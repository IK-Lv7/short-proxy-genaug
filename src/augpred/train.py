import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import build_datasets
from .models import build_model


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


@torch.no_grad()
def train_eval_outputs(model, ds, device, amp):
    """Per-image loss, correctness, true-class probability, and EL2N (|softmax - one-hot|_2) on the training data (no augmentation)."""
    was_training = model.training
    model.eval()
    loader = DataLoader(ds["train_eval"], 512, shuffle=False, num_workers=4)
    out = {"loss": [], "correct": [], "p_true": [], "el2n": []}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        logits = logits.float()
        prob = logits.softmax(1)
        out["loss"].append(F.cross_entropy(logits, y, reduction="none").cpu())
        out["correct"].append((logits.argmax(1) == y).cpu())
        out["p_true"].append(prob.gather(1, y[:, None])[:, 0].cpu())
        out["el2n"].append((prob - F.one_hot(y, prob.shape[1]).float()).norm(dim=1).cpu())
    model.train(was_training)
    return {k: torch.cat(v) for k, v in out.items()}


def cartography_stats(snaps, n_real):
    """From multi-checkpoint outputs (train_eval_outputs), average Dataset Cartography confidence / variability
    (mean and std over checkpoints of true-class probability) and per-checkpoint EL2N, separately for real and generated images."""
    p = torch.stack([s["p_true"] for s in snaps])  # (checkpoint, image)
    conf, var = p.mean(0), p.std(0, unbiased=False)
    out = {}
    for part, sl in [("real", slice(0, n_real)), ("gen", slice(n_real, None))]:
        if conf[sl].numel() == 0:
            continue
        out[f"cart_conf_{part}"] = conf[sl].mean().item()
        out[f"cart_var_{part}"] = var[sl].mean().item()
        out[f"el2n_{part}"] = [s["el2n"][sl].mean().item() for s in snaps]
    return out


@torch.no_grad()
def train_fit_stats(model, ds, device, amp, save_path=None):
    """Measure fit to the training data separately for real and generated images (no augmentation)."""
    o = train_eval_outputs(model, ds, device, amp)
    losses, correct = o["loss"], o["correct"].float()
    n = ds["n_real"]
    if save_path:  # save per-image losses for analysis
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        meta = ds["gen_meta"] or {"src_idx": np.array([], dtype=int), "swapped": np.array([], dtype=bool)}
        np.savez(save_path, loss_real=losses[:n].numpy(), loss_gen=losses[n:].numpy(),
                 correct_real=correct[:n].numpy(), correct_gen=correct[n:].numpy(),
                 y_gen=ds["train_eval"].labels[n:].numpy(), src_idx=meta["src_idx"], swapped=meta["swapped"])
    out = {"fit_loss_real": losses[:n].mean().item(), "fit_acc_real": correct[:n].mean().item()}
    if len(losses) > n:
        out.update(fit_loss_gen=losses[n:].mean().item(), fit_acc_gen=correct[n:].mean().item(),
                   fit_loss_gen_median=losses[n:].median().item())
    return out


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        loss_sum += F.cross_entropy(logits.float(), y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total, loss_sum / total


def train_one(cfg, seed, device="cuda", log=print):
    """Train one condition × one seed; return a result dict."""
    tcfg = cfg["train"]
    set_seed(seed, tcfg["deterministic"])
    ds = build_datasets(cfg)
    g = torch.Generator().manual_seed(seed)
    kw = dict(num_workers=tcfg["num_workers"], pin_memory=True, persistent_workers=tcfg["num_workers"] > 0)
    train_loader = DataLoader(ds["train"], tcfg["batch_size"], shuffle=True, drop_last=False, generator=g, **kw)
    val_loader = DataLoader(ds["val"], 512, **kw)
    test_loader = DataLoader(ds["test"], 512, **kw)

    model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=tcfg["lr"], momentum=tcfg["momentum"],
                          weight_decay=tcfg["weight_decay"], nesterov=True)
    # If steps is set, total iterations are fixed regardless of data size (for compute-matched comparison).
    # Otherwise epochs * (iterations per epoch). History is recorded one row every record_every iterations
    if tcfg.get("steps"):
        steps, record_every = tcfg["steps"], tcfg.get("record_every", 200)
    else:
        steps, record_every = tcfg["epochs"] * len(train_loader), len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tcfg["lr"], total_steps=steps, pct_start=0.15,
                                                anneal_strategy="cos")
    amp = tcfg["amp"]
    history = []
    t0 = time.time()
    # stop_at: keep the LR schedule for steps but stop at an intermediate iteration (for learning-curve extrapolation)
    stop = min(tcfg.get("stop_at") or steps, steps)
    # fit_checkpoints: record per-image outputs on training data k times during training (every 1/k of stop) (for Cartography/EL2N)
    k_fit = tcfg.get("fit_checkpoints") or 0
    fit_steps = {round(stop * (i + 1) / k_fit) for i in range(k_fit)} if k_fit else set()
    snaps = []

    def batches():
        while True:
            yield from train_loader

    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for step, (x, y) in enumerate(batches(), start=1):
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        loss_sum += loss.item() * y.numel()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
        if step in fit_steps:
            snaps.append(train_eval_outputs(model, ds, device, amp))
        if step % record_every and step != stop:
            continue
        rec = {"step": step, "epoch": step / len(train_loader), "train_loss": loss_sum / n, "train_acc": correct / n}
        loss_sum, correct, n = 0.0, 0, 0
        if step == stop or len(history) % 10 == 9 or tcfg.get("val_every_record"):
            rec["val_acc"], rec["val_loss"] = evaluate(model, val_loader, device, amp)
            model.train()
            log(f"  step{step:6d} (ep{rec['epoch']:6.1f}) loss={rec['train_loss']:.3f} "
                f"train_acc={rec['train_acc']:.3f} val_acc={rec['val_acc']:.4f}")
        history.append(rec)
        if step == stop:
            break

    test_acc, test_loss = evaluate(model, test_loader, device, amp)
    fit = train_fit_stats(model, ds, device, amp, save_path=tcfg.get("save_sample_losses"))
    if tcfg.get("save_model"):
        Path(tcfg["save_model"]).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), tcfg["save_model"])
    if snaps:
        fit.update(cartography_stats(snaps, ds["n_real"]), fit_checkpoint_steps=sorted(fit_steps))
    if stop < steps:
        fit["stopped_at"] = stop
    return {
        "seed": seed,
        "test_acc": test_acc,
        **fit,
        "test_loss": test_loss,
        "val_acc": history[-1]["val_acc"],
        "n_train": len(ds["train"]),
        "steps": steps,
        "wall_sec": time.time() - t0,
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "config": cfg,
        "history": history,
    }


def save_result(result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
