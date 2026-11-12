"""Generative augmentation. Generate n_per_image images per real image in the training subset and cache them as uint8 npz at the original resolution."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .classnames import CIFAR100_CLASSES
from .data import DATASETS, get_split, load_dataset


def class_names(dataset):
    """Class names used in prompts (in label order)."""
    if dataset == "cifar100":
        return [c.replace("_", " ") for c in CIFAR100_CLASSES]
    if dataset in ("tinyimagenet", "flowers102", "food101"):
        return json.loads((Path(__file__).parent / f"{dataset}_classes.json").read_text())["names"]
    raise ValueError(dataset)


def upsample(x_uint8, res):
    """(N,3,32,32) uint8 -> (N,3,res,res) float [0,1]"""
    return F.interpolate(x_uint8.float() / 255, size=(res, res), mode="bicubic", align_corners=False).clamp(0, 1)


def downsample(x, size=32):
    """(N,3,H,W) float [0,1] -> (N,3,size,size) uint8"""
    x = F.interpolate(x.float(), size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


def load_source(src_cfg):
    dataset = src_cfg.get("dataset", "cifar100")
    x_tr, y_tr, _, _ = load_dataset(f"data/{dataset}", dataset)
    split = get_split(y_tr, src_cfg["n_per_class"], src_cfg["n_val_per_class"], src_cfg["subset_seed"], dataset)
    idx = np.asarray(split["train"])
    return x_tr[idx], y_tr[idx], idx


class Img2Img:
    def __init__(self, cfg, device="cuda"):
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            cfg["model"], torch_dtype=torch.float16, cache_dir=cfg.get("cache_dir"),
            safety_checker=None, requires_safety_checker=False,
        ).to(device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.vae.enable_slicing()
        self.pipe.set_progress_bar_config(disable=True)
        if cfg.get("ti_embeds"):  # load per-class tokens learned by textual inversion
            from .textual_inversion import load_into_pipeline
            load_into_pipeline(self.pipe, torch.load(cfg["ti_embeds"]), cfg.get("init_word", "flower"))
        self.cfg = cfg

    def __call__(self, x_up, labels, seeds):  # labels: list of class names
        c = self.cfg
        prompts = [c["prompt"].format(cls=n) for n in labels]
        gens = [torch.Generator("cuda").manual_seed(int(s)) for s in seeds]
        out = self.pipe(
            prompt=prompts, negative_prompt=[c.get("negative_prompt", "")] * len(prompts),
            image=x_up, strength=c["strength"], guidance_scale=c["guidance_scale"],
            num_inference_steps=c["num_inference_steps"], generator=gens, output_type="pt",
        ).images
        return out  # (B,3,res,res) float [0,1]


def make_mask(n, res, mode, frac, device):
    """(n,1,res,res) 0/1 mask; 1 marks the region to repaint.
    center: repaint a centered square of side frac / border: keep a centered square of side frac and repaint the surroundings"""
    side = int(round(res * frac))
    lo = (res - side) // 2
    center = torch.zeros(n, 1, res, res, device=device)
    center[:, :, lo:lo + side, lo:lo + side] = 1
    return center if mode == "center" else 1 - center


class Inpaint(Img2Img):
    def __init__(self, cfg, device="cuda"):
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            cfg["model"], torch_dtype=torch.float16, cache_dir=cfg.get("cache_dir"),
            variant=cfg.get("variant"), use_safetensors=True,
            safety_checker=None, requires_safety_checker=False,
        ).to(device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.vae.enable_slicing()
        self.pipe.set_progress_bar_config(disable=True)
        self.cfg = cfg

    def __call__(self, x_up, labels, seeds):  # labels: list of class names
        c = self.cfg
        prompts = [c["prompt"].format(cls=n) for n in labels]
        gens = [torch.Generator("cuda").manual_seed(int(s)) for s in seeds]
        mask = make_mask(len(x_up), x_up.shape[-1], c["mask_mode"], c["mask_frac"], x_up.device)
        out = self.pipe(
            prompt=prompts, negative_prompt=[c.get("negative_prompt", "")] * len(prompts),
            image=x_up, mask_image=mask, strength=c.get("strength", 1.0),
            guidance_scale=c["guidance_scale"], num_inference_steps=c["num_inference_steps"],
            generator=gens, output_type="pt", height=x_up.shape[-1], width=x_up.shape[-1],
        ).images
        return out


class SDXLImg2Img(Img2Img):
    """SDXL base 1.0 img2img. Prompt handling and seeding are the same as the SD 1.5 version."""

    def __init__(self, cfg, device="cuda"):
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline

        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            cfg["model"], torch_dtype=torch.float16, cache_dir=cfg.get("cache_dir"),
            variant=cfg.get("variant", "fp16"), use_safetensors=True,
        ).to(device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.vae.enable_slicing()
        self.pipe.set_progress_bar_config(disable=True)
        self.cfg = cfg


GENERATORS = {"img2img": Img2Img, "inpaint": Inpaint, "sdxl_img2img": SDXLImg2Img}


def generate_variant(name, cfg, limit=None, device="cuda", log=print):
    out_dir = Path(cfg["out_root"]) / name
    if (out_dir / "images.npz").exists() and limit is None:
        log(f"[cache] {name}")
        return out_dir
    x_src, y_src, src_idx = load_source(cfg["source"])
    if limit:
        x_src, y_src, src_idx = x_src[:limit], y_src[:limit], src_idx[:limit]
    res, k = cfg["resolution"], cfg["n_per_image"]
    dataset = cfg["source"].get("dataset", "cifar100")
    names, out_size = class_names(dataset), DATASETS[dataset]["size"]
    if cfg.get("ti_embeds"):  # replace class names in prompts with learned tokens
        from .textual_inversion import token_name
        names = [token_name(c) for c in range(len(names))]
    # wrong_prompt: true generates each image with a prompt of a class different from the original (for label-noise interventions).
    # The saved label y stays the original class; the class used in the prompt is recorded in prompt_y
    prompt_y_src = y_src.copy()
    if cfg.get("wrong_prompt"):
        n_cls = len(names)
        rng = np.random.default_rng(cfg["seed"] + 12345)
        prompt_y_src = (y_src + rng.integers(1, n_cls, size=len(y_src))) % n_cls

    gen = GENERATORS[cfg["method"]](cfg, device) if cfg["method"] in GENERATORS else None
    bs = cfg.get("batch_size", 256)
    xs, ys, pys, srcs, previews = [], [], [], [], []
    t0 = time.time()
    for rep in range(k):
        for i in range(0, len(y_src), bs):
            xb = x_src[i:i + bs]
            x_up = upsample(xb, res).to(device)
            if cfg["method"] == "resize_roundtrip":
                out = x_up
            elif cfg["method"] in GENERATORS:
                # Per-image seed: determined by generation seed, repeat index, and source image index.
                # seed_per_class: true uses the class instead of the source image index (same-class images share noise, lowering diversity)
                keys = y_src[i:i + bs] if cfg.get("seed_per_class") else src_idx[i:i + bs]
                seeds = [cfg["seed"] * 10_000_000 + rep * 100_000 + int(s) for s in keys]
                out = gen(x_up, [names[j] for j in prompt_y_src[i:i + bs]], seeds)
            else:
                raise ValueError(cfg["method"])
            if len(previews) * bs < 64:
                previews.append(out[:64].cpu().half())
            xs.append(downsample(out, out_size).cpu())
            ys.append(y_src[i:i + bs])
            pys.append(prompt_y_src[i:i + bs])
            srcs.append(src_idx[i:i + bs])
            done = rep * len(y_src) + i + len(xb)
            if (i // bs) % 20 == 0:
                el = time.time() - t0
                log(f"  {name}: {done}/{k * len(y_src)} ({el:.0f}s, eta {el / done * (k * len(y_src) - done):.0f}s)")

    x = torch.cat(xs).permute(0, 2, 3, 1).numpy()
    y, src = np.concatenate(ys), np.concatenate(srcs)
    if limit:
        out_dir = out_dir.parent / f"_test_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "images.npz", x=x, y=y, prompt_y=np.concatenate(pys), src_idx=src)
    _save_preview(torch.cat(previews)[:64], x_src[:64], out_dir / "preview.png")
    meta = {"name": name, "config": cfg, "n": len(y), "gen_sec": time.time() - t0,
            "n_black": int((x.reshape(len(x), -1).max(1) < 10).sum())}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"[saved] {out_dir} n={len(y)} ({meta['gen_sec']:.0f}s, black={meta['n_black']})")
    return out_dir


def _save_preview(gen_hr, src_lr, path, n=16):
    """Top row: source images (enlarged); bottom row: generated images (512px)."""
    from torchvision.utils import save_image
    n = min(n, len(gen_hr))
    src = F.interpolate(src_lr[:n].float() / 255, size=gen_hr.shape[-1], mode="nearest")
    save_image(torch.cat([src, gen_hr[:n].float()]), path, nrow=n)
