import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
SPLIT_DIR = Path(__file__).resolve().parents[2] / "splits"

# Selected by data.name. Default is cifar100 (existing configs without name are all cifar100)
DATASETS = {
    "cifar100": {"file": "cifar100.npz", "size": 32, "num_classes": 100, "mean": CIFAR100_MEAN, "std": CIFAR100_STD},
    # mean/std computed from 100000 Tiny-ImageNet train images
    "tinyimagenet": {"file": "tinyimagenet.npz", "size": 64, "num_classes": 200,
                     "mean": (0.4802, 0.4481, 0.3975), "std": (0.2764, 0.2689, 0.2816)},
    # Stored at 256px; randomly cropped to 224px in training (for fine-tuning ImageNet-pretrained models)
    "flowers102": {"file": "flowers102.npz", "size": 256, "crop": 224, "num_classes": 102,
                   "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    # 100 images/class chosen from official train and test, stored at 256px (scripts/prepare_food101.py); cropped to 224px in training
    "food101": {"file": "food101.npz", "size": 256, "crop": 224, "num_classes": 101,
                "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}


IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def dataset_info(dcfg):
    """If data.resize is set, upscale to that resolution and normalize with ImageNet mean/std (for pretrained models)."""
    info = DATASETS[dcfg.get("name", "cifar100")]
    if dcfg.get("resize"):
        info = {**info, "resize": dcfg["resize"], "mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    return info


def get_split(train_labels, n_per_class, n_val_per_class, subset_seed, dataset="cifar100"):
    """Select train/val indices per class from train. The result is saved to splits/ and reused."""
    path = SPLIT_DIR / f"{dataset}_n{n_per_class}_v{n_val_per_class}_s{subset_seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = np.random.default_rng(subset_seed)
    labels = np.asarray(train_labels)
    train_idx, val_idx = [], []
    for c in range(labels.max() + 1):
        idx = rng.permutation(np.flatnonzero(labels == c))
        train_idx += idx[:n_per_class].tolist()
        val_idx += idx[n_per_class:n_per_class + n_val_per_class].tolist()
    split = {"train": sorted(train_idx), "val": sorted(val_idx)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split))
    return split


class TensorImageDataset(Dataset):
    """Holds uint8 images (N, 3, H, W) and labels; applies transform on access."""

    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.labels[i]


def build_transforms(aug_cfg, train: bool, info=DATASETS["cifar100"]):
    ops = []
    if info.get("crop"):  # high-resolution data: RandomResizedCrop for training, center crop for evaluation
        if train and aug_cfg.get("crop_flip", True):
            ops += [v2.RandomResizedCrop(info["crop"], scale=(0.25, 1.0), antialias=True), v2.RandomHorizontalFlip()]
        else:
            ops.append(v2.CenterCrop(info["crop"]))
    elif train and aug_cfg.get("crop_flip", True):
        ops += [v2.RandomCrop(info["size"], padding=info["size"] // 8), v2.RandomHorizontalFlip()]
    if train:
        if aug_cfg.get("randaugment"):
            ops.append(v2.RandAugment(**aug_cfg["randaugment"]))
        if aug_cfg.get("trivialaugment"):
            ops.append(v2.TrivialAugmentWide())
        if aug_cfg.get("autoaugment"):
            ops.append(v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10))
    if info.get("resize"):  # upscale low-res images for pretrained models (after augmentation at original resolution)
        ops.append(v2.Resize(info["resize"], antialias=True))
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(info["mean"], info["std"])]
    return v2.Compose(ops)


def load_dataset(root, name="cifar100"):
    """Load the npz made by scripts/prepare_*.py; return an (N, 3, H, W) uint8 tensor."""
    d = np.load(Path(root) / DATASETS[name]["file"])
    to_t = lambda x: torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
    return to_t(d["x_train"]), d["y_train"], to_t(d["x_test"]), d["y_test"]


def load_cifar100(root):
    return load_dataset(root, "cifar100")


def _load_gen_npz(gdir, dcfg):
    meta = json.loads((gdir / "meta.json").read_text())
    src = meta["config"]["source"]
    if src.get("dataset", "cifar100") != dcfg.get("name", "cifar100"):
        raise ValueError(f"{gdir}: source dataset ({src.get('dataset', 'cifar100')}) differs from the training config")
    for k in ["n_per_class", "n_val_per_class", "subset_seed"]:
        if src[k] != dcfg[k]:
            raise ValueError(f"{gdir}: source subset ({k}={src[k]}) differs from the training config ({dcfg[k]})")
    return np.load(gdir / "images.npz")


def load_generated(gcfg, dcfg, n_real, return_src=False, return_meta=False):
    """Select ratio * n_real images from data/generated/<name>/images.npz.

    gcfg: {name, ratio, sample_seed (default 0), root (default data/generated),
           label_noise: {name: set generated with wrong prompts, rate: fraction to replace} (optional),
           filter: {frac: fraction, by: loss | random, score: npy of per-image losses (when by: loss),
                    replace: set generated with another seed (if omitted, only remove)} (optional)}
    With label_noise, a fraction rate of the selected images is replaced by images generated from the same
    source image with a different class's prompt. Labels stay the original class (= label noise).
    """
    root = Path(gcfg.get("root", "data/generated"))
    d = _load_gen_npz(root / gcfg["name"], dcfg)
    x, y = d["x"], d["y"]
    n = int(round(gcfg["ratio"] * n_real))
    if n > len(y):
        raise ValueError(f"{gcfg['name']}: need {n} images but only {len(y)}")
    rng = np.random.default_rng(gcfg.get("sample_seed", 0))
    idx = np.sort(rng.permutation(len(y))[:n])
    x_sel, y_sel = x[idx], y[idx]

    ln = gcfg.get("label_noise")
    if ln and ln["rate"] > 0:
        w = _load_gen_npz(root / ln["name"], dcfg)
        if not (np.array_equal(w["src_idx"], d["src_idx"]) and np.array_equal(w["y"], y)):
            raise ValueError(f"{ln['name']} and {gcfg['name']} have different source image order")
        swap = np.random.default_rng(gcfg.get("sample_seed", 0) + 1).permutation(n)[:int(round(ln["rate"] * n))]
        x_sel = x_sel.copy()
        x_sel[swap] = w["x"][idx[swap]]
    else:
        swap = np.array([], dtype=int)
    src_sel = d["src_idx"][idx]

    flt = gcfg.get("filter")
    if flt:  # remove/replace some selected images, by highest proxy loss (by: loss) or at random (by: random)
        k = int(round(flt["frac"] * n))
        if flt["by"] == "loss":
            score = np.load(flt["score"])  # per-image losses in generated-set order (scripts/make_filter_scores.py)
            if len(score) != len(y):
                raise ValueError(f"{flt['score']}: has {len(score)} scores but {len(y)} generated images")
            hit = np.argsort(-score[idx], kind="stable")[:k]
        else:
            hit = np.random.default_rng(gcfg.get("sample_seed", 0) + 2).permutation(n)[:k]
        if flt.get("replace"):  # replace with images generated from the same source image with another seed
            r = _load_gen_npz(root / flt["replace"], dcfg)
            if not (np.array_equal(r["src_idx"], d["src_idx"]) and np.array_equal(r["y"], y)):
                raise ValueError(f"{flt['replace']} and {gcfg['name']} have different source image order")
            x_sel = x_sel.copy()
            x_sel[hit] = r["x"][idx[hit]]
            swap = np.setdiff1d(swap, hit)  # replaced images are no longer label noise
        else:
            keep = np.setdiff1d(np.arange(n), hit)
            x_sel, y_sel, src_sel = x_sel[keep], y_sel[keep], src_sel[keep]
            swap = np.flatnonzero(np.isin(keep, swap))
            n = len(keep)
    x_t = torch.from_numpy(x_sel).permute(0, 3, 1, 2).contiguous()
    if return_meta:  # for analysis: source image indices of selected images, and whether each was replaced by label noise
        swapped = np.zeros(n, dtype=bool)
        swapped[swap] = True
        return x_t, y_sel, {"src_idx": src_sel, "swapped": swapped}
    return (x_t, y_sel, src_sel) if return_src else (x_t, y_sel)


def build_datasets(cfg):
    dcfg, aug_cfg = cfg["data"], cfg["aug"]
    name, info = dcfg.get("name", "cifar100"), dataset_info(dcfg)
    x_tr, y_tr, x_te, y_te = load_dataset(dcfg["root"], name)
    split = get_split(y_tr, dcfg["n_per_class"], dcfg["n_val_per_class"], dcfg["subset_seed"], name)

    tr_key, val_key = "train", "val"
    if dcfg.get("train_split") == "val":  # train on validation images (for the Affinity reference model; avoids the training subset)
        tr_key, val_key = "val", "train"
    x_train, y_train = x_tr[split[tr_key]], y_tr[split[tr_key]]
    n_real = len(y_train)
    if aug_cfg.get("generated"):
        x_gen, y_gen, gen_meta = load_generated(aug_cfg["generated"], dcfg, n_real=len(y_train), return_meta=True)
        x_train, y_train = torch.cat([x_train, x_gen]), np.concatenate([y_train, y_gen])
    else:
        gen_meta = None

    eval_tf = build_transforms(aug_cfg, train=False, info=info)
    return {
        "train": TensorImageDataset(x_train, y_train, build_transforms(aug_cfg, train=True, info=info)),
        "val": TensorImageDataset(x_tr[split[val_key]], y_tr[split[val_key]], eval_tf),
        "test": TensorImageDataset(x_te, y_te, eval_tf),
        # for evaluating training data without augmentation (first n_real are real images, the rest generated)
        "train_eval": TensorImageDataset(x_train, y_train, eval_tf),
        "n_real": n_real,
        "gen_meta": gen_meta,
    }
