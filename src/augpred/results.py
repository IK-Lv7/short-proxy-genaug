"""Load results/all_conditions.csv; add shared image metrics, noisy flag, and Holm-adjusted p-values."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

IMAGE_METRICS = ["fid", "kid", "label_acc", "affinity", "ds_nnls", "precision", "recall", "density", "coverage",
                 "cc_kid", "fd_dinov2"]

# (setting name, condition selector, main proxy-training budget)
SETTINGS = [
    ("CIFAR-100, ResNet-18, SD1.5", lambda d: d.group.isin(["main", "heldout"]), 1000),
    ("CIFAR-100, ResNet-18, SDXL", lambda d: d.group == "sdxl", 1000),
    ("CIFAR-100 (2nd subset)", lambda d: d.group == "subset1", 1000),
    ("CIFAR-100, VGG-11", lambda d: d.group == "vgg11", 1000),
    ("Tiny-ImageNet, ResNet-18", lambda d: d.group == "tinyimagenet", 1000),
    ("CIFAR-100, ResNet-50 (pretrained)", lambda d: d.group == "c100pt", 250),
    ("CIFAR-100, ViT-B/16 (pretrained)", lambda d: d.group == "vit", 250),
    ("Flowers102, ResNet-50 (pretrained)", lambda d: d.group == "flowers102", 250),
]


def load():
    d = pd.read_csv("results/all_conditions.csv")
    d = d[d.group != "conventional"].copy()
    if Path("results/metrics2.csv").exists():
        d = d.merge(pd.read_csv("results/metrics2.csv"), on=["experiment", "condition"], how="left")
    for c in IMAGE_METRICS:
        if c not in d:
            d[c] = np.nan
    # Assign main-group image metrics to vgg11 / c100pt / vit
    src = d[d.group.isin(["main", "heldout", "sdxl"])].set_index("condition")
    for g in ["vgg11", "c100pt", "vit"]:
        for c in IMAGE_METRICS + ["ds_lstsq"]:
            m = d.group == g
            d.loc[m, c] = d.loc[m, "condition"].map(src[c])
    d["noisy"] = d.method == "label_noise"
    # Holm correction per setting
    d["p_holm"] = np.nan
    for _, sel, _ in SETTINGS:
        m = sel(d).to_numpy()
        pv = d.loc[m, "p_vs_baseline"].to_numpy()
        order = np.argsort(pv)
        holm = np.empty(len(pv))
        holm[order] = np.maximum.accumulate(np.minimum(1, (len(pv) - np.arange(len(pv))) * pv[order]))
        d.loc[m, "p_holm"] = holm
    return d


def rho(x, y):
    ok = x.notna() & y.notna()
    if ok.sum() < len(x) or len(x) < 4:
        return np.nan
    return stats.spearmanr(x, y)[0]
