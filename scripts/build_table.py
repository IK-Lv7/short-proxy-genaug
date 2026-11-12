"""Combine all conditions × all metrics into one table, results/all_conditions.csv (for paper figures/tables and release).

1 row = (dataset, subset, model, experiment, condition). Δ is the difference (pt) from the baseline with the same data, model, and steps.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

EXPS = ["fixedsteps_gen", "labelnoise_gen", "inpaint_gen", "ratio_gen", "heldout_gen", "sub1_gen", "tin_gen", "tin_gen2",
        "vgg_gen", "conventional_aug", "sdxl_gen", "flowers_gen", "c100pt_gen", "vit_gen", "flowers_ti"]
GROUP = {"fixedsteps_gen": "main", "labelnoise_gen": "main", "inpaint_gen": "main", "ratio_gen": "main",
         "heldout_gen": "heldout", "sub1_gen": "subset1", "tin_gen": "tinyimagenet", "tin_gen2": "tinyimagenet",
         "vgg_gen": "vgg11", "conventional_aug": "conventional", "sdxl_gen": "sdxl", "flowers_gen": "flowers102", "c100pt_gen": "c100pt", "vit_gen": "vit", "flowers_ti": "flowers_ti"}


def method_of(cond):
    if "noise" in cond or "wrong" in cond:  # label noise is classified ahead of method
        return "label_noise"
    for k in ["inpaint_center", "inpaint_border", "roundtrip", "sketch", "g1.0", "noise", "randaug", "trivial", "autoaug"]:
        if k in cond:
            return {"noise": "label_noise", "g1.0": "low_guidance"}.get(k, k)
    return "img2img"


def proxy_stats(exp, cond, B):
    d = Path(f"results/runs/proxyfit{B}__{exp}/{cond}")
    rs = [json.loads(f.read_text()) for f in sorted(d.glob("seed*.json"))]
    if not rs:
        return None, None
    G = np.mean([r["fit_loss_gen"] - r["fit_loss_real"] for r in rs]) if "fit_loss_gen" in rs[0] else np.nan
    return np.mean([r["val_acc"] for r in rs]), G


def main():
    runs = pd.read_csv("results/runs.csv")
    met = pd.read_csv("results/metrics.csv") if Path("results/metrics.csv").exists() else pd.DataFrame()
    dsm = pd.read_csv("results/ds_metrics.csv") if Path("results/ds_metrics.csv").exists() else pd.DataFrame()
    rows = []
    for exp in EXPS:
        f = Path(f"configs/exp/{exp}.yaml")
        if not f.exists():
            continue
        e = yaml.safe_load(f.read_text())
        ref = e.get("reference", "baseline")
        rexp, rcond = ref.split("/") if "/" in ref else (exp, ref)
        base = runs[(runs.experiment == rexp) & (runs.condition == rcond)].test_acc * 100
        base_v, _ = proxy_stats(rexp, rcond, 1000)
        ov = e.get("overrides", {})
        for cond in e["conditions"]:
            if cond == "baseline":
                continue
            a = runs[(runs.experiment == exp) & (runs.condition == cond)].test_acc * 100
            if a.empty:
                continue
            row = {"group": GROUP[exp], "dataset": ov.get("data", {}).get("name", "cifar100"),
                   "subset": ov.get("data", {}).get("subset_seed", 0), "model": ov.get("model", {}).get("name", "resnet18_cifar"),
                   "experiment": exp, "condition": cond, "method": method_of(cond), "n_seeds": len(a),
                   "acc": a.mean(), "acc_std": a.std(), "delta": a.mean() - base.mean(),
                   "p_vs_baseline": stats.ttest_ind(a, base, equal_var=False).pvalue}
            for B in [250, 500, 1000, 2000, 4000]:
                v, G = proxy_stats(exp, cond, B)
                bv, _ = proxy_stats(rexp, rcond, B)
                if v is not None and bv is not None:
                    row[f"P{B}"] = (v - bv) * 100
                    row[f"G{B}"] = G
            rows.append(row)
    df = pd.DataFrame(rows)
    for m in [met, dsm]:
        if not m.empty:
            df = df.merge(m, on=["experiment", "condition"], how="left", suffixes=("", "_dup"))
    df = df[[c for c in df.columns if not c.endswith("_dup")]]
    df.to_csv("results/all_conditions.csv", index=False)
    print(df.groupby("group").size().to_string())
    print(f"saved results/all_conditions.csv ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
