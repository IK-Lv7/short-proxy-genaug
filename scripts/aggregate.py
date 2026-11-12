"""Combine runs under results/runs/ into one table and show per-condition Δaccuracy and seed variance.

Output: results/runs.csv (per condition × seed), results/conditions.csv (per condition)
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--exp", help="show only this experiment")
    args = p.parse_args()
    root = Path(args.results)

    rows = []
    for f in sorted((root / "runs").glob("*/*/seed*.json")):
        r = json.loads(f.read_text())
        rows.append({k: r[k] for k in ["experiment", "condition", "seed", "test_acc", "val_acc", "n_train", "wall_sec"]})
    if not rows:
        print("no runs")
        return
    runs = pd.DataFrame(rows)
    runs.to_csv(root / "runs.csv", index=False)

    exp_refs = {}
    for f in Path("configs/exp").glob("*.yaml"):
        import yaml
        e = yaml.safe_load(f.read_text())
        exp_refs[e["name"]] = e.get("reference", "baseline")

    out = []
    for exp_name, df in runs.groupby("experiment"):
        if args.exp and exp_name != args.exp:
            continue
        ref_name = exp_refs.get(exp_name, "baseline")
        # "experiment/condition" lets a condition from another experiment be the reference
        ref_exp, ref_cond = ref_name.split("/") if "/" in ref_name else (exp_name, ref_name)
        ref = runs[(runs.experiment == ref_exp) & (runs.condition == ref_cond)].set_index("seed").test_acc
        for cond, d in df.groupby("condition"):
            acc = d.set_index("seed").test_acc
            paired = (acc - ref).dropna()  # difference between the same seeds
            out.append({
                "experiment": exp_name, "condition": cond, "n_seeds": len(acc),
                "test_acc_mean": acc.mean(), "test_acc_std": acc.std(ddof=1),
                "delta_mean": acc.mean() - ref.mean() if len(ref) else float("nan"),
                "delta_paired_std": paired.std(ddof=1) if len(paired) > 1 else float("nan"),
            })
    cond = pd.DataFrame(out)
    cond.to_csv(root / "conditions.csv", index=False)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(cond.to_string(index=False))


if __name__ == "__main__":
    main()
