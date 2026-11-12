"""Combine per-condition compute time (generation, production training, proxy training) with Δaccuracy into results/cost.csv.

All times are measured (one RTX 4080 SUPER, GPU jobs run serially one at a time):
  gen_min   : generation time of the images used. gen_sec in data/generated/<name>/meta.json × (images used / images generated)
              Label-noise conditions are artificial interventions, so generation time for the wrong-prompt side is not counted
  train_min : time for one production training run (seed mean, wall_sec in results/runs/<exp>/<cond>/seed*.json)
  proxy_min : total time for 3 proxy-training seeds (results/runs/proxyfit<B>__<exp>/<cond>/)
Example: python scripts/cost_table.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from augpred.data import DATASETS  # noqa: E402
from augpred.results import SETTINGS  # noqa: E402

RUNS = Path("results/runs")
GEN = Path("data/generated")


def runs(exp, cond):
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(RUNS / exp / cond / "seed*.json")))]


def gen_minutes(aug, n_real):
    g = (aug or {}).get("generated")
    if not g:
        return 0.0
    meta = json.loads((GEN / g["name"] / "meta.json").read_text())
    return meta["gen_sec"] / 60 * round(g["ratio"] * n_real) / meta["n"]


def main():
    d = pd.read_csv("results/all_conditions.csv")
    budget = {}
    for _, sel, b in SETTINGS:
        for g in d[sel(d)].group.unique():
            budget[g] = b
    budget["conventional"] = 1000
    rows = []
    for r in d.itertuples():
        rs = runs(r.experiment, r.condition)
        if not rs:
            continue
        cfg = rs[0]["config"]
        n_real = cfg["data"]["n_per_class"] * DATASETS[cfg["data"].get("name", "cifar100")]["num_classes"]
        prs = runs(f"proxyfit{budget[r.group]}__{r.experiment}", r.condition)
        rows.append({
            "group": r.group, "experiment": r.experiment, "condition": r.condition, "method": r.method,
            "delta": r.delta, "p_vs_baseline": r.p_vs_baseline,
            "gen_min": gen_minutes(cfg["aug"], n_real),
            "train_min": np.mean([x["wall_sec"] for x in rs]) / 60,
            "proxy_min": sum(x["wall_sec"] for x in prs) / 60 if len(prs) == 3 else np.nan,
            "n_gen_images": rs[0]["n_train"] - n_real,
            "gen_name": ((cfg["aug"] or {}).get("generated") or {}).get("name", ""),
        })
    out = pd.DataFrame(rows)
    out.to_csv("results/cost.csv", index=False, float_format="%.6g")
    print(f"wrote results/cost.csv ({len(out)} conditions)")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
