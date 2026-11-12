"""Practical scenario: choosing the best generative-augmentation candidate; how much compute a predictor saves.

Strategies (N candidates):
  full-1seed: train every candidate once on production (1 seed) and pick the best. Cost N × 16000 iterations
  <metric> top-k: keep the top k by metric, train those once on production, pick the best.
                  Cost = metric cost + k × 16000 (proxy: 3 seeds × 1000 iterations per candidate)
  random top-k: pick k at random (reference)
Evaluation: regret = best candidate's Δ (5-seed mean) − chosen candidate's Δ (5-seed mean), and the fraction of chosen candidates that hurt (Δ<0).
Production 1-seed noise is reproduced by resampling actual per-seed results (averaged over seed combinations).
"""
import itertools

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# (name, condition selector, proxy budget B, production iterations)
GROUPS = {
    "CIFAR-100 s0 (ResNet-18)": (lambda d: d.group.isin(["main", "heldout"]), 1000, 16000),
    "CIFAR-100 s1": (lambda d: d.group == "subset1", 1000, 16000),
    "Tiny-ImageNet": (lambda d: d.group == "tinyimagenet", 1000, 16000),
    "VGG-11": (lambda d: d.group == "vgg11", 1000, 16000),
    "CIFAR-100 (ResNet-50, pretrained)": (lambda d: d.group == "c100pt", 250, 4000),
    "CIFAR-100 (ViT-B/16, pretrained)": (lambda d: d.group == "vit", 250, 4000),
}
# Metrics: oriented so larger = predicted better. G and P come from the same proxy run (cost 3 seeds × B)
METRICS = {"G (fit gap)": lambda d, B: -d[f"G{B}"], "P1000": lambda d, B: d[f"P{B}"], "FID": lambda d, B: -d.fid,
           "DS (NNLS)": lambda d, B: d.ds_nnls}
PROXY_METRICS = {"G (fit gap)", "P1000"}


def per_seed(runs, exp, cond):
    return runs[(runs.experiment == exp) & (runs.condition == cond)].sort_values("seed").test_acc.to_numpy() * 100


def simulate(d, runs, k, rng, B, full, n_draw=2000):
    """Draw one seed per candidate to mimic a 1-seed production evaluation; return the true Δ of the chosen candidate."""
    seeds = [per_seed(runs, e, c) for e, c in zip(d.experiment, d.condition)]
    true = d.delta.to_numpy()
    best = true.max()
    out = {}
    for name, f in [("full-1seed", None)] + list(METRICS.items()) + [("random", None)]:
        if name not in ("full-1seed", "random") and f(d, B).isna().any():
            continue
        regrets, harm = [], []
        for _ in range(n_draw):
            draw = np.array([s[rng.integers(len(s))] for s in seeds])  # one-seed accuracy per candidate
            if name == "full-1seed":
                cand = np.arange(len(d))
            elif name == "random":
                cand = rng.permutation(len(d))[:k]
            else:
                cand = np.argsort(-f(d, B).to_numpy())[:k]
            pick = cand[np.argmax(draw[cand])]
            regrets.append(best - true[pick])
            harm.append(true[pick] < 0)
        n = len(d)
        proxy = 3 * B if name in PROXY_METRICS else 0
        cost = n * full if name == "full-1seed" else k * full + n * proxy
        out[name] = {"regret": np.mean(regrets), "p_harm": np.mean(harm), "cost_rel": cost / (n * full)}
    return out


def main():
    from augpred.results import load  # table with image-only metrics assigned to settings sharing generated images
    d0 = load()
    runs = pd.read_csv("results/runs.csv")
    rng = np.random.default_rng(0)
    rows = []
    for gname, (sel, B, full) in GROUPS.items():
        d = d0[sel(d0)].reset_index(drop=True)
        for k in [1, 2, 3]:
            for strat, r in simulate(d, runs, k, rng, B, full).items():
                if strat == "full-1seed" and k > 1:
                    continue
                rows.append({"group": gname, "n": len(d), "k": "all" if strat == "full-1seed" else k, "strategy": strat, **r})
    t = pd.DataFrame(rows)
    t.to_csv("results/selection_sim.csv", index=False)
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 200):
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()
