"""Analyses for extra check 3 (existing results only, no GPU). Output goes to results/review3/.

- selection.csv (A3, A4, A6): candidate-selection simulation (same setup as scripts/selection_sim.py), plus
    final selection by validation accuracy, and cost in measured wall time (results/cost.csv)
- ranking.csv (B1): top-rank hit quality. Spearman on conditions whose true Δ is in the top half, NDCG@3, and overlap in the top 3
- lofo.csv (B2): min/max of Spearman(−G, Δ) when leaving out one condition family (generation-method series) at a time
- s_encoding.csv (B5): partial correlation (−G | s, r) under different assignments of s for inpainting, sketch, and roundtrip
Example: python scripts/review3_analyses.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from augpred.results import SETTINGS, load  # noqa: E402
from review2_baselines import partial_spearman, r_of  # noqa: E402
from selection_sim import GROUPS  # noqa: E402

OUT = Path("results/review3")
SCORES = {"G": lambda d, B: -d[f"G{B}"], "P": lambda d, B: d[f"P{B}"], "FID": lambda d, B: -d.fid,
          "DS": lambda d, B: d.ds_nnls}
PROXY = {"G", "P"}


def selection(d0, runs, cost):
    """Draw random numbers in the same order as scripts/selection_sim.py (same seed combinations) and make the final selection by both test and validation accuracy.
    Regret when selecting by test matches results/selection_sim.csv."""
    from selection_sim import METRICS, per_seed
    name_map = {"G (fit gap)": "G", "P1000": "P", "FID": "FID", "DS (NNLS)": "DS"}
    rng = np.random.default_rng(0)
    rows = []
    for gname, (sel, B, full) in GROUPS.items():
        d = d0[sel(d0)].reset_index(drop=True)
        n = len(d)
        test = [per_seed(runs, e, c) for e, c in zip(d.experiment, d.condition)]
        val = [runs[(runs.experiment == e) & (runs.condition == c)].sort_values("seed").val_acc.to_numpy() * 100
               for e, c in zip(d.experiment, d.condition)]
        true = d.delta.to_numpy()
        best = true.max()
        cm = d.merge(cost, on=["experiment", "condition"], how="left")
        train_min, proxy_min = cm.train_min.to_numpy(), cm.proxy_min.to_numpy()
        for k in [1, 2, 3]:
            for name, f in [("full-1seed", None)] + list(METRICS.items()) + [("random", None)]:
                if name not in ("full-1seed", "random") and f(d, B).isna().any():
                    continue
                reg = {"test": [], "val": []}
                for _ in range(2000):
                    idx = [rng.integers(len(t)) for t in test]
                    if name == "full-1seed":
                        cand = np.arange(n)
                    elif name == "random":
                        cand = rng.permutation(n)[:k]
                    else:
                        cand = np.argsort(-f(d, B).to_numpy())[:k]
                    for by, arr in [("test", test), ("val", val)]:
                        score = np.array([arr[j][idx[j]] for j in range(n)])
                        reg[by].append(best - true[cand[np.argmax(score[cand])]])
                if name == "full-1seed" and k > 1:
                    continue
                short = {"full-1seed": "train all", "random": "random"}.get(name, name_map.get(name))
                if name == "full-1seed":
                    c_iter = c_wall = 1.0
                else:
                    proxy = name in ("G (fit gap)", "P1000")
                    c_iter = (k * full + (n * 3 * B if proxy else 0)) / (n * full)
                    trained = k * train_min.mean() if name == "random" else train_min[np.argsort(-f(d, B).to_numpy())[:k]].sum()
                    c_wall = (trained + (proxy_min.sum() if proxy else 0)) / train_min.sum()
                for by in ["test", "val"]:
                    rows.append(dict(pool=gname, n=n, strategy=short, k="all" if name == "full-1seed" else k, pick_by=by,
                                     regret=np.mean(reg[by]), cost_iterations=c_iter, cost_wallclock=c_wall))
    return pd.DataFrame(rows)


def ndcg_at(score, delta, k=3):
    rel = delta - delta.min()  # gain with the worst condition as 0
    order = np.argsort(-score, kind="stable")[:k]
    ideal = np.sort(rel)[::-1][:k]
    disc = 1 / np.log2(np.arange(2, k + 2))
    return (rel[order] * disc).sum() / (ideal * disc).sum()


def ranking(d):
    rows = []
    rng = np.random.default_rng(0)
    for name, sel, B in SETTINGS:
        for subset in ["all", "no label noise"]:
            t = d[sel(d)]
            if subset != "all":
                t = t[~t.noisy]
            if len(t) < 6:
                continue
            delta = t.delta.to_numpy()
            top = delta >= np.median(delta)
            row = {"setting": name, "conditions": subset, "n": len(t), "n_top_half": int(top.sum())}
            for m, f in SCORES.items():
                s = f(t, B).to_numpy()
                if np.isnan(s).any():
                    continue
                row[f"rho_{m}"] = stats.spearmanr(s, delta)[0]
                row[f"rho_top_half_{m}"] = stats.spearmanr(s[top], delta[top])[0]
                row[f"ndcg3_{m}"] = ndcg_at(s, delta)
                row[f"top3_overlap_{m}"] = len(set(np.argsort(-s)[:3]) & set(np.argsort(-delta)[:3]))
            row["ndcg3_random"] = np.mean([ndcg_at(rng.permutation(len(t)).astype(float), delta) for _ in range(2000)])
            rows.append(row)
    return pd.DataFrame(rows)


FAMILY = {"img2img": "img2img strength", "inpaint_center": "inpaint centre", "inpaint_border": "inpaint border",
          "roundtrip": "resize-only", "sketch": "sketch", "low_guidance": "guidance 1.0", "label_noise": "label noise"}


def family_of(r):
    if r.condition.startswith("sdxl"):
        return "SDXL (label noise)" if r.method == "label_noise" else "SDXL"
    return FAMILY[r.method]


def lofo(d):
    rows = []
    for name, sel, B in SETTINGS:
        for subset in ["all", "no label noise"]:
            t = d[sel(d)].copy()
            if subset != "all":
                t = t[~t.noisy]
            if len(t) < 6:
                continue
            t["family"] = [family_of(r) for r in t.itertuples()]
            vals = {}
            for fam in sorted(t.family.unique()):
                u = t[t.family != fam]
                if len(u) >= 4:
                    vals[fam] = stats.spearmanr(-u[f"G{B}"], u.delta)[0]
            worst = min(vals, key=vals.get)
            rows.append(dict(setting=name, conditions=subset, n=len(t), n_families=t.family.nunique(),
                             rho=stats.spearmanr(-t[f"G{B}"], t.delta)[0], rho_min=vals[worst], dropped_at_min=worst,
                             rho_max=max(vals.values())))
    return pd.DataFrame(rows)


def s_variants(cond):
    """Four ways to assign s. default: the assignment in the main text (roundtrip 0, inpainting 1)."""
    import re
    m = re.search(r"(?:^|_)s(\d\.\d+)", cond)
    s_img = float(m.group(1)) if m else None
    area = None
    if "inpaint_center_m" in cond:
        area = float(re.search(r"_m(\d\.\d+)", cond).group(1)) ** 2       # fraction of area repainted
    if "inpaint_border_k" in cond:
        area = 1 - float(re.search(r"_k(\d\.\d+)", cond).group(1)) ** 2  # outside the kept square
    rt = "roundtrip" in cond
    base = 0.0 if rt else (s_img if s_img is not None else 1.0)
    return {"default": base,
            "inpaint = repainted area": 0.0 if rt else (area if area is not None else base),
            "inpaint = 0.5": 0.0 if rt else (0.5 if area is not None else base),
            "sketch, guidance 1.0 = 0.5": 0.0 if rt else (0.5 if ("sketch" in cond or "g1.0" in cond) else base)}


def s_encoding(d):
    rows = []
    for name, sel, B in SETTINGS:
        t = d[sel(d) & ~d.noisy]
        if len(t) < 6:
            continue
        r = t.condition.map(r_of).to_numpy()
        for variant in s_variants("x"):
            s = np.array([s_variants(c)[variant] for c in t.condition])
            rows.append(dict(setting=name, n=len(t), s_encoding=variant,
                             rho_s=stats.spearmanr(s, t.delta)[0] if np.ptp(s) > 0 else np.nan,
                             partial=partial_spearman(-t[f"G{B}"].to_numpy(), t.delta.to_numpy(), [s, r])))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    runs = pd.read_csv("results/runs.csv")
    cost = pd.read_csv("results/cost.csv")[["experiment", "condition", "train_min", "proxy_min"]]
    for fname, df in [("selection", selection(d, runs, cost)), ("ranking", ranking(d)), ("lofo", lofo(d)),
                      ("s_encoding", s_encoding(d))]:
        df.to_csv(OUT / f"{fname}.csv", index=False, float_format="%.6f")
        with pd.option_context("display.width", 250, "display.max_rows", 300, "display.max_columns", 40):
            print(f"\n== {fname}\n" + df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
