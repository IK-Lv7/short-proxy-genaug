"""Extra check 2 (point 1): compare G against early-training baselines with the same budget (from training results of scripts/queue_review2.sh).

- cart<B>__ (per-image outputs at 5 points during proxy training):
    confidence difference (generated − real; larger is better), variability difference, EL2N difference (last vs. first checkpoint; smaller is better),
    and G from the same training (to confirm reproduction)
- trunc<B>__ (stop at B with the production schedule, 10 validation-accuracy points):
    fit a − b·t^(−c) to the seed-mean curve, extrapolate to the end of production, and take the difference from the baseline (larger is better).
    If fitting fails, substitute the difference at the last point (the count is printed)
Output: results/review2/early_baselines.csv
Example: python scripts/review2_early.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from augpred.results import SETTINGS, load  # noqa: E402
from review2_baselines import reference  # noqa: E402

OUT = Path("results/review2")
FULL = {1000: 16000, 250: 4000}


def runs(prefix, exp, cond):
    return [json.loads(f.read_text()) for f in sorted(Path(f"results/runs/{prefix}__{exp}/{cond}").glob("seed*.json"))]


def cart_stats(exp, cond, B):
    rs = runs(f"cart{B}", exp, cond)
    if len(rs) < 3 or "cart_conf_gen" not in rs[0]:
        return {}
    m = lambda f: np.mean([f(r) for r in rs])
    return {"cart_conf": m(lambda r: r["cart_conf_gen"] - r["cart_conf_real"]),
            "cart_var": m(lambda r: r["cart_var_gen"] - r["cart_var_real"]),
            "el2n_last": m(lambda r: r["el2n_gen"][-1] - r["el2n_real"][-1]),
            "el2n_first": m(lambda r: r["el2n_gen"][0] - r["el2n_real"][0]),
            "G_rerun": m(lambda r: r["fit_loss_gen"] - r["fit_loss_real"])}


def curve(exp, cond, B):
    rs = runs(f"trunc{B}", exp, cond)
    if len(rs) < 3:
        return None
    steps = [h["step"] for h in rs[0]["history"] if "val_acc" in h]
    acc = np.mean([[h["val_acc"] for h in r["history"] if "val_acc" in h] for r in rs], 0)
    return np.array(steps, float), acc


def extrapolate(t, y, t_end):
    f = lambda t, a, b, c: a - b * t ** (-c)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = optimize.curve_fit(f, t / t[-1], y, p0=[y[-1] + 0.05, 0.1, 1.0],
                                      bounds=([0, 0, 0.01], [1, 10, 5]), maxfev=20000)
        return float(np.clip(f(t_end / t[-1], *p), 0, 1)), True
    except (RuntimeError, ValueError):
        return float(y[-1]), False


def main():
    d = load()
    rows, fails = [], 0
    for name, sel, B in SETTINGS:
        t0 = d[sel(d)].copy()
        ext = {}
        for e, c in list(zip(t0.experiment, t0.condition)) + [tuple(reference(e)) for e in t0.experiment.unique()]:
            cv = curve(e, c, B)
            if cv is not None and (e, c) not in ext:
                ext[(e, c)], ok = extrapolate(*cv, FULL[B])
                ext[(e, c, "last")] = cv[1][-1]
                fails += not ok
        recs = []
        for r in t0.itertuples():
            ref = tuple(reference(r.experiment))
            rec = {"noisy": r.noisy, "delta": r.delta, "-G": -getattr(r, f"G{B}"), **cart_stats(r.experiment, r.condition, B)}
            if (r.experiment, r.condition) in ext and ref in ext:
                rec["curve_extrap"] = ext[(r.experiment, r.condition)] - ext[ref]
                rec["curve_last"] = ext[(r.experiment, r.condition, "last")] - ext[ref + ("last",)]
            recs.append(rec)
        t1 = pd.DataFrame(recs)
        for subset, t in [("all", t1), ("no label noise", t1[~t1.noisy])]:
            row = {"setting": name, "conditions": subset, "n": len(t)}
            if len(t) >= 5:
                for col, sign in [("-G", 1), ("G_rerun", -1), ("cart_conf", 1), ("cart_var", -1), ("el2n_last", -1),
                                  ("el2n_first", -1), ("curve_extrap", 1), ("curve_last", 1)]:
                    if col in t and t[col].notna().all():
                        row[col] = stats.spearmanr(sign * t[col], t.delta)[0]
            rows.append(row)
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "early_baselines.csv", index=False, float_format="%.6f")
    with pd.option_context("display.width", 220):
        print(out.round(2).to_string(index=False))
    print(f"curve fits that fell back to the last point: {fails}")
    print("Signs: variability and EL2N are assumed smaller = better, and directions are aligned (same for G_rerun)")


if __name__ == "__main__":
    main()
