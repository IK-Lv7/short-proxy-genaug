"""Extra check 2: compare G against trivial predictors and baselines built from existing records.

- Point 2: predictions from generation strength s and mix ratio r alone (s, s×r), and partial rank correlation of G controlling for s and r
  s is read from the condition name (roundtrip is 0, inpainting is strength 1.0). r is _r<value> in the name (1.0 if absent)
- Point 1 (partial): proxy-training validation-loss difference (vs. baseline proxy; smaller predicted better)
Output: results/review2/baselines.csv (setting × all conditions / without label noise)
Example: python scripts/review2_baselines.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from augpred.results import SETTINGS, load  # noqa: E402

OUT = Path("results/review2")


def s_of(cond):
    if "roundtrip" in cond:
        return 0.0
    m = re.search(r"(?:^|_)s(\d\.\d+)", cond)
    return float(m.group(1)) if m else 1.0


def r_of(cond):
    m = re.search(r"_r(\d\.\d+)", cond)
    return float(m.group(1)) if m else 1.0


def proxy_val_loss(exp, cond, B):
    rs = [json.loads(f.read_text()) for f in sorted(Path(f"results/runs/proxyfit{B}__{exp}/{cond}").glob("seed*.json"))]
    return np.mean([r["history"][-1]["val_loss"] for r in rs]) if len(rs) == 3 else np.nan


def reference(exp):
    ref = yaml.safe_load(Path(f"configs/exp/{exp}.yaml").read_text()).get("reference", "baseline")
    return ref.split("/") if "/" in ref else (exp, ref)


def partial_spearman(x, y, controls):
    """Correlation between residuals after rank-transforming and regressing on the ranks of controls."""
    rk = stats.rankdata
    X = np.column_stack([np.ones(len(x))] + [rk(z) for z in controls])
    res = lambda v: rk(v) - X @ np.linalg.lstsq(X, rk(v), rcond=None)[0]
    return stats.pearsonr(res(x), res(y))[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    d["s"], d["r"] = d.condition.map(s_of), d.condition.map(r_of)
    rows = []
    for name, sel, B in SETTINGS:
        t0 = d[sel(d)].copy()
        t0["val_loss_gap"] = [proxy_val_loss(e, c, B) - proxy_val_loss(*reference(e), B)
                              for e, c in zip(t0.experiment, t0.condition)]
        for subset, t in [("all", t0), ("no label noise", t0[~t0.noisy])]:
            t = t.dropna(subset=[f"G{B}", "delta"])
            row = {"setting": name, "conditions": subset, "n": len(t)}
            if len(t) >= 5:
                sp = lambda x: stats.spearmanr(x, t.delta)[0]
                row.update({"-G": sp(-t[f"G{B}"]), "s": sp(t.s), "s*r": sp(t.s * t.r),
                            "-G | s, r (partial)": partial_spearman(-t[f"G{B}"].to_numpy(), t.delta.to_numpy(), [t.s, t.r]),
                            "-proxy val loss": sp(-t.val_loss_gap), "P": sp(t[f"P{B}"])})
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "baselines.csv", index=False, float_format="%.6f")
    with pd.option_context("display.width", 200):
        print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
