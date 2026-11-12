"""Verdict for the sixth check (G + duplication/diversity check). Compare Δ, G, and D of the 4 new conditions where diversity drops naturally (review6_gen)
with the original conditions. Compute D beforehand with scripts/review6_diversity.py.
G follows the paper's definition (real loss − generated loss; larger is better).
Output: results/review6/new_conditions.csv, results/tables/diversity.tex
Example: python scripts/review6_analysis.py
"""
import json
from pathlib import Path
Path("results/tables").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from scipy import stats

RUNS = Path("results/runs")
# new conditions and the original conditions they are compared with (experiment, condition)
SOURCE = {"img2img_s1.0_cseed_r1.0": ("fixedsteps_gen", "img2img_s1.0_r1.0"),
          "img2img_s0.85_cseed_r1.0": ("fixedsteps_gen", "img2img_s0.85_r1.0"),
          "img2img_s1.0_g15_r1.0": ("fixedsteps_gen", "img2img_s1.0_r1.0"),
          "inpaint_center_m0.75_cseed_r1.0": ("inpaint_gen", "inpaint_center_m0.75_r1.0")}
LABEL = {"img2img_s1.0_cseed_r1.0": "img2img $s{=}1.0$, one seed per class",
         "inpaint_center_m0.75_cseed_r1.0": "Inpaint centre 0.75, one seed per class",
         "img2img_s1.0_g15_r1.0": "img2img $s{=}1.0$, guidance 15",
         "img2img_s0.85_cseed_r1.0": "img2img $s{=}0.85$, one seed per class",
         "img2img_s1.0_r1.0": "img2img $s{=}1.0$", "img2img_s0.85_r1.0": "img2img $s{=}0.85$",
         "inpaint_center_m0.75_r1.0": "Inpaint centre 0.75"}


def runs(exp, cond):
    return [json.loads(f.read_text()) for f in sorted((RUNS / exp / cond).glob("seed*.json"))]


def acc(exp, cond):
    return np.array([r["test_acc"] for r in runs(exp, cond)]) * 100


def main():
    base = acc("fixedsteps_gen", "baseline")
    D = pd.read_csv("results/review6/diversity.csv").query("experiment == 'proxyfit1000__review6_gen'").set_index("condition").D
    rows = []
    for c, (se, sc) in SOURCE.items():
        a, s = acc("review6_gen", c), acc(se, sc)
        rows.append(dict(condition=c, n=len(a), G=np.mean([r["fit_loss_real"] - r["fit_loss_gen"] for r in runs("proxyfit1000__review6_gen", c)]),
                         D=D[c], delta=a.mean() - base.mean(), sd=a.std(ddof=1), p_vs_base=stats.ttest_ind(a, base, equal_var=False).pvalue,
                         source=sc, delta_source=s.mean() - base.mean(), p_vs_source=stats.ttest_ind(a, s, equal_var=False).pvalue))
    r = pd.DataFrame(rows).sort_values("G", ascending=False)
    r.to_csv("results/review6/new_conditions.csv", index=False)
    print(r.round(3).to_string(index=False))
    d = pd.read_csv("results/all_conditions.csv").query("group in ['main', 'heldout']")
    nn = d.method != "label_noise"
    print(f"rho(G, Δ): 23 conditions {stats.spearmanr(-d.G1000, d.delta)[0]:.2f}, "
          f"with the 4 new ({len(d) + 4}) {stats.spearmanr(np.r_[-d.G1000, r.G], np.r_[d.delta, r.delta])[0]:.2f}, "
          f"without label noise ({nn.sum() + 4}) {stats.spearmanr(np.r_[-d.G1000[nn], r.G], np.r_[d.delta[nn], r.delta])[0]:.2f}")

    num = lambda x, k=2: f"{x:+.{k}f}".replace("+", "$+$").replace("-", "$-$")
    pv = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Images & Compared with & $G$ & $D$ & $\Delta$ & $\Delta$ of source & $p$ \\", r"\midrule"]
    for x in r.itertuples():
        lines.append(f"{LABEL[x.condition]} & {LABEL[x.source]} & {num(x.G, 3)} & {x.D:.2f} & {num(x.delta)} & "
                     f"{num(x.delta_source)} & {pv(x.p_vs_source)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path("results/tables/diversity.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
