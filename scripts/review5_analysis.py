"""Extra check 5 (points 6, 7) verdict: shortcut / near-duplicates (shortcut_gen) and Flowers102 + TrivialAugment (flowers_ta_gen).
G follows the paper's definition (real loss − generated loss; larger is better).

Example: python scripts/review5_analysis.py predict   # print the G table from proxy training only (before production training)
    python scripts/review5_analysis.py           # also use production results for the verdict. Saved to results/review5/*.csv
"""
import json
import sys
from pathlib import Path
Path("results/tables").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from scipy import stats

RUNS = Path("results/runs")
OUT = Path("results/review5")
# shortcut conditions and the original conditions they are compared with (fixedsteps_gen)
PAIRS = {"img2img_s1.0_patch_r1.0": "img2img_s1.0_r1.0", "roundtrip_patch_r1.0": "roundtrip_r1.0",
         "img2img_s1.0_dup1_r1.0": "img2img_s1.0_r1.0", "img2img_s1.0_dup5_r1.0": "img2img_s1.0_r1.0"}


def runs(exp, cond):
    return [json.loads(f.read_text()) for f in sorted((RUNS / exp / cond).glob("seed*.json"))]


def G(exp, cond):
    r = runs(exp, cond)
    return np.mean([x["fit_loss_real"] - x["fit_loss_gen"] for x in r]) if r else np.nan


def acc(exp, cond):
    return np.array([x["test_acc"] for x in runs(exp, cond)]) * 100


def shortcut(full):
    base = acc("fixedsteps_gen", "baseline") if full else None
    rows = []
    for c, src in PAIRS.items():
        row = dict(condition=c, source=src, G=G("proxyfit1000__shortcut_gen", c), G_source=G("proxyfit1000__fixedsteps_gen", src))
        if full:
            a, s = acc("shortcut_gen", c), acc("fixedsteps_gen", src)
            row.update(n=len(a), delta=a.mean() - base.mean(), delta_source=s.mean() - base.mean(),
                       p_vs_source=stats.ttest_ind(a, s, equal_var=False).pvalue)
            row["fooled"] = bool(row["G"] > row["G_source"] and row["delta"] < row["delta_source"] and row["p_vs_source"] < 0.05)
        rows.append(row)
    return pd.DataFrame(rows)


def flowers(full):
    conds = sorted(p.name for p in (RUNS / "proxyfit250__flowers_ta_gen").iterdir() if p.name != "baseline")
    fid = pd.read_csv("results/all_conditions.csv").query("group == 'flowers102'").set_index("condition").fid
    base = acc("flowers_ta_gen", "baseline") if full else None
    rows = []
    for c in conds:
        row = dict(condition=c, G=G("proxyfit250__flowers_ta_gen", c), fid=fid.get(c, np.nan), noisy="noise" in c)
        if full:
            a = acc("flowers_ta_gen", c)
            row.update(n=len(a), delta=a.mean() - base.mean(), p=stats.ttest_ind(a, base, equal_var=False).pvalue)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("G", ascending=False)


def main():
    full = "predict" not in sys.argv[1:]
    s, f = shortcut(full), flowers(full)
    print("== shortcut (CIFAR-100, ResNet-18)")
    print(s.round(3).to_string(index=False))
    print("\n== Flowers102 + TrivialAugment (ResNet-50 pretrained)")
    print(f.round(3).to_string(index=False))
    if not full:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "shortcut.csv", index=False)
    f.to_csv(OUT / "flowers_ta.csv", index=False)
    base = acc("flowers_ta_gen", "baseline")
    print(f"\nFlowers + TA baseline: {base.mean():.2f} ± {base.std(ddof=1):.2f}")
    nf = f[~f.noisy]
    for name, t in [("all", f), ("no label noise", nf)]:
        print(f"Flowers + TA, {name} (n={len(t)}): rho(G)={stats.spearmanr(t.G, t.delta)[0]:.2f}  "
              f"rho(-FID)={stats.spearmanr(-t.fid, t.delta)[0]:.2f}")
    print(f"Flowers + TA, Δ<0: {(f.delta < 0).sum()} / {len(f)}")
    # Reference: ρ(G, Δ) with the 4 shortcut conditions added to the existing 23
    d = pd.read_csv("results/all_conditions.csv").query("group in ['main', 'heldout']")
    g = np.r_[-d.G1000.to_numpy(), s.G.to_numpy()]
    dl = np.r_[d.delta.to_numpy(), s.delta.to_numpy()]
    print(f"CIFAR-100 ResNet-18: rho(G) 23 conditions = {stats.spearmanr(-d.G1000, d.delta)[0]:.2f}, "
          f"with shortcut ({len(g)}) = {stats.spearmanr(g, dl)[0]:.2f}")
    tables(s, f)


def tables(s, f):
    num = lambda x, k=2: f"{x:+.{k}f}".replace("+", "$+$").replace("-", "$-$")
    pv = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
    label = {"img2img_s1.0_patch_r1.0": "img2img $s{=}1.0$ + colour patch", "roundtrip_patch_r1.0": "Resize-only + colour patch",
             "img2img_s1.0_dup1_r1.0": "img2img $s{=}1.0$, 1 image per class repeated",
             "img2img_s1.0_dup5_r1.0": "img2img $s{=}1.0$, 5 images per class repeated",
             "img2img_s1.0_r1.0": "img2img $s{=}1.0$", "roundtrip_r1.0": "Resize-only"}
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Images & Compared with & $G$ & $G$ of source & $\Delta$ & $\Delta$ of source & $p$ \\", r"\midrule"]
    for x in s.itertuples():
        lines.append(f"{label[x.condition]} & {label[x.source]} & {num(x.G, 3)} & {num(x.G_source, 3)} & {num(x.delta)} & "
                     f"{num(x.delta_source)} & {pv(x.p_vs_source)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path("results/tables/shortcut.tex").write_text("\n".join(lines) + "\n")
    name = {"roundtrip_r1.0": "Resize-only", "img2img_s0.5_r1.0": "img2img $s{=}0.5$", "img2img_s1.0_r1.0": "img2img $s{=}1.0$",
            "img2img_s1.0_r0.5": "img2img $s{=}1.0$, $r{=}0.5$", "inpaint_center_m0.75_r1.0": "Inpaint centre 0.75",
            "img2img_s1.0_sketch_r1.0": "Pencil sketch", "img2img_s1.0_g1.0_r1.0": "Guidance 1.0",
            "s1.0_noise0.25": "Label noise 25\\%", "s1.0_noise0.5": "Label noise 50\\%"}
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Generated images & $\Delta$ over TA & $p$ & $G$ \\", r"\midrule"]
    for x in f.itertuples():
        lines.append(f"{name[x.condition]} & {num(x.delta)} & {pv(x.p)} & {num(x.G, 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path("results/tables/flowers_ta.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
