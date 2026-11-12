"""Verdict for the pre-registered Food-101 hypotheses: Δ (5 seeds, Welch + Holm), G (proxy 250 iter × 3 seeds), Spearman."""
import glob, json, sys
from pathlib import Path
Path("results/tables").mkdir(parents=True, exist_ok=True)
import numpy as np, pandas as pd
from scipy import stats

NOISE = ["s1.0_noise0.25", "s1.0_noise0.5"]
out = []
for m in ["r50", "vit"]:
    def load(exp):
        r = {}
        for f in glob.glob(f"results/runs/{exp}/*/seed*.json"):
            d = json.load(open(f)); r.setdefault(d["condition"], []).append(d)
        return r
    full, prox = load(f"food_{m}_gen"), load(f"proxyfit250__food_{m}_gen")
    base = np.array([d["test_acc"] for d in full["baseline"]]) * 100
    rows = []
    for c, ds in full.items():
        if c == "baseline": continue
        a = np.array([d["test_acc"] for d in ds]) * 100
        p = prox[c]
        G = np.mean([d["fit_loss_real"] - d["fit_loss_gen"] for d in p])
        rows.append(dict(model=m, cond=c, n=len(a), delta=a.mean() - base.mean(), sd=a.std(ddof=1), p=stats.ttest_ind(a, base, equal_var=False).pvalue, G=G))
    df = pd.DataFrame(rows)
    o = np.argsort(df.p.values); k = len(df)
    h = np.empty(k); h[o] = np.maximum.accumulate(np.minimum(1, (k - np.arange(k)) * df.p.values[o])); df["holm"] = h
    print(f"\n== {m}: baseline {base.mean():.2f} ± {base.std(ddof=1):.2f} (n={len(base)})")
    print(df.sort_values("G", ascending=False).round(3).to_string(index=False))
    nf = df[~df.cond.isin(NOISE)]
    print("Spearman(G,Δ) all:", round(stats.spearmanr(df.G, df.delta)[0], 3), " noise-free:", round(stats.spearmanr(nf.G, nf.delta)[0], 3))
    out.append(df)
pd.concat(out).to_csv("results/food_summary.csv", index=False)

# Paper table (results/tables/food.tex). FID depends only on generated images, so use the ResNet-50 row for both models
LABEL = {"sdxl_s1.0_r1.0": "SDXL $s{=}1.0$", "img2img_s1.0_r1.0": "img2img $s{=}1.0$", "img2img_s1.0_r0.5": "img2img $s{=}1.0$, ratio 0.5",
         "inpaint_center_m0.75_r1.0": "Inpaint centre 0.75", "img2img_s0.5_r1.0": "img2img $s{=}0.5$", "img2img_s1.0_sketch_r1.0": "Pencil sketch",
         "inpaint_border_k0.75_r1.0": "Inpaint border 0.75", "roundtrip_r1.0": "Resize-only", "img2img_s1.0_g1.0_r1.0": "Guidance 1.0",
         "s1.0_noise0.25": "Label noise 25\\%", "s1.0_noise0.5": "Label noise 50\\%"}
fid = pd.read_csv("results/metrics.csv").query("experiment == 'food_r50_gen'").set_index("condition").fid
ds = pd.read_csv("results/ds_metrics.csv").query("experiment == 'food_r50_gen'").set_index("condition").ds_nnls
r, v = (x.set_index("cond") for x in out)
num = lambda x, d: f"$-${abs(x):.{d}f}" if x < 0 else f"$+${x:.{d}f}"
dl = lambda row: num(row.delta, 2) + ("$^{*}$" if row.holm < 0.05 else "")
lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r" & \multicolumn{2}{c}{ResNet-50} & \multicolumn{2}{c}{ViT-B/16} & & \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Generated images & $\Delta$ & $G$ & $\Delta$ & $G$ & FID & DS \\", r"\midrule"]
for c in r.sort_values("G", ascending=False).index:
    lines.append(f"{LABEL[c]} & {dl(r.loc[c])} & {num(r.G[c], 3)} & {dl(v.loc[c])} & {num(v.G[c], 3)} & {fid[c]:.1f} & {ds[c]:.3f} \\\\")
rho = lambda d, nf: stats.spearmanr(d.G[~d.index.str.contains('noise')] if nf else d.G, d.delta[~d.index.str.contains('noise')] if nf else d.delta)[0]
def corr(m, d, nf):
    i = d.index[~d.index.str.contains('noise')] if nf else d.index
    return stats.spearmanr(m[i], d.delta[i])[0]
fr = lambda d, nf: corr(-fid, d, nf)
dr = lambda d, nf: corr(ds, d, nf)
lines += [r"\midrule", f"$\\rho$ with $\\Delta$, all 11 & & {rho(r, 0):.2f} & & {rho(v, 0):.2f} & {num(fr(r, 0), 2)} / {num(fr(v, 0), 2)} & {num(dr(r, 0), 2)} / {num(dr(v, 0), 2)} \\\\",
          f"$\\rho$ with $\\Delta$, without label noise & & {rho(r, 1):.2f} & & {rho(v, 1):.2f} & {num(fr(r, 1), 2)} / {num(fr(v, 1), 2)} & {num(dr(r, 1), 2)} / {num(dr(v, 1), 2)} \\\\",
          r"\bottomrule", r"\end{tabular}"]
Path("results/tables/food.tex").write_text("\n".join(lines) + "\n")
