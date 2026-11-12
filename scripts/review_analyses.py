"""Analyses for responding to review (existing logs only, no training). Results are saved to results/review/.

R1: Spearman on conditions without label noise only (all predictors, per setting), with condition-level bootstrap 95% CI
R2: decomposition of G = L_gen - L_real. Spearman on the subset with mix ratio fixed at 1.0. Whether L_real moves with mix ratio
R3: using the P / G of the resize-only control (roundtrip) as the direction threshold instead of 0 (post-hoc exploration)
R4: multiple-comparison correction for significance (Holm and Benjamini-Hochberg)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("results/review")
SETTINGS = [
    ("CIFAR-100, ResNet-18, SD1.5", lambda d: d.group.isin(["main", "heldout"]), 1000),
    ("CIFAR-100, ResNet-18, SDXL", lambda d: d.group == "sdxl", 1000),
    ("CIFAR-100 (2nd subset)", lambda d: d.group == "subset1", 1000),
    ("CIFAR-100, VGG-11", lambda d: d.group == "vgg11", 1000),
    ("Tiny-ImageNet, ResNet-18", lambda d: d.group == "tinyimagenet", 1000),
    ("Flowers102, ResNet-50 (pretrained)", lambda d: d.group == "flowers102", 250),
]
PRED = {"FID": lambda s, B: -s.fid, "KID": lambda s, B: -s.kid, "Label acc.": lambda s, B: s.label_acc,
        "Affinity": lambda s, B: s.affinity, "DS": lambda s, B: s.ds_nnls,
        "P": lambda s, B: s[f"P{B}"], "-G": lambda s, B: -s[f"G{B}"]}


def load():
    d = pd.read_csv("results/all_conditions.csv")
    d = d[d.group != "conventional"].copy()
    src = d[d.group.isin(["main", "heldout"])].set_index("condition")
    for c in ["fid", "kid", "label_acc", "affinity", "ds_nnls"]:  # VGG-11 uses the same generated images
        m = d.group == "vgg11"
        d.loc[m, c] = d.loc[m, "condition"].map(src[c])
    d["noisy"] = d.method == "label_noise"
    return d


def boot_ci(x, y, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) < 3 or len(np.unique(y[i])) < 3:
            continue
        vals.append(stats.spearmanr(x[i], y[i])[0])
    return np.percentile(vals, [2.5, 97.5])


def r1(d):
    rows = []
    for name, sel, B in SETTINGS:
        for subset, m in [("all", np.ones(len(d), bool)), ("no label noise", ~d.noisy)]:
            s = d[sel(d) & m]
            for pname, f in PRED.items():
                v = f(s, B)
                if v.isna().any() or len(s) < 4:
                    continue
                r = stats.spearmanr(v, s.delta)[0]
                lo, hi = boot_ci(v, s.delta) if len(s) >= 6 else (np.nan, np.nan)
                rows.append(dict(setting=name, subset=subset, n=len(s), predictor=pname, rho=r, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def fit_losses(exp, cond, B):
    rs = [json.loads(f.read_text()) for f in Path(f"results/runs/proxyfit{B}__{exp}/{cond}").glob("seed*.json")]
    if not rs or "fit_loss_gen" not in rs[0]:
        return np.nan, np.nan
    return np.mean([r["fit_loss_gen"] for r in rs]), np.mean([r["fit_loss_real"] for r in rs])


def r2(d):
    rows = []
    for name, sel, B in SETTINGS:
        s = d[sel(d)]
        for _, r in s.iterrows():
            lg, lr = fit_losses(r.experiment, r.condition, B)
            ratio = 0.25 if "r0.25" in r.condition else 0.5 if "r0.5" in r.condition else 1.0
            rows.append(dict(setting=name, condition=r.condition, ratio=ratio, noisy=r.noisy, delta=r.delta,
                             L_gen=lg, L_real=lr, G=lg - lr))
    df = pd.DataFrame(rows)
    summ = []
    for name, s in df.groupby("setting", sort=False):
        for sub, m in [("ratio=1.0", s.ratio == 1.0), ("ratio=1.0, no label noise", (s.ratio == 1.0) & ~s.noisy)]:
            t = s[m]
            if len(t) < 4:
                continue
            summ.append(dict(setting=name, subset=sub, n=len(t),
                             rho_negG=stats.spearmanr(-t.G, t.delta)[0],
                             rho_negLgen=stats.spearmanr(-t.L_gen, t.delta)[0],
                             rho_negLreal=stats.spearmanr(-t.L_real, t.delta)[0]))
    return df, pd.DataFrame(summ)


def r3(d):
    rows = []
    for name, sel, B in SETTINGS:
        s = d[sel(d)]
        rt = s[s.method == "roundtrip"]
        sig = s[(s.p_vs_baseline < 0.05) & (s.method != "roundtrip")]
        if rt.empty:
            continue
        p0, g0 = rt[f"P{B}"].iloc[0], rt[f"G{B}"].iloc[0]
        rows.append(dict(setting=name, n_sig=len(sig),
                         P_thr0=((sig[f"P{B}"] > 0) == (sig.delta > 0)).sum(),
                         P_thr_roundtrip=((sig[f"P{B}"] > p0) == (sig.delta > 0)).sum(),
                         G_thr0=((sig[f"G{B}"] <= 0) == (sig.delta > 0)).sum(),
                         G_thr_roundtrip=((sig[f"G{B}"] <= g0) == (sig.delta > 0)).sum(), P_roundtrip=p0, G_roundtrip=g0))
    return pd.DataFrame(rows)


def r4(d):
    p = d.p_vs_baseline.to_numpy()
    n = len(p)
    order = np.argsort(p)
    holm = np.empty(n)
    holm[order] = np.maximum.accumulate(np.minimum(1, (n - np.arange(n)) * p[order]))
    bh = np.empty(n)
    bh[order[::-1]] = np.minimum.accumulate((p[order] * n / (np.arange(n) + 1))[::-1])
    d = d.assign(p_holm=holm, p_bh=np.minimum(bh, 1))
    return d, dict(n=n, raw=(p < 0.05).sum(), holm=(holm < 0.05).sum(), bh=(d.p_bh < 0.05).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    t1 = r1(d)
    t1.to_csv(OUT / "r1_noise_free_rho.csv", index=False)
    piv = t1.pivot_table(index=["setting", "subset", "n"], columns="predictor", values="rho", sort=False)
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 250):
        print("R1 Spearman (all vs no label noise)")
        print(piv[[c for c in PRED if c in piv]].to_string())
        print(t1[t1.predictor.isin(["-G", "P", "FID"])].round(2).to_string(index=False))
    df2, s2 = r2(d)
    df2.to_csv(OUT / "r2_G_decomposition.csv", index=False)
    s2.to_csv(OUT / "r2_ratio_fixed.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 250):
        print("\nR2 mixing-ratio series (CIFAR SD1.5 / TIN, s1.0)")
        print(df2[df2.condition.str.match(r"img2img_s1\.0_r")].to_string(index=False))
        print(s2.to_string(index=False))
    t3 = r3(d)
    t3.to_csv(OUT / "r3_sign_threshold.csv", index=False)
    print("\nR3 sign agreement with threshold 0 vs roundtrip control")
    print(t3.round(3).to_string(index=False))
    d4, s4 = r4(d)
    d4[["group", "condition", "delta", "p_vs_baseline", "p_holm", "p_bh"]].to_csv(OUT / "r4_multiple_testing.csv", index=False)
    print("\nR4 significant conditions:", s4)
    lost = d4[(d4.p_vs_baseline < 0.05) & (d4.p_holm >= 0.05)]
    print("lost under Holm:", lost[["group", "condition", "delta", "p_vs_baseline", "p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
