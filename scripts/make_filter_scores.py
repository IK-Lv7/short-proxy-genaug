"""Average per-image losses saved during proxy training (results/sample_losses/) over seeds, to use as scores for removal/replacement experiments.

Only ratio 1.0 conditions are handled (all generated images are used in generated-set order).
Output: results/filter_scores/<experiment>__<condition>.npy (losses in generated-set order)
Example: python scripts/make_filter_scores.py proxysample1000__labelnoise_gen/s1.0_noise0.5 proxysample1000__fixedsteps_gen/img2img_s1.0_r1.0
"""
import sys
from pathlib import Path

import numpy as np

SRC, OUT = Path("results/sample_losses"), Path("results/filter_scores")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for key in sys.argv[1:]:
        ds = [np.load(f) for f in sorted((SRC / key).glob("seed*.npz"))]
        if len(ds) < 3:
            raise ValueError(f"{key}: only {len(ds)} seeds")
        if any(not np.array_equal(d["src_idx"], ds[0]["src_idx"]) for d in ds):
            raise ValueError(f"{key}: image order differs across seeds")
        if not np.array_equal(np.sort(ds[0]["src_idx"]), ds[0]["src_idx"]):
            raise ValueError(f"{key}: order does not match the generated set (only ratio 1.0 conditions are supported)")
        score = np.mean([d["loss_gen"] for d in ds], 0)
        out = OUT / (key.replace("/", "__") + ".npy")
        np.save(out, score)
        sw = ds[0]["swapped"]
        top = np.argsort(-score)[:len(score) // 4]
        print(f"{out}: n={len(score)} mean={score.mean():.3f}"
              + (f" label noise {sw.mean():.0%} → {sw[top].mean():.0%} in top-25% loss" if sw.any() else ""))


if __name__ == "__main__":
    main()
