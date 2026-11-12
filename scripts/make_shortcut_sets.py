"""Build "too easy" image sets from existing generated images for the extra check 5 (point 7) sanity check. No diffusion model is used.

G treats "generated images are easier to fit = better", so it may overrate sets containing shortcuts or near-duplicates.
For CIFAR-100 subset 0 (ResNet-18), build the following sets in data/generated/.

- <src>_patch: draw a 6x6 square in a per-class color at the top-left (2..7 px) of each image. Colors map 1-to-1 to classes
    and do not appear in real images (including test). A shortcut where class cues exist only in generated images
    (src = img2img_s1.0: generated images + shortcut, src = resize_roundtrip: resized real images + shortcut)
- img2img_s1.0_dup<k>: for each class, pick only k images from img2img_s1.0 and repeat them to the original count (50 per class).
    Near-duplicates of a class "prototype" (exact copies; they vary slightly via crop/flip in training)

Example: python scripts/make_shortcut_sets.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("data/generated")
PATCH = (2, 8)  # row/column range (top-left of a 32px image)


def save(name, x, y, src_idx, base, extra):
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "images.npz", x=x, y=y, src_idx=src_idx)
    meta = json.loads((ROOT / base / "meta.json").read_text())
    meta.update(name=name, n=len(y), derived_from=base, derived=extra)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"{name}: {len(y)} images")


def patch(base, colours):
    d = np.load(ROOT / base / "images.npz")
    x = d["x"].copy()
    a, b = PATCH
    x[:, a:b, a:b, :] = colours[d["y"]][:, None, None, :]
    save(f"{base}_patch", x, d["y"], d["src_idx"], base,
         dict(kind="class_colour_patch", rows_cols=list(PATCH), colour_seed=0))


def dup(base, k, seed=0):
    d = np.load(ROOT / base / "images.npz")
    rng = np.random.default_rng(seed)
    idx = np.empty(len(d["y"]), dtype=np.int64)
    for c in np.unique(d["y"]):
        members = np.flatnonzero(d["y"] == c)
        keep = rng.choice(members, size=k, replace=False)
        idx[members] = np.resize(keep, len(members))  # repeat the selected k images
    save(f"{base}_dup{k}", d["x"][idx], d["y"][idx], d["src_idx"][idx], base,
         dict(kind="near_duplicate", per_class_unique=k, seed=seed))


def main():
    n_cls = int(np.load(ROOT / "img2img_s1.0" / "images.npz")["y"].max()) + 1
    colours = np.random.default_rng(0).integers(0, 256, size=(n_cls, 3), dtype=np.uint8)
    patch("img2img_s1.0", colours)
    patch("resize_roundtrip", colours)
    dup("img2img_s1.0", 1)
    dup("img2img_s1.0", 5)


if __name__ == "__main__":
    main()
