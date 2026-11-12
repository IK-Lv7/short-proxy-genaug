"""Download Food-101 from Hugging Face (ethz/food101) and convert to uint8 npz at 256x256 (resize short side to 256, center crop).

Keeping everything (75,750 train / 25,250 test) at 256px would take about 20 GB, so select a subset per class with a fixed seed:
  x_train, y_train: 100/class from official train (10,100 images). Split into 50 train / 50 val is decided in splits/
  x_test,  y_test : 100/class from official test ("validation" in HF) (10,100 images)
src/augpred/food101_classes.json: class names in label order (from parquet features, with "_" replaced by a space)
Source data (parquet) goes in <root>/raw.
Image use is limited to research purposes per the provider (ETH Zürich). Do not redistribute, including generated images.
Example: python scripts/prepare_food101.py
"""
import argparse
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

BASE = "https://huggingface.co/datasets/ethz/food101/resolve/main/data"
FILES = [f"train-{i:05d}-of-00008.parquet" for i in range(8)] + [f"validation-{i:05d}-of-00003.parquet" for i in range(3)]
CLASSES_OUT = Path(__file__).resolve().parents[1] / "src/augpred/food101_classes.json"
SIZE, PER_CLASS, SEED = 256, 100, 0


def fetch(name, raw):
    path = raw / name
    if not path.exists():
        print(f"download {name}", flush=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(f"{BASE}/{name}", tmp)
        tmp.rename(path)
    return path


def load_image(b):
    im = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = im.size
    s = SIZE / min(w, h)
    im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - SIZE) // 2, (h - SIZE) // 2
    return np.asarray(im.crop((l, t, l + SIZE, t + SIZE)))


def class_names(path):
    meta = json.loads(pq.read_schema(path).metadata[b"huggingface"])
    return meta["info"]["features"]["label"]["names"]


def build(paths, rng):
    """Read only the label column of the parquet, select PER_CLASS images per class, and decode only the selected rows."""
    labels = np.concatenate([pq.read_table(p, columns=["label"])["label"].to_numpy() for p in paths])
    offsets = np.cumsum([0] + [pq.ParquetFile(p).metadata.num_rows for p in paths])
    pick = np.sort(np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:PER_CLASS] for c in range(labels.max() + 1)]))
    xs, ys = [], []
    for k, p in enumerate(paths):
        rows = pick[(pick >= offsets[k]) & (pick < offsets[k + 1])] - offsets[k]
        if len(rows) == 0:
            continue
        t = pq.read_table(p).take(rows)
        imgs, lab = t["image"].to_pylist(), t["label"].to_numpy()
        xs += [load_image(im["bytes"]) for im in imgs]
        ys.append(lab)
        print(f"{p.name}: {len(rows)} images", flush=True)
    return np.stack(xs), np.concatenate(ys).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/food101")
    args = ap.parse_args()
    raw = Path(args.root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    paths = [fetch(f, raw) for f in FILES]
    for p in paths:
        print(f"{p.name} sha256={hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")
    names = [n.replace("_", " ") for n in class_names(paths[0])]
    assert len(names) == 101 and len(set(names)) == 101
    CLASSES_OUT.write_text(json.dumps({"names": names}, indent=0))
    rng = np.random.default_rng(SEED)
    x_tr, y_tr = build([p for p in paths if p.name.startswith("train")], rng)
    x_te, y_te = build([p for p in paths if p.name.startswith("validation")], rng)
    print("train", x_tr.shape, np.bincount(y_tr).min(), np.bincount(y_tr).max(),
          "test", x_te.shape, np.bincount(y_te).min(), np.bincount(y_te).max())
    np.savez(Path(args.root) / "food101.npz", x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)
    print(f"saved {Path(args.root) / 'food101.npz'}")


if __name__ == "__main__":
    main()
