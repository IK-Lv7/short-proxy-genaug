"""Download Oxford Flowers102 from the official source and convert to uint8 npz at 256x256 (resize short side to 256, center crop).

Output: <root>/flowers102.npz
  x_train, y_train: official train + val (20/class, 2040 images). Split into train/val is decided in splits/
  x_test, y_test: official test (6149 images)
src/augpred/flowers102_classes.json: class names in label order (from the README of Hugging Face dpdl-benchmark/oxford_flowers102)
"""
import argparse
import hashlib
import json
import re
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

BASE = "https://thor.robots.ox.ac.uk/flowers/102"
NAMES_URL = "https://huggingface.co/datasets/dpdl-benchmark/oxford_flowers102/resolve/main/README.md"
CLASSES_OUT = Path(__file__).resolve().parents[1] / "src/augpred/flowers102_classes.json"
SIZE = 256


def fetch(url, path):
    if not path.exists():
        print(f"download {url}")
        urllib.request.urlretrieve(url, path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{path.name} sha256={h}")
    return path


def load_image(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = SIZE / min(w, h)
    im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - SIZE) // 2, (h - SIZE) // 2
    return np.asarray(im.crop((l, t, l + SIZE, t + SIZE)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/flowers102")
    args = p.parse_args()
    raw = Path(args.root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    readme = fetch(NAMES_URL, raw / "hf_README.md").read_text()
    names = [n.strip() for _, n in sorted(((int(i), n) for i, n in re.findall(r"'(\d+)': (.+)", readme)))]
    assert len(names) == 102 and len(set(names)) == 102
    CLASSES_OUT.write_text(json.dumps({"names": names}, indent=0))

    tgz = fetch(f"{BASE}/102flowers.tgz", raw / "102flowers.tgz")
    labels = loadmat(fetch(f"{BASE}/imagelabels.mat", raw / "imagelabels.mat"))["labels"][0] - 1  # official labels are 1-based
    setid = loadmat(fetch(f"{BASE}/setid.mat", raw / "setid.mat"))
    img_dir = raw / "jpg"
    if not img_dir.exists():
        with tarfile.open(tgz) as t:
            members = [m for m in t.getmembers() if m.isfile() and re.fullmatch(r"jpg/image_\d{5}\.jpg", m.name)]
            t.extractall(raw, members=members, filter="data")

    def build(ids):
        ids = np.sort(np.asarray(ids).ravel())
        x = np.stack([load_image(img_dir / f"image_{i:05d}.jpg") for i in ids])
        return x, labels[ids - 1].astype(np.int64)

    x_tr, y_tr = build(np.concatenate([setid["trnid"].ravel(), setid["valid"].ravel()]))
    x_te, y_te = build(setid["tstid"])
    print("train", x_tr.shape, np.bincount(y_tr).min(), np.bincount(y_tr).max(), "test", x_te.shape)
    np.savez(Path(args.root) / "flowers102.npz", x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)
    print(f"saved {Path(args.root) / 'flowers102.npz'}")


if __name__ == "__main__":
    main()
