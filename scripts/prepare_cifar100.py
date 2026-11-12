"""Download CIFAR-100 from Hugging Face (uoft-cs/cifar100) and convert to a uint8 npz.

Downloading from the original source (cs.toronto.edu) is extremely slow, so use the HF mirror.
Output: <root>/cifar100.npz  (x_train: (50000,32,32,3) uint8, y_train, x_test, y_test are fine labels)
"""
import argparse
import hashlib
import io
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

BASE = "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100"
FILES = {"train": "train-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/cifar100")
    args = p.parse_args()
    root = Path(args.root)
    (root / "raw").mkdir(parents=True, exist_ok=True)

    arrays = {}
    for split, fname in FILES.items():
        path = root / "raw" / fname
        if not path.exists():
            print(f"download {fname}")
            urllib.request.urlretrieve(f"{BASE}/{fname}", path)
        print(f"{fname} sha256={sha256(path)}")
        t = pq.read_table(path).to_pydict()
        arrays[f"x_{split}"] = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t["img"]])
        arrays[f"y_{split}"] = np.asarray(t["fine_label"], dtype=np.int64)
        print(split, arrays[f"x_{split}"].shape, np.bincount(arrays[f"y_{split}"]).min())

    np.savez(root / "cifar100.npz", **arrays)
    print(f"saved {root / 'cifar100.npz'}")


if __name__ == "__main__":
    main()
