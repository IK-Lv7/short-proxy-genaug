"""Download Tiny-ImageNet from Hugging Face (zh-plus/tiny-imagenet) and convert to a uint8 npz.

Output: <root>/tinyimagenet.npz  (x_train: (100000,64,64,3) uint8, y_train, x_test: official valid (10000,64,64,3), y_test)
      src/augpred/tinyimagenet_classes.json  (class names in label order; used in prompts)
The official test labels are not public, so the official valid set is used as test.
Terms: ImageNet Terms of Access (non-commercial research and education only).
"""
import argparse
import hashlib
import io
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

BASE = "https://huggingface.co/datasets/zh-plus/tiny-imagenet/resolve/main"
FILES = {"train": "data/train-00000-of-00001-1359597a978bc4fa.parquet",
         "test": "data/valid-00000-of-00001-70d52db3c749a935.parquet"}
CLASSES_OUT = Path(__file__).resolve().parents[1] / "src/augpred/tinyimagenet_classes.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, path):
    if not path.exists():
        print(f"download {url}")
        urllib.request.urlretrieve(url, path)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/tinyimagenet")
    args = p.parse_args()
    raw = Path(args.root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # Class names: wnid order in dataset_infos.json -> WordNet dict in classes.py (read as a string, not executed as code)
    infos = json.loads(fetch(f"{BASE}/dataset_infos.json", raw / "dataset_infos.json").read_text())
    wnids = next(iter(infos.values()))["features"]["label"]["names"]
    i2d = dict(re.findall(r'"(n\d{8})": "([^"]*)"', fetch(f"{BASE}/classes.py", raw / "classes.py.txt").read_text()))
    names = [i2d[w].split(",")[0].strip() for w in wnids]
    assert len(names) == 200 and len(set(names)) == 200
    CLASSES_OUT.write_text(json.dumps({"wnids": wnids, "names": names}, indent=0))

    arrays = {}
    for split, rel in FILES.items():
        path = fetch(f"{BASE}/{rel}", raw / Path(rel).name)
        print(f"{path.name} sha256={sha256(path)}")
        t = pq.read_table(path).to_pydict()
        arrays[f"x_{split}"] = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t["image"]])
        arrays[f"y_{split}"] = np.asarray(t["label"], dtype=np.int64)
        cnt = np.bincount(arrays[f"y_{split}"], minlength=200)
        print(split, arrays[f"x_{split}"].shape, "per class", cnt.min(), cnt.max())
    np.savez(Path(args.root) / "tinyimagenet.npz", **arrays)
    print(f"saved {Path(args.root) / 'tinyimagenet.npz'}")


if __name__ == "__main__":
    main()
