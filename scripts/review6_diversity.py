"""Combination of G and the duplication/diversity check (sixth check). For each CIFAR-100 subset 0 generated-image set, ratio of within-class diversity in CLIP features:

    D = (mean pairwise cosine distance within class of generated images) / (same quantity for real images)   (average over classes, then take the ratio)

is computed. No extra real data is used (only the 50/class training real images). CLIP is openai/clip-vit-base-patch32, same as DS.
Label-noise conditions and conditions with different mix ratios use the same D as the original image set (computed from the images actually used in training via load_generated).
Output: results/review6/diversity.csv (experiment, condition, D)
Example: python scripts/review6_diversity.py shortcut_gen fixedsteps_gen ...
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from augpred import ds as DS  # noqa: E402
from augpred.data import get_split, load_dataset, load_generated  # noqa: E402
from augpred.metrics import within_class_diversity  # noqa: E402

OUT = Path("results/review6/diversity.csv")


def div(x, y):
    return within_class_diversity(torch.as_tensor(DS.clip_embed(x)), y)


def main(exps):
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["experiment"], r["condition"]) for r in rows}
    ref = None
    for exp in exps:
        for d in sorted(Path(f"results/runs/{exp}").iterdir()):
            if (exp, d.name) in done or not list(d.glob("seed*.json")):
                continue
            cfg = json.loads(sorted(d.glob("seed*.json"))[0].read_text())["config"]
            dcfg, gcfg = cfg["data"], cfg["aug"].get("generated")
            if not gcfg or dcfg.get("name", "cifar100") != "cifar100" or dcfg["subset_seed"] != 0:
                continue
            if ref is None:
                x_tr, y_tr, _, _ = load_dataset(dcfg["root"], "cifar100")
                idx = np.asarray(get_split(y_tr, dcfg["n_per_class"], dcfg["n_val_per_class"], 0, "cifar100")["train"])
                ref = div(x_tr[idx], y_tr[idx])
                print(f"real: {ref:.4f}", flush=True)
            x, y = load_generated(gcfg, dcfg, n_real=len(idx))
            x = torch.as_tensor(np.asarray(x)) if not torch.is_tensor(x) else x
            if x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2).contiguous()
            D = div(x, np.asarray(y)) / ref
            rows.append(dict(experiment=exp, condition=d.name, D=D))
            print(f"{exp}/{d.name}: D={D:.3f}", flush=True)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
