"""Compute extra baselines for review (Major 6) and save to results/metrics2.csv (src/augpred/metrics2.py).

Example: python scripts/compute_metrics2.py configs/exp/fixedsteps_gen.yaml configs/exp/tin_gen.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from augpred import metrics as M  # noqa: E402
from augpred import metrics2 as M2  # noqa: E402
from augpred.config import expand_experiment  # noqa: E402
from augpred.data import get_split, load_dataset, load_generated  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exps", nargs="+")
    p.add_argument("--out", default="results/metrics2.csv")
    args = p.parse_args()
    refs, rows = {}, []
    for exp_path in args.exps:
        exp, runs = expand_experiment(exp_path)
        seen = set()
        for cond, _, cfg in runs:
            g = cfg["aug"].get("generated")
            if not g or cond in seen:
                continue
            seen.add(cond)
            dcfg = cfg["data"]
            name = dcfg.get("name", "cifar100")
            key = (name, dcfg["subset_seed"])
            if key not in refs:
                x_tr, y_tr, _, _ = load_dataset(dcfg["root"], name)
                split = get_split(y_tr, dcfg["n_per_class"], dcfg["n_val_per_class"], dcfg["subset_seed"], name)
                idx = np.asarray(split["train"])
                x, y = x_tr[idx], y_tr[idx]
                refs[key] = dict(y=y, inc=M.inception_features(x), dino=M2.dinov2_features(x))
                print(f"ref {key} done", flush=True)
            ref = refs[key]
            x_gen, y_gen = load_generated(g, dcfg, n_real=len(ref["y"]))
            inc, dino = M.inception_features(x_gen), M2.dinov2_features(x_gen)
            row = {"experiment": exp["name"], "condition": cond, **M2.prdc(ref["inc"], inc),
                   "cc_kid": M2.class_conditional_kid(ref["inc"], ref["y"], inc, y_gen),
                   "fd_dinov2": M.fid_from_features(dino, ref["dino"])}
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    out = Path(args.out)
    df = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        key = set(zip(df.experiment, df.condition))
        df = pd.concat([old[[k not in key for k in zip(old.experiment, old.condition)]], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
