"""Compute Discriminative Span (multi-class, src/augpred/ds.py) for each generative-augmentation condition and save to results/ds_metrics.csv.

Example: python scripts/compute_ds.py configs/exp/fixedsteps_gen.yaml configs/exp/tin_gen.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from augpred import ds as DS  # noqa: E402
from augpred.config import expand_experiment  # noqa: E402
from augpred.data import dataset_info, get_split, load_dataset, load_generated  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exps", nargs="+")
    p.add_argument("--out", default="results/ds_metrics.csv")
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
                emb = DS.clip_embed(x_tr[idx])
                W, acc = DS.fit_linear(emb, y_tr[idx], dataset_info(dcfg)["num_classes"])
                refs[key] = dict(emb=emb, W=W, pos={int(i): k for k, i in enumerate(idx)})
                print(f"ref {key}: linear probe train acc={acc:.3f}", flush=True)
            ref = refs[key]
            x_gen, y_gen, src = load_generated(g, dcfg, n_real=len(ref["emb"]), return_src=True)
            e_gen = DS.clip_embed(x_gen)
            e_src = ref["emb"][[ref["pos"][int(s)] for s in src]]
            row = {"experiment": exp["name"], "condition": cond, **DS.discriminative_span(ref["W"], e_gen, e_src, y_gen)}
            rows.append(row)
            print(row, flush=True)
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
