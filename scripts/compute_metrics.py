"""Compute metrics for each generative-augmentation condition in experiment files and save to results/metrics.csv.

Example: python scripts/compute_metrics.py configs/exp/fixedsteps_gen.yaml configs/exp/labelnoise_gen.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from augpred import metrics as M  # noqa: E402
from augpred.config import expand_experiment, load_yaml  # noqa: E402
from augpred.data import get_split, load_dataset, load_generated  # noqa: E402
from augpred.models import build_model  # noqa: E402

MODELS = {"oracle": "data/models/oracle_seed0.pt", "affinity": "data/models/affinity_ref_val_seed0.pt"}


def load_model(path, cfg):
    m = build_model(cfg, num_classes=100).cuda()
    m.load_state_dict(torch.load(path, map_location="cuda"))
    return m


class Reference:
    """Quantities on the real-image side (computed once per dataset and subset).

    The oracle classifier and the Affinity reference model were trained for CIFAR-100 subset 0.
    For anything else (other subsets, other datasets), values would be biased by training-data overlap, so they are not computed
    (FID/KID only. For other CIFAR-100 subsets, label_acc, which only judges generated images, is also computed).
    """

    def __init__(self, dcfg, base):
        name = dcfg.get("name", "cifar100")
        x_tr, y_tr, _, _ = load_dataset(dcfg["root"], name)
        split = get_split(y_tr, dcfg["n_per_class"], dcfg["n_val_per_class"], dcfg["subset_seed"], name)
        self.tr_idx = np.asarray(split["train"])
        self.x, self.y = x_tr[self.tr_idx], y_tr[self.tr_idx]
        print(f"inception features (real: {name}, subset {dcfg['subset_seed']})", flush=True)
        self.inc = M.inception_features(self.x)
        self.full = name == "cifar100" and dcfg["subset_seed"] == 0
        self.oracle = load_model(MODELS["oracle"], base) if name == "cifar100" else None
        if self.full:
            self.aff = load_model(MODELS["affinity"], base)
            o_logits, self.o_feat = M.classifier_outputs(self.oracle, self.x)
            # The Affinity reference model is trained on validation images, so the training subset (source real images) is unseen data
            self.aff_acc = M.label_metrics(M.classifier_outputs(self.aff, self.x)[0], self.y)["acc"]
            self.div = M.within_class_diversity(self.o_feat, self.y)
            self.pos = {int(i): k for k, i in enumerate(self.tr_idx)}  # train index -> position in the training subset
            print(f"real: oracle_acc={M.label_metrics(o_logits, self.y)['acc']:.3f} "
                  f"affinity_real_acc={self.aff_acc:.3f} diversity={self.div:.4f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exps", nargs="+")
    p.add_argument("--out", default="results/metrics.csv")
    args = p.parse_args()
    base = load_yaml("configs/base.yaml")
    refs = {}

    rows = []
    for exp_path in args.exps:
        exp, runs = expand_experiment(exp_path)
        seen = set()
        for cond, _, cfg in runs:
            g = cfg["aug"].get("generated")
            if not g or cond in seen:
                continue
            seen.add(cond)
            dcfg = cfg["data"]
            key = (dcfg.get("name", "cifar100"), dcfg["subset_seed"])
            if key not in refs:
                refs[key] = Reference(dcfg, base)
            ref = refs[key]
            x_gen, y_gen, src = load_generated(g, dcfg, n_real=len(ref.y), return_src=True)
            inc_gen = M.inception_features(x_gen)
            row = {"experiment": exp["name"], "condition": cond, "n_gen": len(y_gen),
                   "fid": M.fid_from_features(inc_gen, ref.inc), "kid": M.kid_from_features(inc_gen, ref.inc)}
            if ref.oracle is not None:
                o_logits, o_feat = M.classifier_outputs(ref.oracle, x_gen)
                lab = M.label_metrics(o_logits, y_gen)
                row.update(label_acc=lab["acc"], label_prob=lab["prob"])
            if ref.full:
                aff_acc = M.label_metrics(M.classifier_outputs(ref.aff, x_gen)[0], y_gen)["acc"]
                div = M.within_class_diversity(o_feat, y_gen)
                row.update(mmd_oracle=M.kid_from_features(o_feat.numpy(), ref.o_feat.numpy()),
                           affinity_acc=aff_acc, affinity=aff_acc / ref.aff_acc,
                           diversity=div, diversity_ratio=div / ref.div,
                           novelty=M.novelty(o_feat, ref.o_feat[[ref.pos[int(s)] for s in src]]))
            rows.append(row)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    out = Path(args.out)
    df = pd.DataFrame(rows)
    if out.exists():  # same (experiment, condition) is overwritten; others are kept
        old = pd.read_csv(out)
        key = set(zip(df.experiment, df.condition))
        old = old[[k not in key for k in zip(old.experiment, old.condition)]]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
