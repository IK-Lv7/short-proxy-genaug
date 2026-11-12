"""Train all condition × seed combinations in an experiment file. Skip runs that already have results.

Example: python scripts/run_exp.py configs/exp/phase1_seed_variance.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from augpred.config import deep_update, expand_experiment, run_path  # noqa: E402
from augpred.train import save_result, train_one  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exp")
    p.add_argument("--results", default="results")
    p.add_argument("--only", nargs="*", help="run only these conditions")
    p.add_argument("--force", action="store_true", help="overwrite existing results")
    # For early-training metrics (proxy training): run the same conditions with short iterations and save under a different experiment name
    p.add_argument("--steps", type=int, help="override train.steps")
    p.add_argument("--seeds", type=int, nargs="*", help="override seeds")
    p.add_argument("--prefix", default="", help="prepend to the experiment name (e.g. proxy1000__)")
    p.add_argument("--stop-at", type=int, help="stop at this iteration while keeping the LR schedule. "
                   "record validation accuracy every stop_at/10 (for learning-curve extrapolation)")
    p.add_argument("--fit-checkpoints", type=int, help="record per-image outputs on training data k times during training (for Cartography/EL2N)")
    p.add_argument("--gen-ratio", type=float, help="override the generated-image mix ratio (for experiments computing G from a subset of generated images). "
                   "skip conditions without generated images and those whose mix ratio is not 1.0")
    p.add_argument("--sample-loss-dir", help="save per-image losses to <dir>/<experiment>/<condition>/seed<n>.npz (for analysis)")
    args = p.parse_args()

    exp, runs = expand_experiment(args.exp)
    exp = {**exp, "name": args.prefix + exp["name"]}
    if args.seeds is not None:
        runs = [(c, s, cfg) for c, s, cfg in runs if s == runs[0][1]]
        runs = [(c, s, cfg) for c, _, cfg in runs for s in args.seeds]
    if args.steps:
        runs = [(c, s, deep_update(cfg, {"train": {"steps": args.steps, "record_every": max(args.steps // 10, 1)}}))
                for c, s, cfg in runs]
    if args.stop_at:
        runs = [(c, s, deep_update(cfg, {"train": {"stop_at": args.stop_at, "record_every": max(args.stop_at // 10, 1),
                                                    "val_every_record": True}})) for c, s, cfg in runs]
    if args.fit_checkpoints:
        runs = [(c, s, deep_update(cfg, {"train": {"fit_checkpoints": args.fit_checkpoints}})) for c, s, cfg in runs]
    if args.gen_ratio:
        runs = [(c, s, deep_update(cfg, {"aug": {"generated": {"ratio": args.gen_ratio}}})) for c, s, cfg in runs
                if (cfg["aug"].get("generated") or {}).get("ratio") == 1.0]
    for cond, seed, cfg in runs:
        if args.only and cond not in args.only:
            continue
        out = run_path(args.results, exp["name"], cond, seed)
        if out.exists() and not args.force:
            print(f"[skip] {cond} seed={seed}")
            continue
        if args.sample_loss_dir:
            cfg = deep_update(cfg, {"train": {"save_sample_losses": f"{args.sample_loss_dir}/{exp['name']}/{cond}/seed{seed}.npz"}})
        print(f"[run] {exp['name']} / {cond} / seed={seed}", flush=True)
        res = train_one(cfg, seed, log=lambda s: print(s, flush=True))
        res.update(experiment=exp["name"], condition=cond)
        save_result(res, out)
        print(f"[done] {cond} seed={seed} test_acc={res['test_acc']:.4f} ({res['wall_sec']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
