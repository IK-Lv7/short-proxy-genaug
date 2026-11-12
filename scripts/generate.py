"""Generate generative-augmentation images and cache them in data/generated/<variant>/. Skip if they already exist.

Example: python scripts/generate.py configs/gen/pilot.yaml [--only img2img_s0.5] [--limit 64]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from augpred.config import deep_update, load_yaml  # noqa: E402
from augpred.generate import generate_variant  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--only", nargs="*")
    p.add_argument("--limit", type=int, help="for sanity checks: generate only the first N images and save to _test_<name>")
    args = p.parse_args()
    gcfg = load_yaml(args.config)
    for name, v in gcfg["variants"].items():
        if args.only and name not in args.only:
            continue
        generate_variant(name, deep_update(gcfg["common"], v), limit=args.limit,
                         log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
