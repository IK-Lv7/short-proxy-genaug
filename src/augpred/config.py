import copy
from pathlib import Path

import yaml


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def expand_experiment(exp_path, base_path="configs/base.yaml"):
    """Expand an experiment file into a list of (condition name, seed, full config)."""
    base = load_yaml(base_path)
    exp = load_yaml(exp_path)
    base = deep_update(base, exp.get("overrides", {}))
    runs = []
    for cond_name, cond in exp["conditions"].items():
        cfg = deep_update(base, cond or {})
        for seed in exp["seeds"]:
            runs.append((cond_name, seed, cfg))
    return exp, runs


def run_path(results_root, exp_name, cond_name, seed) -> Path:
    return Path(results_root) / "runs" / exp_name / cond_name / f"seed{seed}.json"
