# eval_loads.py
from __future__ import annotations

import argparse
import os
import shutil

from compare_experiments import (
    DEFAULT_EXPERIMENT_STRING,
    build_comparison_policies,
    parse_experiment_names,
    run_load_comparison,
)
from config import EnvConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility entry for multi-load evaluation. "
            "Uses the same policy implementations as compare_experiments.py."
        )
    )
    parser.add_argument(
        "--experiments",
        default=os.getenv("COMPARE_EXPERIMENTS", DEFAULT_EXPERIMENT_STRING),
        help="Comma-separated experiment names.",
    )
    parser.add_argument("--episodes", type=int, default=int(os.getenv("COMPARE_EPISODES", "50")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("COMPARE_SEED", "123")))
    parser.add_argument(
        "--model-path",
        default=os.getenv("MAPPO_MODEL_PATH", os.path.join("results", "mappo_checkpoint.pt")),
    )
    parser.add_argument(
        "--pruned-model-path",
        default=os.getenv("PRUNED_MAPPO_MODEL_PATH", os.path.join("results", "mappo_actor_pruned.pt")),
    )
    parser.add_argument(
        "--distilled-model-path",
        default=os.getenv(
            "DISTILLED_MAPPO_MODEL_PATH",
            os.path.join("results", "mappo_actor_pruned_distilled.pt"),
        ),
    )
    parser.add_argument(
        "--prune-rates",
        default=os.getenv("PRUNE_RATES", ""),
        help="Comma-separated pruning rates, e.g. 0.10,0.25,0.40,0.50.",
    )
    parser.add_argument(
        "--pruned-model-template",
        default=os.getenv(
            "PRUNED_MODEL_TEMPLATE",
            os.path.join("results", "mappo_actor_pruned_{suffix}.pt"),
        ),
        help="Path template for rate-specific pruned actors.",
    )
    parser.add_argument(
        "--distilled-model-template",
        default=os.getenv(
            "DISTILLED_MODEL_TEMPLATE",
            os.path.join("results", "mappo_actor_pruned_distilled_{suffix}.pt"),
        ),
        help="Path template for rate-specific distilled actors.",
    )
    parser.add_argument("--device", default=os.getenv("COMPARE_DEVICE", "cpu"))
    parser.add_argument(
        "--save-dir",
        default=os.getenv("EVAL_LOADS_SAVE_DIR", os.path.join("results", "eval_loads")),
    )
    parser.add_argument("--load-min", type=float, default=float(os.getenv("LOAD_FACTOR_MIN", "0.70")))
    parser.add_argument("--load-max", type=float, default=float(os.getenv("LOAD_FACTOR_MAX", "1.30")))
    parser.add_argument("--load-points", type=int, default=int(os.getenv("NUM_LOAD_POINTS", "13")))
    return parser.parse_args()


def save_legacy_summary_name(save_dir: str) -> None:
    src = os.path.join(save_dir, "comparison_loads_summary.csv")
    dst = os.path.join(save_dir, "eval_loads_summary.csv")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print(f"Saved legacy CSV: {dst}")


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.load_points <= 0:
        raise ValueError("--load-points must be positive.")
    if args.load_min > args.load_max:
        raise ValueError("--load-min must be less than or equal to --load-max.")

    cfg = EnvConfig()
    experiment_names = parse_experiment_names(args.experiments)
    print("Mode: loads")
    print(f"Episodes per policy/load: {args.episodes}")

    policies = build_comparison_policies(cfg, experiment_names, args)
    print(f"Policies: {', '.join(policy.name for policy in policies)}")
    run_load_comparison(cfg, policies, args)
    save_legacy_summary_name(args.save_dir)


if __name__ == "__main__":
    main()
