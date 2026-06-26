from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

from temporal.config import EnvConfig
from temporal.experiments.common import (
    evaluate_policy,
    plot_bar_dashboard,
    plot_load_curves,
    print_summary_table,
    save_csv,
)
from temporal.experiments.greedy_policy import GreedyPolicy
from temporal.experiments.local_policy import LocalPolicy
from temporal.experiments.mappo_policy import MAPPOPolicy
from temporal.experiments.offload_policy import OffloadPolicy
from temporal.experiments.pruned_mappo_policy import PrunedMAPPOPolicy
from temporal.experiments.random_policy import RandomPolicy
from temporal.experiments.temporal_mappo_policy import TemporalMAPPOPolicy


PolicyFactory = Callable[[EnvConfig, argparse.Namespace], Any]
DEFAULT_EXPERIMENT_NAMES = [
    "mappo",
    "pruned_mappo",
    "distilled_mappo",
    "temporal_distilled_mappo",
    "offload",
    "greedy",
]
DEFAULT_EXPERIMENT_STRING = ",".join(DEFAULT_EXPERIMENT_NAMES)


def build_mappo(cfg: EnvConfig, args: argparse.Namespace) -> MAPPOPolicy:
    return MAPPOPolicy(cfg, model_path=args.model_path, device=args.device)


def build_pruned_mappo(cfg: EnvConfig, args: argparse.Namespace) -> PrunedMAPPOPolicy:
    return PrunedMAPPOPolicy(
        cfg,
        model_path=args.pruned_model_path,
        device=args.device,
        policy_name="pruned_mappo",
    )


def build_distilled_mappo(cfg: EnvConfig, args: argparse.Namespace) -> PrunedMAPPOPolicy:
    return PrunedMAPPOPolicy(
        cfg,
        model_path=args.distilled_model_path,
        device=args.device,
        policy_name="distilled_mappo",
    )


def build_temporal_distilled_mappo(
    cfg: EnvConfig,
    args: argparse.Namespace,
) -> TemporalMAPPOPolicy:
    return TemporalMAPPOPolicy(
        cfg,
        model_path=args.temporal_model_path,
        device=args.device,
        policy_name="temporal_distilled_mappo",
        max_repeat=args.temporal_max_repeat,
        repeat_scale=args.temporal_repeat_scale,
        safety_enabled=not args.disable_safety,
        safe_energy_min=args.safe_energy_min,
        safe_queue_threshold=args.safe_queue_threshold,
        safe_obs_change_threshold=args.safe_obs_change_threshold,
        safe_channel_drop_threshold=args.safe_channel_drop_threshold,
    )


def build_random(cfg: EnvConfig, args: argparse.Namespace) -> RandomPolicy:
    return RandomPolicy(seed=args.seed, offload_probability=args.random_offload_prob)


EXPERIMENTS: Dict[str, PolicyFactory] = {
    "mappo": build_mappo,
    "pruned_mappo": build_pruned_mappo,
    "distilled_mappo": build_distilled_mappo,
    "temporal_distilled_mappo": build_temporal_distilled_mappo,
    "local": lambda cfg, args: LocalPolicy(),
    "offload": lambda cfg, args: OffloadPolicy(),
    "random": build_random,
    "greedy": lambda cfg, args: GreedyPolicy(),
}


def parse_experiment_names(raw_names: str) -> List[str]:
    names = [item.strip().lower() for item in raw_names.split(",") if item.strip()]
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        valid = ", ".join(EXPERIMENTS)
        raise ValueError(f"Unknown experiments: {unknown}. Valid names: {valid}")
    return names


def build_policies(
    cfg: EnvConfig,
    names: List[str],
    args: argparse.Namespace,
) -> List[Any]:
    return [EXPERIMENTS[name](cfg, args) for name in names]


def run_fixed_comparison(
    cfg: EnvConfig,
    policies: List[Any],
    args: argparse.Namespace,
) -> None:
    all_results = []
    for policy in policies:
        print(f"Evaluating policy: {policy.name}")
        summary = evaluate_policy(
            cfg=cfg,
            policy=policy,
            num_eval_episodes=args.episodes,
            seed=args.seed,
        )
        all_results.append(summary)

    print_summary_table(all_results)
    save_csv(args.save_dir, all_results, "comparison_summary.csv")
    plot_bar_dashboard(args.save_dir, all_results)


def run_load_comparison(
    base_cfg: EnvConfig,
    policies: List[Any],
    args: argparse.Namespace,
) -> None:
    load_factors = np.linspace(args.load_min, args.load_max, args.load_points)
    all_results = []

    for idx, load_factor in enumerate(load_factors, start=1):
        task_min_bits = base_cfg.task_min_bits * float(load_factor)
        task_max_bits = base_cfg.task_max_bits * float(load_factor)
        cfg = replace(
            base_cfg,
            task_min_bits=task_min_bits,
            task_max_bits=task_max_bits,
        )
        load_name = f"load_{idx:02d}"
        avg_task_mbits = 0.5 * (task_min_bits + task_max_bits) / 1e6

        print(f"\n========== Load factor: {load_factor:.2f} ==========")
        print(f"task_min_bits={task_min_bits:.1e}, task_max_bits={task_max_bits:.1e}")

        for policy in policies:
            print(f"Evaluating policy: {policy.name}")
            result = evaluate_policy(
                cfg=cfg,
                policy=policy,
                num_eval_episodes=args.episodes,
                seed=args.seed,
            )
            result["load_name"] = load_name
            result["load_factor"] = float(load_factor)
            result["task_min_bits"] = task_min_bits
            result["task_max_bits"] = task_max_bits
            result["avg_task_mbits"] = avg_task_mbits
            all_results.append(result)

    save_csv(args.save_dir, all_results, "comparison_loads_summary.csv")
    plot_load_curves(
        save_dir=args.save_dir,
        all_results=all_results,
        policies=[policy.name for policy in policies],
    )
    print("\nDone. Temporal load comparison results have been saved.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated temporal-distillation comparison experiments."
    )
    parser.add_argument(
        "--experiments",
        default=os.getenv("TEMPORAL_COMPARE_EXPERIMENTS", DEFAULT_EXPERIMENT_STRING),
        help=f"Comma-separated experiment names. Valid names: {', '.join(EXPERIMENTS)}.",
    )
    parser.add_argument(
        "--mode",
        choices=["fixed", "loads"],
        default=os.getenv("TEMPORAL_COMPARE_MODE", "fixed"),
    )
    parser.add_argument("--episodes", type=int, default=int(os.getenv("TEMPORAL_COMPARE_EPISODES", "50")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("TEMPORAL_COMPARE_SEED", "123")))
    parser.add_argument(
        "--model-path",
        default=os.getenv("TEMPORAL_MAPPO_MODEL_PATH", os.path.join("results", "mappo_checkpoint.pt")),
    )
    parser.add_argument(
        "--pruned-model-path",
        default=os.getenv(
            "TEMPORAL_PRUNED_MODEL_PATH",
            os.path.join("temporal", "results", "mappo_actor_pruned_p25.pt"),
        ),
    )
    parser.add_argument(
        "--distilled-model-path",
        default=os.getenv(
            "TEMPORAL_DISTILLED_MODEL_PATH",
            os.path.join("results", "mappo_actor_pruned_distilled_p25.pt"),
        ),
        help="Optional regular distilled baseline. Defaults to the existing root result.",
    )
    parser.add_argument(
        "--temporal-model-path",
        default=os.getenv(
            "TEMPORAL_DISTILLED_MODEL_PATH",
            os.path.join("temporal", "results", "mappo_actor_temporal_distilled_p25.pt"),
        ),
    )
    parser.add_argument("--temporal-repeat-scale", type=float, default=None)
    parser.add_argument("--temporal-max-repeat", type=int, default=None)
    parser.add_argument("--disable-safety", action="store_true")
    parser.add_argument("--safe-energy-min", type=float, default=0.12)
    parser.add_argument("--safe-queue-threshold", type=float, default=0.80)
    parser.add_argument("--safe-obs-change-threshold", type=float, default=0.35)
    parser.add_argument("--safe-channel-drop-threshold", type=float, default=0.35)
    parser.add_argument("--device", default=os.getenv("TEMPORAL_COMPARE_DEVICE", "cpu"))
    parser.add_argument(
        "--save-dir",
        default=os.getenv(
            "TEMPORAL_COMPARE_SAVE_DIR",
            os.path.join("temporal", "results", "compare_temporal"),
        ),
    )
    parser.add_argument("--load-min", type=float, default=float(os.getenv("TEMPORAL_LOAD_FACTOR_MIN", "0.70")))
    parser.add_argument("--load-max", type=float, default=float(os.getenv("TEMPORAL_LOAD_FACTOR_MAX", "1.30")))
    parser.add_argument("--load-points", type=int, default=int(os.getenv("TEMPORAL_NUM_LOAD_POINTS", "13")))
    parser.add_argument("--random-offload-prob", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.load_points <= 0:
        raise ValueError("--load-points must be positive.")
    if args.load_min > args.load_max:
        raise ValueError("--load-min must be less than or equal to --load-max.")
    if not 0.0 <= args.random_offload_prob <= 1.0:
        raise ValueError("--random-offload-prob must be in [0, 1].")

    cfg = EnvConfig()
    experiment_names = parse_experiment_names(args.experiments)
    policies = build_policies(cfg, experiment_names, args)

    print(f"Mode: {args.mode}")
    print(f"Episodes per policy/config: {args.episodes}")
    print(f"Policies: {', '.join(policy.name for policy in policies)}")
    print(f"Safety interrupt: {'off' if args.disable_safety else 'on'}")

    if args.mode == "fixed":
        run_fixed_comparison(cfg, policies, args)
    else:
        run_load_comparison(cfg, policies, args)


if __name__ == "__main__":
    main()
