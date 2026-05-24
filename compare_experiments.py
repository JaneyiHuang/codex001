from __future__ import annotations

import argparse
import os
from dataclasses import replace
from typing import Any, Callable, Dict, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

from config import EnvConfig
from experiments.common import (
    evaluate_policy,
    plot_bar_dashboard,
    plot_load_curves,
    print_summary_table,
    save_csv,
)
from experiments.greedy_policy import GreedyPolicy
from experiments.local_policy import LocalPolicy
from experiments.mappo_policy import MAPPOPolicy
from experiments.offload_policy import OffloadPolicy
from experiments.pruned_mappo_policy import PrunedMAPPOPolicy
from experiments.random_policy import RandomPolicy


PolicyFactory = Callable[[EnvConfig, argparse.Namespace], Any]
DEFAULT_EXPERIMENT_NAMES = [
    "mappo",
    "pruned_mappo",
    "distilled_mappo",
    "offload",
    "greedy",
    "local",
    "random",
]
DEFAULT_EXPERIMENT_STRING = ",".join(DEFAULT_EXPERIMENT_NAMES)
RATE_SWEEP_BASE_EXPERIMENT_NAMES = ["mappo"]
RATE_SWEEP_BASELINE_NAMES = ["offload", "greedy", "local", "random"]


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


def build_random(cfg: EnvConfig, args: argparse.Namespace) -> RandomPolicy:
    return RandomPolicy(seed=args.seed)


EXPERIMENTS: Dict[str, PolicyFactory] = {
    "mappo": build_mappo,
    "pruned_mappo": build_pruned_mappo,
    "distilled_mappo": build_distilled_mappo,
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


def parse_prune_rates(raw_rates: str) -> List[float]:
    rates = []
    for item in raw_rates.split(","):
        item = item.strip()
        if not item:
            continue
        rate = float(item)
        if rate > 1.0:
            rate = rate / 100.0
        if not 0.0 <= rate < 1.0:
            raise ValueError("Prune rates must be in [0, 1), or percentages in [0, 100).")
        rates.append(rate)
    return rates


def format_prune_suffix(rate: float) -> str:
    percent = rate * 100.0
    if abs(percent - round(percent)) < 1e-8:
        return f"p{int(round(percent))}"
    return "p" + f"{percent:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def format_rate_path(template: str, rate: float) -> str:
    suffix = format_prune_suffix(rate)
    percent = rate * 100.0
    return template.format(
        rate=rate,
        percent=percent,
        percent_int=int(round(percent)),
        suffix=suffix,
    )


def build_policies(
    cfg: EnvConfig,
    names: List[str],
    args: argparse.Namespace,
) -> List[Any]:
    return [EXPERIMENTS[name](cfg, args) for name in names]


def build_prune_rate_policies(
    cfg: EnvConfig,
    rates: List[float],
    args: argparse.Namespace,
) -> List[Any]:
    policies: List[Any] = []
    for rate in rates:
        suffix = format_prune_suffix(rate)
        pruned_path = format_rate_path(args.pruned_model_template, rate)
        distilled_path = format_rate_path(args.distilled_model_template, rate)
        policies.append(
            PrunedMAPPOPolicy(
                cfg,
                model_path=pruned_path,
                device=args.device,
                policy_name=f"pruned_{suffix}",
            )
        )
        policies.append(
            PrunedMAPPOPolicy(
                cfg,
                model_path=distilled_path,
                device=args.device,
                policy_name=f"distilled_{suffix}",
            )
        )
    return policies


def build_comparison_policies(
    cfg: EnvConfig,
    experiment_names: List[str],
    args: argparse.Namespace,
) -> List[Any]:
    prune_rates = parse_prune_rates(args.prune_rates)
    if not prune_rates:
        return build_policies(cfg, experiment_names, args)

    if args.experiments == DEFAULT_EXPERIMENT_STRING:
        policies = build_policies(cfg, RATE_SWEEP_BASE_EXPERIMENT_NAMES, args)
        policies.extend(build_prune_rate_policies(cfg, prune_rates, args))
        policies.extend(build_policies(cfg, RATE_SWEEP_BASELINE_NAMES, args))
        return policies

    policies = build_policies(cfg, experiment_names, args)
    policies.extend(build_prune_rate_policies(cfg, prune_rates, args))
    return policies


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
    print("\nDone. Load comparison results have been saved.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected MEC offloading comparison experiments."
    )
    parser.add_argument(
        "--experiments",
        default=os.getenv("COMPARE_EXPERIMENTS", DEFAULT_EXPERIMENT_STRING),
        help=(
            "Comma-separated experiment names. Valid names: "
            f"{', '.join(EXPERIMENTS)}."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["fixed", "loads"],
        default=os.getenv("COMPARE_MODE", "fixed"),
        help="fixed: one scenario; loads: scan task-size load factors.",
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
        help=(
            "Comma-separated pruning rates for a pruning-rate sweep, e.g. "
            "0.10,0.25,0.40,0.50 or 10,25,40,50. When this is set and "
            "--experiments is left at its default, the single pruned_mappo/"
            "distilled_mappo entries are replaced by rate-specific policies."
        ),
    )
    parser.add_argument(
        "--pruned-model-template",
        default=os.getenv(
            "PRUNED_MODEL_TEMPLATE",
            os.path.join("results", "mappo_actor_pruned_{suffix}.pt"),
        ),
        help=(
            "Path template for rate-specific pruned actors. Available fields: "
            "{suffix}, {rate}, {percent}, {percent_int}."
        ),
    )
    parser.add_argument(
        "--distilled-model-template",
        default=os.getenv(
            "DISTILLED_MODEL_TEMPLATE",
            os.path.join("results", "mappo_actor_pruned_distilled_{suffix}.pt"),
        ),
        help=(
            "Path template for rate-specific distilled actors. Available fields: "
            "{suffix}, {rate}, {percent}, {percent_int}."
        ),
    )
    parser.add_argument("--device", default=os.getenv("COMPARE_DEVICE", "cpu"))
    parser.add_argument(
        "--save-dir",
        default=os.getenv("COMPARE_SAVE_DIR", os.path.join("results", "compare_experiments")),
    )
    parser.add_argument("--load-min", type=float, default=float(os.getenv("LOAD_FACTOR_MIN", "0.70")))
    parser.add_argument("--load-max", type=float, default=float(os.getenv("LOAD_FACTOR_MAX", "1.30")))
    parser.add_argument("--load-points", type=int, default=int(os.getenv("NUM_LOAD_POINTS", "13")))
    return parser.parse_args()


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
    print(f"Mode: {args.mode}")
    print(f"Episodes per policy/config: {args.episodes}")
    policies = build_comparison_policies(cfg, experiment_names, args)
    print(f"Policies: {', '.join(policy.name for policy in policies)}")

    if args.mode == "fixed":
        run_fixed_comparison(cfg, policies, args)
    else:
        run_load_comparison(cfg, policies, args)


if __name__ == "__main__":
    main()
