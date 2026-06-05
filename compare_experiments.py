from __future__ import annotations

import argparse
import os
import re
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from QMIX.policy import QMIXPolicy


PolicyFactory = Callable[[EnvConfig, argparse.Namespace], Any]
DEFAULT_EXPERIMENT_NAMES = [
    "mappo",
    "qmix",
]
DEFAULT_EXPERIMENT_STRING = ",".join(DEFAULT_EXPERIMENT_NAMES)
RATE_SWEEP_BASE_EXPERIMENT_NAMES = ["mappo"]
RATE_SWEEP_BASELINE_NAMES = ["offload", "greedy", "local", "random"]


def build_mappo(cfg: EnvConfig, args: argparse.Namespace) -> MAPPOPolicy:
    return MAPPOPolicy(cfg, model_path=args.model_path, device=args.device)


def build_qmix(cfg: EnvConfig, args: argparse.Namespace) -> QMIXPolicy:
    return QMIXPolicy(cfg, model_path=args.qmix_model_path, device=args.device)


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
    return RandomPolicy(seed=args.seed, offload_probability=args.random_offload_prob)


EXPERIMENTS: Dict[str, PolicyFactory] = {
    "mappo": build_mappo,
    "qmix": build_qmix,
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
        normalized = item[1:].replace("p", ".") if item.lower().startswith("p") else item
        rate = float(normalized)
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


def try_parse_rate_item(raw_item: str) -> Optional[float]:
    item = raw_item.strip().lower()
    if not item:
        return None
    if item.endswith(".pt"):
        return None
    if any(separator in item for separator in ("/", "\\")):
        return None

    try:
        normalized = item[1:].replace("p", ".") if item.startswith("p") else item
        rate = float(normalized)
    except ValueError:
        return None

    if rate > 1.0:
        rate = rate / 100.0
    if not 0.0 <= rate < 1.0:
        raise ValueError("Prune rates must be in [0, 1), or percentages in [0, 100).")
    return rate


def sanitize_policy_label(label: str, fallback: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", label.strip()).strip("_").lower()
    return normalized or fallback


def normalize_compressed_policy_name(prefix: str, label: str) -> str:
    label = sanitize_policy_label(label, "model")
    if label.startswith(f"{prefix}_"):
        return label
    return f"{prefix}_{label}"


def infer_model_label(model_path: str, prefix: str, index: int) -> str:
    stem = os.path.splitext(os.path.basename(model_path))[0].lower()
    match = re.search(r"(?:^|_)(p\d+(?:p\d+)?)(?:_|$)", stem)
    if match:
        return match.group(1)

    known_prefixes = [
        "mappo_actor_pruned_distilled",
        "mappo_actor_pruned",
        f"{prefix}_mappo",
        prefix,
    ]
    for known_prefix in known_prefixes:
        if stem == known_prefix:
            return "default"
        if stem.startswith(f"{known_prefix}_"):
            return sanitize_policy_label(stem[len(known_prefix) + 1 :], f"model{index}")

    return sanitize_policy_label(stem, f"model{index}")


def parse_model_specs(raw_specs: str, template: str, prefix: str) -> List[Tuple[str, str]]:
    specs: List[Tuple[str, str]] = []
    seen_names = set()

    for index, item in enumerate(raw_specs.split(","), start=1):
        item = item.strip()
        if not item:
            continue

        label = ""
        value = item
        if "=" in item:
            label, value = item.split("=", 1)
            label = label.strip()
            value = value.strip()
            if not label or not value:
                raise ValueError(
                    f"Invalid {prefix} model spec '{item}'. Use rate, path, or label=path."
                )

        rate = try_parse_rate_item(value)
        if rate is not None:
            suffix = format_prune_suffix(rate)
            model_path = format_rate_path(template, rate)
            policy_name = normalize_compressed_policy_name(prefix, label or suffix)
        else:
            model_path = value
            policy_name = normalize_compressed_policy_name(
                prefix,
                label or infer_model_label(model_path, prefix, index),
            )

        if policy_name in seen_names:
            raise ValueError(f"Duplicate policy name generated from {prefix} specs: {policy_name}")
        seen_names.add(policy_name)
        specs.append((policy_name, model_path))

    return specs


def build_policies(
    cfg: EnvConfig,
    names: List[str],
    args: argparse.Namespace,
) -> List[Any]:
    return [EXPERIMENTS[name](cfg, args) for name in names]


def build_compressed_model_policies(
    cfg: EnvConfig,
    specs: List[Tuple[str, str]],
    args: argparse.Namespace,
) -> List[Any]:
    return [
        PrunedMAPPOPolicy(
            cfg,
            model_path=model_path,
            device=args.device,
            policy_name=policy_name,
        )
        for policy_name, model_path in specs
    ]


def build_rate_policies(
    cfg: EnvConfig,
    rates: List[float],
    args: argparse.Namespace,
    *,
    prefix: str,
    template: str,
) -> List[Any]:
    policies: List[Any] = []
    for rate in rates:
        suffix = format_prune_suffix(rate)
        policies.append(
            PrunedMAPPOPolicy(
                cfg,
                model_path=format_rate_path(template, rate),
                device=args.device,
                policy_name=f"{prefix}_{suffix}",
            )
        )
    return policies


def build_compressed_mappo_policies(cfg: EnvConfig, args: argparse.Namespace) -> List[Any]:
    if args.pruned_models or args.distilled_models:
        specs = []
        specs.extend(parse_model_specs(args.pruned_models, args.pruned_model_template, "pruned"))
        specs.extend(
            parse_model_specs(args.distilled_models, args.distilled_model_template, "distilled")
        )
        return build_compressed_model_policies(cfg, specs, args)

    prune_rates = parse_prune_rates(args.prune_rates)
    if not prune_rates:
        return []

    policies: List[Any] = []
    policies.extend(
        build_rate_policies(
            cfg,
            prune_rates,
            args,
            prefix="pruned",
            template=args.pruned_model_template,
        )
    )
    policies.extend(
        build_rate_policies(
            cfg,
            prune_rates,
            args,
            prefix="distilled",
            template=args.distilled_model_template,
        )
    )
    return policies


def build_comparison_policies(
    cfg: EnvConfig,
    experiment_names: List[str],
    args: argparse.Namespace,
) -> List[Any]:
    compressed_policies = build_compressed_mappo_policies(cfg, args)
    if not compressed_policies:
        return build_policies(cfg, experiment_names, args)

    if args.experiments == DEFAULT_EXPERIMENT_STRING:
        policies = build_policies(cfg, RATE_SWEEP_BASE_EXPERIMENT_NAMES, args)
        policies.extend(compressed_policies)
        policies.extend(build_policies(cfg, RATE_SWEEP_BASELINE_NAMES, args))
        return policies

    base_names = [
        name
        for name in experiment_names
        if name not in {"pruned_mappo", "distilled_mappo"}
    ]
    policies = build_policies(cfg, base_names, args)
    policies.extend(compressed_policies)
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
        "--qmix-model-path",
        default=os.getenv("QMIX_MODEL_PATH", os.path.join("QMIX", "results", "qmix_checkpoint.pt")),
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
            "Legacy convenience option for a pruning-rate sweep, e.g. "
            "0.10,0.25,0.40,0.50 or 10,25,40,50. This includes both pruned "
            "and distilled actors for every rate unless --pruned-models or "
            "--distilled-models is set."
        ),
    )
    parser.add_argument(
        "--pruned-models",
        default=os.getenv("PRUNED_MAPPO_MODELS", ""),
        help=(
            "Comma-separated pruned actors to compare. Each item can be a rate "
            "(10, p10, 0.10), a model path, or label=path. Rate items use "
            "--pruned-model-template."
        ),
    )
    parser.add_argument(
        "--distilled-models",
        default=os.getenv("DISTILLED_MAPPO_MODELS", ""),
        help=(
            "Comma-separated distilled actors to compare. Each item can be a rate "
            "(10, p10, 0.10), a model path, or label=path. Rate items use "
            "--distilled-model-template."
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
    parser.add_argument(
        "--random-offload-prob",
        type=float,
        default=float(os.getenv("RANDOM_OFFLOAD_PROB", "0.50")),
        help=(
            "Probability that the random baseline selects offloading action 1. "
            "The standard unbiased random baseline is 0.50."
        ),
    )
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
