from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EnvConfig
from models import Actor


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "compare_experiments"
MODEL_COLORS = {
    "original_mappo": "#2563EB",
    "pruned_mappo": "#F97316",
    "distilled_mappo": "#DC2626",
}


@dataclass
class ActorBenchmark:
    name: str
    actor: Actor
    checkpoint_path: Path
    hidden_dims: List[int]
    parameters: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare actor inference speed before pruning, after pruning, "
            "and after pruning plus distillation."
        )
    )
    parser.add_argument(
        "--original-model-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "mappo_checkpoint.pt",
        help="Path to the original trained MAPPO checkpoint.",
    )
    parser.add_argument(
        "--pruned-model-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "mappo_actor_pruned_p25.pt",
        help="Path to the pruned actor checkpoint.",
    )
    parser.add_argument(
        "--distilled-model-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "mappo_actor_pruned_distilled_p25.pt",
        help="Path to the distilled pruned actor checkpoint.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for CSV files and the comparison figure.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, such as cpu or cuda.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of per-agent observations inferred together. Defaults to EnvConfig.M.",
    )
    parser.add_argument("--warmup", type=int, default=300, help="Warmup actor calls per model.")
    parser.add_argument("--rounds", type=int, default=10, help="Independent timing rounds.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=2000,
        help="Actor calls measured in each timing round.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Torch CPU threads used by the benchmark.",
    )
    parser.add_argument(
        "--include-argmax",
        action="store_true",
        help="Measure actor forward plus greedy action selection instead of forward only.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_args = {
        "--warmup": args.warmup,
        "--rounds": args.rounds,
        "--repeats": args.repeats,
        "--num-threads": args.num_threads,
        "--dpi": args.dpi,
    }
    for name, value in positive_args.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "actor" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain an actor state dict: {path}")
    return checkpoint


def count_parameters(actor: Actor) -> int:
    return sum(parameter.numel() for parameter in actor.parameters())


def build_actor_benchmark(
    name: str,
    cfg: EnvConfig,
    checkpoint_path: Path,
    device: torch.device,
    hidden_dims: List[int] | None = None,
) -> ActorBenchmark:
    checkpoint = load_checkpoint(checkpoint_path, device)
    resolved_hidden_dims = hidden_dims or checkpoint.get("actor_hidden_dims")
    if resolved_hidden_dims is None:
        raise ValueError(
            f"Compressed checkpoint must contain actor_hidden_dims: {checkpoint_path}"
        )

    resolved_hidden_dims = [int(value) for value in resolved_hidden_dims]
    actor = Actor(
        obs_dim=cfg.obs_dim,
        n_actions=cfg.n_actions,
        hidden_dims=resolved_hidden_dims,
    ).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return ActorBenchmark(
        name=name,
        actor=actor,
        checkpoint_path=checkpoint_path,
        hidden_dims=resolved_hidden_dims,
        parameters=count_parameters(actor),
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def run_inference(
    actor: Actor,
    observations: torch.Tensor,
    include_argmax: bool,
) -> torch.Tensor:
    logits = actor(observations)
    if include_argmax:
        return torch.argmax(logits, dim=-1)
    return logits


def warm_up(
    benchmark: ActorBenchmark,
    observations: torch.Tensor,
    warmup: int,
    include_argmax: bool,
    device: torch.device,
) -> None:
    with torch.inference_mode():
        for _ in range(warmup):
            run_inference(benchmark.actor, observations, include_argmax)
    synchronize(device)


def measure_round(
    benchmark: ActorBenchmark,
    observations: torch.Tensor,
    repeats: int,
    include_argmax: bool,
    device: torch.device,
) -> float:
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeats):
            run_inference(benchmark.actor, observations, include_argmax)
    synchronize(device)
    elapsed_seconds = time.perf_counter() - start
    return elapsed_seconds * 1000.0 / repeats


def benchmark_actors(
    benchmarks: List[ActorBenchmark],
    observations: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, Any]]:
    for benchmark in benchmarks:
        print(f"Warming up {benchmark.name} ...")
        warm_up(benchmark, observations, args.warmup, args.include_argmax, device)

    records: List[Dict[str, Any]] = []
    for round_index in range(args.rounds):
        offset = round_index % len(benchmarks)
        ordered_benchmarks = benchmarks[offset:] + benchmarks[:offset]
        for order_index, benchmark in enumerate(ordered_benchmarks, start=1):
            latency_ms = measure_round(
                benchmark=benchmark,
                observations=observations,
                repeats=args.repeats,
                include_argmax=args.include_argmax,
                device=device,
            )
            records.append(
                {
                    "round": round_index + 1,
                    "measurement_order": order_index,
                    "model": benchmark.name,
                    "latency_ms": latency_ms,
                }
            )
        print(f"Completed timing round {round_index + 1}/{args.rounds}")
    return records


def summarize_results(
    benchmarks: List[ActorBenchmark],
    round_records: List[Dict[str, Any]],
    batch_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, Any]]:
    original_params = benchmarks[0].parameters
    latency_by_model = {
        benchmark.name: [
            float(record["latency_ms"])
            for record in round_records
            if record["model"] == benchmark.name
        ]
        for benchmark in benchmarks
    }
    original_latency_ms = statistics.mean(latency_by_model[benchmarks[0].name])

    summary: List[Dict[str, Any]] = []
    for benchmark in benchmarks:
        latencies = latency_by_model[benchmark.name]
        mean_latency_ms = statistics.mean(latencies)
        summary.append(
            {
                "model": benchmark.name,
                "checkpoint": str(benchmark.checkpoint_path),
                "hidden_dims": "x".join(str(value) for value in benchmark.hidden_dims),
                "parameters": benchmark.parameters,
                "parameter_memory_kb": benchmark.parameters * 4.0 / 1024.0,
                "parameter_reduction_percent": (
                    1.0 - benchmark.parameters / original_params
                )
                * 100.0,
                "latency_mean_ms": mean_latency_ms,
                "latency_std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "latency_p50_ms": float(np.percentile(latencies, 50)),
                "latency_p95_ms": float(np.percentile(latencies, 95)),
                "speedup_vs_original": original_latency_ms / mean_latency_ms,
                "observations_per_second": batch_size * 1000.0 / mean_latency_ms,
                "batch_size": batch_size,
                "device": str(device),
                "include_argmax": args.include_argmax,
                "warmup": args.warmup,
                "rounds": args.rounds,
                "repeats_per_round": args.repeats,
                "num_threads": args.num_threads,
            }
        )
    return summary


def save_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved CSV: {path}")


def plot_results(summary: List[Dict[str, Any]], save_path: Path, dpi: int) -> None:
    labels = [str(record["model"]) for record in summary]
    colors = [MODEL_COLORS[label] for label in labels]
    latency_means = np.array([record["latency_mean_ms"] for record in summary])
    latency_stds = np.array([record["latency_std_ms"] for record in summary])
    parameters = np.array([record["parameters"] for record in summary])
    x = np.arange(len(labels))

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    latency_axis, parameter_axis = axes

    latency_axis.bar(
        x,
        latency_means,
        yerr=latency_stds,
        capsize=4,
        color=colors,
        width=0.64,
    )
    latency_axis.set_title("Actor Inference Latency")
    latency_axis.set_ylabel("Mean Latency per Actor Call (ms)")
    latency_axis.set_xticks(x, labels, rotation=15, ha="right")
    latency_axis.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    latency_axis.set_axisbelow(True)
    for index, record in enumerate(summary):
        latency_axis.text(
            index,
            latency_means[index] + latency_stds[index],
            f"{latency_means[index]:.4f} ms\n{record['speedup_vs_original']:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    parameter_axis.bar(x, parameters, color=colors, width=0.64)
    parameter_axis.set_title("Actor Parameter Count")
    parameter_axis.set_ylabel("Parameters")
    parameter_axis.set_xticks(x, labels, rotation=15, ha="right")
    parameter_axis.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    parameter_axis.set_axisbelow(True)
    for index, parameter_count in enumerate(parameters):
        parameter_axis.text(
            index,
            parameter_count,
            f"{int(parameter_count):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.suptitle("MAPPO Actor Speed Before and After Pruning + Distillation")
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figure: {save_path}")


def print_summary(summary: List[Dict[str, Any]]) -> None:
    print("\n================ Inference Speed Comparison ================")
    print(
        f"{'model':<20} {'params':>10} {'latency(ms)':>14} "
        f"{'p95(ms)':>12} {'speedup':>10} {'obs/s':>14}"
    )
    for record in summary:
        print(
            f"{record['model']:<20} "
            f"{record['parameters']:>10,d} "
            f"{record['latency_mean_ms']:>14.6f} "
            f"{record['latency_p95_ms']:>12.6f} "
            f"{record['speedup_vs_original']:>9.3f}x "
            f"{record['observations_per_second']:>14.2f}"
        )
    print("============================================================\n")


def main() -> None:
    args = parse_args()
    validate_args(args)
    cfg = EnvConfig()
    batch_size = args.batch_size or cfg.M
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    observations = torch.randn(batch_size, cfg.obs_dim, dtype=torch.float32, device=device)

    benchmarks = [
        build_actor_benchmark(
            name="original_mappo",
            cfg=cfg,
            checkpoint_path=args.original_model_path,
            device=device,
            hidden_dims=[128, 128],
        ),
        build_actor_benchmark(
            name="pruned_mappo",
            cfg=cfg,
            checkpoint_path=args.pruned_model_path,
            device=device,
        ),
        build_actor_benchmark(
            name="distilled_mappo",
            cfg=cfg,
            checkpoint_path=args.distilled_model_path,
            device=device,
        ),
    ]

    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Measured operation: actor forward{' + argmax' if args.include_argmax else ''}")
    for benchmark in benchmarks:
        print(
            f"Loaded {benchmark.name}: hidden_dims={benchmark.hidden_dims}, "
            f"parameters={benchmark.parameters}"
        )

    round_records = benchmark_actors(benchmarks, observations, args, device)
    summary = summarize_results(benchmarks, round_records, batch_size, args, device)

    save_csv(args.save_dir / "inference_speed_rounds.csv", round_records)
    save_csv(args.save_dir / "inference_speed_comparison.csv", summary)
    plot_results(
        summary=summary,
        save_path=args.save_dir / "inference_speed_comparison.png",
        dpi=args.dpi,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
