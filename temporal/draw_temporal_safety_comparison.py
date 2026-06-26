from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from temporal.experiments.common import annotate_bars, set_padded_ylim


CASE_COLORS = {
    "Default safety": "#2563EB",
    "No safety": "#DC2626",
    "Relaxed safety": "#16A34A",
}


def read_summary(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_by_policy(rows: List[Dict[str, str]], policy: str) -> Dict[str, str]:
    for row in rows:
        if row["policy"] == policy:
            return row
    raise ValueError(f"Cannot find policy '{policy}' in summary.")


def build_case_record(case_name: str, csv_path: Path) -> Dict[str, Any]:
    rows = read_summary(csv_path)
    mappo = row_by_policy(rows, "mappo")
    temporal = row_by_policy(rows, "temporal_distilled_mappo")

    delay = float(temporal["delay_mean"])
    mappo_delay = float(mappo["delay_mean"])
    drop = float(temporal["drop_rate_mean"])
    mappo_drop = float(mappo["drop_rate_mean"])
    decision_count = float(temporal["decision_count_mean"])
    base_decisions = float(temporal["base_agent_decisions_mean"])
    inference_count = float(temporal["inference_call_count_mean"])
    base_inference = float(temporal["base_inference_calls_mean"])

    return {
        "case": case_name,
        "csv_path": str(csv_path),
        "delay_mean": delay,
        "delay_delta_vs_mappo": delay - mappo_delay,
        "drop_rate_mean": drop,
        "drop_delta_vs_mappo": drop - mappo_drop,
        "reward_mean": float(temporal["reward_mean"]),
        "decision_count_mean": decision_count,
        "decision_saved_percent": (1.0 - decision_count / max(base_decisions, 1.0)) * 100.0,
        "agent_temporal_compression_ratio_mean": float(
            temporal["agent_temporal_compression_ratio_mean"]
        ),
        "inference_call_count_mean": inference_count,
        "inference_saved_percent": (1.0 - inference_count / max(base_inference, 1.0)) * 100.0,
        "inference_temporal_compression_ratio_mean": float(
            temporal["inference_temporal_compression_ratio_mean"]
        ),
        "safety_interrupt_count_mean": float(temporal["safety_interrupt_count_mean"]),
        "predicted_repeat_mean": float(temporal["predicted_repeat_mean"]),
    }


def save_summary_csv(records: List[Dict[str, Any]], out_dir: Path) -> Path:
    csv_path = out_dir / "temporal_safety_summary.csv"
    fieldnames = [
        "case",
        "delay_mean",
        "delay_delta_vs_mappo",
        "drop_rate_mean",
        "drop_delta_vs_mappo",
        "reward_mean",
        "decision_count_mean",
        "decision_saved_percent",
        "agent_temporal_compression_ratio_mean",
        "inference_call_count_mean",
        "inference_saved_percent",
        "inference_temporal_compression_ratio_mean",
        "safety_interrupt_count_mean",
        "predicted_repeat_mean",
        "csv_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return csv_path


def save_summary_markdown(records: List[Dict[str, Any]], out_dir: Path) -> Path:
    md_path = out_dir / "temporal_safety_summary.md"
    headers = [
        "Case",
        "Delay",
        "Delta Delay",
        "Drop",
        "Delta Drop",
        "Decision Saving",
        "Agent CR",
        "Inference Saving",
        "Inference CR",
        "Safety Interrupts",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record["case"]),
                    f"{record['delay_mean']:.3f}",
                    f"{record['delay_delta_vs_mappo']:+.3f}",
                    f"{record['drop_rate_mean']:.4f}",
                    f"{record['drop_delta_vs_mappo']:+.4f}",
                    f"{record['decision_saved_percent']:.2f}%",
                    f"{record['agent_temporal_compression_ratio_mean']:.3f}x",
                    f"{record['inference_saved_percent']:.2f}%",
                    f"{record['inference_temporal_compression_ratio_mean']:.3f}x",
                    f"{record['safety_interrupt_count_mean']:.2f}",
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def plot_performance_gaps(records: List[Dict[str, Any]], out_dir: Path) -> Path:
    labels = [record["case"] for record in records]
    colors = [CASE_COLORS[label] for label in labels]
    x = np.arange(len(labels))

    panels = [
        ("Delay Gap vs MAPPO", "Delta delay", [r["delay_delta_vs_mappo"] for r in records], "{:+.2f}"),
        ("Drop-Rate Gap vs MAPPO", "Delta drop rate", [r["drop_delta_vs_mappo"] for r in records], "{:+.4f}"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, (title, ylabel, values, value_fmt) in zip(axes, panels):
        bars = ax.bar(x, values, color=colors, edgecolor="#334155", linewidth=0.7, width=0.62)
        ax.axhline(0.0, color="#475569", linewidth=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        set_padded_ylim(ax, values, include_zero=True)
        annotate_bars(ax, bars, values, value_fmt)
        ax.grid(True, axis="y", color="#D5DBE5", linewidth=0.6, alpha=0.82)
        ax.set_axisbelow(True)

    fig.suptitle("Temporal Safety Modes: Performance Cost", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "temporal_safety_performance_gaps.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_compression_savings(records: List[Dict[str, Any]], out_dir: Path) -> Path:
    labels = [record["case"] for record in records]
    colors = [CASE_COLORS[label] for label in labels]
    x = np.arange(len(labels))

    panels = [
        (
            "Per-Agent Decision Saving",
            "Saving (%)",
            [r["decision_saved_percent"] for r in records],
            "{:.1f}%",
        ),
        (
            "Slot-Level Inference Saving",
            "Saving (%)",
            [r["inference_saved_percent"] for r in records],
            "{:.1f}%",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, (title, ylabel, values, value_fmt) in zip(axes, panels):
        bars = ax.bar(x, values, color=colors, edgecolor="#334155", linewidth=0.7, width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        set_padded_ylim(ax, values, include_zero=True)
        annotate_bars(ax, bars, values, value_fmt)
        ax.grid(True, axis="y", color="#D5DBE5", linewidth=0.6, alpha=0.82)
        ax.set_axisbelow(True)

    fig.suptitle("Temporal Safety Modes: Compression Benefit", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "temporal_safety_compression_savings.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_tradeoff(records: List[Dict[str, Any]], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    max_interrupts = max(r["safety_interrupt_count_mean"] for r in records) or 1.0
    for record in records:
        label = record["case"]
        x = record["decision_saved_percent"]
        y = record["delay_delta_vs_mappo"]
        size = 120.0 + 360.0 * record["safety_interrupt_count_mean"] / max_interrupts
        ax.scatter(
            x,
            y,
            s=size,
            color=CASE_COLORS[label],
            edgecolor="#111827",
            linewidth=0.7,
            alpha=0.88,
            label=label,
        )
        ax.annotate(
            f"{label}\n{x:.1f}%, {y:+.2f}",
            xy=(x, y),
            xytext=(8, 7),
            textcoords="offset points",
            fontsize=9,
        )

    ax.axhline(0.0, color="#475569", linewidth=0.85)
    ax.set_xlabel("Per-agent decision saving (%)")
    ax.set_ylabel("Delay gap vs MAPPO")
    ax.set_title("Performance-Compression Trade-off")
    ax.grid(True, color="#D5DBE5", linewidth=0.6, alpha=0.82)
    ax.set_axisbelow(True)
    ax.legend(title="Safety mode", loc="best")

    path = out_dir / "temporal_safety_tradeoff.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw clearer cross-safety figures for temporal distillation results."
    )
    parser.add_argument(
        "--default-csv",
        type=Path,
        default=Path("temporal/results/compare_temporal/comparison_summary.csv"),
    )
    parser.add_argument(
        "--no-safety-csv",
        type=Path,
        default=Path("temporal/results/compare_temporal_no_safety/comparison_summary.csv"),
    )
    parser.add_argument(
        "--relaxed-csv",
        type=Path,
        default=Path("temporal/results/compare_temporal_relaxed_safety/comparison_summary.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("temporal/results/safety_comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = [
        build_case_record("Default safety", args.default_csv),
        build_case_record("No safety", args.no_safety_csv),
        build_case_record("Relaxed safety", args.relaxed_csv),
    ]

    saved = [
        save_summary_csv(records, args.out_dir),
        save_summary_markdown(records, args.out_dir),
        plot_performance_gaps(records, args.out_dir),
        plot_compression_savings(records, args.out_dir),
        plot_tradeoff(records, args.out_dir),
    ]

    print(f"Saved {len(saved)} files to {args.out_dir}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
