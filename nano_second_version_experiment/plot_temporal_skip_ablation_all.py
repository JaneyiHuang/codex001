#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot four-way temporal skip ablation results."""

from __future__ import print_function

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("no_skip", "No-skip"),
    ("safety", "Safety-aware"),
    ("relaxed", "Relaxed safety"),
    ("no_safety", "No-safety"),
]

SYSTEM_YLIMS = {
    "average_delay": (195.0, 200.0),
    "average_reward": (-1.0, -0.97),
    "average_drop_rate": (0.20, 0.23),
    "average_offload_rate": (0.74, 0.80),
}

SYSTEM_YTICKS = {
    "average_delay": np.arange(195.0, 200.1, 1.0),
    "average_reward": np.arange(-1.00, -0.969, 0.01),
    "average_drop_rate": np.arange(0.20, 0.231, 0.01),
    "average_offload_rate": np.arange(0.74, 0.801, 0.02),
}

SYSTEM_FORMATS = {
    "average_delay": "{:.2f}",
    "average_reward": "{:.4f}",
    "average_drop_rate": "{:.4f}",
    "average_offload_rate": "{:.4f}",
}

CALL_REDUCTION_YLIM = (0.0, 0.8)
OVERHEAD_YLIM = (0.0, 55.0)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def value(summary, key, default=None):
    raw = summary.get(key, default)
    return None if raw is None else float(raw)


def build_rows(no_skip, safety, relaxed, no_safety):
    summaries = {
        "no_skip": no_skip,
        "safety": safety,
        "relaxed": relaxed,
        "no_safety": no_safety,
    }
    rows = []
    for key, label in METHODS:
        s = summaries[key]
        total_steps = value(s, "total_steps")
        if key == "no_skip":
            nano_calls = total_steps
            nano_reduction = 0.0
            pc_repeat_calls = None
            pc_repeat_reduction = None
            safety_interrupts = None
            pc_safety_interrupts = None
            rtt_per_step = value(s, "average_round_trip_ms")
            infer_per_step = value(s, "average_nano_infer_ms")
        else:
            nano_calls = value(s, "nano_model_call_count")
            nano_reduction = value(s, "nano_call_reduction_rate")
            pc_repeat_calls = value(s, "pc_repeat_model_call_count")
            pc_repeat_reduction = value(s, "pc_repeat_call_reduction_rate")
            safety_interrupts = value(s, "nano_safety_interrupt_count")
            pc_safety_interrupts = value(s, "pc_repeat_safety_interrupt_count")
            rtt_per_step = value(s, "average_round_trip_ms_per_env_step")
            infer_per_step = value(s, "average_nano_infer_ms_per_env_step")

        rows.append(
            {
                "method": label,
                "average_delay": value(s, "average_delay"),
                "average_reward": value(s, "average_reward"),
                "average_drop_rate": value(s, "average_drop_rate"),
                "average_offload_rate": value(s, "average_offload_rate"),
                "target_action_agreement": value(s, "target_action_agreement"),
                "nano_calls": nano_calls,
                "nano_call_reduction_rate": nano_reduction,
                "pc_repeat_calls": pc_repeat_calls,
                "pc_repeat_call_reduction_rate": pc_repeat_reduction,
                "nano_safety_interrupt_count": safety_interrupts,
                "pc_repeat_safety_interrupt_count": pc_safety_interrupts,
                "round_trip_ms_per_step": rtt_per_step,
                "nano_infer_ms_per_step": infer_per_step,
            }
        )
    return rows


def write_csv(path, rows):
    ensure_dir(os.path.dirname(path))
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    ensure_dir(os.path.dirname(path))
    lines = []
    lines.append("# Temporal Skip Ablation")
    lines.append("")
    lines.append(
        "| Method | Delay | Reward | Drop | Offload | Nano call reduction | RTT/step | Nano infer/step |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {method} | {delay:.6g} | {reward:.6g} | {drop:.6g} | {offload:.6g} | {red:.2%} | {rtt:.6g} | {infer:.6g} |".format(
                method=row["method"],
                delay=row["average_delay"],
                reward=row["average_reward"],
                drop=row["average_drop_rate"],
                offload=row["average_offload_rate"],
                red=row["nano_call_reduction_rate"],
                rtt=row["round_trip_ms_per_step"],
                infer=row["nano_infer_ms_per_step"],
            )
        )
    lines.append("")
    lines.append("Safety conditions:")
    lines.append("")
    lines.append("- No-skip: temporal repeat is disabled; every step calls Nano.")
    lines.append("- Safety-aware: energy<=0.12, queue>=0.80, obs-change>=0.35, channel-drop>=0.35 interrupt repeat.")
    lines.append("- Relaxed safety: energy<=0.08, queue>=0.90, obs-change>=0.50, channel-drop>=0.50 interrupt repeat.")
    lines.append("- No-safety: safety interrupts disabled.")
    lines.append("")
    lines.append("RTT and Nano inference overhead are reported separately and are not added to simulated MEC delay.")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def apply_grid(ax):
    ax.grid(True, axis="y", color="#D0D7DE", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def annotate(ax, bars, fmt="{:.3g}", rotation=0, tight=True):
    ymin, ymax = ax.get_ylim()
    offset_ratio = 0.006 if tight else 0.02
    offset = max((ymax - ymin) * offset_ratio, 1e-6)
    for bar in bars:
        top = bar.get_y() + bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            top + offset,
            fmt.format(top),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=rotation,
            color="#111827",
        )


def bar_plot(rows, key, title, ylabel, path, percent=False):
    labels = [row["method"] for row in rows]
    vals = [row[key] for row in rows]
    colors = ["#2563EB", "#16A34A", "#F97316", "#DC2626"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=colors, width=0.58)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    apply_grid(ax)
    ax.tick_params(axis="x", rotation=12)
    if percent:
        ax.set_ylim(*CALL_REDUCTION_YLIM)
    annotate(ax, bars, fmt="{:.1%}" if percent else "{:.4g}")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def multi_metric_plot(rows, path):
    labels = [row["method"] for row in rows]
    colors = ["#2563EB", "#16A34A", "#F97316", "#DC2626"]
    specs = [
        ("average_delay", "Average delay"),
        ("average_reward", "Average reward"),
        ("average_drop_rate", "Drop rate"),
        ("average_offload_rate", "Offload rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    for ax, (key, title) in zip(axes.flatten(), specs):
        vals = [row[key] for row in rows]
        if key in SYSTEM_YLIMS:
            y_min, y_max = SYSTEM_YLIMS[key]
        else:
            y_min = min(vals)
            y_max = max(vals)
            pad = max((y_max - y_min) * 0.18, abs(y_max) * 0.02, 1e-6)
            y_min -= pad
            y_max += pad

        # For truncated axes, draw bars from the visible lower bound. This avoids
        # the "flattened" look that appears when bars start at zero outside view.
        heights = [v - y_min for v in vals]
        bars = ax.bar(labels, heights, bottom=y_min, color=colors, width=0.58)
        ax.set_title(title)
        apply_grid(ax)
        ax.tick_params(axis="x", rotation=12)
        ax.set_ylim(y_min, y_max)
        if key in SYSTEM_YTICKS:
            ax.set_yticks(SYSTEM_YTICKS[key])
        annotate(ax, bars, fmt=SYSTEM_FORMATS.get(key, "{:.3g}"))

    fig.suptitle("Temporal Skip Ablation: System Metrics", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def overhead_plot(rows, path):
    labels = [row["method"] for row in rows]
    rtt = [row["round_trip_ms_per_step"] for row in rows]
    infer = [row["nano_infer_ms_per_step"] for row in rows]
    x = np.arange(len(rows))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width / 2, rtt, width, label="RTT/step", color="#7C3AED")
    b2 = ax.bar(x + width / 2, infer, width, label="Nano infer/step", color="#F97316")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_title("Per-Step Deployment Overhead")
    ax.set_ylabel("Time (ms)")
    ax.legend()
    apply_grid(ax)
    ax.set_ylim(*OVERHEAD_YLIM)
    annotate(ax, b1, fmt="{:.2f}")
    annotate(ax, b2, fmt="{:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot four-way temporal skip ablation.")
    parser.add_argument(
        "--no_skip_summary",
        default=os.path.join("results", "hil_single_ue_onnx_nano_student_others", "hil_single_ue_summary.json"),
    )
    parser.add_argument(
        "--safety_summary",
        default=os.path.join("nano_second_version_experiment", "results", "mixed_skip_20x300", "mixed_temporal_skip_summary.json"),
    )
    parser.add_argument(
        "--relaxed_summary",
        default=os.path.join("nano_second_version_experiment", "results", "mixed_skip_relaxed_safety_20x300", "mixed_temporal_skip_summary.json"),
    )
    parser.add_argument(
        "--no_safety_summary",
        default=os.path.join("nano_second_version_experiment", "results", "mixed_skip_no_safety_20x300", "mixed_temporal_skip_summary.json"),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("nano_second_version_experiment", "results", "temporal_skip_ablation_all", "visuals"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    no_skip = load_json(args.no_skip_summary)
    safety = load_json(args.safety_summary)
    relaxed = load_json(args.relaxed_summary)
    no_safety = load_json(args.no_safety_summary)
    rows = build_rows(no_skip, safety, relaxed, no_safety)
    ensure_dir(args.output_dir)

    table_path = os.path.join(args.output_dir, "temporal_skip_ablation_all.csv")
    report_path = os.path.join(args.output_dir, "temporal_skip_ablation_all.md")
    system_path = os.path.join(args.output_dir, "temporal_skip_system_metrics_all.png")
    reduction_path = os.path.join(args.output_dir, "nano_call_reduction_all.png")
    overhead_path = os.path.join(args.output_dir, "deployment_overhead_all.png")

    write_csv(table_path, rows)
    write_markdown(report_path, rows)
    multi_metric_plot(rows, system_path)
    bar_plot(rows, "nano_call_reduction_rate", "Nano Call Reduction", "Reduction rate", reduction_path, percent=True)
    overhead_plot(rows, overhead_path)

    print(
        json.dumps(
            {
                "output_dir": os.path.abspath(args.output_dir),
                "files": {
                    "table_csv": table_path,
                    "report_md": report_path,
                    "system_metrics": system_path,
                    "nano_call_reduction": reduction_path,
                    "deployment_overhead": overhead_path,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
