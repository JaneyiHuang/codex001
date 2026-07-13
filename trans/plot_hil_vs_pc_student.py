#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot PC-student baseline vs Nano-HIL student comparison."""

from __future__ import print_function

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SYSTEM_METRICS = [
    ("average_reward", "Average reward"),
    ("average_delay", "Average delay"),
    ("average_drop_rate", "Drop rate"),
    ("average_offload_rate", "Offload rate"),
    ("target_action_agreement", "Teacher agreement"),
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, text):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def value(summary, key):
    raw = summary.get(key)
    return None if raw is None else float(raw)


def relative_diff(pc_value, hil_value):
    if pc_value is None or hil_value is None:
        return None
    if abs(pc_value) < 1e-12:
        return None
    return (hil_value - pc_value) / abs(pc_value)


def build_metric_rows(pc_summary, hil_summary, step_stats):
    rows = []
    for key, label in SYSTEM_METRICS:
        pc_value = value(pc_summary, key)
        hil_value = value(hil_summary, key)
        diff = None if pc_value is None or hil_value is None else hil_value - pc_value
        rows.append(
            {
                "metric": key,
                "label": label,
                "pc_student": pc_value,
                "hil_nano": hil_value,
                "diff_hil_minus_pc": diff,
                "relative_diff": relative_diff(pc_value, hil_value),
            }
        )

    rows.extend(
        [
            {
                "metric": "joint_action_match_rate_pc_vs_hil",
                "label": "Joint action match",
                "pc_student": 1.0,
                "hil_nano": step_stats["joint_action_match_rate"],
                "diff_hil_minus_pc": step_stats["joint_action_match_rate"] - 1.0,
                "relative_diff": step_stats["joint_action_match_rate"] - 1.0,
            },
            {
                "metric": "target_action_match_rate_pc_vs_hil",
                "label": "Target action match",
                "pc_student": 1.0,
                "hil_nano": step_stats["target_action_match_rate"],
                "diff_hil_minus_pc": step_stats["target_action_match_rate"] - 1.0,
                "relative_diff": step_stats["target_action_match_rate"] - 1.0,
            },
            {
                "metric": "delay_match_rate_pc_vs_hil",
                "label": "Delay match",
                "pc_student": 1.0,
                "hil_nano": step_stats["delay_match_rate"],
                "diff_hil_minus_pc": step_stats["delay_match_rate"] - 1.0,
                "relative_diff": step_stats["delay_match_rate"] - 1.0,
            },
        ]
    )
    return rows


def compute_step_stats(pc_steps, hil_steps):
    if len(pc_steps) != len(hil_steps):
        raise ValueError("Step CSV lengths differ: {} vs {}".format(len(pc_steps), len(hil_steps)))

    total = max(len(pc_steps), 1)
    same_joint = (pc_steps["joint_actions"].astype(str).values == hil_steps["joint_actions"].astype(str).values)
    same_delay = np.isclose(
        pc_steps["delay"].astype(float).values,
        hil_steps["delay"].astype(float).values,
        atol=1e-12,
        rtol=0.0,
    )
    same_target = (
        pc_steps["student_action_target"].astype(int).values
        == hil_steps["nano_action"].astype(int).values
    )
    return {
        "total_aligned_steps": int(total),
        "joint_action_match_rate": float(np.mean(same_joint)),
        "target_action_match_rate": float(np.mean(same_target)),
        "delay_match_rate": float(np.mean(same_delay)),
        "max_abs_delay_diff": float(
            np.max(
                np.abs(
                    pc_steps["delay"].astype(float).values
                    - hil_steps["delay"].astype(float).values
                )
            )
        ),
    }


def apply_grid(ax):
    ax.grid(True, axis="y", color="#D0D7DE", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def annotate_bars(ax, bars, fmt="{:.3g}"):
    ymin, ymax = ax.get_ylim()
    offset = max((ymax - ymin) * 0.02, 1e-6)
    for bar in bars:
        value = bar.get_height()
        y = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            fmt.format(value),
            ha="center",
            va=va,
            fontsize=8,
            color="#111827",
        )


def plot_system_metrics(pc_summary, hil_summary, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()
    labels = ["PC all-student", "Nano HIL"]
    colors = ["#2563EB", "#F97316"]

    for ax, (key, title) in zip(axes, SYSTEM_METRICS):
        values = [value(pc_summary, key), value(hil_summary, key)]
        bars = ax.bar(labels, values, color=colors, width=0.58)
        ax.set_title(title)
        apply_grid(ax)
        annotate_bars(ax, bars)
        if key in ("average_drop_rate", "average_offload_rate", "target_action_agreement"):
            ax.set_ylim(0.0, max(1.05, max(values) * 1.18))

    axes[-1].axis("off")
    axes[-1].text(
        0.0,
        0.75,
        "A: PC all-student baseline\n"
        "B: Nano target UE + PC student UEs\n"
        "System metrics are produced by env.step().",
        fontsize=11,
        va="top",
        color="#111827",
    )
    fig.suptitle("Pure-PC Student vs Nano-HIL Student", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_metric_deltas(metric_rows, out_path):
    rows = [row for row in metric_rows if row["metric"] in [m[0] for m in SYSTEM_METRICS]]
    labels = [row["label"] for row in rows]
    diffs = [row["diff_hil_minus_pc"] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = ["#64748B" if abs(diff or 0.0) < 1e-12 else "#DC2626" for diff in diffs]
    bars = ax.bar(labels, diffs, color=colors, width=0.6)
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_title("HIL minus PC Baseline")
    ax.set_ylabel("Absolute difference")
    ax.tick_params(axis="x", rotation=18)
    apply_grid(ax)
    annotate_bars(ax, bars, fmt="{:.4g}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_deployment_overhead(pc_summary, hil_summary, hil_steps, out_path):
    pc_infer = value(pc_summary, "average_pc_student_infer_ms")
    nano_infer = value(hil_summary, "average_nano_infer_ms")
    round_trip = value(hil_summary, "average_round_trip_ms")
    labels = ["PC student\nbatch infer", "Nano ONNX\npure infer", "PC-Nano\nround trip"]
    values = [pc_infer, nano_infer, round_trip]
    colors = ["#2563EB", "#F97316", "#7C3AED"]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title("Inference and Communication Overhead")
    ax.set_ylabel("Time (ms)")
    apply_grid(ax)
    annotate_bars(ax, bars, fmt="{:.2f}")

    if "round_trip_ms" in hil_steps.columns:
        vals = hil_steps["round_trip_ms"].astype(float).values
        text = "Round-trip p50={:.2f} ms, p95={:.2f} ms".format(
            float(np.percentile(vals, 50)),
            float(np.percentile(vals, 95)),
        )
        ax.text(0.02, 0.96, text, transform=ax.transAxes, va="top", fontsize=9, color="#334155")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_cumulative_delay(pc_steps, hil_steps, out_path):
    pc_delay = pc_steps["delay"].astype(float).values
    hil_delay = hil_steps["delay"].astype(float).values
    x = np.arange(1, len(pc_delay) + 1)
    pc_cum = np.cumsum(pc_delay) / x
    hil_cum = np.cumsum(hil_delay) / x

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, pc_cum, label="PC all-student", color="#2563EB", linewidth=2.0)
    ax.plot(x, hil_cum, label="Nano HIL", color="#F97316", linewidth=1.5, linestyle="--")
    ax.set_title("Cumulative Average Delay")
    ax.set_xlabel("Step")
    ax.set_ylabel("Delay")
    ax.legend()
    apply_grid(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_consistency(step_stats, hil_summary, out_path):
    labels = ["Joint actions", "Target action", "Delay", "Teacher agreement"]
    values = [
        step_stats["joint_action_match_rate"],
        step_stats["target_action_match_rate"],
        step_stats["delay_match_rate"],
        value(hil_summary, "target_action_agreement"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(labels, values, color=["#16A34A", "#16A34A", "#16A34A", "#0EA5E9"], width=0.58)
    ax.set_title("Action and Metric Consistency")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.06)
    apply_grid(ax)
    annotate_bars(ax, bars, fmt="{:.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_markdown_report(pc_summary, hil_summary, metric_rows, step_stats):
    lines = []
    lines.append("# PC Student vs Nano HIL Student Comparison")
    lines.append("")
    lines.append("| Metric | PC all-student | Nano HIL | HIL - PC | Relative diff |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in metric_rows:
        if row["metric"] not in [m[0] for m in SYSTEM_METRICS]:
            continue
        rel = row["relative_diff"]
        lines.append(
            "| {label} | {pc:.6g} | {hil:.6g} | {diff:.6g} | {rel} |".format(
                label=row["label"],
                pc=row["pc_student"],
                hil=row["hil_nano"],
                diff=row["diff_hil_minus_pc"],
                rel="" if rel is None else "{:.4%}".format(rel),
            )
        )

    lines.append("")
    lines.append("## Deployment Overhead")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append("| PC student batch infer ms | {:.6g} |".format(value(pc_summary, "average_pc_student_infer_ms")))
    lines.append("| Nano ONNX pure infer ms | {:.6g} |".format(value(hil_summary, "average_nano_infer_ms")))
    lines.append("| PC-Nano round-trip ms | {:.6g} |".format(value(hil_summary, "average_round_trip_ms")))

    lines.append("")
    lines.append("## Step-Level Consistency")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append("| Total aligned steps | {} |".format(step_stats["total_aligned_steps"]))
    lines.append("| Joint action match rate | {:.6g} |".format(step_stats["joint_action_match_rate"]))
    lines.append("| Target action match rate | {:.6g} |".format(step_stats["target_action_match_rate"]))
    lines.append("| Delay match rate | {:.6g} |".format(step_stats["delay_match_rate"]))
    lines.append("| Max absolute delay diff | {:.6g} |".format(step_stats["max_abs_delay_diff"]))
    lines.append("")
    lines.append(
        "Note: round-trip latency is deployment communication overhead and is not added "
        "to simulated MEC delay."
    )
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot PC student baseline vs Nano HIL student results.")
    parser.add_argument(
        "--pc_summary",
        default=os.path.join("results", "pc_student_baseline_20x300", "pc_student_baseline_summary.json"),
    )
    parser.add_argument(
        "--pc_steps",
        default=os.path.join("results", "pc_student_baseline_20x300", "pc_student_baseline_steps.csv"),
    )
    parser.add_argument(
        "--hil_summary",
        default=os.path.join(
            "results",
            "hil_single_ue_onnx_nano_student_others",
            "hil_single_ue_summary.json",
        ),
    )
    parser.add_argument(
        "--hil_steps",
        default=os.path.join(
            "results",
            "hil_single_ue_onnx_nano_student_others",
            "hil_single_ue_steps.csv",
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("results", "hil_vs_pc_student_comparison_20x300", "visuals"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pc_summary = load_json(args.pc_summary)
    hil_summary = load_json(args.hil_summary)
    pc_steps = pd.read_csv(args.pc_steps)
    hil_steps = pd.read_csv(args.hil_steps)

    step_stats = compute_step_stats(pc_steps, hil_steps)
    metric_rows = build_metric_rows(pc_summary, hil_summary, step_stats)
    ensure_dir(args.output_dir)

    table_path = os.path.join(args.output_dir, "hil_vs_pc_student_table.csv")
    report_path = os.path.join(args.output_dir, "hil_vs_pc_student_report.md")
    write_csv(
        table_path,
        metric_rows,
        ["metric", "label", "pc_student", "hil_nano", "diff_hil_minus_pc", "relative_diff"],
    )
    write_text(report_path, make_markdown_report(pc_summary, hil_summary, metric_rows, step_stats))

    paths = {
        "system_metrics": os.path.join(args.output_dir, "system_metrics_comparison.png"),
        "metric_deltas": os.path.join(args.output_dir, "metric_deltas_hil_minus_pc.png"),
        "deployment_overhead": os.path.join(args.output_dir, "deployment_overhead.png"),
        "cumulative_delay": os.path.join(args.output_dir, "cumulative_delay_overlay.png"),
        "consistency": os.path.join(args.output_dir, "step_consistency.png"),
        "table_csv": table_path,
        "report_md": report_path,
    }
    plot_system_metrics(pc_summary, hil_summary, paths["system_metrics"])
    plot_metric_deltas(metric_rows, paths["metric_deltas"])
    plot_deployment_overhead(pc_summary, hil_summary, hil_steps, paths["deployment_overhead"])
    plot_cumulative_delay(pc_steps, hil_steps, paths["cumulative_delay"])
    plot_consistency(step_stats, hil_summary, paths["consistency"])

    print(json.dumps({"output_dir": os.path.abspath(args.output_dir), "files": paths}, indent=2))


if __name__ == "__main__":
    main()
