#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot first-version no-skip HIL vs second-version mixed temporal-skip HIL."""

from __future__ import print_function

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SYSTEM_METRICS = [
    ("average_reward", "Average reward"),
    ("average_delay", "Average delay"),
    ("average_drop_rate", "Drop rate"),
    ("average_offload_rate", "Offload rate"),
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def value(summary, key):
    raw = summary.get(key)
    return None if raw is None else float(raw)


def write_csv(path, rows):
    ensure_dir(os.path.dirname(path))
    fieldnames = ["metric", "no_skip", "mixed_skip", "diff_skip_minus_no_skip", "relative_diff"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, no_skip, skip):
    ensure_dir(os.path.dirname(path))
    lines = []
    lines.append("# No-Skip HIL vs Mixed Temporal-Skip HIL")
    lines.append("")
    lines.append("| Metric | No skip | Mixed skip | Skip - No skip | Relative diff |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in rows:
        rel = row["relative_diff"]
        lines.append(
            "| {metric} | {no_skip:.6g} | {mixed_skip:.6g} | {diff:.6g} | {rel} |".format(
                metric=row["metric"],
                no_skip=row["no_skip"],
                mixed_skip=row["mixed_skip"],
                diff=row["diff_skip_minus_no_skip"],
                rel="" if rel is None else "{:.4%}".format(rel),
            )
        )
    lines.append("")
    lines.append("## Temporal Skip Calls")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append("| Total env steps | {} |".format(skip.get("total_steps")))
    lines.append("| Nano model calls | {} |".format(skip.get("nano_model_call_count")))
    lines.append("| Nano call reduction rate | {:.4%} |".format(value(skip, "nano_call_reduction_rate")))
    lines.append("| PC repeat model calls | {} |".format(skip.get("pc_repeat_model_call_count")))
    lines.append("| PC repeat call reduction rate | {:.4%} |".format(value(skip, "pc_repeat_call_reduction_rate")))
    lines.append("| Nano safety interrupts | {} |".format(skip.get("nano_safety_interrupt_count")))
    lines.append("| PC repeat safety interrupts | {} |".format(skip.get("pc_repeat_safety_interrupt_count")))
    lines.append("")
    lines.append("## Deployment Overhead")
    lines.append("")
    lines.append("| Metric | No skip | Mixed skip |")
    lines.append("|---|---:|---:|")
    lines.append(
        "| Nano infer ms per env step | {:.6g} | {:.6g} |".format(
            value(no_skip, "average_nano_infer_ms"),
            value(skip, "average_nano_infer_ms_per_env_step"),
        )
    )
    lines.append(
        "| Round-trip ms per env step | {:.6g} | {:.6g} |".format(
            value(no_skip, "average_round_trip_ms"),
            value(skip, "average_round_trip_ms_per_env_step"),
        )
    )
    lines.append("")
    lines.append("Note: deployment overhead is reported separately and is not added to simulated MEC delay.")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_rows(no_skip, skip):
    rows = []
    for key, label in SYSTEM_METRICS:
        a = value(no_skip, key)
        b = value(skip, key)
        diff = None if a is None or b is None else b - a
        rel = None
        if diff is not None and abs(a) > 1e-12:
            rel = diff / abs(a)
        rows.append(
            {
                "metric": label,
                "no_skip": a,
                "mixed_skip": b,
                "diff_skip_minus_no_skip": diff,
                "relative_diff": rel,
            }
        )
    return rows


def apply_grid(ax):
    ax.grid(True, axis="y", color="#D0D7DE", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def annotate(ax, bars, fmt="{:.3g}"):
    ymin, ymax = ax.get_ylim()
    offset = max((ymax - ymin) * 0.02, 1e-6)
    for bar in bars:
        v = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + offset if v >= 0 else v - offset,
            fmt.format(v),
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8,
            color="#111827",
        )


def plot_system(no_skip, skip, path):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    axes = axes.flatten()
    labels = ["No skip", "Mixed skip"]
    colors = ["#2563EB", "#F97316"]
    for ax, (key, title) in zip(axes, SYSTEM_METRICS):
        vals = [value(no_skip, key), value(skip, key)]
        bars = ax.bar(labels, vals, color=colors, width=0.58)
        ax.set_title(title)
        apply_grid(ax)
        annotate(ax, bars)
        if key in ("average_drop_rate", "average_offload_rate"):
            ax.set_ylim(0.0, max(1.05, max(vals) * 1.18))
    fig.suptitle("No-Skip vs Mixed Temporal-Skip HIL", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_calls(skip, path):
    total = value(skip, "total_steps")
    labels = ["Nano calls", "Nano skips", "PC repeat calls", "PC repeat skips"]
    vals = [
        value(skip, "nano_model_call_count"),
        value(skip, "nano_skip_count"),
        value(skip, "pc_repeat_model_call_count"),
        value(skip, "pc_repeat_skip_count"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=["#F97316", "#FDBA74", "#2563EB", "#93C5FD"], width=0.58)
    ax.set_title("Temporal Skip Call Counts")
    ax.set_ylabel("Steps")
    if total is not None:
        ax.axhline(total, color="#111827", linewidth=1.0, linestyle="--", label="Total env steps")
        ax.legend()
    apply_grid(ax)
    annotate(ax, bars, fmt="{:.0f}")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_overhead(no_skip, skip, path):
    labels = [
        "No-skip\nNano infer/step",
        "Mixed-skip\nNano infer/step",
        "No-skip\nRTT/step",
        "Mixed-skip\nRTT/step",
    ]
    vals = [
        value(no_skip, "average_nano_infer_ms"),
        value(skip, "average_nano_infer_ms_per_env_step"),
        value(no_skip, "average_round_trip_ms"),
        value(skip, "average_round_trip_ms_per_env_step"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=["#F97316", "#FDBA74", "#7C3AED", "#C4B5FD"], width=0.58)
    ax.set_title("Per-Environment-Step Deployment Overhead")
    ax.set_ylabel("Time (ms)")
    apply_grid(ax)
    annotate(ax, bars, fmt="{:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot no-skip vs mixed temporal-skip HIL results.")
    parser.add_argument(
        "--no_skip_summary",
        default=os.path.join(
            "results",
            "hil_single_ue_onnx_nano_student_others",
            "hil_single_ue_summary.json",
        ),
    )
    parser.add_argument(
        "--skip_summary",
        default=os.path.join(
            "nano_second_version_experiment",
            "results",
            "mixed_skip_20x300",
            "mixed_temporal_skip_summary.json",
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            "nano_second_version_experiment",
            "results",
            "mixed_skip_20x300",
            "visuals",
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    no_skip = load_json(args.no_skip_summary)
    skip = load_json(args.skip_summary)
    rows = build_rows(no_skip, skip)
    ensure_dir(args.output_dir)

    table_path = os.path.join(args.output_dir, "no_skip_vs_mixed_skip_table.csv")
    report_path = os.path.join(args.output_dir, "no_skip_vs_mixed_skip_report.md")
    system_path = os.path.join(args.output_dir, "no_skip_vs_mixed_skip_system_metrics.png")
    calls_path = os.path.join(args.output_dir, "temporal_skip_call_counts.png")
    overhead_path = os.path.join(args.output_dir, "temporal_skip_overhead.png")

    write_csv(table_path, rows)
    write_markdown(report_path, rows, no_skip, skip)
    plot_system(no_skip, skip, system_path)
    plot_calls(skip, calls_path)
    plot_overhead(no_skip, skip, overhead_path)

    print(
        json.dumps(
            {
                "output_dir": os.path.abspath(args.output_dir),
                "files": {
                    "table_csv": table_path,
                    "report_md": report_path,
                    "system_metrics": system_path,
                    "call_counts": calls_path,
                    "overhead": overhead_path,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
