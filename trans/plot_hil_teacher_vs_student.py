#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Nano-HIL student vs Nano-HIL teacher comparison."""

from __future__ import print_function

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    ("average_reward", "Average reward"),
    ("average_delay", "Average delay"),
    ("average_drop_rate", "Drop rate"),
    ("average_offload_rate", "Offload rate"),
    ("target_action_agreement", "Agreement with teacher"),
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
    fieldnames = ["metric", "student_hil", "teacher_hil", "diff_student_minus_teacher", "relative_diff"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, student_summary, teacher_summary):
    ensure_dir(os.path.dirname(path))
    lines = []
    lines.append("# Nano-HIL Student vs Nano-HIL Teacher")
    lines.append("")
    lines.append("| Metric | Student HIL | Teacher HIL | Student - Teacher | Relative diff |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in rows:
        rel = row["relative_diff"]
        lines.append(
            "| {metric} | {student:.6g} | {teacher:.6g} | {diff:.6g} | {rel} |".format(
                metric=row["metric"],
                student=row["student_hil"],
                teacher=row["teacher_hil"],
                diff=row["diff_student_minus_teacher"],
                rel="" if rel is None else "{:.4%}".format(rel),
            )
        )
    lines.append("")
    lines.append("## Deployment Overhead")
    lines.append("")
    lines.append("| Metric | Student HIL | Teacher HIL |")
    lines.append("|---|---:|---:|")
    lines.append(
        "| Nano pure infer ms | {:.6g} | {:.6g} |".format(
            value(student_summary, "average_nano_infer_ms"),
            value(teacher_summary, "average_nano_infer_ms"),
        )
    )
    lines.append(
        "| PC-Nano round trip ms | {:.6g} | {:.6g} |".format(
            value(student_summary, "average_round_trip_ms"),
            value(teacher_summary, "average_round_trip_ms"),
        )
    )
    lines.append("")
    lines.append("Note: round-trip latency is not added to simulated MEC delay.")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_rows(student_summary, teacher_summary):
    rows = []
    for key, label in METRICS:
        s = value(student_summary, key)
        t = value(teacher_summary, key)
        diff = None if s is None or t is None else s - t
        rel = None
        if diff is not None and abs(t) > 1e-12:
            rel = diff / abs(t)
        rows.append(
            {
                "metric": label,
                "student_hil": s,
                "teacher_hil": t,
                "diff_student_minus_teacher": diff,
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


def plot_system(student_summary, teacher_summary, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()
    labels = ["Student HIL", "Teacher HIL"]
    colors = ["#F97316", "#2563EB"]
    for ax, (key, title) in zip(axes, METRICS):
        vals = [value(student_summary, key), value(teacher_summary, key)]
        bars = ax.bar(labels, vals, color=colors, width=0.58)
        ax.set_title(title)
        apply_grid(ax)
        annotate(ax, bars)
        if key in ("average_drop_rate", "average_offload_rate", "target_action_agreement"):
            ax.set_ylim(0.0, max(1.05, max(vals) * 1.18))
    axes[-1].axis("off")
    axes[-1].text(
        0.0,
        0.75,
        "Student HIL: Nano ONNX student + PC student UEs\n"
        "Teacher HIL: Nano ONNX teacher + PC teacher UEs\n"
        "Both use the same PC MEC environment protocol.",
        fontsize=11,
        va="top",
        color="#111827",
    )
    fig.suptitle("Nano-HIL Student vs Nano-HIL Teacher", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_overhead(student_summary, teacher_summary, path):
    labels = ["Student\nNano infer", "Teacher\nNano infer", "Student\nround trip", "Teacher\nround trip"]
    vals = [
        value(student_summary, "average_nano_infer_ms"),
        value(teacher_summary, "average_nano_infer_ms"),
        value(student_summary, "average_round_trip_ms"),
        value(teacher_summary, "average_round_trip_ms"),
    ]
    colors = ["#F97316", "#2563EB", "#FB923C", "#60A5FA"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=colors, width=0.58)
    ax.set_title("Nano Inference and Communication Overhead")
    ax.set_ylabel("Time (ms)")
    apply_grid(ax)
    annotate(ax, bars, fmt="{:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Nano-HIL student vs teacher summaries.")
    parser.add_argument(
        "--student_summary",
        default=os.path.join(
            "results",
            "hil_single_ue_onnx_nano_student_others",
            "hil_single_ue_summary.json",
        ),
    )
    parser.add_argument(
        "--teacher_summary",
        default=os.path.join(
            "results",
            "hil_single_ue_onnx_nano_teacher_others",
            "hil_single_ue_summary.json",
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("results", "hil_teacher_vs_student_comparison_20x300", "visuals"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    student = load_json(args.student_summary)
    teacher = load_json(args.teacher_summary)
    rows = build_rows(student, teacher)
    ensure_dir(args.output_dir)

    table_path = os.path.join(args.output_dir, "hil_teacher_vs_student_table.csv")
    report_path = os.path.join(args.output_dir, "hil_teacher_vs_student_report.md")
    system_path = os.path.join(args.output_dir, "hil_teacher_vs_student_system_metrics.png")
    overhead_path = os.path.join(args.output_dir, "hil_teacher_vs_student_overhead.png")

    write_csv(table_path, rows)
    write_markdown(report_path, rows, student, teacher)
    plot_system(student, teacher, system_path)
    plot_overhead(student, teacher, overhead_path)

    print(
        json.dumps(
            {
                "output_dir": os.path.abspath(args.output_dir),
                "files": {
                    "table_csv": table_path,
                    "report_md": report_path,
                    "system_metrics": system_path,
                    "overhead": overhead_path,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
