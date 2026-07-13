#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare pure-PC student baseline against Nano HIL student result."""

from __future__ import print_function

import argparse
import csv
import json
import os


METRICS = [
    "average_reward",
    "average_delay",
    "average_drop_rate",
    "average_offload_rate",
    "target_action_agreement",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path, rows):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fieldnames = ["metric", "pc_student", "hil_nano", "diff_hil_minus_pc", "relative_diff"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def as_float_or_none(value):
    if value is None:
        return None
    return float(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PC student baseline and Nano HIL summaries.")
    parser.add_argument(
        "--pc_summary",
        default=os.path.join("results", "pc_student_baseline", "pc_student_baseline_summary.json"),
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
        "--output_dir",
        default=os.path.join("results", "hil_vs_pc_student_comparison"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pc = load_json(args.pc_summary)
    hil = load_json(args.hil_summary)

    rows = []
    for metric in METRICS:
        pc_value = as_float_or_none(pc.get(metric))
        hil_value = as_float_or_none(hil.get(metric))
        diff = None
        relative = None
        if pc_value is not None and hil_value is not None:
            diff = hil_value - pc_value
            if abs(pc_value) > 1e-12:
                relative = diff / abs(pc_value)
        rows.append(
            {
                "metric": metric,
                "pc_student": pc_value,
                "hil_nano": hil_value,
                "diff_hil_minus_pc": diff,
                "relative_diff": relative,
            }
        )

    comparison = {
        "pc_summary": os.path.abspath(args.pc_summary),
        "hil_summary": os.path.abspath(args.hil_summary),
        "pc_total_steps": pc.get("total_steps"),
        "hil_total_steps": hil.get("total_steps"),
        "pc_average_pc_student_infer_ms": pc.get("average_pc_student_infer_ms"),
        "hil_average_nano_infer_ms": hil.get("average_nano_infer_ms"),
        "hil_average_round_trip_ms": hil.get("average_round_trip_ms"),
        "metrics": rows,
        "note": (
            "diff_hil_minus_pc = HIL Nano result - pure PC student baseline. "
            "The HIL round-trip time is reported separately and is not included "
            "in simulated MEC delay."
        ),
    }

    output_dir = os.path.abspath(args.output_dir)
    csv_path = os.path.join(output_dir, "hil_vs_pc_student_comparison.csv")
    json_path = os.path.join(output_dir, "hil_vs_pc_student_comparison.json")
    write_csv(csv_path, rows)
    write_json(json_path, comparison)

    print("Saved comparison CSV: {}".format(csv_path))
    print("Saved comparison JSON: {}".format(json_path))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
