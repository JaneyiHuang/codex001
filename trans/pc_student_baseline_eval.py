#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure-PC student baseline for the single-UE HIL experiment.

This script evaluates the matching all-simulated baseline:
  - UE0, UE1, UE2, UE3 are all simulated on the PC.
  - All 4 UEs use the same lightweight student actor.
  - The temporal repeat head is recorded by the model checkpoint but is not used
    for action skipping here; every UE makes a decision every slot.

Use this baseline to compare against:
  - UE0 on Jetson Nano running ONNX student
  - UE1-UE3 simulated on PC running student

The comparison answers whether the deployed Nano ONNX student behaves like the
same lightweight student policy in pure simulation.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pc_hil_single_ue_eval import (
    build_actor_policy,
    checked_action,
    ensure_obs_shape,
    extract_metrics,
    import_torch,
    make_env,
    mean_or_none,
    parse_reset_result,
    parse_step_result,
    resolve_model_path,
    write_json,
)


CSV_FIELDS = [
    "episode",
    "step",
    "target_agent_id",
    "student_action_target",
    "teacher_action_target",
    "action_match",
    "reward",
    "delay",
    "drop",
    "offload_rate",
    "pc_student_infer_ms",
    "joint_actions",
]


def write_csv(path, rows):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pure-PC 4-student baseline matching the single-UE HIL setup."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--target_agent_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--student_model_path_pc", default=None)
    parser.add_argument("--teacher_model_path", default=None)
    parser.add_argument("--obs_dim", type=int, default=10)
    parser.add_argument("--n_actions", type=int, default=2)
    parser.add_argument("--num_users", type=int, default=4)
    parser.add_argument("--output_dir", default=os.path.join("results", "pc_student_baseline"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.target_agent_id < 0 or args.target_agent_id >= args.num_users:
        raise ValueError("target_agent_id must be in [0, num_users).")

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import_torch()[0].manual_seed(args.seed)
    except Exception:
        pass

    student_candidates = [
        os.path.join("temporal", "results", "mappo_actor_temporal_distilled_p25.pt"),
        os.path.join("results", "mappo_actor_pruned_distilled_p25.pt"),
        os.path.join("export", "mappo_actor_temporal_distilled_p25.onnx"),
    ]
    teacher_candidates = [
        os.path.join("export", "mappo_teacher_actor.pt"),
        os.path.join("results", "mappo_checkpoint.pt"),
        os.path.join("export", "mappo_teacher_actor.onnx"),
        os.path.join("export", "mappo_teacher_actor_opset11.onnx"),
    ]

    student_path = resolve_model_path(
        args.student_model_path_pc,
        student_candidates,
        "student model for pure-PC baseline",
        required=True,
    )
    student_policy = build_actor_policy(
        student_path,
        obs_dim=args.obs_dim,
        n_actions=args.n_actions,
        device=args.device,
    )
    print("Loaded PC-side student actor: {}".format(student_path))

    teacher_path = resolve_model_path(
        args.teacher_model_path,
        teacher_candidates,
        "teacher model",
        required=False,
    )
    teacher_policy = None
    if teacher_path:
        teacher_policy = build_actor_policy(
            teacher_path,
            obs_dim=args.obs_dim,
            n_actions=args.n_actions,
            device=args.device,
        )
        print("Loaded teacher actor for target agreement: {}".format(teacher_path))

    env = make_env(
        num_users=args.num_users,
        steps=args.steps,
        obs_dim=args.obs_dim,
        n_actions=args.n_actions,
        seed=args.seed,
    )

    rows = []
    rewards = []
    delays = []
    drops = []
    offloads = []
    pc_infer_times = []
    action_matches = []

    print(
        "Starting pure-PC student baseline: episodes={}, steps={}, num_users={}, target_agent_id={}".format(
            args.episodes,
            args.steps,
            args.num_users,
            args.target_agent_id,
        )
    )

    for episode in range(int(args.episodes)):
        obs_all = ensure_obs_shape(parse_reset_result(env.reset()), args.num_users, args.obs_dim)

        for step in range(int(args.steps)):
            target_obs = obs_all[int(args.target_agent_id)]
            teacher_action_target = None
            action_match = None
            if teacher_policy is not None:
                teacher_action_target = checked_action(
                    teacher_policy.predict_one(target_obs),
                    args.n_actions,
                    "teacher target policy",
                )

            start = time.perf_counter()
            joint_actions_np = student_policy.predict_batch(obs_all)
            pc_student_infer_ms = (time.perf_counter() - start) * 1000.0
            joint_actions = [
                checked_action(action, args.n_actions, "PC student policy")
                for action in joint_actions_np.tolist()
            ]
            student_action_target = int(joint_actions[int(args.target_agent_id)])

            if teacher_action_target is not None:
                action_match = int(int(teacher_action_target) == int(student_action_target))
                action_matches.append(action_match)

            try:
                step_result = env.step(joint_actions)
            except (AssertionError, TypeError, ValueError):
                step_result = env.step(np.asarray(joint_actions, dtype=np.int64))

            next_obs_all, reward, done, info = parse_step_result(step_result)
            delay, drop, offload_rate = extract_metrics(info, joint_actions, args.num_users)

            rewards.append(reward)
            if delay is not None:
                delays.append(delay)
            if drop is not None:
                drops.append(drop)
            if offload_rate is not None:
                offloads.append(offload_rate)
            pc_infer_times.append(float(pc_student_infer_ms))

            rows.append(
                {
                    "episode": int(episode),
                    "step": int(step),
                    "target_agent_id": int(args.target_agent_id),
                    "student_action_target": student_action_target,
                    "teacher_action_target": teacher_action_target,
                    "action_match": action_match,
                    "reward": float(reward),
                    "delay": delay,
                    "drop": drop,
                    "offload_rate": offload_rate,
                    "pc_student_infer_ms": float(pc_student_infer_ms),
                    "joint_actions": json.dumps([int(a) for a in joint_actions]),
                }
            )

            obs_all = ensure_obs_shape(next_obs_all, args.num_users, args.obs_dim)
            if done:
                break

        print("Finished episode {}/{}.".format(episode + 1, args.episodes))

    total_steps = len(rows)
    output_dir = os.path.abspath(args.output_dir)
    csv_path = os.path.join(output_dir, "pc_student_baseline_steps.csv")
    summary_path = os.path.join(output_dir, "pc_student_baseline_summary.json")

    agreement = None
    if action_matches:
        agreement = float(np.mean(np.asarray(action_matches, dtype=np.float32)))

    summary = {
        "episodes": int(args.episodes),
        "steps_per_episode": int(args.steps),
        "num_users": int(args.num_users),
        "target_agent_id": int(args.target_agent_id),
        "policy": "pc_all_student",
        "average_reward": mean_or_none(rewards),
        "average_delay": mean_or_none(delays),
        "average_drop_rate": mean_or_none(drops),
        "average_offload_rate": mean_or_none(offloads),
        "average_pc_student_infer_ms": mean_or_none(pc_infer_times),
        "target_action_agreement": agreement,
        "total_steps": int(total_steps),
        "teacher_model_path": teacher_path,
        "student_model_path_pc": student_path,
        "csv_path": csv_path,
        "summary_path": summary_path,
        "note": (
            "Pure-PC baseline: all 4 UEs are simulated and use the lightweight "
            "student actor every slot. Temporal repeat/skip is not used, matching "
            "the first-version HIL protocol."
        ),
    }

    write_csv(csv_path, rows)
    write_json(summary_path, summary)

    print("Saved step CSV: {}".format(csv_path))
    print("Saved summary JSON: {}".format(summary_path))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
