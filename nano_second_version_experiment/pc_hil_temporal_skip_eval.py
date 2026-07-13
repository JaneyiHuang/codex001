#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Second-version Nano HIL experiment with mixed temporal skip.

Experiment design:
  - UE0: Jetson Nano student ONNX, using repeat_pred to reuse cached action.
  - UE1: PC simulated temporal student, also using repeat_pred to reuse action.
  - UE2-UE3: PC simulated student, making a fresh decision every slot.
  - PC: MEC environment, edge queue, reward, delay, and drop calculation remain
    unchanged.

This script is intentionally separate from the first-version HIL evaluator.
The first version records repeat_pred but still calls Nano every slot. This
second version actually uses repeat_pred for UE0 and UE1.
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
TRANS_DIR = os.path.join(PROJECT_ROOT, "trans")
if TRANS_DIR not in sys.path:
    sys.path.insert(0, TRANS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pc_hil_single_ue_eval import (
    JsonLineClient,
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
    "pc_repeat_agent_id",
    "nano_action",
    "nano_called",
    "nano_skip_used",
    "nano_repeat_raw",
    "nano_repeat_slots",
    "nano_remaining_before",
    "nano_remaining_after",
    "nano_safety_interrupted",
    "nano_interrupt_reason",
    "nano_infer_ms",
    "round_trip_ms",
    "pc_repeat_action",
    "pc_repeat_called",
    "pc_repeat_skip_used",
    "pc_repeat_repeat_raw",
    "pc_repeat_repeat_slots",
    "pc_repeat_remaining_before",
    "pc_repeat_remaining_after",
    "pc_repeat_safety_interrupted",
    "pc_repeat_interrupt_reason",
    "pc_repeat_infer_ms",
    "pc_regular_agent_ids",
    "pc_regular_infer_ms",
    "teacher_action_target",
    "target_action_match",
    "reward",
    "delay",
    "drop",
    "offload_rate",
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


def repeat_slots_from_raw(repeat_raw, repeat_scale, max_repeat):
    if repeat_raw is None:
        return 0
    slots = int(round(max(float(repeat_raw), 0.0) * float(repeat_scale)))
    return int(np.clip(slots, 0, int(max_repeat)))


def first_scalar(value):
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    return float(arr[0])


def predict_action_repeat_one(policy, obs, n_actions, repeat_scale, max_repeat):
    """Return action, repeat_raw, repeat_slots, infer_ms for one observation."""
    obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    start = time.perf_counter()

    if hasattr(policy, "session"):
        outputs = policy.session.run(None, {policy.input_name: obs_batch})
        logits = np.asarray(outputs[0], dtype=np.float32)
        repeat_raw = first_scalar(outputs[1]) if len(outputs) >= 2 else None
    else:
        torch = policy.torch
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=policy.device)
        with torch.no_grad():
            output = policy.actor(obs_tensor)
            if isinstance(output, (tuple, list)):
                logits_tensor = output[0]
                repeat_tensor = output[1] if len(output) >= 2 else None
            else:
                logits_tensor = output
                repeat_tensor = None
        logits = logits_tensor.detach().cpu().numpy().astype(np.float32)
        repeat_raw = first_scalar(repeat_tensor.detach().cpu().numpy()) if repeat_tensor is not None else None

    infer_ms = (time.perf_counter() - start) * 1000.0
    action = int(np.argmax(logits, axis=-1).reshape(-1)[0])
    action = checked_action(action, n_actions, "temporal student policy")
    repeat_slots = repeat_slots_from_raw(repeat_raw, repeat_scale, max_repeat)
    return action, repeat_raw, repeat_slots, infer_ms


class TemporalSkipState(object):
    def __init__(self):
        self.cached_action = 0
        self.remaining_repeat = 0
        self.last_decision_obs = None
        self.has_decision = False


def safety_interrupt(
    state,
    obs,
    safety_enabled,
    safe_energy_min,
    safe_queue_threshold,
    safe_obs_change_threshold,
    safe_channel_drop_threshold,
):
    if not safety_enabled or not state.has_decision or state.remaining_repeat <= 0:
        return False, ""

    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    prev = np.asarray(state.last_decision_obs, dtype=np.float32).reshape(-1)
    reasons = []

    if obs.size > 1 and obs[1] <= float(safe_energy_min):
        reasons.append("energy_low")

    if obs.size > 8:
        queue_pressure = max(float(obs[4]), float(obs[5]), float(obs[8]))
        if queue_pressure >= float(safe_queue_threshold):
            reasons.append("queue_pressure")

    critical_indices = [1, 4, 5, 6, 7, 8, 9]
    valid_indices = [idx for idx in critical_indices if idx < obs.size and idx < prev.size]
    if valid_indices:
        critical_change = float(np.max(np.abs(obs[valid_indices] - prev[valid_indices])))
        if critical_change >= float(safe_obs_change_threshold):
            reasons.append("obs_changed")

    if state.cached_action == 1 and obs.size > 3 and prev.size > 3:
        if float(prev[3] - obs[3]) >= float(safe_channel_drop_threshold):
            reasons.append("channel_drop")

    return bool(reasons), "|".join(reasons)


def update_skip_state_after_call(state, action, repeat_slots, obs):
    state.cached_action = int(action)
    state.remaining_repeat = int(repeat_slots)
    state.last_decision_obs = np.asarray(obs, dtype=np.float32).copy()
    state.has_decision = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Second-version HIL: UE0 Nano skip, UE1 PC skip, UE2-UE3 per-step."
    )
    parser.add_argument("--nano_ip", required=True)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--target_agent_id", type=int, default=0)
    parser.add_argument("--pc_repeat_agent_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--student_model_path_pc", default=None)
    parser.add_argument("--teacher_model_path", default=None)
    parser.add_argument("--obs_dim", type=int, default=10)
    parser.add_argument("--n_actions", type=int, default=2)
    parser.add_argument("--num_users", type=int, default=4)
    parser.add_argument("--repeat_scale", type=float, default=5.0)
    parser.add_argument("--max_repeat", type=int, default=5)
    parser.add_argument("--disable_safety", action="store_true")
    parser.add_argument("--safe_energy_min", type=float, default=0.12)
    parser.add_argument("--safe_queue_threshold", type=float, default=0.80)
    parser.add_argument("--safe_obs_change_threshold", type=float, default=0.35)
    parser.add_argument("--safe_channel_drop_threshold", type=float, default=0.35)
    parser.add_argument(
        "--output_dir",
        default=os.path.join("nano_second_version_experiment", "results", "mixed_skip_20x300"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--socket_timeout", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.target_agent_id < 0 or args.target_agent_id >= args.num_users:
        raise ValueError("target_agent_id must be in [0, num_users).")
    if args.pc_repeat_agent_id < 0 or args.pc_repeat_agent_id >= args.num_users:
        raise ValueError("pc_repeat_agent_id must be in [0, num_users).")
    if args.pc_repeat_agent_id == args.target_agent_id:
        raise ValueError("pc_repeat_agent_id must be different from target_agent_id.")
    if args.num_users < 4:
        raise ValueError("This mixed experiment expects num_users >= 4.")

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
        "student model for PC-side UEs",
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
    client = JsonLineClient(args.nano_ip, args.port, timeout=args.socket_timeout)

    safety_enabled = not bool(args.disable_safety)
    per_step_agent_ids = [
        idx
        for idx in range(int(args.num_users))
        if idx not in (int(args.target_agent_id), int(args.pc_repeat_agent_id))
    ]

    rows = []
    rewards = []
    delays = []
    drops = []
    offloads = []
    target_matches = []

    nano_infer_call_times = []
    nano_round_trip_call_times = []
    pc_repeat_infer_call_times = []
    pc_regular_infer_times = []
    nano_repeat_slots_list = []
    pc_repeat_slots_list = []

    nano_model_call_count = 0
    nano_skip_count = 0
    nano_safety_interrupt_count = 0
    pc_repeat_model_call_count = 0
    pc_repeat_skip_count = 0
    pc_repeat_safety_interrupt_count = 0
    pc_regular_model_call_count = 0

    print("Connected to Nano at {}:{}.".format(args.nano_ip, args.port))
    print(
        "Starting mixed temporal-skip HIL: episodes={}, steps={}, target UE={}, PC repeat UE={}, per-step UEs={}".format(
            args.episodes,
            args.steps,
            args.target_agent_id,
            args.pc_repeat_agent_id,
            per_step_agent_ids,
        )
    )

    try:
        for episode in range(int(args.episodes)):
            obs_all = ensure_obs_shape(parse_reset_result(env.reset()), args.num_users, args.obs_dim)
            nano_state = TemporalSkipState()
            pc_repeat_state = TemporalSkipState()

            for step in range(int(args.steps)):
                joint_actions = [0 for _ in range(int(args.num_users))]

                target_obs = obs_all[int(args.target_agent_id)]
                teacher_action_target = None
                target_action_match = None
                if teacher_policy is not None:
                    teacher_action_target = checked_action(
                        teacher_policy.predict_one(target_obs),
                        args.n_actions,
                        "teacher target policy",
                    )

                # ---------------- UE0: Nano student with temporal skip ----------------
                nano_remaining_before = int(nano_state.remaining_repeat)
                nano_interrupted, nano_reason = safety_interrupt(
                    nano_state,
                    target_obs,
                    safety_enabled,
                    args.safe_energy_min,
                    args.safe_queue_threshold,
                    args.safe_obs_change_threshold,
                    args.safe_channel_drop_threshold,
                )
                nano_called = False
                nano_skip_used = False
                nano_repeat_raw = None
                nano_repeat_slots = None
                nano_infer_ms = None
                round_trip_ms = None

                if nano_remaining_before > 0 and not nano_interrupted:
                    nano_action = int(nano_state.cached_action)
                    nano_state.remaining_repeat -= 1
                    nano_skip_used = True
                    nano_skip_count += 1
                else:
                    if nano_interrupted:
                        nano_safety_interrupt_count += 1
                        nano_state.remaining_repeat = 0
                    nano_response, round_trip_ms = client.request_action(episode, step, target_obs)
                    nano_action = checked_action(nano_response.get("action"), args.n_actions, "Nano")
                    nano_infer_ms = float(nano_response.get("infer_ms", 0.0))
                    nano_repeat_raw = nano_response.get("repeat_pred")
                    nano_repeat_slots = repeat_slots_from_raw(
                        nano_repeat_raw,
                        args.repeat_scale,
                        args.max_repeat,
                    )
                    update_skip_state_after_call(nano_state, nano_action, nano_repeat_slots, target_obs)
                    nano_called = True
                    nano_model_call_count += 1
                    nano_infer_call_times.append(nano_infer_ms)
                    nano_round_trip_call_times.append(float(round_trip_ms))
                    nano_repeat_slots_list.append(float(nano_repeat_slots))

                joint_actions[int(args.target_agent_id)] = int(nano_action)
                if teacher_action_target is not None:
                    target_action_match = int(int(teacher_action_target) == int(nano_action))
                    target_matches.append(target_action_match)

                # ---------------- UE1: PC student with temporal skip ----------------
                repeat_idx = int(args.pc_repeat_agent_id)
                repeat_obs = obs_all[repeat_idx]
                pc_remaining_before = int(pc_repeat_state.remaining_repeat)
                pc_interrupted, pc_reason = safety_interrupt(
                    pc_repeat_state,
                    repeat_obs,
                    safety_enabled,
                    args.safe_energy_min,
                    args.safe_queue_threshold,
                    args.safe_obs_change_threshold,
                    args.safe_channel_drop_threshold,
                )
                pc_repeat_called = False
                pc_repeat_skip_used = False
                pc_repeat_repeat_raw = None
                pc_repeat_repeat_slots = None
                pc_repeat_infer_ms = None

                if pc_remaining_before > 0 and not pc_interrupted:
                    pc_repeat_action = int(pc_repeat_state.cached_action)
                    pc_repeat_state.remaining_repeat -= 1
                    pc_repeat_skip_used = True
                    pc_repeat_skip_count += 1
                else:
                    if pc_interrupted:
                        pc_repeat_safety_interrupt_count += 1
                        pc_repeat_state.remaining_repeat = 0
                    (
                        pc_repeat_action,
                        pc_repeat_repeat_raw,
                        pc_repeat_repeat_slots,
                        pc_repeat_infer_ms,
                    ) = predict_action_repeat_one(
                        student_policy,
                        repeat_obs,
                        args.n_actions,
                        args.repeat_scale,
                        args.max_repeat,
                    )
                    update_skip_state_after_call(
                        pc_repeat_state,
                        pc_repeat_action,
                        pc_repeat_repeat_slots,
                        repeat_obs,
                    )
                    pc_repeat_called = True
                    pc_repeat_model_call_count += 1
                    pc_repeat_infer_call_times.append(float(pc_repeat_infer_ms))
                    pc_repeat_slots_list.append(float(pc_repeat_repeat_slots))

                joint_actions[repeat_idx] = int(pc_repeat_action)

                # ---------------- UE2-UE3: PC student every-step decisions ----------------
                start_regular = time.perf_counter()
                regular_actions = student_policy.predict_batch(obs_all[per_step_agent_ids])
                pc_regular_infer_ms = (time.perf_counter() - start_regular) * 1000.0
                pc_regular_model_call_count += 1
                pc_regular_infer_times.append(float(pc_regular_infer_ms))
                for local_idx, agent_id in enumerate(per_step_agent_ids):
                    joint_actions[int(agent_id)] = checked_action(
                        regular_actions[local_idx],
                        args.n_actions,
                        "PC regular student policy",
                    )

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

                rows.append(
                    {
                        "episode": int(episode),
                        "step": int(step),
                        "target_agent_id": int(args.target_agent_id),
                        "pc_repeat_agent_id": int(args.pc_repeat_agent_id),
                        "nano_action": int(nano_action),
                        "nano_called": int(nano_called),
                        "nano_skip_used": int(nano_skip_used),
                        "nano_repeat_raw": nano_repeat_raw,
                        "nano_repeat_slots": nano_repeat_slots,
                        "nano_remaining_before": nano_remaining_before,
                        "nano_remaining_after": int(nano_state.remaining_repeat),
                        "nano_safety_interrupted": int(nano_interrupted),
                        "nano_interrupt_reason": nano_reason,
                        "nano_infer_ms": nano_infer_ms,
                        "round_trip_ms": round_trip_ms,
                        "pc_repeat_action": int(pc_repeat_action),
                        "pc_repeat_called": int(pc_repeat_called),
                        "pc_repeat_skip_used": int(pc_repeat_skip_used),
                        "pc_repeat_repeat_raw": pc_repeat_repeat_raw,
                        "pc_repeat_repeat_slots": pc_repeat_repeat_slots,
                        "pc_repeat_remaining_before": pc_remaining_before,
                        "pc_repeat_remaining_after": int(pc_repeat_state.remaining_repeat),
                        "pc_repeat_safety_interrupted": int(pc_interrupted),
                        "pc_repeat_interrupt_reason": pc_reason,
                        "pc_repeat_infer_ms": pc_repeat_infer_ms,
                        "pc_regular_agent_ids": json.dumps([int(i) for i in per_step_agent_ids]),
                        "pc_regular_infer_ms": float(pc_regular_infer_ms),
                        "teacher_action_target": teacher_action_target,
                        "target_action_match": target_action_match,
                        "reward": float(reward),
                        "delay": delay,
                        "drop": drop,
                        "offload_rate": offload_rate,
                        "joint_actions": json.dumps([int(a) for a in joint_actions]),
                    }
                )

                obs_all = ensure_obs_shape(next_obs_all, args.num_users, args.obs_dim)
                if done:
                    break

            print("Finished episode {}/{}.".format(episode + 1, args.episodes))
    finally:
        client.close()

    total_steps = len(rows)
    output_dir = os.path.abspath(args.output_dir)
    csv_path = os.path.join(output_dir, "mixed_temporal_skip_steps.csv")
    summary_path = os.path.join(output_dir, "mixed_temporal_skip_summary.json")

    target_agreement = None
    if target_matches:
        target_agreement = float(np.mean(np.asarray(target_matches, dtype=np.float32)))

    nano_call_rate = float(nano_model_call_count / float(max(total_steps, 1)))
    pc_repeat_call_rate = float(pc_repeat_model_call_count / float(max(total_steps, 1)))
    summary = {
        "experiment": "mixed_temporal_skip_hil",
        "episodes": int(args.episodes),
        "steps_per_episode": int(args.steps),
        "num_users": int(args.num_users),
        "target_agent_id": int(args.target_agent_id),
        "pc_repeat_agent_id": int(args.pc_repeat_agent_id),
        "pc_regular_agent_ids": [int(i) for i in per_step_agent_ids],
        "repeat_scale": float(args.repeat_scale),
        "max_repeat": int(args.max_repeat),
        "safety_enabled": bool(safety_enabled),
        "average_reward": mean_or_none(rewards),
        "average_delay": mean_or_none(delays),
        "average_drop_rate": mean_or_none(drops),
        "average_offload_rate": mean_or_none(offloads),
        "target_action_agreement": target_agreement,
        "total_steps": int(total_steps),
        "nano_model_call_count": int(nano_model_call_count),
        "nano_skip_count": int(nano_skip_count),
        "nano_model_call_rate": nano_call_rate,
        "nano_call_reduction_rate": float(1.0 - nano_call_rate),
        "nano_safety_interrupt_count": int(nano_safety_interrupt_count),
        "pc_repeat_model_call_count": int(pc_repeat_model_call_count),
        "pc_repeat_skip_count": int(pc_repeat_skip_count),
        "pc_repeat_model_call_rate": pc_repeat_call_rate,
        "pc_repeat_call_reduction_rate": float(1.0 - pc_repeat_call_rate),
        "pc_repeat_safety_interrupt_count": int(pc_repeat_safety_interrupt_count),
        "pc_regular_model_call_count": int(pc_regular_model_call_count),
        "average_nano_infer_ms_per_call": mean_or_none(nano_infer_call_times),
        "average_round_trip_ms_per_call": mean_or_none(nano_round_trip_call_times),
        "average_nano_infer_ms_per_env_step": float(sum(nano_infer_call_times) / float(max(total_steps, 1))),
        "average_round_trip_ms_per_env_step": float(sum(nano_round_trip_call_times) / float(max(total_steps, 1))),
        "average_pc_repeat_infer_ms_per_call": mean_or_none(pc_repeat_infer_call_times),
        "average_pc_repeat_infer_ms_per_env_step": float(sum(pc_repeat_infer_call_times) / float(max(total_steps, 1))),
        "average_pc_regular_infer_ms_per_step": mean_or_none(pc_regular_infer_times),
        "average_nano_repeat_slots_per_call": mean_or_none(nano_repeat_slots_list),
        "average_pc_repeat_slots_per_call": mean_or_none(pc_repeat_slots_list),
        "teacher_model_path": teacher_path,
        "student_model_path_pc": student_path,
        "csv_path": csv_path,
        "summary_path": summary_path,
        "note": (
            "UE0 Nano and UE1 PC use temporal repeat skip. UE2-UE3 keep every-slot "
            "student decisions. reward/delay/drop/offload_rate are still simulated "
            "MEC metrics from PC env.step(); deployment communication overhead is "
            "reported separately."
        ),
    }

    write_csv(csv_path, rows)
    write_json(summary_path, summary)

    print("Saved step CSV: {}".format(csv_path))
    print("Saved summary JSON: {}".format(summary_path))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
