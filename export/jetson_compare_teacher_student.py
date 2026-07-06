#!/usr/bin/env python3
"""
Compare teacher MAPPO actor and temporal-distilled student on Jetson Nano.

The script reports:
  - one-step action agreement on the same simulated trajectory,
  - actor-forward latency,
  - policy-step latency with temporal action reuse,
  - closed-loop simulated delay/drop/reward.

Typical quick run on Jetson:
    python3 export/jetson_compare_teacher_student.py --episodes 10 --agreement-episodes 10

Fair ONNX-vs-ONNX latency run:
    python3 export/jetson_compare_teacher_student.py \
      --teacher-runtime onnx --teacher-model-path export/mappo_teacher_actor.onnx \
      --student-runtime onnx --student-model-path export/mappo_actor_temporal_distilled_p25.onnx
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from jetson_infer import (
    JetsonTemporalPolicy,
    create_onnx_session,
    load_checkpoint,
    load_json_file,
    resolve_metadata_path,
    summarize_latency,
    synchronize_device,
)
from jetson_eval_sim import EnvConfig, MECEnv, build_agent_states, build_edge_state


TEACHER_BASENAME = "mappo_teacher_actor"
STUDENT_BASENAME = "mappo_actor_temporal_distilled_p25"


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLPBlock, self).__init__()
        layers = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TeacherActor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dims):
        super(TeacherActor, self).__init__()
        self.mlp = MLPBlock(
            input_dim=int(obs_dim),
            hidden_dims=[int(value) for value in hidden_dims],
            output_dim=int(n_actions),
        )

    def forward(self, obs):
        return self.mlp(obs)


def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def project_root():
    return os.path.abspath(os.path.join(script_dir(), os.pardir))


def candidate_paths(filename):
    return [
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "export", filename),
        os.path.join(script_dir(), filename),
        os.path.join(project_root(), "export", filename),
    ]


def resolve_existing_path(path, candidates, description):
    if path:
        return os.path.abspath(path)
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0])


def resolve_teacher_model_path(path, runtime):
    runtime = str(runtime or "auto").lower()
    if runtime == "onnx":
        candidates = candidate_paths(TEACHER_BASENAME + ".onnx")
    else:
        candidates = candidate_paths(TEACHER_BASENAME + ".pt")
        candidates.append(os.path.join(project_root(), "results", "mappo_checkpoint.pt"))
    return resolve_existing_path(path, candidates, "teacher model")


def resolve_student_model_path(path, runtime):
    runtime = str(runtime or "auto").lower()
    if runtime == "torch":
        candidates = [
            os.path.join(project_root(), "temporal", "results", STUDENT_BASENAME + ".pt"),
            os.path.join(script_dir(), STUDENT_BASENAME + ".pt"),
            os.path.join(os.getcwd(), "temporal", "results", STUDENT_BASENAME + ".pt"),
        ]
    elif runtime == "onnx":
        candidates = candidate_paths(STUDENT_BASENAME + ".onnx")
    else:
        candidates = candidate_paths(STUDENT_BASENAME + ".onnx") + [
            os.path.join(project_root(), "temporal", "results", STUDENT_BASENAME + ".pt"),
            os.path.join(script_dir(), STUDENT_BASENAME + ".pt"),
        ]
    return resolve_existing_path(path, candidates, "student model")


def resolve_runtime(runtime, model_path):
    runtime = str(runtime or "auto").lower()
    if runtime != "auto":
        return runtime
    if model_path and os.path.splitext(model_path)[1].lower() == ".onnx":
        return "onnx"
    return "torch"


def load_metadata(model_path, metadata_path=None):
    path = metadata_path or resolve_metadata_path(model_path)
    if path and os.path.exists(path):
        data = load_json_file(path)
        if isinstance(data, dict):
            return data
    return {}


def sorted_linear_weight_items(actor_state):
    items = []
    for key, value in actor_state.items():
        if key.startswith("mlp.net.") and key.endswith(".weight"):
            parts = key.split(".")
            try:
                layer_index = int(parts[2])
            except (IndexError, ValueError):
                continue
            items.append((layer_index, key, value))
    return sorted(items, key=lambda item: item[0])


def infer_actor_dims(actor_state):
    weight_items = sorted_linear_weight_items(actor_state)
    if not weight_items:
        raise ValueError("Could not infer teacher actor dimensions from state dict.")
    obs_dim = int(weight_items[0][2].shape[1])
    n_actions = int(weight_items[-1][2].shape[0])
    hidden_dims = [int(item[2].shape[0]) for item in weight_items[:-1]]
    return obs_dim, n_actions, hidden_dims


def count_parameters_from_state(state_dict):
    return int(sum(value.numel() for value in state_dict.values()))


def count_parameters_from_module(module):
    return int(sum(parameter.numel() for parameter in module.parameters()))


class TeacherPolicy(object):
    name = "teacher_mappo_actor"

    def __init__(
        self,
        model_path,
        runtime="torch",
        device="auto",
        metadata_path=None,
        onnx_provider="auto",
    ):
        self.runtime = str(runtime or "torch").lower()
        self.model_path = model_path
        self.actor = None
        self.onnx_session = None
        self.onnx_input_name = None
        self.onnx_providers = []
        metadata = load_metadata(model_path, metadata_path)

        if self.runtime == "torch":
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device

            ckpt = load_checkpoint(model_path, self.device)
            if not isinstance(ckpt, dict) or "actor" not in ckpt:
                raise ValueError("Teacher torch model must contain an actor state dict.")
            actor_state = ckpt["actor"]
            obs_dim, n_actions, hidden_dims = infer_actor_dims(actor_state)

            self.obs_dim = int(ckpt.get("obs_dim", metadata.get("obs_dim", obs_dim)))
            self.n_actions = int(ckpt.get("n_actions", metadata.get("n_actions", n_actions)))
            self.n_agents = int(ckpt.get("n_agents", metadata.get("n_agents", 4)))
            self.actor_hidden_dims = list(ckpt.get("actor_hidden_dims", metadata.get("actor_hidden_dims", hidden_dims)))
            self.actor = TeacherActor(
                obs_dim=self.obs_dim,
                n_actions=self.n_actions,
                hidden_dims=self.actor_hidden_dims,
            ).to(self.device)
            self.actor.load_state_dict(actor_state)
            self.actor.eval()
            self.parameters = count_parameters_from_state(actor_state)
        elif self.runtime == "onnx":
            (
                self.onnx_session,
                self.onnx_input_name,
                self.onnx_providers,
                inferred_obs_dim,
            ) = create_onnx_session(model_path, device=device, provider_mode=onnx_provider)
            self.obs_dim = int(metadata.get("obs_dim", inferred_obs_dim if inferred_obs_dim is not None else 10))
            if inferred_obs_dim is not None and self.obs_dim != inferred_obs_dim:
                raise ValueError(
                    "Teacher ONNX metadata obs_dim={} does not match input obs_dim={}".format(
                        self.obs_dim,
                        inferred_obs_dim,
                    )
                )
            self.n_actions = int(metadata.get("n_actions", 2))
            self.n_agents = int(metadata.get("n_agents", 4))
            self.actor_hidden_dims = list(metadata.get("actor_hidden_dims", []))
            self.parameters = int(metadata.get("parameters", 0))
            self.device = "onnxruntime:{}".format(",".join(self.onnx_providers))
        else:
            raise ValueError("Unsupported teacher runtime: {}".format(runtime))

    def reset(self):
        pass

    def reset_episode(self):
        pass

    def predict_batch(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim:
            raise ValueError("Expected teacher obs batch shape (N, {}), got {}".format(self.obs_dim, obs.shape))

        if self.runtime == "onnx":
            outputs = self.onnx_session.run(None, {self.onnx_input_name: obs})
            logits = np.asarray(outputs[0], dtype=np.float32)
            actions = np.argmax(logits, axis=-1).astype(np.int64)
            return actions, logits

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits_tensor = self.actor(obs_tensor)
        logits = logits_tensor.cpu().numpy().astype(np.float32)
        actions = np.argmax(logits, axis=-1).astype(np.int64)
        return actions, logits

    def select_actions(self, env):
        actions, _ = self.predict_batch(env.get_obs())
        return actions

    def episode_stats(self, n_agents):
        return {
            "model_call_step_ratio": 1.0,
            "decision_agent_ratio": 1.0,
            "safety_interrupt_count": 0.0,
        }


class StudentPolicyAdapter(object):
    name = "student_temporal_distilled_mappo"

    def __init__(
        self,
        model_path,
        runtime,
        device,
        metadata_path,
        onnx_provider,
        safety_enabled,
    ):
        self.policy = JetsonTemporalPolicy(
            model_path=model_path,
            device=device,
            runtime=runtime,
            metadata_path=metadata_path,
            onnx_provider=onnx_provider,
            safety_enabled=safety_enabled,
        )
        self.reset_episode()

    def reset_episode(self):
        self.policy.reset()
        self.steps = 0
        self.model_call_steps = 0
        self.decision_agent_count = 0
        self.safety_interrupt_count = 0

    def select_actions(self, env):
        result = self.policy.select_actions(build_agent_states(env), build_edge_state(env))
        self.steps += 1
        if result["decision_agents"]:
            self.model_call_steps += 1
            self.decision_agent_count += len(result["decision_agents"])
        self.safety_interrupt_count += len(result["safety_interrupted_agents"])
        return np.asarray(result["actions"], dtype=np.int64)

    def episode_stats(self, n_agents):
        total_agent_steps = max(self.steps * n_agents, 1)
        return {
            "model_call_step_ratio": float(self.model_call_steps / float(max(self.steps, 1))),
            "decision_agent_ratio": float(self.decision_agent_count / float(total_agent_steps)),
            "safety_interrupt_count": float(self.safety_interrupt_count),
        }


def save_csv(path, rows):
    if not rows:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def mean_std(records, key, default=0.0):
    values = [float(record.get(key, default)) for record in records]
    return float(np.mean(values)), float(np.std(values))


def evaluate_policy(cfg, policy, episodes, seed):
    records = []
    start_time = time.perf_counter()
    for episode_idx in range(episodes):
        env = MECEnv(cfg)
        env.seed(seed + episode_idx)
        policy.reset_episode()
        env.reset()

        done = False
        reward_sum = 0.0
        delay_values = []
        drop_values = []
        offload_values = []
        energy_values = []

        while not done:
            actions = policy.select_actions(env)
            _, reward, done, info = env.step(actions)
            reward_sum += reward
            delay_values.append(info["delay_mean"])
            drop_values.append(info["drop_rate"])
            offload_values.append(info["offload_rate"])
            energy_values.append(info["energy_mean"])

        record = {
            "reward": float(reward_sum),
            "delay": float(np.mean(delay_values)),
            "drop_rate": float(np.mean(drop_values)),
            "offload_rate": float(np.mean(offload_values)),
            "energy": float(np.mean(energy_values)),
        }
        record.update(policy.episode_stats(cfg.M))
        records.append(record)

    elapsed_s = time.perf_counter() - start_time
    summary = {
        "policy": policy.name,
        "episodes": int(episodes),
        "elapsed_s": float(elapsed_s),
    }
    for key in [
        "reward",
        "delay",
        "drop_rate",
        "offload_rate",
        "energy",
        "model_call_step_ratio",
        "decision_agent_ratio",
        "safety_interrupt_count",
    ]:
        mean_value, std_value = mean_std(records, key)
        summary[key + "_mean"] = mean_value
        summary[key + "_std"] = std_value
    return summary


def compare_action_agreement(cfg, teacher, student_policy, episodes, seed, rollout_policy):
    agent_matches = 0
    agent_total = 0
    slot_matches = 0
    slot_total = 0
    decision_matches = 0
    decision_total = 0
    reused_matches = 0
    reused_total = 0
    student_model_call_steps = 0
    student_decision_agents = 0
    safety_interrupt_count = 0
    confusion = np.zeros((2, 2), dtype=np.int64)

    for episode_idx in range(episodes):
        env = MECEnv(cfg)
        env.seed(seed + episode_idx)
        env.reset()
        student_policy.reset()

        done = False
        while not done:
            obs = env.get_obs()
            teacher_actions, _ = teacher.predict_batch(obs)
            result = student_policy.select_actions(build_agent_states(env), build_edge_state(env))
            student_actions = np.asarray(result["actions"], dtype=np.int64)

            matches = teacher_actions == student_actions
            agent_matches += int(np.sum(matches))
            agent_total += int(cfg.M)
            slot_matches += int(np.all(matches))
            slot_total += 1

            decision_agents = list(result["decision_agents"])
            reused_agents = [idx for idx in range(cfg.M) if idx not in set(decision_agents)]
            if decision_agents:
                decision_matches += int(np.sum(matches[decision_agents]))
                decision_total += len(decision_agents)
                student_model_call_steps += 1
                student_decision_agents += len(decision_agents)
            if reused_agents:
                reused_matches += int(np.sum(matches[reused_agents]))
                reused_total += len(reused_agents)

            safety_interrupt_count += len(result["safety_interrupted_agents"])
            for teacher_action, student_action in zip(teacher_actions, student_actions):
                if 0 <= int(teacher_action) < 2 and 0 <= int(student_action) < 2:
                    confusion[int(teacher_action), int(student_action)] += 1

            if rollout_policy == "student":
                rollout_actions = student_actions
            elif rollout_policy == "teacher":
                rollout_actions = teacher_actions
            else:
                rollout_actions = teacher_actions
            _, _, done, _ = env.step(rollout_actions)

    total_steps = max(slot_total, 1)
    total_agent_steps = max(agent_total, 1)
    return {
        "episodes": int(episodes),
        "rollout_policy": rollout_policy,
        "steps": int(slot_total),
        "agent_comparisons": int(agent_total),
        "agent_action_agreement": float(agent_matches / float(total_agent_steps)),
        "slot_action_agreement": float(slot_matches / float(total_steps)),
        "decision_agent_action_agreement": float(decision_matches / float(max(decision_total, 1))),
        "reused_agent_action_agreement": float(reused_matches / float(max(reused_total, 1))),
        "decision_agent_comparisons": int(decision_total),
        "reused_agent_comparisons": int(reused_total),
        "student_model_call_step_ratio": float(student_model_call_steps / float(total_steps)),
        "student_decision_agent_ratio": float(student_decision_agents / float(total_agent_steps)),
        "student_safety_interrupt_count": int(safety_interrupt_count),
        "teacher_local_student_local": int(confusion[0, 0]),
        "teacher_local_student_offload": int(confusion[0, 1]),
        "teacher_offload_student_local": int(confusion[1, 0]),
        "teacher_offload_student_offload": int(confusion[1, 1]),
    }


def benchmark_actor(policy, obs, iterations, warmup):
    for _ in range(warmup):
        if isinstance(policy, TeacherPolicy):
            policy.predict_batch(obs)
        else:
            policy.predict_from_obs(obs)
    synchronize_device(policy)

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        if isinstance(policy, TeacherPolicy):
            policy.predict_batch(obs)
        else:
            policy.predict_from_obs(obs)
        synchronize_device(policy)
        samples.append((time.perf_counter() - start) * 1000.0)
    return summarize_latency(samples)


def benchmark_policy_step(cfg, teacher, student_policy, iterations, warmup):
    env = MECEnv(cfg)
    env.seed(777)
    env.reset()
    agents = build_agent_states(env)
    edge = build_edge_state(env)
    obs = env.get_obs()

    for _ in range(warmup):
        teacher.predict_batch(obs)
    synchronize_device(teacher)
    teacher_samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        teacher.predict_batch(obs)
        synchronize_device(teacher)
        teacher_samples.append((time.perf_counter() - start) * 1000.0)

    student_policy.reset()
    for _ in range(warmup):
        student_policy.select_actions(agents, edge)
    synchronize_device(student_policy)

    student_policy.reset()
    student_samples = []
    model_call_steps = 0
    decision_agent_count = 0
    for _ in range(iterations):
        start = time.perf_counter()
        result = student_policy.select_actions(agents, edge)
        synchronize_device(student_policy)
        student_samples.append((time.perf_counter() - start) * 1000.0)
        if result["decision_agents"]:
            model_call_steps += 1
            decision_agent_count += len(result["decision_agents"])

    total_agent_steps = max(iterations * cfg.M, 1)
    teacher_row = {
        "policy": "teacher_mappo_actor",
        "policy_step_model_call_step_ratio": 1.0,
        "policy_step_decision_agent_ratio": 1.0,
    }
    teacher_row.update({"policy_step_" + key: value for key, value in summarize_latency(teacher_samples).items()})

    student_row = {
        "policy": "student_temporal_distilled_mappo",
        "policy_step_model_call_step_ratio": float(model_call_steps / float(max(iterations, 1))),
        "policy_step_decision_agent_ratio": float(decision_agent_count / float(total_agent_steps)),
    }
    student_row.update({"policy_step_" + key: value for key, value in summarize_latency(student_samples).items()})
    return [teacher_row, student_row]


def print_closed_loop(rows):
    print("\nClosed-loop simulated evaluation")
    print("policy                         delay_mean  drop_mean  reward_mean  model_calls  decisions")
    print("-----------------------------  ----------  ---------  -----------  -----------  ---------")
    for row in rows:
        print(
            "{:<29}  {:>10.4f}  {:>9.4f}  {:>11.4f}  {:>11.4f}  {:>9.4f}".format(
                row["policy"],
                row["delay_mean"],
                row["drop_rate_mean"],
                row["reward_mean"],
                row["model_call_step_ratio_mean"],
                row["decision_agent_ratio_mean"],
            )
        )


def print_agreement(row):
    print("\nTeacher/student action agreement")
    print("agent agreement:        {:.4f}".format(row["agent_action_agreement"]))
    print("slot agreement:         {:.4f}".format(row["slot_action_agreement"]))
    print("student model calls:    {:.4f}".format(row["student_model_call_step_ratio"]))
    print("student decisions:      {:.4f}".format(row["student_decision_agent_ratio"]))
    print(
        "confusion [[T0/S0,T0/S1],[T1/S0,T1/S1]]: [[{},{}],[{},{}]]".format(
            row["teacher_local_student_local"],
            row["teacher_local_student_offload"],
            row["teacher_offload_student_local"],
            row["teacher_offload_student_offload"],
        )
    )


def flatten_latency(prefix, latency):
    return {prefix + "_" + key: value for key, value in latency.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare teacher and student policies on Jetson Nano.")
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--teacher-metadata-path", default=None)
    parser.add_argument("--teacher-runtime", default="torch", choices=["auto", "torch", "onnx"])
    parser.add_argument("--student-model-path", default=None)
    parser.add_argument("--student-metadata-path", default=None)
    parser.add_argument("--student-runtime", default="auto", choices=["auto", "torch", "onnx"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--onnx-provider", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--agreement-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--load-factor", type=float, default=1.0)
    parser.add_argument("--agreement-rollout", default="teacher", choices=["teacher", "student"])
    parser.add_argument("--benchmark-iters", type=int, default=500)
    parser.add_argument("--benchmark-warmup", type=int, default=50)
    parser.add_argument("--disable-safety", action="store_true")
    parser.add_argument("--save-dir", default="jetson_teacher_student_results")
    parser.add_argument("--skip-closed-loop", action="store_true")
    parser.add_argument("--skip-agreement", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.agreement_episodes <= 0:
        raise ValueError("--agreement-episodes must be positive.")
    if args.load_factor <= 0:
        raise ValueError("--load-factor must be positive.")
    if args.benchmark_iters <= 0:
        raise ValueError("--benchmark-iters must be positive.")
    if args.benchmark_warmup < 0:
        raise ValueError("--benchmark-warmup must be non-negative.")

    teacher_model_path = resolve_teacher_model_path(args.teacher_model_path, args.teacher_runtime)
    student_model_path = resolve_student_model_path(args.student_model_path, args.student_runtime)
    teacher_runtime = resolve_runtime(args.teacher_runtime, teacher_model_path)
    student_runtime = resolve_runtime(args.student_runtime, student_model_path)

    if not os.path.exists(teacher_model_path):
        raise FileNotFoundError("Teacher model not found: {}".format(teacher_model_path))
    if not os.path.exists(student_model_path):
        raise FileNotFoundError("Student model not found: {}".format(student_model_path))

    cfg = EnvConfig(load_factor=args.load_factor)
    teacher = TeacherPolicy(
        model_path=teacher_model_path,
        runtime=teacher_runtime,
        device=args.device,
        metadata_path=args.teacher_metadata_path,
        onnx_provider=args.onnx_provider,
    )
    student_policy = JetsonTemporalPolicy(
        model_path=student_model_path,
        runtime=student_runtime,
        device=args.device,
        metadata_path=args.student_metadata_path,
        onnx_provider=args.onnx_provider,
        safety_enabled=not args.disable_safety,
    )
    student_eval_adapter = StudentPolicyAdapter(
        model_path=student_model_path,
        runtime=student_runtime,
        device=args.device,
        metadata_path=args.student_metadata_path,
        onnx_provider=args.onnx_provider,
        safety_enabled=not args.disable_safety,
    )

    print("Teacher: {} ({}, {})".format(teacher_model_path, teacher_runtime, teacher.device))
    print("Student: {} ({}, {})".format(student_model_path, student_runtime, student_policy.device))
    print("Safety interrupt: {}".format("off" if args.disable_safety else "on"))

    os.makedirs(args.save_dir, exist_ok=True)
    output = {
        "teacher_model_path": teacher_model_path,
        "teacher_runtime": teacher_runtime,
        "student_model_path": student_model_path,
        "student_runtime": student_runtime,
        "load_factor": args.load_factor,
    }

    if not args.skip_agreement:
        agreement = compare_action_agreement(
            cfg=cfg,
            teacher=teacher,
            student_policy=student_policy,
            episodes=args.agreement_episodes,
            seed=args.seed,
            rollout_policy=args.agreement_rollout,
        )
        print_agreement(agreement)
        save_csv(os.path.join(args.save_dir, "teacher_student_agreement.csv"), [agreement])
        output["agreement"] = agreement

    if not args.skip_benchmark:
        rng = np.random.RandomState(args.seed)
        obs = rng.rand(cfg.M, cfg.obs_dim).astype(np.float32)
        teacher_actor_latency = benchmark_actor(teacher, obs, args.benchmark_iters, args.benchmark_warmup)
        student_actor_latency = benchmark_actor(student_policy, obs, args.benchmark_iters, args.benchmark_warmup)
        policy_step_rows = benchmark_policy_step(
            cfg=cfg,
            teacher=teacher,
            student_policy=student_policy,
            iterations=args.benchmark_iters,
            warmup=args.benchmark_warmup,
        )
        latency_rows = [
            {
                "policy": "teacher_mappo_actor",
                "runtime": teacher_runtime,
                "device": str(teacher.device),
                "parameters": int(teacher.parameters),
                **flatten_latency("actor_forward", teacher_actor_latency),
            },
            {
                "policy": "student_temporal_distilled_mappo",
                "runtime": student_runtime,
                "device": str(student_policy.device),
                "parameters": 0,
                **flatten_latency("actor_forward", student_actor_latency),
            },
        ]
        save_csv(os.path.join(args.save_dir, "teacher_student_latency.csv"), latency_rows)
        save_csv(os.path.join(args.save_dir, "teacher_student_policy_step_latency.csv"), policy_step_rows)
        output["latency"] = {
            "actor_forward": latency_rows,
            "policy_step": policy_step_rows,
        }
        print("\nActor-forward latency")
        for row in latency_rows:
            print(
                "{:<31} mean={:.6f} ms p95={:.6f} ms".format(
                    row["policy"],
                    row["actor_forward_mean_ms"],
                    row["actor_forward_p95_ms"],
                )
            )

    if not args.skip_closed_loop:
        closed_loop_rows = [
            evaluate_policy(cfg, teacher, args.episodes, args.seed),
            evaluate_policy(cfg, student_eval_adapter, args.episodes, args.seed),
        ]
        print_closed_loop(closed_loop_rows)
        save_csv(os.path.join(args.save_dir, "teacher_student_closed_loop.csv"), closed_loop_rows)
        output["closed_loop"] = closed_loop_rows

    save_json(os.path.join(args.save_dir, "teacher_student_summary.json"), output)
    print("\nSaved results under: {}".format(os.path.abspath(args.save_dir)))


if __name__ == "__main__":
    main()
