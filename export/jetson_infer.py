#!/usr/bin/env python3
"""
Jetson Nano inference helper for mappo_actor_temporal_distilled_p25.pt.

This script does not generate a random environment for deployment. It converts
real MEC system measurements into the normalized observation format used during
training, then runs the temporal-distilled MAPPO actor.

Typical smoke test on Jetson:
    python3 export/jetson_infer.py --demo

Typical one-step JSON input:
    python3 export/jetson_infer.py --obs-json current_state.json
"""

from __future__ import print_function

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_FILENAME = "mappo_actor_temporal_distilled_p25.pt"


def load_checkpoint(model_path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


class NormalizationConfig(object):
    """Constants must match the training config.py values."""

    def __init__(
        self,
        n_agents=4,
        obs_dim=10,
        task_max_bits=6e6,
        e_max=5.0,
        h_max=0.5,
        q_loc_max=1e7,
        q_tx_max=1e7,
        q_edge_max=4e7,
        episode_limit=200,
        h_norm_clip=10.0,
    ):
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.task_max_bits = float(task_max_bits)
        self.e_max = float(e_max)
        self.h_max = float(h_max)
        self.q_loc_max = float(q_loc_max)
        self.q_tx_max = float(q_tx_max)
        self.q_edge_max = float(q_edge_max)
        self.episode_limit = float(episode_limit)
        self.h_norm_clip = float(h_norm_clip)


class TemporalActor(nn.Module):
    """Small self-contained copy of temporal.models.TemporalActor."""

    def __init__(self, obs_dim, n_actions, hidden_dims):
        super(TemporalActor, self).__init__()
        self.hidden_dims = list(hidden_dims)

        layers = []
        last_dim = int(obs_dim)
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            last_dim = int(hidden_dim)

        self.feature_net = nn.Sequential(*layers)
        self.action_head = nn.Linear(last_dim, int(n_actions))
        self.repeat_head = nn.Linear(last_dim, 1)

    def forward(self, obs):
        features = self.feature_net(obs)
        action_logits = self.action_head(features)
        repeat_raw = self.repeat_head(features).squeeze(-1)
        return action_logits, repeat_raw

    def predict_temporal(self, obs, repeat_scale, max_repeat):
        logits, repeat_raw = self.forward(obs)
        actions = torch.argmax(logits, dim=-1)
        repeats = torch.round(F.relu(repeat_raw) * float(repeat_scale)).to(torch.int64)
        repeats = torch.clamp(repeats, min=0, max=int(max_repeat))
        return actions, repeats


class JetsonTemporalPolicy(object):
    """
    Deployment wrapper.

    The input to select_actions is real system state, not a simulated random
    environment. Each agent state must contain:
        task_bits, energy, harvested_energy, channel_gain,
        q_loc, q_tx, f_loc, f_tx

    Edge state must contain:
        q_edge, f_edge
    """

    def __init__(
        self,
        model_path,
        device="auto",
        normalization=None,
        repeat_scale=None,
        max_repeat=None,
        safety_enabled=True,
        safe_energy_min=0.12,
        safe_queue_threshold=0.80,
        safe_obs_change_threshold=0.35,
        safe_channel_drop_threshold=0.35,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        ckpt = load_checkpoint(model_path, self.device)
        if not isinstance(ckpt, dict):
            raise ValueError("Expected a checkpoint dict, got: {}".format(type(ckpt)))
        if not ckpt.get("temporal_actor", False):
            raise ValueError("Checkpoint is not a temporal actor checkpoint.")

        hidden_dims = ckpt.get("actor_hidden_dims")
        if hidden_dims is None:
            raise ValueError("Checkpoint is missing actor_hidden_dims.")

        self.normalization = normalization or NormalizationConfig(
            n_agents=ckpt.get("n_agents", 4),
            obs_dim=ckpt.get("obs_dim", 10),
        )
        self.n_agents = int(ckpt.get("n_agents", self.normalization.n_agents))
        self.obs_dim = int(ckpt.get("obs_dim", self.normalization.obs_dim))
        self.n_actions = int(ckpt.get("n_actions", 2))

        self.actor = TemporalActor(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()

        self.repeat_scale = float(repeat_scale if repeat_scale is not None else ckpt.get("repeat_scale", 5.0))
        self.max_repeat = int(max_repeat if max_repeat is not None else ckpt.get("max_repeat", 5))

        self.safety_enabled = bool(safety_enabled)
        self.safe_energy_min = float(safe_energy_min)
        self.safe_queue_threshold = float(safe_queue_threshold)
        self.safe_obs_change_threshold = float(safe_obs_change_threshold)
        self.safe_channel_drop_threshold = float(safe_channel_drop_threshold)

        self.reset()

    def reset(self):
        self.cached_actions = np.zeros(self.n_agents, dtype=np.int64)
        self.remaining_repeats = np.zeros(self.n_agents, dtype=np.int64)
        self.last_decision_obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        self.has_decision_obs = np.zeros(self.n_agents, dtype=np.bool_)

    def build_obs(self, agent_states, edge_state):
        if len(agent_states) != self.n_agents:
            raise ValueError("Expected {} agent states, got {}".format(self.n_agents, len(agent_states)))

        cfg = self.normalization
        q_edge = float(edge_state["q_edge"])
        f_edge = float(edge_state["f_edge"])

        obs_rows = []
        for state in agent_states:
            row = np.array(
                [
                    float(state["task_bits"]) / cfg.task_max_bits,
                    float(state["energy"]) / cfg.e_max,
                    float(state["harvested_energy"]) / cfg.h_max,
                    min(float(state["channel_gain"]), cfg.h_norm_clip) / cfg.h_norm_clip,
                    float(state["q_loc"]) / cfg.q_loc_max,
                    float(state["q_tx"]) / cfg.q_tx_max,
                    min(float(state["f_loc"]), cfg.episode_limit) / cfg.episode_limit,
                    min(float(state["f_tx"]), cfg.episode_limit) / cfg.episode_limit,
                    q_edge / cfg.q_edge_max,
                    min(f_edge, cfg.episode_limit) / cfg.episode_limit,
                ],
                dtype=np.float32,
            )
            obs_rows.append(row)

        obs = np.stack(obs_rows, axis=0).astype(np.float32)
        return np.clip(obs, 0.0, 1.0)

    def predict_from_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.n_agents, self.obs_dim):
            raise ValueError("Expected obs shape ({}, {}), got {}".format(self.n_agents, self.obs_dim, obs.shape))
        return self.predict_batch(obs)

    def predict_batch(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim:
            raise ValueError("Expected obs batch shape (N, {}), got {}".format(self.obs_dim, obs.shape))
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions, repeats = self.actor.predict_temporal(
                obs_tensor,
                repeat_scale=self.repeat_scale,
                max_repeat=self.max_repeat,
            )
        return actions.cpu().numpy().astype(np.int64), repeats.cpu().numpy().astype(np.int64)

    def select_actions(self, agent_states, edge_state):
        obs = self.build_obs(agent_states, edge_state)

        actions = np.zeros(self.n_agents, dtype=np.int64)
        decision_agents = []
        interrupted_agents = []

        for agent_idx in range(self.n_agents):
            can_reuse = self.remaining_repeats[agent_idx] > 0
            interrupted = can_reuse and self._safety_interrupt(obs, agent_idx)
            if can_reuse and not interrupted:
                actions[agent_idx] = self.cached_actions[agent_idx]
                self.remaining_repeats[agent_idx] -= 1
            else:
                if interrupted:
                    interrupted_agents.append(agent_idx)
                    self.remaining_repeats[agent_idx] = 0
                decision_agents.append(agent_idx)

        predicted_repeats = np.zeros(self.n_agents, dtype=np.int64)
        if decision_agents:
            batch_obs = obs[decision_agents]
            batch_actions, batch_repeats = self.predict_batch(batch_obs)

            for local_idx, agent_idx in enumerate(decision_agents):
                action = int(batch_actions[local_idx])
                repeat = int(batch_repeats[local_idx])
                actions[agent_idx] = action
                predicted_repeats[agent_idx] = repeat
                self.cached_actions[agent_idx] = action
                self.remaining_repeats[agent_idx] = repeat
                self.last_decision_obs[agent_idx] = obs[agent_idx]
                self.has_decision_obs[agent_idx] = True

        return {
            "actions": actions.tolist(),
            "repeats": predicted_repeats.tolist(),
            "remaining_repeats": self.remaining_repeats.tolist(),
            "decision_agents": decision_agents,
            "safety_interrupted_agents": interrupted_agents,
            "obs": obs.tolist(),
        }

    def _safety_interrupt(self, obs, agent_idx):
        if not self.safety_enabled or not self.has_decision_obs[agent_idx]:
            return False

        current = obs[agent_idx]
        previous = self.last_decision_obs[agent_idx]

        energy_low = current[1] <= self.safe_energy_min
        queue_pressure = max(current[4], current[5], current[8]) >= self.safe_queue_threshold

        critical_indices = np.array([1, 4, 5, 6, 7, 8, 9], dtype=np.int64)
        critical_change = float(np.max(np.abs(current[critical_indices] - previous[critical_indices])))
        obs_changed = critical_change >= self.safe_obs_change_threshold

        channel_drop = False
        if self.cached_actions[agent_idx] == 1:
            channel_drop = (previous[3] - current[3]) >= self.safe_channel_drop_threshold

        return bool(energy_low or queue_pressure or obs_changed or channel_drop)


def resolve_model_path(model_path):
    if model_path:
        return model_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.getcwd(), "temporal", "results", MODEL_FILENAME),
        os.path.join(script_dir, "..", "temporal", "results", MODEL_FILENAME),
        os.path.join(script_dir, MODEL_FILENAME),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return candidates[0]


def load_state_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    if "agents" not in data or "edge" not in data:
        raise ValueError("JSON must contain top-level keys: agents, edge")
    return data["agents"], data["edge"]


def synchronize_device(policy):
    if str(policy.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize_latency(samples_ms):
    samples = np.asarray(samples_ms, dtype=np.float64)
    return {
        "iterations": int(samples.size),
        "mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
    }


def run_benchmark(policy, agents, edge, iterations, warmup):
    obs = policy.build_obs(agents, edge)

    for _ in range(warmup):
        policy.predict_from_obs(obs)
    synchronize_device(policy)

    actor_samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        policy.predict_from_obs(obs)
        synchronize_device(policy)
        actor_samples.append((time.perf_counter() - start) * 1000.0)

    policy.reset()
    for _ in range(warmup):
        policy.select_actions(agents, edge)
    synchronize_device(policy)

    policy.reset()
    policy_samples = []
    model_call_steps = 0
    decision_agent_count = 0
    for _ in range(iterations):
        start = time.perf_counter()
        result = policy.select_actions(agents, edge)
        synchronize_device(policy)
        policy_samples.append((time.perf_counter() - start) * 1000.0)

        if result["decision_agents"]:
            model_call_steps += 1
            decision_agent_count += len(result["decision_agents"])

    total_agent_steps = int(iterations * policy.n_agents)
    return {
        "actor_forward_full_batch": summarize_latency(actor_samples),
        "temporal_policy_step": summarize_latency(policy_samples),
        "temporal_policy_stats": {
            "steps": int(iterations),
            "total_agent_steps": total_agent_steps,
            "model_call_steps": int(model_call_steps),
            "model_call_step_ratio": float(model_call_steps / float(iterations)),
            "decision_agent_count": int(decision_agent_count),
            "decision_agent_ratio": float(decision_agent_count / float(total_agent_steps)),
        },
        "note": (
            "This measures Jetson-side inference cost only. Long-term average "
            "delay must be evaluated with an environment or real closed-loop system."
        ),
    }


def demo_state():
    agents = [
        {
            "task_bits": 2.5e6,
            "energy": 2.5,
            "harvested_energy": 0.2,
            "channel_gain": 1.0,
            "q_loc": 1.0e6,
            "q_tx": 0.5e6,
            "f_loc": 0.0,
            "f_tx": 0.0,
        },
        {
            "task_bits": 3.0e6,
            "energy": 2.0,
            "harvested_energy": 0.3,
            "channel_gain": 2.0,
            "q_loc": 2.0e6,
            "q_tx": 0.8e6,
            "f_loc": 3.0,
            "f_tx": 1.0,
        },
        {
            "task_bits": 4.0e6,
            "energy": 3.0,
            "harvested_energy": 0.2,
            "channel_gain": 0.5,
            "q_loc": 1.5e6,
            "q_tx": 1.2e6,
            "f_loc": 2.0,
            "f_tx": 4.0,
        },
        {
            "task_bits": 1.8e6,
            "energy": 1.8,
            "harvested_energy": 0.4,
            "channel_gain": 1.5,
            "q_loc": 0.5e6,
            "q_tx": 0.2e6,
            "f_loc": 0.0,
            "f_tx": 0.0,
        },
    ]
    edge = {"q_edge": 2.0e6, "f_edge": 1.0}
    return agents, edge


def parse_args():
    parser = argparse.ArgumentParser(description="Run temporal MAPPO actor inference on Jetson Nano.")
    parser.add_argument("--model-path", default=None, help="Path to mappo_actor_temporal_distilled_p25.pt.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument("--obs-json", default=None, help="JSON file containing real agent and edge states.")
    parser.add_argument("--demo", action="store_true", help="Run one smoke-test step using fixed example values.")
    parser.add_argument("--disable-safety", action="store_true", help="Disable temporal safety interrupts.")
    parser.add_argument("--benchmark", action="store_true", help="Measure actor and temporal-policy latency.")
    parser.add_argument("--benchmark-iters", type=int, default=1000, help="Benchmark iterations.")
    parser.add_argument("--benchmark-warmup", type=int, default=100, help="Benchmark warmup iterations.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model checkpoint not found: {}".format(model_path))

    if args.demo:
        agents, edge = demo_state()
    elif args.obs_json:
        agents, edge = load_state_json(args.obs_json)
    else:
        raise SystemExit("Use --demo for a smoke test, or --obs-json with real system state.")

    policy = JetsonTemporalPolicy(
        model_path=model_path,
        device=args.device,
        safety_enabled=not args.disable_safety,
    )
    result = policy.select_actions(agents, edge)
    benchmark = None
    if args.benchmark:
        benchmark = run_benchmark(
            policy=policy,
            agents=agents,
            edge=edge,
            iterations=args.benchmark_iters,
            warmup=args.benchmark_warmup,
        )

    output = {
        "model_path": model_path,
        "device": policy.device,
        "action_meaning": {"0": "local_compute", "1": "offload_to_edge"},
        "result": result,
    }
    if benchmark is not None:
        output["benchmark"] = benchmark

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
