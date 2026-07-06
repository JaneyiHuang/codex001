#!/usr/bin/env python3
"""
Jetson Nano inference helper for mappo_actor_temporal_distilled_p25.

This script does not generate a random environment for deployment. It converts
real MEC system measurements into the normalized observation format used during
training, then runs the temporal-distilled MAPPO actor.

Typical PyTorch smoke test on Jetson:
    python3 export/jetson_infer.py --demo

Typical ONNX Runtime smoke test on Jetson:
    python3 export/jetson_infer.py --runtime onnx --demo

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


MODEL_BASENAME = "mappo_actor_temporal_distilled_p25"
TORCH_MODEL_FILENAME = MODEL_BASENAME + ".pt"
ONNX_MODEL_FILENAME = MODEL_BASENAME + ".onnx"
ONNX_METADATA_FILENAME = MODEL_BASENAME + ".onnx.meta.json"

# Backward-compatible name used by earlier deployment notes.
MODEL_FILENAME = TORCH_MODEL_FILENAME


def load_checkpoint(model_path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


def load_json_file(path):
    with open(path, "r") as f:
        return json.load(f)


def resolve_metadata_path(model_path, metadata_path=None):
    if metadata_path:
        return metadata_path
    root, ext = os.path.splitext(model_path)
    if ext.lower() == ".onnx":
        return root + ".onnx.meta.json"
    return model_path + ".meta.json"


def load_onnx_metadata(model_path, metadata_path=None):
    path = resolve_metadata_path(model_path, metadata_path)
    if path and os.path.exists(path):
        data = load_json_file(path)
        if not isinstance(data, dict):
            raise ValueError("ONNX metadata must be a JSON object: {}".format(path))
        return data
    return {}


def temporal_outputs_from_logits(action_logits, repeat_raw, repeat_scale, max_repeat):
    action_logits = np.asarray(action_logits, dtype=np.float32)
    repeat_raw = np.asarray(repeat_raw, dtype=np.float32)
    actions = np.argmax(action_logits, axis=-1).astype(np.int64)
    repeats = np.rint(np.maximum(repeat_raw, 0.0) * float(repeat_scale)).astype(np.int64)
    repeats = np.clip(repeats, 0, int(max_repeat)).astype(np.int64)
    return actions, repeats


def import_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "Could not import onnxruntime: {}. Install a working onnxruntime "
            "package on Jetson before using --runtime onnx, or use the default "
            "PyTorch runtime.".format(exc)
        )
    return ort


def select_onnx_providers(device="auto", provider_mode="auto"):
    ort = import_onnxruntime()

    available = ort.get_available_providers()
    providers = []

    def add_provider(name):
        if name in available and name not in providers:
            providers.append(name)

    provider_mode = str(provider_mode or "auto").lower()
    device = str(device or "auto").lower()

    if provider_mode == "cpu":
        add_provider("CPUExecutionProvider")
    elif provider_mode == "cuda":
        add_provider("CUDAExecutionProvider")
        add_provider("CPUExecutionProvider")
        if "CUDAExecutionProvider" not in providers:
            raise ValueError(
                "CUDAExecutionProvider is not available in this onnxruntime build. "
                "Available providers: {}".format(available)
            )
    elif provider_mode == "auto":
        if device != "cpu":
            add_provider("CUDAExecutionProvider")
        add_provider("CPUExecutionProvider")
    else:
        raise ValueError("Unsupported ONNX provider mode: {}".format(provider_mode))

    if not providers:
        raise RuntimeError("No usable ONNX Runtime provider found. Available providers: {}".format(available))
    return providers


def create_onnx_session(model_path, device="auto", provider_mode="auto"):
    ort = import_onnxruntime()
    providers = select_onnx_providers(device=device, provider_mode=provider_mode)
    session = ort.InferenceSession(model_path, providers=providers)
    inputs = session.get_inputs()
    if not inputs:
        raise ValueError("ONNX model has no inputs: {}".format(model_path))

    input_name = inputs[0].name
    input_shape = inputs[0].shape
    inferred_obs_dim = None
    if input_shape and isinstance(input_shape[-1], int):
        inferred_obs_dim = int(input_shape[-1])
    return session, input_name, session.get_providers(), inferred_obs_dim


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
        runtime="torch",
        metadata_path=None,
        onnx_provider="auto",
        normalization=None,
        repeat_scale=None,
        max_repeat=None,
        safety_enabled=True,
        safe_energy_min=0.12,
        safe_queue_threshold=0.80,
        safe_obs_change_threshold=0.35,
        safe_channel_drop_threshold=0.35,
    ):
        self.runtime = str(runtime or "torch").lower()
        self.model_path = model_path
        self.actor = None
        self.onnx_session = None
        self.onnx_input_name = None
        self.onnx_providers = []

        if self.runtime == "torch":
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
        elif self.runtime == "onnx":
            metadata = load_onnx_metadata(model_path, metadata_path)
            (
                self.onnx_session,
                self.onnx_input_name,
                self.onnx_providers,
                inferred_obs_dim,
            ) = create_onnx_session(model_path, device=device, provider_mode=onnx_provider)

            metadata_obs_dim = metadata.get("obs_dim", inferred_obs_dim if inferred_obs_dim is not None else 10)
            normalization_data = metadata.get("normalization", {})
            if not isinstance(normalization_data, dict):
                normalization_data = {}
            self.normalization = normalization or NormalizationConfig(
                n_agents=metadata.get("n_agents", 4),
                obs_dim=metadata_obs_dim,
                task_max_bits=normalization_data.get("task_max_bits", 6e6),
                e_max=normalization_data.get("e_max", 5.0),
                h_max=normalization_data.get("h_max", 0.5),
                q_loc_max=normalization_data.get("q_loc_max", 1e7),
                q_tx_max=normalization_data.get("q_tx_max", 1e7),
                q_edge_max=normalization_data.get("q_edge_max", 4e7),
                episode_limit=normalization_data.get("episode_limit", 200),
                h_norm_clip=normalization_data.get("h_norm_clip", 10.0),
            )
            self.n_agents = int(metadata.get("n_agents", self.normalization.n_agents))
            self.obs_dim = int(metadata.get("obs_dim", self.normalization.obs_dim))
            if inferred_obs_dim is not None and self.obs_dim != inferred_obs_dim:
                raise ValueError(
                    "ONNX metadata obs_dim={} does not match model input obs_dim={}".format(
                        self.obs_dim,
                        inferred_obs_dim,
                    )
                )
            self.n_actions = int(metadata.get("n_actions", 2))
            self.repeat_scale = float(
                repeat_scale if repeat_scale is not None else metadata.get("repeat_scale", 5.0)
            )
            self.max_repeat = int(max_repeat if max_repeat is not None else metadata.get("max_repeat", 5))
            self.device = "onnxruntime:{}".format(",".join(self.onnx_providers))
        else:
            raise ValueError("Unsupported runtime: {}".format(runtime))

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

        if self.runtime == "onnx":
            ort_outputs = self.onnx_session.run(None, {self.onnx_input_name: obs})
            if len(ort_outputs) < 2:
                raise ValueError("Expected ONNX outputs: action_logits, repeat_raw")
            return temporal_outputs_from_logits(
                ort_outputs[0],
                ort_outputs[1],
                repeat_scale=self.repeat_scale,
                max_repeat=self.max_repeat,
            )

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


def resolve_model_path(model_path, runtime="torch"):
    if model_path:
        return model_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if runtime == "onnx":
        filename = ONNX_MODEL_FILENAME
        candidates = [
            os.path.join(os.getcwd(), "export", filename),
            os.path.join(script_dir, filename),
            os.path.join(os.getcwd(), "temporal", "results", filename),
            os.path.join(script_dir, "..", "temporal", "results", filename),
        ]
    else:
        filename = TORCH_MODEL_FILENAME
        candidates = [
            os.path.join(os.getcwd(), "temporal", "results", filename),
            os.path.join(script_dir, "..", "temporal", "results", filename),
            os.path.join(script_dir, filename),
        ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return candidates[0]


def resolve_runtime(runtime, model_path):
    runtime = str(runtime or "torch").lower()
    if runtime != "auto":
        return runtime
    if model_path and os.path.splitext(model_path)[1].lower() == ".onnx":
        return "onnx"
    return "torch"


def resolve_onnx_metadata_for_model(model_path, metadata_path):
    if metadata_path:
        return metadata_path
    candidate = resolve_metadata_path(model_path)
    if os.path.exists(candidate):
        return candidate
    return None


def load_state_json(path):
    data = load_json_file(path)
    if "agents" not in data or "edge" not in data:
        raise ValueError("JSON must contain top-level keys: agents, edge")
    return data["agents"], data["edge"]


def synchronize_device(policy):
    if getattr(policy, "runtime", "torch") == "torch" and str(policy.device).startswith("cuda") and torch.cuda.is_available():
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
    parser.add_argument("--model-path", default=None, help="Path to the .pt checkpoint or exported .onnx model.")
    parser.add_argument("--metadata-path", default=None, help="Optional ONNX metadata JSON path.")
    parser.add_argument("--runtime", default="torch", choices=["auto", "torch", "onnx"], help="Inference runtime.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument(
        "--onnx-provider",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="ONNX Runtime execution provider preference.",
    )
    parser.add_argument("--obs-json", default=None, help="JSON file containing real agent and edge states.")
    parser.add_argument("--demo", action="store_true", help="Run one smoke-test step using fixed example values.")
    parser.add_argument("--disable-safety", action="store_true", help="Disable temporal safety interrupts.")
    parser.add_argument("--benchmark", action="store_true", help="Measure actor and temporal-policy latency.")
    parser.add_argument("--benchmark-iters", type=int, default=1000, help="Benchmark iterations.")
    parser.add_argument("--benchmark-warmup", type=int, default=100, help="Benchmark warmup iterations.")
    return parser.parse_args()


def main():
    args = parse_args()
    runtime = resolve_runtime(args.runtime, args.model_path)
    model_path = resolve_model_path(args.model_path, runtime=runtime)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found: {}".format(model_path))

    if args.demo:
        agents, edge = demo_state()
    elif args.obs_json:
        agents, edge = load_state_json(args.obs_json)
    else:
        raise SystemExit("Use --demo for a smoke test, or --obs-json with real system state.")

    policy = JetsonTemporalPolicy(
        model_path=model_path,
        device=args.device,
        runtime=runtime,
        metadata_path=resolve_onnx_metadata_for_model(model_path, args.metadata_path),
        onnx_provider=args.onnx_provider,
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
        "runtime": policy.runtime,
        "device": policy.device,
        "action_meaning": {"0": "local_compute", "1": "offload_to_edge"},
        "result": result,
    }
    if policy.runtime == "onnx":
        output["onnx_providers"] = policy.onnx_providers
    if benchmark is not None:
        output["benchmark"] = benchmark

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
