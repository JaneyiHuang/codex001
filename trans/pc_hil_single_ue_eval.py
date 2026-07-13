#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PC-side single-UE hardware-in-the-loop evaluator.

Experiment meaning:
  - Jetson Nano is NOT the MEC edge server in this experiment.
  - Jetson Nano is a device-side inference node representing only one target UE
    among the 4 UEs used by training/evaluation.
  - The PC runs the complete MEC simulation environment, edge queue, the other
    3 virtual UEs, reward calculation, simulated delay, and drop calculation.
  - Nano only performs:
        target UE local observation -> ONNX student actor -> target UE action
  - This is single-device hardware-in-the-loop evaluation, not a real
    multi-physical-UE system.
  - Nano acting as a TCP server is only a communication implementation detail;
    it does not mean Nano is the MEC edge server.

The closed-loop state transition always uses the target UE action returned by
Nano. Teacher/student/random/local/offload policies on the PC are used only for
the other virtual UEs, except that a teacher model can also be used to report
target-UE action agreement.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import random
import socket
import sys
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import EnvConfig
from env import MECEnv


CSV_FIELDS = [
    "episode",
    "step",
    "target_agent_id",
    "nano_action",
    "teacher_action_target",
    "action_match",
    "repeat_pred",
    "reward",
    "delay",
    "drop",
    "offload_rate",
    "nano_infer_ms",
    "round_trip_ms",
    "other_policy",
    "joint_actions",
]


class JsonLineClient(object):
    """Python-socket JSON-line client."""

    def __init__(self, host, port, timeout=30.0):
        self.sock = socket.create_connection((host, int(port)), timeout=float(timeout))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buffer = b""

    def send_json(self, message):
        data = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        self.sock.sendall(data)

    def recv_json(self):
        while b"\n" not in self.buffer:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError("Nano TCP connection closed while waiting for a response.")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        line = line.strip()
        if not line:
            return {}
        return json.loads(line.decode("utf-8"))

    def request_action(self, episode, step, obs):
        obs_list = np.asarray(obs, dtype=np.float32).reshape(-1).tolist()
        request = {
            "type": "infer",
            "episode": int(episode),
            "step": int(step),
            "obs": obs_list,
        }
        start = time.perf_counter()
        self.send_json(request)
        response = self.recv_json()
        round_trip_ms = (time.perf_counter() - start) * 1000.0

        if response.get("type") != "action":
            raise RuntimeError("Nano returned non-action response: {}".format(response))
        return response, round_trip_ms

    def close(self):
        try:
            self.send_json({"type": "close"})
            try:
                self.recv_json()
            except Exception:
                pass
        finally:
            self.sock.close()


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metadata_path_for(model_path):
    root, ext = os.path.splitext(model_path)
    if ext.lower() == ".onnx":
        return root + ".onnx.meta.json"
    return model_path + ".meta.json"


def load_metadata(model_path):
    path = metadata_path_for(model_path)
    if os.path.exists(path):
        data = load_json_file(path)
        if isinstance(data, dict):
            return data
    return {}


def import_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("PyTorch is required to load a .pt actor: {}".format(exc))
    return torch, nn


def load_torch_checkpoint(path, device):
    torch, _ = import_torch()
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def is_state_dict(value):
    if not isinstance(value, dict) or not value:
        return False
    for item in value.values():
        if not hasattr(item, "shape"):
            return False
    return True


def normalize_state_keys(state):
    normalized = {}
    for key, value in state.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "actor.", "policy.", "model."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        normalized[new_key] = value
    return normalized


def extract_actor_state(ckpt):
    if is_state_dict(ckpt):
        return normalize_state_keys(ckpt)
    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported torch checkpoint type: {}".format(type(ckpt)))
    for key in ("actor", "actor_state_dict", "state_dict", "model_state_dict"):
        value = ckpt.get(key)
        if is_state_dict(value):
            return normalize_state_keys(value)
    raise ValueError("Could not find an actor state dict in checkpoint.")


def sorted_linear_weights(state, prefix):
    items = []
    for key, value in state.items():
        if key.startswith(prefix) and key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) < 3:
                continue
            try:
                layer_index = int(parts[1])
            except ValueError:
                continue
            items.append((layer_index, key, value))
    return sorted(items, key=lambda item: item[0])


def infer_standard_actor_dims(state):
    items = []
    for key, value in state.items():
        if key.startswith("mlp.net.") and key.endswith(".weight"):
            parts = key.split(".")
            try:
                layer_index = int(parts[2])
            except (IndexError, ValueError):
                continue
            items.append((layer_index, key, value))
    items = sorted(items, key=lambda item: item[0])
    if not items:
        raise ValueError("Could not infer standard Actor dimensions from state dict.")
    obs_dim = int(items[0][2].shape[1])
    n_actions = int(items[-1][2].shape[0])
    hidden_dims = [int(item[2].shape[0]) for item in items[:-1]]
    return obs_dim, n_actions, hidden_dims


def infer_temporal_actor_dims(state):
    items = sorted_linear_weights(state, "feature_net.")
    if not items or "action_head.weight" not in state:
        raise ValueError("Could not infer TemporalActor dimensions from state dict.")
    obs_dim = int(items[0][2].shape[1])
    hidden_dims = [int(item[2].shape[0]) for item in items]
    n_actions = int(state["action_head.weight"].shape[0])
    return obs_dim, n_actions, hidden_dims


def create_standard_actor(obs_dim, n_actions, hidden_dims):
    torch, nn = import_torch()

    class StandardActor(nn.Module):
        def __init__(self):
            super(StandardActor, self).__init__()
            layers = []
            last_dim = int(obs_dim)
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(last_dim, int(hidden_dim)))
                layers.append(nn.ReLU())
                last_dim = int(hidden_dim)
            layers.append(nn.Linear(last_dim, int(n_actions)))
            self.mlp = nn.Module()
            self.mlp.net = nn.Sequential(*layers)

        def forward(self, obs):
            return self.mlp.net(obs)

    return StandardActor()


def create_temporal_actor(obs_dim, n_actions, hidden_dims):
    torch, nn = import_torch()

    class TemporalActor(nn.Module):
        def __init__(self):
            super(TemporalActor, self).__init__()
            layers = []
            last_dim = int(obs_dim)
            for hidden_dim in hidden_dims:
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

    return TemporalActor()


class TorchActorPolicy(object):
    """Greedy PyTorch actor wrapper for teacher/student actions on the PC."""

    def __init__(self, model_path, obs_dim=None, n_actions=None, device="cpu"):
        self.model_path = model_path
        self.device = device
        self.torch, _ = import_torch()
        ckpt = load_torch_checkpoint(model_path, device)
        state = extract_actor_state(ckpt)
        metadata = load_metadata(model_path)

        is_temporal = (
            "action_head.weight" in state
            or bool(isinstance(ckpt, dict) and ckpt.get("temporal_actor", False))
        )
        if is_temporal:
            inferred_obs_dim, inferred_n_actions, inferred_hidden = infer_temporal_actor_dims(state)
            self.actor = create_temporal_actor(
                obs_dim=int(obs_dim or metadata.get("obs_dim", inferred_obs_dim)),
                n_actions=int(n_actions or metadata.get("n_actions", inferred_n_actions)),
                hidden_dims=list(metadata.get("actor_hidden_dims", inferred_hidden)),
            )
        else:
            inferred_obs_dim, inferred_n_actions, inferred_hidden = infer_standard_actor_dims(state)
            self.actor = create_standard_actor(
                obs_dim=int(obs_dim or metadata.get("obs_dim", inferred_obs_dim)),
                n_actions=int(n_actions or metadata.get("n_actions", inferred_n_actions)),
                hidden_dims=list(metadata.get("actor_hidden_dims", inferred_hidden)),
            )

        self.obs_dim = int(obs_dim or metadata.get("obs_dim", inferred_obs_dim))
        self.n_actions = int(n_actions or metadata.get("n_actions", inferred_n_actions))
        self.is_temporal = bool(is_temporal)
        self.actor.to(device)
        self.actor.load_state_dict(state, strict=True)
        self.actor.eval()

    def predict_batch(self, obs_batch):
        obs_batch = np.asarray(obs_batch, dtype=np.float32)
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(1, -1)
        if obs_batch.shape[1] != self.obs_dim:
            raise ValueError(
                "Expected obs_dim={} for {}, got {}".format(
                    self.obs_dim, self.model_path, obs_batch.shape
                )
            )
        obs_tensor = self.torch.tensor(obs_batch, dtype=self.torch.float32, device=self.device)
        with self.torch.no_grad():
            output = self.actor(obs_tensor)
            logits_tensor = output[0] if isinstance(output, (tuple, list)) else output
            actions = self.torch.argmax(logits_tensor, dim=-1)
        return actions.cpu().numpy().astype(np.int64)

    def predict_one(self, obs):
        return int(self.predict_batch(np.asarray(obs, dtype=np.float32).reshape(1, -1))[0])


class OnnxActorPolicy(object):
    """Greedy ONNX actor wrapper for teacher/student actions on the PC."""

    def __init__(self, model_path, obs_dim=None, n_actions=None):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime is required to load an ONNX actor: {}".format(exc))

        self.model_path = model_path
        metadata = load_metadata(model_path)
        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else available
        if not providers:
            raise RuntimeError("No ONNX Runtime provider is available.")
        self.session = ort.InferenceSession(model_path, providers=providers)
        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError("ONNX model has no inputs: {}".format(model_path))
        self.input_name = inputs[0].name
        input_shape = inputs[0].shape
        inferred_obs_dim = None
        if input_shape and isinstance(input_shape[-1], int):
            inferred_obs_dim = int(input_shape[-1])
        self.obs_dim = int(obs_dim or metadata.get("obs_dim", inferred_obs_dim or 10))
        self.n_actions = int(n_actions or metadata.get("n_actions", 2))

    def predict_batch(self, obs_batch):
        obs_batch = np.asarray(obs_batch, dtype=np.float32)
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(1, -1)
        if obs_batch.shape[1] != self.obs_dim:
            raise ValueError(
                "Expected obs_dim={} for {}, got {}".format(
                    self.obs_dim, self.model_path, obs_batch.shape
                )
            )
        outputs = self.session.run(None, {self.input_name: obs_batch})
        logits = np.asarray(outputs[0], dtype=np.float32)
        actions = np.argmax(logits, axis=-1).reshape(-1).astype(np.int64)
        return actions

    def predict_one(self, obs):
        return int(self.predict_batch(np.asarray(obs, dtype=np.float32).reshape(1, -1))[0])


def build_actor_policy(model_path, obs_dim, n_actions, device="cpu"):
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".onnx":
        return OnnxActorPolicy(model_path, obs_dim=obs_dim, n_actions=n_actions)
    return TorchActorPolicy(model_path, obs_dim=obs_dim, n_actions=n_actions, device=device)


def resolve_model_path(user_path, candidates, description, required=False):
    if user_path:
        path = os.path.abspath(user_path)
        if not os.path.exists(path):
            raise FileNotFoundError("{} not found: {}".format(description, path))
        return path
    for candidate in candidates:
        path = os.path.abspath(os.path.join(PROJECT_ROOT, candidate))
        if os.path.exists(path):
            return path
    if required:
        raise FileNotFoundError(
            "Could not find {}. Tried: {}".format(
                description, ", ".join(os.path.join(PROJECT_ROOT, c) for c in candidates)
            )
        )
    return None


def make_env(num_users, steps, obs_dim, n_actions, seed):
    cfg = EnvConfig()
    cfg.M = int(num_users)
    cfg.episode_limit = int(steps)
    cfg.obs_dim = int(obs_dim)
    cfg.n_actions = int(n_actions)
    cfg.state_dim = 8 * int(cfg.M) + 2
    env = MECEnv(cfg)
    if hasattr(env, "seed"):
        env.seed(int(seed))
    return env


def extract_obs(container):
    if isinstance(container, dict):
        for key in ("obs", "observation", "observations"):
            if key in container:
                return np.asarray(container[key], dtype=np.float32)
        raise KeyError("Could not find obs in environment result keys: {}".format(list(container.keys())))
    return np.asarray(container, dtype=np.float32)


def parse_reset_result(result):
    if isinstance(result, tuple):
        result = result[0]
    return extract_obs(result)


def done_to_bool(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return bool(arr.item())
    return bool(np.any(arr))


def parse_step_result(result):
    if isinstance(result, dict):
        obs = extract_obs(result)
        reward = float(result.get("reward", 0.0))
        done = done_to_bool(result.get("done", False))
        info = result.get("info", {})
        return obs, reward, done, info

    if not isinstance(result, tuple):
        raise ValueError("Unsupported env.step return type: {}".format(type(result)))
    if len(result) == 4:
        obs, reward, done, info = result
        return extract_obs(obs), float(reward), done_to_bool(done), info
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = done_to_bool(terminated) or done_to_bool(truncated)
        return extract_obs(obs), float(reward), done, info
    raise ValueError("Unsupported env.step tuple length: {}".format(len(result)))


def ensure_obs_shape(obs_all, num_users, obs_dim):
    obs_all = np.asarray(obs_all, dtype=np.float32)
    if obs_all.shape != (int(num_users), int(obs_dim)):
        raise ValueError(
            "obs_all should have shape [{}, {}], got {}".format(
                num_users, obs_dim, obs_all.shape
            )
        )
    return obs_all


def get_numeric_metric(info, aliases, num_users=None):
    if not isinstance(info, dict):
        return None
    for key in aliases:
        if key not in info:
            continue
        value = info[key]
        arr = np.asarray(value)
        if arr.size == 0:
            continue
        metric = float(np.mean(arr.astype(np.float64)))
        lower_key = key.lower()
        if num_users is not None and (
            "count" in lower_key or lower_key.endswith("_num") or lower_key == "drop_num"
        ):
            metric = metric / float(num_users)
        return metric
    return None


def extract_metrics(info, joint_actions, num_users):
    # Simulated MEC delay/drop/offload_rate are produced by env.step().
    # They are not mixed with Nano inference time or TCP round-trip time.
    delay = get_numeric_metric(
        info,
        ["delay", "delay_mean", "avg_delay", "mean_delay", "delay_avg", "delay_arr"],
        num_users=None,
    )
    drop = get_numeric_metric(
        info,
        ["drop_rate", "drop", "avg_drop", "mean_drop", "drop_count", "drop_num", "drop_flags"],
        num_users=num_users,
    )
    offload_rate = get_numeric_metric(
        info,
        ["offload_rate", "avg_offload_rate", "mean_offload_rate", "offload_ratio"],
        num_users=None,
    )
    if offload_rate is None:
        offload_rate = float(np.mean(np.asarray(joint_actions, dtype=np.int64) == 1))
    return delay, drop, offload_rate


def checked_action(action, n_actions, source):
    action = int(action)
    if action < 0 or action >= int(n_actions):
        raise ValueError("{} produced invalid action {} for n_actions={}".format(source, action, n_actions))
    return action


def make_other_actions(other_policy, obs_all, target_agent_id, n_actions, rng, actor_policy):
    num_users = int(obs_all.shape[0])
    actions = [0 for _ in range(num_users)]

    if other_policy == "random":
        for idx in range(num_users):
            if idx != target_agent_id:
                actions[idx] = int(rng.randint(0, int(n_actions)))
        return actions

    if other_policy == "local":
        return actions

    if other_policy == "offload":
        return [1 if idx != target_agent_id else 0 for idx in range(num_users)]

    if other_policy in ("teacher", "student"):
        if actor_policy is None:
            raise RuntimeError("{} policy requested but no actor policy was loaded.".format(other_policy))
        predicted = actor_policy.predict_batch(obs_all)
        for idx in range(num_users):
            if idx != target_agent_id:
                actions[idx] = checked_action(predicted[idx], n_actions, other_policy)
        return actions

    raise ValueError("Unsupported other_policy: {}".format(other_policy))


def mean_or_none(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def write_csv(path, rows):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, data):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PC-side single-target-UE hardware-in-the-loop evaluation."
    )
    parser.add_argument("--nano_ip", required=True, help="Jetson Nano TCP server IP.")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--target_agent_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--other_policy",
        choices=["teacher", "student", "random", "local", "offload"],
        default="teacher",
    )
    parser.add_argument("--teacher_model_path", default=None)
    parser.add_argument("--student_model_path_pc", default=None)
    parser.add_argument("--obs_dim", type=int, default=10)
    parser.add_argument("--n_actions", type=int, default=2)
    parser.add_argument("--num_users", type=int, default=4)
    parser.add_argument("--output_dir", default=os.path.join("results", "hil_single_ue_onnx"))
    parser.add_argument("--device", default="cpu", help="PC torch device for .pt teacher/student actors.")
    parser.add_argument("--socket_timeout", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()

    if int(args.num_users) != 4:
        print("Warning: this HIL design defaults to num_users=4; using --num_users={}.".format(args.num_users))
    if args.target_agent_id < 0 or args.target_agent_id >= args.num_users:
        raise ValueError("target_agent_id must be in [0, num_users).")

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import_torch()[0].manual_seed(args.seed)
    except Exception:
        pass

    teacher_candidates = [
        os.path.join("export", "mappo_teacher_actor.pt"),
        os.path.join("results", "mappo_checkpoint.pt"),
        os.path.join("export", "mappo_teacher_actor.onnx"),
        os.path.join("export", "mappo_teacher_actor_opset11.onnx"),
    ]
    student_candidates = [
        os.path.join("temporal", "results", "mappo_actor_temporal_distilled_p25.pt"),
        os.path.join("results", "mappo_actor_pruned_distilled_p25.pt"),
        os.path.join("export", "mappo_actor_temporal_distilled_p25.onnx"),
    ]

    teacher_path = None
    teacher_policy = None
    if args.teacher_model_path or args.other_policy == "teacher":
        teacher_path = resolve_model_path(
            args.teacher_model_path,
            teacher_candidates,
            "teacher model",
            required=(args.other_policy == "teacher"),
        )
        if teacher_path:
            teacher_policy = build_actor_policy(
                teacher_path,
                obs_dim=args.obs_dim,
                n_actions=args.n_actions,
                device=args.device,
            )
            print("Loaded teacher actor: {}".format(teacher_path))

    student_path = None
    student_policy = None
    if args.other_policy == "student":
        student_path = resolve_model_path(
            args.student_model_path_pc,
            student_candidates,
            "student model for PC-side virtual UEs",
            required=True,
        )
        student_policy = build_actor_policy(
            student_path,
            obs_dim=args.obs_dim,
            n_actions=args.n_actions,
            device=args.device,
        )
        print("Loaded PC-side student actor: {}".format(student_path))

    other_actor_policy = teacher_policy if args.other_policy == "teacher" else student_policy

    env = make_env(
        num_users=args.num_users,
        steps=args.steps,
        obs_dim=args.obs_dim,
        n_actions=args.n_actions,
        seed=args.seed,
    )
    rng = np.random.RandomState(args.seed)
    client = JsonLineClient(args.nano_ip, args.port, timeout=args.socket_timeout)

    rows = []
    rewards = []
    delays = []
    drops = []
    offloads = []
    nano_infer_times = []
    round_trip_times = []
    action_matches = []

    print("Connected to Nano at {}:{}.".format(args.nano_ip, args.port))
    print(
        "Starting HIL evaluation: episodes={}, steps={}, num_users={}, target_agent_id={}, other_policy={}".format(
            args.episodes,
            args.steps,
            args.num_users,
            args.target_agent_id,
            args.other_policy,
        )
    )

    try:
        for episode in range(int(args.episodes)):
            reset_result = env.reset()
            obs_all = ensure_obs_shape(parse_reset_result(reset_result), args.num_users, args.obs_dim)

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

                nano_response, round_trip_ms = client.request_action(episode, step, target_obs)
                nano_action = checked_action(nano_response.get("action"), args.n_actions, "Nano")
                nano_infer_ms = float(nano_response.get("infer_ms", 0.0))
                repeat_pred = nano_response.get("repeat_pred")

                if teacher_action_target is not None:
                    action_match = int(int(teacher_action_target) == int(nano_action))
                    action_matches.append(action_match)

                # First version intentionally does not use temporal repeat/skip.
                # The PC sends the current target-UE observation to Nano every slot.
                other_actions = make_other_actions(
                    args.other_policy,
                    obs_all,
                    int(args.target_agent_id),
                    args.n_actions,
                    rng,
                    other_actor_policy,
                )
                joint_actions = list(other_actions)
                joint_actions[int(args.target_agent_id)] = int(nano_action)

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
                nano_infer_times.append(nano_infer_ms)
                round_trip_times.append(round_trip_ms)

                rows.append(
                    {
                        "episode": int(episode),
                        "step": int(step),
                        "target_agent_id": int(args.target_agent_id),
                        "nano_action": int(nano_action),
                        "teacher_action_target": teacher_action_target,
                        "action_match": action_match,
                        "repeat_pred": repeat_pred,
                        "reward": float(reward),
                        "delay": delay,
                        "drop": drop,
                        "offload_rate": offload_rate,
                        "nano_infer_ms": float(nano_infer_ms),
                        "round_trip_ms": float(round_trip_ms),
                        "other_policy": args.other_policy,
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
    csv_path = os.path.join(output_dir, "hil_single_ue_steps.csv")
    summary_path = os.path.join(output_dir, "hil_single_ue_summary.json")

    agreement = None
    if action_matches:
        agreement = float(np.mean(np.asarray(action_matches, dtype=np.float32)))

    summary = {
        "episodes": int(args.episodes),
        "steps_per_episode": int(args.steps),
        "num_users": int(args.num_users),
        "target_agent_id": int(args.target_agent_id),
        "other_policy": args.other_policy,
        "average_reward": mean_or_none(rewards),
        "average_delay": mean_or_none(delays),
        "average_drop_rate": mean_or_none(drops),
        "average_offload_rate": mean_or_none(offloads),
        "average_nano_infer_ms": mean_or_none(nano_infer_times),
        "average_round_trip_ms": mean_or_none(round_trip_times),
        "target_action_agreement": agreement,
        "total_steps": int(total_steps),
        "teacher_model_path": teacher_path,
        "student_model_path_pc": student_path,
        "csv_path": csv_path,
        "summary_path": summary_path,
        "note": (
            "reward/delay/drop/offload_rate are simulated MEC metrics from PC env.step(); "
            "nano_infer_ms is pure Nano ONNX inference time; round_trip_ms is PC-Nano TCP "
            "communication latency and is not added to simulated MEC delay."
        ),
    }

    write_csv(csv_path, rows)
    write_json(summary_path, summary)

    print("Saved step CSV: {}".format(csv_path))
    print("Saved summary JSON: {}".format(summary_path))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
