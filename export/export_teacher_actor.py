#!/usr/bin/env python3
"""
Export the original MAPPO teacher as an actor-only deployment artifact.

Outputs by default:
  - export/mappo_teacher_actor.pt
  - export/mappo_teacher_actor.pt.meta.json
  - export/mappo_teacher_actor.onnx
  - export/mappo_teacher_actor.onnx.meta.json
"""

from __future__ import print_function

import argparse
import json
import os

import torch
import torch.nn as nn


MODEL_BASENAME = "mappo_teacher_actor"
DEFAULT_HIDDEN_DIMS = [128, 128]


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


class Actor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dims=None):
        super(Actor, self).__init__()
        if hidden_dims is None:
            hidden_dims = DEFAULT_HIDDEN_DIMS
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


def default_source_path():
    return os.path.join(project_root(), "results", "mappo_checkpoint.pt")


def default_pt_path():
    return os.path.join(script_dir(), MODEL_BASENAME + ".pt")


def default_onnx_path():
    return os.path.join(script_dir(), MODEL_BASENAME + ".onnx")


def metadata_path_for(path):
    return path + ".meta.json"


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


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
        raise ValueError("Could not infer actor dimensions from state dict.")

    obs_dim = int(weight_items[0][2].shape[1])
    n_actions = int(weight_items[-1][2].shape[0])
    hidden_dims = [int(item[2].shape[0]) for item in weight_items[:-1]]
    return obs_dim, n_actions, hidden_dims


def count_state_params(state_dict):
    return int(sum(value.numel() for value in state_dict.values()))


def build_metadata(source_path, artifact_path, actor_state, obs_dim, n_actions, hidden_dims, artifact_format):
    return {
        "format": artifact_format,
        "model_name": MODEL_BASENAME,
        "source_checkpoint": os.path.abspath(source_path),
        "artifact": os.path.abspath(artifact_path),
        "teacher_actor_only": True,
        "input_name": "obs",
        "output_names": ["action_logits"],
        "n_agents": 4,
        "obs_dim": int(obs_dim),
        "n_actions": int(n_actions),
        "actor_hidden_dims": [int(value) for value in hidden_dims],
        "parameters": count_state_params(actor_state),
        "normalization": {
            "task_max_bits": 6000000.0,
            "e_max": 5.0,
            "h_max": 0.5,
            "q_loc_max": 10000000.0,
            "q_tx_max": 10000000.0,
            "q_edge_max": 40000000.0,
            "episode_limit": 200.0,
            "h_norm_clip": 10.0,
        },
    }


def save_json(path, data):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def export_pt(source_path, output_path, actor_state, obs_dim, n_actions, hidden_dims):
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)

    checkpoint = {
        "format": "mappo_teacher_actor_only",
        "model_name": MODEL_BASENAME,
        "actor": actor_state,
        "teacher_actor_only": True,
        "actor_hidden_dims": [int(value) for value in hidden_dims],
        "obs_dim": int(obs_dim),
        "n_actions": int(n_actions),
        "n_agents": 4,
        "source_checkpoint": os.path.abspath(source_path),
        "parameters": count_state_params(actor_state),
    }
    torch.save(checkpoint, output_path)
    save_json(
        metadata_path_for(output_path),
        build_metadata(
            source_path=source_path,
            artifact_path=output_path,
            actor_state=actor_state,
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dims=hidden_dims,
            artifact_format="mappo_teacher_actor_pt_metadata",
        ),
    )


def export_onnx(source_path, output_path, actor_state, obs_dim, n_actions, hidden_dims, batch_size, opset):
    actor = Actor(obs_dim=obs_dim, n_actions=n_actions, hidden_dims=hidden_dims)
    actor.load_state_dict(actor_state)
    actor.eval()

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)

    dummy_obs = torch.zeros((int(batch_size), int(obs_dim)), dtype=torch.float32)
    torch.onnx.export(
        actor,
        dummy_obs,
        output_path,
        export_params=True,
        opset_version=int(opset),
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["action_logits"],
        dynamic_axes={
            "obs": {0: "batch"},
            "action_logits": {0: "batch"},
        },
    )
    save_json(
        metadata_path_for(output_path),
        build_metadata(
            source_path=source_path,
            artifact_path=output_path,
            actor_state=actor_state,
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dims=hidden_dims,
            artifact_format="mappo_teacher_actor_onnx_metadata",
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Export original MAPPO teacher actor only.")
    parser.add_argument("--source", default=default_source_path(), help="Path to results/mappo_checkpoint.pt.")
    parser.add_argument("--output", default=default_pt_path(), help="Output actor-only .pt path.")
    parser.add_argument("--onnx-output", default=default_onnx_path(), help="Output actor-only ONNX path.")
    parser.add_argument("--skip-onnx", action="store_true", help="Only write the actor-only .pt artifact.")
    parser.add_argument("--batch-size", type=int, default=4, help="Dummy ONNX export batch size.")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version.")
    return parser.parse_args()


def main():
    args = parse_args()
    source_path = os.path.abspath(args.source)
    output_path = os.path.abspath(args.output)
    onnx_output_path = os.path.abspath(args.onnx_output)

    if not os.path.exists(source_path):
        raise FileNotFoundError("Teacher checkpoint not found: {}".format(source_path))

    checkpoint = load_checkpoint(source_path, "cpu")
    if not isinstance(checkpoint, dict) or "actor" not in checkpoint:
        raise ValueError("Expected a teacher checkpoint dict containing an actor state dict.")

    actor_state = checkpoint["actor"]
    obs_dim, n_actions, hidden_dims = infer_actor_dims(actor_state)

    export_pt(
        source_path=source_path,
        output_path=output_path,
        actor_state=actor_state,
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dims=hidden_dims,
    )

    onnx_written = None
    if not args.skip_onnx:
        export_onnx(
            source_path=source_path,
            output_path=onnx_output_path,
            actor_state=actor_state,
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dims=hidden_dims,
            batch_size=args.batch_size,
            opset=args.opset,
        )
        onnx_written = onnx_output_path

    summary = {
        "source_checkpoint": source_path,
        "actor_pt": output_path,
        "actor_pt_metadata": metadata_path_for(output_path),
        "actor_onnx": onnx_written,
        "actor_onnx_metadata": metadata_path_for(onnx_output_path) if onnx_written else None,
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "actor_hidden_dims": hidden_dims,
        "parameters": count_state_params(actor_state),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
