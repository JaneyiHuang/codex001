#!/usr/bin/env python3
"""
Export the temporal-distilled MAPPO actor checkpoint to ONNX.

Example:
    python3 export/export_temporal_actor_onnx.py --verify
"""

from __future__ import print_function

import argparse
import json
import os

import numpy as np
import torch

from jetson_infer import (
    MODEL_BASENAME,
    ONNX_MODEL_FILENAME,
    NormalizationConfig,
    TemporalActor,
    load_checkpoint,
    resolve_model_path,
    temporal_outputs_from_logits,
)


def default_output_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, ONNX_MODEL_FILENAME)


def default_metadata_path(onnx_path):
    root, _ = os.path.splitext(onnx_path)
    return root + ".onnx.meta.json"


def build_actor_from_checkpoint(ckpt):
    if not isinstance(ckpt, dict):
        raise ValueError("Expected a checkpoint dict, got: {}".format(type(ckpt)))
    if not ckpt.get("temporal_actor", False):
        raise ValueError("Checkpoint is not a temporal actor checkpoint.")
    if "actor" not in ckpt:
        raise ValueError("Checkpoint is missing actor weights.")

    hidden_dims = ckpt.get("actor_hidden_dims")
    if hidden_dims is None:
        raise ValueError("Checkpoint is missing actor_hidden_dims.")

    obs_dim = int(ckpt.get("obs_dim", 10))
    n_actions = int(ckpt.get("n_actions", 2))
    actor = TemporalActor(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dims=hidden_dims,
    )
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def metadata_from_checkpoint(ckpt, model_path, onnx_path):
    n_agents = int(ckpt.get("n_agents", 4))
    obs_dim = int(ckpt.get("obs_dim", 10))
    cfg = NormalizationConfig(n_agents=n_agents, obs_dim=obs_dim)
    return {
        "format": "temporal_mappo_actor_onnx_metadata",
        "model_name": MODEL_BASENAME,
        "source_checkpoint": os.path.abspath(model_path),
        "onnx_model": os.path.abspath(onnx_path),
        "input_name": "obs",
        "output_names": ["action_logits", "repeat_raw"],
        "n_agents": n_agents,
        "obs_dim": obs_dim,
        "n_actions": int(ckpt.get("n_actions", 2)),
        "actor_hidden_dims": list(ckpt.get("actor_hidden_dims", [])),
        "repeat_scale": float(ckpt.get("repeat_scale", 5.0)),
        "max_repeat": int(ckpt.get("max_repeat", 5)),
        "normalization": {
            "task_max_bits": cfg.task_max_bits,
            "e_max": cfg.e_max,
            "h_max": cfg.h_max,
            "q_loc_max": cfg.q_loc_max,
            "q_tx_max": cfg.q_tx_max,
            "q_edge_max": cfg.q_edge_max,
            "episode_limit": cfg.episode_limit,
            "h_norm_clip": cfg.h_norm_clip,
        },
    }


def export_onnx(actor, output_path, obs_dim, opset, batch_size, dynamic_batch):
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)

    dummy_obs = torch.zeros((int(batch_size), int(obs_dim)), dtype=torch.float32)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "obs": {0: "batch"},
            "action_logits": {0: "batch"},
            "repeat_raw": {0: "batch"},
        }

    torch.onnx.export(
        actor,
        dummy_obs,
        output_path,
        export_params=True,
        opset_version=int(opset),
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["action_logits", "repeat_raw"],
        dynamic_axes=dynamic_axes,
    )


def verify_onnx(actor, onnx_path, obs_dim, repeat_scale, max_repeat, batch_size):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("Could not import onnxruntime for --verify: {}".format(exc))

    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if "CPUExecutionProvider" not in available:
        providers = available

    rng = np.random.RandomState(7)
    obs = rng.rand(int(batch_size), int(obs_dim)).astype(np.float32)

    with torch.no_grad():
        torch_logits, torch_repeat_raw = actor(torch.from_numpy(obs))
    torch_logits = torch_logits.cpu().numpy()
    torch_repeat_raw = torch_repeat_raw.cpu().numpy()
    torch_actions, torch_repeats = temporal_outputs_from_logits(
        torch_logits,
        torch_repeat_raw,
        repeat_scale=repeat_scale,
        max_repeat=max_repeat,
    )

    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    onnx_logits, onnx_repeat_raw = session.run(None, {input_name: obs})
    onnx_actions, onnx_repeats = temporal_outputs_from_logits(
        onnx_logits,
        onnx_repeat_raw,
        repeat_scale=repeat_scale,
        max_repeat=max_repeat,
    )

    logits_diff = float(np.max(np.abs(torch_logits - onnx_logits)))
    repeat_diff = float(np.max(np.abs(torch_repeat_raw - onnx_repeat_raw)))
    actions_equal = bool(np.array_equal(torch_actions, onnx_actions))
    repeats_equal = bool(np.array_equal(torch_repeats, onnx_repeats))
    return {
        "providers": providers,
        "logits_max_abs_diff": logits_diff,
        "repeat_raw_max_abs_diff": repeat_diff,
        "actions_equal": actions_equal,
        "repeats_equal": repeats_equal,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Export temporal MAPPO actor checkpoint to ONNX.")
    parser.add_argument("--model-path", default=None, help="Path to mappo_actor_temporal_distilled_p25.pt.")
    parser.add_argument("--output", default=None, help="Output ONNX path.")
    parser.add_argument("--metadata-output", default=None, help="Output metadata JSON path.")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version.")
    parser.add_argument("--batch-size", type=int, default=4, help="Dummy/export verification batch size.")
    parser.add_argument("--static-batch", action="store_true", help="Disable dynamic batch axis in ONNX.")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX output against PyTorch with ONNX Runtime.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model_path, runtime="torch")
    output_path = os.path.abspath(args.output or default_output_path())
    metadata_path = os.path.abspath(args.metadata_output or default_metadata_path(output_path))

    if not os.path.exists(model_path):
        raise FileNotFoundError("Checkpoint not found: {}".format(model_path))

    ckpt = load_checkpoint(model_path, "cpu")
    actor = build_actor_from_checkpoint(ckpt)
    obs_dim = int(ckpt.get("obs_dim", 10))

    export_onnx(
        actor=actor,
        output_path=output_path,
        obs_dim=obs_dim,
        opset=args.opset,
        batch_size=args.batch_size,
        dynamic_batch=not args.static_batch,
    )

    metadata = metadata_from_checkpoint(ckpt, model_path, output_path)
    metadata_parent = os.path.dirname(metadata_path)
    if metadata_parent and not os.path.exists(metadata_parent):
        os.makedirs(metadata_parent)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    summary = {
        "checkpoint": os.path.abspath(model_path),
        "onnx": output_path,
        "metadata": metadata_path,
        "opset": int(args.opset),
        "dynamic_batch": bool(not args.static_batch),
        "obs_dim": obs_dim,
        "n_actions": int(ckpt.get("n_actions", 2)),
    }

    if args.verify:
        summary["verification"] = verify_onnx(
            actor=actor,
            onnx_path=output_path,
            obs_dim=obs_dim,
            repeat_scale=float(ckpt.get("repeat_scale", 5.0)),
            max_repeat=int(ckpt.get("max_repeat", 5)),
            batch_size=args.batch_size,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
