from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn

from config import EnvConfig
from models import Actor


def get_actor_linear_layers(actor: Actor) -> List[nn.Linear]:
    return [module for module in actor.mlp.net if isinstance(module, nn.Linear)]


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def compute_keep_indices(
    actor: Actor,
    pruning_percentage: float,
) -> Dict[int, List[int]]:
    if not 0.0 <= pruning_percentage < 1.0:
        raise ValueError("pruning_percentage must be in [0.0, 1.0).")

    linear_layers = get_actor_linear_layers(actor)
    hidden_layers = linear_layers[:-1]
    keep_indices: Dict[int, List[int]] = {}

    for layer_idx, layer in enumerate(hidden_layers):
        l1_norms = torch.norm(layer.weight.data, p=1, dim=1)
        num_keep = max(1, int(l1_norms.shape[0] * (1.0 - pruning_percentage)))
        _, indices = torch.topk(l1_norms, num_keep)
        indices = torch.sort(indices).values
        keep_indices[layer_idx] = indices.cpu().numpy().astype(int).tolist()
        print(
            f"hidden layer {layer_idx}: keeping {num_keep} "
            f"out of {l1_norms.shape[0]} neurons"
        )

    return keep_indices


def build_pruned_actor(
    cfg: EnvConfig,
    original_actor: Actor,
    keep_indices: Dict[int, List[int]],
) -> Actor:
    linear_layers = get_actor_linear_layers(original_actor)
    if len(linear_layers) != 3:
        raise ValueError(
            "This pruning script expects Actor hidden_dims=[h1, h2], "
            f"but found {len(linear_layers) - 1} hidden Linear layers."
        )

    keep_0 = keep_indices[0]
    keep_1 = keep_indices[1]
    pruned_actor = Actor(
        obs_dim=cfg.obs_dim,
        n_actions=cfg.n_actions,
        hidden_dims=[len(keep_0), len(keep_1)],
    )

    src_layers = get_actor_linear_layers(original_actor)
    dst_layers = get_actor_linear_layers(pruned_actor)

    with torch.no_grad():
        # First hidden layer: select output neurons, keep all input features.
        dst_layers[0].weight.copy_(src_layers[0].weight[keep_0, :])
        dst_layers[0].bias.copy_(src_layers[0].bias[keep_0])

        # Second hidden layer: select output neurons and the kept first-layer inputs.
        dst_layers[1].weight.copy_(src_layers[1].weight[keep_1, :][:, keep_0])
        dst_layers[1].bias.copy_(src_layers[1].bias[keep_1])

        # Output layer: keep all action logits, select kept second-layer inputs.
        dst_layers[2].weight.copy_(src_layers[2].weight[:, keep_1])
        dst_layers[2].bias.copy_(src_layers[2].bias)

    return pruned_actor


def load_original_actor(cfg: EnvConfig, model_path: str, device: str) -> Actor:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find MAPPO checkpoint: {model_path}")

    actor = Actor(
        obs_dim=cfg.obs_dim,
        n_actions=cfg.n_actions,
        hidden_dims=[128, 128],
    ).to(device)
    ckpt = torch.load(model_path, map_location=device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def measure_actor_latency(
    actor: Actor,
    obs_dim: int,
    n_agents: int,
    device: str,
    repeats: int = 2000,
    warmup: int = 200,
) -> float:
    actor.eval()
    obs = torch.randn(n_agents, obs_dim, dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = actor(obs)

        start = time.perf_counter()
        for _ in range(repeats):
            _ = actor(obs)
        elapsed = time.perf_counter() - start

    return elapsed * 1000.0 / repeats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neuron-prune a trained MAPPO actor.")
    parser.add_argument(
        "--model-path",
        default=os.path.join("results", "mappo_checkpoint.pt"),
        help="Path to the trained MAPPO checkpoint.",
    )
    parser.add_argument(
        "--save-path",
        default=os.path.join("results", "mappo_actor_pruned.pt"),
        help="Path for the pruned actor checkpoint.",
    )
    parser.add_argument(
        "--pruning-percentage",
        type=float,
        default=0.25,
        help="Fraction of hidden neurons removed from each hidden layer.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latency-repeats", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EnvConfig()

    original_actor = load_original_actor(cfg, args.model_path, args.device)
    keep_indices = compute_keep_indices(original_actor, args.pruning_percentage)
    pruned_actor = build_pruned_actor(cfg, original_actor, keep_indices).to(args.device)
    pruned_actor.eval()

    original_params = count_parameters(original_actor)
    pruned_params = count_parameters(pruned_actor)
    compression_ratio = original_params / max(pruned_params, 1)

    original_latency_ms = measure_actor_latency(
        original_actor,
        obs_dim=cfg.obs_dim,
        n_agents=cfg.M,
        device=args.device,
        repeats=args.latency_repeats,
    )
    pruned_latency_ms = measure_actor_latency(
        pruned_actor,
        obs_dim=cfg.obs_dim,
        n_agents=cfg.M,
        device=args.device,
        repeats=args.latency_repeats,
    )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(
        {
            "actor": pruned_actor.state_dict(),
            "actor_hidden_dims": [len(keep_indices[0]), len(keep_indices[1])],
            "obs_dim": cfg.obs_dim,
            "n_actions": cfg.n_actions,
            "n_agents": cfg.M,
            "source_model_path": args.model_path,
            "pruning_percentage": args.pruning_percentage,
            "keep_indices": keep_indices,
            "original_params": original_params,
            "pruned_params": pruned_params,
            "compression_ratio": compression_ratio,
            "original_latency_ms": original_latency_ms,
            "pruned_latency_ms": pruned_latency_ms,
        },
        args.save_path,
    )

    print("\n================ Pruning Summary ================\n")
    print(f"Source model: {args.model_path}")
    print(f"Pruned actor: {args.save_path}")
    print(f"Pruning percentage: {args.pruning_percentage:.2f}")
    print(f"Hidden dims: {[128, 128]} -> {[len(keep_indices[0]), len(keep_indices[1])]}")
    print(f"Parameters: {original_params} -> {pruned_params}")
    print(f"Compression ratio: {compression_ratio:.3f}x")
    print(f"Actor latency: {original_latency_ms:.6f} ms -> {pruned_latency_ms:.6f} ms")
    print("\n=================================================\n")


if __name__ == "__main__":
    main()

