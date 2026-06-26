from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from temporal.config import EnvConfig
from temporal.env import MECEnv
from temporal.models import Actor, TemporalActor
from temporal.prune_mappo_actor import count_parameters


def actor_linear_layers(actor: Actor) -> List[nn.Linear]:
    return [module for module in actor.mlp.net if isinstance(module, nn.Linear)]


def temporal_hidden_layers(actor: TemporalActor) -> List[nn.Linear]:
    return [module for module in actor.feature_net if isinstance(module, nn.Linear)]


def copy_pruned_actor_to_temporal(pruned_actor: Actor, temporal_actor: TemporalActor) -> None:
    src_layers = actor_linear_layers(pruned_actor)
    dst_hidden_layers = temporal_hidden_layers(temporal_actor)
    if len(src_layers) != len(dst_hidden_layers) + 1:
        raise ValueError("Pruned actor and temporal actor architectures do not match.")

    with torch.no_grad():
        for src_layer, dst_layer in zip(src_layers[:-1], dst_hidden_layers):
            dst_layer.weight.copy_(src_layer.weight)
            dst_layer.bias.copy_(src_layer.bias)
        temporal_actor.action_head.weight.copy_(src_layers[-1].weight)
        temporal_actor.action_head.bias.copy_(src_layers[-1].bias)
        nn.init.zeros_(temporal_actor.repeat_head.weight)
        nn.init.constant_(temporal_actor.repeat_head.bias, 0.5)


def load_teacher_actor(cfg: EnvConfig, model_path: str, device: str) -> Actor:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find teacher MAPPO checkpoint: {model_path}")

    actor = Actor(
        obs_dim=cfg.obs_dim,
        n_actions=cfg.n_actions,
        hidden_dims=[128, 128],
    ).to(device)
    ckpt = torch.load(model_path, map_location=device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def load_student_temporal_actor(
    cfg: EnvConfig,
    model_path: str,
    device: str,
) -> Tuple[TemporalActor, Dict[str, Any]]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Cannot find pruned student actor: {model_path}\n"
            "Run temporal/prune_mappo_actor.py first, or pass --student-model-path."
        )

    ckpt = torch.load(model_path, map_location=device)
    hidden_dims = ckpt.get("actor_hidden_dims")
    if hidden_dims is None:
        raise ValueError("Student checkpoint must contain actor_hidden_dims.")

    pruned_actor = Actor(
        obs_dim=cfg.obs_dim,
        n_actions=cfg.n_actions,
        hidden_dims=hidden_dims,
    ).to(device)
    pruned_actor.load_state_dict(ckpt["actor"])
    pruned_actor.eval()

    temporal_actor = TemporalActor(
        obs_dim=cfg.obs_dim,
        n_actions=cfg.n_actions,
        hidden_dims=hidden_dims,
    ).to(device)
    copy_pruned_actor_to_temporal(pruned_actor, temporal_actor)
    temporal_actor.train()
    return temporal_actor, ckpt


def compute_repeat_targets(actions: np.ndarray) -> np.ndarray:
    """Return remaining same-action repeats for each time step and agent."""
    if actions.ndim != 2:
        raise ValueError("actions must have shape (T, M).")

    repeats = np.zeros_like(actions, dtype=np.int64)
    if actions.shape[0] == 0:
        return repeats

    for m in range(actions.shape[1]):
        for t in range(actions.shape[0] - 2, -1, -1):
            if actions[t, m] == actions[t + 1, m]:
                repeats[t, m] = repeats[t + 1, m] + 1
            else:
                repeats[t, m] = 0
    return repeats


def reduced_variant_mask(actions: np.ndarray) -> np.ndarray:
    mask = np.ones(actions.shape, dtype=bool)
    if actions.shape[0] <= 1:
        return mask
    mask[1:] = actions[1:] != actions[:-1]
    return mask


def collect_temporal_dataset(
    cfg: EnvConfig,
    teacher: Actor,
    collect_episodes: int,
    device: str,
    seed: int,
    trajectory_variant: str,
    max_repeat: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = MECEnv(cfg)
    env.seed(seed)
    obs_records: List[np.ndarray] = []
    logits_records: List[np.ndarray] = []
    repeat_records: List[np.ndarray] = []

    for episode in range(1, collect_episodes + 1):
        data = env.reset()
        obs = data["obs"]
        done = False

        episode_obs: List[np.ndarray] = []
        episode_logits: List[np.ndarray] = []
        episode_actions: List[np.ndarray] = []

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = teacher(obs_tensor)
                actions = torch.argmax(logits, dim=-1).cpu().numpy()

            episode_obs.append(obs.astype(np.float32, copy=True))
            episode_logits.append(logits.cpu().numpy().astype(np.float32, copy=False))
            episode_actions.append(actions.astype(np.int64, copy=True))

            out = env.step(actions)
            obs = out["obs"]
            done = out["done"]

        obs_arr = np.stack(episode_obs, axis=0)
        logits_arr = np.stack(episode_logits, axis=0)
        actions_arr = np.stack(episode_actions, axis=0)
        repeats_arr = np.minimum(compute_repeat_targets(actions_arr), max_repeat)

        if trajectory_variant == "reduced":
            mask = reduced_variant_mask(actions_arr)
            obs_records.append(obs_arr[mask])
            logits_records.append(logits_arr[mask])
            repeat_records.append(repeats_arr[mask])
        else:
            obs_records.append(obs_arr.reshape(-1, cfg.obs_dim))
            logits_records.append(logits_arr.reshape(-1, cfg.n_actions))
            repeat_records.append(repeats_arr.reshape(-1))

        if episode == 1 or episode % 10 == 0 or episode == collect_episodes:
            kept = int(obs_records[-1].shape[0])
            print(f"Collected teacher episode {episode}/{collect_episodes} | samples: {kept}")

    obs_array = np.concatenate(obs_records, axis=0)
    logits_array = np.concatenate(logits_records, axis=0)
    repeat_array = np.concatenate(repeat_records, axis=0).astype(np.float32)
    print(
        "Collected temporal dataset: "
        f"obs={obs_array.shape}, logits={logits_array.shape}, repeats={repeat_array.shape}"
    )
    print(
        f"Repeat target mean={float(np.mean(repeat_array)):.3f}, "
        f"max={float(np.max(repeat_array)):.1f}"
    )
    return obs_array, logits_array, repeat_array


def split_train_val(
    obs_array: np.ndarray,
    logits_array: np.ndarray,
    repeat_array: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0).")

    rng = np.random.default_rng(seed)
    indices = np.arange(obs_array.shape[0])
    rng.shuffle(indices)

    val_size = int(round(obs_array.shape[0] * val_ratio))
    if val_ratio > 0.0:
        val_size = max(1, val_size)
    val_size = min(val_size, obs_array.shape[0] - 1)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    def tensors(selected: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(obs_array[selected], dtype=torch.float32),
            torch.tensor(logits_array[selected], dtype=torch.float32),
            torch.tensor(repeat_array[selected], dtype=torch.float32),
        )

    return tensors(train_indices), tensors(val_indices)


def action_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def temporal_distillation_loss(
    student_logits: torch.Tensor,
    repeat_raw: torch.Tensor,
    teacher_logits: torch.Tensor,
    repeat_targets: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action_loss = action_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        temperature=args.temperature,
    )
    repeat_pred_scaled = F.relu(repeat_raw)
    repeat_target_scaled = repeat_targets / args.repeat_scale
    repeat_loss = F.mse_loss(repeat_pred_scaled, repeat_target_scaled)
    total_loss = action_loss + args.repeat_loss_weight * repeat_loss
    return total_loss, action_loss, repeat_loss


def evaluate_student_match(
    student: TemporalActor,
    dataset: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
) -> Dict[str, float]:
    obs, teacher_logits, repeat_targets = dataset
    if obs.numel() == 0:
        return {
            "loss": 0.0,
            "action_loss": 0.0,
            "repeat_loss": 0.0,
            "agreement": 0.0,
            "repeat_mae": 0.0,
            "repeat_exact": 0.0,
            "repeat_pred_mean": 0.0,
            "repeat_target_mean": 0.0,
        }

    loader = DataLoader(TensorDataset(obs, teacher_logits, repeat_targets), batch_size=args.batch_size)
    losses = []
    action_losses = []
    repeat_losses = []
    agreements = []
    repeat_maes = []
    repeat_exacts = []
    repeat_pred_means = []
    repeat_target_means = []

    student.eval()
    with torch.no_grad():
        for batch_obs, batch_teacher_logits, batch_repeats in loader:
            batch_obs = batch_obs.to(args.device)
            batch_teacher_logits = batch_teacher_logits.to(args.device)
            batch_repeats = batch_repeats.to(args.device)

            student_logits, repeat_raw = student(batch_obs)
            loss, action_loss, repeat_loss = temporal_distillation_loss(
                student_logits=student_logits,
                repeat_raw=repeat_raw,
                teacher_logits=batch_teacher_logits,
                repeat_targets=batch_repeats,
                args=args,
            )
            teacher_actions = torch.argmax(batch_teacher_logits, dim=-1)
            student_actions = torch.argmax(student_logits, dim=-1)
            pred_repeats = torch.round(F.relu(repeat_raw) * args.repeat_scale)
            pred_repeats = torch.clamp(pred_repeats, min=0, max=args.max_repeat)

            losses.append(float(loss.item()))
            action_losses.append(float(action_loss.item()))
            repeat_losses.append(float(repeat_loss.item()))
            agreements.append(float((teacher_actions == student_actions).float().mean().item()))
            repeat_maes.append(float(torch.abs(pred_repeats - batch_repeats).mean().item()))
            repeat_exacts.append(float((pred_repeats == batch_repeats).float().mean().item()))
            repeat_pred_means.append(float(pred_repeats.float().mean().item()))
            repeat_target_means.append(float(batch_repeats.float().mean().item()))

    student.train()
    return {
        "loss": float(np.mean(losses)),
        "action_loss": float(np.mean(action_losses)),
        "repeat_loss": float(np.mean(repeat_losses)),
        "agreement": float(np.mean(agreements)),
        "repeat_mae": float(np.mean(repeat_maes)),
        "repeat_exact": float(np.mean(repeat_exacts)),
        "repeat_pred_mean": float(np.mean(repeat_pred_means)),
        "repeat_target_mean": float(np.mean(repeat_target_means)),
    }


def train_student(
    student: TemporalActor,
    train_dataset: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_dataset: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
) -> List[Dict[str, float]]:
    train_obs, train_logits, train_repeats = train_dataset
    train_loader = DataLoader(
        TensorDataset(train_obs, train_logits, train_repeats),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    optimizer = Adam(student.parameters(), lr=args.lr)
    records: List[Dict[str, float]] = []

    student.train()
    for epoch in range(1, args.epochs + 1):
        batch_losses = []
        batch_action_losses = []
        batch_repeat_losses = []
        batch_agreements = []
        batch_repeat_maes = []

        for batch_obs, batch_teacher_logits, batch_repeats in train_loader:
            batch_obs = batch_obs.to(args.device)
            batch_teacher_logits = batch_teacher_logits.to(args.device)
            batch_repeats = batch_repeats.to(args.device)

            student_logits, repeat_raw = student(batch_obs)
            loss, action_loss, repeat_loss = temporal_distillation_loss(
                student_logits=student_logits,
                repeat_raw=repeat_raw,
                teacher_logits=batch_teacher_logits,
                repeat_targets=batch_repeats,
                args=args,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                teacher_actions = torch.argmax(batch_teacher_logits, dim=-1)
                student_actions = torch.argmax(student_logits, dim=-1)
                pred_repeats = torch.round(F.relu(repeat_raw) * args.repeat_scale)
                pred_repeats = torch.clamp(pred_repeats, min=0, max=args.max_repeat)
                repeat_mae = torch.abs(pred_repeats - batch_repeats).mean()

            batch_losses.append(float(loss.item()))
            batch_action_losses.append(float(action_loss.item()))
            batch_repeat_losses.append(float(repeat_loss.item()))
            batch_agreements.append(float((teacher_actions == student_actions).float().mean().item()))
            batch_repeat_maes.append(float(repeat_mae.item()))

        val_metrics = evaluate_student_match(student, val_dataset, args)
        record = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(batch_losses)),
            "train_action_loss": float(np.mean(batch_action_losses)),
            "train_repeat_loss": float(np.mean(batch_repeat_losses)),
            "train_agreement": float(np.mean(batch_agreements)),
            "train_repeat_mae": float(np.mean(batch_repeat_maes)),
            "val_loss": val_metrics["loss"],
            "val_action_loss": val_metrics["action_loss"],
            "val_repeat_loss": val_metrics["repeat_loss"],
            "val_agreement": val_metrics["agreement"],
            "val_repeat_mae": val_metrics["repeat_mae"],
            "val_repeat_exact": val_metrics["repeat_exact"],
            "val_repeat_pred_mean": val_metrics["repeat_pred_mean"],
            "val_repeat_target_mean": val_metrics["repeat_target_mean"],
        }
        records.append(record)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {record['train_loss']:.6f} | "
            f"ActLoss: {record['train_action_loss']:.6f} | "
            f"RepLoss: {record['train_repeat_loss']:.6f} | "
            f"Agree: {record['train_agreement']:.4f} | "
            f"RepMAE: {record['train_repeat_mae']:.3f} | "
            f"ValAgree: {record['val_agreement']:.4f} | "
            f"ValRepMAE: {record['val_repeat_mae']:.3f}"
        )

    return records


def save_training_log(save_path: str, records: List[Dict[str, float]]) -> str:
    log_path = os.path.splitext(save_path)[0] + "_log.csv"
    if not records:
        return log_path

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"Temporal distillation log saved to: {log_path}")
    return log_path


def measure_temporal_actor_latency(
    actor: TemporalActor,
    obs_dim: int,
    n_agents: int,
    device: str,
    repeats: int,
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
    parser = argparse.ArgumentParser(
        description="Temporal distillation for a pruned MAPPO actor."
    )
    parser.add_argument(
        "--teacher-model-path",
        default=os.path.join("results", "mappo_checkpoint.pt"),
        help="Path to the original MAPPO checkpoint used as teacher.",
    )
    parser.add_argument(
        "--student-model-path",
        default=os.path.join("temporal", "results", "mappo_actor_pruned_p25.pt"),
        help="Path to a pruned actor checkpoint used as student initialization.",
    )
    parser.add_argument(
        "--save-path",
        default=os.path.join("temporal", "results", "mappo_actor_temporal_distilled_p25.pt"),
        help="Path for the temporal-distilled actor checkpoint.",
    )
    parser.add_argument("--collect-episodes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--repeat-scale", type=float, default=5.0)
    parser.add_argument("--repeat-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-repeat", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--trajectory-variant", choices=["extended", "reduced"], default="extended")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latency-repeats", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.collect_episodes <= 0:
        raise ValueError("--collect-episodes must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if args.repeat_scale <= 0:
        raise ValueError("--repeat-scale must be positive.")
    if args.repeat_loss_weight < 0:
        raise ValueError("--repeat-loss-weight must be non-negative.")
    if args.max_repeat < 0:
        raise ValueError("--max-repeat must be non-negative.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = EnvConfig()
    teacher = load_teacher_actor(cfg, args.teacher_model_path, args.device)
    student, student_ckpt = load_student_temporal_actor(cfg, args.student_model_path, args.device)

    obs_array, logits_array, repeat_array = collect_temporal_dataset(
        cfg=cfg,
        teacher=teacher,
        collect_episodes=args.collect_episodes,
        device=args.device,
        seed=args.seed,
        trajectory_variant=args.trajectory_variant,
        max_repeat=args.max_repeat,
    )
    train_dataset, val_dataset = split_train_val(
        obs_array=obs_array,
        logits_array=logits_array,
        repeat_array=repeat_array,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Train samples: {len(train_dataset[0])}, validation samples: {len(val_dataset[0])}")

    pre_metrics = evaluate_student_match(student, val_dataset, args)
    print(
        "Before temporal distillation | "
        f"ValAgree: {pre_metrics['agreement']:.4f} | "
        f"ValRepMAE: {pre_metrics['repeat_mae']:.3f}"
    )

    records = train_student(student, train_dataset, val_dataset, args)
    post_metrics = evaluate_student_match(student, val_dataset, args)

    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    compression_ratio = teacher_params / max(student_params, 1)
    temporal_latency_ms = measure_temporal_actor_latency(
        actor=student,
        obs_dim=cfg.obs_dim,
        n_agents=cfg.M,
        device=args.device,
        repeats=args.latency_repeats,
    )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    log_path = save_training_log(args.save_path, records)
    torch.save(
        {
            **student_ckpt,
            "actor": student.state_dict(),
            "temporal_actor": True,
            "temporal_distilled": True,
            "actor_hidden_dims": student.hidden_dims,
            "teacher_model_path": args.teacher_model_path,
            "student_init_path": args.student_model_path,
            "collect_episodes": args.collect_episodes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "temperature": args.temperature,
            "repeat_scale": args.repeat_scale,
            "repeat_loss_weight": args.repeat_loss_weight,
            "max_repeat": args.max_repeat,
            "trajectory_variant": args.trajectory_variant,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "distillation_log_path": log_path,
            "pre_val_agreement": pre_metrics["agreement"],
            "pre_val_repeat_mae": pre_metrics["repeat_mae"],
            "post_val_agreement": post_metrics["agreement"],
            "post_val_repeat_mae": post_metrics["repeat_mae"],
            "post_val_repeat_exact": post_metrics["repeat_exact"],
            "original_params": teacher_params,
            "pruned_params": student_params,
            "compression_ratio": compression_ratio,
            "temporal_latency_ms": temporal_latency_ms,
        },
        args.save_path,
    )

    print("\n=========== Temporal Distillation Summary ===========\n")
    print(f"Teacher model: {args.teacher_model_path}")
    print(f"Student init: {args.student_model_path}")
    print(f"Temporal actor: {args.save_path}")
    print(f"Hidden dims: {student.hidden_dims}")
    print(f"Parameters: {teacher_params} -> {student_params}")
    print(f"Compression ratio: {compression_ratio:.3f}x")
    print(f"Temporal actor latency: {temporal_latency_ms:.6f} ms")
    print(f"Val agreement: {pre_metrics['agreement']:.4f} -> {post_metrics['agreement']:.4f}")
    print(f"Val repeat MAE: {pre_metrics['repeat_mae']:.3f} -> {post_metrics['repeat_mae']:.3f}")
    print(f"Val repeat exact: {post_metrics['repeat_exact']:.4f}")
    print("\n=====================================================\n")


if __name__ == "__main__":
    main()
