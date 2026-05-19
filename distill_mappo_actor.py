from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from config import EnvConfig
from env import MECEnv
from models import Actor
from prune_mappo_actor import count_parameters, measure_actor_latency


def load_teacher_actor(cfg: EnvConfig, model_path: str, device: str) -> Actor:# 加载教师模型
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find teacher MAPPO checkpoint: {model_path}")

    actor = Actor(
        obs_dim=cfg.obs_dim,
        n_actions=cfg.n_actions,
        hidden_dims=[128, 128],
    ).to(device)
    ckpt = torch.load(model_path, map_location=device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()#设为eval模式，不训练
    return actor


def load_student_actor(cfg: EnvConfig, model_path: str, device: str) -> Tuple[Actor, Dict[str, Any]]:# 加载学生模型
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Cannot find pruned student actor: {model_path}\n"
            "Please run prune_mappo_actor.py first."
        )

    ckpt = torch.load(model_path, map_location=device)
    hidden_dims = ckpt.get("actor_hidden_dims")
    if hidden_dims is None:
        raise ValueError(
            "Student checkpoint must contain actor_hidden_dims. "
            "Please generate it with prune_mappo_actor.py."
        )

    actor = Actor(
        obs_dim=cfg.obs_dim,
        n_actions=cfg.n_actions,
        hidden_dims=hidden_dims,
    ).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.train()# 设置为训练模式，准备学习教师网络
    return actor, ckpt


def collect_teacher_observations(#用教师模型跑环境，收集观测数据，用来训练学生网络
    cfg: EnvConfig,
    teacher: Actor,
    collect_episodes: int,
    device: str,
    seed: int,
) -> np.ndarray:
    env = MECEnv(cfg)
    env.seed(seed)
    observations: List[np.ndarray] = []

    for episode in range(1, collect_episodes + 1):
        data = env.reset()
        obs = data["obs"]
        done = False

        while not done:
            observations.append(obs.astype(np.float32, copy=True))
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = teacher(obs_tensor)# 教师做出的决策
                actions = torch.argmax(logits, dim=-1).cpu().numpy()# 教师选出的动作

            out = env.step(actions)
            obs = out["obs"]
            done = out["done"]

        if episode == 1 or episode % 10 == 0 or episode == collect_episodes:
            print(f"Collected teacher episode {episode}/{collect_episodes}")

    obs_array = np.concatenate(observations, axis=0)
    print(f"Collected {obs_array.shape[0]} per-agent observations for distillation.")
    return obs_array


def split_train_val(#把数据分成训练集和验证集（打乱数据，90%用来训练学生，10%来测学生学的好不好）
    obs_array: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    train_obs = torch.tensor(obs_array[train_indices], dtype=torch.float32)
    val_obs = torch.tensor(obs_array[val_indices], dtype=torch.float32)
    return train_obs, val_obs


def distillation_loss(# 蒸馏损失，学生网络拟合教师网络的输出
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def evaluate_student_match(# 评估学生网络的输出，统计学生网络和教师网络的动作一致率
    teacher: Actor,
    student: Actor,
    obs: torch.Tensor,
    batch_size: int,
    temperature: float,
    device: str,
) -> Dict[str, float]:
    if obs.numel() == 0:
        return {"loss": 0.0, "agreement": 0.0}

    loader = DataLoader(TensorDataset(obs), batch_size=batch_size, shuffle=False)
    losses = []
    agreements = []

    teacher.eval()
    student.eval()
    with torch.no_grad():
        for (batch_obs,) in loader:
            batch_obs = batch_obs.to(device)
            teacher_logits = teacher(batch_obs)
            student_logits = student(batch_obs)
            loss = distillation_loss(student_logits, teacher_logits, temperature)
            teacher_actions = torch.argmax(teacher_logits, dim=-1)
            student_actions = torch.argmax(student_logits, dim=-1)
            agreement = (teacher_actions == student_actions).float().mean()
            losses.append(float(loss.item()))
            agreements.append(float(agreement.item()))

    student.train()
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "agreement": float(np.mean(agreements)) if agreements else 0.0,
    }


def train_student(#训练学生
    teacher: Actor,
    student: Actor,
    train_obs: torch.Tensor,
    val_obs: torch.Tensor,
    args: argparse.Namespace,
) -> List[Dict[str, float]]:
    train_loader = DataLoader(
        TensorDataset(train_obs),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    optimizer = Adam(student.parameters(), lr=args.lr)# 优化器：调整学生参数
    log_records: List[Dict[str, float]] = []

    teacher.eval()
    student.train()

    for epoch in range(1, args.epochs + 1):# 每一轮训练
        batch_losses = []
        batch_agreements = []

        for (batch_obs,) in train_loader:# 每一批数据:
            batch_obs = batch_obs.to(args.device)

            with torch.no_grad():# 教师网络的输出（标准答案）
                teacher_logits = teacher(batch_obs)

            student_logits = student(batch_obs)# 学生尝试做题
            loss = distillation_loss(student_logits, teacher_logits, args.temperature)#计算学生和老师差多少

            optimizer.zero_grad()#让学生改错，变得和和教师网络接近
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                teacher_actions = torch.argmax(teacher_logits, dim=-1)
                student_actions = torch.argmax(student_logits, dim=-1)
                agreement = (teacher_actions == student_actions).float().mean()

            batch_losses.append(float(loss.item()))
            batch_agreements.append(float(agreement.item()))

        val_metrics = evaluate_student_match(
            teacher=teacher,
            student=student,
            obs=val_obs,
            batch_size=args.batch_size,
            temperature=args.temperature,
            device=args.device,
        )
        record = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(batch_losses)) if batch_losses else 0.0,
            "train_agreement": float(np.mean(batch_agreements)) if batch_agreements else 0.0,
            "val_loss": val_metrics["loss"],
            "val_agreement": val_metrics["agreement"],
        }
        log_records.append(record)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"TrainLoss: {record['train_loss']:.6f} | "
            f"TrainAgree: {record['train_agreement']:.4f} | "
            f"ValLoss: {record['val_loss']:.6f} | "
            f"ValAgree: {record['val_agreement']:.4f}"
        )

    return log_records


def save_distillation_log(save_path: str, records: List[Dict[str, float]]) -> str:# 保存日志
    log_path = os.path.splitext(save_path)[0] + "_log.csv"
    if not records:
        return log_path

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"Distillation log saved to: {log_path}")
    return log_path


def parse_args() -> argparse.Namespace:#命令行参数
    parser = argparse.ArgumentParser(description="Distill a pruned MAPPO actor from the original actor.")
    parser.add_argument(
        "--teacher-model-path",
        default=os.path.join("results", "mappo_checkpoint.pt"),
        help="Path to the trained MAPPO checkpoint used as teacher.",
    )
    parser.add_argument(
        "--student-model-path",
        default=os.path.join("results", "mappo_actor_pruned.pt"),
        help="Path to the pruned actor checkpoint used as student initialization.",
    )
    parser.add_argument(
        "--save-path",
        default=os.path.join("results", "mappo_actor_pruned_distilled.pt"),
        help="Path for the distilled pruned actor checkpoint.",
    )
    parser.add_argument("--collect-episodes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latency-repeats", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EnvConfig()

    if args.collect_episodes <= 0:
        raise ValueError("collect_episodes must be positive.")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 加载教师和学生
    teacher = load_teacher_actor(cfg, args.teacher_model_path, args.device)
    student, student_ckpt = load_student_actor(cfg, args.student_model_path, args.device)

    # 收集教师的经验数据
    obs_array = collect_teacher_observations(
        cfg=cfg,
        teacher=teacher,
        collect_episodes=args.collect_episodes,
        device=args.device,
        seed=args.seed,
    )
    train_obs, val_obs = split_train_val(obs_array, args.val_ratio, args.seed)
    print(f"Train observations: {len(train_obs)}, validation observations: {len(val_obs)}")

    # 蒸馏前，先测一下学生有多像老师
    pre_metrics = evaluate_student_match(
        teacher=teacher,
        student=student,
        obs=val_obs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        device=args.device,
    )
    print(
        f"Before distillation | ValLoss: {pre_metrics['loss']:.6f} | "
        f"ValAgree: {pre_metrics['agreement']:.4f}"
    )

    # 开始训练，老师教学生30轮
    records = train_student(
        teacher=teacher,
        student=student,
        train_obs=train_obs,
        val_obs=val_obs,
        args=args,
    )

    # 蒸馏后，再测学生有多像老师
    post_metrics = evaluate_student_match(
        teacher=teacher,
        student=student,
        obs=val_obs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        device=args.device,
    )

    # 数参数、测速
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    compression_ratio = teacher_params / max(student_params, 1)
    teacher_latency_ms = measure_actor_latency(
        teacher,
        obs_dim=cfg.obs_dim,
        n_agents=cfg.M,
        device=args.device,
        repeats=args.latency_repeats,
    )
    student_latency_ms = measure_actor_latency(
        student,
        obs_dim=cfg.obs_dim,
        n_agents=cfg.M,
        device=args.device,
        repeats=args.latency_repeats,
    )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    log_path = save_distillation_log(args.save_path, records)

    # 保存变聪明的学生模型
    torch.save(
        {
            **student_ckpt,
            "actor": student.state_dict(),
            "distilled": True,
            "teacher_model_path": args.teacher_model_path,
            "student_init_path": args.student_model_path,
            "collect_episodes": args.collect_episodes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "temperature": args.temperature,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "distillation_log_path": log_path,
            "pre_val_loss": pre_metrics["loss"],
            "pre_val_agreement": pre_metrics["agreement"],
            "post_val_loss": post_metrics["loss"],
            "post_val_agreement": post_metrics["agreement"],
            "original_params": teacher_params,
            "pruned_params": student_params,
            "compression_ratio": compression_ratio,
            "original_latency_ms": teacher_latency_ms,
            "pruned_latency_ms": student_latency_ms,
        },
        args.save_path,
    )

    print("\n============== Distillation Summary ==============\n")
    print(f"Teacher model: {args.teacher_model_path}")
    print(f"Student init: {args.student_model_path}")
    print(f"Distilled actor: {args.save_path}")
    print(f"Parameters: {teacher_params} -> {student_params}")
    print(f"Compression ratio: {compression_ratio:.3f}x")
    print(f"Actor latency: {teacher_latency_ms:.6f} ms -> {student_latency_ms:.6f} ms")
    print(f"Val loss: {pre_metrics['loss']:.6f} -> {post_metrics['loss']:.6f}")
    print(f"Val agreement: {pre_metrics['agreement']:.4f} -> {post_metrics['agreement']:.4f}")
    print("\n==================================================\n")


if __name__ == "__main__":
    main()

