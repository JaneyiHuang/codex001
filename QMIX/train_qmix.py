from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
from typing import Any, Dict, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import EnvConfig
from env import MECEnv

try:
    from .qmix import QMIX
    from .replay_buffer import QMIXReplayBuffer
except ImportError:
    from qmix import QMIX
    from replay_buffer import QMIXReplayBuffer


def parse_train_load_factors(raw: str) -> List[float]:
    factors = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not factors:
        raise ValueError("TRAIN_LOAD_FACTORS must contain at least one factor.")
    return factors


def build_episode_cfg(cfg: EnvConfig, load_factor: float) -> EnvConfig:
    return replace(
        cfg,
        task_min_bits=cfg.task_min_bits * load_factor,
        task_max_bits=cfg.task_max_bits * load_factor,
    )


def linear_epsilon(
    global_step: int,
    epsilon_start: float,
    epsilon_finish: float,
    epsilon_anneal_steps: int,
) -> float:
    if epsilon_anneal_steps <= 0:
        return epsilon_finish
    progress = min(float(global_step) / float(epsilon_anneal_steps), 1.0)
    return epsilon_start + progress * (epsilon_finish - epsilon_start)


def moving_average(data: List[float], window: int = 20) -> np.ndarray:
    if not data:
        return np.array([])
    values = np.array(data, dtype=np.float32)
    if len(values) < window:
        return values
    ma = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.concatenate([values[: window - 1], ma], axis=0)


def save_training_csv(save_dir: str, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "qmix_training_log.csv")
    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"QMIX training log saved to: {csv_path}")


def plot_training_curves(save_dir: str, records: List[Dict[str, Any]]) -> None:
    if not records:
        return

    os.makedirs(save_dir, exist_ok=True)
    episodes = [r["episode"] for r in records]
    rewards = [r["episode_reward"] for r in records]
    delays = [r["episode_delay_mean"] for r in records]
    drops = [r["episode_drop_rate"] for r in records]
    losses = [r["qmix_loss"] for r in records]

    plots = [
        ("reward_curve.png", rewards, moving_average(rewards), "Episode Reward", "Reward"),
        ("delay_curve.png", delays, moving_average(delays), "Episode Delay Mean", "Mean Delay"),
        ("drop_rate_curve.png", drops, moving_average(drops), "Episode Drop Rate", "Drop Rate"),
        ("loss_curve.png", losses, moving_average(losses), "QMIX TD Loss", "Loss"),
    ]

    for filename, values, smooth_values, title, ylabel in plots:
        plt.figure(figsize=(8, 5))
        plt.plot(episodes, values, label=title)
        plt.plot(episodes, smooth_values, label="MA(20)")
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(f"QMIX {title}")
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(save_dir, filename)
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"Saved figure: {fig_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QMIX on the MEC offloading environment.")
    parser.add_argument("--episodes", type=int, default=int(os.getenv("QMIX_TRAIN_EPISODES", "2000")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("QMIX_SEED", "52")))
    parser.add_argument("--device", default=os.getenv("QMIX_DEVICE", "cpu"))
    parser.add_argument("--save-dir", default=os.getenv("QMIX_SAVE_DIR", os.path.join("QMIX", "results")))
    parser.add_argument("--buffer-size", type=int, default=int(os.getenv("QMIX_BUFFER_SIZE", "50000")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("QMIX_BATCH_SIZE", "128")))
    parser.add_argument("--warmup-steps", type=int, default=int(os.getenv("QMIX_WARMUP_STEPS", "1000")))
    parser.add_argument("--update-interval", type=int, default=int(os.getenv("QMIX_UPDATE_INTERVAL", "2")))
    parser.add_argument("--updates-per-train-step", type=int, default=int(os.getenv("QMIX_UPDATES_PER_TRAIN_STEP", "1")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("QMIX_LR", "5e-4")))
    parser.add_argument("--gamma", type=float, default=float(os.getenv("QMIX_GAMMA", "0.99")))
    parser.add_argument("--target-update-interval", type=int, default=int(os.getenv("QMIX_TARGET_UPDATE_INTERVAL", "200")))
    parser.add_argument("--epsilon-start", type=float, default=float(os.getenv("QMIX_EPSILON_START", "1.0")))
    parser.add_argument("--epsilon-finish", type=float, default=float(os.getenv("QMIX_EPSILON_FINISH", "0.08")))
    parser.add_argument("--epsilon-anneal-steps", type=int, default=int(os.getenv("QMIX_EPSILON_ANNEAL_STEPS", "60000")))
    parser.add_argument("--train-load-factors", default=os.getenv("TRAIN_LOAD_FACTORS", "0.67,0.83,1.00,1.17,1.33"))
    parser.add_argument("--phase1-ratio", type=float, default=float(os.getenv("QMIX_PHASE1_RATIO", "1.0")))
    parser.add_argument("--phase1-load-factor", type=float, default=float(os.getenv("QMIX_PHASE1_LOAD_FACTOR", "1.0")))
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.update_interval <= 0:
        raise ValueError("--update-interval must be positive.")

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, "qmix_checkpoint.pt")
    last_model_path = os.path.join(args.save_dir, "qmix_last.pt")

    cfg = EnvConfig()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_load_factors = parse_train_load_factors(args.train_load_factors)
    phase1_ratio = min(max(args.phase1_ratio, 0.0), 1.0)
    phase1_episodes = int(round(args.episodes * phase1_ratio))

    agent = QMIX(
        obs_dim=cfg.obs_dim,
        state_dim=cfg.state_dim,
        n_actions=cfg.n_actions,
        n_agents=cfg.M,
        agent_hidden_dims=[64, 64],
        mixer_embed_dim=32,
        mixer_hyper_hidden_dim=64,
        lr=args.lr,
        gamma=args.gamma,
        target_update_interval=args.target_update_interval,
        max_grad_norm=10.0,
        device=args.device,
    )
    replay_buffer = QMIXReplayBuffer(
        capacity=args.buffer_size,
        n_agents=cfg.M,
        obs_dim=cfg.obs_dim,
        state_dim=cfg.state_dim,
        device=args.device,
    )

    training_records: List[Dict[str, Any]] = []
    best_score = -1e18
    global_step = 0
    save_window = 20

    for episode in range(1, args.episodes + 1):
        if episode <= phase1_episodes:
            stage_name = "phase1"
            load_factor = args.phase1_load_factor
        else:
            stage_name = "phase2"
            load_factor = float(rng.choice(train_load_factors))

        episode_cfg = build_episode_cfg(cfg, load_factor)
        env = MECEnv(episode_cfg)
        env.seed(args.seed + episode)

        data = env.reset()
        obs = data["obs"]
        state = data["state"]
        done = False

        episode_reward = 0.0
        delay_list = []
        drop_rate_list = []
        offload_rate_list = []
        energy_mean_list = []
        losses = []
        td_errors = []
        q_totals = []
        target_q_totals = []
        step_count = 0

        while not done:
            epsilon = linear_epsilon(
                global_step,
                args.epsilon_start,
                args.epsilon_finish,
                args.epsilon_anneal_steps,
            )
            actions = agent.select_actions(obs, epsilon=epsilon, deterministic=False)
            out = env.step(actions)
            next_obs = out["obs"]
            next_state = out["state"]
            reward = float(out["reward"])
            done = bool(out["done"])
            info = out["info"]

            replay_buffer.store(
                obs=obs,
                state=state,
                actions=actions,
                reward=reward,
                next_obs=next_obs,
                next_state=next_state,
                done=done,
            )

            if (
                global_step >= args.warmup_steps
                and replay_buffer.can_sample(args.batch_size)
                and global_step % args.update_interval == 0
            ):
                for _ in range(args.updates_per_train_step):
                    batch = replay_buffer.sample(args.batch_size)
                    update_info = agent.update(batch)
                    losses.append(update_info["loss"])
                    td_errors.append(update_info["td_error_abs"])
                    q_totals.append(update_info["q_total_mean"])
                    target_q_totals.append(update_info["target_q_total_mean"])

            episode_reward += reward
            delay_list.append(info["delay_mean"])
            drop_rate_list.append(info["drop_rate"])
            offload_rate_list.append(info["offload_rate"])
            energy_mean_list.append(info["energy_mean"])

            obs = next_obs
            state = next_state
            step_count += 1
            global_step += 1

        recent_rewards = [r["episode_reward"] for r in training_records[-(save_window - 1):]]
        recent_rewards.append(episode_reward)
        reward_ma = float(np.mean(recent_rewards))
        epsilon = linear_epsilon(
            global_step,
            args.epsilon_start,
            args.epsilon_finish,
            args.epsilon_anneal_steps,
        )

        record = {
            "episode": episode,
            "stage": stage_name,
            "load_factor": load_factor,
            "task_min_bits": episode_cfg.task_min_bits,
            "task_max_bits": episode_cfg.task_max_bits,
            "steps": step_count,
            "global_step": global_step,
            "epsilon": epsilon,
            "episode_reward": episode_reward,
            "reward_ma": reward_ma,
            "episode_delay_mean": float(np.mean(delay_list)) if delay_list else 0.0,
            "episode_drop_rate": float(np.mean(drop_rate_list)) if drop_rate_list else 0.0,
            "episode_offload_rate": float(np.mean(offload_rate_list)) if offload_rate_list else 0.0,
            "episode_energy_mean": float(np.mean(energy_mean_list)) if energy_mean_list else 0.0,
            "qmix_loss": float(np.mean(losses)) if losses else 0.0,
            "td_error_abs": float(np.mean(td_errors)) if td_errors else 0.0,
            "q_total_mean": float(np.mean(q_totals)) if q_totals else 0.0,
            "target_q_total_mean": float(np.mean(target_q_totals)) if target_q_totals else 0.0,
            "buffer_size": len(replay_buffer),
        }
        training_records.append(record)

        print(
            f"Episode {episode:4d}/{args.episodes} | "
            f"Stage: {stage_name:6s} | "
            f"Load: {load_factor:4.2f} | "
            f"Eps: {epsilon:5.3f} | "
            f"Reward: {episode_reward:8.3f} | "
            f"RewardMA: {reward_ma:8.3f} | "
            f"Delay: {record['episode_delay_mean']:8.3f} | "
            f"Drop: {record['episode_drop_rate']:6.3f} | "
            f"Offload: {record['episode_offload_rate']:6.3f} | "
            f"Loss: {record['qmix_loss']:8.4f} | "
            f"Buffer: {len(replay_buffer):5d}"
        )

        if reward_ma > best_score and len(replay_buffer) >= args.batch_size:
            best_score = reward_ma
            agent.save(checkpoint_path)

    agent.save(last_model_path)
    if not os.path.exists(checkpoint_path):
        agent.save(checkpoint_path)

    save_training_csv(args.save_dir, training_records)
    plot_training_curves(args.save_dir, training_records)

    print("\nQMIX training finished.")
    print(f"Best QMIX model saved to: {checkpoint_path}")
    print(f"Last QMIX model saved to: {last_model_path}")


if __name__ == "__main__":
    train()
