# train.py
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
from dataclasses import replace
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
import torch

from env import MECEnv
from buffer import RolloutBuffer
from mappo import MAPPO
from config import EnvConfig



def parse_train_load_factors() -> List[float]:
    """
    Parse discrete load factors for multi-load random training.
    """
    raw = os.getenv("TRAIN_LOAD_FACTORS", "0.67,0.83,1.00,1.17,1.33")
    factors = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(factors) == 0:
        raise ValueError("TRAIN_LOAD_FACTORS must contain at least one factor.")
    return factors


def build_episode_cfg(cfg: EnvConfig, load_factor: float) -> EnvConfig:
    return replace(
        cfg,
        task_min_bits=cfg.task_min_bits * load_factor,
        task_max_bits=cfg.task_max_bits * load_factor,
    )


def moving_average(data: List[float], window: int = 20) -> np.ndarray:
    """
    Compute moving average for smoother visualization.
    """
    if len(data) == 0:
        return np.array([])
    data = np.array(data, dtype=np.float32)
    if len(data) < window:
        return data
    ma = np.convolve(data, np.ones(window) / window, mode="valid")
    prefix = data[:window - 1]
    return np.concatenate([prefix, ma], axis=0)


def save_training_csv(save_dir: str, records: List[Dict[str, Any]]) -> None:
    """
    Save training logs to CSV.
    """
    if len(records) == 0:
        return

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "training_log.csv")

    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Training log saved to: {csv_path}")


def plot_training_curves(save_dir: str, records: List[Dict[str, Any]]) -> None:
    """
    Plot reward and delay curves.
    """
    if len(records) == 0:
        return

    os.makedirs(save_dir, exist_ok=True)

    episodes = [r["episode"] for r in records]
    rewards = [r["episode_reward"] for r in records]
    delays = [r["episode_delay_mean"] for r in records]
    drops = [r["episode_drop_rate"] for r in records]

    reward_ma = moving_average(rewards, window=20)
    delay_ma = moving_average(delays, window=20)
    drop_ma = moving_average(drops, window=20)

    # Reward curve
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, rewards, label="Episode Reward")
    plt.plot(episodes, reward_ma, label="Reward MA(20)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.tight_layout()
    reward_fig_path = os.path.join(save_dir, "reward_curve.png")
    plt.savefig(reward_fig_path, dpi=200)
    plt.close()

    # Delay curve
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, delays, label="Episode Delay Mean")
    plt.plot(episodes, delay_ma, label="Delay MA(20)")
    plt.xlabel("Episode")
    plt.ylabel("Mean Delay")
    plt.title("Training Delay Curve")
    plt.legend()
    plt.tight_layout()
    delay_fig_path = os.path.join(save_dir, "delay_curve.png")
    plt.savefig(delay_fig_path, dpi=200)
    plt.close()

    # Drop rate curve
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, drops, label="Episode Drop Rate")
    plt.plot(episodes, drop_ma, label="Drop Rate MA(20)")
    plt.xlabel("Episode")
    plt.ylabel("Drop Rate")
    plt.title("Training Drop Rate Curve")
    plt.legend()
    plt.tight_layout()
    drop_fig_path = os.path.join(save_dir, "drop_rate_curve.png")
    plt.savefig(drop_fig_path, dpi=200)
    plt.close()

    print(f"Reward curve saved to: {reward_fig_path}")
    print(f"Delay curve saved to: {delay_fig_path}")
    print(f"Drop rate curve saved to: {drop_fig_path}")


def train():
    # =========================================================
    # 1. Config
    # =========================================================
    cfg = EnvConfig()

    # Training hyperparameters
    num_episodes = 1000        # 先用 300 跑通，后面可改成 1000/2000   
    save_dir = "results"
    best_model_path = os.path.join(save_dir, "mappo_checkpoint.pt")
    # last_model_path = os.path.join(save_dir, "mappo_last.pt") # 新增一个保存训练最后的模型

    os.makedirs(save_dir, exist_ok=True)
    seed = 42
    num_episodes = int(os.getenv("TRAIN_EPISODES", "2000"))
    save_window = 20
    last_model_path = os.path.join(save_dir, "mappo_last.pt")
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_load_factors = parse_train_load_factors()
    phase1_ratio = float(os.getenv("PHASE1_RATIO", "1.0"))
    phase1_ratio = min(max(phase1_ratio, 0.0), 1.0)
    phase1_episodes = int(round(num_episodes * phase1_ratio))
    phase2_episodes = num_episodes - phase1_episodes
    phase1_load_factor = float(os.getenv("PHASE1_LOAD_FACTOR", "1.0"))

    # =========================================================
    # 2. Build environment and agent
    # =========================================================
    agent = MAPPO(
        obs_dim=cfg.obs_dim,
        state_dim=cfg.state_dim,
        n_actions=cfg.n_actions,
        n_agents=cfg.M,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[256, 256],
        actor_lr=1e-4,
        critic_lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.005,
        critic_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=6,
        minibatch_size=128,
        device="cpu",
    )

    # =========================================================
    # 3. Logging
    # =========================================================
    training_records: List[Dict[str, Any]] = []
    best_score = -1e18

    # =========================================================
    # 4. Main training loop
    # =========================================================
    for episode in range(1, num_episodes + 1):
        if episode <= phase1_episodes:
            stage_name = "phase1"
            load_factor = phase1_load_factor
        else:
            stage_name = "phase2"
            load_factor = float(rng.choice(train_load_factors))

        episode_cfg = build_episode_cfg(cfg, load_factor)
        env = MECEnv(episode_cfg)
        env.seed(seed + episode)

        # create a fresh rollout buffer for each episode
        buffer = RolloutBuffer(
            episode_limit=episode_cfg.episode_limit,
            n_agents=episode_cfg.M,
            obs_dim=episode_cfg.obs_dim,
            state_dim=episode_cfg.state_dim,
            gamma=0.99,
            gae_lambda=0.95,
            device="cpu",
        )

        data = env.reset()
        obs = data["obs"]
        state = data["state"]

        done = False
        episode_reward = 0.0

        delay_list = []
        drop_rate_list = []
        offload_rate_list = []
        energy_mean_list = []

        step_count = 0

        while not done:
            # 1) select actions
            act_out = agent.select_actions(obs, deterministic=False)
            actions = act_out["actions"]
            log_probs = act_out["log_probs"]

            # 2) critic estimate
            value = agent.get_value(state)

            # 3) step environment
            out = env.step(actions)
            next_obs = out["obs"]
            next_state = out["state"]
            reward = out["reward"]
            done = out["done"]
            info = out["info"]

            # 4) store transition
            buffer.store(
                obs=obs,
                state=state,
                actions=actions,
                log_probs=log_probs,
                reward=reward,
                done=done,
                value=value,
            )

            # 5) record episode metrics
            episode_reward += reward
            delay_list.append(info["delay_mean"])
            drop_rate_list.append(info["drop_rate"])
            offload_rate_list.append(info["offload_rate"])
            energy_mean_list.append(info["energy_mean"])

            # 6) move forward
            obs = next_obs
            state = next_state
            step_count += 1

        # =====================================================
        # 5. Compute returns / advantages
        # =====================================================
        last_value = 0.0
        buffer.compute_returns_and_advantages(last_value=last_value)
        buffer.normalize_advantages()
        batch = buffer.get()

        # =====================================================
        # 6. Update MAPPO
        # =====================================================
        update_info = agent.update(batch)

        # =====================================================
        # 7. Logging
        # =====================================================
        episode_delay_mean = float(np.mean(delay_list)) if delay_list else 0.0
        episode_drop_rate = float(np.mean(drop_rate_list)) if drop_rate_list else 0.0
        episode_offload_rate = float(np.mean(offload_rate_list)) if offload_rate_list else 0.0
        episode_energy_mean = float(np.mean(energy_mean_list)) if energy_mean_list else 0.0
        recent_rewards = [r["episode_reward"] for r in training_records[-(save_window - 1):]]
        recent_rewards.append(episode_reward)
        reward_ma = float(np.mean(recent_rewards))

        record = {
            "episode": episode,
            "stage": stage_name,
            "load_factor": load_factor,
            "task_min_bits": episode_cfg.task_min_bits,
            "task_max_bits": episode_cfg.task_max_bits,
            "steps": step_count,
            "episode_reward": episode_reward,
            "reward_ma": reward_ma,
            "episode_delay_mean": episode_delay_mean,
            "episode_drop_rate": episode_drop_rate,
            "episode_offload_rate": episode_offload_rate,
            "episode_energy_mean": episode_energy_mean,
            "actor_loss": update_info["actor_loss"],
            "critic_loss": update_info["critic_loss"],
            "entropy": update_info["entropy"],
            "total_loss": update_info["total_loss"],
        }
        training_records.append(record)

        # print progress
        print(
            f"Episode {episode:4d}/{num_episodes} | "
            f"Stage: {stage_name:6s} | "
            f"Load: {load_factor:4.2f} | "
            f"Reward: {episode_reward:8.3f} | "
            f"RewardMA: {reward_ma:8.3f} | "
            f"Delay: {episode_delay_mean:8.3f} | "
            f"Drop: {episode_drop_rate:6.3f} | "
            f"Offload: {episode_offload_rate:6.3f} | "
            f"ActorLoss: {update_info['actor_loss']:8.4f} | "
            f"CriticLoss: {update_info['critic_loss']:8.4f}"
        )

        # save best model by recent reward moving average
        if reward_ma > best_score:
            best_score = reward_ma
            agent.save(best_model_path)
        
    # 保存最后训练结束时的模型
    # agent.save(last_model_path)

    agent.save(last_model_path)

    # =========================================================
    # 8. Save logs and figures
    # =========================================================
    save_training_csv(save_dir, training_records)
    plot_training_curves(save_dir, training_records)

    print(f"\nTraining finished.")
    print(f"Best model saved to: {best_model_path}")
    print(f"Last model saved to: {last_model_path}")


if __name__ == "__main__":
    train()
