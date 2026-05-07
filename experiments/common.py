from __future__ import annotations

import csv
import os
from typing import Any, Dict, Iterable, List, Protocol

import matplotlib.pyplot as plt
import numpy as np

from env import MECEnv


class Policy(Protocol):
    name: str

    def select_actions(self, env: MECEnv, obs: np.ndarray) -> np.ndarray:
        """Return one action per agent."""


def run_one_episode(env: MECEnv, policy: Policy) -> Dict[str, float]:
    data = env.reset()
    obs = data["obs"]
    done = False

    episode_reward = 0.0
    delay_list: List[float] = []
    drop_rate_list: List[float] = []
    offload_rate_list: List[float] = []
    energy_mean_list: List[float] = []

    while not done:
        actions = policy.select_actions(env, obs)
        out = env.step(actions)
        obs = out["obs"]
        done = out["done"]
        info = out["info"]

        episode_reward += float(out["reward"])
        delay_list.append(float(info["delay_mean"]))
        drop_rate_list.append(float(info["drop_rate"]))
        offload_rate_list.append(float(info["offload_rate"]))
        energy_mean_list.append(float(info["energy_mean"]))

    return {
        "episode_reward": float(episode_reward),
        "episode_delay_mean": float(np.mean(delay_list)) if delay_list else 0.0,
        "episode_drop_rate": float(np.mean(drop_rate_list)) if drop_rate_list else 0.0,
        "episode_offload_rate": float(np.mean(offload_rate_list)) if offload_rate_list else 0.0,
        "episode_energy_mean": float(np.mean(energy_mean_list)) if energy_mean_list else 0.0,
    }


def evaluate_policy(
    cfg: Any,
    policy: Policy,
    num_eval_episodes: int,
    seed: int = 123,
) -> Dict[str, float]:
    env = MECEnv(cfg)
    env.seed(seed)

    rewards = []
    delays = []
    drops = []
    offloads = []
    energies = []

    for _ in range(num_eval_episodes):
        result = run_one_episode(env, policy)
        rewards.append(result["episode_reward"])
        delays.append(result["episode_delay_mean"])
        drops.append(result["episode_drop_rate"])
        offloads.append(result["episode_offload_rate"])
        energies.append(result["episode_energy_mean"])

    return {
        "policy": policy.name,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "delay_mean": float(np.mean(delays)),
        "delay_std": float(np.std(delays)),
        "drop_rate_mean": float(np.mean(drops)),
        "drop_rate_std": float(np.std(drops)),
        "offload_rate_mean": float(np.mean(offloads)),
        "offload_rate_std": float(np.std(offloads)),
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
    }


def save_csv(save_dir: str, records: List[Dict[str, Any]], filename: str) -> None:
    if not records:
        return

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fieldnames = list(records[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved CSV: {path}")


def print_summary_table(results: Iterable[Dict[str, Any]]) -> None:
    results = list(results)
    print("\n================ Comparison Summary ================\n")
    header = (
        f"{'Policy':<12}"
        f"{'DelayMean':>12}"
        f"{'DelayStd':>12}"
        f"{'DropMean':>12}"
        f"{'OffloadMean':>14}"
        f"{'RewardMean':>14}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['policy']:<12}"
            f"{r['delay_mean']:>12.4f}"
            f"{r['delay_std']:>12.4f}"
            f"{r['drop_rate_mean']:>12.4f}"
            f"{r['offload_rate_mean']:>14.4f}"
            f"{r['reward_mean']:>14.4f}"
        )

    print("\n====================================================\n")


def plot_bar_dashboard(save_dir: str, results: List[Dict[str, Any]]) -> None:
    if not results:
        return

    os.makedirs(save_dir, exist_ok=True)
    policies = [r["policy"] for r in results]
    metrics = [
        ("delay_mean", "delay_std", "Mean Delay", "Delay"),
        ("drop_rate_mean", "drop_rate_std", "Drop Rate", "Drop Rate"),
        ("reward_mean", "reward_std", "Reward", "Reward"),
        ("offload_rate_mean", "offload_rate_std", "Offload Rate", "Offload Rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    x = np.arange(len(policies))

    for ax, (mean_key, std_key, ylabel, title) in zip(axes, metrics):
        means = [r[mean_key] for r in results]
        stds = [r[std_key] for r in results]
        ax.bar(x, means, yerr=stds, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=15)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig.suptitle("Selected Policy Comparison", fontsize=14)
    fig.tight_layout()
    fig_path = os.path.join(save_dir, "comparison_dashboard.png")
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    print(f"Saved figure: {fig_path}")


def plot_load_curves(
    save_dir: str,
    all_results: List[Dict[str, Any]],
    policies: List[str],
) -> None:
    if not all_results:
        return

    os.makedirs(save_dir, exist_ok=True)
    metrics = [
        ("delay_mean", "delay_std", "Mean Delay", "loads_delay.png"),
        ("drop_rate_mean", "drop_rate_std", "Drop Rate", "loads_drop_rate.png"),
        ("offload_rate_mean", "offload_rate_std", "Offload Rate", "loads_offload_rate.png"),
        ("reward_mean", "reward_std", "Reward", "loads_reward.png"),
    ]

    for metric_key, std_key, ylabel, filename in metrics:
        plt.figure(figsize=(8, 5))

        for policy_name in policies:
            matched = sorted(
                [r for r in all_results if r["policy"] == policy_name],
                key=lambda r: r["load_factor"],
            )
            x = np.array([r["avg_task_mbits"] for r in matched], dtype=np.float32)
            y = np.array([r[metric_key] for r in matched], dtype=np.float32)
            y_std = np.array([r[std_key] for r in matched], dtype=np.float32)

            plt.plot(x, y, marker="o", linewidth=2.0, label=policy_name)
            plt.fill_between(x, y - y_std, y + y_std, alpha=0.10)

        plt.xlabel("Average Task Size (Mbits)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} under Different Load Levels")
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(save_dir, filename)
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"Saved figure: {fig_path}")

