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


LOAD_CURVE_STYLES: Dict[str, Dict[str, Any]] = {
    "mappo": {"color": "#111827", "linestyle": "-", "marker": "o", "linewidth": 2.8, "zorder": 8},
    "qmix": {"color": "#7C3AED", "linestyle": "-", "marker": "P", "linewidth": 2.2, "zorder": 5},
    "pruned": {"color": "#F97316", "linestyle": "--", "marker": "s", "linewidth": 2.0, "zorder": 6},
    "distilled": {"color": "#16A34A", "linestyle": "-.", "marker": "^", "linewidth": 2.0, "zorder": 7},
    "offload": {"color": "#DC2626", "linestyle": "-", "marker": "D", "linewidth": 2.0, "zorder": 4},
    "greedy": {"color": "#9333EA", "linestyle": "--", "marker": "v", "linewidth": 2.0, "zorder": 3},
    "local": {"color": "#92400E", "linestyle": "-", "marker": "X", "linewidth": 2.0, "zorder": 2},
    "random": {"color": "#DB2777", "linestyle": ":", "marker": "o", "linewidth": 2.2, "zorder": 4},
}

GRID_STYLE: Dict[str, Any] = {
    "color": "#D5DBE5",
    "linewidth": 0.6,
    "alpha": 0.82,
}


def apply_readable_grid(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, which="major", **GRID_STYLE)


def load_curve_style_key(policy_name: str) -> str:
    for prefix in ("pruned", "distilled", "random"):
        if policy_name.startswith(prefix):
            return prefix
    return policy_name


def load_curve_draw_order(policies: List[str]) -> List[str]:
    return [name for name in policies if name != "mappo"] + [
        name for name in policies if name == "mappo"
    ]


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
        apply_readable_grid(ax)

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
        fig, ax = plt.subplots(figsize=(8, 5))
        apply_readable_grid(ax)

        legend_entries = []
        for policy_name in load_curve_draw_order(policies):
            matched = sorted(
                [r for r in all_results if r["policy"] == policy_name],
                key=lambda r: r["load_factor"],
            )
            if not matched:
                continue
            x = np.array([r["avg_task_mbits"] for r in matched], dtype=np.float32)
            y = np.array([r[metric_key] for r in matched], dtype=np.float32)
            y_std = np.array([r[std_key] for r in matched], dtype=np.float32)
            style = LOAD_CURVE_STYLES.get(
                load_curve_style_key(policy_name),
                {"marker": "o", "linewidth": 2.0, "zorder": 3},
            )

            line = ax.plot(x, y, label=policy_name, **style)[0]
            legend_entries.append((policy_name, line))
            ax.fill_between(
                x,
                y - y_std,
                y + y_std,
                color=line.get_color(),
                alpha=0.08,
                zorder=max(float(style.get("zorder", 3)) - 1.0, 0.0),
            )

        ax.set_xlabel("Average Task Size (Mbits)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} under Different Load Levels")
        if legend_entries:
            legend_by_name = {name: line for name, line in legend_entries}
            ordered_names = [name for name in policies if name in legend_by_name]
            ax.legend(
                [legend_by_name[name] for name in ordered_names],
                ordered_names,
            )
        fig.tight_layout()

        fig_path = os.path.join(save_dir, filename)
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"Saved figure: {fig_path}")
