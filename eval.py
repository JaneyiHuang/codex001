# eval.py
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

from config import EnvConfig
from env import MECEnv
from mappo import MAPPO


def moving_average(data: List[float], window: int = 20) -> np.ndarray:
    if len(data) == 0:
        return np.array([])
    arr = np.array(data, dtype=np.float32)
    if len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=np.float32) / window
    ma = np.convolve(arr, kernel, mode="valid")
    prefix = arr[:window - 1]
    return np.concatenate([prefix, ma], axis=0)


def load_training_records(results_dir: str) -> List[Dict[str, float]]:
    csv_path = os.path.join(results_dir, "training_log.csv")
    if not os.path.exists(csv_path):
        return []

    records: List[Dict[str, float]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, float] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    continue
            records.append(parsed)
    return records


# =========================================================
# Helper: Greedy baseline
# =========================================================
def greedy_action_for_agent(env: MECEnv, m: int) -> int:
    """
    Greedy baseline:
    compare estimated local delay and estimated offload delay,
    then choose the smaller one.
    """
    cfg = env.cfg

    task_bits = float(env.lambda_arr[m])
    E_m = float(env.E_arr[m])
    h_m = float(env.h_arr[m])

    # ---------- local estimate ----------
    e_loc = env.local_energy_cost()
    if E_m >= e_loc and cfg.f_local > 0:
        T_loc = env.local_processing_slots(task_bits)
    else:
        T_loc = cfg.psi

    S_loc = max(float(env.t), float(env.F_loc[m]))
    est_local_delay = (S_loc + T_loc - 1) - float(env.t)

    # ---------- offload estimate ----------
    e_tx = env.tx_energy_cost()
    if E_m >= e_tx:
        rate = env.uplink_rate(h_m)
        if rate > 0:
            T_tx = env.tx_slots(task_bits, rate)
        else:
            T_tx = cfg.psi
    else:
        T_tx = cfg.psi

    T_edge = env.edge_processing_slots(task_bits)

    S_tx = max(float(env.t), float(env.F_tx[m]))
    C_tx = S_tx + T_tx - 1
    S_edge = max(C_tx + 1, float(env.F_edge))
    est_offload_delay = (S_edge + T_edge - 1) - float(env.t)

    return 0 if est_local_delay <= est_offload_delay else 1


# =========================================================
# One episode evaluation
# =========================================================
def run_one_episode(
    env: MECEnv,
    policy_name: str,
    agent: MAPPO | None = None,
) -> Dict[str, float]:
    data = env.reset()
    obs = data["obs"]

    done = False

    episode_reward = 0.0
    delay_list = []
    drop_rate_list = []
    offload_rate_list = []
    energy_mean_list = []

    while not done:
        if policy_name == "mappo":
            if agent is None:
                raise ValueError("agent must be provided for policy_name='mappo'")
            act_out = agent.select_actions(obs, deterministic=True)
            actions = act_out["actions"]

        elif policy_name == "local":
            actions = np.zeros(env.M, dtype=np.int64)

        elif policy_name == "offload":
            actions = np.ones(env.M, dtype=np.int64)

        elif policy_name == "random":
            actions = np.random.randint(0, 2, size=env.M, dtype=np.int64)

        elif policy_name == "greedy":
            actions = np.array(
                [greedy_action_for_agent(env, m) for m in range(env.M)],
                dtype=np.int64
            )

        else:
            raise ValueError(f"Unknown policy_name: {policy_name}")

        out = env.step(actions)
        obs = out["obs"]
        done = out["done"]
        info = out["info"]
        reward = out["reward"]

        episode_reward += reward
        delay_list.append(info["delay_mean"])
        drop_rate_list.append(info["drop_rate"])
        offload_rate_list.append(info["offload_rate"])
        energy_mean_list.append(info["energy_mean"])

    result = {
        "episode_reward": float(episode_reward),
        "episode_delay_mean": float(np.mean(delay_list)) if delay_list else 0.0,
        "episode_drop_rate": float(np.mean(drop_rate_list)) if drop_rate_list else 0.0,
        "episode_offload_rate": float(np.mean(offload_rate_list)) if offload_rate_list else 0.0,
        "episode_energy_mean": float(np.mean(energy_mean_list)) if energy_mean_list else 0.0,
    }
    return result


# =========================================================
# Multi-episode evaluation
# =========================================================
def evaluate_policy(
    env: MECEnv,
    policy_name: str,
    num_eval_episodes: int,
    agent: MAPPO | None = None,
) -> Dict[str, float]:
    rewards = []
    delays = []
    drops = []
    offloads = []
    energies = []

    for _ in range(num_eval_episodes):
        result = run_one_episode(env, policy_name=policy_name, agent=agent)
        rewards.append(result["episode_reward"])
        delays.append(result["episode_delay_mean"])
        drops.append(result["episode_drop_rate"])
        offloads.append(result["episode_offload_rate"])
        energies.append(result["episode_energy_mean"])

    summary = {
        "policy": policy_name,
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
    return summary


# =========================================================
# Save results
# =========================================================
def save_eval_csv(save_dir: str, results: List[Dict[str, float]]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "eval_summary.csv")

    if len(results) == 0:
        return

    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Evaluation summary saved to: {csv_path}")


def print_summary_table(results: List[Dict[str, float]]) -> None:
    print("\n================ Evaluation Summary ================\n")
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

    print("\n===================================================\n")


# =========================================================
# Plotting
# =========================================================
def plot_bar_metric(
    save_dir: str,
    results: List[Dict[str, float]],
    metric_mean_key: str,
    metric_std_key: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    policies = [r["policy"] for r in results]
    means = [r[metric_mean_key] for r in results]
    stds = [r[metric_std_key] for r in results]

    x = np.arange(len(policies))

    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, policies)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    fig_path = os.path.join(save_dir, filename)
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"Saved figure: {fig_path}")


def plot_all_eval_figures(save_dir: str, results: List[Dict[str, float]]) -> None:
    plot_bar_metric(
        save_dir=save_dir,
        results=results,
        metric_mean_key="delay_mean",
        metric_std_key="delay_std",
        ylabel="Mean Delay",
        title="Comparison of Mean Delay",
        filename="bar_delay.png",
    )

    plot_bar_metric(
        save_dir=save_dir,
        results=results,
        metric_mean_key="drop_rate_mean",
        metric_std_key="drop_rate_std",
        ylabel="Drop Rate",
        title="Comparison of Drop Rate",
        filename="bar_drop_rate.png",
    )

    plot_bar_metric(
        save_dir=save_dir,
        results=results,
        metric_mean_key="offload_rate_mean",
        metric_std_key="offload_rate_std",
        ylabel="Offload Rate",
        title="Comparison of Offload Rate",
        filename="bar_offload_rate.png",
    )

    plot_bar_metric(
        save_dir=save_dir,
        results=results,
        metric_mean_key="reward_mean",
        metric_std_key="reward_std",
        ylabel="Reward",
        title="Comparison of Reward",
        filename="bar_reward.png",
    )


def plot_eval_dashboard(save_dir: str, results: List[Dict[str, float]]) -> None:
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

    fig.suptitle("Policy Comparison under Fixed Scenario", fontsize=14)
    fig.tight_layout()
    fig_path = os.path.join(save_dir, "eval_dashboard.png")
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    print(f"Saved figure: {fig_path}")


def plot_training_convergence_overview(
    save_dir: str,
    training_records: List[Dict[str, float]],
) -> None:
    if len(training_records) == 0:
        print("Training log not found. Skip convergence overview.")
        return

    os.makedirs(save_dir, exist_ok=True)

    episodes = [r["episode"] for r in training_records]
    rewards = [r["episode_reward"] for r in training_records]
    delays = [r["episode_delay_mean"] for r in training_records]
    drops = [r["episode_drop_rate"] for r in training_records]
    offloads = [r["episode_offload_rate"] for r in training_records]

    reward_ma = moving_average(rewards, window=50)
    delay_ma = moving_average(delays, window=50)
    drop_ma = moving_average(drops, window=50)
    offload_ma = moving_average(offloads, window=50)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    series = [
        (rewards, reward_ma, "Reward", "Episode Reward"),
        (delays, delay_ma, "Delay", "Episode Delay Mean"),
        (drops, drop_ma, "Drop Rate", "Episode Drop Rate"),
        (offloads, offload_ma, "Offload Rate", "Episode Offload Rate"),
    ]

    for ax, (raw, ma, ylabel, title) in zip(axes, series):
        ax.plot(episodes, raw, alpha=0.35, linewidth=1.0, label="Raw")
        ax.plot(episodes, ma, linewidth=2.0, label="MA(50)")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Training Convergence Overview", fontsize=14)
    fig.tight_layout()
    fig_path = os.path.join(save_dir, "training_convergence_overview.png")
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    print(f"Saved figure: {fig_path}")


# =========================================================
# Main
# =========================================================
def main():
    cfg = EnvConfig()

    num_eval_episodes = 50
    model_path = os.path.join("results", "mappo_checkpoint.pt")
    # model_path = os.path.join("results", "mappo_last.pt")
    save_dir = os.path.join("results", "eval_results")
    # save_dir = os.path.join("results", "eval_last_results")


    env = MECEnv(cfg)
    env.seed(123)

    agent = MAPPO(
        obs_dim=cfg.obs_dim,
        state_dim=cfg.state_dim,
        n_actions=cfg.n_actions,
        n_agents=cfg.M,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[256, 256],
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        critic_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        minibatch_size=256,
        device="cpu",
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Cannot find trained model: {model_path}\n"
            f"Please run train.py first."
        )

    agent.load(model_path)
    print(f"Loaded trained MAPPO model from: {model_path}")

    policies = ["mappo", "local", "offload", "random", "greedy"]
    all_results = []

    for policy_name in policies:
        print(f"Evaluating policy: {policy_name}")
        summary = evaluate_policy(
            env=env,
            policy_name=policy_name,
            num_eval_episodes=num_eval_episodes,
            agent=agent if policy_name == "mappo" else None,
        )
        all_results.append(summary)

    print_summary_table(all_results)
    save_eval_csv(save_dir, all_results)
    plot_all_eval_figures(save_dir, all_results)
    plot_eval_dashboard(save_dir, all_results)
    training_records = load_training_records("results")
    plot_training_convergence_overview(save_dir, training_records)


if __name__ == "__main__":
    main()
