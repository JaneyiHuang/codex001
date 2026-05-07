# eval_loads.py
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
from dataclasses import replace
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

from config import EnvConfig
from env import MECEnv
from mappo import MAPPO


POLICY_ORDER = ["mappo", "local", "offload", "random", "greedy"]
POLICY_COLORS = {
    "mappo": "#d62728",
    "local": "#7f7f7f",
    "offload": "#2ca02c",
    "random": "#1f77b4",
    "greedy": "#ff7f0e",
}


# =========================================================
# Helper: Greedy baseline
# =========================================================
def greedy_action_for_agent(env: MECEnv, m: int) -> int:
    cfg = env.cfg

    task_bits = float(env.lambda_arr[m])
    E_m = float(env.E_arr[m])
    h_m = float(env.h_arr[m])

    # local delay estimate
    e_loc = env.local_energy_cost()
    if E_m >= e_loc and cfg.f_local > 0:
        T_loc = env.local_processing_slots(task_bits)
    else:
        T_loc = cfg.psi

    S_loc = max(float(env.t), float(env.F_loc[m]))
    est_local_delay = (S_loc + T_loc - 1) - float(env.t)

    # offload delay estimate
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
# Run one episode
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

    while not done:
        if policy_name == "mappo":
            if agent is None:
                raise ValueError("agent must be provided for mappo policy")
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
        reward = out["reward"]
        info = out["info"]

        episode_reward += reward
        delay_list.append(info["delay_mean"])
        drop_rate_list.append(info["drop_rate"])
        offload_rate_list.append(info["offload_rate"])

    return {
        "episode_reward": float(episode_reward),
        "episode_delay_mean": float(np.mean(delay_list)) if delay_list else 0.0,
        "episode_drop_rate": float(np.mean(drop_rate_list)) if drop_rate_list else 0.0,
        "episode_offload_rate": float(np.mean(offload_rate_list)) if offload_rate_list else 0.0,
    }


# =========================================================
# Evaluate one policy under one config
# =========================================================
def evaluate_policy(
    cfg: EnvConfig,
    policy_name: str,
    num_eval_episodes: int,
    agent: MAPPO | None = None,
) -> Dict[str, float]:
    env = MECEnv(cfg)
    env.seed(123)

    rewards = []
    delays = []
    drops = []
    offloads = []

    for _ in range(num_eval_episodes):
        result = run_one_episode(env, policy_name=policy_name, agent=agent)
        rewards.append(result["episode_reward"])
        delays.append(result["episode_delay_mean"])
        drops.append(result["episode_drop_rate"])
        offloads.append(result["episode_offload_rate"])

    return {
        "policy": policy_name,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "delay_mean": float(np.mean(delays)),
        "delay_std": float(np.std(delays)),
        "drop_rate_mean": float(np.mean(drops)),
        "drop_rate_std": float(np.std(drops)),
        "offload_rate_mean": float(np.mean(offloads)),
        "offload_rate_std": float(np.std(offloads)),
    }


# =========================================================
# Save CSV
# =========================================================
def save_csv(save_dir: str, records: List[Dict[str, Any]], filename: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    if len(records) == 0:
        return

    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved CSV: {path}")


# =========================================================
# Plotting
# =========================================================
def plot_metric_across_loads(
    save_dir: str,
    all_results: List[Dict[str, Any]],
    metric_key: str,
    metric_std_key: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for policy in POLICY_ORDER:
        matched = sorted(
            [r for r in all_results if r["policy"] == policy],
            key=lambda r: r["load_factor"]
        )
        x = np.array([r["avg_task_mbits"] for r in matched], dtype=np.float32)
        y = np.array([r[metric_key] for r in matched], dtype=np.float32)
        y_std = np.array([r[metric_std_key] for r in matched], dtype=np.float32)
        color = POLICY_COLORS[policy]

        plt.plot(
            x,
            y,
            marker="o",
            markersize=5,
            linewidth=2.4,
            color=color,
            label=policy,
            zorder=3 if policy == "mappo" else 2,
        )
        plt.fill_between(
            x,
            y - y_std,
            y + y_std,
            color=color,
            alpha=0.10 if policy == "mappo" else 0.08,
            zorder=1,
        )

    plt.xlabel("Average Task Size (Mbits)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(save_dir, filename)
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"Saved figure: {fig_path}")


def plot_loads_dashboard(save_dir: str, all_results: List[Dict[str, Any]]) -> None:
    os.makedirs(save_dir, exist_ok=True)

    metrics = [
        ("delay_mean", "delay_std", "Mean Delay", "Delay vs Load"),
        ("drop_rate_mean", "drop_rate_std", "Drop Rate", "Drop Rate vs Load"),
        ("reward_mean", "reward_std", "Reward", "Reward vs Load"),
        ("offload_rate_mean", "offload_rate_std", "Offload Rate", "Offload Rate vs Load"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (metric_key, metric_std_key, ylabel, title) in zip(axes, metrics):
        for policy in POLICY_ORDER:
            matched = sorted(
                [r for r in all_results if r["policy"] == policy],
                key=lambda r: r["load_factor"]
            )
            x = np.array([r["avg_task_mbits"] for r in matched], dtype=np.float32)
            y = np.array([r[metric_key] for r in matched], dtype=np.float32)
            y_std = np.array([r[metric_std_key] for r in matched], dtype=np.float32)
            color = POLICY_COLORS[policy]

            ax.plot(
                x,
                y,
                marker="o",
                markersize=4.5,
                linewidth=2.2,
                color=color,
                label=policy,
                zorder=3 if policy == "mappo" else 2,
            )
            ax.fill_between(
                x,
                y - y_std,
                y + y_std,
                color=color,
                alpha=0.09 if policy == "mappo" else 0.07,
                zorder=1,
            )

        ax.set_xlabel("Average Task Size (Mbits)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle("Policy Performance under Dense Load Scan", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    fig_path = os.path.join(save_dir, "loads_dashboard.png")
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    print(f"Saved figure: {fig_path}")


# =========================================================
# Main
# =========================================================
def main():
    base_cfg = EnvConfig()

    model_path = os.path.join("results", "mappo_checkpoint.pt")
    save_dir = os.path.join("results", "eval_loads")
    num_eval_episodes = 50

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Cannot find trained model: {model_path}\nPlease run train.py first."
        )

    # Build agent and load model
    agent = MAPPO(
        obs_dim=base_cfg.obs_dim,
        state_dim=base_cfg.state_dim,
        n_actions=base_cfg.n_actions,
        n_agents=base_cfg.M,
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
    agent.load(model_path)
    print(f"Loaded trained MAPPO model from: {model_path}")

    # Dense load scan with a wider default range.
    load_factor_min = float(os.getenv("LOAD_FACTOR_MIN", "0.70"))
    load_factor_max = float(os.getenv("LOAD_FACTOR_MAX", "1.30"))
    num_load_points = int(os.getenv("NUM_LOAD_POINTS", "13"))
    load_factors = np.linspace(load_factor_min, load_factor_max, num_load_points)

    all_results = []

    for idx, load_factor in enumerate(load_factors, start=1):
        task_min_bits = base_cfg.task_min_bits * float(load_factor)
        task_max_bits = base_cfg.task_max_bits * float(load_factor)
        cfg = replace(
            base_cfg,
            task_min_bits=task_min_bits,
            task_max_bits=task_max_bits,
        )
        load_name = f"load_{idx:02d}"
        avg_task_mbits = 0.5 * (task_min_bits + task_max_bits) / 1e6

        print(f"\n========== Evaluating load factor: {load_factor:.2f} ==========")
        print(f"task_min_bits={task_min_bits:.1e}, task_max_bits={task_max_bits:.1e}")

        for policy_name in POLICY_ORDER:
            print(f"Evaluating policy: {policy_name}")
            result = evaluate_policy(
                cfg=cfg,
                policy_name=policy_name,
                num_eval_episodes=num_eval_episodes,
                agent=agent if policy_name == "mappo" else None,
            )
            result["load_name"] = load_name
            result["load_factor"] = float(load_factor)
            result["task_min_bits"] = task_min_bits
            result["task_max_bits"] = task_max_bits
            result["avg_task_mbits"] = avg_task_mbits
            all_results.append(result)

    # Save all results
    save_csv(save_dir, all_results, "eval_loads_summary.csv")

    # Plot key metrics across loads
    plot_metric_across_loads(
        save_dir=save_dir,
        all_results=all_results,
        metric_key="delay_mean",
        metric_std_key="delay_std",
        ylabel="Mean Delay",
        title="Mean Delay under Different Load Levels",
        filename="loads_delay.png",
    )

    plot_metric_across_loads(
        save_dir=save_dir,
        all_results=all_results,
        metric_key="drop_rate_mean",
        metric_std_key="drop_rate_std",
        ylabel="Drop Rate",
        title="Drop Rate under Different Load Levels",
        filename="loads_drop_rate.png",
    )

    plot_metric_across_loads(
        save_dir=save_dir,
        all_results=all_results,
        metric_key="offload_rate_mean",
        metric_std_key="offload_rate_std",
        ylabel="Offload Rate",
        title="Offload Rate under Different Load Levels",
        filename="loads_offload_rate.png",
    )

    plot_metric_across_loads(
        save_dir=save_dir,
        all_results=all_results,
        metric_key="reward_mean",
        metric_std_key="reward_std",
        ylabel="Reward",
        title="Reward under Different Load Levels",
        filename="loads_reward.png",
    )

    plot_loads_dashboard(save_dir, all_results)

    print("\nDone. Multi-load evaluation results have been saved.")


if __name__ == "__main__":
    main()
