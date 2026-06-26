from __future__ import annotations

import csv
import os
from typing import Any, Dict, Iterable, List, Protocol

import matplotlib.pyplot as plt
import numpy as np

from temporal.env import MECEnv


class Policy(Protocol):
    name: str

    def select_actions(self, env: MECEnv, obs: np.ndarray) -> np.ndarray:
        """Return one action per agent."""


LOAD_CURVE_STYLES: Dict[str, Dict[str, Any]] = {
    "mappo": {"color": "#111827", "linestyle": "-", "marker": "o", "linewidth": 2.8, "zorder": 8},
    "qmix": {"color": "#7C3AED", "linestyle": "-", "marker": "P", "linewidth": 2.2, "zorder": 5},
    "pruned": {"color": "#F97316", "linestyle": "--", "marker": "s", "linewidth": 2.0, "zorder": 6},
    "distilled": {"color": "#16A34A", "linestyle": "-.", "marker": "^", "linewidth": 2.0, "zorder": 7},
    "temporal": {"color": "#2563EB", "linestyle": "-", "marker": "P", "linewidth": 2.2, "zorder": 7},
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

POLICY_COLORS: Dict[str, str] = {
    "mappo": "#111827",
    "pruned_mappo": "#F97316",
    "distilled_mappo": "#16A34A",
    "temporal_distilled_mappo": "#2563EB",
    "offload": "#DC2626",
    "greedy": "#9333EA",
    "local": "#92400E",
    "random": "#DB2777",
}

POLICY_LABELS: Dict[str, str] = {
    "mappo": "MAPPO",
    "pruned_mappo": "Pruned",
    "distilled_mappo": "Distilled",
    "temporal_distilled_mappo": "Temporal",
    "offload": "Offload",
    "greedy": "Greedy",
    "local": "Local",
    "random": "Random",
}


def apply_readable_grid(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, which="major", **GRID_STYLE)


def policy_color(policy_name: str) -> str:
    return POLICY_COLORS.get(policy_name, "#64748B")


def policy_label(policy_name: str) -> str:
    return POLICY_LABELS.get(policy_name, policy_name)


def annotate_bars(ax: plt.Axes, bars: Any, values: List[float], fmt: str) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.025
    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2.0
        if value >= 0:
            y = value + offset
            va = "bottom"
        else:
            y = value - offset
            va = "top"
        ax.text(
            x,
            y,
            fmt.format(value),
            ha="center",
            va=va,
            fontsize=8.5,
            color="#111827",
        )


def set_padded_ylim(ax: plt.Axes, values: List[float], include_zero: bool = True) -> None:
    if not values:
        return
    ymin = min(values)
    ymax = max(values)
    if include_zero:
        ymin = min(ymin, 0.0)
        ymax = max(ymax, 0.0)
    span = ymax - ymin
    pad = max(span * 0.18, 0.08 if span < 1.0 else 0.8)
    ax.set_ylim(ymin - pad, ymax + pad)


def load_curve_style_key(policy_name: str) -> str:
    for prefix in ("temporal", "pruned", "distilled", "random"):
        if policy_name.startswith(prefix):
            return prefix
    return policy_name


def load_curve_draw_order(policies: List[str]) -> List[str]:
    return [name for name in policies if name != "mappo"] + [
        name for name in policies if name == "mappo"
    ]


def run_one_episode(env: MECEnv, policy: Policy) -> Dict[str, float]:
    if hasattr(policy, "reset_episode"):
        policy.reset_episode()

    data = env.reset()
    obs = data["obs"]
    done = False
    step_count = 0

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
        step_count += 1

    base_agent_decisions = float(step_count * env.M)
    base_inference_calls = float(step_count)
    policy_stats: Dict[str, Any] = {}
    if hasattr(policy, "get_episode_stats"):
        policy_stats = dict(policy.get_episode_stats())

    decision_count = float(policy_stats.get("decision_count", base_agent_decisions))
    inference_call_count = float(policy_stats.get("inference_call_count", base_inference_calls))
    reused_action_count = float(policy_stats.get("reused_action_count", 0.0))
    safety_interrupt_count = float(policy_stats.get("safety_interrupt_count", 0.0))
    predicted_repeat_mean = float(policy_stats.get("predicted_repeat_mean", 0.0))

    return {
        "episode_reward": float(episode_reward),
        "episode_delay_mean": float(np.mean(delay_list)) if delay_list else 0.0,
        "episode_drop_rate": float(np.mean(drop_rate_list)) if drop_rate_list else 0.0,
        "episode_offload_rate": float(np.mean(offload_rate_list)) if offload_rate_list else 0.0,
        "episode_energy_mean": float(np.mean(energy_mean_list)) if energy_mean_list else 0.0,
        "episode_steps": float(step_count),
        "base_agent_decisions": base_agent_decisions,
        "decision_count": decision_count,
        "base_inference_calls": base_inference_calls,
        "inference_call_count": inference_call_count,
        "reused_action_count": reused_action_count,
        "safety_interrupt_count": safety_interrupt_count,
        "predicted_repeat_mean": predicted_repeat_mean,
        "agent_temporal_compression_ratio": base_agent_decisions / max(decision_count, 1.0),
        "inference_temporal_compression_ratio": base_inference_calls / max(inference_call_count, 1.0),
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
    steps = []
    base_agent_decisions = []
    decision_counts = []
    base_inference_calls = []
    inference_call_counts = []
    reused_action_counts = []
    safety_interrupt_counts = []
    predicted_repeat_means = []
    agent_compression_ratios = []
    inference_compression_ratios = []

    for _ in range(num_eval_episodes):
        result = run_one_episode(env, policy)
        rewards.append(result["episode_reward"])
        delays.append(result["episode_delay_mean"])
        drops.append(result["episode_drop_rate"])
        offloads.append(result["episode_offload_rate"])
        energies.append(result["episode_energy_mean"])
        steps.append(result["episode_steps"])
        base_agent_decisions.append(result["base_agent_decisions"])
        decision_counts.append(result["decision_count"])
        base_inference_calls.append(result["base_inference_calls"])
        inference_call_counts.append(result["inference_call_count"])
        reused_action_counts.append(result["reused_action_count"])
        safety_interrupt_counts.append(result["safety_interrupt_count"])
        predicted_repeat_means.append(result["predicted_repeat_mean"])
        agent_compression_ratios.append(result["agent_temporal_compression_ratio"])
        inference_compression_ratios.append(result["inference_temporal_compression_ratio"])

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
        "episode_steps_mean": float(np.mean(steps)),
        "base_agent_decisions_mean": float(np.mean(base_agent_decisions)),
        "decision_count_mean": float(np.mean(decision_counts)),
        "base_inference_calls_mean": float(np.mean(base_inference_calls)),
        "inference_call_count_mean": float(np.mean(inference_call_counts)),
        "reused_action_count_mean": float(np.mean(reused_action_counts)),
        "safety_interrupt_count_mean": float(np.mean(safety_interrupt_counts)),
        "predicted_repeat_mean": float(np.mean(predicted_repeat_means)),
        "agent_temporal_compression_ratio_mean": float(np.mean(agent_compression_ratios)),
        "inference_temporal_compression_ratio_mean": float(np.mean(inference_compression_ratios)),
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
        f"{'Policy':<28}"
        f"{'DelayMean':>12}"
        f"{'DelayStd':>12}"
        f"{'DropMean':>12}"
        f"{'OffloadMean':>14}"
        f"{'RewardMean':>14}"
        f"{'Decisions':>12}"
        f"{'AgentCR':>10}"
        f"{'InferCR':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['policy']:<28}"
            f"{r['delay_mean']:>12.4f}"
            f"{r['delay_std']:>12.4f}"
            f"{r['drop_rate_mean']:>12.4f}"
            f"{r['offload_rate_mean']:>14.4f}"
            f"{r['reward_mean']:>14.4f}"
            f"{r['decision_count_mean']:>12.2f}"
            f"{r['agent_temporal_compression_ratio_mean']:>10.3f}"
            f"{r['inference_temporal_compression_ratio_mean']:>10.3f}"
        )

    print("\n====================================================\n")


def plot_bar_dashboard(save_dir: str, results: List[Dict[str, Any]]) -> None:
    if not results:
        return

    os.makedirs(save_dir, exist_ok=True)
    plot_comparison_dashboard(
        save_dir=save_dir,
        results=results,
        filename="comparison_dashboard.png",
        title="Policy Comparison: Performance Cost and Temporal Savings",
    )

    focus_names = {
        "mappo",
        "pruned_mappo",
        "distilled_mappo",
        "temporal_distilled_mappo",
    }
    focus_results = [record for record in results if record["policy"] in focus_names]
    if len(focus_results) >= 2:
        plot_comparison_dashboard(
            save_dir=save_dir,
            results=focus_results,
            filename="compressed_methods_dashboard.png",
            title="Compressed MAPPO Variants: Zoomed Comparison",
        )


def plot_comparison_dashboard(
    save_dir: str,
    results: List[Dict[str, Any]],
    filename: str,
    title: str,
) -> None:
    policies = [r["policy"] for r in results]
    labels = [policy_label(name) for name in policies]
    colors = [policy_color(name) for name in policies]

    mappo_record = next((r for r in results if r["policy"] == "mappo"), results[0])
    mappo_delay = float(mappo_record["delay_mean"])
    mappo_drop = float(mappo_record["drop_rate_mean"])

    delay_delta = [float(r["delay_mean"]) - mappo_delay for r in results]
    drop_delta = [float(r["drop_rate_mean"]) - mappo_drop for r in results]
    agent_saved = [
        (1.0 - float(r["decision_count_mean"]) / max(float(r["base_agent_decisions_mean"]), 1.0))
        * 100.0
        for r in results
    ]
    inference_saved = [
        (
            1.0
            - float(r["inference_call_count_mean"])
            / max(float(r["base_inference_calls_mean"]), 1.0)
        )
        * 100.0
        for r in results
    ]

    panels = [
        ("Delay Gap vs MAPPO", "Delta delay", delay_delta, "{:+.2f}", True),
        ("Drop-Rate Gap vs MAPPO", "Delta drop rate", drop_delta, "{:+.4f}", True),
        ("Per-Agent Decision Saving", "Saving (%)", agent_saved, "{:.1f}%", True),
        ("Slot-Level Inference Saving", "Saving (%)", inference_saved, "{:.1f}%", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    axes = axes.flatten()
    x = np.arange(len(policies))

    for ax, (panel_title, ylabel, values, value_fmt, include_zero) in zip(axes, panels):
        bars = ax.bar(
            x,
            values,
            color=colors,
            edgecolor="#334155",
            linewidth=0.6,
            width=0.68,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.axhline(0.0, color="#475569", linewidth=0.8)
        set_padded_ylim(ax, values, include_zero=include_zero)
        annotate_bars(ax, bars, values, value_fmt)
        apply_readable_grid(ax)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = os.path.join(save_dir, filename)
    fig.savefig(fig_path, dpi=260)
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
        (
            "agent_temporal_compression_ratio_mean",
            None,
            "Agent Temporal Compression Ratio",
            "loads_agent_temporal_compression.png",
        ),
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
            y_std = np.array([r[std_key] for r in matched], dtype=np.float32) if std_key else None
            style = LOAD_CURVE_STYLES.get(
                load_curve_style_key(policy_name),
                {"marker": "o", "linewidth": 2.0, "zorder": 3},
            )

            line = ax.plot(x, y, label=policy_name, **style)[0]
            legend_entries.append((policy_name, line))
            if y_std is not None:
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
