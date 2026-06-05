from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MAPPO_LOG = ROOT_DIR / "results" / "training_log.csv"
DEFAULT_QMIX_LOG = ROOT_DIR / "QMIX" / "results" / "qmix_training_log.csv"
DEFAULT_SAVE_DIR = ROOT_DIR / "QMIX" / "results" / "mappo_qmix_comparison"

MODEL_STYLES = {
    "MAPPO": {
        "color": "#1F77B4",
        "raw_alpha": 0.16,
        "band_alpha": 0.10,
        "linewidth": 2.2,
        "raw_linewidth": 0.75,
        "zorder": 4,
    },
    "QMIX": {
        "color": "#F97316",
        "raw_alpha": 0.16,
        "band_alpha": 0.10,
        "linewidth": 2.2,
        "raw_linewidth": 0.75,
        "zorder": 3,
    },
}

METRICS = [
    ("episode_reward", "Reward", "training_reward"),
    ("episode_delay_mean", "Delay", "training_delay"),
    ("loss_norm", "Normalized Loss", "training_loss"),
    ("episode_energy_mean", "Energy", "training_energy"),
    ("episode_drop_rate", "Drop Rate", "training_drop_rate"),
]


def read_training_log(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Training log not found: {path}")

    rows: List[Dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            rows.append(parsed)
    return rows


def add_mappo_loss(records: List[Dict[str, float]]) -> None:
    for record in records:
        if "total_loss" in record:
            record["loss"] = record["total_loss"]
        elif "actor_loss" in record and "critic_loss" in record:
            record["loss"] = record["actor_loss"] + record["critic_loss"]
        else:
            record["loss"] = np.nan


def add_qmix_loss(records: List[Dict[str, float]]) -> None:
    for record in records:
        record["loss"] = record.get("qmix_loss", np.nan)


def add_normalized_loss(records: List[Dict[str, float]]) -> None:
    losses = np.asarray(
        [
            record.get("loss", np.nan)
            for record in records
            if np.isfinite(record.get("loss", np.nan)) and record.get("loss", np.nan) > 0.0
        ],
        dtype=np.float64,
    )
    if losses.size == 0:
        for record in records:
            record["loss_norm"] = np.nan
        return

    low = float(np.nanpercentile(losses, 1.0))
    high = float(np.nanpercentile(losses, 99.0))
    scale = max(high - low, 1e-12)
    for record in records:
        loss = record.get("loss", np.nan)
        if not np.isfinite(loss) or loss <= 0.0:
            record["loss_norm"] = np.nan
        else:
            record["loss_norm"] = float(np.clip((loss - low) / scale, 0.0, 1.0))


def moving_average(values: Iterable[float], window: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return arr
    if window <= 1:
        return arr

    result = np.empty_like(arr)
    for idx in range(arr.size):
        start = max(0, idx - window + 1)
        result[idx] = np.nanmean(arr[start : idx + 1])
    return result


def rolling_std(values: Iterable[float], window: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return arr
    if window <= 1:
        return np.zeros_like(arr)

    result = np.empty_like(arr)
    for idx in range(arr.size):
        start = max(0, idx - window + 1)
        window_values = arr[start : idx + 1]
        if window_values.size <= 1:
            result[idx] = 0.0
        else:
            result[idx] = np.nanstd(window_values)
    return result


def finite_xy(records: List[Dict[str, float]], metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    episodes = []
    values = []
    for record in records:
        episode = record.get("episode")
        value = record.get(metric_key)
        if episode is None or value is None or not np.isfinite(value):
            continue
        episodes.append(episode)
        values.append(value)
    return np.asarray(episodes, dtype=np.float64), np.asarray(values, dtype=np.float64)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#D5DBE5",
            "grid.linewidth": 0.6,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#CBD5E1",
        }
    )


def plot_metric(
    ax: plt.Axes,
    metric_key: str,
    ylabel: str,
    mappo_records: List[Dict[str, float]],
    qmix_records: List[Dict[str, float]],
    window: int,
) -> None:
    for model_name, records in (("MAPPO", mappo_records), ("QMIX", qmix_records)):
        x, raw = finite_xy(records, metric_key)
        if raw.size == 0:
            continue

        style = MODEL_STYLES[model_name]
        ma = moving_average(raw, window)
        std = rolling_std(raw, window)
        band_center = ma
        lower = band_center - std
        upper = band_center + std

        ax.fill_between(
            x,
            lower,
            upper,
            color=style["color"],
            alpha=style["band_alpha"],
            linewidth=0,
            zorder=style["zorder"] - 2,
        )
        ax.plot(
            x,
            raw,
            color=style["color"],
            alpha=style["raw_alpha"],
            linewidth=style["raw_linewidth"],
            label=f"{model_name} Raw",
            zorder=style["zorder"] - 1,
        )
        ax.plot(
            x,
            ma,
            color=style["color"],
            linewidth=style["linewidth"],
            label=f"{model_name} MA({window})",
            zorder=style["zorder"],
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(True, alpha=0.82)
    ax.legend(loc="best", fontsize=8)


def save_single_metric(
    save_dir: Path,
    metric_key: str,
    ylabel: str,
    filename_stem: str,
    mappo_records: List[Dict[str, float]],
    qmix_records: List[Dict[str, float]],
    window: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_metric(ax, metric_key, ylabel, mappo_records, qmix_records, window)
    fig.tight_layout()
    path = save_dir / f"{filename_stem}.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    print(f"Saved figure: {path}")


def save_raw_loss_figure(
    save_dir: Path,
    mappo_records: List[Dict[str, float]],
    qmix_records: List[Dict[str, float]],
    window: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=False)
    for ax, model_name, records in (
        (axes[0], "MAPPO", mappo_records),
        (axes[1], "QMIX", qmix_records),
    ):
        x, raw = finite_xy(records, "loss")
        if raw.size == 0:
            continue

        style = MODEL_STYLES[model_name]
        ma = moving_average(raw, window)
        std = rolling_std(raw, window)
        ax.fill_between(
            x,
            ma - std,
            ma + std,
            color=style["color"],
            alpha=style["band_alpha"],
            linewidth=0,
        )
        ax.plot(
            x,
            raw,
            color=style["color"],
            alpha=style["raw_alpha"],
            linewidth=style["raw_linewidth"],
            label=f"{model_name} Raw",
        )
        ax.plot(
            x,
            ma,
            color=style["color"],
            linewidth=style["linewidth"],
            label=f"{model_name} MA({window})",
        )
        ax.set_title(f"{model_name} Raw Loss")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.82)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Raw Loss Curves by Algorithm", fontsize=14, fontweight="semibold")
    fig.tight_layout()
    path = save_dir / "training_loss_raw_by_algorithm.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    print(f"Saved figure: {path}")


def plot_overview(
    save_dir: Path,
    mappo_records: List[Dict[str, float]],
    qmix_records: List[Dict[str, float]],
    window: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.flatten()

    for ax, (metric_key, ylabel, _) in zip(axes, METRICS):
        plot_metric(ax, metric_key, ylabel, mappo_records, qmix_records, window)

    axes[-1].axis("off")
    fig.suptitle("MAPPO vs QMIX Training Curves", fontsize=14, fontweight="semibold")
    fig.tight_layout()
    path = save_dir / "mappo_qmix_training_overview.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    print(f"Saved figure: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw MAPPO vs QMIX training-curve comparison.")
    parser.add_argument("--mappo-log", type=Path, default=DEFAULT_MAPPO_LOG)
    parser.add_argument("--qmix-log", type=Path, default=DEFAULT_QMIX_LOG)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--window", type=int, default=50, help="Moving-average and shading window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window <= 0:
        raise ValueError("--window must be positive.")

    configure_style()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    mappo_records = read_training_log(args.mappo_log)
    qmix_records = read_training_log(args.qmix_log)
    add_mappo_loss(mappo_records)
    add_qmix_loss(qmix_records)
    add_normalized_loss(mappo_records)
    add_normalized_loss(qmix_records)

    plot_overview(args.save_dir, mappo_records, qmix_records, args.window)
    for metric_key, ylabel, filename_stem in METRICS:
        save_single_metric(
            args.save_dir,
            metric_key,
            ylabel,
            filename_stem,
            mappo_records,
            qmix_records,
            args.window,
        )
    save_raw_loss_figure(args.save_dir, mappo_records, qmix_records, args.window)


if __name__ == "__main__":
    main()
