from __future__ import annotations

import argparse
import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import MultipleLocator

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - plotly is optional at runtime.
    go = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT_DIR / "results" / "compare_prune_rates_loads" / "comparison_loads_summary.csv"
DEFAULT_OUT_DIR = DEFAULT_CSV.parent / "draw_tmp"
DEFAULT_MODEL_PATH = ROOT_DIR / "results" / "mappo_checkpoint.pt"
DEFAULT_DISTILLED_MODEL_TEMPLATE = str(
    ROOT_DIR / "results" / "mappo_actor_pruned_distilled_{suffix}.pt"
)
DEFAULT_DEPLOY_PRUNE_RATES = "10,25,40,50"

BASELINE_POLICIES = ("greedy", "local", "offload", "random")
DEFAULT_FOCUS_POLICIES = ("distilled_p50", "pruned_p50", "mappo")
METHOD_PREFIXES = ("distilled_", "pruned_")

POLICY_COLORS = {
    "mappo": "#111827",
    "distilled_p10": "#F59E0B",
    "distilled_p25": "#EF4444",
    "distilled_p40": "#DB2777",
    "distilled_p50": "#DC2626",
    "pruned_p10": "#22C55E",
    "pruned_p25": "#10B981",
    "pruned_p40": "#06B6D4",
    "pruned_p50": "#2563EB",
    "greedy": "#7C8A9A",
    "local": "#C0A16B",
    "offload": "#8F969E",
    "random": "#9A8FB8",
}

MATPLOTLIB_LINESTYLES = {
    "mappo": "solid",
    "greedy": (0, (5, 3)),
    "local": (0, (2, 2)),
    "offload": (0, (7, 3)),
    "random": (0, (1, 2)),
}

PLOTLY_LINESTYLES = {
    "mappo": "solid",
    "greedy": "dash",
    "local": "dot",
    "offload": "longdash",
    "random": "dashdot",
}


@dataclass(frozen=True)
class MetricSpec:
    mean_key: str
    std_key: str
    ylabel: str
    stem: str
    y_major_step: float
    y_minor_step: float
    zoom_ylim: Optional[Tuple[float, float]] = None


METRICS = (
    MetricSpec("delay_mean", "delay_std", "Mean Delay", "loads_delay", 5.0, 1.0, (135.0, 160.0)),
    MetricSpec("drop_rate_mean", "drop_rate_std", "Drop Rate", "loads_drop_rate", 0.05, 0.01),
    MetricSpec("offload_rate_mean", "offload_rate_std", "Offload Rate", "loads_offload_rate", 0.05, 0.01),
    MetricSpec("reward_mean", "reward_std", "Reward", "loads_reward", 5.0, 1.0, (-160.0, -135.0)),
)

NUMERIC_FIELDS = {
    "reward_mean",
    "reward_std",
    "delay_mean",
    "delay_std",
    "drop_rate_mean",
    "drop_rate_std",
    "offload_rate_mean",
    "offload_rate_std",
    "energy_mean",
    "energy_std",
    "load_factor",
    "task_min_bits",
    "task_max_bits",
    "avg_task_mbits",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw clearer load-comparison curves from comparison_loads_summary.csv."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input summary CSV.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for figures.")
    parser.add_argument(
        "--focus-policy",
        action="append",
        default=[],
        help="Policy to highlight. Can be passed multiple times. Defaults to p50 methods and MAPPO.",
    )
    parser.add_argument(
        "--main-policy",
        default="mappo",
        help="Original MAPPO policy name used in split comparisons.",
    )
    parser.add_argument(
        "--pruned-policies",
        default="",
        help=(
            "Comma-separated pruned policy names for prune-vs-MAPPO figures. "
            "Defaults to every policy starting with pruned_."
        ),
    )
    parser.add_argument(
        "--distilled-policies",
        default="",
        help=(
            "Comma-separated distilled policy names for distill-vs-MAPPO figures. "
            "Defaults to every policy starting with distilled_."
        ),
    )
    parser.add_argument(
        "--combined-methods",
        action="store_true",
        help="Also draw the old combined pruned+distilled method figures.",
    )
    parser.add_argument(
        "--only-split",
        action="store_true",
        help="Draw only prune-vs-MAPPO and distill-vs-MAPPO load curves.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip interactive Plotly HTML files.",
    )
    parser.add_argument(
        "--show-std",
        action="store_true",
        help="Draw faint standard-deviation bands in static figures.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Original MAPPO checkpoint used for the deployment-size comparison.",
    )
    parser.add_argument(
        "--distilled-model-template",
        default=DEFAULT_DISTILLED_MODEL_TEMPLATE,
        help=(
            "Path template for distilled actors in the deployment-size comparison. "
            "Available fields: {suffix}, {rate}, {percent}, {percent_int}."
        ),
    )
    parser.add_argument(
        "--deploy-prune-rates",
        default=DEFAULT_DEPLOY_PRUNE_RATES,
        help="Comma-separated distilled pruning rates used in the deployment-size comparison.",
    )
    parser.add_argument(
        "--no-model-deploy",
        action="store_true",
        help="Skip the actor parameter/model-size deployment comparison.",
    )
    return parser.parse_args()


def read_records(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records: List[Dict[str, Any]] = []
        for row in reader:
            parsed: Dict[str, Any] = {}
            for key, value in row.items():
                if key in NUMERIC_FIELDS:
                    parsed[key] = float(value)
                else:
                    parsed[key] = value
            records.append(parsed)
    return records


def unique_in_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def parse_policy_list(raw_policies: str) -> List[str]:
    return [item.strip() for item in raw_policies.split(",") if item.strip()]


def choose_group_policies(
    all_policies: Sequence[str],
    main_policy: str,
    prefix: str,
    selected_policies: Sequence[str],
) -> List[str]:
    policy_set = set(all_policies)
    if selected_policies:
        missing = [policy for policy in selected_policies if policy not in policy_set]
        if missing:
            print(f"Skip missing {prefix} policies in CSV: {', '.join(missing)}")
        method_policies = [policy for policy in selected_policies if policy in policy_set]
    else:
        method_policies = [policy for policy in all_policies if policy.startswith(prefix)]

    group: List[str] = []
    if main_policy in policy_set:
        group.append(main_policy)
    elif method_policies:
        print(f"Main policy '{main_policy}' was not found in CSV; drawing methods only.")

    group.extend(policy for policy in method_policies if policy not in group)
    return group


def prune_level(policy: str) -> float:
    for prefix in METHOD_PREFIXES:
        if policy.startswith(prefix):
            try:
                suffix = policy.rsplit("_", 1)[1]
                if suffix.startswith("p"):
                    suffix = suffix[1:]
                return float(suffix.replace("p", "."))
            except (IndexError, ValueError):
                return 0.0
    return 0.0


def method_rank(policy: str, focus_policies: Sequence[str]) -> Tuple[int, float, str]:
    if policy in focus_policies:
        return (0, focus_policies.index(policy), policy)
    if policy.startswith("distilled_"):
        return (1, prune_level(policy), policy)
    if policy.startswith("pruned_"):
        return (2, prune_level(policy), policy)
    if policy in BASELINE_POLICIES:
        return (4, BASELINE_POLICIES.index(policy), policy)
    return (3, 0, policy)


def legend_order(policies: Sequence[str], focus_policies: Sequence[str]) -> List[str]:
    return sorted(policies, key=lambda p: method_rank(p, focus_policies))


def draw_order(policies: Sequence[str], focus_policies: Sequence[str]) -> List[str]:
    baselines = [p for p in policies if p in BASELINE_POLICIES]
    others = [p for p in policies if p not in BASELINE_POLICIES and p not in focus_policies]
    focus = [p for p in policies if p in focus_policies]
    return baselines + others + focus


def records_for_policy(records: Sequence[Dict[str, Any]], policy: str) -> List[Dict[str, Any]]:
    return sorted(
        [record for record in records if record["policy"] == policy],
        key=lambda record: (record["avg_task_mbits"], record["load_factor"]),
    )


def generated_policy_color(policy: str) -> str:
    level = max(0, min(prune_level(policy), 100)) / 100.0
    if policy.startswith("pruned_"):
        rgba = plt.get_cmap("viridis")(0.30 + 0.55 * level)
        return mcolors.to_hex(rgba)
    if policy.startswith("distilled_"):
        rgba = plt.get_cmap("plasma")(0.22 + 0.58 * level)
        return mcolors.to_hex(rgba)
    return "#334155"


def style_for_policy(policy: str, focus_policies: Sequence[str]) -> Dict[str, Any]:
    is_baseline = policy in BASELINE_POLICIES
    is_focus = policy in focus_policies
    is_method = policy.startswith(METHOD_PREFIXES)

    if is_focus:
        linewidth, markersize, alpha, zorder = 1.8, 4.6, 1.0, 5
    elif is_method:
        linewidth, markersize, alpha, zorder = 1.25, 3.6, 0.78, 3
    elif is_baseline:
        linewidth, markersize, alpha, zorder = 1.05, 3.0, 0.50, 2
    else:
        linewidth, markersize, alpha, zorder = 1.15, 3.2, 0.68, 2

    linestyle = MATPLOTLIB_LINESTYLES.get(policy)
    if linestyle is None and policy.startswith("pruned_"):
        linestyle = (0, (4, 2))
    elif linestyle is None:
        linestyle = "solid"

    return {
        "color": POLICY_COLORS.get(policy, generated_policy_color(policy)),
        "linewidth": linewidth,
        "markersize": markersize,
        "alpha": alpha,
        "zorder": zorder,
        "linestyle": linestyle,
    }


def plotly_style_for_policy(policy: str, focus_policies: Sequence[str]) -> Dict[str, Any]:
    style = style_for_policy(policy, focus_policies)
    dash = PLOTLY_LINESTYLES.get(policy)
    if dash is None and policy.startswith("pruned_"):
        dash = "dash"
    elif dash is None:
        dash = "solid"
    return {
        "color": style["color"],
        "width": style["linewidth"] + 0.25,
        "dash": dash,
        "opacity": style["alpha"],
        "markersize": style["markersize"] + 1.0,
    }


def values_for_metric(
    records: Sequence[Dict[str, Any]], policies: Sequence[str], metric_key: str
) -> List[float]:
    policy_set = set(policies)
    return [float(record[metric_key]) for record in records if record["policy"] in policy_set]


def padded_ylim(values: Sequence[float], major_step: float) -> Tuple[float, float]:
    ymin = min(values)
    ymax = max(values)
    span = ymax - ymin
    pad = max(span * 0.06, major_step)
    return ymin - pad, ymax + pad


def configure_axes(ax: plt.Axes, spec: MetricSpec, ylim: Optional[Tuple[float, float]]) -> None:
    ax.set_xlabel("Average Task Size (Mbits)")
    ax.set_ylabel(spec.ylabel)
    ax.set_title(f"{spec.ylabel} under Different Load Levels", pad=12)

    ax.yaxis.set_major_locator(MultipleLocator(spec.y_major_step))
    ax.yaxis.set_minor_locator(MultipleLocator(spec.y_minor_step))
    ax.grid(True, which="major", color="#D5DBE5", linewidth=0.55, alpha=0.82)
    ax.grid(True, which="minor", color="#E8ECF3", linewidth=0.35, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ylim is not None:
        ax.set_ylim(*ylim)


def add_outside_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    policies: Sequence[str],
    focus_policies: Sequence[str],
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    ordered_labels = [label for label in legend_order(policies, focus_policies) if label in handle_by_label]
    ordered_handles = [handle_by_label[label] for label in ordered_labels]

    fig.subplots_adjust(right=0.74, bottom=0.18)
    ax.legend(
        ordered_handles,
        ordered_labels,
        title="Policy",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.95,
        fontsize=8.6,
        title_fontsize=9.2,
    )


def plot_static_metric(
    records: Sequence[Dict[str, Any]],
    policies: Sequence[str],
    focus_policies: Sequence[str],
    spec: MetricSpec,
    out_dir: Path,
    suffix: str = "",
    ylim: Optional[Tuple[float, float]] = None,
    show_std: bool = True,
) -> List[Path]:
    fig, ax = plt.subplots(figsize=(10.2, 5.8))

    x_values = sorted({float(record["avg_task_mbits"]) for record in records})
    ax.set_xticks(x_values)
    ax.set_xticklabels([f"{value:.2f}" for value in x_values], rotation=32, ha="right")

    for policy in draw_order(policies, focus_policies):
        matched = records_for_policy(records, policy)
        if not matched:
            continue

        x = [float(record["avg_task_mbits"]) for record in matched]
        y = [float(record[spec.mean_key]) for record in matched]
        y_std = [float(record[spec.std_key]) for record in matched]
        style = style_for_policy(policy, focus_policies)

        ax.plot(
            x,
            y,
            marker="o",
            label=policy,
            color=style["color"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            alpha=style["alpha"],
            linestyle=style["linestyle"],
            zorder=style["zorder"],
        )

        if show_std and (policy in focus_policies or policy.startswith(METHOD_PREFIXES)):
            lower = [value - std for value, std in zip(y, y_std)]
            upper = [value + std for value, std in zip(y, y_std)]
            ax.fill_between(
                x,
                lower,
                upper,
                color=style["color"],
                alpha=0.035 if policy in focus_policies else 0.02,
                linewidth=0,
                zorder=max(style["zorder"] - 1, 1),
            )

    if ylim is None:
        values = values_for_metric(records, policies, spec.mean_key)
        ylim = padded_ylim(values, spec.y_major_step)

    configure_axes(ax, spec, ylim)
    add_outside_legend(fig, ax, policies, focus_policies)

    png_path = out_dir / f"{spec.stem}{suffix}.png"
    svg_path = out_dir / f"{spec.stem}{suffix}.svg"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, svg_path]


def plot_interactive_metric(
    records: Sequence[Dict[str, Any]],
    policies: Sequence[str],
    focus_policies: Sequence[str],
    spec: MetricSpec,
    out_dir: Path,
    suffix: str = "",
) -> Optional[Path]:
    if go is None:
        return None

    fig = go.Figure()
    rank_by_policy = {
        policy: index
        for index, policy in enumerate(legend_order(policies, focus_policies))
    }

    for policy in draw_order(policies, focus_policies):
        matched = records_for_policy(records, policy)
        if not matched:
            continue

        x = [float(record["avg_task_mbits"]) for record in matched]
        y = [float(record[spec.mean_key]) for record in matched]
        customdata = [
            [float(record["load_factor"]), float(record[spec.std_key])]
            for record in matched
        ]
        style = plotly_style_for_policy(policy, focus_policies)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                customdata=customdata,
                mode="lines+markers",
                name=policy,
                legendrank=rank_by_policy.get(policy, 99),
                opacity=style["opacity"],
                line={
                    "color": style["color"],
                    "width": style["width"],
                    "dash": style["dash"],
                },
                marker={"size": style["markersize"], "color": style["color"]},
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Avg task: %{x:.3f} Mbits<br>"
                    f"{spec.ylabel}: %{{y:.4f}}<br>"
                    "Std: %{customdata[1]:.4f}<br>"
                    "Load factor: %{customdata[0]:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    values = values_for_metric(records, policies, spec.mean_key)
    y0, y1 = padded_ylim(values, spec.y_major_step)
    fig.update_layout(
        template="plotly_white",
        title=f"{spec.ylabel} under Different Load Levels",
        xaxis_title="Average Task Size (Mbits)",
        yaxis_title=spec.ylabel,
        dragmode="zoom",
        hovermode="x unified",
        width=1050,
        height=620,
        margin={"l": 70, "r": 220, "t": 70, "b": 75},
        legend={
            "title": {"text": "Policy"},
            "x": 1.02,
            "y": 1.0,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.92)",
            "bordercolor": "rgba(148,163,184,0.55)",
            "borderwidth": 1,
        },
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=sorted({float(record["avg_task_mbits"]) for record in records}),
        tickformat=".2f",
        showgrid=True,
        gridcolor="rgba(203,213,225,0.75)",
    )
    fig.update_yaxes(
        range=[y0, y1],
        dtick=spec.y_major_step,
        showgrid=True,
        gridcolor="rgba(203,213,225,0.75)",
    )

    html_path = out_dir / f"{spec.stem}{suffix}_interactive.html"
    fig.write_html(
        html_path,
        include_plotlyjs=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {"format": "png", "scale": 3},
        },
    )
    return html_path


def parse_prune_rates(raw_rates: str) -> List[float]:
    rates = []
    for item in raw_rates.split(","):
        item = item.strip()
        if not item:
            continue
        normalized = item[1:].replace("p", ".") if item.lower().startswith("p") else item
        rate = float(normalized)
        if rate > 1.0:
            rate = rate / 100.0
        if not 0.0 <= rate < 1.0:
            raise ValueError("Prune rates must be in [0, 1), or percentages in [0, 100).")
        rates.append(rate)
    return rates


def format_prune_suffix(rate: float) -> str:
    percent = rate * 100.0
    if abs(percent - round(percent)) < 1e-8:
        return f"p{int(round(percent))}"
    return "p" + f"{percent:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def format_rate_path(template: str, rate: float) -> Path:
    suffix = format_prune_suffix(rate)
    percent = rate * 100.0
    path = Path(
        template.format(
            rate=rate,
            percent=percent,
            percent_int=int(round(percent)),
            suffix=suffix,
        )
    )
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def load_torch_checkpoint(path: Path) -> Dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu")


def count_state_dict_params(state_dict: Dict[str, Any]) -> int:
    return int(
        sum(
            tensor.numel()
            for tensor in state_dict.values()
            if hasattr(tensor, "numel")
        )
    )


def state_dict_nbytes(state_dict: Dict[str, Any]) -> int:
    return int(
        sum(
            tensor.numel() * tensor.element_size()
            for tensor in state_dict.values()
            if hasattr(tensor, "numel") and hasattr(tensor, "element_size")
        )
    )


def infer_actor_hidden_dims(state_dict: Dict[str, Any]) -> List[int]:
    linear_weights = [
        tensor
        for key, tensor in state_dict.items()
        if key.endswith(".weight") and getattr(tensor, "ndim", 0) == 2
    ]
    if len(linear_weights) < 2:
        raise ValueError("Cannot infer actor hidden dimensions from checkpoint state_dict.")
    return [int(tensor.shape[0]) for tensor in linear_weights[:-1]]


def infer_actor_io_dims(state_dict: Dict[str, Any]) -> Tuple[int, int]:
    linear_weights = [
        tensor
        for key, tensor in state_dict.items()
        if key.endswith(".weight") and getattr(tensor, "ndim", 0) == 2
    ]
    if not linear_weights:
        raise ValueError("Cannot infer actor input/output dimensions from checkpoint state_dict.")
    return int(linear_weights[0].shape[1]), int(linear_weights[-1].shape[0])


def serialized_actor_size_bytes(
    state_dict: Dict[str, Any],
    hidden_dims: Sequence[int],
    obs_dim: int,
    n_actions: int,
    n_agents: int,
) -> int:
    import torch

    payload = {
        "actor": state_dict,
        "actor_hidden_dims": list(hidden_dims),
        "obs_dim": int(obs_dim),
        "n_actions": int(n_actions),
        "n_agents": int(n_agents),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.tell()


def build_model_record(
    *,
    model: str,
    display_name: str,
    prune_rate: float,
    path: Path,
    state_dict: Dict[str, Any],
    hidden_dims: Sequence[int],
    obs_dim: int,
    n_actions: int,
    n_agents: int,
    original_params: int,
    original_deploy_size_bytes: int,
) -> Dict[str, Any]:
    params = count_state_dict_params(state_dict)
    param_bytes = state_dict_nbytes(state_dict)
    deploy_size_bytes = serialized_actor_size_bytes(
        state_dict=state_dict,
        hidden_dims=hidden_dims,
        obs_dim=obs_dim,
        n_actions=n_actions,
        n_agents=n_agents,
    )
    return {
        "model": model,
        "display_name": display_name,
        "prune_rate": prune_rate,
        "hidden_dims": "-".join(str(dim) for dim in hidden_dims),
        "params": params,
        "param_memory_mb": param_bytes / (1024.0**2),
        "deployment_size_mb": deploy_size_bytes / (1024.0**2),
        "actual_checkpoint_mb": path.stat().st_size / (1024.0**2),
        "param_compression_x": original_params / max(params, 1),
        "size_compression_x": original_deploy_size_bytes / max(deploy_size_bytes, 1),
        "param_reduction_pct": (1.0 - params / max(original_params, 1)) * 100.0,
        "size_reduction_pct": (
            1.0 - deploy_size_bytes / max(original_deploy_size_bytes, 1)
        )
        * 100.0,
        "path": str(path),
    }


def build_model_deployment_records(
    model_path: Path,
    distilled_model_template: str,
    prune_rates: Sequence[float],
) -> List[Dict[str, Any]]:
    model_path = model_path if model_path.is_absolute() else ROOT_DIR / model_path
    if not model_path.exists():
        print(f"Skip deployment comparison: missing original model {model_path}")
        return []

    original_ckpt = load_torch_checkpoint(model_path)
    original_state = original_ckpt["actor"]
    original_hidden_dims = infer_actor_hidden_dims(original_state)
    original_obs_dim, original_n_actions = infer_actor_io_dims(original_state)
    original_n_agents = int(original_ckpt.get("n_agents", 4))
    original_params = count_state_dict_params(original_state)
    original_deploy_size_bytes = serialized_actor_size_bytes(
        state_dict=original_state,
        hidden_dims=original_hidden_dims,
        obs_dim=original_obs_dim,
        n_actions=original_n_actions,
        n_agents=original_n_agents,
    )

    records = [
        build_model_record(
            model="original_mappo_actor",
            display_name="Original MAPPO Actor",
            prune_rate=0.0,
            path=model_path,
            state_dict=original_state,
            hidden_dims=original_hidden_dims,
            obs_dim=original_obs_dim,
            n_actions=original_n_actions,
            n_agents=original_n_agents,
            original_params=original_params,
            original_deploy_size_bytes=original_deploy_size_bytes,
        )
    ]

    for rate in prune_rates:
        suffix = format_prune_suffix(rate)
        path = format_rate_path(distilled_model_template, rate)
        if not path.exists():
            print(f"Skip missing distilled actor for {suffix}: {path}")
            continue

        ckpt = load_torch_checkpoint(path)
        state = ckpt["actor"]
        hidden_dims = ckpt.get("actor_hidden_dims") or infer_actor_hidden_dims(state)
        obs_dim = int(ckpt.get("obs_dim", original_obs_dim))
        n_actions = int(ckpt.get("n_actions", original_n_actions))
        n_agents = int(ckpt.get("n_agents", original_n_agents))
        percent = int(round(rate * 100.0))
        records.append(
            build_model_record(
                model=f"distilled_{suffix}",
                display_name=f"Distilled {percent}%",
                prune_rate=rate,
                path=path,
                state_dict=state,
                hidden_dims=hidden_dims,
                obs_dim=obs_dim,
                n_actions=n_actions,
                n_agents=n_agents,
                original_params=original_params,
                original_deploy_size_bytes=original_deploy_size_bytes,
            )
        )

    return records


def save_model_deployment_csv(records: Sequence[Dict[str, Any]], out_dir: Path) -> Path:
    csv_path = out_dir / "model_deployment_comparison.csv"
    fieldnames = [
        "model",
        "display_name",
        "prune_rate",
        "hidden_dims",
        "params",
        "param_memory_mb",
        "deployment_size_mb",
        "actual_checkpoint_mb",
        "param_compression_x",
        "size_compression_x",
        "param_reduction_pct",
        "size_reduction_pct",
        "path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return csv_path


def annotate_bar(
    ax: plt.Axes,
    bar: Any,
    value_label: str,
    reduction_pct: float,
    show_reduction: bool,
) -> None:
    height = bar.get_height()
    ax.annotate(
        value_label,
        xy=(bar.get_x() + bar.get_width() / 2.0, height),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#111827",
    )
    if show_reduction:
        ax.annotate(
            f"-{reduction_pct:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#047857",
        )


def plot_model_deployment_comparison(
    records: Sequence[Dict[str, Any]],
    out_dir: Path,
) -> List[Path]:
    if not records:
        return []

    labels = [record["display_name"].replace(" ", "\n") for record in records]
    colors = [
        POLICY_COLORS.get(record["model"].replace("original_mappo_actor", "mappo"), "#334155")
        for record in records
    ]
    for idx, record in enumerate(records):
        if str(record["model"]).startswith("distilled_"):
            colors[idx] = POLICY_COLORS.get(str(record["model"]), "#EF4444")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    metrics = [
        (
            "params",
            "Actor Parameters",
            "Parameters",
            lambda value: f"{value / 1000.0:.1f}K",
            "param_reduction_pct",
        ),
        (
            "deployment_size_mb",
            "Actor Deployment Size",
            "Actor-only checkpoint size (MB)",
            lambda value: f"{value * 1024.0:.1f} KB",
            "size_reduction_pct",
        ),
    ]

    for ax, (key, title, ylabel, label_fn, reduction_key) in zip(axes, metrics):
        values = [float(record[key]) for record in records]
        bars = ax.bar(range(len(records)), values, color=colors, width=0.66)
        ax.set_title(title, pad=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(records)))
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(True, axis="y", color="#E2E8F0", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, max(values) * 1.34)

        for idx, (bar, record) in enumerate(zip(bars, records)):
            annotate_bar(
                ax,
                bar,
                label_fn(values[idx]),
                float(record[reduction_key]),
                show_reduction=idx > 0,
            )

    fig.suptitle("Deployment Cost: Original Actor vs Distilled Pruned Actors", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "Actor-only deployment payload is shown because edge devices only need the decision network.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))

    png_path = out_dir / "model_deployment_comparison.png"
    svg_path = out_dir / "model_deployment_comparison.svg"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, svg_path]


def main() -> None:
    args = parse_args()
    records = read_records(args.csv)
    if not records:
        raise SystemExit(f"No records found in {args.csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_policies = unique_in_order(str(record["policy"]) for record in records)
    focus_policies = tuple(args.focus_policy) if args.focus_policy else DEFAULT_FOCUS_POLICIES
    focus_policies = tuple(policy for policy in focus_policies if policy in all_policies)
    method_policies = [
        policy
        for policy in all_policies
        if policy not in BASELINE_POLICIES
    ]
    split_groups = [
        (
            "prune_vs_mappo",
            choose_group_policies(
                all_policies,
                args.main_policy,
                "pruned_",
                parse_policy_list(args.pruned_policies),
            ),
        ),
        (
            "distill_vs_mappo",
            choose_group_policies(
                all_policies,
                args.main_policy,
                "distilled_",
                parse_policy_list(args.distilled_policies),
            ),
        ),
    ]

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 8.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    saved_paths: List[Path] = []
    show_std = args.show_std
    for spec in METRICS:
        if not args.only_split:
            saved_paths.extend(
                plot_static_metric(
                    records,
                    all_policies,
                    focus_policies,
                    spec,
                    args.out_dir,
                    show_std=show_std,
                )
            )

        if args.combined_methods and not args.only_split:
            saved_paths.extend(
                plot_static_metric(
                    records,
                    method_policies,
                    focus_policies,
                    spec,
                    args.out_dir,
                    suffix="_methods_zoom",
                    show_std=show_std,
                )
            )

        for group_name, group_policies in split_groups:
            if len(group_policies) < 2:
                continue
            group_focus = tuple(policy for policy in focus_policies if policy in group_policies)
            if args.main_policy in group_policies and args.main_policy not in group_focus:
                group_focus = (args.main_policy, *group_focus)
            saved_paths.extend(
                plot_static_metric(
                    records,
                    group_policies,
                    group_focus,
                    spec,
                    args.out_dir,
                    suffix=f"_{group_name}",
                    show_std=show_std,
                )
            )

        if spec.zoom_ylim is not None:
            zoom_suffix = f"_zoom_{int(abs(spec.zoom_ylim[0]))}_{int(abs(spec.zoom_ylim[1]))}"
            if not args.only_split:
                saved_paths.extend(
                    plot_static_metric(
                        records,
                        all_policies,
                        focus_policies,
                        spec,
                        args.out_dir,
                        suffix=zoom_suffix,
                        ylim=spec.zoom_ylim,
                        show_std=show_std,
                    )
                )
            for group_name, group_policies in split_groups:
                if len(group_policies) < 2:
                    continue
                group_focus = tuple(policy for policy in focus_policies if policy in group_policies)
                if args.main_policy in group_policies and args.main_policy not in group_focus:
                    group_focus = (args.main_policy, *group_focus)
                saved_paths.extend(
                    plot_static_metric(
                        records,
                        group_policies,
                        group_focus,
                        spec,
                        args.out_dir,
                        suffix=f"_{group_name}{zoom_suffix}",
                        ylim=spec.zoom_ylim,
                        show_std=show_std,
                    )
                )

        if not args.no_html:
            if not args.only_split:
                html_path = plot_interactive_metric(
                    records,
                    all_policies,
                    focus_policies,
                    spec,
                    args.out_dir,
                )
                if html_path is not None:
                    saved_paths.append(html_path)
            for group_name, group_policies in split_groups:
                if len(group_policies) < 2:
                    continue
                group_focus = tuple(policy for policy in focus_policies if policy in group_policies)
                if args.main_policy in group_policies and args.main_policy not in group_focus:
                    group_focus = (args.main_policy, *group_focus)
                html_path = plot_interactive_metric(
                    records,
                    group_policies,
                    group_focus,
                    spec,
                    args.out_dir,
                    suffix=f"_{group_name}",
                )
                if html_path is not None:
                    saved_paths.append(html_path)

    if not args.no_model_deploy and not args.only_split:
        deploy_records = build_model_deployment_records(
            model_path=args.model_path,
            distilled_model_template=args.distilled_model_template,
            prune_rates=parse_prune_rates(args.deploy_prune_rates),
        )
        if deploy_records:
            saved_paths.append(save_model_deployment_csv(deploy_records, args.out_dir))
            saved_paths.extend(plot_model_deployment_comparison(deploy_records, args.out_dir))

    print(f"Saved {len(saved_paths)} files to {args.out_dir}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
