from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - plotly is optional at runtime.
    go = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT_DIR / "results" / "compare_prune_rates_loads" / "comparison_loads_summary.csv"
DEFAULT_OUT_DIR = DEFAULT_CSV.parent / "draw_tmp"

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
        "--no-html",
        action="store_true",
        help="Skip interactive Plotly HTML files.",
    )
    parser.add_argument(
        "--show-std",
        action="store_true",
        help="Draw faint standard-deviation bands in static figures.",
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


def prune_level(policy: str) -> int:
    for prefix in METHOD_PREFIXES:
        if policy.startswith(prefix):
            try:
                return int(policy.rsplit("p", 1)[1])
            except (IndexError, ValueError):
                return 0
    return 0


def method_rank(policy: str, focus_policies: Sequence[str]) -> Tuple[int, int, str]:
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
        "color": POLICY_COLORS.get(policy, "#334155"),
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

    html_path = out_dir / f"{spec.stem}_interactive.html"
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
        if spec.zoom_ylim is not None:
            zoom_suffix = f"_zoom_{int(abs(spec.zoom_ylim[0]))}_{int(abs(spec.zoom_ylim[1]))}"
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

        if not args.no_html:
            html_path = plot_interactive_metric(
                records,
                all_policies,
                focus_policies,
                spec,
                args.out_dir,
            )
            if html_path is not None:
                saved_paths.append(html_path)

    print(f"Saved {len(saved_paths)} files to {args.out_dir}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
