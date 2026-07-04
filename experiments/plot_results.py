"""Plot Hive ABM sweep outputs.

Examples from the project root:
    python experiments/plot_results.py --input-dir outputs/sweeps
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


TOPOLOGY_ORDER = ["centralized", "hierarchical", "decentralized_mesh", "adaptive"]
POLICY_ORDER = ["permissive", "moderate", "strict", "mission_critical"]
PLOT_DPI = 240
BLUE = "#2563eb"
GREEN = "#047857"
RED = "#dc2626"
PURPLE = "#7c3aed"
ORANGE = "#d97706"
GRID = "#e5e7eb"


CAPTIONS = {
    "density_completion_rate.png": "Final mission completion rate as node density increases. Interpret the x-axis as an abstract agent-population parameter; normalized density and tasks-per-node should be used for thesis claims until calibrated against deployed Hive logs.",
    "density_peak_congestion.png": "Final peak congestion as node count changes. This plot is intended to detect carrying-capacity regimes: under-provisioning, efficient operating range, and overload.",
    "density_net_contribution.png": "Mean net contribution proxy by node count, combining resource contribution, completed work, imposed load, and failed attempts.",
    "density_completion_by_tasks_per_node.png": "Completion rate plotted against tasks per node, a more interpretable normalized load measure than raw node count.",
    "density_congestion_by_nodes_per_sector.png": "Peak congestion plotted against nodes per grid sector, a normalized spatial-density measure.",
    "topology_policy_completion_heatmap.png": "Mean final completion rate by orchestration topology and policy strictness.",
    "topology_policy_verified_heatmap.png": "Mean verified completion rate by topology and policy. This is especially important under mission-critical policy.",
    "topology_policy_denials_heatmap.png": "Denied actions by topology and policy, indicating authorization friction and blocked work.",
    "communication_loss_completion.png": "Completion under message-loss probability and assignment horizon. This approximates communication fragility.",
    "communication_loss_messages.png": "Total messages under message-loss probability and assignment horizon, showing coordination overhead.",
    "verification_duration_verified_completion.png": "Verified completion as verification duration increases, measuring verification bottleneck pressure.",
    "verification_duration_queue.png": "Final verification backlog as verification duration increases.",
    "timeseries_completion_by_experiment.png": "Mean completion trajectories by experiment family.",
    "timeseries_congestion_by_experiment.png": "Mean congestion trajectories by experiment family.",
    "timeseries_verification_queue.png": "Verification queue trajectory under verification-duration conditions.",
    "environment_completion.png": "Completion rate across spatial environment presets. This tests whether topology/policy effects survive clustered tasks, noisy regions, resource scarcity, and corridor structure.",
    "environment_congestion.png": "Peak congestion across spatial environment presets.",
}


def ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d1d5db",
            "axes.labelcolor": "#111827",
            "axes.titlecolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.7,
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.0,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def ci95(series: pd.Series) -> float:
    n = series.count()
    if n <= 1:
        return 0.0
    return 1.96 * float(series.std(ddof=1)) / math.sqrt(n)


def plot_line_ci(
    df: pd.DataFrame,
    x: str,
    y: str,
    output: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    group: Optional[str] = None,
) -> None:
    if df.empty or x not in df or y not in df:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    if group is None:
        grouped = df.groupby(x, dropna=True)[y].agg(["mean", "count", "std"]).reset_index()
        grouped["ci"] = df.groupby(x, dropna=True)[y].apply(ci95).values
        grouped = grouped.sort_values(x)
        ax.plot(grouped[x], grouped["mean"], marker="o", linewidth=2.2, color=BLUE)
        ax.fill_between(grouped[x], grouped["mean"] - grouped["ci"], grouped["mean"] + grouped["ci"], color=BLUE, alpha=0.16, label="95% CI")
    else:
        colors = [BLUE, GREEN, RED, PURPLE, ORANGE, "#0284c7"]
        for idx, (group_value, part) in enumerate(df.groupby(group, dropna=True)):
            grouped = part.groupby(x, dropna=True)[y].agg(["mean", "count", "std"]).reset_index()
            grouped["ci"] = part.groupby(x, dropna=True)[y].apply(ci95).values
            grouped = grouped.sort_values(x)
            color = colors[idx % len(colors)]
            ax.plot(grouped[x], grouped["mean"], marker="o", linewidth=2.0, color=color, label=str(group_value))
            ax.fill_between(grouped[x], grouped["mean"] - grouped["ci"], grouped["mean"] + grouped["ci"], color=color, alpha=0.10)
        ax.legend(title=group.replace("_", " "), frameon=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def plot_heatmap(
    df: pd.DataFrame,
    value: str,
    output: Path,
    title: str,
    cmap: str = "viridis",
) -> None:
    required = {"topology", "policy_level", value}
    if df.empty or not required.issubset(df.columns):
        return
    pivot = df.pivot_table(index="policy_level", columns="topology", values=value, aggfunc="mean")
    pivot = pivot.reindex(index=POLICY_ORDER, columns=TOPOLOGY_ORDER)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    image = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            value_at_cell = pivot.iloc[r, c]
            if pd.notna(value_at_cell):
                color = "white" if value_at_cell < pivot.stack().median() else "#111827"
                ax.text(c, r, f"{value_at_cell:.2f}", ha="center", va="center", color=color, fontsize=8, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def plot_bar_ci(df: pd.DataFrame, x: str, y: str, output: Path, title: str, xlabel: str, ylabel: str) -> None:
    if df.empty or x not in df or y not in df:
        return
    grouped = df.groupby(x, dropna=True)[y].agg(["mean", "count", "std"]).reset_index()
    grouped["ci"] = df.groupby(x, dropna=True)[y].apply(ci95).values
    grouped = grouped.sort_values(x)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(grouped[x].astype(str), grouped["mean"], yerr=grouped["ci"], capsize=4, color=PURPLE, alpha=0.84)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def plot_time_series(
    ticks: pd.DataFrame,
    y: str,
    output: Path,
    title: str,
    ylabel: str,
    group: str = "experiment",
) -> None:
    required = {"step", y, group}
    if ticks.empty or not required.issubset(ticks.columns):
        return
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = [BLUE, GREEN, RED, PURPLE, ORANGE, "#0284c7"]
    for idx, (group_value, part) in enumerate(ticks.groupby(group, dropna=True)):
        grouped = part.groupby("step", dropna=True)[y].mean().reset_index()
        ax.plot(grouped["step"], grouped[y], linewidth=2.0, color=colors[idx % len(colors)], label=str(group_value))
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title=group.replace("_", " "), frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def plot_all(input_dir: Path, output_dir: Path) -> None:
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries_path = input_dir / "summaries_flat.csv"
    ticks_path = input_dir / "ticks.csv"
    if not summaries_path.exists():
        raise FileNotFoundError(f"Missing {summaries_path}")

    summaries = pd.read_csv(summaries_path)
    ensure_numeric(
        summaries,
        [
            "final_completion_rate",
            "final_verified_completion_rate",
            "final_peak_congestion",
            "final_mean_congestion",
            "total_messages",
            "total_denied_actions",
            "total_verification_attempts",
            "n_nodes_sweep",
            "message_loss_sweep",
            "assignment_horizon_sweep",
            "verification_duration_max_sweep",
            "tasks_per_node",
            "nodes_per_sector",
            "mean_net_contribution",
        ],
    )

    density = summaries[summaries["experiment"] == "density"] if "experiment" in summaries else pd.DataFrame()
    plot_line_ci(
        density,
        "n_nodes_sweep",
        "final_completion_rate",
        output_dir / "density_completion_rate.png",
        "Node Density vs Completion",
        "Nodes",
        "Final completion rate",
    )
    plot_line_ci(
        density,
        "n_nodes_sweep",
        "final_peak_congestion",
        output_dir / "density_peak_congestion.png",
        "Node Density vs Peak Congestion",
        "Nodes",
        "Final peak congestion",
    )
    plot_line_ci(
        density,
        "n_nodes_sweep",
        "mean_net_contribution",
        output_dir / "density_net_contribution.png",
        "Node Density vs Net Contribution",
        "Nodes",
        "Mean net contribution",
    )
    plot_line_ci(
        density,
        "tasks_per_node",
        "final_completion_rate",
        output_dir / "density_completion_by_tasks_per_node.png",
        "Task Load vs Completion",
        "Tasks per node",
        "Final completion rate",
    )
    plot_line_ci(
        density,
        "nodes_per_sector",
        "final_peak_congestion",
        output_dir / "density_congestion_by_nodes_per_sector.png",
        "Spatial Density vs Peak Congestion",
        "Nodes per sector",
        "Final peak congestion",
    )

    topology_policy = summaries[summaries["experiment"] == "topology_policy"] if "experiment" in summaries else pd.DataFrame()
    plot_heatmap(
        topology_policy,
        "final_completion_rate",
        output_dir / "topology_policy_completion_heatmap.png",
        "Topology x Policy: Completion",
    )
    plot_heatmap(
        topology_policy,
        "final_verified_completion_rate",
        output_dir / "topology_policy_verified_heatmap.png",
        "Topology x Policy: Verified Completion",
    )
    plot_heatmap(
        topology_policy,
        "total_denied_actions",
        output_dir / "topology_policy_denials_heatmap.png",
        "Topology x Policy: Denied Actions",
        cmap="magma",
    )

    communication = summaries[summaries["experiment"] == "communication"] if "experiment" in summaries else pd.DataFrame()
    plot_line_ci(
        communication,
        "message_loss_sweep",
        "final_completion_rate",
        output_dir / "communication_loss_completion.png",
        "Message Loss vs Completion",
        "Message loss probability",
        "Final completion rate",
        group="assignment_horizon_sweep",
    )
    plot_line_ci(
        communication,
        "message_loss_sweep",
        "total_messages",
        output_dir / "communication_loss_messages.png",
        "Message Loss vs Messages",
        "Message loss probability",
        "Total messages",
        group="assignment_horizon_sweep",
    )

    verification = summaries[summaries["experiment"] == "verification"] if "experiment" in summaries else pd.DataFrame()
    plot_line_ci(
        verification,
        "verification_duration_max_sweep",
        "final_verified_completion_rate",
        output_dir / "verification_duration_verified_completion.png",
        "Verification Duration vs Verified Completion",
        "Max verification duration",
        "Final verified completion rate",
    )
    plot_line_ci(
        verification,
        "verification_duration_max_sweep",
        "final_awaiting_verification",
        output_dir / "verification_duration_queue.png",
        "Verification Duration vs Queue Backlog",
        "Max verification duration",
        "Final awaiting-verification tasks",
    )

    environment = summaries[summaries["experiment"] == "environment"] if "experiment" in summaries else pd.DataFrame()
    plot_bar_ci(
        environment,
        "environment_preset",
        "final_completion_rate",
        output_dir / "environment_completion.png",
        "Environment Preset vs Completion",
        "Environment preset",
        "Final completion rate",
    )
    plot_bar_ci(
        environment,
        "environment_preset",
        "final_peak_congestion",
        output_dir / "environment_congestion.png",
        "Environment Preset vs Peak Congestion",
        "Environment preset",
        "Final peak congestion",
    )

    if ticks_path.exists():
        ticks = pd.read_csv(ticks_path)
        ensure_numeric(
            ticks,
            [
                "step",
                "completion_rate",
                "verified_completion_rate",
                "mean_congestion",
                "messages_sent",
                "awaiting_verification_tasks",
                "verification_duration_max_sweep",
            ],
        )
        plot_time_series(
            ticks,
            "completion_rate",
            output_dir / "timeseries_completion_by_experiment.png",
            "Completion Over Time",
            "Completion rate",
        )
        plot_time_series(
            ticks,
            "mean_congestion",
            output_dir / "timeseries_congestion_by_experiment.png",
            "Congestion Over Time",
            "Mean congestion",
        )
        verification_ticks = ticks[ticks["experiment"] == "verification"] if "experiment" in ticks else pd.DataFrame()
        plot_time_series(
            verification_ticks,
            "awaiting_verification_tasks",
            output_dir / "timeseries_verification_queue.png",
            "Verification Queue Over Time",
            "Awaiting-verification tasks",
            group="verification_duration_max_sweep",
        )
    write_captions(output_dir)
    print(f"Wrote plots to {output_dir}")


def write_captions(output_dir: Path) -> None:
    lines = ["# Figure Captions", ""]
    for name, caption in CAPTIONS.items():
        if (output_dir / name).exists():
            lines.append(f"## {name}")
            lines.append("")
            lines.append(caption)
            lines.append("")
    (output_dir / "figure_captions.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("outputs") / "sweeps")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir / "plots"
    plot_all(args.input_dir, output_dir)


if __name__ == "__main__":
    main()
