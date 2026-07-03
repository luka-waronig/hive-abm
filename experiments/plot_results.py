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


def ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


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
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    if group is None:
        grouped = df.groupby(x, dropna=True)[y].agg(["mean", "count", "std"]).reset_index()
        grouped["ci"] = df.groupby(x, dropna=True)[y].apply(ci95).values
        grouped = grouped.sort_values(x)
        ax.plot(grouped[x], grouped["mean"], marker="o", linewidth=2.0)
        ax.fill_between(grouped[x], grouped["mean"] - grouped["ci"], grouped["mean"] + grouped["ci"], alpha=0.18)
    else:
        for group_value, part in df.groupby(group, dropna=True):
            grouped = part.groupby(x, dropna=True)[y].agg(["mean", "count", "std"]).reset_index()
            grouped["ci"] = part.groupby(x, dropna=True)[y].apply(ci95).values
            grouped = grouped.sort_values(x)
            ax.plot(grouped[x], grouped["mean"], marker="o", linewidth=2.0, label=str(group_value))
            ax.fill_between(grouped[x], grouped["mean"] - grouped["ci"], grouped["mean"] + grouped["ci"], alpha=0.12)
        ax.legend(title=group)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
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
                ax.text(c, r, f"{value_at_cell:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
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
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for group_value, part in ticks.groupby(group, dropna=True):
        grouped = part.groupby("step", dropna=True)[y].mean().reset_index()
        ax.plot(grouped["step"], grouped[y], linewidth=2.0, label=str(group_value))
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(title=group)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_all(input_dir: Path, output_dir: Path) -> None:
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
    print(f"Wrote plots to {output_dir}")


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
