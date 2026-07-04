"""Scientific animated 2D grid visualization for Hive ABM runs.

Run from the project root:
    python experiments/animate_grid.py

The default output is:
    outputs/animations/grid_replay.gif
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from hive_abm.simulation import HiveSimulation, SimulationConfig


ROLE_COLORS = {
    "coordinator": "#6d28d9",
    "planner": "#2563eb",
    "worker": "#16a34a",
    "verifier": "#dc2626",
    "relay": "#ea580c",
    "observer": "#0891b2",
    "memory": "#9333ea",
}

STATE_EDGE_COLORS = {
    "idle": "#ffffff",
    "working": "#facc15",
    "blocked": "#ef4444",
    "relaying": "#fb923c",
    "observing": "#67e8f9",
    "failed": "#111827",
}

TASK_MARKERS = {
    "pending": ("o", "#64748b"),
    "in_progress": ("s", "#f59e0b"),
    "awaiting_verification": ("D", "#db2777"),
    "completed": ("*", "#15803d"),
    "failed": ("X", "#991b1b"),
}

LAYER_CMAPS = {
    "congestion": "magma",
    "mission_pressure": "viridis",
    "sensor_coverage": "Greens",
    "bandwidth_capacity": "YlGnBu",
    "bandwidth_demand": "OrRd",
    "compute_capacity": "YlGn",
    "compute_demand": "Oranges",
    "noise": "inferno",
}


def edge_rows(adjacency: Dict[int, set[int]]) -> List[Tuple[int, int]]:
    edges = []
    for source, targets in adjacency.items():
        for target in targets:
            if source < target:
                edges.append((source, target))
    return sorted(edges)


def capture_frame(sim: HiveSimulation) -> Dict[str, Any]:
    tick = sim.metrics.ticks[-1]
    return {
        "step": sim.step_idx,
        "layers": {name: [row[:] for row in grid] for name, grid in sim.environment.layers.items()},
        "nodes": [
            {
                "id": node.node_id,
                "role": node.role,
                "state": node.state,
                "row": node.position[0],
                "col": node.position[1],
                "load": node.current_load,
                "task_id": node.current_task_id,
            }
            for node in sim.nodes
        ],
        "tasks": [
            {
                "id": task.task_id,
                "type": task.task_type,
                "status": task.status,
                "row": task.sector[0],
                "col": task.sector[1],
                "phase": task.active_phase,
                "work_remaining": task.work_remaining,
            }
            for task in sim.mission.tasks.values()
        ],
        "edges": edge_rows(sim.adjacency),
        "metrics": asdict(tick),
    }


def run_frames(config: SimulationConfig) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sim = HiveSimulation(config)
    frames = []
    for step in range(config.max_steps):
        sim.step_idx = step
        sim.step()
        frames.append(capture_frame(sim))
        if sim.mission.is_success():
            break
    return frames, asdict(sim.summary())


def layer_limits(frames: List[Dict[str, Any]], layer: str) -> Tuple[float, float]:
    values = []
    for frame in frames:
        for row in frame["layers"][layer]:
            values.extend(row)
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    return lo, hi if hi > lo else lo + 1.0


def node_lookup(frame: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {node["id"]: node for node in frame["nodes"]}


def draw_edges(ax, frame: Dict[str, Any], enabled: bool) -> None:
    if not enabled:
        return
    nodes = node_lookup(frame)
    segments = []
    for source, target in frame["edges"]:
        a = nodes.get(source)
        b = nodes.get(target)
        if a is None or b is None:
            continue
        segments.append([(a["col"], a["row"]), (b["col"], b["row"])])
    if segments:
        ax.add_collection(LineCollection(segments, colors="#334155", linewidths=0.7, alpha=0.24, zorder=2))


def draw_tasks(ax, frame: Dict[str, Any], show_completed: bool) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for task in frame["tasks"]:
        if task["status"] == "completed" and not show_completed:
            continue
        grouped.setdefault(task["status"], []).append(task)
    for status, tasks in grouped.items():
        marker, color = TASK_MARKERS.get(status, ("o", "#64748b"))
        ax.scatter(
            [task["col"] - 0.22 for task in tasks],
            [task["row"] - 0.22 for task in tasks],
            s=38,
            marker=marker,
            c=color,
            edgecolors="#ffffff",
            linewidths=0.6,
            alpha=0.92,
            zorder=4,
        )


def draw_nodes(ax, frame: Dict[str, Any], label_nodes: bool) -> None:
    for node in frame["nodes"]:
        face = "#111827" if node["state"] == "failed" else ROLE_COLORS.get(node["role"], "#475569")
        edge = STATE_EDGE_COLORS.get(node["state"], "#ffffff")
        ax.scatter(
            [node["col"]],
            [node["row"]],
            s=78,
            c=face,
            edgecolors=edge,
            linewidths=2.0 if node["state"] in {"working", "blocked", "failed"} else 1.0,
            zorder=5,
        )
        if label_nodes:
            ax.text(
                node["col"],
                node["row"] + 0.34,
                str(node["id"]),
                ha="center",
                va="bottom",
                fontsize=6,
                color="#0f172a",
                zorder=6,
            )


def plot_metric_traces(ax, frames: List[Dict[str, Any]], current_idx: int) -> None:
    steps = [frame["step"] for frame in frames]
    completion = [frame["metrics"]["completion_rate"] for frame in frames]
    verified = [frame["metrics"]["verified_completion_rate"] for frame in frames]
    congestion = [frame["metrics"]["mean_congestion"] for frame in frames]
    queue = [frame["metrics"]["awaiting_verification_tasks"] for frame in frames]
    max_queue = max(max(queue), 1)
    queue_scaled = [value / max_queue for value in queue]

    ax.plot(steps, completion, color="#2563eb", linewidth=2.0, label="completion")
    ax.plot(steps, verified, color="#16a34a", linewidth=2.0, label="verified")
    ax.plot(steps, congestion, color="#dc2626", linewidth=1.8, label="mean congestion")
    ax.plot(steps, queue_scaled, color="#db2777", linewidth=1.6, label="verification queue scaled")
    ax.axvline(steps[current_idx], color="#111827", linewidth=1.0, alpha=0.65)
    ax.set_xlim(min(steps), max(steps) if max(steps) > min(steps) else min(steps) + 1)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Step")
    ax.set_ylabel("Metric value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncols=4, fontsize=8, frameon=False)


def role_legend_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="#ffffff", markersize=7, label=role)
        for role, color in ROLE_COLORS.items()
    ]


def task_legend_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], marker=marker, color="none", markerfacecolor=color, markeredgecolor="#ffffff", markersize=7, label=status)
        for status, (marker, color) in TASK_MARKERS.items()
    ]


def make_animation(
    frames: List[Dict[str, Any]],
    config: SimulationConfig,
    summary: Dict[str, Any],
    layer: str,
    output: Path,
    interval: int,
    dpi: int,
    show_edges: bool,
    show_completed_tasks: bool,
    label_nodes: bool,
    style: str,
    color_by_role: bool,
) -> None:
    if not frames:
        raise ValueError("no frames captured")
    if layer not in frames[0]["layers"]:
        raise ValueError(f"unknown layer: {layer}")

    rows = config.rows
    cols = config.cols
    vmin, vmax = layer_limits(frames, layer)
    cmap = LAYER_CMAPS.get(layer, "viridis")

    fig = plt.figure(figsize=(11.5, 8.0))
    gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[3.2, 1.2], height_ratios=[3.4, 1.0])
    ax_grid = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_trace = fig.add_subplot(gs[1, :])
    fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.08, wspace=0.16, hspace=0.25)

    first = frames[0]
    image = ax_grid.imshow(
        first["layers"][layer],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="nearest",
        zorder=1,
    )
    fig.colorbar(image, ax=ax_grid, fraction=0.046, pad=0.02, label=layer)
    ax_grid.set_xlim(-0.5, cols - 0.5)
    ax_grid.set_ylim(rows - 0.5, -0.5)
    x_major = sorted(set([0, cols - 1, *range(0, cols, max(1, cols // 5))]))
    y_major = sorted(set([0, rows - 1, *range(0, rows, max(1, rows // 5))]))
    ax_grid.set_xticks(x_major)
    ax_grid.set_yticks(y_major)
    ax_grid.set_xticks(range(cols), minor=True)
    ax_grid.set_yticks(range(rows), minor=True)
    ax_grid.tick_params(labelsize=7)
    ax_grid.grid(which="minor", color="#ffffff", linewidth=0.35, alpha=0.35)
    ax_grid.set_xlabel("Column")
    ax_grid.set_ylabel("Row")

    edge_collection = LineCollection([], colors="#334155", linewidths=0.7, alpha=0.24, zorder=2)
    ax_grid.add_collection(edge_collection)

    task_scatters = {}
    for status, (marker, color) in TASK_MARKERS.items():
        task_scatters[status] = ax_grid.scatter(
            [],
            [],
            s=38,
            marker=marker,
            c=color,
            edgecolors="#ffffff",
            linewidths=0.6,
            alpha=0.92,
            zorder=4,
        )

    node_scatter = ax_grid.scatter([], [], s=78, zorder=5)
    node_labels = []
    if label_nodes:
        for _ in first["nodes"]:
            node_labels.append(ax_grid.text(0, 0, "", ha="center", va="bottom", fontsize=6, color="#0f172a", zorder=6))

    ax_info.axis("off")
    info_text = ax_info.text(0.0, 0.98, "", ha="left", va="top", fontsize=9, family="monospace")
    if color_by_role:
        leg1 = ax_info.legend(handles=role_legend_handles(), loc="lower left", title="Roles", fontsize=7, title_fontsize=8, frameon=False)
        ax_info.add_artist(leg1)
        ax_info.legend(handles=task_legend_handles(), loc="lower right", title="Tasks", fontsize=7, title_fontsize=8, frameon=False)
    else:
        node_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#111827", markeredgecolor="#ffffff", markersize=7, label="agent"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#111827", markeredgecolor="#facc15", markeredgewidth=2, markersize=7, label="working"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#111827", markeredgecolor="#ef4444", markeredgewidth=2, markersize=7, label="blocked"),
        ]
        leg1 = ax_info.legend(handles=node_handles, loc="lower left", title="Nodes", fontsize=7, title_fontsize=8, frameon=False)
        ax_info.add_artist(leg1)
        ax_info.legend(handles=task_legend_handles(), loc="lower right", title="Tasks", fontsize=7, title_fontsize=8, frameon=False)

    steps = [frame["step"] for frame in frames]
    completion = [frame["metrics"]["completion_rate"] for frame in frames]
    verified = [frame["metrics"]["verified_completion_rate"] for frame in frames]
    congestion = [frame["metrics"]["mean_congestion"] for frame in frames]
    ax_trace.plot(steps, completion, color="#2563eb", linewidth=2.0, label="completion")
    if style == "diagnostic":
        queue = [frame["metrics"]["awaiting_verification_tasks"] for frame in frames]
        max_queue = max(max(queue), 1)
        queue_scaled = [value / max_queue for value in queue]
        ax_trace.plot(steps, verified, color="#16a34a", linewidth=2.0, label="verified")
        ax_trace.plot(steps, congestion, color="#dc2626", linewidth=1.8, label="mean congestion")
        ax_trace.plot(steps, queue_scaled, color="#db2777", linewidth=1.6, label="verification queue scaled")
    else:
        max_congestion = max(max(congestion), 1e-9)
        congestion_scaled = [value / max_congestion for value in congestion]
        ax_trace.plot(steps, verified, color="#16a34a", linewidth=1.7, linestyle="--", label="verified")
        ax_trace.plot(steps, congestion_scaled, color="#dc2626", linewidth=1.7, label="congestion scaled")
    current_line = ax_trace.axvline(steps[0], color="#111827", linewidth=1.0, alpha=0.65)
    ax_trace.set_xlim(min(steps), max(steps) if max(steps) > min(steps) else min(steps) + 1)
    ax_trace.set_ylim(0.0, 1.02)
    ax_trace.set_xlabel("Step")
    ax_trace.set_ylabel("Metric value")
    ax_trace.grid(True, alpha=0.25)
    ax_trace.legend(loc="upper left", ncols=3 if style == "thesis" else 4, fontsize=8, frameon=False)

    def set_offsets(collection, pairs: List[Tuple[float, float]]) -> None:
        collection.set_offsets(np.asarray(pairs, dtype=float) if pairs else np.empty((0, 2)))

    def update(idx: int):
        frame = frames[idx]
        metrics = frame["metrics"]
        image.set_data(frame["layers"][layer])

        nodes = node_lookup(frame)
        if show_edges:
            segments = []
            for source, target in frame["edges"]:
                a = nodes.get(source)
                b = nodes.get(target)
                if a is not None and b is not None:
                    segments.append([(a["col"], a["row"]), (b["col"], b["row"])])
            edge_collection.set_segments(segments)
        else:
            edge_collection.set_segments([])

        for status, scatter in task_scatters.items():
            pairs = []
            for task in frame["tasks"]:
                if task["status"] != status:
                    continue
                if status == "completed" and not show_completed_tasks:
                    continue
                pairs.append((task["col"] - 0.22, task["row"] - 0.22))
            set_offsets(scatter, pairs)

        node_pairs = [(node["col"], node["row"]) for node in frame["nodes"]]
        if color_by_role:
            node_faces = ["#111827" if node["state"] == "failed" else ROLE_COLORS.get(node["role"], "#475569") for node in frame["nodes"]]
        else:
            node_faces = ["#6b7280" if node["state"] == "failed" else "#111827" for node in frame["nodes"]]
        node_edges = [STATE_EDGE_COLORS.get(node["state"], "#ffffff") for node in frame["nodes"]]
        set_offsets(node_scatter, node_pairs)
        node_scatter.set_facecolors(node_faces)
        node_scatter.set_edgecolors(node_edges)
        node_scatter.set_linewidths([2.0 if node["state"] in {"working", "blocked", "failed"} else 1.0 for node in frame["nodes"]])

        for text, node in zip(node_labels, frame["nodes"]):
            text.set_position((node["col"], node["row"] + 0.34))
            text.set_text(str(node["id"]))

        ax_grid.set_title(
            f"{layer} field | step {frame['step']} | completion {metrics['completion_rate']:.2f}",
            fontsize=12,
        )
        if style == "diagnostic":
            info = [
                f"run: {config.run_id}",
                f"topology: {config.topology}",
                f"policy: {config.policy_level}",
                f"nodes/tasks: {config.n_nodes}/{config.n_tasks}",
                "",
                f"mean congestion: {metrics['mean_congestion']:.4f}",
                f"peak congestion: {metrics['peak_congestion']:.4f}",
                f"messages: {metrics['messages_sent']}",
                f"denied actions: {metrics['denied_actions']}",
                f"verification queue: {metrics['awaiting_verification_tasks']}",
                f"working nodes: {metrics['working_nodes']}",
                "",
                f"final completion: {summary['final_completion_rate']:.3f}",
                f"final verified: {summary['final_verified_completion_rate']:.3f}",
            ]
        else:
            info = [
                f"topology: {config.topology}",
                f"policy: {config.policy_level}",
                f"nodes/tasks: {config.n_nodes}/{config.n_tasks}",
                "",
                f"completion: {metrics['completion_rate']:.3f}",
                f"verified: {metrics['verified_completion_rate']:.3f}",
                f"mean congestion: {metrics['mean_congestion']:.4f}",
                f"verification queue: {metrics['awaiting_verification_tasks']}",
            ]
        info_text.set_text("\n".join(info))
        current_line.set_xdata([steps[idx], steps[idx]])
        return [image, edge_collection, node_scatter, info_text, current_line, *task_scatters.values(), *node_labels]

    animation = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False, repeat=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".gif":
        animation.save(output, writer=PillowWriter(fps=max(1, round(1000 / interval))), dpi=dpi)
    elif suffix == ".html":
        output.write_text(animation.to_jshtml(), encoding="utf-8")
    else:
        raise ValueError("output must end in .gif or .html")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "animations" / "grid_replay.gif")
    parser.add_argument("--layer", default="congestion")
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--n-tasks", type=int, default=70)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--topology", default="decentralized_mesh")
    parser.add_argument("--policy-level", default="moderate")
    parser.add_argument("--memory-mode", default="local_and_shared")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--message-loss-probability", type=float, default=0.02)
    parser.add_argument("--node-failure-probability", type=float, default=0.002)
    parser.add_argument("--interval", type=int, default=180)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--style", choices=["thesis", "diagnostic"], default="thesis")
    parser.add_argument("--show-edges", action="store_true")
    parser.add_argument("--hide-edges", action="store_true")
    parser.add_argument("--color-by-role", action="store_true")
    parser.add_argument("--show-completed-tasks", action="store_true")
    parser.add_argument("--label-nodes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimulationConfig(
        rows=args.rows,
        cols=args.cols,
        n_nodes=args.n_nodes,
        n_tasks=args.n_tasks,
        max_steps=args.max_steps,
        topology=args.topology,
        policy_level=args.policy_level,
        memory_mode=args.memory_mode,
        seed=args.seed,
        message_loss_probability=args.message_loss_probability,
        node_failure_probability=args.node_failure_probability,
        run_id=f"animation_{args.topology}_{args.policy_level}_seed{args.seed}",
        metadata={"experiment": "animation"},
    )
    frames, summary = run_frames(cfg)
    make_animation(
        frames=frames,
        config=cfg,
        summary=summary,
        layer=args.layer,
        output=args.output,
        interval=args.interval,
        dpi=args.dpi,
        show_edges=args.show_edges or (args.style == "diagnostic" and not args.hide_edges),
        show_completed_tasks=args.show_completed_tasks,
        label_nodes=args.label_nodes,
        style=args.style,
        color_by_role=args.color_by_role or args.style == "diagnostic",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
