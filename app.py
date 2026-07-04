"""Interactive Hive ABM spatial simulation workbench.

Run from the project root:
    python -m streamlit run app.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
import math
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hive_abm.session import SimulationSession


UI_DEFAULT_VERSION = 3
DEFAULT_MODE = "Edge sensing mesh"

RUN_MODES: Dict[str, Dict[str, Any]] = {
    "Edge sensing mesh": {
        "scenario": "baseline",
        "environment_preset": "clustered_tasks",
        "topology": "decentralized_mesh",
        "policy_level": "moderate",
        "rows": 12,
        "cols": 12,
        "n_nodes": 16,
        "n_tasks": 22,
        "max_steps": 80,
        "message_loss_probability": 0.02,
        "node_failure_probability": 0.001,
    },
    "Balanced run": {
        "scenario": "baseline",
        "environment_preset": "uniform",
        "topology": "decentralized_mesh",
        "policy_level": "moderate",
        "rows": 20,
        "cols": 20,
        "n_nodes": 50,
        "n_tasks": 70,
        "max_steps": 100,
        "message_loss_probability": 0.02,
        "node_failure_probability": 0.002,
    },
    "Clustered tasks": {
        "scenario": "baseline",
        "environment_preset": "clustered_tasks",
        "topology": "decentralized_mesh",
        "policy_level": "moderate",
        "rows": 20,
        "cols": 20,
        "n_nodes": 50,
        "n_tasks": 80,
        "max_steps": 110,
        "message_loss_probability": 0.02,
        "node_failure_probability": 0.002,
    },
    "Noisy region": {
        "scenario": "noisy",
        "environment_preset": "noisy_region",
        "topology": "decentralized_mesh",
        "policy_level": "moderate",
        "rows": 20,
        "cols": 20,
        "n_nodes": 60,
        "n_tasks": 80,
        "max_steps": 110,
        "message_loss_probability": 0.10,
        "node_failure_probability": 0.002,
    },
    "Verification queue": {
        "scenario": "verification_bottleneck",
        "environment_preset": "clustered_tasks",
        "topology": "decentralized_mesh",
        "policy_level": "mission_critical",
        "rows": 20,
        "cols": 20,
        "n_nodes": 60,
        "n_tasks": 90,
        "max_steps": 130,
        "message_loss_probability": 0.03,
        "node_failure_probability": 0.002,
    },
    "Relay corridor": {
        "scenario": "relay_stress",
        "environment_preset": "corridor",
        "topology": "decentralized_mesh",
        "policy_level": "strict",
        "rows": 20,
        "cols": 24,
        "n_nodes": 45,
        "n_tasks": 80,
        "max_steps": 120,
        "message_loss_probability": 0.06,
        "node_failure_probability": 0.002,
    },
    "Dense swarm": {
        "scenario": "dense",
        "environment_preset": "uniform",
        "topology": "adaptive",
        "policy_level": "moderate",
        "rows": 22,
        "cols": 22,
        "n_nodes": 100,
        "n_tasks": 85,
        "max_steps": 100,
        "message_loss_probability": 0.02,
        "node_failure_probability": 0.002,
    },
}

MODE_NOTES = {
    "Edge sensing mesh": "Small readable map. Edge agents sense nearby sectors, coordinate through a local mesh, and work on fixed mission tasks.",
    "Balanced run": "Baseline thesis-scale run for checking ordinary decentralized coordination behavior.",
    "Clustered tasks": "Tasks are spatially concentrated, so the mesh must reallocate agents toward hotspots.",
    "Noisy region": "A region has higher communication interference, stressing routing and assignment.",
    "Verification queue": "Mission-critical policy forces solved tasks through independent verification.",
    "Relay corridor": "A corridor has better communication capacity; relay placement matters more.",
    "Dense swarm": "Many nodes share the same grid, exposing crowding, redundancy, and congestion effects.",
}

TOPOLOGY_LABELS = {
    "centralized": "Central hub",
    "hierarchical": "Coordinator and sector leads",
    "decentralized_mesh": "Local peer mesh",
    "adaptive": "Adaptive local mesh",
    "small_world": "Small-world shortcuts",
    "scale_free": "Hub-heavy network",
}

POLICY_LABELS = {
    "permissive": "Permissive",
    "moderate": "Moderate",
    "strict": "Strict",
    "mission_critical": "Mission critical",
}

FIELD_LABELS = {
    "congestion": "Congestion",
    "mission_pressure": "Mission pressure",
    "sensor_coverage": "Sensor coverage",
    "bandwidth_capacity": "Bandwidth capacity",
    "bandwidth_demand": "Bandwidth demand",
    "compute_capacity": "Compute capacity",
    "compute_demand": "Compute demand",
    "noise": "Noise",
}

MAP_LAYERS = {
    "Mission demand": "mission_pressure",
    "Network load": "congestion",
    "Sensor coverage": "sensor_coverage",
    "Bandwidth capacity": "bandwidth_capacity",
    "Interference": "noise",
}

MAP_LAYER_NOTES = {
    "Mission demand": "where unresolved work is concentrated",
    "Network load": "where communication and resource demand are high",
    "Sensor coverage": "where agents can locally sense the environment",
    "Bandwidth capacity": "where relay and communication capacity exists",
    "Interference": "where communication noise is present",
}

FIELD_MAX = {
    "congestion": 0.35,
    "mission_pressure": 3.0,
    "sensor_coverage": 5.0,
    "bandwidth_capacity": 6.0,
    "bandwidth_demand": 1.5,
    "compute_capacity": 3.0,
    "compute_demand": 1.5,
    "noise": 1.0,
}

FIELD_NOTES = {
    "congestion": "local communication and resource strain",
    "mission_pressure": "unresolved task pressure plus coverage/congestion pressure",
    "sensor_coverage": "how much nearby sensing is available",
    "bandwidth_capacity": "local communication capacity supplied by nodes",
    "bandwidth_demand": "local communication demand imposed by nodes",
    "compute_capacity": "local compute capacity supplied by nodes",
    "compute_demand": "local compute demand imposed by nodes",
    "noise": "persistent interference plus node-induced noise",
}

FIELD_SCALE = {
    "congestion": [[0.0, "#f8fafc"], [0.35, "#fde68a"], [0.7, "#f97316"], [1.0, "#991b1b"]],
    "mission_pressure": [[0.0, "#f8fafc"], [0.35, "#bae6fd"], [0.7, "#2563eb"], [1.0, "#1e1b4b"]],
    "sensor_coverage": [[0.0, "#f8fafc"], [0.4, "#bbf7d0"], [0.75, "#16a34a"], [1.0, "#064e3b"]],
    "bandwidth_capacity": [[0.0, "#f8fafc"], [0.45, "#bfdbfe"], [0.8, "#2563eb"], [1.0, "#172554"]],
    "bandwidth_demand": [[0.0, "#f8fafc"], [0.45, "#fed7aa"], [0.8, "#ea580c"], [1.0, "#7c2d12"]],
    "compute_capacity": [[0.0, "#f8fafc"], [0.45, "#d9f99d"], [0.8, "#65a30d"], [1.0, "#365314"]],
    "compute_demand": [[0.0, "#f8fafc"], [0.45, "#fecaca"], [0.8, "#dc2626"], [1.0, "#7f1d1d"]],
    "noise": [[0.0, "#f8fafc"], [0.4, "#c4b5fd"], [0.75, "#7c3aed"], [1.0, "#3b0764"]],
}

TASK_STYLE = {
    "pending": {"name": "Open task sector", "symbol": "square-open", "color": "#475569", "size": 14, "text": "#111827"},
    "in_progress": {"name": "Active task sector", "symbol": "square", "color": "#2563eb", "size": 15, "text": "#ffffff"},
    "awaiting_verification": {"name": "Verification sector", "symbol": "diamond", "color": "#7c3aed", "size": 15, "text": "#ffffff"},
    "completed": {"name": "Completed sector", "symbol": "square", "color": "#16a34a", "size": 10, "text": "#ffffff"},
    "failed": {"name": "Failed sector", "symbol": "x", "color": "#991b1b", "size": 12, "text": "#ffffff"},
}

NODE_STYLE = {
    "idle": {"name": "Idle agent sector", "symbol": "circle", "color": "#111827", "line": "#ffffff", "size": 9, "text": "#ffffff"},
    "working": {"name": "Working agent sector", "symbol": "circle", "color": "#111827", "line": "#2563eb", "size": 12, "text": "#ffffff"},
    "blocked": {"name": "Blocked agent sector", "symbol": "x", "color": "#dc2626", "line": "#ffffff", "size": 11, "text": "#ffffff"},
    "relaying": {"name": "Relay agent sector", "symbol": "circle", "color": "#111827", "line": "#0f766e", "size": 11, "text": "#ffffff"},
    "observing": {"name": "Sensing agent sector", "symbol": "circle", "color": "#111827", "line": "#0284c7", "size": 11, "text": "#ffffff"},
    "failed": {"name": "Failed agent sector", "symbol": "x", "color": "#6b7280", "line": "#ffffff", "size": 11, "text": "#ffffff"},
}

PLOT_TEMPLATE = {
    "layout": {
        "font": {"family": "Inter, Segoe UI, system-ui, sans-serif", "size": 13, "color": "#111827"},
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "colorway": ["#2563eb", "#047857", "#d97706", "#7c3aed", "#dc2626", "#0284c7"],
        "xaxis": {"gridcolor": "#e5e7eb", "zerolinecolor": "#e5e7eb"},
        "yaxis": {"gridcolor": "#e5e7eb", "zerolinecolor": "#e5e7eb"},
    }
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f6f7f9;
            color: #111827;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e5e7eb;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1480px;
        }
        h1 {
            font-size: 1.45rem;
            letter-spacing: 0;
            margin-bottom: 0.2rem;
        }
        h2, h3 {
            letter-spacing: 0;
        }
        .section-title {
            color: #111827;
            font-size: 0.95rem;
            font-weight: 800;
            margin: 0.2rem 0 0.35rem 0;
        }
        .fine-print {
            color: #4b5563;
            font-size: 0.88rem;
            line-height: 1.42;
            margin: 0.1rem 0 0.55rem 0;
        }
        .scenario-note {
            color: #374151;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            font-size: 0.86rem;
            line-height: 1.38;
            margin: 0.2rem 0 0.85rem 0;
        }
        .cycle-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin: 0.2rem 0 0.85rem 0;
        }
        .cycle-chip {
            background: #ffffff;
            border: 1px solid #d1d5db;
            border-radius: 999px;
            color: #374151;
            font-size: 0.78rem;
            font-weight: 750;
            padding: 0.24rem 0.55rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.7rem 0.85rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetricLabel"] {
            color: #4b5563;
            font-weight: 750;
        }
        div[data-testid="stMetricValue"] {
            color: #111827;
            font-size: 1.25rem;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 7px;
            border: 1px solid #d1d5db;
            font-weight: 750;
            min-height: 2.45rem;
        }
        .stButton > button[kind="primary"] {
            background: #2563eb;
            border-color: #2563eb;
        }
        .legend-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.35rem;
            margin: 0.2rem 0 0.75rem 0;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: #374151;
            font-size: 0.88rem;
        }
        .legend-dot {
            display: inline-block;
            width: 0.78rem;
            height: 0.78rem;
            border-radius: 999px;
            background: #111827;
            border: 2px solid #ffffff;
            box-shadow: 0 0 0 1px #cbd5e1;
        }
        .legend-square {
            display: inline-block;
            width: 0.75rem;
            height: 0.75rem;
            border: 2px solid #64748b;
            background: #ffffff;
        }
        .legend-active {
            display: inline-block;
            width: 0.75rem;
            height: 0.75rem;
            background: #2563eb;
        }
        .legend-diamond {
            display: inline-block;
            width: 0.72rem;
            height: 0.72rem;
            background: #7c3aed;
            transform: rotate(45deg);
            margin-left: 0.08rem;
        }
        .legend-line {
            display: inline-block;
            width: 1.15rem;
            height: 0;
            border-top: 2px solid #334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def mode_defaults(mode: str) -> Dict[str, Any]:
    values = dict(RUN_MODES[mode])
    values.setdefault("seed", 42)
    return values


def ensure_state_defaults() -> None:
    if st.session_state.get("ui_default_version") != UI_DEFAULT_VERSION:
        st.session_state.run_mode = DEFAULT_MODE
        st.session_state.seed = 42
        for key, value in mode_defaults(DEFAULT_MODE).items():
            st.session_state[key] = value
        st.session_state.map_layer = "Mission demand"
        st.session_state.show_agents = True
        st.session_state.show_tasks = True
        st.session_state.show_assignment_links = False
        st.session_state.show_topology_edges = False
        st.session_state.show_completed_tasks = False
        st.session_state.inspect_row = 0
        st.session_state.inspect_col = 0
        st.session_state.ui_default_version = UI_DEFAULT_VERSION
        st.session_state.pop("session", None)
        return

    st.session_state.setdefault("run_mode", DEFAULT_MODE)
    st.session_state.setdefault("seed", 42)
    for key, value in mode_defaults(st.session_state.run_mode).items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("map_layer", "Mission demand")
    st.session_state.setdefault("show_agents", True)
    st.session_state.setdefault("show_tasks", True)
    st.session_state.setdefault("show_assignment_links", False)
    st.session_state.setdefault("show_topology_edges", False)
    st.session_state.setdefault("show_completed_tasks", False)
    st.session_state.setdefault("inspect_row", 0)
    st.session_state.setdefault("inspect_col", 0)


def reset_session() -> None:
    ensure_state_defaults()
    st.session_state.session = SimulationSession(
        scenario=st.session_state.scenario,
        environment_preset=st.session_state.environment_preset,
        seed=st.session_state.seed,
        config_overrides={
            "topology": st.session_state.topology,
            "policy_level": st.session_state.policy_level,
            "n_nodes": st.session_state.n_nodes,
            "n_tasks": st.session_state.n_tasks,
            "rows": st.session_state.rows,
            "cols": st.session_state.cols,
            "max_steps": st.session_state.max_steps,
            "message_loss_probability": st.session_state.message_loss_probability,
            "node_failure_probability": st.session_state.node_failure_probability,
        },
    )


def apply_mode_defaults() -> None:
    for key, value in mode_defaults(st.session_state.run_mode).items():
        st.session_state[key] = value
    st.session_state.inspect_row = 0
    st.session_state.inspect_col = 0
    reset_session()


def ensure_session() -> SimulationSession:
    ensure_state_defaults()
    if "session" not in st.session_state:
        reset_session()
    return st.session_state.session


def metric_frame(session: SimulationSession) -> pd.DataFrame:
    rows = [asdict(tick) for tick in session.sim.metrics.ticks]
    if not rows:
        rows = [
            {
                "step": 0,
                "completion_rate": session.sim.mission.completion_rate(),
                "verified_completion_rate": session.sim.mission.verified_completion_rate(),
            }
        ]
    return pd.DataFrame(rows)


def grouped(records: Iterable[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        out.setdefault(str(record.get(key, "")), []).append(record)
    return out


def node_positions(snapshot: dict) -> Dict[int, Tuple[int, int]]:
    return {int(node["id"]): tuple(node["position"]) for node in snapshot["nodes"]}


def task_status_counts(snapshot: dict) -> Dict[str, int]:
    counts = {"pending": 0, "in_progress": 0, "awaiting_verification": 0, "completed": 0}
    for task in snapshot["tasks"]:
        counts[task["status"]] = counts.get(task["status"], 0) + 1
    return counts


def active_node_count(snapshot: dict) -> int:
    return sum(1 for node in snapshot["nodes"] if node["state"] in {"working", "relaying", "observing"})


TASK_STATUS_PRIORITY = {
    "in_progress": 4,
    "awaiting_verification": 3,
    "pending": 2,
    "failed": 1,
    "completed": 0,
}

NODE_STATE_PRIORITY = {
    "working": 5,
    "blocked": 4,
    "relaying": 3,
    "observing": 2,
    "failed": 1,
    "idle": 0,
}


def compact_counts(counts: Dict[str, int]) -> str:
    return ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()) if value)


def task_sector_summaries(snapshot: dict, show_completed: bool) -> List[Dict[str, Any]]:
    cells: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for task in snapshot["tasks"]:
        if task["status"] == "completed" and not show_completed:
            continue
        row, col = task["sector"]
        cell = cells.setdefault((row, col), {"row": row, "col": col, "count": 0, "status_counts": {}, "types": {}})
        cell["count"] += 1
        cell["status_counts"][task["status"]] = cell["status_counts"].get(task["status"], 0) + 1
        cell["types"][task["type"]] = cell["types"].get(task["type"], 0) + 1
    summaries: List[Dict[str, Any]] = []
    for cell in cells.values():
        dominant = max(cell["status_counts"], key=lambda status: TASK_STATUS_PRIORITY.get(status, -1))
        summaries.append({**cell, "status": dominant})
    return summaries


def agent_sector_summaries(snapshot: dict) -> List[Dict[str, Any]]:
    cells: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for node in snapshot["nodes"]:
        row, col = node["position"]
        cell = cells.setdefault((row, col), {"row": row, "col": col, "count": 0, "state_counts": {}, "roles": {}})
        cell["count"] += 1
        cell["state_counts"][node["state"]] = cell["state_counts"].get(node["state"], 0) + 1
        cell["roles"][node["role"]] = cell["roles"].get(node["role"], 0) + 1
    summaries: List[Dict[str, Any]] = []
    for cell in cells.values():
        dominant = max(cell["state_counts"], key=lambda state: NODE_STATE_PRIORITY.get(state, -1))
        summaries.append({**cell, "state": dominant})
    return summaries


def marker_size(base: int, count: int) -> int:
    return min(28, base + int(4 * math.sqrt(max(0, count - 1))))


def add_assignment_links(fig: go.Figure, snapshot: dict) -> None:
    positions = node_positions(snapshot)
    first = True
    for task in snapshot["tasks"]:
        node_id = task.get("assigned_to")
        if node_id is None or node_id not in positions:
            continue
        node_r, node_c = positions[int(node_id)]
        task_r, task_c = task["sector"]
        fig.add_trace(
            go.Scatter(
                x=[node_c, task_c],
                y=[node_r, task_r],
                mode="lines",
                line={"color": "rgba(15, 23, 42, 0.48)", "width": 1.6, "dash": "dot"},
                name="active assignment",
                showlegend=first,
                hovertemplate=(
                    f"node {node_id} -> task {task['id']}<br>"
                    f"phase={task.get('phase')}<br>"
                    f"work remaining={task.get('work_remaining')}<extra></extra>"
                ),
            )
        )
        first = False


def add_topology_edges(fig: go.Figure, snapshot: dict) -> None:
    positions = node_positions(snapshot)
    first = True
    for source, target in snapshot["edges"]:
        if source not in positions or target not in positions:
            continue
        r1, c1 = positions[source]
        r2, c2 = positions[target]
        fig.add_trace(
            go.Scatter(
                x=[c1, c2],
                y=[r1, r2],
                mode="lines",
                line={"color": "rgba(100, 116, 139, 0.16)", "width": 1},
                hoverinfo="skip",
                name="communication link",
                showlegend=first,
            )
        )
        first = False


def grid_figure(
    snapshot: dict,
    layer: str,
    show_agents: bool,
    show_tasks: bool,
    show_assignment_links: bool,
    show_topology_edges: bool,
    show_completed_tasks: bool,
    selected_cell: Tuple[int, int],
) -> go.Figure:
    z = snapshot["layers"][layer]
    rows = len(z)
    cols = len(z[0]) if rows else 0
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=list(range(cols)),
            y=list(range(rows)),
            colorscale=FIELD_SCALE[layer],
            zmin=0,
            zmax=FIELD_MAX.get(layer, 1.0),
            xgap=1,
            ygap=1,
            colorbar={"title": FIELD_LABELS[layer], "thickness": 14, "len": 0.76},
            hovertemplate="row=%{y}<br>col=%{x}<br>value=%{z:.3f}<extra></extra>",
            name=FIELD_LABELS[layer],
            showscale=True,
        )
    )
    if show_topology_edges:
        add_topology_edges(fig, snapshot)
    if show_assignment_links:
        add_assignment_links(fig, snapshot)
    if show_tasks:
        for status, part in grouped(task_sector_summaries(snapshot, show_completed_tasks), "status").items():
            style = TASK_STYLE.get(status, TASK_STYLE["pending"])
            fig.add_trace(
                go.Scatter(
                    x=[cell["col"] for cell in part],
                    y=[cell["row"] for cell in part],
                    mode="markers+text",
                    marker={
                        "symbol": style["symbol"],
                        "size": [marker_size(style["size"], cell["count"]) for cell in part],
                        "color": style["color"],
                        "opacity": 0.35 if status == "completed" else 0.96,
                        "line": {"color": "#ffffff", "width": 1.4},
                    },
                    text=[str(cell["count"]) if cell["count"] > 1 else "" for cell in part],
                    textfont={"color": style["text"], "size": 10},
                    textposition="middle center",
                    name=style["name"],
                    hovertext=[
                        (
                            f"sector=({cell['row']}, {cell['col']})<br>"
                            f"tasks={cell['count']}<br>"
                            f"status={compact_counts(cell['status_counts'])}<br>"
                            f"types={compact_counts(cell['types'])}"
                        )
                        for cell in part
                    ],
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )
    if show_agents:
        for state, part in grouped(agent_sector_summaries(snapshot), "state").items():
            style = NODE_STYLE.get(state, NODE_STYLE["idle"])
            fig.add_trace(
                go.Scatter(
                    x=[cell["col"] for cell in part],
                    y=[cell["row"] for cell in part],
                    mode="markers+text",
                    marker={
                        "symbol": style["symbol"],
                        "size": [marker_size(style["size"], cell["count"]) for cell in part],
                        "color": style["color"],
                        "line": {"color": style["line"], "width": 2.0 if state in {"working", "relaying", "observing"} else 1.3},
                    },
                    text=[str(cell["count"]) if cell["count"] > 1 else "" for cell in part],
                    textfont={"color": style["text"], "size": 10},
                    textposition="middle center",
                    name=style["name"],
                    showlegend=False,
                    hovertext=[
                        (
                            f"sector=({cell['row']}, {cell['col']})<br>"
                            f"agents={cell['count']}<br>"
                            f"states={compact_counts(cell['state_counts'])}<br>"
                            f"roles={compact_counts(cell['roles'])}"
                        )
                        for cell in part
                    ],
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )
    sr, sc = selected_cell
    if 0 <= sr < rows and 0 <= sc < cols:
        fig.add_shape(
            type="rect",
            x0=sc - 0.5,
            x1=sc + 0.5,
            y0=sr - 0.5,
            y1=sr + 0.5,
            line={"color": "#111827", "width": 2},
            fillcolor="rgba(0,0,0,0)",
        )
    fig.update_yaxes(
        autorange="reversed",
        title="row",
        range=[rows - 0.5, -0.5],
        dtick=max(1, rows // 10),
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    )
    fig.update_xaxes(title="column", range=[-0.5, cols - 0.5], dtick=max(1, cols // 10), constrain="domain")
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=660,
        margin={"l": 12, "r": 12, "t": 16, "b": 8},
        showlegend=False,
    )
    return fig


def metric_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(template=PLOT_TEMPLATE, height=280, margin={"l": 10, "r": 10, "t": 20, "b": 10})
        return fig
    fig.add_trace(
        go.Scatter(
            x=df["step"],
            y=100 * df["completion_rate"],
            mode="lines+markers",
            name="completed",
            line={"width": 3, "color": "#2563eb"},
            hovertemplate="cycle=%{x}<br>completed=%{y:.0f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["step"],
            y=100 * df["verified_completion_rate"],
            mode="lines+markers",
            name="verified",
            line={"width": 3, "color": "#7c3aed"},
            hovertemplate="cycle=%{x}<br>verified=%{y:.0f}%<extra></extra>",
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=295,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis={"title": "mission cycle"},
        yaxis={"title": "tasks completed", "range": [0, 100], "ticksuffix": "%"},
        legend={"orientation": "h", "y": -0.18},
    )
    return fig


def assignment_frame(snapshot: dict) -> pd.DataFrame:
    positions = node_positions(snapshot)
    rows: List[Dict[str, Any]] = []
    for task in snapshot["tasks"]:
        node_id = task.get("assigned_to")
        if node_id is None:
            continue
        node_pos = positions.get(int(node_id))
        task_pos = tuple(task["sector"])
        distance = None
        if node_pos is not None:
            distance = abs(node_pos[0] - task_pos[0]) + abs(node_pos[1] - task_pos[1])
        rows.append(
            {
                "task": task["id"],
                "status": task["status"],
                "phase": task.get("phase"),
                "task sector": task_pos,
                "node": node_id,
                "node sector": node_pos,
                "grid distance": distance,
                "hops": task.get("assignment_hops", 0),
                "work left": task.get("work_remaining", 0),
            }
        )
    return pd.DataFrame(rows)


def queue_frame(snapshot: dict) -> pd.DataFrame:
    counts = task_status_counts(snapshot)
    rows = [
        {"queue": "open", "meaning": "available or waiting on dependencies", "tasks": counts.get("pending", 0)},
        {"queue": "being worked", "meaning": "assigned to agents this cycle", "tasks": counts.get("in_progress", 0)},
        {"queue": "needs verification", "meaning": "solved artifact waiting for verifier", "tasks": counts.get("awaiting_verification", 0)},
        {"queue": "completed", "meaning": "accepted mission work", "tasks": counts.get("completed", 0)},
    ]
    return pd.DataFrame(rows)


def cell_tables(snapshot: dict, row: int, col: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agents = [
        {
            "node": node["id"],
            "role": node["role"],
            "state": node["state"],
            "task": node.get("task_id"),
            "load": round(float(node.get("load", 0.0)), 3),
        }
        for node in snapshot["nodes"]
        if tuple(node["position"]) == (row, col)
    ]
    tasks = [
        {
            "task": task["id"],
            "type": task["type"],
            "status": task["status"],
            "assigned node": task.get("assigned_to"),
            "phase": task.get("phase"),
            "work left": task.get("work_remaining", 0),
        }
        for task in snapshot["tasks"]
        if tuple(task["sector"]) == (row, col)
    ]
    return pd.DataFrame(agents), pd.DataFrame(tasks)


def draw_visual_legend(show_assignment_links: bool) -> None:
    assignment_item = (
        '<div class="legend-item"><span class="legend-line"></span> Active remote assignment</div>'
        if show_assignment_links
        else ""
    )
    st.markdown(
        f"""
        <div class="legend-grid">
            <div class="legend-item"><span class="legend-dot"></span> Agent location</div>
            <div class="legend-item"><span class="legend-square"></span> Open task</div>
            <div class="legend-item"><span class="legend-active"></span> Task being worked</div>
            <div class="legend-item"><span class="legend-diamond"></span> Waiting for verification</div>
            {assignment_item}
        </div>
        """,
        unsafe_allow_html=True,
    )


def draw_reading_note(map_layer: str) -> None:
    st.markdown(
        f"""
        <div class="fine-print">
        This is a field mission map. Each cell is a sector; background color shows {map_layer.lower()};
        black circles are agent sectors; squares/diamonds are mission task sectors.
        </div>
        """,
        unsafe_allow_html=True,
    )


def draw_cycle_note() -> None:
    st.markdown(
        """
        <div class="cycle-row">
            <span class="cycle-chip">sense sectors</span>
            <span class="cycle-chip">assign reachable tasks</span>
            <span class="cycle-chip">work / relay / verify</span>
            <span class="cycle-chip">update congestion</span>
            <span class="cycle-chip">log metrics</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def download_payload(snapshot: dict, metrics_df: pd.DataFrame) -> None:
    st.download_button(
        "Snapshot JSON",
        data=json.dumps(snapshot, indent=2),
        file_name="hive_abm_snapshot.json",
        mime="application/json",
        width="stretch",
    )
    st.download_button(
        "Metrics CSV",
        data=metrics_df.to_csv(index=False),
        file_name="hive_abm_ticks.csv",
        mime="text/csv",
        width="stretch",
    )
    st.download_button(
        "Config JSON",
        data=json.dumps(snapshot["config"], indent=2),
        file_name="hive_abm_config.json",
        mime="application/json",
        width="stretch",
    )


st.set_page_config(page_title="Hive ABM Spatial Workbench", layout="wide")
inject_css()
ensure_state_defaults()

with st.sidebar:
    st.header("Simulation")
    st.selectbox("Mode", list(RUN_MODES), key="run_mode", on_change=apply_mode_defaults)
    st.markdown(f'<div class="scenario-note">{MODE_NOTES[st.session_state.run_mode]}</div>', unsafe_allow_html=True)
    st.number_input("Seed", min_value=0, key="seed", on_change=reset_session)
    st.selectbox(
        "Map layer",
        list(MAP_LAYERS),
        key="map_layer",
    )
    st.caption(MAP_LAYER_NOTES[st.session_state.map_layer])

    with st.expander("Advanced setup", expanded=False):
        st.selectbox(
            "Topology",
            list(TOPOLOGY_LABELS),
            key="topology",
            format_func=lambda item: TOPOLOGY_LABELS[item],
            on_change=reset_session,
        )
        st.selectbox(
            "Policy",
            list(POLICY_LABELS),
            key="policy_level",
            format_func=lambda item: POLICY_LABELS[item],
            on_change=reset_session,
        )
        st.slider("Rows", 6, 40, key="rows", on_change=reset_session)
        st.slider("Columns", 6, 40, key="cols", on_change=reset_session)
        st.slider("Agents", 3, 180, key="n_nodes", on_change=reset_session)
        st.slider("Tasks", 5, 180, key="n_tasks", on_change=reset_session)
        st.slider("Max steps", 10, 250, key="max_steps", on_change=reset_session)
        st.slider("Message loss", 0.0, 0.35, step=0.01, key="message_loss_probability", on_change=reset_session)
        st.slider("Node failure", 0.0, 0.05, step=0.001, key="node_failure_probability", on_change=reset_session)

    st.header("Show")
    st.checkbox("Agents", key="show_agents")
    st.checkbox("Tasks", key="show_tasks")
    st.checkbox("Assignment lines", key="show_assignment_links")
    st.checkbox("Communication graph", key="show_topology_edges")
    st.checkbox("Completed tasks", key="show_completed_tasks")

session = ensure_session()
snapshot = session.snapshot(include_layers=True)
actual_layer = MAP_LAYERS[st.session_state.map_layer]

st.title("Hive ABM Spatial Workbench")
draw_reading_note(st.session_state.map_layer)
draw_cycle_note()

run_cols = st.columns([1, 1, 1, 1])
if run_cols[0].button("Advance one cycle", type="primary", width="stretch"):
    snapshot = session.step()
if run_cols[1].button("Run 10 cycles", width="stretch"):
    snapshot = session.run_steps(10)
if run_cols[2].button("Run full mission", width="stretch"):
    snapshot = session.run_to_end()
if run_cols[3].button("Reset mission", width="stretch"):
    reset_session()
    session = ensure_session()
    snapshot = session.snapshot(include_layers=True)

metrics = snapshot.get("latest_tick", {})
counts = task_status_counts(snapshot)
unresolved = counts.get("pending", 0) + counts.get("in_progress", 0) + counts.get("awaiting_verification", 0)
top = st.columns(6)
top[0].metric("Cycle", snapshot["step"])
top[1].metric("Completed", f"{100 * metrics.get('completion_rate', 0.0):.0f}%")
top[2].metric("Verified", f"{100 * metrics.get('verified_completion_rate', 0.0):.0f}%")
top[3].metric("Open Tasks", f"{unresolved}/{len(snapshot['tasks'])}")
top[4].metric("Network Load", f"{metrics.get('mean_congestion', 0.0):.3f}")
top[5].metric("Busy Agents", f"{active_node_count(snapshot)}/{len(snapshot['nodes'])}")

rows = int(snapshot["config"]["rows"])
cols = int(snapshot["config"]["cols"])
st.session_state.inspect_row = min(int(st.session_state.inspect_row), rows - 1)
st.session_state.inspect_col = min(int(st.session_state.inspect_col), cols - 1)

left, right = st.columns([3.15, 1.05], gap="large")
with left:
    st.markdown('<div class="section-title">Grid</div>', unsafe_allow_html=True)
    selected_cell = (int(st.session_state.inspect_row), int(st.session_state.inspect_col))
    st.plotly_chart(
        grid_figure(
            snapshot,
            actual_layer,
            st.session_state.show_agents,
            st.session_state.show_tasks,
            st.session_state.show_assignment_links,
            st.session_state.show_topology_edges,
            st.session_state.show_completed_tasks,
            selected_cell,
        ),
        width="stretch",
    )

with right:
    st.markdown('<div class="section-title">Legend</div>', unsafe_allow_html=True)
    draw_visual_legend(st.session_state.show_assignment_links)

    st.markdown('<div class="section-title">Inspect Cell</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.number_input("Row", min_value=0, max_value=rows - 1, key="inspect_row")
    c2.number_input("Column", min_value=0, max_value=cols - 1, key="inspect_col")
    row = int(st.session_state.inspect_row)
    col = int(st.session_state.inspect_col)
    layer_value = snapshot["layers"][actual_layer][row][col]
    st.metric(st.session_state.map_layer, f"{layer_value:.3f}")
    agent_df, task_df = cell_tables(snapshot, row, col)
    st.caption("Agents in sector")
    if agent_df.empty:
        st.write("None")
    else:
        st.dataframe(agent_df, hide_index=True, width="stretch")
    st.caption("Tasks in sector")
    if task_df.empty:
        st.write("None")
    else:
        st.dataframe(task_df, hide_index=True, width="stretch")

    with st.expander("Interventions", expanded=False):
        if st.button("Insert relay at pressure peak", width="stretch"):
            session.apply_intervention("add_relay")
            st.rerun()
        if st.button("Fail 5 random agents", width="stretch"):
            session.apply_intervention("fail_nodes", count=5)
            st.rerun()
        noise_amount = st.slider("Noise amount", 0.05, 1.0, 0.25)
        noise_radius = st.slider("Noise radius", 1, 8, 2)
        if st.button("Inject center noise", width="stretch"):
            session.apply_intervention("inject_noise", radius=noise_radius, amount=noise_amount)
            st.rerun()
        if st.button("Add central task cluster", width="stretch"):
            session.apply_intervention("add_task_cluster", count=12, radius=2)
            st.rerun()

metrics_df = metric_frame(session)

lower_left, lower_right = st.columns([2.1, 1], gap="large")
with lower_left:
    st.markdown('<div class="section-title">Mission Trajectory</div>', unsafe_allow_html=True)
    st.plotly_chart(metric_figure(metrics_df), width="stretch")

with lower_right:
    st.markdown('<div class="section-title">Task Queues</div>', unsafe_allow_html=True)
    st.dataframe(queue_frame(snapshot), hide_index=True, width="stretch")

    with st.expander("Active assignment details", expanded=False):
        assignments = assignment_frame(snapshot)
        if assignments.empty:
            st.write("No active assignments.")
        else:
            st.dataframe(assignments, hide_index=True, width="stretch")

    with st.expander("Export", expanded=False):
        download_payload(snapshot, metrics_df)

with st.expander("Intervention log", expanded=False):
    st.dataframe(pd.DataFrame([asdict(record) for record in session.interventions]), width="stretch")
