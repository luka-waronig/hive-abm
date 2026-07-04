"""Interactive simulation session and intervention operators."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .agents import HiveNode
from .mission import Task
from .policy import make_policy
from .scenarios import build_simulation
from .simulation import HiveSimulation, SimulationConfig
from .topology import build_topology


@dataclass
class InterventionRecord:
    step: int
    kind: str
    params: dict


@dataclass
class SimulationSession:
    """Stateful wrapper for chunked runs, interventions, and reproducible export."""

    scenario: str = "baseline"
    environment_preset: str = "uniform"
    seed: int = 42
    config_overrides: dict = field(default_factory=dict)
    sim: HiveSimulation = field(init=False)
    interventions: List[InterventionRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.reset(self.scenario, self.environment_preset, self.seed, **self.config_overrides)

    def reset(self, scenario: Optional[str] = None, environment_preset: Optional[str] = None, seed: Optional[int] = None, **overrides) -> None:
        if scenario is not None:
            self.scenario = scenario
        if environment_preset is not None:
            self.environment_preset = environment_preset
        if seed is not None:
            self.seed = seed
        self.config_overrides = {**self.config_overrides, **overrides}
        self.sim = build_simulation(self.scenario, self.environment_preset, self.seed, **self.config_overrides)
        self.interventions = []

    def step(self) -> dict:
        if self.sim.metrics.ticks and self.sim.mission.is_success():
            return self.snapshot()
        self.sim.step_idx = len(self.sim.metrics.ticks)
        self.sim.step()
        return self.snapshot()

    def run_steps(self, n: int) -> dict:
        for _ in range(max(0, n)):
            self.step()
            if self.sim.mission.is_success():
                break
        return self.snapshot()

    def run_to_end(self) -> dict:
        remaining = max(0, self.sim.config.max_steps - len(self.sim.metrics.ticks))
        return self.run_steps(remaining)

    def run_until(self, condition: str | Callable[[HiveSimulation], bool], max_steps: Optional[int] = None) -> dict:
        limit = max_steps if max_steps is not None else self.sim.config.max_steps
        for _ in range(max(0, limit)):
            if self._condition_met(condition):
                break
            self.step()
            if self.sim.mission.is_success():
                break
        return self.snapshot()

    def _condition_met(self, condition: str | Callable[[HiveSimulation], bool]) -> bool:
        if callable(condition):
            return bool(condition(self.sim))
        condition = condition.strip()
        latest = self.sim.metrics.ticks[-1] if self.sim.metrics.ticks else None
        values = {
            "completion_rate": self.sim.mission.completion_rate(),
            "verified_completion_rate": self.sim.mission.verified_completion_rate(),
            "mean_congestion": latest.mean_congestion if latest else 0.0,
            "peak_congestion": latest.peak_congestion if latest else 0.0,
            "awaiting_verification_tasks": latest.awaiting_verification_tasks if latest else 0,
        }
        for op in (">=", "<=", ">", "<", "=="):
            if op in condition:
                left, right = [part.strip() for part in condition.split(op, 1)]
                observed = values[left]
                target = float(right)
                return {
                    ">=": observed >= target,
                    "<=": observed <= target,
                    ">": observed > target,
                    "<": observed < target,
                    "==": observed == target,
                }[op]
        raise ValueError(f"unsupported condition: {condition}")

    def apply_intervention(self, kind: str, **params) -> dict:
        kind = kind.lower()
        if kind == "add_relay":
            self._add_relay(params.get("position"))
        elif kind == "fail_nodes":
            self._fail_nodes(params.get("ids"), int(params.get("count", 0)))
        elif kind == "inject_noise":
            self._inject_noise(params)
        elif kind == "switch_topology":
            self._switch_topology(str(params["topology"]))
        elif kind == "switch_policy":
            self._switch_policy(str(params["policy_level"]))
        elif kind == "add_task_cluster":
            self._add_task_cluster(params)
        else:
            raise ValueError(f"unknown intervention: {kind}")
        self.interventions.append(InterventionRecord(step=len(self.sim.metrics.ticks), kind=kind, params=dict(params)))
        return self.snapshot()

    def _add_relay(self, position: Optional[Tuple[int, int]]) -> None:
        node_id = max((node.node_id for node in self.sim.nodes), default=-1) + 1
        if position is None:
            _, r, c = self.sim.environment.layer_max("mission_pressure")
            position = (r if r >= 0 else 0, c if c >= 0 else 0)
        relay = HiveNode(
            node_id=node_id,
            role="relay",
            specialization=self.sim._make_specialization("relay"),
            position=position,
            permissions=set(self.sim.policy_permissions_for_role_hint("relay")),
            resources=self.sim._make_resources("relay"),
        )
        self.sim.nodes.append(relay)
        self.sim.config.n_nodes = len(self.sim.nodes)
        self._rebuild_topology()

    def _fail_nodes(self, ids: Optional[List[int]], count: int) -> None:
        if ids is None:
            candidates = [node for node in self.sim.nodes if node.state != "failed"]
            self.sim.rng.shuffle(candidates)
            ids = [node.node_id for node in candidates[: max(0, count)]]
        for node in self.sim.nodes:
            if node.node_id in ids:
                node.state = "failed"
                if node.current_task_id is not None:
                    self.sim.mission.release(node.current_task_id)
                    node.current_task_id = None
        self._rebuild_topology()

    def _inject_noise(self, params: dict) -> None:
        r = int(params.get("row", self.sim.environment.rows // 2))
        c = int(params.get("col", self.sim.environment.cols // 2))
        radius = int(params.get("radius", 2))
        amount = float(params.get("amount", 0.25))
        self.sim.environment.add_noise_region(r - radius, c - radius, r + radius, c + radius, amount)

    def _switch_topology(self, topology: str) -> None:
        self.sim.config.topology = topology
        self._rebuild_topology()

    def _switch_policy(self, policy_level: str) -> None:
        self.sim.config.policy_level = policy_level
        self.sim.policy = make_policy(policy_level)

    def _add_task_cluster(self, params: dict) -> None:
        row = int(params.get("row", self.sim.environment.rows // 2))
        col = int(params.get("col", self.sim.environment.cols // 2))
        count = int(params.get("count", 10))
        radius = int(params.get("radius", 2))
        start_id = max(self.sim.mission.tasks.keys(), default=-1) + 1
        for offset in range(count):
            sector = (
                max(0, min(self.sim.environment.rows - 1, row + self.sim.rng.randint(-radius, radius))),
                max(0, min(self.sim.environment.cols - 1, col + self.sim.rng.randint(-radius, radius))),
            )
            task_type = self.sim.rng.choice(("planning", "coding", "verification", "routing", "retrieval", "analysis"))
            self.sim.mission.tasks[start_id + offset] = Task(
                task_id=start_id + offset,
                task_type=task_type,
                sector=sector,
                difficulty=self.sim.rng.uniform(0.25, 0.85),
                value=self.sim.rng.uniform(0.8, 1.5),
                required_role="verifier" if task_type == "verification" else None,
            )
        self.sim.config.n_tasks = len(self.sim.mission.tasks)

    def _rebuild_topology(self) -> None:
        self.sim.adjacency = build_topology(self.sim.nodes, self.sim.config.topology, self.sim.rng)

    def snapshot(self, include_layers: bool = True) -> dict:
        snap = self.sim.snapshot(include_layers=include_layers)
        snap["session"] = {
            "scenario": self.scenario,
            "environment_preset": self.environment_preset,
            "seed": self.seed,
            "interventions": [asdict(record) for record in self.interventions],
        }
        return snap

    def fork(self) -> "SimulationSession":
        return copy.deepcopy(self)

    def export_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(include_layers=True), indent=2), encoding="utf-8")
