"""Named scenarios and environment presets for thesis-ready experiments."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .environment import SpatialEnvironment
from .mission import MissionGraph, Task, make_random_mission
from .simulation import DEFAULT_TASK_TYPES, HiveSimulation, SimulationConfig


SCENARIOS = (
    "baseline",
    "sparse",
    "dense",
    "mission_critical",
    "noisy",
    "verification_bottleneck",
    "relay_stress",
)

ENVIRONMENTS = (
    "uniform",
    "clustered_tasks",
    "noisy_region",
    "resource_scarce_zone",
    "corridor",
)


@dataclass(frozen=True)
class ScenarioSpec:
    """Reproducible scenario descriptor."""

    name: str = "baseline"
    environment: str = "uniform"
    seed: int = 42
    config: SimulationConfig = field(default_factory=SimulationConfig)

    def to_dict(self) -> dict:
        row = asdict(self)
        row["config"] = asdict(self.config)
        return row


def scenario_config(name: str = "baseline", seed: int = 42, **overrides) -> SimulationConfig:
    """Return a SimulationConfig for a named scenario."""

    name = name.lower()
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {name}")

    cfg = SimulationConfig(seed=seed, run_id=f"{name}_seed{seed}", metadata={"scenario": name})
    if name == "sparse":
        cfg = replace(cfg, n_nodes=16, n_tasks=70, topology="decentralized_mesh", message_loss_probability=0.03)
    elif name == "dense":
        cfg = replace(cfg, n_nodes=100, n_tasks=80, topology="decentralized_mesh")
    elif name == "mission_critical":
        cfg = replace(cfg, n_nodes=60, n_tasks=80, policy_level="mission_critical", verification_duration_max=3)
    elif name == "noisy":
        cfg = replace(cfg, n_nodes=60, n_tasks=80, message_loss_probability=0.10, assignment_horizon_hops=2)
    elif name == "verification_bottleneck":
        cfg = replace(cfg, n_nodes=60, n_tasks=90, policy_level="mission_critical", verification_duration_max=6)
    elif name == "relay_stress":
        cfg = replace(cfg, n_nodes=45, n_tasks=80, topology="decentralized_mesh", assignment_horizon_hops=1, memory_horizon_hops=2)

    if overrides:
        cfg = replace(cfg, **overrides)
    cfg.metadata = {**cfg.metadata, "scenario": name}
    return cfg


def clustered_mission(
    rows: int,
    cols: int,
    n_tasks: int,
    rng: random.Random,
    clusters: int = 3,
    spread: int = 3,
    task_types: Sequence[str] = DEFAULT_TASK_TYPES,
    timeout_steps: int = 100,
) -> MissionGraph:
    """Generate a mission with spatial task clusters instead of uniform placement."""

    centers = [(rng.randrange(rows), rng.randrange(cols)) for _ in range(max(1, clusters))]
    tasks: List[Task] = []
    for task_id in range(n_tasks):
        cr, cc = rng.choice(centers)
        sector = (
            max(0, min(rows - 1, cr + rng.randint(-spread, spread))),
            max(0, min(cols - 1, cc + rng.randint(-spread, spread))),
        )
        task_type = rng.choice(task_types)
        deps = [j for j in range(task_id) if rng.random() < 0.10]
        difficulty = min(0.95, max(0.10, rng.betavariate(2.2, 2.8)))
        value = max(0.5, rng.lognormvariate(0.0, 0.30))
        tasks.append(
            Task(
                task_id=task_id,
                task_type=task_type,
                sector=sector,
                difficulty=difficulty,
                value=value,
                dependencies=deps,
                required_role="verifier" if task_type == "verification" else None,
            )
        )
    return MissionGraph.from_tasks(tasks, timeout_steps=timeout_steps)


def apply_environment_preset(environment: SpatialEnvironment, name: str) -> SpatialEnvironment:
    """Apply persistent environment perturbations."""

    name = name.lower()
    if name not in ENVIRONMENTS:
        raise ValueError(f"unknown environment preset: {name}")
    rows, cols = environment.rows, environment.cols
    if name == "noisy_region":
        environment.add_noise_region(rows // 3, cols // 3, min(rows - 1, rows // 3 + rows // 4), min(cols - 1, cols // 3 + cols // 4), 0.35)
    elif name == "resource_scarce_zone":
        environment.add_capacity_region(rows // 3, cols // 3, min(rows - 1, rows // 3 + rows // 4), min(cols - 1, cols // 3 + cols // 4), bandwidth=-0.55, compute=-0.35)
    elif name == "corridor":
        band = max(1, rows // 8)
        center = rows // 2
        environment.add_noise_region(0, 0, rows - 1, cols - 1, 0.18)
        environment.add_noise_region(max(0, center - band), 0, min(rows - 1, center + band), cols - 1, -0.14)
        environment.add_capacity_region(max(0, center - band), 0, min(rows - 1, center + band), cols - 1, bandwidth=0.6)
    return environment


def build_simulation(
    scenario: str = "baseline",
    environment_preset: str = "uniform",
    seed: int = 42,
    **overrides,
) -> HiveSimulation:
    """Build a HiveSimulation from named scenario/environment presets."""

    cfg = scenario_config(scenario, seed=seed, **overrides)
    rng = random.Random(seed)
    environment = apply_environment_preset(SpatialEnvironment(cfg.rows, cfg.cols), environment_preset)
    if environment_preset == "clustered_tasks":
        mission = clustered_mission(cfg.rows, cfg.cols, cfg.n_tasks, rng, timeout_steps=cfg.max_steps)
    else:
        mission = make_random_mission(cfg.rows, cfg.cols, cfg.n_tasks, rng, task_types=DEFAULT_TASK_TYPES, timeout_steps=cfg.max_steps)
    cfg.metadata = {**cfg.metadata, "environment_preset": environment_preset}
    return HiveSimulation(cfg, mission=mission, environment=environment)
