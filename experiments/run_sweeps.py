"""Run reproducible Hive ABM sweep suites.

Examples from the project root:
    python experiments/run_sweeps.py --preset quick
    python experiments/run_sweeps.py --preset standard --experiments density topology_policy
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from time import perf_counter
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hive_abm.metrics import summary_to_row, tick_rows, write_rows_csv, write_summaries_csv
from hive_abm.scenarios import ENVIRONMENTS, build_simulation
from hive_abm.simulation import HiveSimulation, SimulationConfig


TOPOLOGIES = ["centralized", "hierarchical", "decentralized_mesh", "adaptive"]
POLICIES = ["permissive", "moderate", "strict", "mission_critical"]


def model_fingerprint() -> str:
    """Hash package source files so results can be tied to a code version."""

    digest = hashlib.sha256()
    for path in sorted((PROJECT_ROOT / "hive_abm").glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def preset_replications(preset: str) -> int:
    return {"quick": 2, "standard": 10, "paper": 30}[preset]


def preset_steps(preset: str) -> int:
    return {"quick": 90, "standard": 130, "paper": 160}[preset]


def density_conditions(preset: str, replications: int, seed: int, max_steps: int) -> Iterable[SimulationConfig]:
    densities = [12, 25, 50, 90] if preset == "quick" else [10, 25, 50, 100, 160]
    for n_nodes in densities:
        for rep in range(replications):
            yield SimulationConfig(
                rows=20,
                cols=20,
                n_nodes=n_nodes,
                n_tasks=70,
                max_steps=max_steps,
                topology="decentralized_mesh",
                policy_level="moderate",
                memory_mode="local_and_shared",
                seed=seed + 10_000 + 1_000 * n_nodes + rep,
                run_id=f"density_n{n_nodes}_rep{rep}",
                metadata={
                    "experiment": "density",
                    "condition": f"n_nodes={n_nodes}",
                    "replication": rep,
                    "seed_family": seed + 10_000 + rep,
                    "n_nodes_sweep": n_nodes,
                },
            )


def topology_policy_conditions(preset: str, replications: int, seed: int, max_steps: int) -> Iterable[SimulationConfig]:
    for topology in TOPOLOGIES:
        for policy in POLICIES:
            for rep in range(replications):
                common_seed = seed + 20_000 + rep
                yield SimulationConfig(
                    rows=20,
                    cols=20,
                    n_nodes=60,
                    n_tasks=80,
                    max_steps=max_steps,
                    topology=topology,
                    policy_level=policy,
                    memory_mode="local_and_shared",
                    seed=common_seed,
                    run_id=f"topopol_{topology}_{policy}_rep{rep}",
                    metadata={
                        "experiment": "topology_policy",
                        "condition": f"topology={topology}|policy={policy}",
                        "replication": rep,
                        "common_random_seed": common_seed,
                    },
                )


def communication_conditions(preset: str, replications: int, seed: int, max_steps: int) -> Iterable[SimulationConfig]:
    losses = [0.0, 0.03, 0.08] if preset == "quick" else [0.0, 0.02, 0.05, 0.10, 0.16]
    horizons = [1, 3, 5] if preset == "quick" else [1, 2, 3, 5, 7]
    for loss in losses:
        for horizon in horizons:
            for rep in range(replications):
                common_seed = seed + 30_000 + rep
                yield SimulationConfig(
                    rows=20,
                    cols=20,
                    n_nodes=60,
                    n_tasks=80,
                    max_steps=max_steps,
                    topology="decentralized_mesh",
                    policy_level="moderate",
                    memory_mode="local_and_shared",
                    assignment_horizon_hops=horizon,
                    memory_horizon_hops=max(horizon, 2),
                    direct_handoff_horizon_hops=max(1, horizon - 1),
                    message_loss_probability=loss,
                    seed=common_seed,
                    run_id=f"comm_loss{loss:.2f}_h{horizon}_rep{rep}",
                    metadata={
                        "experiment": "communication",
                        "condition": f"loss={loss:.2f}|horizon={horizon}",
                        "replication": rep,
                        "common_random_seed": common_seed,
                        "message_loss_sweep": loss,
                        "assignment_horizon_sweep": horizon,
                    },
                )


def verification_conditions(preset: str, replications: int, seed: int, max_steps: int) -> Iterable[SimulationConfig]:
    durations = [1, 2, 4] if preset == "quick" else [1, 2, 3, 5, 8]
    for verification_max in durations:
        for rep in range(replications):
            common_seed = seed + 40_000 + rep
            yield SimulationConfig(
                rows=20,
                cols=20,
                n_nodes=60,
                n_tasks=80,
                max_steps=max_steps,
                topology="decentralized_mesh",
                policy_level="mission_critical",
                memory_mode="local_and_shared",
                verification_duration_min=1,
                verification_duration_max=verification_max,
                seed=common_seed,
                run_id=f"verify_dur{verification_max}_rep{rep}",
                metadata={
                    "experiment": "verification",
                    "condition": f"verification_duration_max={verification_max}",
                    "replication": rep,
                    "common_random_seed": common_seed,
                    "verification_duration_max_sweep": verification_max,
                },
            )


def environment_conditions(preset: str, replications: int, seed: int, max_steps: int) -> Iterable[SimulationConfig]:
    presets = ["uniform", "clustered_tasks", "noisy_region"] if preset == "quick" else list(ENVIRONMENTS)
    for environment_preset in presets:
        for rep in range(replications):
            common_seed = seed + 50_000 + rep
            yield SimulationConfig(
                rows=20,
                cols=20,
                n_nodes=60,
                n_tasks=80,
                max_steps=max_steps,
                topology="decentralized_mesh",
                policy_level="moderate",
                memory_mode="local_and_shared",
                seed=common_seed,
                run_id=f"environment_{environment_preset}_rep{rep}",
                metadata={
                    "experiment": "environment",
                    "condition": f"environment={environment_preset}",
                    "replication": rep,
                    "common_random_seed": common_seed,
                    "scenario": "baseline",
                    "environment_preset": environment_preset,
                },
            )


def build_conditions(experiments: List[str], preset: str, replications: int, seed: int) -> List[SimulationConfig]:
    max_steps = preset_steps(preset)
    builders = {
        "density": density_conditions,
        "topology_policy": topology_policy_conditions,
        "communication": communication_conditions,
        "verification": verification_conditions,
        "environment": environment_conditions,
    }
    conditions: List[SimulationConfig] = []
    for experiment in experiments:
        conditions.extend(builders[experiment](preset, replications, seed, max_steps))
    fingerprint = model_fingerprint()
    for cfg in conditions:
        tasks_per_node = cfg.n_tasks / cfg.n_nodes if cfg.n_nodes else 0.0
        nodes_per_sector = cfg.n_nodes / (cfg.rows * cfg.cols)
        cfg.metadata = {
            **cfg.metadata,
            "preset": preset,
            "model_fingerprint": fingerprint,
            "max_steps": cfg.max_steps,
            "rows": cfg.rows,
            "cols": cfg.cols,
            "tasks_per_node": tasks_per_node,
            "nodes_per_sector": nodes_per_sector,
            "node_sector_count": cfg.rows * cfg.cols,
        }
    return conditions


def run_conditions(conditions: List[SimulationConfig], output_dir: Path, write_ticks: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    summary_rows = []
    all_tick_rows = []
    started = perf_counter()
    for idx, cfg in enumerate(conditions, start=1):
        if "environment_preset" in cfg.metadata or "scenario" in cfg.metadata:
            sim = build_simulation(
                str(cfg.metadata.get("scenario", "baseline")),
                str(cfg.metadata.get("environment_preset", "uniform")),
                seed=cfg.seed,
                rows=cfg.rows,
                cols=cfg.cols,
                n_nodes=cfg.n_nodes,
                n_tasks=cfg.n_tasks,
                max_steps=cfg.max_steps,
                topology=cfg.topology,
                policy_level=cfg.policy_level,
                memory_mode=cfg.memory_mode,
                assignment_horizon_hops=cfg.assignment_horizon_hops,
                memory_horizon_hops=cfg.memory_horizon_hops,
                direct_handoff_horizon_hops=cfg.direct_handoff_horizon_hops,
                message_loss_probability=cfg.message_loss_probability,
                node_failure_probability=cfg.node_failure_probability,
                verification_duration_min=cfg.verification_duration_min,
                verification_duration_max=cfg.verification_duration_max,
                run_id=cfg.run_id,
                metadata=cfg.metadata,
            )
        else:
            sim = HiveSimulation(cfg)
        summary = sim.run()
        summaries.append(summary)
        row = summary_to_row(summary)
        summary_rows.append(row)
        if write_ticks:
            all_tick_rows.extend(tick_rows(summary.run_id, sim.metrics.ticks, row))
        elapsed = perf_counter() - started
        print(
            f"[{idx}/{len(conditions)}] {summary.run_id}: "
            f"completion={summary.final_completion_rate:.2f} "
            f"verified={summary.final_verified_completion_rate:.2f} "
            f"congestion={summary.final_peak_congestion:.2f} "
            f"messages={summary.total_messages} elapsed={elapsed:.1f}s"
        )

    write_summaries_csv(summaries, output_dir / "summaries.csv")
    write_rows_csv(summary_rows, output_dir / "summaries_flat.csv")
    if write_ticks:
        write_rows_csv(all_tick_rows, output_dir / "ticks.csv")
    print(f"Wrote summaries to {output_dir / 'summaries.csv'}")
    print(f"Wrote flat summaries to {output_dir / 'summaries_flat.csv'}")
    if write_ticks:
        print(f"Wrote ticks to {output_dir / 'ticks.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["quick", "standard", "paper"], default="quick")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["density", "topology_policy", "communication", "verification", "environment", "all"],
        default=["all"],
    )
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sweeps")
    parser.add_argument("--no-ticks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = ["density", "topology_policy", "communication", "verification", "environment"] if "all" in args.experiments else args.experiments
    replications = args.replications if args.replications is not None else preset_replications(args.preset)
    conditions = build_conditions(experiments, args.preset, replications, args.seed)
    print(f"Running {len(conditions)} conditions with preset={args.preset}, replications={replications}")
    run_conditions(conditions, args.output_dir, write_ticks=not args.no_ticks)


if __name__ == "__main__":
    main()
