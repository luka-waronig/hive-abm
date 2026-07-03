"""Minimal runnable Hive ABM example."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hive_abm.simulation import HiveSimulation, SimulationConfig


def main() -> None:
    cfg = SimulationConfig(
        rows=15,
        cols=15,
        n_nodes=35,
        n_tasks=45,
        max_steps=90,
        topology="decentralized_mesh",
        policy_level="moderate",
        seed=7,
        run_id="single_demo",
    )
    sim = HiveSimulation(cfg)
    summary = sim.run()
    sim.metrics.to_csv(PROJECT_ROOT / "outputs" / "single_run_ticks.csv")
    print(summary)
    print("Tick metrics written to outputs/single_run_ticks.csv")


if __name__ == "__main__":
    main()
