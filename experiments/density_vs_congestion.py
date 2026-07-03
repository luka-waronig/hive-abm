"""Experiment: node density versus congestion and mission performance.

Run from the project root:
    python experiments/density_vs_congestion.py --replications 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hive_abm.metrics import write_summaries_csv
from hive_abm.simulation import HiveSimulation, SimulationConfig


def run_experiment(replications: int, output: Path, seed: int) -> None:
    densities = [10, 25, 50, 100, 160]
    summaries = []
    for n_nodes in densities:
        for rep in range(replications):
            cfg = SimulationConfig(
                rows=20,
                cols=20,
                n_nodes=n_nodes,
                n_tasks=70,
                max_steps=120,
                topology="decentralized_mesh",
                policy_level="moderate",
                memory_mode="local_and_shared",
                seed=seed + 1000 * n_nodes + rep,
                run_id=f"density_n{n_nodes}_rep{rep}",
                metadata={"experiment": "density_vs_congestion", "replication": rep},
            )
            sim = HiveSimulation(cfg)
            summary = sim.run()
            summaries.append(summary)
            print(
                f"{summary.run_id}: success={summary.success} completion={summary.final_completion_rate:.2f} "
                f"peak_congestion={summary.final_peak_congestion:.2f} messages={summary.total_messages}"
            )
    write_summaries_csv(summaries, output)
    print(f"Wrote {len(summaries)} summaries to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=5)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "density_vs_congestion.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment(args.replications, args.output, args.seed)


if __name__ == "__main__":
    main()
