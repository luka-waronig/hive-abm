"""Primary factorial experiment: topology × policy confinement.

Run from the project root:
    python experiments/topology_policy_factorial.py --replications 10
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
    topologies = ["centralized", "hierarchical", "decentralized_mesh", "adaptive"]
    policies = ["permissive", "moderate", "strict", "mission_critical"]
    summaries = []
    idx = 0
    for topology in topologies:
        for policy in policies:
            for rep in range(replications):
                idx += 1
                cfg = SimulationConfig(
                    rows=20,
                    cols=20,
                    n_nodes=60,
                    n_tasks=80,
                    max_steps=130,
                    topology=topology,
                    policy_level=policy,
                    memory_mode="local_and_shared",
                    seed=seed + idx,
                    run_id=f"factorial_{topology}_{policy}_rep{rep}",
                    metadata={"experiment": "topology_policy_factorial", "replication": rep},
                )
                sim = HiveSimulation(cfg)
                summary = sim.run()
                summaries.append(summary)
                print(
                    f"{summary.run_id}: success={summary.success} completion={summary.final_completion_rate:.2f} "
                    f"denied={summary.total_denied_actions} centralization={summary.graph_centralization:.2f}"
                )
    write_summaries_csv(summaries, output)
    print(f"Wrote {len(summaries)} summaries to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=5)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "topology_policy_factorial.csv")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    run_experiment(args.replications, args.output, args.seed)


if __name__ == "__main__":
    main()
