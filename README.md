# Hive ABM

This is a compact, dependency-free Python prototype for modeling decentralized distributed AI-agent coordination as an agent-based model.

> Hive nodes are both cognitive agents and resource-bearing infrastructure. They can contribute reasoning, compute, bandwidth, storage, sensing, relay capacity, and verification, but they can also impose congestion, duplicated work, latency, and policy overhead.

The ABM models the **coordination layer**, not token-level LLM cognition. It supports controlled computational experiments over topology, policy confinement, memory, spatial density, verification, and communication constraints.

## Structure

```text
hive_abm/
  actions.py       # action labels and Action dataclass
  agents.py        # single HiveNode class + ResourceProfile
  environment.py   # 2D sector map and resource layers
  fields.py        # 1D/2D difference-array kernels
  memory.py        # local/shared memory artifacts with provenance
  metrics.py       # tick metrics and run summaries
  mission.py       # task graph, work phases, verification lifecycle
  policy.py        # policy confinement regimes
  simulation.py    # main simulation loop
  topology.py      # centralized/hierarchical/mesh/adaptive graphs
  windows.py       # sliding-window kernels
experiments/
  animate_grid.py
  single_run.py
  density_vs_congestion.py
  plot_results.py
  run_sweeps.py
  topology_policy_factorial.py
tests/
  test_fields.py
  test_simulation.py
```

## Key abstractions

### One agent class

There is only one agent/node class:

```python
HiveNode
```

`relay`, `worker`, `verifier`, `planner`, `observer`, and `coordinator` are **roles**, not subclasses. This keeps the model simple and allows role assignment, misallocation, and adaptive reassignment to become experimental variables.

### Role vs specialization

- `role`: what the system currently authorizes/expects the node to do.
- `specialization`: what the node is technically good at doing.
- `policy`: what the node is allowed to do.

This lets the model represent nodes that are capable but unauthorized, or authorized but poorly specialized.

### Resource fields

The spatial environment stores layered fields:

- `bandwidth_capacity`
- `bandwidth_demand`
- `compute_capacity`
- `compute_demand`
- `sensor_coverage`
- `noise`
- `congestion`
- `mission_pressure`

Agent placement and actions update these layers. The model can then measure coverage, congestion, bottlenecks, safe corridors, and weak deployment zones.

### Mission lifecycle

Tasks are not instantaneous. Assignment creates multi-tick work with explicit service time. Under mission-critical policy, a successful solve produces an artifact and moves the task to `awaiting_verification`; an independent verifier must then access the artifact and complete a verification phase before the task counts as completed.

### Topology-aware coordination

Communication topology is causal in the model. Task assignment depends on either local task sensing or a reachable coordinator/planner/delegation path. Assignment hop length increases message cost, route loss, and success friction. Shared memory artifacts carry writer provenance, so memory benefits depend on graph-local reachability rather than global access.

## Algorithmic kernels

The uploaded exercises are encoded as reusable ABM primitives:

- 1D range updates -> `fields.apply_range_add_1d`
- 2D rectangular range updates -> `fields.apply_rect_add_2d`
- fixed-window relay insertion -> `windows.min_fixed_window`
- safe corridor under threshold -> `windows.max_window_under_threshold`

## Run a demo

Run from the project root:

```bash
python experiments/single_run.py
```

This writes tick-level metrics to:

```text
outputs/single_run_ticks.csv
```

## Install plotting/test dependencies

The simulator itself uses the Python standard library. Tests and plots use a small dev dependency set:

```bash
python -m pip install -e ".[dev]"
```

Or:

```bash
python -m pip install -r requirements-dev.txt
```

## Run sweep suites

The main reproducible runner writes both run summaries and tick-level traces:

```bash
python experiments/run_sweeps.py --preset quick
```

Outputs:

```text
outputs/sweeps/summaries.csv
outputs/sweeps/summaries_flat.csv
outputs/sweeps/ticks.csv
```

Presets:

- `quick`: small exploratory run, intended for fast local plotting.
- `standard`: larger replication count for more stable figures.
- `paper`: high replication count for publication-grade robustness checks.

You can restrict the suite:

```bash
python experiments/run_sweeps.py --preset standard --experiments density topology_policy
python experiments/run_sweeps.py --preset standard --experiments communication verification
```

You can override replications:

```bash
python experiments/run_sweeps.py --preset quick --replications 5
```

## Plot results

After running a sweep:

```bash
python experiments/plot_results.py --input-dir outputs/sweeps
```

Plots are written to:

```text
outputs/sweeps/plots/
```

Generated figures include density line plots, topology-policy heatmaps, communication-loss line plots, and verification queue time series.

## Export a scientific grid animation

For a scientific animated 2D grid plot, use:

```bash
python experiments/animate_grid.py
```

This writes:

```text
outputs/animations/grid_replay.gif
```

Useful variants:

```bash
python experiments/animate_grid.py --layer mission_pressure
python experiments/animate_grid.py --topology adaptive --policy-level mission_critical --layer congestion
python experiments/animate_grid.py --output outputs/animations/grid_replay.html
```

The animation shows a selected spatial field as a heatmap, agents colored by role, task markers by state, communication edges, and synchronized metric traces.

## Legacy experiments

```bash
python experiments/density_vs_congestion.py --replications 10
python experiments/topology_policy_factorial.py --replications 10
```

Outputs are written to `outputs/*.csv`.

## Thesis use

This MVP supports the following thesis-level experiments:

1. **Node density vs congestion**: identify the coordination carrying capacity where adding more agents stops helping and begins hurting.
2. **Topology x policy confinement**: compare centralized, hierarchical, mesh, and adaptive topologies under permissive, moderate, strict, and mission-critical policy regimes.
3. **Spatial resource bottlenecks**: use resource fields to identify peak congestion, weak bandwidth zones, safe corridors, and undercovered sectors.
4. **Role-specialization alignment**: vary `role_specialization_alignment` to study misallocation and adaptive reassignment.
5. **Verification bottlenecks**: measure queues and delays introduced by mission-critical artifact verification.
6. **Memory locality and relay dependence**: compare local, shared, and hybrid memory under constrained communication horizons.

## Calibration to real Hive traces

Later, deployed Hive logs can parameterize the ABM using empirical estimates such as:

- per-role task success rates
- tool-denial rates under policy levels
- message counts and handoff rates
- latency distributions by topology
- failure/retry rates
- memory read/write frequency
- artifact quality scores
- verification acceptance/rejection rates

This closes the loop between the real Hive system and the ABM.
