# Experiment Protocol

## Reproducibility Requirements

Every run should record:

- scenario name
- environment preset
- full `SimulationConfig`
- random seed
- model fingerprint
- topology
- policy level
- replication index
- intervention log, if any

Use common random numbers when comparing conditions: keep the task/environment seed fixed and vary only the experimental factor.

## Professor-Ready Figure Set

Minimum figure set:

1. Density vs completion with confidence intervals.
2. Density vs mean/peak congestion with confidence intervals.
3. Topology x policy completion heatmap.
4. Topology x policy verified-completion heatmap.
5. Communication loss x assignment horizon phase diagram.
6. Verification queue over time under mission-critical policy.
7. Spatial animation frame or GIF showing local congestion/mission pressure and agent movement.

## Interpretation Rules

- Do not interpret raw node counts literally until calibrated.
- Prefer normalized quantities: tasks per node, nodes per sector, capacity-demand ratio, mean degree, and message loss.
- Treat quick preset plots as exploratory.
- Use standard or paper presets for claims.
- Report confidence intervals or bootstrap intervals for sweep plots.

## Calibration Targets

Future Hive logs should estimate:

- agents per mission
- tasks per mission
- messages per task
- solve and verification durations
- verification queue delay
- tool denial frequency
- artifact retrieval latency/failure
- node disconnect/reconnect frequency
- policy-blocked action rates
- mission completion and failure rates

## Interactive Protocol

For dashboard/intervention runs:

1. Select scenario/environment.
2. Reset with a fixed seed.
3. Run to a named condition or step count.
4. Apply a recorded intervention.
5. Continue the run.
6. Export snapshot JSON and result CSV.

Interventions must be included in figure captions when used.
