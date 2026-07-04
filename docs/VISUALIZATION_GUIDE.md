# Visualization Guide

## Principle

Each figure should communicate one mechanism at a time. Avoid combining role colors, topology edges, task states, and multiple fields unless the figure is explicitly diagnostic.

## Recommended Thesis Figures

Use static figures for claims:

- density vs completion with confidence intervals
- normalized task load vs completion
- spatial density vs congestion
- topology x policy heatmaps
- communication loss x assignment horizon lines
- verification queue time series
- environment preset comparison

Use animations for mechanism explanation:

- congestion field with agents and unresolved tasks
- mission pressure field with agent movement
- verification bottleneck under mission-critical policy

## Dashboard Defaults

The dashboard defaults are intended for live exploration:

- heatmap shows one human-named map layer: mission demand, network load, sensor coverage, bandwidth capacity, or interference
- agents are aggregated by sector rather than drawn as every individual node by default
- tasks are aggregated by sector rather than drawn as every individual task by default
- active assignment lines are hidden by default and only shown as a diagnostic overlay
- topology edges are hidden by default
- role colors are hidden by default; role is available in hover/inspection tables
- the sector inspector reports agents, tasks, and selected field value for one grid cell
- intervention controls are grouped separately from view controls
- all state-changing interventions are logged

## Animation Defaults

The `thesis` animation style is the default. It uses:

- one spatial field heatmap
- neutral agent markers
- task state markers
- no topology edges by default
- compact metric traces

Use `--style diagnostic --show-edges --color-by-role` only for debugging or methods appendices.

## Captioning Rule

Every professor-facing figure should state:

- scenario
- environment preset
- topology
- policy level
- seed or replication count
- whether the plot is exploratory or replicated
- what the x-axis means physically or why it is abstract

Raw node counts should be interpreted carefully. Prefer normalized quantities such as tasks per node, nodes per sector, mean degree, and capacity-demand ratio.
