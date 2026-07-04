# Hive ABM ODD Model Description

## Purpose

The model studies decentralized Hive-like agent coordination under policy, topology, spatial resource constraints, verification requirements, failures, and communication loss.

## Entities

- **Hive nodes**: heterogeneous computational agents with role, specialization, permissions, resource profile, reliability, trust, memory, state, and grid position.
- **Tasks**: mission subtasks with type, sector, difficulty, value, dependencies, status, assignment, work remaining, and verification state.
- **Spatial environment**: a 2D sector grid with layered fields for capacity, demand, coverage, noise, congestion, and mission pressure.
- **Communication topology**: graph overlay connecting agents for assignment, memory access, and coordination.
- **Policy regime**: capability authorization layer controlling allowed actions.
- **Artifacts/memory**: task outputs with provenance and graph-local reachability.

## State Variables

Agent variables include `role`, `specialization`, `position`, `permissions`, `resources`, `trust`, `reliability`, `state`, `current_task_id`, `current_load`, and local memory.

Task variables include `status`, `assigned_to`, `active_phase`, `work_remaining`, `attempts`, `verification_attempts`, `quality`, and `verified`.

Environment variables include bandwidth/compute capacity and demand, coverage, noise, congestion, and mission pressure.

## Process Overview

Each step:

1. Adaptive topology may be rebuilt from current positions.
2. The environment recomputes resource layers from node footprints and unresolved tasks.
3. Available solve/verify tasks are assigned subject to policy and graph reachability.
4. Nodes act: solve, verify, relay, observe, move, wait, block, or fail.
5. Task work advances; completed solve artifacts may require verification.
6. Environment layers are recomputed.
7. tick-level metrics are logged.

## Design Concepts

- **Heterogeneity**: nodes differ by role, competence, resources, permissions, reliability, trust, and position.
- **Local sensing**: agents observe local environmental fields and task opportunities through spatial radius and graph reachability.
- **Interaction**: nodes interact through shared resource fields, communication topology, memory artifacts, and policy gates.
- **Emergence**: mission completion, congestion, verification queues, and coordination instability emerge from local actions and constraints.
- **Stochasticity**: randomness enters positions, task sectors/types/dependencies, competence, reliability, failures, route loss, and task outcomes.
- **Observation**: model outputs include completion, verified completion, congestion, messages, denials, graph structure, queue dynamics, and spatial fields.

## Initialization

Scenarios configure grid size, node count, task count, topology, policy, memory mode, failure/message loss rates, task duration, and verification duration. Environment presets can introduce clustered tasks, noisy regions, resource scarcity, or corridor structure.

## Input Data

The current model is synthetic. Future calibration should use Hive logs for message counts, task durations, tool denials, verification delays, handoff latency, artifact retrieval, node disconnects, and mission outcomes.

## Submodels

- **Task assignment**: greedy policy-constrained and graph-reachable assignment based on competence, distance, route hops, load, and role match.
- **Task success**: probabilistic function of competence, difficulty, coverage, memory, congestion, policy friction, route hops, reliability, and message loss.
- **Congestion**: local excess demand over capacity plus noise.
- **Verification**: mission-critical solves produce artifacts that require independent verification before completion.
- **Memory access**: local/shared artifacts are useful only when reachable by local memory or graph hops.
