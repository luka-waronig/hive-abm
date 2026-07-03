"""Main Hive ABM simulation loop."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .actions import Action, MOVE, OBSERVE, RELAY, SOLVE, VERIFY, WAIT
from .agents import HiveNode, ResourceProfile
from .environment import SpatialEnvironment
from .memory import MemoryMode, SharedMemory
from .metrics import MetricsCollector, RunSummary, TickMetrics
from .mission import MissionGraph, Task, make_random_mission
from .policy import PolicyLevel, PolicyRegime, make_policy
from .topology import build_topology, graph_stats, nodes_within_hops, shortest_path_length


DEFAULT_TASK_TYPES = ("planning", "coding", "verification", "routing", "retrieval", "analysis")
DEFAULT_ROLES = ("coordinator", "planner", "worker", "worker", "worker", "verifier", "relay", "observer", "memory")


@dataclass
class SimulationConfig:
    rows: int = 20
    cols: int = 20
    n_nodes: int = 40
    n_tasks: int = 60
    max_steps: int = 100
    task_duration_min: int = 1
    task_duration_max: int = 4
    verification_duration_min: int = 1
    verification_duration_max: int = 2
    quality_threshold: float = 0.45
    topology: str = "decentralized_mesh"
    policy_level: str = "moderate"
    memory_mode: str = "local_and_shared"
    assignment_horizon_hops: int = 3
    memory_horizon_hops: int = 4
    direct_handoff_horizon_hops: int = 3
    role_specialization_alignment: float = 0.75
    node_failure_probability: float = 0.002
    message_loss_probability: float = 0.02
    base_latency: float = 1.0
    congestion_penalty: float = 0.12
    seed: int = 42
    adaptive_reassignment: bool = True
    adaptive_movement: bool = True
    run_id: str = "run"
    metadata: Dict[str, str | int | float] = field(default_factory=dict)


class HiveSimulation:
    """A compact ABM of Hive-like decentralized coordination."""

    def __init__(
        self,
        config: SimulationConfig,
        nodes: Optional[List[HiveNode]] = None,
        mission: Optional[MissionGraph] = None,
        environment: Optional[SpatialEnvironment] = None,
        policy: Optional[PolicyRegime] = None,
    ) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.environment = environment or SpatialEnvironment(config.rows, config.cols)
        self.nodes = nodes or self._make_nodes(config.n_nodes)
        self.mission = mission or make_random_mission(
            config.rows,
            config.cols,
            config.n_tasks,
            self.rng,
            task_types=DEFAULT_TASK_TYPES,
            timeout_steps=config.max_steps,
        )
        self.mission.quality_threshold = config.quality_threshold
        self.policy = policy or make_policy(config.policy_level)
        self.memory = SharedMemory(mode=MemoryMode(config.memory_mode))
        self.adjacency = build_topology(self.nodes, config.topology, self.rng)
        self.metrics = MetricsCollector()
        self.step_idx = 0

    def _make_nodes(self, n_nodes: int) -> List[HiveNode]:
        nodes: List[HiveNode] = []
        for node_id in range(n_nodes):
            role = DEFAULT_ROLES[node_id % len(DEFAULT_ROLES)]
            if node_id == 0:
                role = "coordinator"
            specialization = self._make_specialization(role)
            position = (self.rng.randrange(self.config.rows), self.rng.randrange(self.config.cols))
            permissions = set(self.policy_permissions_for_role_hint(role))
            resources = self._make_resources(role)
            nodes.append(
                HiveNode(
                    node_id=node_id,
                    role=role,
                    specialization=specialization,
                    position=position,
                    permissions=permissions,
                    resources=resources,
                    reliability=self.rng.uniform(0.94, 0.995),
                    trust=self.rng.uniform(0.75, 1.0),
                )
            )
        return nodes

    def policy_permissions_for_role_hint(self, role: str) -> set[str]:
        # Duplicated light role defaults to avoid importing policy globals into generation.
        base = {"wait", "move", "observe_sector"}
        if role in {"coordinator", "planner"}:
            permissions = base | {"solve_task", "read_memory", "write_memory", "delegate_task"}
            if role == "coordinator":
                permissions |= {"verify_task", "relay_message", "assign_task"}
            return permissions
        if role == "worker":
            return base | {"solve_task", "read_memory", "write_memory"}
        if role == "verifier":
            return base | {"verify_task", "read_memory", "write_memory"}
        if role == "relay":
            return base | {"relay_message"}
        if role == "memory":
            return base | {"read_memory", "write_memory", "relay_message"}
        return base | {"read_memory"}

    def _make_specialization(self, role: str) -> Dict[str, float]:
        alignment = max(0.0, min(1.0, self.config.role_specialization_alignment))
        spec = {task: self.rng.uniform(0.15, 0.55) for task in DEFAULT_TASK_TYPES}
        preferred = {
            "coordinator": ["planning", "analysis"],
            "planner": ["planning", "retrieval"],
            "worker": ["coding", "analysis"],
            "verifier": ["verification", "analysis"],
            "relay": ["routing"],
            "observer": ["analysis", "retrieval"],
            "memory": ["retrieval", "analysis"],
        }.get(role, [])
        for task in preferred:
            spec[task] = alignment * self.rng.uniform(0.70, 0.95) + (1.0 - alignment) * self.rng.uniform(0.15, 0.45)
        return spec

    def _make_resources(self, role: str) -> ResourceProfile:
        if role == "relay":
            return ResourceProfile(compute_capacity=0.6, bandwidth_capacity=2.0, relay_capacity=2.5, bandwidth_demand=0.35, communication_radius=5, sensing_radius=1)
        if role == "memory":
            return ResourceProfile(compute_capacity=0.8, bandwidth_capacity=1.2, storage_capacity=3.0, relay_capacity=0.8, bandwidth_demand=0.4, communication_radius=4)
        if role == "observer":
            return ResourceProfile(compute_capacity=0.7, bandwidth_capacity=0.8, sensing_radius=3, bandwidth_demand=0.25, communication_radius=3)
        if role == "verifier":
            return ResourceProfile(compute_capacity=1.2, bandwidth_capacity=0.8, compute_demand=0.35, bandwidth_demand=0.25, communication_radius=3)
        if role in {"coordinator", "planner"}:
            return ResourceProfile(compute_capacity=1.4, bandwidth_capacity=1.2, compute_demand=0.35, bandwidth_demand=0.35, communication_radius=4)
        return ResourceProfile(compute_capacity=1.0, bandwidth_capacity=0.8, compute_demand=0.30, bandwidth_demand=0.25, communication_radius=3)

    def run(self) -> RunSummary:
        for step in range(self.config.max_steps):
            self.step_idx = step
            self.step()
            if self.mission.is_success():
                break
        return self.summary()

    def step(self) -> None:
        # Rebuild adaptive topology before decisions when movement has changed positions.
        if self.config.topology == "adaptive":
            self.adjacency = build_topology(self.nodes, "adaptive", self.rng)

        unresolved = self.mission.unresolved_by_sector(self.environment.rows, self.environment.cols)
        self.environment.recompute_from_nodes(self.nodes, unresolved_tasks=unresolved)
        self._assign_available_tasks()

        denied = 0
        messages = 0
        duplicate_work = 0

        task_assignments_seen: Dict[int, int] = {}
        for node in self.nodes:
            node.messages_sent = 0
            if node.state != "failed" and self.rng.random() < self.config.node_failure_probability:
                node.state = "failed"
                if node.current_task_id is not None:
                    self.mission.release(node.current_task_id)
                    node.current_task_id = None
                continue

            task = self.mission.task_for_node(node.node_id)
            action = node.propose_action(
                has_assigned_task=task is not None,
                task_type=task.task_type if task else None,
                task_phase=task.active_phase if task else None,
            )
            allowed = self.policy.allows(node, action, task)

            if not allowed:
                denied += 1
                node.blocked_steps += 1
                node.current_load += self.policy.denial_cost
                node.state = "blocked"
                if node.current_task_id is not None:
                    self.mission.release(node.current_task_id)
                    node.current_task_id = None
                continue

            if action.kind in {SOLVE, VERIFY} and task is not None:
                if task.task_id in task_assignments_seen:
                    duplicate_work += 1
                task_assignments_seen[task.task_id] = node.node_id
                messages += self._message_cost(node, action, task)
                phase = task.active_phase
                finished_work = self.mission.progress_assigned_task(task.task_id)
                node.current_load += 0.04 + 0.04 * task.difficulty
                node.state = "working"
                if not finished_work:
                    continue

                verify = phase == "verify" or action.kind == VERIFY
                self.mission.record_attempt(task.task_id, phase)
                ok, quality = self._attempt_task(node, task, verify=verify)
                if ok:
                    cid = task.artifact_id or self._artifact_id(task)
                    if self._requires_downstream_verification(action, task, phase):
                        self.mission.mark_awaiting_verification(
                            task.task_id,
                            quality=quality,
                            step=self.step_idx,
                            solved_by=node.node_id,
                            artifact_id=cid,
                        )
                        self._publish_artifact(node, task, cid, quality, verified=False)
                    else:
                        self.mission.complete(task.task_id, quality=quality, step=self.step_idx, verified=verify)
                        node.completed_tasks += 1
                        self._publish_artifact(node, task, cid, quality, verified=verify)
                    node.release_task()
                else:
                    node.failed_attempts += 1
                    self.mission.fail_attempt(task.task_id, phase=phase, release=True, count_attempt=False)
                    node.release_task()
            elif action.kind == RELAY:
                node.current_load = max(0.0, node.current_load * 0.85)
                messages += self._message_cost(node, action)
                node.state = "relaying"
            elif action.kind == OBSERVE:
                node.current_load = max(0.0, node.current_load * 0.90)
                messages += self._message_cost(node, action)
                node.state = "observing"
            else:
                node.current_load = max(0.0, node.current_load * 0.92)
                if self.config.adaptive_movement and node.state in {"idle", "blocked"}:
                    self._move_idle_node(node)
                if node.state != "failed":
                    node.state = "idle"

        self.environment.recompute_from_nodes(self.nodes, unresolved_tasks=self.mission.unresolved_by_sector(self.environment.rows, self.environment.cols))
        self._log_tick(denied=denied, messages=messages, duplicate_work=duplicate_work)

    def _assign_available_tasks(self) -> None:
        available = [(task, "verify") for task in self.mission.available_verification_tasks()]
        available.extend((task, "solve") for task in self.mission.available_tasks())
        if not available:
            return
        idle_nodes = [node for node in self.nodes if node.state in {"idle", "blocked", "observing", "relaying"} and node.current_task_id is None]
        self.rng.shuffle(idle_nodes)

        # Greedy assignment by policy feasibility, topology reachability, and fit.
        for task, phase in list(available):
            candidates: List[Tuple[HiveNode, int]] = []
            for node in idle_nodes:
                if phase == "verify" and node.node_id == task.solved_by:
                    continue
                action_kind = self._action_kind_for_phase(task, phase)
                action = Action(action_kind, node.node_id, task.task_id)
                if not self.policy.allows(node, action, task):
                    continue
                if phase == "verify" and not self._can_access_task_artifact(node, task):
                    continue
                route_hops = self._assignment_route_hops(node, task, phase)
                if route_hops is None:
                    continue
                candidates.append((node, route_hops))
            if not candidates:
                continue
            best, route_hops = max(candidates, key=lambda item: self._assignment_score(item[0], task, phase, item[1]))
            self.mission.assign(
                task.task_id,
                best.node_id,
                phase=phase,
                work_required=self._work_required(task, phase),
                step=self.step_idx,
                assignment_hops=route_hops,
            )
            best.assign_task(task.task_id)
            idle_nodes.remove(best)
            if not idle_nodes:
                break

    def _action_kind_for_phase(self, task: Task, phase: str) -> str:
        if phase == "verify" or task.task_type == "verification":
            return VERIFY
        return SOLVE

    def _assignment_score(self, node: HiveNode, task: Task, phase: str = "solve", route_hops: int = 0) -> float:
        dist = abs(node.position[0] - task.sector[0]) + abs(node.position[1] - task.sector[1])
        dist_score = 1.0 / (1.0 + dist)
        route_score = 1.0 / (1.0 + route_hops)
        competence = node.competence("verification") if phase == "verify" else node.competence(task.task_type)
        role_bonus = 0.20 if phase == "verify" and node.role in {"verifier", "coordinator"} else 0.0
        if task.required_role and task.required_role == node.role:
            role_bonus += 0.15
        return 0.60 * competence + 0.20 * dist_score + 0.10 * route_score + role_bonus - 0.05 * node.current_load

    def _work_required(self, task: Task, phase: str) -> int:
        if phase == "verify":
            low = max(1, self.config.verification_duration_min)
            high = max(low, self.config.verification_duration_max)
        else:
            low = max(1, self.config.task_duration_min)
            high = max(low, self.config.task_duration_max)
        return max(low, min(high, low + round(task.difficulty * (high - low))))

    def _assignment_route_hops(self, node: HiveNode, task: Task, phase: str) -> Optional[int]:
        if node.state == "failed":
            return None
        if self._node_can_self_assign(node, task, phase):
            return 0

        best: Optional[int] = None
        for dispatcher in self._assignment_dispatchers():
            if dispatcher.node_id == node.node_id:
                hops = 0
            else:
                hops = shortest_path_length(
                    self.adjacency,
                    dispatcher.node_id,
                    node.node_id,
                    max_hops=self.config.assignment_horizon_hops,
                )
            if hops is None:
                continue
            best = hops if best is None else min(best, hops)
        return best

    def _assignment_dispatchers(self) -> List[HiveNode]:
        return [
            node
            for node in self.nodes
            if node.state != "failed"
            and (
                node.role in {"coordinator", "planner"}
                or "assign_task" in node.permissions
                or "delegate_task" in node.permissions
            )
        ]

    def _node_can_self_assign(self, node: HiveNode, task: Task, phase: str) -> bool:
        if phase == "verify":
            return self._can_access_task_artifact(node, task)
        return self._node_can_observe_task(node, task)

    def _node_can_observe_task(self, node: HiveNode, task: Task) -> bool:
        dist = abs(node.position[0] - task.sector[0]) + abs(node.position[1] - task.sector[1])
        return dist <= max(1, node.resources.sensing_radius)

    def _attempt_task(self, node: HiveNode, task: Task, verify: bool = False) -> Tuple[bool, float]:
        obs = self.environment.observe(node.position)
        competence = node.competence(task.task_type)
        if verify:
            competence = max(competence, node.competence("verification"))
        congestion = obs["congestion"]
        coverage = obs["sensor_coverage"]
        memory_bonus = self._memory_bonus(node, task, verify=verify)
        route_penalty = 0.03 * task.assignment_hops
        policy_friction = {
            PolicyLevel.PERMISSIVE: 0.00,
            PolicyLevel.MODERATE: 0.03,
            PolicyLevel.STRICT: 0.08,
            PolicyLevel.MISSION_CRITICAL: 0.12,
        }[self.policy.level]
        p_success = (
            0.16
            + 0.82 * competence
            + 0.08 * min(1.0, coverage / 3.0)
            + memory_bonus
            - 0.38 * task.difficulty
            - self.config.congestion_penalty * congestion
            - policy_friction
            - route_penalty
        )
        p_success *= node.reliability
        p_success = max(0.02, min(0.98, p_success))
        route_loss = 1.0 - (1.0 - self.config.message_loss_probability) ** max(1, task.assignment_hops + 1)
        ok = self.rng.random() >= route_loss and self.rng.random() < p_success
        quality = max(0.0, min(1.0, 0.08 + competence * (1.0 - 0.20 * task.difficulty) + self.rng.uniform(-0.06, 0.14)))
        if verify:
            quality = min(1.0, quality + 0.10)
        node.current_load += 0.10 + 0.20 * task.difficulty
        return ok and quality >= self.mission.quality_threshold, quality

    def _memory_bonus(self, node: HiveNode, task: Task, verify: bool = False) -> float:
        bonus = 0.0
        if self.memory.can_use_local() and node.local_memory:
            bonus += 0.03
        if verify and self._can_access_task_artifact(node, task):
            bonus += 0.06
        elif self._has_visible_shared_memory(node):
            bonus += 0.05
        return min(0.08, bonus)

    def _has_visible_shared_memory(self, node: HiveNode) -> bool:
        if not self.memory.can_read_shared() or self.memory.size() == 0:
            return False
        reachable = nodes_within_hops(self.adjacency, node.node_id, self.config.memory_horizon_hops)
        for artifact in self.memory.artifacts.values():
            if artifact.writer_id is None or artifact.writer_id in reachable:
                return True
        return False

    def _can_access_task_artifact(self, node: HiveNode, task: Task) -> bool:
        if task.artifact_id is None:
            return False
        if task.artifact_id in node.local_memory:
            return True
        if task.solved_by is not None:
            direct_hops = shortest_path_length(
                self.adjacency,
                node.node_id,
                task.solved_by,
                max_hops=self.config.direct_handoff_horizon_hops,
            )
            if direct_hops is not None:
                return True
        if not self.memory.can_read_shared():
            return False
        artifact = self.memory.artifact(task.artifact_id)
        if artifact is None:
            return False
        if artifact.writer_id is None:
            return True
        reachable = nodes_within_hops(self.adjacency, node.node_id, self.config.memory_horizon_hops)
        return artifact.writer_id in reachable

    def _requires_downstream_verification(self, action: Action, task: Task, phase: str) -> bool:
        return self.policy.require_verifier_for_completion and phase != "verify" and action.kind != VERIFY

    def _artifact_id(self, task: Task) -> str:
        return task.artifact_id or f"task-{task.task_id}-artifact"

    def _publish_artifact(self, node: HiveNode, task: Task, cid: str, quality: float, verified: bool) -> None:
        task.artifact_id = cid
        label = "verified" if verified else "solved"
        if self.memory.can_write_shared():
            self.memory.write(
                cid,
                f"{label}-task-{task.task_id}-by-{node.node_id}-q{quality:.3f}",
                writer_id=node.node_id,
                sector=task.sector,
                created_step=self.step_idx,
            )
        if self.memory.can_use_local():
            node.local_memory.add(cid)

    def _message_cost(self, node: HiveNode, action: Action, task: Optional[Task] = None) -> int:
        degree = len(self.adjacency.get(node.node_id, []))
        route_hops = task.assignment_hops if task is not None else 0
        base = 1 if action.kind in {SOLVE, VERIFY, OBSERVE} else 2 if action.kind == RELAY else 0
        cost = base + route_hops + max(0, degree // 4)
        node.messages_sent += cost
        return cost

    def _move_idle_node(self, node: HiveNode) -> None:
        # Move toward nearest available task, otherwise toward highest mission pressure.
        target = self.mission.nearest_available_task_sector(node.position)
        if target is None:
            _, r, c = self.environment.layer_max("mission_pressure")
            target = (r, c) if r >= 0 else node.position
        node.move_towards(target, self.environment.rows, self.environment.cols)

    def _log_tick(self, denied: int, messages: int, duplicate_work: int) -> None:
        peak_congestion, pr, pc = self.environment.layer_max("congestion")
        corridor_len, _, _, corridor_sum = self.environment.safest_congestion_corridor_by_row(threshold=self.environment.rows * 0.5)
        stats = graph_stats(self.adjacency)
        task_counts = self.mission.status_counts()
        active_hops = [task.assignment_hops for task in self.mission.tasks.values() if task.status == "in_progress"]
        tick = TickMetrics(
            step=self.step_idx,
            completion_rate=self.mission.completion_rate(),
            verified_completion_rate=self.mission.verified_completion_rate(),
            completed_value=self.mission.total_value_completed(),
            mean_quality=self.mission.mean_quality(),
            mean_congestion=self.environment.layer_mean("congestion"),
            peak_congestion=peak_congestion,
            peak_congestion_row=pr,
            peak_congestion_col=pc,
            mean_coverage=self.environment.layer_mean("sensor_coverage"),
            mean_bandwidth_capacity=self.environment.layer_mean("bandwidth_capacity"),
            mean_bandwidth_demand=self.environment.layer_mean("bandwidth_demand"),
            denied_actions=denied,
            messages_sent=messages,
            duplicate_work=duplicate_work,
            blocked_nodes=sum(1 for n in self.nodes if n.state == "blocked"),
            failed_nodes=sum(1 for n in self.nodes if n.state == "failed"),
            idle_nodes=sum(1 for n in self.nodes if n.state == "idle"),
            working_nodes=sum(1 for n in self.nodes if n.state == "working"),
            pending_tasks=task_counts.get("pending", 0),
            in_progress_tasks=task_counts.get("in_progress", 0),
            awaiting_verification_tasks=task_counts.get("awaiting_verification", 0),
            completed_tasks=task_counts.get("completed", 0),
            verified_tasks=sum(1 for task in self.mission.tasks.values() if task.status == "completed" and task.verified),
            mean_assignment_hops=(sum(active_hops) / len(active_hops)) if active_hops else 0.0,
            safe_corridor_length=corridor_len,
            safe_corridor_sum=corridor_sum,
            graph_centralization=stats["centralization"],
            mean_degree=stats["mean_degree"],
        )
        self.metrics.log_tick(tick)

    def summary(self) -> RunSummary:
        last = self.metrics.ticks[-1] if self.metrics.ticks else None
        steps = (last.step + 1) if last else 0
        stats = graph_stats(self.adjacency)
        task_counts = self.mission.status_counts()
        mean_net = sum(node.net_contribution_proxy() for node in self.nodes) / len(self.nodes) if self.nodes else 0.0
        meta = dict(self.config.metadata)
        meta.update(
            {
                "topology": self.config.topology,
                "policy_level": self.config.policy_level,
                "memory_mode": self.config.memory_mode,
                "n_nodes": self.config.n_nodes,
                "n_tasks": self.config.n_tasks,
                "seed": self.config.seed,
                "quality_threshold": self.config.quality_threshold,
                "role_specialization_alignment": self.config.role_specialization_alignment,
                "task_duration_min": self.config.task_duration_min,
                "task_duration_max": self.config.task_duration_max,
                "verification_duration_min": self.config.verification_duration_min,
                "verification_duration_max": self.config.verification_duration_max,
                "assignment_horizon_hops": self.config.assignment_horizon_hops,
                "memory_horizon_hops": self.config.memory_horizon_hops,
                "direct_handoff_horizon_hops": self.config.direct_handoff_horizon_hops,
            }
        )
        return RunSummary(
            run_id=self.config.run_id,
            success=self.mission.is_success(),
            steps=steps,
            final_completion_rate=self.mission.completion_rate(),
            final_verified_completion_rate=self.mission.verified_completion_rate(),
            final_quality=self.mission.mean_quality(),
            final_awaiting_verification=task_counts.get("awaiting_verification", 0),
            total_attempts=self.mission.total_attempts(),
            total_verification_attempts=self.mission.total_verification_attempts(),
            total_denied_actions=self.metrics.denied_actions_total,
            total_messages=self.metrics.messages_total,
            total_duplicate_work=self.metrics.duplicate_work_total,
            final_mean_congestion=last.mean_congestion if last else 0.0,
            final_peak_congestion=last.peak_congestion if last else 0.0,
            final_mean_coverage=last.mean_coverage if last else 0.0,
            safe_corridor_length=last.safe_corridor_length if last else 0,
            graph_centralization=stats["centralization"],
            mean_net_contribution=mean_net,
            metadata=meta,
        )
