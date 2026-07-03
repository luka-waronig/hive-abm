"""Agent abstraction for the Hive ABM.

There is intentionally only one agent class: ``HiveNode``. Relay, verifier,
planner, worker, etc. are roles, not subclasses. This keeps the model simple
and lets role assignment be an experimental variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

from .actions import Action, MOVE, OBSERVE, RELAY, SOLVE, VERIFY, WAIT


@dataclass
class ResourceProfile:
    """Infrastructure contribution and load profile of a Hive node."""

    compute_capacity: float = 1.0
    bandwidth_capacity: float = 1.0
    storage_capacity: float = 1.0
    relay_capacity: float = 0.0
    sensing_radius: int = 1
    communication_radius: int = 3
    compute_demand: float = 0.2
    bandwidth_demand: float = 0.2
    energy: float = 100.0


@dataclass
class HiveNode:
    """Single node abstraction for cognitive + infrastructure-bearing agents."""

    node_id: int
    role: str
    specialization: Dict[str, float]
    position: Tuple[int, int]
    permissions: Set[str]
    resources: ResourceProfile = field(default_factory=ResourceProfile)
    trust: float = 1.0
    reliability: float = 0.98
    state: str = "idle"
    current_task_id: Optional[int] = None
    current_load: float = 0.0
    blocked_steps: int = 0
    completed_tasks: int = 0
    failed_attempts: int = 0
    messages_sent: int = 0
    local_memory: Set[str] = field(default_factory=set)

    def competence(self, task_type: str) -> float:
        """Return competence for a task type, defaulting to a weak baseline."""

        return max(0.0, min(1.0, self.specialization.get(task_type, 0.25)))

    def assign_task(self, task_id: int) -> None:
        self.current_task_id = task_id
        self.state = "working"

    def release_task(self) -> None:
        self.current_task_id = None
        if self.state != "failed":
            self.state = "idle"

    def move_towards(self, target: Tuple[int, int], rows: int, cols: int) -> Tuple[int, int]:
        """Move one grid step toward target using Manhattan movement."""

        r, c = self.position
        tr, tc = target
        if tr > r:
            r += 1
        elif tr < r:
            r -= 1
        elif tc > c:
            c += 1
        elif tc < c:
            c -= 1
        r = max(0, min(rows - 1, r))
        c = max(0, min(cols - 1, c))
        self.position = (r, c)
        return self.position

    def propose_action(
        self,
        has_assigned_task: bool,
        task_type: Optional[str] = None,
        task_phase: Optional[str] = None,
    ) -> Action:
        """Simple default policy-independent action proposal.

        The simulation supplies task assignment. If assigned, the node proposes
        solving or verifying depending on role/task type. Otherwise, relay nodes
        relay, observer nodes observe, and everyone else waits/moves elsewhere
        through the simulation's movement heuristic.
        """

        if self.state == "failed":
            return Action(WAIT, self.node_id)
        if has_assigned_task and self.current_task_id is not None:
            if task_phase == "verify" or task_type == "verification":
                return Action(VERIFY, self.node_id, self.current_task_id)
            return Action(SOLVE, self.node_id, self.current_task_id)
        if self.role == "relay":
            return Action(RELAY, self.node_id)
        if self.role == "observer":
            return Action(OBSERVE, self.node_id)
        return Action(WAIT, self.node_id)

    def infrastructure_value(self) -> float:
        """Positive resource-bearing contribution, used for summary metrics."""

        return (
            self.resources.compute_capacity
            + self.resources.bandwidth_capacity
            + self.resources.storage_capacity
            + self.resources.relay_capacity
        )

    def imposed_load(self) -> float:
        """Demand/load contribution, used for net-contribution metrics."""

        return self.resources.compute_demand + self.resources.bandwidth_demand + self.current_load

    def net_contribution_proxy(self) -> float:
        """Proxy for useful capacity minus imposed system load."""

        return self.infrastructure_value() + self.completed_tasks - self.imposed_load() - self.failed_attempts
