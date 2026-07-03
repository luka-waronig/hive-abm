"""Policy-confinement regimes for the Hive ABM."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set

from .actions import DELEGATE, MOVE, OBSERVE, QUERY_MEMORY, RELAY, SOLVE, VERIFY, WAIT, WRITE_MEMORY, Action
from .agents import HiveNode
from .mission import Task


class PolicyLevel(str, Enum):
    PERMISSIVE = "permissive"
    MODERATE = "moderate"
    STRICT = "strict"
    MISSION_CRITICAL = "mission_critical"


ACTION_PERMISSION = {
    WAIT: "wait",
    MOVE: "move",
    SOLVE: "solve_task",
    VERIFY: "verify_task",
    RELAY: "relay_message",
    QUERY_MEMORY: "read_memory",
    WRITE_MEMORY: "write_memory",
    DELEGATE: "delegate_task",
    OBSERVE: "observe_sector",
}

DEFAULT_ROLE_PERMISSIONS: Dict[str, Set[str]] = {
    "coordinator": {
        "wait",
        "move",
        "solve_task",
        "verify_task",
        "relay_message",
        "read_memory",
        "write_memory",
        "delegate_task",
        "observe_sector",
        "assign_task",
    },
    "planner": {"wait", "move", "solve_task", "read_memory", "write_memory", "delegate_task", "observe_sector"},
    "worker": {"wait", "move", "solve_task", "read_memory", "write_memory", "observe_sector"},
    "verifier": {"wait", "move", "verify_task", "read_memory", "write_memory", "observe_sector"},
    "relay": {"wait", "move", "relay_message", "observe_sector"},
    "observer": {"wait", "move", "observe_sector", "read_memory"},
    "memory": {"wait", "move", "read_memory", "write_memory", "relay_message"},
}


@dataclass
class PolicyRegime:
    """A simple policy layer: authorization before capability/performance."""

    level: PolicyLevel
    role_permissions: Dict[str, Set[str]] = field(default_factory=lambda: {k: set(v) for k, v in DEFAULT_ROLE_PERMISSIONS.items()})
    require_role_task_match: bool = False
    require_explicit_permission: bool = False
    require_verifier_for_completion: bool = False
    min_competence_for_sensitive: float = 0.0
    denial_cost: float = 0.15

    def permissions_for_role(self, role: str) -> Set[str]:
        return self.role_permissions.get(role, {"wait", "move", "observe_sector"})

    def allows(self, node: HiveNode, action: Action, task: Task | None = None) -> bool:
        """Return whether a proposed action is authorized.

        The simulator still computes success probabilistically after authorization.
        """

        if action.kind == WAIT:
            return True
        permission = ACTION_PERMISSION.get(action.kind)
        if permission is None:
            return False

        role_allowed = permission in self.permissions_for_role(node.role)
        explicit_allowed = permission in node.permissions

        if self.level == PolicyLevel.PERMISSIVE:
            return role_allowed or explicit_allowed

        if self.require_explicit_permission and not explicit_allowed:
            return False
        if not role_allowed:
            return False

        if self.require_role_task_match and task is not None and action.kind in {SOLVE, VERIFY}:
            if action.kind == VERIFY and node.role != "verifier" and node.role != "coordinator":
                return False
            if action.kind == SOLVE and node.role == "verifier":
                return False

        if self.min_competence_for_sensitive > 0.0 and task is not None and action.kind in {SOLVE, VERIFY}:
            competence = node.competence("verification") if action.kind == VERIFY else node.competence(task.task_type)
            if competence < self.min_competence_for_sensitive:
                return False

        return True


def make_policy(level: str | PolicyLevel) -> PolicyRegime:
    """Factory for predefined policy-confinement levels."""

    level = PolicyLevel(level)
    if level == PolicyLevel.PERMISSIVE:
        return PolicyRegime(level=level)
    if level == PolicyLevel.MODERATE:
        return PolicyRegime(level=level, require_role_task_match=False, require_explicit_permission=False)
    if level == PolicyLevel.STRICT:
        return PolicyRegime(
            level=level,
            require_role_task_match=True,
            require_explicit_permission=True,
            min_competence_for_sensitive=0.35,
            denial_cost=0.25,
        )
    if level == PolicyLevel.MISSION_CRITICAL:
        return PolicyRegime(
            level=level,
            require_role_task_match=True,
            require_explicit_permission=True,
            require_verifier_for_completion=True,
            min_competence_for_sensitive=0.50,
            denial_cost=0.35,
        )
    raise ValueError(f"unknown policy level: {level}")
