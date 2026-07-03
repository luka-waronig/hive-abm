"""Action representation for Hive ABM nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Action:
    """A node-level action proposed before policy filtering."""

    kind: str
    node_id: int
    task_id: Optional[int] = None
    target_position: Optional[Tuple[int, int]] = None
    metadata: Dict[str, float | int | str] = field(default_factory=dict)


WAIT = "wait"
MOVE = "move"
SOLVE = "solve"
VERIFY = "verify"
RELAY = "relay"
QUERY_MEMORY = "query_memory"
WRITE_MEMORY = "write_memory"
DELEGATE = "delegate"
OBSERVE = "observe"
