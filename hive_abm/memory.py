"""Memory architecture abstractions for Hive ABM experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple


class MemoryMode(str, Enum):
    NONE = "none"
    LOCAL = "local"
    SHARED = "shared"
    LOCAL_AND_SHARED = "local_and_shared"


@dataclass
class MemoryArtifact:
    """A shared artifact plus provenance needed for topology-aware access."""

    cid: str
    value: str
    writer_id: Optional[int] = None
    sector: Optional[Tuple[int, int]] = None
    created_step: Optional[int] = None


@dataclass
class SharedMemory:
    """Minimal content-addressed shared memory proxy.

    This is not IPFS; it is a simulation stand-in. Artifacts are keyed by a
    synthetic content ID. Access cost can be interpreted as retrieval latency,
    bandwidth use, or shared-index contention.
    """

    mode: MemoryMode = MemoryMode.LOCAL_AND_SHARED
    artifacts: Dict[str, MemoryArtifact] = field(default_factory=dict)
    reads: int = 0
    writes: int = 0
    access_cost: float = 0.05

    def can_read_shared(self) -> bool:
        return self.mode in {MemoryMode.SHARED, MemoryMode.LOCAL_AND_SHARED}

    def can_write_shared(self) -> bool:
        return self.mode in {MemoryMode.SHARED, MemoryMode.LOCAL_AND_SHARED}

    def can_use_local(self) -> bool:
        return self.mode in {MemoryMode.LOCAL, MemoryMode.LOCAL_AND_SHARED}

    def write(
        self,
        cid: str,
        value: str,
        writer_id: Optional[int] = None,
        sector: Optional[Tuple[int, int]] = None,
        created_step: Optional[int] = None,
    ) -> None:
        if not self.can_write_shared():
            return
        self.artifacts[cid] = MemoryArtifact(
            cid=cid,
            value=value,
            writer_id=writer_id,
            sector=sector,
            created_step=created_step,
        )
        self.writes += 1

    def read(self, cid: str) -> Optional[str]:
        if not self.can_read_shared():
            return None
        self.reads += 1
        artifact = self.artifacts.get(cid)
        return artifact.value if artifact is not None else None

    def artifact(self, cid: str) -> Optional[MemoryArtifact]:
        return self.artifacts.get(cid)

    def size(self) -> int:
        return len(self.artifacts)
