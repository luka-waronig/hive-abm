"""Hive ABM: a compact simulation framework for decentralized AI-agent coordination.

The model represents Hive nodes as cognitive + infrastructure-bearing agents embedded
in spatial sectors. Nodes contribute compute/bandwidth/coverage while also imposing
message load and congestion. The simulator is intentionally lightweight and pure Python
so it can be used directly in thesis experiments.
"""

from .agents import HiveNode, ResourceProfile
from .environment import SpatialEnvironment
from .mission import MissionGraph, Task
from .policy import PolicyLevel, PolicyRegime, make_policy
from .simulation import HiveSimulation, SimulationConfig

__all__ = [
    "HiveNode",
    "ResourceProfile",
    "SpatialEnvironment",
    "MissionGraph",
    "Task",
    "PolicyLevel",
    "PolicyRegime",
    "make_policy",
    "HiveSimulation",
    "SimulationConfig",
]
