"""Communication topology generation for Hive node networks."""

from __future__ import annotations

import math
import random
from collections import deque
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

from .agents import HiveNode

Adjacency = Dict[int, Set[int]]


def empty_adjacency(nodes: Iterable[HiveNode]) -> Adjacency:
    return {node.node_id: set() for node in nodes}


def add_edge(adj: Adjacency, a: int, b: int) -> None:
    if a == b:
        return
    adj.setdefault(a, set()).add(b)
    adj.setdefault(b, set()).add(a)


def build_topology(nodes: List[HiveNode], kind: str, rng: random.Random, k: int = 3) -> Adjacency:
    """Build an undirected communication graph for nodes."""

    if not nodes:
        return {}
    kind = kind.lower()
    if kind == "centralized":
        return centralized(nodes)
    if kind == "hierarchical":
        return hierarchical(nodes, rng)
    if kind in {"mesh", "decentralized", "decentralized_mesh"}:
        return spatial_mesh(nodes)
    if kind == "small_world":
        return small_world(nodes, rng, k=max(2, k))
    if kind == "scale_free":
        return scale_free(nodes, rng, m=max(1, k // 2))
    if kind == "adaptive":
        # Starts as spatial mesh; simulation can rebuild as positions/load change.
        return spatial_mesh(nodes)
    raise ValueError(f"unknown topology: {kind}")


def centralized(nodes: List[HiveNode]) -> Adjacency:
    adj = empty_adjacency(nodes)
    coordinator = min(nodes, key=lambda n: (n.role != "coordinator", n.node_id)).node_id
    for node in nodes:
        add_edge(adj, coordinator, node.node_id)
    return adj


def hierarchical(nodes: List[HiveNode], rng: random.Random) -> Adjacency:
    adj = empty_adjacency(nodes)
    coordinator = min(nodes, key=lambda n: (n.role != "coordinator", n.node_id)).node_id
    sector_leads = [n.node_id for n in nodes if n.role in {"planner", "relay"} and n.node_id != coordinator]
    if not sector_leads:
        sector_leads = [n.node_id for n in nodes if n.node_id != coordinator][: max(1, int(math.sqrt(len(nodes))))]
    for lead in sector_leads:
        add_edge(adj, coordinator, lead)
    for node in nodes:
        if node.node_id == coordinator or node.node_id in sector_leads:
            continue
        lead = min(sector_leads, key=lambda lid: abs(_node_by_id(nodes, lid).position[0] - node.position[0]) + abs(_node_by_id(nodes, lid).position[1] - node.position[1]))
        add_edge(adj, lead, node.node_id)
    return adj


def spatial_mesh(nodes: List[HiveNode]) -> Adjacency:
    adj = empty_adjacency(nodes)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            dist = abs(a.position[0] - b.position[0]) + abs(a.position[1] - b.position[1])
            radius = max(a.resources.communication_radius, b.resources.communication_radius)
            if dist <= radius:
                add_edge(adj, a.node_id, b.node_id)
    # Avoid isolated nodes by linking them to nearest neighbor.
    for node in nodes:
        if not adj[node.node_id] and len(nodes) > 1:
            nearest = min((other for other in nodes if other.node_id != node.node_id), key=lambda other: abs(other.position[0] - node.position[0]) + abs(other.position[1] - node.position[1]))
            add_edge(adj, node.node_id, nearest.node_id)
    return adj


def small_world(nodes: List[HiveNode], rng: random.Random, k: int = 3, shortcut_prob: float = 0.15) -> Adjacency:
    adj = empty_adjacency(nodes)
    ids = [node.node_id for node in nodes]
    n = len(ids)
    if n <= 1:
        return adj
    k = min(k, n - 1)
    for i, node_id in enumerate(ids):
        for offset in range(1, k // 2 + 1):
            add_edge(adj, node_id, ids[(i + offset) % n])
            add_edge(adj, node_id, ids[(i - offset) % n])
    for i, node_id in enumerate(ids):
        for j, other_id in enumerate(ids):
            if i < j and rng.random() < shortcut_prob:
                add_edge(adj, node_id, other_id)
    return adj


def scale_free(nodes: List[HiveNode], rng: random.Random, m: int = 1) -> Adjacency:
    adj = empty_adjacency(nodes)
    ids = [node.node_id for node in nodes]
    if len(ids) <= 1:
        return adj
    add_edge(adj, ids[0], ids[1])
    repeated = [ids[0], ids[1]]
    for new_id in ids[2:]:
        targets: Set[int] = set()
        while len(targets) < min(m, len(set(repeated))):
            targets.add(rng.choice(repeated))
        for target in targets:
            add_edge(adj, new_id, target)
        repeated.extend([new_id] * max(1, len(targets)))
        repeated.extend(list(targets))
    return adj


def graph_stats(adj: Adjacency) -> Dict[str, float]:
    n = len(adj)
    if n == 0:
        return {"nodes": 0.0, "edges": 0.0, "mean_degree": 0.0, "max_degree": 0.0, "centralization": 0.0}
    degrees = [len(neigh) for neigh in adj.values()]
    edges = sum(degrees) / 2
    max_degree = max(degrees) if degrees else 0
    mean_degree = sum(degrees) / n
    if n <= 2:
        centralization = 0.0
    else:
        centralization = sum(max_degree - d for d in degrees) / ((n - 1) * (n - 2))
    return {
        "nodes": float(n),
        "edges": float(edges),
        "mean_degree": float(mean_degree),
        "max_degree": float(max_degree),
        "centralization": float(centralization),
        "components": float(component_count(adj)),
    }


def component_count(adj: Adjacency) -> int:
    """Return the number of connected components in an undirected graph."""

    unseen = set(adj)
    count = 0
    while unseen:
        count += 1
        start = unseen.pop()
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adj.get(current, set()):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    return count


def shortest_path_lengths(adj: Adjacency, start: int, max_hops: int | None = None) -> Dict[int, int]:
    """Return hop distances from ``start`` through the communication graph."""

    if start not in adj:
        return {}
    distances = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        current_dist = distances[current]
        if max_hops is not None and current_dist >= max_hops:
            continue
        for neighbor in adj.get(current, set()):
            if neighbor in distances:
                continue
            distances[neighbor] = current_dist + 1
            queue.append(neighbor)
    return distances


def shortest_path_length(adj: Adjacency, start: int, end: int, max_hops: int | None = None) -> int | None:
    """Return the shortest hop count between two nodes, or ``None`` if unreachable."""

    return shortest_path_lengths(adj, start, max_hops=max_hops).get(end)


def nodes_within_hops(adj: Adjacency, start: int, max_hops: int) -> Set[int]:
    """Return nodes reachable from ``start`` within ``max_hops`` graph hops."""

    return set(shortest_path_lengths(adj, start, max_hops=max_hops))


def _node_by_id(nodes: List[HiveNode], node_id: int) -> HiveNode:
    for node in nodes:
        if node.node_id == node_id:
            return node
    raise KeyError(node_id)
