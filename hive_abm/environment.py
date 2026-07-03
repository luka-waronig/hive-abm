"""Spatial sector environment for the Hive ABM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from .agents import HiveNode
from .fields import RectCommand2D, apply_rect_add_2d, argmax_first_2d, argmin_first_2d, prefix_sum_2d, region_sum, zeros_2d
from .windows import max_window_under_threshold, min_fixed_window

Grid = List[List[float]]


@dataclass
class SpatialEnvironment:
    """A 2D sector map with resource/cost layers.

    The environment is recomputed from node footprints each tick. This keeps the
    model transparent: node actions and placements generate capacity, demand,
    coverage, and congestion fields.
    """

    rows: int
    cols: int
    base_bandwidth_capacity: float = 1.0
    base_compute_capacity: float = 1.0
    base_noise: float = 0.05
    layers: Dict[str, Grid] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive")
        self.reset_layers()

    def reset_layers(self) -> None:
        self.layers = {
            "bandwidth_capacity": zeros_2d(self.rows, self.cols, self.base_bandwidth_capacity),
            "bandwidth_demand": zeros_2d(self.rows, self.cols, 0.0),
            "compute_capacity": zeros_2d(self.rows, self.cols, self.base_compute_capacity),
            "compute_demand": zeros_2d(self.rows, self.cols, 0.0),
            "sensor_coverage": zeros_2d(self.rows, self.cols, 0.0),
            "noise": zeros_2d(self.rows, self.cols, self.base_noise),
            "congestion": zeros_2d(self.rows, self.cols, 0.0),
            "mission_pressure": zeros_2d(self.rows, self.cols, 0.0),
        }

    def clip_rect(self, center: Tuple[int, int], radius: int) -> Tuple[int, int, int, int]:
        r, c = center
        return (
            max(0, r - radius),
            max(0, c - radius),
            min(self.rows - 1, r + radius),
            min(self.cols - 1, c + radius),
        )

    def recompute_from_nodes(self, nodes: Iterable[HiveNode], unresolved_tasks: Grid | None = None) -> None:
        """Recompute resource layers based on current node positions/profiles."""

        bw_capacity_cmds: List[RectCommand2D] = []
        bw_demand_cmds: List[RectCommand2D] = []
        compute_capacity_cmds: List[RectCommand2D] = []
        compute_demand_cmds: List[RectCommand2D] = []
        coverage_cmds: List[RectCommand2D] = []
        noise_cmds: List[RectCommand2D] = []

        for node in nodes:
            if node.state == "failed":
                continue
            r, c = node.position
            sensing_rect = self.clip_rect((r, c), max(0, node.resources.sensing_radius))
            comm_rect = self.clip_rect((r, c), max(0, node.resources.communication_radius))

            # Capacity contributions can extend through the communication radius.
            bw_capacity_cmds.append(RectCommand2D(*comm_rect, node.resources.bandwidth_capacity + node.resources.relay_capacity))
            compute_capacity_cmds.append(RectCommand2D(r, c, r, c, node.resources.compute_capacity))
            coverage_cmds.append(RectCommand2D(*sensing_rect, 1.0))

            # Demand/noise are strongest near the node; message-heavy nodes burden nearby sectors.
            local_demand = node.resources.bandwidth_demand + 0.05 * node.current_load + 0.02 * node.messages_sent
            bw_demand_cmds.append(RectCommand2D(r, c, r, c, local_demand))
            compute_demand_cmds.append(RectCommand2D(r, c, r, c, node.resources.compute_demand + node.current_load))
            noise_cmds.append(RectCommand2D(r, c, r, c, max(0.0, local_demand * 0.15)))

        bw_capacity = apply_rect_add_2d(self.rows, self.cols, bw_capacity_cmds)
        bw_demand = apply_rect_add_2d(self.rows, self.cols, bw_demand_cmds)
        compute_capacity = apply_rect_add_2d(self.rows, self.cols, compute_capacity_cmds)
        compute_demand = apply_rect_add_2d(self.rows, self.cols, compute_demand_cmds)
        coverage = apply_rect_add_2d(self.rows, self.cols, coverage_cmds)
        noise_add = apply_rect_add_2d(self.rows, self.cols, noise_cmds)

        self.layers["bandwidth_capacity"] = self._add_base(bw_capacity, self.base_bandwidth_capacity)
        self.layers["bandwidth_demand"] = bw_demand
        self.layers["compute_capacity"] = self._add_base(compute_capacity, self.base_compute_capacity)
        self.layers["compute_demand"] = compute_demand
        self.layers["sensor_coverage"] = coverage
        self.layers["noise"] = self._add_base(noise_add, self.base_noise)
        self.layers["congestion"] = self._compute_congestion()
        if unresolved_tasks is not None:
            self.layers["mission_pressure"] = self._compute_mission_pressure(unresolved_tasks)

    def _add_base(self, grid: Grid, base: float) -> Grid:
        return [[cell + base for cell in row] for row in grid]

    def _compute_congestion(self) -> Grid:
        out = zeros_2d(self.rows, self.cols, 0.0)
        for r in range(self.rows):
            for c in range(self.cols):
                bw_over = max(0.0, self.layers["bandwidth_demand"][r][c] - self.layers["bandwidth_capacity"][r][c])
                compute_over = max(0.0, self.layers["compute_demand"][r][c] - self.layers["compute_capacity"][r][c])
                out[r][c] = bw_over + compute_over + self.layers["noise"][r][c]
        return out

    def _compute_mission_pressure(self, unresolved_tasks: Grid) -> Grid:
        out = zeros_2d(self.rows, self.cols, 0.0)
        for r in range(self.rows):
            for c in range(self.cols):
                coverage_deficit = max(0.0, 1.0 - self.layers["sensor_coverage"][r][c])
                out[r][c] = unresolved_tasks[r][c] + self.layers["congestion"][r][c] + coverage_deficit
        return out

    def observe(self, position: Tuple[int, int]) -> Dict[str, float]:
        r, c = position
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(f"position out of bounds: {position}")
        return {name: grid[r][c] for name, grid in self.layers.items()}

    def layer_mean(self, name: str) -> float:
        grid = self.layers[name]
        total = sum(sum(row) for row in grid)
        return total / (self.rows * self.cols)

    def layer_max(self, name: str) -> Tuple[float, int, int]:
        return argmax_first_2d(self.layers[name])

    def layer_min(self, name: str) -> Tuple[float, int, int]:
        return argmin_first_2d(self.layers[name])

    def row_projection(self, name: str) -> List[float]:
        """Collapse a 2D layer to row-wise sums for corridor/window analyses."""

        return [sum(row) for row in self.layers[name]]

    def col_projection(self, name: str) -> List[float]:
        """Collapse a 2D layer to column-wise sums for corridor/window analyses."""

        return [sum(self.layers[name][r][c] for r in range(self.rows)) for c in range(self.cols)]

    def weakest_bandwidth_window_by_row(self, window_height: int) -> Tuple[float, int]:
        """Find the weakest contiguous horizontal band by bandwidth capacity."""

        return min_fixed_window(self.row_projection("bandwidth_capacity"), window_height)

    def safest_congestion_corridor_by_row(self, threshold: float) -> Tuple[int, int, int, float]:
        """Longest row band whose aggregate congestion/noise is under threshold."""

        return max_window_under_threshold(self.row_projection("congestion"), threshold)

    def min_window_2d(self, layer: str, height: int, width: int) -> Tuple[float, int, int]:
        """Return minimum-sum 2D window and its top-left coordinate."""

        if height <= 0 or width <= 0 or height > self.rows or width > self.cols:
            return (-1.0, -1, -1)
        ps = prefix_sum_2d(self.layers[layer])
        best = float("inf")
        best_r = best_c = 0
        for r in range(self.rows - height + 1):
            for c in range(self.cols - width + 1):
                total = region_sum(ps, r, c, r + height - 1, c + width - 1)
                if total < best:
                    best = total
                    best_r = r
                    best_c = c
        return best, best_r, best_c

    def max_window_2d(self, layer: str, height: int, width: int) -> Tuple[float, int, int]:
        """Return maximum-sum 2D window and its top-left coordinate."""

        if height <= 0 or width <= 0 or height > self.rows or width > self.cols:
            return (-1.0, -1, -1)
        ps = prefix_sum_2d(self.layers[layer])
        best = float("-inf")
        best_r = best_c = 0
        for r in range(self.rows - height + 1):
            for c in range(self.cols - width + 1):
                total = region_sum(ps, r, c, r + height - 1, c + width - 1)
                if total > best:
                    best = total
                    best_r = r
                    best_c = c
        return best, best_r, best_c
