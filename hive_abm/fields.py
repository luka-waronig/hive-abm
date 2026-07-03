"""Efficient spatial field primitives for the Hive ABM.

These are the algorithmic kernels adapted from the uploaded exercises:
- 1D difference arrays for linear sector range updates.
- 2D difference arrays for rectangular grid updates.
- Prefix-sum regional queries and first argmax/argmin helpers.

The functions are deliberately small and dependency-free. They are suitable for
large ABM runs where each node action updates a shared resource field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

Number = float
Grid2D = List[List[Number]]


@dataclass(frozen=True)
class RangeCommand1D:
    """Add ``delta`` to a closed interval [start, end] on a 0-indexed line."""

    start: int
    end: int
    delta: Number


@dataclass(frozen=True)
class RectCommand2D:
    """Add ``delta`` to a closed rectangle [r1:r2, c1:c2] on a 0-indexed grid."""

    r1: int
    c1: int
    r2: int
    c2: int
    delta: Number


def apply_range_add_1d(size: int, commands: Iterable[RangeCommand1D]) -> List[Number]:
    """Materialize a 1D field after inclusive range-add commands.

    This is the 0-indexed analogue of the 1D bandwidth/congestion exercise.
    Invalid or fully out-of-bound commands are clipped rather than raising.
    """

    if size <= 0:
        return []

    diff = [0.0] * (size + 1)
    for cmd in commands:
        start = max(0, cmd.start)
        end = min(size - 1, cmd.end)
        if start > end:
            continue
        diff[start] += cmd.delta
        if end + 1 < size:
            diff[end + 1] -= cmd.delta

    out: List[Number] = []
    current = 0.0
    for i in range(size):
        current += diff[i]
        out.append(current)
    return out


def apply_rect_add_2d(rows: int, cols: int, commands: Iterable[RectCommand2D]) -> Grid2D:
    """Materialize a 2D field after inclusive rectangular range-add commands.

    Correct 2D difference-array markers:
        diff[r1][c1] += k
        diff[r1][c2 + 1] -= k
        diff[r2 + 1][c1] -= k
        diff[r2 + 1][c2 + 1] += k

    Note the subtraction at ``c2 + 1`` rather than ``c1 + 1``: the update must
    continue through the whole rectangle until its right boundary.
    """

    if rows <= 0 or cols <= 0:
        return []

    diff = [[0.0] * (cols + 1) for _ in range(rows + 1)]
    for cmd in commands:
        r1 = max(0, cmd.r1)
        c1 = max(0, cmd.c1)
        r2 = min(rows - 1, cmd.r2)
        c2 = min(cols - 1, cmd.c2)
        if r1 > r2 or c1 > c2:
            continue

        diff[r1][c1] += cmd.delta
        diff[r1][c2 + 1] -= cmd.delta
        diff[r2 + 1][c1] -= cmd.delta
        diff[r2 + 1][c2 + 1] += cmd.delta

    # Convert difference markers into the actual field via 2D prefix accumulation.
    out = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            top = diff[r - 1][c] if r > 0 else 0.0
            left = diff[r][c - 1] if c > 0 else 0.0
            diag = diff[r - 1][c - 1] if r > 0 and c > 0 else 0.0
            diff[r][c] += top + left - diag
            out[r][c] = diff[r][c]
    return out


def add_grids(*grids: Grid2D) -> Grid2D:
    """Elementwise sum of grids with identical shape."""

    if not grids:
        return []
    rows = len(grids[0])
    cols = len(grids[0][0]) if rows else 0
    out = [[0.0] * cols for _ in range(rows)]
    for grid in grids:
        if len(grid) != rows or any(len(row) != cols for row in grid):
            raise ValueError("all grids must have the same shape")
        for r in range(rows):
            for c in range(cols):
                out[r][c] += grid[r][c]
    return out


def zeros_2d(rows: int, cols: int, value: Number = 0.0) -> Grid2D:
    return [[float(value) for _ in range(cols)] for _ in range(rows)]


def prefix_sum_2d(grid: Grid2D) -> Grid2D:
    """Return a padded 2D prefix-sum table of shape (rows+1, cols+1)."""

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    ps = [[0.0] * (cols + 1) for _ in range(rows + 1)]
    for r in range(rows):
        row_acc = 0.0
        for c in range(cols):
            row_acc += grid[r][c]
            ps[r + 1][c + 1] = ps[r][c + 1] + row_acc
    return ps


def region_sum(ps: Grid2D, r1: int, c1: int, r2: int, c2: int) -> Number:
    """Query sum over closed rectangle from a padded prefix-sum table."""

    rows = len(ps) - 1
    cols = len(ps[0]) - 1 if rows >= 0 and ps else 0
    if rows <= 0 or cols <= 0:
        return 0.0
    r1 = max(0, r1)
    c1 = max(0, c1)
    r2 = min(rows - 1, r2)
    c2 = min(cols - 1, c2)
    if r1 > r2 or c1 > c2:
        return 0.0
    return ps[r2 + 1][c2 + 1] - ps[r1][c2 + 1] - ps[r2 + 1][c1] + ps[r1][c1]


def argmax_first_1d(values: Sequence[Number]) -> Tuple[Number, int]:
    """Return (max_value, first_index)."""

    if not values:
        return (0.0, -1)
    best_val = values[0]
    best_idx = 0
    for i, val in enumerate(values[1:], start=1):
        if val > best_val:
            best_val = val
            best_idx = i
    return best_val, best_idx


def argmin_first_1d(values: Sequence[Number]) -> Tuple[Number, int]:
    """Return (min_value, first_index)."""

    if not values:
        return (0.0, -1)
    best_val = values[0]
    best_idx = 0
    for i, val in enumerate(values[1:], start=1):
        if val < best_val:
            best_val = val
            best_idx = i
    return best_val, best_idx


def argmax_first_2d(grid: Grid2D) -> Tuple[Number, int, int]:
    """Return (max_value, row, col), scanning top-to-bottom then left-to-right."""

    if not grid or not grid[0]:
        return (0.0, -1, -1)
    best_val = grid[0][0]
    best_r = best_c = 0
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val > best_val:
                best_val = val
                best_r = r
                best_c = c
    return best_val, best_r, best_c


def argmin_first_2d(grid: Grid2D) -> Tuple[Number, int, int]:
    """Return (min_value, row, col), scanning top-to-bottom then left-to-right."""

    if not grid or not grid[0]:
        return (0.0, -1, -1)
    best_val = grid[0][0]
    best_r = best_c = 0
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val < best_val:
                best_val = val
                best_r = r
                best_c = c
    return best_val, best_r, best_c


def clip_position(pos: Tuple[int, int], rows: int, cols: int) -> Tuple[int, int]:
    r, c = pos
    return max(0, min(rows - 1, r)), max(0, min(cols - 1, c))
