"""Sliding-window primitives for deployment and corridor queries."""

from __future__ import annotations

from typing import List, Sequence, Tuple

Number = float


def min_fixed_window(values: Sequence[Number], width: int) -> Tuple[Number, int]:
    """Return (minimum_sum, first_start_index) for a fixed-size window.

    This generalizes the relay-insertion exercise. It can locate the weakest
    contiguous bandwidth region, least covered corridor, or any other minimum
    resource window.
    """

    n = len(values)
    if width <= 0 or width > n:
        return (-1.0, -1)
    current = sum(values[:width])
    best = current
    best_start = 0
    for start in range(1, n - width + 1):
        current += values[start + width - 1] - values[start - 1]
        if current < best:
            best = current
            best_start = start
    return float(best), best_start


def max_fixed_window(values: Sequence[Number], width: int) -> Tuple[Number, int]:
    """Return (maximum_sum, first_start_index) for a fixed-size window."""

    n = len(values)
    if width <= 0 or width > n:
        return (-1.0, -1)
    current = sum(values[:width])
    best = current
    best_start = 0
    for start in range(1, n - width + 1):
        current += values[start + width - 1] - values[start - 1]
        if current > best:
            best = current
            best_start = start
    return float(best), best_start


def max_window_under_threshold(values: Sequence[Number], threshold: Number) -> Tuple[int, int, int, Number]:
    """Return longest contiguous window whose sum is <= threshold.

    Returns ``(length, start, end, sum)``. If no single item is safe, returns
    ``(0, -1, -1, 0.0)``.

    Assumes non-negative values. This is appropriate for congestion/noise/load
    maps used in the Hive ABM.
    """

    if threshold < 0:
        return (0, -1, -1, 0.0)
    best_len = 0
    best_start = -1
    best_end = -1
    best_sum = 0.0
    left = 0
    current = 0.0
    for right, value in enumerate(values):
        if value < 0:
            raise ValueError("max_window_under_threshold assumes non-negative values")
        current += value
        while current > threshold and left <= right:
            current -= values[left]
            left += 1
        length = right - left + 1
        if length > best_len:
            best_len = length
            best_start = left
            best_end = right
            best_sum = current
    return best_len, best_start, best_end, float(best_sum)


def fixed_window_sums(values: Sequence[Number], width: int) -> List[Number]:
    """Return all fixed-window sums."""

    n = len(values)
    if width <= 0 or width > n:
        return []
    current = sum(values[:width])
    out = [float(current)]
    for start in range(1, n - width + 1):
        current += values[start + width - 1] - values[start - 1]
        out.append(float(current))
    return out
