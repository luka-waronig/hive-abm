"""Metrics collection for Hive ABM simulations."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class TickMetrics:
    step: int
    completion_rate: float
    verified_completion_rate: float
    completed_value: float
    mean_quality: float
    mean_congestion: float
    peak_congestion: float
    peak_congestion_row: int
    peak_congestion_col: int
    mean_coverage: float
    mean_bandwidth_capacity: float
    mean_bandwidth_demand: float
    denied_actions: int
    messages_sent: int
    duplicate_work: int
    blocked_nodes: int
    failed_nodes: int
    idle_nodes: int
    working_nodes: int
    pending_tasks: int
    in_progress_tasks: int
    awaiting_verification_tasks: int
    completed_tasks: int
    verified_tasks: int
    mean_assignment_hops: float
    safe_corridor_length: int
    safe_corridor_sum: float
    graph_centralization: float
    mean_degree: float


@dataclass
class RunSummary:
    run_id: str
    success: bool
    steps: int
    final_completion_rate: float
    final_verified_completion_rate: float
    final_quality: float
    final_awaiting_verification: int
    total_attempts: int
    total_verification_attempts: int
    total_denied_actions: int
    total_messages: int
    total_duplicate_work: int
    final_mean_congestion: float
    final_peak_congestion: float
    final_mean_coverage: float
    safe_corridor_length: int
    graph_centralization: float
    mean_net_contribution: float
    metadata: Dict[str, str | int | float] = field(default_factory=dict)


@dataclass
class MetricsCollector:
    ticks: List[TickMetrics] = field(default_factory=list)
    denied_actions_total: int = 0
    messages_total: int = 0
    duplicate_work_total: int = 0

    def log_tick(self, tick: TickMetrics) -> None:
        self.ticks.append(tick)
        self.denied_actions_total += tick.denied_actions
        self.messages_total += tick.messages_sent
        self.duplicate_work_total += tick.duplicate_work

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.ticks:
            path.write_text("")
            return
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.ticks[0]).keys()))
            writer.writeheader()
            for tick in self.ticks:
                writer.writerow(asdict(tick))


def write_summaries_csv(summaries: Iterable[RunSummary], path: str | Path) -> None:
    summaries = list(summaries)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summaries:
        path.write_text("")
        return
    base_fields = [k for k in asdict(summaries[0]).keys() if k != "metadata"]
    meta_keys = sorted({key for s in summaries for key in s.metadata.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + meta_keys)
        writer.writeheader()
        for summary in summaries:
            row = asdict(summary)
            metadata = row.pop("metadata")
            row.update(metadata)
            writer.writerow(row)


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
