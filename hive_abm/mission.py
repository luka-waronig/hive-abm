"""Mission graph representation for Hive ABM experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class Task:
    """A mission subtask with dependencies and spatial relevance."""

    task_id: int
    task_type: str
    sector: Tuple[int, int]
    difficulty: float = 0.5
    value: float = 1.0
    dependencies: List[int] = field(default_factory=list)
    required_permission: str = "solve_task"
    required_role: Optional[str] = None
    required_tool: Optional[str] = None
    status: str = "pending"  # pending, in_progress, awaiting_verification, completed, failed
    assigned_to: Optional[int] = None
    active_phase: str = "solve"
    work_remaining: int = 0
    assigned_step: Optional[int] = None
    assignment_hops: int = 0
    attempts: int = 0
    verification_attempts: int = 0
    solved_by: Optional[int] = None
    solution_quality: float = 0.0
    solution_step: Optional[int] = None
    artifact_id: Optional[str] = None
    verified: bool = False
    quality: float = 0.0
    completion_step: Optional[int] = None

    @property
    def is_complete(self) -> bool:
        return self.status == "completed"


@dataclass
class MissionGraph:
    """A dependency-aware mission task graph."""

    tasks: Dict[int, Task]
    quality_threshold: float = 0.55
    timeout_steps: int = 100

    @classmethod
    def from_tasks(cls, tasks: Iterable[Task], quality_threshold: float = 0.55, timeout_steps: int = 100) -> "MissionGraph":
        return cls(tasks={task.task_id: task for task in tasks}, quality_threshold=quality_threshold, timeout_steps=timeout_steps)

    def dependencies_complete(self, task: Task) -> bool:
        return all(self.tasks[dep].status == "completed" for dep in task.dependencies)

    def available_tasks(self) -> List[Task]:
        return [
            task
            for task in self.tasks.values()
            if task.status == "pending" and self.dependencies_complete(task)
        ]

    def available_verification_tasks(self) -> List[Task]:
        return [
            task
            for task in self.tasks.values()
            if task.status == "awaiting_verification" and task.artifact_id is not None
        ]

    def assign(
        self,
        task_id: int,
        node_id: int,
        phase: str = "solve",
        work_required: int = 1,
        step: Optional[int] = None,
        assignment_hops: int = 0,
    ) -> None:
        if phase not in {"solve", "verify"}:
            raise ValueError(f"unknown task phase: {phase}")
        task = self.tasks[task_id]
        if phase == "solve" and task.status != "pending":
            raise ValueError(f"cannot solve task {task_id} from status {task.status}")
        if phase == "verify" and task.status != "awaiting_verification":
            raise ValueError(f"cannot verify task {task_id} from status {task.status}")
        task.status = "in_progress"
        task.assigned_to = node_id
        task.active_phase = phase
        task.work_remaining = max(1, work_required)
        task.assigned_step = step
        task.assignment_hops = max(0, assignment_hops)

    def release(self, task_id: int) -> None:
        task = self.tasks[task_id]
        if task.status == "in_progress":
            task.status = "awaiting_verification" if task.active_phase == "verify" else "pending"
            task.assigned_to = None
            task.active_phase = "solve"
            task.work_remaining = 0
            task.assigned_step = None
            task.assignment_hops = 0

    def progress_assigned_task(self, task_id: int) -> bool:
        """Advance assigned work by one tick and return whether it finished."""

        task = self.tasks[task_id]
        if task.status != "in_progress":
            return False
        task.work_remaining = max(0, task.work_remaining - 1)
        return task.work_remaining == 0

    def mark_awaiting_verification(self, task_id: int, quality: float, step: int, solved_by: int, artifact_id: str) -> None:
        task = self.tasks[task_id]
        task.status = "awaiting_verification"
        task.assigned_to = None
        task.active_phase = "verify"
        task.work_remaining = 0
        task.assigned_step = None
        task.assignment_hops = 0
        task.solved_by = solved_by
        task.solution_quality = max(task.solution_quality, quality)
        task.solution_step = step
        task.artifact_id = artifact_id

    def complete(self, task_id: int, quality: float, step: int, verified: bool = False) -> None:
        task = self.tasks[task_id]
        task.status = "completed"
        task.quality = max(task.quality, quality, task.solution_quality)
        task.verified = task.verified or verified
        task.completion_step = step
        task.assigned_to = None
        task.active_phase = "complete"
        task.work_remaining = 0
        task.assigned_step = None
        task.assignment_hops = 0

    def record_attempt(self, task_id: int, phase: str = "solve") -> None:
        task = self.tasks[task_id]
        if phase == "verify":
            task.verification_attempts += 1
        else:
            task.attempts += 1

    def fail_attempt(self, task_id: int, phase: str = "solve", release: bool = True, count_attempt: bool = True) -> None:
        task = self.tasks[task_id]
        if count_attempt:
            self.record_attempt(task_id, phase)
        if release:
            if phase == "verify":
                task.status = "pending"
                task.solved_by = None
                task.solution_quality = 0.0
                task.solution_step = None
                task.artifact_id = None
            else:
                task.status = "pending"
            task.assigned_to = None
            task.active_phase = "solve"
            task.work_remaining = 0
            task.assigned_step = None
            task.assignment_hops = 0

    def is_success(self) -> bool:
        return all(task.status == "completed" for task in self.tasks.values())

    def is_failed(self, step: int) -> bool:
        return step >= self.timeout_steps and not self.is_success()

    def completion_rate(self) -> float:
        if not self.tasks:
            return 1.0
        return sum(1 for task in self.tasks.values() if task.status == "completed") / len(self.tasks)

    def total_value_completed(self) -> float:
        return sum(task.value for task in self.tasks.values() if task.status == "completed")

    def mean_quality(self) -> float:
        completed = [task.quality for task in self.tasks.values() if task.status == "completed"]
        return sum(completed) / len(completed) if completed else 0.0

    def verified_completion_rate(self) -> float:
        if not self.tasks:
            return 1.0
        return sum(1 for task in self.tasks.values() if task.status == "completed" and task.verified) / len(self.tasks)

    def status_counts(self) -> Dict[str, int]:
        counts = {
            "pending": 0,
            "in_progress": 0,
            "awaiting_verification": 0,
            "completed": 0,
            "failed": 0,
        }
        for task in self.tasks.values():
            counts[task.status] = counts.get(task.status, 0) + 1
        return counts

    def total_attempts(self) -> int:
        return sum(task.attempts for task in self.tasks.values())

    def total_verification_attempts(self) -> int:
        return sum(task.verification_attempts for task in self.tasks.values())

    def unresolved_by_sector(self, rows: int, cols: int) -> List[List[float]]:
        grid = [[0.0] * cols for _ in range(rows)]
        for task in self.tasks.values():
            if task.status != "completed":
                r, c = task.sector
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r][c] += task.value
        return grid

    def nearest_available_task_sector(self, position: Tuple[int, int], include_verification: bool = True) -> Optional[Tuple[int, int]]:
        available = self.available_tasks()
        if include_verification:
            available = self.available_verification_tasks() + available
        if not available:
            return None
        r, c = position
        best = min(available, key=lambda t: abs(t.sector[0] - r) + abs(t.sector[1] - c))
        return best.sector

    def task_for_node(self, node_id: int) -> Optional[Task]:
        for task in self.tasks.values():
            if task.assigned_to == node_id and task.status == "in_progress":
                return task
        return None


def make_random_mission(
    rows: int,
    cols: int,
    n_tasks: int,
    rng,
    task_types: Sequence[str] = ("planning", "coding", "verification", "routing", "retrieval", "analysis"),
    dependency_probability: float = 0.12,
    timeout_steps: int = 100,
) -> MissionGraph:
    """Generate a small random DAG-like mission over spatial sectors."""

    tasks: List[Task] = []
    for i in range(n_tasks):
        sector = (rng.randrange(rows), rng.randrange(cols))
        task_type = rng.choice(task_types)
        deps = [j for j in range(i) if rng.random() < dependency_probability]
        difficulty = rng.uniform(0.25, 0.85)
        value = rng.uniform(0.8, 1.5)
        required_role = "verifier" if task_type == "verification" else None
        tasks.append(
            Task(
                task_id=i,
                task_type=task_type,
                sector=sector,
                difficulty=difficulty,
                value=value,
                dependencies=deps,
                required_role=required_role,
            )
        )
    return MissionGraph.from_tasks(tasks, quality_threshold=0.55, timeout_steps=timeout_steps)
