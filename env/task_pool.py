from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from .task_generator import Task


class TaskPool:
    def __init__(self, tasks: Sequence[Task]):
        self.tasks: List[Task] = list(tasks)
        self.by_id: Dict[int, Task] = {t.id: t for t in self.tasks}

    def update_release_and_expiry(self, now: float) -> None:
        for task in self.tasks:
            if task.status == "unreleased" and task.release_time <= now:
                task.status = "unassigned"
            if task.status in {"unassigned", "assigned"} and now > task.deadline:
                task.status = "expired"

    def available_tasks(self, now: float, include_vehicle_only: bool = False) -> List[Task]:
        self.update_release_and_expiry(now)
        out = []
        for task in self.tasks:
            if task.status == "unassigned":
                if include_vehicle_only or task.executor in {"drone", "both"}:
                    out.append(task)
        return out

    def active_tasks(self) -> List[Task]:
        return [t for t in self.tasks if t.status in {"unassigned", "assigned", "executing"}]

    def mark_assigned(self, task_ids: Iterable[int], agent: str, now: float) -> None:
        for tid in task_ids:
            task = self.by_id[int(tid)]
            if task.status == "unassigned":
                task.status = "assigned"
                task.assigned_to = agent

    def mark_executing(self, task_id: int, agent: str, now: float) -> None:
        task = self.by_id[int(task_id)]
        if task.status in {"unassigned", "assigned"}:
            task.status = "executing"
            task.assigned_to = agent
            task.start_time = now

    def mark_done(self, task_id: int, now: float, agent: str) -> None:
        task = self.by_id[int(task_id)]
        task.status = "done"
        task.assigned_to = agent
        if task.start_time is None:
            task.start_time = now
        task.completion_time = now

    def generated_until(self, now: float) -> int:
        return sum(t.release_time <= now for t in self.tasks)
