from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from .task_generator import Task


class TaskPool:
    def __init__(
        self,
        tasks: Sequence[Task],
        *,
        task_release_horizon_s: Optional[float] = None,
        expire_executing: bool = False,
    ):
        self.tasks: List[Task] = list(tasks)
        self.by_id: Dict[int, Task] = {t.id: t for t in self.tasks}
        self.task_release_horizon_s = (
            None if task_release_horizon_s is None else float(task_release_horizon_s)
        )
        self.expire_executing = bool(expire_executing)
        # Per-physical-segment state is intentionally separate from the
        # directed CAV graph.  It lets each periodic obligation retain its own
        # lifecycle while exposing the latest inspection history to the next
        # released cycle.
        self._periodic_segment_state: Dict[str, Dict[str, object]] = {}
        for task in self.tasks:
            if bool(task.metadata.get("periodic_task", False)):
                segment = str(task.metadata.get("inspection_segment_id", task.metadata.get("edge_id", "")))
                self._periodic_segment_state.setdefault(segment, {
                    "last_inspection_time": None,
                    "periodic_completion_count": 0,
                    "periodic_expired_count": 0,
                })

    def update_release_and_expiry(self, now: float) -> None:
        for task in self.tasks:
            release_allowed = (
                self.task_release_horizon_s is None
                or float(task.release_time) < self.task_release_horizon_s
            )
            if task.status == "unreleased" and release_allowed and task.release_time <= now:
                task.status = "unassigned"
                self._update_periodic_metadata(task, now, "active")
            expirable = {"unassigned", "assigned"}
            if self.expire_executing:
                expirable.add("executing")
            # Periodic obligations have a closed release boundary: expire the
            # previous cycle at its deadline before releasing the next one,
            # so a physical segment never has two active cycle obligations at
            # the same timestamp.  Preserve the legacy strict-expiry rule for
            # all non-periodic task streams.
            deadline_reached = now >= task.deadline if bool(task.metadata.get("periodic_task", False)) else now > task.deadline
            if task.status in expirable and deadline_reached:
                task.status = "expired"
                self._update_periodic_metadata(task, now, "expired")

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
        # A task which has already passed its own deadline remains a genuine
        # timeout.  In particular, a late service-complete event must not turn
        # one timeout into a second completion during the drain phase.
        if task.status in {"expired", "censored"}:
            return
        task.status = "done"
        task.assigned_to = agent
        if task.start_time is None:
            task.start_time = now
        task.completion_time = now
        self._update_periodic_metadata(task, now, "completed")

    def _update_periodic_metadata(self, task: Task, now: float, status: str) -> None:
        """Synchronise auditable cycle fields without merging obligations."""
        if not bool(task.metadata.get("periodic_task", False)):
            return
        segment = str(task.metadata.get("inspection_segment_id", task.metadata.get("edge_id", "")))
        state = self._periodic_segment_state.setdefault(segment, {
            "last_inspection_time": None,
            "periodic_completion_count": 0,
            "periodic_expired_count": 0,
        })
        if status == "completed":
            state["last_inspection_time"] = float(now)
            state["periodic_completion_count"] = int(state["periodic_completion_count"]) + 1
        elif status == "expired":
            state["periodic_expired_count"] = int(state["periodic_expired_count"]) + 1
        last = state["last_inspection_time"]
        task.metadata.update(
            current_cycle_status=status,
            last_inspection_time=last,
            age_since_last_inspection_s=None if last is None else max(0.0, float(now) - float(last)),
            periodic_completion_count=int(state["periodic_completion_count"]),
            periodic_expired_count=int(state["periodic_expired_count"]),
        )

    def generated_until(self, now: float) -> int:
        return sum(
            t.release_time <= now
            and (
                self.task_release_horizon_s is None
                or float(t.release_time) < self.task_release_horizon_s
            )
            for t in self.tasks
        )
