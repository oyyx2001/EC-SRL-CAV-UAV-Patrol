from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import math


@dataclass
class EpisodeMetrics:
    seed: int = 0
    method: str = ""
    scenario: str = ""
    generated_tasks: int = 0
    completed_tasks: int = 0
    expired_tasks: int = 0
    timeout_tasks: int = 0
    completion_rate: float = 0.0
    timeout_rate: float = 0.0
    avg_response_time_s: float = math.nan
    high_priority_response_time_s: float = math.nan
    weighted_aoi: float = 0.0
    total_energy: float = 0.0
    vehicle_wait_time_s: float = 0.0
    drone_wait_time_s: float = 0.0
    drone_utilization: float = 0.0
    tasks_per_hour: float = 0.0
    illegal_actions: int = 0
    avg_tasks_per_sortie: float = 0.0
    num_sorties: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_task_metrics(tasks: List[Any], high_priority_threshold: int) -> Dict[str, float]:
    completed = [t for t in tasks if t.status == "done"]
    expired = [t for t in tasks if t.status == "expired"]
    timeouts = [t for t in tasks if t.completion_time is not None and t.completion_time > t.deadline]
    responses = [t.completion_time - t.release_time for t in completed if t.completion_time is not None]
    high_responses = [
        t.completion_time - t.release_time
        for t in completed
        if t.priority >= high_priority_threshold and t.completion_time is not None
    ]
    n = len(tasks)
    return {
        "generated_tasks": n,
        "completed_tasks": len(completed),
        "expired_tasks": len(expired),
        "timeout_tasks": len(timeouts),
        "completion_rate": len(completed) / n if n else 0.0,
        "timeout_rate": (len(expired) + len(timeouts)) / n if n else 0.0,
        "avg_response_time_s": float(sum(responses) / len(responses)) if responses else math.nan,
        "high_priority_response_time_s": float(sum(high_responses) / len(high_responses)) if high_responses else math.nan,
    }
