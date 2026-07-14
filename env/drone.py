from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Drone:
    id: str
    home_vehicle_id: str
    position: int
    battery_capacity: float
    battery: float
    status: str = "idle"  # idle, flying, waiting_recovery, charging
    sortie: Optional[Any] = None
    segment_index: int = 0
    segment_remaining: float = 0.0
    wait_time: float = 0.0
    fly_time: float = 0.0
    service_time: float = 0.0
    energy_used: float = 0.0
    completed_task_ids: List[int] = field(default_factory=list)
    log: List[Tuple[float, str, int]] = field(default_factory=list)

    def is_idle(self) -> bool:
        return self.status == "idle"

    def reset_after_recovery(self, node: int, now: float) -> None:
        self.position = int(node)
        self.battery = self.battery_capacity
        self.status = "idle"
        self.sortie = None
        self.segment_index = 0
        self.segment_remaining = 0.0
        self.log.append((now, "recovered", int(node)))

    def assign_sortie(self, sortie: Any, now: float) -> None:
        if not self.is_idle():
            raise ValueError(f"Drone {self.id} is not idle")
        self.sortie = sortie
        self.status = "flying"
        self.segment_index = 0
        self.segment_remaining = sortie.segments[0]["duration"] if sortie.segments else 0.0
        self.log.append((now, "launch", sortie.launch_node))
