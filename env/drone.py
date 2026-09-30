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
    status: str = "idle"  # idle, launching, flying, waiting_recovery, recovering, charging
    sortie: Optional[Any] = None
    segment_index: int = 0
    segment_remaining: float = 0.0
    wait_time: float = 0.0
    fly_time: float = 0.0
    service_time: float = 0.0
    energy_used: float = 0.0
    completed_task_ids: List[int] = field(default_factory=list)
    log: List[Tuple[float, str, int]] = field(default_factory=list)
    schedule_version: int = 0
    active_voxel_path: List[Tuple[int, int, int]] = field(default_factory=list)
    active_path_xyz: List[Tuple[float, float, float]] = field(default_factory=list)
    path_total_distance_m: float = 0.0
    path_total_flight_time_s: float = 0.0
    path_segment_index: int = 0
    path_progress_m: float = 0.0
    task_arrival_times: Dict[int, float] = field(default_factory=dict)
    recovery_eta: Optional[float] = None
    recovery_remaining_s: float = 0.0
    launch_remaining_s: float = 0.0
    actual_launch_duration_s: float = 0.0

    def is_idle(self) -> bool:
        return self.status == "idle"

    def reset_after_recovery(self, node: int, now: float) -> None:
        self.schedule_version += 1
        self.position = int(node)
        self.battery = self.battery_capacity
        self.status = "idle"
        self.sortie = None
        self.segment_index = 0
        self.segment_remaining = 0.0
        self.active_voxel_path.clear(); self.active_path_xyz.clear(); self.task_arrival_times.clear(); self.recovery_eta = None; self.recovery_remaining_s = 0.0
        self.launch_remaining_s = 0.0; self.actual_launch_duration_s = 0.0
        self.log.append((now, "recovered", int(node)))

    def assign_sortie(self, sortie: Any, now: float) -> None:
        if not self.is_idle():
            raise ValueError(f"Drone {self.id} is not idle")
        self.schedule_version += 1
        self.sortie = sortie
        launch_duration = max(0.0, float(getattr(sortie, "actual_launch_time_s", 0.0)))
        self.actual_launch_duration_s = launch_duration
        self.launch_remaining_s = launch_duration
        self.status = "launching" if launch_duration > 1e-9 else "flying"
        self.segment_index = 0
        self.segment_remaining = sortie.segments[0]["duration"] if sortie.segments else 0.0
        self.active_voxel_path = list(getattr(sortie, "active_voxel_path", [])); self.active_path_xyz = list(getattr(sortie, "active_path_xyz", []))
        self.path_total_distance_m = float(getattr(sortie, "actual_distance_m", 0.0)); self.path_total_flight_time_s = float(getattr(sortie, "actual_flight_time_s", 0.0))
        self.path_segment_index = 0; self.path_progress_m = 0.0; self.task_arrival_times = dict(getattr(sortie, "task_arrival_times", {})); self.recovery_eta = getattr(sortie, "recovery_eta", None)
        self.log.append((now, "launch", sortie.launch_node))
