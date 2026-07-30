from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from itertools import count
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


EVENT_PRIORITIES: Dict[str, int] = {
    "EPISODE_END": 0,
    "CAV_RECOVERY_COMPLETE": 10,
    "UAV_RECOVERED": 11,
    "TASK_SERVICE_COMPLETE": 12,
    "UAV_SERVICE_COMPLETE": 13,
    "UAV_RECOVERY_ARRIVAL": 20,
    "UAV_TASK_ARRIVAL": 21,
    "UAV_EDGE_COVERAGE_COMPLETE": 21,
    "UAV_DESCENT_COMPLETE": 21,
    "UAV_CLIMB_COMPLETE": 21,
    "CAV_RECOVERY_NODE_ARRIVAL": 22,
    "CAV_NODE_ARRIVAL": 23,
    "CAV_WAIT_COMPLETE": 24,
    "TASK_TIMEOUT": 30,
    "TASK_RELEASE": 40,
    "UAV_LAUNCH": 50,
}


@dataclass(order=True, frozen=True)
class Event:
    """A deterministic system-event marker.

    Physical state is advanced by :class:`Simulator`; these markers define the
    exact boundaries at which that state must be inspected and decision
    eligibility may be recomputed.
    """

    sort_key: Tuple[float, int, int] = field(init=False, repr=False)
    time: float
    event_type: str = field(compare=False)
    entity_id: str = field(compare=False, default="")
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)
    priority: int = field(compare=False, default=100)
    sequence_id: int = field(compare=False, default=0)
    schedule_version: int = field(compare=False, default=0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sort_key",
            (float(self.time), int(self.priority), int(self.sequence_id)),
        )


class EventQueue:
    """Stable heap queue with same-time batching and stale-event filtering."""

    def __init__(self, time_tolerance: float = 1e-9) -> None:
        self.time_tolerance = float(time_tolerance)
        self._heap: List[Event] = []
        self._sequence = count()
        self._scheduled_keys: set[Tuple[Any, ...]] = set()
        self._executed_sequence_ids: set[int] = set()
        self.stale_event_count = 0
        self.duplicate_event_count = 0
        self.invalid_time_event_count = 0
        self.processed_event_count = 0

    def __len__(self) -> int:
        return len(self._heap)

    def schedule(
        self,
        *,
        time: float,
        event_type: str,
        entity_id: str = "",
        payload: Optional[Dict[str, Any]] = None,
        priority: Optional[int] = None,
        schedule_version: int = 0,
        current_time: float = 0.0,
    ) -> Optional[Event]:
        event_time = float(time)
        if event_time < float(current_time) - self.time_tolerance:
            self.invalid_time_event_count += 1
            return None
        event_priority = int(
            EVENT_PRIORITIES.get(str(event_type), 100) if priority is None else priority
        )
        key = (
            round(event_time / max(self.time_tolerance, 1e-12)),
            str(event_type),
            str(entity_id),
            int(schedule_version),
        )
        if key in self._scheduled_keys:
            self.duplicate_event_count += 1
            return None
        event = Event(
            time=event_time,
            event_type=str(event_type),
            entity_id=str(entity_id),
            payload=dict(payload or {}),
            priority=event_priority,
            sequence_id=next(self._sequence),
            schedule_version=int(schedule_version),
        )
        heapq.heappush(self._heap, event)
        self._scheduled_keys.add(key)
        return event

    def _key(self, event: Event) -> Tuple[Any, ...]:
        return (
            round(float(event.time) / max(self.time_tolerance, 1e-12)),
            str(event.event_type),
            str(event.entity_id),
            int(event.schedule_version),
        )

    def _is_stale(
        self,
        event: Event,
        version_getter: Optional[Callable[[Event], Optional[int]]],
    ) -> bool:
        if version_getter is None:
            return False
        current_version = version_getter(event)
        return current_version is not None and int(current_version) != int(event.schedule_version)

    def _discard_invalid_head(
        self,
        current_time: float,
        version_getter: Optional[Callable[[Event], Optional[int]]],
    ) -> None:
        while self._heap:
            event = self._heap[0]
            if float(event.time) < float(current_time) - self.time_tolerance:
                heapq.heappop(self._heap)
                self._scheduled_keys.discard(self._key(event))
                self.invalid_time_event_count += 1
                continue
            if self._is_stale(event, version_getter):
                heapq.heappop(self._heap)
                self._scheduled_keys.discard(self._key(event))
                self.stale_event_count += 1
                continue
            break

    def peek_time(
        self,
        current_time: float,
        version_getter: Optional[Callable[[Event], Optional[int]]] = None,
    ) -> Optional[float]:
        self._discard_invalid_head(current_time, version_getter)
        return None if not self._heap else float(self._heap[0].time)

    def pop_same_time(
        self,
        event_time: float,
        current_time: float,
        version_getter: Optional[Callable[[Event], Optional[int]]] = None,
    ) -> List[Event]:
        self._discard_invalid_head(current_time, version_getter)
        batch: List[Event] = []
        while self._heap and abs(float(self._heap[0].time) - float(event_time)) <= self.time_tolerance:
            event = heapq.heappop(self._heap)
            self._scheduled_keys.discard(self._key(event))
            if self._is_stale(event, version_getter):
                self.stale_event_count += 1
                continue
            if event.sequence_id in self._executed_sequence_ids:
                self.duplicate_event_count += 1
                continue
            self._executed_sequence_ids.add(event.sequence_id)
            batch.append(event)
        batch.sort(key=lambda event: event.sort_key)
        self.processed_event_count += len(batch)
        return batch

    def snapshot(self) -> List[Event]:
        return sorted(self._heap)
