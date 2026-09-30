from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .road_network import RoadNetwork


@dataclass
class Vehicle:
    """CAV state with strict edge-by-edge movement on the directed road network.

    The earlier first-stage implementation used the directed shortest-path travel time
    but jumped directly from the current node to the target rendezvous node when the
    travel time elapsed. This version expands the shortest path into directed edges and
    advances the CAV along those edges. Therefore, every logged CAV movement satisfies
    ``network.G.has_edge(u, v)``.
    """

    id: str
    position: int
    target_node: Optional[int] = None
    at_intersection: bool = True

    # Total remaining time to the final target and remaining time on the current edge.
    remaining_time: float = 0.0
    edge_remaining_time: float = 0.0
    edge_total_time: float = 0.0
    edge_remaining_distance_m: float = 0.0
    edge_realized_speed_mps: float = 0.0
    edge_speed_multiplier: float = 1.0
    edge_stage_index: int = -1
    edge_realized: bool = False

    # Current shortest path being executed.
    path_nodes: List[int] = field(default_factory=list)
    path_index: int = 0

    # Node and edge logs. route_log stores node arrivals; route_edge_log stores actual
    # directed road edges traversed by the CAV.
    route_log: List[Tuple[float, int]] = field(default_factory=list)
    route_edge_log: List[Tuple[float, int, int]] = field(default_factory=list)

    wait_time: float = 0.0
    distance_time: float = 0.0
    distance_traveled_m: float = 0.0
    last_arrival_time: Optional[float] = None
    schedule_version: int = 0
    wait_until: Optional[float] = None
    wait_started_at: Optional[float] = None
    operation_hold_until: Optional[float] = None
    operation_hold_started_at: Optional[float] = None
    planned_operation_hold_until: Optional[float] = None
    last_route_change_time: Optional[float] = None
    route_change_times: List[float] = field(default_factory=list)
    route_commitment_durations: List[float] = field(default_factory=list)
    committed_nodes_remaining: int = 0
    target_history: List[int] = field(default_factory=list)

    def is_idle(self) -> bool:
        return self.target_node is None or self.remaining_time <= 1e-9

    @property
    def current_edge(self) -> Optional[Tuple[int, int]]:
        if self.at_intersection or not self.path_nodes or self.path_index >= len(self.path_nodes) - 1:
            return None
        return (int(self.path_nodes[self.path_index]), int(self.path_nodes[self.path_index + 1]))

    def is_waiting(self, now: float) -> bool:
        return self.wait_until is not None and float(now) < float(self.wait_until) - 1e-9

    def begin_wait(self, until: float, now: float) -> None:
        if not self.at_intersection:
            raise ValueError("CAV can only wait at a road node")
        until = float(until)
        if until <= float(now) + 1e-9:
            raise ValueError("CAV wait must end in the future")
        self.schedule_version += 1
        self.wait_started_at = float(now)
        self.wait_until = until
        self.target_node = None
        self.remaining_time = 0.0
        self.edge_remaining_time = 0.0
        self.edge_total_time = 0.0
        self._reset_edge_realization()
        self.path_nodes = []
        self.path_index = 0
        self.at_intersection = True

    def is_operationally_held(self, now: float) -> bool:
        return self.operation_hold_until is not None and float(now) < float(self.operation_hold_until) - 1e-9

    def planned_operation_hold_remaining(self, now: float) -> float:
        if self.planned_operation_hold_until is None:
            return 0.0
        return max(0.0, float(self.planned_operation_hold_until) - float(now))

    def begin_operation_hold(self, until: float, planned_until: float, now: float) -> None:
        """Pause CAV motion for launch handling without discarding its route."""
        if not self.at_intersection:
            raise ValueError("CAV launch handling can only start at a road node")
        until = max(float(now), float(until))
        planned_until = max(float(now), float(planned_until))
        previous = float(self.operation_hold_until or now)
        previous_planned = float(self.planned_operation_hold_until or now)
        new_until = max(previous, until)
        new_planned = max(previous_planned, planned_until)
        if new_until > previous + 1e-9 or new_planned > previous_planned + 1e-9:
            self.schedule_version += 1
        if new_until > float(now) + 1e-9:
            if self.operation_hold_started_at is None:
                self.operation_hold_started_at = float(now)
            self.operation_hold_until = new_until
        self.planned_operation_hold_until = new_planned

    @staticmethod
    def _uses_staged_random_speed(network: RoadNetwork) -> bool:
        return bool(getattr(getattr(network, "traffic", None), "staged_speed", False))

    @staticmethod
    def _uses_execution_time_uncertainty(network: RoadNetwork) -> bool:
        return bool(
            getattr(getattr(network, "traffic", None), "execution_uncertainty_enabled", False)
        )

    @classmethod
    def _defers_edge_realization(cls, network: RoadNetwork) -> bool:
        return cls._uses_staged_random_speed(network) or cls._uses_execution_time_uncertainty(network)

    def _reset_edge_realization(self) -> None:
        self.edge_remaining_distance_m = 0.0
        self.edge_realized_speed_mps = 0.0
        self.edge_speed_multiplier = 1.0
        self.edge_stage_index = -1
        self.edge_realized = False

    def ensure_edge_realization(
        self,
        now: float,
        network: RoadNetwork,
        *,
        force: bool = False,
    ) -> None:
        """Sample physical CAV speed when an edge is actually entered/resumed."""
        if not self._defers_edge_realization(network):
            return
        if self.target_node is None or not self.path_nodes or self.path_index >= len(self.path_nodes) - 1:
            return
        if self.edge_realized and not force:
            return
        u = int(self.path_nodes[self.path_index])
        v = int(self.path_nodes[self.path_index + 1])
        if self.edge_remaining_distance_m <= 1e-9:
            self.edge_remaining_distance_m = float(network.arc_length((u, v)))
        traffic = getattr(network, "traffic")
        realized = traffic.sample_edge_motion(
            u,
            v,
            float(now),
            float(self.edge_remaining_distance_m),
        )
        self.edge_realized_speed_mps = float(realized["actual_speed_mps"])
        self.edge_speed_multiplier = float(realized["speed_multiplier"])
        self.edge_stage_index = int(realized["stage_index"])
        self.edge_total_time = float(realized["duration_s"])
        self.edge_remaining_time = float(realized["duration_s"])
        self.edge_realized = True
        self.at_intersection = False
        future_time = sum(
            network.get_vehicle_edge_time(int(a), int(b), actual=False)
            for a, b in zip(
                self.path_nodes[self.path_index + 1 : -1],
                self.path_nodes[self.path_index + 2 :],
            )
        )
        self.remaining_time = float(self.edge_remaining_time + future_time)
        self.schedule_version += 1

    def planned_remaining_time(self, now: float, network: RoadNetwork) -> float:
        """Planner-facing route ETA that never exposes a realized speed draw."""
        if self.target_node is None or not self.path_nodes or self.path_index >= len(self.path_nodes) - 1:
            return 0.0
        if not self._defers_edge_realization(network):
            return max(0.0, float(self.remaining_time))
        traffic = getattr(network, "traffic")
        u = int(self.path_nodes[self.path_index])
        v = int(self.path_nodes[self.path_index + 1])
        distance = (
            float(self.edge_remaining_distance_m)
            if self.edge_remaining_distance_m > 1e-9
            else float(network.arc_length((u, v)))
        )
        current = distance / max(float(traffic.mean_speed_mps(now)), 1e-9)
        future = sum(
            network.get_vehicle_edge_time(int(a), int(b), actual=False)
            for a, b in zip(
                self.path_nodes[self.path_index + 1 : -1],
                self.path_nodes[self.path_index + 2 :],
            )
        )
        return float(current + future)

    def end_wait(self) -> None:
        """Finish an explicit wait early after the awaited physical event."""
        if self.wait_until is not None:
            self.schedule_version += 1
        self.wait_until = None
        self.wait_started_at = None

    def set_route(
        self,
        path: List[int],
        network: RoadNetwork,
        now: float,
        target: Optional[int] = None,
    ) -> None:
        """Commit an explicitly constructed directed route."""
        if not self.at_intersection:
            raise ValueError("CAV route can only change at a road node")
        path = [int(node) for node in path]
        if not path:
            raise ValueError("Vehicle route cannot be empty")
        if path[0] != int(self.position):
            raise ValueError(f"Vehicle route must start at current node {self.position}")
        for u, v in zip(path[:-1], path[1:]):
            if not network.G.has_edge(u, v):
                raise ValueError(f"Route contains invalid directed edge: {u}->{v}")
        final_target = int(path[-1] if target is None else target)
        if final_target != int(path[-1]):
            raise ValueError("Vehicle target must be the final route node")
        self.schedule_version += 1
        self.wait_until = None
        self.wait_started_at = None
        self.last_arrival_time = None
        if len(path) == 1:
            self.target_node = None
            self.remaining_time = 0.0
            self.edge_remaining_time = 0.0
            self.edge_total_time = 0.0
            self._reset_edge_realization()
            self.path_nodes = []
            self.path_index = 0
            self.at_intersection = True
            return
        self.target_node = final_target
        self.path_nodes = path
        self.path_index = 0
        deferred_realization = self._defers_edge_realization(network)
        self.remaining_time = sum(
            network.get_vehicle_edge_time(u, v, actual=not deferred_realization)
            for u, v in zip(path[:-1], path[1:])
        )
        self.edge_total_time = network.get_vehicle_edge_time(
            path[0], path[1], actual=not deferred_realization
        )
        self.edge_remaining_time = float(self.edge_total_time)
        self.edge_remaining_distance_m = float(network.arc_length((path[0], path[1])))
        self.edge_realized_speed_mps = 0.0
        self.edge_speed_multiplier = 1.0
        self.edge_stage_index = -1
        self.edge_realized = not deferred_realization
        self.at_intersection = True

    def set_target(self, target: int, network: RoadNetwork, now: float) -> None:
        """Set a new CAV target and expand the directed shortest path to it."""
        target = int(target)
        if target == self.position:
            self.schedule_version += 1
            self.wait_until = None
            self.wait_started_at = None
            self.last_arrival_time = None
            self.target_node = None
            self.at_intersection = True
            self.remaining_time = 0.0
            self.edge_remaining_time = 0.0
            self.edge_total_time = 0.0
            self._reset_edge_realization()
            self.path_nodes = []
            self.path_index = 0
            return

        path = network.get_vehicle_path(self.position, target)
        if not path or len(path) < 2:
            raise ValueError(f"Vehicle target unreachable: {self.position}->{target}")

        self.set_route([int(x) for x in path], network, now, target)

    def note_route_change(self, now: float, target: int, cfg: Dict[str, Any]) -> None:
        now = float(now)
        if self.last_route_change_time is not None:
            self.route_commitment_durations.append(now - self.last_route_change_time)
        self.last_route_change_time = now
        self.route_change_times.append(now)
        self.target_history.append(int(target))
        self.committed_nodes_remaining = max(0, int(cfg.get("minimum_commitment_nodes", 1)))

    def recent_route_change_count(self, now: float, horizon_s: float) -> int:
        threshold = float(now) - float(horizon_s)
        return sum(float(value) >= threshold - 1e-9 for value in self.route_change_times)

    def route_change_allowed(self, now: float, cfg: Dict[str, Any]) -> bool:
        if not bool(cfg.get("enabled", True)):
            return True
        if self.committed_nodes_remaining > 0:
            return False
        cooldown = float(cfg.get("reroute_cooldown_s", 120.0))
        if self.last_route_change_time is not None and float(now) - self.last_route_change_time < cooldown - 1e-9:
            return False
        maximum = int(cfg.get("max_route_changes_in_horizon", 2))
        horizon = float(cfg.get("commitment_horizon_s", 180.0))
        return self.recent_route_change_count(now, horizon) < maximum

    def mean_route_commitment_time(self) -> float:
        if not self.route_commitment_durations:
            return 0.0
        return float(sum(self.route_commitment_durations) / len(self.route_commitment_durations))

    def step(self, dt: float, now: float, network: RoadNetwork) -> bool:
        """Advance the CAV strictly along directed road-network edges.

        Parameters
        ----------
        dt:
            Simulation time increment.
        now:
            Start time of the current simulation increment.
        network:
            Directed road network containing edge travel times.

        Returns
        -------
        bool
            True if the CAV reaches its final target during this time increment.
        """
        self.last_arrival_time = None
        if self.is_operationally_held(now):
            held = min(float(dt), max(0.0, float(self.operation_hold_until) - float(now)))
            self.add_wait(held)
            if held >= float(dt) - 1e-9:
                if float(now) + held >= float(self.operation_hold_until) - 1e-9:
                    self.operation_hold_until = None
                    self.operation_hold_started_at = None
                return False
            self.operation_hold_until = None
            self.operation_hold_started_at = None
            now = float(now) + held
            dt = float(dt) - held
        elif self.operation_hold_until is not None:
            self.operation_hold_until = None
            self.operation_hold_started_at = None
        if (
            self.planned_operation_hold_until is not None
            and float(now) >= float(self.planned_operation_hold_until) - 1e-9
        ):
            self.planned_operation_hold_until = None
        if self.is_waiting(now):
            waited = min(float(dt), max(0.0, float(self.wait_until) - float(now)))
            self.add_wait(waited)
            if waited >= float(dt) - 1e-9:
                if float(now) + waited >= float(self.wait_until) - 1e-9:
                    self.wait_until = None
                    self.wait_started_at = None
                return False
            self.end_wait()
            now = float(now) + waited
            dt = float(dt) - waited
        if self.target_node is None or not self.path_nodes:
            self.at_intersection = True
            return False

        self.ensure_edge_realization(now, network)

        remaining_dt = float(dt)
        elapsed = 0.0
        arrived_target = False

        while remaining_dt > 1e-9 and self.target_node is not None:
            if self.path_index >= len(self.path_nodes) - 1:
                # Defensive cleanup for a completed path.
                self.target_node = None
                self.remaining_time = 0.0
                self.edge_remaining_time = 0.0
                self.edge_total_time = 0.0
                self._reset_edge_realization()
                self.path_nodes = []
                self.path_index = 0
                self.at_intersection = True
                arrived_target = True
                self.last_arrival_time = now + elapsed
                break

            u = int(self.path_nodes[self.path_index])
            v = int(self.path_nodes[self.path_index + 1])
            if not network.G.has_edge(u, v):
                raise ValueError(f"CAV attempted invalid directed edge: {u}->{v}")

            self.ensure_edge_realization(now + elapsed, network)
            if self.edge_remaining_time <= 1e-9:
                self.edge_total_time = network.get_vehicle_edge_time(u, v, actual=True)
                self.edge_remaining_time = float(self.edge_total_time)

            edge_time = float(self.edge_total_time)
            move = min(remaining_dt, self.edge_remaining_time)
            if move > 1e-9:
                self.at_intersection = False
                if self.edge_realized and self.edge_realized_speed_mps > 1e-12:
                    distance = min(
                        float(self.edge_remaining_distance_m),
                        float(self.edge_realized_speed_mps) * float(move),
                    )
                    self.edge_remaining_distance_m = max(
                        0.0, float(self.edge_remaining_distance_m) - distance
                    )
                    self.distance_traveled_m += distance
                elif edge_time > 1e-9:
                    self.distance_traveled_m += float(network.arc_length((u, v))) * (float(move) / edge_time)
            self.edge_remaining_time -= move
            self.remaining_time = max(0.0, self.remaining_time - move)
            self.distance_time += move
            remaining_dt -= move
            elapsed += move

            # Still on the same edge.
            if self.edge_remaining_time > 1e-9:
                break

            # The CAV has reached the next road-network node.
            arrival_time = now + elapsed
            self.position = v
            self.at_intersection = True
            self.route_edge_log.append((arrival_time, u, v))
            self.route_log.append((arrival_time, v))
            self.path_index += 1
            if self.committed_nodes_remaining > 0:
                self.committed_nodes_remaining -= 1

            if self.position == self.target_node:
                self.target_node = None
                self.remaining_time = 0.0
                self.edge_remaining_time = 0.0
                self.edge_total_time = 0.0
                self._reset_edge_realization()
                self.path_nodes = []
                self.path_index = 0
                self.at_intersection = True
                arrived_target = True
                self.last_arrival_time = arrival_time
                break

            # Prepare the next directed edge on the path.
            next_u = int(self.path_nodes[self.path_index])
            next_v = int(self.path_nodes[self.path_index + 1])
            if not network.G.has_edge(next_u, next_v):
                raise ValueError(f"CAV attempted invalid directed edge: {next_u}->{next_v}")
            deferred_realization = self._defers_edge_realization(network)
            self.edge_total_time = network.get_vehicle_edge_time(
                next_u, next_v, actual=not deferred_realization
            )
            self.edge_remaining_time = float(self.edge_total_time)
            self.edge_remaining_distance_m = float(network.arc_length((next_u, next_v)))
            self.edge_realized_speed_mps = 0.0
            self.edge_speed_multiplier = 1.0
            self.edge_stage_index = -1
            self.edge_realized = not deferred_realization

        return arrived_target

    def refresh_actual_remaining_time(
        self,
        network: RoadNetwork,
        *,
        now: Optional[float] = None,
        resample_current: bool = False,
    ) -> None:
        """Refresh route timing; staged traffic may resample a mid-edge remainder."""
        if self.target_node is None or not self.path_nodes or self.path_index >= len(self.path_nodes) - 1:
            return
        if self._defers_edge_realization(network):
            current_u = int(self.path_nodes[self.path_index])
            current_v = int(self.path_nodes[self.path_index + 1])
            if self.edge_realized and not self.at_intersection and resample_current:
                self.ensure_edge_realization(float(0.0 if now is None else now), network, force=True)
            elif not self.edge_realized:
                self.edge_total_time = network.get_vehicle_edge_time(
                    current_u, current_v, actual=False
                )
                self.edge_remaining_time = float(self.edge_total_time)
            future_time = sum(
                network.get_vehicle_edge_time(int(u), int(v), actual=False)
                for u, v in zip(
                    self.path_nodes[self.path_index + 1 : -1],
                    self.path_nodes[self.path_index + 2 :],
                )
            )
            self.remaining_time = float(self.edge_remaining_time + future_time)
            return
        current_u = int(self.path_nodes[self.path_index])
        current_v = int(self.path_nodes[self.path_index + 1])
        if self.at_intersection:
            self.edge_total_time = network.get_vehicle_edge_time(current_u, current_v, actual=True)
            self.edge_remaining_time = float(self.edge_total_time)
        future_time = sum(
            network.get_vehicle_edge_time(int(u), int(v), actual=True)
            for u, v in zip(
                self.path_nodes[self.path_index + 1 : -1],
                self.path_nodes[self.path_index + 2 :],
            )
        )
        self.remaining_time = float(self.edge_remaining_time + future_time)

    def add_wait(self, dt: float) -> None:
        self.wait_time += dt
