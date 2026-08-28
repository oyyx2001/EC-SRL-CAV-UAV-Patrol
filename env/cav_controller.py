from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import math
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import Counter


GROUND_ACTIONS = ("keep_route", "wait", "move_to", "reroute")


def is_immediate_route_reversal(
    planned_route: Sequence[int],
    route_edge_log: Sequence[Tuple[float, int, int]],
) -> bool:
    """Return whether the next CAV edge immediately reverses the last edge.

    ``route_edge_log`` stores ``(time, from_node, to_node)``.  When the CAV is
    currently at ``to_node``, selecting ``from_node`` as the next route node is
    the exact directed-edge reversal that the stability guard is meant to
    detect.
    """

    if len(planned_route) <= 1 or not route_edge_log:
        return False
    previous_node = int(route_edge_log[-1][1])
    return int(planned_route[1]) == previous_node


@dataclass
class GroundCandidate:
    """A side-effect-free rolling-horizon CAV action proposal."""

    candidate_id: int
    action_type: str
    target_node: Optional[int] = None
    waypoint_node: Optional[int] = None
    planned_route: List[int] = field(default_factory=list)
    affected_recovery_ids: List[str] = field(default_factory=list)
    recovery_updates: Dict[str, int] = field(default_factory=dict)
    predicted_cav_arrival_time: float = math.inf
    predicted_recovery_schedule: List[Dict[str, Any]] = field(default_factory=list)
    predicted_uav_wait: float = 0.0
    predicted_cav_wait: float = 0.0
    predicted_uav_return_time: float = 0.0
    predicted_uav_return_distance: float = 0.0
    predicted_uav_return_energy: float = 0.0
    predicted_hover_energy: float = 0.0
    cav_travel_time: float = 0.0
    cav_travel_distance: float = 0.0
    detour_time: float = 0.0
    detour_distance: float = 0.0
    detour_ratio: float = 0.0
    task_coverage_distance: float = 0.0
    task_coverage_improvement: float = 0.0
    recovery_improvement: float = 0.0
    min_recovery_slack: float = math.inf
    max_queue_wait: float = 0.0
    route_switch_penalty: float = 0.0
    route_prefix_overlap: float = 1.0
    reverse_route_flag: bool = False
    route_changed: bool = False
    emergency_override: bool = False
    wait_until: Optional[float] = None
    feasible: bool = True
    infeasible_reason: str = "ok"
    score: float = math.inf
    computation_time_ms: float = 0.0
    state_fingerprint: Tuple[Any, ...] = field(default_factory=tuple, repr=False)

    def to_record(self) -> Dict[str, Any]:
        record = asdict(self)
        record["planned_route"] = "|".join(str(node) for node in self.planned_route)
        record["affected_recovery_ids"] = "|".join(self.affected_recovery_ids)
        record["recovery_updates"] = json.dumps(self.recovery_updates, sort_keys=True)
        record["predicted_recovery_schedule"] = json.dumps(
            self.predicted_recovery_schedule, sort_keys=True
        )
        record.pop("state_fingerprint", None)
        return record


class ActiveCAVController:
    """Deterministic rule-based active CAV routing controller.

    The controller never changes the UAV-dispatch action space.  Candidate
    evaluation is pure; only :meth:`commit_cav_action` mutates simulator state.
    """

    def __init__(self, simulator: Any):
        self.sim = simulator
        self.cfg = dict(simulator.cfg.get("cav_control", {}))
        self.commit_cfg = dict(simulator.cfg.get("route_commitment", {}))
        self.cost_cfg = dict(simulator.cfg.get("ground_cost", {}))
        self.mode = str(self.cfg.get("mode", "passive")).lower()
        raw_actions = self.cfg.get("action_set", ["keep_route"])
        self.action_set = tuple(
            action for action in GROUND_ACTIONS if action in {str(x) for x in raw_actions}
        )
        if "keep_route" not in self.action_set:
            self.action_set = ("keep_route", *self.action_set)
        self.last_decision_time: Optional[float] = None
        self.last_selected_action = "keep_route"
        self.action_log: List[Dict[str, Any]] = []
        self.candidate_log: List[Dict[str, Any]] = []
        self.route_change_log: List[Dict[str, Any]] = []
        self.recovery_schedule_log: List[Dict[str, Any]] = []
        self.total_committed_detour_distance = 0.0
        self.pending_realized_gain: Dict[str, Dict[str, float]] = {}
        self.realized_gains: List[float] = []
        self._candidate_serial = 0
        # These counters are intentionally controller-owned: they distinguish
        # state-advance recovery work from initial selected-sortie planning.
        self.replanning_stats: Counter = Counter()
        self._replan_epoch_time: Optional[float] = None
        self._replans_by_drone: Counter = Counter()
        self._replans_in_epoch = 0
        self._last_recovery_plan_failure_reason: Optional[str] = None

    @property
    def active(self) -> bool:
        return self.mode == "active_rule_based" and len(self.action_set) > 1

    @property
    def candidate_generation_enabled(self) -> bool:
        """Whether non-keep proposals may be generated.

        ``joint_plan`` deliberately enables the stage-three proposal machinery
        without enabling its independent rule-based ``decide`` path.
        """

        return self.mode in {"active_rule_based", "joint_plan"} and len(self.action_set) > 1

    @property
    def queue_aware_recovery(self) -> bool:
        """Whether queue state may enter proposal generation or scoring.

        ``queue_blind`` retains the physical FIFO queue and its safety checks,
        but the policy-side candidate generator must not condition its proposed
        actions, candidate scores, or feature values on that queue.
        """

        return bool(self.sim.cfg.get("ablation", {}).get("queue_aware_recovery", True))

    def state_fingerprint(self) -> Tuple[Any, ...]:
        vehicle = self.sim.vehicle
        queue = tuple(
            (
                str(entry.get("sortie_id", entry.get("drone_id", ""))),
                int(entry["recovery_node"]),
                str(entry.get("status", "planned")),
                bool(entry.get("locked", False)),
                round(float(entry.get("eta", 0.0)), 9),
            )
            for entry in self.sim.recovery_queue
        )
        drones = tuple(
            (
                drone.id,
                drone.status,
                int(drone.position),
                round(float(drone.battery), 9),
                int(drone.schedule_version),
            )
            for drone in self.sim.drones
        )
        return (
            round(float(self.sim.now), 9),
            int(vehicle.position),
            vehicle.target_node,
            int(vehicle.schedule_version),
            int(self.sim.recovery_schedule_version),
            queue,
            drones,
        )

    def _next_id(self) -> int:
        value = self._candidate_serial
        self._candidate_serial += 1
        return value

    def _path_time(self, path: Sequence[int]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            if not self.sim.network.G.has_edge(int(u), int(v)):
                return math.inf
            total += float(self.sim.network.G[int(u)][int(v)]["vehicle_time_s"])
        return total

    def _path_distance(self, path: Sequence[int]) -> float:
        if len(path) < 2:
            return 0.0
        return float(
            sum(
                self.sim.network.arc_length((int(u), int(v)))
                for u, v in zip(path[:-1], path[1:])
            )
        )

    @staticmethod
    def _join_paths(*paths: Sequence[int]) -> List[int]:
        joined: List[int] = []
        for path in paths:
            for node in path:
                node = int(node)
                if not joined or joined[-1] != node:
                    joined.append(node)
        return joined

    def _shortest_path(self, start: int, end: int) -> List[int]:
        return [int(x) for x in self.sim.network.get_vehicle_path(int(start), int(end))]

    def _existing_route(self) -> List[int]:
        vehicle = self.sim.vehicle
        if vehicle.path_nodes and vehicle.path_index < len(vehicle.path_nodes):
            route = [int(x) for x in vehicle.path_nodes[vehicle.path_index :]]
            if route and route[0] != int(vehicle.position):
                route.insert(0, int(vehicle.position))
            return route
        return [int(vehicle.position)]

    def _locked_prefix(self, queue: Optional[Sequence[Dict[str, Any]]] = None) -> int:
        return self.sim.recovery_scheduler.locked_prefix_length(
            self.sim.recovery_queue if queue is None else queue
        )

    def _entry_id(self, entry: Dict[str, Any]) -> str:
        return str(entry.get("sortie_id", entry.get("drone_id", entry.get("uav_id", ""))))

    def _entry_is_flexible(self, index: int, entry: Dict[str, Any]) -> bool:
        if not bool(self.cfg.get("allow_flexible_rendezvous_update", True)):
            return False
        if index < self._locked_prefix():
            return False
        if bool(entry.get("locked", False)):
            return False
        if str(entry.get("status", "planned")) in {"returning", "recovering", "completed"}:
            return False
        try:
            drone = self.sim.get_drone(str(entry.get("drone_id", entry.get("uav_id"))))
        except KeyError:
            return False
        if drone.status == "flying" and drone.sortie is not None:
            final_index = max(0, len(drone.sortie.segments or []) - 1)
            if drone.segment_index >= final_index:
                return False
        return True

    def _task_weight(self, task: Any) -> float:
        urgency = 1.0 / max(1.0, float(task.deadline) - float(self.sim.now))
        return max(1e-9, float(task.priority) * urgency)

    def _road_distance(self, start: int, end: int) -> float:
        return self._path_distance(self._shortest_path(start, end))

    def _task_coverage_distance(self, node: int) -> float:
        tasks = self.sim.task_pool.available_tasks(self.sim.now)
        if not tasks:
            return 0.0
        numerator = 0.0
        denominator = 0.0
        for task in tasks:
            weight = self._task_weight(task)
            distance = self._road_distance(int(node), int(task.start_node))
            if not math.isfinite(distance):
                continue
            numerator += weight * distance
            denominator += weight
        return numerator / denominator if denominator > 0.0 else 0.0

    def _next_wait_end(self, next_event_time: Optional[float]) -> float:
        now = float(self.sim.now)
        upper = now + float(self.cfg.get("max_wait_s", 120.0))
        future_times = []
        if self.queue_aware_recovery:
            future_times.extend(
                float(entry.get("eta", math.inf))
                for entry in self.sim.recovery_queue
                if float(entry.get("eta", math.inf)) > now + 1e-9
            )
        if next_event_time is not None and float(next_event_time) > now + 1e-9:
            future_times.append(float(next_event_time))
        future_times.append(upper)
        return min(future_times)

    def _rendezvous_nodes(self, index: int, entry: Dict[str, Any]) -> List[int]:
        if not self._entry_is_flexible(index, entry):
            return []
        node = int(entry["recovery_node"])
        seeds = [node]
        try:
            drone = self.sim.get_drone(str(entry.get("drone_id", entry.get("uav_id"))))
        except KeyError:
            return seeds
        if drone.sortie is not None and drone.sortie.segments:
            final_fly = drone.sortie.segments[-1]
            seeds.append(int(final_fly.get("from", drone.position)))
        seeds.extend(int(v) for _, v in self.sim.network.G.out_edges(node))
        route = self._shortest_path(int(self.sim.vehicle.position), node)
        if len(route) > 2:
            seeds.extend(route[1:-1])
        unique: List[int] = []
        horizon = float(self.cfg.get("prediction_horizon_s", 300.0))
        for candidate_node in seeds:
            if candidate_node in unique:
                continue
            travel = self.sim.network.get_vehicle_time(self.sim.vehicle.position, candidate_node)
            if not math.isfinite(travel) or travel > horizon + 1e-9:
                continue
            prediction = self._predict_entry_update(entry, candidate_node)
            if prediction is None:
                continue
            unique.append(int(candidate_node))
        limit = int(self.cfg.get("max_rendezvous_nodes_per_uav", 3))
        unique.sort(
            key=lambda candidate_node: (
                abs(
                    float(self.sim.now)
                    + self.sim.network.get_vehicle_time(self.sim.vehicle.position, candidate_node)
                    - float(entry.get("eta", self.sim.now))
                ),
                self.sim.network.get_drone_time(int(drone.position), candidate_node),
                candidate_node,
            )
        )
        return unique[:limit]

    def _candidate_node_specs(self) -> List[Tuple[int, Optional[str], str]]:
        specs: Dict[int, Tuple[Optional[str], str, int]] = {}

        def add(node: int, recovery_id: Optional[str], source: str, priority: int) -> None:
            node = int(node)
            old = specs.get(node)
            value = (recovery_id, source, priority)
            if old is None or priority < old[2]:
                specs[node] = value

        if self.queue_aware_recovery:
            for index, entry in enumerate(self.sim.recovery_queue):
                recovery_id = self._entry_id(entry)
                add(int(entry["recovery_node"]), recovery_id if self._entry_is_flexible(index, entry) else None, "recovery", 0)
                for node in self._rendezvous_nodes(index, entry):
                    add(node, recovery_id, "rendezvous", 1)
        current = int(self.sim.vehicle.position)
        for _, node in sorted(self.sim.network.G.out_edges(current)):
            add(int(node), None, "adjacent", 3)
        existing = self._existing_route()
        for node in existing[1:3]:
            add(node, None, "current_route", 2)
        urgent_tasks = sorted(
            self.sim.task_pool.available_tasks(self.sim.now),
            key=lambda task: (-self._task_weight(task), int(task.id)),
        )[:3]
        for task in urgent_tasks:
            add(int(task.start_node), None, "task", 4)
            add(int(task.end_node), None, "task", 4)
        specs.pop(current, None)
        ranked = sorted(
            specs.items(),
            key=lambda item: (
                item[1][2],
                self.sim.network.get_vehicle_time(current, item[0]),
                item[0],
            ),
        )
        limit = int(self.cfg.get("max_ground_candidates", 8))
        return [(node, meta[0], meta[1]) for node, meta in ranked[:limit]]

    def generate_ground_candidates(
        self, next_event_time: Optional[float] = None
    ) -> List[GroundCandidate]:
        self._candidate_serial = 0
        current = int(self.sim.vehicle.position)
        keep = GroundCandidate(self._next_id(), "keep_route", target_node=self.sim.vehicle.target_node)
        keep = self.evaluate_cav_candidate(keep)
        candidates = [keep]
        if not self.candidate_generation_enabled or not self.sim.vehicle.at_intersection:
            return candidates
        if "wait" in self.action_set:
            wait_updates: Dict[str, int] = {}
            wait_affected: List[str] = []
            flexible = []
            if self.queue_aware_recovery:
                flexible = [
                    (index, entry)
                    for index, entry in enumerate(self.sim.recovery_queue)
                    if self._entry_is_flexible(index, entry)
                    and self._predict_entry_update(entry, current) is not None
                ]
            if flexible:
                _, entry = min(
                    flexible,
                    key=lambda item: (float(item[1].get("eta", math.inf)), self._entry_id(item[1])),
                )
                recovery_id = self._entry_id(entry)
                wait_updates[recovery_id] = current
                wait_affected.append(recovery_id)
            wait = GroundCandidate(
                self._next_id(),
                "wait",
                target_node=current,
                wait_until=self._next_wait_end(next_event_time),
                affected_recovery_ids=wait_affected,
                recovery_updates=wait_updates,
            )
            candidates.append(self.evaluate_cav_candidate(wait, keep_reference=keep))
        for node, recovery_id, _source in self._candidate_node_specs():
            updates = {recovery_id: int(node)} if recovery_id is not None else {}
            affected = [recovery_id] if recovery_id is not None else []
            if "move_to" in self.action_set:
                move = GroundCandidate(
                    self._next_id(),
                    "move_to",
                    target_node=int(node),
                    affected_recovery_ids=affected,
                    recovery_updates=updates,
                )
                candidates.append(self.evaluate_cav_candidate(move, keep_reference=keep))
            if "reroute" in self.action_set:
                reroute = GroundCandidate(
                    self._next_id(),
                    "reroute",
                    target_node=int(node),
                    waypoint_node=int(node),
                    affected_recovery_ids=affected,
                    recovery_updates=updates,
                )
                candidates.append(self.evaluate_cav_candidate(reroute, keep_reference=keep))
        return candidates

    def _uses_voxel_recovery_planner(self) -> bool:
        planning = self.sim.cfg.get("uav_path_planning", {})
        return bool(planning.get("enabled", False)) and (
            str(planning.get("planner_type", "")).lower() == "voxel_astar_3d"
        )

    def _voxel_context(self):
        """Return the single per-episode voxel context shared with sorties."""
        if not self._uses_voxel_recovery_planner():
            return None
        planning = self.sim.cfg.get("uav_path_planning", {})
        generator = self.sim.sortie_generator
        if generator is None:
            return None
        try:
            import sys
            from pathlib import Path

            module_root = str(Path(__file__).resolve().parents[1] / "truck_uav_modules")
            if module_root not in sys.path:
                sys.path.insert(0, module_root)
            from uav_path_planner import VoxelPlannerContext

            region = planning.get("region", self.sim.cfg.get("network", {}).get("uav_path_region"))
            if not region:
                return None
            if generator._voxel_context is None:
                generator._voxel_context = VoxelPlannerContext(str(region), planning)
            return generator._voxel_context
        except Exception:
            return None

    def _replan_limits(self) -> Tuple[int, int]:
        cfg=self.sim.cfg.get("uav_path_planning", {}).get("recovery_replanning", {})
        return max(1, int(cfg.get("max_exact_replans_per_uav_per_decision", 1))), max(1, int(cfg.get("max_exact_replans_system_per_decision", 4)))

    def _reserve_exact_replan(self, drone_id: str) -> bool:
        now=float(self.sim.now)
        if self._replan_epoch_time is None or abs(self._replan_epoch_time-now) > 1e-9:
            self._replan_epoch_time=now
            self._replans_by_drone.clear()
            self._replans_in_epoch=0
        per_uav, system=self._replan_limits()
        if self._replans_by_drone[str(drone_id)] >= per_uav or self._replans_in_epoch >= system:
            self.replanning_stats["exact_replan_limit_skip_count"] += 1
            return False
        self._replans_by_drone[str(drone_id)] += 1
        self._replans_in_epoch += 1
        self.replanning_stats["exact_replan_request_count"] += 1
        return True

    def _flight_state(self) -> Dict[str, float]:
        generator=self.sim.sortie_generator
        params=generator._flight_parameters()
        return {
            "cruise_speed": float(params["cruise_speed"]), "climb_speed": float(params["up_speed"]),
            "descent_speed": float(params["down_speed"]), "cruise_power": float(params["cruise_power"]),
            "climb_power": float(params["up_power"]), "descent_power": float(params["down_power"]),
        }

    def _recovery_voxel(self, context: Any, node: int) -> Tuple[int, int, int]:
        planning=self.sim.cfg.get("uav_path_planning", {})
        altitude=float(planning.get("default_task_observation_altitude_m", 20.0))
        xy=self.sim.network.node_xy(int(node))
        return tuple(context.free_near(context.point((xy[0], xy[1], altitude))))

    def _voxel_leg_from_path(self, context: Any, path: Sequence[Tuple[int, int, int]]) -> Optional[Dict[str, Any]]:
        try:
            metrics=context.path_metrics(list(path), self._flight_state())
        except (TypeError, ValueError):
            return None
        return {
            "duration_s": float(metrics["flight_time_s"]), "energy": float(metrics["energy_consumption"]),
            "distance_m": float(metrics["total_distance_m"]), "waypoints_xyz": list(metrics["waypoints_xyz"]),
            "voxel_path": [tuple(map(int, point)) for point in path],
            "segment_result": {
                "start_voxel": tuple(map(int, path[0])), "goal_voxel": tuple(map(int, path[-1])),
                "distance_m": float(metrics["total_distance_m"]), "planning_time_s": 0.0,
                "expanded_nodes": 0, "cache_hit": True, "astar_status": "path_reuse",
                "astar_success": True, "path_voxel_count": len(path), "voxel_path": list(path),
            },
        }

    def _return_path_suffix(self, drone: Any, final_index: int, final: Dict[str, Any]) -> Tuple[List[Tuple[int, int, int]], bool]:
        """Return the unflown voxel suffix when a UAV is already on its return leg."""
        path=[tuple(map(int, point)) for point in final.get("path_voxel_path", final.get("voxel_path", []))]
        if len(path) < 1 or int(drone.segment_index) != int(final_index):
            return path, False
        duration=max(float(final.get("duration", 0.0)), 1e-9)
        remaining=max(0.0, min(duration, float(drone.segment_remaining)))
        progress=1.0-remaining/duration
        index=min(len(path)-1, max(0, int(math.ceil(progress*(len(path)-1)))))
        return path[index:], True

    def _plan_voxel_recovery_leg(
        self, origin: int, recovery_node: int, *, start_voxel: Optional[Tuple[int, int, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run one bounded exact recovery search through the shared voxel LRU."""
        self._last_recovery_plan_failure_reason=None
        context=self._voxel_context()
        if context is None:
            self._last_recovery_plan_failure_reason="voxel_context_unavailable"
            return None
        try:
            import sys
            from pathlib import Path
            module_root=str(Path(__file__).resolve().parents[1] / "truck_uav_modules")
            if module_root not in sys.path:
                sys.path.insert(0, module_root)
            from uav_path_planner import plan_selected_sortie_3d

            planning=self.sim.cfg.get("uav_path_planning", {})
            altitude=float(planning.get("default_task_observation_altitude_m", 20.0))
            start_xy=self.sim.network.node_xy(int(origin))
            goal_xy=self.sim.network.node_xy(int(recovery_node))
            precise=plan_selected_sortie_3d(
                (start_xy[0], start_xy[1], altitude), [], (goal_xy[0], goal_xy[1], altitude),
                self._flight_state(), context, source="inflight_recovery_replan",
                budget_profile="recovery_replan", start_voxel_override=start_voxel,
            )
            if not precise.feasible:
                self._last_recovery_plan_failure_reason=str(precise.failure_reason or precise.termination_reason)
                return None
            self.replanning_stats["exact_replan_success_count"] += 1
            return {
                "duration_s": float(precise.flight_time_s), "energy": float(precise.energy_consumption),
                "distance_m": float(precise.total_distance_m), "waypoints_xyz": list(precise.waypoints_xyz),
                "voxel_path": list(precise.voxel_path),
                "segment_result": dict((precise.segment_results or [{}])[0]),
            }
        except Exception as exc:
            self._last_recovery_plan_failure_reason=f"internal_error:{type(exc).__name__}"
            return None

    def _predict_entry_update(
        self, entry: Dict[str, Any], recovery_node: int
    ) -> Optional[Dict[str, float]]:
        try:
            drone = self.sim.get_drone(str(entry.get("drone_id", entry.get("uav_id"))))
        except KeyError:
            return None
        if drone.sortie is None or not drone.sortie.segments:
            return {
                "eta": float(entry.get("eta", self.sim.now)),
                "battery_at_eta": float(entry.get("battery_at_eta", drone.battery)),
                "return_time": self.sim.network.get_drone_time(int(drone.position), int(recovery_node)),
            }
        segments = drone.sortie.segments
        final_index = next(
            (i for i in range(len(segments) - 1, -1, -1)
            if str(segments[i].get("type")) in {"fly", "horizontal_flight"}),
            len(segments) - 1,
        )
        remaining_time = 0.0
        remaining_energy = 0.0
        return_time = 0.0
        for index in range(drone.segment_index, len(segments)):
            segment = segments[index]
            duration = (
                float(drone.segment_remaining)
                if index == drone.segment_index and drone.segment_remaining > 0.0
                else float(segment.get("duration", 0.0))
            )
            if index == final_index and str(segment.get("type")) in {"fly", "horizontal_flight"}:
                origin = int(segment.get("from", drone.position))
                # This is candidate scoring, not a committed path.  Retain a
                # cheap network-time estimate so ground-plan enumeration stays
                # bounded; the selected recovery update is always replaced by
                # ``_plan_voxel_recovery_leg`` before state is committed.
                duration = self.sim.network.get_drone_time(origin, int(recovery_node))
                return_time = duration
            rate = float(
                self.sim.cfg["agents"].get(
                    "drone_service_power" if segment.get("type") == "service" else "drone_fly_power",
                    0.85 if segment.get("type") == "service" else 1.0,
                )
            )
            remaining_time += duration
            remaining_energy += duration * rate
        if not any(str(segment.get("type")) in {"fly", "horizontal_flight"} for segment in segments):
            origin = int(segments[-1].get("to", drone.position))
            return_time = self.sim.network.get_drone_time(origin, int(recovery_node))
            return_energy = return_time * float(self.sim.cfg["agents"].get("drone_fly_power", 1.0))
            remaining_time += return_time
            remaining_energy += return_energy
        reserve = float(entry.get("battery_reserve", 0.0))
        battery_at_eta = float(drone.battery) - remaining_energy
        if battery_at_eta < reserve - 1e-9:
            return None
        return {
            "eta": float(self.sim.now + remaining_time),
            "battery_at_eta": battery_at_eta,
            "return_time": return_time,
        }

    def _trial_queue(self, candidate: GroundCandidate) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        # This is the *policy* queue.  Physical FIFO validation is carried out
        # later by the recovery scheduler when a selected plan is committed.
        queue = deepcopy(self.sim.recovery_queue) if self.queue_aware_recovery else []
        locked = self._locked_prefix(queue)
        for index, entry in enumerate(queue):
            entry_id = self._entry_id(entry)
            if entry_id not in candidate.recovery_updates:
                continue
            if index < locked or bool(entry.get("locked", False)):
                return queue, "locked_recovery_modified"
            prediction = self._predict_entry_update(entry, candidate.recovery_updates[entry_id])
            if prediction is None:
                return queue, "battery_reserve"
            entry["recovery_node"] = int(candidate.recovery_updates[entry_id])
            entry["eta"] = float(prediction["eta"])
            entry["battery_at_eta"] = float(prediction["battery_at_eta"])
        return queue, None

    def _route_to_first_recovery(
        self, candidate: GroundCandidate, queue: Sequence[Dict[str, Any]]
    ) -> Tuple[List[int], float]:
        current = int(self.sim.vehicle.position)
        wait_duration = max(0.0, float(candidate.wait_until or self.sim.now) - float(self.sim.now))
        first_node = int(queue[0]["recovery_node"]) if queue else None
        if candidate.action_type == "keep_route":
            route = self._existing_route()
        elif candidate.action_type == "wait":
            route = [current]
        elif candidate.action_type == "move_to":
            route = self._shortest_path(current, int(candidate.target_node))
        else:
            to_waypoint = self._shortest_path(current, int(candidate.waypoint_node))
            if first_node is None:
                route = to_waypoint
            else:
                route = self._join_paths(to_waypoint, self._shortest_path(int(candidate.waypoint_node), first_node))
        if not route:
            return [], math.inf
        if first_node is not None and route[-1] != first_node:
            route = self._join_paths(route, self._shortest_path(route[-1], first_node))
        return route, wait_duration

    def _return_metrics(self, queue: Sequence[Dict[str, Any]]) -> Tuple[float, float, float]:
        speed = float(self.sim.cfg["network"].get("drone_speed_mps", 14.0))
        fly_power = float(self.sim.cfg["agents"].get("drone_fly_power", 1.0))
        total_time = 0.0
        for entry in queue:
            try:
                drone = self.sim.get_drone(str(entry.get("drone_id", entry.get("uav_id"))))
            except KeyError:
                continue
            prediction = self._predict_entry_update(entry, int(entry["recovery_node"]))
            if prediction is not None:
                total_time += float(prediction["return_time"])
            elif drone.sortie is not None and drone.sortie.segments:
                total_time += float(drone.sortie.segments[-1].get("duration", 0.0))
        return total_time, total_time * speed, total_time * fly_power

    def evaluate_cav_candidate(
        self,
        candidate: GroundCandidate,
        state: Optional[Any] = None,
        commit: bool = False,
        keep_reference: Optional[GroundCandidate] = None,
    ) -> GroundCandidate:
        """Evaluate a candidate without changing simulator or RNG state."""

        del state
        started = perf_counter()
        candidate.state_fingerprint = self.state_fingerprint()
        vehicle = self.sim.vehicle
        now = float(self.sim.now)
        current = int(vehicle.position)
        candidate.route_changed = candidate.action_type in {"move_to", "reroute"}
        if candidate.action_type not in GROUND_ACTIONS:
            candidate.feasible, candidate.infeasible_reason = False, "unknown_action"
            return candidate
        if candidate.action_type != "keep_route" and not vehicle.at_intersection:
            candidate.feasible, candidate.infeasible_reason = False, "cav_on_edge"
            return candidate

        queue, queue_error = self._trial_queue(candidate)
        if queue_error is not None:
            candidate.feasible, candidate.infeasible_reason = False, queue_error
            return candidate
        route, wait_duration = self._route_to_first_recovery(candidate, queue)
        candidate.planned_route = route
        route_time = self._path_time(route)
        route_distance = self._path_distance(route)
        if not route or not math.isfinite(route_time):
            candidate.feasible, candidate.infeasible_reason = False, "vehicle_unreachable"
            return candidate

        target_route = route
        if candidate.target_node is not None and int(candidate.target_node) in route:
            target_index = route.index(int(candidate.target_node))
            target_route = route[: target_index + 1]
        candidate.predicted_cav_arrival_time = now + wait_duration + self._path_time(target_route)
        candidate.cav_travel_time = wait_duration + route_time
        candidate.cav_travel_distance = route_distance
        first_arrival = now + wait_duration + route_time
        schedule_result = self.sim.recovery_scheduler.simulate_ground_plan(queue, first_arrival)
        candidate.predicted_recovery_schedule = deepcopy(schedule_result.get("schedule", []))
        candidate.predicted_uav_wait = float(
            sum(row.get("predicted_uav_wait", 0.0) for row in candidate.predicted_recovery_schedule)
        )
        candidate.predicted_cav_wait = float(
            sum(row.get("predicted_cav_wait", 0.0) for row in candidate.predicted_recovery_schedule)
        )
        candidate.predicted_hover_energy = float(
            sum(row.get("predicted_hover_energy", 0.0) for row in candidate.predicted_recovery_schedule)
        )
        candidate.max_queue_wait = max(
            (float(row.get("predicted_uav_wait", 0.0)) for row in candidate.predicted_recovery_schedule),
            default=0.0,
        )
        candidate.min_recovery_slack = min(
            (
                (
                    float(row.get("battery_after_wait", math.inf))
                    - float(row.get("battery_reserve", 0.0))
                )
                / max(float(row.get("hover_power", self.sim.cfg["agents"].get("drone_hover_power", 0.65))), 1e-9)
                for row in candidate.predicted_recovery_schedule
            ),
            default=math.inf,
        )
        (
            candidate.predicted_uav_return_time,
            candidate.predicted_uav_return_distance,
            candidate.predicted_uav_return_energy,
        ) = self._return_metrics(queue)
        coverage_node = int(candidate.target_node if candidate.target_node is not None else current)
        candidate.task_coverage_distance = self._task_coverage_distance(coverage_node)
        keep_node = int(vehicle.target_node if vehicle.target_node is not None else current)
        candidate.task_coverage_improvement = (
            self._task_coverage_distance(keep_node) - candidate.task_coverage_distance
        )

        if keep_reference is not None:
            candidate.detour_time = max(0.0, candidate.cav_travel_time - keep_reference.cav_travel_time)
            candidate.detour_distance = max(
                0.0, candidate.cav_travel_distance - keep_reference.cav_travel_distance
            )
            candidate.detour_ratio = candidate.detour_distance / max(
                keep_reference.cav_travel_distance, 1e-9
            )
            candidate.recovery_improvement = (
                keep_reference.predicted_uav_wait - candidate.predicted_uav_wait
            )
            current_route = self._existing_route()
            overlap = 0
            for left, right in zip(current_route, route):
                if left != right:
                    break
                overlap += 1
            candidate.route_prefix_overlap = overlap / max(1, len(current_route))

        candidate.reverse_route_flag = is_immediate_route_reversal(
            route,
            vehicle.route_edge_log,
        )
        recent_changes = vehicle.recent_route_change_count(
            now, float(self.commit_cfg.get("commitment_horizon_s", 180.0))
        )
        # A direct move to an intermediate waypoint requires a second route
        # commitment to resume the queued recovery.  A finite-horizon reroute
        # commits waypoint -> recovery in one stable plan.
        followup_switch = bool(
            candidate.action_type == "move_to"
            and queue
            and candidate.target_node is not None
            and int(candidate.target_node) != int(queue[0]["recovery_node"])
        )
        candidate.route_switch_penalty = float(candidate.route_changed) + float(
            candidate.reverse_route_flag
        ) + recent_changes + float(followup_switch)
        horizon = max(1.0, float(self.cfg.get("prediction_horizon_s", 300.0)))
        distance_reference = float(
            self.cfg.get(
                "task_distance_reference_m",
                max(
                    float(self.sim.cfg["network"].get("width_m", 1.0)),
                    float(self.sim.cfg["network"].get("height_m", 1.0)),
                    1.0,
                ),
            )
        )
        recovery_wait_weight = float(self.cost_cfg.get("weight_recovery_wait", 1.0)) if self.queue_aware_recovery else 0.0
        cav_wait_weight = float(self.cost_cfg.get("weight_cav_wait", 0.25)) if self.queue_aware_recovery else 0.0
        candidate.score = float(
            recovery_wait_weight
            * candidate.predicted_uav_wait
            / horizon
            + float(self.cost_cfg.get("weight_uav_return", 0.5))
            * candidate.predicted_uav_return_time
            / horizon
            + cav_wait_weight
            * candidate.predicted_cav_wait
            / horizon
            + float(self.cost_cfg.get("weight_detour", 0.30))
            * candidate.detour_time
            / horizon
            + float(self.cost_cfg.get("weight_task_coverage", 0.20))
            * candidate.task_coverage_distance
            / max(distance_reference, 1e-9)
            + float(self.cost_cfg.get("weight_route_switch", 0.20))
            * candidate.route_switch_penalty
        )

        reason = "ok"
        if candidate.action_type == "keep_route":
            candidate.feasible = True
        elif not schedule_result.get("feasible", True):
            reason = str(schedule_result.get("reason", "recovery_schedule_infeasible"))
        elif candidate.action_type == "wait" and wait_duration <= 1e-9:
            reason = "no_future_wait_event"
        elif candidate.action_type in {"move_to", "reroute"} and (
            candidate.predicted_cav_arrival_time - now
            > float(self.cfg.get("prediction_horizon_s", 300.0)) + 1e-9
        ):
            reason = "prediction_horizon"
        elif candidate.detour_time > float(self.cfg.get("max_detour_time_s", 120.0)) + 1e-9:
            reason = "max_detour_time"
        elif candidate.detour_distance > float(
            self.cfg.get("max_detour_distance_m", self.cfg.get("max_detour_distance", 1000.0))
        ) + 1e-9:
            reason = "max_detour_distance"
        elif candidate.detour_ratio > float(self.cfg.get("max_detour_ratio", 0.25)) + 1e-9:
            reason = "max_detour_ratio"
        elif candidate.min_recovery_slack < float(
            self.cfg.get("minimum_recovery_slack_s", 30.0)
        ) - 1e-9:
            reason = "minimum_recovery_slack"
        elif candidate.max_queue_wait > float(
            self.cfg.get("maximum_uav_wait_s", math.inf)
        ) + 1e-9:
            reason = "maximum_uav_wait"
        elif candidate.reverse_route_flag and bool(
            self.commit_cfg.get("prevent_immediate_reverse", True)
        ):
            reason = "immediate_reverse"
        elif candidate.route_changed and not vehicle.route_change_allowed(now, self.commit_cfg):
            reason = "route_commitment"
        candidate.feasible = reason == "ok"
        candidate.infeasible_reason = reason
        candidate.computation_time_ms = (perf_counter() - started) * 1000.0
        if commit:
            self.commit_cav_action(candidate)
        return candidate

    def _select_best_candidate(
        self, candidates: Sequence[GroundCandidate]
    ) -> GroundCandidate:
        keep = next(candidate for candidate in candidates if candidate.action_type == "keep_route")
        feasible = [candidate for candidate in candidates if candidate.feasible]
        order = {"keep_route": 0, "wait": 1, "move_to": 2, "reroute": 3}
        best = min(
            feasible,
            key=lambda candidate: (
                round(float(candidate.score), 12),
                order[candidate.action_type],
                int(candidate.route_changed),
                round(float(candidate.detour_distance), 9),
                -round(float(candidate.min_recovery_slack), 9),
                candidate.candidate_id,
            ),
        )
        if best.action_type == "keep_route":
            return keep
        improvement = float(keep.score - best.score)
        relative = improvement / max(abs(float(keep.score)), 1e-9)
        absolute_threshold = float(self.cfg.get("min_absolute_improvement", 0.0))
        relative_threshold = float(self.cfg.get("min_relative_improvement", 0.05))
        absolute_pass = absolute_threshold > 0.0 and improvement >= absolute_threshold
        relative_pass = relative >= relative_threshold
        return best if improvement > 0.0 and (absolute_pass or relative_pass) else keep

    def decide(
        self,
        trigger_events: Optional[Iterable[str]] = None,
        next_event_time: Optional[float] = None,
        force: bool = False,
    ) -> Optional[GroundCandidate]:
        now = float(self.sim.now)
        triggers = list(dict.fromkeys(str(x) for x in (trigger_events or ["STATE_ADVANCE"])))
        allowed = {str(x) for x in self.cfg.get("decision_events", [])}
        if not force and allowed and not (allowed & set(triggers)):
            return None
        if self.last_decision_time is not None and abs(self.last_decision_time - now) <= 1e-9:
            return None
        if not self.sim.vehicle.at_intersection:
            return None
        started = perf_counter()
        candidates = self.generate_ground_candidates(next_event_time=next_event_time)
        selected = self._select_best_candidate(candidates)
        keep = next(candidate for candidate in candidates if candidate.action_type == "keep_route")
        committed = self.commit_cav_action(selected)
        if not committed:
            selected = keep
            self.commit_cav_action(keep)
        self.last_decision_time = now
        self.last_selected_action = selected.action_type
        elapsed_ms = (perf_counter() - started) * 1000.0
        infeasible = sorted(
            {candidate.infeasible_reason for candidate in candidates if not candidate.feasible}
        )
        locked_prefix = self._locked_prefix()
        self.action_log.append(
            {
                "episode": 0,
                "seed": int(self.sim.seed),
                "simulation_time": now,
                "trigger_events": "|".join(triggers),
                "cav_current_node": int(self.sim.vehicle.position),
                "cav_current_edge": self.sim.vehicle.current_edge,
                "current_target": self.sim.vehicle.target_node,
                "queue_length": len(self.sim.recovery_queue),
                "locked_prefix_length": locked_prefix,
                "available_ground_actions": "|".join(self.action_set),
                "candidate_count": len(candidates),
                "node_candidate_count": sum(
                    candidate.action_type in {"move_to", "reroute"} for candidate in candidates
                ),
                "feasible_candidate_count": sum(candidate.feasible for candidate in candidates),
                "selected_action": selected.action_type,
                "selected_target": selected.target_node,
                "selected_route": "|".join(str(x) for x in selected.planned_route),
                "affected_recovery_ids": "|".join(selected.affected_recovery_ids),
                "predicted_cav_arrival": selected.predicted_cav_arrival_time,
                "predicted_uav_wait": selected.predicted_uav_wait,
                "predicted_cav_wait": selected.predicted_cav_wait,
                "predicted_uav_return_time": selected.predicted_uav_return_time,
                "predicted_hover_energy": selected.predicted_hover_energy,
                "predicted_detour_distance": selected.detour_distance,
                "predicted_detour_ratio": selected.detour_ratio,
                "predicted_task_coverage_improvement": selected.task_coverage_improvement,
                "minimum_recovery_slack": selected.min_recovery_slack,
                "selected_cost": selected.score,
                "keep_route_cost": keep.score,
                "predicted_gain_vs_keep": keep.score - selected.score,
                "route_changed": selected.route_changed,
                "emergency_override": selected.emergency_override,
                "infeasible_reasons": "|".join(infeasible),
                "computation_time_ms": elapsed_ms,
            }
        )
        for candidate in candidates:
            row = candidate.to_record()
            row.update({"seed": int(self.sim.seed), "simulation_time": now})
            self.candidate_log.append(row)
        return selected

    def _apply_recovery_update(self, recovery_id: str, node: int) -> bool:
        """Atomically replace one flexible recovery target and its return leg.

        Every exact recovery search is completed and battery-checked before an
        entry, route, ETA or queue member is mutated.  Thus a timeout or node
        budget failure leaves the old legal plan intact even when this helper
        is invoked outside ``commit_cav_action``.
        """
        def retain_old(reason: str) -> bool:
            self.replanning_stats["retained_old_plan_count"] += 1
            self.replanning_stats[f"retained_old_plan_{str(reason).split(':', 1)[0]}_count"] += 1
            return False

        locked = self._locked_prefix()
        for index, entry in enumerate(self.sim.recovery_queue):
            if self._entry_id(entry) != str(recovery_id):
                continue
            if index < locked or not self._entry_is_flexible(index, entry):
                return False
            old_node=int(entry["recovery_node"])
            prediction = self._predict_entry_update(entry, int(node))
            if prediction is None:
                return retain_old("prediction_infeasible")
            drone = self.sim.get_drone(str(entry.get("drone_id", entry.get("uav_id"))))
            # An entry without a physical sortie can only carry its cheap
            # prediction; there is no airborne path to replace.
            if drone.sortie is None or not drone.sortie.segments:
                entry["recovery_node"] = int(node)
                entry["eta"] = float(prediction["eta"])
                entry["battery_at_eta"] = float(prediction["battery_at_eta"])
                self.sim.recovery_schedule_version += 1
                return True

            final_index=next(
                (i for i in range(len(drone.sortie.segments)-1, -1, -1)
                if str(drone.sortie.segments[i].get("type")) in {"fly", "horizontal_flight"}),
                None,
            )
            final=drone.sortie.segments[final_index] if final_index is not None else drone.sortie.segments[-1]
            final_is_flight=str(final.get("type")) in {"fly", "horizontal_flight"}
            origin=int(final.get("from" if final_is_flight else "to", drone.position))
            fly_power=float(self.sim.cfg["agents"].get("drone_fly_power", 1.0))
            old_duration=float(final.get("duration", 0.0)) if final_is_flight else 0.0
            old_energy=float(final.get("path_energy", old_duration*fly_power)) if final_is_flight else 0.0
            suffix, on_return_path=self._return_path_suffix(drone, final_index if final_index is not None else -1, final)
            old_remaining=float(drone.segment_remaining) if on_return_path else old_duration
            old_remaining_energy=old_energy*(old_remaining/max(old_duration, 1e-9)) if on_return_path else old_energy
            planning=self.sim.cfg.get("uav_path_planning", {})
            planned: Optional[Dict[str, Any]]=None
            preserve_existing=False

            if self._uses_voxel_recovery_planner():
                context=self._voxel_context()
                if context is None:
                    return retain_old("voxel_context_unavailable")
                try:
                    old_goal=self._recovery_voxel(context, old_node)
                    new_goal=self._recovery_voxel(context, int(node))
                except Exception:
                    return retain_old("recovery_projection_failed")
                base_path=suffix if suffix else [tuple(map(int, p)) for p in final.get("path_voxel_path", final.get("voxel_path", []))]
                if old_goal == new_goal:
                    self.replanning_stats["same_target_skip_count"] += 1
                    if base_path:
                        planned=self._voxel_leg_from_path(context, base_path)
                        if planned is None:
                            return retain_old("invalid_existing_path")
                        if on_return_path:
                            self.replanning_stats["path_suffix_reuse_count"] += 1
                    else:
                        # The old exact leg remains legal because both ground
                        # points project to the same free voxel.  Do not invent
                        # a new ETA/energy estimate or run A*.
                        preserve_existing=True
                elif (
                    base_path
                    and self.sim.network.euclidean_distance(old_node, int(node)) < float(
                        planning.get("recovery_replanning", {}).get("local_connection_distance_m", 20.0)
                    )
                ):
                    connection=context.local_connection(base_path[-1], new_goal)
                    if connection is not None:
                        planned=self._voxel_leg_from_path(context, list(base_path[:-1])+list(connection))
                        if planned is not None:
                            self.replanning_stats["local_connection_success_count"] += 1
                            if on_return_path:
                                self.replanning_stats["path_suffix_reuse_count"] += 1
                if planned is None and base_path:
                    # A changed road node need not be close to the old road
                    # node for the UAV's already legal return corridor to pass
                    # nearby.  Splice at such a point only when the remaining
                    # local voxel connection is below the same 20 m threshold
                    # and validates step-by-step; otherwise A* still owns the
                    # full replan.
                    local_radius=float(planning.get("recovery_replanning", {}).get("local_connection_distance_m", 20.0))
                    best_path=None
                    best_distance=math.inf
                    for path_index, path_voxel in enumerate(base_path):
                        dx=(int(path_voxel[2])-int(new_goal[2]))*float(context.voxel_size_xyz[0])
                        dy=(int(path_voxel[1])-int(new_goal[1]))*float(context.voxel_size_xyz[1])
                        dz=(int(path_voxel[0])-int(new_goal[0]))*float(context.voxel_size_xyz[2])
                        if math.sqrt(dx*dx+dy*dy+dz*dz) >= local_radius:
                            continue
                        connection=context.local_connection(path_voxel, new_goal)
                        if connection is None:
                            continue
                        spliced=list(base_path[:path_index+1])+list(connection[1:])
                        candidate_path=self._voxel_leg_from_path(context, spliced)
                        if candidate_path is not None and candidate_path["distance_m"] < best_distance:
                            best_path=candidate_path
                            best_distance=float(candidate_path["distance_m"])
                    if best_path is not None:
                        planned=best_path
                        self.replanning_stats["local_connection_success_count"] += 1
                        self.replanning_stats["path_splice_reuse_count"] += 1
                        if on_return_path:
                            self.replanning_stats["path_suffix_reuse_count"] += 1
                if planned is None and not preserve_existing:
                    if not self._reserve_exact_replan(drone.id):
                        return retain_old("exact_replan_limit")
                    planned=self._plan_voxel_recovery_leg(
                        origin, int(node), start_voxel=base_path[0] if (on_return_path and base_path) else None,
                    )
                    if planned is None:
                        return retain_old(self._last_recovery_plan_failure_reason or "exact_replan_failed")
            elif bool(planning.get("enabled", False)):
                # Historical non-voxel scenarios preserve their existing 2-D
                # route behaviour; the optimization never changes that branch.
                try:
                    import sys
                    from pathlib import Path
                    module_root=str(Path(__file__).resolve().parents[1] / "truck_uav_modules")
                    if module_root not in sys.path:
                        sys.path.insert(0, module_root)
                    from uav_path_planner import plan_uav_path
                    region=planning.get("region", self.sim.cfg.get("network", {}).get("uav_path_region"))
                    flight=self.sim.cfg.get("uav_flight", {})
                    legacy=plan_uav_path(str(region), self.sim.network.node_xy(origin), self.sim.network.node_xy(int(node)), candidate_heights=(float(flight.get("cruise_height_m", 60.0)),))
                    if not legacy.get("feasible", False):
                        return retain_old("legacy_replan_failed")
                    planned={"duration_s": float(legacy["distance_m"])/max(float(flight.get("cruise_speed_mps", self.sim.cfg["network"].get("drone_speed_mps", 14.0))), 1e-9), "energy": None, "distance_m": float(legacy["distance_m"])}
                except Exception:
                    return retain_old("legacy_replan_failed")
            else:
                planned={"duration_s": self.sim.network.get_drone_time(origin, int(node)), "energy": None, "distance_m": 0.0}

            if preserve_existing:
                # Commit only the logical ground-node alias; physical path,
                # arrival time and energy remain the old exact values.
                entry["recovery_node"] = int(node)
                if final_is_flight:
                    final["to"] = int(node)
                drone.sortie.recovery_node=int(node)
                drone.schedule_version += 1
                self.sim.recovery_schedule_version += 1
                return True

            assert planned is not None
            new_duration=float(planned["duration_s"])
            new_energy=float(planned.get("energy") if planned.get("energy") is not None else new_duration*fly_power)
            if self._uses_voxel_recovery_planner():
                estimated_time=float(prediction["return_time"])
                exact_battery=float(prediction["battery_at_eta"]) - (new_energy-estimated_time*fly_power)
                if exact_battery < float(entry.get("battery_reserve", 0.0))-1e-9:
                    return retain_old("battery_reserve_after_exact_path")
                new_eta=float(prediction["eta"])+new_duration-estimated_time
            else:
                exact_battery=float(prediction["battery_at_eta"])
                new_eta=float(prediction["eta"])

            if final_is_flight:
                updated_final=dict(final)
                updated_final.update({"to": int(node), "duration": new_duration, "distance_m": float(planned["distance_m"]), "estimated": False})
                if self._uses_voxel_recovery_planner():
                    updated_final.update({
                        "path_mode": "voxel_astar_3d", "waypoints_xyz": list(planned["waypoints_xyz"]),
                        "voxel_path": list(planned["voxel_path"]), "path_voxel_path": list(planned["voxel_path"]),
                        "path_voxel_segment": dict(planned["segment_result"]), "path_energy": new_energy,
                    })
                drone.sortie.segments[final_index]=updated_final
            else:
                updated_final={"type": "fly", "from": origin, "to": int(node), "duration": new_duration, "distance_m": float(planned["distance_m"]), "estimated": False}
                if self._uses_voxel_recovery_planner():
                    updated_final.update({
                        "path_mode": "voxel_astar_3d", "waypoints_xyz": list(planned["waypoints_xyz"]),
                        "voxel_path": list(planned["voxel_path"]), "path_voxel_path": list(planned["voxel_path"]),
                        "path_voxel_segment": dict(planned["segment_result"]), "path_energy": new_energy,
                    })
                drone.sortie.segments.append(updated_final)

            schedule_delta=new_duration-old_remaining
            drone.sortie.recovery_node=int(node)
            drone.sortie.drone_time_s += schedule_delta
            drone.sortie.energy += new_energy-old_remaining_energy
            drone.sortie.flight_service_energy += new_energy-old_remaining_energy
            drone.sortie.finish_time_s += schedule_delta
            drone.sortie.battery_at_eta=exact_battery
            if on_return_path:
                drone.segment_remaining=new_duration
            entry["recovery_node"] = int(node)
            entry["eta"] = new_eta
            entry["battery_at_eta"] = exact_battery
            drone.schedule_version += 1
            self.sim.recovery_schedule_version += 1
            return True
        return False

    def _coalesced_recovery_updates(self, updates: Dict[str, int]) -> List[Tuple[str, int]]:
        """Keep the last target for each UAV at one simulation timestamp."""
        by_drone: Dict[str, Tuple[str, int]] = {}
        for recovery_id, node in updates.items():
            entry=next((item for item in self.sim.recovery_queue if self._entry_id(item) == str(recovery_id)), None)
            if entry is None:
                continue
            drone_id=str(entry.get("drone_id", entry.get("uav_id")))
            if drone_id in by_drone:
                self.replanning_stats["merged_request_count"] += 1
            by_drone[drone_id]=(str(recovery_id), int(node))
        return list(by_drone.values())

    def commit_cav_action(
        self, candidate: GroundCandidate, *, validate_fingerprint: bool = True
    ) -> bool:
        if not candidate.feasible or (
            validate_fingerprint and candidate.state_fingerprint != self.state_fingerprint()
        ):
            return candidate.action_type == "keep_route" and candidate.feasible
        if candidate.action_type == "keep_route":
            self.sim._set_vehicle_next_recovery_target()
            return True
        vehicle = self.sim.vehicle
        now = float(self.sim.now)
        if not vehicle.at_intersection:
            return False
        snapshot = (
            deepcopy(self.sim.recovery_queue),
            deepcopy(vehicle.__dict__),
            [(drone, deepcopy(drone.__dict__)) for drone in self.sim.drones],
            int(self.sim.recovery_schedule_version),
        )
        try:
            for recovery_id, node in self._coalesced_recovery_updates(candidate.recovery_updates):
                if not self._apply_recovery_update(recovery_id, node):
                    raise ValueError("recovery_update_became_infeasible")
            old_target = vehicle.target_node
            old_route = self._existing_route()
            if candidate.action_type == "wait":
                vehicle.begin_wait(float(candidate.wait_until), now)
            else:
                route = list(candidate.planned_route)
                if candidate.action_type == "move_to" and candidate.target_node in route:
                    route = route[: route.index(int(candidate.target_node)) + 1]
                final_target = int(route[-1]) if route else int(candidate.target_node)
                vehicle.set_route(route, self.sim.network, now, final_target)
                vehicle.note_route_change(now, final_target, self.commit_cfg)
                self.total_committed_detour_distance += float(candidate.detour_distance)
            schedule_by_id = {
                str(row["sortie_id"]): row for row in candidate.predicted_recovery_schedule
            }
            for entry in self.sim.recovery_queue:
                row = schedule_by_id.get(self._entry_id(entry))
                if row is None:
                    continue
                for key in ("scheduled_start_time", "scheduled_finish_time", "predicted_uav_wait", "predicted_cav_wait"):
                    entry[key] = float(row[key])
                self.recovery_schedule_log.append(
                    {"seed": int(self.sim.seed), "simulation_time": now, **row}
                )
            keep_wait = {}
            for row in candidate.predicted_recovery_schedule:
                keep_wait[str(row["sortie_id"])] = float(row.get("predicted_uav_wait", 0.0))
            for recovery_id in candidate.affected_recovery_ids:
                self.pending_realized_gain[recovery_id] = {
                    "selected_predicted_wait": keep_wait.get(recovery_id, 0.0),
                    "keep_predicted_wait": keep_wait.get(recovery_id, 0.0)
                    + max(0.0, candidate.recovery_improvement),
                }
            self.sim.event_log.append(
                {
                    "time": now,
                    "event": "cav_ground_action",
                    "agent": vehicle.id,
                    "node": int(vehicle.position),
                    "target": candidate.target_node,
                    "action": candidate.action_type,
                    "tasks": "",
                }
            )
            self.route_change_log.append(
                {
                    "seed": int(self.sim.seed),
                    "simulation_time": now,
                    "action": candidate.action_type,
                    "action_type": candidate.action_type,
                    "route_changed": bool(candidate.route_changed),
                    "old_target": old_target,
                    "new_target": vehicle.target_node,
                    "old_route": "|".join(str(x) for x in old_route),
                    "new_route": "|".join(str(x) for x in candidate.planned_route),
                    "detour_distance": candidate.detour_distance,
                    "detour_ratio": candidate.detour_ratio,
                    "reverse_route_flag": candidate.reverse_route_flag,
                    "emergency_override": candidate.emergency_override,
                }
            )
            return True
        except Exception:
            self.sim.recovery_queue[:] = snapshot[0]
            vehicle.__dict__.clear()
            vehicle.__dict__.update(snapshot[1])
            for drone, state in snapshot[2]:
                drone.__dict__.clear()
                drone.__dict__.update(state)
            self.sim.recovery_schedule_version = snapshot[3]
            return False

    def record_recovery(self, recovery_id: str, actual_wait: float) -> None:
        prediction = self.pending_realized_gain.pop(str(recovery_id), None)
        if prediction is None:
            return
        self.realized_gains.append(float(prediction["keep_predicted_wait"] - actual_wait))

    def summary(self) -> Dict[str, Any]:
        rows = self.action_log
        counts = {
            action: sum(row.get("selected_action") == action for row in rows)
            for action in GROUND_ACTIONS
        }
        selected_non_keep = [row for row in rows if row.get("selected_action") != "keep_route"]
        return {
            "keep_route_count": counts["keep_route"],
            "wait_count": counts["wait"],
            "move_to_count": counts["move_to"],
            "reroute_count": counts["reroute"],
            "non_keep_action_rate": len(selected_non_keep) / len(rows) if rows else 0.0,
            "route_change_count": sum(
                bool(row.get("route_changed")) and not bool(row.get("emergency_override"))
                for row in self.route_change_log
            ),
            "route_reversal_count": sum(
                bool(row.get("reverse_route_flag")) and not bool(row.get("emergency_override"))
                for row in self.route_change_log
            ),
            "emergency_override_count": sum(bool(row.get("emergency_override")) for row in self.route_change_log),
            "mean_route_commitment_time_s": self.sim.vehicle.mean_route_commitment_time(),
            "ground_candidate_count": float(sum(row.get("candidate_count", 0) for row in rows) / len(rows)) if rows else 0.0,
            "ground_action_computation_time_ms": float(sum(row.get("computation_time_ms", 0.0) for row in rows) / len(rows)) if rows else 0.0,
            "mean_predicted_gain_vs_keep": float(sum(row.get("predicted_gain_vs_keep", 0.0) for row in rows) / len(rows)) if rows else 0.0,
            "mean_realized_gain_vs_keep": float(sum(self.realized_gains) / len(self.realized_gains)) if self.realized_gains else 0.0,
            "cav_detour_distance_m": float(self.total_committed_detour_distance),
            "cav_detour_ratio": float(sum(row.get("predicted_detour_ratio", 0.0) for row in selected_non_keep) / len(selected_non_keep)) if selected_non_keep else 0.0,
            "maximum_cav_detour_ratio": max((float(row.get("predicted_detour_ratio", 0.0)) for row in selected_non_keep), default=0.0),
            "minimum_recovery_slack_s": min((float(row.get("minimum_recovery_slack", math.inf)) for row in rows), default=math.inf),
            "maximum_queue_wait_s": max((float(row.get("predicted_uav_wait", 0.0)) for row in rows), default=0.0),
        }
