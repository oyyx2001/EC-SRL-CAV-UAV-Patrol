from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict, fields
from itertools import combinations, permutations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
from time import perf_counter
import json
import math

import numpy as np

from dynamic_patrol_marl.env.road_network import RoadNetwork
from dynamic_patrol_marl.env.task_generator import Task
from dynamic_patrol_marl.env.drone import Drone
from dynamic_patrol_marl.utils.operation_times import planning_operation_time_s


@dataclass
class Sortie:
    drone_id: str
    launch_node: int
    task_ids: List[int]
    task_sequence: List[int]
    recovery_node: int
    drone_time_s: float
    vehicle_time_s: float
    finish_time_s: float
    energy: float
    score: float
    feasible: bool
    reason: str = "ok"
    segments: List[Dict] = None
    tasks_per_sortie: int = 1
    vehicle_anchor_node: Optional[int] = None
    vehicle_queue_time_s: float = 0.0
    flight_service_energy: float = 0.0
    battery_at_eta: float = 0.0
    base_feasible: bool = True
    insertion_evaluation: Optional[Dict[str, Any]] = None
    planned_launch_time_s: float = 0.0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["segments"] = self.segments or []
        d.pop("insertion_evaluation", None)
        return d


class SortieGenerator:
    def __init__(self, cfg: Dict, network: RoadNetwork):
        self.cfg = cfg
        self.network = network
        self.sortie_cfg = cfg["sortie"]
        self.agent_cfg = cfg["agents"]
        self.ablation_cfg = cfg.get("ablation", {})
        self.drone_speed = float(cfg["network"].get("drone_speed_mps", 14.0))
        self.fly_power = float(self.agent_cfg.get("drone_fly_power", 1.0))
        self.hover_power = float(self.agent_cfg.get("drone_hover_power", 0.65))
        self.service_power = float(self.agent_cfg.get("drone_service_power", 0.85))
        self.planned_launch_time_s = planning_operation_time_s(cfg, "launch")
        self.flight_cfg = dict(cfg.get("uav_flight", {}))
        self.path_cfg = dict(cfg.get("uav_path_planning", {}))
        self.vertical_flight_enabled = bool(self.flight_cfg) and bool(self.path_cfg.get("enabled", False))
        # Per-process cache.  Keys use GraphML IDs (when available), rather
        # than transient internal integers, and are map-version scoped.
        self.path_cache: Dict[Tuple[str, str, float, str, str, float], Dict[str, Any]] = {}
        self._voxel_context = None
        self.last_materialization: Dict[str, Any] = {}
        # Candidate V1 stays opt-in so the unoptimised generator remains a
        # reproducible baseline under ``candidate_generation.optimization_enabled: false``.
        self.candidate_cfg = dict(cfg.get("candidate_generation", {}))
        self.candidate_optimization_enabled = bool(self.candidate_cfg.get("optimization_enabled", False))
        self._distance_cache: Dict[Tuple[int, int, str], float] = {}
        self._cav_eta_cache: Dict[Tuple[int, int, str], float] = {}
        self._connectivity_cache: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int], str], Optional[bool]] = {}
        self._recovery_candidate_cache: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
        self._cache_counts = {
            "distance_cache_hits": 0, "distance_cache_misses": 0,
            "cav_eta_cache_hits": 0, "cav_eta_cache_misses": 0,
            "connectivity_cache_hits": 0, "connectivity_cache_misses": 0,
        }
        self._generation_serial = 0
        self.candidate_generation_log: List[Dict[str, Any]] = []
        self.last_candidates_by_drone: Dict[str, List[Sortie]] = {}
        self._connectivity_labels = None
        self._connectivity_node_voxels: Dict[int, Tuple[int, int, int]] = {}
        self._connectivity_version = "disabled"
        self._load_optional_connectivity_labels()

    def _network_map_version(self) -> str:
        return str(self.cfg.get("network", {}).get("map_version", "default"))

    @staticmethod
    def _copy_sortie_for_materialization(sortie: Sortie) -> Sortie:
        """Copy only declared ``Sortie`` fields before adding exact-path data.

        Candidate generation may attach private ranking metadata (for example
        ``_candidate_tie_priority``) to an instance.  That metadata is not a
        constructor argument and must not leak into the commit-time clone.
        """
        return Sortie(**{
            field.name: deepcopy(getattr(sortie, field.name))
            for field in fields(Sortie)
        })

    def _candidate_stats(self, drone: Drone, now: float, available_tasks: Sequence[Task], decision_epoch_id: Optional[int]) -> Dict[str, Any]:
        return {
            "decision_epoch_id": int(self._generation_serial if decision_epoch_id is None else decision_epoch_id),
            "simulation_time": float(now), "uav_id": str(drone.id),
            "idle_uav_count": 1, "available_task_count": len(available_tasks),
            "prefilter_tasks_before": len(available_tasks), "prefilter_tasks_after": 0,
            "single_task_sequence_count": 0, "double_task_sequence_count": 0, "triple_task_sequence_count": 0,
            "connectivity_pruned_count": 0, "deadline_pruned_count": 0,
            "energy_pruned_count": 0, "no_recovery_pruned_count": 0,
            "duplicate_candidate_count": 0, "recovery_nodes_before": 0, "recovery_nodes_after": 0,
            "cheap_candidate_count": 0, "full_evaluation_count": 0,
            "full_evaluation_budget_truncated_count": 0, "feasible_sortie_count": 0,
            "retained_sortie_count": 0, "task_prefilter_time_ms": 0.0,
            "sequence_generation_time_ms": 0.0, "connectivity_filter_time_ms": 0.0,
            "recovery_prefilter_time_ms": 0.0, "sortie_evaluate_time_ms": 0.0,
            "raw_joint_plan_count": 0, "duplicate_joint_plan_count": 0,
            "deduplicated_joint_plan_count": 0, "retained_joint_plan_count": 0,
            "joint_plan_generation_time_ms": 0.0,
            "candidate_generation_total_ms": 0.0,
            "distance_cache_hits": 0, "distance_cache_misses": 0,
            "cav_eta_cache_hits": 0, "cav_eta_cache_misses": 0,
            "connectivity_cache_hits": 0, "connectivity_cache_misses": 0,
            "optimization_enabled": bool(self.candidate_optimization_enabled),
            "connectivity_filter_active": bool(self._connectivity_labels is not None),
        }

    def _load_optional_connectivity_labels(self) -> None:
        """Load precomputed voxel components only when explicitly configured.

        Candidate generation never creates labels or invokes 3-D A*: absent
        labels mean "unknown", which intentionally passes the filter.
        """
        if not bool(self.candidate_cfg.get("enable_connectivity_filter", False)):
            return
        labels_name = self.candidate_cfg.get("connectivity_labels_file")
        voxels_name = self.candidate_cfg.get("connectivity_node_voxels_file")
        if not labels_name or not voxels_name:
            return
        root = Path(__file__).resolve().parents[1]
        labels_path = Path(labels_name)
        voxels_path = Path(voxels_name)
        if not labels_path.is_absolute():
            labels_path = root / labels_path
        if not voxels_path.is_absolute():
            voxels_path = root / voxels_path
        if not labels_path.exists() or not voxels_path.exists():
            return
        try:
            labels = np.load(labels_path, allow_pickle=False)
            raw = json.loads(voxels_path.read_text(encoding="utf-8"))
            raw = raw.get("node_voxels", raw) if isinstance(raw, dict) else {}
            node_voxels = {
                int(node): tuple(int(value) for value in voxel)
                for node, voxel in raw.items()
                if isinstance(voxel, (list, tuple)) and len(voxel) == 3
            }
            if labels.ndim != 3 or not node_voxels:
                return
            self._connectivity_labels = labels
            self._connectivity_node_voxels = node_voxels
            self._connectivity_version = f"{self._network_map_version()}:{labels_path.stat().st_mtime_ns}"
        except (OSError, ValueError, json.JSONDecodeError):
            # The compatibility path must not make an existing simulator fail.
            self._connectivity_labels = None
            self._connectivity_node_voxels = {}
            self._connectivity_version = "unavailable"

    def _fast_distance(self, start: int, goal: int) -> float:
        key = (int(start), int(goal), self._network_map_version())
        if bool(self.candidate_cfg.get("enable_distance_cache", True)) and key in self._distance_cache:
            self._cache_counts["distance_cache_hits"] += 1
            return self._distance_cache[key]
        self._cache_counts["distance_cache_misses"] += 1
        value = float(self.network.euclidean_distance(int(start), int(goal)))
        if bool(self.candidate_cfg.get("enable_distance_cache", True)):
            self._distance_cache[key] = value
        return value

    def _cached_cav_eta(self, start: int, recovery: int) -> float:
        key = (int(start), int(recovery), self._network_map_version())
        if bool(self.candidate_cfg.get("enable_cav_eta_cache", True)) and key in self._cav_eta_cache:
            self._cache_counts["cav_eta_cache_hits"] += 1
            return self._cav_eta_cache[key]
        self._cache_counts["cav_eta_cache_misses"] += 1
        value = float(self.network.get_vehicle_time(int(start), int(recovery)))
        if bool(self.candidate_cfg.get("enable_cav_eta_cache", True)):
            self._cav_eta_cache[key] = value
        return value

    def _connected(self, start: int, goal: int) -> Optional[bool]:
        if self._connectivity_labels is None:
            return None
        start_voxel = self._connectivity_node_voxels.get(int(start))
        goal_voxel = self._connectivity_node_voxels.get(int(goal))
        if start_voxel is None or goal_voxel is None:
            return None
        key = (start_voxel, goal_voxel, self._connectivity_version)
        if bool(self.candidate_cfg.get("enable_connectivity_filter", True)) and key in self._connectivity_cache:
            self._cache_counts["connectivity_cache_hits"] += 1
            return self._connectivity_cache[key]
        self._cache_counts["connectivity_cache_misses"] += 1
        try:
            a = int(self._connectivity_labels[start_voxel])
            b = int(self._connectivity_labels[goal_voxel])
            # Negative labels and out-of-range lookup are unknown, never a
            # reason to reject a sortie before exact materialisation.
            value: Optional[bool] = None if a < 0 or b < 0 else bool(a == b)
        except (IndexError, TypeError):
            value = None
        if bool(self.candidate_cfg.get("enable_connectivity_filter", True)):
            self._connectivity_cache[key] = value
        return value

    @staticmethod
    def _task_nodes(task: Task) -> Tuple[int, ...]:
        return tuple(sorted({int(node) for arc in (task.target_arcs or []) for node in arc}))

    def _any_connected(self, starts: Sequence[int], goals: Sequence[int]) -> Optional[bool]:
        if self._connectivity_labels is None:
            return None
        outcomes = [self._connected(start, goal) for start in starts for goal in goals]
        if any(result is True for result in outcomes):
            return True
        return False if outcomes and all(result is False for result in outcomes) else None

    def _task_min_service_time(self, task: Task) -> float:
        speed = self._flight_parameters()["cruise_speed"] if self.vertical_flight_enabled else self.drone_speed
        coverage = 0.0
        if bool(task.metadata.get("edge_task", False)):
            coverage = sum(self.network.arc_length((int(u), int(v))) for u, v in task.target_arcs) / max(speed, 1e-9)
        return max(0.0, float(task.service_time)) + coverage

    def _task_is_high_priority_or_urgent(self, task: Task, now: float) -> bool:
        high = float(task.priority) >= float(self.candidate_cfg.get("high_priority_threshold", 8.0))
        urgent = float(task.deadline) - float(now) <= float(self.candidate_cfg.get("urgent_deadline_threshold_s", 300.0))
        incident = "incident" in str(task.type).lower() or bool(task.metadata.get("incident_confirmation", False))
        return high or urgent or incident

    def _prefilter_tasks(self, drone: Drone, launch_node: int, now: float, available_tasks: Sequence[Task], stats: Dict[str, Any]) -> List[Task]:
        started = perf_counter()
        valid: List[Tuple[Task, float, bool]] = []
        for task in available_tasks:
            nodes = self._task_nodes(task)
            if (
                str(task.status) != "unassigned"
                or str(task.executor) not in {"drone", "both"}
                or not nodes
                or not math.isfinite(float(task.deadline))
                or float(task.deadline) < float(now)
            ):
                continue
            distance = min(self._fast_distance(int(launch_node), node) for node in nodes)
            flight_service_time = (
                distance / max(self.drone_speed, 1e-9)
                + self._task_min_service_time(task)
            )
            lower_time = self.planned_launch_time_s + flight_service_time
            # Launch preparation delays task completion but does not consume
            # onboard UAV energy; only flight/service time belongs here.
            lower_energy = flight_service_time * min(self.fly_power, self.service_power)
            # Only remove a protected task after a proven optimistic failure.
            if float(now) + lower_time > float(task.deadline) + 1e-9:
                stats["deadline_pruned_count"] += 1
                continue
            if lower_energy > float(drone.battery) + 1e-9:
                stats["energy_pruned_count"] += 1
                continue
            priority = float(task.priority)
            urgency = 1.0 / max(1.0, float(task.deadline) - float(now))
            waiting = max(0.0, float(now) - float(task.release_time))
            score = 8.0 * priority + 5.0 * urgency - 0.01 * lower_time - 0.001 * lower_energy - 0.0001 * waiting
            valid.append((task, score, self._task_is_high_priority_or_urgent(task, now)))
        forced = [row for row in valid if row[2]]
        other = [row for row in valid if not row[2]]
        forced.sort(key=lambda row: (-float(row[0].priority), float(row[0].deadline) - float(now), self._fast_distance(int(launch_node), row[0].start_node), int(row[0].id)))
        other.sort(key=lambda row: (-row[1], int(row[0].id)))
        limit = max(1, int(self.candidate_cfg.get("max_prefiltered_tasks_per_uav", 6)))
        selected = [row[0] for row in (forced + other)[:limit]]
        stats["prefilter_tasks_after"] = len(selected)
        stats["task_prefilter_time_ms"] += (perf_counter() - started) * 1000.0
        return selected

    def _flight_parameters(self) -> Dict[str, float]:
        """Read new parameters while retaining the legacy model when disabled."""
        return {
            "height": float(self.flight_cfg.get("cruise_height_m", 0.0)),
            "up_speed": float(self.flight_cfg.get("climb_speed_mps", self.drone_speed)),
            "down_speed": float(self.flight_cfg.get("descent_speed_mps", self.drone_speed)),
            "cruise_speed": float(self.flight_cfg.get("cruise_speed_mps", self.drone_speed)),
            "up_power": float(self.flight_cfg.get("climb_power", self.fly_power)),
            "cruise_power": float(self.flight_cfg.get("cruise_power", self.fly_power)),
            "down_power": float(self.flight_cfg.get("descent_power", self.fly_power)),
            "service_power": float(self.flight_cfg.get("service_power", self.service_power)),
            "factor": float(self.path_cfg.get("candidate_estimation_factor", 1.15)),
        }

    def _vertical_segment(self, segment_type: str, node: int, duration: float) -> Dict[str, Any]:
        return {"type": segment_type, "from": int(node), "to": int(node), "duration": float(duration)}

    def _horizontal_segment(self, start: int, end: int, task_id: Optional[int] = None) -> Dict[str, Any]:
        p = self._flight_parameters()
        distance = self.network.euclidean_distance(start, end) * p["factor"]
        return {
            "type": "horizontal_flight", "from": int(start), "to": int(end),
            "duration": distance / max(p["cruise_speed"], 1e-9), "distance_m": float(distance),
            "task_id": task_id, "estimated": True,
        }

    def materialize_selected_sortie(
        self, sortie: Sortie, drone: Drone, now: float, task_by_id: Dict[int, Task]
    ) -> Optional[Sortie]:
        """Replace estimated horizontal legs with paths from the real UAV planner.

        This deliberately runs only at commit time.  It returns ``None`` rather
        than silently retaining an estimate when planning is requested but fails.
        """
        if not self.vertical_flight_enabled:
            self.last_materialization = {"materialization_success": True, "failure_stage": None, "failure_reason": None, "segment_results": []}
            return sortie
        region = self.path_cfg.get("region", self.cfg.get("network", {}).get("uav_path_region"))
        if not region:
            self.last_materialization = {"materialization_success": False, "failure_stage": "before_astar", "failure_reason": "invalid_region", "segment_results": []}
            return None
        try:
            import sys
            from pathlib import Path
            module_root = str(Path(__file__).resolve().parents[1] / "truck_uav_modules")
            if module_root not in sys.path:
                sys.path.insert(0, module_root)
            from uav_path_planner import plan_uav_path, VoxelPlannerContext, plan_selected_sortie_3d
        except Exception:
            self.last_materialization = {"materialization_success": False, "failure_stage": "before_astar", "failure_reason": "planner_import_error", "segment_results": []}
            return None
        p = self._flight_parameters()
        use_voxel = str(self.path_cfg.get("planner_type", "")).lower() == "voxel_astar_3d"
        if use_voxel and self._voxel_context is None:
            self._voxel_context = VoxelPlannerContext(str(region), self.path_cfg)
        # A selected sortie is materialized as one ordered 3-D itinerary.  The
        # candidate's horizontal legs define its already-chosen task order; no
        # planner is invoked during candidate generation.
        if use_voxel:
            legs = [s for s in (sortie.segments or []) if s.get("type") == "horizontal_flight"]
            if legs:
                altitude = float(self.path_cfg.get("default_task_observation_altitude_m", 20.0))
                launch = self.network.node_xy(int(legs[0]["from"]))
                # ``horizontal_flight`` is the candidate's explicit airborne
                # itinerary.  Road-edge coverage/service segments between two
                # legs are not additional airborne waypoints.
                goals = [self.network.node_xy(int(s["to"])) for s in legs]
                state = {"cruise_speed": p["cruise_speed"], "climb_speed": p["up_speed"], "descent_speed": p["down_speed"], "cruise_power": p["cruise_power"], "climb_power": p["up_power"], "descent_power": p["down_power"]}
                precise = plan_selected_sortie_3d(
                    (launch[0], launch[1], altitude),
                    [(x, y, altitude) for x, y in goals[:-1]],
                    (goals[-1][0], goals[-1][1], altitude), state, self._voxel_context,
                    source="initial_materialization", budget_profile="materialization",
                )
                if not precise.feasible:
                    self.last_materialization = {"materialization_success": False, "failure_reason": precise.failure_reason, "failure_stage": "during_astar", "astar_3d_called": 1, "segment_results": list(precise.segment_results or []), "segment_count": len(legs)}
                    return None
                # Allocate each A* leg back to execution segments, preserving
                # service/coverage semantics while replacing all flight ETAs.
                per_leg = list(precise.segment_results or [])
                rebuilt=[]; elapsed=float(getattr(sortie, "planned_launch_time_s", self.planned_launch_time_s)); energy=0.0; arrivals={}; leg_i=0
                for source in sortie.segments or []:
                    seg=dict(source); kind=str(seg.get("type"))
                    if kind == "horizontal_flight":
                        item=per_leg[leg_i]; leg_i += 1; seg.update({"distance_m": float(item["distance_m"]), "duration": float(item["distance_m"]) / max(p["cruise_speed"],1e-9), "estimated":False, "path_mode":"voxel_astar_3d", "path_voxel_segment": item, "path_voxel_path": list(item.get("voxel_path", []))})
                    elapsed += float(seg.get("duration",0.0)); rebuilt.append(seg)
                    if kind in {"horizontal_flight","edge_coverage"}: energy += float(seg.get("duration",0))*p["cruise_power"]
                    elif kind == "vertical_climb": energy += float(seg.get("duration",0))*p["up_power"]
                    elif kind == "vertical_descent": energy += float(seg.get("duration",0))*p["down_power"]
                    elif kind == "service":
                        energy += float(seg.get("duration",0))*p["service_power"]
                        if bool(seg.get("service_final",True)): arrivals[int(seg["task_id"])] = float(now)+elapsed
                result=self._copy_sortie_for_materialization(sortie); result.segments=rebuilt; result.drone_time_s=elapsed; result.energy=energy; result.flight_service_energy=energy; result.battery_at_eta=float(drone.battery)-energy; result.finish_time_s=float(now)+max(elapsed,result.vehicle_time_s)
                result.base_feasible=energy <= drone.battery + 1e-9 and all(arrivals[t] <= task_by_id[t].deadline + 1e-9 for t in arrivals); result.feasible=result.base_feasible; result.reason="ok" if result.feasible else "real_path_infeasible"
                result.active_voxel_path=precise.voxel_path; result.active_path_xyz=precise.waypoints_xyz; result.actual_distance_m=precise.total_distance_m; result.actual_flight_time_s=precise.flight_time_s; result.task_arrival_times=arrivals; result.recovery_eta=float(now)+elapsed
                self.last_materialization={"planner_type":"voxel_astar_3d","astar_3d_called":1,"segment_count":len(per_leg),"cache_hit_segments":precise.cache_hit_segments,"cache_miss_segments":precise.cache_miss_segments,"planning_time_ms":1000*precise.planning_time_s,"expanded_nodes":precise.expanded_nodes,"estimated_distance_m":sum(float(s.get("distance_m",0)) for s in legs),"actual_distance_m":precise.total_distance_m,"estimated_flight_time_s":sum(float(s.get("duration",0)) for s in legs),"actual_flight_time_s":precise.flight_time_s,"materialization_success":bool(result.feasible),"failure_stage":None if result.feasible else "after_astar","failure_reason":None if result.feasible else result.reason,"segment_results":per_leg}
                return result if result.feasible else None
        rebuilt: List[Dict[str, Any]] = []
        elapsed = float(getattr(sortie, "planned_launch_time_s", self.planned_launch_time_s))
        energy = 0.0
        task_elapsed: Dict[int, float] = {}
        for segment in sortie.segments or []:
            seg = dict(segment)
            kind = str(seg.get("type"))
            if kind == "horizontal_flight":
                start_xy = self.network.node_xy(int(seg["from"]))
                end_xy = self.network.node_xy(int(seg["to"]))
                start_external = self.network.to_external_node(int(seg["from"]))
                end_external = self.network.to_external_node(int(seg["to"]))
                cache_key = (
                    start_external, end_external, float(p["height"]),
                    str(self.path_cfg.get("planner_mode", "visibility_graph")),
                    str(self.cfg.get("network", {}).get("map_version", "default")),
                    float(self.path_cfg.get("search_buffer_m", 50.0)),
                )
                try:
                    if use_voxel:
                        altitude = float(self.path_cfg.get("default_task_observation_altitude_m", p["height"] or 20.0))
                        result = plan_selected_sortie_3d(
                            (start_xy[0], start_xy[1], altitude), [], (end_xy[0], end_xy[1], altitude),
                            {"cruise_speed": p["cruise_speed"], "climb_speed": p["up_speed"], "descent_speed": p["down_speed"], "cruise_power": p["cruise_power"], "climb_power": p["up_power"], "descent_power": p["down_power"]}, self._voxel_context,
                            source="initial_materialization", budget_profile="materialization",
                        )
                        if not result.feasible:
                            return None
                        planned = {"feasible": True, "distance_m": result.total_distance_m, "path_mode": result.path_type,
                                   "waypoints_xyz": result.waypoints_xyz, "voxel_path": result.voxel_path,
                                   "planning_time_s": result.planning_time_s, "expanded_nodes": result.expanded_nodes}
                    else:
                        if bool(self.path_cfg.get("fallback_to_2d_planner", False)):
                            planned = plan_uav_path(
                                str(region), start_xy, end_xy, candidate_heights=(p["height"],),
                                search_buffer_m=float(self.path_cfg.get("search_buffer_m", 50.0)))
                        else:
                            return None
                except Exception:
                    return None
                if not planned.get("feasible", False) or planned.get("distance_m") is None:
                    return None
                seg["distance_m"] = float(planned["distance_m"])
                seg["duration"] = seg["distance_m"] / max(p["cruise_speed"], 1e-9)
                seg["estimated"] = False
                seg["path_mode"] = str(planned.get("path_mode", "unknown"))
                seg["waypoints_xyz"] = planned.get("waypoints_xyz", [])
                seg["voxel_path"] = planned.get("voxel_path", [])
                seg["path_cache_key"] = cache_key
                seg["start_external_node_id"] = start_external
                seg["end_external_node_id"] = end_external
            rebuilt.append(seg)
            duration = float(seg.get("duration", 0.0))
            elapsed += duration
            if kind == "vertical_climb": energy += duration * p["up_power"]
            elif kind == "vertical_descent": energy += duration * p["down_power"]
            elif kind in {"horizontal_flight", "edge_coverage"}: energy += duration * p["cruise_power"]
            elif kind == "service":
                energy += duration * p["service_power"]
                if bool(seg.get("service_final", True)):
                    task_elapsed[int(seg["task_id"])] = elapsed
        result = self._copy_sortie_for_materialization(sortie)
        result.segments = rebuilt
        result.drone_time_s = elapsed
        result.energy = energy
        result.flight_service_energy = energy
        result.battery_at_eta = max(0.0, float(drone.battery) - energy)
        result.finish_time_s = float(now) + max(float(elapsed), float(result.vehicle_time_s))
        result.base_feasible = result.battery_at_eta >= -1e-9 and all(
            task_by_id[task_id].deadline + 1e-9 >= float(now) + eta
            for task_id, eta in task_elapsed.items()
        )
        result.feasible = bool(result.base_feasible and result.energy <= float(drone.battery) + 1e-9)
        result.reason = "ok" if result.feasible else "real_path_infeasible"
        return result if result.feasible else None


    def generate_for_drone(
        self,
        drone: Drone,
        vehicle_position: int,
        now: float,
        available_tasks: Sequence[Task],
        max_tasks: Optional[int] = None,
        top_k: Optional[int] = None,
        no_priority: bool = False,
        recovery_anchor_node: Optional[int] = None,
        vehicle_queue_time_s: float = 0.0,
        allow_recovery_reschedule: bool = False,
        decision_epoch_id: Optional[int] = None,
    ) -> List[Sortie]:
        if not drone.is_idle() or drone.position != vehicle_position:
            return []
        stats = self._candidate_stats(drone, now, available_tasks, decision_epoch_id)
        cache_before = dict(self._cache_counts)
        started = perf_counter()
        if self.candidate_optimization_enabled:
            result = self._generate_for_drone_optimized(
                drone, vehicle_position, now, available_tasks, max_tasks, top_k,
                no_priority, recovery_anchor_node, vehicle_queue_time_s,
                allow_recovery_reschedule, stats,
            )
        else:
            result = self._generate_for_drone_baseline(
                drone, vehicle_position, now, available_tasks, max_tasks, top_k,
                no_priority, recovery_anchor_node, vehicle_queue_time_s,
                allow_recovery_reschedule, stats,
            )
        stats["retained_sortie_count"] = len(result)
        stats["candidate_generation_total_ms"] = (perf_counter() - started) * 1000.0
        for key in self._cache_counts:
            stats[key] = int(self._cache_counts[key] - cache_before[key])
        self.candidate_generation_log.append(stats)
        self._generation_serial += 1
        return result

    def _generate_for_drone_baseline(
        self,
        drone: Drone,
        vehicle_position: int,
        now: float,
        available_tasks: Sequence[Task],
        max_tasks: Optional[int] = None,
        top_k: Optional[int] = None,
        no_priority: bool = False,
        recovery_anchor_node: Optional[int] = None,
        vehicle_queue_time_s: float = 0.0,
        allow_recovery_reschedule: bool = False,
        stats: Optional[Dict[str, Any]] = None,
    ) -> List[Sortie]:
        if not drone.is_idle() or drone.position != vehicle_position:
            return []
        top_k = int(self.sortie_cfg.get("top_k_per_drone", 8) if top_k is None else top_k)
        max_tasks = int(self.sortie_cfg.get("max_tasks_per_sortie", 3) if max_tasks is None else max_tasks)
        if not bool(self.sortie_cfg.get("allow_multi_task", True)):
            max_tasks = 1
        # Keep candidate enumeration controlled.
        tasks = [t for t in available_tasks if t.executor in {"drone", "both"}]
        if bool(self.ablation_cfg.get("priority_urgency_score", True)):
            tasks = sorted(
                tasks,
                key=lambda t: (
                    -(1 if no_priority else t.priority),
                    t.deadline,
                    self.network.get_drone_time(vehicle_position, t.start_node),
                ),
            )
        else:
            tasks = sorted(tasks, key=lambda t: (self.network.get_drone_time(vehicle_position, t.start_node), t.id))
        tasks = tasks[: int(self.sortie_cfg.get("candidate_pool_tasks", 12))]
        if stats is not None:
            stats["prefilter_tasks_after"] = len(tasks)
        recovery_anchor_node = int(vehicle_position if recovery_anchor_node is None else recovery_anchor_node)
        candidates: List[Sortie] = []
        for length in range(1, max_tasks + 1):
            for subset in combinations(tasks, length):
                # Only permute small subsets. Sort by deadlines first, then optionally test reverse for local variety.
                orders = [tuple(sorted(subset, key=lambda t: t.deadline))]
                if length > 1:
                    rev = tuple(reversed(orders[0]))
                    if rev != orders[0]:
                        orders.append(rev)
                if length <= 3:
                    # Include a few permutations with high score potential.
                    for p in permutations(subset):
                        if p not in orders and len(orders) < 4:
                            orders.append(p)
                for order in orders:
                    if stats is not None:
                        stats[{1: "single_task_sequence_count", 2: "double_task_sequence_count", 3: "triple_task_sequence_count"}.get(length, "triple_task_sequence_count")] += 1
                    rec_nodes = self._recovery_candidates(order)
                    if stats is not None:
                        stats["recovery_nodes_before"] += len(rec_nodes)
                        stats["recovery_nodes_after"] += len(rec_nodes)
                    for rec in rec_nodes:
                        evaluate_started = perf_counter()
                        s = self.evaluate(
                            drone,
                            vehicle_position,
                            now,
                            list(order),
                            rec,
                            no_priority=no_priority,
                            vehicle_anchor_node=recovery_anchor_node,
                            vehicle_queue_time_s=vehicle_queue_time_s,
                        )
                        if stats is not None:
                            stats["full_evaluation_count"] += 1
                            stats["sortie_evaluate_time_ms"] += (perf_counter() - evaluate_started) * 1000.0
                        if s.feasible or (allow_recovery_reschedule and s.base_feasible):
                            candidates.append(s)
                            if stats is not None:
                                stats["feasible_sortie_count"] += 1
        if stats is not None:
            stats["cheap_candidate_count"] = len(candidates)
        # Diversity: avoid returning only same first task.
        candidates.sort(key=lambda s: s.score, reverse=True)
        diverse: List[Sortie] = []
        seen_signature = set()
        for s in candidates:
            sig = (tuple(s.task_ids), s.recovery_node)
            first_sig = s.task_ids[0]
            if sig in seen_signature:
                if stats is not None:
                    stats["duplicate_candidate_count"] += 1
                continue
            if sum(1 for x in diverse if x.task_ids and x.task_ids[0] == first_sig) >= max(2, top_k // 3):
                continue
            diverse.append(s)
            seen_signature.add(sig)
            if len(diverse) >= top_k:
                break
        return diverse

    def _prefix_lower_bound(
        self, launch_node: int, now: float, tasks: Sequence[Task], drone: Drone, stats: Dict[str, Any]
    ) -> Tuple[Optional[Tuple[float, float, Tuple[int, ...]]], Optional[str]]:
        """Return a conservative no-A* lower bound for an ordered task prefix."""
        current_nodes: Tuple[int, ...] = (int(launch_node),)
        elapsed = float(self.planned_launch_time_s)
        energy = 0.0
        speed = self._flight_parameters()["cruise_speed"] if self.vertical_flight_enabled else self.drone_speed
        minimum_power = min(self.fly_power, self.service_power)
        for task in tasks:
            nodes = self._task_nodes(task)
            connectivity_started = perf_counter()
            connected = self._any_connected(current_nodes, nodes)
            stats["connectivity_filter_time_ms"] += (perf_counter() - connectivity_started) * 1000.0
            if connected is False:
                stats["connectivity_pruned_count"] += 1
                return None, "connectivity"
            distance = min(self._fast_distance(current, node) for current in current_nodes for node in nodes)
            elapsed += distance / max(speed, 1e-9) + self._task_min_service_time(task)
            energy += (distance / max(speed, 1e-9) + self._task_min_service_time(task)) * minimum_power
            if float(now) + elapsed > float(task.deadline) + 1e-9:
                stats["deadline_pruned_count"] += 1
                return None, "deadline"
            if energy > float(drone.battery) + 1e-9:
                stats["energy_pruned_count"] += 1
                return None, "energy"
            current_nodes = nodes
        return (elapsed, energy, current_nodes), None

    def _cached_recovery_nodes(self, tasks: Sequence[Task]) -> Tuple[int, ...]:
        key = tuple(int(task.id) for task in tasks)
        cached = self._recovery_candidate_cache.get(key)
        if cached is None:
            cached = tuple(int(node) for node in self._recovery_candidates(tasks))
            self._recovery_candidate_cache[key] = cached
        return cached

    def _cheap_recovery_candidates(
        self,
        tasks: Sequence[Task],
        prefix: Tuple[float, float, Tuple[int, ...]],
        recovery_anchor_node: int,
        vehicle_queue_time_s: float,
        stats: Dict[str, Any],
    ) -> List[Tuple[float, int]]:
        started = perf_counter()
        raw = self._cached_recovery_nodes(tasks)
        stats["recovery_nodes_before"] += len(raw)
        if not raw:
            stats["no_recovery_pruned_count"] += 1
            return []
        elapsed, energy, last_nodes = prefix
        speed = self._flight_parameters()["cruise_speed"] if self.vertical_flight_enabled else self.drone_speed
        flight_power = self._flight_parameters()["cruise_power"] if self.vertical_flight_enabled else self.fly_power
        ranked: List[Tuple[float, int]] = []
        for recovery in raw:
            connectivity_started = perf_counter()
            connected = self._any_connected(last_nodes, (int(recovery),))
            stats["connectivity_filter_time_ms"] += (perf_counter() - connectivity_started) * 1000.0
            if connected is False:
                stats["connectivity_pruned_count"] += 1
                continue
            leg_distance = min(self._fast_distance(last, int(recovery)) for last in last_nodes)
            uav_eta = elapsed + leg_distance / max(speed, 1e-9)
            cav_eta = float(vehicle_queue_time_s) + self._cached_cav_eta(recovery_anchor_node, int(recovery))
            if not math.isfinite(cav_eta):
                continue
            estimated_energy = energy + leg_distance / max(speed, 1e-9) * flight_power
            if estimated_energy > float(self.agent_cfg.get("drone_battery_capacity", math.inf)) + 1e-9:
                stats["energy_pruned_count"] += 1
                continue
            priority_sum = sum(float(task.priority) for task in tasks)
            urgency = sum(max(0.0, 1.0 - (float(task.deadline) - float(stats["simulation_time"])) / max(1.0, float(self.cfg["simulation"].get("horizon_s", 3600)))) for task in tasks)
            cheap_score = (
                8.0 * priority_sum + 5.0 * urgency + 0.6 * max(0, len(tasks) - 1)
                - 0.01 * max(uav_eta, cav_eta) - 0.002 * estimated_energy
                - 0.01 * abs(uav_eta - cav_eta)
            )
            ranked.append((float(cheap_score), int(recovery)))
        limit = max(1, int(self.candidate_cfg.get("max_recovery_nodes_per_sequence", 3)))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        ranked = ranked[:limit]
        stats["recovery_nodes_after"] += len(ranked)
        if not ranked:
            stats["no_recovery_pruned_count"] += 1
        stats["recovery_prefilter_time_ms"] += (perf_counter() - started) * 1000.0
        return ranked

    @staticmethod
    def _sortie_rank_key(sortie: Sortie) -> Tuple[Any, ...]:
        return (
            -round(float(sortie.score), 12),
            -float(getattr(sortie, "_candidate_tie_priority", 0.0)),
            float(getattr(sortie, "_candidate_tie_deadline", math.inf)),
            tuple(int(task_id) for task_id in sortie.task_ids),
            int(sortie.recovery_node), str(sortie.drone_id),
        )

    def _bounded_insert_sortie(self, pool: List[Sortie], sortie: Sortie, top_k: int) -> None:
        """Keep only the bounded, deterministic Top-K sortie objects in memory."""
        first_task = int(sortie.task_ids[0]) if sortie.task_ids else -1
        per_first_limit = max(2, top_k // 3)
        same_first = [item for item in pool if item.task_ids and int(item.task_ids[0]) == first_task]
        if len(same_first) >= per_first_limit:
            worst_same = max(same_first, key=self._sortie_rank_key)
            if self._sortie_rank_key(sortie) >= self._sortie_rank_key(worst_same):
                return
            pool.remove(worst_same)
        if len(pool) < top_k:
            pool.append(sortie)
            return
        worst = max(pool, key=self._sortie_rank_key)
        if self._sortie_rank_key(sortie) < self._sortie_rank_key(worst):
            pool.remove(worst)
            pool.append(sortie)

    def _generate_for_drone_optimized(
        self,
        drone: Drone,
        vehicle_position: int,
        now: float,
        available_tasks: Sequence[Task],
        max_tasks: Optional[int],
        top_k: Optional[int],
        no_priority: bool,
        recovery_anchor_node: Optional[int],
        vehicle_queue_time_s: float,
        allow_recovery_reschedule: bool,
        stats: Dict[str, Any],
    ) -> List[Sortie]:
        top_k = int(self.candidate_cfg.get("final_top_k_per_uav", self.sortie_cfg.get("top_k_per_drone", 12)) if top_k is None else top_k)
        max_tasks = int(self.sortie_cfg.get("max_tasks_per_sortie", 3) if max_tasks is None else max_tasks)
        if not bool(self.sortie_cfg.get("allow_multi_task", True)):
            max_tasks = 1
        max_tasks = min(3, max(1, max_tasks))
        tasks = self._prefilter_tasks(drone, vehicle_position, now, available_tasks, stats)
        protected_task_ids = {
            int(task.id) for task in tasks
            if self._task_is_high_priority_or_urgent(task, now)
            and bool(self.candidate_cfg.get("preserve_high_priority_tasks", True) or self.candidate_cfg.get("preserve_urgent_tasks", True))
        }
        recovery_anchor_node = int(vehicle_position if recovery_anchor_node is None else recovery_anchor_node)
        sequence_started = perf_counter()
        cheap: List[Tuple[float, Tuple[int, Tuple[int, ...], int], Tuple[Task, ...], int]] = []
        seen_sequences: set[Tuple[int, ...]] = set()
        seen_candidates: set[Tuple[str, int, Tuple[int, ...], int]] = set()
        frontier: List[Tuple[Task, ...]] = [tuple()]
        for depth in range(1, max_tasks + 1):
            next_frontier: List[Tuple[Task, ...]] = []
            for prefix_tasks in frontier:
                used = {int(task.id) for task in prefix_tasks}
                for task in tasks:
                    if int(task.id) in used:
                        continue
                    sequence = (*prefix_tasks, task)
                    sequence_key = tuple(int(item.id) for item in sequence)
                    if sequence_key in seen_sequences:
                        stats["duplicate_candidate_count"] += 1
                        continue
                    seen_sequences.add(sequence_key)
                    lower, _reason = self._prefix_lower_bound(vehicle_position, now, sequence, drone, stats)
                    if lower is None:
                        continue
                    stats[{1: "single_task_sequence_count", 2: "double_task_sequence_count", 3: "triple_task_sequence_count"}[depth]] += 1
                    recovery_options = self._cheap_recovery_candidates(
                        sequence, lower, recovery_anchor_node, vehicle_queue_time_s, stats
                    )
                    if not recovery_options:
                        continue
                    for cheap_score, recovery in recovery_options:
                        signature = (str(drone.id), int(vehicle_position), sequence_key, int(recovery))
                        if signature in seen_candidates:
                            stats["duplicate_candidate_count"] += 1
                            continue
                        seen_candidates.add(signature)
                        cheap.append((cheap_score, (int(vehicle_position), sequence_key, int(recovery)), sequence, int(recovery)))
                    if depth < max_tasks:
                        next_frontier.append(sequence)
            frontier = next_frontier
        stats["sequence_generation_time_ms"] += (perf_counter() - sequence_started) * 1000.0
        cheap.sort(key=lambda row: (-row[0], row[1][1], row[1][2], str(drone.id)))
        stats["cheap_candidate_count"] = len(cheap)
        evaluation_limit = max(1, int(self.candidate_cfg.get("max_full_evaluations_per_uav", 100)))
        # Each protected task gets its best cheap candidate into the complete
        # evaluation budget.  The remainder stays globally score-ranked, so
        # high priority or urgent work cannot disappear merely at the budget
        # boundary.
        selected_cheap: List[Tuple[float, Tuple[int, Tuple[int, ...], int], Tuple[Task, ...], int]] = []
        selected_cheap_keys = set()
        for protected_id in sorted(protected_task_ids):
            candidate = next((row for row in cheap if any(int(task.id) == protected_id for task in row[2])), None)
            if candidate is not None and candidate[1] not in selected_cheap_keys:
                selected_cheap.append(candidate)
                selected_cheap_keys.add(candidate[1])
        for candidate in cheap:
            if len(selected_cheap) >= evaluation_limit:
                break
            if candidate[1] not in selected_cheap_keys:
                selected_cheap.append(candidate)
                selected_cheap_keys.add(candidate[1])
        stats["full_evaluation_budget_truncated_count"] = len(cheap) - len(selected_cheap)
        cheap = selected_cheap
        bounded: List[Sortie] = []
        protected_best: Dict[int, Sortie] = {}
        for _cheap_score, _key, sequence, recovery in cheap:
            evaluate_started = perf_counter()
            sortie = self.evaluate(
                drone, vehicle_position, now, list(sequence), recovery,
                no_priority=no_priority, vehicle_anchor_node=recovery_anchor_node,
                vehicle_queue_time_s=vehicle_queue_time_s,
            )
            stats["full_evaluation_count"] += 1
            stats["sortie_evaluate_time_ms"] += (perf_counter() - evaluate_started) * 1000.0
            if sortie.feasible or (allow_recovery_reschedule and sortie.base_feasible):
                sortie._candidate_tie_priority = max(float(task.priority) for task in sequence)
                sortie._candidate_tie_deadline = min(float(task.deadline) for task in sequence)
                stats["feasible_sortie_count"] += 1
                self._bounded_insert_sortie(bounded, sortie, top_k)
                for task in sequence:
                    task_id = int(task.id)
                    if task_id in protected_task_ids:
                        previous = protected_best.get(task_id)
                        if previous is None or self._sortie_rank_key(sortie) < self._sortie_rank_key(previous):
                            protected_best[task_id] = sortie
        # Reserve one best feasible sortie per protected task before filling
        # the remaining fixed-size pool.  This does not change the 12-slot
        # ceiling and falls back naturally if a task is truly infeasible.
        retained: List[Sortie] = []
        retained_signatures = set()
        for sortie in sorted(protected_best.values(), key=self._sortie_rank_key):
            signature = (str(sortie.drone_id), tuple(int(task_id) for task_id in sortie.task_ids), int(sortie.recovery_node))
            if signature not in retained_signatures and len(retained) < top_k:
                retained.append(sortie)
                retained_signatures.add(signature)
        for sortie in sorted(bounded, key=self._sortie_rank_key):
            signature = (str(sortie.drone_id), tuple(int(task_id) for task_id in sortie.task_ids), int(sortie.recovery_node))
            if signature not in retained_signatures and len(retained) < top_k:
                retained.append(sortie)
                retained_signatures.add(signature)
        return sorted(retained, key=self._sortie_rank_key)

    def generate_all(
        self,
        drones: Sequence[Drone],
        vehicle_position: int,
        now: float,
        available_tasks: Sequence[Task],
        max_tasks: Optional[int] = None,
        top_k_per_drone: Optional[int] = None,
        no_priority: bool = False,
        recovery_anchor_node: Optional[int] = None,
        vehicle_queue_time_s: float = 0.0,
        allow_recovery_reschedule: bool = False,
        decision_epoch_id: Optional[int] = None,
    ) -> List[Sortie]:
        out: List[Sortie] = []
        self.last_candidates_by_drone = {}
        for d in drones:
            per_drone = self.generate_for_drone(
                    d,
                    vehicle_position,
                    now,
                    available_tasks,
                    max_tasks,
                    top_k_per_drone,
                    no_priority,
                    recovery_anchor_node=recovery_anchor_node,
                    vehicle_queue_time_s=vehicle_queue_time_s,
                    allow_recovery_reschedule=allow_recovery_reschedule,
                    decision_epoch_id=decision_epoch_id,
                )
            self.last_candidates_by_drone[str(d.id)] = list(per_drone)
            out.extend(per_drone)
        return sorted(out, key=lambda s: s.score, reverse=True)

    def _recovery_candidates(self, tasks: Sequence[Task]) -> List[int]:
        if not tasks:
            return []
        k = int(self.sortie_cfg.get("recovery_candidates", 4))
        required_site_nodes = self._required_vehicle_site_nodes(tasks)
        if required_site_nodes:
            return sorted(required_site_nodes)
        center_nodes = self._center_recovery_candidates(k)
        mode = str(self.sortie_cfg.get("recovery_candidate_mode", "center_only")).lower()
        if center_nodes and mode not in {"mixed", "task_near", "task_near_first"}:
            return center_nodes
        task_near_nodes = []
        # Near the final task is often a good rendezvous point.
        final_arc = tasks[-1].target_arcs[-1]
        task_near_nodes.extend(self.network.nearest_nodes_to_arc(final_arc, k=k))
        # Include both endpoints because inspection tasks are direction-free for
        # rendezvous purposes.
        for task in tasks:
            for u, v in task.target_arcs:
                task_near_nodes.extend([int(u), int(v)])
        nodes = task_near_nodes + center_nodes if mode in {"mixed", "task_near", "task_near_first"} else task_near_nodes
        seen, out = set(), []
        for n in nodes:
            if n not in seen:
                out.append(int(n))
                seen.add(int(n))
        return out[:k]

    def _center_recovery_candidates(self, k: int) -> List[int]:
        radius = float(self.sortie_cfg.get("recovery_center_radius_m", 0.0))
        if radius <= 0.0:
            return []
        center = int(self.network.patrol_center)
        ranked = sorted(
            (int(n) for n in self.network.nodes),
            key=lambda n: (self.network.euclidean_distance(center, n), n),
        )
        within = [n for n in ranked if self.network.euclidean_distance(center, n) <= radius + 1e-9]
        if center not in within:
            within.insert(0, center)
        return within[: max(1, int(k))]

    def _required_vehicle_site_nodes(self, tasks: Sequence[Task]) -> set[int]:
        return {
            int(t.metadata.get("vehicle_site_node", t.end_node))
            for t in tasks
            if bool(t.metadata.get("requires_vehicle_onsite", False))
        }

    def _direction_free_service_arcs(self, current: int, task: Task) -> List[Tuple[int, int]]:
        arcs = [(int(u), int(v)) for u, v in task.target_arcs]
        if len(arcs) <= 1:
            if not arcs:
                return []
            u, v = arcs[0]
            to_u = self.network.get_drone_time(current, u)
            to_v = self.network.get_drone_time(current, v)
            return [(v, u)] if to_v < to_u else [(u, v)]

        forward = arcs
        backward = [(v, u) for u, v in reversed(arcs)]
        forward_time = self.network.get_drone_time(current, forward[0][0])
        backward_time = self.network.get_drone_time(current, backward[0][0])
        return backward if backward_time < forward_time else forward

    def evaluate(
        self,
        drone: Drone,
        launch_node: int,
        now: float,
        tasks: Sequence[Task],
        recovery_node: int,
        no_priority: bool = False,
        vehicle_anchor_node: Optional[int] = None,
        vehicle_queue_time_s: float = 0.0,
    ) -> Sortie:
        segments: List[Dict] = []
        required_site_nodes = self._required_vehicle_site_nodes(tasks)
        if len(required_site_nodes) > 1:
            return Sortie(
                drone.id,
                int(launch_node),
                [int(t.id) for t in tasks],
                [int(t.id) for t in tasks],
                int(recovery_node),
                math.inf,
                math.inf,
                math.inf,
                math.inf,
                -1e9,
                False,
                "multiple_vehicle_sites_required",
                [],
                len(tasks),
                int(launch_node if vehicle_anchor_node is None else vehicle_anchor_node),
                float(vehicle_queue_time_s),
                0.0,
                float(drone.battery),
                False,
            )
        if required_site_nodes and int(recovery_node) not in required_site_nodes:
            return Sortie(
                drone.id,
                int(launch_node),
                [int(t.id) for t in tasks],
                [int(t.id) for t in tasks],
                int(recovery_node),
                math.inf,
                math.inf,
                math.inf,
                math.inf,
                -1e9,
                False,
                "vehicle_site_required",
                [],
                len(tasks),
                int(launch_node if vehicle_anchor_node is None else vehicle_anchor_node),
                float(vehicle_queue_time_s),
                0.0,
                float(drone.battery),
                False,
            )
        current = int(launch_node)
        elapsed = float(self.planned_launch_time_s)
        energy = 0.0
        flight = self._flight_parameters()
        if self.vertical_flight_enabled:
            climb_time = flight["height"] / max(flight["up_speed"], 1e-9)
            segments.append(self._vertical_segment("vertical_climb", current, climb_time))
            elapsed += climb_time
            energy += climb_time * flight["up_power"]
        task_ids: List[int] = []
        deadline_violation = False
        for task in tasks:
            service_arcs = self._direction_free_service_arcs(current, task)
            arc_count = max(1, len(service_arcs))
            service_duration = float(task.service_time) / float(arc_count)
            for arc_index, (service_from, service_to) in enumerate(service_arcs):
                t_transfer = self.network.get_drone_time(current, service_from)
                if t_transfer > 0:
                    if self.vertical_flight_enabled:
                        horizontal = self._horizontal_segment(current, service_from)
                        segments.append(horizontal)
                        elapsed += float(horizontal["duration"])
                        energy += float(horizontal["duration"]) * flight["cruise_power"]
                    else:
                        segments.append({"type": "fly", "from": current, "to": service_from, "duration": t_transfer, "task_id": None})
                        elapsed += t_transfer
                        energy += t_transfer * self.fly_power
                is_edge_task = bool(task.metadata.get("edge_task", False))
                if is_edge_task:
                    coverage_distance = float(
                        task.metadata.get("edge_length_m", self.network.arc_length((service_from, service_to)))
                    )
                    coverage_speed = float(
                        flight["cruise_speed"] if self.vertical_flight_enabled
                        else self.network.cfg.get("network", {}).get("drone_speed_mps", 14.0)
                    )
                    coverage_duration = coverage_distance / max(coverage_speed, 1e-9)
                    geometry_wkt = str(task.metadata.get("edge_geometry_wkt", ""))
                    segments.append(
                        {
                            "type": "edge_coverage",
                            "from": service_from,
                            "to": service_to,
                            "duration": coverage_duration,
                            "distance_m": coverage_distance,
                            "task_id": task.id,
                            "edge_id": str(task.metadata.get("edge_id", "")),
                            "selected_orientation": f"{service_from}->{service_to}",
                            "geometry_wkt": geometry_wkt,
                            "coverage_required": True,
                        }
                    )
                    elapsed += coverage_duration
                    energy += coverage_duration * (
                        flight["cruise_power"] if self.vertical_flight_enabled else self.fly_power
                    )
                segments.append(
                    {
                        "type": "service",
                        "from": service_to if is_edge_task else service_from,
                        "to": service_to,
                        "duration": service_duration,
                        "task_id": task.id,
                        "arc_index": arc_index,
                        "arc_count": arc_count,
                        "service_final": arc_index == arc_count - 1,
                        "requires_edge_coverage": is_edge_task,
                    }
                )
                elapsed += service_duration
                energy += service_duration * (flight["service_power"] if self.vertical_flight_enabled else self.service_power)
                current = service_to
            task_ids.append(task.id)
            if now + elapsed > task.deadline + 1e-9:
                deadline_violation = True
        # Return to recovery node.
        t_return = self.network.get_drone_time(current, recovery_node)
        if t_return > 0:
            if self.vertical_flight_enabled:
                horizontal = self._horizontal_segment(current, int(recovery_node))
                segments.append(horizontal)
                elapsed += float(horizontal["duration"])
                energy += float(horizontal["duration"]) * flight["cruise_power"]
            else:
                segments.append({"type": "fly", "from": current, "to": int(recovery_node), "duration": t_return, "task_id": None})
                elapsed += t_return
                energy += t_return * self.fly_power
        if self.vertical_flight_enabled:
            descent_time = flight["height"] / max(flight["down_speed"], 1e-9)
            segments.append(self._vertical_segment("vertical_descent", int(recovery_node), descent_time))
            elapsed += descent_time
            energy += descent_time * flight["down_power"]
        flight_service_energy = float(energy)
        battery_at_eta = float(drone.battery - flight_service_energy)
        anchor = int(launch_node if vehicle_anchor_node is None else vehicle_anchor_node)
        vehicle_leg_time = self.network.get_vehicle_time(anchor, recovery_node)
        vehicle_time = float(vehicle_queue_time_s) + float(vehicle_leg_time)
        if not math.isfinite(vehicle_time):
            return Sortie(
                drone.id, int(launch_node), task_ids, task_ids, int(recovery_node), elapsed,
                vehicle_time, math.inf, energy, -1e9, False, "vehicle_unreachable", segments,
                len(task_ids), anchor, float(vehicle_queue_time_s), flight_service_energy,
                battery_at_eta, False,
            )
        drone_wait = max(0.0, vehicle_time - elapsed)
        vehicle_wait = max(0.0, elapsed - vehicle_time)
        energy += drone_wait * self.hover_power
        finish = now + max(elapsed, vehicle_time)
        battery_capacity = drone.battery
        base_feasible = True
        feasible = True
        reason = "ok"
        if deadline_violation:
            base_feasible = False
            feasible = False
            reason = "deadline"
        if flight_service_energy > battery_capacity + 1e-9:
            base_feasible = False
            feasible = False
            reason = "battery"
        elif energy > battery_capacity + 1e-9:
            feasible = False
            reason = "battery"
        if bool(self.ablation_cfg.get("priority_urgency_score", True)):
            priority_sum = sum((1.0 if no_priority else t.priority) for t in tasks)
            urgency_bonus = sum(max(0.0, 1.0 - (t.deadline - now) / max(1.0, self.cfg["simulation"].get("horizon_s", 3600))) for t in tasks)
        else:
            priority_sum = 0.0
            urgency_bonus = 0.0
        multitask_bonus = 0.6 * max(0, len(tasks) - 1)
        score = 8.0 * priority_sum + 5.0 * urgency_bonus + 3.0 * multitask_bonus - 0.01 * max(elapsed, vehicle_time) - 0.002 * energy - 0.01 * (drone_wait + vehicle_wait)
        return Sortie(
            drone_id=drone.id,
            launch_node=int(launch_node),
            task_ids=task_ids,
            task_sequence=task_ids,
            recovery_node=int(recovery_node),
            drone_time_s=float(elapsed),
            vehicle_time_s=float(vehicle_time),
            finish_time_s=float(finish),
            energy=float(energy),
            score=float(score),
            feasible=bool(feasible),
            reason=reason,
            segments=segments,
            tasks_per_sortie=len(task_ids),
            vehicle_anchor_node=anchor,
            vehicle_queue_time_s=float(vehicle_queue_time_s),
            flight_service_energy=flight_service_energy,
            battery_at_eta=battery_at_eta,
            base_feasible=bool(base_feasible),
            planned_launch_time_s=float(self.planned_launch_time_s),
        )
