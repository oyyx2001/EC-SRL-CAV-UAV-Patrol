"""Runtime adapter for the validated 3-D voxel A* implementation.

This module intentionally reuses ``build_and_benchmark_3d_voxel_astar`` for
coordinate conversion and search; it does not maintain a second A*.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import OrderedDict, Counter, defaultdict
import json, math, time
import numpy as np
import build_and_benchmark_3d_voxel_astar as core

@dataclass
class UAVPathResult:
    feasible: bool; path_type: str; waypoints_xyz: List[Tuple[float,float,float]]
    voxel_path: List[Tuple[int,int,int]]; horizontal_distance_m: float
    vertical_distance_m: float; total_distance_m: float; climb_distance_m: float
    descent_distance_m: float; flight_time_s: float; energy_consumption: float
    planning_time_s: float; expanded_nodes: int; failure_reason: str | None = None
    segment_results: List[Dict[str, Any]] | None = None
    cache_hit_segments: int = 0
    cache_miss_segments: int = 0
    termination_reason: str = "success"
    budget_limit_ms: float | None = None
    budget_exhausted: bool = False

class VoxelPlannerContext:
    def __init__(self, scene: str, cfg: Dict[str, Any]):
        configured = cfg.get("voxel_map_dir")
        default_dir = str(Path(scene) / "voxel_map_3d") if scene.lower() == "xinjiekou" else "processed_3d_voxel_grids"
        base = Path(configured) if configured else Path(__file__).resolve().parents[1] / "data" / default_dir
        if not base.is_absolute():
            base = Path(__file__).resolve().parents[2] / base
        occupancy_path = base / "occupancy.npy"
        metadata_path = base / "metadata.json"
        if not occupancy_path.exists():
            occupancy_path = base / f"{scene}_occupancy_3d_10m.npy"
        if not metadata_path.exists():
            metadata_path = base / f"{scene}_occupancy_3d_10m_metadata.json"
        self.occupancy=np.load(occupancy_path, allow_pickle=False)
        self.metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
        self.cfg=dict(cfg); self.scene=scene; self.workspace=core.AStarWorkspace.create(self.occupancy.shape)
        self.voxel_size_xyz=(float(self.metadata['voxel_size_x_m']), float(self.metadata['voxel_size_y_m']), float(self.metadata['voxel_size_z_m']))
        # The packaged planner currently exposes the 26-neighbour stencil.  We
        # still retain the configured mode in the cache key so a later map or
        # stencil variant cannot reuse an incompatible route.
        self.neighbor_mode=int(self.cfg.get("neighbor_mode", 26))
        self.prevent_corner_cutting=bool(self.cfg.get("prevent_corner_cutting", True))
        self.offsets=core._neighbor_offsets(self.voxel_size_xyz)
        self._offset_lookup = {
            (dz, dr, dc): (intermediates, float(cost))
            for dz, dr, dc, cost, intermediates in self.offsets
        }
        self.map_version=str(
            self.cfg.get("map_version")
            or self.metadata.get("map_version")
            or f"{self.scene}:{self.occupancy.shape}:{self.voxel_size_xyz}"
        )
        self.height_range=(
            float(self.cfg.get("min_operational_altitude_m", 0.0)),
            float(self.cfg.get("max_operational_altitude_m", float("inf"))),
        )
        self.cache: OrderedDict[Tuple[Any, ...], Tuple[List[Tuple[int, int, int]], float, int, float]] = OrderedDict()
        self.cache_max_entries=max(1, int(self.cfg.get("path_cache_max_entries", 512)))
        self.request_log: List[Dict[str, Any]] = []
        self.stats: Counter = Counter()
        self.source_stats: Dict[str, Counter] = defaultdict(Counter)
    def point(self, xyz):
        return core.world_to_voxel(*xyz, self.metadata['transform'], self.metadata['z_min_m'], self.metadata['voxel_size_z_m'], self.occupancy.shape)
    def free_near(self, voxel):
        if self.occupancy[voxel] == 0: return voxel
        radius=int(self.cfg.get('max_goal_search_radius_voxels', 2))
        for r in range(1,radius+1):
            for dz in range(-r,r+1):
             for dr in range(-r,r+1):
              for dc in range(-r,r+1):
               q=(voxel[0]+dz,voxel[1]+dr,voxel[2]+dc)
               if all(0<=q[i]<self.occupancy.shape[i] for i in range(3)) and self.occupancy[q] == 0: return q
        raise ValueError('no_free_voxel_within_radius')
    def cache_key(self, start: Sequence[int], goal: Sequence[int]) -> Tuple[Any, ...]:
        """Return a map- and movement-rule-scoped LRU key for one leg."""
        return (
            self.scene, self.map_version, tuple(map(int, start)), tuple(map(int, goal)),
            self.height_range, self.neighbor_mode, self.prevent_corner_cutting,
            self.voxel_size_xyz,
        )

    def _record_request(
        self, *, source: str, profile: str, start: Sequence[int], goal: Sequence[int],
        cache_hit: bool, actual_search: bool, elapsed_s: float, expanded_nodes: int,
        status: str, cache_key: Tuple[Any, ...],
    ) -> None:
        source=str(source or "unspecified")
        self.stats["astar_request_count"] += 1
        self.stats["astar_actual_search_count"] += int(actual_search)
        self.stats["astar_cache_hit_count"] += int(cache_hit)
        if status == "direct_voxel_connection":
            self.stats["direct_voxel_connection_count"] += 1
        if status == "reverse_cache_hit_success":
            self.stats["astar_reverse_cache_hit_count"] += 1
        if status in {"timeout", "sortie_timeout"}:
            self.stats["astar_timeout_count"] += 1
        if status == "max_nodes_exceeded":
            self.stats["astar_node_budget_exceeded_count"] += 1
        bucket=self.source_stats[source]
        bucket["request_count"] += 1
        bucket["actual_search_count"] += int(actual_search)
        bucket["cache_hit_count"] += int(cache_hit)
        bucket["elapsed_time_s"] += float(elapsed_s)
        bucket["actual_search_time_s"] += float(elapsed_s) if actual_search else 0.0
        bucket["timeout_count"] += int(status in {"timeout", "sortie_timeout"})
        bucket["node_budget_exceeded_count"] += int(status == "max_nodes_exceeded")
        self.request_log.append({
            "source": source, "budget_profile": str(profile),
            "start_voxel": tuple(map(int, start)), "goal_voxel": tuple(map(int, goal)),
            "cache_key": repr(cache_key), "cache_hit": bool(cache_hit),
            "actual_search": bool(actual_search), "elapsed_time_s": float(elapsed_s),
            "expanded_nodes": int(expanded_nodes), "status": str(status),
        })

    def _put_cache(self, key: Tuple[Any, ...], item: Tuple[List[Tuple[int, int, int]], float, int, float]) -> None:
        if not self.cfg.get('use_path_cache', True):
            return
        self.cache[key]=(list(item[0]), float(item[1]), int(item[2]), float(item[3]))
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_max_entries:
            self.cache.popitem(last=False)
            self.stats["astar_cache_eviction_count"] += 1

    def _audit_path(self, path: Sequence[Tuple[int, int, int]]) -> bool:
        """Record explicit occupancy and move-rule validation for reporting."""
        obstacle_count=0
        illegal_move_count=0
        for point in path:
            if any(not 0 <= int(point[i]) < self.occupancy.shape[i] for i in range(3)) or self.occupancy[tuple(point)] != 0:
                obstacle_count += 1
        for current, neighbor in zip(path[:-1], path[1:]):
            if not self._valid_step(tuple(map(int, current)), tuple(map(int, neighbor))):
                illegal_move_count += 1
        self.stats["validated_path_count"] += 1
        self.stats["path_obstacle_voxel_count"] += int(obstacle_count)
        self.stats["path_illegal_move_count"] += int(illegal_move_count)
        return obstacle_count == 0 and illegal_move_count == 0

    def _path_cost(self, path: Sequence[Tuple[int, int, int]]) -> float:
        return float(sum(
            self._offset_lookup[tuple(int(b[i])-int(a[i]) for i in range(3))][1]
            for a, b in zip(path[:-1], path[1:])
        ))

    def budget_profile(self, profile: str) -> Dict[str, Any]:
        budget_cfg=dict(self.cfg.get("planning_budget", {}))
        profiles=budget_cfg.get("profiles", {})
        selected=dict(profiles.get(profile, {}))
        # Backward compatibility for configurations written before profiles.
        return {**budget_cfg, **selected, "enabled": bool(budget_cfg.get("enabled", True))}

    def segment(
        self, start: Sequence[int], goal: Sequence[int], deadline_perf_counter=None,
        max_expanded_nodes=None, *, source: str = "unspecified",
        budget_profile: str = "materialization",
    ):
        start=tuple(map(int, start)); goal=tuple(map(int, goal))
        key=self.cache_key(start, goal)
        started=time.perf_counter()
        cached=self.cache.get(key) if self.cfg.get('use_path_cache',True) else None
        if cached is not None:
            # LRU promotion and a defensive path copy prevent one committed
            # UAV's mutable route from corrupting a later cache hit.
            self.cache.move_to_end(key)
            item=(list(cached[0]), float(cached[1]), int(cached[2]), float(cached[3]))
            if not self._audit_path(item[0]):
                self.cache.pop(key, None)
                self.stats["invalid_cached_path_rejected_count"] += 1
                cached=None
            else:
                self._record_request(
                    source=source, profile=budget_profile, start=start, goal=goal,
                    cache_hit=True, actual_search=False, elapsed_s=time.perf_counter()-started,
                    expanded_nodes=int(item[2]), status="cache_hit_success", cache_key=key,
                )
                return item, True
        reverse_key=self.cache_key(goal, start)
        reverse_cached=self.cache.get(reverse_key) if self.cfg.get('use_path_cache', True) else None
        if reverse_cached is not None:
            self.cache.move_to_end(reverse_key)
            item=(list(reversed(reverse_cached[0])), float(reverse_cached[1]), int(reverse_cached[2]), float(reverse_cached[3]))
            if not self._audit_path(item[0]):
                self.cache.pop(reverse_key, None)
                self.stats["invalid_cached_path_rejected_count"] += 1
                reverse_cached=None
            else:
                self._put_cache(key, item)
                self._record_request(
                    source=source, profile=budget_profile, start=start, goal=goal,
                    cache_hit=True, actual_search=False, elapsed_s=time.perf_counter()-started,
                    expanded_nodes=int(item[2]), status="reverse_cache_hit_success", cache_key=key,
                )
                return item, True
        # This is not a geometric shortcut: every voxel and every 26-neighbour
        # step is validated against the same occupancy and corner-cutting rule
        # as A*.  It simply avoids opening a search when the straight voxel
        # route is already legal.
        direct=self.local_connection(start, goal)
        if direct is not None:
            item=(list(direct), self._path_cost(direct), 0, time.perf_counter()-started)
            assert self._audit_path(item[0])
            self._put_cache(key, item)
            self._record_request(
                source=source, profile=budget_profile, start=start, goal=goal,
                cache_hit=False, actual_search=False, elapsed_s=time.perf_counter()-started,
                expanded_nodes=0, status="direct_voxel_connection", cache_key=key,
            )
            return item, False
        elevated=self.elevated_connection(start, goal)
        if elevated is not None:
            item=(list(elevated), self._path_cost(elevated), 0, time.perf_counter()-started)
            assert self._audit_path(item[0])
            self._put_cache(key, item)
            self._record_request(
                source=source, profile=budget_profile, start=start, goal=goal,
                cache_hit=False, actual_search=False, elapsed_s=time.perf_counter()-started,
                expanded_nodes=0, status="elevated_voxel_connection", cache_key=key,
            )
            self.stats["elevated_voxel_connection_count"] += 1
            return item, False
        profile_cfg=self.budget_profile(budget_profile)
        budget=core.AStarBudget(
            deadline_perf_counter=deadline_perf_counter,
            max_expanded_nodes=max_expanded_nodes,
            check_interval_expansions=int(profile_cfg.get('check_interval_expansions', 64)),
        )
        result=core.plan_3d_path(
            self.occupancy, start, goal, workspace=self.workspace, offsets=self.offsets,
            voxel_size_xyz=self.voxel_size_xyz, prevent_corner_cutting=self.prevent_corner_cutting,
            budget=budget, return_result=True,
            heuristic_weight=float(self.cfg.get("runtime_heuristic_weight", 1.0)),
        )
        elapsed=time.perf_counter()-started
        self._record_request(
            source=source, profile=budget_profile, start=start, goal=goal,
            cache_hit=False, actual_search=True, elapsed_s=elapsed,
            expanded_nodes=int(result.expanded_nodes), status=str(result.termination_reason), cache_key=key,
        )
        if not result.success:
            return result, False
        item=(list(result.voxel_path),float(result.cost_m),int(result.expanded_nodes),float(result.planning_time_ms)/1000.0)
        if not self._audit_path(item[0]):
            raise RuntimeError("astar_returned_invalid_voxel_path")
        self._put_cache(key, item)
        return (list(item[0]), item[1], item[2], item[3]), False

    def path_metrics(self, path: Sequence[Tuple[int, int, int]], uav_state: Dict[str, float]) -> Dict[str, float]:
        """Calculate exact distance, flight time and energy for a valid voxel path."""
        if not path:
            raise ValueError("empty_voxel_path")
        horizontal=vertical=climb=descent=0.0
        for current, neighbor in zip(path[:-1], path[1:]):
            dz, dr, dc=(int(neighbor[i])-int(current[i]) for i in range(3))
            key=(dz, dr, dc)
            if key not in self._offset_lookup or not self._valid_step(tuple(current), tuple(neighbor)):
                raise ValueError("invalid_voxel_path")
            x0,y0,z0=core.voxel_to_world_center(*current, self.metadata['transform'], self.metadata['z_min_m'], self.metadata['voxel_size_z_m'])
            x1,y1,z1=core.voxel_to_world_center(*neighbor, self.metadata['transform'], self.metadata['z_min_m'], self.metadata['voxel_size_z_m'])
            horizontal += math.hypot(x1-x0, y1-y0)
            vertical += abs(z1-z0); climb += max(0.0, z1-z0); descent += max(0.0, z0-z1)
        flight_time=(horizontal/max(float(uav_state['cruise_speed']),1e-9)
                     +climb/max(float(uav_state['climb_speed']),1e-9)
                     +descent/max(float(uav_state['descent_speed']),1e-9))
        energy=(horizontal/max(float(uav_state['cruise_speed']),1e-9)*float(uav_state['cruise_power'])
                +climb/max(float(uav_state['climb_speed']),1e-9)*float(uav_state['climb_power'])
                +descent/max(float(uav_state['descent_speed']),1e-9)*float(uav_state['descent_power']))
        xyz=[core.voxel_to_world_center(*p, self.metadata['transform'], self.metadata['z_min_m'], self.metadata['voxel_size_z_m']) for p in path]
        return {"horizontal_distance_m": horizontal, "vertical_distance_m": vertical,
                "total_distance_m": horizontal+vertical, "climb_distance_m": climb,
                "descent_distance_m": descent, "flight_time_s": flight_time,
                "energy_consumption": energy, "waypoints_xyz": xyz}

    def _valid_step(self, current: Tuple[int, int, int], neighbor: Tuple[int, int, int]) -> bool:
        delta=tuple(int(neighbor[i])-int(current[i]) for i in range(3))
        item=self._offset_lookup.get(delta)
        if item is None or any(not 0 <= int(neighbor[i]) < self.occupancy.shape[i] for i in range(3)):
            return False
        if self.occupancy[neighbor] != 0:
            return False
        if self.prevent_corner_cutting:
            for dz,dr,dc in item[0]:
                if self.occupancy[current[0]+dz, current[1]+dr, current[2]+dc] != 0:
                    return False
        return True

    def local_connection(self, start: Tuple[int, int, int], goal: Tuple[int, int, int]) -> Optional[List[Tuple[int, int, int]]]:
        """Validate a short 26-neighbour voxel connection without running A*."""
        start=tuple(map(int,start)); goal=tuple(map(int,goal))
        steps=max(abs(goal[i]-start[i]) for i in range(3))
        if steps == 0:
            return [start]
        path=[start]
        for index in range(1, steps+1):
            point=tuple(int(round(start[axis]+(goal[axis]-start[axis])*index/steps)) for axis in range(3))
            if point != path[-1]:
                if not self._valid_step(path[-1], point):
                    return None
                path.append(point)
        return path if path[-1] == goal else None

    def elevated_connection(self, start: Tuple[int, int, int], goal: Tuple[int, int, int]) -> Optional[List[Tuple[int, int, int]]]:
        """Return a fully validated climb--cruise--descent voxel corridor.

        This is only a fast feasibility proof.  Every vertical and horizontal
        step is validated with the same 26-neighbour and corner-cutting test as
        the A* expansion; an unavailable corridor falls straight through to
        the normal A* search.
        """
        if not bool(self.cfg.get("enable_validated_elevated_corridor", True)):
            return None
        start=tuple(map(int,start)); goal=tuple(map(int,goal))
        allowed=[]
        for z_index in range(self.occupancy.shape[0]):
            _, _, altitude=core.voxel_to_world_center(
                z_index, 0, 0, self.metadata['transform'], self.metadata['z_min_m'], self.metadata['voxel_size_z_m']
            )
            if self.height_range[0]-1e-9 <= altitude <= self.height_range[1]+1e-9:
                allowed.append(z_index)
        for z_index in sorted(allowed):
            climb_target=(z_index, start[1], start[2])
            descend_start=(z_index, goal[1], goal[2])
            climb=self.local_connection(start, climb_target)
            if climb is None:
                continue
            cruise=self.local_connection(climb_target, descend_start)
            if cruise is None:
                continue
            descent=self.local_connection(descend_start, goal)
            if descent is None:
                continue
            return list(climb)+list(cruise[1:])+list(descent[1:])
        return None

    def snapshot_stats(self) -> Dict[str, Any]:
        for key in ("path_obstacle_voxel_count", "path_illegal_move_count", "validated_path_count"):
            self.stats.setdefault(key, 0)
        return {
            **{key: int(value) for key, value in self.stats.items()},
            "cache_entry_count": len(self.cache),
            "sources": {
                name: {key: (float(value) if key.endswith("_time_s") else int(value)) for key, value in values.items()}
                for name, values in self.source_stats.items()
            },
        }

def plan_selected_sortie_3d(
    launch_position, ordered_tasks, recovery_position, uav_state, planner_context,
    *, source: str = "initial_materialization", budget_profile: str = "materialization",
    start_voxel_override: Optional[Tuple[int, int, int]] = None,
):
    """Plan and concatenate launch -> ordered task observation points -> recovery."""
    started=time.perf_counter()
    try:
        points=[launch_position]+list(ordered_tasks)+[recovery_position]
        vox=[planner_context.free_near(planner_context.point(p)) for p in points]
        if start_voxel_override is not None:
            vox[0]=tuple(map(int, start_voxel_override))
        full=[]; expanded=0; details=[]; hits=misses=0
        budget_cfg=planner_context.budget_profile(budget_profile)
        # A selected sortie can contain several independently valid A* legs.
        # Materialization therefore applies its configured 1 s ceiling to each
        # *actual search*, rather than incorrectly sharing it across all legs.
        # In-flight recovery is a one-leg plan and keeps its strict aggregate
        # 0.5 s deadline.
        enforce_sortie_deadline=bool(budget_cfg.get('enforce_sortie_deadline', True))
        sortie_deadline=(
            started+float(budget_cfg.get('max_sortie_planning_time_ms',800.0))/1000.0
            if budget_cfg.get('enabled',True) and enforce_sortie_deadline else None
        )
        for a,b in zip(vox[:-1],vox[1:]):
            if sortie_deadline is not None and time.perf_counter() >= sortie_deadline:
                return UAVPathResult(feasible=False,path_type='voxel_astar_3d',waypoints_xyz=[],voxel_path=[],horizontal_distance_m=0,vertical_distance_m=0,total_distance_m=0,climb_distance_m=0,descent_distance_m=0,flight_time_s=0,energy_consumption=0,planning_time_s=time.perf_counter()-started,expanded_nodes=expanded,failure_reason='sortie_timeout',segment_results=details,cache_hit_segments=hits,cache_miss_segments=misses,termination_reason='sortie_timeout',budget_limit_ms=float(budget_cfg.get('max_sortie_planning_time_ms',800.0)),budget_exhausted=True)
            seg_limit=float(budget_cfg.get('max_segment_planning_time_ms',200.0))/1000.0 if budget_cfg.get('enabled',True) else None
            deadline=min(time.perf_counter()+seg_limit,sortie_deadline) if seg_limit is not None and sortie_deadline is not None else (time.perf_counter()+seg_limit if seg_limit is not None else None)
            sortie_limited = sortie_deadline is not None and (seg_limit is None or sortie_deadline <= time.perf_counter() + seg_limit)
            item,hit=planner_context.segment(
                a,b,deadline,
                int(budget_cfg.get('max_expanded_nodes_per_segment',100000)) if budget_cfg.get('enabled',True) else None,
                source=source, budget_profile=budget_profile,
            )
            if hasattr(item,'termination_reason'):
                reason='sortie_timeout' if item.termination_reason == 'timeout' and sortie_limited else item.termination_reason
                details.append({'start_voxel':a,'goal_voxel':b,'distance_m':None,'planning_time_s':float(item.planning_time_ms)/1000.0,'expanded_nodes':int(item.expanded_nodes),'cache_hit':False,'astar_status':str(reason),'astar_success':False, 'source': source, 'budget_profile': budget_profile})
                return UAVPathResult(feasible=False,path_type='voxel_astar_3d',waypoints_xyz=[],voxel_path=[],horizontal_distance_m=0,vertical_distance_m=0,total_distance_m=0,climb_distance_m=0,descent_distance_m=0,flight_time_s=0,energy_consumption=0,planning_time_s=item.planning_time_ms/1000.0,expanded_nodes=expanded+item.expanded_nodes,failure_reason=reason,segment_results=details,cache_hit_segments=hits,cache_miss_segments=misses,termination_reason=reason,budget_limit_ms=float(budget_cfg.get('max_segment_planning_time_ms',200.0)),budget_exhausted=reason in {'timeout','sortie_timeout','max_nodes_exceeded'})
            path,cost,n,plan_time=item; full += path if not full else path[1:]; expanded += n
            hits += int(hit); misses += int(not hit); details.append({'start_voxel':a,'goal_voxel':b,'distance_m':cost,'planning_time_s':plan_time,'expanded_nodes':n,'cache_hit':hit,'astar_status':'cache_hit_success' if hit else 'success','astar_success':True,'path_voxel_count':len(path), 'voxel_path': list(path), 'source': source, 'budget_profile': budget_profile})
        xyz=[core.voxel_to_world_center(*p, planner_context.metadata['transform'], planner_context.metadata['z_min_m'], planner_context.metadata['voxel_size_z_m']) for p in full]
        h=sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(xyz[:-1],xyz[1:])); v=sum(abs(b[2]-a[2]) for a,b in zip(xyz[:-1],xyz[1:]))
        climb=sum(max(0,b[2]-a[2]) for a,b in zip(xyz[:-1],xyz[1:])); descent=sum(max(0,a[2]-b[2]) for a,b in zip(xyz[:-1],xyz[1:]))
        f=uav_state; t=h/max(f['cruise_speed'],1e-9)+climb/max(f['climb_speed'],1e-9)+descent/max(f['descent_speed'],1e-9)
        e=(h/max(f['cruise_speed'],1e-9))*f['cruise_power']+(climb/max(f['climb_speed'],1e-9))*f['climb_power']+(descent/max(f['descent_speed'],1e-9))*f['descent_power']
        return UAVPathResult(True,'voxel_astar_3d',xyz,full,h,v,h+v,climb,descent,t,e,time.perf_counter()-started,expanded,None,details,hits,misses)
    except Exception as exc:
        return UAVPathResult(feasible=False,path_type='voxel_astar_3d',waypoints_xyz=[],voxel_path=[],horizontal_distance_m=0,vertical_distance_m=0,total_distance_m=0,climb_distance_m=0,descent_distance_m=0,flight_time_s=0,energy_consumption=0,planning_time_s=time.perf_counter()-started,expanded_nodes=0,failure_reason=str(exc),termination_reason='internal_error')
