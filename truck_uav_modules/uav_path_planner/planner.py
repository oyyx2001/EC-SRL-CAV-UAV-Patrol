"""Fixed-window, height-progressive UAV routing extracted from the local visibility-graph experiment."""

from __future__ import annotations

import itertools
import heapq
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from shapely import make_valid
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from ground_network import load_ground_network, shortest_ground_path
from ground_network.api import DATA_ROOT, REGION_CONFIG


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "configs" / "planner.yaml"
_PROFILE_STATE: dict[str, Any] | None = None


def _profile_add(stage: str, seconds: float) -> None:
    if _PROFILE_STATE is not None:
        _PROFILE_STATE[stage] = float(_PROFILE_STATE.get(stage, 0.0)) + float(seconds) * 1000.0


def _profile_set(key: str, value: Any) -> None:
    if _PROFILE_STATE is not None:
        _PROFILE_STATE[key] = value


def _settings() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _as_point(value: Point | Iterable[float]) -> Point:
    if isinstance(value, Point):
        return value
    values = list(value)
    if len(values) != 2:
        raise ValueError("start_point and goal_point must be shapely Points or (x, y) pairs.")
    return Point(float(values[0]), float(values[1]))


@lru_cache(maxsize=None)
def _buildings(region: str) -> gpd.GeoDataFrame:
    started = time.perf_counter()
    path = DATA_ROOT / region.lower() / REGION_CONFIG["regions"][region.lower()]["buildings_file"]
    buildings = gpd.read_file(path)
    field = str(_settings()["regions"][region.lower()]["building_height_field"])
    if field not in buildings:
        raise ValueError(f"{region} building data has no {field!r} height field.")
    buildings = buildings[buildings.geometry.notna() & ~buildings.geometry.is_empty].copy()
    buildings["geometry"] = buildings.geometry.map(make_valid)
    buildings["_height_m"] = pd.to_numeric(buildings[field], errors="coerce")
    result = buildings[buildings._height_m.notna() & (buildings._height_m > 0)].reset_index(drop=True)
    _profile_add("building_query_ms", time.perf_counter() - started)
    return result


@lru_cache(maxsize=None)
def _obstacles(region: str, height_m: float) -> gpd.GeoDataFrame:
    settings = _settings()["planner"]
    threshold = float(height_m) - float(settings["vertical_clearance_m"])
    obstacles = _buildings(region)[_buildings(region)._height_m >= threshold].copy()
    buffer_started = time.perf_counter()
    obstacles["geometry"] = obstacles.geometry.buffer(float(settings["horizontal_clearance_m"]))
    _profile_add("building_buffer_ms", time.perf_counter() - buffer_started)
    return obstacles[obstacles.geometry.notna() & ~obstacles.geometry.is_empty].reset_index(drop=True)


def _line_is_clear(line: LineString, obstacles: gpd.GeoDataFrame) -> bool:
    indices = obstacles.sindex.query(line, predicate="intersects")
    return not any(obstacles.iloc[int(index)].geometry.intersects(line) for index in indices)


def _endpoint_is_safe(point: Point, obstacles: gpd.GeoDataFrame) -> bool:
    indices = obstacles.sindex.query(point)
    return not any(obstacles.iloc[int(index)].geometry.covers(point) for index in indices)


def _polygon_parts(geometry: Any) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    valid = make_valid(geometry)
    if valid.geom_type == "Polygon":
        return [valid]
    if valid.geom_type == "MultiPolygon":
        return [item for item in valid.geoms if item.area > 0]
    if valid.geom_type == "GeometryCollection":
        return [part for item in valid.geoms for part in _polygon_parts(item)]
    return []


def _local_obstacles(obstacles: gpd.GeoDataFrame, window: Polygon) -> gpd.GeoDataFrame:
    started = time.perf_counter()
    indices = obstacles.sindex.query(window, predicate="intersects")
    _profile_add("building_query_ms", time.perf_counter() - started)
    return obstacles.iloc[np.asarray(indices, dtype=int)].copy().reset_index(drop=True)


def _key_boundary_vertices(polygon: Polygon, simplify_tolerance_m: float, turn_threshold_deg: float) -> list[Point]:
    """Existing experiment's simplified, high-turn exterior waypoint selection."""
    coordinates = list(polygon.exterior.simplify(simplify_tolerance_m, preserve_topology=True).coords)[:-1]
    if len(coordinates) <= 3:
        return [Point(item) for item in coordinates]
    retained: list[Point] = []
    for index, coordinate in enumerate(coordinates):
        previous = np.asarray(coordinates[index - 1], dtype=float)
        current = np.asarray(coordinate, dtype=float)
        following = np.asarray(coordinates[(index + 1) % len(coordinates)], dtype=float)
        first, second = previous - current, following - current
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator <= 1e-9:
            continue
        direction_change = 180.0 - math.degrees(math.acos(float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))))
        cross = float((following - current)[0] * (previous - current)[1] - (following - current)[1] * (previous - current)[0])
        if cross > 0 or direction_change >= turn_threshold_deg:
            retained.append(Point(coordinate))
    return retained if len(retained) >= 3 else [Point(item) for item in coordinates]


def _waypoints(obstacles: list[Polygon], outset_m: float, spacing_m: float, simplify_tolerance_m: float, turn_threshold_deg: float) -> list[Point]:
    started = time.perf_counter()
    points: list[Point] = []
    for obstacle in obstacles:
        for exterior in _polygon_parts(obstacle.buffer(outset_m, quad_segs=2)):
            for point in _key_boundary_vertices(exterior, simplify_tolerance_m, turn_threshold_deg):
                if any(item.covers(point) for item in obstacles):
                    continue
                if not any(point.distance(previous) < spacing_m for previous in points):
                    points.append(point)
    _profile_add("visibility_node_generation_ms", time.perf_counter() - started)
    _profile_set("visibility_node_count", len(points))
    return points


def _astar_shortest_path(
    graph: nx.Graph,
    points: list[Point],
    start_index: int,
    goal_index: int,
    *,
    timeout_s: float,
    started: float,
) -> tuple[list[int] | None, str | None, int, float]:
    """A* on the unchanged visibility graph, using Euclidean h(n, goal)."""
    search_started = time.perf_counter()
    queue: list[tuple[float, float, int, int]] = []
    serial = itertools.count()
    best_g = {int(start_index): 0.0}
    goal_point = points[int(goal_index)]
    heapq.heappush(queue, (float(points[int(start_index)].distance(goal_point)), 0.0, next(serial), int(start_index)))
    parent: dict[int, int | None] = {int(start_index): None}
    expanded = 0
    while queue:
        if time.perf_counter() - started > timeout_s:
            return None, "visibility_timeout", expanded, time.perf_counter() - search_started
        _, g_score, _, current = heapq.heappop(queue)
        if g_score > best_g.get(current, math.inf) + 1e-12:
            continue
        expanded += 1
        if current == int(goal_index):
            search_finished = time.perf_counter()
            path = []
            reconstruction_started = time.perf_counter()
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            _profile_add("astar_search_ms", search_finished - search_started)
            _profile_add("path_reconstruction_ms", time.perf_counter() - reconstruction_started)
            return path, None, expanded, search_finished - search_started
        for neighbor, edge in graph[current].items():
            tentative_g = float(g_score) + float(edge["weight"])
            if tentative_g + 1e-12 >= best_g.get(int(neighbor), math.inf):
                continue
            best_g[int(neighbor)] = tentative_g
            parent[int(neighbor)] = current
            heuristic = math.hypot(goal_point.x - points[int(neighbor)].x, goal_point.y - points[int(neighbor)].y)
            heapq.heappush(queue, (tentative_g + heuristic, tentative_g, next(serial), int(neighbor)))
    return None, "disconnected_visibility_graph", expanded, time.perf_counter() - search_started


def _visibility_path(start: Point, goal: Point, obstacles: gpd.GeoDataFrame, margin_m: float, timeout_s: float) -> tuple[LineString | None, str | None, int, float]:
    """One fixed local complete visibility-graph attempt, followed by line-of-sight smoothing."""

    started = time.perf_counter()
    minx, miny, maxx, maxy = LineString([start, goal]).bounds
    local = _local_obstacles(obstacles, box(minx - margin_m, miny - margin_m, maxx + margin_m, maxy + margin_m))
    _profile_set("building_count", int(len(local)))
    if local.empty:
        return None, "no_local_obstacles", 0, 0.0
    merged = _polygon_parts(unary_union(list(local.geometry)))
    settings = _settings()["planner"]
    points = [start, goal, *_waypoints(merged, float(settings["waypoint_outset_m"]), float(settings["waypoint_minimum_spacing_m"]), float(settings["waypoint_simplify_tolerance_m"]), float(settings["waypoint_turn_threshold_deg"]))]
    if len(points) <= 2:
        return None, "no_valid_waypoints", 0, 0.0
    merged_gdf = gpd.GeoDataFrame(geometry=merged, crs=obstacles.crs)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(points)))
    _profile_set("candidate_edge_count", int(len(points) * (len(points) - 1) // 2))
    edge_started = time.perf_counter()
    for left, right in itertools.combinations(range(len(points)), 2):
        if time.perf_counter() - started > timeout_s:
            _profile_add("visibility_edge_construction_ms", time.perf_counter() - edge_started)
            _profile_set("visibility_edge_count", int(graph.number_of_edges()))
            return None, "visibility_timeout", 0, 0.0
        line = LineString([points[left], points[right]])
        if _line_is_clear(line, merged_gdf):
            graph.add_edge(left, right, weight=float(line.length))
    _profile_add("visibility_edge_construction_ms", time.perf_counter() - edge_started)
    _profile_set("visibility_edge_count", int(graph.number_of_edges()))
    raw_indices, failure, expanded_nodes, astar_search_time_s = _astar_shortest_path(
        graph, points, 0, 1, timeout_s=timeout_s, started=started
    )
    if raw_indices is None:
        return None, failure, expanded_nodes, astar_search_time_s
    raw = [points[index] for index in raw_indices]
    smoothing_started = time.perf_counter()
    smooth = [raw[0]]
    current = 0
    while current < len(raw) - 1:
        if time.perf_counter() - started > timeout_s:
            return None, "smoothing_timeout", expanded_nodes, astar_search_time_s
        next_index = len(raw) - 1
        while next_index > current + 1 and not _line_is_clear(LineString([raw[current], raw[next_index]]), merged_gdf):
            next_index -= 1
        smooth.append(raw[next_index])
        current = next_index
    path = LineString(smooth)
    _profile_add("path_smoothing_ms", time.perf_counter() - smoothing_started)
    return ((path, None, expanded_nodes, astar_search_time_s)
            if all(_line_is_clear(LineString([a, b]), merged_gdf) for a, b in zip(smooth[:-1], smooth[1:]))
            else (None, "collision_after_smoothing", expanded_nodes, astar_search_time_s))


def _road_fallback(region: str, start: Point, goal: Point, heights: tuple[float, ...]) -> tuple[dict[str, Any] | None, dict[str, str]]:
    network = load_ground_network(region)
    road_path = shortest_ground_path(region, network.nearest_graph_node(start), network.nearest_graph_node(goal))
    geometry = road_path["geometry"]
    failures: dict[str, str] = {}
    for height in heights:
        obstacles = _obstacles(region.lower(), float(height))
        if _line_is_clear(geometry, obstacles):
            return road_path | {"selected_height_m": float(height)}, failures
        failures[str(int(height))] = "road_path_blocked_by_building_clearance"
    return None, failures


def plan_uav_path(region: str, start_point: Point | Iterable[float], goal_point: Point | Iterable[float], candidate_heights: tuple[float, ...] = (20, 40, 60, 80, 100, 120), search_buffer_m: float = 50, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Plan direct, local-detour, then real-road fallback routes in ascending fixed height layers."""

    normalized = str(region).lower()
    if normalized not in _settings()["regions"]:
        raise ValueError(f"Unsupported region {region!r}.")
    start, goal = _as_point(start_point), _as_point(goal_point)
    heights = tuple(float(height) for height in candidate_heights)
    if not heights or any(height <= 0 for height in heights):
        raise ValueError("candidate_heights must contain positive values.")
    started = time.perf_counter()
    global _PROFILE_STATE
    _PROFILE_STATE = profile
    if profile is not None:
        for key in (
            "direct_path_check_ms", "building_query_ms", "building_buffer_ms",
            "visibility_node_generation_ms", "visibility_edge_construction_ms",
            "astar_search_ms", "path_reconstruction_ms", "path_smoothing_ms",
        ):
            profile[key] = 0.0
        profile["building_count"] = 0
        profile["candidate_edge_count"] = 0
        profile["visibility_edge_count"] = 0

    def finish(result: dict[str, Any]) -> dict[str, Any]:
        global _PROFILE_STATE
        if profile is not None:
            total_ms = (time.perf_counter() - started) * 1000.0
            profile["total_planning_ms"] = total_ms
            measured = sum(float(profile.get(key, 0.0)) for key in (
                "direct_path_check_ms", "building_query_ms", "building_buffer_ms",
                "visibility_node_generation_ms", "visibility_edge_construction_ms",
                "astar_search_ms", "path_reconstruction_ms", "path_smoothing_ms",
            ))
            profile["other_overhead_ms"] = total_ms - measured
            result["profiling"] = dict(profile)
        _PROFILE_STATE = None
        return result
    failures: dict[str, str] = {}
    tested: list[float] = []
    expanded_nodes = 0
    astar_search_time_s = 0.0
    timeout_s = float(_settings()["planner"]["planning_timeout_s"])
    for height in heights:
        tested.append(height)
        obstacles = _obstacles(normalized, height)
        key = str(int(height))
        direct_started = time.perf_counter()
        endpoint_safe = _endpoint_is_safe(start, obstacles) and _endpoint_is_safe(goal, obstacles)
        direct = LineString([start, goal])
        direct_clear = endpoint_safe and _line_is_clear(direct, obstacles)
        _profile_add("direct_path_check_ms", time.perf_counter() - direct_started)
        if not endpoint_safe:
            failures[key] = "endpoint_inside_building_clearance"
            continue
        if direct_clear:
            return finish({"feasible": True, "path_mode": "direct", "selected_height_m": height, "path_geometry": direct, "distance_m": float(direct.length), "planning_time_s": time.perf_counter() - started, "astar_search_time_s": astar_search_time_s, "expanded_nodes": expanded_nodes, "tested_heights": tested, "failure_reason_by_height": failures})
        path, failure, expanded, search_time_s = _visibility_path(start, goal, obstacles, float(search_buffer_m), timeout_s)
        expanded_nodes += int(expanded)
        astar_search_time_s += float(search_time_s)
        if path is not None:
            return finish({"feasible": True, "path_mode": "local_detour", "selected_height_m": height, "path_geometry": path, "distance_m": float(path.length), "planning_time_s": time.perf_counter() - started, "astar_search_time_s": astar_search_time_s, "expanded_nodes": expanded_nodes, "tested_heights": tested, "failure_reason_by_height": failures})
        failures[key] = failure or "local_detour_failed"
    fallback, fallback_failures = _road_fallback(normalized, start, goal, heights)
    if fallback is not None:
            return finish({"feasible": True, "path_mode": "road_following", "selected_height_m": fallback["selected_height_m"], "path_geometry": fallback["geometry"], "distance_m": float(fallback["distance_m"]), "planning_time_s": time.perf_counter() - started, "astar_search_time_s": astar_search_time_s, "expanded_nodes": expanded_nodes, "tested_heights": tested, "failure_reason_by_height": failures | {f"road_{key}": value for key, value in fallback_failures.items()}, "ground_node_sequence": fallback["node_sequence"], "ground_edge_sequence": fallback["edge_sequence"]})
    return finish({"feasible": False, "path_mode": "infeasible", "selected_height_m": None, "path_geometry": None, "distance_m": None, "planning_time_s": time.perf_counter() - started, "astar_search_time_s": astar_search_time_s, "expanded_nodes": expanded_nodes, "tested_heights": tested, "failure_reason_by_height": failures | {f"road_{key}": value for key, value in fallback_failures.items()}})
