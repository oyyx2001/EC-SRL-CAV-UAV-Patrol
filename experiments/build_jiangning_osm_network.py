from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import requests


DRIVABLE_HIGHWAYS = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "service",
}

SPEED_MPS = {
    "motorway": 22.2,
    "trunk": 18.0,
    "primary": 15.0,
    "secondary": 12.0,
    "tertiary": 10.0,
    "unclassified": 8.3,
    "residential": 7.0,
    "living_street": 5.0,
    "service": 5.0,
}


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bbox_from_center(center_lon: float, center_lat: float, width_m: float, height_m: float) -> Tuple[float, float, float, float]:
    lat_delta = height_m / 111_320.0
    lon_delta = width_m / (111_320.0 * math.cos(math.radians(center_lat)))
    south = center_lat - lat_delta / 2.0
    north = center_lat + lat_delta / 2.0
    west = center_lon - lon_delta / 2.0
    east = center_lon + lon_delta / 2.0
    return south, west, north, east


def project_lonlat(lon: float, lat: float, origin_lon: float, origin_lat: float) -> Tuple[float, float]:
    x = (lon - origin_lon) * 111_320.0 * math.cos(math.radians(origin_lat))
    y = (lat - origin_lat) * 111_320.0
    return x, y


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def normalize_highway(value) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value)


def is_oneway(tags: Dict[str, str]) -> int:
    raw = str(tags.get("oneway", "")).lower()
    if raw in {"yes", "true", "1"}:
        return 1
    if raw == "-1":
        return -1
    if normalize_highway(tags.get("highway")) == "motorway":
        return 1
    if str(tags.get("junction", "")).lower() == "roundabout":
        return 1
    return 0


def overpass_query(south: float, west: float, north: float, east: float) -> str:
    highway_pattern = "|".join(sorted(DRIVABLE_HIGHWAYS))
    return f"""
[out:json][timeout:180];
(
  way["highway"~"^({highway_pattern})$"]({south:.7f},{west:.7f},{north:.7f},{east:.7f});
);
(._;>;);
out body;
"""


def fetch_osm(south: float, west: float, north: float, east: float, cache_path: Path, endpoint: str) -> Dict:
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    query = overpass_query(south, west, north, east)
    headers = {"User-Agent": "dynamic-patrol-marl-osm-builder/1.0"}
    response = requests.post(endpoint, data={"data": query}, headers=headers, timeout=240)
    response.raise_for_status()
    data = response.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    time.sleep(1.0)
    return data


def build_full_graph(osm: Dict, south: float, west: float) -> nx.DiGraph:
    nodes = {}
    ways = []
    for element in osm.get("elements", []):
        if element.get("type") == "node":
            lon = float(element["lon"])
            lat = float(element["lat"])
            x, y = project_lonlat(lon, lat, west, south)
            nodes[int(element["id"])] = {"osm_id": int(element["id"]), "lon": lon, "lat": lat, "x": x, "y": y}
        elif element.get("type") == "way":
            tags = element.get("tags", {})
            highway = normalize_highway(tags.get("highway"))
            if highway in DRIVABLE_HIGHWAYS:
                ways.append(element)

    graph = nx.DiGraph()
    for osm_id, attrs in nodes.items():
        graph.add_node(osm_id, **attrs)

    for way in ways:
        tags = way.get("tags", {})
        highway = normalize_highway(tags.get("highway"))
        speed = SPEED_MPS.get(highway, SPEED_MPS["unclassified"])
        oneway = is_oneway(tags)
        way_nodes = [int(n) for n in way.get("nodes", []) if int(n) in graph]
        if oneway == -1:
            way_nodes = list(reversed(way_nodes))
            oneway = 1
        for u, v in zip(way_nodes[:-1], way_nodes[1:]):
            if u == v:
                continue
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            length = haversine_m(u_data["lon"], u_data["lat"], v_data["lon"], v_data["lat"])
            edge_attrs = {
                "length_m": length,
                "vehicle_speed_mps": speed,
                "vehicle_time_s": length / speed,
                "highway": highway,
                "name": tags.get("name", ""),
                "osm_id": int(way["id"]),
                "oneway": bool(oneway),
                "source": "osm",
            }
            graph.add_edge(u, v, **edge_attrs)
            if not oneway:
                reverse_attrs = dict(edge_attrs)
                reverse_attrs["oneway"] = False
                graph.add_edge(v, u, **reverse_attrs)
    return largest_weak_component(graph)


def largest_weak_component(graph: nx.DiGraph) -> nx.DiGraph:
    if graph.number_of_nodes() == 0:
        return graph
    components = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
    return graph.subgraph(components[0]).copy()


def make_undirected(graph: nx.DiGraph) -> nx.Graph:
    undirected = nx.Graph()
    for node, attrs in graph.nodes(data=True):
        undirected.add_node(node, **attrs)
    for u, v, attrs in graph.edges(data=True):
        length = float(attrs.get("length_m", 0.0))
        if undirected.has_edge(u, v):
            if length >= undirected[u][v].get("length_m", math.inf):
                continue
        undirected.add_edge(u, v, **attrs)
    return undirected


def path_has_direction(graph: nx.DiGraph, path: Sequence[int]) -> bool:
    return all(graph.has_edge(int(u), int(v)) for u, v in zip(path[:-1], path[1:]))


def edge_attrs_for_path(graph: nx.DiGraph, path: Sequence[int]) -> Dict:
    lengths = []
    speeds = []
    highways = []
    names = []
    osm_ids = []
    for u, v in zip(path[:-1], path[1:]):
        attrs = graph[int(u)][int(v)]
        lengths.append(float(attrs.get("length_m", 0.0)))
        speeds.append(float(attrs.get("vehicle_speed_mps", SPEED_MPS["unclassified"])))
        highways.append(str(attrs.get("highway", "")))
        name = str(attrs.get("name", ""))
        if name:
            names.append(name)
        osm_ids.append(int(attrs.get("osm_id", 0)))
    length = sum(lengths)
    speed = sum(s * l for s, l in zip(speeds, lengths)) / max(length, 1.0)
    return {
        "length_m": length,
        "vehicle_speed_mps": speed,
        "vehicle_time_s": length / max(speed, 1.0),
        "highway": max(set(highways), key=highways.count) if highways else "",
        "name": names[0] if names else "",
        "path_osm_ids": osm_ids,
        "path_node_ids": [int(n) for n in path],
        "source": "osm_simplified",
    }


def edge_attrs_for_any_path(graph: nx.DiGraph, path: Sequence[int]) -> Dict:
    lengths = []
    speeds = []
    highways = []
    names = []
    osm_ids = []
    for u, v in zip(path[:-1], path[1:]):
        if graph.has_edge(int(u), int(v)):
            attrs = graph[int(u)][int(v)]
        else:
            attrs = graph[int(v)][int(u)]
        lengths.append(float(attrs.get("length_m", 0.0)))
        speeds.append(float(attrs.get("vehicle_speed_mps", SPEED_MPS["unclassified"])))
        highways.append(str(attrs.get("highway", "")))
        name = str(attrs.get("name", ""))
        if name:
            names.append(name)
        osm_ids.append(int(attrs.get("osm_id", 0)))
    length = sum(lengths)
    speed = sum(s * l for s, l in zip(speeds, lengths)) / max(length, 1.0)
    return {
        "length_m": length,
        "vehicle_speed_mps": speed,
        "vehicle_time_s": length / max(speed, 1.0),
        "highway": max(set(highways), key=highways.count) if highways else "",
        "name": names[0] if names else "",
        "path_osm_ids": osm_ids,
        "path_node_ids": [int(n) for n in path],
        "source": "osm_sim_abstracted",
    }


def simplify_graph(graph: nx.DiGraph, keep_nodes: Iterable[int] | None = None) -> nx.DiGraph:
    undirected = make_undirected(graph)
    if keep_nodes is None:
        keep = {n for n in undirected.nodes if undirected.degree(n) != 2}
    else:
        keep = {int(n) for n in keep_nodes if n in undirected}
    if not keep:
        keep = set(undirected.nodes)

    simplified = nx.DiGraph()
    for node in keep:
        simplified.add_node(node, **graph.nodes[node])

    visited = set()
    for start in keep:
        for first in undirected.neighbors(start):
            key = tuple(sorted((start, first)))
            if key in visited and keep_nodes is None:
                continue
            path = [start, first]
            prev = start
            current = first
            while current not in keep:
                candidates = [n for n in undirected.neighbors(current) if n != prev]
                if len(candidates) != 1:
                    break
                prev, current = current, candidates[0]
                path.append(current)
            if path[-1] not in keep or path[-1] == start:
                continue
            for u, v in zip(path[:-1], path[1:]):
                visited.add(tuple(sorted((u, v))))
            end = path[-1]
            if path_has_direction(graph, path):
                simplified.add_edge(start, end, **edge_attrs_for_path(graph, path))
            reverse_path = list(reversed(path))
            if path_has_direction(graph, reverse_path):
                simplified.add_edge(end, start, **edge_attrs_for_path(graph, reverse_path))
    return largest_weak_component(simplified)


def select_sim_nodes(graph: nx.DiGraph, target_nodes: int, center_lon: float, center_lat: float) -> List[int]:
    if graph.number_of_nodes() <= target_nodes:
        return list(graph.nodes)
    undirected = graph.to_undirected()
    center = min(
        graph.nodes,
        key=lambda n: haversine_m(center_lon, center_lat, float(graph.nodes[n]["lon"]), float(graph.nodes[n]["lat"])),
    )
    candidates = list(graph.nodes)
    degree = dict(undirected.degree())
    selected = [center]
    while len(selected) < target_nodes:
        best = None
        best_score = -math.inf
        for node in candidates:
            if node in selected:
                continue
            min_dist = min(
                math.hypot(graph.nodes[node]["x"] - graph.nodes[s]["x"], graph.nodes[node]["y"] - graph.nodes[s]["y"])
                for s in selected
            )
            score = min_dist + 120.0 * degree.get(node, 0)
            if score > best_score:
                best = node
                best_score = score
        selected.append(int(best))
    return selected


def select_uniform_nodes(
    graph: nx.DiGraph,
    target_nodes: int,
    center_lon: float,
    center_lat: float,
    min_spacing_m: float,
) -> List[int]:
    """Select road nodes with farthest-point sampling to avoid dense clusters."""
    if graph.number_of_nodes() <= target_nodes:
        return list(graph.nodes)

    candidates = list(graph.nodes)
    center = min(
        candidates,
        key=lambda n: haversine_m(center_lon, center_lat, float(graph.nodes[n]["lon"]), float(graph.nodes[n]["lat"])),
    )
    selected = [int(center)]
    min_distance = {
        int(node): math.hypot(
            float(graph.nodes[node]["x"]) - float(graph.nodes[center]["x"]),
            float(graph.nodes[node]["y"]) - float(graph.nodes[center]["y"]),
        )
        for node in candidates
    }

    while len(selected) < target_nodes:
        best = max((n for n in candidates if n not in selected), key=lambda n: min_distance[int(n)], default=None)
        if best is None or min_distance[int(best)] < min_spacing_m:
            break
        best = int(best)
        selected.append(best)
        bx = float(graph.nodes[best]["x"])
        by = float(graph.nodes[best]["y"])
        for node in candidates:
            node = int(node)
            distance = math.hypot(float(graph.nodes[node]["x"]) - bx, float(graph.nodes[node]["y"]) - by)
            if distance < min_distance[node]:
                min_distance[node] = distance
    return selected


def build_uniform_sim_graph(graph: nx.DiGraph, selected_nodes: Sequence[int]) -> nx.DiGraph:
    """Contract neighboring road-network regions into a sparse, connected graph."""
    undirected = make_undirected(graph)
    selected = [int(n) for n in selected_nodes if n in undirected]
    if not selected:
        return nx.DiGraph()

    _, source_paths = nx.multi_source_dijkstra(undirected, selected, weight="length_m")
    owner = {int(node): int(path[0]) for node, path in source_paths.items()}
    adjacent_regions = {
        tuple(sorted((owner[int(u)], owner[int(v)])))
        for u, v in undirected.edges
        if owner[int(u)] != owner[int(v)]
    }

    sim = nx.DiGraph()
    id_map = {old_id: new_id for new_id, old_id in enumerate(selected)}
    for old_id, new_id in id_map.items():
        attrs = dict(graph.nodes[old_id])
        attrs["osm_id"] = int(old_id)
        sim.add_node(new_id, **attrs)

    for source, target in sorted(adjacent_regions):
        path = nx.shortest_path(undirected, source, target, weight="length_m")
        attrs = edge_attrs_for_any_path(graph, path)
        attrs["source"] = "osm_uniform_contracted"
        sim.add_edge(id_map[source], id_map[target], **attrs)
        reverse_attrs = edge_attrs_for_any_path(graph, list(reversed(path)))
        reverse_attrs["source"] = "osm_uniform_contracted"
        sim.add_edge(id_map[target], id_map[source], **reverse_attrs)
    return largest_weak_component(sim)


def build_sim_graph(graph: nx.DiGraph, selected_nodes: Sequence[int]) -> nx.DiGraph:
    undirected = make_undirected(graph)
    selected = [int(n) for n in selected_nodes if n in graph]
    complete = nx.Graph()
    for node in selected:
        complete.add_node(node)

    paths = {}
    for source in selected:
        lengths, path_map = nx.single_source_dijkstra(undirected, source, weight="length_m")
        nearest = sorted(
            (target for target in selected if target != source and target in lengths),
            key=lambda target: lengths[target],
        )[:4]
        for target in nearest:
            key = tuple(sorted((source, target)))
            paths[key] = path_map[target]
            complete.add_edge(source, target, weight=lengths[target])

    if complete.number_of_edges() == 0:
        return simplify_graph(graph)
    keep_edges = set(tuple(sorted(e)) for e in nx.minimum_spanning_tree(complete, weight="weight").edges())
    for node in selected:
        for _, target in sorted(complete.edges(node), key=lambda e: complete[e[0]][e[1]]["weight"])[:2]:
            keep_edges.add(tuple(sorted((node, target))))

    sim = nx.DiGraph()
    id_map = {old_id: new_id for new_id, old_id in enumerate(selected)}
    for old_id, new_id in id_map.items():
        attrs = dict(graph.nodes[old_id])
        attrs["osm_id"] = int(old_id)
        sim.add_node(new_id, **attrs)
    for u, v in keep_edges:
        path = paths.get(tuple(sorted((u, v))))
        if not path:
            continue
        if path[0] != u:
            path = list(reversed(path))
        attrs = edge_attrs_for_any_path(graph, path)
        sim.add_edge(id_map[u], id_map[v], **attrs)
        sim.add_edge(id_map[v], id_map[u], **dict(attrs))
    return largest_weak_component(sim)


def to_serializable(graph: nx.DiGraph, metadata: Dict) -> Dict:
    nodes = []
    for node, attrs in graph.nodes(data=True):
        row = {"id": int(node)}
        for key, value in attrs.items():
            if isinstance(value, (int, float, str, bool)):
                row[key] = value
        nodes.append(row)
    edges = []
    for u, v, attrs in graph.edges(data=True):
        row = {"u": int(u), "v": int(v)}
        for key, value in attrs.items():
            if isinstance(value, (int, float, str, bool)):
                row[key] = value
            elif key in {"path_osm_ids", "path_node_ids"}:
                row[key] = [int(x) for x in value]
        edges.append(row)
    return {"metadata": metadata, "nodes": nodes, "edges": edges}


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def plot_graphs(full: nx.DiGraph, sim: nx.DiGraph, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    for u, v in full.edges:
        ax.plot(
            [full.nodes[u]["x"], full.nodes[v]["x"]],
            [full.nodes[u]["y"], full.nodes[v]["y"]],
            color="#b8c0c8",
            linewidth=0.45,
            alpha=0.55,
            zorder=1,
        )
    for u, v, attrs in sim.edges(data=True):
        path_nodes = [int(n) for n in attrs.get("path_node_ids", []) if int(n) in full.nodes]
        if len(path_nodes) >= 2:
            xs = [full.nodes[n]["x"] for n in path_nodes]
            ys = [full.nodes[n]["y"] for n in path_nodes]
        else:
            xs = [sim.nodes[u]["x"], sim.nodes[v]["x"]]
            ys = [sim.nodes[u]["y"], sim.nodes[v]["y"]]
        ax.plot(xs, ys, color="#c43c39", linewidth=1.25, alpha=0.82, zorder=3)
    ax.scatter([sim.nodes[n]["x"] for n in sim.nodes], [sim.nodes[n]["y"] for n in sim.nodes], s=16, c="#1f5a96", zorder=4)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_real_map(full: nx.DiGraph, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    for u, v, attrs in full.edges(data=True):
        width = 0.35
        if attrs.get("highway") in {"primary", "secondary", "trunk"}:
            width = 0.75
        ax.plot(
            [full.nodes[u]["x"], full.nodes[v]["x"]],
            [full.nodes[u]["y"], full.nodes[v]["y"]],
            color="#687887",
            linewidth=width,
            alpha=0.72,
            zorder=1,
        )
    ax.set_title("Jiangning OSM real road map")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_sim_graph(sim: nx.DiGraph, out_path: Path, reference: nx.DiGraph | None = None, title: str = "Jiangning simulation graph") -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    reference = reference or sim
    for u, v, attrs in sim.edges(data=True):
        path_nodes = [int(n) for n in attrs.get("path_node_ids", []) if int(n) in reference.nodes]
        xs = [sim.nodes[u]["x"], sim.nodes[v]["x"]]
        ys = [sim.nodes[u]["y"], sim.nodes[v]["y"]]
        if len(path_nodes) >= 2:
            xs = [reference.nodes[n]["x"] for n in path_nodes]
            ys = [reference.nodes[n]["y"] for n in path_nodes]
        ax.plot(xs, ys, color="#c43c39", linewidth=1.45, alpha=0.80, zorder=2)
    node_size = 30 if sim.number_of_nodes() <= 120 else 7
    ax.scatter([sim.nodes[n]["x"] for n in sim.nodes], [sim.nodes[n]["y"] for n in sim.nodes], s=node_size, c="#1f5a96", zorder=3)
    if sim.number_of_nodes() <= 120:
        for node in sim.nodes:
            ax.text(sim.nodes[node]["x"], sim.nodes[node]["y"], str(node), fontsize=5.5, ha="center", va="center", color="white", zorder=4)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_real_vs_sim(full: nx.DiGraph, sim: nx.DiGraph, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for u, v, attrs in full.edges(data=True):
        width = 0.32 if attrs.get("highway") not in {"primary", "secondary", "trunk"} else 0.72
        axes[0].plot(
            [full.nodes[u]["x"], full.nodes[v]["x"]],
            [full.nodes[u]["y"], full.nodes[v]["y"]],
            color="#687887",
            linewidth=width,
            alpha=0.68,
            zorder=1,
        )
    drawn = set()
    for u, v, attrs in sim.edges(data=True):
        key = tuple(sorted((int(u), int(v))))
        if key in drawn:
            continue
        drawn.add(key)
        path_nodes = [int(n) for n in attrs.get("path_node_ids", []) if int(n) in full.nodes]
        xs = [sim.nodes[u]["x"], sim.nodes[v]["x"]]
        ys = [sim.nodes[u]["y"], sim.nodes[v]["y"]]
        if len(path_nodes) >= 2:
            xs = [full.nodes[n]["x"] for n in path_nodes]
            ys = [full.nodes[n]["y"] for n in path_nodes]
        axes[1].plot(xs, ys, color="#c43c39", linewidth=1.05, alpha=0.82, zorder=2)
    axes[1].scatter(
        [sim.nodes[n]["x"] for n in sim.nodes],
        [sim.nodes[n]["y"] for n in sim.nodes],
        s=11,
        c="#1f5a96",
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )
    axes[0].set_title(f"Real OSM road network ({full.number_of_nodes()} polyline nodes)")
    axes[1].set_title(f"Uniform simulation network ({sim.number_of_nodes()} decision nodes)")
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
    fig.tight_layout(w_pad=1.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_config(path: Path, network_json: Path) -> None:
    rel = network_json.relative_to(package_root()).as_posix()
    content = f"""project:
  name: dynamic_patrol_marl_jiangning_osm
  seed: 42
  results_dir: results
simulation:
  horizon_s: 3600
  decision_interval_s: 60
  dt_s: 10
  high_priority_threshold: 8
network:
  type: imported_osm
  path: {rel}
  force_bidirectional: true
  patrol_center: center
  vehicle_speed_mps: 8.3
  drone_speed_mps: 14.0
  risk_seed_shift: 100
  high_risk_fraction: 0.25
agents:
  num_vehicles: 1
  num_drones: 3
  drone_battery_capacity: 1800.0
  drone_fly_power: 1.0
  drone_hover_power: 0.65
  drone_service_power: 0.85
  battery_swap_time_s: 0
  launch_time_s: 0
  recovery_time_s: 0
tasks:
  routine_interval_s: 200
  incident_rate_per_hour: 8
  congestion_rate_per_hour: 10
  congestion_group_max_arcs: 3
  deadline_s:
    routine: 3600
    incident: 420
    congestion: 720
  service_time_s:
    routine: 45
    incident: 60
    congestion: 90
  priority:
    routine: 2
    incident: 10
    congestion: 6
  incident_vehicle_required_prob: 0.30
sortie:
  top_k_per_drone: 8
  max_tasks_per_sortie: 3
  recovery_candidates: 4
  recovery_center_radius_m: 900
  allow_multi_task: true
  candidate_pool_tasks: 12
baseline:
  seeds: 30
  methods: [nearest_task, priority_greedy, edf, score_based, rolling_horizon, static_reopt, random_masked]
training:
  seed_mode: random
reward:
  done_task: 30.0
  priority_bonus: 2.0
  timeout_penalty: -30.0
  undone_task_penalty: -20.0
  response_penalty: -0.01
  energy_penalty: -0.001
  wait_penalty: -0.005
  reasonable_wait_penalty: 0.0
  busy_wait_penalty: 0.0
  routine_wait_penalty: -0.3
  incident_wait_penalty: -2.0
  dynamic_wait_penalty: -1.5
  urgent_wait_penalty: -3.0
  recovery_wait_penalty: -3.0
  recovery_queue_penalty: -0.05
  vehicle_recovery_time_penalty: -0.0005
  urgent_wait_threshold_s: 120
  aoi_penalty: -0.00001
visualization:
  dpi: 160
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Jiangning District OSM road network for the patrol simulator.")
    parser.add_argument("--center-lon", type=float, default=118.8215)
    parser.add_argument("--center-lat", type=float, default=31.9295)
    parser.add_argument("--width-m", type=float, default=5800.0)
    parser.add_argument("--height-m", type=float, default=5800.0)
    parser.add_argument("--target-nodes", type=int, default=60)
    parser.add_argument("--uniform-target-nodes", type=int, default=180)
    parser.add_argument("--min-node-spacing-m", type=float, default=220.0)
    parser.add_argument("--endpoint", default="https://overpass-api.de/api/interpreter")
    parser.add_argument("--out-dir", default=str(package_root() / "data" / "jiangning_osm"))
    args = parser.parse_args()

    if args.width_m > 6000.0 or args.height_m > 6000.0:
        raise ValueError("width-m and height-m must be no more than 6000.")

    out_dir = Path(args.out_dir)
    south, west, north, east = bbox_from_center(args.center_lon, args.center_lat, args.width_m, args.height_m)
    metadata = {
        "place": "Jiangning District, Nanjing, China",
        "center_lon": args.center_lon,
        "center_lat": args.center_lat,
        "bbox": {"south": south, "west": west, "north": north, "east": east},
        "width_m": args.width_m,
        "height_m": args.height_m,
        "source": "OpenStreetMap via Overpass API",
    }

    cache_path = out_dir / "jiangning_baijiahu_overpass_raw.json"
    osm = fetch_osm(south, west, north, east, cache_path, args.endpoint)
    full = build_full_graph(osm, south, west)
    simplified = simplify_graph(full)
    selected = select_sim_nodes(simplified, args.target_nodes, args.center_lon, args.center_lat)
    sim = build_sim_graph(simplified, selected)
    uniform_selected = select_uniform_nodes(
        simplified,
        args.uniform_target_nodes,
        args.center_lon,
        args.center_lat,
        args.min_node_spacing_m,
    )
    uniform_sim = build_uniform_sim_graph(full, uniform_selected)

    full_path = out_dir / "jiangning_osm_full.json"
    simplified_path = out_dir / "jiangning_osm_intersections.json"
    sim_path = out_dir / "jiangning_osm_sim60.json"
    uniform_sim_path = out_dir / "jiangning_osm_uniform.json"
    preview_path = out_dir / "jiangning_osm_preview.png"
    real_map_path = out_dir / "jiangning_osm_real_map.png"
    sim_graph_path = out_dir / "jiangning_osm_sim_graph.png"
    intersection_graph_path = out_dir / "jiangning_osm_intersection_graph.png"
    uniform_graph_path = out_dir / "jiangning_osm_uniform_graph.png"
    comparison_path = out_dir / "jiangning_real_vs_uniform_sim_comparison.png"
    config_path = package_root() / "configs" / "scenario_jiangning_osm.yaml"
    intersection_config_path = package_root() / "configs" / "scenario_jiangning_osm_intersections.yaml"
    uniform_config_path = package_root() / "configs" / "scenario_jiangning_osm_uniform.yaml"

    write_json(full_path, to_serializable(full, {**metadata, "kind": "full_osm_polyline"}))
    write_json(simplified_path, to_serializable(simplified, {**metadata, "kind": "intersection_graph"}))
    write_json(sim_path, to_serializable(sim, {**metadata, "kind": "simulation_graph", "target_nodes": args.target_nodes}))
    write_json(
        uniform_sim_path,
        to_serializable(
            uniform_sim,
            {
                **metadata,
                "kind": "uniform_contracted_simulation_graph",
                "target_nodes": args.uniform_target_nodes,
                "min_node_spacing_m": args.min_node_spacing_m,
            },
        ),
    )
    plot_graphs(full, sim, preview_path, "Jiangning OSM road network: full roads and simulation graph")
    plot_real_map(full, real_map_path)
    plot_sim_graph(sim, sim_graph_path, full, "Jiangning 60-node simulation graph")
    plot_sim_graph(simplified, intersection_graph_path, full, "Jiangning actual intersection simulation graph")
    plot_sim_graph(uniform_sim, uniform_graph_path, full, "Jiangning uniformly simplified simulation graph")
    plot_real_vs_sim(full, uniform_sim, comparison_path)
    write_config(config_path, sim_path)
    write_config(intersection_config_path, simplified_path)
    write_config(uniform_config_path, uniform_sim_path)

    summary = {
        "bbox": metadata["bbox"],
        "full_nodes": full.number_of_nodes(),
        "full_edges": full.number_of_edges(),
        "intersection_nodes": simplified.number_of_nodes(),
        "intersection_edges": simplified.number_of_edges(),
        "sim_nodes": sim.number_of_nodes(),
        "sim_edges": sim.number_of_edges(),
        "uniform_nodes": uniform_sim.number_of_nodes(),
        "uniform_edges": uniform_sim.number_of_edges(),
        "uniform_min_spacing_m": args.min_node_spacing_m,
        "full_json": str(full_path),
        "intersection_json": str(simplified_path),
        "sim_json": str(sim_path),
        "uniform_json": str(uniform_sim_path),
        "preview_png": str(preview_path),
        "real_map_png": str(real_map_path),
        "sim_graph_png": str(sim_graph_path),
        "intersection_graph_png": str(intersection_graph_path),
        "uniform_graph_png": str(uniform_graph_path),
        "uniform_comparison_png": str(comparison_path),
        "config_yaml": str(config_path),
        "intersection_config_yaml": str(intersection_config_path),
        "uniform_config_yaml": str(uniform_config_path),
    }
    write_json(out_dir / "jiangning_osm_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
