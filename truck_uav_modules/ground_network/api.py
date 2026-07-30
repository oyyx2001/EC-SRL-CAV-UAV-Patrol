"""Load the final regional road networks and recover real road paths."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import yaml
from shapely import wkt
from shapely.geometry import LineString, Point


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REGION_CONFIG = yaml.safe_load((PACKAGE_ROOT / "configs" / "regions.yaml").read_text(encoding="utf-8"))
DATA_ROOT = (PACKAGE_ROOT / "configs" / REGION_CONFIG["data_root"]).resolve()
SUPPORTED_REGIONS = set(REGION_CONFIG["regions"])


@dataclass(frozen=True)
class GroundNetwork:
    """Final main-road topology and launch/recovery candidates for one region."""

    region: str
    graph: nx.MultiGraph
    launch_recovery_nodes: gpd.GeoDataFrame
    intersection_nodes: gpd.GeoDataFrame
    supplementary_nodes: gpd.GeoDataFrame
    compressed_edges: gpd.GeoDataFrame

    def nearest_graph_node(self, point: Point) -> str:
        return min(
            self.graph.nodes,
            key=lambda node: point.distance(Point(float(self.graph.nodes[node]["x"]), float(self.graph.nodes[node]["y"]))),
        )


def _region_dir(region: str) -> Path:
    normalized = str(region).strip().lower()
    if normalized not in SUPPORTED_REGIONS:
        raise ValueError(f"Unsupported region {region!r}; choose one of {sorted(SUPPORTED_REGIONS)}.")
    return DATA_ROOT / normalized


def _restore_geometry(graph: nx.MultiGraph) -> nx.MultiGraph:
    for _, data in graph.nodes(data=True):
        geometry = data.get("geometry")
        if isinstance(geometry, str) and geometry:
            data["geometry"] = wkt.loads(geometry)
        point = data.get("geometry")
        if isinstance(point, Point):
            data.setdefault("x", float(point.x))
            data.setdefault("y", float(point.y))
    for _, _, _, data in graph.edges(keys=True, data=True):
        geometry = data.get("geometry")
        if isinstance(geometry, str) and geometry:
            data["geometry"] = wkt.loads(geometry)
    return graph


@lru_cache(maxsize=None)
def load_ground_network(region: str) -> GroundNetwork:
    """Load one region's final, non-fixed-count main-road topology from package data."""

    directory = _region_dir(region)
    files = REGION_CONFIG["regions"][str(region).lower()]
    required = set(files.values()) - {"buildings.geojson", "buildings_local.geojson"}
    missing = sorted(name for name in required if not (directory / name).exists())
    if missing:
        raise FileNotFoundError(f"Missing packaged {region} data: {missing}")
    graph = _restore_geometry(nx.read_graphml(directory / files["major_road_graph_file"], force_multigraph=True))
    return GroundNetwork(
        region=str(region).lower(),
        graph=graph,
        launch_recovery_nodes=gpd.read_file(directory / files["launch_nodes_file"]),
        intersection_nodes=gpd.read_file(directory / files["intersection_nodes_file"]),
        supplementary_nodes=gpd.read_file(directory / files["supplementary_nodes_file"]),
        compressed_edges=gpd.read_file(directory / files["compressed_road_edges_file"]),
    )


def get_launch_recovery_nodes(region: str) -> gpd.GeoDataFrame:
    """Return final intersection and supplementary launch/recovery candidates."""

    return load_ground_network(region).launch_recovery_nodes.copy()


def _edge_data(graph: nx.MultiGraph, source: str, target: str) -> tuple[str, dict[str, Any]]:
    options = graph.get_edge_data(source, target)
    if not options:
        raise nx.NetworkXNoPath(f"No road edge from {source} to {target}.")
    key, data = min(options.items(), key=lambda item: float(item[1].get("travel_time_s", item[1].get("length_m", 1.0))))
    return str(key), data


def _orient(line: LineString, start: Point) -> list[tuple[float, float]]:
    coords = list(line.coords)
    return coords if Point(coords[0]).distance(start) <= Point(coords[-1]).distance(start) else list(reversed(coords))


def shortest_ground_path(region: str, start_node: str, end_node: str) -> dict[str, Any]:
    """Return the shortest real-geometry major-road path between two graph nodes.

    ``start_node`` and ``end_node`` should be final launch-node IDs or any IDs in
    the packaged major-road graph.  Distance is metres and travel time is seconds.
    """

    network = load_ground_network(region)
    source, target = str(start_node), str(end_node)
    if source not in network.graph or target not in network.graph:
        raise KeyError("Both node IDs must exist in the packaged major-road graph.")
    nodes = nx.shortest_path(network.graph, source=source, target=target, weight="travel_time_s")
    coordinates: list[tuple[float, float]] = []
    edge_ids: list[str] = []
    distance_m = travel_time_s = 0.0
    for left, right in zip(nodes[:-1], nodes[1:]):
        edge_id, data = _edge_data(network.graph, left, right)
        line = data.get("geometry")
        if not isinstance(line, LineString):
            line = LineString([(network.graph.nodes[left]["x"], network.graph.nodes[left]["y"]), (network.graph.nodes[right]["x"], network.graph.nodes[right]["y"])])
        oriented = _orient(line, Point(float(network.graph.nodes[left]["x"]), float(network.graph.nodes[left]["y"])))
        coordinates.extend(oriented if not coordinates else oriented[1:])
        edge_ids.append(edge_id)
        distance_m += float(data.get("length_m", line.length))
        travel_time_s += float(data.get("travel_time_s", 0.0))
    geometry = LineString(coordinates) if len(coordinates) >= 2 else Point(float(network.graph.nodes[source]["x"]), float(network.graph.nodes[source]["y"])).buffer(0).boundary
    return {"region": network.region, "node_sequence": [str(node) for node in nodes], "edge_sequence": edge_ids, "geometry": geometry, "distance_m": distance_m, "travel_time_s": travel_time_s}
