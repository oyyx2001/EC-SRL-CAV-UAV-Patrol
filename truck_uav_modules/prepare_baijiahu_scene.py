"""Prepare Baijiahu assets for the current training and 3-D planner stack.

The source assets are retained.  This script creates the same canonical file
layout used by Xinjiekou and converts the existing 10 m vertical occupancy map
to a 10 m x 10 m x 20 m operational map.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PACKAGE_DATA = PROJECT_ROOT / "truck_uav_modules" / "data"
SCENE_DIR = PACKAGE_DATA / "baijiahu"
SOURCE_VOXEL_DIR = PACKAGE_DATA / "processed_3d_voxel_grids"
TASK_DIR = PROJECT_ROOT / "data" / "baijiahu"

CANONICAL_COPIES = {
    "buildings_local.geojson": "buildings.geojson",
    "compressed_major_road_graph.graphml": "road_network.graphml",
    "compressed_major_road_edges.geojson": "road_edges.geojson",
}


def _copy_canonical_assets() -> None:
    for source_name, target_name in CANONICAL_COPIES.items():
        source = SCENE_DIR / source_name
        target = SCENE_DIR / target_name
        if not source.exists():
            raise FileNotFoundError(source)
        if not target.exists() or source.read_bytes() != target.read_bytes():
            shutil.copy2(source, target)


def _validate_scene_assets() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    graph = nx.read_graphml(SCENE_DIR / "road_network.graphml", force_multigraph=True)
    roads = gpd.read_file(SCENE_DIR / "road_edges.geojson")
    nodes = gpd.read_file(SCENE_DIR / "launch_recovery_nodes.geojson")
    buildings = gpd.read_file(SCENE_DIR / "buildings.geojson")

    if graph.number_of_nodes() != 129 or graph.number_of_edges() != 247:
        raise ValueError(
            f"Unexpected Baijiahu graph size: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )
    if not nx.is_connected(nx.Graph(graph)):
        raise ValueError("Baijiahu compressed road graph is not connected")
    if len(roads) != graph.number_of_edges() or roads["compressed_edge_id"].nunique() != len(roads):
        raise ValueError("Baijiahu road edge file does not match the compressed graph")
    if len(nodes) != graph.number_of_nodes() or not set(nodes["node_id"].astype(str)).issubset(graph.nodes):
        raise ValueError("Baijiahu launch/recovery nodes do not match the compressed graph")
    if not nodes["launch_allowed"].astype(bool).all() or not nodes["recovery_allowed"].astype(bool).all():
        raise ValueError("Every packaged Baijiahu key node must support launch and recovery")
    heights = np.asarray(buildings["height_m"], dtype=float)
    if len(buildings) != 6872 or not np.isfinite(heights).all() or (heights < 0.0).any():
        raise ValueError("Baijiahu building heights are incomplete or invalid")
    if str(roads.crs) != "EPSG:32650" or str(nodes.crs) != "EPSG:32650" or str(buildings.crs) != "EPSG:32650":
        raise ValueError("Baijiahu assets must use EPSG:32650")
    return roads, nodes, buildings


def _all_touched_mask(
    geometries: list,
    *,
    rows: int,
    cols: int,
    min_x: float,
    max_y: float,
    cell_width: float,
    cell_height: float,
) -> np.ndarray:
    """Rasterize geometries with the same all-touched rule as the source grids."""
    mask = np.zeros((rows, cols), dtype=bool)
    for geometry in geometries:
        if geometry is None or geometry.is_empty:
            continue
        geom_min_x, geom_min_y, geom_max_x, geom_max_y = geometry.bounds
        min_col = max(0, int(math.floor((geom_min_x - min_x) / cell_width)) - 1)
        max_col = min(cols - 1, int(math.floor((geom_max_x - min_x) / cell_width)) + 1)
        min_row = max(0, int(math.floor((max_y - geom_max_y) / cell_height)) - 1)
        max_row = min(rows - 1, int(math.floor((max_y - geom_min_y) / cell_height)) + 1)
        for row in range(min_row, max_row + 1):
            top = max_y - row * cell_height
            for col in range(min_col, max_col + 1):
                left = min_x + col * cell_width
                if geometry.intersects(box(left, top - cell_height, left + cell_width, top)):
                    mask[row, col] = True
    return mask


def _point_cell(x: float, y: float, metadata: dict) -> tuple[int, int]:
    a, b, c, d, e, f = [float(value) for value in metadata["transform"]]
    if abs(b) > 1e-12 or abs(d) > 1e-12 or a <= 0.0 or e >= 0.0:
        raise ValueError(f"Unsupported transform: {metadata['transform']}")
    return int(math.floor((f - y) / abs(e))), int(math.floor((x - c) / a))


def _build_voxel_map(roads: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, building_count: int) -> None:
    source_grid = np.load(SOURCE_VOXEL_DIR / "baijiahu_occupancy_3d_10m.npy", allow_pickle=False)
    source_metadata = json.loads(
        (SOURCE_VOXEL_DIR / "baijiahu_occupancy_3d_10m_metadata.json").read_text(encoding="utf-8")
    )
    if source_grid.shape != (12, 260, 260) or float(source_metadata["voxel_size_z_m"]) != 10.0:
        raise ValueError(f"Unexpected source Baijiahu voxel map: {source_grid.shape}")

    # Conservative aggregation: either occupied source layer makes the 20 m
    # destination layer occupied.
    occupancy = source_grid.reshape(6, 2, source_grid.shape[1], source_grid.shape[2]).max(axis=1)
    bounds = source_metadata["bounds"]
    road_geometries = [geometry.buffer(10.0) for geometry in roads.geometry]
    road_mask = _all_touched_mask(
        road_geometries,
        rows=occupancy.shape[1],
        cols=occupancy.shape[2],
        min_x=float(bounds["min_x"]),
        max_y=float(bounds["max_y"]),
        cell_width=float(source_metadata["voxel_size_x_m"]),
        cell_height=float(source_metadata["voxel_size_y_m"]),
    )
    occupancy[1:, road_mask] = 0

    node_cells: set[tuple[int, int]] = set()
    for point in nodes.geometry:
        row, col = _point_cell(float(point.x), float(point.y), source_metadata)
        if not (0 <= row < occupancy.shape[1] and 0 <= col < occupancy.shape[2]):
            raise ValueError(f"Launch/recovery point falls outside the voxel map: {(point.x, point.y)}")
        node_cells.add((row, col))
        occupancy[:, row, col] = 0

    if occupancy[1:, road_mask].any():
        raise AssertionError("Operational road clearance validation failed")
    if any(occupancy[:, row, col].any() for row, col in node_cells):
        raise AssertionError("Launch/recovery corridor validation failed")

    output_dir = SCENE_DIR / "voxel_map_3d"
    output_dir.mkdir(parents=True, exist_ok=True)
    occupancy = occupancy.astype(np.uint8, copy=False)
    np.save(output_dir / "occupancy.npy", occupancy)
    np.savez_compressed(output_dir / "occupancy.npz", occupancy_3d=occupancy)

    z_edges = np.arange(0.0, 120.0 + 20.0, 20.0)
    metadata = {
        **source_metadata,
        "scene": "baijiahu",
        "voxel_size_z_m": 20.0,
        "z_edges_m": z_edges.tolist(),
        "z_centers_m": ((z_edges[:-1] + z_edges[1:]) / 2.0).tolist(),
        "n_z": 6,
        "building_count": int(building_count),
        "map_variant": "road_free_10m_xy_20m_z",
        "road_buffer_m": 10.0,
        "road_node_free_layers_m": [20, 40, 60, 80, 100, 120],
        "road_node_free_operational_layers": [1, 2, 3, 4, 5],
        "road_node_free_operational_height_range_m": [20.0, 120.0],
        "vertical_aggregation": "max_of_each_two_10m_source_layers_then_road_node_release",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary = {
        "shape": list(occupancy.shape),
        "total_voxels": int(occupancy.size),
        "free_voxels": int((occupancy == 0).sum()),
        "occupied_voxels": int(occupancy.sum()),
        "operational_road_cells_free": True,
        "operational_node_cells_free": True,
        "output_dir": "truck_uav_modules/data/baijiahu/voxel_map_3d",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _build_task_registry() -> tuple[int, str]:
    from env.road_network import RoadNetwork
    from env.xinjiekou_adapter import generate_edge_task_registry, validate_edge_task_registry

    network = RoadNetwork(
        {
            "project": {"seed": 42},
            "network": {
                "type": "compressed_region_graphml",
                "graphml_file": str(SCENE_DIR / "road_network.graphml"),
                "expected_node_count": 129,
                "expected_edge_count": 247,
                "vehicle_speed_mps": 8.3,
                "drone_speed_mps": 10.0,
                "patrol_center": "center",
            },
        }
    )
    registry, digest = generate_edge_task_registry(
        network,
        str(SCENE_DIR / "road_edges.geojson"),
        mapping_version="baijiahu_direct_compressed_edge_v1",
    )
    checks = validate_edge_task_registry(network, registry)
    if checks["registry_edges"] != 247 or not checks["valid_internal_arc_pairs"] or not checks["positive_finite_lengths"]:
        raise ValueError(f"Invalid Baijiahu edge-task registry: {checks}")
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    registry.to_csv(TASK_DIR / "edge_task_registry.csv", index=False, lineterminator="\n", float_format="%.9f")
    persisted = (TASK_DIR / "edge_task_registry.csv").read_bytes()
    return len(registry), hashlib.sha256(persisted).hexdigest()


def main() -> None:
    _copy_canonical_assets()
    roads, nodes, buildings = _validate_scene_assets()
    _build_voxel_map(roads, nodes, len(buildings))
    registry_count, registry_sha256 = _build_task_registry()
    occupancy = np.load(SCENE_DIR / "voxel_map_3d" / "occupancy.npy", allow_pickle=False)
    print(
        json.dumps(
            {
                "scene": "baijiahu",
                "road_nodes": 129,
                "road_edges": 247,
                "launch_recovery_nodes": len(nodes),
                "buildings": len(buildings),
                "voxel_shape": list(occupancy.shape),
                "voxel_size_m": [10.0, 10.0, 20.0],
                "occupied_voxels": int(occupancy.sum()),
                "edge_task_registry_rows": registry_count,
                "edge_task_registry_sha256": registry_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
