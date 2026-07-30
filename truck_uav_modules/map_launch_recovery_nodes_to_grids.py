"""Map CAV UAV launch/recovery nodes onto existing occupancy grids.

Install dependencies with:
    pip install numpy pandas geopandas matplotlib networkx

This script does not regenerate building occupancy grids and never overwrites
the original ``processed_grid_layers`` files. It creates modified copies in
``processed_grid_layers_with_launch_recovery`` by setting only the single grid
cell containing each in-bounds launch/recovery node to free (0).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


LOGGER = logging.getLogger("launch_recovery_grid_mapping")
SCENES = ("xinjiekou", "baijiahu")
FLIGHT_LAYERS_M = (20, 40, 60, 80, 100, 120)

CONFIG = {
    "grid_size_m": 10.0,
    "flight_layers_m": FLIGHT_LAYERS_M,
    "show_grid_lines": True,
    "grid_line_width": 0.15,
    "grid_line_alpha": 0.55,
    "highlight_launch_cells": True,
    "launch_cell_line_width": 0.8,
    "annotate_node_ids": False,
    "dpi": 300,
    "overwrite_original_grid": False,
    "max_expected_distance_m": 10.0 * math.sqrt(2.0) / 2.0,
}

REGION_FILES = {
    "xinjiekou": {
        "road_graph": "road_network.graphml",
        "launch_recovery": "launch_recovery_nodes.geojson",
    },
    "baijiahu": {
        "road_graph": "major_road_graph.graphml",
        "launch_recovery": "launch_recovery_nodes.geojson",
    },
}


def find_existing_grid_files(data_dir: Path) -> Dict[Tuple[str, int], Path]:
    """Find the existing 10 m NPY/NPZ grid for every scene and layer."""
    grid_dir = data_dir / "processed_grid_layers"
    found: Dict[Tuple[str, int], Path] = {}
    for scene in SCENES:
        for layer in FLIGHT_LAYERS_M:
            stem = f"{scene}_occupancy_grid_{layer}m"
            npy = grid_dir / f"{stem}.npy"
            npz = grid_dir / f"{stem}.npz"
            if npy.exists():
                found[(scene, layer)] = npy
            elif npz.exists():
                found[(scene, layer)] = npz
            else:
                raise FileNotFoundError(f"Missing existing grid: {npy} or {npz}")
    return found


def find_grid_metadata_files(data_dir: Path) -> Dict[str, Path]:
    metadata_dir = data_dir / "processed_grid_layers"
    found = {}
    for scene in SCENES:
        path = metadata_dir / f"{scene}_grid_metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing grid metadata: {path}")
        found[scene] = path
    return found


def find_road_network_files(data_dir: Path) -> Dict[str, Dict[str, Path]]:
    found: Dict[str, Dict[str, Path]] = {}
    for scene, names in REGION_FILES.items():
        scene_dir = data_dir / scene
        found[scene] = {}
        for key, filename in names.items():
            path = scene_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing {key} for {scene}: {path}")
            found[scene][key] = path
    return found


def load_grid_and_metadata(grid_path: Path, metadata_path: Path) -> Tuple[np.ndarray, dict]:
    """Load one existing grid and its affine/CRS metadata."""
    loaded = np.load(grid_path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "occupancy_grid" not in loaded.files:
            loaded.close()
            raise ValueError(f"NPZ has no occupancy_grid key: {grid_path}")
        grid = loaded["occupancy_grid"]
        loaded.close()
    else:
        grid = loaded
    grid = np.asarray(grid)
    if grid.ndim != 2 or not np.isin(grid, (0, 1)).all():
        raise ValueError(f"Invalid occupancy grid values or shape: {grid_path}")
    if grid.dtype != np.uint8:
        grid = grid.astype(np.uint8, copy=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = ("crs", "grid_size_m", "rows", "cols", "bounds", "transform")
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ValueError(f"Metadata missing fields {missing}: {metadata_path}")
    if tuple(grid.shape) != (int(metadata["rows"]), int(metadata["cols"])):
        raise ValueError(f"Grid shape and metadata disagree: {grid_path}")
    return grid, metadata


def load_directed_road_network(scene_name: str, network_files: Dict[str, Dict[str, Path]]) -> nx.Graph:
    """Load the packaged GraphML only for node/topology consistency checks."""
    graph_path = network_files[scene_name]["road_graph"]
    graph = nx.read_graphml(graph_path, force_multigraph=True)
    LOGGER.info(
        "[%s] Road graph: %s; nodes=%d, edges=%d, directed=%s",
        scene_name.title(), graph_path.name, graph.number_of_nodes(), graph.number_of_edges(), graph.is_directed(),
    )
    return graph


def identify_launch_recovery_nodes(scene_name: str, network_files: Dict[str, Dict[str, Path]], graph: nx.Graph) -> gpd.GeoDataFrame:
    """Use the project's explicit launch/recovery node collection."""
    path = network_files[scene_name]["launch_recovery"]
    nodes = gpd.read_file(path)
    if nodes.empty:
        raise ValueError(f"No launch/recovery nodes found: {path}")
    if "node_id" not in nodes.columns:
        raise ValueError(f"Launch/recovery data has no node_id field: {path}")
    if nodes["node_id"].duplicated().any():
        LOGGER.warning("[%s] Duplicate node_id values found; retaining all node records.", scene_name.title())
    missing_ids = sorted(set(nodes["node_id"].astype(str)) - {str(node) for node in graph.nodes})
    if missing_ids:
        raise ValueError(f"{scene_name} launch/recovery nodes missing from road graph: {missing_ids[:10]}")
    for field in ("launch_allowed", "recovery_allowed"):
        if field not in nodes.columns:
            nodes[field] = True
    nodes = nodes.loc[nodes.geometry.notna() & ~nodes.geometry.is_empty].copy()
    return nodes.reset_index(drop=True)


def transform_nodes_to_grid_crs(nodes_gdf: gpd.GeoDataFrame, target_crs: str) -> Tuple[gpd.GeoDataFrame, bool, str]:
    """Project nodes to the grid CRS and report the transformation."""
    if nodes_gdf.crs is None:
        raise ValueError("Launch/recovery nodes have no CRS; refusing implicit coordinate conversion")
    original_crs = str(nodes_gdf.crs)
    if str(nodes_gdf.crs) == str(target_crs):
        return nodes_gdf.copy(), False, original_crs
    return nodes_gdf.to_crs(target_crs), True, original_crs


def _row_col_from_transform(x: float, y: float, transform: Sequence[float]) -> Tuple[int, int]:
    """Apply the north-up affine transform [a,b,c,d,e,f] to a point."""
    a, b, c, d, e, f = [float(value) for value in transform]
    if abs(b) > 1e-12 or abs(d) > 1e-12 or a <= 0 or e >= 0:
        raise ValueError(f"Only north-up transforms are supported, got {transform}")
    return int(math.floor((f - y) / abs(e))), int(math.floor((x - c) / a))


def _grid_center(row: int, col: int, transform: Sequence[float]) -> Tuple[float, float]:
    a, b, c, d, e, f = [float(value) for value in transform]
    return c + (col + 0.5) * a, f + (row + 0.5) * e


def map_nodes_to_grid(nodes_gdf: gpd.GeoDataFrame, metadata: dict) -> gpd.GeoDataFrame:
    """Map node points with affine row/column semantics and no clipping."""
    rows, cols = int(metadata["rows"]), int(metadata["cols"])
    transform = metadata["transform"]
    records = []
    for node in nodes_gdf.itertuples(index=False):
        point = node.geometry
        row, col = _row_col_from_transform(float(point.x), float(point.y), transform)
        in_bounds = 0 <= row < rows and 0 <= col < cols
        center_x, center_y = (np.nan, np.nan)
        distance = np.nan
        if in_bounds:
            center_x, center_y = _grid_center(row, col, transform)
            distance = float(math.hypot(float(point.x) - center_x, float(point.y) - center_y))
        records.append(
            {
                "scene": str(getattr(node, "region", "")) or "",
                "node_id": str(node.node_id),
                "node_x": float(point.x),
                "node_y": float(point.y),
                "row": row,
                "col": col,
                "grid_center_x": center_x,
                "grid_center_y": center_y,
                "node_to_grid_center_distance_m": distance,
                "in_bounds": bool(in_bounds),
                "is_launch_allowed": bool(getattr(node, "launch_allowed", True)),
                "is_recovery_allowed": bool(getattr(node, "recovery_allowed", True)),
            }
        )
    mapping = pd.DataFrame(records)
    if mapping.empty:
        raise ValueError("No nodes could be mapped")
    mapping["scene"] = mapping["scene"].replace("", nodes_gdf["region"].iloc[0] if "region" in nodes_gdf else "")
    shared_counts = mapping.loc[mapping["in_bounds"]].groupby(["row", "col"]).size()
    mapping["shared_cell_node_count"] = [
        int(shared_counts.get((row, col), 0)) if in_bounds else 0
        for row, col, in_bounds in zip(mapping["row"], mapping["col"], mapping["in_bounds"])
    ]
    if "longitude" not in mapping:
        mapping["longitude"] = np.nan
        mapping["latitude"] = np.nan
    return mapping


def validate_node_grid_mapping(mapping: pd.DataFrame, metadata: dict) -> None:
    """Validate bounds and cell-center distance for all mapped nodes."""
    in_bounds = mapping[mapping["in_bounds"]]
    if len(in_bounds) != len(mapping):
        LOGGER.warning("%d nodes fall outside the grid extent.", len(mapping) - len(in_bounds))
    if not in_bounds.empty:
        max_distance = float(in_bounds["node_to_grid_center_distance_m"].max())
        if max_distance > CONFIG["max_expected_distance_m"] + 1e-6:
            raise AssertionError(f"Node-to-cell-center distance exceeds half-cell diagonal: {max_distance}")
    if (mapping["row"] < 0).any() and (mapping["in_bounds"]).any():
        LOGGER.info("Out-of-bounds nodes are recorded without clipping.")


def force_launch_recovery_cells_free(grid: np.ndarray, mapping: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Set only unique in-bounds node cells to free and record modifications."""
    modified = grid.copy()
    details = []
    for row in mapping.itertuples(index=False):
        original = np.nan
        changed = False
        modified_value = np.nan
        if row.in_bounds:
            original = int(grid[row.row, row.col])
            modified[row.row, row.col] = 0
            modified_value = int(modified[row.row, row.col])
            changed = original != modified_value
        details.append(
            {
                "scene": row.scene,
                "flight_layer_m": None,
                "node_id": row.node_id,
                "row": row.row,
                "col": row.col,
                "original_occupancy": original,
                "modified_occupancy": modified_value,
                "occupancy_changed": changed,
                "in_bounds": row.in_bounds,
            }
        )
    return modified, pd.DataFrame(details)


def _grid_edges(metadata: dict) -> Tuple[np.ndarray, np.ndarray]:
    min_x, min_y, max_x, max_y = (float(metadata["bounds"][key]) for key in ("min_x", "min_y", "max_x", "max_y"))
    cols, rows = int(metadata["cols"]), int(metadata["rows"])
    return np.linspace(min_x, max_x, cols + 1), np.linspace(min_y, max_y, rows + 1)


def _node_geometry_for_plot(nodes_gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    return nodes_gdf.to_crs(target_crs) if str(nodes_gdf.crs) != str(target_crs) else nodes_gdf


def plot_grid_with_launch_recovery_nodes(
    grid: np.ndarray,
    original_grid: np.ndarray,
    nodes_gdf: gpd.GeoDataFrame,
    mapping: pd.DataFrame,
    scene_name: str,
    layer_m: int,
    metadata: dict,
    output_path: Path,
) -> None:
    """Render occupancy cells, all aligned grid lines, node points and node cells."""
    x_edges, y_edges = _grid_edges(metadata)
    display_grid = np.flipud(grid)
    cmap = ListedColormap(["#ffffff", "#656a72"])
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.pcolormesh(
        x_edges,
        y_edges,
        display_grid,
        cmap=cmap,
        vmin=0,
        vmax=1,
        shading="flat",
        edgecolors="#9aa0a8" if CONFIG["show_grid_lines"] else "none",
        linewidth=CONFIG["grid_line_width"] if CONFIG["show_grid_lines"] else 0,
        alpha=1.0,
    )
    if CONFIG["highlight_launch_cells"]:
        unique_cells = mapping.loc[mapping["in_bounds"], ["row", "col"]].drop_duplicates()
        for cell in unique_cells.itertuples(index=False):
            x = x_edges[int(cell.col)]
            y = y_edges[int(metadata["rows"]) - int(cell.row) - 1]
            ax.add_patch(Rectangle((x, y), float(metadata["grid_size_m"]), float(metadata["grid_size_m"]), fill=False, edgecolor="#f59e0b", linewidth=CONFIG["launch_cell_line_width"], zorder=3))

    plot_nodes = _node_geometry_for_plot(nodes_gdf, str(metadata["crs"]))
    launch = plot_nodes[plot_nodes["launch_allowed"].astype(bool) & ~plot_nodes["recovery_allowed"].astype(bool)]
    recovery = plot_nodes[~plot_nodes["launch_allowed"].astype(bool) & plot_nodes["recovery_allowed"].astype(bool)]
    both = plot_nodes[plot_nodes["launch_allowed"].astype(bool) & plot_nodes["recovery_allowed"].astype(bool)]
    if not launch.empty:
        ax.scatter(launch.geometry.x, launch.geometry.y, marker="o", s=12, color="#2563eb", edgecolor="white", linewidth=0.25, zorder=5)
    if not recovery.empty:
        ax.scatter(recovery.geometry.x, recovery.geometry.y, marker="^", s=14, color="#16a34a", edgecolor="white", linewidth=0.25, zorder=5)
    if not both.empty:
        ax.scatter(both.geometry.x, both.geometry.y, marker="*", s=20, color="#dc2626", edgecolor="white", linewidth=0.25, zorder=5)
    if CONFIG["annotate_node_ids"]:
        for node in plot_nodes.itertuples():
            ax.annotate(str(node.node_id), (node.geometry.x, node.geometry.y), fontsize=4, xytext=(2, 2), textcoords="offset points", zorder=6)

    changed_cells = mapping.loc[mapping["occupancy_changed"] if "occupancy_changed" in mapping else np.zeros(len(mapping), dtype=bool)]
    blocked_cells = int(original_grid.sum())
    free_cells = int(original_grid.size - blocked_cells)
    changed_count = int(mapping.loc[mapping["in_bounds"], ["row", "col"]].drop_duplicates().shape[0])
    changed_from_blocked = int(mapping["occupancy_changed"].sum()) if "occupancy_changed" in mapping else 0
    stats = (
        f"Grid resolution: {metadata['grid_size_m']:g} m\n"
        f"Rows x Columns: {metadata['rows']} x {metadata['cols']}\n"
        f"Total cells: {original_grid.size:,}\n"
        f"Blocked cells: {blocked_cells:,}\n"
        f"Free cells: {free_cells:,}\n"
        f"Launch/recovery nodes: {len(mapping):,}\n"
        f"Unique launch/recovery cells: {changed_count:,}\n"
        f"Cells changed from blocked to free: {changed_from_blocked:,}\n"
        f"Flight layer: {layer_m} m"
    )
    ax.text(0.01, 0.01, stats, transform=ax.transAxes, va="bottom", fontsize=8, bbox={"facecolor": "white", "edgecolor": "#aeb4bc", "alpha": 0.9, "pad": 4}, zorder=8)
    ax.set_title(f"{scene_name.title()} UAV occupancy grid with launch/recovery nodes at {layer_m} m")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    ax.legend(handles=[
        Patch(facecolor="#ffffff", edgecolor="#777777", label="Free cell"),
        Patch(facecolor="#656a72", edgecolor="#555555", label="Occupied cell"),
        Patch(facecolor="none", edgecolor="#f59e0b", label="Launch/recovery cell"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#dc2626", markeredgecolor="white", markersize=8, label="UAV launch/recovery node"),
    ], loc="upper right", fontsize=8, framealpha=0.95)
    fig.savefig(output_path, dpi=CONFIG["dpi"], facecolor="white", bbox_inches="tight")
    plt.close(fig)


def save_modified_grid(grid: np.ndarray, original_grid: np.ndarray, metadata: dict, mapping: pd.DataFrame, scene: str, layer_m: int, output_dir: Path) -> Tuple[Path, Path]:
    """Save NPY/NPZ copies with mapping metadata."""
    stem = f"{scene}_occupancy_grid_{layer_m}m_with_launch_recovery"
    npy_path = output_dir / f"{stem}.npy"
    npz_path = output_dir / f"{stem}.npz"
    np.save(npy_path, grid.astype(np.uint8, copy=False))
    ids = mapping.loc[mapping["in_bounds"], "node_id"].astype(str).to_numpy()
    np.savez_compressed(
        npz_path,
        occupancy_grid=grid.astype(np.uint8, copy=False),
        scene=np.array(scene),
        flight_layer_m=np.array(layer_m),
        grid_size_m=np.array(metadata["grid_size_m"]),
        rows=np.array(metadata["rows"]),
        cols=np.array(metadata["cols"]),
        transform=np.asarray(metadata["transform"], dtype=float),
        crs=np.array(metadata["crs"]),
        launch_recovery_rows=mapping.loc[mapping["in_bounds"], "row"].to_numpy(dtype=np.int32),
        launch_recovery_cols=mapping.loc[mapping["in_bounds"], "col"].to_numpy(dtype=np.int32),
        launch_recovery_node_ids=ids,
    )
    reloaded = np.load(npy_path, allow_pickle=False)
    if not np.array_equal(grid, reloaded):
        raise AssertionError(f"Saved NPY differs from modified array: {npy_path}")
    if np.array_equal(original_grid, grid) and int(mapping["occupancy_changed"].sum()) > 0:
        raise AssertionError("Expected at least one changed grid cell, but modified grid equals original")
    return npy_path, npz_path


def save_mapping_tables(mapping: pd.DataFrame, modifications: List[pd.DataFrame], output_dir: Path) -> None:
    mapping.to_csv(output_dir / "launch_recovery_grid_mapping.csv", index=False)
    pd.concat(modifications, ignore_index=True).to_csv(output_dir / "launch_recovery_grid_modifications.csv", index=False)


def process_scene(scene: str, grid_files: Dict[Tuple[str, int], Path], metadata: dict, network_files: Dict[str, Dict[str, Path]], figure_dir: Path, output_dir: Path) -> Tuple[pd.DataFrame, List[pd.DataFrame], dict]:
    graph = load_directed_road_network(scene, network_files)
    raw_nodes = identify_launch_recovery_nodes(scene, network_files, graph)
    nodes, transformed, original_node_crs = transform_nodes_to_grid_crs(raw_nodes, str(metadata["crs"]))
    if transformed:
        LOGGER.info("[%s] Coordinate transformation applied: True (%s -> %s)", scene.title(), original_node_crs, metadata["crs"])
    else:
        LOGGER.info("[%s] Coordinate transformation applied: False; CRS=%s", scene.title(), metadata["crs"])
    mapping = map_nodes_to_grid(nodes, metadata)
    validate_node_grid_mapping(mapping, metadata)
    mapping["longitude"] = np.nan
    mapping["latitude"] = np.nan
    if raw_nodes.crs is not None:
        geographic = raw_nodes.to_crs("EPSG:4326")
        mapping["longitude"] = geographic.geometry.x.to_numpy()
        mapping["latitude"] = geographic.geometry.y.to_numpy()
    mapping_path = output_dir / f"{scene}_launch_recovery_grid_mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    modifications = []
    summary_rows = []
    for layer_m in FLIGHT_LAYERS_M:
        original_grid, _ = load_grid_and_metadata(grid_files[(scene, layer_m)], metadata_path=GRID_METADATA_PATHS[scene])
        modified_grid, details = force_launch_recovery_cells_free(original_grid, mapping)
        details["flight_layer_m"] = layer_m
        details["scene"] = scene
        details["occupancy_changed"] = details["occupancy_changed"].astype(bool)
        changed_by_cell = {
            (int(row), int(col)): bool(original_grid[int(row), int(col)] != 0)
            for row, col in mapping.loc[mapping["in_bounds"], ["row", "col"]].drop_duplicates().itertuples(index=False)
        }
        mapping["occupancy_changed"] = [
            changed_by_cell.get((int(row), int(col)), False) if in_bounds else False
            for row, col, in_bounds in zip(mapping["row"], mapping["col"], mapping["in_bounds"])
        ]
        # Only the node cells may differ; preserve all other original cells exactly.
        changed_mask = modified_grid != original_grid
        allowed_mask = np.zeros_like(original_grid, dtype=bool)
        for row, col in mapping.loc[mapping["in_bounds"], ["row", "col"]].drop_duplicates().itertuples(index=False):
            allowed_mask[int(row), int(col)] = True
        if np.any(changed_mask & ~allowed_mask):
            raise AssertionError(f"Unexpected non-node grid change at {scene} {layer_m} m")
        npy_path, npz_path = save_modified_grid(modified_grid, original_grid, metadata, details, scene, layer_m, output_dir)
        figure_path = figure_dir / f"{scene}_grid_with_launch_recovery_{layer_m}m.png"
        plot_mapping = mapping.copy()
        plot_mapping["occupancy_changed"] = details.groupby("node_id")["occupancy_changed"].transform("max").astype(bool)
        plot_grid_with_launch_recovery_nodes(modified_grid, original_grid, nodes, plot_mapping, scene, layer_m, metadata, figure_path)
        modifications.append(details)
        unique_cells = mapping.loc[mapping["in_bounds"], ["row", "col"]].drop_duplicates()
        summary_rows.append({
            "scene": scene,
            "flight_layer_m": layer_m,
            "original_grid_file": str(grid_files[(scene, layer_m)]),
            "modified_npy": str(npy_path),
            "modified_npz": str(npz_path),
            "original_node_count": len(raw_nodes),
            "in_bounds_node_count": int(mapping["in_bounds"].sum()),
            "unique_node_count": int(mapping["node_id"].nunique()),
            "unique_coordinate_count": int(nodes.geometry.apply(lambda p: (round(p.x, 6), round(p.y, 6))).nunique()),
            "unique_launch_recovery_cell_count": len(unique_cells),
            "out_of_bounds_node_count": int((~mapping["in_bounds"]).sum()),
            "original_blocked_cells": int(original_grid.sum()),
            "modified_blocked_cells": int(modified_grid.sum()),
            "cells_changed_from_blocked_to_free": int((original_grid != modified_grid).sum()),
            "grid_rows": original_grid.shape[0],
            "grid_cols": original_grid.shape[1],
            "grid_size_m": metadata["grid_size_m"],
            "grid_crs": metadata["crs"],
            "node_crs": original_node_crs,
            "coordinate_transformation_applied": transformed,
            "road_graph_file": str(network_files[scene]["road_graph"]),
            "road_graph_node_count": graph.number_of_nodes(),
            "road_graph_edge_count": graph.number_of_edges(),
            "road_graph_directed": graph.is_directed(),
        })
        LOGGER.info("[%s][%dm] nodes=%d, in_bounds=%d, unique_cells=%d, changed=%d", scene.title(), layer_m, len(mapping), int(mapping["in_bounds"].sum()), len(unique_cells), int((original_grid != modified_grid).sum()))
    return mapping.drop(columns=["occupancy_changed"], errors="ignore"), modifications, {"scene": scene, "summary": summary_rows}


def validate_saved_outputs(output_dir: Path) -> None:
    """Reload every modified NPY and ensure expected files are present."""
    for scene in SCENES:
        for layer in FLIGHT_LAYERS_M:
            path = output_dir / f"{scene}_occupancy_grid_{layer}m_with_launch_recovery.npy"
            if not path.exists():
                raise FileNotFoundError(path)
            grid = np.load(path, allow_pickle=False)
            if grid.dtype != np.uint8 or grid.ndim != 2 or not np.isin(grid, (0, 1)).all():
                raise AssertionError(f"Invalid saved grid: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--output-grid-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    global GRID_METADATA_PATHS
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir or project_root / "truck_uav_modules" / "data"
    figure_dir = args.figure_dir or project_root / "truck_uav_modules" / "figure"
    output_dir = args.output_grid_dir or data_dir / "processed_grid_layers_with_launch_recovery"
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_files = find_existing_grid_files(data_dir)
    GRID_METADATA_PATHS = find_grid_metadata_files(data_dir)
    network_files = find_road_network_files(data_dir)
    metadata_by_scene = {scene: json.loads(path.read_text(encoding="utf-8")) for scene, path in GRID_METADATA_PATHS.items()}
    for scene in SCENES:
        metadata = metadata_by_scene[scene]
        if float(metadata["grid_size_m"]) != CONFIG["grid_size_m"]:
            raise ValueError(f"{scene} grid resolution is {metadata['grid_size_m']}, expected {CONFIG['grid_size_m']}")
        if len(metadata["transform"]) != 6:
            raise ValueError(f"Unexpected affine transform for {scene}: {metadata['transform']}")

    mapping_tables = []
    modification_tables = []
    summaries = []
    for scene in SCENES:
        mapping, modifications, scene_result = process_scene(scene, grid_files, metadata_by_scene[scene], network_files, figure_dir, output_dir)
        mapping_tables.append(mapping)
        modification_tables.extend(modifications)
        summaries.extend(scene_result["summary"])

    pd.concat(mapping_tables, ignore_index=True).to_csv(output_dir / "launch_recovery_grid_mapping.csv", index=False)
    pd.concat(modification_tables, ignore_index=True).to_csv(output_dir / "launch_recovery_grid_modifications.csv", index=False)
    summary_path = output_dir / "launch_recovery_grid_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    validate_saved_outputs(output_dir)
    LOGGER.info("Road-network CRS and grid CRS were checked; original grids and road networks were not overwritten.")
    LOGGER.info("Saved mapping and modified grids to %s", output_dir)
    print("=" * 72)
    print("LAUNCH/RECOVERY NODE TO OCCUPANCY-GRID MAPPING")
    print(f"Output directory: {output_dir}")
    print(f"Grid CRS: {', '.join(sorted({str(row['grid_crs']) for row in summaries}))}")
    print(f"Total scene-layer outputs: {len(summaries)}")
    for row in summaries:
        print(f"[{row['scene'].title()}][{row['flight_layer_m']}m] nodes={row['original_node_count']}, in_bounds={row['in_bounds_node_count']}, unique_cells={row['unique_launch_recovery_cell_count']}, changed={row['cells_changed_from_blocked_to_free']}")


if __name__ == "__main__":
    GRID_METADATA_PATHS: Dict[str, Path] = {}
    main()
