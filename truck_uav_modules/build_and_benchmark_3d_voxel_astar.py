"""Build 10 m 3-D voxel maps and benchmark 26-neighbor 3-D A*.

Dependencies:
    pip install numpy pandas geopandas shapely matplotlib networkx scipy psutil

The script reuses the existing 2-D grid extent/CRS/affine transform and the
existing building footprints/heights. It writes new 3-D outputs and never
overwrites 2-D grids, building data, or CAV road networks.
"""

from __future__ import annotations

import heapq
import json
import logging
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

try:
    import psutil
except ImportError:
    psutil = None

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


LOGGER = logging.getLogger("voxel_3d_astar")
SCENES = ("xinjiekou", "baijiahu")
FLIGHT_LAYERS_M = (20, 40, 60, 80, 100, 120)

CONFIG = {
    "planner_algorithm": "astar",
    "astar_heuristic": "euclidean",
    "heuristic_weight": 1.0,
    "cost_tolerance": 1e-6,
    "voxel_size_m": 10.0,
    "z_min_m": 0.0,
    "z_max_m": 120.0,
    "min_operational_altitude_m": 20.0,
    "num_trials_per_scene": 1000,
    "warmup_trials": 20,
    "smoke_test_trials": 10,
    "random_seed": 20260728,
    "neighbor_mode": 26,
    "prevent_3d_corner_cutting": True,
    "min_3d_distance_voxels": 10.0,
    "min_horizontal_distance_voxels": 8.0,
    # Keep random benchmark pairs local enough for a reproducible 1000-trial
    # run while still exercising obstacle-aware 3-D detours.
    "max_3d_distance_voxels": 20.0,
    "max_horizontal_distance_voxels": 16.0,
    "max_sampling_attempts": 100000,
    "num_example_paths": 5,
    "horizontal_safety_buffer_m": 0.0,
    "vertical_safety_margin_m": 0.0,
    "all_touched": True,
    "force_launch_recovery_column_free": True,
    "dpi": 300,
}


@dataclass(frozen=True)
class GridSpec:
    rows: int
    cols: int
    grid_size_m: float
    transform: Tuple[float, ...]
    crs: str
    bounds: Tuple[float, float, float, float]

    @property
    def min_x(self) -> float:
        return self.bounds[0]

    @property
    def min_y(self) -> float:
        return self.bounds[1]

    @property
    def max_x(self) -> float:
        return self.bounds[2]

    @property
    def max_y(self) -> float:
        return self.bounds[3]


@dataclass
class AStarWorkspace:
    g_score: np.ndarray
    parent: np.ndarray
    seen: np.ndarray
    closed: np.ndarray
    search_id: int = 0

    @classmethod
    def create(cls, shape: Tuple[int, int, int]) -> "AStarWorkspace":
        size = int(np.prod(shape))
        return cls(
            # Heap costs are Python doubles; retaining scores as float64 avoids
            # silently turning the current heap entry into a false stale pop.
            g_score=np.empty(size, dtype=np.float64),
            parent=np.empty(size, dtype=np.int64),
            seen=np.zeros(size, dtype=np.int32),
            closed=np.zeros(size, dtype=np.int32),
        )

@dataclass(frozen=True)
class AStarBudget:
    deadline_perf_counter: Optional[float] = None
    max_expanded_nodes: Optional[int] = None
    check_interval_expansions: int = 128

@dataclass(frozen=True)
class AStar3DResult:
    success: bool
    voxel_path: Optional[List[Tuple[int, int, int]]]
    cost_m: float
    expanded_nodes: int
    generated_nodes: int
    planning_time_ms: float
    termination_reason: str
    failure_reason: Optional[str] = None
    diagnostics: Optional[Dict[str, float]] = None

def load_grid_metadata(data_dir: Path) -> Dict[str, dict]:
    result = {}
    for scene in SCENES:
        path = data_dir / "processed_grid_layers" / f"{scene}_grid_metadata.json"
        if path.is_file():
            result[scene] = json.loads(path.read_text(encoding="utf-8"))
        else:
            # Offline compatibility only: derive the original extent from local
            # footprints instead of depending on unpublished historical grids.
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import rasterize_building_occupancy as grid_tools
            buildings = load_and_clean_buildings(data_dir, scene)
            result[scene] = grid_tools.derive_grid_spec(buildings).metadata()
    return result


def load_existing_2d_grid(data_dir: Path, scene: str, layer_m: int) -> np.ndarray:
    path = data_dir / "processed_grid_layers" / f"{scene}_occupancy_grid_{layer_m}m.npy"
    return np.load(path, allow_pickle=False)


def load_launch_recovery_mapping(data_dir: Path, scene: str) -> pd.DataFrame:
    path = data_dir / "processed_grid_layers_with_launch_recovery" / "launch_recovery_grid_mapping.csv"
    if path.is_file():
        mapping = pd.read_csv(path)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import rasterize_building_occupancy as grid_tools
        buildings = load_and_clean_buildings(data_dir, scene)
        grid = grid_tools.derive_grid_spec(buildings)
        nodes = gpd.read_file(data_dir / scene / "launch_recovery_nodes.geojson")
        mapping = grid_tools.map_launch_nodes(nodes, grid, scene)
    mapping = mapping[(mapping["scene"].astype(str).str.lower() == scene) & mapping["in_bounds"].astype(bool)].copy()
    required = {"node_id", "row", "col", "node_x", "node_y"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Launch/recovery mapping missing {sorted(missing)}: {path}")
    return mapping


def load_and_clean_buildings(data_dir: Path, scene: str) -> gpd.GeoDataFrame:
    """Reuse the previous building loader/normalizer and standard height field."""
    package_dir = Path(__file__).resolve().parent
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    import rasterize_building_occupancy as grid_tools

    path = data_dir / scene / ("buildings.geojson" if scene.lower() == "xinjiekou" else "buildings_local.geojson")
    if not path.is_file():
        path = data_dir / scene / "buildings.geojson"
    buildings = grid_tools.load_buildings(path)
    buildings = grid_tools.normalize_building_height(grid_tools.clean_building_geometries(buildings))
    return buildings


def build_grid_spec(metadata: dict) -> GridSpec:
    return GridSpec(
        rows=int(metadata["rows"]),
        cols=int(metadata["cols"]),
        grid_size_m=float(metadata["grid_size_m"]),
        transform=tuple(float(value) for value in metadata["transform"]),
        crs=str(metadata["crs"]),
        bounds=tuple(float(metadata["bounds"][key]) for key in ("min_x", "min_y", "max_x", "max_y")),
    )


def _rasterize_buildings(buildings: gpd.GeoDataFrame, grid_spec: GridSpec, z_lower_m: float) -> np.ndarray:
    """Rasterize one horizontal slice using the existing bulk rasterizer."""
    import rasterize_building_occupancy as obstruction_tools

    blocking = buildings.loc[buildings["building_height_m"] + CONFIG["vertical_safety_margin_m"] > z_lower_m]
    transform_values = tuple(float(value) for value in grid_spec.transform)
    transform_object = type("TemporaryTransform", (), {
        "a": transform_values[0], "b": transform_values[1], "c": transform_values[2],
        "d": transform_values[3], "e": transform_values[4], "f": transform_values[5],
        "__iter__": lambda self: iter(transform_values),
    })()
    temporary_spec = type("TemporaryGridSpec", (), {
        "min_x": grid_spec.min_x,
        "min_y": grid_spec.min_y,
        "max_x": grid_spec.max_x,
        "max_y": grid_spec.max_y,
        "grid_size_m": grid_spec.grid_size_m,
        "n_rows": grid_spec.rows,
        "n_cols": grid_spec.cols,
        "transform": transform_object,
    })()
    return obstruction_tools.rasterize_blocking_buildings(
        blocking,
        temporary_spec,
        CONFIG["horizontal_safety_buffer_m"],
        CONFIG["all_touched"],
    )


def apply_launch_recovery_vertical_corridors(
    occupancy_3d: np.ndarray,
    mapping: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Force the complete vertical column at each unique mapped cell to free."""
    modified = occupancy_3d.copy()
    unique_cells = mapping[["row", "col"]].drop_duplicates()
    records = []
    for cell in unique_cells.itertuples(index=False):
        row, col = int(cell.row), int(cell.col)
        original_column = occupancy_3d[:, row, col].copy()
        modified[:, row, col] = 0
        node_ids = mapping.loc[(mapping["row"] == row) & (mapping["col"] == col), "node_id"].astype(str).tolist()
        for z_index, original in enumerate(original_column):
            z_lower = CONFIG["z_min_m"] + z_index * CONFIG["voxel_size_m"]
            records.append({
                "scene": str(mapping["scene"].iloc[0]),
                "node_id": "|".join(node_ids),
                "z_index": z_index,
                "z_lower_m": z_lower,
                "z_upper_m": z_lower + CONFIG["voxel_size_m"],
                "row": row,
                "col": col,
                "original_occupancy": int(original),
                "modified_occupancy": int(modified[z_index, row, col]),
                "occupancy_changed": bool(original != modified[z_index, row, col]),
            })
    return modified, pd.DataFrame(records)


def build_3d_voxel_grid(
    scene: str,
    data_dir: Path,
    metadata: dict,
    mapping: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, GridSpec, dict]:
    """Build true z,row,col voxels from building footprints and heights."""
    started = perf_counter_ns()
    grid_spec = build_grid_spec(metadata)
    if grid_spec.grid_size_m != CONFIG["voxel_size_m"]:
        raise ValueError(f"2-D grid size {grid_spec.grid_size_m} differs from voxel size {CONFIG['voxel_size_m']}")
    buildings = load_and_clean_buildings(data_dir, scene)
    n_z = int(round((CONFIG["z_max_m"] - CONFIG["z_min_m"]) / CONFIG["voxel_size_m"]))
    z_edges = np.arange(CONFIG["z_min_m"], CONFIG["z_max_m"] + CONFIG["voxel_size_m"], CONFIG["voxel_size_m"])
    occupancy = np.zeros((n_z, grid_spec.rows, grid_spec.cols), dtype=np.uint8)
    for z_index, z_lower in enumerate(z_edges[:-1]):
        occupancy[z_index] = _rasterize_buildings(buildings, grid_spec, float(z_lower))
    original = occupancy.copy()
    if CONFIG["force_launch_recovery_column_free"]:
        occupancy, modifications = apply_launch_recovery_vertical_corridors(occupancy, mapping)
    else:
        modifications = pd.DataFrame()
    build_time_s = (perf_counter_ns() - started) / 1_000_000_000.0
    metadata_3d = {
        "scene": scene,
        "voxel_size_x_m": CONFIG["voxel_size_m"],
        "voxel_size_y_m": CONFIG["voxel_size_m"],
        "voxel_size_z_m": CONFIG["voxel_size_m"],
        "z_min_m": CONFIG["z_min_m"],
        "z_max_m": CONFIG["z_max_m"],
        "z_edges_m": z_edges.tolist(),
        "z_centers_m": ((z_edges[:-1] + z_edges[1:]) / 2.0).tolist(),
        "rows": grid_spec.rows,
        "cols": grid_spec.cols,
        "n_z": n_z,
        "transform": list(grid_spec.transform),
        "crs": grid_spec.crs,
        "bounds": dict(zip(("min_x", "min_y", "max_x", "max_y"), grid_spec.bounds)),
        "min_operational_altitude_m": CONFIG["min_operational_altitude_m"],
        "horizontal_safety_buffer_m": CONFIG["horizontal_safety_buffer_m"],
        "vertical_safety_margin_m": CONFIG["vertical_safety_margin_m"],
        "all_touched": CONFIG["all_touched"],
        "force_launch_recovery_column_free": CONFIG["force_launch_recovery_column_free"],
        "building_count": len(buildings),
        "build_time_s": build_time_s,
    }
    return occupancy, original, grid_spec, {"metadata": metadata_3d, "modifications": modifications}


def save_3d_voxel_grid(
    occupancy_3d: np.ndarray,
    metadata: dict,
    output_dir: Path,
    scene: str,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / f"{scene}_occupancy_3d_10m.npy"
    npz_path = output_dir / f"{scene}_occupancy_3d_10m.npz"
    json_path = output_dir / f"{scene}_occupancy_3d_10m_metadata.json"
    np.save(npy_path, occupancy_3d.astype(np.uint8, copy=False))
    np.savez_compressed(
        npz_path,
        occupancy_3d=occupancy_3d.astype(np.uint8, copy=False),
        scene=np.array(scene),
        voxel_size_x_m=np.array(metadata["voxel_size_x_m"]),
        voxel_size_y_m=np.array(metadata["voxel_size_y_m"]),
        voxel_size_z_m=np.array(metadata["voxel_size_z_m"]),
        z_min_m=np.array(metadata["z_min_m"]),
        z_max_m=np.array(metadata["z_max_m"]),
        z_edges_m=np.asarray(metadata["z_edges_m"], dtype=float),
        z_centers_m=np.asarray(metadata["z_centers_m"], dtype=float),
        rows=np.array(metadata["rows"]),
        cols=np.array(metadata["cols"]),
        n_z=np.array(metadata["n_z"]),
        transform=np.asarray(metadata["transform"], dtype=float),
        crs=np.array(metadata["crs"]),
        bounds=np.asarray([metadata["bounds"][key] for key in ("min_x", "min_y", "max_x", "max_y")], dtype=float),
        min_operational_altitude_m=np.array(metadata["min_operational_altitude_m"]),
        horizontal_safety_buffer_m=np.array(metadata["horizontal_safety_buffer_m"]),
        vertical_safety_margin_m=np.array(metadata["vertical_safety_margin_m"]),
        all_touched=np.array(metadata["all_touched"]),
    )
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not np.array_equal(occupancy_3d, np.load(npy_path, allow_pickle=False)):
        raise AssertionError(f"NPY reload mismatch: {npy_path}")
    return npy_path, npz_path


def world_to_voxel(x: float, y: float, z: float, transform: Sequence[float], z_min_m: float, voxel_size_z_m: float, shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Convert world x,y,z to z,row,col using the north-up affine transform."""
    a, b, c, d, e, f = [float(value) for value in transform]
    if abs(b) > 1e-12 or abs(d) > 1e-12 or a <= 0 or e >= 0:
        raise ValueError(f"Unsupported affine transform: {transform}")
    row = int(math.floor((f - y) / abs(e)))
    col = int(math.floor((x - c) / a))
    z_index = int(math.floor((z - z_min_m) / voxel_size_z_m))
    nz, rows, cols = shape
    if not (0 <= z_index < nz and 0 <= row < rows and 0 <= col < cols):
        raise ValueError(f"World point is outside voxel grid: {(x, y, z)} -> {(z_index, row, col)}")
    return z_index, row, col


def voxel_to_world_center(z_index: int, row: int, col: int, transform: Sequence[float], z_min_m: float, voxel_size_z_m: float) -> Tuple[float, float, float]:
    a, b, c, d, e, f = [float(value) for value in transform]
    return c + (col + 0.5) * a, f + (row + 0.5) * e, z_min_m + (z_index + 0.5) * voxel_size_z_m


def validate_voxel_coordinate_conversion(occupancy: np.ndarray, metadata: dict, rng: np.random.Generator, count: int = 100) -> None:
    free = np.argwhere(occupancy == 0)
    if len(free) < count:
        count = len(free)
    selected = free[rng.choice(len(free), size=count, replace=False)]
    for z_index, row, col in selected:
        x, y, z = voxel_to_world_center(int(z_index), int(row), int(col), metadata["transform"], metadata["z_min_m"], metadata["voxel_size_z_m"])
        recovered = world_to_voxel(x, y, z, metadata["transform"], metadata["z_min_m"], metadata["voxel_size_z_m"], occupancy.shape)
        if recovered != (int(z_index), int(row), int(col)):
            raise AssertionError(f"Voxel coordinate round trip failed: {(z_index, row, col)} -> {recovered}")


def _neighbor_offsets(voxel_size_xyz: Optional[Tuple[float, float, float]] = None) -> List[Tuple[int, int, int, float, Tuple[Tuple[int, int, int], ...]]]:
    offsets = []
    # Tuple order is (x, y, z), whereas a voxel index step is (z, row, col).
    # Keeping the uniform-grid default preserves the benchmark behaviour.
    if voxel_size_xyz is None:
        voxel = float(CONFIG["voxel_size_m"])
        voxel_size_xyz = (voxel, voxel, voxel)
    vx, vy, vz = map(float, voxel_size_xyz)
    for dz in (-1, 0, 1):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dz, dr, dc) == (0, 0, 0):
                    continue
                changed = [axis for axis, value in enumerate((dz, dr, dc)) if value]
                intermediates: List[Tuple[int, int, int]] = []
                if CONFIG["prevent_3d_corner_cutting"] and len(changed) > 1:
                    # All non-empty proper subsets of changed axes must be free.
                    for mask in range(1, 2 ** len(changed) - 1):
                        point = [0, 0, 0]
                        for bit, axis in enumerate(changed):
                            if mask & (1 << bit):
                                point[axis] = (dz, dr, dc)[axis]
                        intermediates.append(tuple(point))
                cost = math.sqrt((dz * vz) ** 2 + (dr * vy) ** 2 + (dc * vx) ** 2)
                offsets.append((dz, dr, dc, cost, tuple(intermediates)))
    return offsets


def is_valid_3d_move(occupancy_3d: np.ndarray, current: Tuple[int, int, int], neighbor: Tuple[int, int, int], intermediates: Sequence[Tuple[int, int, int]]) -> bool:
    nz, rows, cols = occupancy_3d.shape
    z, row, col = neighbor
    if not (0 <= z < nz and 0 <= row < rows and 0 <= col < cols):
        return False
    if occupancy_3d[z, row, col] != 0:
        return False
    if CONFIG["prevent_3d_corner_cutting"]:
        for dz, dr, dc in intermediates:
            if occupancy_3d[current[0] + dz, current[1] + dr, current[2] + dc] != 0:
                return False
    return True


def heuristic_3d_euclidean(current: Tuple[int, int, int], goal: Tuple[int, int, int], voxel_size_xyz: Tuple[float, float, float]) -> float:
    """Admissible Euclidean lower bound in physical metres."""
    cz, cr, cc = current
    gz, gr, gc = goal
    dz = abs(gz - cz) * voxel_size_xyz[2]
    dy = abs(gr - cr) * voxel_size_xyz[1]
    dx = abs(gc - cc) * voxel_size_xyz[0]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _plan_3d_path_runtime_fast(
    occupancy_3d: np.ndarray,
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    algorithm: str,
    workspace: Optional[AStarWorkspace],
    offsets: Optional[Sequence[Tuple[int, int, int, float, Tuple[Tuple[int, int, int], ...]]]],
    voxel_size_xyz: Optional[Tuple[float, float, float]],
    prevent_corner_cutting: bool,
    budget: Optional[AStarBudget],
    return_result: bool,
    heuristic_weight: float,
):
    """Lean runtime A* with the same graph and budget semantics.

    The benchmark implementation below retains detailed heap diagnostics.  In
    an online simulator those per-neighbour Python dictionary updates dominate
    long searches, so this path precomputes the admissible Euclidean heuristic
    and emits compact diagnostics while preserving the exact 26-neighbour,
    occupancy and corner-cutting checks.
    """
    started=time.perf_counter()
    algorithm=str(algorithm).lower()
    if algorithm not in {"astar", "dijkstra"}:
        raise ValueError("algorithm must be 'dijkstra' or 'astar'")
    if workspace is None:
        workspace=AStarWorkspace.create(occupancy_3d.shape)
    if voxel_size_xyz is None:
        voxel_size=float(CONFIG["voxel_size_m"])
        voxel_size_xyz=(voxel_size, voxel_size, voxel_size)
    if offsets is None:
        offsets=_neighbor_offsets(voxel_size_xyz)
    nz, rows, cols=occupancy_3d.shape
    def finish(path, cost, expanded, generated, reason, failure=None):
        result=AStar3DResult(
            path is not None, path, float(cost), int(expanded), int(generated),
            (time.perf_counter()-started)*1000.0, reason, failure,
            {"runtime_fast_path": 1.0, "expanded_node_count": float(expanded)},
        )
        return result if return_result else (path, cost, expanded, generated)
    if not (all(0 <= int(start[i]) < occupancy_3d.shape[i] for i in range(3)) and all(0 <= int(goal[i]) < occupancy_3d.shape[i] for i in range(3))):
        return finish(None, math.inf, 0, 0, "invalid_start" if not all(0 <= int(start[i]) < occupancy_3d.shape[i] for i in range(3)) else "invalid_goal", "voxel_out_of_bounds")
    start=tuple(map(int, start)); goal=tuple(map(int, goal))
    if occupancy_3d[start] != 0:
        return finish(None, math.inf, 0, 0, "invalid_start", "occupied_start")
    if occupancy_3d[goal] != 0:
        return finish(None, math.inf, 0, 0, "invalid_goal", "occupied_goal")
    if start == goal:
        return finish([start], 0.0, 0, 0, "success")
    workspace.search_id += 1
    search_id=workspace.search_id
    plane=rows*cols
    start_flat=start[0]*plane+start[1]*cols+start[2]
    goal_flat=goal[0]*plane+goal[1]*cols+goal[2]
    occupancy_flat=occupancy_3d.ravel()
    if algorithm == "astar":
        z_axis=(np.arange(nz, dtype=np.float64)[:, None, None]-float(goal[0]))*float(voxel_size_xyz[2])
        row_axis=(np.arange(rows, dtype=np.float64)[None, :, None]-float(goal[1]))*float(voxel_size_xyz[1])
        col_axis=(np.arange(cols, dtype=np.float64)[None, None, :]-float(goal[2]))*float(voxel_size_xyz[0])
        heuristic_flat=np.sqrt(z_axis*z_axis+row_axis*row_axis+col_axis*col_axis).ravel()
    else:
        heuristic_flat=np.zeros(occupancy_flat.size, dtype=np.float64)
    moves=[]
    for dz, dr, dc, cost, intermediates in offsets:
        moves.append((int(dz), int(dr), int(dc), float(cost), int(dz)*plane+int(dr)*cols+int(dc), tuple(int(iz)*plane+int(ir)*cols+int(ic) for iz,ir,ic in intermediates)))
    workspace.seen[start_flat]=search_id
    workspace.closed[start_flat]=0
    workspace.g_score[start_flat]=0.0
    workspace.parent[start_flat]=-1
    heuristic_weight=max(1.0, float(heuristic_weight))
    heap=[(float(heuristic_weight*heuristic_flat[start_flat]), start_flat)]
    expanded=0; generated=1
    check_interval=max(1, int(budget.check_interval_expansions)) if budget is not None else 1
    while heap:
        popped_f, current_flat=heapq.heappop(heap)
        best_g=float(workspace.g_score[current_flat])
        if popped_f > best_g+heuristic_weight*float(heuristic_flat[current_flat])+1e-9:
            continue
        if workspace.closed[current_flat] == search_id:
            continue
        workspace.closed[current_flat]=search_id
        expanded += 1
        if budget is not None and (expanded % check_interval == 0 or (budget.max_expanded_nodes is not None and expanded >= int(budget.max_expanded_nodes))):
            if budget.max_expanded_nodes is not None and expanded >= int(budget.max_expanded_nodes):
                return finish(None, math.inf, expanded, generated, "max_nodes_exceeded", "max_expanded_nodes")
            if budget.deadline_perf_counter is not None and time.perf_counter() >= float(budget.deadline_perf_counter):
                return finish(None, math.inf, expanded, generated, "timeout", "deadline_exceeded")
        if current_flat == goal_flat:
            path_flat=[current_flat]
            while path_flat[-1] != start_flat:
                path_flat.append(int(workspace.parent[path_flat[-1]]))
            path=[]
            for item in reversed(path_flat):
                z=item//plane; remainder=item-z*plane; row=remainder//cols
                path.append((int(z), int(row), int(remainder-row*cols)))
            return finish(path, best_g, expanded, generated, "success")
        z=current_flat//plane; remainder=current_flat-z*plane; row=remainder//cols; col=remainder-row*cols
        for dz, dr, dc, move_cost, flat_delta, intermediate_deltas in moves:
            next_z=z+dz; next_row=row+dr; next_col=col+dc
            if not (0 <= next_z < nz and 0 <= next_row < rows and 0 <= next_col < cols):
                continue
            neighbor_flat=current_flat+flat_delta
            if occupancy_flat[neighbor_flat] != 0:
                continue
            blocked=False
            if prevent_corner_cutting:
                for intermediate_delta in intermediate_deltas:
                    if occupancy_flat[current_flat+intermediate_delta] != 0:
                        blocked=True
                        break
            if blocked:
                continue
            tentative=best_g+move_cost
            if workspace.seen[neighbor_flat] == search_id and tentative+1e-12 >= float(workspace.g_score[neighbor_flat]):
                continue
            workspace.seen[neighbor_flat]=search_id
            workspace.g_score[neighbor_flat]=tentative
            workspace.parent[neighbor_flat]=current_flat
            if workspace.closed[neighbor_flat] == search_id:
                workspace.closed[neighbor_flat]=0
            heapq.heappush(heap, (tentative+heuristic_weight*float(heuristic_flat[neighbor_flat]), neighbor_flat))
            generated += 1
    return finish(None, math.inf, expanded, generated, "open_set_exhausted", "disconnected_free_space")


def plan_3d_path(
    occupancy_3d: np.ndarray,
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    algorithm: str = "astar",
    neighbor_mode: int = 26,
    prevent_corner_cutting: bool = True,
    workspace: Optional[AStarWorkspace] = None,
    offsets: Optional[Sequence[Tuple[int, int, int, float, Tuple[Tuple[int, int, int], ...]]]] = None,
    voxel_size_xyz: Optional[Tuple[float, float, float]] = None,
    budget: Optional[AStarBudget] = None,
    return_result: bool = False,
    runtime_fast: bool = True,
    heuristic_weight: float = 1.0,
) -> Tuple[Optional[List[Tuple[int, int, int]]], float, int, int]:
    """Run Dijkstra or standard A* over the same 3-D voxel graph.

    Dijkstra is exactly the same search with ``h=0``. Both algorithms share
    the occupancy grid, offsets, move costs, corner checks, workspace and
    path reconstruction; only the queue priority differs.
    """
    if runtime_fast:
        return _plan_3d_path_runtime_fast(
            occupancy_3d, start, goal, algorithm, workspace, offsets,
            voxel_size_xyz, prevent_corner_cutting, budget, return_result, heuristic_weight,
        )
    algorithm = str(algorithm).lower()
    if algorithm not in {"dijkstra", "astar"}:
        raise ValueError("algorithm must be 'dijkstra' or 'astar'")
    if neighbor_mode != 26:
        raise ValueError("This benchmark currently supports neighbor_mode=26 only")
    if workspace is None:
        workspace = AStarWorkspace.create(occupancy_3d.shape)
    if voxel_size_xyz is None:
        voxel_size = float(CONFIG["voxel_size_m"])
        voxel_size_xyz = (voxel_size, voxel_size, voxel_size)
    if offsets is None:
        offsets = _neighbor_offsets(voxel_size_xyz)
    started = time.perf_counter()
    diagnostics = {"heap_push_count": 0, "heap_pop_count": 0, "stale_heap_pop_count": 0,
                   "expanded_node_count": 0, "successful_relaxation_count": 0,
                   "closed_skip_count": 0, "reopened_node_count": 0, "max_open_set_size": 0,
                   "minimum_positive_stale_delta": math.inf, "maximum_stale_delta": 0.0,
                   "sum_positive_stale_delta": 0.0}
    def finish(path, cost, expanded, generated, reason, failure=None):
        diagnostics["expanded_node_count"] = int(expanded)
        count = diagnostics["stale_heap_pop_count"]
        diagnostics["mean_positive_stale_delta"] = diagnostics["sum_positive_stale_delta"] / count if count else 0.0
        if not math.isfinite(diagnostics["minimum_positive_stale_delta"]):
            diagnostics["minimum_positive_stale_delta"] = 0.0
        diagnostics.pop("sum_positive_stale_delta", None)
        result = AStar3DResult(path is not None, path, float(cost), int(expanded), int(generated), (time.perf_counter()-started)*1000.0, reason, failure, dict(diagnostics))
        return result if return_result else (path, cost, expanded, generated)
    if not (all(0 <= int(start[i]) < occupancy_3d.shape[i] for i in range(3)) and all(0 <= int(goal[i]) < occupancy_3d.shape[i] for i in range(3))):
        return finish(None, math.inf, 0, 0, "invalid_start" if not all(0 <= int(start[i]) < occupancy_3d.shape[i] for i in range(3)) else "invalid_goal", "voxel_out_of_bounds")
    if occupancy_3d[start] != 0:
        return finish(None, math.inf, 0, 0, "invalid_start", "occupied_start")
    if occupancy_3d[goal] != 0:
        return finish(None, math.inf, 0, 0, "invalid_goal", "occupied_goal")
    if start == goal:
        return finish([start], 0.0, 0, 0, "success")

    workspace.search_id += 1
    search_id = workspace.search_id
    nz, rows, cols = occupancy_3d.shape
    plane_size = rows * cols
    goal_flat = goal[0] * plane_size + goal[1] * cols + goal[2]
    start_flat = start[0] * plane_size + start[1] * cols + start[2]
    occupancy_flat = occupancy_3d.ravel()
    goal_z, goal_row, goal_col = goal
    use_heuristic = algorithm == "astar"

    def heuristic(z: int, row: int, col: int) -> float:
        if not use_heuristic:
            return 0.0
        return heuristic_3d_euclidean((z, row, col), (goal_z, goal_row, goal_col), voxel_size_xyz)

    workspace.seen[start_flat] = search_id
    workspace.g_score[start_flat] = 0.0
    workspace.parent[start_flat] = -1
    tie_breaker = 0
    start_h = heuristic(*start)
    heap: List[Tuple[float, float, float, int, int]] = [(start_h, 0.0, start_h, tie_breaker, start_flat)]
    diagnostics["heap_push_count"] = 1
    diagnostics["max_open_set_size"] = 1
    expanded = 0
    generated = 1
    while heap:
        _, popped_g, _, _, current_flat = heapq.heappop(heap)
        diagnostics["heap_pop_count"] += 1
        best_g = float(workspace.g_score[current_flat])
        stale_tolerance = 1e-9 + 1e-12 * max(abs(popped_g), abs(best_g))
        if popped_g > best_g + stale_tolerance:
            delta = popped_g - best_g
            diagnostics["stale_heap_pop_count"] += 1
            diagnostics["minimum_positive_stale_delta"] = min(diagnostics["minimum_positive_stale_delta"], delta)
            diagnostics["maximum_stale_delta"] = max(diagnostics["maximum_stale_delta"], delta)
            diagnostics["sum_positive_stale_delta"] += delta
            continue
        if workspace.closed[current_flat] == search_id:
            diagnostics["closed_skip_count"] += 1
            continue
        workspace.closed[current_flat] = search_id
        expanded += 1
        # Honour an expanded-node ceiling exactly even when wall-clock checks
        # are sampled less frequently for throughput.  Recovery replanning
        # uses this guard as a commit boundary, so overshooting the configured
        # ceiling would make the stated safety budget misleading.
        if budget is not None and (
            expanded % max(1, int(budget.check_interval_expansions)) == 0
            or (
                budget.max_expanded_nodes is not None
                and expanded >= int(budget.max_expanded_nodes)
            )
        ):
            if budget.max_expanded_nodes is not None and expanded >= int(budget.max_expanded_nodes):
                return finish(None, math.inf, expanded, generated, "max_nodes_exceeded", "max_expanded_nodes")
            if budget.deadline_perf_counter is not None and time.perf_counter() >= float(budget.deadline_perf_counter):
                return finish(None, math.inf, expanded, generated, "timeout", "deadline_exceeded")
        if current_flat == goal_flat:
            path_flat = [current_flat]
            while path_flat[-1] != start_flat:
                path_flat.append(int(workspace.parent[path_flat[-1]]))
            path_flat.reverse()
            path = []
            for item in path_flat:
                z_item = item // plane_size
                rem_item = item - z_item * plane_size
                row_item = rem_item // cols
                col_item = rem_item - row_item * cols
                path.append((int(z_item), int(row_item), int(col_item)))
            return finish(path, best_g, expanded, generated, "success")

        z = current_flat // plane_size
        rem = current_flat - z * plane_size
        row = rem // cols
        col = rem - row * cols
        for dz, dr, dc, move_cost, intermediates in offsets:
            neighbor_z = z + dz
            neighbor_row = row + dr
            neighbor_col = col + dc
            if not (0 <= neighbor_z < nz and 0 <= neighbor_row < rows and 0 <= neighbor_col < cols):
                continue
            neighbor_flat = neighbor_z * plane_size + neighbor_row * cols + neighbor_col
            if occupancy_flat[neighbor_flat] != 0:
                continue
            if prevent_corner_cutting:
                blocked = False
                for intermediate_dz, intermediate_dr, intermediate_dc in intermediates:
                    intermediate_flat = ((z + intermediate_dz) * rows + row + intermediate_dr) * cols + col + intermediate_dc
                    if occupancy_flat[intermediate_flat] != 0:
                        blocked = True
                        break
                if blocked:
                    continue
            tentative = float(best_g) + float(move_cost)
            known = float(workspace.g_score[neighbor_flat])
            relaxation_tolerance = 1e-12 + 1e-12 * max(abs(tentative), abs(known)) if math.isfinite(known) else 0.0
            if workspace.seen[neighbor_flat] != search_id or tentative + relaxation_tolerance < known:
                workspace.seen[neighbor_flat] = search_id
                workspace.g_score[neighbor_flat] = tentative
                workspace.parent[neighbor_flat] = current_flat
                diagnostics["successful_relaxation_count"] += 1
                if workspace.closed[neighbor_flat] == search_id:
                    workspace.closed[neighbor_flat] = 0
                    diagnostics["reopened_node_count"] += 1
                tie_breaker += 1
                h_score = heuristic(neighbor_z, neighbor_row, neighbor_col)
                heapq.heappush(heap, (float(tentative + h_score), float(tentative), float(h_score), tie_breaker, neighbor_flat))
                diagnostics["heap_push_count"] += 1
                diagnostics["max_open_set_size"] = max(diagnostics["max_open_set_size"], len(heap))
                generated += 1
    return finish(None, math.inf, expanded, generated, "open_set_exhausted", "disconnected_free_space")


def astar_3d(
    occupancy_3d: np.ndarray,
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    workspace: AStarWorkspace,
    offsets: Sequence[Tuple[int, int, int, float, Tuple[Tuple[int, int, int], ...]]],
) -> Tuple[Optional[List[Tuple[int, int, int]]], float, int, int]:
    return plan_3d_path(occupancy_3d, start, goal, algorithm="astar", workspace=workspace, offsets=offsets)


def dijkstra_3d(
    occupancy_3d: np.ndarray,
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    workspace: AStarWorkspace,
    offsets: Sequence[Tuple[int, int, int, float, Tuple[Tuple[int, int, int], ...]]],
) -> Tuple[Optional[List[Tuple[int, int, int]]], float, int, int]:
    return plan_3d_path(occupancy_3d, start, goal, algorithm="dijkstra", workspace=workspace, offsets=offsets)


def validate_3d_path(occupancy: np.ndarray, path: Sequence[Tuple[int, int, int]], start: Tuple[int, int, int], goal: Tuple[int, int, int], offsets: Sequence[Tuple[int, int, int, float, Tuple[Tuple[int, int, int], ...]]], cost: float) -> None:
    if not path or path[0] != start or path[-1] != goal:
        raise AssertionError("3-D path endpoints are invalid")
    offset_lookup = {(dz, dr, dc): (intermediates, move_cost) for dz, dr, dc, move_cost, intermediates in offsets}
    calculated = 0.0
    for current, neighbor in zip(path[:-1], path[1:]):
        delta = tuple(b - a for a, b in zip(current, neighbor))
        if delta not in offset_lookup:
            raise AssertionError(f"Invalid 3-D neighbor step: {delta}")
        if occupancy[neighbor] != 0 or not is_valid_3d_move(occupancy, current, neighbor, offset_lookup[delta][0]):
            raise AssertionError("3-D path crosses occupied voxel or cuts a corner")
        calculated += offset_lookup[delta][1]
    if not math.isclose(calculated, cost, rel_tol=1e-5, abs_tol=1e-4):
        raise AssertionError(f"Path cost mismatch: {calculated} != {cost}")


def compute_3d_connected_components(occupancy: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute conservative 6-connected components of free voxels."""
    structure = ndimage.generate_binary_structure(3, 1) if ndimage is not None else None
    if ndimage is not None:
        labels, count = ndimage.label(occupancy == 0, structure=structure)
        members = [np.argwhere(labels == index) for index in range(1, count + 1)]
        return labels.astype(np.int32), members
    labels = np.full(occupancy.shape, -1, dtype=np.int32)
    members: List[np.ndarray] = []
    for start in np.argwhere(occupancy == 0):
        start_tuple = tuple(int(value) for value in start)
        if labels[start_tuple] >= 0:
            continue
        stack = [start_tuple]
        labels[start_tuple] = len(members)
        current_members = []
        while stack:
            z, row, col = stack.pop()
            current_members.append((z, row, col))
            for dz, dr, dc in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                next_cell = (z + dz, row + dr, col + dc)
                if 0 <= next_cell[0] < occupancy.shape[0] and 0 <= next_cell[1] < occupancy.shape[1] and 0 <= next_cell[2] < occupancy.shape[2] and occupancy[next_cell] == 0 and labels[next_cell] < 0:
                    labels[next_cell] = len(members)
                    stack.append(next_cell)
        members.append(np.asarray(current_members, dtype=np.int32))
    return labels, members


def sample_reachable_3d_start_goal(free_members: Sequence[np.ndarray], labels: np.ndarray, rng: np.random.Generator) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], float, float, int]:
    if not free_members:
        raise RuntimeError("No free 3-D component exists")
    largest_id = max(range(len(free_members)), key=lambda index: len(free_members[index]))
    members = free_members[largest_id]
    for _ in range(CONFIG["max_sampling_attempts"]):
        first, second = rng.integers(0, len(members), size=2)
        if first == second:
            continue
        start = tuple(int(value) for value in members[first])
        goal = tuple(int(value) for value in members[second])
        if min(start[0], goal[0]) * CONFIG["voxel_size_m"] + CONFIG["voxel_size_m"] / 2 < CONFIG["min_operational_altitude_m"]:
            continue
        if min(goal[0], start[0]) < int(math.ceil(CONFIG["min_operational_altitude_m"] / CONFIG["voxel_size_m"])):
            continue
        three_d = float(math.dist(start, goal))
        horizontal = float(math.hypot(start[1] - goal[1], start[2] - goal[2]))
        if (
            CONFIG["min_3d_distance_voxels"] <= three_d <= CONFIG["max_3d_distance_voxels"]
            and CONFIG["min_horizontal_distance_voxels"] <= horizontal <= CONFIG["max_horizontal_distance_voxels"]
            and labels[start] == labels[goal]
        ):
            return start, goal, three_d, horizontal, largest_id
    raise RuntimeError("Could not sample valid reachable 3-D start/goal")


def _trial_record(scene: str, trial_id: int, start: Tuple[int, int, int], goal: Tuple[int, int, int], grid: np.ndarray, elapsed_ms: float, path: Optional[Sequence[Tuple[int, int, int]]], cost: float, expanded: int, generated: int, failure_reason: str = "") -> dict:
    straight = float(math.dist(start, goal) * CONFIG["voxel_size_m"])
    horizontal = float(math.hypot(start[1] - goal[1], start[2] - goal[2]) * CONFIG["voxel_size_m"])
    vertical = float(abs(start[0] - goal[0]) * CONFIG["voxel_size_m"])
    found = path is not None
    return {
        "scene": scene,
        "trial_id": trial_id,
        "start_z": start[0], "start_row": start[1], "start_col": start[2],
        "goal_z": goal[0], "goal_row": goal[1], "goal_col": goal[2],
        "start_x": np.nan, "start_y": np.nan, "start_altitude_m": (start[0] + 0.5) * CONFIG["voxel_size_m"],
        "goal_x": np.nan, "goal_y": np.nan, "goal_altitude_m": (goal[0] + 0.5) * CONFIG["voxel_size_m"],
        "horizontal_distance_m": horizontal,
        "vertical_distance_m": vertical,
        "straight_3d_distance_m": straight,
        "path_found": found,
        "planning_time_ms": elapsed_ms,
        "path_cost_m": cost if found else np.nan,
        "path_node_count": len(path) if found else 0,
        "expanded_nodes": expanded,
        "generated_nodes": generated,
        "detour_ratio": cost / straight if found and straight else np.nan,
        "voxel_nz": grid.shape[0], "voxel_rows": grid.shape[1], "voxel_cols": grid.shape[2],
        "free_voxels": int((grid == 0).sum()), "occupied_voxels": int(grid.sum()), "occupied_ratio": float(grid.mean()),
        "neighbor_mode": CONFIG["neighbor_mode"], "prevent_corner_cutting": CONFIG["prevent_3d_corner_cutting"],
        "random_seed": CONFIG["random_seed"], "failure_reason": failure_reason,
    }


def run_3d_warmup(occupancy: np.ndarray, members: Sequence[np.ndarray], labels: np.ndarray, rng: np.random.Generator, workspace: AStarWorkspace, offsets: Sequence) -> None:
    for _ in range(CONFIG["warmup_trials"]):
        start, goal, _, _, _ = sample_reachable_3d_start_goal(members, labels, rng)
        astar_3d(occupancy, start, goal, workspace, offsets)


def run_scene_3d_benchmark(scene: str, occupancy: np.ndarray, metadata: dict, rng: np.random.Generator) -> Tuple[List[dict], List[dict], dict]:
    labels, members = compute_3d_connected_components(occupancy)
    largest_id = max(range(len(members)), key=lambda index: len(members[index]))
    workspace = AStarWorkspace.create(occupancy.shape)
    offsets = _neighbor_offsets()
    run_3d_warmup(occupancy, members, labels, rng, workspace, offsets)
    records: List[dict] = []
    examples: List[dict] = []
    for trial_id in range(1, CONFIG["num_trials_per_scene"] + 1):
        start, goal, _, _, _ = sample_reachable_3d_start_goal(members, labels, rng)
        start_ns = perf_counter_ns()
        path, cost, expanded, generated = astar_3d(occupancy, start, goal, workspace, offsets)
        elapsed_ms = (perf_counter_ns() - start_ns) / 1_000_000.0
        if path is None:
            record = _trial_record(scene, trial_id, start, goal, occupancy, elapsed_ms, None, cost, expanded, generated, "no_path")
        else:
            validate_3d_path(occupancy, path, start, goal, offsets, cost)
            record = _trial_record(scene, trial_id, start, goal, occupancy, elapsed_ms, path, cost, expanded, generated)
            if len(examples) < CONFIG["num_example_paths"]:
                examples.append({"path": path, "record": record.copy()})
        records.append(record)
        if trial_id % 100 == 0:
            LOGGER.info("[%s] completed %d/%d trials", scene.title(), trial_id, CONFIG["num_trials_per_scene"])
    return records, examples, {"component_count": len(members), "largest_component_voxels": len(members[largest_id]), "free_voxels": int((occupancy == 0).sum()), "occupied_voxels": int(occupancy.sum()), "offset_count": len(offsets)}


def calculate_3d_benchmark_statistics(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scene, group in trials.groupby("scene", sort=False):
        valid = group[group["path_found"] == True]
        times = valid["planning_time_ms"].to_numpy(dtype=float)
        mean_path_cost_m = float(valid["path_cost_m"].mean()) if len(valid) else math.nan
        mean_straight_3d_distance_m = float(valid["straight_3d_distance_m"].mean()) if len(valid) else math.nan
        mean_horizontal_distance_m = float(valid["horizontal_distance_m"].mean()) if len(valid) else math.nan
        mean_vertical_distance_m = float(valid["vertical_distance_m"].mean()) if len(valid) else math.nan
        mean_detour_ratio = float(valid["detour_ratio"].mean()) if len(valid) else math.nan
        mean_expanded_nodes = float(valid["expanded_nodes"].mean()) if len(valid) else math.nan
        mean_generated_nodes = float(valid["generated_nodes"].mean()) if len(valid) else math.nan
        rows.append({
            "scene": scene,
            "num_valid_trials": len(valid), "num_failed_trials": len(group) - len(valid), "success_rate": len(valid) / len(group),
            "mean_time_ms": float(times.mean()) if len(times) else math.nan, "std_time_ms": float(times.std()) if len(times) else math.nan,
            "min_time_ms": float(times.min()) if len(times) else math.nan, "median_time_ms": float(np.median(times)) if len(times) else math.nan,
            "p90_time_ms": float(np.percentile(times, 90)) if len(times) else math.nan, "p95_time_ms": float(np.percentile(times, 95)) if len(times) else math.nan,
            "p99_time_ms": float(np.percentile(times, 99)) if len(times) else math.nan, "max_time_ms": float(times.max()) if len(times) else math.nan,
            "mean_path_length_m": mean_path_cost_m, "mean_3d_straight_distance_m": mean_straight_3d_distance_m,
            "mean_horizontal_distance_m": mean_horizontal_distance_m, "mean_vertical_distance_m": mean_vertical_distance_m,
            "mean_detour_ratio": mean_detour_ratio, "mean_expanded_nodes": mean_expanded_nodes, "mean_generated_nodes": mean_generated_nodes,
            "voxel_nz": int(group["voxel_nz"].iloc[0]), "voxel_rows": int(group["voxel_rows"].iloc[0]), "voxel_cols": int(group["voxel_cols"].iloc[0]),
            "free_voxels": int(group["free_voxels"].iloc[0]), "occupied_voxels": int(group["occupied_voxels"].iloc[0]), "occupied_ratio": float(group["occupied_ratio"].iloc[0]),
        })
    return pd.DataFrame(rows)


def _plot_3d_example_paths(occupancy: np.ndarray, examples: Sequence[dict], metadata: dict, scene: str, output_path: Path) -> None:
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    ax3d = fig.add_subplot(131, projection="3d")
    axxy = fig.add_subplot(132)
    axalt = fig.add_subplot(133)
    colors = ("#d946ef", "#16a34a", "#f97316", "#2563eb", "#dc2626")
    for index, example in enumerate(examples):
        path = np.asarray(example["path"], dtype=int)
        color = colors[index % len(colors)]
        x = metadata["bounds"]["min_x"] + (path[:, 2] + 0.5) * metadata["voxel_size_x_m"]
        y = metadata["bounds"]["max_y"] - (path[:, 1] + 0.5) * metadata["voxel_size_y_m"]
        z = (path[:, 0] + 0.5) * metadata["voxel_size_z_m"]
        label = f"Trial {example['record']['trial_id']}"
        ax3d.plot(x, y, z, color=color, linewidth=1.1, label=label)
        ax3d.scatter([x[0], x[-1]], [y[0], y[-1]], [z[0], z[-1]], color=color, s=18)
        axxy.plot(x, y, color=color, linewidth=1.1, label=label)
        axxy.scatter([x[0], x[-1]], [y[0], y[-1]], color=color, s=18)
        cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(np.column_stack((x, y, z)), axis=0), axis=1))]
        axalt.plot(cumulative, z, color=color, linewidth=1.1, label=label)
    ax3d.set_title("3-D paths"); ax3d.set_xlabel("Easting (m)"); ax3d.set_ylabel("Northing (m)"); ax3d.set_zlabel("Altitude (m)")
    axxy.set_title("XY projection"); axxy.set_xlabel("Easting (m)"); axxy.set_ylabel("Northing (m)"); axxy.set_aspect("equal")
    axalt.set_title("Altitude profile"); axalt.set_xlabel("Cumulative path distance (m)"); axalt.set_ylabel("Flight altitude (m)")
    ax3d.legend(fontsize=7); axxy.legend(fontsize=7); axalt.legend(fontsize=7)
    fig.suptitle(f"{scene.title()} 3-D A* example paths", fontsize=14)
    fig.savefig(output_path, dpi=CONFIG["dpi"], facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_horizontal_slices(occupancy: np.ndarray, metadata: dict, scene: str, output_path: Path) -> None:
    indices = [min(int(round((height - CONFIG["z_min_m"]) / CONFIG["voxel_size_m"])), occupancy.shape[0] - 1) for height in FLIGHT_LAYERS_M]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    cmap = ListedColormap(["#ffffff", "#5f646c"])
    for ax, z_index, height in zip(axes.ravel(), indices, FLIGHT_LAYERS_M):
        display = np.flipud(occupancy[z_index])
        ax.imshow(display, cmap=cmap, vmin=0, vmax=1, origin="upper", interpolation="none", extent=(metadata["bounds"]["min_x"], metadata["bounds"]["max_x"], metadata["bounds"]["min_y"], metadata["bounds"]["max_y"]))
        ax.set_title(f"Voxel slice [{height} m, {height + 10} m)")
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)"); ax.set_aspect("equal")
    fig.suptitle(f"{scene.title()} 3-D voxel horizontal slices", fontsize=14)
    fig.savefig(output_path, dpi=CONFIG["dpi"], facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_time_distribution(trials: pd.DataFrame, scene: str, output_path: Path) -> None:
    values = trials.loc[(trials.scene == scene) & (trials.path_found == True), "planning_time_ms"]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist(values, bins=40, color="#2563eb", alpha=0.8)
    ax.set_xlabel("3-D A* planning time (ms)"); ax.set_ylabel("Trial count"); ax.set_title(f"{scene.title()} 3-D A* time distribution"); ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=CONFIG["dpi"], facecolor="white", bbox_inches="tight"); plt.close(fig)


def _plot_length_vs_time(trials: pd.DataFrame, scene: str, output_path: Path) -> None:
    data = trials[(trials.scene == scene) & (trials.path_found == True)]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.scatter(data["path_cost_m"], data["planning_time_ms"], s=8, alpha=0.25, color="#dc2626")
    ax.set_xlabel("3-D path length (m)"); ax.set_ylabel("3-D A* planning time (ms)"); ax.set_title(f"{scene.title()} 3-D path length versus time"); ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=CONFIG["dpi"], facecolor="white", bbox_inches="tight"); plt.close(fig)


def _system_info() -> dict:
    info = {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()}
    if psutil is not None:
        info["memory_total_mb"] = psutil.virtual_memory().total / 1024 ** 2
        info["memory_available_mb"] = psutil.virtual_memory().available / 1024 ** 2
    return info


def _save_json_safe(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def _run_smoke_test(scene: str, occupancy: np.ndarray, metadata: dict, rng: np.random.Generator) -> None:
    labels, members = compute_3d_connected_components(occupancy)
    workspace = AStarWorkspace.create(occupancy.shape)
    offsets = _neighbor_offsets()
    for _ in range(CONFIG["smoke_test_trials"]):
        start, goal, _, _, _ = sample_reachable_3d_start_goal(members, labels, rng)
        path, cost, _, _ = astar_3d(occupancy, start, goal, workspace, offsets)
        if path is None:
            raise AssertionError(f"Smoke test found no path for {scene}: {start}->{goal}")
        validate_3d_path(occupancy, path, start, goal, offsets, cost)
    LOGGER.info("[%s] smoke test passed (%d paths)", scene.title(), CONFIG["smoke_test_trials"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "truck_uav_modules" / "data"
    voxel_dir = data_dir / "processed_3d_voxel_grids"
    benchmark_dir = data_dir / "path_planning_3d_benchmark"
    figure_dir = project_root / "truck_uav_modules" / "figure"
    voxel_dir.mkdir(parents=True, exist_ok=True); benchmark_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    metadata_by_scene = load_grid_metadata(data_dir)
    rng = np.random.default_rng(CONFIG["random_seed"])
    all_trials: List[dict] = []
    all_summaries = []
    total_started = time.perf_counter()
    for scene in SCENES:
        mapping = load_launch_recovery_mapping(data_dir, scene)
        occupancy, original, grid_spec, details = build_3d_voxel_grid(scene, data_dir, metadata_by_scene[scene], mapping)
        metadata_3d = details["metadata"]
        npy_path, npz_path = save_3d_voxel_grid(occupancy, metadata_3d, voxel_dir, scene)
        details["modifications"].to_csv(voxel_dir / f"{scene}_launch_recovery_3d_voxel_modifications.csv", index=False)
        validate_voxel_coordinate_conversion(occupancy, metadata_3d, rng)
        _run_smoke_test(scene, occupancy, metadata_3d, rng)
        records, examples, component_info = run_scene_3d_benchmark(scene, occupancy, metadata_3d, rng)
        all_trials.extend(records)
        all_summaries.append({"scene": scene, **component_info, **metadata_3d, "npy_path": str(npy_path), "npz_path": str(npz_path)})
        _plot_3d_example_paths(occupancy, examples, metadata_3d, scene, figure_dir / f"{scene}_astar_3d_example_paths.png")
        _plot_horizontal_slices(occupancy, metadata_3d, scene, figure_dir / f"{scene}_3d_voxel_horizontal_slices.png")
    trials = pd.DataFrame(all_trials)
    summary = calculate_3d_benchmark_statistics(trials)
    trials.to_csv(benchmark_dir / "astar_3d_benchmark_trials.csv", index=False)
    summary.to_csv(benchmark_dir / "astar_3d_benchmark_summary.csv", index=False)
    for scene in SCENES:
        _plot_time_distribution(trials, scene, figure_dir / f"{scene}_astar_3d_time_distribution.png")
        _plot_length_vs_time(trials, scene, figure_dir / f"{scene}_astar_3d_length_vs_time.png")
    valid = trials[trials["path_found"] == True]
    report = {
        "benchmark": "3-D voxel A* path planning",
        "config": CONFIG,
        "scenes": all_summaries,
        "summary": summary.to_dict(orient="records"),
        "total_trials": int(len(trials)), "total_valid_trials": int(len(valid)),
        "overall_mean_time_ms": float(valid["planning_time_ms"].mean()),
        "overall_median_time_ms": float(valid["planning_time_ms"].median()),
        "overall_p95_time_ms": float(valid["planning_time_ms"].quantile(0.95)),
        "total_benchmark_runtime_s": time.perf_counter() - total_started,
        "system": _system_info(),
        "outputs": {"voxel_dir": str(voxel_dir), "benchmark_dir": str(benchmark_dir), "figure_dir": str(figure_dir)},
    }
    _save_json_safe(benchmark_dir / "astar_3d_benchmark_report.json", report)
    print("=" * 72)
    print("3D VOXEL A* PATH-PLANNING BENCHMARK")
    print("Voxel size: 10 m x 10 m x 10 m")
    print("Altitude range: [0, 120) m")
    print(f"Minimum operational altitude: {CONFIG['min_operational_altitude_m']} m")
    print("Neighbor mode: 26; prevent 3-D corner cutting: True")
    print(f"Trials per scene: {CONFIG['num_trials_per_scene']}")
    for row in summary.itertuples(index=False):
        print(f"\n[{row.scene.title()}]")
        print(f"Voxel shape: ({row.voxel_nz}, {row.voxel_rows}, {row.voxel_cols})")
        print(f"Occupied voxels: {row.occupied_voxels}; occupied ratio: {row.occupied_ratio:.4%}")
        print(f"Valid trials: {row.num_valid_trials}; failed trials: {row.num_failed_trials}")
        print(f"Mean planning time: {row.mean_time_ms:.4f} ms")
        print(f"Median planning time: {row.median_time_ms:.4f} ms")
        print(f"P95 planning time: {row.p95_time_ms:.4f} ms")
        print(f"Maximum planning time: {row.max_time_ms:.4f} ms")
        print(f"Mean path length: {row.mean_path_length_m:.2f} m")
        print(f"Mean expanded nodes: {row.mean_expanded_nodes:.2f}")
    print("\n" + "=" * 72)
    print("OVERALL RESULT")
    print(f"Total valid trials: {len(valid)}")
    print(f"Overall mean planning time: {valid['planning_time_ms'].mean():.4f} ms")
    print(f"Overall median planning time: {valid['planning_time_ms'].median():.4f} ms")
    print(f"Overall P95 planning time: {valid['planning_time_ms'].quantile(0.95):.4f} ms")
    print(f"Total benchmark runtime: {report['total_benchmark_runtime_s']:.2f} s")
    print("=" * 72)


if __name__ == "__main__":
    main()
