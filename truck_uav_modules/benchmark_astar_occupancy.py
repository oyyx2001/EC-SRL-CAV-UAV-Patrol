"""Benchmark 2-D A* planning on the existing 10 m occupancy grids.

Install dependencies with:
    pip install numpy pandas matplotlib

The benchmark reads precomputed grids only. It does not regenerate grids,
modify the existing planner, run reinforcement learning, or write source data.
Grid values are 0 (free) and 1 (occupied). The default run executes 1,000
trials for each of two scenes and six flight layers.
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import os
import platform
import sys
from collections import deque
from pathlib import Path
from time import perf_counter_ns
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


LOGGER = logging.getLogger("astar_benchmark")
SCENES = ("xinjiekou", "baijiahu")

# Keep all experiment parameters in one place for reproducibility.
CONFIG = {
    "grid_size_m": 10.0,
    "flight_layers_m": [20, 40, 60, 80, 100, 120],
    "num_trials": 1000,
    "warmup_trials": 20,
    "random_seed": 20260727,
    "allow_diagonal": True,
    "prevent_corner_cutting": True,
    "min_euclidean_distance_cells": 10,
    "max_sampling_attempts": 10000,
    "save_example_paths": True,
    "num_example_paths_per_layer": 3,
}

MOVE_4 = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0))
MOVE_DIAGONAL = (
    (-1, -1, float(np.sqrt(2))),
    (-1, 1, float(np.sqrt(2))),
    (1, -1, float(np.sqrt(2))),
    (1, 1, float(np.sqrt(2))),
)


def find_grid_files(grid_data_dir: Path) -> Dict[Tuple[str, int], Path]:
    """Locate existing NPY grids, falling back to NPZ when NPY is absent."""
    files: Dict[Tuple[str, int], Path] = {}
    for scene in SCENES:
        for layer in CONFIG["flight_layers_m"]:
            stem = f"{scene}_occupancy_grid_{layer}m"
            npy = grid_data_dir / f"{stem}.npy"
            npz = grid_data_dir / f"{stem}.npz"
            if npy.exists():
                files[(scene, layer)] = npy
            elif npz.exists():
                files[(scene, layer)] = npz
            else:
                raise FileNotFoundError(f"Missing both {npy.name} and {npz.name} in {grid_data_dir}")
    return files


def load_occupancy_grid(grid_path: Path) -> np.ndarray:
    """Load and validate one uint8 occupancy grid."""
    loaded = np.load(grid_path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "occupancy_grid" not in loaded.files:
            loaded.close()
            raise ValueError(f"NPZ file has no occupancy_grid key: {grid_path}")
        grid = loaded["occupancy_grid"]
        loaded.close()
    else:
        grid = loaded
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError(f"Grid must be 2-D, got {grid.ndim}-D: {grid_path}")
    if not np.isin(grid, (0, 1)).all():
        raise ValueError(f"Grid contains values other than 0 and 1: {grid_path}")
    if grid.dtype != np.uint8:
        LOGGER.warning("Converting %s from %s to uint8.", grid_path.name, grid.dtype)
        grid = grid.astype(np.uint8, copy=False)
    return grid


def _moves(allow_diagonal: bool) -> Tuple[Tuple[int, int, float], ...]:
    return MOVE_4 + MOVE_DIAGONAL if allow_diagonal else MOVE_4


def _valid_neighbor(
    grid: np.ndarray,
    row: int,
    col: int,
    next_row: int,
    next_col: int,
    prevent_corner_cutting: bool,
) -> bool:
    rows, cols = grid.shape
    if not (0 <= next_row < rows and 0 <= next_col < cols):
        return False
    if grid[next_row, next_col] != 0:
        return False
    if prevent_corner_cutting and next_row != row and next_col != col:
        return grid[next_row, col] == 0 and grid[row, next_col] == 0
    return True


def compute_connected_components(grid: np.ndarray, allow_diagonal: bool) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Label free-cell components using the same neighborhood as A*."""
    labels = np.full(grid.shape, -1, dtype=np.int32)
    component_members: List[np.ndarray] = []
    moves = _moves(allow_diagonal)
    component_id = 0
    for start_row, start_col in zip(*np.where(grid == 0)):
        if labels[start_row, start_col] >= 0:
            continue
        queue = deque([(int(start_row), int(start_col))])
        labels[start_row, start_col] = component_id
        members: List[Tuple[int, int]] = []
        while queue:
            row, col = queue.popleft()
            members.append((row, col))
            for delta_row, delta_col, _ in moves:
                next_row, next_col = row + delta_row, col + delta_col
                if not _valid_neighbor(grid, row, col, next_row, next_col, CONFIG["prevent_corner_cutting"]):
                    continue
                if labels[next_row, next_col] < 0:
                    labels[next_row, next_col] = component_id
                    queue.append((next_row, next_col))
        component_members.append(np.asarray(members, dtype=np.int32))
        component_id += 1
    return labels, component_members


def sample_reachable_start_goal(
    free_cells: np.ndarray,
    component_labels: np.ndarray,
    component_members: Sequence[np.ndarray],
    min_distance_cells: int,
    max_sampling_attempts: int,
    rng: np.random.Generator,
) -> Tuple[Tuple[int, int], Tuple[int, int], int, float]:
    """Sample two far enough cells from the largest A*-reachable component."""
    if not component_members:
        raise RuntimeError("Grid has no free connected component")
    largest_id = max(range(len(component_members)), key=lambda index: len(component_members[index]))
    members = component_members[largest_id]
    if len(members) < 2:
        raise RuntimeError("Largest free component has fewer than two cells")
    for _ in range(max_sampling_attempts):
        first_index, second_index = rng.integers(0, len(members), size=2)
        if first_index == second_index:
            continue
        start = tuple(int(value) for value in members[first_index])
        goal = tuple(int(value) for value in members[second_index])
        distance_cells = float(np.hypot(goal[0] - start[0], goal[1] - start[1]))
        if distance_cells >= min_distance_cells and component_labels[start] == component_labels[goal]:
            return start, goal, largest_id, distance_cells
    raise RuntimeError("Could not sample a sufficiently distant reachable pair")


def astar_search(
    occupancy_grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    allow_diagonal: bool = True,
    prevent_corner_cutting: bool = True,
) -> Tuple[Optional[List[Tuple[int, int]]], float, int]:
    """Run standard A* and return (path, cell_cost, expanded_nodes)."""
    if occupancy_grid[start] != 0 or occupancy_grid[goal] != 0:
        return None, float("inf"), 0
    if start == goal:
        return [start], 0.0, 0
    rows, cols = occupancy_grid.shape
    moves = _moves(allow_diagonal)
    goal_row, goal_col = goal

    def heuristic(row: int, col: int) -> float:
        dx = abs(goal_col - col)
        dy = abs(goal_row - row)
        if allow_diagonal:
            return max(dx, dy) + (float(np.sqrt(2)) - 1.0) * min(dx, dy)
        return float(dx + dy)

    open_heap: List[Tuple[float, float, int, int]] = [(heuristic(*start), 0.0, start[0], start[1])]
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    closed: set[Tuple[int, int]] = set()
    expanded_nodes = 0

    while open_heap:
        _, current_cost, row, col = heapq.heappop(open_heap)
        current = (row, col)
        if current in closed:
            continue
        closed.add(current)
        expanded_nodes += 1
        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return path, current_cost, expanded_nodes
        for delta_row, delta_col, step_cost in moves:
            next_row, next_col = row + delta_row, col + delta_col
            if not _valid_neighbor(occupancy_grid, row, col, next_row, next_col, prevent_corner_cutting):
                continue
            neighbor = (next_row, next_col)
            if neighbor in closed:
                continue
            tentative = current_cost + step_cost
            if tentative < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative + heuristic(next_row, next_col), tentative, next_row, next_col))
    return None, float("inf"), expanded_nodes


def validate_path(
    grid: np.ndarray,
    path: Sequence[Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    prevent_corner_cutting: bool,
) -> None:
    """Assert endpoints, occupancy, adjacency and diagonal corner safety."""
    if not path or path[0] != start or path[-1] != goal:
        raise AssertionError("A* path endpoints are invalid")
    if any(grid[row, col] != 0 for row, col in path):
        raise AssertionError("A* path crosses an occupied cell")
    for current, following in zip(path[:-1], path[1:]):
        delta_row = abs(following[0] - current[0])
        delta_col = abs(following[1] - current[1])
        if max(delta_row, delta_col) != 1 or (delta_row == 0 and delta_col == 0):
            raise AssertionError("A* path contains a non-adjacent step")
        if prevent_corner_cutting and delta_row == 1 and delta_col == 1:
            if grid[current[0], following[1]] != 0 or grid[following[0], current[1]] != 0:
                raise AssertionError("A* path cuts through an obstacle corner")


def _path_length_cells(path: Sequence[Tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:])))


def _sample_pair(
    grid: np.ndarray,
    labels: np.ndarray,
    members: Sequence[np.ndarray],
    rng: np.random.Generator,
) -> Tuple[Tuple[int, int], Tuple[int, int], int, float]:
    free_cells = np.argwhere(grid == 0)
    return sample_reachable_start_goal(
        free_cells,
        labels,
        members,
        CONFIG["min_euclidean_distance_cells"],
        CONFIG["max_sampling_attempts"],
        rng,
    )


def _trial_record(
    scene: str,
    layer: int,
    trial_id: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    distance_cells: float,
    grid: np.ndarray,
    elapsed_ms: float,
    path: Optional[Sequence[Tuple[int, int]]],
    path_cost: float,
    expanded_nodes: int,
    failure_reason: str = "",
) -> dict:
    path_found = path is not None
    path_length_m = path_cost * CONFIG["grid_size_m"] if path_found else np.nan
    straight_m = distance_cells * CONFIG["grid_size_m"]
    return {
        "scene": scene,
        "flight_layer_m": layer,
        "trial_id": trial_id,
        "start_row": start[0],
        "start_col": start[1],
        "goal_row": goal[0],
        "goal_col": goal[1],
        "straight_line_distance_cells": distance_cells,
        "straight_line_distance_m": straight_m,
        "path_found": path_found,
        "planning_time_ms": elapsed_ms,
        "path_cost_cells": path_cost if path_found else np.nan,
        "path_length_m": path_length_m,
        "path_node_count": len(path) if path_found else 0,
        "expanded_nodes": expanded_nodes,
        "grid_rows": grid.shape[0],
        "grid_cols": grid.shape[1],
        "blocked_ratio": float(grid.mean()),
        "random_seed": CONFIG["random_seed"],
        "failure_reason": failure_reason,
    }


def run_warmup_trials(grid: np.ndarray, labels: np.ndarray, members: Sequence[np.ndarray], rng: np.random.Generator) -> None:
    """Warm up Python/A* code without including it in timing statistics."""
    for _ in range(CONFIG["warmup_trials"]):
        start, goal, _, _ = _sample_pair(grid, labels, members, rng)
        astar_search(grid, start, goal, CONFIG["allow_diagonal"], CONFIG["prevent_corner_cutting"])


def run_layer_benchmark(scene: str, layer: int, grid: np.ndarray, rng: np.random.Generator) -> Tuple[List[dict], List[dict]]:
    """Run warmup plus the configured number of timed A* trials."""
    labels, members = compute_connected_components(grid, CONFIG["allow_diagonal"])
    largest_id = max(range(len(members)), key=lambda index: len(members[index]))
    LOGGER.info("[%s][%dm] free=%d, components=%d, largest_component=%d", scene.title(), layer, int((grid == 0).sum()), len(members), len(members[largest_id]))
    run_warmup_trials(grid, labels, members, rng)
    records: List[dict] = []
    examples: List[dict] = []
    for trial_id in range(1, CONFIG["num_trials"] + 1):
        start, goal, _, distance_cells = _sample_pair(grid, labels, members, rng)
        start_ns = perf_counter_ns()
        path, path_cost, expanded_nodes = astar_search(
            grid, start, goal, CONFIG["allow_diagonal"], CONFIG["prevent_corner_cutting"]
        )
        elapsed_ms = (perf_counter_ns() - start_ns) / 1_000_000.0
        if path is None:
            record = _trial_record(scene, layer, trial_id, start, goal, distance_cells, grid, elapsed_ms, None, path_cost, expanded_nodes, "no_path")
        else:
            validate_path(grid, path, start, goal, CONFIG["prevent_corner_cutting"])
            record = _trial_record(scene, layer, trial_id, start, goal, distance_cells, grid, elapsed_ms, path, path_cost, expanded_nodes)
            if CONFIG["save_example_paths"] and len(examples) < CONFIG["num_example_paths_per_layer"]:
                examples.append({"start": start, "goal": goal, "path": list(path), "record": record.copy()})
        records.append(record)
        if trial_id % 100 == 0:
            LOGGER.info("[%s][%dm] completed %d/%d trials", scene.title(), layer, trial_id, CONFIG["num_trials"])
    return records, examples


def calculate_group_statistics(trial_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate timing, path and search statistics for each scene/layer."""
    rows = []
    for (scene, layer), group in trial_dataframe.groupby(["scene", "flight_layer_m"], sort=False):
        valid = group[group["path_found"] == True]
        times = valid["planning_time_ms"].to_numpy(dtype=float)
        lengths = valid["path_length_m"].to_numpy(dtype=float)
        straight = valid["straight_line_distance_m"].to_numpy(dtype=float)
        detour = lengths / straight
        expanded = valid["expanded_nodes"].to_numpy(dtype=float)
        total_cells = int(group["grid_rows"].iloc[0] * group["grid_cols"].iloc[0])
        blocked_cells = int(round(float(group["blocked_ratio"].iloc[0]) * total_cells))
        row = {
            "scene": scene,
            "flight_layer_m": int(layer),
            "num_valid_trials": int(len(valid)),
            "num_failed_trials": int(len(group) - len(valid)),
            "success_rate": len(valid) / len(group) if len(group) else 0.0,
            "mean_time_ms": float(np.mean(times)) if len(times) else np.nan,
            "std_time_ms": float(np.std(times)) if len(times) else np.nan,
            "min_time_ms": float(np.min(times)) if len(times) else np.nan,
            "median_time_ms": float(np.median(times)) if len(times) else np.nan,
            "p90_time_ms": float(np.percentile(times, 90)) if len(times) else np.nan,
            "p95_time_ms": float(np.percentile(times, 95)) if len(times) else np.nan,
            "p99_time_ms": float(np.percentile(times, 99)) if len(times) else np.nan,
            "max_time_ms": float(np.max(times)) if len(times) else np.nan,
            "mean_path_length_m": float(np.mean(lengths)) if len(lengths) else np.nan,
            "mean_straight_distance_m": float(np.mean(straight)) if len(straight) else np.nan,
            "mean_detour_ratio": float(np.mean(detour)) if len(detour) else np.nan,
            "mean_expanded_nodes": float(np.mean(expanded)) if len(expanded) else np.nan,
            "grid_rows": int(group["grid_rows"].iloc[0]),
            "grid_cols": int(group["grid_cols"].iloc[0]),
            "free_cells": total_cells - blocked_cells,
            "blocked_cells": blocked_cells,
            "blocked_ratio": float(group["blocked_ratio"].iloc[0]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_mean_time_by_layer(summary: pd.DataFrame, scene: str, output_path: Path) -> None:
    data = summary[summary.scene == scene].sort_values("flight_layer_m")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.errorbar(data.flight_layer_m, data.mean_time_ms, yerr=data.std_time_ms, marker="o", capsize=3, color="#2563eb", label="Mean +/- standard deviation")
    ax.set_xlabel("Flight altitude layer (m)")
    ax.set_ylabel("Mean A* planning time (ms)")
    ax.set_title(f"{scene.title()} A* mean planning time by layer")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save_figure(fig, output_path)


def plot_time_boxplot(trials: pd.DataFrame, scene: str, output_path: Path) -> None:
    data = trials[(trials.scene == scene) & (trials.path_found == True)]
    layers = sorted(data.flight_layer_m.unique())
    values = [data.loc[data.flight_layer_m == layer, "planning_time_ms"] for layer in layers]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.boxplot(values, tick_labels=[str(layer) for layer in layers], showfliers=False)
    ax.set_xlabel("Flight altitude layer (m)")
    ax.set_ylabel("A* planning time (ms)")
    ax.set_title(f"{scene.title()} A* planning-time distribution")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, output_path)


def plot_path_length_vs_time(trials: pd.DataFrame, scene: str, output_path: Path) -> None:
    data = trials[(trials.scene == scene) & (trials.path_found == True)]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for layer, group in data.groupby("flight_layer_m"):
        ax.scatter(group.path_length_m, group.planning_time_ms, s=7, alpha=0.25, label=f"{layer} m")
    ax.set_xlabel("Path length (m)")
    ax.set_ylabel("A* planning time (ms)")
    ax.set_title(f"{scene.title()} path length versus planning time")
    ax.grid(alpha=0.25)
    ax.legend(title="Layer", fontsize=7, ncol=2)
    _save_figure(fig, output_path)


def plot_example_paths(grid: np.ndarray, examples: Sequence[dict], scene: str, layer: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.imshow(grid, cmap=ListedColormap(["#ffffff", "#575b63"]), vmin=0, vmax=1, origin="upper", interpolation="none")
    colors = ("#d946ef", "#16a34a", "#f97316")
    for index, example in enumerate(examples):
        path = np.asarray(example["path"])
        ax.plot(path[:, 1], path[:, 0], color=colors[index % len(colors)], linewidth=1.3, label=f"Trial {example['record']['trial_id']}")
        ax.scatter([example["start"][1]], [example["start"][0]], color=colors[index % len(colors)], marker="o", s=24, edgecolor="black", linewidth=0.4)
        ax.scatter([example["goal"][1]], [example["goal"][0]], color=colors[index % len(colors)], marker="x", s=28, linewidth=1.0)
    ax.set_title(f"{scene.title()} A* example paths at {layer} m")
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    ax.legend(fontsize=8)
    _save_figure(fig, output_path)


def _system_info() -> dict:
    versions = {}
    for module_name in ("numpy", "pandas", "matplotlib"):
        module = __import__(module_name)
        versions[module_name] = getattr(module, "__version__", "unknown")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "dependencies": versions,
    }


def export_results(
    trials: pd.DataFrame,
    summary: pd.DataFrame,
    report: dict,
    benchmark_dir: Path,
) -> None:
    trials.to_csv(benchmark_dir / "astar_benchmark_trials.csv", index=False)
    summary.to_csv(benchmark_dir / "astar_benchmark_summary.csv", index=False)
    (benchmark_dir / "astar_benchmark_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def _print_summary(summary: pd.DataFrame, trials: pd.DataFrame) -> None:
    print("=" * 60)
    print("A* OCCUPANCY-GRID BENCHMARK")
    print(f"Grid resolution: {CONFIG['grid_size_m']} m")
    print(f"Trials per scene-layer: {CONFIG['num_trials']}")
    print("=" * 60)
    for row in summary.itertuples(index=False):
        print(f"\n[{row.scene.title()}][{row.flight_layer_m} m]")
        print(f"Valid trials: {row.num_valid_trials}")
        print(f"Failed trials: {row.num_failed_trials}")
        print(f"Mean planning time: {row.mean_time_ms:.4f} ms")
        print(f"Median planning time: {row.median_time_ms:.4f} ms")
        print(f"P95 planning time: {row.p95_time_ms:.4f} ms")
        print(f"Mean path length: {row.mean_path_length_m:.2f} m")
        print(f"Mean expanded nodes: {row.mean_expanded_nodes:.2f}")
    valid_times = trials.loc[trials.path_found == True, "planning_time_ms"].to_numpy(dtype=float)
    print("\n" + "=" * 60)
    print("OVERALL RESULT")
    print(f"Total valid trials: {len(valid_times)}")
    print(f"Overall mean planning time: {np.mean(valid_times):.4f} ms")
    print(f"Overall median planning time: {np.median(valid_times):.4f} ms")
    print(f"Overall P95 planning time: {np.percentile(valid_times, 95):.4f} ms")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-data-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--benchmark-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    grid_data_dir = args.grid_data_dir or project_root / "truck_uav_modules" / "data" / "processed_grid_layers"
    figure_dir = args.figure_dir or project_root / "truck_uav_modules" / "figure"
    benchmark_dir = args.benchmark_dir or project_root / "truck_uav_modules" / "data" / "path_planning_benchmark"
    figure_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    grid_files = find_grid_files(grid_data_dir)
    grids: Dict[Tuple[str, int], np.ndarray] = {}
    for key, path in grid_files.items():
        grids[key] = load_occupancy_grid(path)
        LOGGER.info("Loaded %s: shape=%s, dtype=%s, blocked_ratio=%.4f", path.name, grids[key].shape, grids[key].dtype, float(grids[key].mean()))

    rng = np.random.default_rng(CONFIG["random_seed"])
    all_records: List[dict] = []
    all_examples: Dict[Tuple[str, int], List[dict]] = {}
    for scene in SCENES:
        for layer in CONFIG["flight_layers_m"]:
            records, examples = run_layer_benchmark(scene, layer, grids[(scene, layer)], rng)
            all_records.extend(records)
            all_examples[(scene, layer)] = examples

    trials = pd.DataFrame(all_records)
    summary = calculate_group_statistics(trials)
    report = {
        "benchmark": "A* occupancy-grid path planning",
        "grid_data_dir": str(grid_data_dir),
        "figure_dir": str(figure_dir),
        "benchmark_dir": str(benchmark_dir),
        "config": CONFIG,
        "grid_files": {f"{scene}_{layer}m": str(path) for (scene, layer), path in grid_files.items()},
        "system": _system_info(),
        "total_trials": int(len(trials)),
        "total_valid_trials": int(trials["path_found"].sum()),
        "overall_mean_time_ms": float(trials.loc[trials.path_found == True, "planning_time_ms"].mean()),
        "overall_median_time_ms": float(trials.loc[trials.path_found == True, "planning_time_ms"].median()),
        "overall_p95_time_ms": float(trials.loc[trials.path_found == True, "planning_time_ms"].quantile(0.95)),
    }
    export_results(trials, summary, report, benchmark_dir)

    for scene in SCENES:
        plot_mean_time_by_layer(summary, scene, figure_dir / f"{scene}_astar_mean_time_by_layer.png")
        plot_time_boxplot(trials, scene, figure_dir / f"{scene}_astar_time_boxplot.png")
        plot_path_length_vs_time(trials, scene, figure_dir / f"{scene}_path_length_vs_time.png")
        if CONFIG["save_example_paths"]:
            for layer in CONFIG["flight_layers_m"]:
                plot_example_paths(grids[(scene, layer)], all_examples[(scene, layer)], scene, layer, figure_dir / f"{scene}_astar_examples_{layer}m.png")

    _print_summary(summary, trials)
    LOGGER.info("Saved benchmark CSV/JSON results to %s", benchmark_dir)
    LOGGER.info("Saved benchmark figures to %s", figure_dir)


if __name__ == "__main__":
    main()
