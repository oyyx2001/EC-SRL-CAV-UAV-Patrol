"""Fair paired Dijkstra versus standard A* benchmark on existing 3-D voxels.

Run from the project root with:
    python -u -B truck_uav_modules/benchmark_dijkstra_vs_astar_3d.py

The script only reads existing voxel NPY files. It does not rebuild or modify
building data, launch/recovery mappings, 2-D grids, road networks, or models.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import sys
from pathlib import Path
from time import perf_counter_ns

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_and_benchmark_3d_voxel_astar as core


LOGGER = logging.getLogger("dijkstra_vs_astar_3d")
NUM_TRIALS = 1000
WARMUP_TRIALS = 20
SMOKE_TEST_TRIALS = 10
RANDOM_SEED = 20260728
TIME_EQUAL_TOLERANCE_MS = 0.01


def _record_base(scene, trial_id, algorithm, start, goal, result, elapsed_ms, occupancy):
    path, cost, expanded, generated = result
    straight = float(math.dist(start, goal) * core.CONFIG["voxel_size_m"])
    found = path is not None
    return {
        "scene": scene, "trial_id": trial_id, "algorithm": algorithm,
        "start_z": start[0], "start_row": start[1], "start_col": start[2],
        "goal_z": goal[0], "goal_row": goal[1], "goal_col": goal[2],
        "start_altitude_m": (start[0] + 0.5) * core.CONFIG["voxel_size_m"],
        "goal_altitude_m": (goal[0] + 0.5) * core.CONFIG["voxel_size_m"],
        "horizontal_distance_m": math.hypot(start[1] - goal[1], start[2] - goal[2]) * core.CONFIG["voxel_size_m"],
        "vertical_distance_m": abs(start[0] - goal[0]) * core.CONFIG["voxel_size_m"],
        "path_found": found, "planning_time_ms": elapsed_ms,
        "path_cost_m": cost if found else np.nan, "path_node_count": len(path) if found else 0,
        "expanded_nodes": expanded, "generated_nodes": generated,
        "straight_3d_distance_m": straight, "detour_ratio": cost / straight if found and straight else np.nan,
        "neighbor_mode": 26, "prevent_corner_cutting": True,
        "heuristic": "euclidean" if algorithm == "astar" else "zero",
        "heuristic_weight": 1.0 if algorithm == "astar" else 0.0,
        "random_seed": RANDOM_SEED, "failure_reason": "" if found else "no_path",
        "voxel_nz": occupancy.shape[0], "voxel_rows": occupancy.shape[1], "voxel_cols": occupancy.shape[2],
    }


def _run_one(occupancy, start, goal, algorithm, workspace, offsets):
    start_ns = perf_counter_ns()
    result = core.plan_3d_path(
        occupancy, start, goal, algorithm=algorithm, neighbor_mode=26,
        prevent_corner_cutting=True, workspace=workspace, offsets=offsets,
    )
    elapsed_ms = (perf_counter_ns() - start_ns) / 1_000_000.0
    return result, elapsed_ms


def _validate_pair(occupancy, start, goal, results):
    dijkstra_result = results["dijkstra"][0]
    astar_result = results["astar"][0]
    d_path, d_cost, _, _ = dijkstra_result
    a_path, a_cost, _, _ = astar_result
    if (d_path is None) != (a_path is None):
        raise AssertionError("Dijkstra and A* reachability differs")
    if d_path is not None:
        offsets = core._neighbor_offsets()
        core.validate_3d_path(occupancy, d_path, start, goal, offsets, d_cost)
        core.validate_3d_path(occupancy, a_path, start, goal, offsets, a_cost)
        if not np.isclose(d_cost, a_cost, rtol=1e-9, atol=core.CONFIG["cost_tolerance"]):
            raise AssertionError(f"Path costs differ: {d_cost} != {a_cost}")


def _paired_record(scene, trial_id, records):
    d = records["dijkstra"]
    a = records["astar"]
    d_time = d["planning_time_ms"]
    a_time = a["planning_time_ms"]
    d_cost = d["path_cost_m"]
    a_cost = a["path_cost_m"]
    d_expanded = d["expanded_nodes"]
    return {
        "scene": scene, "trial_id": trial_id,
        "dijkstra_time_ms": d_time, "astar_time_ms": a_time,
        "dijkstra_cost_m": d_cost, "astar_cost_m": a_cost,
        "cost_equal": bool((pd.isna(d_cost) and pd.isna(a_cost)) or np.isclose(d_cost, a_cost, rtol=1e-9, atol=core.CONFIG["cost_tolerance"])),
        "dijkstra_path_found": d["path_found"], "astar_path_found": a["path_found"],
        "dijkstra_expanded_nodes": d_expanded, "astar_expanded_nodes": a["expanded_nodes"],
        "dijkstra_generated_nodes": d["generated_nodes"], "astar_generated_nodes": a["generated_nodes"],
        "time_saved_ms": d_time - a_time,
        "speedup_ratio": d_time / a_time if a_time > 0 else np.nan,
        "expanded_node_reduction": 1.0 - a["expanded_nodes"] / d_expanded if d_expanded else np.nan,
    }


def _statistics(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scene, algorithm), group in trials.groupby(["scene", "algorithm"], sort=False):
        valid = group[group["path_found"] == True]
        values = valid["planning_time_ms"].to_numpy(dtype=float)
        rows.append({
            "scene": scene, "algorithm": algorithm, "valid_trials": len(valid),
            "failed_trials": len(group) - len(valid), "success_rate": len(valid) / len(group),
            "mean_time_ms": values.mean() if len(values) else np.nan,
            "std_time_ms": values.std() if len(values) else np.nan,
            "min_time_ms": values.min() if len(values) else np.nan,
            "median_time_ms": np.median(values) if len(values) else np.nan,
            "p90_time_ms": np.percentile(values, 90) if len(values) else np.nan,
            "p95_time_ms": np.percentile(values, 95) if len(values) else np.nan,
            "p99_time_ms": np.percentile(values, 99) if len(values) else np.nan,
            "max_time_ms": values.max() if len(values) else np.nan,
            "mean_path_length_m": valid["path_cost_m"].mean() if len(valid) else np.nan,
            "mean_expanded_nodes": valid["expanded_nodes"].mean() if len(valid) else np.nan,
            "mean_generated_nodes": valid["generated_nodes"].mean() if len(valid) else np.nan,
        })
    return pd.DataFrame(rows)


def _plot_mean_time(stats, scene, output_path):
    data = stats[stats.scene == scene].set_index("algorithm").reindex(["dijkstra", "astar"])
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    bars = ax.bar(["Dijkstra", "A*"], data["mean_time_ms"], color=["#6b7280", "#2563eb"])
    for bar, med, p95 in zip(bars, data["median_time_ms"], data["p95_time_ms"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"median {med:.1f}\nP95 {p95:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean planning time (ms)"); ax.set_title(f"{scene.title()} Dijkstra vs A* mean time"); ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight"); plt.close(fig)


def _plot_paired_time(paired, scene, output_path):
    data = paired[paired.scene == scene]
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    x = data["dijkstra_time_ms"].to_numpy(); y = data["astar_time_ms"].to_numpy()
    ax.scatter(x, y, s=8, alpha=0.25, color="#2563eb"); limit = max(float(np.nanmax(x)), float(np.nanmax(y)))
    ax.plot([0, limit], [0, limit], "k--", linewidth=0.8); ax.set_xlabel("Dijkstra planning time (ms)"); ax.set_ylabel("A* planning time (ms)"); ax.set_title(f"{scene.title()} paired planning time"); ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight"); plt.close(fig)


def _plot_speedup(paired, scene, output_path):
    values = paired.loc[paired.scene == scene, "speedup_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True); ax.hist(values, bins=40, color="#16a34a", alpha=0.8); ax.axvline(1, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Dijkstra time / A* time"); ax.set_ylabel("Trial count"); ax.set_title(f"{scene.title()} A* speedup distribution"); ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight"); plt.close(fig)


def _plot_expanded(paired, scene, output_path):
    data = paired[paired.scene == scene]
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True); ax.scatter(data["dijkstra_expanded_nodes"], data["astar_expanded_nodes"], s=8, alpha=0.25, color="#dc2626")
    limit = max(float(data["dijkstra_expanded_nodes"].max()), float(data["astar_expanded_nodes"].max())); ax.plot([0, limit], [0, limit], "k--", linewidth=0.8)
    ax.set_xlabel("Dijkstra expanded nodes"); ax.set_ylabel("A* expanded nodes"); ax.set_title(f"{scene.title()} expanded-node comparison"); ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight"); plt.close(fig)


def _plot_overall_mean(stats, output_path):
    data = stats.groupby("algorithm", sort=False)["mean_time_ms"].mean().reindex(["dijkstra", "astar"])
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True); ax.bar(["Dijkstra", "A*"], data, color=["#6b7280", "#2563eb"]); ax.set_ylabel("Mean planning time (ms)"); ax.set_title("Overall Dijkstra vs A* mean time"); ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight"); plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "truck_uav_modules" / "data"
    voxel_dir = data_dir / "processed_3d_voxel_grids"
    output_dir = data_dir / "path_planning_3d_comparison"
    figure_dir = project_root / "truck_uav_modules" / "figure"
    output_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    all_trials, all_paired, all_od = [], [], []
    examples = {}
    offsets = core._neighbor_offsets()

    for scene in core.SCENES:
        occupancy = np.load(voxel_dir / f"{scene}_occupancy_3d_10m.npy", allow_pickle=False)
        labels, members = core.compute_3d_connected_components(occupancy)
        pairs = []
        for trial_id in range(1, NUM_TRIALS + 1):
            start, goal, _, _, component_id = core.sample_reachable_3d_start_goal(members, labels, rng)
            pairs.append((start, goal, component_id))
            all_od.append({"scene": scene, "trial_id": trial_id, "start_z": start[0], "start_row": start[1], "start_col": start[2], "goal_z": goal[0], "goal_row": goal[1], "goal_col": goal[2], "start_altitude_m": (start[0] + 0.5) * 10.0, "goal_altitude_m": (goal[0] + 0.5) * 10.0, "horizontal_distance_m": math.hypot(start[1] - goal[1], start[2] - goal[2]) * 10.0, "vertical_distance_m": abs(start[0] - goal[0]) * 10.0, "straight_3d_distance_m": math.dist(start, goal) * 10.0, "component_id": component_id, "random_seed": RANDOM_SEED})
        workspace_d = core.AStarWorkspace.create(occupancy.shape); workspace_a = core.AStarWorkspace.create(occupancy.shape)
        for start, goal, _ in pairs[:WARMUP_TRIALS]:
            core.plan_3d_path(occupancy, start, goal, "dijkstra", 26, True, workspace_d, offsets)
            core.plan_3d_path(occupancy, start, goal, "astar", 26, True, workspace_a, offsets)
        for trial_id, (start, goal, _) in enumerate(pairs, start=1):
            order = ("dijkstra", "astar") if trial_id % 2 else ("astar", "dijkstra")
            results, trial_records = {}, {}
            for algorithm in order:
                workspace = workspace_d if algorithm == "dijkstra" else workspace_a
                result, elapsed = _run_one(occupancy, start, goal, algorithm, workspace, offsets)
                results[algorithm] = (result, elapsed)
                trial_records[algorithm] = _record_base(scene, trial_id, algorithm, start, goal, result, elapsed, occupancy)
            if trial_id <= SMOKE_TEST_TRIALS or trial_id <= 20:
                _validate_pair(occupancy, start, goal, results)
            all_trials.extend(trial_records.values()); all_paired.append(_paired_record(scene, trial_id, trial_records))
            if trial_id <= 5: examples.setdefault(scene, (occupancy, start, goal, results))
            if trial_id % 100 == 0: LOGGER.info("[%s] completed paired trials %d/%d", scene.title(), trial_id, NUM_TRIALS)

    trials = pd.DataFrame(all_trials); paired = pd.DataFrame(all_paired); od_pairs = pd.DataFrame(all_od); stats = _statistics(trials)
    trials.to_csv(output_dir / "dijkstra_vs_astar_3d_trials.csv", index=False); paired.to_csv(output_dir / "dijkstra_vs_astar_3d_paired_results.csv", index=False); od_pairs.to_csv(output_dir / "path_planner_comparison_od_pairs.csv", index=False); stats.to_csv(output_dir / "dijkstra_vs_astar_3d_summary.csv", index=False)
    for scene in core.SCENES:
        _plot_mean_time(stats, scene, figure_dir / f"{scene}_dijkstra_vs_astar_mean_time.png"); _plot_paired_time(paired, scene, figure_dir / f"{scene}_dijkstra_vs_astar_paired_time.png"); _plot_speedup(paired, scene, figure_dir / f"{scene}_astar_speedup_distribution.png"); _plot_expanded(paired, scene, figure_dir / f"{scene}_expanded_nodes_comparison.png")
    _plot_overall_mean(stats, figure_dir / "overall_dijkstra_vs_astar_mean_time.png")
    valid_paired = paired[paired["cost_equal"] == True]
    d = valid_paired["dijkstra_time_ms"]; a = valid_paired["astar_time_ms"]
    report = {"benchmark": "paired 3-D Dijkstra versus standard A*", "voxel_size_m": [10.0, 10.0, 10.0], "trials_per_scene": NUM_TRIALS, "total_paired_trials": len(paired), "cost_agreement_rate": float(paired["cost_equal"].mean()), "astar_faster_ratio": float((a < d).mean()), "dijkstra_faster_ratio": float((d < a).mean()), "mean_speedup": float((d / a).mean()), "median_speedup": float((d / a).median()), "mean_time_saved_ms": float((d - a).mean()), "mean_expanded_node_reduction": float(paired["expanded_node_reduction"].mean()), "statistics": stats.to_dict(orient="records"), "config": {"neighbor_mode": 26, "prevent_corner_cutting": True, "heuristic": "euclidean", "heuristic_weight": 1.0, "random_seed": RANDOM_SEED, "warmup_trials": WARMUP_TRIALS, "smoke_test_trials": SMOKE_TEST_TRIALS, "time_equal_tolerance_ms": TIME_EQUAL_TOLERANCE_MS, "min_3d_distance_voxels": core.CONFIG["min_3d_distance_voxels"], "max_3d_distance_voxels": core.CONFIG["max_3d_distance_voxels"], "min_horizontal_distance_voxels": core.CONFIG["min_horizontal_distance_voxels"], "max_horizontal_distance_voxels": core.CONFIG["max_horizontal_distance_voxels"]}, "system": {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()}, "outputs": {"output_dir": str(output_dir), "figure_dir": str(figure_dir)}}
    (output_dir / "dijkstra_vs_astar_3d_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("=" * 72); print("3D DIJKSTRA VS A* BENCHMARK"); print("Voxel size: 10 m x 10 m x 10 m"); print("Neighbor mode: 26; prevent corner cutting: True"); print(f"Trials per scene: {NUM_TRIALS}"); print(f"Overall paired trials: {len(paired)}"); print(f"Path-cost agreement: {report['cost_agreement_rate']:.2%}"); print(f"Mean speedup (Dijkstra/A*): {report['mean_speedup']:.4f}x"); print(f"A* faster ratio: {report['astar_faster_ratio']:.2%}"); print(f"Mean expanded-node reduction: {report['mean_expanded_node_reduction']:.2%}"); print(stats.to_string(index=False)); print("=" * 72)


if __name__ == "__main__":
    main()
