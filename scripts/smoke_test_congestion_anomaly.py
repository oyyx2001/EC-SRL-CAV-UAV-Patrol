"""Generate seed-42 congestion-anomaly tasks without starting training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.road_network import RoadNetwork
from env.xinjiekou_adapter import build_direct_edge_task_stream, load_edge_task_registry
from utils.config import load_config, project_root


SCENE_CONFIGS = {
    "xinjiekou": "default_training.yaml",
    "baijiahu": "baijiahu_training.yaml",
}


def task_signature(tasks):
    return [
        (
            task.metadata["task_id"],
            task.metadata["scenario_id"],
            task.release_time,
            task.deadline,
            task.priority,
            task.metadata["inspection_segment_id"],
            task.metadata["affected_direction"],
            task.metadata["road_length_m"],
            task.service_time,
            task.metadata["spatial_weight"],
            task.metadata["generation_seed"],
        )
        for task in tasks
    ]


def generate_scene(scene: str, seed: int, output_dir: Path) -> tuple[dict, list[float]]:
    root = project_root()
    cfg = load_config(root / "configs" / SCENE_CONFIGS[scene])
    network = RoadNetwork(cfg)

    def generate_once():
        generated = build_direct_edge_task_stream(
            network,
            cfg["edge_tasks"]["registry_path"],
            cfg["edge_tasks"],
            release_horizon_s=cfg["simulation"]["task_release_horizon_s"],
            seed=seed,
        )
        return [task for task in generated if task.type == "congestion_anomaly_scan"]

    tasks = generate_once()
    if task_signature(tasks) != task_signature(generate_once()):
        raise AssertionError(f"{scene} congestion anomaly tasks are not reproducible for seed {seed}")
    if len(tasks) != 12:
        raise AssertionError(f"{scene} expected 12 congestion anomaly tasks, got {len(tasks)}")

    segment_counts = [
        sum(start <= task.release_time < end for task in tasks)
        for start, end in ((0.0, 3600.0), (3600.0, 7200.0), (7200.0, 10800.0))
    ]
    if segment_counts != [2, 8, 2]:
        raise AssertionError(f"{scene} congestion segment counts are {segment_counts}, expected [2, 8, 2]")
    registry_ids = set(
        load_edge_task_registry(network, cfg["edge_tasks"]["registry_path"])["edge_id"].astype(str)
    )
    if any(task.metadata["inspection_segment_id"] not in registry_ids for task in tasks):
        raise AssertionError(f"{scene} congestion tasks reference another scene's registry")

    rows = []
    for task in tasks:
        metadata = task.metadata
        rows.append({
            "task_id": task.id,
            "task_type": task.type,
            "scenario_id": metadata["scenario_id"],
            "release_time_s": task.release_time,
            "deadline_time_s": task.deadline,
            "priority": task.priority,
            "inspection_segment_id": metadata["inspection_segment_id"],
            "affected_direction": metadata["affected_direction"],
            "road_length_m": metadata["road_length_m"],
            "service_time_s": task.service_time,
            "spatial_weight": metadata["spatial_weight"],
            "generation_seed": metadata["generation_seed"],
            "road_classes": metadata["road_classes"],
            "road_class_weight": metadata["road_class_weight"],
            "length_weight": metadata["length_weight"],
            "edge_betweenness_centrality": metadata["edge_betweenness_centrality"],
            "centrality_weight": metadata["centrality_weight"],
            "hotspot_score": metadata["hotspot_score"],
            "active_road_exclusion_fallback": metadata["active_road_exclusion_fallback"],
            "registry_path": metadata["registry_path"],
        })
    frame = pd.DataFrame(rows).sort_values(["release_time_s", "task_id"], kind="mergesort")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{scene}_congestion_anomaly_scan_tasks_seed{seed}.csv"
    frame.to_csv(csv_path, index=False, float_format="%.9f")

    release_times = [float(task.release_time) for task in tasks]
    service_times = [float(task.service_time) for task in tasks]
    return {
        "scene": scene,
        "scenario_id": cfg["experiment"]["scenario_name"],
        "seed": seed,
        "process": "fixed_count_piecewise_random",
        "actual_task_count": len(tasks),
        "time_segment_counts": segment_counts,
        "release_times_s": release_times,
        "service_time_min_s": min(service_times),
        "service_time_max_s": max(service_times),
        "service_time_varies": len({round(value, 9) for value in service_times}) > 1,
        "affected_directions": sorted({task.metadata["affected_direction"] for task in tasks}),
        "active_road_exclusion_fallback_count": sum(
            bool(task.metadata["active_road_exclusion_fallback"]) for task in tasks
        ),
        "registry_path": cfg["edge_tasks"]["registry_path"],
        "task_csv": str(csv_path),
        "reproducible": True,
    }, release_times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or project_root() / "results" / "task_generation_smoke" / f"seed{args.seed}"

    summaries = []
    arrivals_by_scene = {}
    for scene in SCENE_CONFIGS:
        summary, releases = generate_scene(scene, args.seed, output_dir)
        summaries.append(summary)
        arrivals_by_scene[scene] = releases
    if len({tuple(releases) for releases in arrivals_by_scene.values()}) != 1:
        raise AssertionError("Congestion release times differ across scenes for the same seed")

    summary_path = output_dir / f"congestion_anomaly_smoke_summary_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "scenes": summaries}, indent=2))


if __name__ == "__main__":
    main()
