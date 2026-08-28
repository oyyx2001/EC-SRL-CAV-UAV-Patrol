"""Generate and audit seed-42 periodic patrol schedules without running training."""
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
from env.xinjiekou_adapter import build_direct_edge_task_stream, validate_periodic_patrol_schedule
from utils.config import load_config, project_root


SCENE_CONFIGS = {
    "xinjiekou": "default_training.yaml",
    "baijiahu": "baijiahu_training.yaml",
}


def task_signature(tasks):
    return [
        (
            task.metadata["inspection_segment_id"],
            task.metadata["periodic_group_number"],
            task.metadata["cycle_index"],
            task.release_time,
            task.deadline,
        )
        for task in tasks
    ]


def generate_scene(scene: str, seed: int, output_dir: Path) -> dict:
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
        return [task for task in generated if task.metadata.get("periodic_task")]

    tasks = generate_once()
    if task_signature(tasks) != task_signature(generate_once()):
        raise AssertionError(f"{scene} periodic patrol is not reproducible for seed {seed}")

    periodic_cfg = cfg["edge_tasks"]["periodic_patrol"]
    summary = validate_periodic_patrol_schedule(
        tasks,
        selected_segment_count=periodic_cfg["selected_segment_count"],
        batch_size=periodic_cfg["batch_size"],
        revisit_interval_s=periodic_cfg["revisit_interval_s"],
        cycle_count=periodic_cfg["cycle_count"],
        deadline_s=periodic_cfg["deadline_s"],
    )
    if not all(str(task.metadata["mapping_version"]).startswith(scene) for task in tasks):
        raise AssertionError(f"{scene} tasks did not use the {scene} registry")

    task_rows = []
    for task in tasks:
        metadata = task.metadata
        task_rows.append({
            "scene": scene,
            "seed": seed,
            "task_id": task.id,
            "task_instance_id": metadata["task_instance_id"],
            "inspection_segment_id": metadata["inspection_segment_id"],
            "physical_road_key": metadata["physical_road_key"],
            "group_number": metadata["periodic_group_number"],
            "cycle_number": int(metadata["cycle_index"]) + 1,
            "release_time_s": task.release_time,
            "deadline_time_s": task.deadline,
            "priority": task.priority,
            "service_time_s": task.service_time,
            "registry_path": metadata["registry_path"],
            "mapping_version": metadata["mapping_version"],
        })
    task_frame = pd.DataFrame(task_rows).sort_values(["release_time_s", "task_id"], kind="mergesort")

    group_rows = []
    for group_number, group in task_frame.groupby("group_number", sort=True):
        first_cycle = group[group["cycle_number"] == 1].sort_values("task_id")
        release_times = sorted(group["release_time_s"].unique().tolist())
        group_rows.append({
            "scene": scene,
            "seed": seed,
            "group_number": int(group_number),
            "segment_count": len(first_cycle),
            "inspection_segment_ids": ";".join(first_cycle["inspection_segment_id"].tolist()),
            "first_release_time_s": release_times[0],
            "release_times_s": ";".join(f"{value:g}" for value in release_times),
            "revisit_interval_s": periodic_cfg["revisit_interval_s"],
            "registry_path": cfg["edge_tasks"]["registry_path"],
        })
    group_frame = pd.DataFrame(group_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    task_path = output_dir / f"{scene}_periodic_patrol_tasks_seed{seed}.csv"
    group_path = output_dir / f"{scene}_periodic_patrol_groups_seed{seed}.csv"
    task_frame.to_csv(task_path, index=False)
    group_frame.to_csv(group_path, index=False)
    return {
        "scene": scene,
        "seed": seed,
        "registry_path": cfg["edge_tasks"]["registry_path"],
        "task_csv": str(task_path),
        "group_csv": str(group_path),
        "reproducible": True,
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "results" / "task_generation_smoke" / "seed42",
    )
    args = parser.parse_args()
    summaries = [generate_scene(scene, args.seed, args.output_dir) for scene in SCENE_CONFIGS]
    summary_path = args.output_dir / f"periodic_patrol_smoke_summary_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "scenes": summaries}, indent=2))


if __name__ == "__main__":
    main()
