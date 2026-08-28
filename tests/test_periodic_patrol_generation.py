from __future__ import annotations

from collections import Counter

import pytest

from env.road_network import RoadNetwork
from env.xinjiekou_adapter import build_direct_edge_task_stream, validate_periodic_patrol_schedule
from utils.config import load_config, project_root


def _generate(scene: str):
    root = project_root()
    config_name = "default_training.yaml" if scene == "xinjiekou" else "baijiahu_training.yaml"
    cfg = load_config(root / "configs" / config_name)
    network = RoadNetwork(cfg)
    tasks = build_direct_edge_task_stream(
        network,
        cfg["edge_tasks"]["registry_path"],
        cfg["edge_tasks"],
        release_horizon_s=cfg["simulation"]["task_release_horizon_s"],
        seed=42,
    )
    periodic = [task for task in tasks if task.metadata.get("periodic_task")]
    return cfg, periodic


@pytest.mark.parametrize("scene", ["xinjiekou", "baijiahu"])
def test_periodic_patrol_has_fixed_groups_and_expected_schedule(scene: str) -> None:
    cfg, tasks = _generate(scene)
    periodic_cfg = cfg["edge_tasks"]["periodic_patrol"]
    summary = validate_periodic_patrol_schedule(
        tasks,
        selected_segment_count=periodic_cfg["selected_segment_count"],
        batch_size=periodic_cfg["batch_size"],
        revisit_interval_s=periodic_cfg["revisit_interval_s"],
        cycle_count=periodic_cfg["cycle_count"],
        deadline_s=periodic_cfg["deadline_s"],
    )

    assert summary["unique_physical_road_count"] == 10
    assert summary["group_count"] == 5
    assert summary["group_sizes"] == [2, 2, 2, 2, 2]
    assert summary["phase_interval_s"] == pytest.approx(360.0)
    assert summary["task_count"] == 60
    assert summary["last_release_time_s"] == pytest.approx(10440.0)
    assert summary["last_deadline_time_s"] == pytest.approx(12240.0)
    assert set(Counter(task.metadata["inspection_segment_id"] for task in tasks).values()) == {6}
    assert all(task.deadline == pytest.approx(task.release_time + 1800.0) for task in tasks)


@pytest.mark.parametrize("scene", ["xinjiekou", "baijiahu"])
def test_periodic_patrol_is_reproducible_and_uses_scene_registry(scene: str) -> None:
    cfg, first = _generate(scene)
    _, second = _generate(scene)
    signature = lambda tasks: [
        (
            task.id,
            task.metadata["inspection_segment_id"],
            task.metadata["physical_road_key"],
            task.metadata["periodic_group_number"],
            task.metadata["cycle_index"],
            task.release_time,
            task.deadline,
        )
        for task in tasks
    ]
    assert signature(first) == signature(second)
    assert all(task.metadata["registry_path"] == cfg["edge_tasks"]["registry_path"] for task in first)
    assert all(str(task.metadata["mapping_version"]).startswith(scene) for task in first)
    assert cfg["simulation"]["max_drain_time_s"] >= 1800
