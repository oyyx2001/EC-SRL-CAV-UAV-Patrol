from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from env.road_network import RoadNetwork
from env.xinjiekou_adapter import (
    build_congestion_sampling_pool,
    build_direct_edge_task_stream,
    load_edge_task_registry,
)
from utils.config import load_config, project_root


SCENE_CONFIGS = {
    "xinjiekou": "default_training.yaml",
    "baijiahu": "baijiahu_training.yaml",
}


def _signature(tasks):
    return [
        (
            task.id,
            task.metadata["task_id"],
            task.metadata["task_type"],
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


@pytest.fixture(scope="module")
def generated_scenes():
    root = project_root()
    generated = {}
    for scene, config_name in SCENE_CONFIGS.items():
        cfg = load_config(root / "configs" / config_name)
        network = RoadNetwork(cfg)

        def generate_once():
            all_tasks = build_direct_edge_task_stream(
                network,
                cfg["edge_tasks"]["registry_path"],
                cfg["edge_tasks"],
                release_horizon_s=cfg["simulation"]["task_release_horizon_s"],
                seed=42,
            )
            return [task for task in all_tasks if task.type == "congestion_anomaly_scan"]

        registry = load_edge_task_registry(network, cfg["edge_tasks"]["registry_path"])
        eligible = registry[
            registry["eligible_congestion_anomaly_scan"].astype(str).str.lower().isin({"true", "1"})
        ]
        generated[scene] = {
            "cfg": cfg,
            "network": network,
            "registry": registry,
            "sampling_pool": build_congestion_sampling_pool(network, eligible),
            "first": generate_once(),
            "second": generate_once(),
        }
    return generated


@pytest.mark.parametrize("scene", SCENE_CONFIGS)
def test_congestion_has_fixed_piecewise_count_and_is_reproducible(generated_scenes, scene: str) -> None:
    data = generated_scenes[scene]
    tasks = data["first"]
    assert len(tasks) == 12
    assert _signature(tasks) == _signature(data["second"])
    assert [
        sum(start <= task.release_time < end for task in tasks)
        for start, end in ((0.0, 3600.0), (3600.0, 7200.0), (7200.0, 10800.0))
    ] == [2, 8, 2]
    assert [task.release_time for task in tasks] == sorted(task.release_time for task in tasks)


def test_cross_scene_release_times_are_identical(generated_scenes) -> None:
    xinjiekou = generated_scenes["xinjiekou"]["first"]
    baijiahu = generated_scenes["baijiahu"]["first"]
    assert [task.release_time for task in xinjiekou] == [task.release_time for task in baijiahu]


@pytest.mark.parametrize("scene", SCENE_CONFIGS)
def test_congestion_uses_scene_registry_and_required_attributes(generated_scenes, scene: str) -> None:
    data = generated_scenes[scene]
    tasks = data["first"]
    registry_ids = set(data["registry"]["edge_id"].astype(str))
    required_metadata = {
        "task_id",
        "task_type",
        "scenario_id",
        "release_time_s",
        "deadline_time_s",
        "priority",
        "inspection_segment_id",
        "affected_direction",
        "road_length_m",
        "service_time_s",
        "spatial_weight",
        "generation_seed",
    }
    for task in tasks:
        assert required_metadata <= set(task.metadata)
        assert task.metadata["inspection_segment_id"] in registry_ids
        assert task.metadata["registry_path"] == data["cfg"]["edge_tasks"]["registry_path"]
        assert str(task.metadata["mapping_version"]).startswith(scene)
        assert task.metadata["scenario_id"] == data["cfg"]["experiment"]["scenario_name"]
        assert task.priority == pytest.approx(6.0)
        assert task.deadline == pytest.approx(task.release_time + 600.0)
        assert task.metadata["affected_direction"] == "both"
        expected_service = float(np.clip(task.metadata["road_length_m"] / 5.0 + 30.0, 45.0, 120.0))
        assert task.service_time == pytest.approx(expected_service)
        assert task.metadata["service_time_s"] == pytest.approx(expected_service)
        assert task.metadata["generation_seed"] == 42
    assert len({round(task.service_time, 9) for task in tasks}) > 1


@pytest.mark.parametrize("scene", SCENE_CONFIGS)
def test_active_congestion_roads_are_excluded_and_weights_use_all_factors(generated_scenes, scene: str) -> None:
    tasks = generated_scenes[scene]["first"]
    releases_by_segment = {}
    for task in tasks:
        releases_by_segment.setdefault(task.metadata["inspection_segment_id"], []).append(task.release_time)
    for releases in releases_by_segment.values():
        assert all(right - left >= 600.0 for left, right in zip(releases, releases[1:]))

    pool = generated_scenes[scene]["sampling_pool"]
    raw_product = (
        pool["road_class_weight"].to_numpy(float)
        * pool["length_weight"].to_numpy(float)
        * pool["centrality_weight"].to_numpy(float)
        * pool["hotspot_score"].to_numpy(float)
    )
    weights = pool["spatial_weight"].to_numpy(float)
    assert pool["physical_road_key"].is_unique
    assert np.isfinite(weights).all() and np.all(weights > 0.0)
    assert weights.sum() == pytest.approx(1.0)
    assert not np.allclose(weights, weights[0])
    assert np.allclose(pool["congestion_spatial_weight_raw"].to_numpy(float), raw_product)
    assert pool["road_class_weight"].nunique() > 1
    assert pool["length_weight"].nunique() > 1
    assert pool["centrality_weight"].nunique() > 1
    assert pool["hotspot_score"].nunique() > 1
    assert Counter(task.metadata["active_road_exclusion_fallback"] for task in tasks)[True] == 0


def test_old_default_congestion_schedule_is_disabled(generated_scenes) -> None:
    dynamic = generated_scenes["xinjiekou"]["cfg"]["edge_tasks"]["dynamic_tasks"]
    assert "congestion" not in dynamic
    spec = dynamic["congestion_anomaly_scan"]
    assert spec["process"] == "fixed_count_piecewise_random"
    assert "rate_per_hour" not in spec
    assert "additional_release_times_s" not in spec
    assert "service_duration_s" not in spec
