from __future__ import annotations

import math

import numpy as np
import pytest

from env.road_network import RoadNetwork
from env.xinjiekou_adapter import (
    build_direct_edge_task_stream,
    build_incident_sampling_pool,
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
            task.metadata["scenario_id"],
            task.release_time,
            task.deadline,
            task.metadata["inspection_segment_id"],
            task.metadata["affected_direction"],
            task.metadata["road_length_m"],
            task.service_time,
            task.metadata["arrival_rate_per_hour"],
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
            return [task for task in all_tasks if task.type == "incident_confirmation"]

        registry = load_edge_task_registry(network, cfg["edge_tasks"]["registry_path"])
        incident_pool = registry[
            registry["eligible_incident_confirmation"].astype(str).str.lower().isin({"true", "1"})
        ]
        generated[scene] = {
            "cfg": cfg,
            "network": network,
            "registry": registry,
            "sampling_pool": build_incident_sampling_pool(network, incident_pool),
            "first": generate_once(),
            "second": generate_once(),
        }
    return generated


@pytest.mark.parametrize("scene", SCENE_CONFIGS)
def test_incident_confirmation_is_reproducible_and_uses_scene_registry(generated_scenes, scene: str) -> None:
    data = generated_scenes[scene]
    first = data["first"]
    assert _signature(first) == _signature(data["second"])

    registry_ids = set(data["registry"]["edge_id"].astype(str))
    assert all(task.metadata["inspection_segment_id"] in registry_ids for task in first)
    assert all(task.metadata["registry_path"] == data["cfg"]["edge_tasks"]["registry_path"] for task in first)
    assert all(str(task.metadata["mapping_version"]).startswith(scene) for task in first)


def test_same_seed_has_identical_cross_scene_arrivals(generated_scenes) -> None:
    xinjiekou = generated_scenes["xinjiekou"]["first"]
    baijiahu = generated_scenes["baijiahu"]["first"]
    assert [task.release_time for task in xinjiekou] == [task.release_time for task in baijiahu]
    assert len(xinjiekou) == len(baijiahu)

    cfg = generated_scenes["xinjiekou"]["cfg"]
    spec = cfg["edge_tasks"]["dynamic_tasks"]["incident_confirmation"]
    expected_count = spec["arrival_rate_per_hour"] * cfg["simulation"]["task_release_horizon_s"] / 3600.0
    assert expected_count == pytest.approx(6.0)
    assert all(0.0 <= task.release_time < 10800.0 for task in xinjiekou)


@pytest.mark.parametrize("scene", SCENE_CONFIGS)
def test_incident_attributes_and_service_time_formula(generated_scenes, scene: str) -> None:
    cfg = generated_scenes[scene]["cfg"]
    tasks = generated_scenes[scene]["first"]
    required_metadata = {
        "task_id",
        "scenario_id",
        "release_time_s",
        "deadline_time_s",
        "inspection_segment_id",
        "affected_direction",
        "road_length_m",
        "service_time_s",
        "arrival_rate_per_hour",
        "generation_seed",
    }
    for task in tasks:
        assert required_metadata <= set(task.metadata)
        assert task.metadata["scenario_id"] == cfg["experiment"]["scenario_name"]
        assert task.priority == pytest.approx(10.0)
        assert task.deadline == pytest.approx(task.release_time + 420.0)
        assert task.metadata["release_time_s"] == pytest.approx(task.release_time)
        assert task.metadata["deadline_time_s"] == pytest.approx(task.deadline)
        expected_service = float(np.clip(task.metadata["road_length_m"] / 3.0 + 45.0, 60.0, 150.0))
        assert task.service_time == pytest.approx(expected_service)
        assert task.metadata["service_time_s"] == pytest.approx(expected_service)
        assert task.metadata["arrival_rate_per_hour"] == pytest.approx(2.0)
        assert task.metadata["generation_seed"] == 42
        assert task.metadata["affected_direction"] in {"u_to_v", "v_to_u"}


@pytest.mark.parametrize("scene", SCENE_CONFIGS)
def test_incident_sampling_pool_is_unique_and_weighted_by_all_factors(generated_scenes, scene: str) -> None:
    pool = generated_scenes[scene]["sampling_pool"]
    probabilities = pool["incident_sampling_probability"].to_numpy(float)
    component_product = (
        pool["road_class_weight"].to_numpy(float)
        * pool["length_weight"].to_numpy(float)
        * pool["centrality_weight"].to_numpy(float)
    )

    assert pool["edge_id"].is_unique
    assert pool["physical_road_key"].is_unique
    assert np.isfinite(probabilities).all() and np.all(probabilities > 0.0)
    assert probabilities.sum() == pytest.approx(1.0)
    assert not np.allclose(probabilities, probabilities[0])
    assert pool["road_class_weight"].nunique() > 1
    assert pool["length_weight"].nunique() > 1
    assert pool["centrality_weight"].nunique() > 1
    assert np.allclose(pool["incident_sampling_weight"].to_numpy(float), component_product)
    assert math.isfinite(float(pool["edge_betweenness_centrality"].max()))
