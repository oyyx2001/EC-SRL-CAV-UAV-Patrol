"""Release contracts; uses real assets/weights, never starts training."""
import importlib.util
import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from dynamic_patrol_marl.rl.checkpoint_io import CASES, MAIN_METRICS, ROOT, config_hash, load_policy, state_dict_hash
from dynamic_patrol_marl.rl.dqn_networks import masked_q_values
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.rl.evaluate_checkpoint import metric_values, summarize, task_stream_hash
from dynamic_patrol_marl.utils.config import load_config


@pytest.mark.parametrize("case,legacy,nodes,links", [
    (CASES[0], "default_training.yaml", 333, 601),
    (CASES[1], "baijiahu_training.yaml", 129, 247),
])
def test_case_assets_and_seeded_tasks_unchanged(case, legacy, nodes, links):
    config = load_config(ROOT / "configs" / f"{case}.yaml")
    old_config = load_config(ROOT / "configs" / legacy)
    new = DynamicPatrolEnv(config, scenario=case)
    old = DynamicPatrolEnv(old_config, scenario=old_config["rl"]["eval_scenario"])
    obs, info = new.reset(seed=1000)
    old_obs, old_info = old.reset(seed=1000)
    assert new.sim.network.source_edge_count == links
    assert len(new.sim.network.nodes) == nodes
    assert task_stream_hash(new.sim.tasks) == task_stream_hash(old.sim.tasks)
    np.testing.assert_array_equal(obs, old_obs)
    np.testing.assert_array_equal(info["action_mask"], old_info["action_mask"])
    assert (new.obs_dim, new.max_actions, new.plan_feature_dim) == (1025, 25, 38)
    assert len(info["action_mask"]) == 25 and info["action_mask"].any()
    assert config["recovery"]["rule"] == "dynamic_insertion"
    for path in [config["network"]["graphml_file"], config["network"]["buildings_file"],
                 config["edge_tasks"]["registry_path"], config["uav_path_planning"]["voxel_map_dir"]]:
        assert (ROOT / path).exists()


@pytest.mark.parametrize("case", CASES)
def test_frozen_checkpoint_load_and_deterministic_forward(case):
    torch.set_num_threads(1)
    model, config, checkpoint = load_policy(case)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert checkpoint["seed"] == 42 and checkpoint["timesteps"] == 300000
    assert model.global_dim == 113  # Preserve original safe-slot absorption.
    env = DynamicPatrolEnv(config, scenario=case)
    obs, info = env.reset(seed=1000)
    x = torch.as_tensor(obs).unsqueeze(0)
    mask = torch.as_tensor(info["action_mask"]).unsqueeze(0)
    with torch.inference_mode():
        first = model(x, mask)
        model.reset_noise()
        second = model(x, mask)
    assert torch.equal(first, second)  # NoisyNet must be disabled at inference.
    action = masked_q_values(first, mask).argmax().item()
    assert info["action_mask"][action]


def test_only_main_algorithm_is_exposed():
    from dynamic_patrol_marl.rl.dqn_networks import make_q_network
    with pytest.raises(ValueError):
        make_q_network("dqn", 1025, 25, 38)
    with pytest.raises(ValueError):
        make_q_network("rainbow_dqn", 1025, 25, 38, architecture="state_only")


def test_metric_units_and_missing_values_are_not_modified():
    values = metric_values(dict(completion_rate_final=0.9, avg_response_time_s=10,
                                high_priority_response_time_s=float("nan"),
                                uav_energy_per_completed_task=20,
                                cav_travel_distance_m=12000, drone_wait_time_s=300))
    assert values == dict(CR_pct=90, MRT_s=10, HPRT_s=None, ECT=20, CTD_km=12, CURW_s=300)


def test_summary_sample_sd_and_metric_specific_n():
    rows = []
    for seed, cr, hprt in [(1000, 80, None), (1001, 90, 200), (1002, 100, 300)]:
        rows.append(dict(case=CASES[0], eval_seed=seed, CR_pct=cr, MRT_s=10,
                         HPRT_s=hprt, ECT=20, CTD_km=12, CURW_s=300))
    _, summary = summarize(rows)
    cr = summary[summary.metric == "CR_pct"].iloc[0]
    hprt = summary[summary.metric == "HPRT_s"].iloc[0]
    assert cr["mean"] == 90 and cr["std"] == 10
    assert hprt["valid_n"] == 2 and hprt["missing_n"] == 1
    assert not summary.matches_reference_2dp.any()  # A pilot cannot certify reproduction.


@pytest.mark.parametrize("case", CASES)
def test_original_and_released_networks_are_bitwise_identical(case):
    # Optional local provenance test; a public clone does not need historical runs.
    from dynamic_patrol_marl.rl.prepare_pretrained_release import SOURCES
    source = ROOT / SOURCES[case][0]
    if not source.exists():
        pytest.skip("Historical source artifact is intentionally not required by the release")
    original = torch.load(source, map_location="cpu", weights_only=False)
    _, _, published = load_policy(case)
    assert state_dict_hash(original["model_state_dict"]) == published["state_dict_sha256"]
    for key, tensor in original["model_state_dict"].items():
        assert torch.equal(tensor, published["model_state_dict"][key])
    legacy_module = ROOT / ".codex_work/release_audit/original_dqn_networks.py"
    if legacy_module.exists():
        spec = importlib.util.spec_from_file_location("original_network", legacy_module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        old_model = module.make_q_network("rainbow_dqn", 1025, 25, 38)
        old_model.load_state_dict(original["model_state_dict"], strict=True)
        old_model.eval()
        new_model, config, _ = load_policy(case)
        env = DynamicPatrolEnv(config, scenario=case)
        obs, info = env.reset(seed=1000)
        x, mask = torch.as_tensor(obs).unsqueeze(0), torch.as_tensor(info["action_mask"]).unsqueeze(0)
        with torch.inference_mode():
            assert torch.equal(old_model(x, mask), new_model(x, mask))


@pytest.mark.parametrize("case", CASES)
def test_portable_metadata_and_public_configuration_match(case):
    _, config, checkpoint = load_policy(case)
    public_config = load_config(ROOT / "configs" / f"{case}.yaml")
    assert config_hash(config) == config_hash(public_config)
    metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    serialized = json.dumps(metadata)
    assert not re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/Users/|/home/", serialized)
    assert "optimizer_state_dict" not in checkpoint


def test_published_evaluation_tables_are_complete_and_statistics_verified():
    raw = pd.read_csv(ROOT / "reproduction/per_seed_metrics.csv")
    summary = pd.read_csv(ROOT / "reproduction/summary_mean_std.csv")
    assert len(raw) == 40 and not raw.duplicated(["case", "eval_seed"]).any()
    assert not raw[list(MAIN_METRICS)].isna().any().any()
    assert set(raw.case) == set(CASES)
    assert raw.illegal_actions.eq(0).all()
    assert raw.training_seed.eq(42).all() and raw.training_steps.eq(300000).all()
    for case in CASES:
        group = raw.loc[raw.case == case]
        assert sorted(group.eval_seed) == list(range(1000, 1020))
        np.testing.assert_allclose(group.CR_pct, 100 * group.completed_task_count / group.released_task_count)
        for metric in MAIN_METRICS:
            row = summary.loc[(summary.case == case) & (summary.metric == metric)].iloc[0]
            assert row.valid_n == 20 and row.missing_n == 0
            assert row["mean"] == pytest.approx(group[metric].mean(), abs=1e-10)
            assert row["std"] == pytest.approx(group[metric].std(ddof=1), abs=1e-10)
            assert row.matches_reference_2dp and row.matches_reference_std_2dp


@pytest.mark.parametrize("statement", [
    "import truck_uav_modules.uav_path_planner",
    "import dynamic_patrol_marl.truck_uav_modules.uav_path_planner",
    "import sys; sys.path.insert(0, 'truck_uav_modules'); import uav_path_planner",
])
def test_planning_compatibility_imports_in_fresh_interpreter(statement):
    result = subprocess.run([sys.executable, "-c", statement], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
