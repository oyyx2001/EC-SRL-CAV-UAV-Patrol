from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from env.road_network import RoadNetwork
from utils.config import load_config, project_root


def test_baijiahu_training_assets_match_current_model_contract() -> None:
    root = project_root()
    cfg = load_config(root / "configs" / "baijiahu_training.yaml")
    network = RoadNetwork(cfg)

    assert len(network.nodes) == 129
    assert network.source_edge_count == 247
    assert cfg["reward"]["mode"] == "stage5_coordinated"
    assert cfg["rl"]["discount_mode"] == "time_aware"
    assert cfg["rl"]["hyperparams"]["candidate_dim"] == 38

    registry = pd.read_csv(root / cfg["edge_tasks"]["registry_path"])
    assert len(registry) == 247
    assert registry["edge_id"].nunique() == 247
    assert registry["eligible_routine"].astype(bool).all()

    voxel_dir = root / cfg["uav_path_planning"]["voxel_map_dir"]
    occupancy = np.load(voxel_dir / "occupancy.npy", allow_pickle=False)
    metadata = json.loads((voxel_dir / "metadata.json").read_text(encoding="utf-8"))
    assert occupancy.shape == (6, 260, 260)
    assert occupancy.dtype == np.uint8
    assert tuple(metadata[key] for key in ("voxel_size_x_m", "voxel_size_y_m", "voxel_size_z_m")) == (
        10.0,
        10.0,
        20.0,
    )
    assert metadata["road_node_free_operational_layers"] == [1, 2, 3, 4, 5]
