"""Portable, strict loading of the original-weight CS-RDQN release artifacts."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from dynamic_patrol_marl.rl.dqn_networks import make_q_network
from dynamic_patrol_marl.utils.config import project_root

ROOT = project_root()
CASES = ("case1_xinjiekou", "case2_baijiahu")
MAIN_METRICS = ("CR_pct", "MRT_s", "HPRT_s", "ECT", "CTD_km", "CURW_s")
TEST_SEEDS = tuple(range(1000, 1020))
REFERENCE_MEANS = {
    CASES[0]: dict(zip(MAIN_METRICS, (90.27, 365.81, 269.76, 230.23, 15.87, 1130.62))),
    CASES[1]: dict(zip(MAIN_METRICS, (86.53, 436.60, 274.06, 252.10, 12.88, 887.68))),
}
REFERENCE_STDS = {
    CASES[0]: dict(zip(MAIN_METRICS, (3.65, 28.08, 30.08, 17.40, 5.61, 435.81))),
    CASES[1]: dict(zip(MAIN_METRICS, (3.95, 45.83, 48.21, 18.12, 3.76, 527.24))),
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_config(config: dict) -> dict:
    config = deepcopy(config)
    config.pop("_config_path", None)
    return config


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(portable_config(config), sort_keys=True,
                                    separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def reward_signature(config: dict) -> str:
    # Exactly the canonical serialization used by the unchanged training code.
    return hashlib.sha256(json.dumps(dict(config.get("reward", {}) or {}),
                                    sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True).encode()).hexdigest()


def state_dict_hash(state: dict) -> str:
    """Stable hash of tensor names, shapes, dtypes and bytes, including buffers."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_path(case: str) -> Path:
    if case not in CASES:
        raise ValueError(f"Unknown case: {case}")
    return ROOT / "checkpoints" / f"cs_rdqn_{case}.pt"


def load_policy(case: str, path: Path | None = None, device: str = "cpu"):
    """No optimizer, replay, exploration or training is initialized here."""
    path = Path(path) if path is not None else checkpoint_path(case)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint missing: {path.name}; see checkpoints/README.md")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("release_format") != "cs_rdqn_original_weights_v1":
        raise ValueError("Expected an audited portable CS-RDQN checkpoint, not an arbitrary legacy pickle")
    if checkpoint.get("case") != case or checkpoint.get("algo") != "rainbow_dqn":
        raise ValueError("Checkpoint case/algorithm mismatch")
    if checkpoint.get("q_network_architecture") != "candidate_aware":
        raise ValueError("Only the main candidate-aware architecture is supported for reproduction")
    config = portable_config(checkpoint["evaluation_config"])
    if config_hash(config) != checkpoint["evaluation_config_sha256"]:
        raise ValueError("Embedded evaluation configuration checksum mismatch")
    if reward_signature(config) != checkpoint["reward_metadata"]["signature"]:
        raise ValueError("Checkpoint/environment reward configuration mismatch")
    if state_dict_hash(checkpoint["model_state_dict"]) != checkpoint["state_dict_sha256"]:
        raise ValueError("Checkpoint tensor checksum mismatch")
    network = make_q_network(checkpoint["algo"], checkpoint["obs_dim"],
                             checkpoint["action_dim"], checkpoint["candidate_dim"],
                             architecture=checkpoint["q_network_architecture"])
    network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    network.to(torch.device(device)).eval().requires_grad_(False)
    return network, config, checkpoint
