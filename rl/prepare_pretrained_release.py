"""Repackage the two verified original checkpoints; never trains or invents weights.

Private run paths and optimizer state are removed. Every model tensor and buffer
must remain bitwise identical to the original artifact. This is a maintainer
utility; evaluation users can directly use the two released .pt files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from dynamic_patrol_marl.rl.checkpoint_io import (
    CASES, ROOT, checkpoint_path, config_hash, file_hash, load_policy,
    portable_config, reward_signature, state_dict_hash,
)
from dynamic_patrol_marl.utils.config import load_config

SOURCES = {
    CASES[0]: (
        "results/algorithm_comparison/xinjiekou_random_traffic_300k_v1/rainbow_dqn/"
        "rainbow_xinjiekou_random_traffic_300k_seed42_20260918/"
        "rainbow_xinjiekou_random_traffic_300k_seed42_20260918.pt",
        "42c587522c55fa055fc9c77a3120d4a072899bbdc1e751b1f4d8119f568f208f",
    ),
    CASES[1]: (
        "results/algorithm_comparison/baijiahu_random_traffic_300k_v1/rainbow_dqn/"
        "rainbow_baijiahu_random_traffic_300k_seed42_20260920/"
        "rainbow_baijiahu_random_traffic_300k_seed42_20260920.pt",
        "619b90b1a5d7a471d8ed14817a01ad3f07b14edd84532ccb0e3270776cd5dd31",
    ),
}


def publish(case: str, source: Path) -> dict:
    expected_hash = SOURCES[case][1]
    if file_hash(source) != expected_hash:
        raise ValueError(f"Original paper checkpoint hash mismatch for {case}; publication stopped")
    # Only these locally verified originals are loaded using the legacy pickle mode.
    original = torch.load(source, map_location="cpu", weights_only=False)
    record = pd.read_csv(source.parent / "summary.csv").iloc[0]
    steps = sum(int(stage["timesteps"]) for stage in original["training_stages"])
    if steps != 300000 or int(record["timesteps"]) != steps or original["seed"] != 42:
        raise ValueError("Original training evidence does not support seed 42 / final 300k")
    config = portable_config(load_config(ROOT / "configs" / f"{case}.yaml"))
    if reward_signature(config) != original["reward_metadata"]["signature"]:
        raise ValueError("Original checkpoint and release reward signatures differ")
    if (original["obs_dim"], original["action_dim"], original["candidate_dim"]) != (1025, 25, 38):
        raise ValueError("Unexpected original model dimensions")
    # Original final .pt files omit a top-level timesteps key. Evidence is retained,
    # not silently guessed from their filenames.
    metadata = {
        "release_format": "cs_rdqn_original_weights_v1", "case": case,
        "algo": original["algo"], "seed": original["seed"], "timesteps": steps,
        "step_evidence": "Original training_stages plus matching run summary.csv",
        "selection": "original final 300k; not the fleet-sensitivity validation-best model",
        "obs_dim": 1025, "action_dim": 25, "candidate_dim": 38,
        "raw_global_features": 75, "global_encoder_input_dim": 113,
        "q_network_architecture": "candidate_aware", "decision_mode": "joint_plan",
        "hyperparams": original["hyperparams"], "replay_config": original["replay_config"],
        "training_stages": original["training_stages"],
        "feature_schema": original["feature_schema"],
        "reward_metadata": original["reward_metadata"],
        "evaluation_config": config, "evaluation_config_sha256": config_hash(config),
        "original_artifact_sha256": expected_hash,
        "state_dict_sha256": state_dict_hash(original["model_state_dict"]),
        "historical_training_yaml_available": False,
        "configuration_provenance": "Audited current canonical runtime; historical entry YAML is unavailable",
        "config_contract": "Numerical evaluation settings unchanged; public case labels standardized",
        "weight_integrity": "Every parameter and buffer bitwise identical; optimizer/private paths omitted",
    }
    # JSON round-trip makes metadata weights_only-safe and strips no tensor data.
    metadata = json.loads(json.dumps(metadata))
    target = checkpoint_path(case)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = torch.load(target, map_location="cpu", weights_only=True)
        if (existing.get("state_dict_sha256") != metadata["state_dict_sha256"]
                or existing.get("evaluation_config_sha256") != metadata["evaluation_config_sha256"]):
            raise ValueError(f"Refusing to overwrite a different published checkpoint: {target.name}")
    else:
        torch.save({**metadata, "model_state_dict": original["model_state_dict"]}, target)
    published = torch.load(target, map_location="cpu", weights_only=True)
    for key, tensor in original["model_state_dict"].items():
        if not torch.equal(tensor, published["model_state_dict"][key]):
            raise AssertionError(f"Changed original tensor: {key}")
    load_policy(case)
    sidecar = {**metadata, "published_artifact_sha256": file_hash(target),
               "published_size_bytes": target.stat().st_size}
    target.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return {"case": case, "artifact": target.relative_to(ROOT).as_posix(),
            "sha256": sidecar["published_artifact_sha256"],
            "state_dict_sha256": metadata["state_dict_sha256"],
            "original_artifact_sha256": expected_hash, "weight_integrity_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-case1", type=Path)
    parser.add_argument("--source-case2", type=Path)
    args = parser.parse_args()
    sources = [args.source_case1 or ROOT / SOURCES[CASES[0]][0],
               args.source_case2 or ROOT / SOURCES[CASES[1]][0]]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Original checkpoints missing; never generate replacements:\n" + "\n".join(missing))
    records = [publish(case, source) for case, source in zip(CASES, sources)]
    (ROOT / "checkpoints/manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
