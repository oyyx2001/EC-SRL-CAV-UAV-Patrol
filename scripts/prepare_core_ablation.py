"""Create an immutable manifest and effective-config snapshots for core ablations."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from dynamic_patrol_marl.env.simulator import Simulator
from dynamic_patrol_marl.rl.train_masked_dqn import reward_config_metadata
from dynamic_patrol_marl.utils.config import load_config, project_root


VARIANTS = ("full", "fixed_discount", "single_task", "queue_blind")
TRAIN_SEEDS = (0, 1, 2)
EVAL_SEEDS = tuple(range(1000, 1020))
ROOT = project_root()
OUT = ROOT / "results" / "ablations" / "core_20k_repaired"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_run_snapshot(path: Path) -> dict:
    progress_files = sorted(path.rglob("progress.json")) if path.exists() else []
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "progress": [
            {
                "path": str(item.relative_to(ROOT)),
                "mtime_utc": datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc).isoformat(),
                "contents": json.loads(item.read_text(encoding="utf-8")),
            }
            for item in progress_files
        ],
    }


def effective_config(name: str, timesteps: int, *, unmasked: bool = False) -> dict:
    cfg = load_config(ROOT / "configs" / "ablations" / f"{name}.yaml")
    cfg.pop("_config_path", None)
    cfg.setdefault("project", {})["results_dir"] = "results/ablations/core_20k_repaired"
    cfg.setdefault("project", {})["tensorboard_dir"] = "results/ablations/core_20k_repaired/tensorboard_disabled"
    cfg.setdefault("rl", {})["timesteps"] = int(timesteps)
    cfg["rl"]["eval_seeds"] = list(EVAL_SEEDS)
    cfg["_ablation_execution"] = {
        "variant": name,
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "timesteps": int(timesteps),
        "device": "cuda",
        "tensorboard": False,
        "action_masking": not unmasked,
        "unmasked_diagnostic": bool(unmasked),
    }
    return cfg


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    effective = {name: effective_config(name, 20_000) for name in VARIANTS}
    for name, cfg in effective.items():
        (OUT / f"effective_config_{name}.yaml").write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    diagnostic_cfg = effective_config("full", 5_000, unmasked=True)
    (OUT / "effective_config_unmasked_diagnostic.yaml").write_text(
        yaml.safe_dump(diagnostic_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    feature_sim = Simulator(effective["full"], seed=0)
    feature_schema = feature_sim.joint_plan_generator.feature_builder.schema()
    files = [
        ROOT / "configs" / "default_training.yaml",
        *(ROOT / "configs" / "ablations" / f"{name}.yaml" for name in VARIANTS),
        ROOT / "env" / "simulator.py",
        ROOT / "env" / "cav_controller.py",
        ROOT / "planner" / "joint_plan.py",
        ROOT / "rl" / "env_wrapper.py",
        ROOT / "rl" / "train_masked_dqn.py",
    ]
    protected = [
        ROOT / "results" / "default_training" / "rainbow_xinjiekou_1m_seed42",
        ROOT / "results" / "baijiahu" / "default_training" / "rainbow_baijiahu_1m_seed42",
        ROOT / "results" / "xinjiekou_333" / "smdp_time_aware_discount",
        ROOT / "results" / "xinjiekou_333" / "reward_weight_freezing",
    ]
    manifest = {
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "study": "core_20k_minimal_ablation_screening",
        "scope": {
            "variants": list(VARIANTS),
            "control": "full",
            "train_seeds": list(TRAIN_SEEDS),
            "eval_seeds": list(EVAL_SEEDS),
            "timesteps": 20_000,
            "execution": "one new job at a time; cuda; tensorboard disabled",
            "statistical_unit": "paired train-seed / common evaluation-seed screening",
        },
        "protocol_correction": {
            "reason": "Rainbow previously used an untrained NoisyNet policy during replay warm-up; full seed 0 entered a persistent no-launch collapse.",
            "exploration": "uniform valid-action sampling through learning_starts=1000; NoisyNet-only exploration thereafter",
            "comparability": "all retained variants and all train seeds are retrained under the corrected protocol",
        },
        "excluded_variants": {
            "fixed_cav": {
                "status": "excluded_from_core_ablation",
                "reason": "The passive-patrol implementation produced zero completed tasks in all three seeds and hundreds of path-planning failures, so it changed baseline feasibility in addition to active CAV control.",
                "historical_results": "results/ablations/core_20k/fixed_cav_seed0..2",
            }
        },
        "queue_blind_definition": {
            "hidden_policy_features": [
                "recovery_wait", "cav_wait", "existing_queue_delay", "insertion_cost",
                "queue_length", "locked_prefix_length",
            ],
            "candidate_generation_and_scores": "do not use the current recovery queue",
            "physical_fifo": "retained; a private safety validator may reject unsafe execution",
        },
        "baseline": {
            "reward": reward_config_metadata(effective["full"]),
            "feature_schema": feature_schema,
            "map": {
                "version": effective["full"]["network"].get("map_version"),
                "node_count": len(feature_sim.network.nodes),
                "edge_count": len(feature_sim.network.arcs),
                "graphml_sha256": sha256(ROOT / effective["full"]["network"]["graphml_file"]),
            },
            "task_registry": {
                "path": effective["full"]["edge_tasks"]["registry_path"],
                "sha256": sha256(ROOT / effective["full"]["edge_tasks"]["registry_path"]),
                "config": effective["full"]["edge_tasks"],
            },
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "source_hashes_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in files},
        "protected_existing_runs_snapshot": [protected_run_snapshot(path) for path in protected],
        "validation_completed": {
            "targeted_pytest": "49 passed",
            "one_episode_smoke": "all four retained variants completed; dimensions 1025/25; finite primary metrics; zero illegal actions",
        },
    }
    (OUT / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(OUT)


if __name__ == "__main__":
    main()
