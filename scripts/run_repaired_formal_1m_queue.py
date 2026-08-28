"""Run the corrected Xinjiekou/Baijiahu 1M trainings one at a time.

Historical 320k checkpoints are intentionally not resumed: those checkpoints
do not contain replay-buffer, RNG, target-network, or accumulated-step state.
This queue writes to a new output root and never overwrites an incomplete run.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "formal_1m_repaired"
LOGS = OUT / "logs"
EVAL_SEEDS = ",".join(str(seed) for seed in range(1000, 1020))
RUNS = (
    {
        "scene": "xinjiekou",
        "config": "configs/default_training.yaml",
        "run_name": "rainbow_xinjiekou_1m_seed42_repaired",
        "model_name": "rainbow_xinjiekou_1m_seed42_repaired.pt",
    },
    {
        "scene": "baijiahu",
        "config": "configs/baijiahu_training.yaml",
        "run_name": "rainbow_baijiahu_1m_seed42_repaired",
        "model_name": "rainbow_baijiahu_1m_seed42_repaired.pt",
    },
)


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_status(**payload: object) -> None:
    payload["updated_at_utc"] = utc_now()
    write_json(OUT / "queue_status.json", payload)


def sha256(relative_path: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative_path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def completed(directory: Path) -> bool:
    progress = directory / "progress.json"
    summary = directory / "summary.csv"
    if not progress.exists() or not summary.exists():
        return False
    try:
        status = json.loads(progress.read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError):
        return False
    return status in {"complete", "completed"}


def write_manifest() -> None:
    sources = (
        "configs/default_training.yaml",
        "configs/baijiahu_training.yaml",
        "rl/train_masked_dqn.py",
        "rl/env_wrapper.py",
        "env/simulator.py",
        "planner/joint_plan.py",
    )
    manifest = {
        "created_at_utc": utc_now(),
        "study": "repaired_formal_1m_cross_scene_training",
        "decision": {
            "mode": "restart_from_scratch",
            "reason": (
                "The historical 320k checkpoints omit replay-buffer, RNG, target-network, "
                "and accumulated-step state, so --resume-from is not a faithful continuation."
            ),
            "historical_results_preserved": [
                "results/default_training/rainbow_xinjiekou_1m_seed42",
                "results/baijiahu/default_training/rainbow_baijiahu_1m_seed42",
            ],
        },
        "protocol": {
            "execution": "strictly sequential: Xinjiekou then Baijiahu",
            "timesteps_per_scene": 1_000_000,
            "base_seed": 42,
            "seed_mode": "random",
            "eval_seeds": list(range(1000, 1020)),
            "device": "cuda",
            "exploration": "uniform valid actions through learning_starts=1000, then NoisyNet",
            "tensorboard": True,
            "automatic_retry": False,
        },
        "runs": list(RUNS),
        "source_hashes_sha256": {path: sha256(path) for path in sources},
    }
    write_json(OUT / "run_manifest.json", manifest)


def launch(spec: dict[str, str]) -> None:
    scene = spec["scene"]
    run_name = spec["run_name"]
    directory = OUT / run_name
    if completed(directory):
        write_status(status="skipped_completed", scene=scene, run_name=run_name)
        return
    if directory.exists():
        write_status(status="blocked_existing_incomplete_directory", scene=scene, run_name=run_name)
        raise RuntimeError(f"Refusing to overwrite incomplete run directory: {directory}")

    LOGS.mkdir(parents=True, exist_ok=True)
    stdout_path = LOGS / f"{scene}.stdout.log"
    stderr_path = LOGS / f"{scene}.stderr.log"
    command = [
        sys.executable,
        "-B",
        "-m",
        "dynamic_patrol_marl.rl.train_masked_dqn",
        "--config",
        spec["config"],
        "--timesteps",
        "1000000",
        "--seed",
        "42",
        "--seed-mode",
        "random",
        "--out-dir",
        "results/formal_1m_repaired",
        "--tensorboard-dir",
        "results/formal_1m_repaired/tensorboard",
        "--run-name",
        run_name,
        "--model-name",
        spec["model_name"],
        "--eval-seeds",
        EVAL_SEEDS,
        "--device",
        "cuda",
    ]
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(ROOT / ".matplotlib")
    environment["PYTHONUNBUFFERED"] = "1"
    (ROOT / ".matplotlib").mkdir(exist_ok=True)
    write_status(status="running", scene=scene, run_name=run_name, command=command)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
    if result.returncode != 0:
        write_status(
            status="blocked_training_exit",
            scene=scene,
            run_name=run_name,
            returncode=result.returncode,
        )
        raise RuntimeError(f"Training failed without retry: {run_name}, exit={result.returncode}")
    if not completed(directory):
        write_status(status="blocked_missing_completion_artifact", scene=scene, run_name=run_name)
        raise RuntimeError(f"Training exited without complete artifacts: {directory}")
    write_status(status="completed", scene=scene, run_name=run_name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_manifest()
    try:
        for spec in RUNS:
            launch(spec)
        write_status(status="completed_all")
    except Exception as exc:
        failure_path = OUT / "failure_log.md"
        existing = failure_path.read_text(encoding="utf-8") if failure_path.exists() else "# Failure log\n\n"
        failure_path.write_text(existing + f"- {utc_now()}: {exc}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
