"""Run the authorized core 20k screening jobs strictly one at a time."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ablations" / "core_20k_repaired"
LOGS = OUT / "logs"
VARIANTS = ("full", "fixed_discount", "single_task", "queue_blind")
SEEDS = (0, 1, 2)
EVAL_SEEDS = ",".join(str(seed) for seed in range(1000, 1020))
STALE_SECONDS = 20 * 60


def write_status(**payload: object) -> None:
    payload["updated_at_utc"] = datetime.now(tz=timezone.utc).isoformat()
    (OUT / "queue_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def completed(directory: Path) -> bool:
    summary = directory / "summary.csv"
    progress = directory / "progress.json"
    if not summary.exists() or not progress.exists():
        return False
    try:
        return json.loads(progress.read_text(encoding="utf-8")).get("status") in {"complete", "completed"}
    except json.JSONDecodeError:
        return False


def wait_for_existing(directory: Path, variant: str, seed: int) -> None:
    progress = directory / "progress.json"
    while progress.exists() and not completed(directory):
        age = time.time() - progress.stat().st_mtime
        try:
            data = json.loads(progress.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"status": "unreadable_progress"}
        write_status(status="waiting_for_existing_job", variant=variant, seed=seed, progress=data, progress_age_s=age)
        if age > STALE_SECONDS:
            write_status(status="blocked_stale_existing_job", variant=variant, seed=seed, progress=data, progress_age_s=age)
            raise RuntimeError(f"Existing run appears stale: {directory}")
        time.sleep(30)


def launch(variant: str, seed: int) -> None:
    directory = OUT / f"{variant}_seed{seed}"
    if completed(directory):
        write_status(status="skipped_completed", variant=variant, seed=seed)
        return
    wait_for_existing(directory, variant, seed)
    if completed(directory):
        write_status(status="skipped_completed", variant=variant, seed=seed)
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    stdout_path = LOGS / f"{variant}_seed{seed}.stdout.log"
    stderr_path = LOGS / f"{variant}_seed{seed}.stderr.log"
    command = [
        sys.executable, "-m", "dynamic_patrol_marl.rl.train_masked_dqn",
        "--config", f"configs/ablations/{variant}.yaml",
        "--timesteps", "20000",
        "--seed", str(seed),
        "--seed-mode", "fixed",
        "--out-dir", "results/ablations/core_20k_repaired",
        "--run-name", f"{variant}_seed{seed}",
        "--disable-tensorboard",
        "--eval-seeds", EVAL_SEEDS,
        "--device", "cuda",
    ]
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(ROOT / ".matplotlib")
    (ROOT / ".matplotlib").mkdir(exist_ok=True)
    write_status(status="running", variant=variant, seed=seed, command=command)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
    if result.returncode != 0:
        write_status(status="blocked_training_exit", variant=variant, seed=seed, returncode=result.returncode)
        raise RuntimeError(f"Training failed without retry: {variant}_seed{seed}, exit={result.returncode}")
    if not completed(directory):
        write_status(status="blocked_missing_completion_artifact", variant=variant, seed=seed)
        raise RuntimeError(f"Training exited but completion artifacts are missing: {directory}")
    write_status(status="completed", variant=variant, seed=seed)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        for variant in VARIANTS:
            for seed in SEEDS:
                launch(variant, seed)
        write_status(status="aggregating")
        result = subprocess.run([sys.executable, "scripts/analyze_core_ablation.py"], cwd=ROOT)
        if result.returncode != 0:
            write_status(status="blocked_aggregation_exit", returncode=result.returncode)
            raise RuntimeError(f"Aggregation failed, exit={result.returncode}")
        write_status(status="completed_all")
    except Exception as exc:
        failure = OUT / "failure_log.md"
        existing = failure.read_text(encoding="utf-8") if failure.exists() else "# Failure / execution log\n"
        failure.write_text(existing + f"- Queue stopped: {exc}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
