"""Evaluate a frozen CS-RDQN checkpoint. No training or fine-tuning is performed."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd
import torch

from dynamic_patrol_marl.rl.checkpoint_io import (
    CASES, MAIN_METRICS, REFERENCE_MEANS, REFERENCE_STDS, ROOT, TEST_SEEDS,
    checkpoint_path, file_hash, load_policy,
)
from dynamic_patrol_marl.rl.dqn_networks import masked_q_values
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv


def atomic_json(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for directory in ("env", "planner", "rl", "utils", "truck_uav_modules"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def asset_hashes(config: dict) -> dict:
    """Hash the packaged runtime inputs; never include a machine-local path."""
    network = config["network"]
    paths = [network[name] for name in ("graphml_file", "buildings_file", "launch_recovery_file")]
    paths.append(config["edge_tasks"]["registry_path"])
    voxel = Path(config["uav_path_planning"]["voxel_map_dir"])
    paths.extend([voxel / "metadata.json", voxel / "occupancy.npy"])
    return {Path(path).as_posix(): file_hash(ROOT / path) for path in sorted(paths, key=str)}


def task_stream_hash(tasks) -> str:
    rows = [(task.type, task.target_arcs, task.release_time, task.deadline,
             task.priority, task.service_time) for task in tasks]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def metric_values(summary: dict) -> dict:
    values = {
        "CR_pct": 100.0 * summary["completion_rate_final"],
        "MRT_s": summary["avg_response_time_s"],
        "HPRT_s": summary["high_priority_response_time_s"],
        "ECT": summary["uav_energy_per_completed_task"],
        "CTD_km": summary["cav_travel_distance_m"] / 1000.0,
        "CURW_s": summary["drone_wait_time_s"],
    }
    # Undefined response-time metrics remain missing, never replaced with zero.
    return {metric: float(value) if np.isfinite(value) else None
            for metric, value in values.items()}


def evaluate_seed(case: str, seed: int, model_path: str, device: str = "cpu") -> dict:
    started = time.perf_counter()
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    network, config, checkpoint = load_policy(case, Path(model_path), device)
    env = DynamicPatrolEnv(deepcopy(config), scenario=case)
    if (env.obs_dim, env.max_actions, env.plan_feature_dim) != (
        checkpoint["obs_dim"], checkpoint["action_dim"], checkpoint["candidate_dim"]
    ):
        raise ValueError("Environment/checkpoint observation or action dimension mismatch")
    obs, info = env.reset(seed=int(seed))
    mask = info.get("action_mask", env.action_masks())
    stream_hash = task_stream_hash(env.sim.tasks)
    done, episode_reward = False, 0.0
    with torch.inference_mode():
        while not done:
            obs_tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
            mask_tensor = torch.as_tensor(np.asarray(mask, dtype=bool), device=device).unsqueeze(0)
            if not bool(mask_tensor.any()):
                raise ValueError("Empty feasible-action mask")
            action = int(masked_q_values(network(obs_tensor, mask_tensor), mask_tensor).argmax(dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            mask = info.get("action_mask", env.action_masks())
            episode_reward += float(reward)
    summary = env.episode_summary()
    return {
        "case": case, "eval_seed": int(seed), **metric_values(summary),
        "released_task_count": int(summary["released_task_count"]),
        "completed_task_count": int(summary["completed_task_count_final"]),
        "episode_end_time_s": float(summary["episode_end_time_s"]),
        "episode_reward": episode_reward, "task_stream_sha256": stream_hash,
        "training_seed": int(checkpoint["seed"]), "training_steps": int(checkpoint["timesteps"]),
        "checkpoint_sha256": file_hash(Path(model_path)),
        "evaluation_config_sha256": checkpoint["evaluation_config_sha256"],
        "path_planning_failure_count": int(summary["path_planning_failure_count"]),
        "commit_failure_count": int(summary["commit_failure_count"]),
        "illegal_actions": int(summary["illegal_actions"]),
        "terminated": bool(terminated), "truncated": bool(truncated),
        "evaluation_wall_time_s": time.perf_counter() - started,
    }


def summarize(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(records).sort_values(["case", "eval_seed"])
    if raw.duplicated(["case", "eval_seed"]).any():
        raise ValueError("Duplicated evaluation case/seed")
    rows = []
    for case, group in raw.groupby("case", sort=True):
        full_protocol = sorted(group.eval_seed.tolist()) == list(TEST_SEEDS)
        for metric in MAIN_METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            count = int(values.count())
            mean = float(values.mean()) if count else float("nan")
            reference = REFERENCE_MEANS[case][metric]
            std = float(values.std(ddof=1)) if count > 1 else float("nan")
            reference_std = REFERENCE_STDS[case][metric]
            rows.append({
                "case": case, "metric": metric, "episodes": len(group),
                "valid_n": count, "missing_n": len(group) - count,
                "mean": mean, "std": std,
                "reference_mean": reference, "difference": mean - reference,
                "reference_std": reference_std, "std_difference": std - reference_std,
                "relative_difference_pct": 100 * (mean - reference) / reference,
                "full_20_seed_protocol": full_protocol,
                "matches_reference_2dp": bool(full_protocol and count == 20 and abs(mean - reference) < 0.005),
                "matches_reference_std_2dp": bool(full_protocol and count == 20 and abs(std - reference_std) < 0.005),
            })
    return raw, pd.DataFrame(rows)


def run_evaluation(cases, seeds, output_dir: Path, workers=1, device="cpu", checkpoint_overrides=None):
    seeds = list(seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Evaluation seeds must be nonempty and unique")
    if workers < 1 or (device != "cpu" and workers != 1):
        raise ValueError("Use workers>=1; GPU evaluation requires a single worker")
    checkpoint_overrides = checkpoint_overrides or {}
    model_paths = {}
    for case in cases:
        path = Path(checkpoint_overrides.get(case, checkpoint_path(case)))
        model_paths[case] = (path if path.is_absolute() else ROOT / path).resolve()
    # Validate everything before launching any rollout.
    assets = {}
    for case, path in model_paths.items():
        _, config, _ = load_policy(case, path, device)
        assets[case] = asset_hashes(config)
    fingerprint = source_fingerprint()
    identity = {
        "cases": list(cases), "seeds": seeds, "device": device,
        "source_sha256": fingerprint,
        "checkpoint_sha256": {case: file_hash(path) for case, path in model_paths.items()},
        "asset_sha256": assets,
        "policy": "frozen eval-mode masked argmax; NoisyNet disabled; epsilon=0",
        "numerical_settings": "embedded audited configuration; no horizon, reward or task overrides",
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "numpy", "pandas", "networkx", "shapely", "geopandas", "gymnasium")},
        "python": platform.python_version(), "platform": platform.system(),
        "thread_limit": 1, "openmp_duplicate_workaround": os.environ.get("KMP_DUPLICATE_LIB_OK", "unset"),
    }
    output_dir = Path(output_dir)
    manifest_path = output_dir / "evaluation_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != identity:
            raise ValueError("Existing evaluation identity differs; choose a new output directory")
    atomic_json(manifest_path, identity)
    records, pending = [], []
    for case in cases:
        for seed in seeds:
            path = output_dir / "episodes" / f"{case}_seed{seed}.json"
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
                if row["case"] != case or row["eval_seed"] != seed or row["checkpoint_sha256"] != identity["checkpoint_sha256"][case]:
                    raise ValueError("Cached episode identity mismatch")
                records.append(row)
            else:
                pending.append((case, seed, str(model_paths[case]), device))

    def persist(row):
        records.append(row)
        atomic_json(output_dir / "episodes" / f"{row['case']}_seed{row['eval_seed']}.json", row)
        raw, summary = summarize(records)
        raw.to_csv(output_dir / "per_seed_metrics.csv", index=False)
        summary.to_csv(output_dir / "summary_mean_std.csv", index=False)
        print(f"[{len(records)}/{len(cases)*len(seeds)}] {row['case']} seed={row['eval_seed']} "
              f"CR={row['CR_pct']:.2f}% wall={row['evaluation_wall_time_s']:.1f}s", flush=True)

    if workers == 1:
        for job in pending:
            persist(evaluate_seed(*job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(evaluate_seed, *job): job[:2] for job in pending}
            for future in as_completed(futures):
                persist(future.result())  # Errors propagate; no silent retry or fabricated row.
    raw, summary = summarize(records)
    raw.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_mean_std.csv", index=False)
    summary.to_csv(output_dir / "reference_comparison.csv", index=False)
    return raw, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TEST_SEEDS))
    parser.add_argument("--smoke", action="store_true", help="One complete seed-1000 episode, not paper reproduction")
    args = parser.parse_args()
    output = args.output_dir or ROOT / "results" / ("smoke" if args.smoke else "paper_reproduction") / args.case
    if not output.is_absolute():
        output = ROOT / output
    overrides = {args.case: args.checkpoint} if args.checkpoint else None
    _, summary = run_evaluation([args.case], [1000] if args.smoke else args.seeds,
                                output, args.workers, args.device, overrides)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
