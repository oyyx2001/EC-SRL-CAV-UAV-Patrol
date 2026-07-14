from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from dynamic_patrol_marl.env.simulator import Simulator
from dynamic_patrol_marl.planner.greedy_baseline import get_policy
from dynamic_patrol_marl.utils.config import load_config, project_root


METRICS = [
    "completion_rate",
    "timeout_rate",
    "avg_response_time_s",
    "high_priority_response_time_s",
    "weighted_aoi",
    "uav_energy_per_completed_task",
    "mean_uav_recovery_wait_s",
    "p95_uav_recovery_wait_s",
    "total_uav_hover_time_s",
    "total_hover_energy",
    "mean_cav_waiting_time_s",
    "cav_travel_distance_m",
    "recovery_contention_rate",
    "non_append_rate",
    "mean_insertion_displacement",
    "mean_counterfactual_gain",
    "dynamic_insertion_computation_time_ms",
]


def config_for_rule(base_cfg: Dict, rule: str) -> Dict:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("recovery", {})["rule"] = rule
    cfg.setdefault("recovery_insertion", {})["enabled"] = rule == "dynamic_insertion"
    return cfg


def run_paired(
    base_cfg: Dict,
    seeds: Iterable[int],
    method: str,
    scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []
    insertion_rows: List[Dict] = []
    policy = get_policy(method)
    for rule in ("fifo", "dynamic_insertion"):
        cfg = config_for_rule(base_cfg, rule)
        for seed in seeds:
            sim = Simulator(cfg, seed=int(seed), scenario=scenario)
            metrics = sim.run(policy, method_name=method)
            row = metrics.to_dict()
            row["recovery_rule"] = rule
            drone_ids = [str(entry.get("drone_id")) for entry in sim.recovery_queue]
            row["final_queue_duplicate_count"] = len(drone_ids) - len(set(drone_ids))
            row["unreachable_recovery_count"] = sum(
                not math.isfinite(sim.network.get_vehicle_time(sim.vehicle.position, int(entry["recovery_node"])))
                for entry in sim.recovery_queue
            )
            row["negative_battery_count"] = sum(d.battery < -1e-9 for d in sim.drones)
            rows.append(row)
            insertion_rows.extend(dict(event) for event in sim.insertion_log)
    return pd.DataFrame(rows), pd.DataFrame(insertion_rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    fifo = raw[raw["recovery_rule"] == "fifo"].set_index("seed")
    dynamic = raw[raw["recovery_rule"] == "dynamic_insertion"].set_index("seed")
    common = fifo.index.intersection(dynamic.index)
    records = []
    for metric in METRICS:
        fifo_values = pd.to_numeric(fifo.loc[common, metric], errors="coerce")
        dynamic_values = pd.to_numeric(dynamic.loc[common, metric], errors="coerce")
        valid = fifo_values.notna() & dynamic_values.notna()
        f = fifo_values[valid].to_numpy(dtype=float)
        d = dynamic_values[valid].to_numpy(dtype=float)
        diff = d - f
        n = len(diff)
        diff_mean = float(np.mean(diff)) if n else math.nan
        half_width = float(1.96 * np.std(diff, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        fifo_mean = float(np.mean(f)) if n else math.nan
        dynamic_mean = float(np.mean(d)) if n else math.nan
        relative = ((dynamic_mean - fifo_mean) / abs(fifo_mean) * 100.0) if n and abs(fifo_mean) > 1e-12 else math.nan
        records.append(
            {
                "metric": metric,
                "paired_seed_count": n,
                "fifo_mean": fifo_mean,
                "fifo_std": float(np.std(f, ddof=1)) if n > 1 else 0.0,
                "dynamic_mean": dynamic_mean,
                "dynamic_std": float(np.std(d, ddof=1)) if n > 1 else 0.0,
                "relative_change_pct": relative,
                "paired_difference_dynamic_minus_fifo": diff_mean,
                "paired_difference_ci95_low": diff_mean - half_width if n else math.nan,
                "paired_difference_ci95_high": diff_mean + half_width if n else math.nan,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 FIFO vs dynamic recovery insertion experiment")
    parser.add_argument("--config", default="configs/scenario_60node_rainbow_test.yaml")
    parser.add_argument("--seeds", type=int, default=20, help="paired seeds 0..N-1")
    parser.add_argument("--seed-list", default=None, help="comma-separated paired seeds; overrides --seeds")
    parser.add_argument("--method", default="score_based")
    parser.add_argument("--scenario", default="small")
    parser.add_argument("--out-dir", default="results/stage1_dynamic_recovery")
    args = parser.parse_args()

    root = project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seed_list.split(",")] if args.seed_list else list(range(args.seeds))

    raw, insertions = run_paired(load_config(config_path), seeds, args.method, args.scenario)
    summary = summarize(raw)
    raw.to_csv(out_dir / "stage1_raw_results.csv", index=False)
    summary.to_csv(out_dir / "stage1_summary.csv", index=False)
    insertions.to_csv(out_dir / "stage1_insertion_events.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Raw results: {out_dir / 'stage1_raw_results.csv'}")
    print(f"Summary: {out_dir / 'stage1_summary.csv'}")
    print(f"Insertion log: {out_dir / 'stage1_insertion_events.csv'}")


if __name__ == "__main__":
    main()
