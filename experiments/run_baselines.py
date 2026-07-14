from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from dynamic_patrol_marl.env.simulator import Simulator
from dynamic_patrol_marl.planner.greedy_baseline import get_policy
from dynamic_patrol_marl.utils.config import load_config, project_root, ensure_dirs
from dynamic_patrol_marl.utils.visualization import plot_baseline_results, plot_episode


def run_one(cfg, seed: int, method: str, scenario: str = "small", no_multitask: bool = False, no_priority: bool = False):
    sim = Simulator(cfg, seed=seed, scenario=scenario)
    policy = get_policy(method)
    metrics = sim.run(policy, method_name=method, no_multitask=no_multitask, no_priority=no_priority)
    return metrics, sim


def run_many(cfg, seeds: List[int], methods: List[str], scenario: str, out_dir: Path, no_multitask: bool = False, no_priority: bool = False) -> pd.DataFrame:
    rows = []
    sample_sim = None
    for method in methods:
        for seed in seeds:
            metrics, sim = run_one(cfg, seed, method, scenario, no_multitask=no_multitask, no_priority=no_priority)
            rows.append(metrics.to_dict())
            if sample_sim is None:
                sample_sim = sim
    df = pd.DataFrame(rows)
    out_csv = out_dir / f"baseline_results_{scenario}.csv"
    df.to_csv(out_csv, index=False)
    plot_baseline_results(out_csv, out_dir / "plots")
    if sample_sim is not None:
        sample_sim.export_logs(out_dir / "logs", prefix=f"sample_{scenario}")
        plot_episode(sample_sim, out_dir / "plots" / f"sample_episode_{scenario}.png", title=f"Sample episode - {scenario}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scenario_60node.yaml")
    parser.add_argument("--seeds", type=int, default=None, help="number of seeds, overrides config baseline.seeds")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--scenario", default="small")
    parser.add_argument("--no-multitask", action="store_true")
    parser.add_argument("--no-priority", action="store_true")
    args = parser.parse_args()
    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(cfg_path)
    out_dir = ensure_dirs(cfg) / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    num_seeds = args.seeds if args.seeds is not None else int(cfg.get("baseline", {}).get("seeds", 30))
    methods = args.methods or cfg.get("baseline", {}).get("methods", ["nearest", "priority", "rolling_horizon"])
    seeds = list(range(num_seeds))
    df = run_many(cfg, seeds, methods, args.scenario, out_dir, no_multitask=args.no_multitask, no_priority=args.no_priority)
    print(df.groupby("method")[["completion_rate", "timeout_rate", "avg_response_time_s", "weighted_aoi", "total_energy"]].mean().round(3))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
