from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from dynamic_patrol_marl.experiments.run_baselines import run_many
from dynamic_patrol_marl.utils.config import load_config, project_root, ensure_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-configs", nargs="*", default=["configs/scenario_low.yaml", "configs/scenario_high.yaml", "configs/scenario_battery_limited.yaml"])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["nearest_task", "priority_greedy", "edf", "score_based", "rolling_horizon", "static_reopt", "random_masked"],
    )
    args = parser.parse_args()
    root = project_root()
    all_rows = []
    for cfg_name in args.scenario_configs:
        cfg_path = Path(cfg_name)
        if not cfg_path.is_absolute():
            cfg_path = root / cfg_path
        cfg = load_config(cfg_path)
        scenario = cfg_path.stem.replace("scenario_", "")
        out_dir = ensure_dirs(cfg) / "scenarios" / scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        df = run_many(cfg, list(range(args.seeds)), args.methods, scenario, out_dir)
        all_rows.append(df)
    combined = pd.concat(all_rows, ignore_index=True)
    out = root / "results" / "scenarios" / "comparison_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    summary = combined.groupby(["scenario", "method"])[["completion_rate", "timeout_rate", "avg_response_time_s", "weighted_aoi", "total_energy"]].agg(["mean", "std"])
    summary.to_csv(out.parent / "comparison_summary.csv")
    print(summary.round(3))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
