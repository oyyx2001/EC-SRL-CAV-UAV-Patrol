from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from dynamic_patrol_marl.env.simulator import Simulator
from dynamic_patrol_marl.planner.greedy_baseline import get_policy
from dynamic_patrol_marl.utils.config import load_config, project_root, ensure_dirs, deep_update
from dynamic_patrol_marl.utils.visualization import plot_baseline_results


def run_variant(cfg, seed: int, variant: str, method: str = "rolling_horizon"):
    local_cfg = cfg
    no_multitask = False
    no_priority = False
    if variant == "no_multitask":
        no_multitask = True
        local_cfg = deep_update(cfg, {"sortie": {"allow_multi_task": False, "max_tasks_per_sortie": 1}})
    elif variant == "no_priority":
        no_priority = True
    elif variant == "no_aoi":
        local_cfg = deep_update(cfg, {"reward": {"aoi_penalty": 0.0}})
    elif variant == "no_rolling_replan":
        method = "static_reopt"
    elif variant == "full":
        pass
    else:
        raise ValueError(variant)
    sim = Simulator(local_cfg, seed=seed, scenario=variant)
    metrics = sim.run(get_policy(method), method_name=variant, no_multitask=no_multitask, no_priority=no_priority)
    return metrics.to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scenario_60node.yaml")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--variants", nargs="*", default=["full", "no_multitask", "no_priority", "no_aoi", "no_rolling_replan"])
    args = parser.parse_args()
    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(cfg_path)
    out_dir = ensure_dirs(cfg) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in args.variants:
        for seed in range(args.seeds):
            rows.append(run_variant(cfg, seed, variant))
    df = pd.DataFrame(rows)
    out_csv = out_dir / "ablation_results.csv"
    df.to_csv(out_csv, index=False)
    plot_baseline_results(out_csv, out_dir / "ablation_plots")
    summary = df.groupby("method")[["completion_rate", "timeout_rate", "avg_response_time_s", "weighted_aoi", "total_energy", "avg_tasks_per_sortie"]].agg(["mean", "std"])
    summary.to_csv(out_dir / "ablation_summary.csv")
    with open(out_dir / "ablation_summary.md", "w", encoding="utf-8") as f:
        f.write("# Ablation summary\n\n")
        f.write(summary.round(3).to_markdown())
    print(summary.round(3))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
