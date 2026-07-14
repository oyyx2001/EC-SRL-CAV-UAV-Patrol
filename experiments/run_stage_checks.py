from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dynamic_patrol_marl.env.road_network import RoadNetwork
from dynamic_patrol_marl.env.task_generator import TaskGenerator
from dynamic_patrol_marl.env.simulator import Simulator
from dynamic_patrol_marl.planner.greedy_baseline import rolling_horizon_policy
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.utils.config import load_config, project_root, ensure_dirs
from dynamic_patrol_marl.utils.visualization import plot_network, plot_task_stats, plot_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scenario_60node.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(cfg_path)
    cfg["project"]["seed"] = args.seed
    out_dir = ensure_dirs(cfg) / "stage_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []

    # S0 config check.
    with open(out_dir / "loaded_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    report.append("S0 config: OK - loaded_config.json written")

    # S1 network and shortest path.
    network = RoadNetwork(cfg)
    network.save_matrices(out_dir)
    net_val = network.validate(sample_pairs=100)
    plot_network(network, out_dir / "network_preview.png", task_arcs=network.high_risk_arcs(15), title="Network preview with high-risk arcs")
    report.append(f"S1 network: {net_val}")

    # S2 task generator.
    tg = TaskGenerator(cfg, network, args.seed)
    tasks = tg.generate_episode_tasks()
    task_df = TaskGenerator.to_dataframe(tasks)
    task_df.to_csv(out_dir / f"tasks_seed_{args.seed}.csv", index=False)
    plot_task_stats(task_df, out_dir / "task_stats.png")
    report.append(f"S2 tasks: {TaskGenerator.validate(tasks)}")

    # S3-S5 simulator + candidate sorties + baseline sample.
    sim_cand = Simulator(cfg, seed=args.seed, scenario="stage_check")
    # Inspect candidates after a short warm-up so incident/congestion tasks are active too.
    sim_cand.now = min(1200.0, float(cfg["simulation"].get("horizon_s", 3600)) / 2.0)
    sim_cand.task_pool.update_release_and_expiry(sim_cand.now)
    candidates = sim_cand.get_candidates()
    with open(out_dir / "candidate_sorties.json", "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in candidates[:30]], f, ensure_ascii=False, indent=2)
    multi = sum(len(c.task_ids) >= 2 for c in candidates)
    report.append(f"S4 candidates: count={len(candidates)}, multi_task={multi}")
    sim = Simulator(cfg, seed=args.seed, scenario="stage_check")
    metrics = sim.run(rolling_horizon_policy, method_name="rolling_horizon")
    sim.export_logs(out_dir, prefix="stage_sample")
    plot_episode(sim, out_dir / "stage_sample_episode.png", title="Stage check sample episode")
    report.append(f"S5 baseline sample: {metrics.to_dict()}")

    # S6 env API and action mask.
    env = DynamicPatrolEnv(cfg)
    reset_out = env.reset(seed=args.seed)
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    mask = env.action_masks()
    illegal = 0
    for _ in range(20):
        action = env.sample_masked_action()
        out = env.step(action)
        if len(out) == 5:
            _, _, terminated, truncated, info = out
            done = terminated or truncated
        else:
            _, _, done, info = out
        illegal += int(info.get("illegal", False))
        if done:
            break
    report.append(f"S6 env: obs_shape={getattr(obs, 'shape', None)}, mask_shape={mask.shape}, illegal_in_20_steps={illegal}")

    with open(out_dir / "stage_check_report.md", "w", encoding="utf-8") as f:
        f.write("# Stage check report\n\n")
        for line in report:
            f.write(f"- {line}\n")
    print("\n".join(report))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
