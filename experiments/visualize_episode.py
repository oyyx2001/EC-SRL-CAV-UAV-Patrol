from __future__ import annotations

import argparse
from pathlib import Path

from dynamic_patrol_marl.env.simulator import Simulator
from dynamic_patrol_marl.planner.greedy_baseline import get_policy
from dynamic_patrol_marl.utils.config import load_config, project_root, ensure_dirs
from dynamic_patrol_marl.utils.visualization import plot_episode, plot_episode_animation, plot_episode_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scenario_60node.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default="rolling_horizon")
    parser.add_argument("--animation-frames", type=int, default=60)
    parser.add_argument("--animation-fps", type=int, default=5)
    parser.add_argument("--skip-animation", action="store_true")
    args = parser.parse_args()
    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(cfg_path)
    out_dir = ensure_dirs(cfg) / "episode_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = Simulator(cfg, seed=args.seed, scenario="visual")
    metrics = sim.run(get_policy(args.method), method_name=args.method)
    sim.export_logs(out_dir, prefix=f"episode_seed_{args.seed}_{args.method}")
    out_path = out_dir / f"episode_seed_{args.seed}_{args.method}.png"
    plot_episode(sim, out_path, title=f"{args.method}, seed={args.seed}")
    tasks_path = out_dir / f"episode_seed_{args.seed}_{args.method}_tasks_map.png"
    plot_episode_tasks(sim, tasks_path, title=f"generated tasks, seed={args.seed}")
    animation_path = out_dir / f"episode_seed_{args.seed}_{args.method}_animation.gif"
    if not args.skip_animation:
        plot_episode_animation(
            sim,
            animation_path,
            title=f"{args.method}, seed={args.seed}",
            frames=args.animation_frames,
            fps=args.animation_fps,
        )
    print(metrics.to_dict())
    print(f"Saved: {out_path}")
    print(f"Saved task map: {tasks_path}")
    if not args.skip_animation:
        print(f"Saved animation: {animation_path}")


if __name__ == "__main__":
    main()
