from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from dynamic_patrol_marl.rl.train_masked_dqn import (
    ALGORITHMS,
    DQNHyperParams,
    parse_eval_seeds,
    train_masked_dqn,
)
from dynamic_patrol_marl.rl.paths import resolve_dqn_output_dirs
from dynamic_patrol_marl.rl.seed_scheduler import SEED_MODES, parse_int_list
from dynamic_patrol_marl.utils.config import load_config, project_root


DEFAULT_ALGO_PARAMS: Dict[str, DQNHyperParams] = {
    "dqn": DQNHyperParams(),
    "double_dqn": DQNHyperParams(),
    "dueling_double_dqn": DQNHyperParams(),
    "rainbow_dqn": DQNHyperParams(epsilon_start=0.0, epsilon_final=0.0, epsilon_decay_steps=1),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scenario_60node.yaml")
    parser.add_argument("--algos", nargs="*", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seeds", default="1000,1001,1002,1003,1004")
    parser.add_argument("--scenario", default="small")
    parser.add_argument("--eval-scenario", default="eval")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default=None, help="default: results/rl/dqn")
    parser.add_argument("--tensorboard-dir", default=None, help="default: results/rl/tensorboard/dqn")
    parser.add_argument("--disable-tensorboard", action="store_true", help="train without writing TensorBoard event files")
    parser.add_argument("--seed-mode", choices=SEED_MODES, default=None, help="default: training.seed_mode from config, else fixed")
    parser.add_argument("--stage-timesteps", default=None, help="comma-separated stage timesteps; one value or one per stage")
    parser.add_argument("--model-name", default=None, help="optional model filename; multi-algo runs append the algo name")
    args = parser.parse_args()

    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(cfg_path)
    out_dir, tensorboard_dir = resolve_dqn_output_dirs(root, args.out_dir, args.tensorboard_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_paths: List[Path] = []
    for algo in args.algos:
        run_name = f"{algo}_seed{args.seed}"
        if args.model_name and len(args.algos) > 1:
            name_path = Path(args.model_name)
            model_name = f"{name_path.stem}_{algo}{name_path.suffix or '.pt'}"
        else:
            model_name = args.model_name or f"{run_name}.pt"
        train_masked_dqn(
            cfg=cfg,
            cfg_path=str(cfg_path),
            algo=algo,
            total_timesteps=args.timesteps,
            seed=args.seed,
            out_dir=out_dir,
            hp=DEFAULT_ALGO_PARAMS[algo],
            eval_seeds=parse_eval_seeds(args.eval_seeds),
            eval_scenario=args.eval_scenario,
            device_name=args.device,
            model_name=model_name,
            train_scenario=args.scenario,
            run_name=run_name,
            tensorboard_dir=tensorboard_dir,
            enable_tensorboard=not args.disable_tensorboard,
            seed_mode=args.seed_mode,
            stage_timesteps=parse_int_list(args.stage_timesteps),
        )
        summary_paths.append(out_dir / run_name / "summary.csv")

    summaries = [pd.read_csv(path) for path in summary_paths if path.exists()]
    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        comparison_dir = out_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(comparison_dir / "rl_comparison_summary.csv", index=False)
        cols = [c for c in ["algo", "final_completion_rate", "final_timeout_rate", "final_episode_reward"] if c in combined.columns]
        print(combined[cols].to_string(index=False))
    print(f"Saved comparison outputs: {out_dir}")


if __name__ == "__main__":
    main()
