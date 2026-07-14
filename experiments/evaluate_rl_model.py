from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.utils.config import load_config, project_root


def _load_maskable_ppo(model_path: Path):
    try:
        from sb3_contrib import MaskablePPO
    except Exception as exc:
        raise RuntimeError(
            "RL evaluation requires optional dependencies: "
            "gymnasium, stable-baselines3, sb3-contrib, and torch."
        ) from exc
    return MaskablePPO.load(model_path)


def evaluate_model(cfg: Dict, model_path: Path, seeds: int, scenario: str) -> pd.DataFrame:
    model = _load_maskable_ppo(model_path)
    rows: List[Dict] = []
    for seed in range(int(seeds)):
        env = DynamicPatrolEnv(cfg, scenario=scenario)
        obs, info = env.reset(seed=seed)
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += float(reward)
            done = bool(terminated or truncated)
        row = env.sim.metrics("rl_model").to_dict()
        row["episode_reward"] = episode_reward
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scenario_60node.yaml")
    parser.add_argument("--model", default="results/rl/trained_model.zip")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--scenario", default="eval")
    parser.add_argument("--out", default="results/rl/eval_rl_model.csv")
    args = parser.parse_args()

    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path

    cfg = load_config(cfg_path)
    df = evaluate_model(cfg, model_path=model_path, seeds=args.seeds, scenario=args.scenario)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    metrics = [
        "completion_rate",
        "timeout_rate",
        "avg_response_time_s",
        "high_priority_response_time_s",
        "weighted_aoi",
        "total_energy",
        "vehicle_wait_time_s",
        "drone_wait_time_s",
        "drone_utilization",
        "tasks_per_hour",
        "illegal_actions",
        "avg_tasks_per_sortie",
        "num_sorties",
        "episode_reward",
    ]
    print(df[metrics].mean(numeric_only=True).round(4).to_string())
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
