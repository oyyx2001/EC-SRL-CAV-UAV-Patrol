from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from dynamic_patrol_marl.env.simulator import Simulator
from dynamic_patrol_marl.experiments.evaluate_masked_rl_models import (
    ModelSpec,
    _load_masked_dqn,
    _predict_dqn,
    _reset_env,
    _step_env,
)
from dynamic_patrol_marl.planner.greedy_baseline import get_policy
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.utils.config import load_config, project_root


BASELINE_METHODS: List[Tuple[str, str]] = [
    ("Nearest-task Greedy", "nearest_task"),
    ("Priority-based Greedy", "priority_greedy"),
    ("Deadline-based Greedy", "edf"),
    ("Score-based Heuristic", "score_based"),
    ("Rolling-horizon Heuristic", "rolling_horizon"),
    ("Random Masked Policy", "random_masked"),
]

RL_MODELS: List[ModelSpec] = [
    ModelSpec(
        label="Masked PPO",
        family="ppo",
        path=Path("results/rl/MaskPPO_trained_model_60node_random_seed.zip"),
    ),
    ModelSpec(
        label="Masked DQN",
        family="dqn",
        path=Path("results/rl/MaskedDQN_trained_model_60node_random_seed.zip"),
        algo="dqn",
    ),
    ModelSpec(
        label="Masked Rainbow DQN",
        family="dqn",
        path=Path("results/rl/RainbowFullMaskedDQN_60node.pt"),
        algo="rainbow_dqn",
    ),
]

SUMMARY_METRICS: List[Tuple[str, str, int]] = [
    ("completion_rate", "Completion rate", 3),
    ("timeout_rate", "Timeout rate", 3),
    ("avg_response_time_s", "Avg. response time (s)", 1),
    ("high_priority_response_time_s", "High-priority response time (s)", 1),
    ("weighted_aoi", "Weighted-AoI", 1),
    ("energy_per_completed_task", "UAV energy / completed task", 1),
    ("drone_wait_time_s", "UAV waiting time (s)", 1),
    ("vehicle_wait_time_s", "CAV waiting time (s)", 1),
    ("total_recovery_wait_time_s", "Total recovery waiting time (s)", 1),
]


def resolve_path(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def add_derived_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    completed = float(row.get("completed_tasks", 0) or 0)
    total_energy = float(row.get("total_energy", np.nan))
    row["energy_per_completed_task"] = total_energy / completed if completed > 0 else np.nan
    row["total_recovery_wait_time_s"] = float(row.get("drone_wait_time_s", 0.0) or 0.0) + float(
        row.get("vehicle_wait_time_s", 0.0) or 0.0
    )
    return row


def evaluate_baselines(cfg: Dict[str, Any], seeds: Iterable[int], scenario: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for label, policy_name in BASELINE_METHODS:
        policy = get_policy(policy_name)
        print(f"Evaluating {label}")
        for seed in seeds:
            sim = Simulator(cfg, seed=int(seed), scenario=scenario)
            metrics = sim.run(policy, method_name=label)
            row = metrics.to_dict()
            row["method"] = label
            rows.append(add_derived_metrics(row))
    return pd.DataFrame(rows)


def load_maskable_ppo(model_path: Path, cfg: Dict[str, Any], scenario: str, device_name: str) -> Any:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

    env = DynamicPatrolEnv(cfg, scenario=scenario)
    custom_objects = {
        "policy_class": MaskableActorCriticPolicy,
        "_last_obs": None,
        "_last_episode_starts": None,
        "ep_info_buffer": deque(maxlen=100),
        "ep_success_buffer": deque(maxlen=100),
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "rollout_buffer_class": MaskableRolloutBuffer,
        "clip_range": lambda _: 0.2,
        "lr_schedule": lambda _: 0.0003,
    }
    return MaskablePPO.load(model_path, custom_objects=custom_objects, device=device_name)


def evaluate_rl_models(
    cfg: Dict[str, Any],
    seeds: Iterable[int],
    scenario: str,
    root: Path,
    device_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for spec in RL_MODELS:
        model_path = resolve_path(spec.path, root)
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model for {spec.label}: {model_path}")
        print(f"Evaluating {spec.label}: {model_path}")
        if spec.family == "ppo":
            model = load_maskable_ppo(model_path, cfg, scenario, "cpu" if device_name == "auto" else device_name)
        elif spec.family == "dqn":
            model = _load_masked_dqn(model_path, spec.algo, device_name, spec.action_masking)
        else:
            raise ValueError(f"Unknown RL model family: {spec.family}")

        for seed in seeds:
            env = DynamicPatrolEnv(cfg, scenario=scenario)
            obs, info = _reset_env(env, int(seed))
            done = False
            episode_reward = 0.0
            while not done:
                action_mask = np.asarray(info["action_mask"], dtype=bool)
                if spec.family == "ppo":
                    action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
                    action = int(action)
                else:
                    action = _predict_dqn(model, obs, action_mask)
                obs, reward, done, info = _step_env(env, action)
                episode_reward += reward

            row = env.sim.metrics(spec.label).to_dict()
            row["method"] = spec.label
            row["episode_reward"] = float(episode_reward)
            row["model_path"] = str(model_path)
            rows.append(add_derived_metrics(row))
    return pd.DataFrame(rows)


def format_mean_sd(values: pd.Series, decimals: int) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return ""
    mean = float(numeric.mean())
    sd = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
    return f"{mean:.{decimals}f} ({sd:.{decimals}f})"


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    method_order = [label for label, _ in BASELINE_METHODS] + [spec.label for spec in RL_MODELS]
    rows: List[Dict[str, str]] = []
    for method in method_order:
        group = df[df["method"] == method]
        if group.empty:
            continue
        row = {"Method": method}
        for metric, title, decimals in SUMMARY_METRICS:
            row[title] = format_mean_sd(group[metric], decimals)
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown_table(table: pd.DataFrame, path: Path) -> None:
    cols = list(table.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all baseline and trained RL policies on shared seeds.")
    parser.add_argument("--config", default="configs/scenario_60node.yaml")
    parser.add_argument("--scenario", default="small")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="results/evaluation_100seeds")
    args = parser.parse_args()

    root = project_root()
    cfg = load_config(resolve_path(args.config, root))
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))

    baseline_df = evaluate_baselines(cfg, seeds, args.scenario)
    rl_df = evaluate_rl_models(cfg, seeds, args.scenario, root, args.device)
    df = pd.concat([baseline_df, rl_df], ignore_index=True)

    out_dir = resolve_path(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "eval_all_models_100seeds_raw.csv"
    summary_path = out_dir / "eval_all_models_100seeds_summary.csv"
    markdown_path = out_dir / "eval_all_models_100seeds_summary.md"

    df.to_csv(raw_path, index=False)
    table = make_summary_table(df)
    table.to_csv(summary_path, index=False)
    write_markdown_table(table, markdown_path)

    print(table.to_string(index=False))
    print(f"Saved raw results: {raw_path}")
    print(f"Saved summary table: {summary_path}")
    print(f"Saved markdown table: {markdown_path}")


if __name__ == "__main__":
    main()
