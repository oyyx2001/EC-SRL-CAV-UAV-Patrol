from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from dynamic_patrol_marl.experiments.evaluate_all_models_100seeds import (
    SUMMARY_METRICS,
    add_derived_metrics,
    format_mean_sd,
    resolve_path,
    write_markdown_table,
)
from dynamic_patrol_marl.experiments.evaluate_masked_rl_models import (
    _load_masked_dqn,
    _predict_dqn,
    _reset_env,
    _step_env,
)
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.utils.config import load_config, project_root


@dataclass(frozen=True)
class AblationSpec:
    label: str
    config: Path
    model: Path
    action_masking: bool = True


ABLATIONS: List[AblationSpec] = [
    AblationSpec(
        label="Full Masked Rainbow DQN",
        config=Path("configs/ablations/rainbow_60node_full_masked.yaml"),
        model=Path("results/rl/rainbow_dqn_ablation/rainbow_full_masked_60node/RainbowFullMaskedDQN_60node.pt"),
    ),
    AblationSpec(
        label="w/o Action Mask",
        config=Path("configs/ablations/rainbow_60node_wo_action_mask.yaml"),
        model=Path("results/rl/rainbow_dqn_ablation/rainbow_wo_action_mask_60node/RainbowDQN_wo_action_mask_60node.pt"),
        action_masking=False,
    ),
    AblationSpec(
        label="w/o Candidate-sortie Features",
        config=Path("configs/ablations/rainbow_60node_wo_candidate_sortie_features.yaml"),
        model=Path(
            "results/rl/rainbow_dqn_ablation/rainbow_wo_candidate_sortie_features_60node/"
            "RainbowDQN_wo_candidate_sortie_features_60node.pt"
        ),
    ),
    AblationSpec(
        label="w/o Queue-aware Recovery",
        config=Path("configs/ablations/rainbow_60node_wo_queue_aware_recovery.yaml"),
        model=Path(
            "results/rl/rainbow_dqn_ablation/rainbow_wo_queue_aware_recovery_60node/"
            "RainbowDQN_wo_queue_aware_recovery_60node.pt"
        ),
    ),
    AblationSpec(
        label="Single-task Sorties",
        config=Path("configs/ablations/rainbow_60node_max_tasks_1.yaml"),
        model=Path("results/rl/rainbow_dqn_ablation/rainbow_max_tasks_1_60node/RainbowDQN_max_tasks_1_60node.pt"),
    ),
    AblationSpec(
        label="w/o Priority-urgency Score",
        config=Path("configs/ablations/rainbow_60node_wo_priority_urgency_score.yaml"),
        model=Path(
            "results/rl/rainbow_dqn_ablation/rainbow_wo_priority_urgency_score_60node/"
            "RainbowDQN_wo_priority_urgency_score_60node.pt"
        ),
    ),
]


def evaluate_spec(
    spec: AblationSpec,
    root: Path,
    seeds: Iterable[int],
    scenario: str,
    device_name: str,
) -> pd.DataFrame:
    cfg_path = resolve_path(spec.config, root)
    model_path = resolve_path(spec.model, root)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config for {spec.label}: {cfg_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model for {spec.label}: {model_path}")

    cfg = load_config(cfg_path)
    model = _load_masked_dqn(model_path, "rainbow_dqn", device_name, spec.action_masking)
    rows: List[Dict[str, Any]] = []
    print(f"Evaluating {spec.label}: {model_path}")
    for seed in seeds:
        env = DynamicPatrolEnv(cfg, scenario=scenario)
        obs, info = _reset_env(env, int(seed))
        done = False
        episode_reward = 0.0
        while not done:
            action_mask = np.asarray(info["action_mask"], dtype=bool)
            action = _predict_dqn(model, obs, action_mask)
            obs, reward, done, info = _step_env(env, action)
            episode_reward += reward

        row = env.sim.metrics(spec.label).to_dict()
        row["method"] = spec.label
        row["seed"] = int(seed)
        row["episode_reward"] = float(episode_reward)
        row["config_path"] = str(cfg_path)
        row["model_path"] = str(model_path)
        rows.append(add_derived_metrics(row))
    return pd.DataFrame(rows)


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for spec in ABLATIONS:
        group = df[df["method"] == spec.label]
        if group.empty:
            continue
        row = {"Model": spec.label}
        for metric, title, decimals in SUMMARY_METRICS:
            row[title] = format_mean_sd(group[metric], decimals)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Rainbow DQN ablation checkpoints on shared seeds.")
    parser.add_argument("--scenario", default="small")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="results/evaluation_ablation_100seeds")
    args = parser.parse_args()

    root = project_root()
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))
    frames = [evaluate_spec(spec, root, seeds, args.scenario, args.device) for spec in ABLATIONS]
    df = pd.concat(frames, ignore_index=True)

    out_dir = resolve_path(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "eval_rainbow_ablation_100seeds_raw.csv"
    summary_path = out_dir / "eval_rainbow_ablation_100seeds_summary.csv"
    markdown_path = out_dir / "eval_rainbow_ablation_100seeds_summary.md"

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
