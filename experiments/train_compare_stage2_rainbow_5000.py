from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from dynamic_patrol_marl.experiments.evaluate_masked_rl_models import (
    _load_masked_dqn,
    _predict_dqn,
    _reset_env,
    _step_env,
)
from dynamic_patrol_marl.experiments.run_stage2_event_smdp import paired_statistics, parse_seeds
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.rl.train_masked_dqn import DQNHyperParams, train_masked_dqn
from dynamic_patrol_marl.utils.config import load_config, project_root


EVALUATION_METRICS = [
    "completion_rate",
    "timeout_rate",
    "avg_response_time_s",
    "high_priority_response_time_s",
    "weighted_aoi",
    "uav_energy_per_completed_task",
    "mean_uav_recovery_wait_s",
    "p95_uav_recovery_wait_s",
    "total_hover_energy",
    "vehicle_wait_time_s",
    "cav_travel_distance_m",
    "non_append_rate",
    "episode_reward",
    "decisions_per_episode",
    "forced_wait_decision_rate",
    "mean_delta_t",
]


def make_shared_hyperparams(cfg: Dict[str, Any], args: argparse.Namespace) -> DQNHyperParams:
    values = asdict(DQNHyperParams())
    configured = cfg.get("rl", {}).get("hyperparams", {}) or {}
    values.update({key: value for key, value in configured.items() if key in values})
    values.update(
        {
            "batch_size": int(args.batch_size),
            "learning_starts": int(args.learning_starts),
            "train_frequency": int(args.train_frequency),
            "target_update_interval": int(args.target_update_interval),
            "eval_interval": int(args.epochs),
            "replay_buffer_size": max(int(values["replay_buffer_size"]), int(args.epochs) * 2),
        }
    )
    return DQNHyperParams(**values)


def assert_fair_configuration(
    fixed_cfg: Dict[str, Any],
    event_cfg: Dict[str, Any],
    shared_hp: DQNHyperParams,
) -> Dict[str, Any]:
    required_equal_sections = [
        "project",
        "simulation",
        "network",
        "agents",
        "recovery",
        "recovery_insertion",
        "tasks",
        "sortie",
        "reward",
        "ablation",
    ]
    mismatches = []
    for section in required_equal_sections:
        if fixed_cfg.get(section, {}) != event_cfg.get(section, {}):
            mismatches.append(section)
    if mismatches:
        raise ValueError(f"B1/B2 fairness violation in shared config sections: {mismatches}")
    fixed_smdp = fixed_cfg.get("smdp", {})
    event_smdp = event_cfg.get("smdp", {})
    for key in ("reference_interval_s", "gamma_reference", "store_delta_t", "interval_normalize_reward", "n_step"):
        if fixed_smdp.get(key) != event_smdp.get(key):
            raise ValueError(f"B1/B2 fairness violation in smdp.{key}")
    if fixed_cfg["reward"].get("accounting_mode") != "interval_increment":
        raise ValueError("fixed mode must use interval_increment reward accounting")
    fixed_env = DynamicPatrolEnv(fixed_cfg, scenario="small")
    event_env = DynamicPatrolEnv(event_cfg, scenario="small")
    fixed_env.reset(seed=0)
    event_env.reset(seed=0)
    if fixed_env.observation_space.shape != event_env.observation_space.shape:
        raise ValueError("observation dimensions differ")
    if fixed_env.action_space.n != event_env.action_space.n:
        raise ValueError("action dimensions differ")
    return {
        "fairness_check_passed": True,
        "operational_epoch_definition": "one environment transition and at most one scheduled gradient update",
        "training_epochs": None,
        "training_seed": None,
        "equal_sections": required_equal_sections,
        "reward_configuration": fixed_cfg["reward"],
        "shared_hyperparameters": asdict(shared_hp),
        "observation_dim": int(fixed_env.observation_space.shape[0]),
        "action_dim": int(fixed_env.action_space.n),
        "allowed_differences": {
            "time.mode": ["fixed_interval", "event_driven"],
            "event_driven.enabled": [False, True],
            "smdp.enabled": [False, True],
            "smdp.discount_mode": ["fixed_decision", "continuous_time"],
        },
    }


def train_one(
    cfg: Dict[str, Any],
    cfg_path: Path,
    label: str,
    epochs: int,
    seed: int,
    hp: DQNHyperParams,
    out_dir: Path,
    device: str,
    seed_mode: str,
) -> Tuple[Path, Dict[str, Any]]:
    run_name = f"rainbow_{label}_{epochs}epochs_{seed_mode}_seed{seed}"
    model_name = f"Rainbow_{label}_{epochs}epochs_{seed_mode}_seed{seed}.pt"
    start = time.perf_counter()
    model_path = train_masked_dqn(
        cfg=deepcopy(cfg),
        cfg_path=str(cfg_path),
        algo="rainbow_dqn",
        total_timesteps=int(epochs),
        seed=int(seed),
        out_dir=out_dir / "training" / label,
        hp=deepcopy(hp),
        eval_seeds=(1000, 1001, 1002),
        eval_scenario="small",
        device_name=device,
        model_name=model_name,
        train_scenario="small",
        run_name=run_name,
        enable_tensorboard=False,
        seed_mode=seed_mode,
        action_masking=True,
    )
    elapsed = time.perf_counter() - start
    replay_config = json.loads((model_path.parent / "replay_config.json").read_text(encoding="utf-8"))
    curve = pd.read_csv(model_path.parent / "training_curve.csv")
    return model_path, {
        "mode": label,
        "model_path": str(model_path),
        "training_epochs": int(epochs),
        "training_seed": int(seed),
        "training_seed_mode": seed_mode,
        "wall_clock_training_time_s": elapsed,
        "gradient_updates": int(replay_config["gradient_updates"]),
        "replay_transitions": int(replay_config["size"]),
        "training_simulated_seconds": float(replay_config["training_simulated_seconds"]),
        "completed_or_partial_episodes": int(len(curve)),
        "final_training_episode_reward": float(curve.iloc[-1]["episode_reward"]),
    }


def evaluate_model(
    cfg: Dict[str, Any],
    model_path: Path,
    label: str,
    seeds: Sequence[int],
    device: str,
) -> pd.DataFrame:
    loaded = _load_masked_dqn(model_path, "rainbow_dqn", device, action_masking=True)
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        env = DynamicPatrolEnv(cfg, scenario="small")
        obs, info = _reset_env(env, int(seed))
        done = False
        episode_reward = 0.0
        decisions = 0
        forced_wait = 0
        delta_values: List[float] = []
        discounts: List[float] = []
        start = time.perf_counter()
        while not done:
            mask = np.asarray(info["action_mask"], dtype=bool)
            action = _predict_dqn(loaded, obs, mask)
            forced_wait += int(action == 0 and not mask[1:].any())
            obs, reward, done, info = _step_env(env, action)
            episode_reward += float(reward)
            decisions += 1
            delta_values.append(float(info.get("delta_t", 0.0)))
            discounts.append(float(info.get("transition_discount", 0.0)))
            if decisions > 10000:
                raise RuntimeError("evaluation decision safety limit exceeded")
        elapsed = time.perf_counter() - start
        delta = np.asarray(delta_values, dtype=np.float64)
        metrics = env.sim.metrics(label).to_dict()
        metrics.update(
            {
                "mode": label,
                "seed": int(seed),
                "episode_reward": float(episode_reward),
                "decisions_per_episode": int(decisions),
                "forced_wait_decisions": int(forced_wait),
                "forced_wait_decision_rate": forced_wait / decisions if decisions else 0.0,
                "mean_delta_t": float(delta.mean()) if delta.size else 0.0,
                "median_delta_t": float(np.median(delta)) if delta.size else 0.0,
                "p95_delta_t": float(np.percentile(delta, 95)) if delta.size else 0.0,
                "zero_duration_transition_rate": float(np.mean(delta <= 1e-9)) if delta.size else 0.0,
                "mean_transition_discount": float(np.mean(discounts)) if discounts else 0.0,
                "system_events_per_episode": int(len(env.sim.event_log)),
                "simulated_time": float(env.sim.now),
                "wall_clock_evaluation_time_s": elapsed,
            }
        )
        rows.append(metrics)
        print(
            f"eval {label} seed={seed} reward={episode_reward:.2f} "
            f"completion={metrics['completion_rate']:.3f} timeout={metrics['timeout_rate']:.3f}"
        )
    return pd.DataFrame(rows)


def confidence_summary(fixed: pd.DataFrame, event: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw, broad_summary = paired_statistics(fixed, event)
    summary = broad_summary[broad_summary["metric"].isin(EVALUATION_METRICS)].copy()
    missing = [metric for metric in EVALUATION_METRICS if metric not in set(summary["metric"])]
    if missing:
        merged = fixed.merge(event, on="seed", suffixes=("_fixed60", "_event_smdp"))
        try:
            from scipy import stats
        except Exception:
            stats = None
        extra = []
        for metric in missing:
            left = pd.to_numeric(merged[f"{metric}_fixed60"], errors="coerce")
            right = pd.to_numeric(merged[f"{metric}_event_smdp"], errors="coerce")
            diff = (right - left).dropna()
            n = len(diff)
            mean_diff = float(diff.mean()) if n else math.nan
            sd = float(diff.std(ddof=1)) if n > 1 else math.nan
            critical = float(stats.t.ppf(0.975, n - 1)) if stats is not None and n > 1 else 1.96
            half = critical * sd / math.sqrt(n) if n > 1 else math.nan
            extra.append(
                {
                    "metric": metric,
                    "n": n,
                    "fixed_mean": float(left.mean()),
                    "fixed_sd": float(left.std(ddof=1)),
                    "event_mean": float(right.mean()),
                    "event_sd": float(right.std(ddof=1)),
                    "paired_mean_difference": mean_diff,
                    "relative_change": mean_diff / abs(float(left.mean())) if float(left.mean()) != 0 else math.nan,
                    "ci95_low": mean_diff - half,
                    "ci95_high": mean_diff + half,
                    "paired_t_p": float(stats.ttest_rel(right, left, nan_policy="omit").pvalue) if stats is not None else math.nan,
                    "wilcoxon_p": math.nan,
                }
            )
            raw[f"{metric}_fixed60"] = left
            raw[f"{metric}_event_smdp"] = right
            raw[f"{metric}_paired_difference"] = right - left
        summary = pd.concat([summary, pd.DataFrame(extra)], ignore_index=True)
    return raw, summary


def plot_training_curves(fixed_model: Path, event_model: Path, out_dir: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    frames = []
    colors = {"fixed60": "#4C78A8", "event_smdp": "#E45756"}
    for label, model in (("fixed60", fixed_model), ("event_smdp", event_model)):
        curve = pd.read_csv(model.parent / "training_curve.csv")
        curve["mode"] = label
        curve["smoothed_episode_reward"] = curve["episode_reward"].rolling(15, min_periods=1).mean()
        frames.append(curve)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "training_curves_combined.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for label, group in combined.groupby("mode"):
        ordered = group.sort_values("step")
        ax.plot(ordered["step"], ordered["episode_reward"], color=colors[label], alpha=0.18, linewidth=0.8)
        ax.plot(
            ordered["step"],
            ordered["smoothed_episode_reward"],
            color=colors[label],
            linewidth=2.0,
            label=f"{label} (15-episode mean)",
        )
    ax.set(xlabel="Training epoch (environment transition)", ylabel="Episode return")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "training_reward_curve.png", dpi=240)
    plt.close(fig)
    if "training_simulated_seconds" in combined.columns:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for label, group in combined.groupby("mode"):
            ordered = group.dropna(subset=["training_simulated_seconds"]).sort_values("training_simulated_seconds")
            ax.plot(
                ordered["training_simulated_seconds"] / 3600.0,
                ordered["smoothed_episode_reward"],
                color=colors[label],
                linewidth=2.0,
                label=f"{label} (15-episode mean)",
            )
        ax.set(xlabel="Cumulative simulated time (h)", ylabel="Episode return")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "training_reward_curve_simulated_time.png", dpi=240)
        plt.close(fig)
    return combined


def plot_evaluation_metrics(fixed: pd.DataFrame, event: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    specs = [
        ("completion_rate", "Completion rate"),
        ("timeout_rate", "Timeout rate"),
        ("avg_response_time_s", "Average response (s)"),
        ("weighted_aoi", "Weighted-AoI"),
        ("uav_energy_per_completed_task", "UAV energy / completed task"),
        ("episode_reward", "Episode return"),
    ]
    colors = ["#4C78A8", "#E45756"]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4))
    for ax, (metric, title) in zip(axes.ravel(), specs):
        means = [fixed[metric].mean(), event[metric].mean()]
        errors = [fixed[metric].std(ddof=1), event[metric].std(ddof=1)]
        ax.bar(["Fixed 60 s", "Event-SMDP"], means, yerr=errors, color=colors, capsize=3)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=12)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "trained_model_metric_comparison.png", dpi=240)
    plt.close(fig)


def write_report(
    out_dir: Path,
    epochs: int,
    seed: int,
    eval_seeds: Sequence[int],
    training: pd.DataFrame,
    summary: pd.DataFrame,
    seed_mode: str,
) -> None:
    selected = summary.set_index("metric")
    lines = [
        f"# {epochs}-Epoch Fixed Rainbow vs Event-SMDP Rainbow",
        "",
        f"- Operational epoch: one environment transition and at most one scheduled gradient update.",
        f"- Training epochs: {epochs} per model.",
        f"- Shared training seed: {seed}.",
        f"- Training scenario seed mode: `{seed_mode}` (a newly sampled scenario seed for every episode).",
        f"- Shared evaluation seeds: {len(eval_seeds)} ({min(eval_seeds)}–{max(eval_seeds)}).",
        "- Reward accounting: identical `interval_increment` components and weights in both modes.",
        "- Allowed differences: decision clock and fixed/continuous-time bootstrap discount.",
        "",
        "## Training",
        "",
        "| Mode | Epochs | Gradient updates | Replay transitions | Simulated seconds | Wall-clock (s) | Model |",
        "|---|---:|---:|---:|---:|---:|---|",
        *[
            f"| {row['mode']} | {int(row['training_epochs'])} | {int(row['gradient_updates'])} | "
            f"{int(row['replay_transitions'])} | {row['training_simulated_seconds']:.3f} | "
            f"{row['wall_clock_training_time_s']:.3f} | `{Path(row['model_path']).name}` |"
            for _, row in training.iterrows()
        ],
        "",
        "## Paired evaluation",
        "",
        "| Metric | Fixed mean | Event mean | Relative change | 95% CI of paired difference | p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in EVALUATION_METRICS:
        if metric not in selected.index:
            continue
        row = selected.loc[metric]
        lines.append(
            f"| {metric} | {row['fixed_mean']:.6g} | {row['event_mean']:.6g} | "
            f"{row['relative_change']:.2%} | [{row['ci95_low']:.6g}, {row['ci95_high']:.6g}] | "
            f"{row['paired_t_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "The comparison uses a single trained model per mode. The paired seed statistics quantify evaluation variability, not training-seed variability.",
        ]
    )
    (out_dir / "TRAINED_MODEL_COMPARISON_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def overwrite_previous_models(
    out_dir: Path,
    fixed_model: Path,
    event_model: Path,
    epochs: int,
    seed: int,
    seed_mode: str,
) -> Dict[str, str]:
    targets = {
        "fixed60": out_dir / "training" / "fixed60" / "rainbow_fixed60_5000epochs_seed42" / "Rainbow_fixed60_5000epochs_seed42.pt",
        "event_smdp": out_dir / "training" / "event_smdp" / "rainbow_event_smdp_5000epochs_seed42" / "Rainbow_event_smdp_5000epochs_seed42.pt",
    }
    sources = {"fixed60": fixed_model, "event_smdp": event_model}
    deployed: Dict[str, str] = {}
    for label, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sources[label], target)
        notice = {
            "replaced_model_path": str(target),
            "source_model_path": str(sources[label]),
            "actual_training_epochs": int(epochs),
            "training_seed": int(seed),
            "training_seed_mode": seed_mode,
        }
        (target.parent / "MODEL_REPLACEMENT_MANIFEST.json").write_text(
            json.dumps(notice, indent=2), encoding="utf-8"
        )
        deployed[label] = str(target)
    return deployed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare fixed/event Rainbow with matched settings")
    parser.add_argument("--fixed-config", default="configs/scenario_60node_stage2_fixed_dynamic.yaml")
    parser.add_argument("--event-config", default="configs/scenario_60node_stage2_event_dynamic.yaml")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seeds", default="1000:1020")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=1000)
    parser.add_argument("--seed-mode", choices=("fixed", "random", "curriculum"), default=None)
    parser.add_argument("--overwrite-previous-models", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="results/stage2_rainbow_5000epoch_comparison")
    args = parser.parse_args()

    root = project_root()
    fixed_path = root / args.fixed_config
    event_path = root / args.event_config
    fixed_cfg = load_config(fixed_path)
    event_cfg = load_config(event_path)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    hp = make_shared_hyperparams(fixed_cfg, args)
    seed_mode = str(args.seed_mode or fixed_cfg.get("training", {}).get("seed_mode", "fixed"))
    manifest = assert_fair_configuration(fixed_cfg, event_cfg, hp)
    manifest["training_epochs"] = int(args.epochs)
    manifest["training_seed"] = int(args.seed)
    manifest["training_seed_mode"] = seed_mode
    manifest["fixed_config"] = str(fixed_path)
    manifest["event_config"] = str(event_path)
    (out_dir / "fairness_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    fixed_model, fixed_training = train_one(
        fixed_cfg, fixed_path, "fixed60", args.epochs, args.seed, hp, out_dir, args.device, seed_mode
    )
    event_model, event_training = train_one(
        event_cfg, event_path, "event_smdp", args.epochs, args.seed, hp, out_dir, args.device, seed_mode
    )
    training_df = pd.DataFrame([fixed_training, event_training])
    training_df.to_csv(out_dir / "training_summary.csv", index=False)
    plot_training_curves(fixed_model, event_model, out_dir)

    eval_seeds = parse_seeds(args.eval_seeds)
    fixed_eval = evaluate_model(fixed_cfg, fixed_model, "fixed60", eval_seeds, args.device)
    event_eval = evaluate_model(event_cfg, event_model, "event_smdp", eval_seeds, args.device)
    fixed_eval.to_csv(out_dir / "fixed60_model_evaluation.csv", index=False)
    event_eval.to_csv(out_dir / "event_smdp_model_evaluation.csv", index=False)
    raw, summary = confidence_summary(fixed_eval, event_eval)
    raw.to_csv(out_dir / "paired_model_comparison.csv", index=False)
    summary.to_csv(out_dir / "paired_model_statistical_summary.csv", index=False)
    plot_evaluation_metrics(fixed_eval, event_eval, out_dir)
    write_report(out_dir, args.epochs, args.seed, eval_seeds, training_df, summary, seed_mode)
    deployed_models = (
        overwrite_previous_models(out_dir, fixed_model, event_model, args.epochs, args.seed, seed_mode)
        if args.overwrite_previous_models
        else {}
    )
    summary_json = {
        "epochs_per_model": int(args.epochs),
        "training_seed": int(args.seed),
        "training_seed_mode": seed_mode,
        "evaluation_seed_count": len(eval_seeds),
        "fixed_model": str(fixed_model),
        "event_model": str(event_model),
        "deployed_over_previous_models": deployed_models,
        "fairness_check_passed": True,
        "reward_configuration_identical": True,
        "shared_hyperparameters": asdict(hp),
        "metrics": summary.set_index("metric").to_dict(orient="index"),
    }
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(
            summary_json,
            indent=2,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        ),
        encoding="utf-8",
    )
    print(f"Saved {args.epochs}-epoch trained-model comparison to {out_dir}")


if __name__ == "__main__":
    main()
