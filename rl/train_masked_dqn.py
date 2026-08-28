from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - clear runtime message.
    raise RuntimeError("Masked DQN training requires PyTorch. Install torch to use this module.") from exc

from dynamic_patrol_marl.rl.dqn_networks import make_q_network, masked_q_values, reset_noisy_layers
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.rl.paths import resolve_dqn_output_dirs
from dynamic_patrol_marl.rl.replay_buffer import MaskedReplayBuffer, PrioritizedMaskedReplayBuffer
from dynamic_patrol_marl.rl.smdp import (
    NStepSMDPAccumulator,
    NStepTransition,
    one_step_smdp_target,
    resolve_discount_settings,
    stage5_td_error_delta_t_bin,
)
from dynamic_patrol_marl.rl.seed_scheduler import (
    CurriculumSeedScheduler,
    SEED_MODES,
    build_training_stages,
    parse_int_list,
)
from dynamic_patrol_marl.utils.config import load_config, project_root


ALGORITHMS = ("dqn", "double_dqn", "dueling_double_dqn", "rainbow_dqn")
STAGE5_EVAL_METRICS = (
    "high_priority_response_time_s",
    "total_energy",
    "total_hover_energy",
    "uav_energy_per_completed_task",
    "avg_tasks_per_sortie",
    "uav_return_distance_m",
    "cav_travel_distance_m",
    "drone_wait_time_s",
    "vehicle_wait_time_s",
    "mean_uav_recovery_wait_s",
    "queue_occupancy_time_s",
    "route_change_count",
    "route_reversal_count",
    "emergency_override_count",
    "safe_plan_rate",
    "safe_only_decision_rate",
    "type_I_selected_rate",
    "type_II_selected_rate",
    "type_III_selected_rate",
    "joint_keep_route_rate",
    "joint_wait_rate",
    "joint_move_to_rate",
    "joint_reroute_rate",
    "keep_route_count",
    "wait_count",
    "move_to_count",
    "reroute_count",
    "commit_failure_count",
    "path_planning_failure_count",
    "rollback_count",
    "infeasible_recoveries",
    "minimum_battery_observed",
    "mean_plan_generation_time_ms",
)

STAGE5_EPISODE_COLUMNS = (
    "task_release_horizon_s", "episode_end_time_s", "terminated", "truncated",
    "released_task_count", "completed_task_count_final", "timed_out_task_count_final",
    "pending_tasks_at_cutoff", "backlog_rate_at_cutoff", "completed_during_drain",
    "timed_out_during_drain", "drain_duration_s", "censored_task_count",
    "censoring_rate", "completion_rate_final", "timeout_rate_final",
    "operation_reward", "drain_reward", "episode_reward_total", "illegal_actions",
    "reward_per_released_task", "reward_per_operation_hour",
    "physical_episode_reward", "smdp_aggregated_episode_reward",
    "physical_event_count", "atomic_event_transition_count", "policy_decision_count",
    "replay_transition_count", "single_valid_action_count", "multi_valid_action_count",
    "average_valid_action_count", "automatic_action_count",
    "initial_automatic_action_count", "initial_automatic_reward",
    "average_smdp_duration", "maximum_smdp_duration", "non_safe_action_count",
)

SMDP_EPISODE_COLUMNS = (
    "delta_t_mean", "delta_t_min", "delta_t_max",
    "effective_gamma_mean", "effective_gamma_min", "effective_gamma_max",
)

REWARD_FREEZING_COMPONENT_COLUMNS = (
    "completion_reward", "priority_reward", "timeout_penalty",
    "weighted_aoi_penalty", "flight_energy_penalty", "hover_energy_penalty",
    "uav_return_distance_penalty", "cav_distance_penalty",
    "recovery_wait_penalty", "queue_occupancy_penalty",
    "route_change_penalty", "route_reversal_penalty", "failure_penalty",
    "episode_total_reward", "reward_conservation_error",
)

EVAL_COLUMNS = [
    "step",
    "algo",
    "seed",
    "completion_rate",
    "timeout_rate",
    "avg_response_time",
    "weighted_aoi",
    "energy",
    "uav_waiting_time",
    "cav_waiting_time",
    "episode_reward",
] + list(STAGE5_EVAL_METRICS) + list(STAGE5_EPISODE_COLUMNS) + list(SMDP_EPISODE_COLUMNS) + list(REWARD_FREEZING_COMPONENT_COLUMNS) + ["action_entropy"]

STAGE5_DT_BINS = (
    "dt_eq_0",
    "dt_0_10",
    "dt_10_30",
    "dt_30_60",
    "dt_60_120",
    "dt_over_120",
)


def reward_config_metadata(cfg: Dict) -> Dict:
    reward_cfg = dict(cfg.get("reward", {}) or {})
    canonical = json.dumps(reward_cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    normalization = dict(reward_cfg.get("normalization", {}) or {})
    return {
        "config": reward_cfg,
        "signature": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "mode": str(reward_cfg.get("mode", "legacy_interval")),
        "scale_file": str(normalization.get("scale_file", "")),
        "scale_version": str(normalization.get("scale_version", "")),
    }


def _action_entropy(metrics: Dict) -> float:
    probabilities = np.asarray(
        [
            metrics.get("safe_plan_rate", 0.0),
            metrics.get("type_I_selected_rate", 0.0),
            metrics.get("type_II_selected_rate", 0.0),
            metrics.get("type_III_selected_rate", 0.0),
        ],
        dtype=float,
    )
    probabilities = probabilities[np.isfinite(probabilities) & (probabilities > 0.0)]
    return float(-np.sum(probabilities * np.log(probabilities))) if probabilities.size else 0.0


def _safe_correlation(left: List[float], right: List[float], *, ranked: bool = False) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    x = pd.Series(left, dtype=float)
    y = pd.Series(right, dtype=float)
    if ranked:
        x, y = x.rank(method="average"), y.rank(method="average")
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan")
    return float(x.corr(y))


def stage5_event_categories(event_types: str, *, instantaneous_dispatch: bool = False) -> List[str]:
    categories: List[str] = []
    for event_type in str(event_types).split("|"):
        event_type = event_type.strip().upper()
        if not event_type:
            continue
        if event_type == "TASK_RELEASE":
            categories.append("task_release")
        elif event_type == "TASK_SERVICE_COMPLETE":
            categories.append("task_completion")
        elif event_type == "TASK_TIMEOUT":
            categories.append("task_timeout")
        elif event_type == "UAV_LAUNCH":
            categories.append("uav_launch")
        elif event_type in {"UAV_RECOVERED", "CAV_RECOVERY_COMPLETE"}:
            categories.append("uav_recovery")
        elif event_type in {"UAV_RECOVERY_ARRIVAL", "RECOVERY_QUEUE_UPDATED"}:
            categories.append("queue_change")
        elif event_type.startswith("CAV_") and "ARRIVAL" in event_type:
            categories.append("cav_arrival")
        elif event_type == "EPISODE_END":
            categories.append("episode_terminal")
    if instantaneous_dispatch:
        categories.append("instantaneous_multi_dispatch")
    return sorted(set(categories)) or ["other"]


def validate_resume_checkpoint(
    checkpoint: Dict,
    *,
    expected_obs_dim: int,
    expected_action_dim: int,
    expected_candidate_dim: int,
    expected_decision_mode: str,
    current_reward_metadata: Dict,
) -> None:
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("unsupported resume checkpoint")
    expected = (
        int(expected_obs_dim),
        int(expected_action_dim),
        int(expected_candidate_dim),
        str(expected_decision_mode),
    )
    actual = (
        int(checkpoint.get("obs_dim", -1)),
        int(checkpoint.get("action_dim", -1)),
        int(checkpoint.get("candidate_dim", -1)),
        str(checkpoint.get("decision_mode", "")),
    )
    if actual != expected:
        raise ValueError(f"resume checkpoint/environment mismatch: checkpoint={actual}, environment={expected}")
    saved_reward = dict(checkpoint.get("reward_metadata", {}) or {})
    if not saved_reward.get("signature"):
        raise ValueError("resume checkpoint has no reward configuration signature")
    if saved_reward["signature"] != current_reward_metadata["signature"]:
        raise ValueError(
            "resume checkpoint reward configuration mismatch: "
            f"checkpoint={saved_reward['signature']} current={current_reward_metadata['signature']}"
        )


@dataclass
class DQNHyperParams:
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 128
    replay_buffer_size: int = 100000
    learning_starts: int = 5000
    train_frequency: int = 1
    target_update_interval: int = 2000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 100000
    max_grad_norm: float = 10.0
    eval_interval: int = 20000
    candidate_dim: int = 6
    prioritized_replay_alpha: float = 0.6
    prioritized_replay_beta_start: float = 0.4
    prioritized_replay_beta_final: float = 1.0
    prioritized_replay_beta_steps: int = 200000
    priority_epsilon: float = 1e-6
    checkpoint_interval: int = 0
    diagnostic_sample_stride: int = 1


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def reset_env(env: DynamicPatrolEnv, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        mask = info.get("action_mask", env.action_masks())
    else:
        obs = out
        mask = env.action_masks()
    return np.asarray(obs, dtype=np.float32), np.asarray(mask, dtype=bool)


def step_env(env: DynamicPatrolEnv, action: int) -> Tuple[np.ndarray, float, bool, np.ndarray, Dict]:
    out = env.step(int(action))
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        info = dict(info)
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
    else:
        obs, reward, done, info = out
    mask = info.get("action_mask", env.action_masks())
    info = dict(info)
    info.setdefault("terminated", bool(done))
    info.setdefault("truncated", False)
    return np.asarray(obs, dtype=np.float32), float(reward), bool(done), np.asarray(mask, dtype=bool), info


def epsilon_by_step(step: int, hp: DQNHyperParams) -> float:
    frac = min(1.0, max(0.0, float(step) / float(max(1, hp.epsilon_decay_steps))))
    return hp.epsilon_start + frac * (hp.epsilon_final - hp.epsilon_start)


def exploration_epsilon(algo: str, step: int, hp: DQNHyperParams) -> float:
    """Return the epsilon used to collect a training transition.

    Rainbow normally relies on NoisyNet after learning starts, but an
    untrained candidate-aware network can otherwise fill its entire initial
    replay window with the no-launch action for an unlucky initialization.
    Uniform valid-action replay warm-up makes ``learning_starts`` a real
    exploration boundary and prevents that absorbing failure mode.
    """

    if algo == "rainbow_dqn":
        return 1.0 if int(step) <= int(hp.learning_starts) else 0.0
    return epsilon_by_step(step, hp)


def prioritized_beta_by_step(step: int, hp: DQNHyperParams) -> float:
    frac = min(1.0, max(0.0, float(step) / float(max(1, hp.prioritized_replay_beta_steps))))
    return hp.prioritized_replay_beta_start + frac * (hp.prioritized_replay_beta_final - hp.prioritized_replay_beta_start)


def sample_valid_action(action_mask: np.ndarray, rng: np.random.Generator) -> int:
    valid = np.flatnonzero(np.asarray(action_mask, dtype=bool))
    if len(valid) == 0:
        return 0
    return int(rng.choice(valid))


def policy_action_mask(action_mask: np.ndarray, action_masking: bool) -> np.ndarray:
    """Return the mask used by the policy.

    In unmasked experiments every discrete action is available to the policy;
    infeasible actions are still handled by the environment as illegal actions.
    """
    mask = np.asarray(action_mask, dtype=bool)
    return mask if action_masking else np.ones_like(mask, dtype=bool)


@torch.no_grad()
def select_action(
    q_network: nn.Module,
    obs: np.ndarray,
    action_mask: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    exploring = rng.random() < float(epsilon)
    if exploring:
        return sample_valid_action(action_mask, rng)
    obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
    mask_t = torch.as_tensor(action_mask[None, :], device=device, dtype=torch.bool)
    q_values = q_network(obs_t, mask_t)
    return int(masked_q_values(q_values, mask_t).argmax(dim=1).item())


def compute_dqn_loss(
    algo: str,
    online_net: nn.Module,
    target_net: nn.Module,
    batch,
    gamma: float,
    weights: torch.Tensor | None = None,
    action_masking: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action_masks = batch.action_masks if action_masking else torch.ones_like(batch.action_masks, dtype=torch.bool)
    next_action_masks = batch.next_action_masks if action_masking else torch.ones_like(batch.next_action_masks, dtype=torch.bool)

    q_values = online_net(batch.obs, action_masks)
    current_q = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if algo == "dqn":
            target_next_q = target_net(batch.next_obs, next_action_masks)
            next_q = masked_q_values(target_next_q, next_action_masks).max(dim=1).values
        else:
            online_next_q = online_net(batch.next_obs, next_action_masks)
            next_actions = masked_q_values(online_next_q, next_action_masks).argmax(dim=1)
            target_next_q = target_net(batch.next_obs, next_action_masks)
            next_q = target_next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        discounts = getattr(batch, "bootstrap_discounts", None)
        if discounts is None:
            discounts = torch.full_like(batch.rewards, float(gamma))
        # The experiment contract uses the complete done mask: neither a
        # natural terminal nor a time-limit terminal bootstraps from next_q.
        target_q = one_step_smdp_target(batch.rewards, discounts, batch.dones, next_q)
    per_sample_loss = nn.functional.smooth_l1_loss(current_q, target_q, reduction="none")
    if weights is not None:
        per_sample_loss = per_sample_loss * weights
    td_errors = current_q - target_q
    # Keep the sampled online-Q values for diagnostics.  This avoids a second
    # network forward pass solely for experiment reporting.
    return per_sample_loss.mean(), td_errors, current_q.detach()


def materialize_q_range(
    minimum_q_value: torch.Tensor | None,
    maximum_q_value: torch.Tensor | None,
) -> tuple[float, float]:
    """Transfer the accumulated Q range to the host with one CUDA sync."""

    if minimum_q_value is None or maximum_q_value is None:
        return float("nan"), float("nan")
    values = torch.stack((minimum_q_value, maximum_q_value)).detach().cpu().numpy()
    return float(values[0]), float(values[1])


def metric_row(step: int, algo: str, seed: int, metrics: Dict, episode_reward: float) -> Dict:
    row = {
        "step": int(step),
        "algo": algo,
        "seed": int(seed),
        "completion_rate": metrics.get("completion_rate_final", metrics.get("completion_rate", np.nan)),
        "timeout_rate": metrics.get("timeout_rate_final", metrics.get("timeout_rate", np.nan)),
        "avg_response_time": metrics.get("avg_response_time_s", np.nan),
        "weighted_aoi": metrics.get("weighted_aoi", np.nan),
        "energy": metrics.get("total_energy", np.nan),
        "uav_waiting_time": metrics.get("drone_wait_time_s", np.nan),
        "cav_waiting_time": metrics.get("vehicle_wait_time_s", np.nan),
        "episode_reward": float(episode_reward),
    }
    row.update({key: metrics.get(key, np.nan) for key in STAGE5_EVAL_METRICS})
    row.update({key: metrics.get(key, np.nan) for key in STAGE5_EPISODE_COLUMNS})
    row.update({key: metrics.get(key, np.nan) for key in SMDP_EPISODE_COLUMNS})
    row.update({key: metrics.get(key, np.nan) for key in REWARD_FREEZING_COMPONENT_COLUMNS})
    row["action_entropy"] = _action_entropy(metrics)
    return row


def train_metric_row(
    step: int,
    episode: int,
    episode_steps: int,
    episode_reward: float,
    epsilon: float,
    latest_loss: float,
    metrics: Dict,
    partial_episode: bool = False,
    seed_mode: str = "",
    stage: int | None = None,
    stage_name: str = "",
    q_value_min: float = np.nan,
    q_value_max: float = np.nan,
    q_value_mean: float = np.nan,
    td_error_mean: float = np.nan,
    td_error_max: float = np.nan,
) -> Dict:
    row = {
        "step": int(step),
        "episode": int(episode),
        "episode_steps": int(episode_steps),
        "episode_reward": float(episode_reward),
        "epsilon": float(epsilon),
        "loss": float(latest_loss) if np.isfinite(latest_loss) else 0.0,
        "loss_available": bool(np.isfinite(latest_loss)),
        "completion_rate": metrics.get("completion_rate_final", metrics.get("completion_rate", np.nan)),
        "timeout_rate": metrics.get("timeout_rate_final", metrics.get("timeout_rate", np.nan)),
        "illegal_actions": metrics.get("illegal_actions", np.nan),
        "partial_episode": bool(partial_episode),
        "seed_mode": seed_mode,
        "stage": np.nan if stage is None else int(stage),
        "stage_name": stage_name,
        "q_value_min": float(q_value_min),
        "q_value_max": float(q_value_max),
        "q_value_mean": float(q_value_mean),
        "td_error_mean": float(td_error_mean),
        "td_error_max": float(td_error_max),
    }
    row.update({key: metrics.get(key, np.nan) for key in STAGE5_EVAL_METRICS})
    row.update({key: metrics.get(key, np.nan) for key in STAGE5_EPISODE_COLUMNS})
    row.update({key: metrics.get(key, np.nan) for key in SMDP_EPISODE_COLUMNS})
    row.update({key: metrics.get(key, np.nan) for key in REWARD_FREEZING_COMPONENT_COLUMNS})
    row["action_entropy"] = _action_entropy(metrics)
    return row


def write_train_scalars(writer, row: Dict, step: int) -> None:
    writer.add_scalar("train/episode_reward", float(row["episode_reward"]), step)
    writer.add_scalar("train/completion_rate", float(row["completion_rate"]), step)
    writer.add_scalar("train/timeout_rate", float(row["timeout_rate"]), step)
    for name in (
        "reward_per_released_task", "reward_per_operation_hour", "backlog_rate_at_cutoff",
        "drain_duration_s", "censoring_rate", "safe_plan_rate",
    ):
        value = row.get(name, np.nan)
        if np.isfinite(value):
            writer.add_scalar(f"train/{name}", float(value), step)
    writer.add_scalar("train/illegal_actions", float(row["illegal_actions"]), step)
    writer.add_scalar("train/episode_steps", int(row["episode_steps"]), step)
    writer.add_scalar("train/epsilon", float(row["epsilon"]), step)
    if np.isfinite(row.get("stage", np.nan)):
        writer.add_scalar("train/curriculum_stage", float(row["stage"]), step)
    if np.isfinite(row.get("loss", np.nan)):
        writer.add_scalar("train/loss", float(row["loss"]), step)
    for name in (
        "delta_t_mean", "delta_t_min", "delta_t_max",
        "effective_gamma_mean", "effective_gamma_min", "effective_gamma_max",
        "td_error_mean", "td_error_max",
        "q_value_mean", "q_value_min", "q_value_max",
    ):
        value = row.get(name, np.nan)
        if np.isfinite(value):
            writer.add_scalar(f"train/{name}", float(value), step)


@torch.no_grad()
def evaluate_policy(
    cfg: Dict,
    q_network: nn.Module,
    device: torch.device,
    algo: str,
    step: int,
    eval_seeds: Iterable[int],
    scenario: str,
    action_masking: bool = True,
    transition_audit_rows: List[Dict] | None = None,
    route_event_rows: List[Dict] | None = None,
    task_audit_rows: List[Dict] | None = None,
) -> pd.DataFrame:
    q_network.eval()
    rows: List[Dict] = []
    rng = np.random.default_rng(0)
    method_name = algo if action_masking else f"unmasked_{algo}"
    for seed in eval_seeds:
        env = DynamicPatrolEnv(cfg, scenario=scenario)
        obs, mask = reset_env(env, seed=int(seed))
        done = False
        episode_reward = 0.0
        while not done:
            action = select_action(
                q_network,
                obs,
                policy_action_mask(mask, action_masking),
                epsilon=0.0,
                rng=rng,
                device=device,
            )
            obs, reward, done, mask, _ = step_env(env, action)
            episode_reward += reward
        rows.append(metric_row(step, algo, int(seed), env.episode_summary(), episode_reward))
        if transition_audit_rows is not None:
            transition_audit_rows.extend(
                {
                    "evaluation_step": int(step),
                    "eval_seed": int(seed),
                    **dict(row),
                }
                for row in env.transition_log
            )
        if route_event_rows is not None:
            route_event_rows.extend(
                {
                    "evaluation_step": int(step),
                    "eval_seed": int(seed),
                    **dict(row),
                }
                for row in getattr(env.sim.ground_controller, "route_change_log", [])
            )
        if task_audit_rows is not None:
            task_audit_rows.extend(
                {
                    "evaluation_step": int(step),
                    "eval_seed": int(seed),
                    "task_id": int(task.id),
                    "priority": float(task.priority),
                    "release_time_s": float(task.release_time),
                    "completion_time_s": float(task.completion_time),
                    "end_to_end_response_time_s": float(task.completion_time - task.release_time),
                }
                for task in env.sim.tasks
                if task.status == "done" and task.completion_time is not None
            )
    q_network.train()
    return pd.DataFrame(rows, columns=EVAL_COLUMNS)


def save_outputs(
    out_dir: Path,
    run_dir: Path,
    algo: str,
    seed: int,
    cfg_path: str,
    hp: DQNHyperParams,
    online_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs_dim: int,
    action_dim: int,
    training_rows: List[Dict],
    eval_rows: List[Dict],
    total_timesteps: int,
    model_name: str,
    train_scenario: str,
    tensorboard_log_dir: Path | None,
    seed_mode: str,
    stages: List[Dict],
    action_masking: bool,
    replay_config: Dict | None = None,
    feature_schema: Dict | None = None,
    decision_mode: str = "sortie_only",
    reward_metadata: Dict | None = None,
    resume_from: str = "",
    wall_clock_time_s: float = 0.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / model_name
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algo": algo,
            "seed": int(seed),
            "config_path": cfg_path,
            "train_scenario": train_scenario,
            "run_dir": str(run_dir),
            "tensorboard_log_dir": str(tensorboard_log_dir) if tensorboard_log_dir is not None else "",
            "seed_mode": seed_mode,
            "action_masking": bool(action_masking),
            "mask_mode": "masked" if action_masking else "unmasked",
            "training_stages": stages,
            "hyperparams": asdict(hp),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "candidate_dim": int(hp.candidate_dim),
            "replay_config": dict(replay_config or {}),
            "decision_mode": str(decision_mode),
            "feature_schema": dict(feature_schema or {}),
            "reward_metadata": dict(reward_metadata or {}),
            "resume_from": str(resume_from),
            "model_state_dict": online_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        model_path,
    )
    pd.DataFrame(training_rows).to_csv(run_dir / "training_curve.csv", index=False)
    pd.DataFrame(eval_rows, columns=EVAL_COLUMNS).to_csv(run_dir / "eval_curve.csv", index=False)

    eval_df = pd.DataFrame(eval_rows)
    final_eval = eval_df[eval_df["step"] == eval_df["step"].max()] if not eval_df.empty else pd.DataFrame()
    summary = {
        "algo": algo,
        "seed": int(seed),
        "run_name": run_dir.name,
        "timesteps": int(total_timesteps),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "model_path": str(model_path),
        "run_dir": str(run_dir),
        "tensorboard_log_dir": str(tensorboard_log_dir) if tensorboard_log_dir is not None else "",
        "seed_mode": seed_mode,
        "action_masking": bool(action_masking),
        "mask_mode": "masked" if action_masking else "unmasked",
        "training_stages": json.dumps(stages),
        "replay_schema_version": int((replay_config or {}).get("schema_version", 1)),
        "time_mode": str((replay_config or {}).get("time_mode", "fixed_interval")),
        "decision_mode": str(decision_mode),
        "feature_schema_version": int((feature_schema or {}).get("schema_version", 0)),
        "reward_mode": str((reward_metadata or {}).get("mode", "")),
        "reward_config_signature": str((reward_metadata or {}).get("signature", "")),
        "reward_scale_file": str((reward_metadata or {}).get("scale_file", "")),
        "reward_scale_version": str((reward_metadata or {}).get("scale_version", "")),
        "resume_from": str(resume_from),
        "wall_clock_time_s": float(wall_clock_time_s),
        **asdict(hp),
    }
    if not final_eval.empty:
        for col in EVAL_COLUMNS:
            if col not in {"step", "algo", "seed"}:
                summary[f"final_{col}"] = float(pd.to_numeric(final_eval[col], errors="coerce").mean())
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)
    summary_path = out_dir / "summary.csv"
    summary_df = pd.DataFrame([summary])
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        if "run_name" in existing.columns:
            existing = existing[existing["run_name"] != run_dir.name]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    with open(run_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def save_intermediate_checkpoint(
    *,
    run_dir: Path,
    step: int,
    algo: str,
    seed: int,
    online_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs_dim: int,
    action_dim: int,
    candidate_dim: int,
    decision_mode: str,
    reward_metadata: Dict,
    feature_schema: Dict,
    hyperparams: DQNHyperParams,
) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_step_{int(step)}.pt"
    torch.save(
        {
            "algo": algo,
            "seed": int(seed),
            "timesteps": int(step),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "candidate_dim": int(candidate_dim),
            "decision_mode": str(decision_mode),
            "feature_schema": dict(feature_schema),
            "reward_metadata": dict(reward_metadata),
            "hyperparams": asdict(hyperparams),
            "model_state_dict": online_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    return path


def train_masked_dqn(
    cfg: Dict,
    cfg_path: str,
    algo: str,
    total_timesteps: int | None,
    seed: int | None,
    out_dir: Path,
    hp: DQNHyperParams | None = None,
    eval_seeds: Iterable[int] = (1000, 1001, 1002, 1003, 1004),
    eval_scenario: str = "eval",
    device_name: str = "auto",
    model_name: str | None = "model.pt",
    train_scenario: str = "small",
    run_name: str | None = None,
    tensorboard_dir: Path | None = None,
    enable_tensorboard: bool = True,
    seed_mode: str | None = None,
    stage_timesteps: List[int] | None = None,
    action_masking: bool = True,
    resume_from: Path | None = None,
    max_episodes: int | None = None,
    timing_out_dir: Path | None = None,
    reward_audit_out_dir: Path | None = None,
) -> Path:
    if algo not in ALGORITHMS:
        raise ValueError(f"--algo must be one of {ALGORITHMS}")
    hp = hp or DQNHyperParams()
    resolved_seed_mode, base_seed, stages = build_training_stages(
        cfg=cfg,
        seed_mode=seed_mode,
        seed=seed,
        total_timesteps=total_timesteps,
        stage_timesteps=stage_timesteps,
        default_timesteps=200000,
    )
    effective_timesteps = int(sum(stage.timesteps for stage in stages))
    set_global_seed(base_seed)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    rng = np.random.default_rng(int(base_seed))
    run_prefix = algo if action_masking else f"unmasked_{algo}"
    run_name = run_name or f"{run_prefix}_{resolved_seed_mode}_seed{base_seed}"
    model_name = Path(model_name).name if model_name else f"{run_name}.pt"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"
    progress_started_at = perf_counter()
    gradient_updates = 0

    def write_progress(step_value: int, status: str) -> None:
        progress_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "step": int(step_value),
                    "total_timesteps": int(effective_timesteps),
                    "progress_fraction": float(step_value / max(1, effective_timesteps)),
                    "training_seed": int(base_seed),
                    "gradient_updates": int(gradient_updates),
                    "elapsed_wall_time_s": float(perf_counter() - progress_started_at),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    write_progress(0, "initializing")
    tensorboard_log_dir = tensorboard_dir / run_name if tensorboard_dir is not None else out_dir / "tensorboard" / run_name
    writer = None
    if enable_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
        except Exception as exc:
            raise RuntimeError(
                "TensorBoard logging is enabled but unavailable. Install tensorboard "
                "or rerun with --disable-tensorboard."
            ) from exc
    else:
        tensorboard_log_dir = None

    scheduler = CurriculumSeedScheduler(stages, rng_seed=base_seed)
    env = DynamicPatrolEnv(cfg, scenario=train_scenario, seed_scheduler=scheduler)
    if max_episodes == 1:
        print("ENV_CREATED", flush=True)
    timing_enabled = max_episodes is not None or timing_out_dir is not None
    phase_names = [
        "environment_reset_time", "observation_building_time", "task_update_time",
        "candidate_generation_time", "candidate_feature_time", "uav_path_planning_time",
        "selected_sortie_materialization_time", "voxel_astar_search_time",
        "action_mask_time", "policy_inference_time", "environment_step_time",
        "reward_calculation_time", "replay_buffer_time", "network_training_time",
        "target_network_update_time", "logging_time", "checkpoint_time",
    ]
    phase_totals = {name: 0.0 for name in phase_names}
    phase_calls = {name: 0 for name in phase_names}
    episode_timing_rows: list[dict] = []
    current_episode_timing = {name: 0.0 for name in phase_names}
    current_episode_calls = {name: 0 for name in phase_names}
    current_planner = {"calls": 0, "direct": 0, "local_detour": 0, "failed": 0, "full_time_s": 0.0, "edge_time_s": 0.0, "astar_time_s": 0.0}
    simple_counts = {
        "safe_plan_count": 0, "selected_uav_plan_count": 0,
        "materialization_failure_count": 0, "voxel_astar_call_count": 0,
        "voxel_astar_timeout_count": 0,
    }
    # Per-attempt diagnostics are deliberately collected at the existing
    # commit-time materialization boundary; candidate generation remains
    # estimate-only.  ``selected_uav_action_count`` means calls entering
    # materialize_selected_sortie, while ``committed_uav_sortie_count`` means
    # the returned precise sortie was subsequently accepted by commit logic.
    materialization_attempt_rows: List[Dict[str, Any]] = []
    astar_segment_rows: List[Dict[str, Any]] = []
    planner_original = None

    def add_phase(name: str, elapsed: float) -> None:
        if not timing_enabled:
            return
        phase_totals[name] += float(elapsed)
        phase_calls[name] += 1
        current_episode_timing[name] += float(elapsed)
        current_episode_calls[name] += 1

    def wrap_method(obj, attr: str, phase: str) -> None:
        original = getattr(obj, attr)
        def timed(*args, **kwargs):
            started = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                add_phase(phase, perf_counter() - started)
        setattr(obj, attr, timed)

    if timing_enabled:
        wrap_method(env, "_observation", "observation_building_time")
        wrap_method(env, "action_masks", "action_mask_time")
        wrap_method(env, "_reward", "reward_calculation_time")
        wrap_method(env, "_event_reward_components", "reward_calculation_time")
    def install_sim_timing_wrappers() -> None:
        if not timing_enabled or env.sim is None:
            return
        wrap_method(env.sim, "get_candidates", "candidate_generation_time")
        if env.sim.joint_plan_generator is not None:
            wrap_method(env.sim.joint_plan_generator, "generate_joint_plans", "candidate_generation_time")
        if env.sim.task_pool is not None:
            wrap_method(env.sim.task_pool, "update_release_and_expiry", "task_update_time")
        if env.sim.sortie_generator is not None:
            original_materialize = env.sim.sortie_generator.materialize_selected_sortie
            def timed_materialize(*args, **kwargs):
                started = perf_counter()
                sortie_arg = args[0] if args else kwargs.get("sortie")
                now_arg = float(args[2] if len(args) > 2 else kwargs.get("now", getattr(env.sim, "now", 0.0)))
                attempt_id = f"episode_{episode_idx}_materialization_{len(materialization_attempt_rows)+1:04d}"
                try:
                    result = original_materialize(*args, **kwargs)
                    simple_counts["selected_uav_plan_count"] += int(result is not None)
                    simple_counts["materialization_failure_count"] += int(result is None)
                    failure = str(getattr(env.sim.sortie_generator, "last_materialization", {}).get("failure_reason", ""))
                    simple_counts["voxel_astar_timeout_count"] += int("timeout" in failure)
                    meta = dict(getattr(env.sim.sortie_generator, "last_materialization", {}) or {})
                    details = list(meta.get("segment_results", []) or [])
                    required = len([s for s in (getattr(sortie_arg, "segments", None) or []) if str(s.get("type")) == "horizontal_flight"])
                    success_segments = sum(bool(d.get("astar_success", d.get("cache_hit", False))) for d in details)
                    failed_segments = len(details) - success_segments
                    failure_stage = meta.get("failure_stage")
                    if result is None and not failure_stage:
                        failure_stage = "during_astar" if details else "before_astar"
                    task_ids = list(getattr(sortie_arg, "task_ids", []) or [])
                    attempt_row = {
                        "episode_id": f"episode_{episode_idx}", "decision_epoch_id": len(materialization_attempt_rows),
                        "simulation_time": now_arg, "action_slot": None, "joint_plan_id": None,
                        "materialization_attempt_id": attempt_id, "uav_id": getattr(sortie_arg, "drone_id", None),
                        "task_count": len(task_ids), "task_ids": ";".join(map(str, task_ids)),
                        "task_types": None, "task_priorities": None, "task_deadlines": None,
                        "launch_node": getattr(sortie_arg, "launch_node", None), "recovery_node": getattr(sortie_arg, "recovery_node", None),
                        "expected_distance_before_astar_m": sum(float(s.get("distance_m", 0.0)) for s in (getattr(sortie_arg, "segments", None) or []) if s.get("type") == "horizontal_flight"),
                        "expected_duration_before_astar_s": getattr(sortie_arg, "drone_time_s", None), "expected_energy_before_astar": getattr(sortie_arg, "energy", None),
                        "expected_recovery_eta_before_astar": (now_arg + float(getattr(sortie_arg, "finish_time_s", now_arg) - now_arg)) if sortie_arg is not None else None,
                        "segment_count_required": required, "astar_segments_attempted": len(details),
                        "astar_segments_succeeded": int(success_segments), "astar_segments_failed": int(failed_segments),
                        "exact_total_distance_m": meta.get("actual_distance_m"), "exact_total_duration_s": meta.get("actual_flight_time_s"),
                        "exact_total_energy": getattr(result, "energy", None) if result is not None else None,
                        "exact_recovery_eta": getattr(result, "recovery_eta", None) if result is not None else None,
                        "materialization_success": bool(result is not None), "failure_stage": failure_stage,
                        "failure_reason": None if result is not None else (failure or "invalid_materialization_result"),
                        "failure_detail": None if result is not None else str(meta), "elapsed_time_ms": (perf_counter() - started) * 1000.0,
                    }
                    materialization_attempt_rows.append(attempt_row)
                    for index, detail in enumerate(details):
                        astar_segment_rows.append({
                            "episode_id": f"episode_{episode_idx}", "decision_epoch_id": attempt_row["decision_epoch_id"], "simulation_time": now_arg,
                            "materialization_attempt_id": attempt_id, "segment_plan_id": f"{attempt_id}_segment_{index:03d}", "segment_index": index,
                            "segment_type": "launch_to_task" if index == 0 else ("task_to_recovery" if index == required - 1 else "task_to_task"),
                            "origin_type": None, "origin_id": None, "origin_x": None, "origin_y": None, "origin_z": None,
                            "destination_type": None, "destination_id": None, "destination_x": None, "destination_y": None, "destination_z": None,
                            "euclidean_distance_m": detail.get("distance_m"), "voxel_start": detail.get("start_voxel"), "voxel_goal": detail.get("goal_voxel"),
                            "astar_called": True, "astar_success": bool(detail.get("astar_success", detail.get("cache_hit", False))),
                            "astar_status": detail.get("astar_status", "success"), "astar_failure_reason": None if detail.get("astar_success", detail.get("cache_hit", False)) else meta.get("failure_reason"),
                            "planning_time_ms": float(detail.get("planning_time_s", 0.0)) * 1000.0, "expanded_nodes": detail.get("expanded_nodes"), "generated_nodes": None,
                            "open_set_peak_size": None, "path_voxel_count": detail.get("path_voxel_count"), "path_distance_m": detail.get("distance_m"),
                            "cache_hit": bool(detail.get("cache_hit", False)), "cache_key": None,
                        })
                    return result
                finally:
                    add_phase("selected_sortie_materialization_time", perf_counter() - started)
            env.sim.sortie_generator.materialize_selected_sortie = timed_materialize

    if timing_enabled:
        uav_module_root = project_root() / "truck_uav_modules"
        if str(uav_module_root) not in sys.path:
            sys.path.insert(0, str(uav_module_root))
        import uav_path_planner as _uav_path_planner
        planner_original = _uav_path_planner.plan_uav_path
        def timed_uav_path(*args, **kwargs):
            profile = kwargs.setdefault("profile", {})
            started = perf_counter()
            result = planner_original(*args, **kwargs)
            elapsed = perf_counter() - started
            add_phase("uav_path_planning_time", elapsed)
            current_planner["calls"] += 1
            current_planner["full_time_s"] += elapsed
            mode = str(result.get("path_mode", ""))
            if mode == "direct":
                current_planner["direct"] += 1
            elif mode == "local_detour":
                current_planner["local_detour"] += 1
            if not bool(result.get("feasible")):
                current_planner["failed"] += 1
            current_planner["edge_time_s"] += float(profile.get("visibility_edge_construction_ms", 0.0)) / 1000.0
            current_planner["astar_time_s"] += float(profile.get("astar_search_ms", 0.0)) / 1000.0
            return result
        _uav_path_planner.plan_uav_path = timed_uav_path
        import build_and_benchmark_3d_voxel_astar as _voxel_core
        voxel_original = _voxel_core.plan_3d_path
        def timed_voxel_astar(*args, **kwargs):
            started = perf_counter()
            try:
                simple_counts["voxel_astar_call_count"] += 1
                return voxel_original(*args, **kwargs)
            finally:
                add_phase("voxel_astar_search_time", perf_counter() - started)
        _voxel_core.plan_3d_path = timed_voxel_astar
    reset_started = perf_counter()
    obs, mask = reset_env(env)
    add_phase("environment_reset_time", perf_counter() - reset_started)
    install_sim_timing_wrappers()
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    if env.decision_mode == "joint_plan":
        expected_action_dim = 1 + int(cfg.get("joint_plan", {}).get("max_non_safe_plans", 24))
        hp.candidate_dim = int(env.plan_feature_dim)
    else:
        expected_action_dim = 1 + int(cfg["agents"].get("num_drones", 2)) * int(cfg["sortie"].get("top_k_per_drone", 8))
    if action_dim != expected_action_dim:
        raise ValueError(f"action_dim mismatch: env={action_dim}, expected={expected_action_dim}")

    online_net = make_q_network(algo, obs_dim, action_dim, candidate_dim=hp.candidate_dim).to(device)
    target_net = make_q_network(algo, obs_dim, action_dim, candidate_dim=hp.candidate_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(online_net.parameters(), lr=hp.lr)
    current_reward_metadata = reward_config_metadata(cfg)
    resolved_resume_from = ""
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            resume_path = project_root() / resume_path
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:  # PyTorch < 2.0
            checkpoint = torch.load(resume_path, map_location=device)
        validate_resume_checkpoint(
            checkpoint,
            expected_obs_dim=obs_dim,
            expected_action_dim=action_dim,
            expected_candidate_dim=int(hp.candidate_dim),
            expected_decision_mode=env.decision_mode,
            current_reward_metadata=current_reward_metadata,
        )
        online_net.load_state_dict(checkpoint["model_state_dict"])
        target_net.load_state_dict(online_net.state_dict())
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        resolved_resume_from = str(resume_path)
    time_mode = str(cfg.get("time", {}).get("mode", cfg.get("time_mode", "fixed_interval"))).lower()
    discount_mode, configured_gamma, reference_interval_s, n_step = resolve_discount_settings(cfg)
    if not np.isclose(configured_gamma, float(hp.gamma), rtol=0.0, atol=1e-12):
        raise ValueError(
            "discount gamma mismatch between rl.gamma and DQN hyperparameters: "
            f"discount={configured_gamma}, optimizer={hp.gamma}"
        )
    n_step_accumulator = NStepSMDPAccumulator(n_step)
    replay_kwargs = {
        "default_transition_discount": float(hp.gamma),
        "time_mode": time_mode,
        "require_time_fields": time_mode == "event_driven",
    }
    if algo == "rainbow_dqn":
        replay = PrioritizedMaskedReplayBuffer(
            hp.replay_buffer_size,
            obs_dim,
            action_dim,
            seed=base_seed,
            alpha=hp.prioritized_replay_alpha,
            priority_epsilon=hp.priority_epsilon,
            **replay_kwargs,
        )
    else:
        replay = MaskedReplayBuffer(
            hp.replay_buffer_size,
            obs_dim,
            action_dim,
            seed=base_seed,
            **replay_kwargs,
        )

    training_rows: List[Dict] = []
    eval_rows: List[Dict] = []
    final_evaluation_transition_rows: List[Dict] = []
    final_evaluation_route_event_rows: List[Dict] = []
    final_evaluation_task_rows: List[Dict] = []
    episode_reward = 0.0
    episode_steps = 0
    episode_replay_transition_count = 0
    episode_idx = 0
    latest_loss = np.nan
    gradient_updates = 0
    training_simulated_seconds = 0.0
    td_error_diagnostics = {
        key: {
            "transition_count": 0,
            "rewards": [],
            "td_errors": [],
            "priorities": [],
            "gradient_norms": [],
        }
        for key in STAGE5_DT_BINS
    }
    all_sampled_delta_t: List[float] = []
    all_sampled_td_error: List[float] = []
    all_transition_delta_t: List[float] = []
    all_effective_gamma: List[float] = []
    episode_td_errors: List[float] = []
    episode_q_values: List[float] = []
    q_value_sum = 0.0
    q_value_count = 0
    replay_event_counts: Counter[str] = Counter()
    sampled_event_counts: Counter[str] = Counter()
    policy_decision_event_counts: Counter[str] = Counter()
    reward_component_sums: Counter[str] = Counter()
    reward_component_nonzero: Counter[str] = Counter()
    reward_component_transitions = 0
    nonfinite_loss_count = 0
    nonfinite_gradient_count = 0
    maximum_gradient_norm = 0.0
    # Keep the range on the training device.  Calling ``.item()`` here on
    # every update would force CUDA/CPU synchronization and materially slow
    # the environment/training loop.
    minimum_q_value_device: torch.Tensor | None = None
    maximum_q_value_device: torch.Tensor | None = None
    minimum_q_value = float("nan")
    maximum_q_value = float("nan")
    gradient_norm_history: List[float] = []
    training_started_at = perf_counter()
    stage_end_steps = np.cumsum([int(stage.timesteps) for stage in stages]).tolist()
    current_stage_idx = 0
    scheduler.set_stage(current_stage_idx)
    episode_wall_started = perf_counter()
    if max_episodes == 1:
        print("EPISODE_STARTED", flush=True)
    print(
        f"DQN seed_mode={resolved_seed_mode}; mask_mode={'masked' if action_masking else 'unmasked'}; "
        f"stage 1/{len(stages)} {stages[0].name}, seeds={stages[0].seed_pool}, timesteps={stages[0].timesteps}"
    )

    for step in range(1, effective_timesteps + 1):
        while current_stage_idx + 1 < len(stage_end_steps) and step > stage_end_steps[current_stage_idx]:
            current_stage_idx += 1
            scheduler.set_stage(current_stage_idx)
            stage = stages[current_stage_idx]
            print(
                f"DQN seed_mode={resolved_seed_mode}; mask_mode={'masked' if action_masking else 'unmasked'}; "
                f"stage {current_stage_idx + 1}/{len(stages)} {stage.name}, "
                f"seeds={stage.seed_pool}, timesteps={stage.timesteps}"
            )
        epsilon = exploration_epsilon(algo, step, hp)
        if algo == "rainbow_dqn":
            reset_noisy_layers(online_net)
        stored_mask = policy_action_mask(mask, action_masking)
        policy_started = perf_counter()
        action = select_action(online_net, obs, stored_mask, epsilon, rng, device)
        simple_counts["safe_plan_count"] += int(action == 0)
        add_phase("policy_inference_time", perf_counter() - policy_started)
        step_started = perf_counter()
        next_obs, reward, done, next_mask, transition_info = step_env(env, action)
        add_phase("environment_step_time", perf_counter() - step_started)
        ledger_values = dict(transition_info.get("reward_ledger_components", {}) or {})
        if ledger_values:
            reward_component_transitions += 1
            for name, value in ledger_values.items():
                number = float(value)
                if not np.isfinite(number):
                    raise FloatingPointError(f"non-finite reward component {name} at step {step}: {number}")
                reward_component_sums[name] += number
                reward_component_nonzero[name] += int(abs(number) > 1e-12)
        if time_mode == "event_driven":
            delta_t = float(transition_info["delta_t"])
            bootstrap_discount = float(transition_info["transition_discount"])
        else:
            delta_t = reference_interval_s
            bootstrap_discount = float(hp.gamma)
        training_simulated_seconds += float(transition_info.get("delta_t", delta_t))
        raw_transition = NStepTransition(
            obs=np.asarray(obs, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_obs=np.asarray(next_obs, dtype=np.float32),
            terminated=bool(transition_info.get("terminated", done)),
            truncated=bool(transition_info.get("truncated", False)),
            action_mask=np.asarray(stored_mask, dtype=bool),
            next_action_mask=np.asarray(policy_action_mask(next_mask, action_masking), dtype=bool),
            delta_t=delta_t,
            bootstrap_discount=bootstrap_discount,
            decision_time=float(transition_info.get("decision_time", 0.0)),
            trigger_event_type="|".join(transition_info.get("trigger_events", [])),
            instantaneous_dispatch=bool(transition_info.get("instantaneous_dispatch", False)),
        )
        replay_started = perf_counter()
        stored_transitions = n_step_accumulator.append(raw_transition)
        episode_replay_transition_count += len(stored_transitions)
        for stored_transition in stored_transitions:
            all_transition_delta_t.append(float(stored_transition.delta_t))
            all_effective_gamma.append(float(stored_transition.bootstrap_discount))
            replay_event_counts.update(
                stage5_event_categories(
                    stored_transition.trigger_event_type,
                    instantaneous_dispatch=stored_transition.instantaneous_dispatch,
                )
            )
            replay.add(
                stored_transition.obs,
                stored_transition.action,
                stored_transition.reward,
                stored_transition.next_obs,
                stored_transition.done,
                stored_transition.action_mask,
                stored_transition.next_action_mask,
                terminated=stored_transition.terminated,
                truncated=stored_transition.truncated,
                delta_t=stored_transition.delta_t,
                transition_discount=stored_transition.bootstrap_discount,
                decision_time=stored_transition.decision_time,
                trigger_event_type=stored_transition.trigger_event_type,
                instantaneous_dispatch=stored_transition.instantaneous_dispatch,
            )
        add_phase("replay_buffer_time", perf_counter() - replay_started)

        obs = next_obs
        mask = next_mask
        episode_reward += reward
        episode_steps += 1

        if (
            step >= hp.learning_starts
            and len(replay) >= hp.batch_size
            and step % max(1, int(hp.train_frequency)) == 0
        ):
            network_started = perf_counter()
            if algo == "rainbow_dqn":
                reset_noisy_layers(online_net)
                batch = replay.sample(hp.batch_size, device, beta=prioritized_beta_by_step(step, hp))
            else:
                batch = replay.sample(hp.batch_size, device)
            weights = batch.weights if algo == "rainbow_dqn" else None
            loss, td_errors, current_q_values = compute_dqn_loss(
                algo,
                online_net,
                target_net,
                batch,
                hp.gamma,
                weights=weights,
                action_masking=action_masking,
            )
            if not bool(torch.isfinite(loss).item()) or not bool(torch.isfinite(td_errors).all().item()):
                nonfinite_loss_count += 1
                raise FloatingPointError(f"non-finite loss or TD-error at training step {step}")
            # ``current_q_values`` is detached.  Accumulating scalar tensors on
            # the device adds no autograd graph and avoids two host syncs per
            # gradient update.  The existing finite loss/TD-error guard also
            # catches non-finite current Q values.
            batch_q_min = current_q_values.min()
            batch_q_max = current_q_values.max()
            minimum_q_value_device = (
                batch_q_min
                if minimum_q_value_device is None
                else torch.minimum(minimum_q_value_device, batch_q_min)
            )
            maximum_q_value_device = (
                batch_q_max
                if maximum_q_value_device is None
                else torch.maximum(maximum_q_value_device, batch_q_max)
            )
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = float(nn.utils.clip_grad_norm_(online_net.parameters(), hp.max_grad_norm).item())
            if not np.isfinite(gradient_norm):
                nonfinite_gradient_count += 1
                raise FloatingPointError(f"non-finite gradient norm at training step {step}")
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
            gradient_norm_history.append(gradient_norm)
            optimizer.step()
            gradient_updates += 1
            if algo == "rainbow_dqn" and batch.indices is not None:
                replay.update_priorities(batch.indices, td_errors)
            durations = batch.delta_ts.detach().cpu().numpy()
            absolute_errors = td_errors.detach().abs().cpu().numpy()
            sampled_q_values = current_q_values.detach().cpu().numpy()
            episode_td_errors.extend(float(value) for value in absolute_errors)
            episode_q_values.extend(float(value) for value in sampled_q_values)
            q_value_sum += float(sampled_q_values.sum())
            q_value_count += int(sampled_q_values.size)
            sampled_rewards = batch.rewards.detach().cpu().numpy()
            sampled_priorities = (
                batch.sampled_priorities.detach().cpu().numpy()
                if batch.sampled_priorities is not None
                else np.full_like(absolute_errors, np.nan, dtype=float)
            )
            labels = np.asarray([stage5_td_error_delta_t_bin(float(value)) for value in durations])
            for label in STAGE5_DT_BINS:
                td_error_diagnostics[label]["transition_count"] += int(np.sum(labels == label))
            capture_diagnostics = gradient_updates % max(1, int(hp.diagnostic_sample_stride)) == 0
            if capture_diagnostics:
                all_sampled_delta_t.extend(float(value) for value in durations)
                all_sampled_td_error.extend(float(value) for value in absolute_errors)
                for label in STAGE5_DT_BINS:
                    selected = labels == label
                    if not selected.any():
                        continue
                    bucket = td_error_diagnostics[label]
                    bucket["rewards"].extend(float(value) for value in sampled_rewards[selected])
                    bucket["td_errors"].extend(float(value) for value in absolute_errors[selected])
                    bucket["priorities"].extend(
                        float(value) for value in sampled_priorities[selected] if np.isfinite(value)
                    )
                    bucket["gradient_norms"].extend([gradient_norm] * int(np.sum(selected)))
            if capture_diagnostics and batch.trigger_event_types is not None:
                instantaneous = batch.instantaneous_dispatches.detach().cpu().numpy()
                for event_types, is_instantaneous in zip(batch.trigger_event_types, instantaneous):
                    sampled_event_counts.update(
                        stage5_event_categories(
                            str(event_types), instantaneous_dispatch=bool(is_instantaneous)
                        )
                    )
            latest_loss = float(loss.item())
            add_phase("network_training_time", perf_counter() - network_started)

        if step % hp.target_update_interval == 0:
            target_started = perf_counter()
            target_net.load_state_dict(online_net.state_dict())
            add_phase("target_network_update_time", perf_counter() - target_started)

        if hp.checkpoint_interval > 0 and step % int(hp.checkpoint_interval) == 0 and step < effective_timesteps:
            checkpoint_started = perf_counter()
            save_intermediate_checkpoint(
                run_dir=run_dir,
                step=step,
                algo=algo,
                seed=base_seed,
                online_net=online_net,
                optimizer=optimizer,
                obs_dim=obs_dim,
                action_dim=action_dim,
                candidate_dim=int(hp.candidate_dim),
                decision_mode=env.decision_mode,
                reward_metadata=current_reward_metadata,
                feature_schema=(
                    env.sim.joint_plan_generator.feature_builder.schema()
                    if env.sim is not None and env.sim.joint_plan_generator is not None
                    else {}
                ),
                hyperparams=hp,
            )
            add_phase("checkpoint_time", perf_counter() - checkpoint_started)

        if step % 1000 == 0:
            write_progress(step, "training")

        if done:
            metrics = env.episode_summary()
            metrics["replay_transition_count"] = int(episode_replay_transition_count)
            metrics["smdp_aggregated_episode_reward"] = float(episode_reward)
            policy_decision_event_counts.update(
                dict(metrics.get("policy_trigger_event_counts", {}) or {})
            )
            minimum_q_value, maximum_q_value = materialize_q_range(
                minimum_q_value_device, maximum_q_value_device
            )
            episode_td = np.asarray(episode_td_errors, dtype=float)
            episode_q = np.asarray(episode_q_values, dtype=float)
            row = train_metric_row(
                step=step,
                episode=episode_idx,
                episode_steps=episode_steps,
                episode_reward=episode_reward,
                epsilon=epsilon,
                latest_loss=latest_loss,
                metrics=metrics,
                seed_mode=resolved_seed_mode,
                stage=current_stage_idx + 1,
                stage_name=stages[current_stage_idx].name,
                q_value_min=float(episode_q.min()) if episode_q.size else np.nan,
                q_value_max=float(episode_q.max()) if episode_q.size else np.nan,
                q_value_mean=float(episode_q.mean()) if episode_q.size else np.nan,
                td_error_mean=float(episode_td.mean()) if episode_td.size else np.nan,
                td_error_max=float(episode_td.max()) if episode_td.size else np.nan,
            )
            training_rows.append(row)
            row["simulated_time_s"] = float(env.sim.now)
            row["decisions_per_episode"] = int(getattr(env, "decision_index", episode_steps))
            row["gradient_updates"] = int(gradient_updates)
            row["replay_transitions"] = int(len(replay))
            row["training_simulated_seconds"] = float(training_simulated_seconds)
            if writer is not None:
                logging_started = perf_counter()
                write_train_scalars(writer, row, step)
                add_phase("logging_time", perf_counter() - logging_started)
            if timing_enabled:
                episode_timing_rows.append({
                    "episode": int(episode_idx), "episode_reward": float(episode_reward),
                    "episode_transitions": int(episode_steps),
                    "episode_simulation_time_s": float(env.sim.now),
                    "episode_wall_time_s": float(perf_counter() - episode_wall_started),
                    "transitions_per_second": float(episode_steps / max(1e-9, perf_counter() - episode_wall_started)),
                    **{name: float(current_episode_timing[name]) for name in phase_names},
                    "uav_planner_calls": int(current_planner["calls"]),
                    "direct_path_calls": int(current_planner["direct"]),
                    "local_detour_calls": int(current_planner["local_detour"]),
                    "failed_path_calls": int(current_planner["failed"]),
                    "uav_full_planning_time_s": float(current_planner["full_time_s"]),
                    "uav_edge_construction_time_s": float(current_planner["edge_time_s"]),
                    "uav_astar_search_time_s": float(current_planner["astar_time_s"]),
                })
            episode_idx += 1
            if max_episodes is not None and episode_idx >= int(max_episodes):
                if max_episodes == 1:
                    print("EPISODE_COMPLETED", flush=True)
                break
            current_episode_timing = {name: 0.0 for name in phase_names}
            current_episode_calls = {name: 0 for name in phase_names}
            current_planner = {"calls": 0, "direct": 0, "local_detour": 0, "failed": 0, "full_time_s": 0.0, "edge_time_s": 0.0, "astar_time_s": 0.0}
            episode_td_errors = []
            episode_q_values = []
            episode_wall_started = perf_counter()
            obs, mask = reset_env(env)
            install_sim_timing_wrappers()
            episode_reward = 0.0
            episode_steps = 0
            episode_replay_transition_count = 0

        should_eval = step % hp.eval_interval == 0 or step == effective_timesteps
        if should_eval:
            write_progress(step, "evaluating")
            capture_final_audit = step == effective_timesteps
            eval_df = evaluate_policy(
                cfg,
                online_net,
                device,
                algo,
                step,
                eval_seeds,
                eval_scenario,
                action_masking=action_masking,
                transition_audit_rows=(
                    final_evaluation_transition_rows if capture_final_audit else None
                ),
                route_event_rows=(
                    final_evaluation_route_event_rows if capture_final_audit else None
                ),
                task_audit_rows=(
                    final_evaluation_task_rows if capture_final_audit else None
                ),
            )
            eval_rows.extend(eval_df.to_dict("records"))
            eval_mean = eval_df[["completion_rate", "timeout_rate", "episode_reward"]].mean(numeric_only=True)
            print(
                f"[{algo}] step={step} epsilon={epsilon:.3f} "
                f"completion={eval_mean.get('completion_rate', np.nan):.3f} "
                f"timeout={eval_mean.get('timeout_rate', np.nan):.3f} "
                f"eval_reward={eval_mean.get('episode_reward', np.nan):.2f}"
            )

    if max_episodes is not None and episode_idx >= int(max_episodes):
        effective_timesteps = int(step)
    if episode_steps > 0:
        metrics = env.sim.metrics(algo).to_dict()
        metrics["replay_transition_count"] = int(episode_replay_transition_count)
        metrics["smdp_aggregated_episode_reward"] = float(episode_reward)
        partial_delta = np.asarray(getattr(env, "_delta_t_history", []), dtype=float)
        partial_gamma = np.asarray(getattr(env, "_discount_history", []), dtype=float)
        metrics.update({
            "delta_t_mean": float(partial_delta.mean()) if partial_delta.size else 0.0,
            "delta_t_min": float(partial_delta.min()) if partial_delta.size else 0.0,
            "delta_t_max": float(partial_delta.max()) if partial_delta.size else 0.0,
            "effective_gamma_mean": float(partial_gamma.mean()) if partial_gamma.size else 0.0,
            "effective_gamma_min": float(partial_gamma.min()) if partial_gamma.size else 0.0,
            "effective_gamma_max": float(partial_gamma.max()) if partial_gamma.size else 0.0,
        })
        minimum_q_value, maximum_q_value = materialize_q_range(
            minimum_q_value_device, maximum_q_value_device
        )
        episode_td = np.asarray(episode_td_errors, dtype=float)
        episode_q = np.asarray(episode_q_values, dtype=float)
        row = train_metric_row(
            step=effective_timesteps,
            episode=episode_idx,
            episode_steps=episode_steps,
            episode_reward=episode_reward,
            epsilon=exploration_epsilon(algo, effective_timesteps, hp),
            latest_loss=latest_loss,
            metrics=metrics,
            partial_episode=True,
            seed_mode=resolved_seed_mode,
            stage=current_stage_idx + 1,
            stage_name=stages[current_stage_idx].name,
            q_value_min=float(episode_q.min()) if episode_q.size else np.nan,
            q_value_max=float(episode_q.max()) if episode_q.size else np.nan,
            q_value_mean=float(episode_q.mean()) if episode_q.size else np.nan,
            td_error_mean=float(episode_td.mean()) if episode_td.size else np.nan,
            td_error_max=float(episode_td.max()) if episode_td.size else np.nan,
        )
        training_rows.append(row)
        row["simulated_time_s"] = float(env.sim.now)
        row["decisions_per_episode"] = int(getattr(env, "decision_index", episode_steps))
        row["gradient_updates"] = int(gradient_updates)
        row["replay_transitions"] = int(len(replay))
        row["training_simulated_seconds"] = float(training_simulated_seconds)
        if writer is not None:
            write_train_scalars(writer, row, effective_timesteps)

    wall_clock_time_s = perf_counter() - training_started_at
    save_outputs(
        out_dir=out_dir,
        run_dir=run_dir,
        algo=algo,
        seed=base_seed,
        cfg_path=cfg_path,
        hp=hp,
        online_net=online_net,
        optimizer=optimizer,
        obs_dim=obs_dim,
        action_dim=action_dim,
        training_rows=training_rows,
        eval_rows=eval_rows,
        total_timesteps=effective_timesteps,
        model_name=model_name,
        train_scenario=train_scenario,
        tensorboard_log_dir=tensorboard_log_dir,
        seed_mode=resolved_seed_mode,
        stages=[
            {
                "name": stage.name,
                "timesteps": int(stage.timesteps),
                "seeds": stage.seeds,
                "random_seed": bool(stage.random_seed),
            }
            for stage in stages
        ],
        action_masking=action_masking,
        decision_mode=env.decision_mode,
        feature_schema=(
            env.sim.joint_plan_generator.feature_builder.schema()
            if env.sim is not None and env.sim.joint_plan_generator is not None
            else {}
        ),
        replay_config={
            **replay.state_dict(),
            "n_step": n_step,
            "reference_interval_s": reference_interval_s,
            "gamma_reference": float(hp.gamma),
            "discount_mode": discount_mode,
            "exploration_policy": (
                f"uniform_valid_actions_through_step_{int(hp.learning_starts)}_then_noisynet"
                if algo == "rainbow_dqn"
                else "epsilon_greedy_schedule"
            ),
        },
        reward_metadata=current_reward_metadata,
        resume_from=resolved_resume_from,
        wall_clock_time_s=wall_clock_time_s,
    )
    pd.DataFrame(final_evaluation_transition_rows).to_csv(
        run_dir / "evaluation_transition_audit.csv",
        index=False,
    )
    pd.DataFrame(final_evaluation_route_event_rows).to_csv(
        run_dir / "evaluation_route_events.csv",
        index=False,
    )
    pd.DataFrame(final_evaluation_task_rows).to_csv(
        run_dir / "evaluation_task_audit.csv",
        index=False,
    )
    td_rows = []
    for key, values in td_error_diagnostics.items():
        rewards = np.asarray(values["rewards"], dtype=float)
        errors = np.asarray(values["td_errors"], dtype=float)
        priorities = np.asarray(values["priorities"], dtype=float)
        gradient_norms = np.asarray(values["gradient_norms"], dtype=float)
        td_rows.append(
            {
                "delta_t_bin": key,
                "transition_count": int(values["transition_count"]),
                "diagnostic_sample_count": int(errors.size),
                "reward_mean": float(rewards.mean()) if rewards.size else np.nan,
                "reward_std": float(rewards.std(ddof=0)) if rewards.size else np.nan,
                "absolute_td_error_mean": float(errors.mean()) if errors.size else np.nan,
                "absolute_td_error_median": float(np.median(errors)) if errors.size else np.nan,
                "absolute_td_error_p95": float(np.percentile(errors, 95)) if errors.size else np.nan,
                "sampled_priority_mean": float(priorities.mean()) if priorities.size else np.nan,
                "gradient_norm_mean": float(gradient_norms.mean()) if gradient_norms.size else np.nan,
            }
        )
    pd.DataFrame(td_rows).to_csv(run_dir / "td_error_by_delta_t.csv", index=False)
    total_replay_events = max(1, sum(replay_event_counts.values()))
    total_sampled_events = max(1, sum(sampled_event_counts.values()))
    event_names = sorted(set(replay_event_counts) | set(sampled_event_counts))
    pd.DataFrame(
        [
            {
                "event_type": event_type,
                "replay_count": replay_event_counts[event_type],
                "replay_ratio": replay_event_counts[event_type] / total_replay_events,
                "sampled_count": sampled_event_counts[event_type],
                "sampled_ratio": sampled_event_counts[event_type] / total_sampled_events,
            }
            for event_type in event_names
        ]
    ).to_csv(run_dir / "replay_event_composition.csv", index=False)
    pd.DataFrame(
        [
            {"event_type": event_type, "policy_decision_count": int(count)}
            for event_type, count in sorted(policy_decision_event_counts.items())
        ]
    ).to_csv(run_dir / "policy_decision_event_counts.csv", index=False)
    pd.DataFrame(
        [
            {
                "component": name,
                "episode_training_sum": float(reward_component_sums[name]),
                "transition_mean": float(reward_component_sums[name]) / max(1, reward_component_transitions),
                "nonzero_ratio": float(reward_component_nonzero[name]) / max(1, reward_component_transitions),
            }
            for name in sorted(reward_component_sums)
        ]
    ).to_csv(run_dir / "reward_component_training_summary.csv", index=False)
    transition_delta_array = np.asarray(all_transition_delta_t, dtype=float)
    effective_gamma_array = np.asarray(all_effective_gamma, dtype=float)
    sampled_td_error_array = np.asarray(all_sampled_td_error, dtype=float)
    minimum_q_value, maximum_q_value = materialize_q_range(
        minimum_q_value_device, maximum_q_value_device
    )
    diagnostics = {
        "delta_t_mean": float(transition_delta_array.mean()) if transition_delta_array.size else 0.0,
        "delta_t_min": float(transition_delta_array.min()) if transition_delta_array.size else 0.0,
        "delta_t_max": float(transition_delta_array.max()) if transition_delta_array.size else 0.0,
        "effective_gamma_mean": float(effective_gamma_array.mean()) if effective_gamma_array.size else 0.0,
        "effective_gamma_min": float(effective_gamma_array.min()) if effective_gamma_array.size else 0.0,
        "effective_gamma_max": float(effective_gamma_array.max()) if effective_gamma_array.size else 0.0,
        "td_error_mean": float(sampled_td_error_array.mean()) if sampled_td_error_array.size else 0.0,
        "td_error_max": float(sampled_td_error_array.max()) if sampled_td_error_array.size else 0.0,
        "q_value_mean": float(q_value_sum / q_value_count) if q_value_count else 0.0,
        "q_value_min": float(minimum_q_value) if np.isfinite(minimum_q_value) else None,
        "q_value_max": float(maximum_q_value) if np.isfinite(maximum_q_value) else None,
        "pearson_abs_td_error_delta_t": _safe_correlation(all_sampled_delta_t, all_sampled_td_error),
        "spearman_abs_td_error_delta_t": _safe_correlation(
            all_sampled_delta_t, all_sampled_td_error, ranked=True
        ),
        "nonfinite_loss_count": int(nonfinite_loss_count),
        "nonfinite_gradient_count": int(nonfinite_gradient_count),
        "maximum_gradient_norm": float(maximum_gradient_norm),
        "minimum_q_value": float(minimum_q_value) if np.isfinite(minimum_q_value) else None,
        "maximum_q_value": float(maximum_q_value) if np.isfinite(maximum_q_value) else None,
        "mean_gradient_norm": float(np.mean(gradient_norm_history)) if gradient_norm_history else 0.0,
        "p95_gradient_norm": float(np.percentile(gradient_norm_history, 95)) if gradient_norm_history else 0.0,
        "p99_gradient_norm": float(np.percentile(gradient_norm_history, 99)) if gradient_norm_history else 0.0,
        "gradient_updates": int(gradient_updates),
        "training_simulated_seconds": float(training_simulated_seconds),
        "wall_clock_time_s": float(wall_clock_time_s),
        "simulated_seconds_per_wall_second": float(training_simulated_seconds / max(wall_clock_time_s, 1e-12)),
        "reward_component_transitions": int(reward_component_transitions),
        "diagnostic_sample_stride": int(hp.diagnostic_sample_stride),
        "resume_from": resolved_resume_from,
        "reward_metadata": current_reward_metadata,
    }
    with open(run_dir / "stage5_training_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    with open(run_dir / "replay_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **replay.state_dict(),
                "n_step": n_step,
                "reference_interval_s": reference_interval_s,
                "gamma_reference": float(hp.gamma),
                "discount_mode": discount_mode,
                "exploration_policy": (
                    f"uniform_valid_actions_through_step_{int(hp.learning_starts)}_then_noisynet"
                    if algo == "rainbow_dqn"
                    else "epsilon_greedy_schedule"
                ),
                "gradient_updates": gradient_updates,
                "training_simulated_seconds": training_simulated_seconds,
                "wall_clock_time_s": wall_clock_time_s,
                "reward_config_signature": current_reward_metadata["signature"],
                "resume_from": resolved_resume_from,
            },
            f,
            indent=2,
        )
    if reward_audit_out_dir is not None:
        # The exporter reads the already-emitted transition ledger; it neither
        # changes the reward returned by the environment nor updates the model.
        from experiments.reward_audit import export_reward_audit

        audit_summary = export_reward_audit(
            out_dir=Path(reward_audit_out_dir),
            env=env,
            cfg=cfg,
            hp=hp,
            seed=base_seed,
            episode_reward=episode_reward,
            algo=algo,
        )
        print(
            "Saved reward audit: "
            f"{Path(reward_audit_out_dir)} (reward={audit_summary['episode_reward_from_transitions']:.6f})",
            flush=True,
        )
    if timing_enabled:
        timing_dir = Path(timing_out_dir or (run_dir / "timing"))
        timing_dir.mkdir(parents=True, exist_ok=True)
        # The context is shared by initial materialization and in-flight CAV
        # recovery updates.  Export its request log separately so the two
        # sources remain auditable even though their A* time is nested in the
        # existing environment phases.
        voxel_context = (
            getattr(getattr(env.sim, "sortie_generator", None), "_voxel_context", None)
            if env.sim is not None else None
        )
        voxel_execution_stats = (
            voxel_context.snapshot_stats() if voxel_context is not None else {}
        )
        recovery_replanning_stats = (
            {key: int(value) for key, value in getattr(env.sim.ground_controller, "replanning_stats", {}).items()}
            if env.sim is not None and getattr(env.sim, "ground_controller", None) is not None else {}
        )
        mandatory_recovery_counters = (
            "exact_replan_request_count", "exact_replan_success_count", "same_target_skip_count",
            "path_suffix_reuse_count", "local_connection_success_count", "merged_request_count",
            "exact_replan_limit_skip_count", "retained_old_plan_count",
        )
        for key in mandatory_recovery_counters:
            recovery_replanning_stats.setdefault(key, 0)
        for key in ("astar_request_count", "astar_actual_search_count", "astar_cache_hit_count", "astar_timeout_count", "astar_node_budget_exceeded_count"):
            voxel_execution_stats.setdefault(key, 0)
        request_rows = list(getattr(voxel_context, "request_log", []) or [])
        pd.DataFrame(request_rows).to_csv(timing_dir / "voxel_astar_request_log.csv", index=False)
        (timing_dir / "recovery_replanning_statistics.json").write_text(
            json.dumps(
                {
                    "voxel_execution": voxel_execution_stats,
                    "inflight_recovery_replanning": recovery_replanning_stats,
                    "call_chain": [
                        "ActiveCAVController.commit_cav_action",
                        "ActiveCAVController._apply_recovery_update",
                        "ActiveCAVController._plan_voxel_recovery_leg",
                        "uav_path_planner.plan_selected_sortie_3d",
                        "VoxelPlannerContext.segment",
                        "build_and_benchmark_3d_voxel_astar.plan_3d_path",
                    ],
                }, ensure_ascii=False, indent=2
            ), encoding="utf-8",
        )
        timing_frame = pd.DataFrame(episode_timing_rows)
        timing_frame.to_csv(timing_dir / "training_10_episode_timing.csv", index=False)
        measured_total = float(sum(phase_totals.values()))
        process_rows = []
        for name in phase_names:
            process_rows.append({
                "process_name": name, "total_time_s": float(phase_totals[name]),
                "mean_time_per_call_ms": float(phase_totals[name] * 1000.0 / max(1, phase_calls[name])),
                "call_count": int(phase_calls[name]),
                "percentage_of_total_training_time": float(phase_totals[name] / max(wall_clock_time_s, 1e-12) * 100.0),
            })
        other = max(0.0, wall_clock_time_s - measured_total)
        process_rows.append({
            "process_name": "other_overhead_time", "total_time_s": other,
            "mean_time_per_call_ms": other * 1000.0 / max(1, len(episode_timing_rows)),
            "call_count": len(episode_timing_rows),
            "percentage_of_total_training_time": other / max(wall_clock_time_s, 1e-12) * 100.0,
        })
        process_rows.append({
            "process_name": "total_training_time", "total_time_s": wall_clock_time_s,
            "mean_time_per_call_ms": wall_clock_time_s * 1000.0 / max(1, len(episode_timing_rows)),
            "call_count": len(episode_timing_rows), "percentage_of_total_training_time": 100.0,
        })
        pd.DataFrame(process_rows).to_csv(timing_dir / "training_10_episode_process_summary.csv", index=False)
        total_transitions = int(timing_frame["episode_transitions"].sum()) if not timing_frame.empty else 0
        report = {
            "config": str(cfg_path), "training_entry": "dynamic_patrol_marl.rl.train_masked_dqn",
            "algorithm": algo, "planner": "A* visibility graph (Euclidean heuristic)",
            "seed": int(base_seed), "episodes_completed": int(len(timing_frame)),
            "max_episodes": int(max_episodes) if max_episodes is not None else None,
            "total_training_time_s": wall_clock_time_s, "total_transitions": total_transitions,
            "mean_transitions_per_episode": total_transitions / max(1, len(timing_frame)),
            "transitions_per_second": total_transitions / max(wall_clock_time_s, 1e-12),
            "transitions_per_minute": total_transitions / max(wall_clock_time_s, 1e-12) * 60.0,
            "transitions_per_hour": total_transitions / max(wall_clock_time_s, 1e-12) * 3600.0,
            "phase_totals_s": {**{k: float(v) for k, v in phase_totals.items()}, "other_overhead_time": other},
            "uav_planner": {
                "calls": int(sum(row["uav_planner_calls"] for row in episode_timing_rows)),
                "direct_paths": int(sum(row["direct_path_calls"] for row in episode_timing_rows)),
                "local_detours": int(sum(row["local_detour_calls"] for row in episode_timing_rows)),
                "failed_paths": int(sum(row["failed_path_calls"] for row in episode_timing_rows)),
                "full_planning_time_s": float(sum(row["uav_full_planning_time_s"] for row in episode_timing_rows)),
                "edge_construction_time_s": float(sum(row["uav_edge_construction_time_s"] for row in episode_timing_rows)),
                "astar_search_time_s": float(sum(row["uav_astar_search_time_s"] for row in episode_timing_rows)),
            },
        }
        (timing_dir / "training_10_episode_timing_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if max_episodes == 1 and not timing_frame.empty:
            episode_row = timing_frame.iloc[0]
            environment_initialization = float(phase_totals["environment_reset_time"])
            candidate_generation = float(phase_totals["candidate_generation_time"])
            observation_and_action_mask = float(
                phase_totals["observation_building_time"]
                + phase_totals["candidate_feature_time"]
                + phase_totals["action_mask_time"]
            )
            policy_inference = float(phase_totals["policy_inference_time"])
            selected_materialization = float(phase_totals["selected_sortie_materialization_time"])
            voxel_astar_time = float(phase_totals["voxel_astar_search_time"])
            reward_calculation = float(phase_totals["reward_calculation_time"])
            replay_and_training = float(
                phase_totals["replay_buffer_time"]
                + phase_totals["network_training_time"]
                + phase_totals["target_network_update_time"]
            )
            simulation_and_state_update = max(
                0.0,
                float(phase_totals["environment_step_time"])
                - candidate_generation
                - observation_and_action_mask
                - selected_materialization
                - reward_calculation,
            )
            episode_wall = float(wall_clock_time_s + environment_initialization)
            top_level = {
                "environment_initialization": environment_initialization,
                "candidate_generation": candidate_generation,
                "observation_and_action_mask": observation_and_action_mask,
                "policy_inference": policy_inference,
                "selected_sortie_materialization": selected_materialization,
                "simulation_and_state_update": simulation_and_state_update,
                "reward_calculation": reward_calculation,
                "replay_buffer_and_training": replay_and_training,
            }
            other_overhead = max(0.0, episode_wall - sum(top_level.values()))
            top_level["other_overhead"] = other_overhead
            simple_rows = []
            simple_call_counts = {
                "environment_initialization": 1,
                "candidate_generation": int(phase_calls["candidate_generation_time"]),
                "observation_and_action_mask": int(
                    phase_calls["observation_building_time"]
                    + phase_calls["candidate_feature_time"]
                    + phase_calls["action_mask_time"]
                ),
                "policy_inference": int(phase_calls["policy_inference_time"]),
                "selected_sortie_materialization": int(phase_calls["selected_sortie_materialization_time"]),
                "simulation_and_state_update": int(phase_calls["environment_step_time"]),
                "reward_calculation": int(phase_calls["reward_calculation_time"]),
                "replay_buffer_and_training": int(
                    phase_calls["replay_buffer_time"]
                    + phase_calls["network_training_time"]
                    + phase_calls["target_network_update_time"]
                ),
                "other_overhead": 1,
            }
            for process_name, total_time in top_level.items():
                calls = simple_call_counts[process_name]
                simple_rows.append({
                    "process_name": process_name,
                    "call_count": calls,
                    "total_time_s": total_time,
                    "mean_time_ms": total_time * 1000.0 / max(1, calls),
                    "percentage_of_episode_wall_time": total_time / max(episode_wall, 1e-12) * 100.0,
                })
            pd.DataFrame(simple_rows).to_csv(timing_dir / "one_episode_simple_timing.csv", index=False)
            simple_summary = {
                "episode_wall_time_s": episode_wall,
                "episode_transitions": int(episode_row["episode_transitions"]),
                "transitions_per_second": int(episode_row["episode_transitions"]) / max(episode_wall, 1e-12),
                "episode_reward": float(episode_row["episode_reward"]),
                "candidate_generation_calls": int(phase_calls["candidate_generation_time"]),
                "selected_uav_plan_count": int(simple_counts["selected_uav_plan_count"]),
                "safe_plan_count": int(simple_counts["safe_plan_count"]),
                "voxel_astar_call_count": int(simple_counts["voxel_astar_call_count"]),
                "voxel_astar_timeout_count": int(simple_counts["voxel_astar_timeout_count"]),
                "materialization_failure_count": int(simple_counts["materialization_failure_count"]),
                "network_update_count": int(gradient_updates),
                "learning_starts": int(hp.learning_starts),
                "voxel_astar_search_time_s": voxel_astar_time,
                "selected_sortie_materialization_time_s": selected_materialization,
                "voxel_execution": voxel_execution_stats,
                "inflight_recovery_replanning": recovery_replanning_stats,
                "processes": simple_rows,
                "timing_note": "voxel_astar_search_time is nested in selected_sortie_materialization and is not added to top-level totals",
            }
            (timing_dir / "one_episode_simple_summary.json").write_text(
                json.dumps(simple_summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            baseline = dict(cfg.get("recovery_replanning_optimization", {}).get("baseline", {}))
            if baseline:
                metric_row = next(
                    (row for row in training_rows if not bool(row.get("partial_episode", False))),
                    training_rows[0] if training_rows else {},
                )
                def metric_value(*names, default=0.0):
                    for name in names:
                        if name in metric_row and pd.notna(metric_row[name]):
                            return float(metric_row[name])
                    return float(default)
                current = {
                    "seed": int(base_seed), "task_count": metric_value("released_task_count", "total_tasks", default=0),
                    "episode_total_time_s": episode_wall,
                    "voxel_astar_call_count": int(voxel_execution_stats.get("astar_actual_search_count", 0)),
                    "voxel_astar_total_time_s": voxel_astar_time,
                    "simulation_and_state_update_time_s": simulation_and_state_update,
                    "selected_sortie_materialization_time_s": selected_materialization,
                    "completion_rate": metric_value("completion_rate_final", "completion_rate"),
                    "episode_reward": float(episode_row["episode_reward"]),
                    "illegal_actions": metric_value("illegal_actions", "illegal_action_count"),
                    "commit_failures": metric_value("commit_failure_count", "commit_failures"),
                }
                (timing_dir / "recovery_replanning_comparison.json").write_text(
                    json.dumps(
                        {
                            "baseline": baseline, "optimized": current,
                            "delta_optimized_minus_baseline": {
                                key: current[key] - float(value)
                                for key, value in baseline.items() if key in current and isinstance(value, (int, float))
                            },
                        }, ensure_ascii=False, indent=2
                    ), encoding="utf-8",
                )
            import matplotlib.pyplot as plt
            figure_rows = sorted(simple_rows, key=lambda row: row["total_time_s"])
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.barh(
                [row["process_name"] for row in figure_rows],
                [row["total_time_s"] for row in figure_rows],
                color="#4472C4",
            )
            ax.set_xlabel("Wall time (s)")
            ax.set_title("Stage 5 one-episode simple timing")
            fig.tight_layout()
            fig.savefig(timing_dir / "one_episode_simple_timing.png", dpi=180)
            plt.close(fig)
            ranked = sorted(top_level.items(), key=lambda item: item[1], reverse=True)
            print("=" * 60, flush=True)
            print("ONE EPISODE SIMPLE TIMING", flush=True)
            print("=" * 60, flush=True)
            print(f"Episode wall time: {episode_wall:.3f} s", flush=True)
            print(f"Transitions: {int(episode_row['episode_transitions'])}", flush=True)
            print(f"Transitions per second: {simple_summary['transitions_per_second']:.6f}", flush=True)
            print(f"Episode reward: {float(episode_row['episode_reward']):.6f}", flush=True)
            for name, value in top_level.items():
                print(f"{name}: {value:.3f} s ({value / max(episode_wall, 1e-12) * 100.0:.2f}%)", flush=True)
            print(f"  Voxel A* search: {voxel_astar_time:.6f} s", flush=True)
            print(f"Candidate generation calls: {simple_summary['candidate_generation_calls']}", flush=True)
            print(f"Selected UAV plans: {simple_summary['selected_uav_plan_count']}", flush=True)
            print(f"Voxel A* calls: {simple_summary['voxel_astar_call_count']}", flush=True)
            print(f"Voxel A* timeouts: {simple_summary['voxel_astar_timeout_count']}", flush=True)
            print(f"Network updates: {gradient_updates}", flush=True)
            print("Top 3 time-consuming modules:", flush=True)
            for index, (name, value) in enumerate(ranked[:3], 1):
                print(f"{index}. {name}: {value:.3f} s", flush=True)
            print("=" * 60, flush=True)
            print("RESULTS_SAVED", flush=True)
    if max_episodes == 1 and timing_out_dir is not None:
        diag_dir = Path(timing_out_dir)
        diag_dir.mkdir(parents=True, exist_ok=True)
        attempts_df = pd.DataFrame(materialization_attempt_rows)
        segments_df = pd.DataFrame(astar_segment_rows)
        attempts_df.to_csv(diag_dir / "materialization_attempts.csv", index=False)
        segments_df.to_csv(diag_dir / "astar_segment_calls.csv", index=False)
        failure_summary = attempts_df[attempts_df.get("materialization_success", pd.Series(dtype=bool)) == False].groupby(["failure_stage", "failure_reason"], dropna=False).size().reset_index(name="count") if not attempts_df.empty else pd.DataFrame(columns=["failure_stage", "failure_reason", "count"])
        failure_summary.to_csv(diag_dir / "materialization_failure_summary.csv", index=False)
        status_summary = segments_df.groupby(["astar_status", "astar_success"], dropna=False).size().reset_index(name="count") if not segments_df.empty else pd.DataFrame(columns=["astar_status", "astar_success", "count"])
        status_summary.to_csv(diag_dir / "astar_status_summary.csv", index=False)
        stage_timing = pd.DataFrame([{"stage": "materialization_total", "total_time_s": float(attempts_df["elapsed_time_ms"].sum() / 1000.0) if not attempts_df.empty else 0.0}, {"stage": "astar_total", "total_time_s": float(segments_df["planning_time_ms"].sum() / 1000.0) if not segments_df.empty else 0.0}])
        stage_timing.to_csv(diag_dir / "materialization_stage_timing.csv", index=False)
        attempt_count = len(attempts_df); success_count = int(attempts_df["materialization_success"].sum()) if not attempts_df.empty else 0
        seg_count = len(segments_df); seg_success = int(segments_df["astar_success"].sum()) if not segments_df.empty else 0
        status_counts = Counter(segments_df["astar_status"].astype(str).tolist()) if not segments_df.empty else Counter()
        diag_summary = {
            "state_transition_count": int(episode_steps), "safe_plan_count": int(simple_counts["safe_plan_count"]),
            "selected_uav_action_count": attempt_count, "materialization_attempt_count": attempt_count,
            "materialization_success_count": success_count, "materialization_failure_count": attempt_count - success_count,
            "committed_uav_sortie_count": success_count, "astar_path_request_count": seg_count,
            "astar_actual_search_count": int(sum(not bool(x) for x in segments_df.get("cache_hit", pd.Series(dtype=bool)))) if not segments_df.empty else 0,
            "astar_segment_success_count": seg_success, "astar_segment_failure_count": seg_count - seg_success,
            "astar_timeout_count": int(status_counts.get("timeout", 0) + status_counts.get("sortie_timeout", 0)), "astar_no_path_count": int(status_counts.get("no_path", 0)),
            "astar_invalid_start_count": int(status_counts.get("invalid_start", 0)), "astar_invalid_goal_count": int(status_counts.get("invalid_goal", 0)),
            "astar_projection_failure_count": int(sum(status_counts.get(x, 0) for x in ("start_projection_failed", "goal_projection_failed"))),
            "astar_internal_error_count": int(status_counts.get("internal_error", 0)), "astar_cache_hit_count": int(segments_df["cache_hit"].sum()) if not segments_df.empty else 0,
            "failure_before_astar_count": int((attempts_df["failure_stage"] == "before_astar").sum()) if not attempts_df.empty else 0,
            "failure_during_astar_count": int((attempts_df["failure_stage"] == "during_astar").sum()) if not attempts_df.empty else 0,
            "failure_after_astar_count": int((attempts_df["failure_stage"] == "after_astar").sum()) if not attempts_df.empty else 0,
            "mean_segments_per_materialization_attempt": seg_count / max(1, attempt_count), "mean_segments_per_successful_sortie": float(attempts_df.loc[attempts_df["materialization_success"], "segment_count_required"].mean()) if success_count else 0.0,
            "mean_segments_per_failed_sortie": float(attempts_df.loc[~attempts_df["materialization_success"], "segment_count_required"].mean()) if attempt_count-success_count else 0.0,
            "mean_successful_materialization_time_s": float(attempts_df.loc[attempts_df["materialization_success"], "elapsed_time_ms"].mean()/1000.0) if success_count else None,
            "mean_failed_materialization_time_s": float(attempts_df.loc[~attempts_df["materialization_success"], "elapsed_time_ms"].mean()/1000.0) if attempt_count-success_count else None,
            "total_astar_time_s": float(segments_df["planning_time_ms"].sum()/1000.0) if not segments_df.empty else 0.0,
            "consistency_attempts": attempt_count == success_count + (attempt_count-success_count), "consistency_astar": seg_count == seg_success + (seg_count-seg_success),
        }
        (diag_dir / "materialization_diagnostics_summary.json").write_text(json.dumps(diag_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        failure_summary.to_csv(diag_dir / "materialization_failure_summary.csv", index=False)
        print(f"MATERIALIZATION_DIAGNOSTICS_SAVED {diag_dir}", flush=True)
    if writer is not None:
        writer.flush()
        writer.close()
    write_progress(effective_timesteps, "complete")
    return run_dir / model_name


def parse_eval_seeds(value) -> List[int]:
    if value is None:
        return [1000, 1001, 1002, 1003, 1004]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def build_hyperparams(args: argparse.Namespace, cfg: Dict | None = None) -> DQNHyperParams:
    values = asdict(DQNHyperParams())
    rl_cfg = (cfg or {}).get("rl", {})
    config_hyperparams = rl_cfg.get("hyperparams", rl_cfg.get("dqn_hyperparams", {})) or {}
    values.update({k: v for k, v in config_hyperparams.items() if k in values})
    if "gamma" in rl_cfg:
        values["gamma"] = rl_cfg["gamma"]
    if args.params:
        root = project_root()
        params_path = Path(args.params)
        if not params_path.is_absolute():
            params_path = root / params_path
        with open(params_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        values.update({k: v for k, v in loaded.items() if k in values})
    for key in values:
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            values[key] = cli_value
    return DQNHyperParams(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_training.yaml")
    parser.add_argument("--algo", choices=ALGORITHMS, default=None, help="default: rl.algo from config, else dqn")
    parser.add_argument("--timesteps", type=int, default=None, help="override total training timesteps")
    parser.add_argument("--seed", type=int, default=None, help="base seed for fixed/curriculum sampling")
    parser.add_argument("--seed-mode", choices=SEED_MODES, default=None, help="default: training.seed_mode from config, else fixed")
    parser.add_argument("--stage-timesteps", default=None, help="comma-separated stage timesteps; one value or one per stage")
    parser.add_argument("--out-dir", default=None, help="override project.results_dir from the config")
    parser.add_argument("--run-name", default=None, help="default: <algo>_seed<seed>; used for CSV and TensorBoard subdirs")
    parser.add_argument("--tensorboard-dir", default=None, help="default: results/rl/tensorboard/dqn")
    parser.add_argument("--disable-tensorboard", action="store_true", help="train without writing TensorBoard event files")
    parser.add_argument("--unmasked", action="store_true", help="train without using the action mask for selection or target Q values")
    parser.add_argument("--model-name", default=None, help="override rl.model_name from the config")
    parser.add_argument("--resume-from", default=None, help="Stage 5 checkpoint whose model and optimizer state are loaded")
    parser.add_argument("--params", default=None, help="optional DQN hyperparameter YAML file")
    parser.add_argument("--eval-seeds", default=None)
    parser.add_argument("--scenario", default=None, help="training scenario label passed to DynamicPatrolEnv")
    parser.add_argument("--eval-scenario", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--replay-buffer-size", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--train-frequency", type=int, default=None)
    parser.add_argument("--target-update-interval", type=int, default=None)
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--epsilon-final", type=float, default=None)
    parser.add_argument("--epsilon-decay-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--candidate-dim", type=int, default=None)
    parser.add_argument("--prioritized-replay-alpha", type=float, default=None)
    parser.add_argument("--prioritized-replay-beta-start", type=float, default=None)
    parser.add_argument("--prioritized-replay-beta-final", type=float, default=None)
    parser.add_argument("--prioritized-replay-beta-steps", type=int, default=None)
    parser.add_argument("--priority-epsilon", type=float, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--diagnostic-sample-stride", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None, help="optional hard stop after complete episodes")
    parser.add_argument("--timing-out-dir", default=None, help="write optional per-episode timing profile")
    parser.add_argument("--reward-audit-out-dir", default=None, help="optional reward-ledger audit directory for the completed episode")
    args = parser.parse_args()

    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(cfg_path)
    rl_cfg = cfg.get("rl", {})
    algo = args.algo or rl_cfg.get("algo", "dqn")
    run_name = args.run_name or rl_cfg.get("run_name")
    model_name = args.model_name or rl_cfg.get("model_name")
    total_timesteps = args.timesteps if args.timesteps is not None else rl_cfg.get("timesteps")
    eval_seeds = parse_eval_seeds(args.eval_seeds if args.eval_seeds is not None else rl_cfg.get("eval_seeds"))
    train_scenario = args.scenario or rl_cfg.get("train_scenario", "small")
    eval_scenario = args.eval_scenario or rl_cfg.get("eval_scenario", "eval")
    config_action_masking = bool(rl_cfg.get("action_masking", True))
    action_masking = False if args.unmasked else config_action_masking
    project_cfg = cfg.get("project", {})
    configured_out_dir = args.out_dir if args.out_dir is not None else project_cfg.get("results_dir")
    configured_tensorboard_dir = (
        args.tensorboard_dir
        if args.tensorboard_dir is not None
        else project_cfg.get("tensorboard_dir")
    )
    out_dir, tensorboard_dir = resolve_dqn_output_dirs(
        root,
        configured_out_dir,
        configured_tensorboard_dir,
    )
    # The long-running core-ablation queue may begin while an externally
    # launched first job is still active.  A completed run is immutable: skip
    # rather than overwrite it if the ablation configuration opts into this
    # guard.  This guard has no effect on model, environment, or evaluation.
    if bool(cfg.get("ablation", {}).get("skip_completed_run", False)):
        existing_run_dir = out_dir / str(run_name)
        progress_path = existing_run_dir / "progress.json"
        summary_path = existing_run_dir / "summary.csv"
        if progress_path.exists() and summary_path.exists():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                progress = {}
            if progress.get("status") in {"complete", "completed"}:
                print(f"Skipping completed ablation run: {existing_run_dir}")
                return
    reward_audit_out_dir = None
    if args.reward_audit_out_dir is not None:
        reward_audit_out_dir = Path(args.reward_audit_out_dir)
        if not reward_audit_out_dir.is_absolute():
            reward_audit_out_dir = root / reward_audit_out_dir

    model_path = train_masked_dqn(
        cfg=cfg,
        cfg_path=str(cfg_path),
        algo=algo,
        total_timesteps=total_timesteps,
        seed=args.seed,
        out_dir=out_dir,
        hp=build_hyperparams(args, cfg),
        eval_seeds=eval_seeds,
        eval_scenario=eval_scenario,
        device_name=args.device,
        model_name=model_name,
        train_scenario=train_scenario,
        run_name=run_name,
        tensorboard_dir=tensorboard_dir,
        enable_tensorboard=not args.disable_tensorboard,
        seed_mode=args.seed_mode,
        stage_timesteps=parse_int_list(args.stage_timesteps),
        action_masking=action_masking,
        resume_from=None if args.resume_from is None else Path(args.resume_from),
        max_episodes=args.max_episodes,
        timing_out_dir=None if args.timing_out_dir is None else Path(args.timing_out_dir),
        reward_audit_out_dir=reward_audit_out_dir,
    )
    run_dir = model_path.parent
    run_name = run_dir.name
    print(f"Saved model: {model_path}")
    print(f"Saved curves: {run_dir / 'training_curve.csv'}, {run_dir / 'eval_curve.csv'}")
    print(f"Saved summary: {run_dir / 'summary.csv'}")
    print(f"Saved summary index: {out_dir / 'summary.csv'}")
    print(f"Saved TensorBoard logs: {tensorboard_dir / run_name}")


if __name__ == "__main__":
    main()
