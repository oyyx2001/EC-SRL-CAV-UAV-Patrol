"""Reproducible, side-effect-free reward audit exports for one RL episode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPONENT_COLUMNS = (
    "task_completion_reward",
    "priority_reward",
    "deadline_reward",
    "response_time_reward",
    "weighted_aoi_reward",
    "recovery_reward",
    "energy_penalty",
    "uav_distance_penalty",
    "cav_distance_penalty",
    "waiting_penalty",
    "timeout_penalty",
    "invalid_action_penalty",
    "other_reward",
)


def _ids(value: Any) -> List[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [int(item) for item in str(value).split("|") if str(item).strip()]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _save_figures(frame: pd.DataFrame, task_frame: pd.DataFrame, out_dir: Path, support: Dict[str, Any]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    totals = frame.loc[:, list(COMPONENT_COLUMNS)].sum(axis=0).sort_values(key=np.abs, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    totals.plot.bar(ax=ax, color=["#2a9d8f" if value >= 0 else "#e76f51" for value in totals])
    ax.set_ylabel("Cumulative reward")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "01_reward_components_total.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for name in COMPONENT_COLUMNS:
        values = frame[name].to_numpy(dtype=float)
        if np.any(np.abs(values) > 1e-12):
            ax.plot(frame["decision_step"], values, linewidth=0.8, label=name)
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Component reward")
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "02_reward_components_by_step.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(frame["total_reward"], bins=min(40, max(10, len(frame) // 5)), color="#457b9d", edgecolor="white")
    ax.set_xlabel("Per-transition reward")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "03_step_reward_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame["decision_step"], frame["total_reward"].cumsum(), color="#2a9d8f")
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Cumulative reward")
    fig.tight_layout()
    fig.savefig(out_dir / "04_cumulative_reward_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    completed = task_frame[task_frame["status"] == "done"].copy()
    ax.bar(completed["task_id"].astype(str), completed["lifecycle_task_specific_reward"], color="#2a9d8f")
    ax.set_xlabel("Completed task ID")
    ax.set_ylabel("Task-specific reward")
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    fig.tight_layout()
    fig.savefig(out_dir / "05_reward_per_completed_task.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    current = support.get("current_c51_support")
    proposed = support.get("hypothetical_scheme_c_support")
    if current is None:
        ax.text(0.5, 0.58, "Current Rainbow implementation is scalar-Q, not C51.", ha="center", va="center", fontsize=12)
        ax.text(0.5, 0.42, f"If C51 is introduced: [{proposed['v_min']}, {proposed['v_max']}], "
                f"{proposed['num_atoms']} atoms, spacing {proposed['atom_spacing']:.2f}", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.vlines(current["atoms"], 0, 1, label="current")
        ax.vlines(proposed["atoms"], 0, 0.6, label="hypothetical scheme C")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "06_rainbow_value_support_comparison.png", dpi=160)
    plt.close(fig)


def _reward_function_audit_markdown(cfg: Dict[str, Any], summary: Dict[str, Any]) -> str:
    reward = cfg.get("reward", {})
    weights = reward.get("stage5_weights", {})
    references = reward.get("stage5_references", {})
    return f"""# Reward-function audit (seed {summary['seed']})

## Active path

The active configuration uses `reward.mode: {reward.get('mode')}` and
`reward.accounting_mode: {reward.get('accounting_mode')}`.  Each event-driven
transition calls `DynamicPatrolEnv._event_reward_components`, which derives
new task sets by `after.done_ids - before.done_ids` and
`after.expired_ids - before.expired_ids`, then calls
`_stage5_coordinated_reward_components`.

| Component | Formula in the active path | Trigger / timing | Scaling |
|---|---|---|---|
| task completion | `{weights.get('complete')} × number of newly done tasks` | task first enters `done_ids` | no `delta_t`; no clip/normalization |
| priority | `{weights.get('priority')} × sum(priority of newly done tasks)` | same one-time completion event | no `delta_t`; additive to completion, not a duplicate completion payment |
| timeout | `-{weights.get('timeout')} × number of newly expired tasks` | task first enters `expired_ids` | no `delta_t` |
| weighted AoI | `-{weights.get('aoi')} × ΔAoI / {references.get('time_s', 60)}` | every state-advance interval | increment, not cumulative state |
| flight / hover energy | `-{weights.get('flight_energy')} × ΔE_flight`, `-{weights.get('hover_energy')} × ΔE_hover` | every interval | increment, no `delta_t` multiplier because energy is already integrated |
| UAV/CAV distance | `-{weights.get('return_distance')} × Δreturn/{references.get('return_distance_m', 1000)}`, `-{weights.get('cav_distance')} × ΔCAV/{references.get('cav_distance_m', 1000)}` | every interval | distance increments |
| recovery waiting/queue | `-{weights.get('recovery_wait')} × Δwait/{references.get('time_s', 60)}`, `-{weights.get('recovery_queue')} × Δqueue-occupancy/{references.get('time_s', 60)}` | every interval | integrated increments |
| route/co-ordination/illegal | negative configured cost times event delta | event only | no positive recovery bonus is present |

The configured `smdp.interval_normalize_reward` only divides AoI by the
reference interval.  There is no active reward clipping, reward scaling, or
post-hoc normalization.  The legacy `_reward` function contains cumulative
state terms, but it is not called by this `stage5_coordinated` event-driven
episode.

## Duplicate-settlement checks

Completion and priority rewards use the same newly-done set and are intended
as two distinct terms for one completion.  A task removed from availability
remains in `done_ids`, so it cannot re-enter the set difference in a later
transition.  The audit run found **{summary['duplicate_completion_task_count']}**
tasks with more than one completion reward event.  Recovery has no positive
reward component, and CAV/UAV do not maintain separate per-task completion
reward streams.

## Conservation

Every emitted transition is checked at the environment boundary with
`abs(total_reward - sum(audit_components)) < 1e-8`.  Maximum observed absolute
residual: `{summary['max_reward_conservation_error']:.3e}`.
"""


def export_reward_audit(
    *,
    out_dir: Path,
    env: Any,
    cfg: Dict[str, Any],
    hp: Any,
    seed: int,
    episode_reward: float,
    algo: str,
) -> Dict[str, Any]:
    """Export one completed episode's ledger without changing its reward."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(list(env.transition_log))
    if raw.empty:
        raise ValueError("reward audit requires a completed episode with transition_log rows")
    for name in COMPONENT_COLUMNS:
        raw[name] = pd.to_numeric(raw.get(name, 0.0), errors="coerce").fillna(0.0)
    frame = pd.DataFrame({
        "decision_step": pd.to_numeric(raw["decision_index"], errors="coerce").fillna(-1).astype(int),
        "simulation_time": pd.to_numeric(raw["current_time"], errors="coerce").fillna(0.0),
        "selected_action": pd.to_numeric(raw.get("selected_action", raw["action"]), errors="coerce").fillna(-1).astype(int),
        "selected_plan_type": raw.get("selected_plan_type", raw.get("action_type", "")).fillna(""),
        "completed_task_ids": raw.get("completed_task_ids", "").fillna(""),
        "timed_out_task_ids": raw.get("timed_out_task_ids", "").fillna(""),
    })
    for name in COMPONENT_COLUMNS:
        frame[name] = raw[name].astype(float)
    frame["total_reward"] = pd.to_numeric(raw["reward_total"], errors="coerce").fillna(0.0)
    frame["component_sum"] = frame.loc[:, list(COMPONENT_COLUMNS)].sum(axis=1)
    frame["reward_conservation_error"] = frame["total_reward"] - frame["component_sum"]
    max_error = float(frame["reward_conservation_error"].abs().max())
    if not bool((frame["reward_conservation_error"].abs() < 1e-8).all()):
        raise AssertionError(f"audit reward conservation failed: max_error={max_error}")
    frame.to_csv(out_dir / "reward_component_by_step.csv", index=False)

    total = float(frame["total_reward"].sum())
    summary_rows = []
    for name in (*COMPONENT_COLUMNS, "total_reward"):
        series = frame[name]
        cumulative = float(series.sum())
        summary_rows.append({
            "reward_component": name,
            "cumulative_value": cumulative,
            "share_of_episode_reward": cumulative / total if abs(total) > 1e-12 else np.nan,
            "mean_per_step": float(series.mean()),
            "std_per_step": float(series.std(ddof=0)),
            "min_per_step": float(series.min()),
            "max_per_step": float(series.max()),
            "nonzero_steps": int((series.abs() > 1e-12).sum()),
            "over_50_percent_of_total": bool(abs(cumulative) > 0.5 * abs(total)),
        })
    component_summary = pd.DataFrame(summary_rows)
    component_summary.to_csv(out_dir / "reward_component_summary.csv", index=False)

    events: Dict[int, List[Dict[str, Any]]] = {}
    for _, row in frame.iterrows():
        for task_id in _ids(row["completed_task_ids"]):
            events.setdefault(task_id, []).append({
                "decision_step": int(row.decision_step), "simulation_time": float(row.simulation_time),
                "event": "completion", "task_completion_reward": float(row.task_completion_reward) /
                max(1, len(_ids(row.completed_task_ids))),
                "priority_reward": float(row.priority_reward) / max(1, len(_ids(row.completed_task_ids))),
                "timeout_penalty": 0.0,
            })
        for task_id in _ids(row["timed_out_task_ids"]):
            events.setdefault(task_id, []).append({
                "decision_step": int(row.decision_step), "simulation_time": float(row.simulation_time),
                "event": "timeout", "task_completion_reward": 0.0, "priority_reward": 0.0,
                "timeout_penalty": float(row.timeout_penalty) / max(1, len(_ids(row.timed_out_task_ids))),
            })
    task_rows = []
    for task in sorted(env.sim.tasks, key=lambda item: int(item.id)):
        task_events = events.get(int(task.id), [])
        completion_events = [event for event in task_events if event["event"] == "completion"]
        timeout_events = [event for event in task_events if event["event"] == "timeout"]
        task_rows.append({
            "task_id": int(task.id), "priority": _finite(getattr(task, "priority", 0.0)),
            "release_time": _finite(getattr(task, "release_time", 0.0)), "deadline": _finite(getattr(task, "deadline", 0.0)),
            "completion_time": getattr(task, "completion_time", None), "status": str(getattr(task, "status", "")),
            "completion_reward_event_count": len(completion_events), "timeout_reward_event_count": len(timeout_events),
            "task_completion_reward_total": float(sum(event["task_completion_reward"] for event in completion_events)),
            "priority_reward_total": float(sum(event["priority_reward"] for event in completion_events)),
            "timeout_penalty_total": float(sum(event["timeout_penalty"] for event in timeout_events)),
            "lifecycle_task_specific_reward": float(sum(
                event["task_completion_reward"] + event["priority_reward"] + event["timeout_penalty"]
                for event in task_events
            )),
            "reward_events_json": json.dumps(task_events, ensure_ascii=False),
        })
    task_frame = pd.DataFrame(task_rows)
    task_frame.to_csv(out_dir / "reward_by_task.csv", index=False)
    duplicates = task_frame[task_frame["completion_reward_event_count"] > 1].copy()
    duplicates.to_csv(out_dir / "duplicate_reward_events.csv", index=False)

    max_priority = max((_finite(getattr(task, "priority", 0.0)) for task in env.sim.tasks), default=0.0)
    normal_single_task_upper = float(cfg.get("reward", {}).get("stage5_weights", {}).get("complete", 30.0)) + \
        float(cfg.get("reward", {}).get("stage5_weights", {}).get("priority", 2.0)) * max_priority
    large_threshold = max(100.0, 2.0 * normal_single_task_upper)
    large_events = frame[(frame["total_reward"].abs() > large_threshold) | (
        (frame["completed_task_ids"].astype(str) == "") & (frame["total_reward"] > normal_single_task_upper)
    )].copy()
    large_events["large_reward_threshold"] = large_threshold
    large_events.to_csv(out_dir / "large_reward_events.csv", index=False)

    gamma = float(getattr(hp, "gamma", 0.99))
    rewards = frame["total_reward"].to_numpy(dtype=float)
    positive_rewards = rewards[rewards > 0.0]
    discounted_returns = np.zeros_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        discounted_returns[index] = running
    mean_reward = float(rewards.mean())
    positive_mean = float(positive_rewards.mean()) if positive_rewards.size else 0.0
    scale = {
        "mean_reward_per_step": mean_reward,
        "max_reward_per_step": float(rewards.max()),
        "min_reward_per_step": float(rewards.min()),
        "positive_mean_reward_per_step": positive_mean,
        "estimated_discounted_return_scale": mean_reward / max(1e-12, 1.0 - gamma),
        "mean_value_scale": mean_reward / max(1e-12, 1.0 - gamma),
        "positive_value_scale": positive_mean / max(1e-12, 1.0 - gamma),
        "empirical_discounted_return_min": float(discounted_returns.min()),
        "empirical_discounted_return_max": float(discounted_returns.max()),
    }
    # This is only a migration target for a future categorical-C51 network.
    # It encloses the observed Monte-Carlo return range with room for poorer
    # policies, while 5 reward units per atom still resolves the 30-point
    # completion event and 2-point priority increments usefully.
    hypothetical_c = {"v_min": -1000.0, "v_max": 1000.0, "num_atoms": 401, "atom_spacing": 5.0}
    reward_cfg = cfg.get("reward", {})
    rainbow_audit = {
        "algorithm": str(algo), "is_c51_distributional": False,
        "gamma": gamma, "n_step": int(cfg.get("smdp", {}).get("n_step", 1)),
        "v_min": None, "v_max": None, "num_atoms": None,
        "reward_clip": None, "reward_scale": 1.0,
        "reward_scale_configured": "reward_scale" in reward_cfg,
        "reward_clip_configured": "reward_clip" in reward_cfg,
        **scale,
        "estimated_vmin_saturation_ratio": None, "estimated_vmax_saturation_ratio": None,
        "distribution_atom_spacing": None,
        "assessment": "Not applicable: CandidateAwareRainbowQNetwork emits scalar Q-values; it has NoisyLinear heads and prioritized replay but no C51 atom support or projection.",
        "hypothetical_scheme_c_support": hypothetical_c,
    }
    (out_dir / "rainbow_value_range_audit.json").write_text(json.dumps(rainbow_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison_rows = []
    for label, factor, v_min, v_max, atoms, applicable in (
        ("A_keep_original_reward", 1.0, None, None, None, False),
        ("B_uniform_reward_scale_0_1", 0.1, None, None, None, False),
        ("C_keep_reward_expand_hypothetical_C51_support", 1.0, -1000.0, 1000.0, 401, True),
    ):
        comparison_rows.append({
            "scheme": label, "reward_scale": factor,
            "step_reward_min": float(rewards.min() * factor), "step_reward_max": float(rewards.max() * factor),
            "episode_reward": float(total * factor),
            "mean_discounted_value_scale": float(scale["mean_value_scale"] * factor),
            "positive_discounted_value_scale": float(scale["positive_value_scale"] * factor),
            "v_min": v_min, "v_max": v_max, "num_atoms": atoms,
            "atom_spacing": None if atoms is None else (v_max - v_min) / (atoms - 1),
            "c51_applicable_to_current_code": applicable,
            "relative_reward_weights_preserved": True,
        })
    pd.DataFrame(comparison_rows).to_csv(out_dir / "reward_scaling_comparison.csv", index=False)

    top20 = frame.reindex(frame["total_reward"].abs().nlargest(min(20, len(frame))).index).sort_values("total_reward", key=np.abs, ascending=False)
    summary = {
        "seed": int(seed), "algorithm": str(algo), "transition_count": int(len(frame)),
        "episode_reward_from_transitions": total, "episode_reward_from_trainer": float(episode_reward),
        "episode_reward_difference": total - float(episode_reward),
        "max_reward_conservation_error": max_error,
        "completed_task_count": int((task_frame["status"] == "done").sum()),
        "timeout_task_count": int((task_frame["status"] == "expired").sum()),
        "duplicate_completion_task_count": int(len(duplicates)),
        "positive_reward_steps": int((rewards > 1e-12).sum()),
        "negative_reward_steps": int((rewards < -1e-12).sum()),
        "zero_reward_steps": int((np.abs(rewards) <= 1e-12).sum()),
        "mean_reward_per_step": mean_reward,
        "reward_per_completed_task": total / max(1, int((task_frame["status"] == "done").sum())),
        "max_step_reward": float(rewards.max()), "min_step_reward": float(rewards.min()),
        "normal_single_task_reward_upper": normal_single_task_upper,
        "large_reward_threshold": large_threshold, "large_reward_event_count": int(len(large_events)),
        "positive_without_task_completion_count": int(((frame["completed_task_ids"].astype(str) == "") & (frame["total_reward"] > normal_single_task_upper)).sum()),
        "top_20_absolute_reward_steps": top20.to_dict(orient="records"),
        "rainbow_value_range": rainbow_audit,
        "recommendation": "A_keep_original_reward: the observed total is explained by one-time completion plus priority rewards, and this implementation has no C51 support to saturate. Do not add reward_scale or C51 bounds unless the network is deliberately migrated to categorical C51.",
    }
    (out_dir / "reward_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "reward_function_audit.md").write_text(_reward_function_audit_markdown(cfg, summary), encoding="utf-8")
    _save_figures(frame, task_frame, out_dir, {"current_c51_support": None, "hypothetical_scheme_c_support": hypothetical_c})
    return summary
