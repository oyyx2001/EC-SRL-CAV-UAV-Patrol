from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import tracemalloc
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.utils.config import load_config


METHOD_CONFIGS = {
    "Legacy-C3": "configs/scenario_60node_stage3_c3_reroute.yaml",
    "J0-Service": "configs/scenario_60node_stage4_joint_service_only.yaml",
    "J1-Rendezvous": "configs/scenario_60node_stage4_joint_service_rendezvous.yaml",
    "J2-Full-Guard": "configs/scenario_60node_stage4_joint_full.yaml",
    "J3-Full-NoGuard": "configs/scenario_60node_stage4_joint_full_no_return_guard.yaml",
}


def _reset(env: DynamicPatrolEnv, seed: int):
    out = env.reset(seed=seed)
    return out if isinstance(out, tuple) else (out, {"action_mask": env.action_masks()})


def _step(env: DynamicPatrolEnv, action: int):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated or truncated), info
    return out


def _action(env: DynamicPatrolEnv) -> int:
    if env.decision_mode == "joint_plan":
        return env.sim.joint_plan_generator.select_heuristic(env.joint_plan_cache)
    mask = env.action_masks()
    valid = np.flatnonzero(mask[1:]) + 1
    if not valid.size:
        return 0
    return int(min(valid, key=lambda action: (-float(env.candidates[action - 1].score), int(action))))


def run_episode(label: str, config_path: str, seed: int):
    cfg = load_config(config_path)
    env = DynamicPatrolEnv(cfg)
    _reset(env, seed)
    selected_rows: List[Dict[str, Any]] = []
    tracemalloc.start()
    wall_start = time.perf_counter()
    done = False
    steps = 0
    while not done and steps < 10000:
        action = _action(env)
        if env.decision_mode == "joint_plan":
            plan = env.joint_plan_cache.plan_slots[action]
            if plan is not None:
                row = plan.to_record()
                row.update({"method": label, "seed": seed, "decision_index": steps, "selected": True})
                selected_rows.append(row)
        _, _, done, _ = _step(env, action)
        steps += 1
    wall_s = time.perf_counter() - wall_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metric = env.episode_summary()
    metric.update({
        "method": label, "seed": seed, "config_path": config_path,
        "wall_clock_episode_s": wall_s, "peak_memory_mb": peak_bytes / (1024.0 ** 2),
        "safety_step_limit_hit": steps >= 10000,
    })
    generation: List[Dict] = []
    rejection: List[Dict] = []
    commits: List[Dict] = []
    if env.sim.joint_plan_generator is not None:
        generation = [{"method": label, "seed": seed, **row} for row in env.sim.joint_plan_generator.generation_log]
        rejection = [{"method": label, "seed": seed, **row} for row in env.sim.joint_plan_generator.rejection_log]
        commits = [{"method": label, "seed": seed, **row} for row in env.sim.joint_plan_generator.commit_log]
    transitions = [{"method": label, "seed": seed, **row} for row in env.transition_log]
    extra: Dict[str, List[Dict[str, Any]]] = {
        "plan": [], "slot": [], "feature": [], "rejection": [],
        "commit": [], "return_guard": [], "mask_audit": [],
    }
    if env.sim.joint_plan_generator is not None:
        generator = env.sim.joint_plan_generator
        sources = {
            "plan": generator.plan_log, "slot": generator.slot_log,
            "feature": generator.feature_log, "rejection": generator.rejection_log,
            "commit": generator.commit_log, "return_guard": generator.return_guard_log,
            "mask_audit": generator.mask_audit_log,
        }
        for key, source in sources.items():
            extra[key] = [{"method": label, "seed": seed, **row} for row in source]
    return metric, generation, selected_rows, rejection + commits, transitions, extra


def run_group(methods: Iterable[str], seeds: Iterable[int]):
    metrics: List[Dict] = []
    generation: List[Dict] = []
    selected: List[Dict] = []
    events: List[Dict] = []
    transitions: List[Dict] = []
    extra = {key: [] for key in ("plan", "slot", "feature", "rejection", "commit", "return_guard", "mask_audit")}
    for label in methods:
        for seed in seeds:
            result = run_episode(label, METHOD_CONFIGS[label], int(seed))
            metrics.append(result[0]); generation.extend(result[1]); selected.extend(result[2]); events.extend(result[3]); transitions.extend(result[4])
            for key in extra:
                extra[key].extend(result[5][key])
            print(f"{label} seed={seed} completion={result[0]['completion_rate']:.3f} valid={result[0].get('mean_valid_plans', 0):.2f}")
    return tuple(pd.DataFrame(rows) for rows in (metrics, generation, selected, events, transitions, *extra.values()))


def run_k_sensitivity(seeds: Iterable[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for k in (8, 16, 24, 32):
        path = f"configs/scenario_60node_stage4_joint_k{k}.yaml"
        for seed in seeds:
            metric, *_ = run_episode(f"K{k}", path, int(seed))
            metric["K"] = k
            rows.append(metric)
    return pd.DataFrame(rows)


def run_guard_sensitivity(seeds: Iterable[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for token, ratio in (("025", 0.25), ("050", 0.50), ("075", 0.75), ("100", 1.00)):
        path = f"configs/scenario_60node_stage4_joint_guard{token}.yaml"
        for seed in seeds:
            metric, *_ = run_episode(f"Guard-{ratio:.2f}", path, int(seed))
            metric["guard_ratio"] = ratio
            rows.append(metric)
    return pd.DataFrame(rows)


def _save_summary(metrics: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    numeric = metrics.select_dtypes(include=[np.number]).columns.tolist()
    summary = metrics.groupby("method", as_index=False)[numeric].agg(["mean", "std"])
    summary.columns = ["_".join(part for part in col if part) for col in summary.columns.to_flat_index()]
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def _plot_placeholder(ax, title: str, message: str = "No observations"):
    ax.set_title(title); ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes); ax.set_axis_off()


def make_plots(metrics: pd.DataFrame, generation: pd.DataFrame, selected: pd.DataFrame, k_df: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("01_valid_plan_distribution.png", "Valid JointPlan count", generation, "retained_plans", "hist"),
        ("02_generated_plan_types.png", "Generated plan types", generation, "type_I_generated", "types"),
        ("03_selected_plan_types.png", "Selected plan types", selected, "plan_type", "count"),
        ("04_padding_rate.png", "Padding slots", generation, "padding_slots", "bar"),
        ("05_return_guard_rejection.png", "Return Guard rejection", metrics, "return_guard_rejection_rate", "bar"),
        ("06_system_metrics.png", "J0-J3 completion rate", metrics, "completion_rate", "bar"),
        ("07_wait_return_tradeoff.png", "Recovery wait vs return distance", metrics, "tradeoff", "scatter"),
        ("08_k_time_relation.png", "Candidate K vs generation time", k_df, "mean_plan_generation_time_ms", "k"),
        ("09_generation_timing.png", "Mean/P95 generation time", metrics, "timing", "timing"),
        ("10_predicted_realized_wait.png", "Predicted vs realized recovery wait", selected, "wait", "scatter_wait"),
    ]
    for filename, title, frame, column, kind in specs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if frame.empty:
            _plot_placeholder(ax, title)
        elif kind == "hist" and column in frame:
            ax.hist(frame[column].dropna(), bins=range(1, 27)); ax.set_title(title); ax.set_xlabel(column)
        elif kind == "types":
            cols = [c for c in ("type_I_generated", "type_II_generated", "type_III_generated") if c in frame]
            frame[cols].sum().plot(kind="bar", ax=ax, title=title)
        elif kind == "count" and column in frame:
            frame[column].value_counts().plot(kind="bar", ax=ax, title=title)
        elif kind == "bar" and column in frame:
            frame.groupby("method")[column].mean().plot(kind="bar", ax=ax, title=title)
        elif kind == "scatter" and {"mean_uav_recovery_wait_s", "uav_return_distance_m"}.issubset(frame.columns):
            for method, group in frame.groupby("method"):
                ax.scatter(group["mean_uav_recovery_wait_s"], group["uav_return_distance_m"], label=method)
            ax.set_title(title); ax.set_xlabel("recovery wait (s)"); ax.set_ylabel("return distance (m)"); ax.legend(fontsize=7)
        elif kind == "k" and {"K", column}.issubset(frame.columns):
            frame.groupby("K")[column].mean().plot(marker="o", ax=ax, title=title)
        elif kind == "timing" and {"mean_plan_generation_time_ms", "p95_plan_generation_time_ms"}.issubset(frame.columns):
            frame.groupby("method")[["mean_plan_generation_time_ms", "p95_plan_generation_time_ms"]].mean().plot(kind="bar", ax=ax, title=title)
        elif kind == "scatter_wait" and {"predicted_recovery_wait", "predicted_cav_wait"}.issubset(frame.columns):
            ax.scatter(frame["predicted_recovery_wait"], frame["predicted_cav_wait"], alpha=0.5); ax.set_title(title)
        else:
            _plot_placeholder(ax, title)
        fig.tight_layout(); fig.savefig(plot_dir / filename, dpi=160); plt.close(fig)


def write_required_outputs(frames, k_df: pd.DataFrame, out_dir: Path) -> None:
    metrics, generation, selected, events, transitions, plan, slot, feature, rejection, commit, return_guard, mask_audit = frames
    method_files = {
        "Legacy-C3": "legacy_c3_results.csv", "J0-Service": "j0_service_only_results.csv",
        "J1-Rendezvous": "j1_service_rendezvous_results.csv",
        "J2-Full-Guard": "j2_full_guarded_results.csv",
        "J3-Full-NoGuard": "j3_full_no_guard_results.csv",
    }
    for method, filename in method_files.items():
        metrics[metrics["method"] == method].to_csv(out_dir / filename, index=False)
    paired = metrics.pivot(index="seed", columns="method")
    paired.columns = [f"{metric}__{method}" for metric, method in paired.columns]
    paired = paired.reset_index()
    for metric in ("completion_rate", "uav_return_distance_m", "mean_uav_recovery_wait_s", "safe_only_decision_rate"):
        left, right = f"{metric}__J2-Full-Guard", f"{metric}__J3-Full-NoGuard"
        if left in paired and right in paired:
            paired[f"{metric}__J2_minus_J3"] = paired[left] - paired[right]
    paired.to_csv(out_dir / "paired_comparison.csv", index=False)
    plan.to_csv(out_dir / "joint_plan_log.csv", index=False)
    slot.to_csv(out_dir / "joint_plan_slot_log.csv", index=False)
    feature.to_csv(out_dir / "joint_plan_feature_log.csv", index=False)
    rejection.to_csv(out_dir / "joint_plan_rejection_log.csv", index=False)
    commit.to_csv(out_dir / "joint_plan_commit_log.csv", index=False)
    commit[commit.get("success", pd.Series(dtype=bool)).astype(str).str.lower() == "false"].to_csv(
        out_dir / "joint_plan_rollback_log.csv", index=False
    )
    return_guard.to_csv(out_dir / "return_guard_log.csv", index=False)
    mask_audit.to_csv(out_dir / "mask_audit.csv", index=False)
    if not k_df.empty:
        k_df.to_csv(out_dir / "candidate_size_sensitivity.csv", index=False)
    test_results = out_dir / "unit_test_results.csv"
    if test_results.exists():
        tests = pd.read_csv(test_results)
        tests[tests["module"].astype(str).str.contains("stage4_micro")].to_csv(
            out_dir / "micro_scenario_results.csv", index=False
        )
    training_curve = out_dir / "short_training" / "stage4_joint_short_train_seed0" / "training_curve.csv"
    if training_curve.exists():
        pd.read_csv(training_curve).to_csv(out_dir / "short_training_log.csv", index=False)
    grouped = metrics.groupby("method").mean(numeric_only=True)
    j2 = grouped.loc["J2-Full-Guard"] if "J2-Full-Guard" in grouped.index else pd.Series(dtype=float)
    j3 = grouped.loc["J3-Full-NoGuard"] if "J3-Full-NoGuard" in grouped.index else pd.Series(dtype=float)
    summary = {
        "schema_version": 1, "observation_dim": 1025, "action_dim": 25,
        "joint_feature_dim": 38, "main_K": 24,
        "methods": grouped.reset_index().to_dict("records"),
        "acceptance": {
            "safe_plan_always_present": bool(mask_audit.get("safe_present", pd.Series([False])).all()),
            "safe_plan_always_masked": bool(mask_audit.get("safe_mask", pd.Series([False])).all()),
            "padding_mask_violations": int(mask_audit.get("padding_mask_violation_count", pd.Series([0])).sum()),
            "slot_index_violations": int(mask_audit.get("slot_index_violation_count", pd.Series([0])).sum()),
            "all_features_finite": bool(mask_audit.get("finite_feature_matrix", pd.Series([False])).all()),
            "commit_failures": int(metrics.get("commit_failure_count", pd.Series([0])).sum()),
            "rollbacks": int(metrics.get("rollback_count", pd.Series([0])).sum()),
            "stale_cache_count": int(metrics.get("stale_plan_cache_count", pd.Series([0])).sum()),
            "mask_inconsistency_count": int(metrics.get("mask_inconsistency_count", pd.Series([0])).sum()),
            "j2_return_distance_reduction_vs_j3_m": float(j3.get("uav_return_distance_m", 0.0) - j2.get("uav_return_distance_m", 0.0)),
            "j2_safe_only_increase_vs_j3": float(j2.get("safe_only_decision_rate", 0.0) - j3.get("safe_only_decision_rate", 0.0)),
            "j2_mean_generation_time_ms": float(j2.get("mean_plan_generation_time_ms", 0.0)),
            "j2_p95_generation_time_ms": float(j2.get("p95_plan_generation_time_ms", 0.0)),
        },
    }
    if test_results.exists():
        tests = pd.read_csv(test_results)
        summary["tests"] = {
            "total": int(len(tests)),
            "passed": int((tests["status"] == "passed").sum()),
            "failed": int((tests["status"] != "passed").sum()),
            "stage4_unit": int(tests["module"].astype(str).str.contains("stage4_joint_plan").sum()),
            "micro_scenarios": int(tests["module"].astype(str).str.contains("stage4_micro").sum()),
        }
    if training_curve.exists():
        training = pd.read_csv(training_curve)
        numeric = training.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        summary["short_training"] = {
            "transitions": int(training["step"].max()),
            "gradient_updates": int(training["gradient_updates"].max()),
            "replay_transitions": int(training["replay_transitions"].max()),
            "all_numeric_finite": bool(np.isfinite(numeric).all()),
            "checkpoint_obs_dim": 1025, "checkpoint_action_dim": 25,
            "checkpoint_candidate_dim": 38, "replay_schema_version": 2,
        }
    if not k_df.empty:
        summary["K_sensitivity"] = k_df.groupby("K").mean(numeric_only=True).reset_index().to_dict("records")
    guard_path = out_dir / "return_guard_sensitivity_3seed.csv"
    if guard_path.exists():
        guard = pd.read_csv(guard_path)
        summary["return_guard_sensitivity"] = guard.groupby("guard_ratio").mean(numeric_only=True).reset_index().to_dict("records")
    with open(out_dir / "stage4_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 structural evaluation")
    parser.add_argument("--mode", choices=("smoke", "formal", "k", "guard", "all"), default="all")
    parser.add_argument("--out-dir", default="results/stage4_joint_plan_generation")
    parser.add_argument("--smoke-seeds", type=int, default=3)
    parser.add_argument("--formal-seeds", type=int, default=20)
    parser.add_argument("--k-seeds", type=int, default=3)
    parser.add_argument("--guard-seeds", type=int, default=3)
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    if args.mode in {"smoke", "all"}:
        smoke = run_group(("J0-Service", "J1-Rendezvous", "J2-Full-Guard", "J3-Full-NoGuard"), range(args.smoke_seeds))
        smoke[0].to_csv(out_dir / "smoke_3seed_metrics.csv", index=False)
        frames.append(smoke)
    if args.mode in {"formal", "all"}:
        formal = run_group(METHOD_CONFIGS, range(args.formal_seeds))
        formal[0].to_csv(out_dir / "formal_20seed_episode_metrics.csv", index=False)
        frames.append(formal)
    k_df = pd.DataFrame()
    if args.mode in {"k", "all"}:
        k_df = run_k_sensitivity(range(args.k_seeds)); k_df.to_csv(out_dir / "k_sensitivity_3seed.csv", index=False)
    if args.mode in {"guard", "all"}:
        guard_df = run_guard_sensitivity(range(args.guard_seeds))
        guard_df.to_csv(out_dir / "return_guard_sensitivity_3seed.csv", index=False)
    if frames:
        combined = tuple(pd.concat([group[i] for group in frames], ignore_index=True) for i in range(12))
        names = (
            "episode_metrics.csv", "joint_plan_generation.csv", "selected_plans.csv",
            "commit_and_rejection_log.csv", "transition_log.csv", "joint_plan_log.csv",
            "joint_plan_slot_log.csv", "joint_plan_feature_log.csv", "joint_plan_rejection_log.csv",
            "joint_plan_commit_log.csv", "return_guard_log.csv", "mask_audit.csv",
        )
        for frame, name in zip(combined, names): frame.to_csv(out_dir / name, index=False)
        _save_summary(combined[0], out_dir)
        if k_df.empty and (out_dir / "k_sensitivity_3seed.csv").exists():
            k_df = pd.read_csv(out_dir / "k_sensitivity_3seed.csv")
        write_required_outputs(combined, k_df, out_dir)
        make_plots(combined[0], combined[1], combined[2], k_df, out_dir)


if __name__ == "__main__":
    main()
