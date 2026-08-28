from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import math


@dataclass
class EpisodeMetrics:
    seed: int = 0
    method: str = ""
    scenario: str = ""
    generated_tasks: int = 0
    completed_tasks: int = 0
    expired_tasks: int = 0
    timeout_tasks: int = 0
    completion_rate: float = 0.0
    timeout_rate: float = 0.0
    avg_response_time_s: float = math.nan
    high_priority_response_time_s: float = math.nan
    weighted_aoi: float = 0.0
    total_energy: float = 0.0
    vehicle_wait_time_s: float = 0.0
    drone_wait_time_s: float = 0.0
    drone_utilization: float = 0.0
    tasks_per_hour: float = 0.0
    illegal_actions: int = 0
    avg_tasks_per_sortie: float = 0.0
    num_sorties: int = 0
    uav_energy_per_completed_task: float = math.nan
    mean_uav_recovery_wait_s: float = 0.0
    p95_uav_recovery_wait_s: float = 0.0
    total_uav_hover_time_s: float = 0.0
    total_hover_energy: float = 0.0
    mean_cav_waiting_time_s: float = 0.0
    cav_travel_distance_m: float = 0.0
    recovery_contention_rate: float = 0.0
    non_append_rate: float = 0.0
    mean_insertion_displacement: float = 0.0
    mean_counterfactual_gain: float = 0.0
    locked_queue_rate: float = 0.0
    no_feasible_insertion_rate: float = 0.0
    dynamic_insertion_computation_time_ms: float = 0.0
    recovery_attempts: int = 0
    infeasible_recoveries: int = 0
    minimum_battery_observed: float = 0.0
    cav_detour_distance_m: float = 0.0
    cav_detour_ratio: float = 0.0
    cav_waiting_time_s: float = 0.0
    uav_return_distance_m: float = 0.0
    uav_return_time_s: float = 0.0
    minimum_recovery_slack_s: float = math.inf
    maximum_queue_wait_s: float = 0.0
    keep_route_count: int = 0
    wait_count: int = 0
    move_to_count: int = 0
    reroute_count: int = 0
    non_keep_action_rate: float = 0.0
    route_change_count: int = 0
    route_reversal_count: int = 0
    emergency_override_count: int = 0
    mean_route_commitment_time_s: float = 0.0
    ground_candidate_count: float = 0.0
    ground_action_computation_time_ms: float = 0.0
    mean_predicted_gain_vs_keep: float = 0.0
    mean_realized_gain_vs_keep: float = 0.0
    maximum_cav_detour_ratio: float = 0.0
    mean_generated_plans: float = 0.0
    mean_valid_plans: float = 0.0
    minimum_valid_plans: int = 0
    maximum_valid_plans: int = 0
    safe_plan_rate: float = 0.0
    safe_only_decision_rate: float = 0.0
    padding_rate: float = 0.0
    ground_only_plan_rate: float = 0.0
    joint_launch_ground_rate: float = 0.0
    type_I_generated_rate: float = 0.0
    type_II_generated_rate: float = 0.0
    type_III_generated_rate: float = 0.0
    type_I_selected_rate: float = 0.0
    type_II_selected_rate: float = 0.0
    type_III_selected_rate: float = 0.0
    joint_keep_route_rate: float = 0.0
    joint_wait_rate: float = 0.0
    joint_move_to_rate: float = 0.0
    joint_reroute_rate: float = 0.0
    return_guard_rejection_rate: float = 0.0
    commit_failure_count: int = 0
    path_planning_failure_count: int = 0
    rollback_count: int = 0
    atomic_commit_failure_rate: float = 0.0
    rollback_rate: float = 0.0
    candidate_truncation_count: int = 0
    stale_plan_cache_count: int = 0
    mask_inconsistency_count: int = 0
    mean_plan_generation_time_ms: float = 0.0
    p95_plan_generation_time_ms: float = 0.0
    mean_plan_evaluation_time_ms: float = 0.0
    p95_plan_evaluation_time_ms: float = 0.0
    plan_feature_clipping_rate: float = 0.0
    joint_plan_decisions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_task_metrics(tasks: List[Any], high_priority_threshold: int) -> Dict[str, float]:
    completed = [t for t in tasks if t.status == "done"]
    expired = [t for t in tasks if t.status == "expired"]
    timeouts = [t for t in tasks if t.completion_time is not None and t.completion_time > t.deadline]
    responses = [t.completion_time - t.release_time for t in completed if t.completion_time is not None]
    high_responses = [
        t.completion_time - t.release_time
        for t in completed
        if t.priority >= high_priority_threshold and t.completion_time is not None
    ]
    n = len(tasks)
    return {
        "generated_tasks": n,
        "completed_tasks": len(completed),
        "expired_tasks": len(expired),
        "timeout_tasks": len(timeouts),
        "completion_rate": len(completed) / n if n else 0.0,
        "timeout_rate": (len(expired) + len(timeouts)) / n if n else 0.0,
        "avg_response_time_s": float(sum(responses) / len(responses)) if responses else math.nan,
        "high_priority_response_time_s": float(sum(high_responses) / len(high_responses)) if high_responses else math.nan,
    }
