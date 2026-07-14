from __future__ import annotations

from typing import List, Sequence

from .sortie_generator import Sortie


def select_non_conflicting(sorted_candidates: Sequence[Sortie], max_count: int) -> List[Sortie]:
    selected: List[Sortie] = []
    used_drones, used_tasks = set(), set()
    for s in sorted_candidates:
        if not s.feasible:
            continue
        if s.drone_id in used_drones:
            continue
        if any(t in used_tasks for t in s.task_ids):
            continue
        selected.append(s)
        used_drones.add(s.drone_id)
        used_tasks.update(s.task_ids)
        if len(selected) >= max_count:
            break
    return selected


def _tasks(sim, sortie: Sortie):
    return [sim.task_pool.by_id[tid] for tid in sortie.task_ids]


def _max_dispatch_count(sim) -> int:
    return max(1, len(sim.idle_drones_on_vehicle()))


def nearest_task_policy(sim, candidates: List[Sortie]):
    """Nearest-task greedy baseline.

    Rank candidates by the distance/time from the current launch point to the
    closest task in the sortie. This is a deliberately simple spatial rule: it
    ignores task priority and deadline except as tie-breakers.
    """

    def nearest_time(s: Sortie) -> float:
        tasks = _tasks(sim, s)
        if not tasks:
            return float("inf")
        return min(sim.network.get_drone_time(s.launch_node, t.start_node) for t in tasks)

    ranked = sorted(candidates, key=lambda s: (nearest_time(s), s.finish_time_s, s.drone_time_s + s.vehicle_time_s))
    return select_non_conflicting(ranked, max_count=_max_dispatch_count(sim))


def earliest_finish_policy(sim, candidates: List[Sortie]):
    ranked = sorted(candidates, key=lambda s: (s.finish_time_s, s.drone_time_s + s.vehicle_time_s))
    return select_non_conflicting(ranked, max_count=max(1, len(sim.idle_drones_on_vehicle())))


def priority_greedy_policy(sim, candidates: List[Sortie]):
    """Priority-based greedy baseline, independent from deadline pressure."""

    def priority_score(s: Sortie) -> float:
        tasks = _tasks(sim, s)
        if not tasks:
            return -float("inf")
        max_priority = max(t.priority for t in tasks)
        priority_sum = sum(t.priority for t in tasks)
        return 1000.0 * max_priority + 10.0 * priority_sum + len(tasks) - 0.005 * (s.drone_time_s + s.vehicle_time_s)

    ranked = sorted(candidates, key=priority_score, reverse=True)
    return select_non_conflicting(ranked, max_count=_max_dispatch_count(sim))


def priority_deadline_policy(sim, candidates: List[Sortie]):
    def priority_deadline_score(s: Sortie) -> float:
        tasks = _tasks(sim, s)
        urgency = sum(t.priority / max(1.0, t.deadline - sim.now) for t in tasks)
        return urgency * 1000.0 + len(tasks) * 3.0 - 0.005 * (s.drone_time_s + s.vehicle_time_s)

    ranked = sorted(candidates, key=priority_deadline_score, reverse=True)
    return select_non_conflicting(ranked, max_count=_max_dispatch_count(sim))


def edf_policy(sim, candidates: List[Sortie]):
    """Earliest-deadline-first baseline over currently released tasks only."""

    def edf_key(s: Sortie):
        tasks = _tasks(sim, s)
        if not tasks:
            return (float("inf"), 0.0, float("inf"))
        earliest_deadline = min(t.deadline for t in tasks)
        max_priority = max(t.priority for t in tasks)
        return (earliest_deadline, -max_priority, s.finish_time_s)

    ranked = sorted(candidates, key=edf_key)
    return select_non_conflicting(ranked, max_count=_max_dispatch_count(sim))


def score_based_policy(sim, candidates: List[Sortie]):
    """Choose the single legal candidate sortie with the highest hand-crafted score."""

    ranked = sorted((s for s in candidates if s.feasible), key=lambda s: s.score, reverse=True)
    return ranked[:1]


def rolling_horizon_policy(sim, candidates: List[Sortie]):
    ranked = sorted(candidates, key=lambda s: s.score, reverse=True)
    return select_non_conflicting(ranked, max_count=_max_dispatch_count(sim))


def static_reopt_policy(sim, candidates: List[Sortie]):
    # Static re-optimization is over currently released tasks only; future task
    # arrivals are not visible to this policy.
    def score(s: Sortie) -> float:
        tasks = _tasks(sim, s)
        deadline_pressure = sum((t.priority + 1.0) / max(30.0, t.deadline - sim.now) for t in tasks)
        spatial_saving = max(0, len(tasks) - 1) * 5.0
        return 900.0 * deadline_pressure + spatial_saving - 0.006 * max(s.drone_time_s, s.vehicle_time_s)
    ranked = sorted(candidates, key=score, reverse=True)
    return select_non_conflicting(ranked, max_count=_max_dispatch_count(sim))


def random_masked_policy(sim, candidates: List[Sortie]):
    """Random policy over the legal action mask: wait plus feasible candidates."""

    feasible = [s for s in candidates if s.feasible]
    action = int(sim.rng.integers(0, len(feasible) + 1))
    if action == 0:
        return []
    return [feasible[action - 1]]


def wait_policy(sim, candidates: List[Sortie]):
    return []


def get_policy(name: str):
    name = name.lower()
    if name in {"nearest", "nearest_task", "nearest-task", "nearest_task_greedy"}:
        return nearest_task_policy
    if name in {"earliest_finish", "finish_time"}:
        return earliest_finish_policy
    if name in {"priority", "priority_greedy", "priority-based", "priority_based"}:
        return priority_greedy_policy
    if name in {"priority_deadline", "priority_edf"}:
        return priority_deadline_policy
    if name in {"edf", "deadline", "deadline_greedy"}:
        return edf_policy
    if name in {"score", "score_based", "score-based"}:
        return score_based_policy
    if name in {"rolling", "rolling_horizon"}:
        return rolling_horizon_policy
    if name in {"static", "static_reopt", "static_reoptimization"}:
        return static_reopt_policy
    if name in {"random", "random_masked", "random_masked_policy"}:
        return random_masked_policy
    if name == "wait":
        return wait_policy
    raise ValueError(f"Unknown policy: {name}")
