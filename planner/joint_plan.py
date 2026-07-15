from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dynamic_patrol_marl.env.cav_controller import GroundCandidate
from dynamic_patrol_marl.planner.sortie_generator import Sortie


PLAN_SAFE = "safe"
PLAN_TYPE_I = "type_I_service"
PLAN_TYPE_II = "type_II_rendezvous"
PLAN_TYPE_III = "type_III_reroute"


@dataclass
class JointPlan:
    plan_id: str
    slot_index: int
    plan_type: str
    ground_candidate: GroundCandidate
    sortie: Optional[Sortie] = None
    insertion_index: Optional[int] = None
    insertion_evaluation: Optional[Dict[str, Any]] = None
    cav_route: List[int] = field(default_factory=list)
    cav_target_node: Optional[int] = None
    cav_wait_until: Optional[float] = None
    uav_route: List[int] = field(default_factory=list)
    recovery_schedule: List[Dict[str, Any]] = field(default_factory=list)
    task_ids: List[int] = field(default_factory=list)
    uav_id: Optional[str] = None
    affected_recovery_ids: List[str] = field(default_factory=list)
    task_benefit: float = 0.0
    predicted_aoi_reduction: float = 0.0
    predicted_completion_time: float = 0.0
    predicted_uav_energy: float = 0.0
    predicted_uav_return_energy: float = 0.0
    predicted_uav_return_distance: float = 0.0
    predicted_uav_return_time: float = 0.0
    predicted_battery_slack: float = 0.0
    predicted_cav_distance: float = 0.0
    predicted_cav_time: float = 0.0
    predicted_cav_wait: float = 0.0
    predicted_detour_distance: float = 0.0
    predicted_detour_ratio: float = 0.0
    predicted_recovery_wait: float = 0.0
    predicted_existing_queue_delay: float = 0.0
    predicted_hover_energy: float = 0.0
    predicted_rendezvous_slack: float = 0.0
    predicted_min_recovery_slack: float = 0.0
    queue_length_after: int = 0
    route_switch_flag: bool = False
    reroute_flag: bool = False
    ground_already_committed: bool = False
    features: List[float] = field(default_factory=list)
    feasible: bool = True
    mask: bool = True
    infeasible_reasons: List[str] = field(default_factory=list)
    pre_score: float = 0.0
    generation_time_ms: float = 0.0
    evaluation_time_ms: float = 0.0
    state_fingerprint: Tuple[Any, ...] = field(default_factory=tuple, repr=False)
    created_at: float = 0.0
    schedule_versions: Tuple[int, ...] = field(default_factory=tuple)

    @property
    def is_safe(self) -> bool:
        return self.plan_type == PLAN_SAFE

    def to_record(self) -> Dict[str, Any]:
        record = asdict(self)
        record.pop("state_fingerprint", None)
        record["ground_action"] = self.ground_candidate.action_type
        record["task_ids"] = "|".join(str(x) for x in self.task_ids)
        record["cav_route"] = "|".join(str(x) for x in self.cav_route)
        record["uav_route"] = "|".join(str(x) for x in self.uav_route)
        record["infeasible_reasons"] = "|".join(self.infeasible_reasons)
        record.pop("ground_candidate", None)
        record.pop("sortie", None)
        record.pop("insertion_evaluation", None)
        return record


@dataclass(frozen=True)
class JointPlanCache:
    decision_time: float
    decision_epoch_id: int
    state_fingerprint: Tuple[Any, ...]
    plan_slots: Tuple[Optional[JointPlan], ...]
    feature_matrix: np.ndarray
    action_mask: np.ndarray
    schedule_versions: Tuple[int, ...]
    event_sequence_id: int
    generation_time_ms: float


class JointPlanFeatureBuilder:
    """The single authoritative 38-feature schema for stage four."""

    FEATURE_NAMES = (
        "normalized_task_count", "priority_sum", "maximum_priority",
        "maximum_urgency", "minimum_deadline_slack", "predicted_aoi_reduction",
        "task_coverage_gain", "sortie_duration", "outbound_and_service_energy",
        "return_distance", "return_time", "return_energy", "total_uav_energy",
        "battery_slack", "cav_travel_time", "cav_travel_distance",
        "detour_distance", "detour_ratio", "route_switch_flag", "ground_action_type",
        "recovery_wait", "cav_wait", "existing_queue_delay", "hover_energy",
        "insertion_cost", "rendezvous_slack", "minimum_recovery_slack",
        "queue_length", "locked_prefix_length", "idle_uav_ratio", "pending_task_count",
        "high_priority_task_count", "event_type", "instantaneous_dispatch_flag",
        "is_safe", "is_type_I", "is_type_II", "is_type_III",
    )

    def __init__(self, simulator: Any):
        self.sim = simulator
        cfg = simulator.cfg.get("joint_plan_features", {})
        self.distance_scale = float(cfg.get("distance_scale_m", 3600.0))
        self.time_scale = float(cfg.get("time_scale_s", 600.0))
        self.energy_scale = float(cfg.get("energy_scale", simulator.cfg["agents"].get("drone_battery_capacity", 1800.0)))
        self.task_scale = float(cfg.get("task_scale", 10.0))
        self.clip_low = float(cfg.get("clip_low", -5.0))
        self.clip_high = float(cfg.get("clip_high", 5.0))
        self.clipped_values = 0
        self.total_values = 0

    @property
    def dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def schema(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "feature_dim": self.dim,
            "feature_names": list(self.FEATURE_NAMES),
            "scales": {
                "distance_m": self.distance_scale,
                "time_s": self.time_scale,
                "energy": self.energy_scale,
                "tasks": self.task_scale,
            },
            "clip": [self.clip_low, self.clip_high],
            "missing_value": 0.0,
        }

    def build(self, plan: JointPlan, context: Optional[Dict[str, Any]] = None) -> List[float]:
        context = context or {}
        sim = self.sim
        tasks = [sim.task_pool.by_id[x] for x in plan.task_ids if x in sim.task_pool.by_id]
        priorities = [float(task.priority) for task in tasks]
        slacks = [float(task.deadline - sim.now) for task in tasks]
        urgency = [1.0 / max(1.0, slack) for slack in slacks]
        ground_code = {"keep_route": 0.0, "wait": 1.0 / 3.0, "move_to": 2.0 / 3.0, "reroute": 1.0}.get(
            plan.ground_candidate.action_type, 0.0
        )
        locked = sim.recovery_scheduler.locked_prefix_length(sim.recovery_queue)
        pending = len(sim.task_pool.active_tasks())
        high_threshold = float(sim.cfg.get("simulation", {}).get("high_priority_threshold", 8))
        high_count = sum(float(task.priority) >= high_threshold for task in sim.task_pool.active_tasks())
        outbound_energy = max(0.0, plan.predicted_uav_energy - plan.predicted_uav_return_energy)
        insertion_cost = 0.0
        if plan.insertion_evaluation:
            insertion_cost = float(plan.insertion_evaluation.get("insertion_cost", 0.0))
        raw = [
            len(tasks) / max(1.0, float(sim.cfg.get("sortie", {}).get("max_tasks_per_sortie", 3))),
            sum(priorities) / 36.0,
            max(priorities, default=0.0) / 12.0,
            max(urgency, default=0.0) * self.time_scale,
            min(slacks, default=0.0) / self.time_scale,
            plan.predicted_aoi_reduction / self.time_scale,
            plan.ground_candidate.task_coverage_improvement / self.distance_scale,
            (float(plan.sortie.drone_time_s) if plan.sortie else 0.0) / self.time_scale,
            outbound_energy / self.energy_scale,
            plan.predicted_uav_return_distance / self.distance_scale,
            plan.predicted_uav_return_time / self.time_scale,
            plan.predicted_uav_return_energy / self.energy_scale,
            plan.predicted_uav_energy / self.energy_scale,
            plan.predicted_battery_slack / self.energy_scale,
            plan.predicted_cav_time / self.time_scale,
            plan.predicted_cav_distance / self.distance_scale,
            plan.predicted_detour_distance / self.distance_scale,
            plan.predicted_detour_ratio,
            float(plan.route_switch_flag),
            ground_code,
            plan.predicted_recovery_wait / self.time_scale,
            plan.predicted_cav_wait / self.time_scale,
            plan.predicted_existing_queue_delay / self.time_scale,
            plan.predicted_hover_energy / self.energy_scale,
            insertion_cost / self.time_scale,
            plan.predicted_rendezvous_slack / self.time_scale,
            plan.predicted_min_recovery_slack / self.time_scale,
            plan.queue_length_after / max(1.0, len(sim.drones)),
            locked / max(1.0, len(sim.drones)),
            len(sim.idle_drones_on_vehicle()) / max(1.0, len(sim.drones)),
            pending / self.task_scale,
            high_count / self.task_scale,
            float(context.get("event_type_code", 0.0)),
            float(bool(context.get("instantaneous_dispatch", False))),
            float(plan.plan_type == PLAN_SAFE),
            float(plan.plan_type == PLAN_TYPE_I),
            float(plan.plan_type == PLAN_TYPE_II),
            float(plan.plan_type == PLAN_TYPE_III),
        ]
        values = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0.0, posinf=self.clip_high, neginf=self.clip_low)
        self.clipped_values += int(np.count_nonzero((values < self.clip_low) | (values > self.clip_high)))
        self.total_values += int(values.size)
        return np.clip(values, self.clip_low, self.clip_high).astype(np.float32).tolist()


class JointPlanGenerator:
    """Deterministic, side-effect-free candidate generation and atomic commit."""

    def __init__(self, simulator: Any):
        self.sim = simulator
        self.cfg = dict(simulator.cfg.get("joint_plan", {}))
        self.feature_builder = JointPlanFeatureBuilder(simulator)
        self.non_safe_slots = int(self.cfg.get("max_non_safe_plans", 24))
        self.action_slots = 1 + self.non_safe_slots
        self.max_ground = int(self.cfg.get("max_ground_candidates", 4))
        self.max_sorties_per_ground = int(self.cfg.get("max_sorties_per_ground", 6))
        self.max_insertions = int(self.cfg.get("max_insertions_per_pair", 1))
        quotas = self.cfg.get("type_quotas", {})
        self.quotas = {
            PLAN_TYPE_I: int(quotas.get("type_I", 12)),
            PLAN_TYPE_II: int(quotas.get("type_II", 8)),
            PLAN_TYPE_III: int(quotas.get("type_III", 4)),
        }
        self.return_cfg = dict(simulator.cfg.get("uav_return_guard", {}))
        self.cost_cfg = dict(simulator.cfg.get("joint_plan_cost", {}))
        self.rejection_log: List[Dict[str, Any]] = []
        self.generation_log: List[Dict[str, Any]] = []
        self.commit_log: List[Dict[str, Any]] = []
        self.plan_log: List[Dict[str, Any]] = []
        self.slot_log: List[Dict[str, Any]] = []
        self.feature_log: List[Dict[str, Any]] = []
        self.return_guard_log: List[Dict[str, Any]] = []
        self.mask_audit_log: List[Dict[str, Any]] = []
        self.generation_times_ms: List[float] = []
        self.evaluation_times_ms: List[float] = []
        self.return_guard_evaluations = 0
        self.return_guard_rejections = 0
        self.commit_failures = 0
        self.rollback_count = 0
        self.stale_cache_count = 0
        self.mask_inconsistency_count = 0
        self.selected_type_counts = {PLAN_SAFE: 0, PLAN_TYPE_I: 0, PLAN_TYPE_II: 0, PLAN_TYPE_III: 0}
        self.selected_ground_only = 0
        self.selected_joint_launch_ground = 0
        self.candidate_truncation_count = 0

    def state_fingerprint(self) -> Tuple[Any, ...]:
        tasks = tuple(
            (int(task.id), str(task.status), task.assigned_to)
            for task in self.sim.tasks
        )
        return (self.sim.ground_controller.state_fingerprint(), tasks)

    def schedule_versions(self) -> Tuple[int, ...]:
        return (
            int(self.sim.vehicle.schedule_version),
            int(self.sim.recovery_schedule_version),
            *(int(drone.schedule_version) for drone in self.sim.drones),
        )

    def _ground_candidates(self, context: Dict[str, Any]) -> List[GroundCandidate]:
        old_serial = self.sim.ground_controller._candidate_serial
        candidates = self.sim.ground_controller.generate_ground_candidates(context.get("next_event_time"))
        self.sim.ground_controller._candidate_serial = old_serial
        feasible = [candidate for candidate in candidates if candidate.feasible]
        keep = next((candidate for candidate in feasible if candidate.action_type == "keep_route"), candidates[0])
        if context.get("ground_action_committed_at_timestamp", False):
            return [keep]
        selected = [keep]
        for action_type in ("wait", "move_to", "reroute"):
            group = [candidate for candidate in feasible if candidate.action_type == action_type]
            group.sort(key=lambda c: (round(float(c.score), 12), c.target_node if c.target_node is not None else -1, c.candidate_id))
            if group:
                selected.append(group[0])
        return selected[: self.max_ground]

    def _sorties(self, context: Optional[Dict[str, Any]] = None) -> List[Sortie]:
        sim = self.sim
        context = context or {}
        sim.task_pool.update_release_and_expiry(sim.now)
        available = sim.task_pool.available_tasks(sim.now)
        drones = sim.idle_drones_on_vehicle()
        if context.get("ground_action_committed_at_timestamp", False):
            # A committed route may mark the CAV as being on its first edge even
            # though no physical time has elapsed.  Remaining UAVs are still at
            # the launch node and may be dispatched at this same timestamp.
            drones = [
                drone for drone in sim.drones
                if drone.is_idle() and int(drone.position) == int(sim.vehicle.position)
            ]
        return sim.sortie_generator.generate_all(
            drones, sim.vehicle.position, sim.now, available,
            recovery_anchor_node=sim.recovery_queue_anchor_node(),
            vehicle_queue_time_s=sim.recovery_queue_travel_time_s(),
            allow_recovery_reschedule=sim.dynamic_insertion_enabled,
        )

    def _retarget_sortie(self, sortie: Sortie, ground: GroundCandidate) -> Sortie:
        target = ground.target_node
        if ground.action_type == "keep_route" or target is None or int(target) == int(sortie.recovery_node):
            return deepcopy(sortie)
        trial = deepcopy(sortie)
        if not trial.segments:
            trial.feasible = False
            trial.reason = "missing_return_segment"
            return trial
        final = trial.segments[-1]
        if str(final.get("type")) == "fly":
            origin = int(final.get("from", trial.launch_node))
            old_duration = float(final.get("duration", 0.0))
        else:
            origin = int(final.get("to", trial.launch_node))
            old_duration = 0.0
        new_duration = float(self.sim.network.get_drone_time(origin, int(target)))
        delta = new_duration - old_duration
        if str(final.get("type")) == "fly":
            final["to"] = int(target)
            final["duration"] = new_duration
        else:
            trial.segments.append(
                {"type": "fly", "from": origin, "to": int(target), "duration": new_duration}
            )
        trial.recovery_node = int(target)
        trial.drone_time_s += delta
        trial.finish_time_s += delta
        fly_power = float(self.sim.cfg["agents"].get("drone_fly_power", 1.0))
        trial.energy += delta * fly_power
        trial.battery_at_eta -= delta * fly_power
        trial.feasible = bool(trial.base_feasible and trial.battery_at_eta >= -1e-9)
        trial.reason = "ok" if trial.feasible else "battery_reserve"
        return trial

    def _ground_override(self, ground: GroundCandidate) -> Dict[str, Any]:
        queue, reason = self.sim.ground_controller._trial_queue(ground)
        if reason is not None:
            return {"queue": queue, "first_cav_arrival_time": math.inf}
        arrival = float(ground.predicted_cav_arrival_time)
        if not math.isfinite(arrival):
            arrival = float(self.sim.now)
        return {"queue": queue, "first_cav_arrival_time": arrival}

    def _return_distance(self, sortie: Sortie) -> float:
        if not sortie.segments:
            return 0.0
        final = sortie.segments[-1]
        if str(final.get("type")) != "fly":
            return 0.0
        return float(self.sim.network.euclidean_distance(int(final.get("from", sortie.launch_node)), int(final.get("to", sortie.recovery_node))))

    def _reference_distances(self, sorties: Sequence[Sortie], keep: GroundCandidate) -> Dict[Tuple[str, Tuple[int, ...]], float]:
        refs: Dict[Tuple[str, Tuple[int, ...]], float] = {}
        for sortie in sorties:
            trial = self._retarget_sortie(sortie, keep)
            ranked = self.sim.recovery_scheduler.evaluate_insertions(trial, max_results=1)
            if ranked:
                refs[(trial.drone_id, tuple(trial.task_ids))] = self._return_distance(trial)
        return refs

    def _make_plan(
        self, ground: GroundCandidate, sortie: Optional[Sortie], evaluation: Optional[Dict[str, Any]],
        plan_type: str, fingerprint: Tuple[Any, ...], context: Dict[str, Any], serial: int,
    ) -> JointPlan:
        started = perf_counter()
        diagnostics = evaluation.get("diagnostics", {}) if evaluation else {}
        schedule = deepcopy(evaluation.get("predicted_schedule", [])) if evaluation else deepcopy(ground.predicted_recovery_schedule)
        new_row = None
        if sortie is not None and evaluation is not None:
            sid = evaluation["entry"]["sortie_id"]
            new_row = next((row for row in schedule if str(row.get("sortie_id")) == str(sid)), None)
        tasks = [self.sim.task_pool.by_id[x] for x in sortie.task_ids] if sortie else []
        task_benefit = sum(float(task.priority) for task in tasks)
        completion = max((float(task.deadline) for task in tasks), default=float(self.sim.now))
        return_time = (
            float(sortie.segments[-1].get("duration", 0.0))
            if sortie and sortie.segments and str(sortie.segments[-1].get("type")) == "fly"
            else 0.0
        )
        return_energy = return_time * float(self.sim.cfg["agents"].get("drone_fly_power", 1.0))
        battery_capacity = float(self.sim.cfg["agents"].get("drone_battery_capacity", 1800.0))
        min_slack = float(ground.min_recovery_slack)
        if not math.isfinite(min_slack):
            min_slack = 0.0
        plan = JointPlan(
            plan_id=f"jp:{self.sim.now:.9f}:{serial}", slot_index=-1, plan_type=plan_type,
            ground_candidate=ground, sortie=sortie,
            insertion_index=None if evaluation is None else int(evaluation["best_index"]),
            insertion_evaluation=evaluation, cav_route=list(ground.planned_route),
            cav_target_node=ground.target_node, cav_wait_until=ground.wait_until,
            uav_route=[] if sortie is None else [int(segment.get("to", sortie.launch_node)) for segment in sortie.segments if segment.get("type") == "fly"],
            recovery_schedule=schedule, task_ids=[] if sortie is None else list(sortie.task_ids),
            uav_id=None if sortie is None else str(sortie.drone_id),
            affected_recovery_ids=list(ground.affected_recovery_ids), task_benefit=task_benefit,
            predicted_aoi_reduction=sum(max(0.0, float(self.sim.now) - float(task.release_time)) for task in tasks),
            predicted_completion_time=completion,
            predicted_uav_energy=0.0 if sortie is None else float(sortie.energy),
            predicted_uav_return_energy=return_energy,
            predicted_uav_return_distance=0.0 if sortie is None else self._return_distance(sortie),
            predicted_uav_return_time=return_time,
            predicted_battery_slack=(
                battery_capacity if sortie is None else float(
                    (new_row or {}).get("battery_after_wait", sortie.battery_at_eta)
                    - (new_row or {}).get("battery_reserve", 0.0)
                )
            ),
            predicted_cav_distance=float(ground.cav_travel_distance), predicted_cav_time=float(ground.cav_travel_time),
            predicted_cav_wait=float(diagnostics.get("predicted_cav_wait", ground.predicted_cav_wait)),
            predicted_detour_distance=float(ground.detour_distance), predicted_detour_ratio=float(ground.detour_ratio),
            predicted_recovery_wait=float((new_row or {}).get("predicted_uav_wait", ground.predicted_uav_wait)),
            predicted_existing_queue_delay=float(diagnostics.get("predicted_existing_delay", 0.0)),
            predicted_hover_energy=float(diagnostics.get("predicted_hover_energy", ground.predicted_hover_energy)),
            predicted_rendezvous_slack=float((new_row or {}).get("battery_after_wait", 0.0) - (new_row or {}).get("battery_reserve", 0.0)),
            predicted_min_recovery_slack=min_slack, queue_length_after=len(self.sim.recovery_queue) + int(sortie is not None),
            route_switch_flag=bool(ground.route_changed), reroute_flag=ground.action_type == "reroute",
            ground_already_committed=bool(context.get("ground_action_committed_at_timestamp", False)),
            state_fingerprint=fingerprint, created_at=float(self.sim.now), schedule_versions=self.schedule_versions(),
        )
        weights = {
            "task": float(self.cost_cfg.get("weight_task_benefit", 1.0)),
            "uav": float(self.cost_cfg.get("weight_uav_energy", 0.2)),
            "return": float(self.cost_cfg.get("weight_uav_return", 0.4)),
            "cav": float(self.cost_cfg.get("weight_cav", 0.2)),
            "recovery": float(self.cost_cfg.get("weight_recovery", 0.5)),
            "stability": float(self.cost_cfg.get("weight_stability", 0.2)),
        }
        plan.pre_score = (
            -weights["task"] * plan.task_benefit
            + weights["uav"] * plan.predicted_uav_energy / 100.0
            + weights["return"] * plan.predicted_uav_return_distance / 1000.0
            + weights["cav"] * plan.predicted_cav_time / 60.0
            + weights["recovery"] * plan.predicted_recovery_wait / 60.0
            + weights["stability"] * float(plan.route_switch_flag)
        )
        plan.features = self.feature_builder.build(plan, context)
        plan.evaluation_time_ms = (perf_counter() - started) * 1000.0
        self.evaluation_times_ms.append(plan.evaluation_time_ms)
        return plan

    def _guard_allows(self, plan: JointPlan, references: Dict[Tuple[str, Tuple[int, ...]], float]) -> bool:
        if plan.sortie is None or not bool(self.return_cfg.get("enabled", True)):
            return True
        key = (plan.sortie.drone_id, tuple(plan.sortie.task_ids))
        reference = float(references.get(key, plan.predicted_uav_return_distance))
        floor = float(self.return_cfg.get("return_distance_floor_m", 100.0))
        ratio = (plan.predicted_uav_return_distance - reference) / max(reference, floor)
        self.return_guard_evaluations += 1
        maximum = float(self.return_cfg.get("max_return_distance_increase_ratio", 0.75))
        battery_ok = plan.predicted_battery_slack >= float(self.return_cfg.get("minimum_battery_slack", 0.0))
        allowed = ratio <= maximum + 1e-12 and (
            battery_ok or not bool(self.return_cfg.get("enforce_energy_feasibility", True))
        )
        self.return_guard_log.append(
            {
                "time": float(self.sim.now), "plan_id": plan.plan_id,
                "uav_id": plan.uav_id, "task_ids": "|".join(str(x) for x in plan.task_ids),
                "return_distance_m": plan.predicted_uav_return_distance,
                "reference_distance_m": reference, "return_increase_ratio": ratio,
                "maximum_ratio": maximum, "battery_slack": plan.predicted_battery_slack,
                "allowed": bool(allowed),
            }
        )
        if not allowed:
            self.return_guard_rejections += 1
            reason = "return_distance_guard" if ratio > maximum else "minimum_battery_slack"
            plan.feasible = False
            plan.mask = False
            plan.infeasible_reasons.append(reason)
            if bool(self.return_cfg.get("log_rejected_plans", True)):
                self.rejection_log.append({
                    "time": float(self.sim.now), "plan_id": plan.plan_id, "reason": reason,
                    "return_distance_m": plan.predicted_uav_return_distance,
                    "reference_distance_m": reference, "return_increase_ratio": ratio,
                })
        return allowed

    def generate_joint_plans(self, observation_context: Optional[Dict[str, Any]] = None) -> JointPlanCache:
        context = dict(observation_context or {})
        started = perf_counter()
        fingerprint = self.state_fingerprint()
        grounds = self._ground_candidates(context)
        keep = grounds[0]
        sorties = self._sorties(context)
        references = self._reference_distances(sorties, keep)
        pools = {PLAN_TYPE_I: [], PLAN_TYPE_II: [], PLAN_TYPE_III: []}
        serial = 1
        for ground in grounds:
            if ground.action_type != "keep_route":
                ptype = PLAN_TYPE_III if ground.action_type == "reroute" else PLAN_TYPE_II
                pools[ptype].append(self._make_plan(ground, None, None, ptype, fingerprint, context, serial))
                serial += 1
            override = self._ground_override(ground)
            for base_sortie in sorties[: self.max_sorties_per_ground]:
                sortie = self._retarget_sortie(base_sortie, ground)
                if not sortie.feasible:
                    continue
                ranked = self.sim.recovery_scheduler.evaluate_insertions(
                    sortie, ground_plan_override=override, max_results=self.max_insertions
                )
                for evaluation in ranked:
                    ptype = PLAN_TYPE_I if ground.action_type == "keep_route" else (
                        PLAN_TYPE_III if ground.action_type == "reroute" else PLAN_TYPE_II
                    )
                    plan = self._make_plan(ground, sortie, evaluation, ptype, fingerprint, context, serial)
                    serial += 1
                    if self._guard_allows(plan, references):
                        pools[ptype].append(plan)
        for group in pools.values():
            group.sort(key=lambda p: (round(float(p.pre_score), 12), p.uav_id or "", tuple(p.task_ids), p.cav_target_node or -1, p.insertion_index or -1))
        retained: List[JointPlan] = []
        leftovers: List[JointPlan] = []
        for ptype in (PLAN_TYPE_I, PLAN_TYPE_II, PLAN_TYPE_III):
            quota = self.quotas[ptype]
            retained.extend(pools[ptype][:quota])
            leftovers.extend(pools[ptype][quota:])
        if len(retained) < self.non_safe_slots:
            leftovers.sort(key=lambda p: (round(float(p.pre_score), 12), p.plan_type, p.plan_id))
            retained.extend(leftovers[: self.non_safe_slots - len(retained)])
        retained = retained[: self.non_safe_slots]
        safe = self._make_plan(keep, None, None, PLAN_SAFE, fingerprint, context, 0)
        safe.plan_id = f"jp:{self.sim.now:.9f}:safe"
        safe.pre_score = 0.0
        slots: List[Optional[JointPlan]] = [safe, *retained]
        for index, plan in enumerate(slots):
            if plan is not None:
                plan.slot_index = index
        slots.extend([None] * (self.action_slots - len(slots)))
        matrix = np.zeros((self.action_slots, self.feature_builder.dim), dtype=np.float32)
        mask = np.zeros(self.action_slots, dtype=bool)
        for index, plan in enumerate(slots):
            if plan is not None:
                matrix[index] = np.asarray(plan.features, dtype=np.float32)
                mask[index] = bool(plan.feasible and plan.mask)
        mask[0] = True
        elapsed = (perf_counter() - started) * 1000.0
        self.generation_times_ms.append(elapsed)
        generated = 1 + sum(len(group) for group in pools.values())
        if generated > self.action_slots:
            self.candidate_truncation_count += 1
        self.generation_log.append({
            "time": float(self.sim.now), "generated_plans": generated,
            "retained_plans": int(mask.sum()), "padding_slots": int((~mask).sum()),
            "type_I_generated": len(pools[PLAN_TYPE_I]), "type_II_generated": len(pools[PLAN_TYPE_II]),
            "type_III_generated": len(pools[PLAN_TYPE_III]), "generation_time_ms": elapsed,
        })
        epoch = int(context.get("decision_epoch_id", 0))
        for index, plan in enumerate(slots):
            self.slot_log.append(
                {
                    "time": float(self.sim.now), "decision_epoch_id": epoch,
                    "slot_index": index, "plan_id": "" if plan is None else plan.plan_id,
                    "plan_type": "padding" if plan is None else plan.plan_type,
                    "ground_action": "" if plan is None else plan.ground_candidate.action_type,
                    "mask": bool(mask[index]), "padding": plan is None,
                }
            )
            feature_row = {
                "time": float(self.sim.now), "decision_epoch_id": epoch,
                "slot_index": index, "plan_id": "" if plan is None else plan.plan_id,
            }
            feature_row.update(
                {
                    name: float(matrix[index, feature_index])
                    for feature_index, name in enumerate(self.feature_builder.FEATURE_NAMES)
                }
            )
            self.feature_log.append(feature_row)
            if plan is not None:
                plan_row = plan.to_record()
                plan_row.update({"time": float(self.sim.now), "decision_epoch_id": epoch})
                self.plan_log.append(plan_row)
        self.mask_audit_log.append(
            {
                "time": float(self.sim.now), "decision_epoch_id": epoch,
                "action_slots": self.action_slots, "valid_actions": int(mask.sum()),
                "padding_slots": sum(plan is None for plan in slots),
                "safe_present": bool(slots[0] is not None and slots[0].is_safe),
                "safe_mask": bool(mask[0]),
                "padding_mask_violation_count": sum(
                    bool(mask[index]) for index, plan in enumerate(slots) if plan is None
                ),
                "slot_index_violation_count": sum(
                    int(plan.slot_index) != index for index, plan in enumerate(slots) if plan is not None
                ),
                "finite_feature_matrix": bool(np.isfinite(matrix).all()),
            }
        )
        for plan in retained:
            plan.generation_time_ms = elapsed
        return JointPlanCache(
            decision_time=float(self.sim.now), decision_epoch_id=int(context.get("decision_epoch_id", 0)),
            state_fingerprint=fingerprint, plan_slots=tuple(slots), feature_matrix=matrix,
            action_mask=mask, schedule_versions=self.schedule_versions(),
            event_sequence_id=int(context.get("event_sequence_id", 0)), generation_time_ms=elapsed,
        )

    def validate_cache(self, cache: JointPlanCache) -> bool:
        valid = (
            abs(float(cache.decision_time) - float(self.sim.now)) <= 1e-9
            and cache.state_fingerprint == self.state_fingerprint()
            and cache.schedule_versions == self.schedule_versions()
        )
        if not valid:
            self.stale_cache_count += 1
        return valid

    def _snapshot(self) -> Dict[str, Any]:
        sim = self.sim
        controller = sim.ground_controller
        return {
            "vehicle": deepcopy(sim.vehicle.__dict__),
            "drones": [deepcopy(drone.__dict__) for drone in sim.drones],
            "tasks": [deepcopy(task.__dict__) for task in sim.tasks],
            "queue": deepcopy(sim.recovery_queue),
            "recovery_version": int(sim.recovery_schedule_version),
            "rng": deepcopy(sim.rng.bit_generator.state),
            "event_log": deepcopy(sim.event_log), "sortie_log": deepcopy(sim.sortie_log),
            "insertion_log": deepcopy(sim.insertion_log),
            "controller": {
                key: deepcopy(getattr(controller, key)) for key in (
                    "action_log", "candidate_log", "route_change_log", "recovery_schedule_log",
                    "pending_realized_gain", "realized_gains", "total_committed_detour_distance",
                    "last_decision_time", "last_selected_action",
                )
            },
        }

    def _restore(self, snapshot: Dict[str, Any]) -> None:
        sim = self.sim
        sim.vehicle.__dict__.clear(); sim.vehicle.__dict__.update(snapshot["vehicle"])
        for drone, state in zip(sim.drones, snapshot["drones"]):
            drone.__dict__.clear(); drone.__dict__.update(state)
        for task, state in zip(sim.tasks, snapshot["tasks"]):
            task.__dict__.clear(); task.__dict__.update(state)
        sim.recovery_queue[:] = snapshot["queue"]
        sim.recovery_schedule_version = snapshot["recovery_version"]
        sim.rng.bit_generator.state = snapshot["rng"]
        sim.event_log[:] = snapshot["event_log"]
        sim.sortie_log[:] = snapshot["sortie_log"]
        sim.insertion_log[:] = snapshot["insertion_log"]
        for key, value in snapshot["controller"].items():
            setattr(sim.ground_controller, key, value)

    def commit_joint_plan(self, cache: JointPlanCache, action: int) -> Tuple[bool, Optional[JointPlan]]:
        if not self.validate_cache(cache) or action < 0 or action >= self.action_slots or not bool(cache.action_mask[action]):
            self.commit_failures += 1
            return False, None
        plan = cache.plan_slots[int(action)]
        if plan is None:
            self.commit_failures += 1
            return False, None
        if int(plan.slot_index) != int(action):
            self.mask_inconsistency_count += 1
            self.commit_failures += 1
            return False, None
        snapshot = self._snapshot()
        try:
            injection_point = self.cfg.get("inject_commit_failure_at", "")
            if bool(self.cfg.get("inject_commit_failure", False)) or injection_point == "before_commit":
                raise RuntimeError("injected_joint_commit_failure")
            if plan.is_safe:
                self.selected_type_counts[PLAN_SAFE] += 1
                self.commit_log.append({"time": self.sim.now, "plan_id": plan.plan_id, "success": True, "plan_type": plan.plan_type})
                return True, plan
            if not plan.ground_already_committed and not self.sim.ground_controller.commit_cav_action(plan.ground_candidate, validate_fingerprint=True):
                raise RuntimeError("ground_commit_failed")
            if injection_point == "after_ground":
                raise RuntimeError("injected_failure_after_ground_commit")
            if plan.sortie is not None:
                sortie = plan.sortie
                drone = self.sim.get_drone(sortie.drone_id)
                if not drone.is_idle() or drone.position != self.sim.vehicle.position or (
                    not self.sim.vehicle.at_intersection and not plan.ground_already_committed
                ):
                    raise RuntimeError("uav_launch_state_changed")
                if any(self.sim.task_pool.by_id[tid].status != "unassigned" for tid in sortie.task_ids):
                    raise RuntimeError("task_state_changed")
                evaluation = plan.insertion_evaluation or {}
                entry = dict(evaluation["entry"])
                index = int(plan.insertion_index)
                if index < 0 or index > len(self.sim.recovery_queue):
                    raise RuntimeError("insertion_index_invalid")
                locked = self.sim.recovery_scheduler.locked_prefix_length(self.sim.recovery_queue)
                if index < locked:
                    raise RuntimeError("locked_prefix_modified")
                self.sim.recovery_queue.insert(index, entry)
                self.sim.recovery_schedule_version += 1
                schedule_by_id = {str(row["sortie_id"]): row for row in evaluation["predicted_schedule"]}
                for queue_entry in self.sim.recovery_queue:
                    row = schedule_by_id.get(str(queue_entry.get("sortie_id")))
                    if row is not None:
                        for key in ("scheduled_start_time", "scheduled_finish_time", "predicted_uav_wait", "predicted_cav_wait"):
                            queue_entry[key] = float(row[key])
                if injection_point == "after_insertion":
                    raise RuntimeError("injected_failure_after_recovery_insertion")
                self.sim._record_insertion_attempt(sortie, evaluation)
                self.sim.task_pool.mark_assigned(sortie.task_ids, drone.id, self.sim.now)
                drone.assign_sortie(sortie, self.sim.now)
                self.sim.sortie_log.append({"time": self.sim.now, **sortie.to_dict()})
                self.sim.event_log.append({"time": self.sim.now, "event": "launch", "agent": drone.id, "node": sortie.launch_node, "tasks": ";".join(map(str, sortie.task_ids))})
                self.sim.event_log.append({"time": self.sim.now, "event": "recovery_queue_insert", "agent": drone.id, "node": int(sortie.recovery_node), "queue_position": index, "tasks": ";".join(map(str, sortie.task_ids))})
                if plan.ground_candidate.action_type == "keep_route" and not plan.ground_already_committed:
                    self.sim._set_vehicle_next_recovery_target()
            self.selected_type_counts[plan.plan_type] += 1
            if plan.sortie is None:
                self.selected_ground_only += 1
            elif plan.ground_candidate.action_type != "keep_route":
                self.selected_joint_launch_ground += 1
            self.commit_log.append({"time": self.sim.now, "plan_id": plan.plan_id, "success": True, "plan_type": plan.plan_type})
            return True, plan
        except Exception as exc:
            self._restore(snapshot)
            self.commit_failures += 1
            self.rollback_count += 1
            self.commit_log.append({"time": self.sim.now, "plan_id": plan.plan_id, "success": False, "reason": str(exc), "plan_type": plan.plan_type})
            return False, plan

    def select_heuristic(self, cache: JointPlanCache) -> int:
        valid = [plan for plan in cache.plan_slots if plan is not None and plan.mask and plan.feasible]
        if len(valid) <= 1:
            return 0
        selected = min(valid[1:], key=lambda plan: (round(float(plan.pre_score), 12), plan.slot_index))
        return int(selected.slot_index)

    def summary(self) -> Dict[str, Any]:
        generation = np.asarray(self.generation_times_ms, dtype=float)
        evaluation = np.asarray(self.evaluation_times_ms, dtype=float)
        decisions = max(1, len(self.generation_log))
        selections = max(1, sum(self.selected_type_counts.values()))
        generated_by_type = {
            PLAN_TYPE_I: sum(row.get("type_I_generated", 0) for row in self.generation_log),
            PLAN_TYPE_II: sum(row.get("type_II_generated", 0) for row in self.generation_log),
            PLAN_TYPE_III: sum(row.get("type_III_generated", 0) for row in self.generation_log),
        }
        total_generated_typed = max(1, sum(generated_by_type.values()))
        return {
            "mean_generated_plans": float(np.mean([row["generated_plans"] for row in self.generation_log])) if self.generation_log else 0.0,
            "mean_valid_plans": float(np.mean([row["retained_plans"] for row in self.generation_log])) if self.generation_log else 0.0,
            "minimum_valid_plans": min((row["retained_plans"] for row in self.generation_log), default=0),
            "maximum_valid_plans": max((row["retained_plans"] for row in self.generation_log), default=0),
            "safe_plan_rate": self.selected_type_counts[PLAN_SAFE] / selections,
            "safe_only_decision_rate": sum(row.get("retained_plans", 0) == 1 for row in self.generation_log) / decisions,
            "padding_rate": float(np.mean([row.get("padding_slots", 0) / self.action_slots for row in self.generation_log])) if self.generation_log else 0.0,
            "ground_only_plan_rate": self.selected_ground_only / selections,
            "joint_launch_ground_rate": self.selected_joint_launch_ground / selections,
            "type_I_generated_rate": generated_by_type[PLAN_TYPE_I] / total_generated_typed,
            "type_II_generated_rate": generated_by_type[PLAN_TYPE_II] / total_generated_typed,
            "type_III_generated_rate": generated_by_type[PLAN_TYPE_III] / total_generated_typed,
            "type_I_selected_rate": self.selected_type_counts[PLAN_TYPE_I] / selections,
            "type_II_selected_rate": self.selected_type_counts[PLAN_TYPE_II] / selections,
            "type_III_selected_rate": self.selected_type_counts[PLAN_TYPE_III] / selections,
            "return_guard_rejection_rate": self.return_guard_rejections / max(1, self.return_guard_evaluations),
            "commit_failure_count": self.commit_failures, "rollback_count": self.rollback_count,
            "atomic_commit_failure_rate": self.commit_failures / max(1, len(self.commit_log)),
            "rollback_rate": self.rollback_count / max(1, len(self.commit_log)),
            "candidate_truncation_count": self.candidate_truncation_count,
            "stale_plan_cache_count": self.stale_cache_count, "mask_inconsistency_count": self.mask_inconsistency_count,
            "mean_plan_generation_time_ms": float(generation.mean()) if generation.size else 0.0,
            "p95_plan_generation_time_ms": float(np.percentile(generation, 95)) if generation.size else 0.0,
            "mean_plan_evaluation_time_ms": float(evaluation.mean()) if evaluation.size else 0.0,
            "p95_plan_evaluation_time_ms": float(np.percentile(evaluation, 95)) if evaluation.size else 0.0,
            "plan_feature_clipping_rate": self.feature_builder.clipped_values / max(1, self.feature_builder.total_values),
            "joint_plan_decisions": decisions,
        }
