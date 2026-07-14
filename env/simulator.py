from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Any, Union
import json
import math

import numpy as np
import pandas as pd

from .road_network import RoadNetwork, build_network
from .task_generator import Task, TaskGenerator
from .task_pool import TaskPool
from .vehicle import Vehicle
from .drone import Drone
from .recovery_scheduler import RecoveryInsertionScheduler
from dynamic_patrol_marl.planner.sortie_generator import Sortie, SortieGenerator
from dynamic_patrol_marl.utils.metrics import EpisodeMetrics, summarize_task_metrics


PolicyFn = Callable[["Simulator", List[Sortie]], Union[Sequence[Sortie], Sortie, None]]


class Simulator:
    """Discrete-time simulator for one CAV patrol vehicle with multiple UAVs.

    The implementation deliberately keeps the first-stage MVP simple and auditable:
    - one vehicle acts as launch/recovery/charging platform;
    - multiple drones can be launched from the current vehicle node;
    - each drone sortie may inspect one or multiple dynamic tasks;
    - the vehicle travels on the directed graph to recover drones;
    - hard constraints are handled before dispatch by the candidate sortie generator.
    """

    def __init__(self, cfg: Dict, seed: Optional[int] = None, scenario: str = "small"):
        self.cfg = cfg
        self.seed = int(cfg["project"].get("seed", 42) if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)
        self.scenario = scenario
        self.network = RoadNetwork(cfg)
        self.task_generator = TaskGenerator(cfg, self.network, self.seed)
        self.tasks = self.task_generator.generate_episode_tasks()
        self.task_pool = TaskPool(self.tasks)
        self.sortie_generator = SortieGenerator(cfg, self.network)
        self.vehicle = Vehicle(id="veh0", position=self.network.patrol_center)
        self.vehicle.route_log.append((0.0, self.vehicle.position))
        self.drones = [
            Drone(
                id=f"drone{i}",
                home_vehicle_id=self.vehicle.id,
                position=self.vehicle.position,
                battery_capacity=float(cfg["agents"].get("drone_battery_capacity", 1800.0)),
                battery=float(cfg["agents"].get("drone_battery_capacity", 1800.0)),
            )
            for i in range(int(cfg["agents"].get("num_drones", 2)))
        ]
        self.now = 0.0
        self.task_pool.update_release_and_expiry(self.now)
        self.horizon = float(cfg["simulation"].get("horizon_s", 3600))
        self.dt = float(cfg["simulation"].get("dt_s", 10))
        self.decision_interval = float(cfg["simulation"].get("decision_interval_s", 60))
        self.event_log: List[Dict[str, Any]] = []
        self.sortie_log: List[Dict[str, Any]] = []
        self.recovery_queue: List[Dict[str, Any]] = []
        recovery_cfg = cfg.get("recovery", {})
        self.recovery_rule = str(recovery_cfg.get("rule", cfg.get("recovery_rule", "fifo"))).lower()
        if self.recovery_rule not in {"fifo", "dynamic_insertion"}:
            raise ValueError(f"Unknown recovery rule: {self.recovery_rule}")
        self.recovery_insertion_cfg = cfg.get("recovery_insertion", {})
        self.dynamic_insertion_enabled = self.recovery_rule == "dynamic_insertion" and bool(
            self.recovery_insertion_cfg.get("enabled", True)
        )
        self.recovery_scheduler = RecoveryInsertionScheduler(self)
        self.insertion_log: List[Dict[str, Any]] = []
        self.recovery_wait_records: List[float] = []
        self.insertion_evaluation_count = 0
        self.no_feasible_insertion_count = 0
        self.infeasible_recoveries = 0
        self.minimum_battery_observed = min((d.battery for d in self.drones), default=0.0)
        self.last_step_vehicle_arrivals: List[Tuple[float, int]] = []
        self.cover_last_time = {arc: 0.0 for arc in self.network.arcs}
        self.aoi_integral = 0.0
        self.illegal_actions = 0
        self.reward_total = 0.0
        self.completed_since_last = 0
        self.expired_since_last = 0
        self._last_expired_ids = set()

    def reset(self) -> None:
        self.__init__(self.cfg, self.seed, self.scenario)

    def active_drones(self) -> List[Drone]:
        return [d for d in self.drones if not d.is_idle()]

    def idle_drones_on_vehicle(self) -> List[Drone]:
        if not self.vehicle.at_intersection:
            return []
        return [d for d in self.drones if d.is_idle() and d.position == self.vehicle.position]

    def get_candidates(self, no_multitask: bool = False, no_priority: bool = False) -> List[Sortie]:
        self.task_pool.update_release_and_expiry(self.now)
        available = self.task_pool.available_tasks(self.now)
        max_tasks = 1 if no_multitask else None
        queue_aware_recovery = bool(self.cfg.get("ablation", {}).get("queue_aware_recovery", True))
        recovery_anchor_node = self.recovery_queue_anchor_node() if queue_aware_recovery else self.vehicle.position
        vehicle_queue_time_s = self.recovery_queue_travel_time_s() if queue_aware_recovery else 0.0
        candidates = self.sortie_generator.generate_all(
            self.idle_drones_on_vehicle(),
            self.vehicle.position,
            self.now,
            available,
            max_tasks=max_tasks,
            no_priority=no_priority,
            recovery_anchor_node=recovery_anchor_node,
            vehicle_queue_time_s=vehicle_queue_time_s,
            allow_recovery_reschedule=self.dynamic_insertion_enabled,
        )
        if self.dynamic_insertion_enabled:
            for candidate in candidates:
                evaluation = self.evaluate_insertion(candidate, commit=False)
                if bool(self.recovery_insertion_cfg.get("cache_candidate_evaluation", True)):
                    candidate.insertion_evaluation = evaluation
                candidate.feasible = bool(candidate.base_feasible and evaluation["feasible"])
                candidate.reason = "ok" if candidate.feasible else str(evaluation["diagnostics"]["fallback_reason"])
                self.insertion_evaluation_count += 1
                if not evaluation["feasible"]:
                    self.no_feasible_insertion_count += 1
        return candidates

    def evaluate_insertion(self, candidate_sortie: Sortie, commit: bool = False) -> Dict[str, Any]:
        rule = "dynamic_insertion" if self.dynamic_insertion_enabled else "fifo"
        evaluation = self.recovery_scheduler.evaluate(candidate_sortie, rule=rule)
        if commit and evaluation["feasible"]:
            self.commit_insertion(candidate_sortie, evaluation=evaluation)
        return evaluation

    def commit_insertion(
        self,
        candidate_sortie: Sortie,
        best_index: Optional[int] = None,
        evaluation: Optional[Dict[str, Any]] = None,
    ) -> bool:
        evaluation = evaluation or self.evaluate_insertion(candidate_sortie, commit=False)
        if evaluation.get("state_fingerprint") != self.recovery_scheduler.state_fingerprint():
            evaluation = self.evaluate_insertion(candidate_sortie, commit=False)
        if not evaluation["feasible"]:
            self.infeasible_recoveries += 1
            self._record_insertion_attempt(candidate_sortie, evaluation)
            return False
        index = int(evaluation["best_index"] if best_index is None else best_index)
        if index != int(evaluation["best_index"]):
            return False
        entry = dict(evaluation["entry"])
        self.recovery_queue.insert(index, entry)
        schedule_by_id = {str(row["sortie_id"]): row for row in evaluation["predicted_schedule"]}
        for queue_entry in self.recovery_queue:
            row = schedule_by_id.get(str(queue_entry.get("sortie_id")))
            if row is None:
                continue
            queue_entry["scheduled_start_time"] = float(row["scheduled_start_time"])
            queue_entry["scheduled_finish_time"] = float(row["scheduled_finish_time"])
            queue_entry["predicted_uav_wait"] = float(row["predicted_uav_wait"])
            queue_entry["predicted_cav_wait"] = float(row["predicted_cav_wait"])
        self._record_insertion_attempt(candidate_sortie, evaluation)
        return True

    def _record_insertion_attempt(self, sortie: Sortie, evaluation: Dict[str, Any]) -> None:
        if not bool(self.recovery_insertion_cfg.get("log_insertion_details", True)) and self.dynamic_insertion_enabled:
            return
        diagnostics = evaluation["diagnostics"]
        fields = (
            "queue_length_before", "locked_prefix_length", "feasible_insertion_count",
            "append_index", "selected_index", "insertion_displacement", "selected_cost",
            "append_cost", "counterfactual_gain", "predicted_new_uav_wait",
            "predicted_total_uav_wait", "predicted_existing_delay", "predicted_cav_wait",
            "predicted_route_time_change", "predicted_hover_energy", "selected_is_non_append",
            "insertion_feasible", "fallback_reason", "computation_time_ms",
        )
        self.insertion_log.append(
            {
                "simulation_time": float(self.now),
                "episode": 0,
                "seed": int(self.seed),
                "recovery_rule": self.recovery_rule,
                "uav_id": str(sortie.drone_id),
                "sortie_id": self.recovery_scheduler.sortie_id(sortie),
                **{key: diagnostics.get(key) for key in fields},
            }
        )

    def dispatch_sorties(self, selected: Sequence[Sortie] | Sortie | None) -> int:
        if selected is None:
            return 0
        if isinstance(selected, Sortie):
            selected = [selected]
        # Filter conflicts defensively.
        used_drones, used_tasks = set(), set()
        accepted: List[Sortie] = []
        for s in selected:
            if not s or not s.feasible:
                self.illegal_actions += 1
                continue
            if s.drone_id in used_drones or any(t in used_tasks for t in s.task_ids):
                continue
            drone = self.get_drone(s.drone_id)
            if not drone.is_idle() or drone.position != self.vehicle.position or not self.vehicle.at_intersection:
                self.illegal_actions += 1
                continue
            if any(self.task_pool.by_id[tid].status != "unassigned" for tid in s.task_ids):
                continue
            accepted.append(s)
            used_drones.add(s.drone_id)
            used_tasks.update(s.task_ids)
        if not accepted:
            return 0
        launched = 0
        for s in accepted:
            drone = self.get_drone(s.drone_id)
            evaluation = self.evaluate_insertion(s, commit=False)
            if self.dynamic_insertion_enabled:
                if not evaluation["feasible"]:
                    self.illegal_actions += 1
                    self.infeasible_recoveries += 1
                    self._record_insertion_attempt(s, evaluation)
                    continue
                if not self.commit_insertion(s, evaluation=evaluation):
                    self.illegal_actions += 1
                    continue
            else:
                self._append_fifo_entry(s, evaluation)
            self.task_pool.mark_assigned(s.task_ids, drone.id, self.now)
            drone.assign_sortie(s, self.now)
            self.sortie_log.append({"time": self.now, **s.to_dict()})
            self.event_log.append({"time": self.now, "event": "launch", "agent": drone.id, "node": s.launch_node, "tasks": ";".join(map(str, s.task_ids))})
            self.event_log.append(
                {
                    "time": self.now,
                    "event": "recovery_queue_insert" if self.dynamic_insertion_enabled else "recovery_queue_append",
                    "agent": drone.id,
                    "node": int(s.recovery_node),
                    "queue_position": int(evaluation["best_index"]) if self.dynamic_insertion_enabled else int(evaluation["diagnostics"]["append_index"]),
                    "tasks": ";".join(map(str, s.task_ids)),
                }
            )
            launched += 1
        self._set_vehicle_next_recovery_target()
        return launched

    def _append_fifo_entry(self, sortie: Sortie, evaluation: Dict[str, Any]) -> None:
        entry = evaluation.get("entry") or self.recovery_scheduler.make_entry(sortie)
        self.recovery_queue.append(dict(entry))
        if evaluation.get("predicted_schedule"):
            schedule_by_id = {str(row["sortie_id"]): row for row in evaluation["predicted_schedule"]}
            for queue_entry in self.recovery_queue:
                row = schedule_by_id.get(str(queue_entry.get("sortie_id")))
                if row is not None:
                    queue_entry["scheduled_start_time"] = float(row["scheduled_start_time"])
                    queue_entry["scheduled_finish_time"] = float(row["scheduled_finish_time"])
                    queue_entry["predicted_uav_wait"] = float(row["predicted_uav_wait"])
                    queue_entry["predicted_cav_wait"] = float(row["predicted_cav_wait"])
        self._record_insertion_attempt(sortie, evaluation)

    def get_drone(self, drone_id: str) -> Drone:
        for d in self.drones:
            if d.id == drone_id:
                return d
        raise KeyError(drone_id)

    def run(self, policy: PolicyFn, method_name: str = "policy", no_multitask: bool = False, no_priority: bool = False) -> EpisodeMetrics:
        next_decision = 0.0
        while self.now < self.horizon:
            self.task_pool.update_release_and_expiry(self.now)
            if self.now + 1e-9 >= next_decision:
                candidates = self.get_candidates(no_multitask=no_multitask, no_priority=no_priority)
                selected = policy(self, candidates) if candidates else None
                self.dispatch_sorties(selected)
                next_decision += self.decision_interval
            self.step(min(self.dt, self.horizon - self.now))
        return self.metrics(method_name)

    def step(self, dt: float) -> None:
        if dt <= 0:
            return
        start = self.now
        self.last_step_vehicle_arrivals = []
        self.completed_since_last = 0
        self.expired_since_last = 0
        self._integrate_aoi(dt)
        # Expiry before movement.
        before_expired = {t.id for t in self.tasks if t.status == "expired"}
        self.task_pool.update_release_and_expiry(self.now)
        after_expired = {t.id for t in self.tasks if t.status == "expired"}
        self.expired_since_last += len(after_expired - before_expired - self._last_expired_ids)
        self._last_expired_ids.update(after_expired)
        # Vehicle waiting if it reaches a rendezvous before a drone.
        front = self.recovery_queue[0] if self.recovery_queue else None
        if self.vehicle.is_idle() and front is not None and int(front["recovery_node"]) == self.vehicle.position and self.get_drone(str(front["drone_id"])).status != "idle":
            self.vehicle.add_wait(dt)
        # Vehicle update.
        route_log_len = len(self.vehicle.route_log)
        arrived = self.vehicle.step(dt, self.now, self.network)
        new_arrivals = self.vehicle.route_log[route_log_len:]
        if new_arrivals:
            self.last_step_vehicle_arrivals.extend((float(t), int(n)) for t, n in new_arrivals)
            self._sync_idle_drones_with_vehicle()
        # Drone update.
        for drone in self.drones:
            self._advance_drone(drone, dt, start)
        for arrival_time, node in self.last_step_vehicle_arrivals:
            self.event_log.append({"time": arrival_time, "event": "vehicle_arrive", "agent": self.vehicle.id, "node": node, "tasks": ""})
        if arrived:
            self._recover_waiting_drones()
        # Recover if vehicle is already at a waiting drone's recovery point.
        self._recover_waiting_drones()
        if self.vehicle.is_idle():
            self._set_vehicle_next_recovery_target()
        self.now += dt
        self.task_pool.update_release_and_expiry(self.now)

    def _advance_drone(self, drone: Drone, dt: float, step_start: float) -> None:
        if drone.status == "idle":
            return
        if drone.status == "waiting_recovery":
            self._consume(drone, dt, "hover")
            drone.wait_time += dt
            return
        if drone.status != "flying" or drone.sortie is None:
            return
        remain = dt
        local_time = step_start
        while remain > 1e-9 and drone.status == "flying":
            if drone.segment_index >= len(drone.sortie.segments):
                self._drone_arrived_recovery(drone, local_time)
                break
            seg = drone.sortie.segments[drone.segment_index]
            if drone.segment_index >= len(drone.sortie.segments) - 1 and seg.get("type") == "fly":
                entry = self._queue_entry_for_drone(drone.id)
                if entry is not None:
                    entry["status"] = "returning"
            if drone.segment_remaining <= 1e-9:
                drone.segment_remaining = float(seg["duration"])
            use = min(remain, drone.segment_remaining)
            self._consume(drone, use, seg["type"])
            drone.segment_remaining -= use
            remain -= use
            local_time += use
            if drone.segment_remaining <= 1e-9:
                # Finish segment.
                if seg["type"] == "fly":
                    drone.position = int(seg["to"])
                elif seg["type"] == "service":
                    tid = int(seg["task_id"])
                    if self.task_pool.by_id[tid].status in {"assigned", "unassigned"}:
                        self.task_pool.mark_executing(tid, drone.id, max(step_start, local_time - float(seg["duration"])))
                    if bool(seg.get("service_final", True)):
                        self.task_pool.mark_done(tid, local_time, drone.id)
                        drone.completed_task_ids.append(tid)
                        self.completed_since_last += 1
                        for arc in self.task_pool.by_id[tid].target_arcs:
                            self.cover_last_time[arc] = local_time
                        self.event_log.append({"time": local_time, "event": "task_done", "agent": drone.id, "node": int(seg["to"]), "tasks": str(tid)})
                    drone.position = int(seg["to"])
                drone.segment_index += 1
                if drone.segment_index >= len(drone.sortie.segments):
                    self._drone_arrived_recovery(drone, local_time)
                    break
                else:
                    drone.segment_remaining = float(drone.sortie.segments[drone.segment_index]["duration"])

    def _consume(self, drone: Drone, dt: float, mode: str) -> None:
        if mode == "service":
            rate = float(self.cfg["agents"].get("drone_service_power", 0.85))
            drone.service_time += dt
        elif mode == "hover":
            rate = float(self.cfg["agents"].get("drone_hover_power", 0.65))
        else:
            rate = float(self.cfg["agents"].get("drone_fly_power", 1.0))
            drone.fly_time += dt
        energy = rate * dt
        drone.energy_used += energy
        drone.battery = max(0.0, drone.battery - energy)
        self.minimum_battery_observed = min(self.minimum_battery_observed, drone.battery)

    def _drone_arrived_recovery(self, drone: Drone, t: float) -> None:
        rec = int(drone.sortie.recovery_node)
        drone.position = rec
        entry = self._queue_entry_for_drone(drone.id)
        if entry is not None:
            entry["status"] = "returning"
            entry["recovery_arrival_time"] = float(t)
        if self._front_queue_matches(drone.id, rec) and self.vehicle.position == rec and self.vehicle.is_idle():
            self._recover_drone(drone, t)
        else:
            drone.status = "waiting_recovery"
            drone.log.append((t, "waiting_recovery", rec))
            self.event_log.append({"time": t, "event": "drone_wait_recovery", "agent": drone.id, "node": rec, "tasks": ""})

    def _recover_waiting_drones(self) -> None:
        if not self.recovery_queue or not self.vehicle.is_idle():
            return
        front = self.recovery_queue[0]
        drone = self.get_drone(str(front["drone_id"]))
        rec = int(front["recovery_node"])
        if drone.status == "waiting_recovery" and drone.position == self.vehicle.position == rec:
            front["status"] = "recovering"
            self._recover_drone(drone, self.now)

    def _recover_drone(self, drone: Drone, t: float) -> None:
        rec = drone.position
        entry = self._queue_entry_for_drone(drone.id)
        if entry is not None:
            arrival_time = entry.get("recovery_arrival_time")
            actual_wait = max(0.0, float(t) - float(arrival_time)) if arrival_time is not None else 0.0
            self.recovery_wait_records.append(actual_wait)
            entry["status"] = "completed"
        drone.reset_after_recovery(rec, t)
        if self.recovery_queue and str(self.recovery_queue[0]["drone_id"]) == drone.id:
            finished = self.recovery_queue.pop(0)
            self.event_log.append(
                {
                    "time": t,
                    "event": "recovery_queue_pop",
                    "agent": drone.id,
                    "node": int(finished["recovery_node"]),
                    "tasks": ";".join(map(str, finished.get("task_ids", []))),
                }
            )
        self.event_log.append({"time": t, "event": "recover", "agent": drone.id, "node": rec, "tasks": ""})
        self._sync_idle_drones_with_vehicle()

    def _set_vehicle_next_recovery_target(self) -> None:
        if not self.vehicle.is_idle():
            return
        if not self.recovery_queue:
            return
        target = int(self.recovery_queue[0]["recovery_node"])
        if target == self.vehicle.position:
            return
        self.vehicle.set_target(target, self.network, self.now)
        self.event_log.append({"time": self.now, "event": "vehicle_depart", "agent": self.vehicle.id, "node": self.vehicle.position, "target": target, "tasks": ""})

    def _sync_idle_drones_with_vehicle(self) -> None:
        for drone in self.drones:
            if drone.is_idle():
                drone.position = int(self.vehicle.position)

    def _front_queue_matches(self, drone_id: str, recovery_node: int) -> bool:
        if not self.recovery_queue:
            return False
        front = self.recovery_queue[0]
        return str(front["drone_id"]) == str(drone_id) and int(front["recovery_node"]) == int(recovery_node)

    def _queue_entry_for_drone(self, drone_id: str) -> Optional[Dict[str, Any]]:
        for entry in self.recovery_queue:
            if str(entry.get("drone_id", entry.get("uav_id"))) == str(drone_id):
                return entry
        return None

    def recovery_queue_anchor_node(self) -> int:
        if self.recovery_queue:
            return int(self.recovery_queue[-1]["recovery_node"])
        if self.vehicle.target_node is not None:
            return int(self.vehicle.target_node)
        return int(self.vehicle.position)

    def recovery_queue_travel_time_s(self) -> float:
        if not self.recovery_queue:
            return float(self.vehicle.remaining_time if self.vehicle.target_node is not None else 0.0)
        total = float(self.vehicle.remaining_time if self.vehicle.target_node is not None else 0.0)
        nodes = [int(x["recovery_node"]) for x in self.recovery_queue]
        start_idx = 0
        if self.vehicle.target_node is None:
            total += self.network.get_vehicle_time(self.vehicle.position, nodes[0])
        else:
            start_idx = 1
        for u, v in zip(nodes[start_idx:-1], nodes[start_idx + 1 :]):
            total += self.network.get_vehicle_time(u, v)
        return float(total)

    def _integrate_aoi(self, dt: float) -> None:
        if not self.cover_last_time:
            return
        total = 0.0
        for arc in self.network.arcs:
            risk = self.network.arc_risk(arc)
            last = self.cover_last_time.get(arc, 0.0)
            total += risk * max(0.0, self.now - last)
        self.aoi_integral += (total / max(1, len(self.network.arcs))) * dt

    def _all_terminal(self) -> bool:
        if any(d.status != "idle" for d in self.drones):
            return False
        return all(t.status in {"done", "expired"} for t in self.tasks)

    def metrics(self, method_name: str) -> EpisodeMetrics:
        base = summarize_task_metrics(self.tasks, int(self.cfg["simulation"].get("high_priority_threshold", 8)))
        total_drone_time = sum(d.fly_time + d.service_time + d.wait_time for d in self.drones)
        possible_drone_time = max(1.0, len(self.drones) * max(self.now, 1.0))
        total_energy = sum(d.energy_used for d in self.drones)
        hover_time = sum(d.wait_time for d in self.drones)
        hover_energy = hover_time * float(self.cfg.get("agents", {}).get("drone_hover_power", 0.65))
        completed_recoveries = len(self.recovery_wait_records)
        insertion_count = len(self.insertion_log)
        non_append = [row for row in self.insertion_log if bool(row.get("selected_is_non_append"))]
        cav_distance = sum(
            self.network.arc_length((int(u), int(v)))
            for _, u, v in self.vehicle.route_edge_log
        )
        num_sorties = len(self.sortie_log)
        avg_tasks_per_sortie = (sum(len(str(x.get("task_ids", [])).split(",")) if isinstance(x.get("task_ids"), str) else len(x.get("task_ids", [])) for x in self.sortie_log) / num_sorties) if num_sorties else 0.0
        m = EpisodeMetrics(
            seed=self.seed,
            method=method_name,
            scenario=self.scenario,
            weighted_aoi=self.aoi_integral / max(1.0, self.now),
            total_energy=total_energy,
            vehicle_wait_time_s=self.vehicle.wait_time,
            drone_wait_time_s=sum(d.wait_time for d in self.drones),
            drone_utilization=total_drone_time / possible_drone_time,
            tasks_per_hour=base["completed_tasks"] / max(1.0, self.now / 3600.0),
            illegal_actions=self.illegal_actions,
            avg_tasks_per_sortie=avg_tasks_per_sortie,
            num_sorties=num_sorties,
            uav_energy_per_completed_task=total_energy / base["completed_tasks"] if base["completed_tasks"] else math.nan,
            mean_uav_recovery_wait_s=float(np.mean(self.recovery_wait_records)) if self.recovery_wait_records else 0.0,
            p95_uav_recovery_wait_s=float(np.percentile(self.recovery_wait_records, 95)) if self.recovery_wait_records else 0.0,
            total_uav_hover_time_s=hover_time,
            total_hover_energy=hover_energy,
            mean_cav_waiting_time_s=self.vehicle.wait_time / completed_recoveries if completed_recoveries else 0.0,
            cav_travel_distance_m=float(cav_distance),
            recovery_contention_rate=(sum(int(row.get("queue_length_before", 0)) >= 2 for row in self.insertion_log) / insertion_count) if insertion_count else 0.0,
            non_append_rate=len(non_append) / insertion_count if insertion_count else 0.0,
            mean_insertion_displacement=float(np.mean([row.get("insertion_displacement", 0) for row in self.insertion_log])) if insertion_count else 0.0,
            mean_counterfactual_gain=float(np.mean([row.get("counterfactual_gain", 0.0) for row in self.insertion_log])) if insertion_count else 0.0,
            locked_queue_rate=(sum(int(row.get("locked_prefix_length", 0)) > 0 for row in self.insertion_log) / insertion_count) if insertion_count else 0.0,
            no_feasible_insertion_rate=self.no_feasible_insertion_count / self.insertion_evaluation_count if self.insertion_evaluation_count else 0.0,
            dynamic_insertion_computation_time_ms=float(np.mean([row.get("computation_time_ms", 0.0) for row in self.insertion_log])) if self.dynamic_insertion_enabled and insertion_count else 0.0,
            recovery_attempts=insertion_count,
            infeasible_recoveries=self.infeasible_recoveries,
            minimum_battery_observed=float(self.minimum_battery_observed),
        )
        for k, v in base.items():
            setattr(m, k, v)
        return m

    def export_logs(self, out_dir: str | Path, prefix: str = "episode") -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([t.to_record() for t in self.tasks]).to_csv(out / f"{prefix}_tasks.csv", index=False)
        pd.DataFrame(self.event_log).to_csv(out / f"{prefix}_events.csv", index=False)
        pd.DataFrame(self.sortie_log).to_csv(out / f"{prefix}_sorties.csv", index=False)
        pd.DataFrame(self.vehicle.route_log, columns=["time", "node"]).to_csv(out / f"{prefix}_vehicle_route.csv", index=False)
        pd.DataFrame(self.vehicle.route_edge_log, columns=["time", "from_node", "to_node"]).to_csv(out / f"{prefix}_vehicle_edges.csv", index=False)
        pd.DataFrame(self.insertion_log).to_csv(out / f"{prefix}_recovery_insertions.csv", index=False)
