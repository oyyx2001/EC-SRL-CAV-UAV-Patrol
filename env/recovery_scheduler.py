from __future__ import annotations

from copy import deepcopy
import math
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple


class RecoveryInsertionScheduler:
    """Side-effect-free evaluator for FIFO and dynamic recovery insertion."""

    def __init__(self, simulator: Any):
        self.sim = simulator
        self.cfg = simulator.cfg.get("recovery_insertion", {})
        self.agent_cfg = simulator.cfg.get("agents", {})

    def sortie_id(self, sortie: Any) -> str:
        tasks = "-".join(str(int(x)) for x in sortie.task_ids)
        return f"{sortie.drone_id}:{self.sim.now:.6f}:{tasks}:{int(sortie.recovery_node)}"

    def make_entry(self, sortie: Any) -> Dict[str, Any]:
        drone = self.sim.get_drone(str(sortie.drone_id))
        base_energy = float(getattr(sortie, "flight_service_energy", 0.0))
        if base_energy <= 0.0 and getattr(sortie, "segments", None):
            base_energy = self._segment_energy(sortie.segments)
        battery_at_eta = float(getattr(sortie, "battery_at_eta", drone.battery - base_energy))
        reserve = float(self.cfg.get("battery_reserve", self.agent_cfg.get("drone_battery_reserve", 0.0)))
        return {
            "uav_id": str(drone.id),
            "drone_id": str(drone.id),
            "recovery_node": int(sortie.recovery_node),
            "eta": float(self.sim.now + sortie.drone_time_s),
            "recovery_duration": float(self.agent_cfg.get("recovery_time_s", 0.0)),
            "battery_at_eta": battery_at_eta,
            "battery_reserve": reserve,
            "hover_power": float(self.agent_cfg.get("drone_hover_power", 0.65)),
            "sortie_id": self.sortie_id(sortie),
            "status": "planned",
            "locked": False,
            "original_queue_index": len(self.sim.recovery_queue),
            "scheduled_start_time": None,
            "scheduled_finish_time": None,
            "predicted_uav_wait": None,
            "predicted_cav_wait": None,
            "launch_time": float(self.sim.now),
            "task_ids": list(sortie.task_ids),
            "recovery_arrival_time": None,
        }

    def evaluate(self, sortie: Any, rule: str = "dynamic_insertion") -> Dict[str, Any]:
        started = perf_counter()
        queue = [dict(entry) for entry in self.sim.recovery_queue]
        new_entry = self.make_entry(sortie)
        append_index = len(queue)
        locked_prefix = self.locked_prefix_length(queue)
        state_fingerprint = self.state_fingerprint()

        duplicate = any(str(entry.get("drone_id", entry.get("uav_id"))) == str(sortie.drone_id) for entry in queue)
        if duplicate:
            return self._infeasible_result(
                append_index,
                locked_prefix,
                state_fingerprint,
                "duplicate_recovery_entry",
                started,
            )

        append_queue = queue + [new_entry]
        append_sim = self.simulate(append_queue, new_entry["sortie_id"])
        append_cost_parts = self._cost_parts(append_sim, append_sim, queue)
        append_cost = self._weighted_cost(append_cost_parts)

        positions: Sequence[int]
        if str(rule).lower() == "fifo":
            positions = [append_index]
        else:
            positions = range(locked_prefix, append_index + 1)

        feasible_options: List[Dict[str, Any]] = []
        position_results: List[Dict[str, Any]] = []
        for index in positions:
            trial_queue = [dict(entry) for entry in queue]
            trial_queue.insert(int(index), dict(new_entry))
            simulated = self.simulate(trial_queue, new_entry["sortie_id"])
            cost_parts = self._cost_parts(simulated, append_sim, queue)
            cost = self._weighted_cost(cost_parts)
            option = {
                "index": int(index),
                "feasible": bool(simulated["feasible"]),
                "reason": str(simulated["reason"]),
                "cost": float(cost),
                "cost_parts": cost_parts,
                "schedule": simulated["schedule"],
            }
            position_results.append(option)
            if option["feasible"]:
                feasible_options.append(option)

        if not feasible_options:
            result = self._infeasible_result(
                append_index,
                locked_prefix,
                state_fingerprint,
                position_results[0]["reason"] if position_results else append_sim["reason"],
                started,
            )
            result["diagnostics"]["position_results"] = position_results
            result["diagnostics"]["append_cost"] = float(append_cost)
            return result

        def tie_key(option: Dict[str, Any]) -> Tuple[float, float, int, int, int]:
            parts = option["cost_parts"]
            index = int(option["index"])
            return (
                round(float(option["cost"]), 12),
                round(float(parts["existing_delay"]), 12),
                append_index - index,
                -index,
                index,
            )

        selected = min(feasible_options, key=tie_key)
        selected_index = int(selected["index"])
        selected_parts = selected["cost_parts"]
        selected_schedule = selected["schedule"]
        new_row = next(row for row in selected_schedule if row["sortie_id"] == new_entry["sortie_id"])
        fallback_reason = "dynamic_insertion"
        if selected_index == append_index:
            fallback_reason = "append_only_feasible" if len(feasible_options) == 1 else "best_is_append"
        elapsed_ms = (perf_counter() - started) * 1000.0
        diagnostics = {
            "queue_length_before": len(queue),
            "locked_prefix_length": locked_prefix,
            "feasible_insertion_count": len(feasible_options),
            "append_index": append_index,
            "selected_index": selected_index,
            "insertion_displacement": append_index - selected_index,
            "selected_cost": float(selected["cost"]),
            "append_cost": float(append_cost),
            "counterfactual_gain": float(append_cost - selected["cost"]),
            "predicted_new_uav_wait": float(new_row["predicted_uav_wait"]),
            "predicted_total_uav_wait": float(selected_parts["total_uav_wait"]),
            "predicted_existing_delay": float(selected_parts["existing_delay"]),
            "predicted_cav_wait": float(selected_parts["total_cav_wait"]),
            "predicted_route_time_change": float(selected_parts["route_time_change"]),
            "predicted_hover_energy": float(selected_parts["hover_energy"]),
            "selected_is_non_append": selected_index < append_index,
            "insertion_feasible": True,
            "fallback_reason": fallback_reason,
            "computation_time_ms": float(elapsed_ms),
            "position_results": position_results,
        }
        return {
            "feasible": True,
            "best_index": selected_index,
            "insertion_cost": float(selected["cost"]),
            "predicted_schedule": deepcopy(selected_schedule),
            "diagnostics": diagnostics,
            "entry": dict(new_entry),
            "state_fingerprint": state_fingerprint,
        }

    def evaluate_insertions(
        self,
        sortie: Any,
        ground_plan_override: Optional[Dict[str, Any]] = None,
        max_results: int = 1,
        commit: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return ranked feasible insertion alternatives without side effects.

        The legacy :meth:`evaluate` result and tie-breaking remain unchanged.
        A ground override may supply a trial ``queue`` and the physical
        ``first_cav_arrival_time`` produced by a stage-three GroundCandidate.
        Committing here is intentionally unsupported: stage four commits the
        selected ground and air components in one atomic transaction.
        """

        if commit:
            raise ValueError("ranked insertion evaluation is side-effect-free; commit the JointPlan")
        max_results = max(0, int(max_results))
        if max_results == 0:
            return []
        if ground_plan_override is None:
            result = self.evaluate(sortie, rule="dynamic_insertion")
            if not result.get("feasible", False):
                return []
            diagnostics = result.get("diagnostics", {})
            append_index = int(diagnostics.get("append_index", len(self.sim.recovery_queue)))
            feasible = [
                option
                for option in diagnostics.get("position_results", [])
                if bool(option.get("feasible", False))
            ]
            feasible.sort(
                key=lambda option: (
                    round(float(option.get("cost", math.inf)), 12),
                    round(float(option.get("cost_parts", {}).get("existing_delay", math.inf)), 12),
                    append_index - int(option.get("index", append_index)),
                    -int(option.get("index", append_index)),
                    int(option.get("index", append_index)),
                )
            )
            ranked: List[Dict[str, Any]] = []
            for option in feasible[:max_results]:
                ranked.append(
                    {
                        "feasible": True,
                        "best_index": int(option["index"]),
                        "insertion_cost": float(option["cost"]),
                        "predicted_schedule": deepcopy(option["schedule"]),
                        "diagnostics": {
                            **diagnostics,
                            "selected_index": int(option["index"]),
                            "selected_cost": float(option["cost"]),
                        },
                        "entry": dict(result["entry"]),
                        "state_fingerprint": result["state_fingerprint"],
                    }
                )
            return ranked

        started = perf_counter()
        queue = [dict(entry) for entry in ground_plan_override.get("queue", self.sim.recovery_queue)]
        new_entry = self.make_entry(sortie)
        if any(
            str(entry.get("drone_id", entry.get("uav_id"))) == str(sortie.drone_id)
            for entry in queue
        ):
            return []
        append_index = len(queue)
        locked_prefix = self.locked_prefix_length(queue)
        first_arrival = float(
            ground_plan_override.get("first_cav_arrival_time", self.sim.now)
        )
        options: List[Dict[str, Any]] = []
        append_queue = [dict(entry) for entry in queue] + [dict(new_entry)]
        append_sim = self.simulate_ground_plan(append_queue, first_arrival)
        append_parts = self._cost_parts(append_sim, append_sim, queue)
        append_cost = self._weighted_cost(append_parts)
        for index in range(locked_prefix, append_index + 1):
            trial = [dict(entry) for entry in queue]
            trial.insert(index, dict(new_entry))
            simulated = self.simulate_ground_plan(trial, first_arrival)
            if not simulated.get("feasible", False):
                continue
            parts = self._cost_parts(simulated, append_sim, queue)
            options.append(
                {
                    "index": index,
                    "cost": self._weighted_cost(parts),
                    "cost_parts": parts,
                    "schedule": simulated["schedule"],
                }
            )
        options.sort(
            key=lambda option: (
                round(float(option["cost"]), 12),
                round(float(option["cost_parts"]["existing_delay"]), 12),
                append_index - int(option["index"]),
                -int(option["index"]),
                int(option["index"]),
            )
        )
        ranked = []
        for option in options[:max_results]:
            selected_index = int(option["index"])
            schedule = deepcopy(option["schedule"])
            new_row = next(row for row in schedule if row["sortie_id"] == new_entry["sortie_id"])
            ranked.append(
                {
                    "feasible": True,
                    "best_index": selected_index,
                    "insertion_cost": float(option["cost"]),
                    "predicted_schedule": schedule,
                    "diagnostics": {
                        "queue_length_before": len(queue),
                        "locked_prefix_length": locked_prefix,
                        "feasible_insertion_count": len(options),
                        "append_index": append_index,
                        "selected_index": selected_index,
                        "insertion_displacement": append_index - selected_index,
                        "selected_cost": float(option["cost"]),
                        "append_cost": float(append_cost),
                        "counterfactual_gain": float(append_cost - option["cost"]),
                        "predicted_new_uav_wait": float(new_row["predicted_uav_wait"]),
                        "predicted_total_uav_wait": float(option["cost_parts"]["total_uav_wait"]),
                        "predicted_existing_delay": float(option["cost_parts"]["existing_delay"]),
                        "predicted_cav_wait": float(option["cost_parts"]["total_cav_wait"]),
                        "predicted_route_time_change": float(option["cost_parts"]["route_time_change"]),
                        "predicted_hover_energy": float(option["cost_parts"]["hover_energy"]),
                        "selected_is_non_append": selected_index < append_index,
                        "insertion_feasible": True,
                        "fallback_reason": "ground_plan_override",
                        "computation_time_ms": (perf_counter() - started) * 1000.0,
                    },
                    "entry": dict(new_entry),
                    "state_fingerprint": self.state_fingerprint(),
                }
            )
        return ranked

    def simulate(self, queue: Sequence[Dict[str, Any]], new_sortie_id: Optional[str] = None) -> Dict[str, Any]:
        schedule: List[Dict[str, Any]] = []
        current_time = float(self.sim.now)
        current_node = int(self.sim.vehicle.position)
        route_time = 0.0
        active_target = self.sim.vehicle.target_node
        use_battery = bool(self.cfg.get("use_battery_feasibility", True))

        if active_target is not None and queue:
            if int(queue[0]["recovery_node"]) != int(active_target):
                return {"feasible": False, "reason": "active_vehicle_route_changed", "schedule": [], "route_time": math.inf}

        for index, entry in enumerate(queue):
            node = int(entry["recovery_node"])
            if index == 0 and active_target is not None:
                travel_time = float(self.sim.vehicle.remaining_time)
            else:
                travel_time = float(self.sim.network.get_vehicle_time(current_node, node))
            if not math.isfinite(travel_time):
                return {"feasible": False, "reason": "vehicle_unreachable", "schedule": schedule, "route_time": math.inf}
            cav_arrive = current_time + travel_time
            eta = float(entry.get("eta", self.sim.now))
            start = max(cav_arrive, eta)
            finish = start + float(entry.get("recovery_duration", 0.0))
            uav_wait = max(0.0, start - eta)
            cav_wait = max(0.0, eta - cav_arrive)
            hover_power = float(entry.get("hover_power", self.agent_cfg.get("drone_hover_power", 0.65)))
            hover_energy = hover_power * uav_wait
            battery_after_wait = float(entry.get("battery_at_eta", math.inf)) - hover_energy
            battery_reserve = float(entry.get("battery_reserve", self.cfg.get("battery_reserve", 0.0)))
            row = {
                "queue_index": index,
                "uav_id": str(entry.get("uav_id", entry.get("drone_id"))),
                "drone_id": str(entry.get("drone_id", entry.get("uav_id"))),
                "sortie_id": str(entry.get("sortie_id", entry.get("drone_id"))),
                "recovery_node": node,
                "eta": eta,
                "cav_arrival_time": float(cav_arrive),
                "scheduled_start_time": float(start),
                "scheduled_finish_time": float(finish),
                "predicted_uav_wait": float(uav_wait),
                "predicted_cav_wait": float(cav_wait),
                "predicted_hover_energy": float(hover_energy),
                "battery_after_wait": float(battery_after_wait),
                "battery_reserve": float(battery_reserve),
                "is_new": str(entry.get("sortie_id")) == str(new_sortie_id),
            }
            schedule.append(row)
            if use_battery and battery_after_wait < battery_reserve - 1e-9:
                return {"feasible": False, "reason": "battery_reserve", "schedule": schedule, "route_time": route_time + travel_time}
            route_time += travel_time
            current_time = finish
            current_node = node

        return {"feasible": True, "reason": "ok", "schedule": schedule, "route_time": float(route_time)}

    def simulate_ground_plan(
        self,
        queue: Sequence[Dict[str, Any]],
        first_cav_arrival_time: float,
    ) -> Dict[str, Any]:
        """Reuse the stage-one recovery model after a proposed ground-route prefix.

        ``first_cav_arrival_time`` is the physical arrival at the first queued
        recovery node after executing the candidate CAV route.  No simulator
        state is read or written except for immutable travel-time lookups.
        """
        if not queue:
            return {"feasible": True, "reason": "ok", "schedule": [], "route_time": 0.0}
        schedule: List[Dict[str, Any]] = []
        current_time = float(first_cav_arrival_time)
        current_node = int(queue[0]["recovery_node"])
        route_time = max(0.0, current_time - float(self.sim.now))
        use_battery = bool(self.cfg.get("use_battery_feasibility", True))
        for index, entry in enumerate(queue):
            node = int(entry["recovery_node"])
            if index == 0:
                cav_arrive = current_time
            else:
                travel_time = float(self.sim.network.get_vehicle_time(current_node, node))
                if not math.isfinite(travel_time):
                    return {
                        "feasible": False,
                        "reason": "vehicle_unreachable",
                        "schedule": schedule,
                        "route_time": math.inf,
                    }
                route_time += travel_time
                cav_arrive = current_time + travel_time
            eta = float(entry.get("eta", self.sim.now))
            start = max(cav_arrive, eta)
            finish = start + float(entry.get("recovery_duration", 0.0))
            uav_wait = max(0.0, start - eta)
            cav_wait = max(0.0, eta - cav_arrive)
            hover_power = float(entry.get("hover_power", self.agent_cfg.get("drone_hover_power", 0.65)))
            hover_energy = hover_power * uav_wait
            battery_after_wait = float(entry.get("battery_at_eta", math.inf)) - hover_energy
            reserve = float(entry.get("battery_reserve", self.cfg.get("battery_reserve", 0.0)))
            row = {
                "queue_index": index,
                "uav_id": str(entry.get("uav_id", entry.get("drone_id"))),
                "drone_id": str(entry.get("drone_id", entry.get("uav_id"))),
                "sortie_id": str(entry.get("sortie_id", entry.get("drone_id"))),
                "recovery_node": node,
                "eta": eta,
                "cav_arrival_time": float(cav_arrive),
                "scheduled_start_time": float(start),
                "scheduled_finish_time": float(finish),
                "predicted_uav_wait": float(uav_wait),
                "predicted_cav_wait": float(cav_wait),
                "predicted_hover_energy": float(hover_energy),
                "hover_power": float(hover_power),
                "battery_after_wait": float(battery_after_wait),
                "battery_reserve": float(reserve),
                "is_new": False,
            }
            schedule.append(row)
            if use_battery and battery_after_wait < reserve - 1e-9:
                return {
                    "feasible": False,
                    "reason": "battery_reserve",
                    "schedule": schedule,
                    "route_time": float(route_time),
                }
            current_time = finish
            current_node = node
        return {
            "feasible": True,
            "reason": "ok",
            "schedule": schedule,
            "route_time": float(route_time),
        }

    def locked_prefix_length(self, queue: Sequence[Dict[str, Any]]) -> int:
        preserve = bool(self.cfg.get("preserve_locked_prefix", True))
        horizon = float(self.cfg.get("lock_horizon_s", 120.0))
        last_locked = -1
        for index, entry in enumerate(queue):
            status = self._runtime_status(entry)
            locked = bool(entry.get("locked", False))
            if index == 0 and self.sim.vehicle.target_node is not None:
                locked = True  # The existing simulator has no CAV reroute action.
            if status == "recovering":
                locked = True
            if status == "returning" and float(entry.get("eta", self.sim.now)) - self.sim.now <= horizon + 1e-9:
                locked = True
            if locked:
                last_locked = index
        if not preserve:
            return 1 if queue and self.sim.vehicle.target_node is not None else 0
        return last_locked + 1

    def state_fingerprint(self) -> Tuple[Any, ...]:
        queue_state = tuple(
            (
                str(entry.get("sortie_id", entry.get("drone_id"))),
                int(entry["recovery_node"]),
                str(entry.get("status", "planned")),
            )
            for entry in self.sim.recovery_queue
        )
        drone_state = tuple((d.id, d.status, int(d.position), round(float(d.battery), 9)) for d in self.sim.drones)
        return (
            round(float(self.sim.now), 9),
            int(self.sim.vehicle.position),
            self.sim.vehicle.target_node,
            round(float(self.sim.vehicle.remaining_time), 9),
            queue_state,
            drone_state,
        )

    def _runtime_status(self, entry: Dict[str, Any]) -> str:
        stored = str(entry.get("status", "planned"))
        if stored in {"returning", "recovering", "completed"}:
            return stored
        try:
            drone = self.sim.get_drone(str(entry.get("drone_id", entry.get("uav_id"))))
        except KeyError:
            return stored
        if drone.status == "waiting_recovery":
            return "recovering" if self.sim.vehicle.position == drone.position else "returning"
        if drone.status == "flying" and drone.sortie is not None:
            last_index = max(0, len(drone.sortie.segments or []) - 1)
            if drone.segment_index >= last_index:
                return "returning"
        return stored

    def _cost_parts(
        self,
        simulated: Dict[str, Any],
        append_sim: Dict[str, Any],
        old_queue: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        schedule = simulated.get("schedule", [])
        append_schedule = append_sim.get("schedule", [])
        append_starts = {str(row["sortie_id"]): float(row["scheduled_start_time"]) for row in append_schedule}
        existing_ids = {str(entry.get("sortie_id", entry.get("drone_id"))) for entry in old_queue}
        existing_delay = sum(
            max(0.0, float(row["scheduled_start_time"]) - append_starts.get(str(row["sortie_id"]), float(row["scheduled_start_time"])))
            for row in schedule
            if str(row["sortie_id"]) in existing_ids
        )
        route_time = float(simulated.get("route_time", math.inf))
        append_route_time = float(append_sim.get("route_time", math.inf))
        route_change = max(0.0, route_time - append_route_time) if math.isfinite(route_time) and math.isfinite(append_route_time) else math.inf
        return {
            "total_uav_wait": float(sum(row["predicted_uav_wait"] for row in schedule)),
            "existing_delay": float(existing_delay),
            "total_cav_wait": float(sum(row["predicted_cav_wait"] for row in schedule)),
            "route_time_change": float(route_change),
            "hover_energy": float(sum(row["predicted_hover_energy"] for row in schedule)),
        }

    def _weighted_cost(self, parts: Dict[str, float]) -> float:
        return float(
            float(self.cfg.get("weight_uav_wait", 1.0)) * parts["total_uav_wait"]
            + float(self.cfg.get("weight_existing_delay", 0.5)) * parts["existing_delay"]
            + float(self.cfg.get("weight_cav_wait", 0.25)) * parts["total_cav_wait"]
            + float(self.cfg.get("weight_route_time", 0.10)) * parts["route_time_change"]
        )

    def _segment_energy(self, segments: Sequence[Dict[str, Any]]) -> float:
        rates = {
            "fly": float(self.agent_cfg.get("drone_fly_power", 1.0)),
            "service": float(self.agent_cfg.get("drone_service_power", 0.85)),
        }
        return float(sum(float(seg.get("duration", 0.0)) * rates.get(str(seg.get("type")), rates["fly"]) for seg in segments))

    def _infeasible_result(
        self,
        append_index: int,
        locked_prefix: int,
        state_fingerprint: Tuple[Any, ...],
        reason: str,
        started: float,
    ) -> Dict[str, Any]:
        diagnostics = {
            "queue_length_before": append_index,
            "locked_prefix_length": locked_prefix,
            "feasible_insertion_count": 0,
            "append_index": append_index,
            "selected_index": None,
            "insertion_displacement": 0,
            "selected_cost": math.inf,
            "append_cost": math.inf,
            "counterfactual_gain": 0.0,
            "predicted_new_uav_wait": math.nan,
            "predicted_total_uav_wait": math.nan,
            "predicted_existing_delay": math.nan,
            "predicted_cav_wait": math.nan,
            "predicted_route_time_change": math.nan,
            "predicted_hover_energy": math.nan,
            "selected_is_non_append": False,
            "insertion_feasible": False,
            "fallback_reason": reason,
            "computation_time_ms": float((perf_counter() - started) * 1000.0),
            "position_results": [],
        }
        return {
            "feasible": False,
            "best_index": None,
            "insertion_cost": math.inf,
            "predicted_schedule": [],
            "diagnostics": diagnostics,
            "entry": None,
            "state_fingerprint": state_fingerprint,
        }
