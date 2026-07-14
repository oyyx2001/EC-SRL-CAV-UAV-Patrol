from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations, permutations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math

import numpy as np

from dynamic_patrol_marl.env.road_network import RoadNetwork
from dynamic_patrol_marl.env.task_generator import Task
from dynamic_patrol_marl.env.drone import Drone


@dataclass
class Sortie:
    drone_id: str
    launch_node: int
    task_ids: List[int]
    task_sequence: List[int]
    recovery_node: int
    drone_time_s: float
    vehicle_time_s: float
    finish_time_s: float
    energy: float
    score: float
    feasible: bool
    reason: str = "ok"
    segments: List[Dict] = None
    tasks_per_sortie: int = 1
    vehicle_anchor_node: Optional[int] = None
    vehicle_queue_time_s: float = 0.0
    flight_service_energy: float = 0.0
    battery_at_eta: float = 0.0
    base_feasible: bool = True
    insertion_evaluation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["segments"] = self.segments or []
        d.pop("insertion_evaluation", None)
        return d


class SortieGenerator:
    def __init__(self, cfg: Dict, network: RoadNetwork):
        self.cfg = cfg
        self.network = network
        self.sortie_cfg = cfg["sortie"]
        self.agent_cfg = cfg["agents"]
        self.ablation_cfg = cfg.get("ablation", {})
        self.drone_speed = float(cfg["network"].get("drone_speed_mps", 14.0))
        self.fly_power = float(self.agent_cfg.get("drone_fly_power", 1.0))
        self.hover_power = float(self.agent_cfg.get("drone_hover_power", 0.65))
        self.service_power = float(self.agent_cfg.get("drone_service_power", 0.85))

    def generate_for_drone(
        self,
        drone: Drone,
        vehicle_position: int,
        now: float,
        available_tasks: Sequence[Task],
        max_tasks: Optional[int] = None,
        top_k: Optional[int] = None,
        no_priority: bool = False,
        recovery_anchor_node: Optional[int] = None,
        vehicle_queue_time_s: float = 0.0,
        allow_recovery_reschedule: bool = False,
    ) -> List[Sortie]:
        if not drone.is_idle() or drone.position != vehicle_position:
            return []
        top_k = int(self.sortie_cfg.get("top_k_per_drone", 8) if top_k is None else top_k)
        max_tasks = int(self.sortie_cfg.get("max_tasks_per_sortie", 3) if max_tasks is None else max_tasks)
        if not bool(self.sortie_cfg.get("allow_multi_task", True)):
            max_tasks = 1
        # Keep candidate enumeration controlled.
        tasks = [t for t in available_tasks if t.executor in {"drone", "both"}]
        if bool(self.ablation_cfg.get("priority_urgency_score", True)):
            tasks = sorted(
                tasks,
                key=lambda t: (
                    -(1 if no_priority else t.priority),
                    t.deadline,
                    self.network.get_drone_time(vehicle_position, t.start_node),
                ),
            )
        else:
            tasks = sorted(tasks, key=lambda t: (self.network.get_drone_time(vehicle_position, t.start_node), t.id))
        tasks = tasks[: int(self.sortie_cfg.get("candidate_pool_tasks", 12))]
        recovery_anchor_node = int(vehicle_position if recovery_anchor_node is None else recovery_anchor_node)
        candidates: List[Sortie] = []
        for length in range(1, max_tasks + 1):
            for subset in combinations(tasks, length):
                # Only permute small subsets. Sort by deadlines first, then optionally test reverse for local variety.
                orders = [tuple(sorted(subset, key=lambda t: t.deadline))]
                if length > 1:
                    rev = tuple(reversed(orders[0]))
                    if rev != orders[0]:
                        orders.append(rev)
                if length <= 3:
                    # Include a few permutations with high score potential.
                    for p in permutations(subset):
                        if p not in orders and len(orders) < 4:
                            orders.append(p)
                for order in orders:
                    rec_nodes = self._recovery_candidates(order)
                    for rec in rec_nodes:
                        s = self.evaluate(
                            drone,
                            vehicle_position,
                            now,
                            list(order),
                            rec,
                            no_priority=no_priority,
                            vehicle_anchor_node=recovery_anchor_node,
                            vehicle_queue_time_s=vehicle_queue_time_s,
                        )
                        if s.feasible or (allow_recovery_reschedule and s.base_feasible):
                            candidates.append(s)
        # Diversity: avoid returning only same first task.
        candidates.sort(key=lambda s: s.score, reverse=True)
        diverse: List[Sortie] = []
        seen_signature = set()
        for s in candidates:
            sig = (tuple(s.task_ids), s.recovery_node)
            first_sig = s.task_ids[0]
            if sig in seen_signature:
                continue
            if sum(1 for x in diverse if x.task_ids and x.task_ids[0] == first_sig) >= max(2, top_k // 3):
                continue
            diverse.append(s)
            seen_signature.add(sig)
            if len(diverse) >= top_k:
                break
        return diverse

    def generate_all(
        self,
        drones: Sequence[Drone],
        vehicle_position: int,
        now: float,
        available_tasks: Sequence[Task],
        max_tasks: Optional[int] = None,
        top_k_per_drone: Optional[int] = None,
        no_priority: bool = False,
        recovery_anchor_node: Optional[int] = None,
        vehicle_queue_time_s: float = 0.0,
        allow_recovery_reschedule: bool = False,
    ) -> List[Sortie]:
        out: List[Sortie] = []
        for d in drones:
            out.extend(
                self.generate_for_drone(
                    d,
                    vehicle_position,
                    now,
                    available_tasks,
                    max_tasks,
                    top_k_per_drone,
                    no_priority,
                    recovery_anchor_node=recovery_anchor_node,
                    vehicle_queue_time_s=vehicle_queue_time_s,
                    allow_recovery_reschedule=allow_recovery_reschedule,
                )
            )
        return sorted(out, key=lambda s: s.score, reverse=True)

    def _recovery_candidates(self, tasks: Sequence[Task]) -> List[int]:
        if not tasks:
            return []
        k = int(self.sortie_cfg.get("recovery_candidates", 4))
        required_site_nodes = self._required_vehicle_site_nodes(tasks)
        if required_site_nodes:
            return sorted(required_site_nodes)
        center_nodes = self._center_recovery_candidates(k)
        mode = str(self.sortie_cfg.get("recovery_candidate_mode", "center_only")).lower()
        if center_nodes and mode not in {"mixed", "task_near", "task_near_first"}:
            return center_nodes
        task_near_nodes = []
        # Near the final task is often a good rendezvous point.
        final_arc = tasks[-1].target_arcs[-1]
        task_near_nodes.extend(self.network.nearest_nodes_to_arc(final_arc, k=k))
        # Include both endpoints because inspection tasks are direction-free for
        # rendezvous purposes.
        for task in tasks:
            for u, v in task.target_arcs:
                task_near_nodes.extend([int(u), int(v)])
        nodes = task_near_nodes + center_nodes if mode in {"mixed", "task_near", "task_near_first"} else task_near_nodes
        seen, out = set(), []
        for n in nodes:
            if n not in seen:
                out.append(int(n))
                seen.add(int(n))
        return out[:k]

    def _center_recovery_candidates(self, k: int) -> List[int]:
        radius = float(self.sortie_cfg.get("recovery_center_radius_m", 0.0))
        if radius <= 0.0:
            return []
        center = int(self.network.patrol_center)
        ranked = sorted(
            (int(n) for n in self.network.nodes),
            key=lambda n: (self.network.euclidean_distance(center, n), n),
        )
        within = [n for n in ranked if self.network.euclidean_distance(center, n) <= radius + 1e-9]
        if center not in within:
            within.insert(0, center)
        return within[: max(1, int(k))]

    def _required_vehicle_site_nodes(self, tasks: Sequence[Task]) -> set[int]:
        return {
            int(t.metadata.get("vehicle_site_node", t.end_node))
            for t in tasks
            if bool(t.metadata.get("requires_vehicle_onsite", False))
        }

    def _direction_free_service_arcs(self, current: int, task: Task) -> List[Tuple[int, int]]:
        arcs = [(int(u), int(v)) for u, v in task.target_arcs]
        if len(arcs) <= 1:
            if not arcs:
                return []
            u, v = arcs[0]
            to_u = self.network.get_drone_time(current, u)
            to_v = self.network.get_drone_time(current, v)
            return [(v, u)] if to_v < to_u else [(u, v)]

        forward = arcs
        backward = [(v, u) for u, v in reversed(arcs)]
        forward_time = self.network.get_drone_time(current, forward[0][0])
        backward_time = self.network.get_drone_time(current, backward[0][0])
        return backward if backward_time < forward_time else forward

    def evaluate(
        self,
        drone: Drone,
        launch_node: int,
        now: float,
        tasks: Sequence[Task],
        recovery_node: int,
        no_priority: bool = False,
        vehicle_anchor_node: Optional[int] = None,
        vehicle_queue_time_s: float = 0.0,
    ) -> Sortie:
        segments: List[Dict] = []
        required_site_nodes = self._required_vehicle_site_nodes(tasks)
        if len(required_site_nodes) > 1:
            return Sortie(
                drone.id,
                int(launch_node),
                [int(t.id) for t in tasks],
                [int(t.id) for t in tasks],
                int(recovery_node),
                math.inf,
                math.inf,
                math.inf,
                math.inf,
                -1e9,
                False,
                "multiple_vehicle_sites_required",
                [],
                len(tasks),
                int(launch_node if vehicle_anchor_node is None else vehicle_anchor_node),
                float(vehicle_queue_time_s),
                0.0,
                float(drone.battery),
                False,
            )
        if required_site_nodes and int(recovery_node) not in required_site_nodes:
            return Sortie(
                drone.id,
                int(launch_node),
                [int(t.id) for t in tasks],
                [int(t.id) for t in tasks],
                int(recovery_node),
                math.inf,
                math.inf,
                math.inf,
                math.inf,
                -1e9,
                False,
                "vehicle_site_required",
                [],
                len(tasks),
                int(launch_node if vehicle_anchor_node is None else vehicle_anchor_node),
                float(vehicle_queue_time_s),
                0.0,
                float(drone.battery),
                False,
            )
        current = int(launch_node)
        elapsed = 0.0
        energy = 0.0
        task_ids: List[int] = []
        deadline_violation = False
        for task in tasks:
            service_arcs = self._direction_free_service_arcs(current, task)
            arc_count = max(1, len(service_arcs))
            service_duration = float(task.service_time) / float(arc_count)
            for arc_index, (service_from, service_to) in enumerate(service_arcs):
                t_transfer = self.network.get_drone_time(current, service_from)
                if t_transfer > 0:
                    segments.append({"type": "fly", "from": current, "to": service_from, "duration": t_transfer, "task_id": None})
                    elapsed += t_transfer
                    energy += t_transfer * self.fly_power
                segments.append(
                    {
                        "type": "service",
                        "from": service_from,
                        "to": service_to,
                        "duration": service_duration,
                        "task_id": task.id,
                        "arc_index": arc_index,
                        "arc_count": arc_count,
                        "service_final": arc_index == arc_count - 1,
                    }
                )
                elapsed += service_duration
                energy += service_duration * self.service_power
                current = service_to
            task_ids.append(task.id)
            if now + elapsed > task.deadline + 1e-9:
                deadline_violation = True
        # Return to recovery node.
        t_return = self.network.get_drone_time(current, recovery_node)
        if t_return > 0:
            segments.append({"type": "fly", "from": current, "to": int(recovery_node), "duration": t_return, "task_id": None})
            elapsed += t_return
            energy += t_return * self.fly_power
        flight_service_energy = float(energy)
        battery_at_eta = float(drone.battery - flight_service_energy)
        anchor = int(launch_node if vehicle_anchor_node is None else vehicle_anchor_node)
        vehicle_leg_time = self.network.get_vehicle_time(anchor, recovery_node)
        vehicle_time = float(vehicle_queue_time_s) + float(vehicle_leg_time)
        if not math.isfinite(vehicle_time):
            return Sortie(
                drone.id, int(launch_node), task_ids, task_ids, int(recovery_node), elapsed,
                vehicle_time, math.inf, energy, -1e9, False, "vehicle_unreachable", segments,
                len(task_ids), anchor, float(vehicle_queue_time_s), flight_service_energy,
                battery_at_eta, False,
            )
        drone_wait = max(0.0, vehicle_time - elapsed)
        vehicle_wait = max(0.0, elapsed - vehicle_time)
        energy += drone_wait * self.hover_power
        finish = now + max(elapsed, vehicle_time)
        battery_capacity = drone.battery
        base_feasible = True
        feasible = True
        reason = "ok"
        if deadline_violation:
            base_feasible = False
            feasible = False
            reason = "deadline"
        if flight_service_energy > battery_capacity + 1e-9:
            base_feasible = False
            feasible = False
            reason = "battery"
        elif energy > battery_capacity + 1e-9:
            feasible = False
            reason = "battery"
        if bool(self.ablation_cfg.get("priority_urgency_score", True)):
            priority_sum = sum((1.0 if no_priority else t.priority) for t in tasks)
            urgency_bonus = sum(max(0.0, 1.0 - (t.deadline - now) / max(1.0, self.cfg["simulation"].get("horizon_s", 3600))) for t in tasks)
        else:
            priority_sum = 0.0
            urgency_bonus = 0.0
        multitask_bonus = 0.6 * max(0, len(tasks) - 1)
        score = 8.0 * priority_sum + 5.0 * urgency_bonus + 3.0 * multitask_bonus - 0.01 * max(elapsed, vehicle_time) - 0.002 * energy - 0.01 * (drone_wait + vehicle_wait)
        return Sortie(
            drone_id=drone.id,
            launch_node=int(launch_node),
            task_ids=task_ids,
            task_sequence=task_ids,
            recovery_node=int(recovery_node),
            drone_time_s=float(elapsed),
            vehicle_time_s=float(vehicle_time),
            finish_time_s=float(finish),
            energy=float(energy),
            score=float(score),
            feasible=bool(feasible),
            reason=reason,
            segments=segments,
            tasks_per_sortie=len(task_ids),
            vehicle_anchor_node=anchor,
            vehicle_queue_time_s=float(vehicle_queue_time_s),
            flight_service_energy=flight_service_energy,
            battery_at_eta=battery_at_eta,
            base_feasible=bool(base_feasible),
        )
