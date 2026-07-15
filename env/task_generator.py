from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import numpy as np
import pandas as pd

from .road_network import RoadNetwork

TASK_ROUTINE = "routine"
TASK_INCIDENT = "incident"
TASK_CONGESTION = "congestion"


@dataclass
class Task:
    id: int
    type: str
    target_arcs: List[Tuple[int, int]]
    release_time: float
    deadline: float
    priority: float
    service_time: float
    executor: str = "drone"  # drone, vehicle, both
    status: str = "unreleased"  # unreleased, unassigned, assigned, executing, done, expired
    assigned_to: Optional[str] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    risk: float = 0.0
    metadata: Dict = field(default_factory=dict)

    @property
    def start_node(self) -> int:
        return int(self.target_arcs[0][0])

    @property
    def end_node(self) -> int:
        return int(self.target_arcs[-1][1])

    @property
    def is_active(self) -> bool:
        return self.status in {"unassigned", "assigned", "executing"}

    @property
    def is_available(self) -> bool:
        return self.status == "unassigned"

    def to_record(self) -> Dict:
        d = asdict(self)
        d["target_arcs"] = ";".join([f"{u}-{v}" for u, v in self.target_arcs])
        return d


class TaskGenerator:
    """Generate heterogeneous dynamic patrol tasks with reproducible random seed."""

    def __init__(self, cfg: Dict, network: RoadNetwork, seed: Optional[int] = None):
        self.cfg = cfg
        self.network = network
        self.seed = int(cfg["project"].get("seed", 42) if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)
        self.task_cfg = cfg["tasks"]
        self.next_id = 0

    def generate_episode_tasks(self) -> List[Task]:
        horizon = float(self.cfg["simulation"].get("horizon_s", 3600))
        tasks: List[Task] = []
        tasks.extend(self._routine_tasks(horizon))
        tasks.extend(self._poisson_tasks(horizon, TASK_INCIDENT))
        tasks.extend(self._poisson_tasks(horizon, TASK_CONGESTION))
        tasks.sort(key=lambda t: (t.release_time, t.id))
        return tasks

    def _new_id(self) -> int:
        tid = self.next_id
        self.next_id += 1
        return tid

    def _deadline(self, typ: str, release: float) -> float:
        return release + float(self.task_cfg["deadline_s"][typ])

    def _service_time(self, typ: str, arcs: Sequence[Tuple[int, int]]) -> float:
        base = float(self.task_cfg["service_time_s"][typ])
        if typ == TASK_CONGESTION:
            return base * max(1, len(arcs))
        return base

    def _priority(self, typ: str, risk: float) -> float:
        base = float(self.task_cfg["priority"][typ])
        return base + 2.0 * float(risk)

    def _routine_tasks(self, horizon: float) -> List[Task]:
        interval = float(self.task_cfg.get("routine_interval_s", 900))
        high_arcs = self.network.high_risk_arcs(max(4, int(len(self.network.arcs) * 0.15)))
        tasks: List[Task] = []
        nominal_time = 0.0
        idx = 0
        while nominal_time < horizon:
            arc = high_arcs[idx % len(high_arcs)]
            risk = self.network.arc_risk(arc)
            # Routine patrol requirements are released periodically during the
            # episode. The nominal time is the actual release time.
            tasks.append(Task(
                id=self._new_id(),
                type=TASK_ROUTINE,
                target_arcs=[arc],
                release_time=nominal_time,
                deadline=self._deadline(TASK_ROUTINE, nominal_time),
                priority=self._priority(TASK_ROUTINE, risk),
                service_time=self._service_time(TASK_ROUTINE, [arc]),
                executor="drone",
                risk=risk,
                metadata={"periodic_index": idx, "nominal_time": nominal_time},
            ))
            nominal_time += interval
            idx += 1
        return tasks

    def _poisson_tasks(self, horizon: float, typ: str) -> List[Task]:
        rate = float(self.task_cfg.get(f"{typ}_rate_per_hour", 0))
        expected = max(0.0, rate * horizon / 3600.0)
        n = int(self.rng.poisson(expected))
        if n <= 0:
            return []
        releases = sorted(self.rng.uniform(0, horizon * 0.92, size=n).tolist())
        tasks: List[Task] = []
        concentration = str(self.task_cfg.get("spatial_concentration", "medium")).lower()
        if concentration == "low":
            # Uniform spatial support; task count and arrival process are unchanged.
            risk_arcs = sorted(self.network.arcs)
            weights = np.ones(len(risk_arcs), dtype=float)
        elif concentration == "high":
            all_arcs = self.network.high_risk_arcs()
            fraction = float(self.task_cfg.get("hotspot_fraction", 0.12))
            hotspot_count = max(1, int(math.ceil(len(all_arcs) * fraction)))
            risk_arcs = all_arcs[:hotspot_count]
            weights = np.array([self.network.arc_risk(e) + 0.05 for e in risk_arcs], dtype=float)
        else:
            risk_arcs = self.network.high_risk_arcs()
            weights = np.array([self.network.arc_risk(e) + 0.05 for e in risk_arcs], dtype=float)
        weights = weights / weights.sum()
        for rel in releases:
            arc = risk_arcs[int(self.rng.choice(len(risk_arcs), p=weights))]
            if typ == TASK_CONGESTION:
                arcs = self._neighbor_arc_group(arc)
            else:
                arcs = [arc]
            executor = "drone"
            requires_vehicle_onsite = bool(
                typ == TASK_INCIDENT
                and self.rng.random() < float(self.task_cfg.get("incident_vehicle_required_prob", 0.0))
            )
            risk = float(np.mean([self.network.arc_risk(a) for a in arcs]))
            tasks.append(Task(
                id=self._new_id(),
                type=typ,
                target_arcs=list(arcs),
                release_time=float(rel),
                deadline=self._deadline(typ, float(rel)),
                priority=self._priority(typ, risk),
                service_time=self._service_time(typ, arcs),
                executor=executor,
                risk=risk,
                metadata={
                    "generator": "poisson",
                    "requires_vehicle_onsite": requires_vehicle_onsite,
                    "vehicle_site_node": int(arcs[-1][1]) if requires_vehicle_onsite else None,
                },
            ))
        return tasks

    def _neighbor_arc_group(self, seed_arc: Tuple[int, int]) -> List[Tuple[int, int]]:
        max_arcs = int(self.task_cfg.get("congestion_group_max_arcs", 3))
        arcs = [seed_arc]
        current = seed_arc[1]
        for _ in range(max_arcs - 1):
            outs = list(self.network.G.out_edges(current))
            if not outs:
                break
            # Pick outgoing arc with high risk/congestion.
            outs = sorted(outs, key=lambda e: self.network.G[e[0]][e[1]].get("risk", 0.0), reverse=True)
            nxt = outs[int(self.rng.integers(0, min(len(outs), 3)))]
            if nxt in arcs:
                break
            arcs.append(nxt)
            current = nxt[1]
        return arcs

    @staticmethod
    def to_dataframe(tasks: Sequence[Task]) -> pd.DataFrame:
        return pd.DataFrame([t.to_record() for t in tasks])

    @staticmethod
    def validate(tasks: Sequence[Task]) -> Dict[str, float]:
        release_ok = all(t.release_time <= t.deadline for t in tasks)
        service_ok = all(t.service_time > 0 for t in tasks)
        priority_ok = all(t.priority > 0 for t in tasks)
        return {
            "num_tasks": len(tasks),
            "release_deadline_ok": float(release_ok),
            "service_positive_ok": float(service_ok),
            "priority_positive_ok": float(priority_ok),
            "num_routine": sum(t.type == TASK_ROUTINE for t in tasks),
            "num_incident": sum(t.type == TASK_INCIDENT for t in tasks),
            "num_congestion": sum(t.type == TASK_CONGESTION for t in tasks),
        }
