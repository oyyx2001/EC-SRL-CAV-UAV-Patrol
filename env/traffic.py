from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class TrafficSnapshot:
    epoch: int
    time_s: float
    global_multiplier: float
    mean_multiplier: float
    p95_multiplier: float
    maximum_multiplier: float
    congested_arc_fraction: float
    planner_recomputed: bool
    stage_index: int = -1
    stage_mean_speed_mps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traffic_epoch": int(self.epoch),
            "traffic_time_s": float(self.time_s),
            "traffic_global_multiplier": float(self.global_multiplier),
            "traffic_mean_multiplier": float(self.mean_multiplier),
            "traffic_p95_multiplier": float(self.p95_multiplier),
            "traffic_maximum_multiplier": float(self.maximum_multiplier),
            "traffic_congested_arc_fraction": float(self.congested_arc_fraction),
            "traffic_planner_recomputed": bool(self.planner_recomputed),
            "traffic_stage_index": int(self.stage_index),
            "traffic_stage_mean_speed_mps": float(self.stage_mean_speed_mps),
        }


class TimeVaryingTraffic:
    """Seeded, piecewise-constant road traffic shared by planning ablations.

    ``actual_vehicle_time_s`` is the physical traversal time used by the CAV.
    ``vehicle_time_s`` remains the planner-facing weight.  In the adaptive
    condition the latter is refreshed and shortest paths are recomputed; in
    the stale-planner condition it remains at the nominal value.
    """

    def __init__(self, network: Any, cfg: Dict[str, Any], seed: int):
        self.network = network
        self.cfg = dict(cfg.get("time_varying_traffic", {}) or {})
        self.enabled = bool(self.cfg.get("enabled", False))
        self.mode = str(self.cfg.get("mode", "network_multiplier")).lower()
        self.staged_speed = self.enabled and self.mode in {
            "staged_random_speed",
            "piecewise_random_speed",
        }
        self.planner_recompute = bool(self.cfg.get("planner_recompute", False))
        self.update_interval_s = float(self.cfg.get("update_interval_s", 900.0))
        self.seed = int(seed)
        self.seed_offset = int(self.cfg.get("seed_offset", 73031))
        self.moderate_fraction = float(self.cfg.get("moderate_fraction", 0.25))
        self.severe_fraction = float(self.cfg.get("severe_fraction", 0.08))
        self.moderate_multiplier = float(self.cfg.get("moderate_multiplier", 1.35))
        self.severe_multiplier = float(self.cfg.get("severe_multiplier", 1.85))
        self.jitter_std = float(self.cfg.get("jitter_std", 0.05))
        self.profile = list(self.cfg.get("temporal_profile", []) or [])
        self.speed_stages = list(self.cfg.get("speed_stages", []) or [])
        self.speed_multiplier_mean = float(self.cfg.get("speed_multiplier_mean", 1.0))
        self.speed_multiplier_std = float(self.cfg.get("speed_multiplier_std", 0.1))
        self.speed_multiplier_min = float(self.cfg.get("speed_multiplier_min", 0.8))
        self.speed_multiplier_max = float(self.cfg.get("speed_multiplier_max", 1.2))
        execution_root = dict(cfg.get("operational_execution_uncertainty", {}) or {})
        self.cav_execution_cfg = dict(execution_root.get("cav_travel_time", {}) or {})
        self.execution_uncertainty_enabled = bool(execution_root.get("enabled", False)) and bool(
            self.cav_execution_cfg.get("enabled", False)
        )
        self.cav_execution_delta = float(self.cav_execution_cfg.get("delta", 0.0))
        if not math.isfinite(self.cav_execution_delta) or not 0.0 <= self.cav_execution_delta < 1.0:
            raise ValueError("operational_execution_uncertainty.cav_travel_time.delta must be in [0, 1)")
        if str(self.cav_execution_cfg.get("distribution", "uniform")).lower() != "uniform":
            raise ValueError("CAV travel-time execution uncertainty currently supports only uniform")
        self.speed_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.seed_offset, 0x434156])
        )
        # Keep this stream separate from the existing staged-traffic stream so
        # delta=0 reproduces the nominal evaluation exactly.
        self.execution_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.seed_offset, 0x45584543])
        )
        self.realization_log: List[Dict[str, Any]] = []
        self.revision = 0
        self.current_epoch = -1
        self.current_snapshot: TrafficSnapshot | None = None
        self.nominal_times: Dict[Tuple[int, int], float] = {}
        self.actual_times: Dict[Tuple[int, int], float] = {}
        self._validate()
        for u, v, data in self.network.G.edges(data=True):
            edge = (int(u), int(v))
            nominal = float(data["vehicle_time_s"])
            self.nominal_times[edge] = nominal
            self.actual_times[edge] = nominal
            data["nominal_vehicle_time_s"] = nominal
            data["actual_vehicle_time_s"] = nominal

    def _validate(self) -> None:
        if not self.enabled:
            return
        if self.staged_speed:
            if not self.speed_stages:
                raise ValueError("time_varying_traffic.speed_stages must not be empty")
            previous_end = None
            for index, row in enumerate(self.speed_stages):
                start = float(row.get("start_s", 0.0))
                end = float(row.get("end_s", math.inf))
                speed_kmh = float(row.get("mean_speed_kmh", 0.0))
                if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                    raise ValueError(f"invalid staged traffic interval at index {index}")
                if speed_kmh <= 0.0 or not math.isfinite(speed_kmh):
                    raise ValueError(f"invalid staged traffic speed at index {index}")
                if previous_end is not None and abs(start - previous_end) > 1e-9:
                    raise ValueError("staged traffic intervals must be contiguous")
                previous_end = end
            if float(self.speed_stages[0].get("start_s", 0.0)) > 1e-9:
                raise ValueError("staged traffic must begin at time zero")
            if not (
                0.0 < self.speed_multiplier_min
                <= self.speed_multiplier_mean
                <= self.speed_multiplier_max
            ):
                raise ValueError("invalid staged traffic speed-multiplier bounds")
            if self.speed_multiplier_std <= 0.0 or not math.isfinite(self.speed_multiplier_std):
                raise ValueError("speed_multiplier_std must be finite and positive")
            return
        if not math.isfinite(self.update_interval_s) or self.update_interval_s <= 0.0:
            raise ValueError("time_varying_traffic.update_interval_s must be positive")
        if self.moderate_fraction < 0.0 or self.severe_fraction < 0.0:
            raise ValueError("traffic fractions must be non-negative")
        if self.moderate_fraction + self.severe_fraction > 1.0 + 1e-12:
            raise ValueError("moderate_fraction + severe_fraction must not exceed 1")
        if self.moderate_multiplier < 1.0 or self.severe_multiplier < self.moderate_multiplier:
            raise ValueError("traffic multipliers must satisfy severe >= moderate >= 1")
        if self.jitter_std < 0.0 or not math.isfinite(self.jitter_std):
            raise ValueError("time_varying_traffic.jitter_std must be finite and non-negative")

    def update_times(self, horizon_s: float) -> List[float]:
        if not self.enabled:
            return []
        if self.staged_speed:
            return sorted(
                {
                    float(row.get("start_s", 0.0))
                    for row in self.speed_stages
                    if 1e-9 < float(row.get("start_s", 0.0)) < float(horizon_s) - 1e-9
                }
            )
        count = int(math.floor((float(horizon_s) - 1e-9) / self.update_interval_s))
        return [self.update_interval_s * index for index in range(1, count + 1)]

    def _speed_stage(self, time_s: float) -> Tuple[int, Dict[str, Any]]:
        value = float(time_s)
        for index, row in enumerate(self.speed_stages):
            start = float(row.get("start_s", 0.0))
            end = float(row.get("end_s", math.inf))
            if start <= value < end or (
                index == len(self.speed_stages) - 1 and abs(value - end) <= 1e-9
            ):
                return index, row
        raise ValueError(f"No staged traffic speed is defined at t={value}")

    def mean_speed_mps(self, time_s: float) -> float:
        if not self.staged_speed:
            return float(self.network.cfg["network"].get("vehicle_speed_mps", 8.3))
        _, row = self._speed_stage(time_s)
        return float(row["mean_speed_kmh"]) / 3.6

    def next_stage_change_after(self, time_s: float) -> float | None:
        if not self.staged_speed:
            return None
        value = float(time_s)
        future = [
            float(row.get("start_s", 0.0))
            for row in self.speed_stages
            if float(row.get("start_s", 0.0)) > value + 1e-9
        ]
        return min(future) if future else None

    def _sample_speed_multiplier(self) -> float:
        for _ in range(1024):
            value = float(self.speed_rng.normal(self.speed_multiplier_mean, self.speed_multiplier_std))
            if self.speed_multiplier_min <= value <= self.speed_multiplier_max:
                return value
        return float(
            np.clip(
                self.speed_rng.normal(self.speed_multiplier_mean, self.speed_multiplier_std),
                self.speed_multiplier_min,
                self.speed_multiplier_max,
            )
        )

    def cav_execution_multiplier(self) -> float:
        """Sample one hidden CAV travel-time multiplier for a new edge."""
        if not self.execution_uncertainty_enabled or self.cav_execution_delta <= 1e-15:
            return 1.0
        delta = float(self.cav_execution_delta)
        return float(self.execution_rng.uniform(1.0 - delta, 1.0 + delta))

    def sample_edge_motion(
        self,
        u: int,
        v: int,
        entry_time_s: float,
        distance_m: float,
    ) -> Dict[str, float]:
        """Realize one physical CAV speed only after the edge is entered."""
        if not self.staged_speed:
            base_duration = self.actual_edge_time(u, v)
            execution_multiplier = self.cav_execution_multiplier()
            duration = float(base_duration) * execution_multiplier
            actual_speed = float(distance_m) / max(duration, 1e-9)
            self.realization_log.append(
                {
                    "time_s": float(entry_time_s),
                    "from_node": int(u),
                    "to_node": int(v),
                    "stage_index": -1,
                    "mean_speed_mps": float(distance_m) / max(float(base_duration), 1e-9),
                    "speed_multiplier": 1.0,
                    "actual_speed_mps": actual_speed,
                    "remaining_distance_m": float(distance_m),
                    "base_duration_s": float(base_duration),
                    "realized_duration_s": float(duration),
                    "cav_execution_multiplier": float(execution_multiplier),
                    "cav_execution_delta": float(self.cav_execution_delta),
                }
            )
            return {
                "stage_index": -1.0,
                "mean_speed_mps": float(distance_m) / max(float(base_duration), 1e-9),
                "speed_multiplier": 1.0,
                "actual_speed_mps": actual_speed,
                "duration_s": float(duration),
                "cav_execution_multiplier": float(execution_multiplier),
            }
        stage_index, _ = self._speed_stage(entry_time_s)
        mean_speed = self.mean_speed_mps(entry_time_s)
        multiplier = self._sample_speed_multiplier()
        base_speed = mean_speed * multiplier
        base_duration = float(distance_m) / max(base_speed, 1e-9)
        execution_multiplier = self.cav_execution_multiplier()
        duration = base_duration * execution_multiplier
        actual_speed = float(distance_m) / max(duration, 1e-9)
        record = {
            "time_s": float(entry_time_s),
            "from_node": int(u),
            "to_node": int(v),
            "stage_index": int(stage_index),
            "mean_speed_mps": float(mean_speed),
            "speed_multiplier": float(multiplier),
            "actual_speed_mps": float(actual_speed),
            "remaining_distance_m": float(distance_m),
            "base_duration_s": float(base_duration),
            "realized_duration_s": float(duration),
            "cav_execution_multiplier": float(execution_multiplier),
            "cav_execution_delta": float(self.cav_execution_delta),
        }
        self.realization_log.append(record)
        return {
            "stage_index": float(stage_index),
            "mean_speed_mps": float(mean_speed),
            "speed_multiplier": float(multiplier),
            "actual_speed_mps": float(actual_speed),
            "duration_s": float(duration),
            "cav_execution_multiplier": float(execution_multiplier),
        }

    def _global_multiplier(self, time_s: float) -> float:
        for row in self.profile:
            start = float(row.get("start_s", 0.0))
            end = float(row.get("end_s", math.inf))
            if start <= float(time_s) < end:
                value = float(row.get("multiplier", 1.0))
                if value < 1.0 or not math.isfinite(value):
                    raise ValueError("temporal traffic multipliers must be finite and >= 1")
                return value
        return 1.0

    def _physical_road_groups(self) -> List[List[Tuple[int, int]]]:
        grouped: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for edge in sorted(self.nominal_times):
            key = tuple(sorted(edge))
            grouped.setdefault(key, []).append(edge)
        return [grouped[key] for key in sorted(grouped)]

    def update(self, time_s: float) -> TrafficSnapshot | None:
        if not self.enabled:
            return None
        if self.staged_speed:
            stage_index, _ = self._speed_stage(time_s)
            if stage_index == self.current_epoch:
                return None
            mean_speed = self.mean_speed_mps(time_s)
            factors: List[float] = []
            for edge, nominal in self.nominal_times.items():
                data = self.network.G[edge[0]][edge[1]]
                planned = float(data["length_m"]) / max(mean_speed, 1e-9)
                factor = planned / max(nominal, 1e-9)
                self.actual_times[edge] = planned
                data["actual_vehicle_time_s"] = planned
                data["traffic_multiplier"] = factor
                data["vehicle_time_s"] = planned
                factors.append(factor)
            self.network._compute_shortest_paths()
            self.current_epoch = stage_index
            self.revision += 1
            values = np.asarray(factors, dtype=float)
            self.current_snapshot = TrafficSnapshot(
                epoch=stage_index,
                time_s=float(time_s),
                global_multiplier=float(values.mean()),
                mean_multiplier=float(values.mean()),
                p95_multiplier=float(np.percentile(values, 95)),
                maximum_multiplier=float(values.max()),
                congested_arc_fraction=0.0,
                planner_recomputed=True,
                stage_index=stage_index,
                stage_mean_speed_mps=mean_speed,
            )
            return self.current_snapshot
        epoch = int(math.floor((float(time_s) + 1e-9) / self.update_interval_s))
        if epoch == self.current_epoch:
            return None
        groups = self._physical_road_groups()
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.seed_offset, epoch])
        )
        order = rng.permutation(len(groups))
        severe_count = min(len(groups), int(round(len(groups) * self.severe_fraction)))
        moderate_count = min(
            len(groups) - severe_count,
            int(round(len(groups) * self.moderate_fraction)),
        )
        severe_groups = set(int(value) for value in order[:severe_count])
        moderate_groups = set(
            int(value) for value in order[severe_count : severe_count + moderate_count]
        )
        global_multiplier = self._global_multiplier(time_s)
        factors: List[float] = []
        for group_index, group in enumerate(groups):
            local = 1.0
            if group_index in severe_groups:
                local = self.severe_multiplier
            elif group_index in moderate_groups:
                local = self.moderate_multiplier
            jitter = float(
                rng.lognormal(mean=-0.5 * self.jitter_std**2, sigma=self.jitter_std)
            )
            factor = max(1.0, global_multiplier * local * jitter)
            for edge in group:
                nominal = self.nominal_times[edge]
                actual = nominal * factor
                self.actual_times[edge] = actual
                data = self.network.G[edge[0]][edge[1]]
                data["actual_vehicle_time_s"] = actual
                data["traffic_multiplier"] = factor
                data["vehicle_time_s"] = actual if self.planner_recompute else nominal
                factors.append(factor)
        if self.planner_recompute:
            self.network._compute_shortest_paths()
        self.current_epoch = epoch
        self.revision += 1
        values = np.asarray(factors, dtype=float)
        self.current_snapshot = TrafficSnapshot(
            epoch=epoch,
            time_s=float(time_s),
            global_multiplier=float(global_multiplier),
            mean_multiplier=float(values.mean()),
            p95_multiplier=float(np.percentile(values, 95)),
            maximum_multiplier=float(values.max()),
            congested_arc_fraction=float(np.mean(values > global_multiplier * 1.05)),
            planner_recomputed=bool(self.planner_recompute),
        )
        return self.current_snapshot

    def actual_edge_time(self, u: int, v: int) -> float:
        edge = (int(u), int(v))
        return float(self.actual_times.get(edge, self.network.G[edge[0]][edge[1]]["vehicle_time_s"]))
