from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import time
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - fallback keeps baseline code runnable without gymnasium.
    gym = None
    spaces = None

from dynamic_patrol_marl.env.simulator import Simulator
from dynamic_patrol_marl.env.events import Event, EventQueue
from dynamic_patrol_marl.planner.sortie_generator import Sortie
from dynamic_patrol_marl.planner.joint_plan import JointPlan, JointPlanCache
from dynamic_patrol_marl.rl.smdp import transition_discount


class SimpleDiscrete:
    def __init__(self, n: int):
        self.n = int(n)

    def sample(self) -> int:
        return int(np.random.randint(self.n))


class SimpleBox:
    def __init__(self, shape):
        self.shape = tuple(shape)


BaseEnv = gym.Env if gym is not None else object


class DynamicPatrolEnv(BaseEnv):
    """Gymnasium-like single-agent wrapper with action masking.

    Action = index of a candidate sortie at the current rolling-horizon decision epoch.
    0 is always wait/no-op. All infeasible candidates are masked before sampling.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Dict,
        scenario: str = "small",
        no_multitask: bool = False,
        no_priority: bool = False,
        random_episode_seed: bool = False,
        episode_seed_start: Optional[int] = None,
        seed_scheduler: Optional[Any] = None,
    ):
        if gym is not None:
            super().__init__()
        self.cfg = cfg
        self.scenario = scenario
        self.no_multitask = no_multitask
        self.no_priority = no_priority
        self.random_episode_seed = bool(random_episode_seed)
        self.episode_seed_start = int(cfg["project"].get("seed", 42) if episode_seed_start is None else episode_seed_start)
        self.episode_idx = 0
        self.seed_scheduler = seed_scheduler
        inferred_mode = "separate_control" if str(cfg.get("cav_control", {}).get("mode", "passive")).lower() == "active_rule_based" else "sortie_only"
        self.decision_mode = str(cfg.get("decision_mode", inferred_mode)).lower()
        self.plan_feature_dim = 38 if self.decision_mode == "joint_plan" else 6
        if self.decision_mode == "joint_plan":
            self.max_actions = 1 + int(cfg.get("joint_plan", {}).get("max_non_safe_plans", 24))
        else:
            self.max_actions = 1 + int(cfg["agents"].get("num_drones", 2)) * int(cfg["sortie"].get("top_k_per_drone", 8))
        self.max_tasks_obs = 10
        self.max_candidates_obs = self.max_actions if self.decision_mode == "joint_plan" else self.max_actions - 1
        self.global_obs_dim = 6 + int(cfg["agents"].get("num_drones", 2)) * 3 + self.max_tasks_obs * 6
        self.obs_dim = self.global_obs_dim + self.max_candidates_obs * self.plan_feature_dim
        self.action_space = spaces.Discrete(self.max_actions) if spaces else SimpleDiscrete(self.max_actions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32) if spaces else SimpleBox((self.obs_dim,))
        self.sim: Optional[Simulator] = None
        self.candidates: List[Sortie] = []
        self.joint_plan_cache: Optional[JointPlanCache] = None
        self._ground_action_committed_time: Optional[float] = None
        self.last_metrics = None
        time_cfg = cfg.get("time", {})
        self.time_mode = str(time_cfg.get("mode", cfg.get("time_mode", "fixed_interval"))).lower()
        self.event_cfg = dict(cfg.get("event_driven", {}))
        self.smdp_cfg = dict(cfg.get("smdp", {}))
        self.reference_interval_s = float(self.smdp_cfg.get("reference_interval_s", 60.0))
        self.gamma_reference = float(
            self.smdp_cfg.get(
                "gamma_reference",
                cfg.get("rl", {}).get("hyperparams", {}).get("gamma", 0.99),
            )
        )
        self.discount_mode = str(self.smdp_cfg.get("discount_mode", "continuous_time"))
        self.event_queue: Optional[EventQueue] = None
        self.transition_log: List[Dict[str, Any]] = []
        self.event_timeline: List[Dict[str, Any]] = []
        self.decision_index = 0
        self._decision_epoch_serial = 0
        self.meaningful_decisions = 0
        self.forced_wait_decisions = 0
        self.instantaneous_decisions = 0
        self.zero_duration_transitions = 0
        self.system_events_processed = 0
        self._instantaneous_time: Optional[float] = None
        self._instantaneous_count = 0
        self._dynamic_event_signatures: Dict[str, Tuple[int, float, str]] = {}
        self._delta_t_history: List[float] = []
        self._discount_history: List[float] = []
        self._pending_ground_trigger_events: List[str] = ["INITIAL_DECISION"]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if self.seed_scheduler is not None:
            sim_seed = int(self.seed_scheduler.sample_seed())
        elif seed is None and self.random_episode_seed:
            sim_seed = self.episode_seed_start + self.episode_idx
            self.episode_idx += 1
        else:
            sim_seed = int(self.cfg["project"].get("seed", 42) if seed is None else seed)
            if seed is not None and self.random_episode_seed:
                self.episode_idx = max(self.episode_idx, sim_seed - self.episode_seed_start + 1)
        self.sim = Simulator(self.cfg, seed=sim_seed, scenario=self.scenario)
        self.time_mode = self.sim.time_mode
        self.transition_log = []
        self.event_timeline = []
        self.decision_index = 0
        self._decision_epoch_serial = 0
        self.meaningful_decisions = 0
        self.forced_wait_decisions = 0
        self.instantaneous_decisions = 0
        self.zero_duration_transitions = 0
        self.system_events_processed = 0
        self._instantaneous_time = None
        self._instantaneous_count = 0
        self._dynamic_event_signatures = {}
        self._delta_t_history = []
        self._discount_history = []
        self._pending_ground_trigger_events = ["INITIAL_DECISION"]
        if self.time_mode == "event_driven":
            self._initialize_event_queue()
        self.joint_plan_cache = None
        self._ground_action_committed_time = None
        self._prepare_decision_epoch(["INITIAL_DECISION"])
        obs = self._observation()
        initial_mask = self.action_masks()
        info = {
            "action_mask": initial_mask,
            "time": self.sim.now,
            "decision_time": self.sim.now,
            "recovery_queue_len": len(self.sim.recovery_queue),
            "time_mode": self.time_mode,
            "valid_action_count": int(initial_mask.sum()),
            "valid_non_wait_action_count": int(initial_mask[1:].sum()),
        }
        return (obs, info) if gym is not None else obs

    def step(self, action: int):
        if self.time_mode == "event_driven":
            return self._step_event_driven(action)
        return self._step_fixed_interval(action)

    def _prepare_decision_epoch(self, trigger_events: Optional[List[str]] = None) -> None:
        assert self.sim is not None
        if self.decision_mode != "joint_plan":
            self.candidates = self.sim.get_candidates(
                no_multitask=self.no_multitask, no_priority=self.no_priority
            )
            return
        assert self.sim.joint_plan_generator is not None
        if self.joint_plan_cache is not None:
            cache = self.joint_plan_cache
            if (
                abs(float(cache.decision_time) - float(self.sim.now)) <= 1e-9
                and cache.state_fingerprint == self.sim.joint_plan_generator.state_fingerprint()
                and cache.schedule_versions == self.sim.joint_plan_generator.schedule_versions()
            ):
                return
        next_event_time = None
        if self.event_queue is not None:
            next_event_time = self.event_queue.peek_time(self.sim.now, self._event_version)
        same_timestamp = (
            self._ground_action_committed_time is not None
            and abs(float(self._ground_action_committed_time) - float(self.sim.now)) <= 1e-9
        )
        event_types = list(trigger_events or [])
        event_code = 0.0
        if event_types:
            event_code = (sum(ord(ch) for ch in event_types[0]) % 101) / 100.0
        self.joint_plan_cache = self.sim.joint_plan_generator.generate_joint_plans(
            {
                "decision_epoch_id": int(self._decision_epoch_serial),
                "event_sequence_id": int(self.system_events_processed),
                "next_event_time": next_event_time,
                "event_type_code": event_code,
                "instantaneous_dispatch": same_timestamp,
                "ground_action_committed_at_timestamp": same_timestamp,
            }
        )
        self._decision_epoch_serial += 1
        self.candidates = []

    def _step_fixed_interval(self, action: int):
        assert self.sim is not None, "call reset first"
        wall_start = time.perf_counter()
        decision_time = float(self.sim.now)
        use_interval_reward = str(
            self.cfg.get("reward", {}).get("accounting_mode", "legacy_cumulative")
        ).lower() == "interval_increment"
        before_reward_snapshot = self._reward_snapshot() if use_interval_reward else None
        mask = self.action_masks()
        illegal = False
        selected = None
        selected_plan = None
        if action < 0 or action >= self.max_actions or not bool(mask[action]):
            illegal = True
            self.sim.illegal_actions += 1
        elif self.decision_mode == "joint_plan":
            assert self.joint_plan_cache is not None and self.sim.joint_plan_generator is not None
            committed, selected_plan = self.sim.joint_plan_generator.commit_joint_plan(
                self.joint_plan_cache, int(action)
            )
            self.joint_plan_cache = None
            if not committed:
                illegal = True
                self.sim.illegal_actions += 1
            elif selected_plan is not None:
                selected = selected_plan.sortie
        elif action > 0:
            idx = action - 1
            if idx < len(self.candidates):
                selected = self.candidates[idx]
        before_done_ids = {t.id for t in self.sim.tasks if t.status == "done"}
        before_done = len(before_done_ids)
        before_expired = sum(t.status == "expired" for t in self.sim.tasks)
        wait_penalty = self._wait_penalty(action)
        launched = int(selected is not None) if self.decision_mode == "joint_plan" and not illegal else self.sim.dispatch_sorties(selected)
        # A launch consumes a UAV but not a CAV movement decision. If another
        # idle UAV remains on the CAV, return immediately so PPO can launch
        # multiple UAVs from the same intersection through consecutive actions.
        allow_same_node_followup = launched > 0 and bool(self.sim.idle_drones_on_vehicle())
        if not allow_same_node_followup:
            # Advance until the next intersection decision point. If the CAV is
            # not moving, use the configured decision interval as a waiting/update window.
            target_time = min(self.sim.now + self.sim.decision_interval, self.sim.horizon)
            while self.sim.now < target_time - 1e-9:
                self.sim.step(min(self.sim.dt, target_time - self.sim.now))
                if self.sim.last_step_vehicle_arrivals:
                    break
        after_done_ids = {t.id for t in self.sim.tasks if t.status == "done"}
        after_done = len(after_done_ids)
        after_expired = sum(t.status == "expired" for t in self.sim.tasks)
        terminated = self.sim.now >= self.sim.horizon or self.sim._all_terminal()
        new_done_tasks = [t for t in self.sim.tasks if t.id in after_done_ids - before_done_ids]
        terminal_penalty = self._terminal_undone_penalty() if terminated else 0.0
        if use_interval_reward:
            reward_components = self._event_reward_components(
                before_reward_snapshot,
                self._reward_snapshot(),
                illegal=illegal,
                wait_penalty=wait_penalty,
                delta_t=max(0.0, float(self.sim.now) - decision_time),
                terminated=terminated,
            )
            reward = float(sum(reward_components.values()))
            self.sim.reward_total += reward
        else:
            reward = self._reward(
                after_done - before_done,
                after_expired - before_expired,
                illegal,
                wait_penalty,
                new_done_tasks,
                terminal_penalty,
            )
            reward_components = {"other": float(reward)}
        truncated = False
        self._prepare_decision_epoch(["FIXED_INTERVAL"])
        obs = self._observation()
        info = {
            "action_mask": self.action_masks(),
            "time": self.sim.now,
            "illegal": illegal,
            "recovery_queue_len": len(self.sim.recovery_queue),
            "time_mode": "fixed_interval",
            "decision_time": decision_time,
            "next_decision_time": float(self.sim.now),
            "delta_t": float(self.sim.now - decision_time),
            "transition_discount": float(self.gamma_reference),
            "trigger_events": ["FIXED_INTERVAL"],
            "processed_event_count": 0,
            "stale_event_count": 0,
            "duplicate_event_count": 0,
            "invalid_time_event_count": 0,
            "reward_components": reward_components,
            "valid_action_count": int(self.action_masks().sum()),
            "valid_non_wait_action_count": int(self.action_masks()[1:].sum()),
            "forced_wait": bool(action == 0 and int(mask[1:].sum()) == 0),
            "instantaneous_dispatch": bool(self.sim.now == decision_time and launched > 0),
            "wall_clock_step_time_ms": (time.perf_counter() - wall_start) * 1000.0,
            **self.sim.metrics("rl_env").to_dict(),
        }
        if gym is not None:
            return obs, float(reward), bool(terminated), bool(truncated), info
        return obs, float(reward), bool(terminated), info

    def _initialize_event_queue(self) -> None:
        assert self.sim is not None
        tolerance = float(self.event_cfg.get("event_time_tolerance", 1e-9))
        self.event_queue = EventQueue(time_tolerance=tolerance)
        now = float(self.sim.now)
        for task in sorted(self.sim.tasks, key=lambda item: (float(item.release_time), int(item.id))):
            if now + tolerance < float(task.release_time) < self.sim.horizon - tolerance:
                self.event_queue.schedule(
                    time=float(task.release_time),
                    event_type="TASK_RELEASE",
                    entity_id=f"task:{task.id}",
                    payload={"task_id": int(task.id)},
                    current_time=now,
                )
            timeout_time = float(np.nextafter(float(task.deadline), math.inf))
            if timeout_time < self.sim.horizon - tolerance and timeout_time > now + tolerance:
                self.event_queue.schedule(
                    time=timeout_time,
                    event_type="TASK_TIMEOUT",
                    entity_id=f"task:{task.id}",
                    payload={"task_id": int(task.id)},
                    current_time=now,
                )
        self.event_queue.schedule(
            time=float(self.sim.horizon),
            event_type="EPISODE_END",
            entity_id="episode",
            current_time=now,
        )
        self._schedule_dynamic_events()

    def _event_version(self, event: Event) -> Optional[int]:
        assert self.sim is not None
        if event.entity_id == self.sim.vehicle.id:
            return int(self.sim.vehicle.schedule_version)
        for drone in self.sim.drones:
            if event.entity_id == drone.id:
                return int(drone.schedule_version)
        return None

    def _schedule_entity_event(
        self,
        entity_id: str,
        version: int,
        event_time: float,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert self.sim is not None and self.event_queue is not None
        signature = (int(version), float(event_time), str(event_type))
        if self._dynamic_event_signatures.get(str(entity_id)) == signature:
            return
        self._dynamic_event_signatures[str(entity_id)] = signature
        self.event_queue.schedule(
            time=float(event_time),
            event_type=str(event_type),
            entity_id=str(entity_id),
            payload=payload,
            schedule_version=int(version),
            current_time=float(self.sim.now),
        )

    def _schedule_dynamic_events(self) -> None:
        if self.event_queue is None or self.sim is None:
            return
        tolerance = self.event_queue.time_tolerance
        vehicle = self.sim.vehicle
        if vehicle.is_waiting(self.sim.now):
            self._schedule_entity_event(
                vehicle.id,
                vehicle.schedule_version,
                float(vehicle.wait_until),
                "CAV_WAIT_COMPLETE",
                {"node": int(vehicle.position)},
            )
        elif vehicle.target_node is not None and vehicle.edge_remaining_time > tolerance:
            reaches_recovery = vehicle.edge_remaining_time >= vehicle.remaining_time - tolerance
            event_type = "CAV_RECOVERY_NODE_ARRIVAL" if reaches_recovery else "CAV_NODE_ARRIVAL"
            next_node = (
                int(vehicle.path_nodes[vehicle.path_index + 1])
                if vehicle.path_nodes and vehicle.path_index + 1 < len(vehicle.path_nodes)
                else int(vehicle.target_node)
            )
            self._schedule_entity_event(
                vehicle.id,
                vehicle.schedule_version,
                self.sim.now + vehicle.edge_remaining_time,
                event_type,
                {"node": next_node},
            )
        for drone in self.sim.drones:
            if drone.status != "flying" or drone.sortie is None:
                continue
            if drone.segment_index >= len(drone.sortie.segments):
                continue
            segment = drone.sortie.segments[drone.segment_index]
            remaining = float(drone.segment_remaining or segment.get("duration", 0.0))
            if remaining < -tolerance:
                continue
            if segment.get("type") == "service":
                event_type = "UAV_SERVICE_COMPLETE"
            elif drone.segment_index >= len(drone.sortie.segments) - 1:
                event_type = "UAV_RECOVERY_ARRIVAL"
            else:
                event_type = "UAV_TASK_ARRIVAL"
            self._schedule_entity_event(
                drone.id,
                drone.schedule_version,
                self.sim.now + max(0.0, remaining),
                event_type,
                {"segment_index": int(drone.segment_index)},
            )

    def _state_summary(self) -> Dict[str, Any]:
        assert self.sim is not None
        return {
            "time": float(self.sim.now),
            "cav_node": int(self.sim.vehicle.position),
            "cav_target": None if self.sim.vehicle.target_node is None else int(self.sim.vehicle.target_node),
            "uav_status": {drone.id: drone.status for drone in self.sim.drones},
            "task_status": {
                status: sum(task.status == status for task in self.sim.tasks)
                for status in ("unreleased", "unassigned", "assigned", "executing", "done", "expired")
            },
            "queue_length": len(self.sim.recovery_queue),
        }

    def _reward_snapshot(self) -> Dict[str, Any]:
        assert self.sim is not None
        return {
            "done_ids": {task.id for task in self.sim.tasks if task.status == "done"},
            "expired_ids": {task.id for task in self.sim.tasks if task.status == "expired"},
            "energy": float(sum(drone.energy_used for drone in self.sim.drones)),
            "aoi_integral": float(self.sim.aoi_integral),
            "uav_wait": float(sum(drone.wait_time for drone in self.sim.drones)),
            "cav_wait": float(self.sim.vehicle.wait_time),
            "queue_length": float(len(self.sim.recovery_queue)),
            "queue_travel_time": float(self.sim.recovery_queue_travel_time_s()),
        }

    def _episode_terminated(self) -> bool:
        assert self.sim is not None
        if self.sim.now >= self.sim.horizon - 1e-9:
            return True
        terminate_on_all = bool(self.cfg.get("simulation", {}).get("terminate_on_all_terminal", True))
        return bool(terminate_on_all and self.sim._all_terminal())

    def _event_reward_components(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
        *,
        illegal: bool,
        wait_penalty: float,
        delta_t: float,
        terminated: bool,
    ) -> Dict[str, float]:
        assert self.sim is not None
        r_cfg = self.cfg.get("reward", {})
        new_done_ids = set(after["done_ids"]) - set(before["done_ids"])
        new_expired_ids = set(after["expired_ids"]) - set(before["expired_ids"])
        done_tasks = [self.sim.task_pool.by_id[task_id] for task_id in sorted(new_done_ids)]
        completion = len(new_done_ids) * float(r_cfg.get("done_task", 10.0))
        completion += float(r_cfg.get("priority_bonus", 0.0)) * sum(float(task.priority) for task in done_tasks)
        timeout = len(new_expired_ids) * float(r_cfg.get("timeout_penalty", -8.0))
        normalize = bool(self.smdp_cfg.get("interval_normalize_reward", True))
        interval_divisor = self.reference_interval_s if normalize else 1.0
        aoi_increment = max(0.0, float(after["aoi_integral"]) - float(before["aoi_integral"]))
        reward_aoi = float(r_cfg.get("aoi_penalty", -0.00001)) * aoi_increment / interval_divisor
        energy_increment = max(0.0, float(after["energy"]) - float(before["energy"]))
        reward_energy = float(r_cfg.get("energy_penalty", -0.001)) * energy_increment * 0.001
        interval_scale = max(0.0, float(delta_t)) / self.reference_interval_s
        mean_queue = 0.5 * (float(before["queue_length"]) + float(after["queue_length"]))
        mean_queue_travel = 0.5 * (float(before["queue_travel_time"]) + float(after["queue_travel_time"]))
        reward_uav_wait = float(r_cfg.get("recovery_queue_penalty", -0.05)) * mean_queue * interval_scale
        reward_cav_wait = float(r_cfg.get("vehicle_recovery_time_penalty", -0.0005)) * mean_queue_travel * interval_scale
        other = (-15.0 if illegal else 0.0) + float(wait_penalty) * interval_scale
        if terminated:
            other += self._terminal_undone_penalty()
        components = {
            "completion": float(completion),
            "timeout": float(timeout),
            "aoi": float(reward_aoi),
            "energy": float(reward_energy),
            "uav_wait": float(reward_uav_wait),
            "cav_wait": float(reward_cav_wait),
            "other": float(other),
        }
        return components

    def _step_event_driven(self, action: int):
        assert self.sim is not None and self.event_queue is not None, "call reset first"
        wall_start = time.perf_counter()
        decision_time = float(self.sim.now)
        source_mask = self.action_masks()
        valid_action_count = int(source_mask.sum())
        valid_non_wait_count = int(source_mask[1:].sum())
        forced_wait = bool(int(action) == 0 and valid_non_wait_count == 0)
        if forced_wait:
            self.forced_wait_decisions += 1
        if valid_non_wait_count > 0:
            self.meaningful_decisions += 1

        illegal = bool(action < 0 or action >= self.max_actions or not bool(source_mask[action]))
        selected: Optional[Sortie] = None
        selected_plan: Optional[JointPlan] = None
        ground_action = None
        if illegal:
            self.sim.illegal_actions += 1
        elif self.decision_mode == "joint_plan":
            assert self.joint_plan_cache is not None and self.sim.joint_plan_generator is not None
            committed, selected_plan = self.sim.joint_plan_generator.commit_joint_plan(
                self.joint_plan_cache, int(action)
            )
            self.joint_plan_cache = None
            if not committed:
                illegal = True
                self.sim.illegal_actions += 1
            elif selected_plan is not None:
                selected = selected_plan.sortie
                ground_action = selected_plan.ground_candidate
                self._ground_action_committed_time = decision_time
        elif int(action) > 0:
            candidate_index = int(action) - 1
            if candidate_index < len(self.candidates):
                selected = self.candidates[candidate_index]
        selected_sortie_id = ""
        if selected is not None:
            selected_sortie_id = self.sim.recovery_scheduler.sortie_id(selected)
        before = self._reward_snapshot()
        wait_penalty = self._wait_penalty(action)
        launched = int(selected is not None) if self.decision_mode == "joint_plan" and not illegal else self.sim.dispatch_sorties(selected)
        trigger_events: List[str] = ["UAV_LAUNCH"] if launched > 0 else []
        processed_event_count = len(trigger_events)

        if self._instantaneous_time is None or abs(self._instantaneous_time - decision_time) > 1e-9:
            self._instantaneous_time = decision_time
            self._instantaneous_count = 0
        if launched > 0:
            self._instantaneous_count += 1
        configured_max = self.event_cfg.get("max_instantaneous_decisions", "auto")
        max_instantaneous = len(self.sim.drones) if str(configured_max).lower() == "auto" else int(configured_max)
        self._prepare_decision_epoch(["UAV_LAUNCH"] if launched > 0 else [])
        allow_instantaneous = bool(self.event_cfg.get("allow_instantaneous_multi_dispatch", True))
        instantaneous_dispatch = bool(
            allow_instantaneous
            and launched > 0
            and bool(self.sim.idle_drones_on_vehicle())
            and self._instantaneous_count < max(1, max_instantaneous)
        )

        if launched > 0:
            self._pending_ground_trigger_events.extend(
                ["UAV_LAUNCH", "RECOVERY_QUEUE_UPDATED", "DYNAMIC_RECOVERY_INSERTION_COMPLETE"]
            )
        if not instantaneous_dispatch and self.decision_mode != "joint_plan":
            next_ground_event = self.event_queue.peek_time(self.sim.now, self._event_version)
            ground_action = self.sim.ground_controller.decide(
                list(dict.fromkeys(self._pending_ground_trigger_events)),
                next_event_time=next_ground_event,
            )
            self._pending_ground_trigger_events = []
            self._prepare_decision_epoch(self._pending_ground_trigger_events)

        if instantaneous_dispatch:
            self.instantaneous_decisions += 1
            self._schedule_dynamic_events()
        else:
            while not self._episode_terminated():
                self._schedule_dynamic_events()
                next_time = self.event_queue.peek_time(self.sim.now, self._event_version)
                if next_time is None:
                    next_time = float(self.sim.horizon)
                next_time = min(float(next_time), float(self.sim.horizon))
                if next_time < self.sim.now - self.event_queue.time_tolerance:
                    self.event_queue.invalid_time_event_count += 1
                    continue
                batch = self.event_queue.pop_same_time(next_time, self.sim.now, self._event_version)
                for event in batch:
                    self._dynamic_event_signatures.pop(event.entity_id, None)
                state_before = self._state_summary()
                event_log_start = len(self.sim.event_log)
                if next_time > self.sim.now + self.event_queue.time_tolerance:
                    self.sim.step(next_time - self.sim.now)
                else:
                    # Complete zero-duration physical segments represented by
                    # an event at the current timestamp.  Without this flush,
                    # the same dynamic event is immediately scheduled again.
                    self.sim.step(0.0)
                derived_map = {
                    "task_done": "TASK_SERVICE_COMPLETE",
                    "recover": "UAV_RECOVERED",
                    "vehicle_arrive": "CAV_NODE_ARRIVAL",
                    "drone_wait_recovery": "UAV_RECOVERY_ARRIVAL",
                }
                batch_types = [event.event_type for event in batch]
                for row in self.sim.event_log[event_log_start:]:
                    if abs(float(row.get("time", self.sim.now)) - float(self.sim.now)) <= 1e-6:
                        mapped = derived_map.get(str(row.get("event")))
                        if mapped is not None:
                            batch_types.append(mapped)
                batch_types = list(dict.fromkeys(batch_types))
                self._pending_ground_trigger_events.extend(batch_types)
                if any(
                    event_type in {"UAV_RECOVERED", "CAV_RECOVERY_COMPLETE"}
                    for event_type in batch_types
                ):
                    self._pending_ground_trigger_events.append("RECOVERY_QUEUE_UPDATED")
                state_after = self._state_summary()
                if self._ground_action_committed_time is not None and self.sim.now > self._ground_action_committed_time + 1e-9:
                    self._ground_action_committed_time = None
                self._prepare_decision_epoch(batch_types)
                next_mask_for_decision = self.action_masks()
                decision_triggered = bool(next_mask_for_decision[1:].any()) and not self._episode_terminated()
                if not decision_triggered and not self._episode_terminated() and self.decision_mode != "joint_plan":
                    next_ground_event = self.event_queue.peek_time(self.sim.now, self._event_version)
                    automatic_ground_action = self.sim.ground_controller.decide(
                        list(dict.fromkeys(self._pending_ground_trigger_events)),
                        next_event_time=next_ground_event,
                    )
                    if automatic_ground_action is not None:
                        ground_action = automatic_ground_action
                    self._pending_ground_trigger_events = []
                    self._prepare_decision_epoch(self._pending_ground_trigger_events)
                    next_mask_for_decision = self.action_masks()
                    decision_triggered = bool(next_mask_for_decision[1:].any()) and not self._episode_terminated()
                for event_type in batch_types or ["STATE_ADVANCE"]:
                    self.event_timeline.append(
                        {
                            "event_time": float(self.sim.now),
                            "event_type": str(event_type),
                            "entity_id": "|".join(event.entity_id for event in batch),
                            "state_before": state_before,
                            "state_after": state_after,
                            "decision_triggered": bool(decision_triggered),
                            "decision_index": int(self.decision_index) if decision_triggered else -1,
                        }
                    )
                trigger_events.extend(batch_types)
                processed_event_count += max(len(batch), len(batch_types))
                self.system_events_processed += max(len(batch), len(batch_types))
                self._schedule_dynamic_events()
                if self._episode_terminated() or decision_triggered:
                    break

        terminated = self._episode_terminated()
        truncated = False
        delta_t = max(0.0, float(self.sim.now) - decision_time)
        discount = transition_discount(
            self.gamma_reference,
            delta_t,
            self.reference_interval_s,
            self.discount_mode,
        )
        if delta_t <= 1e-9:
            self.zero_duration_transitions += 1
        after = self._reward_snapshot()
        reward_components = self._event_reward_components(
            before,
            after,
            illegal=illegal,
            wait_penalty=wait_penalty,
            delta_t=delta_t,
            terminated=terminated,
        )
        reward = float(sum(reward_components.values()))
        self.sim.reward_total += reward
        self._delta_t_history.append(delta_t)
        self._discount_history.append(discount)
        self._prepare_decision_epoch(trigger_events)
        next_mask = self.action_masks()
        obs = self._observation()
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        trigger_events = list(dict.fromkeys(trigger_events))
        action_type = "illegal" if illegal else (
            selected_plan.plan_type if selected_plan is not None else ("wait" if int(action) == 0 else "sortie")
        )
        info = {
            "action_mask": next_mask,
            "time": float(self.sim.now),
            "time_mode": "event_driven",
            "illegal": illegal,
            "delta_t": delta_t,
            "transition_discount": discount,
            "decision_time": decision_time,
            "next_decision_time": float(self.sim.now),
            "trigger_events": trigger_events,
            "processed_event_count": int(processed_event_count),
            "stale_event_count": int(self.event_queue.stale_event_count),
            "duplicate_event_count": int(self.event_queue.duplicate_event_count),
            "invalid_time_event_count": int(self.event_queue.invalid_time_event_count),
            "reward_components": reward_components,
            "valid_action_count": int(next_mask.sum()),
            "valid_non_wait_action_count": int(next_mask[1:].sum()),
            "forced_wait": forced_wait,
            "instantaneous_dispatch": instantaneous_dispatch,
            "instantaneous_decision_count": int(self._instantaneous_count),
            "selected_sortie_id": selected_sortie_id,
            "action_type": action_type,
            "ground_action": None if ground_action is None else ground_action.action_type,
            "recovery_queue_len": len(self.sim.recovery_queue),
            "wall_clock_step_time_ms": wall_ms,
            **self.sim.metrics("rl_env").to_dict(),
        }
        log_row = {
            "episode": 0,
            "decision_index": int(self.decision_index),
            "current_time": decision_time,
            "next_decision_time": float(self.sim.now),
            "delta_t": delta_t,
            "transition_discount": discount,
            "action": int(action),
            "action_type": action_type,
            "ground_action": None if ground_action is None else ground_action.action_type,
            "selected_sortie_id": selected_sortie_id,
            "trigger_events": "|".join(trigger_events),
            "processed_event_count": int(processed_event_count),
            "valid_action_count": valid_action_count,
            "valid_non_wait_action_count": valid_non_wait_count,
            "forced_wait": forced_wait,
            "instantaneous_dispatch": instantaneous_dispatch,
            "reward_total": reward,
            "reward_completion": reward_components["completion"],
            "reward_timeout": reward_components["timeout"],
            "reward_aoi": reward_components["aoi"],
            "reward_energy": reward_components["energy"],
            "reward_uav_wait": reward_components["uav_wait"],
            "reward_cav_wait": reward_components["cav_wait"],
            "reward_other": reward_components["other"],
            "queue_length": len(self.sim.recovery_queue),
            "recovery_rule": self.sim.recovery_rule,
            "stale_event_count": int(self.event_queue.stale_event_count),
            "duplicate_event_count": int(self.event_queue.duplicate_event_count),
            "invalid_time_event_count": int(self.event_queue.invalid_time_event_count),
            "wall_clock_step_time_ms": wall_ms,
        }
        self.transition_log.append(log_row)
        self.decision_index += 1
        if gym is not None:
            return obs, reward, bool(terminated), bool(truncated), info
        return obs, reward, bool(terminated), info

    def episode_summary(self) -> Dict[str, Any]:
        assert self.sim is not None
        delta = np.asarray(self._delta_t_history, dtype=np.float64)
        discounts = np.asarray(self._discount_history, dtype=np.float64)
        metrics = self.sim.metrics("rl_env").to_dict()
        decisions = len(self.transition_log)
        return {
            **metrics,
            "episode_return": float(sum(row["reward_total"] for row in self.transition_log)),
            "simulated_time": float(self.sim.now),
            "decisions_per_episode": int(decisions),
            "system_events_per_episode": int(self.system_events_processed),
            "meaningful_decisions": int(self.meaningful_decisions),
            "forced_wait_decisions": int(self.forced_wait_decisions),
            "instantaneous_decisions": int(self.instantaneous_decisions),
            "zero_duration_transitions": int(self.zero_duration_transitions),
            "mean_delta_t": float(delta.mean()) if delta.size else 0.0,
            "median_delta_t": float(np.median(delta)) if delta.size else 0.0,
            "p95_delta_t": float(np.percentile(delta, 95)) if delta.size else 0.0,
            "min_delta_t": float(delta.min()) if delta.size else 0.0,
            "max_delta_t": float(delta.max()) if delta.size else 0.0,
            "mean_transition_discount": float(discounts.mean()) if discounts.size else 0.0,
            "forced_wait_decision_rate": self.forced_wait_decisions / decisions if decisions else 0.0,
            "zero_duration_transition_rate": self.zero_duration_transitions / decisions if decisions else 0.0,
            "simulated_seconds_per_decision": float(self.sim.now) / decisions if decisions else 0.0,
            "replay_transitions_per_simulated_hour": decisions / max(self.sim.now / 3600.0, 1e-12),
            "wall_clock_time_ms": float(sum(row["wall_clock_step_time_ms"] for row in self.transition_log)),
        }

    def action_masks(self) -> np.ndarray:
        if self.decision_mode == "joint_plan":
            if self.joint_plan_cache is None:
                raise RuntimeError("joint plan decision cache has not been prepared")
            return self.joint_plan_cache.action_mask.copy()
        mask = np.zeros(self.max_actions, dtype=bool)
        mask[0] = True
        for i, c in enumerate(self.candidates[: self.max_actions - 1], start=1):
            mask[i] = bool(c.feasible)
        return mask

    def sample_masked_action(self) -> int:
        mask = self.action_masks()
        valid = np.flatnonzero(mask)
        return int(np.random.choice(valid)) if len(valid) else 0

    def _wait_penalty(self, action: int) -> float:
        """State-dependent shaping for the no-op/wait action.

        Waiting is neutral when nothing can usefully be done, but it is penalized
        when there are open tasks or recovery demands that the CAV/UAV system
        should react to.
        """
        if int(action) != 0:
            return 0.0
        sim = self.sim
        assert sim is not None
        r_cfg = self.cfg.get("reward", {})
        active_tasks = sim.task_pool.active_tasks()
        available_tasks = sim.task_pool.available_tasks(sim.now)
        recovery_demand = bool(sim.recovery_queue) or any(d.status == "waiting_recovery" for d in sim.drones)
        all_idle = sim.vehicle.is_idle() and not sim.recovery_queue and all(d.status == "idle" for d in sim.drones)

        if not active_tasks and not recovery_demand and all_idle:
            return float(r_cfg.get("reasonable_wait_penalty", 0.0))
        if recovery_demand:
            return float(r_cfg.get("recovery_wait_penalty", -2.0))
        if not all_idle:
            return float(r_cfg.get("busy_wait_penalty", 0.0))
        if not available_tasks:
            return float(r_cfg.get("reasonable_wait_penalty", 0.0))

        high_priority_threshold = float(self.cfg.get("simulation", {}).get("high_priority_threshold", 8))
        urgent_threshold = float(r_cfg.get("urgent_wait_threshold_s", sim.decision_interval * 2.0))
        has_high_priority = any(t.priority >= high_priority_threshold for t in available_tasks)
        has_urgent = any(t.deadline - sim.now <= urgent_threshold for t in available_tasks)
        has_incident = any(t.type == "incident" for t in available_tasks)
        has_dynamic = any(t.type in {"incident", "congestion"} for t in available_tasks)

        if has_urgent or has_high_priority:
            return float(r_cfg.get("urgent_wait_penalty", -1.0))
        if has_incident:
            return float(r_cfg.get("incident_wait_penalty", -1.0))
        if has_dynamic:
            return float(r_cfg.get("dynamic_wait_penalty", -1.0))
        return float(r_cfg.get("routine_wait_penalty", -0.1))

    def _terminal_undone_penalty(self) -> float:
        sim = self.sim
        assert sim is not None
        undone = sum(t.status != "done" for t in sim.tasks)
        return undone * float(self.cfg.get("reward", {}).get("undone_task_penalty", -20.0))

    def _reward(
        self,
        new_done: int,
        new_expired: int,
        illegal: bool,
        wait_penalty: float = 0.0,
        new_done_tasks: Optional[List[Any]] = None,
        terminal_penalty: float = 0.0,
    ) -> float:
        r_cfg = self.cfg.get("reward", {})
        reward = new_done * float(r_cfg.get("done_task", 10.0)) + new_expired * float(r_cfg.get("timeout_penalty", -8.0))
        if new_done_tasks:
            reward += float(r_cfg.get("priority_bonus", 0.0)) * sum(float(t.priority) for t in new_done_tasks)
        if illegal:
            reward -= 15.0
        reward += float(wait_penalty)
        reward += float(terminal_penalty)
        reward += float(r_cfg.get("recovery_queue_penalty", -0.05)) * len(self.sim.recovery_queue)
        reward += float(r_cfg.get("vehicle_recovery_time_penalty", -0.0005)) * self.sim.recovery_queue_travel_time_s()
        # Penalize continuous costs at decision granularity.
        reward += float(r_cfg.get("energy_penalty", -0.001)) * sum(d.energy_used for d in self.sim.drones) * 0.001
        reward += float(r_cfg.get("aoi_penalty", -0.00001)) * self.sim.aoi_integral
        return float(reward)

    def _observation(self) -> np.ndarray:
        sim = self.sim
        assert sim is not None
        horizon = max(1.0, sim.horizon)
        n_nodes = max(1, len(sim.network.nodes) - 1)
        ablation_cfg = self.cfg.get("ablation", {})
        use_candidate_features = bool(ablation_cfg.get("candidate_sortie_features", True))
        use_queue_aware_recovery = bool(ablation_cfg.get("queue_aware_recovery", True))
        obs: List[float] = [
            sim.now / horizon,
            sim.vehicle.position / n_nodes,
            (sim.vehicle.target_node if sim.vehicle.target_node is not None else sim.vehicle.position) / n_nodes,
            sim.vehicle.remaining_time / horizon,
            (len(sim.recovery_queue) / max(1.0, len(sim.drones))) if use_queue_aware_recovery else 0.0,
            1.0 if sim.vehicle.at_intersection else 0.0,
        ]
        for d in sim.drones:
            obs.extend([
                1.0 if d.status == "idle" else 0.0,
                d.position / n_nodes,
                d.battery / max(1.0, d.battery_capacity),
            ])
        tasks = sorted(sim.task_pool.available_tasks(sim.now), key=lambda t: (t.deadline, -t.priority))[: self.max_tasks_obs]
        for t in tasks:
            obs.extend([
                t.release_time / horizon,
                max(0.0, t.deadline - sim.now) / horizon,
                t.priority / 12.0,
                t.service_time / 300.0,
                t.start_node / n_nodes,
                t.end_node / n_nodes,
            ])
        obs.extend([0.0] * ((self.max_tasks_obs - len(tasks)) * 6))
        if self.decision_mode == "joint_plan":
            if self.joint_plan_cache is None:
                raise RuntimeError("joint plan decision cache has not been prepared")
            obs.extend(self.joint_plan_cache.feature_matrix.reshape(-1).tolist())
        elif use_candidate_features:
            for c in self.candidates[: self.max_candidates_obs]:
                obs.extend([
                    c.score / 100.0,
                    c.drone_time_s / horizon,
                    c.vehicle_time_s / horizon,
                    c.energy / max(1.0, sim.drones[0].battery_capacity),
                    len(c.task_ids) / max(1, int(sim.cfg["sortie"].get("max_tasks_per_sortie", 3))),
                    c.recovery_node / n_nodes,
                ])
            obs.extend([0.0] * ((self.max_candidates_obs - min(len(self.candidates), self.max_candidates_obs)) * 6))
        else:
            obs.extend([0.0] * (self.max_candidates_obs * 6))
        return np.array(obs, dtype=np.float32)
