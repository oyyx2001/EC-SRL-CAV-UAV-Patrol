from __future__ import annotations

from collections import Counter
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
from dynamic_patrol_marl.rl.reward_ledger import RewardComponents, RewardLedger
from dynamic_patrol_marl.rl.smdp import resolve_discount_settings, transition_discount


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
        (
            self.discount_mode,
            self.gamma_reference,
            self.reference_interval_s,
            self.n_step,
        ) = resolve_discount_settings(cfg)
        self.event_queue: Optional[EventQueue] = None
        self.transition_log: List[Dict[str, Any]] = []
        self.event_timeline: List[Dict[str, Any]] = []
        self.reward_ledger = RewardLedger()
        self._last_reward_ledger_components: Dict[str, float] = {}
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
        self._cutoff_snapshot: Optional[Dict[str, Any]] = None
        self._operation_reward = 0.0
        self._drain_reward = 0.0
        self._censored_task_ids: set[int] = set()
        # Event-driven diagnostics distinguish physical state changes from
        # actual calls to the policy.  ``transition_log`` intentionally keeps
        # the physical sub-transitions for reward auditing, while the public
        # step API coalesces forced actions into one SMDP transition.
        self.policy_decision_count = 0
        self.single_valid_action_count = 0
        self.multi_valid_action_count = 0
        self.automatic_action_count = 0
        self.non_safe_action_count = 0
        self._valid_action_counts: List[int] = []
        self._smdp_duration_history: List[float] = []
        self.policy_trigger_event_counts: Counter[str] = Counter()
        self.initial_automatic_action_count = 0
        self.initial_automatic_reward = 0.0

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
        self._last_reward_ledger_components = {}
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
        self._cutoff_snapshot = None
        self._operation_reward = 0.0
        self._drain_reward = 0.0
        self._censored_task_ids = set()
        self.policy_decision_count = 0
        self.single_valid_action_count = 0
        self.multi_valid_action_count = 0
        self.automatic_action_count = 0
        self.non_safe_action_count = 0
        self._valid_action_counts = []
        self._smdp_duration_history = []
        self.policy_trigger_event_counts = Counter()
        self.initial_automatic_action_count = 0
        self.initial_automatic_reward = 0.0
        if self.time_mode == "event_driven":
            self._initialize_event_queue()
        self.joint_plan_cache = None
        self._ground_action_committed_time = None
        self._prepare_decision_epoch(["INITIAL_DECISION"])
        if self.time_mode == "event_driven":
            self._settle_initial_forced_actions()
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

    @staticmethod
    def _as_step_parts(step_output: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Normalize Gymnasium and legacy step signatures for coalescing."""

        if len(step_output) == 5:
            obs, reward, terminated, truncated, info = step_output
            return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), dict(info)
        obs, reward, done, info = step_output
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), False, dict(info)

    def _format_step_parts(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ):
        if gym is not None:
            return obs, float(reward), bool(terminated), bool(truncated), info
        return obs, float(reward), bool(terminated or truncated), info

    @staticmethod
    def _ordered_unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(str(value) for value in values if str(value)))

    def _coalesce_event_steps(
        self,
        steps: List[Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Return one time-consistent SMDP transition from atomic event steps.

        Each atomic event step already has its own effective discount.  The
        accumulated reward therefore uses the product of preceding discounts,
        and the returned bootstrap discount is the full product.  This is the
        same composition used by ``NStepSMDPAccumulator`` and works for both
        fixed-decision and time-aware discount modes.
        """

        if not steps:
            raise ValueError("cannot coalesce an empty event-step sequence")
        first_obs, _first_reward, _first_terminated, _first_truncated, first_info = steps[0]
        final_obs, _final_reward, terminated, truncated, final_info = steps[-1]
        accumulated_reward = 0.0
        raw_physical_reward = 0.0
        discount_product = 1.0
        total_delta_t = 0.0
        processed_events = 0
        automatic_count = 0
        trigger_events: List[str] = []
        ledger_components: Dict[str, float] = {}
        reward_components: Dict[str, float] = {}
        reward_audit_components: Dict[str, float] = {}
        completed_task_ids: set[int] = set()
        timed_out_task_ids: set[int] = set()

        for _obs, reward, _terminated, _truncated, info in steps:
            accumulated_reward += discount_product * float(reward)
            raw_physical_reward += float(reward)
            total_delta_t += float(info.get("delta_t", 0.0))
            processed_events += int(info.get("processed_event_count", 0))
            automatic_count += int(bool(info.get("automatic_action", False)))
            trigger_events.extend(str(item) for item in info.get("trigger_events", []))
            for name, value in dict(info.get("reward_ledger_components", {}) or {}).items():
                ledger_components[name] = ledger_components.get(name, 0.0) + discount_product * float(value)
            for name, value in dict(info.get("reward_components", {}) or {}).items():
                reward_components[name] = reward_components.get(name, 0.0) + discount_product * float(value)
            for name, value in dict(info.get("reward_audit_components", {}) or {}).items():
                reward_audit_components[name] = reward_audit_components.get(name, 0.0) + discount_product * float(value)
            completed_task_ids.update(int(item) for item in info.get("completed_task_ids", []) or [])
            timed_out_task_ids.update(int(item) for item in info.get("timed_out_task_ids", []) or [])
            discount_product *= float(info["transition_discount"])

        if not math.isclose(
            accumulated_reward,
            float(sum(reward_components.values())),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise AssertionError(
                "coalesced reward ledger does not conserve discounted reward: "
                f"reward={accumulated_reward} components={sum(reward_components.values())}"
            )

        combined_info = dict(first_info)
        combined_info.update({
            "action_mask": np.asarray(final_info["action_mask"], dtype=bool),
            "time": float(final_info.get("time", 0.0)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "delta_t": float(total_delta_t),
            "transition_discount": float(discount_product),
            "next_decision_time": float(final_info.get("next_decision_time", final_info.get("time", 0.0))),
            "trigger_events": self._ordered_unique(trigger_events),
            "processed_event_count": int(processed_events),
            "stale_event_count": int(final_info.get("stale_event_count", 0)),
            "duplicate_event_count": int(final_info.get("duplicate_event_count", 0)),
            "invalid_time_event_count": int(final_info.get("invalid_time_event_count", 0)),
            "reward_components": reward_components,
            "reward_ledger_components": ledger_components,
            "reward_audit_components": reward_audit_components,
            "valid_action_count": int(np.asarray(final_info["action_mask"], dtype=bool).sum()),
            "valid_non_wait_action_count": int(np.asarray(final_info["action_mask"], dtype=bool)[1:].sum()),
            "completed_task_ids": sorted(completed_task_ids),
            "timed_out_task_ids": sorted(timed_out_task_ids),
            "automatic_action_count": int(automatic_count),
            "smdp_subtransition_count": int(len(steps)),
            "raw_physical_reward": float(raw_physical_reward),
            "smdp_aggregated_reward": float(accumulated_reward),
            "reward_conservation_error": float(accumulated_reward - sum(reward_components.values())),
            "wall_clock_step_time_ms": float(sum(float(item[4].get("wall_clock_step_time_ms", 0.0)) for item in steps)),
        })
        return final_obs, float(accumulated_reward), bool(terminated), bool(truncated), combined_info

    def _settle_initial_forced_actions(self) -> None:
        """Hide deterministic initial states from the policy and replay.

        There is no preceding policy action at reset to which an initial forced
        interval could be attached.  We still execute it physically and keep it
        in the raw event/reward audit, but expose the first state with a real
        choice to the policy.
        """

        max_steps = int(self.event_cfg.get("max_automatic_advance_steps", 100000))
        for _ in range(max_steps):
            if self._episode_done() or int(self.action_masks().sum()) != 1:
                return
            parts = self._as_step_parts(self._step_event_driven_once(0, automatic_action=True))
            self.initial_automatic_action_count += 1
            self.initial_automatic_reward += float(parts[1])
        raise RuntimeError("initial forced-action settlement exceeded max_automatic_advance_steps")

    def _step_event_driven(self, action: int):
        """Execute one policy choice and coalesce any following forced actions."""

        assert self.sim is not None and self.event_queue is not None, "call reset first"
        source_mask = self.action_masks()
        is_automatic_source = int(source_mask.sum()) == 1
        if not is_automatic_source:
            self.policy_decision_count += 1

        steps: List[Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]] = []
        current_action = int(np.flatnonzero(source_mask)[0]) if is_automatic_source else int(action)
        automatic = bool(is_automatic_source)
        max_steps = int(self.event_cfg.get("max_automatic_advance_steps", 100000))
        for _ in range(max_steps):
            parts = self._as_step_parts(
                self._step_event_driven_once(current_action, automatic_action=automatic)
            )
            steps.append(parts)
            _obs, _reward, terminated, truncated, info = parts
            if terminated or truncated:
                break
            next_mask = np.asarray(info["action_mask"], dtype=bool)
            if int(next_mask.sum()) >= 2:
                break
            if int(next_mask.sum()) != 1:
                raise RuntimeError("action mask must retain the safe action during automatic advance")
            current_action = int(np.flatnonzero(next_mask)[0])
            automatic = True
        else:
            raise RuntimeError("forced-action coalescing exceeded max_automatic_advance_steps")

        obs, reward, terminated, truncated, info = self._coalesce_event_steps(steps)
        self._smdp_duration_history.append(float(info["delta_t"]))
        return self._format_step_parts(obs, reward, terminated, truncated, info)

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
        terminated, truncated = self._episode_status()
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
        self._accumulate_phase_reward(decision_time, reward)
        self._prepare_decision_epoch(["FIXED_INTERVAL"])
        obs = self._observation()
        info = {
            "action_mask": self.action_masks(),
            "time": self.sim.now,
            "illegal": illegal,
            "recovery_queue_len": len(self.sim.recovery_queue),
            "time_mode": "fixed_interval",
            "terminated": bool(terminated),
            "truncated": bool(truncated),
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
            "reward_ledger_components": dict(self._last_reward_ledger_components),
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
        release_horizon = float(self.sim.task_release_horizon)
        hard_horizon = float(self.sim.hard_horizon)
        for task in sorted(self.sim.tasks, key=lambda item: (float(item.release_time), int(item.id))):
            if now + tolerance < float(task.release_time) < release_horizon - tolerance:
                self.event_queue.schedule(
                    time=float(task.release_time),
                    event_type="TASK_RELEASE",
                    entity_id=f"task:{task.id}",
                    payload={"task_id": int(task.id)},
                    current_time=now,
                )
            timeout_time = float(np.nextafter(float(task.deadline), math.inf))
            if timeout_time < hard_horizon - tolerance and timeout_time > now + tolerance:
                self.event_queue.schedule(
                    time=timeout_time,
                    event_type="TASK_TIMEOUT",
                    entity_id=f"task:{task.id}",
                    payload={"task_id": int(task.id)},
                    current_time=now,
                )
        if self.sim.drain_phase_enabled and release_horizon > now + tolerance:
            self.event_queue.schedule(
                time=release_horizon,
                event_type="TASK_RELEASE_CUTOFF",
                entity_id="episode:release_cutoff",
                current_time=now,
            )
        self.event_queue.schedule(
            time=hard_horizon,
            event_type="EPISODE_HARD_LIMIT",
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
            if segment.get("type") == "vertical_climb":
                event_type = "UAV_CLIMB_COMPLETE"
            elif segment.get("type") == "vertical_descent":
                event_type = "UAV_DESCENT_COMPLETE"
            elif segment.get("type") == "service":
                event_type = "UAV_SERVICE_COMPLETE"
            elif segment.get("type") == "edge_coverage":
                event_type = "UAV_EDGE_COVERAGE_COMPLETE"
            elif drone.segment_index >= len(drone.sortie.segments) - 2 and segment.get("type") in {"fly", "horizontal_flight"}:
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

    def _scheduling_fingerprint(self) -> Tuple[Any, ...]:
        """Return only state fields that can create a new dispatch choice.

        Flight-segment progress and the current clock are deliberately absent:
        a UAV climbing or changing between ordinary flight segments is a
        physical event, not automatically a policy opportunity.  In contrast,
        released/unassigned work, UAV availability on the CAV, the CAV's
        decision node/route and recovery assignments are all represented.
        """

        assert self.sim is not None
        vehicle = self.sim.vehicle
        available = tuple(
            sorted(
                (
                    int(task.id),
                    str(task.status),
                    str(getattr(task, "executor", "")),
                )
                for task in self.sim.task_pool.available_tasks(self.sim.now)
            )
        )
        idle_on_vehicle = tuple(
            sorted(
                (
                    str(drone.id),
                    int(drone.position),
                    round(float(drone.battery), 6),
                )
                for drone in self.sim.idle_drones_on_vehicle()
            )
        )
        route = (
            int(vehicle.position),
            None if vehicle.target_node is None else int(vehicle.target_node),
            bool(vehicle.at_intersection),
            tuple(int(node) for node in getattr(vehicle, "path_nodes", []) or []),
            int(getattr(vehicle, "schedule_version", 0)),
        )
        recovery = tuple(
            sorted(
                (
                    str(entry.get("sortie_id", entry.get("drone_id", ""))),
                    str(entry.get("drone_id", entry.get("uav_id", ""))),
                    int(entry.get("recovery_node", -1)),
                )
                for entry in self.sim.recovery_queue
            )
        )
        return available, idle_on_vehicle, route, recovery

    def _candidate_signature(self) -> Tuple[Any, ...]:
        """Identify substantive action-set changes without comparing features."""

        if self.decision_mode == "joint_plan":
            cache = self.joint_plan_cache
            if cache is None:
                return tuple()
            rows = []
            for index, plan in enumerate(cache.plan_slots):
                if plan is None:
                    rows.append((index, False, "padding"))
                    continue
                ground = getattr(plan, "ground_candidate", None)
                sortie = getattr(plan, "sortie", None)
                rows.append(
                    (
                        int(index),
                        bool(cache.action_mask[index]),
                        str(getattr(plan, "plan_type", "")),
                        str(getattr(ground, "action_type", "")),
                        None if sortie is None else str(getattr(sortie, "drone_id", "")),
                        tuple(int(task_id) for task_id in (getattr(sortie, "task_ids", []) or [])),
                        None if sortie is None else int(getattr(sortie, "recovery_node", -1)),
                    )
                )
            return tuple(rows)
        return tuple(
            (
                bool(getattr(candidate, "feasible", False)),
                str(getattr(candidate, "drone_id", "")),
                tuple(int(task_id) for task_id in (getattr(candidate, "task_ids", []) or [])),
                int(getattr(candidate, "recovery_node", -1)),
            )
            for candidate in self.candidates
        )

    @staticmethod
    def _decision_trigger_sources(
        batch_types: List[str],
        before: Tuple[Any, ...],
        after: Tuple[Any, ...],
        candidate_changed: bool,
    ) -> List[str]:
        """Attribute a decision to the state change that created it.

        Several physical events can share a timestamp.  Counting every event
        in such a batch would incorrectly make, for example,
        ``UAV_CLIMB_COMPLETE`` look like a policy trigger when a simultaneous
        CAV wait completion actually created the scheduling opportunity.
        """

        sources: List[str] = []
        event_set = {str(event_type) for event_type in batch_types}
        if before[0] != after[0]:
            sources.extend(sorted(event_set & {"TASK_RELEASE", "TASK_TIMEOUT"}) or ["TASK_POOL_CHANGED"])
        if before[1] != after[1]:
            sources.extend(
                sorted(event_set & {"UAV_RECOVERED", "CAV_RECOVERY_COMPLETE", "UAV_RECOVERY_ARRIVAL"})
                or ["UAV_AVAILABILITY_CHANGED"]
            )
        if before[2] != after[2]:
            sources.extend(
                sorted(event_type for event_type in event_set if event_type.startswith("CAV_"))
                or ["CAV_ROUTE_OR_NODE_CHANGED"]
            )
        if before[3] != after[3]:
            sources.extend(
                sorted(event_set & {"UAV_RECOVERED", "CAV_RECOVERY_COMPLETE", "UAV_RECOVERY_ARRIVAL"})
                or ["RECOVERY_PLAN_CHANGED"]
            )
        if candidate_changed:
            sources.append("CANDIDATE_SET_CHANGED")
        return list(dict.fromkeys(sources or ["SCHEDULING_STATE_CHANGED"]))

    def _reward_snapshot(self) -> Dict[str, Any]:
        assert self.sim is not None
        return {
            "done_ids": {task.id for task in self.sim.tasks if task.status == "done"},
            "expired_ids": {task.id for task in self.sim.tasks if task.status == "expired"},
            "energy": float(sum(drone.energy_used for drone in self.sim.drones)),
            "aoi_integral": float(self.sim.aoi_integral),
            "uav_wait": float(sum(drone.wait_time for drone in self.sim.drones)),
            "cav_wait": float(self.sim.vehicle.wait_time),
            "hover_time": float(sum(drone.wait_time for drone in self.sim.drones)),
            "return_distance": float(getattr(self.sim, "uav_return_distance_m", 0.0)),
            "cav_distance": float(getattr(self.sim.vehicle, "distance_traveled_m", 0.0)),
            "queue_length": float(len(self.sim.recovery_queue)),
            "queue_occupancy_integral": float(
                getattr(self.sim, "recovery_queue_occupancy_integral_s", 0.0)
            ),
            "queue_travel_time": float(self.sim.recovery_queue_travel_time_s()),
            # Emergency overrides resolve a hard feasibility conflict and are
            # intentionally excluded from the soft route-stability penalty.
            "route_change_count": float(
                sum(
                    bool(row.get("route_changed", row.get("action_type") in {"move_to", "reroute"}))
                    and not bool(row.get("emergency_override"))
                    for row in getattr(self.sim.ground_controller, "route_change_log", [])
                )
            ),
            "route_reverse_count": float(
                sum(
                    bool(row.get("reverse_route_flag")) and not bool(row.get("emergency_override"))
                    for row in getattr(self.sim.ground_controller, "route_change_log", [])
                )
            ),
            "coordination_failure_count": float(
                getattr(getattr(self.sim, "joint_plan_generator", None), "commit_failures", 0)
                + getattr(self.sim, "infeasible_recoveries", 0)
            ),
        }

    def _capture_cutoff_state(self) -> None:
        """Freeze the operational backlog exactly once at release cutoff."""

        assert self.sim is not None
        if self._cutoff_snapshot is not None:
            return
        if self.sim.now + 1e-9 < float(self.sim.task_release_horizon):
            return
        cohort = self._release_cohort_tasks()
        pending = sum(task.status in {"unassigned", "assigned", "executing"} for task in cohort)
        self._cutoff_snapshot = {
            "released_task_count": int(len(cohort)),
            "pending_tasks_at_cutoff": int(pending),
            "backlog_rate_at_cutoff": float(pending / len(cohort)) if cohort else 0.0,
        }

    def _release_cohort_tasks(self) -> List[Any]:
        assert self.sim is not None
        return [
            task
            for task in self.sim.tasks
            if float(task.release_time) < float(self.sim.task_release_horizon)
        ]

    def _mark_censored_at_hard_limit(self) -> None:
        """Mark unresolved released work without converting it to timeout."""

        assert self.sim is not None
        for task in self._release_cohort_tasks():
            if task.status in {"done", "expired", "censored"}:
                continue
            task.status = "censored"
            task.metadata["censored_at_s"] = float(self.sim.now)
            task.metadata["censor_reason"] = "episode_hard_limit"
            self._censored_task_ids.add(int(task.id))

    def _episode_status(self) -> Tuple[bool, bool]:
        assert self.sim is not None
        self._capture_cutoff_state()
        # During the normal operating window the future task stream must keep
        # the episode alive even if the currently released backlog is empty.
        if self.sim.now + 1e-9 < float(self.sim.task_release_horizon):
            return False, False
        # Drain-aware Stage 5 episodes always use the explicit quiescence
        # rule.  Historical configs may set ``terminate_on_all_terminal`` to
        # false, but that must not suppress the required natural termination
        # once the released cohort and recovery workflow are empty.
        terminate_on_all = bool(
            self.sim.drain_phase_enabled
            or self.cfg.get("simulation", {}).get("terminate_on_all_terminal", True)
        )
        if terminate_on_all and self.sim._all_terminal():
            return True, False
        if self.sim.now >= float(self.sim.hard_horizon) - 1e-9:
            self._mark_censored_at_hard_limit()
            return False, True
        return False, False

    def _episode_terminated(self) -> bool:
        return self._episode_status()[0]

    def _episode_done(self) -> bool:
        terminated, truncated = self._episode_status()
        return bool(terminated or truncated)

    def _accumulate_phase_reward(self, decision_time: float, reward: float) -> None:
        assert self.sim is not None
        if float(decision_time) < float(self.sim.task_release_horizon) - 1e-9:
            self._operation_reward += float(reward)
        else:
            self._drain_reward += float(reward)

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
        if str(r_cfg.get("mode", "legacy_interval")).lower() == "stage5_coordinated":
            return self._stage5_coordinated_reward_components(
                before,
                after,
                new_done_ids=new_done_ids,
                new_expired_ids=new_expired_ids,
                done_tasks=done_tasks,
                illegal=illegal,
                wait_penalty=wait_penalty,
                delta_t=delta_t,
                terminated=terminated,
            )
        completion_base = len(new_done_ids) * float(r_cfg.get("done_task", 10.0))
        priority_reward = float(r_cfg.get("priority_bonus", 0.0)) * sum(float(task.priority) for task in done_tasks)
        completion = completion_base + priority_reward
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
        illegal_reward = -15.0 if illegal else 0.0
        wait_action_reward = float(wait_penalty) * interval_scale
        terminal_undone_reward = 0.0
        other = illegal_reward + wait_action_reward
        if terminated:
            terminal_undone_reward = self._terminal_undone_penalty()
            other += terminal_undone_reward
        ledger_components = self.reward_ledger.from_legacy_interval(
            completion_reward=completion_base,
            priority_reward=priority_reward,
            timeout_reward=timeout,
            aoi_reward=reward_aoi,
            energy_reward=reward_energy,
            legacy_queue_reward=reward_uav_wait,
            legacy_cav_queue_reward=reward_cav_wait,
            wait_action_reward=wait_action_reward,
            illegal_action_reward=illegal_reward,
            terminal_undone_reward=terminal_undone_reward,
        )
        components = {
            "completion": float(completion),
            "timeout": float(timeout),
            "aoi": float(reward_aoi),
            "energy": float(reward_energy),
            "uav_wait": float(reward_uav_wait),
            "cav_wait": float(reward_cav_wait),
            "other": float(other),
        }
        self.reward_ledger.assert_conserved(ledger_components, sum(components.values()))
        self._last_reward_ledger_components = ledger_components.to_dict()
        return components

    def _stage5_coordinated_reward_components(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
        *,
        new_done_ids: set,
        new_expired_ids: set,
        done_tasks: List[Any],
        illegal: bool,
        wait_penalty: float,
        delta_t: float,
        terminated: bool,
    ) -> Dict[str, float]:
        assert self.sim is not None
        r_cfg = self.cfg.get("reward", {})
        weights = dict(r_cfg.get("stage5_weights", {}))
        references = dict(r_cfg.get("stage5_references", {}))
        reference_time = float(references.get("time_s", self.reference_interval_s))
        return_reference = max(float(references.get("return_distance_m", 1000.0)), 1e-9)
        cav_reference = max(float(references.get("cav_distance_m", 1000.0)), 1e-9)
        normalize_interval = bool(self.smdp_cfg.get("interval_normalize_reward", True))
        interval_divisor = self.reference_interval_s if normalize_interval else 1.0

        completion_reward = float(weights.get("complete", r_cfg.get("done_task", 30.0))) * len(new_done_ids)
        priority_reward = float(weights.get("priority", r_cfg.get("priority_bonus", 2.0))) * sum(
            float(task.priority) for task in done_tasks
        )
        timeout_base = float(weights.get("timeout", abs(float(r_cfg.get("timeout_penalty", -30.0)))))
        timeout_priority = float(weights.get("timeout_priority", 0.0))
        timeout_cost = sum(
            timeout_base + timeout_priority * float(self.sim.task_pool.by_id[task_id].priority)
            for task_id in new_expired_ids
        )

        aoi_increment = max(0.0, float(after["aoi_integral"]) - float(before["aoi_integral"]))
        aoi_cost = float(weights.get("aoi", abs(float(r_cfg.get("aoi_penalty", -0.00001))))) * aoi_increment / interval_divisor

        energy_increment = max(0.0, float(after["energy"]) - float(before["energy"]))
        hover_time_increment = max(0.0, float(after.get("hover_time", 0.0)) - float(before.get("hover_time", 0.0)))
        hover_energy_increment = hover_time_increment * float(self.cfg.get("agents", {}).get("drone_hover_power", 0.65))
        flight_energy_increment = max(0.0, energy_increment - hover_energy_increment)
        flight_energy_cost = float(weights.get("flight_energy", 1.0e-6)) * flight_energy_increment
        hover_energy_cost = float(weights.get("hover_energy", 1.0e-6)) * hover_energy_increment

        return_distance_increment = max(0.0, float(after.get("return_distance", 0.0)) - float(before.get("return_distance", 0.0)))
        cav_distance_increment = max(0.0, float(after.get("cav_distance", 0.0)) - float(before.get("cav_distance", 0.0)))
        return_distance_cost = float(weights.get("return_distance", 0.0)) * return_distance_increment / return_reference
        cav_distance_cost = float(weights.get("cav_distance", 0.0)) * cav_distance_increment / cav_reference

        recovery_wait_increment = max(0.0, float(after.get("uav_wait", 0.0)) - float(before.get("uav_wait", 0.0)))
        recovery_wait_cost = float(weights.get("recovery_wait", 0.0)) * recovery_wait_increment / max(reference_time, 1e-9)
        queue_occupancy_increment = max(
            0.0,
            float(after.get("queue_occupancy_integral", 0.0))
            - float(before.get("queue_occupancy_integral", 0.0)),
        )
        recovery_queue_cost = (
            float(weights.get("recovery_queue", 0.0))
            * queue_occupancy_increment
            / max(reference_time, 1e-9)
        )

        route_change_delta = max(0.0, float(after.get("route_change_count", 0.0)) - float(before.get("route_change_count", 0.0)))
        route_reverse_delta = max(0.0, float(after.get("route_reverse_count", 0.0)) - float(before.get("route_reverse_count", 0.0)))
        failure_delta = max(
            0.0,
            float(after.get("coordination_failure_count", 0.0)) - float(before.get("coordination_failure_count", 0.0)),
        )

        # Keep invalid-action cost independent from an actual coordination
        # failure.  Reward-freezing experiments vary both terms explicitly;
        # fall back to the historical shared value for existing configurations.
        illegal_weight = float(weights.get("illegal_action", weights.get("coordination_failure", 15.0)))
        illegal_reward = -illegal_weight if illegal else 0.0
        terminal_undone_reward = self._terminal_undone_penalty() if terminated else 0.0

        ledger_components = RewardComponents(
            completion_reward=completion_reward,
            priority_reward=priority_reward,
            timeout_cost=timeout_cost,
            aoi_cost=aoi_cost,
            flight_energy_cost=flight_energy_cost,
            hover_energy_cost=hover_energy_cost,
            return_distance_cost=return_distance_cost,
            cav_distance_cost=cav_distance_cost,
            recovery_wait_cost=recovery_wait_cost,
            recovery_queue_cost=recovery_queue_cost,
            route_change_cost=float(weights.get("route_change", 0.0)) * route_change_delta,
            route_reverse_cost=float(weights.get("route_reverse", 0.0)) * route_reverse_delta,
            coordination_failure_cost=float(weights.get("coordination_failure", 15.0)) * failure_delta,
            terminal_undone_cost=max(0.0, -terminal_undone_reward),
            wait_action_cost=0.0,
            illegal_action_cost=max(0.0, -illegal_reward),
        ).finalize()
        components = {
            "completion_reward": float(ledger_components.completion_reward),
            "priority_reward": float(ledger_components.priority_reward),
            "timeout_cost": float(-ledger_components.timeout_cost),
            "aoi_cost": float(-ledger_components.aoi_cost),
            "flight_energy_cost": float(-ledger_components.flight_energy_cost),
            "hover_energy_cost": float(-ledger_components.hover_energy_cost),
            "return_distance_cost": float(-ledger_components.return_distance_cost),
            "cav_distance_cost": float(-ledger_components.cav_distance_cost),
            "recovery_wait_cost": float(-ledger_components.recovery_wait_cost),
            "recovery_queue_cost": float(-ledger_components.recovery_queue_cost),
            "route_change_cost": float(-ledger_components.route_change_cost),
            "route_reverse_cost": float(-ledger_components.route_reverse_cost),
            "coordination_failure_cost": float(-ledger_components.coordination_failure_cost),
            "terminal_undone_cost": float(-ledger_components.terminal_undone_cost),
            "illegal_action_cost": float(-ledger_components.illegal_action_cost),
        }
        component_total = float(sum(components.values()))
        if abs(float(ledger_components.final_reward) - component_total) >= 1e-8:
            raise AssertionError(
                "stage5 reward components do not conserve transition reward: "
                f"reward={ledger_components.final_reward} "
                f"components={component_total} "
                f"diff={ledger_components.final_reward - component_total}"
            )
        self.reward_ledger.assert_conserved(ledger_components, component_total)
        self._last_reward_ledger_components = ledger_components.to_dict()
        self._last_reward_ledger_components.update(
            {
                "raw_aoi_increment": float(aoi_increment),
                "raw_energy_increment": float(energy_increment),
                "raw_hover_energy_increment": float(hover_energy_increment),
                "raw_flight_energy_increment": float(flight_energy_increment),
                "raw_return_distance_increment": float(return_distance_increment),
                "raw_cav_distance_increment": float(cav_distance_increment),
                "raw_recovery_wait_increment": float(recovery_wait_increment),
                "raw_recovery_queue_occupancy_increment": float(queue_occupancy_increment),
                "raw_route_change_delta": float(route_change_delta),
                "raw_route_reverse_delta": float(route_reverse_delta),
                "raw_coordination_failure_delta": float(failure_delta),
            }
        )
        return components

    def _reward_audit_components(self, reward: float) -> Dict[str, float]:
        """Return a stable, signed reward decomposition for offline audits.

        This is deliberately a view of the existing ledger rather than a
        second reward calculation.  Keeping it here makes the conservation
        check run at the exact transition boundary used by the trainer.
        """
        ledger = self._last_reward_ledger_components
        value = lambda name: float(ledger.get(name, 0.0))
        components = {
            "task_completion_reward": value("completion_reward"),
            "priority_reward": value("priority_reward"),
            # Stage 5 has no separate deadline or response-time bonus.  They
            # remain explicit zero columns to prevent silently reclassifying
            # a component in downstream analyses.
            "deadline_reward": 0.0,
            "response_time_reward": 0.0,
            "weighted_aoi_reward": -value("aoi_cost"),
            "recovery_reward": 0.0,
            "energy_penalty": -value("flight_energy_cost") - value("hover_energy_cost"),
            "uav_distance_penalty": -value("return_distance_cost"),
            "cav_distance_penalty": -value("cav_distance_cost"),
            "waiting_penalty": -value("recovery_wait_cost") - value("recovery_queue_cost"),
            "timeout_penalty": -value("timeout_cost"),
            "invalid_action_penalty": -value("illegal_action_cost"),
            "other_reward": -(
                value("route_change_cost")
                + value("route_reverse_cost")
                + value("coordination_failure_cost")
                + value("terminal_undone_cost")
                + value("wait_action_cost")
                + value("legacy_queue_cost")
                + value("legacy_cav_queue_cost")
                + value("other_cost")
            ),
        }
        # This stricter assertion is intentionally separate from the
        # historical 1e-6 ledger tolerance: the audit CSV must be exactly
        # reconcilable with the scalar reward emitted to Rainbow.
        conservation_error = float(reward) - float(sum(components.values()))
        if abs(conservation_error) >= 1e-8:
            raise AssertionError(
                "reward audit components do not conserve transition reward: "
                f"reward={reward} components={sum(components.values())} diff={conservation_error}"
            )
        components["reward_conservation_error"] = conservation_error
        return components

    def _step_event_driven_once(self, action: int, *, automatic_action: bool = False):
        assert self.sim is not None and self.event_queue is not None, "call reset first"
        wall_start = time.perf_counter()
        decision_time = float(self.sim.now)
        source_mask = self.action_masks()
        valid_action_count = int(source_mask.sum())
        valid_non_wait_count = int(source_mask[1:].sum())
        self._valid_action_counts.append(valid_action_count)
        if valid_action_count == 1:
            self.single_valid_action_count += 1
        else:
            self.multi_valid_action_count += 1
        if automatic_action:
            self.automatic_action_count += 1
        forced_wait = bool(int(action) == 0 and valid_non_wait_count == 0)
        if forced_wait:
            self.forced_wait_decisions += 1
        if valid_non_wait_count > 0:
            self.meaningful_decisions += 1

        # Capture the transition baseline before a joint plan is committed.
        # Joint-plan commit mutates the CAV route and appends route events; a
        # post-commit baseline would place the same event in before and after,
        # forcing its transition-local reward delta to zero.
        before = self._reward_snapshot()
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
        if not illegal and int(action) != 0:
            self.non_safe_action_count += 1
        selected_sortie_id = ""
        if selected is not None:
            selected_sortie_id = self.sim.recovery_scheduler.sortie_id(selected)
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
            while not self._episode_done():
                self._schedule_dynamic_events()
                next_time = self.event_queue.peek_time(self.sim.now, self._event_version)
                if next_time is None:
                    next_time = float(self.sim.hard_horizon)
                next_time = min(float(next_time), float(self.sim.hard_horizon))
                if next_time < self.sim.now - self.event_queue.time_tolerance:
                    self.event_queue.invalid_time_event_count += 1
                    continue
                batch = self.event_queue.pop_same_time(next_time, self.sim.now, self._event_version)
                for event in batch:
                    self._dynamic_event_signatures.pop(event.entity_id, None)
                state_before = self._state_summary()
                scheduling_before = self._scheduling_fingerprint()
                candidate_before = self._candidate_signature()
                event_log_start = len(self.sim.event_log)
                if next_time > self.sim.now + self.event_queue.time_tolerance:
                    self.sim.step(next_time - self.sim.now)
                else:
                    # Complete zero-duration physical segments represented by
                    # an event at the current timestamp.  Without this flush,
                    # the same dynamic event is immediately scheduled again.
                    self.sim.step(0.0)
                self._capture_cutoff_state()
                derived_map = {
                    "task_done": "TASK_SERVICE_COMPLETE",
                    "edge_coverage_done": "UAV_EDGE_COVERAGE_COMPLETE",
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
                state_after = self._state_summary()
                scheduling_after = self._scheduling_fingerprint()
                scheduling_changed = scheduling_before != scheduling_after
                if self._ground_action_committed_time is not None and self.sim.now > self._ground_action_committed_time + 1e-9:
                    self._ground_action_committed_time = None

                # A batch may contain many physical events at the same
                # timestamp.  Rebuild candidates only after the full batch and
                # only when its processed state has changed dispatch-relevant
                # fields.  This is what suppresses spurious decisions at
                # climb/ordinary flight segment boundaries.
                candidate_changed = False
                if scheduling_changed:
                    self._pending_ground_trigger_events.extend(batch_types)
                    if any(
                        event_type in {"UAV_RECOVERED", "CAV_RECOVERY_COMPLETE"}
                        for event_type in batch_types
                    ):
                        self._pending_ground_trigger_events.append("RECOVERY_QUEUE_UPDATED")
                    self._prepare_decision_epoch(batch_types)
                    next_mask_for_decision = self.action_masks()
                    candidate_changed = candidate_before != self._candidate_signature()
                else:
                    next_mask_for_decision = self.action_masks()
                decision_triggered = bool(
                    scheduling_changed
                    and (candidate_changed or scheduling_before != scheduling_after)
                    and bool(next_mask_for_decision[1:].any())
                    and not self._episode_done()
                )
                if decision_triggered:
                    self.policy_trigger_event_counts.update(
                        self._decision_trigger_sources(
                            batch_types,
                            scheduling_before,
                            scheduling_after,
                            candidate_changed,
                        )
                    )
                if scheduling_changed and not decision_triggered and not self._episode_done() and self.decision_mode != "joint_plan":
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
                    decision_triggered = bool(next_mask_for_decision[1:].any()) and not self._episode_done()
                    if decision_triggered:
                        self.policy_trigger_event_counts.update(
                            self._decision_trigger_sources(
                                batch_types,
                                scheduling_before,
                                scheduling_after,
                                True,
                            )
                        )
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
                if self._episode_done() or decision_triggered:
                    break

        terminated, truncated = self._episode_status()
        # Both values are simulator seconds.  Do not clamp: a backwards clock
        # must fail in transition_discount instead of silently becoming dt=0.
        delta_t = float(self.sim.now) - decision_time
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
        completed_task_ids = sorted(set(after["done_ids"]) - set(before["done_ids"]))
        timed_out_task_ids = sorted(set(after["expired_ids"]) - set(before["expired_ids"]))
        reward_audit = self._reward_audit_components(reward)
        self.sim.reward_total += reward
        self._accumulate_phase_reward(decision_time, reward)
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
            "terminated": bool(terminated),
            "truncated": bool(truncated),
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
            "reward_ledger_components": dict(self._last_reward_ledger_components),
            "valid_action_count": int(next_mask.sum()),
            "valid_non_wait_action_count": int(next_mask[1:].sum()),
            "forced_wait": forced_wait,
            "automatic_action": bool(automatic_action),
            "instantaneous_dispatch": instantaneous_dispatch,
            "instantaneous_decision_count": int(self._instantaneous_count),
            "selected_sortie_id": selected_sortie_id,
            "action_type": action_type,
            "completed_task_ids": completed_task_ids,
            "timed_out_task_ids": timed_out_task_ids,
            "reward_audit_components": dict(reward_audit),
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
            "selected_action": int(action),
            "action_type": action_type,
            "selected_plan_type": action_type,
            "ground_action": None if ground_action is None else ground_action.action_type,
            "selected_sortie_id": selected_sortie_id,
            "completed_task_ids": "|".join(map(str, completed_task_ids)),
            "timed_out_task_ids": "|".join(map(str, timed_out_task_ids)),
            "trigger_events": "|".join(trigger_events),
            "processed_event_count": int(processed_event_count),
            "valid_action_count": valid_action_count,
            "valid_non_wait_action_count": valid_non_wait_count,
            "forced_wait": forced_wait,
            "automatic_action": bool(automatic_action),
            "instantaneous_dispatch": instantaneous_dispatch,
            "reward_total": reward,
            "reward_completion": reward_components.get("completion", reward_components.get("completion_reward", 0.0)),
            "reward_timeout": reward_components.get("timeout", reward_components.get("timeout_cost", 0.0)),
            "reward_aoi": reward_components.get("aoi", reward_components.get("aoi_cost", 0.0)),
            "reward_energy": reward_components.get("energy", reward_components.get("flight_energy_cost", 0.0)),
            "reward_uav_wait": reward_components.get("uav_wait", reward_components.get("recovery_wait_cost", 0.0)),
            "reward_cav_wait": reward_components.get("cav_wait", reward_components.get("cav_distance_cost", 0.0)),
            "reward_other": reward_components.get("other", 0.0),
            **reward_audit,
            "stage5_reward_final": self._last_reward_ledger_components.get("final_reward", reward),
            "stage5_reward_completion": self._last_reward_ledger_components.get("completion_reward", 0.0),
            "stage5_reward_priority": self._last_reward_ledger_components.get("priority_reward", 0.0),
            "stage5_cost_timeout": self._last_reward_ledger_components.get("timeout_cost", 0.0),
            "stage5_cost_aoi": self._last_reward_ledger_components.get("aoi_cost", 0.0),
            "stage5_cost_flight_energy": self._last_reward_ledger_components.get("flight_energy_cost", 0.0),
            "stage5_cost_hover_energy": self._last_reward_ledger_components.get("hover_energy_cost", 0.0),
            "stage5_cost_return_distance": self._last_reward_ledger_components.get("return_distance_cost", 0.0),
            "stage5_cost_cav_distance": self._last_reward_ledger_components.get("cav_distance_cost", 0.0),
            "stage5_cost_recovery_wait": self._last_reward_ledger_components.get("recovery_wait_cost", 0.0),
            "stage5_cost_recovery_queue": self._last_reward_ledger_components.get("recovery_queue_cost", 0.0),
            "stage5_cost_route_change": self._last_reward_ledger_components.get("route_change_cost", 0.0),
            "stage5_cost_route_reverse": self._last_reward_ledger_components.get("route_reverse_cost", 0.0),
            "raw_route_change_delta": self._last_reward_ledger_components.get("raw_route_change_delta", 0.0),
            "raw_route_reverse_delta": self._last_reward_ledger_components.get("raw_route_reverse_delta", 0.0),
            "stage5_cost_coordination_failure": self._last_reward_ledger_components.get("coordination_failure_cost", 0.0),
            "stage5_cost_terminal_undone": self._last_reward_ledger_components.get("terminal_undone_cost", 0.0),
            "stage5_cost_illegal_action": self._last_reward_ledger_components.get("illegal_action_cost", 0.0),
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
        valid_counts = np.asarray(self._valid_action_counts, dtype=np.float64)
        smdp_durations = np.asarray(self._smdp_duration_history, dtype=np.float64)
        metrics = self.sim.metrics("rl_env").to_dict()
        decisions = len(self.transition_log)
        self._capture_cutoff_state()
        cohort = self._release_cohort_tasks()
        released = len(cohort)
        completed = sum(task.status == "done" for task in cohort)
        timed_out = sum(task.status == "expired" for task in cohort)
        censored = sum(task.status == "censored" for task in cohort)
        cutoff = self._cutoff_snapshot or {
            "released_task_count": released,
            "pending_tasks_at_cutoff": 0,
            "backlog_rate_at_cutoff": 0.0,
        }
        total_reward = float(sum(row["reward_total"] for row in self.transition_log))
        component_sum = lambda key: float(sum(float(row.get(key, 0.0)) for row in self.transition_log))
        reward_components = {
            "completion_reward": component_sum("stage5_reward_completion"),
            "priority_reward": component_sum("stage5_reward_priority"),
            "timeout_penalty": -component_sum("stage5_cost_timeout"),
            "weighted_aoi_penalty": -component_sum("stage5_cost_aoi"),
            "flight_energy_penalty": -component_sum("stage5_cost_flight_energy"),
            "hover_energy_penalty": -component_sum("stage5_cost_hover_energy"),
            "uav_return_distance_penalty": -component_sum("stage5_cost_return_distance"),
            "cav_distance_penalty": -component_sum("stage5_cost_cav_distance"),
            "recovery_wait_penalty": -component_sum("stage5_cost_recovery_wait"),
            "queue_occupancy_penalty": -component_sum("stage5_cost_recovery_queue"),
            "route_change_penalty": -component_sum("stage5_cost_route_change"),
            "route_reversal_penalty": -component_sum("stage5_cost_route_reverse"),
            "failure_penalty": -(
                component_sum("stage5_cost_coordination_failure")
                + component_sum("stage5_cost_terminal_undone")
                + component_sum("stage5_cost_illegal_action")
            ),
            "episode_total_reward": total_reward,
        }
        decomposed_total = float(
            sum(
                value
                for name, value in reward_components.items()
                if name != "episode_total_reward"
            )
        )
        reward_conservation_error = total_reward - decomposed_total
        if abs(reward_conservation_error) >= 1e-8:
            raise AssertionError(
                "episode reward components do not conserve total reward: "
                f"reward={total_reward} components={decomposed_total} "
                f"diff={reward_conservation_error}"
            )
        joint_generator = getattr(self.sim, "joint_plan_generator", None)
        reliability_metrics = {
            "illegal_actions": int(getattr(self.sim, "illegal_actions", 0)),
            "commit_failure_count": int(getattr(joint_generator, "commit_failures", 0)),
            "path_planning_failure_count": int(
                metrics.get("path_planning_failure_count", 0)
            ),
            "infeasible_recoveries": int(getattr(self.sim, "infeasible_recoveries", 0)),
            "queue_occupancy_time_s": float(
                getattr(self.sim, "recovery_queue_occupancy_integral_s", 0.0)
            ),
        }
        terminated, truncated = self._episode_status()
        return {
            **metrics,
            "episode_return": total_reward,
            "physical_episode_reward": total_reward,
            "simulated_time": float(self.sim.now),
            "task_release_horizon_s": float(self.sim.task_release_horizon),
            "episode_end_time_s": float(self.sim.now),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "released_task_count": int(released),
            "completed_task_count_final": int(completed),
            "timed_out_task_count_final": int(timed_out),
            "pending_tasks_at_cutoff": int(cutoff["pending_tasks_at_cutoff"]),
            "backlog_rate_at_cutoff": float(cutoff["backlog_rate_at_cutoff"]),
            "completed_during_drain": int(sum(
                task.status == "done"
                and task.completion_time is not None
                and float(task.completion_time) >= float(self.sim.task_release_horizon)
                for task in cohort
            )),
            "timed_out_during_drain": int(sum(
                task.status == "expired" and float(task.deadline) >= float(self.sim.task_release_horizon)
                for task in cohort
            )),
            "drain_duration_s": max(0.0, float(self.sim.now) - float(self.sim.task_release_horizon)),
            "censored_task_count": int(censored),
            "censoring_rate": float(censored / released) if released else 0.0,
            "unresolved_task_count": int(sum(
                task.status not in {"done", "expired"} for task in cohort
            )),
            "unfinished_uav_states": {
                drone.id: drone.status
                for drone in self.sim.drones
                if drone.status != "idle"
            },
            "recovery_queue_at_end": int(len(self.sim.recovery_queue)),
            "completion_rate_final": float(completed / released) if released else 0.0,
            "timeout_rate_final": float(timed_out / released) if released else 0.0,
            "operation_reward": float(self._operation_reward),
            "drain_reward": float(self._drain_reward),
            "episode_reward_total": total_reward,
            **reward_components,
            "reward_conservation_error": float(reward_conservation_error),
            **reliability_metrics,
            "reward_per_released_task": float(total_reward / released) if released else 0.0,
            "reward_per_operation_hour": float(total_reward / max(float(self.sim.task_release_horizon) / 3600.0, 1e-12)),
            "decisions_per_episode": int(decisions),
            "system_events_per_episode": int(self.system_events_processed),
            "physical_event_count": int(self.system_events_processed),
            "atomic_event_transition_count": int(decisions),
            "policy_decision_count": int(self.policy_decision_count),
            "single_valid_action_count": int(self.single_valid_action_count),
            "multi_valid_action_count": int(self.multi_valid_action_count),
            "average_valid_action_count": float(valid_counts.mean()) if valid_counts.size else 0.0,
            "automatic_action_count": int(self.automatic_action_count),
            "initial_automatic_action_count": int(self.initial_automatic_action_count),
            "initial_automatic_reward": float(self.initial_automatic_reward),
            "average_smdp_duration": float(smdp_durations.mean()) if smdp_durations.size else 0.0,
            "maximum_smdp_duration": float(smdp_durations.max()) if smdp_durations.size else 0.0,
            "non_safe_action_count": int(self.non_safe_action_count),
            "policy_trigger_event_counts": dict(sorted(self.policy_trigger_event_counts.items())),
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
            "delta_t_mean": float(delta.mean()) if delta.size else 0.0,
            "delta_t_min": float(delta.min()) if delta.size else 0.0,
            "delta_t_max": float(delta.max()) if delta.size else 0.0,
            "effective_gamma_mean": float(discounts.mean()) if discounts.size else 0.0,
            "effective_gamma_min": float(discounts.min()) if discounts.size else 0.0,
            "effective_gamma_max": float(discounts.max()) if discounts.size else 0.0,
            "forced_wait_decision_rate": self.forced_wait_decisions / decisions if decisions else 0.0,
            "zero_duration_transition_rate": self.zero_duration_transitions / decisions if decisions else 0.0,
            "simulated_seconds_per_decision": float(self.sim.now) / decisions if decisions else 0.0,
            "policy_decisions_per_simulated_hour": self.policy_decision_count / max(self.sim.now / 3600.0, 1e-12),
            "replay_transitions_per_simulated_hour": self.policy_decision_count / max(self.sim.now / 3600.0, 1e-12),
            "wall_clock_time_ms": float(sum(row.get("wall_clock_step_time_ms", 0.0) for row in self.transition_log)),
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
        # Direct-edge Xinjiekou experiments may include additional dynamic
        # inspection categories.  Every non-routine task should receive the
        # dynamic wait shaping rather than being treated as ordinary patrol.
        has_dynamic = any(t.type != "routine" for t in available_tasks)

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
        # Drain-aware episodes never transform unresolved work at the hard
        # limit into an implicit timeout-like terminal cost.  True deadline
        # expiry has already been charged by the normal timeout path.
        if sim.drain_phase_enabled:
            return 0.0
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
