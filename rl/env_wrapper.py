from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - fallback keeps baseline code runnable without gymnasium.
    gym = None
    spaces = None

from dynamic_patrol_marl.env.simulator import Simulator
from dynamic_patrol_marl.planner.sortie_generator import Sortie


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
        self.max_actions = 1 + int(cfg["agents"].get("num_drones", 2)) * int(cfg["sortie"].get("top_k_per_drone", 8))
        self.max_tasks_obs = 10
        self.max_candidates_obs = self.max_actions - 1
        self.obs_dim = 6 + int(cfg["agents"].get("num_drones", 2)) * 3 + self.max_tasks_obs * 6 + self.max_candidates_obs * 6
        self.action_space = spaces.Discrete(self.max_actions) if spaces else SimpleDiscrete(self.max_actions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32) if spaces else SimpleBox((self.obs_dim,))
        self.sim: Optional[Simulator] = None
        self.candidates: List[Sortie] = []
        self.last_metrics = None

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
        self.candidates = self.sim.get_candidates(no_multitask=self.no_multitask, no_priority=self.no_priority)
        obs = self._observation()
        info = {"action_mask": self.action_masks(), "time": self.sim.now, "recovery_queue_len": len(self.sim.recovery_queue)}
        return (obs, info) if gym is not None else obs

    def step(self, action: int):
        assert self.sim is not None, "call reset first"
        mask = self.action_masks()
        illegal = False
        selected = None
        if action < 0 or action >= self.max_actions or not bool(mask[action]):
            illegal = True
            self.sim.illegal_actions += 1
        elif action > 0:
            idx = action - 1
            if idx < len(self.candidates):
                selected = self.candidates[idx]
        before_done_ids = {t.id for t in self.sim.tasks if t.status == "done"}
        before_done = len(before_done_ids)
        before_expired = sum(t.status == "expired" for t in self.sim.tasks)
        wait_penalty = self._wait_penalty(action)
        launched = self.sim.dispatch_sorties(selected)
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
        reward = self._reward(
            after_done - before_done,
            after_expired - before_expired,
            illegal,
            wait_penalty,
            new_done_tasks,
            terminal_penalty,
        )
        truncated = False
        self.candidates = self.sim.get_candidates(no_multitask=self.no_multitask, no_priority=self.no_priority)
        obs = self._observation()
        info = {
            "action_mask": self.action_masks(),
            "time": self.sim.now,
            "illegal": illegal,
            "recovery_queue_len": len(self.sim.recovery_queue),
            **self.sim.metrics("rl_env").to_dict(),
        }
        if gym is not None:
            return obs, float(reward), bool(terminated), bool(truncated), info
        return obs, float(reward), bool(terminated), info

    def action_masks(self) -> np.ndarray:
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
        if use_candidate_features:
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
