from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    action_masks: torch.Tensor
    next_action_masks: torch.Tensor
    terminateds: torch.Tensor
    truncateds: torch.Tensor
    delta_ts: torch.Tensor
    bootstrap_discounts: torch.Tensor
    decision_times: torch.Tensor
    instantaneous_dispatches: torch.Tensor
    trigger_event_types: np.ndarray | None = None
    sampled_priorities: torch.Tensor | None = None
    indices: torch.Tensor | None = None
    weights: torch.Tensor | None = None


class MaskedReplayBuffer:
    """Fixed-size replay buffer that stores current and next action masks."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        seed: int = 0,
        default_transition_discount: float = 0.99,
        time_mode: str = "fixed_interval",
        require_time_fields: bool = False,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(int(seed))
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.action_masks = np.zeros((self.capacity, self.action_dim), dtype=bool)
        self.next_action_masks = np.zeros((self.capacity, self.action_dim), dtype=bool)
        self.terminateds = np.zeros(self.capacity, dtype=np.float32)
        self.truncateds = np.zeros(self.capacity, dtype=np.float32)
        self.delta_ts = np.zeros(self.capacity, dtype=np.float32)
        self.bootstrap_discounts = np.zeros(self.capacity, dtype=np.float32)
        self.decision_times = np.zeros(self.capacity, dtype=np.float64)
        self.instantaneous_dispatches = np.zeros(self.capacity, dtype=bool)
        self.trigger_event_types = np.full(self.capacity, "", dtype="<U256")
        self.default_transition_discount = float(default_transition_discount)
        self.time_mode = str(time_mode)
        self.require_time_fields = bool(require_time_fields)
        self.schema_version = 2
        self.pos = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
        *,
        terminated: bool | None = None,
        truncated: bool = False,
        delta_t: float | None = None,
        transition_discount: float | None = None,
        decision_time: float = 0.0,
        trigger_event_type: str = "",
        instantaneous_dispatch: bool = False,
    ) -> None:
        if self.require_time_fields and (delta_t is None or transition_discount is None):
            raise ValueError(
                "event-driven replay requires explicit delta_t and transition_discount; "
                "refusing to silently apply a fixed-step discount"
            )
        idx = self.pos
        self.obs[idx] = np.asarray(obs, dtype=np.float32)
        self.next_obs[idx] = np.asarray(next_obs, dtype=np.float32)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = float(done)
        self.terminateds[idx] = float(done if terminated is None else terminated)
        self.truncateds[idx] = float(truncated)
        self.delta_ts[idx] = float(60.0 if delta_t is None else delta_t)
        self.bootstrap_discounts[idx] = float(
            self.default_transition_discount if transition_discount is None else transition_discount
        )
        self.decision_times[idx] = float(decision_time)
        self.trigger_event_types[idx] = str(trigger_event_type)[:256]
        self.instantaneous_dispatches[idx] = bool(instantaneous_dispatch)
        self.action_masks[idx] = np.asarray(action_mask, dtype=bool)
        self.next_action_masks[idx] = np.asarray(next_action_mask, dtype=bool)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if self.size < int(batch_size):
            raise ValueError("not enough samples in replay buffer")
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return ReplayBatch(
            obs=torch.as_tensor(self.obs[idx], device=device),
            actions=torch.as_tensor(self.actions[idx], device=device, dtype=torch.long),
            rewards=torch.as_tensor(self.rewards[idx], device=device),
            next_obs=torch.as_tensor(self.next_obs[idx], device=device),
            dones=torch.as_tensor(self.dones[idx], device=device),
            action_masks=torch.as_tensor(self.action_masks[idx], device=device, dtype=torch.bool),
            next_action_masks=torch.as_tensor(self.next_action_masks[idx], device=device, dtype=torch.bool),
            terminateds=torch.as_tensor(self.terminateds[idx], device=device),
            truncateds=torch.as_tensor(self.truncateds[idx], device=device),
            delta_ts=torch.as_tensor(self.delta_ts[idx], device=device),
            bootstrap_discounts=torch.as_tensor(self.bootstrap_discounts[idx], device=device),
            decision_times=torch.as_tensor(self.decision_times[idx], device=device),
            instantaneous_dispatches=torch.as_tensor(self.instantaneous_dispatches[idx], device=device, dtype=torch.bool),
            trigger_event_types=self.trigger_event_types[idx].copy(),
        )

    def state_dict(self) -> Dict[str, int]:
        return {
            "schema_version": self.schema_version,
            "capacity": self.capacity,
            "size": self.size,
            "pos": self.pos,
            "time_mode": self.time_mode,
            "default_transition_discount": self.default_transition_discount,
            "stores_delta_t": True,
        }


class PrioritizedMaskedReplayBuffer(MaskedReplayBuffer):
    """Prioritized replay buffer for Rainbow-style DQN updates."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        seed: int = 0,
        alpha: float = 0.6,
        priority_epsilon: float = 1e-6,
        default_transition_discount: float = 0.99,
        time_mode: str = "fixed_interval",
        require_time_fields: bool = False,
    ) -> None:
        super().__init__(
            capacity,
            obs_dim,
            action_dim,
            seed=seed,
            default_transition_discount=default_transition_discount,
            time_mode=time_mode,
            require_time_fields=require_time_fields,
        )
        self.alpha = float(alpha)
        self.priority_epsilon = float(priority_epsilon)
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.max_priority = 1.0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
        *,
        terminated: bool | None = None,
        truncated: bool = False,
        delta_t: float | None = None,
        transition_discount: float | None = None,
        decision_time: float = 0.0,
        trigger_event_type: str = "",
        instantaneous_dispatch: bool = False,
    ) -> None:
        idx = self.pos
        super().add(
            obs,
            action,
            reward,
            next_obs,
            done,
            action_mask,
            next_action_mask,
            terminated=terminated,
            truncated=truncated,
            delta_t=delta_t,
            transition_discount=transition_discount,
            decision_time=decision_time,
            trigger_event_type=trigger_event_type,
            instantaneous_dispatch=instantaneous_dispatch,
        )
        self.priorities[idx] = self.max_priority

    def sample(self, batch_size: int, device: torch.device, beta: float = 0.4) -> ReplayBatch:
        if self.size < int(batch_size):
            raise ValueError("not enough samples in replay buffer")
        priorities = self.priorities[: self.size].astype(np.float64)
        if priorities.sum() <= 0:
            probabilities = np.full(self.size, 1.0 / self.size, dtype=np.float64)
        else:
            scaled = priorities ** self.alpha
            probabilities = scaled / scaled.sum()
        idx = self.rng.choice(self.size, size=int(batch_size), replace=True, p=probabilities)
        weights = (self.size * probabilities[idx]) ** (-float(beta))
        weights = weights / max(weights.max(), self.priority_epsilon)
        batch = ReplayBatch(
            obs=torch.as_tensor(self.obs[idx], device=device),
            actions=torch.as_tensor(self.actions[idx], device=device, dtype=torch.long),
            rewards=torch.as_tensor(self.rewards[idx], device=device),
            next_obs=torch.as_tensor(self.next_obs[idx], device=device),
            dones=torch.as_tensor(self.dones[idx], device=device),
            action_masks=torch.as_tensor(self.action_masks[idx], device=device, dtype=torch.bool),
            next_action_masks=torch.as_tensor(self.next_action_masks[idx], device=device, dtype=torch.bool),
            terminateds=torch.as_tensor(self.terminateds[idx], device=device),
            truncateds=torch.as_tensor(self.truncateds[idx], device=device),
            delta_ts=torch.as_tensor(self.delta_ts[idx], device=device),
            bootstrap_discounts=torch.as_tensor(self.bootstrap_discounts[idx], device=device),
            decision_times=torch.as_tensor(self.decision_times[idx], device=device),
            instantaneous_dispatches=torch.as_tensor(self.instantaneous_dispatches[idx], device=device, dtype=torch.bool),
            trigger_event_types=self.trigger_event_types[idx].copy(),
            sampled_priorities=torch.as_tensor(self.priorities[idx], device=device, dtype=torch.float32),
            indices=torch.as_tensor(idx, device=device, dtype=torch.long),
            weights=torch.as_tensor(weights, device=device, dtype=torch.float32),
        )
        return batch

    def update_priorities(self, indices: torch.Tensor | np.ndarray, td_errors: torch.Tensor | np.ndarray) -> None:
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().abs().cpu().numpy()
        priorities = np.asarray(td_errors, dtype=np.float32) + self.priority_epsilon
        self.priorities[np.asarray(indices, dtype=np.int64)] = priorities
        self.max_priority = max(self.max_priority, float(np.max(priorities)))

    def state_dict(self) -> Dict[str, int | float]:
        state = super().state_dict()
        state.update({"alpha": self.alpha, "max_priority": self.max_priority})
        return state
