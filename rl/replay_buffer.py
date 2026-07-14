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
    indices: torch.Tensor | None = None
    weights: torch.Tensor | None = None


class MaskedReplayBuffer:
    """Fixed-size replay buffer that stores current and next action masks."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int = 0) -> None:
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
    ) -> None:
        idx = self.pos
        self.obs[idx] = np.asarray(obs, dtype=np.float32)
        self.next_obs[idx] = np.asarray(next_obs, dtype=np.float32)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = float(done)
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
        )

    def state_dict(self) -> Dict[str, int]:
        return {"capacity": self.capacity, "size": self.size, "pos": self.pos}


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
    ) -> None:
        super().__init__(capacity, obs_dim, action_dim, seed=seed)
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
    ) -> None:
        idx = self.pos
        super().add(obs, action, reward, next_obs, done, action_mask, next_action_mask)
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
