from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Sequence

import numpy as np


def transition_discount(
    gamma_reference: float,
    delta_t: float,
    reference_interval_s: float = 60.0,
    discount_mode: str = "continuous_time",
) -> float:
    """Return the effective SMDP bootstrap discount for one transition."""

    gamma = float(gamma_reference)
    duration = float(delta_t)
    reference = float(reference_interval_s)
    if duration < -1e-9:
        raise ValueError(f"delta_t must be non-negative, got {duration}")
    if reference <= 0.0:
        raise ValueError("reference_interval_s must be positive")
    if discount_mode == "fixed_decision":
        return gamma
    if discount_mode != "continuous_time":
        raise ValueError(f"unknown discount_mode: {discount_mode}")
    return float(gamma ** (max(0.0, duration) / reference))


def one_step_smdp_target(
    rewards: Any,
    bootstrap_discounts: Any,
    dones: Any,
    next_q: Any,
) -> Any:
    return rewards + (1.0 - dones) * bootstrap_discounts * next_q


@dataclass
class NStepTransition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray
    delta_t: float
    bootstrap_discount: float
    decision_time: float
    trigger_event_type: str = ""
    instantaneous_dispatch: bool = False

    @property
    def done(self) -> bool:
        return bool(self.terminated or self.truncated)


class NStepSMDPAccumulator:
    """Aggregate transition-specific discounts without assuming ``gamma**n``."""

    def __init__(self, n_step: int) -> None:
        if int(n_step) < 1:
            raise ValueError("n_step must be >= 1")
        self.n_step = int(n_step)
        self._queue: Deque[NStepTransition] = deque()

    def __len__(self) -> int:
        return len(self._queue)

    def _aggregate_prefix(self, length: int) -> NStepTransition:
        items = list(self._queue)[: int(length)]
        first = items[0]
        reward = 0.0
        discount_product = 1.0
        total_delta_t = 0.0
        terminal_index = len(items) - 1
        for index, item in enumerate(items):
            reward += discount_product * float(item.reward)
            total_delta_t += float(item.delta_t)
            discount_product *= float(item.bootstrap_discount)
            terminal_index = index
            if item.done:
                discount_product = 0.0
                break
        last = items[terminal_index]
        return NStepTransition(
            obs=np.asarray(first.obs, dtype=np.float32),
            action=int(first.action),
            reward=float(reward),
            next_obs=np.asarray(last.next_obs, dtype=np.float32),
            terminated=bool(last.terminated),
            truncated=bool(last.truncated),
            action_mask=np.asarray(first.action_mask, dtype=bool),
            next_action_mask=np.asarray(last.next_action_mask, dtype=bool),
            delta_t=float(total_delta_t),
            bootstrap_discount=float(discount_product),
            decision_time=float(first.decision_time),
            trigger_event_type=str(last.trigger_event_type),
            instantaneous_dispatch=bool(first.instantaneous_dispatch),
        )

    def append(self, transition: NStepTransition) -> List[NStepTransition]:
        self._queue.append(transition)
        output: List[NStepTransition] = []
        if transition.done:
            while self._queue:
                output.append(self._aggregate_prefix(min(self.n_step, len(self._queue))))
                self._queue.popleft()
        elif len(self._queue) >= self.n_step:
            output.append(self._aggregate_prefix(self.n_step))
            self._queue.popleft()
        return output

    def clear(self) -> None:
        self._queue.clear()


def c51_project_distribution(
    rewards: Any,
    bootstrap_discounts: Any,
    dones: Any,
    next_probabilities: Any,
    support: Any,
) -> Any:
    """Project an SMDP C51 target while preserving the input array backend.

    This repository's current Rainbow head is scalar (not C51).  Keeping the
    projection independent makes the time-correct target fully tested without
    changing the frozen network architecture in stage two.
    """

    try:
        import torch

        if isinstance(next_probabilities, torch.Tensor):
            probs = next_probabilities
            z = support.to(device=probs.device, dtype=probs.dtype)
            r = torch.as_tensor(rewards, device=probs.device, dtype=probs.dtype).reshape(-1, 1)
            d = torch.as_tensor(bootstrap_discounts, device=probs.device, dtype=probs.dtype).reshape(-1, 1)
            terminal = torch.as_tensor(dones, device=probs.device, dtype=probs.dtype).reshape(-1, 1)
            v_min, v_max = z[0], z[-1]
            atoms = int(z.numel())
            delta_z = (v_max - v_min) / max(1, atoms - 1)
            tz = (r + (1.0 - terminal) * d * z.reshape(1, -1)).clamp(v_min, v_max)
            b = (tz - v_min) / delta_z
            lower = b.floor().long()
            upper = b.ceil().long()
            projection = torch.zeros_like(probs)
            offset = torch.arange(probs.shape[0], device=probs.device).unsqueeze(1) * atoms
            projection.view(-1).index_add_(0, (lower + offset).view(-1), (probs * (upper.to(probs.dtype) - b)).view(-1))
            projection.view(-1).index_add_(0, (upper + offset).view(-1), (probs * (b - lower.to(probs.dtype))).view(-1))
            equal = upper == lower
            if equal.any():
                projection.view(-1).index_add_(0, (lower + offset)[equal], probs[equal])
            return projection
    except ImportError:  # pragma: no cover
        pass

    probs_np = np.asarray(next_probabilities, dtype=np.float64)
    z_np = np.asarray(support, dtype=np.float64)
    rewards_np = np.asarray(rewards, dtype=np.float64).reshape(-1, 1)
    discounts_np = np.asarray(bootstrap_discounts, dtype=np.float64).reshape(-1, 1)
    dones_np = np.asarray(dones, dtype=np.float64).reshape(-1, 1)
    v_min, v_max = float(z_np[0]), float(z_np[-1])
    atoms = len(z_np)
    delta_z = (v_max - v_min) / max(1, atoms - 1)
    tz = np.clip(rewards_np + (1.0 - dones_np) * discounts_np * z_np.reshape(1, -1), v_min, v_max)
    b = (tz - v_min) / delta_z
    lower = np.floor(b).astype(np.int64)
    upper = np.ceil(b).astype(np.int64)
    projection = np.zeros_like(probs_np)
    for batch_index in range(probs_np.shape[0]):
        for atom_index in range(atoms):
            probability = probs_np[batch_index, atom_index]
            lo, hi = lower[batch_index, atom_index], upper[batch_index, atom_index]
            if lo == hi:
                projection[batch_index, lo] += probability
            else:
                projection[batch_index, lo] += probability * (hi - b[batch_index, atom_index])
                projection[batch_index, hi] += probability * (b[batch_index, atom_index] - lo)
    return projection


def td_error_delta_t_bin(delta_t: float) -> str:
    value = float(delta_t)
    if value < 30.0:
        return "delta_0_30"
    if value < 60.0:
        return "delta_30_60"
    if value <= 120.0:
        return "delta_60_120"
    return "delta_over_120"
