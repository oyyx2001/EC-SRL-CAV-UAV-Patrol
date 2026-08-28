from __future__ import annotations

import numpy as np
import pytest
import torch

from dynamic_patrol_marl.rl.replay_buffer import MaskedReplayBuffer
from dynamic_patrol_marl.rl.smdp import (
    NStepSMDPAccumulator,
    NStepTransition,
    one_step_smdp_target,
    resolve_discount_settings,
    transition_discount,
)


@pytest.mark.parametrize(
    ("delta_t", "expected"),
    [
        (60.0, 0.99),
        (120.0, 0.99**2),
        (30.0, np.sqrt(0.99)),
    ],
)
def test_time_aware_discount_examples(delta_t: float, expected: float) -> None:
    assert transition_discount(0.99, delta_t, 60.0, "time_aware") == pytest.approx(expected)


def test_done_transition_does_not_bootstrap() -> None:
    rewards = torch.tensor([3.0, 3.0])
    discounts = torch.tensor([0.99, 0.99])
    dones = torch.tensor([1.0, 0.0])
    next_q = torch.tensor([100.0, 100.0])

    target = one_step_smdp_target(rewards, discounts, dones, next_q)

    assert target[0].item() == pytest.approx(3.0)
    assert target[1].item() == pytest.approx(102.0)


def test_fixed_decision_ignores_duration() -> None:
    assert transition_discount(0.99, 0.0, 60.0, "fixed_decision") == pytest.approx(0.99)
    assert transition_discount(0.99, 1.0e9, 60.0, "fixed_decision") == pytest.approx(0.99)


def test_zero_tiny_and_very_long_time_aware_durations() -> None:
    assert transition_discount(0.99, 0.0, 60.0, "time_aware") == pytest.approx(1.0)
    tiny = transition_discount(0.99, 1.0e-12, 60.0, "time_aware")
    very_long = transition_discount(0.99, 1.0e9, 60.0, "time_aware")
    assert 0.99 < tiny <= 1.0
    assert 0.0 <= very_long < 0.99


@pytest.mark.parametrize("delta_t", [-1.0e-12, -1.0, -np.inf, np.nan])
def test_invalid_delta_t_fails_fast(delta_t: float) -> None:
    with pytest.raises(ValueError, match="delta_t"):
        transition_discount(0.99, delta_t, 60.0, "time_aware")


def test_replay_rejects_negative_delta_t() -> None:
    replay = MaskedReplayBuffer(2, 2, 2, require_time_fields=True)
    with pytest.raises(ValueError, match="delta_t"):
        replay.add(
            np.zeros(2),
            0,
            0.0,
            np.zeros(2),
            False,
            np.ones(2, dtype=bool),
            np.ones(2, dtype=bool),
            delta_t=-1.0e-12,
            transition_discount=0.99,
        )


def test_public_rl_discount_configuration_takes_precedence() -> None:
    cfg = {
        "rl": {
            "discount_mode": "time_aware",
            "gamma": 0.99,
            "discount_reference_time_s": 60.0,
            "n_step": 1,
        },
        "smdp": {
            "discount_mode": "fixed_decision",
            "gamma_reference": 0.5,
            "reference_interval_s": 1.0,
            "n_step": 3,
        },
    }
    assert resolve_discount_settings(cfg) == ("time_aware", 0.99, 60.0, 1)


def test_terminal_transition_keeps_effective_gamma_for_logging() -> None:
    transition = NStepTransition(
        obs=np.zeros(2, dtype=np.float32),
        action=0,
        reward=1.0,
        next_obs=np.ones(2, dtype=np.float32),
        terminated=True,
        truncated=False,
        action_mask=np.ones(2, dtype=bool),
        next_action_mask=np.ones(2, dtype=bool),
        delta_t=60.0,
        bootstrap_discount=0.99,
        decision_time=0.0,
    )
    stored = NStepSMDPAccumulator(n_step=1).append(transition)
    assert len(stored) == 1
    assert stored[0].done is True
    assert stored[0].bootstrap_discount == pytest.approx(0.99)
