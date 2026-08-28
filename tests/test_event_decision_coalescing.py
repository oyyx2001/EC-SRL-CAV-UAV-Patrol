from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from dynamic_patrol_marl.env.events import EventQueue
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.rl.replay_buffer import PrioritizedMaskedReplayBuffer
from dynamic_patrol_marl.utils.config import load_config


def _atomic_info(*, reward: float, discount: float, delta_t: float, mask: list[bool], terminated: bool = False) -> dict:
    return {
        "action_mask": np.asarray(mask, dtype=bool),
        "time": delta_t,
        "terminated": terminated,
        "truncated": False,
        "delta_t": delta_t,
        "transition_discount": discount,
        "decision_time": 0.0,
        "next_decision_time": delta_t,
        "trigger_events": ["STATE_ADVANCE"],
        "processed_event_count": 1,
        "stale_event_count": 0,
        "duplicate_event_count": 0,
        "invalid_time_event_count": 0,
        "reward_components": {"component": reward},
        "reward_ledger_components": {"final_reward": reward},
        "reward_audit_components": {"component": reward},
        "completed_task_ids": [],
        "timed_out_task_ids": [],
        "automatic_action": False,
        "wall_clock_step_time_ms": 0.0,
    }


def _coalescing_harness(terminal_on_auto: bool = False) -> tuple[DynamicPatrolEnv, list[tuple[int, bool]]]:
    env = DynamicPatrolEnv.__new__(DynamicPatrolEnv)
    env.sim = SimpleNamespace()
    env.event_queue = SimpleNamespace()
    env.event_cfg = {"max_automatic_advance_steps": 10}
    env.policy_decision_count = 0
    env._smdp_duration_history = []
    calls: list[tuple[int, bool]] = []
    env.action_masks = MethodType(lambda self: np.asarray([True, True], dtype=bool), env)

    def atomic(self, action: int, *, automatic_action: bool = False):
        calls.append((int(action), bool(automatic_action)))
        if len(calls) == 1:
            info = _atomic_info(reward=2.0, discount=0.5, delta_t=10.0, mask=[True])
            info["automatic_action"] = bool(automatic_action)
            return (
                np.asarray([1.0], dtype=np.float32),
                2.0,
                False,
                False,
                info,
            )
        info = _atomic_info(
            reward=4.0,
            discount=0.5,
            delta_t=20.0,
            mask=[True] if terminal_on_auto else [True, True],
            terminated=terminal_on_auto,
        )
        info["automatic_action"] = bool(automatic_action)
        return (
            np.asarray([2.0], dtype=np.float32),
            4.0,
            bool(terminal_on_auto),
            False,
            info,
        )

    env._step_event_driven_once = MethodType(atomic, env)
    return env, calls


def test_same_timestamp_events_are_popped_as_one_batch() -> None:
    queue = EventQueue()
    queue.schedule(time=10.0, event_type="UAV_CLIMB_COMPLETE", entity_id="drone0")
    queue.schedule(time=10.0, event_type="TASK_RELEASE", entity_id="task:1")
    queue.schedule(time=10.0, event_type="CAV_NODE_ARRIVAL", entity_id="cav0")

    batch = queue.pop_same_time(10.0, current_time=0.0)

    assert len(batch) == 3
    assert {event.event_type for event in batch} == {
        "UAV_CLIMB_COMPLETE", "TASK_RELEASE", "CAV_NODE_ARRIVAL",
    }
    assert queue.peek_time(0.0) is None


def test_uav_climb_only_progress_does_not_change_scheduling_fingerprint() -> None:
    env = DynamicPatrolEnv.__new__(DynamicPatrolEnv)
    drone = SimpleNamespace(id="drone0", status="flying", position=10, battery=100.0, segment_index=0)
    vehicle = SimpleNamespace(
        position=10,
        target_node=11,
        at_intersection=False,
        path_nodes=[10, 11],
        schedule_version=3,
    )
    env.sim = SimpleNamespace(
        now=20.0,
        vehicle=vehicle,
        recovery_queue=[],
        task_pool=SimpleNamespace(available_tasks=lambda now: []),
        idle_drones_on_vehicle=lambda: [],
    )

    before = env._scheduling_fingerprint()
    drone.segment_index = 1  # UAV_CLIMB_COMPLETE -> ordinary flight
    after = env._scheduling_fingerprint()

    assert before == after


def test_simultaneous_climb_is_not_credited_when_cav_event_creates_choice() -> None:
    before = (tuple(), tuple(), (1, 2, False, (1, 2), 1), tuple())
    after = (tuple(), tuple(), (2, None, True, (2,), 2), tuple())

    sources = DynamicPatrolEnv._decision_trigger_sources(
        ["UAV_CLIMB_COMPLETE", "CAV_WAIT_COMPLETE"],
        before,
        after,
        candidate_changed=True,
    )

    assert "CAV_WAIT_COMPLETE" in sources
    assert "UAV_CLIMB_COMPLETE" not in sources


def test_single_valid_action_is_automatically_coalesced_with_correct_discount() -> None:
    env, calls = _coalescing_harness()

    obs, reward, terminated, truncated, info = env._step_event_driven(1)

    assert calls == [(1, False), (0, True)]
    assert obs.tolist() == [2.0]
    assert terminated is False and truncated is False
    assert reward == pytest.approx(2.0 + 0.5 * 4.0)
    assert info["delta_t"] == pytest.approx(30.0)
    assert info["transition_discount"] == pytest.approx(0.25)
    assert info["automatic_action_count"] == 1
    assert info["smdp_subtransition_count"] == 2
    assert info["reward_conservation_error"] == pytest.approx(0.0)
    assert env.policy_decision_count == 1


def test_terminal_during_automatic_advance_returns_final_transition() -> None:
    env, calls = _coalescing_harness(terminal_on_auto=True)

    _obs, reward, terminated, truncated, info = env._step_event_driven(1)

    assert calls == [(1, False), (0, True)]
    assert terminated is True and truncated is False
    assert reward == pytest.approx(4.0)
    assert info["transition_discount"] == pytest.approx(0.25)
    assert info["smdp_subtransition_count"] == 2
    assert info["automatic_action_count"] == 1


def test_coalesced_transition_keeps_time_fields_and_per_priority_path() -> None:
    env, _calls = _coalescing_harness()
    obs, reward, terminated, truncated, info = env._step_event_driven(1)
    replay = PrioritizedMaskedReplayBuffer(
        capacity=2,
        obs_dim=2,
        action_dim=2,
        require_time_fields=True,
    )

    replay.add(
        np.asarray([0.0, 0.0], dtype=np.float32),
        1,
        reward,
        np.asarray([obs[0], 0.0], dtype=np.float32),
        terminated or truncated,
        np.asarray([True, True]),
        np.asarray(info["action_mask"], dtype=bool),
        terminated=terminated,
        truncated=truncated,
        delta_t=info["delta_t"],
        transition_discount=info["transition_discount"],
    )
    replay.update_priorities(np.asarray([0]), np.asarray([2.0]))

    assert replay.rewards[0] == pytest.approx(4.0)
    assert replay.delta_ts[0] == pytest.approx(30.0)
    assert replay.bootstrap_discounts[0] == pytest.approx(0.25)
    assert replay.priorities[0] > 2.0


def test_task_and_operational_reward_components_remain_conservative_after_coalescing() -> None:
    env = DynamicPatrolEnv.__new__(DynamicPatrolEnv)
    first = _atomic_info(reward=10.0, discount=0.5, delta_t=5.0, mask=[True])
    first["reward_components"] = {"completion_reward": 12.0, "weighted_aoi_cost": -2.0}
    first["reward_audit_components"] = dict(first["reward_components"])
    second = _atomic_info(reward=-4.0, discount=0.5, delta_t=15.0, mask=[True, True])
    second["reward_components"] = {"weighted_aoi_cost": -3.0, "flight_energy_cost": -1.0}
    second["reward_audit_components"] = dict(second["reward_components"])

    _obs, reward, _terminated, _truncated, info = env._coalesce_event_steps(
        [
            (np.zeros(1, dtype=np.float32), 10.0, False, False, first),
            (np.ones(1, dtype=np.float32), -4.0, False, False, second),
        ]
    )

    assert reward == pytest.approx(8.0)
    assert info["reward_components"]["completion_reward"] == pytest.approx(12.0)
    assert info["reward_components"]["weighted_aoi_cost"] == pytest.approx(-3.5)
    assert info["reward_components"]["flight_energy_cost"] == pytest.approx(-0.5)
    assert sum(info["reward_components"].values()) == pytest.approx(reward)


def test_default_joint_action_capacity_remains_unchanged() -> None:
    cfg = load_config("configs/default_training.yaml")

    assert cfg["joint_plan"]["max_non_safe_plans"] == 24
    assert 1 + cfg["joint_plan"]["max_non_safe_plans"] == 25
