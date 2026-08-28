from __future__ import annotations

from types import SimpleNamespace

import pytest

from dynamic_patrol_marl.env.cav_controller import is_immediate_route_reversal
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.rl.reward_ledger import RewardLedger


def _snapshot(**overrides):
    values = {
        "done_ids": set(),
        "expired_ids": set(),
        "energy": 0.0,
        "aoi_integral": 0.0,
        "uav_wait": 0.0,
        "cav_wait": 0.0,
        "hover_time": 0.0,
        "return_distance": 0.0,
        "cav_distance": 0.0,
        "queue_length": 0.0,
        "queue_occupancy_integral": 0.0,
        "queue_travel_time": 0.0,
        "route_change_count": 0.0,
        "route_reverse_count": 0.0,
        "coordination_failure_count": 0.0,
    }
    values.update(overrides)
    return values


def _minimal_reward_env(weights: dict[str, float]) -> DynamicPatrolEnv:
    env = DynamicPatrolEnv.__new__(DynamicPatrolEnv)
    env.cfg = {
        "reward": {
            "mode": "stage5_coordinated",
            "stage5_weights": weights,
            "stage5_references": {
                "time_s": 60.0,
                "return_distance_m": 1000.0,
                "cav_distance_m": 1000.0,
            },
        },
        "agents": {"drone_hover_power": 0.65},
    }
    env.smdp_cfg = {"interval_normalize_reward": False}
    env.reference_interval_s = 60.0
    env.reward_ledger = RewardLedger(tolerance=1e-8)
    env.sim = SimpleNamespace(task_pool=SimpleNamespace(by_id={}))
    env._last_reward_ledger_components = {}
    return env


def test_route_penalties_use_only_current_transition_deltas() -> None:
    env = _minimal_reward_env(
        {
            "route_change": 0.10,
            "route_reverse": 0.40,
            "coordination_failure": 30.0,
            "illegal_action": 30.0,
        }
    )

    components = env._stage5_coordinated_reward_components(
        _snapshot(route_change_count=5.0, route_reverse_count=2.0),
        _snapshot(route_change_count=7.0, route_reverse_count=3.0),
        new_done_ids=set(),
        new_expired_ids=set(),
        done_tasks=[],
        illegal=False,
        wait_penalty=0.0,
        delta_t=10.0,
        terminated=False,
    )

    assert components["route_change_cost"] == pytest.approx(-0.20)
    assert components["route_reverse_cost"] == pytest.approx(-0.40)
    assert sum(components.values()) == pytest.approx(-0.60)
    assert env._last_reward_ledger_components["raw_route_change_delta"] == 2.0
    assert env._last_reward_ledger_components["raw_route_reverse_delta"] == 1.0


def test_timeout_penalty_includes_task_priority() -> None:
    env = _minimal_reward_env(
        {
            "timeout": 30.0,
            "timeout_priority": 2.0,
            "coordination_failure": 30.0,
            "illegal_action": 30.0,
        }
    )
    env.sim.task_pool.by_id[7] = SimpleNamespace(priority=3.0)

    components = env._stage5_coordinated_reward_components(
        _snapshot(),
        _snapshot(expired_ids={7}),
        new_done_ids=set(),
        new_expired_ids={7},
        done_tasks=[],
        illegal=False,
        wait_penalty=0.0,
        delta_t=10.0,
        terminated=False,
    )

    assert components["timeout_cost"] == pytest.approx(-(30.0 + 2.0 * 3.0))


def test_reward_snapshot_counts_real_non_emergency_route_events() -> None:
    env = DynamicPatrolEnv.__new__(DynamicPatrolEnv)
    controller = SimpleNamespace(
        route_change_log=[
            {"action_type": "keep_route", "route_changed": False},
            {"action_type": "reroute", "route_changed": True},
            {"action_type": "move_to", "route_changed": True, "emergency_override": True},
            {"action_type": "reroute", "route_changed": True, "reverse_route_flag": True},
        ]
    )
    env.sim = SimpleNamespace(
        now=0.0,
        tasks=[],
        drones=[],
        vehicle=SimpleNamespace(
            position=0,
            target_node=None,
            wait_time=0.0,
            distance_traveled_m=0.0,
        ),
        aoi_integral=0.0,
        recovery_queue=[],
        uav_return_distance_m=0.0,
        recovery_queue_occupancy_integral_s=0.0,
        recovery_queue_travel_time_s=lambda: 0.0,
        ground_controller=controller,
        joint_plan_generator=SimpleNamespace(commit_failures=0),
        infeasible_recoveries=0,
    )

    snapshot = env._reward_snapshot()

    assert snapshot["route_change_count"] == 2.0
    assert snapshot["route_reverse_count"] == 1.0


def test_route_reversal_detector_uses_previous_directed_edge() -> None:
    edge_log = [(120.0, 4, 9)]

    assert is_immediate_route_reversal([9, 4, 2], edge_log) is True
    assert is_immediate_route_reversal([9, 5, 2], edge_log) is False
    assert is_immediate_route_reversal([9], edge_log) is False
    assert is_immediate_route_reversal([9, 4], []) is False
