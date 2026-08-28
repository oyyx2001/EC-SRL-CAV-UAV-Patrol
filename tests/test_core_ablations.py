from __future__ import annotations

from pathlib import Path

import numpy as np

from dynamic_patrol_marl.env.cav_controller import GroundCandidate
from dynamic_patrol_marl.env.simulator import Simulator
from dynamic_patrol_marl.planner.joint_plan import JointPlan
from dynamic_patrol_marl.rl.train_masked_dqn import DQNHyperParams, epsilon_by_step, exploration_epsilon
from dynamic_patrol_marl.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _config(name: str):
    return load_config(ROOT / "configs" / "ablations" / f"{name}.yaml")


def _queue_fixture(sim: Simulator, node: int, eta: float) -> dict:
    return {
        "sortie_id": "ablation-fixture",
        "drone_id": "drone0",
        "uav_id": "drone0",
        "recovery_node": int(node),
        "eta": float(eta),
        "battery_at_eta": 1500.0,
        "battery_reserve": 0.0,
        "hover_power": 0.65,
        "status": "planned",
        "locked": False,
        "recovery_duration": 0.0,
    }


def _ground_signature(config_name: str, recovery_node: int, eta: float):
    sim = Simulator(_config(config_name), seed=0)
    sim.recovery_queue.append(_queue_fixture(sim, recovery_node, eta))
    return [
        (
            candidate.action_type,
            candidate.target_node,
            candidate.waypoint_node,
            round(float(candidate.score), 9),
            tuple(candidate.planned_route),
        )
        for candidate in sim.ground_controller.generate_ground_candidates()
    ]


def test_queue_blind_candidates_ignore_changed_queue_but_full_does_not() -> None:
    probe = Simulator(_config("full"), seed=0)
    nodes = sorted(probe.network.nodes)

    blind_a = _ground_signature("queue_blind", nodes[3], 100.0)
    blind_b = _ground_signature("queue_blind", nodes[-4], 500.0)
    assert blind_a == blind_b

    full_a = _ground_signature("full", nodes[3], 100.0)
    full_b = _ground_signature("full", nodes[-4], 500.0)
    assert full_a != full_b


def test_queue_blind_zeroes_all_declared_queue_policy_features() -> None:
    blind_sim = Simulator(_config("queue_blind"), seed=0)
    full_sim = Simulator(_config("full"), seed=0)
    candidate = GroundCandidate(candidate_id=0, action_type="keep_route")

    def make_plan(recovery_wait: float, cav_wait: float, existing_delay: float, insertion_cost: float, queue_len: int):
        return JointPlan(
            plan_id="fixture",
            slot_index=0,
            plan_type="safe",
            ground_candidate=candidate,
            predicted_recovery_wait=recovery_wait,
            predicted_cav_wait=cav_wait,
            predicted_existing_queue_delay=existing_delay,
            insertion_evaluation={"insertion_cost": insertion_cost},
            queue_length_after=queue_len,
        )

    low = make_plan(2.0, 3.0, 5.0, 7.0, 1)
    high = make_plan(200.0, 300.0, 500.0, 700.0, 3)
    blind_low = blind_sim.joint_plan_generator.feature_builder.build(low)
    blind_high = blind_sim.joint_plan_generator.feature_builder.build(high)
    full_low = full_sim.joint_plan_generator.feature_builder.build(low)
    full_high = full_sim.joint_plan_generator.feature_builder.build(high)
    names = blind_sim.joint_plan_generator.feature_builder.FEATURE_NAMES
    hidden = (
        "recovery_wait",
        "cav_wait",
        "existing_queue_delay",
        "insertion_cost",
        "queue_length",
        "locked_prefix_length",
    )
    hidden_indices = [names.index(name) for name in hidden]

    assert all(blind_low[index] == 0.0 and blind_high[index] == 0.0 for index in hidden_indices)
    assert np.allclose(blind_low, blind_high)
    assert any(full_low[index] != full_high[index] for index in hidden_indices)


def test_fixed_cav_uses_a_nonstationary_directed_passive_patrol() -> None:
    sim = Simulator(_config("fixed_cav"), seed=0)
    initial_position = int(sim.vehicle.position)
    assert sim.fixed_cav_patrol_enabled
    assert len(sim.vehicle.path_nodes) >= 2
    for u, v in zip(sim.vehicle.path_nodes[:-1], sim.vehicle.path_nodes[1:]):
        assert sim.network.G.has_edge(int(u), int(v))

    sim.step(120.0)

    assert sim.vehicle.distance_traveled_m > 0.0
    assert sim.vehicle.route_edge_log
    assert int(sim.vehicle.position) != initial_position or len(sim.vehicle.route_edge_log) >= 2


def test_rainbow_uses_uniform_replay_warmup_before_noisynet_only_exploration() -> None:
    hp = DQNHyperParams(learning_starts=1000)

    assert exploration_epsilon("rainbow_dqn", 1, hp) == 1.0
    assert exploration_epsilon("rainbow_dqn", 1000, hp) == 1.0
    assert exploration_epsilon("rainbow_dqn", 1001, hp) == 0.0
    assert exploration_epsilon("double_dqn", 1001, hp) == epsilon_by_step(1001, hp)
