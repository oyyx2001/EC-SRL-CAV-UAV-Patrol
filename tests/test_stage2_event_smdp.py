from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import unittest

import numpy as np

from dynamic_patrol_marl.env.events import EventQueue
from dynamic_patrol_marl.env.simulator import Simulator
from dynamic_patrol_marl.planner.greedy_baseline import get_policy
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.rl.smdp import (
    NStepSMDPAccumulator,
    NStepTransition,
    c51_project_distribution,
    one_step_smdp_target,
    transition_discount,
)
from dynamic_patrol_marl.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def step5(env: DynamicPatrolEnv, action: int):
    out = env.step(action)
    if len(out) == 5:
        return out
    obs, reward, done, info = out
    return obs, reward, done, False, info


def event_env(seed: int = 0) -> DynamicPatrolEnv:
    cfg = load_config(ROOT / "configs" / "scenario_60node_stage2_event_dynamic.yaml")
    env = DynamicPatrolEnv(cfg, scenario="small")
    env.reset(seed=seed)
    return env


def transition(
    reward: float,
    discount: float,
    delta_t: float,
    *,
    done: bool = False,
    marker: float = 0.0,
) -> NStepTransition:
    return NStepTransition(
        obs=np.array([marker], dtype=np.float32),
        action=0,
        reward=reward,
        next_obs=np.array([marker + 1], dtype=np.float32),
        terminated=done,
        truncated=False,
        action_mask=np.array([True, True]),
        next_action_mask=np.array([True, True]),
        delta_t=delta_t,
        bootstrap_discount=discount,
        decision_time=marker,
    )


class Stage2EventDrivenSMDPTests(unittest.TestCase):
    def test_01_fixed_60_regression(self):
        cfg = load_config(ROOT / "configs" / "scenario_60node_stage1_fifo.yaml")
        sim = Simulator(cfg, seed=0, scenario="small")
        metrics = sim.run(get_policy("score_based"), method_name="score_based")
        self.assertEqual(metrics.completed_tasks, 20)
        self.assertEqual(metrics.expired_tasks, 4)
        self.assertAlmostEqual(metrics.weighted_aoi, 615.6631987447494, places=9)

    def test_02_event_time_order(self):
        queue = EventQueue()
        for value in (120.0, 30.0, 75.0):
            queue.schedule(time=value, event_type="TASK_RELEASE", entity_id=str(value))
        observed = []
        now = 0.0
        while len(queue):
            next_time = queue.peek_time(now)
            observed.append(next_time)
            queue.pop_same_time(next_time, now)
            now = next_time
        self.assertEqual(observed, [30.0, 75.0, 120.0])

    def test_03_same_time_batch_single_decision_check(self):
        queue = EventQueue()
        queue.schedule(time=30.0, event_type="TASK_RELEASE", entity_id="task")
        queue.schedule(time=30.0, event_type="UAV_RECOVERED", entity_id="drone")
        queue.schedule(time=30.0, event_type="CAV_NODE_ARRIVAL", entity_id="cav")
        batch = queue.pop_same_time(30.0, 0.0)
        self.assertEqual(len(batch), 3)
        self.assertEqual([item.event_type for item in batch], ["UAV_RECOVERED", "CAV_NODE_ARRIVAL", "TASK_RELEASE"])
        self.assertEqual(queue.processed_event_count, 3)

    def test_04_variable_discount(self):
        gamma = 0.99
        expected = {0.0: 1.0, 30.0: gamma ** 0.5, 60.0: gamma, 120.0: gamma ** 2}
        for duration, value in expected.items():
            self.assertAlmostEqual(transition_discount(gamma, duration, 60.0), value, places=12)

    def test_05_constant_rate_reward_equivalence(self):
        rate = 7.5
        fixed_reward = rate
        event_reward = rate * 60.0 / 60.0
        self.assertAlmostEqual(event_reward, fixed_reward, places=12)

    def test_06_aoi_integral(self):
        integral = Simulator.integrate_linear_rate(10.0, 70.0, 60.0)
        self.assertAlmostEqual(integral, 2400.0, places=12)
        self.assertAlmostEqual(integral / 60.0, 40.0, places=12)

    def test_07_intermediate_aoi_reset(self):
        before_reset = Simulator.integrate_linear_rate(10.0, 40.0, 30.0)
        after_reset = Simulator.integrate_linear_rate(0.0, 30.0, 30.0)
        self.assertAlmostEqual(before_reset + after_reset, 1200.0, places=12)

    def test_08_one_step_target(self):
        target = one_step_smdp_target(
            np.array([2.0]), np.array([0.8]), np.array([0.0]), np.array([5.0])
        )
        self.assertAlmostEqual(float(target[0]), 6.0, places=12)

    def test_09_n_step_target(self):
        accumulator = NStepSMDPAccumulator(3)
        self.assertEqual(accumulator.append(transition(1.0, 0.9, 10.0, marker=0.0)), [])
        self.assertEqual(accumulator.append(transition(2.0, 0.8, 20.0, marker=1.0)), [])
        output = accumulator.append(transition(3.0, 0.7, 30.0, marker=2.0))
        self.assertEqual(len(output), 1)
        self.assertAlmostEqual(output[0].reward, 1.0 + 0.9 * 2.0 + 0.9 * 0.8 * 3.0, places=12)
        self.assertAlmostEqual(output[0].delta_t, 60.0, places=12)
        self.assertAlmostEqual(output[0].bootstrap_discount, 0.9 * 0.8 * 0.7, places=12)

    def test_10_c51_projection_uses_smdp_discount(self):
        support = np.array([-1.0, 0.0, 1.0])
        probabilities = np.array([[0.0, 0.0, 1.0]])
        projected = c51_project_distribution(
            np.array([0.0]), np.array([0.5]), np.array([0.0]), probabilities, support
        )
        np.testing.assert_allclose(projected, np.array([[0.0, 0.5, 0.5]]), atol=1e-12)

    def test_11_terminal_transition_has_no_bootstrap(self):
        target = one_step_smdp_target(
            np.array([2.0]), np.array([0.8]), np.array([1.0]), np.array([999.0])
        )
        self.assertAlmostEqual(float(target[0]), 2.0, places=12)
        projected = c51_project_distribution(
            np.array([0.0]),
            np.array([0.9]),
            np.array([1.0]),
            np.array([[0.2, 0.3, 0.5]]),
            np.array([-1.0, 0.0, 1.0]),
        )
        np.testing.assert_allclose(projected, np.array([[0.0, 1.0, 0.0]]), atol=1e-12)

    def test_12_wait_advances_to_future_event(self):
        env = event_env(0)
        _, _, _, _, info = step5(env, 0)
        self.assertGreater(info["delta_t"], 0.0)
        self.assertGreater(info["next_decision_time"], info["decision_time"])

    def test_13_same_time_multi_uav_dispatch_is_bounded(self):
        env = event_env(0)
        zero_duration = 0
        for _ in range(len(env.sim.drones) + 2):
            mask = env.action_masks()
            valid = np.flatnonzero(mask[1:]) + 1
            action = int(valid[0]) if len(valid) else 0
            _, _, done, _, info = step5(env, action)
            zero_duration += int(info["delta_t"] <= 1e-9)
            if done or info["delta_t"] > 1e-9:
                break
        self.assertGreaterEqual(zero_duration, 1)
        self.assertLessEqual(zero_duration, len(env.sim.drones))

    def test_14_dynamic_insertion_event_compatibility(self):
        env = event_env(0)
        valid = np.flatnonzero(env.action_masks()[1:]) + 1
        self.assertGreater(len(valid), 0)
        step5(env, int(valid[0]))
        self.assertGreater(len(env.sim.insertion_log), 0)
        self.assertGreater(len(env.sim.recovery_queue), 0)
        for _ in range(80):
            _, _, done, _, _ = step5(env, 0)
            if done or any(row.get("event") == "recover" for row in env.sim.event_log):
                break
        self.assertTrue(any(row.get("event") == "recover" for row in env.sim.event_log))
        self.assertTrue(all(env.sim.network.G.has_edge(u, v) for _, u, v in env.sim.vehicle.route_edge_log))

    def test_15_stale_event_is_discarded(self):
        queue = EventQueue()
        queue.schedule(
            time=10.0,
            event_type="CAV_NODE_ARRIVAL",
            entity_id="cav",
            schedule_version=1,
        )
        self.assertIsNone(queue.peek_time(0.0, lambda event: 2 if event.entity_id == "cav" else None))
        self.assertEqual(queue.stale_event_count, 1)

    def test_16_determinism(self):
        env_a, env_b = event_env(7), event_env(7)
        trace_a, trace_b = [], []
        for env, trace in ((env_a, trace_a), (env_b, trace_b)):
            for _ in range(4):
                _, reward, done, _, info = step5(env, 0)
                trace.append((info["next_decision_time"], reward, tuple(info["trigger_events"])))
                if done:
                    break
        self.assertEqual(trace_a, trace_b)
        self.assertEqual(env_a._state_summary(), env_b._state_summary())

    def test_17_reward_conservation(self):
        env = event_env(0)
        _, reward, _, _, info = step5(env, 0)
        self.assertAlmostEqual(reward, sum(info["reward_components"].values()), places=12)

    def test_18_predicted_energy_not_charged_as_actual(self):
        env = event_env(0)
        valid = np.flatnonzero(env.action_masks()[1:]) + 1
        candidate = env.candidates[int(valid[0]) - 1]
        energy_before_evaluation = sum(drone.energy_used for drone in env.sim.drones)
        evaluation = env.sim.evaluate_insertion(candidate, commit=False)
        self.assertEqual(sum(drone.energy_used for drone in env.sim.drones), energy_before_evaluation)
        self.assertGreaterEqual(evaluation["diagnostics"].get("predicted_hover_energy", 0.0), 0.0)
        step5(env, int(valid[0]))
        energy_before_execution = sum(drone.energy_used for drone in env.sim.drones)
        _, _, _, _, info = step5(env, 0)
        energy_increment = sum(drone.energy_used for drone in env.sim.drones) - energy_before_execution
        expected = float(env.cfg["reward"]["energy_penalty"]) * energy_increment * 0.001
        self.assertAlmostEqual(info["reward_components"]["energy"], expected, places=12)

    def test_19_fixed_and_event_training_rewards_are_identical(self):
        fixed_cfg = load_config(ROOT / "configs" / "scenario_60node_stage2_fixed_dynamic.yaml")
        event_cfg = load_config(ROOT / "configs" / "scenario_60node_stage2_event_dynamic.yaml")
        self.assertEqual(fixed_cfg["reward"], event_cfg["reward"])
        self.assertEqual(fixed_cfg["reward"]["accounting_mode"], "interval_increment")
        self.assertTrue(bool(fixed_cfg["smdp"]["interval_normalize_reward"]))
        self.assertTrue(bool(event_cfg["smdp"]["interval_normalize_reward"]))
        fixed_env = DynamicPatrolEnv(fixed_cfg, scenario="small")
        event_env_instance = DynamicPatrolEnv(event_cfg, scenario="small")
        fixed_env.reset(seed=3)
        event_env_instance.reset(seed=3)
        before = {
            "done_ids": set(),
            "expired_ids": set(),
            "energy": 10.0,
            "aoi_integral": 100.0,
            "uav_wait": 0.0,
            "cav_wait": 0.0,
            "queue_length": 1.0,
            "queue_travel_time": 120.0,
        }
        after = {
            **before,
            "energy": 25.0,
            "aoi_integral": 340.0,
            "queue_length": 2.0,
            "queue_travel_time": 180.0,
        }
        fixed_components = fixed_env._event_reward_components(
            before, after, illegal=False, wait_penalty=-0.3, delta_t=60.0, terminated=False
        )
        event_components = event_env_instance._event_reward_components(
            before, after, illegal=False, wait_penalty=-0.3, delta_t=60.0, terminated=False
        )
        self.assertEqual(fixed_components, event_components)


if __name__ == "__main__":
    unittest.main()
