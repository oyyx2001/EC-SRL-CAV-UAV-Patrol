from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from dynamic_patrol_marl.planner.joint_plan import PLAN_SAFE, PLAN_TYPE_I, PLAN_TYPE_III
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.utils.config import load_config


def env_for(seed=0):
    env = DynamicPatrolEnv(load_config("configs/scenario_60node_stage4_joint_full.yaml"))
    env.reset(seed=seed)
    return env


class Stage4MicroScenarios(unittest.TestCase):
    def test_micro_01_no_available_task_safe_only(self):
        env = env_for()
        for task in env.sim.tasks:
            if task.status == "unassigned":
                task.status = "unreleased"
                task.release_time = env.sim.now + 100.0
        env.sim.ground_controller.action_set = ("keep_route",)
        env.joint_plan_cache = None
        env._prepare_decision_epoch(["MICRO_NO_TASK"])
        self.assertEqual(int(env.action_masks().sum()), 1)
        self.assertEqual(env.joint_plan_cache.plan_slots[0].plan_type, PLAN_SAFE)

    def test_micro_02_single_task_service_launch(self):
        env = env_for()
        plan = next(p for p in env.joint_plan_cache.plan_slots if p and p.plan_type == PLAN_TYPE_I)
        out = env.step(plan.slot_index)
        self.assertEqual(out[-1]["selected_sortie_id"].split(":")[0], plan.uav_id)

    def test_micro_03_active_queue_creates_reroute(self):
        env = env_for()
        env.step(1)
        env.step(0)
        self.assertTrue(any(p and p.plan_type == PLAN_TYPE_III for p in env.joint_plan_cache.plan_slots))

    def test_micro_04_extreme_return_is_guarded(self):
        env = env_for()
        generator = env.sim.joint_plan_generator
        generator.return_cfg["max_return_distance_increase_ratio"] = -0.5
        generator.generate_joint_plans({"decision_epoch_id": 4})
        self.assertGreater(generator.return_guard_rejections, 0)

    def test_micro_05_low_battery_removes_sorties(self):
        env = env_for()
        for drone in env.sim.drones:
            drone.battery = 0.0
        env.joint_plan_cache = None
        env._prepare_decision_epoch(["MICRO_LOW_BATTERY"])
        self.assertTrue(all(not p.sortie for p in env.joint_plan_cache.plan_slots if p))

    def test_micro_06_locked_recovery_prefix_preserved(self):
        env = env_for()
        plan = next(p for p in env.joint_plan_cache.plan_slots if p and p.sortie)
        entry = env.sim.recovery_scheduler.make_entry(plan.sortie)
        entry["locked"] = True
        entry["status"] = "returning"
        env.sim.recovery_queue.append(entry)
        env.joint_plan_cache = None
        env._prepare_decision_epoch(["MICRO_LOCKED"])
        for candidate in env.joint_plan_cache.plan_slots:
            if candidate and candidate.insertion_index is not None:
                self.assertGreaterEqual(candidate.insertion_index, 1)

    def test_micro_07_sparse_candidates_are_zero_padded(self):
        env = env_for()
        padding = [i for i, p in enumerate(env.joint_plan_cache.plan_slots) if p is None]
        self.assertTrue(padding)
        self.assertTrue(all(np.all(env.joint_plan_cache.feature_matrix[i] == 0) for i in padding))

    def test_micro_08_same_time_route_cannot_change_twice(self):
        env = env_for()
        env._ground_action_committed_time = env.sim.now
        env.joint_plan_cache = None
        env._prepare_decision_epoch(["MICRO_SAME_TIME"])
        actions = {p.ground_candidate.action_type for p in env.joint_plan_cache.plan_slots if p}
        self.assertEqual(actions, {"keep_route"})

    def test_micro_09_safe_plan_forces_positive_delta_t(self):
        env = env_for()
        info = env.step(0)[-1]
        self.assertGreater(info["delta_t"], 0.0)

    def test_micro_10_atomic_failure_restores_rng_and_tasks(self):
        env = env_for()
        generator = env.sim.joint_plan_generator
        generator.cfg["inject_commit_failure"] = True
        rng = deepcopy(env.sim.rng.bit_generator.state)
        tasks = deepcopy([task.__dict__ for task in env.sim.tasks])
        action = next(p.slot_index for p in env.joint_plan_cache.plan_slots if p and not p.is_safe)
        ok, _ = generator.commit_joint_plan(env.joint_plan_cache, action)
        self.assertFalse(ok)
        self.assertEqual(rng, env.sim.rng.bit_generator.state)
        self.assertEqual(tasks, [task.__dict__ for task in env.sim.tasks])


if __name__ == "__main__":
    unittest.main()
