from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dynamic_patrol_marl.env.simulator import Simulator
from dynamic_patrol_marl.planner.joint_plan import (
    PLAN_SAFE, PLAN_TYPE_I, PLAN_TYPE_II, PLAN_TYPE_III,
)
from dynamic_patrol_marl.rl.dqn_networks import make_q_network
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.rl.replay_buffer import MaskedReplayBuffer
from dynamic_patrol_marl.rl.smdp import transition_discount
from dynamic_patrol_marl.utils.config import load_config


CONFIG = "configs/scenario_60node_stage4_joint_full.yaml"


def make_env(seed: int = 0, config: str = CONFIG) -> DynamicPatrolEnv:
    env = DynamicPatrolEnv(load_config(config))
    env.reset(seed=seed)
    return env


def plans(env: DynamicPatrolEnv):
    return [plan for plan in env.joint_plan_cache.plan_slots if plan is not None]


def find_plan(env: DynamicPatrolEnv, predicate, max_decisions: int = 100):
    """Follow the deterministic heuristic until a requested joint-plan type appears."""
    for _ in range(max_decisions):
        matching = [plan for plan in plans(env) if predicate(plan)]
        if matching:
            return matching[0]
        action = env.sim.joint_plan_generator.select_heuristic(env.joint_plan_cache)
        out = env.step(action)
        done = bool(out[2] or out[3]) if len(out) == 5 else bool(out[2])
        if done:
            break
    raise AssertionError(f"requested plan did not appear within {max_decisions} decisions")


class Stage4JointPlanTests(unittest.TestCase):
    def test_01_legacy_sortie_only_regression_dimensions(self):
        env = DynamicPatrolEnv(load_config("configs/scenario_60node_stage2_event_dynamic.yaml"))
        obs = env.reset(seed=0)[0]
        self.assertEqual(env.decision_mode, "sortie_only")
        self.assertEqual((obs.shape[0], env.action_space.n), (219, 25))

    def test_02_safe_plan_is_slot_zero_and_valid(self):
        env = make_env()
        safe = env.joint_plan_cache.plan_slots[0]
        self.assertEqual(safe.plan_type, PLAN_SAFE)
        self.assertTrue(env.action_masks()[0])
        sim = env.sim
        before = (
            deepcopy(sim.vehicle.__dict__), deepcopy([d.__dict__ for d in sim.drones]),
            deepcopy([t.__dict__ for t in sim.tasks]), deepcopy(sim.recovery_queue),
            sim.recovery_schedule_version, deepcopy(sim.event_log), deepcopy(sim.sortie_log),
            deepcopy(sim.insertion_log), deepcopy(sim.ground_controller.action_log),
        )
        ok, committed = sim.joint_plan_generator.commit_joint_plan(env.joint_plan_cache, 0)
        self.assertTrue(ok)
        self.assertTrue(committed.is_safe)
        after = (
            sim.vehicle.__dict__, [d.__dict__ for d in sim.drones],
            [t.__dict__ for t in sim.tasks], sim.recovery_queue,
            sim.recovery_schedule_version, sim.event_log, sim.sortie_log,
            sim.insertion_log, sim.ground_controller.action_log,
        )
        self.assertEqual(before, after)

    def test_03_type_i_service_plan_exists(self):
        candidates = [p for p in plans(make_env()) if p.plan_type == PLAN_TYPE_I and p.sortie]
        self.assertTrue(candidates)
        self.assertTrue(all(p.ground_candidate.action_type == "keep_route" for p in candidates))
        self.assertTrue(all(p.insertion_index is not None for p in candidates))

    def test_04_type_ii_rendezvous_plan_exists(self):
        candidates = [p for p in plans(make_env()) if p.plan_type == PLAN_TYPE_II]
        self.assertTrue(any(p.sortie is None for p in candidates))
        self.assertTrue(any(p.sortie is not None for p in candidates))
        self.assertTrue(all(p.ground_candidate.action_type in {"wait", "move_to"} for p in candidates))

    def test_05_type_iii_reroute_plan_exists_when_queue_active(self):
        env = make_env()
        plan = find_plan(
            env,
            lambda p: p.plan_type == PLAN_TYPE_III and p.sortie is not None,
            max_decisions=100,
        )
        self.assertEqual(plan.ground_candidate.action_type, "reroute")
        self.assertIsNotNone(plan.insertion_index)

    def test_06_launch_is_from_current_cav_node(self):
        env = make_env()
        for plan in plans(env):
            if plan.sortie:
                self.assertEqual(plan.sortie.launch_node, env.sim.vehicle.position)

    def test_07_all_cav_route_arcs_exist(self):
        env = make_env()
        for plan in plans(env):
            for u, v in zip(plan.cav_route[:-1], plan.cav_route[1:]):
                self.assertTrue(env.sim.network.G.has_edge(u, v))
        env.sim.vehicle.at_intersection = False
        env.joint_plan_cache = None
        env._prepare_decision_epoch(["ARC_STATE_TEST"])
        self.assertTrue(all(p.ground_candidate.action_type == "keep_route" for p in plans(env)))
        self.assertTrue(all(p.sortie is None for p in plans(env)))

    def test_08_locked_prefix_is_never_crossed(self):
        env = make_env()
        source = next(p for p in plans(env) if p.sortie is not None)
        locked_entry = env.sim.recovery_scheduler.make_entry(source.sortie)
        locked_entry["locked"] = True
        env.sim.recovery_queue.append(locked_entry)
        env.sim.recovery_schedule_version += 1
        cache = env.sim.joint_plan_generator.generate_joint_plans({"decision_epoch_id": 8})
        retained = [p for p in cache.plan_slots if p is not None and p.insertion_index is not None]
        locked = env.sim.recovery_scheduler.locked_prefix_length(env.sim.recovery_queue)
        self.assertEqual(locked, 1)
        self.assertTrue(retained)
        self.assertTrue(all(p.insertion_index >= locked for p in retained))

    def test_09_ranked_insertions_respect_k(self):
        sim = Simulator(load_config(CONFIG), seed=0)
        candidates = sim.sortie_generator.generate_all(
            sim.idle_drones_on_vehicle(), sim.vehicle.position, sim.now,
            sim.task_pool.available_tasks(sim.now), allow_recovery_reschedule=True,
        )
        ranked = sim.recovery_scheduler.evaluate_insertions(candidates[0], max_results=2)
        self.assertLessEqual(len(ranked), 2)
        self.assertTrue(ranked and ranked[0]["feasible"])
        self.assertEqual(
            [row["insertion_cost"] for row in ranked],
            sorted(row["insertion_cost"] for row in ranked),
        )
        repeated = sim.recovery_scheduler.evaluate_insertions(candidates[0], max_results=2)
        self.assertEqual(
            [(row["best_index"], row["insertion_cost"]) for row in ranked],
            [(row["best_index"], row["insertion_cost"]) for row in repeated],
        )

    def test_10_battery_slack_is_nonnegative_for_retained_sorties(self):
        env = make_env()
        self.assertTrue(all(p.predicted_battery_slack >= -1e-9 for p in plans(env) if p.sortie))

    def test_11_return_guard_can_reject_plans(self):
        env = make_env()
        generator = env.sim.joint_plan_generator
        generator.return_cfg["max_return_distance_increase_ratio"] = -1.0
        cache = generator.generate_joint_plans({"decision_epoch_id": 99})
        self.assertGreater(generator.return_guard_rejections, 0)
        self.assertTrue(cache.action_mask[0])

    def test_12_return_guard_floor_prevents_division_by_zero(self):
        env = make_env()
        generator = env.sim.joint_plan_generator
        plan = next(p for p in plans(env) if p.sortie)
        generator.return_cfg["return_distance_floor_m"] = 100.0
        generator.return_cfg["max_return_distance_increase_ratio"] = 100.0
        self.assertTrue(generator._guard_allows(plan, {(plan.uav_id, tuple(plan.task_ids)): 0.0}))

    def test_13_duplicate_uav_recovery_is_filtered(self):
        env = make_env()
        plan = next(p for p in plans(env) if p.sortie)
        entry = env.sim.recovery_scheduler.make_entry(plan.sortie)
        env.sim.recovery_queue.append(entry)
        self.assertEqual(env.sim.recovery_scheduler.evaluate_insertions(plan.sortie), [])
        assigned_task = int(plan.task_ids[0])
        env.sim.task_pool.mark_assigned([assigned_task], "external_uav", env.sim.now)
        cache = env.sim.joint_plan_generator.generate_joint_plans({"decision_epoch_id": 13})
        for candidate in (p for p in cache.plan_slots if p is not None):
            self.assertEqual(len(candidate.task_ids), len(set(candidate.task_ids)))
            self.assertNotIn(assigned_task, candidate.task_ids)

    def test_14_candidate_caps_are_enforced(self):
        env = make_env()
        self.assertLessEqual(len(plans(env)), 25)
        self.assertEqual(len(env.joint_plan_cache.plan_slots), 25)

    def test_15_type_quotas_are_enforced_before_fill(self):
        env = make_env()
        counts = {ptype: sum(p.plan_type == ptype for p in plans(env)) for ptype in (PLAN_TYPE_I, PLAN_TYPE_II, PLAN_TYPE_III)}
        self.assertLessEqual(counts[PLAN_TYPE_I], 12)
        self.assertLessEqual(counts[PLAN_TYPE_II], 8)
        self.assertLessEqual(counts[PLAN_TYPE_III], 4)

    def test_16_unused_quota_slots_are_deterministically_filled_or_padded(self):
        env = make_env()
        generator = env.sim.joint_plan_generator
        generator.quotas = {PLAN_TYPE_I: 0, PLAN_TYPE_II: 0, PLAN_TYPE_III: 0}
        cache = generator.generate_joint_plans({"decision_epoch_id": 16})
        self.assertGreater(int(cache.action_mask.sum()), 1)
        first_padding = next((i for i, p in enumerate(cache.plan_slots) if p is None), len(cache.plan_slots))
        self.assertTrue(all(p is None for p in cache.plan_slots[first_padding:]))
        self.assertTrue(cache.action_mask[:first_padding].all())

    def test_17_padding_has_zero_features_and_false_mask(self):
        env = make_env()
        for index, plan in enumerate(env.joint_plan_cache.plan_slots):
            if plan is None:
                self.assertFalse(env.action_masks()[index])
                self.assertTrue(np.all(env.joint_plan_cache.feature_matrix[index] == 0.0))

    def test_18_generation_is_deterministic(self):
        a, b = make_env(), make_env()
        sig = lambda e: [(p.plan_type, p.uav_id, tuple(p.task_ids), p.cav_target_node) if p else None for p in e.joint_plan_cache.plan_slots]
        self.assertEqual(sig(a), sig(b))
        np.testing.assert_array_equal(a.joint_plan_cache.feature_matrix, b.joint_plan_cache.feature_matrix)

    def test_19_joint_observation_dimension(self):
        env = make_env()
        self.assertEqual((env.obs_dim, env.action_space.n), (1025, 25))

    def test_20_features_are_finite(self):
        env = make_env()
        self.assertTrue(np.isfinite(env.joint_plan_cache.feature_matrix).all())

    def test_21_action_mask_matches_slot_feasibility(self):
        env = make_env()
        for i, plan in enumerate(env.joint_plan_cache.plan_slots):
            self.assertEqual(bool(env.action_masks()[i]), bool(plan and plan.feasible and plan.mask))

    def test_22_observation_and_mask_reuse_one_cache(self):
        env = make_env()
        cache_id = id(env.joint_plan_cache)
        epoch_id = env.joint_plan_cache.decision_epoch_id
        env._observation(); env.action_masks(); env._observation()
        self.assertEqual(cache_id, id(env.joint_plan_cache))
        self.assertEqual(epoch_id, env.joint_plan_cache.decision_epoch_id)
        env.step(0)
        self.assertGreater(env.joint_plan_cache.decision_epoch_id, epoch_id)

    def test_23_stale_cache_is_rejected(self):
        env = make_env()
        cache = env.joint_plan_cache
        env.sim.now += 1.0
        ok, _ = env.sim.joint_plan_generator.commit_joint_plan(cache, 0)
        self.assertFalse(ok)
        self.assertEqual(env.sim.joint_plan_generator.stale_cache_count, 1)

    def test_24_generation_does_not_mutate_physical_state(self):
        env = make_env()
        sim = env.sim
        before = (
            sim.now, deepcopy(sim.vehicle.__dict__), deepcopy([d.__dict__ for d in sim.drones]),
            deepcopy([t.__dict__ for t in sim.tasks]), deepcopy(sim.recovery_queue),
            sim.recovery_schedule_version, deepcopy(sim.rng.bit_generator.state),
            deepcopy(sim.event_log), deepcopy(sim.sortie_log), deepcopy(sim.insertion_log),
            deepcopy(sim.ground_controller.action_log), deepcopy(sim.ground_controller.route_change_log),
            sim.illegal_actions, sim.reward_total, tuple(env.event_queue.snapshot()), id(env.joint_plan_cache),
        )
        env.sim.joint_plan_generator.generate_joint_plans({"decision_epoch_id": 7})
        after = (
            sim.now, sim.vehicle.__dict__, [d.__dict__ for d in sim.drones],
            [t.__dict__ for t in sim.tasks], sim.recovery_queue,
            sim.recovery_schedule_version, sim.rng.bit_generator.state,
            sim.event_log, sim.sortie_log, sim.insertion_log,
            sim.ground_controller.action_log, sim.ground_controller.route_change_log,
            sim.illegal_actions, sim.reward_total, tuple(env.event_queue.snapshot()), id(env.joint_plan_cache),
        )
        self.assertEqual(before, after)

    def test_25_atomic_commit_success(self):
        env = make_env()
        selected = next(p for p in plans(env) if p.plan_type == PLAN_TYPE_II and p.sortie)
        action = selected.slot_index
        before = (
            len(env.sim.recovery_queue), len(env.sim.sortie_log), len(env.sim.insertion_log),
            len(env.sim.event_log), len(env.sim.ground_controller.route_change_log),
        )
        ok, plan = env.sim.joint_plan_generator.commit_joint_plan(env.joint_plan_cache, action)
        self.assertTrue(ok)
        self.assertEqual(env.sim.get_drone(plan.uav_id).status, "flying")
        self.assertEqual(
            (
                len(env.sim.recovery_queue), len(env.sim.sortie_log), len(env.sim.insertion_log),
                len(env.sim.event_log), len(env.sim.ground_controller.route_change_log),
            ),
            (before[0] + 1, before[1] + 1, before[2] + 1, before[3] + 3, before[4] + 1),
        )
        self.assertEqual(env.sim.event_log[-3]["event"], "cav_ground_action")
        self.assertEqual(
            env.sim.recovery_queue[plan.insertion_index]["sortie_id"],
            plan.insertion_evaluation["entry"]["sortie_id"],
        )
        self.assertTrue(all(env.sim.task_pool.by_id[tid].status == "assigned" for tid in plan.task_ids))

    def test_26_atomic_rollback_on_injected_failure(self):
        env = make_env()
        generator = env.sim.joint_plan_generator
        generator.cfg["inject_commit_failure_at"] = "after_ground"
        sim = env.sim
        before = (
            deepcopy(sim.vehicle.__dict__), deepcopy([d.__dict__ for d in sim.drones]),
            deepcopy(sim.recovery_queue), deepcopy([t.__dict__ for t in sim.tasks]),
            sim.recovery_schedule_version, deepcopy(sim.rng.bit_generator.state),
            deepcopy(sim.event_log), deepcopy(sim.sortie_log), deepcopy(sim.insertion_log),
            tuple(env.event_queue.snapshot()),
        )
        action = next(p.slot_index for p in plans(env) if p.plan_type == PLAN_TYPE_II)
        ok, _ = generator.commit_joint_plan(env.joint_plan_cache, action)
        self.assertFalse(ok)
        after = (
            sim.vehicle.__dict__, [d.__dict__ for d in sim.drones], sim.recovery_queue,
            [t.__dict__ for t in sim.tasks], sim.recovery_schedule_version,
            sim.rng.bit_generator.state, sim.event_log, sim.sortie_log, sim.insertion_log,
            tuple(env.event_queue.snapshot()),
        )
        self.assertEqual(before, after)
        self.assertEqual(generator.rollback_count, 1)

    def test_27_route_commit_updates_schedule_version(self):
        env = make_env(); env.step(1); env.step(0)
        plan = next(p for p in plans(env) if p.ground_candidate.action_type in {"move_to", "reroute"})
        before = env.sim.vehicle.schedule_version
        ok, _ = env.sim.joint_plan_generator.commit_joint_plan(env.joint_plan_cache, plan.slot_index)
        self.assertTrue(ok)
        self.assertGreater(env.sim.vehicle.schedule_version, before)

    def test_28_same_timestamp_ground_is_keep_only(self):
        env = make_env()
        env._ground_action_committed_time = env.sim.now
        env.joint_plan_cache = None
        env._prepare_decision_epoch(["UAV_LAUNCH"])
        self.assertTrue(all(p.ground_candidate.action_type == "keep_route" for p in plans(env)))

    def test_29_safe_plan_advances_to_future_event(self):
        env = make_env()
        out = env.step(0)
        info = out[-1]
        self.assertGreater(info["delta_t"], 0.0)

    def test_30_replay_constructor_accepts_joint_dimensions(self):
        import torch
        replay = MaskedReplayBuffer(
            capacity=8, obs_dim=1025, action_dim=25, seed=0,
            time_mode="event_driven", require_time_fields=True,
        )
        obs = np.zeros(1025, dtype=np.float32)
        next_obs = np.ones(1025, dtype=np.float32)
        mask = np.zeros(25, dtype=bool); mask[0] = True
        replay.add(
            obs, 0, 1.5, next_obs, False, mask, mask,
            delta_t=60.0, transition_discount=0.99, decision_time=12.0,
            trigger_event_type="TEST",
        )
        batch = replay.sample(1, torch.device("cpu"))
        self.assertEqual((len(replay), tuple(batch.obs.shape), tuple(batch.action_masks.shape)), (1, (1, 1025), (1, 25)))
        self.assertAlmostEqual(float(batch.bootstrap_discounts.item()), 0.99, places=6)

    def test_31_smdp_discount_is_unchanged(self):
        import torch
        from dynamic_patrol_marl.rl.dqn_networks import masked_q_values
        self.assertAlmostEqual(transition_discount(0.99, 60.0, 60.0, "continuous_time"), 0.99)
        next_q = torch.tensor([[100.0, 4.0]])
        next_mask = torch.tensor([[False, True]])
        target = 2.0 + 0.9 * float(masked_q_values(next_q, next_mask).max().item())
        self.assertAlmostEqual(target, 5.6)

    def test_32_network_forward_supports_joint_shape(self):
        import torch
        from dynamic_patrol_marl.experiments.evaluate_masked_rl_models import (
            ModelSpec, _evaluate_spec, _load_masked_dqn,
        )
        net = make_q_network("dqn", 1025, 25, candidate_dim=38)
        q = net(torch.zeros((2, 1025), dtype=torch.float32))
        self.assertEqual(tuple(q.shape), (2, 25))
        with tempfile.TemporaryDirectory() as temp_dir:
            new_path = Path(temp_dir) / "joint.pt"
            old_path = Path(temp_dir) / "legacy.pt"
            torch.save({
                "model_state_dict": net.state_dict(), "algo": "dqn",
                "obs_dim": 1025, "action_dim": 25, "candidate_dim": 38,
                "decision_mode": "joint_plan", "feature_schema": {"feature_dim": 38},
                "replay_config": {"schema_version": 2}, "action_masking": True,
            }, new_path)
            loaded = _load_masked_dqn(new_path, "dqn", "cpu", True)
            self.assertEqual(
                (loaded["obs_dim"], loaded["action_dim"], loaded["candidate_dim"], loaded["decision_mode"]),
                (1025, 25, 38, "joint_plan"),
            )
            old_net = make_q_network("dqn", 219, 25, candidate_dim=6)
            torch.save({
                "model_state_dict": old_net.state_dict(), "algo": "dqn",
                "obs_dim": 219, "action_dim": 25, "candidate_dim": 6,
                "replay_config": {"schema_version": 2},
            }, old_path)
            spec = ModelSpec("legacy", "dqn", old_path, algo="dqn")
            with self.assertRaisesRegex(ValueError, "legacy sortie-only checkpoints"):
                _evaluate_spec(load_config(CONFIG), spec, [0], "case", "cpu")

    def test_33_heuristic_selector_is_deterministic(self):
        env = make_env()
        a = env.sim.joint_plan_generator.select_heuristic(env.joint_plan_cache)
        b = env.sim.joint_plan_generator.select_heuristic(env.joint_plan_cache)
        self.assertEqual(a, b)

    def test_34_generation_timing_is_recorded(self):
        env = make_env()
        self.assertGreaterEqual(env.joint_plan_cache.generation_time_ms, 0.0)
        self.assertTrue(env.sim.joint_plan_generator.generation_times_ms)
        self.assertTrue(env.sim.joint_plan_generator.evaluation_times_ms)
        self.assertTrue(all(value >= 0.0 for value in env.sim.joint_plan_generator.evaluation_times_ms))

    def test_35_stage4_metrics_are_exported(self):
        env = make_env()
        plan = next(p for p in plans(env) if p.sortie is not None)
        ok, committed = env.sim.joint_plan_generator.commit_joint_plan(env.joint_plan_cache, plan.slot_index)
        self.assertTrue(ok)
        metrics = env.sim.metrics("test").to_dict()
        for key in ("mean_valid_plans", "return_guard_rejection_rate", "commit_failure_count", "mean_plan_generation_time_ms"):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["num_sorties"], len(env.sim.sortie_log))
        self.assertEqual(metrics["recovery_attempts"], len(env.sim.insertion_log))
        self.assertEqual(metrics["num_sorties"], 1)
        self.assertEqual(metrics["avg_tasks_per_sortie"], len(committed.task_ids))
        self.assertEqual(len(env.sim.recovery_queue), 1)

    def test_36_separate_control_preserves_25_actions(self):
        cases = (
            ("configs/scenario_60node_stage3_c0_passive.yaml", "sortie_only"),
            ("configs/scenario_60node_stage3_c3_reroute.yaml", "separate_control"),
        )
        for config, mode in cases:
            env = DynamicPatrolEnv(load_config(config))
            env.reset(seed=0)
            self.assertEqual((env.decision_mode, env.action_space.n), (mode, 25))

    def test_37_joint_mode_disables_independent_ground_decide(self):
        env = make_env()
        self.assertFalse(env.sim.ground_controller.active)
        self.assertTrue(env.sim.ground_controller.candidate_generation_enabled)

    def test_38_fixed_interval_joint_safe_step_runs(self):
        cfg = load_config(CONFIG)
        cfg["time"]["mode"] = "fixed_interval"
        env = DynamicPatrolEnv(cfg); env.reset(seed=0)
        out = env.step(0)
        self.assertGreater(out[-1]["delta_t"], 0.0)

    def test_39_feature_schema_has_fixed_order(self):
        schema = make_env().sim.joint_plan_generator.feature_builder.schema()
        self.assertEqual(schema["feature_dim"], 38)
        self.assertEqual(len(schema["feature_names"]), 38)
        published = json.loads((Path(__file__).parents[1] / "joint_plan_feature_schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema, published)

    def test_40_k_configs_define_expected_action_dimensions(self):
        for k in (8, 16, 24, 32):
            env = make_env(config=f"configs/scenario_60node_stage4_joint_k{k}.yaml")
            self.assertEqual(env.action_space.n, k + 1)
            self.assertEqual(env.obs_dim, env.global_obs_dim + (k + 1) * 38)


if __name__ == "__main__":
    unittest.main()
