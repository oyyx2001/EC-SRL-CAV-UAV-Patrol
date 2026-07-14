from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_patrol_marl.env.simulator import Simulator
from dynamic_patrol_marl.planner.greedy_baseline import get_policy
from dynamic_patrol_marl.planner.sortie_generator import Sortie
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.utils.config import load_config


class RecoveryInsertionTests(unittest.TestCase):
    def setUp(self):
        cfg = load_config(ROOT / "configs" / "scenario_60node_stage1_dynamic.yaml")
        self.cfg = deepcopy(cfg)
        self.cfg["recovery_insertion"].update(
            {
                "weight_uav_wait": 1.0,
                "weight_existing_delay": 0.5,
                "weight_cav_wait": 0.0,
                "weight_route_time": 0.0,
                "battery_reserve": 0.0,
            }
        )
        self.cfg["agents"]["drone_hover_power"] = 1.0
        self.sim = Simulator(self.cfg, seed=123, scenario="small")
        self.node = int(self.sim.vehicle.position)

    def entry(self, drone_id, eta, battery=10000.0, locked=False, sortie_id=None):
        return {
            "uav_id": str(drone_id),
            "drone_id": str(drone_id),
            "recovery_node": self.node,
            "eta": float(eta),
            "recovery_duration": 0.0,
            "battery_at_eta": float(battery),
            "battery_reserve": 0.0,
            "hover_power": 1.0,
            "sortie_id": sortie_id or f"old-{drone_id}",
            "status": "planned",
            "locked": bool(locked),
            "original_queue_index": 0,
            "scheduled_start_time": None,
            "scheduled_finish_time": None,
            "predicted_uav_wait": None,
            "predicted_cav_wait": None,
            "launch_time": 0.0,
            "task_ids": [],
            "recovery_arrival_time": None,
        }

    def sortie(self, eta, battery=10000.0):
        return Sortie(
            drone_id="drone0",
            launch_node=self.node,
            task_ids=[],
            task_sequence=[],
            recovery_node=self.node,
            drone_time_s=float(eta),
            vehicle_time_s=0.0,
            finish_time_s=float(eta),
            energy=0.0,
            score=0.0,
            feasible=True,
            segments=[],
            tasks_per_sortie=0,
            vehicle_anchor_node=self.node,
            vehicle_queue_time_s=0.0,
            flight_service_energy=0.0,
            battery_at_eta=float(battery),
            base_feasible=True,
        )

    def evaluate(self, sortie):
        return self.sim.recovery_scheduler.evaluate(sortie, rule="dynamic_insertion")

    def test_02_tail_is_optimal_without_contention(self):
        self.sim.recovery_queue = [self.entry("drone1", 0.0), self.entry("drone2", 0.0)]
        result = self.evaluate(self.sortie(1000.0))
        self.assertTrue(result["feasible"])
        self.assertEqual(result["best_index"], 2)

    def test_03_middle_insertion_is_optimal(self):
        self.sim.recovery_queue = [self.entry("drone1", 0.0), self.entry("drone2", 1000.0)]
        result = self.evaluate(self.sortie(100.0))
        self.assertTrue(result["feasible"])
        self.assertEqual(result["best_index"], 1)
        self.assertGreater(result["diagnostics"]["counterfactual_gain"], 0.0)

    def test_04_head_insertion_for_low_battery_uav(self):
        self.sim.recovery_queue = [self.entry("drone1", 1000.0)]
        result = self.evaluate(self.sortie(0.0, battery=1.0))
        self.assertTrue(result["feasible"])
        self.assertEqual(result["best_index"], 0)

    def test_05_locked_prefix_cannot_be_crossed(self):
        self.sim.recovery_queue = [self.entry("drone1", 1000.0, locked=True)]
        result = self.evaluate(self.sortie(0.0, battery=2000.0))
        self.assertTrue(result["feasible"])
        self.assertEqual(result["diagnostics"]["locked_prefix_length"], 1)
        self.assertGreaterEqual(result["best_index"], 1)

    def test_06_battery_infeasible_position_is_filtered(self):
        self.sim.recovery_queue = [self.entry("drone1", 100.0)]
        result = self.evaluate(self.sortie(0.0, battery=10.0))
        positions = {row["index"]: row for row in result["diagnostics"]["position_results"]}
        self.assertTrue(positions[0]["feasible"])
        self.assertFalse(positions[1]["feasible"])
        self.assertEqual(positions[1]["reason"], "battery_reserve")

    def test_07_all_positions_infeasible_masks_action_without_mutation(self):
        before = deepcopy(self.sim.recovery_queue)
        candidate = self.sortie(0.0, battery=-1.0)
        result = self.evaluate(candidate)
        candidate.feasible = bool(result["feasible"])
        env = DynamicPatrolEnv(self.cfg)
        env.sim = self.sim
        env.candidates = [candidate]
        self.assertFalse(result["feasible"])
        self.assertFalse(bool(env.action_masks()[1]))
        self.assertEqual(self.sim.recovery_queue, before)

    def test_08_evaluation_has_no_side_effects(self):
        self.sim.recovery_queue = [self.entry("drone1", 0.0), self.entry("drone2", 1000.0)]
        queue_before = deepcopy(self.sim.recovery_queue)
        vehicle_before = deepcopy(self.sim.vehicle.__dict__)
        drones_before = deepcopy([d.__dict__ for d in self.sim.drones])
        self.evaluate(self.sortie(100.0))
        self.assertEqual(self.sim.recovery_queue, queue_before)
        self.assertEqual(self.sim.vehicle.__dict__, vehicle_before)
        self.assertEqual([d.__dict__ for d in self.sim.drones], drones_before)

    def test_09_evaluation_is_deterministic(self):
        self.sim.recovery_queue = [self.entry("drone1", 0.0), self.entry("drone2", 1000.0)]
        first = self.evaluate(self.sortie(100.0))
        second = self.evaluate(self.sortie(100.0))
        self.assertEqual(first["best_index"], second["best_index"])
        self.assertEqual(first["insertion_cost"], second["insertion_cost"])
        self.assertEqual(first["predicted_schedule"], second["predicted_schedule"])

    def test_10_predicted_hover_energy_is_not_deducted(self):
        self.sim.recovery_queue = [self.entry("drone1", 100.0, locked=True)]
        candidate = self.sortie(0.0, battery=200.0)
        drone = self.sim.get_drone("drone0")
        battery_before = drone.battery
        result = self.evaluate(candidate)
        self.assertGreater(result["diagnostics"]["predicted_hover_energy"], 0.0)
        self.assertEqual(drone.battery, battery_before)
        self.sim._consume(drone, 1.0, "hover")
        self.assertAlmostEqual(drone.battery, battery_before - 1.0, places=12)


class FifoGoldenRegressionTest(unittest.TestCase):
    def test_01_fifo_matches_pre_change_golden_episode(self):
        cfg = load_config(ROOT / "configs" / "scenario_60node_stage1_fifo.yaml")
        sim = Simulator(cfg, seed=0, scenario="small")
        metrics = sim.run(get_policy("score_based"), method_name="score_based")
        self.assertEqual(metrics.completed_tasks, 20)
        self.assertEqual(metrics.expired_tasks, 4)
        self.assertEqual(metrics.num_sorties, 9)
        self.assertAlmostEqual(metrics.completion_rate, 0.6451612903225806, places=12)
        self.assertAlmostEqual(metrics.avg_response_time_s, 638.8317880031684, places=9)
        self.assertAlmostEqual(metrics.drone_wait_time_s, 400.0, places=12)
        self.assertAlmostEqual(metrics.vehicle_wait_time_s, 300.0, places=12)
        self.assertAlmostEqual(metrics.total_energy, 3680.0885128707264, places=9)
        self.assertAlmostEqual(metrics.weighted_aoi, 615.6631987447494, places=9)
        append_events = [
            (event.get("time"), event.get("agent"), event.get("node"), event.get("queue_position"))
            for event in sim.event_log
            if event.get("event") == "recovery_queue_append"
        ]
        self.assertEqual(
            append_events,
            [
                (0.0, "drone0", 4, 0),
                (180.0, "drone0", 46, 0),
                (600.0, "drone0", 6, 0),
                (1140.0, "drone1", 19, 1),
                (1200.0, "drone2", 19, 2),
                (1740.0, "drone0", 8, 0),
                (2220.0, "drone1", 45, 1),
                (2880.0, "drone0", 40, 0),
                (3240.0, "drone0", 44, 0),
            ],
        )


if __name__ == "__main__":
    unittest.main()
