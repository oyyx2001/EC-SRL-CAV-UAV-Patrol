from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import unittest

from dynamic_patrol_marl.env.cav_controller import GroundCandidate
from dynamic_patrol_marl.env.events import EventQueue
from dynamic_patrol_marl.env.simulator import Simulator
from dynamic_patrol_marl.experiments.run_stage2_event_smdp import run_episode
from dynamic_patrol_marl.planner.sortie_generator import Sortie
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.rl.smdp import transition_discount
from dynamic_patrol_marl.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def config(name: str = "scenario_60node_stage3_c3_reroute.yaml"):
    cfg = load_config(ROOT / "configs" / name)
    cfg["cav_control"]["minimum_recovery_slack_s"] = 0.0
    cfg["cav_control"]["maximum_uav_wait_s"] = 10000.0
    cfg["cav_control"]["max_detour_time_s"] = 10000.0
    cfg["cav_control"]["max_detour_distance_m"] = 100000.0
    cfg["cav_control"]["max_detour_ratio"] = 100.0
    cfg["cav_control"]["min_relative_improvement"] = 0.0
    cfg["ground_cost"]["weight_route_switch"] = 0.0
    return cfg


def farthest_node(sim: Simulator, origin: int | None = None) -> int:
    origin = int(sim.vehicle.position if origin is None else origin)
    return int(
        max(
            sim.network.nodes,
            key=lambda node: (sim.network.get_vehicle_time(origin, int(node)), int(node)),
        )
    )


def adjacent_node(sim: Simulator, origin: int | None = None) -> int:
    origin = int(sim.vehicle.position if origin is None else origin)
    return int(sorted(v for _, v in sim.network.G.out_edges(origin))[0])


def install_flexible_return(
    sim: Simulator,
    recovery_node: int,
    *,
    origin: int | None = None,
    battery: float = 1500.0,
    service_s: float = 20.0,
    locked: bool = False,
    sortie_id: str = "stage3-test-sortie",
):
    drone = sim.drones[0]
    origin = int(adjacent_node(sim) if origin is None else origin)
    return_s = float(sim.network.get_drone_time(origin, int(recovery_node)))
    fly_power = float(sim.cfg["agents"].get("drone_fly_power", 1.0))
    service_power = float(sim.cfg["agents"].get("drone_service_power", 0.85))
    energy = service_s * service_power + return_s * fly_power
    sortie = Sortie(
        drone_id=drone.id,
        launch_node=int(sim.vehicle.position),
        task_ids=[999],
        task_sequence=[999],
        recovery_node=int(recovery_node),
        drone_time_s=service_s + return_s,
        vehicle_time_s=sim.network.get_vehicle_time(sim.vehicle.position, recovery_node),
        finish_time_s=sim.now + service_s + return_s,
        energy=energy,
        score=0.0,
        feasible=True,
        segments=[
            {"type": "service", "from": origin, "to": origin, "duration": service_s, "task_id": 999},
            {"type": "fly", "from": origin, "to": int(recovery_node), "duration": return_s},
        ],
        flight_service_energy=energy,
        battery_at_eta=battery - energy,
    )
    drone.status = "flying"
    drone.position = origin
    drone.battery = float(battery)
    drone.sortie = sortie
    drone.segment_index = 0
    drone.segment_remaining = float(service_s)
    entry = {
        "uav_id": drone.id,
        "drone_id": drone.id,
        "recovery_node": int(recovery_node),
        "eta": float(sim.now + service_s + return_s),
        "recovery_duration": 0.0,
        "battery_at_eta": float(battery - energy),
        "battery_reserve": 0.0,
        "hover_power": float(sim.cfg["agents"].get("drone_hover_power", 0.65)),
        "sortie_id": sortie_id,
        "status": "planned",
        "locked": bool(locked),
        "scheduled_start_time": None,
        "scheduled_finish_time": None,
        "predicted_uav_wait": None,
        "predicted_cav_wait": None,
        "task_ids": [999],
    }
    sim.recovery_queue[:] = [entry]
    sim.recovery_schedule_version += 1
    return drone, entry


class Stage3CAVActiveRoutingTests(unittest.TestCase):
    def test_01_passive_regression_matches_stage2(self):
        stage2, _, _ = run_episode(
            load_config(ROOT / "configs" / "scenario_60node_stage2_event_dynamic.yaml"),
            0,
            "stage2",
        )
        c0, _, _ = run_episode(
            load_config(ROOT / "configs" / "scenario_60node_stage3_c0_passive.yaml"),
            0,
            "c0",
        )
        for key in (
            "completed_tasks",
            "expired_tasks",
            "completion_rate",
            "weighted_aoi",
            "total_energy",
            "mean_uav_recovery_wait_s",
            "cav_travel_distance_m",
            "decisions_per_episode",
            "mean_delta_t",
        ):
            self.assertEqual(stage2[key], c0[key], key)

    def test_02_keep_route_consistency(self):
        sim = Simulator(config("scenario_60node_stage3_c0_passive.yaml"), seed=2)
        target = adjacent_node(sim)
        sim.vehicle.set_target(target, sim.network, sim.now)
        before = deepcopy(sim.vehicle.__dict__), deepcopy(sim.recovery_queue)
        selected = sim.ground_controller.decide(["CAV_NODE_ARRIVAL"], force=True)
        self.assertEqual(selected.action_type, "keep_route")
        self.assertEqual(before, (sim.vehicle.__dict__, sim.recovery_queue))

    def test_03_wait_is_effective(self):
        cfg = config("scenario_60node_stage3_c1_wait.yaml")
        sim = Simulator(cfg, seed=3)
        install_flexible_return(sim, farthest_node(sim), origin=sim.vehicle.position)
        candidates = sim.ground_controller.generate_ground_candidates(next_event_time=sim.recovery_queue[0]["eta"])
        keep = next(row for row in candidates if row.action_type == "keep_route")
        wait = next(row for row in candidates if row.action_type == "wait")
        self.assertTrue(wait.feasible)
        self.assertLess(wait.predicted_uav_wait, keep.predicted_uav_wait)
        self.assertEqual(sim.ground_controller._select_best_candidate(candidates).action_type, "wait")
        self.assertTrue(sim.ground_controller.commit_cav_action(wait))
        sim._recover_drone(sim.drones[0], min(float(wait.wait_until), sim.now + 1.0))
        self.assertIsNone(sim.vehicle.wait_until)

    def test_04_wait_filtered_for_locked_recovery(self):
        cfg = config("scenario_60node_stage3_c1_wait.yaml")
        cfg["cav_control"]["maximum_uav_wait_s"] = 1.0
        sim = Simulator(cfg, seed=4)
        _, entry = install_flexible_return(sim, farthest_node(sim), battery=100.0, locked=True)
        entry["eta"] = sim.now
        entry["battery_at_eta"] = 1.0
        wait = next(
            row
            for row in sim.ground_controller.generate_ground_candidates(next_event_time=sim.now + 60.0)
            if row.action_type == "wait"
        )
        self.assertFalse(wait.feasible)

    def test_05_move_to_effective(self):
        cfg = config("scenario_60node_stage3_c2_move.yaml")
        cfg["cav_control"]["action_set"] = ["keep_route", "move_to"]
        sim = Simulator(cfg, seed=5)
        install_flexible_return(sim, farthest_node(sim), origin=adjacent_node(sim))
        candidates = sim.ground_controller.generate_ground_candidates()
        moves = [row for row in candidates if row.action_type == "move_to" and row.feasible]
        self.assertTrue(moves)
        self.assertLess(min(row.predicted_uav_return_time for row in moves), candidates[0].predicted_uav_return_time)

    def test_06_unreachable_target_filtered(self):
        sim = Simulator(config(), seed=6)
        candidate = GroundCandidate(100, "move_to", target_node=999999)
        evaluated = sim.ground_controller.evaluate_cav_candidate(candidate)
        self.assertFalse(evaluated.feasible)
        self.assertEqual(evaluated.infeasible_reason, "vehicle_unreachable")

    def test_07_detour_constraint(self):
        cfg = config()
        cfg["cav_control"]["max_detour_ratio"] = 0.0
        cfg["cav_control"]["prediction_horizon_s"] = 10000.0
        sim = Simulator(cfg, seed=7)
        head = farthest_node(sim)
        install_flexible_return(sim, head)
        keep = sim.ground_controller.evaluate_cav_candidate(GroundCandidate(0, "keep_route"))
        best = None
        for node in sim.network.nodes:
            candidate = GroundCandidate(1, "reroute", target_node=int(node), waypoint_node=int(node))
            candidate = sim.ground_controller.evaluate_cav_candidate(candidate, keep_reference=keep)
            if candidate.detour_ratio > 1e-9:
                best = candidate
                break
        self.assertIsNotNone(best)
        self.assertFalse(best.feasible)
        self.assertEqual(best.infeasible_reason, "max_detour_ratio")

    def test_08_locked_recovery_protected(self):
        sim = Simulator(config(), seed=8)
        _, entry = install_flexible_return(sim, farthest_node(sim), locked=True)
        original = int(entry["recovery_node"])
        candidate = GroundCandidate(
            1,
            "move_to",
            target_node=adjacent_node(sim),
            recovery_updates={entry["sortie_id"]: adjacent_node(sim)},
        )
        evaluated = sim.ground_controller.evaluate_cav_candidate(candidate)
        self.assertFalse(evaluated.feasible)
        self.assertEqual(sim.recovery_queue[0]["recovery_node"], original)

    def test_09_flexible_recovery_update(self):
        sim = Simulator(config("scenario_60node_stage3_c2_move.yaml"), seed=9)
        drone, entry = install_flexible_return(sim, farthest_node(sim))
        old = int(entry["recovery_node"])
        candidates = sim.ground_controller.generate_ground_candidates()
        candidate = next(
            row
            for row in candidates
            if row.feasible and row.action_type == "move_to" and row.recovery_updates
        )
        self.assertTrue(sim.ground_controller.commit_cav_action(candidate))
        self.assertNotEqual(sim.recovery_queue[0]["recovery_node"], old)
        self.assertEqual(drone.sortie.recovery_node, sim.recovery_queue[0]["recovery_node"])

    def test_10_uav_battery_constraint(self):
        sim = Simulator(config(), seed=10)
        _, entry = install_flexible_return(sim, adjacent_node(sim), origin=farthest_node(sim), battery=1.0)
        target = farthest_node(sim, origin=int(entry["recovery_node"]))
        candidate = GroundCandidate(
            1,
            "move_to",
            target_node=target,
            recovery_updates={entry["sortie_id"]: target},
        )
        evaluated = sim.ground_controller.evaluate_cav_candidate(candidate)
        self.assertFalse(evaluated.feasible)
        self.assertEqual(evaluated.infeasible_reason, "battery_reserve")

    def test_11_recovery_capacity_is_sequential(self):
        sim = Simulator(config(), seed=11)
        _, first = install_flexible_return(sim, adjacent_node(sim))
        second = dict(first)
        second.update({"drone_id": "drone1", "uav_id": "drone1", "sortie_id": "second", "eta": first["eta"]})
        sim.recovery_queue.append(second)
        result = sim.recovery_scheduler.simulate_ground_plan(sim.recovery_queue, sim.now)
        self.assertTrue(result["feasible"])
        self.assertGreaterEqual(result["schedule"][1]["scheduled_start_time"], result["schedule"][0]["scheduled_finish_time"])

    def test_12_route_commitment_until_next_node(self):
        cfg = config()
        cfg["cav_control"]["prediction_horizon_s"] = 10000.0
        sim = Simulator(cfg, seed=12)
        target = adjacent_node(sim)
        candidate = sim.ground_controller.evaluate_cav_candidate(GroundCandidate(1, "move_to", target_node=target))
        self.assertTrue(candidate.feasible)
        self.assertTrue(sim.ground_controller.commit_cav_action(candidate))
        other = farthest_node(sim)
        evaluated = sim.ground_controller.evaluate_cav_candidate(GroundCandidate(2, "move_to", target_node=other))
        self.assertFalse(evaluated.feasible)
        self.assertEqual(evaluated.infeasible_reason, "route_commitment")

    def test_13_immediate_reverse_filtered(self):
        sim = Simulator(config(), seed=13)
        current = int(sim.vehicle.position)
        previous = adjacent_node(sim)
        sim.vehicle.route_edge_log.append((sim.now, previous, current))
        candidate = sim.ground_controller.evaluate_cav_candidate(GroundCandidate(1, "move_to", target_node=previous))
        self.assertTrue(candidate.reverse_route_flag)
        self.assertFalse(candidate.feasible)

    def test_14_cooldown_prevents_oscillation(self):
        cfg = config()
        cfg["cav_control"]["prediction_horizon_s"] = 10000.0
        sim = Simulator(cfg, seed=14)
        sim.vehicle.note_route_change(sim.now, adjacent_node(sim), sim.cfg["route_commitment"])
        sim.vehicle.committed_nodes_remaining = 0
        candidate = sim.ground_controller.evaluate_cav_candidate(
            GroundCandidate(1, "move_to", target_node=farthest_node(sim))
        )
        self.assertFalse(candidate.feasible)
        self.assertEqual(candidate.infeasible_reason, "route_commitment")

    def test_15_evaluation_has_no_side_effects(self):
        sim = Simulator(config(), seed=15)
        install_flexible_return(sim, farthest_node(sim))
        before = (
            sim.now,
            deepcopy(sim.vehicle.__dict__),
            deepcopy(sim.recovery_queue),
            [deepcopy(drone.__dict__) for drone in sim.drones],
            deepcopy(sim.rng.bit_generator.state),
            sim.recovery_schedule_version,
        )
        sim.ground_controller.generate_ground_candidates(next_event_time=100.0)
        after = (
            sim.now,
            sim.vehicle.__dict__,
            sim.recovery_queue,
            [drone.__dict__ for drone in sim.drones],
            sim.rng.bit_generator.state,
            sim.recovery_schedule_version,
        )
        self.assertEqual(before, after)

    def test_16_stale_commit_rolls_back_to_keep(self):
        sim = Simulator(config(), seed=16)
        install_flexible_return(sim, farthest_node(sim))
        candidate = next(row for row in sim.ground_controller.generate_ground_candidates() if row.action_type == "move_to")
        state = deepcopy(sim.vehicle.__dict__), deepcopy(sim.recovery_queue)
        sim.recovery_schedule_version += 1
        self.assertFalse(sim.ground_controller.commit_cav_action(candidate))
        self.assertEqual(state, (sim.vehicle.__dict__, sim.recovery_queue))

    def test_17_route_change_invalidates_old_event(self):
        sim = Simulator(config(), seed=17)
        old_version = sim.vehicle.schedule_version
        queue = EventQueue()
        queue.schedule(time=10.0, event_type="CAV_NODE_ARRIVAL", entity_id=sim.vehicle.id, schedule_version=old_version)
        candidate = sim.ground_controller.evaluate_cav_candidate(
            GroundCandidate(1, "move_to", target_node=adjacent_node(sim))
        )
        self.assertTrue(sim.ground_controller.commit_cav_action(candidate))
        self.assertIsNone(queue.peek_time(0.0, lambda event: sim.vehicle.schedule_version))
        self.assertEqual(queue.stale_event_count, 1)

    def test_18_stage1_dynamic_insertion_compatible(self):
        sim = Simulator(config(), seed=18)
        move = sim.ground_controller.evaluate_cav_candidate(
            GroundCandidate(1, "move_to", target_node=adjacent_node(sim))
        )
        self.assertTrue(sim.ground_controller.commit_cav_action(move))
        candidates = sim.get_candidates()
        if candidates:
            result = sim.evaluate_insertion(candidates[0], commit=False)
            self.assertIn("feasible", result)
        self.assertTrue(all(sim.network.G.has_edge(u, v) for u, v in zip(sim.vehicle.path_nodes[:-1], sim.vehicle.path_nodes[1:])))

    def test_19_stage2_discount_compatible(self):
        cfg = load_config(ROOT / "configs" / "scenario_60node_stage3_c0_passive.yaml")
        env = DynamicPatrolEnv(cfg)
        env.reset(seed=19)
        output = env.step(0)
        info = output[-1]
        expected = transition_discount(0.99, info["delta_t"], 60.0, "continuous_time")
        self.assertAlmostEqual(info["transition_discount"], expected, places=12)

    def test_20_only_one_decision_per_timestamp(self):
        sim = Simulator(config(), seed=20)
        first = sim.ground_controller.decide(["TASK_RELEASE", "CAV_NODE_ARRIVAL"], force=True)
        second = sim.ground_controller.decide(["UAV_SERVICE_COMPLETE"], force=True)
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(len(sim.ground_controller.action_log), 1)

    def test_21_candidate_node_limit(self):
        sim = Simulator(config(), seed=21)
        install_flexible_return(sim, farthest_node(sim))
        self.assertLessEqual(
            len(sim.ground_controller._candidate_node_specs()),
            int(sim.cfg["cav_control"]["max_ground_candidates"]),
        )

    def test_22_deterministic_candidate_and_selection(self):
        sim = Simulator(config(), seed=22)
        install_flexible_return(sim, farthest_node(sim))
        left = sim.ground_controller.generate_ground_candidates(next_event_time=100.0)
        right = sim.ground_controller.generate_ground_candidates(next_event_time=100.0)
        compact = lambda rows: [
            (row.action_type, row.target_node, row.feasible, round(row.score, 12), tuple(row.planned_route))
            for row in rows
        ]
        self.assertEqual(compact(left), compact(right))
        self.assertEqual(
            sim.ground_controller._select_best_candidate(left).action_type,
            sim.ground_controller._select_best_candidate(right).action_type,
        )

    def test_23_keep_route_is_safe_fallback(self):
        sim = Simulator(config(), seed=23)
        keep = sim.ground_controller.evaluate_cav_candidate(GroundCandidate(0, "keep_route"))
        bad = sim.ground_controller.evaluate_cav_candidate(GroundCandidate(1, "move_to", target_node=999999))
        self.assertTrue(keep.feasible)
        self.assertEqual(sim.ground_controller._select_best_candidate([keep, bad]).action_type, "keep_route")

    def test_24_distance_metric_conservation(self):
        sim = Simulator(config("scenario_60node_stage3_c0_passive.yaml"), seed=24)
        target = farthest_node(sim)
        sim.vehicle.set_target(target, sim.network, sim.now)
        while not sim.vehicle.is_idle():
            sim.step(min(sim.dt, sim.vehicle.remaining_time))
        expected = sum(sim.network.arc_length((u, v)) for _, u, v in sim.vehicle.route_edge_log)
        self.assertAlmostEqual(sim.metrics("test").cav_travel_distance_m, expected, places=9)
        self.assertAlmostEqual(sim.metrics("test").vehicle_wait_time_s, sim.vehicle.wait_time, places=9)

    def test_25_zero_duration_return_segment_completes_at_same_timestamp(self):
        sim = Simulator(config("scenario_60node_stage3_c1_wait.yaml"), seed=25)
        drone, entry = install_flexible_return(
            sim,
            sim.vehicle.position,
            origin=sim.vehicle.position,
        )
        drone.segment_index = len(drone.sortie.segments) - 1
        drone.sortie.segments[-1]["duration"] = 0.0
        drone.segment_remaining = 0.0
        drone.sortie.recovery_node = int(sim.vehicle.position)
        entry["recovery_node"] = int(sim.vehicle.position)
        entry["eta"] = float(sim.now)

        sim.step(0.0)

        self.assertTrue(drone.is_idle())
        self.assertEqual(sim.recovery_queue, [])
        self.assertEqual(sim.now, 0.0)


if __name__ == "__main__":
    unittest.main()
