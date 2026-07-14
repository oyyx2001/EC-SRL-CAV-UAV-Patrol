from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class MultiVehicleRoadmap:
    """S9 extension scaffold.

    The submitted MVP intentionally solves the first-stage problem: one CAV patrol
    vehicle with multiple drones. This file keeps the interface for the second-stage
    MAPPO extension explicit instead of hiding an unvalidated implementation.
    """

    num_vehicles: int = 2
    drones_per_vehicle: int = 2
    cross_recovery: bool = False


def build_multi_vehicle_spec(cfg: Dict) -> Dict:
    return {
        "num_vehicles": int(cfg.get("agents", {}).get("num_vehicles", 1)),
        "num_drones_total": int(cfg.get("agents", {}).get("num_drones", 2)),
        "state_design": ["fleet positions", "drone batteries", "active task pool", "candidate fleet-level sorties"],
        "action_design": ["fleet-task allocation", "fleet-level rendezvous choice", "candidate sortie choice"],
        "recommended_algorithm": "MAPPO after S0-S8 are stable",
    }


if __name__ == "__main__":
    import json
    from dynamic_patrol_marl.utils.config import load_config, project_root
    cfg = load_config(project_root() / "configs/scenario_60node.yaml")
    print(json.dumps(build_multi_vehicle_spec(cfg), ensure_ascii=False, indent=2))
