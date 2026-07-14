from __future__ import annotations

from dynamic_patrol_marl.rl.multi_vehicle_env import build_multi_vehicle_spec
from dynamic_patrol_marl.utils.config import load_config, project_root


def main():
    cfg = load_config(project_root() / "configs/scenario_60node.yaml")
    spec = build_multi_vehicle_spec(cfg)
    print("MAPPO is a second-stage extension. Current MVP provides this interface spec:")
    print(spec)
    print("Recommended next step: copy DynamicPatrolEnv into a PettingZoo/MARL wrapper after baseline and PPO results are stable.")


if __name__ == "__main__":
    main()
