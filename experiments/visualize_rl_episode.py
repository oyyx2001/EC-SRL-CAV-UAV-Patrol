from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.utils.config import ensure_dirs, load_config, project_root
from dynamic_patrol_marl.utils.visualization import plot_episode, plot_episode_animation, plot_episode_tasks


def _load_maskable_ppo(model_path: Path):
    try:
        from sb3_contrib import MaskablePPO
    except Exception as exc:
        raise RuntimeError(
            "RL visualization requires optional dependencies: "
            "gymnasium, stable-baselines3, sb3-contrib, and torch."
        ) from exc
    return MaskablePPO.load(model_path)


def _candidate_record(candidates: List[Any], action: int) -> Dict[str, Any]:
    if action <= 0:
        return {
            "selected_drone": "",
            "selected_tasks": "",
            "selected_recovery_node": "",
            "selected_vehicle_anchor_node": "",
            "selected_vehicle_queue_time_s": "",
            "selected_vehicle_time_s": "",
            "selected_drone_time_s": "",
            "selected_score": "",
        }
    idx = int(action) - 1
    if idx >= len(candidates):
        return {
            "selected_drone": "",
            "selected_tasks": "",
            "selected_recovery_node": "",
            "selected_vehicle_anchor_node": "",
            "selected_vehicle_queue_time_s": "",
            "selected_vehicle_time_s": "",
            "selected_drone_time_s": "",
            "selected_score": "",
        }
    sortie = candidates[idx]
    return {
        "selected_drone": sortie.drone_id,
        "selected_tasks": ";".join(map(str, sortie.task_ids)),
        "selected_recovery_node": sortie.recovery_node,
        "selected_vehicle_anchor_node": sortie.vehicle_anchor_node,
        "selected_vehicle_queue_time_s": sortie.vehicle_queue_time_s,
        "selected_vehicle_time_s": sortie.vehicle_time_s,
        "selected_drone_time_s": sortie.drone_time_s,
        "selected_score": sortie.score,
    }


def run_rl_episode(cfg: Dict, seed: int, model_path: Path, scenario: str = "visual"):
    env = DynamicPatrolEnv(cfg, scenario=scenario)
    model = _load_maskable_ppo(model_path)
    obs, info = env.reset(seed=seed)
    done = False
    action_log: List[Dict[str, Any]] = []

    while not done:
        candidates = list(env.candidates)
        queue = list(env.sim.recovery_queue)
        action, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
        action = int(action)
        action_log.append(
            {
                "decision_step": len(action_log),
                "time": float(env.sim.now),
                "cav_node": int(env.sim.vehicle.position),
                "cav_target": "" if env.sim.vehicle.target_node is None else int(env.sim.vehicle.target_node),
                "cav_remaining_time_s": float(env.sim.vehicle.remaining_time),
                "recovery_queue_len": len(queue),
                "recovery_queue": "|".join(f"{q['drone_id']}@{q['recovery_node']}" for q in queue),
                "front_recovery_node": "" if not queue else int(queue[0]["recovery_node"]),
                "action": action,
                "valid_actions": int(info["action_mask"].sum()),
                "num_candidates": len(candidates),
                **_candidate_record(candidates, action),
            }
        )
        step_out = env.step(action)
        obs, reward, terminated, truncated, info = step_out
        action_log[-1]["reward"] = float(reward)
        done = bool(terminated or truncated)

    return env.sim, action_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scenario_60node.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="results/rl/trained_model.zip")
    parser.add_argument("--scenario", default="visual")
    parser.add_argument("--animation-frames", type=int, default=60)
    parser.add_argument("--animation-fps", type=int, default=5)
    parser.add_argument("--skip-animation", action="store_true")
    args = parser.parse_args()

    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path

    cfg = load_config(cfg_path)
    out_dir = ensure_dirs(cfg) / "episode_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    sim, action_log = run_rl_episode(cfg, seed=args.seed, model_path=model_path, scenario=args.scenario)
    prefix = f"episode_seed_{args.seed}_rl_model"
    sim.export_logs(out_dir, prefix=prefix)
    pd.DataFrame(action_log).to_csv(out_dir / f"{prefix}_actions.csv", index=False)

    image_path = out_dir / f"{prefix}.png"
    tasks_path = out_dir / f"{prefix}_tasks_map.png"
    plot_episode(sim, image_path, title=f"RL model, seed={args.seed}")
    plot_episode_tasks(sim, tasks_path, title=f"RL generated tasks, seed={args.seed}")

    if not args.skip_animation:
        animation_path = out_dir / f"{prefix}_animation.gif"
        plot_episode_animation(
            sim,
            animation_path,
            title=f"RL model, seed={args.seed}",
            frames=args.animation_frames,
            fps=args.animation_fps,
        )
    else:
        animation_path = None

    print(sim.metrics("rl_model").to_dict())
    print(f"Saved RL image: {image_path}")
    print(f"Saved RL task map: {tasks_path}")
    print(f"Saved RL actions: {out_dir / f'{prefix}_actions.csv'}")
    if animation_path is not None:
        print(f"Saved RL animation: {animation_path}")


if __name__ == "__main__":
    main()
