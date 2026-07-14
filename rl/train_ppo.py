from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.rl.seed_scheduler import (
    CurriculumSeedScheduler,
    CurriculumStage,
    SEED_MODES,
    build_training_stages,
    parse_int_list,
)
from dynamic_patrol_marl.utils.config import load_config, project_root


PPO_TENSORBOARD_DIRNAME = "ppo"
DEFAULT_EVAL_SEEDS = (0, 1, 2, 3, 4)


def run_random_masked(cfg, episodes: int, out_csv: Path):
    rows = []
    for seed in range(episodes):
        env = DynamicPatrolEnv(cfg)
        env.reset(seed=seed)
        done = False
        while not done:
            action = env.sample_masked_action()
            step_out = env.step(action)
            if len(step_out) == 5:
                _, _, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                _, _, done, _ = step_out
        rows.append(env.sim.metrics("random_masked").to_dict())
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def parse_eval_seeds(value: str | None) -> List[int]:
    parsed = parse_int_list(value)
    return parsed if parsed is not None else list(DEFAULT_EVAL_SEEDS)


def evaluate_ppo_policy(
    model,
    cfg: Dict,
    eval_seeds: Iterable[int],
    scenario: str,
    action_masking: bool = True,
    method_name: str = "ppo_eval",
) -> pd.DataFrame:
    rows: List[Dict] = []
    for seed in eval_seeds:
        env = DynamicPatrolEnv(cfg, scenario=scenario)
        reset_out = env.reset(seed=int(seed))
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, info = reset_out
        else:
            obs = reset_out
            info = {"action_mask": env.action_masks()}
        done = False
        episode_reward = 0.0
        while not done:
            if action_masking:
                action, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(int(action))
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_out
            episode_reward += float(reward)
        row = env.sim.metrics(method_name).to_dict()
        row["episode_reward"] = episode_reward
        rows.append(row)
    return pd.DataFrame(rows)


def stage_log_row(
    stage_idx: int,
    stage: CurriculumStage,
    seed_mode: str,
    cumulative_timesteps: int,
    eval_df: pd.DataFrame,
) -> Dict:
    return {
        "stage": int(stage_idx),
        "stage_name": stage.name,
        "seed_mode": seed_mode,
        "seed_pool": stage.seed_pool,
        "stage_timesteps": int(stage.timesteps),
        "cumulative_timesteps": int(cumulative_timesteps),
        "mean_reward": float(pd.to_numeric(eval_df.get("episode_reward"), errors="coerce").mean()),
        "mean_completion_rate": float(pd.to_numeric(eval_df.get("completion_rate"), errors="coerce").mean()),
        "mean_timeout_rate": float(pd.to_numeric(eval_df.get("timeout_rate"), errors="coerce").mean()),
    }


def train_maskable_ppo(
    cfg,
    total_timesteps: int | None,
    out_dir: Path,
    continue_from: Path | None = None,
    model_name: str = "trained_model.zip",
    run_name: str = "ppo",
    tensorboard_dir: Path | None = None,
    enable_tensorboard: bool = True,
    seed: int | None = None,
    seed_mode: str | None = None,
    stage_timesteps: Sequence[int] | None = None,
    eval_seeds: Sequence[int] = DEFAULT_EVAL_SEEDS,
    eval_scenario: str = "eval",
    action_masking: bool = True,
    device_name: str = "auto",
):
    try:
        if action_masking:
            from sb3_contrib import MaskablePPO
            from sb3_contrib.common.wrappers import ActionMasker
        else:
            from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except Exception as e:
        raise RuntimeError(
            "PPO training requires optional packages: pip install gymnasium stable-baselines3 sb3-contrib. "
            "Core baselines and random masked evaluation run without them."
        ) from e

    class TrainMetricsCallback(BaseCallback):
        def __init__(self, log_dir: Path | None, scheduler: CurriculumSeedScheduler, seed_mode_name: str):
            super().__init__()
            self.log_dir = log_dir
            self.scheduler = scheduler
            self.seed_mode_name = seed_mode_name
            self.writer = None
            self.episode_rewards: List[float] = [0.0]
            self.episode_steps: List[int] = [0]

        def _on_training_start(self) -> None:
            if self.log_dir is None:
                return
            try:
                from torch.utils.tensorboard import SummaryWriter
            except Exception as exc:
                raise RuntimeError(
                    "TensorBoard logging is enabled but unavailable. Install tensorboard "
                    "or rerun with --disable-tensorboard."
                ) from exc
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards", [])
            dones = self.locals.get("dones", [])
            infos = self.locals.get("infos", [])
            for idx, reward in enumerate(rewards):
                if idx >= len(self.episode_rewards):
                    self.episode_rewards.append(0.0)
                    self.episode_steps.append(0)
                self.episode_rewards[idx] += float(reward)
                self.episode_steps[idx] += 1
                done = bool(dones[idx]) if idx < len(dones) else False
                if done:
                    info: Dict = infos[idx] if idx < len(infos) else {}
                    if self.writer is not None:
                        self.writer.add_scalar("train/episode_reward", self.episode_rewards[idx], self.num_timesteps)
                        self.writer.add_scalar("train/completion_rate", float(info.get("completion_rate", 0.0)), self.num_timesteps)
                        self.writer.add_scalar("train/timeout_rate", float(info.get("timeout_rate", 0.0)), self.num_timesteps)
                        self.writer.add_scalar("train/illegal_actions", float(info.get("illegal_actions", 0.0)), self.num_timesteps)
                        self.writer.add_scalar("train/episode_steps", self.episode_steps[idx], self.num_timesteps)
                        self.writer.add_scalar("train/curriculum_stage", float(self.scheduler.stage_idx + 1), self.num_timesteps)
                    self.episode_rewards[idx] = 0.0
                    self.episode_steps[idx] = 0
            return True

        def _on_training_end(self) -> None:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

    resolved_seed_mode, base_seed, stages = build_training_stages(
        cfg=cfg,
        seed_mode=seed_mode,
        seed=seed,
        total_timesteps=total_timesteps,
        stage_timesteps=stage_timesteps,
    )
    scheduler = CurriculumSeedScheduler(stages, rng_seed=base_seed)
    base_env = DynamicPatrolEnv(cfg, seed_scheduler=scheduler)
    if action_masking:
        def mask_fn(env):
            return env.action_masks()

        env = ActionMasker(base_env, mask_fn)
        model_cls = MaskablePPO
        policy_name = "MaskPPO"
        method_name = "ppo_eval"
    else:
        env = base_env
        model_cls = PPO
        policy_name = "UnmaskPPO"
        method_name = "unmask_ppo_eval"
    tensorboard_log_dir = tensorboard_dir / run_name if enable_tensorboard and tensorboard_dir is not None else None
    callback = TrainMetricsCallback(tensorboard_log_dir, scheduler, resolved_seed_mode)

    if continue_from is not None and continue_from.exists():
        model = model_cls.load(continue_from, env=env, verbose=1, device=device_name)
    else:
        model = model_cls("MlpPolicy", env, verbose=1, seed=base_seed, device=device_name)

    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    stage_log_path = out_dir / "curriculum_training_log.csv"
    stage_rows: List[Dict] = []

    for stage_idx, stage in enumerate(stages):
        scheduler.set_stage(stage_idx)
        print(
            f"{policy_name} {resolved_seed_mode} stage {stage_idx + 1}/{len(stages)}: "
            f"{stage.name}, seeds={stage.seed_pool}, timesteps={stage.timesteps}"
        )
        reset_steps = bool(stage_idx == 0 and continue_from is None)
        model.learn(
            total_timesteps=int(stage.timesteps),
            reset_num_timesteps=reset_steps,
            callback=callback,
        )

        cumulative_timesteps = int(model.num_timesteps)
        checkpoint_path = checkpoints_dir / f"model_stage_{stage_idx + 1}.zip"
        model.save(checkpoint_path)

        eval_df = evaluate_ppo_policy(
            model,
            cfg,
            eval_seeds=eval_seeds,
            scenario=eval_scenario,
            action_masking=action_masking,
            method_name=method_name,
        )
        stage_rows.append(
            stage_log_row(
                stage_idx=stage_idx + 1,
                stage=stage,
                seed_mode=resolved_seed_mode,
                cumulative_timesteps=cumulative_timesteps,
                eval_df=eval_df,
            )
        )
        pd.DataFrame(stage_rows).to_csv(stage_log_path, index=False)
        print(
            f"Saved checkpoint: {checkpoint_path}; "
            f"mean_reward={stage_rows[-1]['mean_reward']:.3f}, "
            f"completion={stage_rows[-1]['mean_completion_rate']:.3f}"
        )

    if resolved_seed_mode == "curriculum" and model_name == "trained_model.zip":
        final_name = "trained_model_curriculum.zip" if action_masking else "unmasked_trained_model_curriculum.zip"
    else:
        final_name = model_name
    model_path = out_dir / final_name
    model.save(model_path)
    metadata = {
        "seed_mode": resolved_seed_mode,
        "action_masking": bool(action_masking),
        "mask_mode": "masked" if action_masking else "unmasked",
        "base_seed": int(base_seed),
        "stages": [
            {
                "name": s.name,
                "timesteps": int(s.timesteps),
                "seeds": s.seeds,
                "random_seed": bool(s.random_seed),
            }
            for s in stages
        ],
        "eval_seeds": list(map(int, eval_seeds)),
        "eval_scenario": eval_scenario,
        "model_path": str(model_path),
        "stage_log_path": str(stage_log_path),
    }
    with open(out_dir / "curriculum_training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scenario_60node.yaml")
    parser.add_argument("--timesteps", type=int, default=None, help="backward-compatible alias for --total-timesteps")
    parser.add_argument("--total-timesteps", type=int, default=None, help="override total training timesteps")
    parser.add_argument("--stage-timesteps", default=None, help="comma-separated stage timesteps; one value or one per stage")
    parser.add_argument("--seed", type=int, default=None, help="base seed for PPO policy and fixed/curriculum sampling")
    parser.add_argument("--seed-mode", choices=SEED_MODES, default=None, help="fixed, random, or curriculum")
    parser.add_argument("--random-only", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--continue-from", default=None, help="optional existing MaskablePPO model to continue training")
    parser.add_argument("--model-name", default="trained_model.zip", help="output model filename for non-curriculum PPO")
    parser.add_argument("--run-name", default="ppo", help="TensorBoard run name under results/rl/tensorboard/ppo")
    parser.add_argument("--out-dir", default=None, help="default: results/rl, or results/rl/unmasked_ppo with --unmasked")
    parser.add_argument("--tensorboard-dir", default=None, help="default: results/rl/tensorboard/ppo")
    parser.add_argument("--disable-tensorboard", action="store_true", help="train without writing custom TensorBoard metrics")
    parser.add_argument("--eval-seeds", default=None, help="comma-separated deterministic seeds for stage-end evaluation")
    parser.add_argument("--eval-scenario", default="eval")
    parser.add_argument("--device", default="auto", help="PPO device, e.g. auto, cpu, cuda")
    parser.add_argument("--unmasked", action="store_true", help="train standard PPO without action masking")
    args = parser.parse_args()

    root = project_root()
    cfg = load_config(root / args.config if not str(args.config).startswith("/") else args.config)
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = root / out_dir
    elif args.unmasked:
        out_dir = root / "results" / "rl" / "unmasked_ppo"
    else:
        out_dir = root / "results" / "rl"
    out_dir.mkdir(parents=True, exist_ok=True)
    random_csv = run_random_masked(cfg, args.episodes, out_dir / "random_masked_eval.csv")
    print(f"Saved random masked evaluation: {random_csv}")

    if not args.random_only:
        continue_from = Path(args.continue_from) if args.continue_from else None
        if continue_from is not None and not continue_from.is_absolute():
            continue_from = root / continue_from
        tensorboard_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else root / "results" / "rl" / "tensorboard" / PPO_TENSORBOARD_DIRNAME
        if not tensorboard_dir.is_absolute():
            tensorboard_dir = root / tensorboard_dir

        total_timesteps = args.total_timesteps if args.total_timesteps is not None else args.timesteps
        model_path = train_maskable_ppo(
            cfg,
            total_timesteps=total_timesteps,
            out_dir=out_dir,
            continue_from=continue_from,
            model_name=args.model_name,
            run_name=args.run_name,
            tensorboard_dir=tensorboard_dir,
            enable_tensorboard=not args.disable_tensorboard,
            seed=args.seed,
            seed_mode=args.seed_mode,
            stage_timesteps=parse_int_list(args.stage_timesteps),
            eval_seeds=parse_eval_seeds(args.eval_seeds),
            eval_scenario=args.eval_scenario,
            action_masking=not args.unmasked,
            device_name=args.device,
        )
        print(f"Saved model: {model_path}")
        print(f"Saved TensorBoard logs: {tensorboard_dir / args.run_name}")
        print(f"Saved curriculum log: {out_dir / 'curriculum_training_log.csv'}")


if __name__ == "__main__":
    main()
