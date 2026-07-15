from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - clear runtime message.
    raise RuntimeError("Masked DQN training requires PyTorch. Install torch to use this module.") from exc

from dynamic_patrol_marl.rl.dqn_networks import make_q_network, masked_q_values, reset_noisy_layers
from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.rl.paths import resolve_dqn_output_dirs
from dynamic_patrol_marl.rl.replay_buffer import MaskedReplayBuffer, PrioritizedMaskedReplayBuffer
from dynamic_patrol_marl.rl.smdp import (
    NStepSMDPAccumulator,
    NStepTransition,
    one_step_smdp_target,
    td_error_delta_t_bin,
)
from dynamic_patrol_marl.rl.seed_scheduler import (
    CurriculumSeedScheduler,
    SEED_MODES,
    build_training_stages,
    parse_int_list,
)
from dynamic_patrol_marl.utils.config import load_config, project_root


ALGORITHMS = ("dqn", "double_dqn", "dueling_double_dqn", "rainbow_dqn")
EVAL_COLUMNS = [
    "step",
    "algo",
    "seed",
    "completion_rate",
    "timeout_rate",
    "avg_response_time",
    "weighted_aoi",
    "energy",
    "uav_waiting_time",
    "cav_waiting_time",
    "episode_reward",
]


@dataclass
class DQNHyperParams:
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 128
    replay_buffer_size: int = 100000
    learning_starts: int = 5000
    train_frequency: int = 1
    target_update_interval: int = 2000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 100000
    max_grad_norm: float = 10.0
    eval_interval: int = 20000
    candidate_dim: int = 6
    prioritized_replay_alpha: float = 0.6
    prioritized_replay_beta_start: float = 0.4
    prioritized_replay_beta_final: float = 1.0
    prioritized_replay_beta_steps: int = 200000
    priority_epsilon: float = 1e-6


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def reset_env(env: DynamicPatrolEnv, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        mask = info.get("action_mask", env.action_masks())
    else:
        obs = out
        mask = env.action_masks()
    return np.asarray(obs, dtype=np.float32), np.asarray(mask, dtype=bool)


def step_env(env: DynamicPatrolEnv, action: int) -> Tuple[np.ndarray, float, bool, np.ndarray, Dict]:
    out = env.step(int(action))
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        info = dict(info)
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
    else:
        obs, reward, done, info = out
    mask = info.get("action_mask", env.action_masks())
    info = dict(info)
    info.setdefault("terminated", bool(done))
    info.setdefault("truncated", False)
    return np.asarray(obs, dtype=np.float32), float(reward), bool(done), np.asarray(mask, dtype=bool), info


def epsilon_by_step(step: int, hp: DQNHyperParams) -> float:
    frac = min(1.0, max(0.0, float(step) / float(max(1, hp.epsilon_decay_steps))))
    return hp.epsilon_start + frac * (hp.epsilon_final - hp.epsilon_start)


def prioritized_beta_by_step(step: int, hp: DQNHyperParams) -> float:
    frac = min(1.0, max(0.0, float(step) / float(max(1, hp.prioritized_replay_beta_steps))))
    return hp.prioritized_replay_beta_start + frac * (hp.prioritized_replay_beta_final - hp.prioritized_replay_beta_start)


def sample_valid_action(action_mask: np.ndarray, rng: np.random.Generator) -> int:
    valid = np.flatnonzero(np.asarray(action_mask, dtype=bool))
    if len(valid) == 0:
        return 0
    return int(rng.choice(valid))


def policy_action_mask(action_mask: np.ndarray, action_masking: bool) -> np.ndarray:
    """Return the mask used by the policy.

    In unmasked experiments every discrete action is available to the policy;
    infeasible actions are still handled by the environment as illegal actions.
    """
    mask = np.asarray(action_mask, dtype=bool)
    return mask if action_masking else np.ones_like(mask, dtype=bool)


@torch.no_grad()
def select_action(
    q_network: nn.Module,
    obs: np.ndarray,
    action_mask: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    if rng.random() < float(epsilon):
        return sample_valid_action(action_mask, rng)
    obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
    mask_t = torch.as_tensor(action_mask[None, :], device=device, dtype=torch.bool)
    q_values = q_network(obs_t, mask_t)
    return int(masked_q_values(q_values, mask_t).argmax(dim=1).item())


def compute_dqn_loss(
    algo: str,
    online_net: nn.Module,
    target_net: nn.Module,
    batch,
    gamma: float,
    weights: torch.Tensor | None = None,
    action_masking: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    action_masks = batch.action_masks if action_masking else torch.ones_like(batch.action_masks, dtype=torch.bool)
    next_action_masks = batch.next_action_masks if action_masking else torch.ones_like(batch.next_action_masks, dtype=torch.bool)

    q_values = online_net(batch.obs, action_masks)
    current_q = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if algo == "dqn":
            target_next_q = target_net(batch.next_obs, next_action_masks)
            next_q = masked_q_values(target_next_q, next_action_masks).max(dim=1).values
        else:
            online_next_q = online_net(batch.next_obs, next_action_masks)
            next_actions = masked_q_values(online_next_q, next_action_masks).argmax(dim=1)
            target_next_q = target_net(batch.next_obs, next_action_masks)
            next_q = target_next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        discounts = getattr(batch, "bootstrap_discounts", None)
        if discounts is None:
            discounts = torch.full_like(batch.rewards, float(gamma))
        target_q = one_step_smdp_target(batch.rewards, discounts, batch.dones, next_q)
    per_sample_loss = nn.functional.smooth_l1_loss(current_q, target_q, reduction="none")
    if weights is not None:
        per_sample_loss = per_sample_loss * weights
    td_errors = current_q - target_q
    return per_sample_loss.mean(), td_errors


def metric_row(step: int, algo: str, seed: int, metrics: Dict, episode_reward: float) -> Dict:
    return {
        "step": int(step),
        "algo": algo,
        "seed": int(seed),
        "completion_rate": metrics.get("completion_rate", np.nan),
        "timeout_rate": metrics.get("timeout_rate", np.nan),
        "avg_response_time": metrics.get("avg_response_time_s", np.nan),
        "weighted_aoi": metrics.get("weighted_aoi", np.nan),
        "energy": metrics.get("total_energy", np.nan),
        "uav_waiting_time": metrics.get("drone_wait_time_s", np.nan),
        "cav_waiting_time": metrics.get("vehicle_wait_time_s", np.nan),
        "episode_reward": float(episode_reward),
    }


def train_metric_row(
    step: int,
    episode: int,
    episode_steps: int,
    episode_reward: float,
    epsilon: float,
    latest_loss: float,
    metrics: Dict,
    partial_episode: bool = False,
    seed_mode: str = "",
    stage: int | None = None,
    stage_name: str = "",
) -> Dict:
    return {
        "step": int(step),
        "episode": int(episode),
        "episode_steps": int(episode_steps),
        "episode_reward": float(episode_reward),
        "epsilon": float(epsilon),
        "loss": float(latest_loss) if np.isfinite(latest_loss) else np.nan,
        "completion_rate": metrics.get("completion_rate", np.nan),
        "timeout_rate": metrics.get("timeout_rate", np.nan),
        "illegal_actions": metrics.get("illegal_actions", np.nan),
        "partial_episode": bool(partial_episode),
        "seed_mode": seed_mode,
        "stage": np.nan if stage is None else int(stage),
        "stage_name": stage_name,
    }


def write_train_scalars(writer, row: Dict, step: int) -> None:
    writer.add_scalar("train/episode_reward", float(row["episode_reward"]), step)
    writer.add_scalar("train/completion_rate", float(row["completion_rate"]), step)
    writer.add_scalar("train/timeout_rate", float(row["timeout_rate"]), step)
    writer.add_scalar("train/illegal_actions", float(row["illegal_actions"]), step)
    writer.add_scalar("train/episode_steps", int(row["episode_steps"]), step)
    writer.add_scalar("train/epsilon", float(row["epsilon"]), step)
    if np.isfinite(row.get("stage", np.nan)):
        writer.add_scalar("train/curriculum_stage", float(row["stage"]), step)
    if np.isfinite(row.get("loss", np.nan)):
        writer.add_scalar("train/loss", float(row["loss"]), step)


@torch.no_grad()
def evaluate_policy(
    cfg: Dict,
    q_network: nn.Module,
    device: torch.device,
    algo: str,
    step: int,
    eval_seeds: Iterable[int],
    scenario: str,
    action_masking: bool = True,
) -> pd.DataFrame:
    q_network.eval()
    rows: List[Dict] = []
    rng = np.random.default_rng(0)
    method_name = algo if action_masking else f"unmasked_{algo}"
    for seed in eval_seeds:
        env = DynamicPatrolEnv(cfg, scenario=scenario)
        obs, mask = reset_env(env, seed=int(seed))
        done = False
        episode_reward = 0.0
        while not done:
            action = select_action(
                q_network,
                obs,
                policy_action_mask(mask, action_masking),
                epsilon=0.0,
                rng=rng,
                device=device,
            )
            obs, reward, done, mask, _ = step_env(env, action)
            episode_reward += reward
        rows.append(metric_row(step, algo, int(seed), env.sim.metrics(method_name).to_dict(), episode_reward))
    q_network.train()
    return pd.DataFrame(rows, columns=EVAL_COLUMNS)


def save_outputs(
    out_dir: Path,
    run_dir: Path,
    algo: str,
    seed: int,
    cfg_path: str,
    hp: DQNHyperParams,
    online_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs_dim: int,
    action_dim: int,
    training_rows: List[Dict],
    eval_rows: List[Dict],
    total_timesteps: int,
    model_name: str,
    train_scenario: str,
    tensorboard_log_dir: Path | None,
    seed_mode: str,
    stages: List[Dict],
    action_masking: bool,
    replay_config: Dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / model_name
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algo": algo,
            "seed": int(seed),
            "config_path": cfg_path,
            "train_scenario": train_scenario,
            "run_dir": str(run_dir),
            "tensorboard_log_dir": str(tensorboard_log_dir) if tensorboard_log_dir is not None else "",
            "seed_mode": seed_mode,
            "action_masking": bool(action_masking),
            "mask_mode": "masked" if action_masking else "unmasked",
            "training_stages": stages,
            "hyperparams": asdict(hp),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "candidate_dim": int(hp.candidate_dim),
            "replay_config": dict(replay_config or {}),
            "model_state_dict": online_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        model_path,
    )
    pd.DataFrame(training_rows).to_csv(run_dir / "training_curve.csv", index=False)
    pd.DataFrame(eval_rows, columns=EVAL_COLUMNS).to_csv(run_dir / "eval_curve.csv", index=False)

    eval_df = pd.DataFrame(eval_rows)
    final_eval = eval_df[eval_df["step"] == eval_df["step"].max()] if not eval_df.empty else pd.DataFrame()
    summary = {
        "algo": algo,
        "seed": int(seed),
        "run_name": run_dir.name,
        "timesteps": int(total_timesteps),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "model_path": str(model_path),
        "run_dir": str(run_dir),
        "tensorboard_log_dir": str(tensorboard_log_dir) if tensorboard_log_dir is not None else "",
        "seed_mode": seed_mode,
        "action_masking": bool(action_masking),
        "mask_mode": "masked" if action_masking else "unmasked",
        "training_stages": json.dumps(stages),
        "replay_schema_version": int((replay_config or {}).get("schema_version", 1)),
        "time_mode": str((replay_config or {}).get("time_mode", "fixed_interval")),
        **asdict(hp),
    }
    if not final_eval.empty:
        for col in EVAL_COLUMNS:
            if col not in {"step", "algo", "seed"}:
                summary[f"final_{col}"] = float(pd.to_numeric(final_eval[col], errors="coerce").mean())
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)
    summary_path = out_dir / "summary.csv"
    summary_df = pd.DataFrame([summary])
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        if "run_name" in existing.columns:
            existing = existing[existing["run_name"] != run_dir.name]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    with open(run_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def train_masked_dqn(
    cfg: Dict,
    cfg_path: str,
    algo: str,
    total_timesteps: int | None,
    seed: int | None,
    out_dir: Path,
    hp: DQNHyperParams | None = None,
    eval_seeds: Iterable[int] = (1000, 1001, 1002, 1003, 1004),
    eval_scenario: str = "eval",
    device_name: str = "auto",
    model_name: str | None = "model.pt",
    train_scenario: str = "small",
    run_name: str | None = None,
    tensorboard_dir: Path | None = None,
    enable_tensorboard: bool = True,
    seed_mode: str | None = None,
    stage_timesteps: List[int] | None = None,
    action_masking: bool = True,
) -> Path:
    if algo not in ALGORITHMS:
        raise ValueError(f"--algo must be one of {ALGORITHMS}")
    hp = hp or DQNHyperParams()
    resolved_seed_mode, base_seed, stages = build_training_stages(
        cfg=cfg,
        seed_mode=seed_mode,
        seed=seed,
        total_timesteps=total_timesteps,
        stage_timesteps=stage_timesteps,
        default_timesteps=200000,
    )
    effective_timesteps = int(sum(stage.timesteps for stage in stages))
    set_global_seed(base_seed)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    rng = np.random.default_rng(int(base_seed))
    run_prefix = algo if action_masking else f"unmasked_{algo}"
    run_name = run_name or f"{run_prefix}_{resolved_seed_mode}_seed{base_seed}"
    model_name = Path(model_name).name if model_name else f"{run_name}.pt"
    run_dir = out_dir / run_name
    tensorboard_log_dir = tensorboard_dir / run_name if tensorboard_dir is not None else out_dir / "tensorboard" / run_name
    writer = None
    if enable_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
        except Exception as exc:
            raise RuntimeError(
                "TensorBoard logging is enabled but unavailable. Install tensorboard "
                "or rerun with --disable-tensorboard."
            ) from exc
    else:
        tensorboard_log_dir = None

    scheduler = CurriculumSeedScheduler(stages, rng_seed=base_seed)
    env = DynamicPatrolEnv(cfg, scenario=train_scenario, seed_scheduler=scheduler)
    obs, mask = reset_env(env)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    expected_action_dim = 1 + int(cfg["agents"].get("num_drones", 2)) * int(cfg["sortie"].get("top_k_per_drone", 8))
    if action_dim != expected_action_dim:
        raise ValueError(f"action_dim mismatch: env={action_dim}, expected={expected_action_dim}")

    online_net = make_q_network(algo, obs_dim, action_dim, candidate_dim=hp.candidate_dim).to(device)
    target_net = make_q_network(algo, obs_dim, action_dim, candidate_dim=hp.candidate_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(online_net.parameters(), lr=hp.lr)
    time_mode = str(cfg.get("time", {}).get("mode", cfg.get("time_mode", "fixed_interval"))).lower()
    smdp_cfg = cfg.get("smdp", {})
    reference_interval_s = float(smdp_cfg.get("reference_interval_s", 60.0))
    n_step = int(smdp_cfg.get("n_step", 1))
    n_step_accumulator = NStepSMDPAccumulator(n_step)
    replay_kwargs = {
        "default_transition_discount": float(hp.gamma),
        "time_mode": time_mode,
        "require_time_fields": time_mode == "event_driven",
    }
    if algo == "rainbow_dqn":
        replay = PrioritizedMaskedReplayBuffer(
            hp.replay_buffer_size,
            obs_dim,
            action_dim,
            seed=base_seed,
            alpha=hp.prioritized_replay_alpha,
            priority_epsilon=hp.priority_epsilon,
            **replay_kwargs,
        )
    else:
        replay = MaskedReplayBuffer(
            hp.replay_buffer_size,
            obs_dim,
            action_dim,
            seed=base_seed,
            **replay_kwargs,
        )

    training_rows: List[Dict] = []
    eval_rows: List[Dict] = []
    episode_reward = 0.0
    episode_steps = 0
    episode_idx = 0
    latest_loss = np.nan
    gradient_updates = 0
    training_simulated_seconds = 0.0
    td_error_diagnostics: Dict[str, List[float]] = {
        "delta_0_30": [],
        "delta_30_60": [],
        "delta_60_120": [],
        "delta_over_120": [],
    }
    stage_end_steps = np.cumsum([int(stage.timesteps) for stage in stages]).tolist()
    current_stage_idx = 0
    scheduler.set_stage(current_stage_idx)
    print(
        f"DQN seed_mode={resolved_seed_mode}; mask_mode={'masked' if action_masking else 'unmasked'}; "
        f"stage 1/{len(stages)} {stages[0].name}, seeds={stages[0].seed_pool}, timesteps={stages[0].timesteps}"
    )

    for step in range(1, effective_timesteps + 1):
        while current_stage_idx + 1 < len(stage_end_steps) and step > stage_end_steps[current_stage_idx]:
            current_stage_idx += 1
            scheduler.set_stage(current_stage_idx)
            stage = stages[current_stage_idx]
            print(
                f"DQN seed_mode={resolved_seed_mode}; mask_mode={'masked' if action_masking else 'unmasked'}; "
                f"stage {current_stage_idx + 1}/{len(stages)} {stage.name}, "
                f"seeds={stage.seed_pool}, timesteps={stage.timesteps}"
            )
        epsilon = 0.0 if algo == "rainbow_dqn" else epsilon_by_step(step, hp)
        if algo == "rainbow_dqn":
            reset_noisy_layers(online_net)
        stored_mask = policy_action_mask(mask, action_masking)
        action = select_action(online_net, obs, stored_mask, epsilon, rng, device)
        next_obs, reward, done, next_mask, transition_info = step_env(env, action)
        if time_mode == "event_driven":
            delta_t = float(transition_info["delta_t"])
            bootstrap_discount = float(transition_info["transition_discount"])
        else:
            delta_t = reference_interval_s
            bootstrap_discount = float(hp.gamma)
        training_simulated_seconds += float(transition_info.get("delta_t", delta_t))
        raw_transition = NStepTransition(
            obs=np.asarray(obs, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_obs=np.asarray(next_obs, dtype=np.float32),
            terminated=bool(transition_info.get("terminated", done)),
            truncated=bool(transition_info.get("truncated", False)),
            action_mask=np.asarray(stored_mask, dtype=bool),
            next_action_mask=np.asarray(policy_action_mask(next_mask, action_masking), dtype=bool),
            delta_t=delta_t,
            bootstrap_discount=bootstrap_discount,
            decision_time=float(transition_info.get("decision_time", 0.0)),
            trigger_event_type="|".join(transition_info.get("trigger_events", [])),
            instantaneous_dispatch=bool(transition_info.get("instantaneous_dispatch", False)),
        )
        for stored_transition in n_step_accumulator.append(raw_transition):
            replay.add(
                stored_transition.obs,
                stored_transition.action,
                stored_transition.reward,
                stored_transition.next_obs,
                stored_transition.done,
                stored_transition.action_mask,
                stored_transition.next_action_mask,
                terminated=stored_transition.terminated,
                truncated=stored_transition.truncated,
                delta_t=stored_transition.delta_t,
                transition_discount=stored_transition.bootstrap_discount,
                decision_time=stored_transition.decision_time,
                trigger_event_type=stored_transition.trigger_event_type,
                instantaneous_dispatch=stored_transition.instantaneous_dispatch,
            )

        obs = next_obs
        mask = next_mask
        episode_reward += reward
        episode_steps += 1

        if (
            step >= hp.learning_starts
            and len(replay) >= hp.batch_size
            and step % max(1, int(hp.train_frequency)) == 0
        ):
            if algo == "rainbow_dqn":
                reset_noisy_layers(online_net)
                batch = replay.sample(hp.batch_size, device, beta=prioritized_beta_by_step(step, hp))
            else:
                batch = replay.sample(hp.batch_size, device)
            weights = batch.weights if algo == "rainbow_dqn" else None
            loss, td_errors = compute_dqn_loss(
                algo,
                online_net,
                target_net,
                batch,
                hp.gamma,
                weights=weights,
                action_masking=action_masking,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online_net.parameters(), hp.max_grad_norm)
            optimizer.step()
            gradient_updates += 1
            if algo == "rainbow_dqn" and batch.indices is not None:
                replay.update_priorities(batch.indices, td_errors)
            for duration, error in zip(
                batch.delta_ts.detach().cpu().numpy(),
                td_errors.detach().abs().cpu().numpy(),
            ):
                td_error_diagnostics[td_error_delta_t_bin(float(duration))].append(float(error))
            latest_loss = float(loss.item())

        if step % hp.target_update_interval == 0:
            target_net.load_state_dict(online_net.state_dict())

        if done:
            metrics = env.sim.metrics(algo).to_dict()
            row = train_metric_row(
                step=step,
                episode=episode_idx,
                episode_steps=episode_steps,
                episode_reward=episode_reward,
                epsilon=epsilon,
                latest_loss=latest_loss,
                metrics=metrics,
                seed_mode=resolved_seed_mode,
                stage=current_stage_idx + 1,
                stage_name=stages[current_stage_idx].name,
            )
            training_rows.append(row)
            row["simulated_time_s"] = float(env.sim.now)
            row["decisions_per_episode"] = int(getattr(env, "decision_index", episode_steps))
            row["gradient_updates"] = int(gradient_updates)
            row["replay_transitions"] = int(len(replay))
            row["training_simulated_seconds"] = float(training_simulated_seconds)
            if writer is not None:
                write_train_scalars(writer, row, step)
            episode_idx += 1
            obs, mask = reset_env(env)
            episode_reward = 0.0
            episode_steps = 0

        should_eval = step % hp.eval_interval == 0 or step == effective_timesteps
        if should_eval:
            eval_df = evaluate_policy(
                cfg,
                online_net,
                device,
                algo,
                step,
                eval_seeds,
                eval_scenario,
                action_masking=action_masking,
            )
            eval_rows.extend(eval_df.to_dict("records"))
            eval_mean = eval_df[["completion_rate", "timeout_rate", "episode_reward"]].mean(numeric_only=True)
            print(
                f"[{algo}] step={step} epsilon={epsilon:.3f} "
                f"completion={eval_mean.get('completion_rate', np.nan):.3f} "
                f"timeout={eval_mean.get('timeout_rate', np.nan):.3f} "
                f"eval_reward={eval_mean.get('episode_reward', np.nan):.2f}"
            )

    if episode_steps > 0:
        metrics = env.sim.metrics(algo).to_dict()
        row = train_metric_row(
            step=effective_timesteps,
            episode=episode_idx,
            episode_steps=episode_steps,
            episode_reward=episode_reward,
            epsilon=epsilon_by_step(effective_timesteps, hp),
            latest_loss=latest_loss,
            metrics=metrics,
            partial_episode=True,
            seed_mode=resolved_seed_mode,
            stage=current_stage_idx + 1,
            stage_name=stages[current_stage_idx].name,
        )
        training_rows.append(row)
        if writer is not None:
            write_train_scalars(writer, row, effective_timesteps)

    save_outputs(
        out_dir=out_dir,
        run_dir=run_dir,
        algo=algo,
        seed=base_seed,
        cfg_path=cfg_path,
        hp=hp,
        online_net=online_net,
        optimizer=optimizer,
        obs_dim=obs_dim,
        action_dim=action_dim,
        training_rows=training_rows,
        eval_rows=eval_rows,
        total_timesteps=effective_timesteps,
        model_name=model_name,
        train_scenario=train_scenario,
        tensorboard_log_dir=tensorboard_log_dir,
        seed_mode=resolved_seed_mode,
        stages=[
            {
                "name": stage.name,
                "timesteps": int(stage.timesteps),
                "seeds": stage.seeds,
                "random_seed": bool(stage.random_seed),
            }
            for stage in stages
        ],
        action_masking=action_masking,
        replay_config={
            **replay.state_dict(),
            "n_step": n_step,
            "reference_interval_s": reference_interval_s,
            "gamma_reference": float(hp.gamma),
            "discount_mode": str(smdp_cfg.get("discount_mode", "fixed_decision")),
        },
    )
    pd.DataFrame(
        [
            {
                "delta_t_bin": key,
                "mean_td_error": float(np.mean(values)) if values else np.nan,
                "sample_count": len(values),
            }
            for key, values in td_error_diagnostics.items()
        ]
    ).to_csv(run_dir / "td_error_by_delta_t.csv", index=False)
    with open(run_dir / "replay_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **replay.state_dict(),
                "n_step": n_step,
                "reference_interval_s": reference_interval_s,
                "gamma_reference": float(hp.gamma),
                "discount_mode": str(smdp_cfg.get("discount_mode", "fixed_decision")),
                "gradient_updates": gradient_updates,
                "training_simulated_seconds": training_simulated_seconds,
            },
            f,
            indent=2,
        )
    if writer is not None:
        writer.flush()
        writer.close()
    return run_dir / model_name


def parse_eval_seeds(value) -> List[int]:
    if value is None:
        return [1000, 1001, 1002, 1003, 1004]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def build_hyperparams(args: argparse.Namespace, cfg: Dict | None = None) -> DQNHyperParams:
    values = asdict(DQNHyperParams())
    rl_cfg = (cfg or {}).get("rl", {})
    config_hyperparams = rl_cfg.get("hyperparams", rl_cfg.get("dqn_hyperparams", {})) or {}
    values.update({k: v for k, v in config_hyperparams.items() if k in values})
    if args.params:
        root = project_root()
        params_path = Path(args.params)
        if not params_path.is_absolute():
            params_path = root / params_path
        with open(params_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        values.update({k: v for k, v in loaded.items() if k in values})
    for key in values:
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            values[key] = cli_value
    return DQNHyperParams(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scenario_60node.yaml")
    parser.add_argument("--algo", choices=ALGORITHMS, default=None, help="default: rl.algo from config, else dqn")
    parser.add_argument("--timesteps", type=int, default=None, help="override total training timesteps")
    parser.add_argument("--seed", type=int, default=None, help="base seed for fixed/curriculum sampling")
    parser.add_argument("--seed-mode", choices=SEED_MODES, default=None, help="default: training.seed_mode from config, else fixed")
    parser.add_argument("--stage-timesteps", default=None, help="comma-separated stage timesteps; one value or one per stage")
    parser.add_argument("--out-dir", default=None, help="default: results/rl/dqn")
    parser.add_argument("--run-name", default=None, help="default: <algo>_seed<seed>; used for CSV and TensorBoard subdirs")
    parser.add_argument("--tensorboard-dir", default=None, help="default: results/rl/tensorboard/dqn")
    parser.add_argument("--disable-tensorboard", action="store_true", help="train without writing TensorBoard event files")
    parser.add_argument("--unmasked", action="store_true", help="train without using the action mask for selection or target Q values")
    parser.add_argument("--model-name", default=None, help="output model filename under results/rl/dqn/<run-name>")
    parser.add_argument("--params", default=None, help="optional DQN hyperparameter YAML file")
    parser.add_argument("--eval-seeds", default=None)
    parser.add_argument("--scenario", default=None, help="training scenario label passed to DynamicPatrolEnv")
    parser.add_argument("--eval-scenario", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--replay-buffer-size", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--train-frequency", type=int, default=None)
    parser.add_argument("--target-update-interval", type=int, default=None)
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--epsilon-final", type=float, default=None)
    parser.add_argument("--epsilon-decay-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--candidate-dim", type=int, default=None)
    parser.add_argument("--prioritized-replay-alpha", type=float, default=None)
    parser.add_argument("--prioritized-replay-beta-start", type=float, default=None)
    parser.add_argument("--prioritized-replay-beta-final", type=float, default=None)
    parser.add_argument("--prioritized-replay-beta-steps", type=int, default=None)
    parser.add_argument("--priority-epsilon", type=float, default=None)
    args = parser.parse_args()

    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(cfg_path)
    rl_cfg = cfg.get("rl", {})
    algo = args.algo or rl_cfg.get("algo", "dqn")
    run_name = args.run_name or rl_cfg.get("run_name")
    model_name = args.model_name or rl_cfg.get("model_name")
    total_timesteps = args.timesteps if args.timesteps is not None else rl_cfg.get("timesteps")
    eval_seeds = parse_eval_seeds(args.eval_seeds if args.eval_seeds is not None else rl_cfg.get("eval_seeds"))
    train_scenario = args.scenario or rl_cfg.get("train_scenario", "small")
    eval_scenario = args.eval_scenario or rl_cfg.get("eval_scenario", "eval")
    config_action_masking = bool(rl_cfg.get("action_masking", True))
    action_masking = False if args.unmasked else config_action_masking
    out_dir, tensorboard_dir = resolve_dqn_output_dirs(root, args.out_dir, args.tensorboard_dir)

    model_path = train_masked_dqn(
        cfg=cfg,
        cfg_path=str(cfg_path),
        algo=algo,
        total_timesteps=total_timesteps,
        seed=args.seed,
        out_dir=out_dir,
        hp=build_hyperparams(args, cfg),
        eval_seeds=eval_seeds,
        eval_scenario=eval_scenario,
        device_name=args.device,
        model_name=model_name,
        train_scenario=train_scenario,
        run_name=run_name,
        tensorboard_dir=tensorboard_dir,
        enable_tensorboard=not args.disable_tensorboard,
        seed_mode=args.seed_mode,
        stage_timesteps=parse_int_list(args.stage_timesteps),
        action_masking=action_masking,
    )
    run_dir = model_path.parent
    run_name = run_dir.name
    print(f"Saved model: {model_path}")
    print(f"Saved curves: {run_dir / 'training_curve.csv'}, {run_dir / 'eval_curve.csv'}")
    print(f"Saved summary: {run_dir / 'summary.csv'}")
    print(f"Saved summary index: {out_dir / 'summary.csv'}")
    print(f"Saved TensorBoard logs: {tensorboard_dir / run_name}")


if __name__ == "__main__":
    main()
