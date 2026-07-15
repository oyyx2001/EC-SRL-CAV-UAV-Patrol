from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from dynamic_patrol_marl.rl.env_wrapper import DynamicPatrolEnv
from dynamic_patrol_marl.utils.config import load_config, project_root


@dataclass(frozen=True)
class ModelSpec:
    label: str
    family: str
    path: Path
    algo: str | None = None
    action_masking: bool = True


DEFAULT_MODELS: Dict[str, ModelSpec] = {
    "maskppo": ModelSpec(
        label="maskppo",
        family="ppo",
        path=Path("results/rl/MaskPPO_trained_model_60node_random_seed.zip"),
    ),
    "maskdqn": ModelSpec(
        label="maskdqn",
        family="dqn",
        path=Path("results/rl/MaskedDQN_trained_model_60node_random_seed.zip"),
        algo="dqn",
    ),
    "mask_rainbow_dqn": ModelSpec(
        label="mask_rainbow_dqn",
        family="dqn",
        path=Path("results/rl/RainbowDQN_trained_model_60node.pt"),
        algo="rainbow_dqn",
    ),
    "mask_double_dqn": ModelSpec(
        label="mask_double_dqn",
        family="dqn",
        path=Path("results/rl/MaskedDoubleDQN_trained_model_60node_ramdom_seed.zip"),
        algo="double_dqn",
    ),
    "unmask_ppo": ModelSpec(
        label="unmask_ppo",
        family="ppo",
        path=Path("results/rl/unmasked_ppo/UnmaskPPO_trained_model_60node.zip"),
        action_masking=False,
    ),
    "unmask_rainbow_dqn": ModelSpec(
        label="unmask_rainbow_dqn",
        family="dqn",
        path=Path("results/rl/dqn/unmasked_rainbow_dqn_60node/UnmaskRainbowDQN_trained_model_60node.pt"),
        algo="rainbow_dqn",
        action_masking=False,
    ),
}
DEFAULT_MODEL_KEYS = ["maskppo", "maskdqn", "mask_rainbow_dqn", "mask_double_dqn"]

TABLE_METRICS = [
    ("completion_rate", "Completion", 3),
    ("timeout_rate", "Timeout", 3),
    ("avg_response_time_s", "Avg. response (s)", 1),
    ("high_priority_response_time_s", "High-priority response (s)", 1),
    ("weighted_aoi", "Weighted-AoI", 1),
    ("total_energy", "Energy", 0),
    ("avg_tasks_per_sortie", "Tasks/sortie", 3),
]


def _resolve(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _load_ppo(model_path: Path, action_masking: bool) -> Any:
    try:
        if action_masking:
            from sb3_contrib import MaskablePPO
            return MaskablePPO.load(model_path)
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    except Exception as exc:  # pragma: no cover - optional dependency message.
        raise RuntimeError(
            "PPO evaluation requires optional RL dependencies: "
            "torch, gymnasium, stable-baselines3, and sb3-contrib."
        ) from exc


def _torch_load(model_path: Path, device: Any) -> Any:
    import torch

    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:  # Older PyTorch does not expose weights_only.
        return torch.load(model_path, map_location=device)


def _load_masked_dqn(
    model_path: Path,
    algo_override: str | None,
    device_name: str,
    action_masking: bool,
) -> Dict[str, Any]:
    try:
        import torch
        from dynamic_patrol_marl.rl.dqn_networks import make_q_network, masked_q_values
    except Exception as exc:  # pragma: no cover - optional dependency message.
        raise RuntimeError("Masked DQN evaluation requires PyTorch.") from exc

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    checkpoint = _torch_load(model_path, device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        algo = algo_override or checkpoint.get("algo")
        hyperparams = checkpoint.get("hyperparams", {}) or {}
        obs_dim = int(checkpoint["obs_dim"])
        action_dim = int(checkpoint["action_dim"])
        candidate_dim = int(checkpoint.get("candidate_dim", hyperparams.get("candidate_dim", 6)))
        state_dict = checkpoint["model_state_dict"]
        replay_config = checkpoint.get("replay_config", {}) or {}
        if int(replay_config.get("schema_version", 1)) < 2:
            warnings.warn(
                "Loading a legacy fixed-interval checkpoint without SMDP replay metadata. "
                "It may be used for an event-driven smoke test only; decision-frequency "
                "and state-distribution shift make it unsuitable as a final performance result.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        raise ValueError(
            f"{model_path} is not a supported Masked DQN checkpoint. "
            "Expected keys: model_state_dict, obs_dim, action_dim."
        )

    if not algo:
        raise ValueError(f"Cannot infer DQN algorithm for {model_path}; pass the correct model argument.")
    network = make_q_network(str(algo), obs_dim, action_dim, candidate_dim=candidate_dim).to(device)
    network.load_state_dict(state_dict)
    network.eval()
    return {
        "network": network,
        "torch": torch,
        "device": device,
        "masked_q_values": masked_q_values,
        "action_masking": bool(checkpoint.get("action_masking", action_masking)),
        "replay_config": replay_config,
        "legacy_fixed_interval_checkpoint": int(replay_config.get("schema_version", 1)) < 2,
    }


def _reset_env(env: DynamicPatrolEnv, seed: int) -> tuple[np.ndarray, Dict[str, Any]]:
    out = env.reset(seed=int(seed))
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {"action_mask": env.action_masks()}
    return np.asarray(obs, dtype=np.float32), dict(info)


def _step_env(env: DynamicPatrolEnv, action: int) -> tuple[np.ndarray, float, bool, Dict[str, Any]]:
    out = env.step(int(action))
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = out
    info = dict(info)
    info.setdefault("action_mask", env.action_masks())
    return np.asarray(obs, dtype=np.float32), float(reward), bool(done), info


def _predict_dqn(loaded: Dict[str, Any], obs: np.ndarray, action_mask: np.ndarray) -> int:
    torch = loaded["torch"]
    device = loaded["device"]
    network = loaded["network"]
    masked_q_values = loaded["masked_q_values"]
    policy_mask = action_mask if loaded.get("action_masking", True) else np.ones_like(action_mask, dtype=bool)
    with torch.no_grad():
        obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
        mask_t = torch.as_tensor(policy_mask[None, :], device=device, dtype=torch.bool)
        q_values = network(obs_t, mask_t)
        return int(masked_q_values(q_values, mask_t).argmax(dim=1).item())


def _evaluate_spec(
    cfg: Dict[str, Any],
    spec: ModelSpec,
    seeds: Iterable[int],
    scenario: str,
    device_name: str,
) -> pd.DataFrame:
    if spec.family == "ppo":
        model = _load_ppo(spec.path, spec.action_masking)
    elif spec.family == "dqn":
        model = _load_masked_dqn(spec.path, spec.algo, device_name, spec.action_masking)
    else:
        raise ValueError(f"Unknown model family: {spec.family}")

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        env = DynamicPatrolEnv(cfg, scenario=scenario)
        obs, info = _reset_env(env, int(seed))
        done = False
        episode_reward = 0.0
        while not done:
            action_mask = np.asarray(info["action_mask"], dtype=bool)
            if spec.family == "ppo":
                if spec.action_masking:
                    action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = _predict_dqn(model, obs, action_mask)
            obs, reward, done, info = _step_env(env, action)
            episode_reward += reward

        row = env.sim.metrics(spec.label).to_dict()
        row["method"] = spec.label
        row["seed"] = int(seed)
        row["episode_reward"] = float(episode_reward)
        row["model_path"] = str(spec.path)
        rows.append(row)
    return pd.DataFrame(rows)


def _format_mean_sd(values: pd.Series, decimals: int) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return ""
    mean = float(numeric.mean())
    sd = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
    return f"{mean:.{decimals}f} ({sd:.{decimals}f})"


def make_mean_sd_table(df: pd.DataFrame, method_order: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for method in method_order:
        group = df[df["method"] == method]
        if group.empty:
            continue
        row = {"Method": method}
        for metric, title, decimals in TABLE_METRICS:
            row[title] = _format_mean_sd(group[metric], decimals) if metric in group.columns else ""
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown_table(table: pd.DataFrame, path: Path) -> None:
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_model_specs(args: argparse.Namespace, root: Path) -> List[ModelSpec]:
    selected = args.models or DEFAULT_MODEL_KEYS
    overrides = {
        "maskppo": args.maskppo_model,
        "maskdqn": args.maskdqn_model,
        "mask_rainbow_dqn": args.mask_rainbow_dqn_model,
        "mask_double_dqn": args.mask_double_dqn_model,
        "unmask_ppo": args.unmask_ppo_model,
        "unmask_rainbow_dqn": args.unmask_rainbow_dqn_model,
    }
    specs: List[ModelSpec] = []
    for key in selected:
        spec = DEFAULT_MODELS[key]
        model_path = _resolve(overrides.get(key) or spec.path, root)
        specs.append(
            ModelSpec(
                label=spec.label,
                family=spec.family,
                path=model_path,
                algo=spec.algo,
                action_masking=spec.action_masking,
            )
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained masked or unmasked PPO/DQN models on the 60-node scenario."
    )
    parser.add_argument("--config", default="configs/scenario_60node.yaml")
    parser.add_argument("--scenario", default="small", help="Use 'small' to match the baseline script default.")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", nargs="*", choices=list(DEFAULT_MODELS), default=None)
    parser.add_argument("--maskppo-model", default=None)
    parser.add_argument("--maskdqn-model", default=None)
    parser.add_argument("--mask-rainbow-dqn-model", default=None)
    parser.add_argument("--mask-double-dqn-model", default=None)
    parser.add_argument("--unmask-ppo-model", default=None)
    parser.add_argument("--unmask-rainbow-dqn-model", default=None)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--out", default="results/rl/eval_masked_rl_models_30seeds.csv")
    parser.add_argument("--summary-out", default="results/rl/eval_masked_rl_models_30seeds_mean_sd_table.csv")
    parser.add_argument("--markdown-out", default="results/rl/eval_masked_rl_models_30seeds_mean_sd_table.md")
    args = parser.parse_args()

    root = project_root()
    cfg_path = _resolve(args.config, root)
    out_path = _resolve(args.out, root)
    summary_path = _resolve(args.summary_out, root)
    markdown_path = _resolve(args.markdown_out, root)
    cfg = load_config(cfg_path)

    specs = build_model_specs(args, root)
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))
    frames: List[pd.DataFrame] = []
    evaluated_methods: List[str] = []
    for spec in specs:
        if not spec.path.exists():
            message = f"Missing model for {spec.label}: {spec.path}"
            if args.skip_missing:
                print(message)
                continue
            raise FileNotFoundError(message)
        print(f"Evaluating {spec.label}: {spec.path}")
        frames.append(_evaluate_spec(cfg, spec, seeds, args.scenario, args.device))
        evaluated_methods.append(spec.label)

    if not frames:
        raise RuntimeError("No models were evaluated.")

    df = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    table = make_mean_sd_table(df, evaluated_methods)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(summary_path, index=False)
    write_markdown_table(table, markdown_path)

    print(table.to_string(index=False))
    print(f"Saved per-seed results: {out_path}")
    print(f"Saved Mean(SD) table: {summary_path}")
    print(f"Saved markdown table: {markdown_path}")


if __name__ == "__main__":
    main()
