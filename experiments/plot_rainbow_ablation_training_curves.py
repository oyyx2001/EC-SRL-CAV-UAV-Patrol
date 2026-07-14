from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import pandas as pd

from dynamic_patrol_marl.utils.config import project_root


@dataclass(frozen=True)
class CurveSpec:
    label: str
    run_dir: str
    color: str


CURVES: List[CurveSpec] = [
    CurveSpec("Full Masked Rainbow DQN", "rainbow_full_masked_60node", "#F44336"),
    CurveSpec("w/o Candidate-sortie Features", "rainbow_wo_candidate_sortie_features_60node", "#2196F3"),
    CurveSpec("w/o Multi-task Sorties", "rainbow_max_tasks_1_60node", "#4CAF50"),
    CurveSpec("w/o Priority-urgency Score", "rainbow_wo_priority_urgency_score_60node", "#FF9800"),
]


METRICS = [
    ("episode_reward", "Reward", "(a)"),
    ("completion_rate", "Completion rate", "(b)"),
    ("timeout_rate", "Timeout rate", "(c)"),
]


def _step_formatter(value: float, _pos: int) -> str:
    if value >= 1_000_000:
        if abs(value % 1_000_000) < 1e-6:
            return f"{value / 1_000_000:.0f}M"
        return f"{value / 1_000_000:.1f}M"
    if value >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{value:.0f}"


def _smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(3, window // 5), center=True).mean()


def _lighten_color(color: str, amount: float = 0.72) -> str:
    rgb = mcolors.to_rgb(color)
    return mcolors.to_hex(tuple((1.0 - amount) * channel + amount for channel in rgb))


def _read_training_curve(base_dir: Path, spec: CurveSpec) -> pd.DataFrame:
    path = base_dir / spec.run_dir / "training_curve.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing training curve for {spec.label}: {path}")
    df = pd.read_csv(path)
    required = {"step", "episode_reward", "completion_rate", "timeout_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.sort_values("step").copy()
    return df


def plot_ablation_training_curves(
    base_dir: Path,
    out_path: Path,
    specs: Iterable[CurveSpec] = CURVES,
    smooth_window: int = 135,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.linewidth"] = 1.35

    fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.9), sharex=True)
    all_steps = []
    loaded = []
    for spec in specs:
        df = _read_training_curve(base_dir, spec)
        loaded.append((spec, df))
        all_steps.extend(pd.to_numeric(df["step"], errors="coerce").dropna().tolist())

    y_limits = {
        "episode_reward": (-120, 360),
        "completion_rate": (0.68, 0.88),
        "timeout_rate": (0.06, 0.18),
    }

    for ax, (metric, ylabel, panel_label) in zip(axes, METRICS):
        for spec, df in loaded:
            x = pd.to_numeric(df["step"], errors="coerce")
            y = pd.to_numeric(df[metric], errors="coerce")
            valid = x.notna() & y.notna()
            x = x[valid]
            y = y[valid]
            ax.plot(x, y, color=_lighten_color(spec.color), alpha=0.18, linewidth=0.48)
            ax.plot(x, _smooth(y, smooth_window), color=spec.color, alpha=0.98, linewidth=1.72, label=spec.label)

        ax.set_ylabel(ylabel, fontsize=20)
        ax.text(
            0.5,
            -0.34,
            f"{panel_label} {ylabel}",
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            ha="center",
            va="top",
        )
        ax.grid(True, color="#bfbfbf", linewidth=1.05, alpha=0.9)
        ax.tick_params(axis="both", labelsize=20, width=1.25)
        for spine in ax.spines.values():
            spine.set_linewidth(1.35)
            spine.set_color("black")
        if metric in {"completion_rate", "timeout_rate"}:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(*y_limits[metric])

    max_step = max(all_steps) if all_steps else 1.0
    for ax in axes:
        ax.set_xlim(0, max_step)
        ax.xaxis.set_major_formatter(FuncFormatter(_step_formatter))
        ax.set_xlabel("Timestep", fontsize=20, labelpad=5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        prop={"family": "Times New Roman", "size": 20},
        columnspacing=1.8,
        handlelength=2.4,
        handletextpad=0.6,
    )

    fig.subplots_adjust(left=0.07, right=0.995, top=0.82, bottom=0.32, wspace=0.30)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Rainbow DQN ablation training curves.")
    parser.add_argument("--base-dir", default="results/rl/rainbow_dqn_ablation")
    parser.add_argument("--out", default="results/rl/rainbow_dqn_ablation/ablation_training_curves_3panel.png")
    parser.add_argument("--smooth-window", type=int, default=135)
    args = parser.parse_args()

    root = project_root()
    base_dir = Path(args.base_dir)
    if not base_dir.is_absolute():
        base_dir = root / base_dir
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    saved = plot_ablation_training_curves(base_dir, out_path, smooth_window=int(args.smooth_window))
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
