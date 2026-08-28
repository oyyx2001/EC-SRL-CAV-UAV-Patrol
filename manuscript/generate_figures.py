from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def save_framework() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 3.5), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.02, 0.56, 0.15, 0.27, "Event-driven\nurban system", "#E8F1F8"),
        (0.22, 0.56, 0.17, 0.27, "Candidate-sortie\ngeneration", "#E8F5EE"),
        (0.44, 0.56, 0.13, 0.27, "Feasibility\nmask", "#FFF3D8"),
        (0.62, 0.56, 0.16, 0.27, "Candidate-aware\nDQN", "#F0EAF8"),
        (0.83, 0.56, 0.15, 0.27, "Atomic plan\ncommit", "#FCE8E6"),
    ]
    for x, y, w, h, label, color in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.2, edgecolor="#263238", facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10.5)

    for x0, x1 in [(0.17, 0.22), (0.39, 0.44), (0.57, 0.62), (0.78, 0.83)]:
        ax.annotate("", xy=(x1, 0.695), xytext=(x0, 0.695),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#455A64"))

    ax.text(0.305, 0.37, "Up to 3 tasks per sortie\nqueue-aware rendezvous", ha="center", fontsize=9, color="#37474F")
    ax.text(0.505, 0.37, "25 fixed slots\n38 features per plan", ha="center", fontsize=9, color="#37474F")
    ax.text(0.70, 0.37, "SMDP discount\nvalid-action Bellman target", ha="center", fontsize=9, color="#37474F")
    ax.text(0.905, 0.37, "CAV route + UAV sortie\nrevalidated before dispatch", ha="center", fontsize=9, color="#37474F")

    ax.annotate(
        "",
        xy=(0.095, 0.53), xytext=(0.905, 0.25),
        arrowprops=dict(arrowstyle="-|>", lw=1.25, color="#607D8B", connectionstyle="arc3,rad=-0.10"),
    )
    ax.text(0.50, 0.10, "Elapsed-time reward, task updates, vehicle/UAV state, and recovery-queue feedback",
            ha="center", fontsize=9.5, color="#37474F")
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "framework.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_ablation_summary() -> None:
    path = ROOT / "results" / "ablations" / "core_20k_repaired" / "ablation_summary.csv"
    data = pd.read_csv(path)
    variants = ["full", "fixed_discount", "single_task", "queue_blind"]
    labels = ["Full", "Fixed discount", "Single task", "Queue blind"]
    metrics = [
        ("completion_rate", "Completion rate", 100.0, "%"),
        ("high_priority_response_time_s", "High-priority response", 1.0, "s"),
        ("avg_response_time", "Mean response", 1.0, "s"),
        ("mean_uav_recovery_wait_s", "Mean recovery wait", 1.0, "s"),
    ]
    colors = ["#205493", "#5C85B5", "#C68B3C", "#A64B4B"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), dpi=220)
    for ax, (metric, title, scale, unit) in zip(axes.ravel(), metrics):
        sub = data[data["metric"] == metric].set_index("variant").loc[variants]
        means = sub["mean"].to_numpy(float) * scale
        errs = sub["train_seed_sd"].to_numpy(float) * scale
        x = np.arange(len(variants))
        ax.bar(x, means, yerr=errs, capsize=3, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "completion_rate":
            ax.set_ylim(90, 98)
    fig.suptitle("20k-transition ablation screening (mean ± SD across three training seeds)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "ablation_summary.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_discount_tradeoff() -> None:
    path = ROOT / "results" / "xinjiekou_333" / "smdp_time_aware_discount" / "formal_100k" / "formal_100k_paired_differences.csv"
    data = pd.read_csv(path).set_index("metric")
    metrics = [
        "avg_response_time",
        "weighted_aoi",
        "uav_energy_per_completed_task",
        "mean_uav_recovery_wait_s",
        "queue_occupancy_time_s",
        "cav_travel_distance_m",
    ]
    labels = [
        "Mean response time",
        "Weighted AoI",
        "UAV energy/completed task",
        "Mean UAV recovery wait",
        "Recovery-queue occupancy",
        "CAV travel distance",
    ]
    fixed = data.loc[metrics, "fixed_mean"].to_numpy(float)
    diff = data.loc[metrics, "difference_time_aware_minus_fixed"].to_numpy(float)
    low = data.loc[metrics, "paired_bootstrap_ci_low"].to_numpy(float)
    high = data.loc[metrics, "paired_bootstrap_ci_high"].to_numpy(float)
    pct = 100 * diff / fixed
    low_pct = 100 * low / fixed
    high_pct = 100 * high / fixed
    y = np.arange(len(metrics))[::-1]
    colors = np.where(pct <= 0, "#2A7F62", "#B4473D")
    fig, ax = plt.subplots(figsize=(8.3, 4.4), dpi=220)
    ax.axvline(0, color="#263238", linewidth=1)
    for yi, p, lo, hi, color in zip(y, pct, low_pct, high_pct, colors):
        ax.errorbar(p, yi, xerr=[[p - lo], [hi - p]], fmt="o", color=color,
                    ecolor=color, capsize=3, markersize=6, linewidth=1.5)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Change under time-aware discount relative to fixed per-decision discount (%)")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("100k-transition single-training-seed comparison\n(error bars: paired bootstrap over 20 evaluation seeds)",
                 fontsize=11, fontweight="bold")
    ax.text(0.01, -0.22, "Green denotes improvement for lower-is-better metrics; red denotes a cost increase.",
            transform=ax.transAxes, fontsize=8.5, color="#455A64")
    fig.tight_layout()
    fig.savefig(OUT / "discount_tradeoff.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    save_framework()
    save_ablation_summary()
    save_discount_tradeoff()
    print(f"Wrote figures to {OUT}")
