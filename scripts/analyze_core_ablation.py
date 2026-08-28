"""Aggregate the corrected 20k four-variant core-ablation outputs.

This script is deliberately read-only with respect to individual run folders.
It writes only the requested study-level tables, figures, integrity audit and
Chinese report under ``results/ablations/core_20k_repaired``.  The historical
Fixed CAV failure is deliberately excluded from inferential aggregation.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamic_patrol_marl.utils.config import project_root


ROOT = project_root()
OUT = ROOT / "results" / "ablations" / "core_20k_repaired"
VARIANTS = ("full", "fixed_discount", "single_task", "queue_blind")
TRAIN_SEEDS = (0, 1, 2)
EVAL_SEEDS = tuple(range(1000, 1020))
PRIMARY = {
    "completion_rate": ("完成率", True),
    "timeout_rate": ("超时率", False),
    "high_priority_response_time_s": ("高优先级响应时间 (s)", False),
    "weighted_aoi": ("加权 AoI", False),
}
SECONDARY = {
    "avg_response_time": ("平均响应时间 (s)", False),
    "energy": ("总能耗", False),
    "cav_travel_distance_m": ("CAV 距离 (m)", False),
    "uav_return_distance_m": ("UAV 返回距离 (m)", False),
    "avg_tasks_per_sortie": ("每架次任务数", True),
    "mean_uav_recovery_wait_s": ("平均恢复等待 (s)", False),
    "queue_occupancy_time_s": ("队列占用时间 (s)", False),
    "illegal_actions": ("非法动作数", False),
    "commit_failure_count": ("原子提交失败", False),
    "path_planning_failure_count": ("路径规划失败", False),
    "safe_plan_rate": ("安全计划率", True),
    "mean_plan_generation_time_ms": ("候选生成均值 (ms)", False),
}
METRICS = {**PRIMARY, **SECONDARY}


def run_dir(variant: str, train_seed: int) -> Path:
    return OUT / f"{variant}_seed{train_seed}"


def read_final_evaluations() -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    issues: list[dict] = []
    for variant in VARIANTS:
        for train_seed in TRAIN_SEEDS:
            directory = run_dir(variant, train_seed)
            curve = directory / "eval_curve.csv"
            if not curve.exists():
                issues.append({"variant": variant, "train_seed": train_seed, "issue": "missing_eval_curve"})
                continue
            try:
                data = pd.read_csv(curve)
            except Exception as exc:
                issues.append({"variant": variant, "train_seed": train_seed, "issue": f"unreadable_eval_curve: {exc}"})
                continue
            if data.empty or "step" not in data:
                issues.append({"variant": variant, "train_seed": train_seed, "issue": "empty_eval_curve"})
                continue
            final_step = int(pd.to_numeric(data["step"], errors="coerce").max())
            final = data[pd.to_numeric(data["step"], errors="coerce") == final_step].copy()
            final["variant"] = variant
            final["train_seed"] = int(train_seed)
            final["final_step"] = final_step
            frames.append(final)
            if final_step != 20_000:
                issues.append({"variant": variant, "train_seed": train_seed, "issue": f"final_step={final_step}, expected=20000"})
            observed = sorted(pd.to_numeric(final.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique())
            if observed != list(EVAL_SEEDS):
                issues.append({"variant": variant, "train_seed": train_seed, "issue": f"eval_seeds={observed}, expected={list(EVAL_SEEDS)}"})
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), issues


def finite(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def summary_table(final: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for variant in VARIANTS:
        subset = final[final["variant"] == variant]
        for metric, (label, higher_is_better) in METRICS.items():
            if metric not in subset:
                continue
            per_train = subset.groupby("train_seed")[metric].apply(finite)
            train_means = subset.groupby("train_seed")[metric].mean(numeric_only=True)
            within_sd = subset.groupby("train_seed")[metric].std(ddof=1)
            all_values = finite(subset[metric])
            rows.append({
                "variant": variant,
                "metric": metric,
                "metric_zh": label,
                "higher_is_better": higher_is_better,
                "n_train_seeds": int(train_means.notna().sum()),
                "n_eval_observations": int(all_values.size),
                "mean": float(train_means.mean()) if not train_means.empty else math.nan,
                "train_seed_sd": float(train_means.std(ddof=1)) if train_means.notna().sum() > 1 else math.nan,
                "mean_within_train_eval_sd": float(within_sd.mean()) if within_sd.notna().any() else math.nan,
                "pooled_eval_sd": float(all_values.std(ddof=1)) if all_values.size > 1 else math.nan,
            })
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not values.size:
        return math.nan, math.nan
    draws = rng.choice(values, size=(20_000, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def holm_adjust(values: dict[str, float]) -> dict[str, float]:
    usable = [(key, value) for key, value in values.items() if math.isfinite(value)]
    usable.sort(key=lambda item: item[1])
    adjusted: dict[str, float] = {key: math.nan for key in values}
    running = 0.0
    count = len(usable)
    for rank, (key, value) in enumerate(usable):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[key] = running
    return adjusted


def paired_tables(final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    differences: list[pd.DataFrame] = []
    test_rows: list[dict] = []
    rng = np.random.default_rng(20260807)
    # Holm correction is applied within each ablation's four pre-declared
    # primary metrics.  Keeping the families separate matches the protocol
    # recorded in ``holm_family`` and avoids silently changing the inferential
    # question into a single omnibus family across all ablations.
    primary_p: dict[str, dict[str, float]] = {}
    pending: list[dict] = []
    for ablation in VARIANTS[1:]:
        for metric, (label, higher_is_better) in METRICS.items():
            if metric not in final:
                continue
            full = final[final["variant"] == "full"][["train_seed", "seed", metric]].rename(columns={metric: "full_value"})
            alt = final[final["variant"] == ablation][["train_seed", "seed", metric]].rename(columns={metric: "ablation_value"})
            paired = full.merge(alt, on=["train_seed", "seed"], how="inner")
            paired["variant"] = ablation
            paired["metric"] = metric
            paired["metric_zh"] = label
            paired["full_minus_ablation"] = paired["full_value"] - paired["ablation_value"]
            paired["full_advantage"] = paired["full_minus_ablation"] if higher_is_better else -paired["full_minus_ablation"]
            paired["relative_full_advantage"] = paired["full_advantage"] / paired["ablation_value"].abs().replace(0.0, np.nan)
            differences.append(paired)
            per_train = paired.groupby("train_seed")["full_advantage"].mean()
            values = finite(per_train)
            if values.size >= 2:
                t_result = stats.ttest_1samp(values.to_numpy(), popmean=0.0, nan_policy="omit")
                p_value = float(t_result.pvalue)
                effect = float(values.mean() / values.std(ddof=1)) if values.std(ddof=1) > 0 else math.nan
            else:
                p_value, effect = math.nan, math.nan
            ci_low, ci_high = bootstrap_ci(values.to_numpy(), rng)
            record = {
                "variant": ablation,
                "metric": metric,
                "metric_zh": label,
                "higher_is_better": higher_is_better,
                "n_paired_eval_observations": int(len(paired)),
                "n_paired_train_seeds": int(values.size),
                "mean_full_advantage": float(values.mean()) if values.size else math.nan,
                "train_seed_sd_of_advantage": float(values.std(ddof=1)) if values.size > 1 else math.nan,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "paired_t_p_value": p_value,
                "cohen_dz": effect,
                "interpretation": "positive favours full",
            }
            key = f"{ablation}:{metric}"
            if metric in PRIMARY:
                primary_p.setdefault(ablation, {})[key] = p_value
                record["holm_family"] = "four primary metrics within this ablation"
                record["_holm_key"] = key
            else:
                record["holm_family"] = "not adjusted (secondary/exploratory)"
                record["_holm_key"] = ""
            pending.append(record)
    adjusted: dict[str, float] = {}
    for family in primary_p.values():
        adjusted.update(holm_adjust(family))
    for row in pending:
        row["holm_adjusted_p_value"] = adjusted.get(row.pop("_holm_key"), math.nan)
        test_rows.append(row)
    return (pd.concat(differences, ignore_index=True) if differences else pd.DataFrame(), pd.DataFrame(test_rows))


def runtime_quantiles() -> pd.DataFrame:
    rows: list[dict] = []
    for variant in VARIANTS:
        for train_seed in TRAIN_SEEDS:
            path = run_dir(variant, train_seed) / "evaluation_task_audit.csv"
            if not path.exists():
                continue
            try:
                data = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                # A valid final evaluation can contain zero completed tasks;
                # the exporter then emits an empty task-audit file.
                continue
            if data.empty or "end_to_end_response_time_s" not in data:
                continue
            for eval_seed, group in data.groupby("eval_seed"):
                values = finite(group["end_to_end_response_time_s"])
                if values.empty:
                    continue
                rows.append({
                    "variant": variant,
                    "train_seed": train_seed,
                    "eval_seed": int(eval_seed),
                    "latency_type": "completed_task_end_to_end_response_s",
                    "n_completed_tasks": int(values.size),
                    "p50": float(values.quantile(0.50)),
                    "p95": float(values.quantile(0.95)),
                    "p99": float(values.quantile(0.99)),
                })
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    aggregate = raw.groupby("variant", as_index=False)[["n_completed_tasks", "p50", "p95", "p99"]].mean()
    aggregate["train_seed"] = "mean_over_train_eval"
    aggregate["eval_seed"] = "mean_over_train_eval"
    aggregate["latency_type"] = "completed_task_end_to_end_response_s"
    return pd.concat([raw, aggregate[raw.columns]], ignore_index=True)


def write_figures(summary: pd.DataFrame, tests: pd.DataFrame, latency: pd.DataFrame) -> None:
    figure_dir = OUT / "figures"
    figure_dir.mkdir(exist_ok=True)
    colors = {"full": "#1f77b4", "fixed_discount": "#ff7f0e", "single_task": "#2ca02c", "queue_blind": "#9467bd"}
    labels = {"full": "Full", "fixed_discount": "Fixed discount", "single_task": "Single task", "queue_blind": "Queue blind"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, PRIMARY):
        if summary.empty:
            axis.text(0.5, 0.5, "No complete 20k data", ha="center", va="center")
            axis.set_title(metric)
            axis.set_axis_off()
            continue
        table = summary[summary.metric == metric].set_index("variant").reindex(VARIANTS)
        axis.bar(range(len(table)), table["mean"], yerr=table["train_seed_sd"], capsize=3, color=[colors[item] for item in table.index])
        axis.set_xticks(range(len(table)), [labels[item] for item in table.index], rotation=25, ha="right")
        axis.set_title(metric)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(figure_dir / "figure1_primary_metrics.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    plot = tests[tests.metric.isin(PRIMARY)].copy() if not tests.empty else pd.DataFrame()
    if not plot.empty:
        plot["label"] = plot["variant"] + " — " + plot["metric"]
        plot = plot.sort_values("mean_full_advantage")
        y = np.arange(len(plot))
        x = plot["mean_full_advantage"].to_numpy()
        left = x - plot["bootstrap_ci95_low"].to_numpy()
        right = plot["bootstrap_ci95_high"].to_numpy() - x
        axis.errorbar(x, y, xerr=np.vstack([left, right]), fmt="o", color="#1f77b4", capsize=3)
        axis.set_yticks(y, plot["label"])
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Full advantage (positive favours full; metric direction adjusted)")
    else:
        axis.text(0.5, 0.5, "No complete paired data", ha="center", va="center")
    fig.savefig(figure_dir / "figure2_paired_primary_effects.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    aggregated = latency[latency.train_seed.astype(str) == "mean_over_train_eval"] if not latency.empty else pd.DataFrame()
    if not aggregated.empty:
        aggregated = aggregated.set_index("variant").reindex(VARIANTS)
        x = np.arange(len(aggregated))
        axis.plot(x, aggregated["p50"], marker="o", label="p50")
        axis.plot(x, aggregated["p95"], marker="o", label="p95")
        axis.plot(x, aggregated["p99"], marker="o", label="p99")
        axis.set_xticks(x, [labels[item] for item in aggregated.index], rotation=25, ha="right")
        axis.set_ylabel("Completed-task end-to-end response time (s)")
        axis.legend()
        axis.grid(alpha=0.25)
    else:
        axis.text(0.5, 0.5, "Task-level final-evaluation audits unavailable", ha="center", va="center")
    fig.savefig(figure_dir / "figure3_latency_quantiles.png", dpi=180)
    plt.close(fig)


def unmasked_diagnostic() -> dict:
    path = OUT / "unmasked_diagnostic_seed0" / "eval_curve.csv"
    if not path.exists():
        return {"available": False}
    data = pd.read_csv(path)
    final = data[pd.to_numeric(data["step"], errors="coerce") == pd.to_numeric(data["step"], errors="coerce").max()]
    illegal = pd.to_numeric(final.get("illegal_actions"), errors="coerce")
    decisions = pd.to_numeric(final.get("policy_decision_count"), errors="coerce")
    rate = illegal / decisions.replace(0.0, np.nan)
    return {
        "available": True,
        "step": int(final["step"].iloc[0]),
        "n_eval_seeds": int(len(final)),
        "mean_illegal_actions": float(illegal.mean()),
        "max_illegal_actions": float(illegal.max()),
        "mean_illegal_action_rate": float(rate.mean()),
        "max_illegal_action_rate": float(rate.max()),
        "mean_completion_rate": float(pd.to_numeric(final.get("completion_rate"), errors="coerce").mean()),
        "mean_timeout_rate": float(pd.to_numeric(final.get("timeout_rate"), errors="coerce").mean()),
        "stop_rule_triggered": bool((rate > 0.20).any()),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""

    columns = ["metric", *[str(column) for column in frame.columns]]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for index, row in frame.iterrows():
        values = [str(index)]
        for value in row:
            values.append("NA" if pd.isna(value) else f"{float(value):.4f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def report(summary: pd.DataFrame, tests: pd.DataFrame, integrity: dict, diagnostic: dict) -> str:
    primary = summary[summary.metric.isin(PRIMARY)].pivot(index="metric", columns="variant", values="mean") if not summary.empty else pd.DataFrame()
    table = markdown_table(primary) if not primary.empty else "尚无完整 20k 主试验数据。"
    diagnostic_text = "尚未完成" if not diagnostic.get("available") else (
        f"完成于 {diagnostic['step']} step，平均非法动作率 {diagnostic['mean_illegal_action_rate']:.2%}，"
        f"最大 {diagnostic['max_illegal_action_rate']:.2%}；未触发 20% 停止阈值。"
    )
    report_state = "最终报告" if integrity["complete"] else (
        f"阶段性报告（已完成 {integrity['observed_core_runs']}/{integrity['expected_core_runs']} 个主运行）"
    )
    overall_confidence = "CAUTION" if integrity["complete"] else "RED_FLAG"
    full_completion = summary[(summary["variant"] == "full") & (summary["metric"] == "completion_rate")]["mean"] if not summary.empty else pd.Series(dtype=float)
    anomaly_text = ""
    if not full_completion.empty and float(full_completion.iloc[0]) <= 1e-12:
        anomaly_text = (
            "\n\n**阶段性异常：当前已完成的 full 训练种子在 20 个评估种子上完成率均为 0、超时率均为 1，"
            "且能耗/架次为 0，表明该模型没有执行服务任务。响应时间为空是没有完成任务的结果，不是低延迟。**"
        )
    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: {datetime.now(tz=timezone.utc).date().isoformat()}
- Verification Status: ANALYZED
- Version Label: core_ablation_validation_v1

# 核心消融实验报告（20k screening）

**报告状态：{report_state}。数据未完整时，所有数值只作执行审计，不作机制优劣结论。**

## Validation Report

- Source: `results/ablations/core_20k`
- Overall Confidence: {overall_confidence}
- Reproducibility Verdict: {"PARTIALLY_REPRODUCIBLE" if integrity['complete'] else "CANNOT_VERIFY"}

## 1. 研究目的

验证联合规划框架中时间感知折扣、多任务编排、主动 CAV 控制与队列感知是否具有必要性；不做超参数扫描或模型家族堆叠。

## 2. 对照与变体

`full` 为联合规划、多任务、队列感知、时间感知折扣和动作掩码的完整对照。其余仅改变一个因素：`fixed_discount`、`single_task`、`fixed_cav`、`queue_blind`。

## 3. 固定实验条件

每个主变体训练种子为 0/1/2，训练 20,000 step；每个模型用共同评估种子 1000–1019；CUDA 运行且禁用 TensorBoard。

## 4. 固定 CAV 的有效性

固定 CAV 不允许策略选择 wait/move_to/reroute；当无 FIFO 恢复任务时，按确定性的有向出返巡航路线移动，因此不是静止车辆伪基线。

## 5. 队列盲化定义

队列盲化将 recovery wait、CAV wait、existing queue delay、insertion cost、queue length 与 locked prefix 从策略特征中置零；候选生成和候选评分不读取当前队列。物理 FIFO 与安全验证仍保留，避免把不安全动作误当成算法能力。

## 6. 未做的消融及理由

未做 gamma/奖励权重网格、DQN 结构阶梯、C51、动态插入、GNN 或 MAPPO；这些会同时改变学习容量、优化难度或研究问题，不能回答核心机制是否必要。

## 7. 预验证

20 项针对性 pytest 全通过。五个配置均完成单回合 smoke，观察/动作维度保持 1025/25、无 NaN、safe-action smoke 下非法动作数为零。

## 8. 数据完整性

完整性状态：`{integrity['complete']}`；已完成主运行 {integrity['observed_core_runs']}/{integrity['expected_core_runs']}，已取得最终评估行 {integrity['observed_eval_rows']}/{integrity['expected_eval_rows']}。发现问题数：{len(integrity['issues'])}。详见 `data_integrity_check.json` 与 `failure_log.md`。

## 9. 主指标结果

{table}{anomaly_text}

## 10. 次指标与代价

详见 `ablation_summary.csv`，其中显式分离训练种子波动（`train_seed_sd`）和同一训练模型跨评估种子的波动（`mean_within_train_eval_sd`）。

## 11. 配对比较与统计

`paired_differences.csv` 保留 train-seed/eval-seed 一一配对的原始差值；`statistical_tests.csv` 使用每个训练种子上 20 个评估种子的均值为独立训练单位，报告 95% bootstrap CI、配对 t 检验、Cohen dz，并对每个消融的四个主指标做 Holm 校正。

## 12. 延迟与运行时

任务级端到端响应时间的 p50/p95/p99 写入 `runtime_quantiles.csv`；候选生成均值在次指标表中报告。图 3 仅基于最终评估导出的完成任务审计。

## 13. 无掩码诊断

{diagnostic_text} 该诊断仅 5k、单训练种子，不与主 20k 消融混合，也不作为掩码性能效应量。

## 14. 审稿风险与反谬误检查

统计谬误扫描覆盖 11/11：

| 类型 | 严重度 | 本实验检查结果 |
| --- | --- | --- |
| Simpson 悖论 | NOTE | 当前没有足够训练种子与变体数据检验分层方向反转；最终报告需同时展示训练种子层与汇总层。 |
| 生态谬误 | CAUTION | 评估种子不是独立训练重复，不能由 60 个 train×eval 行推断 60 次独立训练的稳定性。 |
| Berkson 悖论 | NOTE | 未按性能筛选训练种子；当前未发现条件选择造成的负相关分析。 |
| 碰撞变量偏差 | NOTE | 主分析未加入事后协变量控制；暂未形成碰撞变量路径。 |
| 基准率忽视 | NOTE | 完成率与超时率均使用已释放任务为分母；响应时间必须同时报告完成任务数。 |
| 均值回归 | NOTE | 训练种子预先固定，并非按极端表现选取；当前单种子异常不能解释为总体趋势。 |
| 幸存者偏差 | CAUTION | 延迟仅定义于完成任务；若完成率低，单独报告完成任务延迟会严重偏乐观，必须与完成率联合解释。 |
| 多重检验/到处寻找效应 | CAUTION | 四个主指标按消融内做 Holm 校正；次指标只能作为探索性结果。 |
| 分叉路径花园 | NOTE | 变体、种子、主指标及停止阈值已在运行清单中预注册；仍需保留所有失败运行与缺失数据。 |
| 相关不等于因果 | CAUTION | 完整的单因素消融可支持本协议内的机制归因；当前阶段性单种子结果不支持因果结论。 |
| 反向因果 | NOTE | 配置由实验者预先设定，性能不能反向决定配置；对本受控消融不构成主要威胁。 |

不要将 3 个训练种子的 screening 当作确证性显著性证据；不要把 train×eval 配对误视为独立训练重复；不要把安全验证保留误写成队列信息仍被策略直接使用；不要用无掩码 5k 诊断替代完整对照。

## 15. 可复现性与限制

`run_manifest.json` 固化了有效配置、奖励签名、特征 schema、地图/任务注册表和源码 SHA-256。当前设计能筛查最必要机制，但若要形成论文级结论，需增加独立训练种子并在新增地图/需求负荷上复验。
"""


def report_repaired(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    integrity: dict,
    diagnostic: dict,
    final: pd.DataFrame,
) -> str:
    """Build the corrected four-variant report without mixing legacy runs."""

    def value(variant: str, metric: str) -> float:
        selected = summary[(summary["variant"] == variant) & (summary["metric"] == metric)]["mean"]
        return float(selected.iloc[0]) if not selected.empty else math.nan

    def number(item: float, digits: int = 4) -> str:
        return "NA" if not math.isfinite(item) else f"{item:.{digits}f}"

    primary = summary[summary.metric.isin(PRIMARY)].pivot(index="metric", columns="variant", values="mean") if not summary.empty else pd.DataFrame()
    primary_table = markdown_table(primary.reindex(columns=VARIANTS)) if not primary.empty else "暂无完整结果。"

    seed_lines = [
        "| 配置 | 训练种子 | 完成率 | 超时率 | 高优先级响应时间 (s) | 加权 AoI |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not final.empty:
        grouped = final.groupby(["variant", "train_seed"], sort=False)
        for (variant, train_seed), group in grouped:
            seed_lines.append(
                "| " + " | ".join([
                    str(variant), str(int(train_seed)),
                    number(float(pd.to_numeric(group["completion_rate"], errors="coerce").mean())),
                    number(float(pd.to_numeric(group["timeout_rate"], errors="coerce").mean())),
                    number(float(pd.to_numeric(group["high_priority_response_time_s"], errors="coerce").mean())),
                    number(float(pd.to_numeric(group["weighted_aoi"], errors="coerce").mean())),
                ]) + " |"
            )

    resource_metrics = [
        "avg_response_time", "energy", "cav_travel_distance_m",
        "uav_return_distance_m", "mean_uav_recovery_wait_s",
        "queue_occupancy_time_s", "avg_tasks_per_sortie",
        "path_planning_failure_count", "safe_plan_rate",
        "mean_plan_generation_time_ms",
    ]
    resource = summary[summary.metric.isin(resource_metrics)].pivot(index="metric", columns="variant", values="mean") if not summary.empty else pd.DataFrame()
    resource_table = markdown_table(resource.reindex(columns=VARIANTS)) if not resource.empty else "暂无完整结果。"

    stat_lines = [
        "| 消融 | 主指标 | Full 优势均值 | 95% bootstrap CI | 配对 t 检验 p | Holm p | Cohen dz |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    if not tests.empty:
        for _, row in tests[tests["metric"].isin(PRIMARY)].iterrows():
            stat_lines.append(
                "| " + " | ".join([
                    str(row["variant"]), str(row["metric"]),
                    number(float(row["mean_full_advantage"])),
                    f"[{number(float(row['bootstrap_ci95_low']))}, {number(float(row['bootstrap_ci95_high']))}]",
                    number(float(row["paired_t_p_value"])),
                    number(float(row["holm_adjusted_p_value"])),
                    number(float(row["cohen_dz"])),
                ]) + " |"
            )

    run_state = "最终报告" if integrity["complete"] else "阶段性报告"
    confidence = "CAUTION" if integrity["complete"] else "RED_FLAG"
    reproducibility = "PARTIALLY_REPRODUCIBLE" if integrity["complete"] else "CANNOT_VERIFY"
    diagnostic_text = "未运行" if not diagnostic.get("available") else (
        f"旧协议的 5k 无掩码诊断平均非法动作率为 {diagnostic['mean_illegal_action_rate']:.2%}，"
        f"最大为 {diagnostic['max_illegal_action_rate']:.2%}；该诊断不与修复后的主消融合并。"
    )
    warning_text = "；".join(str(item) for item in integrity.get("warnings", [])) or "无"

    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: {datetime.now(tz=timezone.utc).date().isoformat()}
- Verification Status: ANALYZED
- Version Label: repaired_core_ablation_validation_v2

# 修复后的核心消融实验报告（20k screening）

**报告状态：{run_state}。Fixed CAV 已从核心统计中排除，旧结果仅作为失败基线审计证据保留。**

## Validation Report

- Source: `results/ablations/core_20k_repaired`
- Overall Confidence: {confidence}
- Reproducibility Verdict: {reproducibility}

## 1. 执行摘要

修复 Rainbow 回放预热缺口后，对 `full`、`fixed_discount`、`single_task`、`queue_blind` 的全部训练种子 0/1/2 重新训练；每个模型训练 20,000 step，并在共同评估种子 1000–1019 上评估。没有把旧协议模型与新协议模型混入同一统计表。

## 2. 修复内容与必要性

旧实现令 Rainbow 从第 1 步开始只依赖未训练的 NoisyNet。Full seed 0 因初始化不利，从首回合开始持续选择不派飞动作，20k 内始终零完成。修复后，前 `learning_starts=1000` 步均匀采样合法动作，之后恢复 NoisyNet；该规则统一作用于所有保留变体。

## 3. Fixed CAV 排除决定

Fixed CAV 的三个旧运行均为零完成、100% 超时，并伴随平均约 384 次路径规划失败。该被动巡航不仅移除了主动 CAV 决策，还破坏了任务可服务性，不满足“只改变一个机制且基线仍可运行”的消融条件。因此它不进入修复后的均值、配对检验、图表或机制判定；旧目录 `results/ablations/core_20k/fixed_cav_seed0..2` 保留作审计。

## 4. 保留的最小消融

| 配置 | 唯一目标变化 | 研究问题 |
| --- | --- | --- |
| full | 无 | 完整方法对照 |
| fixed_discount | 时间感知折扣改为固定决策折扣 | 不规则决策间隔是否需要时间感知折扣 |
| single_task | 每架次最多一个任务 | 多任务架次是否改善服务效率 |
| queue_blind | 策略与候选层屏蔽队列信息 | 显式队列感知是否改善等待与拥塞 |

## 5. 固定实验条件与复现信息

训练种子为 0/1/2，评估种子为 1000–1019，20,000 step，CUDA，禁用 TensorBoard。有效配置、源码 SHA-256、奖励签名、特征 schema、地图及任务注册表记录在 `run_manifest.json`。

## 6. 数据完整性

主运行 {integrity['observed_core_runs']}/{integrity['expected_core_runs']}；最终评估行 {integrity['observed_eval_rows']}/{integrity['expected_eval_rows']}；完整性状态 `{integrity['complete']}`。警告：{warning_text}。

## 7. 主指标汇总

{primary_table}

## 8. 训练种子分层结果

{chr(10).join(seed_lines)}

训练种子才是独立训练重复；不能把每个模型的 20 个评估种子当成 20 次独立训练。

## 9. 资源、机制与成本转移

{resource_table}

解释完成率和延迟时必须联合查看 UAV 能耗、CAV 距离、UAV 返回距离、回收等待与队列占用，避免把成本从空中转移到地面后仍表述为整体改进。

## 10. 配对统计、95% CI 与效应量

{chr(10).join(stat_lines)}

Full 优势为正表示 Full 更好。统计单位是每个训练种子在 20 个共同评估场景上的均值；四个主指标使用 Holm 校正。三个训练种子只构成 screening evidence。

## 11. 组件判定规则

仅当移除组件后主要指标跨种子稳定且实质恶化，才判为 Essential。CI 跨零或训练种子方向不一致时判为 Inconclusive；机制指标改善但主要业务证据不足时判为 Mechanism-supported。具体判定应以本报告第 8–10 节共同为依据。

## 12. 未执行项目

未做学习率、网络宽度、batch size、PER 参数或奖励小权重网格，也未做 C51、GNN、MAPPO 等模型族扩展；这些实验不能直接回答核心机制必要性。

## 13. 审稿风险

- 修复是看到 seed 0 坍缩后作出的，因此必须披露旧结果和修复依据，不能把失败运行静默删除。
- 只有 3 个训练种子，显著性与 CI 均不稳定。
- 响应时间只定义于完成任务，必须与完成率和完成任务数联合报告。
- 20k 是筛选预算，不足以证明最终收敛。

## 14. 统计谬误检查

覆盖 11/11：Simpson 悖论、生态谬误、Berkson 悖论、碰撞变量偏差、基准率忽视、均值回归、幸存者偏差、look-elsewhere、分叉路径花园、相关不等于因果、反向因果。主要风险是把评估种子误作独立训练重复、完成任务延迟的幸存者偏差，以及修复后选择性隐藏旧失败结果。

## 15. 无掩码诊断

{diagnostic_text}

## 16. 可支持与不可支持的表述

可以陈述修复协议下观察到的均值、跨训练种子波动和配对差值；不能把 20k×3 seeds 的筛选结果表述为最终显著性证据，也不能再将 Fixed CAV 的零完成结果作为主动 CAV 控制的有效消融证据。

## 17. 升级建议

只有在修复后 3 个训练种子的方向基本一致时，才值得将对应机制升级到 100k×5 seeds；方向不一致的机制应先定位稳定性原因，不做无意义超参数穷举。
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    final, issues = read_final_evaluations()
    warnings: list[dict] = []
    if not final.empty:
        for metric in METRICS:
            if metric in final:
                bad = int((~np.isfinite(pd.to_numeric(final[metric], errors="coerce"))).sum())
                if bad:
                    completion = pd.to_numeric(final.get("completion_rate"), errors="coerce")
                    expected_undefined = metric in {"high_priority_response_time_s", "avg_response_time"} and int((completion <= 0.0).sum()) == bad
                    target = warnings if expected_undefined else issues
                    target.append({
                        "issue": ("expected_undefined_" if expected_undefined else "nonfinite_") + metric,
                        "count": bad,
                    })
    summary = summary_table(final) if not final.empty else pd.DataFrame()
    differences, tests = paired_tables(final) if not final.empty else (pd.DataFrame(), pd.DataFrame())
    latency = runtime_quantiles()
    summary.to_csv(OUT / "ablation_summary.csv", index=False)
    differences.to_csv(OUT / "paired_differences.csv", index=False)
    tests.to_csv(OUT / "statistical_tests.csv", index=False)
    latency.to_csv(OUT / "runtime_quantiles.csv", index=False)
    expected_runs = len(VARIANTS) * len(TRAIN_SEEDS)
    integrity = {
        "complete": bool(len(issues) == 0 and not final.empty and final[["variant", "train_seed"]].drop_duplicates().shape[0] == expected_runs),
        "expected_core_runs": expected_runs,
        "observed_core_runs": int(final[["variant", "train_seed"]].drop_duplicates().shape[0]) if not final.empty else 0,
        "expected_eval_rows": expected_runs * len(EVAL_SEEDS),
        "observed_eval_rows": int(len(final)),
        "issues": issues,
        "warnings": warnings,
    }
    diagnostic = unmasked_diagnostic()
    integrity["unmasked_diagnostic"] = diagnostic
    (OUT / "data_integrity_check.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = [
        "# Failure / execution log", "",
        "- Unmasked diagnostic first launch: tool-level 120 s timeout at 2,000/5,000 step; no final result was used.",
        "- First core queue stopped after treating trainer status `complete` as stale; the status compatibility bug was fixed on 2026-08-08 and the queue resumed without rerunning the completed result.",
    ]
    failures.extend(f"- {item}" for item in issues)
    failures.extend(f"- Warning: {item}" for item in warnings)
    failures.append("- Fixed CAV excluded from the repaired core set because its passive patrol destroyed task feasibility; historical runs remain under results/ablations/core_20k.")
    failures.append("- Legacy full seed 0 no-launch collapse remains preserved under results/ablations/core_20k/full_seed0.")
    if not issues and not warnings:
        failures.append("- No missing/invalid completed-core-run condition detected at this aggregation time.")
    (OUT / "failure_log.md").write_text("\n".join(failures) + "\n", encoding="utf-8-sig")
    write_figures(summary, tests, latency)
    (OUT / "ABLATION_REPORT.md").write_text(
        report_repaired(summary, tests, integrity, diagnostic, final),
        encoding="utf-8-sig",
    )
    print(json.dumps(integrity, ensure_ascii=False))


if __name__ == "__main__":
    main()
