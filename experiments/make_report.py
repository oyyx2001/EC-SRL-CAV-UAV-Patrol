from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from dynamic_patrol_marl.utils.config import project_root


def table_or_note(csv_path: Path, group_col: str = "method") -> str:
    if not csv_path.exists():
        return f"Missing data file: `{csv_path}`. Run the corresponding experiment first."
    df = pd.read_csv(csv_path)
    metrics = [c for c in ["completion_rate", "timeout_rate", "avg_response_time_s", "high_priority_response_time_s", "weighted_aoi", "total_energy", "avg_tasks_per_sortie"] if c in df.columns]
    if group_col not in df.columns:
        group_col = "method" if "method" in df.columns else df.columns[0]
    summary = df.groupby(group_col)[metrics].agg(["mean", "std"]).round(3)
    return summary.to_markdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/experiment_report.md")
    args = parser.parse_args()
    root = project_root()
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    baseline_csv = root / "results" / "baselines" / "baseline_results_small.csv"
    scenario_csv = root / "results" / "scenarios" / "comparison_results.csv"
    ablation_csv = root / "results" / "ablation" / "ablation_results.csv"
    text = "# Experiment report\n\n"
    text += "## Baseline comparison\n\n" + table_or_note(baseline_csv) + "\n\n"
    text += "## Scenario comparison\n\n" + (table_or_note(scenario_csv, group_col="scenario") if scenario_csv.exists() else "Run `python -m dynamic_patrol_marl.experiments.run_scenarios --seeds 30` first.") + "\n\n"
    text += "## Ablation study\n\n" + table_or_note(ablation_csv) + "\n\n"
    text += "## Traceability\n\nEach figure in `results/**/plots` is generated from the CSV file in the same experiment folder.\n"
    out.write_text(text, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
