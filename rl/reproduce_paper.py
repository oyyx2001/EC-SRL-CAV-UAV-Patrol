"""Run both original pretrained policies on the 20 matched unseen test seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

from dynamic_patrol_marl.rl.checkpoint_io import CASES, ROOT, TEST_SEEDS
from dynamic_patrol_marl.rl.evaluate_checkpoint import run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or ROOT / "results" / ("paper_smoke" if args.smoke else "paper_reproduction_20seeds")
    if not output.is_absolute():
        output = ROOT / output
    _, summary = run_evaluation(CASES, [1000] if args.smoke else TEST_SEEDS, output, args.workers)
    wide = summary.pivot(index="case", columns="metric", values="mean")
    wide.to_csv(output / "paper_summary.csv")
    print(summary.to_string(index=False))
    complete = bool(summary.full_20_seed_protocol.all())
    matched = bool(complete and summary.matches_reference_2dp.all())
    print("Full 20-seed protocol:", complete)
    print("All six Table VIII means match at displayed precision:", matched)
    print("All six Table VIII sample SDs match at displayed precision:",
          bool(complete and summary.matches_reference_std_2dp.all()))
    if not matched:
        print("No values were adjusted. Inspect reference_comparison.csv and REPRODUCIBILITY.md.")


if __name__ == "__main__":
    main()
