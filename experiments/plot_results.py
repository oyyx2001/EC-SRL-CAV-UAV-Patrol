from __future__ import annotations

import argparse
from pathlib import Path

from dynamic_patrol_marl.utils.visualization import plot_baseline_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()
    paths = plot_baseline_results(args.csv, args.out_dir)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
