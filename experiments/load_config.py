from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamic_patrol_marl.utils.config import load_config, project_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scenario_60node.yaml")
    args = parser.parse_args()
    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(cfg_path)
    print(json.dumps(cfg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
