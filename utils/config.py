from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        root = project_root()
        candidate = root / "configs" / str(path)
        if candidate.exists():
            path = candidate
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_name = cfg.pop("base", None)
    if base_name:
        base_path = path.parent / base_name
        base_cfg = load_config(base_path)
        cfg = deep_update(base_cfg, cfg)
    cfg.setdefault("_config_path", str(path))
    return cfg


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dirs(cfg: Dict[str, Any]) -> Path:
    root = project_root()
    results_dir = root / cfg.get("project", {}).get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    return results_dir
