from __future__ import annotations

from typing import Any, Dict

import numpy as np


_AGENT_KEYS = {
    "launch": "launch_time_s",
    "recovery": "recovery_time_s",
}


def operation_time_block(cfg: Dict[str, Any], operation: str) -> Dict[str, Any]:
    operation = str(operation).lower()
    if operation not in _AGENT_KEYS:
        raise ValueError(f"Unknown operation time: {operation}")
    root = dict(cfg.get("operational_time_uncertainty", {}) or {})
    return dict(root.get(operation, {}) or {})


def operation_time_uncertainty_enabled(cfg: Dict[str, Any], operation: str) -> bool:
    root = dict(cfg.get("operational_time_uncertainty", {}) or {})
    block = operation_time_block(cfg, operation)
    return bool(root.get("enabled", False)) and bool(block.get("enabled", True))


def planning_operation_time_s(cfg: Dict[str, Any], operation: str) -> float:
    """Return the duration visible to planners, never the realized sample."""
    operation = str(operation).lower()
    block = operation_time_block(cfg, operation)
    fallback = float(cfg.get("agents", {}).get(_AGENT_KEYS[operation], 0.0))
    if not operation_time_uncertainty_enabled(cfg, operation):
        return max(0.0, fallback)

    mode = str(block.get("planning_estimate_mode", "fixed")).lower()
    if mode in {"upper", "upper_bound", "maximum", "max"}:
        value = block.get("max_s", fallback)
    elif mode in {"mean", "expected", "center"}:
        value = block.get("mean_s", fallback)
    elif mode in {"fixed", "configured"}:
        value = block.get("planning_estimate_s", fallback)
    else:
        raise ValueError(f"Unknown planning_estimate_mode for {operation}: {mode}")
    return max(0.0, float(value))


def sample_actual_operation_time_s(
    cfg: Dict[str, Any],
    operation: str,
    rng: np.random.Generator,
) -> float:
    """Sample a true truncated-normal actual duration within its bounds."""
    operation = str(operation).lower()
    if not operation_time_uncertainty_enabled(cfg, operation):
        return planning_operation_time_s(cfg, operation)

    block = operation_time_block(cfg, operation)
    distribution = str(block.get("distribution", "truncated_normal")).lower()
    lower = float(block.get("min_s", 0.0))
    upper = float(block.get("max_s", lower))
    if lower < 0.0 or upper < lower:
        raise ValueError(f"Invalid {operation} duration bounds: [{lower}, {upper}]")
    if upper <= lower + 1e-12:
        return max(0.0, lower)
    mean = float(block.get("mean_s", 0.5 * (lower + upper)))
    if not lower <= mean <= upper:
        raise ValueError(f"{operation} mean {mean} is outside [{lower}, {upper}]")
    if distribution not in {"truncated_normal", "truncnorm", "bounded_normal"}:
        raise ValueError(f"Unsupported {operation} duration distribution: {distribution}")
    std = float(block.get("std_s", (upper - lower) / 6.0))
    if std <= 0.0:
        raise ValueError(f"{operation} std_s must be positive")

    for _ in range(1024):
        value = float(rng.normal(mean, std))
        if lower <= value <= upper:
            return value
    return float(np.clip(rng.normal(mean, std), lower, upper))
