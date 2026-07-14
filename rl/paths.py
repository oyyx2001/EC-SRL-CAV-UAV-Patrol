from __future__ import annotations

from pathlib import Path
from typing import Tuple


DQN_RESULTS_DIRNAME = "dqn"
DQN_TENSORBOARD_DIRNAME = "dqn"


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def resolve_dqn_output_dirs(
    root: Path,
    out_dir: str | Path | None = None,
    tensorboard_dir: str | Path | None = None,
) -> Tuple[Path, Path]:
    """Resolve DQN outputs into dedicated folders under results/rl by default."""
    rl_dir = root / "results" / "rl"

    if out_dir is None:
        resolved_out_dir = rl_dir / DQN_RESULTS_DIRNAME
    else:
        resolved_out_dir = Path(out_dir)
        if not resolved_out_dir.is_absolute():
            resolved_out_dir = root / resolved_out_dir
        if _same_path(resolved_out_dir, rl_dir):
            resolved_out_dir = resolved_out_dir / DQN_RESULTS_DIRNAME

    rl_tensorboard_dir = rl_dir / "tensorboard"
    if tensorboard_dir is None:
        resolved_tensorboard_dir = rl_tensorboard_dir / DQN_TENSORBOARD_DIRNAME
    else:
        resolved_tensorboard_dir = Path(tensorboard_dir)
        if not resolved_tensorboard_dir.is_absolute():
            resolved_tensorboard_dir = root / resolved_tensorboard_dir
        if _same_path(resolved_tensorboard_dir, rl_tensorboard_dir):
            resolved_tensorboard_dir = resolved_tensorboard_dir / DQN_TENSORBOARD_DIRNAME

    return resolved_out_dir, resolved_tensorboard_dir
