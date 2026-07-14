from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np


SEED_MODES = ("fixed", "random", "curriculum")


@dataclass
class CurriculumStage:
    name: str
    timesteps: int
    seeds: List[int] | None = None
    random_seed: bool = False

    @property
    def seed_pool(self) -> str:
        if self.random_seed:
            return "random"
        return ",".join(map(str, self.seeds or []))


class CurriculumSeedScheduler:
    """Provide episode seeds for fixed, random, and curriculum RL training.

    Curriculum training starts on a narrow seed pool so the model can learn the
    basic launch/recovery/task-completion mechanics, then gradually expands the
    seed pool and finally switches to fully random seeds. This lowers early
    training variance while still pushing the final policy toward cross-scenario
    generalization.
    """

    def __init__(
        self,
        stages: Sequence[CurriculumStage | Dict[str, Any]],
        rng_seed: int = 0,
        rank: int = 0,
        rank_stride: int = 10000,
    ):
        if not stages:
            raise ValueError("CurriculumSeedScheduler requires at least one stage")
        self.stages = [self._coerce_stage(s, idx) for idx, s in enumerate(stages)]
        self.rng = np.random.default_rng(int(rng_seed))
        self.rank = int(rank)
        self.rank_stride = int(rank_stride)
        self.stage_idx = 0

    def _coerce_stage(self, stage: CurriculumStage | Dict[str, Any], idx: int) -> CurriculumStage:
        if isinstance(stage, CurriculumStage):
            return stage
        seeds = stage.get("seeds")
        return CurriculumStage(
            name=str(stage.get("name", f"stage_{idx + 1}")),
            timesteps=int(stage.get("timesteps", 0)),
            seeds=[int(s) for s in seeds] if seeds is not None else None,
            random_seed=bool(stage.get("random_seed", False)),
        )

    def set_stage(self, stage_idx: int) -> None:
        if stage_idx < 0 or stage_idx >= len(self.stages):
            raise IndexError(f"stage index out of range: {stage_idx}")
        self.stage_idx = int(stage_idx)

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.stage_idx]

    def sample_seed(self) -> int:
        stage = self.current_stage
        if stage.random_seed:
            base_seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
        else:
            if not stage.seeds:
                raise ValueError(f"stage {stage.name!r} must define seeds or random_seed=true")
            base_seed = int(self.rng.choice(stage.seeds))
        return base_seed + self.rank * self.rank_stride


def default_curriculum_stages() -> List[CurriculumStage]:
    return [
        CurriculumStage("stage_1_single_seed", timesteps=200000, seeds=[0]),
        CurriculumStage("stage_2_five_seeds", timesteps=300000, seeds=[0, 1, 2, 3, 4]),
        CurriculumStage("stage_3_twenty_seeds", timesteps=500000, seeds=list(range(20))),
        CurriculumStage("stage_4_random_seeds", timesteps=1000000, random_seed=True),
    ]


def parse_int_list(value: str | None) -> List[int] | None:
    if value is None or str(value).strip() == "":
        return None
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def normalize_stage(stage: Dict, default_name: str) -> CurriculumStage:
    seeds = stage.get("seeds")
    return CurriculumStage(
        name=str(stage.get("name", default_name)),
        timesteps=int(stage.get("timesteps", 0)),
        seeds=[int(s) for s in seeds] if seeds is not None else None,
        random_seed=bool(stage.get("random_seed", False)),
    )


def scale_stage_timesteps(stages: Sequence[CurriculumStage], total_timesteps: int) -> List[CurriculumStage]:
    total_timesteps = int(total_timesteps)
    if total_timesteps < len(stages):
        raise ValueError("total_timesteps must be at least the number of curriculum stages")
    current_total = sum(max(0, int(s.timesteps)) for s in stages)
    if current_total <= 0:
        base = total_timesteps // max(1, len(stages))
        extra = total_timesteps % max(1, len(stages))
        return [
            CurriculumStage(s.name, base + (1 if i < extra else 0), seeds=s.seeds, random_seed=s.random_seed)
            for i, s in enumerate(stages)
        ]
    raw = [total_timesteps * max(0, int(s.timesteps)) / current_total for s in stages]
    ints = [max(1, int(np.floor(v))) for v in raw]
    diff = total_timesteps - sum(ints)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - np.floor(raw[i]), reverse=True)
    while diff > 0:
        for i in order:
            if diff <= 0:
                break
            ints[i] += 1
            diff -= 1
    while diff < 0:
        for i in reversed(order):
            if diff >= 0:
                break
            if ints[i] > 1:
                ints[i] -= 1
                diff += 1
    return [
        CurriculumStage(s.name, ints[i], seeds=s.seeds, random_seed=s.random_seed)
        for i, s in enumerate(stages)
    ]


def override_stage_timesteps(stages: Sequence[CurriculumStage], stage_timesteps: Sequence[int] | None) -> List[CurriculumStage]:
    if not stage_timesteps:
        return list(stages)
    if len(stage_timesteps) == 1:
        values = [int(stage_timesteps[0])] * len(stages)
    elif len(stage_timesteps) == len(stages):
        values = [int(v) for v in stage_timesteps]
    else:
        raise ValueError("--stage-timesteps must provide one value or one value per stage")
    return [
        CurriculumStage(stage.name, values[i], seeds=stage.seeds, random_seed=stage.random_seed)
        for i, stage in enumerate(stages)
    ]


def build_training_stages(
    cfg: Dict,
    seed_mode: str | None,
    seed: int | None,
    total_timesteps: int | None,
    stage_timesteps: Sequence[int] | None,
    default_timesteps: int = 2000,
) -> tuple[str, int, List[CurriculumStage]]:
    training_cfg = cfg.get("training", {}) or {}
    resolved_seed_mode = str(seed_mode or training_cfg.get("seed_mode", "fixed")).lower()
    if resolved_seed_mode not in SEED_MODES:
        raise ValueError(f"seed_mode must be one of {SEED_MODES}, got {resolved_seed_mode!r}")

    base_seed = int(cfg.get("project", {}).get("seed", 42) if seed is None else seed)
    single_stage_timesteps = int(
        total_timesteps
        or training_cfg.get("total_timesteps")
        or training_cfg.get("timesteps")
        or default_timesteps
    )

    if resolved_seed_mode == "curriculum":
        raw_stages = training_cfg.get("curriculum_stages") or []
        stages = [
            normalize_stage(stage, f"stage_{idx + 1}")
            for idx, stage in enumerate(raw_stages)
        ] or default_curriculum_stages()
        if total_timesteps is not None:
            stages = scale_stage_timesteps(stages, int(total_timesteps))
    elif resolved_seed_mode == "random":
        stages = [CurriculumStage("random_seeds", timesteps=single_stage_timesteps, random_seed=True)]
    else:
        stages = [CurriculumStage("fixed_seed", timesteps=single_stage_timesteps, seeds=[base_seed])]

    stages = override_stage_timesteps(stages, stage_timesteps)
    for stage in stages:
        if stage.timesteps <= 0:
            raise ValueError(f"stage {stage.name!r} timesteps must be positive")
    return resolved_seed_mode, base_seed, stages
