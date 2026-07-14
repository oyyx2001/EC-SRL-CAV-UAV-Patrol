from __future__ import annotations

from typing import List

from .sortie_generator import Sortie
from .greedy_baseline import select_non_conflicting


class RollingHorizonPlanner:
    """Greedy rolling horizon baseline.

    At each decision epoch it regenerates feasible candidate sorties and chooses a
    non-conflicting batch with the best dynamic score. This mirrors the online
    re-planning idea in the experiment guide while keeping the MVP fully runnable.
    """

    def __init__(self, max_batch: int | None = None):
        self.max_batch = max_batch

    def __call__(self, sim, candidates: List[Sortie]):
        max_count = self.max_batch if self.max_batch is not None else max(1, len(sim.idle_drones_on_vehicle()))
        ranked = sorted(candidates, key=lambda s: s.score, reverse=True)
        return select_non_conflicting(ranked, max_count=max_count)
