from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Iterable, Mapping


POSITIVE_COMPONENTS = (
    "completion_reward",
    "priority_reward",
)

COST_COMPONENTS = (
    "timeout_cost",
    "aoi_cost",
    "flight_energy_cost",
    "hover_energy_cost",
    "return_distance_cost",
    "cav_distance_cost",
    "recovery_wait_cost",
    "recovery_queue_cost",
    "route_change_cost",
    "route_reverse_cost",
    "coordination_failure_cost",
)

DIAGNOSTIC_COMPONENTS = (
    "terminal_undone_cost",
    "wait_action_cost",
    "illegal_action_cost",
    "legacy_queue_cost",
    "legacy_cav_queue_cost",
    "other_cost",
)


@dataclass
class RewardComponents:
    completion_reward: float = 0.0
    priority_reward: float = 0.0

    timeout_cost: float = 0.0
    aoi_cost: float = 0.0

    flight_energy_cost: float = 0.0
    hover_energy_cost: float = 0.0
    return_distance_cost: float = 0.0

    cav_distance_cost: float = 0.0
    recovery_wait_cost: float = 0.0
    recovery_queue_cost: float = 0.0

    route_change_cost: float = 0.0
    route_reverse_cost: float = 0.0
    coordination_failure_cost: float = 0.0

    terminal_undone_cost: float = 0.0
    wait_action_cost: float = 0.0
    illegal_action_cost: float = 0.0
    legacy_queue_cost: float = 0.0
    legacy_cav_queue_cost: float = 0.0
    other_cost: float = 0.0

    raw_total: float = 0.0
    normalized_total: float = 0.0
    final_reward: float = 0.0

    def compute_total(self) -> float:
        positive = sum(float(getattr(self, name)) for name in POSITIVE_COMPONENTS)
        costs = sum(float(getattr(self, name)) for name in COST_COMPONENTS)
        diagnostics = sum(float(getattr(self, name)) for name in DIAGNOSTIC_COMPONENTS)
        return float(positive - costs - diagnostics)

    def finalize(self) -> "RewardComponents":
        total = self.compute_total()
        self.raw_total = total
        self.normalized_total = total
        self.final_reward = total
        return self

    def to_dict(self) -> Dict[str, float]:
        return {field.name: float(getattr(self, field.name)) for field in fields(self)}

    def legacy_reward_components(self) -> Dict[str, float]:
        return {
            "completion": float(self.completion_reward + self.priority_reward),
            "timeout": float(-self.timeout_cost),
            "aoi": float(-self.aoi_cost),
            "energy": float(-self.flight_energy_cost),
            "uav_wait": float(-self.legacy_queue_cost),
            "cav_wait": float(-self.legacy_cav_queue_cost),
            "other": float(
                -self.terminal_undone_cost
                - self.wait_action_cost
                - self.illegal_action_cost
                - self.other_cost
            ),
        }


class RewardLedger:
    """Builds reward component records without changing Stage 2 reward semantics."""

    def __init__(self, *, tolerance: float = 1e-6) -> None:
        self.tolerance = float(tolerance)

    def from_legacy_interval(
        self,
        *,
        completion_reward: float,
        priority_reward: float,
        timeout_reward: float,
        aoi_reward: float,
        energy_reward: float,
        legacy_queue_reward: float,
        legacy_cav_queue_reward: float,
        wait_action_reward: float,
        illegal_action_reward: float,
        terminal_undone_reward: float,
        other_reward: float = 0.0,
    ) -> RewardComponents:
        components = RewardComponents(
            completion_reward=max(0.0, float(completion_reward)),
            priority_reward=max(0.0, float(priority_reward)),
            timeout_cost=max(0.0, -float(timeout_reward)),
            aoi_cost=max(0.0, -float(aoi_reward)),
            flight_energy_cost=max(0.0, -float(energy_reward)),
            legacy_queue_cost=max(0.0, -float(legacy_queue_reward)),
            legacy_cav_queue_cost=max(0.0, -float(legacy_cav_queue_reward)),
            wait_action_cost=max(0.0, -float(wait_action_reward)),
            illegal_action_cost=max(0.0, -float(illegal_action_reward)),
            terminal_undone_cost=max(0.0, -float(terminal_undone_reward)),
            other_cost=max(0.0, -float(other_reward)),
        )
        return components.finalize()

    def assert_conserved(self, components: RewardComponents, reward: float) -> None:
        diff = abs(float(components.final_reward) - float(reward))
        if diff > self.tolerance:
            raise AssertionError(
                f"reward components do not conserve total: "
                f"components={components.final_reward} reward={reward} diff={diff}"
            )


def reward_component_names() -> Iterable[str]:
    return tuple(field.name for field in fields(RewardComponents))


def finite_component_dict(values: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in values.items():
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError(f"non-finite reward component {key}: {value}")
        out[str(key)] = number
    return out
