from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixedScalarHMCTargetFixture:
    fixture_id: str
    dimension: int
    matrix: tuple[tuple[float, ...], ...]
    beta: float
    gamma: float
    c: tuple[float, ...]
    evaluation_point: tuple[float, ...]
    inverse_mass: tuple[tuple[float, ...], ...]
    initial_momentum: tuple[float, ...]
    leapfrog_step_size: float
    leapfrog_step_count: int


def build_fixed_scalar_hmc_target_fixture() -> FixedScalarHMCTargetFixture:
    return FixedScalarHMCTargetFixture(
        fixture_id="fixed_scalar_hmc_target_fixture",
        dimension=2,
        matrix=((1.4, 0.2), (0.2, 0.9)),
        beta=0.07,
        gamma=0.11,
        c=(0.6, -0.4),
        evaluation_point=(0.4, -0.25),
        inverse_mass=((1.2, 0.0), (0.0, 0.8)),
        initial_momentum=(0.3, -0.2),
        leapfrog_step_size=0.05,
        leapfrog_step_count=3,
    )
