from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticAffineFlowFixture:
    fixture_id: str
    dimension: int
    matrix: tuple[tuple[float, ...], ...]
    offset: tuple[float, ...]
    base_mean: tuple[float, ...]
    base_covariance: tuple[tuple[float, ...], ...]
    base_particles: tuple[tuple[float, ...], ...]
    observation_particles: tuple[tuple[float, ...], ...]
    prior_log_density: tuple[float, ...]
    target_log_density: tuple[float, ...]
    proposal_base_log_density: tuple[float, ...]
    proposal_noise_mean: tuple[float, ...]
    proposal_noise_covariance: tuple[tuple[float, ...], ...]


def build_synthetic_affine_flow_fixture() -> SyntheticAffineFlowFixture:
    return SyntheticAffineFlowFixture(
        fixture_id="synthetic_affine_flow_fixture",
        dimension=2,
        matrix=((1.5, 0.25), (-0.4, 0.9)),
        offset=(0.2, -0.1),
        base_mean=(0.1, -0.3),
        base_covariance=((1.2, 0.1), (0.1, 0.8)),
        base_particles=((0.2, -0.1), (1.0, 0.5), (-0.7, 1.2)),
        observation_particles=((0.45, -0.05), (1.3, 0.2), (-0.1, 0.8)),
        prior_log_density=(-1.7, -0.8, -1.2),
        target_log_density=(-0.9, -0.4, -1.5),
        proposal_base_log_density=(-0.6, -1.1, -0.7),
        proposal_noise_mean=(0.05, -0.15),
        proposal_noise_covariance=((0.9, 0.2), (0.2, 0.7)),
    )
