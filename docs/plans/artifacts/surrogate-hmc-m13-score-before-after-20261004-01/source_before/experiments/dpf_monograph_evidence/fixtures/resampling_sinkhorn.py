from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SoftResamplingBiasFixture:
    fixture_id: str
    particles: tuple[float, float]
    base_weights: tuple[float, float]
    relaxed_mixture_parameter: float
    nonlinear_test_values: tuple[float, float]


@dataclass(frozen=True)
class SinkhornResidualFixture:
    fixture_id: str
    source_marginal: tuple[float, ...]
    target_marginal: tuple[float, ...]
    cost_matrix: tuple[tuple[float, ...], ...]
    epsilon: float
    stabilization_mode: str
    budget_ladder: tuple[int, ...]


def build_soft_resampling_bias_fixture() -> SoftResamplingBiasFixture:
    return SoftResamplingBiasFixture(
        fixture_id="soft_resampling_bias_fixture",
        particles=(-1.0, 2.0),
        base_weights=(0.25, 0.75),
        relaxed_mixture_parameter=0.2,
        nonlinear_test_values=(1.0, 4.0),
    )


def build_sinkhorn_residual_fixture() -> SinkhornResidualFixture:
    return SinkhornResidualFixture(
        fixture_id="sinkhorn_residual_fixture",
        source_marginal=(0.5, 0.3, 0.2),
        target_marginal=(0.4, 0.35, 0.25),
        cost_matrix=(
            (0.0, 1.0, 2.0),
            (1.0, 0.0, 1.0),
            (2.0, 1.0, 0.0),
        ),
        epsilon=0.3,
        stabilization_mode="plain_log_domain_scaling",
        budget_ladder=(5, 20, 100),
    )
