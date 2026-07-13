from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

from bayesfilter.inference.hmc import PrecomputedMassArtifact
from bayesfilter.inference.hmc_coordinates import (
    AffineCoordinateTransform,
    KernelState,
    MomentumMetric,
    PositionCovarianceEstimate,
    WarmupTrajectoryPolicy,
    transform_from_precomputed_mass_artifact,
)


def _estimate(covariance: np.ndarray) -> PositionCovarianceEstimate:
    dimension = covariance.shape[0]
    return PositionCovarianceEstimate(
        center=np.linspace(-0.2, 0.3, dimension),
        covariance=covariance,
        source_coordinate_signature="test-source-coordinate",
        estimator_family="analytical_oracle",
        state_count=128,
        effective_rank=dimension,
        regularization_report={"method": "none"},
        adequacy_report={"passed": True},
    )


def _kernel_state(transform: AffineCoordinateTransform) -> KernelState:
    latent = np.array([[0.2] * transform.dimension, [-0.1] * transform.dimension])
    theta = transform.latent_to_theta(latent).numpy()
    return KernelState(
        canonical_theta=theta,
        active_latent=latent,
        transform=transform,
        momentum_metric=MomentumMetric.identity_for(transform),
        epsilon=None,
        trajectory_policy=WarmupTrajectoryPolicy(3, 16),
        adaptation_generation=0,
        seed_lineage=(20260711, 2),
        evidence_status="initialized",
    )


@pytest.mark.parametrize(
    "covariance",
    (
        np.eye(3),
        np.array([[4.0, 1.2, -0.3], [1.2, 2.0, 0.4], [-0.3, 0.4, 1.5]]),
    ),
)
def test_affine_transform_round_trip_and_score_map(covariance: np.ndarray) -> None:
    transform = AffineCoordinateTransform.from_covariance_estimate(_estimate(covariance))
    latent = np.array([[0.2, -0.4, 0.8], [-1.0, 0.5, 0.3]])
    theta = transform.latent_to_theta(latent)

    np.testing.assert_allclose(transform.theta_to_latent(theta), latent, atol=1.0e-12)
    np.testing.assert_allclose(transform.covariance, covariance, atol=1.0e-12)

    theta_score = -theta.numpy()
    expected_latent_score = theta_score @ transform.factor
    np.testing.assert_allclose(
        transform.theta_score_to_latent_score(theta_score),
        expected_latent_score,
        atol=1.0e-12,
    )


def test_affine_rescaling_preserves_canonical_theta_and_invalidates_epsilon() -> None:
    first = AffineCoordinateTransform.from_covariance_estimate(_estimate(np.eye(2)))
    state = _kernel_state(first).with_epsilon(0.15, evidence_status="step_tuned")
    rotated = np.array([[2.0, 0.7], [0.7, 1.0]])
    second = AffineCoordinateTransform.from_covariance_estimate(_estimate(rotated))

    remapped = state.remap(
        second,
        adaptation_generation=1,
        evidence_status="metric_updated",
    )

    np.testing.assert_allclose(remapped.canonical_theta, state.canonical_theta)
    np.testing.assert_allclose(
        second.latent_to_theta(remapped.active_latent),
        state.canonical_theta,
        atol=1.0e-12,
    )
    assert remapped.epsilon is None
    assert remapped.epsilon_context_signature is None


def test_kernel_state_rejects_cross_coordinate_epsilon_context() -> None:
    transform = AffineCoordinateTransform.from_covariance_estimate(_estimate(np.eye(2)))
    state = _kernel_state(transform).with_epsilon(0.1, evidence_status="step_tuned")

    with pytest.raises(ValueError, match="epsilon context is stale"):
        replace(state, epsilon_context_signature="stale-coordinate-context")


def test_kernel_state_rejects_stale_metric_and_corrupt_state() -> None:
    transform = AffineCoordinateTransform.from_covariance_estimate(_estimate(np.eye(2)))
    state = _kernel_state(transform)
    stale = MomentumMetric(np.eye(2), np.eye(2), "different-coordinate")

    with pytest.raises(ValueError, match="coordinate signature is stale"):
        replace(state, momentum_metric=stale)
    with pytest.raises(ValueError, match="do not round trip"):
        replace(state, canonical_theta=state.canonical_theta + 1.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("adaptation_generation", 0.5),
        ("adaptation_generation", False),
        ("seed_lineage", (20260711.5, 2)),
        ("seed_lineage", (20260711, True)),
    ],
)
def test_kernel_state_rejects_noninteger_lineage_fields(
    field: str,
    value: object,
) -> None:
    transform = AffineCoordinateTransform.from_covariance_estimate(_estimate(np.eye(2)))
    state = _kernel_state(transform)

    with pytest.raises(ValueError, match="integer scalar"):
        replace(state, **{field: value})


@pytest.mark.parametrize(
    ("leapfrog", "maximum"),
    [(3.5, 16), (True, 16), (3, 16.5), (3, False)],
)
def test_warmup_trajectory_policy_rejects_noninteger_counts(
    leapfrog: object,
    maximum: object,
) -> None:
    with pytest.raises(ValueError, match="integer scalar"):
        WarmupTrajectoryPolicy(leapfrog, maximum)


def test_active_whitening_metric_rejects_nonidentity_momentum() -> None:
    transform = AffineCoordinateTransform.from_covariance_estimate(_estimate(np.eye(2)))
    with pytest.raises(ValueError, match="requires identity momentum"):
        MomentumMetric(
            momentum_covariance=np.diag([2.0, 1.0]),
            kinetic_precision=np.diag([0.5, 1.0]),
            coordinate_signature=transform.signature,
        )


def test_legacy_mass_compatibility_preserves_arrays_but_not_adequacy_claim() -> None:
    covariance = np.array([[2.0, 0.4], [0.4, 0.8]])
    artifact = PrecomputedMassArtifact.from_covariance(
        position=[0.5, -0.3],
        covariance=covariance,
        adapter_signature="test-legacy-mass-adapter",
        covariance_source="test_covariance",
        jitter=0.0,
    )

    estimate, transform = transform_from_precomputed_mass_artifact(
        artifact,
        source_coordinate_signature="legacy-source",
    )

    np.testing.assert_allclose(estimate.covariance, covariance)
    np.testing.assert_allclose(transform.covariance, covariance)
    assert estimate.adequacy_report["operational_metric_adequacy_not_inferred"] is True


def test_coordinate_arrays_are_immutable_and_signatures_include_contents() -> None:
    first = _estimate(np.eye(2))
    second = _estimate(np.diag([1.0, 2.0]))

    assert first.signature != second.signature
    with pytest.raises(ValueError):
        first.covariance[0, 0] = 2.0
