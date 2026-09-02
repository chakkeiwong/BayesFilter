"""Focused tests for the frozen C2 transformed-observation Student guide."""

import math

import pytest
import tensorflow as tf

from bayesfilter.highdim.c2_transformed_observation_student_proposal_tf import (
    LOG_CHI_SQUARE_MEAN,
    LOG_CHI_SQUARE_VARIANCE,
    build_c2_transformed_observation_student_proposal,
    raw_observation_zero_cross_covariance,
    student_scale_from_covariance,
    transformed_log_square_observation,
)


DTYPE = tf.float64


def _proposal():
    theta = tf.constant([0.6, math.log(0.4)], DTYPE)
    transition = tf.constant([[0.6, 0.04], [-0.02, 0.6]], DTYPE)
    process = tf.eye(2, dtype=DTYPE)
    observation = tf.constant([0.35, -0.21], DTYPE)
    return build_c2_transformed_observation_student_proposal(
        transition_matrix=transition,
        process_covariance=process,
        observation=observation,
        theta_reference=theta,
        nu=8.0,
        time_index=1,
    )


def test_log_chi_square_constants_and_raw_zero_gain_negative_control() -> None:
    assert math.isclose(
        LOG_CHI_SQUARE_MEAN, -0.5772156649015329 - math.log(2.0), abs_tol=1e-15
    )
    assert math.isclose(LOG_CHI_SQUARE_VARIANCE, math.pi**2 / 2.0, abs_tol=1e-15)
    tf.debugging.assert_equal(
        raw_observation_zero_cross_covariance(3), tf.zeros([3, 3], DTYPE)
    )


def test_transformed_guide_has_checked_kalman_geometry_and_student_scale() -> None:
    proposal = _proposal()
    parents = tf.constant(
        [[-0.5, 0.2], [0.3, -0.7], [1.1, 0.4]], DTYPE
    )
    means = proposal.conditional_mean(parents)
    assert means.shape == (3, 2)
    tf.debugging.assert_near(
        proposal.scale,
        student_scale_from_covariance(proposal.posterior_covariance, 8.0),
        atol=2e-14,
    )
    assert proposal.solve_residual_max <= 2e-11
    assert float(tf.reduce_min(tf.linalg.eigvalsh(proposal.posterior_covariance)).numpy()) > 0.0
    samples = proposal.sample_with_seed(parents, 3, (91, 7), jit_compile=False)
    assert bool(samples["finite"].numpy())
    tf.debugging.assert_near(
        samples["physical_log_density"], proposal.log_density(samples["physical_points"], parents), atol=2e-12
    )


def test_student_sampler_matches_covariance_moment_convention() -> None:
    proposal = _proposal()
    count = 8192
    parents = tf.zeros([count, 2], DTYPE)
    samples = proposal.sample_with_seed(parents, count, (193, 17), jit_compile=False)
    points = samples["physical_points"]
    empirical_mean = tf.reduce_mean(points, axis=0)
    centered = points - empirical_mean[None, :]
    empirical_covariance = tf.linalg.matmul(centered, centered, transpose_a=True) / tf.cast(
        count - 1, DTYPE
    )
    tf.debugging.assert_near(empirical_mean, proposal.conditional_mean(parents[:1])[0], atol=0.08)
    tf.debugging.assert_near(empirical_covariance, proposal.posterior_covariance, atol=0.12)


def test_transformed_observation_rejects_exact_zero() -> None:
    theta = tf.constant([0.6, math.log(0.4)], DTYPE)
    with pytest.raises(ValueError, match="nonzero observations"):
        transformed_log_square_observation(tf.constant([0.0, 0.2], DTYPE), theta)


def test_student_scale_rejects_nonpositive_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="greater than two"):
        student_scale_from_covariance(tf.eye(2, dtype=DTYPE), 2.0)
    with pytest.raises(ValueError, match="positive definite"):
        student_scale_from_covariance(
            tf.constant([[1.0, 0.0], [0.0, -1.0]], DTYPE), 8.0
        )
