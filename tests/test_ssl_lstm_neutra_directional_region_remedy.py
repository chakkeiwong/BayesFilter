from __future__ import annotations

import dataclasses
import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.predictive_equivalence import (
    PredictiveContractError,
    batched_split_quadratic_loss_confidence_bounds,
    classify_split_proper_score_equivalence,
    conditional_mean_log_variance_influence,
    proper_score_loss,
    split_quadratic_loss_confidence_bounds,
)


def _average_loss():
    return proper_score_loss(tf.fill([10], tf.constant(0.1, tf.float64)))


def test_split_region_uses_prospective_dimensions_and_union_allocation() -> None:
    bounds = split_quadratic_loss_confidence_bounds(
        tf.zeros([20], tf.float64),
        tf.eye(20, dtype=tf.float64),
        _average_loss(),
        jit_compile=False,
    )
    expected_average = tfp.distributions.Chi2(
        df=tf.constant(20.0, tf.float64)
    ).quantile(
        tf.constant(0.975, tf.float64)
    )
    expected_horizon = tfp.distributions.Chi2(
        df=tf.constant(2.0, tf.float64)
    ).quantile(
        tf.constant(0.9975, tf.float64)
    )
    tf.debugging.assert_near(
        bounds.average_confidence_radius_squared, expected_average
    )
    tf.debugging.assert_near(
        bounds.horizon_confidence_radii_squared,
        tf.fill([10], expected_horizon),
    )
    assert float(bounds.allocated_familywise_alpha) == pytest.approx(0.05)
    historical = tfp.distributions.Chi2(
        df=tf.constant(20.0, tf.float64)
    ).quantile(tf.constant(0.95, tf.float64))
    assert float(expected_horizon / historical) == pytest.approx(0.3814951915)


def test_split_region_rejects_alpha_overallocation() -> None:
    with pytest.raises(PredictiveContractError, match="exceeds familywise"):
        split_quadratic_loss_confidence_bounds(
            tf.zeros([20], tf.float64),
            tf.eye(20, dtype=tf.float64),
            _average_loss(),
            average_alpha=0.03,
            horizon_alpha=0.003,
            familywise_alpha=0.05,
            jit_compile=False,
        )


def test_horizon_projection_selects_mean_and_matching_log_variance() -> None:
    diagonal = tf.constant(
        [float(index + 1) for index in range(20)], tf.float64
    )
    estimate = tf.zeros([20], tf.float64)
    estimate = tf.tensor_scatter_nd_update(estimate, [[3], [13]], [0.2, -0.4])
    bounds = split_quadratic_loss_confidence_bounds(
        estimate,
        tf.linalg.diag(diagonal) * tf.constant(1.0e-5, tf.float64),
        _average_loss(),
        jit_compile=False,
    )
    expected_point = 0.5 * 0.2**2 + 0.25 * 0.4**2
    assert float(bounds.horizon_point_losses[3]) == pytest.approx(expected_point)
    zeros = tf.concat(
        (bounds.horizon_point_losses[:3], bounds.horizon_point_losses[4:]), axis=0
    )
    tf.debugging.assert_near(zeros, tf.zeros([9], tf.float64))


def test_batched_split_bounds_match_scalar_results() -> None:
    estimates = tf.stack(
        (
            tf.zeros([20], tf.float64),
            tf.linspace(tf.constant(-0.04, tf.float64), 0.05, 20),
        )
    )
    base = tf.linalg.diag(tf.linspace(tf.constant(1.0e-5, tf.float64), 3.0e-5, 20))
    covariances = tf.stack((base, 1.5 * base))
    batched = batched_split_quadratic_loss_confidence_bounds(
        estimates, covariances, _average_loss(), jit_compile=False
    )
    assert bool(tf.reduce_all(batched.inference_admissible))
    for index in range(2):
        scalar = split_quadratic_loss_confidence_bounds(
            estimates[index], covariances[index], _average_loss(), jit_compile=False
        )
        tf.debugging.assert_near(
            batched.average_lower_bound[index], scalar.average_lower_bound
        )
        tf.debugging.assert_near(
            batched.average_upper_bound[index], scalar.average_upper_bound
        )
        tf.debugging.assert_near(
            batched.horizon_lower_bounds[index], scalar.horizon_lower_bounds
        )
        tf.debugging.assert_near(
            batched.horizon_upper_bounds[index], scalar.horizon_upper_bounds
        )


def test_split_classification_rejects_tampered_bounds() -> None:
    bounds = split_quadratic_loss_confidence_bounds(
        tf.zeros([20], tf.float64),
        tf.eye(20, dtype=tf.float64) * tf.constant(1.0e-8, tf.float64),
        _average_loss(),
        jit_compile=False,
    )
    threshold = tf.constant(0.0068491, tf.float64)
    decision = classify_split_proper_score_equivalence(
        bounds,
        acceptable_average_loss=threshold,
        acceptable_horizon_loss=threshold,
    )
    assert decision.status == "PASS"
    tampered = dataclasses.replace(
        bounds, horizon_upper_bounds=tf.zeros([10], tf.float64)
    )
    rejected = classify_split_proper_score_equivalence(
        tampered,
        acceptable_average_loss=threshold,
        acceptable_horizon_loss=threshold,
    )
    assert rejected.status == "INVALID_HARD_VETO"
    assert rejected.hard_veto_codes == ("SPLIT_LOSS_BOUNDS_UNAUTHENTICATED",)


def test_conditional_moment_estimator_matches_total_variance_algebra() -> None:
    means = tf.reshape(
        tf.linspace(tf.constant(-0.7, tf.float64), 0.8, 4 * 5 * 2 * 10),
        [4, 5, 2, 10],
    )
    variances = tf.reshape(
        tf.linspace(tf.constant(0.2, tf.float64), 0.5, 4 * 5 * 2 * 10),
        [4, 5, 2, 10],
    )
    result = conditional_mean_log_variance_influence(
        means, variances, jit_compile=False
    )
    expected_mean = tf.reduce_mean(means, axis=[0, 1, 2])
    expected_variance = tf.reduce_mean(
        variances + tf.square(means), axis=[0, 1, 2]
    ) - tf.square(expected_mean)
    tf.debugging.assert_near(result.standardized_means, expected_mean)
    tf.debugging.assert_near(result.log_variances, tf.math.log(expected_variance))
    tf.debugging.assert_near(
        tf.reduce_mean(result.influence_values, axis=[0, 1]),
        tf.zeros([20], tf.float64),
        atol=1.0e-14,
    )


def test_conditional_moment_scale_contract_is_invariant_when_applied_correctly() -> None:
    base = tf.reshape(tf.range(4 * 3 * 2 * 10, dtype=tf.float64), [4, 3, 2, 10])
    means = (base - tf.reduce_mean(base)) / 20.0
    variances = tf.ones_like(means) * 0.4
    scale = tf.linspace(tf.constant(0.5, tf.float64), 2.0, 10)
    reference = conditional_mean_log_variance_influence(
        means, variances, jit_compile=False
    )
    physical_means = means * scale
    physical_variances = variances * tf.square(scale)
    rescaled = conditional_mean_log_variance_influence(
        physical_means / scale,
        physical_variances / tf.square(scale),
        jit_compile=False,
    )
    tf.debugging.assert_near(reference.feature_estimate, rescaled.feature_estimate)
    tf.debugging.assert_near(reference.influence_values, rescaled.influence_values)


def test_boundary_constants_match_chapter() -> None:
    threshold = 0.0068491
    assert math.sqrt(2.0 * threshold) == pytest.approx(0.1170393096)
    assert math.exp(2.0 * math.sqrt(threshold)) == pytest.approx(1.1800048858)
    assert math.exp(-2.0 * math.sqrt(threshold)) == pytest.approx(0.8474541182)


def test_controlled_law_hac_bias_calculation_matches_chapter() -> None:
    exact = 0.825 + 2.0 * 0.65 * 0.6 / (1.0 - 0.6)
    bandwidth = 20
    bartlett = 0.825 + 2.0 * 0.65 * sum(
        (1.0 - lag / (bandwidth + 1.0)) * 0.6**lag
        for lag in range(1, bandwidth + 1)
    )
    assert exact == pytest.approx(2.775)
    assert bartlett == pytest.approx(2.5428622354)
    assert (exact - bartlett) / exact == pytest.approx(0.0836532485)
