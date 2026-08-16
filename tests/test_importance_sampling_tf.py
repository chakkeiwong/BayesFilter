"""Analytic tests for corrected TensorFlow importance diagnostics."""

from __future__ import annotations

import math

import pytest
import tensorflow as tf

from bayesfilter.testing.importance_sampling_tf import (
    gaussian_mixture_log_prob,
    gaussian_mixture_log_prob_responsibilities_score,
    independent_batch_interval,
    sample_gaussian_mixture,
    self_normalized_importance_diagnostics,
    validate_gaussian_mixture,
)


DTYPE = tf.float64


def _fixture():
    means = tf.constant(((-3.0,), (2.0,)), DTYPE)
    covariances = tf.constant((((0.4**2,),), ((1.2**2,),)), DTYPE)
    return means, covariances


def test_gaussian_mixture_log_prob_matches_direct_formula() -> None:
    means, covariances = _fixture()
    probabilities = tf.constant((0.7, 0.3), DTYPE)
    rows = tf.constant(((-3.0,), (0.0,), (2.0,)), DTYPE)
    actual = gaussian_mixture_log_prob(rows, probabilities, means, covariances)
    components = []
    for index in range(2):
        variance = covariances[index, 0, 0]
        components.append(
            tf.math.log(probabilities[index])
            - 0.5 * tf.square(rows[:, 0] - means[index, 0]) / variance
            - 0.5 * tf.math.log(tf.constant(2.0 * math.pi, DTYPE) * variance)
        )
    expected = tf.reduce_logsumexp(tf.stack(components, axis=1), axis=1)
    tf.debugging.assert_near(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_gaussian_mixture_responsibilities_and_score_match_autodiff() -> None:
    means, covariances = _fixture()
    probabilities = tf.constant((0.7, 0.3), DTYPE)
    rows = tf.constant(((-3.0,), (-0.4,), (2.0,)), DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(rows)
        reference = gaussian_mixture_log_prob(
            rows, probabilities, means, covariances
        )
        reference_sum = tf.reduce_sum(reference)
    reference_score = tape.gradient(reference_sum, rows)
    value, responsibilities, score = (
        gaussian_mixture_log_prob_responsibilities_score(
            rows, probabilities, means, covariances
        )
    )
    tf.debugging.assert_near(value, reference, atol=1.0e-12, rtol=1.0e-12)
    tf.debugging.assert_near(score, reference_score, atol=1.0e-12, rtol=1.0e-12)
    tf.debugging.assert_near(
        tf.reduce_sum(responsibilities, axis=1), tf.ones(3, DTYPE), atol=1.0e-14
    )


def test_corrected_importance_recovers_unequal_weight_unequal_scale_target() -> None:
    means, covariances = _fixture()
    proposal_probabilities = tf.constant((0.5, 0.5), DTYPE)
    target_probabilities = tf.constant((0.8, 0.2), DTYPE)
    rows, labels = sample_gaussian_mixture(
        50_000,
        proposal_probabilities,
        means,
        covariances,
        seed=(20260810, 3101),
    )
    proposal = gaussian_mixture_log_prob(
        rows, proposal_probabilities, means, covariances
    )
    target = gaussian_mixture_log_prob(
        rows, target_probabilities, means, covariances
    )
    diagnostics = self_normalized_importance_diagnostics(
        target, proposal, rows[:, 0] < 0.0
    )
    # The overlap across zero is tiny but nonzero, so use the exact target law's
    # negative probability rather than the component probability 0.8.
    standard_normal = tf.math.erf
    left_negative = 0.5 * (
        1.0
        + standard_normal(
            (0.0 - means[:, 0])
            / tf.sqrt(2.0 * covariances[:, 0, 0])
        )
    )
    expected = tf.reduce_sum(target_probabilities * left_negative)
    tf.debugging.assert_near(
        diagnostics["negative_region_probability"], expected, atol=0.01
    )
    assert 0.45 <= float(tf.reduce_mean(tf.cast(labels == 0, DTYPE)).numpy()) <= 0.55
    assert float(diagnostics["effective_sample_size_fraction"].numpy()) > 0.7
    assert float(diagnostics["maximum_normalized_weight"].numpy()) < 1.0e-4


def test_weight_degeneracy_is_visible_in_ess_and_maximum_weight() -> None:
    target = tf.constant((100.0, 0.0, 0.0, 0.0), DTYPE)
    proposal = tf.zeros(4, DTYPE)
    result = self_normalized_importance_diagnostics(
        target, proposal, tf.constant((True, False, False, False))
    )
    assert float(result["effective_sample_size_fraction"].numpy()) < 0.26
    assert float(result["maximum_normalized_weight"].numpy()) > 0.99


def test_independent_batch_interval_uses_sample_uncertainty() -> None:
    values = tf.constant((0.4, 0.5, 0.6, 0.5, 0.4, 0.6, 0.5, 0.5), DTYPE)
    interval = independent_batch_interval(values)
    tf.debugging.assert_near(interval["mean"], tf.constant(0.5, DTYPE), atol=1.0e-14)
    assert 0.0 < float(interval["half_width"].numpy()) < 0.10
    assert int(interval["batch_count"].numpy()) == 8


def test_invalid_mixture_configuration_fails_closed() -> None:
    means, covariances = _fixture()
    with pytest.raises(tf.errors.InvalidArgumentError):
        validate_gaussian_mixture(
            tf.constant((0.6, 0.6), DTYPE), means, covariances
        )
