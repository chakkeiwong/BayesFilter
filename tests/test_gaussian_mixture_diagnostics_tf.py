"""Independent CPU checks for generic Gaussian-mixture diagnostics."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.testing.gaussian_mixture_diagnostics_tf import (
    gaussian_mixture_moments,
    hard_assignment_transition_counts,
    retained_gaussian_mixture_diagnostics,
)
from bayesfilter.testing.importance_sampling_tf import sample_gaussian_mixture


DTYPE = tf.float64


def _three_component_target():
    probabilities = tf.constant((0.5, 0.3, 0.2), DTYPE)
    means = tf.constant(
        ((-4.5, -1.0, 0.8, -0.4), (4.0, -1.8, -0.7, 0.5), (0.5, 4.8, 0.2, -0.6)),
        DTYPE,
    )
    factors = tf.constant(
        (
            ((0.75, 0.0, 0.0, 0.0), (0.18, 0.55, 0.0, 0.0), (0.05, 0.10, 0.45, 0.0), (0.08, -0.03, 0.12, 0.38)),
            ((0.48, 0.0, 0.0, 0.0), (-0.16, 0.88, 0.0, 0.0), (0.08, -0.22, 0.68, 0.0), (0.02, 0.14, -0.09, 0.52)),
            ((0.62, 0.0, 0.0, 0.0), (0.28, 0.58, 0.0, 0.0), (-0.12, 0.16, 0.82, 0.0), (0.10, 0.04, 0.20, 0.44)),
        ),
        DTYPE,
    )
    covariances = tf.matmul(factors, factors, transpose_b=True)
    return probabilities, means, covariances


def test_exact_three_component_moments_match_total_moment_formula() -> None:
    probabilities, means, covariances = _three_component_target()
    actual = gaussian_mixture_moments(probabilities, means, covariances)
    expected_mean = tf.reduce_sum(probabilities[:, None] * means, axis=0)
    expected_second = tf.reduce_sum(
        probabilities[:, None, None]
        * (covariances + means[:, :, None] * means[:, None, :]),
        axis=0,
    )
    tf.debugging.assert_near(actual["mean"], expected_mean, atol=1.0e-14)
    tf.debugging.assert_near(
        actual["covariance"],
        expected_second - expected_mean[:, None] * expected_mean[None, :],
        atol=1.0e-14,
    )


def test_transition_matrix_counts_each_directed_pair_and_chain() -> None:
    labels = tf.constant(((0, 2), (1, 2), (2, 1), (0, 0)), tf.int32)
    counts = hard_assignment_transition_counts(labels, component_count=3)
    expected = tf.constant(
        (
            ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
            ((0, 0, 0), (1, 0, 0), (0, 1, 1)),
        ),
        tf.int64,
    )
    tf.debugging.assert_equal(counts, expected)


def test_exact_three_component_draws_pass_mass_and_transition_screens() -> None:
    probabilities, means, covariances = _three_component_target()
    rows, _ = sample_gaussian_mixture(
        60_000, probabilities, means, covariances, seed=(20260812, 12001)
    )
    diagnostics = retained_gaussian_mixture_diagnostics(
        tf.reshape(rows, (15_000, 4, 4)), probabilities, means, covariances
    )
    assert diagnostics["component_count"] == 3
    assert diagnostics["gates"]["all_components_observed_overall"] is True
    assert diagnostics["gates"]["all_component_mass_intervals_contain_truth"] is True
    assert diagnostics["gates"]["every_component_involved_in_transition"] is True
    assert diagnostics["joint_test_performed"] is False
    tf.debugging.assert_near(
        diagnostics["component_conditional_mean"], means, atol=0.035, rtol=0.0
    )
    tf.debugging.assert_near(
        diagnostics["component_conditional_covariance"],
        covariances,
        atol=0.035,
        rtol=0.0,
    )


def test_component_permutation_only_permutes_component_outputs() -> None:
    probabilities, means, covariances = _three_component_target()
    rows, _ = sample_gaussian_mixture(
        24_000, probabilities, means, covariances, seed=(20260812, 12002)
    )
    samples = tf.reshape(rows, (6_000, 4, 4))
    direct = retained_gaussian_mixture_diagnostics(
        samples, probabilities, means, covariances
    )
    permutation = tf.constant((2, 0, 1), tf.int32)
    permuted = retained_gaussian_mixture_diagnostics(
        samples,
        tf.gather(probabilities, permutation),
        tf.gather(means, permutation),
        tf.gather(covariances, permutation),
    )
    tf.debugging.assert_near(
        permuted["component_mass"],
        tf.gather(direct["component_mass"], permutation),
        atol=1.0e-13,
        rtol=1.0e-13,
    )
    tf.debugging.assert_near(permuted["sample_mean"], direct["sample_mean"], atol=1.0e-14)


def test_missing_component_is_detected_without_joint_test() -> None:
    probabilities, means, covariances = _three_component_target()
    rows, _ = sample_gaussian_mixture(
        8_000,
        tf.constant((0.625, 0.375), DTYPE),
        means[:2],
        covariances[:2],
        seed=(20260812, 12003),
    )
    diagnostics = retained_gaussian_mixture_diagnostics(
        tf.reshape(rows, (2_000, 4, 4)), probabilities, means, covariances
    )
    assert diagnostics["gates"]["all_components_observed_overall"] is False
    assert diagnostics["gates"]["all_component_mass_intervals_contain_truth"] is False
    assert diagnostics["passed_primary_screens"] is False
    assert diagnostics["joint_test_performed"] is False


def test_unimodal_target_uses_the_same_generic_path() -> None:
    probabilities = tf.constant((1.0,), DTYPE)
    means = tf.constant(((0.5, -0.25),), DTYPE)
    covariances = tf.constant((((1.0, 0.2), (0.2, 0.8)),), DTYPE)
    rows, _ = sample_gaussian_mixture(
        16_000, probabilities, means, covariances, seed=(20260812, 12004)
    )
    diagnostics = retained_gaussian_mixture_diagnostics(
        tf.reshape(rows, (4_000, 4, 2)), probabilities, means, covariances
    )
    assert diagnostics["component_count"] == 1
    assert diagnostics["gates"]["all_components_observed_overall"] is True
    assert diagnostics["gates"]["all_component_mass_intervals_contain_truth"] is True
    assert diagnostics["gates"]["every_component_involved_in_transition"] is True
    assert diagnostics["transition_requirement_applicable"] is False
    assert diagnostics["passed_primary_screens"] is True
