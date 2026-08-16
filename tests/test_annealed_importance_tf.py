"""Known-law tests for bridge-correct fixed-HMC AIS."""

from __future__ import annotations

import math

import pytest
import tensorflow as tf

from bayesfilter.testing.annealed_importance_tf import run_linear_ais_fixed_hmc
from bayesfilter.testing.importance_sampling_tf import (
    gaussian_mixture_log_prob,
    sample_gaussian_mixture,
    self_normalized_importance_diagnostics,
)


DTYPE = tf.float64


def test_identical_normalized_proposal_target_has_zero_log_weights() -> None:
    log_two_pi = tf.math.log(tf.constant(2.0 * math.pi, DTYPE))

    def normal(state: tf.Tensor) -> tf.Tensor:
        return -0.5 * tf.reduce_sum(tf.square(state), axis=1) - 0.5 * log_two_pi

    initial = tf.reshape(tf.linspace(tf.constant(-2.0, DTYPE), tf.constant(2.0, DTYPE), 16), (-1, 1))
    result = run_linear_ais_fixed_hmc(
        normal,
        normal,
        initial,
        num_steps=8,
        step_size=0.2,
        num_leapfrog_steps=3,
        seed=(20260810, 6101),
        jit_compile=True,
    )
    tf.debugging.assert_near(result["log_weights"], tf.zeros(16, DTYPE), atol=1.0e-12)
    assert bool(tf.reduce_all(result["path_all_finite"]).numpy())


def test_ais_recovers_unequal_weight_unequal_scale_mixture() -> None:
    means = tf.constant(((-3.0,), (2.0,)), DTYPE)
    covariances = tf.constant((((0.4**2,),), ((1.2**2,),)), DTYPE)
    proposal_probabilities = tf.constant((0.5, 0.5), DTYPE)
    target_probabilities = tf.constant((0.8, 0.2), DTYPE)
    initial, _labels = sample_gaussian_mixture(
        4096,
        proposal_probabilities,
        means,
        covariances,
        seed=(20260810, 6201),
    )

    def proposal(state: tf.Tensor) -> tf.Tensor:
        return gaussian_mixture_log_prob(
            state, proposal_probabilities, means, covariances
        )

    def target(state: tf.Tensor) -> tf.Tensor:
        return gaussian_mixture_log_prob(
            state, target_probabilities, means, covariances
        )

    result = run_linear_ais_fixed_hmc(
        proposal,
        target,
        initial,
        num_steps=24,
        step_size=0.2,
        num_leapfrog_steps=3,
        seed=(20260810, 6202),
        rejuvenation_interval=4,
        jit_compile=True,
    )
    diagnostics = self_normalized_importance_diagnostics(
        result["log_weights"],
        tf.zeros_like(result["log_weights"]),
        result["terminal_state"][:, 0] < 0.0,
    )
    expected_negative = tf.reduce_sum(
        target_probabilities
        * 0.5
        * (
            1.0
            + tf.math.erf(
                (0.0 - means[:, 0])
                / tf.sqrt(2.0 * covariances[:, 0, 0])
            )
        )
    )
    tf.debugging.assert_near(
        diagnostics["negative_region_probability"], expected_negative, atol=0.025
    )
    assert bool(tf.reduce_all(result["path_all_finite"]).numpy())
    assert float(diagnostics["effective_sample_size_fraction"].numpy()) > 0.7
    assert float(diagnostics["maximum_normalized_weight"].numpy()) < 0.001


def test_sparse_rejuvenation_preserves_exact_identical_target_weights() -> None:
    log_two_pi = tf.math.log(tf.constant(2.0 * math.pi, DTYPE))

    def normal(state: tf.Tensor) -> tf.Tensor:
        return -0.5 * tf.reduce_sum(tf.square(state), axis=1) - 0.5 * log_two_pi

    initial = tf.reshape(
        tf.linspace(tf.constant(-2.0, DTYPE), tf.constant(2.0, DTYPE), 16),
        (-1, 1),
    )
    result = run_linear_ais_fixed_hmc(
        normal,
        normal,
        initial,
        num_steps=16,
        step_size=0.2,
        num_leapfrog_steps=3,
        seed=(20260810, 6301),
        rejuvenation_interval=4,
        jit_compile=True,
    )
    tf.debugging.assert_near(result["log_weights"], tf.zeros(16, DTYPE), atol=1.0e-12)
    tf.debugging.assert_equal(result["rejuvenation_count"], 4)
    assert bool(tf.reduce_all(result["path_all_finite"]).numpy())
    assert bool(
        tf.reduce_all(
            tf.logical_and(
                result["acceptance_fraction"] >= 0.0,
                result["acceptance_fraction"] <= 1.0,
            )
        ).numpy()
    )


@pytest.mark.parametrize("interval", (0, 3))
def test_rejuvenation_interval_must_be_positive_divisor(interval: int) -> None:
    def normal(state: tf.Tensor) -> tf.Tensor:
        return -0.5 * tf.reduce_sum(tf.square(state), axis=1)

    with pytest.raises(ValueError):
        run_linear_ais_fixed_hmc(
            normal,
            normal,
            tf.zeros((4, 1), DTYPE),
            num_steps=8,
            step_size=0.2,
            num_leapfrog_steps=3,
            seed=(20260810, 6401),
            rejuvenation_interval=interval,
            jit_compile=True,
        )
