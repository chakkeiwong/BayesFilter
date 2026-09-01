from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.linear.rectangular_factor_tf import (
    batched_direct_support_conditional,
    batched_direct_stack_svd_factor,
    batched_fixed_pivot_rectangular_qr,
    batched_fixed_support_qr_likelihood,
    batched_fixed_support_qr_update,
    batched_support_gaussian_log_likelihood,
)


def test_fixed_pivot_rectangular_qr_reconstructs_exact_rank_two_stack() -> None:
    stack = tf.constant(
        [[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
        tf.float64,
    )
    factor, _, diagnostics = batched_fixed_pivot_rectangular_qr(
        stack, tf.constant([0, 1, 2], tf.int32), 2
    )
    np.testing.assert_allclose(
        factor @ tf.transpose(factor, [0, 2, 1]),
        stack @ tf.transpose(stack, [0, 2, 1]),
        rtol=1e-12,
        atol=1e-12,
    )
    assert bool(diagnostics["chart_valid"][0])


def test_fixed_pivot_rectangular_qr_derivative_matches_centered_finite_difference() -> None:
    stack = tf.constant(
        [[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
        tf.float64,
    )
    derivative = tf.constant(
        [[[[0.1, 0.0, 0.1, 0.0], [0.0, 0.2, 0.0, 0.2], [0.1, 0.2, 0.1, 0.2]]]],
        tf.float64,
    )
    permutation = tf.constant([0, 1, 2], tf.int32)
    factor, d_factor, diagnostics = batched_fixed_pivot_rectangular_qr(
        stack, permutation, 2, derivative
    )
    eps = 1e-6
    plus = batched_fixed_pivot_rectangular_qr(stack + eps * derivative[:, 0], permutation, 2)[0]
    minus = batched_fixed_pivot_rectangular_qr(stack - eps * derivative[:, 0], permutation, 2)[0]
    np.testing.assert_allclose((plus - minus) / (2.0 * eps), d_factor[:, 0], rtol=2e-6, atol=2e-8)
    assert bool(diagnostics["derivative_chart_valid"][0])


def test_rectangular_qr_chart_failure_and_bad_permutation_are_explicit() -> None:
    stack = tf.constant([[[1.0, 0.0], [0.0, 0.0]]], tf.float64)
    with pytest.raises(ValueError, match="bijection"):
        batched_fixed_pivot_rectangular_qr(stack, tf.constant([0, 0]), 1)
    outside = tf.constant([[[1.0, 0.0], [0.0, 1.0]]], tf.float64)
    _, _, diagnostics = batched_fixed_pivot_rectangular_qr(
        outside, tf.constant([0, 1]), 1, residual_tolerance=1e-14
    )
    assert not bool(diagnostics["chart_valid"][0])


def test_direct_stack_svd_handles_rank_zero_repeated_values_and_cutoff_branch() -> None:
    rank_zero = tf.zeros([1, 3, 4], tf.float64)
    factor, singular, _, diagnostics = batched_direct_stack_svd_factor(rank_zero)
    assert diagnostics["rank"].numpy().tolist() == [0]
    assert factor.shape == (1, 3, 3)
    repeated = tf.constant([[[2.0, 0.0], [0.0, 2.0]]], tf.float64)
    _, _, _, repeated_diagnostics = batched_direct_stack_svd_factor(repeated)
    assert repeated_diagnostics["rank"].numpy().tolist() == [2]
    near = tf.constant([[[1.0, 0.0], [0.0, 1e-10]]], tf.float64)
    assert batched_direct_stack_svd_factor(near, relative_cutoff=1e-12)[3]["rank"].numpy().tolist() == [2]
    assert batched_direct_stack_svd_factor(near, relative_cutoff=1e-8)[3]["rank"].numpy().tolist() == [1]
    assert singular.shape == (1, 3)


def test_support_likelihood_on_and_off_affine_support() -> None:
    stack = tf.constant([[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], tf.float64)
    on_support, rank, diagnostics = batched_support_gaussian_log_likelihood(
        tf.constant([[1.0, 0.0]], tf.float64), stack
    )
    off_support, _, off_diagnostics = batched_support_gaussian_log_likelihood(
        tf.constant([[1.0, 1e-3]], tf.float64), stack
    )
    expected = -0.5 * (np.log(2.0 * np.pi) + np.log(4.0) + 0.25)
    np.testing.assert_allclose(on_support, [expected], rtol=1e-12, atol=1e-12)
    assert rank.numpy().tolist() == [1]
    assert bool(diagnostics["on_support"][0])
    assert not bool(off_diagnostics["on_support"][0])
    assert np.isneginf(off_support.numpy()[0])


def test_direct_support_conditional_is_value_only_and_removes_observed_stack() -> None:
    observation_stack = tf.constant([[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], tf.float64)
    state_stack = tf.constant([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]], tf.float64)
    log_likelihood, gain, conditional_factor, rank, diagnostics = batched_direct_support_conditional(
        observation_stack,
        state_stack,
        tf.constant([[1.0, 0.0]], tf.float64),
    )
    assert rank.numpy().tolist() == [1]
    assert bool(diagnostics["value_only"])
    assert bool(diagnostics["on_support"][0])
    assert np.isfinite(log_likelihood.numpy()[0])
    observed_state = diagnostics["conditional_stack"]
    # The residual state loading has no component in the observed column span.
    np.testing.assert_allclose(
        observed_state @ tf.transpose(observation_stack, [0, 2, 1]),
        np.zeros([1, 2, 2]),
        atol=1e-12,
    )
    assert conditional_factor.shape[0] == 1
    assert gain.shape == (1, 2, 2)


def test_rectangular_primitives_reject_nonfinite_and_invalid_shapes() -> None:
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        batched_direct_stack_svd_factor(tf.constant(np.nan, shape=[1, 2, 2], dtype=tf.float64))
    with pytest.raises(ValueError):
        batched_direct_stack_svd_factor(tf.zeros([2, 2], tf.float64))
    with pytest.raises(ValueError):
        batched_fixed_pivot_rectangular_qr(tf.zeros([1, 3, 2], tf.float64), tf.constant([0, 1, 2]), 3)


def test_fixed_support_likelihood_derivative_and_renormalized_epsilon_limit() -> None:
    factor = tf.constant([[[2.0], [0.0]]], tf.float64)
    innovation = tf.constant([[1.0, 0.0]], tf.float64)
    d_factor = tf.constant([[[[0.1], [0.0]]]], tf.float64)
    d_innovation = tf.constant([[[0.2, 0.0]]], tf.float64)
    value, score, diagnostics = batched_fixed_support_qr_likelihood(
        innovation, factor, d_innovation, d_factor
    )
    eps = 1.0e-6
    plus = batched_fixed_support_qr_likelihood(
        innovation + eps * d_innovation[:, 0], factor + eps * d_factor[:, 0]
    )[0]
    minus = batched_fixed_support_qr_likelihood(
        innovation - eps * d_innovation[:, 0], factor - eps * d_factor[:, 0]
    )[0]
    np.testing.assert_allclose(score[:, 0], (plus - minus) / (2.0 * eps), rtol=2e-7, atol=2e-9)
    assert bool(diagnostics["score_valid"][0])

    # Ambient N(0, GG' + epsilon I) diverges on support.  After subtracting
    # the null-space normalization, it converges to the affine-support value.
    for regularizer in (1.0e-3, 1.0e-5, 1.0e-7):
        covariance = factor[0] @ tf.transpose(factor[0]) + regularizer * tf.eye(2, dtype=tf.float64)
        ambient = -0.5 * (
            2.0 * np.log(2.0 * np.pi)
            + tf.linalg.logdet(covariance)
            + tf.einsum("i,ij,j->", innovation[0], tf.linalg.inv(covariance), innovation[0])
        )
        renormalized = ambient + 0.5 * np.log(2.0 * np.pi * regularizer)
        np.testing.assert_allclose(renormalized, value[0], rtol=0.0, atol=2.0 * regularizer)


def test_fixed_support_conditional_value_and_derivatives_match_centered_fd() -> None:
    observation_permutation = tf.constant([0, 1], tf.int32)
    conditional_permutation = tf.constant([1, 0], tf.int32)

    def evaluate(theta: float, with_derivative: bool):
        observation_stack = tf.constant(
            [[[2.0 + 0.1 * theta, 0.0, 0.0], [0.0, 0.0, 0.0]]], tf.float64
        )
        state_stack = tf.constant(
            [[[1.0 + 0.2 * theta, 0.0, 1.0 + 0.2 * theta],
              [0.0, 1.0 + 0.1 * theta, 1.0 + 0.1 * theta]]], tf.float64
        )
        innovation = tf.constant([[1.0 + 0.3 * theta, 0.0]], tf.float64)
        if not with_derivative:
            return batched_fixed_support_qr_update(
                observation_stack, state_stack, innovation,
                observation_permutation, 1, conditional_permutation, 2,
            )
        return batched_fixed_support_qr_update(
            observation_stack,
            state_stack,
            innovation,
            observation_permutation,
            1,
            conditional_permutation,
            2,
            tf.constant([[[[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]]], tf.float64),
            tf.constant([[[[0.2, 0.0, 0.2], [0.0, 0.1, 0.1]]]], tf.float64),
            tf.constant([[[0.3, 0.0]]], tf.float64),
        )

    result = evaluate(0.0, True)
    eps = 1.0e-6
    plus, minus = evaluate(eps, False), evaluate(-eps, False)
    for value_index, derivative_index in ((0, 1), (2, 3), (4, 5)):
        np.testing.assert_allclose(
            result[derivative_index][:, 0],
            (plus[value_index] - minus[value_index]) / (2.0 * eps),
            rtol=2e-6,
            atol=2e-8,
        )
    assert bool(result[-1]["chart_valid"][0])
    assert bool(result[-1]["conditional_chart_valid"][0])


def test_fixed_chart_anisotropic_near_rank_change_fails_closed() -> None:
    stack = tf.constant([[[1.0, 0.0], [0.0, 1.0e-14]]], tf.float64)
    _, _, diagnostics = batched_fixed_pivot_rectangular_qr(
        stack, tf.constant([0, 1]), 2
    )
    assert not bool(diagnostics["chart_valid"][0])


def test_rank_two_fixed_support_conditional_matches_dense_authority_and_fd() -> None:
    observation = tf.constant([[[1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]], tf.float64)
    state = tf.constant(
        [[[1.0, -0.2, 0.3, 0.7], [0.1, 0.9, -0.4, 0.5], [0.6, 0.2, 0.8, -0.1]]],
        tf.float64,
    )
    innovation = tf.constant([[0.2, -0.3]], tf.float64)
    d_observation = tf.constant(
        [[[[0.1, -0.05, 0.02, 0.0], [0.03, 0.04, -0.02, 0.01]]]], tf.float64
    )
    d_state = tf.constant(
        [[[[0.02, 0.01, -0.03, 0.04], [-0.01, 0.03, 0.02, -0.02], [0.04, -0.02, 0.01, 0.03]]]],
        tf.float64,
    )
    d_innovation = tf.constant([[[0.05, -0.02]]], tf.float64)
    result = batched_fixed_support_qr_update(
        observation,
        state,
        innovation,
        tf.constant([0, 1]),
        2,
        tf.constant([0, 1, 2]),
        2,
        d_observation,
        d_state,
        d_innovation,
    )
    s = observation @ tf.transpose(observation, [0, 2, 1])
    expected_gain = state @ tf.transpose(observation, [0, 2, 1]) @ tf.linalg.inv(s)
    expected_covariance = (
        state @ tf.transpose(state, [0, 2, 1])
        - expected_gain @ s @ tf.transpose(expected_gain, [0, 2, 1])
    )
    np.testing.assert_allclose(result[-1]["gain"], expected_gain, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        result[4] @ tf.transpose(result[4], [0, 2, 1]),
        expected_covariance,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(result[2], tf.einsum("bnm,bm->bn", expected_gain, innovation), rtol=2e-12, atol=2e-12)
    eps = 1.0e-6
    plus = batched_fixed_support_qr_update(
        observation + eps * d_observation[:, 0],
        state + eps * d_state[:, 0],
        innovation + eps * d_innovation[:, 0],
        tf.constant([0, 1]), 2, tf.constant([0, 1, 2]), 2,
    )
    minus = batched_fixed_support_qr_update(
        observation - eps * d_observation[:, 0],
        state - eps * d_state[:, 0],
        innovation - eps * d_innovation[:, 0],
        tf.constant([0, 1]), 2, tf.constant([0, 1, 2]), 2,
    )
    for value_index, derivative_index in ((0, 1), (2, 3), (4, 5)):
        np.testing.assert_allclose(
            result[derivative_index][:, 0],
            (plus[value_index] - minus[value_index]) / (2.0 * eps),
            rtol=3e-6,
            atol=3e-8,
        )
