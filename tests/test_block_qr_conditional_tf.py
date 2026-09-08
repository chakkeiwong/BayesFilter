from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.linear.block_qr_conditional_tf import batched_block_qr_conditional


def _stacks(seed: int = 12):
    rng = np.random.default_rng(seed)
    observation = tf.constant(rng.normal(size=(2, 2, 7)), tf.float64)
    state = tf.constant(rng.normal(size=(2, 3, 7)), tf.float64)
    return observation, state


def test_block_qr_reconstructs_gain_and_conditional_schur_identity() -> None:
    observation, state = _stacks()
    ly, gain, lf, *_rest, diagnostics = batched_block_qr_conditional(
        observation, state, compute_covariance_diagnostics=True
    )
    s = observation @ tf.transpose(observation, [0, 2, 1])
    pxy = state @ tf.transpose(observation, [0, 2, 1])
    pminus = state @ tf.transpose(state, [0, 2, 1])
    np.testing.assert_allclose(ly @ tf.transpose(ly, [0, 2, 1]), s, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gain, pxy @ tf.linalg.inv(s), rtol=1e-11, atol=1e-12)
    expected = pminus - pxy @ tf.linalg.solve(s, tf.transpose(pxy, [0, 2, 1]))
    np.testing.assert_allclose(lf @ tf.transpose(lf, [0, 2, 1]), expected, rtol=1e-11, atol=1e-12)
    assert float(tf.reduce_max(diagnostics["conditional_factor_reconstruction_residual"])) < 1e-11


def test_block_qr_derivatives_match_centered_finite_difference_without_covariance_diagnostics() -> None:
    observation, state = _stacks(21)
    rng = np.random.default_rng(22)
    d_observation = tf.constant(rng.normal(size=(2, 2, 2, 7)), tf.float64)
    d_state = tf.constant(rng.normal(size=(2, 2, 3, 7)), tf.float64)
    result = batched_block_qr_conditional(
        observation,
        state,
        d_observation,
        d_state,
        compute_covariance_diagnostics=False,
    )
    eps = 1e-6
    plus = batched_block_qr_conditional(
        observation + eps * d_observation[:, 0],
        state + eps * d_state[:, 0],
        compute_covariance_diagnostics=False,
    )
    minus = batched_block_qr_conditional(
        observation - eps * d_observation[:, 0],
        state - eps * d_state[:, 0],
        compute_covariance_diagnostics=False,
    )
    for value_index, derivative_index in ((0, 3), (1, 4), (2, 5)):
        finite_difference = (plus[value_index] - minus[value_index]) / (2.0 * eps)
        np.testing.assert_allclose(
            finite_difference,
            result[derivative_index][:, 0],
            rtol=2e-6,
            atol=2e-8,
        )
    assert "conditional_factor_reconstruction_residual" in result[-1]
    assert float(tf.reduce_max(result[-1]["stack_reconstruction_residual"])) < 1e-11


def test_block_qr_rejects_bad_dimensions_and_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        batched_block_qr_conditional(tf.zeros([1, 2, 3], tf.float64), tf.zeros([1, 3, 3], tf.float64))
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        batched_block_qr_conditional(
            tf.constant(np.nan, shape=[1, 2, 5], dtype=tf.float64),
            tf.zeros([1, 3, 5], tf.float64),
        )


def test_block_qr_relative_pivot_floor_fails_closed() -> None:
    observation = tf.constant([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0e-14, 0.0, 0.0]]], tf.float64)
    state = tf.constant([[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]], tf.float64)
    with pytest.raises(tf.errors.InvalidArgumentError, match="qr_relative_pivot"):
        batched_block_qr_conditional(
            observation,
            state,
            relative_pivot_tolerance=1.0e-8,
            compute_covariance_diagnostics=False,
        )
