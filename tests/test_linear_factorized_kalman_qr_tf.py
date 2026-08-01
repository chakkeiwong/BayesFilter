from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.linear.kalman_qr_tf import (
    tf_qr_sqrt_factorized_kalman_log_likelihood,
    tf_qr_sqrt_factorized_kalman_log_likelihood_with_increments,
    tf_qr_sqrt_kalman_log_likelihood_while_loop,
)


def _inputs():
    dtype = tf.float64
    observations = tf.constant(
        [[0.2, -0.1], [0.05, 0.08], [-0.03, 0.04]], dtype=dtype
    )
    transition_offset = tf.constant([0.01, -0.02], dtype=dtype)
    transition_matrix = tf.constant([[0.7, 0.1], [-0.2, 0.5]], dtype=dtype)
    transition_noise_factor = tf.constant([[0.3], [0.15]], dtype=dtype)
    observation_offset = tf.constant([0.0, 0.03], dtype=dtype)
    observation_matrix = tf.constant([[1.0, 0.2], [-0.1, 0.8]], dtype=dtype)
    observation_covariance = tf.constant([[0.09, 0.01], [0.01, 0.12]], dtype=dtype)
    initial_mean = tf.constant([0.0, 0.0], dtype=dtype)
    initial_factor = tf.linalg.cholesky(
        tf.constant([[0.4, 0.05], [0.05, 0.3]], dtype=dtype)
    )
    return (
        observations,
        transition_offset,
        transition_matrix,
        transition_noise_factor,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_mean,
        initial_factor,
    )


def test_rectangular_factor_matches_full_rank_covariance_qr_value_and_gradient():
    values = _inputs()
    factor0 = values[3]
    covariance = factor0 @ tf.transpose(factor0)
    covariance += tf.eye(2, dtype=tf.float64) * tf.constant(0.04, tf.float64)
    factor = tf.linalg.cholesky(covariance)

    @tf.function(
        input_signature=(tf.TensorSpec((2, 2), tf.float64),),
        jit_compile=True,
        autograph=False,
    )
    def value_and_gradient(process_factor):
        with tf.GradientTape() as factor_tape:
            factor_tape.watch(process_factor)
            value = tf_qr_sqrt_factorized_kalman_log_likelihood(
                values[0],
                values[1],
                values[2],
                process_factor,
                *values[4:],
                jitter=0.0,
            )
        return value, factor_tape.gradient(value, process_factor)

    factorized, factor_gradient = value_and_gradient(factor)

    reference = tf_qr_sqrt_kalman_log_likelihood_while_loop(
        observations=values[0],
        transition_offset=values[1],
        transition_matrix=values[2],
        transition_covariance=covariance,
        observation_offset=values[4],
        observation_matrix=values[5],
        observation_covariance=values[6],
        initial_state_mean=values[7],
        initial_state_covariance=values[8] @ tf.transpose(values[8]),
        jitter=0.0,
    )
    np.testing.assert_allclose(factorized.numpy(), reference.numpy(), atol=2.0e-12)
    assert factor_gradient.shape == factor.shape
    assert bool(tf.reduce_all(tf.math.is_finite(factor_gradient)).numpy())


def test_rank_deficient_factor_is_finite_and_increments_sum_to_total():
    values = _inputs()
    total, increments = (
        tf_qr_sqrt_factorized_kalman_log_likelihood_with_increments(
            *values, jitter=0.0
        )
    )
    assert np.isfinite(total.numpy())
    assert increments.shape == (3,)
    np.testing.assert_allclose(total.numpy(), tf.reduce_sum(increments).numpy(), atol=0.0)


def test_factorized_qr_rejects_bad_factor_shapes_and_has_no_host_callback():
    values = _inputs()
    with pytest.raises(ValueError, match="rank 2"):
        tf_qr_sqrt_factorized_kalman_log_likelihood(
            values[0], values[1], values[2], tf.ones((2,), tf.float64), *values[4:]
        )
    with pytest.raises(ValueError, match="first dimension"):
        tf_qr_sqrt_factorized_kalman_log_likelihood(
            values[0], values[1], values[2], tf.ones((3, 1), tf.float64), *values[4:]
        )

    concrete = tf_qr_sqrt_factorized_kalman_log_likelihood.get_concrete_function(
        *values, jitter=tf.constant(0.0, tf.float64)
    )
    operation_types = {operation.type for operation in concrete.graph.get_operations()}
    assert not operation_types.intersection({"PyFunc", "EagerPyFunc"})
    source = inspect.getsource(
        __import__(
            "bayesfilter.linear.kalman_qr_tf", fromlist=["kalman_qr_tf"]
        )
    )
    assert "import numpy" not in source
    assert "tf.vectorized_map" not in source
