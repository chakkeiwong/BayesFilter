from __future__ import annotations

import inspect

import numpy as np
import tensorflow as tf

import bayesfilter.linear as linear
from bayesfilter.linear.experimental_batched_kalman_tf import (
    tf_batched_kalman_value_and_score,
)
from bayesfilter.linear.kalman_covariance_derivatives_tf import (
    tf_batched_covariance_kalman_value_and_score,
)


def _payload(batch: int = 2, parameter_dim: int = 3) -> dict[str, tf.Tensor]:
    state_dim = 2
    observation_dim = 1
    def zeros(shape: list[int]) -> tf.Tensor:
        return tf.zeros(shape, dtype=tf.float64)
    identity = tf.eye(state_dim, batch_shape=[batch], dtype=tf.float64)
    observation_covariance = tf.eye(
        observation_dim, batch_shape=[batch], dtype=tf.float64
    ) * 0.2
    return {
        "observations": zeros([4, observation_dim]),
        "transition_offset": zeros([batch, state_dim]),
        "transition_matrix": identity * 0.5,
        "transition_covariance": zeros([batch, state_dim, state_dim]),
        "observation_offset": zeros([batch, observation_dim]),
        "observation_matrix": tf.ones(
            [batch, observation_dim, state_dim], dtype=tf.float64
        ),
        "observation_covariance": observation_covariance,
        "initial_state_mean": zeros([batch, state_dim]),
        "initial_state_covariance": zeros([batch, state_dim, state_dim]),
        "d_initial_state_mean": zeros([batch, parameter_dim, state_dim]),
        "d_initial_state_covariance": zeros(
            [batch, parameter_dim, state_dim, state_dim]
        ),
        "d_transition_offset": zeros([batch, parameter_dim, state_dim]),
        "d_transition_matrix": zeros(
            [batch, parameter_dim, state_dim, state_dim]
        ),
        "d_transition_covariance": zeros(
            [batch, parameter_dim, state_dim, state_dim]
        ),
        "d_observation_offset": zeros([batch, parameter_dim, observation_dim]),
        "d_observation_matrix": zeros(
            [batch, parameter_dim, observation_dim, state_dim]
        ),
        "d_observation_covariance": zeros(
            [batch, parameter_dim, observation_dim, observation_dim]
        ),
        "jitter": tf.constant(0.0, tf.float64),
    }


def test_covariance_kernel_is_public_and_not_experimental() -> None:
    assert "tf_batched_covariance_kalman_value_and_score" in linear.__all__
    assert hasattr(linear, "tf_batched_covariance_kalman_value_and_score")
    assert "tf_batched_covariance_kalman_filter" not in linear.__all__
    source = inspect.getsource(
        tf_batched_covariance_kalman_value_and_score.python_function
    )
    assert "GradientTape" not in source
    assert "ForwardAccumulator" not in source
    assert "tf_batched_kalman_value_and_score" not in source


def test_covariance_kernel_supports_positive_semidefinite_state_covariances() -> None:
    payload = _payload()
    value, score = tf_batched_covariance_kalman_value_and_score(**payload)
    assert value.shape == (2,)
    assert score.shape == (2, 3)
    tf.debugging.assert_all_finite(value, "PSD covariance value must be finite")
    tf.debugging.assert_all_finite(score, "PSD covariance score must be finite")


def test_experimental_compatibility_name_matches_reviewed_kernel() -> None:
    payload = _payload()
    reviewed_value, reviewed_score = (
        tf_batched_covariance_kalman_value_and_score(**payload)
    )
    compatibility_value, compatibility_score = tf_batched_kalman_value_and_score(
        **payload
    )
    np.testing.assert_allclose(
        compatibility_value.numpy(), reviewed_value.numpy(), atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        compatibility_score.numpy(), reviewed_score.numpy(), atol=0.0, rtol=0.0
    )


def test_covariance_kernel_target_only_xla_parity() -> None:
    payload = _payload()
    eager_value, eager_score = tf_batched_covariance_kalman_value_and_score(
        **payload
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(**kwargs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return tf_batched_covariance_kalman_value_and_score(**kwargs)

    xla_value, xla_score = compiled(**payload)
    tf.debugging.assert_near(xla_value, eager_value, atol=1.0e-10, rtol=1.0e-10)
    tf.debugging.assert_near(xla_score, eager_score, atol=1.0e-10, rtol=1.0e-10)
