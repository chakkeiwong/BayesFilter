from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.linear.experimental_batched_kalman_tf import (
    tf_batched_kalman_filter,
    tf_batched_kalman_value_and_score,
)
from bayesfilter.linear.kalman_qr_derivatives_tf import tf_qr_linear_gaussian_score
from bayesfilter.linear.kalman_tf import tf_linear_gaussian_log_likelihood
from bayesfilter.linear.types_tf import (
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceDerivatives,
)

JITTER = 1.0e-9


def _observations() -> tf.Tensor:
    return tf.constant([[0.18], [0.05], [0.16], [0.11]], dtype=tf.float64)


def _model_and_derivatives(
    params: tf.Tensor,
) -> tuple[TFLinearGaussianStateSpace, TFLinearGaussianStateSpaceDerivatives]:
    rho_param, log_measurement_noise = tf.unstack(params)
    rho = 0.75 * tf.math.tanh(rho_param)
    drho = 0.75 * (1.0 - tf.math.tanh(rho_param) ** 2)
    d2rho = -1.5 * tf.math.tanh(rho_param) * (1.0 - tf.math.tanh(rho_param) ** 2)
    measurement_variance = tf.exp(2.0 * log_measurement_noise)
    d_measurement_variance = 2.0 * measurement_variance
    d2_measurement_variance = 4.0 * measurement_variance

    model = TFLinearGaussianStateSpace(
        initial_mean=tf.constant([0.1], dtype=tf.float64),
        initial_covariance=tf.constant([[0.35]], dtype=tf.float64),
        transition_offset=tf.constant([0.02], dtype=tf.float64),
        transition_matrix=tf.reshape(rho, [1, 1]),
        transition_covariance=tf.constant([[0.07]], dtype=tf.float64),
        observation_offset=tf.constant([0.01], dtype=tf.float64),
        observation_matrix=tf.constant([[1.2]], dtype=tf.float64),
        observation_covariance=tf.reshape(measurement_variance, [1, 1]),
    )
    derivatives = TFLinearGaussianStateSpaceDerivatives(
        d_initial_mean=tf.zeros([2, 1], dtype=tf.float64),
        d_initial_covariance=tf.zeros([2, 1, 1], dtype=tf.float64),
        d_transition_offset=tf.zeros([2, 1], dtype=tf.float64),
        d_transition_matrix=tf.reshape(tf.stack([drho, 0.0]), [2, 1, 1]),
        d_transition_covariance=tf.zeros([2, 1, 1], dtype=tf.float64),
        d_observation_offset=tf.zeros([2, 1], dtype=tf.float64),
        d_observation_matrix=tf.zeros([2, 1, 1], dtype=tf.float64),
        d_observation_covariance=tf.reshape(
            tf.stack([0.0, d_measurement_variance]),
            [2, 1, 1],
        ),
        d2_initial_mean=tf.zeros([2, 2, 1], dtype=tf.float64),
        d2_initial_covariance=tf.zeros([2, 2, 1, 1], dtype=tf.float64),
        d2_transition_offset=tf.zeros([2, 2, 1], dtype=tf.float64),
        d2_transition_matrix=tf.reshape(
            tf.stack([d2rho, 0.0, 0.0, 0.0]),
            [2, 2, 1, 1],
        ),
        d2_transition_covariance=tf.zeros([2, 2, 1, 1], dtype=tf.float64),
        d2_observation_offset=tf.zeros([2, 2, 1], dtype=tf.float64),
        d2_observation_matrix=tf.zeros([2, 2, 1, 1], dtype=tf.float64),
        d2_observation_covariance=tf.reshape(
            tf.stack([0.0, 0.0, 0.0, d2_measurement_variance]),
            [2, 2, 1, 1],
        ),
    )
    return model, derivatives


def _batch_model_and_derivatives(theta_batch: tf.Tensor) -> tuple[dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    models = []
    derivatives = []
    for row in tf.unstack(tf.convert_to_tensor(theta_batch, dtype=tf.float64), axis=0):
        model, derivative = _model_and_derivatives(row)
        models.append(model)
        derivatives.append(derivative)
    model_batch = {
        "initial_state_mean": tf.stack([model.initial_mean for model in models], axis=0),
        "initial_state_covariance": tf.stack(
            [model.initial_covariance for model in models],
            axis=0,
        ),
        "transition_offset": tf.stack([model.transition_offset for model in models], axis=0),
        "transition_matrix": tf.stack([model.transition_matrix for model in models], axis=0),
        "transition_covariance": tf.stack(
            [model.transition_covariance for model in models],
            axis=0,
        ),
        "observation_offset": tf.stack([model.observation_offset for model in models], axis=0),
        "observation_matrix": tf.stack([model.observation_matrix for model in models], axis=0),
        "observation_covariance": tf.stack(
            [model.observation_covariance for model in models],
            axis=0,
        ),
    }
    derivative_batch = {
        "d_initial_state_mean": tf.stack(
            [derivative.d_initial_mean for derivative in derivatives],
            axis=0,
        ),
        "d_initial_state_covariance": tf.stack(
            [derivative.d_initial_covariance for derivative in derivatives],
            axis=0,
        ),
        "d_transition_offset": tf.stack(
            [derivative.d_transition_offset for derivative in derivatives],
            axis=0,
        ),
        "d_transition_matrix": tf.stack(
            [derivative.d_transition_matrix for derivative in derivatives],
            axis=0,
        ),
        "d_transition_covariance": tf.stack(
            [derivative.d_transition_covariance for derivative in derivatives],
            axis=0,
        ),
        "d_observation_offset": tf.stack(
            [derivative.d_observation_offset for derivative in derivatives],
            axis=0,
        ),
        "d_observation_matrix": tf.stack(
            [derivative.d_observation_matrix for derivative in derivatives],
            axis=0,
        ),
        "d_observation_covariance": tf.stack(
            [derivative.d_observation_covariance for derivative in derivatives],
            axis=0,
        ),
    }
    return model_batch, derivative_batch


def _batched_value(theta_batch: tf.Tensor, *, return_filtered: bool = False):
    model_batch, _derivative_batch = _batch_model_and_derivatives(theta_batch)
    return tf_batched_kalman_filter(
        _observations(),
        jitter=tf.constant(JITTER, dtype=tf.float64),
        return_filtered=return_filtered,
        **model_batch,
    )


def _batched_value_and_score(theta_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    model_batch, derivative_batch = _batch_model_and_derivatives(theta_batch)
    return tf_batched_kalman_value_and_score(
        _observations(),
        jitter=tf.constant(JITTER, dtype=tf.float64),
        **model_batch,
        **derivative_batch,
    )


def _scalar_value_and_score(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    model, derivatives = _model_and_derivatives(theta)
    value = tf_linear_gaussian_log_likelihood(
        _observations(),
        model,
        jitter=tf.constant(JITTER, dtype=tf.float64),
    ).log_likelihood
    score_result = tf_qr_linear_gaussian_score(
        _observations(),
        model,
        derivatives,
        jitter=tf.constant(JITTER, dtype=tf.float64),
    )
    return value, score_result.score


def _scalar_filtered(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    model, _derivatives = _model_and_derivatives(theta)
    result = tf_linear_gaussian_log_likelihood(
        _observations(),
        model,
        jitter=tf.constant(JITTER, dtype=tf.float64),
        return_filtered=True,
    )
    assert result.filtered_means is not None
    assert result.filtered_covariances is not None
    return result.filtered_means, result.filtered_covariances


def _theta_batch() -> tf.Tensor:
    return tf.constant(
        [
            [0.25, -1.10],
            [0.05, -1.25],
            [0.40, -0.95],
        ],
        dtype=tf.float64,
    )


def test_experimental_batched_kalman_value_matches_scalar_rows() -> None:
    theta_batch = _theta_batch()
    batched_value, _means, _covariances = _batched_value(theta_batch)
    expected = tf.stack(
        [_scalar_value_and_score(row)[0] for row in tf.unstack(theta_batch)],
        axis=0,
    )

    np.testing.assert_allclose(batched_value.numpy(), expected.numpy(), atol=1e-10)


def test_experimental_batched_kalman_filtered_states_match_scalar_rows() -> None:
    theta_batch = _theta_batch()
    _value, filtered_means, filtered_covariances = _batched_value(
        theta_batch,
        return_filtered=True,
    )
    assert filtered_means is not None
    assert filtered_covariances is not None
    expected_means = []
    expected_covariances = []
    for row in tf.unstack(theta_batch):
        means, covariances = _scalar_filtered(row)
        expected_means.append(means)
        expected_covariances.append(covariances)

    np.testing.assert_allclose(
        filtered_means.numpy(),
        tf.stack(expected_means, axis=1).numpy(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        filtered_covariances.numpy(),
        tf.stack(expected_covariances, axis=1).numpy(),
        atol=1e-10,
    )


def test_experimental_batched_kalman_value_and_score_matches_scalar_qr_score() -> None:
    theta_batch = _theta_batch()
    batched_value, batched_score = _batched_value_and_score(theta_batch)
    expected_values = []
    expected_scores = []
    for row in tf.unstack(theta_batch):
        value, score = _scalar_value_and_score(row)
        expected_values.append(value)
        expected_scores.append(score)

    np.testing.assert_allclose(
        batched_value.numpy(),
        tf.stack(expected_values, axis=0).numpy(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        batched_score.numpy(),
        tf.stack(expected_scores, axis=0).numpy(),
        rtol=1e-8,
        atol=1e-8,
    )


def test_experimental_batched_kalman_value_and_score_supports_singleton_batch() -> None:
    theta_batch = _theta_batch()[:1]
    batched_value, batched_score = _batched_value_and_score(theta_batch)
    expected_value, expected_score = _scalar_value_and_score(theta_batch[0])

    np.testing.assert_allclose(
        batched_value.numpy(),
        expected_value[tf.newaxis].numpy(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        batched_score.numpy(),
        expected_score[tf.newaxis, :].numpy(),
        rtol=1e-8,
        atol=1e-8,
    )


def test_experimental_batched_kalman_row_permutation_preserves_order() -> None:
    theta_batch = _theta_batch()
    base_value, base_score = _batched_value_and_score(theta_batch)
    permutation = tf.constant([2, 0, 1], dtype=tf.int32)
    permuted_value, permuted_score = _batched_value_and_score(
        tf.gather(theta_batch, permutation)
    )

    np.testing.assert_allclose(
        permuted_value.numpy(),
        tf.gather(base_value, permutation).numpy(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        permuted_score.numpy(),
        tf.gather(base_score, permutation).numpy(),
        atol=1e-10,
    )


def test_experimental_batched_kalman_score_matches_tape_jacobian_reference() -> None:
    theta_batch = _theta_batch()

    def scalar_sum(values: tf.Tensor) -> tf.Tensor:
        batch_value, _batch_score = _batched_value_and_score(values)
        return tf.reduce_sum(batch_value)

    # Differentiate inside the same compiled boundary as the recurrence.
    @tf.function(jit_compile=True)
    def compiled_reference(values):
        with tf.GradientTape() as tape:
            tape.watch(values)
            value = scalar_sum(values)
        return tape.gradient(value, values)

    autodiff_score = compiled_reference(theta_batch)
    _batched_value, analytic_score = _batched_value_and_score(theta_batch)

    assert autodiff_score is not None
    np.testing.assert_allclose(
        analytic_score.numpy(),
        autodiff_score.numpy(),
        rtol=1e-8,
        atol=1e-8,
    )


def test_experimental_batched_kalman_graph_and_cpu_xla_parity() -> None:
    theta_batch = _theta_batch()
    eager_value, eager_score = _batched_value_and_score(theta_batch)

    @tf.function(reduce_retracing=True)
    def graph(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return _batched_value_and_score(values)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def xla(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return _batched_value_and_score(values)

    graph_value, graph_score = graph(theta_batch)
    xla_value, xla_score = xla(theta_batch)

    np.testing.assert_allclose(graph_value.numpy(), eager_value.numpy(), atol=1e-10)
    np.testing.assert_allclose(graph_score.numpy(), eager_score.numpy(), atol=1e-10)
    np.testing.assert_allclose(xla_value.numpy(), eager_value.numpy(), atol=1e-10)
    np.testing.assert_allclose(xla_score.numpy(), eager_score.numpy(), atol=1e-10)
    assert len(graph._list_all_concrete_functions_for_serialization()) == 1


def test_experimental_batched_kalman_shape_mismatch_fails_closed() -> None:
    model_batch, derivative_batch = _batch_model_and_derivatives(_theta_batch())
    bad_derivatives = dict(derivative_batch)
    bad_derivatives["d_observation_matrix"] = bad_derivatives["d_observation_matrix"][
        :,
        :1,
    ]

    with pytest.raises((tf.errors.InvalidArgumentError, ValueError), match="shape|rank"):
        tf_batched_kalman_value_and_score(
            _observations(),
            jitter=tf.constant(JITTER, dtype=tf.float64),
            **model_batch,
            **bad_derivatives,
        )


def test_experimental_batched_kalman_cpu_only_hides_gpu() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert tf.config.list_physical_devices("GPU") == []
