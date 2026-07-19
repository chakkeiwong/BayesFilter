"""Trusted GPU-XLA regression for batched principal-root score layouts."""

from __future__ import annotations

import os

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.nonlinear import (
    tf_batched_svd_sigma_point_value_and_score_custom_gradient,
)
from bayesfilter.nonlinear.experimental_batched_svd_sigma_point_tf import (
    TFBatchedStructuralFirstDerivatives,
    TFBatchedStructuralStateSpace,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("BAYESFILTER_RUN_GPU_XLA_LAYOUT_TEST") != "1",
    reason="set BAYESFILTER_RUN_GPU_XLA_LAYOUT_TEST=1 for trusted GPU-XLA checks",
)


def _value_score_diagnostics(
    theta: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
    """Build a nonlinear batch-4 target with the DZ4 failure dimensions."""

    theta = tf.ensure_shape(tf.convert_to_tensor(theta, tf.float64), [4, 11])
    batch_size = 4
    parameter_dim = 11
    state_dim = 8
    observation_dim = 18
    phi = tf.linalg.diag(
        tf.constant([0.91, 0.87, 0.82, 0.89, 0.84, 0.79, 0.76, 0.72], tf.float64)
    )
    eye = tf.eye(state_dim, dtype=tf.float64)
    initial_covariance = tf.linalg.diag(
        tf.constant([2.0e-5, 2.4e-5, 2.8e-5, 2.2e-5, 2.6e-5, 3.0e-5, 1.8e-5, 2.1e-5], tf.float64)
    )
    innovation_covariance = tf.linalg.diag(
        tf.constant([1.1e-6, 1.3e-6, 1.5e-6, 1.2e-6, 1.4e-6, 1.6e-6, 0.9e-6, 1.0e-6], tf.float64)
    )
    projection = tf.constant(
        [
            [1.0, 0.0],
            [0.5, 0.2],
            [0.2, 0.5],
            [0.0, 1.0],
            [-0.2, 0.4],
            [0.3, -0.1],
            [0.8, 0.3],
            [-0.4, 0.7],
        ],
        tf.float64,
    )
    state_mean = theta[:, :state_dim]
    transition_offset = tf.linalg.matvec(eye - phi, state_mean)

    row_groups = tf.constant(
        np.vstack(
            (
                np.r_[np.ones(6), np.zeros(12)],
                np.r_[np.zeros(6), np.ones(6), np.zeros(6)],
                np.r_[np.zeros(12), np.ones(6)],
            )
        ),
        tf.float64,
    )
    variances = tf.exp(2.0 * theta[:, 8:11])
    observation_diagonal = tf.einsum("bg,gm->bm", variances, row_groups)
    observation_covariance = tf.linalg.diag(observation_diagonal)

    selector = tf.concat(
        (tf.zeros([8, 3], tf.float64), tf.eye(3, dtype=tf.float64)), axis=0
    )
    d_variances = 2.0 * selector[tf.newaxis, :, :] * variances[:, tf.newaxis, :]
    d_observation_covariance = tf.linalg.diag(
        tf.einsum("bpg,gm->bpm", d_variances, row_groups)
    )

    def transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        return (
            transition_offset[:, tf.newaxis, :]
            + tf.linalg.matmul(previous, phi, transpose_b=True)
            + innovation
        )

    def observe(points: tf.Tensor) -> tf.Tensor:
        nonlinear = 0.25 * points + 0.05 * tf.square(points)
        projected = tf.linalg.matmul(points, projection)
        return tf.concat((points, nonlinear, projected), axis=-1)

    def transition_state_jacobian(
        previous: tf.Tensor, innovation: tf.Tensor
    ) -> tf.Tensor:
        del innovation
        return tf.broadcast_to(
            phi[tf.newaxis, tf.newaxis, :, :],
            [batch_size, tf.shape(previous)[1], state_dim, state_dim],
        )

    def transition_innovation_jacobian(
        previous: tf.Tensor, innovation: tf.Tensor
    ) -> tf.Tensor:
        del innovation
        return tf.broadcast_to(
            eye[tf.newaxis, tf.newaxis, :, :],
            [batch_size, tf.shape(previous)[1], state_dim, state_dim],
        )

    offset_derivative = tf.transpose(
        tf.concat((eye - phi, tf.zeros([state_dim, 3], tf.float64)), axis=1)
    )

    def d_transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        del innovation
        return tf.broadcast_to(
            offset_derivative[tf.newaxis, :, tf.newaxis, :],
            [batch_size, parameter_dim, tf.shape(previous)[1], state_dim],
        )

    def observation_state_jacobian(points: tf.Tensor) -> tf.Tensor:
        point_count = tf.shape(points)[1]
        linear = tf.broadcast_to(
            eye[tf.newaxis, tf.newaxis, :, :],
            [batch_size, point_count, state_dim, state_dim],
        )
        nonlinear = tf.linalg.diag(0.25 + 0.10 * points)
        projected = tf.broadcast_to(
            tf.transpose(projection)[tf.newaxis, tf.newaxis, :, :],
            [batch_size, point_count, 2, state_dim],
        )
        return tf.concat((linear, nonlinear, projected), axis=-2)

    def d_observation(points: tf.Tensor) -> tf.Tensor:
        return tf.zeros(
            [batch_size, parameter_dim, tf.shape(points)[1], observation_dim],
            tf.float64,
        )

    d_initial_mean = tf.concat(
        (
            tf.broadcast_to(
                eye[tf.newaxis, :, :], [batch_size, state_dim, state_dim]
            ),
            tf.zeros([batch_size, 3, state_dim], tf.float64),
        ),
        axis=1,
    )
    model = TFBatchedStructuralStateSpace(
        initial_mean=state_mean,
        initial_covariance=tf.broadcast_to(
            initial_covariance[tf.newaxis, :, :], [batch_size, state_dim, state_dim]
        ),
        innovation_covariance=tf.broadcast_to(
            innovation_covariance[tf.newaxis, :, :],
            [batch_size, state_dim, state_dim],
        ),
        observation_covariance=observation_covariance,
        transition_fn=transition,
        observation_fn=observe,
        name="gpu_xla_layout_batch4_fixture",
    )
    derivatives = TFBatchedStructuralFirstDerivatives(
        d_initial_mean=d_initial_mean,
        d_initial_covariance=tf.zeros(
            [batch_size, parameter_dim, state_dim, state_dim], tf.float64
        ),
        d_innovation_covariance=tf.zeros(
            [batch_size, parameter_dim, state_dim, state_dim], tf.float64
        ),
        d_observation_covariance=d_observation_covariance,
        transition_state_jacobian_fn=transition_state_jacobian,
        transition_innovation_jacobian_fn=transition_innovation_jacobian,
        d_transition_fn=d_transition,
        observation_state_jacobian_fn=observation_state_jacobian,
        d_observation_fn=d_observation,
        name="gpu_xla_layout_batch4_derivatives",
    )
    time = tf.cast(tf.range(12)[:, tf.newaxis], tf.float64)
    columns = tf.cast(tf.range(observation_dim)[tf.newaxis, :], tf.float64)
    observations = 2.0e-3 * tf.sin(0.3 * time + 0.2 * columns)
    value, score, diagnostics = (
        tf_batched_svd_sigma_point_value_and_score_custom_gradient(
            theta,
            observations,
            model,
            derivatives,
            backend="tf_principal_sqrt_ukf",
            innovation_floor=tf.constant(1.0e-12, tf.float64),
            spectral_gap_tolerance=tf.constant(1.0e-10, tf.float64),
            fixed_null_tolerance=tf.constant(1.0e-10, tf.float64),
        )
    )
    return value, score, dict(diagnostics)


def test_batch4_nonlinear_score_gpu_xla_matches_eager() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    assert gpus, "trusted GPU-XLA layout test requires a physical GPU"
    base = np.array(
        [0.010, -0.004, 0.002, 0.008, -0.003, 0.001, 0.0005, -0.0003,
         np.log(2.0e-3), np.log(2.5e-3), np.log(3.0e-3)],
        dtype=np.float64,
    )
    offsets = np.array(
        [
            np.zeros(11),
            [1e-3, -5e-4, 2e-4, -7e-4, 3e-4, -1e-4, 2e-4, -2e-4, 0.05, -0.03, 0.02],
            [-8e-4, 2e-4, -3e-4, 6e-4, -2e-4, 2e-4, -1e-4, 3e-4, -0.04, 0.02, -0.01],
            [4e-4, 6e-4, -2e-4, -5e-4, -4e-4, 1e-4, 3e-4, 1e-4, 0.03, 0.01, -0.02],
        ],
        dtype=np.float64,
    )
    theta = tf.constant(base[np.newaxis, :] + offsets, tf.float64)
    eager_value, eager_score, eager_diagnostics = _value_score_diagnostics(theta)

    @tf.function(
        input_signature=[tf.TensorSpec([4, 11], tf.float64)],
        jit_compile=True,
        reduce_retracing=True,
    )
    def compiled(x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        value, score, diagnostics = _value_score_diagnostics(x)
        return (
            value,
            score,
            diagnostics["principal_sqrt_target_classified_invalid_count"],
            diagnostics["principal_sqrt_target_roundoff_repair_count"],
        )

    xla_value, xla_score, invalid, repairs = compiled(theta)
    np.testing.assert_allclose(
        xla_value.numpy(), eager_value.numpy(), rtol=1.0e-11, atol=1.0e-7
    )
    np.testing.assert_allclose(
        xla_score.numpy(), eager_score.numpy(), rtol=1.0e-10, atol=1.0e-6
    )
    np.testing.assert_array_equal(invalid.numpy(), np.zeros(4, dtype=np.int32))
    np.testing.assert_array_equal(repairs.numpy(), np.zeros(4, dtype=np.int32))
    np.testing.assert_array_equal(
        eager_diagnostics["principal_sqrt_target_classified_invalid_count"].numpy(),
        np.zeros(4, dtype=np.int32),
    )
    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1

