import numpy as np
import tensorflow as tf

from bayesfilter import StatePartition, affine_structural_to_linear_gaussian_tf
from bayesfilter.linear.kalman_qr_tf import tf_qr_linear_gaussian_log_likelihood
from bayesfilter.structural_tf import make_affine_structural_tf


def _affine_ar2_pair():
    partition = StatePartition(
        state_names=("m", "lag_m"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    phi1 = tf.constant(0.35, dtype=tf.float64)
    phi2 = tf.constant(-0.10, dtype=tf.float64)
    sigma = tf.constant(0.25, dtype=tf.float64)
    transition_matrix = tf.stack(
        [
            tf.stack([phi1, phi2]),
            tf.constant([1.0, 0.0], dtype=tf.float64),
        ]
    )
    innovation_matrix = tf.reshape(tf.stack([sigma, tf.constant(0.0, dtype=tf.float64)]), [2, 1])
    structural = make_affine_structural_tf(
        partition=partition,
        initial_mean=tf.zeros([2], dtype=tf.float64),
        initial_covariance=tf.eye(2, dtype=tf.float64),
        transition_offset=tf.zeros([2], dtype=tf.float64),
        transition_matrix=transition_matrix,
        innovation_matrix=innovation_matrix,
        innovation_covariance=tf.eye(1, dtype=tf.float64),
        observation_offset=tf.zeros([1], dtype=tf.float64),
        observation_matrix=tf.constant([[1.0, 0.0]], dtype=tf.float64),
        observation_covariance=tf.constant([[0.15**2]], dtype=tf.float64),
    )
    linear = affine_structural_to_linear_gaussian_tf(structural)
    return structural, linear


def test_affine_structural_lgssm_reduces_to_linear_backend_value() -> None:
    structural, linear = _affine_ar2_pair()
    direct_linear = affine_structural_to_linear_gaussian_tf(structural)
    observations = tf.constant([[0.2], [0.05], [-0.1], [0.15], [0.0]], dtype=tf.float64)

    from_structural = tf_qr_linear_gaussian_log_likelihood(
        observations,
        direct_linear,
        backend="tf_qr",
        jitter=tf.constant(1e-9, dtype=tf.float64),
    )
    from_linear = tf_qr_linear_gaussian_log_likelihood(
        observations,
        linear,
        backend="tf_qr",
        jitter=tf.constant(1e-9, dtype=tf.float64),
    )

    np.testing.assert_allclose(
        from_structural.log_likelihood.numpy(),
        from_linear.log_likelihood.numpy(),
        atol=1e-12,
    )
    assert from_structural.metadata.partition is structural.partition
    assert from_structural.metadata.integration_space == "full_state"


def test_affine_structural_transition_covariance_is_declared_pushforward() -> None:
    structural, linear = _affine_ar2_pair()
    expected_covariance = (
        structural.innovation_matrix
        @ structural.innovation_covariance
        @ tf.transpose(structural.innovation_matrix)
    )

    np.testing.assert_allclose(
        linear.transition_covariance.numpy(),
        expected_covariance.numpy(),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        linear.transition_covariance.numpy(),
        np.array([[0.25**2, 0.0], [0.0, 0.0]]),
        atol=1e-14,
    )


def test_affine_structural_observation_map_accepts_batched_points() -> None:
    structural, _linear = _affine_ar2_pair()
    points = tf.constant([[0.2, 0.1], [-0.3, 0.4]], dtype=tf.float64)

    observations = structural.observe(points)

    np.testing.assert_allclose(observations.numpy(), np.array([[0.2], [-0.3]]), atol=1e-14)
