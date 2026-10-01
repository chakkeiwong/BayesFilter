"""TensorFlow Kalman reference for the experimental LGSSM fixture."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.fixtures.lgssm_tf import (
    DTYPE,
    LGSSMTFFixture,
    gaussian_logpdf_zero_mean_tf,
)


@dataclass(frozen=True)
class KalmanTFResult:
    filtered_means: tf.Tensor
    filtered_covariances: tf.Tensor
    predictive_means: tf.Tensor
    predictive_covariances: tf.Tensor
    log_likelihood: tf.Tensor
    finite: bool
    reference_id: str = "kalman_lgssm_tf_exact_reference"


def run_kalman_filter_tf(fixture: LGSSMTFFixture) -> KalmanTFResult:
    m_prev = tf.identity(fixture.m0)
    p_prev = tf.identity(fixture.P0)
    eye = tf.eye(fixture.state_dim, dtype=DTYPE)
    filtered_means = []
    filtered_covariances = []
    predictive_means = []
    predictive_covariances = []
    log_likelihood = tf.constant(0.0, dtype=DTYPE)

    for observation in tf.unstack(fixture.observations, axis=0):
        m_pred = tf.linalg.matvec(fixture.A, m_prev)
        p_pred = fixture.A @ p_prev @ tf.transpose(fixture.A) + fixture.Q
        y_pred = tf.linalg.matvec(fixture.C, m_pred)
        s_mat = fixture.C @ p_pred @ tf.transpose(fixture.C) + fixture.R
        residual = tf.reshape(observation, [-1]) - y_pred
        log_likelihood = log_likelihood + gaussian_logpdf_zero_mean_tf(
            tf.reshape(residual, [1, -1]),
            s_mat,
        )[0]
        k_gain = tf.transpose(
            tf.linalg.solve(s_mat, fixture.C @ p_pred)
        )
        m_filt = m_pred + tf.linalg.matvec(k_gain, residual)
        p_filt = (eye - k_gain @ fixture.C) @ p_pred
        p_filt = _symmetrize(p_filt)
        predictive_means.append(m_pred)
        predictive_covariances.append(p_pred)
        filtered_means.append(m_filt)
        filtered_covariances.append(p_filt)
        m_prev = m_filt
        p_prev = p_filt

    filtered_means_tensor = tf.stack(filtered_means, axis=0)
    filtered_covariances_tensor = tf.stack(filtered_covariances, axis=0)
    predictive_means_tensor = tf.stack(predictive_means, axis=0)
    predictive_covariances_tensor = tf.stack(predictive_covariances, axis=0)
    finite = bool(
        tf.reduce_all(tf.math.is_finite(filtered_means_tensor)).numpy()
        and tf.reduce_all(tf.math.is_finite(filtered_covariances_tensor)).numpy()
        and tf.reduce_all(tf.math.is_finite(predictive_means_tensor)).numpy()
        and tf.reduce_all(tf.math.is_finite(predictive_covariances_tensor)).numpy()
        and tf.math.is_finite(log_likelihood).numpy()
    )
    return KalmanTFResult(
        filtered_means=filtered_means_tensor,
        filtered_covariances=filtered_covariances_tensor,
        predictive_means=predictive_means_tensor,
        predictive_covariances=predictive_covariances_tensor,
        log_likelihood=log_likelihood,
        finite=finite,
    )


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.transpose(matrix))
