import numpy as np
import pytest
import tensorflow as tf

from bayesfilter import StatePartition
from bayesfilter.nonlinear.svd_cut_derivatives_tf import tf_svd_cut4_score_hessian
from bayesfilter.nonlinear.svd_cut_tf import tf_svd_cut4_log_likelihood
from bayesfilter.structural_tf import make_affine_structural_tf


def _model_from_params(params: tf.Tensor, *, repeated_spectrum: bool = False):
    phi1 = params[0]
    sigma = params[1]
    partition = StatePartition(
        state_names=("m", "lag_m"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    initial_covariance = (
        tf.eye(2, dtype=tf.float64)
        if repeated_spectrum
        else tf.linalg.diag(tf.constant([1.2, 0.7], dtype=tf.float64))
    )
    innovation_variance = (
        tf.constant([[1.0]], dtype=tf.float64)
        if repeated_spectrum
        else tf.constant([[0.43]], dtype=tf.float64)
    )
    return make_affine_structural_tf(
        partition=partition,
        initial_mean=tf.constant([0.1, -0.2], dtype=tf.float64),
        initial_covariance=initial_covariance,
        transition_offset=tf.zeros([2], dtype=tf.float64),
        transition_matrix=tf.stack(
            [
                tf.stack([phi1, tf.constant(-0.12, dtype=tf.float64)]),
                tf.constant([1.0, 0.0], dtype=tf.float64),
            ]
        ),
        innovation_matrix=tf.reshape(tf.stack([sigma, tf.constant(0.0, dtype=tf.float64)]), [2, 1]),
        innovation_covariance=innovation_variance,
        observation_offset=tf.zeros([1], dtype=tf.float64),
        observation_matrix=tf.constant([[1.0, 0.25]], dtype=tf.float64),
        observation_covariance=tf.constant([[0.19]], dtype=tf.float64),
    )


def _value(params: tf.Tensor) -> tf.Tensor:
    value, _means, _covs, _diagnostics = tf_svd_cut4_log_likelihood(
        tf.constant([[0.2], [-0.05], [0.15]], dtype=tf.float64),
        _model_from_params(params),
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
    )
    return value


def _finite_difference_score_hessian(theta: np.ndarray, step: float = 1e-4):
    theta = np.asarray(theta, dtype=np.float64)
    p = theta.size
    score = np.zeros(p, dtype=np.float64)
    hessian = np.zeros((p, p), dtype=np.float64)

    def f(values):
        return float(_value(tf.constant(values, dtype=tf.float64)).numpy())

    center = f(theta)
    for i in range(p):
        direction = np.zeros(p, dtype=np.float64)
        direction[i] = step
        score[i] = (f(theta + direction) - f(theta - direction)) / (2.0 * step)
        hessian[i, i] = (f(theta + direction) - 2.0 * center + f(theta - direction)) / (
            step**2
        )
        for j in range(i + 1, p):
            direction_j = np.zeros(p, dtype=np.float64)
            direction_j[j] = step
            hessian[i, j] = (
                f(theta + direction + direction_j)
                - f(theta + direction - direction_j)
                - f(theta - direction + direction_j)
                + f(theta - direction - direction_j)
            ) / (4.0 * step**2)
            hessian[j, i] = hessian[i, j]
    return score, hessian


def test_svd_cut4_score_hessian_matches_finite_differences_on_smooth_branch() -> None:
    observations = tf.constant([[0.2], [-0.05], [0.15]], dtype=tf.float64)
    params = tf.constant([0.31, 0.27], dtype=tf.float64)

    result = tf_svd_cut4_score_hessian(
        observations,
        params,
        _model_from_params,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        spectral_gap_tolerance=tf.constant(1e-7, dtype=tf.float64),
    )
    fd_score, fd_hessian = _finite_difference_score_hessian(params.numpy())

    np.testing.assert_allclose(result.score.numpy(), fd_score, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(result.hessian.numpy(), fd_hessian, rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(result.hessian.numpy(), result.hessian.numpy().T, atol=1e-10)
    assert result.metadata.filter_name == "tf_svd_cut4_score_hessian"
    assert result.metadata.differentiability_status == "smooth_branch_score_hessian"
    assert result.diagnostics.regularization.derivative_target == "implemented_regularized_law"
    assert result.diagnostics.extra["derivative_branch"] == "smooth_separated_spectrum"


def test_svd_cut4_score_hessian_matches_direct_tensorflow_autodiff() -> None:
    observations = tf.constant([[0.2], [-0.05], [0.15]], dtype=tf.float64)
    params = tf.constant([0.31, 0.27], dtype=tf.float64)

    result = tf_svd_cut4_score_hessian(observations, params, _model_from_params)
    with tf.GradientTape() as outer:
        outer.watch(params)
        with tf.GradientTape() as inner:
            inner.watch(params)
            direct_value = _value(params)
        direct_score = inner.gradient(direct_value, params)
    direct_hessian = outer.jacobian(direct_score, params)

    np.testing.assert_allclose(result.log_likelihood.numpy(), direct_value.numpy(), atol=1e-12)
    np.testing.assert_allclose(result.score.numpy(), direct_score.numpy(), atol=1e-12)
    np.testing.assert_allclose(result.hessian.numpy(), direct_hessian.numpy(), atol=1e-12)


def test_svd_cut4_score_hessian_blocks_active_floor() -> None:
    observations = tf.constant([[0.2]], dtype=tf.float64)
    params = tf.constant([0.31, 0.27], dtype=tf.float64)

    with pytest.raises(tf.errors.InvalidArgumentError, match="blocked_active_floor"):
        tf_svd_cut4_score_hessian(
            observations,
            params,
            _model_from_params,
            placement_floor=tf.constant(10.0, dtype=tf.float64),
        )


def test_svd_cut4_score_hessian_blocks_weak_spectral_gap() -> None:
    observations = tf.constant([[0.2]], dtype=tf.float64)
    params = tf.constant([0.31, 0.27], dtype=tf.float64)

    def repeated_builder(values: tf.Tensor):
        return _model_from_params(values, repeated_spectrum=True)

    with pytest.raises(tf.errors.InvalidArgumentError, match="blocked_weak_spectral_gap"):
        tf_svd_cut4_score_hessian(
            observations,
            params,
            repeated_builder,
            spectral_gap_tolerance=tf.constant(1e-7, dtype=tf.float64),
        )
