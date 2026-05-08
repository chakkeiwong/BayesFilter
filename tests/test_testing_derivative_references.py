from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from bayesfilter.testing.tf_covariance_differentiated_kalman_reference import (
    tf_differentiated_kalman_loglik,
    tf_differentiated_kalman_loglik_grad,
)
from bayesfilter.testing.tf_solve_differentiated_kalman_reference import (
    tf_solve_differentiated_kalman_loglik,
)


JITTER = 1e-9


def _make_model_and_derivatives(params: tf.Tensor):
    rho_param, log_measurement_noise = tf.unstack(params)
    rho = 0.75 * tf.math.tanh(rho_param)
    drho = 0.75 * (1.0 - tf.math.tanh(rho_param) ** 2)
    d2rho = -1.5 * tf.math.tanh(rho_param) * (1.0 - tf.math.tanh(rho_param) ** 2)
    measurement_variance = tf.exp(2.0 * log_measurement_noise)
    d_measurement_variance = 2.0 * measurement_variance
    d2_measurement_variance = 4.0 * measurement_variance

    model = SimpleNamespace(
        initial_mean=tf.constant([0.1], dtype=tf.float64),
        initial_covariance=tf.constant([[0.35]], dtype=tf.float64),
        transition_offset=tf.constant([0.02], dtype=tf.float64),
        transition_matrix=tf.reshape(rho, [1, 1]),
        transition_covariance=tf.constant([[0.07]], dtype=tf.float64),
        observation_offset=tf.constant([0.01], dtype=tf.float64),
        observation_matrix=tf.constant([[1.2]], dtype=tf.float64),
        observation_covariance=tf.reshape(measurement_variance, [1, 1]),
    )
    derivatives = SimpleNamespace(
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


def _observations() -> tf.Tensor:
    return tf.constant([[0.18], [0.05], [0.16], [0.11]], dtype=tf.float64)


def _autodiff_reference(params: tf.Tensor):
    with tf.GradientTape() as hessian_tape:
        hessian_tape.watch(params)
        with tf.GradientTape() as gradient_tape:
            gradient_tape.watch(params)
            model, _ = _make_model_and_derivatives(params)
            value = tf_solve_differentiated_kalman_loglik(
                _observations(),
                model,
                _make_model_and_derivatives(tf.stop_gradient(params))[1],
                jitter=JITTER,
            )[0]
        gradient = gradient_tape.gradient(value, params)
    hessian = hessian_tape.jacobian(gradient, params)
    return value, gradient, hessian


def test_testing_solve_and_covariance_derivative_references_match_autodiff() -> None:
    params = tf.constant([0.25, -1.1], dtype=tf.float64)
    model, derivatives = _make_model_and_derivatives(params)

    solve_loglik, solve_grad, solve_hess = tf_solve_differentiated_kalman_loglik(
        _observations(),
        model,
        derivatives,
        jitter=JITTER,
    )
    cov_loglik, cov_grad, cov_hess = tf_differentiated_kalman_loglik(
        _observations(),
        model,
        derivatives,
        jitter=JITTER,
    )
    autodiff_loglik, autodiff_grad, autodiff_hess = _autodiff_reference(params)

    np.testing.assert_allclose(solve_loglik.numpy(), cov_loglik.numpy(), atol=1e-10)
    np.testing.assert_allclose(solve_grad.numpy(), cov_grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(solve_hess.numpy(), cov_hess.numpy(), atol=1e-10)
    np.testing.assert_allclose(solve_loglik.numpy(), autodiff_loglik.numpy(), atol=1e-10)
    np.testing.assert_allclose(solve_grad.numpy(), autodiff_grad.numpy(), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(solve_hess.numpy(), autodiff_hess.numpy(), rtol=1e-6, atol=1e-7)


def test_testing_covariance_score_only_reference_matches_full_score() -> None:
    params = tf.constant([0.25, -1.1], dtype=tf.float64)
    model, derivatives = _make_model_and_derivatives(params)

    full_loglik, full_grad, _ = tf_differentiated_kalman_loglik(
        _observations(),
        model,
        derivatives,
        jitter=JITTER,
    )
    score_loglik, score_grad = tf_differentiated_kalman_loglik_grad(
        _observations(),
        model,
        derivatives,
        jitter=JITTER,
    )

    np.testing.assert_allclose(score_loglik.numpy(), full_loglik.numpy(), atol=1e-10)
    np.testing.assert_allclose(score_grad.numpy(), full_grad.numpy(), atol=1e-10)


def test_testing_reference_modules_are_not_production_exports() -> None:
    import bayesfilter

    assert not hasattr(bayesfilter, "tf_solve_differentiated_kalman_loglik")
    assert not hasattr(bayesfilter, "tf_differentiated_kalman_loglik")
