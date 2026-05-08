from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.linear.kalman_qr_derivatives_tf import (
    tf_qr_linear_gaussian_score_hessian,
    tf_qr_sqrt_kalman_score_hessian,
)
from bayesfilter.linear.kalman_qr_tf import tf_qr_sqrt_kalman_log_likelihood
from bayesfilter.linear.types_tf import (
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceDerivatives,
)
from bayesfilter.testing.tf_solve_differentiated_kalman_reference import (
    tf_solve_differentiated_kalman_loglik,
)


ROOT = Path(__file__).resolve().parents[1]
JITTER = 1e-9


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


def _observations() -> tf.Tensor:
    return tf.constant([[0.18], [0.05], [0.16], [0.11]], dtype=tf.float64)


def _qr_log_likelihood(observations: tf.Tensor, model: TFLinearGaussianStateSpace) -> tf.Tensor:
    return tf_qr_sqrt_kalman_log_likelihood(
        observations=observations,
        transition_offset=model.transition_offset,
        transition_matrix=model.transition_matrix,
        transition_covariance=model.transition_covariance,
        observation_offset=model.observation_offset,
        observation_matrix=model.observation_matrix,
        observation_covariance=model.observation_covariance,
        initial_state_mean=model.initial_mean,
        initial_state_covariance=model.initial_covariance,
        jitter=tf.constant(JITTER, dtype=tf.float64),
    )


def _autodiff_reference(params: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as hessian_tape:
        hessian_tape.watch(params)
        with tf.GradientTape() as gradient_tape:
            gradient_tape.watch(params)
            model, _ = _model_and_derivatives(params)
            value = _qr_log_likelihood(_observations(), model)
        gradient = gradient_tape.gradient(value, params)
    hessian = hessian_tape.jacobian(gradient, params)
    return value, gradient, hessian


def test_qr_score_hessian_matches_value_and_solve_reference() -> None:
    params = tf.constant([0.25, -1.1], dtype=tf.float64)
    model, derivatives = _model_and_derivatives(params)

    qr_loglik, qr_score, qr_hessian = tf_qr_sqrt_kalman_score_hessian(
        observations=_observations(),
        transition_offset=model.transition_offset,
        transition_matrix=model.transition_matrix,
        transition_covariance=model.transition_covariance,
        observation_offset=model.observation_offset,
        observation_matrix=model.observation_matrix,
        observation_covariance=model.observation_covariance,
        initial_state_mean=model.initial_mean,
        initial_state_covariance=model.initial_covariance,
        d_initial_state_mean=derivatives.d_initial_mean,
        d_initial_state_covariance=derivatives.d_initial_covariance,
        d_transition_offset=derivatives.d_transition_offset,
        d_transition_matrix=derivatives.d_transition_matrix,
        d_transition_covariance=derivatives.d_transition_covariance,
        d_observation_offset=derivatives.d_observation_offset,
        d_observation_matrix=derivatives.d_observation_matrix,
        d_observation_covariance=derivatives.d_observation_covariance,
        d2_initial_state_mean=derivatives.d2_initial_mean,
        d2_initial_state_covariance=derivatives.d2_initial_covariance,
        d2_transition_offset=derivatives.d2_transition_offset,
        d2_transition_matrix=derivatives.d2_transition_matrix,
        d2_transition_covariance=derivatives.d2_transition_covariance,
        d2_observation_offset=derivatives.d2_observation_offset,
        d2_observation_matrix=derivatives.d2_observation_matrix,
        d2_observation_covariance=derivatives.d2_observation_covariance,
        jitter=tf.constant(JITTER, dtype=tf.float64),
    )
    value_loglik = _qr_log_likelihood(_observations(), model)
    solve_loglik, solve_score, solve_hessian = tf_solve_differentiated_kalman_loglik(
        _observations(),
        model,
        derivatives,
        jitter=JITTER,
    )

    np.testing.assert_allclose(qr_loglik.numpy(), value_loglik.numpy(), atol=1e-10)
    np.testing.assert_allclose(qr_loglik.numpy(), solve_loglik.numpy(), atol=1e-10)
    np.testing.assert_allclose(qr_score.numpy(), solve_score.numpy(), rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(
        qr_hessian.numpy(),
        solve_hessian.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_allclose(qr_hessian.numpy(), qr_hessian.numpy().T, atol=1e-12)


def test_qr_score_hessian_matches_autodiff_on_tiny_model() -> None:
    params = tf.constant([0.25, -1.1], dtype=tf.float64)
    model, derivatives = _model_and_derivatives(params)

    qr_loglik, qr_score, qr_hessian = tf_qr_sqrt_kalman_score_hessian(
        observations=_observations(),
        transition_offset=model.transition_offset,
        transition_matrix=model.transition_matrix,
        transition_covariance=model.transition_covariance,
        observation_offset=model.observation_offset,
        observation_matrix=model.observation_matrix,
        observation_covariance=model.observation_covariance,
        initial_state_mean=model.initial_mean,
        initial_state_covariance=model.initial_covariance,
        d_initial_state_mean=derivatives.d_initial_mean,
        d_initial_state_covariance=derivatives.d_initial_covariance,
        d_transition_offset=derivatives.d_transition_offset,
        d_transition_matrix=derivatives.d_transition_matrix,
        d_transition_covariance=derivatives.d_transition_covariance,
        d_observation_offset=derivatives.d_observation_offset,
        d_observation_matrix=derivatives.d_observation_matrix,
        d_observation_covariance=derivatives.d_observation_covariance,
        d2_initial_state_mean=derivatives.d2_initial_mean,
        d2_initial_state_covariance=derivatives.d2_initial_covariance,
        d2_transition_offset=derivatives.d2_transition_offset,
        d2_transition_matrix=derivatives.d2_transition_matrix,
        d2_transition_covariance=derivatives.d2_transition_covariance,
        d2_observation_offset=derivatives.d2_observation_offset,
        d2_observation_matrix=derivatives.d2_observation_matrix,
        d2_observation_covariance=derivatives.d2_observation_covariance,
        jitter=tf.constant(JITTER, dtype=tf.float64),
    )
    autodiff_loglik, autodiff_score, autodiff_hessian = _autodiff_reference(params)

    np.testing.assert_allclose(qr_loglik.numpy(), autodiff_loglik.numpy(), atol=1e-10)
    np.testing.assert_allclose(
        qr_score.numpy(),
        autodiff_score.numpy(),
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        qr_hessian.numpy(),
        autodiff_hessian.numpy(),
        rtol=1e-5,
        atol=1e-6,
    )


def test_qr_derivative_wrapper_metadata_and_backend_validation() -> None:
    params = tf.constant([0.25, -1.1], dtype=tf.float64)
    model, derivatives = _model_and_derivatives(params)

    result = tf_qr_linear_gaussian_score_hessian(
        _observations(),
        model,
        derivatives,
        jitter=tf.constant(JITTER, dtype=tf.float64),
    )

    assert result.metadata.filter_name == "tf_qr_sqrt_differentiated_kalman"
    assert result.metadata.differentiability_status == "analytic_score_hessian"
    assert result.metadata.compiled_status == "tf_function"
    assert result.diagnostics.regularization.branch_label == "qr_square_root"
    assert result.hessian is not None

    with pytest.raises(ValueError, match="unknown TensorFlow QR derivative backend"):
        tf_qr_linear_gaussian_score_hessian(
            _observations(),
            model,
            derivatives,
            backend="not_qr_sqrt",
        )


def test_qr_derivative_tf_function_reuses_same_shape_concrete_function() -> None:
    params = tf.constant([0.25, -1.1], dtype=tf.float64)
    model, derivatives = _model_and_derivatives(params)
    observations = _observations()
    eager = tf_qr_sqrt_kalman_score_hessian(
        observations=observations,
        transition_offset=model.transition_offset,
        transition_matrix=model.transition_matrix,
        transition_covariance=model.transition_covariance,
        observation_offset=model.observation_offset,
        observation_matrix=model.observation_matrix,
        observation_covariance=model.observation_covariance,
        initial_state_mean=model.initial_mean,
        initial_state_covariance=model.initial_covariance,
        d_initial_state_mean=derivatives.d_initial_mean,
        d_initial_state_covariance=derivatives.d_initial_covariance,
        d_transition_offset=derivatives.d_transition_offset,
        d_transition_matrix=derivatives.d_transition_matrix,
        d_transition_covariance=derivatives.d_transition_covariance,
        d_observation_offset=derivatives.d_observation_offset,
        d_observation_matrix=derivatives.d_observation_matrix,
        d_observation_covariance=derivatives.d_observation_covariance,
        d2_initial_state_mean=derivatives.d2_initial_mean,
        d2_initial_state_covariance=derivatives.d2_initial_covariance,
        d2_transition_offset=derivatives.d2_transition_offset,
        d2_transition_matrix=derivatives.d2_transition_matrix,
        d2_transition_covariance=derivatives.d2_transition_covariance,
        d2_observation_offset=derivatives.d2_observation_offset,
        d2_observation_matrix=derivatives.d2_observation_matrix,
        d2_observation_covariance=derivatives.d2_observation_covariance,
        jitter=tf.constant(JITTER, dtype=tf.float64),
    )

    @tf.function(reduce_retracing=True)
    def compiled(observation_shift: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return tf_qr_sqrt_kalman_score_hessian(
            observations=observations + observation_shift,
            transition_offset=model.transition_offset,
            transition_matrix=model.transition_matrix,
            transition_covariance=model.transition_covariance,
            observation_offset=model.observation_offset,
            observation_matrix=model.observation_matrix,
            observation_covariance=model.observation_covariance,
            initial_state_mean=model.initial_mean,
            initial_state_covariance=model.initial_covariance,
            d_initial_state_mean=derivatives.d_initial_mean,
            d_initial_state_covariance=derivatives.d_initial_covariance,
            d_transition_offset=derivatives.d_transition_offset,
            d_transition_matrix=derivatives.d_transition_matrix,
            d_transition_covariance=derivatives.d_transition_covariance,
            d_observation_offset=derivatives.d_observation_offset,
            d_observation_matrix=derivatives.d_observation_matrix,
            d_observation_covariance=derivatives.d_observation_covariance,
            d2_initial_state_mean=derivatives.d2_initial_mean,
            d2_initial_state_covariance=derivatives.d2_initial_covariance,
            d2_transition_offset=derivatives.d2_transition_offset,
            d2_transition_matrix=derivatives.d2_transition_matrix,
            d2_transition_covariance=derivatives.d2_transition_covariance,
            d2_observation_offset=derivatives.d2_observation_offset,
            d2_observation_matrix=derivatives.d2_observation_matrix,
            d2_observation_covariance=derivatives.d2_observation_covariance,
            jitter=tf.constant(JITTER, dtype=tf.float64),
        )

    first = compiled(tf.zeros_like(observations))
    second = compiled(tf.zeros_like(observations))

    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1
    for first_value, second_value, eager_value in zip(first, second, eager):
        np.testing.assert_allclose(first_value.numpy(), eager_value.numpy(), atol=1e-10)
        np.testing.assert_allclose(second_value.numpy(), eager_value.numpy(), atol=1e-10)


def test_qr_derivative_module_does_not_import_numpy_or_call_dot_numpy() -> None:
    text = (
        ROOT / "bayesfilter" / "linear" / "kalman_qr_derivatives_tf.py"
    ).read_text(encoding="utf-8")

    assert "import numpy" not in text
    assert "from numpy" not in text
    assert ".numpy(" not in text
