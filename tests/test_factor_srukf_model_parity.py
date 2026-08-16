"""Model-level parity checks for the direct-factor SR-UKF candidate.

These are bounded diagnostic comparisons against the existing principal-root
UKF. They do not change the repository default route.
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf

from bayesfilter.nonlinear.factor_srukf_tf import (
    TFFactorSRUKFDerivatives,
    TFFactorSRUKFModel,
    tf_factor_srukf_value_and_score,
)
from bayesfilter.nonlinear.svd_sigma_point_derivatives_tf import (
    TFStructuralFirstDerivatives,
    tf_principal_sqrt_ukf_score,
)
from bayesfilter.structural import StatePartition, StructuralFilterConfig
from bayesfilter.structural_tf import make_affine_structural_tf
from bayesfilter.testing import (
    make_nonlinear_accumulation_first_derivatives_tf,
    make_nonlinear_accumulation_model_tf,
    make_univariate_nonlinear_growth_first_derivatives_tf,
    make_univariate_nonlinear_growth_model_tf,
    model_a_observations_tf,
    model_b_observations_tf,
    model_c_observations_tf,
)


def _batch_factor(value: tf.Tensor, batch: int = 1) -> tf.Tensor:
    return tf.broadcast_to(value[None, ...], [batch, *value.shape.as_list()])


def _model_a(theta: tf.Tensor) -> tuple[TFFactorSRUKFModel, TFFactorSRUKFDerivatives, object, TFStructuralFirstDerivatives]:
    theta = tf.convert_to_tensor(theta, tf.float64)
    partition = StatePartition(
        state_names=("m", "lag_m"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    config = StructuralFilterConfig(
        integration_space="innovation",
        deterministic_completion="required",
        approximation_label="model_a_factor_parity",
    )
    transition_matrix = tf.constant([[0.35, -0.10], [1.0, 0.0]], tf.float64)
    innovation_matrix = tf.constant([[0.25], [0.0]], tf.float64)

    def transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        return tf.einsum("ij,brj->bri", transition_matrix, previous) + tf.einsum(
            "iq,brq->bri", innovation_matrix, innovation
        )

    def observe(states: tf.Tensor) -> tf.Tensor:
        return states[:, :, 0:1] + theta

    old_model = make_affine_structural_tf(
        partition=partition,
        initial_mean=tf.zeros([2], tf.float64),
        initial_covariance=tf.eye(2, dtype=tf.float64),
        transition_offset=tf.zeros([2], tf.float64),
        transition_matrix=transition_matrix,
        innovation_matrix=innovation_matrix,
        innovation_covariance=tf.ones([1, 1], tf.float64),
        observation_offset=tf.reshape(theta, [1]),
        observation_matrix=tf.constant([[1.0, 0.0]], tf.float64),
        observation_covariance=tf.constant([[0.15**2]], tf.float64),
    )

    def old_transition_jac(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        del innovation
        return tf.broadcast_to(transition_matrix[None, :, :], [tf.shape(previous)[0], 2, 2])

    def old_innovation_jac(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        del previous
        return tf.broadcast_to(innovation_matrix[None, :, :], [tf.shape(innovation)[0], 2, 1])

    def old_d_transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        del innovation
        return tf.zeros([1, tf.shape(previous)[0], 2], tf.float64)

    def old_observation_jac(states: tf.Tensor) -> tf.Tensor:
        return tf.broadcast_to(tf.constant([[[1.0, 0.0]]], tf.float64), [tf.shape(states)[0], 1, 2])

    def old_d_observation(states: tf.Tensor) -> tf.Tensor:
        return tf.ones([1, tf.shape(states)[0], 1], tf.float64)

    old_derivatives = TFStructuralFirstDerivatives(
        d_initial_mean=tf.zeros([1, 2], tf.float64),
        d_initial_covariance=tf.zeros([1, 2, 2], tf.float64),
        d_innovation_covariance=tf.zeros([1, 1, 1], tf.float64),
        d_observation_covariance=tf.zeros([1, 1, 1], tf.float64),
        transition_state_jacobian_fn=old_transition_jac,
        transition_innovation_jacobian_fn=old_innovation_jac,
        d_transition_fn=old_d_transition,
        observation_state_jacobian_fn=old_observation_jac,
        d_observation_fn=old_d_observation,
        name="model_a_parity_derivatives",
    )

    def transition_jac(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        return tf.broadcast_to(transition_matrix[None, None, :, :], [1, tf.shape(previous)[1], 2, 2])

    def innovation_jac(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        return tf.broadcast_to(innovation_matrix[None, None, :, :], [1, tf.shape(previous)[1], 2, 1])

    def d_transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        return tf.zeros([1, 1, tf.shape(previous)[1], 2], tf.float64)

    def observation_jac(states: tf.Tensor) -> tf.Tensor:
        return tf.broadcast_to(tf.constant([[[1.0, 0.0]]], tf.float64)[None, ...], [1, tf.shape(states)[1], 1, 2])

    def d_observation(states: tf.Tensor) -> tf.Tensor:
        return tf.ones([1, 1, tf.shape(states)[1], 1], tf.float64)

    factor_model = TFFactorSRUKFModel(
        initial_mean=tf.zeros([1, 2], tf.float64),
        initial_factor=tf.eye(2, batch_shape=[1], dtype=tf.float64),
        process_factor=tf.ones([1, 1, 1], tf.float64),
        observation_factor=tf.constant([[[0.15]]], tf.float64),
        transition_fn=transition,
        observation_fn=observe,
        name="model_a_direct_factor_parity",
    )
    factor_derivatives = TFFactorSRUKFDerivatives(
        d_initial_mean=tf.zeros([1, 1, 2], tf.float64),
        d_initial_factor=tf.zeros([1, 1, 2, 2], tf.float64),
        d_process_factor=tf.zeros([1, 1, 1, 1], tf.float64),
        d_observation_factor=tf.zeros([1, 1, 1, 1], tf.float64),
        transition_state_jacobian_fn=transition_jac,
        transition_process_jacobian_fn=innovation_jac,
        d_transition_fn=d_transition,
        observation_state_jacobian_fn=observation_jac,
        d_observation_fn=d_observation,
    )
    return factor_model, factor_derivatives, old_model, old_derivatives


def _model_b(theta: tf.Tensor) -> tuple[TFFactorSRUKFModel, TFFactorSRUKFDerivatives, object, TFStructuralFirstDerivatives]:
    rho, sigma, beta = tf.unstack(tf.convert_to_tensor(theta, tf.float64))
    old_model = make_nonlinear_accumulation_model_tf(rho=rho, sigma=sigma, beta=beta)
    old_derivatives = make_nonlinear_accumulation_first_derivatives_tf(rho=rho, sigma=sigma, beta=beta)
    alpha = tf.constant(0.55, tf.float64)

    def transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        eps = innovation[:, :, 0]
        m = rho * previous[:, :, 0] + sigma * eps
        k = alpha * previous[:, :, 1] + beta * tf.math.tanh(m)
        return tf.stack([m, k], axis=2)

    def observe(states: tf.Tensor) -> tf.Tensor:
        return (states[:, :, 0] + states[:, :, 1])[:, :, None]

    def state_jac(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        eps = innovation[:, :, 0]
        m = rho * previous[:, :, 0] + sigma * eps
        sech2 = 1.0 - tf.square(tf.math.tanh(m))
        return tf.stack(
            [tf.stack([tf.fill(tf.shape(m), rho), tf.zeros_like(m)], 2), tf.stack([beta * sech2 * rho, tf.fill(tf.shape(m), alpha)], 2)], 2
        )

    def process_jac(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        eps = innovation[:, :, 0]
        m = rho * previous[:, :, 0] + sigma * eps
        sech2 = 1.0 - tf.square(tf.math.tanh(m))
        return tf.stack([tf.fill(tf.shape(m), sigma), beta * sech2 * sigma], 2)[:, :, :, None]

    def d_transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        eps = innovation[:, :, 0]
        m = rho * previous[:, :, 0] + sigma * eps
        tanh_m = tf.math.tanh(m)
        sech2 = 1.0 - tf.square(tanh_m)
        return tf.stack(
            [
                tf.stack([previous[:, :, 0], beta * sech2 * previous[:, :, 0]], 2),
                tf.stack([eps, beta * sech2 * eps], 2),
                tf.stack([tf.zeros_like(eps), tanh_m], 2),
            ],
            axis=1,
        )

    def obs_jac(states: tf.Tensor) -> tf.Tensor:
        return tf.broadcast_to(tf.constant([[[1.0, 1.0]]], tf.float64)[None, ...], [1, tf.shape(states)[1], 1, 2])

    def d_obs(states: tf.Tensor) -> tf.Tensor:
        return tf.zeros([1, 3, tf.shape(states)[1], 1], tf.float64)

    return (
        TFFactorSRUKFModel(
            initial_mean=tf.zeros([1, 2], tf.float64),
            initial_factor=tf.linalg.diag(
                tf.stack([
                    tf.constant(0.5, tf.float64),
                    tf.sqrt(tf.constant(0.2, tf.float64)),
                ])
            )[None, ...],
            process_factor=tf.ones([1, 1, 1], tf.float64),
            observation_factor=tf.constant([[[0.3]]], tf.float64),
            transition_fn=transition,
            observation_fn=observe,
            name="model_b_direct_factor_parity",
        ),
        TFFactorSRUKFDerivatives(
            d_initial_mean=tf.zeros([1, 3, 2], tf.float64),
            d_initial_factor=tf.zeros([1, 3, 2, 2], tf.float64),
            d_process_factor=tf.zeros([1, 3, 1, 1], tf.float64),
            d_observation_factor=tf.zeros([1, 3, 1, 1], tf.float64),
            transition_state_jacobian_fn=state_jac,
            transition_process_jacobian_fn=process_jac,
            d_transition_fn=d_transition,
            observation_state_jacobian_fn=obs_jac,
            d_observation_fn=d_obs,
        ),
        old_model,
        old_derivatives,
    )


def _model_c(theta: tf.Tensor) -> tuple[TFFactorSRUKFModel, TFFactorSRUKFDerivatives, object, TFStructuralFirstDerivatives]:
    process_sigma, observation_sigma, initial_variance = tf.unstack(tf.convert_to_tensor(theta, tf.float64))
    old_model = make_univariate_nonlinear_growth_model_tf(
        process_sigma=process_sigma,
        observation_sigma=observation_sigma,
        initial_variance=initial_variance,
        initial_phase_variance=tf.constant(0.15, tf.float64),
    )
    old_derivatives = make_univariate_nonlinear_growth_first_derivatives_tf(
        process_sigma=process_sigma, observation_sigma=observation_sigma
    )

    def transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        x = previous[:, :, 0]
        tau = previous[:, :, 1]
        x_next = 0.5 * x + 25.0 * x / (1.0 + tf.square(x)) + 8.0 * tf.math.cos(1.2 * tau) + process_sigma * innovation[:, :, 0]
        return tf.stack([x_next, tau + 1.0], axis=2)

    def observe(states: tf.Tensor) -> tf.Tensor:
        return (tf.square(states[:, :, 0]) / 20.0)[:, :, None]

    def state_jac(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        x, tau = previous[:, :, 0], previous[:, :, 1]
        dx = 0.5 + 25.0 * (1.0 - tf.square(x)) / tf.square(1.0 + tf.square(x))
        return tf.stack([tf.stack([dx, -9.6 * tf.math.sin(1.2 * tau)], 2), tf.stack([tf.zeros_like(x), tf.ones_like(x)], 2)], 2)

    def process_jac(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        return tf.stack([tf.fill(tf.shape(previous[:, :, 0]), process_sigma), tf.zeros_like(previous[:, :, 0])], 2)[:, :, :, None]

    def d_transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        eps = innovation[:, :, 0]
        zeros = tf.zeros_like(eps)
        d_process = tf.stack([eps, zeros], axis=2)
        return tf.stack([d_process, tf.zeros_like(d_process), tf.zeros_like(d_process)], axis=1)

    def obs_jac(states: tf.Tensor) -> tf.Tensor:
        return tf.stack([states[:, :, 0] / 10.0, tf.zeros_like(states[:, :, 0])], 2)[:, :, None, :]

    def d_obs(states: tf.Tensor) -> tf.Tensor:
        return tf.zeros([1, 3, tf.shape(states)[1], 1], tf.float64)

    initial_factor = tf.linalg.diag(tf.stack([tf.sqrt(initial_variance), tf.sqrt(tf.constant(0.15, tf.float64))], axis=0))[None, ...]
    return (
        TFFactorSRUKFModel(
            initial_mean=tf.constant([[0.0, 1.0]], tf.float64),
            initial_factor=initial_factor,
            process_factor=tf.ones([1, 1, 1], tf.float64),
            observation_factor=tf.reshape(observation_sigma, [1, 1, 1]),
            transition_fn=transition,
            observation_fn=observe,
            name="model_c_direct_factor_parity",
        ),
        TFFactorSRUKFDerivatives(
            d_initial_mean=tf.zeros([1, 3, 2], tf.float64),
            d_initial_factor=tf.concat(
                [
                    tf.zeros([1, 1, 2, 2], tf.float64),
                    tf.zeros([1, 1, 2, 2], tf.float64),
                    tf.reshape(
                        tf.stack(
                            [
                                tf.stack(
                                    [
                                        0.5 / tf.sqrt(initial_variance),
                                        tf.constant(0.0, tf.float64),
                                    ]
                                ),
                                tf.constant([0.0, 0.0], tf.float64),
                            ]
                        ),
                        [1, 1, 2, 2],
                    ),
                ],
                axis=1,
            ),
            d_process_factor=tf.zeros([1, 3, 1, 1], tf.float64),
            d_observation_factor=tf.constant([[[[0.0]], [[1.0]], [[0.0]]]], tf.float64),
            transition_state_jacobian_fn=state_jac,
            transition_process_jacobian_fn=process_jac,
            d_transition_fn=d_transition,
            observation_state_jacobian_fn=obs_jac,
            d_observation_fn=d_obs,
        ),
        old_model,
        old_derivatives,
    )


def _run_case(name: str, builder, theta: tf.Tensor, observations: tf.Tensor) -> dict[str, object]:
    factor_model, factor_derivatives, old_model, old_derivatives = builder(theta)
    old = tf_principal_sqrt_ukf_score(observations, old_model, old_derivatives)
    new = tf_factor_srukf_value_and_score(observations[None, ...], factor_model, factor_derivatives, jit_compile=False)
    old_value = float(old.log_likelihood.numpy())
    old_score = old.score.numpy()
    new_value = float(new.log_likelihood.numpy()[0])
    new_score = new.score.numpy()[0]
    return {
        "model": name,
        "old_value": old_value,
        "new_value": new_value,
        "value_abs_diff": abs(new_value - old_value),
        "value_rel_diff": abs(new_value - old_value) / max(1.0, abs(old_value)),
        "old_score": old_score.tolist(),
        "new_score": new_score.tolist(),
        "score_max_abs_diff": float(np.max(np.abs(new_score - old_score))),
        "score_max_rel_diff": float(np.max(np.abs(new_score - old_score) / np.maximum(1.0, np.abs(old_score)))),
        "new_min_qr_pivot": float(tf.reduce_min(new.diagnostics["minimum_qr_pivot"]).numpy()),
        "new_min_downdate_margin": float(tf.reduce_min(new.diagnostics["minimum_downdate_margin"]).numpy()),
    }


def test_non_ssl_model_value_score_parity_campaign() -> None:
    cases = [
        _run_case("model_a_affine", _model_a, tf.constant(0.0, tf.float64), model_a_observations_tf()),
        _run_case("model_b_nonlinear_accumulation", _model_b, tf.constant([0.70, 0.25, 0.80], tf.float64), model_b_observations_tf()),
        _run_case("model_c_nonlinear_growth", _model_c, tf.constant([1.0, 1.0, 0.20], tf.float64), model_c_observations_tf()),
    ]
    for case in cases:
        assert np.isfinite(case["old_value"])
        assert np.isfinite(case["new_value"])
        assert np.isfinite(case["score_max_abs_diff"])
        assert case["new_min_qr_pivot"] > 0.0
        assert case["new_min_downdate_margin"] > 0.0
    # Model A is affine and should agree to numerical precision. Models B/C
    # are nonlinear orientation comparisons; the artifact records differences.
    assert cases[0]["value_abs_diff"] < 1.0e-8
    assert cases[0]["score_max_abs_diff"] < 1.0e-7


def test_non_ssl_direct_factor_scores_match_centered_fd_of_same_program() -> None:
    cases = [
        ("model_a_affine", _model_a, tf.constant([0.0], tf.float64), model_a_observations_tf()),
        (
            "model_b_nonlinear_accumulation",
            _model_b,
            tf.constant([0.70, 0.25, 0.80], tf.float64),
            model_b_observations_tf(),
        ),
        (
            "model_c_nonlinear_growth",
            _model_c,
            tf.constant([1.0, 1.0, 0.20], tf.float64),
            model_c_observations_tf(),
        ),
    ]
    for name, builder, theta, observations in cases:
        del name
        base_theta = theta[0] if theta.shape[0] == 1 else theta
        model, derivatives, _, _ = builder(base_theta)
        result = tf_factor_srukf_value_and_score(
            observations[None, ...], model, derivatives, jit_compile=False
        )
        fd_columns = []
        step = 1.0e-5
        for parameter_index in range(int(theta.shape[0])):
            direction = tf.one_hot(parameter_index, int(theta.shape[0]), dtype=tf.float64)
            plus_theta = theta + step * direction
            minus_theta = theta - step * direction
            plus_base = plus_theta[0] if theta.shape[0] == 1 else plus_theta
            minus_base = minus_theta[0] if theta.shape[0] == 1 else minus_theta
            plus_model, plus_derivatives, _, _ = builder(plus_base)
            minus_model, minus_derivatives, _, _ = builder(minus_base)
            plus_value = tf_factor_srukf_value_and_score(
                observations[None, ...], plus_model, plus_derivatives, jit_compile=False
            ).log_likelihood[0]
            minus_value = tf_factor_srukf_value_and_score(
                observations[None, ...], minus_model, minus_derivatives, jit_compile=False
            ).log_likelihood[0]
            fd_columns.append((plus_value - minus_value) / (2.0 * step))
        finite_difference = tf.stack(fd_columns)
        np.testing.assert_allclose(
            result.score[0].numpy(), finite_difference.numpy(), rtol=2.0e-6, atol=2.0e-8
        )
