from __future__ import annotations

import hashlib

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.models import parameterized_zhao_cui_sir_austria_model
from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
    SIR_HORIZON,
    SIR_OBSERVATION_SHA256,
    SIR_PRIOR_SCALE,
    SIR_STATE_SHA256,
    SIR_WRONG_TIME_ORDER_SHA256,
    generate_frozen_sir_dataset_tf,
    make_sir_sgqf_neutra_adapter,
    make_sir_ukf_neutra_adapter,
    sir_identity_chart_jacobian_value_score,
    sir_bootstrap_pf_log_likelihood_tf,
    sir_prior_value_score,
    sir_prior_predictive_tf,
    sir_rk4_transition_value,
    sir_rk4_transition_value_state_source_jacobians,
    sir_sgqf_likelihood_value_score_status,
    sir_sgqf_likelihood_value_only_status,
    sir_sgqf_posterior_value_only,
    sir_ukf_likelihood_value_score_status,
)


def _tensor_hash(value: tf.Tensor) -> str:
    return hashlib.sha256(bytes(tf.io.serialize_tensor(value).numpy())).hexdigest()


def test_frozen_sir_dataset_uses_y1_y20_and_clipping_is_inactive() -> None:
    states, observations, all_observations = generate_frozen_sir_dataset_tf()

    assert states.shape == (SIR_HORIZON + 1, 18)
    assert observations.shape == (SIR_HORIZON, 9)
    assert all_observations.shape == (SIR_HORIZON + 1, 9)
    assert _tensor_hash(states) == SIR_STATE_SHA256
    assert _tensor_hash(observations) == SIR_OBSERVATION_SHA256
    assert _tensor_hash(all_observations[:SIR_HORIZON]) == SIR_WRONG_TIME_ORDER_SHA256
    tf.debugging.assert_equal(observations, all_observations[1:])
    assert float(tf.reduce_min(states[:, 0::2]).numpy()) > 0.0


def test_sir_prior_and_identity_chart_match_declared_gaussian() -> None:
    theta = tf.constant([[0.0, 0.5, -0.5], [0.2, -0.3, 0.4]], tf.float64)
    value, score = sir_prior_value_score(theta)
    jacobian_value, jacobian_score = sir_identity_chart_jacobian_value_score(theta)
    expected = tf.reduce_sum(
        -0.5 * tf.square(theta / SIR_PRIOR_SCALE)
        - tf.math.log(SIR_PRIOR_SCALE)
        - 0.5 * tf.math.log(tf.constant(2.0 * np.pi, tf.float64)),
        axis=1,
    )

    tf.debugging.assert_near(value, expected, atol=1e-14)
    tf.debugging.assert_near(score, -theta / tf.square(SIR_PRIOR_SCALE), atol=1e-14)
    tf.debugging.assert_equal(jacobian_value, tf.zeros([2], tf.float64))
    tf.debugging.assert_equal(jacobian_score, tf.zeros_like(theta))


def test_sir_prior_predictive_is_batched_unprojected_and_replayable() -> None:
    seed = tf.constant([20260716, 16201], tf.int32)

    @tf.function(jit_compile=True)
    def compiled():
        return dict(sir_prior_predictive_tf(batch_size=8, horizon=3, seed=seed))

    first = compiled()
    second = compiled()
    assert first["theta"].shape == (8, 3)
    assert first["initial_state"].shape == (8, 18)
    assert first["states"].shape == (8, 3, 18)
    assert first["observations"].shape == (8, 3, 9)
    for key in first:
        tf.debugging.assert_equal(first[key], second[key])


def test_sir_target_rejects_wrong_theta_dtype() -> None:
    with np.testing.assert_raises_regex(ValueError, "float64 theta"):
        sir_prior_value_score(tf.zeros([1, 3], tf.float32))


def test_sir_bootstrap_pf_reference_is_compiled_batched_and_replayable() -> None:
    _states, observations, _all = generate_frozen_sir_dataset_tf()
    seed = tf.constant([20260716, 16202], tf.int32)

    @tf.function(jit_compile=True)
    def compiled():
        return sir_bootstrap_pf_log_likelihood_tf(
            tf.zeros([3], tf.float64),
            observations=observations[:2],
            particle_count=32,
            replicate_count=2,
            seed=seed,
        )

    first = compiled()
    second = compiled()
    assert first.shape == (2,)
    assert bool(tf.reduce_all(tf.math.is_finite(first)).numpy())
    tf.debugging.assert_equal(first, second)


def test_sir_rk4_value_matches_parameterized_author_variant() -> None:
    model = parameterized_zhao_cui_sir_austria_model()
    theta = tf.constant([[0.03, -0.02, 0.01], [-0.04, 0.05, -0.03]], tf.float64)
    offsets = tf.stack(
        (
            tf.linspace(tf.constant(-0.03, tf.float64), tf.constant(0.03, tf.float64), 18),
            tf.linspace(tf.constant(0.02, tf.float64), tf.constant(-0.02, tf.float64), 18),
            tf.zeros([18], tf.float64),
        ),
        axis=0,
    )
    previous = model.base_model.initial_mean[None, None, :] + offsets[None, :, :]
    previous = tf.broadcast_to(previous, [2, 3, 18])

    actual, _state_jacobian, _source_jacobian = (
        sir_rk4_transition_value_state_source_jacobians(theta, previous)
    )
    value_only = sir_rk4_transition_value(theta, previous)
    expected = tf.stack(
        [model.transition_mean(theta[index], previous[index]) for index in range(2)],
        axis=0,
    )
    tf.debugging.assert_near(actual, expected, atol=2e-10, rtol=2e-10)
    tf.debugging.assert_equal(value_only, actual)


def test_sir_rk4_source_and_state_jacobians_match_autodiff() -> None:
    theta = tf.constant([[0.03, -0.02, 0.01]], tf.float64)
    previous = tf.constant(
        parameterized_zhao_cui_sir_austria_model().base_model.initial_mean[None, None, :]
    )
    actual, state_jacobian, source_jacobian = (
        sir_rk4_transition_value_state_source_jacobians(theta, previous)
    )

    with tf.GradientTape(persistent=True) as tape:
        tape.watch((theta, previous))
        value = sir_rk4_transition_value_state_source_jacobians(theta, previous)[0]
    autodiff_source = tape.jacobian(value, theta)
    autodiff_state = tape.jacobian(value, previous)
    del tape

    tf.debugging.assert_near(actual, value, atol=0.0)
    tf.debugging.assert_near(
        source_jacobian[0, :, 0, :],
        tf.transpose(autodiff_source[0, 0, :, 0, :]),
        atol=2e-9,
        rtol=2e-9,
    )
    tf.debugging.assert_near(
        state_jacobian[0, 0],
        autodiff_state[0, 0, :, 0, 0, :],
        atol=2e-9,
        rtol=2e-9,
    )


def _central_value_gradient(value_fn, point: tf.Tensor, step: float = 1e-5) -> tf.Tensor:
    basis = tf.eye(3, dtype=tf.float64)
    stencil = tf.concat(
        (point[None, :] + step * basis, point[None, :] - step * basis), axis=0
    )
    values = value_fn(stencil)
    return (values[:3] - values[3:]) / (2.0 * step)


def test_sir_ukf_t3_score_matches_central_value_difference() -> None:
    _states, observations, _all = generate_frozen_sir_dataset_tf()
    point = tf.constant([0.05, -0.04, 0.03], tf.float64)

    value, score, status = sir_ukf_likelihood_value_score_status(
        point[None, :], observations=observations[:3]
    )
    finite_difference = _central_value_gradient(
        lambda theta: sir_ukf_likelihood_value_score_status(
            theta, observations=observations[:3]
        )[0],
        point,
    )

    assert bool(status["valid_pre_regularized_score"][0].numpy())
    assert bool(tf.math.is_finite(value[0]).numpy())
    tf.debugging.assert_near(score[0], finite_difference, atol=2e-4, rtol=2e-5)


def test_sir_ukf_full_horizon_score_matches_same_mode_value_difference() -> None:
    _states, observations, _all = generate_frozen_sir_dataset_tf()
    point = tf.constant([0.0, -1.0, 0.0], tf.float64)

    _value, score, status = sir_ukf_likelihood_value_score_status(
        point[None, :], observations=observations
    )
    finite_difference = _central_value_gradient(
        lambda theta: sir_ukf_likelihood_value_score_status(
            theta, observations=observations
        )[0],
        point,
        step=5.0e-5,
    )

    assert bool(status["valid_pre_regularized_score"][0].numpy())
    tf.debugging.assert_near(score[0], finite_difference, atol=5e-3, rtol=5e-4)


def test_sir_sgqf_t3_score_matches_central_value_difference() -> None:
    _states, observations, _all = generate_frozen_sir_dataset_tf()
    adapter = make_sir_sgqf_neutra_adapter(observations=observations)
    point = tf.constant([0.05, -0.04, 0.03], tf.float64)

    value, score, status = sir_sgqf_likelihood_value_score_status(
        point[None, :],
        observations=observations[:3],
        nodes=adapter.nodes,
        weights=adapter.weights,
    )
    finite_difference = _central_value_gradient(
        lambda theta: sir_sgqf_likelihood_value_score_status(
            theta,
            observations=observations[:3],
            nodes=adapter.nodes,
            weights=adapter.weights,
        )[0],
        point,
    )

    assert bool(status["valid_pre_regularized_score"][0].numpy())
    assert bool(tf.math.is_finite(value[0]).numpy())
    tf.debugging.assert_near(score[0], finite_difference, atol=2e-4, rtol=2e-5)


def test_sir_sgqf_value_only_matches_complete_posterior_under_cpu_xla() -> None:
    _states, observations, _all = generate_frozen_sir_dataset_tf()
    adapter = make_sir_sgqf_neutra_adapter(observations=observations)
    theta = tf.constant([[0.0, 0.0, 0.0], [0.1, -0.2, 0.05]], tf.float64)

    @tf.function(jit_compile=True)
    def compiled(values):
        likelihood, status = sir_sgqf_likelihood_value_only_status(
            values, observations=observations, nodes=adapter.nodes,
            weights=adapter.weights
        )
        posterior = sir_sgqf_posterior_value_only(
            values, observations=observations, nodes=adapter.nodes,
            weights=adapter.weights
        )
        return likelihood, status, posterior

    likelihood, status, posterior = compiled(theta)
    reference, _score, reference_status = (
        adapter.neutra_batch_log_prob_and_grad_status(theta)
    )
    tf.debugging.assert_near(posterior, reference, atol=1e-8, rtol=1e-11)
    tf.debugging.assert_equal(status["status_code"], reference_status["status_code"])
    assert bool(tf.reduce_all(tf.math.is_finite(likelihood)).numpy())


def test_sir_posterior_adapters_add_exactly_prior_and_zero_chart_term() -> None:
    _states, observations, _all = generate_frozen_sir_dataset_tf()
    theta = tf.constant([[0.0, 0.0, 0.0], [0.1, -0.1, 0.05]], tf.float64)
    for adapter in (
        make_sir_ukf_neutra_adapter(observations=observations),
        make_sir_sgqf_neutra_adapter(observations=observations),
    ):
        posterior_value, posterior_score = adapter.log_prob_and_grad(theta)
        if adapter.target_scope.startswith("SIR-UKF"):
            likelihood_value, likelihood_score, _status = (
                sir_ukf_likelihood_value_score_status(
                    theta, observations=observations
                )
            )
        else:
            likelihood_value, likelihood_score, _status = (
                sir_sgqf_likelihood_value_score_status(
                    theta,
                    observations=observations,
                    nodes=adapter.nodes,
                    weights=adapter.weights,
                )
            )
        prior_value, prior_score = sir_prior_value_score(theta)
        tf.debugging.assert_near(posterior_value, likelihood_value + prior_value)
        tf.debugging.assert_near(posterior_score, likelihood_score + prior_score)
