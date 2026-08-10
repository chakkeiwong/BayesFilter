from __future__ import annotations

import numpy as np
import tensorflow as tf

from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (
    FREE_NAMES,
    PRIOR_CENTER,
    STATIC_DATA_CONSTRUCTION_POLICY,
    complexity_posterior_target,
    make_complexity_config,
    make_full_fixture,
    make_synthetic_observations,
)
from bayesfilter.nonlinear.ssl_lstm_posterior_tf import locked_ssl_lstm_posterior_target
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import (
    make_ssl_lstm_svd_ukf_components,
)
from bayesfilter.nonlinear.svd_sigma_point_derivatives_tf import (
    tf_principal_sqrt_ukf_score,
)


PLAN = "docs/plans/bayesfilter-ssl-lstm-neutra-hmc-state-complexity-ladder-plan-2026-07-19.md"


def test_q1_principal_target_is_close_to_historical_locked_target() -> None:
    locked = locked_ssl_lstm_posterior_target()
    candidate = complexity_posterior_target(1)
    for offset in (0.0, 0.1, -0.15):
        point = PRIOR_CENTER + tf.constant((offset, -offset, offset / 2.0, -offset / 2.0), tf.float64)
        expected_value, expected_score = locked.value_and_score(point)
        actual_value, actual_score = candidate.value_and_score(point)
        tf.debugging.assert_near(actual_value, expected_value, atol=1.5e-5, rtol=0.0)
        tf.debugging.assert_near(actual_score, expected_score, atol=1.0e-4, rtol=0.0)


def test_selected_score_equals_full_score_at_q2() -> None:
    target = complexity_posterior_target(2, jit_compile=False)
    full = target.full_theta(PRIOR_CENTER)
    selected_components = make_ssl_lstm_svd_ukf_components(
        full,
        target.config.static_config,
        evidence_path=PLAN,
        derivative_parameter_indices=target.config.free_indices,
    )
    complete_components = make_ssl_lstm_svd_ukf_components(
        full,
        target.config.static_config,
        evidence_path=PLAN,
    )
    selected = tf_principal_sqrt_ukf_score(
        target.config.observations,
        selected_components.model,
        selected_components.derivatives,
        innovation_floor=tf.constant(1e-12, tf.float64),
    )
    complete = tf_principal_sqrt_ukf_score(
        target.config.observations,
        complete_components.model,
        complete_components.derivatives,
        innovation_floor=tf.constant(1e-12, tf.float64),
    )
    tf.debugging.assert_near(
        selected.score,
        tf.gather(complete.score, target.config.free_indices),
        atol=1e-10,
        rtol=1e-10,
    )


def test_all_rungs_have_four_direction_derivative_surfaces() -> None:
    for q in (1, 2, 5, 10, 20):
        target = complexity_posterior_target(q, jit_compile=False)
        components = make_ssl_lstm_svd_ukf_components(
            target.full_theta(PRIOR_CENTER),
            target.config.static_config,
            evidence_path=PLAN,
            derivative_parameter_indices=target.config.free_indices,
        )
        derivatives = components.derivatives
        assert derivatives.parameter_dim == 4
        assert derivatives.d_initial_covariance.shape[0] == 4
        assert derivatives.d_innovation_covariance.shape[0] == 4
        assert derivatives.d_observation_covariance.shape[0] == 4


def test_fixture_and_synthetic_data_are_deterministic() -> None:
    for q in (2, 5, 10, 20):
        config = make_complexity_config(q)
        first_fixture = make_full_fixture(config)
        second_fixture = make_full_fixture(config)
        tf.debugging.assert_equal(first_fixture, second_fixture)
        first = make_synthetic_observations(config, first_fixture)
        second = make_synthetic_observations(config, second_fixture)
        tf.debugging.assert_equal(first, second)
        assert first.shape == (30, 1)
        assert bool(tf.reduce_all(tf.math.is_finite(first)).numpy())


def test_target_chart_contract_is_four_dimensional_and_identity_oriented() -> None:
    target = complexity_posterior_target(5, jit_compile=False)
    assert target.parameter_dim == 4
    assert target.parameter_names == FREE_NAMES
    assert target.config.signature_payload()["parameter_transform"] == {
        "orientation": "identity",
        "inverse_orientation": "identity",
    }
    assert target.config.signature_payload()["filter_backend"] == "tf_principal_sqrt_ukf"
    assert target.config.signature_payload()["score_backend"] == "tf_principal_sqrt_ukf_score"
    assert target.config.signature_payload()["static_data_construction_policy"] == (
        STATIC_DATA_CONSTRUCTION_POLICY
    )
    assert "CPU:0" in target.config.fixture.device
    assert "CPU:0" in target.config.observations.device
    assert len(target.target_signature()) == 64
    assert len(target.adapter_signature()) == 64


def test_target_status_instrumentation_preserves_value_and_score() -> None:
    target = complexity_posterior_target(2, jit_compile=False)
    point = PRIOR_CENTER + tf.constant((0.03, -0.02, 0.01, -0.04), tf.float64)
    expected_value, expected_score = target.value_and_score(point)
    value, score, status = target.log_prob_and_grad_status(point)
    tf.debugging.assert_equal(value, expected_value)
    tf.debugging.assert_equal(score, expected_score)
    assert set(status) == {
        "status_code",
        "valid_pre_regularized_score",
        "floor_count_value",
        "min_innovation_eigenvalue",
        "innovation_condition_estimate",
    }
    assert int(status["status_code"].numpy()) == 0
    assert bool(status["valid_pre_regularized_score"].numpy())
    assert int(status["floor_count_value"].numpy()) == 0
    assert float(status["min_innovation_eigenvalue"].numpy()) > 0.0
    tf.debugging.assert_all_finite(
        status["innovation_condition_estimate"], "condition estimate must be finite"
    )


def test_batched_target_status_preserves_batch_value_and_score() -> None:
    target = complexity_posterior_target(2, jit_compile=False)
    points = tf.stack((PRIOR_CENTER, PRIOR_CENTER + 0.01), axis=0)
    expected_value, expected_score = target.batch_value_and_score(points)
    value, score, status = target.log_prob_and_grad_status(points)
    tf.debugging.assert_equal(value, expected_value)
    tf.debugging.assert_equal(score, expected_score)
    assert status["status_code"].shape == (2,)
    tf.debugging.assert_equal(status["status_code"], tf.zeros((2,), tf.int32))
    tf.debugging.assert_equal(
        status["valid_pre_regularized_score"], tf.ones((2,), tf.bool)
    )


def test_q2_directional_score_matches_finite_difference() -> None:
    target = complexity_posterior_target(2, jit_compile=False)
    point = PRIOR_CENTER + tf.constant((0.07, -0.05, 0.04, -0.03), tf.float64)
    _value, score = target.value_and_score(point)
    step = tf.constant(1e-5, tf.float64)
    finite = []
    for index in range(4):
        direction = tf.one_hot(index, 4, dtype=tf.float64)
        plus = target.eager_value(point + step * direction)
        minus = target.eager_value(point - step * direction)
        finite.append((plus - minus) / (2.0 * step))
    tf.debugging.assert_near(score, tf.stack(finite), atol=2e-7, rtol=2e-6)


def test_q5_principal_score_handles_observed_weak_eigen_gap_point() -> None:
    target = complexity_posterior_target(5, jit_compile=False)
    point = tf.constant(
        (
            1.7911783477746424,
            2.1641609074420747,
            -0.4040704194454182,
            -0.1681555034132438,
        ),
        tf.float64,
    )
    value, score = target.eager_value_and_score(point)
    tf.debugging.assert_all_finite(value, "q=5 weak-gap value must be finite")
    tf.debugging.assert_all_finite(score, "q=5 weak-gap score must be finite")
    step = tf.constant(1.0e-5, tf.float64)
    finite = []
    for index in range(4):
        direction = tf.one_hot(index, 4, dtype=tf.float64)
        plus = target.eager_value(point + step * direction)
        minus = target.eager_value(point - step * direction)
        finite.append((plus - minus) / (2.0 * step))
    np.testing.assert_allclose(
        score.numpy(), tf.stack(finite).numpy(), rtol=3.0e-6, atol=3.0e-7
    )
