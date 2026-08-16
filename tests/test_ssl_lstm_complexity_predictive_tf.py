from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.nonlinear.ssl_lstm_complexity_predictive_tf import (
    ComplexityPredictiveError,
    _value_model,
    calibration_from_observation_banks,
    calibration_seed_roots,
    forecast_complexity_conditional_moments,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (
    complexity_posterior_target,
)
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import (
    unpack_ssl_lstm_parameters,
)


def test_value_only_model_uses_zero_derivative_directions() -> None:
    target = complexity_posterior_target(2, jit_compile=False)
    full = target.full_theta(target.config.prior_center)
    params, model = _value_model(full, target)
    assert params.d_initial_mean.shape == (0, 6)
    assert params.d_initial_covariance.shape == (0, 6, 6)
    assert model.partition.innovation_dim == 2


def test_q1_compiled_forecast_is_finite_replayable_and_rao_variance_is_exact() -> None:
    draws = tf.constant(
        [[0.35, -0.08, 0.65, 0.05], [0.36, -0.07, 0.64, 0.04]],
        tf.float64,
    )
    first = forecast_complexity_conditional_moments(
        draws, q=1, seed=(20260719, 49911)
    )
    second = forecast_complexity_conditional_moments(
        draws, q=1, seed=(20260719, 49911)
    )
    assert first.conditional_means.shape == (2, 2, 10)
    assert first.terminal_states.shape == (2, 2, 3)
    tf.debugging.assert_equal(first.conditional_means, second.conditional_means)
    tf.debugging.assert_equal(first.observations, second.observations)
    target = complexity_posterior_target(1, jit_compile=False)
    variances = []
    for draw in draws:
        params = unpack_ssl_lstm_parameters(
            target.full_theta(draw),
            target.config.static_config,
            derivative_parameter_indices=(),
        )
        variances.append(float(tf.square(params.observation_std[0]).numpy()))
    expected = tf.broadcast_to(
        tf.constant(variances, tf.float64)[:, tf.newaxis, tf.newaxis],
        (2, 2, 10),
    )
    tf.debugging.assert_near(first.conditional_variances, expected)
    assert first.target_signature == target.target_signature()


def test_q2_compiled_forecast_uses_stable_cholesky_terminal_factor() -> None:
    draw = tf.constant([[0.35, -0.08, 0.65, 0.05]], tf.float64)
    forecast = forecast_complexity_conditional_moments(
        draw, q=2, seed=(20260719, 49002)
    )
    assert forecast.conditional_means.shape == (1, 2, 10)
    assert forecast.terminal_states.shape == (1, 2, 6)
    assert bool(tf.reduce_all(forecast.status).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(forecast.observations)).numpy())


def test_compiled_forecast_supports_arbitrary_positive_horizon_and_replay() -> None:
    draw = tf.constant([[0.35, -0.08, 0.65, 0.05]], tf.float64)
    first = forecast_complexity_conditional_moments(
        draw, q=1, seed=(20260809, 49003), replication_count=3, horizon=20
    )
    second = forecast_complexity_conditional_moments(
        draw, q=1, seed=(20260809, 49003), replication_count=3, horizon=20
    )
    assert first.observations.shape == (1, 3, 20)
    assert first.horizon == 20
    tf.debugging.assert_equal(first.observations, second.observations)


@pytest.mark.parametrize("horizon", (0, -1, True, 1.5))
def test_compiled_forecast_rejects_invalid_horizon(horizon: object) -> None:
    with pytest.raises(ComplexityPredictiveError, match="horizon"):
        forecast_complexity_conditional_moments(
            tf.constant([[0.35, -0.08, 0.65, 0.05]], tf.float64),
            q=1,
            seed=(20260809, 49004),
            horizon=horizon,  # type: ignore[arg-type]
        )


def test_calibration_math_is_unbiased_positive_and_seed_bound() -> None:
    first = tf.reshape(tf.cast(tf.range(40), tf.float64), (2, 2, 10))
    second = first + 1.0
    roots = ((20260719, 4101), (20260719, 4102))
    calibration = calibration_from_observation_banks(
        (first, second),
        q=2,
        seed_roots=roots,
        target_signature="target",
        forecast_signatures=("a", "b"),
    )
    pooled = tf.concat((first, second), axis=0)
    expected_center = tf.reduce_mean(pooled, axis=(0, 1))
    centered = pooled - expected_center
    expected_scale = tf.sqrt(
        tf.reduce_sum(tf.square(centered), axis=(0, 1)) / 7.0
    )
    tf.debugging.assert_near(calibration.center, expected_center)
    tf.debugging.assert_near(calibration.scale, expected_scale)
    assert calibration.seed_roots == roots
    assert calibration.calibration_signature


def test_calibration_rejects_nonpositive_or_misaligned_banks() -> None:
    constant = tf.ones((2, 2, 10), tf.float64)
    with pytest.raises(ComplexityPredictiveError, match="invalid"):
        calibration_from_observation_banks(
            (constant, constant),
            q=1,
            seed_roots=((1, 2), (3, 4)),
            target_signature="target",
            forecast_signatures=("a", "b"),
        )
    with pytest.raises(ComplexityPredictiveError, match="one forecast signature"):
        calibration_from_observation_banks(
            (constant, constant + tf.range(10, dtype=tf.float64)),
            q=1,
            seed_roots=((1, 2), (3, 4)),
            target_signature="target",
            forecast_signatures=("a",),
        )


def test_calibration_seed_domains_are_q_specific() -> None:
    q1 = calibration_seed_roots(1)
    q20 = calibration_seed_roots(20)
    assert len(q1) == 4
    assert set(q1).isdisjoint(q20)
