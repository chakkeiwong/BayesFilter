from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters as ssl_adapters
import bayesfilter.nonlinear.ssl_lstm_zhaocui_fixed_adapter as zhaocui_adapter
from bayesfilter.nonlinear.fixed_sgqf_derivatives_tf import tf_fixed_sgqf_score
from bayesfilter.nonlinear.ssl_lstm_protocol import SSLLSTMStaticConfig
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import (
    make_ssl_lstm_fixed_sgqf_components,
    make_ssl_lstm_svd_ukf_components,
    ssl_lstm_observation,
    ssl_lstm_observation_state_jacobian,
    ssl_lstm_observation_state_jvp,
    ssl_lstm_parameter_slices,
    ssl_lstm_transition,
    ssl_lstm_transition_state_jacobian,
    ssl_lstm_transition_state_jvp,
    unpack_ssl_lstm_parameters,
)
from bayesfilter.nonlinear.ssl_lstm_zhaocui_fixed_adapter import (
    SSLLSTMZhaoCuiFixedManifest,
    tf_ssl_lstm_zhaocui_fixed_score,
)
from bayesfilter.nonlinear.svd_sigma_point_derivatives_tf import (
    tf_principal_sqrt_ukf_score,
)


PLAN = "docs/plans/bayesfilter-ssl-lstm-matrix-free-filter-derivatives-plan-2026-07-20.md"


def _config(q: int, *, horizon: int = 2) -> SSLLSTMStaticConfig:
    return SSLLSTMStaticConfig(
        horizon=horizon,
        latent_dim=q,
        hidden_dim=q,
        observation_dim=1,
    )


def _theta(config: SSLLSTMStaticConfig) -> tf.Tensor:
    indices = tf.cast(tf.range(config.parameter_dim), tf.float64)
    return 0.08 * tf.sin(0.37 * indices) - 0.03 * tf.cos(0.19 * indices)


def _selected_indices(config: SSLLSTMStaticConfig) -> tuple[int, ...]:
    slices = ssl_lstm_parameter_slices(config)
    return (
        slices.latent_weight_start,
        slices.latent_bias_start,
        slices.observation_weight_start,
        slices.observation_bias_start,
    )


def _points(config: SSLLSTMStaticConfig, point_count: int) -> tf.Tensor:
    size = point_count * config.augmented_state_dim
    return tf.reshape(
        0.2 * tf.sin(0.13 * tf.cast(tf.range(size), tf.float64)),
        [point_count, config.augmented_state_dim],
    )


def _tangents(
    config: SSLLSTMStaticConfig,
    point_count: int,
    direction_count: int = 4,
) -> tf.Tensor:
    size = direction_count * point_count * config.augmented_state_dim
    return tf.reshape(
        0.1 * tf.cos(0.17 * tf.cast(tf.range(size), tf.float64)),
        [direction_count, point_count, config.augmented_state_dim],
    )


@pytest.mark.parametrize("q", (1, 2, 5))
def test_ssl_lstm_local_jvps_match_dense_jacobian_products(q: int) -> None:
    config = _config(q)
    params = unpack_ssl_lstm_parameters(
        _theta(config),
        config,
        derivative_parameter_indices=_selected_indices(config),
    )
    points = _points(config, point_count=7)
    tangents = _tangents(config, point_count=7)

    dense_transition = tf.einsum(
        "roi,pri->pro",
        ssl_lstm_transition_state_jacobian(params, points),
        tangents,
    )
    direct_transition = ssl_lstm_transition_state_jvp(params, points, tangents)
    dense_observation = tf.einsum(
        "rdi,pri->prd",
        ssl_lstm_observation_state_jacobian(params, points),
        tangents,
    )
    direct_observation = ssl_lstm_observation_state_jvp(params, points, tangents)

    tf.debugging.assert_near(direct_transition, dense_transition, atol=2e-13, rtol=2e-13)
    tf.debugging.assert_near(direct_observation, dense_observation, atol=2e-13, rtol=2e-13)


def test_ssl_lstm_transition_jvp_matches_directional_finite_difference() -> None:
    config = _config(2)
    params = unpack_ssl_lstm_parameters(
        _theta(config),
        config,
        derivative_parameter_indices=_selected_indices(config),
    )
    points = _points(config, point_count=5)
    tangents = _tangents(config, point_count=5)
    actual = ssl_lstm_transition_state_jvp(params, points, tangents)
    step = tf.constant(1.0e-6, tf.float64)
    finite = tf.stack(
        [
            (
                ssl_lstm_transition(params, points + step * tangent)
                - ssl_lstm_transition(params, points - step * tangent)
            )
            / (2.0 * step)
            for tangent in tf.unstack(tangents, axis=0)
        ],
        axis=0,
    )
    tf.debugging.assert_near(actual, finite, atol=2e-10, rtol=2e-8)


def test_selected_ukf_score_matches_dense_fallback() -> None:
    config = _config(2)
    theta = _theta(config)
    observations = tf.constant([[0.12], [-0.03]], tf.float64)
    components = make_ssl_lstm_svd_ukf_components(
        theta,
        config,
        evidence_path=PLAN,
        derivative_parameter_indices=_selected_indices(config),
    )
    dense_derivatives = replace(
        components.derivatives,
        transition_jvp_fn=None,
        observation_jvp_fn=None,
    )

    candidate = tf_principal_sqrt_ukf_score(
        observations,
        components.model,
        components.derivatives,
        innovation_floor=tf.constant(1.0e-12, tf.float64),
    )
    baseline = tf_principal_sqrt_ukf_score(
        observations,
        components.model,
        dense_derivatives,
        innovation_floor=tf.constant(1.0e-12, tf.float64),
    )

    tf.debugging.assert_near(candidate.log_likelihood, baseline.log_likelihood, atol=1e-13, rtol=0.0)
    tf.debugging.assert_near(candidate.score, baseline.score, atol=2e-12, rtol=2e-12)


def test_fixed_sgqf_score_matches_dense_fallback() -> None:
    config = _config(1)
    theta = _theta(config)
    observations = tf.constant([[0.12], [-0.03]], tf.float64)
    components = make_ssl_lstm_fixed_sgqf_components(
        theta,
        config,
        evidence_path=PLAN,
        sparse_level=2,
    )
    dense_derivatives = replace(
        components.derivatives,
        transition_jvp_fn=None,
        observation_jvp_fn=None,
    )
    candidate = tf_fixed_sgqf_score(
        observations,
        components.model,
        components.derivatives,
        cloud=components.cloud,
        branch_config=components.branch_config,
    )
    baseline = tf_fixed_sgqf_score(
        observations,
        components.model,
        dense_derivatives,
        cloud=components.cloud,
        branch_config=components.branch_config,
    )

    assert candidate.failure is None
    assert baseline.failure is None
    tf.debugging.assert_near(candidate.log_likelihood, baseline.log_likelihood, atol=1e-13, rtol=0.0)
    tf.debugging.assert_near(candidate.score, baseline.score, atol=2e-12, rtol=2e-12)


def test_fixed_replay_score_matches_dense_local_products(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(1)
    theta = _theta(config)
    observations = tf.constant([[0.12], [-0.03]], tf.float64)
    manifest = SSLLSTMZhaoCuiFixedManifest(
        reference_sample_count=9,
        initial_seed=(20260720, 11),
        process_seed=(20260720, 13),
    )
    candidate, _ = tf_ssl_lstm_zhaocui_fixed_score(
        observations,
        theta,
        config,
        evidence_path=PLAN,
        manifest=manifest,
    )

    def dense_transition_jvp(params, points, tangents):
        return tf.einsum(
            "roi,pri->pro",
            ssl_lstm_transition_state_jacobian(params, points),
            tangents,
        )

    def dense_observation_jvp(params, points, tangents):
        return tf.einsum(
            "rdi,pri->prd",
            ssl_lstm_observation_state_jacobian(params, points),
            tangents,
        )

    monkeypatch.setattr(zhaocui_adapter, "ssl_lstm_transition_state_jvp", dense_transition_jvp)
    monkeypatch.setattr(zhaocui_adapter, "ssl_lstm_observation_state_jvp", dense_observation_jvp)
    baseline, _ = tf_ssl_lstm_zhaocui_fixed_score(
        observations,
        theta,
        config,
        evidence_path=PLAN,
        manifest=manifest,
    )

    tf.debugging.assert_near(candidate.log_likelihood, baseline.log_likelihood, atol=1e-13, rtol=0.0)
    tf.debugging.assert_near(candidate.score, baseline.score, atol=2e-12, rtol=2e-12)


def test_selected_ukf_and_fixed_replay_do_not_call_dense_ssl_lstm_jacobians(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(1)
    theta = _theta(config)
    observations = tf.constant([[0.12], [-0.03]], tf.float64)
    components = make_ssl_lstm_svd_ukf_components(
        theta,
        config,
        evidence_path=PLAN,
        derivative_parameter_indices=_selected_indices(config),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense SSL-LSTM Jacobian builder was called")

    monkeypatch.setattr(ssl_adapters, "ssl_lstm_transition_state_jacobian", forbidden)
    monkeypatch.setattr(ssl_adapters, "ssl_lstm_observation_state_jacobian", forbidden)
    ukf = tf_principal_sqrt_ukf_score(
        observations,
        components.model,
        components.derivatives,
        innovation_floor=tf.constant(1.0e-12, tf.float64),
    )
    replay, _ = tf_ssl_lstm_zhaocui_fixed_score(
        observations,
        theta,
        config,
        evidence_path=PLAN,
        manifest=SSLLSTMZhaoCuiFixedManifest(reference_sample_count=5),
    )

    assert bool(tf.reduce_all(tf.math.is_finite(ukf.score)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(replay.score)).numpy())


def test_q20_matrix_free_local_shape_without_dense_jacobian() -> None:
    config = _config(20)
    params = unpack_ssl_lstm_parameters(
        _theta(config),
        config,
        derivative_parameter_indices=_selected_indices(config),
    )
    point_count = 8 * 20 + 1
    points = _points(config, point_count)
    tangents = _tangents(config, point_count)
    transition = ssl_lstm_transition_state_jvp(params, points, tangents)
    observation = ssl_lstm_observation_state_jvp(params, points, tangents)

    assert transition.shape == (4, 161, 60)
    assert observation.shape == (4, 161, 1)
    assert bool(tf.reduce_all(tf.math.is_finite(transition)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(observation)).numpy())


def test_ssl_lstm_local_jvp_cpu_xla_parity() -> None:
    config = _config(2)
    theta = _theta(config)
    points = _points(config, point_count=5)
    tangents = _tangents(config, point_count=5)
    params = unpack_ssl_lstm_parameters(
        theta,
        config,
        derivative_parameter_indices=_selected_indices(config),
    )
    expected = (
        ssl_lstm_transition_state_jvp(params, points, tangents),
        ssl_lstm_observation_state_jvp(params, points, tangents),
    )

    @tf.function(jit_compile=True)
    def compiled(theta_value: tf.Tensor, point_values: tf.Tensor, tangent_values: tf.Tensor):
        local = unpack_ssl_lstm_parameters(
            theta_value,
            config,
            derivative_parameter_indices=_selected_indices(config),
        )
        return (
            ssl_lstm_transition_state_jvp(local, point_values, tangent_values),
            ssl_lstm_observation_state_jvp(local, point_values, tangent_values),
        )

    try:
        actual = compiled(theta, points, tangents)
    except tf.errors.InvalidArgumentError as exc:
        pytest.skip(f"local CPU XLA unavailable for SSL-LSTM JVP smoke: {exc}")
    tf.debugging.assert_near(actual[0], expected[0], atol=2e-13, rtol=2e-13)
    tf.debugging.assert_near(actual[1], expected[1], atol=2e-13, rtol=2e-13)
