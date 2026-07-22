from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim.generalized_sv_sgqf_tf import (
    GENERALIZED_SV_SGQF_OBSERVATION_SHA256,
    GENERALIZED_SV_SGQF_STATE_SHA256,
    generalized_sv_dense_value_reference_status,
    generalized_sv_sgqf_value_only_status,
    generalized_sv_sgqf_value_score_status,
    make_generalized_sv_sgqf_route,
)
from bayesfilter.nonlinear.fixed_sgqf_tf import tf_fixed_sgqf_cloud


def _central_fd(theta: tf.Tensor, observations: tf.Tensor, index: int) -> float:
    h = 1.0e-5
    direction = tf.one_hot(index, 3, dtype=tf.float64)
    plus, _ = generalized_sv_sgqf_value_only_status(
        theta + h * direction, observations
    )
    minus, _ = generalized_sv_sgqf_value_only_status(
        theta - h * direction, observations
    )
    return float((plus - minus).numpy() / (2.0 * h))


def test_generalized_sv_t1_is_exactly_one_source_transition() -> None:
    route = make_generalized_sv_sgqf_route()
    value, _score, status = generalized_sv_sgqf_value_score_status(
        route.theta, route.observations[:1]
    )

    gamma = tfp.distributions.Normal(
        loc=tf.constant(0.0, tf.float64), scale=tf.constant(1.0, tf.float64)
    ).cdf(route.theta[0])
    tau = tf.exp(route.theta[1])
    mu = route.theta[2] * tau
    stationary_variance = 1.0 / (1.0 - tf.square(gamma))
    cloud = tf_fixed_sgqf_cloud(dim=1, sparse_level=3)
    points = mu + tf.sqrt(stationary_variance) * tf.reshape(cloud.points, [-1])
    log_variance = tau * points
    y = route.observations[0, 0]
    observation_log = -0.5 * (
        tf.math.log(tf.constant(2.0 * np.pi, tf.float64))
        + log_variance
        + tf.square(y) * tf.exp(-log_variance)
    )
    expected = tf.reduce_logsumexp(tf.math.log(cloud.weights) + observation_log)

    assert int(status["transition_count"].numpy()) == 1
    tf.debugging.assert_near(value, expected, atol=1.0e-12, rtol=1.0e-12)


def test_generalized_sv_t2_same_scalar_manual_score_matches_fd() -> None:
    route = make_generalized_sv_sgqf_route()
    observations = route.observations[:2]
    value, score, status = generalized_sv_sgqf_value_score_status(
        route.theta, observations
    )
    value_only, value_status = generalized_sv_sgqf_value_only_status(
        route.theta, observations
    )
    finite_difference = np.array(
        [_central_fd(route.theta, observations, index) for index in range(3)]
    )

    assert int(status["transition_count"].numpy()) == 2
    assert int(status["status_code"].numpy()) == 0
    assert int(value_status["status_code"].numpy()) == 0
    tf.debugging.assert_near(value, value_only, atol=1.0e-12, rtol=1.0e-12)
    np.testing.assert_allclose(score.numpy(), finite_difference, atol=2.0e-7, rtol=2.0e-6)


def test_generalized_sv_asymmetric_prefix_score_matches_fd() -> None:
    route = make_generalized_sv_sgqf_route()
    theta = route.theta + tf.constant([0.13, -0.08, 0.17], tf.float64)
    observations = route.observations[:17]
    _value, score, status = generalized_sv_sgqf_value_score_status(
        theta, observations
    )
    finite_difference = np.array(
        [_central_fd(theta, observations, index) for index in range(3)]
    )

    assert int(status["status_code"].numpy()) == 0
    np.testing.assert_allclose(score.numpy(), finite_difference, atol=2.0e-6, rtol=2.0e-5)


def test_generalized_sv_manual_route_has_no_runtime_gradient_tape(monkeypatch) -> None:
    route = make_generalized_sv_sgqf_route()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("GradientTape is forbidden in the admitted SGQF score route")

    monkeypatch.setattr(tf, "GradientTape", forbidden)
    value, score, status = generalized_sv_sgqf_value_score_status(
        route.theta, route.observations[:3]
    )
    assert bool(tf.math.is_finite(value).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    assert int(status["status_code"].numpy()) == 0


def test_generalized_sv_full_route_identity_and_value_are_finite() -> None:
    route = make_generalized_sv_sgqf_route()
    value, score, status = route.value_score_status()

    assert route.states.shape == (1008, 1)
    assert route.observations.shape == (1008, 1)
    assert route.manifest["state_sha256"] == GENERALIZED_SV_SGQF_STATE_SHA256
    assert route.manifest["observation_sha256"] == (
        GENERALIZED_SV_SGQF_OBSERVATION_SHA256
    )
    assert route.manifest["model_family"] == "GeneralizedSVPriorMeanSSM"
    assert route.manifest["approximation"] == (
        "sequential_gaussian_projection_after_direct_likelihood_quadrature"
    )
    assert route.manifest["cloud_level"] == 3
    assert any(
        "not NativeGeneralizedSVSSM" in item
        for item in route.manifest["nonclaims"]
    )
    assert int(status["transition_count"].numpy()) == 1008
    assert int(status["status_code"].numpy()) == 0
    assert bool(tf.math.is_finite(value).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())


def test_generalized_sv_level3_tracks_level5_and_dense_reference() -> None:
    route = make_generalized_sv_sgqf_route()
    level3, _score3, status3 = generalized_sv_sgqf_value_score_status(
        route.theta, route.observations, sparse_level=3
    )
    level5, _score5, status5 = generalized_sv_sgqf_value_score_status(
        route.theta, route.observations, sparse_level=5
    )
    dense, dense_status = generalized_sv_dense_value_reference_status(
        route.theta, route.observations, order=41
    )

    assert int(status3["status_code"].numpy()) == 0
    assert int(status5["status_code"].numpy()) == 0
    assert int(dense_status["status_code"].numpy()) == 0
    assert abs(float(level3 - level5)) < 1.0e-4
    assert abs(float(level3 - dense)) < 1.0e-4


def test_generalized_sv_route_rejects_mutated_canonical_data() -> None:
    route = make_generalized_sv_sgqf_route()
    observations = tf.tensor_scatter_nd_add(
        route.observations, [[0, 0]], tf.constant([1.0e-6], tf.float64)
    )
    with pytest.raises(ValueError, match="observation identity rejected"):
        replace(route, observations=observations)


def test_generalized_sv_route_identity_is_cpu_pinned() -> None:
    route = make_generalized_sv_sgqf_route()
    assert route.manifest["state_sha256"] == GENERALIZED_SV_SGQF_STATE_SHA256
    assert route.manifest["observation_sha256"] == (
        GENERALIZED_SV_SGQF_OBSERVATION_SHA256
    )
