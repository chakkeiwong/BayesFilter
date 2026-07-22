from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss

from bayesfilter.highdim import ledh_contract_e_tp_scalar_sv_tf as model
from bayesfilter.highdim.ledh_forward_contract import (
    ACTUAL_SV_ROW_ID,
    GENERALIZED_SV_ROW_ID,
    KSC_SV_ROW_ID,
)


DTYPE = tf.float64
THETA = {
    ACTUAL_SV_ROW_ID: tf.constant([0.2533471031357997, -0.916290731874155], DTYPE),
    KSC_SV_ROW_ID: tf.constant([0.2533471031357997, -0.916290731874155], DTYPE),
    GENERALIZED_SV_ROW_ID: tf.constant(
        [1.0824113944610982, -2.076793740349318, 0.0], DTYPE
    ),
}


def _hermite_rule(order: int) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, weights = hermgauss(order)
    return (
        tf.constant(np.sqrt(2.0) * nodes, DTYPE),
        tf.constant(weights / np.sqrt(np.pi), DTYPE),
    )


def _legendre_rule(order: int, radius: float) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, weights = leggauss(order)
    return (
        tf.constant(radius * nodes, DTYPE),
        tf.constant(radius * weights, DTYPE),
    )


@pytest.mark.parametrize(
    "row_id",
    (ACTUAL_SV_ROW_ID, KSC_SV_ROW_ID, GENERALIZED_SV_ROW_ID),
)
def test_target_continuation_one_step_matches_direct_quadrature(row_id: str) -> None:
    spec = model.make_scalar_sv_spec(row_id)
    theta = THETA[row_id]
    grid, weights = _legendre_rule(65, 10.0)
    points = tf.constant([[-1.2], [0.0], [0.9]], DTYPE)
    target_observation, _ = model.target_and_flow_observations(
        spec, tf.constant([[0.7]], DTYPE)
    )

    actual = model.target_continuation_log_likelihood(
        spec,
        theta,
        points,
        target_observation,
        grid,
        weights,
        first_future_time_index=1,
    )
    transition = model._pairwise_transition_log_density(
        spec, theta, points, grid[:, None], 1
    )
    observation = spec.model.observation_log_density(
        theta, grid[:, None], target_observation[0], t=1
    )
    expected = tf.reduce_logsumexp(
        transition + tf.math.log(weights)[None, :] + observation[None, :],
        axis=1,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize(
    "row_id",
    (ACTUAL_SV_ROW_ID, KSC_SV_ROW_ID, GENERALIZED_SV_ROW_ID),
)
def test_target_continuation_total_derivative_matches_fd(row_id: str) -> None:
    spec = model.make_scalar_sv_spec(row_id)
    theta = THETA[row_id]
    grid, weights = _legendre_rule(65, 10.0)
    target_observations, _ = model.target_and_flow_observations(
        spec, tf.constant([[0.7], [-0.4]], DTYPE)
    )

    def scalar(value: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(
            model.target_continuation_log_likelihood(
                spec,
                value,
                tf.constant([[-0.2], [0.6]], DTYPE),
                target_observations,
                grid,
                weights,
                first_future_time_index=1,
            )
        )

    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = scalar(theta)
    gradient = tape.gradient(value, theta)
    steps = tf.constant([2.0e-5] * spec.parameter_dimension, DTYPE)
    fd = []
    for index in range(spec.parameter_dimension):
        direction = tf.one_hot(index, spec.parameter_dimension, dtype=DTYPE)
        fd.append(
            (scalar(theta + steps[index] * direction) - scalar(theta - steps[index] * direction))
            / (2.0 * steps[index])
        )
    np.testing.assert_allclose(gradient, tf.stack(fd), rtol=2e-7, atol=2e-8)


def test_generalized_sv_keeps_raw_target_and_log_square_flow_separate() -> None:
    spec = model.make_scalar_sv_spec(GENERALIZED_SV_ROW_ID)
    raw = tf.constant([[0.7], [-0.4]], DTYPE)
    target, flow = model.target_and_flow_observations(spec, raw)
    np.testing.assert_allclose(target, raw, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        flow, tf.math.log(tf.square(raw) + 1.0e-6), rtol=0.0, atol=2.0e-14
    )
    assert not np.allclose(target.numpy(), flow.numpy())
    assert spec.transition_before_first_observation


def test_progressive_features_deduplicate_identical_late_horizon_marks() -> None:
    spec = model.make_scalar_sv_spec(GENERALIZED_SV_ROW_ID)
    theta = THETA[GENERALIZED_SV_ROW_ID]
    grid, weights = _legendre_rule(65, 10.0)
    target, _flow = model.target_and_flow_observations(
        spec, tf.constant([[0.7], [-0.4], [0.2]], DTYPE)
    )
    points = tf.constant([[-0.8], [0.0], [0.9]], DTYPE)

    assert model.effective_progressive_lookaheads((1, 3, 8), 3) == (1, 3)
    features = model.progressive_features(
        spec,
        theta,
        points,
        target,
        grid,
        weights,
        first_future_time_index=1,
        requested_lookaheads=(1, 3, 8),
    )

    assert features.shape == (5, 3)
    np.testing.assert_allclose(features[0], 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(features[1], points[:, 0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(features[2], tf.square(points[:, 0]), rtol=0.0, atol=0.0)
    assert not np.allclose(features[3].numpy(), features[4].numpy())


def test_progressive_feature_total_derivative_matches_fd() -> None:
    spec = model.make_scalar_sv_spec(GENERALIZED_SV_ROW_ID)
    theta = THETA[GENERALIZED_SV_ROW_ID]
    grid, weights = _legendre_rule(65, 10.0)
    target, _flow = model.target_and_flow_observations(
        spec, tf.constant([[0.7], [-0.4], [0.2]], DTYPE)
    )

    def scalar(value: tf.Tensor) -> tf.Tensor:
        features = model.progressive_features(
            spec,
            value,
            tf.constant([[-0.8], [0.0], [0.9]], DTYPE),
            target,
            grid,
            weights,
            first_future_time_index=1,
            requested_lookaheads=(1, 3),
        )
        return tf.reduce_sum(features[3:])

    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = scalar(theta)
    gradient = tape.gradient(value, theta)
    step = tf.constant(2.0e-5, DTYPE)
    finite_difference = []
    for index in range(spec.parameter_dimension):
        direction = tf.one_hot(index, spec.parameter_dimension, dtype=DTYPE)
        finite_difference.append(
            (scalar(theta + step * direction) - scalar(theta - step * direction))
            / (2.0 * step)
        )
    np.testing.assert_allclose(
        gradient, tf.stack(finite_difference), rtol=4e-7, atol=4e-8
    )


@pytest.mark.parametrize(
    "row_id",
    (ACTUAL_SV_ROW_ID, KSC_SV_ROW_ID, GENERALIZED_SV_ROW_ID),
)
def test_t1_corrected_ledh_scalar_converges_to_initial_quadrature(row_id: str) -> None:
    spec = model.make_scalar_sv_spec(row_id)
    theta = THETA[row_id]
    grid, grid_weights = _legendre_rule(33, 10.0)
    target, flow = model.target_and_flow_observations(
        spec, tf.constant([[0.7]], DTYPE)
    )
    gaps = []
    for order in (25, 41, 65):
        nodes, weights = _hermite_rule(order)
        result = model.contract_e_tp_scalar_sv_recursive_core(
            spec,
            theta,
            target,
            flow,
            nodes,
            weights,
            tf.zeros([0, model.FEATURE_COUNT], tf.int32),
            tf.zeros([0, model.FEATURE_COUNT], DTYPE),
            grid,
            grid_weights,
            lookahead_steps=1,
        )
        initial_points, _log_weights, _nodes, _node_log_weights = model.initial_rule(
            spec, theta, nodes, weights
        )
        if spec.transition_before_first_observation:
            previous = tf.repeat(initial_points, order)
            _mean, _scale, gamma, process_scale = model._dynamics(spec, theta)
            following = (
                _mean
                + gamma * (previous - _mean)
                + process_scale * tf.tile(nodes, [order])
            )
            direct = tf.reduce_logsumexp(
                tf.repeat(tf.math.log(weights), order)
                + tf.tile(tf.math.log(weights), [order])
                + spec.model.observation_log_density(
                    theta, following[:, None], target[0], t=0
                )
            )
        else:
            direct = tf.reduce_logsumexp(
                tf.math.log(weights)
                + spec.model.observation_log_density(
                    theta, initial_points[:, None], target[0], t=0
                )
            )
        gaps.append(float(tf.abs(result["objective"] - direct).numpy()))
        assert bool(tf.reduce_all(result["valid_history"]).numpy())
    if spec.transition_before_first_observation:
        assert max(gaps) < 2.0e-12
    else:
        assert gaps[-1] < gaps[0]
    assert gaps[-1] < 2.0e-7
