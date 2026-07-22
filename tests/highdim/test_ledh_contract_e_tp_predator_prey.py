from __future__ import annotations

import numpy as np
import tensorflow as tf
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss

from bayesfilter.highdim import ledh_contract_e_tp_predator_prey_tf as route
from bayesfilter.highdim.models import p30_predator_prey_fixture_model
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _predator_prey_dataset,
)


DTYPE = tf.float64


def _hermite(order: int) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, weights = hermgauss(order)
    return (
        tf.constant(np.sqrt(2.0) * nodes, DTYPE),
        tf.constant(weights / np.sqrt(np.pi), DTYPE),
    )


def _grid(order: int) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, weights = leggauss(order)
    prey = 65.0 + 55.0 * nodes
    predator = 6.0 + 12.0 * nodes
    prey_weight = 55.0 * weights
    predator_weight = 12.0 * weights
    first, second = np.meshgrid(prey, predator, indexing="ij")
    first_weight, second_weight = np.meshgrid(
        prey_weight, predator_weight, indexing="ij"
    )
    return (
        tf.constant(np.stack([first.ravel(), second.ravel()], axis=1), DTYPE),
        tf.constant((first_weight * second_weight).ravel(), DTYPE),
    )


def test_predator_continuation_one_step_matches_direct_quadrature() -> None:
    model = p30_predator_prey_fixture_model()
    theta = model.true_parameters()
    observation = tf.constant([[80.0, 3.8]], DTYPE)
    points = tf.constant([[48.0, 4.0], [55.0, 7.0]], DTYPE)
    grid, weights = _grid(9)
    actual = route.target_continuation_log_likelihood(
        model,
        theta,
        points,
        observation,
        grid,
        weights,
        first_future_time_index=1,
    )
    transition = route._pairwise_transition(model, theta, points, grid, 1)
    observation_log = model.observation_log_density(
        theta, grid, observation[0], t=1
    )
    expected = tf.reduce_logsumexp(
        transition + tf.math.log(weights)[None, :] + observation_log[None, :],
        axis=1,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_predator_analytic_one_step_continuation_matches_gaussian_quadrature() -> None:
    model = p30_predator_prey_fixture_model()
    theta = model.true_parameters()
    points = tf.constant([[48.0, 4.0], [55.0, 7.0]], DTYPE)
    observation = tf.constant([80.0, 3.8], DTYPE)
    means = model.transition_mean(theta, points)
    analytic = route.one_step_target_continuation_log_likelihood(
        model, theta, points, observation
    )
    errors = []
    for order in (13, 19, 25):
        nodes, weights = _hermite(order)
        standard, product_weights = route._product_rule(nodes, weights)
        process_chol = tf.linalg.cholesky(model.process_covariance)
        children = means[:, None, :] + tf.linalg.matmul(
            standard, process_chol, transpose_b=True
        )[None, :, :]
        quadrature = tf.reduce_logsumexp(
            tf.math.log(product_weights)[None, :]
            + tf.reshape(
                model.observation_log_density(
                    theta,
                    tf.reshape(children, [-1, 2]),
                    observation,
                    t=1,
                ),
                [2, -1],
            ),
            axis=1,
        )
        errors.append(float(tf.reduce_max(tf.abs(analytic - quadrature)).numpy()))
    assert errors[-1] < errors[0]
    assert errors[-1] < 2.0e-12


def test_predator_t1_is_initial_law_first_and_matches_direct_rule() -> None:
    model = p30_predator_prey_fixture_model()
    theta = model.true_parameters()
    observation = tf.convert_to_tensor(
        _predator_prey_dataset(81104)["observations"][:1], DTYPE
    )
    grid, grid_weights = _grid(7)
    gaps = []
    for order in (5, 7, 9):
        nodes, weights = _hermite(order)
        result = route.contract_e_tp_predator_prey_recursive_core(
            theta,
            observation,
            nodes,
            weights,
            tf.zeros([0, route.FEATURE_COUNT], tf.int32),
            tf.zeros([0, route.FEATURE_COUNT], DTYPE),
            grid,
            grid_weights,
            lookahead_steps=1,
        )
        initial_points, initial_log_weights, _standard, _standard_log = route.initial_rule(
            model, nodes, weights
        )
        expected = tf.reduce_logsumexp(
            initial_log_weights
            + model.observation_log_density(theta, initial_points, observation[0], t=0)
        )
        gaps.append(float(tf.abs(result["objective"] - expected).numpy()))
        assert bool(tf.reduce_all(result["valid_history"]).numpy())
    assert gaps[-1] < gaps[0]
    assert gaps[-1] < 2.0e-9


def test_predator_t1_total_score_matches_finite_difference() -> None:
    model = p30_predator_prey_fixture_model()
    theta = model.true_parameters()
    observation = tf.convert_to_tensor(
        _predator_prey_dataset(81104)["observations"][:1], DTYPE
    )
    nodes, weights = _hermite(7)
    grid, grid_weights = _grid(7)

    def scalar(value: tf.Tensor) -> tf.Tensor:
        return route.contract_e_tp_predator_prey_recursive_core(
            value,
            observation,
            nodes,
            weights,
            tf.zeros([0, route.FEATURE_COUNT], tf.int32),
            tf.zeros([0, route.FEATURE_COUNT], DTYPE),
            grid,
            grid_weights,
            lookahead_steps=1,
        )["objective"]

    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = scalar(theta)
    score = tape.gradient(value, theta)
    fd = []
    for index in range(6):
        step = tf.constant(1.0e-5, DTYPE) * tf.maximum(
            tf.constant(1.0, DTYPE), tf.abs(theta[index])
        )
        direction = tf.one_hot(index, 6, dtype=DTYPE)
        fd.append(
            (scalar(theta + step * direction) - scalar(theta - step * direction))
            / (2.0 * step)
        )
    np.testing.assert_allclose(score, tf.stack(fd), rtol=4e-6, atol=2e-7)
