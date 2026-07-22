from __future__ import annotations

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.sir_latent_preclip_reference_tf import (
    dense_latent_sir_value,
    dense_latent_sir_value_and_manual_score,
    prepare_reduced_dense_grids,
    reduced_latent_preclip_sir_model,
)


DTYPE = tf.float64


def _fixture(time_steps: int, *, order: int, radius: float):
    model = reduced_latent_preclip_sir_model()
    theta = tf.constant([0.03, -0.02, 0.04], DTYPE)
    observations = tf.constant([[0.15], [0.1], [0.05]], DTYPE)[: time_steps + 1]
    grids = prepare_reduced_dense_grids(
        model, theta, time_steps=time_steps, order=order, radius=radius
    )
    return model, theta, observations, grids


def _autodiff_score(model, theta, observations, grids) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = dense_latent_sir_value(model, theta, observations, grids)
    score = tape.gradient(value, theta)
    assert score is not None
    return score


def _central_fd(model, theta, observations, grids, step: float = 1.0e-5) -> tf.Tensor:
    columns = []
    for index in range(3):
        direction = tf.one_hot(index, 3, dtype=DTYPE)
        plus = dense_latent_sir_value(
            model, theta + step * direction, observations, grids
        )
        minus = dense_latent_sir_value(
            model, theta - step * direction, observations, grids
        )
        columns.append((plus - minus) / (2.0 * step))
    return tf.stack(columns)


def test_manual_filtering_score_matches_autodiff_and_same_scalar_fd_t1_t2() -> None:
    for time_steps in (1, 2):
        model, theta, observations, grids = _fixture(
            time_steps, order=15, radius=7.0
        )
        result = dense_latent_sir_value_and_manual_score(
            model, theta, observations, grids
        )
        autodiff = _autodiff_score(model, theta, observations, grids)
        finite_difference = _central_fd(model, theta, observations, grids)
        tf.debugging.assert_near(result["score"], autodiff, atol=2.0e-9, rtol=2.0e-9)
        tf.debugging.assert_near(result["score"], finite_difference, atol=2.0e-7, rtol=2.0e-7)
        tf.debugging.assert_near(
            result["objective"], tf.reduce_sum(result["increment_history"]), atol=0.0
        )
        assert bool(tf.reduce_all(tf.math.is_finite(result["boundary_mass_history"])).numpy())


def test_previous_marginal_score_is_required_at_t2() -> None:
    model, theta, observations, grids = _fixture(2, order=15, radius=7.0)
    total = dense_latent_sir_value_and_manual_score(model, theta, observations, grids)
    stopped = dense_latent_sir_value_and_manual_score(
        model,
        theta,
        observations,
        grids,
        stop_previous_marginal_score=True,
    )
    autodiff = _autodiff_score(model, theta, observations, grids)
    total_error = tf.reduce_max(tf.abs(total["score"] - autodiff))
    stopped_error = tf.reduce_max(tf.abs(stopped["score"] - autodiff))
    assert float(total_error.numpy()) < 2.0e-9
    assert float(stopped_error.numpy()) > 1.0e-4
    assert stopped["previous_marginal_score_status"] == "stopped_negative_control"


def test_reference_order_and_range_refinement_is_small_and_boundary_mass_decays() -> None:
    configurations = ((29, 6.0), (33, 6.0), (33, 7.0))
    results = []
    for order, radius in configurations:
        model, theta, observations, grids = _fixture(2, order=order, radius=radius)
        results.append(
            dense_latent_sir_value_and_manual_score(model, theta, observations, grids)
        )
    order_value_gap = abs(float((results[1]["objective"] - results[0]["objective"]).numpy()))
    range_value_gap = abs(float((results[2]["objective"] - results[1]["objective"]).numpy()))
    order_score_gap = float(
        tf.reduce_max(tf.abs(results[1]["score"] - results[0]["score"])).numpy()
    )
    range_score_gap = float(
        tf.reduce_max(tf.abs(results[2]["score"] - results[1]["score"])).numpy()
    )
    assert order_value_gap < 2.0e-4
    assert range_value_gap < 2.0e-4
    assert order_score_gap < 2.0e-3
    assert range_score_gap < 2.0e-3
    assert float(tf.reduce_max(results[1]["boundary_mass_history"]).numpy()) < 5.0e-10
    assert float(tf.reduce_max(results[2]["boundary_mass_history"]).numpy()) < 1.0e-12
    assert np.isfinite([order_value_gap, range_value_gap, order_score_gap, range_score_gap]).all()
