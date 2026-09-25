"""Pre-unification row-independence gates for the fused LEDH lane.

The legacy fused lane implements batch and multi-direction semantics but not
Contract-E or higher-moment controls. Full-union row isolation is therefore a
post-extraction gate; these tests freeze the batch behavior that already exists.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    PerPointScoreModel,
    canonical_batch_fused_value_score,
)


def _model(dtype: tf.DType) -> PerPointScoreModel:
    def transition(theta_rows, points):
        return points + theta_rows[:, :1] * tf.sin(points)

    def transition_tangent(theta_rows, points, d_points, d_theta_rows):
        return (
            d_theta_rows[:, :1] * tf.sin(points)
            + d_points
            + theta_rows[:, :1] * tf.cos(points) * d_points
        )

    def observation(points):
        return points

    return PerPointScoreModel(
        transition_mean_fn=transition,
        transition_mean_tangent_fn=transition_tangent,
        observation_fn=observation,
        observation_jacobian_fn=lambda points: tf.broadcast_to(
            tf.eye(2, dtype=dtype), [tf.shape(points)[0], 2, 2]
        ),
        observation_tangent_fn=lambda points, d_points: d_points,
        process_covariance=tf.constant(0.4 * np.eye(2), dtype),
        observation_covariance=tf.constant(0.6 * np.eye(2), dtype),
    )


def _fixture(dtype: tf.DType):
    rng = np.random.default_rng(20260910)
    n, horizon = 6, 2
    return (
        tf.constant(rng.standard_normal((n, 2)), dtype),
        tf.constant(np.stack([np.eye(2)] * n), dtype),
        tf.constant(rng.standard_normal((horizon, n, 2)), dtype),
        tf.constant(rng.standard_normal((horizon, 2)), dtype),
    )


def _run(theta: tf.Tensor, directions: tf.Tensor, dtype: tf.DType):
    initial, covariances, noises, observations = _fixture(dtype)
    return canonical_batch_fused_value_score(
        _model(dtype),
        theta,
        directions,
        initial,
        covariances,
        noises,
        observations,
        substeps=4,
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("direction_count", [1, 2, 5])
def test_identical_rows_produce_identical_outputs(
    batch_size: int, direction_count: int
):
    dtype = tf.float64
    theta = tf.fill([batch_size, 1], tf.constant(0.6, dtype))
    one_row = tf.reshape(tf.linspace(tf.constant(0.25, dtype), 1.0, direction_count), [1, direction_count, 1])
    directions = tf.tile(one_row, [batch_size, 1, 1])
    values, scores, diagnostics = _run(theta, directions, dtype)

    assert bool(tf.reduce_all(diagnostics["program_valid"]).numpy())
    for row in range(1, batch_size):
        np.testing.assert_array_equal(values[0].numpy(), values[row].numpy())
        np.testing.assert_array_equal(scores[0].numpy(), scores[row].numpy())


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("direction_count", [1, 2, 5])
def test_perturbing_one_row_does_not_change_other_rows(
    batch_size: int, direction_count: int
):
    dtype = tf.float64
    theta = tf.reshape(tf.linspace(tf.constant(0.45, dtype), 0.9, batch_size), [batch_size, 1])
    direction_row = tf.reshape(tf.linspace(tf.constant(0.25, dtype), 1.0, direction_count), [1, direction_count, 1])
    directions = tf.tile(direction_row, [batch_size, 1, 1])
    baseline_values, baseline_scores, _ = _run(theta, directions, dtype)

    for changed_row in range(batch_size):
        changed_theta = tf.tensor_scatter_nd_add(
            theta, [[changed_row, 0]], [tf.constant(0.35, dtype)]
        )
        changed_values, changed_scores, _ = _run(
            changed_theta, directions, dtype
        )
        assert not np.array_equal(
            changed_values[changed_row].numpy(),
            baseline_values[changed_row].numpy(),
        )
        for row in range(batch_size):
            if row == changed_row:
                continue
            np.testing.assert_array_equal(
                changed_values[row].numpy(), baseline_values[row].numpy()
            )
            np.testing.assert_array_equal(
                changed_scores[row].numpy(), baseline_scores[row].numpy()
            )
