"""Fused batch lane parity gates (NeuTra-eligibility prerequisite).

P-1 fused: batch-size-1 value AND score parity vs the single-cloud
canonical authority; multi-row independence; tf.function compilability
(the batch-native requirement). Declared tolerance: rtol 5e-4 (op-order),
recorded errors.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    PerPointScoreModel,
    canonical_batch_fused_value_score,
)
from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel,
    canonical_value_and_analytical_score,
)

DTYPE = tf.float64


def _single_model():
    def transition_mean_fn(theta, points):
        return points + theta[0] * tf.sin(points)

    def transition_mean_tangent_fn(theta, points, d_points):
        return (
            tf.sin(points)
            + d_points
            + theta[0] * tf.cos(points) * d_points
        )

    def observation_fn(points):
        return points

    def observation_jacobian_fn(points):
        return tf.broadcast_to(
            tf.eye(2, dtype=DTYPE), [tf.shape(points)[0], 2, 2]
        )

    def observation_tangent_fn(points, d_points):
        return d_points

    return NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_tangent_fn=observation_tangent_fn,
        process_covariance=tf.constant(0.4 * np.eye(2), DTYPE),
        observation_covariance=tf.constant(0.6 * np.eye(2), DTYPE),
    )


def _fused_model():
    def transition_mean_fn(theta, points):
        # per-point theta rows [M, P]
        return points + theta[:, 0:1] * tf.sin(points)

    def transition_mean_tangent_fn(theta, points, d_points, d_theta):
        return (
            d_theta[:, 0:1] * tf.sin(points)
            + d_points
            + theta[:, 0:1] * tf.cos(points) * d_points
        )

    def observation_fn(points):
        return points

    def observation_jacobian_fn(points):
        return tf.broadcast_to(
            tf.eye(2, dtype=DTYPE), [tf.shape(points)[0], 2, 2]
        )

    def observation_tangent_fn(points, d_points):
        return d_points

    return PerPointScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_tangent_fn=observation_tangent_fn,
        process_covariance=tf.constant(0.4 * np.eye(2), DTYPE),
        observation_covariance=tf.constant(0.6 * np.eye(2), DTYPE),
    )


def _fixture(seed: int, n: int = 6, horizon: int = 3):
    rng = np.random.default_rng(seed)
    initial = tf.constant(rng.standard_normal((n, 2)), DTYPE)
    covs = tf.constant(np.stack([np.eye(2)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, 2)), DTYPE)
    observations = tf.constant(rng.standard_normal((horizon, 2)), DTYPE)
    return initial, covs, noises, observations


def test_fused_batch_size_one_parity():
    initial, covs, noises, observations = _fixture(91)
    single = _single_model()
    value_ref, score_ref = canonical_value_and_analytical_score(
        single, tf.constant([0.6], DTYPE), initial, covs, noises,
        observations, flow_substeps=10, with_score=True,
    )
    fused = _fused_model()
    value_b, score_b, diag = canonical_batch_fused_value_score(
        fused,
        tf.constant([[0.6]], DTYPE),
        tf.constant([[1.0]], DTYPE),
        initial, covs, noises, observations,
        flow_substeps=10,
    )
    assert bool(diag["program_valid"][0].numpy())
    v_err = abs(float(value_b[0].numpy()) - float(value_ref.numpy()))
    s_err = abs(float(score_b[0].numpy()) - float(score_ref[0].numpy()))
    assert v_err < 5.0e-4 * max(abs(float(value_ref.numpy())), 1.0), v_err
    assert s_err < 5.0e-4 * max(abs(float(score_ref[0].numpy())), 1.0), s_err


def test_fused_rows_independent_and_distinct():
    initial, covs, noises, observations = _fixture(93)
    fused = _fused_model()
    value_b, score_b, _ = canonical_batch_fused_value_score(
        fused,
        tf.constant([[0.6], [0.9]], DTYPE),
        tf.constant([[1.0], [1.0]], DTYPE),
        initial, covs, noises, observations,
        flow_substeps=10,
    )
    value_row0, _, _ = canonical_batch_fused_value_score(
        fused,
        tf.constant([[0.6]], DTYPE),
        tf.constant([[1.0]], DTYPE),
        initial, covs, noises, observations,
        flow_substeps=10,
    )
    assert abs(
        float(value_b[0].numpy()) - float(value_row0[0].numpy())
    ) < 1.0e-12, "row 0 changed when row 1 appended"
    assert abs(
        float(value_b[0].numpy()) - float(value_b[1].numpy())
    ) > 1.0e-6, "distinct theta rows produced identical values"


def test_fused_lane_is_tf_function_compilable():
    """Batch-native requirement: the fused program must trace under
    tf.function with no Python-loop-over-rows dependence on batch size
    (rows enter only through tensor shapes)."""
    initial, covs, noises, observations = _fixture(97)
    fused = _fused_model()
    compiled = tf.function(
        lambda t, d: canonical_batch_fused_value_score(
            fused, t, d, initial, covs, noises, observations, flow_substeps=8
        ),
        autograph=False,
    )
    value_g, score_g, _ = compiled(
        tf.constant([[0.6], [0.8], [1.1]], DTYPE),
        tf.constant([[1.0], [1.0], [1.0]], DTYPE),
    )
    assert value_g.shape == (3,)
    assert bool(tf.reduce_all(tf.math.is_finite(value_g)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score_g)).numpy())
