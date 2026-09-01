from __future__ import annotations

import numpy as np
import tensorflow as tf

from bayesfilter.nonlinear.rectangular_srukf_tf import (
    TFRectangularSRUKFDerivatives,
    TFRectangularSRUKFFixedBranch,
    TFRectangularSRUKFModel,
    tf_rectangular_srukf_value,
    tf_rectangular_srukf_value_and_score,
)


def _model() -> TFRectangularSRUKFModel:
    return TFRectangularSRUKFModel(
        tf.constant([[0.0]], tf.float64),
        tf.constant([[[0.5]]], tf.float64),
        tf.constant([[[0.1]]], tf.float64),
        tf.constant([[[0.0], [0.0]]], tf.float64),
        lambda state, process: state + process,
        lambda state: tf.concat([state, state], axis=-1),
    )


def test_rectangular_temporal_value_route_handles_on_support_singular_observations() -> None:
    result = tf_rectangular_srukf_value(
        tf.constant([[[0.1, 0.1], [0.2, 0.2]]], tf.float64),
        _model(),
        jit_compile=False,
    )
    assert bool(result.diagnostics["value_only"])
    assert result.diagnostics["rank_branch_status"].numpy() == b"value_only_rank_discovery"
    assert bool(tf.reduce_all(result.diagnostics["on_support"]))
    assert bool(tf.reduce_all(tf.math.is_finite(result.log_likelihood)))
    assert result.filtered_factor.shape == (1, 1, 1)


def test_rectangular_temporal_value_route_marks_off_support_without_nan() -> None:
    result = tf_rectangular_srukf_value(
        tf.constant([[[0.1, 0.2]]], tf.float64),
        _model(),
        jit_compile=False,
    )
    assert not bool(result.diagnostics["on_support"][0])
    assert np.isneginf(result.log_likelihood.numpy()[0])
    assert not bool(tf.reduce_any(tf.math.is_nan(result.log_likelihood)))


def _fixed_branch_model(theta: float = 0.0):
    model = TFRectangularSRUKFModel(
        tf.constant([[0.0, 0.0]], tf.float64),
        tf.constant([[[0.0], [0.5]]], tf.float64),
        tf.constant([[[0.2 + 0.1 * theta]]], tf.float64),
        tf.constant([[[0.0], [0.0]]], tf.float64),
        lambda state, process: tf.stack(
            [state[..., 0] + process[..., 0], state[..., 1]], axis=-1
        ),
        lambda state: tf.stack([state[..., 0], state[..., 0]], axis=-1),
    )
    derivatives = TFRectangularSRUKFDerivatives(
        tf.zeros([1, 1, 2], tf.float64),
        tf.zeros([1, 1, 2, 1], tf.float64),
        tf.constant([[[[0.1]]]], tf.float64),
        tf.zeros([1, 1, 2, 1], tf.float64),
        lambda state, process: tf.broadcast_to(
            tf.eye(2, dtype=tf.float64), [1, tf.shape(state)[1], 2, 2]
        ),
        lambda state, process: tf.broadcast_to(
            tf.constant([[[1.0], [0.0]]], tf.float64),
            [1, tf.shape(state)[1], 2, 1],
        ),
        lambda state, process: tf.zeros([1, 1, tf.shape(state)[1], 2], tf.float64),
        lambda state: tf.broadcast_to(
            tf.constant([[[1.0, 0.0], [1.0, 0.0]]], tf.float64),
            [1, tf.shape(state)[1], 2, 2],
        ),
        lambda state: tf.zeros([1, 1, tf.shape(state)[1], 2], tf.float64),
    )
    return model, derivatives


def _fixed_branch() -> TFRectangularSRUKFFixedBranch:
    return TFRectangularSRUKFFixedBranch(
        predicted_rank=2,
        predicted_permutation=(0, 1),
        innovation_rank=1,
        innovation_permutation=(0, 1),
        filtered_rank=1,
        filtered_permutation=(1, 0),
    )


def test_fixed_rank_temporal_score_matches_fd_and_cpu_xla() -> None:
    observations = tf.constant([[[0.1, 0.1], [0.2, 0.2]]], tf.float64)
    result = tf_rectangular_srukf_value_and_score(
        observations, *_fixed_branch_model(), _fixed_branch(), jit_compile=False
    )
    eps = 1.0e-6
    plus = tf_rectangular_srukf_value_and_score(
        observations, *_fixed_branch_model(eps), _fixed_branch(), jit_compile=False
    )
    minus = tf_rectangular_srukf_value_and_score(
        observations, *_fixed_branch_model(-eps), _fixed_branch(), jit_compile=False
    )
    np.testing.assert_allclose(
        result.score[:, 0],
        (plus.log_likelihood - minus.log_likelihood) / (2.0 * eps),
        rtol=2e-6,
        atol=2e-8,
    )
    compiled = tf_rectangular_srukf_value_and_score(
        observations, *_fixed_branch_model(), _fixed_branch(), jit_compile=True
    )
    np.testing.assert_allclose(compiled.log_likelihood, result.log_likelihood, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(compiled.score, result.score, rtol=1e-11, atol=1e-12)
    assert bool(result.diagnostics["score_valid"][0])
    assert result.diagnostics["branch_status"][0].numpy() == b"fixed_branch_valid"


def test_fixed_rank_temporal_off_support_invalidates_score_without_nan_value() -> None:
    result = tf_rectangular_srukf_value_and_score(
        tf.constant([[[0.1, 0.2]]], tf.float64),
        *_fixed_branch_model(),
        _fixed_branch(),
        jit_compile=False,
    )
    assert np.isneginf(result.log_likelihood.numpy()[0])
    assert np.isnan(result.score.numpy()[0, 0])
    assert not bool(result.diagnostics["score_valid"][0])
