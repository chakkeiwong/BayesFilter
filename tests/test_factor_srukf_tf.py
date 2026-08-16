from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.nonlinear.factor_srukf_tf import (
    TFFactorSRUKFDerivatives,
    TFFactorSRUKFModel,
    tf_factor_srukf_value_and_score,
)
from bayesfilter.nonlinear.factor_srukf_tf import _right_solve


def _model(batch: int = 2, parameter_dim: int = 1):
    mean = tf.constant([[0.1], [0.2]][:batch], tf.float64)
    sf = tf.constant([[[0.7]], [[0.8]]][:batch], tf.float64)
    qf = tf.constant([[[0.2]], [[0.25]]][:batch], tf.float64)
    rf = tf.constant([[[0.3]], [[0.35]]][:batch], tf.float64)
    def transition(x, q): return x + q
    def observe(x): return 2.0 * x
    def sj(x, q): return tf.ones([batch, tf.shape(x)[1], 1, 1], tf.float64)
    def qj(x, q): return tf.ones([batch, tf.shape(x)[1], 1, 1], tf.float64)
    def sd(x, q): return tf.ones([batch, parameter_dim, tf.shape(x)[1], 1], tf.float64)
    def oj(x): return tf.fill([batch, tf.shape(x)[1], 1, 1], tf.constant(2.0, tf.float64))
    def od(x): return tf.ones([batch, parameter_dim, tf.shape(x)[1], 1], tf.float64)
    model = TFFactorSRUKFModel(mean, sf, qf, rf, transition, observe)
    derivatives = TFFactorSRUKFDerivatives(
        tf.zeros([batch, parameter_dim, 1], tf.float64),
        tf.zeros([batch, parameter_dim, 1, 1], tf.float64),
        tf.zeros([batch, parameter_dim, 1, 1], tf.float64),
        tf.zeros([batch, parameter_dim, 1, 1], tf.float64),
        sj, qj, sd, oj, od,
    )
    return model, derivatives


def test_factor_filter_batch_and_modes_are_finite() -> None:
    model, derivatives = _model()
    observations = tf.constant([[[0.3], [0.4]], [[0.5], [0.6]]], tf.float64)
    eager = tf_factor_srukf_value_and_score(observations, model, derivatives, jit_compile=False)
    graph = tf_factor_srukf_value_and_score(observations, model, derivatives, jit_compile=True)
    for result in (eager, graph):
        for value in (result.log_likelihood, result.score, result.filtered_mean, result.filtered_factor, result.d_filtered_mean, result.d_filtered_factor):
            assert bool(tf.reduce_all(tf.math.is_finite(value)))
    np.testing.assert_allclose(eager.log_likelihood, graph.log_likelihood, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(eager.score, graph.score, rtol=1e-10, atol=1e-11)
    duplicated_model, duplicated_derivatives = _model(batch=2)
    duplicated_model = TFFactorSRUKFModel(
        tf.repeat(duplicated_model.initial_mean[:1], 2, axis=0),
        tf.repeat(duplicated_model.initial_factor[:1], 2, axis=0),
        tf.repeat(duplicated_model.process_factor[:1], 2, axis=0),
        tf.repeat(duplicated_model.observation_factor[:1], 2, axis=0),
        duplicated_model.transition_fn,
        duplicated_model.observation_fn,
    )
    duplicated_derivatives = TFFactorSRUKFDerivatives(
        tf.repeat(duplicated_derivatives.d_initial_mean[:1], 2, axis=0),
        tf.repeat(duplicated_derivatives.d_initial_factor[:1], 2, axis=0),
        tf.repeat(duplicated_derivatives.d_process_factor[:1], 2, axis=0),
        tf.repeat(duplicated_derivatives.d_observation_factor[:1], 2, axis=0),
        duplicated_derivatives.transition_state_jacobian_fn,
        duplicated_derivatives.transition_process_jacobian_fn,
        duplicated_derivatives.d_transition_fn,
        duplicated_derivatives.observation_state_jacobian_fn,
        duplicated_derivatives.d_observation_fn,
    )
    duplicate_obs = tf.repeat(observations[:1], 2, axis=0)
    duplicate = tf_factor_srukf_value_and_score(duplicate_obs, duplicated_model, duplicated_derivatives, jit_compile=False)
    np.testing.assert_allclose(duplicate.log_likelihood[0], duplicate.log_likelihood[1], rtol=1e-12, atol=1e-12)


def test_factor_filter_rejects_nonfinite_observations() -> None:
    model, derivatives = _model()
    with pytest.raises(tf.errors.InvalidArgumentError):
        tf_factor_srukf_value_and_score(tf.fill([2, 1, 1], tf.constant(np.nan, tf.float64)), model, derivatives, jit_compile=False).log_likelihood.numpy()


def test_gain_right_triangular_solve_matches_dense_orientation() -> None:
    factor = tf.constant([[[2.0, 0.0], [0.4, 1.5]]], tf.float64)
    matrix = tf.constant([[[1.2, -0.3], [0.7, 2.1]]], tf.float64)
    solved = _right_solve(factor, matrix)
    expected = matrix @ tf.linalg.inv(factor)
    np.testing.assert_allclose(solved, expected, rtol=1e-13, atol=1e-13)


def test_factor_filter_score_and_carried_derivatives_match_centered_fd() -> None:
    def evaluate(theta: float):
        model, derivatives = _model(batch=2, parameter_dim=1)
        model = TFFactorSRUKFModel(
            model.initial_mean[:1], model.initial_factor[:1], model.process_factor[:1],
            model.observation_factor[:1], model.transition_fn,
            lambda x, theta=theta: 2.0 * x + theta,
        )
        derivatives = TFFactorSRUKFDerivatives(
            derivatives.d_initial_mean[:1], derivatives.d_initial_factor[:1],
            derivatives.d_process_factor[:1], derivatives.d_observation_factor[:1],
            lambda x, q: tf.ones([tf.shape(x)[0], tf.shape(x)[1], 1, 1], tf.float64),
            lambda x, q: tf.ones([tf.shape(x)[0], tf.shape(x)[1], 1, 1], tf.float64),
            lambda x, q: tf.zeros([tf.shape(x)[0], 1, tf.shape(x)[1], 1], tf.float64),
            lambda x: tf.fill([1, tf.shape(x)[1], 1, 1], tf.constant(2.0, tf.float64)),
            lambda x: tf.ones([tf.shape(x)[0], 1, tf.shape(x)[1], 1], tf.float64),
        )
        return tf_factor_srukf_value_and_score(
            tf.constant([[[0.3], [0.4]]], tf.float64), model, derivatives, jit_compile=False
        )

    result = evaluate(0.0)
    eps = 1.0e-6
    plus, minus = evaluate(eps), evaluate(-eps)
    fd = (plus.log_likelihood - minus.log_likelihood) / (2.0 * eps)
    np.testing.assert_allclose(result.score[0], fd, rtol=1e-7, atol=1e-9)
    fd_mean = (plus.filtered_mean - minus.filtered_mean) / (2.0 * eps)
    np.testing.assert_allclose(result.d_filtered_mean[0], fd_mean, rtol=1e-7, atol=1e-9)
