"""Independent covariance/finite-difference references for the pruned SRUKF."""

from dataclasses import replace
import math
import os

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.linear.stack_qr_tf import batched_semidefinite_stack_qr_lower
from bayesfilter.nonlinear.pruned_srukf_tf import (
    TFPrunedSRUKFModel,
    make_pruned_srukf_value_and_score,
    tf_pruned_srukf_filter,
)


def _model(parameters):
    parameters = tf.convert_to_tensor(parameters, tf.float64)
    batch = parameters.shape[0]
    slope, curvature, log_noise = tf.unstack(parameters, axis=1)
    transition_hessian = tf.stack([curvature, curvature * 0.3, curvature * 0.3, curvature * 0.2], axis=1)
    return TFPrunedSRUKFModel(
        initial_mean=tf.broadcast_to(tf.constant([[0.1, 0.0]], tf.float64), [batch, 2]),
        initial_factor=tf.broadcast_to(tf.constant([[[0.2], [0.0]]], tf.float64), [batch, 2, 1]),
        transition_matrix=slope[:, None, None],
        innovation_loading=tf.ones([batch, 1, 1], tf.float64) * 0.15,
        transition_hessian=tf.reshape(transition_hessian, [batch, 1, 2, 2]),
        second_order_constant=tf.ones([batch, 1], tf.float64) * 0.01,
        observation_constant=tf.ones([batch, 1], tf.float64) * 0.02,
        observation_matrix=tf.ones([batch, 1, 1], tf.float64),
        observation_hessian=tf.ones([batch, 1, 1, 1], tf.float64) * 0.1,
        innovation_factor=tf.ones([batch, 1, 1], tf.float64),
        observation_factor=tf.exp(log_noise[:, None, None]),
    )


def _covariance_reference(parameters, observations):
    slope, curvature, log_noise = parameters
    mean = np.array([0.1, 0.0])
    factor = np.diag([0.2, 0.0])
    value = 0.0
    offsets = np.vstack([np.zeros((1, 3)), np.sqrt(3) * np.eye(3), -np.sqrt(3) * np.eye(3)])
    weights = np.array([0.0] + [1 / 6] * 6)
    covariance_weights = np.array([2.0] + [1 / 6] * 6)
    for observed in observations:
        augmented = np.zeros((3, 3))
        augmented[:2, :2] = factor
        augmented[2, 2] = 1.0
        points = np.r_[mean, 0.0] + offsets @ augmented.T
        first, second, noise = points.T
        predicted = np.column_stack([
            slope * first + 0.15 * noise,
            slope * second + 0.01 + 0.5 * curvature * (first**2 + 0.6 * first * noise + 0.2 * noise**2),
        ])
        measurements = 0.02 + predicted[:, 0] + predicted[:, 1] + 0.05 * predicted[:, 0]**2
        mean_predicted = weights @ predicted
        observation_predicted = weights @ measurements
        state_residuals = predicted - mean_predicted
        observation_residuals = measurements - observation_predicted
        covariance = (state_residuals.T * covariance_weights) @ state_residuals
        variance = covariance_weights @ observation_residuals**2 + np.exp(2 * log_noise)
        cross = (state_residuals.T * covariance_weights) @ observation_residuals
        innovation = observed - observation_predicted
        value += -0.5 * (math.log(2 * math.pi * variance) + innovation**2 / variance)
        mean = mean_predicted + cross * innovation / variance
        covariance -= np.outer(cross, cross) / variance
        factor = np.linalg.cholesky(covariance)
    return value, mean, factor @ factor.T


def test_pruning_mixed_terms_half_convention_and_no_second_order_feedback():
    model = _model([[0.6, 0.4, -2.0]])
    state = tf.constant([[[0.2, 0.3], [0.2, 1.3]]], tf.float64)
    noise = tf.constant([[[0.7], [0.7]]], tf.float64)
    result = model.transition(state, noise).numpy()[0]
    expected_second = 0.6 * 0.3 + 0.01 + 0.2 * (0.2**2 + 0.6 * 0.2 * 0.7 + 0.2 * 0.7**2)
    np.testing.assert_allclose(result[0], [0.6 * 0.2 + 0.15 * 0.7, expected_second], atol=1e-15)
    np.testing.assert_allclose(result[1] - result[0], [0.0, 0.6], atol=1e-15)
    np.testing.assert_allclose(model.observe(state).numpy()[0, :, 0], [0.522, 1.522], atol=1e-15)


@pytest.mark.parametrize("time_steps", [1, 5, 12])
def test_matches_independent_covariance_recursion(time_steps):
    parameters = [0.65, 0.3, -2.0]
    observations = np.linspace(-0.1, 0.25, time_steps)
    result = tf_pruned_srukf_filter(observations[None, :, None], _model([parameters]), jit_compile=False)
    expected_value, expected_mean, expected_covariance = _covariance_reference(parameters, observations)
    assert result.diagnostics["valid"].numpy().tolist() == [True]
    np.testing.assert_allclose(result.log_likelihood, [expected_value], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(result.filtered_mean[0], expected_mean, rtol=1e-11, atol=1e-12)
    factor = result.filtered_factor.numpy()[0]
    np.testing.assert_allclose(factor @ factor.T, expected_covariance, rtol=1e-11, atol=1e-12)


def test_linear_kalman_limit_with_structurally_zero_second_order_state():
    base = _model([[0.7, 0.0, -2.0]])
    model = replace(base, second_order_constant=tf.zeros_like(base.second_order_constant), observation_hessian=tf.zeros_like(base.observation_hessian))
    observations = np.array([0.1, -0.2, 0.3, 0.0])
    result = tf_pruned_srukf_filter(observations[None, :, None], model, jit_compile=False)
    mean, variance, value = 0.1, 0.04, 0.0
    for observed in observations:
        mean *= 0.7
        variance = 0.7**2 * variance + 0.15**2
        innovation_variance = variance + np.exp(-4)
        innovation = observed - 0.02 - mean
        value -= 0.5 * (np.log(2 * np.pi * innovation_variance) + innovation**2 / innovation_variance)
        mean += variance / innovation_variance * innovation
        variance -= variance**2 / innovation_variance
    np.testing.assert_allclose(result.log_likelihood, [value], atol=1e-12)
    np.testing.assert_allclose(result.filtered_mean, [[mean, 0.0]], atol=1e-12)
    factor = result.filtered_factor.numpy()[0]
    np.testing.assert_allclose(factor @ factor.T, [[variance, 0], [0, 0]], atol=1e-12)


@pytest.mark.parametrize("singular", [False, True])
def test_rank_tolerant_qr_preserves_gram_matrix_and_gradients(singular):
    rng = np.random.default_rng(82)
    stack = rng.normal(size=(2, 3, 9))
    if singular:
        stack[:, 2, :] = 0.0
    parameters = tf.constant([0.4, 0.7], tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(parameters)
        factor = batched_semidefinite_stack_qr_lower(stack * parameters[:, None, None])
        loss = tf.reduce_sum(tf.square(factor))
    derivative = tape.gradient(loss, parameters)
    expected = stack @ stack.transpose(0, 2, 1) * parameters.numpy()[:, None, None]**2
    np.testing.assert_allclose(factor @ tf.linalg.matrix_transpose(factor), expected, atol=2e-14)
    np.testing.assert_allclose(derivative, 2 * parameters.numpy() * np.sum(stack**2, axis=(1, 2)), rtol=1e-13)


def test_scale_separation_does_not_trigger_a_condition_cap():
    stack = tf.constant([[[1e12, 1e12, 0.0], [0.0, 0.0, 1e-12]]], tf.float64)
    factor = batched_semidefinite_stack_qr_lower(stack).numpy()[0]
    np.testing.assert_allclose(np.diag(factor), [np.sqrt(2) * 1e12, 1e-12], rtol=1e-14)


def test_total_score_and_batch_independence():
    parameters = np.array([[0.65, 0.3, -2.0], [0.45, 0.18, -1.5]])
    observations = np.array([[[0.1], [-0.2], [0.15]], [[0.0], [0.12], [-0.04]]])
    call = make_pruned_srukf_value_and_score(_model, tf.TensorSpec([2, 3], tf.float64), tf.TensorSpec([2, 3, 1], tf.float64), jit_compile=False)
    value, score, diagnostics = call(parameters, observations)
    assert np.all(diagnostics["valid"])
    for row in range(2):
        reference = _covariance_reference(parameters[row], observations[row, :, 0])[0]
        np.testing.assert_allclose(value[row], reference, atol=1e-11)
        for step in [1e-4, 1e-5]:
            for parameter in range(3):
                shift = np.zeros(3)
                shift[parameter] = step
                plus = _covariance_reference(parameters[row] + shift, observations[row, :, 0])[0]
                minus = _covariance_reference(parameters[row] - shift, observations[row, :, 0])[0]
                np.testing.assert_allclose(score[row, parameter], (plus - minus) / (2 * step), rtol=2e-6, atol=2e-7)
    call(parameters + 0.01, observations)
    assert call.experimental_get_tracing_count() == 1
    graph = call.get_concrete_function().graph.as_graph_def()
    operations = {node.op for node in graph.node}
    operations.update(node.op for function in graph.library.function for node in function.node_def)
    assert not operations.intersection({"PyFunc", "EagerPyFunc", "Svd", "SelfAdjointEigV2", "Cholesky", "Qr"})


def test_empty_observations_preserve_initial_cross_covariance():
    base = _model([[0.6, 0.2, -2.0]])
    model = replace(base, initial_factor=tf.constant([[[0.2, 0.0], [0.1, 0.05]]], tf.float64))
    result = tf_pruned_srukf_filter(tf.zeros([1, 0, 1], tf.float64), model, jit_compile=False)
    np.testing.assert_array_equal(result.filtered_factor, model.initial_factor)
    np.testing.assert_array_equal(result.filtered_mean, model.initial_mean)
    np.testing.assert_array_equal(result.log_likelihood, [0.0])


def test_nonzero_initial_cross_covariance_is_conditioned_jointly():
    base = _model([[0.6, 0.2, -2.0]])
    initial_factor = tf.constant([[[0.2, 0.0], [0.1, 0.05]]], tf.float64)
    model = replace(
        base,
        initial_factor=initial_factor,
        transition_matrix=tf.constant([[[0.7]]], tf.float64),
        second_order_constant=tf.zeros_like(base.second_order_constant),
        transition_hessian=tf.zeros_like(base.transition_hessian),
        observation_hessian=tf.zeros_like(base.observation_hessian),
    )
    observed = 0.13
    result = tf_pruned_srukf_filter(
        tf.constant([[[observed]]], tf.float64), model, jit_compile=False
    )

    transition = np.diag([0.7, 0.7])
    initial_covariance = np.array([[0.04, 0.02], [0.02, 0.0125]])
    predicted_mean = transition @ np.array([0.1, 0.0])
    predicted_covariance = transition @ initial_covariance @ transition.T
    predicted_covariance[0, 0] += 0.15**2
    observation_matrix = np.array([1.0, 1.0])
    innovation_variance = (
        observation_matrix @ predicted_covariance @ observation_matrix
        + np.exp(-4.0)
    )
    innovation = observed - (0.02 + observation_matrix @ predicted_mean)
    gain = predicted_covariance @ observation_matrix / innovation_variance
    expected_mean = predicted_mean + gain * innovation
    expected_covariance = predicted_covariance - np.outer(gain, gain) * innovation_variance
    expected_value = -0.5 * (
        np.log(2.0 * np.pi * innovation_variance) + innovation**2 / innovation_variance
    )

    np.testing.assert_allclose(result.filtered_mean, [expected_mean], atol=1e-12)
    np.testing.assert_allclose(result.log_likelihood, [expected_value], atol=1e-12)
    factor = result.filtered_factor.numpy()[0]
    np.testing.assert_allclose(factor @ factor.T, expected_covariance, atol=1e-12)


@pytest.mark.parametrize("invalid", ["observation", "coefficient", "singular_innovation"])
def test_invalid_rows_fail_closed(invalid):
    model = _model([[0.6, 0.2, -2.0]])
    observations = tf.zeros([1, 2, 1], tf.float64)
    if invalid == "observation":
        observations = tf.fill([1, 2, 1], tf.constant(float("nan"), tf.float64))
    elif invalid == "coefficient":
        model = replace(model, transition_matrix=tf.fill([1, 1, 1], tf.constant(float("inf"), tf.float64)))
    else:
        model = replace(model, observation_matrix=tf.zeros_like(model.observation_matrix), observation_hessian=tf.zeros_like(model.observation_hessian), observation_factor=tf.zeros_like(model.observation_factor))
    result = tf_pruned_srukf_filter(observations, model, jit_compile=False)
    assert result.diagnostics["valid"].numpy().tolist() == [False]
    assert np.isneginf(result.log_likelihood.numpy()[0])


@pytest.mark.parametrize("field,shape", [("initial_mean", [1, 1]), ("initial_factor", [1, 2, 3]), ("transition_hessian", [1, 1, 1, 1]), ("observation_hessian", [1, 1, 2, 2])])
def test_shape_errors_are_explicit(field, shape):
    with pytest.raises(ValueError, match=field):
        replace(_model([[0.6, 0.2, -2.0]]), **{field: tf.zeros(shape, tf.float64)})


@pytest.mark.skipif(os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") != "visible", reason="requires trusted GPU launch")
def test_gpu_xla_value_score_and_invalid_status():
    assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") == "true"
    physical = tf.config.list_physical_devices("GPU")
    assert len(physical) == 1
    tf.config.experimental.set_memory_growth(physical[0], True)
    parameters = tf.constant([[0.65, 0.3, -2.0], [0.4, 0.2, -1.7]], tf.float64)
    panel = tf.constant([[[0.1], [-0.1]], [[0.0], [0.2]]], tf.float64)
    reference = make_pruned_srukf_value_and_score(_model, tf.TensorSpec([2, 3], tf.float64), tf.TensorSpec([2, 2, 1], tf.float64), jit_compile=False)
    expected_value, expected_score, _ = reference(parameters, panel)
    compiled = make_pruned_srukf_value_and_score(_model, tf.TensorSpec([2, 3], tf.float64), tf.TensorSpec([2, 2, 1], tf.float64))
    with tf.device("/GPU:0"):
        value, score, diagnostics = compiled(parameters, panel)
        invalid_panel = tf.tensor_scatter_nd_update(panel, [[1, 0, 0]], [tf.constant(float("nan"), tf.float64)])
        invalid_value, _, invalid_diagnostics = compiled(parameters, invalid_panel)
    assert "GPU:0" in value.device
    np.testing.assert_allclose(value, expected_value, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(score, expected_score, atol=1e-10, rtol=1e-10)
    assert diagnostics["valid"].numpy().tolist() == [True, True]
    assert invalid_diagnostics["valid"].numpy().tolist() == [True, False]
    assert np.isneginf(invalid_value.numpy()[1])
