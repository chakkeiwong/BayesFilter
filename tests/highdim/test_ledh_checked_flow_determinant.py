"""Independent numerical and failure checks for the shared flow determinant."""

import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_numerical_safety_tf import checked_log_abs_det_tangent


def compiled_helper(shape, dtype):
    return tf.function(checked_log_abs_det_tangent, jit_compile=True, autograph=False,
                       input_signature=[tf.TensorSpec(shape, dtype)] * 2)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("order", [1, 2, 4, 18])
def test_batched_xla_matches_independent_determinant_and_solve(dtype, order):
    shape = [2, 3, order, order]
    matrix = tf.random.stateless_normal(shape, [712, order], dtype=dtype) * .05
    matrix += tf.eye(order, batch_shape=[2, 3], dtype=dtype)
    matrix = tf.reverse(matrix, axis=[-2])
    tangent = tf.random.stateless_normal(shape, [912, order], dtype=dtype)
    program = compiled_helper(shape, dtype)
    log_det, derivative, value_valid, force_valid = program(matrix, tangent)
    absolute, relative = (2e-10, 2e-9) if dtype == tf.float64 else (2e-5, 2e-4)
    with tf.device("/CPU:0"):
        reference_log = tf.linalg.slogdet(matrix)[1]
        reference_derivative = tf.linalg.trace(tf.linalg.solve(matrix, tangent))
    tf.debugging.assert_equal(value_valid & force_valid, tf.ones([2, 3], tf.bool))
    tf.debugging.assert_near(log_det, reference_log, atol=absolute, rtol=relative)
    tf.debugging.assert_near(derivative, reference_derivative, atol=absolute, rtol=relative)
    program(matrix, tf.zeros_like(tangent))
    assert program.experimental_get_tracing_count() == 1


def test_directional_derivative_matches_central_difference():
    matrix = tf.constant([[0., 2., 1.], [3., 1., 2.], [1., 4., 3.]], tf.float64)
    tangent = tf.constant([[.2, -.1, .3], [.1, .3, -.2], [.4, -.1, .2]], tf.float64)
    program = compiled_helper([3, 3], tf.float64)
    derivative = program(matrix, tangent)[1]
    step = tf.constant(1e-5, tf.float64)
    difference = (program(matrix + step*tangent, tangent)[0]
                  - program(matrix - step*tangent, tangent)[0]) / (2*step)
    tf.debugging.assert_near(derivative, difference, atol=2e-10, rtol=2e-9)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_invalid_inputs_reject_without_poisoning_healthy_batch(dtype):
    matrix = tf.constant([[[1., 0.], [0., 1.]], [[0., 1.], [0., 2.]],
                          [[1., 2.], [2., 4.]], [[float('nan'), 0.], [0., 1.]],
                          [[1., 0.], [float('inf'), 1.]], [[-1., 0.], [0., 2.]]], dtype)
    tangent = tf.eye(2, batch_shape=[6], dtype=dtype)
    result = compiled_helper([6, 2, 2], dtype)(matrix, tangent)
    expected = tf.constant([True, False, False, False, False, True])
    tf.debugging.assert_equal(result[2], expected)
    tf.debugging.assert_equal(result[3], expected)
    tf.debugging.assert_all_finite(result[0], "determinant placeholders must be finite")
    tf.debugging.assert_all_finite(result[1], "derivative placeholders must be finite")


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_nonfinite_tangent_rejects_force_only(dtype):
    matrix = tf.eye(2, batch_shape=[2], dtype=dtype)
    tangent = tf.constant([[[float('nan'), 0.], [0., 1.]],
                           [[1., float('inf')], [0., 1.]]], dtype)
    log_det, derivative, value_valid, force_valid = compiled_helper([2, 2, 2], dtype)(matrix, tangent)
    tf.debugging.assert_equal(value_valid, [True, True])
    tf.debugging.assert_equal(force_valid, [False, False])
    tf.debugging.assert_equal(log_det, tf.zeros([2], dtype))
    tf.debugging.assert_equal(derivative, tf.zeros([2], dtype))


@pytest.mark.parametrize("dtype,small", [(tf.float32, 1e-6), (tf.float64, 1e-12)])
def test_ill_conditioned_nonzero_pivot_is_not_clipped(dtype, small):
    matrix = tf.linalg.diag(tf.constant([1., small, -2.], dtype))
    log_det, derivative, value_valid, force_valid = compiled_helper([3, 3], dtype)(matrix, matrix)
    tf.debugging.assert_equal(value_valid & force_valid, True)
    tf.debugging.assert_near(log_det, tf.math.log(tf.constant(2*small, dtype)))
    tf.debugging.assert_near(derivative, tf.constant(3., dtype))


def test_flow_consumer_uses_shared_checked_factorization(monkeypatch):
    from bayesfilter.highdim import ledh_canonical_score_tf as shared
    from bayesfilter.highdim.ledh_canonical_batch_fused_tf import _single_cloud_model
    from bayesfilter.testing.marginal_weight_campaign import specification

    calls = []

    def recording(matrix, tangent):
        calls.append(matrix.shape)
        return checked_log_abs_det_tangent(matrix, tangent)

    monkeypatch.setattr(shared, "checked_log_abs_det_tangent", recording)
    spec = specification("m13-t1", tf.float64)
    theta = tf.constant([-.3, .2], tf.float64)
    model = _single_cloud_model(spec["model"], theta, tf.constant([.3, -.2], tf.float64))
    anchors = tf.constant([[1.2, .7, .18, -.06], [1., .8, .15, .03]], tf.float64)
    covariance = .04 * tf.eye(4, batch_shape=[2], dtype=tf.float64)
    observation_covariance = model.observation_covariance
    outputs = shared._flow_substeps_with_tangent(
        model, anchors, .01 * anchors, anchors, .02 * anchors,
        covariance, .03 * covariance, spec["fixture"].observations[0],
        observation_covariance, None, tf.linalg.inv(observation_covariance), None,
        substeps=2, eye=tf.eye(4, dtype=tf.float64), return_affine=True)
    assert calls
    assert all(shape[-2:] == (2, 2) for shape in calls)
    tf.debugging.assert_equal(tf.reduce_all(outputs[4] & outputs[5]), True)
    matrix, tangent = outputs[-2:]
    tf.debugging.assert_near(outputs[2], tf.linalg.slogdet(matrix)[1], atol=2e-12, rtol=2e-12)
    tf.debugging.assert_near(outputs[3], tf.linalg.trace(tf.linalg.solve(matrix, tangent)),
                             atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("observation_dimension", [1, 2, 3])
def test_flow_dimension_selection_preserves_total_tangent(monkeypatch, dtype, observation_dimension):
    from bayesfilter.highdim import ledh_canonical_score_tf as shared

    calls = []

    def recording(matrix, tangent):
        calls.append(matrix.shape[-1])
        return checked_log_abs_det_tangent(matrix, tangent)

    monkeypatch.setattr(shared, "checked_log_abs_det_tangent", recording)
    observation_matrix = tf.random.stateless_normal([observation_dimension, 2], [641, 22], dtype=dtype)
    observation_matrix_tangent = .03 * observation_matrix
    observation_covariance = .2 * tf.eye(observation_dimension, dtype=dtype)
    covariance_tangent = .02 * observation_covariance
    model = shared.NonlinearScoreModel(
        transition_mean_fn=lambda states, theta: states,
        transition_mean_tangent_fn=lambda states, tangent, theta: tangent,
        observation_fn=lambda states: tf.matmul(states, observation_matrix, transpose_b=True),
        observation_jacobian_fn=lambda states: tf.broadcast_to(
            observation_matrix, [tf.shape(states)[0], observation_dimension, 2]),
        observation_tangent_fn=lambda states, tangent: (
            tf.matmul(tangent, observation_matrix, transpose_b=True)
            + tf.matmul(states, observation_matrix_tangent, transpose_b=True)),
        observation_jacobian_tangent_fn=lambda states, tangent: tf.broadcast_to(
            observation_matrix_tangent, [tf.shape(states)[0], observation_dimension, 2]),
        process_covariance=tf.eye(2, dtype=dtype),
        observation_covariance=observation_covariance)
    anchors = tf.constant([[1.2, .7], [1., .8]], dtype)
    covariance = tf.broadcast_to(tf.constant([[.1, .02], [.02, .06]], dtype), [2, 2, 2])
    inverse = tf.linalg.inv(observation_covariance)
    outputs = shared._flow_substeps_with_tangent(
        model, anchors, .01 * anchors, anchors, .02 * anchors,
        covariance, .03 * covariance, tf.ones([observation_dimension], dtype),
        observation_covariance, covariance_tangent, inverse,
        -tf.matmul(tf.matmul(inverse, covariance_tangent), inverse),
        substeps=2, eye=tf.eye(2, dtype=dtype), return_affine=True)
    assert calls and set(calls) == {min(2, observation_dimension)}
    tf.debugging.assert_equal(tf.reduce_all(outputs[4] & outputs[5]), True)
    matrix, tangent = outputs[-2:]
    tolerance = 5e-6 if dtype == tf.float32 else 2e-12
    tf.debugging.assert_near(outputs[2], tf.linalg.slogdet(matrix)[1], atol=tolerance, rtol=tolerance)
    tf.debugging.assert_near(outputs[3], tf.linalg.trace(tf.linalg.solve(matrix, tangent)),
                             atol=tolerance, rtol=tolerance)


def linear_conditioning_case(covariance, covariance_tangent, observation_variance):
    from bayesfilter.highdim import ledh_canonical_score_tf as shared

    dtype = covariance.dtype
    observation_matrix = tf.constant([[1., .2, -.3], [.1, -.4, 1.]], dtype)
    observation_covariance = observation_variance * tf.eye(2, dtype=dtype)
    model = shared.NonlinearScoreModel(
        transition_mean_fn=lambda states, theta: states,
        transition_mean_tangent_fn=lambda states, tangent, theta: tangent,
        observation_fn=lambda states: tf.matmul(states, observation_matrix, transpose_b=True),
        observation_jacobian_fn=lambda states: tf.broadcast_to(observation_matrix, [1, 2, 3]),
        observation_tangent_fn=lambda states, tangent: tf.matmul(
            tangent, observation_matrix, transpose_b=True),
        observation_jacobian_tangent_fn=lambda states, tangent: tf.zeros([1, 2, 3], dtype),
        process_covariance=tf.eye(3, dtype=dtype),
        observation_covariance=observation_covariance)
    anchors = tf.constant([[.2, -.3, .5]], dtype)
    inverse = tf.eye(2, dtype=dtype) / observation_variance
    return shared._flow_substeps_with_tangent(
        model, anchors, .01 * anchors, anchors, .02 * anchors,
        covariance, covariance_tangent, tf.constant([.1, .2], dtype),
        observation_covariance, .03 * observation_covariance, inverse, -.03 * inverse,
        substeps=3, eye=tf.eye(3, dtype=dtype), return_affine=True)


@pytest.mark.parametrize("condition", [1., 1e6, 1e12])
@pytest.mark.parametrize("observation_variance", [1., 1e-8])
def test_flow_determinant_matches_full_affine_map_across_conditioning(
        condition, observation_variance):
    dtype = tf.float64
    rotation = tf.linalg.qr(tf.constant([[1., 2., 3.], [3., 1., 2.], [2., 3., 1.]], dtype))[0]
    spectrum = tf.linalg.diag(tf.constant([1., condition ** -.5, 1. / condition], dtype))
    covariance = tf.matmul(tf.matmul(rotation, spectrum), rotation, transpose_b=True)[None]
    outputs = linear_conditioning_case(covariance, .02 * covariance, observation_variance)
    tf.debugging.assert_equal(outputs[4] & outputs[5], [True])
    matrix, tangent = outputs[-2:]
    tf.debugging.assert_near(outputs[2], tf.linalg.slogdet(matrix)[1], atol=1e-8, rtol=1e-8)
    tf.debugging.assert_near(outputs[3], tf.linalg.trace(tf.linalg.solve(matrix, tangent)),
                             atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("failure", ["nonfinite_value", "invalid_innovation", "nonfinite_force"])
def test_flow_determinant_preserves_failure_classification(failure):
    covariance = tf.eye(3, batch_shape=[1], dtype=tf.float64)
    covariance_tangent = .02 * covariance
    if failure == "nonfinite_value":
        covariance *= float("nan")
    elif failure == "invalid_innovation":
        covariance *= -100.
    else:
        covariance_tangent *= float("nan")
    outputs = linear_conditioning_case(covariance, covariance_tangent, 1.)
    tf.debugging.assert_equal(outputs[4], [failure == "nonfinite_force"])
    tf.debugging.assert_equal(outputs[5], [False])
    for output in outputs[:4]:
        tf.debugging.assert_all_finite(output, "rejected flow must use finite placeholders")
