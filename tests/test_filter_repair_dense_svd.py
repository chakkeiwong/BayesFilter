"""Dense SVD values/pullbacks against independent diagnostic references."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.score_curvature_tf import _singular_value_program

D = tf.float64


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('case', ['partial_cloud', 'repeated', 'near_repeated', 'singular'])
def test_dense_svd_values_and_pullback(dimension, case):
    if case == 'partial_cloud':
        cloud = tf.random.stateless_uniform([33, dimension], [-1729, 2], minval=-.03,
            maxval=.03, dtype=D).numpy()
        matrix = np.linalg.qr(cloud, mode='reduced')[1]
    else:
        matrix = np.eye(dimension)
        if case == 'near_repeated':
            matrix += 1e-8 * np.arange(dimension * dimension).reshape(dimension, dimension)
        elif case == 'singular':
            matrix[-1, -1] = 0.
    value = tf.constant(matrix, D)
    actual_program = _singular_value_program(dimension)

    def make(jit):
        @tf.function(input_signature=[tf.TensorSpec([dimension, dimension], D)],
            autograph=False, jit_compile=jit)
        def evaluate(matrix):
            with tf.GradientTape() as tape:
                tape.watch(matrix)
                singular = actual_program(matrix) if jit else tf.linalg.svd(matrix, compute_uv=False)
                objective = .5 * tf.reduce_sum(tf.square(singular))
            return singular, objective, tape.gradient(objective, matrix)
        return evaluate

    actual = make(True)(value)
    graph = make(False)(value)
    for left, right in zip(actual, graph, strict=True):
        np.testing.assert_allclose(left, right, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual[0], np.linalg.svd(matrix, compute_uv=False), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual[1], .5 * np.sum(matrix * matrix), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual[2], matrix, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('dimension', [3, 5])
def test_dense_condition_pullback_matches_independent_simple_spectrum(dimension):
    rng = np.random.default_rng(1729 + dimension)
    matrix = rng.normal(size=(dimension, dimension)) + 3. * np.eye(dimension)
    left, values, right_transpose = np.linalg.svd(matrix)
    reference = (np.outer(left[:, 0], right_transpose[0]) / values[-1]
        - values[0] * np.outer(left[:, -1], right_transpose[-1]) / values[-1] ** 2)
    program = _singular_value_program(dimension)

    @tf.function(input_signature=program.input_signature, autograph=False, jit_compile=True)
    def pullback(matrix):
        with tf.GradientTape() as tape:
            tape.watch(matrix)
            singular = program(matrix)
            condition = tf.reduce_max(singular) / tf.reduce_min(singular)
        return condition, tape.gradient(condition, matrix)

    condition, gradient = pullback(tf.constant(matrix, D))
    np.testing.assert_allclose(condition, values[0] / values[-1], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(gradient, reference, atol=1e-10, rtol=1e-10)
