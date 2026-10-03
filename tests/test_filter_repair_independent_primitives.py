"""Independent-reference derivatives and graph lifetime for shape-only solves."""

import gc
import weakref

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.sequential_score_fit_tf import _score_lstsq_program

D = tf.float64


@pytest.mark.parametrize('jit', [False, True])
def test_nested_cod_keeps_full_pullback_without_retaining_parent(jit):
    # First construct this signature from inside a parent graph. It must not
    # keep that graph alive through the custom-gradient registry.
    primitive = []

    @tf.function(input_signature=[tf.TensorSpec([9, 4], D), tf.TensorSpec([9, 1], D)],
                 jit_compile=jit, autograph=False)
    def evaluate(matrix, rhs):
        program = _score_lstsq_program(9, 4, jit)
        primitive.append(program)
        with tf.GradientTape() as tape:
            tape.watch((matrix, rhs))
            solution = program(matrix, rhs)
            objective = tf.reduce_sum(solution ** 2)
        return solution, tape.gradient(objective, (matrix, rhs))

    matrix = tf.constant([[2., .1, .2, 0.], [.1, 3., 0., .2], [.2, 0., 4., .1],
        [0., .2, .1, 5.], [.2, .1, -.1, .2], [-.1, .2, .1, .3],
        [.2, -.1, .3, .1], [.1, .2, .3, -.2], [.3, .1, -.2, .2]], D)
    rhs = tf.reshape(tf.linspace(tf.constant(-.3, D), .5, 9), [9, 1])
    for multiplier in (1., 1.1):
        point = matrix * multiplier
        with tf.GradientTape() as tape:
            tape.watch((point, rhs))
            # Well-conditioned independent normal-equation reference only.
            expected = tf.linalg.lstsq(point, rhs, fast=True)
            objective = tf.reduce_sum(expected ** 2)
        expected_gradients = tape.gradient(objective, (point, rhs))
        actual, gradients = evaluate(point, rhs)
        for left, right in zip(tf.nest.flatten((actual, gradients)),
                               tf.nest.flatten((expected, expected_gradients)), strict=True):
            np.testing.assert_allclose(left, right, atol=1e-11, rtol=1e-11)
    assert len(primitive) == 1
    assert not primitive[0].get_concrete_function().captured_inputs
    assert primitive[0].experimental_get_tracing_count() == 1
    assert evaluate.experimental_get_tracing_count() == 1
    ref = weakref.ref(evaluate.get_concrete_function().graph)
    del evaluate
    gc.collect()
    assert ref() is None
