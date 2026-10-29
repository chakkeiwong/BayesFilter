"""Independent diagnostics for captured-input VJPs and compiled conditionals."""

import numpy as np
import pytest
import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.native_execution_tf import (
    compile_recomputed_diagnostic, conditional_step, scan_steps,
)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("captured_variable", [False, True])
def test_callback_captures_remain_connected_through_conditional_loop(jit, captured_variable):
    scale = tf.Variable(0.7, dtype=tf.float64)
    with tf.GradientTape() as outer:
        coefficient = scale if captured_variable else scale * 0.9

        def iterate(x, date):
            _, value = tf.while_loop(lambda k, _: k < 3,
                lambda k, value: (k + 1, value * coefficient), (tf.constant(0), x),
                maximum_iterations=3)
            return value, {"valid": tf.constant(True), "date": date}

        step, _ = conditional_step(iterate, [tf.TensorSpec([], tf.float64),
            tf.TensorSpec([], tf.int32)], jit_compile=jit,
            fallback=lambda specs, x, date: (x, {"valid": tf.constant(True), "date": date}))

        def evaluate(x):
            def advance(date, state):
                value, _ = step(date == 1, state, date)
                return value, value
            return scan_steps(advance, x, 3)

        call = compile_recomputed_diagnostic(evaluate, [tf.TensorSpec([], tf.float64)], jit_compile=jit)
        final, history = call(tf.constant(2., tf.float64))
    derivative = outer.gradient(final, scale)
    expected_scale = .7 if captured_variable else .63
    np.testing.assert_allclose(history, [2., 2. * expected_scale**3, 2. * expected_scale**3], rtol=1e-12)
    assert derivative is not None
    np.testing.assert_allclose(derivative, 6. * expected_scale**2 * (1. if captured_variable else .9), rtol=1e-12)


@pytest.mark.parametrize("jit", [False, True])
def test_mixed_captures_both_branches_and_variable_updates(jit):
    """Cached VJPs must read current variables and retain each capture edge."""
    scale = tf.Variable(.7, dtype=tf.float64)
    offset = tf.Variable(.2, dtype=tf.float64)
    tensor = tf.constant(.3, tf.float64)
    signature = [tf.TensorSpec([], tf.float64)]
    choose, _ = conditional_step(lambda x: x * scale + tensor, signature,
        fallback=lambda specs, x: x**2 + offset * tensor, jit_compile=jit)
    for predicate, new_scale, new_offset in ((True, .7, .2), (False, 1.2, .4), (True, 1.4, .5)):
        scale.assign(new_scale)
        offset.assign(new_offset)
        x = tf.constant(2., tf.float64)
        with tf.GradientTape() as tape:
            tape.watch((x, tensor))
            result = choose(tf.constant(predicate), x)
        dx, dt, ds, do = tape.gradient(result, (x, tensor, scale, offset),
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        expected = (2. * new_scale + .3, new_scale, 1., 2., 0.) if predicate else (
            4. + new_offset * .3, 4., new_offset, 0., .3)
        np.testing.assert_allclose([result, dx, dt, ds, do], expected, rtol=1e-12, atol=1e-12)
