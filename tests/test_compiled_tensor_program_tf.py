"""Independent pullback checks for enclosing native TensorFlow control flow."""

from contextlib import nullcontext

import pytest
import tensorflow as tf

from bayesfilter.ops.compiled_tensor_program_tf import tensor_program


@pytest.mark.parametrize("jit_compile", [False, True])
@pytest.mark.parametrize("nesting", ["none", "cond", "while_cond"])
@pytest.mark.parametrize("shape", [(), (256,)])
def test_value_only_preparation_defers_pullback_with_backward_only_capture(jit_compile, nesting, shape):
    coefficient = tf.fill(shape, tf.constant(3., tf.float64))
    pullback_traces = []

    @tf.custom_gradient
    def cubic(x):
        def pullback(incoming):
            pullback_traces.append(True)
            return incoming * coefficient * x**2

        return x**3, pullback

    signature = [tf.TensorSpec(shape, tf.float64)]

    def enclosing(x):
        local = tensor_program(cubic, signature, False)

        def branch():
            return tf.cond(tf.reduce_all(x > 0.), lambda: local.python_function(x), lambda: x**3)

        if nesting == "cond":
            return branch()
        _, output = tf.while_loop(lambda index, _: index < 2,
            lambda index, total: (index + 1, total + branch()),
            (tf.constant(0), tf.zeros_like(x)), maximum_iterations=2)
        return output * .5

    with tf.init_scope():
        program = tensor_program(cubic if nesting == "none" else enclosing, signature, jit_compile)
    value = tf.fill(shape, tf.constant(2., tf.float64))
    tf.debugging.assert_equal(program(value), tf.fill(shape, tf.constant(8., tf.float64)))
    assert not pullback_traces
    for _ in range(2):
        with tf.GradientTape() as tape:
            tape.watch(value)
            output = program(value)
        tf.debugging.assert_near(tape.gradient(output, value), tf.fill(shape, tf.constant(12., tf.float64)),
                                 atol=1e-12, rtol=1e-12)
    assert len(pullback_traces) == 1
    assert program.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("jit_compile", [False, True])
def test_program_prepared_inside_initialization_retains_later_derivatives(jit_compile):
    # Marginal/metadata preparation may lift cold program construction out of
    # tracing. Pausing that outer tape must not freeze a zero pullback forever.
    with tf.init_scope():
        program = tensor_program(lambda x: x**3, [tf.TensorSpec([], tf.float64)], jit_compile)
    value = tf.constant(2., tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(value)
        output = program(value)
    tf.debugging.assert_near(tape.gradient(output, value), tf.constant(12., tf.float64),
                             atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("jit_compile", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("prepared_in_init_scope", [False, True])
def test_complete_pullback_keeps_inputs_tensor_captures_and_resources(
        jit_compile, nested, prepared_in_init_scope):
    dtype = tf.float64
    query = tf.constant([.2, -.4], dtype)
    coefficient = tf.constant([1.3, .7], dtype)
    linear = tf.Variable(.6, dtype=dtype)
    signature = [tf.TensorSpec([2], dtype)]

    def finite_program(x):
        def step(index, value):
            return index + 1, value + tf.cast(index + 1, dtype) * coefficient * x**2 + linear * x

        _, value = tf.while_loop(lambda index, _: index < 3, step,
                                (tf.constant(0), tf.zeros_like(x)), maximum_iterations=3)
        # Unused numeric outputs receive a zero cotangent; status leaves have
        # no derivative. Neither may erase input, capture or variable VJPs.
        return value, 2. * value, tf.constant(3), tf.reduce_all(tf.math.is_finite(value)), 3. * value

    def enclosing(x):
        def branch(scale):
            local = tensor_program(finite_program, signature, False)
            return lambda: scale * local.python_function(x)[0]

        return tf.map_fn(lambda index: tf.switch_case(index, (branch(1.), branch(2.))),
                         tf.range(2), fn_output_signature=tf.TensorSpec([2], dtype),
                         parallel_iterations=1)

    with tf.init_scope() if prepared_in_init_scope else nullcontext():
        program = tensor_program(enclosing if nested else finite_program, signature, jit_compile)
    for value in (.6, .9):
        linear.assign(value)
        with tf.GradientTape() as tape:
            tape.watch((query, coefficient))
            values = program(query)
            loss = tf.reduce_sum(values) if nested else tf.reduce_sum(values[0] + values[1])
        gradients = tape.gradient(loss, (query, coefficient, linear))
        expected_value = 6. * coefficient * query**2 + 3. * linear * query
        tf.debugging.assert_near(values[0], expected_value, atol=1e-12, rtol=1e-12)
        tf.debugging.assert_near(values[1], 2. * expected_value, atol=1e-12, rtol=1e-12)
        if not nested:
            tf.debugging.assert_equal(values[3], True)
        expected = (3. * (12. * coefficient * query + 3. * linear),
                    18. * query**2, 9. * tf.reduce_sum(query))
        for actual, reference in zip(gradients, expected, strict=True):
            assert actual is not None
            tf.debugging.assert_near(actual, reference, atol=1e-12, rtol=1e-12)
    assert program.experimental_get_tracing_count() == 1
