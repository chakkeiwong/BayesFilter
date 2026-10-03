"""Actual public endpoints retain stable, bounded XLA input signatures."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.ops.fixed_signature_tf import fixed_signature_function


def test_numeric_scalar_values_do_not_retrace_and_cache_is_bounded():
    @fixed_signature_function(static_parameters=("negate",), max_specializations=2)
    def operation(x, offset=0.0, negate=False):
        return (x + offset) * (-1.0 if negate else 1.0)

    first = operation.get_concrete_function(tf.TensorSpec([2], tf.float64))
    assert first.function_def.attr["_XlaMustCompile"].b
    for value in (0.0, 1.0, 2.0):
        np.testing.assert_allclose(
            operation(tf.ones([2], tf.float64), offset=value), 1.0 + value
        )
        assert (
            operation.get_concrete_function(tf.ones([2], tf.float64), offset=value)
            is first
        )
    assert operation.experimental_get_tracing_count() == 1
    np.testing.assert_allclose(operation(tf.ones([2], tf.float64), negate=True), -1.0)
    operation(tf.ones([3], tf.float64))
    assert operation.specialization_cache_info().currsize == 2
    with pytest.raises(TypeError, match="static bool"):
        operation(tf.ones([2], tf.float64), negate=tf.constant(True))


def test_public_correlated_filter_signature():
    from bayesfilter.linear.correlated_kalman_tf import tf_correlated_kalman_filter
    from tests.test_linear_correlated_kalman_tf import _fixture

    inputs = {name: tf.constant(value, tf.float64) for name, value in _fixture().items()}
    program = tf_correlated_kalman_filter.get_concrete_function(**inputs)
    actual = tf_correlated_kalman_filter(**inputs)
    expected = tf_correlated_kalman_filter.python_function(**inputs)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(left, right, atol=1e-10, rtol=1e-10)
    assert tf_correlated_kalman_filter.get_concrete_function(**inputs) is program
    assert program.function_def.attr["_XlaMustCompile"].b


def test_public_float64_filter_preserves_python_list_precision():
    from bayesfilter.linear.correlated_kalman_tf import tf_correlated_kalman_filter
    from tests.test_linear_correlated_kalman_tf import _fixture

    inputs = {name: value.tolist() for name, value in _fixture().items()}
    expected = tf_correlated_kalman_filter.python_function(**inputs)
    for actual, reference in zip(tf_correlated_kalman_filter(**inputs), expected, strict=True):
        np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=1e-12)


def test_dtype_anchor_list_controls_and_enclosing_variable_gradient():
    @fixed_signature_function(dtype_like="x", tensor_dtypes={"indices": tf.int32})
    def operation(x, indices, offset):
        return tf.reduce_sum(tf.gather(x + offset, indices) ** 2)

    @tf.function(input_signature=[tf.TensorSpec([2], tf.float64)], jit_compile=True)
    def enclosing(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            value = operation(x, [0, 1], [0.123456789123, 0.0])
        return value, tf.convert_to_tensor(tape.gradient(value, x))

    variable = tf.Variable([1.0, 2.0], dtype=tf.float64)
    expected = 2.0 * (variable.numpy() + [0.123456789123, 0.0])
    np.testing.assert_allclose(enclosing(variable)[1], expected, atol=1e-13, rtol=1e-13)
    variable.assign([3.0, 4.0])
    expected = 2.0 * (variable.numpy() + [0.123456789123, 0.0])
    np.testing.assert_allclose(enclosing(variable)[1], expected, atol=1e-13, rtol=1e-13)
    with tf.GradientTape() as tape:
        result = operation(variable, [0, 1], [0.123456789123, 0.0])
    np.testing.assert_allclose(tf.convert_to_tensor(tape.gradient(result, variable)), expected, atol=1e-13, rtol=1e-13)
    assert operation.specialization_cache_info().currsize == 1


def test_keyword_arguments_bind_into_enclosing_xla_without_input_capture():
    @fixed_signature_function()
    def square(x):
        return x * x

    @tf.function(input_signature=[tf.TensorSpec([2], tf.float64)], jit_compile=True)
    def enclosing(x):
        return square(x=x) + square(x=x + 1.0)

    np.testing.assert_array_equal(
        enclosing(tf.constant([1.0, 2.0], tf.float64)), [5.0, 13.0]
    )
    np.testing.assert_array_equal(
        enclosing(tf.constant([3.0, 4.0], tf.float64)), [25.0, 41.0]
    )
    assert square.specialization_cache_info().currsize == 1
