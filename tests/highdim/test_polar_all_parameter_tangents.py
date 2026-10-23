"""Independent value/finite-difference diagnostics for the shared polar geometry."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_range_bearing_target_tf import polar_value_tangents


def reference_value(x, jacobian):
    """Direct regularized polar formulas, without importing tangent callbacks."""
    a, b = x[:, 0], x[:, 1]
    square = a*a+b*b
    radius = tf.sqrt(square+tf.constant(1e-12, x.dtype))
    if not jacobian:
        return tf.stack([radius, tf.atan2(b, a)], -1)
    zero = tf.zeros_like(a)
    return tf.stack([tf.stack([a/radius, b/radius, zero, zero], -1),
                     tf.stack([-b/square, a/square, zero, zero], -1)], -2)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("jacobian", [False, True])
def test_all_directions_against_independent_finite_differences(dtype, jacobian):
    # Include the regularization scale as well as ordinary ranges. Directions
    # are small relative to each radius, so the FD ladder stays on one chart.
    x = tf.constant([[3., 4., .7, -.3], [.2, -.4, 1., 2.],
                     [1e-6, -2e-6, -1., .4]], dtype)
    scale = tf.sqrt(tf.reduce_sum(x[:, :2]**2, -1))
    dx = tf.random.stateless_normal([3, 3, 4], [261021, 91], dtype=dtype)
    dx *= tf.concat([.1*scale[:, None]*tf.ones([3, 2], dtype),
                     tf.ones([3, 2], dtype)], -1)[None]

    @tf.function(input_signature=[tf.TensorSpec(x.shape, dtype),
                                  tf.TensorSpec(dx.shape, dtype)],
                 jit_compile=True, autograph=False)
    def evaluate(x, dx):
        return polar_value_tangents(x, dx, jacobian=jacobian)

    value, tangent = evaluate(x, dx)
    xx, dd = tf.cast(x, tf.float64), tf.cast(dx, tf.float64)
    tolerance = 4e-5 if dtype == tf.float32 else 1e-8
    tf.debugging.assert_near(tf.cast(value, tf.float64), reference_value(xx, jacobian),
                             atol=tolerance, rtol=tolerance)
    for direction in range(3):
        for step in (1e-3, 3e-4, 1e-4):
            finite_difference = (reference_value(xx+step*dd[direction], jacobian)
                                 -reference_value(xx-step*dd[direction], jacobian))/(2*step)
            half_step = (reference_value(xx+.5*step*dd[direction], jacobian)
                         -reference_value(xx-.5*step*dd[direction], jacobian))/step
            # Remove the O(h^2) truncation error rather than loosening the
            # derivative tolerance near the regularized origin.
            extrapolated = (4*half_step-finite_difference)/3
            tf.debugging.assert_near(tf.cast(tangent[direction], tf.float64),
                extrapolated, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("jacobian", [False, True])
def test_origin_and_nonfinite_rows_are_flagged_in_every_direction(jacobian):
    x = tf.constant([[0., 0., 0., 0.], [1., 2., float("inf"), 0.],
                     [2., 3., 0., 0.]], tf.float32)
    dx = tf.ones([3, 3, 4], tf.float32)
    value, tangent = polar_value_tangents(x, dx, jacobian=jacobian)
    tf.debugging.assert_equal(tf.reduce_all(tf.math.is_nan(value[:2])), True)
    tf.debugging.assert_equal(tf.reduce_all(tf.math.is_nan(tangent[:, :2])), True)
    tf.debugging.assert_equal(tf.reduce_all(tf.math.is_finite(value[2:])), True)
    tf.debugging.assert_equal(tf.reduce_all(tf.math.is_finite(tangent[:, 2:])), True)
