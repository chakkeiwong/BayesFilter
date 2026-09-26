"""Uninstalled TensorFlow dot/pullback candidate for execution diagnostics.

No admitted runtime imports this module. Arrays and all numerical derivatives
use TensorFlow; only the diagnostic tests use independent NumPy references.
"""

import tensorflow as tf
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.compiler.xla import xla_data_pb2


def _contract(left, right, left_axis, right_axis):
    dimensions = xla_data_pb2.DotDimensionNumbers(
        lhs_contracting_dimensions=[left_axis],
        rhs_contracting_dimensions=[right_axis])
    precision = xla_data_pb2.PrecisionConfig(operand_precision=[
        xla_data_pb2.PrecisionConfig.HIGHEST,
        xla_data_pb2.PrecisionConfig.HIGHEST])

    @tf.custom_gradient
    def operation(lhs, rhs):
        value = gen_xla_ops.xla_dot_v2(lhs, rhs,
            dimension_numbers=dimensions.SerializeToString(),
            precision_config=precision.SerializeToString(),
            preferred_element_type=lhs.dtype)
        value = tf.ensure_shape(value,
            [lhs.shape[1 - left_axis], rhs.shape[1 - right_axis]])

        def pullback(cotangent):
            # In oriented coordinates C=L R: dL=G R^T, dR=L^T G.
            # Native reductions also have registered higher derivatives and
            # retain FP32 multiplication with global TF32 enabled. Recursive
            # custom raw-dot gradients failed TensorFlow's nested graph scope
            # check (04168); no derivative is detached or replaced by zeros.
            oriented_left = tf.transpose(lhs) if left_axis == 0 else lhs
            oriented_right = tf.transpose(rhs) if right_axis == 1 else rhs
            grad_left = tf.reduce_sum(
                cotangent[:, :, None] * tf.transpose(oriented_right)[None, :, :], axis=1)
            grad_right = tf.reduce_sum(
                tf.transpose(oriented_left)[:, :, None] * cotangent[None, :, :], axis=1)
            if left_axis == 0:
                grad_left = tf.transpose(grad_left)
            if right_axis == 1:
                grad_right = tf.transpose(grad_right)
            return grad_left, grad_right

        return value, pullback

    return operation(left, right)


class HighestPrecisionDotProgram:
    """An explicitly shaped, owned XLA function with differentiable pullback."""

    def __init__(self, left_shape, right_shape, left_axis, right_axis, dtype):
        dtype = tf.as_dtype(dtype)
        if dtype not in (tf.float32, tf.float64):
            raise ValueError("highest-precision dot requires float32 or float64")
        if left_axis not in (0, 1) or right_axis not in (0, 1):
            raise ValueError("matrix contraction axes must be zero or one")
        left_shape, right_shape = tf.TensorShape(left_shape), tf.TensorShape(right_shape)
        if (left_shape.rank != 2 or right_shape.rank != 2
                or not left_shape.is_fully_defined() or not right_shape.is_fully_defined()):
            raise ValueError("matrix contraction requires fixed rank-two shapes")
        if left_shape[left_axis] != right_shape[right_axis]:
            raise ValueError("contracting dimensions must agree")

        def calculate(left, right):
            return _contract(left, right, left_axis, right_axis)

        self.function = tf.function(calculate, input_signature=[
            tf.TensorSpec(left_shape, dtype), tf.TensorSpec(right_shape, dtype)],
            jit_compile=True, autograph=False)

    def __call__(self, left, right):
        return self.function(left, right)


class HighestPrecisionDotFamily:
    """Test-local shape registry; dropping the family drops its owned functions."""

    def __init__(self):
        self.programs = {}

    def __call__(self, left, right, left_axis, right_axis):
        key = (tuple(left.shape), tuple(right.shape), left_axis, right_axis, left.dtype)
        if key not in self.programs:
            self.programs[key] = HighestPrecisionDotProgram(*key)
        return self.programs[key](left, right)
