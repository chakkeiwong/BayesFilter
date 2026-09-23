"""Magnitude-normalized real SVD with binary64 XLA convergence.

This evaluates A=U diag(s) V^T without changing rank cutoffs or singular values.
The real thin-SVD pullback follows TensorFlow 2.19 linalg_grad.py::_SvdGrad,
including its existing broadened reciprocal convention at repeated singular
values. Analytical filtering scores remain on their existing QR route.
"""

import sys
from functools import lru_cache

import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_svd

from bayesfilter.inference.program_cache_scope import independent_trace_scope
from bayesfilter.ops.compiled_tensor_program_tf import in_xla_context


def _safe_reciprocal(value):
    # TensorFlow's existing _SafeReciprocal, not a new regularization policy.
    return value * tf.math.reciprocal(tf.square(value) + tf.constant(1e-20, value.dtype))


@tf.custom_gradient
def _xla_thin_svd(matrix):
    rows, columns = matrix.shape[-2:]
    extent = min(rows, columns)
    magnitude = tf.reduce_max(tf.abs(matrix), axis=[-2, -1], keepdims=True)
    normalized = matrix / tf.where(magnitude > 0., magnitude, tf.ones_like(magnitude))
    decomposition = xla_svd(normalized, max_iter=100,
        epsilon=sys.float_info.epsilon, precision_config='')
    batch_shape = matrix.shape[:-2]
    singular = tf.ensure_shape(decomposition.s, batch_shape.concatenate([extent])) * magnitude[..., 0]
    left = tf.ensure_shape(decomposition.u, batch_shape.concatenate([rows, rows]))[..., :extent]
    right = tf.ensure_shape(decomposition.v, batch_shape.concatenate([columns, columns]))[..., :extent]

    def pullback(ds, du, dv):
        ds = tf.zeros_like(singular) if ds is None else ds
        du = tf.zeros_like(left) if du is None else du
        dv = tf.zeros_like(right) if dv is None else dv
        # TF's real thin-SVD convention computes the wide case and transposes
        # the tall case. This preserves its gap and null-space derivatives.
        u, v = (right, left) if rows > columns else (left, right)
        gu, gv = (dv, du) if rows > columns else (du, dv)
        square = tf.square(singular)
        gaps = square[..., None, :] - square[..., :, None]
        inverse = tf.linalg.set_diag(_safe_reciprocal(gaps), tf.zeros_like(singular))
        left_local = inverse * tf.matmul(u, gu, transpose_a=True)
        right_local = inverse * tf.matmul(v, gv, transpose_a=True)
        diagonal = tf.linalg.diag(singular)
        local = (tf.linalg.diag(ds) +
            tf.matmul(left_local + tf.linalg.matrix_transpose(left_local), diagonal) +
            tf.matmul(diagonal, right_local + tf.linalg.matrix_transpose(right_local)))
        gradient = tf.matmul(u, tf.matmul(local, v, transpose_b=True))
        if rows != columns:
            transposed = tf.linalg.matrix_transpose(gv)
            orthogonal = transposed - tf.matmul(tf.matmul(transposed, v), v, transpose_b=True)
            gradient += tf.matmul(u * _safe_reciprocal(singular)[..., None, :], orthogonal)
        return tf.linalg.matrix_transpose(gradient) if rows > columns else gradient

    return (singular, left, right), pullback


@lru_cache(maxsize=64)
def _xla_svd_program(shape):
    # Custom-gradient registry closures must own only this shape-only graph,
    # never a calling endpoint's graph or target/resource dependencies.
    with independent_trace_scope():
        program = tf.function(_xla_thin_svd, autograph=False, jit_compile=True,
            input_signature=[tf.TensorSpec(shape, tf.float64)])
        program.get_concrete_function()
    return program


def accurate_svd(matrix, *, compute_uv=True):
    """Binary64 thin SVD; use the enclosing XLA or explicit graph reference.

    Matrix extents must be static for XLA; batch dimensions are preserved.
    The graph/eager reference keeps TensorFlow's standard supported SVD.
    """
    matrix = tf.convert_to_tensor(matrix, tf.float64)
    if not in_xla_context():
        return tf.linalg.svd(matrix, full_matrices=False, compute_uv=compute_uv)
    if matrix.shape.rank is None or matrix.shape.rank < 2 or None in matrix.shape[-2:]:
        raise ValueError('accurate XLA SVD requires static matrix extents')
    if min(matrix.shape[-2:]) == 0:
        return tf.linalg.svd(matrix, full_matrices=False, compute_uv=compute_uv)
    result = _xla_svd_program(tuple(matrix.shape))(matrix)
    return result if compute_uv else result[0]
