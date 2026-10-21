"""Shared TensorFlow arithmetic for small real matrix factorizations/solves.

Static orders one and two use scalar elimination so GPU XLA can fuse them
into surrounding kernels. Larger/unknown orders and other dtypes retain the
TensorFlow library operations. This changes no ridge, clipping or model rule.
Use safe_cholesky for a covariance with a propagated validity mask.
"""
from __future__ import annotations

import tensorflow as tf


def _small_order(matrix):
    if (matrix.dtype not in (tf.float32, tf.float64)
            or matrix.shape.rank is None or matrix.shape.rank < 2):
        return None
    n = matrix.shape[-1]
    return n if n in (1, 2) and matrix.shape[-2] == n else None


def cholesky(matrix, name=None):
    """Lower Cholesky factor with the same lower-triangle convention as TF.

    A nonpositive/nonfinite pivot returns NaNs. The caller's safe_cholesky
    converts rejected factors to identity and retains false validity. Positive
    pivots are never clipped. Broadcasting is inherited from elementwise ops.
    """
    matrix = tf.convert_to_tensor(matrix)
    n = _small_order(matrix)
    if n is None:
        return tf.linalg.cholesky(matrix, name=name)
    with tf.name_scope(name or "small_cholesky"):
        a = matrix[..., 0, 0]
        first_ok = tf.math.is_finite(a) & (a > 0)
        l00 = tf.sqrt(tf.where(first_ok, a, tf.ones_like(a)))
        if n == 1:
            factor = l00[..., None, None]
            valid = first_ok
        else:
            l10 = matrix[..., 1, 0] / l00
            pivot = matrix[..., 1, 1] - l10*l10
            valid = first_ok & tf.math.is_finite(l10) & tf.math.is_finite(pivot) & (pivot > 0)
            l11 = tf.sqrt(tf.where(valid, pivot, tf.ones_like(pivot)))
            factor = tf.stack([tf.stack([l00, tf.zeros_like(l00)], -1), tf.stack([l10, l11], -1)], -2)
        return tf.where(valid[..., None, None], factor, tf.constant(float("nan"), matrix.dtype))


def triangular_solve(matrix, rhs, lower=True, adjoint=False, name=None):
    """Solve triangular systems, including broadcast batch/direction axes."""
    matrix = tf.convert_to_tensor(matrix)
    rhs = tf.convert_to_tensor(rhs, dtype=matrix.dtype)
    n = _small_order(matrix)
    if n is None:
        return tf.linalg.triangular_solve(matrix, rhs, lower=lower, adjoint=adjoint, name=name)
    if rhs.shape.rank is not None and (rhs.shape.rank < 2 or rhs.shape[-2] not in (None, n)):
        raise ValueError("triangular solve RHS has incompatible matrix dimension")
    with tf.name_scope(name or "small_triangular_solve"):
        if n == 1:
            return rhs / matrix[..., 0, 0, None, None]
        diagonal0, diagonal1 = matrix[..., 0, 0, None], matrix[..., 1, 1, None]
        off_diagonal = matrix[..., 1, 0, None] if lower else matrix[..., 0, 1, None]
        if lower != adjoint:
            x0 = rhs[..., 0, :] / diagonal0
            x1 = (rhs[..., 1, :] - off_diagonal*x0) / diagonal1
        else:
            x1 = rhs[..., 1, :] / diagonal1
            x0 = (rhs[..., 0, :] - off_diagonal*x1) / diagonal0
        return tf.stack([x0, x1], -2)


def cholesky_solve(chol, rhs, name=None):
    """Solve LL' x = rhs using the same shared triangular operations."""
    with tf.name_scope(name or "small_cholesky_solve"):
        return triangular_solve(chol, triangular_solve(chol, rhs), adjoint=True)


def solve(matrix, rhs, adjoint=False, name=None):
    """General solve by partial-pivot elimination at static order one/two.

    Callers retain responsibility for rejecting singular/nonfinite systems,
    exactly as for triangular_solve. No pivot regularization is introduced.
    """
    matrix = tf.convert_to_tensor(matrix)
    rhs = tf.convert_to_tensor(rhs, dtype=matrix.dtype)
    n = _small_order(matrix)
    if n is None:
        return tf.linalg.solve(matrix, rhs, adjoint=adjoint, name=name)
    if rhs.shape.rank is not None and (rhs.shape.rank < 2 or rhs.shape[-2] not in (None, n)):
        raise ValueError("solve RHS has incompatible matrix dimension")
    with tf.name_scope(name or "small_matrix_solve"):
        if n == 1:
            return rhs / matrix[..., 0, 0, None, None]
        if adjoint:
            matrix = tf.linalg.matrix_transpose(matrix)
        a, b, c, d = (matrix[..., i, j, None] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
        swap = tf.abs(c) > tf.abs(a)
        a, b, c, d = tf.where(swap, c, a), tf.where(swap, d, b), tf.where(swap, a, c), tf.where(swap, b, d)
        r0, r1 = tf.where(swap, rhs[..., 1, :], rhs[..., 0, :]), tf.where(swap, rhs[..., 0, :], rhs[..., 1, :])
        multiplier = c/a
        x1 = (r1-multiplier*r0)/(d-multiplier*b)
        x0 = (r0-b*x1)/a
        return tf.stack([x0, x1], -2)
