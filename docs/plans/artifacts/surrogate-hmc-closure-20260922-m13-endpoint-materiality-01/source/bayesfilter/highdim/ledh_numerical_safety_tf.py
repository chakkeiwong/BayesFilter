"""Fail-closed numerical guards for the canonical LEDH recursion.

Invalid factors are identity placeholders under a false validity mask. They
keep subsequent analytical solves defined; they are never accepted covariances.
These checks establish arithmetic validity, not conditioning or posterior validity.
"""
from __future__ import annotations

import tensorflow as tf

import bayesfilter.highdim.ledh_small_matrix_tf as small_matrix


def all_finite(*values: tf.Tensor) -> tf.Tensor:
    """Scalar finite check without converting a scientific tensor to NumPy."""
    checks = [tf.reduce_all(tf.math.is_finite(value)) for value in values if value is not None]
    return tf.reduce_all(tf.stack(checks)) if checks else tf.constant(True)


def finite_or_zero(value: tf.Tensor) -> tf.Tensor:
    """Arithmetic placeholder; callers must retain the original validity."""
    return tf.where(tf.math.is_finite(value), value, tf.zeros_like(value))


def safe_cholesky(matrix: tf.Tensor, name: str = 'cholesky') -> tuple[tf.Tensor, tf.Tensor]:
    """Factor a full symmetric covariance [..., n, n], preserving batch shape.

    Both triangles must be finite and agree within 8*n machine epsilons after
    scaling by the largest entry. This permits ordinary accumulation roundoff
    without accepting a corrupted unused triangle. No symmetrization, ridge or
    clipping changes an accepted input. A successful factor must be finite and
    have strictly positive diagonal. No condition-number guarantee is implied.

    Returns a boolean of shape [...] and a factor of shape [..., n, n]. An
    invalid input returns an identity factor, safe for downstream solves only
    when its false mask is propagated. Callers must never interpret that factor
    as a repaired covariance.
    """
    matrix = tf.convert_to_tensor(matrix)
    if matrix.dtype not in (tf.float32, tf.float64):
        raise TypeError('safe_cholesky requires float32 or float64')
    if matrix.shape.rank is not None and matrix.shape.rank < 2:
        raise ValueError('covariance must have rank at least two')
    if matrix.shape[-1] is not None and matrix.shape[-2] is not None and matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError('covariance must be square')
    with tf.name_scope(name):
        identity = tf.eye(tf.shape(matrix)[-1], batch_shape=tf.shape(matrix)[:-2], dtype=matrix.dtype)
        finite = tf.reduce_all(tf.math.is_finite(matrix), axis=[-2, -1])
        clean = tf.where(finite[..., None, None], matrix, identity)
        scale = tf.reduce_max(tf.abs(clean), axis=[-2, -1], keepdims=True)
        normalized = clean / tf.where(scale > 0, scale, tf.ones_like(scale))
        epsilon = 2.0 ** (-52 if matrix.dtype == tf.float64 else -23)
        tolerance = tf.cast(8 * tf.shape(matrix)[-1], matrix.dtype) * tf.constant(epsilon, matrix.dtype)
        symmetric = tf.reduce_max(tf.abs(normalized-tf.linalg.matrix_transpose(normalized)), axis=[-2,-1]) <= tolerance
        input_valid = finite & symmetric
        attempt = small_matrix.cholesky(tf.where(input_valid[...,None,None], matrix, identity))
        valid = input_valid & tf.reduce_all(tf.math.is_finite(attempt), axis=[-2,-1])
        valid &= tf.reduce_all(tf.linalg.diag_part(attempt) > 0, axis=-1)
        return valid, tf.where(valid[...,None,None], attempt, identity)


def cholesky_differential(chol: tf.Tensor, d_matrix: tf.Tensor,
                          valid: tf.Tensor | None = None) -> tf.Tensor:
    """Analytical Cholesky differential; suppress tangents of rejected factors.

    A nonfinite tangent on an otherwise valid factor is left detectable by the
    consumer's independent force check. A rejected factor's tangent is zero.
    Leading direction dimensions may broadcast over factor batch dimensions.
    """
    if valid is not None:
        d_matrix = tf.where(valid[...,None,None], d_matrix, tf.zeros_like(d_matrix))
    left = small_matrix.triangular_solve(chol, d_matrix)
    middle = tf.linalg.matrix_transpose(small_matrix.triangular_solve(chol, tf.linalg.matrix_transpose(left)))
    phi = tf.linalg.band_part(middle,-1,0) - 0.5*tf.linalg.diag(tf.linalg.diag_part(middle))
    result = tf.linalg.matmul(chol,phi)
    return result if valid is None else tf.where(valid[...,None,None], result, tf.zeros_like(result))
