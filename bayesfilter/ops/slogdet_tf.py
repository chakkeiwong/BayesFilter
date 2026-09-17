"""Signed log determinant using native pivoted elimination for XLA."""

import tensorflow as tf


@tf.custom_gradient
def slogdet_tf(matrix):
    """Real square matrix determinant and its usual inverse-transpose VJP.

    Partial pivoting chooses the same largest-column rule as LU. Singular
    matrices return sign zero and log magnitude minus infinity. Derivatives
    are defined only on the nonsingular branch, as for TensorFlow slogdet.
    """
    count = matrix.shape[-1]
    if matrix.shape.rank != 2 or count is None or matrix.shape[-2] != count:
        raise ValueError("slogdet_tf requires a fixed square real matrix")
    indices = tf.range(count)

    def body(column, work, sign, log_abs):
        eligible = indices >= column
        magnitudes = tf.where(eligible, tf.abs(work[:, column]), tf.cast(-1., matrix.dtype))
        pivot = tf.argmax(magnitudes, output_type=tf.int32)
        permutation = tf.where(indices == column, pivot, tf.where(indices == pivot, column, indices))
        work = tf.gather(work, permutation)
        diagonal = work[column, column]
        sign *= tf.where(pivot == column, tf.cast(1., matrix.dtype), tf.cast(-1., matrix.dtype)) * tf.sign(diagonal)
        log_abs += tf.math.log(tf.abs(diagonal))
        safe_diagonal = tf.where(diagonal != 0., diagonal, tf.ones_like(diagonal))
        multiplier = tf.where(indices > column, work[:, column] / safe_diagonal, tf.zeros([count], matrix.dtype))
        updated = work - multiplier[:, None] * work[column, :][None, :]
        work = tf.where((indices[:, None] > column) & (indices[None, :] >= column), updated, work)
        return column + 1, work, sign, log_abs

    _, _, sign, log_abs = tf.while_loop(lambda column, *_: column < count, body,
        (tf.constant(0), matrix, tf.cast(1., matrix.dtype), tf.cast(0., matrix.dtype)),
        maximum_iterations=count, parallel_iterations=1)

    def grad(_d_sign, d_log_abs):
        return d_log_abs * tf.linalg.solve(tf.transpose(matrix), tf.eye(count, dtype=matrix.dtype))

    return (sign, log_abs), grad
