"""Signed log determinant using native pivoted elimination for XLA."""

import tensorflow as tf


def _pivoted_diagonal(matrix):
    """LU diagonal and row-permutation sign, including leading batch axes."""
    count = matrix.shape[-1]
    rank = matrix.shape.rank
    if rank is None or rank < 2 or count is None or matrix.shape[-2] != count:
        raise ValueError("native determinant requires fixed square real matrices")
    indices = tf.range(count)

    def body(column, work, sign):
        eligible = indices >= column
        magnitudes = tf.where(eligible, tf.abs(work[..., :, column]), tf.cast(-1., matrix.dtype))
        pivot = tf.argmax(magnitudes, axis=-1, output_type=tf.int32)
        permutation = tf.where(indices == column, pivot[..., None],
            tf.where(indices == pivot[..., None], column, indices))
        work = tf.gather(work, permutation, axis=-2, batch_dims=rank - 2)
        diagonal = work[..., column, column]
        sign *= tf.where(pivot == column, tf.cast(1., matrix.dtype), tf.cast(-1., matrix.dtype))
        safe_diagonal = tf.where(diagonal != 0., diagonal, tf.ones_like(diagonal))
        multiplier = tf.where(indices > column, work[..., :, column] / safe_diagonal[..., None],
            tf.zeros([count], matrix.dtype))
        updated = work - multiplier[..., :, None] * work[..., column, None, :]
        work = tf.where((indices[:, None] > column) & (indices[None, :] >= column), updated, work)
        return column + 1, work, sign

    _, work, sign = tf.while_loop(lambda column, *_: column < count, body,
        (tf.constant(0), matrix, tf.ones(tf.shape(matrix)[:-2], matrix.dtype)),
        maximum_iterations=count, parallel_iterations=1)
    return tf.linalg.diag_part(work), sign


def determinant_tf(matrix):
    """Native LU determinant with product arithmetic (also for singular input).

    This preserves the determinant-then-log value program at legacy call sites;
    use ``slogdet_tf`` when the intended finite program sums diagonal logs.
    """
    diagonal, sign = _pivoted_diagonal(matrix)
    return sign * tf.reduce_prod(diagonal, axis=-1)


@tf.custom_gradient
def slogdet_tf(matrix):
    """Real square matrix signed log determinant and inverse-transpose VJP.

    Partial pivoting chooses the same largest-column rule as LU. Singular
    matrices return sign zero and log magnitude minus infinity. Derivatives
    are defined only on the nonsingular branch, as for TensorFlow slogdet.
    """
    diagonal, permutation_sign = _pivoted_diagonal(matrix)
    sign = permutation_sign * tf.reduce_prod(tf.sign(diagonal), axis=-1)
    # Retain sequential log accumulation from the original scalar helper.
    def accumulate(column, total):
        return column + 1, total + tf.math.log(tf.abs(diagonal[..., column]))

    log_abs = tf.while_loop(lambda column, _: column < matrix.shape[-1], accumulate,
        (tf.constant(0), tf.zeros(tf.shape(matrix)[:-2], matrix.dtype)), parallel_iterations=1)[1]

    def grad(_d_sign, d_log_abs):
        identity = tf.eye(matrix.shape[-1], batch_shape=tf.shape(matrix)[:-2], dtype=matrix.dtype)
        return d_log_abs[..., None, None] * tf.linalg.solve(tf.linalg.matrix_transpose(matrix), identity)

    return (sign, log_abs), grad
