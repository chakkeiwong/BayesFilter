"""Tensor-native packed upper-triangle score-regression design."""

import tensorflow as tf


def upper_triangle_indices(dimension):
    row = tf.range(dimension)
    starts = row * (2 * dimension - row + 1) // 2
    index = tf.range(dimension * (dimension + 1) // 2)
    rows = (
        tf.reduce_sum(tf.cast(index[:, None] >= starts[None, :], tf.int32), axis=1) - 1
    )
    return rows, rows + index - tf.gather(starts, rows)


def symmetric_score_design(z, dimension):
    rows, columns = upper_triangle_indices(dimension)
    direct = (
        tf.gather(z, columns, axis=1)[:, None, :]
        * tf.transpose(tf.one_hot(rows, dimension, dtype=z.dtype))[None, :, :]
    )
    reflected = (
        tf.gather(z, rows, axis=1)[:, None, :]
        * tf.transpose(tf.one_hot(columns, dimension, dtype=z.dtype))[None, :, :]
    )
    return direct + tf.where(
        (rows != columns)[None, None, :], reflected, tf.zeros_like(reflected)
    )


def unpack_symmetric(coefficients, dimension):
    rows, columns = upper_triangle_indices(dimension)
    upper = tf.scatter_nd(
        tf.stack((rows, columns), axis=1), coefficients, [dimension, dimension]
    )
    return upper + tf.transpose(
        tf.linalg.set_diag(upper, tf.zeros([dimension], upper.dtype))
    )
