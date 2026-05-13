"""TensorFlow Conjugate Unscented Transform rules."""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.nonlinear.sigma_points_tf import TFSigmaPointRule


def _axis_offsets(dim: int, distance: tf.Tensor) -> tf.Tensor:
    eye = tf.eye(dim, dtype=tf.float64)
    return tf.concat([distance * eye, -distance * eye], axis=0)


def _corner_offsets(dim: int, distance: tf.Tensor) -> tf.Tensor:
    count = 2**dim
    values = tf.range(count, dtype=tf.int32)[:, tf.newaxis]
    bits = tf.range(dim, dtype=tf.int32)[tf.newaxis, :]
    signs = tf.where(
        tf.equal(tf.bitwise.bitwise_and(values, tf.bitwise.left_shift(1, bits)), 0),
        tf.ones([count, dim], dtype=tf.float64),
        -tf.ones([count, dim], dtype=tf.float64),
    )
    return distance * signs


def tf_cut4g_sigma_point_rule(dim: int) -> TFSigmaPointRule:
    """Return the positive-weight Gaussian CUT4-G rule.

    The rule uses ``2 * dim + 2**dim`` points.  The closed-form constants are
    the generic CUT4-G constants used by the DSGE experimental CUTSRUKF donor.
    """

    dim = int(dim)
    if dim < 3:
        raise ValueError("CUT4-G requires dim >= 3")
    n = float(dim)
    axis_distance = tf.sqrt(tf.constant((n + 2.0) / 2.0, dtype=tf.float64))
    corner_distance = tf.sqrt(tf.constant((n + 2.0) / (n - 2.0), dtype=tf.float64))
    axis_weight = tf.constant(4.0 / (n + 2.0) ** 2, dtype=tf.float64)
    corner_weight = tf.constant(
        (n - 2.0) ** 2 / (2.0**dim * (n + 2.0) ** 2),
        dtype=tf.float64,
    )
    axis = _axis_offsets(dim, axis_distance)
    corners = _corner_offsets(dim, corner_distance)
    offsets = tf.concat([axis, corners], axis=0)
    mean_weights = tf.concat(
        [
            tf.fill([2 * dim], axis_weight),
            tf.fill([2**dim], corner_weight),
        ],
        axis=0,
    )
    return TFSigmaPointRule(
        name="CUT4-G",
        dim=dim,
        offsets=offsets,
        mean_weights=mean_weights,
        covariance_weights=mean_weights,
        polynomial_degree=5,
    )
