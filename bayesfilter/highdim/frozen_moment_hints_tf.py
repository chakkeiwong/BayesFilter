"""Tensor-indexed access to already frozen moment-hint artifacts."""

import tensorflow as tf


def frozen_moment_hint_callbacks(hints, horizon):
    """Decode an ordered artifact schema without date-dependent Python calls.

    This does not estimate moments: its inputs must contain the frozen means
    and covariances. The list iteration decodes artifact records only.
    """
    hints = tuple(hints[:horizon])
    if horizon < 1 or len(hints) != horizon:
        raise ValueError("frozen hint count must match positive horizon")
    if any(int(row["time_index"]) != index for index,row in enumerate(hints)):
        raise ValueError("frozen hints must be ordered with consecutive dates")
    initial_mean = tf.convert_to_tensor(hints[0]["mean"],tf.float64)
    initial_cov = tf.convert_to_tensor(hints[0]["covariance"],tf.float64)
    n = initial_mean.shape[0]
    means = tf.reshape(tf.convert_to_tensor([row["mean"] for row in hints[1:]],tf.float64),[horizon-1,2*n])
    covariances = tf.reshape(tf.convert_to_tensor([row["covariance"] for row in hints[1:]],tf.float64),[horizon-1,2*n,2*n])
    if initial_cov.shape != (n,n):
        raise ValueError("initial frozen covariance shape mismatch")
    if not bool(tf.reduce_all(tf.stack(tuple(tf.reduce_all(tf.math.is_finite(value))
        for value in (initial_mean,initial_cov,means,covariances)))).numpy()):
        raise ValueError("frozen moments must be finite")
    return (lambda observation: (initial_mean,initial_cov),
        lambda date,observation: (tf.gather(means,date-1),tf.gather(covariances,date-1)))
