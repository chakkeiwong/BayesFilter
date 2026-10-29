"""Frozen two-observation KSC guide diagnostics, never fitted KDM evidence.

The mixture represents the exact future likelihood up to an x-independent
constant. Its positive precision field is responsibility-averaged, not the
negative Hessian. Every proposal and analytical score uses the shared consumer.
"""

import tensorflow as tf

from bayesfilter.highdim.sqmc_ksc_tf import MEANS, VARIANCES, WEIGHTS, persistence


CONDITIONS = {
    "ordinary": [[0.2], [-0.4]],
    "negative_future_tail": [[0.2], [-8.0]],
    "positive_future_tail": [[0.2], [5.0]],
}
ARMS = (
    ("unguided_identity", "none", (1.0, 0.0, 0.0)),
    ("unguided_global", "none", (0.1, 0.9, 0.0)),
    ("unguided_local", "none", (0.1, 0.0, 0.9)),
    ("unguided_equal", "none", (1.0 / 3.0,) * 3),
    ("gaussian_future_global", "gaussian", (0.1, 0.9, 0.0)),
    ("gaussian_future_equal", "gaussian", (1.0 / 3.0,) * 3),
    ("mixture_future_global", "mixture", (0.1, 0.9, 0.0)),
    ("mixture_future_equal", "mixture", (1.0 / 3.0,) * 3),
)


def future_guides(theta, observations):
    """Construct once at base theta; freeze all coefficients during score checks."""
    theta = tf.convert_to_tensor(theta, tf.float64)
    observations = tf.ensure_shape(tf.convert_to_tensor(observations, tf.float64), [2, 1])
    weights = tf.constant(WEIGHTS, theta.dtype)
    centers = (observations[1, 0] - 2.0 * theta[1] - tf.constant(MEANS, theta.dtype)) / persistence(theta)
    variances = (1.0 + tf.constant(VARIANCES, theta.dtype)) / tf.square(persistence(theta))
    mean = tf.reduce_sum(weights * centers)
    variance = tf.reduce_sum(weights * (variances + tf.square(centers))) - tf.square(mean)
    quadratic = (tf.reshape(tf.stack([1.0 / variance, tf.zeros([], theta.dtype)]), [2, 1, 1]),
                 tf.reshape(tf.stack([mean / variance, tf.zeros([], theta.dtype)]), [2, 1]))
    mixture = (tf.broadcast_to(tf.math.log(weights)[None], [2, len(WEIGHTS)]),
               tf.stack([centers[:, None], tf.zeros([len(WEIGHTS), 1], theta.dtype)]),
               tf.stack([variances[:, None, None], tf.ones([len(WEIGHTS), 1, 1], theta.dtype)]),
               tf.constant([True, False]))
    return {"none": (tf.zeros([2, 1, 1], theta.dtype), tf.zeros([2, 1], theta.dtype)),
            "gaussian": quadratic, "mixture": mixture}


def cloud_seed(condition_index, replicate):
    if condition_index not in range(len(CONDITIONS)) or replicate not in range(8):
        raise ValueError("cloud index is outside the frozen diagnostic design")
    return 617000 + 100 * condition_index + replicate


def make_clouds(particles, seed):
    """Independent stateless streams, shared by all eight arms within a cloud."""
    return (tf.random.stateless_normal([particles, 1], [seed, 1], dtype=tf.float64),
            tf.random.stateless_normal([2, particles, 1], [seed, 2], dtype=tf.float64),
            tf.random.stateless_uniform([2, particles, 2], [seed, 3], dtype=tf.float64))
