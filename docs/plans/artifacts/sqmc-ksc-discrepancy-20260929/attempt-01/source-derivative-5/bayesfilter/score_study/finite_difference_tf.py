"""TF finite-difference operators and covariance-aware diagnostics.

FD scores are explicitly independent reference/candidate estimates. These
operators do not issue canonical analytical LEDH gradient identity.
"""
from functools import lru_cache

import tensorflow as tf


DESIGNS = {
    "central3": ((-1., 0., 1.), (-.5, 0., .5)),
    "central5": ((-2., -1., 0., 1., 2.), (1/12, -2/3, 0., 2/3, -1/12)),
    "cubic11": (tuple(float(j) for j in range(-5, 6)),
                tuple((41030*j-1958*j**3)/679536 for j in range(-5, 6))),
}


def stencil(kind, dtype):
    nodes, coefficients = DESIGNS[kind]
    return tf.constant(nodes, dtype), tf.constant(coefficients, dtype)


@lru_cache(maxsize=16)
def make_stencil_kernel(kind, dtype_name="float64", jit_compile=True):
    dtype = tf.as_dtype(dtype_name)
    size = len(DESIGNS[kind][0])
    _, coefficients = stencil(kind, dtype)
    @tf.function(input_signature=[tf.TensorSpec([None, size], dtype), tf.TensorSpec([], dtype)], jit_compile=jit_compile)
    def kernel(values, h):
        valid = tf.math.is_finite(h) & (h > 0) & tf.reduce_all(tf.math.is_finite(values))
        derivative = tf.linalg.matvec(values, coefficients) / tf.where(valid, h, tf.ones_like(h))
        return tf.where(valid, derivative, tf.fill(tf.shape(derivative), tf.cast(float("nan"), dtype))), valid
    return kernel


@lru_cache(maxsize=16)
def make_direction_solver(dimension, directions, dtype_name="float64", jit_compile=True):
    """Solve V^T s=b with SVD and return numerical rank and error amplification.

    The dtype/dimension rank guard is necessary but not a scientific accuracy
    threshold. A downstream directional error budget must still be propagated.
    """
    dtype = tf.as_dtype(dtype_name)
    epsilon = 2**(-52 if dtype == tf.float64 else -23)
    @tf.function(input_signature=[tf.TensorSpec([dimension, directions], dtype),
                                  tf.TensorSpec([directions], dtype)], jit_compile=jit_compile)
    def kernel(V, b):
        singular, left, right = tf.linalg.svd(V, full_matrices=False)
        margin = tf.cast(max(dimension, directions) * epsilon, dtype) * singular[0]
        rank = tf.reduce_sum(tf.cast(singular > margin, tf.int32))
        valid = (rank == dimension) & tf.reduce_all(tf.math.is_finite(V)) & tf.reduce_all(tf.math.is_finite(b))
        denominator = tf.where(singular > margin, singular, tf.ones_like(singular))
        inverse_map = (left / denominator[None, :]) @ tf.transpose(right)
        solution = tf.linalg.matvec(inverse_map, b)
        solution = tf.where(valid, solution, tf.fill([dimension], tf.cast(float("nan"), dtype)))
        amplification = tf.where(valid, 1 / denominator[-1], tf.cast(float("inf"), dtype))
        return solution, rank, singular, amplification, valid
    return kernel


def checked_nodes(theta, direction, h, kind, lower, upper):
    """Host-side validity guard before a likelihood is evaluated."""
    nodes, _ = stencil(kind, theta.dtype)
    points = theta[None, :] + nodes[:, None] * h * direction[None, :]
    if not bool(tf.reduce_all(tf.math.is_finite(points))) or not bool(tf.math.is_finite(h) & (h > 0)):
        raise ValueError("non-finite perturbations or nonpositive h")
    if not bool(tf.reduce_all((points >= lower) & (points <= upper))):
        raise ValueError("stencil leaves the model parameter domain")
    if not bool(tf.reduce_all(tf.reduce_any(points[1:] != points[:-1], axis=1))):
        raise ValueError("perturbations coincide at the working dtype")
    return points


def replicated_stencil_diagnostics(values, coefficients, h, exact_nodes, exact_score):
    """Unbiased sample variance/covariance and direct empirical oracle MSE.

    values is [independent particle replicate, node]. Dataset nesting is
    handled by the study report, not flattened into this conditional estimate.
    """
    count = int(values.shape[0])
    if count < 2:
        raise ValueError("at least two independent particle replicates are required")
    derivatives = tf.linalg.matvec(values, coefficients) / h
    sample_mean = tf.reduce_mean(derivatives)
    centered = values - tf.reduce_mean(values, axis=0)
    covariance = tf.transpose(centered) @ centered / tf.cast(count - 1, values.dtype)
    variance = tf.tensordot(coefficients, tf.linalg.matvec(covariance, coefficients), 1) / h**2
    mse = tf.reduce_mean((derivatives - exact_score)**2)
    truncation = tf.reduce_sum(exact_nodes * coefficients) / h - exact_score
    particle_bias = sample_mean - exact_score - truncation
    # This correction estimates squared total bias without folding MC variance
    # of its estimated mean into the bias. It can legitimately be negative.
    squared_bias_unbiased = (sample_mean-exact_score)**2 - variance / count
    return {"mse": mse, "sample_variance": variance, "squared_bias_unbiased": squared_bias_unbiased,
            "truncation": truncation, "particle_bias_estimate": particle_bias,
            "covariance": covariance, "mean": sample_mean, "count": count,
            "empirical_mse_identity": (sample_mean-exact_score)**2 + variance * (count-1)/count}
