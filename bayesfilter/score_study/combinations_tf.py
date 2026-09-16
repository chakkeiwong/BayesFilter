"""Frozen score combinations with explicit centering and calibration contracts.

Fit coefficients on data independent of evaluation. A control with known zero
mean preserves the baseline's expectation, including its finite-particle bias.
An oracle-fitted blend minimizes calibration squared error; it has no such
expectation identity. Numerical kernels cannot establish sample independence:
the study coordinator must enforce the disjoint partitions.
"""
from functools import lru_cache

import tensorflow as tf


@lru_cache(maxsize=8)
def make_combination_kernels(parameter_dimension, control_dimension, dtype_name="float64",
                             jit_compile=True):
    dtype = tf.as_dtype(dtype_name)
    p, k = parameter_dimension, control_dimension
    epsilon = tf.constant(2**-23 if dtype == tf.float32 else 2**-52, dtype)

    @tf.function(input_signature=[tf.TensorSpec([None, p], dtype),
                                 tf.TensorSpec([None, k], dtype)], jit_compile=jit_compile)
    def fit_control(scores, zero_mean_controls):
        # Center calibration rows to estimate covariance; evaluation uses the
        # proven zero center, never the estimated calibration control mean.
        x = zero_mean_controls - tf.reduce_mean(zero_mean_controls, axis=0)
        y = scores - tf.reduce_mean(scores, axis=0)
        singular, u, v = tf.linalg.svd(x, full_matrices=False)
        tolerance = epsilon * tf.cast(tf.maximum(tf.shape(x)[0], k), dtype) * singular[0]
        retained = singular > tolerance
        inverse = tf.where(retained, tf.math.divide_no_nan(tf.ones_like(singular), singular), 0)
        coefficient = (v * inverse[None, :]) @ tf.transpose(u) @ y
        valid = (tf.shape(x)[0] > 1) & tf.reduce_all(tf.math.is_finite(coefficient))
        return coefficient, tf.reduce_sum(tf.cast(retained, tf.int32)), valid

    @tf.function(input_signature=[tf.TensorSpec([None, p], dtype),
                                 tf.TensorSpec([None, k], dtype),
                                 tf.TensorSpec([k, p], dtype)], jit_compile=jit_compile)
    def apply_control(scores, zero_mean_controls, coefficient):
        return scores - zero_mean_controls @ coefficient

    @tf.function(input_signature=[tf.TensorSpec([None, p], dtype),
                                 tf.TensorSpec([None, p], dtype),
                                 tf.TensorSpec([None, p], dtype)], jit_compile=jit_compile)
    def fit_blend(baseline, auxiliary, oracle):
        difference = auxiliary - baseline
        denominator = tf.reduce_sum(difference**2)
        numerator = tf.reduce_sum((baseline - oracle) * difference)
        # Exact equality makes all coefficients equivalent; choose the original
        # baseline. No ridge or unreported convex-weight clipping is introduced.
        alpha = -tf.math.divide_no_nan(numerator, denominator)
        residual = baseline + alpha * difference - oracle
        return alpha, tf.reduce_mean(tf.reduce_sum(residual**2, axis=1))

    @tf.function(input_signature=[tf.TensorSpec([None, p], dtype),
                                 tf.TensorSpec([None, p], dtype),
                                 tf.TensorSpec([], dtype)], jit_compile=jit_compile)
    def apply_blend(baseline, auxiliary, alpha):
        return baseline + alpha * (auxiliary - baseline)

    return fit_control, apply_control, fit_blend, apply_blend


@lru_cache(maxsize=8)
def make_frozen_kdm_controls(component_count, dimension, sample_count,
                             dtype_name="float64", jit_compile=True):
    """Translation and common log-scale scores of a frozen Gaussian mixture.

Draw a component using the supplied independent uniforms and its Gaussian
innovation. Innovations may be shared with a baseline estimator, preserving
both marginal laws. Centers, weights and covariance are fixed independently
of these evaluation streams. The returned controls have exact mean zero by
the normalized-density derivative identity (IWSG with integrand one).
"""
    from bayesfilter.highdim.ledh_younis_kdm_tf import make_gaussian_kdm_kernel
    dtype = tf.as_dtype(dtype_name)
    M, d, n = component_count, dimension, sample_count
    density = make_gaussian_kdm_kernel(
        evaluation_count=n, component_count=M, dimension=d, direction_count=d+1,
        dtype=dtype, jit_compile=jit_compile,
        normalization_tolerance=1e-5 if dtype == tf.float32 else 1e-8)

    @tf.function(input_signature=[tf.TensorSpec([M], dtype),
        tf.TensorSpec([M, d], dtype), tf.TensorSpec([M, d, d], dtype),
        tf.TensorSpec([n], dtype), tf.TensorSpec([n, d], dtype)], jit_compile=jit_compile)
    def kernel(weights, means, covariance, uniforms, innovations):
        indices = tf.minimum(tf.searchsorted(tf.cumsum(weights), uniforms, side="right"), M-1)
        z = tf.gather(means, indices) + tf.einsum(
            "nij,nj->ni", tf.linalg.cholesky(tf.gather(covariance, indices)), innovations)
        d_means = tf.concat([tf.broadcast_to(tf.eye(d, dtype=dtype)[:, None, :], [d, M, d]),
                             tf.zeros([1, M, d], dtype)], axis=0)
        d_covariance = tf.concat([tf.zeros([d, M, d, d], dtype),
                                  2*covariance[None, ...]], axis=0)
        result = density(z, weights, means, covariance, tf.zeros([d+1, n, d], dtype),
                         tf.zeros([d+1, M], dtype), d_means, d_covariance)
        valid = (tf.reduce_all(result["valid"]) & tf.reduce_all(tf.math.is_finite(uniforms))
                 & tf.reduce_all((uniforms >= 0) & (uniforms < 1)))
        controls = tf.transpose(result["d_log_density"])
        return tf.where(valid, controls, tf.constant(float("nan"), dtype)), valid
    return kernel
