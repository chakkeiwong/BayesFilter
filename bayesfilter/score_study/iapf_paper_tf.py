"""Candidate value-only GJL Algorithm 5 for explicit linear-Gaussian models.

The initial state follows mu directly. This does not implement twist fitting,
adaptive iAPF iterations, scores, or the separate score-study model defaults.
"""
from functools import lru_cache
import tensorflow as tf


def _log_normal(points, mean, covariance):
    lower = tf.linalg.cholesky(covariance)
    whitened = tf.linalg.triangular_solve(lower, tf.transpose(points - mean))
    normalizer = tf.cast(points.shape[-1], points.dtype) * tf.math.log(
        tf.constant(2.0 * 3.141592653589793, points.dtype))
    return -.5 * (normalizer + tf.reduce_sum(tf.square(whitened), axis=0)) - tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(lower)))


def _add_floor(log_density, log_floor):
    return tf.reduce_logsumexp(tf.stack([
        log_density, tf.fill(tf.shape(log_density), log_floor)], axis=0), axis=0)


@lru_cache(maxsize=16)
def make_paper_apf_value_kernel(d, o, n, horizon, dtype_name="float64", jit_compile=True):
    """Stable graph; supplied random arrays define a reproducible finite value."""
    if min(d, o, horizon) < 1 or n < 2:
        raise ValueError("positive dimensions/horizon and at least two particles required")
    dtype = tf.as_dtype(dtype_name)
    spec = lambda shape: tf.TensorSpec(shape, dtype)

    @tf.function(input_signature=[
        spec([d, d]), spec([o, d]), spec([d]), spec([d, d]), spec([d, d]),
        spec([o, o]), spec([horizon, o]), spec([horizon, d]),
        spec([horizon, d, d]), spec([horizon]),
        tf.TensorSpec([horizon], tf.bool), spec([horizon, n, d]),
        spec([horizon, n]), spec([horizon, n]), spec([])], jit_compile=jit_compile)
    def kernel(A, C, initial_mean, P, Q, R, observations, centers, covariances,
               floors, constants, noise, mixture_uniforms, ancestor_uniforms, kappa):
        count = tf.cast(n, dtype)
        log_count = tf.math.log(count)
        def logmean(logweights):
            return tf.reduce_logsumexp(logweights) - log_count
        def ess(logweights):
            return tf.math.reciprocal(tf.reduce_sum(tf.square(tf.nn.softmax(logweights))))
        def integral(means, covariance, t):
            return tf.cond(constants[t], lambda: tf.zeros([n], dtype),
                lambda: _add_floor(_log_normal(means, centers[t],
                                               covariance + covariances[t]), floors[t]))
        def proposal(means, covariance, t):
            base = means + tf.linalg.matmul(noise[t], tf.linalg.cholesky(covariance),
                                           transpose_b=True)
            def twisted():
                V = covariances[t]
                gain = tf.transpose(tf.linalg.cholesky_solve(
                    tf.linalg.cholesky(covariance + V), tf.transpose(covariance)))
                posterior = covariance - tf.matmul(gain, covariance)
                posterior = .5 * (posterior + tf.transpose(posterior))
                # Separate the time-varying center from the batched GEMM.
                adapted_mean = means - tf.linalg.matmul(means, gain, transpose_b=True)
                adapted_mean += tf.linalg.matvec(gain, centers[t])[None, :]
                adapted = adapted_mean + tf.linalg.matmul(
                    noise[t], tf.linalg.cholesky(posterior), transpose_b=True)
                mass = integral(means, covariance, t)
                floor_probability = tf.exp(floors[t] - mass)
                return tf.where((mixture_uniforms[t] < floor_probability)[:, None], base, adapted)
            return tf.cond(constants[t], lambda: base, twisted)

        clouds = tf.TensorArray(dtype, horizon, element_shape=[n, d])
        weights_history = tf.TensorArray(dtype, horizon, element_shape=[n])
        ancestors = tf.TensorArray(tf.int32, horizon, element_shape=[n])
        ess_history = tf.TensorArray(dtype, horizon, element_shape=[])
        flags = tf.TensorArray(tf.bool, horizon, element_shape=[])
        prefixes = tf.TensorArray(dtype, horizon, element_shape=[])

        def step(t, x, weights, accumulated, clouds, weights_history, ancestors,
                 ess_history, flags, prefixes):
            # ESS <= N by Cauchy-Schwarz; enforce the exact kappa=1 endpoint.
            resample = tf.logical_and(t > 0, tf.logical_or(
                kappa == tf.ones([], dtype), ess(weights) <= kappa * count))
            def draw_ancestors():
                cdf = tf.cumsum(tf.nn.softmax(weights))
                return tf.minimum(tf.searchsorted(cdf, ancestor_uniforms[t], side="right"), n - 1)
            indices = tf.cond(resample, draw_ancestors, lambda: tf.range(n))
            accumulated += tf.where(resample, logmean(weights), tf.zeros([], dtype))
            previous = tf.where(resample, tf.zeros([n], dtype), weights)
            means = tf.cond(t == 0,
                lambda: tf.broadcast_to(initial_mean, [n, d]),
                lambda: tf.linalg.matmul(tf.gather(x, indices), A, transpose_b=True))
            covariance = tf.cond(t == 0, lambda: P, lambda: Q)
            x = proposal(means, covariance, t)
            log_observation = _log_normal(
                tf.broadcast_to(observations[t], [n, o]),
                tf.linalg.matmul(x, C, transpose_b=True), R)
            log_twist = tf.cond(constants[t], lambda: tf.zeros([n], dtype),
                lambda: _add_floor(_log_normal(x, centers[t], covariances[t]), floors[t]))
            future = tf.cond(t + 1 < horizon,
                lambda: integral(tf.linalg.matmul(x, A, transpose_b=True), Q, t + 1),
                lambda: tf.zeros([n], dtype))
            weights = previous + log_observation + future - log_twist
            weights += tf.cond(t == 0, lambda: integral(means, covariance, t),
                               lambda: tf.zeros([n], dtype))
            return (t + 1, x, weights, accumulated, clouds.write(t, x),
                weights_history.write(t, weights), ancestors.write(t, indices),
                ess_history.write(t, ess(weights)), flags.write(t, resample),
                prefixes.write(t, accumulated + logmean(weights - future)))

        result = tf.while_loop(lambda t, *_: t < horizon, step,
            (tf.constant(0), tf.zeros([n, d], dtype), tf.zeros([n], dtype),
             tf.zeros([], dtype), clouds, weights_history, ancestors,
             ess_history, flags, prefixes), parallel_iterations=1)
        return (result[3] + logmean(result[2]), result[4].stack(), result[5].stack(),
                result[6].stack(), result[7].stack(), result[8].stack(), result[9].stack())
    return kernel
