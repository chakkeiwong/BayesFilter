"""
Resampling algorithms for particle filters — TensorFlow implementation.

All functions are tf.function-compatible (no Python-side randomness, no NumPy).
Uses tf.random.stateless_* or tf.random.Generator for reproducibility.

Mirrors: utils/resampling.py (NumPy version)
"""

import tensorflow as tf


@tf.function
def systematic_resample(weights: tf.Tensor, rng: tf.random.Generator) -> tf.Tensor:
    """
    Systematic resampling.

    Deterministic spacing with single random offset. Low variance.

    Args:
        weights: [N] Normalized weights (must sum to 1), float32/64
        rng: tf.random.Generator

    Returns:
        indices: [N] Resampled particle indices, int32
    """
    N = tf.shape(weights)[0]
    N_f = tf.cast(N, weights.dtype)

    # Cumulative sum
    cdf = tf.cumsum(weights)
    # Force last element to exactly 1.0
    cdf = tf.concat([cdf[:-1], tf.ones([1], dtype=weights.dtype)], axis=0)

    # Single uniform offset in [0, 1/N)
    u0 = rng.uniform(shape=[], minval=0.0, maxval=1.0 / N_f, dtype=weights.dtype)
    u = u0 + tf.cast(tf.range(N), weights.dtype) / N_f

    # searchsorted: find index where cdf >= u
    indices = tf.searchsorted(cdf, u, side='left')
    indices = tf.minimum(indices, N - 1)

    return tf.cast(indices, tf.int32)


@tf.function
def stratified_resample(weights: tf.Tensor, rng: tf.random.Generator) -> tf.Tensor:
    """
    Stratified resampling.

    Independent random draw within each stratum.

    Args:
        weights: [N] Normalized weights (must sum to 1)
        rng: tf.random.Generator

    Returns:
        indices: [N] Resampled particle indices, int32
    """
    N = tf.shape(weights)[0]
    N_f = tf.cast(N, weights.dtype)

    cdf = tf.cumsum(weights)
    cdf = tf.concat([cdf[:-1], tf.ones([1], dtype=weights.dtype)], axis=0)

    # Stratified positions: (i + U_i) / N, U_i ~ Uniform(0, 1)
    u_i = rng.uniform(shape=[N], minval=0.0, maxval=1.0, dtype=weights.dtype)
    u = (tf.cast(tf.range(N), weights.dtype) + u_i) / N_f

    indices = tf.searchsorted(cdf, u, side='left')
    indices = tf.minimum(indices, N - 1)

    return tf.cast(indices, tf.int32)


@tf.function
def multinomial_resample(weights: tf.Tensor, rng: tf.random.Generator) -> tf.Tensor:
    """
    Multinomial resampling.

    Standard resampling with replacement. Higher variance than systematic.

    Args:
        weights: [N] Normalized weights (must sum to 1)
        rng: tf.random.Generator

    Returns:
        indices: [N] Resampled particle indices, int32
    """
    N = tf.shape(weights)[0]
    # tf.random.categorical expects log-probabilities and shape [batch, num_classes]
    log_weights = tf.math.log(weights + 1e-38)  # avoid log(0)
    log_weights = tf.expand_dims(log_weights, axis=0)  # [1, N]

    # Draw N samples; categorical returns [1, N]
    # Note: tf.random.categorical uses its own RNG; for determinism we
    # use the Generator to produce uniform samples + searchsorted instead.
    N_f = tf.cast(N, weights.dtype)
    cdf = tf.cumsum(weights)
    cdf = tf.concat([cdf[:-1], tf.ones([1], dtype=weights.dtype)], axis=0)

    u = rng.uniform(shape=[N], minval=0.0, maxval=1.0, dtype=weights.dtype)
    indices = tf.searchsorted(cdf, u, side='left')
    indices = tf.minimum(indices, N - 1)

    return tf.cast(indices, tf.int32)


@tf.function
def effective_sample_size(weights: tf.Tensor) -> tf.Tensor:
    """
    Compute effective sample size (ESS).

    ESS = 1 / sum(w_i^2), where weights are normalized.

    Args:
        weights: [N] Normalized weights (must sum to 1)

    Returns:
        ESS value (scalar), same dtype as weights
    """
    return 1.0 / tf.reduce_sum(weights ** 2)


@tf.function
def normalize_log_weights(log_weights: tf.Tensor):
    """
    Normalize log weights to get normalized weights.

    Args:
        log_weights: [N] Unnormalized log weights

    Returns:
        weights: [N] Normalized weights (sum to 1)
        log_normalizer: Log of the normalizing constant (scalar)
    """
    max_log = tf.reduce_max(log_weights)
    log_sum = max_log + tf.math.log(
        tf.reduce_sum(tf.exp(log_weights - max_log))
    )
    weights = tf.exp(log_weights - log_sum)
    return weights, log_sum
