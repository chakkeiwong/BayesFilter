"""Shared TensorFlow arithmetic for HMC diagnostic reports.

Inputs are [draw, chain, parameter]. Rank, fold and split follow Vehtari et al.
(2021), equations 3 and 14. The positive-pair ESS intentionally retains the
TFP 0.25 estimator; it is identified separately from Stan's monotone estimator.
These post-chunk diagnostics are graph/reference operations, not HMC transitions.
"""
from __future__ import annotations
import math
from functools import lru_cache, wraps
from typing import Any
import tensorflow as tf
import tensorflow_probability as tfp

DIAGNOSTIC_VERSION = "bayesfilter.hmc_diagnostic_math.v2"


def _diagnostic_graph(function):
    """Bound tracing to static shapes at a post-chunk diagnostic boundary.

    Rank grouping uses data-dependent segment reductions. This reporting-only
    graph deliberately uses non-XLA execution; the HMC transition keeps its
    independently qualified XLA policy.
    """
    @lru_cache(maxsize=32)
    def program(shape):
        return tf.function(function, input_signature=[tf.TensorSpec(shape, tf.float64)],
                           autograph=False, jit_compile=False)

    @wraps(function)
    def evaluate(values):
        values = tf.cast(tf.convert_to_tensor(values), tf.float64)
        if values.shape.rank != 3 or any(d is None for d in values.shape):
            raise ValueError("diagnostics require static [draw, chain, parameter]")
        return program(tuple(values.shape))(values)
    return evaluate


@_diagnostic_graph
def _rank_normalize(values: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(values)
    flat = tf.reshape(values, [-1, shape[-1]])
    count = tf.shape(flat)[0]

    def rank_column(column: tf.Tensor) -> tf.Tensor:
        order = tf.argsort(column, stable=True)
        sorted_column = tf.gather(column, order)
        new_group = tf.concat(
            [
                tf.constant([True]),
                tf.not_equal(sorted_column[1:], sorted_column[:-1]),
            ],
            axis=0,
        )
        group = tf.cumsum(tf.cast(new_group, tf.int32)) - 1
        ranks = tf.cast(tf.range(1, count + 1), tf.float64)
        rank_sum = tf.math.segment_sum(ranks, group)
        group_count = tf.math.segment_sum(tf.ones_like(ranks), group)
        sorted_ranks = tf.gather(rank_sum / group_count, group)
        return tf.gather(sorted_ranks, tf.argsort(order, stable=True))

    ranks = tf.map_fn(
        rank_column,
        tf.transpose(flat, [1, 0]),
        fn_output_signature=tf.TensorSpec(shape=(None,), dtype=tf.float64),
    )
    ranks = tf.transpose(ranks, [1, 0])
    probability = (ranks - 3.0 / 8.0) / (tf.cast(count, tf.float64) + 1.0 / 4.0)
    normal = tfp.distributions.Normal(
        loc=tf.constant(0.0, tf.float64),
        scale=tf.constant(1.0, tf.float64),
    )
    return tf.reshape(normal.quantile(probability), shape)


@_diagnostic_graph
def _cross_chain_ess(sample_major: Any) -> Any:
    """Vehtari cross-chain ESS with a warning-free real FFT covariance.

    TFP 0.25 computes the same autocovariance through a complex FFT and then
    casts ``complex128`` to ``float64``.  The mathematical result is real, but
    that cast emits a lossy-conversion warning for every ESS call.  Using
    ``rfft``/``irfft`` preserves the real-valued contract directly.
    """

    import tensorflow as tf

    values = tf.cast(tf.convert_to_tensor(sample_major), tf.float64)
    if values.shape.rank != 3 or any(dim is None for dim in values.shape):
        raise ValueError("cross-chain ESS requires static [draw, chain, parameter]")
    draw_count, chain_count, _ = (int(dim) for dim in values.shape)
    rotated = tf.transpose(values, (1, 2, 0))
    centered = rotated - tf.reduce_mean(rotated, axis=-1, keepdims=True)
    fft_length = 1 << int(math.ceil(math.log2(2 * draw_count)))
    padded = tf.pad(centered, ((0, 0), (0, 0), (0, fft_length - draw_count)))
    spectrum = tf.signal.rfft(padded)
    autocov_rotated = tf.signal.irfft(
        spectrum * tf.math.conj(spectrum), fft_length=(fft_length,)
    )[..., :draw_count]
    denominators = tf.cast(
        tf.range(draw_count, 0, -1), tf.float64
    )
    autocov = tf.transpose(
        autocov_rotated / denominators[tf.newaxis, tf.newaxis, :],
        (2, 0, 1),
    )

    chain_means = tf.reduce_mean(values, axis=0)
    between_div_n = tf.math.reduce_variance(chain_means, axis=0) * (
        tf.cast(chain_count, tf.float64)
        / tf.cast(chain_count - 1, tf.float64)
    )
    biased_within = tf.reduce_mean(autocov[0], axis=0)
    variance_plus = biased_within + between_div_n
    mean_autocov = tf.reduce_mean(autocov, axis=1)
    autocorrelation = 1.0 - (
        biased_within[tf.newaxis, :] - mean_autocov
    ) / variance_plus[tf.newaxis, :]
    lag_weight = tf.cast(
        tf.range(draw_count, 0, -1), tf.float64
    ) / tf.cast(draw_count, tf.float64)
    weighted = autocorrelation * lag_weight[:, tf.newaxis]

    even_count = draw_count - draw_count % 2
    pair_shape = (even_count // 2, 2, int(values.shape[2]))
    pair_correlation = tf.reduce_sum(
        tf.reshape(autocorrelation[:even_count], pair_shape), axis=1
    )
    positive_mask = tf.maximum(
        1.0
        - tf.cumsum(tf.cast(pair_correlation < 0.0, tf.float64), axis=0),
        0.0,
    )
    weighted_pairs = tf.reduce_sum(
        tf.reshape(weighted[:even_count], pair_shape), axis=1
    ) * positive_mask
    return (
        tf.cast(chain_count * draw_count, tf.float64)
        / (-1.0 + 2.0 * tf.reduce_sum(weighted_pairs, axis=0))
    )


@_diagnostic_graph
def _split_rhat_from_sample_major(sample_major: Any) -> Any:
    """Return the square-root between/within-chain variance diagnostic."""

    import tensorflow as tf

    values = tf.cast(sample_major, tf.float64)
    draws = tf.cast(tf.shape(values)[0], tf.float64)
    chain_means = tf.reduce_mean(values, axis=0)
    within_chain_variance = tf.math.reduce_variance(values, axis=0) * draws / (draws - 1.)
    within = tf.reduce_mean(within_chain_variance, axis=0)
    chains = tf.cast(tf.shape(values)[1], tf.float64)
    between_over_draws = tf.math.reduce_variance(chain_means, axis=0) * chains / (chains - 1.)
    variance_plus = ((draws - 1.0) / draws) * within + between_over_draws
    return tf.sqrt(variance_plus / within)
