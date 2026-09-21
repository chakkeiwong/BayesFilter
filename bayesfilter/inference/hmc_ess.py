"""Stan/ArviZ ESS arithmetic for preserved-transition posterior diagnostics.

The separately identified TFP mean-precision estimator remains in
hmc_diagnostic_math. All call sites declare which convention they use.
"""
from __future__ import annotations
import math
from functools import lru_cache
from typing import Any

STAN_ESS_VERSION = "bayesfilter.stan_initial_positive_monotone_ess.v1"


@lru_cache(maxsize=32)
def _compiled_ess(shape: tuple[int, ...]) -> Any:
    """Cache one XLA recursion per static input shape, not per coordinate/call."""
    import tensorflow as tf

    return tf.function(_cross_chain_ess_impl,
        input_signature=[tf.TensorSpec(shape, tf.float64)],
        autograph=False, jit_compile=True)


def stan_cross_chain_ess(sample_major: Any) -> Any:
    """Run the standard ESS recursion in its cached, fixed-shape XLA graph."""
    import tensorflow as tf

    values = tf.cast(tf.convert_to_tensor(sample_major), tf.float64)
    if values.shape.rank != 3 or any(dim is None for dim in values.shape):
        raise ValueError("cross-chain ESS requires static [draw, chain, parameter]")
    return _compiled_ess(tuple(int(dim) for dim in values.shape))(values)


def _cross_chain_ess_impl(sample_major: Any) -> Any:
    r"""Geyer initial-positive/monotone ESS for [draw, chain, parameter].

    This follows Vehtari et al. (2021) and the Stan posterior/ArviZ estimator:
    biased lag autocovariances, unbiased within-chain variance, monotone paired
    autocorrelations and the final even-lag improvement. The finite-sample
    safeguard tau >= 1/log10(S) prevents unstable negative estimates; it is not
    a theoretical upper bound on the true ESS of antithetic chains. No lag
    weights are applied to the between-chain variance correction.

    One chain is supported for per-chain MCSE. Constant pooled samples return
    NaN, keeping them non-promotable rather than assigning spurious precision.
    """
    import tensorflow as tf

    values = tf.cast(tf.convert_to_tensor(sample_major), tf.float64)
    if values.shape.rank != 3 or any(dim is None for dim in values.shape):
        raise ValueError("cross-chain ESS requires static [draw, chain, parameter]")
    draw_count, chain_count, parameter_count = (int(dim) for dim in values.shape)
    if draw_count < 2 or chain_count < 1 or parameter_count < 1:
        raise ValueError("ESS requires at least two draws, one chain and one parameter")
    rotated = tf.transpose(values, (1, 2, 0))
    centered = rotated - tf.reduce_mean(rotated, axis=-1, keepdims=True)
    fft_length = 1 << math.ceil(math.log2(2 * draw_count))
    padded = tf.pad(centered, ((0, 0), (0, 0), (0, fft_length - draw_count)))
    spectrum = tf.signal.rfft(padded)
    autocov = tf.transpose(tf.signal.irfft(
        spectrum * tf.math.conj(spectrum), fft_length=(fft_length,)
    )[..., :draw_count] / tf.cast(draw_count, tf.float64), (2, 0, 1))
    mean_autocov = tf.reduce_mean(autocov, axis=1)
    within = mean_autocov[0] * draw_count / (draw_count - 1.0)
    variance_plus = mean_autocov[0]
    if chain_count > 1:
        chain_means = tf.reduce_mean(values, axis=0)
        variance_plus += tf.math.reduce_variance(chain_means, axis=0) * (
            chain_count / (chain_count - 1.0)
        )
    rho = tf.concat((tf.ones_like(mean_autocov[:1]),
        1.0 - (within[tf.newaxis, :] - mean_autocov[1:]) /
        variance_plus[tf.newaxis, :]), axis=0)

    # The terminal pair contributes only its even lag. This preserves the
    # reference end-of-array rule as well as the first nonpositive-pair stop.
    # Each parameter stops separately; the TF loop accumulates only vectors.
    pair_count = max(1, (draw_count - 3) // 2 + 1)
    pairs = tf.reduce_sum(tf.reshape(rho[:2 * pair_count],
        [pair_count, 2, parameter_count]), axis=1)
    nonpositive = pairs <= 0.0
    stop = tf.where(tf.reduce_any(nonpositive, axis=0),
        tf.argmax(tf.cast(nonpositive, tf.int32), axis=0, output_type=tf.int32),
        tf.fill([parameter_count], pair_count - 1))

    def accumulate(index, previous_pair, total):
        monotone_pair = tf.minimum(previous_pair, pairs[index])
        return index + 1, monotone_pair, total + tf.where(
            index < stop, monotone_pair, tf.zeros_like(monotone_pair))

    _, _, pair_sum = tf.while_loop(
        lambda index, previous_pair, total: index < pair_count - 1,
        accumulate, (tf.constant(0),
            tf.fill([parameter_count], tf.constant(float("inf"), tf.float64)),
            tf.zeros([parameter_count], tf.float64)), parallel_iterations=1)
    indices = tf.stack((2 * stop, tf.range(parameter_count)), axis=1)
    even = tf.gather_nd(rho, indices)
    terminal_pair = tf.gather_nd(pairs, tf.stack((stop, tf.range(parameter_count)), axis=1))
    last_even = tf.where(terminal_pair >= 0.0, even, tf.maximum(even, 0.0))
    total_draws = tf.cast(chain_count * draw_count, tf.float64)
    tau_bound = tf.math.log(tf.constant(10.0, tf.float64)) / tf.math.log(total_draws)
    tau = tf.maximum(-1.0 + 2.0 * pair_sum + last_even, tau_bound)
    ess = total_draws / tau
    valid = tf.logical_and(variance_plus > 0.0,
        tf.reduce_all(tf.math.is_finite(values), axis=(0, 1)))
    return tf.where(valid, ess, tf.fill([parameter_count], tf.constant(float("nan"), tf.float64)))
