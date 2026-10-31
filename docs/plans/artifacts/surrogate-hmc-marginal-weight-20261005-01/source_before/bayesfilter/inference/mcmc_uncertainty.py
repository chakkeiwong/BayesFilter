"""Chain-preserving batch covariance primitives, without an admission policy.

Vats--Flegal, arXiv:1809.04541, section 4.3, equation (7). Consistency requires
the paper's stationary/mixing and increasing-batch assumptions. A finite
positive estimate does not establish those assumptions or sufficient burn-in.
No covariance repair, zero-variance substitution, or chain concatenation occurs.
"""
from __future__ import annotations

from functools import lru_cache
import math

import tensorflow as tf

BATCH_COVARIANCE_VERSION = "bayesfilter.chain_batch_covariance.v1"


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@lru_cache(maxsize=32)
def _program(shape, batch_size, small_batch, c, minimum, jit_compile):
    _, n, chains, quantities = shape

    @tf.function(input_signature=[tf.TensorSpec(shape, tf.float64)],
                 autograph=False, jit_compile=jit_compile)
    def compute(values):
        def covariance(b):
            count = n // b
            means = tf.reduce_mean(tf.reshape(values[:, :count*b],
                [shape[0], count, b, chains, quantities]), axis=2)
            centered = means - tf.reduce_mean(means, axis=1, keepdims=True)
            return b * tf.einsum("racp,racq->rcpq", centered, centered) / (count - 1.)

        mean = tf.reduce_mean(values, axis=1)
        marginal_variance = tf.math.reduce_variance(values, axis=1) * n / (n-1.)
        enough = n // batch_size >= minimum and n // small_batch >= minimum
        if enough:
            plain = covariance(batch_size)
            lrv = (plain - c * covariance(small_batch)) / (1. - c)
        else:
            plain = tf.fill([shape[0], chains, quantities, quantities],
                            tf.constant(float("nan"), tf.float64))
            lrv = plain
        diagonal = tf.linalg.diag_part(lrv)
        # Compare values, not rounded variance, for exact constancy under XLA.
        nonconstant = tf.reduce_any(values != values[:, :1], axis=1)
        valid = (tf.reduce_all(tf.math.is_finite(values), axis=1) & nonconstant
                 & tf.math.is_finite(diagonal) & (diagonal > 0.)
                 & tf.math.is_finite(marginal_variance) & (marginal_variance > 0.))
        variance = tf.where(valid, diagonal / n, float("nan"))
        return {
            "mean_by_chain": mean,
            "long_run_covariance_by_chain": lrv,
            "plain_long_run_covariance_by_chain": plain,
            "variance_of_mean_by_chain": variance,
            "mcse_by_chain": tf.sqrt(variance),
            "effective_information_by_chain": marginal_variance / variance,
            "valid_by_chain": valid,
            "pooled_mean": tf.reduce_mean(mean, axis=1),
            "pooled_variance_of_mean": tf.reduce_sum(variance, axis=1) / chains**2,
        }
    return compute


def chain_batch_covariance(values, *, batch_size: int, min_batches: int,
                           method="lugsail", lugsail_r=3, lugsail_c=.5,
                           jit_compile=True):
    """Return tensors for [draw, chain, quantity], optionally replication first.

    Means use all draws. LRV estimates use complete batches, centered on their
    own complete-batch means. Covariances between quantities are preserved;
    marginal validity does not assert positive definiteness of the full matrix.
    Unavailable estimates retain NaN internally and never become zero MCSE.
    """
    _integer(batch_size, "batch_size", 1)
    _integer(min_batches, "min_batches", 2)
    _integer(lugsail_r, "lugsail_r", 1)
    if method not in {"batch_means", "lugsail"}:
        raise ValueError("method must be batch_means or lugsail")
    if not math.isfinite(lugsail_c) or not 0. <= lugsail_c < 1.:
        raise ValueError("lugsail_c must be in [0,1)")
    if type(jit_compile) is not bool:
        raise TypeError("jit_compile must be boolean")
    values = (tf.cast(values, tf.float64) if tf.is_tensor(values)
              else tf.convert_to_tensor(values, tf.float64))
    unbatched = values.shape.rank == 3
    if unbatched:
        values = values[None]
    if values.shape.rank != 4 or any(d is None or d < 1 for d in values.shape):
        raise ValueError("static [replication, draw, chain, quantity] required")
    shape = tuple(map(int, values.shape))
    if shape[1] < 2:
        raise ValueError("at least two draws required")
    small = batch_size if method == "batch_means" else batch_size // lugsail_r
    if small < 1:
        raise ValueError("batch_size is smaller than lugsail_r")
    c = 0. if method == "batch_means" else float(lugsail_c)
    tensors = _program(shape, batch_size, small, c, min_batches, jit_compile)(values)
    if unbatched:
        tensors = {key: value[0] for key, value in tensors.items()}
    return {**tensors, "schema": BATCH_COVARIANCE_VERSION,
            "method": method, "batch_size": batch_size, "small_batch_size": small,
            "batch_count": shape[1] // batch_size,
            "small_batch_count": shape[1] // small,
            "unused_terminal_draws": shape[1] % batch_size,
            "small_unused_terminal_draws": shape[1] % small,
            "min_batches": min_batches, "lugsail_r": lugsail_r, "lugsail_c": c,
            "jit_compile": jit_compile}
