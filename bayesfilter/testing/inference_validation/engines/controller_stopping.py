"""Exact Gaussian diagnostic fixtures for the real sequential controller.

This module is a validation/reference tool, not an HMC tuner. Fixed-count
Gaussian mean calculations are exact for the declared start distribution;
they are not confidence statements at a random stopping time.
"""
from __future__ import annotations

from functools import lru_cache
import math

import tensorflow as tf

from bayesfilter.inference.hmc_precision import mean_precision


class GaussianAR1Transition:
    """Compiled, batch-native exact transition preserving N(0,1)."""

    def __init__(self, rho, *, chains=4, jit_compile=True):
        if not math.isfinite(rho) or not -1 < rho < 1:
            raise ValueError("rho must be finite and strictly between -1 and 1")
        if type(chains) is not int or chains < 4:
            raise ValueError("at least four chains are required")
        self.rho = float(rho)
        self.chains = chains
        self.jit_compile = bool(jit_compile)
        self.signature = f"diagnostic_gaussian_ar1:{rho!r}:{chains}"

    @lru_cache(maxsize=24)
    def program(self, num_results):
        if type(num_results) is not int or num_results < 1:
            raise ValueError("num_results must be a positive integer")
        shape = (self.chains, 1)
        rho = tf.constant(self.rho, tf.float64)
        scale = tf.constant(math.sqrt(1 - self.rho**2), tf.float64)

        @tf.function(input_signature=[tf.TensorSpec(shape, tf.float64),
                                     tf.TensorSpec((2,), tf.int32)],
                     autograph=False, jit_compile=self.jit_compile)
        def transition(state, seed):
            noise = tf.random.stateless_normal((num_results, *shape), seed, dtype=tf.float64)
            return tf.scan(lambda last, z: rho * last + scale * z, noise,
                           initializer=state, parallel_iterations=1)

        return transition

    def __call__(self, state, *, num_results, seed, stage):
        if stage not in {"warmup", "retained", "fixed"}:
            raise ValueError("unknown diagnostic stage")
        samples = self.program(num_results)(tf.convert_to_tensor(state, tf.float64), seed)
        finite = bool(tf.reduce_all(tf.math.is_finite(samples)))
        return {
            "transition_signature": self.signature,
            "posterior_stream_only": True,
            "posterior_temperature": 1.0,
            "posterior_samples": samples,
            "posterior_replica_identities": tf.broadcast_to(
                tf.range(self.chains)[None, :], (num_results, self.chains)),
            "final_transition_state": samples[-1],
            "health": {"passed": finite, "hard_vetoes": () if finite else ("nonfinite_ar1",)},
        }


def fixed_mean_law(rho, starts, *, warmup, draws, stationary_start=False):
    """Exact marginal mean/variance of the pooled mean at deterministic counts.

    X_t = rho**t X_0 + sqrt(1-rho**2) sum_{j=1}^t rho**(t-j) Z_j.
    Conditional covariance is rho**abs(s-t) - rho**(s+t). Integrating
    independent stationary X_0 restores the subtracted outer product.
    """
    if not math.isfinite(rho) or not -1 < rho < 1:
        raise ValueError("invalid AR(1) coefficient")
    if type(warmup) is not int or warmup < 0 or type(draws) is not int or draws < 1:
        raise ValueError("fixed counts must be nonnegative warmup and positive draws")
    starts = tuple(float(v) for v in starts)
    if not starts or any(not math.isfinite(v) for v in starts):
        raise ValueError("finite starting values required")
    powers = math.fsum(rho**t for t in range(warmup + 1, warmup + draws + 1))
    stationary_sum = draws + 2 * math.fsum((draws - k) * rho**k for k in range(1, draws))
    variance = (stationary_sum - (0. if stationary_start else powers**2)) / (len(starts) * draws**2)
    if variance <= 0 or not math.isfinite(variance):
        raise ValueError("invalid exact mean variance")
    mean = 0. if stationary_start else math.fsum(starts) * powers / (len(starts) * draws)
    return {"mean": mean, "variance": variance, "mcse": math.sqrt(variance),
            "stationary_start": stationary_start, "fixed_counts_only": True,
            "last_initial_mean_by_chain": [v * rho**warmup for v in starts]}


def mean_interval(samples, *, jit_compile):
    """Report ordinary normal lugsail intervals, including unavailable estimates."""
    if int(samples.shape[0]) < 4:
        return {"available": False, "covered": False, "estimate": None, "mcse": None}
    report = mean_precision(samples, method="lugsail", jit_compile=jit_compile)
    valid = bool(report["valid"][0])
    estimate = float(report["estimate"][0])
    se = float(report["mcse"][0]) if valid else None
    return {"available": valid, "covered": bool(valid and abs(estimate) <= 1.959963984540054 * se),
            "estimate": estimate, "mcse": se,
            "interval": [estimate - 1.959963984540054 * se, estimate + 1.959963984540054 * se] if valid else None,
            "reference": 0., "method": "lugsail_normal_95", "anytime_coverage_claim": False}
