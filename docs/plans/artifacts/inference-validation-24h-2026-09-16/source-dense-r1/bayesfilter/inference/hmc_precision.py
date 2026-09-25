"""Estimand-specific posterior precision; never a tuning admission policy.

Batch means follow Vats--Flegal (2021), section 4, equation 7. Quantile MCSE
follows Vehtari et al. (2021), section 4.4. Finite estimates require the stated
moment and mixing assumptions; diagnostics cannot establish these assumptions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from typing import Any, Mapping

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.hmc_diagnostic_math import _cross_chain_ess


@dataclass(frozen=True)
class HMCPrecisionTarget:
    name: str
    kind: str = "mean"
    probability: float | None = None
    mcse_absolute_max: float | None = None
    mcse_sd_ratio_max: float | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or self.kind not in {"mean", "quantile"}:
            raise ValueError("precision target requires a name and mean/quantile kind")
        if self.kind == "quantile":
            if self.probability is None or not 0. < self.probability < 1.:
                raise ValueError("quantile probability must be strictly between zero and one")
        elif self.probability is not None:
            raise ValueError("mean target does not take a quantile probability")
        if self.mcse_absolute_max is None and self.mcse_sd_ratio_max is None:
            raise ValueError("precision requires an explicit MCSE tolerance")
        for v in (self.mcse_absolute_max, self.mcse_sd_ratio_max):
            if v is not None and (not math.isfinite(v) or v <= 0.):
                raise ValueError("MCSE tolerances must be finite and positive")


@dataclass(frozen=True)
class HMCPrecisionPolicy:
    targets: tuple[HMCPrecisionTarget, ...]
    method: str = "autocorrelation"
    batch_size: int | None = None
    min_batches: int = 20
    lugsail_r: int = 3
    lugsail_c: float = 0.5
    jit_compile: bool = True

    def __post_init__(self):
        object.__setattr__(self, "targets", tuple(self.targets))
        if not self.targets or any(not isinstance(t, HMCPrecisionTarget) for t in self.targets):
            raise ValueError("precision targets must be a nonempty typed sequence")
        if len({(t.name, t.kind, t.probability) for t in self.targets}) != len(self.targets):
            raise ValueError("duplicate precision target")
        if self.method not in {"autocorrelation", "batch_means", "lugsail"}:
            raise ValueError("unknown precision estimator")
        for v, lower in ((self.min_batches, 2), (self.lugsail_r, 1)):
            if type(v) is not int or v < lower:
                raise ValueError("invalid batch count or lugsail ratio")
        if self.batch_size is not None and (type(self.batch_size) is not int or self.batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        if not math.isfinite(self.lugsail_c) or not 0. <= self.lugsail_c < 1.:
            raise ValueError("lugsail c must be in [0, 1)")
        if type(self.jit_compile) is not bool:
            raise TypeError("jit_compile must be boolean")

    def payload(self):
        return {"schema": "bayesfilter.hmc_precision_policy.v1", **asdict(self),
                "batch_provenance": "explicit" if self.batch_size else "sqrt(n), literature baseline",
                "minimum_batches_provenance": "operational stability floor; not calibrated coverage",
                "moment_assumption": "finite variance and applicable mean CLT; not proved by these diagnostics",
                "stopping_interpretation": "repeated-look operational MCSE screen; no anytime coverage"}


@lru_cache(maxsize=32)
def _batch_program(shape, batch, small_batch, c, jit_compile):
    @tf.function(input_signature=[tf.TensorSpec(shape, tf.float64)],
                 autograph=False, jit_compile=jit_compile)
    def compute(values):
        def estimate(b):
            count = shape[0] // b
            used = values[:count * b]
            means = tf.reduce_mean(tf.reshape(used, (count, b, shape[1], shape[2])), axis=1)
            # Center on the complete-batch mean. Unused terminal draws remain
            # in the posterior mean; report their exclusion from LRV estimation.
            return b * tf.reduce_sum(tf.square(means - tf.reduce_mean(means, axis=0)), axis=0) / (count - 1.)
        plain = estimate(batch)
        small = estimate(small_batch)
        return plain, (plain - c * small) / (1. - c)
    return compute


def mean_precision(samples: Any, *, method="autocorrelation", batch_size=None,
                   min_batches=20, lugsail_r=3, lugsail_c=0.5, jit_compile=True):
    """Return tensors for [draw, chain, quantity], preserving chain boundaries."""
    values = tf.convert_to_tensor(samples, tf.float64)
    if values.shape.rank != 3 or any(d is None for d in values.shape):
        raise ValueError("precision samples require static [draw, chain, quantity]")
    n, m, p = map(int, values.shape)
    if n < 4 or m < 2 or p < 1:
        raise ValueError("precision requires four draws and at least two chains")
    tf.debugging.assert_all_finite(values, "precision samples must be finite")
    total = n * m
    if type(min_batches) is not int or min_batches < 2:
        raise ValueError("min_batches must be an integer at least two")
    sd = tf.sqrt(tf.math.reduce_variance(values, axis=(0, 1)) * total / (total - 1.))
    metadata = {"method": method, "draws_per_chain": n, "chain_count": m,
                "long_run_variance_assumption": "stationary/CLT approximation for each independent chain"}
    if method == "autocorrelation":
        ess = _cross_chain_ess(values)
        variance = tf.square(sd) / ess
        metadata["estimator"] = "tfp_0.25_positive_pairs_real_fft"
    elif method in {"batch_means", "lugsail"}:
        b = int(math.sqrt(n)) if batch_size is None else batch_size
        if type(b) is not int or b < 1 or type(lugsail_r) is not int or lugsail_r < 1:
            raise ValueError("invalid batch size or lugsail r")
        if not math.isfinite(lugsail_c) or not 0 <= lugsail_c < 1:
            raise ValueError("invalid lugsail c")
        small = b if method == "batch_means" else b // lugsail_r
        metadata.update(estimator=method, batch_size=b, small_batch_size=small, batch_count=n // b,
                        unused_terminal_draws=n % b, lugsail_r=lugsail_r,
                        lugsail_c=0. if method == "batch_means" else lugsail_c)
        if min_batches < 2 or small < 1 or n // b < min_batches or n // small < min_batches:
            variance = tf.fill((p,), tf.constant(float("nan"), tf.float64))
            metadata["unavailable_reason"] = "insufficient_complete_batches"
        else:
            plain, lrv = _batch_program(tuple(values.shape), b, small,
                0. if method == "batch_means" else lugsail_c, jit_compile)(values)
            variance = tf.reduce_sum(lrv, axis=0) / (m * m * n)
            # A negative per-chain estimate cannot be hidden by pooling.
            variance = tf.where(tf.reduce_all(lrv > 0., axis=0), variance, float("nan"))
            metadata.update(per_chain_long_run_variance=lrv,
                            per_chain_plain_long_run_variance=plain)
        ess = tf.square(sd) / variance
    else:
        raise ValueError("unknown MCSE estimator")
    valid = tf.math.is_finite(variance) & (variance > 0.) & tf.math.is_finite(sd) & (sd > 0.)
    mcse = tf.where(valid, tf.sqrt(variance), float("nan"))
    return {**metadata, "estimate": tf.reduce_mean(values, axis=(0, 1)),
            "posterior_sd": sd, "mcse": mcse, "mean_ess": ess,
            "mcse_sd_ratio": mcse / sd, "valid": valid}


def quantile_precision(samples: Any, probability: float):
    """Indicator ESS mapped through order statistics (Vehtari section 4.4)."""
    values = tf.convert_to_tensor(samples, tf.float64)
    if values.shape.rank != 3 or any(d is None for d in values.shape):
        raise ValueError("quantile samples require static [draw, chain, quantity]")
    if values.shape[0] < 4 or values.shape[1] < 2 or values.shape[2] < 1:
        raise ValueError("quantile precision requires four draws and two chains")
    tf.debugging.assert_all_finite(values, "quantile samples must be finite")
    if not 0. < probability < 1.:
        raise ValueError("quantile probability must be in (0, 1)")
    q = tfp.stats.percentile(values, 100. * probability, axis=(0, 1), interpolation="linear")
    n, m, p = map(int, values.shape)
    half = n // 2
    split = tf.concat((values[:half], values[-half:]), axis=1)
    ess = _cross_chain_ess(tf.cast(split <= q, tf.float64))
    ess_valid = tf.math.is_finite(ess) & (ess > 0.)
    safe_ess = tf.where(ess_valid, ess, tf.ones_like(ess))
    beta = tfp.distributions.Beta(safe_ess * probability + 1., safe_ess * (1. - probability) + 1.)
    normal = tfp.distributions.Normal(tf.constant(0., tf.float64), tf.constant(1., tf.float64))
    bounds = beta.quantile(normal.cdf(tf.constant([[-1.], [1.]], tf.float64)))
    ordered = tf.sort(tf.reshape(values, (n * m, p)), axis=0)
    lower = tf.clip_by_value(tf.cast(tf.floor(bounds[0] * n * m), tf.int32) - 1, 0, n * m - 1)
    upper = tf.clip_by_value(tf.cast(tf.math.ceil(bounds[1] * n * m), tf.int32) - 1, 0, n * m - 1)
    columns = tf.range(p)
    a = tf.gather_nd(ordered, tf.stack((lower, columns), axis=1))
    b = tf.gather_nd(ordered, tf.stack((upper, columns), axis=1))
    valid = ess_valid & tf.reduce_all(tf.math.is_finite(bounds), axis=0) & tf.math.is_finite(a) & tf.math.is_finite(b) & (b > a)
    return {"estimate": q, "mcse": tf.where(valid, (b - a) / 2., float("nan")),
            "valid": valid, "indicator_ess": ess,
            "method": "vehtari_quantile_order_statistics_tfp_positive_pairs"}


def precision_report(samples, names, policy: HMCPrecisionPolicy | None):
    """Host report with explicit missing/invalid precision; JSON-safe values."""
    if policy is None:
        return {"status": "precision_not_requested", "passed": True, "targets": ()}
    names = tuple(names)
    if len(names) != int(samples.shape[-1]) or len(set(names)) != len(names):
        raise ValueError("precision quantity names must uniquely cover the trailing axis")
    if any(t.name not in names for t in policy.targets):
        raise ValueError("unknown precision quantity")
    mean = mean_precision(samples, method=policy.method, batch_size=policy.batch_size,
        min_batches=policy.min_batches, lugsail_r=policy.lugsail_r,
        lugsail_c=policy.lugsail_c, jit_compile=policy.jit_compile)
    quantiles = {t.probability: quantile_precision(samples, t.probability)
                 for t in policy.targets if t.kind == "quantile"}
    rows = []
    for target in policy.targets:
        index = names.index(target.name)
        report = mean if target.kind == "mean" else quantiles[target.probability]
        valid = bool(report["valid"][index])
        mcse = float(report["mcse"][index]) if valid else None
        sd = float(mean["posterior_sd"][index])
        ratio = mcse / sd if valid and sd > 0. else None
        passed = valid and (target.mcse_absolute_max is None or mcse <= target.mcse_absolute_max)
        passed = passed and (target.mcse_sd_ratio_max is None or (ratio is not None and ratio <= target.mcse_sd_ratio_max))
        rows.append({**asdict(target), "estimate": float(report["estimate"][index]),
            "mcse": mcse, "mcse_sd_ratio": ratio, "valid": valid, "passed": bool(passed),
            "estimator": report.get("estimator", report["method"])})
    batch_metadata = {k: v for k, v in mean.items() if k in {
        "batch_size", "small_batch_size", "batch_count", "unused_terminal_draws", "unavailable_reason"}}
    for key in ("per_chain_long_run_variance", "per_chain_plain_long_run_variance"):
        if key in mean:
            batch_metadata[key] = tuple(tuple(float(v) if math.isfinite(float(v)) else None
                for v in tf.unstack(row)) for row in tf.unstack(mean[key]))
    return {"schema": "bayesfilter.hmc_precision_report.v1",
            "status": "precision_met" if all(r["passed"] for r in rows) else "precision_insufficient",
            "passed": all(r["passed"] for r in rows), "targets": tuple(rows),
            "policy": policy.payload(),
            "batch_metadata": batch_metadata}
