"""Null-calibrated joint predictive-consistency diagnostics.

This module calibrates a joint nonconformity threshold from independent
same-parameter outputs. It does not estimate or validate an observed-data
score; it only provides a finite-sample predictive coverage construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import tensorflow as tf
import tensorflow_probability as tfp


DTYPE = tf.float64


@dataclass(frozen=True)
class SVDGeometry:
    center: tf.Tensor
    singular_values: tf.Tensor
    left_vectors: tf.Tensor
    rank: int
    omitted_variance: tf.Tensor
    threshold: tf.Tensor

    def squared_distance(self, values: tf.Tensor) -> tf.Tensor:
        observations = tf.cast(tf.convert_to_tensor(values), DTYPE)
        if observations.shape.rank != 2 or observations.shape[1] != self.center.shape[0]:
            raise ValueError("values must have shape [batch, dimension]")
        projected = tf.matmul(observations - self.center[None, :], self.left_vectors[:, : self.rank])
        return tf.reduce_sum(tf.square(projected) / self.singular_values[None, : self.rank], axis=1)

    def distance(self, values: tf.Tensor) -> tf.Tensor:
        return tf.sqrt(tf.maximum(self.squared_distance(values), tf.constant(0.0, DTYPE)))


def fit_svd_geometry(values: tf.Tensor, *, min_rank: int = 2) -> SVDGeometry:
    observations = tf.cast(tf.convert_to_tensor(values), DTYPE)
    if observations.shape.rank != 2 or observations.shape[0] is None or observations.shape[1] is None:
        raise ValueError("values must have static shape [batch, dimension]")
    count = int(observations.shape[0])
    dimension = int(observations.shape[1])
    if count < 2:
        raise ValueError("at least two null-fit paths are required")
    center = tf.reduce_mean(observations, axis=0)
    centered = observations - center[None, :]
    covariance = tf.matmul(centered, centered, transpose_a=True) / tf.cast(count - 1, DTYPE)
    singular_values, left_vectors, _ = tf.linalg.svd(covariance, full_matrices=False, compute_uv=True)
    threshold = tf.cast(tf.experimental.numpy.finfo(DTYPE.as_numpy_dtype).eps, DTYPE) * tf.maximum(tf.reduce_max(singular_values), tf.constant(0.0, DTYPE)) * tf.cast(dimension, DTYPE)
    retained = singular_values > threshold
    rank = int(tf.reduce_sum(tf.cast(retained, tf.int32)).numpy())
    if rank < int(min_rank):
        raise ValueError(f"null-fit covariance rank {rank} is below min_rank={min_rank}")
    omitted = tf.reduce_sum(tf.boolean_mask(singular_values, ~retained))
    return SVDGeometry(
        center=tf.identity(center),
        singular_values=tf.identity(singular_values),
        left_vectors=tf.identity(left_vectors),
        rank=rank,
        omitted_variance=tf.identity(omitted),
        threshold=tf.identity(threshold),
    )


def conformal_order_rank(count: int, *, coverage: float = 0.95, tolerance_confidence: float | None = None) -> int:
    """Return a one-based order rank for a split-conformal/tolerance bound."""
    n = int(count)
    if n < 1 or not 0.0 < float(coverage) < 1.0:
        raise ValueError("count must be positive and coverage must lie in (0,1)")
    if tolerance_confidence is None:
        rank = math.ceil(float(coverage) * (n + 1))
    else:
        confidence = float(tolerance_confidence)
        if not 0.0 < confidence < 1.0:
            raise ValueError("tolerance_confidence must lie in (0,1)")
        # Distribution-free one-sided tolerance rank. The smallest rank k
        # satisfying P{Binomial(n, coverage) <= k-1} >= confidence is used.
        distribution = tfp.distributions.Binomial(total_count=n, probs=tf.cast(coverage, DTYPE))
        rank = next(
            (k for k in range(1, n + 1) if float(distribution.cdf(tf.cast(k - 1, DTYPE)).numpy()) >= confidence),
            n,
        )
    return max(1, min(n, int(rank)))


def conformal_threshold(scores: tf.Tensor, *, coverage: float = 0.95, tolerance_confidence: float | None = None) -> tf.Tensor:
    values = tf.sort(tf.cast(tf.reshape(tf.convert_to_tensor(scores), [-1]), DTYPE))
    if values.shape[0] is None or int(values.shape[0]) < 1:
        raise ValueError("scores must be non-empty")
    if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
        raise ValueError("scores contain non-finite values")
    rank = conformal_order_rank(int(values.shape[0]), coverage=coverage, tolerance_confidence=tolerance_confidence)
    return tf.identity(values[rank - 1])


def clopper_pearson_lower(failures: int, trials: int, *, confidence: float = 0.95) -> float:
    f = int(failures)
    n = int(trials)
    if f < 0 or f > n or n < 1 or not 0.0 < float(confidence) < 1.0:
        raise ValueError("invalid binomial inputs")
    if f == 0:
        return 0.0
    value = tfp.distributions.Beta(tf.cast(f, DTYPE), tf.cast(n - f + 1, DTYPE)).quantile(tf.cast(1.0 - confidence, DTYPE))
    return float(value.numpy())


def audit_failure(failures: int, trials: int, *, target_failure: float = 0.05, confidence: float = 0.95) -> dict[str, object]:
    lower = clopper_pearson_lower(failures, trials, confidence=confidence)
    return {
        "failures": int(failures),
        "trials": int(trials),
        "failure_rate": float(failures) / float(trials),
        "one_sided_lower_bound": lower,
        "target_failure": float(target_failure),
        "falsified": lower > float(target_failure),
    }


def zero_mean_max_t_diagnostic(values: tf.Tensor, *, bootstrap_replicates: int = 1000, confidence: float = 0.95, seed: tuple[int, int] = (731, 911)) -> dict[str, object]:
    """Report a simultaneous multiplier-bootstrap diagnostic for mean zero."""
    observations = tf.cast(tf.convert_to_tensor(values), DTYPE)
    if observations.shape.rank != 2 or observations.shape[0] is None or observations.shape[1] is None:
        raise ValueError("values must have static shape [batch, dimension]")
    count = int(observations.shape[0])
    if count < 2:
        raise ValueError("at least two values are required")
    mean = tf.reduce_mean(observations, axis=0)
    centered = observations - mean[None, :]
    standard_error = tf.math.reduce_std(observations, axis=0) / tf.sqrt(tf.cast(count, DTYPE))
    safe_se = tf.maximum(standard_error, tf.constant(tf.experimental.numpy.finfo(DTYPE.as_numpy_dtype).eps, DTYPE))
    multipliers = tf.random.stateless_normal([int(bootstrap_replicates), count], seed=seed, dtype=DTYPE)
    bootstrap_means = tf.matmul(multipliers, centered) / tf.cast(count, DTYPE)
    bootstrap_t = tf.reduce_max(tf.abs(bootstrap_means / safe_se[None, :]), axis=1)
    critical = conformal_threshold(bootstrap_t, coverage=confidence)
    observed_t = tf.reduce_max(tf.abs(mean / safe_se))
    return {
        "mean": tf.identity(mean),
        "standard_error": tf.identity(standard_error),
        "critical_value": tf.identity(critical),
        "observed_max_t": tf.identity(observed_t),
        "contains_zero": bool(observed_t.numpy() <= critical.numpy()),
        "bootstrap_replicates": int(bootstrap_replicates),
    }


__all__ = [
    "DTYPE",
    "SVDGeometry",
    "audit_failure",
    "clopper_pearson_lower",
    "conformal_order_rank",
    "conformal_threshold",
    "fit_svd_geometry",
    "zero_mean_max_t_diagnostic",
]
