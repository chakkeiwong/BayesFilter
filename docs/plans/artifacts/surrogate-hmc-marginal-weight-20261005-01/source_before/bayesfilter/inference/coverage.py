"""Joint credible-region coverage diagnostics.

Phase 0 Task 0.3. Implements the Option A decision: joint 95% Mahalanobis
region rather than a conjunction of independent marginal intervals.

Coverage is an EXPLANATORY diagnostic in this program, not a promotion
criterion. The primary criterion is reference-sampler agreement between the
exact-force and damped-force arms on the same frozen target. Coverage of the
true parameter confounds sampler correctness with finite-N bias and with a
property of the particular dataset, so it cannot certify Corollary 5.2. It is
reported because it is informative, not because it decides promotion.
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp


def joint_mahalanobis_coverage(
    samples: tf.Tensor,
    true_theta: tf.Tensor,
    alpha: float = 0.05,
    ridge: float = 1.0e-10,
) -> dict[str, float | bool]:
    """
    Check whether true_theta lies inside the joint (1-alpha) credible region.

    Uses the Mahalanobis distance under the empirical posterior covariance and
    compares it to a chi-squared quantile with parameter_dim degrees of
    freedom.

    Why joint rather than marginal: requiring all P marginal 95% intervals to
    cover simultaneously has probability 0.95**P for a correct sampler, which
    is 0.774 at P=5. That criterion rejects a correct sampler about one run in
    four. The joint region has coverage 1-alpha by construction for a
    correctly-sampled Gaussian-ish posterior.

    Args:
        samples: [num_samples, parameter_dim] posterior draws, chains pooled.
        true_theta: [parameter_dim] generating parameter.
        alpha: Tail mass outside the region. 0.05 gives a 95% region.
        ridge: Relative ridge added to the covariance diagonal before
            factorization, as a fraction of the mean diagonal entry. Guards a
            near-singular empirical covariance when num_samples is close to
            parameter_dim.

    Returns:
        covers: whether true_theta is inside the region.
        mahalanobis_squared: the D^2 statistic.
        threshold: the chi-squared (1-alpha) quantile at parameter_dim df.
        p_value: 1 - F_chi2(D^2), the tail mass beyond the observed distance.
        parameter_dim, num_samples: shape provenance.

    Raises:
        ValueError: on rank mismatch, shape mismatch, non-finite input, or
            too few samples to estimate a covariance.
    """
    samples = tf.convert_to_tensor(samples)
    true_theta = tf.cast(tf.convert_to_tensor(true_theta), samples.dtype)

    if samples.shape.rank != 2:
        raise ValueError(
            f"samples must have rank 2 [num_samples, parameter_dim], got rank "
            f"{samples.shape.rank}"
        )
    if true_theta.shape.rank != 1:
        raise ValueError(
            f"true_theta must have rank 1 [parameter_dim], got rank "
            f"{true_theta.shape.rank}"
        )

    num_samples = int(samples.shape[0])
    parameter_dim = int(samples.shape[1])

    if int(true_theta.shape[0]) != parameter_dim:
        raise ValueError(
            f"true_theta dimension {int(true_theta.shape[0])} does not match "
            f"samples parameter_dim {parameter_dim}"
        )
    if num_samples <= parameter_dim:
        raise ValueError(
            f"need num_samples > parameter_dim to estimate a covariance, got "
            f"{num_samples} samples for dimension {parameter_dim}"
        )
    if not bool(tf.reduce_all(tf.math.is_finite(samples)).numpy()):
        raise ValueError("samples contain non-finite values")
    if not bool(tf.reduce_all(tf.math.is_finite(true_theta)).numpy()):
        raise ValueError("true_theta contains non-finite values")

    posterior_mean = tf.reduce_mean(samples, axis=0)
    covariance = tfp.stats.covariance(samples, sample_axis=0, event_axis=-1)

    # Scale-aware ridge: an absolute floor would silently expire as the
    # posterior scale changes with the model.
    mean_diagonal = tf.reduce_mean(tf.linalg.diag_part(covariance))
    ridge_scale = tf.cast(ridge, samples.dtype) * mean_diagonal
    covariance = covariance + ridge_scale * tf.eye(
        parameter_dim, dtype=samples.dtype
    )

    chol = tf.linalg.cholesky(covariance)
    if not bool(tf.reduce_all(tf.math.is_finite(chol)).numpy()):
        raise ValueError(
            "posterior covariance is not positive definite after ridging; "
            "cannot form a Mahalanobis region"
        )

    delta = tf.reshape(true_theta - posterior_mean, [parameter_dim, 1])
    solved = tf.linalg.cholesky_solve(chol, delta)
    mahalanobis_squared = tf.reduce_sum(delta * solved)

    chi2 = tfp.distributions.Chi2(
        df=tf.cast(parameter_dim, samples.dtype)
    )
    threshold = chi2.quantile(tf.cast(1.0 - alpha, samples.dtype))
    p_value = tf.cast(1.0, samples.dtype) - chi2.cdf(mahalanobis_squared)

    return {
        "covers": bool((mahalanobis_squared < threshold).numpy()),
        "mahalanobis_squared": float(mahalanobis_squared.numpy()),
        "threshold": float(threshold.numpy()),
        "p_value": float(p_value.numpy()),
        "parameter_dim": parameter_dim,
        "num_samples": num_samples,
        "alpha": float(alpha),
    }


def marginal_coverage_diagnostic(
    samples: tf.Tensor,
    true_theta: tf.Tensor,
    alpha: float = 0.05,
) -> dict[str, object]:
    """
    Per-parameter marginal interval coverage, reported alongside the joint region.

    Included because a joint pass with several marginal misses localizes which
    coordinates are off, which the single joint number hides. This is strictly
    explanatory: the conjunction of marginals is NOT used as a criterion, for
    the 0.95**P reason documented in joint_mahalanobis_coverage.

    Returns per-coordinate coverage flags, the count covered, and the
    probability that a correct sampler would cover all P marginals, so the
    reader can see how weak that conjunction is.
    """
    samples = tf.convert_to_tensor(samples)
    true_theta = tf.cast(tf.convert_to_tensor(true_theta), samples.dtype)

    parameter_dim = int(samples.shape[1])

    lower = tfp.stats.percentile(
        samples, 100.0 * (alpha / 2.0), axis=0, interpolation="linear"
    )
    upper = tfp.stats.percentile(
        samples, 100.0 * (1.0 - alpha / 2.0), axis=0, interpolation="linear"
    )

    covered = tf.logical_and(true_theta >= lower, true_theta <= upper)
    covered_list = [bool(value) for value in covered.numpy().tolist()]

    return {
        "covered_per_parameter": covered_list,
        "count_covered": int(sum(covered_list)),
        "parameter_dim": parameter_dim,
        "all_marginals_covered": all(covered_list),
        "probability_all_covered_if_correct": float((1.0 - alpha) ** parameter_dim),
        "lower": [float(v) for v in lower.numpy().tolist()],
        "upper": [float(v) for v in upper.numpy().tolist()],
    }


__all__ = ["joint_mahalanobis_coverage", "marginal_coverage_diagnostic"]
