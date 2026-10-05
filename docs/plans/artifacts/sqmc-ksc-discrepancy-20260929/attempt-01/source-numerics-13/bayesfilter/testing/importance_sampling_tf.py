"""TensorFlow diagnostics for corrected Gaussian-mixture importance sampling.

The routines are diagnostic authorities for known proposals and target values.
They do not discover modes, define an inference default, or turn a local-Laplace
proposal into a posterior approximation without the explicit importance weights.
"""

from __future__ import annotations

from typing import Any, Mapping

import tensorflow as tf


def validate_gaussian_mixture(
    component_probabilities: Any,
    component_means: Any,
    component_covariances: Any,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Validate and return probabilities, means, covariances, and Cholesky factors."""

    means = tf.convert_to_tensor(component_means)
    if not means.dtype.is_floating or means.shape.rank != 2:
        raise ValueError("component_means must be a floating [component, dimension] tensor")
    probabilities = tf.convert_to_tensor(component_probabilities, means.dtype)
    covariances = tf.convert_to_tensor(component_covariances, means.dtype)
    component_count = means.shape[0]
    dimension = means.shape[1]
    if component_count is None or dimension is None:
        raise ValueError("mixture component and dimension sizes must be static")
    if probabilities.shape != (component_count,):
        raise ValueError("component_probabilities shape must match means")
    if covariances.shape != (component_count, dimension, dimension):
        raise ValueError("component_covariances shape must match means")
    tf.debugging.assert_all_finite(probabilities, "mixture probabilities")
    tf.debugging.assert_all_finite(means, "mixture means")
    tf.debugging.assert_all_finite(covariances, "mixture covariances")
    tf.debugging.assert_positive(probabilities, "mixture probabilities must be positive")
    tf.debugging.assert_near(
        tf.reduce_sum(probabilities),
        tf.constant(1.0, means.dtype),
        atol=tf.constant(1.0e-12, means.dtype),
        message="mixture probabilities must sum to one",
    )
    tf.debugging.assert_near(
        covariances,
        tf.linalg.matrix_transpose(covariances),
        atol=tf.constant(1.0e-12, means.dtype),
        message="mixture covariances must be symmetric",
    )
    factors = tf.linalg.cholesky(covariances)
    tf.debugging.assert_all_finite(factors, "mixture covariance Cholesky factors")
    return probabilities, means, covariances, factors


def gaussian_mixture_log_prob(
    values: Any,
    component_probabilities: Any,
    component_means: Any,
    component_covariances: Any,
) -> tf.Tensor:
    """Evaluate a normalized full-covariance Gaussian mixture log density."""

    probabilities, means, _covariances, factors = validate_gaussian_mixture(
        component_probabilities, component_means, component_covariances
    )
    rows = tf.convert_to_tensor(values, means.dtype)
    if rows.shape.rank != 2 or rows.shape[1] != means.shape[1]:
        raise ValueError("values must have shape [row, dimension]")
    centered = rows[:, tf.newaxis, :] - means[tf.newaxis, :, :]
    solved = tf.linalg.triangular_solve(
        factors[tf.newaxis, :, :, :],
        centered[:, :, :, tf.newaxis],
        lower=True,
    )[..., 0]
    quadratic = tf.reduce_sum(tf.square(solved), axis=2)
    log_determinant = 2.0 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(factors)), axis=1
    )
    dimension = tf.cast(tf.shape(means)[1], means.dtype)
    log_normalizer = dimension * tf.math.log(
        tf.constant(2.0 * 3.141592653589793, means.dtype)
    ) + log_determinant
    component_log_prob = -0.5 * (quadratic + log_normalizer[tf.newaxis, :])
    return tf.reduce_logsumexp(
        tf.math.log(probabilities)[tf.newaxis, :] + component_log_prob,
        axis=1,
    )


def gaussian_mixture_log_prob_responsibilities_score(
    values: Any,
    component_probabilities: Any,
    component_means: Any,
    component_covariances: Any,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return normalized log density, component responsibilities, and score."""

    probabilities, means, _covariances, factors = validate_gaussian_mixture(
        component_probabilities, component_means, component_covariances
    )
    rows = tf.convert_to_tensor(values, means.dtype)
    if rows.shape.rank != 2 or rows.shape[1] != means.shape[1]:
        raise ValueError("values must have shape [row, dimension]")
    centered = rows[:, tf.newaxis, :] - means[tf.newaxis, :, :]
    solved = tf.linalg.triangular_solve(
        factors[tf.newaxis, :, :, :],
        centered[:, :, :, tf.newaxis],
        lower=True,
    )
    quadratic = tf.reduce_sum(tf.square(solved[..., 0]), axis=2)
    log_determinant = 2.0 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(factors)), axis=1
    )
    dimension = tf.cast(tf.shape(means)[1], means.dtype)
    log_normalizer = dimension * tf.math.log(
        tf.constant(2.0 * 3.141592653589793, means.dtype)
    ) + log_determinant
    component_log_prob = -0.5 * (quadratic + log_normalizer[tf.newaxis, :])
    joint = tf.math.log(probabilities)[tf.newaxis, :] + component_log_prob
    log_prob = tf.reduce_logsumexp(joint, axis=1)
    responsibilities = tf.exp(joint - log_prob[:, tf.newaxis])
    component_score = -tf.linalg.cholesky_solve(
        factors[tf.newaxis, :, :, :], centered[:, :, :, tf.newaxis]
    )[..., 0]
    score = tf.reduce_sum(responsibilities[:, :, tf.newaxis] * component_score, axis=1)
    return log_prob, responsibilities, score


def sample_gaussian_mixture(
    sample_count: int,
    component_probabilities: Any,
    component_means: Any,
    component_covariances: Any,
    *,
    seed: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor]:
    """Draw stateless mixture rows and return their component labels."""

    if isinstance(sample_count, bool) or int(sample_count) <= 0:
        raise ValueError("sample_count must be a positive integer")
    probabilities, means, _covariances, factors = validate_gaussian_mixture(
        component_probabilities, component_means, component_covariances
    )
    split = tf.random.experimental.stateless_split(tf.constant(seed, tf.int32), 2)
    labels = tf.cast(
        tf.reshape(
            tf.random.stateless_categorical(
                tf.math.log(probabilities)[tf.newaxis, :],
                int(sample_count),
                seed=split[0],
            ),
            (-1,),
        ),
        tf.int32,
    )
    noise = tf.random.stateless_normal(
        (int(sample_count), int(means.shape[1])),
        seed=split[1],
        dtype=means.dtype,
    )
    selected_means = tf.gather(means, labels)
    selected_factors = tf.gather(factors, labels)
    rows = selected_means + tf.linalg.matvec(selected_factors, noise)
    return rows, labels


def self_normalized_importance_diagnostics(
    target_log_prob: Any,
    proposal_log_prob: Any,
    negative_region: Any,
) -> Mapping[str, tf.Tensor]:
    """Return corrected weights, ESS, and one weighted region probability."""

    target = tf.convert_to_tensor(target_log_prob)
    proposal = tf.convert_to_tensor(proposal_log_prob, target.dtype)
    indicator = tf.convert_to_tensor(negative_region, tf.bool)
    if target.shape.rank != 1 or proposal.shape != target.shape or indicator.shape != target.shape:
        raise ValueError("target, proposal, and indicator must be equal-length vectors")
    tf.debugging.assert_all_finite(target, "importance target log density")
    tf.debugging.assert_all_finite(proposal, "importance proposal log density")
    log_weights = target - proposal
    normalized = tf.nn.softmax(log_weights)
    ess = tf.math.reciprocal(tf.reduce_sum(tf.square(normalized)))
    estimate = tf.reduce_sum(normalized * tf.cast(indicator, target.dtype))
    entropy = -tf.reduce_sum(
        tf.math.xlogy(normalized, tf.maximum(normalized, tf.constant(1.0e-300, target.dtype)))
    )
    return {
        "log_weights": log_weights,
        "normalized_weights": normalized,
        "effective_sample_size": ess,
        "effective_sample_size_fraction": ess / tf.cast(tf.size(target), target.dtype),
        "maximum_normalized_weight": tf.reduce_max(normalized),
        "negative_region_probability": estimate,
        "weight_entropy": entropy,
        "log_normalizer_ratio_estimate": tf.reduce_logsumexp(log_weights)
        - tf.math.log(tf.cast(tf.size(target), target.dtype)),
    }


def independent_batch_interval(
    batch_estimates: Any,
    *,
    critical_value: float = 2.364624251,
) -> Mapping[str, tf.Tensor]:
    """Return a two-sided t interval across independent batch estimates.

    The default critical value is the measured Student-t 97.5th percentile for
    seven degrees of freedom and therefore applies to the planned eight batches.
    Callers with another batch count must supply the matching reviewed value.
    """

    values = tf.convert_to_tensor(batch_estimates)
    if not values.dtype.is_floating or values.shape.rank != 1:
        raise ValueError("batch_estimates must be a floating vector")
    if values.shape[0] is None or int(values.shape[0]) < 2:
        raise ValueError("at least two static independent batches are required")
    tf.debugging.assert_all_finite(values, "independent batch estimates")
    mean = tf.reduce_mean(values)
    count = tf.cast(tf.size(values), values.dtype)
    sample_variance = tf.reduce_sum(tf.square(values - mean)) / (count - 1.0)
    standard_deviation = tf.sqrt(sample_variance)
    standard_error = standard_deviation / tf.sqrt(count)
    half_width = tf.cast(critical_value, values.dtype) * standard_error
    return {
        "batch_count": tf.size(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "critical_value": tf.cast(critical_value, values.dtype),
        "half_width": half_width,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }
