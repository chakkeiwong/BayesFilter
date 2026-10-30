"""TensorFlow diagnostics for retained draws from analytic Gaussian mixtures.

The component index is generic and has no distinguished minority component.
Individual intervals are reported separately and are never combined into an
omnibus test or joint p-value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import tensorflow as tf

from bayesfilter.testing.importance_sampling_tf import (
    gaussian_mixture_log_prob_responsibilities_score,
    validate_gaussian_mixture,
)


def gaussian_mixture_moments(
    component_probabilities: Any,
    component_means: Any,
    component_covariances: Any,
) -> Mapping[str, tf.Tensor]:
    """Return exact mean, covariance, and component second moments."""

    probabilities, means, covariances, _ = validate_gaussian_mixture(
        component_probabilities, component_means, component_covariances
    )
    mean = tf.reduce_sum(probabilities[:, tf.newaxis] * means, axis=0)
    centered = means - mean
    covariance = tf.reduce_sum(
        probabilities[:, tf.newaxis, tf.newaxis]
        * (
            covariances
            + centered[:, :, tf.newaxis] * centered[:, tf.newaxis, :]
        ),
        axis=0,
    )
    return {
        "mean": mean,
        "covariance": covariance,
        "component_second_moments": covariances
        + means[:, :, tf.newaxis] * means[:, tf.newaxis, :],
    }


def hard_assignment_transition_counts(
    hard_assignments: Any,
    *,
    component_count: int,
) -> tf.Tensor:
    """Count directed hard-assignment transitions for each chain."""

    labels = tf.convert_to_tensor(hard_assignments, tf.int32)
    if labels.shape.rank != 2:
        raise ValueError("hard_assignments must have shape [draw, chain]")
    if labels.shape[0] is None or int(labels.shape[0]) < 2:
        raise ValueError("transition counts require at least two draws")
    if labels.shape[1] is None or int(labels.shape[1]) < 1:
        raise ValueError("transition counts require at least one chain")
    if isinstance(component_count, bool) or int(component_count) < 1:
        raise ValueError("component_count must be positive")
    tf.debugging.assert_greater_equal(labels, tf.constant(0, tf.int32))
    tf.debugging.assert_less(labels, tf.constant(int(component_count), tf.int32))
    previous = labels[:-1]
    current = labels[1:]
    pair = previous * int(component_count) + current
    return tf.stack(
        [
            tf.reshape(
                tf.math.bincount(
                    pair[:, chain],
                    minlength=int(component_count) ** 2,
                    maxlength=int(component_count) ** 2,
                    dtype=tf.int64,
                ),
                (int(component_count), int(component_count)),
            )
            for chain in range(int(labels.shape[1]))
        ]
    )


def retained_gaussian_mixture_diagnostics(
    physical_samples: Any,
    component_probabilities: Any,
    component_means: Any,
    component_covariances: Any,
    *,
    confidence_z: float = 2.5758293035489004,
) -> Mapping[str, Any]:
    """Compare retained ``[draw, chain, dimension]`` draws with exact truth."""

    probabilities, means, covariances, _ = validate_gaussian_mixture(
        component_probabilities, component_means, component_covariances
    )
    samples = tf.convert_to_tensor(physical_samples, means.dtype)
    component_count = int(means.shape[0])
    dimension = int(means.shape[1])
    if (
        samples.shape.rank != 3
        or samples.shape[-1] != dimension
        or samples.shape[0] is None
        or samples.shape[1] is None
    ):
        raise ValueError(
            "physical_samples must have static shape [draw, chain, dimension]"
        )
    draws = int(samples.shape[0])
    chains = int(samples.shape[1])
    if draws < 4 or chains < 2:
        raise ValueError("retained diagnostics require at least four draws and two chains")
    tf.debugging.assert_all_finite(samples, "retained Gaussian-mixture samples")
    z_value = tf.cast(confidence_z, means.dtype)
    if not math.isfinite(float(confidence_z)) or float(confidence_z) <= 0.0:
        raise ValueError("confidence_z must be finite and positive")

    flat = tf.reshape(samples, (-1, dimension))
    _value, flat_responsibilities, _score = (
        gaussian_mixture_log_prob_responsibilities_score(
            flat, probabilities, means, covariances
        )
    )
    responsibilities = tf.reshape(
        flat_responsibilities, (draws, chains, component_count)
    )
    mass = tf.reduce_mean(responsibilities, axis=(0, 1))
    mass_mcse = tf.stack(
        [_batch_means_mcse(responsibilities[:, :, k]) for k in range(component_count)]
    )
    mass_lower = mass - z_value * mass_mcse
    mass_upper = mass + z_value * mass_mcse
    mass_contains_truth = tf.logical_and(
        mass_lower <= probabilities, probabilities <= mass_upper
    )

    weighted_sum = tf.einsum("dck,dcj->kj", responsibilities, samples)
    responsibility_sum = tf.reduce_sum(responsibilities, axis=(0, 1))
    conditional_mean = weighted_sum / responsibility_sum[:, tf.newaxis]
    centered_by_component = (
        samples[:, :, tf.newaxis, :] - conditional_mean[tf.newaxis, tf.newaxis, :, :]
    )
    conditional_covariance = tf.einsum(
        "dck,dcki,dckj->kij",
        responsibilities,
        centered_by_component,
        centered_by_component,
    ) / responsibility_sum[:, tf.newaxis, tf.newaxis]

    moments = gaussian_mixture_moments(probabilities, means, covariances)
    sample_mean = tf.reduce_mean(samples, axis=(0, 1))
    mean_mcse = tf.stack(
        [_batch_means_mcse(samples[:, :, index]) for index in range(dimension)]
    )
    mean_lower = sample_mean - z_value * mean_mcse
    mean_upper = sample_mean + z_value * mean_mcse
    mean_contains_truth = tf.logical_and(
        mean_lower <= moments["mean"], moments["mean"] <= mean_upper
    )
    centered_truth = samples - moments["mean"]
    covariance_moment = tf.einsum("dci,dcj->dcij", centered_truth, centered_truth)
    covariance_estimate = tf.reduce_mean(covariance_moment, axis=(0, 1))
    covariance_mcse = tf.stack(
        [
            tf.stack(
                [
                    _batch_means_mcse(covariance_moment[:, :, i, j])
                    for j in range(dimension)
                ]
            )
            for i in range(dimension)
        ]
    )
    covariance_lower = covariance_estimate - z_value * covariance_mcse
    covariance_upper = covariance_estimate + z_value * covariance_mcse
    covariance_contains_truth = tf.logical_and(
        covariance_lower <= moments["covariance"],
        moments["covariance"] <= covariance_upper,
    )

    hard = tf.reshape(
        tf.argmax(flat_responsibilities, axis=1, output_type=tf.int32),
        (draws, chains),
    )
    hard_counts = tf.stack(
        [
            tf.math.bincount(
                hard[:, chain],
                minlength=component_count,
                maxlength=component_count,
                dtype=tf.int64,
            )
            for chain in range(chains)
        ]
    )
    transitions = hard_assignment_transition_counts(
        hard, component_count=component_count
    )
    aggregate_transitions = tf.reduce_sum(transitions, axis=0)
    off_diagonal = aggregate_transitions * (
        tf.ones((component_count, component_count), tf.int64)
        - tf.eye(component_count, dtype=tf.int64)
    )
    transition_involvement = tf.reduce_sum(off_diagonal, axis=0) + tf.reduce_sum(
        off_diagonal, axis=1
    )
    component_observed_overall = tf.reduce_sum(hard_counts, axis=0) > 0
    component_observed_per_chain = hard_counts > 0
    component_has_transition = transition_involvement > 0
    transition_requirement_applicable = component_count > 1
    primary_gates = {
        "all_finite": bool(
            tf.reduce_all(tf.math.is_finite(responsibilities)).numpy()
        ),
        "all_components_observed_overall": bool(
            tf.reduce_all(component_observed_overall).numpy()
        ),
        "all_component_mass_intervals_contain_truth": bool(
            tf.reduce_all(mass_contains_truth).numpy()
        ),
        "every_component_involved_in_transition": bool(
            (not transition_requirement_applicable)
            or tf.reduce_all(component_has_transition).numpy()
        ),
    }
    return {
        "schema": "bayesfilter.gaussian_mixture_retained_diagnostics.v1",
        "sample_shape": tuple(int(value) for value in samples.shape),
        "component_count": component_count,
        "dimension": dimension,
        "confidence_level": 0.99,
        "component_mass": mass,
        "component_mass_batch_means_mcse": mass_mcse,
        "component_mass_interval_lower": mass_lower,
        "component_mass_interval_upper": mass_upper,
        "analytic_component_mass": probabilities,
        "component_mass_interval_contains_truth": mass_contains_truth,
        "component_conditional_mean": conditional_mean,
        "analytic_component_conditional_mean": means,
        "component_conditional_covariance": conditional_covariance,
        "analytic_component_conditional_covariance": covariances,
        "sample_mean": sample_mean,
        "analytic_mean": moments["mean"],
        "mean_batch_means_mcse": mean_mcse,
        "mean_interval_contains_truth": mean_contains_truth,
        "covariance_moment_estimate": covariance_estimate,
        "analytic_covariance": moments["covariance"],
        "covariance_moment_batch_means_mcse": covariance_mcse,
        "covariance_interval_contains_truth": covariance_contains_truth,
        "hard_assignment_counts_per_chain": hard_counts,
        "component_observed_overall": component_observed_overall,
        "component_observed_per_chain": component_observed_per_chain,
        "hard_assignment_transition_counts_per_chain": transitions,
        "hard_assignment_transition_counts_aggregate": aggregate_transitions,
        "component_involved_in_transition": component_has_transition,
        "transition_requirement_applicable": transition_requirement_applicable,
        "gates": primary_gates,
        "passed_primary_screens": all(primary_gates.values()),
        "joint_test_performed": False,
        "marginal_moment_role": "explanatory_only_not_joint_veto",
        "conditional_moment_role": "explanatory_only_not_joint_veto",
        "multiple_testing_note": (
            "Component-mass intervals are separate primary diagnostics; moment "
            "intervals are marginal explanatory diagnostics and no joint p-value "
            "or omnibus rejection is computed."
        ),
    }


def _batch_means_mcse(values: tf.Tensor) -> tf.Tensor:
    rows = tf.convert_to_tensor(values)
    if not rows.dtype.is_floating or rows.shape.rank != 2:
        raise ValueError("batch-means input must be a floating [draw, chain] tensor")
    draws = int(rows.shape[0])
    chains = int(rows.shape[1])
    batch_length = max(2, int(math.sqrt(draws)))
    batch_count = draws // batch_length
    if batch_count < 2:
        return tf.math.reduce_std(tf.reshape(rows, (-1,))) / tf.sqrt(
            tf.cast(draws * chains, rows.dtype)
        )
    trimmed = rows[: batch_count * batch_length]
    batched = tf.reshape(trimmed, (batch_count, batch_length, chains))
    batch_means = tf.reduce_mean(batched, axis=1)
    variance = tf.math.reduce_variance(tf.reshape(batch_means, (-1,)))
    return tf.sqrt(variance / tf.cast(batch_count * chains, rows.dtype))


__all__ = [
    "gaussian_mixture_moments",
    "hard_assignment_transition_counts",
    "retained_gaussian_mixture_diagnostics",
]
