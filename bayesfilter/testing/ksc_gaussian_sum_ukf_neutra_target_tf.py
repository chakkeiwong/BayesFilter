"""Batch-native mass-preserving Gaussian-sum repair for the KSC target."""

from __future__ import annotations

from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
    source_chart_physical_parameters,
)


_LOG_TWO_PI = tf.constant(1.8378770664093453, tf.float64)
_INVALID_LOG_WEIGHT = tf.constant(-1.0e100, tf.float64)
_MASS_FLOOR = tf.constant(1.0e-300, tf.float64)


def ksc_gaussian_sum_ukf_likelihood_value_score_status(
    theta: Any,
    *,
    transformed_observations: tf.Tensor,
    mixture_weights: tf.Tensor,
    mixture_means: tf.Tensor,
    mixture_variances: tf.Tensor,
    component_cap: int,
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    """Evaluate a deterministic reduced Gaussian-sum UKF approximation.

    Each affine KSC observation component uses the exact scalar Kalman/UKF
    update. Unlike the historical route, multiple posterior components are
    retained. Components outside the fixed cap are assigned to a retained
    top-weight center and moment-merged, preserving all normalized mass.
    """

    values = tf.convert_to_tensor(theta, tf.float64)
    if values.shape.rank != 2 or values.shape[-1] != 2:
        raise ValueError("theta must have shape [batch, 2]")
    if int(component_cap) < 7:
        raise ValueError("component_cap must be at least the KSC component count")
    observations = tf.reshape(
        tf.convert_to_tensor(transformed_observations, tf.float64), (-1,)
    )
    weights = tf.reshape(tf.convert_to_tensor(mixture_weights, tf.float64), (-1,))
    locations = tf.reshape(tf.convert_to_tensor(mixture_means, tf.float64), (-1,))
    variances = tf.reshape(
        tf.convert_to_tensor(mixture_variances, tf.float64), (-1,)
    )
    if int(weights.shape[0]) != 7:
        raise ValueError("the KSC repair requires exactly seven mixture components")

    with tf.GradientTape() as tape:
        tape.watch(values)
        value, diagnostics = _ksc_gaussian_sum_value(
            values,
            observations=observations,
            mixture_weights=weights,
            mixture_means=locations,
            mixture_variances=variances,
            component_cap=int(component_cap),
        )
    score = tape.gradient(value, values)
    if score is None:
        raise RuntimeError("TensorFlow did not produce the Gaussian-sum score")
    finite = tf.logical_and(
        tf.math.is_finite(value), tf.reduce_all(tf.math.is_finite(score), axis=1)
    )
    finite = tf.logical_and(finite, diagnostics["minimum_component_variance"] > 0.0)
    return value, score, {
        "status_code": tf.where(
            finite, tf.zeros_like(value, tf.int32), tf.ones_like(value, tf.int32)
        ),
        "valid_pre_regularized_score": finite,
        "minimum_component_variance": diagnostics["minimum_component_variance"],
        "minimum_retained_mass_fraction": diagnostics[
            "minimum_retained_mass_fraction"
        ],
        "minimum_premerge_top_weight_mass_fraction": diagnostics[
            "minimum_premerge_top_weight_mass_fraction"
        ],
        "maximum_active_component_count": diagnostics[
            "maximum_active_component_count"
        ],
        "component_cap": tf.fill(tf.shape(value), tf.cast(component_cap, tf.int32)),
    }


def _ksc_gaussian_sum_value(
    theta: tf.Tensor,
    *,
    observations: tf.Tensor,
    mixture_weights: tf.Tensor,
    mixture_means: tf.Tensor,
    mixture_variances: tf.Tensor,
    component_cap: int,
) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
    batch_size = tf.shape(theta)[0]
    gamma, beta = source_chart_physical_parameters(theta)
    initial_variance = tf.math.reciprocal(1.0 - tf.square(gamma))
    means = tf.zeros([batch_size, component_cap], tf.float64)
    component_variances = tf.concat(
        (
            initial_variance[:, None],
            tf.ones([batch_size, component_cap - 1], tf.float64),
        ),
        axis=1,
    )
    log_weights = tf.concat(
        (
            tf.zeros([batch_size, 1], tf.float64),
            tf.fill([batch_size, component_cap - 1], _INVALID_LOG_WEIGHT),
        ),
        axis=1,
    )
    value = tf.zeros([batch_size], tf.float64)
    minimum_variance = initial_variance
    minimum_retained_mass = tf.ones([batch_size], tf.float64)
    minimum_premerge_mass = tf.ones([batch_size], tf.float64)
    maximum_active = tf.ones([batch_size], tf.int32)
    log_mixture_weights = tf.math.log(mixture_weights)

    def condition(index, *_):
        return index < tf.shape(observations)[0]

    def body(
        index,
        current_means,
        current_variances,
        current_log_weights,
        current_value,
        current_minimum_variance,
        current_minimum_retained_mass,
        current_minimum_premerge_mass,
        current_maximum_active,
    ):
        positive_time = index > 0
        predicted_means = tf.where(
            positive_time, gamma[:, None] * current_means, current_means
        )
        predicted_variances = tf.where(
            positive_time,
            tf.square(gamma)[:, None] * current_variances + 1.0,
            current_variances,
        )
        innovation_variances = (
            predicted_variances[:, :, None]
            + mixture_variances[None, None, :]
        )
        observation_offsets = (
            2.0 * tf.math.log(beta)[:, None, None]
            + mixture_means[None, None, :]
        )
        innovation = (
            observations[index]
            - predicted_means[:, :, None]
            - observation_offsets
        )
        expanded_log_weights = (
            current_log_weights[:, :, None]
            + log_mixture_weights[None, None, :]
            - 0.5
            * (
                _LOG_TWO_PI
                + tf.math.log(innovation_variances)
                + tf.square(innovation) / innovation_variances
            )
        )
        gains = predicted_variances[:, :, None] / innovation_variances
        posterior_means = predicted_means[:, :, None] + gains * innovation
        posterior_variances = (
            predicted_variances[:, :, None]
            * mixture_variances[None, None, :]
            / innovation_variances
        )
        flat_log_weights = tf.reshape(expanded_log_weights, [batch_size, -1])
        flat_means = tf.reshape(posterior_means, [batch_size, -1])
        flat_variances = tf.reshape(posterior_variances, [batch_size, -1])
        log_increment = tf.reduce_logsumexp(flat_log_weights, axis=1)
        normalized_flat = flat_log_weights - log_increment[:, None]
        center_log_weights, retained_indices = tf.math.top_k(
            normalized_flat, k=component_cap, sorted=True
        )
        center_means = tf.gather(
            flat_means, retained_indices, axis=1, batch_dims=1
        )
        center_variances = tf.gather(
            flat_variances, retained_indices, axis=1, batch_dims=1
        )
        premerge_mass = tf.exp(tf.reduce_logsumexp(center_log_weights, axis=1))

        # Assign every expanded component to a retained center, then preserve
        # its first two moments. The discrete assignment is deterministic; the
        # selected branch remains differentiable in its continuous quantities.
        sorted_center_indices = tf.argsort(
            center_means, axis=1, direction="ASCENDING", stable=True
        )
        sorted_center_means = tf.gather(
            center_means, sorted_center_indices, axis=1, batch_dims=1
        )
        insertion = tf.searchsorted(
            sorted_center_means, flat_means, side="left", out_type=tf.int32
        )
        left_position = tf.maximum(insertion - 1, 0)
        right_position = tf.minimum(insertion, component_cap - 1)
        left_mean = tf.gather(
            sorted_center_means, left_position, axis=1, batch_dims=1
        )
        right_mean = tf.gather(
            sorted_center_means, right_position, axis=1, batch_dims=1
        )
        selected_position = tf.where(
            tf.abs(flat_means - left_mean) <= tf.abs(flat_means - right_mean),
            left_position,
            right_position,
        )
        assignment = tf.gather(
            sorted_center_indices, selected_position, axis=1, batch_dims=1
        )
        flat_weights = tf.exp(normalized_flat)
        batch_offset = tf.range(batch_size, dtype=tf.int32)[:, None] * component_cap
        global_assignment = tf.reshape(assignment + batch_offset, (-1,))
        segment_count = batch_size * component_cap
        retained_mass = tf.reshape(
            tf.math.unsorted_segment_sum(
                tf.reshape(flat_weights, (-1,)),
                global_assignment,
                segment_count,
            ),
            [batch_size, component_cap],
        )
        active = retained_mass > 0.0
        safe_mass = tf.where(active, retained_mass, tf.ones_like(retained_mass))
        retained_means = tf.reshape(
            tf.math.unsorted_segment_sum(
                tf.reshape(flat_weights * flat_means, (-1,)),
                global_assignment,
                segment_count,
            ),
            [batch_size, component_cap],
        ) / safe_mass
        retained_second = tf.reshape(
            tf.math.unsorted_segment_sum(
                tf.reshape(
                    flat_weights * (flat_variances + tf.square(flat_means)), (-1,)
                ),
                global_assignment,
                segment_count,
            ),
            [batch_size, component_cap],
        ) / safe_mass
        retained_variances = retained_second - tf.square(retained_means)
        retained_means = tf.where(active, retained_means, center_means)
        retained_variances = tf.where(
            active, retained_variances, center_variances
        )
        retained_log_weights = tf.where(
            active,
            tf.math.log(tf.maximum(retained_mass, _MASS_FLOOR)),
            _INVALID_LOG_WEIGHT,
        )
        total_retained_mass = tf.reduce_sum(retained_mass, axis=1)
        active_count = tf.reduce_sum(
            tf.cast(active, tf.int32), axis=1
        )
        return (
            index + 1,
            retained_means,
            retained_variances,
            retained_log_weights,
            current_value + log_increment,
            tf.minimum(
                current_minimum_variance,
                tf.reduce_min(retained_variances, axis=1),
            ),
            tf.minimum(current_minimum_retained_mass, total_retained_mass),
            tf.minimum(current_minimum_premerge_mass, premerge_mass),
            tf.maximum(current_maximum_active, active_count),
        )

    result = tf.while_loop(
        condition,
        body,
        (
            tf.constant(0, tf.int32),
            means,
            component_variances,
            log_weights,
            value,
            minimum_variance,
            minimum_retained_mass,
            minimum_premerge_mass,
            maximum_active,
        ),
        maximum_iterations=int(observations.shape[0]),
        parallel_iterations=1,
    )
    return result[4], {
        "minimum_component_variance": result[5],
        "minimum_retained_mass_fraction": result[6],
        "minimum_premerge_top_weight_mass_fraction": result[7],
        "maximum_active_component_count": result[8],
    }


__all__ = ["ksc_gaussian_sum_ukf_likelihood_value_score_status"]
