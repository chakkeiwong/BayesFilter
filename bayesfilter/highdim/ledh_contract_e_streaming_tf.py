"""Streaming row quotient composed with the canonical Contract E-Chol reset."""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from bayesfilter.highdim.transport_chunk_policy import validate_transport_chunks

from bayesfilter.highdim import ledh_contract_e_reset_tf as cloud_reset
from bayesfilter.highdim.ledh_contract_e_reset_tf import (
    _contract_e_chol_cloud_forward_core,
    _contract_e_chol_cloud_jvp_from_forward_core,
    _contract_e_chol_cloud_jvp_core,
    _contract_e_chol_cloud_vjp_core,
)
from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    _STREAMING_LOG_ZERO,
    _filterflow_streaming_column_log_normalizer,
    _filterflow_streaming_finite_sinkhorn_potentials_jvp_total,
    _filterflow_streaming_finite_sinkhorn_potentials_total_vjp,
    _filterflow_streaming_finite_sinkhorn_potentials_vjp_total,
    _filterflow_streaming_terminal_balance_potential,
    _filterflow_streaming_terminal_balance_potential_jvp,
    _filterflow_streaming_terminal_balance_potential_vjp,
    _filterflow_streaming_transport_from_potentials,
    _filterflow_streaming_transport_from_potentials_jvp,
    _filterflow_streaming_transport_from_potentials_vjp,
    _half_pairwise_squared_cross_jvp,
    _pairwise_squared_cross,
    _slice_axis1_padded_2d,
    _slice_axis1_padded_3d,
)


_FLOAT64_UNIT_ROUNDOFF = 2.220446049250313e-16
_FLOAT32_UNIT_ROUNDOFF = 1.1920928955078125e-7
TV_COLUMN_TOLERANCE = 1.0e-4
MAXIMUM_ROW_ERROR_TOLERANCE = 1.0e-2


def _marginal_roundoff_tolerance(
    particle_count: tf.Tensor,
    column_target: tf.Tensor,
) -> tf.Tensor:
    """Conservative backward-error envelope for streamed marginal sums."""

    dtype = column_target.dtype
    n = tf.cast(particle_count, dtype)
    unit_roundoff_value = (
        _FLOAT32_UNIT_ROUNDOFF if dtype == tf.float32 else _FLOAT64_UNIT_ROUNDOFF
    )
    unit_roundoff = tf.cast(unit_roundoff_value, dtype)
    operation_depth = 16.0 * n
    gamma_n = operation_depth * unit_roundoff / (
        1.0 - operation_depth * unit_roundoff
    )
    maximum_target = tf.reduce_max(tf.abs(column_target), axis=1)
    target_scale = tf.where(
        maximum_target > 1.0,
        maximum_target,
        tf.ones([tf.shape(column_target)[0]], dtype),
    )
    # The 16N operation-depth envelope covers exponentiation inputs, column
    # normalization, streamed reductions, terminal scaling, and row quotient.
    return gamma_n * target_scale


def _balanced_transport_value_core(
    scaled_geometry: tf.Tensor,
    payload: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    particle_count = tf.shape(scaled_geometry)[1]
    float_n = tf.cast(particle_count, scaled_geometry.dtype)
    uniform_log_weight = -tf.math.log(float_n) * tf.ones_like(
        normalized_log_weights
    )
    row_potential, column_potential = (
        _filterflow_streaming_finite_sinkhorn_potentials_total_vjp(
            normalized_log_weights,
            uniform_log_weight,
            scaled_geometry,
            epsilon,
            epsilon0,
            scaling,
            steps=steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
    )
    row_potential = _filterflow_streaming_terminal_balance_potential(
        normalized_log_weights,
        scaled_geometry,
        row_potential,
        epsilon,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    return _filterflow_streaming_transport_from_potentials(
        scaled_geometry,
        payload,
        row_potential,
        column_potential,
        epsilon,
        normalized_log_weights,
        float_n,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )


def _balanced_transport_jvp_core(
    scaled_geometry: tf.Tensor,
    payload: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    scaled_geometry_tangent: tf.Tensor,
    payload_tangent: tf.Tensor,
    normalized_log_weights_tangent: tf.Tensor,
    epsilon0_tangent: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    particle_count = tf.shape(scaled_geometry)[1]
    float_n = tf.cast(particle_count, scaled_geometry.dtype)
    uniform_log_weight = -tf.math.log(float_n) * tf.ones_like(
        normalized_log_weights
    )
    uniform_tangent = tf.zeros_like(normalized_log_weights_tangent)
    (
        row_potential,
        column_potential,
        row_potential_tangent,
        column_potential_tangent,
        _running,
        _running_tangent,
    ) = _filterflow_streaming_finite_sinkhorn_potentials_jvp_total(
        normalized_log_weights,
        uniform_log_weight,
        scaled_geometry,
        normalized_log_weights_tangent,
        uniform_tangent,
        scaled_geometry_tangent,
        epsilon0_tangent,
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    row_potential, row_potential_tangent = (
        _filterflow_streaming_terminal_balance_potential_jvp(
            normalized_log_weights,
            scaled_geometry,
            row_potential,
            normalized_log_weights_tangent,
            scaled_geometry_tangent,
            row_potential_tangent,
            epsilon,
            balance_steps=balance_steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
    )
    return _filterflow_streaming_transport_from_potentials_jvp(
        scaled_geometry,
        payload,
        row_potential,
        column_potential,
        epsilon,
        normalized_log_weights,
        float_n,
        scaled_geometry_tangent,
        payload_tangent,
        row_potential_tangent,
        column_potential_tangent,
        normalized_log_weights_tangent,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )


def _balanced_transport_forward_jvp_state_core(
    scaled_geometry: tf.Tensor,
    payload: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    scaled_geometry_tangent: tf.Tensor,
    payload_tangent: tf.Tensor,
    normalized_log_weights_tangent: tf.Tensor,
    epsilon0_tangent: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, tf.Tensor]:
    """Build one shared primal/JVP OT state for an exact one-tile traversal."""

    particle_count = scaled_geometry.shape[1]
    if particle_count is None:
        raise ValueError("shared OT state requires a static particle count")
    validate_transport_chunks(
        int(particle_count),
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    if row_chunk_size != int(particle_count) or col_chunk_size != int(particle_count):
        raise ValueError("shared same-pass marginal state currently requires K=N")
    dtype = scaled_geometry.dtype
    float_n = tf.cast(particle_count, dtype)
    uniform_log_weight = -tf.math.log(float_n) * tf.ones_like(
        normalized_log_weights
    )
    uniform_tangent = tf.zeros_like(normalized_log_weights_tangent)
    (
        initial_row_potential,
        column_potential,
        initial_row_potential_tangent,
        column_potential_tangent,
        _running,
        _running_tangent,
    ) = _filterflow_streaming_finite_sinkhorn_potentials_jvp_total(
        normalized_log_weights,
        uniform_log_weight,
        scaled_geometry,
        normalized_log_weights_tangent,
        uniform_tangent,
        scaled_geometry_tangent,
        epsilon0_tangent,
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    row_potential, row_potential_tangent = (
        _filterflow_streaming_terminal_balance_potential_jvp(
            normalized_log_weights,
            scaled_geometry,
            initial_row_potential,
            normalized_log_weights_tangent,
            scaled_geometry_tangent,
            initial_row_potential_tangent,
            epsilon,
            balance_steps=balance_steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
    )
    epsilon = tf.reshape(tf.cast(epsilon, dtype), [-1])
    cost = 0.5 * _pairwise_squared_cross(scaled_geometry, scaled_geometry)
    cost_tangent = _half_pairwise_squared_cross_jvp(
        scaled_geometry,
        scaled_geometry,
        scaled_geometry_tangent,
        scaled_geometry_tangent,
    )
    logits = (
        row_potential[:, :, None]
        + column_potential[:, None, :]
        - cost
    ) / epsilon[:, None, None]
    logits_tangent = (
        row_potential_tangent[:, :, None, :]
        + column_potential_tangent[:, None, :, :]
        - cost_tangent
    ) / epsilon[:, None, None, None]
    column_log_normalizer = tf.reduce_logsumexp(logits, axis=1)
    normalized_columns = tf.exp(logits - column_log_normalizer[:, None, :])
    column_log_normalizer_tangent = tf.reduce_sum(
        normalized_columns[:, :, :, None] * logits_tangent,
        axis=1,
    )
    log_transport = (
        logits
        - column_log_normalizer[:, None, :]
        + tf.math.log(float_n)
        + normalized_log_weights[:, None, :]
    )
    log_transport_tangent = (
        logits_tangent
        - column_log_normalizer_tangent[:, None, :, :]
        + normalized_log_weights_tangent[:, None, :, :]
    )
    transport = tf.exp(log_transport)
    transport_tangent = transport[:, :, :, None] * log_transport_tangent
    augmented_numerator = tf.einsum("bij,bjd->bid", transport, payload)
    augmented_tangent = (
        tf.einsum("bijp,bjd->bidp", transport_tangent, payload)
        + tf.einsum("bij,bjdp->bidp", transport, payload_tangent)
    )
    row_mass = augmented_numerator[:, :, -1]
    column_mass = tf.reduce_sum(transport, axis=1)
    post_quotient_column_mass = tf.reduce_sum(
        transport / row_mass[:, :, None], axis=1
    )
    return {
        "augmented_numerator": augmented_numerator,
        "augmented_tangent": augmented_tangent,
        "row_potential": row_potential,
        "column_potential": column_potential,
        "row_potential_tangent": row_potential_tangent,
        "column_potential_tangent": column_potential_tangent,
        "column_mass": column_mass,
        "post_quotient_column_mass": post_quotient_column_mass,
        "sinkhorn_state_constructions": tf.ones([], tf.int32),
        "terminal_balance_state_constructions": tf.ones([], tf.int32),
        "transport_tile_sweeps": tf.ones([], tf.int32),
        "marginal_tile_sweeps": tf.zeros([], tf.int32),
        "diagnostic_solver_reconstructions": tf.zeros([], tf.int32),
    }


def _balanced_transport_pullback_core(
    scaled_geometry: tf.Tensor,
    payload: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    upstream: tf.Tensor,
    *,
    steps: int,
    balance_steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    particle_count = tf.shape(scaled_geometry)[1]
    float_n = tf.cast(particle_count, scaled_geometry.dtype)
    uniform_log_weight = -tf.math.log(float_n) * tf.ones_like(
        normalized_log_weights
    )
    initial_row_potential, column_potential = (
        _filterflow_streaming_finite_sinkhorn_potentials_total_vjp(
            normalized_log_weights,
            uniform_log_weight,
            scaled_geometry,
            epsilon,
            epsilon0,
            scaling,
            steps=steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
    )
    row_potential = _filterflow_streaming_terminal_balance_potential(
        normalized_log_weights,
        scaled_geometry,
        initial_row_potential,
        epsilon,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    (
        geometry_transport_bar,
        payload_bar,
        row_potential_bar,
        column_potential_bar,
        log_weights_transport_bar,
    ) = _filterflow_streaming_transport_from_potentials_vjp(
        scaled_geometry,
        payload,
        row_potential,
        column_potential,
        epsilon,
        normalized_log_weights,
        float_n,
        upstream,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    (
        initial_row_potential_bar,
        log_weights_balance_bar,
        geometry_balance_bar,
    ) = _filterflow_streaming_terminal_balance_potential_vjp(
        normalized_log_weights,
        scaled_geometry,
        initial_row_potential,
        epsilon,
        row_potential_bar,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    (
        log_weights_sinkhorn_bar,
        _uniform_log_weight_bar,
        geometry_sinkhorn_bar,
        epsilon0_bar,
    ) = _filterflow_streaming_finite_sinkhorn_potentials_vjp_total(
        normalized_log_weights,
        uniform_log_weight,
        scaled_geometry,
        initial_row_potential_bar,
        column_potential_bar,
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    return (
        geometry_transport_bar + geometry_balance_bar + geometry_sinkhorn_bar,
        payload_bar,
        log_weights_transport_bar
        + log_weights_balance_bar
        + log_weights_sinkhorn_bar,
        epsilon0_bar,
    )


def _row_quotient_forward_core(
    numerator: tf.Tensor,
    mass: tf.Tensor,
) -> dict[str, tf.Tensor]:
    particles = numerator / mass[:, :, None]
    mass_finite = tf.reduce_all(tf.math.is_finite(mass), axis=1)
    mass_positive = tf.reduce_all(mass > 0, axis=1)
    particles_finite = tf.reduce_all(tf.math.is_finite(particles), axis=[1, 2])
    return {
        "numerator": numerator,
        "mass": mass,
        "particles": particles,
        "mass_finite": mass_finite,
        "mass_positive": mass_positive,
        "particles_finite": particles_finite,
        "valid_chart": mass_finite & mass_positive & particles_finite,
        "minimum_mass": tf.reduce_min(mass, axis=1),
        "row_residual_by_batch": tf.reduce_max(tf.abs(mass - 1.0), axis=1),
    }


def _row_quotient_jvp_core(
    numerator: tf.Tensor,
    mass: tf.Tensor,
    numerator_tangent: tf.Tensor,
    mass_tangent: tf.Tensor,
) -> dict[str, tf.Tensor]:
    forward = _row_quotient_forward_core(numerator, mass)
    denominator = mass[:, :, None, None]
    particles_tangent = (
        numerator_tangent * denominator
        - numerator[:, :, :, None] * mass_tangent[:, :, None, :]
    ) / (denominator * denominator)
    return {
        **forward,
        "numerator_tangent": numerator_tangent,
        "mass_tangent": mass_tangent,
        "particles_tangent": particles_tangent,
    }


def _row_quotient_vjp_core(
    numerator: tf.Tensor,
    mass: tf.Tensor,
    upstream_particles: tf.Tensor,
) -> dict[str, tf.Tensor]:
    forward = _row_quotient_forward_core(numerator, mass)
    numerator_bar = upstream_particles / mass[:, :, None]
    mass_bar = -tf.reduce_sum(
        upstream_particles * forward["particles"], axis=2
    ) / mass
    return {
        **forward,
        "numerator_bar": numerator_bar,
        "mass_bar": mass_bar,
    }


def _augmented_payload(particles: tf.Tensor) -> tf.Tensor:
    return tf.concat([particles, tf.ones_like(particles[:, :, :1])], axis=2)


def _augmented_payload_tangent(particles_tangent: tf.Tensor) -> tf.Tensor:
    return tf.concat(
        [particles_tangent, tf.zeros_like(particles_tangent[:, :, :1, :])],
        axis=2,
    )


def _streaming_column_masses_from_potentials_core(
    scaled_geometry: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    row_mass: tf.Tensor,
    row_potential: tf.Tensor,
    column_potential: tf.Tensor,
    epsilon: tf.Tensor,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Report raw and row-quotient column masses with O(N) retained state."""

    dtype = scaled_geometry.dtype
    batch_size = tf.shape(scaled_geometry)[0]
    particle_count = tf.shape(scaled_geometry)[1]
    float_n = tf.cast(particle_count, dtype)
    log_n = tf.math.log(float_n)
    epsilon = tf.reshape(tf.cast(epsilon, dtype), [-1])
    row_chunk = tf.cast(row_chunk_size, tf.int32)
    col_chunk = tf.cast(col_chunk_size, tf.int32)
    column_log_normalizer = (
        _filterflow_streaming_column_log_normalizer(
            scaled_geometry,
            row_potential,
            column_potential,
            epsilon,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
    )
    num_col_blocks = (particle_count + col_chunk - 1) // col_chunk
    blocks = tf.TensorArray(
        dtype=dtype,
        size=num_col_blocks,
        element_shape=tf.TensorShape([2, None, col_chunk_size]),
    )

    def col_cond(col_start: tf.Tensor, _blocks: tf.TensorArray) -> tf.Tensor:
        return col_start < particle_count

    def col_body(col_start: tf.Tensor, blocks_ta: tf.TensorArray):
        key_block = _slice_axis1_padded_3d(
            scaled_geometry, col_start, col_chunk_size
        )
        column_potential_block = _slice_axis1_padded_2d(
            column_potential,
            col_start,
            col_chunk_size,
            pad_value=float(_STREAMING_LOG_ZERO),
        )
        log_weight_block = _slice_axis1_padded_2d(
            normalized_log_weights,
            col_start,
            col_chunk_size,
            pad_value=float(_STREAMING_LOG_ZERO),
        )
        normalizer_block = _slice_axis1_padded_2d(
            column_log_normalizer,
            col_start,
            col_chunk_size,
            pad_value=0.0,
        )
        column_mass = tf.zeros([batch_size, col_chunk_size], dtype)
        quotient_column_mass = tf.zeros([batch_size, col_chunk_size], dtype)

        def row_cond(
            row_start: tf.Tensor, _mass: tf.Tensor, _quotient_mass: tf.Tensor
        ) -> tf.Tensor:
            return row_start < particle_count

        def row_body(
            row_start: tf.Tensor, mass: tf.Tensor, quotient_mass: tf.Tensor
        ):
            query_block = _slice_axis1_padded_3d(
                scaled_geometry, row_start, row_chunk_size
            )
            row_potential_block = _slice_axis1_padded_2d(
                row_potential,
                row_start,
                row_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            row_mass_block = _slice_axis1_padded_2d(
                row_mass,
                row_start,
                row_chunk_size,
                pad_value=1.0,
            )
            cost = 0.5 * _pairwise_squared_cross(
                query_block, key_block
            )
            log_transport = (
                row_potential_block[:, :, None]
                + column_potential_block[:, None, :]
                - cost
            ) / epsilon[:, None, None]
            log_transport = (
                log_transport
                - normalizer_block[:, None, :]
                + log_n
                + log_weight_block[:, None, :]
            )
            transport = tf.exp(log_transport)
            return (
                row_start + row_chunk,
                mass + tf.reduce_sum(transport, axis=1),
                quotient_mass
                + tf.reduce_sum(
                    transport / row_mass_block[:, :, None], axis=1
                ),
            )

        _, column_mass, quotient_column_mass = tf.while_loop(
            row_cond,
            row_body,
            loop_vars=(
                tf.constant(0, tf.int32),
                column_mass,
                quotient_column_mass,
            ),
            maximum_iterations=(particle_count + row_chunk - 1) // row_chunk,
        )
        blocks_ta = blocks_ta.write(
            col_start // col_chunk,
            tf.stack([column_mass, quotient_column_mass], axis=0),
        )
        return col_start + col_chunk, blocks_ta

    _, blocks = tf.while_loop(
        col_cond,
        col_body,
        loop_vars=(tf.constant(0, tf.int32), blocks),
        maximum_iterations=num_col_blocks,
    )
    stacked = tf.transpose(blocks.stack(), [1, 2, 0, 3])
    flat = tf.reshape(
        stacked, [2, batch_size, num_col_blocks * col_chunk]
    )
    return flat[0, :, :particle_count], flat[1, :, :particle_count]


def _streaming_column_mass_from_potentials_core(
    scaled_geometry: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    row_potential: tf.Tensor,
    column_potential: tf.Tensor,
    epsilon: tf.Tensor,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tf.Tensor:
    """Compatibility wrapper for historical raw-column-mass diagnostics."""

    row_mass = tf.ones(tf.shape(normalized_log_weights), scaled_geometry.dtype)
    raw_mass, _ = _streaming_column_masses_from_potentials_core(
        scaled_geometry,
        normalized_log_weights,
        row_mass,
        row_potential,
        column_potential,
        epsilon,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    return raw_mass


def _streaming_marginal_diagnostics_core(
    scaled_geometry: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    row_mass: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, tf.Tensor]:
    particle_count = tf.shape(scaled_geometry)[1]
    dtype = scaled_geometry.dtype
    float_n = tf.cast(particle_count, dtype)
    uniform_log_weight = -tf.math.log(float_n) * tf.ones_like(
        normalized_log_weights
    )
    row_potential, column_potential = (
        _filterflow_streaming_finite_sinkhorn_potentials_total_vjp(
            normalized_log_weights,
            uniform_log_weight,
            scaled_geometry,
            epsilon,
            epsilon0,
            scaling,
            steps=steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
    )
    row_potential = _filterflow_streaming_terminal_balance_potential(
        normalized_log_weights,
        scaled_geometry,
        row_potential,
        epsilon,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    column_mass, quotient_column_mass = _streaming_column_masses_from_potentials_core(
        scaled_geometry,
        normalized_log_weights,
        row_mass,
        row_potential,
        column_potential,
        epsilon,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    row_target = tf.ones_like(row_mass)
    column_target = float_n * tf.exp(normalized_log_weights)
    row_signed_residual = row_mass - row_target
    column_signed_residual = column_mass - column_target
    quotient_column_signed_residual = quotient_column_mass - column_target
    maximum_column_target = tf.reduce_max(tf.abs(column_target), axis=1)
    column_residual_scale = tf.where(
        maximum_column_target > 1.0,
        maximum_column_target,
        tf.ones_like(maximum_column_target),
    )
    return {
        "row_target": row_target,
        "row_signed_residual": row_signed_residual,
        "column_mass": column_mass,
        "column_target": column_target,
        "column_signed_residual": column_signed_residual,
        "post_quotient_column_mass": quotient_column_mass,
        "post_quotient_column_signed_residual": quotient_column_signed_residual,
        "maximum_row_absolute_residual": tf.reduce_max(
            tf.abs(row_signed_residual), axis=1
        ),
        "maximum_column_absolute_residual": tf.reduce_max(
            tf.abs(column_signed_residual), axis=1
        ),
        "maximum_post_quotient_column_absolute_residual": tf.reduce_max(
            tf.abs(quotient_column_signed_residual), axis=1
        ),
        "row_residual_scale": tf.ones([tf.shape(row_mass)[0]], dtype),
        "column_residual_scale": column_residual_scale,
        "marginal_roundoff_tolerance": _marginal_roundoff_tolerance(
            particle_count, column_target
        ),
    }


def _streaming_row_quotient_forward_core(
    scaled_geometry: tf.Tensor,
    particles: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, tf.Tensor]:
    augmented_numerator, helper_row_residual = _balanced_transport_value_core(
        scaled_geometry,
        _augmented_payload(particles),
        normalized_log_weights,
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    result = _row_quotient_forward_core(
        augmented_numerator[:, :, :-1],
        augmented_numerator[:, :, -1],
    )
    marginal_diagnostics = _streaming_marginal_diagnostics_core(
        scaled_geometry,
        normalized_log_weights,
        result["mass"],
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    marginal_tolerance = marginal_diagnostics["marginal_roundoff_tolerance"]
    marginal_valid = (
        marginal_diagnostics["maximum_row_absolute_residual"]
        <= marginal_tolerance
    ) & (
        marginal_diagnostics["maximum_post_quotient_column_absolute_residual"]
        <= marginal_tolerance
    )
    return {
        **result,
        **marginal_diagnostics,
        "marginal_valid": marginal_valid,
        "helper_row_residual": helper_row_residual,
    }


def _streaming_row_quotient_jvp_core(
    scaled_geometry: tf.Tensor,
    particles: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    scaled_geometry_tangent: tf.Tensor,
    particles_tangent: tf.Tensor,
    normalized_log_weights_tangent: tf.Tensor,
    epsilon0_tangent: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, tf.Tensor]:
    augmented_numerator, augmented_tangent, helper_row_residual = (
        _balanced_transport_jvp_core(
            scaled_geometry,
            _augmented_payload(particles),
            normalized_log_weights,
            scaled_geometry_tangent,
            _augmented_payload_tangent(particles_tangent),
            normalized_log_weights_tangent,
            epsilon0_tangent,
            epsilon,
            epsilon0,
            scaling,
            steps=steps,
            balance_steps=balance_steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
    )
    result = _row_quotient_jvp_core(
        augmented_numerator[:, :, :-1],
        augmented_numerator[:, :, -1],
        augmented_tangent[:, :, :-1, :],
        augmented_tangent[:, :, -1, :],
    )
    return {**result, "helper_row_residual": helper_row_residual}


def _streaming_row_quotient_forward_jvp_core(
    scaled_geometry: tf.Tensor,
    particles: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    scaled_geometry_tangent: tf.Tensor,
    particles_tangent: tf.Tensor,
    normalized_log_weights_tangent: tf.Tensor,
    epsilon0_tangent: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, tf.Tensor]:
    """Return quotient value, JVP, and marginals from one shared OT state."""

    state = _balanced_transport_forward_jvp_state_core(
        scaled_geometry,
        _augmented_payload(particles),
        normalized_log_weights,
        scaled_geometry_tangent,
        _augmented_payload_tangent(particles_tangent),
        normalized_log_weights_tangent,
        epsilon0_tangent,
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    augmented_numerator = state["augmented_numerator"]
    augmented_tangent = state["augmented_tangent"]
    result = _row_quotient_jvp_core(
        augmented_numerator[:, :, :-1],
        augmented_numerator[:, :, -1],
        augmented_tangent[:, :, :-1, :],
        augmented_tangent[:, :, -1, :],
    )
    dtype = scaled_geometry.dtype
    particle_count = tf.shape(scaled_geometry)[1]
    float_n = tf.cast(particle_count, dtype)
    row_target = tf.ones_like(result["mass"])
    column_target = float_n * tf.exp(normalized_log_weights)
    row_signed_residual = result["mass"] - row_target
    column_signed_residual = state["column_mass"] - column_target
    post_quotient_column_signed_residual = (
        state["post_quotient_column_mass"] - column_target
    )
    maximum_row_error = tf.reduce_max(tf.abs(row_signed_residual), axis=1)
    tv_column_error = tf.reduce_sum(
        tf.abs(post_quotient_column_signed_residual), axis=1
    ) / (2.0 * float_n)
    maximum_target = tf.reduce_max(tf.abs(column_target), axis=1)
    column_residual_scale = tf.where(
        maximum_target > 1.0, maximum_target, tf.ones_like(maximum_target)
    )
    marginal_valid = (
        maximum_row_error <= tf.cast(MAXIMUM_ROW_ERROR_TOLERANCE, dtype)
    ) & (tv_column_error <= tf.cast(TV_COLUMN_TOLERANCE, dtype))
    return {
        **result,
        "row_target": row_target,
        "row_signed_residual": row_signed_residual,
        "column_mass": state["column_mass"],
        "column_target": column_target,
        "column_signed_residual": column_signed_residual,
        "post_quotient_column_mass": state["post_quotient_column_mass"],
        "post_quotient_column_signed_residual": post_quotient_column_signed_residual,
        "maximum_row_absolute_residual": maximum_row_error,
        "maximum_column_absolute_residual": tf.reduce_max(
            tf.abs(column_signed_residual), axis=1
        ),
        "maximum_post_quotient_column_absolute_residual": tf.reduce_max(
            tf.abs(post_quotient_column_signed_residual), axis=1
        ),
        "tv_column_error": tv_column_error,
        "maximum_row_error": maximum_row_error,
        "tv_column_tolerance": tf.fill(
            tf.shape(maximum_row_error), tf.cast(TV_COLUMN_TOLERANCE, dtype)
        ),
        "maximum_row_error_tolerance": tf.fill(
            tf.shape(maximum_row_error),
            tf.cast(MAXIMUM_ROW_ERROR_TOLERANCE, dtype),
        ),
        "row_residual_scale": tf.ones_like(maximum_row_error),
        "column_residual_scale": column_residual_scale,
        "marginal_roundoff_tolerance": _marginal_roundoff_tolerance(
            particle_count, column_target
        ),
        "marginal_valid": marginal_valid,
        "work": {
            "sinkhorn_state_constructions": state["sinkhorn_state_constructions"],
            "terminal_balance_state_constructions": state[
                "terminal_balance_state_constructions"
            ],
            "transport_tile_sweeps": state["transport_tile_sweeps"],
            "marginal_tile_sweeps": state["marginal_tile_sweeps"],
            "diagnostic_solver_reconstructions": state[
                "diagnostic_solver_reconstructions"
            ],
        },
    }


def _streaming_row_quotient_vjp_core(
    scaled_geometry: tf.Tensor,
    particles: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    upstream_particles: tf.Tensor,
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, tf.Tensor]:
    forward = _streaming_row_quotient_forward_core(
        scaled_geometry,
        particles,
        normalized_log_weights,
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    quotient_vjp = _row_quotient_vjp_core(
        forward["numerator"], forward["mass"], upstream_particles
    )
    augmented_upstream = tf.concat(
        [quotient_vjp["numerator_bar"], quotient_vjp["mass_bar"][:, :, None]],
        axis=2,
    )
    (
        scaled_geometry_bar,
        augmented_payload_bar,
        normalized_log_weights_bar,
        epsilon0_bar,
    ) = _balanced_transport_pullback_core(
        scaled_geometry,
        _augmented_payload(particles),
        normalized_log_weights,
        epsilon,
        epsilon0,
        scaling,
        augmented_upstream,
        steps=steps,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    return {
        **quotient_vjp,
        "scaled_geometry": scaled_geometry_bar,
        "particles": augmented_payload_bar[:, :, :-1],
        "constant_payload": augmented_payload_bar[:, :, -1],
        "normalized_log_weights": normalized_log_weights_bar,
        "epsilon0": epsilon0_bar,
        "augmented_upstream": augmented_upstream,
    }


def _contract_e_streaming_forward_core(
    scaled_geometry: tf.Tensor,
    source_particles: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    normalized_weights: tf.Tensor,
    residual_design: tf.Tensor,
    ridge: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, Any]:
    particle_count = source_particles.shape[1]
    if particle_count is None:
        raise ValueError("Contract E streaming requires static particle count")
    validate_transport_chunks(
        int(particle_count),
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    quotient = _streaming_row_quotient_forward_core(
        scaled_geometry,
        source_particles,
        normalized_log_weights,
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    reset = _contract_e_chol_cloud_forward_core(
        source_particles,
        normalized_weights,
        quotient["particles"],
        residual_design,
        ridge,
    )
    return {"particles": reset["particles"], "quotient": quotient, "reset": reset}


def _contract_e_streaming_jvp_core(
    scaled_geometry: tf.Tensor,
    source_particles: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    normalized_weights: tf.Tensor,
    residual_design: tf.Tensor,
    ridge: tf.Tensor,
    scaled_geometry_tangent: tf.Tensor,
    source_particles_tangent: tf.Tensor,
    normalized_log_weights_tangent: tf.Tensor,
    normalized_weights_tangent: tf.Tensor,
    residual_design_tangent: tf.Tensor,
    ridge_tangent: tf.Tensor,
    epsilon0_tangent: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, Any]:
    particle_count = source_particles.shape[1]
    if particle_count is None:
        raise ValueError("Contract E streaming JVP requires static particle count")
    validate_transport_chunks(
        int(particle_count),
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    quotient = _streaming_row_quotient_jvp_core(
        scaled_geometry,
        source_particles,
        normalized_log_weights,
        scaled_geometry_tangent,
        source_particles_tangent,
        normalized_log_weights_tangent,
        epsilon0_tangent,
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    parameter_count = tf.shape(source_particles_tangent)[3]
    output_signature = tf.TensorSpec(
        shape=source_particles.shape, dtype=source_particles.dtype
    )

    def one_direction(index: tf.Tensor) -> tf.Tensor:
        return _contract_e_chol_cloud_jvp_core(
            source_particles,
            normalized_weights,
            quotient["particles"],
            residual_design,
            ridge,
            source_particles_tangent[:, :, :, index],
            normalized_weights_tangent[:, :, index],
            quotient["particles_tangent"][:, :, :, index],
            residual_design_tangent[:, :, :, index],
            ridge_tangent[:, index],
        )["particles"]

    reset_particles = tf.transpose(
        tf.map_fn(
            one_direction,
            tf.range(parameter_count),
            fn_output_signature=output_signature,
            parallel_iterations=1,
        ),
        [1, 2, 3, 0],
    )
    return {
        "particles": reset_particles,
        "quotient": quotient,
        "reset": {"particles": reset_particles},
    }


def _contract_e_streaming_forward_jvp_core(
    scaled_geometry: tf.Tensor,
    source_particles: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    normalized_weights: tf.Tensor,
    residual_design: tf.Tensor,
    ridge: tf.Tensor,
    scaled_geometry_tangent: tf.Tensor,
    source_particles_tangent: tf.Tensor,
    normalized_log_weights_tangent: tf.Tensor,
    normalized_weights_tangent: tf.Tensor,
    residual_design_tangent: tf.Tensor,
    ridge_tangent: tf.Tensor,
    epsilon0_tangent: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, Any]:
    """Evaluate canonical Contract E value and all tangents from shared state."""

    quotient = _streaming_row_quotient_forward_jvp_core(
        scaled_geometry,
        source_particles,
        normalized_log_weights,
        scaled_geometry_tangent,
        source_particles_tangent,
        normalized_log_weights_tangent,
        epsilon0_tangent,
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    reset = _contract_e_chol_cloud_forward_core(
        source_particles,
        normalized_weights,
        quotient["particles"],
        residual_design,
        ridge,
    )
    parameter_count = tf.shape(source_particles_tangent)[3]
    output_signature = tf.TensorSpec(
        shape=source_particles.shape, dtype=source_particles.dtype
    )

    def one_direction(index: tf.Tensor) -> tf.Tensor:
        return _contract_e_chol_cloud_jvp_from_forward_core(
            reset,
            source_particles,
            normalized_weights,
            quotient["particles"],
            residual_design,
            ridge,
            source_particles_tangent[:, :, :, index],
            normalized_weights_tangent[:, :, index],
            quotient["particles_tangent"][:, :, :, index],
            residual_design_tangent[:, :, :, index],
            ridge_tangent[:, index],
        )["particles"]

    reset_tangent = tf.transpose(
        tf.map_fn(
            one_direction,
            tf.range(parameter_count),
            fn_output_signature=output_signature,
            parallel_iterations=1,
        ),
        [1, 2, 3, 0],
    )
    return {
        "particles": reset["particles"],
        "particles_tangent": reset_tangent,
        "quotient": quotient,
        "reset": reset,
        "work": quotient["work"],
    }


def _contract_e_streaming_vjp_core(
    scaled_geometry: tf.Tensor,
    source_particles: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    normalized_weights: tf.Tensor,
    residual_design: tf.Tensor,
    ridge: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    upstream_particles: tf.Tensor,
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, Any]:
    particle_count = source_particles.shape[1]
    if particle_count is None:
        raise ValueError("Contract E streaming VJP requires static particle count")
    validate_transport_chunks(
        int(particle_count),
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    quotient_forward = _streaming_row_quotient_forward_core(
        scaled_geometry,
        source_particles,
        normalized_log_weights,
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    reset = _contract_e_chol_cloud_vjp_core(
        source_particles,
        normalized_weights,
        quotient_forward["particles"],
        residual_design,
        ridge,
        upstream_particles,
    )
    quotient = _streaming_row_quotient_vjp_core(
        scaled_geometry,
        source_particles,
        normalized_log_weights,
        epsilon,
        epsilon0,
        scaling,
        reset["transported_particles"],
        steps=steps,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    normalized_log_weights_moment = (
        normalized_weights * reset["normalized_weights"]
    )
    return {
        "source_particles": reset["source_particles"] + quotient["particles"],
        "source_particles_direct": reset["source_particles"],
        "source_particles_transport": quotient["particles"],
        "normalized_weights_probability": reset["normalized_weights"],
        "normalized_log_weights": (
            quotient["normalized_log_weights"] + normalized_log_weights_moment
        ),
        "normalized_log_weights_moment": normalized_log_weights_moment,
        "normalized_log_weights_transport": quotient["normalized_log_weights"],
        "scaled_geometry": quotient["scaled_geometry"],
        "residual_design": reset["residual_design"],
        "ridge": reset["ridge"],
        "epsilon0": quotient["epsilon0"],
        "constant_payload": quotient["constant_payload"],
        "quotient": quotient,
        "reset": reset,
    }


@tf.function(jit_compile=True, reduce_retracing=True)
def contract_e_streaming_forward_tf(
    scaled_geometry: tf.Tensor,
    source_particles: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    normalized_weights: tf.Tensor,
    residual_design: tf.Tensor,
    ridge: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tf.Tensor:
    """Return canonical Contract E particles from row-quotient transport."""

    return _contract_e_streaming_forward_core(
        scaled_geometry,
        source_particles,
        normalized_log_weights,
        normalized_weights,
        residual_design,
        ridge,
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )["particles"]


@tf.function(jit_compile=True, reduce_retracing=True)
def contract_e_streaming_jvp_tf(
    scaled_geometry: tf.Tensor,
    source_particles: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    normalized_weights: tf.Tensor,
    residual_design: tf.Tensor,
    ridge: tf.Tensor,
    scaled_geometry_tangent: tf.Tensor,
    source_particles_tangent: tf.Tensor,
    normalized_log_weights_tangent: tf.Tensor,
    normalized_weights_tangent: tf.Tensor,
    residual_design_tangent: tf.Tensor,
    ridge_tangent: tf.Tensor,
    epsilon0_tangent: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tf.Tensor:
    """Return the local Contract E streaming output tangent."""

    return _contract_e_streaming_jvp_core(
        scaled_geometry,
        source_particles,
        normalized_log_weights,
        normalized_weights,
        residual_design,
        ridge,
        scaled_geometry_tangent,
        source_particles_tangent,
        normalized_log_weights_tangent,
        normalized_weights_tangent,
        residual_design_tangent,
        ridge_tangent,
        epsilon0_tangent,
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )["particles"]


@tf.function(jit_compile=True, reduce_retracing=True)
def contract_e_streaming_vjp_tf(
    scaled_geometry: tf.Tensor,
    source_particles: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    normalized_weights: tf.Tensor,
    residual_design: tf.Tensor,
    ridge: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    upstream_particles: tf.Tensor,
    *,
    steps: int,
    balance_steps: int = 0,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, tf.Tensor]:
    """Return separated direct and transport cotangents for local composition."""

    result = _contract_e_streaming_vjp_core(
        scaled_geometry,
        source_particles,
        normalized_log_weights,
        normalized_weights,
        residual_design,
        ridge,
        epsilon,
        epsilon0,
        scaling,
        upstream_particles,
        steps=steps,
        balance_steps=balance_steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    return {
        name: result[name]
        for name in (
            "source_particles",
            "source_particles_direct",
            "source_particles_transport",
            "normalized_weights_probability",
            "normalized_log_weights",
            "normalized_log_weights_moment",
            "normalized_log_weights_transport",
            "scaled_geometry",
            "residual_design",
            "ridge",
            "epsilon0",
            "constant_payload",
        )
    }
