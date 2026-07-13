"""TensorFlow filterflow-style annealed regularized transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorflow as tf


DEFAULT_DTYPE = tf.float64
DTYPE = DEFAULT_DTYPE
DEFAULT_STREAMING_CHUNK_SIZE = 1024
_STREAMING_LOG_ZERO = -1.0e30
TRANSPORT_AD_MODES = {
    "stabilized",
    "diff-scale",
    "diff-keys",
    "diff-potentials",
    "full",
}


@dataclass(frozen=True)
class AnnealedTransportTFResult:
    particles: tf.Tensor
    log_weights: tf.Tensor
    transport_matrix: tf.Tensor
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class AnnealedTransportWarmstartStateTF:
    a_y: tf.Tensor
    b_x: tf.Tensor
    a_x: tf.Tensor
    b_y: tf.Tensor
    valid_mask: tf.Tensor


def build_annealed_transport_warmstart_state_tf(
    a_y: tf.Tensor,
    b_x: tf.Tensor,
    a_x: tf.Tensor,
    b_y: tf.Tensor,
    valid_mask: tf.Tensor,
) -> AnnealedTransportWarmstartStateTF:
    return AnnealedTransportWarmstartStateTF(
        a_y=tf.cast(a_y, DTYPE),
        b_x=tf.cast(b_x, DTYPE),
        a_x=tf.cast(a_x, DTYPE),
        b_y=tf.cast(b_y, DTYPE),
        valid_mask=tf.reshape(tf.cast(valid_mask, tf.bool), [-1]),
    )


def build_annealed_transport_coldstart_state_tf(
    scaled_particles: tf.Tensor,
    log_weights: tf.Tensor,
    *,
    epsilon: float | tf.Tensor,
    scaling: float | tf.Tensor,
    transport_plan_mode: str = "streaming",
    row_chunk_size: int = DEFAULT_STREAMING_CHUNK_SIZE,
    col_chunk_size: int = DEFAULT_STREAMING_CHUNK_SIZE,
) -> AnnealedTransportWarmstartStateTF:
    x = tf.cast(scaled_particles, DTYPE)
    logw = tf.cast(log_weights, DTYPE)
    if len(x.shape) != 3 or len(logw.shape) != 2:
        raise ValueError("scaled_particles must be [B,N,D] and log_weights must be [B,N]")
    batch_size = tf.shape(x)[0]
    num_particles = tf.shape(x)[1]
    if transport_plan_mode not in {"dense", "streaming"}:
        raise ValueError("transport_plan_mode must be 'dense' or 'streaming'")
    float_n = tf.cast(num_particles, DTYPE)
    log_n = tf.math.log(float_n)
    uniform_log_weight = -log_n * tf.ones_like(logw)
    epsilon_tensor = _epsilon_per_batch(tf.convert_to_tensor(epsilon, dtype=DTYPE), batch_size)
    scaling_tensor = tf.reshape(tf.cast(scaling, DTYPE), [])
    particles_diameter = _filterflow_exact_max_min(x, x)
    epsilon_0 = particles_diameter ** 2
    if transport_plan_mode == "streaming":
        a_y = _filterflow_streaming_softmin(
            epsilon_0,
            x,
            x,
            logw,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        b_x = _filterflow_streaming_softmin(
            epsilon_0,
            x,
            x,
            uniform_log_weight,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        a_x = _filterflow_streaming_softmin(
            epsilon_0,
            x,
            x,
            logw,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        b_y = _filterflow_streaming_softmin(
            epsilon_0,
            x,
            x,
            uniform_log_weight,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
    else:
        cost = _filterflow_exact_cost(x, x)
        a_y = _filterflow_exact_softmin(epsilon_0, cost, logw)
        b_x = _filterflow_exact_softmin(epsilon_0, cost, uniform_log_weight)
        a_x = _filterflow_exact_softmin(epsilon_0, cost, logw)
        b_y = _filterflow_exact_softmin(epsilon_0, cost, uniform_log_weight)
    return build_annealed_transport_warmstart_state_tf(
        a_y,
        b_x,
        a_x,
        b_y,
        tf.ones([batch_size], dtype=tf.bool),
    )


def annealed_transport_resample_tf(
    particles: tf.Tensor,
    log_weights: tf.Tensor,
    *,
    epsilon: float = 0.5,
    scaling: float = 0.9,
    convergence_threshold: float = 1e-3,
    max_iterations: int = 100,
    ess_mask: tf.Tensor | None = None,
    transport_gradient_mode: str = "filterflow_clipped",
    application_mode: str = "active_rows_only",
    transport_plan_mode: str = "dense",
    transport_ad_mode: str = "stabilized",
    row_chunk_size: int = DEFAULT_STREAMING_CHUNK_SIZE,
    col_chunk_size: int = DEFAULT_STREAMING_CHUNK_SIZE,
    warmstart_state: AnnealedTransportWarmstartStateTF | None = None,
) -> AnnealedTransportTFResult:
    """Apply filterflow RegularisedTransform-style annealed transport.

    The mathematical target is the executable filterflow RegularisedTransform
    transport, not the local fixed-target Sinkhorn comparator.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not 0.0 < scaling <= 1.0:
        raise ValueError("scaling must be in (0, 1]")
    if convergence_threshold <= 0.0:
        raise ValueError("convergence_threshold must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if transport_gradient_mode not in {"filterflow_clipped", "filterflow_custom_op", "raw"}:
        raise ValueError(
            "transport_gradient_mode must be 'filterflow_clipped', "
            "'filterflow_custom_op', or 'raw'"
        )
    if application_mode not in {"active_rows_only", "filterflow_all_rows"}:
        raise ValueError("application_mode must be 'active_rows_only' or 'filterflow_all_rows'")
    if transport_plan_mode not in {"dense", "streaming"}:
        raise ValueError("transport_plan_mode must be 'dense' or 'streaming'")
    _validate_transport_ad_mode(transport_ad_mode)
    if transport_plan_mode == "streaming" and transport_gradient_mode != "raw":
        raise ValueError("streaming transport currently supports transport_gradient_mode='raw' only")
    if row_chunk_size <= 0 or col_chunk_size <= 0:
        raise ValueError("row_chunk_size and col_chunk_size must be positive")

    original_particle_rank = len(particles.shape)
    original_weight_rank = len(log_weights.shape)
    x = tf.cast(particles, DTYPE)
    logw = tf.cast(log_weights, DTYPE)
    if original_particle_rank == 2:
        x = x[None, :, :]
    if original_weight_rank == 1:
        logw = logw[None, :]
    if len(x.shape) != 3 or len(logw.shape) != 2:
        raise ValueError("particles must be [N,D] or [B,N,D]; log_weights must be [N] or [B,N]")
    if int(x.shape[1] or 0) and int(logw.shape[1] or 0) and x.shape[1] != logw.shape[1]:
        raise ValueError("particles and log_weights must agree on particle count")

    batch_size = tf.shape(x)[0]
    num_particles = tf.shape(x)[1]
    if ess_mask is None:
        mask = tf.ones([batch_size], dtype=tf.bool)
    else:
        mask = tf.reshape(tf.cast(ess_mask, tf.bool), [-1])
    log_weight_normalization_residual = tf.reduce_max(
        tf.abs(tf.reduce_logsumexp(logw, axis=1))
    )
    uniform_log = tf.fill(
        [batch_size, num_particles],
        -tf.math.log(tf.cast(num_particles, DTYPE)),
    )
    out_particles = tf.identity(x)
    out_log_weights = tf.where(mask[:, None], uniform_log, logw)
    if transport_plan_mode == "dense":
        full_transport = tf.zeros([batch_size, num_particles, num_particles], dtype=DTYPE)
    else:
        full_transport = tf.zeros([batch_size, 0, 0], dtype=DTYPE)
    triggered_count = tf.reduce_sum(tf.cast(mask, DTYPE))
    skipped_count = tf.cast(batch_size, DTYPE) - triggered_count

    if application_mode == "filterflow_all_rows":
        transported, transport_matrix, active_diag = _transport_active(
            x,
            logw,
            epsilon=epsilon,
            scaling=scaling,
            convergence_threshold=convergence_threshold,
            max_iterations=max_iterations,
            transport_gradient_mode=transport_gradient_mode,
            transport_plan_mode=transport_plan_mode,
            transport_ad_mode=transport_ad_mode,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            warmstart_state=warmstart_state,
        )
        out_particles = tf.where(mask[:, None, None], transported, x)
        if transport_plan_mode == "dense":
            full_transport = transport_matrix
        max_row_residual = active_diag["max_row_residual"]
        max_column_residual = active_diag["max_column_residual"]
        max_cost_scale = active_diag["max_cost_scale"]
        min_cost_scale = active_diag["min_cost_scale"]
        max_iterations_used = active_diag["max_iterations_used"]
        finite_transport = active_diag["finite_transport"]
        finite_particles = active_diag["finite_particles"]
    elif bool(tf.reduce_any(mask).numpy()):
        active_x = tf.boolean_mask(x, mask)
        active_logw = tf.boolean_mask(logw, mask)
        transported, transport_matrix, active_diag = _transport_active(
            active_x,
            active_logw,
            epsilon=epsilon,
            scaling=scaling,
            convergence_threshold=convergence_threshold,
            max_iterations=max_iterations,
            transport_gradient_mode=transport_gradient_mode,
            transport_plan_mode=transport_plan_mode,
            transport_ad_mode=transport_ad_mode,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        active_indices = tf.where(mask)
        out_particles = tf.tensor_scatter_nd_update(out_particles, active_indices, transported)
        if transport_plan_mode == "dense":
            full_transport = tf.tensor_scatter_nd_update(
                full_transport,
                active_indices,
                transport_matrix,
            )
        max_row_residual = active_diag["max_row_residual"]
        max_column_residual = active_diag["max_column_residual"]
        max_cost_scale = active_diag["max_cost_scale"]
        min_cost_scale = active_diag["min_cost_scale"]
        max_iterations_used = active_diag["max_iterations_used"]
        finite_transport = active_diag["finite_transport"]
        finite_particles = active_diag["finite_particles"]
    else:
        max_row_residual = tf.constant(0.0, DTYPE)
        max_column_residual = tf.constant(0.0, DTYPE)
        max_cost_scale = tf.constant(0.0, DTYPE)
        min_cost_scale = tf.constant(0.0, DTYPE)
        max_iterations_used = tf.constant(0.0, DTYPE)
        finite_transport = True
        finite_particles = True

    diagnostics = {
        "component_id": "filterflow_style_annealed_transport_tf",
        "reference_algorithm": "canonical_executable_filterflow_regularised_transform",
        "mathematical_object": "annealed_regularized_transport_transform",
        "fixed_target_sinkhorn_status": "not_this_algorithm_local_comparator_only",
        "epsilon": float(epsilon),
        "scaling": float(scaling),
        "convergence_threshold": float(convergence_threshold),
        "max_iterations": int(max_iterations),
        "transport_gradient_mode": transport_gradient_mode,
        "transport_ad_mode": transport_ad_mode,
        "application_mode": application_mode,
        "transport_plan_mode": transport_plan_mode,
        "warmstart_used": bool(warmstart_state is not None),
        "transport_matrix_materialized": transport_plan_mode == "dense",
        "row_chunk_size": int(row_chunk_size),
        "col_chunk_size": int(col_chunk_size),
        "transport_backward_rule": _transport_backward_rule(transport_gradient_mode),
        "triggered_rows": _float(triggered_count),
        "skipped_rows": _float(skipped_count),
        "max_row_residual": _float(max_row_residual),
        "max_column_residual": _float(max_column_residual),
        "max_cost_scale": _float(max_cost_scale),
        "min_cost_scale": _float(min_cost_scale),
        "max_iterations_used": _float(max_iterations_used),
        "finite_transport": bool(finite_transport),
        "finite_particles": bool(finite_particles),
        "max_log_weight_normalization_residual": _float(log_weight_normalization_residual),
        "backend": "tensorflow",
        "resampling_status": "annealed_transport_triggered_rows_only",
    }
    if not diagnostics["finite_transport"] or not diagnostics["finite_particles"]:
        raise FloatingPointError("annealed transport emitted non-finite values")
    result_particles = out_particles[0] if original_particle_rank == 2 else out_particles
    result_log_weights = out_log_weights[0] if original_weight_rank == 1 else out_log_weights
    result_transport = full_transport[0] if original_particle_rank == 2 else full_transport
    return AnnealedTransportTFResult(
        particles=result_particles,
        log_weights=result_log_weights,
        transport_matrix=result_transport,
        diagnostics=diagnostics,
    )


def _transport_active(
    particles: tf.Tensor,
    log_weights: tf.Tensor,
    *,
    epsilon: float,
    scaling: float,
    convergence_threshold: float,
    max_iterations: int,
    transport_gradient_mode: str,
    transport_plan_mode: str = "dense",
    transport_ad_mode: str = "stabilized",
    row_chunk_size: int = DEFAULT_STREAMING_CHUNK_SIZE,
    col_chunk_size: int = DEFAULT_STREAMING_CHUNK_SIZE,
    warmstart_state: AnnealedTransportWarmstartStateTF | None = None,
) -> tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor | bool]]:
    _validate_transport_ad_mode(transport_ad_mode)
    x = tf.cast(particles, DTYPE)
    logw = tf.cast(log_weights, DTYPE)
    batch_size = tf.shape(x)[0]
    num_particles = tf.shape(x)[1]
    center = tf.reduce_mean(x, axis=1, keepdims=True)
    if _transport_ad_stop_scale(transport_ad_mode):
        center = tf.stop_gradient(center)
    centered = x - center
    scale = _filterflow_scale(x)
    if _transport_ad_stop_scale(transport_ad_mode):
        scale_for_division = tf.stop_gradient(scale)
    else:
        scale_for_division = scale
    scaled_x = centered / scale_for_division[:, None, None]
    epsilon_tensor = tf.constant(epsilon, DTYPE)
    scaling_tensor = tf.constant(scaling, DTYPE)
    threshold_tensor = tf.constant(convergence_threshold, DTYPE)
    max_iterations_tensor = tf.constant(max_iterations, tf.int32)
    if transport_plan_mode == "streaming":
        if transport_gradient_mode != "raw":
            raise ValueError("streaming transport currently supports raw gradients only")
        transport_matrix = tf.zeros([batch_size, 0, 0], dtype=DTYPE)
        transported, iterations, row_residual, column_residual = _filterflow_streaming_transport(
            scaled_x,
            x,
            logw,
            epsilon_tensor,
            scaling_tensor,
            threshold_tensor,
            max_iterations_tensor,
            num_particles,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            transport_ad_mode=transport_ad_mode,
            warmstart_state=warmstart_state,
        )
    elif transport_gradient_mode == "filterflow_custom_op":
        transport_matrix, iterations = _filterflow_custom_gradient_transport_matrix(
            scaled_x,
            logw,
            epsilon_tensor,
            scaling_tensor,
            threshold_tensor,
            max_iterations_tensor,
            num_particles,
        )
        transported = tf.linalg.matmul(transport_matrix, x)
        row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(transport_matrix, axis=2) - 1.0))
        source_weights = tf.exp(logw)
        column_mass = tf.reduce_sum(transport_matrix, axis=1)
        column_target = source_weights * tf.cast(num_particles, DTYPE)
        column_residual = tf.reduce_max(tf.abs(column_mass - column_target))
    else:
        transport_matrix, iterations = _filterflow_exact_transport_matrix(
            scaled_x,
            logw,
            epsilon_tensor,
            scaling_tensor,
            threshold_tensor,
            max_iterations_tensor,
            num_particles,
            transport_ad_mode=transport_ad_mode,
            warmstart_state=warmstart_state,
        )
        if transport_gradient_mode == "filterflow_clipped":
            transport_matrix = _clip_transport_upstream_gradient(transport_matrix)
        transported = tf.linalg.matmul(transport_matrix, x)
        row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(transport_matrix, axis=2) - 1.0))
        source_weights = tf.exp(logw)
        column_mass = tf.reduce_sum(transport_matrix, axis=1)
        column_target = source_weights * tf.cast(num_particles, DTYPE)
        column_residual = tf.reduce_max(tf.abs(column_mass - column_target))
    return transported, transport_matrix, {
        "max_row_residual": tf.cast(row_residual, DTYPE),
        "max_column_residual": tf.cast(column_residual, DTYPE),
        "max_cost_scale": tf.reduce_max(scale),
        "min_cost_scale": tf.reduce_min(scale),
        "max_iterations_used": tf.reduce_max(tf.cast(iterations, DTYPE)),
        "finite_transport": bool(tf.reduce_all(tf.math.is_finite(transport_matrix)).numpy()),
        "finite_particles": bool(tf.reduce_all(tf.math.is_finite(transported)).numpy()),
    }


def _validate_transport_ad_mode(mode: str) -> str:
    if mode not in TRANSPORT_AD_MODES:
        raise ValueError(
            "transport_ad_mode must be one of "
            f"{sorted(TRANSPORT_AD_MODES)}"
        )
    return mode


def _transport_ad_stop_scale(mode: str) -> bool:
    return mode not in {"diff-scale", "full"}


def _transport_ad_stop_keys(mode: str) -> bool:
    return mode not in {"diff-keys", "full"}


def _transport_ad_stop_potentials(mode: str) -> bool:
    return mode not in {"diff-potentials", "full"}


def _maybe_stop(value: tf.Tensor, *, stop: bool) -> tf.Tensor:
    return tf.stop_gradient(value) if stop else value


def _validate_chunk_size(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _slice_axis1_padded_3d(
    tensor: tf.Tensor,
    start: tf.Tensor,
    chunk_size: int,
) -> tf.Tensor:
    chunk_size = _validate_chunk_size(chunk_size, "chunk_size")
    total = tf.shape(tensor)[1]
    raw_indices = start + tf.range(tf.cast(chunk_size, tf.int32), dtype=tf.int32)
    safe_indices = tf.minimum(raw_indices, total - 1)
    gathered = tf.gather(tensor, safe_indices, axis=1)
    gathered.set_shape([tensor.shape[0], chunk_size, tensor.shape[2]])
    return gathered


def _slice_axis1_padded_4d(
    tensor: tf.Tensor,
    start: tf.Tensor,
    chunk_size: int,
) -> tf.Tensor:
    chunk_size = _validate_chunk_size(chunk_size, "chunk_size")
    total = tf.shape(tensor)[1]
    raw_indices = start + tf.range(tf.cast(chunk_size, tf.int32), dtype=tf.int32)
    safe_indices = tf.minimum(raw_indices, total - 1)
    gathered = tf.gather(tensor, safe_indices, axis=1)
    gathered.set_shape([tensor.shape[0], chunk_size, tensor.shape[2], tensor.shape[3]])
    return gathered


def _slice_axis1_padded_2d(
    tensor: tf.Tensor,
    start: tf.Tensor,
    chunk_size: int,
    *,
    pad_value: float,
) -> tf.Tensor:
    chunk_size = _validate_chunk_size(chunk_size, "chunk_size")
    total = tf.shape(tensor)[1]
    raw_indices = start + tf.range(tf.cast(chunk_size, tf.int32), dtype=tf.int32)
    safe_indices = tf.minimum(raw_indices, total - 1)
    gathered = tf.gather(tensor, safe_indices, axis=1)
    valid = raw_indices < total
    padded = tf.where(
        valid[None, :],
        gathered,
        tf.fill(tf.shape(gathered), tf.cast(pad_value, tensor.dtype)),
    )
    padded.set_shape([tensor.shape[0], chunk_size])
    return padded


def _scatter_axis1_add_2d(
    accum: tf.Tensor,
    start: tf.Tensor,
    updates: tf.Tensor,
) -> tf.Tensor:
    total = tf.shape(accum)[1]
    batch_size = tf.shape(accum)[0]
    raw_indices = start + tf.range(tf.shape(updates)[1], dtype=tf.int32)
    valid = raw_indices < total
    positions = tf.reshape(tf.where(valid), [-1])
    axis1_indices = tf.gather(raw_indices, positions)
    valid_updates = tf.gather(updates, positions, axis=1)
    num_valid = tf.shape(axis1_indices)[0]
    batch_indices = tf.repeat(tf.range(batch_size), num_valid)
    scatter_indices = tf.stack(
        [batch_indices, tf.tile(axis1_indices, [batch_size])],
        axis=1,
    )
    scatter_updates = tf.reshape(valid_updates, [-1])
    return tf.tensor_scatter_nd_add(accum, scatter_indices, scatter_updates)


def _scatter_axis1_add_3d(
    accum: tf.Tensor,
    start: tf.Tensor,
    updates: tf.Tensor,
) -> tf.Tensor:
    total = tf.shape(accum)[1]
    batch_size = tf.shape(accum)[0]
    state_dim = tf.shape(accum)[2]
    raw_indices = start + tf.range(tf.shape(updates)[1], dtype=tf.int32)
    valid = raw_indices < total
    positions = tf.reshape(tf.where(valid), [-1])
    axis1_indices = tf.gather(raw_indices, positions)
    valid_updates = tf.gather(updates, positions, axis=1)
    num_valid = tf.shape(axis1_indices)[0]
    batch_indices = tf.repeat(tf.range(batch_size), num_valid)
    scatter_indices = tf.stack(
        [batch_indices, tf.tile(axis1_indices, [batch_size])],
        axis=1,
    )
    scatter_updates = tf.reshape(valid_updates, [-1, state_dim])
    return tf.tensor_scatter_nd_add(accum, scatter_indices, scatter_updates)


def _epsilon_per_batch(epsilon: tf.Tensor, batch_size: tf.Tensor) -> tf.Tensor:
    epsilon = tf.cast(epsilon, DTYPE)
    static_rank = epsilon.shape.rank
    if static_rank == 0:
        return tf.fill([batch_size], epsilon)
    if static_rank == 1:
        return tf.reshape(epsilon, [-1])
    return tf.reshape(epsilon, [tf.shape(epsilon)[0], -1])[:, 0]


def _streaming_log_zero(dtype: tf.DType) -> tf.Tensor:
    return tf.cast(_STREAMING_LOG_ZERO, dtype)


def _logaddexp(lhs: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
    return tf.reduce_logsumexp(tf.stack([lhs, rhs], axis=0), axis=0)


def _pairwise_squared_cross(query: tf.Tensor, key: tf.Tensor) -> tf.Tensor:
    xx = tf.reduce_sum(query * query, axis=2, keepdims=True)
    xy = tf.matmul(query, key, transpose_b=True)
    yy = tf.expand_dims(tf.reduce_sum(key * key, axis=-1), axis=1)
    return tf.clip_by_value(xx - 2.0 * xy + yy, 0.0, float("inf"))


def _half_pairwise_squared_cross_jvp(
    query: tf.Tensor,
    key: tf.Tensor,
    d_query: tf.Tensor,
    d_key: tf.Tensor,
) -> tf.Tensor:
    """JVP of `0.5 * _pairwise_squared_cross` without a `[B,R,C,D,P]` product."""

    query_dot_d_query = tf.einsum("brd,brdp->brp", query, d_query)
    key_dot_d_key = tf.einsum("bcd,bcdp->bcp", key, d_key)
    key_dot_d_query = tf.einsum("bcd,brdp->brcp", key, d_query)
    query_dot_d_key = tf.einsum("brd,bcdp->brcp", query, d_key)
    return (
        query_dot_d_query[:, :, None, :]
        - key_dot_d_query
        - query_dot_d_key
        + key_dot_d_key[:, None, :, :]
    )


def _filterflow_streaming_softmin(
    epsilon: tf.Tensor,
    query: tf.Tensor,
    key: tf.Tensor,
    values: tf.Tensor,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tf.Tensor:
    """Compute FilterFlow softmin without materializing the cost matrix."""

    row_chunk_size = _validate_chunk_size(row_chunk_size, "row_chunk_size")
    col_chunk_size = _validate_chunk_size(col_chunk_size, "col_chunk_size")
    query = tf.cast(query, DTYPE)
    key = tf.cast(key, DTYPE)
    values = tf.cast(values, DTYPE)
    batch_size = tf.shape(query)[0]
    epsilon = _epsilon_per_batch(epsilon, batch_size)
    num_rows = tf.shape(query)[1]
    num_cols = tf.shape(key)[1]
    row_chunk_tensor = tf.cast(row_chunk_size, tf.int32)
    col_chunk_tensor = tf.cast(col_chunk_size, tf.int32)
    num_row_blocks = (num_rows + row_chunk_tensor - 1) // row_chunk_tensor
    blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_row_blocks,
        element_shape=tf.TensorShape([None, row_chunk_size]),
    )

    def row_cond(row_start: tf.Tensor, _blocks: tf.TensorArray) -> tf.Tensor:
        return row_start < num_rows

    def row_body(row_start: tf.Tensor, blocks_ta: tf.TensorArray):
        query_block = _slice_axis1_padded_3d(query, row_start, row_chunk_size)
        running = tf.fill([batch_size, row_chunk_size], tf.constant(-float("inf"), DTYPE))

        def col_cond(col_start: tf.Tensor, _running: tf.Tensor) -> tf.Tensor:
            return col_start < num_cols

        def col_body(col_start: tf.Tensor, running_logsum: tf.Tensor):
            key_block = _slice_axis1_padded_3d(key, col_start, col_chunk_size)
            values_block = _slice_axis1_padded_2d(
                values,
                col_start,
                col_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            cost = 0.5 * _pairwise_squared_cross(query_block, key_block)
            logits = values_block[:, None, :] - cost / epsilon[:, None, None]
            block_logsum = tf.reduce_logsumexp(logits, axis=2)
            return col_start + col_chunk_tensor, _logaddexp(running_logsum, block_logsum)

        _, row_logsum = tf.while_loop(
            col_cond,
            col_body,
            loop_vars=(tf.constant(0, tf.int32), running),
            maximum_iterations=(num_cols + col_chunk_tensor - 1) // col_chunk_tensor,
        )
        result_block = -epsilon[:, None] * row_logsum
        block_index = row_start // row_chunk_tensor
        blocks_ta = blocks_ta.write(block_index, result_block)
        return row_start + row_chunk_tensor, blocks_ta

    _, blocks = tf.while_loop(
        row_cond,
        row_body,
        loop_vars=(tf.constant(0, tf.int32), blocks),
        maximum_iterations=num_row_blocks,
    )
    stacked = blocks.stack()
    transposed = tf.transpose(stacked, [1, 0, 2])
    flat = tf.reshape(transposed, [batch_size, num_row_blocks * row_chunk_tensor])
    return flat[:, :num_rows]


def _filterflow_streaming_softmin_jvp(
    epsilon: tf.Tensor,
    query: tf.Tensor,
    key: tf.Tensor,
    values: tf.Tensor,
    d_epsilon: tf.Tensor,
    d_query: tf.Tensor,
    d_key: tf.Tensor,
    d_values: tf.Tensor,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Value and forward tangent for `_filterflow_streaming_softmin`.

    Tangents carry a final parameter axis.  This helper is intentionally
    no-tape; it differentiates the same blockwise finite softmin value used by
    the transport route.
    """

    row_chunk_size = _validate_chunk_size(row_chunk_size, "row_chunk_size")
    col_chunk_size = _validate_chunk_size(col_chunk_size, "col_chunk_size")
    query = tf.cast(query, DTYPE)
    key = tf.cast(key, DTYPE)
    values = tf.cast(values, DTYPE)
    d_query = tf.cast(d_query, DTYPE)
    d_key = tf.cast(d_key, DTYPE)
    d_values = tf.cast(d_values, DTYPE)
    batch_size = tf.shape(query)[0]
    epsilon = _epsilon_per_batch(tf.cast(epsilon, DTYPE), batch_size)
    d_epsilon = tf.cast(d_epsilon, DTYPE)
    num_rows = tf.shape(query)[1]
    num_cols = tf.shape(key)[1]
    param_dim = tf.shape(d_values)[2]
    row_chunk_tensor = tf.cast(row_chunk_size, tf.int32)
    col_chunk_tensor = tf.cast(col_chunk_size, tf.int32)
    num_row_blocks = (num_rows + row_chunk_tensor - 1) // row_chunk_tensor
    num_col_blocks = (num_cols + col_chunk_tensor - 1) // col_chunk_tensor
    value_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_row_blocks,
        element_shape=tf.TensorShape([None, row_chunk_size]),
    )
    tangent_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_row_blocks,
        element_shape=tf.TensorShape([None, row_chunk_size, None]),
    )

    def row_cond(
        row_start: tf.Tensor,
        _value_blocks: tf.TensorArray,
        _tangent_blocks: tf.TensorArray,
    ) -> tf.Tensor:
        return row_start < num_rows

    def row_body(
        row_start: tf.Tensor,
        value_ta: tf.TensorArray,
        tangent_ta: tf.TensorArray,
    ):
        query_block = _slice_axis1_padded_3d(query, row_start, row_chunk_size)
        d_query_block = _slice_axis1_padded_4d(d_query, row_start, row_chunk_size)
        running = tf.fill([batch_size, row_chunk_size], tf.constant(-float("inf"), DTYPE))
        d_running = tf.zeros([batch_size, row_chunk_size, param_dim], dtype=DTYPE)

        def col_cond(
            col_start: tf.Tensor,
            _running: tf.Tensor,
            _d_running: tf.Tensor,
        ) -> tf.Tensor:
            return col_start < num_cols

        def col_body(
            col_start: tf.Tensor,
            running_logsum: tf.Tensor,
            running_tangent: tf.Tensor,
        ):
            key_block = _slice_axis1_padded_3d(key, col_start, col_chunk_size)
            d_key_block = _slice_axis1_padded_4d(d_key, col_start, col_chunk_size)
            values_block = _slice_axis1_padded_2d(
                values,
                col_start,
                col_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            d_values_block = _slice_axis1_padded_3d(
                d_values,
                col_start,
                col_chunk_size,
            )
            cost = 0.5 * _pairwise_squared_cross(query_block, key_block)
            logits = values_block[:, None, :] - cost / epsilon[:, None, None]
            d_cost = _half_pairwise_squared_cross_jvp(
                query_block,
                key_block,
                d_query_block,
                d_key_block,
            )
            d_logits = (
                d_values_block[:, None, :, :]
                - d_cost / epsilon[:, None, None, None]
                + cost[:, :, :, None]
                * d_epsilon[:, None, None, :]
                / (epsilon[:, None, None, None] * epsilon[:, None, None, None])
            )
            block_logsum = tf.reduce_logsumexp(logits, axis=2)
            block_probs = tf.exp(logits - block_logsum[:, :, None])
            block_tangent = tf.reduce_sum(block_probs[:, :, :, None] * d_logits, axis=2)
            combined = _logaddexp(running_logsum, block_logsum)
            running_weight = tf.exp(running_logsum - combined)
            block_weight = tf.exp(block_logsum - combined)
            combined_tangent = (
                running_weight[:, :, None] * running_tangent
                + block_weight[:, :, None] * block_tangent
            )
            return col_start + col_chunk_tensor, combined, combined_tangent

        _, row_logsum, row_d_logsum = tf.while_loop(
            col_cond,
            col_body,
            loop_vars=(tf.constant(0, tf.int32), running, d_running),
            maximum_iterations=num_col_blocks,
        )
        result_block = -epsilon[:, None] * row_logsum
        tangent_block = (
            -d_epsilon[:, None, :] * row_logsum[:, :, None]
            -epsilon[:, None, None] * row_d_logsum
        )
        block_index = row_start // row_chunk_tensor
        value_ta = value_ta.write(block_index, result_block)
        tangent_ta = tangent_ta.write(block_index, tangent_block)
        return row_start + row_chunk_tensor, value_ta, tangent_ta

    _, value_blocks, tangent_blocks = tf.while_loop(
        row_cond,
        row_body,
        loop_vars=(tf.constant(0, tf.int32), value_blocks, tangent_blocks),
        maximum_iterations=num_row_blocks,
    )
    stacked_value = value_blocks.stack()
    transposed_value = tf.transpose(stacked_value, [1, 0, 2])
    flat_value = tf.reshape(transposed_value, [batch_size, num_row_blocks * row_chunk_tensor])
    stacked_tangent = tangent_blocks.stack()
    transposed_tangent = tf.transpose(stacked_tangent, [1, 0, 2, 3])
    flat_tangent = tf.reshape(
        transposed_tangent,
        [batch_size, num_row_blocks * row_chunk_tensor, param_dim],
    )
    return flat_value[:, :num_rows], flat_tangent[:, :num_rows, :]


def _filterflow_streaming_softmin_vjp(
    epsilon: tf.Tensor,
    query: tf.Tensor,
    key: tf.Tensor,
    values: tf.Tensor,
    upstream: tf.Tensor,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
    stop_keys: bool,
    return_epsilon: bool = False,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Blockwise VJP for `_filterflow_streaming_softmin`."""

    row_chunk_size = _validate_chunk_size(row_chunk_size, "row_chunk_size")
    col_chunk_size = _validate_chunk_size(col_chunk_size, "col_chunk_size")
    raw_epsilon = tf.cast(epsilon, DTYPE)
    query = tf.cast(query, DTYPE)
    key = tf.cast(key, DTYPE)
    values = tf.cast(values, DTYPE)
    upstream = tf.cast(upstream, DTYPE)
    batch_size = tf.shape(query)[0]
    epsilon = _epsilon_per_batch(raw_epsilon, batch_size)
    num_rows = tf.shape(query)[1]
    num_cols = tf.shape(key)[1]
    state_dim = tf.shape(query)[2]
    row_chunk_tensor = tf.cast(row_chunk_size, tf.int32)
    col_chunk_tensor = tf.cast(col_chunk_size, tf.int32)
    num_row_blocks = (num_rows + row_chunk_tensor - 1) // row_chunk_tensor
    num_col_blocks = (num_cols + col_chunk_tensor - 1) // col_chunk_tensor
    row_logsum_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_row_blocks,
        element_shape=tf.TensorShape([None, row_chunk_size]),
        clear_after_read=False,
    )
    query_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_row_blocks,
        element_shape=tf.TensorShape([None, row_chunk_size, None]),
    )
    key_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_col_blocks,
        element_shape=tf.TensorShape([None, col_chunk_size, None]),
    )
    value_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_col_blocks,
        element_shape=tf.TensorShape([None, col_chunk_size]),
    )
    initial_d_epsilon = tf.zeros([batch_size], dtype=DTYPE)

    def row_cond(
        row_start: tf.Tensor,
        _row_logsum_blocks: tf.TensorArray,
        _query_blocks: tf.TensorArray,
        _d_epsilon: tf.Tensor,
    ) -> tf.Tensor:
        return row_start < num_rows

    def row_body(
        row_start: tf.Tensor,
        row_logsum_ta: tf.TensorArray,
        query_ta: tf.TensorArray,
        d_epsilon_accum: tf.Tensor,
    ):
        query_block = _slice_axis1_padded_3d(query, row_start, row_chunk_size)
        upstream_block = _slice_axis1_padded_2d(
            upstream,
            row_start,
            row_chunk_size,
            pad_value=0.0,
        )
        d_query_block = tf.zeros([batch_size, row_chunk_size, state_dim], dtype=DTYPE)
        running = tf.fill([batch_size, row_chunk_size], tf.constant(-float("inf"), DTYPE))

        def logsum_col_cond(col_start: tf.Tensor, _running: tf.Tensor) -> tf.Tensor:
            return col_start < num_cols

        def logsum_col_body(col_start: tf.Tensor, running_logsum: tf.Tensor):
            key_block = _slice_axis1_padded_3d(key, col_start, col_chunk_size)
            values_block = _slice_axis1_padded_2d(
                values,
                col_start,
                col_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            cost = 0.5 * _pairwise_squared_cross(query_block, key_block)
            logits = values_block[:, None, :] - cost / epsilon[:, None, None]
            block_logsum = tf.reduce_logsumexp(logits, axis=2)
            return col_start + col_chunk_tensor, _logaddexp(running_logsum, block_logsum)

        _, row_logsum = tf.while_loop(
            logsum_col_cond,
            logsum_col_body,
            loop_vars=(tf.constant(0, tf.int32), running),
            maximum_iterations=num_col_blocks,
        )
        row_d_epsilon = tf.reduce_sum(upstream_block * (-row_logsum), axis=1)

        def col_cond(
            col_start: tf.Tensor,
            _d_query_block: tf.Tensor,
            _row_d_epsilon: tf.Tensor,
        ) -> tf.Tensor:
            return col_start < num_cols

        def col_body(
            col_start: tf.Tensor,
            d_query_block_accum: tf.Tensor,
            inner_row_d_epsilon: tf.Tensor,
        ):
            key_block = _slice_axis1_padded_3d(key, col_start, col_chunk_size)
            values_block = _slice_axis1_padded_2d(
                values,
                col_start,
                col_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            cost = 0.5 * _pairwise_squared_cross(query_block, key_block)
            logits = values_block[:, None, :] - cost / epsilon[:, None, None]
            probs = tf.exp(logits - row_logsum[:, :, None])
            d_cost = upstream_block[:, :, None] * probs
            diff = query_block[:, :, None, :] - key_block[:, None, :, :]
            d_query_next = d_query_block_accum + tf.reduce_sum(
                d_cost[:, :, :, None] * diff,
                axis=2,
            )
            expected_cost_block = tf.reduce_sum(probs * cost, axis=2)
            d_epsilon_from_cost = tf.reduce_sum(
                upstream_block * (-expected_cost_block / epsilon[:, None]),
                axis=1,
            )
            return (
                col_start + col_chunk_tensor,
                d_query_next,
                inner_row_d_epsilon + d_epsilon_from_cost,
            )

        _, d_query_block, row_d_epsilon = tf.while_loop(
            col_cond,
            col_body,
            loop_vars=(
                tf.constant(0, tf.int32),
                d_query_block,
                row_d_epsilon,
            ),
            maximum_iterations=num_col_blocks,
        )
        block_index = row_start // row_chunk_tensor
        row_logsum_ta = row_logsum_ta.write(block_index, row_logsum)
        query_ta = query_ta.write(block_index, d_query_block)
        return (
            row_start + row_chunk_tensor,
            row_logsum_ta,
            query_ta,
            d_epsilon_accum + row_d_epsilon,
        )

    _, row_logsum_blocks, query_blocks, d_epsilon = tf.while_loop(
        row_cond,
        row_body,
        loop_vars=(
            tf.constant(0, tf.int32),
            row_logsum_blocks,
            query_blocks,
            initial_d_epsilon,
        ),
        shape_invariants=(
            tf.TensorShape([]),
            tf.TensorShape(None),
            tf.TensorShape(None),
            tf.TensorShape([None]),
        ),
        maximum_iterations=num_row_blocks,
    )

    def key_cond(
        col_start: tf.Tensor,
        _key_blocks: tf.TensorArray,
        _value_blocks: tf.TensorArray,
    ) -> tf.Tensor:
        return col_start < num_cols

    def key_body(
        col_start: tf.Tensor,
        key_ta: tf.TensorArray,
        value_ta: tf.TensorArray,
    ):
        key_block = _slice_axis1_padded_3d(key, col_start, col_chunk_size)
        values_block = _slice_axis1_padded_2d(
            values,
            col_start,
            col_chunk_size,
            pad_value=float(_STREAMING_LOG_ZERO),
        )
        d_key_block = tf.zeros([batch_size, col_chunk_size, state_dim], dtype=DTYPE)
        d_values_block = tf.zeros([batch_size, col_chunk_size], dtype=DTYPE)

        def row_accum_cond(
            row_start: tf.Tensor,
            _d_key_block: tf.Tensor,
            _d_values_block: tf.Tensor,
        ) -> tf.Tensor:
            return row_start < num_rows

        def row_accum_body(
            row_start: tf.Tensor,
            d_key_block_accum: tf.Tensor,
            d_values_block_accum: tf.Tensor,
        ):
            block_index = row_start // row_chunk_tensor
            query_block = _slice_axis1_padded_3d(query, row_start, row_chunk_size)
            upstream_block = _slice_axis1_padded_2d(
                upstream,
                row_start,
                row_chunk_size,
                pad_value=0.0,
            )
            row_logsum = row_logsum_blocks.read(block_index)
            cost = 0.5 * _pairwise_squared_cross(query_block, key_block)
            logits = values_block[:, None, :] - cost / epsilon[:, None, None]
            probs = tf.exp(logits - row_logsum[:, :, None])
            d_values_next = d_values_block_accum - epsilon[:, None] * tf.reduce_sum(
                upstream_block[:, :, None] * probs,
                axis=1,
            )
            if stop_keys:
                d_key_next = d_key_block_accum
            else:
                d_cost = upstream_block[:, :, None] * probs
                diff = query_block[:, :, None, :] - key_block[:, None, :, :]
                d_key_next = d_key_block_accum + tf.reduce_sum(
                    d_cost[:, :, :, None] * (-diff),
                    axis=1,
                )
            return row_start + row_chunk_tensor, d_key_next, d_values_next

        _, d_key_block, d_values_block = tf.while_loop(
            row_accum_cond,
            row_accum_body,
            loop_vars=(tf.constant(0, tf.int32), d_key_block, d_values_block),
            maximum_iterations=num_row_blocks,
        )
        block_index = col_start // col_chunk_tensor
        key_ta = key_ta.write(block_index, d_key_block)
        value_ta = value_ta.write(block_index, d_values_block)
        return col_start + col_chunk_tensor, key_ta, value_ta

    _, key_blocks, value_blocks = tf.while_loop(
        key_cond,
        key_body,
        loop_vars=(tf.constant(0, tf.int32), key_blocks, value_blocks),
        maximum_iterations=num_col_blocks,
    )

    stacked = query_blocks.stack()
    transposed = tf.transpose(stacked, [1, 0, 2, 3])
    d_query = tf.reshape(
        transposed,
        [batch_size, num_row_blocks * row_chunk_tensor, state_dim],
    )[:, :num_rows, :]
    stacked_key = key_blocks.stack()
    transposed_key = tf.transpose(stacked_key, [1, 0, 2, 3])
    d_key = tf.reshape(
        transposed_key,
        [batch_size, num_col_blocks * col_chunk_tensor, state_dim],
    )[:, :num_cols, :]
    stacked_values = value_blocks.stack()
    transposed_values = tf.transpose(stacked_values, [1, 0, 2])
    d_values = tf.reshape(
        transposed_values,
        [batch_size, num_col_blocks * col_chunk_tensor],
    )[:, :num_cols]
    if return_epsilon:
        d_epsilon = _match_epsilon_cotangent_shape(raw_epsilon, d_epsilon)
        return d_query, d_key, d_values, d_epsilon
    return d_query, d_key, d_values


def _filterflow_streaming_same_points_softmin_vjp(
    epsilon: tf.Tensor,
    points: tf.Tensor,
    values: tf.Tensor,
    upstream: tf.Tensor,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
    return_epsilon: bool = False,
) -> tuple[tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Blockwise VJP for softmin when query/key are the same particle tensor.

    This is algebraically the same as calling `_filterflow_streaming_softmin_vjp`
    with `query=key=points` and summing the query/key cotangents, but it avoids
    stacking both full cotangent outputs on the equal-chunk path used by the
    memory-style total Sinkhorn reverse route.
    """

    row_chunk_size = _validate_chunk_size(row_chunk_size, "row_chunk_size")
    col_chunk_size = _validate_chunk_size(col_chunk_size, "col_chunk_size")
    if row_chunk_size != col_chunk_size:
        if return_epsilon:
            d_query, d_key, d_values, d_epsilon = _filterflow_streaming_softmin_vjp(
                epsilon,
                points,
                points,
                values,
                upstream,
                row_chunk_size=row_chunk_size,
                col_chunk_size=col_chunk_size,
                stop_keys=False,
                return_epsilon=True,
            )
            return d_query + d_key, d_values, d_epsilon
        d_query, d_key, d_values = _filterflow_streaming_softmin_vjp(
            epsilon,
            points,
            points,
            values,
            upstream,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            stop_keys=False,
        )
        return d_query + d_key, d_values

    raw_epsilon = tf.cast(epsilon, DTYPE)
    points = tf.cast(points, DTYPE)
    values = tf.cast(values, DTYPE)
    upstream = tf.cast(upstream, DTYPE)
    batch_size = tf.shape(points)[0]
    epsilon = _epsilon_per_batch(raw_epsilon, batch_size)
    num_points = tf.shape(points)[1]
    state_dim = tf.shape(points)[2]
    chunk_tensor = tf.cast(row_chunk_size, tf.int32)
    num_blocks = (num_points + chunk_tensor - 1) // chunk_tensor
    row_logsum_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_blocks,
        element_shape=tf.TensorShape([None, row_chunk_size]),
        clear_after_read=False,
    )
    combined_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_blocks,
        element_shape=tf.TensorShape([None, row_chunk_size, None]),
    )
    value_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_blocks,
        element_shape=tf.TensorShape([None, row_chunk_size]),
    )
    initial_d_epsilon = tf.zeros([batch_size], dtype=DTYPE)

    def logsum_cond(
        row_start: tf.Tensor,
        _row_logsum_blocks: tf.TensorArray,
    ) -> tf.Tensor:
        return row_start < num_points

    def logsum_body(
        row_start: tf.Tensor,
        row_logsum_ta: tf.TensorArray,
    ):
        query_block = _slice_axis1_padded_3d(points, row_start, row_chunk_size)
        running = tf.fill([batch_size, row_chunk_size], tf.constant(-float("inf"), DTYPE))

        def col_cond(col_start: tf.Tensor, _running: tf.Tensor) -> tf.Tensor:
            return col_start < num_points

        def col_body(col_start: tf.Tensor, running_logsum: tf.Tensor):
            key_block = _slice_axis1_padded_3d(points, col_start, row_chunk_size)
            values_block = _slice_axis1_padded_2d(
                values,
                col_start,
                row_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            cost = 0.5 * _pairwise_squared_cross(query_block, key_block)
            logits = values_block[:, None, :] - cost / epsilon[:, None, None]
            block_logsum = tf.reduce_logsumexp(logits, axis=2)
            return col_start + chunk_tensor, _logaddexp(running_logsum, block_logsum)

        _, row_logsum = tf.while_loop(
            col_cond,
            col_body,
            loop_vars=(tf.constant(0, tf.int32), running),
            maximum_iterations=num_blocks,
        )
        block_index = row_start // chunk_tensor
        return row_start + chunk_tensor, row_logsum_ta.write(block_index, row_logsum)

    _, row_logsum_blocks = tf.while_loop(
        logsum_cond,
        logsum_body,
        loop_vars=(tf.constant(0, tf.int32), row_logsum_blocks),
        maximum_iterations=num_blocks,
    )

    def combined_cond(
        block_start: tf.Tensor,
        _combined_blocks: tf.TensorArray,
        _value_blocks: tf.TensorArray,
        _d_epsilon: tf.Tensor,
    ) -> tf.Tensor:
        return block_start < num_points

    def combined_body(
        block_start: tf.Tensor,
        combined_ta: tf.TensorArray,
        value_ta: tf.TensorArray,
        d_epsilon_accum: tf.Tensor,
    ):
        block_index = block_start // chunk_tensor
        point_block = _slice_axis1_padded_3d(points, block_start, row_chunk_size)
        value_block = _slice_axis1_padded_2d(
            values,
            block_start,
            row_chunk_size,
            pad_value=float(_STREAMING_LOG_ZERO),
        )
        upstream_block = _slice_axis1_padded_2d(
            upstream,
            block_start,
            row_chunk_size,
            pad_value=0.0,
        )
        row_logsum = row_logsum_blocks.read(block_index)
        d_query_block = tf.zeros([batch_size, row_chunk_size, state_dim], dtype=DTYPE)
        row_d_epsilon = tf.reduce_sum(upstream_block * (-row_logsum), axis=1)

        def query_col_cond(
            col_start: tf.Tensor,
            _d_query_block: tf.Tensor,
            _row_d_epsilon: tf.Tensor,
        ) -> tf.Tensor:
            return col_start < num_points

        def query_col_body(
            col_start: tf.Tensor,
            d_query_accum: tf.Tensor,
            inner_row_d_epsilon: tf.Tensor,
        ):
            key_block = _slice_axis1_padded_3d(points, col_start, row_chunk_size)
            values_block = _slice_axis1_padded_2d(
                values,
                col_start,
                row_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            cost = 0.5 * _pairwise_squared_cross(point_block, key_block)
            logits = values_block[:, None, :] - cost / epsilon[:, None, None]
            probs = tf.exp(logits - row_logsum[:, :, None])
            d_cost = upstream_block[:, :, None] * probs
            diff = point_block[:, :, None, :] - key_block[:, None, :, :]
            d_query_next = d_query_accum + tf.reduce_sum(
                d_cost[:, :, :, None] * diff,
                axis=2,
            )
            expected_cost_block = tf.reduce_sum(probs * cost, axis=2)
            d_epsilon_from_cost = tf.reduce_sum(
                upstream_block * (-expected_cost_block / epsilon[:, None]),
                axis=1,
            )
            return (
                col_start + chunk_tensor,
                d_query_next,
                inner_row_d_epsilon + d_epsilon_from_cost,
            )

        _, d_query_block, row_d_epsilon = tf.while_loop(
            query_col_cond,
            query_col_body,
            loop_vars=(
                tf.constant(0, tf.int32),
                d_query_block,
                row_d_epsilon,
            ),
            maximum_iterations=num_blocks,
        )

        d_key_block = tf.zeros([batch_size, row_chunk_size, state_dim], dtype=DTYPE)
        d_values_block = tf.zeros([batch_size, row_chunk_size], dtype=DTYPE)

        def key_row_cond(
            row_start: tf.Tensor,
            _d_key_block: tf.Tensor,
            _d_values_block: tf.Tensor,
        ) -> tf.Tensor:
            return row_start < num_points

        def key_row_body(
            row_start: tf.Tensor,
            d_key_accum: tf.Tensor,
            d_values_accum: tf.Tensor,
        ):
            row_index = row_start // chunk_tensor
            query_block = _slice_axis1_padded_3d(points, row_start, row_chunk_size)
            upstream_row = _slice_axis1_padded_2d(
                upstream,
                row_start,
                row_chunk_size,
                pad_value=0.0,
            )
            other_row_logsum = row_logsum_blocks.read(row_index)
            cost = 0.5 * _pairwise_squared_cross(query_block, point_block)
            logits = value_block[:, None, :] - cost / epsilon[:, None, None]
            probs = tf.exp(logits - other_row_logsum[:, :, None])
            d_values_next = d_values_accum - epsilon[:, None] * tf.reduce_sum(
                upstream_row[:, :, None] * probs,
                axis=1,
            )
            d_cost = upstream_row[:, :, None] * probs
            diff = query_block[:, :, None, :] - point_block[:, None, :, :]
            d_key_next = d_key_accum + tf.reduce_sum(
                d_cost[:, :, :, None] * (-diff),
                axis=1,
            )
            return row_start + chunk_tensor, d_key_next, d_values_next

        _, d_key_block, d_values_block = tf.while_loop(
            key_row_cond,
            key_row_body,
            loop_vars=(tf.constant(0, tf.int32), d_key_block, d_values_block),
            maximum_iterations=num_blocks,
        )

        combined_ta = combined_ta.write(block_index, d_query_block + d_key_block)
        value_ta = value_ta.write(block_index, d_values_block)
        return (
            block_start + chunk_tensor,
            combined_ta,
            value_ta,
            d_epsilon_accum + row_d_epsilon,
        )

    _, combined_blocks, value_blocks, d_epsilon = tf.while_loop(
        combined_cond,
        combined_body,
        loop_vars=(
            tf.constant(0, tf.int32),
            combined_blocks,
            value_blocks,
            initial_d_epsilon,
        ),
        shape_invariants=(
            tf.TensorShape([]),
            tf.TensorShape(None),
            tf.TensorShape(None),
            tf.TensorShape([None]),
        ),
        maximum_iterations=num_blocks,
    )

    stacked_combined = combined_blocks.stack()
    transposed_combined = tf.transpose(stacked_combined, [1, 0, 2, 3])
    d_points = tf.reshape(
        transposed_combined,
        [batch_size, num_blocks * chunk_tensor, state_dim],
    )[:, :num_points, :]
    stacked_values = value_blocks.stack()
    transposed_values = tf.transpose(stacked_values, [1, 0, 2])
    d_values = tf.reshape(
        transposed_values,
        [batch_size, num_blocks * chunk_tensor],
    )[:, :num_points]
    if return_epsilon:
        d_epsilon = _match_epsilon_cotangent_shape(raw_epsilon, d_epsilon)
        return d_points, d_values, d_epsilon
    return d_points, d_values


def _match_epsilon_cotangent_shape(
    epsilon: tf.Tensor,
    d_epsilon_per_batch: tf.Tensor,
) -> tf.Tensor:
    """Map per-batch epsilon cotangents to the shape used by `_epsilon_per_batch`."""

    epsilon = tf.cast(epsilon, DTYPE)
    d_epsilon_per_batch = tf.cast(d_epsilon_per_batch, DTYPE)
    static_rank = epsilon.shape.rank
    if static_rank == 0:
        return tf.reduce_sum(d_epsilon_per_batch)
    if static_rank == 1:
        return tf.reshape(d_epsilon_per_batch, tf.shape(epsilon))
    flat = tf.reshape(tf.zeros_like(epsilon), [tf.shape(epsilon)[0], -1])
    batch_indices = tf.range(tf.shape(flat)[0], dtype=tf.int32)
    scatter_indices = tf.stack(
        [batch_indices, tf.zeros_like(batch_indices)],
        axis=1,
    )
    flat = tf.tensor_scatter_nd_add(flat, scatter_indices, d_epsilon_per_batch)
    return tf.reshape(flat, tf.shape(epsilon))


def _filterflow_streaming_sinkhorn_potentials(
    log_alpha: tf.Tensor,
    x: tf.Tensor,
    log_beta: tf.Tensor,
    y: tf.Tensor,
    epsilon: tf.Tensor,
    scaling: tf.Tensor,
    threshold: tf.Tensor,
    max_iter: tf.Tensor,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
    transport_ad_mode: str = "stabilized",
    warmstart_state: AnnealedTransportWarmstartStateTF | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    _validate_transport_ad_mode(transport_ad_mode)
    batch_size = tf.shape(log_alpha)[0]
    continue_flag = tf.ones([batch_size], dtype=tf.bool)
    epsilon_0 = _maybe_stop(
        _filterflow_exact_max_min(x, y),
        stop=_transport_ad_stop_scale(transport_ad_mode),
    ) ** 2
    scaling_factor = scaling ** 2
    x_key = _maybe_stop(x, stop=_transport_ad_stop_keys(transport_ad_mode))
    y_key = _maybe_stop(y, stop=_transport_ad_stop_keys(transport_ad_mode))

    if warmstart_state is None:
        a_y_init = _filterflow_streaming_softmin(
            epsilon_0,
            y,
            x_key,
            log_alpha,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        b_x_init = _filterflow_streaming_softmin(
            epsilon_0,
            x,
            y_key,
            log_beta,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        a_x_init = _filterflow_streaming_softmin(
            epsilon_0,
            x,
            x_key,
            log_alpha,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        b_y_init = _filterflow_streaming_softmin(
            epsilon_0,
            y,
            y_key,
            log_beta,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
    else:
        warm_mask = tf.reshape(tf.cast(warmstart_state.valid_mask, tf.bool), [-1])
        a_y_cold = _filterflow_streaming_softmin(
            epsilon_0,
            y,
            x_key,
            log_alpha,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        b_x_cold = _filterflow_streaming_softmin(
            epsilon_0,
            x,
            y_key,
            log_beta,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        a_x_cold = _filterflow_streaming_softmin(
            epsilon_0,
            x,
            x_key,
            log_alpha,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        b_y_cold = _filterflow_streaming_softmin(
            epsilon_0,
            y,
            y_key,
            log_beta,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        a_y_init = tf.where(warm_mask[:, None], tf.cast(warmstart_state.a_y, DTYPE), a_y_cold)
        b_x_init = tf.where(warm_mask[:, None], tf.cast(warmstart_state.b_x, DTYPE), b_x_cold)
        a_x_init = tf.where(warm_mask[:, None], tf.cast(warmstart_state.a_x, DTYPE), a_x_cold)
        b_y_init = tf.where(warm_mask[:, None], tf.cast(warmstart_state.b_y, DTYPE), b_y_cold)

    def stop_condition(i, _a_y, _b_x, _a_x, _b_y, continue_, _running_epsilon):
        n_iter_cond = i < max_iter - 1
        return tf.logical_and(n_iter_cond, tf.reduce_all(continue_))

    def apply_one(a_y, b_x, a_x, b_y, continue_, running_epsilon):
        running_epsilon_ = tf.reshape(running_epsilon, [-1, 1])
        continue_reshaped = tf.reshape(continue_, [-1, 1])
        at_y_raw = _filterflow_streaming_softmin(
            running_epsilon,
            y,
            x_key,
            log_alpha + b_x / running_epsilon_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        bt_x_raw = _filterflow_streaming_softmin(
            running_epsilon,
            x,
            y_key,
            log_beta + a_y / running_epsilon_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        at_x_raw = _filterflow_streaming_softmin(
            running_epsilon,
            x,
            x_key,
            log_alpha + a_x / running_epsilon_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        bt_y_raw = _filterflow_streaming_softmin(
            running_epsilon,
            y,
            y_key,
            log_beta + b_y / running_epsilon_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        at_y = tf.where(continue_reshaped, at_y_raw, a_y)
        bt_x = tf.where(continue_reshaped, bt_x_raw, b_x)
        at_x = tf.where(continue_reshaped, at_x_raw, a_x)
        bt_y = tf.where(continue_reshaped, bt_y_raw, b_y)
        a_y_new = (a_y + at_y) / 2
        b_x_new = (b_x + bt_x) / 2
        a_x_new = (a_x + at_x) / 2
        b_y_new = (b_y + bt_y) / 2
        a_y_diff = tf.reduce_max(tf.abs(a_y_new - a_y), 1)
        b_x_diff = tf.reduce_max(tf.abs(b_x_new - b_x), 1)
        local_continue = tf.logical_or(a_y_diff > threshold, b_x_diff > threshold)
        return a_y_new, b_x_new, a_x_new, b_y_new, local_continue

    def body(i, a_y, b_x, a_x, b_y, continue_, running_epsilon):
        new_a_y, new_b_x, new_a_x, new_b_y, local_continue = apply_one(
            a_y,
            b_x,
            a_x,
            b_y,
            continue_,
            running_epsilon,
        )
        new_epsilon = tf.maximum(running_epsilon * scaling_factor, epsilon)
        global_continue = tf.logical_or(new_epsilon < running_epsilon, local_continue)
        return i + 1, new_a_y, new_b_x, new_a_x, new_b_y, global_continue, new_epsilon

    (
        total_iter,
        converged_a_y,
        converged_b_x,
        converged_a_x,
        converged_b_y,
        _,
        _final_epsilon,
    ) = tf.while_loop(
        stop_condition,
        body,
        loop_vars=(
            tf.constant(0, tf.int32),
            a_y_init,
            b_x_init,
            a_x_init,
            b_y_init,
            continue_flag,
            epsilon_0,
        ),
        maximum_iterations=max_iter,
    )
    if _transport_ad_stop_potentials(transport_ad_mode):
        (
            converged_a_y,
            converged_b_x,
            converged_a_x,
            converged_b_y,
        ) = tf.nest.map_structure(
            tf.stop_gradient,
            (converged_a_y, converged_b_x, converged_a_x, converged_b_y),
        )
    epsilon_ = tf.reshape(epsilon, [-1, 1])
    final_a_y = _filterflow_streaming_softmin(
        epsilon,
        y,
        x_key,
        log_alpha + converged_b_x / epsilon_,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    final_b_x = _filterflow_streaming_softmin(
        epsilon,
        x,
        y_key,
        log_beta + converged_a_y / epsilon_,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    final_a_x = _filterflow_streaming_softmin(
        epsilon,
        x,
        x_key,
        log_alpha + converged_a_x / epsilon_,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    final_b_y = _filterflow_streaming_softmin(
        epsilon,
        y,
        y_key,
        log_beta + converged_b_y / epsilon_,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    return final_a_y, final_b_x, final_a_x, final_b_y, total_iter + 2


def _filterflow_streaming_column_log_normalizer(
    x: tf.Tensor,
    f: tf.Tensor,
    g: tf.Tensor,
    eps: tf.Tensor,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tf.Tensor:
    row_chunk_size = _validate_chunk_size(row_chunk_size, "row_chunk_size")
    col_chunk_size = _validate_chunk_size(col_chunk_size, "col_chunk_size")
    batch_size = tf.shape(x)[0]
    num_particles = tf.shape(x)[1]
    row_chunk_tensor = tf.cast(row_chunk_size, tf.int32)
    eps = _epsilon_per_batch(eps, batch_size)
    col_chunk_tensor = tf.cast(col_chunk_size, tf.int32)
    num_col_blocks = (num_particles + col_chunk_tensor - 1) // col_chunk_tensor
    blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_col_blocks,
        element_shape=tf.TensorShape([None, col_chunk_size]),
    )

    def col_cond(col_start: tf.Tensor, _blocks: tf.TensorArray) -> tf.Tensor:
        return col_start < num_particles

    def col_body(col_start: tf.Tensor, blocks_ta: tf.TensorArray):
        key_block = _slice_axis1_padded_3d(x, col_start, col_chunk_size)
        g_block = _slice_axis1_padded_2d(
            g,
            col_start,
            col_chunk_size,
            pad_value=float(_STREAMING_LOG_ZERO),
        )
        running = tf.fill(
            [batch_size, col_chunk_size],
            _streaming_log_zero(DTYPE),
        )

        def row_cond(row_start: tf.Tensor, _running: tf.Tensor) -> tf.Tensor:
            return row_start < num_particles

        def row_body(row_start: tf.Tensor, running_logsum: tf.Tensor):
            query_block = _slice_axis1_padded_3d(x, row_start, row_chunk_size)
            f_block = _slice_axis1_padded_2d(
                f,
                row_start,
                row_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            cost = 0.5 * _pairwise_squared_cross(query_block, key_block)
            logits = (
                f_block[:, :, None]
                + g_block[:, None, :]
                - cost
            ) / eps[:, None, None]
            block_logsum = tf.reduce_logsumexp(logits, axis=1)
            return row_start + row_chunk_tensor, _logaddexp(running_logsum, block_logsum)

        _, col_logsum = tf.while_loop(
            row_cond,
            row_body,
            loop_vars=(tf.constant(0, tf.int32), running),
            maximum_iterations=(num_particles + row_chunk_tensor - 1) // row_chunk_tensor,
        )
        block_index = col_start // col_chunk_tensor
        blocks_ta = blocks_ta.write(block_index, col_logsum)
        return col_start + col_chunk_tensor, blocks_ta

    _, blocks = tf.while_loop(
        col_cond,
        col_body,
        loop_vars=(tf.constant(0, tf.int32), blocks),
        maximum_iterations=num_col_blocks,
    )
    stacked = blocks.stack()
    transposed = tf.transpose(stacked, [1, 0, 2])
    flat = tf.reshape(transposed, [batch_size, num_col_blocks * col_chunk_tensor])
    return flat[:, :num_particles]


def _filterflow_streaming_column_log_normalizer_jvp(
    x: tf.Tensor,
    f: tf.Tensor,
    g: tf.Tensor,
    eps: tf.Tensor,
    d_x: tf.Tensor,
    d_f: tf.Tensor,
    d_g: tf.Tensor,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    row_chunk_size = _validate_chunk_size(row_chunk_size, "row_chunk_size")
    col_chunk_size = _validate_chunk_size(col_chunk_size, "col_chunk_size")
    x = tf.cast(x, DTYPE)
    f = tf.cast(f, DTYPE)
    g = tf.cast(g, DTYPE)
    d_x = tf.cast(d_x, DTYPE)
    d_f = tf.cast(d_f, DTYPE)
    d_g = tf.cast(d_g, DTYPE)
    batch_size = tf.shape(x)[0]
    num_particles = tf.shape(x)[1]
    param_dim = tf.shape(d_f)[2]
    row_chunk_tensor = tf.cast(row_chunk_size, tf.int32)
    col_chunk_tensor = tf.cast(col_chunk_size, tf.int32)
    eps = _epsilon_per_batch(eps, batch_size)
    num_col_blocks = (num_particles + col_chunk_tensor - 1) // col_chunk_tensor
    value_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_col_blocks,
        element_shape=tf.TensorShape([None, col_chunk_size]),
    )
    tangent_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_col_blocks,
        element_shape=tf.TensorShape([None, col_chunk_size, None]),
    )

    def col_cond(
        col_start: tf.Tensor,
        _value_blocks: tf.TensorArray,
        _tangent_blocks: tf.TensorArray,
    ) -> tf.Tensor:
        return col_start < num_particles

    def col_body(
        col_start: tf.Tensor,
        value_ta: tf.TensorArray,
        tangent_ta: tf.TensorArray,
    ):
        key_block = _slice_axis1_padded_3d(x, col_start, col_chunk_size)
        d_key_block = _slice_axis1_padded_4d(d_x, col_start, col_chunk_size)
        g_block = _slice_axis1_padded_2d(
            g,
            col_start,
            col_chunk_size,
            pad_value=float(_STREAMING_LOG_ZERO),
        )
        d_g_block = _slice_axis1_padded_3d(d_g, col_start, col_chunk_size)
        running = tf.fill(
            [batch_size, col_chunk_size],
            _streaming_log_zero(DTYPE),
        )
        d_running = tf.zeros([batch_size, col_chunk_size, param_dim], dtype=DTYPE)

        def row_cond(
            row_start: tf.Tensor,
            _running: tf.Tensor,
            _d_running: tf.Tensor,
        ) -> tf.Tensor:
            return row_start < num_particles

        def row_body(
            row_start: tf.Tensor,
            running_logsum: tf.Tensor,
            running_tangent: tf.Tensor,
        ):
            query_block = _slice_axis1_padded_3d(x, row_start, row_chunk_size)
            d_query_block = _slice_axis1_padded_4d(d_x, row_start, row_chunk_size)
            f_block = _slice_axis1_padded_2d(
                f,
                row_start,
                row_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            d_f_block = _slice_axis1_padded_3d(d_f, row_start, row_chunk_size)
            cost = 0.5 * _pairwise_squared_cross(query_block, key_block)
            logits = (
                f_block[:, :, None]
                + g_block[:, None, :]
                - cost
            ) / eps[:, None, None]
            d_cost = _half_pairwise_squared_cross_jvp(
                query_block,
                key_block,
                d_query_block,
                d_key_block,
            )
            d_logits = (
                d_f_block[:, :, None, :]
                + d_g_block[:, None, :, :]
                - d_cost
            ) / eps[:, None, None, None]
            block_logsum = tf.reduce_logsumexp(logits, axis=1)
            block_probs = tf.exp(logits - block_logsum[:, None, :])
            block_tangent = tf.reduce_sum(block_probs[:, :, :, None] * d_logits, axis=1)
            combined = _logaddexp(running_logsum, block_logsum)
            running_weight = tf.exp(running_logsum - combined)
            block_weight = tf.exp(block_logsum - combined)
            combined_tangent = (
                running_weight[:, :, None] * running_tangent
                + block_weight[:, :, None] * block_tangent
            )
            return row_start + row_chunk_tensor, combined, combined_tangent

        _, col_logsum, col_d_logsum = tf.while_loop(
            row_cond,
            row_body,
            loop_vars=(tf.constant(0, tf.int32), running, d_running),
            maximum_iterations=(num_particles + row_chunk_tensor - 1) // row_chunk_tensor,
        )
        block_index = col_start // col_chunk_tensor
        value_ta = value_ta.write(block_index, col_logsum)
        tangent_ta = tangent_ta.write(block_index, col_d_logsum)
        return col_start + col_chunk_tensor, value_ta, tangent_ta

    _, value_blocks, tangent_blocks = tf.while_loop(
        col_cond,
        col_body,
        loop_vars=(tf.constant(0, tf.int32), value_blocks, tangent_blocks),
        maximum_iterations=num_col_blocks,
    )
    stacked_value = value_blocks.stack()
    transposed_value = tf.transpose(stacked_value, [1, 0, 2])
    flat_value = tf.reshape(transposed_value, [batch_size, num_col_blocks * col_chunk_tensor])
    stacked_tangent = tangent_blocks.stack()
    transposed_tangent = tf.transpose(stacked_tangent, [1, 0, 2, 3])
    flat_tangent = tf.reshape(
        transposed_tangent,
        [batch_size, num_col_blocks * col_chunk_tensor, param_dim],
    )
    return flat_value[:, :num_particles], flat_tangent[:, :num_particles, :]


def _filterflow_streaming_transport_from_potentials(
    scaled_x: tf.Tensor,
    particles: tf.Tensor,
    f: tf.Tensor,
    g: tf.Tensor,
    eps: tf.Tensor,
    logw: tf.Tensor,
    float_n: tf.Tensor,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    row_chunk_size = _validate_chunk_size(row_chunk_size, "row_chunk_size")
    col_chunk_size = _validate_chunk_size(col_chunk_size, "col_chunk_size")
    batch_size = tf.shape(scaled_x)[0]
    num_particles = tf.shape(scaled_x)[1]
    state_dim = tf.shape(particles)[2]
    eps = tf.reshape(tf.cast(eps, DTYPE), [-1])
    log_n = tf.math.log(float_n)
    column_log_norm = _filterflow_streaming_column_log_normalizer(
        scaled_x,
        f,
        g,
        eps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    row_chunk_tensor = tf.cast(row_chunk_size, tf.int32)
    col_chunk_tensor = tf.cast(col_chunk_size, tf.int32)
    num_row_blocks = (num_particles + row_chunk_tensor - 1) // row_chunk_tensor
    num_col_blocks = (num_particles + col_chunk_tensor - 1) // col_chunk_tensor
    particle_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_row_blocks,
        element_shape=tf.TensorShape([None, row_chunk_size, None]),
    )
    mass_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_row_blocks,
        element_shape=tf.TensorShape([None, row_chunk_size]),
    )

    def row_cond(row_start: tf.Tensor, _particle_blocks, _mass_blocks) -> tf.Tensor:
        return row_start < num_particles

    def row_body(row_start: tf.Tensor, particle_ta: tf.TensorArray, mass_ta: tf.TensorArray):
        query_block = _slice_axis1_padded_3d(scaled_x, row_start, row_chunk_size)
        f_block = _slice_axis1_padded_2d(
            f,
            row_start,
            row_chunk_size,
            pad_value=float(_STREAMING_LOG_ZERO),
        )
        carried = tf.zeros([batch_size, row_chunk_size, state_dim], dtype=DTYPE)
        mass = tf.zeros([batch_size, row_chunk_size], dtype=DTYPE)

        def col_cond(col_start: tf.Tensor, _carried: tf.Tensor, _mass: tf.Tensor) -> tf.Tensor:
            return col_start < num_particles

        def col_body(col_start: tf.Tensor, carried_accum: tf.Tensor, mass_accum: tf.Tensor):
            key_block = _slice_axis1_padded_3d(scaled_x, col_start, col_chunk_size)
            particle_block = _slice_axis1_padded_3d(particles, col_start, col_chunk_size)
            g_block = _slice_axis1_padded_2d(
                g,
                col_start,
                col_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            logw_block = _slice_axis1_padded_2d(
                logw,
                col_start,
                col_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            norm_block = _slice_axis1_padded_2d(
                column_log_norm,
                col_start,
                col_chunk_size,
                pad_value=0.0,
            )
            cost = 0.5 * _pairwise_squared_cross(query_block, key_block)
            log_transport = (
                f_block[:, :, None]
                + g_block[:, None, :]
                - cost
            ) / eps[:, None, None]
            log_transport = (
                log_transport
                - norm_block[:, None, :]
                + log_n
                + logw_block[:, None, :]
            )
            transport_block = tf.exp(log_transport)
            carried_next = carried_accum + tf.matmul(transport_block, particle_block)
            mass_next = mass_accum + tf.reduce_sum(transport_block, axis=2)
            return col_start + col_chunk_tensor, carried_next, mass_next

        _, carried, mass = tf.while_loop(
            col_cond,
            col_body,
            loop_vars=(tf.constant(0, tf.int32), carried, mass),
            maximum_iterations=num_col_blocks,
        )
        block_index = row_start // row_chunk_tensor
        particle_ta = particle_ta.write(block_index, carried)
        mass_ta = mass_ta.write(block_index, mass)
        return row_start + row_chunk_tensor, particle_ta, mass_ta

    _, particle_blocks, mass_blocks = tf.while_loop(
        row_cond,
        row_body,
        loop_vars=(tf.constant(0, tf.int32), particle_blocks, mass_blocks),
        maximum_iterations=num_row_blocks,
    )
    stacked_particles = particle_blocks.stack()
    transposed_particles = tf.transpose(stacked_particles, [1, 0, 2, 3])
    transported = tf.reshape(
        transposed_particles,
        [batch_size, num_row_blocks * row_chunk_tensor, state_dim],
    )[:, :num_particles, :]
    stacked_mass = mass_blocks.stack()
    transposed_mass = tf.transpose(stacked_mass, [1, 0, 2])
    row_mass = tf.reshape(
        transposed_mass,
        [batch_size, num_row_blocks * row_chunk_tensor],
    )[:, :num_particles]
    row_residual = tf.reduce_max(tf.abs(row_mass - 1.0))
    return transported, tf.cast(row_residual, DTYPE)


def _filterflow_streaming_transport_from_potentials_jvp(
    scaled_x: tf.Tensor,
    particles: tf.Tensor,
    f: tf.Tensor,
    g: tf.Tensor,
    eps: tf.Tensor,
    logw: tf.Tensor,
    float_n: tf.Tensor,
    d_scaled_x: tf.Tensor,
    d_particles: tf.Tensor,
    d_f: tf.Tensor,
    d_g: tf.Tensor,
    d_logw: tf.Tensor,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    row_chunk_size = _validate_chunk_size(row_chunk_size, "row_chunk_size")
    col_chunk_size = _validate_chunk_size(col_chunk_size, "col_chunk_size")
    scaled_x = tf.cast(scaled_x, DTYPE)
    particles = tf.cast(particles, DTYPE)
    f = tf.cast(f, DTYPE)
    g = tf.cast(g, DTYPE)
    logw = tf.cast(logw, DTYPE)
    d_scaled_x = tf.cast(d_scaled_x, DTYPE)
    d_particles = tf.cast(d_particles, DTYPE)
    d_f = tf.cast(d_f, DTYPE)
    d_g = tf.cast(d_g, DTYPE)
    d_logw = tf.cast(d_logw, DTYPE)
    batch_size = tf.shape(scaled_x)[0]
    num_particles = tf.shape(scaled_x)[1]
    state_dim = tf.shape(particles)[2]
    param_dim = tf.shape(d_particles)[3]
    eps = tf.reshape(tf.cast(eps, DTYPE), [-1])
    log_n = tf.math.log(float_n)
    column_log_norm, d_column_log_norm = _filterflow_streaming_column_log_normalizer_jvp(
        scaled_x,
        f,
        g,
        eps,
        d_scaled_x,
        d_f,
        d_g,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    row_chunk_tensor = tf.cast(row_chunk_size, tf.int32)
    col_chunk_tensor = tf.cast(col_chunk_size, tf.int32)
    num_row_blocks = (num_particles + row_chunk_tensor - 1) // row_chunk_tensor
    num_col_blocks = (num_particles + col_chunk_tensor - 1) // col_chunk_tensor
    particle_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_row_blocks,
        element_shape=tf.TensorShape([None, row_chunk_size, None]),
    )
    tangent_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_row_blocks,
        element_shape=tf.TensorShape([None, row_chunk_size, None, None]),
    )
    mass_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_row_blocks,
        element_shape=tf.TensorShape([None, row_chunk_size]),
    )

    def row_cond(
        row_start: tf.Tensor,
        _particle_blocks: tf.TensorArray,
        _tangent_blocks: tf.TensorArray,
        _mass_blocks: tf.TensorArray,
    ) -> tf.Tensor:
        return row_start < num_particles

    def row_body(
        row_start: tf.Tensor,
        particle_ta: tf.TensorArray,
        tangent_ta: tf.TensorArray,
        mass_ta: tf.TensorArray,
    ):
        query_block = _slice_axis1_padded_3d(scaled_x, row_start, row_chunk_size)
        d_query_block = _slice_axis1_padded_4d(d_scaled_x, row_start, row_chunk_size)
        f_block = _slice_axis1_padded_2d(
            f,
            row_start,
            row_chunk_size,
            pad_value=float(_STREAMING_LOG_ZERO),
        )
        d_f_block = _slice_axis1_padded_3d(d_f, row_start, row_chunk_size)
        carried = tf.zeros([batch_size, row_chunk_size, state_dim], dtype=DTYPE)
        d_carried = tf.zeros(
            [batch_size, row_chunk_size, state_dim, param_dim],
            dtype=DTYPE,
        )
        mass = tf.zeros([batch_size, row_chunk_size], dtype=DTYPE)

        def col_cond(
            col_start: tf.Tensor,
            _carried: tf.Tensor,
            _d_carried: tf.Tensor,
            _mass: tf.Tensor,
        ) -> tf.Tensor:
            return col_start < num_particles

        def col_body(
            col_start: tf.Tensor,
            carried_accum: tf.Tensor,
            d_carried_accum: tf.Tensor,
            mass_accum: tf.Tensor,
        ):
            key_block = _slice_axis1_padded_3d(scaled_x, col_start, col_chunk_size)
            d_key_block = _slice_axis1_padded_4d(d_scaled_x, col_start, col_chunk_size)
            particle_block = _slice_axis1_padded_3d(particles, col_start, col_chunk_size)
            d_particle_block = _slice_axis1_padded_4d(
                d_particles,
                col_start,
                col_chunk_size,
            )
            g_block = _slice_axis1_padded_2d(
                g,
                col_start,
                col_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            d_g_block = _slice_axis1_padded_3d(d_g, col_start, col_chunk_size)
            logw_block = _slice_axis1_padded_2d(
                logw,
                col_start,
                col_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            d_logw_block = _slice_axis1_padded_3d(d_logw, col_start, col_chunk_size)
            norm_block = _slice_axis1_padded_2d(
                column_log_norm,
                col_start,
                col_chunk_size,
                pad_value=0.0,
            )
            d_norm_block = _slice_axis1_padded_3d(
                d_column_log_norm,
                col_start,
                col_chunk_size,
            )
            cost = 0.5 * _pairwise_squared_cross(query_block, key_block)
            logit = (
                f_block[:, :, None]
                + g_block[:, None, :]
                - cost
            ) / eps[:, None, None]
            log_transport = (
                logit
                - norm_block[:, None, :]
                + log_n
                + logw_block[:, None, :]
            )
            transport_block = tf.exp(log_transport)
            d_cost = _half_pairwise_squared_cross_jvp(
                query_block,
                key_block,
                d_query_block,
                d_key_block,
            )
            d_logit = (
                d_f_block[:, :, None, :]
                + d_g_block[:, None, :, :]
                - d_cost
            ) / eps[:, None, None, None]
            d_log_transport = (
                d_logit
                - d_norm_block[:, None, :, :]
                + d_logw_block[:, None, :, :]
            )
            transported_increment = tf.matmul(transport_block, particle_block)
            d_particle_increment = tf.einsum(
                "brc,bcdp->brdp",
                transport_block,
                d_particle_block,
            )
            d_transport_increment = tf.einsum(
                "brc,bcd,brcp->brdp",
                transport_block,
                particle_block,
                d_log_transport,
            )
            d_increment = d_particle_increment + d_transport_increment
            carried_next = carried_accum + transported_increment
            d_carried_next = d_carried_accum + d_increment
            mass_next = mass_accum + tf.reduce_sum(transport_block, axis=2)
            return (
                col_start + col_chunk_tensor,
                carried_next,
                d_carried_next,
                mass_next,
            )

        _, carried, d_carried, mass = tf.while_loop(
            col_cond,
            col_body,
            loop_vars=(tf.constant(0, tf.int32), carried, d_carried, mass),
            maximum_iterations=num_col_blocks,
        )
        block_index = row_start // row_chunk_tensor
        particle_ta = particle_ta.write(block_index, carried)
        tangent_ta = tangent_ta.write(block_index, d_carried)
        mass_ta = mass_ta.write(block_index, mass)
        return row_start + row_chunk_tensor, particle_ta, tangent_ta, mass_ta

    _, particle_blocks, tangent_blocks, mass_blocks = tf.while_loop(
        row_cond,
        row_body,
        loop_vars=(tf.constant(0, tf.int32), particle_blocks, tangent_blocks, mass_blocks),
        maximum_iterations=num_row_blocks,
    )
    stacked_particles = particle_blocks.stack()
    transposed_particles = tf.transpose(stacked_particles, [1, 0, 2, 3])
    transported = tf.reshape(
        transposed_particles,
        [batch_size, num_row_blocks * row_chunk_tensor, state_dim],
    )[:, :num_particles, :]
    stacked_tangents = tangent_blocks.stack()
    transposed_tangents = tf.transpose(stacked_tangents, [1, 0, 2, 3, 4])
    d_transported = tf.reshape(
        transposed_tangents,
        [batch_size, num_row_blocks * row_chunk_tensor, state_dim, param_dim],
    )[:, :num_particles, :, :]
    stacked_mass = mass_blocks.stack()
    transposed_mass = tf.transpose(stacked_mass, [1, 0, 2])
    row_mass = tf.reshape(
        transposed_mass,
        [batch_size, num_row_blocks * row_chunk_tensor],
    )[:, :num_particles]
    row_residual = tf.reduce_max(tf.abs(row_mass - 1.0))
    return transported, d_transported, tf.cast(row_residual, DTYPE)


def _filterflow_streaming_transport_from_potentials_vjp(
    scaled_x: tf.Tensor,
    particles: tf.Tensor,
    f: tf.Tensor,
    g: tf.Tensor,
    eps: tf.Tensor,
    logw: tf.Tensor,
    float_n: tf.Tensor,
    upstream: tf.Tensor,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    row_chunk_size = _validate_chunk_size(row_chunk_size, "row_chunk_size")
    col_chunk_size = _validate_chunk_size(col_chunk_size, "col_chunk_size")
    scaled_x = tf.cast(scaled_x, DTYPE)
    particles = tf.cast(particles, DTYPE)
    f = tf.cast(f, DTYPE)
    g = tf.cast(g, DTYPE)
    logw = tf.cast(logw, DTYPE)
    upstream = tf.cast(upstream, DTYPE)
    batch_size = tf.shape(scaled_x)[0]
    num_particles = tf.shape(scaled_x)[1]
    state_dim = tf.shape(scaled_x)[2]
    eps = _epsilon_per_batch(eps, batch_size)
    log_n = tf.math.log(tf.cast(float_n, DTYPE))
    column_log_norm = _filterflow_streaming_column_log_normalizer(
        scaled_x,
        f,
        g,
        eps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    row_chunk_tensor = tf.cast(row_chunk_size, tf.int32)
    col_chunk_tensor = tf.cast(col_chunk_size, tf.int32)
    num_row_blocks = (num_particles + row_chunk_tensor - 1) // row_chunk_tensor
    num_col_blocks = (num_particles + col_chunk_tensor - 1) // col_chunk_tensor
    s_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_col_blocks,
        element_shape=tf.TensorShape([None, col_chunk_size]),
        clear_after_read=False,
    )
    particle_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_col_blocks,
        element_shape=tf.TensorShape([None, col_chunk_size, None]),
    )

    def col_cond(
        col_start: tf.Tensor,
        _s_blocks: tf.TensorArray,
        _particle_blocks: tf.TensorArray,
    ) -> tf.Tensor:
        return col_start < num_particles

    def col_body(
        col_start: tf.Tensor,
        s_ta: tf.TensorArray,
        particle_ta: tf.TensorArray,
    ):
        key_block = _slice_axis1_padded_3d(scaled_x, col_start, col_chunk_size)
        particle_block = _slice_axis1_padded_3d(
            particles,
            col_start,
            col_chunk_size,
        )
        g_block = _slice_axis1_padded_2d(
            g,
            col_start,
            col_chunk_size,
            pad_value=float(_STREAMING_LOG_ZERO),
        )
        logw_block = _slice_axis1_padded_2d(
            logw,
            col_start,
            col_chunk_size,
            pad_value=float(_STREAMING_LOG_ZERO),
        )
        norm_block = _slice_axis1_padded_2d(
            column_log_norm,
            col_start,
            col_chunk_size,
            pad_value=0.0,
        )
        s_block = tf.zeros([batch_size, col_chunk_size], dtype=DTYPE)
        d_particle_block = tf.zeros([batch_size, col_chunk_size, state_dim], dtype=DTYPE)

        def first_row_cond(
            row_start: tf.Tensor,
            _s_block: tf.Tensor,
            _d_particle_block: tf.Tensor,
        ) -> tf.Tensor:
            return row_start < num_particles

        def first_row_body(
            row_start: tf.Tensor,
            s_accum: tf.Tensor,
            d_particle_accum: tf.Tensor,
        ):
            query_block = _slice_axis1_padded_3d(scaled_x, row_start, row_chunk_size)
            f_block = _slice_axis1_padded_2d(
                f,
                row_start,
                row_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            upstream_block = _slice_axis1_padded_3d(
                upstream,
                row_start,
                row_chunk_size,
            )
            cost = 0.5 * _pairwise_squared_cross(query_block, key_block)
            log_transport = (
                f_block[:, :, None]
                + g_block[:, None, :]
                - cost
            ) / eps[:, None, None]
            log_transport = (
                log_transport
                - norm_block[:, None, :]
                + log_n
                + logw_block[:, None, :]
            )
            transport_block = tf.exp(log_transport)
            d_transport = tf.reduce_sum(
                upstream_block[:, :, None, :] * particle_block[:, None, :, :],
                axis=3,
            )
            weighted = d_transport * transport_block
            s_next = s_accum + tf.reduce_sum(weighted, axis=1)
            d_particle_next = d_particle_accum + tf.matmul(
                transport_block,
                upstream_block,
                transpose_a=True,
            )
            return row_start + row_chunk_tensor, s_next, d_particle_next

        _, s_block, d_particle_block = tf.while_loop(
            first_row_cond,
            first_row_body,
            loop_vars=(tf.constant(0, tf.int32), s_block, d_particle_block),
            maximum_iterations=num_row_blocks,
        )
        block_index = col_start // col_chunk_tensor
        s_ta = s_ta.write(block_index, s_block)
        particle_ta = particle_ta.write(block_index, d_particle_block)
        return col_start + col_chunk_tensor, s_ta, particle_ta

    _, s_blocks, particle_blocks = tf.while_loop(
        col_cond,
        col_body,
        loop_vars=(tf.constant(0, tf.int32), s_blocks, particle_blocks),
        maximum_iterations=num_col_blocks,
    )

    query_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_row_blocks,
        element_shape=tf.TensorShape([None, row_chunk_size, None]),
    )
    f_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_row_blocks,
        element_shape=tf.TensorShape([None, row_chunk_size]),
    )

    def query_cond(
        row_start: tf.Tensor,
        _query_blocks: tf.TensorArray,
        _f_blocks: tf.TensorArray,
    ) -> tf.Tensor:
        return row_start < num_particles

    def query_body(
        row_start: tf.Tensor,
        query_ta: tf.TensorArray,
        f_ta: tf.TensorArray,
    ):
        query_block = _slice_axis1_padded_3d(scaled_x, row_start, row_chunk_size)
        f_block = _slice_axis1_padded_2d(
            f,
            row_start,
            row_chunk_size,
            pad_value=float(_STREAMING_LOG_ZERO),
        )
        upstream_block = _slice_axis1_padded_3d(
            upstream,
            row_start,
            row_chunk_size,
        )
        d_query_block = tf.zeros([batch_size, row_chunk_size, state_dim], dtype=DTYPE)
        d_f_block = tf.zeros([batch_size, row_chunk_size], dtype=DTYPE)

        def col_accum_cond(
            col_start: tf.Tensor,
            _d_query_block: tf.Tensor,
            _d_f_block: tf.Tensor,
        ) -> tf.Tensor:
            return col_start < num_particles

        def col_accum_body(
            col_start: tf.Tensor,
            d_query_accum: tf.Tensor,
            d_f_accum: tf.Tensor,
        ):
            col_index = col_start // col_chunk_tensor
            s_block = s_blocks.read(col_index)
            key_block = _slice_axis1_padded_3d(scaled_x, col_start, col_chunk_size)
            particle_block = _slice_axis1_padded_3d(
                particles,
                col_start,
                col_chunk_size,
            )
            g_block = _slice_axis1_padded_2d(
                g,
                col_start,
                col_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            logw_block = _slice_axis1_padded_2d(
                logw,
                col_start,
                col_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            norm_block = _slice_axis1_padded_2d(
                column_log_norm,
                col_start,
                col_chunk_size,
                pad_value=0.0,
            )
            cost = 0.5 * _pairwise_squared_cross(query_block, key_block)
            logit = (
                f_block[:, :, None]
                + g_block[:, None, :]
                - cost
            ) / eps[:, None, None]
            log_transport = (
                logit
                - norm_block[:, None, :]
                + log_n
                + logw_block[:, None, :]
            )
            transport_block = tf.exp(log_transport)
            probs = tf.exp(logit - norm_block[:, None, :])
            d_transport = tf.reduce_sum(
                upstream_block[:, :, None, :] * particle_block[:, None, :, :],
                axis=3,
            )
            weighted = d_transport * transport_block
            d_logit = weighted - probs * s_block[:, None, :]
            d_cost = -d_logit / eps[:, None, None]
            diff = query_block[:, :, None, :] - key_block[:, None, :, :]
            d_query_next = d_query_accum + tf.reduce_sum(
                d_cost[:, :, :, None] * diff,
                axis=2,
            )
            d_f_next = d_f_accum + tf.reduce_sum(
                d_logit / eps[:, None, None],
                axis=2,
            )
            return col_start + col_chunk_tensor, d_query_next, d_f_next

        _, d_query_block, d_f_block = tf.while_loop(
            col_accum_cond,
            col_accum_body,
            loop_vars=(tf.constant(0, tf.int32), d_query_block, d_f_block),
            maximum_iterations=num_col_blocks,
        )
        block_index = row_start // row_chunk_tensor
        query_ta = query_ta.write(block_index, d_query_block)
        f_ta = f_ta.write(block_index, d_f_block)
        return row_start + row_chunk_tensor, query_ta, f_ta

    _, query_blocks, f_blocks = tf.while_loop(
        query_cond,
        query_body,
        loop_vars=(tf.constant(0, tf.int32), query_blocks, f_blocks),
        maximum_iterations=num_row_blocks,
    )

    key_blocks = tf.TensorArray(
        dtype=DTYPE,
        size=num_col_blocks,
        element_shape=tf.TensorShape([None, col_chunk_size, None]),
    )

    def key_cond(
        col_start: tf.Tensor,
        _key_blocks: tf.TensorArray,
    ) -> tf.Tensor:
        return col_start < num_particles

    def key_body(
        col_start: tf.Tensor,
        key_ta: tf.TensorArray,
    ):
        col_index = col_start // col_chunk_tensor
        s_block = s_blocks.read(col_index)
        key_block = _slice_axis1_padded_3d(scaled_x, col_start, col_chunk_size)
        particle_block = _slice_axis1_padded_3d(
            particles,
            col_start,
            col_chunk_size,
        )
        g_block = _slice_axis1_padded_2d(
            g,
            col_start,
            col_chunk_size,
            pad_value=float(_STREAMING_LOG_ZERO),
        )
        logw_block = _slice_axis1_padded_2d(
            logw,
            col_start,
            col_chunk_size,
            pad_value=float(_STREAMING_LOG_ZERO),
        )
        norm_block = _slice_axis1_padded_2d(
            column_log_norm,
            col_start,
            col_chunk_size,
            pad_value=0.0,
        )
        d_key_block = tf.zeros([batch_size, col_chunk_size, state_dim], dtype=DTYPE)

        def row_accum_cond(
            row_start: tf.Tensor,
            _d_key_block: tf.Tensor,
        ) -> tf.Tensor:
            return row_start < num_particles

        def row_accum_body(
            row_start: tf.Tensor,
            d_key_accum: tf.Tensor,
        ):
            query_block = _slice_axis1_padded_3d(scaled_x, row_start, row_chunk_size)
            f_block = _slice_axis1_padded_2d(
                f,
                row_start,
                row_chunk_size,
                pad_value=float(_STREAMING_LOG_ZERO),
            )
            upstream_block = _slice_axis1_padded_3d(
                upstream,
                row_start,
                row_chunk_size,
            )
            cost = 0.5 * _pairwise_squared_cross(query_block, key_block)
            logit = (
                f_block[:, :, None]
                + g_block[:, None, :]
                - cost
            ) / eps[:, None, None]
            log_transport = (
                logit
                - norm_block[:, None, :]
                + log_n
                + logw_block[:, None, :]
            )
            transport_block = tf.exp(log_transport)
            probs = tf.exp(logit - norm_block[:, None, :])
            d_transport = tf.reduce_sum(
                upstream_block[:, :, None, :] * particle_block[:, None, :, :],
                axis=3,
            )
            weighted = d_transport * transport_block
            d_logit = weighted - probs * s_block[:, None, :]
            d_cost = -d_logit / eps[:, None, None]
            diff = query_block[:, :, None, :] - key_block[:, None, :, :]
            d_key_next = d_key_accum + tf.reduce_sum(
                d_cost[:, :, :, None] * (-diff),
                axis=1,
            )
            return row_start + row_chunk_tensor, d_key_next

        _, d_key_block = tf.while_loop(
            row_accum_cond,
            row_accum_body,
            loop_vars=(tf.constant(0, tf.int32), d_key_block),
            maximum_iterations=num_row_blocks,
        )
        key_ta = key_ta.write(col_index, d_key_block)
        return col_start + col_chunk_tensor, key_ta

    _, key_blocks = tf.while_loop(
        key_cond,
        key_body,
        loop_vars=(tf.constant(0, tf.int32), key_blocks),
        maximum_iterations=num_col_blocks,
    )

    stacked_query = query_blocks.stack()
    transposed_query = tf.transpose(stacked_query, [1, 0, 2, 3])
    d_query = tf.reshape(
        transposed_query,
        [batch_size, num_row_blocks * row_chunk_tensor, state_dim],
    )[:, :num_particles, :]
    stacked_key = key_blocks.stack()
    transposed_key = tf.transpose(stacked_key, [1, 0, 2, 3])
    d_key = tf.reshape(
        transposed_key,
        [batch_size, num_col_blocks * col_chunk_tensor, state_dim],
    )[:, :num_particles, :]
    stacked_particles = particle_blocks.stack()
    transposed_particles = tf.transpose(stacked_particles, [1, 0, 2, 3])
    d_particles = tf.reshape(
        transposed_particles,
        [batch_size, num_col_blocks * col_chunk_tensor, state_dim],
    )[:, :num_particles, :]
    stacked_f = f_blocks.stack()
    transposed_f = tf.transpose(stacked_f, [1, 0, 2])
    d_f = tf.reshape(
        transposed_f,
        [batch_size, num_row_blocks * row_chunk_tensor],
    )[:, :num_particles]
    stacked_s = s_blocks.stack()
    transposed_s = tf.transpose(stacked_s, [1, 0, 2])
    d_logw = tf.reshape(
        transposed_s,
        [batch_size, num_col_blocks * col_chunk_tensor],
    )[:, :num_particles]
    d_scaled_x = d_query + d_key
    d_g = tf.zeros_like(g)
    return d_scaled_x, d_particles, d_f, d_g, d_logw


def _filterflow_streaming_transport(
    scaled_x: tf.Tensor,
    particles: tf.Tensor,
    logw: tf.Tensor,
    eps: tf.Tensor,
    scaling: tf.Tensor,
    threshold: tf.Tensor,
    max_iter: tf.Tensor,
    n: tf.Tensor,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
    transport_ad_mode: str = "stabilized",
    warmstart_state: AnnealedTransportWarmstartStateTF | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    _validate_transport_ad_mode(transport_ad_mode)
    float_n = tf.cast(n, DTYPE)
    log_n = tf.math.log(float_n)
    uniform_log_weight = -log_n * tf.ones_like(logw)
    alpha, beta, _, _, iterations = _filterflow_streaming_sinkhorn_potentials(
        logw,
        scaled_x,
        uniform_log_weight,
        scaled_x,
        eps,
        scaling,
        threshold,
        max_iter,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        transport_ad_mode=transport_ad_mode,
        warmstart_state=warmstart_state,
    )
    transported, row_residual = _filterflow_streaming_transport_from_potentials(
        scaled_x,
        particles,
        alpha,
        beta,
        eps,
        logw,
        float_n,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    column_residual = tf.constant(0.0, DTYPE)
    return transported, iterations, row_residual, column_residual


def _filterflow_streaming_finite_sinkhorn_potentials_stopped_scale_keys(
    log_alpha: tf.Tensor,
    log_beta: tf.Tensor,
    x: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    steps = _validate_manual_dense_finite_route_inputs(epsilon, steps)
    key_x = tf.stop_gradient(x)
    running = tf.cast(epsilon0, x.dtype)
    eps = tf.cast(epsilon, x.dtype)
    scaling_factor = tf.cast(scaling, x.dtype) ** 2
    a_y = _filterflow_streaming_softmin(
        running,
        x,
        key_x,
        log_alpha,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    b_x = _filterflow_streaming_softmin(
        running,
        x,
        key_x,
        log_beta,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    a_x = _filterflow_streaming_softmin(
        running,
        x,
        key_x,
        log_alpha,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    b_y = _filterflow_streaming_softmin(
        running,
        x,
        key_x,
        log_beta,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    steps_tensor = tf.constant(steps, tf.int32)

    def cond(
        iteration: tf.Tensor,
        _running: tf.Tensor,
        _a_y: tf.Tensor,
        _b_x: tf.Tensor,
        _a_x: tf.Tensor,
        _b_y: tf.Tensor,
    ) -> tf.Tensor:
        del _running, _a_y, _b_x, _a_x, _b_y
        return iteration < steps_tensor

    def body(
        iteration: tf.Tensor,
        running_value: tf.Tensor,
        a_y_value: tf.Tensor,
        b_x_value: tf.Tensor,
        a_x_value: tf.Tensor,
        b_y_value: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        running_ = tf.reshape(running_value, [-1, 1])
        at_y = _filterflow_streaming_softmin(
            running_value,
            x,
            key_x,
            log_alpha + b_x_value / running_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        bt_x = _filterflow_streaming_softmin(
            running_value,
            x,
            key_x,
            log_beta + a_y_value / running_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        at_x = _filterflow_streaming_softmin(
            running_value,
            x,
            key_x,
            log_alpha + a_x_value / running_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        bt_y = _filterflow_streaming_softmin(
            running_value,
            x,
            key_x,
            log_beta + b_y_value / running_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        return (
            iteration + 1,
            tf.maximum(running_value * scaling_factor, eps),
            0.5 * (a_y_value + at_y),
            0.5 * (b_x_value + bt_x),
            0.5 * (a_x_value + at_x),
            0.5 * (b_y_value + bt_y),
        )

    _, running, a_y, b_x, a_x, b_y = tf.while_loop(
        cond,
        body,
        loop_vars=(tf.constant(0, tf.int32), running, a_y, b_x, a_x, b_y),
        maximum_iterations=steps,
    )
    eps_ = tf.reshape(eps, [-1, 1])
    return (
        _filterflow_streaming_softmin(
            eps,
            x,
            key_x,
            log_alpha + b_x / eps_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        ),
        _filterflow_streaming_softmin(
            eps,
            x,
            key_x,
            log_beta + a_y / eps_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        ),
    )


def _filterflow_streaming_finite_sinkhorn_potentials_total_vjp(
    log_alpha: tf.Tensor,
    log_beta: tf.Tensor,
    x: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Finite streaming Sinkhorn potentials for the total-derivative route.

    Unlike the historical ``*_stopped_scale_keys`` helper, this value helper
    does not detach the cost keys.  It is intended for small/reference
    total-VJP repairs where TensorFlow differentiates the local finite Sinkhorn
    map while the outer filter still uses an explicit reverse scan.
    """

    steps = _validate_manual_dense_finite_route_inputs(epsilon, steps)
    key_x = x
    running = tf.cast(epsilon0, x.dtype)
    eps = tf.cast(epsilon, x.dtype)
    scaling_factor = tf.cast(scaling, x.dtype) ** 2
    a_y = _filterflow_streaming_softmin(
        running,
        x,
        key_x,
        log_alpha,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    b_x = _filterflow_streaming_softmin(
        running,
        x,
        key_x,
        log_beta,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    a_x = _filterflow_streaming_softmin(
        running,
        x,
        key_x,
        log_alpha,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    b_y = _filterflow_streaming_softmin(
        running,
        x,
        key_x,
        log_beta,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    steps_tensor = tf.constant(steps, tf.int32)

    def cond(
        iteration: tf.Tensor,
        _running: tf.Tensor,
        _a_y: tf.Tensor,
        _b_x: tf.Tensor,
        _a_x: tf.Tensor,
        _b_y: tf.Tensor,
    ) -> tf.Tensor:
        del _running, _a_y, _b_x, _a_x, _b_y
        return iteration < steps_tensor

    def body(
        iteration: tf.Tensor,
        running_value: tf.Tensor,
        a_y_value: tf.Tensor,
        b_x_value: tf.Tensor,
        a_x_value: tf.Tensor,
        b_y_value: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        running_ = tf.reshape(running_value, [-1, 1])
        at_y = _filterflow_streaming_softmin(
            running_value,
            x,
            key_x,
            log_alpha + b_x_value / running_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        bt_x = _filterflow_streaming_softmin(
            running_value,
            x,
            key_x,
            log_beta + a_y_value / running_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        at_x = _filterflow_streaming_softmin(
            running_value,
            x,
            key_x,
            log_alpha + a_x_value / running_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        bt_y = _filterflow_streaming_softmin(
            running_value,
            x,
            key_x,
            log_beta + b_y_value / running_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        return (
            iteration + 1,
            tf.maximum(running_value * scaling_factor, eps),
            0.5 * (a_y_value + at_y),
            0.5 * (b_x_value + bt_x),
            0.5 * (a_x_value + at_x),
            0.5 * (b_y_value + bt_y),
        )

    _, running, a_y, b_x, a_x, b_y = tf.while_loop(
        cond,
        body,
        loop_vars=(tf.constant(0, tf.int32), running, a_y, b_x, a_x, b_y),
        maximum_iterations=steps,
    )
    eps_ = tf.reshape(eps, [-1, 1])
    return (
        _filterflow_streaming_softmin(
            eps,
            x,
            key_x,
            log_alpha + b_x / eps_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        ),
        _filterflow_streaming_softmin(
            eps,
            x,
            key_x,
            log_beta + a_y / eps_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        ),
    )


def _filterflow_streaming_finite_sinkhorn_potentials_vjp_stopped_scale_keys(
    log_alpha: tf.Tensor,
    log_beta: tf.Tensor,
    x: tf.Tensor,
    upstream_alpha: tf.Tensor,
    upstream_beta: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    steps = _validate_manual_dense_finite_route_inputs(epsilon, steps)
    log_alpha = tf.cast(log_alpha, DTYPE)
    log_beta = tf.cast(log_beta, DTYPE)
    x = tf.cast(x, DTYPE)
    upstream_alpha = tf.cast(upstream_alpha, DTYPE)
    upstream_beta = tf.cast(upstream_beta, DTYPE)
    key_x = tf.stop_gradient(x)
    running = tf.cast(epsilon0, DTYPE)
    eps = _epsilon_per_batch(tf.cast(epsilon, DTYPE), tf.shape(x)[0])
    scaling_factor = tf.cast(scaling, DTYPE) ** 2
    a_y = _filterflow_streaming_softmin(
        running,
        x,
        key_x,
        log_alpha,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    b_x = _filterflow_streaming_softmin(
        running,
        x,
        key_x,
        log_beta,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    a_x = _filterflow_streaming_softmin(
        running,
        x,
        key_x,
        log_alpha,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    b_y = _filterflow_streaming_softmin(
        running,
        x,
        key_x,
        log_beta,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    running_ta = tf.TensorArray(
        dtype=DTYPE,
        size=steps,
        element_shape=running.shape,
        clear_after_read=False,
    )
    a_y_ta = tf.TensorArray(
        dtype=DTYPE,
        size=steps,
        element_shape=a_y.shape,
        clear_after_read=False,
    )
    b_x_ta = tf.TensorArray(
        dtype=DTYPE,
        size=steps,
        element_shape=b_x.shape,
        clear_after_read=False,
    )
    a_x_ta = tf.TensorArray(
        dtype=DTYPE,
        size=steps,
        element_shape=a_x.shape,
        clear_after_read=False,
    )
    b_y_ta = tf.TensorArray(
        dtype=DTYPE,
        size=steps,
        element_shape=b_y.shape,
        clear_after_read=False,
    )
    steps_tensor = tf.constant(steps, tf.int32)

    def forward_cond(
        iteration: tf.Tensor,
        _running: tf.Tensor,
        _a_y: tf.Tensor,
        _b_x: tf.Tensor,
        _a_x: tf.Tensor,
        _b_y: tf.Tensor,
        _running_ta: tf.TensorArray,
        _a_y_ta: tf.TensorArray,
        _b_x_ta: tf.TensorArray,
        _a_x_ta: tf.TensorArray,
        _b_y_ta: tf.TensorArray,
    ) -> tf.Tensor:
        del _running, _a_y, _b_x, _a_x, _b_y
        del _running_ta, _a_y_ta, _b_x_ta, _a_x_ta, _b_y_ta
        return iteration < steps_tensor

    def forward_body(
        iteration: tf.Tensor,
        running_value: tf.Tensor,
        a_y_value: tf.Tensor,
        b_x_value: tf.Tensor,
        a_x_value: tf.Tensor,
        b_y_value: tf.Tensor,
        running_records: tf.TensorArray,
        a_y_records: tf.TensorArray,
        b_x_records: tf.TensorArray,
        a_x_records: tf.TensorArray,
        b_y_records: tf.TensorArray,
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.TensorArray,
        tf.TensorArray,
        tf.TensorArray,
        tf.TensorArray,
        tf.TensorArray,
    ]:
        running_records = running_records.write(iteration, running_value)
        a_y_records = a_y_records.write(iteration, a_y_value)
        b_x_records = b_x_records.write(iteration, b_x_value)
        a_x_records = a_x_records.write(iteration, a_x_value)
        b_y_records = b_y_records.write(iteration, b_y_value)
        running_ = tf.reshape(running_value, [-1, 1])
        at_y = _filterflow_streaming_softmin(
            running_value,
            x,
            key_x,
            log_alpha + b_x_value / running_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        bt_x = _filterflow_streaming_softmin(
            running_value,
            x,
            key_x,
            log_beta + a_y_value / running_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        at_x = _filterflow_streaming_softmin(
            running_value,
            x,
            key_x,
            log_alpha + a_x_value / running_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        bt_y = _filterflow_streaming_softmin(
            running_value,
            x,
            key_x,
            log_beta + b_y_value / running_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        return (
            iteration + 1,
            tf.maximum(running_value * scaling_factor, eps),
            0.5 * (a_y_value + at_y),
            0.5 * (b_x_value + bt_x),
            0.5 * (a_x_value + at_x),
            0.5 * (b_y_value + bt_y),
            running_records,
            a_y_records,
            b_x_records,
            a_x_records,
            b_y_records,
        )

    (
        _,
        running,
        a_y,
        b_x,
        a_x,
        b_y,
        running_ta,
        a_y_ta,
        b_x_ta,
        a_x_ta,
        b_y_ta,
    ) = tf.while_loop(
        forward_cond,
        forward_body,
        loop_vars=(
            tf.constant(0, tf.int32),
            running,
            a_y,
            b_x,
            a_x,
            b_y,
            running_ta,
            a_y_ta,
            b_x_ta,
            a_x_ta,
            b_y_ta,
        ),
        maximum_iterations=steps,
    )

    d_log_alpha = tf.zeros_like(log_alpha)
    d_log_beta = tf.zeros_like(log_beta)
    d_x = tf.zeros_like(x)
    d_a_y = tf.zeros_like(log_alpha)
    d_b_x = tf.zeros_like(log_beta)
    d_a_x = tf.zeros_like(log_alpha)
    d_b_y = tf.zeros_like(log_beta)
    eps_ = tf.reshape(eps, [-1, 1])

    dc, dv = _filterflow_streaming_softmin_vjp(
        eps,
        x,
        key_x,
        log_alpha + b_x / eps_,
        upstream_alpha,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        stop_keys=True,
    )[0::2]
    d_x += dc
    d_log_alpha += dv
    d_b_x += dv / eps_
    dc, dv = _filterflow_streaming_softmin_vjp(
        eps,
        x,
        key_x,
        log_beta + a_y / eps_,
        upstream_beta,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        stop_keys=True,
    )[0::2]
    d_x += dc
    d_log_beta += dv
    d_a_y += dv / eps_

    def reverse_cond(
        iteration: tf.Tensor,
        _d_log_alpha: tf.Tensor,
        _d_log_beta: tf.Tensor,
        _d_x: tf.Tensor,
        _d_a_y: tf.Tensor,
        _d_b_x: tf.Tensor,
        _d_a_x: tf.Tensor,
        _d_b_y: tf.Tensor,
    ) -> tf.Tensor:
        del _d_log_alpha, _d_log_beta, _d_x, _d_a_y, _d_b_x, _d_a_x, _d_b_y
        return iteration > 0

    def reverse_body(
        iteration: tf.Tensor,
        d_log_alpha_value: tf.Tensor,
        d_log_beta_value: tf.Tensor,
        d_x_value: tf.Tensor,
        d_a_y_value: tf.Tensor,
        d_b_x_value: tf.Tensor,
        d_a_x_value: tf.Tensor,
        d_b_y_value: tf.Tensor,
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
    ]:
        state_index = iteration - 1
        old_running = running_ta.read(state_index)
        old_a_y = a_y_ta.read(state_index)
        old_b_x = b_x_ta.read(state_index)
        old_a_x = a_x_ta.read(state_index)
        old_b_y = b_y_ta.read(state_index)
        running_ = tf.reshape(old_running, [-1, 1])
        bar_at_y = 0.5 * d_a_y_value
        bar_bt_x = 0.5 * d_b_x_value
        bar_at_x = 0.5 * d_a_x_value
        bar_bt_y = 0.5 * d_b_y_value

        old_d_a_y = 0.5 * d_a_y_value
        old_d_b_x = 0.5 * d_b_x_value
        old_d_a_x = 0.5 * d_a_x_value
        old_d_b_y = 0.5 * d_b_y_value

        dc, dv = _filterflow_streaming_softmin_vjp(
            old_running,
            x,
            key_x,
            log_alpha + old_b_x / running_,
            bar_at_y,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            stop_keys=True,
        )[0::2]
        d_x_value += dc
        d_log_alpha_value += dv
        old_d_b_x += dv / running_
        dc, dv = _filterflow_streaming_softmin_vjp(
            old_running,
            x,
            key_x,
            log_beta + old_a_y / running_,
            bar_bt_x,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            stop_keys=True,
        )[0::2]
        d_x_value += dc
        d_log_beta_value += dv
        old_d_a_y += dv / running_
        dc, dv = _filterflow_streaming_softmin_vjp(
            old_running,
            x,
            key_x,
            log_alpha + old_a_x / running_,
            bar_at_x,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            stop_keys=True,
        )[0::2]
        d_x_value += dc
        d_log_alpha_value += dv
        old_d_a_x += dv / running_
        dc, dv = _filterflow_streaming_softmin_vjp(
            old_running,
            x,
            key_x,
            log_beta + old_b_y / running_,
            bar_bt_y,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            stop_keys=True,
        )[0::2]
        d_x_value += dc
        d_log_beta_value += dv
        old_d_b_y += dv / running_

        return (
            state_index,
            d_log_alpha_value,
            d_log_beta_value,
            d_x_value,
            old_d_a_y,
            old_d_b_x,
            old_d_a_x,
            old_d_b_y,
        )

    (
        _,
        d_log_alpha,
        d_log_beta,
        d_x,
        d_a_y,
        d_b_x,
        d_a_x,
        d_b_y,
    ) = tf.while_loop(
        reverse_cond,
        reverse_body,
        loop_vars=(
            tf.constant(steps, tf.int32),
            d_log_alpha,
            d_log_beta,
            d_x,
            d_a_y,
            d_b_x,
            d_a_x,
            d_b_y,
        ),
        maximum_iterations=steps,
    )

    dc, dv = _filterflow_streaming_softmin_vjp(
        epsilon0,
        x,
        key_x,
        log_beta,
        d_b_y,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        stop_keys=True,
    )[0::2]
    d_x += dc
    d_log_beta += dv
    dc, dv = _filterflow_streaming_softmin_vjp(
        epsilon0,
        x,
        key_x,
        log_alpha,
        d_a_x,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        stop_keys=True,
    )[0::2]
    d_x += dc
    d_log_alpha += dv
    dc, dv = _filterflow_streaming_softmin_vjp(
        epsilon0,
        x,
        key_x,
        log_beta,
        d_b_x,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        stop_keys=True,
    )[0::2]
    d_x += dc
    d_log_beta += dv
    dc, dv = _filterflow_streaming_softmin_vjp(
        epsilon0,
        x,
        key_x,
        log_alpha,
        d_a_y,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        stop_keys=True,
    )[0::2]
    d_x += dc
    d_log_alpha += dv
    return d_log_alpha, d_log_beta, d_x


def _filterflow_streaming_finite_sinkhorn_potentials_vjp_total(
    log_alpha: tf.Tensor,
    log_beta: tf.Tensor,
    x: tf.Tensor,
    upstream_alpha: tf.Tensor,
    upstream_beta: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """No-tape total VJP for the finite streaming Sinkhorn potential recursion."""

    steps = _validate_manual_dense_finite_route_inputs(epsilon, steps)
    log_alpha = tf.cast(log_alpha, DTYPE)
    log_beta = tf.cast(log_beta, DTYPE)
    x = tf.cast(x, DTYPE)
    upstream_alpha = tf.cast(upstream_alpha, DTYPE)
    upstream_beta = tf.cast(upstream_beta, DTYPE)
    batch_size = tf.shape(x)[0]
    key_x = x
    initial_running = _epsilon_per_batch(tf.cast(epsilon0, DTYPE), batch_size)
    running = initial_running
    eps = _epsilon_per_batch(tf.cast(epsilon, DTYPE), batch_size)
    raw_epsilon0 = tf.cast(epsilon0, DTYPE)
    scaling_factor = tf.cast(scaling, DTYPE) ** 2
    a_y = _filterflow_streaming_softmin(
        running,
        x,
        key_x,
        log_alpha,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    b_x = _filterflow_streaming_softmin(
        running,
        x,
        key_x,
        log_beta,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    a_x = _filterflow_streaming_softmin(
        running,
        x,
        key_x,
        log_alpha,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    b_y = _filterflow_streaming_softmin(
        running,
        x,
        key_x,
        log_beta,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    running_ta = tf.TensorArray(
        dtype=DTYPE,
        size=steps,
        element_shape=running.shape,
        clear_after_read=False,
    )
    a_y_ta = tf.TensorArray(
        dtype=DTYPE,
        size=steps,
        element_shape=a_y.shape,
        clear_after_read=False,
    )
    b_x_ta = tf.TensorArray(
        dtype=DTYPE,
        size=steps,
        element_shape=b_x.shape,
        clear_after_read=False,
    )
    a_x_ta = tf.TensorArray(
        dtype=DTYPE,
        size=steps,
        element_shape=a_x.shape,
        clear_after_read=False,
    )
    b_y_ta = tf.TensorArray(
        dtype=DTYPE,
        size=steps,
        element_shape=b_y.shape,
        clear_after_read=False,
    )
    steps_tensor = tf.constant(steps, tf.int32)

    def forward_cond(
        iteration: tf.Tensor,
        _running: tf.Tensor,
        _a_y: tf.Tensor,
        _b_x: tf.Tensor,
        _a_x: tf.Tensor,
        _b_y: tf.Tensor,
        _running_ta: tf.TensorArray,
        _a_y_ta: tf.TensorArray,
        _b_x_ta: tf.TensorArray,
        _a_x_ta: tf.TensorArray,
        _b_y_ta: tf.TensorArray,
    ) -> tf.Tensor:
        del _running, _a_y, _b_x, _a_x, _b_y
        del _running_ta, _a_y_ta, _b_x_ta, _a_x_ta, _b_y_ta
        return iteration < steps_tensor

    def forward_body(
        iteration: tf.Tensor,
        running_value: tf.Tensor,
        a_y_value: tf.Tensor,
        b_x_value: tf.Tensor,
        a_x_value: tf.Tensor,
        b_y_value: tf.Tensor,
        running_records: tf.TensorArray,
        a_y_records: tf.TensorArray,
        b_x_records: tf.TensorArray,
        a_x_records: tf.TensorArray,
        b_y_records: tf.TensorArray,
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.TensorArray,
        tf.TensorArray,
        tf.TensorArray,
        tf.TensorArray,
        tf.TensorArray,
    ]:
        running_records = running_records.write(iteration, running_value)
        a_y_records = a_y_records.write(iteration, a_y_value)
        b_x_records = b_x_records.write(iteration, b_x_value)
        a_x_records = a_x_records.write(iteration, a_x_value)
        b_y_records = b_y_records.write(iteration, b_y_value)
        running_ = tf.reshape(running_value, [-1, 1])
        at_y = _filterflow_streaming_softmin(
            running_value,
            x,
            key_x,
            log_alpha + b_x_value / running_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        bt_x = _filterflow_streaming_softmin(
            running_value,
            x,
            key_x,
            log_beta + a_y_value / running_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        at_x = _filterflow_streaming_softmin(
            running_value,
            x,
            key_x,
            log_alpha + a_x_value / running_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        bt_y = _filterflow_streaming_softmin(
            running_value,
            x,
            key_x,
            log_beta + b_y_value / running_,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        return (
            iteration + 1,
            tf.maximum(running_value * scaling_factor, eps),
            0.5 * (a_y_value + at_y),
            0.5 * (b_x_value + bt_x),
            0.5 * (a_x_value + at_x),
            0.5 * (b_y_value + bt_y),
            running_records,
            a_y_records,
            b_x_records,
            a_x_records,
            b_y_records,
        )

    (
        _,
        running,
        a_y,
        b_x,
        a_x,
        b_y,
        running_ta,
        a_y_ta,
        b_x_ta,
        a_x_ta,
        b_y_ta,
    ) = tf.while_loop(
        forward_cond,
        forward_body,
        loop_vars=(
            tf.constant(0, tf.int32),
            running,
            a_y,
            b_x,
            a_x,
            b_y,
            running_ta,
            a_y_ta,
            b_x_ta,
            a_x_ta,
            b_y_ta,
        ),
        maximum_iterations=steps,
    )

    d_log_alpha = tf.zeros_like(log_alpha)
    d_log_beta = tf.zeros_like(log_beta)
    d_x = tf.zeros_like(x)
    d_running = tf.zeros_like(running)
    d_a_y = tf.zeros_like(log_alpha)
    d_b_x = tf.zeros_like(log_beta)
    d_a_x = tf.zeros_like(log_alpha)
    d_b_y = tf.zeros_like(log_beta)
    eps_ = tf.reshape(eps, [-1, 1])

    dx_part, dv, d_soft_eps = _filterflow_streaming_same_points_softmin_vjp(
        eps,
        x,
        log_alpha + b_x / eps_,
        upstream_alpha,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        return_epsilon=True,
    )
    d_x += dx_part
    d_log_alpha += dv
    d_b_x += dv / eps_
    dx_part, dv, d_soft_eps_beta = _filterflow_streaming_same_points_softmin_vjp(
        eps,
        x,
        log_beta + a_y / eps_,
        upstream_beta,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        return_epsilon=True,
    )
    d_x += dx_part
    d_log_beta += dv
    d_a_y += dv / eps_
    # `epsilon` is constant in this primitive target; these cotangents are
    # computed to keep the total local reverse formula explicit.
    del d_soft_eps, d_soft_eps_beta

    def reverse_cond(
        iteration: tf.Tensor,
        _d_log_alpha: tf.Tensor,
        _d_log_beta: tf.Tensor,
        _d_x: tf.Tensor,
        _d_running: tf.Tensor,
        _d_a_y: tf.Tensor,
        _d_b_x: tf.Tensor,
        _d_a_x: tf.Tensor,
        _d_b_y: tf.Tensor,
    ) -> tf.Tensor:
        del _d_log_alpha, _d_log_beta, _d_x, _d_running
        del _d_a_y, _d_b_x, _d_a_x, _d_b_y
        return iteration > 0

    def reverse_body(
        iteration: tf.Tensor,
        d_log_alpha_value: tf.Tensor,
        d_log_beta_value: tf.Tensor,
        d_x_value: tf.Tensor,
        d_running_value: tf.Tensor,
        d_a_y_value: tf.Tensor,
        d_b_x_value: tf.Tensor,
        d_a_x_value: tf.Tensor,
        d_b_y_value: tf.Tensor,
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
    ]:
        state_index = iteration - 1
        old_running = running_ta.read(state_index)
        old_a_y = a_y_ta.read(state_index)
        old_b_x = b_x_ta.read(state_index)
        old_a_x = a_x_ta.read(state_index)
        old_b_y = b_y_ta.read(state_index)
        running_ = tf.reshape(old_running, [-1, 1])
        old_d_a_y = 0.5 * d_a_y_value
        old_d_b_x = 0.5 * d_b_x_value
        old_d_a_x = 0.5 * d_a_x_value
        old_d_b_y = 0.5 * d_b_y_value
        active_running = tf.cast(old_running * scaling_factor >= eps, DTYPE)
        old_d_running = d_running_value * scaling_factor * active_running

        dx_part, dv, d_soft_running = _filterflow_streaming_same_points_softmin_vjp(
            old_running,
            x,
            log_alpha + old_b_x / running_,
            0.5 * d_a_y_value,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            return_epsilon=True,
        )
        d_x_value += dx_part
        d_log_alpha_value += dv
        old_d_b_x += dv / running_
        old_d_running += d_soft_running
        old_d_running += -tf.reduce_sum(dv * old_b_x / (running_ * running_), axis=1)

        dx_part, dv, d_soft_running = _filterflow_streaming_same_points_softmin_vjp(
            old_running,
            x,
            log_beta + old_a_y / running_,
            0.5 * d_b_x_value,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            return_epsilon=True,
        )
        d_x_value += dx_part
        d_log_beta_value += dv
        old_d_a_y += dv / running_
        old_d_running += d_soft_running
        old_d_running += -tf.reduce_sum(dv * old_a_y / (running_ * running_), axis=1)

        dx_part, dv, d_soft_running = _filterflow_streaming_same_points_softmin_vjp(
            old_running,
            x,
            log_alpha + old_a_x / running_,
            0.5 * d_a_x_value,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            return_epsilon=True,
        )
        d_x_value += dx_part
        d_log_alpha_value += dv
        old_d_a_x += dv / running_
        old_d_running += d_soft_running
        old_d_running += -tf.reduce_sum(dv * old_a_x / (running_ * running_), axis=1)

        dx_part, dv, d_soft_running = _filterflow_streaming_same_points_softmin_vjp(
            old_running,
            x,
            log_beta + old_b_y / running_,
            0.5 * d_b_y_value,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            return_epsilon=True,
        )
        d_x_value += dx_part
        d_log_beta_value += dv
        old_d_b_y += dv / running_
        old_d_running += d_soft_running
        old_d_running += -tf.reduce_sum(dv * old_b_y / (running_ * running_), axis=1)

        return (
            state_index,
            d_log_alpha_value,
            d_log_beta_value,
            d_x_value,
            old_d_running,
            old_d_a_y,
            old_d_b_x,
            old_d_a_x,
            old_d_b_y,
        )

    (
        _,
        d_log_alpha,
        d_log_beta,
        d_x,
        d_running,
        d_a_y,
        d_b_x,
        d_a_x,
        d_b_y,
    ) = tf.while_loop(
        reverse_cond,
        reverse_body,
        loop_vars=(
            tf.constant(steps, tf.int32),
            d_log_alpha,
            d_log_beta,
            d_x,
            d_running,
            d_a_y,
            d_b_x,
            d_a_x,
            d_b_y,
        ),
        maximum_iterations=steps,
    )

    dx_part, dv, d_soft_running = _filterflow_streaming_same_points_softmin_vjp(
        initial_running,
        x,
        log_beta,
        d_b_y,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        return_epsilon=True,
    )
    d_x += dx_part
    d_log_beta += dv
    d_running += d_soft_running
    dx_part, dv, d_soft_running = _filterflow_streaming_same_points_softmin_vjp(
        initial_running,
        x,
        log_alpha,
        d_a_x,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        return_epsilon=True,
    )
    d_x += dx_part
    d_log_alpha += dv
    d_running += d_soft_running
    dx_part, dv, d_soft_running = _filterflow_streaming_same_points_softmin_vjp(
        initial_running,
        x,
        log_beta,
        d_b_x,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        return_epsilon=True,
    )
    d_x += dx_part
    d_log_beta += dv
    d_running += d_soft_running
    dx_part, dv, d_soft_running = _filterflow_streaming_same_points_softmin_vjp(
        initial_running,
        x,
        log_alpha,
        d_a_y,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
        return_epsilon=True,
    )
    d_x += dx_part
    d_log_alpha += dv
    d_running += d_soft_running
    d_epsilon0 = _match_epsilon_cotangent_shape(raw_epsilon0, d_running)
    return d_log_alpha, d_log_beta, d_x, d_epsilon0


def _filterflow_manual_streaming_finite_transport_value_stopped_scale_keys(
    scaled_x: tf.Tensor,
    particles: tf.Tensor,
    logw: tf.Tensor,
    eps: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    steps = _validate_manual_dense_finite_route_inputs(eps, steps)
    float_n = tf.cast(tf.shape(scaled_x)[1], scaled_x.dtype)
    log_n = tf.math.log(float_n)
    uniform_log_weight = -log_n * tf.ones_like(logw)
    alpha, beta = _filterflow_streaming_finite_sinkhorn_potentials_stopped_scale_keys(
        logw,
        uniform_log_weight,
        scaled_x,
        eps,
        epsilon0,
        scaling,
        steps=steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    return _filterflow_streaming_transport_from_potentials(
        scaled_x,
        particles,
        alpha,
        beta,
        eps,
        logw,
        float_n,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )


def _filterflow_manual_streaming_finite_transport_value_total_vjp(
    scaled_x: tf.Tensor,
    particles: tf.Tensor,
    logw: tf.Tensor,
    eps: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Finite streaming transport value for the total-derivative route."""

    steps = _validate_manual_dense_finite_route_inputs(eps, steps)
    float_n = tf.cast(tf.shape(scaled_x)[1], scaled_x.dtype)
    log_n = tf.math.log(float_n)
    uniform_log_weight = -log_n * tf.ones_like(logw)
    alpha, beta = _filterflow_streaming_finite_sinkhorn_potentials_total_vjp(
        logw,
        uniform_log_weight,
        scaled_x,
        eps,
        epsilon0,
        scaling,
        steps=steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    return _filterflow_streaming_transport_from_potentials(
        scaled_x,
        particles,
        alpha,
        beta,
        eps,
        logw,
        float_n,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )


def _filterflow_streaming_finite_sinkhorn_potentials_jvp_total(
    log_alpha: tf.Tensor,
    log_beta: tf.Tensor,
    x: tf.Tensor,
    d_log_alpha: tf.Tensor,
    d_log_beta: tf.Tensor,
    d_x: tf.Tensor,
    d_epsilon0: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Finite streaming Sinkhorn potentials and no-tape forward tangents."""

    steps = _validate_manual_dense_finite_route_inputs(epsilon, steps)
    log_alpha = tf.cast(log_alpha, DTYPE)
    log_beta = tf.cast(log_beta, DTYPE)
    x = tf.cast(x, DTYPE)
    d_log_alpha = tf.cast(d_log_alpha, DTYPE)
    d_log_beta = tf.cast(d_log_beta, DTYPE)
    d_x = tf.cast(d_x, DTYPE)
    d_epsilon0 = tf.cast(d_epsilon0, DTYPE)
    running = tf.cast(epsilon0, DTYPE)
    d_running = d_epsilon0
    eps = tf.cast(epsilon, DTYPE)
    scaling_factor = tf.cast(scaling, DTYPE) ** 2
    zero_eps_tangent = tf.zeros_like(d_running)
    a_y, d_a_y = _filterflow_streaming_softmin_jvp(
        running,
        x,
        x,
        log_alpha,
        d_running,
        d_x,
        d_x,
        d_log_alpha,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    b_x, d_b_x = _filterflow_streaming_softmin_jvp(
        running,
        x,
        x,
        log_beta,
        d_running,
        d_x,
        d_x,
        d_log_beta,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    a_x, d_a_x = _filterflow_streaming_softmin_jvp(
        running,
        x,
        x,
        log_alpha,
        d_running,
        d_x,
        d_x,
        d_log_alpha,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    b_y, d_b_y = _filterflow_streaming_softmin_jvp(
        running,
        x,
        x,
        log_beta,
        d_running,
        d_x,
        d_x,
        d_log_beta,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    steps_tensor = tf.constant(steps, tf.int32)

    def cond(
        iteration: tf.Tensor,
        _running: tf.Tensor,
        _d_running: tf.Tensor,
        _a_y: tf.Tensor,
        _d_a_y: tf.Tensor,
        _b_x: tf.Tensor,
        _d_b_x: tf.Tensor,
        _a_x: tf.Tensor,
        _d_a_x: tf.Tensor,
        _b_y: tf.Tensor,
        _d_b_y: tf.Tensor,
    ) -> tf.Tensor:
        del _running, _d_running, _a_y, _d_a_y, _b_x, _d_b_x
        del _a_x, _d_a_x, _b_y, _d_b_y
        return iteration < steps_tensor

    def body(
        iteration: tf.Tensor,
        running_value: tf.Tensor,
        d_running_value: tf.Tensor,
        a_y_value: tf.Tensor,
        d_a_y_value: tf.Tensor,
        b_x_value: tf.Tensor,
        d_b_x_value: tf.Tensor,
        a_x_value: tf.Tensor,
        d_a_x_value: tf.Tensor,
        b_y_value: tf.Tensor,
        d_b_y_value: tf.Tensor,
    ):
        running_ = tf.reshape(running_value, [-1, 1])
        d_running_ = d_running_value[:, None, :]
        at_y_values = log_alpha + b_x_value / running_
        d_at_y_values = (
            d_log_alpha
            + d_b_x_value / running_[:, :, None]
            - b_x_value[:, :, None]
            * d_running_
            / (running_[:, :, None] * running_[:, :, None])
        )
        bt_x_values = log_beta + a_y_value / running_
        d_bt_x_values = (
            d_log_beta
            + d_a_y_value / running_[:, :, None]
            - a_y_value[:, :, None]
            * d_running_
            / (running_[:, :, None] * running_[:, :, None])
        )
        at_x_values = log_alpha + a_x_value / running_
        d_at_x_values = (
            d_log_alpha
            + d_a_x_value / running_[:, :, None]
            - a_x_value[:, :, None]
            * d_running_
            / (running_[:, :, None] * running_[:, :, None])
        )
        bt_y_values = log_beta + b_y_value / running_
        d_bt_y_values = (
            d_log_beta
            + d_b_y_value / running_[:, :, None]
            - b_y_value[:, :, None]
            * d_running_
            / (running_[:, :, None] * running_[:, :, None])
        )
        at_y, d_at_y = _filterflow_streaming_softmin_jvp(
            running_value,
            x,
            x,
            at_y_values,
            d_running_value,
            d_x,
            d_x,
            d_at_y_values,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        bt_x, d_bt_x = _filterflow_streaming_softmin_jvp(
            running_value,
            x,
            x,
            bt_x_values,
            d_running_value,
            d_x,
            d_x,
            d_bt_x_values,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        at_x, d_at_x = _filterflow_streaming_softmin_jvp(
            running_value,
            x,
            x,
            at_x_values,
            d_running_value,
            d_x,
            d_x,
            d_at_x_values,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        bt_y, d_bt_y = _filterflow_streaming_softmin_jvp(
            running_value,
            x,
            x,
            bt_y_values,
            d_running_value,
            d_x,
            d_x,
            d_bt_y_values,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        next_running = tf.maximum(running_value * scaling_factor, eps)
        active = tf.cast(running_value * scaling_factor >= eps, DTYPE)
        next_d_running = d_running_value * scaling_factor * active[:, None]
        return (
            iteration + 1,
            next_running,
            next_d_running,
            0.5 * (a_y_value + at_y),
            0.5 * (d_a_y_value + d_at_y),
            0.5 * (b_x_value + bt_x),
            0.5 * (d_b_x_value + d_bt_x),
            0.5 * (a_x_value + at_x),
            0.5 * (d_a_x_value + d_at_x),
            0.5 * (b_y_value + bt_y),
            0.5 * (d_b_y_value + d_bt_y),
        )

    (
        _,
        running,
        d_running,
        a_y,
        d_a_y,
        b_x,
        d_b_x,
        _a_x,
        _d_a_x,
        _b_y,
        _d_b_y,
    ) = tf.while_loop(
        cond,
        body,
        loop_vars=(
            tf.constant(0, tf.int32),
            running,
            d_running,
            a_y,
            d_a_y,
            b_x,
            d_b_x,
            a_x,
            d_a_x,
            b_y,
            d_b_y,
        ),
        maximum_iterations=steps,
    )
    eps_ = tf.reshape(eps, [-1, 1])
    a_values = log_alpha + b_x / eps_
    d_a_values = d_log_alpha + d_b_x / eps_[:, :, None]
    b_values = log_beta + a_y / eps_
    d_b_values = d_log_beta + d_a_y / eps_[:, :, None]
    final_a, d_final_a = _filterflow_streaming_softmin_jvp(
        eps,
        x,
        x,
        a_values,
        zero_eps_tangent,
        d_x,
        d_x,
        d_a_values,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    final_b, d_final_b = _filterflow_streaming_softmin_jvp(
        eps,
        x,
        x,
        b_values,
        zero_eps_tangent,
        d_x,
        d_x,
        d_b_values,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    return final_a, final_b, d_final_a, d_final_b, running, d_running


def _filterflow_manual_streaming_finite_transport_value_and_jvp_total(
    scaled_x: tf.Tensor,
    particles: tf.Tensor,
    logw: tf.Tensor,
    d_scaled_x: tf.Tensor,
    d_particles: tf.Tensor,
    d_logw: tf.Tensor,
    d_epsilon0: tf.Tensor,
    eps: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Finite streaming transport value and no-tape total JVP."""

    steps = _validate_manual_dense_finite_route_inputs(eps, steps)
    scaled_x = tf.cast(scaled_x, DTYPE)
    particles = tf.cast(particles, DTYPE)
    logw = tf.cast(logw, DTYPE)
    d_scaled_x = tf.cast(d_scaled_x, DTYPE)
    d_particles = tf.cast(d_particles, DTYPE)
    d_logw = tf.cast(d_logw, DTYPE)
    d_epsilon0 = tf.cast(d_epsilon0, DTYPE)
    float_n = tf.cast(tf.shape(scaled_x)[1], scaled_x.dtype)
    log_n = tf.math.log(float_n)
    uniform_log_weight = -log_n * tf.ones_like(logw)
    d_uniform_log_weight = tf.zeros_like(d_logw)
    alpha, beta, d_alpha, d_beta, _running, _d_running = (
        _filterflow_streaming_finite_sinkhorn_potentials_jvp_total(
            logw,
            uniform_log_weight,
            scaled_x,
            d_logw,
            d_uniform_log_weight,
            d_scaled_x,
            d_epsilon0,
            eps,
            epsilon0,
            scaling,
            steps=steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
    )
    return _filterflow_streaming_transport_from_potentials_jvp(
        scaled_x,
        particles,
        alpha,
        beta,
        eps,
        logw,
        float_n,
        d_scaled_x,
        d_particles,
        d_alpha,
        d_beta,
        d_logw,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )


def _filterflow_manual_streaming_finite_transport_total_pullback(
    scaled_x: tf.Tensor,
    particles: tf.Tensor,
    logw: tf.Tensor,
    eps: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    upstream: tf.Tensor,
    *,
    steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """No-tape VJP for the finite streaming transport total-value helper."""

    steps = _validate_manual_dense_finite_route_inputs(eps, steps)
    float_n = tf.cast(tf.shape(scaled_x)[1], scaled_x.dtype)
    log_n = tf.math.log(float_n)
    uniform_log_weight = -log_n * tf.ones_like(logw)
    alpha, beta = _filterflow_streaming_finite_sinkhorn_potentials_total_vjp(
        logw,
        uniform_log_weight,
        scaled_x,
        eps,
        epsilon0,
        scaling,
        steps=steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    (
        d_scaled_x_transport,
        d_particles,
        d_alpha,
        d_beta,
        d_logw_transport,
    ) = _filterflow_streaming_transport_from_potentials_vjp(
        scaled_x,
        particles,
        alpha,
        beta,
        eps,
        logw,
        float_n,
        upstream,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    (
        d_log_alpha,
        _d_log_beta,
        d_scaled_x_sinkhorn,
        d_epsilon0,
    ) = _filterflow_streaming_finite_sinkhorn_potentials_vjp_total(
        logw,
        uniform_log_weight,
        scaled_x,
        d_alpha,
        d_beta,
        eps,
        epsilon0,
        scaling,
        steps=steps,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    d_scaled_x = d_scaled_x_transport + d_scaled_x_sinkhorn
    d_logw = d_logw_transport + d_log_alpha
    return d_scaled_x, d_particles, d_logw, d_epsilon0


def _filterflow_manual_streaming_finite_transport_total_vjp(
    scaled_x: tf.Tensor,
    particles: tf.Tensor,
    logw: tf.Tensor,
    eps: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Streaming finite transport with a total-derivative custom VJP.

    This route differentiates the finite fixed-iteration transport value that it
    executes.  It intentionally does not use the historical stopped-scale/key
    derivative.
    """

    steps = _validate_manual_dense_finite_route_inputs(eps, steps)
    row_chunk_size = _validate_chunk_size(row_chunk_size, "row_chunk_size")
    col_chunk_size = _validate_chunk_size(col_chunk_size, "col_chunk_size")

    @tf.custom_gradient
    def _transport(
        inner_scaled_x: tf.Tensor,
        inner_particles: tf.Tensor,
        inner_logw: tf.Tensor,
        inner_eps: tf.Tensor,
        inner_epsilon0: tf.Tensor,
        inner_scaling: tf.Tensor,
    ) -> tuple[tuple[tf.Tensor, tf.Tensor], Any]:
        transported, row_residual = (
            _filterflow_manual_streaming_finite_transport_value_total_vjp(
                inner_scaled_x,
                inner_particles,
                inner_logw,
                inner_eps,
                inner_epsilon0,
                inner_scaling,
                steps=steps,
                row_chunk_size=row_chunk_size,
                col_chunk_size=col_chunk_size,
            )
        )

        def grad(
            d_transported: tf.Tensor,
            d_row_residual: tf.Tensor | None = None,
        ) -> tuple[tf.Tensor | None, ...]:
            del d_row_residual
            (
                d_scaled_x,
                d_particles,
                d_logw,
                d_epsilon0,
            ) = _filterflow_manual_streaming_finite_transport_total_pullback(
                inner_scaled_x,
                inner_particles,
                inner_logw,
                inner_eps,
                inner_epsilon0,
                inner_scaling,
                d_transported,
                row_chunk_size=row_chunk_size,
                col_chunk_size=col_chunk_size,
                steps=steps,
            )
            return d_scaled_x, d_particles, d_logw, None, d_epsilon0, None

        return (transported, row_residual), grad

    return _transport(scaled_x, particles, logw, eps, epsilon0, scaling)


def _filterflow_manual_streaming_finite_transport_stopped_scale_keys(
    scaled_x: tf.Tensor,
    particles: tf.Tensor,
    logw: tf.Tensor,
    eps: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Private streaming/checkpointed route for the stopped-scale/key scalar."""

    steps = _validate_manual_dense_finite_route_inputs(eps, steps)
    row_chunk_size = _validate_chunk_size(row_chunk_size, "row_chunk_size")
    col_chunk_size = _validate_chunk_size(col_chunk_size, "col_chunk_size")

    @tf.custom_gradient
    def _transport(
        inner_scaled_x: tf.Tensor,
        inner_particles: tf.Tensor,
        inner_logw: tf.Tensor,
        inner_eps: tf.Tensor,
        inner_epsilon0: tf.Tensor,
        inner_scaling: tf.Tensor,
    ) -> tuple[tuple[tf.Tensor, tf.Tensor], Any]:
        transported, row_residual = (
            _filterflow_manual_streaming_finite_transport_value_stopped_scale_keys(
                inner_scaled_x,
                inner_particles,
                inner_logw,
                inner_eps,
                inner_epsilon0,
                inner_scaling,
                steps=steps,
                row_chunk_size=row_chunk_size,
                col_chunk_size=col_chunk_size,
            )
        )

        def grad(
            d_transported: tf.Tensor,
            d_row_residual: tf.Tensor | None = None,
        ) -> tuple[tf.Tensor | None, ...]:
            del d_row_residual
            float_n = tf.cast(tf.shape(inner_scaled_x)[1], inner_scaled_x.dtype)
            log_n = tf.math.log(float_n)
            uniform_log_weight = -log_n * tf.ones_like(inner_logw)
            alpha, beta = _filterflow_streaming_finite_sinkhorn_potentials_stopped_scale_keys(
                inner_logw,
                uniform_log_weight,
                inner_scaled_x,
                inner_eps,
                inner_epsilon0,
                inner_scaling,
                steps=steps,
                row_chunk_size=row_chunk_size,
                col_chunk_size=col_chunk_size,
            )
            (
                d_scaled_x_transport,
                d_particles,
                d_alpha,
                d_beta,
                d_logw_transport,
            ) = _filterflow_streaming_transport_from_potentials_vjp(
                inner_scaled_x,
                inner_particles,
                alpha,
                beta,
                inner_eps,
                inner_logw,
                float_n,
                d_transported,
                row_chunk_size=row_chunk_size,
                col_chunk_size=col_chunk_size,
            )
            (
                d_log_alpha,
                _d_log_beta,
                d_scaled_x_sinkhorn,
            ) = _filterflow_streaming_finite_sinkhorn_potentials_vjp_stopped_scale_keys(
                inner_logw,
                uniform_log_weight,
                inner_scaled_x,
                d_alpha,
                d_beta,
                inner_eps,
                inner_epsilon0,
                inner_scaling,
                steps=steps,
                row_chunk_size=row_chunk_size,
                col_chunk_size=col_chunk_size,
            )
            d_scaled_x = d_scaled_x_transport + d_scaled_x_sinkhorn
            d_logw = d_logw_transport + d_log_alpha
            return d_scaled_x, d_particles, d_logw, None, None, None

        return (transported, row_residual), grad

    return _transport(scaled_x, particles, logw, eps, epsilon0, scaling)


def _filterflow_manual_streaming_blockwise_vjp_finite_transport_stopped_scale_keys(
    scaled_x: tf.Tensor,
    particles: tf.Tensor,
    logw: tf.Tensor,
    eps: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Private streaming route with hand-coded blockwise backward pass."""

    steps = _validate_manual_dense_finite_route_inputs(eps, steps)
    row_chunk_size = _validate_chunk_size(row_chunk_size, "row_chunk_size")
    col_chunk_size = _validate_chunk_size(col_chunk_size, "col_chunk_size")

    @tf.custom_gradient
    def _transport(
        inner_scaled_x: tf.Tensor,
        inner_particles: tf.Tensor,
        inner_logw: tf.Tensor,
        inner_eps: tf.Tensor,
        inner_epsilon0: tf.Tensor,
        inner_scaling: tf.Tensor,
    ) -> tuple[tuple[tf.Tensor, tf.Tensor], Any]:
        transported, row_residual = (
            _filterflow_manual_streaming_finite_transport_value_stopped_scale_keys(
                inner_scaled_x,
                inner_particles,
                inner_logw,
                inner_eps,
                inner_epsilon0,
                inner_scaling,
                steps=steps,
                row_chunk_size=row_chunk_size,
                col_chunk_size=col_chunk_size,
            )
        )

        def grad(
            d_transported: tf.Tensor,
            d_row_residual: tf.Tensor | None = None,
        ) -> tuple[tf.Tensor | None, ...]:
            del d_row_residual
            float_n = tf.cast(tf.shape(inner_scaled_x)[1], inner_scaled_x.dtype)
            log_n = tf.math.log(float_n)
            uniform_log_weight = -log_n * tf.ones_like(inner_logw)
            alpha, beta = _filterflow_streaming_finite_sinkhorn_potentials_stopped_scale_keys(
                inner_logw,
                uniform_log_weight,
                inner_scaled_x,
                inner_eps,
                inner_epsilon0,
                inner_scaling,
                steps=steps,
                row_chunk_size=row_chunk_size,
                col_chunk_size=col_chunk_size,
            )
            (
                d_scaled_x_transport,
                d_particles,
                d_alpha,
                d_beta,
                d_logw_transport,
            ) = _filterflow_streaming_transport_from_potentials_vjp(
                inner_scaled_x,
                inner_particles,
                alpha,
                beta,
                inner_eps,
                inner_logw,
                float_n,
                d_transported,
                row_chunk_size=row_chunk_size,
                col_chunk_size=col_chunk_size,
            )
            (
                d_log_alpha,
                _d_log_beta,
                d_scaled_x_sinkhorn,
            ) = _filterflow_streaming_finite_sinkhorn_potentials_vjp_stopped_scale_keys(
                inner_logw,
                uniform_log_weight,
                inner_scaled_x,
                d_alpha,
                d_beta,
                inner_eps,
                inner_epsilon0,
                inner_scaling,
                steps=steps,
                row_chunk_size=row_chunk_size,
                col_chunk_size=col_chunk_size,
            )
            d_scaled_x = d_scaled_x_transport + d_scaled_x_sinkhorn
            d_logw = d_logw_transport + d_log_alpha
            return d_scaled_x, d_particles, d_logw, None, None, None

        return (transported, row_residual), grad

    return _transport(scaled_x, particles, logw, eps, epsilon0, scaling)


def _transport_backward_rule(mode: str) -> str:
    if mode == "filterflow_clipped":
        return "clip_upstream_d_transport_to_[-1,1]_at_transport_matrix_output"
    if mode == "filterflow_custom_op":
        return "filterflow_style_whole_transport_custom_gradient"
    return "raw_tensorflow_transport_gradient"


@tf.custom_gradient
def _clip_transport_upstream_gradient(
    transport_matrix: tf.Tensor,
) -> tuple[tf.Tensor, Any]:
    """Mirror FilterFlow RegularisedTransform transport backward clipping."""

    def grad(d_transport: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(d_transport, -1.0, 1.0)

    return transport_matrix, grad


@tf.custom_gradient
def _filterflow_custom_gradient_transport_matrix(
    x: tf.Tensor,
    logw: tf.Tensor,
    eps: tf.Tensor,
    scaling: tf.Tensor,
    threshold: tf.Tensor,
    max_iter: tf.Tensor,
    n: tf.Tensor,
) -> tuple[tuple[tf.Tensor, tf.Tensor], Any]:
    transport_matrix, iterations = _filterflow_exact_transport_matrix(
        x,
        logw,
        eps,
        scaling,
        threshold,
        max_iter,
        n,
    )

    def grad(
        d_transport: tf.Tensor,
        d_iterations: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor | None, ...]:
        del d_iterations
        clipped = tf.clip_by_value(d_transport, -1.0, 1.0)
        with tf.GradientTape() as tape:
            tape.watch([x, logw])
            raw_transport_matrix, _raw_iterations = _filterflow_exact_transport_matrix(
                x,
                logw,
                eps,
                scaling,
                threshold,
                max_iter,
                n,
            )
        dx, dlogw = tape.gradient(
            raw_transport_matrix,
            [x, logw],
            output_gradients=clipped,
        )
        return dx, dlogw, None, None, None, None, None

    return (transport_matrix, iterations), grad


def _validate_manual_dense_finite_route_inputs(
    eps: tf.Tensor,
    steps: int,
) -> int:
    if eps.shape.rank is not None and eps.shape.rank != 0:
        raise ValueError(
            "manual dense finite transport currently supports scalar epsilon only"
        )
    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    return steps


def _filterflow_manual_dense_finite_softmin_vjp(
    epsilon: tf.Tensor,
    cost_matrix: tf.Tensor,
    values: tf.Tensor,
    upstream: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    dtype = cost_matrix.dtype
    epsilon = tf.reshape(tf.cast(epsilon, dtype), [-1, 1, 1])
    logits = values[:, None, :] - cost_matrix / epsilon
    probs = tf.nn.softmax(logits, axis=2)
    d_values = -tf.reshape(epsilon, [-1, 1]) * tf.reduce_sum(
        upstream[:, :, None] * probs,
        axis=1,
    )
    d_cost = upstream[:, :, None] * probs
    return d_cost, d_values


def _filterflow_manual_dense_finite_sinkhorn_outputs(
    log_alpha: tf.Tensor,
    log_beta: tf.Tensor,
    cost_xy: tf.Tensor,
    cost_yx: tf.Tensor,
    cost_xx: tf.Tensor,
    cost_yy: tf.Tensor,
    *,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    steps: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    dtype = log_alpha.dtype
    running = tf.cast(epsilon0, dtype)
    eps = tf.cast(epsilon, dtype)
    scaling_factor = tf.cast(scaling, dtype) ** 2
    a_y = _filterflow_exact_softmin(running, cost_yx, log_alpha)
    b_x = _filterflow_exact_softmin(running, cost_xy, log_beta)
    a_x = _filterflow_exact_softmin(running, cost_xx, log_alpha)
    b_y = _filterflow_exact_softmin(running, cost_yy, log_beta)

    def cond(
        iteration: tf.Tensor,
        _running: tf.Tensor,
        _a_y: tf.Tensor,
        _b_x: tf.Tensor,
        _a_x: tf.Tensor,
        _b_y: tf.Tensor,
    ) -> tf.Tensor:
        del _running, _a_y, _b_x, _a_x, _b_y
        return iteration < steps

    def body(
        iteration: tf.Tensor,
        running_value: tf.Tensor,
        a_y_value: tf.Tensor,
        b_x_value: tf.Tensor,
        a_x_value: tf.Tensor,
        b_y_value: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        running_ = tf.reshape(running_value, [-1, 1])
        at_y = _filterflow_exact_softmin(
            running_value,
            cost_yx,
            log_alpha + b_x_value / running_,
        )
        bt_x = _filterflow_exact_softmin(
            running_value,
            cost_xy,
            log_beta + a_y_value / running_,
        )
        at_x = _filterflow_exact_softmin(
            running_value,
            cost_xx,
            log_alpha + a_x_value / running_,
        )
        bt_y = _filterflow_exact_softmin(
            running_value,
            cost_yy,
            log_beta + b_y_value / running_,
        )
        return (
            iteration + 1,
            tf.maximum(running_value * scaling_factor, eps),
            0.5 * (a_y_value + at_y),
            0.5 * (b_x_value + bt_x),
            0.5 * (a_x_value + at_x),
            0.5 * (b_y_value + bt_y),
        )

    _, running, a_y, b_x, a_x, b_y = tf.while_loop(
        cond,
        body,
        loop_vars=(tf.constant(0, tf.int32), running, a_y, b_x, a_x, b_y),
        maximum_iterations=steps,
    )
    eps_ = tf.reshape(eps, [-1, 1])
    return (
        _filterflow_exact_softmin(eps, cost_yx, log_alpha + b_x / eps_),
        _filterflow_exact_softmin(eps, cost_xy, log_beta + a_y / eps_),
        _filterflow_exact_softmin(eps, cost_xx, log_alpha + a_x / eps_),
        _filterflow_exact_softmin(eps, cost_yy, log_beta + b_y / eps_),
    )


def _filterflow_manual_dense_finite_sinkhorn_vjp(
    log_alpha: tf.Tensor,
    log_beta: tf.Tensor,
    cost_xy: tf.Tensor,
    cost_yx: tf.Tensor,
    cost_xx: tf.Tensor,
    cost_yy: tf.Tensor,
    upstreams: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    *,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    steps: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    dtype = log_alpha.dtype
    running = tf.cast(epsilon0, dtype)
    eps = tf.cast(epsilon, dtype)
    scaling_factor = tf.cast(scaling, dtype) ** 2
    a_y = _filterflow_exact_softmin(running, cost_yx, log_alpha)
    b_x = _filterflow_exact_softmin(running, cost_xy, log_beta)
    a_x = _filterflow_exact_softmin(running, cost_xx, log_alpha)
    b_y = _filterflow_exact_softmin(running, cost_yy, log_beta)
    running_ta = tf.TensorArray(
        dtype=dtype,
        size=steps,
        element_shape=running.shape,
        clear_after_read=False,
    )
    a_y_ta = tf.TensorArray(
        dtype=dtype,
        size=steps,
        element_shape=a_y.shape,
        clear_after_read=False,
    )
    b_x_ta = tf.TensorArray(
        dtype=dtype,
        size=steps,
        element_shape=b_x.shape,
        clear_after_read=False,
    )
    a_x_ta = tf.TensorArray(
        dtype=dtype,
        size=steps,
        element_shape=a_x.shape,
        clear_after_read=False,
    )
    b_y_ta = tf.TensorArray(
        dtype=dtype,
        size=steps,
        element_shape=b_y.shape,
        clear_after_read=False,
    )

    def forward_cond(
        iteration: tf.Tensor,
        _running: tf.Tensor,
        _a_y: tf.Tensor,
        _b_x: tf.Tensor,
        _a_x: tf.Tensor,
        _b_y: tf.Tensor,
        _running_ta: tf.TensorArray,
        _a_y_ta: tf.TensorArray,
        _b_x_ta: tf.TensorArray,
        _a_x_ta: tf.TensorArray,
        _b_y_ta: tf.TensorArray,
    ) -> tf.Tensor:
        del _running, _a_y, _b_x, _a_x, _b_y
        del _running_ta, _a_y_ta, _b_x_ta, _a_x_ta, _b_y_ta
        return iteration < steps

    def forward_body(
        iteration: tf.Tensor,
        running_value: tf.Tensor,
        a_y_value: tf.Tensor,
        b_x_value: tf.Tensor,
        a_x_value: tf.Tensor,
        b_y_value: tf.Tensor,
        running_records: tf.TensorArray,
        a_y_records: tf.TensorArray,
        b_x_records: tf.TensorArray,
        a_x_records: tf.TensorArray,
        b_y_records: tf.TensorArray,
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.TensorArray,
        tf.TensorArray,
        tf.TensorArray,
        tf.TensorArray,
        tf.TensorArray,
    ]:
        running_records = running_records.write(iteration, running_value)
        a_y_records = a_y_records.write(iteration, a_y_value)
        b_x_records = b_x_records.write(iteration, b_x_value)
        a_x_records = a_x_records.write(iteration, a_x_value)
        b_y_records = b_y_records.write(iteration, b_y_value)
        running_ = tf.reshape(running_value, [-1, 1])
        at_y = _filterflow_exact_softmin(
            running_value,
            cost_yx,
            log_alpha + b_x_value / running_,
        )
        bt_x = _filterflow_exact_softmin(
            running_value,
            cost_xy,
            log_beta + a_y_value / running_,
        )
        at_x = _filterflow_exact_softmin(
            running_value,
            cost_xx,
            log_alpha + a_x_value / running_,
        )
        bt_y = _filterflow_exact_softmin(
            running_value,
            cost_yy,
            log_beta + b_y_value / running_,
        )
        return (
            iteration + 1,
            tf.maximum(running_value * scaling_factor, eps),
            0.5 * (a_y_value + at_y),
            0.5 * (b_x_value + bt_x),
            0.5 * (a_x_value + at_x),
            0.5 * (b_y_value + bt_y),
            running_records,
            a_y_records,
            b_x_records,
            a_x_records,
            b_y_records,
        )

    (
        _,
        running,
        a_y,
        b_x,
        a_x,
        b_y,
        running_ta,
        a_y_ta,
        b_x_ta,
        a_x_ta,
        b_y_ta,
    ) = tf.while_loop(
        forward_cond,
        forward_body,
        loop_vars=(
            tf.constant(0, tf.int32),
            running,
            a_y,
            b_x,
            a_x,
            b_y,
            running_ta,
            a_y_ta,
            b_x_ta,
            a_x_ta,
            b_y_ta,
        ),
        maximum_iterations=steps,
    )

    d_log_alpha = tf.zeros_like(log_alpha)
    d_log_beta = tf.zeros_like(log_beta)
    d_cost_xy = tf.zeros_like(cost_xy)
    d_cost_yx = tf.zeros_like(cost_yx)
    d_cost_xx = tf.zeros_like(cost_xx)
    d_cost_yy = tf.zeros_like(cost_yy)
    d_a_y = tf.zeros_like(log_alpha)
    d_b_x = tf.zeros_like(log_beta)
    d_a_x = tf.zeros_like(log_alpha)
    d_b_y = tf.zeros_like(log_beta)
    eps_ = tf.reshape(eps, [-1, 1])

    dc, dv = _filterflow_manual_dense_finite_softmin_vjp(
        eps,
        cost_yx,
        log_alpha + b_x / eps_,
        upstreams[0],
    )
    d_cost_yx += dc
    d_log_alpha += dv
    d_b_x += dv / eps_
    dc, dv = _filterflow_manual_dense_finite_softmin_vjp(
        eps,
        cost_xy,
        log_beta + a_y / eps_,
        upstreams[1],
    )
    d_cost_xy += dc
    d_log_beta += dv
    d_a_y += dv / eps_
    dc, dv = _filterflow_manual_dense_finite_softmin_vjp(
        eps,
        cost_xx,
        log_alpha + a_x / eps_,
        upstreams[2],
    )
    d_cost_xx += dc
    d_log_alpha += dv
    d_a_x += dv / eps_
    dc, dv = _filterflow_manual_dense_finite_softmin_vjp(
        eps,
        cost_yy,
        log_beta + b_y / eps_,
        upstreams[3],
    )
    d_cost_yy += dc
    d_log_beta += dv
    d_b_y += dv / eps_

    def reverse_cond(
        iteration: tf.Tensor,
        _d_log_alpha: tf.Tensor,
        _d_log_beta: tf.Tensor,
        _d_cost_xy: tf.Tensor,
        _d_cost_yx: tf.Tensor,
        _d_cost_xx: tf.Tensor,
        _d_cost_yy: tf.Tensor,
        _d_a_y: tf.Tensor,
        _d_b_x: tf.Tensor,
        _d_a_x: tf.Tensor,
        _d_b_y: tf.Tensor,
    ) -> tf.Tensor:
        del _d_log_alpha, _d_log_beta, _d_cost_xy, _d_cost_yx
        del _d_cost_xx, _d_cost_yy, _d_a_y, _d_b_x, _d_a_x, _d_b_y
        return iteration > 0

    def reverse_body(
        iteration: tf.Tensor,
        d_log_alpha_value: tf.Tensor,
        d_log_beta_value: tf.Tensor,
        d_cost_xy_value: tf.Tensor,
        d_cost_yx_value: tf.Tensor,
        d_cost_xx_value: tf.Tensor,
        d_cost_yy_value: tf.Tensor,
        d_a_y_value: tf.Tensor,
        d_b_x_value: tf.Tensor,
        d_a_x_value: tf.Tensor,
        d_b_y_value: tf.Tensor,
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
    ]:
        state_index = iteration - 1
        old_running = running_ta.read(state_index)
        old_a_y = a_y_ta.read(state_index)
        old_b_x = b_x_ta.read(state_index)
        old_a_x = a_x_ta.read(state_index)
        old_b_y = b_y_ta.read(state_index)
        running_ = tf.reshape(old_running, [-1, 1])
        bar_at_y = 0.5 * d_a_y_value
        bar_bt_x = 0.5 * d_b_x_value
        bar_at_x = 0.5 * d_a_x_value
        bar_bt_y = 0.5 * d_b_y_value

        old_d_a_y = 0.5 * d_a_y_value
        old_d_b_x = 0.5 * d_b_x_value
        old_d_a_x = 0.5 * d_a_x_value
        old_d_b_y = 0.5 * d_b_y_value

        dc, dv = _filterflow_manual_dense_finite_softmin_vjp(
            old_running,
            cost_yx,
            log_alpha + old_b_x / running_,
            bar_at_y,
        )
        d_cost_yx_value += dc
        d_log_alpha_value += dv
        old_d_b_x += dv / running_
        dc, dv = _filterflow_manual_dense_finite_softmin_vjp(
            old_running,
            cost_xy,
            log_beta + old_a_y / running_,
            bar_bt_x,
        )
        d_cost_xy_value += dc
        d_log_beta_value += dv
        old_d_a_y += dv / running_
        dc, dv = _filterflow_manual_dense_finite_softmin_vjp(
            old_running,
            cost_xx,
            log_alpha + old_a_x / running_,
            bar_at_x,
        )
        d_cost_xx_value += dc
        d_log_alpha_value += dv
        old_d_a_x += dv / running_
        dc, dv = _filterflow_manual_dense_finite_softmin_vjp(
            old_running,
            cost_yy,
            log_beta + old_b_y / running_,
            bar_bt_y,
        )
        d_cost_yy_value += dc
        d_log_beta_value += dv
        old_d_b_y += dv / running_

        return (
            state_index,
            d_log_alpha_value,
            d_log_beta_value,
            d_cost_xy_value,
            d_cost_yx_value,
            d_cost_xx_value,
            d_cost_yy_value,
            old_d_a_y,
            old_d_b_x,
            old_d_a_x,
            old_d_b_y,
        )

    (
        _,
        d_log_alpha,
        d_log_beta,
        d_cost_xy,
        d_cost_yx,
        d_cost_xx,
        d_cost_yy,
        d_a_y,
        d_b_x,
        d_a_x,
        d_b_y,
    ) = tf.while_loop(
        reverse_cond,
        reverse_body,
        loop_vars=(
            tf.constant(steps, tf.int32),
            d_log_alpha,
            d_log_beta,
            d_cost_xy,
            d_cost_yx,
            d_cost_xx,
            d_cost_yy,
            d_a_y,
            d_b_x,
            d_a_x,
            d_b_y,
        ),
        maximum_iterations=steps,
    )

    dc, dv = _filterflow_manual_dense_finite_softmin_vjp(
        epsilon0,
        cost_yy,
        log_beta,
        d_b_y,
    )
    d_cost_yy += dc
    d_log_beta += dv
    dc, dv = _filterflow_manual_dense_finite_softmin_vjp(
        epsilon0,
        cost_xx,
        log_alpha,
        d_a_x,
    )
    d_cost_xx += dc
    d_log_alpha += dv
    dc, dv = _filterflow_manual_dense_finite_softmin_vjp(
        epsilon0,
        cost_xy,
        log_beta,
        d_b_x,
    )
    d_cost_xy += dc
    d_log_beta += dv
    dc, dv = _filterflow_manual_dense_finite_softmin_vjp(
        epsilon0,
        cost_yx,
        log_alpha,
        d_a_y,
    )
    d_cost_yx += dc
    d_log_alpha += dv
    return d_log_alpha, d_log_beta, d_cost_xy, d_cost_yx, d_cost_xx, d_cost_yy


def _filterflow_manual_transport_from_potentials_vjp(
    particles: tf.Tensor,
    f: tf.Tensor,
    g: tf.Tensor,
    eps: tf.Tensor,
    logw: tf.Tensor,
    upstream: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    del g
    dtype = particles.dtype
    eps_ = tf.reshape(tf.cast(eps, dtype), [-1, 1, 1])
    n = tf.cast(tf.shape(particles)[1], dtype)
    diff = particles[:, :, None, :] - particles[:, None, :, :]
    cost = 0.5 * tf.reduce_sum(diff * diff, axis=3)
    logits = (f[:, :, None] + tf.zeros_like(cost) - cost) / eps_
    probs = tf.nn.softmax(logits, axis=1)
    transport = tf.exp(logw[:, None, :]) * n * probs
    d_temp = upstream * transport
    d_logits = d_temp - probs * tf.reduce_sum(d_temp, axis=1, keepdims=True)
    d_logw = tf.reduce_sum(d_temp, axis=1)
    d_f = tf.reduce_sum(d_logits / eps_, axis=2)
    d_g = tf.reduce_sum(d_logits / eps_, axis=1)
    d_cost = -d_logits / eps_
    row_part = tf.reduce_sum(d_cost[:, :, :, None] * diff, axis=2)
    col_part = tf.reduce_sum(d_cost[:, :, :, None] * (-diff), axis=1)
    return row_part + col_part, d_f, d_g, d_logw


def _filterflow_manual_same_particles_cost_vjp(
    particles: tf.Tensor,
    d_cost: tf.Tensor,
    *,
    stop_keys: bool,
) -> tf.Tensor:
    diff = particles[:, :, None, :] - particles[:, None, :, :]
    row_part = tf.reduce_sum(d_cost[:, :, :, None] * diff, axis=2)
    if stop_keys:
        return row_part
    col_part = tf.reduce_sum(d_cost[:, :, :, None] * (-diff), axis=1)
    return row_part + col_part


def _filterflow_manual_dense_finite_transport_matrix_value_stopped_scale_keys(
    x: tf.Tensor,
    logw: tf.Tensor,
    eps: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
) -> tf.Tensor:
    steps = _validate_manual_dense_finite_route_inputs(eps, steps)
    float_n = tf.cast(tf.shape(x)[1], x.dtype)
    log_n = tf.math.log(float_n)
    uniform_log_weight = -log_n * tf.ones_like(logw)
    key_x = tf.stop_gradient(x)
    cost = _filterflow_exact_cost(x, key_x)
    alpha, beta, _, _ = _filterflow_manual_dense_finite_sinkhorn_outputs(
        logw,
        uniform_log_weight,
        cost,
        cost,
        cost,
        cost,
        epsilon=eps,
        epsilon0=epsilon0,
        scaling=scaling,
        steps=steps,
    )
    return _filterflow_exact_transport_from_potentials(
        x,
        alpha,
        beta,
        eps,
        logw,
        float_n,
    )


def _filterflow_manual_dense_finite_transport_matrix_vjp_stopped_scale_keys(
    x: tf.Tensor,
    logw: tf.Tensor,
    eps: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    upstream: tf.Tensor,
    *,
    steps: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    steps = _validate_manual_dense_finite_route_inputs(eps, steps)
    float_n = tf.cast(tf.shape(x)[1], x.dtype)
    log_n = tf.math.log(float_n)
    uniform_log_weight = -log_n * tf.ones_like(logw)
    key_x = tf.stop_gradient(x)
    cost = _filterflow_exact_cost(x, key_x)
    alpha, beta, _, _ = _filterflow_manual_dense_finite_sinkhorn_outputs(
        logw,
        uniform_log_weight,
        cost,
        cost,
        cost,
        cost,
        epsilon=eps,
        epsilon0=epsilon0,
        scaling=scaling,
        steps=steps,
    )
    d_x_transport, d_alpha, d_beta, d_logw_transport = (
        _filterflow_manual_transport_from_potentials_vjp(
            x,
            alpha,
            beta,
            eps,
            logw,
            upstream,
        )
    )
    zero = tf.zeros_like(alpha)
    (
        d_log_alpha,
        _d_log_beta,
        d_cost_xy,
        d_cost_yx,
        d_cost_xx,
        d_cost_yy,
    ) = _filterflow_manual_dense_finite_sinkhorn_vjp(
        logw,
        uniform_log_weight,
        cost,
        cost,
        cost,
        cost,
        (d_alpha, d_beta, zero, zero),
        epsilon=eps,
        epsilon0=epsilon0,
        scaling=scaling,
        steps=steps,
    )
    d_x_cost = tf.add_n(
        [
            _filterflow_manual_same_particles_cost_vjp(x, d_cost, stop_keys=True)
            for d_cost in (d_cost_xy, d_cost_yx, d_cost_xx, d_cost_yy)
        ]
    )
    return d_x_transport + d_x_cost, d_logw_transport + d_log_alpha


def _filterflow_manual_dense_finite_transport_matrix_stopped_scale_keys(
    x: tf.Tensor,
    logw: tf.Tensor,
    eps: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
    *,
    steps: int,
) -> tf.Tensor:
    """Private M3 prototype for the dense finite stopped-scale/key route.

    This helper is intentionally not wired into public/default transport modes.
    It supports the tiny dense finite route used by the manual-adjoint M2/M3
    gates; gradients through epsilon, epsilon0, scaling, and iteration count are
    out of scope.
    """

    steps = _validate_manual_dense_finite_route_inputs(eps, steps)

    @tf.custom_gradient
    def _transport(
        inner_x: tf.Tensor,
        inner_logw: tf.Tensor,
        inner_eps: tf.Tensor,
        inner_epsilon0: tf.Tensor,
        inner_scaling: tf.Tensor,
    ) -> tuple[tf.Tensor, Any]:
        transport_matrix = (
            _filterflow_manual_dense_finite_transport_matrix_value_stopped_scale_keys(
                inner_x,
                inner_logw,
                inner_eps,
                inner_epsilon0,
                inner_scaling,
                steps=steps,
            )
        )

        def grad(d_transport: tf.Tensor) -> tuple[tf.Tensor | None, ...]:
            d_x, d_logw = (
                _filterflow_manual_dense_finite_transport_matrix_vjp_stopped_scale_keys(
                    inner_x,
                    inner_logw,
                    inner_eps,
                    inner_epsilon0,
                    inner_scaling,
                    d_transport,
                    steps=steps,
                )
            )
            return d_x, d_logw, None, None, None

        return transport_matrix, grad

    return _transport(x, logw, eps, epsilon0, scaling)


@tf.function
def _filterflow_exact_transport_matrix(
    x: tf.Tensor,
    logw: tf.Tensor,
    eps: tf.Tensor,
    scaling: tf.Tensor,
    threshold: tf.Tensor,
    max_iter: tf.Tensor,
    n: tf.Tensor,
    transport_ad_mode: str = "stabilized",
    warmstart_state: AnnealedTransportWarmstartStateTF | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    _validate_transport_ad_mode(transport_ad_mode)
    float_n = tf.cast(n, x.dtype)
    log_n = tf.math.log(float_n)
    uniform_log_weight = -log_n * tf.ones_like(logw)
    alpha, beta, _, _, iterations = _filterflow_exact_sinkhorn_potentials(
        logw,
        x,
        uniform_log_weight,
        x,
        eps,
        scaling,
        threshold,
        max_iter,
        transport_ad_mode=transport_ad_mode,
        warmstart_state=warmstart_state,
    )
    transport_matrix = _filterflow_exact_transport_from_potentials(
        x,
        alpha,
        beta,
        eps,
        logw,
        float_n,
    )
    return transport_matrix, iterations


@tf.function
def _filterflow_exact_transport_from_potentials(
    x: tf.Tensor,
    f: tf.Tensor,
    g: tf.Tensor,
    eps: tf.Tensor,
    logw: tf.Tensor,
    float_n: tf.Tensor,
) -> tf.Tensor:
    log_n = tf.math.log(float_n)
    cost_matrix = _filterflow_exact_cost(x, x)
    fg = tf.expand_dims(f, 2) + tf.expand_dims(g, 1)
    temp = fg - cost_matrix
    temp = temp / eps
    temp = temp - tf.reduce_logsumexp(temp, 1, keepdims=True) + log_n
    temp = temp + tf.expand_dims(logw, 1)
    return tf.math.exp(temp)


@tf.function
def _filterflow_exact_sinkhorn_potentials(
    log_alpha: tf.Tensor,
    x: tf.Tensor,
    log_beta: tf.Tensor,
    y: tf.Tensor,
    epsilon: tf.Tensor,
    scaling: tf.Tensor,
    threshold: tf.Tensor,
    max_iter: tf.Tensor,
    transport_ad_mode: str = "stabilized",
    warmstart_state: AnnealedTransportWarmstartStateTF | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    _validate_transport_ad_mode(transport_ad_mode)
    key_x = _maybe_stop(x, stop=_transport_ad_stop_keys(transport_ad_mode))
    key_y = _maybe_stop(y, stop=_transport_ad_stop_keys(transport_ad_mode))
    cost_xy = _filterflow_exact_cost(x, key_y)
    cost_yx = _filterflow_exact_cost(y, key_x)
    cost_xx = _filterflow_exact_cost(x, key_x)
    cost_yy = _filterflow_exact_cost(y, key_y)
    scale = _maybe_stop(
        _filterflow_exact_max_min(x, y),
        stop=_transport_ad_stop_scale(transport_ad_mode),
    )
    return _filterflow_exact_sinkhorn_loop(
        log_alpha,
        log_beta,
        cost_xy,
        cost_yx,
        cost_xx,
        cost_yy,
        epsilon,
        scale,
        scaling,
        threshold,
        max_iter,
        transport_ad_mode=transport_ad_mode,
        warmstart_state=warmstart_state,
    )


@tf.function
def _filterflow_exact_sinkhorn_loop(
    log_alpha: tf.Tensor,
    log_beta: tf.Tensor,
    cost_xy: tf.Tensor,
    cost_yx: tf.Tensor,
    cost_xx: tf.Tensor,
    cost_yy: tf.Tensor,
    epsilon: tf.Tensor,
    particles_diameter: tf.Tensor,
    scaling: tf.Tensor,
    threshold: tf.Tensor,
    max_iter: tf.Tensor,
    transport_ad_mode: str = "stabilized",
    warmstart_state: AnnealedTransportWarmstartStateTF | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    _validate_transport_ad_mode(transport_ad_mode)
    batch_size = tf.shape(log_alpha)[0]
    continue_flag = tf.ones([batch_size], dtype=tf.bool)
    epsilon_0 = particles_diameter ** 2
    scaling_factor = scaling ** 2

    if warmstart_state is None:
        a_y_init = _filterflow_exact_softmin(epsilon_0, cost_yx, log_alpha)
        b_x_init = _filterflow_exact_softmin(epsilon_0, cost_xy, log_beta)
        a_x_init = _filterflow_exact_softmin(epsilon_0, cost_xx, log_alpha)
        b_y_init = _filterflow_exact_softmin(epsilon_0, cost_yy, log_beta)
    else:
        warm_mask = tf.reshape(tf.cast(warmstart_state.valid_mask, tf.bool), [-1])
        a_y_cold = _filterflow_exact_softmin(epsilon_0, cost_yx, log_alpha)
        b_x_cold = _filterflow_exact_softmin(epsilon_0, cost_xy, log_beta)
        a_x_cold = _filterflow_exact_softmin(epsilon_0, cost_xx, log_alpha)
        b_y_cold = _filterflow_exact_softmin(epsilon_0, cost_yy, log_beta)
        a_y_init = tf.where(
            warm_mask[:, None],
            tf.cast(warmstart_state.a_y, DTYPE),
            a_y_cold,
        )
        b_x_init = tf.where(
            warm_mask[:, None],
            tf.cast(warmstart_state.b_x, DTYPE),
            b_x_cold,
        )
        a_x_init = tf.where(
            warm_mask[:, None],
            tf.cast(warmstart_state.a_x, DTYPE),
            a_x_cold,
        )
        b_y_init = tf.where(
            warm_mask[:, None],
            tf.cast(warmstart_state.b_y, DTYPE),
            b_y_cold,
        )

    def stop_condition(i, _a_y, _b_x, _a_x, _b_y, continue_, _running_epsilon):
        n_iter_cond = i < max_iter - 1
        return tf.logical_and(n_iter_cond, tf.reduce_all(continue_))

    def apply_one(a_y, b_x, a_x, b_y, continue_, running_epsilon):
        running_epsilon_ = tf.reshape(running_epsilon, [-1, 1])
        continue_reshaped = tf.reshape(continue_, [-1, 1])
        at_y = tf.where(
            continue_reshaped,
            _filterflow_exact_softmin(
                running_epsilon,
                cost_yx,
                log_alpha + b_x / running_epsilon_,
            ),
            a_y,
        )
        bt_x = tf.where(
            continue_reshaped,
            _filterflow_exact_softmin(
                running_epsilon,
                cost_xy,
                log_beta + a_y / running_epsilon_,
            ),
            b_x,
        )
        at_x = tf.where(
            continue_reshaped,
            _filterflow_exact_softmin(
                running_epsilon,
                cost_xx,
                log_alpha + a_x / running_epsilon_,
            ),
            a_x,
        )
        bt_y = tf.where(
            continue_reshaped,
            _filterflow_exact_softmin(
                running_epsilon,
                cost_yy,
                log_beta + b_y / running_epsilon_,
            ),
            b_y,
        )
        a_y_new = (a_y + at_y) / 2
        b_x_new = (b_x + bt_x) / 2
        a_x_new = (a_x + at_x) / 2
        b_y_new = (b_y + bt_y) / 2
        a_y_diff = tf.reduce_max(tf.abs(a_y_new - a_y), 1)
        b_x_diff = tf.reduce_max(tf.abs(b_x_new - b_x), 1)
        local_continue = tf.logical_or(a_y_diff > threshold, b_x_diff > threshold)
        return a_y_new, b_x_new, a_x_new, b_y_new, local_continue

    def body(i, a_y, b_x, a_x, b_y, continue_, running_epsilon):
        new_a_y, new_b_x, new_a_x, new_b_y, local_continue = apply_one(
            a_y,
            b_x,
            a_x,
            b_y,
            continue_,
            running_epsilon,
        )
        new_epsilon = tf.maximum(running_epsilon * scaling_factor, epsilon)
        global_continue = tf.logical_or(new_epsilon < running_epsilon, local_continue)
        return (
            i + 1,
            new_a_y,
            new_b_x,
            new_a_x,
            new_b_y,
            global_continue,
            new_epsilon,
        )

    n_iter = tf.constant(0)
    (
        total_iter,
        converged_a_y,
        converged_b_x,
        converged_a_x,
        converged_b_y,
        _,
        _final_epsilon,
    ) = tf.while_loop(
        stop_condition,
        body,
        loop_vars=(
            n_iter,
            a_y_init,
            b_x_init,
            a_x_init,
            b_y_init,
            continue_flag,
            epsilon_0,
        ),
        maximum_iterations=max_iter,
    )
    if _transport_ad_stop_potentials(transport_ad_mode):
        (
            converged_a_y,
            converged_b_x,
            converged_a_x,
            converged_b_y,
        ) = tf.nest.map_structure(
            tf.stop_gradient,
            (converged_a_y, converged_b_x, converged_a_x, converged_b_y),
        )
    epsilon_ = tf.reshape(epsilon, [-1, 1])
    final_a_y = _filterflow_exact_softmin(
        epsilon,
        cost_yx,
        log_alpha + converged_b_x / epsilon_,
    )
    final_b_x = _filterflow_exact_softmin(
        epsilon,
        cost_xy,
        log_beta + converged_a_y / epsilon_,
    )
    final_a_x = _filterflow_exact_softmin(
        epsilon,
        cost_xx,
        log_alpha + converged_a_x / epsilon_,
    )
    final_b_y = _filterflow_exact_softmin(
        epsilon,
        cost_yy,
        log_beta + converged_b_y / epsilon_,
    )
    return final_a_y, final_b_x, final_a_x, final_b_y, total_iter + 2


def _filterflow_exact_softmin(
    epsilon: tf.Tensor,
    cost_matrix: tf.Tensor,
    values: tf.Tensor,
) -> tf.Tensor:
    n = cost_matrix.shape[1]
    batch_size = tf.shape(cost_matrix)[0]
    values_reshaped = tf.reshape(values, (batch_size, 1, n))
    temp = values_reshaped - cost_matrix / tf.reshape(epsilon, (-1, 1, 1))
    log_sum_exp = tf.reduce_logsumexp(temp, axis=2)
    return -tf.reshape(epsilon, (-1, 1)) * log_sum_exp


@tf.function
def _filterflow_exact_cost(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return _filterflow_exact_squared_distances(x, y) / 2.0


@tf.function
def _filterflow_exact_squared_distances(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    xx = tf.reduce_sum(x * x, axis=2, keepdims=True)
    xy = tf.matmul(x, y, transpose_b=True)
    yy = tf.expand_dims(tf.reduce_sum(y * y, axis=-1), 1)
    return tf.clip_by_value(xx - 2 * xy + yy, 0.0, float("inf"))


@tf.function
def _filterflow_exact_max_min(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    max_max = tf.maximum(tf.math.reduce_max(x, [1, 2]), tf.math.reduce_max(y, [1, 2]))
    min_min = tf.minimum(tf.math.reduce_min(x, [1, 2]), tf.math.reduce_min(y, [1, 2]))
    return max_max - min_min


def _filterflow_scale(x: tf.Tensor) -> tf.Tensor:
    std = tf.math.reduce_std(tf.cast(x, DTYPE), axis=1)
    diameter = tf.reduce_max(std, axis=1)
    diameter = tf.where(diameter == 0.0, tf.ones_like(diameter), diameter)
    dimension = tf.cast(tf.shape(x)[2], DTYPE)
    return diameter * tf.sqrt(dimension)


def _filterflow_epsilon_start(scaled_x: tf.Tensor) -> tf.Tensor:
    coordinate_range = tf.reduce_max(scaled_x, axis=[1, 2]) - tf.reduce_min(scaled_x, axis=[1, 2])
    return tf.maximum(coordinate_range * coordinate_range, tf.constant(1e-6, DTYPE))


def _pairwise_squared(x: tf.Tensor) -> tf.Tensor:
    xx = tf.reduce_sum(x * x, axis=2, keepdims=True)
    xy = tf.matmul(x, x, transpose_b=True)
    yy = tf.expand_dims(tf.reduce_sum(x * x, axis=-1), axis=1)
    return tf.clip_by_value(xx - 2.0 * xy + yy, 0.0, float("inf"))


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())
