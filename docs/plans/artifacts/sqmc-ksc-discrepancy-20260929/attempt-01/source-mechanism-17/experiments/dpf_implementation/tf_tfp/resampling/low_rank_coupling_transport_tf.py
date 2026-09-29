"""Experimental TensorFlow low-rank coupling transport object.

This Phase 6 module validates the BayesFilter transport object route for
direct low-rank couplings.  It is a semantic-replacement candidate and a
transport-object fixture route, not a port of the full low-rank Sinkhorn
solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorflow as tf


DEFAULT_DTYPE = tf.float64
DTYPE = DEFAULT_DTYPE


@dataclass(frozen=True)
class LowRankCouplingTransportTFResult:
    particles: tf.Tensor
    log_weights: tf.Tensor
    transport_matrix: tf.Tensor
    q_factor: tf.Tensor
    r_factor: tf.Tensor
    g_weights: tf.Tensor
    diagnostics: dict[str, Any]


def low_rank_coupling_transport_resample_tf(
    particles: tf.Tensor,
    log_weights: tf.Tensor,
    *,
    rank: int,
    assignment_epsilon: float = 0.5,
    max_iterations: int = 160,
    convergence_threshold: float = 1.0e-5,
    denominator_floor: float = 1.0e-30,
) -> LowRankCouplingTransportTFResult:
    """Apply a deterministic low-rank coupling transport-object fixture."""

    if rank <= 0:
        raise ValueError("rank must be positive")
    if assignment_epsilon <= 0.0:
        raise ValueError("assignment_epsilon must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if convergence_threshold <= 0.0:
        raise ValueError("convergence_threshold must be positive")
    if denominator_floor <= 0.0:
        raise ValueError("denominator_floor must be positive")

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

    num_particles = tf.shape(x)[1]
    if int(x.shape[1] or 0) and rank > int(x.shape[1]):
        raise ValueError("rank must be <= particle count")
    if not int(x.shape[1] or 0):
        rank_tensor = tf.cast(rank, tf.int32)
        with tf.control_dependencies([
            tf.debugging.assert_less_equal(rank_tensor, num_particles, message="rank must be <= particle count")
        ]):
            x = tf.identity(x)

    batch_size = tf.shape(x)[0]
    float_n = tf.cast(num_particles, DTYPE)
    float_rank = tf.cast(rank, DTYPE)
    source_weights = tf.exp(logw)
    source_weights = source_weights / tf.reduce_sum(source_weights, axis=1, keepdims=True)
    target_weights = tf.ones([batch_size, num_particles], dtype=DTYPE) / float_n
    g_weights = tf.ones([batch_size, rank], dtype=DTYPE) / float_rank

    centered = x - tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
    scale = _filterflow_scale(x)
    scaled_x = centered / tf.stop_gradient(scale[:, None, None])
    landmark_indices = _deterministic_landmark_indices(num_particles, rank)
    latent_points = tf.gather(scaled_x, landmark_indices, axis=1)
    assignment_kernel = _assignment_kernel(
        scaled_x,
        latent_points,
        epsilon=tf.constant(assignment_epsilon, DTYPE),
    )
    q_factor, q_scale_diag = _scale_positive_matrix(
        assignment_kernel,
        target_weights,
        g_weights,
        max_iterations=max_iterations,
        convergence_threshold=tf.constant(convergence_threshold, DTYPE),
        denominator_floor=tf.constant(denominator_floor, DTYPE),
    )
    r_factor, r_scale_diag = _scale_positive_matrix(
        assignment_kernel,
        source_weights,
        g_weights,
        max_iterations=max_iterations,
        convergence_threshold=tf.constant(convergence_threshold, DTYPE),
        denominator_floor=tf.constant(denominator_floor, DTYPE),
    )

    unscaled_transport = _low_rank_apply(q_factor, r_factor, g_weights, x)
    transported = float_n * unscaled_transport
    row_mass = float_n * _low_rank_matvec(q_factor, r_factor, g_weights, tf.ones_like(source_weights))
    q_colsum = tf.reduce_sum(q_factor, axis=1)
    column_mass = float_n * tf.reduce_sum(r_factor * (q_colsum / g_weights)[:, None, :], axis=2)
    row_target = tf.ones([batch_size, num_particles], dtype=DTYPE)
    column_target = source_weights * float_n

    q_row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(q_factor, axis=2) - target_weights))
    q_col_residual = tf.reduce_max(tf.abs(tf.reduce_sum(q_factor, axis=1) - g_weights))
    r_row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(r_factor, axis=2) - source_weights))
    r_col_residual = tf.reduce_max(tf.abs(tf.reduce_sum(r_factor, axis=1) - g_weights))
    induced_row_residual = tf.reduce_max(tf.abs(row_mass - row_target))
    induced_column_residual = tf.reduce_max(tf.abs(column_mass - column_target))
    max_factor_marginal_residual = tf.reduce_max(
        tf.stack([q_row_residual, q_col_residual, r_row_residual, r_col_residual])
    )
    uniform_log = tf.fill([batch_size, num_particles], -tf.math.log(float_n))
    transport_matrix = tf.zeros([batch_size, 0, 0], dtype=DTYPE)
    finite_factors = (
        tf.reduce_all(tf.math.is_finite(q_factor))
        & tf.reduce_all(tf.math.is_finite(r_factor))
        & tf.reduce_all(tf.math.is_finite(g_weights))
    )
    nonnegative_factors = tf.reduce_all(q_factor >= 0.0) & tf.reduce_all(r_factor >= 0.0)
    positive_g = tf.reduce_all(g_weights > 0.0)
    finite_particles = tf.reduce_all(tf.math.is_finite(transported))
    diagnostics = {
        "component_id": "low_rank_coupling_transport_tf",
        "mathematical_object": "low_rank_coupling_transport_object_fixture",
        "source_status": "source_locked",
        "semantic_class": "semantic_replacement",
        "implementation_scope": "transport_object_fixture_route",
        "source_route": "extension_or_invention",
        "source_route_components": {
            "factored_coupling_parameterization": "source_faithful",
            "low_rank_lazy_apply": "source_faithful",
            "factor_marginal_diagnostics": "source_faithful",
            "phase1_scaled_transport_adapter": "fixed_hmc_adaptation",
            "deterministic_latent_assignment_factors": "extension_or_invention",
        },
        "solver_fidelity": "not_claimed_transport_object_fixture_route",
        "backend": "tensorflow",
        "transport_object_kind": "low_rank_coupling_factors",
        "transport_matrix_materialized": False,
        "not_materialized_reason": "low_rank_coupling_factors_nonmaterialized",
        "orientation": "target_rows_source_columns_phase1_scaled",
        "phase1_scaled_transport": "Z = N * Q diag(1/g) R^T",
        "rank": int(rank),
        "latent_weight_rule": "uniform_simplex",
        "assignment_rule": "deterministic_landmark_gaussian_assignments",
        "landmark_indices": [int(v) for v in landmark_indices.numpy().tolist()],
        "assignment_epsilon": float(assignment_epsilon),
        "max_iterations": int(max_iterations),
        "q_iterations_used": int(q_scale_diag["iterations_used"].numpy()),
        "r_iterations_used": int(r_scale_diag["iterations_used"].numpy()),
        "convergence_threshold": float(convergence_threshold),
        "denominator_floor": float(denominator_floor),
        "factor_shapes": {
            "Q": q_factor.shape.as_list(),
            "R": r_factor.shape.as_list(),
            "g": g_weights.shape.as_list(),
        },
        "max_q_row_residual": _float(q_row_residual),
        "max_q_column_residual": _float(q_col_residual),
        "max_r_row_residual": _float(r_row_residual),
        "max_r_column_residual": _float(r_col_residual),
        "max_factor_marginal_residual": _float(max_factor_marginal_residual),
        "max_induced_row_residual": _float(induced_row_residual),
        "max_induced_column_residual": _float(induced_column_residual),
        "total_mass_residual": _float(tf.reduce_max(tf.abs(tf.reduce_sum(row_mass, axis=1) - float_n))),
        "min_g": _float(tf.reduce_min(g_weights)),
        "min_factor": _float(tf.minimum(tf.reduce_min(q_factor), tf.reduce_min(r_factor))),
        "q_min_denominator": _float(q_scale_diag["min_denominator"]),
        "r_min_denominator": _float(r_scale_diag["min_denominator"]),
        "q_denominator_floor_hits": _float(q_scale_diag["floor_hits"]),
        "r_denominator_floor_hits": _float(r_scale_diag["floor_hits"]),
        "finite_factors": bool(finite_factors.numpy()),
        "nonnegative_factors": bool(nonnegative_factors.numpy()),
        "positive_g": bool(positive_g.numpy()),
        "finite_particles": bool(finite_particles.numpy()),
        "nonclaims": [
            "experimental Phase 6 semantic-replacement candidate only",
            "transport-object fixture route, not low-rank Sinkhorn solver fidelity",
            "no dense Sinkhorn equivalence claim",
            "no speedup claim",
            "no ranking claim",
            "no production default change",
            "no posterior correctness claim",
        ],
    }
    if not diagnostics["finite_factors"] or not diagnostics["finite_particles"]:
        raise FloatingPointError("low-rank coupling transport emitted non-finite values")
    if not diagnostics["nonnegative_factors"]:
        raise FloatingPointError("low-rank coupling factors contain negative entries")
    if not diagnostics["positive_g"]:
        raise FloatingPointError("low-rank coupling g weights must be strictly positive")

    result_particles = transported[0] if original_particle_rank == 2 else transported
    result_log_weights = uniform_log[0] if original_weight_rank == 1 else uniform_log
    result_transport = transport_matrix[0] if original_particle_rank == 2 else transport_matrix
    result_q = q_factor[0] if original_particle_rank == 2 else q_factor
    result_r = r_factor[0] if original_particle_rank == 2 else r_factor
    result_g = g_weights[0] if original_particle_rank == 2 else g_weights
    return LowRankCouplingTransportTFResult(
        particles=result_particles,
        log_weights=result_log_weights,
        transport_matrix=result_transport,
        q_factor=result_q,
        r_factor=result_r,
        g_weights=result_g,
        diagnostics=diagnostics,
    )


def low_rank_coupling_scaled_matrix_tf(
    q_factor: tf.Tensor,
    r_factor: tf.Tensor,
    g_weights: tf.Tensor,
) -> tf.Tensor:
    """Materialize the Phase-1-scaled low-rank transport for tiny checks."""

    original_rank = len(q_factor.shape)
    q = tf.cast(q_factor, DTYPE)
    r = tf.cast(r_factor, DTYPE)
    g = tf.cast(g_weights, DTYPE)
    if original_rank == 2:
        q = q[None, :, :]
        r = r[None, :, :]
        g = g[None, :]
    scale = tf.cast(tf.shape(q)[1], DTYPE)
    matrix = scale * tf.einsum("bnk,bk,bmk->bnm", q, 1.0 / g, r)
    return matrix[0] if original_rank == 2 else matrix


def _scale_positive_matrix(
    kernel: tf.Tensor,
    row_target: tf.Tensor,
    column_target: tf.Tensor,
    *,
    max_iterations: int,
    convergence_threshold: tf.Tensor,
    denominator_floor: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    u = tf.ones_like(row_target)
    v = tf.ones_like(column_target)
    min_denominator = tf.constant(float("inf"), dtype=DTYPE)
    floor_hits = tf.constant(0.0, dtype=DTYPE)
    iterations_used = tf.constant(max_iterations, dtype=tf.int32)
    for iteration in range(1, max_iterations + 1):
        raw_kv = tf.einsum("bnr,br->bn", kernel, v)
        min_denominator = tf.minimum(min_denominator, tf.reduce_min(raw_kv))
        floor_hits += tf.reduce_sum(tf.cast(raw_kv <= denominator_floor, DTYPE))
        u = row_target / tf.maximum(raw_kv, denominator_floor)
        raw_ktu = tf.einsum("bnr,bn->br", kernel, u)
        min_denominator = tf.minimum(min_denominator, tf.reduce_min(raw_ktu))
        floor_hits += tf.reduce_sum(tf.cast(raw_ktu <= denominator_floor, DTYPE))
        v = column_target / tf.maximum(raw_ktu, denominator_floor)
        row_residual = tf.reduce_max(tf.abs(u * tf.einsum("bnr,br->bn", kernel, v) - row_target))
        col_residual = tf.reduce_max(tf.abs(v * tf.einsum("bnr,bn->br", kernel, u) - column_target))
        if bool((tf.maximum(row_residual, col_residual) <= convergence_threshold).numpy()):
            iterations_used = tf.constant(iteration, dtype=tf.int32)
            break
    scaled = u[:, :, None] * kernel * v[:, None, :]
    return scaled, {
        "iterations_used": iterations_used,
        "min_denominator": min_denominator,
        "floor_hits": floor_hits,
    }


def _low_rank_apply(
    q_factor: tf.Tensor,
    r_factor: tf.Tensor,
    g_weights: tf.Tensor,
    matrix: tf.Tensor,
) -> tf.Tensor:
    projected = tf.einsum("bnr,bnd->brd", r_factor, matrix)
    weighted = projected / g_weights[:, :, None]
    return tf.einsum("bnr,brd->bnd", q_factor, weighted)


def _low_rank_matvec(
    q_factor: tf.Tensor,
    r_factor: tf.Tensor,
    g_weights: tf.Tensor,
    vector: tf.Tensor,
) -> tf.Tensor:
    projected = tf.einsum("bnr,bn->br", r_factor, vector)
    weighted = projected / g_weights
    return tf.einsum("bnr,br->bn", q_factor, weighted)


def _deterministic_landmark_indices(num_particles: tf.Tensor, rank: int) -> tf.Tensor:
    if rank == 1:
        return tf.zeros([1], dtype=tf.int32)
    end = tf.cast(num_particles - 1, DTYPE)
    indices = tf.cast(tf.round(tf.linspace(tf.constant(0.0, DTYPE), end, rank)), tf.int32)
    return indices


def _assignment_kernel(x: tf.Tensor, latent_points: tf.Tensor, *, epsilon: tf.Tensor) -> tf.Tensor:
    cost = 0.5 * _pairwise_squared_cross(x, latent_points)
    logits = -cost / epsilon
    logits = logits - tf.reduce_max(logits, axis=2, keepdims=True)
    return tf.maximum(tf.exp(logits), tf.constant(1.0e-300, DTYPE))


def _pairwise_squared_cross(query: tf.Tensor, key: tf.Tensor) -> tf.Tensor:
    xx = tf.reduce_sum(query * query, axis=2, keepdims=True)
    xy = tf.matmul(query, key, transpose_b=True)
    yy = tf.expand_dims(tf.reduce_sum(key * key, axis=-1), axis=1)
    return tf.clip_by_value(xx - 2.0 * xy + yy, 0.0, float("inf"))


def _filterflow_scale(x: tf.Tensor) -> tf.Tensor:
    std = tf.math.reduce_std(tf.cast(x, DTYPE), axis=1)
    diameter = tf.reduce_max(std, axis=1)
    diameter = tf.where(diameter == 0.0, tf.ones_like(diameter), diameter)
    dimension = tf.cast(tf.shape(x)[2], DTYPE)
    return diameter * tf.sqrt(dimension)


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())
