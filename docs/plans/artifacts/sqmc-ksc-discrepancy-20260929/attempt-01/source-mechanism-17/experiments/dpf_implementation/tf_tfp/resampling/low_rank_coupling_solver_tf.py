"""Experimental TensorFlow low-rank coupling solver-route prototype.

This Agent C module is a P12 diagnostic lane.  It mirrors the anchored
low-rank coupling factor form and Dykstra-style marginal projection route, but
uses deterministic local initialization and a simplified fixed update.  The
whole route is therefore a diagnostic semantic-replacement prototype, not a
claim of dense Sinkhorn equivalence or production/default readiness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import tensorflow as tf


DEFAULT_DTYPE = tf.float64
DTYPE = DEFAULT_DTYPE


@dataclass(frozen=True)
class LowRankCouplingSolverTFResult:
    particles: tf.Tensor
    log_weights: tf.Tensor
    transport_matrix: tf.Tensor
    q_factor: tf.Tensor
    r_factor: tf.Tensor
    g_weights: tf.Tensor
    diagnostics: dict[str, Any]


class LowRankCouplingSolverTensorsTF(NamedTuple):
    particles: tf.Tensor
    log_weights: tf.Tensor
    transport_matrix: tf.Tensor
    q_factor: tf.Tensor
    r_factor: tf.Tensor
    g_weights: tf.Tensor
    projection_iterations_used: tf.Tensor
    max_q_row_residual: tf.Tensor
    max_q_column_residual: tf.Tensor
    max_r_row_residual: tf.Tensor
    max_r_column_residual: tf.Tensor
    max_factor_marginal_residual: tf.Tensor
    max_induced_row_residual: tf.Tensor
    max_induced_column_residual: tf.Tensor
    projection_error: tf.Tensor
    projection_floor_hits: tf.Tensor
    projection_min_denominator: tf.Tensor
    min_q: tf.Tensor
    min_r: tf.Tensor
    min_g: tf.Tensor
    finite_factors: tf.Tensor
    finite_particles: tf.Tensor
    nonnegative_factors: tf.Tensor
    positive_g: tf.Tensor


def low_rank_coupling_solver_resample_tensors_tf(
    particles: tf.Tensor,
    log_weights: tf.Tensor,
    *,
    rank: int,
    assignment_epsilon: float = 0.5,
    alpha: float = 1.0e-8,
    max_projection_iterations: int = 240,
    convergence_threshold: float = 1.0e-6,
    denominator_floor: float = 1.0e-30,
) -> LowRankCouplingSolverTensorsTF:
    """Apply the deterministic low-rank coupling route with tensor outputs."""

    if rank <= 0:
        raise ValueError("rank must be positive")
    if assignment_epsilon <= 0.0:
        raise ValueError("assignment_epsilon must be positive")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if max_projection_iterations <= 0:
        raise ValueError("max_projection_iterations must be positive")
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
    if int(x.shape[1] or 0) and rank > int(x.shape[1]):
        raise ValueError("rank must be <= particle count")

    static_batch_size = int(x.shape[0] or 0)
    static_num_particles = int(x.shape[1] or 0)
    batch_size = static_batch_size if static_batch_size else tf.shape(x)[0]
    num_particles = static_num_particles if static_num_particles else tf.shape(x)[1]
    if not int(x.shape[1] or 0):
        with tf.control_dependencies([
            tf.debugging.assert_less_equal(tf.cast(rank, tf.int32), num_particles, message="rank must be <= particle count")
        ]):
            x = tf.identity(x)

    float_n = tf.cast(num_particles, DTYPE)
    source_weights = tf.exp(logw)
    source_weights = source_weights / tf.reduce_sum(source_weights, axis=1, keepdims=True)
    target_weights = tf.ones([batch_size, num_particles], dtype=DTYPE) / float_n
    rank_tensor = tf.cast(rank, DTYPE)

    g_init = tf.ones([batch_size, rank], dtype=DTYPE) / rank_tensor
    alpha_tensor = tf.constant(alpha, dtype=DTYPE)
    if alpha * rank >= 1.0:
        raise ValueError("alpha must be smaller than 1/rank")

    scaled_x = _scaled_centered_particles(x)
    latent = tf.gather(scaled_x, _deterministic_landmark_indices(num_particles, rank), axis=1)
    base_kernel = _assignment_kernel(scaled_x, latent, epsilon=tf.constant(assignment_epsilon, DTYPE))
    eps_q = tf.maximum(base_kernel, tf.constant(denominator_floor, dtype=DTYPE))
    eps_r = tf.maximum(_cost_nudged_assignment_kernel(scaled_x, latent, assignment_epsilon), tf.constant(denominator_floor, dtype=DTYPE))
    eps_g = tf.maximum(g_init, alpha_tensor)

    q_factor, r_factor, g_weights, projection_diag = _lr_dykstra_projection(
        eps_q,
        eps_r,
        eps_g,
        target_weights,
        source_weights,
        alpha=alpha_tensor,
        stop_threshold=tf.constant(convergence_threshold, dtype=DTYPE),
        max_iterations=max_projection_iterations,
        denominator_floor=tf.constant(denominator_floor, dtype=DTYPE),
    )

    transported_unit_mass = low_rank_coupling_apply_tf(q_factor, r_factor, g_weights, x)
    transported = float_n * transported_unit_mass
    row_mass = float_n * low_rank_coupling_matvec_tf(q_factor, r_factor, g_weights, tf.ones_like(source_weights))
    column_mass = float_n * low_rank_coupling_transpose_matvec_tf(
        q_factor,
        r_factor,
        g_weights,
        tf.ones_like(target_weights),
    )
    row_target = tf.ones([batch_size, num_particles], dtype=DTYPE)
    column_target = source_weights * float_n
    q_row = tf.reduce_sum(q_factor, axis=2)
    q_col = tf.reduce_sum(q_factor, axis=1)
    r_row = tf.reduce_sum(r_factor, axis=2)
    r_col = tf.reduce_sum(r_factor, axis=1)
    q_row_residual = tf.reduce_max(tf.abs(q_row - target_weights))
    q_col_residual = tf.reduce_max(tf.abs(q_col - g_weights))
    r_row_residual = tf.reduce_max(tf.abs(r_row - source_weights))
    r_col_residual = tf.reduce_max(tf.abs(r_col - g_weights))
    factor_residual = tf.reduce_max(tf.stack([q_row_residual, q_col_residual, r_row_residual, r_col_residual]))
    induced_row_residual = tf.reduce_max(tf.abs(row_mass - row_target))
    induced_col_residual = tf.reduce_max(tf.abs(column_mass - column_target))

    finite_factors = (
        tf.reduce_all(tf.math.is_finite(q_factor))
        & tf.reduce_all(tf.math.is_finite(r_factor))
        & tf.reduce_all(tf.math.is_finite(g_weights))
    )
    finite_particles = tf.reduce_all(tf.math.is_finite(transported))
    nonnegative_factors = tf.reduce_all(q_factor >= 0.0) & tf.reduce_all(r_factor >= 0.0)
    positive_g = tf.reduce_all(g_weights > 0.0)
    uniform_log = tf.fill([batch_size, num_particles], -tf.math.log(float_n))
    transport_matrix = tf.zeros([batch_size, 0, 0], dtype=DTYPE)
    return LowRankCouplingSolverTensorsTF(
        particles=transported[0] if original_particle_rank == 2 else transported,
        log_weights=uniform_log[0] if original_weight_rank == 1 else uniform_log,
        transport_matrix=transport_matrix[0] if original_particle_rank == 2 else transport_matrix,
        q_factor=q_factor[0] if original_particle_rank == 2 else q_factor,
        r_factor=r_factor[0] if original_particle_rank == 2 else r_factor,
        g_weights=g_weights[0] if original_particle_rank == 2 else g_weights,
        projection_iterations_used=projection_diag["iterations_used"],
        max_q_row_residual=tf.cast(q_row_residual, DTYPE),
        max_q_column_residual=tf.cast(q_col_residual, DTYPE),
        max_r_row_residual=tf.cast(r_row_residual, DTYPE),
        max_r_column_residual=tf.cast(r_col_residual, DTYPE),
        max_factor_marginal_residual=tf.cast(factor_residual, DTYPE),
        max_induced_row_residual=tf.cast(induced_row_residual, DTYPE),
        max_induced_column_residual=tf.cast(induced_col_residual, DTYPE),
        projection_error=tf.cast(projection_diag["projection_error"], DTYPE),
        projection_floor_hits=tf.cast(projection_diag["floor_hits"], DTYPE),
        projection_min_denominator=tf.cast(projection_diag["min_denominator"], DTYPE),
        min_q=tf.cast(tf.reduce_min(q_factor), DTYPE),
        min_r=tf.cast(tf.reduce_min(r_factor), DTYPE),
        min_g=tf.cast(tf.reduce_min(g_weights), DTYPE),
        finite_factors=finite_factors,
        finite_particles=finite_particles,
        nonnegative_factors=nonnegative_factors,
        positive_g=positive_g,
    )


def low_rank_coupling_solver_resample_tf(
    particles: tf.Tensor,
    log_weights: tf.Tensor,
    *,
    rank: int,
    assignment_epsilon: float = 0.5,
    alpha: float = 1.0e-8,
    max_projection_iterations: int = 240,
    convergence_threshold: float = 1.0e-6,
    denominator_floor: float = 1.0e-30,
) -> LowRankCouplingSolverTFResult:
    """Apply a deterministic low-rank coupling solver-route diagnostic."""

    tensors = low_rank_coupling_solver_resample_tensors_tf(
        particles,
        log_weights,
        rank=rank,
        assignment_epsilon=assignment_epsilon,
        alpha=alpha,
        max_projection_iterations=max_projection_iterations,
        convergence_threshold=convergence_threshold,
        denominator_floor=denominator_floor,
    )
    diagnostics = {
        "component_id": "low_rank_coupling_solver_tf",
        "mathematical_object": "low_rank_coupling_solver_route",
        "source_status": "source_locked",
        "semantic_class": "semantic_replacement",
        "implementation_scope": "solver_route_dykstra_projection_diagnostic",
        "source_route": "extension_or_invention",
        "source_route_components": {
            "factored_coupling_parameterization": "source_faithful",
            "low_rank_lazy_apply": "source_faithful",
            "factor_marginal_diagnostics": "source_faithful",
            "dykstra_style_projection": "source_faithful",
            "deterministic_initialization": "fixed_hmc_adaptation",
            "fixed_iteration_schedule": "fixed_hmc_adaptation",
            "phase1_scaled_transport_adapter": "fixed_hmc_adaptation",
            "cost_nudged_assignment_kernel": "extension_or_invention",
        },
        "solver_fidelity": "diagnostic_solver_route_not_full_lowrank_sinkhorn_fidelity",
        "backend": "tensorflow",
        "transport_object_kind": "low_rank_coupling_factors",
        "transport_matrix_materialized": False,
        "not_materialized_reason": "low_rank_coupling_factors_nonmaterialized",
        "orientation": "target_rows_source_columns_phase1_scaled",
        "semantic_output": "full_state_particles",
        "rank": int(rank),
        "assignment_epsilon": float(assignment_epsilon),
        "alpha": float(alpha),
        "max_projection_iterations": int(max_projection_iterations),
        "projection_iterations_used": tensors.projection_iterations_used,
        "convergence_threshold": float(convergence_threshold),
        "denominator_floor": float(denominator_floor),
        "factor_shapes": {
            "Q": tensors.q_factor.shape.as_list(),
            "R": tensors.r_factor.shape.as_list(),
            "g": tensors.g_weights.shape.as_list(),
        },
        "max_q_row_residual": tensors.max_q_row_residual,
        "max_q_column_residual": tensors.max_q_column_residual,
        "max_r_row_residual": tensors.max_r_row_residual,
        "max_r_column_residual": tensors.max_r_column_residual,
        "max_factor_marginal_residual": tensors.max_factor_marginal_residual,
        "max_induced_row_residual": tensors.max_induced_row_residual,
        "max_induced_column_residual": tensors.max_induced_column_residual,
        "projection_error": tensors.projection_error,
        "projection_floor_hits": tensors.projection_floor_hits,
        "projection_min_denominator": tensors.projection_min_denominator,
        "min_q": tensors.min_q,
        "min_r": tensors.min_r,
        "min_g": tensors.min_g,
        "finite_factors": tensors.finite_factors,
        "finite_particles": tensors.finite_particles,
        "nonnegative_factors": tensors.nonnegative_factors,
        "positive_g": tensors.positive_g,
        "nonclaims": [
            "Agent C P12 solver-route diagnostic only",
            "semantic replacement, not dense Sinkhorn equivalence",
            "no full low-rank Sinkhorn solver-fidelity claim for extension components",
            "no speedup claim",
            "no ranking claim",
            "no production default change",
            "no posterior correctness claim",
            "no HMC readiness claim",
            "no public API readiness claim",
        ],
    }
    return LowRankCouplingSolverTFResult(
        particles=tensors.particles,
        log_weights=tensors.log_weights,
        transport_matrix=tensors.transport_matrix,
        q_factor=tensors.q_factor,
        r_factor=tensors.r_factor,
        g_weights=tensors.g_weights,
        diagnostics=diagnostics,
    )


def low_rank_coupling_scaled_matrix_tf(q_factor: tf.Tensor, r_factor: tf.Tensor, g_weights: tf.Tensor) -> tf.Tensor:
    """Materialize the Phase-1-scaled factor coupling for tiny diagnostics."""

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


def low_rank_coupling_apply_tf(q_factor: tf.Tensor, r_factor: tf.Tensor, g_weights: tf.Tensor, matrix: tf.Tensor) -> tf.Tensor:
    projected = tf.einsum("bnr,bnd->brd", r_factor, matrix)
    weighted = projected / g_weights[:, :, None]
    return tf.einsum("bnr,brd->bnd", q_factor, weighted)


def low_rank_coupling_matvec_tf(q_factor: tf.Tensor, r_factor: tf.Tensor, g_weights: tf.Tensor, vector: tf.Tensor) -> tf.Tensor:
    projected = tf.einsum("bnr,bn->br", r_factor, vector)
    weighted = projected / g_weights
    return tf.einsum("bnr,br->bn", q_factor, weighted)


def low_rank_coupling_transpose_matvec_tf(
    q_factor: tf.Tensor,
    r_factor: tf.Tensor,
    g_weights: tf.Tensor,
    vector: tf.Tensor,
) -> tf.Tensor:
    projected = tf.einsum("bnr,bn->br", q_factor, vector)
    weighted = projected / g_weights
    return tf.einsum("bnr,br->bn", r_factor, weighted)


def _lr_dykstra_projection(
    eps_q: tf.Tensor,
    eps_r: tf.Tensor,
    eps_g: tf.Tensor,
    target_weights: tf.Tensor,
    source_weights: tf.Tensor,
    *,
    alpha: tf.Tensor,
    stop_threshold: tf.Tensor,
    max_iterations: int,
    denominator_floor: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
    """Mirror the anchored LR-Dykstra projection onto factor marginals."""

    g_tilde = tf.identity(eps_g)
    q3_1 = tf.ones_like(eps_g)
    q3_2 = tf.ones_like(eps_g)
    v1_tilde = tf.ones_like(eps_g)
    v2_tilde = tf.ones_like(eps_g)
    q1 = tf.ones_like(eps_g)
    q2 = tf.ones_like(eps_g)
    min_denominator = tf.constant(float("inf"), dtype=DTYPE)
    floor_hits = tf.constant(0.0, dtype=DTYPE)
    projection_error = tf.constant(float("inf"), dtype=DTYPE)
    iterations_used = tf.constant(max_iterations, dtype=tf.int32)
    q_factor = eps_q
    r_factor = eps_r
    g = tf.maximum(eps_g, alpha)

    max_iterations_tensor = tf.constant(max_iterations, dtype=tf.int32)

    def cond(
        iteration: tf.Tensor,
        _g_tilde: tf.Tensor,
        _q3_1: tf.Tensor,
        _q3_2: tf.Tensor,
        _v1_tilde: tf.Tensor,
        _v2_tilde: tf.Tensor,
        _q1: tf.Tensor,
        _q2: tf.Tensor,
        _min_denominator: tf.Tensor,
        _floor_hits: tf.Tensor,
        current_projection_error: tf.Tensor,
        _q_factor: tf.Tensor,
        _r_factor: tf.Tensor,
        _g: tf.Tensor,
    ) -> tf.Tensor:
        return tf.logical_and(
            iteration < max_iterations_tensor,
            current_projection_error > stop_threshold,
        )

    def body(
        iteration: tf.Tensor,
        g_tilde: tf.Tensor,
        q3_1: tf.Tensor,
        q3_2: tf.Tensor,
        v1_tilde: tf.Tensor,
        v2_tilde: tf.Tensor,
        q1: tf.Tensor,
        q2: tf.Tensor,
        min_denominator: tf.Tensor,
        floor_hits: tf.Tensor,
        _projection_error: tf.Tensor,
        _q_factor: tf.Tensor,
        _r_factor: tf.Tensor,
        _g: tf.Tensor,
    ):
        raw_q_den = tf.einsum("bnr,br->bn", eps_q, v1_tilde)
        raw_r_den = tf.einsum("bnr,br->bn", eps_r, v2_tilde)
        min_denominator = tf.minimum(min_denominator, tf.reduce_min(tf.minimum(raw_q_den, raw_r_den)))
        floor_hits += tf.reduce_sum(tf.cast(raw_q_den <= denominator_floor, DTYPE))
        floor_hits += tf.reduce_sum(tf.cast(raw_r_den <= denominator_floor, DTYPE))
        u1 = target_weights / tf.maximum(raw_q_den, denominator_floor)
        u2 = source_weights / tf.maximum(raw_r_den, denominator_floor)

        g = tf.maximum(alpha, g_tilde * q3_1)
        q3_1 = (g_tilde * q3_1) / g
        g_tilde = tf.identity(g)

        prod1 = (v1_tilde * q1) * tf.einsum("bnr,bn->br", eps_q, u1)
        prod2 = (v2_tilde * q2) * tf.einsum("bnr,bn->br", eps_r, u2)
        g = tf.pow(tf.maximum(g_tilde * q3_2 * prod1 * prod2, denominator_floor), tf.constant(1.0 / 3.0, DTYPE))

        den_v1 = tf.einsum("bnr,bn->br", eps_q, u1)
        den_v2 = tf.einsum("bnr,bn->br", eps_r, u2)
        min_denominator = tf.minimum(min_denominator, tf.reduce_min(tf.minimum(den_v1, den_v2)))
        floor_hits += tf.reduce_sum(tf.cast(den_v1 <= denominator_floor, DTYPE))
        floor_hits += tf.reduce_sum(tf.cast(den_v2 <= denominator_floor, DTYPE))
        v1 = g / tf.maximum(den_v1, denominator_floor)
        v2 = g / tf.maximum(den_v2, denominator_floor)

        q1 = (v1_tilde * q1) / tf.maximum(v1, denominator_floor)
        q2 = (v2_tilde * q2) / tf.maximum(v2, denominator_floor)
        q3_2 = (g_tilde * q3_2) / tf.maximum(g, denominator_floor)
        v1_tilde = tf.identity(v1)
        v2_tilde = tf.identity(v2)
        g_tilde = tf.identity(g)

        q_factor = u1[:, :, None] * eps_q * v1[:, None, :]
        r_factor = u2[:, :, None] * eps_r * v2[:, None, :]
        err1 = tf.reduce_sum(tf.abs(tf.reduce_sum(q_factor, axis=2) - target_weights))
        err2 = tf.reduce_sum(tf.abs(tf.reduce_sum(r_factor, axis=2) - source_weights))
        err3 = tf.reduce_sum(tf.abs(tf.reduce_sum(q_factor, axis=1) - tf.reduce_sum(r_factor, axis=1)))
        projection_error = err1 + err2 + err3
        return (
            iteration + tf.constant(1, dtype=tf.int32),
            g_tilde,
            q3_1,
            q3_2,
            v1_tilde,
            v2_tilde,
            q1,
            q2,
            min_denominator,
            floor_hits,
            projection_error,
            q_factor,
            r_factor,
            g,
        )

    (
        iterations_used,
        g_tilde,
        q3_1,
        q3_2,
        v1_tilde,
        v2_tilde,
        q1,
        q2,
        min_denominator,
        floor_hits,
        projection_error,
        q_factor,
        r_factor,
        g,
    ) = tf.while_loop(
        cond,
        body,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            g_tilde,
            q3_1,
            q3_2,
            v1_tilde,
            v2_tilde,
            q1,
            q2,
            min_denominator,
            floor_hits,
            projection_error,
            q_factor,
            r_factor,
            g,
        ),
        parallel_iterations=1,
        maximum_iterations=max_iterations,
    )

    return q_factor, r_factor, g, {
        "iterations_used": iterations_used,
        "projection_error": projection_error,
        "min_denominator": min_denominator,
        "floor_hits": floor_hits,
    }


def _scaled_centered_particles(x: tf.Tensor) -> tf.Tensor:
    centered = x - tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
    std = tf.math.reduce_std(x, axis=1)
    scale = tf.reduce_max(std, axis=1)
    scale = tf.where(scale == 0.0, tf.ones_like(scale), scale)
    static_dim = int(x.shape[2] or 0)
    dim = tf.cast(static_dim if static_dim else tf.shape(x)[2], DTYPE)
    return centered / tf.stop_gradient(scale[:, None, None] * tf.sqrt(dim))


def _deterministic_landmark_indices(num_particles: tf.Tensor, rank: int) -> tf.Tensor:
    if rank == 1:
        return tf.zeros([1], dtype=tf.int32)
    end = tf.cast(num_particles - 1, DTYPE)
    return tf.cast(tf.round(tf.linspace(tf.constant(0.0, DTYPE), end, rank)), tf.int32)


def _assignment_kernel(x: tf.Tensor, latent: tf.Tensor, *, epsilon: tf.Tensor) -> tf.Tensor:
    cost = 0.5 * _pairwise_squared_cross(x, latent)
    logits = -cost / epsilon
    logits = logits - tf.reduce_max(logits, axis=2, keepdims=True)
    return tf.maximum(tf.exp(logits), tf.constant(1.0e-300, dtype=DTYPE))


def _cost_nudged_assignment_kernel(x: tf.Tensor, latent: tf.Tensor, epsilon: tf.Tensor) -> tf.Tensor:
    reflected = -x
    return _assignment_kernel(reflected, latent, epsilon=tf.cast(epsilon, DTYPE))


def _pairwise_squared_cross(query: tf.Tensor, key: tf.Tensor) -> tf.Tensor:
    xx = tf.reduce_sum(query * query, axis=2, keepdims=True)
    xy = tf.matmul(query, key, transpose_b=True)
    yy = tf.expand_dims(tf.reduce_sum(key * key, axis=-1), axis=1)
    return tf.clip_by_value(xx - 2.0 * xy + yy, 0.0, float("inf"))
