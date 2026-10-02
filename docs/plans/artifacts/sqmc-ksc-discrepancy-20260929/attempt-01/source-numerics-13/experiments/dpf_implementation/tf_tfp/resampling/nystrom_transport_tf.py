"""Experimental TensorFlow fixed-rank Nystrom annealed transport.

This module is experimental infrastructure for the scalable OT program.  It is
an experimental candidate path only; it does not change BayesFilter defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import tensorflow as tf


DEFAULT_DTYPE = tf.float64
DTYPE = DEFAULT_DTYPE


@dataclass(frozen=True)
class NystromTransportTFResult:
    particles: tf.Tensor
    log_weights: tf.Tensor
    transport_matrix: tf.Tensor
    left_factor: tf.Tensor
    core_matrix: tf.Tensor
    scaling_u: tf.Tensor
    scaling_v: tf.Tensor
    landmark_indices: tf.Tensor
    diagnostics: dict[str, Any]


class NystromTransportTFTensors(NamedTuple):
    particles: tf.Tensor
    log_weights: tf.Tensor
    transport_matrix: tf.Tensor
    left_factor: tf.Tensor
    core_matrix: tf.Tensor
    scaling_u: tf.Tensor
    scaling_v: tf.Tensor
    landmark_indices: tf.Tensor
    max_row_residual: tf.Tensor
    max_column_residual: tf.Tensor
    min_kernel_denominator: tf.Tensor
    denominator_floor_hits: tf.Tensor
    max_abs_log_scaling_gauge_shift: tf.Tensor
    scaling_normalization_applications: tf.Tensor
    max_factor_diag_error: tf.Tensor
    min_factor_diagonal: tf.Tensor
    max_factor_diagonal: tf.Tensor
    landmark_core_min_eigenvalue: tf.Tensor
    landmark_core_max_eigenvalue: tf.Tensor
    landmark_core_condition_proxy: tf.Tensor
    landmark_core_effective_rank: tf.Tensor
    left_factor_min: tf.Tensor
    left_factor_max: tf.Tensor
    core_matrix_min: tf.Tensor
    core_matrix_max: tf.Tensor
    raw_kernel_min: tf.Tensor
    projected_kernel_min: tf.Tensor
    projection_floor_hits: tf.Tensor
    scaling_u_min: tf.Tensor
    scaling_u_max: tf.Tensor
    scaling_v_min: tf.Tensor
    scaling_v_max: tf.Tensor
    finite_factors: tf.Tensor
    finite_particles: tf.Tensor
    iterations_used: tf.Tensor


def nystrom_transport_resample_tensors_tf(
    particles: tf.Tensor,
    log_weights: tf.Tensor,
    *,
    rank: int,
    epsilon: float = 0.5,
    max_iterations: int = 80,
    convergence_threshold: float = 1.0e-4,
    cholesky_jitter: float = 1.0e-8,
    denominator_floor: float = 1.0e-30,
    core_solver: str = "cholesky",
    core_rcond: float = 1.0e-6,
    kernel_mode: str = "raw",
    scaling_normalization: str = "none",
    diagnostics_enabled: bool = False,
) -> NystromTransportTFTensors:
    """Apply fixed-rank Nystrom transport and return tensor diagnostics only."""

    _validate_nystrom_args(
        rank=rank,
        epsilon=epsilon,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        cholesky_jitter=cholesky_jitter,
        denominator_floor=denominator_floor,
        core_solver=core_solver,
        core_rcond=core_rcond,
        kernel_mode=kernel_mode,
        scaling_normalization=scaling_normalization,
        diagnostics_enabled=diagnostics_enabled,
    )
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
    if int(x.shape[1] or 0) and rank > int(x.shape[1]):
        raise ValueError("rank must be <= particle count")
    if not int(x.shape[1] or 0):
        rank_tensor = tf.cast(rank, tf.int32)
        with tf.control_dependencies([
            tf.debugging.assert_less_equal(rank_tensor, num_particles, message="rank must be <= particle count")
        ]):
            x = tf.identity(x)

    centered = x - tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
    scale = _filterflow_scale(x)
    scaled_x = centered / tf.stop_gradient(scale[:, None, None])
    landmark_indices = _deterministic_landmark_indices(num_particles, rank)
    landmarks = tf.gather(scaled_x, landmark_indices, axis=1)
    left_factor, core_matrix, factor_diag = _nystrom_factors(
        scaled_x,
        landmarks,
        epsilon=tf.constant(epsilon, DTYPE),
        cholesky_jitter=tf.constant(cholesky_jitter, DTYPE),
        core_solver=core_solver,
        core_rcond=tf.constant(core_rcond, DTYPE),
        diagnostics_enabled=diagnostics_enabled,
    )
    source_weights = tf.exp(logw)
    float_n = tf.cast(num_particles, DTYPE)
    row_target = tf.ones([batch_size, num_particles], dtype=DTYPE)
    column_target = source_weights * float_n
    denominator_floor_tensor = tf.constant(denominator_floor, DTYPE)
    if kernel_mode == "raw":
        matvec_fn = lambda vector: _factor_matvec(left_factor, core_matrix, vector)
        matmul_fn = lambda matrix: _factor_matmul(left_factor, core_matrix, matrix)
        kernel_diag = {
            "raw_kernel_min": tf.constant(float("nan"), dtype=DTYPE),
            "projected_kernel_min": tf.constant(float("nan"), dtype=DTYPE),
            "projection_floor_hits": tf.constant(0.0, dtype=DTYPE),
        }
    else:
        projected_kernel, kernel_diag = _positive_projected_kernel(
            left_factor,
            core_matrix,
            denominator_floor=denominator_floor_tensor,
        )
        matvec_fn = lambda vector: tf.einsum("bnm,bm->bn", projected_kernel, vector)
        matmul_fn = lambda matrix: tf.einsum("bnm,bmd->bnd", projected_kernel, matrix)
    scaling_u, scaling_v, iterations_used, scale_diag = _sinkhorn_scale_factors(
        matvec_fn,
        row_target,
        column_target,
        max_iterations=max_iterations,
        convergence_threshold=tf.constant(convergence_threshold, DTYPE),
        denominator_floor=denominator_floor_tensor,
        scaling_normalization=scaling_normalization,
    )
    transported = scaling_u[:, :, None] * matmul_fn(scaling_v[:, :, None] * x)
    row_mass = scaling_u * matvec_fn(scaling_v)
    column_mass = scaling_v * matvec_fn(scaling_u)
    row_residual = tf.reduce_max(tf.abs(row_mass - row_target))
    column_residual = tf.reduce_max(tf.abs(column_mass - column_target))
    uniform_log = tf.fill(
        [batch_size, num_particles],
        -tf.math.log(float_n),
    )
    transport_matrix = tf.zeros([batch_size, 0, 0], dtype=DTYPE)
    finite_factors = (
        tf.reduce_all(tf.math.is_finite(left_factor))
        & tf.reduce_all(tf.math.is_finite(core_matrix))
        & tf.reduce_all(tf.math.is_finite(scaling_u))
        & tf.reduce_all(tf.math.is_finite(scaling_v))
    )
    finite_particles = tf.reduce_all(tf.math.is_finite(transported))
    result_particles = transported[0] if original_particle_rank == 2 else transported
    result_log_weights = uniform_log[0] if original_weight_rank == 1 else uniform_log
    result_transport = transport_matrix[0] if original_particle_rank == 2 else transport_matrix
    result_left = left_factor[0] if original_particle_rank == 2 else left_factor
    result_core = core_matrix[0] if original_particle_rank == 2 else core_matrix
    result_u = scaling_u[0] if original_particle_rank == 2 else scaling_u
    result_v = scaling_v[0] if original_particle_rank == 2 else scaling_v
    return NystromTransportTFTensors(
        particles=result_particles,
        log_weights=result_log_weights,
        transport_matrix=result_transport,
        left_factor=result_left,
        core_matrix=result_core,
        scaling_u=result_u,
        scaling_v=result_v,
        landmark_indices=landmark_indices,
        max_row_residual=row_residual,
        max_column_residual=column_residual,
        min_kernel_denominator=scale_diag["min_denominator"],
        denominator_floor_hits=scale_diag["floor_hits"],
        max_abs_log_scaling_gauge_shift=scale_diag["max_abs_log_scaling_gauge_shift"],
        scaling_normalization_applications=scale_diag["scaling_normalization_applications"],
        max_factor_diag_error=factor_diag["max_diag_error"],
        min_factor_diagonal=factor_diag["min_factor_diagonal"],
        max_factor_diagonal=factor_diag["max_factor_diagonal"],
        landmark_core_min_eigenvalue=factor_diag["landmark_core_min_eigenvalue"],
        landmark_core_max_eigenvalue=factor_diag["landmark_core_max_eigenvalue"],
        landmark_core_condition_proxy=factor_diag["landmark_core_condition_proxy"],
        landmark_core_effective_rank=factor_diag["landmark_core_effective_rank"],
        left_factor_min=tf.reduce_min(left_factor),
        left_factor_max=tf.reduce_max(left_factor),
        core_matrix_min=tf.reduce_min(core_matrix),
        core_matrix_max=tf.reduce_max(core_matrix),
        raw_kernel_min=kernel_diag["raw_kernel_min"],
        projected_kernel_min=kernel_diag["projected_kernel_min"],
        projection_floor_hits=kernel_diag["projection_floor_hits"],
        scaling_u_min=tf.reduce_min(scaling_u),
        scaling_u_max=tf.reduce_max(scaling_u),
        scaling_v_min=tf.reduce_min(scaling_v),
        scaling_v_max=tf.reduce_max(scaling_v),
        finite_factors=finite_factors,
        finite_particles=finite_particles,
        iterations_used=iterations_used,
    )


def nystrom_transport_resample_tf(
    particles: tf.Tensor,
    log_weights: tf.Tensor,
    *,
    rank: int,
    epsilon: float = 0.5,
    max_iterations: int = 80,
    convergence_threshold: float = 1.0e-4,
    cholesky_jitter: float = 1.0e-8,
    denominator_floor: float = 1.0e-30,
    core_solver: str = "cholesky",
    core_rcond: float = 1.0e-6,
    kernel_mode: str = "raw",
    scaling_normalization: str = "none",
    diagnostics_enabled: bool = False,
) -> NystromTransportTFResult:
    """Apply a fixed-rank Nystrom kernel transport candidate.

    The source-faithful core is the Nystrom factorization
    ``K ~= V A^{-1} V^T`` and low-rank Sinkhorn scaling through these factors.
    The local FilterFlow scaling and deterministic landmark rule are Phase 4
    adapters, not paper-faithful adaptive-rank claims.
    """

    tensors = nystrom_transport_resample_tensors_tf(
        particles,
        log_weights,
        rank=rank,
        epsilon=epsilon,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        cholesky_jitter=cholesky_jitter,
        denominator_floor=denominator_floor,
        core_solver=core_solver,
        core_rcond=core_rcond,
        kernel_mode=kernel_mode,
        scaling_normalization=scaling_normalization,
        diagnostics_enabled=diagnostics_enabled,
    )
    finite_factors = bool(tensors.finite_factors.numpy())
    finite_particles = bool(tensors.finite_particles.numpy())
    diagnostics = {
        "component_id": "fixed_rank_nystrom_transport_tf",
        "mathematical_object": "approximate_kernel_nystrom_transport",
        "source_status": "source_locked",
        "semantic_class": "approximate_kernel",
        "source_route": "fixed_hmc_adaptation",
        "source_route_components": {
            "nystrom_factors": "source_faithful",
            "low_rank_scaling": "source_faithful",
            "filterflow_cost_scaling_adapter": "fixed_hmc_adaptation",
            "deterministic_landmarks": "fixed_hmc_adaptation",
            "cholesky_jitter": "fixed_hmc_adaptation",
            "positive_projected_kernel": "extension_or_invention",
            "scaling_gauge_normalization": "extension_or_invention",
        },
        "backend": "tensorflow",
        "transport_object_kind": "kernel_factors",
        "transport_matrix_materialized": False,
        "not_materialized_reason": "kernel_factors_nonmaterialized",
        "orientation": "source_rows_target_columns",
        "rank": int(rank),
        "landmark_rule": "deterministic_evenly_spaced_indices",
        "landmark_indices": [int(v) for v in tensors.landmark_indices.numpy().tolist()],
        "epsilon": float(epsilon),
        "eta_map": "eta=1/(2*epsilon) for exp(-0.5*||x-y||^2/epsilon)",
        "reg_sigma_map": "sigma=sqrt(epsilon) for POT exp(-dist/(2*sigma^2)) convention",
        "max_iterations": int(max_iterations),
        "iterations_used": int(tensors.iterations_used.numpy()),
        "convergence_threshold": float(convergence_threshold),
        "cholesky_jitter": float(cholesky_jitter),
        "denominator_floor": float(denominator_floor),
        "core_solver": core_solver,
        "core_rcond": float(core_rcond),
        "kernel_mode": kernel_mode,
        "kernel_mode_scope": (
            "diagnostic_dense_positive_projection"
            if kernel_mode == "positive_projected"
            else "raw_factor_application"
        ),
        "scaling_normalization": scaling_normalization,
        "scaling_normalization_scope": (
            "none"
            if scaling_normalization == "none"
            else "opt_in_batchwise_sinkhorn_factor_gauge_balancing"
        ),
        "diagnostics_enabled": bool(diagnostics_enabled),
        "factor_shapes": {
            "V": tensors.left_factor.shape.as_list(),
            "A_inv": tensors.core_matrix.shape.as_list(),
            "scaling_u": tensors.scaling_u.shape.as_list(),
            "scaling_v": tensors.scaling_v.shape.as_list(),
        },
        "max_row_residual": _float(tensors.max_row_residual),
        "max_column_residual": _float(tensors.max_column_residual),
        "min_kernel_denominator": _float(tensors.min_kernel_denominator),
        "denominator_floor_hits": _float(tensors.denominator_floor_hits),
        "max_abs_log_scaling_gauge_shift": _float(tensors.max_abs_log_scaling_gauge_shift),
        "scaling_normalization_applications": _float(tensors.scaling_normalization_applications),
        "max_factor_diag_error": _float(tensors.max_factor_diag_error),
        "min_factor_diagonal": _float(tensors.min_factor_diagonal),
        "max_factor_diagonal": _float(tensors.max_factor_diagonal),
        "landmark_core_min_eigenvalue": _float(tensors.landmark_core_min_eigenvalue),
        "landmark_core_max_eigenvalue": _float(tensors.landmark_core_max_eigenvalue),
        "landmark_core_condition_proxy": _float(tensors.landmark_core_condition_proxy),
        "landmark_core_effective_rank": _float(tensors.landmark_core_effective_rank),
        "left_factor_min": _float(tensors.left_factor_min),
        "left_factor_max": _float(tensors.left_factor_max),
        "core_matrix_min": _float(tensors.core_matrix_min),
        "core_matrix_max": _float(tensors.core_matrix_max),
        "raw_kernel_min": _float(tensors.raw_kernel_min),
        "projected_kernel_min": _float(tensors.projected_kernel_min),
        "projection_floor_hits": _float(tensors.projection_floor_hits),
        "scaling_u_min": _float(tensors.scaling_u_min),
        "scaling_u_max": _float(tensors.scaling_u_max),
        "scaling_v_min": _float(tensors.scaling_v_min),
        "scaling_v_max": _float(tensors.scaling_v_max),
        "finite_factors": finite_factors,
        "finite_particles": finite_particles,
        "nonclaims": [
            "experimental scalable OT candidate only",
            "no speedup claim",
            "no ranking claim",
            "no production default change",
            "no posterior correctness claim",
        ],
    }
    if not finite_factors or not finite_particles:
        raise FloatingPointError("Nystrom transport emitted non-finite values")
    return NystromTransportTFResult(
        particles=tensors.particles,
        log_weights=tensors.log_weights,
        transport_matrix=tensors.transport_matrix,
        left_factor=tensors.left_factor,
        core_matrix=tensors.core_matrix,
        scaling_u=tensors.scaling_u,
        scaling_v=tensors.scaling_v,
        landmark_indices=tensors.landmark_indices,
        diagnostics=diagnostics,
    )


def _validate_nystrom_args(
    *,
    rank: int,
    epsilon: float,
    max_iterations: int,
    convergence_threshold: float,
    cholesky_jitter: float,
    denominator_floor: float,
    core_solver: str,
    core_rcond: float,
    kernel_mode: str,
    scaling_normalization: str,
    diagnostics_enabled: bool,
) -> None:
    if rank <= 0:
        raise ValueError("rank must be positive")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if convergence_threshold <= 0.0:
        raise ValueError("convergence_threshold must be positive")
    if cholesky_jitter < 0.0:
        raise ValueError("cholesky_jitter must be non-negative")
    if denominator_floor <= 0.0:
        raise ValueError("denominator_floor must be positive")
    if core_solver not in {"cholesky", "eigh_truncated", "svd_truncated"}:
        raise ValueError("core_solver must be cholesky, eigh_truncated, or svd_truncated")
    if core_rcond <= 0.0:
        raise ValueError("core_rcond must be positive")
    if kernel_mode not in {"raw", "positive_projected"}:
        raise ValueError("kernel_mode must be raw or positive_projected")
    if scaling_normalization not in {"none", "balanced"}:
        raise ValueError("scaling_normalization must be none or balanced")
    if not isinstance(diagnostics_enabled, bool):
        raise ValueError("diagnostics_enabled must be a bool")


def _deterministic_landmark_indices(num_particles: tf.Tensor, rank: int) -> tf.Tensor:
    if rank == 1:
        return tf.zeros([1], dtype=tf.int32)
    end = tf.cast(num_particles - 1, DTYPE)
    indices = tf.cast(tf.round(tf.linspace(tf.constant(0.0, DTYPE), end, rank)), tf.int32)
    return indices


def _nystrom_factors(
    x: tf.Tensor,
    landmarks: tf.Tensor,
    *,
    epsilon: tf.Tensor,
    cholesky_jitter: tf.Tensor,
    core_solver: str,
    core_rcond: tf.Tensor,
    diagnostics_enabled: bool,
) -> tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
    v = _gaussian_kernel(x, landmarks, epsilon)
    a = _gaussian_kernel(landmarks, landmarks, epsilon)
    rank = tf.shape(a)[1]
    eye = tf.eye(rank, dtype=DTYPE)[None, :, :]
    jittered = a + cholesky_jitter * eye
    core = _nystrom_core_inverse(jittered, eye, core_solver=core_solver, core_rcond=core_rcond)
    diagonal = tf.einsum("bnr,brs,bns->bn", v, core, v)
    spectrum_diag = _landmark_core_spectrum_diagnostics(
        jittered,
        core_rcond=core_rcond,
        diagnostics_enabled=diagnostics_enabled,
    )
    return v, core, {
        "max_diag_error": tf.reduce_max(tf.abs(1.0 - diagonal)),
        "min_factor_diagonal": tf.reduce_min(diagonal),
        "max_factor_diagonal": tf.reduce_max(diagonal),
        **spectrum_diag,
    }


def _landmark_core_spectrum_diagnostics(
    jittered: tf.Tensor,
    *,
    core_rcond: tf.Tensor,
    diagnostics_enabled: bool,
) -> dict[str, tf.Tensor]:
    if not diagnostics_enabled:
        nan = tf.constant(float("nan"), dtype=DTYPE)
        return {
            "landmark_core_min_eigenvalue": nan,
            "landmark_core_max_eigenvalue": nan,
            "landmark_core_condition_proxy": nan,
            "landmark_core_effective_rank": nan,
        }
    diagnostic_core = _core_solve_precision_tensor(jittered)
    diagnostic_rcond = tf.cast(core_rcond, diagnostic_core.dtype)
    eigenvalues = tf.linalg.eigvalsh(diagnostic_core)
    abs_eigenvalues = tf.abs(eigenvalues)
    max_abs = tf.reduce_max(abs_eigenvalues, axis=-1)
    min_abs = tf.reduce_min(abs_eigenvalues, axis=-1)
    tiny = tf.cast(tf.experimental.numpy.finfo(diagnostic_core.dtype.as_numpy_dtype).tiny, diagnostic_core.dtype)
    cutoff = diagnostic_rcond * max_abs
    effective_rank = tf.reduce_sum(
        tf.cast(abs_eigenvalues > cutoff[:, None], diagnostic_core.dtype),
        axis=-1,
    )
    condition_proxy = max_abs / tf.maximum(min_abs, tiny)
    return {
        "landmark_core_min_eigenvalue": tf.cast(tf.reduce_min(eigenvalues), DTYPE),
        "landmark_core_max_eigenvalue": tf.cast(tf.reduce_max(eigenvalues), DTYPE),
        "landmark_core_condition_proxy": tf.cast(tf.reduce_max(condition_proxy), DTYPE),
        "landmark_core_effective_rank": tf.cast(tf.reduce_min(effective_rank), DTYPE),
    }


def _core_solve_precision_tensor(tensor: tf.Tensor) -> tf.Tensor:
    if tensor.dtype == tf.float32:
        return tf.cast(tensor, tf.float64)
    return tensor


def _nystrom_core_inverse(
    jittered: tf.Tensor,
    eye: tf.Tensor,
    *,
    core_solver: str,
    core_rcond: tf.Tensor,
) -> tf.Tensor:
    if core_solver == "cholesky":
        chol = tf.linalg.cholesky(jittered)
        return tf.linalg.cholesky_solve(chol, eye)
    if core_solver == "eigh_truncated":
        eigenvalues, eigenvectors = tf.linalg.eigh(jittered)
        max_eigenvalue = tf.reduce_max(tf.abs(eigenvalues), axis=-1, keepdims=True)
        cutoff = core_rcond * max_eigenvalue
        inverse_values = tf.where(
            eigenvalues > cutoff,
            tf.math.reciprocal(eigenvalues),
            tf.zeros_like(eigenvalues),
        )
        core = tf.matmul(eigenvectors * inverse_values[:, None, :], eigenvectors, transpose_b=True)
        return 0.5 * (core + tf.linalg.matrix_transpose(core))
    if core_solver == "svd_truncated":
        solve_core = _core_solve_precision_tensor(jittered)
        solve_rcond = tf.cast(core_rcond, solve_core.dtype)
        singular_values, left_vectors, right_vectors = tf.linalg.svd(solve_core, full_matrices=False)
        max_singular_value = tf.reduce_max(singular_values, axis=-1, keepdims=True)
        cutoff = solve_rcond * max_singular_value
        inverse_values = tf.where(
            singular_values > cutoff,
            tf.math.reciprocal(singular_values),
            tf.zeros_like(singular_values),
        )
        core = tf.matmul(right_vectors * inverse_values[:, None, :], left_vectors, transpose_b=True)
        sym_core = 0.5 * (core + tf.linalg.matrix_transpose(core))
        return tf.cast(sym_core, jittered.dtype)
    raise ValueError("core_solver must be cholesky, eigh_truncated, or svd_truncated")


def _sinkhorn_scale_factors(
    matvec_fn,
    row_target: tf.Tensor,
    column_target: tf.Tensor,
    *,
    max_iterations: int,
    convergence_threshold: tf.Tensor,
    denominator_floor: tf.Tensor,
    scaling_normalization: str,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
    u = tf.ones_like(row_target)
    v = tf.ones_like(column_target)
    min_denominator = tf.constant(float("inf"), dtype=DTYPE)
    floor_hits = tf.constant(0.0, dtype=DTYPE)
    max_abs_log_scaling_gauge_shift = tf.constant(0.0, dtype=DTYPE)
    scaling_normalization_applications = tf.constant(0.0, dtype=DTYPE)
    iterations_used = tf.constant(max_iterations, dtype=tf.int32)
    has_converged = tf.constant(False)

    def cond(
        iteration: tf.Tensor,
        _u: tf.Tensor,
        _v: tf.Tensor,
        _min_denominator: tf.Tensor,
        _floor_hits: tf.Tensor,
        _max_abs_log_scaling_gauge_shift: tf.Tensor,
        _scaling_normalization_applications: tf.Tensor,
        _iterations_used: tf.Tensor,
        has_converged: tf.Tensor,
    ) -> tf.Tensor:
        return tf.logical_and(
            iteration <= tf.constant(max_iterations, dtype=tf.int32),
            tf.logical_not(has_converged),
        )

    def body(
        iteration: tf.Tensor,
        current_u: tf.Tensor,
        current_v: tf.Tensor,
        min_denominator: tf.Tensor,
        floor_hits: tf.Tensor,
        max_abs_log_scaling_gauge_shift: tf.Tensor,
        scaling_normalization_applications: tf.Tensor,
        iterations_used: tf.Tensor,
        _has_converged: tf.Tensor,
    ):
        raw_kv = matvec_fn(current_v)
        updated_min_denominator = tf.minimum(min_denominator, tf.reduce_min(raw_kv))
        safe_kv = tf.maximum(raw_kv, denominator_floor)
        updated_floor_hits = floor_hits + tf.reduce_sum(tf.cast(raw_kv <= denominator_floor, DTYPE))
        proposed_u = row_target / safe_kv
        raw_ktu = matvec_fn(proposed_u)
        updated_min_denominator = tf.minimum(updated_min_denominator, tf.reduce_min(raw_ktu))
        safe_ktu = tf.maximum(raw_ktu, denominator_floor)
        updated_floor_hits += tf.reduce_sum(tf.cast(raw_ktu <= denominator_floor, DTYPE))
        proposed_v = column_target / safe_ktu
        if scaling_normalization == "balanced":
            proposed_u, proposed_v, log_c = _balanced_scaling_gauge_normalize(
                proposed_u,
                proposed_v,
                denominator_floor=denominator_floor,
            )
            max_abs_log_scaling_gauge_shift = tf.maximum(
                max_abs_log_scaling_gauge_shift,
                tf.reduce_max(tf.abs(log_c)),
            )
            scaling_normalization_applications += tf.cast(tf.shape(proposed_u)[0], DTYPE)
        row_residual = tf.reduce_max(tf.abs(proposed_u * matvec_fn(proposed_v) - row_target))
        column_residual = tf.reduce_max(
            tf.abs(proposed_v * matvec_fn(proposed_u) - column_target)
        )
        converged = tf.maximum(row_residual, column_residual) <= convergence_threshold
        iterations_used = tf.where(
            converged,
            iteration,
            iterations_used,
        )
        return (
            iteration + tf.constant(1, dtype=tf.int32),
            proposed_u,
            proposed_v,
            updated_min_denominator,
            updated_floor_hits,
            max_abs_log_scaling_gauge_shift,
            scaling_normalization_applications,
            iterations_used,
            converged,
        )

    (
        _,
        u,
        v,
        min_denominator,
        floor_hits,
        max_abs_log_scaling_gauge_shift,
        scaling_normalization_applications,
        iterations_used,
        _,
    ) = tf.while_loop(
        cond,
        body,
        loop_vars=(
            tf.constant(1, dtype=tf.int32),
            u,
            v,
            min_denominator,
            floor_hits,
            max_abs_log_scaling_gauge_shift,
            scaling_normalization_applications,
            iterations_used,
            has_converged,
        ),
        parallel_iterations=1,
        maximum_iterations=max_iterations,
    )
    return u, v, iterations_used, {
        "min_denominator": min_denominator,
        "floor_hits": floor_hits,
        "max_abs_log_scaling_gauge_shift": max_abs_log_scaling_gauge_shift,
        "scaling_normalization_applications": scaling_normalization_applications,
    }


def _balanced_scaling_gauge_normalize(
    u: tf.Tensor,
    v: tf.Tensor,
    *,
    denominator_floor: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    clipped_u = tf.maximum(u, denominator_floor)
    clipped_v = tf.maximum(v, denominator_floor)
    mean_log_u = tf.reduce_mean(tf.math.log(clipped_u), axis=1, keepdims=True)
    mean_log_v = tf.reduce_mean(tf.math.log(clipped_v), axis=1, keepdims=True)
    log_c = 0.5 * (mean_log_u - mean_log_v)
    c = tf.exp(log_c)
    return u / c, v * c, log_c


def _factor_matvec(left_factor: tf.Tensor, core_matrix: tf.Tensor, vector: tf.Tensor) -> tf.Tensor:
    compute_dtype = _core_solve_precision_tensor(left_factor).dtype
    left = tf.cast(left_factor, compute_dtype)
    core = tf.cast(core_matrix, compute_dtype)
    vec = tf.cast(vector, compute_dtype)
    projected = tf.einsum("bnr,bn->br", left, vec)
    solved = tf.einsum("brs,bs->br", core, projected)
    result = tf.einsum("bnr,br->bn", left, solved)
    return tf.cast(result, left_factor.dtype)


def _factor_matmul(left_factor: tf.Tensor, core_matrix: tf.Tensor, matrix: tf.Tensor) -> tf.Tensor:
    compute_dtype = _core_solve_precision_tensor(left_factor).dtype
    left = tf.cast(left_factor, compute_dtype)
    core = tf.cast(core_matrix, compute_dtype)
    mat = tf.cast(matrix, compute_dtype)
    projected = tf.einsum("bnr,bnd->brd", left, mat)
    solved = tf.einsum("brs,bsd->brd", core, projected)
    result = tf.einsum("bnr,brd->bnd", left, solved)
    return tf.cast(result, left_factor.dtype)


def _positive_projected_kernel(
    left_factor: tf.Tensor,
    core_matrix: tf.Tensor,
    *,
    denominator_floor: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    kernel = tf.einsum("bnr,brs,bms->bnm", left_factor, core_matrix, left_factor)
    projected = tf.maximum(kernel, denominator_floor)
    return projected, {
        "raw_kernel_min": tf.reduce_min(kernel),
        "projected_kernel_min": tf.reduce_min(projected),
        "projection_floor_hits": tf.reduce_sum(tf.cast(kernel <= denominator_floor, DTYPE)),
    }


def _gaussian_kernel(x: tf.Tensor, y: tf.Tensor, epsilon: tf.Tensor) -> tf.Tensor:
    cost = 0.5 * _pairwise_squared_cross(x, y)
    return tf.exp(-cost / epsilon)


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
