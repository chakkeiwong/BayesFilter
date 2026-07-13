"""Same-target fixed-SIR LEDH score adapter.

This module gives the amended fixed SIR row its own public score identity while
reusing the repaired no-tape total VJP implementation from the parameterized
SIR diagnostic harness.  The target is the finite-N fixed-randomness
observed-data LEDH likelihood scalar for
``zhao_cui_spatial_sir_austria_j9_T20`` with
``sir_log_scale_theta = (log_kappa_scale, log_nu_scale,
log_obs_noise_scale)``.

The older parameterized diagnostic row remains historical/diagnostic.  Calls
through this adapter must use the full manual transport derivative, not the
stopped-scale diagnostic route.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.ledh_forward_contract import (
    FIXED_SIR_AUSTRIA_ROW_ID,
    LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD,
    LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
    MAIN_OBSERVED_DATA_ROW_SCOPE,
    make_fixed_sir_logscale_forward_contract,
    validate_ledh_forward_scalar_artifact,
    validate_ledh_forward_contract_manifest,
)
from bayesfilter.highdim.ledh_score_contract import (
    LEDH_SCORE_ADMISSION_STATUS_BLOCKED_NOT_RUN,
    LEDH_SCORE_ADMISSION_STATUS_FULL,
    LEDH_SCORE_ADMISSION_STATUS_TINY,
    LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
    LEDH_SCORE_COMPACT_FIXED_SIR_PROVENANCE,
    LEDH_SCORE_MEMORY_STYLE_FIXED_SIR_PROVENANCE,
    LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
    LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
    validate_ledh_score_artifact,
)
from docs.benchmarks import benchmark_p8p_parameterized_sir_gradient as p8p


FIXED_SIR_MANUAL_SCORE_ROUTE_ID = (
    "manual_total_vjp_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot"
)
FIXED_SIR_MEMORY_STYLE_SCORE_ROUTE_ID = LEDH_SCORE_MEMORY_STYLE_FIXED_SIR_PROVENANCE
FIXED_SIR_COMPACT_SCORE_ROUTE_ID = LEDH_SCORE_COMPACT_FIXED_SIR_PROVENANCE
FIXED_SIR_SAME_SCALAR_ROUTE_ID = (
    "same_target_fixed_sir_logscale_ledh_pfpf_ot_streaming_manual_total"
)
PARAMETER_NAMES = p8p.PARAMETER_NAMES
MANUAL_SCORE_COMPONENT_NAMES = p8p.MANUAL_SCORE_COMPONENT_NAMES
NONCLAIMS = (
    "not N=10000 memory evidence",
    "not full leaderboard admission",
    "not exact nonlinear likelihood correctness",
    "not HMC/NUTS readiness",
    "not posterior correctness",
    "not Zhao-Cui TT/SIRT source-faithfulness",
)
FULL_ROW_BATCH_SEEDS = (81120, 81121, 81122, 81123, 81124)
FULL_ROW_NUM_PARTICLES = 10000
FULL_ROW_TIME_STEPS = 20
ROOT = Path(__file__).resolve().parents[2]


def _configure_precision(args: argparse.Namespace) -> dict[str, Any]:
    return p8p._configure_precision(args)  # noqa: SLF001


def _score_precision_metadata(precision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dtype": str(precision.get("dtype")),
        "active_dtype": str(precision.get("active_dtype")),
        "tf_dtype": str(precision.get("tf_dtype")),
        "tf32_mode": str(precision.get("tf32_mode")),
        "tf32_execution_enabled": bool(precision.get("tf32_execution_enabled")),
    }


def _full_leaderboard_identity(args: argparse.Namespace) -> bool:
    return bool(
        int(args.time_steps) == FULL_ROW_TIME_STEPS
        and int(args.num_particles) == FULL_ROW_NUM_PARTICLES
        and tuple(int(seed) for seed in args.batch_seeds)
        == FULL_ROW_BATCH_SEEDS
        and args.transport_policy == "active-all"
        and int(args.sinkhorn_iterations) == 10
        and float(args.sinkhorn_epsilon) == 1.0
    )


def _fixed_sir_forward_contract(args: argparse.Namespace) -> dict[str, Any]:
    contract = make_fixed_sir_logscale_forward_contract(
        time_steps=int(args.time_steps),
        num_particles=int(args.num_particles),
        batch_seeds=[int(seed) for seed in args.batch_seeds],
        full_leaderboard_row=_full_leaderboard_identity(args),
    ).to_manifest()
    return validate_ledh_forward_contract_manifest(contract)


def _require_fixed_sir_score_args(args: argparse.Namespace) -> None:
    if args.transport_plan_mode != "streaming":
        raise ValueError("fixed-SIR admitted score requires streaming transport")
    if args.transport_ad_mode != "full":
        raise ValueError(
            "fixed-SIR admitted score requires transport_ad_mode='full'; "
            "stopped-scale/stabilized routes remain diagnostic only"
        )
    if args.transport_gradient_mode not in {
        p8p.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        p8p.core_tf.MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
    }:
        raise ValueError("fixed-SIR admitted score requires a manual streaming transport VJP")
    if args.transport_policy == "no-resampling":
        raise ValueError("fixed-SIR admitted score requires active relaxed Sinkhorn transport")


def _theta_values(theta_values: Sequence[float] | None) -> list[float]:
    if theta_values is None:
        return [0.0, 0.0, 0.0]
    values = [float(value) for value in theta_values]
    if len(values) != len(PARAMETER_NAMES):
        raise ValueError("fixed-SIR theta must have three log-scale entries")
    return values


def _build_fixed_sir_tensors(
    args: argparse.Namespace,
) -> tuple[dict[str, tf.Tensor], dict[str, Any]]:
    tensors, semantics = p8p._build_base_tensors(args)  # noqa: SLF001
    semantics = dict(semantics)
    semantics.update(
        {
            "row_id": FIXED_SIR_AUSTRIA_ROW_ID,
            "row_scope": MAIN_OBSERVED_DATA_ROW_SCOPE,
            "theta_coordinate_system": "sir_log_scale_theta",
            "score_adapter": "fixed_sir_same_target_score_adapter",
            "diagnostic_module_reused_for_vjp": (
                "docs.benchmarks.benchmark_p8p_parameterized_sir_gradient"
            ),
        }
    )
    return tensors, semantics


def _fixed_sir_manual_score_diagnostic(
    args: argparse.Namespace,
    theta_values: Sequence[float] | None = None,
    *,
    return_score_decomposition: bool = True,
) -> dict[str, Any]:
    """Compute the fixed-SIR no-tape score for the same forward scalar."""

    _require_fixed_sir_score_args(args)
    theta = _theta_values(theta_values)
    tensors, semantics = _build_fixed_sir_tensors(args)
    result = p8p._manual_value_and_score_from_components(  # noqa: SLF001
        tensors,
        args,
        p8p._theta_components(theta),  # noqa: SLF001
        return_score_decomposition=return_score_decomposition,
    )
    contract = _fixed_sir_forward_contract(args)
    component_names = list(MANUAL_SCORE_COMPONENT_NAMES)
    diagnostic: dict[str, Any] = {
        **result,
        "row_id": FIXED_SIR_AUSTRIA_ROW_ID,
        "target_scope": MAIN_OBSERVED_DATA_ROW_SCOPE,
        "forward_contract": contract,
        "target_scalar": contract["target_scalar"],
        "target_output_tensor_field": contract["output_tensor_field"],
        "target_density_fields": contract["target_density_fields"],
        "proposal_flow_fields": contract["proposal_flow_fields"],
        "correction_formula": contract["correction_formula"],
        "sir_semantics": semantics,
        "theta_values": theta,
        "theta_coordinate_system": "sir_log_scale_theta",
        "parameter_order": list(PARAMETER_NAMES),
        "score_route": FIXED_SIR_MEMORY_STYLE_SCORE_ROUTE_ID,
        "historical_manual_score_route": FIXED_SIR_MANUAL_SCORE_ROUTE_ID,
        "score_route_id": FIXED_SIR_SAME_SCALAR_ROUTE_ID,
        "value_score_route_status": "same_route_value_score",
        "value_score_same_transport_algorithm": True,
        "no_autodiff_score_route": True,
        "old_parameterized_route_status": "historical_diagnostic_only",
        "manual_score_component_names": component_names,
        "transport": {
            "value_core_mode": "manual_forward_replay_same_scalar",
            "transport_plan_mode": args.transport_plan_mode,
            "transport_ad_mode": args.transport_ad_mode,
            "gradient_mode": args.transport_gradient_mode,
            "row_chunk_size": int(args.row_chunk_size),
            "col_chunk_size": int(args.col_chunk_size),
            "particle_chunk_size": int(args.particle_chunk_size),
            "sinkhorn_iterations": int(args.sinkhorn_iterations),
            "sinkhorn_epsilon": float(args.sinkhorn_epsilon),
            "dense_transport_matrix_materialized": False,
        },
        "nonclaims": list(NONCLAIMS),
    }
    if return_score_decomposition:
        component_tensor = tf.convert_to_tensor(
            diagnostic["manual_score_components"],
            dtype=p8p.DTYPE,
        )
        diagnostic["manual_score_component_sum"] = tf.reduce_sum(component_tensor, axis=0)
        diagnostic["manual_score_component_reconstruction_max_abs_delta"] = tf.reduce_max(
            tf.abs(
                diagnostic["manual_score_component_sum"]
                - tf.convert_to_tensor(diagnostic["per_seed_gradient"], dtype=p8p.DTYPE)
            )
        )
    return diagnostic


def _fixed_sir_same_scalar_fd_diagnostic(
    args: argparse.Namespace,
    theta_values: Sequence[float] | None = None,
    *,
    direction: Sequence[float] = (0.2, -0.1, 0.3),
) -> dict[str, Any]:
    """Centered directional FD against the same fixed-SIR scalar."""

    _require_fixed_sir_score_args(args)
    theta = tf.constant(_theta_values(theta_values), dtype=p8p.DTYPE)
    direction_tensor = tf.constant([float(value) for value in direction], dtype=p8p.DTYPE)
    if int(direction_tensor.shape[0]) != len(PARAMETER_NAMES):
        raise ValueError("fixed-SIR FD direction must have three entries")
    step = tf.constant(float(args.fd_step), dtype=p8p.DTYPE)
    base = _fixed_sir_manual_score_diagnostic(
        args,
        theta.numpy().tolist(),
        return_score_decomposition=True,
    )
    tensors, _semantics = _build_fixed_sir_tensors(args)
    plus_theta = (theta + step * direction_tensor).numpy().tolist()
    minus_theta = (theta - step * direction_tensor).numpy().tolist()
    plus = p8p._manual_value_and_score_from_components(  # noqa: SLF001
        tensors,
        args,
        p8p._theta_components(plus_theta),  # noqa: SLF001
    )
    minus = p8p._manual_value_and_score_from_components(  # noqa: SLF001
        tensors,
        args,
        p8p._theta_components(minus_theta),  # noqa: SLF001
    )
    score_directional = tf.reduce_sum(
        tf.convert_to_tensor(base["gradient_tensor"], dtype=p8p.DTYPE)
        * direction_tensor
    )
    fd_directional = (plus["objective"] - minus["objective"]) / (2.0 * step)
    abs_error = tf.abs(score_directional - fd_directional)
    rel_error = abs_error / tf.maximum(
        tf.maximum(tf.abs(score_directional), tf.abs(fd_directional)),
        tf.constant(1.0e-12, dtype=p8p.DTYPE),
    )
    atol = tf.constant(float(getattr(args, "score_fd_atol", 1.0e-2)), dtype=p8p.DTYPE)
    rtol = tf.constant(float(getattr(args, "score_fd_rtol", 5.0e-2)), dtype=p8p.DTYPE)
    passed = bool((abs_error <= atol or rel_error <= rtol).numpy())
    return {
        "status": "pass" if passed else "fail",
        "row_id": FIXED_SIR_AUSTRIA_ROW_ID,
        "target_scope": MAIN_OBSERVED_DATA_ROW_SCOPE,
        "score_route": FIXED_SIR_MEMORY_STYLE_SCORE_ROUTE_ID,
        "theta_values": [float(value) for value in theta.numpy().tolist()],
        "direction": [float(value) for value in direction_tensor.numpy().tolist()],
        "fd_step": float(step.numpy()),
        "score_directional": float(score_directional.numpy()),
        "fd_directional": float(fd_directional.numpy()),
        "abs_error": float(abs_error.numpy()),
        "rel_error": float(rel_error.numpy()),
        "atol": float(atol.numpy()),
        "rtol": float(rtol.numpy()),
        "base": base,
    }


def _normalize_log_weights_jvp(
    corrected_log_weights: tf.Tensor,
    d_corrected_log_weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    corrected_log_weights = tf.convert_to_tensor(corrected_log_weights, dtype=p8p.DTYPE)
    d_corrected_log_weights = tf.convert_to_tensor(d_corrected_log_weights, dtype=p8p.DTYPE)
    weights, incremental = p8p.core_tf._normalize_log_weights(corrected_log_weights)  # noqa: SLF001
    d_incremental = tf.reduce_sum(weights[:, :, None] * d_corrected_log_weights, axis=1)
    normalized_log_weights = tf.math.log(
        tf.maximum(weights, p8p.core_tf._log_weight_floor())  # noqa: SLF001
    )
    floor_active = weights <= p8p.core_tf._log_weight_floor()  # noqa: SLF001
    d_normalized = tf.where(
        floor_active[:, :, None],
        tf.zeros_like(d_corrected_log_weights),
        d_corrected_log_weights - d_incremental[:, None, :],
    )
    return normalized_log_weights, d_normalized, incremental, d_incremental


def _batched_gaussian_logpdf_jvp(
    residuals: tf.Tensor,
    covariance: tf.Tensor,
    d_residuals: tf.Tensor,
    d_covariance: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    residuals = tf.convert_to_tensor(residuals, dtype=p8p.DTYPE)
    covariance = tf.convert_to_tensor(covariance, dtype=p8p.DTYPE)
    d_residuals = tf.convert_to_tensor(d_residuals, dtype=p8p.DTYPE)
    d_covariance = tf.convert_to_tensor(d_covariance, dtype=p8p.DTYPE)
    chol = tf.linalg.cholesky(covariance)
    precision = tf.linalg.cholesky_solve(
        chol,
        tf.eye(int(covariance.shape[-1]), dtype=p8p.DTYPE)[tf.newaxis, :, :],
    )
    solved = tf.einsum("bij,bnj->bni", precision, residuals)
    value = p8p.core_tf._batched_gaussian_logpdf(residuals, covariance)  # noqa: SLF001
    covariance_bar_per_particle = tf.constant(0.5, dtype=p8p.DTYPE) * (
        tf.einsum("bni,bnj->bnij", solved, solved)
        - precision[:, tf.newaxis, :, :]
    )
    tangent = (
        -tf.reduce_sum(solved[:, :, :, tf.newaxis] * d_residuals, axis=2)
        + tf.reduce_sum(
            covariance_bar_per_particle[:, :, :, :, tf.newaxis]
            * d_covariance[:, tf.newaxis, :, :, :],
            axis=[2, 3],
        )
    )
    return value, tangent


def _cholesky_jvp_matrix(matrix: tf.Tensor, d_matrix: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    matrix = tf.convert_to_tensor(matrix, dtype=p8p.DTYPE)
    d_matrix = tf.convert_to_tensor(d_matrix, dtype=p8p.DTYPE)
    chol = tf.linalg.cholesky(matrix)
    tangent_columns = []
    param_dim = int(d_matrix.shape[-1])
    for index in range(param_dim):
        tangent = d_matrix[..., index]
        inner = tf.linalg.triangular_solve(chol, tangent, lower=True)
        inner = tf.linalg.triangular_solve(chol, tf.linalg.matrix_transpose(inner), lower=True)
        inner = tf.linalg.matrix_transpose(inner)
        lower = tf.linalg.band_part(inner, -1, 0)
        diag = tf.linalg.diag(tf.linalg.diag_part(lower) * tf.constant(0.5, dtype=p8p.DTYPE))
        phi = lower - tf.linalg.diag(tf.linalg.diag_part(lower)) + diag
        tangent_columns.append(tf.matmul(chol, phi))
    return chol, tf.stack(tangent_columns, axis=-1)


def _filterflow_scale_jvp(particles: tf.Tensor, d_particles: tf.Tensor) -> tf.Tensor:
    particles = tf.convert_to_tensor(particles, dtype=p8p.DTYPE)
    d_particles = tf.convert_to_tensor(d_particles, dtype=p8p.DTYPE)
    num_particles = tf.cast(tf.shape(particles)[1], p8p.DTYPE)
    dimension = tf.cast(tf.shape(particles)[2], p8p.DTYPE)
    mean = tf.reduce_mean(particles, axis=1, keepdims=True)
    d_mean = tf.reduce_mean(d_particles, axis=1, keepdims=True)
    centered = particles - mean
    d_centered = d_particles - d_mean
    variance = tf.reduce_mean(centered * centered, axis=1)
    d_variance = (tf.constant(2.0, dtype=p8p.DTYPE) / num_particles) * tf.reduce_sum(
        centered[:, :, :, None] * d_centered,
        axis=1,
    )
    std = tf.sqrt(variance)
    safe_std = tf.where(std > 0.0, std, tf.ones_like(std))
    d_std = tf.where(
        std[:, :, None] > 0.0,
        d_variance / (tf.constant(2.0, dtype=p8p.DTYPE) * safe_std[:, :, None]),
        tf.zeros_like(d_variance),
    )
    diameter = tf.reduce_max(std, axis=1)
    max_mask = tf.cast(std == diameter[:, None], p8p.DTYPE)
    max_count = tf.reduce_sum(max_mask, axis=1, keepdims=True)
    d_diameter = tf.reduce_sum(
        d_std * max_mask[:, :, None] / max_count[:, :, None],
        axis=1,
    )
    active = tf.cast(diameter != 0.0, p8p.DTYPE)
    return tf.sqrt(dimension) * d_diameter * active[:, None]


def _filterflow_epsilon_start_jvp(
    scaled_x: tf.Tensor,
    d_scaled_x: tf.Tensor,
) -> tf.Tensor:
    scaled_x = tf.convert_to_tensor(scaled_x, dtype=p8p.DTYPE)
    d_scaled_x = tf.convert_to_tensor(d_scaled_x, dtype=p8p.DTYPE)
    max_value = tf.reduce_max(scaled_x, axis=[1, 2])
    min_value = tf.reduce_min(scaled_x, axis=[1, 2])
    coordinate_range = max_value - min_value
    max_mask = tf.cast(scaled_x == max_value[:, None, None], p8p.DTYPE)
    min_mask = tf.cast(scaled_x == min_value[:, None, None], p8p.DTYPE)
    max_count = tf.reduce_sum(max_mask, axis=[1, 2], keepdims=True)
    min_count = tf.reduce_sum(min_mask, axis=[1, 2], keepdims=True)
    d_max = tf.reduce_sum(
        d_scaled_x * max_mask[:, :, :, None] / max_count[:, :, :, None],
        axis=[1, 2],
    )
    d_min = tf.reduce_sum(
        d_scaled_x * min_mask[:, :, :, None] / min_count[:, :, :, None],
        axis=[1, 2],
    )
    active = tf.cast(
        coordinate_range * coordinate_range >= tf.constant(1.0e-6, p8p.DTYPE),
        p8p.DTYPE,
    )
    return tf.constant(2.0, dtype=p8p.DTYPE) * coordinate_range[:, None] * (d_max - d_min) * active[:, None]


def _sir_flatten_tangent_components(susceptible: tf.Tensor, infectious: tf.Tensor) -> tf.Tensor:
    susceptible = tf.convert_to_tensor(susceptible, dtype=p8p.DTYPE)
    infectious = tf.convert_to_tensor(infectious, dtype=p8p.DTYPE)
    output_shape = tf.concat(
        [
            tf.shape(susceptible)[:-2],
            [tf.shape(susceptible)[-2] * 2, tf.shape(susceptible)[-1]],
        ],
        axis=0,
    )
    return tf.reshape(tf.stack([susceptible, infectious], axis=3), output_shape)


def _compact_forward_transport_jvp_tf(
    *,
    post_flow: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    d_post_flow: tf.Tensor,
    d_normalized_log_weights: tf.Tensor,
    mask: tf.Tensor,
    args: argparse.Namespace,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    if args.transport_policy == "no-resampling":
        return post_flow, normalized_log_weights, d_post_flow, d_normalized_log_weights
    if args.transport_gradient_mode != p8p.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE:
        raise ValueError("compact fixed-SIR score requires manual streaming finite transport")
    if args.transport_ad_mode != "full":
        raise ValueError("compact fixed-SIR score requires transport_ad_mode='full'")
    batch_size, num_particles, _state_dim = p8p.core_tf._static_shape(  # noqa: SLF001
        post_flow,
        "post_flow",
    )
    center = tf.reduce_mean(post_flow, axis=1, keepdims=True)
    d_center = tf.reduce_mean(d_post_flow, axis=1, keepdims=True)
    scale = p8p.annealed_transport_tf._filterflow_scale(post_flow)  # noqa: SLF001
    d_scale = _filterflow_scale_jvp(post_flow, d_post_flow)
    scaled_x = (post_flow - center) / scale[:, None, None]
    d_scaled_x = (
        (d_post_flow - d_center) / scale[:, None, None, None]
        - (post_flow - center)[:, :, :, None]
        * d_scale[:, None, None, :]
        / (scale[:, None, None, None] * scale[:, None, None, None])
    )
    epsilon = tf.convert_to_tensor(args.sinkhorn_epsilon, dtype=p8p.DTYPE)
    epsilon0 = p8p.annealed_transport_tf._filterflow_epsilon_start(scaled_x)  # noqa: SLF001
    d_epsilon0 = _filterflow_epsilon_start_jvp(scaled_x, d_scaled_x)
    scaling = tf.convert_to_tensor(args.annealed_scaling, dtype=p8p.DTYPE)
    steps = p8p.core_tf._manual_dense_finite_steps(args.sinkhorn_iterations)  # noqa: SLF001
    transported, d_transported, _row_residual = (
        p8p.annealed_transport_tf._filterflow_manual_streaming_finite_transport_value_and_jvp_total(  # noqa: SLF001
            scaled_x,
            post_flow,
            normalized_log_weights,
            d_scaled_x,
            d_post_flow,
            d_normalized_log_weights,
            d_epsilon0,
            epsilon,
            epsilon0,
            scaling,
            steps=steps,
            row_chunk_size=args.row_chunk_size,
            col_chunk_size=args.col_chunk_size,
        )
    )
    raw_transport = p8p.core_tf.batched_annealed_transport_core_tf(
        post_flow,
        normalized_log_weights,
        mask,
        epsilon=args.sinkhorn_epsilon,
        scaling=args.annealed_scaling,
        convergence_threshold=args.annealed_convergence_threshold,
        max_iterations=args.sinkhorn_iterations,
        transport_gradient_mode="raw",
        transport_plan_mode="streaming",
        transport_ad_mode="stabilized",
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
    )
    uniform_log_weights = p8p.core_tf.uniform_log_weights(batch_size, num_particles)
    d_uniform = tf.zeros_like(d_normalized_log_weights)
    next_d_particles = tf.where(mask[:, None, None, None], d_transported, d_post_flow)
    next_d_log_weights = tf.where(mask[:, None, None], d_uniform, d_normalized_log_weights)
    del transported, uniform_log_weights
    return raw_transport.particles, raw_transport.log_weights, next_d_particles, next_d_log_weights


def _sir_rhs_jvp_tf(
    state: tf.Tensor,
    d_state: tf.Tensor,
    *,
    kappa: tf.Tensor,
    d_kappa: tf.Tensor,
    nu: tf.Tensor,
    d_nu: tf.Tensor,
    adjacency: tf.Tensor,
    neighbor_degree: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    rhs = p8p._sir_rhs_tf(  # noqa: SLF001
        state,
        kappa=kappa,
        nu=nu,
        adjacency=adjacency,
        neighbor_degree=neighbor_degree,
    )
    susceptible = state[:, :, 0::2]
    infectious = state[:, :, 1::2]
    d_susceptible = d_state[:, :, 0::2, :]
    d_infectious = d_state[:, :, 1::2, :]
    susceptible_neighbor_tangent = (
        tf.einsum("bnrp,jr->bnjp", d_susceptible, adjacency)
        - d_susceptible * neighbor_degree[tf.newaxis, tf.newaxis, :, tf.newaxis]
    )
    infectious_neighbor_tangent = (
        tf.einsum("bnrp,jr->bnjp", d_infectious, adjacency)
        - d_infectious * neighbor_degree[tf.newaxis, tf.newaxis, :, tf.newaxis]
    )
    d_infection = (
        d_kappa[tf.newaxis, tf.newaxis, :, :]
        * susceptible[:, :, :, None]
        * infectious[:, :, :, None]
        + kappa[tf.newaxis, tf.newaxis, :, None]
        * d_susceptible
        * infectious[:, :, :, None]
        + kappa[tf.newaxis, tf.newaxis, :, None]
        * susceptible[:, :, :, None]
        * d_infectious
    )
    d_s_rhs = -d_infection + tf.constant(0.5, dtype=p8p.DTYPE) * susceptible_neighbor_tangent
    d_i_rhs = (
        d_infection
        - d_nu[tf.newaxis, tf.newaxis, :, :] * infectious[:, :, :, None]
        - nu[tf.newaxis, tf.newaxis, :, None] * d_infectious
        + tf.constant(0.5, dtype=p8p.DTYPE) * infectious_neighbor_tangent
    )
    d_rhs = tf.reshape(
        tf.stack([d_s_rhs, d_i_rhs], axis=3),
        tf.concat([tf.shape(state), [len(PARAMETER_NAMES)]], axis=0),
    )
    d_rhs.set_shape(state.shape.as_list() + [len(PARAMETER_NAMES)])
    return rhs, d_rhs


def _sir_transition_mean_jvp_tf(
    points: tf.Tensor,
    d_points: tf.Tensor,
    *,
    kappa: tf.Tensor,
    d_kappa: tf.Tensor,
    nu: tf.Tensor,
    d_nu: tf.Tensor,
    adjacency: tf.Tensor,
    neighbor_degree: tf.Tensor,
    substeps: int,
    step_size: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    state = tf.convert_to_tensor(points, dtype=p8p.DTYPE)
    d_state = tf.convert_to_tensor(d_points, dtype=p8p.DTYPE)
    half = tf.constant(0.5, dtype=p8p.DTYPE)
    sixth = step_size / tf.constant(6.0, dtype=p8p.DTYPE)
    for _ in range(int(substeps)):
        k1, d_k1 = _sir_rhs_jvp_tf(
            state,
            d_state,
            kappa=kappa,
            d_kappa=d_kappa,
            nu=nu,
            d_nu=d_nu,
            adjacency=adjacency,
            neighbor_degree=neighbor_degree,
        )
        k2_input = state + half * step_size * k1
        d_k2_input = d_state + half * step_size * d_k1
        k2, d_k2 = _sir_rhs_jvp_tf(
            k2_input,
            d_k2_input,
            kappa=kappa,
            d_kappa=d_kappa,
            nu=nu,
            d_nu=d_nu,
            adjacency=adjacency,
            neighbor_degree=neighbor_degree,
        )
        k3_input = state + half * step_size * k2
        d_k3_input = d_state + half * step_size * d_k2
        k3, d_k3 = _sir_rhs_jvp_tf(
            k3_input,
            d_k3_input,
            kappa=kappa,
            d_kappa=d_kappa,
            nu=nu,
            d_nu=d_nu,
            adjacency=adjacency,
            neighbor_degree=neighbor_degree,
        )
        k4_input = state + half * step_size * k3
        d_k4_input = d_state + half * step_size * d_k3
        k4, d_k4 = _sir_rhs_jvp_tf(
            k4_input,
            d_k4_input,
            kappa=kappa,
            d_kappa=d_kappa,
            nu=nu,
            d_nu=d_nu,
            adjacency=adjacency,
            neighbor_degree=neighbor_degree,
        )
        state = state + sixth * (
            k1 + tf.constant(2.0, dtype=p8p.DTYPE) * k2 + tf.constant(2.0, dtype=p8p.DTYPE) * k3 + k4
        )
        d_state = d_state + sixth * (
            d_k1 + tf.constant(2.0, dtype=p8p.DTYPE) * d_k2 + tf.constant(2.0, dtype=p8p.DTYPE) * d_k3 + d_k4
        )
    return state, d_state


def _compact_ledh_flow_jvp_tf(
    *,
    pre_flow: tf.Tensor,
    d_pre_flow: tf.Tensor,
    prior_means: tf.Tensor,
    d_prior_means: tf.Tensor,
    observation: tf.Tensor,
    observation_jacobian: tf.Tensor,
    observation_covariance: tf.Tensor,
    d_observation_covariance: tf.Tensor,
    transition_covariance: tf.Tensor,
) -> tuple[p8p.streaming_tf.StreamingLEDHFlowTensors, tf.Tensor, tf.Tensor, tf.Tensor]:
    residual = observation[tf.newaxis, tf.newaxis, :] - tf.gather(
        pre_flow,
        p8p._SIR_INFECTIOUS_INDICES,  # noqa: SLF001
        axis=2,
    )
    d_residual = -tf.gather(
        d_pre_flow,
        p8p._SIR_INFECTIOUS_INDICES,  # noqa: SLF001
        axis=2,
    )
    flow, aux = p8p.core_tf._batched_ledh_linearized_flow_with_aux_tf(  # noqa: SLF001
        pre_flow_particles=pre_flow,
        prior_means=prior_means,
        observation_jacobian=observation_jacobian,
        observation_residual=residual,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
    )
    batch_size, num_particles, state_dim = p8p.core_tf._static_shape(pre_flow, "pre_flow")  # noqa: SLF001
    param_dim = len(PARAMETER_NAMES)
    obs_dim = len(p8p._SIR_INFECTIOUS_INDICES)  # noqa: SLF001
    prior_chol = aux.prior_chol
    prior_precision = aux.prior_precision
    obs_precision = aux.obs_precision
    d_obs_precision = -tf.einsum(
        "bij,bjkp,bkl->bilp",
        obs_precision,
        d_observation_covariance,
        obs_precision,
    )
    d_pseudo_observation = (
        tf.einsum("bnod,bndp->bnop", observation_jacobian, d_pre_flow)
        + d_residual
    )
    d_post_precision = tf.einsum(
        "bnod,boqp,bnqe->bndep",
        observation_jacobian,
        d_obs_precision,
        observation_jacobian,
    )
    post_covariance = aux.post_covariance
    d_post_covariance = -tf.einsum(
        "bnij,bnjkp,bnkl->bnilp",
        post_covariance,
        d_post_precision,
        post_covariance,
    )
    info = aux.info
    d_info = (
        tf.einsum("bde,bnep->bndp", prior_precision, d_prior_means)
        + tf.einsum("bnod,boqp,bnq->bndp", observation_jacobian, d_obs_precision, aux.pseudo_observation)
        + tf.einsum("bnod,boq,bnqp->bndp", observation_jacobian, obs_precision, d_pseudo_observation)
    )
    post_mean = flow.local_posterior_means
    d_post_mean = (
        tf.einsum("bndep,bne->bndp", d_post_covariance, info)
        + tf.einsum("bnde,bnep->bndp", post_covariance, d_info)
    )
    post_chol, d_post_chol = _cholesky_jvp_matrix(post_covariance, d_post_covariance)
    prior_inv = aux.prior_inv
    affine_transform = aux.affine_transform
    d_affine_transform = tf.einsum("bnijp,bjk->bnikp", d_post_chol, prior_inv)
    delta = pre_flow - prior_means
    d_delta = d_pre_flow - d_prior_means
    d_post_flow = (
        d_post_mean
        + tf.einsum("bnijp,bnj->bnip", d_affine_transform, delta)
        + tf.einsum("bnij,bnjp->bnip", affine_transform, d_delta)
    )
    d_logdet_post_chol = tf.reduce_sum(
        tf.einsum("bniip->bnip", d_post_chol)
        / tf.linalg.diag_part(post_chol)[:, :, :, None],
        axis=2,
    )
    d_forward_log_det = d_logdet_post_chol
    _transition_unused = tf.zeros([batch_size, state_dim, state_dim, param_dim], dtype=p8p.DTYPE)
    _obs_unused = tf.zeros([batch_size, obs_dim, obs_dim, param_dim], dtype=p8p.DTYPE)
    del _transition_unused, _obs_unused, num_particles
    _pre_density, d_pre_density = _batched_gaussian_logpdf_jvp(
        pre_flow - prior_means,
        transition_covariance,
        d_pre_flow - d_prior_means,
        tf.zeros([batch_size, state_dim, state_dim, param_dim], dtype=p8p.DTYPE),
    )
    return (
        p8p.streaming_tf.StreamingLEDHFlowTensors(
            post_flow_particles=flow.post_flow_particles,
            pre_flow_log_density=flow.pre_flow_log_density,
            forward_log_det=flow.forward_log_det,
        ),
        d_post_flow,
        d_pre_density,
        d_forward_log_det,
    )


def _compact_value_and_score_from_components(
    args: argparse.Namespace,
    theta_values: Sequence[float] | None = None,
) -> dict[str, tf.Tensor]:
    """Compute fixed-SIR same-scalar score with compact forward sensitivity."""

    _configure_precision(args)
    _require_fixed_sir_score_args(args)
    theta = _theta_values(theta_values)
    tensors, _semantics = _build_fixed_sir_tensors(args)
    scaled = p8p._scaled_parameters(p8p._theta_components(theta))  # noqa: SLF001
    kappa = tf.convert_to_tensor(scaled["kappa"], dtype=p8p.DTYPE)
    nu = tf.convert_to_tensor(scaled["nu"], dtype=p8p.DTYPE)
    observation_covariance = p8p._batch_matrix_parameter(  # noqa: SLF001
        scaled["observation_covariance"],
        len(args.batch_seeds),
    )
    batch_size = len(args.batch_seeds)
    param_dim = len(PARAMETER_NAMES)
    d_kappa = tf.stack(
        [kappa, tf.zeros_like(kappa), tf.zeros_like(kappa)],
        axis=-1,
    )
    d_nu = tf.stack(
        [tf.zeros_like(nu), nu, tf.zeros_like(nu)],
        axis=-1,
    )
    d_observation_covariance = tf.stack(
        [
            tf.zeros_like(observation_covariance),
            tf.zeros_like(observation_covariance),
            tf.constant(2.0, dtype=p8p.DTYPE) * observation_covariance,
        ],
        axis=-1,
    )
    adjacency = tf.cast(p8p._SIR_ADJACENCY_MATRIX, p8p.DTYPE)  # noqa: SLF001
    neighbor_degree = tf.cast(p8p._SIR_NEIGHBOR_DEGREE, p8p.DTYPE)  # noqa: SLF001
    step_size = tf.cast(p8p._SIR_DELTA, p8p.DTYPE) / tf.cast(  # noqa: SLF001
        int(p8p._SIR_RK4_SUBSTEPS),  # noqa: SLF001
        p8p.DTYPE,
    )
    process_chol = tf.linalg.cholesky(tf.cast(tensors["transition_covariance"], p8p.DTYPE))
    observations = tf.cast(tensors["observations"], p8p.DTYPE)
    fixed_resampling_mask = tf.convert_to_tensor(tensors["fixed_resampling_mask"], dtype=tf.bool)
    transition_covariance = tf.cast(tensors["transition_covariance"], p8p.DTYPE)
    transition_noise = tf.cast(tensors["transition_noise"], p8p.DTYPE)
    particles = tf.cast(tensors["initial_particles"], p8p.DTYPE)
    batch_size_static, num_particles, state_dim = p8p.core_tf._static_shape(  # noqa: SLF001
        particles,
        "initial_particles",
    )
    if batch_size_static != batch_size or state_dim != 18:
        raise ValueError("fixed-SIR compact route expected batch/static state dimensions")
    h_jac_full = tf.tile(
        tf.cast(p8p._SIR_INFECTIOUS_SELECTOR, p8p.DTYPE)[tf.newaxis, tf.newaxis, :, :],  # noqa: SLF001
        [batch_size, num_particles, 1, 1],
    )
    d_particles = tf.zeros([batch_size, num_particles, state_dim, param_dim], dtype=p8p.DTYPE)
    log_weights = p8p.core_tf.uniform_log_weights(batch_size, num_particles)
    d_log_weights = tf.zeros([batch_size, num_particles, param_dim], dtype=p8p.DTYPE)
    log_likelihood = tf.zeros([batch_size], dtype=p8p.DTYPE)
    d_log_likelihood = tf.zeros([batch_size, param_dim], dtype=p8p.DTYPE)

    for time_index in range(int(args.time_steps)):
        observation = observations[time_index]
        prior_means, d_prior_means = _sir_transition_mean_jvp_tf(
            particles,
            d_particles,
            kappa=kappa,
            d_kappa=d_kappa,
            nu=nu,
            d_nu=d_nu,
            adjacency=adjacency,
            neighbor_degree=neighbor_degree,
            substeps=int(p8p._SIR_RK4_SUBSTEPS),  # noqa: SLF001
            step_size=step_size,
        )
        noise_tensor = transition_noise[:, time_index, :, :]
        pushed = prior_means + tf.einsum("bnd,bed->bne", noise_tensor, process_chol)
        d_pushed = d_prior_means
        pushed_susceptible = pushed[:, :, 0::2]
        pre_flow = p8p._sir_flatten_components(  # noqa: SLF001
            tf.maximum(pushed_susceptible, tf.constant(0.0, dtype=p8p.DTYPE)),
            pushed[:, :, 1::2],
        )
        d_pre_flow = _sir_flatten_tangent_components(
            tf.where(
                pushed_susceptible[:, :, :, None] > 0.0,
                d_pushed[:, :, 0::2, :],
                tf.zeros_like(d_pushed[:, :, 0::2, :]),
            ),
            d_pushed[:, :, 1::2, :],
        )
        flow, d_post_flow, d_pre_log_density, d_forward_log_det = _compact_ledh_flow_jvp_tf(
            pre_flow=pre_flow,
            d_pre_flow=d_pre_flow,
            prior_means=prior_means,
            d_prior_means=d_prior_means,
            observation=observation,
            observation_jacobian=h_jac_full,
            observation_covariance=observation_covariance,
            d_observation_covariance=d_observation_covariance,
            transition_covariance=transition_covariance,
        )
        post_flow = flow.post_flow_particles
        transition_log_density, d_transition_log_density = _batched_gaussian_logpdf_jvp(
            post_flow - prior_means,
            transition_covariance,
            d_post_flow - d_prior_means,
            tf.zeros([batch_size, state_dim, state_dim, param_dim], dtype=p8p.DTYPE),
        )
        predicted_observation = tf.gather(
            post_flow,
            p8p._SIR_INFECTIOUS_INDICES,  # noqa: SLF001
            axis=2,
        )
        d_predicted_observation = tf.gather(
            d_post_flow,
            p8p._SIR_INFECTIOUS_INDICES,  # noqa: SLF001
            axis=2,
        )
        observation_log_density, d_observation_log_density = _batched_gaussian_logpdf_jvp(
            predicted_observation - observation[tf.newaxis, tf.newaxis, :],
            observation_covariance,
            d_predicted_observation,
            d_observation_covariance,
        )
        corrected_log_weights = (
            log_weights
            + transition_log_density
            + observation_log_density
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        d_corrected_log_weights = (
            d_log_weights
            + d_transition_log_density
            + d_observation_log_density
            - d_pre_log_density
            + d_forward_log_det
        )
        (
            normalized_log_weights,
            d_normalized_log_weights,
            incremental,
            d_incremental,
        ) = _normalize_log_weights_jvp(corrected_log_weights, d_corrected_log_weights)
        mask = fixed_resampling_mask[:, time_index]
        particles, log_weights, d_particles, d_log_weights = _compact_forward_transport_jvp_tf(
            post_flow=post_flow,
            normalized_log_weights=normalized_log_weights,
            d_post_flow=d_post_flow,
            d_normalized_log_weights=d_normalized_log_weights,
            mask=mask,
            args=args,
        )
        log_likelihood = log_likelihood + incremental
        d_log_likelihood = d_log_likelihood + d_incremental

    return {
        "objective": tf.reduce_mean(log_likelihood),
        "log_likelihood": log_likelihood,
        "gradient_tensor": tf.reduce_mean(d_log_likelihood, axis=0),
        "per_seed_gradient": d_log_likelihood,
        "score_route": FIXED_SIR_COMPACT_SCORE_ROUTE_ID,
        "no_autodiff_score_route": True,
        "value_score_route_status": "same_route_value_score",
        "batch_seeds": [int(seed) for seed in args.batch_seeds],
        "time_steps": int(args.time_steps),
        "num_particles": int(args.num_particles),
        "transport": {
            "value_core_mode": "compact_forward_sensitivity_same_scalar",
            "transport_plan_mode": args.transport_plan_mode,
            "transport_ad_mode": args.transport_ad_mode,
            "gradient_mode": args.transport_gradient_mode,
            "row_chunk_size": int(args.row_chunk_size),
            "col_chunk_size": int(args.col_chunk_size),
            "particle_chunk_size": int(args.particle_chunk_size),
            "sinkhorn_iterations": int(args.sinkhorn_iterations),
            "sinkhorn_epsilon": float(args.sinkhorn_epsilon),
            "dense_transport_matrix_materialized": False,
        },
    }


def _fixed_sir_compact_coordinate_fd_diagnostic(
    args: argparse.Namespace,
    theta_values: Sequence[float] | None = None,
    *,
    fd_step: float | None = None,
) -> dict[str, Any]:
    precision = _configure_precision(args)
    theta = tf.constant(_theta_values(theta_values), dtype=p8p.DTYPE)
    step = tf.constant(float(fd_step if fd_step is not None else args.fd_step), dtype=p8p.DTYPE)
    base = _compact_value_and_score_from_components(args, theta.numpy().tolist())
    fd_values = []
    for index in range(len(PARAMETER_NAMES)):
        basis = tf.one_hot(index, len(PARAMETER_NAMES), dtype=p8p.DTYPE)
        plus = _compact_value_and_score_from_components(args, (theta + step * basis).numpy().tolist())
        minus = _compact_value_and_score_from_components(args, (theta - step * basis).numpy().tolist())
        fd_values.append((plus["objective"] - minus["objective"]) / (tf.constant(2.0, dtype=p8p.DTYPE) * step))
    fd_score = tf.stack(fd_values)
    score = tf.convert_to_tensor(base["gradient_tensor"], dtype=p8p.DTYPE)
    abs_error = tf.abs(score - fd_score)
    rel_error = abs_error / tf.maximum(
        tf.maximum(tf.abs(score), tf.abs(fd_score)),
        tf.constant(1.0e-12, dtype=p8p.DTYPE),
    )
    max_abs_error = tf.reduce_max(abs_error)
    max_rel_error = tf.reduce_max(rel_error)
    atol = tf.constant(float(getattr(args, "score_fd_atol", 1.0e-2)), dtype=p8p.DTYPE)
    rtol = tf.constant(float(getattr(args, "score_fd_rtol", 5.0e-2)), dtype=p8p.DTYPE)
    passed = bool((max_abs_error <= atol or max_rel_error <= rtol).numpy())
    return {
        "status": "pass" if passed else "fail",
        "base": base,
        "score": score,
        "fd_score": fd_score,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "per_coordinate_abs_error": abs_error,
        "per_coordinate_rel_error": rel_error,
        "parameter_names": list(PARAMETER_NAMES),
        "fd_step": float(step.numpy()),
        "atol": float(atol.numpy()),
        "rtol": float(rtol.numpy()),
        "score_precision": _score_precision_metadata(precision),
    }


def _fixed_sir_compact_score_artifact_from_diagnostic(
    diagnostic: Mapping[str, Any],
    *,
    source_value_artifact: Mapping[str, Any],
    source_value_artifact_path: str,
    require_all_parameter_correctness: bool = False,
    memory_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize compact fixed-SIR score evidence into the Phase 1 schema."""

    value_core = validate_ledh_forward_scalar_artifact(
        source_value_artifact,
        expected_row_id=FIXED_SIR_AUSTRIA_ROW_ID,
        require_admitted=True,
    )
    base_raw = diagnostic.get("base")
    if not isinstance(base_raw, Mapping):
        raise ValueError("fixed-SIR compact diagnostic must include base mapping")
    base = base_raw
    if base.get("score_route") != FIXED_SIR_COMPACT_SCORE_ROUTE_ID:
        raise ValueError("fixed-SIR compact diagnostic must use compact score route")
    if base.get("no_autodiff_score_route") is not True:
        raise ValueError("fixed-SIR compact diagnostic must declare no_autodiff_score_route")
    if base.get("value_score_route_status") != LEDH_SCORE_VALUE_ROUTE_STATUS_SAME:
        raise ValueError("fixed-SIR compact diagnostic must be same_route_value_score")
    score = tf.convert_to_tensor(base.get("gradient_tensor"), dtype=p8p.DTYPE)
    if tuple(diagnostic.get("parameter_names", ())) != tuple(PARAMETER_NAMES):
        raise ValueError("fixed-SIR compact diagnostic parameter_names must match parameter order")
    memory = dict(memory_diagnostics or {})
    memory_pass = bool(memory.get("n10000_memory_pass") is True)
    if require_all_parameter_correctness:
        if diagnostic.get("status") != "pass":
            raise ValueError("fixed-SIR compact all-parameter correctness status must pass")
        if not memory_pass:
            raise ValueError("fixed-SIR compact full admission requires N=10000 memory pass")
        if int(base.get("num_particles", -1)) != int(value_core["num_particles"]):
            raise ValueError("fixed-SIR compact full admission requires N=10000 diagnostic shape")
        if int(base.get("time_steps", -1)) != int(value_core["time_steps"]):
            raise ValueError("fixed-SIR compact full admission requires full time_steps")
        if tuple(int(seed) for seed in base.get("batch_seeds", ())) != tuple(
            int(seed) for seed in value_core["batch_seeds"]
        ):
            raise ValueError("fixed-SIR compact full admission requires full batch_seeds")
    score_correctness = {
        "kind": "same_scalar_finite_difference",
        "status": str(diagnostic.get("status", "fail")),
        "max_abs_error": float(
            tf.convert_to_tensor(diagnostic["max_abs_error"], dtype=p8p.DTYPE).numpy()
        ),
        "max_rel_error": float(
            tf.convert_to_tensor(diagnostic["max_rel_error"], dtype=p8p.DTYPE).numpy()
        ),
    }
    for optional_key in ("fd_step", "atol", "rtol"):
        if optional_key in diagnostic:
            score_correctness[optional_key] = float(diagnostic[optional_key])

    artifact = {
        "schema_version": LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
        "row_id": FIXED_SIR_AUSTRIA_ROW_ID,
        "source_value_artifact": source_value_artifact_path,
        "score_target_kind": LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
        "target_scalar": LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
        "target_output_tensor_field": LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD,
        "target_observation_policy": value_core["target_observation_policy"],
        "theta_coordinate_system": value_core["theta_coordinate_system"],
        "score_parameter_names": list(PARAMETER_NAMES),
        "score": [float(value) for value in score.numpy().reshape(-1)],
        "score_derivative_provenance": str(
            base.get("score_route", FIXED_SIR_COMPACT_SCORE_ROUTE_ID)
        ),
        "value_score_route_status": LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
        "value_score_same_transport_algorithm": True,
        "no_autodiff_score_route": True,
        "uses_gradient_tape": False,
        "uses_forward_accumulator": False,
        "uses_stopped_partial_derivative": False,
        "score_correctness": score_correctness,
        "score_admission_status": (
            LEDH_SCORE_ADMISSION_STATUS_FULL
            if require_all_parameter_correctness
            else LEDH_SCORE_ADMISSION_STATUS_TINY
            if diagnostic.get("status") == "pass"
            else LEDH_SCORE_ADMISSION_STATUS_BLOCKED_NOT_RUN
        ),
        "score_precision": _score_precision_metadata(
            diagnostic.get("score_precision", {})
        ),
        "memory_diagnostics": memory,
    }
    if require_all_parameter_correctness:
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=source_value_artifact,
            expected_row_id=FIXED_SIR_AUSTRIA_ROW_ID,
            require_admitted=True,
        )
    return artifact


def _fixed_sir_score_artifact_from_memory_result(
    result: Mapping[str, Any],
    *,
    source_value_artifact: Mapping[str, Any],
    source_value_artifact_path: str,
    require_all_parameter_correctness: bool = False,
) -> dict[str, Any]:
    """Normalize fixed-SIR score-memory evidence into the Phase 1 schema."""

    value_core = validate_ledh_forward_scalar_artifact(
        source_value_artifact,
        expected_row_id=FIXED_SIR_AUSTRIA_ROW_ID,
        require_admitted=True,
    )
    row_id = str(result.get("row_id"))
    if row_id != FIXED_SIR_AUSTRIA_ROW_ID:
        raise ValueError("fixed-SIR score row_id must be the main fixed-SIR row")
    if result.get("target_scope", MAIN_OBSERVED_DATA_ROW_SCOPE) != MAIN_OBSERVED_DATA_ROW_SCOPE:
        raise ValueError("fixed-SIR score target_scope must be main observed-data row")
    num_particles = int(result.get("num_particles"))
    if num_particles != int(value_core["num_particles"]):
        raise ValueError("fixed-SIR score num_particles must match admitted value artifact")
    if int(value_core["time_steps"]) != FULL_ROW_TIME_STEPS:
        raise ValueError("fixed-SIR source value artifact time_steps must be 20")
    score_route = str(result.get("score_route"))
    if score_route != FIXED_SIR_MANUAL_SCORE_ROUTE_ID:
        raise ValueError("fixed-SIR score route must be the manual total VJP route")
    score = result.get("score")
    if not isinstance(score, Sequence) or isinstance(score, (str, bytes)):
        raise ValueError("fixed-SIR score must be a sequence")
    if len(score) != len(PARAMETER_NAMES):
        raise ValueError("fixed-SIR score length must match parameter order")
    peak_mib = float(result.get("gpu_memory_peak_mib"))
    budget_mib = float(result.get("memory_budget_mib", 14000.0))
    memory_pass = bool(result.get("primary_pass") is True and peak_mib <= budget_mib)
    if require_all_parameter_correctness:
        raise ValueError(
            "fixed-SIR historical memory/manual score artifacts are diagnostic only; "
            "full admission requires the compact score diagnostic path"
        )
    artifact = {
        "schema_version": LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
        "row_id": FIXED_SIR_AUSTRIA_ROW_ID,
        "source_value_artifact": source_value_artifact_path,
        "score_target_kind": LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
        "target_scalar": LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
        "target_output_tensor_field": LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD,
        "target_observation_policy": value_core["target_observation_policy"],
        "theta_coordinate_system": value_core["theta_coordinate_system"],
        "score_parameter_names": list(PARAMETER_NAMES),
        "score": [float(value) for value in score],
        "score_derivative_provenance": FIXED_SIR_MEMORY_STYLE_SCORE_ROUTE_ID,
        "historical_manual_score_route": FIXED_SIR_MANUAL_SCORE_ROUTE_ID,
        "historical_memory_style_score_route": FIXED_SIR_MEMORY_STYLE_SCORE_ROUTE_ID,
        "value_score_route_status": LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
        "value_score_same_transport_algorithm": True,
        "no_autodiff_score_route": True,
        "uses_gradient_tape": False,
        "uses_forward_accumulator": False,
        "uses_stopped_partial_derivative": False,
        "score_correctness": {
            "kind": "same_scalar_directional_finite_difference",
            "status": "pass" if result.get("primary_pass") is True else "fail",
            "abs_error": float(result.get("abs_error")),
            "rel_error": float(result.get("rel_error")),
        },
        "score_admission_status": (
            LEDH_SCORE_ADMISSION_STATUS_TINY
            if result.get("primary_pass") is True
            else LEDH_SCORE_ADMISSION_STATUS_BLOCKED_NOT_RUN
        ),
        "memory_diagnostics": {
            "n10000_memory_pass": memory_pass,
            "source": "trusted_gpu_score_memory_artifact",
            "peak_mib": peak_mib,
            "budget_mib": budget_mib,
        },
    }
    if require_all_parameter_correctness:
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=source_value_artifact,
            expected_row_id=FIXED_SIR_AUSTRIA_ROW_ID,
            require_admitted=memory_pass,
        )
    return artifact
