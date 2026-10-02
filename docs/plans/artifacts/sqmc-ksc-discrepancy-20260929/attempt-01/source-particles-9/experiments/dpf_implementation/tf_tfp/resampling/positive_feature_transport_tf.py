"""Experimental TensorFlow positive-feature transport.

This Phase 5 module is a semantic-replacement candidate.  The positive-feature
kernel is treated as the transport kernel being tested; dense-reference errors
are explanatory unless a separate approximation contract is written.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorflow as tf


DEFAULT_DTYPE = tf.float64
DTYPE = DEFAULT_DTYPE


@dataclass(frozen=True)
class PositiveFeatureTransportTFResult:
    particles: tf.Tensor
    log_weights: tf.Tensor
    transport_matrix: tf.Tensor
    left_features: tf.Tensor
    right_features: tf.Tensor
    scaling_u: tf.Tensor
    scaling_v: tf.Tensor
    diagnostics: dict[str, Any]


def positive_feature_transport_resample_tf(
    particles: tf.Tensor,
    log_weights: tf.Tensor,
    *,
    num_features: int,
    epsilon: float = 0.5,
    max_iterations: int = 120,
    convergence_threshold: float = 1.0e-4,
    denominator_floor: float = 1.0e-30,
) -> PositiveFeatureTransportTFResult:
    """Apply a deterministic positive-feature semantic-replacement transport."""

    if num_features <= 0:
        raise ValueError("num_features must be positive")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
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

    batch_size = tf.shape(x)[0]
    num_particles = tf.shape(x)[1]
    centered = x - tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
    scale = _filterflow_scale(x)
    scaled_x = centered / tf.stop_gradient(scale[:, None, None])
    basis = _deterministic_feature_basis(tf.shape(x)[2], num_features)
    features = _positive_feature_map(scaled_x, basis, epsilon=tf.constant(epsilon, DTYPE))
    source_weights = tf.exp(logw)
    float_n = tf.cast(num_particles, DTYPE)
    row_target = tf.ones([batch_size, num_particles], dtype=DTYPE)
    column_target = source_weights * float_n
    scaling_u, scaling_v, iterations_used, scale_diag = _sinkhorn_scale_features(
        features,
        features,
        row_target,
        column_target,
        max_iterations=max_iterations,
        convergence_threshold=tf.constant(convergence_threshold, DTYPE),
        denominator_floor=tf.constant(denominator_floor, DTYPE),
    )
    transported = scaling_u[:, :, None] * _feature_matmul(
        features,
        features,
        scaling_v[:, :, None] * x,
    )
    row_mass = scaling_u * _feature_matvec(features, features, scaling_v)
    column_mass = scaling_v * _feature_matvec(features, features, scaling_u)
    row_residual = tf.reduce_max(tf.abs(row_mass - row_target))
    column_residual = tf.reduce_max(tf.abs(column_mass - column_target))
    uniform_log = tf.fill([batch_size, num_particles], -tf.math.log(float_n))
    transport_matrix = tf.zeros([batch_size, 0, 0], dtype=DTYPE)
    finite_features = (
        tf.reduce_all(tf.math.is_finite(features))
        & tf.reduce_all(tf.math.is_finite(scaling_u))
        & tf.reduce_all(tf.math.is_finite(scaling_v))
    )
    positive_features = tf.reduce_all(features > 0.0)
    finite_particles = tf.reduce_all(tf.math.is_finite(transported))
    diagnostics = {
        "component_id": "positive_feature_transport_tf",
        "mathematical_object": "positive_feature_semantic_replacement_transport",
        "source_status": "source_locked",
        "semantic_class": "semantic_replacement",
        "source_route": "fixed_hmc_adaptation",
        "source_route_components": {
            "positive_feature_factorization": "source_faithful",
            "linear_feature_scaling": "source_faithful",
            "filterflow_cost_scaling_adapter": "fixed_hmc_adaptation",
            "deterministic_feature_basis": "fixed_hmc_adaptation",
        },
        "backend": "tensorflow",
        "transport_object_kind": "kernel_factors",
        "transport_matrix_materialized": False,
        "not_materialized_reason": "positive_feature_factors_nonmaterialized",
        "orientation": "source_rows_target_columns",
        "num_features": int(num_features),
        "semantic_replacement_reason": "deterministic positive features define a feature kernel instead of proving dense Gibbs approximation",
        "feature_rule": "deterministic_sinusoidal_positive_features",
        "epsilon": float(epsilon),
        "max_iterations": int(max_iterations),
        "iterations_used": int(iterations_used.numpy()),
        "convergence_threshold": float(convergence_threshold),
        "denominator_floor": float(denominator_floor),
        "factor_shapes": {
            "left_features": features.shape.as_list(),
            "right_features": features.shape.as_list(),
            "scaling_u": scaling_u.shape.as_list(),
            "scaling_v": scaling_v.shape.as_list(),
        },
        "max_row_residual": _float(row_residual),
        "max_column_residual": _float(column_residual),
        "min_kernel_denominator": _float(scale_diag["min_denominator"]),
        "denominator_floor_hits": _float(scale_diag["floor_hits"]),
        "min_feature_value": _float(tf.reduce_min(features)),
        "finite_features": bool(finite_features.numpy()),
        "positive_features": bool(positive_features.numpy()),
        "finite_particles": bool(finite_particles.numpy()),
        "nonclaims": [
            "experimental Phase 5 semantic-replacement candidate only",
            "no dense Gibbs equivalence claim",
            "no speedup claim",
            "no ranking claim",
            "no production default change",
            "no posterior correctness claim",
        ],
    }
    if not diagnostics["finite_features"] or not diagnostics["finite_particles"]:
        raise FloatingPointError("positive-feature transport emitted non-finite values")
    if not diagnostics["positive_features"]:
        raise FloatingPointError("positive-feature map emitted non-positive values")
    result_particles = transported[0] if original_particle_rank == 2 else transported
    result_log_weights = uniform_log[0] if original_weight_rank == 1 else uniform_log
    result_transport = transport_matrix[0] if original_particle_rank == 2 else transport_matrix
    result_features = features[0] if original_particle_rank == 2 else features
    result_u = scaling_u[0] if original_particle_rank == 2 else scaling_u
    result_v = scaling_v[0] if original_particle_rank == 2 else scaling_v
    return PositiveFeatureTransportTFResult(
        particles=result_particles,
        log_weights=result_log_weights,
        transport_matrix=result_transport,
        left_features=result_features,
        right_features=result_features,
        scaling_u=result_u,
        scaling_v=result_v,
        diagnostics=diagnostics,
    )


def _deterministic_feature_basis(state_dim: tf.Tensor, num_features: int) -> tf.Tensor:
    dim = tf.cast(state_dim, tf.int32)
    feature_ids = tf.cast(tf.range(num_features), DTYPE)[:, None]
    coord_ids = tf.cast(tf.range(dim), DTYPE)[None, :]
    basis = tf.sin((feature_ids + 1.0) * (coord_ids + 1.0) * 0.37)
    basis += tf.cos((feature_ids + 2.0) * (coord_ids + 1.0) * 0.19)
    norms = tf.maximum(tf.norm(basis, axis=1, keepdims=True), tf.constant(1.0e-12, DTYPE))
    return basis / norms


def _positive_feature_map(x: tf.Tensor, basis: tf.Tensor, *, epsilon: tf.Tensor) -> tf.Tensor:
    projections = tf.einsum("bnd,rd->bnr", x, basis)
    squared_norm = tf.reduce_sum(x * x, axis=2, keepdims=True)
    logits = projections / tf.sqrt(epsilon) - 0.25 * squared_norm / epsilon
    logits = logits - tf.reduce_max(logits, axis=2, keepdims=True)
    features = tf.exp(logits) / tf.sqrt(tf.cast(tf.shape(basis)[0], DTYPE))
    return tf.maximum(features, tf.constant(1.0e-300, DTYPE))


def _sinkhorn_scale_features(
    left_features: tf.Tensor,
    right_features: tf.Tensor,
    row_target: tf.Tensor,
    column_target: tf.Tensor,
    *,
    max_iterations: int,
    convergence_threshold: tf.Tensor,
    denominator_floor: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
    u = tf.ones_like(row_target)
    v = tf.ones_like(column_target)
    min_denominator = tf.constant(float("inf"), dtype=DTYPE)
    floor_hits = tf.constant(0.0, dtype=DTYPE)
    iterations_used = tf.constant(max_iterations, dtype=tf.int32)
    for iteration in range(1, max_iterations + 1):
        raw_kv = _feature_matvec(left_features, right_features, v)
        min_denominator = tf.minimum(min_denominator, tf.reduce_min(raw_kv))
        floor_hits += tf.reduce_sum(tf.cast(raw_kv <= denominator_floor, DTYPE))
        u = row_target / tf.maximum(raw_kv, denominator_floor)
        raw_ktu = _feature_matvec(right_features, left_features, u)
        min_denominator = tf.minimum(min_denominator, tf.reduce_min(raw_ktu))
        floor_hits += tf.reduce_sum(tf.cast(raw_ktu <= denominator_floor, DTYPE))
        v = column_target / tf.maximum(raw_ktu, denominator_floor)
        row_residual = tf.reduce_max(tf.abs(u * _feature_matvec(left_features, right_features, v) - row_target))
        column_residual = tf.reduce_max(
            tf.abs(v * _feature_matvec(right_features, left_features, u) - column_target)
        )
        if bool((tf.maximum(row_residual, column_residual) <= convergence_threshold).numpy()):
            iterations_used = tf.constant(iteration, dtype=tf.int32)
            break
    return u, v, iterations_used, {
        "min_denominator": min_denominator,
        "floor_hits": floor_hits,
    }


def _feature_matvec(left_features: tf.Tensor, right_features: tf.Tensor, vector: tf.Tensor) -> tf.Tensor:
    projected = tf.einsum("bnr,bn->br", right_features, vector)
    return tf.einsum("bnr,br->bn", left_features, projected)


def _feature_matmul(left_features: tf.Tensor, right_features: tf.Tensor, matrix: tf.Tensor) -> tf.Tensor:
    projected = tf.einsum("bnr,bnd->brd", right_features, matrix)
    return tf.einsum("bnr,brd->bnd", left_features, projected)


def _filterflow_scale(x: tf.Tensor) -> tf.Tensor:
    std = tf.math.reduce_std(tf.cast(x, DTYPE), axis=1)
    diameter = tf.reduce_max(std, axis=1)
    diameter = tf.where(diameter == 0.0, tf.ones_like(diameter), diameter)
    dimension = tf.cast(tf.shape(x)[2], DTYPE)
    return diameter * tf.sqrt(dimension)


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())
