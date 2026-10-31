"""TensorFlow finite Sinkhorn relaxed resampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorflow as tf


DTYPE = tf.float64
_CANONICAL_GAUGE_POLICY = "mean_log_u_zero"


@dataclass(frozen=True)
class SinkhornLogStateTF:
    log_u: tf.Tensor
    log_v: tf.Tensor
    gauge_policy: str = "none"


@dataclass(frozen=True)
class SinkhornTFResult:
    particles: tf.Tensor
    coupling: tf.Tensor
    source_weights: tf.Tensor
    target_weights: tf.Tensor
    final_state: SinkhornLogStateTF
    canonicalized_final_state: SinkhornLogStateTF
    diagnostics: dict[str, Any]


def pairwise_squared_euclidean_tf(x: tf.Tensor, y: tf.Tensor | None = None) -> tf.Tensor:
    x = tf.cast(x, DTYPE)
    y = x if y is None else tf.cast(y, DTYPE)
    diff = x[:, None, :] - y[None, :, :]
    return tf.reduce_sum(diff * diff, axis=2)


def build_sinkhorn_log_state_tf(
    log_u: tf.Tensor,
    log_v: tf.Tensor,
    *,
    gauge_policy: str = "none",
) -> SinkhornLogStateTF:
    return SinkhornLogStateTF(
        log_u=tf.reshape(tf.cast(log_u, DTYPE), [-1]),
        log_v=tf.reshape(tf.cast(log_v, DTYPE), [-1]),
        gauge_policy=gauge_policy,
    )


def canonicalize_sinkhorn_log_state_tf(
    state_or_log_u: SinkhornLogStateTF | tf.Tensor,
    log_v: tf.Tensor | None = None,
    *,
    gauge_policy: str = _CANONICAL_GAUGE_POLICY,
) -> SinkhornLogStateTF:
    if gauge_policy != _CANONICAL_GAUGE_POLICY:
        raise ValueError(f"unsupported gauge policy: {gauge_policy}")
    state = _coerce_state(state_or_log_u, log_v=log_v)
    offset = tf.reduce_mean(state.log_u)
    return SinkhornLogStateTF(
        log_u=state.log_u - offset,
        log_v=state.log_v + offset,
        gauge_policy=gauge_policy,
    )


def sinkhorn_coupling_from_log_state_tf(
    state_or_log_u: SinkhornLogStateTF | tf.Tensor,
    *,
    epsilon: float,
    cost: tf.Tensor,
    log_v: tf.Tensor | None = None,
) -> tf.Tensor:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    state = _coerce_state(state_or_log_u, log_v=log_v)
    cost_matrix = tf.cast(cost, DTYPE)
    kernel_log = -cost_matrix / tf.constant(epsilon, dtype=DTYPE)
    return tf.exp(state.log_u[:, None] + kernel_log + state.log_v[None, :])


def sinkhorn_resample_tf(
    particles: tf.Tensor,
    weights: tf.Tensor,
    *,
    epsilon: float = 0.5,
    max_iterations: int = 80,
    tolerance: float = 1e-7,
    cost: tf.Tensor | None = None,
    stabilization: str = "log_domain",
    initial_state: SinkhornLogStateTF | None = None,
    initial_log_u: tf.Tensor | None = None,
    initial_log_v: tf.Tensor | None = None,
) -> SinkhornTFResult:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if stabilization != "log_domain":
        raise ValueError("only log_domain stabilization is implemented")
    if initial_state is not None and (initial_log_u is not None or initial_log_v is not None):
        raise ValueError("pass either initial_state or initial_log_u/initial_log_v, not both")
    if (initial_log_u is None) != (initial_log_v is None):
        raise ValueError("initial_log_u and initial_log_v must be provided together")

    x = tf.cast(particles, DTYPE)
    if len(x.shape) == 1:
        x = x[:, None]
    source = tf.reshape(tf.cast(weights, DTYPE), [-1])
    n = tf.shape(x)[0]
    if int(x.shape[0] or 0) and int(source.shape[0] or 0) and x.shape[0] != source.shape[0]:
        raise ValueError("particles and weights must agree on particle count")
    source_total = tf.reduce_sum(source)
    source = source / source_total
    target = tf.ones([n], dtype=DTYPE) / tf.cast(n, DTYPE)
    cost_matrix = pairwise_squared_euclidean_tf(x) if cost is None else tf.cast(cost, DTYPE)
    log_source = tf.math.log(tf.maximum(source, tf.constant(1e-300, dtype=DTYPE)))
    log_target = tf.math.log(target)
    kernel_log = -cost_matrix / tf.constant(epsilon, dtype=DTYPE)

    provided_initial_state = _resolve_initial_state(
        n=n,
        initial_state=initial_state,
        initial_log_u=initial_log_u,
        initial_log_v=initial_log_v,
    )
    if provided_initial_state is None:
        log_u = tf.zeros([n], dtype=DTYPE)
        log_v = tf.zeros([n], dtype=DTYPE)
        initialization_policy = "zeros"
        initial_state_gauge_policy = "none"
    else:
        log_u = provided_initial_state.log_u
        log_v = provided_initial_state.log_v
        initialization_policy = "provided_log_state"
        initial_state_gauge_policy = provided_initial_state.gauge_policy

    iterations_used = max_iterations

    for iteration in range(1, max_iterations + 1):
        log_u = log_source - tf.reduce_logsumexp(kernel_log + log_v[None, :], axis=1)
        log_v = log_target - tf.reduce_logsumexp(kernel_log + log_u[:, None], axis=0)
        if iteration == max_iterations or iteration % 10 == 0:
            coupling_probe = tf.exp(log_u[:, None] + kernel_log + log_v[None, :])
            row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling_probe, axis=1) - source))
            column_residual = tf.reduce_max(tf.abs(tf.reduce_sum(coupling_probe, axis=0) - target))
            if bool((tf.maximum(row_residual, column_residual) <= tolerance).numpy()):
                iterations_used = iteration
                break

    final_state = build_sinkhorn_log_state_tf(log_u, log_v)
    canonicalized_final_state = canonicalize_sinkhorn_log_state_tf(final_state)
    coupling = sinkhorn_coupling_from_log_state_tf(final_state, epsilon=epsilon, cost=cost_matrix)
    relaxed_particles = _relaxed_particles_from_coupling_tf(coupling, x)
    row_residuals = tf.reduce_sum(coupling, axis=1) - source
    column_residuals = tf.reduce_sum(coupling, axis=0) - target
    diagnostics = {
        "component_id": "finite_sinkhorn_relaxed_resampler_tf",
        "mathematical_object": "finite_budget_entropic_ot_coupling",
        "epsilon": float(epsilon),
        "max_iterations": int(max_iterations),
        "iterations_used": int(iterations_used),
        "tolerance": float(tolerance),
        "stabilization": stabilization,
        "cost_function": "pairwise_squared_euclidean",
        "target_marginal": "uniform",
        "initialization_policy": initialization_policy,
        "initial_state_gauge_policy": initial_state_gauge_policy,
        "final_state_gauge_policy": final_state.gauge_policy,
        "canonicalized_final_state_gauge_policy": canonicalized_final_state.gauge_policy,
        "max_row_residual": _float(tf.reduce_max(tf.abs(row_residuals))),
        "max_column_residual": _float(tf.reduce_max(tf.abs(column_residuals))),
        "total_mass_residual": _float(tf.abs(tf.reduce_sum(coupling) - 1.0)),
        "min_coupling": _float(tf.reduce_min(coupling)),
        "finite_coupling": bool(tf.reduce_all(tf.math.is_finite(coupling)).numpy()),
        "finite_particles": bool(tf.reduce_all(tf.math.is_finite(relaxed_particles)).numpy()),
        "resampling_status": "relaxed_finite_sinkhorn_not_categorical",
        "backend": "tensorflow",
    }
    if diagnostics["max_row_residual"] > tolerance * 10.0:
        raise FloatingPointError("Sinkhorn row residual exceeded tolerance envelope")
    if diagnostics["max_column_residual"] > tolerance * 10.0:
        raise FloatingPointError("Sinkhorn column residual exceeded tolerance envelope")
    if diagnostics["min_coupling"] < -1e-12:
        raise FloatingPointError("Sinkhorn coupling has negative entries")
    if not diagnostics["finite_coupling"] or not diagnostics["finite_particles"]:
        raise FloatingPointError("Sinkhorn emitted non-finite values")
    return SinkhornTFResult(
        particles=relaxed_particles,
        coupling=coupling,
        source_weights=source,
        target_weights=target,
        final_state=final_state,
        canonicalized_final_state=canonicalized_final_state,
        diagnostics=diagnostics,
    )


def _resolve_initial_state(
    *,
    n: tf.Tensor,
    initial_state: SinkhornLogStateTF | None,
    initial_log_u: tf.Tensor | None,
    initial_log_v: tf.Tensor | None,
) -> SinkhornLogStateTF | None:
    if initial_state is not None:
        state = build_sinkhorn_log_state_tf(
            initial_state.log_u,
            initial_state.log_v,
            gauge_policy=initial_state.gauge_policy,
        )
    elif initial_log_u is not None and initial_log_v is not None:
        state = build_sinkhorn_log_state_tf(initial_log_u, initial_log_v)
    else:
        return None
    if bool((tf.size(state.log_u) != n).numpy()) or bool((tf.size(state.log_v) != n).numpy()):
        raise ValueError("initial Sinkhorn state must match particle count")
    return state


def _coerce_state(
    state_or_log_u: SinkhornLogStateTF | tf.Tensor,
    *,
    log_v: tf.Tensor | None,
) -> SinkhornLogStateTF:
    if isinstance(state_or_log_u, SinkhornLogStateTF):
        return build_sinkhorn_log_state_tf(
            state_or_log_u.log_u,
            state_or_log_u.log_v,
            gauge_policy=state_or_log_u.gauge_policy,
        )
    if log_v is None:
        raise ValueError("log_v is required when passing raw log_u")
    return build_sinkhorn_log_state_tf(state_or_log_u, log_v, gauge_policy="none")


def _relaxed_particles_from_coupling_tf(coupling: tf.Tensor, particles: tf.Tensor) -> tf.Tensor:
    column_mass = tf.reduce_sum(coupling, axis=0)
    safe_column_mass = tf.maximum(column_mass, tf.constant(1e-300, dtype=DTYPE))
    return tf.linalg.matmul(coupling, particles, transpose_a=True) / safe_column_mass[:, None]


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())
