"""TensorFlow finite Sinkhorn relaxed resampling."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple

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


class SinkhornTensors(NamedTuple):
    particles: tf.Tensor
    coupling: tf.Tensor
    source: tf.Tensor
    target: tf.Tensor
    log_u: tf.Tensor
    log_v: tf.Tensor
    iterations_used: tf.Tensor
    row_residual: tf.Tensor
    column_residual: tf.Tensor
    mass_residual: tf.Tensor
    min_coupling: tf.Tensor
    finite_coupling: tf.Tensor
    finite_particles: tf.Tensor
    valid: tf.Tensor


def sinkhorn_resample_core_tf(x, weights, cost, log_u, log_v, *, epsilon,
                             max_iterations, tolerance):
    """Native finite iteration with the original ten-step stopping cadence.

    The enclosing filter must propagate ``valid`` to its host veto. This core
    contains no host decisions and can be compiled with the complete filter.
    """
    source = weights / tf.reduce_sum(weights)
    target = tf.ones_like(source) / tf.cast(tf.shape(x)[0], DTYPE)
    log_source = tf.math.log(tf.maximum(source, tf.constant(1e-300, DTYPE)))
    log_target = tf.math.log(target)
    kernel_log = -cost / tf.convert_to_tensor(epsilon, DTYPE)

    def body(iteration, log_u, log_v, active):
        del active
        log_u = log_source - tf.reduce_logsumexp(kernel_log + log_v[None, :], axis=1)
        log_v = log_target - tf.reduce_logsumexp(kernel_log + log_u[:, None], axis=0)
        iteration += 1

        def converged():
            coupling = tf.exp(log_u[:, None] + kernel_log + log_v[None, :])
            row_error = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=1) - source))
            col_error = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=0) - target))
            return tf.maximum(row_error, col_error) <= tolerance

        stopped = tf.cond((iteration == max_iterations) | (iteration % 10 == 0),
                          converged, lambda: tf.constant(False))
        return iteration, log_u, log_v, ~stopped

    iterations, log_u, log_v, _ = tf.while_loop(
        lambda i, _u, _v, active: (i < max_iterations) & active, body,
        (tf.constant(0), log_u, log_v, tf.constant(True)),
        maximum_iterations=max_iterations, parallel_iterations=1)
    coupling = tf.exp(log_u[:, None] + kernel_log + log_v[None, :])
    particles = _relaxed_particles_from_coupling_tf(coupling, x)
    row_error = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=1) - source))
    col_error = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=0) - target))
    mass_error = tf.abs(tf.reduce_sum(coupling) - 1.0)
    minimum = tf.reduce_min(coupling)
    finite_coupling = tf.reduce_all(tf.math.is_finite(coupling))
    finite_particles = tf.reduce_all(tf.math.is_finite(particles))
    valid = ((row_error <= tolerance * 10.0) & (col_error <= tolerance * 10.0)
             & (minimum >= -1e-12) & finite_coupling & finite_particles)
    return SinkhornTensors(particles, coupling, source, target, log_u, log_v,
        iterations, row_error, col_error, mass_error, minimum,
        finite_coupling, finite_particles, valid)


@lru_cache(maxsize=8)
def make_sinkhorn_resample_tf(particle_spec, *, epsilon=0.5, max_iterations=80,
                             tolerance=1e-7, supplied_cost=False, jit_compile=True):
    """Bind a complete resampling calculation to a stable tensor signature."""
    count = particle_spec.shape[0]
    if count is None or count < 1:
        raise ValueError("Sinkhorn requires a static positive particle count")
    signature = [particle_spec, tf.TensorSpec([count], DTYPE),
                 tf.TensorSpec([count, count] if supplied_cost else [0, 0], DTYPE), tf.TensorSpec([count], DTYPE),
                 tf.TensorSpec([count], DTYPE)]

    def evaluate(x, weights, cost, log_u, log_v):
        if not supplied_cost:
            cost = pairwise_squared_euclidean_tf(x)
        return sinkhorn_resample_core_tf(x, weights, cost, log_u, log_v,
            epsilon=epsilon, max_iterations=max_iterations, tolerance=tolerance)

    forward = tf.function(evaluate, input_signature=signature,
                          jit_compile=jit_compile, autograph=False)
    # A tape outside an XLA call cannot receive the TensorLists of its native
    # loop. Save only explicit inputs and recompute the same finite iteration
    # inside the compiled VJP. This is diagnostic AD, not an analytical LEDH
    # score. The stopping predicate and all differentiable outputs are retained.
    differentiable_indices = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10)
    result_signature = (
        particle_spec, tf.TensorSpec([count, count], DTYPE),
        tf.TensorSpec([count], DTYPE), tf.TensorSpec([count], DTYPE),
        tf.TensorSpec([count], DTYPE), tf.TensorSpec([count], DTYPE),
        tf.TensorSpec([], DTYPE), tf.TensorSpec([], DTYPE),
        tf.TensorSpec([], DTYPE), tf.TensorSpec([], DTYPE),
    )

    @tf.function(input_signature=[*signature, *result_signature],
                 jit_compile=jit_compile, autograph=False)
    def backward(x, weights, cost, log_u, log_v, *cotangents):
        inputs = (x, weights, cost, log_u, log_v)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            result = evaluate(*inputs)
            outputs = tuple(result[index] for index in differentiable_indices)
        return tape.gradient(outputs, inputs, output_gradients=cotangents,
                             unconnected_gradients=tf.UnconnectedGradients.ZERO)

    @tf.custom_gradient
    def differentiable_evaluate(x, weights, cost, log_u, log_v):
        inputs = (x, weights, cost, log_u, log_v)
        result = forward(*tf.nest.map_structure(tf.stop_gradient, inputs))

        def vjp(*cotangents):
            upstream = tuple(
                tf.zeros_like(result[index]) if cotangents[index] is None
                else cotangents[index] for index in differentiable_indices
            )
            return backward(*inputs, *upstream)

        return result, vjp

    return tf.function(differentiable_evaluate, input_signature=signature,
                       jit_compile=jit_compile, autograph=False)


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
    if not tf.is_tensor(epsilon) and epsilon <= 0.0:
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
    jit_compile: bool = True,
) -> SinkhornTFResult:
    if not tf.is_tensor(epsilon) and epsilon <= 0.0:
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
    n = x.shape[0]
    if n is None or n < 1:
        raise ValueError("Sinkhorn requires a static positive particle count")
    if int(x.shape[0] or 0) and int(source.shape[0] or 0) and x.shape[0] != source.shape[0]:
        raise ValueError("particles and weights must agree on particle count")
    cost_input = tf.zeros([0, 0], DTYPE) if cost is None else tf.cast(cost, DTYPE)

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

    if tf.inside_function():
        result = sinkhorn_resample_core_tf(x, source,
            pairwise_squared_euclidean_tf(x) if cost is None else cost_input,
            log_u, log_v, epsilon=epsilon, max_iterations=max_iterations,
            tolerance=tolerance)
    else:
        result = make_sinkhorn_resample_tf(tf.TensorSpec(x.shape, x.dtype),
            epsilon=epsilon, max_iterations=max_iterations, tolerance=tolerance,
            supplied_cost=cost is not None, jit_compile=jit_compile)(x, source, cost_input, log_u, log_v)
    final_state = build_sinkhorn_log_state_tf(result.log_u, result.log_v)
    canonicalized_final_state = canonicalize_sinkhorn_log_state_tf(final_state)
    diagnostics = {
        "component_id": "finite_sinkhorn_relaxed_resampler_tf",
        "mathematical_object": "finite_budget_entropic_ot_coupling",
        "epsilon": _report_scalar(epsilon),
        "max_iterations": int(max_iterations),
        "iterations_used": _report_scalar(result.iterations_used),
        "tolerance": float(tolerance),
        "stabilization": stabilization,
        "cost_function": "pairwise_squared_euclidean",
        "target_marginal": "uniform",
        "initialization_policy": initialization_policy,
        "initial_state_gauge_policy": initial_state_gauge_policy,
        "final_state_gauge_policy": final_state.gauge_policy,
        "canonicalized_final_state_gauge_policy": canonicalized_final_state.gauge_policy,
        "max_row_residual": _float(result.row_residual),
        "max_column_residual": _float(result.column_residual),
        "total_mass_residual": _float(result.mass_residual),
        "min_coupling": _float(result.min_coupling),
        "finite_coupling": _report_scalar(result.finite_coupling),
        "finite_particles": _report_scalar(result.finite_particles),
        "resampling_status": "relaxed_finite_sinkhorn_not_categorical",
        "backend": "tensorflow",
        "jit_compile": bool(jit_compile),
        "valid": _report_scalar(result.valid),
    }
    if tf.executing_eagerly():
        if diagnostics["max_row_residual"] > tolerance * 10.0:
            raise FloatingPointError("Sinkhorn row residual exceeded tolerance envelope")
        if diagnostics["max_column_residual"] > tolerance * 10.0:
            raise FloatingPointError("Sinkhorn column residual exceeded tolerance envelope")
        if diagnostics["min_coupling"] < -1e-12:
            raise FloatingPointError("Sinkhorn coupling has negative entries")
        if not diagnostics["finite_coupling"] or not diagnostics["finite_particles"]:
            raise FloatingPointError("Sinkhorn emitted non-finite values")
    return SinkhornTFResult(
        particles=result.particles,
        coupling=result.coupling,
        source_weights=result.source,
        target_weights=result.target,
        final_state=final_state,
        canonicalized_final_state=canonicalized_final_state,
        diagnostics=diagnostics,
    )


def _resolve_initial_state(
    *,
    n: int,
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
    if state.log_u.shape != (n,) or state.log_v.shape != (n,):
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
    return _report_scalar(tf.cast(value, DTYPE))


def _report_scalar(value):
    """Materialize only an eager report; compiled consumers must check valid."""
    if not tf.is_tensor(value):
        return value
    return value.numpy().item() if tf.executing_eagerly() else value
