"""Generic frozen deterministic-mixture importance and control-variate kernels.

This module is deliberately model independent.  A caller supplies the exact
nonnegative target values, normalized component log densities, and the declared
row masses.  The complete mixture (all components, not the selected component)
is used in every denominator.  The optional control variate is a nonnegative
function with a known integral, such as a squared TT whose Gram integral has
already been computed.

The directional routine is an explicit tangent calculation.  It is not a
``GradientTape`` or a pfor implementation: callers supply total directional
values, including pathwise map/proposal terms when those are part of the
declared program.  Proposal rows, masses, mixture weights, and discrete choices
are frozen by the surrounding route.  The fixed-shape factories provide the repeated
TensorFlow graph used by candidate routes; XLA is enabled by default and can
be disabled only for a documented diagnostic/reference run.

Classification: generic candidate/diagnostic route.  This module does not
change any model's production default and does not claim universal finite
variance or exact posterior inference.
"""

from __future__ import annotations

from typing import Mapping

import tensorflow as tf


DTYPE = tf.float64
ROUTE_ID = "generic_frozen_complete_dmis_control_variate_v1"
ROUTE_CLASSIFICATION = "candidate_diagnostic_only"


def _as_float64(value: tf.Tensor, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, dtype=DTYPE)
    if tensor.dtype != DTYPE:
        raise TypeError(f"{name} must use float64")
    return tensor


def validate_mixture_inputs(
    component_weights: tf.Tensor,
    base_masses: tf.Tensor,
    *,
    tolerance: float = 1.0e-12,
) -> None:
    """Fail closed unless component weights and row masses are valid."""

    alpha = _as_float64(component_weights, "component_weights")
    masses = _as_float64(base_masses, "base_masses")
    if alpha.shape.rank != 1 or masses.shape.rank != 1:
        raise ValueError("component_weights and base_masses must be vectors")
    tf.debugging.assert_all_finite(alpha, "component_weights must be finite")
    tf.debugging.assert_all_finite(masses, "base_masses must be finite")
    tf.debugging.assert_positive(alpha, "every mixture component weight must be positive")
    tf.debugging.assert_positive(masses, "every deterministic row mass must be positive")
    tf.debugging.assert_near(
        tf.reduce_sum(alpha),
        tf.constant(1.0, DTYPE),
        atol=tf.constant(tolerance, DTYPE),
        rtol=tf.constant(tolerance, DTYPE),
        message="component weights must sum to one",
    )
    tf.debugging.assert_near(
        tf.reduce_sum(masses),
        tf.constant(1.0, DTYPE),
        atol=tf.constant(tolerance, DTYPE),
        rtol=tf.constant(tolerance, DTYPE),
        message="base masses must sum to one",
    )


def complete_mixture_log_density(
    component_log_densities: tf.Tensor,
    component_weights: tf.Tensor,
) -> tf.Tensor:
    """Return ``log(sum_j alpha_j q_j)`` for rows shaped ``[N, J]``.

    Components with zero density at a row may be represented by ``-inf``.  The
    mixture remains valid when at least one component has finite density.
    """

    logs = _as_float64(component_log_densities, "component_log_densities")
    alpha = _as_float64(component_weights, "component_weights")
    if logs.shape.rank != 2 or alpha.shape.rank != 1:
        raise ValueError("component_log_densities must be [N,J] and weights [J]")
    if logs.shape[1] is not None and alpha.shape[0] is not None:
        if int(logs.shape[1]) != int(alpha.shape[0]):
            raise ValueError("component count mismatch")
    return tf.reduce_logsumexp(
        logs + tf.math.log(alpha)[tf.newaxis, :], axis=1
    )


def _safe_responsibilities(
    component_log_densities: tf.Tensor,
    component_weights: tf.Tensor,
    mixture_log_density: tf.Tensor,
) -> tf.Tensor:
    logits = component_log_densities + tf.math.log(component_weights)[None, :]
    finite_logits = tf.math.is_finite(logits)
    raw = tf.where(
        finite_logits,
        tf.exp(logits - mixture_log_density[:, None]),
        tf.zeros_like(logits),
    )
    denominator = tf.reduce_sum(raw, axis=1, keepdims=True)
    return tf.math.divide_no_nan(raw, denominator)


def _masked_inverse_density(mixture_log_density: tf.Tensor) -> tf.Tensor:
    """Return a finite inverse where possible and zero on zero-density rows."""

    finite = tf.math.is_finite(mixture_log_density)
    return tf.where(
        finite,
        tf.exp(-mixture_log_density),
        tf.zeros_like(mixture_log_density),
    )


def _validate_value_shapes(
    target_values: tf.Tensor,
    component_log_densities: tf.Tensor,
    component_weights: tf.Tensor,
    base_masses: tf.Tensor,
    control_values: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    target = _as_float64(target_values, "target_values")
    logs = _as_float64(component_log_densities, "component_log_densities")
    alpha = _as_float64(component_weights, "component_weights")
    masses = _as_float64(base_masses, "base_masses")
    control = _as_float64(control_values, "control_values")
    if target.shape.rank != 1 or control.shape.rank != 1:
        raise ValueError("target_values and control_values must be vectors")
    if logs.shape.rank != 2:
        raise ValueError("component_log_densities must be a matrix")
    if target.shape[0] is not None and logs.shape[0] is not None:
        if int(target.shape[0]) != int(logs.shape[0]):
            raise ValueError("row count mismatch")
    if target.shape[0] is not None and control.shape[0] is not None:
        if int(target.shape[0]) != int(control.shape[0]):
            raise ValueError("control row count mismatch")
    validate_mixture_inputs(alpha, masses)
    tf.debugging.assert_all_finite(target, "target values must be finite")
    tf.debugging.assert_all_finite(control, "control values must be finite")
    tf.debugging.assert_greater_equal(target, tf.zeros_like(target))
    tf.debugging.assert_greater_equal(control, tf.zeros_like(control))
    # A component may be zero at a row, but NaNs and positive infinities are
    # never valid log densities.  ``-inf`` is handled by the mixture routine.
    tf.debugging.assert_equal(
        tf.reduce_all(~(tf.math.is_nan(logs) | tf.math.is_inf(logs) & (logs > 0.0))),
        tf.constant(True),
        message="component log densities may contain -inf but not NaN/+inf",
    )
    return target, logs, alpha, masses, control


def _value_core(
    target_values: tf.Tensor,
    component_log_densities: tf.Tensor,
    component_weights: tf.Tensor,
    base_masses: tf.Tensor,
    control_values: tf.Tensor,
    control_normalizer: tf.Tensor,
    *,
    validate: bool,
) -> Mapping[str, tf.Tensor]:
    if validate:
        target, logs, alpha, masses, control = _validate_value_shapes(
            target_values,
            component_log_densities,
            component_weights,
            base_masses,
            control_values,
        )
    else:
        target = tf.convert_to_tensor(target_values, DTYPE)
        logs = tf.convert_to_tensor(component_log_densities, DTYPE)
        alpha = tf.convert_to_tensor(component_weights, DTYPE)
        masses = tf.convert_to_tensor(base_masses, DTYPE)
        control = tf.convert_to_tensor(control_values, DTYPE)
    z_control = tf.reshape(tf.convert_to_tensor(control_normalizer, DTYPE), [])
    if validate:
        tf.debugging.assert_all_finite(
            z_control, "control normalizer must be finite"
        )
        tf.debugging.assert_greater_equal(
            z_control, tf.constant(0.0, DTYPE),
            message="control normalizer must be nonnegative",
        )

    log_q = complete_mixture_log_density(logs, alpha)
    # A zero proposal density is represented by ``-inf``.  Do not evaluate
    # ``0 * exp(+inf)``: mask the ratio first, then fail the validity flag if
    # a nonzero target/residual needs that unsupported row.
    log_q_finite = tf.math.is_finite(log_q)
    inverse_q = _masked_inverse_density(log_q)
    residual = target - control
    target_weights = target * inverse_q
    residual_weights = residual * inverse_q
    dmis_normalizer = tf.reduce_sum(masses * target_weights)
    normalizer = z_control + tf.reduce_sum(masses * residual_weights)

    weighted_target = masses * target_weights
    safe_dmis = tf.where(
        dmis_normalizer > 0.0,
        dmis_normalizer,
        tf.constant(1.0, DTYPE),
    )
    normalized_target_weights = weighted_target / safe_dmis
    ess_denominator = tf.reduce_sum(tf.square(normalized_target_weights))
    target_ess = tf.math.divide_no_nan(
        tf.constant(1.0, DTYPE), ess_denominator
    )
    target_max_weight = tf.reduce_max(normalized_target_weights)
    residual_second_moment = tf.reduce_sum(
        masses * tf.square(residual_weights)
    )
    finite = tf.reduce_all(
        tf.math.is_finite(
            tf.concat(
                [
                    tf.where(
                        tf.math.is_inf(log_q) & (log_q < 0.0),
                        tf.zeros_like(log_q),
                        log_q,
                    ),
                    target_weights,
                    residual_weights,
                    tf.reshape(dmis_normalizer, [1]),
                    tf.reshape(normalizer, [1]),
                    tf.reshape(residual_second_moment, [1]),
                ],
                axis=0,
            )
        )
    )
    target_support_valid = tf.reduce_all(
        tf.where(target > 0.0, log_q_finite, tf.ones_like(log_q_finite))
    )
    support_needed = tf.not_equal(residual, 0.0)
    residual_support_valid = tf.reduce_all(
        tf.where(support_needed, log_q_finite, tf.ones_like(log_q_finite))
    )
    normalizer_positive = normalizer > 0.0
    log_normalizer = tf.math.log(tf.maximum(normalizer, tf.constant(1.0e-300, DTYPE)))
    return {
        "normalizer": normalizer,
        "log_normalizer": log_normalizer,
        "dmis_normalizer": dmis_normalizer,
        "complete_mixture_log_density": log_q,
        "target_weights": target_weights,
        "residual_weights": residual_weights,
        "normalized_target_weights": normalized_target_weights,
        "target_effective_sample_size": target_ess,
        "target_effective_sample_size_fraction": target_ess
        / tf.cast(tf.shape(target)[0], DTYPE),
        "maximum_normalized_target_weight": target_max_weight,
        "residual_second_moment": residual_second_moment,
        "residual_mean_estimate": tf.reduce_sum(masses * residual_weights),
        "finite": finite,
        # Plain DMIS needs target support; the residual control-variate
        # correction only needs support where its residual is nonzero.
        "target_support_valid": target_support_valid,
        "residual_support_valid": residual_support_valid,
        "support_valid": residual_support_valid,
        "dmis_valid": finite & target_support_valid & (dmis_normalizer > 0.0),
        "normalizer_positive": normalizer_positive,
        "valid": finite & residual_support_valid & normalizer_positive,
    }


def frozen_importance_estimate(
    target_values: tf.Tensor,
    component_log_densities: tf.Tensor,
    component_weights: tf.Tensor,
    base_masses: tf.Tensor,
    *,
    control_values: tf.Tensor | None = None,
    control_normalizer: tf.Tensor | float = 0.0,
) -> Mapping[str, tf.Tensor]:
    """Evaluate plain DMIS or the known-integral control-variate estimator."""

    target = _as_float64(target_values, "target_values")
    control = (
        tf.zeros_like(target)
        if control_values is None
        else _as_float64(control_values, "control_values")
    )
    return _value_core(
        target,
        component_log_densities,
        component_weights,
        base_masses,
        control,
        control_normalizer,
        validate=True,
    )


def frozen_importance_directional_estimate(
    target_values: tf.Tensor,
    component_log_densities: tf.Tensor,
    component_weights: tf.Tensor,
    base_masses: tf.Tensor,
    target_tangent: tf.Tensor,
    component_log_density_tangent: tf.Tensor,
    *,
    control_values: tf.Tensor | None = None,
    control_normalizer: tf.Tensor | float = 0.0,
    control_tangent: tf.Tensor | None = None,
    control_normalizer_tangent: tf.Tensor | None = None,
) -> Mapping[str, tf.Tensor]:
    """Return values and explicit directional derivatives of the frozen program.

    Tangent shapes are ``[N,K]`` for the target/control *values* and
    ``[N,J,K]`` for the component log densities.  ``K`` may be one for a
    directional derivative or the parameter dimension for a complete Jacobian.
    Thus ``target_tangent`` is ``d gamma`` (not ``d log gamma``), while the
    component tangent is ``d log q_j``.  Mixture weights and row masses are
    frozen; their derivatives are outside this kernel's contract.
    """

    target = _as_float64(target_values, "target_values")
    logs = _as_float64(component_log_densities, "component_log_densities")
    alpha = _as_float64(component_weights, "component_weights")
    masses = _as_float64(base_masses, "base_masses")
    control = tf.zeros_like(target) if control_values is None else _as_float64(control_values, "control_values")
    target_dot = _as_float64(target_tangent, "target_tangent")
    component_dot = _as_float64(component_log_density_tangent, "component_log_density_tangent")
    if target_dot.shape.rank != 2 or component_dot.shape.rank != 3:
        raise ValueError("target_tangent must be [N,K], component tangent [N,J,K]")
    if target_dot.shape[0] is not None and target.shape[0] is not None:
        if int(target_dot.shape[0]) != int(target.shape[0]):
            raise ValueError("target tangent row count mismatch")
    if component_dot.shape[0] is not None and logs.shape[0] is not None:
        if int(component_dot.shape[0]) != int(logs.shape[0]):
            raise ValueError("component tangent row count mismatch")
    if component_dot.shape[1] is not None and logs.shape[1] is not None:
        if int(component_dot.shape[1]) != int(logs.shape[1]):
            raise ValueError("component tangent count mismatch")
    k = tf.shape(target_dot)[1]
    control_dot = (
        tf.zeros([tf.shape(target)[0], k], DTYPE)
        if control_tangent is None
        else _as_float64(control_tangent, "control_tangent")
    )
    z_dot = (
        tf.zeros([k], DTYPE)
        if control_normalizer_tangent is None
        else tf.reshape(_as_float64(control_normalizer_tangent, "control_normalizer_tangent"), [-1])
    )
    if control_dot.shape.rank != 2 or z_dot.shape.rank != 1:
        raise ValueError("control tangent must be [N,K] and normalizer tangent [K]")
    if target_dot.shape[0] is not None and control_dot.shape[0] is not None:
        if int(target_dot.shape[0]) != int(control_dot.shape[0]):
            raise ValueError("control tangent row count mismatch")
    if target_dot.shape[1] is not None and control_dot.shape[1] is not None:
        if int(target_dot.shape[1]) != int(control_dot.shape[1]):
            raise ValueError("control tangent dimension mismatch")
    if target_dot.shape[1] is not None and z_dot.shape[0] is not None:
        if int(target_dot.shape[1]) != int(z_dot.shape[0]):
            raise ValueError("normalizer tangent dimension mismatch")
    tf.debugging.assert_all_finite(target_dot, "target tangents must be finite")
    tf.debugging.assert_all_finite(component_dot, "component tangents must be finite")
    tf.debugging.assert_all_finite(control_dot, "control tangents must be finite")
    tf.debugging.assert_all_finite(z_dot, "normalizer tangents must be finite")

    value = _value_core(
        target,
        logs,
        alpha,
        masses,
        control,
        control_normalizer,
        validate=True,
    )
    log_q = value["complete_mixture_log_density"]
    tangent_support_valid = tf.reduce_all(tf.math.is_finite(log_q))
    responsibilities = _safe_responsibilities(logs, alpha, log_q)
    log_q_dot = tf.einsum("nj,njk->nk", responsibilities, component_dot)
    residual = target - control
    residual_dot = target_dot - control_dot
    contribution_dot = masses[:, None] * _masked_inverse_density(log_q)[:, None] * (
        residual_dot - residual[:, None] * log_q_dot
    )
    normalizer_dot = z_dot + tf.reduce_sum(contribution_dot, axis=0)
    safe_normalizer = tf.where(
        value["normalizer"] > 0.0,
        value["normalizer"],
        tf.constant(1.0, DTYPE),
    )
    log_normalizer_dot = normalizer_dot / safe_normalizer
    result = dict(value)
    result.update(
        {
            "mixture_responsibilities": responsibilities,
            "mixture_log_density_tangent": log_q_dot,
            "normalizer_tangent": normalizer_dot,
            "log_normalizer_tangent": log_normalizer_dot,
            "tangent_support_valid": tangent_support_valid,
            "tangent_valid": tf.reduce_all(tf.math.is_finite(normalizer_dot))
            & value["valid"]
            & tangent_support_valid,
        }
    )
    return result


def make_compiled_value_kernel(
    sample_count: int,
    component_count: int,
    *,
    jit_compile: bool = True,
):
    """Create a fixed-shape graph for repeated value evaluations."""

    n = int(sample_count)
    j = int(component_count)
    if n <= 0 or j <= 0:
        raise ValueError("sample_count and component_count must be positive")

    @tf.function(
        input_signature=[
            tf.TensorSpec([n], DTYPE),
            tf.TensorSpec([n, j], DTYPE),
            tf.TensorSpec([j], DTYPE),
            tf.TensorSpec([n], DTYPE),
            tf.TensorSpec([n], DTYPE),
            tf.TensorSpec([], DTYPE),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
    )
    def kernel(target, component_logs, alpha, masses, control, z_control):
        return _value_core(
            target,
            component_logs,
            alpha,
            masses,
            control,
            z_control,
            validate=False,
        )

    return kernel


def make_compiled_directional_kernel(
    sample_count: int,
    component_count: int,
    tangent_count: int,
    *,
    jit_compile: bool = True,
):
    """Create a fixed-shape graph for values and explicit tangents."""

    n = int(sample_count)
    j = int(component_count)
    k = int(tangent_count)
    if n <= 0 or j <= 0 or k <= 0:
        raise ValueError("all static dimensions must be positive")

    @tf.function(
        input_signature=[
            tf.TensorSpec([n], DTYPE),
            tf.TensorSpec([n, j], DTYPE),
            tf.TensorSpec([j], DTYPE),
            tf.TensorSpec([n], DTYPE),
            tf.TensorSpec([n], DTYPE),
            tf.TensorSpec([], DTYPE),
            tf.TensorSpec([n, k], DTYPE),
            tf.TensorSpec([n, j, k], DTYPE),
            tf.TensorSpec([n, k], DTYPE),
            tf.TensorSpec([k], DTYPE),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
    )
    def kernel(
        target,
        component_logs,
        alpha,
        masses,
        control,
        z_control,
        target_dot,
        component_log_dot,
        control_dot,
        z_control_dot,
    ):
        return _directional_core_unchecked(
            target,
            component_logs,
            alpha,
            masses,
            control,
            z_control,
            target_dot,
            component_log_dot,
            control_dot,
            z_control_dot,
        )

    return kernel


def _directional_core_unchecked(
    target: tf.Tensor,
    logs: tf.Tensor,
    alpha: tf.Tensor,
    masses: tf.Tensor,
    control: tf.Tensor,
    z_control: tf.Tensor,
    target_dot: tf.Tensor,
    component_dot: tf.Tensor,
    control_dot: tf.Tensor,
    z_dot: tf.Tensor,
) -> Mapping[str, tf.Tensor]:
    value = _value_core(
        target,
        logs,
        alpha,
        masses,
        control,
        z_control,
        validate=False,
    )
    responsibilities = _safe_responsibilities(
        logs, alpha, value["complete_mixture_log_density"]
    )
    log_q_dot = tf.einsum("nj,njk->nk", responsibilities, component_dot)
    residual = target - control
    residual_dot = target_dot - control_dot
    mixture_log_density = value["complete_mixture_log_density"]
    tangent_support_valid = tf.reduce_all(tf.math.is_finite(mixture_log_density))
    normalizer_dot = z_dot + tf.reduce_sum(
        masses[:, None] * _masked_inverse_density(mixture_log_density)[:, None]
        * (residual_dot - residual[:, None] * log_q_dot),
        axis=0,
    )
    safe_normalizer = tf.where(
        value["normalizer"] > 0.0,
        value["normalizer"],
        tf.constant(1.0, DTYPE),
    )
    result = dict(value)
    result.update(
        {
            "mixture_responsibilities": responsibilities,
            "mixture_log_density_tangent": log_q_dot,
            "normalizer_tangent": normalizer_dot,
            "log_normalizer_tangent": normalizer_dot / safe_normalizer,
            "tangent_support_valid": tangent_support_valid,
            "tangent_valid": tf.reduce_all(tf.math.is_finite(normalizer_dot))
            & value["valid"]
            & tangent_support_valid,
        }
    )
    return result


__all__ = [
    "DTYPE",
    "ROUTE_ID",
    "ROUTE_CLASSIFICATION",
    "complete_mixture_log_density",
    "frozen_importance_estimate",
    "frozen_importance_directional_estimate",
    "make_compiled_value_kernel",
    "make_compiled_directional_kernel",
    "validate_mixture_inputs",
]
