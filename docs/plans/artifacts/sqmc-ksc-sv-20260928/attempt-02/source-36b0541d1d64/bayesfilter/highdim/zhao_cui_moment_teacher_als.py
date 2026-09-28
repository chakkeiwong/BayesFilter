"""Reference fixed-ALS value/JVP replay for the Zhao-Cui moment teacher.

This is a setup-static TensorFlow mechanics implementation.  Python loops over
the fixed axis/sweep schedule are intentional here; the module is not an XLA
runtime path.  The value and tangent use the same fixed-design ridge equations
and fail closed on solve residuals or conditioning failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import tensorflow as tf

from bayesfilter.highdim.derivatives import (
    differentiate_design_matrix,
    fixed_design_lsq_derivative,
)
from bayesfilter.highdim.fitting import (
    FixedTTFitConfig,
    FixedTTFitter,
    _solve_scaled_augmented_ridge,
)
from bayesfilter.highdim.diagnostics import HighDimStatus, MeasureConvention
from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.squared_tt import (
    SquaredTTDensity,
    TensorProductReferenceDensity,
)
from bayesfilter.highdim.zhao_cui_moment_teacher import (
    TTNormalizedMarginalJVP,
    TTShapeTargetsJVP,
    squared_tt_normalized_marginal_jvp,
    squared_tt_shape_targets_jvp,
)
from bayesfilter.highdim.tt import FunctionalTT, TTCore


ROUTE_ID = "zhao_cui_fixed_als_value_jvp_reference_v1"
ROUTE_CLASSIFICATION = "extension_or_invention"


@dataclass(frozen=True)
class SquareRootTargetJVP:
    """Conditioned square-root fit rows and their directional derivative."""

    values: tf.Tensor
    tangent: tf.Tensor
    log_scale_shift: tf.Tensor
    dot_log_scale_shift: tf.Tensor
    scale_shift_index: int


@dataclass(frozen=True)
class DefensiveScaleJVP:
    """Defensive coefficient expressed in max-log-scaled TT units."""

    tau: tf.Tensor
    dot_tau: tf.Tensor


@dataclass(frozen=True)
class FixedALSTangentResult:
    """Terminal cores and total directional derivatives of a fixed ALS replay."""

    cores: tuple[TTCore, ...]
    dot_cores: tuple[TTCore, ...]
    update_diagnostics: tuple[dict[str, object], ...]
    route_id: str = ROUTE_ID
    route_classification: str = ROUTE_CLASSIFICATION


@dataclass(frozen=True)
class FixedTTTeacherStepJVP:
    """One reference square-root fit step and its fitted-core direction."""

    density: SquaredTTDensity
    dot_cores: tuple[TTCore, ...]
    square_root_target: SquareRootTargetJVP
    defensive_scale: DefensiveScaleJVP
    update_diagnostics: tuple[dict[str, object], ...]
    route_id: str = ROUTE_ID
    route_classification: str = ROUTE_CLASSIFICATION


@dataclass(frozen=True)
class FixedTTTeacherRecursionJVP:
    """Setup-static reference recursion over fixed conditional target rows."""

    steps: tuple[FixedTTTeacherStepJVP, ...]
    carried_marginals: tuple[TTNormalizedMarginalJVP, ...]
    shape_targets: tuple[TTShapeTargetsJVP, ...]
    route_id: str = "zhao_cui_fixed_tt_teacher_recursion_jvp_reference_v1"
    route_classification: str = ROUTE_CLASSIFICATION


def square_root_target_jvp(
    log_pulled_target: tf.Tensor,
    dot_log_pulled_target: tf.Tensor,
    *,
    scale_shift_index: int | None = None,
) -> SquareRootTargetJVP:
    """Build ``exp((log_target-c)/2)`` and its fixed-branch JVP.

    The maximizing row is selected by the primal value unless a replay index is
    supplied.  Its index is a discrete fixed-branch choice; the tangent of the
    selected log value is still included in every target row.
    """

    log_target = tf.convert_to_tensor(log_pulled_target, tf.float64)
    dot_log_target = tf.convert_to_tensor(dot_log_pulled_target, tf.float64)
    if log_target.shape.rank != 1 or dot_log_target.shape != log_target.shape:
        raise ValueError("log target and tangent must be matching vectors")
    if log_target.shape[0] is None or int(log_target.shape[0]) < 1:
        raise ValueError("log target must be nonempty with a static row count")
    if not bool(
        tf.reduce_all(tf.math.is_finite(log_target)).numpy()
        and tf.reduce_all(tf.math.is_finite(dot_log_target)).numpy()
    ):
        raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
    index = (
        int(tf.argmax(log_target).numpy())
        if scale_shift_index is None
        else int(scale_shift_index)
    )
    if index < 0 or index >= int(log_target.shape[0]):
        raise IndexError("scale_shift_index is out of range")
    shift = log_target[index]
    dot_shift = dot_log_target[index]
    values = tf.exp(0.5 * (log_target - shift))
    tangent = 0.5 * values * (dot_log_target - dot_shift)
    if not bool(
        tf.reduce_all(tf.math.is_finite(values)).numpy()
        and tf.reduce_all(tf.math.is_finite(tangent)).numpy()
    ):
        raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
    return SquareRootTargetJVP(values, tangent, shift, dot_shift, index)


def scaled_defensive_weight_jvp(
    unscaled_weight: tf.Tensor,
    dot_unscaled_weight: tf.Tensor,
    log_scale_shift: tf.Tensor,
    dot_log_scale_shift: tf.Tensor,
) -> DefensiveScaleJVP:
    """Map an unscaled defensive weight into square-root-fit scale units."""

    weight = tf.convert_to_tensor(unscaled_weight, tf.float64)
    dot_weight = tf.convert_to_tensor(dot_unscaled_weight, tf.float64)
    shift = tf.convert_to_tensor(log_scale_shift, tf.float64)
    dot_shift = tf.convert_to_tensor(dot_log_scale_shift, tf.float64)
    if any(value.shape.rank != 0 for value in (weight, dot_weight, shift, dot_shift)):
        raise ValueError("defensive scaling inputs must be scalar")
    values = tf.stack([weight, dot_weight, shift, dot_shift])
    if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()) or bool(
        (weight < 0.0).numpy()
    ):
        raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
    scale = tf.exp(-shift)
    tau = scale * weight
    dot_tau = scale * (dot_weight - weight * dot_shift)
    if not bool(tf.reduce_all(tf.math.is_finite(tf.stack([tau, dot_tau]))).numpy()):
        raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
    return DefensiveScaleJVP(tau, dot_tau)


def fixed_tt_teacher_step_jvp(
    product_basis: ProductBasis,
    points: tf.Tensor,
    log_pulled_target: tf.Tensor,
    dot_log_pulled_target: tf.Tensor,
    weights: tf.Tensor,
    config: FixedTTFitConfig,
    initial_cores: Sequence[TTCore],
    initial_dot_cores: Sequence[TTCore],
    measure_convention: MeasureConvention,
    *,
    unscaled_defensive_weight: tf.Tensor = tf.constant(0.0, tf.float64),
    dot_unscaled_defensive_weight: tf.Tensor = tf.constant(0.0, tf.float64),
    normalizer_floor: tf.Tensor = tf.constant(1e-12, tf.float64),
    denominator_floor: tf.Tensor = tf.constant(1e-12, tf.float64),
    scale_shift_index: int | None = None,
    condition_number_veto: float | None = None,
) -> FixedTTTeacherStepJVP:
    """Fit one conditional squared-TT teacher step on a fixed reference design."""

    target = square_root_target_jvp(
        log_pulled_target,
        dot_log_pulled_target,
        scale_shift_index=scale_shift_index,
    )
    defensive = scaled_defensive_weight_jvp(
        unscaled_defensive_weight,
        dot_unscaled_defensive_weight,
        target.log_scale_shift,
        target.dot_log_scale_shift,
    )
    fit = fixed_als_value_jvp(
        product_basis,
        points,
        target.values,
        weights,
        target.tangent,
        config,
        initial_cores,
        initial_dot_cores,
        measure_convention,
        condition_number_veto=condition_number_veto,
    )
    sqrt_tt = FunctionalTT(fit.cores, product_basis, measure_convention)
    defensive_density = TensorProductReferenceDensity(
        product_basis, measure_convention
    )
    identity = SquaredTTDensity.expected_branch_identity(
        sqrt_tt=sqrt_tt,
        defensive_density=defensive_density,
        tau=defensive.tau,
        normalizer_floor=normalizer_floor,
        denominator_floor=denominator_floor,
        measure_convention=measure_convention,
    )
    density = SquaredTTDensity(
        sqrt_tt=sqrt_tt,
        defensive_density=defensive_density,
        tau=defensive.tau,
        normalizer_floor=normalizer_floor,
        denominator_floor=denominator_floor,
        measure_convention=measure_convention,
        branch_identity=identity,
    )
    return FixedTTTeacherStepJVP(
        density,
        fit.dot_cores,
        target,
        defensive,
        fit.update_diagnostics,
    )


def fixed_tt_teacher_recursion_jvp(
    product_basis: ProductBasis,
    points: tf.Tensor,
    base_log_pulled_targets: tf.Tensor,
    dot_base_log_pulled_targets: tf.Tensor,
    weights: tf.Tensor,
    config: FixedTTFitConfig,
    initial_cores: Sequence[TTCore],
    initial_dot_cores: Sequence[TTCore],
    measure_convention: MeasureConvention,
    *,
    carried_keep_axes: Sequence[int],
    previous_query_points: tf.Tensor,
    state_offset: tf.Tensor,
    state_matrix: tf.Tensor,
    pair_indices: Sequence[tuple[int, int]] = (),
    dot_state_offset: tf.Tensor | None = None,
    dot_state_matrix: tf.Tensor | None = None,
    unscaled_defensive_weight: tf.Tensor = tf.constant(0.0, tf.float64),
    dot_unscaled_defensive_weight: tf.Tensor = tf.constant(0.0, tf.float64),
    normalizer_floor: tf.Tensor = tf.constant(1e-12, tf.float64),
    denominator_floor: tf.Tensor = tf.constant(1e-12, tf.float64),
    condition_number_veto: float | None = None,
) -> FixedTTTeacherRecursionJVP:
    """Run a fixed-branch TT teacher recursion and one analytical direction.

    Each row of ``base_log_pulled_targets`` contains the model transition,
    observation, chart-Jacobian, and reference-measure terms at fixed ``theta``.
    From the second time onward this function adds the log normalized marginal
    carried by the previous fitted teacher, including its quotient-rule JVP.
    The returned shape targets can be passed directly to
    ``higher_moment_shape_jvp``; no TT normalizer is a particle likelihood.
    """

    points = tf.convert_to_tensor(points, tf.float64)
    base_targets = tf.convert_to_tensor(base_log_pulled_targets, tf.float64)
    dot_base_targets = tf.convert_to_tensor(
        dot_base_log_pulled_targets, tf.float64
    )
    query_points = tf.convert_to_tensor(previous_query_points, tf.float64)
    if base_targets.shape.rank != 2 or dot_base_targets.shape != base_targets.shape:
        raise ValueError("base log targets and tangents must be matching matrices")
    if points.shape.rank != 2 or base_targets.shape[1] != points.shape[0]:
        raise ValueError("every target row must match the fixed fit design")
    if query_points.shape.rank != 2 or query_points.shape[0] != points.shape[0]:
        raise ValueError("previous_query_points must align with fit rows")
    horizon = int(base_targets.shape[0])
    if horizon < 1:
        raise ValueError("teacher recursion horizon must be positive")
    active_cores = tuple(initial_cores)
    active_dot_cores = tuple(initial_dot_cores)
    previous_marginal: TTNormalizedMarginalJVP | None = None
    steps = []
    marginals = []
    shapes = []
    for time_index in range(horizon):
        log_target = base_targets[time_index]
        dot_log_target = dot_base_targets[time_index]
        if previous_marginal is not None:
            if not bool(tf.reduce_all(previous_marginal.values > 0.0).numpy()):
                raise ValueError(HighDimStatus.NORMALIZER_FLOOR_EXCEEDED.value)
            log_target = log_target + tf.math.log(previous_marginal.values)
            dot_log_target = (
                dot_log_target
                + previous_marginal.tangent / previous_marginal.values
            )
        step = fixed_tt_teacher_step_jvp(
            product_basis,
            points,
            log_target,
            dot_log_target,
            weights,
            config,
            active_cores,
            active_dot_cores,
            measure_convention,
            unscaled_defensive_weight=unscaled_defensive_weight,
            dot_unscaled_defensive_weight=dot_unscaled_defensive_weight,
            normalizer_floor=normalizer_floor,
            denominator_floor=denominator_floor,
            condition_number_veto=condition_number_veto,
        )
        shape = squared_tt_shape_targets_jvp(
            step.density,
            state_offset,
            state_matrix,
            step.dot_cores,
            pair_indices=pair_indices,
            dot_state_offset=dot_state_offset,
            dot_state_matrix=dot_state_matrix,
            dot_tau=step.defensive_scale.dot_tau,
        )
        previous_marginal = squared_tt_normalized_marginal_jvp(
            step.density,
            carried_keep_axes,
            query_points,
            step.dot_cores,
            dot_tau=step.defensive_scale.dot_tau,
        )
        steps.append(step)
        shapes.append(shape)
        marginals.append(previous_marginal)
        active_cores = step.density.sqrt_tt.cores
        active_dot_cores = step.dot_cores
    return FixedTTTeacherRecursionJVP(
        tuple(steps), tuple(marginals), tuple(shapes)
    )


def fixed_als_value_jvp(
    product_basis: ProductBasis,
    points: tf.Tensor,
    target_values: tf.Tensor,
    weights: tf.Tensor,
    dot_target_values: tf.Tensor,
    config: FixedTTFitConfig,
    initial_cores: Sequence[TTCore],
    initial_dot_cores: Sequence[TTCore],
    measure_convention: MeasureConvention,
    *,
    dot_weights: tf.Tensor | None = None,
    condition_number_veto: float | None = None,
) -> FixedALSTangentResult:
    """Replay the fixed ALS schedule and its forward JVP.

    ``points`` and the product basis are fixed in this reference route.  The
    tangent therefore covers fitted-core, target, and optional weight
    dependence with a fixed ridge.  Every update uses the current value/tangent
    cores, so later design tangents include all earlier ALS updates.  Moving
    rows, moving bases, and a parameter-dependent ridge are not implemented.
    """
    if not isinstance(product_basis, ProductBasis):
        raise TypeError("product_basis must be a ProductBasis")
    if measure_convention != product_basis.convention:
        raise ValueError("measure convention mismatch")
    points = tf.convert_to_tensor(points, tf.float64)
    target_values = tf.convert_to_tensor(target_values, tf.float64)
    dot_target_values = tf.convert_to_tensor(dot_target_values, tf.float64)
    weights = tf.convert_to_tensor(weights, tf.float64)
    if dot_weights is None:
        dot_weights = tf.zeros_like(weights)
    else:
        dot_weights = tf.convert_to_tensor(dot_weights, tf.float64)
    if points.shape.rank != 2 or target_values.shape.rank != 1:
        raise ValueError("invalid fixed-ALS tensor shapes")
    if dot_target_values.shape != target_values.shape or weights.shape != target_values.shape:
        raise ValueError("target/weight shapes must match")
    if dot_weights.shape != weights.shape:
        raise ValueError("dot_weights shape must match weights")
    fitter = FixedTTFitter()
    cores = tuple(initial_cores)
    dot_cores = tuple(initial_dot_cores)
    if len(cores) != product_basis.dimension or len(dot_cores) != len(cores):
        raise ValueError("core count does not match product basis")
    for core, dot_core in zip(cores, dot_cores):
        if not isinstance(core, TTCore) or not isinstance(dot_core, TTCore):
            raise TypeError("cores must be TTCore objects")
        if core.values.shape != dot_core.values.shape:
            raise ValueError("core tangent shape mismatch")
    veto = float(config.condition_number_veto if condition_number_veto is None else condition_number_veto)
    records: list[dict[str, object]] = []

    for sweep_index in range(config.max_sweeps):
        for core_index in config.sweep_order:
            system = fitter.build_core_update_system(
                product_basis,
                points,
                target_values,
                weights,
                cores,
                core_index=core_index,
                config=config,
            )
            design = system.design_matrix
            dot_design = differentiate_design_matrix(
                product_basis,
                points,
                cores,
                dot_cores,
                core_index=core_index,
            )
            solve = _solve_scaled_augmented_ridge(
                design=design,
                target_values=target_values,
                weights=weights,
                ridge=config.ridge,
                column_scale_floor=config.column_scale_floor,
                stabilization_policy_id=config.stabilization_policy_id,
                solver_backend=config.solver_backend,
            )
            condition = float(solve.scaled_augmented_condition_number)
            if not tf.math.is_finite(tf.constant(condition, tf.float64)) or condition > veto:
                raise ValueError(HighDimStatus.CONDITION_NUMBER_VETO.value)
            solution = tf.reshape(solve.solution, cores[core_index].values.shape)
            coeff = tf.reshape(solution, [-1])
            derivative = fixed_design_lsq_derivative(
                design_matrix=design,
                target_values=target_values,
                weights=weights,
                coefficients=coeff,
                dot_target_values=dot_target_values,
                dot_design_matrix=dot_design,
                dot_weights=dot_weights,
                ridge=config.ridge,
                condition_number_veto=veto,
            )
            if derivative.status is not HighDimStatus.OK:
                raise ValueError(derivative.status.value)
            dot_solution = tf.reshape(
                derivative.dot_coefficients,
                cores[core_index].values.shape,
            )
            value_system_residual = tf.linalg.matvec(system.normal_matrix, coeff) - system.rhs
            jvp_system_residual = (
                tf.linalg.matvec(system.normal_matrix, derivative.dot_coefficients)
                + tf.linalg.matvec(derivative.dot_normal_matrix, coeff)
                - derivative.dot_rhs
            )
            value_system_residual_norm = tf.linalg.norm(value_system_residual)
            jvp_system_residual_norm = tf.linalg.norm(jvp_system_residual)
            fit_residual = tf.linalg.matvec(design, coeff) - target_values
            fit_jvp_residual = (
                tf.linalg.matvec(design, derivative.dot_coefficients)
                + tf.linalg.matvec(dot_design, coeff)
                - dot_target_values
            )
            fit_residual_norm = tf.sqrt(
                tf.reduce_sum(weights * tf.square(fit_residual)) / tf.reduce_sum(weights)
            )
            fit_jvp_residual_norm = tf.sqrt(
                tf.reduce_sum(weights * tf.square(fit_jvp_residual))
                / tf.reduce_sum(weights)
            )
            if not bool(
                tf.reduce_all(
                    tf.math.is_finite(
                        [
                            value_system_residual_norm,
                            jvp_system_residual_norm,
                            fit_residual_norm,
                            fit_jvp_residual_norm,
                        ]
                    )
                ).numpy()
            ):
                raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
            records.append(
                {
                    "sweep_index": sweep_index,
                    "core_index": int(core_index),
                    "condition_number": condition,
                    "jvp_normal_condition_number": derivative.condition_number,
                    "value_solve_residual": value_system_residual_norm,
                    "jvp_solve_residual": jvp_system_residual_norm,
                    "weighted_fit_residual": fit_residual_norm,
                    "weighted_fit_jvp_residual": fit_jvp_residual_norm,
                    "design_tangent_included": True,
                    "weight_tangent_included": True,
                    "ridge_tangent": 0.0,
                    "fit_rows_and_basis_fixed": True,
                    "value_solve_residual_relative": value_system_residual_norm
                    / tf.maximum(tf.linalg.norm(system.rhs), tf.constant(1.0, tf.float64)),
                    "jvp_solve_residual_relative": jvp_system_residual_norm
                    / tf.maximum(tf.linalg.norm(derivative.dot_rhs), tf.constant(1.0, tf.float64)),
                }
            )
            updated = list(cores)
            updated_dot = list(dot_cores)
            updated[core_index] = TTCore(solution)
            updated_dot[core_index] = TTCore(dot_solution)
            cores = tuple(updated)
            dot_cores = tuple(updated_dot)
    return FixedALSTangentResult(cores, dot_cores, tuple(records))


__all__ = [
    "ROUTE_ID",
    "ROUTE_CLASSIFICATION",
    "SquareRootTargetJVP",
    "DefensiveScaleJVP",
    "FixedALSTangentResult",
    "FixedTTTeacherStepJVP",
    "FixedTTTeacherRecursionJVP",
    "square_root_target_jvp",
    "scaled_defensive_weight_jvp",
    "fixed_tt_teacher_step_jvp",
    "fixed_tt_teacher_recursion_jvp",
    "fixed_als_value_jvp",
]
