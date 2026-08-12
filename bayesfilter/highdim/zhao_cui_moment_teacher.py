"""Squared-TT moment contractions for the opt-in Contract E teacher.

This module is an ``extension_or_invention``.  Zhao-Cui squared-TT density and
paired-core marginal operations are the source-grounded primitives; using the
resulting moments as Contract E targets is BayesFilter-owned.

The current API is an FP64 TensorFlow mechanics reference.  It iterates over a
setup-static tuple of variable-rank TT cores in Python, so it is not eligible
for the repository XLA runtime until the cores have a graph-native padded or
masked representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.bases import (
    BoundedInterval,
    LagrangePiecewiseBasis1D,
    LegendreBasis1D,
)
from bayesfilter.highdim.diagnostics import HighDimStatus, MassMeasure
from bayesfilter.highdim.squared_tt import (
    SquaredTTDensity,
    TensorProductReferenceDensity,
)
from bayesfilter.highdim.tt import TTCore
from bayesfilter.highdim.higher_moment_contract_e import higher_moment_shape_jvp


ROUTE_CLASSIFICATION = "extension_or_invention"
ROUTE_ID = "zhao_cui_squared_tt_contract_e_moment_teacher_reference_v1"


@dataclass(frozen=True)
class TTObservableJVP:
    """Normalized squared-TT observable and one directional derivative."""

    value: tf.Tensor
    tangent: tf.Tensor
    numerator: tf.Tensor
    numerator_tangent: tf.Tensor
    normalizer: tf.Tensor
    normalizer_tangent: tf.Tensor

    def __post_init__(self) -> None:
        for name in (
            "value",
            "tangent",
            "numerator",
            "numerator_tangent",
            "normalizer",
            "normalizer_tangent",
        ):
            value = tf.convert_to_tensor(getattr(self, name), dtype=tf.float64)
            if value.shape.rank != 0:
                raise ValueError(f"{name} must be scalar")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class TTReferenceMoments:
    """First two moments of selected reference coordinates."""

    axes: tuple[int, ...]
    mean: tf.Tensor
    raw_second: tf.Tensor
    covariance: tf.Tensor

    def __post_init__(self) -> None:
        axes = tuple(int(axis) for axis in self.axes)
        mean = tf.convert_to_tensor(self.mean, dtype=tf.float64)
        raw_second = tf.convert_to_tensor(self.raw_second, dtype=tf.float64)
        covariance = tf.convert_to_tensor(self.covariance, dtype=tf.float64)
        size = len(axes)
        if mean.shape != (size,):
            raise ValueError("mean shape must match axes")
        if raw_second.shape != (size, size) or covariance.shape != (size, size):
            raise ValueError("second-moment shapes must match axes")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "raw_second", raw_second)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class TTNormalizedMarginalJVP:
    """Normalized retained squared-TT density values and one JVP."""

    values: tf.Tensor
    tangent: tf.Tensor
    numerator: tf.Tensor
    numerator_tangent: tf.Tensor
    normalizer: tf.Tensor
    normalizer_tangent: tf.Tensor


@dataclass(frozen=True)
class FrozenTTShapeTargets:
    """Shape targets for an explicitly frozen finite TT teacher diagnostic."""

    skew: tf.Tensor
    kurtosis: tf.Tensor
    pairwise_co_skew: tf.Tensor
    pairwise_co_kurtosis: tf.Tensor
    pairwise_co_skew_mask: tf.Tensor
    pairwise_co_kurtosis_mask: tf.Tensor
    skew_tangent: tf.Tensor
    kurtosis_tangent: tf.Tensor
    pairwise_co_skew_tangent: tf.Tensor
    pairwise_co_kurtosis_tangent: tf.Tensor
    route_id: str = "zhao_cui_frozen_tt_shape_targets_diagnostic_v1"
    route_classification: str = ROUTE_CLASSIFICATION
    tangent_semantics: str = "zero_tangent_of_explicitly_frozen_teacher_not_refitted_score"

    def explicit_target_kwargs(self) -> dict[str, tf.Tensor]:
        return {
            "explicit_target_skew": self.skew,
            "explicit_target_kurtosis": self.kurtosis,
            "explicit_target_skew_tangent": self.skew_tangent,
            "explicit_target_kurtosis_tangent": self.kurtosis_tangent,
            "explicit_target_pairwise_co_skew": self.pairwise_co_skew,
            "explicit_target_pairwise_co_kurtosis": self.pairwise_co_kurtosis,
            "explicit_target_pairwise_co_skew_tangent": self.pairwise_co_skew_tangent,
            "explicit_target_pairwise_co_kurtosis_tangent": self.pairwise_co_kurtosis_tangent,
            "pairwise_co_skew_target_mask": self.pairwise_co_skew_mask,
            "pairwise_co_kurtosis_target_mask": self.pairwise_co_kurtosis_mask,
        }


@dataclass(frozen=True)
class TTShapeTargetsJVP:
    """Analytical shape targets and one directional tangent."""

    skew: tf.Tensor
    kurtosis: tf.Tensor
    pairwise_co_skew: tf.Tensor
    pairwise_co_kurtosis: tf.Tensor
    pairwise_co_skew_mask: tf.Tensor
    pairwise_co_kurtosis_mask: tf.Tensor
    skew_tangent: tf.Tensor
    kurtosis_tangent: tf.Tensor
    pairwise_co_skew_tangent: tf.Tensor
    pairwise_co_kurtosis_tangent: tf.Tensor
    route_id: str = "zhao_cui_recursive_tt_shape_targets_reference_v1"
    route_classification: str = ROUTE_CLASSIFICATION
    tangent_semantics: str = "manual_total_jvp_of_fixed_tt_teacher_shape_targets"

    def explicit_target_kwargs(self) -> dict[str, tf.Tensor]:
        return {
            "explicit_target_skew": self.skew,
            "explicit_target_kurtosis": self.kurtosis,
            "explicit_target_skew_tangent": self.skew_tangent,
            "explicit_target_kurtosis_tangent": self.kurtosis_tangent,
            "explicit_target_pairwise_co_skew": self.pairwise_co_skew,
            "explicit_target_pairwise_co_kurtosis": self.pairwise_co_kurtosis,
            "explicit_target_pairwise_co_skew_tangent": self.pairwise_co_skew_tangent,
            "explicit_target_pairwise_co_kurtosis_tangent": self.pairwise_co_kurtosis_tangent,
            "pairwise_co_skew_target_mask": self.pairwise_co_skew_mask,
            "pairwise_co_kurtosis_target_mask": self.pairwise_co_kurtosis_mask,
        }


@dataclass(frozen=True)
class TTParticleContractEStepJVP:
    """One reference particle OT/Contract E/TT-shape composition result."""

    particle_log_increment: tf.Tensor
    particle_log_increment_tangent: tf.Tensor
    particles: tf.Tensor
    particles_tangent: tf.Tensor
    diagnostics: Mapping[str, tf.Tensor]
    route_id: str = "zhao_cui_tt_particle_contract_e_step_jvp_reference_v1"
    route_classification: str = ROUTE_CLASSIFICATION


def legendre_monomial_operator_matrix(
    basis: object,
    power: int,
    measure: MassMeasure,
) -> tf.Tensor:
    """Return ``integral phi phi.T x**power dmeasure`` by exact quadrature."""

    power = int(power)
    if power < 0:
        raise ValueError("power must be nonnegative")
    if not isinstance(measure, MassMeasure):
        raise TypeError("measure must be a MassMeasure")
    delegate = getattr(basis, "delegate", basis)
    if isinstance(delegate, LegendreBasis1D):
        polynomial_degree = 2 * int(delegate.max_degree) + power
        order = max(2, (polynomial_degree + 2) // 2)
        nodes, weights = _legendre_gauss_nodes_weights(order)
        midpoint = 0.5 * (delegate.domain.left + delegate.domain.right)
        half_length = 0.5 * delegate.domain.length
        points = midpoint + half_length * nodes
        active_weights = 0.5 * weights
        if measure is MassMeasure.REFERENCE_LEBESGUE:
            active_weights = active_weights * delegate.domain.length
        elif measure is not MassMeasure.REFERENCE_MEASURE:
            raise TypeError("unsupported mass measure")
        values = basis.evaluate(points)
        return tf.einsum(
            "n,nl,nm->lm",
            active_weights * tf.pow(points, power),
            values,
            values,
        )
    if not isinstance(delegate, LagrangePiecewiseBasis1D):
        raise TypeError(
            "moment-teacher reference requires Legendre or piecewise Lagrange basis"
        )
    # Lane B evaluates cardinal polynomials at r(u), but the declared mass
    # measure is uniform in the bounded reference coordinate u.
    polynomial_degree = 2 * int(delegate.order) + power
    order = max(2, (polynomial_degree + 2) // 2)
    nodes, weights = _legendre_gauss_nodes_weights(order)
    element_length = tf.constant(2.0 / float(delegate.num_elems), tf.float64)
    result = tf.zeros([basis.basis_dim, basis.basis_dim], tf.float64)
    for element in range(int(delegate.num_elems)):
        left = tf.constant(-1.0, tf.float64) + tf.cast(
            element, tf.float64
        ) * element_length
        midpoint = left + 0.5 * element_length
        reference_points = midpoint + 0.5 * element_length * nodes
        points = delegate.domain.from_reference(reference_points)
        active_weights = 0.5 * element_length * weights
        if measure is MassMeasure.REFERENCE_MEASURE:
            active_weights = 0.5 * active_weights
        elif measure is not MassMeasure.REFERENCE_LEBESGUE:
            raise TypeError("unsupported mass measure")
        values = basis.evaluate(points)
        result += tf.einsum(
            "n,nl,nm->lm",
            active_weights * tf.pow(reference_points, power),
            values,
            values,
        )
    return result


def _constant_basis_coefficients(
    basis: object,
) -> tf.Tensor:
    delegate = getattr(basis, "delegate", basis)
    if isinstance(delegate, LegendreBasis1D):
        return tf.one_hot(0, basis.basis_dim, dtype=tf.float64)
    if isinstance(delegate, LagrangePiecewiseBasis1D):
        return tf.ones([basis.basis_dim], tf.float64)
    raise TypeError("unsupported constant-function basis representation")


def monomial_operator_matrices(
    density: SquaredTTDensity,
    powers: Sequence[int],
) -> tuple[tf.Tensor, ...]:
    """Build one monomial operator matrix for every TT axis."""

    power_tuple = tuple(int(power) for power in powers)
    dimension = len(density.sqrt_tt.cores)
    if len(power_tuple) != dimension:
        raise ValueError("powers must have one entry per TT axis")
    return tuple(
        legendre_monomial_operator_matrix(
            density.sqrt_tt.product_basis.bases[axis],
            power_tuple[axis],
            density.measure_convention.mass_measure,
        )
        for axis in range(dimension)
    )


def squared_tt_separable_observable(
    density: SquaredTTDensity,
    operator_matrices: Sequence[tf.Tensor],
    *,
    defensive_moment: tf.Tensor,
) -> tf.Tensor:
    """Return the normalized expectation of one separable observable."""

    sqrt_numerator = _paired_contraction(
        density.sqrt_tt.cores,
        _operator_tuple(density, operator_matrices),
    )
    q0_moment = _finite_scalar(defensive_moment, "defensive_moment")
    numerator = sqrt_numerator + density.tau * q0_moment
    value = numerator / density.normalizer()
    _assert_finite(value, "normalized observable")
    return value


def squared_tt_separable_observable_jvp(
    density: SquaredTTDensity,
    operator_matrices: Sequence[tf.Tensor],
    dot_cores: Sequence[TTCore],
    *,
    defensive_moment: tf.Tensor,
    dot_operator_matrices: Sequence[tf.Tensor] | None = None,
    dot_defensive_moment: tf.Tensor = tf.constant(0.0, tf.float64),
    dot_tau: tf.Tensor = tf.constant(0.0, tf.float64),
    dot_defensive_mass: tf.Tensor = tf.constant(0.0, tf.float64),
) -> TTObservableJVP:
    """Differentiate a normalized observable on one fixed squared-TT branch."""

    operators = _operator_tuple(density, operator_matrices)
    dot_operators = (
        tuple(tf.zeros_like(operator) for operator in operators)
        if dot_operator_matrices is None
        else _operator_tuple(density, dot_operator_matrices)
    )
    core_directions = _core_direction_tuple(density, dot_cores)
    sqrt_numerator, dot_sqrt_numerator = _paired_contraction_jvp(
        density.sqrt_tt.cores,
        core_directions,
        operators,
        dot_operators,
    )
    q0_moment = _finite_scalar(defensive_moment, "defensive_moment")
    dot_q0_moment = _finite_scalar(dot_defensive_moment, "dot_defensive_moment")
    tau_dot = _finite_scalar(dot_tau, "dot_tau")
    numerator = sqrt_numerator + density.tau * q0_moment
    numerator_tangent = (
        dot_sqrt_numerator + tau_dot * q0_moment + density.tau * dot_q0_moment
    )

    mass_operators = tuple(
        basis.mass_matrix(density.measure_convention.mass_measure)
        for basis in density.sqrt_tt.product_basis.bases
    )
    _, dot_sqrt_mass = _paired_contraction_jvp(
        density.sqrt_tt.cores,
        core_directions,
        mass_operators,
        tuple(tf.zeros_like(operator) for operator in mass_operators),
    )
    q0_mass = density.defensive_density.normalizer(
        density.measure_convention.mass_measure
    )
    q0_mass_dot = _finite_scalar(dot_defensive_mass, "dot_defensive_mass")
    normalizer = density.normalizer()
    normalizer_tangent = (
        dot_sqrt_mass + tau_dot * q0_mass + density.tau * q0_mass_dot
    )
    value = numerator / normalizer
    tangent = (
        numerator_tangent * normalizer - numerator * normalizer_tangent
    ) / tf.square(normalizer)
    for name, value_to_check in (
        ("observable", value),
        ("observable tangent", tangent),
        ("normalizer tangent", normalizer_tangent),
    ):
        _assert_finite(value_to_check, name)
    return TTObservableJVP(
        value=value,
        tangent=tangent,
        numerator=numerator,
        numerator_tangent=numerator_tangent,
        normalizer=normalizer,
        normalizer_tangent=normalizer_tangent,
    )


def tensor_product_reference_monomial_moment(
    density: SquaredTTDensity,
    powers: Sequence[int],
) -> tf.Tensor:
    """Return the unnormalized monomial moment of the supported defensive density."""

    if not isinstance(density.defensive_density, TensorProductReferenceDensity):
        raise NotImplementedError(
            "moment teacher currently requires TensorProductReferenceDensity"
        )
    power_tuple = tuple(int(power) for power in powers)
    if len(power_tuple) != len(density.sqrt_tt.cores):
        raise ValueError("powers must have one entry per TT axis")
    value = tf.constant(1.0, tf.float64) + density.defensive_density.floor
    for axis, power in enumerate(power_tuple):
        operator = legendre_monomial_operator_matrix(
            density.sqrt_tt.product_basis.bases[axis],
            power,
            density.measure_convention.mass_measure,
        )
        constant = _constant_basis_coefficients(
            density.sqrt_tt.product_basis.bases[axis]
        )
        value = value * tf.einsum("l,lm,m->", constant, operator, constant)
    return value


def squared_tt_raw_moment(
    density: SquaredTTDensity,
    powers: Sequence[int],
) -> tf.Tensor:
    """Return a normalized reference-coordinate raw moment."""

    return squared_tt_separable_observable(
        density,
        monomial_operator_matrices(density, powers),
        defensive_moment=tensor_product_reference_monomial_moment(density, powers),
    )


def squared_tt_reference_moments(
    density: SquaredTTDensity,
    axes: Sequence[int],
) -> TTReferenceMoments:
    """Return mean and covariance for selected reference coordinates."""

    dimension = len(density.sqrt_tt.cores)
    axis_tuple = tuple(int(axis) for axis in axes)
    if len(set(axis_tuple)) != len(axis_tuple):
        raise ValueError("axes must be unique")
    if any(axis < 0 or axis >= dimension for axis in axis_tuple):
        raise IndexError("axis out of range")
    means = []
    for axis in axis_tuple:
        powers = [0] * dimension
        powers[axis] = 1
        means.append(squared_tt_raw_moment(density, powers))
    mean = tf.stack(means)
    rows = []
    for left_axis in axis_tuple:
        row = []
        for right_axis in axis_tuple:
            powers = [0] * dimension
            powers[left_axis] += 1
            powers[right_axis] += 1
            row.append(squared_tt_raw_moment(density, powers))
        rows.append(tf.stack(row))
    raw_second = tf.stack(rows)
    covariance = raw_second - mean[:, None] * mean[None, :]
    covariance = 0.5 * (covariance + tf.transpose(covariance))
    return TTReferenceMoments(
        axes=axis_tuple,
        mean=mean,
        raw_second=raw_second,
        covariance=covariance,
    )


def squared_tt_normalized_marginal_jvp(
    density: SquaredTTDensity,
    keep_axes: Sequence[int],
    points: tf.Tensor,
    dot_cores: Sequence[TTCore],
    *,
    dot_tau: tf.Tensor = tf.constant(0.0, tf.float64),
) -> TTNormalizedMarginalJVP:
    """Evaluate a source-style squared-TT marginal and its manual JVP."""

    axes = tuple(sorted(set(int(axis) for axis in keep_axes)))
    dimension = len(density.sqrt_tt.cores)
    if not axes or any(axis < 0 or axis >= dimension for axis in axes):
        raise IndexError("keep_axes must contain valid retained coordinates")
    values = tf.convert_to_tensor(points, tf.float64)
    if values.shape.rank == 1 and len(axes) == 1:
        values = values[:, tf.newaxis]
    if values.shape.rank != 2 or values.shape[1] != len(axes):
        raise ValueError("points must have one column per retained axis")
    directions = _core_direction_tuple(density, dot_cores)
    tau_dot = _finite_scalar(dot_tau, "dot_tau")
    point_axis = {axis: index for index, axis in enumerate(axes)}
    count = tf.shape(values)[0]
    vector = tf.ones([count, 1], tf.float64)
    dot_vector = tf.zeros_like(vector)
    for axis, (core, dot_core) in enumerate(zip(density.sqrt_tt.cores, directions)):
        if axis in point_axis:
            basis_values = density.sqrt_tt.product_basis.evaluate_axis(
                axis, values[:, point_axis[axis]]
            )
            paired = tf.einsum(
                "nl,nm,alb,AmB->naAbB",
                basis_values,
                basis_values,
                core.values,
                core.values,
            )
            dot_paired = tf.einsum(
                "nl,nm,alb,AmB->naAbB",
                basis_values,
                basis_values,
                dot_core.values,
                core.values,
            ) + tf.einsum(
                "nl,nm,alb,AmB->naAbB",
                basis_values,
                basis_values,
                core.values,
                dot_core.values,
            )
        else:
            mass = density.sqrt_tt.product_basis.bases[axis].mass_matrix(
                density.measure_convention.mass_measure
            )
            base = tf.einsum(
                "alb,AmB,lm->aAbB", core.values, core.values, mass
            )
            dot_base = tf.einsum(
                "alb,AmB,lm->aAbB", dot_core.values, core.values, mass
            ) + tf.einsum(
                "alb,AmB,lm->aAbB", core.values, dot_core.values, mass
            )
            shape = [
                count,
                core.left_rank,
                core.left_rank,
                core.right_rank,
                core.right_rank,
            ]
            paired = tf.broadcast_to(base[tf.newaxis], shape)
            dot_paired = tf.broadcast_to(dot_base[tf.newaxis], shape)
        matrices = tf.reshape(
            paired,
            [count, core.left_rank**2, core.right_rank**2],
        )
        dot_matrices = tf.reshape(
            dot_paired,
            [count, core.left_rank**2, core.right_rank**2],
        )
        next_dot = tf.einsum("na,nab->nb", dot_vector, matrices) + tf.einsum(
            "na,nab->nb", vector, dot_matrices
        )
        vector = tf.einsum("na,nab->nb", vector, matrices)
        dot_vector = next_dot
    sqrt_numerator = tf.reshape(vector, [count])
    dot_sqrt_numerator = tf.reshape(dot_vector, [count])
    defensive = density._defensive_marginal_values(axes, values)
    numerator = sqrt_numerator + density.tau * defensive
    numerator_tangent = dot_sqrt_numerator + tau_dot * defensive
    mass_operators = tuple(
        basis.mass_matrix(density.measure_convention.mass_measure)
        for basis in density.sqrt_tt.product_basis.bases
    )
    _, dot_sqrt_mass = _paired_contraction_jvp(
        density.sqrt_tt.cores,
        directions,
        mass_operators,
        tuple(tf.zeros_like(operator) for operator in mass_operators),
    )
    defensive_mass = density.defensive_density.normalizer(
        density.measure_convention.mass_measure
    )
    normalizer = density.normalizer()
    normalizer_tangent = dot_sqrt_mass + tau_dot * defensive_mass
    normalized = numerator / normalizer
    tangent = (
        numerator_tangent * normalizer - numerator * normalizer_tangent
    ) / tf.square(normalizer)
    _assert_finite(normalized, "normalized marginal")
    _assert_finite(tangent, "normalized marginal tangent")
    return TTNormalizedMarginalJVP(
        normalized,
        tangent,
        numerator,
        numerator_tangent,
        normalizer,
        normalizer_tangent,
    )


def squared_tt_affine_form_moment_jvp(
    density: SquaredTTDensity,
    first_coefficients: tf.Tensor,
    first_offset: tf.Tensor,
    first_power: int,
    dot_cores: Sequence[TTCore],
    *,
    dot_first_coefficients: tf.Tensor | None = None,
    dot_first_offset: tf.Tensor = tf.constant(0.0, tf.float64),
    second_coefficients: tf.Tensor | None = None,
    second_offset: tf.Tensor = tf.constant(0.0, tf.float64),
    second_power: int = 0,
    dot_second_coefficients: tf.Tensor | None = None,
    dot_second_offset: tf.Tensor = tf.constant(0.0, tf.float64),
    dot_tau: tf.Tensor = tf.constant(0.0, tf.float64),
) -> TTObservableJVP:
    """Contract one product of affine-form powers and its manual JVP."""

    dimension = len(density.sqrt_tt.cores)
    first = _coefficient_vector(first_coefficients, dimension, "first_coefficients")
    dot_first = (
        tf.zeros_like(first)
        if dot_first_coefficients is None
        else _coefficient_vector(
            dot_first_coefficients, dimension, "dot_first_coefficients"
        )
    )
    if second_coefficients is None:
        second = tf.zeros_like(first)
    else:
        second = _coefficient_vector(
            second_coefficients, dimension, "second_coefficients"
        )
    dot_second = (
        tf.zeros_like(second)
        if dot_second_coefficients is None
        else _coefficient_vector(
            dot_second_coefficients, dimension, "dot_second_coefficients"
        )
    )
    first_power = int(first_power)
    second_power = int(second_power)
    if first_power < 0 or second_power < 0:
        raise ValueError("affine-form powers must be nonnegative")
    core_directions = _core_direction_tuple(density, dot_cores)
    max_power = first_power + second_power
    axis_operators = tuple(
        tuple(
            legendre_monomial_operator_matrix(
                density.sqrt_tt.product_basis.bases[axis],
                power,
                density.measure_convention.mass_measure,
            )
            for power in range(max_power + 1)
        )
        for axis in range(dimension)
    )
    sqrt_value, sqrt_tangent = _affine_automaton_contraction_jvp(
        density.sqrt_tt.cores,
        core_directions,
        axis_operators,
        first,
        _finite_scalar(first_offset, "first_offset"),
        first_power,
        dot_first,
        _finite_scalar(dot_first_offset, "dot_first_offset"),
        second,
        _finite_scalar(second_offset, "second_offset"),
        second_power,
        dot_second,
        _finite_scalar(dot_second_offset, "dot_second_offset"),
    )
    defensive_value, defensive_tangent = _defensive_affine_moment_jvp(
        density,
        axis_operators,
        first,
        _finite_scalar(first_offset, "first_offset"),
        first_power,
        dot_first,
        _finite_scalar(dot_first_offset, "dot_first_offset"),
        second,
        _finite_scalar(second_offset, "second_offset"),
        second_power,
        dot_second,
        _finite_scalar(dot_second_offset, "dot_second_offset"),
    )
    tau_dot = _finite_scalar(dot_tau, "dot_tau")
    numerator = sqrt_value + density.tau * defensive_value
    numerator_tangent = (
        sqrt_tangent + tau_dot * defensive_value + density.tau * defensive_tangent
    )
    mass_operators = tuple(
        basis.mass_matrix(density.measure_convention.mass_measure)
        for basis in density.sqrt_tt.product_basis.bases
    )
    _, dot_sqrt_mass = _paired_contraction_jvp(
        density.sqrt_tt.cores,
        core_directions,
        mass_operators,
        tuple(tf.zeros_like(operator) for operator in mass_operators),
    )
    q0_mass = density.defensive_density.normalizer(
        density.measure_convention.mass_measure
    )
    normalizer = density.normalizer()
    normalizer_tangent = dot_sqrt_mass + tau_dot * q0_mass
    value = numerator / normalizer
    tangent = (
        numerator_tangent * normalizer - numerator * normalizer_tangent
    ) / tf.square(normalizer)
    _assert_finite(value, "affine-form moment")
    _assert_finite(tangent, "affine-form moment tangent")
    return TTObservableJVP(
        value=value,
        tangent=tangent,
        numerator=numerator,
        numerator_tangent=numerator_tangent,
        normalizer=normalizer,
        normalizer_tangent=normalizer_tangent,
    )


def squared_tt_affine_form_moment(
    density: SquaredTTDensity,
    first_coefficients: tf.Tensor,
    first_offset: tf.Tensor,
    first_power: int,
    *,
    second_coefficients: tf.Tensor | None = None,
    second_offset: tf.Tensor = tf.constant(0.0, tf.float64),
    second_power: int = 0,
) -> tf.Tensor:
    """Return one normalized affine-form mixed moment."""

    zero_cores = tuple(TTCore(tf.zeros_like(core.values)) for core in density.sqrt_tt.cores)
    return squared_tt_affine_form_moment_jvp(
        density,
        first_coefficients,
        first_offset,
        first_power,
        zero_cores,
        second_coefficients=second_coefficients,
        second_offset=second_offset,
        second_power=second_power,
    ).value


def frozen_squared_tt_shape_targets(
    density: SquaredTTDensity,
    state_offset: tf.Tensor,
    state_matrix: tf.Tensor,
    *,
    pair_indices: Sequence[tuple[int, int]] = (),
    parameter_count: int,
) -> FrozenTTShapeTargets:
    """Build standardized shape targets from one frozen affine-chart TT density.

    This is a diagnostic composition.  The returned target tangents are zero
    because the supplied density and chart are explicitly frozen.  They are
    not the tangents of a recursively refitted Zhao-Cui teacher.
    """

    offset = tf.convert_to_tensor(state_offset, tf.float64)
    matrix = tf.convert_to_tensor(state_matrix, tf.float64)
    tt_dimension = len(density.sqrt_tt.cores)
    if offset.shape.rank != 1:
        raise ValueError("state_offset must be a vector")
    state_dimension = int(offset.shape[0])
    if matrix.shape != (state_dimension, tt_dimension):
        raise ValueError("state_matrix must have shape [state_dimension,tt_dimension]")
    if int(parameter_count) < 1:
        raise ValueError("parameter_count must be positive")
    reference = squared_tt_reference_moments(density, tuple(range(tt_dimension)))
    state_mean = offset + tf.linalg.matvec(matrix, reference.mean)
    state_covariance = matrix @ reference.covariance @ tf.transpose(matrix)
    state_covariance = 0.5 * (state_covariance + tf.transpose(state_covariance))
    chol = tf.linalg.cholesky(state_covariance)
    # Column-vector whitening z=L^{-1}(x-mu) gives coefficients L^{-1} B.
    standardized_matrix = tf.linalg.triangular_solve(chol, matrix)
    standardized_offset = tf.linalg.triangular_solve(
        chol,
        tf.reshape(offset - state_mean, [state_dimension, 1]),
    )[:, 0]
    skew = []
    kurtosis = []
    for axis in range(state_dimension):
        coefficients = standardized_matrix[axis]
        local_offset = standardized_offset[axis]
        skew.append(
            squared_tt_affine_form_moment(
                density, coefficients, local_offset, 3
            )
        )
        kurtosis.append(
            squared_tt_affine_form_moment(
                density, coefficients, local_offset, 4
            )
        )
    skew_tensor = tf.stack(skew)
    kurtosis_tensor = tf.stack(kurtosis)
    co_skew = tf.zeros([state_dimension, state_dimension], tf.float64)
    co_kurtosis = tf.zeros_like(co_skew)
    co_skew_mask = tf.zeros_like(co_skew)
    co_kurtosis_mask = tf.zeros_like(co_skew)
    updates3 = []
    co_skew_indices = []
    co_kurtosis_indices = []
    co_kurtosis_updates = []
    seen_ordered_pairs: set[tuple[int, int]] = set()
    seen_unordered_pairs: set[tuple[int, int]] = set()
    for left, right in tuple((int(i), int(j)) for i, j in pair_indices):
        if left == right or left < 0 or right < 0:
            raise ValueError("pair indices must be distinct nonnegative coordinates")
        if left >= state_dimension or right >= state_dimension:
            raise IndexError("pair index out of range")
        if (left, right) in seen_ordered_pairs:
            raise ValueError("pair_indices must not contain duplicate ordered pairs")
        seen_ordered_pairs.add((left, right))
        co_skew_indices.append([left, right])
        updates3.append(
            squared_tt_affine_form_moment(
                density,
                standardized_matrix[left],
                standardized_offset[left],
                2,
                second_coefficients=standardized_matrix[right],
                second_offset=standardized_offset[right],
                second_power=1,
            )
        )
        unordered_pair = (min(left, right), max(left, right))
        if unordered_pair in seen_unordered_pairs:
            continue
        seen_unordered_pairs.add(unordered_pair)
        co_kurtosis_value = squared_tt_affine_form_moment(
            density,
            standardized_matrix[left],
            standardized_offset[left],
            2,
            second_coefficients=standardized_matrix[right],
            second_offset=standardized_offset[right],
            second_power=2,
        )
        co_kurtosis_indices.extend(([left, right], [right, left]))
        co_kurtosis_updates.extend((co_kurtosis_value, co_kurtosis_value))
    if co_skew_indices:
        co_skew = tf.tensor_scatter_nd_update(
            co_skew, co_skew_indices, tf.stack(updates3)
        )
        co_kurtosis = tf.tensor_scatter_nd_update(
            co_kurtosis, co_kurtosis_indices, tf.stack(co_kurtosis_updates)
        )
        co_skew_mask = tf.tensor_scatter_nd_update(
            co_skew_mask,
            co_skew_indices,
            tf.ones([len(co_skew_indices)], tf.float64),
        )
        co_kurtosis_mask = tf.tensor_scatter_nd_update(
            co_kurtosis_mask,
            co_kurtosis_indices,
            tf.ones([len(co_kurtosis_indices)], tf.float64),
        )
    parameter_count = int(parameter_count)
    return FrozenTTShapeTargets(
        skew=skew_tensor,
        kurtosis=kurtosis_tensor,
        pairwise_co_skew=co_skew,
        pairwise_co_kurtosis=co_kurtosis,
        pairwise_co_skew_mask=co_skew_mask,
        pairwise_co_kurtosis_mask=co_kurtosis_mask,
        skew_tangent=tf.zeros([state_dimension, parameter_count], tf.float64),
        kurtosis_tangent=tf.zeros([state_dimension, parameter_count], tf.float64),
        pairwise_co_skew_tangent=tf.zeros(
            [state_dimension, state_dimension, parameter_count], tf.float64
        ),
        pairwise_co_kurtosis_tangent=tf.zeros(
            [state_dimension, state_dimension, parameter_count], tf.float64
        ),
    )


def _cholesky_jvp_single(chol: tf.Tensor, covariance_tangent: tf.Tensor) -> tf.Tensor:
    """Manual lower-Cholesky JVP for one symmetric covariance direction."""

    left = tf.linalg.triangular_solve(chol, covariance_tangent)
    inner = tf.linalg.triangular_solve(chol, tf.transpose(left))
    lower = tf.linalg.band_part(inner, -1, 0)
    phi = lower - 0.5 * tf.linalg.diag(tf.linalg.diag_part(lower))
    return chol @ phi


def squared_tt_shape_targets_jvp(
    density: SquaredTTDensity,
    state_offset: tf.Tensor,
    state_matrix: tf.Tensor,
    dot_cores: Sequence[TTCore],
    *,
    pair_indices: Sequence[tuple[int, int]] = (),
    dot_state_offset: tf.Tensor | None = None,
    dot_state_matrix: tf.Tensor | None = None,
    dot_tau: tf.Tensor = tf.constant(0.0, tf.float64),
) -> TTShapeTargetsJVP:
    """Compute TT standardized shape targets and a manual one-direction JVP."""

    offset = tf.convert_to_tensor(state_offset, tf.float64)
    matrix = tf.convert_to_tensor(state_matrix, tf.float64)
    dimension = len(density.sqrt_tt.cores)
    if offset.shape.rank != 1 or matrix.shape.rank != 2:
        raise ValueError("state chart must be a vector and matrix")
    state_dimension = int(offset.shape[0])
    if matrix.shape != (state_dimension, dimension):
        raise ValueError("state_matrix must have shape [state_dimension,tt_dimension]")
    dot_offset = (
        tf.zeros_like(offset)
        if dot_state_offset is None
        else tf.convert_to_tensor(dot_state_offset, tf.float64)
    )
    dot_matrix = (
        tf.zeros_like(matrix)
        if dot_state_matrix is None
        else tf.convert_to_tensor(dot_state_matrix, tf.float64)
    )
    if dot_offset.shape != offset.shape or dot_matrix.shape != matrix.shape:
        raise ValueError("chart tangent shapes must match chart shapes")
    dot_tau_value = _finite_scalar(dot_tau, "dot_tau")
    means = []
    mean_tangents = []
    for axis in range(dimension):
        powers = [0] * dimension
        powers[axis] = 1
        observable = squared_tt_separable_observable_jvp(
            density,
            monomial_operator_matrices(density, powers),
            dot_cores,
            defensive_moment=tensor_product_reference_monomial_moment(density, powers),
            dot_tau=dot_tau_value,
        )
        means.append(observable.value)
        mean_tangents.append(observable.tangent)
    mean = tf.stack(means)
    mean_tangent = tf.stack(mean_tangents)
    raw_second_rows = []
    raw_second_tangent_rows = []
    for left_axis in range(dimension):
        row = []
        tangent_row = []
        for right_axis in range(dimension):
            powers = [0] * dimension
            powers[left_axis] += 1
            powers[right_axis] += 1
            observable = squared_tt_separable_observable_jvp(
                density,
                monomial_operator_matrices(density, powers),
                dot_cores,
                defensive_moment=tensor_product_reference_monomial_moment(density, powers),
                dot_tau=dot_tau_value,
            )
            row.append(observable.value)
            tangent_row.append(observable.tangent)
        raw_second_rows.append(tf.stack(row))
        raw_second_tangent_rows.append(tf.stack(tangent_row))
    raw_second = tf.stack(raw_second_rows)
    raw_second_tangent = tf.stack(raw_second_tangent_rows)
    covariance = 0.5 * (
        raw_second - mean[:, None] * mean[None, :]
        + tf.transpose(raw_second - mean[:, None] * mean[None, :])
    )
    covariance_tangent_raw = (
        raw_second_tangent
        - mean_tangent[:, None] * mean[None, :]
        - mean[:, None] * mean_tangent[None, :]
    )
    covariance_tangent = 0.5 * (
        covariance_tangent_raw + tf.transpose(covariance_tangent_raw)
    )
    state_mean = offset + tf.linalg.matvec(matrix, mean)
    state_mean_tangent = dot_offset + tf.linalg.matvec(dot_matrix, mean) + tf.linalg.matvec(matrix, mean_tangent)
    state_covariance_raw = matrix @ covariance @ tf.transpose(matrix)
    state_covariance = 0.5 * (
        state_covariance_raw + tf.transpose(state_covariance_raw)
    )
    state_covariance_tangent_raw = (
        dot_matrix @ covariance @ tf.transpose(matrix)
        + matrix @ covariance_tangent @ tf.transpose(matrix)
        + matrix @ covariance @ tf.transpose(dot_matrix)
    )
    state_covariance_tangent = 0.5 * (
        state_covariance_tangent_raw + tf.transpose(state_covariance_tangent_raw)
    )
    chol = tf.linalg.cholesky(state_covariance)
    chol_tangent = _cholesky_jvp_single(chol, state_covariance_tangent)
    standardized_matrix = tf.linalg.triangular_solve(chol, matrix)
    standardized_matrix_tangent = tf.linalg.triangular_solve(
        chol,
        dot_matrix - chol_tangent @ standardized_matrix,
    )
    standardized_offset = tf.linalg.triangular_solve(
        chol, tf.reshape(offset - state_mean, [state_dimension, 1])
    )[:, 0]
    standardized_offset_tangent = tf.linalg.triangular_solve(
        chol,
        tf.reshape(dot_offset - state_mean_tangent, [state_dimension, 1])
        - chol_tangent @ tf.reshape(standardized_offset, [state_dimension, 1]),
    )[:, 0]

    def moment_jvp(left: int, left_power: int, right: int | None = None, right_power: int = 0):
        return squared_tt_affine_form_moment_jvp(
            density,
            standardized_matrix[left],
            standardized_offset[left],
            left_power,
            dot_cores,
            dot_first_coefficients=standardized_matrix_tangent[left],
            dot_first_offset=standardized_offset_tangent[left],
            second_coefficients=(standardized_matrix[right] if right is not None else None),
            second_offset=(standardized_offset[right] if right is not None else tf.constant(0.0, tf.float64)),
            second_power=right_power,
            dot_second_coefficients=(standardized_matrix_tangent[right] if right is not None else None),
            dot_second_offset=(standardized_offset_tangent[right] if right is not None else tf.constant(0.0, tf.float64)),
            dot_tau=dot_tau_value,
        )

    skew_values = []
    kurt_values = []
    skew_tangents = []
    kurt_tangents = []
    for axis in range(state_dimension):
        skew = moment_jvp(axis, 3)
        kurt = moment_jvp(axis, 4)
        skew_values.append(skew.value)
        kurt_values.append(kurt.value)
        skew_tangents.append(skew.tangent)
        kurt_tangents.append(kurt.tangent)
    co_skew = tf.zeros([state_dimension, state_dimension], tf.float64)
    co_kurtosis = tf.zeros_like(co_skew)
    co_skew_tangent = tf.zeros_like(co_skew)
    co_kurtosis_tangent = tf.zeros_like(co_skew)
    skew_indices = []
    skew_values_updates = []
    skew_tangent_updates = []
    kurt_indices = []
    kurt_values_updates = []
    kurt_tangent_updates = []
    seen_ordered: set[tuple[int, int]] = set()
    seen_unordered: set[tuple[int, int]] = set()
    for left, right in tuple((int(i), int(j)) for i, j in pair_indices):
        if left == right or left < 0 or right < 0 or left >= state_dimension or right >= state_dimension:
            raise ValueError("pair indices must be distinct in-range coordinates")
        if (left, right) in seen_ordered:
            raise ValueError("pair_indices must not contain duplicate ordered pairs")
        seen_ordered.add((left, right))
        value = moment_jvp(left, 2, right, 1)
        skew_indices.append([left, right])
        skew_values_updates.append(value.value)
        skew_tangent_updates.append(value.tangent)
        unordered = (min(left, right), max(left, right))
        if unordered not in seen_unordered:
            seen_unordered.add(unordered)
            value = moment_jvp(left, 2, right, 2)
            kurt_indices.extend(([left, right], [right, left]))
            kurt_values_updates.extend((value.value, value.value))
            kurt_tangent_updates.extend((value.tangent, value.tangent))
    skew_mask = tf.zeros_like(co_skew)
    kurt_mask = tf.zeros_like(co_kurtosis)
    if skew_indices:
        co_skew = tf.tensor_scatter_nd_update(co_skew, skew_indices, tf.stack(skew_values_updates))
        co_skew_tangent = tf.tensor_scatter_nd_update(co_skew_tangent, skew_indices, tf.stack(skew_tangent_updates))
        skew_mask = tf.tensor_scatter_nd_update(skew_mask, skew_indices, tf.ones([len(skew_indices)], tf.float64))
    if kurt_indices:
        co_kurtosis = tf.tensor_scatter_nd_update(co_kurtosis, kurt_indices, tf.stack(kurt_values_updates))
        co_kurtosis_tangent = tf.tensor_scatter_nd_update(co_kurtosis_tangent, kurt_indices, tf.stack(kurt_tangent_updates))
        kurt_mask = tf.tensor_scatter_nd_update(kurt_mask, kurt_indices, tf.ones([len(kurt_indices)], tf.float64))
    return TTShapeTargetsJVP(
        tf.stack(skew_values),
        tf.stack(kurt_values),
        co_skew,
        co_kurtosis,
        skew_mask,
        kurt_mask,
        tf.stack(skew_tangents)[:, None],
        tf.stack(kurt_tangents)[:, None],
        co_skew_tangent[:, :, None],
        co_kurtosis_tangent[:, :, None],
    )


def stack_squared_tt_shape_targets_jvp(
    density: SquaredTTDensity,
    state_offset: tf.Tensor,
    state_matrix: tf.Tensor,
    tangent_cores: Sequence[Sequence[tf.Tensor | TTCore]],
    *,
    pair_indices: Sequence[tuple[int, int]] = (),
) -> TTShapeTargetsJVP:
    """Assemble all parameter directions from a Lane-B-style tangent bank."""

    banks = tuple(tuple(bank) for bank in tangent_cores)
    if len(banks) != len(density.sqrt_tt.cores) or not banks:
        raise ValueError("tangent bank must have one row per TT core")
    parameter_count = len(banks[0])
    if parameter_count < 1 or any(len(bank) != parameter_count for bank in banks):
        raise ValueError("tangent bank must have a fixed positive parameter count")
    directions = []
    for parameter in range(parameter_count):
        direction = tuple(
            value if isinstance(value, TTCore) else TTCore(tf.convert_to_tensor(value, tf.float64))
            for value in (bank[parameter] for bank in banks)
        )
        directions.append(
            squared_tt_shape_targets_jvp(
                density,
                state_offset,
                state_matrix,
                direction,
                pair_indices=pair_indices,
            )
        )
    reference = directions[0]
    for candidate in directions[1:]:
        for name in (
            "skew",
            "kurtosis",
            "pairwise_co_skew",
            "pairwise_co_kurtosis",
            "pairwise_co_skew_mask",
            "pairwise_co_kurtosis_mask",
        ):
            tf.debugging.assert_near(
                getattr(candidate, name),
                getattr(reference, name),
                atol=tf.constant(2.0e-12, tf.float64),
                rtol=tf.constant(2.0e-12, tf.float64),
                message=f"parameter-direction target values disagree: {name}",
            )
    return TTShapeTargetsJVP(
        skew=reference.skew,
        kurtosis=reference.kurtosis,
        pairwise_co_skew=reference.pairwise_co_skew,
        pairwise_co_kurtosis=reference.pairwise_co_kurtosis,
        pairwise_co_skew_mask=reference.pairwise_co_skew_mask,
        pairwise_co_kurtosis_mask=reference.pairwise_co_kurtosis_mask,
        skew_tangent=tf.concat([item.skew_tangent for item in directions], axis=1),
        kurtosis_tangent=tf.concat(
            [item.kurtosis_tangent for item in directions], axis=1
        ),
        pairwise_co_skew_tangent=tf.concat(
            [item.pairwise_co_skew_tangent for item in directions], axis=2
        ),
        pairwise_co_kurtosis_tangent=tf.concat(
            [item.pairwise_co_kurtosis_tangent for item in directions], axis=2
        ),
    )


def apply_tt_shape_targets_reference_jvp(
    source: tf.Tensor,
    weights: tf.Tensor,
    source_tangent: tf.Tensor,
    weights_tangent: tf.Tensor,
    points: tf.Tensor,
    points_tangent: tf.Tensor,
    targets: TTShapeTargetsJVP,
    *,
    correction_steps: int,
    strength: float,
    floor: float,
    pairwise_correction_steps: int = 0,
    pairwise_strength: float = 0.0,
    pairwise_floor: float = 1.0e-6,
) -> dict[str, tf.Tensor]:
    """Apply TT shape values/tangents through the existing Contract E map.

    This is the setup-static FP64 reference composition.  It does not compute
    the particle likelihood or OT transport; callers must pass the cloud after
    those stages and keep their scalar increment separate.
    """

    return higher_moment_shape_jvp(
        source,
        weights,
        source_tangent,
        weights_tangent,
        points,
        points_tangent,
        correction_steps=correction_steps,
        strength=strength,
        floor=floor,
        pairwise_correction_steps=pairwise_correction_steps,
        pairwise_strength=pairwise_strength,
        pairwise_floor=pairwise_floor,
        **targets.explicit_target_kwargs(),
    )


def tt_particle_contract_e_step_reference_jvp(
    source: tf.Tensor,
    normalized_weights: tf.Tensor,
    source_tangent: tf.Tensor,
    normalized_weights_tangent: tf.Tensor,
    particle_log_increment: tf.Tensor,
    particle_log_increment_tangent: tf.Tensor,
    residual_design: tf.Tensor,
    targets: TTShapeTargetsJVP,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int,
    ridge: float,
    correction_steps: int,
    strength: float,
    floor: float,
    pairwise_correction_steps: int = 0,
    pairwise_strength: float = 0.0,
    pairwise_floor: float = 1.0e-6,
) -> TTParticleContractEStepJVP:
    """Compose the particle scalar, OT, Contract E, and TT shape correction.

    The likelihood increment and its tangent are inputs owned by the particle
    model adapter and are returned exactly unchanged.  This setup-static FP64
    reference reuses the established OT/Contract E JVP and is not an XLA route.
    """

    # Local import avoids making the standalone TT contraction module own the
    # generic particle-filter implementation.
    from bayesfilter.highdim.cubature_genut_filter import _restore_cloud_jvp_core

    source = tf.convert_to_tensor(source, tf.float64)
    weights = tf.convert_to_tensor(normalized_weights, tf.float64)
    source_dot = tf.convert_to_tensor(source_tangent, tf.float64)
    weights_dot = tf.convert_to_tensor(normalized_weights_tangent, tf.float64)
    increment = _finite_scalar(particle_log_increment, "particle_log_increment")
    increment_dot = tf.convert_to_tensor(particle_log_increment_tangent, tf.float64)
    design = tf.convert_to_tensor(residual_design, tf.float64)
    if source.shape.rank != 2 or source_dot.shape.rank != 3:
        raise ValueError("source and source tangent must have ranks two and three")
    if source_dot.shape[:2] != source.shape:
        raise ValueError("source tangent leading dimensions must match source")
    if weights.shape != source.shape[:1] or weights_dot.shape[:1] != weights.shape:
        raise ValueError("weight shapes must match source rows")
    parameter_count = int(source_dot.shape[-1])
    if increment_dot.shape != (parameter_count,) or weights_dot.shape != (
        int(source.shape[0]),
        parameter_count,
    ):
        raise ValueError("particle increment/weight tangent parameter shapes mismatch")
    restored = _restore_cloud_jvp_core(
        source,
        weights,
        source_dot,
        weights_dot,
        design,
        epsilon=epsilon,
        sinkhorn_steps=sinkhorn_steps,
        balance_steps=balance_steps,
        ridge=ridge,
        parameter_count=parameter_count,
    )
    if not bool(restored["reset_valid"].numpy()):
        raise ValueError("particle OT/Contract E reference reset is invalid")
    corrected = apply_tt_shape_targets_reference_jvp(
        source,
        weights,
        source_dot,
        weights_dot,
        restored["particles"],
        restored["particles_tangent"],
        targets,
        correction_steps=correction_steps,
        strength=strength,
        floor=floor,
        pairwise_correction_steps=pairwise_correction_steps,
        pairwise_strength=pairwise_strength,
        pairwise_floor=pairwise_floor,
    )
    if not bool(corrected["valid"].numpy()):
        raise ValueError("TT higher-moment correction reference is invalid")
    return TTParticleContractEStepJVP(
        increment,
        increment_dot,
        corrected["particles"],
        corrected["particles_tangent"],
        {
            "reset_valid": restored["reset_valid"],
            "mean_residual": restored["mean_residual"],
            "maximum_raw_row_residual": restored["maximum_raw_row_residual"],
            "maximum_post_quotient_column_residual": restored[
                "maximum_post_quotient_column_residual"
            ],
            "shape_valid": corrected["valid"],
            "skew_residual": corrected["skew_residual"],
            "kurtosis_residual": corrected["kurtosis_residual"],
            "pairwise_co_skew_residual": corrected["pairwise_co_skew_residual"],
            "pairwise_co_kurtosis_residual": corrected[
                "pairwise_co_kurtosis_residual"
            ],
        },
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def padded_squared_tt_observable_jvp_xla(
    cores: tf.Tensor,
    dot_cores: tf.Tensor,
    observable_operators: tf.Tensor,
    dot_observable_operators: tf.Tensor,
    mass_operators: tf.Tensor,
    dot_mass_operators: tf.Tensor,
    tau: tf.Tensor,
    dot_tau: tf.Tensor,
    defensive_observable: tf.Tensor,
    dot_defensive_observable: tf.Tensor,
    defensive_mass: tf.Tensor,
    dot_defensive_mass: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Graph-native normalized observable JVP for padded equal-rank TT cores.

    ``cores`` has shape ``[axis, rank, basis, rank]``. Boundary and inactive
    rank channels must be exactly zero-padded, with the physical scalar path at
    rank index zero. This kernel performs no fitting and owns no branch policy;
    it is the XLA contraction primitive used after setup-time padding.
    """

    cores = tf.convert_to_tensor(cores)
    dot_cores = tf.convert_to_tensor(dot_cores, dtype=cores.dtype)
    observable_operators = tf.convert_to_tensor(
        observable_operators, dtype=cores.dtype
    )
    dot_observable_operators = tf.convert_to_tensor(
        dot_observable_operators, dtype=cores.dtype
    )
    mass_operators = tf.convert_to_tensor(mass_operators, dtype=cores.dtype)
    dot_mass_operators = tf.convert_to_tensor(
        dot_mass_operators, dtype=cores.dtype
    )
    if cores.shape.rank != 4 or cores.shape[1] is None:
        raise ValueError("padded XLA cores require a static rank dimension")
    axis_count = tf.shape(cores)[0]
    rank = int(cores.shape[1])
    pair_rank = rank * rank
    initial = tf.one_hot(0, pair_rank, dtype=cores.dtype)

    def contract(
        operators: tf.Tensor, dot_operators: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        def body(
            axis: tf.Tensor,
            vector: tf.Tensor,
            dot_vector: tf.Tensor,
        ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            core = cores[axis]
            dot_core = dot_cores[axis]
            operator = operators[axis]
            dot_operator = dot_operators[axis]
            paired = tf.einsum("alb,AmB,lm->aAbB", core, core, operator)
            dot_paired = (
                tf.einsum("alb,AmB,lm->aAbB", dot_core, core, operator)
                + tf.einsum("alb,AmB,lm->aAbB", core, dot_core, operator)
                + tf.einsum("alb,AmB,lm->aAbB", core, core, dot_operator)
            )
            matrix = tf.reshape(paired, [pair_rank, pair_rank])
            dot_matrix = tf.reshape(dot_paired, [pair_rank, pair_rank])
            next_dot = tf.linalg.matvec(
                matrix, dot_vector, transpose_a=True
            ) + tf.linalg.matvec(dot_matrix, vector, transpose_a=True)
            next_vector = tf.linalg.matvec(matrix, vector, transpose_a=True)
            return (
                axis + 1,
                tf.ensure_shape(next_vector, [pair_rank]),
                tf.ensure_shape(next_dot, [pair_rank]),
            )

        _, vector, dot_vector = tf.while_loop(
            lambda axis, *_: axis < axis_count,
            body,
            (tf.zeros([], tf.int32), initial, tf.zeros_like(initial)),
            parallel_iterations=1,
        )
        return vector[0], dot_vector[0]

    sqrt_observable, dot_sqrt_observable = contract(
        observable_operators, dot_observable_operators
    )
    sqrt_mass, dot_sqrt_mass = contract(mass_operators, dot_mass_operators)
    tau = tf.cast(tau, cores.dtype)
    dot_tau = tf.cast(dot_tau, cores.dtype)
    defensive_observable = tf.cast(defensive_observable, cores.dtype)
    dot_defensive_observable = tf.cast(dot_defensive_observable, cores.dtype)
    defensive_mass = tf.cast(defensive_mass, cores.dtype)
    dot_defensive_mass = tf.cast(dot_defensive_mass, cores.dtype)
    numerator = sqrt_observable + tau * defensive_observable
    dot_numerator = (
        dot_sqrt_observable
        + dot_tau * defensive_observable
        + tau * dot_defensive_observable
    )
    normalizer = sqrt_mass + tau * defensive_mass
    dot_normalizer = (
        dot_sqrt_mass + dot_tau * defensive_mass + tau * dot_defensive_mass
    )
    value = numerator / normalizer
    tangent = (
        dot_numerator * normalizer - numerator * dot_normalizer
    ) / tf.square(normalizer)
    return value, tangent, normalizer, dot_normalizer


def _affine_automaton_contraction_jvp(
    cores: Sequence[TTCore],
    dot_cores: Sequence[TTCore],
    axis_operators: Sequence[Sequence[tf.Tensor]],
    first: tf.Tensor,
    first_offset: tf.Tensor,
    first_power: int,
    dot_first: tf.Tensor,
    dot_first_offset: tf.Tensor,
    second: tf.Tensor,
    second_offset: tf.Tensor,
    second_power: int,
    dot_second: tf.Tensor,
    dot_second_offset: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    auto_size = (first_power + 1) * (second_power + 1)
    initial = []
    dot_initial = []
    for first_degree in range(first_power + 1):
        for second_degree in range(second_power + 1):
            value = tf.pow(first_offset, first_degree) * tf.pow(
                second_offset, second_degree
            )
            tangent = (
                _power_jvp(first_offset, dot_first_offset, first_degree)
                * tf.pow(second_offset, second_degree)
                + tf.pow(first_offset, first_degree)
                * _power_jvp(second_offset, dot_second_offset, second_degree)
            )
            initial.append(value)
            dot_initial.append(tangent)
    state = tf.reshape(tf.stack(initial), [auto_size, 1])
    dot_state = tf.reshape(tf.stack(dot_initial), [auto_size, 1])

    for axis, (core, dot_core) in enumerate(zip(cores, dot_cores)):
        local_rows = []
        dot_local_rows = []
        for out_first in range(first_power + 1):
            for out_second in range(second_power + 1):
                row = []
                dot_row = []
                for in_first in range(first_power + 1):
                    for in_second in range(second_power + 1):
                        if in_first > out_first or in_second > out_second:
                            paired = _zero_paired_matrix(core)
                            dot_paired = tf.zeros_like(paired)
                        else:
                            delta_first = out_first - in_first
                            delta_second = out_second - in_second
                            power = delta_first + delta_second
                            coefficient = tf.cast(
                                math.comb(out_first, in_first)
                                * math.comb(out_second, in_second),
                                tf.float64,
                            )
                            coefficient = (
                                coefficient
                                * tf.pow(first[axis], delta_first)
                                * tf.pow(second[axis], delta_second)
                            )
                            coefficient_tangent = tf.cast(
                                math.comb(out_first, in_first)
                                * math.comb(out_second, in_second),
                                tf.float64,
                            ) * (
                                _power_jvp(
                                    first[axis], dot_first[axis], delta_first
                                )
                                * tf.pow(second[axis], delta_second)
                                + tf.pow(first[axis], delta_first)
                                * _power_jvp(
                                    second[axis], dot_second[axis], delta_second
                                )
                            )
                            base = _paired_core_matrix(
                                core, core, axis_operators[axis][power]
                            )
                            dot_base = _paired_core_matrix(
                                dot_core, core, axis_operators[axis][power]
                            ) + _paired_core_matrix(
                                core, dot_core, axis_operators[axis][power]
                            )
                            paired = coefficient * base
                            dot_paired = coefficient_tangent * base + coefficient * dot_base
                        row.append(paired)
                        dot_row.append(dot_paired)
                local_rows.append(tf.stack(row, axis=0))
                dot_local_rows.append(tf.stack(dot_row, axis=0))
        local = tf.stack(local_rows, axis=0)
        dot_local = tf.stack(dot_local_rows, axis=0)
        dot_state = tf.einsum("ul,vulr->vr", dot_state, local) + tf.einsum(
            "ul,vulr->vr", state, dot_local
        )
        state = tf.einsum("ul,vulr->vr", state, local)
    output_index = first_power * (second_power + 1) + second_power
    return state[output_index, 0], dot_state[output_index, 0]


def _defensive_affine_moment_jvp(
    density: SquaredTTDensity,
    axis_operators: Sequence[Sequence[tf.Tensor]],
    first: tf.Tensor,
    first_offset: tf.Tensor,
    first_power: int,
    dot_first: tf.Tensor,
    dot_first_offset: tf.Tensor,
    second: tf.Tensor,
    second_offset: tf.Tensor,
    second_power: int,
    dot_second: tf.Tensor,
    dot_second_offset: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    if not isinstance(density.defensive_density, TensorProductReferenceDensity):
        raise NotImplementedError(
            "affine-form defensive moment requires TensorProductReferenceDensity"
        )
    scalar_cores = tuple(
        TTCore(tf.ones([1, 1, 1], tf.float64)) for _ in density.sqrt_tt.cores
    )
    zero_cores = tuple(TTCore(tf.zeros([1, 1, 1], tf.float64)) for _ in scalar_cores)
    scalar_operators = []
    for axis, powers in enumerate(axis_operators):
        constant = _constant_basis_coefficients(
            density.sqrt_tt.product_basis.bases[axis]
        )
        scalar_operators.append(
            tuple(
                tf.reshape(tf.einsum("l,lm,m->", constant, operator, constant), [1, 1])
                for operator in powers
            )
        )
    value, tangent = _affine_automaton_contraction_jvp(
        scalar_cores,
        zero_cores,
        tuple(scalar_operators),
        first,
        first_offset,
        first_power,
        dot_first,
        dot_first_offset,
        second,
        second_offset,
        second_power,
        dot_second,
        dot_second_offset,
    )
    scale = tf.constant(1.0, tf.float64) + density.defensive_density.floor
    return scale * value, scale * tangent


def _paired_contraction(
    cores: Sequence[TTCore],
    operators: Sequence[tf.Tensor],
) -> tf.Tensor:
    value, _ = _paired_contraction_jvp(
        cores,
        tuple(TTCore(tf.zeros_like(core.values)) for core in cores),
        operators,
        tuple(tf.zeros_like(operator) for operator in operators),
    )
    return value


def _paired_contraction_jvp(
    cores: Sequence[TTCore],
    dot_cores: Sequence[TTCore],
    operators: Sequence[tf.Tensor],
    dot_operators: Sequence[tf.Tensor],
) -> tuple[tf.Tensor, tf.Tensor]:
    vector = tf.ones([1], tf.float64)
    dot_vector = tf.zeros([1], tf.float64)
    for core, dot_core, operator, dot_operator in zip(
        cores, dot_cores, operators, dot_operators
    ):
        matrix = _paired_core_matrix(core, core, operator)
        dot_matrix = (
            _paired_core_matrix(dot_core, core, operator)
            + _paired_core_matrix(core, dot_core, operator)
            + _paired_core_matrix(core, core, dot_operator)
        )
        dot_vector = tf.einsum("a,ab->b", dot_vector, matrix) + tf.einsum(
            "a,ab->b", vector, dot_matrix
        )
        vector = tf.einsum("a,ab->b", vector, matrix)
    return tf.reshape(vector, []), tf.reshape(dot_vector, [])


def _paired_core_matrix(
    left_core: TTCore,
    right_core: TTCore,
    operator: tf.Tensor,
) -> tf.Tensor:
    matrix = tf.convert_to_tensor(operator, tf.float64)
    if matrix.shape != (left_core.basis_dim, right_core.basis_dim):
        raise ValueError("operator matrix does not match core basis dimensions")
    paired = tf.einsum(
        "alb,AmB,lm->aAbB", left_core.values, right_core.values, matrix
    )
    return tf.reshape(
        paired,
        [
            left_core.left_rank * right_core.left_rank,
            left_core.right_rank * right_core.right_rank,
        ],
    )


def _zero_paired_matrix(core: TTCore) -> tf.Tensor:
    return tf.zeros(
        [core.left_rank * core.left_rank, core.right_rank * core.right_rank],
        tf.float64,
    )


def _operator_tuple(
    density: SquaredTTDensity,
    operators: Sequence[tf.Tensor],
) -> tuple[tf.Tensor, ...]:
    result = tuple(tf.convert_to_tensor(operator, tf.float64) for operator in operators)
    if len(result) != len(density.sqrt_tt.cores):
        raise ValueError("operator count must match TT dimension")
    for core, operator in zip(density.sqrt_tt.cores, result):
        if operator.shape != (core.basis_dim, core.basis_dim):
            raise ValueError("operator matrix has the wrong shape")
        _assert_finite(operator, "operator matrix")
    return result


def _core_direction_tuple(
    density: SquaredTTDensity,
    dot_cores: Sequence[TTCore],
) -> tuple[TTCore, ...]:
    result = tuple(dot_cores)
    if len(result) != len(density.sqrt_tt.cores):
        raise ValueError("core direction count must match TT dimension")
    for core, dot_core in zip(density.sqrt_tt.cores, result):
        if dot_core.values.shape != core.values.shape:
            raise ValueError("core direction shape mismatch")
    return result


def _coefficient_vector(value: tf.Tensor, dimension: int, name: str) -> tf.Tensor:
    result = tf.convert_to_tensor(value, tf.float64)
    if result.shape != (dimension,):
        raise ValueError(f"{name} must have shape [dimension]")
    _assert_finite(result, name)
    return result


def _power_jvp(value: tf.Tensor, tangent: tf.Tensor, power: int) -> tf.Tensor:
    if int(power) == 0:
        return tf.constant(0.0, tf.float64)
    return tf.cast(power, tf.float64) * tf.pow(value, int(power) - 1) * tangent


def _finite_scalar(value: tf.Tensor, name: str) -> tf.Tensor:
    result = tf.convert_to_tensor(value, tf.float64)
    if result.shape.rank != 0:
        raise ValueError(f"{name} must be scalar")
    _assert_finite(result, name)
    return result


def _assert_finite(value: tf.Tensor, name: str) -> None:
    if not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
        raise ValueError(f"{name}: {HighDimStatus.NONFINITE_VALUE.value}")


def _legendre_gauss_nodes_weights(order: int) -> tuple[tf.Tensor, tf.Tensor]:
    k = tf.cast(tf.range(1, int(order), dtype=tf.int32), tf.float64)
    beta = k / tf.sqrt(4.0 * tf.square(k) - 1.0)
    jacobi = tf.linalg.diag(beta, k=1) + tf.linalg.diag(beta, k=-1)
    eigenvalues, eigenvectors = tf.linalg.eigh(jacobi)
    return eigenvalues, 2.0 * tf.square(eigenvectors[0, :])


__all__ = [
    "ROUTE_CLASSIFICATION",
    "ROUTE_ID",
    "TTObservableJVP",
    "TTReferenceMoments",
    "TTShapeTargetsJVP",
    "TTParticleContractEStepJVP",
    "TTNormalizedMarginalJVP",
    "FrozenTTShapeTargets",
    "frozen_squared_tt_shape_targets",
    "legendre_monomial_operator_matrix",
    "monomial_operator_matrices",
    "squared_tt_affine_form_moment",
    "squared_tt_affine_form_moment_jvp",
    "padded_squared_tt_observable_jvp_xla",
    "squared_tt_raw_moment",
    "squared_tt_normalized_marginal_jvp",
    "squared_tt_shape_targets_jvp",
    "stack_squared_tt_shape_targets_jvp",
    "apply_tt_shape_targets_reference_jvp",
    "tt_particle_contract_e_step_reference_jvp",
    "squared_tt_reference_moments",
    "squared_tt_separable_observable",
    "squared_tt_separable_observable_jvp",
    "tensor_product_reference_monomial_moment",
]
