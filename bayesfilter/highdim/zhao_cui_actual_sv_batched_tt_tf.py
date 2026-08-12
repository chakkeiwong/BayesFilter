"""Batch-native fixed-capacity SVX-ZC likelihood for NeuTra/HMC evaluation."""

from __future__ import annotations

import math
from typing import Mapping

import tensorflow as tf


DTYPE = tf.float64
ROUTE_ID = "zhao_cui_fixed_adjacent_state_squared_tt_v1"
BATCH_ROUTE_ID = "zhao_cui_actual_sv_batched_fixed_tt_v1"
DEGREE = 10
RANK = 2
ORDER = 25
COORDINATE_HALF_WIDTH = 8.0
RIDGE = 1.0e-10
MAX_SWEEPS = 2
NORMALIZER_FLOOR = 1.0e-14
_LOG_TWO_PI = tf.constant(math.log(2.0 * math.pi), DTYPE)
_LOG_COORDINATE_REFERENCE_FACTOR = tf.math.log(
    tf.constant(2.0 * COORDINATE_HALF_WIDTH, DTYPE)
)
_SQRT_TWO = tf.sqrt(tf.constant(2.0, DTYPE))
_FLOAT64_EPS = tf.constant(2.220446049250313e-16, DTYPE)


def source_chart_physical_parameters(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Map source-probit coordinates to physical gamma and beta."""

    values = _rank2_theta(theta)
    probabilities = 0.5 * (1.0 + tf.math.erf(values / _SQRT_TWO))
    physical = 0.1 + 0.8 * probabilities
    return physical[:, 0], physical[:, 1]


def internal_likelihood_coordinates(theta: tf.Tensor) -> tf.Tensor:
    """Convert source-probit coordinates to the admitted likelihood chart."""

    gamma, beta = source_chart_physical_parameters(theta)
    gamma_coordinate = _SQRT_TWO * tf.math.erfinv(2.0 * gamma - 1.0)
    return tf.stack((gamma_coordinate, tf.math.log(beta)), axis=1)


def source_uniform_prior_value_score(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Uniform physical-box prior expressed before the chart Jacobian."""

    values = _rank2_theta(theta)
    value = tf.fill(tf.shape(values)[:1], -2.0 * tf.math.log(tf.constant(0.8, DTYPE)))
    return value, tf.zeros_like(values)


def source_two_probit_jacobian_value_score(
    theta: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Full log-Jacobian for the source two-probit chart."""

    values = _rank2_theta(theta)
    log_density = -0.5 * tf.square(values) - 0.5 * _LOG_TWO_PI
    value = tf.reduce_sum(tf.math.log(tf.constant(0.8, DTYPE)) + log_density, axis=1)
    return value, -values


def batched_fixed_tt_likelihood_value_status(
    theta: tf.Tensor,
    *,
    transformed_observations: tf.Tensor,
    initial_core: tf.Tensor,
    adjacent_core0: tf.Tensor,
    adjacent_core1: tf.Tensor,
    reference_nodes: tf.Tensor,
    reference_weights: tf.Tensor,
    reference_grid: tf.Tensor,
    reference_grid_weights: tf.Tensor,
    basis_nodes: tf.Tensor,
    basis_grid_axis0: tf.Tensor,
    basis_grid_axis1: tf.Tensor,
) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
    """Evaluate all parameter rows through one batch-native TT recursion."""

    values = _rank2_theta(theta)
    observations = tf.reshape(tf.convert_to_tensor(transformed_observations, DTYPE), [-1])
    if observations.shape[0] is None or int(observations.shape[0]) < 1:
        raise ValueError("SVX-ZC requires a static positive horizon")
    batch_size = tf.shape(values)[0]
    gamma, beta = source_chart_physical_parameters(values)
    beta_log_shift = 2.0 * tf.math.log(beta)
    one_minus_gamma_sq = 1.0 - tf.square(gamma)
    stationary_variance = tf.math.reciprocal(one_minus_gamma_sq)
    nodes = tf.reshape(tf.convert_to_tensor(reference_nodes, DTYPE), [-1])
    node_weights = tf.reshape(tf.convert_to_tensor(reference_weights, DTYPE), [-1])
    grid = tf.convert_to_tensor(reference_grid, DTYPE)
    grid_weights = tf.reshape(tf.convert_to_tensor(reference_grid_weights, DTYPE), [-1])
    basis1 = tf.convert_to_tensor(basis_nodes, DTYPE)
    basis0 = tf.convert_to_tensor(basis_grid_axis0, DTYPE)
    basis_previous = tf.convert_to_tensor(basis_grid_axis1, DTYPE)
    physical_nodes = COORDINATE_HALF_WIDTH * nodes
    physical_current = COORDINATE_HALF_WIDTH * grid[:, 0]
    physical_previous = COORDINATE_HALF_WIDTH * grid[:, 1]

    frozen_initial = tf.broadcast_to(
        tf.convert_to_tensor(initial_core, DTYPE)[tf.newaxis, ...],
        tf.concat(([batch_size], tf.shape(initial_core)), axis=0),
    )
    frozen_adjacent0 = tf.broadcast_to(
        tf.convert_to_tensor(adjacent_core0, DTYPE)[tf.newaxis, ...],
        tf.concat(([batch_size], tf.shape(adjacent_core0)), axis=0),
    )
    frozen_adjacent1 = tf.broadcast_to(
        tf.convert_to_tensor(adjacent_core1, DTYPE)[tf.newaxis, ...],
        tf.concat(([batch_size], tf.shape(adjacent_core1)), axis=0),
    )

    log_increments = []
    finite = tf.ones([batch_size], tf.bool)
    floor_count = tf.zeros([batch_size], tf.int32)
    maximum_condition = tf.ones([batch_size], DTYPE)
    minimum_normalizer = tf.fill([batch_size], tf.constant(float("inf"), DTYPE))
    previous_dimension = 0
    previous_core0 = frozen_initial
    previous_core1 = frozen_adjacent1

    for time_index in range(int(observations.shape[0])):
        if time_index == 0:
            initial_log = -0.5 * (
                _LOG_TWO_PI
                + tf.math.log(stationary_variance)[:, None]
                + tf.square(physical_nodes)[None, :] / stationary_variance[:, None]
            )
            residual = (
                observations[time_index]
                - beta_log_shift[:, None]
                - physical_nodes[None, :]
            )
            observation_log = _exact_log_chi_square_log_density(residual)
            log_target = initial_log + observation_log + _LOG_COORDINATE_REFERENCE_FACTOR
            log_shift = tf.reduce_max(log_target, axis=1)
            sqrt_target = tf.exp(0.5 * (log_target - log_shift[:, None]))
            fitted_core, condition = _fit_one_axis(
                basis1, sqrt_target, node_weights, frozen_initial
            )
            normalizer = _one_axis_normalizer(fitted_core)
            previous_core0 = fitted_core
            previous_dimension = 1
        else:
            if previous_dimension == 1:
                previous_density = _one_axis_density_values(
                    previous_core0, basis_previous
                )
            else:
                previous_density = _two_axis_marginal_values(
                    previous_core0, previous_core1, basis_previous
                )
            transition_residual = (
                physical_current[None, :]
                - gamma[:, None] * physical_previous[None, :]
            )
            transition_log = -0.5 * (
                _LOG_TWO_PI + tf.square(transition_residual)
            )
            observation_residual = (
                observations[time_index]
                - beta_log_shift[:, None]
                - physical_current[None, :]
            )
            observation_log = _exact_log_chi_square_log_density(observation_residual)
            log_target = (
                tf.math.log(previous_density)
                + transition_log
                + observation_log
                + _LOG_COORDINATE_REFERENCE_FACTOR
            )
            log_shift = tf.reduce_max(log_target, axis=1)
            sqrt_target = tf.exp(0.5 * (log_target - log_shift[:, None]))
            start0 = frozen_adjacent0 if time_index == 1 else previous_core0
            start1 = frozen_adjacent1 if time_index == 1 else previous_core1
            fitted0, fitted1, condition = _fit_two_axes(
                basis0,
                basis_previous,
                sqrt_target,
                grid_weights,
                start0,
                start1,
            )
            normalizer = _two_axis_normalizer(fitted0, fitted1)
            previous_core0 = fitted0
            previous_core1 = fitted1
            previous_dimension = 2
        increment = tf.math.log(normalizer) + log_shift
        finite = tf.logical_and(
            finite,
            tf.logical_and(
                tf.math.is_finite(increment),
                tf.logical_and(
                    tf.math.is_finite(condition),
                    tf.reduce_all(tf.math.is_finite(sqrt_target), axis=1),
                ),
            ),
        )
        below_floor = normalizer <= NORMALIZER_FLOOR
        floor_count = floor_count + tf.cast(below_floor, tf.int32)
        finite = tf.logical_and(finite, tf.logical_not(below_floor))
        maximum_condition = tf.maximum(maximum_condition, condition)
        minimum_normalizer = tf.minimum(minimum_normalizer, normalizer)
        log_increments.append(increment)

    value = tf.reduce_sum(tf.stack(log_increments, axis=1), axis=1)
    finite = tf.logical_and(finite, tf.math.is_finite(value))
    return value, {
        "status_code": tf.where(finite, tf.zeros_like(floor_count), tf.ones_like(floor_count)),
        "valid_pre_regularized_score": finite,
        "floor_count_value": floor_count,
        "minimum_normalizer": minimum_normalizer,
        # Compatibility alias required by the shared NeuTra status schema.
        # SVX-ZC has no Kalman innovation covariance; this is not an eigenvalue.
        "min_innovation_eigenvalue": minimum_normalizer,
        "innovation_condition_estimate": maximum_condition,
    }


def batched_fixed_tt_likelihood_value_score_status(
    theta: tf.Tensor,
    **program_tensors: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    """Differentiate the whole batch-native finite likelihood program."""

    values = _rank2_theta(theta)
    with tf.GradientTape() as tape:
        tape.watch(values)
        likelihood, status = batched_fixed_tt_likelihood_value_status(
            values, **program_tensors
        )
        # Keep the first-order gradient request in the tape context.  On the
        # repository TensorFlow build this is required for the batched
        # Cholesky solve backward path to remain connected to ``values``.
        score = tape.gradient(tf.reduce_sum(likelihood), values)
    if score is None:
        score = tf.fill(tf.shape(values), tf.constant(float("nan"), DTYPE))
    score = tf.convert_to_tensor(score, DTYPE)
    score_finite = tf.reduce_all(tf.math.is_finite(score), axis=1)
    valid = tf.logical_and(status["valid_pre_regularized_score"], score_finite)
    normalized = dict(status)
    normalized["valid_pre_regularized_score"] = valid
    normalized["status_code"] = tf.where(
        valid,
        tf.convert_to_tensor(status["status_code"], tf.int32),
        tf.ones_like(tf.convert_to_tensor(status["status_code"], tf.int32)),
    )
    return likelihood, score, normalized


def _fit_one_axis(
    basis: tf.Tensor,
    target: tf.Tensor,
    weights: tf.Tensor,
    initial_core: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(target)[0]
    design = tf.broadcast_to(
        basis[tf.newaxis, :, :],
        [batch_size, tf.shape(basis)[0], tf.shape(basis)[1]],
    )
    core = initial_core
    conditions = []
    for _sweep_index in range(MAX_SWEEPS):
        solution, condition = _scaled_qr_solve(design, target, weights)
        core = tf.reshape(solution, [batch_size, 1, DEGREE + 1, 1])
        conditions.append(condition)
    return core, tf.reduce_max(tf.stack(conditions, axis=1), axis=1)


def _fit_two_axes(
    basis0: tf.Tensor,
    basis1: tf.Tensor,
    target: tf.Tensor,
    weights: tf.Tensor,
    initial_core0: tf.Tensor,
    initial_core1: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(target)[0]
    core0 = initial_core0
    core1 = initial_core1
    conditions = []
    for _sweep_index in range(MAX_SWEEPS):
        for axis in (0, 1, 1, 0):
            matrix0 = tf.einsum("nl,balr->bnar", basis0, core0)
            matrix1 = tf.einsum("nl,balr->bnar", basis1, core1)
            if axis == 0:
                right = tf.squeeze(matrix1, axis=3)
                blocks = tf.einsum("nl,bnr->bnlr", basis0, right)
                design = tf.reshape(
                    blocks, [batch_size, tf.shape(target)[1], (DEGREE + 1) * RANK]
                )
                solution, condition = _scaled_qr_solve(design, target, weights)
                core0 = tf.reshape(solution, [batch_size, 1, DEGREE + 1, RANK])
            else:
                left = tf.squeeze(matrix0, axis=2)
                blocks = tf.einsum("bnr,nl->bnrl", left, basis1)
                design = tf.reshape(
                    blocks, [batch_size, tf.shape(target)[1], RANK * (DEGREE + 1)]
                )
                solution, condition = _scaled_qr_solve(design, target, weights)
                core1 = tf.reshape(solution, [batch_size, RANK, DEGREE + 1, 1])
            conditions.append(condition)
    return core0, core1, tf.reduce_max(tf.stack(conditions, axis=1), axis=1)


def _scaled_qr_solve(
    design: tf.Tensor,
    target: tf.Tensor,
    weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Solve the scaled ridge least-squares problem with batch-safe AD.

    The previous QR-on-an-augmented-matrix implementation was numerically
    valid in the forward pass, but its TensorFlow QR/triangular-solve
    derivative produced NaNs for repeated rows.  Forming the regularized
    normal equations is algebraically equivalent here and gives a stable,
    batch-native Cholesky backward path.
    """
    raw_norms = tf.sqrt(
        tf.reduce_sum(weights[None, :, None] * tf.square(design), axis=1)
    )
    scale_floor = tf.maximum(
        tf.sqrt(_FLOAT64_EPS) * tf.reduce_max(raw_norms, axis=1, keepdims=True),
        _FLOAT64_EPS,
    )
    scales = tf.maximum(raw_norms, scale_floor)
    scaled_design = design / scales[:, None, :]
    normal = tf.einsum(
        "n,bnp,bnq->bpq", weights, scaled_design, scaled_design
    )
    ridge = tf.linalg.diag(
        tf.broadcast_to(
            tf.constant(RIDGE, DTYPE) / tf.square(scales),
            tf.shape(normal)[:2],
        )
    )
    normal = normal + ridge
    rhs = tf.einsum("n,bnp,bn->bp", weights, scaled_design, target)
    factor = tf.linalg.cholesky(normal)
    scaled_solution = tf.linalg.cholesky_solve(factor, rhs[:, :, None])[:, :, 0]
    solution = scaled_solution / scales
    diagonal = tf.abs(tf.linalg.diag_part(factor))
    condition_proxy = tf.reduce_max(diagonal, axis=1) / tf.reduce_min(diagonal, axis=1)
    return solution, condition_proxy


def _one_axis_normalizer(core: tf.Tensor) -> tf.Tensor:
    coefficients = tf.squeeze(core, axis=(1, 3))
    return tf.reduce_sum(tf.square(coefficients), axis=1)


def _one_axis_density_values(core: tf.Tensor, basis: tf.Tensor) -> tf.Tensor:
    coefficients = tf.squeeze(core, axis=(1, 3))
    amplitude = tf.einsum("nl,bl->bn", basis, coefficients)
    return tf.square(amplitude) / _one_axis_normalizer(core)[:, None]


def _two_axis_normalizer(core0: tf.Tensor, core1: tf.Tensor) -> tf.Tensor:
    left = tf.squeeze(core0, axis=1)
    right = tf.squeeze(core1, axis=3)
    left_mass = tf.einsum("blr,bls->brs", left, left)
    right_mass = tf.einsum("brl,bsl->brs", right, right)
    return tf.reduce_sum(left_mass * right_mass, axis=(1, 2))


def _two_axis_marginal_values(
    core0: tf.Tensor,
    core1: tf.Tensor,
    basis: tf.Tensor,
) -> tf.Tensor:
    left = tf.squeeze(core0, axis=1)
    right = tf.squeeze(core1, axis=3)
    evaluated = tf.einsum("nl,blr->bnr", basis, left)
    right_mass = tf.einsum("brl,bsl->brs", right, right)
    numerator = tf.einsum("bnr,bns,brs->bn", evaluated, evaluated, right_mass)
    return numerator / _two_axis_normalizer(core0, core1)[:, None]


def _exact_log_chi_square_log_density(value: tf.Tensor) -> tf.Tensor:
    return 0.5 * value - 0.5 * tf.exp(value) - 0.5 * _LOG_TWO_PI


def _rank2_theta(theta: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(theta, DTYPE)
    if values.shape.rank != 2 or values.shape[-1] != 2:
        raise ValueError("SVX-ZC batch target requires theta shape [batch, 2]")
    return values


__all__ = [
    "BATCH_ROUTE_ID",
    "COORDINATE_HALF_WIDTH",
    "DEGREE",
    "ORDER",
    "RANK",
    "batched_fixed_tt_likelihood_value_score_status",
    "batched_fixed_tt_likelihood_value_status",
    "internal_likelihood_coordinates",
    "source_chart_physical_parameters",
    "source_two_probit_jacobian_value_score",
    "source_uniform_prior_value_score",
]
