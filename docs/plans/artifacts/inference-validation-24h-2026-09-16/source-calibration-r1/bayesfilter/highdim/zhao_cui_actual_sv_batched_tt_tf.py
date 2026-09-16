"""Batch-native fixed-capacity SVX-ZC likelihood for NeuTra/HMC evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _OneAxisFitTrace:
    fitted_core: tf.Tensor
    condition_by_sweep: tf.Tensor
    design: tf.Tensor
    target: tf.Tensor
    weights: tf.Tensor


@dataclass(frozen=True)
class _TwoAxisSweepTrace:
    sweep_index: int
    axis: int
    design: tf.Tensor
    solution: tf.Tensor
    condition: tf.Tensor
    core0_before: tf.Tensor
    core1_before: tf.Tensor
    core0_after: tf.Tensor
    core1_after: tf.Tensor


@dataclass(frozen=True)
class _TwoAxisFitTrace:
    fitted0: tf.Tensor
    fitted1: tf.Tensor
    condition_by_update: tf.Tensor
    sweeps: tuple[_TwoAxisSweepTrace, ...]
    target: tf.Tensor
    weights: tf.Tensor


@dataclass(frozen=True)
class _SVXStepTrace:
    time_index: int
    target_kind: str
    log_target: tf.Tensor
    log_shift: tf.Tensor
    sqrt_target: tf.Tensor
    normalizer: tf.Tensor
    condition: tf.Tensor
    previous_density: tf.Tensor | None
    one_axis_fit: _OneAxisFitTrace | None
    two_axis_fit: _TwoAxisFitTrace | None


@dataclass(frozen=True)
class _SVXLikelihoodTrace:
    value: tf.Tensor
    status: Mapping[str, tf.Tensor]
    steps: tuple[_SVXStepTrace, ...]


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

    trace = batched_fixed_tt_likelihood_value_trace(
        theta,
        transformed_observations=transformed_observations,
        initial_core=initial_core,
        adjacent_core0=adjacent_core0,
        adjacent_core1=adjacent_core1,
        reference_nodes=reference_nodes,
        reference_weights=reference_weights,
        reference_grid=reference_grid,
        reference_grid_weights=reference_grid_weights,
        basis_nodes=basis_nodes,
        basis_grid_axis0=basis_grid_axis0,
        basis_grid_axis1=basis_grid_axis1,
    )
    return trace.value, trace.status


def batched_fixed_tt_likelihood_value_trace(
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
) -> _SVXLikelihoodTrace:
    """Return the active batched SVX value path together with sweep-local traces."""

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
    step_traces = []
    finite = tf.ones([batch_size], tf.bool)
    floor_count = tf.zeros([batch_size], tf.int32)
    maximum_condition = tf.ones([batch_size], DTYPE)
    minimum_normalizer = tf.fill([batch_size], tf.constant(float("inf"), DTYPE))
    previous_dimension = 0
    previous_core0 = frozen_initial
    previous_core1 = frozen_adjacent1

    for time_index in range(int(observations.shape[0])):
        one_axis_fit = None
        two_axis_fit = None
        previous_density = None
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
            one_axis_fit = _fit_one_axis_with_trace(
                basis1, sqrt_target, node_weights, frozen_initial
            )
            fitted_core = one_axis_fit.fitted_core
            condition = tf.reduce_max(one_axis_fit.condition_by_sweep, axis=1)
            normalizer = _one_axis_normalizer(fitted_core)
            previous_core0 = fitted_core
            previous_dimension = 1
            target_kind = "initial_state_observation"
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
            two_axis_fit = _fit_two_axes_with_trace(
                basis0,
                basis_previous,
                sqrt_target,
                grid_weights,
                start0,
                start1,
            )
            fitted0 = two_axis_fit.fitted0
            fitted1 = two_axis_fit.fitted1
            condition = tf.reduce_max(two_axis_fit.condition_by_update, axis=1)
            normalizer = _two_axis_normalizer(fitted0, fitted1)
            previous_core0 = fitted0
            previous_core1 = fitted1
            previous_dimension = 2
            target_kind = "adjacent_state_update"
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
        step_traces.append(
            _SVXStepTrace(
                time_index=time_index,
                target_kind=target_kind,
                log_target=log_target,
                log_shift=log_shift,
                sqrt_target=sqrt_target,
                normalizer=normalizer,
                condition=condition,
                previous_density=previous_density,
                one_axis_fit=one_axis_fit,
                two_axis_fit=two_axis_fit,
            )
        )

    value = tf.reduce_sum(tf.stack(log_increments, axis=1), axis=1)
    finite = tf.logical_and(finite, tf.math.is_finite(value))
    status = {
        "status_code": tf.where(finite, tf.zeros_like(floor_count), tf.ones_like(floor_count)),
        "valid_pre_regularized_score": finite,
        "floor_count_value": floor_count,
        "minimum_normalizer": minimum_normalizer,
        "min_innovation_eigenvalue": minimum_normalizer,
        "innovation_condition_estimate": maximum_condition,
    }
    return _SVXLikelihoodTrace(
        value=value,
        status=status,
        steps=tuple(step_traces),
    )


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


def batched_fixed_tt_likelihood_analytic_score_status(
    theta: tf.Tensor,
    **program_tensors: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    """Compute the same-program SVX-ZC score from a manual directional replay."""

    values = _rank2_theta(theta)
    trace = batched_fixed_tt_likelihood_value_trace(values, **program_tensors)
    observations = tf.reshape(
        tf.convert_to_tensor(program_tensors["transformed_observations"], DTYPE), [-1]
    )
    basis_previous = tf.convert_to_tensor(program_tensors["basis_grid_axis1"], DTYPE)
    basis_nodes = tf.convert_to_tensor(program_tensors["basis_nodes"], DTYPE)
    basis_current = tf.convert_to_tensor(program_tensors["basis_grid_axis0"], DTYPE)
    physical_nodes = COORDINATE_HALF_WIDTH * tf.reshape(
        tf.convert_to_tensor(program_tensors["reference_nodes"], DTYPE), [-1]
    )
    grid = tf.convert_to_tensor(program_tensors["reference_grid"], DTYPE)
    physical_current = COORDINATE_HALF_WIDTH * grid[:, 0]
    physical_previous = COORDINATE_HALF_WIDTH * grid[:, 1]
    gamma, beta = source_chart_physical_parameters(values)
    normal_pdf = tf.exp(-0.5 * tf.square(values) - 0.5 * _LOG_TWO_PI)
    d_gamma = 0.8 * normal_pdf[:, 0]
    d_beta = 0.8 * normal_pdf[:, 1]
    zero_batch = tf.zeros_like(gamma)
    floor = tf.constant(1e-300, DTYPE)
    sqrt_eps = tf.sqrt(_FLOAT64_EPS)

    def _batch_gather(values_2d: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
        return tf.gather(values_2d, indices, batch_dims=1)

    def _one_axis_dot_normalizer(core: tf.Tensor, dot_core: tf.Tensor) -> tf.Tensor:
        coefficients = tf.squeeze(core, axis=(1, 3))
        dot_coefficients = tf.squeeze(dot_core, axis=(1, 3))
        return 2.0 * tf.reduce_sum(coefficients * dot_coefficients, axis=1)

    def _one_axis_log_density_directional(
        core: tf.Tensor,
        dot_core: tf.Tensor,
        basis: tf.Tensor,
    ) -> tf.Tensor:
        coefficients = tf.squeeze(core, axis=(1, 3))
        dot_coefficients = tf.squeeze(dot_core, axis=(1, 3))
        amplitude = tf.einsum("nl,bl->bn", basis, coefficients)
        dot_amplitude = tf.einsum("nl,bl->bn", basis, dot_coefficients)
        normalizer = _one_axis_normalizer(core)
        dot_normalizer = _one_axis_dot_normalizer(core, dot_core)
        density = tf.square(amplitude) / normalizer[:, None]
        dot_density = (
            2.0 * amplitude * dot_amplitude / normalizer[:, None]
            - density * dot_normalizer[:, None] / normalizer[:, None]
        )
        return dot_density / tf.maximum(density, floor)

    def _two_axis_dot_normalizer(
        core0: tf.Tensor,
        core1: tf.Tensor,
        dot_core0: tf.Tensor,
        dot_core1: tf.Tensor,
    ) -> tf.Tensor:
        left = tf.squeeze(core0, axis=1)
        right = tf.squeeze(core1, axis=3)
        dot_left = tf.squeeze(dot_core0, axis=1)
        dot_right = tf.squeeze(dot_core1, axis=3)
        left_mass = tf.einsum("blr,bls->brs", left, left)
        right_mass = tf.einsum("brl,bsl->brs", right, right)
        dot_left_mass = tf.einsum("blr,bls->brs", dot_left, left) + tf.einsum(
            "blr,bls->brs", left, dot_left
        )
        dot_right_mass = tf.einsum("brl,bsl->brs", dot_right, right) + tf.einsum(
            "brl,bsl->brs", right, dot_right
        )
        return tf.reduce_sum(
            dot_left_mass * right_mass + left_mass * dot_right_mass, axis=(1, 2)
        )

    def _two_axis_log_density_directional(
        core0: tf.Tensor,
        core1: tf.Tensor,
        dot_core0: tf.Tensor,
        dot_core1: tf.Tensor,
        basis: tf.Tensor,
    ) -> tf.Tensor:
        left = tf.squeeze(core0, axis=1)
        right = tf.squeeze(core1, axis=3)
        dot_left = tf.squeeze(dot_core0, axis=1)
        dot_right = tf.squeeze(dot_core1, axis=3)
        evaluated = tf.einsum("nl,blr->bnr", basis, left)
        dot_evaluated = tf.einsum("nl,blr->bnr", basis, dot_left)
        right_mass = tf.einsum("brl,bsl->brs", right, right)
        dot_right_mass = tf.einsum("brl,bsl->brs", dot_right, right) + tf.einsum(
            "brl,bsl->brs", right, dot_right
        )
        numerator = tf.einsum("bnr,bns,brs->bn", evaluated, evaluated, right_mass)
        dot_numerator = (
            tf.einsum("bnr,bns,brs->bn", dot_evaluated, evaluated, right_mass)
            + tf.einsum("bnr,bns,brs->bn", evaluated, dot_evaluated, right_mass)
            + tf.einsum("bnr,bns,brs->bn", evaluated, evaluated, dot_right_mass)
        )
        normalizer = _two_axis_normalizer(core0, core1)
        dot_normalizer = _two_axis_dot_normalizer(core0, core1, dot_core0, dot_core1)
        density = numerator / normalizer[:, None]
        dot_density = dot_numerator / normalizer[:, None] - density * dot_normalizer[:, None] / normalizer[:, None]
        return dot_density / tf.maximum(density, floor)

    def _scaled_solve_directional(
        design: tf.Tensor,
        target: tf.Tensor,
        weights: tf.Tensor,
        solution: tf.Tensor,
        dot_design: tf.Tensor,
        dot_target: tf.Tensor,
    ) -> tf.Tensor:
        raw_norms = tf.sqrt(
            tf.reduce_sum(weights[None, :, None] * tf.square(design), axis=1)
        )
        safe_raw_norms = tf.maximum(raw_norms, _FLOAT64_EPS)
        dot_raw_norms = tf.reduce_sum(
            weights[None, :, None] * design * dot_design, axis=1
        ) / safe_raw_norms
        max_indices = tf.argmax(raw_norms, axis=1, output_type=tf.int32)
        max_raw_norms = tf.reduce_max(raw_norms, axis=1)
        dot_max_raw_norms = _batch_gather(dot_raw_norms, max_indices)
        floor_from_max = sqrt_eps * max_raw_norms
        scale_floor = tf.maximum(floor_from_max, _FLOAT64_EPS)
        dot_scale_floor = tf.where(
            floor_from_max >= _FLOAT64_EPS,
            sqrt_eps * dot_max_raw_norms,
            tf.zeros_like(dot_max_raw_norms),
        )
        scales = tf.maximum(raw_norms, scale_floor[:, None])
        dot_scales = tf.where(
            raw_norms >= scale_floor[:, None],
            dot_raw_norms,
            dot_scale_floor[:, None],
        )
        scaled_design = design / scales[:, None, :]
        dot_scaled_design = dot_design / scales[:, None, :] - design * dot_scales[:, None, :] / tf.square(scales[:, None, :])
        scaled_normal = tf.einsum(
            "n,bnp,bnq->bpq", weights, scaled_design, scaled_design
        ) + tf.linalg.diag(tf.constant(RIDGE, DTYPE) / tf.square(scales))
        dot_scaled_normal = (
            tf.einsum("n,bnp,bnq->bpq", weights, dot_scaled_design, scaled_design)
            + tf.einsum("n,bnp,bnq->bpq", weights, scaled_design, dot_scaled_design)
            + tf.linalg.diag(-2.0 * tf.constant(RIDGE, DTYPE) * dot_scales / tf.pow(scales, 3.0))
        )
        scaled_rhs = tf.einsum("n,bnp,bn->bp", weights, scaled_design, target)
        dot_scaled_rhs = (
            tf.einsum("n,bnp,bn->bp", weights, dot_scaled_design, target)
            + tf.einsum("n,bnp,bn->bp", weights, scaled_design, dot_target)
        )
        scaled_solution = solution * scales
        factor = tf.linalg.cholesky(scaled_normal)
        dot_scaled_solution = tf.linalg.cholesky_solve(
            factor,
            (dot_scaled_rhs - tf.linalg.matvec(dot_scaled_normal, scaled_solution))[:, :, None],
        )[:, :, 0]
        return dot_scaled_solution / scales - scaled_solution * dot_scales / tf.square(scales)

    def _two_axis_fit_directional(
        fit_trace: _TwoAxisFitTrace,
        dot_target: tf.Tensor,
        start_dot_core0: tf.Tensor,
        start_dot_core1: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        dot_core0 = start_dot_core0
        dot_core1 = start_dot_core1
        for sweep in fit_trace.sweeps:
            if sweep.axis == 0:
                dot_matrix1 = tf.einsum("nl,balr->bnar", basis_previous, dot_core1)
                right_dot = tf.squeeze(dot_matrix1, axis=3)
                dot_blocks = tf.einsum("nl,bnr->bnlr", basis_current, right_dot)
                dot_design = tf.reshape(dot_blocks, tf.shape(sweep.design))
                dot_solution = _scaled_solve_directional(
                    sweep.design,
                    fit_trace.target,
                    fit_trace.weights,
                    sweep.solution,
                    dot_design,
                    dot_target,
                )
                dot_core0 = tf.reshape(dot_solution, tf.shape(sweep.core0_after))
            else:
                dot_matrix0 = tf.einsum("nl,balr->bnar", basis_current, dot_core0)
                left_dot = tf.squeeze(dot_matrix0, axis=2)
                dot_blocks = tf.einsum("bnr,nl->bnrl", left_dot, basis_previous)
                dot_design = tf.reshape(dot_blocks, tf.shape(sweep.design))
                dot_solution = _scaled_solve_directional(
                    sweep.design,
                    fit_trace.target,
                    fit_trace.weights,
                    sweep.solution,
                    dot_design,
                    dot_target,
                )
                dot_core1 = tf.reshape(dot_solution, tf.shape(sweep.core1_after))
        return dot_core0, dot_core1

    score_columns = []
    for parameter_index in range(2):
        dot_gamma = d_gamma if parameter_index == 0 else zero_batch
        dot_beta = d_beta if parameter_index == 1 else zero_batch
        dot_log_previous_density = None
        dot_previous_core0 = None
        dot_previous_core1 = None
        score_column = tf.zeros_like(gamma)
        for step in trace.steps:
            if step.one_axis_fit is not None:
                physical = physical_nodes[None, :]
                stationary_inverse = 1.0 / (1.0 - tf.square(gamma))
                dot_initial = dot_gamma[:, None] * gamma[:, None] * (
                    tf.square(physical) - stationary_inverse[:, None]
                )
                residual = (
                    observations[step.time_index]
                    - 2.0 * tf.math.log(beta)[:, None]
                    - physical
                )
                dot_observation = dot_beta[:, None] * (
                    tf.exp(residual) - 1.0
                ) / beta[:, None]
                dot_log_target = dot_initial + dot_observation
                max_indices = tf.argmax(step.log_target, axis=1, output_type=tf.int32)
                dot_shift = _batch_gather(dot_log_target, max_indices)
                dot_sqrt = 0.5 * step.sqrt_target * (
                    dot_log_target - dot_shift[:, None]
                )
                coefficients = tf.squeeze(step.one_axis_fit.fitted_core, axis=(1, 3))
                dot_coefficients = _scaled_solve_directional(
                    step.one_axis_fit.design,
                    step.one_axis_fit.target,
                    step.one_axis_fit.weights,
                    coefficients,
                    tf.zeros_like(step.one_axis_fit.design),
                    dot_sqrt,
                )
                dot_core = tf.reshape(dot_coefficients, tf.shape(step.one_axis_fit.fitted_core))
                dot_normalizer = _one_axis_dot_normalizer(
                    step.one_axis_fit.fitted_core, dot_core
                )
                score_column = score_column + dot_shift + dot_normalizer / step.normalizer
                dot_log_previous_density = _one_axis_log_density_directional(
                    step.one_axis_fit.fitted_core,
                    dot_core,
                    basis_previous,
                )
                dot_previous_core0 = dot_core
                dot_previous_core1 = None
            else:
                transition_dot = dot_gamma[:, None] * (
                    physical_current[None, :]
                    - gamma[:, None] * physical_previous[None, :]
                ) * physical_previous[None, :]
                residual = (
                    observations[step.time_index]
                    - 2.0 * tf.math.log(beta)[:, None]
                    - physical_current[None, :]
                )
                observation_dot = dot_beta[:, None] * (
                    tf.exp(residual) - 1.0
                ) / beta[:, None]
                dot_log_target = dot_log_previous_density + transition_dot + observation_dot
                max_indices = tf.argmax(step.log_target, axis=1, output_type=tf.int32)
                dot_shift = _batch_gather(dot_log_target, max_indices)
                dot_sqrt = 0.5 * step.sqrt_target * (
                    dot_log_target - dot_shift[:, None]
                )
                start_dot_core0 = (
                    tf.zeros_like(step.two_axis_fit.fitted0)
                    if dot_previous_core1 is None
                    else dot_previous_core0
                )
                start_dot_core1 = (
                    tf.zeros_like(step.two_axis_fit.fitted1)
                    if dot_previous_core1 is None
                    else dot_previous_core1
                )
                dot_core0, dot_core1 = _two_axis_fit_directional(
                    step.two_axis_fit,
                    dot_sqrt,
                    start_dot_core0,
                    start_dot_core1,
                )
                dot_normalizer = _two_axis_dot_normalizer(
                    step.two_axis_fit.fitted0,
                    step.two_axis_fit.fitted1,
                    dot_core0,
                    dot_core1,
                )
                score_column = score_column + dot_shift + dot_normalizer / step.normalizer
                dot_log_previous_density = _two_axis_log_density_directional(
                    step.two_axis_fit.fitted0,
                    step.two_axis_fit.fitted1,
                    dot_core0,
                    dot_core1,
                    basis_previous,
                )
                dot_previous_core0 = dot_core0
                dot_previous_core1 = dot_core1
        score_columns.append(score_column)
    score = tf.stack(score_columns, axis=1)
    score_finite = tf.reduce_all(tf.math.is_finite(score), axis=1)
    valid = tf.logical_and(trace.status["valid_pre_regularized_score"], score_finite)
    normalized = dict(trace.status)
    normalized["valid_pre_regularized_score"] = valid
    normalized["status_code"] = tf.where(
        valid,
        tf.convert_to_tensor(trace.status["status_code"], tf.int32),
        tf.ones_like(tf.convert_to_tensor(trace.status["status_code"], tf.int32)),
    )
    return trace.value, score, normalized


    trace = _fit_one_axis_with_trace(basis, target, weights, initial_core)
    return trace.fitted_core, tf.reduce_max(trace.condition_by_sweep, axis=1)


def _fit_one_axis_with_trace(
    basis: tf.Tensor,
    target: tf.Tensor,
    weights: tf.Tensor,
    initial_core: tf.Tensor,
) -> _OneAxisFitTrace:
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
    return _OneAxisFitTrace(
        fitted_core=core,
        condition_by_sweep=tf.stack(conditions, axis=1),
        design=design,
        target=target,
        weights=weights,
    )


def _fit_two_axes_with_trace(
    basis0: tf.Tensor,
    basis1: tf.Tensor,
    target: tf.Tensor,
    weights: tf.Tensor,
    initial_core0: tf.Tensor,
    initial_core1: tf.Tensor,
) -> _TwoAxisFitTrace:
    batch_size = tf.shape(target)[0]
    core0 = initial_core0
    core1 = initial_core1
    conditions = []
    sweeps = []
    for sweep_index in range(MAX_SWEEPS):
        for axis in (0, 1, 1, 0):
            core0_before = core0
            core1_before = core1
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
            sweeps.append(
                _TwoAxisSweepTrace(
                    sweep_index=sweep_index,
                    axis=axis,
                    design=design,
                    solution=solution,
                    condition=condition,
                    core0_before=core0_before,
                    core1_before=core1_before,
                    core0_after=core0,
                    core1_after=core1,
                )
            )
            conditions.append(condition)
    return _TwoAxisFitTrace(
        fitted0=core0,
        fitted1=core1,
        condition_by_update=tf.stack(conditions, axis=1),
        sweeps=tuple(sweeps),
        target=target,
        weights=weights,
    )


def _fit_two_axes(
    basis0: tf.Tensor,
    basis1: tf.Tensor,
    target: tf.Tensor,
    weights: tf.Tensor,
    initial_core0: tf.Tensor,
    initial_core1: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    trace = _fit_two_axes_with_trace(
        basis0,
        basis1,
        target,
        weights,
        initial_core0,
        initial_core1,
    )
    return trace.fitted0, trace.fitted1, tf.reduce_max(trace.condition_by_update, axis=1)


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
