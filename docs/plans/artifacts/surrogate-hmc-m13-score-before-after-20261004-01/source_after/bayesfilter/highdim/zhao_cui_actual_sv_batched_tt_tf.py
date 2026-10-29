"""Batch-native fixed-capacity SVX-ZC likelihood for NeuTra/HMC evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

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



class _TensorSweep(NamedTuple):
    sweep_index: tf.Tensor
    axis: tf.Tensor
    design: tf.Tensor
    solution: tf.Tensor
    condition: tf.Tensor
    core0_before: tf.Tensor
    core1_before: tf.Tensor
    core0_after: tf.Tensor
    core1_after: tf.Tensor


class _TensorFit(NamedTuple):
    fitted0: tf.Tensor
    fitted1: tf.Tensor
    condition_by_update: tf.Tensor
    sweeps: _TensorSweep
    target: tf.Tensor
    weights: tf.Tensor


class _TensorStep(NamedTuple):
    log_target: tf.Tensor
    log_shift: tf.Tensor
    sqrt_target: tf.Tensor
    normalizer: tf.Tensor
    condition: tf.Tensor
    previous_density: tf.Tensor
    two_axis_fit: _TensorFit


def _history_arrays(value, size):
    # Fixed tensor schema plumbing; no numerical iteration in Python.
    return tf.nest.map_structure(
        lambda x: tf.TensorArray(x.dtype, size=size, element_shape=x.shape,
                                clear_after_read=False), value)


def _history_write(arrays, index, value):
    return tf.nest.map_structure(lambda a, x: a.write(index, x), arrays, value)


def _history_read(arrays, index):
    return tf.nest.map_structure(lambda x: x[index], arrays)


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

    value, status, _initial, _adjacent = _likelihood_tensor_trace(
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
    return value, status


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
    """Diagnostic formatting of native tensor histories into time-local records."""

    value, status, initial, adjacent = _likelihood_tensor_trace(
        theta, transformed_observations=transformed_observations,
        initial_core=initial_core, adjacent_core0=adjacent_core0,
        adjacent_core1=adjacent_core1, reference_nodes=reference_nodes,
        reference_weights=reference_weights, reference_grid=reference_grid,
        reference_grid_weights=reference_grid_weights, basis_nodes=basis_nodes,
        basis_grid_axis0=basis_grid_axis0, basis_grid_axis1=basis_grid_axis1,
        _record_design=True,
    )
    steps = [initial]
    # Reporting only. Numerical endpoints use tensor histories directly.
    for i in range(int(tf.convert_to_tensor(transformed_observations).shape[0]) - 1):
        step = _history_read(adjacent, i)
        steps.append(_SVXStepTrace(
            i + 1, "adjacent_state_update", step.log_target, step.log_shift,
            step.sqrt_target, step.normalizer, step.condition,
            step.previous_density, None, _report_two_axis_fit(step.two_axis_fit)))
    return _SVXLikelihoodTrace(value, status, tuple(steps))


def _likelihood_tensor_trace(
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
    _record_design: bool = False,
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

    initial_log = -0.5 * (
        _LOG_TWO_PI + tf.math.log(stationary_variance)[:, None]
        + tf.square(physical_nodes)[None, :] / stationary_variance[:, None])
    residual = observations[0] - beta_log_shift[:, None] - physical_nodes[None, :]
    log_target = initial_log + _exact_log_chi_square_log_density(residual) + _LOG_COORDINATE_REFERENCE_FACTOR
    log_shift = tf.reduce_max(log_target, axis=1)
    sqrt_target = tf.exp(0.5 * (log_target - log_shift[:, None]))
    fit = _fit_one_axis_with_trace(basis1, sqrt_target, node_weights, frozen_initial)
    condition = tf.reduce_max(fit.condition_by_sweep, axis=1)
    normalizer = _one_axis_normalizer(fit.fitted_core)
    initial = _SVXStepTrace(0, "initial_state_observation", log_target, log_shift,
                           sqrt_target, normalizer, condition, None, fit, None)
    increments = tf.TensorArray(DTYPE, size=int(observations.shape[0]),
                                element_shape=gamma.shape, clear_after_read=False)
    increments = increments.write(0, tf.math.log(normalizer) + log_shift)

    def accumulate(step, finite, floor_count, max_condition, min_normalizer):
        increment = tf.math.log(step.normalizer) + step.log_shift
        finite = finite & tf.math.is_finite(increment) & tf.math.is_finite(step.condition)
        finite = finite & tf.reduce_all(tf.math.is_finite(step.sqrt_target), axis=1)
        below_floor = step.normalizer <= NORMALIZER_FLOOR
        return (finite & ~below_floor, floor_count + tf.cast(below_floor, tf.int32),
                tf.maximum(max_condition, step.condition),
                tf.minimum(min_normalizer, step.normalizer))

    status_state = accumulate(initial, tf.ones([batch_size], tf.bool),
                              tf.zeros([batch_size], tf.int32),
                              tf.ones([batch_size], DTYPE),
                              tf.fill([batch_size], tf.constant(float("inf"), DTYPE)))

    def adjacent_step(time_index, previous_density, core0, core1):
        transition_residual = physical_current[None, :] - gamma[:, None] * physical_previous[None, :]
        transition_log = -0.5 * (_LOG_TWO_PI + tf.square(transition_residual))
        observation_residual = observations[time_index] - beta_log_shift[:, None] - physical_current[None, :]
        log_target = (tf.math.log(previous_density) + transition_log
                      + _exact_log_chi_square_log_density(observation_residual)
                      + _LOG_COORDINATE_REFERENCE_FACTOR)
        log_shift = tf.reduce_max(log_target, axis=1)
        sqrt_target = tf.exp(0.5 * (log_target - log_shift[:, None]))
        fit = _fit_two_axes_tensor(basis0, basis_previous, sqrt_target,
                                   grid_weights, core0, core1, record_design=_record_design)
        return _TensorStep(log_target, log_shift, sqrt_target,
                           _two_axis_normalizer(fit.fitted0, fit.fitted1),
                           tf.reduce_max(fit.condition_by_update, axis=1),
                           previous_density, fit)

    history = None
    horizon = int(observations.shape[0])
    if horizon > 1:
        # Initial one-axis rank differs. Enter the fixed-shape recurrence only
        # after the first adjacent-state update.
        first = adjacent_step(1, _one_axis_density_values(fit.fitted_core, basis_previous),
                              frozen_adjacent0, frozen_adjacent1)
        history = _history_write(_history_arrays(first, horizon - 1), 0, first)
        increments = increments.write(1, tf.math.log(first.normalizer) + first.log_shift)
        status_state = accumulate(first, *status_state)

        def body(t, core0, core1, history, increments, status_state):
            step = adjacent_step(t, _two_axis_marginal_values(core0, core1, basis_previous), core0, core1)
            return (t + 1, step.two_axis_fit.fitted0, step.two_axis_fit.fitted1,
                    _history_write(history, t - 1, step),
                    increments.write(t, tf.math.log(step.normalizer) + step.log_shift),
                    accumulate(step, *status_state))

        if horizon > 2:
            _, _, _, history, increments, status_state = tf.while_loop(
                lambda t, *_: t < horizon, body,
                (tf.constant(2), first.two_axis_fit.fitted0, first.two_axis_fit.fitted1,
                 history, increments, status_state),
                parallel_iterations=1, maximum_iterations=horizon - 2)
        history = tf.nest.map_structure(lambda x: x.stack(), history)
    value = tf.reduce_sum(tf.transpose(increments.stack()), axis=1)
    finite, floor_count, maximum_condition, minimum_normalizer = status_state
    finite = finite & tf.math.is_finite(value)
    status = {
        "status_code": tf.where(finite, tf.zeros_like(floor_count), tf.ones_like(floor_count)),
        "valid_pre_regularized_score": finite,
        "floor_count_value": floor_count,
        "minimum_normalizer": minimum_normalizer,
        "min_innovation_eigenvalue": minimum_normalizer,
        "innovation_condition_estimate": maximum_condition,
    }
    return value, status, initial, history


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


def batched_fixed_tt_likelihood_analytic_score_status(theta, **program_tensors):
    """Manual same-program derivatives with both parameter directions batched.

    Dates and ALS updates use native loops. The final parameter axis carries
    the two directional derivatives through each linear solve simultaneously;
    there is no row-mapped scalar target or autodiff score.
    """
    values = _rank2_theta(theta)
    likelihood, status, initial, adjacent = _likelihood_tensor_trace(values, **program_tensors)
    observations = tf.reshape(tf.convert_to_tensor(program_tensors['transformed_observations'], DTYPE), [-1])
    basis_previous = tf.convert_to_tensor(program_tensors['basis_grid_axis1'], DTYPE)
    basis_current = tf.convert_to_tensor(program_tensors['basis_grid_axis0'], DTYPE)
    physical_nodes = COORDINATE_HALF_WIDTH * tf.reshape(tf.convert_to_tensor(program_tensors['reference_nodes'], DTYPE), [-1])
    grid = tf.convert_to_tensor(program_tensors['reference_grid'], DTYPE)
    physical_current, physical_previous = COORDINATE_HALF_WIDTH * grid[:, 0], COORDINATE_HALF_WIDTH * grid[:, 1]
    gamma, beta = source_chart_physical_parameters(values)
    pdf = tf.exp(-.5 * tf.square(values) - .5 * _LOG_TWO_PI)
    dot_gamma = .8 * pdf[:, :1] * tf.constant([[1., 0.]], DTYPE)
    dot_beta = .8 * pdf[:, 1:] * tf.constant([[0., 1.]], DTYPE)
    floor = tf.constant(1e-300, DTYPE)

    def dot_shape(value):
        return tf.concat((tf.shape(value), [2]), axis=0)

    def one_axis_dots(core, dot_core):
        coefficients = tf.squeeze(core, axis=(1, 3))
        dot_coefficients = tf.squeeze(dot_core, axis=(1, 3))
        normalizer = _one_axis_normalizer(core)
        dot_normalizer = 2. * tf.reduce_sum(coefficients[:, :, None] * dot_coefficients, axis=1)
        amplitude = tf.einsum('nl,bl->bn', basis_previous, coefficients)
        dot_amplitude = tf.einsum('nl,blp->bnp', basis_previous, dot_coefficients)
        density = tf.square(amplitude) / normalizer[:, None]
        dot_density = (2. * amplitude[:, :, None] * dot_amplitude / normalizer[:, None, None]
                       - density[:, :, None] * dot_normalizer[:, None, :] / normalizer[:, None, None])
        return dot_normalizer, dot_density / tf.maximum(density[:, :, None], floor)

    def two_axis_dots(core0, core1, dot0, dot1):
        left, right = tf.squeeze(core0, axis=1), tf.squeeze(core1, axis=3)
        dot_left, dot_right = tf.squeeze(dot0, axis=1), tf.squeeze(dot1, axis=3)
        left_mass = tf.einsum('blr,bls->brs', left, left)
        right_mass = tf.einsum('brl,bsl->brs', right, right)
        dot_left_mass = tf.einsum('blrp,bls->brsp', dot_left, left) + tf.einsum('blr,blsp->brsp', left, dot_left)
        dot_right_mass = tf.einsum('brlp,bsl->brsp', dot_right, right) + tf.einsum('brl,bslp->brsp', right, dot_right)
        dot_normalizer = tf.reduce_sum(dot_left_mass * right_mass[:, :, :, None]
                                       + left_mass[:, :, :, None] * dot_right_mass, axis=(1, 2))
        evaluated = tf.einsum('nl,blr->bnr', basis_previous, left)
        dot_evaluated = tf.einsum('nl,blrp->bnrp', basis_previous, dot_left)
        numerator = tf.einsum('bnr,bns,brs->bn', evaluated, evaluated, right_mass)
        dot_numerator = (tf.einsum('bnrp,bns,brs->bnp', dot_evaluated, evaluated, right_mass)
                         + tf.einsum('bnr,bnsp,brs->bnp', evaluated, dot_evaluated, right_mass)
                         + tf.einsum('bnr,bns,brsp->bnp', evaluated, evaluated, dot_right_mass))
        normalizer = _two_axis_normalizer(core0, core1)
        density = numerator / normalizer[:, None]
        dot_density = dot_numerator / normalizer[:, None, None] - density[:, :, None] * dot_normalizer[:, None, :] / normalizer[:, None, None]
        return dot_normalizer, dot_density / tf.maximum(density[:, :, None], floor)

    def solve_directional(design, target, weights, solution, dot_design, dot_target):
        raw_norms = tf.sqrt(tf.reduce_sum(weights[None, :, None] * tf.square(design), axis=1))
        dot_raw = tf.reduce_sum(weights[None, :, None, None] * design[:, :, :, None] * dot_design, axis=1) / tf.maximum(raw_norms[:, :, None], _FLOAT64_EPS)
        indices = tf.argmax(raw_norms, axis=1, output_type=tf.int32)
        dot_max = tf.gather(dot_raw, indices, batch_dims=1)
        sqrt_eps = tf.sqrt(_FLOAT64_EPS)
        from_max = sqrt_eps * tf.reduce_max(raw_norms, axis=1)
        scale_floor = tf.maximum(from_max, _FLOAT64_EPS)
        dot_floor = tf.where(from_max[:, None] >= _FLOAT64_EPS, sqrt_eps * dot_max, tf.zeros_like(dot_max))
        scales = tf.maximum(raw_norms, scale_floor[:, None])
        dot_scales = tf.where(raw_norms[:, :, None] >= scale_floor[:, None, None], dot_raw, dot_floor[:, None, :])
        scaled = design / scales[:, None, :]
        dot_scaled = dot_design / scales[:, None, :, None] - design[:, :, :, None] * dot_scales[:, None, :, :] / tf.square(scales[:, None, :, None])
        normal = tf.einsum('n,bnc,bnd->bcd', weights, scaled, scaled) + tf.linalg.diag(tf.constant(RIDGE, DTYPE) / tf.square(scales))
        dot_ridge = -2. * tf.constant(RIDGE, DTYPE) * dot_scales / tf.pow(scales[:, :, None], 3.)
        dot_normal = (tf.einsum('n,bncp,bnd->bcdp', weights, dot_scaled, scaled)
                      + tf.einsum('n,bnc,bndp->bcdp', weights, scaled, dot_scaled)
                      + tf.einsum('bcp,cd->bcdp', dot_ridge, tf.eye(tf.shape(scales)[1], dtype=DTYPE)))
        dot_rhs = tf.einsum('n,bncp,bn->bcp', weights, dot_scaled, target) + tf.einsum('n,bnc,bnp->bcp', weights, scaled, dot_target)
        scaled_solution = solution * scales
        factor = tf.linalg.cholesky(normal)
        dot_solution = tf.linalg.cholesky_solve(factor, dot_rhs - tf.einsum('bcdp,bd->bcp', dot_normal, scaled_solution))
        return dot_solution / scales[:, :, None] - scaled_solution[:, :, None] * dot_scales / tf.square(scales[:, :, None])

    def fit_directional(fit, dot_target, dot0, dot1):
        def body(index, dot0, dot1):
            sweep = _history_read(fit.sweeps, index)
            def axis0():
                before = tf.einsum('nl,balr->bnar', basis_previous, sweep.core1_before)
                design = tf.reshape(tf.einsum('nl,bnr->bnlr', basis_current, tf.squeeze(before, axis=3)),
                                    [tf.shape(fit.target)[0], tf.shape(fit.target)[1], (DEGREE + 1) * RANK])
                matrix = tf.einsum('nl,balrp->bnarp', basis_previous, dot1)
                blocks = tf.einsum('nl,bnrp->bnlrp', basis_current, tf.squeeze(matrix, axis=3))
                dot_design = tf.reshape(blocks, dot_shape(design))
                solution = solve_directional(design, fit.target, fit.weights, sweep.solution, dot_design, dot_target)
                return tf.reshape(solution, dot_shape(fit.fitted0)), dot1
            def axis1():
                before = tf.einsum('nl,balr->bnar', basis_current, sweep.core0_before)
                design = tf.reshape(tf.einsum('bnr,nl->bnrl', tf.squeeze(before, axis=2), basis_previous),
                                    [tf.shape(fit.target)[0], tf.shape(fit.target)[1], RANK * (DEGREE + 1)])
                matrix = tf.einsum('nl,balrp->bnarp', basis_current, dot0)
                blocks = tf.einsum('bnrp,nl->bnrlp', tf.squeeze(matrix, axis=2), basis_previous)
                dot_design = tf.reshape(blocks, dot_shape(design))
                solution = solve_directional(design, fit.target, fit.weights, sweep.solution, dot_design, dot_target)
                return dot0, tf.reshape(solution, dot_shape(fit.fitted1))
            dot0, dot1 = tf.cond(sweep.axis == 0, axis0, axis1)
            return index + 1, dot0, dot1
        _, dot0, dot1 = tf.while_loop(lambda i, *_: i < 4 * MAX_SWEEPS, body,
            (tf.constant(0), dot0, dot1), maximum_iterations=4 * MAX_SWEEPS, parallel_iterations=1)
        return dot0, dot1

    def shift_derivative(step, dot_log_target):
        indices = tf.argmax(step.log_target, axis=1, output_type=tf.int32)
        dot_shift = tf.gather(dot_log_target, indices, batch_dims=1)
        return dot_shift, .5 * step.sqrt_target[:, :, None] * (dot_log_target - dot_shift[:, None, :])

    physical = physical_nodes[None, :]
    stationary_inverse = 1. / (1. - tf.square(gamma))
    dot_initial = dot_gamma[:, None, :] * gamma[:, None, None] * (tf.square(physical) - stationary_inverse[:, None])[:, :, None]
    residual = observations[0] - 2. * tf.math.log(beta)[:, None] - physical
    dot_observation = dot_beta[:, None, :] * (tf.exp(residual) - 1.)[:, :, None] / beta[:, None, None]
    dot_shift, dot_sqrt = shift_derivative(initial, dot_initial + dot_observation)
    fit = initial.one_axis_fit
    dot_coefficients = solve_directional(fit.design, fit.target, fit.weights,
        tf.squeeze(fit.fitted_core, axis=(1, 3)), tf.zeros(dot_shape(fit.design), DTYPE), dot_sqrt)
    dot_core = tf.reshape(dot_coefficients, dot_shape(fit.fitted_core))
    dot_normalizer, dot_log_density = one_axis_dots(fit.fitted_core, dot_core)
    score = tf.zeros_like(values) + dot_shift + dot_normalizer / initial.normalizer[:, None]
    if adjacent is not None:
        def date_body(t, score, dot_log_density, dot0, dot1):
            step = _history_read(adjacent, t - 1)
            transition_dot = dot_gamma[:, None, :] * (physical_current[None, :] - gamma[:, None] * physical_previous[None, :])[:, :, None] * physical_previous[None, :, None]
            residual = observations[t] - 2. * tf.math.log(beta)[:, None] - physical_current[None, :]
            observation_dot = dot_beta[:, None, :] * (tf.exp(residual) - 1.)[:, :, None] / beta[:, None, None]
            dot_shift, dot_sqrt = shift_derivative(step, dot_log_density + transition_dot + observation_dot)
            dot0, dot1 = fit_directional(step.two_axis_fit, dot_sqrt, dot0, dot1)
            dot_normalizer, dot_log_density = two_axis_dots(step.two_axis_fit.fitted0, step.two_axis_fit.fitted1, dot0, dot1)
            score = score + dot_shift + dot_normalizer / step.normalizer[:, None]
            return t + 1, score, dot_log_density, dot0, dot1
        _, score, _, _, _ = tf.while_loop(lambda t, *_: t < int(observations.shape[0]), date_body,
            (tf.constant(1), score, dot_log_density,
             tf.zeros(dot_shape(adjacent.two_axis_fit.fitted0[0]), DTYPE),
             tf.zeros(dot_shape(adjacent.two_axis_fit.fitted1[0]), DTYPE)),
            maximum_iterations=int(observations.shape[0]) - 1, parallel_iterations=1)
    valid = status['valid_pre_regularized_score'] & tf.reduce_all(tf.math.is_finite(score), axis=1)
    normalized = dict(status)
    normalized['valid_pre_regularized_score'] = valid
    normalized['status_code'] = tf.where(valid, status['status_code'], tf.ones_like(status['status_code']))
    return likelihood, score, normalized


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
    # Every sweep solves the identical one-axis system: no fitted-core input
    # enters its design. Repeating its condition is exact common-expression reuse.
    solution, condition = _scaled_qr_solve(design, target, weights)
    core = tf.reshape(solution, [batch_size, 1, DEGREE + 1, 1])
    return _OneAxisFitTrace(
        fitted_core=core,
        condition_by_sweep=tf.repeat(condition[:, None], MAX_SWEEPS, axis=1),
        design=design,
        target=target,
        weights=weights,
    )


def _fit_two_axes_tensor(basis0, basis1, target, weights, initial_core0, initial_core1, *, record_design=False):
    batch_size = tf.shape(target)[0]
    order = tf.constant([0, 1, 1, 0], tf.int32)

    def update(index, core0, core1):
        axis = order[index % 4]
        def axis0():
            matrix1 = tf.einsum("nl,balr->bnar", basis1, core1)
            right = tf.squeeze(matrix1, axis=3)
            blocks = tf.einsum("nl,bnr->bnlr", basis0, right)
            return tf.reshape(blocks, [batch_size, tf.shape(target)[1], (DEGREE + 1) * RANK])
        def axis1():
            matrix0 = tf.einsum("nl,balr->bnar", basis0, core0)
            left = tf.squeeze(matrix0, axis=2)
            blocks = tf.einsum("bnr,nl->bnrl", left, basis1)
            return tf.reshape(blocks, [batch_size, tf.shape(target)[1], RANK * (DEGREE + 1)])
        design = tf.cond(axis == 0, axis0, axis1)
        solution, condition = _scaled_qr_solve(design, target, weights)
        new0, new1 = tf.cond(axis == 0,
            lambda: (tf.reshape(solution, [batch_size, 1, DEGREE + 1, RANK]), core1),
            lambda: (core0, tf.reshape(solution, [batch_size, RANK, DEGREE + 1, 1])))
        # The analytical replay reconstructs designs exactly from small core
        # checkpoints. Large design histories are retained only for diagnostics.
        saved_design = design if record_design else tf.zeros([batch_size, 0, 0], DTYPE)
        return _TensorSweep(index // 4, axis, saved_design, solution, condition,
                            core0, core1, new0, new1)

    first = update(tf.constant(0), initial_core0, initial_core1)
    history = _history_write(_history_arrays(first, 4 * MAX_SWEEPS), 0, first)
    def body(index, core0, core1, history):
        sweep = update(index, core0, core1)
        return index + 1, sweep.core0_after, sweep.core1_after, _history_write(history, index, sweep)
    _, core0, core1, history = tf.while_loop(
        lambda i, *_: i < 4 * MAX_SWEEPS, body,
        (tf.constant(1), first.core0_after, first.core1_after, history),
        parallel_iterations=1, maximum_iterations=4 * MAX_SWEEPS - 1)
    sweeps = tf.nest.map_structure(lambda x: x.stack(), history)
    return _TensorFit(core0, core1, tf.transpose(sweeps.condition), sweeps, target, weights)


def _report_two_axis_fit(fit):
    # Diagnostic record formatting only; no runtime numerical caller uses it.
    sweeps = tuple(_TwoAxisSweepTrace(
        i // 4, (0, 1, 1, 0)[i % 4], *(_history_read(fit.sweeps, i)[2:]))
        for i in range(4 * MAX_SWEEPS))
    return _TwoAxisFitTrace(fit.fitted0, fit.fitted1, fit.condition_by_update,
                            sweeps, fit.target, fit.weights)


def _fit_two_axes_with_trace(basis0, basis1, target, weights, initial_core0, initial_core1):
    """Diagnostic record view of the native ALS recurrence."""
    return _report_two_axis_fit(_fit_two_axes_tensor(
        basis0, basis1, target, weights, initial_core0, initial_core1, record_design=True))


def _fit_two_axes(
    basis0: tf.Tensor,
    basis1: tf.Tensor,
    target: tf.Tensor,
    weights: tf.Tensor,
    initial_core0: tf.Tensor,
    initial_core1: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    trace = _fit_two_axes_tensor(
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
