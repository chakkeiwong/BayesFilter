"""Graph-native predator-prey principal-square-root-UKF posterior for P4."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.highdim.models import p30_predator_prey_fixture_model
from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.nonlinear.batched_svd_sigma_point_tf import (
    tf_batched_svd_sigma_point_value_and_score_custom_gradient,
)
from bayesfilter.nonlinear.experimental_batched_svd_sigma_point_tf import (
    TFBatchedStructuralFirstDerivatives,
    TFBatchedStructuralStateSpace,
)
from bayesfilter.nonlinear.sigma_points_tf import tf_unit_sigma_point_rule
from bayesfilter.ssm import (
    BayesianSSMProblem,
    FilterProgram,
    ParameterChart,
    ParameterPrior,
    SSMDataSignature,
    SSMStaticShape,
    SSMTargetContract,
    stable_ssm_target_signature,
)


PP_DATASET_ID = "zhao_cui_predator_prey_T20"
PP_DATASET_SEED = 81104
PP_HORIZON = 20
PP_STATE_SHA256 = "ebd1caca85d589bfa61801e92b112b7a8e0b9d5504763cdb67b82100422f7da2"
PP_OBSERVATION_SHA256 = (
    "dc63294b6e77913aef0c92796dd2d3c7a1721a766f976fcc392cd02a70754387"
)
PP_PARAMETER_NAMES = (
    "r_source_probit",
    "K_source_probit",
    "a_source_probit",
    "s_source_probit",
    "u_source_probit",
    "v_source_probit",
)
PP_PARAMETER_LOWER = tf.constant([0.1, 110.0, 20.0, 0.1, 0.0, 0.0], tf.float64)
PP_PARAMETER_UPPER = tf.constant([1.1, 130.0, 30.0, 1.1, 1.0, 1.0], tf.float64)
PP_PARAMETER_WIDTH = PP_PARAMETER_UPPER - PP_PARAMETER_LOWER
PP_TRUTH_PHYSICAL = tf.constant([0.6, 114.0, 25.0, 0.3, 0.5, 0.5], tf.float64)
PP_UKF_SCOPE = "PP-UKF-six-probit-initial-observation-first-v1"
PP_UKF_NONCLAIMS = (
    "principal-square-root UKF approximate predator-prey filter posterior",
    "initial observation is assimilated before the first RK4 transition",
    "no positivity projection despite negative states in the frozen trajectory",
    "no HMC convergence, NeuTra training, calibration, or readiness claim",
)

_LOG_TWO_PI = tf.constant(math.log(2.0 * math.pi), tf.float64)
_INITIAL_MEAN = tf.constant([50.0, 5.0], tf.float64)
_INITIAL_COVARIANCE = tf.eye(2, dtype=tf.float64)
_PROCESS_COVARIANCE = 4.0 * tf.eye(2, dtype=tf.float64)
_OBSERVATION_COVARIANCE = 4.0 * tf.eye(2, dtype=tf.float64)
_RK4_DELTA = tf.constant(2.0, tf.float64)
_RK4_INTERNAL_STEP = tf.constant(0.1, tf.float64)
_RK4_SUBSTEPS = 20


def generate_frozen_predator_prey_dataset_tf() -> tuple[tf.Tensor, tf.Tensor]:
    """Replay the preserved seed-81104 trajectory on CPU."""

    with tf.device("/CPU:0"):
        model = p30_predator_prey_fixture_model()
        states, observations = model.simulate(
            theta=model.true_parameters(), final_time=PP_HORIZON - 1, seed=PP_DATASET_SEED
        )
        return (
            tf.convert_to_tensor(states, tf.float64),
            tf.convert_to_tensor(observations, tf.float64),
        )


def source_chart_physical_parameters(theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
    """Map six source coordinates to the declared physical parameter box."""

    values = _rank2_theta(theta)
    probabilities = 0.5 * (
        1.0 + tf.math.erf(values / tf.sqrt(tf.constant(2.0, tf.float64)))
    )
    density = tf.exp(-0.5 * tf.square(values) - 0.5 * _LOG_TWO_PI)
    physical = PP_PARAMETER_LOWER[None, :] + PP_PARAMETER_WIDTH[None, :] * probabilities
    derivative = PP_PARAMETER_WIDTH[None, :] * density
    return physical, derivative


def source_uniform_prior_value_score(theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
    """Return the independent physical Uniform-box prior value and score."""

    values = _rank2_theta(theta)
    log_volume = tf.reduce_sum(tf.math.log(PP_PARAMETER_WIDTH))
    return tf.fill(tf.shape(values)[:1], -log_volume), tf.zeros_like(values)


def source_six_probit_jacobian_value_score(
    theta: Any,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return the complete six-probit chart log Jacobian and score."""

    values = _rank2_theta(theta)
    log_density = -0.5 * tf.square(values) - 0.5 * _LOG_TWO_PI
    value = tf.reduce_sum(tf.math.log(PP_PARAMETER_WIDTH)[None, :] + log_density, axis=1)
    return value, -values


def _initial_observation_update(
    observations: tf.Tensor, batch_size: int
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Assimilate y0 under the fixed initial and observation Gaussians."""

    y0 = tf.convert_to_tensor(observations, tf.float64)[0]
    innovation_covariance = _INITIAL_COVARIANCE + _OBSERVATION_COVARIANCE
    factor = tf.linalg.cholesky(innovation_covariance)
    innovation = y0 - _INITIAL_MEAN
    solve = tf.linalg.cholesky_solve(factor, innovation[:, None])[:, 0]
    log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)))
    value = -0.5 * (
        tf.constant(2.0, tf.float64) * _LOG_TWO_PI
        + log_det
        + tf.reduce_sum(innovation * solve)
    )
    gain = _INITIAL_COVARIANCE @ tf.linalg.cholesky_solve(
        factor, tf.eye(2, dtype=tf.float64)
    )
    mean = _INITIAL_MEAN + tf.linalg.matvec(gain, innovation)
    covariance = _INITIAL_COVARIANCE - gain @ _INITIAL_COVARIANCE
    return (
        tf.broadcast_to(mean[None, :], [batch_size, 2]),
        tf.broadcast_to(covariance[None, :, :], [batch_size, 2, 2]),
        tf.fill([batch_size], value),
    )


def _rhs_value_jacobians(
    physical: tf.Tensor,
    dphysical: tf.Tensor,
    state: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return predator-prey RHS, state Jacobian, and source partial derivative."""

    r, capacity, half_sat, s_rate, u_rate, v_rate = tf.unstack(physical, axis=1)
    prey = state[..., 0]
    predator = state[..., 1]
    denominator = half_sat[:, None] + prey
    interaction = prey * predator / denominator
    logistic = prey * (1.0 - prey / capacity[:, None])
    value = tf.stack(
        (
            r[:, None] * logistic - s_rate[:, None] * interaction,
            u_rate[:, None] * interaction - v_rate[:, None] * predator,
        ),
        axis=-1,
    )

    d_interaction_prey = predator * half_sat[:, None] / tf.square(denominator)
    d_interaction_predator = prey / denominator
    j11 = (
        r[:, None] * (1.0 - 2.0 * prey / capacity[:, None])
        - s_rate[:, None] * d_interaction_prey
    )
    j12 = -s_rate[:, None] * d_interaction_predator
    j21 = u_rate[:, None] * d_interaction_prey
    j22 = u_rate[:, None] * d_interaction_predator - v_rate[:, None]
    state_jacobian = tf.stack(
        (tf.stack((j11, j12), axis=-1), tf.stack((j21, j22), axis=-1)),
        axis=-2,
    )

    interaction_a = -prey * predator / tf.square(denominator)
    zero = tf.zeros_like(prey)
    physical_partials = tf.stack(
        (
            tf.stack((logistic, zero), axis=-1),
            tf.stack(
                (r[:, None] * tf.square(prey) / tf.square(capacity)[:, None], zero),
                axis=-1,
            ),
            tf.stack(
                (-s_rate[:, None] * interaction_a, u_rate[:, None] * interaction_a),
                axis=-1,
            ),
            tf.stack((-interaction, zero), axis=-1),
            tf.stack((zero, interaction), axis=-1),
            tf.stack((zero, -predator), axis=-1),
        ),
        axis=1,
    )
    source_partials = physical_partials * dphysical[:, :, None, None]
    return value, state_jacobian, source_partials


def rk4_transition_value_state_source_jacobians(
    theta: Any, previous_points: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Propagate batched points and exact RK4 state/source sensitivities."""

    values = _rank2_theta(theta)
    points = tf.convert_to_tensor(previous_points, tf.float64)
    if points.shape.rank != 3 or points.shape[0] != values.shape[0] or points.shape[2] != 2:
        raise ValueError("predator-prey points require shape [batch, point, 2]")
    physical, dphysical = source_chart_physical_parameters(values)
    batch_size = int(values.shape[0])
    point_count = tf.shape(points)[1]
    state = points
    state_jacobian = tf.broadcast_to(
        tf.eye(2, dtype=tf.float64)[None, None, :, :],
        [batch_size, point_count, 2, 2],
    )
    source_jacobian = tf.zeros(
        [batch_size, 6, point_count, 2], dtype=tf.float64
    )

    def stage(
        stage_state: tf.Tensor,
        stage_state_jacobian: tf.Tensor,
        stage_source_jacobian: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        rhs, rhs_state_jacobian, rhs_source_partial = _rhs_value_jacobians(
            physical, dphysical, stage_state
        )
        rhs_state_sensitivity = tf.einsum(
            "brij,brjk->brik", rhs_state_jacobian, stage_state_jacobian
        )
        rhs_source_sensitivity = (
            tf.einsum(
                "brij,bprj->bpri", rhs_state_jacobian, stage_source_jacobian
            )
            + rhs_source_partial
        )
        return rhs, rhs_state_sensitivity, rhs_source_sensitivity

    def body(
        index: tf.Tensor,
        current_state: tf.Tensor,
        current_state_jacobian: tf.Tensor,
        current_source_jacobian: tf.Tensor,
    ):
        del index
        step = _RK4_INTERNAL_STEP
        k1, a1, b1 = stage(
            current_state, current_state_jacobian, current_source_jacobian
        )
        k2, a2, b2 = stage(
            current_state + 0.5 * step * k1,
            current_state_jacobian + 0.5 * step * a1,
            current_source_jacobian + 0.5 * step * b1,
        )
        k3, a3, b3 = stage(
            current_state + 0.5 * step * k2,
            current_state_jacobian + 0.5 * step * a2,
            current_source_jacobian + 0.5 * step * b2,
        )
        k4, a4, b4 = stage(
            current_state + step * k3,
            current_state_jacobian + step * a3,
            current_source_jacobian + step * b3,
        )
        scale = step / 6.0
        return (
            tf.constant(1, tf.int32),
            current_state + scale * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
            current_state_jacobian + scale * (a1 + 2.0 * a2 + 2.0 * a3 + a4),
            current_source_jacobian + scale * (b1 + 2.0 * b2 + 2.0 * b3 + b4),
        )

    def loop_body(index, current_state, current_state_jacobian, current_source_jacobian):
        _unused, next_state, next_state_jacobian, next_source_jacobian = body(
            index, current_state, current_state_jacobian, current_source_jacobian
        )
        return index + 1, next_state, next_state_jacobian, next_source_jacobian

    result = tf.while_loop(
        lambda index, *_unused: index < tf.constant(_RK4_SUBSTEPS, tf.int32),
        loop_body,
        (
            tf.constant(0, tf.int32),
            state,
            state_jacobian,
            source_jacobian,
        ),
        parallel_iterations=1,
    )
    return result[1], result[2], result[3]


def rk4_transition_value(theta: Any, previous_points: tf.Tensor) -> tf.Tensor:
    """Propagate batched predator-prey points without derivative allocation."""

    values = _rank2_theta(theta)
    points = tf.convert_to_tensor(previous_points, tf.float64)
    if points.shape.rank != 3 or points.shape[0] != values.shape[0] or points.shape[2] != 2:
        raise ValueError("predator-prey points require shape [batch, point, 2]")
    physical, _dphysical = source_chart_physical_parameters(values)

    def rhs(state: tf.Tensor) -> tf.Tensor:
        r, capacity, half_sat, s_rate, u_rate, v_rate = tf.unstack(
            physical, axis=1
        )
        prey = state[..., 0]
        predator = state[..., 1]
        interaction = prey * predator / (half_sat[:, None] + prey)
        return tf.stack(
            (
                r[:, None] * prey * (1.0 - prey / capacity[:, None])
                - s_rate[:, None] * interaction,
                u_rate[:, None] * interaction - v_rate[:, None] * predator,
            ),
            axis=-1,
        )

    def body(index: tf.Tensor, state: tf.Tensor):
        del index
        step = _RK4_INTERNAL_STEP
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * step * k1)
        k3 = rhs(state + 0.5 * step * k2)
        k4 = rhs(state + step * k3)
        return (
            tf.constant(1, tf.int32),
            state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
        )

    def loop_body(index: tf.Tensor, state: tf.Tensor):
        _unused, next_state = body(index, state)
        return index + 1, next_state

    return tf.while_loop(
        lambda index, *_unused: index < tf.constant(_RK4_SUBSTEPS, tf.int32),
        loop_body,
        (tf.constant(0, tf.int32), points),
        parallel_iterations=1,
    )[1]


def _principal_sqrt_value_factor(
    covariance: tf.Tensor, *, singular_floor: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Reproduce the admitted principal-sqrt value branch without derivatives."""

    covariance = 0.5 * (
        tf.convert_to_tensor(covariance, tf.float64)
        + tf.linalg.matrix_transpose(tf.convert_to_tensor(covariance, tf.float64))
    )
    eigenvalues, eigenvectors = tf.linalg.eigh(covariance)
    finite = tf.logical_and(
        tf.reduce_all(tf.math.is_finite(covariance), axis=(-2, -1)),
        tf.reduce_all(tf.math.is_finite(eigenvalues), axis=-1),
    )
    diagnostic_large = tf.constant(1.0e100, tf.float64)
    min_eigenvalue = tf.reduce_min(
        tf.where(
            tf.math.is_finite(eigenvalues),
            eigenvalues,
            tf.fill(tf.shape(eigenvalues), -diagnostic_large),
        ),
        axis=-1,
    )
    max_abs_entry = tf.reduce_max(
        tf.where(
            tf.math.is_finite(covariance),
            tf.abs(covariance),
            tf.fill(tf.shape(covariance), diagnostic_large),
        ),
        axis=(-2, -1),
    )
    repair_floor = tf.maximum(
        tf.convert_to_tensor(singular_floor, tf.float64),
        tf.constant(1.0e-14, tf.float64),
    )
    strict_spd = tf.logical_and(finite, min_eigenvalue >= repair_floor)
    near_spd = tf.logical_and(
        finite,
        tf.logical_and(
            min_eigenvalue >= tf.constant(-1.0e-14, tf.float64),
            max_abs_entry <= tf.constant(1.0e8, tf.float64),
        ),
    )
    roundoff_repaired = tf.logical_and(near_spd, min_eigenvalue < repair_floor)
    classified_invalid = tf.logical_not(
        tf.logical_or(strict_spd, roundoff_repaired)
    )
    dimension = int(covariance.shape[-1])
    identity = tf.eye(dimension, dtype=tf.float64)[None, :, :]
    repaired = covariance + identity * repair_floor
    accepted = tf.where(
        roundoff_repaired[:, None, None], repaired, covariance
    )
    # This guard margin is part of the admitted value/score implementation.
    accepted = accepted + identity * repair_floor
    replacement = identity * tf.maximum(repair_floor, tf.constant(1.0, tf.float64))
    safe_covariance = tf.where(
        classified_invalid[:, None, None], replacement, accepted
    )
    safe_values, safe_vectors = tf.linalg.eigh(safe_covariance)
    factor = (
        safe_vectors
        @ tf.linalg.diag(tf.sqrt(tf.maximum(safe_values, 0.0)))
        @ tf.linalg.matrix_transpose(safe_vectors)
    )
    factor = 0.5 * (factor + tf.linalg.matrix_transpose(factor))
    implemented = factor @ tf.linalg.matrix_transpose(factor)
    return (
        factor,
        implemented,
        classified_invalid,
        roundoff_repaired,
        min_eigenvalue,
    )


def pp_ukf_likelihood_value_only_status(
    theta: Any, *, observations: tf.Tensor
) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
    """Evaluate the admitted PP-UKF scalar without score-side allocation."""

    values = _rank2_theta(theta)
    y = tf.convert_to_tensor(observations, tf.float64)
    batch_size = int(values.shape[0])
    mean, covariance, total_value = _initial_observation_update(y, batch_size)
    rule = tf_unit_sigma_point_rule(4, rule="unscented")
    process_covariance = tf.broadcast_to(
        _PROCESS_COVARIANCE[None, :, :], [batch_size, 2, 2]
    )
    observation_covariance = tf.broadcast_to(
        _OBSERVATION_COVARIANCE[None, :, :], [batch_size, 2, 2]
    )
    identity2 = tf.eye(2, batch_shape=[batch_size], dtype=tf.float64)
    invalid_count = tf.zeros([batch_size], tf.int32)
    roundoff_count = tf.zeros([batch_size], tf.int32)
    min_innovation_eigenvalue = tf.fill(
        [batch_size], tf.constant(float("inf"), tf.float64)
    )

    def body(index, current_mean, current_covariance, value_total,
             current_invalid, current_roundoff, current_min_innovation):
        upper = tf.concat(
            (current_covariance, tf.zeros([batch_size, 2, 2], tf.float64)),
            axis=2,
        )
        lower = tf.concat(
            (tf.zeros([batch_size, 2, 2], tf.float64), process_covariance),
            axis=2,
        )
        augmented_covariance = tf.concat((upper, lower), axis=1)
        placement_factor, _placement_covariance, placement_invalid, placement_repair, _ = (
            _principal_sqrt_value_factor(
                augmented_covariance, singular_floor=tf.constant(0.0, tf.float64)
            )
        )
        augmented_mean = tf.concat(
            (current_mean, tf.zeros([batch_size, 2], tf.float64)), axis=1
        )
        points = augmented_mean[:, None, :] + tf.einsum(
            "ra,bda->brd", rule.offsets, placement_factor
        )
        predicted_points = (
            rk4_transition_value(values, points[:, :, :2]) + points[:, :, 2:]
        )
        predicted_mean = tf.einsum(
            "r,bri->bi", rule.mean_weights, predicted_points
        )
        centered = predicted_points - predicted_mean[:, None, :]
        predicted_covariance = tf.einsum(
            "r,bri,brj->bij", rule.covariance_weights, centered, centered
        )
        predicted_covariance = 0.5 * (
            predicted_covariance + tf.linalg.matrix_transpose(predicted_covariance)
        )
        raw_innovation_covariance = predicted_covariance + observation_covariance
        (
            _innovation_sqrt,
            implemented_innovation_covariance,
            innovation_invalid,
            innovation_repair,
            innovation_minimum,
        ) = _principal_sqrt_value_factor(
            raw_innovation_covariance,
            singular_floor=tf.constant(1.0e-12, tf.float64),
        )
        innovation_factor = tf.linalg.cholesky(implemented_innovation_covariance)
        innovation = y[index][None, :] - predicted_mean
        solve = tf.linalg.cholesky_solve(
            innovation_factor, innovation[:, :, None]
        )[:, :, 0]
        precision = tf.linalg.cholesky_solve(innovation_factor, identity2)
        log_det = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(innovation_factor)), axis=1
        )
        increment = -0.5 * (
            2.0 * _LOG_TWO_PI
            + log_det
            + tf.reduce_sum(innovation * solve, axis=1)
        )
        cross_covariance = tf.einsum(
            "r,bri,brj->bij", rule.covariance_weights, centered, centered
        )
        gain = cross_covariance @ precision
        filtered_mean = predicted_mean + tf.linalg.matvec(gain, innovation)
        filtered_covariance = (
            predicted_covariance
            - gain
            @ implemented_innovation_covariance
            @ tf.linalg.matrix_transpose(gain)
        )
        filtered_covariance = 0.5 * (
            filtered_covariance + tf.linalg.matrix_transpose(filtered_covariance)
        )
        step_invalid = tf.cast(
            tf.logical_or(placement_invalid, innovation_invalid), tf.int32
        )
        step_repair = tf.cast(
            tf.logical_or(placement_repair, innovation_repair), tf.int32
        )
        return (
            index + 1,
            filtered_mean,
            filtered_covariance,
            value_total + increment,
            tf.maximum(current_invalid, step_invalid),
            tf.maximum(current_roundoff, step_repair),
            tf.minimum(current_min_innovation, innovation_minimum),
        )

    result = tf.while_loop(
        lambda index, *_unused: index < tf.shape(y)[0],
        body,
        (
            tf.constant(1, tf.int32),
            mean,
            covariance,
            total_value,
            invalid_count,
            roundoff_count,
            min_innovation_eigenvalue,
        ),
        parallel_iterations=1,
    )
    valid = tf.logical_and(
        result[4] == 0, tf.math.is_finite(result[3])
    )
    checked_value = tf.where(
        valid,
        result[3],
        tf.fill(tf.shape(result[3]), tf.constant(-1.0e100, tf.float64)),
    )
    return checked_value, {
        "status_code": tf.where(
            valid, tf.zeros_like(result[4]), tf.ones_like(result[4])
        ),
        "valid_value": valid,
        "roundoff_repair_count": result[5],
        "min_innovation_eigenvalue": result[6],
    }


def pp_ukf_posterior_value_only(theta: Any, *, observations: tf.Tensor) -> tf.Tensor:
    """Return the complete source-coordinate PP-UKF posterior scalar only."""

    likelihood, _status = pp_ukf_likelihood_value_only_status(
        theta, observations=observations
    )
    prior, _ = source_uniform_prior_value_score(theta)
    jacobian, _ = source_six_probit_jacobian_value_score(theta)
    return likelihood + prior + jacobian


def _build_ukf_model_and_derivatives(
    theta: tf.Tensor, observations: tf.Tensor
) -> tuple[TFBatchedStructuralStateSpace, TFBatchedStructuralFirstDerivatives, tf.Tensor]:
    values = _rank2_theta(theta)
    batch_size = values.shape[0]
    if batch_size is None:
        raise ValueError("PP-UKF requires a static batch dimension for XLA")
    batch_size = int(batch_size)
    initial_mean, initial_covariance, initial_value = _initial_observation_update(
        observations, batch_size
    )
    zeros_parameter_state = tf.zeros([batch_size, 6, 2], tf.float64)
    zeros_parameter_covariance = tf.zeros([batch_size, 6, 2, 2], tf.float64)

    def transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        next_state, _state_jacobian, _source_jacobian = (
            rk4_transition_value_state_source_jacobians(values, previous)
        )
        return next_state + innovation

    def observe(states: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(states, tf.float64)

    def transition_state_jacobian(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        del innovation
        _next_state, state_jacobian, _source_jacobian = (
            rk4_transition_value_state_source_jacobians(values, previous)
        )
        return state_jacobian

    def transition_innovation_jacobian(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        del previous
        return tf.broadcast_to(
            tf.eye(2, dtype=tf.float64)[None, None, :, :],
            [batch_size, tf.shape(innovation)[1], 2, 2],
        )

    def d_transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        del innovation
        _next_state, _state_jacobian, source_jacobian = (
            rk4_transition_value_state_source_jacobians(values, previous)
        )
        return source_jacobian

    def observation_state_jacobian(states: tf.Tensor) -> tf.Tensor:
        return tf.broadcast_to(
            tf.eye(2, dtype=tf.float64)[None, None, :, :],
            [batch_size, tf.shape(states)[1], 2, 2],
        )

    def d_observation(states: tf.Tensor) -> tf.Tensor:
        return tf.zeros([batch_size, 6, tf.shape(states)[1], 2], tf.float64)

    model = TFBatchedStructuralStateSpace(
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        innovation_covariance=tf.broadcast_to(
            _PROCESS_COVARIANCE[None, :, :], [batch_size, 2, 2]
        ),
        observation_covariance=tf.broadcast_to(
            _OBSERVATION_COVARIANCE[None, :, :], [batch_size, 2, 2]
        ),
        transition_fn=transition,
        observation_fn=observe,
        name="predator_prey_initial_observation_first_principal_sqrt_ukf",
    )
    derivatives = TFBatchedStructuralFirstDerivatives(
        d_initial_mean=zeros_parameter_state,
        d_initial_covariance=zeros_parameter_covariance,
        d_innovation_covariance=zeros_parameter_covariance,
        d_observation_covariance=zeros_parameter_covariance,
        transition_state_jacobian_fn=transition_state_jacobian,
        transition_innovation_jacobian_fn=transition_innovation_jacobian,
        d_transition_fn=d_transition,
        observation_state_jacobian_fn=observation_state_jacobian,
        d_observation_fn=d_observation,
        name="predator_prey_manual_rk4_source_derivatives",
    )
    return model, derivatives, initial_value


def pp_ukf_likelihood_value_score_status(
    theta: Any,
    *,
    observations: tf.Tensor,
    principal_sqrt_backend: str = "tensorflow_eigh",
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    """Evaluate the corrected-time-order PP-UKF likelihood and score."""

    values = _rank2_theta(theta)
    y = tf.convert_to_tensor(observations, tf.float64)
    model, derivatives, initial_value = _build_ukf_model_and_derivatives(values, y)
    remaining_value, score, diagnostics = (
        tf_batched_svd_sigma_point_value_and_score_custom_gradient(
            values,
            y[1:],
            model,
            derivatives,
            backend="tf_principal_sqrt_ukf",
            placement_floor=tf.constant(0.0, tf.float64),
            innovation_floor=tf.constant(1.0e-12, tf.float64),
            spectral_gap_tolerance=tf.constant(1.0e-8, tf.float64),
            fixed_null_tolerance=tf.constant(1.0e-10, tf.float64),
            principal_sqrt_backend=principal_sqrt_backend,
            jitter=tf.constant(0.0, tf.float64),
        )
    )
    value = initial_value + remaining_value
    valid = tf.logical_and(
        tf.equal(diagnostics["principal_sqrt_target_valid_count"], 1),
        tf.logical_and(
            tf.math.is_finite(value), tf.reduce_all(tf.math.is_finite(score), axis=1)
        ),
    )
    floor_count = (
        diagnostics["placement_floor_count"] + diagnostics["innovation_floor_count"]
    )
    condition_estimate = diagnostics["max_innovation_covariance_abs_entry"] / tf.maximum(
        diagnostics["min_innovation_eigenvalue"], tf.constant(1.0e-300, tf.float64)
    )
    return value, score, {
        "status_code": tf.where(
            valid, tf.zeros_like(value, tf.int32), tf.ones_like(value, tf.int32)
        ),
        "valid_pre_regularized_score": valid,
        "floor_count_value": floor_count,
        "min_innovation_eigenvalue": diagnostics["min_innovation_eigenvalue"],
        "innovation_condition_estimate": condition_estimate,
        "principal_sqrt_target_row_class_code": diagnostics[
            "principal_sqrt_target_row_class_code"
        ],
        "roundoff_repair_count": diagnostics[
            "principal_sqrt_target_roundoff_repair_count"
        ],
        "principal_sqrt_backend_code": tf.fill(
            tf.shape(value),
            tf.constant(
                0 if principal_sqrt_backend == "compiled_custom_op" else 1,
                tf.int32,
            ),
        ),
    }


def pp_ukf_likelihood_value_score(
    theta: Any, *, observations: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    value, score, _status = pp_ukf_likelihood_value_score_status(
        theta, observations=observations
    )
    return value, score


def _posterior_value_score(
    theta: Any, *, observations: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    likelihood_value, likelihood_score = pp_ukf_likelihood_value_score(
        theta, observations=observations
    )
    prior_value, prior_score = source_uniform_prior_value_score(theta)
    jacobian_value, jacobian_score = source_six_probit_jacobian_value_score(theta)
    return (
        likelihood_value + prior_value + jacobian_value,
        likelihood_score + prior_score + jacobian_score,
    )


class PredatorPreyUKFNeuTraAdapter:
    """Batch-native corrected-time-order PP-UKF posterior adapter."""

    dtype = tf.float64
    parameter_dim = 6
    parameter_names = PP_PARAMETER_NAMES

    def __init__(self, *, observations: tf.Tensor, contract: SSMTargetContract) -> None:
        self.observations = tf.convert_to_tensor(observations, tf.float64)
        self.contract = contract
        self.target_scope = PP_UKF_SCOPE
        self.supports_retained_flat_batch = True
        self.supports_retained_value_score_status = True
        payload = {
            "schema": "bayesfilter.testing.predator_prey_ukf_neutra_adapter.v1",
            "target_signature": stable_ssm_target_signature(contract),
            "dtype": self.dtype.name,
            "parameter_names": self.parameter_names,
            "time_order": "initial_observation_then_transitions",
        }
        self._adapter_signature = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def adapter_signature(self) -> str:
        return self._adapter_signature

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_batched_principal_sqrt_ukf_predator_prey",
            evidence_path="bayesfilter/testing/predator_prey_ukf_neutra_target_tf.py",
            target_scope=self.target_scope,
            nonclaims=PP_UKF_NONCLAIMS,
        )

    def log_prob(self, theta: Any) -> tf.Tensor:
        value, _score = _posterior_value_score(theta, observations=self.observations)
        return value

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, _status = self.neutra_batch_log_prob_and_grad_status(theta)
        return value, score

    def neutra_batch_log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        likelihood_value, likelihood_score, status = pp_ukf_likelihood_value_score_status(
            theta, observations=self.observations
        )
        prior_value, prior_score = source_uniform_prior_value_score(theta)
        jacobian_value, jacobian_score = source_six_probit_jacobian_value_score(theta)
        return (
            likelihood_value + prior_value + jacobian_value,
            likelihood_score + prior_score + jacobian_score,
            status,
        )

    def target_status_telemetry(self, theta: Any) -> Mapping[str, tf.Tensor]:
        _value, _score, status = pp_ukf_likelihood_value_score_status(
            theta, observations=self.observations
        )
        return {
            "status_code": status["status_code"],
            "valid_pre_regularized_score": status[
                "valid_pre_regularized_score"
            ],
        }


class PredatorPreyUKFLikelihoodRecomposer:
    """Independent PP-UKF likelihood component for recomposition."""

    def __init__(self, adapter: PredatorPreyUKFNeuTraAdapter) -> None:
        self.observations = adapter.observations

    def __call__(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return pp_ukf_likelihood_value_score(theta, observations=self.observations)


def make_predator_prey_ukf_neutra_adapter(
    *, observations: tf.Tensor | None = None
) -> PredatorPreyUKFNeuTraAdapter:
    """Build the PP-UKF adapter bound to the frozen observations."""

    if observations is None:
        _states, observations = generate_frozen_predator_prey_dataset_tf()
    y = tf.convert_to_tensor(observations, tf.float64)
    if y.shape.rank != 2 or y.shape[1] != 2:
        raise ValueError("predator-prey observations require shape [time, 2]")
    if not bool(tf.reduce_all(tf.math.is_finite(y)).numpy()):
        raise ValueError("predator-prey observations must be finite")
    data_hash = _tensor_hash(y)
    contract = make_predator_prey_ukf_target_contract(
        horizon=int(y.shape[0]), data_hash=data_hash
    )
    return PredatorPreyUKFNeuTraAdapter(observations=y, contract=contract)


def make_predator_prey_ukf_target_contract(
    *, horizon: int, data_hash: str
) -> SSMTargetContract:
    shape = SSMStaticShape(
        horizon=int(horizon),
        state_dim=2,
        observation_dim=2,
        innovation_dim=2,
        parameter_dim=6,
    )
    model_semantics = {
        "model_id": "predator-prey-rk4-additive-gaussian-seed81104",
        "physical_truth": tuple(float(item) for item in PP_TRUTH_PHYSICAL.numpy()),
        "truth_role": "explanatory_only",
        "parameter_box": tuple(
            (float(lower), float(upper))
            for lower, upper in zip(PP_PARAMETER_LOWER.numpy(), PP_PARAMETER_UPPER.numpy())
        ),
        "initial_mean": (50.0, 5.0),
        "initial_covariance": ((1.0, 0.0), (0.0, 1.0)),
        "process_covariance": ((4.0, 0.0), (0.0, 4.0)),
        "observation_covariance": ((4.0, 0.0), (0.0, 4.0)),
        "rk4_delta": 2.0,
        "rk4_internal_step": 0.1,
        "domain_policy": "diagnose_negative_after_noise_no_projection",
        "time_order": "y0_observes_initial_state_then_transition_for_y1_onward",
    }
    problem = BayesianSSMProblem(
        problem_id="predator-prey-principal-sqrt-ukf-six-probit",
        static_shape=shape,
        data_signature=SSMDataSignature(
            dataset_id=PP_DATASET_ID,
            observation_shape=(int(horizon), 2),
            data_hash=f"sha256:{data_hash}",
        ),
        target_coordinate_convention="unconstrained",
        model_manifest={
            **model_semantics,
            "model_hash": f"sha256:{_semantic_hash(model_semantics)}",
        },
    )
    chart_semantics = {
        "transform_id": "predator-prey-six-probit-uniform-box-chart",
        "parameter_order": PP_PARAMETER_NAMES,
        "lower": tuple(float(item) for item in PP_PARAMETER_LOWER.numpy()),
        "upper": tuple(float(item) for item in PP_PARAMETER_UPPER.numpy()),
    }
    chart = ParameterChart(
        parameter_names=PP_PARAMETER_NAMES,
        unconstrained_dim=6,
        constrained_shape=(6,),
        transform_manifest={
            **chart_semantics,
            "transform_hash": f"sha256:{_semantic_hash(chart_semantics)}",
        },
        log_jacobian_convention="included_in_chart",
    )
    prior_semantics = {
        "prior_id": "predator-prey-independent-uniform-parameter-box",
        "physical_support": tuple(
            (float(lower), float(upper))
            for lower, upper in zip(PP_PARAMETER_LOWER.numpy(), PP_PARAMETER_UPPER.numpy())
        ),
        "parameter_order": ("r", "K", "a", "s", "u", "v"),
    }
    prior = ParameterPrior(
        prior_manifest={
            **prior_semantics,
            "prior_hash": f"sha256:{_semantic_hash(prior_semantics)}",
        },
        support_policy="enforced_by_transform",
        log_density_authority="graph_native",
    )
    filter_semantics = {
        "filter_id": "predator-prey-batched-principal-sqrt-ukf-y0-first-v1",
        "engine": "tf_batched_svd_sigma_point_value_and_score_custom_gradient",
        "backend": "tf_principal_sqrt_ukf",
        "principal_sqrt_backend": "tensorflow_eigh_xla_portable",
        "score": "manual_reverse_principal_sqrt_plus_manual_rk4_jacobians",
        "time_order": "analytic_y0_update_then_tf_while_loop_y1_to_y19",
        "innovation_floor": 1.0e-12,
        "positivity_projection": False,
    }
    filter_program = FilterProgram(
        filter_id=str(filter_semantics["filter_id"]),
        required_model_capabilities=(
            "predator_prey_rk4",
            "additive_gaussian_process_observation",
            "principal_square_root_ukf",
        ),
        deterministic_target_policy="deterministic",
        approximation_semantics="deterministic_approximation",
        filter_manifest={
            **filter_semantics,
            "filter_hash": f"sha256:{_semantic_hash(filter_semantics)}",
        },
    )
    return SSMTargetContract(
        problem=problem,
        chart=chart,
        prior=prior,
        filter_program=filter_program,
        frozen_transport=None,
    )


def _rank2_theta(theta: Any) -> tf.Tensor:
    values = tf.convert_to_tensor(theta, tf.float64)
    if values.shape.rank != 2 or values.shape[-1] != 6:
        raise ValueError("predator-prey target requires theta shape [batch, 6]")
    if values.shape[0] is None:
        raise ValueError("predator-prey target requires a static batch dimension")
    return values


def _tensor_hash(value: tf.Tensor) -> str:
    return hashlib.sha256(bytes(tf.io.serialize_tensor(value).numpy())).hexdigest()


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
