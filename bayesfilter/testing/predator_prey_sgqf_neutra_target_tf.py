"""Graph-native fixed-SGQF predator-prey posterior for P4."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.nonlinear.fixed_sgqf_tf import tf_fixed_sgqf_cloud
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
from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
    PP_DATASET_ID,
    PP_OBSERVATION_SHA256,
    PP_PARAMETER_LOWER,
    PP_PARAMETER_NAMES,
    PP_PARAMETER_UPPER,
    PP_TRUTH_PHYSICAL,
    _INITIAL_COVARIANCE,
    _INITIAL_MEAN,
    _LOG_TWO_PI,
    _OBSERVATION_COVARIANCE,
    _PROCESS_COVARIANCE,
    _rank2_theta,
    _semantic_hash,
    _tensor_hash,
    generate_frozen_predator_prey_dataset_tf,
    rk4_transition_value,
    rk4_transition_value_state_source_jacobians,
    source_six_probit_jacobian_value_score,
    source_uniform_prior_value_score,
)


PP_SGQF_PARAMETER_NAMES = PP_PARAMETER_NAMES
PP_SGQF_SCOPE_PREFIX = "PP-SGQF-six-probit-initial-observation-first-level"
PP_SGQF_NONCLAIMS = (
    "fixed sparse-Gaussian-quadrature approximate predator-prey filter posterior",
    "initial observation is assimilated before the first RK4 transition",
    "no positivity projection despite negative states in the frozen trajectory",
    "no HMC convergence, NeuTra training, calibration, or readiness claim",
)
_MIN_VARIANCE = tf.constant(1.0e-12, tf.float64)


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def _cholesky_derivative(
    factor: tf.Tensor, d_covariance: tf.Tensor
) -> tf.Tensor:
    """Differentiate a batched lower Cholesky factor."""

    factor_inverse = tf.linalg.triangular_solve(
        factor, tf.eye(2, batch_shape=[tf.shape(factor)[0]], dtype=tf.float64)
    )
    inner = tf.einsum(
        "bij,bpjk,bkl->bpil",
        factor_inverse,
        d_covariance,
        tf.linalg.matrix_transpose(factor_inverse),
    )
    lower = tf.linalg.band_part(inner, -1, 0)
    phi = lower - 0.5 * tf.linalg.diag(tf.linalg.diag_part(lower))
    return tf.einsum("bij,bpjk->bpik", factor, phi)


def _weighted_covariance(
    centered: tf.Tensor, weights: tf.Tensor
) -> tf.Tensor:
    return _symmetrize(tf.einsum("r,bri,brj->bij", weights, centered, centered))


def _weighted_covariance_derivative(
    centered: tf.Tensor, d_centered: tf.Tensor, weights: tf.Tensor
) -> tf.Tensor:
    return _symmetrize(
        tf.einsum("r,bpri,brj->bpij", weights, d_centered, centered)
        + tf.einsum("r,bri,bprj->bpij", weights, centered, d_centered)
    )


def pp_sgqf_likelihood_value_score_status(
    theta: Any,
    *,
    observations: tf.Tensor,
    nodes: tf.Tensor,
    weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    """Evaluate corrected-time-order PP-SGQF likelihood and manual score."""

    values = _rank2_theta(theta)
    y = tf.convert_to_tensor(observations, tf.float64)
    cloud_points = tf.convert_to_tensor(nodes, tf.float64)
    cloud_weights = tf.convert_to_tensor(weights, tf.float64)
    batch_size = int(values.shape[0])
    parameter_dim = 6

    initial_innovation_covariance = _INITIAL_COVARIANCE + _OBSERVATION_COVARIANCE
    initial_factor = tf.linalg.cholesky(initial_innovation_covariance)
    initial_innovation = y[0] - _INITIAL_MEAN
    initial_solve = tf.linalg.cholesky_solve(
        initial_factor, initial_innovation[:, None]
    )[:, 0]
    initial_log_det = 2.0 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(initial_factor))
    )
    initial_value = -0.5 * (
        2.0 * _LOG_TWO_PI
        + initial_log_det
        + tf.reduce_sum(initial_innovation * initial_solve)
    )
    initial_gain = _INITIAL_COVARIANCE @ tf.linalg.cholesky_solve(
        initial_factor, tf.eye(2, dtype=tf.float64)
    )
    initial_filtered_mean = _INITIAL_MEAN + tf.linalg.matvec(
        initial_gain, initial_innovation
    )
    initial_filtered_covariance = _symmetrize(
        _INITIAL_COVARIANCE
        - initial_gain @ initial_innovation_covariance @ tf.transpose(initial_gain)
    )
    mean = tf.broadcast_to(initial_filtered_mean[None, :], [batch_size, 2])
    covariance = tf.broadcast_to(
        initial_filtered_covariance[None, :, :], [batch_size, 2, 2]
    )
    d_mean = tf.zeros([batch_size, parameter_dim, 2], tf.float64)
    d_covariance = tf.zeros([batch_size, parameter_dim, 2, 2], tf.float64)
    total_value = tf.fill([batch_size], initial_value)
    total_score = tf.zeros([batch_size, parameter_dim], tf.float64)
    valid = tf.ones([batch_size], tf.bool)
    min_predictive_eigenvalue = tf.fill(
        [batch_size], tf.constant(float("inf"), tf.float64)
    )
    min_innovation_eigenvalue = tf.fill(
        [batch_size], tf.constant(float("inf"), tf.float64)
    )
    min_filtered_eigenvalue = tf.reduce_min(
        tf.linalg.eigvalsh(covariance), axis=1
    )

    def condition(index, *_loop_values):
        return index < tf.shape(y)[0]

    def body(
        index,
        current_mean,
        current_covariance,
        current_d_mean,
        current_d_covariance,
        value_total,
        score_total,
        current_valid,
        min_predictive,
        min_innovation,
        min_filtered,
    ):
        previous_factor = tf.linalg.cholesky(current_covariance)
        d_previous_factor = _cholesky_derivative(
            previous_factor, current_d_covariance
        )
        previous_points = current_mean[:, None, :] + tf.einsum(
            "rd,bnd->brn", cloud_points, previous_factor
        )
        d_previous_points = current_d_mean[:, :, None, :] + tf.einsum(
            "rd,bpnd->bprn", cloud_points, d_previous_factor
        )
        transition_values, transition_state_jacobian, d_transition_direct = (
            rk4_transition_value_state_source_jacobians(values, previous_points)
        )
        d_transition_values = (
            tf.einsum(
                "brij,bprj->bpri", transition_state_jacobian, d_previous_points
            )
            + d_transition_direct
        )
        predicted_mean = tf.einsum(
            "r,bri->bi", cloud_weights, transition_values
        )
        d_predicted_mean = tf.einsum(
            "r,bpri->bpi", cloud_weights, d_transition_values
        )
        centered_predicted = transition_values - predicted_mean[:, None, :]
        d_centered_predicted = (
            d_transition_values - d_predicted_mean[:, :, None, :]
        )
        predicted_covariance = _symmetrize(
            _PROCESS_COVARIANCE[None, :, :]
            + _weighted_covariance(centered_predicted, cloud_weights)
        )
        d_predicted_covariance = _weighted_covariance_derivative(
            centered_predicted, d_centered_predicted, cloud_weights
        )
        predicted_factor = tf.linalg.cholesky(predicted_covariance)
        d_predicted_factor = _cholesky_derivative(
            predicted_factor, d_predicted_covariance
        )
        predictive_points = predicted_mean[:, None, :] + tf.einsum(
            "rd,bnd->brn", cloud_points, predicted_factor
        )
        d_predictive_points = d_predicted_mean[:, :, None, :] + tf.einsum(
            "rd,bpnd->bprn", cloud_points, d_predicted_factor
        )
        observation_mean = tf.einsum(
            "r,bri->bi", cloud_weights, predictive_points
        )
        d_observation_mean = tf.einsum(
            "r,bpri->bpi", cloud_weights, d_predictive_points
        )
        centered_observation = predictive_points - observation_mean[:, None, :]
        d_centered_observation = (
            d_predictive_points - d_observation_mean[:, :, None, :]
        )
        innovation_covariance = _symmetrize(
            _OBSERVATION_COVARIANCE[None, :, :]
            + _weighted_covariance(centered_observation, cloud_weights)
        )
        d_innovation_covariance = _weighted_covariance_derivative(
            centered_observation, d_centered_observation, cloud_weights
        )
        centered_state = predictive_points - predicted_mean[:, None, :]
        d_centered_state = d_predictive_points - d_predicted_mean[:, :, None, :]
        cross_covariance = tf.einsum(
            "r,bri,brj->bij",
            cloud_weights,
            centered_state,
            centered_observation,
        )
        d_cross_covariance = (
            tf.einsum(
                "r,bpri,brj->bpij",
                cloud_weights,
                d_centered_state,
                centered_observation,
            )
            + tf.einsum(
                "r,bri,bprj->bpij",
                cloud_weights,
                centered_state,
                d_centered_observation,
            )
        )
        innovation = y[index][None, :] - observation_mean
        d_innovation = -d_observation_mean
        innovation_factor = tf.linalg.cholesky(innovation_covariance)
        innovation_precision = tf.linalg.cholesky_solve(
            innovation_factor,
            tf.broadcast_to(tf.eye(2, dtype=tf.float64)[None, :, :], [batch_size, 2, 2]),
        )
        innovation_solve = tf.linalg.cholesky_solve(
            innovation_factor, innovation[:, :, None]
        )[:, :, 0]
        log_det = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(innovation_factor)), axis=1
        )
        increment = -0.5 * (
            2.0 * _LOG_TWO_PI
            + log_det
            + tf.reduce_sum(innovation * innovation_solve, axis=1)
        )
        trace_term = tf.einsum(
            "bij,bpji->bp", innovation_precision, d_innovation_covariance
        )
        innovation_term = 2.0 * tf.einsum(
            "bpi,bi->bp", d_innovation, innovation_solve
        )
        quadratic_term = tf.einsum(
            "bi,bpij,bj->bp",
            innovation_solve,
            d_innovation_covariance,
            innovation_solve,
        )
        score_increment = -0.5 * (
            trace_term + innovation_term - quadratic_term
        )
        gain = tf.matmul(cross_covariance, innovation_precision)
        d_gain = (
            tf.einsum(
                "bpij,bjk->bpik", d_cross_covariance, innovation_precision
            )
            - tf.einsum(
                "bij,bpjk,bkl->bpil",
                gain,
                d_innovation_covariance,
                innovation_precision,
            )
        )
        filtered_mean = predicted_mean + tf.einsum(
            "bij,bj->bi", gain, innovation
        )
        d_filtered_mean = (
            d_predicted_mean
            + tf.einsum("bpij,bj->bpi", d_gain, innovation)
            + tf.einsum("bij,bpj->bpi", gain, d_innovation)
        )
        filtered_covariance = _symmetrize(
            predicted_covariance
            - gain
            @ innovation_covariance
            @ tf.linalg.matrix_transpose(gain)
        )
        d_filtered_covariance = _symmetrize(
            d_predicted_covariance
            - tf.einsum(
                "bpij,bjk,bkl->bpil",
                d_gain,
                innovation_covariance,
                tf.linalg.matrix_transpose(gain),
            )
            - tf.einsum(
                "bij,bpjk,bkl->bpil",
                gain,
                d_innovation_covariance,
                tf.linalg.matrix_transpose(gain),
            )
            - tf.einsum(
                "bij,bjk,bpkl->bpil",
                gain,
                innovation_covariance,
                tf.linalg.matrix_transpose(d_gain),
            )
        )
        predictive_eigenvalue = tf.reduce_min(
            tf.linalg.eigvalsh(predicted_covariance), axis=1
        )
        innovation_eigenvalue = tf.reduce_min(
            tf.linalg.eigvalsh(innovation_covariance), axis=1
        )
        filtered_eigenvalue = tf.reduce_min(
            tf.linalg.eigvalsh(filtered_covariance), axis=1
        )
        step_valid = tf.logical_and(
            predictive_eigenvalue > _MIN_VARIANCE,
            tf.logical_and(
                innovation_eigenvalue > _MIN_VARIANCE,
                filtered_eigenvalue > _MIN_VARIANCE,
            ),
        )
        step_valid = tf.logical_and(
            step_valid,
            tf.logical_and(
                tf.math.is_finite(increment),
                tf.reduce_all(tf.math.is_finite(score_increment), axis=1),
            ),
        )
        return (
            index + 1,
            filtered_mean,
            filtered_covariance,
            d_filtered_mean,
            d_filtered_covariance,
            value_total + increment,
            score_total + score_increment,
            tf.logical_and(current_valid, step_valid),
            tf.minimum(min_predictive, predictive_eigenvalue),
            tf.minimum(min_innovation, innovation_eigenvalue),
            tf.minimum(min_filtered, filtered_eigenvalue),
        )

    result = tf.while_loop(
        condition,
        body,
        (
            tf.constant(1, tf.int32),
            mean,
            covariance,
            d_mean,
            d_covariance,
            total_value,
            total_score,
            valid,
            min_predictive_eigenvalue,
            min_innovation_eigenvalue,
            min_filtered_eigenvalue,
        ),
        parallel_iterations=1,
    )
    value = result[5]
    score = result[6]
    valid = result[7]
    minimum_innovation = result[9]
    condition_estimate = tf.ones_like(value)
    return value, score, {
        "status_code": tf.where(
            valid, tf.zeros_like(value, tf.int32), tf.ones_like(value, tf.int32)
        ),
        "valid_pre_regularized_score": valid,
        "floor_count_value": tf.zeros_like(value, tf.int32),
        "min_innovation_eigenvalue": minimum_innovation,
        "innovation_condition_estimate": condition_estimate,
        "min_predictive_eigenvalue": result[8],
        "min_filtered_eigenvalue": result[10],
    }


def pp_sgqf_likelihood_value_only_status(
    theta: Any,
    *,
    observations: tf.Tensor,
    nodes: tf.Tensor,
    weights: tf.Tensor,
) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
    """Evaluate the fixed-SGQF scalar recursion without parameter derivatives."""

    values = _rank2_theta(theta)
    y = tf.convert_to_tensor(observations, tf.float64)
    cloud_points = tf.convert_to_tensor(nodes, tf.float64)
    cloud_weights = tf.convert_to_tensor(weights, tf.float64)
    batch_size = int(values.shape[0])
    initial_covariance = _INITIAL_COVARIANCE + _OBSERVATION_COVARIANCE
    initial_factor = tf.linalg.cholesky(initial_covariance)
    initial_innovation = y[0] - _INITIAL_MEAN
    initial_solve = tf.linalg.cholesky_solve(
        initial_factor, initial_innovation[:, None]
    )[:, 0]
    initial_value = -0.5 * (
        2.0 * _LOG_TWO_PI
        + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(initial_factor)))
        + tf.reduce_sum(initial_innovation * initial_solve)
    )
    initial_gain = _INITIAL_COVARIANCE @ tf.linalg.cholesky_solve(
        initial_factor, tf.eye(2, dtype=tf.float64)
    )
    mean = tf.broadcast_to(
        (_INITIAL_MEAN + tf.linalg.matvec(initial_gain, initial_innovation))[None, :],
        [batch_size, 2],
    )
    covariance = tf.broadcast_to(
        _symmetrize(
            _INITIAL_COVARIANCE
            - initial_gain @ initial_covariance @ tf.transpose(initial_gain)
        )[None, :, :],
        [batch_size, 2, 2],
    )
    total_value = tf.fill([batch_size], initial_value)
    valid = tf.ones([batch_size], tf.bool)
    min_predictive = tf.fill([batch_size], tf.constant(float("inf"), tf.float64))
    min_innovation = tf.fill([batch_size], tf.constant(float("inf"), tf.float64))
    min_filtered = tf.reduce_min(tf.linalg.eigvalsh(covariance), axis=1)
    identity = tf.eye(2, batch_shape=[batch_size], dtype=tf.float64)

    def body(index, current_mean, current_covariance, value_total,
             current_valid, current_min_predictive, current_min_innovation,
             current_min_filtered):
        previous_factor = tf.linalg.cholesky(current_covariance)
        previous_points = current_mean[:, None, :] + tf.einsum(
            "rd,bnd->brn", cloud_points, previous_factor
        )
        transition_values = rk4_transition_value(values, previous_points)
        predicted_mean = tf.einsum(
            "r,bri->bi", cloud_weights, transition_values
        )
        centered_predicted = transition_values - predicted_mean[:, None, :]
        predicted_covariance = _symmetrize(
            _PROCESS_COVARIANCE[None, :, :]
            + _weighted_covariance(centered_predicted, cloud_weights)
        )
        predicted_factor = tf.linalg.cholesky(predicted_covariance)
        predictive_points = predicted_mean[:, None, :] + tf.einsum(
            "rd,bnd->brn", cloud_points, predicted_factor
        )
        observation_mean = tf.einsum(
            "r,bri->bi", cloud_weights, predictive_points
        )
        centered_observation = predictive_points - observation_mean[:, None, :]
        innovation_covariance = _symmetrize(
            _OBSERVATION_COVARIANCE[None, :, :]
            + _weighted_covariance(centered_observation, cloud_weights)
        )
        centered_state = predictive_points - predicted_mean[:, None, :]
        cross_covariance = tf.einsum(
            "r,bri,brj->bij",
            cloud_weights,
            centered_state,
            centered_observation,
        )
        innovation = y[index][None, :] - observation_mean
        innovation_factor = tf.linalg.cholesky(innovation_covariance)
        precision = tf.linalg.cholesky_solve(innovation_factor, identity)
        solve = tf.linalg.cholesky_solve(
            innovation_factor, innovation[:, :, None]
        )[:, :, 0]
        increment = -0.5 * (
            2.0 * _LOG_TWO_PI
            + 2.0 * tf.reduce_sum(
                tf.math.log(tf.linalg.diag_part(innovation_factor)), axis=1
            )
            + tf.reduce_sum(innovation * solve, axis=1)
        )
        gain = cross_covariance @ precision
        filtered_mean = predicted_mean + tf.linalg.matvec(gain, innovation)
        filtered_covariance = _symmetrize(
            predicted_covariance
            - gain
            @ innovation_covariance
            @ tf.linalg.matrix_transpose(gain)
        )
        predictive_eigenvalue = tf.reduce_min(
            tf.linalg.eigvalsh(predicted_covariance), axis=1
        )
        innovation_eigenvalue = tf.reduce_min(
            tf.linalg.eigvalsh(innovation_covariance), axis=1
        )
        filtered_eigenvalue = tf.reduce_min(
            tf.linalg.eigvalsh(filtered_covariance), axis=1
        )
        step_valid = tf.logical_and(
            predictive_eigenvalue > _MIN_VARIANCE,
            tf.logical_and(
                innovation_eigenvalue > _MIN_VARIANCE,
                filtered_eigenvalue > _MIN_VARIANCE,
            ),
        )
        step_valid = tf.logical_and(step_valid, tf.math.is_finite(increment))
        return (
            index + 1,
            filtered_mean,
            filtered_covariance,
            value_total + increment,
            tf.logical_and(current_valid, step_valid),
            tf.minimum(current_min_predictive, predictive_eigenvalue),
            tf.minimum(current_min_innovation, innovation_eigenvalue),
            tf.minimum(current_min_filtered, filtered_eigenvalue),
        )

    result = tf.while_loop(
        lambda index, *_unused: index < tf.shape(y)[0],
        body,
        (
            tf.constant(1, tf.int32),
            mean,
            covariance,
            total_value,
            valid,
            min_predictive,
            min_innovation,
            min_filtered,
        ),
        parallel_iterations=1,
    )
    return result[3], {
        "status_code": tf.where(
            result[4],
            tf.zeros([batch_size], tf.int32),
            tf.ones([batch_size], tf.int32),
        ),
        "valid_value": result[4],
        "min_predictive_eigenvalue": result[5],
        "min_innovation_eigenvalue": result[6],
        "min_filtered_eigenvalue": result[7],
    }


def pp_sgqf_posterior_value_only(
    theta: Any,
    *,
    observations: tf.Tensor,
    nodes: tf.Tensor,
    weights: tf.Tensor,
) -> tf.Tensor:
    """Return the complete source-coordinate PP-SGQF posterior scalar only."""

    likelihood, _status = pp_sgqf_likelihood_value_only_status(
        theta, observations=observations, nodes=nodes, weights=weights
    )
    prior, _ = source_uniform_prior_value_score(theta)
    jacobian, _ = source_six_probit_jacobian_value_score(theta)
    return likelihood + prior + jacobian


def pp_sgqf_likelihood_value_score(
    theta: Any,
    *,
    observations: tf.Tensor,
    nodes: tf.Tensor,
    weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    value, score, _status = pp_sgqf_likelihood_value_score_status(
        theta, observations=observations, nodes=nodes, weights=weights
    )
    return value, score


def _posterior_value_score(
    theta: Any,
    *,
    observations: tf.Tensor,
    nodes: tf.Tensor,
    weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    likelihood_value, likelihood_score = pp_sgqf_likelihood_value_score(
        theta, observations=observations, nodes=nodes, weights=weights
    )
    prior_value, prior_score = source_uniform_prior_value_score(theta)
    jacobian_value, jacobian_score = source_six_probit_jacobian_value_score(theta)
    return (
        likelihood_value + prior_value + jacobian_value,
        likelihood_score + prior_score + jacobian_score,
    )


class PredatorPreySGQFNeuTraAdapter:
    """Batch-native corrected-time-order PP-SGQF posterior adapter."""

    dtype = tf.float64
    parameter_dim = 6
    parameter_names = PP_SGQF_PARAMETER_NAMES

    def __init__(
        self,
        *,
        observations: tf.Tensor,
        sparse_level: int,
        contract: SSMTargetContract,
    ) -> None:
        cloud = tf_fixed_sgqf_cloud(dim=2, sparse_level=int(sparse_level))
        self.observations = tf.convert_to_tensor(observations, tf.float64)
        self.nodes = tf.convert_to_tensor(cloud.points, tf.float64)
        self.weights = tf.convert_to_tensor(cloud.weights, tf.float64)
        self.sparse_level = int(sparse_level)
        self.contract = contract
        self.target_scope = f"{PP_SGQF_SCOPE_PREFIX}-{self.sparse_level}"
        payload = {
            "schema": "bayesfilter.testing.predator_prey_sgqf_neutra_adapter.v1",
            "target_signature": stable_ssm_target_signature(contract),
            "dtype": self.dtype.name,
            "parameter_names": self.parameter_names,
            "sparse_level": self.sparse_level,
            "point_count": int(cloud.point_count),
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
            runtime_backend="tensorflow_fixed_sgqf_predator_prey",
            evidence_path="bayesfilter/testing/predator_prey_sgqf_neutra_target_tf.py",
            target_scope=self.target_scope,
            nonclaims=PP_SGQF_NONCLAIMS,
        )

    def log_prob(self, theta: Any) -> tf.Tensor:
        value, _score = _posterior_value_score(
            theta,
            observations=self.observations,
            nodes=self.nodes,
            weights=self.weights,
        )
        return value

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return _posterior_value_score(
            theta,
            observations=self.observations,
            nodes=self.nodes,
            weights=self.weights,
        )

    def neutra_batch_log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        likelihood_value, likelihood_score, status = (
            pp_sgqf_likelihood_value_score_status(
                theta,
                observations=self.observations,
                nodes=self.nodes,
                weights=self.weights,
            )
        )
        prior_value, prior_score = source_uniform_prior_value_score(theta)
        jacobian_value, jacobian_score = source_six_probit_jacobian_value_score(theta)
        return (
            likelihood_value + prior_value + jacobian_value,
            likelihood_score + prior_score + jacobian_score,
            status,
        )

    def target_status_telemetry(self, theta: Any) -> Mapping[str, tf.Tensor]:
        _value, _score, status = pp_sgqf_likelihood_value_score_status(
            theta,
            observations=self.observations,
            nodes=self.nodes,
            weights=self.weights,
        )
        return {
            "status_code": status["status_code"],
            "valid_pre_regularized_score": status[
                "valid_pre_regularized_score"
            ],
        }


class PredatorPreySGQFLikelihoodRecomposer:
    def __init__(self, adapter: PredatorPreySGQFNeuTraAdapter) -> None:
        self.observations = adapter.observations
        self.nodes = adapter.nodes
        self.weights = adapter.weights

    def __call__(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return pp_sgqf_likelihood_value_score(
            theta,
            observations=self.observations,
            nodes=self.nodes,
            weights=self.weights,
        )


def make_predator_prey_sgqf_neutra_adapter(
    *, sparse_level: int, observations: tf.Tensor | None = None
) -> PredatorPreySGQFNeuTraAdapter:
    if observations is None:
        _states, observations = generate_frozen_predator_prey_dataset_tf()
    y = tf.convert_to_tensor(observations, tf.float64)
    cloud = tf_fixed_sgqf_cloud(dim=2, sparse_level=int(sparse_level))
    # TensorFlow serialization can differ by device; bind identity to the
    # backend-independent cloud construction manifest instead of raw tensor
    # bytes while leaving the numerical cloud used by the filter unchanged.
    cloud_manifest = dict(cloud.manifest_payload())
    for device_value_field in (
        "points",
        "weights",
        "weight_total",
        "negative_weight_count",
    ):
        cloud_manifest.pop(device_value_field, None)
    cloud_hash = hashlib.sha256(
        json.dumps(cloud_manifest, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    contract = make_predator_prey_sgqf_target_contract(
        horizon=int(y.shape[0]),
        data_hash=_tensor_hash(y),
        sparse_level=int(sparse_level),
        cloud_hash=cloud_hash,
        point_count=int(cloud.point_count),
        negative_weight_count=int(cloud.negative_weight_count),
    )
    return PredatorPreySGQFNeuTraAdapter(
        observations=y, sparse_level=int(sparse_level), contract=contract
    )


def make_predator_prey_sgqf_target_contract(
    *,
    horizon: int,
    data_hash: str,
    sparse_level: int,
    cloud_hash: str,
    point_count: int,
    negative_weight_count: int,
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
        problem_id=f"predator-prey-fixed-sgqf-level-{int(sparse_level)}-six-probit",
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
        "parameter_order": PP_SGQF_PARAMETER_NAMES,
        "lower": tuple(float(item) for item in PP_PARAMETER_LOWER.numpy()),
        "upper": tuple(float(item) for item in PP_PARAMETER_UPPER.numpy()),
    }
    chart = ParameterChart(
        parameter_names=PP_SGQF_PARAMETER_NAMES,
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
        "filter_id": f"predator-prey-fixed-sgqf-y0-first-level-{int(sparse_level)}",
        "sparse_level": int(sparse_level),
        "cloud_construction_manifest_hash": f"sha256:{cloud_hash}",
        "point_count": int(point_count),
        "negative_weight_count": int(negative_weight_count),
        "backend": "tensorflow_xla_tf_while_loop_tensorized_batch_cloud",
        "score": "manual_cholesky_moment_rk4_forward_sensitivity",
        "time_order": "analytic_y0_update_then_tf_while_loop_y1_to_y19",
        "positivity_projection": False,
    }
    filter_program = FilterProgram(
        filter_id=str(filter_semantics["filter_id"]),
        required_model_capabilities=(
            "predator_prey_rk4",
            "additive_gaussian_process_observation",
            "fixed_sparse_gaussian_quadrature",
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
