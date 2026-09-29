"""Actual-SV augmented-noise analytical SR-UKF adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.nonlinear.srukf_factor_tf import (
    TFSRUKFStepDerivatives,
    tf_srukf_factor_score_step,
    tf_srukf_unit_sigma_point_rule,
)

_STD_NORMAL = tfp.distributions.Normal(
    loc=tf.constant(0.0, dtype=tf.float64),
    scale=tf.constant(1.0, dtype=tf.float64),
)


@dataclass(frozen=True)
class ActualSVSRUKFPanelScoreResult:
    """Independent-panel actual-SV SR-UKF value and score result."""

    log_likelihood: tf.Tensor
    score: tf.Tensor
    log_normalizers: tf.Tensor
    mean_path: tf.Tensor
    covariance_path: tf.Tensor
    factor_path: tf.Tensor
    d_mean_path: tf.Tensor
    d_covariance_path: tf.Tensor
    d_factor_path: tf.Tensor
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in (
            "log_likelihood",
            "score",
            "log_normalizers",
            "mean_path",
            "covariance_path",
            "factor_path",
            "d_mean_path",
            "d_covariance_path",
            "d_factor_path",
        ):
            object.__setattr__(self, name, tf.convert_to_tensor(getattr(self, name), dtype=tf.float64))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def actual_transformed_sv_independent_panel_augmented_noise_srukf_score(
    observations: tf.Tensor,
    *,
    gamma: float | tf.Tensor,
    beta: float | tf.Tensor,
    sigma: float | tf.Tensor,
    filtered_jitter: float | tf.Tensor = 1e-12,
    rule: str = "cubature",
) -> ActualSVSRUKFPanelScoreResult:
    """Evaluate the historical tiny-fixture augmented-noise SR-UKF contract."""

    y = _as_observation_matrix(observations)
    dim = int(y.shape[1])
    gamma_vector = _as_panel_parameter(gamma, dim, "gamma")
    beta_vector = _as_panel_parameter(beta, dim, "beta")
    sigma_vector = _as_panel_parameter(sigma, dim, "sigma")
    _validate_panel_parameters(gamma_vector, beta_vector, sigma_vector)
    srukf_rule = tf_srukf_unit_sigma_point_rule(3, rule=rule)

    # This compatibility route preserves the independent scalar panel law.
    # A fixed-shape TensorFlow map composes the legacy scalar step with XLA;
    # it is not an admitted batch-native NeuTra training target.
    time_count = int(y.shape[0])
    output_signature = {
        "log_terms": tf.TensorSpec([time_count], tf.float64),
        "score": tf.TensorSpec([2], tf.float64),
        "means": tf.TensorSpec([time_count], tf.float64),
        "variances": tf.TensorSpec([time_count], tf.float64),
        "factors": tf.TensorSpec([time_count], tf.float64),
        "d_means": tf.TensorSpec([time_count, 2], tf.float64),
        "d_variances": tf.TensorSpec([time_count, 2], tf.float64),
        "d_factors": tf.TensorSpec([time_count, 2], tf.float64),
        "reconstruction_residuals": tf.TensorSpec([time_count], tf.float64),
        "derivative_residuals": tf.TensorSpec([time_count], tf.float64),
        "solve_residuals": tf.TensorSpec([time_count], tf.float64),
    }

    @tf.function(input_signature=(), jit_compile=True)
    def run_panel():
        return tf.map_fn(
            lambda row: _actual_sv_srukf_axis_score(
                row[0], gamma=row[1], beta=row[2], sigma=row[3],
                filtered_jitter=filtered_jitter, rule=srukf_rule),
            (tf.transpose(y), gamma_vector, beta_vector, sigma_vector),
            fn_output_signature=output_signature, parallel_iterations=1,
        )

    panel = run_panel()
    stacked_log_terms = panel["log_terms"]
    stacked_means = tf.transpose(panel["means"])
    covariance_path = tf.linalg.diag(tf.transpose(panel["variances"]))
    factor_path = tf.linalg.diag(tf.transpose(panel["factors"]))
    d_mean_path = _embed_axis_vector_derivatives(panel["d_means"], dim)
    d_covariance_path = _embed_axis_matrix_derivatives(panel["d_variances"], dim)
    d_factor_path = _embed_axis_matrix_derivatives(panel["d_factors"], dim)

    diagnostics = {
        "backend": "srukf_independent_panel_actual_transformed_sv_augmented_noise_gaussian_closure_score",
        "panel_dim": dim,
        "target": "raw actual SV augmented-noise Gaussian-closure approximate likelihood",
        "target_scope": "actual_transformed_sv_augmented_noise_gaussian_closure_tiny_fixture",
        "lane_id": "lane_b_augmented_noise_gaussian_closure",
        "parameterization": "theta=[probit_gamma, log_beta] per coordinate with fixed sigma",
        "score_provenance": "manual_factor_branch_analytical_score",
        "wrapper_score_contract": "factor_propagating_srukf_manual_score",
        "augmented_variable": "A_t=(H_{t-1}, U_t, E_t)",
        "transition_map": "H_t=gamma*H_{t-1}+U_t",
        "observation_map": "Y_t=beta*exp(H_t/2)*E_t",
        "rule": srukf_rule.name,
        "point_count_trace": (srukf_rule.point_count,) * (dim * time_count),
        "filtered_jitter": float(tf.convert_to_tensor(filtered_jitter, dtype=tf.float64).numpy()),
        "max_factor_reconstruction_residual": float(tf.reduce_max(panel["reconstruction_residuals"]).numpy()),
        "max_factor_derivative_residual": float(tf.reduce_max(panel["derivative_residuals"]).numpy()),
        "max_innovation_solve_residual": float(tf.reduce_max(panel["solve_residuals"]).numpy()),
        "non_claims": (
            "not exact transformed same-target admission",
            "not direct actual-SV likelihood quadrature",
            "not KSC Gaussian mixture approximation",
            "not coupled multivariate Zhao-Cui TT",
            "no generalized SV/CNS estimator",
            "no leaderboard admission before the downstream ladder",
            "no GPU or HMC readiness claim",
        ),
    }
    return ActualSVSRUKFPanelScoreResult(
        log_likelihood=tf.reduce_sum(stacked_log_terms),
        score=tf.reshape(panel["score"], [-1]),
        log_normalizers=tf.reduce_sum(stacked_log_terms, axis=0),
        mean_path=stacked_means,
        covariance_path=covariance_path,
        factor_path=factor_path,
        d_mean_path=d_mean_path,
        d_covariance_path=d_covariance_path,
        d_factor_path=d_factor_path,
        diagnostics=diagnostics,
    )


def _actual_sv_srukf_axis_score(
    observations: tf.Tensor,
    *,
    gamma: tf.Tensor,
    beta: tf.Tensor,
    sigma: tf.Tensor,
    filtered_jitter: float | tf.Tensor,
    rule,
) -> Mapping[str, object]:
    observations = tf.reshape(tf.convert_to_tensor(observations, dtype=tf.float64), [-1])
    gamma = tf.reshape(tf.convert_to_tensor(gamma, dtype=tf.float64), [])
    beta = tf.reshape(tf.convert_to_tensor(beta, dtype=tf.float64), [])
    sigma = tf.reshape(tf.convert_to_tensor(sigma, dtype=tf.float64), [])
    d_gamma = _gamma_theta_derivative(gamma)

    current_mean = tf.constant([0.0], dtype=tf.float64)
    current_factor = tf.reshape(sigma / tf.sqrt(1.0 - tf.square(gamma)), [1, 1])
    current_covariance = current_factor @ tf.transpose(current_factor)
    d_current_mean = tf.zeros([2, 1], dtype=tf.float64)
    d_initial_factor_gamma = sigma * gamma * d_gamma / tf.pow(1.0 - tf.square(gamma), 1.5)
    d_current_factor = tf.reshape(
        tf.stack([d_initial_factor_gamma, tf.constant(0.0, dtype=tf.float64)]),
        [2, 1, 1],
    )
    d_current_covariance = _scalar_factor_covariance_derivative(current_factor, d_current_factor)

    def advance(carry, time_index):
        current_mean, current_factor, d_current_mean, d_current_factor = carry[:4]
        step = tf_srukf_factor_score_step(
            tf.reshape(observations[time_index], [1]),
            tf.stack([current_mean[0], 0.0, 0.0]),
            tf.linalg.diag(tf.stack([current_factor[0, 0], sigma, 1.0])),
            transition_fn=lambda points, gamma=gamma: _actual_sv_transition(points, gamma),
            observation_fn=lambda points, gamma=gamma, beta=beta: _actual_sv_observation(
                points,
                gamma,
                beta,
            ),
            derivatives=_actual_sv_step_derivatives(
                d_current_mean=d_current_mean,
                d_current_factor=d_current_factor,
                gamma=gamma,
                beta=beta,
                d_gamma=d_gamma,
            ),
            rule=rule,
            filtered_jitter=filtered_jitter,
            branch_label="actual_sv_augmented_noise_qr_positive_weight_factor_branch",
        )
        return (
            step.filtered_mean, step.filtered_factor, step.d_filtered_mean, step.d_filtered_factor,
            step.log_likelihood, step.score, step.filtered_covariance[0, 0],
            step.d_filtered_covariance[:, 0, 0],
            tf.maximum(step.diagnostics["state_factor_reconstruction_residual"],
                       tf.maximum(step.diagnostics["innovation_factor_reconstruction_residual"],
                                  step.diagnostics["filtered_factor_reconstruction_residual"])),
            step.diagnostics["filtered_factor_derivative_residual"],
            step.diagnostics["innovation_solve_residual"],
        )

    trace = tf.scan(advance, tf.range(tf.shape(observations)[0]), initializer=(
        current_mean, current_factor, d_current_mean, d_current_factor,
        tf.zeros([], tf.float64), tf.zeros([2], tf.float64), current_covariance[0, 0],
        d_current_covariance[:, 0, 0], tf.zeros([], tf.float64), tf.zeros([], tf.float64), tf.zeros([], tf.float64),
    ), parallel_iterations=1)
    return {
        "log_terms": trace[4], "score": tf.reduce_sum(trace[5], axis=0),
        "means": trace[0][:, 0], "variances": trace[6], "factors": trace[1][:, 0, 0],
        "d_means": trace[2][:, :, 0], "d_variances": trace[7], "d_factors": trace[3][:, :, 0, 0],
        "reconstruction_residuals": trace[8], "derivative_residuals": trace[9], "solve_residuals": trace[10],
    }


def _actual_sv_step_derivatives(
    *,
    d_current_mean: tf.Tensor,
    d_current_factor: tf.Tensor,
    gamma: tf.Tensor,
    beta: tf.Tensor,
    d_gamma: tf.Tensor,
) -> TFSRUKFStepDerivatives:
    d_augmented_mean = tf.stack(
        [
            tf.stack([d_current_mean[0, 0], 0.0, 0.0]),
            tf.stack([d_current_mean[1, 0], 0.0, 0.0]),
        ],
        axis=0,
    )
    d_augmented_factor = tf.zeros([2, 3, 3], dtype=tf.float64)
    d_augmented_factor = tf.tensor_scatter_nd_update(
        d_augmented_factor,
        indices=[[0, 0, 0], [1, 0, 0]],
        updates=[d_current_factor[0, 0, 0], d_current_factor[1, 0, 0]],
    )

    def transition_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
        point_count = tf.shape(tf.convert_to_tensor(points, dtype=tf.float64))[0]
        row = tf.reshape(tf.stack([gamma, 1.0, 0.0]), [1, 1, 3])
        return tf.broadcast_to(row, [point_count, 1, 3])

    def d_transition_fn(points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        first = d_gamma * values[:, 0]
        second = tf.zeros_like(first)
        return tf.stack([first[:, tf.newaxis], second[:, tf.newaxis]], axis=0)

    def observation_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        latent = gamma * values[:, 0] + values[:, 1]
        scale = beta * tf.exp(0.5 * latent)
        jac_h = 0.5 * gamma * scale * values[:, 2]
        jac_u = 0.5 * scale * values[:, 2]
        jac_e = scale
        return tf.stack([jac_h, jac_u, jac_e], axis=1)[:, tf.newaxis, :]

    def d_observation_fn(points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        latent = gamma * values[:, 0] + values[:, 1]
        base = beta * tf.exp(0.5 * latent) * values[:, 2]
        d_gamma_component = 0.5 * d_gamma * values[:, 0] * base
        d_beta_component = base
        return tf.stack([d_gamma_component[:, tf.newaxis], d_beta_component[:, tf.newaxis]], axis=0)

    return TFSRUKFStepDerivatives(
        d_augmented_mean=d_augmented_mean,
        d_augmented_factor=d_augmented_factor,
        transition_jacobian_fn=transition_jacobian_fn,
        d_transition_fn=d_transition_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        d_observation_fn=d_observation_fn,
    )


def _actual_sv_transition(points: tf.Tensor, gamma: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(points, dtype=tf.float64)
    return (gamma * values[:, 0:1]) + values[:, 1:2]


def _actual_sv_observation(points: tf.Tensor, gamma: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(points, dtype=tf.float64)
    latent = gamma * values[:, 0:1] + values[:, 1:2]
    return beta * tf.exp(0.5 * latent) * values[:, 2:3]


def _scalar_factor_covariance_derivative(factor: tf.Tensor, d_factor: tf.Tensor) -> tf.Tensor:
    return (
        d_factor @ tf.transpose(factor)
        + factor[tf.newaxis, :, :] @ tf.linalg.matrix_transpose(d_factor)
    )


def _as_observation_matrix(values: tf.Tensor) -> tf.Tensor:
    tensor = tf.convert_to_tensor(values, dtype=tf.float64)
    if tensor.shape.rank == 1:
        tensor = tensor[:, tf.newaxis]
    if tensor.shape.rank != 2:
        raise ValueError("observations must be one- or two-dimensional")
    if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
        raise ValueError("observations must be finite")
    return tensor


def _as_panel_parameter(value: float | tf.Tensor, dim: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    if tensor.shape.rank == 0:
        tensor = tf.fill([int(dim)], tensor)
    if tensor.shape.rank != 1 or int(tensor.shape[0]) != int(dim):
        raise ValueError(f"{name} must be scalar or length panel_dim")
    if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _validate_panel_parameters(gamma: tf.Tensor, beta: tf.Tensor, sigma: tf.Tensor) -> None:
    if not bool(tf.reduce_all((gamma > 0.0) & (gamma < 1.0)).numpy()):
        raise ValueError("gamma entries must lie in (0, 1)")
    if not bool(tf.reduce_all(beta > 0.0).numpy()):
        raise ValueError("beta entries must be positive")
    if not bool(tf.reduce_all(sigma > 0.0).numpy()):
        raise ValueError("sigma entries must be positive")


def _gamma_theta_derivative(gamma: tf.Tensor) -> tf.Tensor:
    probit = _STD_NORMAL.quantile(tf.reshape(tf.convert_to_tensor(gamma, dtype=tf.float64), []))
    return tf.exp(-0.5 * tf.square(probit)) / tf.sqrt(tf.constant(2.0 * math.pi, dtype=tf.float64))


def _diagonal_panel_path(diagonal_entries: tf.Tensor) -> tf.Tensor:
    return tf.linalg.diag(diagonal_entries)


def _embed_axis_vector_derivatives(axis_values: tf.Tensor, panel_dim: int) -> tf.Tensor:
    embedded = tf.einsum("atp,an->tapn", axis_values, tf.eye(panel_dim, dtype=tf.float64))
    return tf.reshape(embedded, [tf.shape(axis_values)[1], panel_dim * 2, panel_dim])


def _embed_axis_matrix_derivatives(axis_values: tf.Tensor, panel_dim: int) -> tf.Tensor:
    return tf.linalg.diag(_embed_axis_vector_derivatives(axis_values, panel_dim))
