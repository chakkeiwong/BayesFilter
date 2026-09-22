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
    """Evaluate the admitted analytical SR-UKF score for the raw actual-SV lane."""

    y = _as_observation_matrix(observations)
    dim = int(y.shape[1])
    gamma_vector = _as_panel_parameter(gamma, dim, "gamma")
    beta_vector = _as_panel_parameter(beta, dim, "beta")
    sigma_vector = _as_panel_parameter(sigma, dim, "sigma")
    _validate_panel_parameters(gamma_vector, beta_vector, sigma_vector)
    srukf_rule = tf_srukf_unit_sigma_point_rule(3, rule=rule)

    axis_log_terms = []
    axis_scores = []
    axis_means = []
    axis_variances = []
    axis_factors = []
    axis_d_means = []
    axis_d_variances = []
    axis_d_factors = []
    point_counts = []
    reconstruction_residuals = []
    derivative_residuals = []
    solve_residuals = []

    for axis in range(dim):
        axis_result = _actual_sv_srukf_axis_score(
            y[:, axis],
            gamma=gamma_vector[axis],
            beta=beta_vector[axis],
            sigma=sigma_vector[axis],
            filtered_jitter=filtered_jitter,
            rule=srukf_rule,
        )
        axis_log_terms.append(axis_result["log_terms"])
        axis_scores.append(axis_result["score"])
        axis_means.append(axis_result["means"])
        axis_variances.append(axis_result["variances"])
        axis_factors.append(axis_result["factors"])
        axis_d_means.append(axis_result["d_means"])
        axis_d_variances.append(axis_result["d_variances"])
        axis_d_factors.append(axis_result["d_factors"])
        point_counts.extend(axis_result["point_counts"])
        reconstruction_residuals.extend(axis_result["reconstruction_residuals"])
        derivative_residuals.extend(axis_result["derivative_residuals"])
        solve_residuals.extend(axis_result["solve_residuals"])

    stacked_log_terms = tf.stack(axis_log_terms, axis=0)
    stacked_means = tf.transpose(tf.stack(axis_means, axis=0))
    stacked_variances = tf.transpose(tf.stack(axis_variances, axis=0))
    stacked_factors = tf.transpose(tf.stack(axis_factors, axis=0))
    covariance_path = _diagonal_panel_path(stacked_variances)
    factor_path = _diagonal_panel_path(stacked_factors)
    d_mean_path = _embed_axis_vector_derivatives(tf.stack(axis_d_means, axis=0), dim)
    d_covariance_path = _embed_axis_matrix_derivatives(tf.stack(axis_d_variances, axis=0), dim)
    d_factor_path = _embed_axis_matrix_derivatives(tf.stack(axis_d_factors, axis=0), dim)

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
        "point_count_trace": tuple(point_counts),
        "filtered_jitter": float(tf.convert_to_tensor(filtered_jitter, dtype=tf.float64).numpy()),
        "max_factor_reconstruction_residual": _max_or_zero(reconstruction_residuals),
        "max_factor_derivative_residual": _max_or_zero(derivative_residuals),
        "max_innovation_solve_residual": _max_or_zero(solve_residuals),
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
        score=tf.reshape(tf.stack(axis_scores, axis=0), [-1]),
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

    log_terms = []
    score_terms = []
    means = []
    variances = []
    factors = []
    d_means = []
    d_variances = []
    d_factors = []
    point_counts = []
    reconstruction_residuals = []
    derivative_residuals = []
    solve_residuals = []

    for time_index in range(int(observations.shape[0])):
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
        current_mean = step.filtered_mean
        current_factor = step.filtered_factor
        current_covariance = step.filtered_covariance
        d_current_mean = step.d_filtered_mean
        d_current_factor = step.d_filtered_factor
        d_current_covariance = step.d_filtered_covariance

        log_terms.append(step.log_likelihood)
        score_terms.append(step.score)
        means.append(current_mean[0])
        variances.append(current_covariance[0, 0])
        factors.append(current_factor[0, 0])
        d_means.append(d_current_mean[:, 0])
        d_variances.append(d_current_covariance[:, 0, 0])
        d_factors.append(d_current_factor[:, 0, 0])
        point_counts.append(int(step.diagnostics["point_count"].numpy()))
        reconstruction_residuals.extend(
            [
                step.diagnostics["state_factor_reconstruction_residual"],
                step.diagnostics["innovation_factor_reconstruction_residual"],
                step.diagnostics["filtered_factor_reconstruction_residual"],
            ]
        )
        derivative_residuals.append(step.diagnostics["filtered_factor_derivative_residual"])
        solve_residuals.append(step.diagnostics["innovation_solve_residual"])

    del d_current_covariance
    return {
        "log_terms": tf.stack(log_terms),
        "score": tf.reduce_sum(tf.stack(score_terms), axis=0),
        "means": tf.stack(means),
        "variances": tf.stack(variances),
        "factors": tf.stack(factors),
        "d_means": tf.stack(d_means),
        "d_variances": tf.stack(d_variances),
        "d_factors": tf.stack(d_factors),
        "point_counts": tuple(point_counts),
        "reconstruction_residuals": tuple(reconstruction_residuals),
        "derivative_residuals": tuple(derivative_residuals),
        "solve_residuals": tuple(solve_residuals),
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
    rows = []
    for time_index in range(int(diagonal_entries.shape[0])):
        rows.append(tf.linalg.diag(diagonal_entries[time_index]))
    return tf.stack(rows, axis=0)


def _embed_axis_vector_derivatives(axis_values: tf.Tensor, panel_dim: int) -> tf.Tensor:
    time_count = int(axis_values.shape[1])
    full = tf.zeros([time_count, 2 * panel_dim, panel_dim], dtype=tf.float64)
    for axis in range(panel_dim):
        local = axis_values[axis]
        for parameter_index in range(2):
            full = tf.tensor_scatter_nd_update(
                full,
                indices=[[time_index, 2 * axis + parameter_index, axis] for time_index in range(time_count)],
                updates=tf.reshape(local[:, parameter_index], [-1]),
            )
    return full


def _embed_axis_matrix_derivatives(axis_values: tf.Tensor, panel_dim: int) -> tf.Tensor:
    time_count = int(axis_values.shape[1])
    full = tf.zeros([time_count, 2 * panel_dim, panel_dim, panel_dim], dtype=tf.float64)
    for axis in range(panel_dim):
        local = axis_values[axis]
        for parameter_index in range(2):
            full = tf.tensor_scatter_nd_update(
                full,
                indices=[
                    [time_index, 2 * axis + parameter_index, axis, axis]
                    for time_index in range(time_count)
                ],
                updates=tf.reshape(local[:, parameter_index], [-1]),
            )
    return full


def _max_or_zero(values: tuple[tf.Tensor, ...] | list[tf.Tensor]) -> float:
    if not values:
        return 0.0
    stacked = tf.stack([tf.reshape(tf.convert_to_tensor(value, dtype=tf.float64), []) for value in values])
    return float(tf.reduce_max(stacked).numpy())
