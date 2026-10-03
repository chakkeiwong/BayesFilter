"""Historical covariance-refactorizing SR-UKF value and score primitives.

This module is retained for compatibility and bounded regression comparisons.
New compatible SR-UKF code must use the repository-default direct-factor route
in :mod:`bayesfilter.nonlinear.factor_srukf_tf`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

import tensorflow as tf

from bayesfilter.linear.qr_factor_tf import (
    cholesky_factor,
    cholesky_factor_first_derivatives,
    factor_solve,
    lower_factor_from_horizontal_stack,
    symmetrize,
)


TFSRUKFMapFn = Callable[[tf.Tensor], tf.Tensor]
TFSRUKFPointJacobianFn = Callable[[tf.Tensor], tf.Tensor]
TFSRUKFParameterDerivativeFn = Callable[[tf.Tensor], tf.Tensor]


@dataclass(frozen=True)
class TFSRUKFSigmaPointRule:
    """Fixed standardized offsets and weights for SR-UKF placement."""

    name: str
    dim: int
    offsets: tf.Tensor
    mean_weights: tf.Tensor
    covariance_weights: tf.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "offsets", tf.convert_to_tensor(self.offsets, dtype=tf.float64))
        object.__setattr__(self, "mean_weights", tf.convert_to_tensor(self.mean_weights, dtype=tf.float64))
        object.__setattr__(
            self,
            "covariance_weights",
            tf.convert_to_tensor(self.covariance_weights, dtype=tf.float64),
        )
        if int(self.dim) <= 0:
            raise ValueError("dim must be positive")
        if self.offsets.shape.rank != 2 or self.offsets.shape[1] != int(self.dim):
            raise ValueError("offsets must have shape [point_count, dim]")
        if self.mean_weights.shape.rank != 1:
            raise ValueError("mean_weights must be one-dimensional")
        if self.covariance_weights.shape.rank != 1:
            raise ValueError("covariance_weights must be one-dimensional")
        if self.mean_weights.shape[0] != self.offsets.shape[0]:
            raise ValueError("mean_weights length must match point_count")
        if self.covariance_weights.shape[0] != self.offsets.shape[0]:
            raise ValueError("covariance_weights length must match point_count")

    @property
    def point_count(self) -> int:
        return int(self.offsets.shape[0])


@dataclass(frozen=True)
class TFSRUKFStepDerivatives:
    """First derivatives required by one augmented SR-UKF step."""

    d_augmented_mean: tf.Tensor
    d_augmented_factor: tf.Tensor
    transition_jacobian_fn: TFSRUKFPointJacobianFn
    d_transition_fn: TFSRUKFParameterDerivativeFn
    observation_jacobian_fn: TFSRUKFPointJacobianFn
    d_observation_fn: TFSRUKFParameterDerivativeFn

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "d_augmented_mean",
            tf.convert_to_tensor(self.d_augmented_mean, dtype=tf.float64),
        )
        object.__setattr__(
            self,
            "d_augmented_factor",
            tf.convert_to_tensor(self.d_augmented_factor, dtype=tf.float64),
        )
        if self.d_augmented_mean.shape.rank != 2:
            raise ValueError("d_augmented_mean must have shape [parameter_dim, dim]")
        if self.d_augmented_factor.shape.rank != 3:
            raise ValueError("d_augmented_factor must have shape [parameter_dim, dim, rank]")
        if self.d_augmented_factor.shape[0] != self.d_augmented_mean.shape[0]:
            raise ValueError("derivative parameter dimensions must match")
        if self.d_augmented_factor.shape[1] != self.d_augmented_mean.shape[1]:
            raise ValueError("derivative state dimensions must match")

    @property
    def parameter_dim(self) -> int:
        return int(self.d_augmented_mean.shape[0])


@dataclass(frozen=True)
class TFSRUKFStepResult:
    """Analytical one-step SR-UKF result and handoff tensors."""

    log_likelihood: tf.Tensor
    score: tf.Tensor
    filtered_mean: tf.Tensor
    filtered_factor: tf.Tensor
    filtered_covariance: tf.Tensor
    d_filtered_mean: tf.Tensor
    d_filtered_factor: tf.Tensor
    d_filtered_covariance: tf.Tensor
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in (
            "log_likelihood",
            "score",
            "filtered_mean",
            "filtered_factor",
            "filtered_covariance",
            "d_filtered_mean",
            "d_filtered_factor",
            "d_filtered_covariance",
        ):
            object.__setattr__(self, name, tf.convert_to_tensor(getattr(self, name), dtype=tf.float64))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def tf_srukf_unit_sigma_point_rule(dim: int, *, rule: str = "cubature") -> TFSRUKFSigmaPointRule:
    """Return a standardized nonnegative-weight Gaussian point rule."""

    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive")
    eye = tf.eye(dim, dtype=tf.float64)
    if rule == "cubature":
        scale = tf.sqrt(tf.cast(dim, tf.float64))
        offsets = tf.concat([scale * eye, -scale * eye], axis=0)
        weights = tf.fill([2 * dim], tf.constant(1.0 / (2.0 * dim), dtype=tf.float64))
        return TFSRUKFSigmaPointRule(
            name="cubature",
            dim=dim,
            offsets=offsets,
            mean_weights=weights,
            covariance_weights=weights,
        )
    if rule == "unscented_unit_spread":
        scale = tf.sqrt(tf.cast(dim, tf.float64))
        offsets = tf.concat([tf.zeros([1, dim], dtype=tf.float64), scale * eye, -scale * eye], axis=0)
        axis_weight = tf.constant(1.0 / (2.0 * dim), dtype=tf.float64)
        mean_weights = tf.concat([tf.zeros([1], dtype=tf.float64), tf.fill([2 * dim], axis_weight)], axis=0)
        covariance_weights = tf.concat(
            [tf.constant([2.0], dtype=tf.float64), tf.fill([2 * dim], axis_weight)],
            axis=0,
        )
        return TFSRUKFSigmaPointRule(
            name="unscented_unit_spread",
            dim=dim,
            offsets=offsets,
            mean_weights=mean_weights,
            covariance_weights=covariance_weights,
        )
    raise ValueError(f"unknown SR-UKF rule: {rule}")


def _as_vector(value: tf.Tensor, *, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    if tensor.shape.rank != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return tensor


def _as_matrix(value: tf.Tensor, *, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    if tensor.shape.rank != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return tensor


def _weighted_mean(points: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return tf.einsum("r,rd->d", weights, points)


def _weighted_covariance(centered_left: tf.Tensor, centered_right: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return tf.einsum("r,ri,rj->ij", weights, centered_left, centered_right)


def _weighted_covariance_first_derivative(
    centered_left: tf.Tensor,
    centered_right: tf.Tensor,
    d_centered_left: tf.Tensor,
    d_centered_right: tf.Tensor,
    weights: tf.Tensor,
) -> tf.Tensor:
    return (
        tf.einsum("r,pri,rj->pij", weights, d_centered_left, centered_right)
        + tf.einsum("r,ri,prj->pij", weights, centered_left, d_centered_right)
    )


def _factor_from_centered(centered: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    with tf.control_dependencies(
        [
            tf.debugging.assert_greater_equal(
                tf.reduce_min(weights),
                tf.constant(0.0, dtype=tf.float64),
                message="SR-UKF Phase 5 backend supports nonnegative covariance weights only",
            )
        ]
    ):
        centered = tf.identity(centered)
    scaled = tf.transpose(centered * tf.sqrt(weights)[:, tf.newaxis])
    return lower_factor_from_horizontal_stack(scaled)


def _factor_reconstruction_residual(factor: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    return tf.linalg.norm(factor @ tf.transpose(factor) - covariance)


def _factor_derivative_residual(factor: tf.Tensor, d_factor: tf.Tensor, d_covariance: tf.Tensor) -> tf.Tensor:
    residuals = []
    for parameter_index in range(int(d_factor.shape[0])):
        reconstructed = (
            d_factor[parameter_index] @ tf.transpose(factor)
            + factor @ tf.transpose(d_factor[parameter_index])
        )
        residuals.append(tf.linalg.norm(reconstructed - d_covariance[parameter_index]))
    return tf.reduce_max(tf.stack(residuals)) if residuals else tf.constant(0.0, dtype=tf.float64)


def _right_solve_spd(covariance_factor: tf.Tensor, matrix: tf.Tensor) -> tf.Tensor:
    solved_t = factor_solve(covariance_factor, tf.transpose(matrix))
    return tf.transpose(solved_t)


def _logdet_from_lower_factor(factor: tf.Tensor) -> tf.Tensor:
    return 2.0 * tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(factor))))


def tf_srukf_factor_score_step(
    observation: tf.Tensor,
    augmented_mean: tf.Tensor,
    augmented_factor: tf.Tensor,
    *,
    transition_fn: TFSRUKFMapFn,
    observation_fn: TFSRUKFMapFn,
    derivatives: TFSRUKFStepDerivatives,
    rule: TFSRUKFSigmaPointRule | None = None,
    filtered_jitter: tf.Tensor | float = 0.0,
    branch_label: str = "qr_positive_weight_factor_branch",
) -> TFSRUKFStepResult:
    """Evaluate one fixed-branch factor SR-UKF value and analytical score step."""

    augmented_mean = _as_vector(augmented_mean, name="augmented_mean")
    augmented_factor = _as_matrix(augmented_factor, name="augmented_factor")
    observation = _as_vector(observation, name="observation")
    if augmented_factor.shape[0] != augmented_mean.shape[0]:
        raise ValueError("augmented_factor row count must match augmented_mean")
    rule = rule or tf_srukf_unit_sigma_point_rule(int(augmented_factor.shape[1]), rule="cubature")
    if int(rule.dim) != int(augmented_factor.shape[1]):
        raise ValueError("rule dimension must match augmented factor rank")
    if derivatives.d_augmented_mean.shape[1] != augmented_mean.shape[0]:
        raise ValueError("derivative dimension must match augmented mean")
    if derivatives.d_augmented_factor.shape[1:] != augmented_factor.shape:
        raise ValueError("d_augmented_factor shape must match augmented_factor")

    points = augmented_mean[tf.newaxis, :] + rule.offsets @ tf.transpose(augmented_factor)
    d_points = derivatives.d_augmented_mean[:, tf.newaxis, :] + tf.einsum(
        "rk,pdk->prd",
        rule.offsets,
        derivatives.d_augmented_factor,
    )

    state_points = tf.convert_to_tensor(transition_fn(points), dtype=tf.float64)
    observation_points = tf.convert_to_tensor(observation_fn(points), dtype=tf.float64)
    if state_points.shape.rank != 2:
        raise ValueError("transition_fn must return shape [point_count, state_dim]")
    if observation_points.shape.rank != 2:
        raise ValueError("observation_fn must return shape [point_count, observation_dim]")

    transition_jacobian = tf.convert_to_tensor(derivatives.transition_jacobian_fn(points), dtype=tf.float64)
    observation_jacobian = tf.convert_to_tensor(derivatives.observation_jacobian_fn(points), dtype=tf.float64)
    d_transition_direct = tf.convert_to_tensor(derivatives.d_transition_fn(points), dtype=tf.float64)
    d_observation_direct = tf.convert_to_tensor(derivatives.d_observation_fn(points), dtype=tf.float64)

    d_state_points = tf.einsum("rda,pra->prd", transition_jacobian, d_points) + d_transition_direct
    d_observation_points = tf.einsum("rma,pra->prm", observation_jacobian, d_points) + d_observation_direct

    mean_weights = rule.mean_weights
    covariance_weights = rule.covariance_weights
    predicted_mean = _weighted_mean(state_points, mean_weights)
    predicted_observation = _weighted_mean(observation_points, mean_weights)
    d_predicted_mean = tf.einsum("r,prd->pd", mean_weights, d_state_points)
    d_predicted_observation = tf.einsum("r,prm->pm", mean_weights, d_observation_points)

    centered_state = state_points - predicted_mean[tf.newaxis, :]
    centered_observation = observation_points - predicted_observation[tf.newaxis, :]
    d_centered_state = d_state_points - d_predicted_mean[:, tf.newaxis, :]
    d_centered_observation = d_observation_points - d_predicted_observation[:, tf.newaxis, :]

    state_covariance = symmetrize(_weighted_covariance(centered_state, centered_state, covariance_weights))
    innovation_covariance = symmetrize(
        _weighted_covariance(centered_observation, centered_observation, covariance_weights)
    )
    cross_covariance = _weighted_covariance(centered_state, centered_observation, covariance_weights)
    d_state_covariance = symmetrize(
        _weighted_covariance_first_derivative(
            centered_state,
            centered_state,
            d_centered_state,
            d_centered_state,
            covariance_weights,
        )
    )
    d_innovation_covariance = symmetrize(
        _weighted_covariance_first_derivative(
            centered_observation,
            centered_observation,
            d_centered_observation,
            d_centered_observation,
            covariance_weights,
        )
    )
    d_cross_covariance = _weighted_covariance_first_derivative(
        centered_state,
        centered_observation,
        d_centered_state,
        d_centered_observation,
        covariance_weights,
    )

    state_factor = _factor_from_centered(centered_state, covariance_weights)
    innovation_factor = _factor_from_centered(centered_observation, covariance_weights)
    innovation = observation - predicted_observation
    solve_weight = factor_solve(innovation_factor, innovation)
    log_likelihood = -0.5 * (
        tf.cast(tf.shape(observation)[0], tf.float64) * tf.constant(math.log(2.0 * math.pi), dtype=tf.float64)
        + _logdet_from_lower_factor(innovation_factor)
        + tf.tensordot(innovation, solve_weight, axes=1)
    )

    score_terms = []
    for parameter_index in range(derivatives.parameter_dim):
        d_innovation = -d_predicted_observation[parameter_index]
        d_s = d_innovation_covariance[parameter_index]
        trace_term = tf.linalg.trace(factor_solve(innovation_factor, d_s))
        quadratic_term = tf.tensordot(solve_weight, tf.linalg.matvec(d_s, solve_weight), axes=1)
        score_terms.append(
            -0.5
            * (
                trace_term
                + 2.0 * tf.tensordot(d_innovation, solve_weight, axes=1)
                - quadratic_term
            )
        )
    score = tf.stack(score_terms)

    gain = _right_solve_spd(innovation_factor, cross_covariance)
    d_gain_rows = []
    for parameter_index in range(derivatives.parameter_dim):
        rhs = d_cross_covariance[parameter_index] - gain @ d_innovation_covariance[parameter_index]
        d_gain_rows.append(_right_solve_spd(innovation_factor, rhs))
    d_gain = tf.stack(d_gain_rows)

    filtered_mean = predicted_mean + tf.linalg.matvec(gain, innovation)
    d_filtered_mean = (
        d_predicted_mean
        + tf.einsum("pdm,m->pd", d_gain, innovation)
        - tf.einsum("dm,pm->pd", gain, d_predicted_observation)
    )
    filtered_covariance = symmetrize(state_covariance - gain @ innovation_covariance @ tf.transpose(gain))
    d_filtered_covariance_rows = []
    for parameter_index in range(derivatives.parameter_dim):
        d_filtered_covariance_rows.append(
            symmetrize(
                d_state_covariance[parameter_index]
                - d_gain[parameter_index] @ innovation_covariance @ tf.transpose(gain)
                - gain @ d_innovation_covariance[parameter_index] @ tf.transpose(gain)
                - gain @ innovation_covariance @ tf.transpose(d_gain[parameter_index])
            )
        )
    d_filtered_covariance = tf.stack(d_filtered_covariance_rows)
    filtered_factor, d_filtered_factor = cholesky_factor_first_derivatives(
        filtered_covariance,
        d_filtered_covariance,
        jitter=filtered_jitter,
    )

    diagnostics = {
        "backend": "tf_srukf_factor_score",
        "backend_status": "historical_reference_only",
        "default_backend": "direct_factor_srukf",
        "default_backend_contract": "TFFactorSRUKFModel",
        "branch_label": branch_label,
        "score_provenance": "manual_factor_branch_analytical_score",
        "point_count": tf.convert_to_tensor(rule.point_count, dtype=tf.int32),
        "rule": rule.name,
        "state_factor_reconstruction_residual": _factor_reconstruction_residual(
            state_factor,
            state_covariance,
        ),
        "innovation_factor_reconstruction_residual": _factor_reconstruction_residual(
            innovation_factor,
            innovation_covariance,
        ),
        "filtered_factor_reconstruction_residual": _factor_reconstruction_residual(
            filtered_factor,
            filtered_covariance,
        ),
        "filtered_factor_derivative_residual": _factor_derivative_residual(
            filtered_factor,
            d_filtered_factor,
            d_filtered_covariance,
        ),
        "innovation_solve_residual": tf.linalg.norm(
            tf.linalg.matvec(innovation_covariance, solve_weight) - innovation
        ),
        "derivative_target": "implemented_factor_branch",
    }
    return TFSRUKFStepResult(
        log_likelihood=log_likelihood,
        score=score,
        filtered_mean=filtered_mean,
        filtered_factor=filtered_factor,
        filtered_covariance=filtered_covariance,
        d_filtered_mean=d_filtered_mean,
        d_filtered_factor=d_filtered_factor,
        d_filtered_covariance=d_filtered_covariance,
        diagnostics=diagnostics,
    )
