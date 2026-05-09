"""TensorFlow SVD/eigen sigma-point value filters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

import tensorflow as tf

from bayesfilter.diagnostics import TFFilterDiagnostics, TFRegularizationDiagnostics
from bayesfilter.linear.svd_factor_tf import (
    eigh_logdet,
    eigh_solve,
    floor_count,
    psd_eigh,
    symmetrize,
)
from bayesfilter.results_tf import TFFilterValueResult
from bayesfilter.structural_tf import (
    TFStructuralStateSpace,
    structural_block_metadata,
    structural_filter_metadata,
)


TFSigmaPointValueBackend = Literal["tf_svd_cubature", "tf_svd_ukf"]
TFSigmaPointRuleName = Literal["cubature", "unscented"]


@dataclass(frozen=True)
class TFSigmaPointRule:
    """Fixed standardized sigma-point offsets and weights."""

    name: str
    dim: int
    offsets: tf.Tensor
    mean_weights: tf.Tensor
    covariance_weights: tf.Tensor
    polynomial_degree: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "offsets",
            tf.convert_to_tensor(self.offsets, dtype=tf.float64),
        )
        object.__setattr__(
            self,
            "mean_weights",
            tf.convert_to_tensor(self.mean_weights, dtype=tf.float64),
        )
        object.__setattr__(
            self,
            "covariance_weights",
            tf.convert_to_tensor(self.covariance_weights, dtype=tf.float64),
        )
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.offsets.shape.rank != 2 or self.offsets.shape[-1] != self.dim:
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
    def point_count(self) -> int | None:
        return self.offsets.shape[0]


@dataclass(frozen=True)
class TFSigmaPointDiagnostics:
    """Diagnostics for SVD/eigen sigma-point placement."""

    rank: tf.Tensor
    floor_count: tf.Tensor
    psd_projection_residual: tf.Tensor
    support_residual: tf.Tensor
    implemented_covariance: tf.Tensor
    raw_eigenvalues: tf.Tensor
    floored_eigenvalues: tf.Tensor


def _covariance_from_points(
    centered: tf.Tensor,
    weights: tf.Tensor,
) -> tf.Tensor:
    return symmetrize(tf.transpose(centered) @ (centered * weights[:, tf.newaxis]))


def _min_eigen_gap(eigenvalues: tf.Tensor) -> tf.Tensor:
    gaps = eigenvalues[1:] - eigenvalues[:-1]
    return tf.cond(
        tf.size(gaps) > 0,
        lambda: tf.reduce_min(tf.abs(gaps)),
        lambda: tf.constant(float("inf"), dtype=tf.float64),
    )


def _as_observation_matrix(observations: tf.Tensor) -> tf.Tensor:
    y = tf.convert_to_tensor(observations, dtype=tf.float64)
    if y.shape.rank == 1:
        y = y[:, tf.newaxis]
    if y.shape.rank != 2:
        raise ValueError("observations must be one- or two-dimensional")
    return y


def _static_num_timesteps(observations: tf.Tensor) -> int:
    n_timesteps = observations.shape[0]
    if n_timesteps is None:
        raise ValueError("SVD sigma-point filters require a static observation length")
    return int(n_timesteps)


def tf_unit_sigma_point_rule(
    dim: int,
    *,
    rule: TFSigmaPointRuleName = "cubature",
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> TFSigmaPointRule:
    """Return a standardized Gaussian sigma-point rule in ``R^dim``."""

    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive")
    eye = tf.eye(dim, dtype=tf.float64)
    if rule == "cubature":
        scale = tf.sqrt(tf.cast(dim, tf.float64))
        offsets = tf.concat([scale * eye, -scale * eye], axis=0)
        weights = tf.fill([2 * dim], tf.constant(1.0 / (2.0 * dim), dtype=tf.float64))
        return TFSigmaPointRule(
            name="cubature",
            dim=dim,
            offsets=offsets,
            mean_weights=weights,
            covariance_weights=weights,
            polynomial_degree=3,
        )
    if rule == "unscented":
        spread = alpha * alpha * (float(dim) + kappa)
        lambda_value = spread - float(dim)
        if spread <= 0.0:
            raise ValueError("alpha**2 * (dim + kappa) must be positive")
        axis_scale = tf.sqrt(tf.constant(spread, dtype=tf.float64))
        offsets = tf.concat(
            [tf.zeros([1, dim], dtype=tf.float64), axis_scale * eye, -axis_scale * eye],
            axis=0,
        )
        axis_weight = tf.constant(1.0 / (2.0 * spread), dtype=tf.float64)
        mean_weights = tf.concat(
            [
                tf.constant([lambda_value / spread], dtype=tf.float64),
                tf.fill([2 * dim], axis_weight),
            ],
            axis=0,
        )
        covariance_weights = tf.concat(
            [
                tf.constant(
                    [lambda_value / spread + (1.0 - alpha * alpha + beta)],
                    dtype=tf.float64,
                ),
                tf.fill([2 * dim], axis_weight),
            ],
            axis=0,
        )
        return TFSigmaPointRule(
            name="unscented",
            dim=dim,
            offsets=offsets,
            mean_weights=mean_weights,
            covariance_weights=covariance_weights,
            polynomial_degree=3,
        )
    raise ValueError(f"unknown sigma-point rule: {rule}")


def tf_svd_sigma_point_placement(
    mean: tf.Tensor,
    covariance: tf.Tensor,
    rule: TFSigmaPointRule,
    *,
    singular_floor: tf.Tensor | float = 0.0,
    rank_tolerance: tf.Tensor | float = 1e-12,
) -> tuple[tf.Tensor, TFSigmaPointDiagnostics]:
    """Place standardized sigma points using a symmetric SVD/eigen factor."""

    mean = tf.convert_to_tensor(mean, dtype=tf.float64)
    covariance = symmetrize(tf.convert_to_tensor(covariance, dtype=tf.float64))
    singular_floor_tensor = tf.cast(singular_floor, tf.float64)
    rank_tolerance_tensor = tf.cast(rank_tolerance, tf.float64)
    eigenvalues, floored, eigenvectors, implemented_covariance, residual = psd_eigh(
        covariance,
        singular_floor_tensor,
    )
    factor = eigenvectors @ tf.linalg.diag(tf.sqrt(floored))
    point_offsets = rule.offsets @ tf.transpose(factor)
    points = mean[tf.newaxis, :] + point_offsets
    active = eigenvalues > rank_tolerance_tensor
    rank = tf.reduce_sum(tf.cast(active, tf.int32))
    null_weights = tf.cast(tf.logical_not(active), tf.float64)
    null_projector = (
        eigenvectors
        @ tf.linalg.diag(null_weights)
        @ tf.linalg.matrix_transpose(eigenvectors)
    )
    support_residual = tf.reduce_max(tf.linalg.norm(point_offsets @ null_projector, axis=1))
    diagnostics = TFSigmaPointDiagnostics(
        rank=rank,
        floor_count=floor_count(eigenvalues, singular_floor_tensor),
        psd_projection_residual=residual,
        support_residual=support_residual,
        implemented_covariance=implemented_covariance,
        raw_eigenvalues=eigenvalues,
        floored_eigenvalues=floored,
    )
    return points, diagnostics


def tf_svd_sigma_point_log_likelihood(
    observations: tf.Tensor,
    model: TFStructuralStateSpace,
    *,
    rule: TFSigmaPointRuleName = "cubature",
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1e-12,
    rank_tolerance: tf.Tensor | float = 1e-12,
    jitter: tf.Tensor | float = 0.0,
    return_filtered: bool = False,
) -> tuple[tf.Tensor, tf.Tensor | None, tf.Tensor | None, Mapping[str, tf.Tensor]]:
    """Evaluate a structural SVD/eigen sigma-point Gaussian likelihood."""

    return tf_svd_sigma_point_log_likelihood_with_rule(
        observations,
        model,
        sigma_rule=tf_unit_sigma_point_rule(
            model.partition.state_dim + model.partition.innovation_dim,
            rule=rule,
        ),
        placement_floor=placement_floor,
        innovation_floor=innovation_floor,
        rank_tolerance=rank_tolerance,
        jitter=jitter,
        return_filtered=return_filtered,
    )


def tf_svd_sigma_point_log_likelihood_with_rule(
    observations: tf.Tensor,
    model: TFStructuralStateSpace,
    *,
    sigma_rule: TFSigmaPointRule,
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1e-12,
    rank_tolerance: tf.Tensor | float = 1e-12,
    jitter: tf.Tensor | float = 0.0,
    return_filtered: bool = False,
) -> tuple[tf.Tensor, tf.Tensor | None, tf.Tensor | None, Mapping[str, tf.Tensor]]:
    """Evaluate a structural sigma-point likelihood with a fixed rule."""

    y = _as_observation_matrix(observations)
    n_timesteps = _static_num_timesteps(y)
    aug_dim = model.partition.state_dim + model.partition.innovation_dim
    if sigma_rule.dim != aug_dim:
        raise ValueError("sigma_rule dimension must equal state_dim + innovation_dim")
    mean = tf.convert_to_tensor(model.initial_mean, dtype=tf.float64)
    covariance = symmetrize(model.initial_covariance)
    innovation_covariance = symmetrize(model.innovation_covariance)
    observation_covariance = symmetrize(model.observation_covariance)
    placement_floor_tensor = tf.cast(placement_floor, tf.float64)
    innovation_floor_tensor = tf.cast(innovation_floor, tf.float64)
    rank_tolerance_tensor = tf.cast(rank_tolerance, tf.float64)
    jitter_tensor = tf.cast(jitter, tf.float64)

    state_dim = model.partition.state_dim
    innovation_dim = model.partition.innovation_dim
    observation_dim = model.observation_dim
    obs_identity = tf.eye(observation_dim, dtype=tf.float64)
    two_pi = tf.constant(2.0 * math.pi, dtype=tf.float64)
    log_likelihood = tf.constant(0.0, dtype=tf.float64)
    max_placement_floor_count = tf.constant(0, dtype=tf.int32)
    max_innovation_floor_count = tf.constant(0, dtype=tf.int32)
    max_placement_residual = tf.constant(0.0, dtype=tf.float64)
    max_innovation_residual = tf.constant(0.0, dtype=tf.float64)
    max_support_residual = tf.constant(0.0, dtype=tf.float64)
    max_deterministic_residual = tf.constant(0.0, dtype=tf.float64)
    max_rank = tf.constant(0, dtype=tf.int32)
    min_placement_eigen_gap = tf.constant(float("inf"), dtype=tf.float64)
    min_innovation_eigen_gap = tf.constant(float("inf"), dtype=tf.float64)
    last_implemented_innovation_covariance = tf.zeros(
        [observation_dim, observation_dim],
        dtype=tf.float64,
    )
    means = []
    covariances = []

    for t in range(n_timesteps):
        aug_mean = tf.concat(
            [mean, tf.zeros([innovation_dim], dtype=tf.float64)],
            axis=0,
        )
        upper = tf.concat(
            [covariance, tf.zeros([state_dim, innovation_dim], dtype=tf.float64)],
            axis=1,
        )
        lower = tf.concat(
            [tf.zeros([innovation_dim, state_dim], dtype=tf.float64), innovation_covariance],
            axis=1,
        )
        aug_covariance = tf.concat([upper, lower], axis=0)
        aug_points, placement = tf_svd_sigma_point_placement(
            aug_mean,
            aug_covariance,
            sigma_rule,
            singular_floor=placement_floor_tensor,
            rank_tolerance=rank_tolerance_tensor,
        )
        previous_points = aug_points[:, :state_dim]
        innovation_points = aug_points[:, state_dim:]
        predicted_points = model.transition(previous_points, innovation_points)
        residuals = model.deterministic_residual(
            previous_points,
            innovation_points,
            predicted_points,
        )
        if residuals.shape[-1] == 0:
            deterministic_residual = tf.constant(0.0, dtype=tf.float64)
        else:
            deterministic_residual = tf.reduce_max(tf.abs(residuals))
        predicted_mean = tf.linalg.matvec(
            tf.transpose(predicted_points),
            sigma_rule.mean_weights,
        )
        centered_x = predicted_points - predicted_mean[tf.newaxis, :]
        predicted_covariance = _covariance_from_points(
            centered_x,
            sigma_rule.covariance_weights,
        )
        observation_points = model.observe(predicted_points)
        observation_mean = tf.linalg.matvec(
            tf.transpose(observation_points),
            sigma_rule.mean_weights,
        )
        centered_y = observation_points - observation_mean[tf.newaxis, :]
        raw_innovation_covariance = symmetrize(
            _covariance_from_points(centered_y, sigma_rule.covariance_weights)
            + observation_covariance
            + jitter_tensor * obs_identity
        )
        cross_covariance = tf.transpose(centered_x) @ (
            centered_y * sigma_rule.covariance_weights[:, tf.newaxis]
        )
        innovation = y[t] - observation_mean
        (
            innovation_eigenvalues,
            innovation_floored,
            innovation_eigenvectors,
            implemented_innovation_covariance,
            innovation_residual,
        ) = psd_eigh(raw_innovation_covariance, innovation_floor_tensor)
        solve_innovation = eigh_solve(
            innovation_eigenvectors,
            innovation_floored,
            innovation,
        )
        innovation_precision = eigh_solve(
            innovation_eigenvectors,
            innovation_floored,
            obs_identity,
        )
        mahalanobis = tf.reduce_sum(innovation * solve_innovation)
        log_det = eigh_logdet(innovation_floored)
        contribution = -0.5 * (
            tf.cast(observation_dim, tf.float64) * tf.math.log(two_pi)
            + log_det
            + mahalanobis
        )
        kalman_gain = cross_covariance @ innovation_precision
        mean = predicted_mean + tf.linalg.matvec(kalman_gain, innovation)
        covariance = symmetrize(
            predicted_covariance
            - kalman_gain
            @ implemented_innovation_covariance
            @ tf.transpose(kalman_gain)
        )
        log_likelihood = log_likelihood + contribution
        max_placement_floor_count = tf.maximum(
            max_placement_floor_count,
            placement.floor_count,
        )
        max_innovation_floor_count = tf.maximum(
            max_innovation_floor_count,
            floor_count(innovation_eigenvalues, innovation_floor_tensor),
        )
        max_placement_residual = tf.maximum(
            max_placement_residual,
            placement.psd_projection_residual,
        )
        max_innovation_residual = tf.maximum(max_innovation_residual, innovation_residual)
        max_support_residual = tf.maximum(max_support_residual, placement.support_residual)
        min_placement_eigen_gap = tf.minimum(
            min_placement_eigen_gap,
            _min_eigen_gap(placement.raw_eigenvalues),
        )
        min_innovation_eigen_gap = tf.minimum(
            min_innovation_eigen_gap,
            _min_eigen_gap(innovation_eigenvalues),
        )
        max_deterministic_residual = tf.maximum(
            max_deterministic_residual,
            deterministic_residual,
        )
        max_rank = tf.maximum(max_rank, placement.rank)
        last_implemented_innovation_covariance = implemented_innovation_covariance
        if return_filtered:
            means.append(mean)
            covariances.append(covariance)

    filtered_means = tf.stack(means, axis=0) if return_filtered else None
    filtered_covariances = tf.stack(covariances, axis=0) if return_filtered else None
    diagnostics = {
        "rule": tf.constant(sigma_rule.name),
        "augmented_dim": tf.constant(aug_dim, dtype=tf.int32),
        "point_count": tf.constant(sigma_rule.point_count, dtype=tf.int32),
        "polynomial_degree": tf.constant(sigma_rule.polynomial_degree, dtype=tf.int32),
        "max_integration_rank": max_rank,
        "placement_floor_count": max_placement_floor_count,
        "innovation_floor_count": max_innovation_floor_count,
        "placement_psd_projection_residual": max_placement_residual,
        "innovation_psd_projection_residual": max_innovation_residual,
        "support_residual": max_support_residual,
        "deterministic_residual": max_deterministic_residual,
        "min_placement_eigen_gap": min_placement_eigen_gap,
        "min_innovation_eigen_gap": min_innovation_eigen_gap,
        "implemented_innovation_covariance": last_implemented_innovation_covariance,
    }
    return log_likelihood, filtered_means, filtered_covariances, diagnostics


def _backend_rule(backend: TFSigmaPointValueBackend) -> TFSigmaPointRuleName:
    if backend == "tf_svd_cubature":
        return "cubature"
    if backend == "tf_svd_ukf":
        return "unscented"
    raise ValueError(f"unknown TensorFlow SVD sigma-point backend: {backend}")


def tf_svd_sigma_point_filter(
    observations: tf.Tensor,
    model: TFStructuralStateSpace,
    *,
    backend: TFSigmaPointValueBackend = "tf_svd_cubature",
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1e-12,
    rank_tolerance: tf.Tensor | float = 1e-12,
    jitter: tf.Tensor | float = 0.0,
    return_filtered: bool = False,
) -> TFFilterValueResult:
    """Dispatch to a TF structural SVD/eigen sigma-point value filter."""

    rule = _backend_rule(backend)
    value, filtered_means, filtered_covariances, raw_diagnostics = (
        tf_svd_sigma_point_log_likelihood(
            observations,
            model,
            rule=rule,
            placement_floor=placement_floor,
            innovation_floor=innovation_floor,
            rank_tolerance=rank_tolerance,
            jitter=jitter,
            return_filtered=return_filtered,
        )
    )
    block_metadata = dict(structural_block_metadata(model))
    extra = {
        **block_metadata,
        "rule": rule,
        "augmented_dim": raw_diagnostics["augmented_dim"],
        "point_count": raw_diagnostics["point_count"],
        "polynomial_degree": raw_diagnostics["polynomial_degree"],
        "max_integration_rank": raw_diagnostics["max_integration_rank"],
        "support_residual": raw_diagnostics["support_residual"],
        "deterministic_residual": raw_diagnostics["deterministic_residual"],
        "min_placement_eigen_gap": raw_diagnostics["min_placement_eigen_gap"],
        "min_innovation_eigen_gap": raw_diagnostics["min_innovation_eigen_gap"],
        "factorization": "tf.linalg.eigh",
        "derivative_status_reason": "SVD sigma-point derivatives are not certified in value Phase 4.",
    }
    diagnostics = TFFilterDiagnostics(
        backend=backend,
        mask_convention="none",
        regularization=TFRegularizationDiagnostics(
            jitter=tf.convert_to_tensor(jitter, dtype=tf.float64),
            singular_floor=tf.convert_to_tensor(innovation_floor, dtype=tf.float64),
            floor_count=raw_diagnostics["innovation_floor_count"],
            psd_projection_residual=raw_diagnostics["innovation_psd_projection_residual"],
            implemented_covariance=raw_diagnostics["implemented_innovation_covariance"],
            branch_label="svd_sigma_point_value",
            derivative_target="blocked",
        ),
        extra=extra,
    )
    return TFFilterValueResult(
        log_likelihood=value,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
        metadata=structural_filter_metadata(
            model,
            filter_name=f"{backend}_filter",
            differentiability_status="value_only",
            compiled_status="eager_tf",
        ),
        diagnostics=diagnostics,
    )
