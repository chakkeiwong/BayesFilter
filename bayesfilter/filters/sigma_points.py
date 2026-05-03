"""Structural SVD/eigen sigma-point reference backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bayesfilter.structural import (
    FilterRunMetadata,
    StatePartition,
    StructuralFilterConfig,
    validate_filter_config,
)


@dataclass(frozen=True)
class CubatureRule:
    """Spherical-radial cubature rule for Gaussian moments."""

    dim: int

    def sigma_points(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.asarray(mean, dtype=float)
        covariance = _symmetrize(np.asarray(covariance, dtype=float))
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if mean.shape != (self.dim,):
            raise ValueError("mean dimension does not match rule dimension")
        if covariance.shape != (self.dim, self.dim):
            raise ValueError("covariance dimension does not match rule dimension")
        factor, eigenvalues = _spectral_factor(covariance)
        scale = np.sqrt(float(self.dim))
        offsets = np.concatenate([scale * factor.T, -scale * factor.T], axis=0)
        weights = np.full(2 * self.dim, 1.0 / (2.0 * self.dim))
        return mean[None, :] + offsets, weights, eigenvalues


@dataclass(frozen=True)
class SigmaPointResult:
    log_likelihood: float
    filtered_means: np.ndarray | None
    filtered_covariances: np.ndarray | None
    metadata: FilterRunMetadata
    diagnostics: dict[str, float | int | str]


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _spectral_factor(covariance: np.ndarray, floor: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(_symmetrize(covariance))
    if np.min(values) < -1e-9:
        raise np.linalg.LinAlgError("covariance has a negative eigenvalue")
    values = np.maximum(values, floor)
    return vectors * np.sqrt(values)[None, :], values


def _as_observation_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


class StructuralSVDSigmaPointFilter:
    """Approximate Gaussian structural sigma-point filter.

    The prediction step generates cubature points over the joint Gaussian
    approximation to `(previous_state, innovation)`, then calls the model's
    structural transition.  Deterministic coordinates are therefore completed
    pointwise by the model rather than noised independently.
    """

    def __init__(
        self,
        config: StructuralFilterConfig | None = None,
        *,
        eigen_floor: float = 0.0,
    ) -> None:
        self.config = config or StructuralFilterConfig(
            integration_space="innovation",
            deterministic_completion="required",
        )
        self.eigen_floor = float(eigen_floor)

    def filter(
        self,
        model,
        observations: np.ndarray,
        theta=None,
        *,
        mask: np.ndarray | None = None,
        return_filtered: bool = False,
    ) -> SigmaPointResult:
        partition: StatePartition = model.partition
        validate_filter_config(partition, self.config)
        if self.config.integration_space != "innovation":
            raise NotImplementedError(
                "the reference structural sigma-point backend currently supports innovation integration"
            )

        mean = np.asarray(model.initial_mean(theta), dtype=float).copy()
        covariance = np.asarray(model.initial_cov(theta), dtype=float).copy()
        innovation_cov = np.asarray(model.innovation_cov(theta), dtype=float)
        observation_cov = np.asarray(model.observation_cov(theta), dtype=float)
        observations = _as_observation_matrix(observations)
        if observations.shape[1] != observation_cov.shape[0]:
            raise ValueError("observations and observation_cov dimensions do not match")
        obs_mask = np.isfinite(observations) if mask is None else np.asarray(mask, dtype=bool)
        if obs_mask.shape != observations.shape:
            raise ValueError("mask must have the same shape as observations")

        aug_dim = partition.state_dim + partition.innovation_dim
        rule = CubatureRule(aug_dim)
        log_likelihood = 0.0
        means: list[np.ndarray] = []
        covariances: list[np.ndarray] = []
        min_prediction_eigenvalue = np.inf
        min_update_eigenvalue = np.inf

        for row, row_mask in zip(observations, obs_mask, strict=True):
            aug_mean = np.concatenate([mean, np.zeros(partition.innovation_dim)])
            aug_cov = np.zeros((aug_dim, aug_dim), dtype=float)
            aug_cov[: partition.state_dim, : partition.state_dim] = covariance
            aug_cov[partition.state_dim :, partition.state_dim :] = innovation_cov
            aug_points, weights, _ = rule.sigma_points(aug_mean, aug_cov)

            predicted_points = np.asarray(
                [
                    model.transition(
                        point[: partition.state_dim],
                        point[partition.state_dim :],
                        theta,
                    )
                    for point in aug_points
                ],
                dtype=float,
            )
            predicted_mean = weights @ predicted_points
            centered_x = predicted_points - predicted_mean[None, :]
            predicted_covariance = _symmetrize(centered_x.T @ (centered_x * weights[:, None]))
            pred_eigs = np.linalg.eigvalsh(predicted_covariance)
            min_prediction_eigenvalue = min(min_prediction_eigenvalue, float(np.min(pred_eigs)))

            if not np.any(row_mask):
                mean = predicted_mean
                covariance = predicted_covariance
            else:
                obs_points = _as_observation_matrix(model.observe(predicted_points, theta))
                obs_selected = obs_points[:, row_mask]
                R = observation_cov[np.ix_(row_mask, row_mask)]
                obs_mean = weights @ obs_selected
                centered_y = obs_selected - obs_mean[None, :]
                innovation_covariance = _symmetrize(
                    centered_y.T @ (centered_y * weights[:, None]) + R
                )
                cross_covariance = centered_x.T @ (centered_y * weights[:, None])
                innovation = row[row_mask] - obs_mean
                chol = np.linalg.cholesky(innovation_covariance)
                whitened = np.linalg.solve(chol, innovation)
                log_det = 2.0 * np.sum(np.log(np.diag(chol)))
                quad = float(whitened @ whitened)
                log_likelihood += -0.5 * (
                    row_mask.sum() * np.log(2.0 * np.pi) + log_det + quad
                )
                gain = np.linalg.solve(innovation_covariance.T, cross_covariance.T).T
                mean = predicted_mean + gain @ innovation
                covariance = _symmetrize(
                    predicted_covariance - gain @ innovation_covariance @ gain.T
                )

            update_eigs = np.linalg.eigvalsh(_symmetrize(covariance))
            min_update_eigenvalue = min(min_update_eigenvalue, float(np.min(update_eigs)))
            if return_filtered:
                means.append(mean.copy())
                covariances.append(covariance.copy())

        metadata = FilterRunMetadata(
            filter_name="structural_svd_sigma_point",
            partition=partition,
            integration_space=self.config.integration_space,
            deterministic_completion=self.config.deterministic_completion,
            approximation_label=self.config.approximation_label or "sigma_point_gaussian_closure",
            differentiability_status="finite_difference_smoke_only",
            compiled_status="eager_numpy",
        )
        diagnostics = {
            "rule": "cubature",
            "augmented_dim": aug_dim,
            "min_prediction_eigenvalue": min_prediction_eigenvalue,
            "min_update_eigenvalue": min_update_eigenvalue,
        }
        return SigmaPointResult(
            log_likelihood=float(log_likelihood),
            filtered_means=np.stack(means) if return_filtered else None,
            filtered_covariances=np.stack(covariances) if return_filtered else None,
            metadata=metadata,
            diagnostics=diagnostics,
        )
