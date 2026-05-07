"""Exact covariance-form Kalman reference backend."""

from __future__ import annotations

from typing import Literal

import numpy as np

from bayesfilter.linear.types import LinearGaussianStateSpace
from bayesfilter.results import FilterValueResult
from bayesfilter.structural import FilterRunMetadata


KalmanResult = FilterValueResult
LinearValueBackend = Literal["covariance", "solve", "svd"]


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _selected_mask(observations: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.isfinite(observations)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != observations.shape:
        raise ValueError("mask must have the same shape as observations")
    return selected


def _as_observation_matrix(observations: np.ndarray) -> np.ndarray:
    y = np.asarray(observations, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2:
        raise ValueError("observations must be one- or two-dimensional")
    return y


def _innovation_objects(
    row: np.ndarray,
    row_mask: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_covariance: np.ndarray,
    model: LinearGaussianStateSpace,
    jitter: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H = model.observation_matrix[row_mask]
    offset = model.observation_offset[row_mask]
    R = model.observation_covariance[np.ix_(row_mask, row_mask)]
    if jitter:
        R = R + float(jitter) * np.eye(R.shape[0])
    innovation = row[row_mask] - (offset + H @ predicted_mean)
    innovation_covariance = _symmetrize(H @ predicted_covariance @ H.T + R)
    return H, R, innovation, innovation_covariance


def _metadata(filter_name: str, model: LinearGaussianStateSpace) -> FilterRunMetadata:
    return FilterRunMetadata(
        filter_name=filter_name,
        partition=model.partition,
        integration_space="full_state",
        deterministic_completion="none",
        approximation_label=None,
        differentiability_status="value_only",
        compiled_status="eager",
    )


def _value_result(
    *,
    log_likelihood: float,
    means: list[np.ndarray],
    covariances: list[np.ndarray],
    return_filtered: bool,
    metadata: FilterRunMetadata,
    diagnostics: dict[str, object],
) -> KalmanResult:
    return KalmanResult(
        log_likelihood=float(log_likelihood),
        filtered_means=np.stack(means) if return_filtered else None,
        filtered_covariances=np.stack(covariances) if return_filtered else None,
        metadata=metadata,
        diagnostics=diagnostics,
    )


def kalman_log_likelihood(
    observations: np.ndarray,
    model: LinearGaussianStateSpace,
    *,
    mask: np.ndarray | None = None,
    jitter: float = 0.0,
    return_filtered: bool = False,
) -> KalmanResult:
    """Evaluate the exact linear Gaussian prediction-error likelihood.

    Singular process covariance is allowed.  The selected innovation covariance
    must be positive definite; otherwise NumPy raises a linear-algebra error so
    callers cannot silently promote an invalid target.
    """

    y = _as_observation_matrix(observations)
    if y.shape[1] != model.observation_dim:
        raise ValueError("observations have incompatible observation dimension")
    obs_mask = _selected_mask(y, mask)

    mean = model.initial_mean.copy()
    covariance = model.initial_covariance.copy()
    identity = np.eye(model.state_dim)
    log_likelihood = 0.0
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []

    for row, row_mask in zip(y, obs_mask, strict=True):
        predicted_mean = model.transition_offset + model.transition_matrix @ mean
        predicted_covariance = _symmetrize(
            model.transition_matrix @ covariance @ model.transition_matrix.T
            + model.transition_covariance
        )

        if not np.any(row_mask):
            mean = predicted_mean
            covariance = predicted_covariance
        else:
            H, R, innovation, innovation_covariance = _innovation_objects(
                row,
                row_mask,
                predicted_mean,
                predicted_covariance,
                model,
                jitter,
            )
            chol = np.linalg.cholesky(innovation_covariance)
            whitened = np.linalg.solve(chol, innovation)
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            quad = float(whitened @ whitened)
            log_likelihood += -0.5 * (
                row_mask.sum() * np.log(2.0 * np.pi) + log_det + quad
            )

            gain = np.linalg.solve(
                innovation_covariance.T, (predicted_covariance @ H.T).T
            ).T
            mean = predicted_mean + gain @ innovation
            left = identity - gain @ H
            covariance = _symmetrize(left @ predicted_covariance @ left.T + gain @ R @ gain.T)

        if return_filtered:
            means.append(mean.copy())
            covariances.append(covariance.copy())

    return _value_result(
        log_likelihood=log_likelihood,
        means=means,
        covariances=covariances,
        return_filtered=return_filtered,
        metadata=_metadata("covariance_kalman", model),
        diagnostics={
            "backend": "numpy_covariance",
            "jitter": float(jitter),
            "mask_convention": "row_selection",
        },
    )


def solve_kalman_log_likelihood(
    observations: np.ndarray,
    model: LinearGaussianStateSpace,
    *,
    mask: np.ndarray | None = None,
    jitter: float = 0.0,
    return_filtered: bool = False,
) -> KalmanResult:
    """Evaluate the Kalman likelihood using Cholesky solves as the primitive."""

    y = _as_observation_matrix(observations)
    if y.shape[1] != model.observation_dim:
        raise ValueError("observations have incompatible observation dimension")
    obs_mask = _selected_mask(y, mask)

    mean = model.initial_mean.copy()
    covariance = model.initial_covariance.copy()
    identity = np.eye(model.state_dim)
    log_likelihood = 0.0
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    min_cholesky_pivot = np.inf
    max_solve_residual = 0.0

    for row, row_mask in zip(y, obs_mask, strict=True):
        predicted_mean = model.transition_offset + model.transition_matrix @ mean
        predicted_covariance = _symmetrize(
            model.transition_matrix @ covariance @ model.transition_matrix.T
            + model.transition_covariance
        )

        if not np.any(row_mask):
            mean = predicted_mean
            covariance = predicted_covariance
        else:
            H, R, innovation, innovation_covariance = _innovation_objects(
                row,
                row_mask,
                predicted_mean,
                predicted_covariance,
                model,
                jitter,
            )
            chol = np.linalg.cholesky(innovation_covariance)
            solve_innovation = np.linalg.solve(
                chol.T,
                np.linalg.solve(chol, innovation),
            )
            precision = np.linalg.solve(
                chol.T,
                np.linalg.solve(chol, np.eye(chol.shape[0])),
            )
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            quad = float(innovation @ solve_innovation)
            log_likelihood += -0.5 * (
                row_mask.sum() * np.log(2.0 * np.pi) + log_det + quad
            )
            max_solve_residual = max(
                max_solve_residual,
                float(np.linalg.norm(innovation_covariance @ solve_innovation - innovation)),
            )
            min_cholesky_pivot = min(min_cholesky_pivot, float(np.min(np.diag(chol))))

            gain = predicted_covariance @ H.T @ precision
            mean = predicted_mean + gain @ innovation
            left = identity - gain @ H
            covariance = _symmetrize(left @ predicted_covariance @ left.T + gain @ R @ gain.T)

        if return_filtered:
            means.append(mean.copy())
            covariances.append(covariance.copy())

    return _value_result(
        log_likelihood=log_likelihood,
        means=means,
        covariances=covariances,
        return_filtered=return_filtered,
        metadata=_metadata("solve_kalman", model),
        diagnostics={
            "backend": "numpy_solve",
            "jitter": float(jitter),
            "mask_convention": "row_selection",
            "min_cholesky_pivot": float(min_cholesky_pivot),
            "max_solve_residual": float(max_solve_residual),
        },
    )


def _svd_psd(covariance: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(_symmetrize(covariance))
    floored = np.maximum(values, float(floor))
    return vectors, floored


def _svd_solve(vectors: np.ndarray, values: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    projected = vectors.T @ rhs
    if projected.ndim == 1:
        return vectors @ (projected / values)
    return vectors @ (projected / values[:, None])


def svd_kalman_log_likelihood(
    observations: np.ndarray,
    model: LinearGaussianStateSpace,
    *,
    mask: np.ndarray | None = None,
    jitter: float = 0.0,
    singular_floor: float = 1e-12,
    return_filtered: bool = False,
) -> KalmanResult:
    """Evaluate the linear Gaussian likelihood with SVD/eigen solves.

    This backend reconstructs and updates covariance matrices but uses floored
    eigenvalues for innovation solves and log determinants.  Diagnostics report
    the implemented floor branch.
    """

    y = _as_observation_matrix(observations)
    if y.shape[1] != model.observation_dim:
        raise ValueError("observations have incompatible observation dimension")
    obs_mask = _selected_mask(y, mask)

    mean = model.initial_mean.copy()
    covariance = _symmetrize(model.initial_covariance.copy())
    identity = np.eye(model.state_dim)
    log_likelihood = 0.0
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    min_innovation_eigenvalue = np.inf
    max_floor_count = 0
    max_solve_residual = 0.0

    for row, row_mask in zip(y, obs_mask, strict=True):
        predicted_mean = model.transition_offset + model.transition_matrix @ mean
        predicted_covariance = _symmetrize(
            model.transition_matrix @ covariance @ model.transition_matrix.T
            + model.transition_covariance
        )

        if not np.any(row_mask):
            mean = predicted_mean
            covariance = predicted_covariance
        else:
            H, R, innovation, innovation_covariance = _innovation_objects(
                row,
                row_mask,
                predicted_mean,
                predicted_covariance,
                model,
                jitter,
            )
            vectors, floored_values = _svd_psd(innovation_covariance, singular_floor)
            raw_values = np.linalg.eigvalsh(_symmetrize(innovation_covariance))
            solve_innovation = _svd_solve(vectors, floored_values, innovation)
            precision = _svd_solve(vectors, floored_values, np.eye(floored_values.size))
            log_det = float(np.sum(np.log(floored_values)))
            quad = float(innovation @ solve_innovation)
            log_likelihood += -0.5 * (
                row_mask.sum() * np.log(2.0 * np.pi) + log_det + quad
            )

            min_innovation_eigenvalue = min(
                min_innovation_eigenvalue,
                float(np.min(raw_values)),
            )
            max_floor_count = max(
                max_floor_count,
                int(np.sum(raw_values <= float(singular_floor))),
            )
            max_solve_residual = max(
                max_solve_residual,
                float(np.linalg.norm(innovation_covariance @ solve_innovation - innovation)),
            )

            gain = predicted_covariance @ H.T @ precision
            mean = predicted_mean + gain @ innovation
            left = identity - gain @ H
            covariance = _symmetrize(left @ predicted_covariance @ left.T + gain @ R @ gain.T)

        if return_filtered:
            means.append(mean.copy())
            covariances.append(covariance.copy())

    return _value_result(
        log_likelihood=log_likelihood,
        means=means,
        covariances=covariances,
        return_filtered=return_filtered,
        metadata=_metadata("svd_kalman", model),
        diagnostics={
            "backend": "numpy_svd",
            "jitter": float(jitter),
            "singular_floor": float(singular_floor),
            "mask_convention": "row_selection",
            "min_innovation_eigenvalue": float(min_innovation_eigenvalue),
            "max_floor_count": int(max_floor_count),
            "max_solve_residual_pre_floor": float(max_solve_residual),
        },
    )


def linear_gaussian_log_likelihood(
    observations: np.ndarray,
    model: LinearGaussianStateSpace,
    *,
    backend: LinearValueBackend = "covariance",
    mask: np.ndarray | None = None,
    jitter: float = 0.0,
    singular_floor: float = 1e-12,
    return_filtered: bool = False,
) -> KalmanResult:
    """Dispatch to a NumPy linear Gaussian value backend."""

    if backend == "covariance":
        return kalman_log_likelihood(
            observations,
            model,
            mask=mask,
            jitter=jitter,
            return_filtered=return_filtered,
        )
    if backend == "solve":
        return solve_kalman_log_likelihood(
            observations,
            model,
            mask=mask,
            jitter=jitter,
            return_filtered=return_filtered,
        )
    if backend == "svd":
        return svd_kalman_log_likelihood(
            observations,
            model,
            mask=mask,
            jitter=jitter,
            singular_floor=singular_floor,
            return_filtered=return_filtered,
        )
    raise ValueError(f"unknown linear Gaussian backend: {backend}")
