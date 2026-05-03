"""Exact covariance-form Kalman reference backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bayesfilter.structural import FilterRunMetadata, StatePartition


@dataclass(frozen=True)
class LinearGaussianStateSpace:
    """Linear Gaussian state-space object with explicit initialization."""

    initial_mean: np.ndarray
    initial_covariance: np.ndarray
    transition_offset: np.ndarray
    transition_matrix: np.ndarray
    transition_covariance: np.ndarray
    observation_offset: np.ndarray
    observation_matrix: np.ndarray
    observation_covariance: np.ndarray
    partition: StatePartition | None = None

    def __post_init__(self) -> None:
        arrays = {
            name: np.asarray(getattr(self, name), dtype=float)
            for name in (
                "initial_mean",
                "initial_covariance",
                "transition_offset",
                "transition_matrix",
                "transition_covariance",
                "observation_offset",
                "observation_matrix",
                "observation_covariance",
            )
        }
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        self.validate()

    @property
    def state_dim(self) -> int:
        return int(self.transition_matrix.shape[0])

    @property
    def observation_dim(self) -> int:
        return int(self.observation_matrix.shape[0])

    def validate(self) -> None:
        n = self.transition_matrix.shape[0]
        m = self.observation_matrix.shape[0]
        expected = {
            "initial_mean": (n,),
            "initial_covariance": (n, n),
            "transition_offset": (n,),
            "transition_matrix": (n, n),
            "transition_covariance": (n, n),
            "observation_offset": (m,),
            "observation_matrix": (m, n),
            "observation_covariance": (m, m),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")
        if self.partition is not None and self.partition.state_dim != n:
            raise ValueError("partition state_dim does not match transition_matrix")


@dataclass(frozen=True)
class KalmanResult:
    log_likelihood: float
    filtered_means: np.ndarray | None
    filtered_covariances: np.ndarray | None
    metadata: FilterRunMetadata


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _selected_mask(observations: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.isfinite(observations)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != observations.shape:
        raise ValueError("mask must have the same shape as observations")
    return selected


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

    y = np.asarray(observations, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
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
            H = model.observation_matrix[row_mask]
            offset = model.observation_offset[row_mask]
            R = model.observation_covariance[np.ix_(row_mask, row_mask)]
            if jitter:
                R = R + float(jitter) * np.eye(R.shape[0])
            innovation = row[row_mask] - (offset + H @ predicted_mean)
            innovation_covariance = _symmetrize(H @ predicted_covariance @ H.T + R)
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

    metadata = FilterRunMetadata(
        filter_name="covariance_kalman",
        partition=model.partition,
        integration_space="full_state",
        deterministic_completion="none",
        approximation_label=None,
        differentiability_status="value_only",
        compiled_status="eager",
    )
    return KalmanResult(
        log_likelihood=float(log_likelihood),
        filtered_means=np.stack(means) if return_filtered else None,
        filtered_covariances=np.stack(covariances) if return_filtered else None,
        metadata=metadata,
    )
