"""Unscented Kalman reference for the experimental range-bearing fixture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.dpf_implementation.fixtures.range_bearing import (
    RangeBearingFixture,
    observation_residual,
    range_bearing_observation,
)


@dataclass(frozen=True)
class UKFResult:
    filtered_means: np.ndarray
    filtered_covariances: np.ndarray
    predicted_observations: np.ndarray
    finite: bool
    approximate_reference: bool = True
    reference_id: str = "ukf_range_bearing_approximate_reference"


def run_range_bearing_ukf(
    fixture: RangeBearingFixture,
    *,
    alpha: float = 0.5,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> UKFResult:
    """Run a bounded UKF reference on the range-bearing fixture."""

    n = fixture.state_dim
    lam = alpha * alpha * (n + kappa) - n
    wm, wc = _unscented_weights(n, lam, alpha, beta)
    mean = fixture.m0.copy()
    covariance = fixture.P0.copy()
    filtered_means = []
    filtered_covariances = []
    predicted_observations = []

    for observation in fixture.observations:
        mean_pred = fixture.A @ mean
        covariance_pred = fixture.A @ covariance @ fixture.A.T + fixture.Q
        covariance_pred = _stabilize_covariance(covariance_pred)
        sigma = _sigma_points(mean_pred, covariance_pred, lam)
        obs_sigma = range_bearing_observation(sigma)
        obs_mean = _weighted_angle_observation_mean(obs_sigma, wm)
        obs_residuals = observation_residual(obs_mean, obs_sigma)
        state_residuals = sigma - mean_pred
        S = fixture.R.copy()
        cross = np.zeros((n, fixture.obs_dim), dtype=np.float64)
        for i in range(sigma.shape[0]):
            S += wc[i] * np.outer(obs_residuals[i], obs_residuals[i])
            cross += wc[i] * np.outer(state_residuals[i], obs_residuals[i])
        S = _stabilize_covariance(S)
        innovation = observation_residual(obs_mean, observation)
        K = np.linalg.solve(S, cross.T).T
        mean = mean_pred + K @ innovation
        covariance = covariance_pred - K @ S @ K.T
        covariance = _stabilize_covariance(covariance)
        filtered_means.append(mean)
        filtered_covariances.append(covariance)
        predicted_observations.append(obs_mean)

    means_array = np.asarray(filtered_means, dtype=np.float64)
    covariances_array = np.asarray(filtered_covariances, dtype=np.float64)
    observations_array = np.asarray(predicted_observations, dtype=np.float64)
    finite = bool(
        np.all(np.isfinite(means_array))
        and np.all(np.isfinite(covariances_array))
        and np.all(np.isfinite(observations_array))
    )
    return UKFResult(
        filtered_means=means_array,
        filtered_covariances=covariances_array,
        predicted_observations=observations_array,
        finite=finite,
    )


def _unscented_weights(
    n: int,
    lam: float,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    denom = n + lam
    wm = np.full(2 * n + 1, 1.0 / (2.0 * denom), dtype=np.float64)
    wc = wm.copy()
    wm[0] = lam / denom
    wc[0] = lam / denom + (1.0 - alpha * alpha + beta)
    return wm, wc


def _sigma_points(mean: np.ndarray, covariance: np.ndarray, lam: float) -> np.ndarray:
    n = mean.shape[0]
    scale = n + lam
    root = np.linalg.cholesky(_stabilize_covariance(scale * covariance))
    points = [mean]
    for i in range(n):
        points.append(mean + root[:, i])
        points.append(mean - root[:, i])
    return np.asarray(points, dtype=np.float64)


def _weighted_angle_observation_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    range_mean = float(np.sum(weights * values[:, 0]))
    sin_mean = float(np.sum(weights * np.sin(values[:, 1])))
    cos_mean = float(np.sum(weights * np.cos(values[:, 1])))
    return np.array([range_mean, np.arctan2(sin_mean, cos_mean)], dtype=np.float64)


def _stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
    sym = 0.5 * (covariance + covariance.T)
    min_eig = float(np.min(np.linalg.eigvalsh(sym)))
    if min_eig < 1e-10:
        sym = sym + np.eye(sym.shape[0], dtype=np.float64) * (1e-10 - min_eig)
    return sym
