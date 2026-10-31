"""Kalman reference for the experimental LGSSM fixture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.dpf_implementation.fixtures.lgssm import LGSSMFixture


@dataclass(frozen=True)
class KalmanResult:
    filtered_means: np.ndarray
    filtered_covariances: np.ndarray
    predictive_means: np.ndarray
    predictive_covariances: np.ndarray
    log_likelihood: float
    finite: bool
    reference_id: str = "kalman_lgssm_exact_reference"


def run_kalman_filter(fixture: LGSSMFixture) -> KalmanResult:
    """Run the exact Kalman filter for the LGSSM fixture."""

    m_prev = fixture.m0.copy()
    P_prev = fixture.P0.copy()
    filtered_means = []
    filtered_covariances = []
    predictive_means = []
    predictive_covariances = []
    log_likelihood = 0.0
    eye = np.eye(fixture.state_dim, dtype=np.float64)

    for observation in fixture.observations:
        m_pred = fixture.A @ m_prev
        P_pred = fixture.A @ P_prev @ fixture.A.T + fixture.Q
        y_pred = fixture.C @ m_pred
        S = fixture.C @ P_pred @ fixture.C.T + fixture.R
        residual = observation.reshape(-1) - y_pred
        log_likelihood += _gaussian_logpdf_scalar(residual, S)
        K = np.linalg.solve(S, fixture.C @ P_pred).T
        m_filt = m_pred + K @ residual
        P_filt = (eye - K @ fixture.C) @ P_pred
        P_filt = 0.5 * (P_filt + P_filt.T)
        predictive_means.append(m_pred)
        predictive_covariances.append(P_pred)
        filtered_means.append(m_filt)
        filtered_covariances.append(P_filt)
        m_prev = m_filt
        P_prev = P_filt

    filtered_means_array = np.asarray(filtered_means, dtype=np.float64)
    filtered_covariances_array = np.asarray(filtered_covariances, dtype=np.float64)
    predictive_means_array = np.asarray(predictive_means, dtype=np.float64)
    predictive_covariances_array = np.asarray(predictive_covariances, dtype=np.float64)
    finite = bool(
        np.isfinite(log_likelihood)
        and np.all(np.isfinite(filtered_means_array))
        and np.all(np.isfinite(filtered_covariances_array))
        and np.all(np.isfinite(predictive_means_array))
        and np.all(np.isfinite(predictive_covariances_array))
    )
    return KalmanResult(
        filtered_means=filtered_means_array,
        filtered_covariances=filtered_covariances_array,
        predictive_means=predictive_means_array,
        predictive_covariances=predictive_covariances_array,
        log_likelihood=float(log_likelihood),
        finite=finite,
    )


def _gaussian_logpdf_scalar(residual: np.ndarray, covariance: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("innovation covariance must be positive definite")
    solved = np.linalg.solve(covariance, residual)
    quad = float(residual.T @ solved)
    dim = covariance.shape[0]
    return float(-0.5 * (dim * np.log(2.0 * np.pi) + logdet + quad))
