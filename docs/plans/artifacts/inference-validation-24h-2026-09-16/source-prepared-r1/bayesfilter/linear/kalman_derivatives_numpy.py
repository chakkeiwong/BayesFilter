"""NumPy solve-form analytic Kalman score and Hessian."""

from __future__ import annotations

import numpy as np

from bayesfilter.linear.types import (
    LinearGaussianStateSpace,
    LinearGaussianStateSpaceDerivatives,
)
from bayesfilter.results import FilterDerivativeResult
from bayesfilter.structural import FilterRunMetadata


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _cholesky_solve(chol: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))


def _as_observation_matrix(observations: np.ndarray) -> np.ndarray:
    y = np.asarray(observations, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2:
        raise ValueError("observations must be one- or two-dimensional")
    return y


def solve_kalman_score_hessian(
    observations: np.ndarray,
    model: LinearGaussianStateSpace,
    derivatives: LinearGaussianStateSpaceDerivatives,
    *,
    jitter: float = 0.0,
    return_trace: bool = False,
) -> FilterDerivativeResult:
    """Evaluate analytic log likelihood, score, and Hessian.

    This dense NumPy backend is the BayesFilter port of the MacroFinance
    solve-form derivative recursion.  It currently handles fully observed,
    time-invariant LGSSMs; masked derivative support is a later phase.
    """

    y = _as_observation_matrix(observations)
    if y.shape[1] != model.observation_dim:
        raise ValueError("observations have incompatible observation dimension")
    if derivatives.state_dim != model.state_dim:
        raise ValueError("derivative state dimension does not match model")
    if derivatives.observation_dim != model.observation_dim:
        raise ValueError("derivative observation dimension does not match model")

    p = derivatives.parameter_dim
    n = model.state_dim
    m = model.observation_dim
    identity_state = np.eye(n)
    identity_obs = np.eye(m)
    observation_noise = model.observation_covariance + float(jitter) * identity_obs

    mean = model.initial_mean.copy()
    covariance = model.initial_covariance.copy()
    dmean = derivatives.d_initial_mean.copy()
    dcov = derivatives.d_initial_covariance.copy()
    ddmean = derivatives.d2_initial_mean.copy()
    ddcov = derivatives.d2_initial_covariance.copy()

    loglik = 0.0
    score = np.zeros(p)
    hessian = np.zeros((p, p))
    trace_rows: list[dict[str, object]] = []
    min_cholesky_pivot = np.inf
    max_solve_residual = 0.0

    for t, row in enumerate(y):
        predicted_mean = (
            model.transition_offset + model.transition_matrix @ mean
        )
        predicted_covariance = _symmetrize(
            model.transition_matrix @ covariance @ model.transition_matrix.T
            + model.transition_covariance
        )

        dpred_mean = np.zeros_like(dmean)
        dpred_cov = np.zeros_like(dcov)
        d2pred_mean = np.zeros_like(ddmean)
        d2pred_cov = np.zeros_like(ddcov)

        for i in range(p):
            dpred_mean[i] = (
                derivatives.d_transition_offset[i]
                + derivatives.d_transition_matrix[i] @ mean
                + model.transition_matrix @ dmean[i]
            )
            dpred_cov[i] = _symmetrize(
                derivatives.d_transition_matrix[i]
                @ covariance
                @ model.transition_matrix.T
                + model.transition_matrix
                @ dcov[i]
                @ model.transition_matrix.T
                + model.transition_matrix
                @ covariance
                @ derivatives.d_transition_matrix[i].T
                + derivatives.d_transition_covariance[i]
            )
            for j in range(p):
                d2pred_mean[i, j] = (
                    derivatives.d2_transition_offset[i, j]
                    + derivatives.d2_transition_matrix[i, j] @ mean
                    + derivatives.d_transition_matrix[i] @ dmean[j]
                    + derivatives.d_transition_matrix[j] @ dmean[i]
                    + model.transition_matrix @ ddmean[i, j]
                )
                d2pred_cov[i, j] = _symmetrize(
                    derivatives.d2_transition_matrix[i, j]
                    @ covariance
                    @ model.transition_matrix.T
                    + derivatives.d_transition_matrix[i]
                    @ dcov[j]
                    @ model.transition_matrix.T
                    + derivatives.d_transition_matrix[i]
                    @ covariance
                    @ derivatives.d_transition_matrix[j].T
                    + derivatives.d_transition_matrix[j]
                    @ dcov[i]
                    @ model.transition_matrix.T
                    + model.transition_matrix
                    @ ddcov[i, j]
                    @ model.transition_matrix.T
                    + model.transition_matrix
                    @ dcov[i]
                    @ derivatives.d_transition_matrix[j].T
                    + derivatives.d_transition_matrix[j]
                    @ covariance
                    @ derivatives.d_transition_matrix[i].T
                    + model.transition_matrix
                    @ dcov[j]
                    @ derivatives.d_transition_matrix[i].T
                    + model.transition_matrix
                    @ covariance
                    @ derivatives.d2_transition_matrix[i, j].T
                    + derivatives.d2_transition_covariance[i, j]
                )

        innovation = row - (
            model.observation_offset
            + model.observation_matrix @ predicted_mean
        )
        innovation_covariance = _symmetrize(
            model.observation_matrix
            @ predicted_covariance
            @ model.observation_matrix.T
            + observation_noise
        )
        chol = np.linalg.cholesky(innovation_covariance)
        innovation_solve = _cholesky_solve(chol, innovation)
        innovation_precision = _cholesky_solve(chol, identity_obs)
        min_cholesky_pivot = min(min_cholesky_pivot, float(np.min(np.diag(chol))))
        max_solve_residual = max(
            max_solve_residual,
            float(np.linalg.norm(innovation_covariance @ innovation_solve - innovation)),
        )

        dinnovation = np.zeros((p, m))
        dS = np.zeros((p, m, m))
        dw = np.zeros((p, m))
        dSinv = np.zeros((p, m, m))
        d2innovation = np.zeros((p, p, m))
        d2S = np.zeros((p, p, m, m))
        d2Sinv = np.zeros((p, p, m, m))
        grad_contrib = np.zeros(p)
        hess_contrib = np.zeros((p, p))

        for i in range(p):
            dinnovation[i] = (
                -derivatives.d_observation_offset[i]
                - derivatives.d_observation_matrix[i] @ predicted_mean
                - model.observation_matrix @ dpred_mean[i]
            )
            dS[i] = _symmetrize(
                derivatives.d_observation_matrix[i]
                @ predicted_covariance
                @ model.observation_matrix.T
                + model.observation_matrix
                @ dpred_cov[i]
                @ model.observation_matrix.T
                + model.observation_matrix
                @ predicted_covariance
                @ derivatives.d_observation_matrix[i].T
                + derivatives.d_observation_covariance[i]
            )
            dw[i] = _cholesky_solve(chol, dinnovation[i] - dS[i] @ innovation_solve)
            dSinv[i] = -_cholesky_solve(chol, dS[i] @ innovation_precision)
            grad_contrib[i] = -0.5 * (
                np.trace(innovation_precision @ dS[i])
                + 2.0 * dinnovation[i] @ innovation_solve
                - innovation_solve @ dS[i] @ innovation_solve
            )
            score[i] += grad_contrib[i]

        for i in range(p):
            for j in range(p):
                d2innovation[i, j] = (
                    -derivatives.d2_observation_offset[i, j]
                    - derivatives.d2_observation_matrix[i, j] @ predicted_mean
                    - derivatives.d_observation_matrix[i] @ dpred_mean[j]
                    - derivatives.d_observation_matrix[j] @ dpred_mean[i]
                    - model.observation_matrix @ d2pred_mean[i, j]
                )
                d2S[i, j] = _symmetrize(
                    derivatives.d2_observation_matrix[i, j]
                    @ predicted_covariance
                    @ model.observation_matrix.T
                    + derivatives.d_observation_matrix[i]
                    @ dpred_cov[j]
                    @ model.observation_matrix.T
                    + derivatives.d_observation_matrix[i]
                    @ predicted_covariance
                    @ derivatives.d_observation_matrix[j].T
                    + derivatives.d_observation_matrix[j]
                    @ dpred_cov[i]
                    @ model.observation_matrix.T
                    + model.observation_matrix
                    @ d2pred_cov[i, j]
                    @ model.observation_matrix.T
                    + model.observation_matrix
                    @ dpred_cov[i]
                    @ derivatives.d_observation_matrix[j].T
                    + derivatives.d_observation_matrix[j]
                    @ predicted_covariance
                    @ derivatives.d_observation_matrix[i].T
                    + model.observation_matrix
                    @ dpred_cov[j]
                    @ derivatives.d_observation_matrix[i].T
                    + model.observation_matrix
                    @ predicted_covariance
                    @ derivatives.d2_observation_matrix[i, j].T
                    + derivatives.d2_observation_covariance[i, j]
                )
                d2Sinv[i, j] = _cholesky_solve(
                    chol,
                    dS[j] @ innovation_precision @ dS[i] @ innovation_precision
                    + dS[i] @ innovation_precision @ dS[j] @ innovation_precision
                    - d2S[i, j] @ innovation_precision,
                )
                trace_term = np.trace(dSinv[j] @ dS[i] + innovation_precision @ d2S[i, j])
                linear_term = (
                    2.0 * d2innovation[i, j] @ innovation_solve
                    + 2.0 * dinnovation[i] @ dw[j]
                )
                curvature_term = (
                    dinnovation[j] @ innovation_precision @ dS[i] @ innovation_solve
                    + innovation @ dSinv[j] @ dS[i] @ innovation_solve
                    + innovation @ innovation_precision @ d2S[i, j] @ innovation_solve
                    + innovation @ innovation_precision @ dS[i] @ dw[j]
                )
                hess_contrib[i, j] = -0.5 * (
                    trace_term + linear_term - curvature_term
                )
                hessian[i, j] += hess_contrib[i, j]

        kalman_gain = (
            predicted_covariance
            @ model.observation_matrix.T
            @ innovation_precision
        )
        dK = np.zeros((p, n, m))
        d2K = np.zeros((p, p, n, m))
        for i in range(p):
            dK[i] = (
                dpred_cov[i] @ model.observation_matrix.T @ innovation_precision
                + predicted_covariance @ derivatives.d_observation_matrix[i].T @ innovation_precision
                + predicted_covariance @ model.observation_matrix.T @ dSinv[i]
            )
            for j in range(p):
                d2K[i, j] = (
                    d2pred_cov[i, j] @ model.observation_matrix.T @ innovation_precision
                    + dpred_cov[i] @ derivatives.d_observation_matrix[j].T @ innovation_precision
                    + dpred_cov[i] @ model.observation_matrix.T @ dSinv[j]
                    + dpred_cov[j] @ derivatives.d_observation_matrix[i].T @ innovation_precision
                    + predicted_covariance @ derivatives.d2_observation_matrix[i, j].T @ innovation_precision
                    + predicted_covariance @ derivatives.d_observation_matrix[i].T @ dSinv[j]
                    + dpred_cov[j] @ model.observation_matrix.T @ dSinv[i]
                    + predicted_covariance @ derivatives.d_observation_matrix[j].T @ dSinv[i]
                    + predicted_covariance @ model.observation_matrix.T @ d2Sinv[i, j]
                )

        mean = predicted_mean + kalman_gain @ innovation
        joseph_left = identity_state - kalman_gain @ model.observation_matrix
        covariance = _symmetrize(
            joseph_left @ predicted_covariance @ joseph_left.T
            + kalman_gain @ observation_noise @ kalman_gain.T
        )

        d_joseph_left = np.zeros((p, n, n))
        d2_joseph_left = np.zeros((p, p, n, n))
        for i in range(p):
            d_joseph_left[i] = (
                -dK[i] @ model.observation_matrix
                - kalman_gain @ derivatives.d_observation_matrix[i]
            )
            for j in range(p):
                d2_joseph_left[i, j] = (
                    -d2K[i, j] @ model.observation_matrix
                    - dK[i] @ derivatives.d_observation_matrix[j]
                    - dK[j] @ derivatives.d_observation_matrix[i]
                    - kalman_gain @ derivatives.d2_observation_matrix[i, j]
                )

        for i in range(p):
            dmean[i] = (
                dpred_mean[i] + dK[i] @ innovation + kalman_gain @ dinnovation[i]
            )
            dcov[i] = _symmetrize(
                d_joseph_left[i] @ predicted_covariance @ joseph_left.T
                + joseph_left @ dpred_cov[i] @ joseph_left.T
                + joseph_left @ predicted_covariance @ d_joseph_left[i].T
                + dK[i] @ observation_noise @ kalman_gain.T
                + kalman_gain @ derivatives.d_observation_covariance[i] @ kalman_gain.T
                + kalman_gain @ observation_noise @ dK[i].T
            )
            for j in range(p):
                ddmean[i, j] = (
                    d2pred_mean[i, j]
                    + d2K[i, j] @ innovation
                    + dK[i] @ dinnovation[j]
                    + dK[j] @ dinnovation[i]
                    + kalman_gain @ d2innovation[i, j]
                )
                ddcov[i, j] = _symmetrize(
                    d2_joseph_left[i, j] @ predicted_covariance @ joseph_left.T
                    + d_joseph_left[i] @ dpred_cov[j] @ joseph_left.T
                    + d_joseph_left[i] @ predicted_covariance @ d_joseph_left[j].T
                    + d_joseph_left[j] @ dpred_cov[i] @ joseph_left.T
                    + joseph_left @ d2pred_cov[i, j] @ joseph_left.T
                    + joseph_left @ dpred_cov[i] @ d_joseph_left[j].T
                    + d_joseph_left[j] @ predicted_covariance @ d_joseph_left[i].T
                    + joseph_left @ dpred_cov[j] @ d_joseph_left[i].T
                    + joseph_left @ predicted_covariance @ d2_joseph_left[i, j].T
                    + d2K[i, j] @ observation_noise @ kalman_gain.T
                    + dK[i] @ derivatives.d_observation_covariance[j] @ kalman_gain.T
                    + dK[i] @ observation_noise @ dK[j].T
                    + dK[j] @ derivatives.d_observation_covariance[i] @ kalman_gain.T
                    + kalman_gain @ derivatives.d2_observation_covariance[i, j] @ kalman_gain.T
                    + kalman_gain @ derivatives.d_observation_covariance[i] @ dK[j].T
                    + dK[j] @ observation_noise @ dK[i].T
                    + kalman_gain @ derivatives.d_observation_covariance[j] @ dK[i].T
                    + kalman_gain @ observation_noise @ d2K[i, j].T
                )

        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        quad = float(innovation @ innovation_solve)
        contrib = -0.5 * (m * np.log(2.0 * np.pi) + log_det + quad)
        loglik += contrib

        if return_trace:
            trace_rows.append(
                {
                    "t": t,
                    "innovation": innovation.copy(),
                    "innovation_covariance": innovation_covariance.copy(),
                    "innovation_solve": innovation_solve.copy(),
                    "grad_contrib": grad_contrib.copy(),
                    "hess_contrib": hess_contrib.copy(),
                    "loglik": loglik,
                    "contrib": contrib,
                    "cholesky_min_pivot": float(np.min(np.diag(chol))),
                    "solve_residual_norm": float(
                        np.linalg.norm(innovation_covariance @ innovation_solve - innovation)
                    ),
                }
            )

    hessian = _symmetrize(hessian)
    metadata = FilterRunMetadata(
        filter_name="solve_kalman_score_hessian",
        partition=model.partition,
        integration_space="full_state",
        deterministic_completion="none",
        approximation_label=None,
        differentiability_status="analytic_score_hessian",
        compiled_status="eager_numpy",
    )
    return FilterDerivativeResult(
        log_likelihood=loglik,
        score=score,
        hessian=hessian,
        metadata=metadata,
        diagnostics={
            "backend": "numpy_solve_analytic",
            "jitter": float(jitter),
            "mask_convention": "dense_only",
            "min_cholesky_pivot": float(min_cholesky_pivot),
            "max_solve_residual": float(max_solve_residual),
        },
        trace=tuple(trace_rows) if return_trace else None,
    )
