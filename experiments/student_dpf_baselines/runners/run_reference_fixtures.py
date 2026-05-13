"""Write independent Kalman references for common student-baseline fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.student_dpf_baselines.adapters.common import write_json
from experiments.student_dpf_baselines.fixtures.common_fixtures import (
    LinearGaussianFixture,
    fixture_names,
    make_fixture,
)


OUTPUT_DIR = Path(
    "experiments/student_dpf_baselines/reports/outputs/references"
)


def run_kalman_reference(fixture: LinearGaussianFixture) -> dict:
    means, covariances, log_likelihoods = _kalman_filter(
        fixture.A,
        fixture.C,
        fixture.Q,
        fixture.R,
        fixture.m0,
        fixture.P0,
        fixture.observations,
    )
    return {
        "fixture_name": fixture.name,
        "reference": "independent_numpy_kalman",
        "state_dim": fixture.state_dim,
        "obs_dim": fixture.obs_dim,
        "horizon": fixture.horizon,
        "log_likelihood": float(np.sum(log_likelihoods)),
        "log_likelihood_increments": log_likelihoods,
        "filtered_means": means,
        "filtered_covariances": covariances,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in fixture_names():
        fixture = make_fixture(name)
        record = run_kalman_reference(fixture)
        output_path = OUTPUT_DIR / f"{name}.json"
        write_json(output_path, record)
        summary.append(
            {
                "fixture_name": name,
                "output": str(output_path),
                "log_likelihood": record["log_likelihood"],
                "horizon": record["horizon"],
            }
        )
    write_json(OUTPUT_DIR / "summary.json", {"references": summary})


def _kalman_filter(
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon = observations.shape[0]
    nx = A.shape[0]
    means = np.zeros((horizon + 1, nx), dtype=float)
    covariances = np.zeros((horizon + 1, nx, nx), dtype=float)
    log_likelihoods = np.zeros(horizon, dtype=float)
    means[0] = m0
    covariances[0] = P0
    for t, obs in enumerate(observations):
        m_pred = A @ means[t]
        p_pred = A @ covariances[t] @ A.T + Q
        innovation = obs - C @ m_pred
        S = C @ p_pred @ C.T + R
        K = p_pred @ C.T @ np.linalg.inv(S)
        means[t + 1] = m_pred + K @ innovation
        covariances[t + 1] = (np.eye(nx) - K @ C) @ p_pred
        covariances[t + 1] = 0.5 * (covariances[t + 1] + covariances[t + 1].T)
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            raise np.linalg.LinAlgError("innovation covariance is not positive definite")
        quad = float(innovation.T @ np.linalg.solve(S, innovation))
        ny = obs.shape[0]
        log_likelihoods[t] = -0.5 * (ny * np.log(2.0 * np.pi) + logdet + quad)
    return means, covariances, log_likelihoods


if __name__ == "__main__":
    main()
