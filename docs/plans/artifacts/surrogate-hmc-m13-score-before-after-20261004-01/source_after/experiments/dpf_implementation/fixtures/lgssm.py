"""Clean-room LGSSM fixture for experimental OT-DPF diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LGSSMFixture:
    """Linear Gaussian state-space model fixture."""

    name: str
    A: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    m0: np.ndarray
    P0: np.ndarray
    states: np.ndarray
    observations: np.ndarray
    fixture_generation_seed: int
    model_checksum: str
    observation_checksum: str

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.C.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[0])

    def model_definition(self) -> dict[str, Any]:
        return {
            "model_id": self.name,
            "state_equation": "x_t = A x_{t-1} + q_t, q_t ~ Normal(0, Q)",
            "observation_equation": "y_t = C x_t + r_t, r_t ~ Normal(0, R)",
            "initial_distribution": "x_0 ~ Normal(m0, P0)",
            "A": self.A.tolist(),
            "C": self.C.tolist(),
            "Q": self.Q.tolist(),
            "R": self.R.tolist(),
            "m0": self.m0.tolist(),
            "P0": self.P0.tolist(),
            "horizon": self.horizon,
            "fixture_generation_seed": self.fixture_generation_seed,
            "model_checksum": self.model_checksum,
            "observation_checksum": self.observation_checksum,
        }


def build_lgssm_fixture(
    *,
    horizon: int = 25,
    fixture_generation_seed: int = 2026052801,
) -> LGSSMFixture:
    """Build a fixed, small LGSSM fixture."""

    A = np.array([[0.92, 0.18], [-0.04, 0.86]], dtype=np.float64)
    C = np.array([[1.0, 0.35]], dtype=np.float64)
    Q = np.array([[0.08, 0.015], [0.015, 0.05]], dtype=np.float64)
    R = np.array([[0.18]], dtype=np.float64)
    m0 = np.array([0.25, -0.15], dtype=np.float64)
    P0 = np.array([[0.45, 0.04], [0.04, 0.32]], dtype=np.float64)
    rng = np.random.default_rng(fixture_generation_seed)
    states = np.zeros((horizon + 1, 2), dtype=np.float64)
    observations = np.zeros((horizon, 1), dtype=np.float64)
    states[0] = rng.multivariate_normal(m0, P0)
    for t in range(horizon):
        states[t + 1] = rng.multivariate_normal(A @ states[t], Q)
        observations[t] = rng.multivariate_normal(C @ states[t + 1], R)
    model_payload = {
        "name": "lgssm_ot_dpf_smoke",
        "A": A.tolist(),
        "C": C.tolist(),
        "Q": Q.tolist(),
        "R": R.tolist(),
        "m0": m0.tolist(),
        "P0": P0.tolist(),
        "horizon": horizon,
        "fixture_generation_seed": fixture_generation_seed,
    }
    model_checksum = _stable_digest(model_payload)
    observation_checksum = _stable_digest(
        {"states": states.tolist(), "observations": observations.tolist()}
    )
    return LGSSMFixture(
        name="lgssm_ot_dpf_smoke",
        A=A,
        C=C,
        Q=Q,
        R=R,
        m0=m0,
        P0=P0,
        states=states,
        observations=observations,
        fixture_generation_seed=fixture_generation_seed,
        model_checksum=model_checksum,
        observation_checksum=observation_checksum,
    )


def transition_sample(
    rng: np.random.Generator,
    previous_particles: np.ndarray,
    fixture: LGSSMFixture,
) -> np.ndarray:
    mean = previous_particles @ fixture.A.T
    noise = rng.multivariate_normal(
        np.zeros(fixture.state_dim, dtype=np.float64),
        fixture.Q,
        size=previous_particles.shape[0],
    )
    return mean + noise


def initial_sample(
    rng: np.random.Generator,
    num_particles: int,
    fixture: LGSSMFixture,
) -> np.ndarray:
    return rng.multivariate_normal(fixture.m0, fixture.P0, size=num_particles)


def observation_log_density(
    particles: np.ndarray,
    observation: np.ndarray,
    fixture: LGSSMFixture,
) -> np.ndarray:
    predicted = particles @ fixture.C.T
    residual = np.asarray(observation, dtype=np.float64).reshape(1, -1) - predicted
    return _gaussian_logpdf_zero_mean(residual, fixture.R)


def _gaussian_logpdf_zero_mean(residuals: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("covariance must be positive definite")
    solved = np.linalg.solve(covariance, residuals.T).T
    quad = np.sum(residuals * solved, axis=1)
    dim = covariance.shape[0]
    return -0.5 * (dim * np.log(2.0 * np.pi) + logdet + quad)


def _stable_digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
