"""Clean-room Gaussian range-bearing fixture for OT-DPF diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RangeBearingFixture:
    """Gaussian range-bearing nonlinear state-space fixture."""

    name: str
    A: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    m0: np.ndarray
    P0: np.ndarray
    states: np.ndarray
    observations: np.ndarray
    dt: float
    fixture_generation_seed: int
    model_checksum: str
    observation_checksum: str

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.R.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[0])

    def model_definition(self) -> dict[str, Any]:
        return {
            "model_id": self.name,
            "state_equation": "x_t = A x_{t-1} + q_t, q_t ~ Normal(0, Q)",
            "observation_equation": "y_t = [sqrt(px^2+py^2), atan2(py,px)] + r_t",
            "initial_distribution": "x_0 ~ Normal(m0, P0)",
            "A": self.A.tolist(),
            "Q": self.Q.tolist(),
            "R": self.R.tolist(),
            "m0": self.m0.tolist(),
            "P0": self.P0.tolist(),
            "horizon": self.horizon,
            "fixture_generation_seed": self.fixture_generation_seed,
            "model_checksum": self.model_checksum,
            "observation_checksum": self.observation_checksum,
            "reference_status": "UKF is approximate for this nonlinear fixture.",
        }


def make_fixture(name: str = "range_bearing_gaussian_moderate") -> RangeBearingFixture:
    """Create a fixed local range-bearing fixture."""

    if name != "range_bearing_gaussian_moderate":
        raise ValueError("only range_bearing_gaussian_moderate is authorized in this lane")
    dt = 0.1
    A = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    Q = np.diag([0.0015, 0.0015, 0.0008, 0.0008]).astype(np.float64)
    sigma_range = 0.12
    sigma_bearing = 0.04
    R = np.diag([sigma_range**2, sigma_bearing**2]).astype(np.float64)
    m0 = np.array([1.2, 0.7, 0.18, -0.06], dtype=np.float64)
    P0 = np.diag([0.04, 0.04, 0.01, 0.01]).astype(np.float64)
    seed = 701
    states, observations = _simulate_range_bearing(
        A=A,
        Q=Q,
        R=R,
        m0=m0,
        P0=P0,
        horizon=20,
        seed=seed,
    )
    model_payload = {
        "name": name,
        "A": A.tolist(),
        "Q": Q.tolist(),
        "R": R.tolist(),
        "m0": m0.tolist(),
        "P0": P0.tolist(),
        "horizon": 20,
        "seed": seed,
    }
    model_checksum = _stable_digest(model_payload)
    observation_checksum = _stable_digest(
        {"states": states.tolist(), "observations": observations.tolist()}
    )
    return RangeBearingFixture(
        name=name,
        A=A,
        Q=Q,
        R=R,
        m0=m0,
        P0=P0,
        states=states,
        observations=observations,
        dt=dt,
        fixture_generation_seed=seed,
        model_checksum=model_checksum,
        observation_checksum=observation_checksum,
    )


def transition_sample(
    rng: np.random.Generator,
    previous_particles: np.ndarray,
    fixture: RangeBearingFixture,
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
    fixture: RangeBearingFixture,
) -> np.ndarray:
    return rng.multivariate_normal(fixture.m0, fixture.P0, size=num_particles)


def observation_log_density(
    particles: np.ndarray,
    observation: np.ndarray,
    fixture: RangeBearingFixture,
) -> np.ndarray:
    predicted = range_bearing_observation(particles)
    residual = observation_residual(predicted, observation)
    return _gaussian_logpdf_zero_mean(residual, fixture.R)


def range_bearing_observation(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Evaluate h(x) = [range, bearing] for one or more states."""

    x_arr = np.asarray(x, dtype=np.float64)
    single = x_arr.ndim == 1
    if single:
        x_arr = x_arr[None, :]
    px = x_arr[:, 0]
    py = x_arr[:, 1]
    obs = np.stack(
        [np.sqrt(px**2 + py**2 + eps), np.arctan2(py, px)],
        axis=1,
    )
    return obs[0] if single else obs


def observation_residual(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Return observed-minus-predicted residuals with wrapped bearing."""

    residual = np.asarray(observed, dtype=np.float64) - np.asarray(
        predicted, dtype=np.float64
    )
    residual[..., 1] = wrap_angle(residual[..., 1])
    return residual


def wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to [-pi, pi)."""

    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def _simulate_range_bearing(
    *,
    A: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
    horizon: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    state_dim = int(A.shape[0])
    obs_dim = int(R.shape[0])
    states = np.zeros((horizon + 1, state_dim), dtype=np.float64)
    observations = np.zeros((horizon, obs_dim), dtype=np.float64)
    states[0] = rng.multivariate_normal(m0, P0)
    for t in range(horizon):
        states[t + 1] = rng.multivariate_normal(A @ states[t], Q)
        obs_mean = range_bearing_observation(states[t + 1])
        obs = rng.multivariate_normal(obs_mean, R)
        obs[1] = wrap_angle(obs[1])
        observations[t] = obs
    return states, observations


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
