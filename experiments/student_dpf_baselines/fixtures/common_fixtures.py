"""Small linear-Gaussian fixtures for student DPF baseline comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LinearGaussianFixture:
    """Serializable linear-Gaussian state-space fixture."""

    name: str
    A: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    m0: np.ndarray
    P0: np.ndarray
    states: np.ndarray
    observations: np.ndarray
    reference: str = "kalman"

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.C.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[0])


def make_fixture(name: str) -> LinearGaussianFixture:
    fixtures = {
        "lgssm_1d_short": make_lgssm_1d_short,
        "lgssm_cv_2d_short": make_lgssm_cv_2d_short,
        "lgssm_cv_2d_low_particles": make_lgssm_cv_2d_low_particles,
    }
    try:
        return fixtures[name]()
    except KeyError as exc:
        known = ", ".join(sorted(fixtures))
        raise ValueError(f"unknown fixture {name!r}; known fixtures: {known}") from exc


def fixture_names() -> list[str]:
    return [
        "lgssm_1d_short",
        "lgssm_cv_2d_short",
        "lgssm_cv_2d_low_particles",
    ]


def make_lgssm_1d_short() -> LinearGaussianFixture:
    A = np.array([[0.8]], dtype=float)
    C = np.array([[1.0]], dtype=float)
    Q = np.array([[0.1]], dtype=float)
    R = np.array([[0.2]], dtype=float)
    m0 = np.array([0.0], dtype=float)
    P0 = np.array([[1.0]], dtype=float)
    states, observations = _simulate_lgssm(A, C, Q, R, m0, P0, horizon=8, seed=101)
    return LinearGaussianFixture(
        name="lgssm_1d_short",
        A=A,
        C=C,
        Q=Q,
        R=R,
        m0=m0,
        P0=P0,
        states=states,
        observations=observations,
    )


def make_lgssm_cv_2d_short() -> LinearGaussianFixture:
    A, C, Q, R, m0, P0 = _constant_velocity_matrices(r=0.5)
    states, observations = _simulate_lgssm(A, C, Q, R, m0, P0, horizon=12, seed=202)
    return LinearGaussianFixture(
        name="lgssm_cv_2d_short",
        A=A,
        C=C,
        Q=Q,
        R=R,
        m0=m0,
        P0=P0,
        states=states,
        observations=observations,
    )


def make_lgssm_cv_2d_low_particles() -> LinearGaussianFixture:
    A, C, Q, R, m0, P0 = _constant_velocity_matrices(r=0.15)
    states, observations = _simulate_lgssm(A, C, Q, R, m0, P0, horizon=12, seed=303)
    return LinearGaussianFixture(
        name="lgssm_cv_2d_low_particles",
        A=A,
        C=C,
        Q=Q,
        R=R,
        m0=m0,
        P0=P0,
        states=states,
        observations=observations,
    )


def _constant_velocity_matrices(
    *,
    r: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = 1.0
    A = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    C = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=float,
    )
    q = 0.1
    Q = q * np.array(
        [
            [dt**3 / 3, 0, dt**2 / 2, 0],
            [0, dt**3 / 3, 0, dt**2 / 2],
            [dt**2 / 2, 0, dt, 0],
            [0, dt**2 / 2, 0, dt],
        ],
        dtype=float,
    )
    R = r * np.eye(2)
    m0 = np.array([0.0, 0.0, 1.0, 0.5], dtype=float)
    P0 = np.eye(4) * 0.1
    return A, C, Q, R, m0, P0


def _simulate_lgssm(
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
    *,
    horizon: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    nx = A.shape[0]
    ny = C.shape[0]
    states = np.zeros((horizon + 1, nx), dtype=float)
    observations = np.zeros((horizon, ny), dtype=float)
    states[0] = rng.multivariate_normal(m0, P0)
    for t in range(horizon):
        states[t + 1] = rng.multivariate_normal(A @ states[t], Q)
        observations[t] = rng.multivariate_normal(C @ states[t + 1], R)
    return states, observations
