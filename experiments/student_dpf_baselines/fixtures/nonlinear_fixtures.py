"""Nonlinear fixtures for student DPF baseline comparisons."""

from __future__ import annotations

from dataclasses import dataclass

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
    target: str = "gaussian_range_bearing"

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.R.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[0])


def nonlinear_fixture_names() -> list[str]:
    return [
        "range_bearing_gaussian_moderate",
        "range_bearing_gaussian_low_noise",
    ]


def make_nonlinear_fixture(name: str) -> RangeBearingFixture:
    fixtures = {
        "range_bearing_gaussian_moderate": make_range_bearing_gaussian_moderate,
        "range_bearing_gaussian_low_noise": make_range_bearing_gaussian_low_noise,
    }
    try:
        return fixtures[name]()
    except KeyError as exc:
        known = ", ".join(sorted(fixtures))
        raise ValueError(f"unknown nonlinear fixture {name!r}; known fixtures: {known}") from exc


def make_range_bearing_gaussian_moderate() -> RangeBearingFixture:
    return _make_range_bearing_fixture(
        name="range_bearing_gaussian_moderate",
        sigma_range=0.12,
        sigma_bearing=0.04,
        seed=701,
    )


def make_range_bearing_gaussian_low_noise() -> RangeBearingFixture:
    return _make_range_bearing_fixture(
        name="range_bearing_gaussian_low_noise",
        sigma_range=0.035,
        sigma_bearing=0.012,
        seed=702,
    )


def range_bearing_observation(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    single = x_arr.ndim == 1
    if single:
        x_arr = x_arr[None, :]
    px = x_arr[:, 0]
    py = x_arr[:, 1]
    obs = np.stack(
        [
            np.sqrt(px**2 + py**2 + eps),
            np.arctan2(py, px),
        ],
        axis=1,
    )
    return obs[0] if single else obs


def range_bearing_jacobian(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    px, py = x_arr[0], x_arr[1]
    r2 = px**2 + py**2 + eps
    r = np.sqrt(r2)
    return np.array(
        [
            [px / r, py / r, 0.0, 0.0],
            [-py / r2, px / r2, 0.0, 0.0],
        ],
        dtype=float,
    )


def wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def observation_residual(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    residual = np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)
    residual[..., 1] = wrap_angle(residual[..., 1])
    return residual


def _make_range_bearing_fixture(
    *,
    name: str,
    sigma_range: float,
    sigma_bearing: float,
    seed: int,
) -> RangeBearingFixture:
    dt = 0.1
    A = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    Q = np.diag([0.0015, 0.0015, 0.0008, 0.0008])
    R = np.diag([sigma_range**2, sigma_bearing**2])
    m0 = np.array([1.2, 0.7, 0.18, -0.06], dtype=float)
    P0 = np.diag([0.04, 0.04, 0.01, 0.01])
    states, observations = _simulate_range_bearing(
        A=A,
        Q=Q,
        R=R,
        m0=m0,
        P0=P0,
        horizon=20,
        seed=seed,
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
    )


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
    state_dim = A.shape[0]
    obs_dim = R.shape[0]
    states = np.zeros((horizon + 1, state_dim), dtype=float)
    observations = np.zeros((horizon, obs_dim), dtype=float)
    states[0] = rng.multivariate_normal(m0, P0)
    for t in range(horizon):
        states[t + 1] = rng.multivariate_normal(A @ states[t], Q)
        observation_mean = range_bearing_observation(states[t + 1])
        observation = rng.multivariate_normal(observation_mean, R)
        observation[1] = wrap_angle(observation[1])
        observations[t] = observation
    return states, observations
