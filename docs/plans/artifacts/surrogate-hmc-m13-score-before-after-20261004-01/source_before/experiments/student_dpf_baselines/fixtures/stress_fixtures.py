"""Stress fixtures for follow-on student DPF baseline comparisons."""

from __future__ import annotations

import numpy as np

from .common_fixtures import LinearGaussianFixture, _constant_velocity_matrices, _simulate_lgssm


def stress_fixture_names() -> list[str]:
    return [
        "lgssm_1d_long",
        "lgssm_cv_2d_long",
        "lgssm_cv_2d_low_noise",
    ]


def make_stress_fixture(name: str) -> LinearGaussianFixture:
    fixtures = {
        "lgssm_1d_long": make_lgssm_1d_long,
        "lgssm_cv_2d_long": make_lgssm_cv_2d_long,
        "lgssm_cv_2d_low_noise": make_lgssm_cv_2d_low_noise,
    }
    try:
        return fixtures[name]()
    except KeyError as exc:
        known = ", ".join(sorted(fixtures))
        raise ValueError(f"unknown stress fixture {name!r}; known fixtures: {known}") from exc


def make_lgssm_1d_long() -> LinearGaussianFixture:
    A = np.array([[0.9]], dtype=float)
    C = np.array([[1.0]], dtype=float)
    Q = np.array([[0.05]], dtype=float)
    R = np.array([[0.15]], dtype=float)
    m0 = np.array([0.0], dtype=float)
    P0 = np.array([[1.0]], dtype=float)
    states, observations = _simulate_lgssm(A, C, Q, R, m0, P0, horizon=60, seed=401)
    return LinearGaussianFixture(
        name="lgssm_1d_long",
        A=A,
        C=C,
        Q=Q,
        R=R,
        m0=m0,
        P0=P0,
        states=states,
        observations=observations,
    )


def make_lgssm_cv_2d_long() -> LinearGaussianFixture:
    A, C, Q, R, m0, P0 = _constant_velocity_matrices(r=0.5)
    states, observations = _simulate_lgssm(A, C, Q, R, m0, P0, horizon=50, seed=402)
    return LinearGaussianFixture(
        name="lgssm_cv_2d_long",
        A=A,
        C=C,
        Q=Q,
        R=R,
        m0=m0,
        P0=P0,
        states=states,
        observations=observations,
    )


def make_lgssm_cv_2d_low_noise() -> LinearGaussianFixture:
    A, C, Q, R, m0, P0 = _constant_velocity_matrices(r=0.03)
    states, observations = _simulate_lgssm(A, C, Q, R, m0, P0, horizon=50, seed=403)
    return LinearGaussianFixture(
        name="lgssm_cv_2d_low_noise",
        A=A,
        C=C,
        Q=Q,
        R=R,
        m0=m0,
        P0=P0,
        states=states,
        observations=observations,
    )
