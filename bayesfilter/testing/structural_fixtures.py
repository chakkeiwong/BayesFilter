"""Toy structural models used by BayesFilter tests and documentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bayesfilter.filters.kalman import LinearGaussianStateSpace
from bayesfilter.structural import StatePartition


@dataclass(frozen=True)
class AR2StructuralModel:
    phi1: float = 0.45
    phi2: float = -0.15
    sigma: float = 0.30
    observation_sigma: float = 0.20

    @property
    def partition(self) -> StatePartition:
        return StatePartition(
            state_names=("m", "lag_m"),
            stochastic_indices=(0,),
            deterministic_indices=(1,),
            innovation_dim=1,
        )

    def initial_mean(self, theta=None) -> np.ndarray:
        return np.zeros(2)

    def initial_cov(self, theta=None) -> np.ndarray:
        return np.eye(2)

    def innovation_cov(self, theta=None) -> np.ndarray:
        return np.array([[1.0]], dtype=float)

    def observation_cov(self, theta=None) -> np.ndarray:
        return np.array([[self.observation_sigma**2]], dtype=float)

    def transition(self, previous_state, innovation, theta=None) -> np.ndarray:
        previous = np.asarray(previous_state, dtype=float)
        eps = float(np.asarray(innovation, dtype=float)[0])
        m_next = self.phi1 * previous[0] + self.phi2 * previous[1] + self.sigma * eps
        lag_next = previous[0]
        return np.array([m_next, lag_next], dtype=float)

    def observe(self, state_points, theta=None) -> np.ndarray:
        states = np.asarray(state_points, dtype=float)
        return states[:, :1]

    def as_lgssm(self) -> LinearGaussianStateSpace:
        return LinearGaussianStateSpace(
            initial_mean=self.initial_mean(),
            initial_covariance=self.initial_cov(),
            transition_offset=np.zeros(2),
            transition_matrix=np.array([[self.phi1, self.phi2], [1.0, 0.0]], dtype=float),
            transition_covariance=np.array([[self.sigma**2, 0.0], [0.0, 0.0]], dtype=float),
            observation_offset=np.zeros(1),
            observation_matrix=np.array([[1.0, 0.0]], dtype=float),
            observation_covariance=self.observation_cov(),
            partition=self.partition,
        )


@dataclass(frozen=True)
class NonlinearAccumulationModel:
    rho: float = 0.70
    sigma: float = 0.25
    alpha: float = 0.55
    beta: float = 0.80
    observation_sigma: float = 0.30

    @property
    def partition(self) -> StatePartition:
        return StatePartition(
            state_names=("m", "k"),
            stochastic_indices=(0,),
            deterministic_indices=(1,),
            innovation_dim=1,
        )

    def initial_mean(self, theta=None) -> np.ndarray:
        return np.zeros(2)

    def initial_cov(self, theta=None) -> np.ndarray:
        return np.diag([0.25, 0.20])

    def innovation_cov(self, theta=None) -> np.ndarray:
        return np.array([[1.0]], dtype=float)

    def observation_cov(self, theta=None) -> np.ndarray:
        return np.array([[self.observation_sigma**2]], dtype=float)

    def transition(self, previous_state, innovation, theta=None) -> np.ndarray:
        previous = np.asarray(previous_state, dtype=float)
        eps = float(np.asarray(innovation, dtype=float)[0])
        m_next = self.rho * previous[0] + self.sigma * eps
        k_next = self.alpha * previous[1] + self.beta * np.tanh(m_next)
        return np.array([m_next, k_next], dtype=float)

    def observe(self, state_points, theta=None) -> np.ndarray:
        states = np.asarray(state_points, dtype=float)
        return (states[:, 0] + states[:, 1])[:, None]
