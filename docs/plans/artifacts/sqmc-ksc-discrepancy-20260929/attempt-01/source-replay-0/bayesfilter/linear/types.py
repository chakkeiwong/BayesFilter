"""Core linear Gaussian state-space types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bayesfilter.structural import StatePartition


def _array(value, *, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"expected array with ndim={ndim}, got {arr.ndim}")
    return arr


@dataclass(frozen=True)
class LinearGaussianStateSpace:
    """Time-invariant linear Gaussian state-space object.

    Observations satisfy
    ``y_t = observation_offset + observation_matrix @ x_t + noise_t`` and the
    transition satisfies
    ``x_t = transition_offset + transition_matrix @ x_{t-1} + shock_t``.
    """

    initial_mean: np.ndarray
    initial_covariance: np.ndarray
    transition_offset: np.ndarray
    transition_matrix: np.ndarray
    transition_covariance: np.ndarray
    observation_offset: np.ndarray
    observation_matrix: np.ndarray
    observation_covariance: np.ndarray
    partition: StatePartition | None = None

    def __post_init__(self) -> None:
        for name, ndim in (
            ("initial_mean", 1),
            ("initial_covariance", 2),
            ("transition_offset", 1),
            ("transition_matrix", 2),
            ("transition_covariance", 2),
            ("observation_offset", 1),
            ("observation_matrix", 2),
            ("observation_covariance", 2),
        ):
            object.__setattr__(self, name, _array(getattr(self, name), ndim=ndim))
        self.validate()

    @property
    def state_dim(self) -> int:
        return int(self.transition_matrix.shape[0])

    @property
    def observation_dim(self) -> int:
        return int(self.observation_matrix.shape[0])

    def validate(self) -> None:
        n = self.transition_matrix.shape[0]
        m = self.observation_matrix.shape[0]
        expected = {
            "initial_mean": (n,),
            "initial_covariance": (n, n),
            "transition_offset": (n,),
            "transition_matrix": (n, n),
            "transition_covariance": (n, n),
            "observation_offset": (m,),
            "observation_matrix": (m, n),
            "observation_covariance": (m, m),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(
                    f"{name} has shape {getattr(self, name).shape}, expected {shape}"
                )
        if self.partition is not None and self.partition.state_dim != n:
            raise ValueError("partition state_dim does not match transition_matrix")


@dataclass(frozen=True)
class LinearGaussianStateSpaceDerivatives:
    """First- and second-order derivatives of a time-invariant LGSSM."""

    d_initial_mean: np.ndarray
    d_initial_covariance: np.ndarray
    d_transition_offset: np.ndarray
    d_transition_matrix: np.ndarray
    d_transition_covariance: np.ndarray
    d_observation_offset: np.ndarray
    d_observation_matrix: np.ndarray
    d_observation_covariance: np.ndarray
    d2_initial_mean: np.ndarray
    d2_initial_covariance: np.ndarray
    d2_transition_offset: np.ndarray
    d2_transition_matrix: np.ndarray
    d2_transition_covariance: np.ndarray
    d2_observation_offset: np.ndarray
    d2_observation_matrix: np.ndarray
    d2_observation_covariance: np.ndarray

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _array(getattr(self, name)))
        self.validate()

    @property
    def parameter_dim(self) -> int:
        return int(self.d_initial_mean.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.d_initial_mean.shape[1])

    @property
    def observation_dim(self) -> int:
        return int(self.d_observation_offset.shape[1])

    def validate(self) -> None:
        p = self.parameter_dim
        n = self.state_dim
        m = self.observation_dim
        expected = {
            "d_initial_mean": (p, n),
            "d_initial_covariance": (p, n, n),
            "d_transition_offset": (p, n),
            "d_transition_matrix": (p, n, n),
            "d_transition_covariance": (p, n, n),
            "d_observation_offset": (p, m),
            "d_observation_matrix": (p, m, n),
            "d_observation_covariance": (p, m, m),
            "d2_initial_mean": (p, p, n),
            "d2_initial_covariance": (p, p, n, n),
            "d2_transition_offset": (p, p, n),
            "d2_transition_matrix": (p, p, n, n),
            "d2_transition_covariance": (p, p, n, n),
            "d2_observation_offset": (p, p, m),
            "d2_observation_matrix": (p, p, m, n),
            "d2_observation_covariance": (p, p, m, m),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(
                    f"{name} has shape {getattr(self, name).shape}, expected {shape}"
                )
