"""TensorFlow linear Gaussian state-space contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from bayesfilter.structural import StatePartition


def _tensor(value: Any, *, rank: int | tuple[int, ...]) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    ranks = (rank,) if isinstance(rank, int) else tuple(rank)
    if tensor.shape.rank not in ranks:
        raise ValueError(f"expected tensor rank in {ranks}, got {tensor.shape.rank}")
    return tensor


def _last_dims(tensor: tf.Tensor, rank: int) -> tuple[int | None, ...]:
    shape = tensor.shape.as_list()
    return tuple(shape[-rank:])


@dataclass(frozen=True)
class TFLinearGaussianStateSpace:
    """TensorFlow linear Gaussian state-space object.

    Time-invariant tensors are accepted first.  Per-time tensors are allowed for
    offsets, matrices, and covariances when their leading dimension is time.
    """

    initial_mean: tf.Tensor
    initial_covariance: tf.Tensor
    transition_offset: tf.Tensor
    transition_matrix: tf.Tensor
    transition_covariance: tf.Tensor
    observation_offset: tf.Tensor
    observation_matrix: tf.Tensor
    observation_covariance: tf.Tensor
    observation_mask: tf.Tensor | None = None
    partition: StatePartition | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_mean", _tensor(self.initial_mean, rank=1))
        object.__setattr__(
            self,
            "initial_covariance",
            _tensor(self.initial_covariance, rank=2),
        )
        object.__setattr__(
            self,
            "transition_offset",
            _tensor(self.transition_offset, rank=(1, 2)),
        )
        object.__setattr__(
            self,
            "transition_matrix",
            _tensor(self.transition_matrix, rank=(2, 3)),
        )
        object.__setattr__(
            self,
            "transition_covariance",
            _tensor(self.transition_covariance, rank=(2, 3)),
        )
        object.__setattr__(
            self,
            "observation_offset",
            _tensor(self.observation_offset, rank=(1, 2)),
        )
        object.__setattr__(
            self,
            "observation_matrix",
            _tensor(self.observation_matrix, rank=(2, 3)),
        )
        object.__setattr__(
            self,
            "observation_covariance",
            _tensor(self.observation_covariance, rank=(2, 3)),
        )
        if self.observation_mask is not None:
            object.__setattr__(
                self,
                "observation_mask",
                tf.convert_to_tensor(self.observation_mask, dtype=tf.bool),
            )
        self.validate_static_shapes()

    @property
    def state_dim(self) -> int | None:
        return self.initial_mean.shape[-1]

    @property
    def observation_dim(self) -> int | None:
        return self.observation_offset.shape[-1]

    def validate_static_shapes(self) -> None:
        n = self.state_dim
        m = self.observation_dim
        if n is not None:
            expected_state_cov = (n, n)
            if _last_dims(self.initial_covariance, 2) != expected_state_cov:
                raise ValueError("initial_covariance has incompatible shape")
            for name in ("transition_matrix", "transition_covariance"):
                if _last_dims(getattr(self, name), 2) != expected_state_cov:
                    raise ValueError(f"{name} has incompatible state shape")
            if self.transition_offset.shape[-1] not in (n, None):
                raise ValueError("transition_offset has incompatible state shape")
            if self.partition is not None and self.partition.state_dim != n:
                raise ValueError("partition state_dim does not match state_dim")
        if m is not None:
            if _last_dims(self.observation_covariance, 2) != (m, m):
                raise ValueError("observation_covariance has incompatible shape")
            if self.observation_matrix.shape[-2] not in (m, None):
                raise ValueError("observation_matrix has incompatible observation shape")
        if n is not None and self.observation_matrix.shape[-1] not in (n, None):
            raise ValueError("observation_matrix has incompatible state shape")


@dataclass(frozen=True)
class TFLinearGaussianStateSpaceDerivatives:
    """First- and second-order TF derivatives of an LGSSM."""

    d_initial_mean: tf.Tensor
    d_initial_covariance: tf.Tensor
    d_transition_offset: tf.Tensor
    d_transition_matrix: tf.Tensor
    d_transition_covariance: tf.Tensor
    d_observation_offset: tf.Tensor
    d_observation_matrix: tf.Tensor
    d_observation_covariance: tf.Tensor
    d2_initial_mean: tf.Tensor
    d2_initial_covariance: tf.Tensor
    d2_transition_offset: tf.Tensor
    d2_transition_matrix: tf.Tensor
    d2_transition_covariance: tf.Tensor
    d2_observation_offset: tf.Tensor
    d2_observation_matrix: tf.Tensor
    d2_observation_covariance: tf.Tensor

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                tf.convert_to_tensor(getattr(self, name), dtype=tf.float64),
            )
        self.validate_static_shapes()

    @property
    def parameter_dim(self) -> int | None:
        return self.d_initial_mean.shape[0]

    @property
    def state_dim(self) -> int | None:
        return self.d_initial_mean.shape[-1]

    @property
    def observation_dim(self) -> int | None:
        return self.d_observation_offset.shape[-1]

    def validate_static_shapes(self) -> None:
        p = self.parameter_dim
        n = self.state_dim
        m = self.observation_dim
        if p is None or n is None or m is None:
            return
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
            if tuple(getattr(self, name).shape.as_list()) != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")
