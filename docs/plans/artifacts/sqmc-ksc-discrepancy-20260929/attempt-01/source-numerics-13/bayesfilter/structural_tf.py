"""TensorFlow structural nonlinear state-space contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

import tensorflow as tf

from bayesfilter.diagnostics import TFFilterDiagnostics, TFRegularizationDiagnostics
from bayesfilter.linear.types_tf import TFLinearGaussianStateSpace
from bayesfilter.structural import (
    FilterRunMetadata,
    StatePartition,
    StructuralFilterConfig,
    validate_filter_config,
)


TFTransitionFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
TFObservationFn = Callable[[tf.Tensor], tf.Tensor]
TFDeterministicResidualFn = Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]


def _tensor(value: object, *, rank: int | tuple[int, ...], name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    ranks = (rank,) if isinstance(rank, int) else tuple(rank)
    if tensor.shape.rank not in ranks:
        raise ValueError(f"{name} expected tensor rank in {ranks}, got {tensor.shape.rank}")
    return tensor


def _as_points(values: tf.Tensor, *, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(values, dtype=tf.float64)
    if tensor.shape.rank == 1:
        return tensor[tf.newaxis, :]
    if tensor.shape.rank == 2:
        return tensor
    raise ValueError(f"{name} must be one- or two-dimensional")


def _maybe_squeeze_points(values: tf.Tensor, was_vector: bool) -> tf.Tensor:
    return values[0] if was_vector else values


def _row_gather(matrix: tf.Tensor, indices: tuple[int, ...]) -> tf.Tensor:
    return tf.gather(matrix, tf.constant(indices, dtype=tf.int32), axis=0)


def _vector_gather(vector: tf.Tensor, indices: tuple[int, ...]) -> tf.Tensor:
    return tf.gather(vector, tf.constant(indices, dtype=tf.int32), axis=0)


def _freeze(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({str(key): value for key, value in values.items()})


@dataclass(frozen=True)
class TFStructuralStateSpace:
    """Generic TensorFlow structural state-space contract.

    The contract separates the declared innovation integration space from the
    completed state transition.  A backend places points in the innovation
    block, calls ``transition_fn`` pointwise or batched, and can verify
    deterministic identities through ``deterministic_residual_fn``.
    """

    partition: StatePartition
    config: StructuralFilterConfig
    initial_mean: tf.Tensor
    initial_covariance: tf.Tensor
    innovation_covariance: tf.Tensor
    observation_covariance: tf.Tensor
    transition_fn: TFTransitionFn
    observation_fn: TFObservationFn
    deterministic_residual_fn: TFDeterministicResidualFn | None = None
    transition_offset: tf.Tensor | None = None
    transition_matrix: tf.Tensor | None = None
    innovation_matrix: tf.Tensor | None = None
    observation_offset: tf.Tensor | None = None
    observation_matrix: tf.Tensor | None = None
    name: str = "tf_structural_state_space"

    def __post_init__(self) -> None:
        validate_filter_config(self.partition, self.config)
        object.__setattr__(
            self,
            "initial_mean",
            _tensor(self.initial_mean, rank=1, name="initial_mean"),
        )
        object.__setattr__(
            self,
            "initial_covariance",
            _tensor(self.initial_covariance, rank=2, name="initial_covariance"),
        )
        object.__setattr__(
            self,
            "innovation_covariance",
            _tensor(self.innovation_covariance, rank=2, name="innovation_covariance"),
        )
        object.__setattr__(
            self,
            "observation_covariance",
            _tensor(self.observation_covariance, rank=2, name="observation_covariance"),
        )
        for name in (
            "transition_offset",
            "transition_matrix",
            "innovation_matrix",
            "observation_offset",
            "observation_matrix",
        ):
            value = getattr(self, name)
            if value is not None:
                expected_rank = 1 if name.endswith("_offset") else 2
                object.__setattr__(
                    self,
                    name,
                    _tensor(value, rank=expected_rank, name=name),
                )
        self.validate_static_shapes()

    @property
    def state_dim(self) -> int | None:
        return self.initial_mean.shape[-1]

    @property
    def innovation_dim(self) -> int | None:
        return self.innovation_covariance.shape[-1]

    @property
    def observation_dim(self) -> int | None:
        return self.observation_covariance.shape[-1]

    @property
    def is_affine(self) -> bool:
        return (
            self.transition_offset is not None
            and self.transition_matrix is not None
            and self.innovation_matrix is not None
            and self.observation_offset is not None
            and self.observation_matrix is not None
        )

    def validate_static_shapes(self) -> None:
        n = self.state_dim
        q = self.innovation_dim
        m = self.observation_dim
        if n is not None:
            if self.partition.state_dim != n:
                raise ValueError("partition state_dim does not match initial_mean")
            if tuple(self.initial_covariance.shape.as_list()) != (n, n):
                raise ValueError("initial_covariance has incompatible state shape")
            if self.transition_offset is not None and self.transition_offset.shape[-1] != n:
                raise ValueError("transition_offset has incompatible state shape")
            if self.transition_matrix is not None and tuple(self.transition_matrix.shape.as_list()) != (n, n):
                raise ValueError("transition_matrix has incompatible state shape")
            if self.observation_matrix is not None and self.observation_matrix.shape[-1] != n:
                raise ValueError("observation_matrix has incompatible state shape")
        if q is not None:
            if self.partition.innovation_dim != q:
                raise ValueError("partition innovation_dim does not match innovation_covariance")
            if tuple(self.innovation_covariance.shape.as_list()) != (q, q):
                raise ValueError("innovation_covariance has incompatible innovation shape")
            if self.innovation_matrix is not None and self.innovation_matrix.shape[-1] != q:
                raise ValueError("innovation_matrix has incompatible innovation shape")
        if n is not None and q is not None and self.innovation_matrix is not None:
            if tuple(self.innovation_matrix.shape.as_list()) != (n, q):
                raise ValueError("innovation_matrix has incompatible state/innovation shape")
        if m is not None:
            if tuple(self.observation_covariance.shape.as_list()) != (m, m):
                raise ValueError("observation_covariance has incompatible observation shape")
            if self.observation_offset is not None and self.observation_offset.shape[-1] != m:
                raise ValueError("observation_offset has incompatible observation shape")
            if self.observation_matrix is not None and self.observation_matrix.shape[0] != m:
                raise ValueError("observation_matrix has incompatible observation shape")

    def transition(self, previous_state: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        previous = tf.convert_to_tensor(previous_state, dtype=tf.float64)
        innovation = tf.convert_to_tensor(innovation, dtype=tf.float64)
        return self.transition_fn(previous, innovation)

    def observe(self, state_points: tf.Tensor) -> tf.Tensor:
        return self.observation_fn(tf.convert_to_tensor(state_points, dtype=tf.float64))

    def deterministic_residual(
        self,
        previous_state: tf.Tensor,
        innovation: tf.Tensor,
        next_state: tf.Tensor,
    ) -> tf.Tensor:
        previous = tf.convert_to_tensor(previous_state, dtype=tf.float64)
        innovation = tf.convert_to_tensor(innovation, dtype=tf.float64)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float64)
        if self.deterministic_residual_fn is None:
            return tf.zeros(
                [tf.shape(_as_points(next_state, name="next_state"))[0], 0],
                dtype=tf.float64,
            )
        return self.deterministic_residual_fn(previous, innovation, next_state)


def make_affine_structural_tf(
    *,
    partition: StatePartition,
    initial_mean: tf.Tensor,
    initial_covariance: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    innovation_matrix: tf.Tensor,
    innovation_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    config: StructuralFilterConfig | None = None,
    name: str = "tf_affine_structural_state_space",
) -> TFStructuralStateSpace:
    """Build a structural TF model from an affine transition and observation."""

    config = config or StructuralFilterConfig(
        integration_space="innovation",
        deterministic_completion="required" if partition.deterministic_dim else "none",
    )
    transition_offset = _tensor(transition_offset, rank=1, name="transition_offset")
    transition_matrix = _tensor(transition_matrix, rank=2, name="transition_matrix")
    innovation_matrix = _tensor(innovation_matrix, rank=2, name="innovation_matrix")
    observation_offset = _tensor(observation_offset, rank=1, name="observation_offset")
    observation_matrix = _tensor(observation_matrix, rank=2, name="observation_matrix")

    def transition_fn(previous_state: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        was_vector = tf.convert_to_tensor(previous_state).shape.rank == 1
        previous_points = _as_points(previous_state, name="previous_state")
        innovation_points = _as_points(innovation, name="innovation")
        next_points = (
            transition_offset[tf.newaxis, :]
            + previous_points @ tf.transpose(transition_matrix)
            + innovation_points @ tf.transpose(innovation_matrix)
        )
        return _maybe_squeeze_points(next_points, was_vector)

    def observation_fn(state_points: tf.Tensor) -> tf.Tensor:
        was_vector = tf.convert_to_tensor(state_points).shape.rank == 1
        points = _as_points(state_points, name="state_points")
        observations = observation_offset[tf.newaxis, :] + points @ tf.transpose(observation_matrix)
        return _maybe_squeeze_points(observations, was_vector)

    def residual_fn(
        previous_state: tf.Tensor,
        innovation: tf.Tensor,
        next_state: tf.Tensor,
    ) -> tf.Tensor:
        if not partition.deterministic_indices:
            return tf.zeros([tf.shape(_as_points(next_state, name="next_state"))[0], 0], dtype=tf.float64)
        previous_points = _as_points(previous_state, name="previous_state")
        innovation_points = _as_points(innovation, name="innovation")
        next_points = _as_points(next_state, name="next_state")
        deterministic_rows = partition.deterministic_indices
        expected = (
            _vector_gather(transition_offset, deterministic_rows)[tf.newaxis, :]
            + previous_points @ tf.transpose(_row_gather(transition_matrix, deterministic_rows))
            + innovation_points @ tf.transpose(_row_gather(innovation_matrix, deterministic_rows))
        )
        actual = tf.gather(
            next_points,
            tf.constant(deterministic_rows, dtype=tf.int32),
            axis=1,
        )
        return actual - expected

    return TFStructuralStateSpace(
        partition=partition,
        config=config,
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        innovation_covariance=innovation_covariance,
        observation_covariance=observation_covariance,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        deterministic_residual_fn=residual_fn,
        transition_offset=transition_offset,
        transition_matrix=transition_matrix,
        innovation_matrix=innovation_matrix,
        observation_offset=observation_offset,
        observation_matrix=observation_matrix,
        name=name,
    )


def affine_structural_to_linear_gaussian_tf(
    model: TFStructuralStateSpace,
    *,
    observation_mask: tf.Tensor | None = None,
) -> TFLinearGaussianStateSpace:
    """Convert an affine structural TF model to its full-state LGSSM law."""

    if not model.is_affine:
        raise ValueError("only affine structural models can be converted to an LGSSM")
    transition_covariance = (
        model.innovation_matrix
        @ model.innovation_covariance
        @ tf.transpose(model.innovation_matrix)
    )
    return TFLinearGaussianStateSpace(
        initial_mean=model.initial_mean,
        initial_covariance=model.initial_covariance,
        transition_offset=model.transition_offset,
        transition_matrix=model.transition_matrix,
        transition_covariance=transition_covariance,
        observation_offset=model.observation_offset,
        observation_matrix=model.observation_matrix,
        observation_covariance=model.observation_covariance,
        observation_mask=observation_mask,
        partition=model.partition,
    )


def pointwise_deterministic_residuals(
    model: TFStructuralStateSpace,
    previous_points: tf.Tensor,
    innovation_points: tf.Tensor,
) -> tf.Tensor:
    """Propagate points and return deterministic-completion residuals."""

    next_points = model.transition(previous_points, innovation_points)
    return model.deterministic_residual(previous_points, innovation_points, next_points)


def structural_block_metadata(model: TFStructuralStateSpace) -> Mapping[str, object]:
    """Return explicit state-block metadata for structural diagnostics."""

    partition = model.partition
    return _freeze(
        {
            "state_names": partition.state_names,
            "stochastic_indices": partition.stochastic_indices,
            "deterministic_indices": partition.deterministic_indices,
            "auxiliary_indices": partition.auxiliary_indices,
            "external_indices": partition.external_indices,
            "innovation_dim": partition.innovation_dim,
            "exogenous_block": {
                "innovation_dim": partition.innovation_dim,
                "external_indices": partition.external_indices,
            },
            "endogenous_block": {
                "stochastic_indices": partition.stochastic_indices,
                "deterministic_indices": partition.deterministic_indices,
                "auxiliary_indices": partition.auxiliary_indices,
            },
            "deterministic_completion_block": {
                "policy": model.config.deterministic_completion,
                "indices": partition.deterministic_indices,
            },
            "integration_space": model.config.integration_space,
            "collapsed_full_state": (
                model.config.integration_space == "full_state" and partition.is_mixed
            ),
        }
    )


def structural_filter_metadata(
    model: TFStructuralStateSpace,
    *,
    filter_name: str,
    differentiability_status: str = "value_protocol",
    compiled_status: str = "eager_tf",
) -> FilterRunMetadata:
    """Build common provenance for TF structural filter-like results."""

    return FilterRunMetadata(
        filter_name=filter_name,
        partition=model.partition,
        integration_space=model.config.integration_space,
        deterministic_completion=model.config.deterministic_completion,
        approximation_label=model.config.approximation_label,
        differentiability_status=differentiability_status,
        compiled_status=compiled_status,
    )


def structural_filter_diagnostics(
    model: TFStructuralStateSpace,
    *,
    backend: str = "tf_structural_protocol",
    mask_convention: str = "none",
) -> TFFilterDiagnostics:
    """Build diagnostics that separate structural law from regularization."""

    return TFFilterDiagnostics(
        backend=backend,
        mask_convention=mask_convention,
        regularization=TFRegularizationDiagnostics(
            jitter=tf.constant(0.0, dtype=tf.float64),
            singular_floor=tf.constant(0.0, dtype=tf.float64),
            floor_count=tf.constant(0, dtype=tf.int32),
            psd_projection_residual=tf.constant(0.0, dtype=tf.float64),
            implemented_covariance=None,
            branch_label="structural_protocol",
            derivative_target="none",
        ),
        extra=structural_block_metadata(model),
    )
