"""TensorFlow Fixed-SGQF cloud, branch, and value-path helpers."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import tensorflow as tf


FIXED_SGQF_RUNTIME_MODE = "eager_only_python_branch_records"


@dataclass(frozen=True)
class TFFixedSGQFBranchHash:
    """SHA-256 digest over a canonical Fixed-SGQF branch manifest."""

    value: str

    def __post_init__(self) -> None:
        value = str(self.value)
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("branch hash must be a lowercase SHA-256 hex digest")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class TFFixedSGQFBranchManifest:
    """Canonical Fixed-SGQF branch manifest."""

    version: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not str(self.version).strip():
            raise ValueError("version must be nonempty")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")

    def to_canonical_bytes(self) -> bytes:
        canonical = {
            "type": "TFFixedSGQFBranchManifest",
            "version": str(self.version),
            "payload": _canonicalize(self.payload),
        }
        return json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def sha256(self) -> TFFixedSGQFBranchHash:
        return TFFixedSGQFBranchHash(hashlib.sha256(self.to_canonical_bytes()).hexdigest())


@dataclass(frozen=True)
class TFFixedSGQFBranchIdentity:
    """Validated identity tying a Fixed-SGQF manifest to its full hash."""

    manifest: TFFixedSGQFBranchManifest
    hash: TFFixedSGQFBranchHash

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, TFFixedSGQFBranchManifest):
            raise TypeError("manifest must be a TFFixedSGQFBranchManifest")
        if not isinstance(self.hash, TFFixedSGQFBranchHash):
            raise TypeError("hash must be a TFFixedSGQFBranchHash")
        expected = self.manifest.sha256()
        if self.hash != expected:
            raise ValueError("fixed_sgqf_branch_hash_rejected")


@dataclass(frozen=True)
class TFFixedSGQF1DLevelRule:
    """One-dimensional standard-normal GHQ level rule."""

    level: int
    nodes: tf.Tensor
    weights: tf.Tensor
    polynomial_degree: int
    family: str = "standard_normal_ghq"

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tf.convert_to_tensor(self.nodes, dtype=tf.float64))
        object.__setattr__(self, "weights", tf.convert_to_tensor(self.weights, dtype=tf.float64))
        if int(self.level) <= 0:
            raise ValueError("level must be positive")
        if self.nodes.shape.rank != 1 or self.weights.shape.rank != 1:
            raise ValueError("nodes and weights must be one-dimensional")
        if self.nodes.shape[0] != self.weights.shape[0]:
            raise ValueError("nodes and weights must have the same length")
        expected_count = 2 * int(self.level) - 1
        if self.nodes.shape[0] != expected_count:
            raise ValueError(f"level {self.level} must use {expected_count} points")
        if self.polynomial_degree != 2 * expected_count - 1:
            raise ValueError("polynomial_degree must match the GHQ point count")
        if self.family != "standard_normal_ghq":
            raise ValueError("family must be standard_normal_ghq")

    @property
    def point_count(self) -> int:
        return int(self.nodes.shape[0])


@dataclass(frozen=True)
class TFFixedSGQFCloud:
    """Merged standardized sparse-grid cloud for the Fixed-SGQF lane."""

    dim: int
    sparse_level: int
    points: tf.Tensor
    weights: tf.Tensor
    active_multi_indices: tuple[tuple[int, ...], ...]
    combination_coefficients: tuple[int, ...]
    merge_tolerance: float
    zero_weight_tolerance: float
    node_ordering: str = "lexicographic"
    stored_representation: str = "points_and_weights"
    rule_family: str = "standard_normal_ghq"
    merge_tolerance_policy: str = "scaled_by_max_1_supnorm"

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tf.convert_to_tensor(self.points, dtype=tf.float64))
        object.__setattr__(self, "weights", tf.convert_to_tensor(self.weights, dtype=tf.float64))
        object.__setattr__(self, "merge_tolerance", float(self.merge_tolerance))
        object.__setattr__(self, "zero_weight_tolerance", float(self.zero_weight_tolerance))
        if int(self.dim) <= 0:
            raise ValueError("dim must be positive")
        if int(self.sparse_level) <= 0:
            raise ValueError("sparse_level must be positive")
        if self.points.shape.rank != 2 or self.points.shape[1] != int(self.dim):
            raise ValueError("points must have shape [point_count, dim]")
        if self.weights.shape.rank != 1 or self.weights.shape[0] != self.points.shape[0]:
            raise ValueError("weights must have shape [point_count]")
        if self.merge_tolerance < 0.0 or self.zero_weight_tolerance < 0.0:
            raise ValueError("merge and zero-weight tolerances must be nonnegative")
        if self.node_ordering != "lexicographic":
            raise ValueError("node_ordering must be lexicographic")
        if self.stored_representation != "points_and_weights":
            raise ValueError("stored_representation must be points_and_weights")
        if self.rule_family != "standard_normal_ghq":
            raise ValueError("rule_family must be standard_normal_ghq")
        if self.merge_tolerance_policy != "scaled_by_max_1_supnorm":
            raise ValueError("merge_tolerance_policy must be scaled_by_max_1_supnorm")
        if len(self.active_multi_indices) != len(self.combination_coefficients):
            raise ValueError("active_multi_indices and combination_coefficients must align")
        for multi_index in self.active_multi_indices:
            if len(multi_index) != int(self.dim):
                raise ValueError("active multi-index dimension mismatch")
            if any(int(level) <= 0 for level in multi_index):
                raise ValueError("active multi-indices must be positive")

    @property
    def point_count(self) -> int:
        return int(self.weights.shape[0])

    @property
    def max_univariate_level(self) -> int:
        return max(max(multi_index) for multi_index in self.active_multi_indices)

    @property
    def weight_total(self) -> float:
        return float(tf.reduce_sum(self.weights))

    @property
    def negative_weight_count(self) -> int:
        return int(tf.reduce_sum(tf.cast(self.weights < 0.0, tf.int32)))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "FixedSGQFCloud",
            "dim": int(self.dim),
            "sparse_level": int(self.sparse_level),
            "rule_family": self.rule_family,
            "active_multi_indices": self.active_multi_indices,
            "combination_coefficients": tuple(int(value) for value in self.combination_coefficients),
            "merge_tolerance": float(self.merge_tolerance),
            "merge_tolerance_policy": self.merge_tolerance_policy,
            "zero_weight_tolerance": float(self.zero_weight_tolerance),
            "node_ordering": self.node_ordering,
            "stored_representation": self.stored_representation,
            "max_univariate_level": int(self.max_univariate_level),
            "weight_total": float(self.weight_total),
            "negative_weight_count": int(self.negative_weight_count),
            "points": self.points,
            "weights": self.weights,
        }


@dataclass(frozen=True)
class TFFixedSGQFBranchConfig:
    """Declared branch payload for Fixed-SGQF same-scalar checks."""

    observation_preprocessing: str = "identity"
    initial_condition_policy: str = "theta_defined_initial_law"
    failure_record_policy: str = "time_stage_reason_diagnostics"
    factor_branch: str = "chol_lower_positive_diag"
    predictive_epsilon: float = 1e-10
    innovation_epsilon: float = 1e-10
    additive_noise_policy: str = "analytic_q_r_no_state_augmentation"
    veto_policy: str = "symmetrize_then_veto"
    manifest_version: str = "fixed_sgqf.branch.v1"

    def __post_init__(self) -> None:
        for name in (
            "observation_preprocessing",
            "initial_condition_policy",
            "failure_record_policy",
            "factor_branch",
            "additive_noise_policy",
            "veto_policy",
            "manifest_version",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be nonempty")
        object.__setattr__(self, "predictive_epsilon", float(self.predictive_epsilon))
        object.__setattr__(self, "innovation_epsilon", float(self.innovation_epsilon))
        if self.predictive_epsilon < 0.0 or self.innovation_epsilon < 0.0:
            raise ValueError("branch epsilons must be nonnegative")

    def manifest_payload(self, cloud: TFFixedSGQFCloud) -> Mapping[str, object]:
        return {
            "family": "FixedSGQFBranch",
            "observation_preprocessing": self.observation_preprocessing,
            "initial_condition_policy": self.initial_condition_policy,
            "failure_record_policy": self.failure_record_policy,
            "factor_branch": self.factor_branch,
            "predictive_epsilon": float(self.predictive_epsilon),
            "innovation_epsilon": float(self.innovation_epsilon),
            "additive_noise_policy": self.additive_noise_policy,
            "veto_policy": self.veto_policy,
            "cloud": cloud.manifest_payload(),
        }

    def branch_manifest(self, cloud: TFFixedSGQFCloud) -> TFFixedSGQFBranchManifest:
        return TFFixedSGQFBranchManifest(self.manifest_version, self.manifest_payload(cloud))

    def branch_identity(self, cloud: TFFixedSGQFCloud) -> TFFixedSGQFBranchIdentity:
        manifest = self.branch_manifest(cloud)
        return TFFixedSGQFBranchIdentity(manifest=manifest, hash=manifest.sha256())


@dataclass(frozen=True)
class TFFixedSGQFStepFailure:
    """Failure record for one Fixed-SGQF stage-time branch exit."""

    time_index: int
    stage: str
    reason: str
    diagnostics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if int(self.time_index) < 0:
            raise ValueError("time_index must be nonnegative")
        if not str(self.stage).strip():
            raise ValueError("stage must be nonempty")
        if not str(self.reason).strip():
            raise ValueError("reason must be nonempty")
        object.__setattr__(self, "time_index", int(self.time_index))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics or {})))


@dataclass(frozen=True)
class TFFixedSGQFStepResult:
    """Accepted or blocked one-step Fixed-SGQF recursion result."""

    accepted: bool
    time_index: int
    log_likelihood_increment: tf.Tensor | None
    predicted_mean: tf.Tensor | None
    predicted_covariance: tf.Tensor | None
    predicted_factor: tf.Tensor | None
    observation_mean: tf.Tensor | None
    innovation_covariance: tf.Tensor | None
    innovation_factor: tf.Tensor | None
    cross_covariance: tf.Tensor | None
    innovation: tf.Tensor | None
    innovation_solve: tf.Tensor | None
    gain: tf.Tensor | None
    filtered_mean: tf.Tensor | None
    filtered_covariance: tf.Tensor | None
    branch_identity: TFFixedSGQFBranchIdentity
    diagnostics: Mapping[str, object]
    failure: TFFixedSGQFStepFailure | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_index", int(self.time_index))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        for name in (
            "log_likelihood_increment",
            "predicted_mean",
            "predicted_covariance",
            "predicted_factor",
            "observation_mean",
            "innovation_covariance",
            "innovation_factor",
            "cross_covariance",
            "innovation",
            "innovation_solve",
            "gain",
            "filtered_mean",
            "filtered_covariance",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tf.convert_to_tensor(value, dtype=tf.float64))
        if not isinstance(self.branch_identity, TFFixedSGQFBranchIdentity):
            raise TypeError("branch_identity must be a TFFixedSGQFBranchIdentity")
        if self.accepted:
            if self.failure is not None:
                raise ValueError("accepted steps cannot carry a failure record")
            if self.log_likelihood_increment is None:
                raise ValueError("accepted steps require a log_likelihood_increment")
            if self.filtered_mean is None or self.filtered_covariance is None:
                raise ValueError("accepted steps require filtered state outputs")
        else:
            if self.failure is None:
                raise ValueError("blocked steps require a failure record")


@dataclass(frozen=True)
class TFFixedSGQFValueResult:
    """Whole-run Fixed-SGQF value result with optional failure."""

    log_likelihood: tf.Tensor | None
    filtered_means: tf.Tensor | None
    filtered_covariances: tf.Tensor | None
    step_results: tuple[TFFixedSGQFStepResult, ...]
    branch_identity: TFFixedSGQFBranchIdentity
    diagnostics: Mapping[str, object]
    failure: TFFixedSGQFStepFailure | None = None

    def __post_init__(self) -> None:
        if self.log_likelihood is not None:
            object.__setattr__(self, "log_likelihood", tf.convert_to_tensor(self.log_likelihood, dtype=tf.float64))
        if self.filtered_means is not None:
            object.__setattr__(self, "filtered_means", tf.convert_to_tensor(self.filtered_means, dtype=tf.float64))
        if self.filtered_covariances is not None:
            object.__setattr__(self, "filtered_covariances", tf.convert_to_tensor(self.filtered_covariances, dtype=tf.float64))
        object.__setattr__(self, "step_results", tuple(self.step_results))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        if not isinstance(self.branch_identity, TFFixedSGQFBranchIdentity):
            raise TypeError("branch_identity must be a TFFixedSGQFBranchIdentity")
        if self.failure is not None and not isinstance(self.failure, TFFixedSGQFStepFailure):
            raise TypeError("failure must be a TFFixedSGQFStepFailure")


@dataclass(frozen=True)
class TFFixedSGQFNonlinearModel:
    """Minimal additive nonlinear model for the p47 Fixed-SGQF lane."""

    initial_mean: tf.Tensor
    initial_covariance: tf.Tensor
    process_covariance: tf.Tensor
    observation_covariance: tf.Tensor
    transition_fn: Callable[[tf.Tensor], tf.Tensor]
    observation_fn: Callable[[tf.Tensor], tf.Tensor]
    name: str = "fixed_sgqf_nonlinear_model"

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_mean", tf.convert_to_tensor(self.initial_mean, dtype=tf.float64))
        object.__setattr__(self, "initial_covariance", tf.convert_to_tensor(self.initial_covariance, dtype=tf.float64))
        object.__setattr__(self, "process_covariance", tf.convert_to_tensor(self.process_covariance, dtype=tf.float64))
        object.__setattr__(self, "observation_covariance", tf.convert_to_tensor(self.observation_covariance, dtype=tf.float64))
        if self.initial_mean.shape.rank != 1:
            raise ValueError("initial_mean must be rank 1")
        state_dim = int(self.initial_mean.shape[0])
        if self.initial_covariance.shape != (state_dim, state_dim):
            raise ValueError("initial_covariance shape mismatch")
        if self.process_covariance.shape != (state_dim, state_dim):
            raise ValueError("process_covariance shape mismatch")
        if self.observation_covariance.shape.rank != 2:
            raise ValueError("observation_covariance must be rank 2")

    @property
    def state_dim(self) -> int:
        return int(self.initial_mean.shape[0])

    @property
    def observation_dim(self) -> int:
        return int(self.observation_covariance.shape[0])

    def transition(self, points: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(self.transition_fn(tf.convert_to_tensor(points, dtype=tf.float64)), dtype=tf.float64)

    def observe(self, points: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(self.observation_fn(tf.convert_to_tensor(points, dtype=tf.float64)), dtype=tf.float64)


@dataclass(frozen=True)
class TFFixedSGQFAffineModel:
    """Minimal affine Gaussian model for exact-collapse SGQF tieouts."""

    initial_mean: tf.Tensor
    initial_covariance: tf.Tensor
    transition_matrix: tf.Tensor
    process_covariance: tf.Tensor
    observation_matrix: tf.Tensor
    observation_covariance: tf.Tensor
    transition_offset: tf.Tensor | None = None
    observation_offset: tf.Tensor | None = None
    name: str = "fixed_sgqf_affine_model"

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_mean", tf.convert_to_tensor(self.initial_mean, dtype=tf.float64))
        object.__setattr__(self, "initial_covariance", tf.convert_to_tensor(self.initial_covariance, dtype=tf.float64))
        object.__setattr__(self, "transition_matrix", tf.convert_to_tensor(self.transition_matrix, dtype=tf.float64))
        object.__setattr__(self, "process_covariance", tf.convert_to_tensor(self.process_covariance, dtype=tf.float64))
        object.__setattr__(self, "observation_matrix", tf.convert_to_tensor(self.observation_matrix, dtype=tf.float64))
        object.__setattr__(self, "observation_covariance", tf.convert_to_tensor(self.observation_covariance, dtype=tf.float64))
        if self.transition_offset is None:
            object.__setattr__(self, "transition_offset", tf.zeros_like(self.initial_mean))
        else:
            object.__setattr__(self, "transition_offset", tf.convert_to_tensor(self.transition_offset, dtype=tf.float64))
        observation_dim = int(self.observation_matrix.shape[0])
        if self.observation_offset is None:
            object.__setattr__(self, "observation_offset", tf.zeros([observation_dim], dtype=tf.float64))
        else:
            object.__setattr__(self, "observation_offset", tf.convert_to_tensor(self.observation_offset, dtype=tf.float64))
        if self.initial_mean.shape.rank != 1:
            raise ValueError("initial_mean must be rank 1")
        state_dim = int(self.initial_mean.shape[0])
        if self.initial_covariance.shape != (state_dim, state_dim):
            raise ValueError("initial_covariance shape mismatch")
        if self.transition_matrix.shape != (state_dim, state_dim):
            raise ValueError("transition_matrix shape mismatch")
        if self.process_covariance.shape != (state_dim, state_dim):
            raise ValueError("process_covariance shape mismatch")
        if self.transition_offset.shape != (state_dim,):
            raise ValueError("transition_offset shape mismatch")
        if self.observation_matrix.shape.rank != 2 or self.observation_matrix.shape[1] != state_dim:
            raise ValueError("observation_matrix shape mismatch")
        if self.observation_covariance.shape != (observation_dim, observation_dim):
            raise ValueError("observation_covariance shape mismatch")
        if self.observation_offset.shape != (observation_dim,):
            raise ValueError("observation_offset shape mismatch")

    @property
    def state_dim(self) -> int:
        return int(self.initial_mean.shape[0])

    @property
    def observation_dim(self) -> int:
        return int(self.observation_matrix.shape[0])

    def transition(self, points: tf.Tensor) -> tf.Tensor:
        values = _as_points(points, name="points")
        next_points = self.transition_offset[tf.newaxis, :] + values @ tf.transpose(self.transition_matrix)
        return next_points

    def observe(self, points: tf.Tensor) -> tf.Tensor:
        values = _as_points(points, name="points")
        observations = self.observation_offset[tf.newaxis, :] + values @ tf.transpose(self.observation_matrix)
        return observations


@dataclass(frozen=True)
class TFFixedSGQFNonGaussianModel:
    """Fixed-SGQF model with direct non-Gaussian likelihood reweighting."""

    initial_mean: tf.Tensor
    initial_covariance: tf.Tensor
    process_covariance: tf.Tensor
    transition_fn: Callable[[tf.Tensor], tf.Tensor]
    observation_log_density_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
    observation_dim: int = 1
    name: str = "fixed_sgqf_nongaussian_model"

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_mean", tf.convert_to_tensor(self.initial_mean, dtype=tf.float64))
        object.__setattr__(self, "initial_covariance", tf.convert_to_tensor(self.initial_covariance, dtype=tf.float64))
        object.__setattr__(self, "process_covariance", tf.convert_to_tensor(self.process_covariance, dtype=tf.float64))
        object.__setattr__(self, "observation_dim", int(self.observation_dim))
        if self.initial_mean.shape.rank != 1:
            raise ValueError("initial_mean must be rank 1")
        state_dim = int(self.initial_mean.shape[0])
        if self.initial_covariance.shape != (state_dim, state_dim):
            raise ValueError("initial_covariance shape mismatch")
        if self.process_covariance.shape != (state_dim, state_dim):
            raise ValueError("process_covariance shape mismatch")
        if int(self.observation_dim) <= 0:
            raise ValueError("observation_dim must be positive")

    @property
    def state_dim(self) -> int:
        return int(self.initial_mean.shape[0])

    def transition(self, points: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(
            self.transition_fn(tf.convert_to_tensor(points, dtype=tf.float64)),
            dtype=tf.float64,
        )

    def observation_log_density(self, points: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(
            self.observation_log_density_fn(
                tf.convert_to_tensor(points, dtype=tf.float64),
                tf.convert_to_tensor(observation, dtype=tf.float64),
            ),
            dtype=tf.float64,
        )
        if values.shape.rank != 1:
            raise ValueError("observation_log_density_fn must return a length-point_count vector")
        return values


@dataclass(frozen=True)
class TFFixedSGQFOneStepOracle:
    """p47 one-step scalar nonlinear oracle parameters and expectations."""

    beta: tf.Tensor
    initial_mean: tf.Tensor
    initial_covariance: tf.Tensor
    process_covariance: tf.Tensor
    observation_covariance: tf.Tensor
    observation: tf.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "beta", tf.convert_to_tensor(self.beta, dtype=tf.float64))
        object.__setattr__(self, "initial_mean", tf.convert_to_tensor(self.initial_mean, dtype=tf.float64))
        object.__setattr__(self, "initial_covariance", tf.convert_to_tensor(self.initial_covariance, dtype=tf.float64))
        object.__setattr__(self, "process_covariance", tf.convert_to_tensor(self.process_covariance, dtype=tf.float64))
        object.__setattr__(self, "observation_covariance", tf.convert_to_tensor(self.observation_covariance, dtype=tf.float64))
        object.__setattr__(self, "observation", tf.convert_to_tensor(self.observation, dtype=tf.float64))

    def model(self) -> TFFixedSGQFNonlinearModel:
        beta = tf.convert_to_tensor(self.beta, dtype=tf.float64)

        def transition_fn(points: tf.Tensor) -> tf.Tensor:
            values = _as_points(points, name="points")
            return beta * values

        def observation_fn(points: tf.Tensor) -> tf.Tensor:
            values = _as_points(points, name="points")
            return tf.square(values)

        return TFFixedSGQFNonlinearModel(
            initial_mean=self.initial_mean,
            initial_covariance=self.initial_covariance,
            process_covariance=self.process_covariance,
            observation_covariance=self.observation_covariance,
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            name="p47_fixed_sgqf_scalar_oracle",
        )


def _canonicalize(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return _canonicalize_tensor(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_canonicalize(item) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_canonicalize(item) for item in value]}
    if value is None:
        return {"type": "none", "value": None}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("branch manifest floats must be finite")
        return {"type": "float64", "hex": struct.pack(">d", value).hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    raise TypeError(f"unsupported branch manifest value type: {type(value)!r}")


def _canonicalize_tensor(tensor: tf.Tensor) -> Mapping[str, object]:
    value = tf.convert_to_tensor(tensor)
    flat = tf.reshape(value, [-1])
    if value.dtype.is_floating:
        cast_flat = tf.cast(flat, tf.float64)
        encoded = [struct.pack(">d", float(item)).hex() for item in tf.unstack(cast_flat)]
    elif value.dtype.is_integer:
        encoded = [str(int(item)) for item in tf.unstack(flat)]
    elif value.dtype == tf.bool:
        encoded = [bool(item) for item in tf.unstack(flat)]
    else:
        raise TypeError(f"unsupported tensor dtype for branch manifest: {value.dtype}")
    return {
        "type": "tensor",
        "dtype": value.dtype.name,
        "shape": list(value.shape.as_list()),
        "encoding": "big_endian_scalar_list",
        "values": encoded,
    }


def _gauss_hermite_nodes_weights(order: int) -> tuple[tf.Tensor, tf.Tensor]:
    if int(order) <= 0:
        raise ValueError("order must be positive")
    if int(order) == 1:
        return (
            tf.zeros([1], dtype=tf.float64),
            tf.sqrt(tf.constant(math.pi, dtype=tf.float64))[tf.newaxis],
        )
    k = tf.cast(tf.range(1, int(order), dtype=tf.int32), tf.float64)
    beta = tf.sqrt(k / 2.0)
    jacobi = tf.linalg.diag(beta, k=1) + tf.linalg.diag(beta, k=-1)
    eigenvalues, eigenvectors = tf.linalg.eigh(jacobi)
    weights = tf.sqrt(tf.constant(math.pi, dtype=tf.float64)) * tf.square(eigenvectors[0, :])
    return eigenvalues, weights


def tf_standard_normal_ghq_level_rule(level: int) -> TFFixedSGQF1DLevelRule:
    """Return the p47 GHQ level rule ``I_level`` under the standard normal."""

    level = int(level)
    if level <= 0:
        raise ValueError("level must be positive")
    order = 2 * level - 1
    hermite_nodes, hermite_weights = _gauss_hermite_nodes_weights(order)
    nodes = tf.sqrt(tf.constant(2.0, dtype=tf.float64)) * hermite_nodes
    weights = hermite_weights / tf.sqrt(tf.constant(math.pi, dtype=tf.float64))
    return TFFixedSGQF1DLevelRule(
        level=level,
        nodes=nodes,
        weights=weights,
        polynomial_degree=2 * order - 1,
    )


def _positive_multi_indices(dim: int, total: int) -> tuple[tuple[int, ...], ...]:
    if dim == 1:
        return ((int(total),),)
    indices = []
    max_first = int(total) - int(dim) + 1
    for first in range(1, max_first + 1):
        for rest in _positive_multi_indices(dim - 1, total - first):
            indices.append((first, *rest))
    return tuple(indices)


def tf_fixed_sgqf_active_multi_indices(dim: int, sparse_level: int) -> tuple[tuple[int, ...], ...]:
    """Return the active multi-indices in the p47 Smolyak band."""

    dim = int(dim)
    sparse_level = int(sparse_level)
    if dim <= 0 or sparse_level <= 0:
        raise ValueError("dim and sparse_level must be positive")
    active = []
    for total in range(sparse_level, sparse_level + dim):
        active.extend(_positive_multi_indices(dim, total))
    return tuple(sorted(active, key=lambda multi_index: (sum(multi_index), multi_index)))


def tf_fixed_sgqf_combination_coefficient(
    dim: int,
    sparse_level: int,
    multi_index: Sequence[int],
) -> int:
    """Return the direct tensor-rule Smolyak coefficient for one active index."""

    dim = int(dim)
    sparse_level = int(sparse_level)
    multi_index_tuple = tuple(int(level) for level in multi_index)
    if len(multi_index_tuple) != dim or any(level <= 0 for level in multi_index_tuple):
        raise ValueError("multi_index must be positive and match dim")
    total = sum(multi_index_tuple)
    upper = sparse_level + dim - 1
    if total < sparse_level or total > upper:
        raise ValueError("multi_index is outside the active Smolyak band")
    gap = upper - total
    return int(((-1) ** gap) * math.comb(dim - 1, gap))


def _merge_key(point: tuple[float, ...], tolerance: float) -> tuple[object, ...]:
    if tolerance == 0.0:
        return tuple(float(value).hex() for value in point)
    return tuple(int(round(float(value) / tolerance)) for value in point)


def _find_merged_point(
    merged: dict[tuple[object, ...], dict[str, object]],
    point: tuple[float, ...],
    tolerance: float,
) -> tuple[object, ...] | None:
    key = _merge_key(point, tolerance)
    candidate_keys = [key]
    if tolerance != 0.0:
        for offsets in product((-1, 0, 1), repeat=len(point)):
            neighbor = tuple(component + offset for component, offset in zip(key, offsets))
            if neighbor not in candidate_keys:
                candidate_keys.append(neighbor)
    for candidate_key in candidate_keys:
        existing = merged.get(candidate_key)
        if existing is None:
            continue
        existing_point = tuple(float(value) for value in existing["point"])
        if max(abs(point_component - existing_component) for point_component, existing_component in zip(point, existing_point)) <= tolerance:
            return candidate_key
    return None


def _unstack_floats(values: tf.Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in tf.unstack(values))


def tf_fixed_sgqf_cloud(
    dim: int,
    sparse_level: int,
    *,
    merge_tolerance: float = 1e-12,
    zero_weight_tolerance: float = 1e-14,
    node_ordering: str = "lexicographic",
) -> TFFixedSGQFCloud:
    """Build the merged standardized sparse-grid cloud for Fixed-SGQF."""

    dim = int(dim)
    sparse_level = int(sparse_level)
    if dim <= 0 or sparse_level <= 0:
        raise ValueError("dim and sparse_level must be positive")
    if merge_tolerance < 0.0 or zero_weight_tolerance < 0.0:
        raise ValueError("merge and zero-weight tolerances must be nonnegative")
    if node_ordering != "lexicographic":
        raise ValueError("node_ordering must be lexicographic")

    active_multi_indices = tf_fixed_sgqf_active_multi_indices(dim, sparse_level)
    max_level = max(max(multi_index) for multi_index in active_multi_indices)
    rules = {level: tf_standard_normal_ghq_level_rule(level) for level in range(1, max_level + 1)}

    merged: dict[tuple[object, ...], dict[str, object]] = {}
    combination_coefficients = []
    for multi_index in active_multi_indices:
        coefficient = tf_fixed_sgqf_combination_coefficient(dim, sparse_level, multi_index)
        combination_coefficients.append(coefficient)
        rule_nodes = [_unstack_floats(rules[level].nodes) for level in multi_index]
        rule_weights = [_unstack_floats(rules[level].weights) for level in multi_index]
        axis_ranges = [range(len(nodes)) for nodes in rule_nodes]
        for choice in product(*axis_ranges):
            point = tuple(rule_nodes[axis][index] for axis, index in enumerate(choice))
            weight = float(coefficient)
            for axis, index in enumerate(choice):
                weight *= rule_weights[axis][index]
            existing_key = _find_merged_point(merged, point, merge_tolerance)
            if existing_key is not None:
                merged[existing_key]["weight"] = float(merged[existing_key]["weight"]) + weight
            else:
                merged[_merge_key(point, merge_tolerance)] = {"point": point, "weight": weight}

    items = [item for item in merged.values() if abs(float(item["weight"])) > zero_weight_tolerance]
    items.sort(key=lambda item: tuple(float(value) for value in item["point"]))
    points = tf.constant([item["point"] for item in items], dtype=tf.float64)
    weights = tf.constant([item["weight"] for item in items], dtype=tf.float64)
    return TFFixedSGQFCloud(
        dim=dim,
        sparse_level=sparse_level,
        points=points,
        weights=weights,
        active_multi_indices=active_multi_indices,
        combination_coefficients=tuple(combination_coefficients),
        merge_tolerance=merge_tolerance,
        zero_weight_tolerance=zero_weight_tolerance,
        node_ordering=node_ordering,
    )


def tf_fixed_sgqf_level2_axis_cloud(
    dim: int,
    *,
    zero_weight_tolerance: float = 1e-14,
) -> TFFixedSGQFCloud:
    """Build the exact level-2 Smolyak cloud without neighbor-key search.

    At level two the merged cloud is available in closed form: the origin and
    the two ``sqrt(3)`` points on every coordinate axis. This avoids the
    generic merge routine's exponentially large tolerance-neighbor search.
    """

    dimension = int(dim)
    tolerance = float(zero_weight_tolerance)
    if dimension <= 0:
        raise ValueError("dim must be positive")
    if tolerance < 0.0:
        raise ValueError("zero_weight_tolerance must be nonnegative")

    radius = math.sqrt(3.0)
    zero = (0.0,) * dimension
    items: list[tuple[tuple[float, ...], float]] = [
        (zero, 1.0 - float(dimension) / 3.0)
    ]
    for axis in range(dimension):
        for sign in (-1.0, 1.0):
            point = tuple(
                sign * radius if index == axis else 0.0
                for index in range(dimension)
            )
            items.append((point, 1.0 / 6.0))
    items = [item for item in items if abs(item[1]) > tolerance]
    items.sort(key=lambda item: item[0])

    active_multi_indices = tf_fixed_sgqf_active_multi_indices(dimension, 2)
    combination_coefficients = tuple(
        tf_fixed_sgqf_combination_coefficient(dimension, 2, multi_index)
        for multi_index in active_multi_indices
    )
    return TFFixedSGQFCloud(
        dim=dimension,
        sparse_level=2,
        points=tf.constant([point for point, _weight in items], tf.float64),
        weights=tf.constant([weight for _point, weight in items], tf.float64),
        active_multi_indices=active_multi_indices,
        combination_coefficients=combination_coefficients,
        merge_tolerance=0.0,
        zero_weight_tolerance=tolerance,
        node_ordering="lexicographic",
    )


def tf_fixed_sgqf_branch_identity(
    cloud: TFFixedSGQFCloud,
    *,
    observation_preprocessing: str = "identity",
    predictive_epsilon: float = 1e-10,
    innovation_epsilon: float = 1e-10,
    factor_branch: str = "chol_lower_positive_diag",
    additive_noise_policy: str = "analytic_q_r_no_state_augmentation",
    veto_policy: str = "symmetrize_then_veto",
    initial_condition_policy: str = "theta_defined_initial_law",
    failure_record_policy: str = "time_stage_reason_diagnostics",
) -> TFFixedSGQFBranchIdentity:
    """Return the canonical M1 branch identity for a fixed SGQF cloud."""

    return TFFixedSGQFBranchConfig(
        observation_preprocessing=observation_preprocessing,
        initial_condition_policy=initial_condition_policy,
        failure_record_policy=failure_record_policy,
        predictive_epsilon=predictive_epsilon,
        innovation_epsilon=innovation_epsilon,
        factor_branch=factor_branch,
        additive_noise_policy=additive_noise_policy,
        veto_policy=veto_policy,
    ).branch_identity(cloud)


def _as_observation_matrix(observations: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(observations, dtype=tf.float64)
    if values.shape.rank == 1:
        return values[:, tf.newaxis]
    if values.shape.rank != 2:
        raise ValueError("observations must be one- or two-dimensional")
    return values


def _as_points(points: tf.Tensor, *, name: str) -> tf.Tensor:
    values = tf.convert_to_tensor(points, dtype=tf.float64)
    if values.shape.rank == 1:
        return values[:, tf.newaxis]
    if values.shape.rank != 2:
        raise ValueError(f"{name} must be one- or two-dimensional")
    return values


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(matrix, dtype=tf.float64)
    return 0.5 * (values + tf.linalg.matrix_transpose(values))


def _weighted_mean(points: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return tf.linalg.matvec(tf.transpose(points), weights)


def _weighted_covariance(centered: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return _symmetrize(tf.transpose(centered) @ (centered * weights[:, tf.newaxis]))


def _freeze_mapping(values: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(values or {}))


def _common_result_diagnostics(
    *,
    branch_identity: TFFixedSGQFBranchIdentity,
    branch_config: TFFixedSGQFBranchConfig,
    cloud: TFFixedSGQFCloud,
    accepted_steps: int,
    failure: TFFixedSGQFStepFailure | None = None,
) -> Mapping[str, object]:
    diagnostics = {
        "branch_hash": branch_identity.hash.value,
        "cloud_point_count": tf.convert_to_tensor(cloud.point_count, dtype=tf.int32),
        "weight_total": tf.convert_to_tensor(cloud.weight_total, dtype=tf.float64),
        "negative_weight_count": tf.convert_to_tensor(cloud.negative_weight_count, dtype=tf.int32),
        "rule_family": cloud.rule_family,
        "runtime_mode": FIXED_SGQF_RUNTIME_MODE,
        "observation_preprocessing": branch_config.observation_preprocessing,
        "initial_condition_policy": branch_config.initial_condition_policy,
        "failure_record_policy": branch_config.failure_record_policy,
        "factor_branch": branch_config.factor_branch,
        "additive_noise_policy": branch_config.additive_noise_policy,
        "veto_policy": branch_config.veto_policy,
        "predictive_epsilon": tf.convert_to_tensor(branch_config.predictive_epsilon, dtype=tf.float64),
        "innovation_epsilon": tf.convert_to_tensor(branch_config.innovation_epsilon, dtype=tf.float64),
        "accepted_steps": tf.convert_to_tensor(accepted_steps, dtype=tf.int32),
    }
    if failure is not None:
        diagnostics["failure_stage"] = failure.stage
        diagnostics["failure_reason"] = failure.reason
        diagnostics["failure_time_index"] = tf.convert_to_tensor(failure.time_index, dtype=tf.int32)
    return _freeze_mapping(diagnostics)


def _cholesky_factor_or_failure(
    matrix: tf.Tensor,
    *,
    epsilon: float,
    time_index: int,
    stage: str,
) -> tuple[tf.Tensor | None, Mapping[str, object], TFFixedSGQFStepFailure | None]:
    sym_matrix = _symmetrize(matrix)
    eigenvalues, eigenvectors = tf.linalg.eigh(sym_matrix)
    min_eigenvalue = tf.reduce_min(eigenvalues)
    diagnostics = {
        "min_eigenvalue": min_eigenvalue,
        "epsilon": tf.convert_to_tensor(epsilon, dtype=tf.float64),
        "symmetrized_matrix": sym_matrix,
    }
    if min_eigenvalue < tf.convert_to_tensor(0.0, dtype=tf.float64):
        return None, diagnostics, TFFixedSGQFStepFailure(
            time_index=time_index,
            stage=stage,
            reason="positive_definiteness_veto",
            diagnostics=diagnostics,
        )
    if min_eigenvalue >= tf.convert_to_tensor(epsilon, dtype=tf.float64):
        factor = tf.linalg.cholesky(sym_matrix)
    else:
        factor_eigenvalues = tf.maximum(eigenvalues, tf.convert_to_tensor(0.0, dtype=tf.float64))
        factor = eigenvectors @ tf.linalg.diag(tf.sqrt(factor_eigenvalues))
    return factor, diagnostics, None


def _branch_failure_result(
    *,
    branch_identity: TFFixedSGQFBranchIdentity,
    time_index: int,
    failure: TFFixedSGQFStepFailure,
    diagnostics: Mapping[str, object],
) -> TFFixedSGQFStepResult:
    return TFFixedSGQFStepResult(
        accepted=False,
        time_index=time_index,
        log_likelihood_increment=None,
        predicted_mean=None,
        predicted_covariance=None,
        predicted_factor=None,
        observation_mean=None,
        innovation_covariance=None,
        innovation_factor=None,
        cross_covariance=None,
        innovation=None,
        innovation_solve=None,
        gain=None,
        filtered_mean=None,
        filtered_covariance=None,
        branch_identity=branch_identity,
        diagnostics=diagnostics,
        failure=failure,
    )


def tf_fixed_sgqf_filter(
    observations: tf.Tensor,
    model: TFFixedSGQFNonGaussianModel | TFFixedSGQFNonlinearModel | TFFixedSGQFAffineModel,
    *,
    cloud: TFFixedSGQFCloud,
    branch_config: TFFixedSGQFBranchConfig | None = None,
    branch_identity: TFFixedSGQFBranchIdentity | None = None,
    return_filtered: bool = True,
) -> TFFixedSGQFValueResult:
    """Evaluate the p47 Fixed-SGQF value recursion on a fixed branch."""

    y = _as_observation_matrix(observations)
    branch_config = branch_config or TFFixedSGQFBranchConfig()
    branch_identity = branch_identity or branch_config.branch_identity(cloud)
    weight_sum_tolerance = max(cloud.zero_weight_tolerance, 1e-12)
    if abs(cloud.weight_total - 1.0) > weight_sum_tolerance:
        failure = TFFixedSGQFStepFailure(
            time_index=0,
            stage="cloud",
            reason="weight_sum_failure",
            diagnostics={
                "weight_total": tf.convert_to_tensor(cloud.weight_total, dtype=tf.float64),
                "tolerance": tf.convert_to_tensor(weight_sum_tolerance, dtype=tf.float64),
            },
        )
        step = _branch_failure_result(
            branch_identity=branch_identity,
            time_index=0,
            failure=failure,
            diagnostics={
                "branch_hash": branch_identity.hash.value,
                "cloud_point_count": cloud.point_count,
            },
        )
        return TFFixedSGQFValueResult(
            log_likelihood=None,
            filtered_means=None,
            filtered_covariances=None,
            step_results=(step,),
            branch_identity=branch_identity,
            diagnostics=_common_result_diagnostics(
                branch_identity=branch_identity,
                branch_config=branch_config,
                cloud=cloud,
                accepted_steps=0,
                failure=failure,
            ),
            failure=failure,
        )

    mean = tf.convert_to_tensor(model.initial_mean, dtype=tf.float64)
    covariance = _symmetrize(model.initial_covariance)
    process_covariance = _symmetrize(model.process_covariance)
    observation_covariance = _symmetrize(model.observation_covariance)
    state_dim = int(mean.shape[0])
    observation_dim = int(observation_covariance.shape[0])
    weights = tf.convert_to_tensor(cloud.weights, dtype=tf.float64)
    log_likelihood = tf.constant(0.0, dtype=tf.float64)
    step_results: list[TFFixedSGQFStepResult] = []
    filtered_means = []
    filtered_covariances = []
    accepted_steps = 0

    for time_index in range(int(y.shape[0])):
        previous_factor, previous_diag, previous_failure = _cholesky_factor_or_failure(
            covariance,
            epsilon=branch_config.predictive_epsilon,
            time_index=time_index,
            stage="previous_covariance",
        )
        if previous_failure is not None:
            step = _branch_failure_result(
                branch_identity=branch_identity,
                time_index=time_index,
                failure=previous_failure,
                diagnostics={
                    "branch_hash": branch_identity.hash.value,
                    "previous_covariance_diagnostics": previous_diag,
                },
            )
            step_results.append(step)
            return TFFixedSGQFValueResult(
                log_likelihood=None,
                filtered_means=None,
                filtered_covariances=None,
                step_results=tuple(step_results),
                branch_identity=branch_identity,
                diagnostics=_common_result_diagnostics(
                    branch_identity=branch_identity,
                    branch_config=branch_config,
                    cloud=cloud,
                    accepted_steps=accepted_steps,
                    failure=previous_failure,
                ),
                failure=previous_failure,
            )

        previous_points = mean[tf.newaxis, :] + cloud.points @ tf.transpose(previous_factor)
        transition_values = model.transition(previous_points)
        predicted_mean = _weighted_mean(transition_values, weights)
        centered_predicted = transition_values - predicted_mean[tf.newaxis, :]
        predicted_covariance = _symmetrize(process_covariance + _weighted_covariance(centered_predicted, weights))
        predicted_factor, predictive_diag, predictive_failure = _cholesky_factor_or_failure(
            predicted_covariance,
            epsilon=branch_config.predictive_epsilon,
            time_index=time_index,
            stage="predictive_covariance",
        )
        if predictive_failure is not None:
            step = _branch_failure_result(
                branch_identity=branch_identity,
                time_index=time_index,
                failure=predictive_failure,
                diagnostics={
                    "branch_hash": branch_identity.hash.value,
                    "previous_covariance_diagnostics": previous_diag,
                    "predictive_covariance_diagnostics": predictive_diag,
                    "predicted_mean": predicted_mean,
                },
            )
            step_results.append(step)
            return TFFixedSGQFValueResult(
                log_likelihood=None,
                filtered_means=None,
                filtered_covariances=None,
                step_results=tuple(step_results),
                branch_identity=branch_identity,
                diagnostics=_common_result_diagnostics(
                    branch_identity=branch_identity,
                    branch_config=branch_config,
                    cloud=cloud,
                    accepted_steps=accepted_steps,
                    failure=predictive_failure,
                ),
                failure=predictive_failure,
            )

        predictive_points = predicted_mean[tf.newaxis, :] + cloud.points @ tf.transpose(predicted_factor)
        observation_values = model.observe(predictive_points)
        observation_mean = _weighted_mean(observation_values, weights)
        centered_observation = observation_values - observation_mean[tf.newaxis, :]
        innovation_covariance = _symmetrize(observation_covariance + _weighted_covariance(centered_observation, weights))
        innovation_factor, innovation_diag, innovation_failure = _cholesky_factor_or_failure(
            innovation_covariance,
            epsilon=branch_config.innovation_epsilon,
            time_index=time_index,
            stage="innovation_covariance",
        )
        cross_covariance = tf.transpose(predictive_points - predicted_mean[tf.newaxis, :]) @ (
            centered_observation * weights[:, tf.newaxis]
        )
        innovation = y[time_index] - observation_mean
        if innovation_failure is not None:
            step = _branch_failure_result(
                branch_identity=branch_identity,
                time_index=time_index,
                failure=innovation_failure,
                diagnostics={
                    "branch_hash": branch_identity.hash.value,
                    "predicted_mean": predicted_mean,
                    "predicted_covariance": predicted_covariance,
                    "observation_mean": observation_mean,
                    "innovation_covariance_diagnostics": innovation_diag,
                    "cross_covariance": cross_covariance,
                },
            )
            step_results.append(step)
            return TFFixedSGQFValueResult(
                log_likelihood=None,
                filtered_means=None,
                filtered_covariances=None,
                step_results=tuple(step_results),
                branch_identity=branch_identity,
                diagnostics=_common_result_diagnostics(
                    branch_identity=branch_identity,
                    branch_config=branch_config,
                    cloud=cloud,
                    accepted_steps=accepted_steps,
                    failure=innovation_failure,
                ),
                failure=innovation_failure,
            )

        innovation_solve = tf.linalg.cholesky_solve(innovation_factor, innovation[:, tf.newaxis])[:, 0]
        innovation_precision = tf.linalg.cholesky_solve(
            innovation_factor,
            tf.eye(observation_dim, dtype=tf.float64),
        )
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))
        log_likelihood_increment = -0.5 * (
            tf.cast(observation_dim, tf.float64) * tf.math.log(tf.constant(2.0 * math.pi, dtype=tf.float64))
            + log_det
            + tf.reduce_sum(innovation * innovation_solve)
        )
        gain = cross_covariance @ innovation_precision
        filtered_mean = predicted_mean + tf.linalg.matvec(gain, innovation)
        filtered_covariance = _symmetrize(
            predicted_covariance - gain @ innovation_covariance @ tf.transpose(gain)
        )
        _carried_factor, carried_diag, carried_failure = _cholesky_factor_or_failure(
            filtered_covariance,
            epsilon=branch_config.predictive_epsilon,
            time_index=time_index,
            stage="carried_covariance",
        )
        if carried_failure is not None:
            step = _branch_failure_result(
                branch_identity=branch_identity,
                time_index=time_index,
                failure=carried_failure,
                diagnostics={
                    "branch_hash": branch_identity.hash.value,
                    "predicted_mean": predicted_mean,
                    "predicted_covariance": predicted_covariance,
                    "observation_mean": observation_mean,
                    "innovation_covariance": innovation_covariance,
                    "cross_covariance": cross_covariance,
                    "innovation": innovation,
                    "gain": gain,
                    "filtered_covariance_diagnostics": carried_diag,
                },
            )
            step_results.append(step)
            return TFFixedSGQFValueResult(
                log_likelihood=None,
                filtered_means=None,
                filtered_covariances=None,
                step_results=tuple(step_results),
                branch_identity=branch_identity,
                diagnostics=_common_result_diagnostics(
                    branch_identity=branch_identity,
                    branch_config=branch_config,
                    cloud=cloud,
                    accepted_steps=accepted_steps,
                    failure=carried_failure,
                ),
                failure=carried_failure,
            )

        step = TFFixedSGQFStepResult(
            accepted=True,
            time_index=time_index,
            log_likelihood_increment=log_likelihood_increment,
            predicted_mean=predicted_mean,
            predicted_covariance=predicted_covariance,
            predicted_factor=predicted_factor,
            observation_mean=observation_mean,
            innovation_covariance=innovation_covariance,
            innovation_factor=innovation_factor,
            cross_covariance=cross_covariance,
            innovation=innovation,
            innovation_solve=innovation_solve,
            gain=gain,
            filtered_mean=filtered_mean,
            filtered_covariance=filtered_covariance,
            branch_identity=branch_identity,
            diagnostics={
                "branch_hash": branch_identity.hash.value,
                "previous_covariance_diagnostics": previous_diag,
                "predictive_covariance_diagnostics": predictive_diag,
                "innovation_covariance_diagnostics": innovation_diag,
                "cloud_point_count": tf.convert_to_tensor(cloud.point_count, dtype=tf.int32),
                "weight_total": tf.convert_to_tensor(cloud.weight_total, dtype=tf.float64),
                "negative_weight_count": tf.convert_to_tensor(cloud.negative_weight_count, dtype=tf.int32),
            },
        )
        step_results.append(step)
        log_likelihood = log_likelihood + log_likelihood_increment
        accepted_steps += 1
        mean = filtered_mean
        covariance = filtered_covariance
        if return_filtered:
            filtered_means.append(filtered_mean)
            filtered_covariances.append(filtered_covariance)

    return TFFixedSGQFValueResult(
        log_likelihood=log_likelihood,
        filtered_means=tf.stack(filtered_means, axis=0) if return_filtered else None,
        filtered_covariances=tf.stack(filtered_covariances, axis=0) if return_filtered else None,
        step_results=tuple(step_results),
        branch_identity=branch_identity,
        diagnostics=_common_result_diagnostics(
            branch_identity=branch_identity,
            branch_config=branch_config,
            cloud=cloud,
            accepted_steps=accepted_steps,
            failure=None,
        ),
        failure=None,
    )


def tf_fixed_sgqf_p47_one_step_oracle() -> tuple[TFFixedSGQFOneStepOracle, TFFixedSGQFCloud]:
    """Return the p47 one-step scalar nonlinear oracle and its cloud."""

    oracle = TFFixedSGQFOneStepOracle(
        beta=tf.constant(1.0, dtype=tf.float64),
        initial_mean=tf.constant([0.5], dtype=tf.float64),
        initial_covariance=tf.constant([[1.0]], dtype=tf.float64),
        process_covariance=tf.constant([[0.25]], dtype=tf.float64),
        observation_covariance=tf.constant([[1.0]], dtype=tf.float64),
        observation=tf.constant([[2.0]], dtype=tf.float64),
    )
    return oracle, tf_fixed_sgqf_cloud(dim=1, sparse_level=2)
