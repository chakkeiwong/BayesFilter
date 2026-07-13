"""Explicit coordinate and metric contracts for BayesFilter HMC tuning.

The active kernel uses the affine map ``theta = center + A z`` and identity
momentum covariance in ``z``.  Arrays stored by these types are immutable
boundary metadata; TensorFlow methods perform all runtime coordinate maps.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np


def _immutable_array(value: Any, *, name: str, rank: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float).copy()
    if rank is not None and array.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _signature(label: str, payload: Mapping[str, Any], arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(str(label).encode("utf-8"))
    digest.update(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _positive_int(value: Any, *, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer scalar")
    result = int(value)
    minimum = 0 if allow_zero else 1
    if result < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return result


@dataclass(frozen=True)
class PositionCovarianceEstimate:
    """Canonical-position covariance estimate ``Sigma_theta``.

    The estimate is not a momentum mass matrix.  With identity momentum in the
    active whitened coordinates, its inverse is the implied canonical momentum
    covariance.
    """

    center: Any
    covariance: Any
    source_coordinate_signature: str
    estimator_family: str
    state_count: int
    effective_rank: int
    regularization_report: Mapping[str, Any]
    adequacy_report: Mapping[str, Any]
    evidence_role: str = "adaptation_only_position_covariance"

    def __post_init__(self) -> None:
        center = _immutable_array(self.center, name="center", rank=1)
        covariance = _immutable_array(self.covariance, name="covariance", rank=2)
        dimension = int(center.shape[0])
        if dimension <= 0 or covariance.shape != (dimension, dimension):
            raise ValueError("covariance shape must match non-empty center")
        if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("covariance must be symmetric")
        eigenvalues = np.linalg.eigvalsh(covariance)
        if float(np.min(eigenvalues)) <= 0.0:
            raise ValueError("covariance must be positive definite")
        source_signature = str(self.source_coordinate_signature)
        estimator = str(self.estimator_family)
        role = str(self.evidence_role)
        if not source_signature or not estimator or not role:
            raise ValueError("covariance provenance fields must be non-empty")
        state_count = _positive_int(self.state_count, name="state_count")
        effective_rank = _positive_int(self.effective_rank, name="effective_rank")
        if effective_rank > dimension:
            raise ValueError("effective_rank must not exceed covariance dimension")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "source_coordinate_signature", source_signature)
        object.__setattr__(self, "estimator_family", estimator)
        object.__setattr__(self, "state_count", state_count)
        object.__setattr__(self, "effective_rank", effective_rank)
        object.__setattr__(self, "regularization_report", dict(self.regularization_report))
        object.__setattr__(self, "adequacy_report", dict(self.adequacy_report))
        object.__setattr__(self, "evidence_role", role)

    @property
    def dimension(self) -> int:
        return int(self.center.shape[0])

    @property
    def signature(self) -> str:
        return _signature(
            "bayesfilter.position_covariance_estimate.v2",
            {
                "source_coordinate_signature": self.source_coordinate_signature,
                "estimator_family": self.estimator_family,
                "state_count": self.state_count,
                "effective_rank": self.effective_rank,
                "regularization_report": self.regularization_report,
                "adequacy_report": self.adequacy_report,
                "evidence_role": self.evidence_role,
            },
            (self.center, self.covariance),
        )


@dataclass(frozen=True)
class AffineCoordinateTransform:
    """Row-vector affine map ``theta = center + z @ factor.T``."""

    center: Any
    factor: Any
    covariance_signature: str
    coordinate_name: str = "active_whitened_latent"

    def __post_init__(self) -> None:
        center = _immutable_array(self.center, name="center", rank=1)
        factor = _immutable_array(self.factor, name="factor", rank=2)
        dimension = int(center.shape[0])
        if dimension <= 0 or factor.shape != (dimension, dimension):
            raise ValueError("factor shape must match non-empty center")
        if np.linalg.matrix_rank(factor) != dimension:
            raise ValueError("factor must be nonsingular")
        covariance_signature = str(self.covariance_signature)
        coordinate_name = str(self.coordinate_name)
        if not covariance_signature or not coordinate_name:
            raise ValueError("transform signatures must be non-empty")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "covariance_signature", covariance_signature)
        object.__setattr__(self, "coordinate_name", coordinate_name)

    @classmethod
    def from_covariance_estimate(
        cls,
        estimate: PositionCovarianceEstimate,
        *,
        coordinate_name: str = "active_whitened_latent",
    ) -> "AffineCoordinateTransform":
        if not isinstance(estimate, PositionCovarianceEstimate):
            raise TypeError("estimate must be a PositionCovarianceEstimate")
        return cls(
            center=estimate.center,
            factor=np.linalg.cholesky(estimate.covariance),
            covariance_signature=estimate.signature,
            coordinate_name=coordinate_name,
        )

    @property
    def dimension(self) -> int:
        return int(self.center.shape[0])

    @property
    def covariance(self) -> np.ndarray:
        covariance = np.asarray(self.factor @ self.factor.T, dtype=float)
        covariance.setflags(write=False)
        return covariance

    @property
    def signature(self) -> str:
        return _signature(
            "bayesfilter.affine_coordinate_transform.v2",
            {
                "covariance_signature": self.covariance_signature,
                "coordinate_name": self.coordinate_name,
                "orientation": "theta_equals_center_plus_z_right_factor_transpose",
            },
            (self.center, self.factor),
        )

    def latent_to_theta(self, latent: Any) -> Any:
        import tensorflow as tf

        z = tf.convert_to_tensor(latent, dtype=tf.float64)
        self._validate_tensor(z, name="latent")
        center = tf.convert_to_tensor(self.center, dtype=z.dtype)
        factor = tf.convert_to_tensor(self.factor, dtype=z.dtype)
        return center + tf.tensordot(z, factor, axes=[[-1], [1]])

    def theta_to_latent(self, theta: Any) -> Any:
        import tensorflow as tf

        value = tf.convert_to_tensor(theta, dtype=tf.float64)
        self._validate_tensor(value, name="theta")
        centered = value - tf.convert_to_tensor(self.center, dtype=value.dtype)
        factor = tf.convert_to_tensor(self.factor, dtype=value.dtype)
        flat = tf.reshape(centered, [-1, self.dimension])
        solved = tf.linalg.triangular_solve(factor, tf.transpose(flat), lower=True)
        return tf.reshape(tf.transpose(solved), tf.shape(centered))

    def theta_score_to_latent_score(self, theta_score: Any) -> Any:
        import tensorflow as tf

        score = tf.convert_to_tensor(theta_score, dtype=tf.float64)
        self._validate_tensor(score, name="theta_score")
        factor = tf.convert_to_tensor(self.factor, dtype=score.dtype)
        return tf.tensordot(score, factor, axes=[[-1], [0]])

    def _validate_tensor(self, value: Any, *, name: str) -> None:
        rank = value.shape.rank
        if rank is None or rank < 1 or value.shape[-1] != self.dimension:
            raise ValueError(f"{name} must have static trailing transform dimension")


@dataclass(frozen=True)
class MomentumMetric:
    """Explicit momentum covariance and kinetic precision in one coordinate."""

    momentum_covariance: Any
    kinetic_precision: Any
    coordinate_signature: str
    convention: str = "identity_momentum_in_active_whitened_coordinates"

    def __post_init__(self) -> None:
        covariance = _immutable_array(
            self.momentum_covariance, name="momentum_covariance", rank=2
        )
        precision = _immutable_array(
            self.kinetic_precision, name="kinetic_precision", rank=2
        )
        if covariance.shape[0] == 0 or covariance.shape != precision.shape:
            raise ValueError("momentum covariance and precision shapes must match")
        identity = np.eye(covariance.shape[0])
        if not np.allclose(covariance @ precision, identity, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("momentum covariance and kinetic precision must be inverses")
        coordinate_signature = str(self.coordinate_signature)
        convention = str(self.convention)
        if not coordinate_signature or not convention:
            raise ValueError("metric provenance fields must be non-empty")
        if convention == "identity_momentum_in_active_whitened_coordinates":
            if not np.allclose(covariance, identity, rtol=0.0, atol=1.0e-12):
                raise ValueError("active whitening route requires identity momentum covariance")
        object.__setattr__(self, "momentum_covariance", covariance)
        object.__setattr__(self, "kinetic_precision", precision)
        object.__setattr__(self, "coordinate_signature", coordinate_signature)
        object.__setattr__(self, "convention", convention)

    @classmethod
    def identity_for(cls, transform: AffineCoordinateTransform) -> "MomentumMetric":
        identity = np.eye(transform.dimension)
        return cls(identity, identity, transform.signature)

    @property
    def signature(self) -> str:
        return _signature(
            "bayesfilter.momentum_metric.v2",
            {
                "coordinate_signature": self.coordinate_signature,
                "convention": self.convention,
            },
            (self.momentum_covariance, self.kinetic_precision),
        )


@dataclass(frozen=True)
class WarmupTrajectoryPolicy:
    """Bounded deterministic fixed-trajectory policy used during adaptation."""

    num_leapfrog_steps: int
    max_leapfrog_steps: int
    policy_family: str = "deterministic_fixed_l"

    def __post_init__(self) -> None:
        leapfrog = _positive_int(self.num_leapfrog_steps, name="num_leapfrog_steps")
        maximum = _positive_int(self.max_leapfrog_steps, name="max_leapfrog_steps")
        family = str(self.policy_family)
        if leapfrog > maximum:
            raise ValueError("num_leapfrog_steps must not exceed max_leapfrog_steps")
        if family != "deterministic_fixed_l":
            raise ValueError("R0-R8 support only deterministic_fixed_l")
        object.__setattr__(self, "num_leapfrog_steps", leapfrog)
        object.__setattr__(self, "max_leapfrog_steps", maximum)
        object.__setattr__(self, "policy_family", family)

    @property
    def signature(self) -> str:
        return _signature(
            "bayesfilter.warmup_trajectory_policy.v2",
            {
                "num_leapfrog_steps": self.num_leapfrog_steps,
                "max_leapfrog_steps": self.max_leapfrog_steps,
                "policy_family": self.policy_family,
            },
            (),
        )


@dataclass(frozen=True)
class KernelState:
    """Coordinate-complete immutable state for HMC tuning transitions."""

    canonical_theta: Any
    active_latent: Any
    transform: AffineCoordinateTransform
    momentum_metric: MomentumMetric
    epsilon: float | None
    trajectory_policy: WarmupTrajectoryPolicy
    adaptation_generation: int
    seed_lineage: tuple[int, int]
    evidence_status: str
    epsilon_context_signature: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transform, AffineCoordinateTransform):
            raise TypeError("transform must be an AffineCoordinateTransform")
        if not isinstance(self.momentum_metric, MomentumMetric):
            raise TypeError("momentum_metric must be a MomentumMetric")
        if not isinstance(self.trajectory_policy, WarmupTrajectoryPolicy):
            raise TypeError("trajectory_policy must be a WarmupTrajectoryPolicy")
        theta = _immutable_array(self.canonical_theta, name="canonical_theta")
        latent = _immutable_array(self.active_latent, name="active_latent")
        if theta.ndim < 1 or theta.shape != latent.shape:
            raise ValueError("canonical_theta and active_latent shapes must match")
        if theta.shape[-1] != self.transform.dimension:
            raise ValueError("state trailing dimension must match transform")
        if self.momentum_metric.coordinate_signature != self.transform.signature:
            raise ValueError("momentum metric coordinate signature is stale")
        mapped = np.asarray(self.transform.latent_to_theta(latent).numpy(), dtype=float)
        if not np.allclose(mapped, theta, rtol=1.0e-10, atol=1.0e-10):
            raise ValueError("canonical and latent states do not round trip")
        generation = _positive_int(
            self.adaptation_generation, name="adaptation_generation", allow_zero=True
        )
        try:
            raw_seed = tuple(self.seed_lineage)
        except TypeError as exc:
            raise ValueError(
                "seed_lineage must contain exactly two integer scalars"
            ) from exc
        if len(raw_seed) != 2:
            raise ValueError("seed_lineage must contain exactly two integer scalars")
        seed = tuple(
            _positive_int(item, name="seed_lineage item", allow_zero=True)
            for item in raw_seed
        )
        status = str(self.evidence_status)
        if not status:
            raise ValueError("evidence_status must be non-empty")
        epsilon = None if self.epsilon is None else float(self.epsilon)
        expected_context = self.current_epsilon_context_signature
        context = self.epsilon_context_signature
        if epsilon is None:
            if context is not None:
                raise ValueError("epsilon context must be absent when epsilon is absent")
        else:
            if not np.isfinite(epsilon) or epsilon <= 0.0:
                raise ValueError("epsilon must be positive and finite")
            if context != expected_context:
                raise ValueError("epsilon context is stale for coordinate/metric/trajectory")
        object.__setattr__(self, "canonical_theta", theta)
        object.__setattr__(self, "active_latent", latent)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "adaptation_generation", generation)
        object.__setattr__(self, "seed_lineage", seed)
        object.__setattr__(self, "evidence_status", status)

    @property
    def current_epsilon_context_signature(self) -> str:
        return _signature(
            "bayesfilter.kernel_epsilon_context.v2",
            {
                "transform_signature": self.transform.signature,
                "metric_signature": self.momentum_metric.signature,
                "trajectory_signature": self.trajectory_policy.signature,
            },
            (),
        )

    def with_epsilon(self, epsilon: float, *, evidence_status: str) -> "KernelState":
        return replace(
            self,
            epsilon=float(epsilon),
            evidence_status=str(evidence_status),
            epsilon_context_signature=self.current_epsilon_context_signature,
        )

    def invalidate_epsilon(self, *, evidence_status: str) -> "KernelState":
        return replace(
            self,
            epsilon=None,
            evidence_status=str(evidence_status),
            epsilon_context_signature=None,
        )

    def remap(
        self,
        transform: AffineCoordinateTransform,
        *,
        adaptation_generation: int,
        evidence_status: str,
    ) -> "KernelState":
        latent = transform.theta_to_latent(self.canonical_theta)
        return KernelState(
            canonical_theta=self.canonical_theta,
            active_latent=latent.numpy(),
            transform=transform,
            momentum_metric=MomentumMetric.identity_for(transform),
            epsilon=None,
            trajectory_policy=self.trajectory_policy,
            adaptation_generation=adaptation_generation,
            seed_lineage=self.seed_lineage,
            evidence_status=evidence_status,
            epsilon_context_signature=None,
        )


def transform_from_precomputed_mass_artifact(
    artifact: Any,
    *,
    source_coordinate_signature: str,
    estimator_family: str = "legacy_precomputed_position_covariance",
) -> tuple[PositionCovarianceEstimate, AffineCoordinateTransform]:
    """Map the historical covariance artifact into the explicit v2 contract."""

    center = np.asarray(artifact.position, dtype=float)
    covariance = np.asarray(artifact.covariance, dtype=float)
    estimate = PositionCovarianceEstimate(
        center=center,
        covariance=covariance,
        source_coordinate_signature=source_coordinate_signature,
        estimator_family=estimator_family,
        state_count=max(1, int(getattr(artifact, "dimension", center.shape[0]))),
        effective_rank=int(np.linalg.matrix_rank(covariance)),
        regularization_report=dict(getattr(artifact, "regularization_report", {}) or {}),
        adequacy_report={
            "legacy_compatibility_adapter": True,
            "operational_metric_adequacy_not_inferred": True,
        },
        evidence_role="legacy_compatibility_only",
    )
    transform = AffineCoordinateTransform(
        center=center,
        factor=np.asarray(artifact.factor, dtype=float),
        covariance_signature=estimate.signature,
    )
    if not np.allclose(transform.covariance, covariance, rtol=1.0e-10, atol=1.0e-10):
        raise ValueError("legacy factor does not reconstruct covariance")
    return estimate, transform
