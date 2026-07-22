"""Candidate-only positive Cubature/GenUT designs and route identities.

This module deliberately does not select or replace the canonical Contract E
route.  It supplies reusable standardized designs and an immutable scope
identity for later finite value/score adapters.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import tensorflow as tf


CUBATURE_GENUT_CANDIDATE_ROUTE_ID = (
    "cubature_genut_nonfused_positive_ot_row_quotient_candidate_v2"
)
CUBATURE_GENUT_CANDIDATE_FACTORY_ID = (
    "bayesfilter.highdim.cubature_genut_candidate_factory.v1"
)
CUBATURE_GENUT_IDENTITY_SCHEMA = "bayesfilter.highdim.cubature_genut_identity.v1"

# Identity issuance is deliberately process-local.  A digest alone is not an
# issuance authority: callers must receive an object sealed by this module.
_AUTHORIZED_CANDIDATE_SEALS: dict[object, str] = {}
_REPOSITORY_CANDIDATE_SEALS: set[object] = set()

_REPOSITORY_ADAPTER_REGISTRY = {
    "exact_transformed_sv_v1": {
        "callable_dependency_ids": (
            "bayesfilter.highdim.cubature_genut_adapters:exact_transformed_sv_candidate_adapter",
            "bayesfilter.highdim.cubature_genut_filter:finite_value_score",
            "bayesfilter.highdim.ledh_contract_e_reset_tf:_contract_e_chol_cloud_forward_core",
            "bayesfilter.highdim.ledh_contract_e_reset_tf:_contract_e_chol_cloud_jvp_from_forward_core",
        ),
    },
    "predator_prey_additive_gaussian_v1": {
        "callable_dependency_ids": (
            "bayesfilter.highdim.cubature_genut_adapters:predator_prey_candidate_adapter",
            "bayesfilter.highdim.cubature_genut_filter:finite_value_score",
            "bayesfilter.highdim.cubature_genut_candidate:gaussian_genut_design",
            "bayesfilter.highdim.cubature_genut_candidate:replicate_positive_genut",
            "bayesfilter.highdim.ledh_contract_e_reset_tf:_contract_e_chol_cloud_forward_core",
            "bayesfilter.highdim.ledh_contract_e_reset_tf:_contract_e_chol_cloud_jvp_from_forward_core",
        ),
    },
    "chapter18b_structural_shared_primitives_v1": {
        "callable_dependency_ids": (
            "bayesfilter.highdim.cubature_genut_adapters:structural_ukf_candidate_adapter",
            "bayesfilter.highdim.cubature_genut_filter:finite_value_score",
            "bayesfilter.highdim.cubature_genut_candidate:gaussian_genut_design",
            "bayesfilter.highdim.cubature_genut_candidate:replicate_positive_genut",
            "bayesfilter.testing.structural_ukf_neutra_target_design_tf:structural_source_chart_dtype",
            "bayesfilter.testing.structural_ukf_neutra_target_design_tf:structural_transition_value_dtype",
            "bayesfilter.testing.structural_ukf_neutra_target_design_tf:structural_transition_tangent_dtype",
            "bayesfilter.testing.structural_ukf_neutra_target_design_tf:structural_transition_residual_dtype",
            "bayesfilter.testing.structural_ukf_neutra_target_design_tf:structural_observation_log_density_dtype",
            "bayesfilter.testing.structural_ukf_neutra_target_design_tf:structural_observation_log_density_tangent_dtype",
            "bayesfilter.highdim.ledh_contract_e_reset_tf:_contract_e_chol_cloud_forward_core",
            "bayesfilter.highdim.ledh_contract_e_reset_tf:_contract_e_chol_cloud_jvp_from_forward_core",
        ),
    },
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _finite_float(value: float, *, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def cubature_design(
    *, dim: int, num_particles: int, dtype: tf.dtypes.DType = tf.float32
) -> tf.Tensor:
    """Return the replicated spherical-radial third-degree design.

    The population covariance denominator is ``num_particles``.  Replication
    is intentional: it gives an exact equal-weight positive design whenever
    ``num_particles`` is divisible by ``2 * dim``.
    """

    if dim < 1 or num_particles < 1 or num_particles % (2 * dim) != 0:
        raise ValueError("num_particles must be divisible by 2*dim")
    eye = tf.eye(dim, dtype=dtype)
    radius = tf.sqrt(tf.cast(dim, dtype))
    base = radius * tf.concat([eye, -eye], axis=0)
    return tf.repeat(base, repeats=num_particles // (2 * dim), axis=0)


@dataclass(frozen=True)
class GenUTDesign:
    """Weighted standardized GenUT design before an optional replication."""

    points: tf.Tensor = field(repr=False, compare=False)
    weights: tf.Tensor = field(repr=False, compare=False)
    standardized_skewness: tuple[float, ...]
    standardized_kurtosis: tuple[float, ...]
    central_weight: float
    point_count: int

    @property
    def positive(self) -> bool:
        return bool(tf.reduce_all(self.weights >= 0).numpy())


def genut_design(
    *,
    standardized_skewness: Sequence[float],
    standardized_kurtosis: Sequence[float],
    dtype: tf.dtypes.DType = tf.float32,
) -> GenUTDesign:
    """Construct the unconstrained Ebeigbe-style axis GenUT rule.

    The returned rule is usable as a positive OT marginal only when every
    weight is nonnegative.  In particular, a nonnegative central weight is a
    separate requirement from the per-axis feasibility condition.
    """

    if len(standardized_skewness) != len(standardized_kurtosis):
        raise ValueError("skewness and kurtosis dimensions must match")
    dim = len(standardized_skewness)
    if dim < 1:
        raise ValueError("GenUT dimension must be positive")

    points: list[list[float]] = [[0.0] * dim]
    weights: list[float] = []
    for axis, (skewness_raw, kurtosis_raw) in enumerate(
        zip(standardized_skewness, standardized_kurtosis)
    ):
        skewness = _finite_float(skewness_raw, label="standardized skewness")
        kurtosis = _finite_float(kurtosis_raw, label="standardized kurtosis")
        discriminant = 4.0 * kurtosis - 3.0 * skewness * skewness
        if kurtosis <= skewness * skewness or discriminant <= 0.0:
            raise ValueError("GenUT axis moments are not feasible")
        u = (-skewness + math.sqrt(discriminant)) / 2.0
        v = u + skewness
        if u <= 0.0 or v <= 0.0:
            raise ValueError("GenUT axis points must be positive distances")
        b = 1.0 / (u * (u + v))
        c = 1.0 / (v * (u + v))
        negative = [0.0] * dim
        positive = [0.0] * dim
        negative[axis] = -u
        positive[axis] = v
        points.extend([negative, positive])
        weights.extend([b, c])

    central = 1.0 - sum(weights)
    # The Gaussian specialization has an analytically zero central weight;
    # normalize tiny host-side roundoff before constructing the TF tensor.
    if abs(central) <= 1.0e-12:
        central = 0.0
    all_weights = [central, *weights]
    point_tensor = tf.constant(points, dtype=dtype)
    weight_tensor = tf.constant(all_weights, dtype=dtype)
    return GenUTDesign(
        points=point_tensor,
        weights=weight_tensor,
        standardized_skewness=tuple(float(v) for v in standardized_skewness),
        standardized_kurtosis=tuple(float(v) for v in standardized_kurtosis),
        central_weight=central,
        point_count=1 + 2 * dim,
    )


def gaussian_genut_design(
    *, dim: int, dtype: tf.dtypes.DType = tf.float32
) -> GenUTDesign:
    """Return the Gaussian GenUT specialization.

    The rule has one central point and two points per axis.  Its central weight
    is ``1 - dim / 3``, so it reduces to the spherical-radial Cubature measure
    only when ``dim == 3``.
    """

    if dim < 1:
        raise ValueError("GenUT dimension must be positive")
    return genut_design(
        standardized_skewness=(0.0,) * dim,
        standardized_kurtosis=(3.0,) * dim,
        dtype=dtype,
    )


def replicate_positive_genut(
    design: GenUTDesign,
    *,
    num_particles: int,
    tolerance: float = 1.0e-6,
) -> tf.Tensor:
    """Replicate a positive weighted rule into exact equal-weight rows.

    Exact replication is deliberately strict.  If ``N * weight`` is not an
    integer within the declared tolerance, the design cannot be used as the
    positive equal-weight residual cloud required by this candidate route.
    """

    if num_particles < 1 or tolerance < 0.0:
        raise ValueError("invalid replication arguments")
    if not design.positive:
        raise ValueError("signed GenUT weights cannot be used as OT masses")
    counts = tf.cast(tf.round(tf.cast(num_particles, tf.float32) * design.weights), tf.int32)
    residual = tf.abs(
        tf.cast(counts, tf.float32) / tf.cast(num_particles, tf.float32)
        - design.weights
    )
    if bool(tf.reduce_any(residual > tf.cast(tolerance, tf.float32)).numpy()):
        raise ValueError("GenUT weights are not exactly representable at N")
    if int(tf.reduce_sum(counts).numpy()) != num_particles:
        raise ValueError("GenUT replication counts do not sum to N")
    return tf.repeat(design.points, counts, axis=0)


def map_standardized_design(
    design: GenUTDesign, *, mean: tf.Tensor, square_root: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Map standardized points through ``x = mean + C z``."""

    mean = tf.convert_to_tensor(mean, dtype=design.points.dtype)
    square_root = tf.convert_to_tensor(square_root, dtype=design.points.dtype)
    if mean.shape.rank != 1 or square_root.shape.rank != 2:
        raise ValueError("mean and square_root must have ranks one and two")
    points = mean[None, :] + tf.linalg.matmul(
        design.points, square_root, transpose_b=True
    )
    return points, design.weights


@dataclass(frozen=True)
class CandidateRouteScope:
    """All fields that bind a candidate finite computation scope."""

    model_id: str
    target_id: str
    horizon: int
    particle_count: int
    state_dimension: int
    parameter_count: int
    dtype: str
    tf32_enabled: bool
    jit_compile: bool
    design_family: str
    control_family_id: str

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.particle_count < 1:
            raise ValueError("candidate scope sizes must be positive")
        if self.state_dimension < 1 or self.parameter_count < 1:
            raise ValueError("candidate scope dimensions must be positive")
        if self.dtype != "float32" and self.tf32_enabled:
            raise ValueError("TF32 requires float32 candidate arithmetic")
        if self.design_family not in {"cubature", "genut"}:
            raise ValueError("unsupported candidate design family")


@dataclass(frozen=True)
class CandidateRouteIdentity:
    """Factory-issued, immutable candidate route identity."""

    scope: CandidateRouteScope
    prepared_data_id: str
    residual_design_id: str
    controls: tuple[tuple[str, str], ...]
    callable_dependency_ids: tuple[str, ...]
    source_dependency_closure_id: str
    identity_sha256: str = field(repr=False)
    _factory_seal: object = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CUBATURE_GENUT_IDENTITY_SCHEMA,
            "factory_id": CUBATURE_GENUT_CANDIDATE_FACTORY_ID,
            "route_id": CUBATURE_GENUT_CANDIDATE_ROUTE_ID,
            "scope": self.scope.__dict__,
            "prepared_data_id": self.prepared_data_id,
            "residual_design_id": self.residual_design_id,
            "controls": dict(self.controls),
            "callable_dependency_ids": list(self.callable_dependency_ids),
            "source_dependency_closure_id": self.source_dependency_closure_id,
            "identity_sha256": self.identity_sha256,
        }


def issue_candidate_route_identity(
    scope: CandidateRouteScope,
    *,
    prepared_data_id: str,
    residual_design_id: str,
    controls: Mapping[str, str],
    callable_dependency_ids: Sequence[str],
    source_dependency_closure_id: str,
) -> CandidateRouteIdentity:
    """Issue an identity from repository-owned candidate fields."""

    if not prepared_data_id or not residual_design_id:
        raise ValueError("prepared data and residual design identities are required")
    dependencies = tuple(str(item) for item in callable_dependency_ids)
    if not dependencies or any(not item for item in dependencies):
        raise ValueError("callable dependency identities are required")
    if not source_dependency_closure_id:
        raise ValueError("source dependency closure identity is required")
    frozen_controls = tuple(sorted((str(k), str(v)) for k, v in controls.items()))
    payload = {
        "schema": CUBATURE_GENUT_IDENTITY_SCHEMA,
        "factory_id": CUBATURE_GENUT_CANDIDATE_FACTORY_ID,
        "route_id": CUBATURE_GENUT_CANDIDATE_ROUTE_ID,
        "scope": scope.__dict__,
        "prepared_data_id": prepared_data_id,
        "residual_design_id": residual_design_id,
        "controls": dict(frozen_controls),
        "callable_dependency_ids": list(dependencies),
        "source_dependency_closure_id": source_dependency_closure_id,
    }
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    issuance_seal = object()
    _AUTHORIZED_CANDIDATE_SEALS[issuance_seal] = digest
    return CandidateRouteIdentity(
        scope=scope,
        prepared_data_id=prepared_data_id,
        residual_design_id=residual_design_id,
        controls=frozen_controls,
        callable_dependency_ids=dependencies,
        source_dependency_closure_id=source_dependency_closure_id,
        identity_sha256=digest,
        _factory_seal=issuance_seal,
    )


def _resolve_registered_symbol(symbol: str) -> Any:
    module_name, separator, qualname = symbol.partition(":")
    if not separator:
        raise ValueError(f"invalid registered candidate symbol: {symbol!r}")
    value: Any = importlib.import_module(module_name)
    for component in qualname.split("."):
        value = getattr(value, component)
    return value


def _registered_source_closure_digest(symbols: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for symbol in symbols:
        value = _resolve_registered_symbol(symbol)
        source = inspect.getsource(getattr(value, "python_function", value)).encode("utf-8")
        digest.update(symbol.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source)
        digest.update(b"\0")
    return digest.hexdigest()


def issue_repository_candidate_route_identity(
    scope: CandidateRouteScope,
    *,
    prepared_data_id: str,
    residual_design_id: str,
    controls: Mapping[str, str],
    adapter_id: str,
) -> CandidateRouteIdentity:
    """Issue an identity only for a repository-registered candidate adapter."""

    registration = _REPOSITORY_ADAPTER_REGISTRY.get(adapter_id)
    if registration is None:
        raise ValueError(f"candidate adapter is not repository-registered: {adapter_id}")
    symbols = tuple(registration["callable_dependency_ids"])
    closure_digest = _registered_source_closure_digest(symbols)
    identity = issue_candidate_route_identity(
        scope,
        prepared_data_id=prepared_data_id,
        residual_design_id=residual_design_id,
        controls={**controls, "adapter_id": adapter_id},
        callable_dependency_ids=symbols,
        source_dependency_closure_id=closure_digest,
    )
    _REPOSITORY_CANDIDATE_SEALS.add(identity._factory_seal)
    return identity


def validate_repository_candidate_route_identity(
    identity: CandidateRouteIdentity,
) -> None:
    """Require both a valid digest and repository-owned adapter issuance."""

    validate_candidate_route_identity(identity)
    if identity._factory_seal not in _REPOSITORY_CANDIDATE_SEALS:
        raise ValueError("candidate identity was not issued by the repository factory")
    adapter_items = dict(identity.controls)
    adapter_id = adapter_items.get("adapter_id")
    registration = _REPOSITORY_ADAPTER_REGISTRY.get(adapter_id)
    if registration is None:
        raise ValueError("candidate identity has no registered adapter")
    expected_symbols = tuple(registration["callable_dependency_ids"])
    if identity.callable_dependency_ids != expected_symbols:
        raise ValueError("candidate identity callable closure differs from registry")
    if identity.source_dependency_closure_id != _registered_source_closure_digest(expected_symbols):
        raise ValueError("candidate identity source closure differs from registry")


def validate_candidate_route_identity(identity: CandidateRouteIdentity) -> None:
    """Fail closed if an identity's digest or route fields were altered."""

    if not isinstance(identity, CandidateRouteIdentity):
        raise TypeError("candidate identity must be factory-issued")
    registered_digest = _AUTHORIZED_CANDIDATE_SEALS.get(identity._factory_seal)
    if registered_digest is None:
        raise ValueError("candidate route identity has no authorized issuance seal")
    if identity.scope.design_family not in {"cubature", "genut"}:
        raise ValueError("candidate identity has an unsupported design family")
    expected = issue_candidate_route_identity(
        identity.scope,
        prepared_data_id=identity.prepared_data_id,
        residual_design_id=identity.residual_design_id,
        controls=dict(identity.controls),
        callable_dependency_ids=identity.callable_dependency_ids,
        source_dependency_closure_id=identity.source_dependency_closure_id,
    )
    if expected.identity_sha256 != identity.identity_sha256:
        raise ValueError("candidate route identity digest mismatch")
    if registered_digest != identity.identity_sha256:
        raise ValueError("candidate route identity differs from its issuance")


__all__ = [
    "CUBATURE_GENUT_CANDIDATE_FACTORY_ID",
    "CUBATURE_GENUT_CANDIDATE_ROUTE_ID",
    "CandidateRouteIdentity",
    "CandidateRouteScope",
    "GenUTDesign",
    "cubature_design",
    "gaussian_genut_design",
    "genut_design",
    "issue_candidate_route_identity",
    "issue_repository_candidate_route_identity",
    "map_standardized_design",
    "replicate_positive_genut",
    "validate_candidate_route_identity",
    "validate_repository_candidate_route_identity",
]
