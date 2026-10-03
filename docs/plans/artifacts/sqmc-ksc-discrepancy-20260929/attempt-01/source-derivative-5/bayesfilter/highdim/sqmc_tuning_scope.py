"""Repository-issued tuning identities for the SQMC initialization experiment."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SQMCOrderingScope:
    model_id: str
    target_id: str
    arm_id: str
    experiment_scope: str
    horizon: int
    particle_count: int
    state_dimension: int
    parameter_count: int
    dtype: str
    jit_compile: bool
    reset_contract_id: str
    score_route_id: str
    prepared_data_id: str
    calibration_partition_sha256: str
    state_map_id: str
    state_map_location: tuple[float, ...]
    state_map_scale: tuple[float, ...]
    saturation_policy_id: str
    hilbert_implementation_id: str
    hilbert_bits: int
    hilbert_tie_policy_id: str
    point_set_id: str
    point_set_dimension: int
    row_sort_policy_id: str
    endpoint_policy_id: str
    ancestor_cdf_policy_id: str

    def __post_init__(self) -> None:
        if self.experiment_scope not in {"initial_only", "all_innovations"}:
            raise ValueError("invalid SQMC experiment scope")
        if self.horizon < 1 or self.particle_count < 2:
            raise ValueError("invalid SQMC scope size")
        if len(self.state_map_location) != self.state_dimension:
            raise ValueError("state-map location dimension mismatch")
        if len(self.state_map_scale) != self.state_dimension:
            raise ValueError("state-map scale dimension mismatch")
        if any(value <= 0.0 for value in self.state_map_scale):
            raise ValueError("state-map scales must be positive")
        if self.hilbert_bits < 1:
            raise ValueError("Hilbert bit depth must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def scope_sha256(self) -> str:
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SQMCTuningControls:
    epsilon: float
    sinkhorn_steps: int
    balance_steps: int
    ridge: float

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0 or self.sinkhorn_steps < 1:
            raise ValueError("invalid SQMC tuning controls")
        if self.balance_steps < 0 or self.ridge <= 0.0:
            raise ValueError("invalid SQMC tuning controls")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SQMCTuningArtifact:
    scope: SQMCOrderingScope
    controls: SQMCTuningControls
    calibration_metric: float
    candidate_count: int
    calibration_seed_count: int
    _seal: str = field(repr=False, compare=False)

    @property
    def artifact_id(self) -> str:
        payload = {
            "scope_sha256": self.scope.scope_sha256,
            "controls": self.controls.as_dict(),
            "calibration_metric": self.calibration_metric,
            "candidate_count": self.candidate_count,
            "calibration_seed_count": self.calibration_seed_count,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
        return hashlib.sha256(encoded).hexdigest()

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "bayesfilter.sqmc_tuning_artifact.v1",
            "artifact_id": self.artifact_id,
            "scope": self.scope.as_dict(),
            "scope_sha256": self.scope.scope_sha256,
            "controls": self.controls.as_dict(),
            "calibration_metric": self.calibration_metric,
            "candidate_count": self.candidate_count,
            "calibration_seed_count": self.calibration_seed_count,
        }


_ISSUED_SEALS: set[str] = set()


def issue_sqmc_tuning_artifact(
    scope: SQMCOrderingScope,
    controls: SQMCTuningControls,
    *,
    calibration_metric: float,
    candidate_count: int,
    calibration_seed_count: int,
) -> SQMCTuningArtifact:
    seal = secrets.token_hex(24)
    _ISSUED_SEALS.add(seal)
    return SQMCTuningArtifact(
        scope=scope,
        controls=controls,
        calibration_metric=float(calibration_metric),
        candidate_count=int(candidate_count),
        calibration_seed_count=int(calibration_seed_count),
        _seal=seal,
    )


def require_sqmc_tuning_artifact(
    artifact: SQMCTuningArtifact, expected_scope: SQMCOrderingScope
) -> SQMCTuningControls:
    if not isinstance(artifact, SQMCTuningArtifact):
        raise TypeError("SQMC claim execution requires a repository-issued artifact")
    if artifact._seal not in _ISSUED_SEALS:
        raise TypeError("SQMC tuning artifact is caller-stamped or stale")
    if artifact.scope.scope_sha256 != expected_scope.scope_sha256:
        raise ValueError("SQMC tuning artifact scope does not match claim scope")
    return artifact.controls


def scope_from_mapping(values: Mapping[str, Any]) -> SQMCOrderingScope:
    payload = dict(values)
    payload["state_map_location"] = tuple(payload["state_map_location"])
    payload["state_map_scale"] = tuple(payload["state_map_scale"])
    return SQMCOrderingScope(**payload)


__all__ = [
    "SQMCOrderingScope",
    "SQMCTuningArtifact",
    "SQMCTuningControls",
    "issue_sqmc_tuning_artifact",
    "require_sqmc_tuning_artifact",
    "scope_from_mapping",
]
