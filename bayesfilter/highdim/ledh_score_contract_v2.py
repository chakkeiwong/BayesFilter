"""Schema-v2 score candidate artifacts for factory-bound Contract E routes."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from bayesfilter.highdim.ledh_contract_e_identity import (
    CONTRACT_E_PHASE2_CANDIDATE_STATUS,
    _FactoryIssuedRouteIdentity,
    _require_factory_identity,
    _require_production_factory_identity,
)
from bayesfilter.highdim.ledh_forward_contract_v2 import (
    ledh_forward_v2_evidence_gates,
    ledh_forward_v2_artifact_sha256,
    validate_ledh_forward_scalar_artifact_v2,
)


LEDH_SCORE_ARTIFACT_V2_SCHEMA_VERSION = "bayesfilter.highdim.ledh_score_artifact.v2"
LEDH_SCORE_V2_PHASE2_ADMISSION_BLOCKED = (
    "LEDH score artifact v2 is a Phase 2 identity candidate; same-scalar, "
    "production, numerical, and scientific admission gates have not passed"
)
LEDH_SCORE_V2_NONCLAIMS = (
    "not admitted",
    "not production total-gradient correctness",
    "not same-scalar finite-difference evidence",
    "not Kalman gradient equivalence",
    "not leaderboard, default, or HMC eligible",
)
_LEDH_SCORE_V2_CORRECTNESS_ITEMS = (
    ("kind", "not_evidence_phase2_schema_only"),
    ("status", "blocked_phase5_not_run"),
)


def ledh_score_v2_correctness() -> dict[str, str]:
    """Return a fresh serialization copy of the immutable correctness blocker."""

    return dict(_LEDH_SCORE_V2_CORRECTNESS_ITEMS)

_SCORE_KEYS = frozenset(
    {
        "schema_version",
        "route_identity",
        "route_identity_sha256",
        "source_forward_artifact_sha256",
        "row_id",
        "target_scalar",
        "target_output_tensor_field",
        "theta_coordinate_system",
        "score_parameter_names",
        "score",
        "value_score_route_status",
        "score_correctness",
        "evidence_gates",
        "score_admission_status",
        "admitted",
        "canonical_admission_eligible",
        "nonclaims",
    }
)


def _finite_score(value: Any) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("score must be a numeric sequence")
    output = [float(item) for item in value]
    if not output or any(not math.isfinite(item) for item in output):
        raise ValueError("score must be nonempty and finite")
    return output


def validate_ledh_score_artifact_v2(
    artifact: Mapping[str, Any],
    *,
    source_forward_artifact: Mapping[str, Any],
    expected_route_identity: _FactoryIssuedRouteIdentity,
    expected_row_id: str | None = None,
    require_admitted: bool = False,
) -> dict[str, Any]:
    """Validate an unadmitted score candidate against one exact forward artifact."""

    identity = _require_factory_identity(expected_route_identity)
    validate_ledh_forward_scalar_artifact_v2(
        source_forward_artifact,
        expected_route_identity=identity,
        expected_row_id=expected_row_id,
        require_admitted=False,
    )
    if not isinstance(artifact, Mapping):
        raise TypeError("score artifact must be a mapping")
    payload = dict(artifact)
    missing = sorted(_SCORE_KEYS - set(payload))
    extra = sorted(set(payload) - _SCORE_KEYS)
    if missing or extra:
        raise ValueError(f"score v2 field mismatch: missing={missing}, extra={extra}")
    if payload["schema_version"] != LEDH_SCORE_ARTIFACT_V2_SCHEMA_VERSION:
        raise ValueError("invalid LEDH score v2 schema_version")
    if payload["route_identity"] != identity.to_dict():
        raise ValueError("score v2 route identity does not match the expected factory product")
    if payload["route_identity_sha256"] != identity.identity_sha256:
        raise ValueError("score v2 route identity digest mismatch")
    expected_forward_digest = ledh_forward_v2_artifact_sha256(source_forward_artifact)
    if payload["source_forward_artifact_sha256"] != expected_forward_digest:
        raise ValueError("score v2 does not bind the exact source forward artifact")
    semantic_fields = {
        "row_id": identity.row_id,
        "target_scalar": identity.target_scalar,
        "target_output_tensor_field": identity.target_output_tensor_field,
        "theta_coordinate_system": identity.theta_coordinate_system,
        "score_parameter_names": list(identity.parameter_names),
    }
    for name, expected in semantic_fields.items():
        if payload[name] != expected:
            raise ValueError(f"score v2 {name} must be issued by the route specification")
    if expected_row_id is not None and identity.row_id != expected_row_id:
        raise ValueError(f"score v2 row_id must match expected row {expected_row_id}")
    score = _finite_score(payload["score"])
    if len(score) != len(identity.parameter_names):
        raise ValueError("score length must match factory parameter ordering")
    if payload["value_score_route_status"] != "same_factory_identity_candidate":
        raise ValueError("score v2 must bind the same factory identity as its value")
    if payload["score_correctness"] != ledh_score_v2_correctness():
        raise ValueError("score v2 correctness must remain blocked in Phase 2")
    if payload["evidence_gates"] != ledh_forward_v2_evidence_gates():
        raise ValueError("score v2 evidence gates must preserve all Phase 2 blockers")
    if payload["score_admission_status"] != CONTRACT_E_PHASE2_CANDIDATE_STATUS:
        raise ValueError("score v2 admission status must remain a Phase 2 candidate")
    if payload["admitted"] is not False or payload["canonical_admission_eligible"] is not False:
        raise ValueError("score v2 Phase 2 candidate cannot claim admission eligibility")
    if payload["nonclaims"] != list(LEDH_SCORE_V2_NONCLAIMS):
        raise ValueError("score v2 nonclaims must match the Phase 2 boundary")
    if require_admitted:
        raise ValueError(LEDH_SCORE_V2_PHASE2_ADMISSION_BLOCKED)
    return {
        "schema_version": LEDH_SCORE_ARTIFACT_V2_SCHEMA_VERSION,
        "route_identity": identity.to_dict(),
        "route_identity_sha256": identity.identity_sha256,
        "source_forward_artifact_sha256": expected_forward_digest,
        "row_id": identity.row_id,
        "target_scalar": identity.target_scalar,
        "target_output_tensor_field": identity.target_output_tensor_field,
        "theta_coordinate_system": identity.theta_coordinate_system,
        "score_parameter_names": list(identity.parameter_names),
        "score": list(score),
        "value_score_route_status": "same_factory_identity_candidate",
        "score_correctness": ledh_score_v2_correctness(),
        "evidence_gates": ledh_forward_v2_evidence_gates(),
        "score_admission_status": CONTRACT_E_PHASE2_CANDIDATE_STATUS,
        "admitted": False,
        "canonical_admission_eligible": False,
        "nonclaims": list(LEDH_SCORE_V2_NONCLAIMS),
    }


def build_ledh_score_artifact_v2(
    *,
    route_identity: _FactoryIssuedRouteIdentity,
    source_forward_artifact: Mapping[str, Any],
    score: Sequence[float],
) -> dict[str, Any]:
    """Build a production-factory score candidate; test scopes are rejected."""

    identity = _require_production_factory_identity(route_identity)
    artifact = {
        "schema_version": LEDH_SCORE_ARTIFACT_V2_SCHEMA_VERSION,
        "route_identity": identity.to_dict(),
        "route_identity_sha256": identity.identity_sha256,
        "source_forward_artifact_sha256": ledh_forward_v2_artifact_sha256(
            source_forward_artifact
        ),
        "row_id": identity.row_id,
        "target_scalar": identity.target_scalar,
        "target_output_tensor_field": identity.target_output_tensor_field,
        "theta_coordinate_system": identity.theta_coordinate_system,
        "score_parameter_names": list(identity.parameter_names),
        "score": [float(item) for item in score],
        "value_score_route_status": "same_factory_identity_candidate",
        "score_correctness": ledh_score_v2_correctness(),
        "evidence_gates": ledh_forward_v2_evidence_gates(),
        "score_admission_status": CONTRACT_E_PHASE2_CANDIDATE_STATUS,
        "admitted": False,
        "canonical_admission_eligible": False,
        "nonclaims": list(LEDH_SCORE_V2_NONCLAIMS),
    }
    return validate_ledh_score_artifact_v2(
        artifact,
        source_forward_artifact=source_forward_artifact,
        expected_route_identity=identity,
        require_admitted=False,
    )
