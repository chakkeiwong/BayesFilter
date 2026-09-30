"""Schema-v2 forward candidate artifacts for factory-bound Contract E routes."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from bayesfilter.highdim.ledh_contract_e_identity import (
    CONTRACT_E_PHASE2_CANDIDATE_STATUS,
    _FactoryIssuedRouteIdentity,
    _require_factory_identity,
    _require_production_factory_identity,
)


LEDH_FORWARD_SCALAR_ARTIFACT_V2_SCHEMA_VERSION = (
    "bayesfilter.highdim.ledh_forward_scalar_artifact.v2"
)
LEDH_FORWARD_V2_PHASE2_ADMISSION_BLOCKED = (
    "LEDH forward artifact v2 is a Phase 2 identity candidate; production "
    "implementation and numerical/scientific admission gates have not passed"
)
LEDH_FORWARD_V2_NONCLAIMS = (
    "not admitted",
    "not production implementation correctness",
    "not numerical adequacy",
    "not same-scalar finite-difference evidence",
    "not Kalman equivalence",
    "not leaderboard, default, or HMC eligible",
)
_LEDH_FORWARD_V2_EVIDENCE_GATE_ITEMS = (
    ("lgssm_oracle_equivalence", "blocked_phase8_not_run"),
    ("nonlinear_validation", "blocked_phase9_not_run"),
    ("production_reset_implementation", "blocked_phase3_not_run"),
    ("same_scalar_value_gradient_graph", "blocked_phase5_not_run"),
    ("streaming_composition_and_feasibility", "blocked_phase4_not_run"),
)


def ledh_forward_v2_evidence_gates() -> dict[str, str]:
    """Return a fresh serialization copy of the immutable Phase 2 blockers."""

    return dict(_LEDH_FORWARD_V2_EVIDENCE_GATE_ITEMS)

_FORWARD_KEYS = frozenset(
    {
        "schema_version",
        "route_identity",
        "route_identity_sha256",
        "row_id",
        "target_scalar",
        "target_output_tensor_field",
        "theta_coordinate_system",
        "parameter_names",
        "theta_values",
        "batch_seeds",
        "time_steps",
        "num_particles",
        "log_likelihood_by_seed",
        "average_log_likelihood_by_seed",
        "finite_output",
        "evidence_gates",
        "admission_status",
        "admitted",
        "canonical_admission_eligible",
        "nonclaims",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def ledh_forward_v2_artifact_sha256(artifact: Mapping[str, Any]) -> str:
    if not isinstance(artifact, Mapping):
        raise TypeError("forward artifact must be a mapping")
    return hashlib.sha256(_canonical_bytes(dict(artifact))).hexdigest()


def _require_exact_keys(payload: Mapping[str, Any]) -> None:
    missing = sorted(_FORWARD_KEYS - set(payload))
    extra = sorted(set(payload) - _FORWARD_KEYS)
    if missing or extra:
        raise ValueError(f"forward v2 field mismatch: missing={missing}, extra={extra}")


def _finite_float_list(name: str, value: Any, *, allow_empty: bool = False) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a numeric sequence")
    output = [float(item) for item in value]
    if not allow_empty and not output:
        raise ValueError(f"{name} must be nonempty")
    if any(not math.isfinite(item) for item in output):
        raise ValueError(f"{name} must contain only finite values")
    return output


def _int_list(name: str, value: Any) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an integer sequence")
    output: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{name} must contain only integers")
        output.append(item)
    if not output:
        raise ValueError(f"{name} must be nonempty")
    return output


def validate_ledh_forward_scalar_artifact_v2(
    artifact: Mapping[str, Any],
    *,
    expected_route_identity: _FactoryIssuedRouteIdentity,
    expected_row_id: str | None = None,
    require_admitted: bool = False,
) -> dict[str, Any]:
    """Validate an unadmitted v2 artifact against an independent factory product."""

    identity = _require_factory_identity(expected_route_identity)
    if not isinstance(artifact, Mapping):
        raise TypeError("forward artifact must be a mapping")
    payload = dict(artifact)
    _require_exact_keys(payload)
    if payload["schema_version"] != LEDH_FORWARD_SCALAR_ARTIFACT_V2_SCHEMA_VERSION:
        raise ValueError("invalid LEDH forward v2 schema_version")
    expected_identity = identity.to_dict()
    if payload["route_identity"] != expected_identity:
        raise ValueError("forward v2 route identity does not match the expected factory product")
    if payload["route_identity_sha256"] != identity.identity_sha256:
        raise ValueError("forward v2 route identity digest mismatch")
    semantic_fields = {
        "row_id": identity.row_id,
        "target_scalar": identity.target_scalar,
        "target_output_tensor_field": identity.target_output_tensor_field,
        "theta_coordinate_system": identity.theta_coordinate_system,
        "parameter_names": list(identity.parameter_names),
    }
    for name, expected in semantic_fields.items():
        if payload[name] != expected:
            raise ValueError(f"forward v2 {name} must be issued by the route specification")
    if expected_row_id is not None and identity.row_id != expected_row_id:
        raise ValueError(f"forward v2 row_id must match expected row {expected_row_id}")
    theta_values = _finite_float_list("theta_values", payload["theta_values"])
    if len(theta_values) != len(identity.parameter_names):
        raise ValueError("theta_values length must match factory parameter ordering")
    batch_seeds = _int_list("batch_seeds", payload["batch_seeds"])
    time_steps = payload["time_steps"]
    num_particles = payload["num_particles"]
    if isinstance(time_steps, bool) or not isinstance(time_steps, int) or time_steps <= 0:
        raise ValueError("time_steps must be a positive integer")
    if (
        isinstance(num_particles, bool)
        or not isinstance(num_particles, int)
        or num_particles <= 0
    ):
        raise ValueError("num_particles must be a positive integer")
    values = _finite_float_list(
        "log_likelihood_by_seed", payload["log_likelihood_by_seed"]
    )
    averages = _finite_float_list(
        "average_log_likelihood_by_seed",
        payload["average_log_likelihood_by_seed"],
    )
    if len(values) != len(batch_seeds) or len(averages) != len(batch_seeds):
        raise ValueError("forward v2 value arrays must match batch_seeds")
    if payload["finite_output"] is not True:
        raise ValueError("forward v2 finite_output must be true")
    if payload["evidence_gates"] != ledh_forward_v2_evidence_gates():
        raise ValueError("forward v2 evidence gates must preserve all Phase 2 blockers")
    if payload["admission_status"] != CONTRACT_E_PHASE2_CANDIDATE_STATUS:
        raise ValueError("forward v2 admission_status must remain the Phase 2 candidate status")
    if payload["admitted"] is not False or payload["canonical_admission_eligible"] is not False:
        raise ValueError("forward v2 Phase 2 candidate cannot claim admission eligibility")
    if payload["nonclaims"] != list(LEDH_FORWARD_V2_NONCLAIMS):
        raise ValueError("forward v2 nonclaims must match the Phase 2 boundary")
    if require_admitted:
        raise ValueError(LEDH_FORWARD_V2_PHASE2_ADMISSION_BLOCKED)
    return {
        "schema_version": LEDH_FORWARD_SCALAR_ARTIFACT_V2_SCHEMA_VERSION,
        "route_identity": identity.to_dict(),
        "route_identity_sha256": identity.identity_sha256,
        "row_id": identity.row_id,
        "target_scalar": identity.target_scalar,
        "target_output_tensor_field": identity.target_output_tensor_field,
        "theta_coordinate_system": identity.theta_coordinate_system,
        "parameter_names": list(identity.parameter_names),
        "theta_values": list(theta_values),
        "batch_seeds": list(batch_seeds),
        "time_steps": time_steps,
        "num_particles": num_particles,
        "log_likelihood_by_seed": list(values),
        "average_log_likelihood_by_seed": list(averages),
        "finite_output": True,
        "evidence_gates": ledh_forward_v2_evidence_gates(),
        "admission_status": CONTRACT_E_PHASE2_CANDIDATE_STATUS,
        "admitted": False,
        "canonical_admission_eligible": False,
        "nonclaims": list(LEDH_FORWARD_V2_NONCLAIMS),
    }


def build_ledh_forward_scalar_artifact_v2(
    *,
    route_identity: _FactoryIssuedRouteIdentity,
    theta_values: Sequence[float],
    batch_seeds: Sequence[int],
    time_steps: int,
    num_particles: int,
    log_likelihood_by_seed: Sequence[float],
    average_log_likelihood_by_seed: Sequence[float],
) -> dict[str, Any]:
    """Build a production-factory candidate; test-factory products are rejected."""

    identity = _require_production_factory_identity(route_identity)
    artifact = {
        "schema_version": LEDH_FORWARD_SCALAR_ARTIFACT_V2_SCHEMA_VERSION,
        "route_identity": identity.to_dict(),
        "route_identity_sha256": identity.identity_sha256,
        "row_id": identity.row_id,
        "target_scalar": identity.target_scalar,
        "target_output_tensor_field": identity.target_output_tensor_field,
        "theta_coordinate_system": identity.theta_coordinate_system,
        "parameter_names": list(identity.parameter_names),
        "theta_values": [float(item) for item in theta_values],
        "batch_seeds": list(batch_seeds),
        "time_steps": time_steps,
        "num_particles": num_particles,
        "log_likelihood_by_seed": [float(item) for item in log_likelihood_by_seed],
        "average_log_likelihood_by_seed": [
            float(item) for item in average_log_likelihood_by_seed
        ],
        "finite_output": True,
        "evidence_gates": ledh_forward_v2_evidence_gates(),
        "admission_status": CONTRACT_E_PHASE2_CANDIDATE_STATUS,
        "admitted": False,
        "canonical_admission_eligible": False,
        "nonclaims": list(LEDH_FORWARD_V2_NONCLAIMS),
    }
    return validate_ledh_forward_scalar_artifact_v2(
        artifact,
        expected_route_identity=identity,
        require_admitted=False,
    )
