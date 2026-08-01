"""Strict proposal-only certificate for HMC semantic-identity migration."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bayesfilter.inference.hmc_identity import (
    artifact_file_sha256,
    canonical_artifact_payload_hash,
)
from bayesfilter.inference.hmc_identity_integration import (
    HMC_IDENTITY_INPUT_INTEGRITY_MANIFEST_SCHEMA_V1,
    HMC_IDENTITY_OUTPUT_INTEGRITY_MANIFEST_SCHEMA_V1,
    HMC_IDENTITY_PRIVATE_SIDECAR_SCHEMA_V1,
    HMC_IDENTITY_PUBLIC_VALIDATION_SCHEMA_V1,
    parse_private_identity_sidecar,
    parse_public_validation_record,
    public_record_matches_private_sidecar,
    verify_input_integrity_manifest,
    verify_output_integrity_manifest,
)
from bayesfilter.runtime import atomic_write_json, stable_config_hash


HMC_MIGRATION_CERTIFICATE_SCHEMA_V1 = (
    "bayesfilter.hmc_semantic_identity_migration_certificate.v1"
)
HMC_MIGRATION_CERTIFICATE_PUBLIC_PROPOSAL_SCHEMA_V1 = (
    "bayesfilter.hmc_semantic_identity_migration_certificate_proposal.v1"
)
HMC_MIGRATION_CERTIFICATE_OUTPUT_MANIFEST_SCHEMA_V1 = (
    "bayesfilter.hmc_semantic_identity_migration_certificate_output_manifest.v1"
)
HMC_MIGRATION_COMPARISON_SCHEMA_V1 = (
    "bayesfilter.hmc_semantic_identity_migration_comparison.v1"
)
HMC_MIGRATION_SOURCE_REFERENCE_SCHEMA_V1 = (
    "bayesfilter.hmc_semantic_identity_migration_source_reference.v1"
)

CERTIFICATE_STATUS = "proposal_only_pending_human_approval"
CERTIFICATE_DECISION = "PROPOSE_TYPED_IDENTITY_BASELINE_MIGRATION"
ADOPTION_STATUS = "pending_human_approval"
ACTIVE_GATE_STATUS = "legacy_gate_remains_binding"
PROPOSED_ACTION = (
    "adopt the refreshed typed transition/execution/provenance identity bundle "
    "as a new reviewed baseline only after explicit human approval"
)
_CLASSIFICATIONS = frozenset({"equal", "different", "unsupported", "not_checked"})
_HEX_DIGITS = frozenset("0123456789abcdef")
LEGACY_HASH_KEYS = (
    "fixture",
    "xla_compile",
    "geometry",
    "mass",
    "adapter",
    "selected_step",
    "public_final_kernel",
    "private_loop_final_kernel",
    "selected_trajectory",
)
_TAGGED_LEGACY_HASH_KEYS = frozenset(
    {"fixture", "xla_compile", "geometry", "mass"}
)
SOURCE_REFERENCE_KEYS = (
    "phase7_config",
    "refreshed_kernel",
    "refreshed_private_replay",
    "phase3_sidecar",
    "phase3_input_manifest",
    "phase3_public_record",
    "phase3_output_manifest",
)
SOURCE_SCHEMAS = {
    "phase7_config": "bayesfilter.deterministic_lgssm_hmc_phase7_config.v1",
    "refreshed_kernel": "bayesfilter.deterministic_lgssm_hmc_tuning_kernel.v1",
    "refreshed_private_replay": (
        "bayesfilter.deterministic_lgssm_hmc_private_tuning_replay.v1"
    ),
    "phase3_sidecar": HMC_IDENTITY_PRIVATE_SIDECAR_SCHEMA_V1,
    "phase3_input_manifest": HMC_IDENTITY_INPUT_INTEGRITY_MANIFEST_SCHEMA_V1,
    "phase3_public_record": HMC_IDENTITY_PUBLIC_VALIDATION_SCHEMA_V1,
    "phase3_output_manifest": HMC_IDENTITY_OUTPUT_INTEGRITY_MANIFEST_SCHEMA_V1,
}

CERTIFICATE_NONCLAIMS = (
    "historical typed transition identity unsupported",
    "historical typed execution identity unsupported",
    "not baseline adoption",
    "not active-gate replacement",
    "not Phase 7 readiness or execution",
    "not posterior convergence or recovery evidence",
    "not production, default, GPU, NeuTra, or scientific evidence",
)
PUBLIC_PROPOSAL_NONCLAIMS = (
    "migration proposal only",
    "human baseline-adoption approval required",
    "historical typed identity equality unsupported",
    "legacy gate remains binding",
    "not Phase 7 readiness or execution",
    "not posterior convergence, recovery, production, default, GPU, NeuTra, or scientific evidence",
)

COMPARISON_CONTRACT = (
    (
        "historical_refreshed_fixture_hash",
        "equal",
        "historical and refreshed fixture hashes are identical",
    ),
    (
        "historical_refreshed_xla_compile_hash",
        "equal",
        "historical and refreshed XLA compile-gate hashes are identical",
    ),
    (
        "historical_refreshed_geometry_hash",
        "equal",
        "historical and refreshed geometry hashes are identical",
    ),
    (
        "historical_refreshed_mass_hash",
        "equal",
        "historical and refreshed mass hashes are identical",
    ),
    (
        "historical_refreshed_adapter_signature",
        "equal",
        "historical and refreshed base-adapter signatures are identical",
    ),
    (
        "historical_refreshed_selected_step_hash",
        "equal",
        "historical and refreshed selected-step hashes are identical",
    ),
    (
        "historical_refreshed_public_final_kernel_hash",
        "different",
        "legacy whole-payload public final-kernel hashes differ",
    ),
    (
        "historical_refreshed_private_loop_hash",
        "different",
        "legacy whole-payload private-loop hashes differ",
    ),
    (
        "historical_refreshed_selected_trajectory_hash",
        "different",
        "legacy whole-payload selected-trajectory hashes differ",
    ),
    (
        "historical_refreshed_typed_transition_identity",
        "unsupported",
        "historical private transition-bearing payload is unavailable",
    ),
    (
        "historical_refreshed_typed_execution_identity",
        "unsupported",
        "historical typed execution contract does not exist",
    ),
    (
        "refreshed_internal_candidate_reconstruction",
        "equal",
        "Phase 3 live reconstruction and sidecar/public identities agree",
    ),
    (
        "posterior_convergence_recovery",
        "not_checked",
        "no sampler transition or diagnostic run occurred",
    ),
    (
        "baseline_adoption",
        "not_checked",
        "human baseline-adoption decision is pending",
    ),
)

_CERTIFICATE_FIELDS = (
    "schema",
    "status",
    "decision",
    "adoption_status",
    "active_gate_status",
    "source_references",
    "historical_expected_hashes",
    "refreshed_legacy_hashes",
    "refreshed_typed_identities",
    "evidence_availability",
    "comparisons",
    "proposed_action",
    "human_approval_required",
    "nonclaims",
    "artifact_hash",
)
_PUBLIC_FIELDS = (
    "schema",
    "status",
    "decision",
    "adoption_status",
    "active_gate_status",
    "classifications",
    "protected_certificate_reference",
    "phase3_public_record_reference",
    "proposed_action",
    "human_approval_required",
    "redaction",
    "nonclaims",
    "artifact_hash",
)


def _require_exact_keys(
    payload: Mapping[str, Any],
    *,
    required: Sequence[str],
    label: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{label} keys must be strings")
    expected = frozenset(required)
    observed = frozenset(payload)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"{label} fields mismatch: missing={missing}, extra={extra}")


def _require_nonblank(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonblank trimmed string")
    return value


def _require_sha256(value: Any, *, label: str, tagged: bool) -> str:
    text = _require_nonblank(value, label=label)
    digest = text.removeprefix("sha256:") if tagged else text
    if tagged != text.startswith("sha256:"):
        raise ValueError(f"{label} SHA-256 prefix mismatch")
    if len(digest) != 64 or any(char not in _HEX_DIGITS for char in digest):
        raise ValueError(f"{label} must be a complete lowercase SHA-256")
    return text


def _require_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _require_ordered(value: Any, *, expected: Sequence[str], label: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    if tuple(value) != tuple(expected):
        raise ValueError(f"{label} must match the closed ordered contract")


def _embed_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "artifact_hash" in result:
        raise ValueError("artifact_hash must not be prepopulated")
    result["artifact_hash"] = canonical_artifact_payload_hash(result)
    return result


def _verify_hash(payload: Mapping[str, Any], *, label: str) -> str:
    observed = _require_sha256(
        payload.get("artifact_hash"),
        label=f"{label} artifact_hash",
        tagged=True,
    )
    expected = canonical_artifact_payload_hash(
        {key: value for key, value in payload.items() if key != "artifact_hash"}
    )
    if observed != expected:
        raise ValueError(f"{label} embedded artifact hash mismatch")
    return observed


def _verify_source_embedded_hash(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> str:
    observed = _require_sha256(
        payload.get("artifact_hash"),
        label=f"{label} artifact_hash",
        tagged=True,
    )
    expected = "sha256:" + stable_config_hash(
        {key: value for key, value in payload.items() if key != "artifact_hash"}
    )
    if observed != expected:
        raise ValueError(f"{label} embedded artifact hash mismatch")
    return observed


def _source_reference(path: str | Path, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    source = Path(path)
    loaded = _read_json_object(source)
    if canonical_artifact_payload_hash(loaded) != canonical_artifact_payload_hash(
        payload
    ):
        raise ValueError(f"source path does not contain supplied payload: {source.name}")
    return {
        "schema": HMC_MIGRATION_SOURCE_REFERENCE_SCHEMA_V1,
        "source_schema": payload.get("schema"),
        "embedded_artifact_hash": payload.get("artifact_hash"),
        "canonical_payload_hash": canonical_artifact_payload_hash(payload),
        "file_sha256": artifact_file_sha256(source),
        "byte_count": source.stat().st_size,
    }


def _parse_source_reference(payload: Mapping[str, Any], *, label: str) -> None:
    _require_exact_keys(
        payload,
        required=(
            "schema",
            "source_schema",
            "embedded_artifact_hash",
            "canonical_payload_hash",
            "file_sha256",
            "byte_count",
        ),
        label=label,
    )
    if payload["schema"] != HMC_MIGRATION_SOURCE_REFERENCE_SCHEMA_V1:
        raise ValueError(f"{label} schema mismatch")
    _require_nonblank(payload["source_schema"], label=f"{label} source_schema")
    embedded = payload["embedded_artifact_hash"]
    if embedded is not None:
        _require_sha256(
            embedded,
            label=f"{label} embedded_artifact_hash",
            tagged=True,
        )
    _require_sha256(
        payload["canonical_payload_hash"],
        label=f"{label} canonical_payload_hash",
        tagged=True,
    )
    _require_sha256(payload["file_sha256"], label=f"{label} file_sha256", tagged=False)
    _require_int(payload["byte_count"], label=f"{label} byte_count")


def _comparison(
    comparison_id: str,
    classification: str,
    basis: str,
    *,
    left_value: str | None,
    right_value: str | None,
) -> Mapping[str, Any]:
    return {
        "schema": HMC_MIGRATION_COMPARISON_SCHEMA_V1,
        "comparison_id": comparison_id,
        "classification": classification,
        "basis": basis,
        "left_value": left_value,
        "right_value": right_value,
    }


def _parse_comparisons(payload: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError("certificate comparisons must be a sequence")
    comparisons = tuple(payload)
    if len(comparisons) != len(COMPARISON_CONTRACT):
        raise ValueError("certificate comparison count mismatch")
    for item, (comparison_id, classification, basis) in zip(
        comparisons,
        COMPARISON_CONTRACT,
        strict=True,
    ):
        _require_exact_keys(
            item,
            required=(
                "schema",
                "comparison_id",
                "classification",
                "basis",
                "left_value",
                "right_value",
            ),
            label="certificate comparison",
        )
        if item["schema"] != HMC_MIGRATION_COMPARISON_SCHEMA_V1:
            raise ValueError("certificate comparison schema mismatch")
        if (
            item["comparison_id"],
            item["classification"],
            item["basis"],
        ) != (comparison_id, classification, basis):
            raise ValueError("certificate comparison contract mismatch")
        if item["classification"] not in _CLASSIFICATIONS:
            raise ValueError("unsupported certificate classification")
        for name in ("left_value", "right_value"):
            value = item[name]
            if value is not None:
                _require_nonblank(value, label=f"comparison {name}")
        if classification in {"unsupported", "not_checked"} and (
            item["left_value"] is not None or item["right_value"] is not None
        ):
            raise ValueError("unsupported/not-checked comparisons cannot assert values")
        if classification in {"equal", "different"} and (
            item["left_value"] is None or item["right_value"] is None
        ):
            raise ValueError("equal/different comparisons require both values")
        if classification == "equal" and (
            item["left_value"] != item["right_value"]
        ):
            raise ValueError("equal comparison values differ")
        if classification == "different" and (
            item["left_value"] == item["right_value"]
        ):
            raise ValueError("different comparison values are equal")
    return comparisons


def build_migration_certificate(
    *,
    phase7_config_payload: Mapping[str, Any],
    refreshed_kernel_payload: Mapping[str, Any],
    refreshed_private_replay_payload: Mapping[str, Any],
    phase3_sidecar: Mapping[str, Any],
    phase3_input_manifest: Mapping[str, Any],
    phase3_public_record: Mapping[str, Any],
    phase3_output_manifest: Mapping[str, Any],
    phase3_paths: Mapping[str, str | Path],
    source_paths: Mapping[str, str | Path],
) -> Mapping[str, Any]:
    """Build a proposal-only certificate from already-validated source evidence."""

    _require_exact_keys(
        phase3_paths,
        required=("sidecar", "input_manifest", "public_record", "output_manifest"),
        label="Phase 3 paths",
    )
    _require_exact_keys(
        source_paths,
        required=("phase7_config", "refreshed_kernel", "refreshed_private_replay"),
        label="certificate source paths",
    )
    parse_private_identity_sidecar(phase3_sidecar)
    verify_input_integrity_manifest(phase3_input_manifest)
    parse_public_validation_record(phase3_public_record)
    verify_output_integrity_manifest(
        phase3_output_manifest,
        sidecar_path=phase3_paths["sidecar"],
        input_manifest_path=phase3_paths["input_manifest"],
        public_record_path=phase3_paths["public_record"],
    )
    if not public_record_matches_private_sidecar(
        public_record=phase3_public_record,
        sidecar_payload=phase3_sidecar,
        sidecar_path=phase3_paths["sidecar"],
        input_integrity_manifest=phase3_input_manifest,
    ):
        raise ValueError("Phase 3 public/private certificate inputs do not cross-link")
    for name, source in (
        ("phase7_config", phase7_config_payload),
        ("refreshed_kernel", refreshed_kernel_payload),
        ("refreshed_private_replay", refreshed_private_replay_payload),
    ):
        if source.get("schema") != SOURCE_SCHEMAS[name]:
            raise ValueError(f"certificate source schema mismatch: {name}")
    _verify_source_embedded_hash(refreshed_kernel_payload, label="refreshed kernel")
    _verify_source_embedded_hash(
        refreshed_private_replay_payload,
        label="refreshed private replay",
    )
    expected = phase7_config_payload.get("expected_hashes")
    refreshed_final = refreshed_kernel_payload.get("final_kernel_payload")
    if not isinstance(expected, Mapping) or not isinstance(refreshed_final, Mapping):
        raise ValueError("certificate source hash mappings are missing")
    _require_exact_keys(
        expected,
        required=LEGACY_HASH_KEYS,
        label="historical expected hashes",
    )
    refreshed = {
        "fixture": refreshed_private_replay_payload.get("fixture_hash"),
        "xla_compile": refreshed_private_replay_payload.get("xla_compile_hash"),
        "geometry": refreshed_private_replay_payload.get("geometry_hash"),
        "mass": refreshed_private_replay_payload.get("mass_hash"),
        "adapter": refreshed_private_replay_payload.get("adapter_signature"),
        "selected_step": refreshed_private_replay_payload.get("selected_step_hash"),
        "public_final_kernel": refreshed_kernel_payload.get("final_kernel_hash"),
        "private_loop_final_kernel": refreshed_private_replay_payload.get(
            "private_loop_final_kernel_hash"
        ),
        "selected_trajectory": refreshed_private_replay_payload.get(
            "selected_trajectory_hash"
        ),
    }
    historical = {name: expected.get(name) for name in LEGACY_HASH_KEYS}
    for label, values in (("historical", historical), ("refreshed", refreshed)):
        for name, value in values.items():
            _require_sha256(
                value,
                label=f"{label} {name}",
                tagged=name in _TAGGED_LEGACY_HASH_KEYS,
            )
    for name in (
        "fixture",
        "xla_compile",
        "geometry",
        "mass",
        "adapter",
        "selected_step",
    ):
        if historical[name] != refreshed[name]:
            raise ValueError(f"{name} comparison no longer satisfies equal contract")
    for name in (
        "public_final_kernel",
        "private_loop_final_kernel",
        "selected_trajectory",
    ):
        if historical[name] == refreshed[name]:
            raise ValueError(f"{name} comparison no longer satisfies different contract")
    if not all(phase3_public_record["candidate_checks"].values()):
        raise ValueError("Phase 3 candidate reconstruction did not pass")
    refreshed_final_values = {
        "public_final_kernel": refreshed_kernel_payload.get("final_kernel_hash"),
        "private_loop_final_kernel": refreshed_final.get("phase7_final_kernel_hash"),
        "selected_step": refreshed_final.get("selected_step_hash"),
        "selected_trajectory": refreshed_final.get("selected_trajectory_hash"),
        "adapter": refreshed_kernel_payload.get("adapter_signature"),
    }
    for name, value in refreshed_final_values.items():
        if value != refreshed[name]:
            raise ValueError(f"kernel/private replay source mismatch: {name}")
    if refreshed_private_replay_payload.get("public_final_kernel_hash") != refreshed[
        "public_final_kernel"
    ]:
        raise ValueError("kernel/private replay source mismatch: public_final_kernel")

    replay_path = Path(source_paths["refreshed_private_replay"])
    replay_integrity = phase3_sidecar["legacy_private_replay_integrity"]
    replay_reference = refreshed_kernel_payload.get("private_replay_reference")
    if not isinstance(replay_reference, Mapping):
        raise ValueError("refreshed kernel private replay reference is missing")
    expected_replay_reference = {
        "artifact_hash": refreshed_private_replay_payload["artifact_hash"],
        "file_sha256": artifact_file_sha256(replay_path),
        "byte_count": replay_path.stat().st_size,
        "public_final_kernel_hash": refreshed["public_final_kernel"],
        "private_loop_final_kernel_hash": refreshed["private_loop_final_kernel"],
    }
    if any(
        replay_reference.get(name) != value
        for name, value in expected_replay_reference.items()
    ):
        raise ValueError("refreshed kernel private replay reference mismatch")
    if replay_integrity != {
        "schema": replay_integrity["schema"],
        "embedded_artifact_hash": refreshed_private_replay_payload["artifact_hash"],
        "canonical_payload_hash": canonical_artifact_payload_hash(
            refreshed_private_replay_payload
        ),
        "file_sha256": artifact_file_sha256(replay_path),
        "byte_count": replay_path.stat().st_size,
    }:
        raise ValueError("Phase 3 sidecar does not bind the refreshed private replay")
    tuning_payload = refreshed_private_replay_payload.get("tuning_payload")
    if not isinstance(tuning_payload, Mapping) or canonical_artifact_payload_hash(
        tuning_payload
    ) != phase3_sidecar["complete_tuning_payload_hash"]:
        raise ValueError("Phase 3 sidecar does not bind the complete tuning payload")
    if (
        phase3_sidecar["selection_provenance"]["selected_step_hash"]
        != refreshed["selected_step"]
        or phase3_sidecar["selection_provenance"]["selected_trajectory_hash"]
        != refreshed["selected_trajectory"]
        or phase3_sidecar["selection_provenance"]["tuning_config_hash"]
        != refreshed_kernel_payload.get("config_hash")
        or refreshed_kernel_payload.get("config_hash")
        != refreshed_private_replay_payload.get("config_hash")
        or refreshed_kernel_payload.get("config_hash")
        != phase7_config_payload.get("source_tuning_config_hash")
    ):
        raise ValueError("selection or tuning-config source cross-link mismatch")

    legacy_comparison_keys = LEGACY_HASH_KEYS
    comparisons = (
        *(
            _comparison(
                *contract,
                left_value=historical[name],
                right_value=refreshed[name],
            )
            for contract, name in zip(
                COMPARISON_CONTRACT[: len(legacy_comparison_keys)],
                legacy_comparison_keys,
                strict=True,
            )
        ),
        *(
            _comparison(*contract, left_value=None, right_value=None)
            for contract in COMPARISON_CONTRACT[
                len(legacy_comparison_keys) : len(legacy_comparison_keys) + 2
            ]
        ),
        _comparison(
            *COMPARISON_CONTRACT[len(legacy_comparison_keys) + 2],
            left_value=phase3_sidecar["transition_identity_hash"],
            right_value=phase3_public_record["transition_identity_hash"],
        ),
        *(
            _comparison(*contract, left_value=None, right_value=None)
            for contract in COMPARISON_CONTRACT[len(legacy_comparison_keys) + 3 :]
        ),
    )
    _parse_comparisons(comparisons)
    all_source_paths = {
        "phase7_config": Path(source_paths["phase7_config"]),
        "refreshed_kernel": Path(source_paths["refreshed_kernel"]),
        "refreshed_private_replay": Path(source_paths["refreshed_private_replay"]),
        "phase3_sidecar": Path(phase3_paths["sidecar"]),
        "phase3_input_manifest": Path(phase3_paths["input_manifest"]),
        "phase3_public_record": Path(phase3_paths["public_record"]),
        "phase3_output_manifest": Path(phase3_paths["output_manifest"]),
    }
    all_source_payloads = {
        "phase7_config": phase7_config_payload,
        "refreshed_kernel": refreshed_kernel_payload,
        "refreshed_private_replay": refreshed_private_replay_payload,
        "phase3_sidecar": phase3_sidecar,
        "phase3_input_manifest": phase3_input_manifest,
        "phase3_public_record": phase3_public_record,
        "phase3_output_manifest": phase3_output_manifest,
    }
    source_references = {
        name: _source_reference(all_source_paths[name], all_source_payloads[name])
        for name in SOURCE_REFERENCE_KEYS
    }
    governed_entries = {
        item["path"]: item for item in phase3_input_manifest["post_snapshot"]
    }
    for name in ("phase7_config", "refreshed_kernel", "refreshed_private_replay"):
        source_path = str(all_source_paths[name].resolve())
        governed = governed_entries.get(source_path)
        reference = source_references[name]
        if governed is None or (
            governed["file_sha256"],
            governed["byte_count"],
        ) != (
            reference["file_sha256"],
            reference["byte_count"],
        ):
            raise ValueError(f"certificate source is not the governed input: {name}")
    payload = {
        "schema": HMC_MIGRATION_CERTIFICATE_SCHEMA_V1,
        "status": CERTIFICATE_STATUS,
        "decision": CERTIFICATE_DECISION,
        "adoption_status": ADOPTION_STATUS,
        "active_gate_status": ACTIVE_GATE_STATUS,
        "source_references": source_references,
        "historical_expected_hashes": historical,
        "refreshed_legacy_hashes": refreshed,
        "refreshed_typed_identities": {
            "transition_identity_hash": phase3_sidecar["transition_identity_hash"],
            "serious_execution_contract_hash": phase3_sidecar[
                "serious_execution_contract_hash"
            ],
            "smoke_execution_contract_hash": phase3_sidecar[
                "smoke_execution_contract_hash"
            ],
            "selection_provenance_hash": phase3_sidecar[
                "selection_provenance_hash"
            ],
            "complete_tuning_payload_hash": phase3_sidecar[
                "complete_tuning_payload_hash"
            ],
            "legacy_replay_canonical_payload_hash": phase3_sidecar[
                "legacy_private_replay_integrity"
            ]["canonical_payload_hash"],
            "legacy_replay_file_sha256": phase3_sidecar[
                "legacy_private_replay_integrity"
            ]["file_sha256"],
        },
        "evidence_availability": {
            "historical_expected_pins_available": True,
            "historical_private_transition_payload_available": False,
            "historical_typed_transition_identity_available": False,
            "historical_typed_execution_identity_available": False,
            "refreshed_private_replay_available": True,
            "refreshed_live_reconstruction_passed": True,
        },
        "comparisons": comparisons,
        "proposed_action": PROPOSED_ACTION,
        "human_approval_required": True,
        "nonclaims": CERTIFICATE_NONCLAIMS,
    }
    return _embed_hash(payload)


def parse_migration_certificate(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(payload, required=_CERTIFICATE_FIELDS, label="migration certificate")
    if payload.get("schema") != HMC_MIGRATION_CERTIFICATE_SCHEMA_V1:
        raise ValueError("unsupported migration certificate schema")
    if (
        payload.get("status"),
        payload.get("decision"),
        payload.get("adoption_status"),
        payload.get("active_gate_status"),
        payload.get("human_approval_required"),
    ) != (
        CERTIFICATE_STATUS,
        CERTIFICATE_DECISION,
        ADOPTION_STATUS,
        ACTIVE_GATE_STATUS,
        True,
    ):
        raise ValueError("migration certificate approval boundary mismatch")
    if _require_bool(
        payload.get("human_approval_required"),
        label="human_approval_required",
    ) is not True:
        raise ValueError("migration certificate requires human approval")
    sources = payload.get("source_references")
    _require_exact_keys(
        sources,
        required=SOURCE_REFERENCE_KEYS,
        label="migration certificate source references",
    )
    for name, source in sources.items():
        _parse_source_reference(source, label=f"certificate source {name}")
        if source["source_schema"] != SOURCE_SCHEMAS[name]:
            raise ValueError(f"certificate source schema mismatch: {name}")
        if (name == "phase7_config") != (source["embedded_artifact_hash"] is None):
            raise ValueError(f"certificate source embedded-hash mismatch: {name}")
    for label in ("historical_expected_hashes", "refreshed_legacy_hashes"):
        values = payload.get(label)
        _require_exact_keys(
            values,
            required=LEGACY_HASH_KEYS,
            label=label,
        )
        for name, value in values.items():
            _require_sha256(
                value,
                label=f"{label}.{name}",
                tagged=name in _TAGGED_LEGACY_HASH_KEYS,
            )
    typed = payload.get("refreshed_typed_identities")
    _require_exact_keys(
        typed,
        required=(
            "transition_identity_hash",
            "serious_execution_contract_hash",
            "smoke_execution_contract_hash",
            "selection_provenance_hash",
            "complete_tuning_payload_hash",
            "legacy_replay_canonical_payload_hash",
            "legacy_replay_file_sha256",
        ),
        label="refreshed typed identities",
    )
    for name, value in typed.items():
        _require_sha256(
            value,
            label=f"refreshed typed identities.{name}",
            tagged=name != "legacy_replay_file_sha256",
        )
    availability = payload.get("evidence_availability")
    _require_exact_keys(
        availability,
        required=(
            "historical_expected_pins_available",
            "historical_private_transition_payload_available",
            "historical_typed_transition_identity_available",
            "historical_typed_execution_identity_available",
            "refreshed_private_replay_available",
            "refreshed_live_reconstruction_passed",
        ),
        label="certificate evidence availability",
    )
    if availability != {
        "historical_expected_pins_available": True,
        "historical_private_transition_payload_available": False,
        "historical_typed_transition_identity_available": False,
        "historical_typed_execution_identity_available": False,
        "refreshed_private_replay_available": True,
        "refreshed_live_reconstruction_passed": True,
    }:
        raise ValueError("certificate evidence availability mismatch")
    comparisons = _parse_comparisons(payload.get("comparisons"))
    historical = payload["historical_expected_hashes"]
    refreshed = payload["refreshed_legacy_hashes"]
    for index, name in enumerate(LEGACY_HASH_KEYS):
        if (
            comparisons[index]["left_value"],
            comparisons[index]["right_value"],
        ) != (historical[name], refreshed[name]):
            raise ValueError("certificate comparison does not match owned hash maps")
    internal = comparisons[len(LEGACY_HASH_KEYS) + 2]
    if internal["left_value"] != typed["transition_identity_hash"]:
        raise ValueError("internal reconstruction comparison identity mismatch")
    if payload.get("proposed_action") != PROPOSED_ACTION:
        raise ValueError("migration certificate proposed action mismatch")
    _require_ordered(
        payload.get("nonclaims"),
        expected=CERTIFICATE_NONCLAIMS,
        label="certificate nonclaims",
    )
    _verify_hash(payload, label="migration certificate")
    return payload


def verify_migration_certificate_sources(
    payload: Mapping[str, Any],
    *,
    source_paths: Mapping[str, str | Path],
) -> Mapping[str, Any]:
    """Reopen all seven sources and reconstruct the proposal-only certificate."""

    parse_migration_certificate(payload)
    _require_exact_keys(
        source_paths,
        required=SOURCE_REFERENCE_KEYS,
        label="migration certificate live source paths",
    )
    paths = {name: Path(source_paths[name]) for name in SOURCE_REFERENCE_KEYS}
    loaded = {name: _read_json_object(paths[name]) for name in SOURCE_REFERENCE_KEYS}
    for name in SOURCE_REFERENCE_KEYS:
        expected_reference = _source_reference(paths[name], loaded[name])
        if payload["source_references"][name] != expected_reference:
            raise ValueError(f"migration certificate source reference mismatch: {name}")

    rebuilt = build_migration_certificate(
        phase7_config_payload=loaded["phase7_config"],
        refreshed_kernel_payload=loaded["refreshed_kernel"],
        refreshed_private_replay_payload=loaded["refreshed_private_replay"],
        phase3_sidecar=loaded["phase3_sidecar"],
        phase3_input_manifest=loaded["phase3_input_manifest"],
        phase3_public_record=loaded["phase3_public_record"],
        phase3_output_manifest=loaded["phase3_output_manifest"],
        phase3_paths={
            "sidecar": paths["phase3_sidecar"],
            "input_manifest": paths["phase3_input_manifest"],
            "public_record": paths["phase3_public_record"],
            "output_manifest": paths["phase3_output_manifest"],
        },
        source_paths={
            "phase7_config": paths["phase7_config"],
            "refreshed_kernel": paths["refreshed_kernel"],
            "refreshed_private_replay": paths["refreshed_private_replay"],
        },
    )
    if json.loads(json.dumps(payload, sort_keys=True)) != json.loads(
        json.dumps(rebuilt, sort_keys=True)
    ):
        raise ValueError("migration certificate does not match current source evidence")
    return payload


_PUBLIC_REDACTION_FIELDS = (
    "observations_publicized",
    "transform_arrays_publicized",
    "hmc_mechanics_publicized",
    "seeds_publicized",
    "runtime_versions_publicized",
    "private_paths_publicized",
    "stage_lineage_publicized",
    "adapter_mass_signatures_publicized",
)


def build_public_certificate_proposal(
    *,
    certificate: Mapping[str, Any],
    certificate_path: str | Path,
    phase3_public_record: Mapping[str, Any],
    phase3_public_record_path: str | Path,
) -> Mapping[str, Any]:
    parse_migration_certificate(certificate)
    parse_public_validation_record(phase3_public_record)
    path = Path(certificate_path)
    public_path = Path(phase3_public_record_path)
    if canonical_artifact_payload_hash(_read_json_object(path)) != (
        canonical_artifact_payload_hash(certificate)
    ):
        raise ValueError("certificate path does not contain supplied certificate")
    if canonical_artifact_payload_hash(_read_json_object(public_path)) != (
        canonical_artifact_payload_hash(phase3_public_record)
    ):
        raise ValueError("Phase 3 public path does not contain supplied record")
    payload = {
        "schema": HMC_MIGRATION_CERTIFICATE_PUBLIC_PROPOSAL_SCHEMA_V1,
        "status": CERTIFICATE_STATUS,
        "decision": CERTIFICATE_DECISION,
        "adoption_status": ADOPTION_STATUS,
        "active_gate_status": ACTIVE_GATE_STATUS,
        "classifications": {
            item["comparison_id"]: item["classification"]
            for item in certificate["comparisons"]
        },
        "protected_certificate_reference": {
            "schema": HMC_MIGRATION_CERTIFICATE_SCHEMA_V1,
            "artifact_hash": certificate["artifact_hash"],
            "file_sha256": artifact_file_sha256(path),
            "byte_count": path.stat().st_size,
        },
        "phase3_public_record_reference": {
            "schema": HMC_IDENTITY_PUBLIC_VALIDATION_SCHEMA_V1,
            "artifact_hash": phase3_public_record["artifact_hash"],
            "file_sha256": artifact_file_sha256(public_path),
            "byte_count": public_path.stat().st_size,
        },
        "proposed_action": certificate["proposed_action"],
        "human_approval_required": True,
        "redaction": {name: False for name in _PUBLIC_REDACTION_FIELDS},
        "nonclaims": PUBLIC_PROPOSAL_NONCLAIMS,
    }
    return _embed_hash(payload)


def parse_public_certificate_proposal(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(payload, required=_PUBLIC_FIELDS, label="public certificate proposal")
    if payload.get("schema") != HMC_MIGRATION_CERTIFICATE_PUBLIC_PROPOSAL_SCHEMA_V1:
        raise ValueError("unsupported public certificate proposal schema")
    if (
        payload.get("status"),
        payload.get("decision"),
        payload.get("adoption_status"),
        payload.get("active_gate_status"),
        payload.get("human_approval_required"),
    ) != (
        CERTIFICATE_STATUS,
        CERTIFICATE_DECISION,
        ADOPTION_STATUS,
        ACTIVE_GATE_STATUS,
        True,
    ):
        raise ValueError("public certificate approval boundary mismatch")
    if _require_bool(
        payload.get("human_approval_required"),
        label="public human_approval_required",
    ) is not True:
        raise ValueError("public certificate requires human approval")
    classifications = payload.get("classifications")
    _require_exact_keys(
        classifications,
        required=tuple(item[0] for item in COMPARISON_CONTRACT),
        label="public certificate classifications",
    )
    if classifications != {item[0]: item[1] for item in COMPARISON_CONTRACT}:
        raise ValueError("public certificate classification mismatch")
    for name, expected_schema in (
        ("protected_certificate_reference", HMC_MIGRATION_CERTIFICATE_SCHEMA_V1),
        ("phase3_public_record_reference", HMC_IDENTITY_PUBLIC_VALIDATION_SCHEMA_V1),
    ):
        reference = payload.get(name)
        _require_exact_keys(
            reference,
            required=("schema", "artifact_hash", "file_sha256", "byte_count"),
            label=name,
        )
        if reference["schema"] != expected_schema:
            raise ValueError(f"{name} schema mismatch")
        _require_sha256(reference["artifact_hash"], label=f"{name} artifact_hash", tagged=True)
        _require_sha256(reference["file_sha256"], label=f"{name} file_sha256", tagged=False)
        _require_int(reference["byte_count"], label=f"{name} byte_count")
    if payload.get("proposed_action") != PROPOSED_ACTION:
        raise ValueError("public certificate proposed action mismatch")
    redaction = payload.get("redaction")
    _require_exact_keys(redaction, required=_PUBLIC_REDACTION_FIELDS, label="public redaction")
    if any(value is not False for value in redaction.values()):
        raise ValueError("public redaction attestations must all be false")
    _require_ordered(
        payload.get("nonclaims"),
        expected=PUBLIC_PROPOSAL_NONCLAIMS,
        label="public proposal nonclaims",
    )
    assert_public_certificate_redacted(payload)
    _verify_hash(payload, label="public certificate proposal")
    return payload


def assert_public_certificate_redacted(payload: Mapping[str, Any]) -> None:
    forbidden_keys = {
        "observations",
        "transforms",
        "step_size",
        "num_leapfrog_steps",
        "root_seed",
        "tensorflow_version",
        "tfp_version",
        "python_version",
        "stage_lineage",
        "reconstruction_links",
        "historical_expected_hashes",
        "refreshed_legacy_hashes",
        "refreshed_typed_identities",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            overlap = forbidden_keys.intersection(value)
            if overlap:
                raise ValueError(f"public certificate contains forbidden keys: {sorted(overlap)}")
            for item in value.values():
                walk(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                walk(item)
        elif isinstance(value, str) and value.startswith("/"):
            raise ValueError("public certificate contains an absolute private path")

    walk(payload)


def build_certificate_output_manifest(
    *,
    certificate_path: str | Path,
    public_proposal_path: str | Path,
) -> Mapping[str, Any]:
    paths = (
        ("protected_certificate", Path(certificate_path)),
        ("public_proposal", Path(public_proposal_path)),
    )
    outputs = []
    loaded: dict[str, Mapping[str, Any]] = {}
    for role, path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if role == "protected_certificate":
            parse_migration_certificate(payload)
        else:
            parse_public_certificate_proposal(payload)
        loaded[role] = payload
        serialized = json.dumps(payload, sort_keys=True)
        if HMC_MIGRATION_CERTIFICATE_OUTPUT_MANIFEST_SCHEMA_V1 in serialized:
            raise ValueError("certificate output references terminal manifest")
        outputs.append(
            {
                "role": role,
                "schema": payload["schema"],
                "artifact_hash": payload["artifact_hash"],
                "file_sha256": artifact_file_sha256(path),
                "byte_count": path.stat().st_size,
            }
        )
    reference = loaded["public_proposal"]["protected_certificate_reference"]
    certificate = loaded["protected_certificate"]
    path = Path(certificate_path)
    if reference != {
        "schema": certificate["schema"],
        "artifact_hash": certificate["artifact_hash"],
        "file_sha256": artifact_file_sha256(path),
        "byte_count": path.stat().st_size,
    }:
        raise ValueError("public proposal does not reference the protected certificate")
    phase3_reference = loaded["public_proposal"]["phase3_public_record_reference"]
    source = certificate["source_references"]["phase3_public_record"]
    if phase3_reference != {
        "schema": source["source_schema"],
        "artifact_hash": source["embedded_artifact_hash"],
        "file_sha256": source["file_sha256"],
        "byte_count": source["byte_count"],
    }:
        raise ValueError("public proposal does not reference Phase 3 public evidence")
    return _embed_hash(
        {
            "schema": HMC_MIGRATION_CERTIFICATE_OUTPUT_MANIFEST_SCHEMA_V1,
            "terminal_manifest": True,
            "outputs": tuple(outputs),
        }
    )


def parse_certificate_output_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(
        payload,
        required=("schema", "terminal_manifest", "outputs", "artifact_hash"),
        label="certificate output manifest",
    )
    if payload.get("schema") != HMC_MIGRATION_CERTIFICATE_OUTPUT_MANIFEST_SCHEMA_V1:
        raise ValueError("unsupported certificate output manifest schema")
    if payload.get("terminal_manifest") is not True:
        raise ValueError("certificate output manifest must be terminal")
    outputs = payload.get("outputs")
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise ValueError("certificate output entries must be a sequence")
    expected = (
        ("protected_certificate", HMC_MIGRATION_CERTIFICATE_SCHEMA_V1),
        ("public_proposal", HMC_MIGRATION_CERTIFICATE_PUBLIC_PROPOSAL_SCHEMA_V1),
    )
    if tuple((item.get("role"), item.get("schema")) for item in outputs) != expected:
        raise ValueError("certificate output role/schema mismatch")
    for item in outputs:
        _require_exact_keys(
            item,
            required=("role", "schema", "artifact_hash", "file_sha256", "byte_count"),
            label="certificate output entry",
        )
        _require_sha256(item["artifact_hash"], label="output artifact_hash", tagged=True)
        _require_sha256(item["file_sha256"], label="output file_sha256", tagged=False)
        _require_int(item["byte_count"], label="output byte_count")
    _verify_hash(payload, label="certificate output manifest")
    return payload


def verify_certificate_output_manifest(
    payload: Mapping[str, Any],
    *,
    certificate_path: str | Path,
    public_proposal_path: str | Path,
) -> Mapping[str, Any]:
    parse_certificate_output_manifest(payload)
    expected = build_certificate_output_manifest(
        certificate_path=certificate_path,
        public_proposal_path=public_proposal_path,
    )
    if json.loads(json.dumps(payload, sort_keys=True)) != json.loads(
        json.dumps(expected, sort_keys=True)
    ):
        raise ValueError("certificate output manifest does not match current bytes")
    return payload


def write_migration_certificate(path: str | Path, payload: Mapping[str, Any]) -> None:
    parse_migration_certificate(payload)
    atomic_write_json(path, payload)
    parse_migration_certificate(_read_json_object(Path(path)))


def write_public_certificate_proposal(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    parse_public_certificate_proposal(payload)
    atomic_write_json(path, payload)
    parse_public_certificate_proposal(_read_json_object(Path(path)))


def write_certificate_output_manifest(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    parse_certificate_output_manifest(payload)
    atomic_write_json(path, payload)
    parse_certificate_output_manifest(_read_json_object(Path(path)))


def _read_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path.name}")
    return payload
