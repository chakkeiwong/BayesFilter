"""Strict Phase 5 materialization for an approved HMC identity baseline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bayesfilter.inference.hmc_identity import (
    artifact_file_sha256,
    canonical_artifact_payload_hash,
)
from bayesfilter.inference.hmc_identity_migration_certificate import (
    CERTIFICATE_DECISION,
    HMC_MIGRATION_CERTIFICATE_OUTPUT_MANIFEST_SCHEMA_V1,
    HMC_MIGRATION_CERTIFICATE_PUBLIC_PROPOSAL_SCHEMA_V1,
    HMC_MIGRATION_CERTIFICATE_SCHEMA_V1,
    LEGACY_HASH_KEYS,
    parse_certificate_output_manifest,
    parse_migration_certificate,
    parse_public_certificate_proposal,
)
from bayesfilter.runtime import atomic_write_json, stable_config_hash


PHASE7_CONFIG_SCHEMA_V1 = "bayesfilter.deterministic_lgssm_hmc_phase7_config.v1"
PHASE7_CONFIG_SCHEMA_V2 = "bayesfilter.deterministic_lgssm_hmc_phase7_config.v2"
HMC_PHASE5_ARTIFACT_REFERENCE_SCHEMA_V1 = (
    "bayesfilter.hmc_identity_phase5_artifact_reference.v1"
)
HMC_PHASE5_ADOPTION_RECORD_SCHEMA_V1 = (
    "bayesfilter.hmc_identity_baseline_adoption_record.v1"
)
HMC_PHASE5_PREFLIGHT_REPORT_SCHEMA_V1 = (
    "bayesfilter.hmc_identity_phase5_preflight_report.v1"
)
HMC_PHASE5_OUTPUT_MANIFEST_SCHEMA_V1 = (
    "bayesfilter.hmc_identity_phase5_output_manifest.v1"
)

APPROVED_CERTIFICATE_ARTIFACT_HASH = (
    "sha256:684c5ae23c48f0d233fb8797927cb1574836ba0c0af3bf362c71b55e1aa1fc7f"
)
HUMAN_APPROVAL_STATEMENT = (
    "I approve PROPOSE_TYPED_IDENTITY_BASELINE_MIGRATION bound to certificate "
    "sha256:684c5ae23c48f0d233fb8797927cb1574836ba0c0af3bf362c71b55e1aa1fc7f."
)
HUMAN_APPROVAL_DATE = "2026-07-11"
V2_CONFIG_ID = "multidim_lgssm_phase7_typed_identity_baseline_2026_07_11"
V2_PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-hmc-semantic-identity-migration-phase5-"
    "adversarial-validation-subplan-2026-07-11.md"
)
V2_SOURCE_TUNING_CONFIG_PATH = (
    "docs/benchmarks/configs/"
    "multidim_lgssm_serious_hmc_tuning_2026_07_09.json"
)
V2_SOURCE_TUNING_CONFIG_HASH = (
    "sha256:1b5683e2f210e3976fca712ec2970f8327831596c0b67776316efbd0b6b46729"
)
V2_ARTIFACT_ROOT = (
    "docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09"
)
ADOPTION_STATUS = "approved_baseline_materialized_runtime_not_authorized"
LEGACY_GATE_STATUS = "historical_audit_only"
PREFLIGHT_DECISION = "PASS_PHASE5_TYPED_BASELINE_PREFLIGHT_NO_RUNTIME"

ADOPTED_IDENTITY_KEYS = (
    "transition_identity_hash",
    "serious_execution_contract_hash",
    "smoke_execution_contract_hash",
    "selection_provenance_hash",
    "complete_tuning_payload_hash",
    "legacy_replay_canonical_payload_hash",
    "legacy_replay_file_sha256",
)
GOVERNED_SOURCE_KEYS = (
    "fixture",
    "xla_compile",
    "geometry",
    "mass",
    "kernel",
    "private_replay",
    "source_tuning_config",
    "historical_v1_config",
    "source_contract",
)
GOVERNED_SOURCE_SCHEMAS = {
    "fixture": "bayesfilter.deterministic_lgssm_hmc_tuning_fixture.v1",
    "xla_compile": "bayesfilter.deterministic_lgssm_hmc_tuning_xla_score_gate.v1",
    "geometry": "bayesfilter.deterministic_lgssm_hmc_tuning_geometry.v1",
    "mass": "bayesfilter.deterministic_lgssm_hmc_tuning_mass.v1",
    "kernel": "bayesfilter.deterministic_lgssm_hmc_tuning_kernel.v1",
    "private_replay": (
        "bayesfilter.deterministic_lgssm_hmc_private_tuning_replay.v1"
    ),
    "source_tuning_config": (
        "bayesfilter.deterministic_lgssm_hmc_tuning_config.v1"
    ),
    "historical_v1_config": PHASE7_CONFIG_SCHEMA_V1,
    "source_contract": "bayesfilter.multidim_triangular_lgssm.contract.v1",
}
GOVERNED_SOURCE_HASH_RULES = {
    name: ("stable_without_hash" if name in {
        "fixture",
        "xla_compile",
        "geometry",
        "mass",
        "kernel",
        "private_replay",
    } else "none")
    for name in GOVERNED_SOURCE_KEYS
}

V2_NONCLAIMS = (
    "approved typed baseline configuration only",
    "historical typed identity equality unsupported",
    "legacy whole-payload hashes retained as historical audit evidence",
    "runtime authority false",
    "not Phase 7 smoke or serious execution",
    "not convergence, recovery, production, default, GPU, NeuTra, or scientific evidence",
)
ADOPTION_NONCLAIMS = (
    "human approval records a new refreshed baseline, not historical equality",
    "runtime authority false",
    "not HMC smoke, burn-in, sampling, Phase 8, or NeuTra authority",
    "not convergence, recovery, production, default, GPU, or scientific evidence",
)
PREFLIGHT_NONCLAIMS = (
    "typed baseline engineering preflight only",
    "no HMC transition or worker executed",
    "legacy whole-payload differences are historical audit evidence",
    "historical typed identity equality unsupported",
    "not convergence, recovery, production, default, GPU, NeuTra, or scientific evidence",
)

_HASH_RULES = frozenset({"none", "stable_without_hash", "canonical_without_hash"})
_HEX = frozenset("0123456789abcdef")
_V2_FIELDS = (
    "schema",
    "config_id",
    "plan_path",
    "source_tuning_config_path",
    "source_tuning_config_hash",
    "artifact_root",
    "execution",
    "burnin",
    "retained",
    "diagnostics",
    "artifacts",
    "baseline_adoption",
    "adopted_identities",
    "governed_source_references",
    "historical_legacy_hashes",
    "legacy_whole_payload_gate_status",
    "runtime_authority",
    "nonclaims",
    "artifact_hash",
)
_V2_SECTION_FIELDS = {
    "execution": (
        "worker_count",
        "chains_per_worker",
        "root_seed",
        "cuda_visible_devices",
        "jit_compile",
        "use_xla",
        "chain_execution_mode",
        "compile_workers_sequentially",
        "wall_time_cap_seconds",
        "thread_environment",
    ),
    "burnin": (
        "initial_results_per_chain",
        "extension_results_per_chain",
        "check_window_results_per_chain",
        "max_results_per_chain",
    ),
    "retained": (
        "initial_results_per_chain",
        "extension_results_per_chain",
        "check_interval_results_per_chain",
        "max_results_per_chain",
    ),
    "diagnostics": (
        "rhat_max",
        "bulk_ess_min",
        "tail_ess_min",
        "all_parameters_required",
        "coordinate_system",
    ),
    "artifacts": (
        "public_result",
        "public_progress",
        "private_replay",
        "private_retained_samples",
    ),
}
_THREAD_ENVIRONMENT_FIELDS = (
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)
_V2_EXPECTED_SECTIONS = {
    "execution": {
        "worker_count": 2,
        "chains_per_worker": 2,
        "root_seed": [20260711, 701],
        "cuda_visible_devices": "-1",
        "jit_compile": True,
        "use_xla": True,
        "chain_execution_mode": "tf_function",
        "compile_workers_sequentially": True,
        "wall_time_cap_seconds": 28800,
        "thread_environment": {
            "TF_NUM_INTRAOP_THREADS": "8",
            "TF_NUM_INTEROP_THREADS": "1",
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    },
    "burnin": {
        "initial_results_per_chain": 2000,
        "extension_results_per_chain": 1000,
        "check_window_results_per_chain": 1000,
        "max_results_per_chain": 16000,
    },
    "retained": {
        "initial_results_per_chain": 4000,
        "extension_results_per_chain": 2000,
        "check_interval_results_per_chain": 2000,
        "max_results_per_chain": 40000,
    },
    "diagnostics": {
        "rhat_max": 1.01,
        "bulk_ess_min": 1000.0,
        "tail_ess_min": 400.0,
        "all_parameters_required": True,
        "coordinate_system": "raw_lgssm_parameters_after_two_mass_transforms",
    },
    "artifacts": {
        "public_result": (
            "docs/benchmarks/artifacts/"
            "multidim_lgssm_serious_hmc_tuning_2026_07_09/burnin_sampling.json"
        ),
        "public_progress": (
            "docs/benchmarks/artifacts/"
            "multidim_lgssm_serious_hmc_tuning_2026_07_09/"
            "burnin_sampling_progress.json"
        ),
        "private_replay": (
            "docs/benchmarks/artifacts/"
            "multidim_lgssm_serious_hmc_tuning_2026_07_09/"
            "private_diagnostics/kernel_tuning_replay.json"
        ),
        "private_retained_samples": (
            "docs/benchmarks/artifacts/"
            "multidim_lgssm_serious_hmc_tuning_2026_07_09/"
            "private_diagnostics/phase7_retained_samples.npz"
        ),
    },
}
_ADOPTION_FIELDS = (
    "schema",
    "status",
    "decision",
    "human_approval_statement",
    "human_approval_date",
    "certificate_reference",
    "public_proposal_reference",
    "phase4_output_manifest_reference",
    "historical_v1_config_reference",
    "v2_config_reference",
    "adopted_identities",
    "legacy_whole_payload_gate_status",
    "runtime_authority",
    "nonclaims",
    "artifact_hash",
)
_PREFLIGHT_FIELDS = (
    "schema",
    "passed",
    "decision",
    "config_reference",
    "adoption_record_reference",
    "identity_hashes",
    "identity_checks",
    "integrity_checks",
    "legacy_audit",
    "private_replay_artifact_hash",
    "private_replay_file_sha256",
    "parameter_names",
    "target_scope",
    "runtime_authority",
    "runtime_executed",
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


def _require_sha256(value: Any, *, label: str, tagged: bool = True) -> str:
    text = _require_nonblank(value, label=label)
    digest = text.removeprefix("sha256:") if tagged else text
    if tagged != text.startswith("sha256:"):
        raise ValueError(f"{label} SHA-256 prefix mismatch")
    if len(digest) != 64 or any(char not in _HEX for char in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
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
    )
    expected = canonical_artifact_payload_hash(
        {key: value for key, value in payload.items() if key != "artifact_hash"}
    )
    if observed != expected:
        raise ValueError(f"{label} embedded artifact hash mismatch")
    return observed


def _read_json(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {source.name}")
    return payload


def build_phase5_artifact_reference(
    path: str | Path,
    *,
    embedded_hash_rule: str,
) -> Mapping[str, Any]:
    """Bind schema, canonical payload, native envelope, and exact bytes."""

    source = Path(path)
    payload = _read_json(source)
    rule = _require_nonblank(embedded_hash_rule, label="embedded_hash_rule")
    if rule not in _HASH_RULES:
        raise ValueError("unsupported embedded hash rule")
    embedded = payload.get("artifact_hash")
    if rule == "none":
        if embedded is not None:
            raise ValueError("none hash rule requires no embedded artifact hash")
    else:
        _require_sha256(embedded, label="source artifact_hash")
        bare = {key: value for key, value in payload.items() if key != "artifact_hash"}
        expected = (
            "sha256:" + stable_config_hash(bare)
            if rule == "stable_without_hash"
            else canonical_artifact_payload_hash(bare)
        )
        if embedded != expected:
            raise ValueError("source embedded artifact hash mismatch")
    return {
        "schema": HMC_PHASE5_ARTIFACT_REFERENCE_SCHEMA_V1,
        "source_schema": _require_nonblank(
            payload.get("schema"),
            label="source schema",
        ),
        "embedded_hash_rule": rule,
        "embedded_artifact_hash": embedded,
        "canonical_payload_hash": canonical_artifact_payload_hash(payload),
        "resolved_path_sha256": hashlib.sha256(
            str(source.resolve()).encode("utf-8")
        ).hexdigest(),
        "file_sha256": artifact_file_sha256(source),
        "byte_count": source.stat().st_size,
    }


def parse_phase5_artifact_reference(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(
        payload,
        required=(
            "schema",
            "source_schema",
            "embedded_hash_rule",
            "embedded_artifact_hash",
            "canonical_payload_hash",
            "resolved_path_sha256",
            "file_sha256",
            "byte_count",
        ),
        label="Phase 5 artifact reference",
    )
    if payload.get("schema") != HMC_PHASE5_ARTIFACT_REFERENCE_SCHEMA_V1:
        raise ValueError("unsupported Phase 5 artifact reference schema")
    _require_nonblank(payload.get("source_schema"), label="source_schema")
    rule = _require_nonblank(payload.get("embedded_hash_rule"), label="hash rule")
    if rule not in _HASH_RULES:
        raise ValueError("unsupported embedded hash rule")
    embedded = payload.get("embedded_artifact_hash")
    if rule == "none":
        if embedded is not None:
            raise ValueError("none hash rule cannot carry an embedded hash")
    else:
        _require_sha256(embedded, label="embedded_artifact_hash")
    _require_sha256(payload.get("canonical_payload_hash"), label="canonical hash")
    _require_sha256(
        payload.get("resolved_path_sha256"),
        label="resolved_path_sha256",
        tagged=False,
    )
    _require_sha256(payload.get("file_sha256"), label="file_sha256", tagged=False)
    _require_int(payload.get("byte_count"), label="byte_count")
    return payload


def verify_phase5_artifact_reference(
    payload: Mapping[str, Any],
    *,
    path: str | Path,
) -> Mapping[str, Any]:
    parse_phase5_artifact_reference(payload)
    expected = build_phase5_artifact_reference(
        path,
        embedded_hash_rule=payload["embedded_hash_rule"],
    )
    if json.loads(json.dumps(payload, sort_keys=True)) != json.loads(
        json.dumps(expected, sort_keys=True)
    ):
        raise ValueError("Phase 5 artifact reference does not match current bytes")
    return payload


def _parse_adopted_identities(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(
        payload,
        required=ADOPTED_IDENTITY_KEYS,
        label="adopted identities",
    )
    for name, value in payload.items():
        _require_sha256(
            value,
            label=f"adopted identities.{name}",
            tagged=name != "legacy_replay_file_sha256",
        )
    return payload


def _parse_baseline_adoption(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(
        payload,
        required=(
            "decision",
            "human_approval_statement",
            "human_approval_date",
            "certificate_reference",
            "public_proposal_reference",
            "phase4_output_manifest_reference",
        ),
        label="V2 baseline adoption",
    )
    if (
        payload.get("decision"),
        payload.get("human_approval_statement"),
        payload.get("human_approval_date"),
    ) != (CERTIFICATE_DECISION, HUMAN_APPROVAL_STATEMENT, HUMAN_APPROVAL_DATE):
        raise ValueError("V2 baseline human approval mismatch")
    expected_schemas = {
        "certificate_reference": HMC_MIGRATION_CERTIFICATE_SCHEMA_V1,
        "public_proposal_reference": (
            HMC_MIGRATION_CERTIFICATE_PUBLIC_PROPOSAL_SCHEMA_V1
        ),
        "phase4_output_manifest_reference": (
            HMC_MIGRATION_CERTIFICATE_OUTPUT_MANIFEST_SCHEMA_V1
        ),
    }
    for name, schema in expected_schemas.items():
        reference = parse_phase5_artifact_reference(payload[name])
        if reference["source_schema"] != schema:
            raise ValueError(f"V2 baseline {name} schema mismatch")
    if (
        payload["certificate_reference"]["embedded_artifact_hash"]
        != APPROVED_CERTIFICATE_ARTIFACT_HASH
    ):
        raise ValueError("V2 baseline certificate hash mismatch")
    return payload


def build_phase7_v2_config(
    *,
    historical_config: Mapping[str, Any],
    certificate: Mapping[str, Any],
    certificate_path: str | Path,
    public_proposal_path: str | Path,
    phase4_output_manifest_path: str | Path,
    governed_source_paths: Mapping[str, str | Path],
) -> Mapping[str, Any]:
    """Build the strict runtime-inert V2 config from approved evidence."""

    if historical_config.get("schema") != PHASE7_CONFIG_SCHEMA_V1:
        raise ValueError("historical Phase 7 config schema mismatch")
    parse_migration_certificate(certificate)
    if certificate.get("artifact_hash") != APPROVED_CERTIFICATE_ARTIFACT_HASH:
        raise ValueError("approved certificate artifact hash mismatch")
    proposal = _read_json(public_proposal_path)
    manifest = _read_json(phase4_output_manifest_path)
    parse_public_certificate_proposal(proposal)
    parse_certificate_output_manifest(manifest)
    _require_exact_keys(
        governed_source_paths,
        required=GOVERNED_SOURCE_KEYS,
        label="governed source paths",
    )
    source_references = {
        name: build_phase5_artifact_reference(
            governed_source_paths[name],
            embedded_hash_rule=GOVERNED_SOURCE_HASH_RULES[name],
        )
        for name in GOVERNED_SOURCE_KEYS
    }
    for name, reference in source_references.items():
        if reference["source_schema"] != GOVERNED_SOURCE_SCHEMAS[name]:
            raise ValueError(f"governed source schema mismatch: {name}")

    historical_hashes = dict(historical_config.get("expected_hashes", {}))
    _require_exact_keys(
        historical_hashes,
        required=LEGACY_HASH_KEYS,
        label="historical legacy hashes",
    )
    if historical_hashes != certificate["historical_expected_hashes"]:
        raise ValueError("certificate does not own historical V1 hashes")
    adopted = dict(certificate["refreshed_typed_identities"])
    _parse_adopted_identities(adopted)
    payload = {
        "schema": PHASE7_CONFIG_SCHEMA_V2,
        "config_id": V2_CONFIG_ID,
        "plan_path": V2_PLAN_PATH,
        "source_tuning_config_path": historical_config["source_tuning_config_path"],
        "source_tuning_config_hash": historical_config["source_tuning_config_hash"],
        "artifact_root": historical_config["artifact_root"],
        "execution": historical_config["execution"],
        "burnin": historical_config["burnin"],
        "retained": historical_config["retained"],
        "diagnostics": historical_config["diagnostics"],
        "artifacts": historical_config["artifacts"],
        "baseline_adoption": {
            "decision": CERTIFICATE_DECISION,
            "human_approval_statement": HUMAN_APPROVAL_STATEMENT,
            "human_approval_date": HUMAN_APPROVAL_DATE,
            "certificate_reference": build_phase5_artifact_reference(
                certificate_path,
                embedded_hash_rule="canonical_without_hash",
            ),
            "public_proposal_reference": build_phase5_artifact_reference(
                public_proposal_path,
                embedded_hash_rule="canonical_without_hash",
            ),
            "phase4_output_manifest_reference": build_phase5_artifact_reference(
                phase4_output_manifest_path,
                embedded_hash_rule="canonical_without_hash",
            ),
        },
        "adopted_identities": adopted,
        "governed_source_references": source_references,
        "historical_legacy_hashes": historical_hashes,
        "legacy_whole_payload_gate_status": LEGACY_GATE_STATUS,
        "runtime_authority": False,
        "nonclaims": V2_NONCLAIMS,
    }
    result = _embed_hash(payload)
    parse_phase7_v2_config(result)
    validate_phase7_v2_config_against_historical(
        result,
        historical_config=historical_config,
        certificate=certificate,
    )
    return result


def parse_phase7_v2_config(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(payload, required=_V2_FIELDS, label="Phase 7 V2 config")
    if payload.get("schema") != PHASE7_CONFIG_SCHEMA_V2:
        raise ValueError("unsupported Phase 7 V2 config schema")
    if payload.get("config_id") != V2_CONFIG_ID or payload.get("plan_path") != V2_PLAN_PATH:
        raise ValueError("Phase 7 V2 config identity mismatch")
    if (
        payload.get("source_tuning_config_path"),
        payload.get("source_tuning_config_hash"),
        payload.get("artifact_root"),
    ) != (
        V2_SOURCE_TUNING_CONFIG_PATH,
        V2_SOURCE_TUNING_CONFIG_HASH,
        V2_ARTIFACT_ROOT,
    ):
        raise ValueError("Phase 7 V2 source or artifact-root contract mismatch")
    for name, fields in _V2_SECTION_FIELDS.items():
        section = payload.get(name)
        _require_exact_keys(
            section,
            required=fields,
            label=f"Phase 7 V2 {name}",
        )
    _require_exact_keys(
        payload["execution"].get("thread_environment"),
        required=_THREAD_ENVIRONMENT_FIELDS,
        label="Phase 7 V2 thread_environment",
    )
    for name, expected in _V2_EXPECTED_SECTIONS.items():
        if payload[name] != expected:
            raise ValueError(f"Phase 7 V2 {name} contract mismatch")
    _parse_baseline_adoption(payload["baseline_adoption"])
    _parse_adopted_identities(payload["adopted_identities"])
    sources = payload.get("governed_source_references")
    _require_exact_keys(
        sources,
        required=GOVERNED_SOURCE_KEYS,
        label="V2 governed source references",
    )
    for name, reference in sources.items():
        parse_phase5_artifact_reference(reference)
        if reference["source_schema"] != GOVERNED_SOURCE_SCHEMAS[name] or (
            reference["embedded_hash_rule"] != GOVERNED_SOURCE_HASH_RULES[name]
        ):
            raise ValueError(f"V2 governed source reference mismatch: {name}")
    historical = payload.get("historical_legacy_hashes")
    _require_exact_keys(
        historical,
        required=LEGACY_HASH_KEYS,
        label="V2 historical legacy hashes",
    )
    for name, value in historical.items():
        _require_sha256(
            value,
            label=f"historical legacy hash.{name}",
            tagged=name in {"fixture", "xla_compile", "geometry", "mass"},
        )
    if payload.get("legacy_whole_payload_gate_status") != LEGACY_GATE_STATUS:
        raise ValueError("V2 legacy whole-payload status mismatch")
    if _require_bool(payload.get("runtime_authority"), label="runtime_authority"):
        raise ValueError("Phase 5 V2 config cannot authorize runtime")
    _require_ordered(payload.get("nonclaims"), expected=V2_NONCLAIMS, label="V2 nonclaims")
    _verify_hash(payload, label="Phase 7 V2 config")
    return payload


def validate_phase7_v2_config_against_historical(
    payload: Mapping[str, Any],
    *,
    historical_config: Mapping[str, Any],
    certificate: Mapping[str, Any],
) -> Mapping[str, Any]:
    parse_phase7_v2_config(payload)
    parse_migration_certificate(certificate)
    if historical_config.get("schema") != PHASE7_CONFIG_SCHEMA_V1:
        raise ValueError("historical Phase 7 config schema mismatch")
    for name in (
        "source_tuning_config_path",
        "source_tuning_config_hash",
        "artifact_root",
        "execution",
        "burnin",
        "retained",
        "diagnostics",
        "artifacts",
    ):
        if payload[name] != historical_config[name]:
            raise ValueError(f"V2 changed historical runtime field: {name}")
    if payload["historical_legacy_hashes"] != historical_config["expected_hashes"]:
        raise ValueError("V2 historical legacy hashes mismatch")
    if payload["adopted_identities"] != certificate["refreshed_typed_identities"]:
        raise ValueError("V2 adopted identities do not match certificate")
    if (
        payload["baseline_adoption"]["certificate_reference"][
            "embedded_artifact_hash"
        ]
        != certificate["artifact_hash"]
    ):
        raise ValueError("V2 does not bind the supplied certificate")
    return payload


def verify_phase7_v2_sources(
    payload: Mapping[str, Any],
    *,
    source_paths: Mapping[str, str | Path],
) -> Mapping[str, Any]:
    parse_phase7_v2_config(payload)
    _require_exact_keys(
        source_paths,
        required=GOVERNED_SOURCE_KEYS,
        label="V2 current source paths",
    )
    for name in GOVERNED_SOURCE_KEYS:
        verify_phase5_artifact_reference(
            payload["governed_source_references"][name],
            path=source_paths[name],
        )
    return payload


def build_phase5_adoption_record(
    *,
    v2_config: Mapping[str, Any],
    v2_config_path: str | Path,
    historical_v1_config_path: str | Path,
    certificate_path: str | Path,
    public_proposal_path: str | Path,
    phase4_output_manifest_path: str | Path,
) -> Mapping[str, Any]:
    parse_phase7_v2_config(v2_config)
    if canonical_artifact_payload_hash(_read_json(v2_config_path)) != (
        canonical_artifact_payload_hash(v2_config)
    ):
        raise ValueError("V2 config path does not contain supplied payload")
    result = _embed_hash(
        {
            "schema": HMC_PHASE5_ADOPTION_RECORD_SCHEMA_V1,
            "status": ADOPTION_STATUS,
            "decision": CERTIFICATE_DECISION,
            "human_approval_statement": HUMAN_APPROVAL_STATEMENT,
            "human_approval_date": HUMAN_APPROVAL_DATE,
            "certificate_reference": build_phase5_artifact_reference(
                certificate_path,
                embedded_hash_rule="canonical_without_hash",
            ),
            "public_proposal_reference": build_phase5_artifact_reference(
                public_proposal_path,
                embedded_hash_rule="canonical_without_hash",
            ),
            "phase4_output_manifest_reference": build_phase5_artifact_reference(
                phase4_output_manifest_path,
                embedded_hash_rule="canonical_without_hash",
            ),
            "historical_v1_config_reference": build_phase5_artifact_reference(
                historical_v1_config_path,
                embedded_hash_rule="none",
            ),
            "v2_config_reference": build_phase5_artifact_reference(
                v2_config_path,
                embedded_hash_rule="canonical_without_hash",
            ),
            "adopted_identities": dict(v2_config["adopted_identities"]),
            "legacy_whole_payload_gate_status": LEGACY_GATE_STATUS,
            "runtime_authority": False,
            "nonclaims": ADOPTION_NONCLAIMS,
        }
    )
    parse_phase5_adoption_record(result)
    return result


def parse_phase5_adoption_record(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(payload, required=_ADOPTION_FIELDS, label="Phase 5 adoption record")
    if payload.get("schema") != HMC_PHASE5_ADOPTION_RECORD_SCHEMA_V1:
        raise ValueError("unsupported Phase 5 adoption record schema")
    if (
        payload.get("status"),
        payload.get("decision"),
        payload.get("human_approval_statement"),
        payload.get("human_approval_date"),
    ) != (
        ADOPTION_STATUS,
        CERTIFICATE_DECISION,
        HUMAN_APPROVAL_STATEMENT,
        HUMAN_APPROVAL_DATE,
    ):
        raise ValueError("Phase 5 adoption decision or approval mismatch")
    expected_schemas = {
        "certificate_reference": HMC_MIGRATION_CERTIFICATE_SCHEMA_V1,
        "public_proposal_reference": HMC_MIGRATION_CERTIFICATE_PUBLIC_PROPOSAL_SCHEMA_V1,
        "phase4_output_manifest_reference": HMC_MIGRATION_CERTIFICATE_OUTPUT_MANIFEST_SCHEMA_V1,
        "historical_v1_config_reference": PHASE7_CONFIG_SCHEMA_V1,
        "v2_config_reference": PHASE7_CONFIG_SCHEMA_V2,
    }
    for name, schema in expected_schemas.items():
        reference = parse_phase5_artifact_reference(payload[name])
        if reference["source_schema"] != schema:
            raise ValueError(f"Phase 5 adoption {name} schema mismatch")
    if (
        payload["certificate_reference"]["embedded_artifact_hash"]
        != APPROVED_CERTIFICATE_ARTIFACT_HASH
    ):
        raise ValueError("Phase 5 adoption certificate mismatch")
    _parse_adopted_identities(payload["adopted_identities"])
    if payload.get("legacy_whole_payload_gate_status") != LEGACY_GATE_STATUS:
        raise ValueError("Phase 5 adoption legacy status mismatch")
    if _require_bool(payload.get("runtime_authority"), label="runtime_authority"):
        raise ValueError("Phase 5 adoption cannot authorize runtime")
    _require_ordered(
        payload.get("nonclaims"),
        expected=ADOPTION_NONCLAIMS,
        label="adoption nonclaims",
    )
    _verify_hash(payload, label="Phase 5 adoption record")
    return payload


def verify_phase5_adoption_record(
    payload: Mapping[str, Any],
    *,
    v2_config_path: str | Path,
    historical_v1_config_path: str | Path,
    certificate_path: str | Path,
    public_proposal_path: str | Path,
    phase4_output_manifest_path: str | Path,
) -> Mapping[str, Any]:
    parse_phase5_adoption_record(payload)
    paths = {
        "certificate_reference": certificate_path,
        "public_proposal_reference": public_proposal_path,
        "phase4_output_manifest_reference": phase4_output_manifest_path,
        "historical_v1_config_reference": historical_v1_config_path,
        "v2_config_reference": v2_config_path,
    }
    for name, path in paths.items():
        verify_phase5_artifact_reference(payload[name], path=path)
    certificate = _read_json(certificate_path)
    public_proposal = _read_json(public_proposal_path)
    phase4_manifest = _read_json(phase4_output_manifest_path)
    parse_migration_certificate(certificate)
    parse_public_certificate_proposal(public_proposal)
    parse_certificate_output_manifest(phase4_manifest)
    from bayesfilter.inference.hmc_identity_migration_certificate import (
        verify_certificate_output_manifest,
    )

    verify_certificate_output_manifest(
        phase4_manifest,
        certificate_path=certificate_path,
        public_proposal_path=public_proposal_path,
    )
    v2_config = _read_json(v2_config_path)
    parse_phase7_v2_config(v2_config)
    if payload["adopted_identities"] != v2_config["adopted_identities"]:
        raise ValueError("adoption record does not match V2 identities")
    v2_baseline = v2_config["baseline_adoption"]
    for adoption_name, baseline_name in (
        ("certificate_reference", "certificate_reference"),
        ("public_proposal_reference", "public_proposal_reference"),
        ("phase4_output_manifest_reference", "phase4_output_manifest_reference"),
    ):
        if payload[adoption_name] != v2_baseline[baseline_name]:
            raise ValueError(f"adoption record does not match V2 {baseline_name}")
    return payload


def build_phase5_preflight_report(
    *,
    config_reference: Mapping[str, Any],
    adoption_record_reference: Mapping[str, Any],
    identity_hashes: Mapping[str, Any],
    identity_checks: Mapping[str, Any],
    integrity_checks: Mapping[str, Any],
    legacy_audit: Mapping[str, Any],
    private_replay_artifact_hash: str,
    private_replay_file_sha256: str,
    parameter_names: Sequence[str],
    target_scope: str,
) -> Mapping[str, Any]:
    payload = {
        "schema": HMC_PHASE5_PREFLIGHT_REPORT_SCHEMA_V1,
        "passed": True,
        "decision": PREFLIGHT_DECISION,
        "config_reference": dict(config_reference),
        "adoption_record_reference": dict(adoption_record_reference),
        "identity_hashes": dict(identity_hashes),
        "identity_checks": dict(identity_checks),
        "integrity_checks": dict(integrity_checks),
        "legacy_audit": dict(legacy_audit),
        "private_replay_artifact_hash": private_replay_artifact_hash,
        "private_replay_file_sha256": private_replay_file_sha256,
        "parameter_names": tuple(parameter_names),
        "target_scope": target_scope,
        "runtime_authority": False,
        "runtime_executed": False,
        "nonclaims": PREFLIGHT_NONCLAIMS,
    }
    result = _embed_hash(payload)
    parse_phase5_preflight_report(result)
    return result


def parse_phase5_preflight_report(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(payload, required=_PREFLIGHT_FIELDS, label="Phase 5 preflight")
    if payload.get("schema") != HMC_PHASE5_PREFLIGHT_REPORT_SCHEMA_V1:
        raise ValueError("unsupported Phase 5 preflight schema")
    if payload.get("passed") is not True or payload.get("decision") != PREFLIGHT_DECISION:
        raise ValueError("Phase 5 preflight status mismatch")
    for name, schema in (
        ("config_reference", PHASE7_CONFIG_SCHEMA_V2),
        ("adoption_record_reference", HMC_PHASE5_ADOPTION_RECORD_SCHEMA_V1),
    ):
        reference = parse_phase5_artifact_reference(payload[name])
        if reference["source_schema"] != schema:
            raise ValueError(f"Phase 5 preflight {name} schema mismatch")
    identities = payload.get("identity_hashes")
    _require_exact_keys(
        identities,
        required=(
            "transition_identity_hash",
            "serious_execution_contract_hash",
            "smoke_execution_contract_hash",
            "selection_provenance_hash",
            "complete_tuning_payload_hash",
            "legacy_replay_canonical_payload_hash",
        ),
        label="preflight identity hashes",
    )
    for name, value in identities.items():
        _require_sha256(value, label=f"preflight identity hash.{name}")
    checks = payload.get("identity_checks")
    _require_exact_keys(
        checks,
        required=("transition", "serious_execution", "smoke_execution", "provenance"),
        label="preflight identity checks",
    )
    if any(_require_bool(value, label=f"identity check.{name}") is not True for name, value in checks.items()):
        raise ValueError("all Phase 5 identity checks must pass")
    integrity = payload.get("integrity_checks")
    _require_exact_keys(
        integrity,
        required=(
            *GOVERNED_SOURCE_KEYS,
            "certificate",
            "public_proposal",
            "phase4_output_manifest",
            "adoption_record",
        ),
        label="preflight integrity checks",
    )
    if any(_require_bool(value, label=f"integrity check.{name}") is not True for name, value in integrity.items()):
        raise ValueError("all Phase 5 integrity checks must pass")
    audit = payload.get("legacy_audit")
    _require_exact_keys(
        audit,
        required=("status", "public_final_kernel", "private_loop_final_kernel", "selected_trajectory"),
        label="preflight legacy audit",
    )
    if audit != {
        "status": LEGACY_GATE_STATUS,
        "public_final_kernel": "different",
        "private_loop_final_kernel": "different",
        "selected_trajectory": "different",
    }:
        raise ValueError("Phase 5 legacy audit mismatch")
    _require_sha256(payload.get("private_replay_artifact_hash"), label="private replay artifact")
    _require_sha256(
        payload.get("private_replay_file_sha256"),
        label="private replay file",
        tagged=False,
    )
    names = payload.get("parameter_names")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or not names:
        raise ValueError("preflight parameter_names must be non-empty")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("preflight parameter name is invalid")
    _require_nonblank(payload.get("target_scope"), label="preflight target_scope")
    if _require_bool(payload.get("runtime_authority"), label="runtime_authority"):
        raise ValueError("Phase 5 preflight cannot authorize runtime")
    if _require_bool(payload.get("runtime_executed"), label="runtime_executed"):
        raise ValueError("Phase 5 preflight cannot report runtime execution")
    _require_ordered(
        payload.get("nonclaims"),
        expected=PREFLIGHT_NONCLAIMS,
        label="preflight nonclaims",
    )
    _verify_hash(payload, label="Phase 5 preflight")
    return payload


def build_phase5_output_manifest(
    *,
    v2_config_path: str | Path,
    adoption_record_path: str | Path,
    preflight_report_path: str | Path,
) -> Mapping[str, Any]:
    paths = (
        ("v2_config", Path(v2_config_path), PHASE7_CONFIG_SCHEMA_V2),
        (
            "adoption_record",
            Path(adoption_record_path),
            HMC_PHASE5_ADOPTION_RECORD_SCHEMA_V1,
        ),
        (
            "preflight_report",
            Path(preflight_report_path),
            HMC_PHASE5_PREFLIGHT_REPORT_SCHEMA_V1,
        ),
    )
    outputs = []
    loaded: dict[str, Mapping[str, Any]] = {}
    for role, path, schema in paths:
        payload = _read_json(path)
        if role == "v2_config":
            parse_phase7_v2_config(payload)
        elif role == "adoption_record":
            parse_phase5_adoption_record(payload)
        else:
            parse_phase5_preflight_report(payload)
        loaded[role] = payload
        if HMC_PHASE5_OUTPUT_MANIFEST_SCHEMA_V1 in json.dumps(payload, sort_keys=True):
            raise ValueError("Phase 5 output references terminal manifest")
        outputs.append(
            {
                "role": role,
                "schema": schema,
                "artifact_hash": payload["artifact_hash"],
                "file_sha256": artifact_file_sha256(path),
                "byte_count": path.stat().st_size,
            }
        )
    expected_v2_reference = build_phase5_artifact_reference(
        v2_config_path,
        embedded_hash_rule="canonical_without_hash",
    )
    expected_adoption_reference = build_phase5_artifact_reference(
        adoption_record_path,
        embedded_hash_rule="canonical_without_hash",
    )
    if loaded["adoption_record"]["v2_config_reference"] != expected_v2_reference:
        raise ValueError("Phase 5 adoption does not reference the current V2 config")
    if loaded["preflight_report"]["config_reference"] != expected_v2_reference:
        raise ValueError("Phase 5 preflight does not reference the current V2 config")
    if (
        loaded["preflight_report"]["adoption_record_reference"]
        != expected_adoption_reference
    ):
        raise ValueError("Phase 5 preflight does not reference the adoption record")
    return _embed_hash(
        {
            "schema": HMC_PHASE5_OUTPUT_MANIFEST_SCHEMA_V1,
            "terminal_manifest": True,
            "outputs": tuple(outputs),
        }
    )


def parse_phase5_output_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(
        payload,
        required=("schema", "terminal_manifest", "outputs", "artifact_hash"),
        label="Phase 5 output manifest",
    )
    if payload.get("schema") != HMC_PHASE5_OUTPUT_MANIFEST_SCHEMA_V1:
        raise ValueError("unsupported Phase 5 output manifest schema")
    if payload.get("terminal_manifest") is not True:
        raise ValueError("Phase 5 output manifest must be terminal")
    outputs = payload.get("outputs")
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise ValueError("Phase 5 manifest outputs must be a sequence")
    expected = (
        ("v2_config", PHASE7_CONFIG_SCHEMA_V2),
        ("adoption_record", HMC_PHASE5_ADOPTION_RECORD_SCHEMA_V1),
        ("preflight_report", HMC_PHASE5_PREFLIGHT_REPORT_SCHEMA_V1),
    )
    if tuple((item.get("role"), item.get("schema")) for item in outputs) != expected:
        raise ValueError("Phase 5 output role/schema mismatch")
    for item in outputs:
        _require_exact_keys(
            item,
            required=("role", "schema", "artifact_hash", "file_sha256", "byte_count"),
            label="Phase 5 output entry",
        )
        _require_sha256(item["artifact_hash"], label="output artifact_hash")
        _require_sha256(item["file_sha256"], label="output file_sha256", tagged=False)
        _require_int(item["byte_count"], label="output byte_count")
    _verify_hash(payload, label="Phase 5 output manifest")
    return payload


def verify_phase5_output_manifest(
    payload: Mapping[str, Any],
    *,
    v2_config_path: str | Path,
    adoption_record_path: str | Path,
    preflight_report_path: str | Path,
) -> Mapping[str, Any]:
    parse_phase5_output_manifest(payload)
    expected = build_phase5_output_manifest(
        v2_config_path=v2_config_path,
        adoption_record_path=adoption_record_path,
        preflight_report_path=preflight_report_path,
    )
    if json.loads(json.dumps(payload, sort_keys=True)) != json.loads(
        json.dumps(expected, sort_keys=True)
    ):
        raise ValueError("Phase 5 output manifest does not match current bytes")
    return payload


def write_phase5_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    parser: Any,
) -> Mapping[str, Any]:
    parser(payload)
    atomic_write_json(path, payload)
    restored = _read_json(path)
    parser(restored)
    return restored
