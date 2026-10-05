"""Governed serialization for HMC semantic-identity migration evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayesfilter.inference.hmc_identity import (
    FROZEN_HMC_EXECUTION_CONTRACT_SCHEMA_V1,
    FROZEN_HMC_TRANSITION_IDENTITY_SCHEMA_V1,
    SELECTION_PROVENANCE_IDENTITY_SCHEMA_V1,
    FrozenHMCExecutionContractV1,
    FrozenHMCTransitionIdentityV1,
    SelectionProvenanceIdentityV1,
    SelectionStageIdentityV1,
    artifact_file_sha256,
    canonical_artifact_payload_hash,
)
from bayesfilter.runtime import atomic_write_json, stable_config_hash


HMC_IDENTITY_PRIVATE_SIDECAR_SCHEMA_V1 = (
    "bayesfilter.hmc_semantic_identity_private_sidecar.v1"
)
HMC_IDENTITY_PUBLIC_VALIDATION_SCHEMA_V1 = (
    "bayesfilter.hmc_semantic_identity_public_validation.v1"
)
HMC_IDENTITY_INPUT_INTEGRITY_MANIFEST_SCHEMA_V1 = (
    "bayesfilter.hmc_semantic_identity_input_integrity_manifest.v1"
)
HMC_IDENTITY_OUTPUT_INTEGRITY_MANIFEST_SCHEMA_V1 = (
    "bayesfilter.hmc_semantic_identity_output_integrity_manifest.v1"
)
HMC_IDENTITY_RECONSTRUCTION_LINKS_SCHEMA_V1 = (
    "bayesfilter.hmc_semantic_identity_reconstruction_links.v1"
)
HMC_IDENTITY_LEGACY_REPLAY_INTEGRITY_SCHEMA_V1 = (
    "bayesfilter.hmc_semantic_identity_legacy_replay_integrity.v1"
)
HMC_IDENTITY_LEGACY_VALIDATOR_RESULT_SCHEMA_V1 = (
    "bayesfilter.hmc_semantic_identity_legacy_validator_result.v1"
)

PHASE3_STATUS = "blocked_legacy_gate"
PHASE3_DECISION = (
    "CANDIDATE_IDENTITIES_RECORDED_LEGACY_GATE_REMAINS_BINDING"
)
PHASE3_LEGACY_VETO_CODE = "LEGACY_WHOLE_PAYLOAD_HASH_MISMATCH"
PHASE3_LEGACY_EXCEPTION_TYPE = "DeterministicLGSSMPhase7Error"
PHASE3_LEGACY_EXCEPTION_MESSAGE = "public final kernel hash mismatch"
PHASE3_CANDIDATE_CHECK_KEYS = (
    "transition_reconstructed",
    "serious_execution_reconstructed",
    "smoke_execution_reconstructed",
    "selection_provenance_reconstructed",
    "private_sidecar_round_trip",
    "public_private_hashes_match",
    "governed_inputs_unchanged",
    "public_redaction_passed",
)
PHASE3_PUBLIC_NONCLAIMS = (
    "candidate semantic identity engineering evidence only",
    "legacy whole-payload gate remains binding",
    "not baseline adoption",
    "not Phase 7 readiness or execution",
    "not posterior convergence or recovery evidence",
    "not production, default, GPU, NeuTra, or scientific evidence",
)
PHASE3_PRIVATE_NONCLAIMS = (
    "private candidate semantic identity engineering evidence only",
    "legacy whole-payload gate remains binding",
    "not baseline adoption or Phase 7 execution",
    "not convergence, recovery, production, default, GPU, NeuTra, or scientific evidence",
)

_HEX_DIGITS = frozenset("0123456789abcdef")
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
_SIDECAR_FIELDS = (
    "schema",
    "transition_identity",
    "transition_identity_hash",
    "serious_execution_contract",
    "serious_execution_contract_hash",
    "smoke_execution_contract",
    "smoke_execution_contract_hash",
    "selection_provenance",
    "selection_provenance_hash",
    "complete_tuning_payload_hash",
    "legacy_private_replay_integrity",
    "legacy_validator_result",
    "reconstruction_links",
    "nonclaims",
    "artifact_hash",
)
_PUBLIC_FIELDS = (
    "schema",
    "status",
    "decision",
    "transition_identity_schema",
    "transition_identity_hash",
    "serious_execution_contract_schema",
    "serious_execution_contract_hash",
    "smoke_execution_contract_schema",
    "smoke_execution_contract_hash",
    "selection_provenance_schema",
    "selection_provenance_hash",
    "candidate_checks",
    "legacy_gate",
    "legacy_private_replay_reference",
    "private_sidecar_reference",
    "input_integrity_manifest_hash",
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
    prefix = "sha256:"
    digest = text.removeprefix(prefix) if tagged else text
    if tagged != text.startswith(prefix):
        form = "tagged" if tagged else "bare"
        raise ValueError(f"{label} must be a {form} SHA-256")
    if len(digest) != 64 or any(char not in _HEX_DIGITS for char in digest):
        raise ValueError(f"{label} must be a complete lowercase SHA-256")
    return text


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _require_ordered_strings(
    value: Any,
    *,
    expected: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    observed = tuple(value)
    if observed != tuple(expected):
        raise ValueError(f"{label} must match the closed ordered contract")
    return observed


def _without_artifact_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "artifact_hash"}


def _json_normalize(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=True))


def _embed_artifact_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "artifact_hash" in result:
        raise ValueError("artifact_hash must not be prepopulated")
    result["artifact_hash"] = canonical_artifact_payload_hash(result)
    return result


def _verify_embedded_artifact_hash(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> str:
    observed = _require_sha256(
        payload.get("artifact_hash"),
        label=f"{label} artifact_hash",
        tagged=True,
    )
    expected = canonical_artifact_payload_hash(_without_artifact_hash(payload))
    if observed != expected:
        raise ValueError(f"{label} embedded artifact hash mismatch")
    return observed


@dataclass(frozen=True)
class LegacyReplayIntegrityV1:
    embedded_artifact_hash: str
    canonical_payload_hash: str
    file_sha256: str
    byte_count: int
    schema: str = HMC_IDENTITY_LEGACY_REPLAY_INTEGRITY_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != HMC_IDENTITY_LEGACY_REPLAY_INTEGRITY_SCHEMA_V1:
            raise ValueError("unsupported legacy replay-integrity schema")
        _require_sha256(
            self.embedded_artifact_hash,
            label="embedded_artifact_hash",
            tagged=True,
        )
        _require_sha256(
            self.canonical_payload_hash,
            label="canonical_payload_hash",
            tagged=True,
        )
        _require_sha256(self.file_sha256, label="file_sha256", tagged=False)
        _require_int(self.byte_count, label="byte_count", minimum=1)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LegacyReplayIntegrityV1":
        _require_exact_keys(
            payload,
            required=tuple(cls.__dataclass_fields__),
            label="legacy replay integrity",
        )
        return cls(**dict(payload))

    def payload(self) -> Mapping[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class LegacyValidatorResultV1:
    passed: bool
    exception_type: str
    message: str
    veto_code: str
    remains_binding: bool
    schema: str = HMC_IDENTITY_LEGACY_VALIDATOR_RESULT_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != HMC_IDENTITY_LEGACY_VALIDATOR_RESULT_SCHEMA_V1:
            raise ValueError("unsupported legacy validator-result schema")
        if _require_bool(self.passed, label="legacy validator passed") is not False:
            raise ValueError("Phase 3 legacy validator result must be failed")
        if self.exception_type != PHASE3_LEGACY_EXCEPTION_TYPE:
            raise ValueError("legacy validator exception type mismatch")
        if self.message != PHASE3_LEGACY_EXCEPTION_MESSAGE:
            raise ValueError("legacy validator exception message mismatch")
        if self.veto_code != PHASE3_LEGACY_VETO_CODE:
            raise ValueError("legacy validator veto code mismatch")
        if _require_bool(
            self.remains_binding,
            label="legacy remains_binding",
        ) is not True:
            raise ValueError("legacy validator veto must remain binding")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LegacyValidatorResultV1":
        _require_exact_keys(
            payload,
            required=tuple(cls.__dataclass_fields__),
            label="legacy validator result",
        )
        return cls(**dict(payload))

    def payload(self) -> Mapping[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ReconstructionLinksV1:
    base_adapter_signature: str
    phase4_hmc_adapter_signature: str
    final_hmc_adapter_signature: str
    geometry_mass_artifact_signature: str
    adapted_mass_artifact_signature: str
    schema: str = HMC_IDENTITY_RECONSTRUCTION_LINKS_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != HMC_IDENTITY_RECONSTRUCTION_LINKS_SCHEMA_V1:
            raise ValueError("unsupported reconstruction-links schema")
        for name in (
            "base_adapter_signature",
            "phase4_hmc_adapter_signature",
            "final_hmc_adapter_signature",
            "geometry_mass_artifact_signature",
            "adapted_mass_artifact_signature",
        ):
            _require_sha256(getattr(self, name), label=name, tagged=False)

    @classmethod
    def from_replay(cls, replay: Any) -> "ReconstructionLinksV1":
        contract = getattr(replay, "contract", None)
        if not isinstance(contract, Mapping):
            raise ValueError("replay reconstruction contract is missing")
        return cls(
            base_adapter_signature=contract.get("base_adapter_signature"),
            phase4_hmc_adapter_signature=contract.get(
                "phase4_hmc_adapter_signature"
            ),
            final_hmc_adapter_signature=contract.get("final_hmc_adapter_signature"),
            geometry_mass_artifact_signature=contract.get(
                "geometry_mass_artifact_signature"
            ),
            adapted_mass_artifact_signature=contract.get(
                "adapted_mass_artifact_signature"
            ),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ReconstructionLinksV1":
        _require_exact_keys(
            payload,
            required=tuple(cls.__dataclass_fields__),
            label="reconstruction links",
        )
        return cls(**dict(payload))

    def payload(self) -> Mapping[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def build_selection_provenance_from_tuning_payload(
    *,
    tuning_payload: Mapping[str, Any],
    tuning_config_hash: str,
) -> SelectionProvenanceIdentityV1:
    """Bind the full selection payload and its named selected-attempt lineage."""

    if not isinstance(tuning_payload, Mapping):
        raise TypeError("tuning_payload must be a mapping")
    loop = tuning_payload.get("tune_verify_repair_loop")
    final_kernel = loop.get("final_kernel_payload") if isinstance(loop, Mapping) else None
    attempts = loop.get("attempts") if isinstance(loop, Mapping) else None
    if not isinstance(final_kernel, Mapping) or not isinstance(attempts, Sequence):
        raise ValueError("tuning payload is missing the selected repair-loop history")
    passed_attempts = [
        item
        for item in attempts
        if isinstance(item, Mapping)
        and item.get("passed") is True
        and item.get("final_status") == "passed"
    ]
    if len(passed_attempts) != 1:
        raise ValueError("tuning payload must contain exactly one selected passed attempt")
    selected_attempt = passed_attempts[0]
    selected_index = _require_int(
        selected_attempt.get("attempt_index"),
        label="selected attempt index",
    )
    if selected_index >= len(attempts) or attempts[selected_index] != selected_attempt:
        raise ValueError("selected attempt index does not match attempt ordering")
    private_final_kernel = loop.get("final_kernel_payload")
    public_final_kernel = tuning_payload.get("final_kernel_payload")
    private_final_hash = loop.get("final_kernel_hash")
    public_final_hash = tuning_payload.get("final_kernel_hash")
    if not isinstance(private_final_kernel, Mapping) or not isinstance(
        public_final_kernel, Mapping
    ):
        raise ValueError("tuning payload is missing private or public final kernel")
    for label, source, expected_hash in (
        ("private final kernel", private_final_kernel, private_final_hash),
        ("public final kernel", public_final_kernel, public_final_hash),
    ):
        _require_sha256(expected_hash, label=f"{label} hash", tagged=False)
        if stable_config_hash(source) != expected_hash:
            raise ValueError(f"{label} hash does not match source payload")

    stage_sources = (
        (
            "bootstrap",
            tuning_payload.get("bootstrap"),
            final_kernel.get("bootstrap_artifact_hash"),
            None,
        ),
        (
            "geometry",
            tuning_payload.get("geometry"),
            final_kernel.get("geometry_artifact_hash"),
            None,
        ),
        (
            "windowed_mass",
            selected_attempt.get("windowed_stage"),
            final_kernel.get("windowed_stage_artifact_hash"),
            selected_index,
        ),
        (
            "fixed_mass_step",
            selected_attempt.get("fixed_mass_step_stage"),
            final_kernel.get("fixed_mass_step_stage_artifact_hash"),
            selected_index,
        ),
        (
            "frozen_step_trajectory",
            selected_attempt.get("frozen_step_trajectory_stage"),
            final_kernel.get("frozen_step_trajectory_stage_artifact_hash"),
            selected_index,
        ),
        (
            "fresh_verification",
            private_final_kernel,
            private_final_hash,
            selected_index,
        ),
        (
            "tune_verify_repair_loop",
            loop,
            tuning_payload.get("loop_artifact_hash"),
            selected_index,
        ),
    )
    lineage: list[SelectionStageIdentityV1] = []
    for stage_id, source, legacy_hash, index in stage_sources:
        if not isinstance(source, Mapping):
            raise ValueError(f"selection stage is missing: {stage_id}")
        source_schema = source.get("schema")
        if stage_id == "geometry":
            if source.get("artifact_type") != (
                "bayesfilter_hmc_geometry_initialization_result"
            ) or source.get("schema_version") != 1:
                raise ValueError("geometry selection stage schema mismatch")
            source_schema = "bayesfilter.hmc_geometry_initialization_result.v1"
        _require_sha256(legacy_hash, label=f"{stage_id} legacy stage hash", tagged=False)
        if stage_id == "geometry":
            observed_legacy_hash = tuning_payload.get("geometry_artifact_hash")
        else:
            observed_legacy_hash = stable_config_hash(source)
        if observed_legacy_hash != legacy_hash:
            raise ValueError(f"{stage_id} legacy stage hash does not match source payload")
        lineage.append(
            SelectionStageIdentityV1(
                stage_id=stage_id,
                source_schema=source_schema,
                canonical_payload_hash=f"sha256:{legacy_hash}",
                selected_index=index,
            )
        )
    return SelectionProvenanceIdentityV1.from_source_payload(
        source_selection_payload=tuning_payload,
        tuning_config_hash=tuning_config_hash,
        stage_lineage=lineage,
        selected_step_hash=final_kernel.get("selected_step_hash"),
        selected_trajectory_hash=final_kernel.get("selected_trajectory_hash"),
    )


def build_private_identity_sidecar(
    *,
    transition: FrozenHMCTransitionIdentityV1,
    serious_execution: FrozenHMCExecutionContractV1,
    smoke_execution: FrozenHMCExecutionContractV1,
    selection_provenance: SelectionProvenanceIdentityV1,
    complete_tuning_payload: Mapping[str, Any],
    legacy_private_replay_payload: Mapping[str, Any],
    legacy_private_replay_path: str | Path,
    legacy_private_replay_reference: Mapping[str, Any],
    replay: Any,
    legacy_validator_result: LegacyValidatorResultV1,
) -> Mapping[str, Any]:
    if serious_execution.run_mode != "serious" or smoke_execution.run_mode != "smoke":
        raise ValueError("sidecar requires serious and smoke execution contracts")
    if serious_execution.transition_identity_hash != transition.identity_hash or (
        smoke_execution.transition_identity_hash != transition.identity_hash
    ):
        raise ValueError("execution contracts must bind the sidecar transition")
    complete_tuning_hash = canonical_artifact_payload_hash(complete_tuning_payload)
    if selection_provenance.source_selection_payload_hash != complete_tuning_hash:
        raise ValueError(
            "selection provenance must bind the complete tuning payload"
        )
    _require_exact_keys(
        legacy_private_replay_reference,
        required=("artifact_hash", "file_sha256", "byte_count"),
        label="legacy private replay reference",
    )
    _strict_public_legacy_reference(legacy_private_replay_reference)
    legacy_integrity = LegacyReplayIntegrityV1(
        embedded_artifact_hash=legacy_private_replay_reference["artifact_hash"],
        canonical_payload_hash=canonical_artifact_payload_hash(
            legacy_private_replay_payload
        ),
        file_sha256=legacy_private_replay_reference["file_sha256"],
        byte_count=legacy_private_replay_reference["byte_count"],
    )
    replay_path = Path(legacy_private_replay_path)
    if legacy_integrity.embedded_artifact_hash != legacy_private_replay_payload.get(
        "artifact_hash"
    ):
        raise ValueError("legacy replay embedded artifact reference mismatch")
    recomputed_embedded_hash = (
        f"sha256:{stable_config_hash(_without_artifact_hash(legacy_private_replay_payload))}"
    )
    if legacy_integrity.embedded_artifact_hash != recomputed_embedded_hash:
        raise ValueError("legacy replay embedded artifact hash mismatch")
    if legacy_integrity.file_sha256 != artifact_file_sha256(replay_path) or (
        legacy_integrity.byte_count != replay_path.stat().st_size
    ):
        raise ValueError("legacy replay exact-file reference mismatch")

    payload = {
        "schema": HMC_IDENTITY_PRIVATE_SIDECAR_SCHEMA_V1,
        "transition_identity": transition.payload(),
        "transition_identity_hash": transition.identity_hash,
        "serious_execution_contract": serious_execution.payload(),
        "serious_execution_contract_hash": serious_execution.identity_hash,
        "smoke_execution_contract": smoke_execution.payload(),
        "smoke_execution_contract_hash": smoke_execution.identity_hash,
        "selection_provenance": selection_provenance.payload(),
        "selection_provenance_hash": selection_provenance.identity_hash,
        "complete_tuning_payload_hash": complete_tuning_hash,
        "legacy_private_replay_integrity": legacy_integrity.payload(),
        "legacy_validator_result": legacy_validator_result.payload(),
        "reconstruction_links": ReconstructionLinksV1.from_replay(replay).payload(),
        "nonclaims": PHASE3_PRIVATE_NONCLAIMS,
    }
    return _embed_artifact_hash(payload)


def parse_private_identity_sidecar(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require_exact_keys(
        payload,
        required=_SIDECAR_FIELDS,
        label="private identity sidecar",
    )
    if payload.get("schema") != HMC_IDENTITY_PRIVATE_SIDECAR_SCHEMA_V1:
        raise ValueError("unsupported private identity sidecar schema")
    transition = FrozenHMCTransitionIdentityV1.from_payload(
        payload["transition_identity"]
    )
    serious = FrozenHMCExecutionContractV1.from_payload(
        payload["serious_execution_contract"]
    )
    smoke = FrozenHMCExecutionContractV1.from_payload(
        payload["smoke_execution_contract"]
    )
    provenance = SelectionProvenanceIdentityV1.from_payload(
        payload["selection_provenance"]
    )
    expected = {
        "transition_identity_hash": transition.identity_hash,
        "serious_execution_contract_hash": serious.identity_hash,
        "smoke_execution_contract_hash": smoke.identity_hash,
        "selection_provenance_hash": provenance.identity_hash,
        "complete_tuning_payload_hash": provenance.source_selection_payload_hash,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"private sidecar {name} mismatch")
    if serious.run_mode != "serious" or smoke.run_mode != "smoke":
        raise ValueError("private sidecar execution modes mismatch")
    if serious.transition_identity_hash != transition.identity_hash or (
        smoke.transition_identity_hash != transition.identity_hash
    ):
        raise ValueError("private sidecar execution transition mismatch")
    _require_sha256(
        payload.get("complete_tuning_payload_hash"),
        label="complete_tuning_payload_hash",
        tagged=True,
    )
    LegacyReplayIntegrityV1.from_payload(payload["legacy_private_replay_integrity"])
    LegacyValidatorResultV1.from_payload(payload["legacy_validator_result"])
    ReconstructionLinksV1.from_payload(payload["reconstruction_links"])
    _require_ordered_strings(
        payload.get("nonclaims"),
        expected=PHASE3_PRIVATE_NONCLAIMS,
        label="private sidecar nonclaims",
    )
    _verify_embedded_artifact_hash(payload, label="private identity sidecar")
    return payload


def write_private_identity_sidecar(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    parse_private_identity_sidecar(payload)
    atomic_write_json(path, payload)
    restored = json.loads(Path(path).read_text(encoding="utf-8"))
    parse_private_identity_sidecar(restored)
    return restored


def snapshot_governed_inputs(paths: Sequence[str | Path]) -> tuple[Mapping[str, Any], ...]:
    if not paths:
        raise ValueError("governed input paths must be non-empty")
    entries: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for item in paths:
        path = Path(item).resolve()
        normalized = str(path)
        if normalized in seen:
            raise ValueError("governed input paths must be unique")
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(normalized)
        entries.append(
            {
                "path": normalized,
                "file_sha256": artifact_file_sha256(path),
                "byte_count": path.stat().st_size,
            }
        )
    return tuple(entries)


def build_input_integrity_manifest(
    *,
    pre_snapshot: Sequence[Mapping[str, Any]],
    post_snapshot: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    pre = tuple(dict(item) for item in pre_snapshot)
    post = tuple(dict(item) for item in post_snapshot)
    for label, entries in (("pre_snapshot", pre), ("post_snapshot", post)):
        for entry in entries:
            _require_exact_keys(
                entry,
                required=("path", "file_sha256", "byte_count"),
                label=f"input manifest {label} entry",
            )
            _require_nonblank(entry["path"], label=f"{label} path")
            _require_sha256(
                entry["file_sha256"],
                label=f"{label} file_sha256",
                tagged=False,
            )
            _require_int(entry["byte_count"], label=f"{label} byte_count", minimum=1)
    if pre != post:
        raise ValueError("governed inputs changed during Phase 3 evidence generation")
    return _embed_artifact_hash(
        {
            "schema": HMC_IDENTITY_INPUT_INTEGRITY_MANIFEST_SCHEMA_V1,
            "governed_inputs_unchanged": True,
            "pre_snapshot": pre,
            "post_snapshot": post,
        }
    )


def parse_input_integrity_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(
        payload,
        required=(
            "schema",
            "governed_inputs_unchanged",
            "pre_snapshot",
            "post_snapshot",
            "artifact_hash",
        ),
        label="input integrity manifest",
    )
    if payload.get("schema") != HMC_IDENTITY_INPUT_INTEGRITY_MANIFEST_SCHEMA_V1:
        raise ValueError("unsupported input integrity manifest schema")
    if _require_bool(
        payload.get("governed_inputs_unchanged"),
        label="governed_inputs_unchanged",
    ) is not True:
        raise ValueError("input integrity manifest must prove unchanged inputs")
    rebuilt = build_input_integrity_manifest(
        pre_snapshot=payload["pre_snapshot"],
        post_snapshot=payload["post_snapshot"],
    )
    if _json_normalize(rebuilt) != _json_normalize(payload):
        raise ValueError("input integrity manifest content mismatch")
    _verify_embedded_artifact_hash(payload, label="input integrity manifest")
    return payload


def verify_input_integrity_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Reopen every governed input and verify the recorded post-snapshot."""

    parse_input_integrity_manifest(payload)
    current = snapshot_governed_inputs(
        tuple(item["path"] for item in payload["post_snapshot"])
    )
    if _json_normalize(current) != _json_normalize(payload["post_snapshot"]):
        raise ValueError("governed inputs no longer match the integrity manifest")
    return payload


def _strict_public_legacy_reference(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(
        payload,
        required=("artifact_hash", "file_sha256", "byte_count"),
        label="public legacy private replay reference",
    )
    _require_sha256(payload["artifact_hash"], label="artifact_hash", tagged=True)
    _require_sha256(payload["file_sha256"], label="file_sha256", tagged=False)
    _require_int(payload["byte_count"], label="byte_count", minimum=1)
    return payload


def _strict_public_sidecar_reference(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(
        payload,
        required=(
            "schema",
            "artifact_hash",
            "file_sha256",
            "byte_count",
            *_PUBLIC_REDACTION_FIELDS,
        ),
        label="public private-sidecar reference",
    )
    if payload.get("schema") != HMC_IDENTITY_PRIVATE_SIDECAR_SCHEMA_V1:
        raise ValueError("public private-sidecar schema mismatch")
    _require_sha256(payload["artifact_hash"], label="artifact_hash", tagged=True)
    _require_sha256(payload["file_sha256"], label="file_sha256", tagged=False)
    _require_int(payload["byte_count"], label="byte_count", minimum=1)
    for name in _PUBLIC_REDACTION_FIELDS:
        if _require_bool(payload[name], label=name) is not False:
            raise ValueError(f"public sidecar reference requires {name}=false")
    return payload


def build_public_validation_record(
    *,
    sidecar_payload: Mapping[str, Any],
    sidecar_path: str | Path,
    input_integrity_manifest: Mapping[str, Any],
    legacy_private_replay_reference: Mapping[str, Any],
    candidate_checks: Mapping[str, Any],
) -> Mapping[str, Any]:
    parse_private_identity_sidecar(sidecar_payload)
    parse_input_integrity_manifest(input_integrity_manifest)
    _strict_public_legacy_reference(legacy_private_replay_reference)
    _require_exact_keys(
        candidate_checks,
        required=PHASE3_CANDIDATE_CHECK_KEYS,
        label="candidate checks",
    )
    normalized_checks = {
        name: _require_bool(candidate_checks[name], label=f"candidate_checks.{name}")
        for name in PHASE3_CANDIDATE_CHECK_KEYS
    }
    if not all(normalized_checks.values()):
        raise ValueError("all Phase 3 candidate checks must pass before publication")
    sidecar_file = Path(sidecar_path)
    payload = {
        "schema": HMC_IDENTITY_PUBLIC_VALIDATION_SCHEMA_V1,
        "status": PHASE3_STATUS,
        "decision": PHASE3_DECISION,
        "transition_identity_schema": sidecar_payload["transition_identity"]["schema"],
        "transition_identity_hash": sidecar_payload["transition_identity_hash"],
        "serious_execution_contract_schema": sidecar_payload[
            "serious_execution_contract"
        ]["schema"],
        "serious_execution_contract_hash": sidecar_payload[
            "serious_execution_contract_hash"
        ],
        "smoke_execution_contract_schema": sidecar_payload[
            "smoke_execution_contract"
        ]["schema"],
        "smoke_execution_contract_hash": sidecar_payload[
            "smoke_execution_contract_hash"
        ],
        "selection_provenance_schema": sidecar_payload["selection_provenance"][
            "schema"
        ],
        "selection_provenance_hash": sidecar_payload["selection_provenance_hash"],
        "candidate_checks": normalized_checks,
        "legacy_gate": {
            "passed": False,
            "veto_code": PHASE3_LEGACY_VETO_CODE,
            "remains_binding": True,
        },
        "legacy_private_replay_reference": dict(legacy_private_replay_reference),
        "private_sidecar_reference": {
            "schema": HMC_IDENTITY_PRIVATE_SIDECAR_SCHEMA_V1,
            "artifact_hash": sidecar_payload["artifact_hash"],
            "file_sha256": artifact_file_sha256(sidecar_file),
            "byte_count": sidecar_file.stat().st_size,
            **{name: False for name in _PUBLIC_REDACTION_FIELDS},
        },
        "input_integrity_manifest_hash": input_integrity_manifest["artifact_hash"],
        "nonclaims": PHASE3_PUBLIC_NONCLAIMS,
    }
    return _embed_artifact_hash(payload)


def parse_public_validation_record(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(
        payload,
        required=_PUBLIC_FIELDS,
        label="public validation record",
    )
    if payload.get("schema") != HMC_IDENTITY_PUBLIC_VALIDATION_SCHEMA_V1:
        raise ValueError("unsupported public validation record schema")
    if payload.get("status") != PHASE3_STATUS or payload.get("decision") != PHASE3_DECISION:
        raise ValueError("public validation status or decision mismatch")
    expected_schemas = {
        "transition_identity_schema": FROZEN_HMC_TRANSITION_IDENTITY_SCHEMA_V1,
        "serious_execution_contract_schema": FROZEN_HMC_EXECUTION_CONTRACT_SCHEMA_V1,
        "smoke_execution_contract_schema": FROZEN_HMC_EXECUTION_CONTRACT_SCHEMA_V1,
        "selection_provenance_schema": SELECTION_PROVENANCE_IDENTITY_SCHEMA_V1,
    }
    if any(payload.get(name) != expected for name, expected in expected_schemas.items()):
        raise ValueError("public identity schema mismatch")
    for name in (
        "transition_identity_hash",
        "serious_execution_contract_hash",
        "smoke_execution_contract_hash",
        "selection_provenance_hash",
        "input_integrity_manifest_hash",
    ):
        _require_sha256(payload.get(name), label=name, tagged=True)
    checks = payload.get("candidate_checks")
    _require_exact_keys(
        checks,
        required=PHASE3_CANDIDATE_CHECK_KEYS,
        label="candidate checks",
    )
    for name in PHASE3_CANDIDATE_CHECK_KEYS:
        if _require_bool(checks[name], label=f"candidate_checks.{name}") is not True:
            raise ValueError("public candidate checks must all pass")
    legacy = payload.get("legacy_gate")
    _require_exact_keys(
        legacy,
        required=("passed", "veto_code", "remains_binding"),
        label="legacy gate",
    )
    if legacy != {
        "passed": False,
        "veto_code": PHASE3_LEGACY_VETO_CODE,
        "remains_binding": True,
    }:
        raise ValueError("public legacy gate must preserve the binding veto")
    _strict_public_legacy_reference(payload["legacy_private_replay_reference"])
    _strict_public_sidecar_reference(payload["private_sidecar_reference"])
    _require_ordered_strings(
        payload.get("nonclaims"),
        expected=PHASE3_PUBLIC_NONCLAIMS,
        label="public nonclaims",
    )
    assert_public_validation_redacted(payload)
    _verify_embedded_artifact_hash(payload, label="public validation record")
    return payload


def assert_public_validation_redacted(
    payload: Mapping[str, Any],
    *,
    forbidden_values: Sequence[str] = (),
) -> None:
    """Defense-in-depth scan supplementing the exact public schema."""

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
        "base_adapter_signature",
        "phase4_hmc_adapter_signature",
        "final_hmc_adapter_signature",
        "geometry_mass_artifact_signature",
        "adapted_mass_artifact_signature",
        "complete_tuning_payload_hash",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            overlap = forbidden_keys.intersection(value)
            if overlap:
                raise ValueError(f"public validation contains forbidden keys: {sorted(overlap)}")
            for item in value.values():
                walk(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                walk(item)
        elif isinstance(value, str) and value.startswith("/"):
            raise ValueError("public validation contains an absolute private path")

    walk(payload)
    serialized = json.dumps(payload, sort_keys=True)
    for secret in forbidden_values:
        text = _require_nonblank(secret, label="forbidden public value")
        if text in serialized:
            raise ValueError("public validation contains a forbidden private value")


def public_record_matches_private_sidecar(
    *,
    public_record: Mapping[str, Any],
    sidecar_payload: Mapping[str, Any],
    sidecar_path: str | Path,
    input_integrity_manifest: Mapping[str, Any],
) -> bool:
    parse_public_validation_record(public_record)
    parse_private_identity_sidecar(sidecar_payload)
    parse_input_integrity_manifest(input_integrity_manifest)
    expected = {
        "transition_identity_hash": sidecar_payload["transition_identity_hash"],
        "serious_execution_contract_hash": sidecar_payload[
            "serious_execution_contract_hash"
        ],
        "smoke_execution_contract_hash": sidecar_payload[
            "smoke_execution_contract_hash"
        ],
        "selection_provenance_hash": sidecar_payload["selection_provenance_hash"],
        "input_integrity_manifest_hash": input_integrity_manifest["artifact_hash"],
    }
    if any(public_record.get(name) != value for name, value in expected.items()):
        return False
    reference = public_record["private_sidecar_reference"]
    public_legacy = public_record["legacy_private_replay_reference"]
    private_legacy = sidecar_payload["legacy_private_replay_integrity"]
    path = Path(sidecar_path)
    return bool(
        reference["artifact_hash"] == sidecar_payload["artifact_hash"]
        and reference["file_sha256"] == artifact_file_sha256(path)
        and reference["byte_count"] == path.stat().st_size
        and public_legacy["artifact_hash"]
        == private_legacy["embedded_artifact_hash"]
        and public_legacy["file_sha256"] == private_legacy["file_sha256"]
        and public_legacy["byte_count"] == private_legacy["byte_count"]
    )


def build_output_integrity_manifest(
    *,
    sidecar_path: str | Path,
    input_manifest_path: str | Path,
    public_record_path: str | Path,
) -> Mapping[str, Any]:
    paths = (
        ("private_sidecar", Path(sidecar_path)),
        ("input_integrity_manifest", Path(input_manifest_path)),
        ("public_validation_record", Path(public_record_path)),
    )
    outputs = []
    loaded: dict[str, Mapping[str, Any]] = {}
    for role, path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"output must contain a JSON object: {role}")
        if role == "private_sidecar":
            parse_private_identity_sidecar(payload)
        elif role == "input_integrity_manifest":
            verify_input_integrity_manifest(payload)
        else:
            parse_public_validation_record(payload)
        loaded[role] = payload
        serialized = json.dumps(payload, sort_keys=True)
        if HMC_IDENTITY_OUTPUT_INTEGRITY_MANIFEST_SCHEMA_V1 in serialized or (
            "output_integrity_manifest" in serialized
        ):
            raise ValueError("a terminal-manifest input references the terminal manifest")
        outputs.append(
            {
                "role": role,
                "schema": payload.get("schema"),
                "artifact_hash": payload.get("artifact_hash"),
                "file_sha256": artifact_file_sha256(path),
                "byte_count": path.stat().st_size,
            }
        )
    if not public_record_matches_private_sidecar(
        public_record=loaded["public_validation_record"],
        sidecar_payload=loaded["private_sidecar"],
        sidecar_path=Path(sidecar_path),
        input_integrity_manifest=loaded["input_integrity_manifest"],
    ):
        raise ValueError("terminal outputs do not satisfy public/private cross-links")
    return _embed_artifact_hash(
        {
            "schema": HMC_IDENTITY_OUTPUT_INTEGRITY_MANIFEST_SCHEMA_V1,
            "terminal_manifest": True,
            "outputs": tuple(outputs),
        }
    )


def parse_output_integrity_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(
        payload,
        required=("schema", "terminal_manifest", "outputs", "artifact_hash"),
        label="output integrity manifest",
    )
    if payload.get("schema") != HMC_IDENTITY_OUTPUT_INTEGRITY_MANIFEST_SCHEMA_V1:
        raise ValueError("unsupported output integrity manifest schema")
    if _require_bool(payload.get("terminal_manifest"), label="terminal_manifest") is not True:
        raise ValueError("output integrity manifest must be terminal")
    outputs = payload.get("outputs")
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise ValueError("output integrity manifest outputs must be a sequence")
    roles = ("private_sidecar", "input_integrity_manifest", "public_validation_record")
    if tuple(item.get("role") for item in outputs if isinstance(item, Mapping)) != roles:
        raise ValueError("output integrity manifest roles or ordering mismatch")
    expected_schemas = {
        "private_sidecar": HMC_IDENTITY_PRIVATE_SIDECAR_SCHEMA_V1,
        "input_integrity_manifest": HMC_IDENTITY_INPUT_INTEGRITY_MANIFEST_SCHEMA_V1,
        "public_validation_record": HMC_IDENTITY_PUBLIC_VALIDATION_SCHEMA_V1,
    }
    for item in outputs:
        _require_exact_keys(
            item,
            required=(
                "role",
                "schema",
                "artifact_hash",
                "file_sha256",
                "byte_count",
            ),
            label="output integrity manifest entry",
        )
        if item["schema"] != expected_schemas[item["role"]]:
            raise ValueError("output integrity manifest role/schema mismatch")
        _require_sha256(item["artifact_hash"], label="artifact_hash", tagged=True)
        _require_sha256(item["file_sha256"], label="file_sha256", tagged=False)
        _require_int(item["byte_count"], label="byte_count", minimum=1)
    if HMC_IDENTITY_OUTPUT_INTEGRITY_MANIFEST_SCHEMA_V1 in json.dumps(
        outputs, sort_keys=True
    ):
        raise ValueError("terminal output manifest must not reference itself")
    if "output_integrity_manifest" in json.dumps(outputs, sort_keys=True):
        raise ValueError("terminal output manifest must not contain a self role")
    _verify_embedded_artifact_hash(payload, label="output integrity manifest")
    return payload


def verify_output_integrity_manifest(
    payload: Mapping[str, Any],
    *,
    sidecar_path: str | Path,
    input_manifest_path: str | Path,
    public_record_path: str | Path,
) -> Mapping[str, Any]:
    """Reopen every terminal-manifest output and verify exact bytes."""

    parse_output_integrity_manifest(payload)
    expected = build_output_integrity_manifest(
        sidecar_path=sidecar_path,
        input_manifest_path=input_manifest_path,
        public_record_path=public_record_path,
    )
    if _json_normalize(payload) != _json_normalize(expected):
        raise ValueError("output integrity manifest does not match current output bytes")
    return payload
