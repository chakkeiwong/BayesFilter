"""Fail-closed authority artifacts for the Phase 7 typed-identity smoke."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import math
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from bayesfilter.inference.hmc_identity import (
    artifact_file_sha256,
    canonical_artifact_payload_hash,
)
from bayesfilter.inference.hmc_identity_adoption import (
    HMC_PHASE5_ADOPTION_RECORD_SCHEMA_V1,
    HMC_PHASE5_OUTPUT_MANIFEST_SCHEMA_V1,
    HMC_PHASE5_PREFLIGHT_REPORT_SCHEMA_V1,
    PHASE7_CONFIG_SCHEMA_V2,
    build_phase5_artifact_reference,
    parse_phase5_artifact_reference,
    verify_phase5_artifact_reference,
)
HMC_PHASE6_SMOKE_AUTHORITY_PROPOSAL_SCHEMA_V1 = (
    "bayesfilter.hmc_phase6_smoke_authority_proposal.v1"
)
HMC_PHASE6_SMOKE_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V1 = (
    "bayesfilter.hmc_phase6_smoke_authority_proposal_manifest.v1"
)
HMC_PHASE6_SMOKE_AUTHORITY_SCHEMA_V1 = (
    "bayesfilter.hmc_phase6_smoke_authority.v1"
)
HMC_PHASE6_SMOKE_LAUNCH_CLAIM_SCHEMA_V1 = (
    "bayesfilter.hmc_phase6_smoke_launch_claim.v1"
)
HMC_PHASE6_FILE_REFERENCE_SCHEMA_V1 = "bayesfilter.hmc_phase6_file_reference.v1"
HMC_PHASE6_SMOKE_RESULT_SCHEMA_V1 = "bayesfilter.hmc_phase6_smoke_result.v1"
HMC_PHASE6_SMOKE_FAILURE_SCHEMA_V1 = "bayesfilter.hmc_phase6_smoke_failure.v1"
HMC_PHASE6_SMOKE_PROGRESS_SCHEMA_V1 = "bayesfilter.hmc_phase6_smoke_progress.v1"
HMC_PHASE6_SMOKE_OUTPUT_MANIFEST_SCHEMA_V1 = (
    "bayesfilter.hmc_phase6_smoke_output_manifest.v1"
)
HMC_PHASE6_SMOKE_INFRASTRUCTURE_FAILURE_SCHEMA_V1 = (
    "bayesfilter.hmc_phase6_smoke_infrastructure_failure.v1"
)
HMC_PHASE6_SMOKE_INFRASTRUCTURE_MANIFEST_SCHEMA_V1 = (
    "bayesfilter.hmc_phase6_smoke_infrastructure_manifest.v1"
)

SMOKE_AUTHORITY_DECISION = (
    "AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SMOKE"
)
SMOKE_AUTHORITY_STATUS_PENDING = "pending_human_smoke_approval"
SMOKE_AUTHORITY_STATUS_APPROVED = "approved_one_smoke_launch_only"
SMOKE_PASS_DECISION = (
    "PASS_PHASE7_TYPED_IDENTITY_SMOKE_MECHANICS_ONLY_"
    "STOP_BEFORE_SERIOUS_APPROVAL"
)
SMOKE_BLOCK_DECISION = "BLOCK_PHASE7_TYPED_IDENTITY_SMOKE_STOP_BEFORE_SERIOUS"
SMOKE_INFRASTRUCTURE_BLOCK_DECISION = (
    "BLOCK_PHASE7_TYPED_IDENTITY_SMOKE_INFRASTRUCTURE_STOP_BEFORE_SERIOUS"
)

TRANSITION_IDENTITY_HASH = (
    "sha256:10d9a9d2d71562d0c278b5bbc0ba0bb3eed3fc2ae77510a6d09e5c16a6f16d6a"
)
SMOKE_EXECUTION_IDENTITY_HASH = (
    "sha256:fc85f9b1e0bb406593de9f5b8195ced6e86b10ee8fd549b1ecd1a8a24d6ac604"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE6_PUBLIC_ROOT = REPO_ROOT / (
    "docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11"
)
PHASE6_SUBPLAN_PATH = REPO_ROOT / (
    "docs/plans/bayesfilter-hmc-semantic-identity-migration-"
    "phase6-smoke-subplan-2026-07-11.md"
)
V2_CONFIG_PATH = REPO_ROOT / (
    "docs/benchmarks/configs/"
    "multidim_lgssm_phase7_typed_identity_baseline_2026_07_11.json"
)
ADOPTION_RECORD_PATH = PHASE6_PUBLIC_ROOT / "typed_identity_baseline_adoption_record.json"
PREFLIGHT_PATH = PHASE6_PUBLIC_ROOT / "typed_identity_baseline_preflight.json"
PHASE5_MANIFEST_PATH = PHASE6_PUBLIC_ROOT / "phase5_output_integrity_manifest.json"
SUPERSEDED_PROPOSAL_PATH = (
    PHASE6_PUBLIC_ROOT / "phase6_smoke_authority_proposal.json"
)
SUPERSEDED_PROPOSAL_MANIFEST_PATH = PHASE6_PUBLIC_ROOT / (
    "phase6_smoke_authority_proposal_manifest.json"
)
SUPERSEDED_PROPOSAL_V2_PATH = (
    PHASE6_PUBLIC_ROOT / "phase6_smoke_authority_proposal_v2.json"
)
SUPERSEDED_PROPOSAL_MANIFEST_V2_PATH = PHASE6_PUBLIC_ROOT / (
    "phase6_smoke_authority_proposal_manifest_v2.json"
)
SUPERSEDED_AUTHORITY_PATH = PHASE6_PUBLIC_ROOT / "phase6_smoke_authority.json"
SUPERSEDED_CLAIM_PATH = PHASE6_PUBLIC_ROOT / "phase6_smoke_launch_claim.json"
SUPERSEDED_PUBLIC_RESULT_PATH = PHASE6_PUBLIC_ROOT / "phase6_smoke_result.json"
SUPERSEDED_PUBLIC_PROGRESS_PATH = PHASE6_PUBLIC_ROOT / "phase6_smoke_progress.json"
SUPERSEDED_OUTPUT_MANIFEST_PATH = (
    PHASE6_PUBLIC_ROOT / "phase6_smoke_output_manifest.json"
)
SUPERSEDED_INFRASTRUCTURE_FAILURE_PATH = PHASE6_PUBLIC_ROOT / (
    "phase6_smoke_infrastructure_failure.json"
)
SUPERSEDED_INFRASTRUCTURE_MANIFEST_PATH = PHASE6_PUBLIC_ROOT / (
    "phase6_smoke_infrastructure_manifest.json"
)
SUPERSEDED_PRIVATE_SAMPLES_PATH = REPO_ROOT / (
    "docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/"
    "private_diagnostics/phase6_typed_identity_smoke_retained_samples.npz"
)
SUPERSEDED_LOG_PATH = REPO_ROOT / (
    "docs/plans/logs/hmc-semantic-identity-migration-2026-07-11/phase6_smoke.log"
)
PROPOSAL_PATH = PHASE6_PUBLIC_ROOT / "phase6_smoke_authority_proposal_v3.json"
PROPOSAL_MANIFEST_PATH = PHASE6_PUBLIC_ROOT / (
    "phase6_smoke_authority_proposal_manifest_v3.json"
)
AUTHORITY_PATH = PHASE6_PUBLIC_ROOT / "phase6_smoke_attempt2_authority.json"
CLAIM_PATH = PHASE6_PUBLIC_ROOT / "phase6_smoke_attempt2_launch_claim.json"
PUBLIC_RESULT_PATH = PHASE6_PUBLIC_ROOT / "phase6_smoke_attempt2_result.json"
PUBLIC_PROGRESS_PATH = PHASE6_PUBLIC_ROOT / "phase6_smoke_attempt2_progress.json"
OUTPUT_MANIFEST_PATH = PHASE6_PUBLIC_ROOT / "phase6_smoke_attempt2_output_manifest.json"
INFRASTRUCTURE_FAILURE_PATH = PHASE6_PUBLIC_ROOT / (
    "phase6_smoke_attempt2_infrastructure_failure.json"
)
INFRASTRUCTURE_MANIFEST_PATH = PHASE6_PUBLIC_ROOT / (
    "phase6_smoke_attempt2_infrastructure_manifest.json"
)
PRIVATE_SAMPLES_PATH = REPO_ROOT / (
    "docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/"
    "private_diagnostics/phase6_typed_identity_smoke_attempt2_retained_samples.npz"
)
LOG_PATH = REPO_ROOT / (
    "docs/plans/logs/hmc-semantic-identity-migration-2026-07-11/"
    "phase6_smoke_attempt2.log"
)
LAUNCHER_PATH = REPO_ROOT / "scripts/run_hmc_phase6_typed_identity_smoke.py"
PROPOSAL_BUILDER_PATH = REPO_ROOT / (
    "scripts/build_hmc_phase6_smoke_authority_proposal.py"
)
AUTHORITY_BUILDER_PATH = REPO_ROOT / "scripts/build_hmc_phase6_smoke_authority.py"
CONTROLLER_PATH = REPO_ROOT / (
    "bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py"
)
AUTHORITY_MODULE_PATH = Path(__file__).resolve()
AUTHORITY_TEST_PATH = REPO_ROOT / "tests/test_hmc_smoke_authority.py"
CONTROLLER_TEST_PATH = REPO_ROOT / "tests/test_deterministic_lgssm_hmc_phase7_tf.py"
DRIVER_TEST_PATH = REPO_ROOT / "tests/test_deterministic_lgssm_hmc_tuning_driver.py"
BENCHMARK_DRIVER_PATH = REPO_ROOT / (
    "docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py"
)
CONVERGENCE_TEST_PATH = REPO_ROOT / "tests/test_hmc_convergence.py"
IDENTITY_TEST_PATH = REPO_ROOT / "tests/test_hmc_identity.py"
ADOPTION_TEST_PATH = REPO_ROOT / "tests/test_hmc_identity_adoption.py"
INTEGRATION_TEST_PATH = REPO_ROOT / "tests/test_hmc_identity_integration.py"
CERTIFICATE_TEST_PATH = REPO_ROOT / "tests/test_hmc_identity_migration_certificate.py"
_PHASE6_RUNTIME_ROOT_PATHS = (
    AUTHORITY_MODULE_PATH,
    CONTROLLER_PATH,
    BENCHMARK_DRIVER_PATH,
    LAUNCHER_PATH,
    PROPOSAL_BUILDER_PATH,
    AUTHORITY_BUILDER_PATH,
)
_PHASE6_REVIEW_TEST_PATHS = (
    IDENTITY_TEST_PATH,
    ADOPTION_TEST_PATH,
    INTEGRATION_TEST_PATH,
    CERTIFICATE_TEST_PATH,
    CONVERGENCE_TEST_PATH,
    CONTROLLER_TEST_PATH,
    DRIVER_TEST_PATH,
    AUTHORITY_TEST_PATH,
)
SMOKE_THREAD_ENVIRONMENT = {
    "TF_NUM_INTRAOP_THREADS": "8",
    "TF_NUM_INTEROP_THREADS": "1",
    "OMP_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def _consumed_attempt1_evidence_expectations() -> tuple[tuple[Any, ...], ...]:
    """Return the exact immutable evidence required before an attempt-2 action."""

    return (
        (
            "original_proposal",
            SUPERSEDED_PROPOSAL_PATH,
            193504,
            "16df0bdb62f45e9b2c304a7030c5c7d08497720f42c43dbf489b694dc9497d0d",
            0o600,
        ),
        (
            "original_proposal_manifest",
            SUPERSEDED_PROPOSAL_MANIFEST_PATH,
            848,
            "b31d93a568bd30458c56bc87d9eca17ea73ea3579f973591e00d0a9a80696c3c",
            0o600,
        ),
        (
            "attempt1_proposal",
            SUPERSEDED_PROPOSAL_V2_PATH,
            30416,
            "f8c1d301186e9b1df390dbc4248c95932737bf2a7d8f50c6af985129bc7755c8",
            0o600,
        ),
        (
            "attempt1_proposal_manifest",
            SUPERSEDED_PROPOSAL_MANIFEST_V2_PATH,
            847,
            "29dbba924ce899189e178d624ddc26c1fdfaaf46674244c3547f44c7ee591527",
            0o600,
        ),
        (
            "attempt1_authority",
            SUPERSEDED_AUTHORITY_PATH,
            1712,
            "e6be84a875ded5b880eef7d7445e645aefdf86c61dcbc5b4ad744d6d1bec126c",
            0o600,
        ),
        (
            "attempt1_claim",
            SUPERSEDED_CLAIM_PATH,
            1886,
            "d1424c0cf4bc616bcab1de7efda29a9f0c465d0496cb8c1eb6d720faba8d54d3",
            0o400,
        ),
        (
            "attempt1_result",
            SUPERSEDED_PUBLIC_RESULT_PATH,
            5668,
            "28f7866d6e2fc1419a010b70a9b9e4f9f45da3f2c7c0f259c48b47fc9bf09fe9",
            0o400,
        ),
        (
            "attempt1_progress",
            SUPERSEDED_PUBLIC_PROGRESS_PATH,
            949,
            "55a973c4df278ce137793df767ff80864f17e5c90a130c3a59b4fa2cecfed24c",
            0o400,
        ),
        (
            "attempt1_output_manifest",
            SUPERSEDED_OUTPUT_MANIFEST_PATH,
            3836,
            "c144596ac8f0d2be33e6cf65d4c60d64cc2703b0990ee37005707c6a4f773900",
            0o400,
        ),
        (
            "attempt1_infrastructure_failure_reservation",
            SUPERSEDED_INFRASTRUCTURE_FAILURE_PATH,
            0,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            0o400,
        ),
        (
            "attempt1_infrastructure_manifest_reservation",
            SUPERSEDED_INFRASTRUCTURE_MANIFEST_PATH,
            0,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            0o400,
        ),
        (
            "attempt1_private_sample_reservation",
            SUPERSEDED_PRIVATE_SAMPLES_PATH,
            0,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            0o400,
        ),
        (
            "attempt1_log",
            SUPERSEDED_LOG_PATH,
            2564,
            "6dee7ec170811c18c87fc1ee3fa0397213325363a5c1e4e2c294874cc5e7bf80",
            0o400,
        ),
    )


def _repository_file_role(path: Path) -> str:
    return f"repository_file:{path.relative_to(REPO_ROOT).as_posix()}"


def _resolve_bayesfilter_module_paths(module_name: str) -> tuple[Path, ...]:
    """Resolve one static BayesFilter import plus its package initializers."""

    if module_name != "bayesfilter" and not module_name.startswith("bayesfilter."):
        return ()
    parts = module_name.split(".")
    resolved: list[Path] = []
    for index in range(1, len(parts)):
        package_init = REPO_ROOT.joinpath(*parts[:index], "__init__.py")
        if package_init.is_file():
            resolved.append(package_init.resolve())
    module_path = REPO_ROOT.joinpath(*parts).with_suffix(".py")
    package_path = REPO_ROOT.joinpath(*parts, "__init__.py")
    if module_path.is_file():
        resolved.append(module_path.resolve())
    elif package_path.is_file():
        resolved.append(package_path.resolve())
    return tuple(dict.fromkeys(resolved))


def _static_bayesfilter_imports(path: Path) -> tuple[str, ...]:
    """Return statically named BayesFilter modules from every code branch."""

    tree = ast.parse(path.read_bytes(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom) or node.level != 0 or not node.module:
            continue
        modules.add(node.module)
        # ``from package import module`` may load a submodule even when the
        # package initializer does not expose it eagerly.
        for alias in node.names:
            if alias.name != "*":
                candidate = f"{node.module}.{alias.name}"
                if _resolve_bayesfilter_module_paths(candidate):
                    modules.add(candidate)
    return tuple(
        sorted(
            name
            for name in modules
            if name == "bayesfilter" or name.startswith("bayesfilter.")
        )
    )


def _phase6_runtime_source_closure() -> tuple[Path, ...]:
    """Close the exact Phase 6 runtime roots over static project imports."""

    pending = list(_PHASE6_RUNTIME_ROOT_PATHS)
    closed: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if path in closed:
            continue
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Phase 6 runtime source is not a real file: {path}")
        if REPO_ROOT not in path.parents:
            raise ValueError(f"Phase 6 runtime source escapes repository: {path}")
        closed.add(path)
        if path.suffix != ".py":
            continue
        for module_name in _static_bayesfilter_imports(path):
            for imported_path in _resolve_bayesfilter_module_paths(module_name):
                if imported_path not in closed:
                    pending.append(imported_path)
    return tuple(sorted(closed))


def default_implementation_paths(
    python_executable: str | Path,
) -> Mapping[str, Path]:
    # Other agents may add unrelated repository code while this lane is
    # awaiting approval. Bind the reviewed runtime closure and exact review
    # files, not the mutable repository-wide Python namespace.
    sources = (*_phase6_runtime_source_closure(), *_PHASE6_REVIEW_TEST_PATHS)
    inventory: dict[str, Path] = {}
    for path in sources:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Phase 6 implementation source is not a real file: {path}")
        resolved = path.resolve()
        if REPO_ROOT not in resolved.parents:
            raise ValueError(f"Phase 6 implementation source escapes repository: {path}")
        role = _repository_file_role(resolved)
        if role in inventory and inventory[role] != resolved:
            raise ValueError(f"duplicate Phase 6 implementation role: {role}")
        inventory[role] = resolved
    inventory["python_executable"] = Path(python_executable).resolve()
    return dict(sorted(inventory.items()))


def implementation_reference_roles(
    python_executable: str | Path,
    *,
    implementation_paths: Mapping[str, str | Path] | None = None,
) -> tuple[str, ...]:
    paths = (
        default_implementation_paths(python_executable)
        if implementation_paths is None
        else implementation_paths
    )
    return tuple(sorted(paths))


@dataclass(frozen=True)
class Phase6SmokeLaunchContext:
    config: Any
    preflight: Mapping[str, Any]
    proposal: Mapping[str, Any]
    proposal_manifest: Mapping[str, Any]
    authority: Mapping[str, Any]
    authority_reference: Mapping[str, Any]
    claim: Mapping[str, Any]
    paths: Mapping[str, Path]
    command: tuple[str, ...]
    implementation_source_bundle: Mapping[str, bytes]
    output_directories: Any
    claim_fd: int
    consumed_evidence_session: Any | None = None
    output_session: Any | None = None
    prepared_snapshot_hash: str | None = None
    _prepared_token: Any | None = None


_PREPARED_CONTEXT_TOKENS: dict[int, object] = {}
_PREPARED_CONTEXT_EVIDENCE_SESSIONS: dict[int, Any] = {}

SMOKE_NONCLAIMS = (
    "tiny actual-target HMC mechanics smoke executed",
    "historical typed identity equality unsupported",
    "serious convergence thresholds not evaluated",
    "not serious Phase 7, Phase 8, or NeuTra execution authority",
    "not convergence, recovery, production, default, GPU, or scientific evidence",
)
SMOKE_FAILURE_NONCLAIMS = (
    "tiny actual-target HMC mechanics smoke attempted after authority consumption",
    "historical typed identity equality unsupported",
    "serious convergence thresholds not evaluated",
    "not serious Phase 7, Phase 8, or NeuTra execution authority",
    "not evidence against the target, math, or scientific direction without a classified validity failure",
)
SMOKE_INFRASTRUCTURE_FAILURE_NONCLAIMS = (
    "smoke authority was consumed before a launcher infrastructure failure",
    "any valid primary controller result is preserved without overwrite",
    "historical typed identity equality unsupported",
    "not serious Phase 7, Phase 8, or NeuTra execution authority",
    "not convergence, recovery, production, default, GPU, or scientific evidence",
)
PROPOSAL_NONCLAIMS = (
    "proposal and local no-runtime evidence only",
    "human smoke approval not yet recorded",
    "not HMC transition, worker, smoke, burn-in, or sampling authority",
    "not serious Phase 7, Phase 8, or NeuTra authority",
    "not convergence, recovery, production, default, GPU, or scientific evidence",
)
AUTHORITY_NONCLAIMS = (
    "one tiny actual-target two-worker CPU/XLA HMC mechanics smoke launch only",
    "authority is consumed permanently before log, output, or worker creation",
    "historical typed identity equality unsupported",
    "not serious Phase 7, Phase 8, or NeuTra authority",
    "not convergence, recovery, production, default, GPU, or scientific evidence",
)

_HEX = frozenset("0123456789abcdef")
_ARTIFACT_REFERENCE_SCHEMAS = {
    "v2_config_reference": PHASE7_CONFIG_SCHEMA_V2,
    "adoption_record_reference": HMC_PHASE5_ADOPTION_RECORD_SCHEMA_V1,
    "preflight_reference": HMC_PHASE5_PREFLIGHT_REPORT_SCHEMA_V1,
    "phase5_manifest_reference": HMC_PHASE5_OUTPUT_MANIFEST_SCHEMA_V1,
}
_RUNTIME_FIELDS = (
    "mode",
    "worker_count",
    "chains_per_worker",
    "chain_count",
    "burnin_results_per_chain",
    "retained_results_per_chain",
    "cuda_visible_devices",
    "dtype",
    "jit_compile",
    "use_xla",
    "compile_workers_sequentially",
    "root_seed",
    "tensorflow_version",
    "tfp_version",
    "python_version",
    "thread_environment",
    "wall_time_cap_seconds",
)
_PATH_FIELDS = (
    "claim_path",
    "log_path",
    "public_result_path",
    "public_progress_path",
    "output_manifest_path",
    "infrastructure_failure_path",
    "infrastructure_manifest_path",
    "private_samples_path",
)


def _require_exact_keys(
    payload: Mapping[str, Any], *, required: Sequence[str], label: str
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


def _require_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonblank trimmed string")
    return value


def _require_iso_date(value: Any, *, label: str) -> str:
    text = _require_string(value, label=label)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{label} must use YYYY-MM-DD") from error
    if parsed.isoformat() != text:
        raise ValueError(f"{label} must use YYYY-MM-DD")
    return text


def _require_sha256(value: Any, *, label: str, tagged: bool = True) -> str:
    text = _require_string(value, label=label)
    digest = text.removeprefix("sha256:") if tagged else text
    if tagged != text.startswith("sha256:"):
        raise ValueError(f"{label} SHA-256 prefix mismatch")
    if len(digest) != 64 or any(char not in _HEX for char in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _require_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
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
    observed = _require_sha256(payload.get("artifact_hash"), label=f"{label} hash")
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


def _path_hash(path: str | Path) -> str:
    return hashlib.sha256(str(Path(path).resolve()).encode("utf-8")).hexdigest()


def build_file_reference(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "schema": HMC_PHASE6_FILE_REFERENCE_SCHEMA_V1,
        "resolved_path_sha256": _path_hash(source),
        "file_sha256": artifact_file_sha256(source),
        "byte_count": source.stat().st_size,
    }


def parse_file_reference(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(
        payload,
        required=("schema", "resolved_path_sha256", "file_sha256", "byte_count"),
        label="Phase 6 file reference",
    )
    if payload.get("schema") != HMC_PHASE6_FILE_REFERENCE_SCHEMA_V1:
        raise ValueError("unsupported Phase 6 file reference schema")
    _require_sha256(
        payload.get("resolved_path_sha256"),
        label="resolved_path_sha256",
        tagged=False,
    )
    _require_sha256(payload.get("file_sha256"), label="file_sha256", tagged=False)
    _require_nonnegative_int(payload.get("byte_count"), label="byte_count")
    return payload


def verify_file_reference(
    payload: Mapping[str, Any], *, path: str | Path
) -> Mapping[str, Any]:
    parse_file_reference(payload)
    if dict(payload) != dict(build_file_reference(path)):
        raise ValueError("Phase 6 file reference does not match current bytes")
    return payload


def verify_artifact_reference_snapshot(
    reference: Mapping[str, Any], *, snapshot: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Verify a Phase 5 reference against the exact bytes already consumed."""

    from bayesfilter.runtime import stable_config_hash

    parse_phase5_artifact_reference(reference)
    _require_exact_keys(
        snapshot,
        required=("path", "payload", "file_sha256", "byte_count"),
        label="governed source snapshot",
    )
    path = Path(snapshot["path"])
    payload = snapshot["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("governed source snapshot payload must be a mapping")
    rule = reference["embedded_hash_rule"]
    embedded = payload.get("artifact_hash")
    if rule == "none":
        expected_embedded = None
    elif rule == "stable_without_hash":
        expected_embedded = "sha256:" + stable_config_hash(
            {key: value for key, value in payload.items() if key != "artifact_hash"}
        )
    elif rule == "canonical_without_hash":
        expected_embedded = canonical_artifact_payload_hash(
            {key: value for key, value in payload.items() if key != "artifact_hash"}
        )
    else:
        raise ValueError("unsupported governed snapshot hash rule")
    expected = {
        "schema": reference["schema"],
        "source_schema": payload.get("schema"),
        "embedded_hash_rule": rule,
        "embedded_artifact_hash": expected_embedded,
        "canonical_payload_hash": canonical_artifact_payload_hash(payload),
        "resolved_path_sha256": _path_hash(path),
        "file_sha256": snapshot["file_sha256"],
        "byte_count": snapshot["byte_count"],
    }
    if dict(reference) != expected or embedded != expected_embedded:
        raise ValueError("governed source snapshot reference mismatch")
    return reference


def _strict_runtime(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(payload, required=_RUNTIME_FIELDS, label="smoke runtime")
    expected_scalars = {
        "mode": "smoke",
        "worker_count": 2,
        "chains_per_worker": 2,
        "chain_count": 4,
        "burnin_results_per_chain": 4,
        "retained_results_per_chain": 8,
        "cuda_visible_devices": "-1",
        "dtype": "float64",
        "jit_compile": True,
        "use_xla": True,
        "compile_workers_sequentially": True,
        "root_seed": [20260711, 701],
        "wall_time_cap_seconds": 28800,
    }
    for name, expected in expected_scalars.items():
        if payload.get(name) != expected:
            raise ValueError(f"smoke runtime {name} mismatch")
    for name in ("tensorflow_version", "tfp_version", "python_version"):
        _require_string(payload.get(name), label=name)
    threads = payload.get("thread_environment")
    expected_threads = {
        "TF_NUM_INTRAOP_THREADS": "8",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    if threads != expected_threads:
        raise ValueError("smoke runtime thread environment mismatch")
    return payload


def _strict_paths(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(payload, required=_PATH_FIELDS, label="smoke paths")
    paths = tuple(_require_string(payload[name], label=name) for name in _PATH_FIELDS)
    if len(set(paths)) != len(paths):
        raise ValueError("smoke paths must be distinct")
    if any(Path(value).is_absolute() for value in paths):
        raise ValueError("smoke paths must be repository-relative")
    if not payload["private_samples_path"].startswith(
        "docs/benchmarks/artifacts/"
    ) or "/private_diagnostics/" not in payload["private_samples_path"]:
        raise ValueError("smoke private samples path is outside protected storage")
    for name in (
        "claim_path",
        "log_path",
        "public_result_path",
        "public_progress_path",
        "output_manifest_path",
        "infrastructure_failure_path",
        "infrastructure_manifest_path",
    ):
        if not payload[name].startswith("docs/plans/"):
            raise ValueError(f"{name} must remain under docs/plans")
    for value in paths:
        if "\\" in value or any(part in {"", ".", ".."} for part in value.split("/")):
            raise ValueError("smoke paths must use normalized repository components")
    return payload


def _parse_command(command: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        raise ValueError("smoke command must be a sequence")
    normalized = tuple(
        _require_string(item, label="smoke command item") for item in command
    )
    expected_tail = (
        "scripts/run_hmc_phase6_typed_identity_smoke.py",
        "--stage",
        "burnin_sampling",
        "--phase7-smoke",
        "--phase7-smoke-authority",
    )
    if len(normalized) != 7 or normalized[1:6] != expected_tail:
        raise ValueError("smoke command shape mismatch")
    if str(Path(normalized[0]).resolve()) != normalized[0]:
        raise ValueError("smoke Python executable must be an absolute resolved path")
    authority_path = normalized[6]
    authority_root = (
        "docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/"
    )
    authority_name = authority_path.removeprefix(authority_root)
    if (
        not authority_path.startswith(authority_root)
        or "/" in authority_name
        or not authority_name.startswith("phase6_smoke_")
        or not authority_name.endswith("authority.json")
        or "\\" in authority_path
    ):
        raise ValueError("smoke authority command path mismatch")
    return normalized


def _strict_command(command: Sequence[str]) -> tuple[str, ...]:
    """Require the exact currently reviewed attempt command."""

    normalized = _parse_command(command)
    if normalized != expected_launcher_command(normalized[0]):
        raise ValueError("smoke command differs from the reviewed command")
    return normalized


def default_smoke_paths() -> Mapping[str, str]:
    return {
        "claim_path": str(CLAIM_PATH.relative_to(REPO_ROOT)),
        "log_path": str(LOG_PATH.relative_to(REPO_ROOT)),
        "public_result_path": str(PUBLIC_RESULT_PATH.relative_to(REPO_ROOT)),
        "public_progress_path": str(PUBLIC_PROGRESS_PATH.relative_to(REPO_ROOT)),
        "output_manifest_path": str(OUTPUT_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "infrastructure_failure_path": str(
            INFRASTRUCTURE_FAILURE_PATH.relative_to(REPO_ROOT)
        ),
        "infrastructure_manifest_path": str(
            INFRASTRUCTURE_MANIFEST_PATH.relative_to(REPO_ROOT)
        ),
        "private_samples_path": str(PRIVATE_SAMPLES_PATH.relative_to(REPO_ROOT)),
    }


def default_smoke_runtime() -> Mapping[str, Any]:
    _require_smoke_parent_environment()
    import platform

    import tensorflow as tf
    import tensorflow_probability as tfp

    return {
        "mode": "smoke",
        "worker_count": 2,
        "chains_per_worker": 2,
        "chain_count": 4,
        "burnin_results_per_chain": 4,
        "retained_results_per_chain": 8,
        "cuda_visible_devices": "-1",
        "dtype": "float64",
        "jit_compile": True,
        "use_xla": True,
        "compile_workers_sequentially": True,
        "root_seed": [20260711, 701],
        "tensorflow_version": tf.__version__,
        "tfp_version": tfp.__version__,
        "python_version": platform.python_version(),
        "thread_environment": {
            name: os.environ.get(name) for name in SMOKE_THREAD_ENVIRONMENT
        },
        "wall_time_cap_seconds": 28800,
    }


def _require_smoke_parent_environment() -> Mapping[str, str]:
    observed = {name: os.environ.get(name) for name in SMOKE_THREAD_ENVIRONMENT}
    if observed != SMOKE_THREAD_ENVIRONMENT:
        raise ValueError(
            "smoke parent thread environment mismatch before framework initialization"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("smoke parent requires CUDA_VISIBLE_DEVICES=-1")
    return dict(SMOKE_THREAD_ENVIRONMENT)


def expected_launcher_command(python_executable: str | Path) -> tuple[str, ...]:
    return (
        str(Path(python_executable).resolve()),
        str(LAUNCHER_PATH.relative_to(REPO_ROOT)),
        "--stage",
        "burnin_sampling",
        "--phase7-smoke",
        "--phase7-smoke-authority",
        str(AUTHORITY_PATH.relative_to(REPO_ROOT)),
    )


def build_default_smoke_authority_proposal(
    *, python_executable: str | Path
) -> Mapping[str, Any]:
    return build_smoke_authority_proposal(
        phase6_subplan_path=PHASE6_SUBPLAN_PATH,
        v2_config_path=V2_CONFIG_PATH,
        adoption_record_path=ADOPTION_RECORD_PATH,
        preflight_path=PREFLIGHT_PATH,
        phase5_manifest_path=PHASE5_MANIFEST_PATH,
        runtime=default_smoke_runtime(),
        paths=default_smoke_paths(),
        command=expected_launcher_command(python_executable),
        implementation_references={
            name: build_file_reference(path)
            for name, path in default_implementation_paths(
                python_executable
            ).items()
        },
    )


def build_smoke_authority_proposal(
    *,
    phase6_subplan_path: str | Path,
    v2_config_path: str | Path,
    adoption_record_path: str | Path,
    preflight_path: str | Path,
    phase5_manifest_path: str | Path,
    runtime: Mapping[str, Any],
    paths: Mapping[str, Any],
    command: Sequence[str],
    implementation_references: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Build the pending, runtime-inert Phase 6 authority proposal."""

    _strict_runtime(runtime)
    _strict_paths(paths)
    command_tuple = _strict_command(command)
    refs = {
        "v2_config_reference": build_phase5_artifact_reference(
            v2_config_path, embedded_hash_rule="canonical_without_hash"
        ),
        "adoption_record_reference": build_phase5_artifact_reference(
            adoption_record_path, embedded_hash_rule="canonical_without_hash"
        ),
        "preflight_reference": build_phase5_artifact_reference(
            preflight_path, embedded_hash_rule="canonical_without_hash"
        ),
        "phase5_manifest_reference": build_phase5_artifact_reference(
            phase5_manifest_path, embedded_hash_rule="canonical_without_hash"
        ),
    }
    expected_roles = implementation_reference_roles(command_tuple[0])
    if tuple(sorted(implementation_references)) != expected_roles:
        raise ValueError("smoke proposal implementation roles mismatch")
    implementation = {
        name: dict(reference)
        for name, reference in sorted(implementation_references.items())
    }
    return _embed_hash(
        {
            "schema": HMC_PHASE6_SMOKE_AUTHORITY_PROPOSAL_SCHEMA_V1,
            "status": SMOKE_AUTHORITY_STATUS_PENDING,
            "decision": SMOKE_AUTHORITY_DECISION,
            "phase6_subplan_reference": {
                "path_sha256": _path_hash(phase6_subplan_path),
                "file_sha256": artifact_file_sha256(phase6_subplan_path),
                "byte_count": Path(phase6_subplan_path).stat().st_size,
            },
            **refs,
            "transition_identity_hash": TRANSITION_IDENTITY_HASH,
            "smoke_execution_identity_hash": SMOKE_EXECUTION_IDENTITY_HASH,
            "runtime": dict(runtime),
            "paths": dict(paths),
            "command": command_tuple,
            "implementation_references": implementation,
            "serious_runtime_authority": False,
            "phase8_authority": False,
            "neutra_authority": False,
            "nonclaims": PROPOSAL_NONCLAIMS,
        }
    )


def parse_smoke_authority_proposal(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = (
        "schema",
        "status",
        "decision",
        "phase6_subplan_reference",
        *_ARTIFACT_REFERENCE_SCHEMAS,
        "transition_identity_hash",
        "smoke_execution_identity_hash",
        "runtime",
        "paths",
        "command",
        "implementation_references",
        "serious_runtime_authority",
        "phase8_authority",
        "neutra_authority",
        "nonclaims",
        "artifact_hash",
    )
    _require_exact_keys(payload, required=fields, label="smoke authority proposal")
    if payload.get("schema") != HMC_PHASE6_SMOKE_AUTHORITY_PROPOSAL_SCHEMA_V1:
        raise ValueError("unsupported smoke authority proposal schema")
    if payload.get("status") != SMOKE_AUTHORITY_STATUS_PENDING:
        raise ValueError("smoke proposal must remain pending")
    if payload.get("decision") != SMOKE_AUTHORITY_DECISION:
        raise ValueError("smoke proposal decision mismatch")
    subplan = payload.get("phase6_subplan_reference")
    _require_exact_keys(
        subplan,
        required=("path_sha256", "file_sha256", "byte_count"),
        label="Phase 6 subplan reference",
    )
    _require_sha256(subplan["path_sha256"], label="subplan path hash", tagged=False)
    _require_sha256(subplan["file_sha256"], label="subplan file hash", tagged=False)
    _require_positive_int(subplan["byte_count"], label="subplan byte_count")
    for name, schema in _ARTIFACT_REFERENCE_SCHEMAS.items():
        reference = parse_phase5_artifact_reference(payload[name])
        if reference["source_schema"] != schema:
            raise ValueError(f"smoke proposal {name} schema mismatch")
    if payload.get("transition_identity_hash") != TRANSITION_IDENTITY_HASH:
        raise ValueError("smoke proposal transition identity mismatch")
    if payload.get("smoke_execution_identity_hash") != SMOKE_EXECUTION_IDENTITY_HASH:
        raise ValueError("smoke proposal execution identity mismatch")
    _strict_runtime(payload["runtime"])
    _strict_paths(payload["paths"])
    command = _parse_command(payload.get("command"))
    implementation = payload.get("implementation_references")
    if not isinstance(implementation, Mapping) or not implementation:
        raise ValueError("smoke proposal implementation references must be a mapping")
    if "python_executable" not in implementation:
        raise ValueError("smoke proposal Python executable reference is missing")
    for name, reference in implementation.items():
        _require_string(name, label="implementation role")
        parse_file_reference(reference)
        if name == "python_executable":
            expected_path = Path(command[0])
        elif name.startswith("repository_file:"):
            relative = name.removeprefix("repository_file:")
            if (
                not relative.endswith(".py")
                or "\\" in relative
                or any(part in {"", ".", ".."} for part in relative.split("/"))
            ):
                raise ValueError("smoke proposal repository role is invalid")
            expected_path = REPO_ROOT / relative
        else:
            raise ValueError("smoke proposal implementation role is invalid")
        if reference["resolved_path_sha256"] != _path_hash(expected_path):
            raise ValueError("smoke proposal implementation role/path mismatch")
    for name in ("serious_runtime_authority", "phase8_authority", "neutra_authority"):
        if _require_bool(payload.get(name), label=name):
            raise ValueError(f"smoke proposal cannot grant {name}")
    _require_ordered(payload.get("nonclaims"), expected=PROPOSAL_NONCLAIMS, label="proposal nonclaims")
    _verify_hash(payload, label="smoke authority proposal")
    return payload


def verify_smoke_authority_proposal(
    payload: Mapping[str, Any],
    *,
    phase6_subplan_path: str | Path,
    artifact_paths: Mapping[str, str | Path],
    implementation_paths: Mapping[str, str | Path] | None,
) -> Mapping[str, Any]:
    parse_smoke_authority_proposal(payload)
    if payload["runtime"] != default_smoke_runtime():
        raise ValueError("smoke proposal runtime differs from the live environment")
    if payload["paths"] != default_smoke_paths():
        raise ValueError("smoke proposal paths differ from the reviewed paths")
    if tuple(payload["command"]) != expected_launcher_command(
        payload["command"][0]
    ):
        raise ValueError("smoke proposal command differs from the reviewed command")
    subplan = payload["phase6_subplan_reference"]
    if subplan != {
        "path_sha256": _path_hash(phase6_subplan_path),
        "file_sha256": artifact_file_sha256(phase6_subplan_path),
        "byte_count": Path(phase6_subplan_path).stat().st_size,
    }:
        raise ValueError("smoke proposal subplan reference mismatch")
    for name in _ARTIFACT_REFERENCE_SCHEMAS:
        verify_phase5_artifact_reference(payload[name], path=artifact_paths[name])
    resolved_implementation_paths = (
        default_implementation_paths(payload["command"][0])
        if implementation_paths is None
        else implementation_paths
    )
    verify_implementation_reference_inventory(
        payload["implementation_references"],
        python_executable=payload["command"][0],
        implementation_paths=resolved_implementation_paths,
    )
    if Path(resolved_implementation_paths["python_executable"]).resolve() != Path(
        payload["command"][0]
    ).resolve():
        raise ValueError("smoke proposal Python executable reference mismatch")
    return payload


def verify_implementation_reference_inventory(
    references: Mapping[str, Mapping[str, Any]],
    *,
    python_executable: str | Path,
    implementation_paths: Mapping[str, str | Path] | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Verify the closed project-owned Python/runtime source inventory."""

    paths = (
        default_implementation_paths(python_executable)
        if implementation_paths is None
        else dict(implementation_paths)
    )
    if tuple(sorted(paths)) != implementation_reference_roles(
        python_executable,
        implementation_paths=paths,
    ) or tuple(sorted(references)) != tuple(sorted(paths)):
        raise ValueError("smoke proposal implementation roles mismatch")
    for name, path in paths.items():
        verify_file_reference(references[name], path=path)
    return references


def _implementation_source_roles(
    python_executable: str | Path,
    *,
    implementation_paths: Mapping[str, str | Path] | None = None,
) -> tuple[str, ...]:
    return tuple(
        role
        for role in implementation_reference_roles(
            python_executable,
            implementation_paths=implementation_paths,
        )
        if role.startswith("repository_file:")
    )


def implementation_source_bundle_hash(
    source_bundle: Mapping[str, bytes],
) -> str:
    """Hash role-delimited source bytes captured for child imports."""

    digest = hashlib.sha256()
    for role, source in sorted(source_bundle.items()):
        if not isinstance(role, str) or not isinstance(source, bytes):
            raise TypeError("implementation source bundle must map roles to bytes")
        role_bytes = role.encode("utf-8")
        digest.update(len(role_bytes).to_bytes(8, "big"))
        digest.update(role_bytes)
        digest.update(len(source).to_bytes(8, "big"))
        digest.update(source)
    return "sha256:" + digest.hexdigest()


def verify_implementation_source_bundle(
    references: Mapping[str, Mapping[str, Any]],
    source_bundle: Mapping[str, bytes],
    *,
    python_executable: str | Path,
    implementation_paths: Mapping[str, str | Path] | None = None,
) -> Mapping[str, bytes]:
    """Verify the exact source bytes that an isolated child importer will load."""

    expected_roles = _implementation_source_roles(
        python_executable,
        implementation_paths=implementation_paths,
    )
    if tuple(sorted(source_bundle)) != expected_roles:
        raise ValueError("smoke child implementation source roles mismatch")
    for role in expected_roles:
        reference = references.get(role)
        if not isinstance(reference, Mapping):
            raise ValueError("smoke child implementation reference is missing")
        parse_file_reference(reference)
        source = source_bundle[role]
        if not isinstance(source, bytes):
            raise TypeError("smoke child implementation source must be bytes")
        if hashlib.sha256(source).hexdigest() != reference["file_sha256"] or len(
            source
        ) != reference["byte_count"]:
            raise ValueError("smoke child implementation source bytes mismatch")
    implementation_source_bundle_hash(source_bundle)
    return source_bundle


def build_verified_implementation_source_bundle(
    references: Mapping[str, Mapping[str, Any]],
    *,
    python_executable: str | Path,
    implementation_paths: Mapping[str, str | Path] | None = None,
) -> Mapping[str, bytes]:
    """Capture approved package bytes once so children never import pathname bytes."""

    paths = (
        default_implementation_paths(python_executable)
        if implementation_paths is None
        else dict(implementation_paths)
    )
    verify_implementation_reference_inventory(
        references,
        python_executable=python_executable,
        implementation_paths=paths,
    )
    source_roles = _implementation_source_roles(
        python_executable,
        implementation_paths=paths,
    )
    bundle: dict[str, bytes] = {}
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    for role in source_roles:
        path = paths[role]
        fd = os.open(path, flags)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("smoke implementation source is not a regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise RuntimeError("smoke implementation source changed during capture")
        source = b"".join(chunks)
        reference = references[role]
        if (
            hashlib.sha256(source).hexdigest() != reference["file_sha256"]
            or len(source) != reference["byte_count"]
            or _path_hash(path) != reference["resolved_path_sha256"]
        ):
            raise ValueError("smoke implementation source changed before capture")
        bundle[role] = source
    return verify_implementation_source_bundle(
        references,
        bundle,
        python_executable=python_executable,
        implementation_paths=paths,
    )


CHILD_SOURCE_LOADER_BOOTSTRAP = r'''
import builtins as _bf_builtins
import hashlib as _bf_hashlib
import importlib.abc as _bf_importlib_abc
import importlib.util as _bf_importlib_util
import os as _bf_os
from pathlib import Path as _BFPath
import sys as _bf_sys

for _bf_name, _bf_value in _worker_environment.items():
    _bf_os.environ[str(_bf_name)] = str(_bf_value)
_bf_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def _bf_bundle_hash(_bf_bundle):
    _bf_digest = _bf_hashlib.sha256()
    for _bf_role, _bf_source in sorted(_bf_bundle.items()):
        _bf_role_bytes = _bf_role.encode("utf-8")
        _bf_digest.update(len(_bf_role_bytes).to_bytes(8, "big"))
        _bf_digest.update(_bf_role_bytes)
        _bf_digest.update(len(_bf_source).to_bytes(8, "big"))
        _bf_digest.update(_bf_source)
    return "sha256:" + _bf_digest.hexdigest()

_bf_expected_roles = tuple(
    _bf_role
    for _bf_role in sorted(_implementation_references)
    if _bf_role.startswith("repository_file:")
)
if tuple(sorted(_implementation_source_bundle)) != _bf_expected_roles:
    raise RuntimeError("Phase 6 child source bundle roles mismatch")
if _bf_bundle_hash(_implementation_source_bundle) != _implementation_source_bundle_hash:
    raise RuntimeError("Phase 6 child source bundle hash mismatch")

_bf_module_sources = {}
for _bf_role in _bf_expected_roles:
    _bf_source = _implementation_source_bundle[_bf_role]
    _bf_reference = _implementation_references[_bf_role]
    if (
        _bf_hashlib.sha256(_bf_source).hexdigest() != _bf_reference["file_sha256"]
        or len(_bf_source) != _bf_reference["byte_count"]
    ):
        raise RuntimeError("Phase 6 child source reference mismatch")
    _bf_relative = _bf_role.removeprefix("repository_file:")
    _bf_path = _BFPath(_approved_repository_root) / _bf_relative
    _bf_live = _bf_path.read_bytes()
    if (
        _bf_hashlib.sha256(_bf_live).hexdigest() != _bf_reference["file_sha256"]
        or len(_bf_live) != _bf_reference["byte_count"]
        or _bf_hashlib.sha256(str(_bf_path.resolve()).encode("utf-8")).hexdigest()
        != _bf_reference["resolved_path_sha256"]
    ):
        raise RuntimeError("Phase 6 live child source mismatch")
    if _bf_relative.startswith("bayesfilter/"):
        _bf_module_relative = _bf_relative[:-3]
        _bf_is_package = _bf_module_relative.endswith("/__init__")
        if _bf_is_package:
            _bf_module_relative = _bf_module_relative[:-len("/__init__")]
        _bf_module_name = _bf_module_relative.replace("/", ".")
        _bf_module_sources[_bf_module_name] = (
            _bf_role,
            _bf_source,
            str(_bf_path),
            _bf_is_package,
        )
    elif _bf_relative == _benchmark_driver_relative_path:
        _bf_module_sources[_benchmark_driver_module] = (
            _bf_role,
            _bf_source,
            str(_bf_path),
            False,
        )

_bf_synthetic_namespaces = ("docs", "docs.benchmarks")

class _BayesFilterPhase6NamespaceLoader(_bf_importlib_abc.Loader):
    def __init__(self, fullname):
        self.fullname = fullname

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__phase6_synthetic_namespace__ = self.fullname
        module.__phase6_source_bundle_hash__ = _implementation_source_bundle_hash

class _BayesFilterPhase6SourceLoader(_bf_importlib_abc.Loader):
    def __init__(self, fullname, entry):
        self.fullname = fullname
        self.role, self.source, self.filename, self.is_package = entry

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = self.filename
        module.__phase6_source_role__ = self.role
        module.__phase6_source_sha256__ = _bf_hashlib.sha256(self.source).hexdigest()
        module.__phase6_source_bundle_hash__ = _implementation_source_bundle_hash
        _bf_code = compile(self.source, self.filename, "exec", dont_inherit=True)
        exec(_bf_code, module.__dict__)

class _BayesFilterPhase6SourceFinder(_bf_importlib_abc.MetaPathFinder):
    phase6_source_bundle_hash = _implementation_source_bundle_hash

    def find_spec(self, fullname, path=None, target=None):
        if fullname in _bf_synthetic_namespaces:
            _bf_loader = _BayesFilterPhase6NamespaceLoader(fullname)
            return _bf_importlib_util.spec_from_loader(
                fullname,
                _bf_loader,
                origin="phase6-synthetic-namespace",
                is_package=True,
            )
        _bf_entry = _bf_module_sources.get(fullname)
        if _bf_entry is not None:
            _bf_loader = _BayesFilterPhase6SourceLoader(fullname, _bf_entry)
            return _bf_importlib_util.spec_from_loader(
                fullname,
                _bf_loader,
                origin=_bf_entry[2],
                is_package=_bf_entry[3],
            )
        if fullname == "bayesfilter" or fullname.startswith("bayesfilter."):
            raise ImportError("unapproved BayesFilter module outside Phase 6 source bundle")
        if fullname == "docs" or fullname.startswith("docs."):
            raise ImportError("unapproved docs module outside Phase 6 source bundle")
        return None

_bf_finder = _BayesFilterPhase6SourceFinder()
for _bf_loaded_name in tuple(_bf_sys.modules):
    if (
        _bf_loaded_name == "bayesfilter"
        or _bf_loaded_name.startswith("bayesfilter.")
        or _bf_loaded_name == "docs"
        or _bf_loaded_name.startswith("docs.")
    ):
        del _bf_sys.modules[_bf_loaded_name]
_bf_sys.meta_path.insert(0, _bf_finder)
_bf_builtins._BAYESFILTER_PHASE6_SOURCE_BOOTSTRAP = {
    "bundle_hash": _implementation_source_bundle_hash,
    "repository_roles": _bf_expected_roles,
    "synthetic_namespaces": _bf_synthetic_namespaces,
    "finder": _bf_finder,
}
'''


def child_source_loader_initializer(
    *,
    references: Mapping[str, Mapping[str, Any]],
    source_bundle: Mapping[str, bytes],
    worker_environment: Mapping[str, str],
    python_executable: str | Path,
    implementation_paths: Mapping[str, str | Path] | None = None,
) -> tuple[Any, tuple[Any, ...]]:
    """Return a stdlib-only initializer that runs before worker task unpickling."""

    verify_implementation_source_bundle(
        references,
        source_bundle,
        python_executable=python_executable,
        implementation_paths=implementation_paths,
    )
    bootstrap_globals = {
        "__name__": "_bayesfilter_phase6_child_source_bootstrap",
        "_worker_environment": dict(worker_environment),
        "_implementation_references": {
            role: dict(reference) for role, reference in references.items()
        },
        "_implementation_source_bundle": dict(source_bundle),
        "_implementation_source_bundle_hash": implementation_source_bundle_hash(
            source_bundle
        ),
        "_approved_repository_root": str(REPO_ROOT),
        "_benchmark_driver_relative_path": str(
            BENCHMARK_DRIVER_PATH.relative_to(REPO_ROOT)
        ),
        "_benchmark_driver_module": (
            "docs.benchmarks."
            "run_multidim_lgssm_serious_hmc_tuning_2026_07_09"
        ),
    }
    return exec, (CHILD_SOURCE_LOADER_BOOTSTRAP, bootstrap_globals)


def build_smoke_authority_proposal_manifest(
    *, proposal_path: str | Path
) -> Mapping[str, Any]:
    proposal = _read_json(proposal_path)
    parse_smoke_authority_proposal(proposal)
    return _embed_hash(
        {
            "schema": HMC_PHASE6_SMOKE_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V1,
            "terminal_manifest": True,
            "proposal_reference": build_phase5_artifact_reference(
                proposal_path, embedded_hash_rule="canonical_without_hash"
            ),
        }
    )


def parse_smoke_authority_proposal_manifest(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require_exact_keys(
        payload,
        required=("schema", "terminal_manifest", "proposal_reference", "artifact_hash"),
        label="smoke proposal manifest",
    )
    if payload.get("schema") != HMC_PHASE6_SMOKE_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V1:
        raise ValueError("unsupported smoke proposal manifest schema")
    if payload.get("terminal_manifest") is not True:
        raise ValueError("smoke proposal manifest must be terminal")
    reference = parse_phase5_artifact_reference(payload["proposal_reference"])
    if reference["source_schema"] != HMC_PHASE6_SMOKE_AUTHORITY_PROPOSAL_SCHEMA_V1:
        raise ValueError("smoke proposal manifest reference schema mismatch")
    _verify_hash(payload, label="smoke proposal manifest")
    return payload


def verify_smoke_authority_proposal_manifest(
    payload: Mapping[str, Any], *, proposal_path: str | Path
) -> Mapping[str, Any]:
    parse_smoke_authority_proposal_manifest(payload)
    verify_phase5_artifact_reference(payload["proposal_reference"], path=proposal_path)
    return payload


def expected_smoke_approval_statement(proposal_manifest_hash: str) -> str:
    _require_sha256(proposal_manifest_hash, label="proposal manifest hash")
    return (
        f"I approve {SMOKE_AUTHORITY_DECISION} bound to Phase 6 authority "
        f"proposal manifest {proposal_manifest_hash}."
    )


def build_smoke_authority(
    *,
    proposal_manifest_path: str | Path,
    human_approval_statement: str,
    human_approval_date: str,
) -> Mapping[str, Any]:
    manifest = _read_json(proposal_manifest_path)
    parse_smoke_authority_proposal_manifest(manifest)
    statement = expected_smoke_approval_statement(manifest["artifact_hash"])
    if human_approval_statement != statement:
        raise ValueError("smoke human approval statement mismatch")
    approval_date = _require_iso_date(
        human_approval_date,
        label="human approval date",
    )
    return _embed_hash(
        {
            "schema": HMC_PHASE6_SMOKE_AUTHORITY_SCHEMA_V1,
            "status": SMOKE_AUTHORITY_STATUS_APPROVED,
            "decision": SMOKE_AUTHORITY_DECISION,
            "human_approval_statement": statement,
            "human_approval_date": approval_date,
            "proposal_manifest_reference": build_phase5_artifact_reference(
                proposal_manifest_path, embedded_hash_rule="canonical_without_hash"
            ),
            "launches_authorized": 1,
            "mode": "smoke",
            "serious_runtime_authority": False,
            "phase8_authority": False,
            "neutra_authority": False,
            "nonclaims": AUTHORITY_NONCLAIMS,
        }
    )


def parse_smoke_authority(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = (
        "schema",
        "status",
        "decision",
        "human_approval_statement",
        "human_approval_date",
        "proposal_manifest_reference",
        "launches_authorized",
        "mode",
        "serious_runtime_authority",
        "phase8_authority",
        "neutra_authority",
        "nonclaims",
        "artifact_hash",
    )
    _require_exact_keys(payload, required=fields, label="smoke authority")
    if payload.get("schema") != HMC_PHASE6_SMOKE_AUTHORITY_SCHEMA_V1:
        raise ValueError("unsupported smoke authority schema")
    if payload.get("status") != SMOKE_AUTHORITY_STATUS_APPROVED:
        raise ValueError("smoke authority status mismatch")
    if payload.get("decision") != SMOKE_AUTHORITY_DECISION:
        raise ValueError("smoke authority decision mismatch")
    reference = parse_phase5_artifact_reference(payload["proposal_manifest_reference"])
    if reference["source_schema"] != HMC_PHASE6_SMOKE_AUTHORITY_PROPOSAL_MANIFEST_SCHEMA_V1:
        raise ValueError("smoke authority manifest schema mismatch")
    if payload.get("human_approval_statement") != expected_smoke_approval_statement(
        reference["embedded_artifact_hash"]
    ):
        raise ValueError("smoke authority approval mismatch")
    _require_iso_date(payload.get("human_approval_date"), label="approval date")
    if payload.get("launches_authorized") != 1 or payload.get("mode") != "smoke":
        raise ValueError("smoke authority scope mismatch")
    for name in ("serious_runtime_authority", "phase8_authority", "neutra_authority"):
        if _require_bool(payload.get(name), label=name):
            raise ValueError(f"smoke authority cannot grant {name}")
    _require_ordered(
        payload.get("nonclaims"),
        expected=AUTHORITY_NONCLAIMS,
        label="smoke authority nonclaims",
    )
    _verify_hash(payload, label="smoke authority")
    return payload


def verify_smoke_authority(
    payload: Mapping[str, Any], *, proposal_manifest_path: str | Path
) -> Mapping[str, Any]:
    parse_smoke_authority(payload)
    verify_phase5_artifact_reference(
        payload["proposal_manifest_reference"], path=proposal_manifest_path
    )
    return payload


def build_launch_claim(
    *,
    authority: Mapping[str, Any],
    proposal_manifest: Mapping[str, Any],
    command: Sequence[str],
    paths: Mapping[str, Any],
    pid: int,
    started_at_utc: str | None = None,
) -> Mapping[str, Any]:
    parse_smoke_authority(authority)
    parse_smoke_authority_proposal_manifest(proposal_manifest)
    _strict_paths(paths)
    return _embed_hash(
        {
            "schema": HMC_PHASE6_SMOKE_LAUNCH_CLAIM_SCHEMA_V1,
            "authority_artifact_hash": authority["artifact_hash"],
            "proposal_manifest_artifact_hash": proposal_manifest["artifact_hash"],
            "command": tuple(command),
            "paths": dict(paths),
            "pid": _require_positive_int(pid, label="claim pid"),
            "started_at_utc": started_at_utc
            or datetime.now(timezone.utc).isoformat(),
            "file_mode": "0400",
            "permanent_authority_consumption": True,
        }
    )


def parse_launch_claim(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = (
        "schema",
        "authority_artifact_hash",
        "proposal_manifest_artifact_hash",
        "command",
        "paths",
        "pid",
        "started_at_utc",
        "file_mode",
        "permanent_authority_consumption",
        "artifact_hash",
    )
    _require_exact_keys(payload, required=fields, label="smoke launch claim")
    if payload.get("schema") != HMC_PHASE6_SMOKE_LAUNCH_CLAIM_SCHEMA_V1:
        raise ValueError("unsupported smoke launch claim schema")
    _require_sha256(payload.get("authority_artifact_hash"), label="authority hash")
    _require_sha256(payload.get("proposal_manifest_artifact_hash"), label="manifest hash")
    command = payload.get("command")
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)) or not command:
        raise ValueError("smoke claim command must be a non-empty sequence")
    for item in command:
        _require_string(item, label="claim command item")
    _strict_paths(payload["paths"])
    _require_positive_int(payload.get("pid"), label="claim pid")
    _require_string(payload.get("started_at_utc"), label="claim start time")
    if payload.get("file_mode") != "0400":
        raise ValueError("smoke claim file mode mismatch")
    if _require_bool(
        payload.get("permanent_authority_consumption"),
        label="permanent_authority_consumption",
    ) is not True:
        raise ValueError("smoke claim must permanently consume authority")
    _verify_hash(payload, label="smoke launch claim")
    return payload


def _prepared_context_snapshot_hash(
    *,
    config: Any,
    authority_reference: Mapping[str, Any],
    proposal: Mapping[str, Any],
    proposal_manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    claim: Mapping[str, Any],
    preflight: Mapping[str, Any],
    command: Sequence[str],
    paths: Mapping[str, Path],
    implementation_source_bundle: Mapping[str, bytes],
) -> str:
    return canonical_artifact_payload_hash(
        {
            "config_hash": config.hash,
            "config_path": str(config.path.resolve()),
            "authority_reference_hash": canonical_artifact_payload_hash(
                authority_reference
            ),
            "proposal_snapshot_hash": canonical_artifact_payload_hash(proposal),
            "proposal_manifest_snapshot_hash": canonical_artifact_payload_hash(
                proposal_manifest
            ),
            "authority_snapshot_hash": canonical_artifact_payload_hash(authority),
            "claim_snapshot_hash": canonical_artifact_payload_hash(claim),
            "preflight_snapshot_hash": canonical_artifact_payload_hash(preflight),
            "command": tuple(command),
            "paths": {
                name: str(path.resolve()) for name, path in sorted(paths.items())
            },
            "implementation_source_bundle_hash": implementation_source_bundle_hash(
                implementation_source_bundle
            ),
        }
    )


def attach_prepared_output_session(
    context: Phase6SmokeLaunchContext,
    session: Any,
) -> Phase6SmokeLaunchContext:
    """Attach reserved outputs without weakening prepared-context provenance."""

    verify_prepared_smoke_launch_context(context)
    if session.directories is not context.output_directories:
        raise ValueError("prepared smoke output directories mismatch")
    if session.fds.get("claim_path") != context.claim_fd:
        raise ValueError("prepared smoke retained claim descriptor mismatch")
    if session.consumed_evidence_session is not context.consumed_evidence_session:
        raise ValueError("prepared smoke consumed evidence session mismatch")
    session.validate_for_runtime()
    token = getattr(context, "_prepared_token", None)
    assert token is not None
    attached = Phase6SmokeLaunchContext(
        config=context.config,
        preflight=context.preflight,
        proposal=context.proposal,
        proposal_manifest=context.proposal_manifest,
        authority=context.authority,
        authority_reference=context.authority_reference,
        claim=context.claim,
        paths=context.paths,
        implementation_source_bundle=context.implementation_source_bundle,
        command=context.command,
        output_directories=context.output_directories,
        claim_fd=context.claim_fd,
        consumed_evidence_session=context.consumed_evidence_session,
        output_session=session,
        prepared_snapshot_hash=context.prepared_snapshot_hash,
        _prepared_token=token,
    )
    del _PREPARED_CONTEXT_TOKENS[id(context)]
    del _PREPARED_CONTEXT_EVIDENCE_SESSIONS[id(context)]
    _PREPARED_CONTEXT_TOKENS[id(attached)] = token
    _PREPARED_CONTEXT_EVIDENCE_SESSIONS[id(attached)] = (
        attached.consumed_evidence_session
    )
    return attached


def verify_prepared_smoke_launch_context(
    context: Phase6SmokeLaunchContext,
    *,
    consume: bool = False,
) -> Phase6SmokeLaunchContext:
    """Reject contexts not issued by ``prepare_smoke_launch`` in this process."""

    token = context._prepared_token
    if token is None or _PREPARED_CONTEXT_TOKENS.get(id(context)) is not token:
        raise ValueError("smoke launch context was not issued by prepare_smoke_launch")
    if _PREPARED_CONTEXT_EVIDENCE_SESSIONS.get(id(context)) is not (
        context.consumed_evidence_session
    ):
        raise ValueError("smoke launch context evidence session identity mismatch")
    if context.consumed_evidence_session is None:
        raise ValueError("smoke launch context lacks retained consumed evidence")
    context.consumed_evidence_session.verify()
    expected = _prepared_context_snapshot_hash(
        config=context.config,
        authority_reference=context.authority_reference,
        proposal=context.proposal,
        proposal_manifest=context.proposal_manifest,
        authority=context.authority,
        claim=context.claim,
        preflight=context.preflight,
        command=context.command,
        paths=context.paths,
        implementation_source_bundle=context.implementation_source_bundle,
    )
    if context.prepared_snapshot_hash != expected:
        raise ValueError("prepared smoke launch context snapshot mismatch")
    if consume:
        del _PREPARED_CONTEXT_TOKENS[id(context)]
        del _PREPARED_CONTEXT_EVIDENCE_SESSIONS[id(context)]
    return context


def discard_prepared_smoke_launch_context(context: Phase6SmokeLaunchContext) -> None:
    """Drop any unconsumed in-process launch capability during teardown."""

    _PREPARED_CONTEXT_TOKENS.pop(id(context), None)
    _PREPARED_CONTEXT_EVIDENCE_SESSIONS.pop(id(context), None)


def _parse_smoke_links(payload: Mapping[str, Any]) -> None:
    for name in (
        "smoke_authority_artifact_hash",
        "smoke_launch_claim_artifact_hash",
        "smoke_proposal_manifest_artifact_hash",
        "preflight_before_runtime_artifact_hash",
    ):
        _require_sha256(payload.get(name), label=name)


def _require_nonnegative_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def _parse_preflight_before_runtime(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    from bayesfilter.inference.hmc_identity_adoption import (
        parse_phase5_preflight_report,
    )

    parse_phase5_preflight_report(payload)
    if payload.get("runtime_authority") is not False or (
        payload.get("runtime_executed") is not False
    ):
        raise ValueError("smoke preflight must precede runtime")
    return payload


def _parse_smoke_diagnostics(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = (
        "schema",
        "passed",
        "input_all_finite",
        "diagnostics_all_finite",
        "draw_count_per_chain",
        "chain_count",
        "parameter_count",
        "split_draw_count_per_chain",
        "split_chain_count",
        "thresholds",
        "definitions",
        "max_rhat",
        "min_bulk_ess",
        "min_tail_ess",
        "parameter_diagnostics",
        "hard_vetoes",
        "nonclaims",
        "smoke_gate",
    )
    _require_exact_keys(payload, required=fields, label="smoke diagnostics")
    if payload.get("schema") != "bayesfilter.rank_normalized_hmc_diagnostics.v1":
        raise ValueError("unsupported smoke diagnostics schema")
    for name in ("passed", "input_all_finite", "diagnostics_all_finite"):
        if _require_bool(payload.get(name), label=f"diagnostics {name}") is not True:
            raise ValueError(f"smoke diagnostics {name} must be true")
    expected_counts = {
        "draw_count_per_chain": 8,
        "chain_count": 4,
        "parameter_count": 18,
        "split_draw_count_per_chain": 4,
        "split_chain_count": 8,
    }
    for name, expected in expected_counts.items():
        if payload.get(name) != expected:
            raise ValueError(f"smoke diagnostics {name} mismatch")
    thresholds = payload.get("thresholds")
    _require_exact_keys(
        thresholds,
        required=("rhat_max", "bulk_ess_min", "tail_ess_min"),
        label="smoke diagnostic thresholds",
    )
    if thresholds != {
        "rhat_max": 1.01,
        "bulk_ess_min": 1000.0,
        "tail_ess_min": 400.0,
    }:
        raise ValueError("smoke diagnostic thresholds mismatch")
    definitions = payload.get("definitions")
    _require_exact_keys(
        definitions,
        required=(
            "rank_transform",
            "rhat",
            "bulk_ess",
            "tail_ess",
            "autocorrelation_truncation",
            "quantile_interpolation",
        ),
        label="smoke diagnostic definitions",
    )
    for name in definitions:
        _require_string(definitions[name], label=f"diagnostic definition {name}")
    for name in ("max_rhat", "min_bulk_ess", "min_tail_ess"):
        _require_nonnegative_number(payload.get(name), label=name)
    rows = payload.get("parameter_diagnostics")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != 18:
        raise ValueError("smoke diagnostics require 18 parameter rows")
    row_fields = (
        "parameter",
        "rank_normalized_split_rhat",
        "folded_rank_normalized_split_rhat",
        "rhat",
        "bulk_ess",
        "tail_ess",
        "lower_tail_ess",
        "upper_tail_ess",
        "passed",
    )
    seen: set[str] = set()
    for row in rows:
        _require_exact_keys(row, required=row_fields, label="smoke parameter row")
        name = _require_string(row.get("parameter"), label="parameter")
        if name in seen:
            raise ValueError("smoke parameter rows contain duplicates")
        seen.add(name)
        for field in row_fields[1:-1]:
            _require_nonnegative_number(row.get(field), label=f"{name}.{field}")
        if _require_bool(row.get("passed"), label=f"{name}.passed") is not True:
            raise ValueError(f"smoke diagnostic row failed: {name}")
    if tuple(payload.get("hard_vetoes", ())) != ():
        raise ValueError("smoke diagnostics contain hard vetoes")
    _require_ordered(
        payload.get("nonclaims"),
        expected=(
            "finite-only smoke engineering diagnostic screen",
            "R-hat and ESS values are explanatory only",
            "no posterior recovery or HMC convergence claim",
            "no sampler superiority, production, or default readiness claim",
        ),
        label="diagnostic nonclaims",
    )
    if payload.get("smoke_gate") != "finite_diagnostics_only_non_promoting":
        raise ValueError("smoke diagnostic gate mismatch")
    return payload


def _parse_progress_check(payload: Mapping[str, Any], *, stage: str) -> None:
    _require_exact_keys(
        payload,
        required=(
            "stage",
            "completed_results_per_chain",
            "passed",
            "max_rhat",
            "min_bulk_ess",
            "min_tail_ess",
            "input_all_finite",
            "diagnostics_all_finite",
            "hard_vetoes",
        ),
        label=f"smoke {stage} progress check",
    )
    if payload.get("stage") != stage:
        raise ValueError(f"smoke {stage} progress stage mismatch")
    expected_count = 4 if stage == "burnin" else 8
    if payload.get("completed_results_per_chain") != expected_count:
        raise ValueError(f"smoke {stage} progress count mismatch")
    for name in ("passed", "input_all_finite", "diagnostics_all_finite"):
        if _require_bool(payload.get(name), label=f"{stage}.{name}") is not True:
            raise ValueError(f"smoke {stage} progress {name} must be true")
    for name in ("max_rhat", "min_bulk_ess", "min_tail_ess"):
        _require_nonnegative_number(payload.get(name), label=f"{stage}.{name}")
    if tuple(payload.get("hard_vetoes", ())) != ():
        raise ValueError(f"smoke {stage} progress contains hard vetoes")


def parse_smoke_result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = (
        "schema",
        "passed",
        "decision",
        "smoke",
        "smoke_authority_artifact_hash",
        "smoke_launch_claim_artifact_hash",
        "smoke_proposal_manifest_artifact_hash",
        "preflight_before_runtime_artifact_hash",
        "config_hash",
        "preflight_before_runtime",
        "burnin_results_per_chain",
        "retained_results_per_chain",
        "final_diagnostics",
        "worker_count",
        "chains_per_worker",
        "chain_count",
        "worker_pids",
        "worker_metadata",
        "private_retained_sample_reference",
        "jit_compile",
        "jit_compile_false_runtime_executed",
        "cuda_visible_devices",
        "elapsed_seconds",
        "serious_runtime_executed",
        "neutra_executed",
        "phase8_executed",
        "nonclaims",
        "artifact_hash",
    )
    _require_exact_keys(payload, required=fields, label="smoke result")
    if payload.get("schema") != HMC_PHASE6_SMOKE_RESULT_SCHEMA_V1:
        raise ValueError("unsupported smoke result schema")
    if payload.get("passed") is not True or payload.get("smoke") is not True:
        raise ValueError("smoke result pass/mode mismatch")
    if payload.get("decision") != SMOKE_PASS_DECISION:
        raise ValueError("smoke result decision mismatch")
    _parse_smoke_links(payload)
    _require_sha256(payload.get("config_hash"), label="config_hash")
    preflight = _parse_preflight_before_runtime(payload["preflight_before_runtime"])
    if preflight["artifact_hash"] != payload["preflight_before_runtime_artifact_hash"]:
        raise ValueError("smoke result preflight cross-link mismatch")
    expected_counts = {
        "burnin_results_per_chain": 4,
        "retained_results_per_chain": 8,
        "worker_count": 2,
        "chains_per_worker": 2,
        "chain_count": 4,
    }
    for name, expected in expected_counts.items():
        if payload.get(name) != expected:
            raise ValueError(f"smoke result {name} mismatch")
    diagnostics = _parse_smoke_diagnostics(payload.get("final_diagnostics"))
    pids = payload.get("worker_pids")
    if not isinstance(pids, Sequence) or isinstance(pids, (str, bytes)):
        raise ValueError("smoke worker PIDs must be a sequence")
    normalized_pids = tuple(_require_positive_int(item, label="worker PID") for item in pids)
    if len(normalized_pids) != 2 or len(set(normalized_pids)) != 2:
        raise ValueError("smoke requires two distinct worker PIDs")
    metadata = payload.get("worker_metadata")
    if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes)) or len(metadata) != 2:
        raise ValueError("smoke requires two worker metadata entries")
    allowed_worker_fields = (
        "jit_compile",
        "use_xla",
        "compile_trace_count",
        "first_call_s",
        "warm_call_s",
        "tensorflow_version",
        "tfp_version",
        "python_version",
        "cuda_visible_devices",
        "thread_environment",
        "child_source_references_verified",
        "child_implementation_references_verified",
        "child_loaded_source_bytes_verified",
        "child_implementation_source_bundle_hash",
        "child_transition_identity_verified",
        "child_transition_identity_hash",
    )
    for item in metadata:
        _require_exact_keys(item, required=allowed_worker_fields, label="smoke worker metadata")
        if item.get("jit_compile") is not True or item.get("use_xla") is not True:
            raise ValueError("smoke worker XLA/JIT metadata mismatch")
        if item.get("cuda_visible_devices") != "-1":
            raise ValueError("smoke worker CPU hiding mismatch")
        if item.get("child_source_references_verified") is not True or (
            item.get("child_implementation_references_verified") is not True
        ) or (
            item.get("child_loaded_source_bytes_verified") is not True
        ) or (
            item.get("child_transition_identity_verified") is not True
        ) or item.get("child_transition_identity_hash") != TRANSITION_IDENTITY_HASH:
            raise ValueError("smoke worker child identity gate mismatch")
        _require_sha256(
            item.get("child_implementation_source_bundle_hash"),
            label="child implementation source bundle hash",
        )
        for name in ("tensorflow_version", "tfp_version", "python_version"):
            _require_string(item.get(name), label=f"worker {name}")
        _require_positive_int(
            item.get("compile_trace_count"), label="worker compile_trace_count"
        )
        for name in ("first_call_s", "warm_call_s"):
            _require_nonnegative_number(item.get(name), label=f"worker {name}")
        expected_worker_environment = {
            "TF_NUM_INTRAOP_THREADS": "8",
            "TF_NUM_INTEROP_THREADS": "1",
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "-1",
            "TF_CPP_MIN_LOG_LEVEL": "1",
            "MPLCONFIGDIR": "/tmp/matplotlib-bayesfilter-phase7-worker",
        }
        if item.get("thread_environment") != expected_worker_environment:
            raise ValueError("smoke worker thread environment mismatch")
    private = payload.get("private_retained_sample_reference")
    _require_exact_keys(
        private,
        required=(
            "file_sha256",
            "byte_count",
            "shape_verified",
            "finite_verified",
            "provenance_verified",
            "path_publicized",
            "raw_samples_publicized",
        ),
        label="smoke private sample reference",
    )
    _require_sha256(private.get("file_sha256"), label="private sample hash", tagged=False)
    _require_positive_int(private.get("byte_count"), label="private sample byte_count")
    for name in ("shape_verified", "finite_verified", "provenance_verified"):
        if _require_bool(private.get(name), label=name) is not True:
            raise ValueError(f"smoke private sample {name} failed")
    for name in ("path_publicized", "raw_samples_publicized"):
        if _require_bool(private.get(name), label=name) is not False:
            raise ValueError(f"smoke private sample {name} must be false")
    if payload.get("jit_compile") is not True or (
        payload.get("jit_compile_false_runtime_executed") is not False
    ) or payload.get("cuda_visible_devices") != "-1":
        raise ValueError("smoke result runtime metadata mismatch")
    for name in ("serious_runtime_executed", "neutra_executed", "phase8_executed"):
        if _require_bool(payload.get(name), label=name) is not False:
            raise ValueError(f"smoke result {name} must be false")
    _require_nonnegative_number(payload.get("elapsed_seconds"), label="elapsed_seconds")
    _require_ordered(payload.get("nonclaims"), expected=SMOKE_NONCLAIMS, label="smoke nonclaims")
    _verify_hash(payload, label="smoke result")
    return payload


def parse_smoke_failure(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = (
        "schema",
        "passed",
        "decision",
        "smoke",
        "smoke_authority_artifact_hash",
        "smoke_launch_claim_artifact_hash",
        "smoke_proposal_manifest_artifact_hash",
        "preflight_before_runtime_artifact_hash",
        "stage",
        "reason",
        "config_hash",
        "preflight_before_runtime",
        "worker_pids",
        "final_diagnostics",
        "jit_compile_false_runtime_executed",
        "cuda_visible_devices",
        "elapsed_seconds",
        "serious_runtime_executed",
        "neutra_executed",
        "phase8_executed",
        "nonclaims",
        "artifact_hash",
    )
    _require_exact_keys(payload, required=fields, label="smoke failure")
    if payload.get("schema") != HMC_PHASE6_SMOKE_FAILURE_SCHEMA_V1:
        raise ValueError("unsupported smoke failure schema")
    if payload.get("passed") is not False or payload.get("smoke") is not True:
        raise ValueError("smoke failure pass/mode mismatch")
    if payload.get("decision") != SMOKE_BLOCK_DECISION:
        raise ValueError("smoke failure decision mismatch")
    _parse_smoke_links(payload)
    _require_string(payload.get("stage"), label="failure stage")
    _require_string(payload.get("reason"), label="failure reason")
    _require_sha256(payload.get("config_hash"), label="config_hash")
    preflight = _parse_preflight_before_runtime(payload["preflight_before_runtime"])
    if preflight["artifact_hash"] != payload["preflight_before_runtime_artifact_hash"]:
        raise ValueError("smoke failure preflight cross-link mismatch")
    pids = payload.get("worker_pids")
    if not isinstance(pids, Sequence) or isinstance(pids, (str, bytes)):
        raise ValueError("smoke failure worker PIDs must be a sequence")
    for item in pids:
        _require_positive_int(item, label="failure worker PID")
    final_diagnostics = payload.get("final_diagnostics")
    if final_diagnostics is not None:
        _parse_smoke_diagnostics(final_diagnostics)
    if payload.get("jit_compile_false_runtime_executed") is not False or (
        payload.get("cuda_visible_devices") != "-1"
    ):
        raise ValueError("smoke failure runtime metadata mismatch")
    for name in ("serious_runtime_executed", "neutra_executed", "phase8_executed"):
        if _require_bool(payload.get(name), label=name) is not False:
            raise ValueError(f"smoke failure {name} must be false")
    _require_nonnegative_number(payload.get("elapsed_seconds"), label="elapsed_seconds")
    _require_ordered(
        payload.get("nonclaims"),
        expected=SMOKE_FAILURE_NONCLAIMS,
        label="smoke failure nonclaims",
    )
    _verify_hash(payload, label="smoke failure")
    return payload


def parse_smoke_terminal_result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    schema = payload.get("schema")
    if schema == HMC_PHASE6_SMOKE_RESULT_SCHEMA_V1:
        return parse_smoke_result(payload)
    if schema == HMC_PHASE6_SMOKE_FAILURE_SCHEMA_V1:
        return parse_smoke_failure(payload)
    raise ValueError("unsupported smoke terminal result schema")


def build_smoke_infrastructure_failure(
    *,
    context: Phase6SmokeLaunchContext,
    stage: str,
    error: BaseException,
    primary_result_preserved: bool,
    primary_result_artifact_hash: str | None,
) -> Mapping[str, Any]:
    if primary_result_preserved:
        _require_sha256(
            primary_result_artifact_hash, label="primary result artifact hash"
        )
    elif primary_result_artifact_hash is not None:
        raise ValueError("unpreserved primary result cannot carry an artifact hash")
    return _embed_hash(
        {
            "schema": HMC_PHASE6_SMOKE_INFRASTRUCTURE_FAILURE_SCHEMA_V1,
            "passed": False,
            "decision": SMOKE_INFRASTRUCTURE_BLOCK_DECISION,
            "smoke": True,
            "smoke_authority_artifact_hash": context.authority["artifact_hash"],
            "smoke_launch_claim_artifact_hash": context.claim["artifact_hash"],
            "smoke_proposal_manifest_artifact_hash": context.proposal_manifest[
                "artifact_hash"
            ],
            "preflight_before_runtime_artifact_hash": context.preflight[
                "artifact_hash"
            ],
            "stage": _require_string(stage, label="infrastructure stage"),
            "reason": f"infrastructure_error:{type(error).__name__}",
            "primary_result_preserved": bool(primary_result_preserved),
            "primary_result_artifact_hash": primary_result_artifact_hash,
            "serious_runtime_executed": False,
            "neutra_executed": False,
            "phase8_executed": False,
            "nonclaims": SMOKE_INFRASTRUCTURE_FAILURE_NONCLAIMS,
        }
    )


def parse_smoke_infrastructure_failure(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    fields = (
        "schema",
        "passed",
        "decision",
        "smoke",
        "smoke_authority_artifact_hash",
        "smoke_launch_claim_artifact_hash",
        "smoke_proposal_manifest_artifact_hash",
        "preflight_before_runtime_artifact_hash",
        "stage",
        "reason",
        "primary_result_preserved",
        "primary_result_artifact_hash",
        "serious_runtime_executed",
        "neutra_executed",
        "phase8_executed",
        "nonclaims",
        "artifact_hash",
    )
    _require_exact_keys(payload, required=fields, label="smoke infrastructure failure")
    if payload.get("schema") != HMC_PHASE6_SMOKE_INFRASTRUCTURE_FAILURE_SCHEMA_V1:
        raise ValueError("unsupported smoke infrastructure failure schema")
    if payload.get("passed") is not False or payload.get("smoke") is not True:
        raise ValueError("smoke infrastructure failure mode mismatch")
    if payload.get("decision") != SMOKE_INFRASTRUCTURE_BLOCK_DECISION:
        raise ValueError("smoke infrastructure failure decision mismatch")
    _parse_smoke_links(payload)
    _require_string(payload.get("stage"), label="infrastructure stage")
    reason = _require_string(payload.get("reason"), label="infrastructure reason")
    if not reason.startswith("infrastructure_error:"):
        raise ValueError("smoke infrastructure failure reason mismatch")
    preserved = _require_bool(
        payload.get("primary_result_preserved"), label="primary_result_preserved"
    )
    if preserved:
        _require_sha256(
            payload.get("primary_result_artifact_hash"),
            label="primary_result_artifact_hash",
        )
    elif payload.get("primary_result_artifact_hash") is not None:
        raise ValueError("unpreserved primary result cannot carry an artifact hash")
    for name in ("serious_runtime_executed", "neutra_executed", "phase8_executed"):
        if _require_bool(payload.get(name), label=name) is not False:
            raise ValueError(f"smoke infrastructure failure {name} must be false")
    _require_ordered(
        payload.get("nonclaims"),
        expected=SMOKE_INFRASTRUCTURE_FAILURE_NONCLAIMS,
        label="smoke infrastructure nonclaims",
    )
    _verify_hash(payload, label="smoke infrastructure failure")
    return payload


def parse_smoke_progress(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = (
        "schema",
        "status",
        "config_hash",
        "smoke",
        "smoke_authority_artifact_hash",
        "smoke_launch_claim_artifact_hash",
        "smoke_proposal_manifest_artifact_hash",
        "preflight_before_runtime_artifact_hash",
        "burnin_checks",
        "retained_checks",
        "completed",
        "passed",
        "result_artifact_hash",
        "artifact_hash",
    )
    _require_exact_keys(payload, required=fields, label="smoke progress")
    if payload.get("schema") != HMC_PHASE6_SMOKE_PROGRESS_SCHEMA_V1:
        raise ValueError("unsupported smoke progress schema")
    if payload.get("status") not in {"result_written", "blocked_result_written"}:
        raise ValueError("smoke progress is not terminal")
    if payload.get("smoke") is not True or payload.get("completed") is not True:
        raise ValueError("smoke progress mode/completion mismatch")
    _require_bool(payload.get("passed"), label="progress passed")
    _require_sha256(payload.get("config_hash"), label="config_hash")
    _require_sha256(payload.get("result_artifact_hash"), label="result artifact hash")
    _parse_smoke_links(payload)
    for name in ("burnin_checks", "retained_checks"):
        if not isinstance(payload.get(name), Sequence) or isinstance(
            payload.get(name), (str, bytes)
        ):
            raise ValueError(f"smoke progress {name} must be a sequence")
    if len(payload["burnin_checks"]) > 1 or len(payload["retained_checks"]) > 1:
        raise ValueError("smoke progress permits at most one check per stage")
    for item in payload["burnin_checks"]:
        _parse_progress_check(item, stage="burnin")
    for item in payload["retained_checks"]:
        _parse_progress_check(item, stage="retained")
    if payload.get("passed") is True:
        if len(payload["burnin_checks"]) != 1 or len(payload["retained_checks"]) != 1:
            raise ValueError("passing smoke progress requires one check per stage")
        _parse_progress_check(payload["burnin_checks"][0], stage="burnin")
        _parse_progress_check(payload["retained_checks"][0], stage="retained")
    _verify_hash(payload, label="smoke progress")
    return payload


def build_smoke_output_manifest(
    *,
    proposal_path: str | Path,
    proposal_manifest_path: str | Path,
    authority_path: str | Path,
    claim_path: str | Path,
    progress_path: str | Path,
    result_path: str | Path,
    log_path: str | Path,
    private_samples_path: str | Path,
    infrastructure_failure_path: str | Path,
    infrastructure_manifest_path: str | Path,
    output_session: SecureSmokeOutputSession | None = None,
    launch_context: Phase6SmokeLaunchContext | None = None,
) -> Mapping[str, Any]:
    if output_session is None:
        authority = _read_json(authority_path)
        claim = _read_json(claim_path)
        progress = _read_json(progress_path)
        result = _read_json(result_path)
        proposal_manifest = _read_json(proposal_manifest_path)
        proposal = _read_json(proposal_path)
        preflight_hash = _read_json(PREFLIGHT_PATH)["artifact_hash"]
        authority_reference = build_phase5_artifact_reference(
            authority_path, embedded_hash_rule="canonical_without_hash"
        )
    else:
        if launch_context is None:
            raise ValueError("secure smoke manifest requires captured launch context")
        authority = launch_context.authority
        claim = output_session.read_json("claim_path")
        progress = output_session.read_json("public_progress_path")
        result = output_session.read_json("public_result_path")
        proposal_manifest = launch_context.proposal_manifest
        proposal = launch_context.proposal
        preflight_hash = launch_context.preflight["artifact_hash"]
        authority_reference = launch_context.authority_reference
    parse_smoke_authority(authority)
    parse_launch_claim(claim)
    parse_smoke_progress(progress)
    parse_smoke_terminal_result(result)
    if output_session is None:
        verify_smoke_authority(
            authority, proposal_manifest_path=proposal_manifest_path
        )
    else:
        if authority["proposal_manifest_reference"]["embedded_artifact_hash"] != (
            proposal_manifest["artifact_hash"]
        ):
            raise ValueError("smoke authority proposal reference mismatch")
    parse_smoke_authority_proposal_manifest(proposal_manifest)
    if output_session is None:
        verify_smoke_authority_proposal_manifest(
            proposal_manifest, proposal_path=proposal_path
        )
    parse_smoke_authority_proposal(proposal)
    if proposal_manifest["proposal_reference"]["embedded_artifact_hash"] != (
        proposal["artifact_hash"]
    ) or proposal_manifest["proposal_reference"]["canonical_payload_hash"] != (
        canonical_artifact_payload_hash(proposal)
    ):
        raise ValueError("smoke proposal/manifest captured payload mismatch")
    if authority["proposal_manifest_reference"]["embedded_artifact_hash"] != (
        proposal_manifest["artifact_hash"]
    ) or authority["proposal_manifest_reference"]["canonical_payload_hash"] != (
        canonical_artifact_payload_hash(proposal_manifest)
    ):
        raise ValueError("smoke authority/manifest captured payload mismatch")
    if output_session is not None and canonical_artifact_payload_hash(
        claim
    ) != canonical_artifact_payload_hash(launch_context.claim):
        raise ValueError("smoke claim differs from captured launch claim")
    if claim["authority_artifact_hash"] != authority["artifact_hash"] or (
        claim["proposal_manifest_artifact_hash"]
        != proposal_manifest["artifact_hash"]
    ):
        raise ValueError("smoke claim authority/proposal cross-link mismatch")
    if tuple(claim["command"]) != tuple(proposal["command"]) or (
        claim["paths"] != proposal["paths"]
    ):
        raise ValueError("smoke claim command/path cross-link mismatch")
    for name in (
        "smoke_authority_artifact_hash",
        "smoke_launch_claim_artifact_hash",
        "smoke_proposal_manifest_artifact_hash",
        "preflight_before_runtime_artifact_hash",
    ):
        expected = {
            "smoke_authority_artifact_hash": authority["artifact_hash"],
            "smoke_launch_claim_artifact_hash": claim["artifact_hash"],
            "smoke_proposal_manifest_artifact_hash": proposal_manifest["artifact_hash"],
            "preflight_before_runtime_artifact_hash": preflight_hash,
        }[name]
        if result[name] != expected or progress[name] != expected:
            raise ValueError(f"smoke terminal cross-link mismatch: {name}")
    if progress["result_artifact_hash"] != result["artifact_hash"] or (
        progress["passed"] != result["passed"]
    ):
        raise ValueError("smoke progress/result cross-link mismatch")
    if progress["config_hash"] != result["config_hash"]:
        raise ValueError("smoke progress/result config mismatch")
    if output_session is not None:
        if result["config_hash"] != launch_context.config.hash:
            raise ValueError("smoke result differs from captured config")
        runtime = proposal["runtime"]
        if result["cuda_visible_devices"] != runtime["cuda_visible_devices"]:
            raise ValueError("smoke result runtime environment mismatch")
        if result["passed"] is True:
            expected_source_bundle_hash = implementation_source_bundle_hash(
                launch_context.implementation_source_bundle
            )
            for metadata in result["worker_metadata"]:
                if metadata["child_implementation_source_bundle_hash"] != (
                    expected_source_bundle_hash
                ):
                    raise ValueError(
                        "smoke worker source bundle differs from captured bytes"
                    )
                for name in (
                    "tensorflow_version",
                    "tfp_version",
                    "python_version",
                    "cuda_visible_devices",
                ):
                    if metadata[name] != runtime[name]:
                        raise ValueError(f"smoke worker {name} differs from authority")
                for name, value in runtime["thread_environment"].items():
                    if metadata["thread_environment"].get(name) != value:
                        raise ValueError(
                            f"smoke worker thread environment differs: {name}"
                        )
    private_path = Path(private_samples_path)
    private_available = (
        private_path.is_file() and private_path.stat().st_size > 0
        if output_session is None
        else output_session.nonempty("private_samples_path")
    )
    if result["passed"] is True and not private_available:
        raise ValueError("passing smoke result requires private samples")
    if private_available:
        private_reference = (
            build_file_reference(private_path)
            if output_session is None
            else output_session.file_reference("private_samples_path")
        )
        if result["passed"] is True:
            declared_private = result["private_retained_sample_reference"]
            if declared_private["file_sha256"] != private_reference["file_sha256"] or (
                declared_private["byte_count"] != private_reference["byte_count"]
            ):
                raise ValueError("smoke result/private sample exact-byte mismatch")
    else:
        private_reference = None
    if output_session is None:
        infrastructure_failure_reference = build_file_reference(
            infrastructure_failure_path
        )
        infrastructure_manifest_reference = build_file_reference(
            infrastructure_manifest_path
        )
    else:
        infrastructure_failure_reference = output_session.file_reference(
            "infrastructure_failure_path"
        )
        infrastructure_manifest_reference = output_session.file_reference(
            "infrastructure_manifest_path"
        )
    if infrastructure_failure_reference["byte_count"] != 0 or (
        infrastructure_manifest_reference["byte_count"] != 0
    ):
        raise ValueError("normal smoke cannot contain infrastructure failure bytes")
    return _embed_hash(
        {
            "schema": HMC_PHASE6_SMOKE_OUTPUT_MANIFEST_SCHEMA_V1,
            "terminal_manifest": True,
            "passed": bool(result["passed"]),
            "authority_reference": authority_reference,
            "claim_reference": (
                build_phase5_artifact_reference(
                    claim_path, embedded_hash_rule="canonical_without_hash"
                )
                if output_session is None
                else output_session.artifact_reference("claim_path")
            ),
            "progress_reference": (
                build_phase5_artifact_reference(
                    progress_path, embedded_hash_rule="canonical_without_hash"
                )
                if output_session is None
                else output_session.artifact_reference("public_progress_path")
            ),
            "result_reference": (
                build_phase5_artifact_reference(
                    result_path, embedded_hash_rule="canonical_without_hash"
                )
                if output_session is None
                else output_session.artifact_reference("public_result_path")
            ),
            "log_reference": (
                build_file_reference(log_path)
                if output_session is None
                else output_session.file_reference("log_path")
            ),
            "private_samples_available": private_available,
            "private_samples_reference": (
                private_reference
            ),
            "infrastructure_failure_written": False,
            "infrastructure_failure_reservation_reference": (
                infrastructure_failure_reference
            ),
            "infrastructure_manifest_written": False,
            "infrastructure_manifest_reservation_reference": (
                infrastructure_manifest_reference
            ),
        }
    )


def build_smoke_infrastructure_manifest(
    *,
    context: Phase6SmokeLaunchContext,
    session: SecureSmokeOutputSession,
) -> Mapping[str, Any]:
    failure = session.read_json(
        "infrastructure_failure_path", require_path_match=False
    )
    parse_smoke_infrastructure_failure(failure)
    expected_links = {
        "smoke_authority_artifact_hash": context.authority["artifact_hash"],
        "smoke_launch_claim_artifact_hash": context.claim["artifact_hash"],
        "smoke_proposal_manifest_artifact_hash": context.proposal_manifest[
            "artifact_hash"
        ],
        "preflight_before_runtime_artifact_hash": context.preflight["artifact_hash"],
    }
    for name, expected in expected_links.items():
        if failure[name] != expected:
            raise ValueError(f"infrastructure failure cross-link mismatch: {name}")
    claim = session.read_json("claim_path", require_path_match=False)
    parse_launch_claim(claim)
    if canonical_artifact_payload_hash(claim) != canonical_artifact_payload_hash(
        context.claim
    ):
        raise ValueError("infrastructure claim differs from captured claim")
    output_roles = (
        "public_result_path",
        "public_progress_path",
        "output_manifest_path",
        "log_path",
        "private_samples_path",
    )
    output_evidence: dict[str, Any] = {}
    for role in output_roles:
        prefix = {
            "public_result_path": "public_result",
            "public_progress_path": "public_progress",
            "output_manifest_path": "output_manifest",
            "log_path": "log",
            "private_samples_path": "private_samples",
        }[role]
        reserved = session.has_role(role)
        reference = (
            session.file_reference(role, require_path_match=False)
            if reserved
            else None
        )
        output_evidence[f"{prefix}_reserved"] = reserved
        output_evidence[f"{prefix}_nonempty"] = bool(
            reference is not None and reference["byte_count"] > 0
        )
        output_evidence[f"{prefix}_path_intact"] = (
            session.path_intact(role) if reserved else False
        )
        output_evidence[f"{prefix}_reference"] = reference
    return _embed_hash(
        {
            "schema": HMC_PHASE6_SMOKE_INFRASTRUCTURE_MANIFEST_SCHEMA_V1,
            "terminal_manifest": True,
            "passed": False,
            "authority_reference": context.authority_reference,
            "claim_path_intact": session.path_intact("claim_path"),
            "claim_reference": session.artifact_reference(
                "claim_path", require_path_match=False
            ),
            "infrastructure_failure_reference": session.artifact_reference(
                "infrastructure_failure_path"
            ),
            **output_evidence,
        }
    )


def write_smoke_infrastructure_terminal(
    *,
    context: Phase6SmokeLaunchContext,
    session: SecureSmokeOutputSession,
    stage: str,
    error: BaseException,
    max_attempts: int = 3,
) -> Mapping[str, Any]:
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts < 1
    ):
        raise ValueError("infrastructure terminal max_attempts must be positive")
    primary_preserved = False
    primary_result_artifact_hash: str | None = None
    if session.available_at_reviewed_path("public_result_path"):
        try:
            primary = session.read_json("public_result_path")
            parse_smoke_terminal_result(primary)
            primary_preserved = all(
                primary[name]
                == {
                    "smoke_authority_artifact_hash": context.authority[
                        "artifact_hash"
                    ],
                    "smoke_launch_claim_artifact_hash": context.claim[
                        "artifact_hash"
                    ],
                    "smoke_proposal_manifest_artifact_hash": (
                        context.proposal_manifest["artifact_hash"]
                    ),
                    "preflight_before_runtime_artifact_hash": context.preflight[
                        "artifact_hash"
                    ],
                }[name]
                for name in (
                    "smoke_authority_artifact_hash",
                    "smoke_launch_claim_artifact_hash",
                    "smoke_proposal_manifest_artifact_hash",
                    "preflight_before_runtime_artifact_hash",
                )
            )
            if primary_preserved:
                primary_result_artifact_hash = primary["artifact_hash"]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            primary_preserved = False
    if not session.has_role("infrastructure_failure_path"):
        raise RuntimeError("infrastructure failure descriptor was not reserved")
    last_error: BaseException | None = None
    for _attempt in range(max_attempts):
        try:
            failure = build_smoke_infrastructure_failure(
                context=context,
                stage=stage,
                error=error,
                primary_result_preserved=primary_preserved,
                primary_result_artifact_hash=primary_result_artifact_hash,
            )
            _write_json_idempotent_with_retry(
                session=session,
                role="infrastructure_failure_path",
                payload=failure,
                parser=parse_smoke_infrastructure_failure,
                max_attempts=max_attempts,
            )
            break
        except Exception as retry_error:
            last_error = retry_error
    else:
        assert last_error is not None
        raise RuntimeError(
            "secure smoke infrastructure failure could not be built and sealed "
            f"after {max_attempts} attempts"
        ) from last_error
    if not session.has_role("infrastructure_manifest_path"):
        raise RuntimeError("infrastructure manifest descriptor was not reserved")
    last_error = None
    for _attempt in range(max_attempts):
        try:
            manifest = build_smoke_infrastructure_manifest(
                context=context,
                session=session,
            )
            return _write_json_idempotent_with_retry(
                session=session,
                role="infrastructure_manifest_path",
                payload=manifest,
                parser=parse_smoke_infrastructure_manifest,
                max_attempts=max_attempts,
            )
        except Exception as retry_error:
            last_error = retry_error
    assert last_error is not None
    raise RuntimeError(
        "secure smoke infrastructure manifest could not be built and sealed "
        f"after {max_attempts} attempts"
    ) from last_error


def _write_json_idempotent_with_retry(
    *,
    session: SecureSmokeOutputSession,
    role: str,
    payload: Mapping[str, Any],
    parser: Any,
    max_attempts: int,
) -> Mapping[str, Any]:
    """Seal one reserved artifact without replacing valid completed bytes."""

    last_error: BaseException | None = None
    for _attempt in range(max_attempts):
        try:
            existing: Mapping[str, Any] | None = None
            if session.nonempty(role, require_path_match=False):
                try:
                    candidate = session.read_json(role, require_path_match=False)
                    parser(candidate)
                    existing = candidate
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    # A prior interrupted attempt may have left partial bytes.
                    existing = None
            if existing is not None:
                if canonical_artifact_payload_hash(existing) != (
                    canonical_artifact_payload_hash(payload)
                ):
                    raise RuntimeError(
                        f"secure smoke {role} already contains a different "
                        "valid terminal artifact"
                    )
                fd = session.fd(role)
                os.fsync(fd)
                session.directories.assert_fd_matches_path(role, fd)
                return existing
            return session.write_json(
                role,
                payload,
                parser=parser,
            )
        except Exception as retry_error:
            last_error = retry_error
    assert last_error is not None
    raise RuntimeError(
        f"secure smoke {role} could not be sealed after {max_attempts} attempts"
    ) from last_error


def parse_smoke_infrastructure_manifest(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    fields = (
        "schema",
        "terminal_manifest",
        "passed",
        "authority_reference",
        "claim_path_intact",
        "claim_reference",
        "infrastructure_failure_reference",
        "public_result_nonempty",
        "public_result_reserved",
        "public_result_path_intact",
        "public_result_reference",
        "public_progress_nonempty",
        "public_progress_reserved",
        "public_progress_path_intact",
        "public_progress_reference",
        "output_manifest_nonempty",
        "output_manifest_reserved",
        "output_manifest_path_intact",
        "output_manifest_reference",
        "log_nonempty",
        "log_reserved",
        "log_path_intact",
        "log_reference",
        "private_samples_nonempty",
        "private_samples_reserved",
        "private_samples_path_intact",
        "private_samples_reference",
        "artifact_hash",
    )
    _require_exact_keys(payload, required=fields, label="infrastructure manifest")
    if payload.get("schema") != HMC_PHASE6_SMOKE_INFRASTRUCTURE_MANIFEST_SCHEMA_V1:
        raise ValueError("unsupported smoke infrastructure manifest schema")
    if payload.get("terminal_manifest") is not True or payload.get("passed") is not False:
        raise ValueError("smoke infrastructure manifest terminal status mismatch")
    expected = {
        "authority_reference": HMC_PHASE6_SMOKE_AUTHORITY_SCHEMA_V1,
        "claim_reference": HMC_PHASE6_SMOKE_LAUNCH_CLAIM_SCHEMA_V1,
        "infrastructure_failure_reference": (
            HMC_PHASE6_SMOKE_INFRASTRUCTURE_FAILURE_SCHEMA_V1
        ),
    }
    _require_bool(payload.get("claim_path_intact"), label="claim_path_intact")
    for name, schema in expected.items():
        if parse_phase5_artifact_reference(payload[name])["source_schema"] != schema:
            raise ValueError(f"infrastructure manifest {name} schema mismatch")
    for prefix in (
        "public_result",
        "public_progress",
        "output_manifest",
        "log",
        "private_samples",
    ):
        reserved = _require_bool(
            payload.get(f"{prefix}_reserved"), label=f"{prefix}_reserved"
        )
        nonempty = _require_bool(
            payload.get(f"{prefix}_nonempty"), label=f"{prefix}_nonempty"
        )
        path_intact = _require_bool(
            payload.get(f"{prefix}_path_intact"), label=f"{prefix}_path_intact"
        )
        reference_payload = payload[f"{prefix}_reference"]
        if not reserved:
            if nonempty or path_intact or reference_payload is not None:
                raise ValueError(
                    f"unreserved infrastructure output {prefix} carries evidence"
                )
            continue
        reference = parse_file_reference(reference_payload)
        if nonempty != (reference["byte_count"] > 0):
            raise ValueError(f"infrastructure manifest {prefix} size mismatch")
    _verify_hash(payload, label="smoke infrastructure manifest")
    return payload


def verify_smoke_infrastructure_manifest(
    payload: Mapping[str, Any],
    *,
    authority_path: str | Path,
    claim_path: str | Path,
    infrastructure_failure_path: str | Path,
    public_result_path: str | Path,
    public_progress_path: str | Path,
    output_manifest_path: str | Path,
    log_path: str | Path,
    private_samples_path: str | Path,
) -> Mapping[str, Any]:
    parse_smoke_infrastructure_manifest(payload)
    verify_phase5_artifact_reference(
        payload["authority_reference"], path=authority_path
    )
    if payload["claim_path_intact"] is True:
        verify_phase5_artifact_reference(payload["claim_reference"], path=claim_path)
    verify_phase5_artifact_reference(
        payload["infrastructure_failure_reference"],
        path=infrastructure_failure_path,
    )
    role_paths = {
        "public_result": public_result_path,
        "public_progress": public_progress_path,
        "output_manifest": output_manifest_path,
        "log": log_path,
        "private_samples": private_samples_path,
    }
    for prefix, path in role_paths.items():
        if payload[f"{prefix}_reserved"] is True:
            if payload[f"{prefix}_path_intact"] is True:
                verify_file_reference(payload[f"{prefix}_reference"], path=path)
        elif Path(path).exists() or Path(path).is_symlink():
            raise ValueError(f"unreserved infrastructure output exists: {prefix}")
    failure = _read_json(infrastructure_failure_path)
    parse_smoke_infrastructure_failure(failure)
    if failure["smoke_authority_artifact_hash"] != (
        payload["authority_reference"]["embedded_artifact_hash"]
    ) or failure["smoke_launch_claim_artifact_hash"] != (
        payload["claim_reference"]["embedded_artifact_hash"]
    ):
        raise ValueError("infrastructure failure/manifest cross-link mismatch")
    if failure["primary_result_preserved"] is True:
        if payload["public_result_path_intact"] is True:
            result = _read_json(public_result_path)
            parse_smoke_terminal_result(result)
            if result["artifact_hash"] != failure["primary_result_artifact_hash"]:
                raise ValueError("preserved primary result hash mismatch")
        elif not payload["public_result_reserved"] or not payload[
            "public_result_nonempty"
        ]:
            raise ValueError("replaced preserved primary lacks held-byte evidence")
    return payload


def parse_smoke_output_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = (
        "schema",
        "terminal_manifest",
        "passed",
        "authority_reference",
        "claim_reference",
        "progress_reference",
        "result_reference",
        "log_reference",
        "private_samples_available",
        "private_samples_reference",
        "infrastructure_failure_written",
        "infrastructure_failure_reservation_reference",
        "infrastructure_manifest_written",
        "infrastructure_manifest_reservation_reference",
        "artifact_hash",
    )
    _require_exact_keys(payload, required=fields, label="smoke output manifest")
    if payload.get("schema") != HMC_PHASE6_SMOKE_OUTPUT_MANIFEST_SCHEMA_V1:
        raise ValueError("unsupported smoke output manifest schema")
    if payload.get("terminal_manifest") is not True:
        raise ValueError("smoke output manifest must be terminal")
    _require_bool(payload.get("passed"), label="manifest passed")
    expected_schemas = {
        "authority_reference": HMC_PHASE6_SMOKE_AUTHORITY_SCHEMA_V1,
        "claim_reference": HMC_PHASE6_SMOKE_LAUNCH_CLAIM_SCHEMA_V1,
        "progress_reference": HMC_PHASE6_SMOKE_PROGRESS_SCHEMA_V1,
        "result_reference": (
            HMC_PHASE6_SMOKE_RESULT_SCHEMA_V1
            if payload.get("passed") is True
            else HMC_PHASE6_SMOKE_FAILURE_SCHEMA_V1
        ),
    }
    for name, schema in expected_schemas.items():
        reference = parse_phase5_artifact_reference(payload[name])
        if reference["source_schema"] != schema:
            raise ValueError(f"smoke output {name} schema mismatch")
    parse_file_reference(payload["log_reference"])
    available = _require_bool(
        payload.get("private_samples_available"),
        label="private_samples_available",
    )
    if available:
        parse_file_reference(payload["private_samples_reference"])
    elif payload.get("private_samples_reference") is not None:
        raise ValueError("unavailable private samples cannot carry a reference")
    if payload.get("passed") is True and not available:
        raise ValueError("passing output manifest requires private samples")
    for prefix in ("infrastructure_failure", "infrastructure_manifest"):
        if _require_bool(payload.get(f"{prefix}_written"), label=prefix) is not False:
            raise ValueError("normal smoke manifest cannot record infrastructure failure")
        reference = parse_file_reference(
            payload[f"{prefix}_reservation_reference"]
        )
        if reference["byte_count"] != 0:
            raise ValueError("normal smoke emergency reservation must remain empty")
    _verify_hash(payload, label="smoke output manifest")
    return payload


def verify_smoke_output_manifest(
    payload: Mapping[str, Any],
    *,
    proposal_path: str | Path,
    proposal_manifest_path: str | Path,
    authority_path: str | Path,
    claim_path: str | Path,
    progress_path: str | Path,
    result_path: str | Path,
    log_path: str | Path,
    private_samples_path: str | Path,
    infrastructure_failure_path: str | Path,
    infrastructure_manifest_path: str | Path,
) -> Mapping[str, Any]:
    parse_smoke_output_manifest(payload)
    verify_phase5_artifact_reference(
        payload["authority_reference"], path=authority_path
    )
    verify_phase5_artifact_reference(payload["claim_reference"], path=claim_path)
    verify_phase5_artifact_reference(
        payload["progress_reference"], path=progress_path
    )
    verify_phase5_artifact_reference(payload["result_reference"], path=result_path)
    verify_file_reference(payload["log_reference"], path=log_path)
    if payload["private_samples_available"] is True:
        verify_file_reference(
            payload["private_samples_reference"], path=private_samples_path
        )
    verify_file_reference(
        payload["infrastructure_failure_reservation_reference"],
        path=infrastructure_failure_path,
    )
    verify_file_reference(
        payload["infrastructure_manifest_reservation_reference"],
        path=infrastructure_manifest_path,
    )
    expected = build_smoke_output_manifest(
        proposal_path=proposal_path,
        proposal_manifest_path=proposal_manifest_path,
        authority_path=authority_path,
        claim_path=claim_path,
        progress_path=progress_path,
        result_path=result_path,
        log_path=log_path,
        private_samples_path=private_samples_path,
        infrastructure_failure_path=infrastructure_failure_path,
        infrastructure_manifest_path=infrastructure_manifest_path,
    )
    if canonical_artifact_payload_hash(payload) != canonical_artifact_payload_hash(
        expected
    ):
        raise ValueError("smoke output manifest does not match current bytes")
    return payload


def _consumed_evidence_signature(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


class ConsumedAttempt1EvidenceSession:
    """Pin, lock, and repeatedly verify immutable attempt-1 evidence."""

    def __init__(self, *, entries: Mapping[str, Mapping[str, Any]]) -> None:
        self.entries = {name: dict(entry) for name, entry in entries.items()}
        self.snapshots: dict[str, bytes] = {}
        self.report: dict[str, Mapping[str, Any]] = {}
        self._closed = False

    @classmethod
    def open(cls) -> "ConsumedAttempt1EvidenceSession":
        entries: dict[str, Mapping[str, Any]] = {}
        opened: list[int] = []
        try:
            for label, raw_path, size, digest, mode in (
                _consumed_attempt1_evidence_expectations()
            ):
                path = Path(raw_path)
                if path.resolve(strict=True) != path:
                    raise ValueError("consumed smoke evidence path contains a symlink")
                parent_fd = os.open(path.parent, _directory_open_flags())
                opened.append(parent_fd)
                parent_held = os.fstat(parent_fd)
                parent_current = os.stat(path.parent, follow_symlinks=False)
                if not stat.S_ISDIR(parent_current.st_mode) or (
                    parent_held.st_dev,
                    parent_held.st_ino,
                ) != (parent_current.st_dev, parent_current.st_ino):
                    raise RuntimeError("consumed smoke evidence parent changed")
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                fd = os.open(path.name, flags, dir_fd=parent_fd)
                opened.append(fd)
                try:
                    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(
                        "consumed smoke evidence is locked for mutation"
                    ) from error
                entries[label] = {
                    "path": path,
                    "parent_fd": parent_fd,
                    "fd": fd,
                    "expected_size": size,
                    "expected_sha256": digest,
                    "expected_mode": mode,
                    "parent_identity": (parent_held.st_dev, parent_held.st_ino),
                }
            session = cls(entries=entries)
            session.verify(capture=True)
            session.verify_semantics()
            return session
        except BaseException:
            for fd in reversed(opened):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise

    def __enter__(self) -> "ConsumedAttempt1EvidenceSession":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _assert_parent(self, entry: Mapping[str, Any]) -> None:
        path = entry["path"]
        held = os.fstat(entry["parent_fd"])
        current = os.stat(path.parent, follow_symlinks=False)
        identity = entry["parent_identity"]
        if not stat.S_ISDIR(current.st_mode) or (
            held.st_dev,
            held.st_ino,
        ) != identity or (current.st_dev, current.st_ino) != identity:
            raise RuntimeError("consumed smoke evidence parent identity changed")

    def _read_stable(self, label: str, *, capture_signature: bool = False) -> bytes:
        if self._closed:
            raise RuntimeError("consumed smoke evidence session is closed")
        entry = self.entries[label]
        self._assert_parent(entry)
        fd = entry["fd"]
        before = os.fstat(fd)
        current_before = os.stat(
            entry["path"].name,
            dir_fd=entry["parent_fd"],
            follow_symlinks=False,
        )
        before_signature = _consumed_evidence_signature(before)
        if (
            before_signature != _consumed_evidence_signature(current_before)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_gid != os.getgid()
        ):
            raise RuntimeError(
                "consumed smoke evidence identity, owner, or link count changed"
            )

        def read_once() -> bytes:
            chunks: list[bytes] = []
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(
                    fd,
                    min(1024 * 1024, before.st_size - offset),
                    offset,
                )
                if not chunk:
                    raise RuntimeError("consumed smoke evidence read ended early")
                chunks.append(chunk)
                offset += len(chunk)
            return b"".join(chunks)

        first = read_once()
        middle = os.fstat(fd)
        current_middle = os.stat(
            entry["path"].name,
            dir_fd=entry["parent_fd"],
            follow_symlinks=False,
        )
        second = read_once()
        after = os.fstat(fd)
        current_after = os.stat(
            entry["path"].name,
            dir_fd=entry["parent_fd"],
            follow_symlinks=False,
        )
        signatures = {
            before_signature,
            _consumed_evidence_signature(middle),
            _consumed_evidence_signature(after),
            _consumed_evidence_signature(current_middle),
            _consumed_evidence_signature(current_after),
        }
        self._assert_parent(entry)
        if len(signatures) != 1 or first != second:
            raise RuntimeError("consumed smoke evidence changed during verification")
        stable_signature = next(iter(signatures))
        if capture_signature:
            if "capture_signature" in entry:
                raise RuntimeError("consumed smoke evidence signature was recaptured")
            entry["capture_signature"] = stable_signature
        elif stable_signature != entry.get("capture_signature"):
            raise RuntimeError(
                "consumed smoke evidence changed since its capture-time signature"
            )
        if len(first) != entry["expected_size"]:
            raise ValueError("consumed smoke evidence byte count mismatch")
        if hashlib.sha256(first).hexdigest() != entry["expected_sha256"]:
            raise ValueError("consumed smoke evidence exact bytes mismatch")
        if stat.S_IMODE(after.st_mode) != entry["expected_mode"]:
            raise ValueError("consumed smoke evidence file mode mismatch")
        if after.st_nlink != 1:
            raise ValueError("consumed smoke evidence must have one hard link")
        return first

    def verify(self, *, capture: bool = False) -> Mapping[str, Mapping[str, Any]]:
        try:
            observed = {
                label: self._read_stable(label, capture_signature=capture)
                for label in self.entries
            }
        except ConsumedAttempt1EvidenceDriftError:
            raise
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            raise ConsumedAttempt1EvidenceDriftError(str(error)) from error
        if capture:
            self.snapshots = observed
            self.report = {
                label: {
                    "path": str(entry["path"]),
                    "byte_count": entry["expected_size"],
                    "file_sha256": entry["expected_sha256"],
                    "file_mode": f"{entry['expected_mode']:04o}",
                    "hard_link_count": 1,
                }
                for label, entry in self.entries.items()
            }
        elif observed != self.snapshots:
            raise ConsumedAttempt1EvidenceDriftError(
                "consumed smoke evidence differs from pinned snapshot"
            )
        return self.report

    def verify_semantics(self) -> Mapping[str, Mapping[str, Any]]:
        self.verify()
        _verify_consumed_attempt1_snapshot_semantics(self.snapshots)
        self.verify()
        return self.report

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for entry in reversed(tuple(self.entries.values())):
            for fd_name in ("fd", "parent_fd"):
                try:
                    os.close(entry[fd_name])
                except OSError:
                    pass


def _verify_consumed_attempt1_snapshot_semantics(
    snapshots: Mapping[str, bytes],
) -> None:
    """Validate proposal and terminal meaning from the pinned exact snapshots."""

    original_proposal = json.loads(snapshots["original_proposal"])
    original_manifest = json.loads(snapshots["original_proposal_manifest"])
    attempt1_proposal = json.loads(snapshots["attempt1_proposal"])
    attempt1_manifest = json.loads(snapshots["attempt1_proposal_manifest"])
    parse_smoke_authority_proposal(original_proposal)
    parse_smoke_authority_proposal_manifest(original_manifest)
    parse_smoke_authority_proposal(attempt1_proposal)
    parse_smoke_authority_proposal_manifest(attempt1_manifest)
    verify_smoke_authority_proposal_manifest(
        original_manifest,
        proposal_path=SUPERSEDED_PROPOSAL_PATH,
    )
    verify_smoke_authority_proposal_manifest(
        attempt1_manifest,
        proposal_path=SUPERSEDED_PROPOSAL_V2_PATH,
    )
    output_manifest = json.loads(snapshots["attempt1_output_manifest"])
    verify_smoke_output_manifest(
        output_manifest,
        proposal_path=SUPERSEDED_PROPOSAL_V2_PATH,
        proposal_manifest_path=SUPERSEDED_PROPOSAL_MANIFEST_V2_PATH,
        authority_path=SUPERSEDED_AUTHORITY_PATH,
        claim_path=SUPERSEDED_CLAIM_PATH,
        progress_path=SUPERSEDED_PUBLIC_PROGRESS_PATH,
        result_path=SUPERSEDED_PUBLIC_RESULT_PATH,
        log_path=SUPERSEDED_LOG_PATH,
        private_samples_path=SUPERSEDED_PRIVATE_SAMPLES_PATH,
        infrastructure_failure_path=SUPERSEDED_INFRASTRUCTURE_FAILURE_PATH,
        infrastructure_manifest_path=SUPERSEDED_INFRASTRUCTURE_MANIFEST_PATH,
    )
    result = json.loads(snapshots["attempt1_result"])
    progress = json.loads(snapshots["attempt1_progress"])
    if (
        result.get("passed") is not False
        or result.get("stage") != "preflight_passed"
        or result.get("reason") != "runtime_error:BrokenProcessPool"
        or result.get("worker_pids") != []
        or result.get("final_diagnostics") is not None
        or progress.get("burnin_checks") != []
        or progress.get("retained_checks") != []
        or output_manifest.get("private_samples_available") is not False
    ):
        raise ValueError("consumed smoke attempt-1 terminal classification mismatch")


class ConsumedAttempt1EvidenceDriftError(RuntimeError):
    """Signal that no new output bytes or worker may be created."""


def verify_consumed_attempt1_evidence() -> Mapping[str, Mapping[str, Any]]:
    """Verify attempt-1 evidence for a bounded read-only action."""

    with ConsumedAttempt1EvidenceSession.open() as session:
        return dict(session.verify_semantics())


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _exclusive_file_flags() -> int:
    return (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise OSError("secure smoke artifact write made no progress")
        offset += written


class PinnedSmokeOutputDirectories:
    """Pin every reviewed output parent without following repository symlinks."""

    def __init__(
        self,
        *,
        entries: Mapping[str, tuple[int, str, Path]],
        owned_fds: Sequence[int],
        parent_identities: Mapping[int, tuple[Path, int, int]],
    ) -> None:
        self.entries = dict(entries)
        self._owned_fds = list(owned_fds)
        self._parent_identities = dict(parent_identities)
        self._closed = False

    @classmethod
    def open(
        cls,
        paths: Mapping[str, Path],
        *,
        repo_root: Path = REPO_ROOT,
        existing_roles: Sequence[str] = (),
    ) -> "PinnedSmokeOutputDirectories":
        root = Path(repo_root)
        root_fd = os.open(root, _directory_open_flags())
        owned = [root_fd]
        parents: dict[tuple[str, ...], int] = {}
        parent_identities: dict[int, tuple[Path, int, int]] = {}
        entries: dict[str, tuple[int, str, Path]] = {}
        try:
            for role, path in paths.items():
                relative = path.relative_to(root)
                components = relative.parts
                parent_key = components[:-1]
                if parent_key not in parents:
                    current = os.dup(root_fd)
                    owned.append(current)
                    for component in parent_key:
                        next_fd = os.open(
                            component,
                            _directory_open_flags(),
                            dir_fd=current,
                        )
                        owned.append(next_fd)
                        current = next_fd
                    parents[parent_key] = current
                    parent_path = root.joinpath(*parent_key)
                    held = os.fstat(current)
                    observed = os.stat(parent_path, follow_symlinks=False)
                    if not stat.S_ISDIR(observed.st_mode) or (
                        held.st_dev,
                        held.st_ino,
                    ) != (observed.st_dev, observed.st_ino):
                        raise RuntimeError("smoke output parent changed while pinning")
                    parent_identities[current] = (
                        parent_path,
                        held.st_dev,
                        held.st_ino,
                    )
                entries[role] = (parents[parent_key], components[-1], path)
            instance = cls(
                entries=entries,
                owned_fds=owned,
                parent_identities=parent_identities,
            )
            existing = frozenset(existing_roles)
            if not existing.issubset(instance.entries):
                raise ValueError("existing output roles are not reviewed paths")
            for role in instance.entries:
                if role in existing:
                    fd = instance.open_existing_readonly(role)
                    try:
                        instance.assert_fd_matches_path(role, fd)
                    finally:
                        os.close(fd)
                else:
                    instance.assert_absent(role)
            return instance
        except Exception:
            for fd in reversed(owned):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise

    def _entry(self, role: str) -> tuple[int, str, Path]:
        if self._closed:
            raise RuntimeError("pinned smoke output directories are closed")
        try:
            return self.entries[role]
        except KeyError as error:
            raise KeyError(f"unknown smoke output role: {role}") from error

    def assert_absent(self, role: str) -> None:
        directory_fd, name, _path = self._entry(role)
        self._assert_parent_matches(directory_fd)
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FileExistsError(f"smoke {role} already exists")

    def assert_all_absent(self) -> None:
        for role in self.entries:
            self.assert_absent(role)

    def open_exclusive(self, role: str) -> int:
        directory_fd, name, _path = self._entry(role)
        self._assert_parent_matches(directory_fd)
        fd = os.open(name, _exclusive_file_flags(), 0o400, dir_fd=directory_fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise ValueError(f"smoke {role} is not a regular file")
        self._assert_parent_matches(directory_fd)
        return fd

    def open_existing_readonly(self, role: str) -> int:
        directory_fd, name, _path = self._entry(role)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        return os.open(name, flags, dir_fd=directory_fd)

    def unlink_existing(self, role: str, *, expected_fd: int) -> None:
        directory_fd, name, _path = self._entry(role)
        self.assert_fd_matches_path(role, expected_fd)
        os.unlink(name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        self.assert_absent(role)

    def assert_fd_matches_path(self, role: str, fd: int) -> None:
        directory_fd, name, _path = self._entry(role)
        self._assert_parent_matches(directory_fd)
        held = os.fstat(fd)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or (
            held.st_dev,
            held.st_ino,
        ) != (current.st_dev, current.st_ino):
            raise RuntimeError(f"smoke {role} path no longer names its pinned file")
        if held.st_nlink != 1 or current.st_nlink != 1:
            raise RuntimeError(f"smoke {role} link count changed")

    def _assert_parent_matches(self, directory_fd: int) -> None:
        path, device, inode = self._parent_identities[directory_fd]
        held = os.fstat(directory_fd)
        try:
            current = os.stat(path, follow_symlinks=False)
        except FileNotFoundError as error:
            raise RuntimeError("smoke output parent disappeared") from error
        if not stat.S_ISDIR(current.st_mode) or (
            held.st_dev,
            held.st_ino,
        ) != (device, inode) or (current.st_dev, current.st_ino) != (device, inode):
            raise RuntimeError("smoke output parent identity changed")

    def fsync_parent(self, role: str) -> None:
        directory_fd, _name, _path = self._entry(role)
        self._assert_parent_matches(directory_fd)
        os.fsync(directory_fd)
        self._assert_parent_matches(directory_fd)

    def assert_parent_for_role(self, role: str) -> None:
        directory_fd, _name, _path = self._entry(role)
        self._assert_parent_matches(directory_fd)

    def fsync_all_parents(self) -> None:
        seen: set[int] = set()
        for directory_fd, _name, _path in self.entries.values():
            if directory_fd not in seen:
                os.fsync(directory_fd)
                seen.add(directory_fd)

    def path(self, role: str) -> Path:
        return self._entry(role)[2]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fd in reversed(self._owned_fds):
            try:
                os.close(fd)
            except OSError:
                pass


def create_durable_launch_claim(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    pinned_directories: PinnedSmokeOutputDirectories | None = None,
    keep_open: bool = False,
    parser: Any = parse_launch_claim,
) -> int | None:
    """Atomically consume authority and persist the directory entry."""

    parser(payload)
    destination = Path(path)
    own_directory_fd = False
    if pinned_directories is None:
        if not destination.parent.is_dir() or destination.parent.is_symlink():
            raise FileNotFoundError("smoke claim parent must be an existing real directory")
        directory_fd = os.open(destination.parent, _directory_open_flags())
        own_directory_fd = True
        name = destination.name
    else:
        directory_fd, name, reviewed = pinned_directories._entry("claim_path")
        if destination != reviewed:
            raise ValueError("smoke claim path differs from pinned review path")
        pinned_directories.assert_parent_for_role("claim_path")
    fd: int | None = None
    try:
        fd = os.open(name, _exclusive_file_flags(), 0o400, dir_fd=directory_fd)
        encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
        _write_all(fd, encoded)
        os.fchmod(fd, stat.S_IRUSR)
        os.fsync(fd)
        os.fsync(directory_fd)
        if stat.S_IMODE(os.fstat(fd).st_mode) != stat.S_IRUSR:
            raise RuntimeError("smoke claim did not become owner-read-only")
        if pinned_directories is not None:
            pinned_directories.assert_parent_for_role("claim_path")
            pinned_directories.assert_fd_matches_path("claim_path", fd)
        if keep_open:
            if pinned_directories is None:
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                retained = os.open(name, flags, dir_fd=directory_fd)
                written_info = os.fstat(fd)
                retained_info = os.fstat(retained)
                if (written_info.st_dev, written_info.st_ino) != (
                    retained_info.st_dev,
                    retained_info.st_ino,
                ):
                    os.close(retained)
                    raise RuntimeError("smoke claim changed before read-only retention")
            else:
                retained = pinned_directories.open_existing_readonly("claim_path")
                try:
                    pinned_directories.assert_fd_matches_path("claim_path", retained)
                except BaseException:
                    os.close(retained)
                    raise
            return retained
        return None
    finally:
        if fd is not None:
            os.close(fd)
        if own_directory_fd:
            os.close(directory_fd)


def create_durable_launch_claim_with_consumed_evidence(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    pinned_directories: PinnedSmokeOutputDirectories,
    consumed_evidence_session: ConsumedAttempt1EvidenceSession,
    parser: Any = parse_launch_claim,
) -> int:
    """Couple exact evidence checks to the irreversible claim boundary."""

    consumed_evidence_session.verify()
    retained = create_durable_launch_claim(
        path,
        payload,
        pinned_directories=pinned_directories,
        keep_open=True,
        parser=parser,
    )
    assert retained is not None
    try:
        # A failure here deliberately leaves the permanent claim consumed.
        consumed_evidence_session.verify()
    except BaseException:
        os.close(retained)
        raise
    return retained


class SmokeOutputReservationError(RuntimeError):
    def __init__(
        self,
        *,
        role: str,
        session: "SecureSmokeOutputSession",
        cause: BaseException,
    ) -> None:
        super().__init__(f"secure smoke output reservation failed for {role}: {cause}")
        self.role = role
        self.session = session
        self.cause = cause
        self.__cause__ = cause


class SecureSmokeOutputSession:
    """Own exclusive output descriptors for the lifetime of one consumed smoke."""

    _RESERVATION_ORDER = (
        "infrastructure_failure_path",
        "infrastructure_manifest_path",
        "output_manifest_path",
        "log_path",
        "public_result_path",
        "public_progress_path",
        "private_samples_path",
    )

    def __init__(
        self,
        *,
        directories: PinnedSmokeOutputDirectories,
        claim_fd: int,
        consumed_evidence_session: Any | None = None,
    ) -> None:
        self.directories = directories
        self.fds: dict[str, int] = {"claim_path": claim_fd}
        self.consumed_evidence_session = consumed_evidence_session
        self._closed = False

    @classmethod
    def reserve(
        cls,
        *,
        directories: PinnedSmokeOutputDirectories,
        claim_fd: int,
        consumed_evidence_session: Any | None = None,
    ) -> "SecureSmokeOutputSession":
        session = cls(
            directories=directories,
            claim_fd=claim_fd,
            consumed_evidence_session=consumed_evidence_session,
        )
        failure: tuple[str, BaseException] | None = None
        if consumed_evidence_session is not None:
            consumed_evidence_session.verify()
        for role in cls._RESERVATION_ORDER:
            try:
                if consumed_evidence_session is not None:
                    consumed_evidence_session.verify()
                session.fds[role] = directories.open_exclusive(role)
                directories.assert_fd_matches_path(role, session.fds[role])
                directories.fsync_parent(role)
            except ConsumedAttempt1EvidenceDriftError as error:
                raise SmokeOutputReservationError(
                    role="consumed_attempt1_evidence",
                    session=session,
                    cause=error,
                ) from error
            except BaseException as error:
                if not isinstance(error, Exception):
                    failure = (role, error)
                    break
                if failure is None:
                    failure = (role, error)
        if failure is not None:
            role, error = failure
            raise SmokeOutputReservationError(
                role=role,
                session=session,
                cause=error,
            ) from error
        if consumed_evidence_session is not None:
            try:
                consumed_evidence_session.verify()
            except ConsumedAttempt1EvidenceDriftError as error:
                raise SmokeOutputReservationError(
                    role="consumed_attempt1_evidence",
                    session=session,
                    cause=error,
                ) from error
        return session

    def has_role(self, role: str) -> bool:
        return role in self.fds and not self._closed

    def fd(self, role: str) -> int:
        if self._closed:
            raise RuntimeError("secure smoke output session is closed")
        if self.consumed_evidence_session is not None:
            self.consumed_evidence_session.verify()
        try:
            fd = self.fds[role]
        except KeyError as error:
            raise KeyError(f"smoke output role was not reserved: {role}") from error
        self.directories.assert_fd_matches_path(role, fd)
        return fd

    def _read_bytes(self, role: str, *, require_path_match: bool = True) -> bytes:
        fd = self.fd(role) if require_path_match else self.fds[role]
        size = os.fstat(fd).st_size
        return os.pread(fd, size, 0)

    def read_bytes(self, role: str) -> bytes:
        return self._read_bytes(role)

    def write_bytes(self, role: str, data: bytes) -> None:
        if role == "claim_path":
            raise RuntimeError("permanent smoke claim is read-only")
        fd = self.fd(role)
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        _write_all(fd, data)
        os.fsync(fd)
        self.directories.assert_fd_matches_path(role, fd)

    def begin_binary_write(self, role: str) -> Any:
        if role == "claim_path":
            raise RuntimeError("permanent smoke claim is read-only")
        fd = self.fd(role)
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        return os.fdopen(os.dup(fd), "wb")

    def finish_binary_write(self, role: str) -> None:
        fd = self.fd(role)
        os.fsync(fd)
        self.directories.assert_fd_matches_path(role, fd)

    def write_json(
        self,
        role: str,
        payload: Mapping[str, Any],
        *,
        parser: Any | None = None,
    ) -> Mapping[str, Any]:
        if parser is not None:
            parser(payload)
        encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
        self.write_bytes(role, encoded)
        restored = self.read_json(role)
        if parser is not None:
            parser(restored)
        return restored

    def read_json(
        self, role: str, *, require_path_match: bool = True
    ) -> Mapping[str, Any]:
        payload = json.loads(
            self._read_bytes(role, require_path_match=require_path_match).decode(
                "utf-8"
            )
        )
        if not isinstance(payload, Mapping):
            raise ValueError(f"secure smoke {role} must contain a JSON object")
        return payload

    def nonempty(self, role: str, *, require_path_match: bool = True) -> bool:
        if not self.has_role(role):
            return False
        if require_path_match:
            self.fd(role)
        return os.fstat(self.fds[role]).st_size > 0

    def available_at_reviewed_path(self, role: str) -> bool:
        try:
            return self.nonempty(role, require_path_match=True)
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            return False

    def path_intact(self, role: str) -> bool:
        try:
            self.fd(role)
            return True
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            return False

    def file_reference(
        self, role: str, *, require_path_match: bool = True
    ) -> Mapping[str, Any]:
        data = self._read_bytes(role, require_path_match=require_path_match)
        path = self.directories.path(role)
        return {
            "schema": HMC_PHASE6_FILE_REFERENCE_SCHEMA_V1,
            "resolved_path_sha256": hashlib.sha256(
                str(path).encode("utf-8")
            ).hexdigest(),
            "file_sha256": hashlib.sha256(data).hexdigest(),
            "byte_count": len(data),
        }

    def artifact_reference(
        self,
        role: str,
        *,
        embedded_hash_rule: str = "canonical_without_hash",
        require_path_match: bool = True,
    ) -> Mapping[str, Any]:
        from bayesfilter.runtime import stable_config_hash

        data = self._read_bytes(role, require_path_match=require_path_match)
        payload = json.loads(data.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"secure smoke {role} must contain a JSON object")
        embedded = payload.get("artifact_hash")
        if embedded_hash_rule == "canonical_without_hash":
            expected = canonical_artifact_payload_hash(
                {key: value for key, value in payload.items() if key != "artifact_hash"}
            )
        elif embedded_hash_rule == "stable_without_hash":
            expected = "sha256:" + stable_config_hash(
                {key: value for key, value in payload.items() if key != "artifact_hash"}
            )
        elif embedded_hash_rule == "none":
            expected = None
        else:
            raise ValueError("unsupported embedded hash rule")
        if embedded != expected:
            raise ValueError(f"secure smoke {role} embedded hash mismatch")
        path = self.directories.path(role)
        return {
            "schema": "bayesfilter.hmc_identity_phase5_artifact_reference.v1",
            "source_schema": _require_string(
                payload.get("schema"), label=f"{role} source schema"
            ),
            "embedded_hash_rule": embedded_hash_rule,
            "embedded_artifact_hash": embedded,
            "canonical_payload_hash": canonical_artifact_payload_hash(payload),
            "resolved_path_sha256": hashlib.sha256(
                str(path).encode("utf-8")
            ).hexdigest(),
            "file_sha256": hashlib.sha256(data).hexdigest(),
            "byte_count": len(data),
        }

    def validate_for_runtime(self) -> None:
        if self.consumed_evidence_session is not None:
            self.consumed_evidence_session.verify()
        required = {"claim_path", *self._RESERVATION_ORDER}
        if set(self.fds) != required:
            raise RuntimeError("secure smoke output reservation is incomplete")
        for role, fd in self.fds.items():
            self.directories.assert_fd_matches_path(role, fd)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fd in self.fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self.directories.close()


def _resolve_reviewed_paths(paths: Mapping[str, Any]) -> Mapping[str, Path]:
    _strict_paths(paths)
    resolved = {
        name: (REPO_ROOT / str(paths[name])).resolve() for name in _PATH_FIELDS
    }
    for name, path in resolved.items():
        expected = REPO_ROOT / str(paths[name])
        if path != expected:
            raise ValueError(f"smoke {name} contains a symlink or path escape")
        if path == REPO_ROOT or REPO_ROOT not in path.parents:
            raise ValueError(f"smoke {name} escapes the repository")
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("smoke resolved paths must be distinct")
    return resolved


def _verify_no_path_alias_or_existing_output(
    *, proposal: Mapping[str, Any], config: Any
) -> Mapping[str, Path]:
    resolved = _resolve_reviewed_paths(proposal["paths"])
    governed = _protected_smoke_paths(config)
    if governed.intersection(resolved.values()):
        raise ValueError("smoke output aliases a governed or serious artifact")
    for name, path in resolved.items():
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"smoke {name} already exists")
        if not path.parent.is_dir() or path.parent.is_symlink():
            raise ValueError(f"smoke {name} parent must be an existing real directory")
    return resolved


def _protected_smoke_paths(config: Any) -> set[Path]:
    """Return every source or serious path forbidden to smoke outputs."""

    governed = {Path(item).resolve() for item in (
        V2_CONFIG_PATH,
        ADOPTION_RECORD_PATH,
        PREFLIGHT_PATH,
        PHASE5_MANIFEST_PATH,
        PROPOSAL_PATH,
        PROPOSAL_MANIFEST_PATH,
        AUTHORITY_PATH,
        SUPERSEDED_PROPOSAL_PATH,
        SUPERSEDED_PROPOSAL_MANIFEST_PATH,
        SUPERSEDED_PROPOSAL_V2_PATH,
        SUPERSEDED_PROPOSAL_MANIFEST_V2_PATH,
        SUPERSEDED_AUTHORITY_PATH,
        SUPERSEDED_CLAIM_PATH,
        SUPERSEDED_PUBLIC_RESULT_PATH,
        SUPERSEDED_PUBLIC_PROGRESS_PATH,
        SUPERSEDED_OUTPUT_MANIFEST_PATH,
        SUPERSEDED_INFRASTRUCTURE_FAILURE_PATH,
        SUPERSEDED_INFRASTRUCTURE_MANIFEST_PATH,
        SUPERSEDED_PRIVATE_SAMPLES_PATH,
        SUPERSEDED_LOG_PATH,
        PHASE6_SUBPLAN_PATH,
        *default_implementation_paths(sys.executable).values(),
    )}
    from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (
        phase7_governed_source_paths,
    )

    governed.update(path.resolve() for path in phase7_governed_source_paths(config).values())
    for value in config.payload["artifacts"].values():
        governed.add((REPO_ROOT / str(value)).resolve())
    source_config_path = REPO_ROOT / str(config.payload["source_tuning_config_path"])
    source_config = _read_json(source_config_path)
    for value in source_config.get("artifact_paths", {}).values():
        governed.add((REPO_ROOT / str(value)).resolve())
    fixture_path = source_config.get("truth_and_data", {}).get("artifact_path")
    if fixture_path:
        governed.add((REPO_ROOT / str(fixture_path)).resolve())
    return governed


def prepare_smoke_launch(
    *, authority_path: str | Path, current_command: Sequence[str]
) -> Phase6SmokeLaunchContext:
    """Verify and permanently consume one reviewed smoke authority."""

    normalized_command = tuple(
        _require_string(item, label="current command item") for item in current_command
    )
    authority_source = Path(authority_path)
    if authority_source.resolve() != AUTHORITY_PATH:
        raise ValueError("smoke authority path mismatch")
    consumed_evidence_session = ConsumedAttempt1EvidenceSession.open()
    pinned_directories: PinnedSmokeOutputDirectories | None = None
    claim_fd: int | None = None
    try:
        proposal, manifest, config, live_preflight = (
            verify_default_smoke_authority_proposal_bundle(
                python_executable=sys.executable,
                consumed_evidence_session=consumed_evidence_session,
            )
        )
        authority = _read_json(authority_source)
        parse_smoke_authority(authority)
        if authority["proposal_manifest_reference"]["embedded_artifact_hash"] != (
            manifest["artifact_hash"]
        ) or authority["proposal_manifest_reference"][
            "canonical_payload_hash"
        ] != canonical_artifact_payload_hash(manifest):
            raise ValueError("captured smoke authority/manifest mismatch")
        verify_smoke_authority(
            authority,
            proposal_manifest_path=PROPOSAL_MANIFEST_PATH,
        )
        if normalized_command != tuple(proposal["command"]):
            raise ValueError("smoke command does not match the reviewed proposal")
        implementation_source_bundle = build_verified_implementation_source_bundle(
            proposal["implementation_references"],
            python_executable=proposal["command"][0],
        )
        authority_reference = build_phase5_artifact_reference(
            authority_source,
            embedded_hash_rule="canonical_without_hash",
        )
        if authority_reference["embedded_artifact_hash"] != authority[
            "artifact_hash"
        ] or authority_reference["canonical_payload_hash"] != (
            canonical_artifact_payload_hash(authority)
        ):
            raise ValueError("captured smoke authority/reference mismatch")
        resolved_paths = _verify_no_path_alias_or_existing_output(
            proposal=proposal,
            config=config,
        )
        pinned_directories = PinnedSmokeOutputDirectories.open(resolved_paths)
        claim = build_launch_claim(
            authority=authority,
            proposal_manifest=manifest,
            command=normalized_command,
            paths=proposal["paths"],
            pid=os.getpid(),
        )
        claim_fd = create_durable_launch_claim_with_consumed_evidence(
            resolved_paths["claim_path"],
            claim,
            pinned_directories=pinned_directories,
            consumed_evidence_session=consumed_evidence_session,
        )
    except BaseException:
        if claim_fd is not None:
            os.close(claim_fd)
        if pinned_directories is not None:
            pinned_directories.close()
        consumed_evidence_session.close()
        raise
    token = object()
    prepared_snapshot_hash = _prepared_context_snapshot_hash(
        config=config,
        authority_reference=authority_reference,
        proposal=proposal,
        proposal_manifest=manifest,
        authority=authority,
        claim=claim,
        preflight=live_preflight,
        command=normalized_command,
        paths=resolved_paths,
        implementation_source_bundle=implementation_source_bundle,
    )
    context = Phase6SmokeLaunchContext(
        config=config,
        preflight=live_preflight,
        proposal=proposal,
        proposal_manifest=manifest,
        authority=authority,
        authority_reference=authority_reference,
        claim=claim,
        paths=resolved_paths,
        command=normalized_command,
        implementation_source_bundle=implementation_source_bundle,
        output_directories=pinned_directories,
        claim_fd=claim_fd,
        consumed_evidence_session=consumed_evidence_session,
        output_session=None,
        prepared_snapshot_hash=prepared_snapshot_hash,
        _prepared_token=token,
    )
    _PREPARED_CONTEXT_TOKENS[id(context)] = token
    _PREPARED_CONTEXT_EVIDENCE_SESSIONS[id(context)] = consumed_evidence_session
    return context


def verify_default_smoke_authority_proposal_bundle(
    *,
    python_executable: str | Path,
    consumed_evidence_session: ConsumedAttempt1EvidenceSession | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Any, Mapping[str, Any]]:
    """Verify the exact proposal, Phase 5 evidence, and live preflight bundle."""

    proposal = _read_json(PROPOSAL_PATH)
    manifest = _read_json(PROPOSAL_MANIFEST_PATH)
    parse_smoke_authority_proposal(proposal)
    parse_smoke_authority_proposal_manifest(manifest)
    if manifest["proposal_reference"]["embedded_artifact_hash"] != (
        proposal["artifact_hash"]
    ) or manifest["proposal_reference"]["canonical_payload_hash"] != (
        canonical_artifact_payload_hash(proposal)
    ):
        raise ValueError("captured smoke proposal/manifest mismatch")
    verify_smoke_authority_proposal_manifest(manifest, proposal_path=PROPOSAL_PATH)
    config, live_preflight = verify_default_smoke_authority_proposal_candidate(
        proposal,
        python_executable=python_executable,
        consumed_evidence_session=consumed_evidence_session,
    )
    return proposal, manifest, config, live_preflight


def verify_default_smoke_authority_proposal_candidate(
    proposal: Mapping[str, Any],
    *,
    python_executable: str | Path,
    consumed_evidence_session: ConsumedAttempt1EvidenceSession | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Verify all live proposal inputs before any proposal artifact is written."""

    if consumed_evidence_session is None:
        verify_consumed_attempt1_evidence()
    else:
        consumed_evidence_session.verify_semantics()

    from bayesfilter.inference.hmc_identity_adoption import (
        parse_phase5_preflight_report,
        verify_phase5_output_manifest,
    )
    from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (
        DeterministicLGSSMPhase7Config,
        validate_phase7_inputs,
    )

    parse_smoke_authority_proposal(proposal)
    verify_smoke_authority_proposal(
        proposal,
        phase6_subplan_path=PHASE6_SUBPLAN_PATH,
        artifact_paths={
            "v2_config_reference": V2_CONFIG_PATH,
            "adoption_record_reference": ADOPTION_RECORD_PATH,
            "preflight_reference": PREFLIGHT_PATH,
            "phase5_manifest_reference": PHASE5_MANIFEST_PATH,
        },
        implementation_paths=default_implementation_paths(python_executable),
    )
    config = DeterministicLGSSMPhase7Config.load(V2_CONFIG_PATH)
    stored_preflight = _read_json(PREFLIGHT_PATH)
    parse_phase5_preflight_report(stored_preflight)
    verify_phase5_output_manifest(
        _read_json(PHASE5_MANIFEST_PATH),
        v2_config_path=V2_CONFIG_PATH,
        adoption_record_path=ADOPTION_RECORD_PATH,
        preflight_report_path=PREFLIGHT_PATH,
    )
    live_preflight = validate_phase7_inputs(config)
    if canonical_artifact_payload_hash(live_preflight) != (
        canonical_artifact_payload_hash(stored_preflight)
    ):
        raise ValueError("live smoke preflight differs from the Phase 5 artifact")
    if live_preflight["identity_hashes"]["transition_identity_hash"] != (
        proposal["transition_identity_hash"]
    ) or live_preflight["identity_hashes"]["smoke_execution_contract_hash"] != (
        proposal["smoke_execution_identity_hash"]
    ):
        raise ValueError("live smoke typed identity mismatch")
    if default_smoke_runtime() != proposal["runtime"]:
        raise ValueError("live smoke runtime environment mismatch")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("smoke parent requires CUDA_VISIBLE_DEVICES=-1")
    return config, live_preflight


def write_phase6_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    parser: Any,
    consumed_evidence_session: ConsumedAttempt1EvidenceSession | None = None,
) -> Mapping[str, Any]:
    """Create once, or recover only identical complete bytes, crash-durably."""

    parser(payload)
    if consumed_evidence_session is not None:
        consumed_evidence_session.verify()
    destination = Path(path)
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise FileNotFoundError("Phase 6 artifact parent must be an existing directory")
    directory_fd = os.open(destination.parent, _directory_open_flags())
    fd: int | None = None
    try:
        encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
        parent_identity = os.fstat(directory_fd)
        current_parent = os.stat(destination.parent, follow_symlinks=False)
        if (parent_identity.st_dev, parent_identity.st_ino) != (
            current_parent.st_dev,
            current_parent.st_ino,
        ):
            raise RuntimeError("Phase 6 artifact parent identity changed")
        try:
            fd = os.open(
                destination.name,
                _exclusive_file_flags(),
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            fd = os.open(destination.name, flags, dir_fd=directory_fd)
            existing = os.pread(fd, os.fstat(fd).st_size, 0)
            if existing != encoded:
                raise FileExistsError(
                    "Phase 6 artifact already exists with different or partial bytes"
                )
        else:
            _write_all(fd, encoded)
        os.fsync(fd)
        os.fsync(directory_fd)
        written = os.fstat(fd)
        current = os.stat(
            destination.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        current_parent = os.stat(destination.parent, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or (
            written.st_dev,
            written.st_ino,
        ) != (current.st_dev, current.st_ino):
            raise RuntimeError("Phase 6 artifact path changed during creation")
        if written.st_nlink != 1 or current.st_nlink != 1:
            raise RuntimeError("Phase 6 artifact link count changed")
        if (parent_identity.st_dev, parent_identity.st_ino) != (
            current_parent.st_dev,
            current_parent.st_ino,
        ):
            raise RuntimeError("Phase 6 artifact parent changed during creation")
        restored_bytes = os.pread(fd, written.st_size, 0)
        if restored_bytes != encoded:
            raise RuntimeError("Phase 6 artifact bytes changed during creation")
        restored = json.loads(restored_bytes.decode("utf-8"))
        if not isinstance(restored, Mapping):
            raise ValueError("Phase 6 JSON artifact must contain an object")
        parser(restored)
        if consumed_evidence_session is not None:
            consumed_evidence_session.verify()
        return restored
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory_fd)
