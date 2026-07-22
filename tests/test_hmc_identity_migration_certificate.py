from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bayesfilter.inference.hmc_identity import (
    artifact_file_sha256,
    canonical_artifact_payload_hash,
)
from bayesfilter.inference.hmc_identity_migration_certificate import (
    ACTIVE_GATE_STATUS,
    ADOPTION_STATUS,
    CERTIFICATE_DECISION,
    CERTIFICATE_STATUS,
    COMPARISON_CONTRACT,
    HMC_MIGRATION_CERTIFICATE_OUTPUT_MANIFEST_SCHEMA_V1,
    LEGACY_HASH_KEYS,
    SOURCE_REFERENCE_KEYS,
    assert_public_certificate_redacted,
    build_certificate_output_manifest,
    build_migration_certificate,
    build_public_certificate_proposal,
    parse_certificate_output_manifest,
    parse_migration_certificate,
    parse_public_certificate_proposal,
    verify_certificate_output_manifest,
    verify_migration_certificate_sources,
    write_certificate_output_manifest,
    write_migration_certificate,
    write_public_certificate_proposal,
)
from bayesfilter.runtime import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / (
    "docs/benchmarks/artifacts/"
    "multidim_lgssm_serious_hmc_tuning_2026_07_09"
)
PUBLIC_ROOT = ROOT / (
    "docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11"
)
SOURCE_PATHS = {
    "phase7_config": ROOT
    / "docs/benchmarks/configs/multidim_lgssm_phase7_burnin_sampling_2026_07_11.json",
    "refreshed_kernel": ARTIFACT_ROOT / "kernel_tuning.json",
    "refreshed_private_replay": ARTIFACT_ROOT
    / "private_diagnostics/kernel_tuning_replay.json",
    "phase3_sidecar": ARTIFACT_ROOT
    / "private_diagnostics/hmc_semantic_identity_phase3_sidecar.json",
    "phase3_input_manifest": ARTIFACT_ROOT
    / "private_diagnostics/hmc_semantic_identity_phase3_input_integrity_manifest.json",
    "phase3_public_record": PUBLIC_ROOT / "candidate_semantic_validation.json",
    "phase3_output_manifest": PUBLIC_ROOT / "output_integrity_manifest.json",
}

HISTORICAL_HASHES = {
    "fixture": "sha256:5b8f4ae78e00b69fb4b75deb1ccd3facfd7869f5d9fc0c7cb87eafdad8c8793e",
    "xla_compile": "sha256:8941b369f6280ebc3c124220a9bab21f6889228deb92121d63f2fefba3ea6842",
    "geometry": "sha256:e2b9531e86f85a662c4da26595e0ab082dd8a1a29d2dbb83b31b076bbf7683ac",
    "mass": "sha256:92536fbd13e1ba89c53bfcc874355194b8d2d097ea498d22b5ccd7c318490d8e",
    "adapter": "3a71b33479f6eb3681584d3a7a31550a17a5731116253131e9a21a9b5d21af08",
    "selected_step": "ec7db59e51465eee95658167e1f7596e21d9ab0efdac11f54c2d397aa270ab40",
    "public_final_kernel": "8ddf25a3b572893e19e814fad5ca5b6150718e36f760c159b47db1231d92ffff",
    "private_loop_final_kernel": "391558a9b5f4cdc1b9dff9a5e9bceba668dedded7298c1d8c76daea42f42039a",
    "selected_trajectory": "6eaf7a563353b278a71dcfbe2515fda6d46c47ab2e38996b6b61fab1bbbd13b3",
}
REFRESHED_HASHES = {
    **{
        name: HISTORICAL_HASHES[name]
        for name in (
            "fixture",
            "xla_compile",
            "geometry",
            "mass",
            "adapter",
            "selected_step",
        )
    },
    "public_final_kernel": "07910941750ad6b882d357411c8ed9a1faa36b886f6125e78af8306ccdae7fbf",
    "private_loop_final_kernel": "2823e20048c0969b79931604462ba142a34aed06fd8cfab3baf03eab89c0168f",
    "selected_trajectory": "3f4b33680ed1e8365670772afe313e479a3a43a4a1c3f2ac2a77c49795aeb04b",
}


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="module")
def source_payloads() -> dict[str, dict]:
    return {name: _read_json(path) for name, path in SOURCE_PATHS.items()}


def _build_certificate(source_payloads: dict[str, dict]) -> dict:
    return dict(
        build_migration_certificate(
            phase7_config_payload=source_payloads["phase7_config"],
            refreshed_kernel_payload=source_payloads["refreshed_kernel"],
            refreshed_private_replay_payload=source_payloads[
                "refreshed_private_replay"
            ],
            phase3_sidecar=source_payloads["phase3_sidecar"],
            phase3_input_manifest=source_payloads["phase3_input_manifest"],
            phase3_public_record=source_payloads["phase3_public_record"],
            phase3_output_manifest=source_payloads["phase3_output_manifest"],
            phase3_paths={
                "sidecar": SOURCE_PATHS["phase3_sidecar"],
                "input_manifest": SOURCE_PATHS["phase3_input_manifest"],
                "public_record": SOURCE_PATHS["phase3_public_record"],
                "output_manifest": SOURCE_PATHS["phase3_output_manifest"],
            },
            source_paths={
                name: SOURCE_PATHS[name]
                for name in (
                    "phase7_config",
                    "refreshed_kernel",
                    "refreshed_private_replay",
                )
            },
        )
    )


@pytest.fixture(scope="module")
def certificate(source_payloads: dict[str, dict]) -> dict:
    return _build_certificate(source_payloads)


def _rehash(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    result.pop("artifact_hash", None)
    result["artifact_hash"] = canonical_artifact_payload_hash(result)
    return result


def _write_proposal_bundle(
    tmp_path: Path,
    *,
    certificate: dict,
    source_payloads: dict[str, dict],
) -> tuple[Path, Path, dict]:
    certificate_path = tmp_path / "certificate.json"
    proposal_path = tmp_path / "proposal.json"
    write_migration_certificate(certificate_path, certificate)
    proposal = dict(
        build_public_certificate_proposal(
            certificate=certificate,
            certificate_path=certificate_path,
            phase3_public_record=source_payloads["phase3_public_record"],
            phase3_public_record_path=SOURCE_PATHS["phase3_public_record"],
        )
    )
    write_public_certificate_proposal(proposal_path, proposal)
    return certificate_path, proposal_path, proposal


def test_real_certificate_round_trip_owns_exact_hashes_and_classifications(
    certificate: dict,
) -> None:
    assert parse_migration_certificate(certificate) == certificate
    assert verify_migration_certificate_sources(
        certificate,
        source_paths=SOURCE_PATHS,
    ) == certificate
    assert (
        certificate["status"],
        certificate["decision"],
        certificate["adoption_status"],
        certificate["active_gate_status"],
    ) == (
        CERTIFICATE_STATUS,
        CERTIFICATE_DECISION,
        ADOPTION_STATUS,
        ACTIVE_GATE_STATUS,
    )
    assert certificate["human_approval_required"] is True
    assert tuple(certificate["historical_expected_hashes"]) == LEGACY_HASH_KEYS
    assert tuple(certificate["refreshed_legacy_hashes"]) == LEGACY_HASH_KEYS
    assert certificate["historical_expected_hashes"] == HISTORICAL_HASHES
    assert certificate["refreshed_legacy_hashes"] == REFRESHED_HASHES
    assert tuple(
        (item["comparison_id"], item["classification"], item["basis"])
        for item in certificate["comparisons"]
    ) == COMPARISON_CONTRACT
    assert tuple(certificate["source_references"]) == SOURCE_REFERENCE_KEYS
    for name, reference in certificate["source_references"].items():
        assert reference["file_sha256"] == artifact_file_sha256(SOURCE_PATHS[name])
        assert reference["byte_count"] == SOURCE_PATHS[name].stat().st_size


def test_unavailable_historical_typed_identities_are_strictly_unsupported(
    certificate: dict,
) -> None:
    comparisons = {item["comparison_id"]: item for item in certificate["comparisons"]}
    for name in (
        "historical_refreshed_typed_transition_identity",
        "historical_refreshed_typed_execution_identity",
    ):
        assert comparisons[name]["classification"] == "unsupported"
        assert comparisons[name]["left_value"] is None
        assert comparisons[name]["right_value"] is None

    tampered = copy.deepcopy(certificate)
    item = tampered["comparisons"][9]
    item["classification"] = "equal"
    item["left_value"] = certificate["refreshed_typed_identities"][
        "transition_identity_hash"
    ]
    item["right_value"] = item["left_value"]
    with pytest.raises(ValueError, match="comparison contract mismatch"):
        parse_migration_certificate(_rehash(tampered))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "approved"),
        ("decision", "ADOPT_TYPED_IDENTITY_BASELINE"),
        ("adoption_status", "adopted"),
        ("active_gate_status", "typed_gate_active"),
        ("human_approval_required", False),
    ],
)
def test_certificate_rejects_approval_state_tampering(
    certificate: dict,
    field: str,
    value: object,
) -> None:
    tampered = copy.deepcopy(certificate)
    tampered[field] = value
    with pytest.raises(ValueError, match="approval boundary mismatch"):
        parse_migration_certificate(_rehash(tampered))


def test_live_source_verifier_rejects_byte_and_path_substitution(
    tmp_path: Path,
    certificate: dict,
) -> None:
    byte_tampered = tmp_path / "kernel-byte-tampered.json"
    byte_tampered.write_bytes(SOURCE_PATHS["refreshed_kernel"].read_bytes() + b" ")
    paths = dict(SOURCE_PATHS)
    paths["refreshed_kernel"] = byte_tampered
    with pytest.raises(ValueError, match="source reference mismatch: refreshed_kernel"):
        verify_migration_certificate_sources(certificate, source_paths=paths)

    copied = tmp_path / "phase7-config-copy.json"
    copied.write_bytes(SOURCE_PATHS["phase7_config"].read_bytes())
    paths = dict(SOURCE_PATHS)
    paths["phase7_config"] = copied
    with pytest.raises(ValueError, match="not the governed input: phase7_config"):
        verify_migration_certificate_sources(certificate, source_paths=paths)


def test_public_proposal_is_redacted_and_proposal_only(
    tmp_path: Path,
    certificate: dict,
    source_payloads: dict[str, dict],
) -> None:
    _certificate_path, proposal_path, proposal = _write_proposal_bundle(
        tmp_path,
        certificate=certificate,
        source_payloads=source_payloads,
    )
    restored = _read_json(proposal_path)
    assert parse_public_certificate_proposal(restored) == restored
    assert_public_certificate_redacted(restored)
    assert restored["adoption_status"] == ADOPTION_STATUS
    assert restored["active_gate_status"] == ACTIVE_GATE_STATUS
    assert restored["human_approval_required"] is True
    serialized = json.dumps(proposal, sort_keys=True)
    for forbidden in (
        "observations",
        "step_size",
        "num_leapfrog_steps",
        "root_seed",
        "tensorflow_version",
        "stage_lineage",
        "historical_expected_hashes",
        "refreshed_typed_identities",
    ):
        assert f'"{forbidden}":' not in serialized

    with pytest.raises(ValueError, match="forbidden keys"):
        assert_public_certificate_redacted({"nested": {"step_size": 0.1}})
    with pytest.raises(ValueError, match="absolute private path"):
        assert_public_certificate_redacted({"nested": "/private/evidence.json"})


def test_terminal_manifest_rejects_protected_public_reference_mismatch(
    tmp_path: Path,
    certificate: dict,
    source_payloads: dict[str, dict],
) -> None:
    certificate_path, proposal_path, proposal = _write_proposal_bundle(
        tmp_path,
        certificate=certificate,
        source_payloads=source_payloads,
    )
    tampered = copy.deepcopy(proposal)
    tampered["protected_certificate_reference"]["file_sha256"] = "f" * 64
    atomic_write_json(proposal_path, _rehash(tampered))
    with pytest.raises(
        ValueError,
        match="does not reference the protected certificate",
    ):
        build_certificate_output_manifest(
            certificate_path=certificate_path,
            public_proposal_path=proposal_path,
        )


def test_terminal_manifest_is_acyclic_and_detects_exact_byte_tamper(
    tmp_path: Path,
    certificate: dict,
    source_payloads: dict[str, dict],
) -> None:
    certificate_path, proposal_path, _proposal = _write_proposal_bundle(
        tmp_path,
        certificate=certificate,
        source_payloads=source_payloads,
    )
    manifest = dict(
        build_certificate_output_manifest(
            certificate_path=certificate_path,
            public_proposal_path=proposal_path,
        )
    )
    manifest_path = tmp_path / "terminal-manifest.json"
    write_certificate_output_manifest(manifest_path, manifest)
    restored = _read_json(manifest_path)
    assert parse_certificate_output_manifest(restored) == restored
    assert verify_certificate_output_manifest(
        restored,
        certificate_path=certificate_path,
        public_proposal_path=proposal_path,
    ) == restored
    assert HMC_MIGRATION_CERTIFICATE_OUTPUT_MANIFEST_SCHEMA_V1 not in json.dumps(
        restored["outputs"],
        sort_keys=True,
    )

    proposal_path.write_bytes(proposal_path.read_bytes() + b" ")
    assert parse_public_certificate_proposal(_read_json(proposal_path))
    with pytest.raises(ValueError, match="current bytes"):
        verify_certificate_output_manifest(
            restored,
            certificate_path=certificate_path,
            public_proposal_path=proposal_path,
        )
