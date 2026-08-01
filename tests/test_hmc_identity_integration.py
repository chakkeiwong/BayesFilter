from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bayesfilter.inference.hmc_identity import canonical_artifact_payload_hash
from bayesfilter.inference.hmc_identity_integration import (
    HMC_IDENTITY_INPUT_INTEGRITY_MANIFEST_SCHEMA_V1,
    HMC_IDENTITY_OUTPUT_INTEGRITY_MANIFEST_SCHEMA_V1,
    HMC_IDENTITY_PRIVATE_SIDECAR_SCHEMA_V1,
    HMC_IDENTITY_PUBLIC_VALIDATION_SCHEMA_V1,
    LegacyValidatorResultV1,
    PHASE3_CANDIDATE_CHECK_KEYS,
    PHASE3_DECISION,
    PHASE3_LEGACY_VETO_CODE,
    PHASE3_PUBLIC_NONCLAIMS,
    PHASE3_STATUS,
    assert_public_validation_redacted,
    build_input_integrity_manifest,
    build_output_integrity_manifest,
    build_private_identity_sidecar,
    build_public_validation_record,
    build_selection_provenance_from_tuning_payload,
    parse_input_integrity_manifest,
    parse_output_integrity_manifest,
    parse_private_identity_sidecar,
    parse_public_validation_record,
    public_record_matches_private_sidecar,
    snapshot_governed_inputs,
    verify_input_integrity_manifest,
    verify_output_integrity_manifest,
    write_private_identity_sidecar,
)
from bayesfilter.runtime import atomic_write_json
from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (
    DEFAULT_CONFIG_PATH,
    HISTORICAL_V1_CONFIG_PATH,
    DeterministicLGSSMPhase7Config,
    DeterministicLGSSMPhase7Error,
    generate_phase3_candidate_identity_evidence,
    validate_phase7_inputs,
)
from tests.test_hmc_identity import (
    SHA_A,
    _Replay,
    _execution,
    _provenance,
    _transition,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_REPLAY_PATH = ROOT / (
    "docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/"
    "private_diagnostics/kernel_tuning_replay.json"
)
def _rehash(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    result.pop("artifact_hash", None)
    result["artifact_hash"] = canonical_artifact_payload_hash(result)
    return result


def _legacy_validator_result() -> LegacyValidatorResultV1:
    return LegacyValidatorResultV1(
        passed=False,
        exception_type="DeterministicLGSSMPhase7Error",
        message="public final kernel hash mismatch",
        veto_code=PHASE3_LEGACY_VETO_CODE,
        remains_binding=True,
    )


def _write_legacy_replay(tmp_path: Path) -> tuple[Path, dict, dict]:
    tuning_payload = {
        "schema": "test.selection.v1",
        "final_status": "passed",
        "handoff_screen_policy": "current",
        "diagnostics": {"acceptance": 0.7},
    }
    payload = {
        "schema": "test.legacy.private_replay.v1",
        "tuning_payload": tuning_payload,
    }
    from bayesfilter.runtime import stable_config_hash

    payload["artifact_hash"] = f"sha256:{stable_config_hash(payload)}"
    path = tmp_path / "legacy_replay.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    import hashlib

    reference = {
        "artifact_hash": payload["artifact_hash"],
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }
    return path, payload, reference


def _sidecar(tmp_path: Path) -> tuple[dict, Path, dict]:
    legacy_path, legacy_payload, legacy_reference = _write_legacy_replay(tmp_path)
    replay = _Replay()
    replay.contract.update(
        {
            "base_adapter_signature": "1" * 64,
            "phase4_hmc_adapter_signature": "2" * 64,
            "final_hmc_adapter_signature": "3" * 64,
            "geometry_mass_artifact_signature": "4" * 64,
            "adapted_mass_artifact_signature": "5" * 64,
        }
    )
    payload = build_private_identity_sidecar(
        transition=_transition(),
        serious_execution=_execution(smoke=False),
        smoke_execution=_execution(smoke=True),
        selection_provenance=_provenance(),
        complete_tuning_payload=legacy_payload["tuning_payload"],
        legacy_private_replay_payload=legacy_payload,
        legacy_private_replay_path=legacy_path,
        legacy_private_replay_reference=legacy_reference,
        replay=replay,
        legacy_validator_result=_legacy_validator_result(),
    )
    path = tmp_path / "private_sidecar.json"
    write_private_identity_sidecar(path, payload)
    return payload, path, legacy_reference


def _input_manifest(tmp_path: Path) -> tuple[dict, Path]:
    governed = tmp_path / "governed.json"
    governed.write_text('{"governed":true}\n', encoding="utf-8")
    before = snapshot_governed_inputs((governed,))
    after = snapshot_governed_inputs((governed,))
    payload = build_input_integrity_manifest(
        pre_snapshot=before,
        post_snapshot=after,
    )
    path = tmp_path / "input_manifest.json"
    atomic_write_json(path, payload)
    return payload, path


def _candidate_checks() -> dict[str, bool]:
    return {name: True for name in PHASE3_CANDIDATE_CHECK_KEYS}


def _public_bundle(tmp_path: Path) -> tuple[dict, Path, dict, Path, dict, Path]:
    sidecar, sidecar_path, legacy_reference = _sidecar(tmp_path)
    input_manifest, input_path = _input_manifest(tmp_path)
    public = build_public_validation_record(
        sidecar_payload=sidecar,
        sidecar_path=sidecar_path,
        input_integrity_manifest=input_manifest,
        legacy_private_replay_reference=legacy_reference,
        candidate_checks=_candidate_checks(),
    )
    public_path = tmp_path / "public_record.json"
    atomic_write_json(public_path, public)
    return public, public_path, sidecar, sidecar_path, input_manifest, input_path


def test_private_sidecar_round_trip_separates_all_identity_domains(
    tmp_path: Path,
) -> None:
    sidecar, path, _reference = _sidecar(tmp_path)
    restored = json.loads(path.read_text(encoding="utf-8"))

    assert parse_private_identity_sidecar(restored) == restored
    assert restored["schema"] == HMC_IDENTITY_PRIVATE_SIDECAR_SCHEMA_V1
    assert restored["transition_identity_hash"] == _transition().identity_hash
    assert restored["serious_execution_contract_hash"] == _execution().identity_hash
    assert restored["smoke_execution_contract_hash"] == _execution(smoke=True).identity_hash
    assert restored["selection_provenance_hash"] == _provenance().identity_hash
    assert restored["complete_tuning_payload_hash"] == canonical_artifact_payload_hash(
        {
            "schema": "test.selection.v1",
            "final_status": "passed",
            "handoff_screen_policy": "current",
            "diagnostics": {"acceptance": 0.7},
        }
    )
    assert "file_sha256" not in sidecar
    assert "byte_count" not in sidecar


def test_private_sidecar_rejects_semantic_tamper_even_with_rehashed_envelope(
    tmp_path: Path,
) -> None:
    sidecar, _path, _reference = _sidecar(tmp_path)
    tampered = copy.deepcopy(sidecar)
    tampered["transition_identity_hash"] = "sha256:" + "f" * 64
    tampered = _rehash(tampered)

    with pytest.raises(ValueError, match="transition_identity_hash mismatch"):
        parse_private_identity_sidecar(tampered)

    extra = copy.deepcopy(sidecar)
    extra["legacy_private_replay_integrity"]["unexpected"] = True
    extra = _rehash(extra)
    with pytest.raises(ValueError, match="fields mismatch"):
        parse_private_identity_sidecar(extra)

    provenance_mismatch = copy.deepcopy(sidecar)
    provenance_mismatch["complete_tuning_payload_hash"] = "sha256:" + "e" * 64
    provenance_mismatch = _rehash(provenance_mismatch)
    with pytest.raises(ValueError, match="complete_tuning_payload_hash mismatch"):
        parse_private_identity_sidecar(provenance_mismatch)


def test_public_record_has_exact_closed_schema_and_no_private_mechanics(
    tmp_path: Path,
) -> None:
    public, _path, sidecar, sidecar_path, input_manifest, _input_path = (
        _public_bundle(tmp_path)
    )

    assert parse_public_validation_record(public) == public
    assert public["schema"] == HMC_IDENTITY_PUBLIC_VALIDATION_SCHEMA_V1
    assert public["status"] == PHASE3_STATUS
    assert public["decision"] == PHASE3_DECISION
    assert tuple(public["candidate_checks"]) == PHASE3_CANDIDATE_CHECK_KEYS
    assert public["legacy_gate"] == {
        "passed": False,
        "veto_code": PHASE3_LEGACY_VETO_CODE,
        "remains_binding": True,
    }
    assert tuple(public["nonclaims"]) == PHASE3_PUBLIC_NONCLAIMS
    assert all(
        value is False
        for name, value in public["private_sidecar_reference"].items()
        if name.endswith("_publicized")
    )
    assert public_record_matches_private_sidecar(
        public_record=public,
        sidecar_payload=sidecar,
        sidecar_path=sidecar_path,
        input_integrity_manifest=input_manifest,
    )
    serialized = json.dumps(public, sort_keys=True)
    for forbidden in (
        "step_size",
        "num_leapfrog_steps",
        "base_adapter_signature",
        "tensorflow_version",
        "test_lgssm_scope",
        "reconstruction-only-final-signature",
    ):
        assert forbidden not in serialized
    assert '"stage_lineage":' not in serialized


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(status="passed"), "status or decision"),
        (
            lambda value: value["legacy_gate"].update(remains_binding=False),
            "legacy gate",
        ),
        (
            lambda value: value["candidate_checks"].update(unknown=True),
            "fields mismatch",
        ),
        (
            lambda value: value["private_sidecar_reference"].update(
                seeds_publicized=True
            ),
            "requires seeds_publicized=false",
        ),
        (
            lambda value: value["legacy_private_replay_reference"].update(
                path="/private/replay.json"
            ),
            "fields mismatch",
        ),
        (
            lambda value: value.update(
                nonclaims=list(reversed(value["nonclaims"]))
            ),
            "closed ordered contract",
        ),
    ],
)
def test_public_parser_rejects_altered_fixed_values_and_nested_fields(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    public, *_rest = _public_bundle(tmp_path)
    tampered = copy.deepcopy(public)
    mutation(tampered)
    tampered = _rehash(tampered)

    with pytest.raises(ValueError, match=match):
        parse_public_validation_record(tampered)


def test_public_redaction_scan_rejects_recursive_secrets() -> None:
    with pytest.raises(ValueError, match="forbidden keys"):
        assert_public_validation_redacted({"nested": {"step_size": 0.1}})
    with pytest.raises(ValueError, match="absolute private path"):
        assert_public_validation_redacted({"nested": "/private/path"})
    with pytest.raises(ValueError, match="forbidden private value"):
        assert_public_validation_redacted(
            {"safe": "prefix-secret-suffix"},
            forbidden_values=("secret",),
        )


def test_input_manifest_fails_closed_when_governed_bytes_change(tmp_path: Path) -> None:
    governed = tmp_path / "governed.json"
    governed.write_text('{"value":1}\n', encoding="utf-8")
    before = snapshot_governed_inputs((governed,))
    governed.write_text('{"value":2}\n', encoding="utf-8")
    after = snapshot_governed_inputs((governed,))

    with pytest.raises(ValueError, match="changed"):
        build_input_integrity_manifest(pre_snapshot=before, post_snapshot=after)


def test_terminal_manifest_is_acyclic_and_detects_exact_byte_tamper(
    tmp_path: Path,
) -> None:
    public, public_path, _sidecar, sidecar_path, _input, input_path = (
        _public_bundle(tmp_path)
    )
    manifest = build_output_integrity_manifest(
        sidecar_path=sidecar_path,
        input_manifest_path=input_path,
        public_record_path=public_path,
    )
    manifest_path = tmp_path / "output_manifest.json"
    atomic_write_json(manifest_path, manifest)
    restored = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert parse_output_integrity_manifest(restored) == restored
    assert verify_output_integrity_manifest(
        restored,
        sidecar_path=sidecar_path,
        input_manifest_path=input_path,
        public_record_path=public_path,
    ) == restored
    assert restored["schema"] == HMC_IDENTITY_OUTPUT_INTEGRITY_MANIFEST_SCHEMA_V1
    assert tuple(item["role"] for item in restored["outputs"]) == (
        "private_sidecar",
        "input_integrity_manifest",
        "public_validation_record",
    )
    assert all("path" not in item for item in restored["outputs"])
    assert "output_integrity_manifest" not in json.dumps(restored["outputs"])

    public_path.write_text(
        public_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    assert parse_public_validation_record(public) == public
    with pytest.raises(ValueError, match="current output bytes"):
        verify_output_integrity_manifest(
            restored,
            sidecar_path=sidecar_path,
            input_manifest_path=input_path,
            public_record_path=public_path,
        )


def test_terminal_manifest_builder_rejects_valid_but_cross_linked_wrong_sidecar(
    tmp_path: Path,
) -> None:
    _public, public_path, sidecar, sidecar_path, _input, input_path = (
        _public_bundle(tmp_path)
    )
    changed = copy.deepcopy(sidecar)
    changed["legacy_validator_result"]["remains_binding"] = True
    changed["legacy_validator_result"]["message"] = "public final kernel hash mismatch"
    changed["complete_tuning_payload_hash"] = changed[
        "selection_provenance"
    ]["source_selection_payload_hash"]
    changed["nonclaims"] = list(changed["nonclaims"])
    changed = _rehash(changed)
    # Change one valid private cross-link without changing the public record.
    changed["reconstruction_links"]["base_adapter_signature"] = "9" * 64
    changed = _rehash(changed)
    write_private_identity_sidecar(sidecar_path, changed)

    with pytest.raises(ValueError, match="public/private cross-links"):
        build_output_integrity_manifest(
            sidecar_path=sidecar_path,
            input_manifest_path=input_path,
            public_record_path=public_path,
        )


def test_public_private_reference_mismatch_is_not_suppressed(tmp_path: Path) -> None:
    public, _path, sidecar, sidecar_path, input_manifest, _input_path = (
        _public_bundle(tmp_path)
    )
    sidecar_path.write_text(
        sidecar_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    assert public_record_matches_private_sidecar(
        public_record=public,
        sidecar_payload=sidecar,
        sidecar_path=sidecar_path,
        input_integrity_manifest=input_manifest,
    ) is False


def test_public_legacy_replay_reference_mismatch_is_not_suppressed(
    tmp_path: Path,
) -> None:
    public, _path, sidecar, sidecar_path, input_manifest, _input_path = (
        _public_bundle(tmp_path)
    )
    mismatched = copy.deepcopy(public)
    mismatched["legacy_private_replay_reference"]["file_sha256"] = "f" * 64
    mismatched = _rehash(mismatched)

    assert parse_public_validation_record(mismatched) == mismatched
    assert public_record_matches_private_sidecar(
        public_record=mismatched,
        sidecar_payload=sidecar,
        sidecar_path=sidecar_path,
        input_integrity_manifest=input_manifest,
    ) is False


def test_sidecar_builder_rejects_provenance_for_a_different_tuning_payload(
    tmp_path: Path,
) -> None:
    legacy_path, legacy_payload, legacy_reference = _write_legacy_replay(tmp_path)
    replay = _Replay()
    replay.contract.update(
        {
            "base_adapter_signature": "1" * 64,
            "phase4_hmc_adapter_signature": "2" * 64,
            "final_hmc_adapter_signature": "3" * 64,
            "geometry_mass_artifact_signature": "4" * 64,
            "adapted_mass_artifact_signature": "5" * 64,
        }
    )
    changed_payload = copy.deepcopy(legacy_payload["tuning_payload"])
    changed_payload["handoff_screen_policy"] = "changed"

    with pytest.raises(ValueError, match="complete tuning payload"):
        build_private_identity_sidecar(
            transition=_transition(),
            serious_execution=_execution(smoke=False),
            smoke_execution=_execution(smoke=True),
            selection_provenance=_provenance(),
            complete_tuning_payload=changed_payload,
            legacy_private_replay_payload=legacy_payload,
            legacy_private_replay_path=legacy_path,
            legacy_private_replay_reference=legacy_reference,
            replay=replay,
            legacy_validator_result=_legacy_validator_result(),
        )


def test_real_selection_provenance_binds_named_selected_attempt_lineage() -> None:
    private_replay = json.loads(REAL_REPLAY_PATH.read_text(encoding="utf-8"))
    provenance = build_selection_provenance_from_tuning_payload(
        tuning_payload=private_replay["tuning_payload"],
        tuning_config_hash=private_replay["config_hash"],
    )

    assert tuple(stage.stage_id for stage in provenance.stage_lineage) == (
        "bootstrap",
        "geometry",
        "windowed_mass",
        "fixed_mass_step",
        "frozen_step_trajectory",
        "fresh_verification",
        "tune_verify_repair_loop",
    )
    assert tuple(stage.selected_index for stage in provenance.stage_lineage) == (
        None,
        None,
        2,
        2,
        2,
        2,
        2,
    )
    assert tuple(stage.canonical_payload_hash for stage in provenance.stage_lineage) == (
        "sha256:cf53d964cf896c2e9532758490e17d5ad80ce1883c10bf77b286dd310e9df97c",
        "sha256:1b56467521d5b0a2c600e44c2553c9b8e37d01d63f2ca344674372b83da4f8a1",
        "sha256:975c6dd505929ee5effc2430a4ac3bb8959fd835db4fcbe1bb92bcf01e85e9e1",
        "sha256:d3044b746745d6aa626988837fe42193d2f9bf38496f416cc361c1ed761a6e6d",
        "sha256:40d6c2af382ac823c2b247bedfcecfe5d5c1fb3b1bde2bc90d9ace434b0ae94c",
        "sha256:2823e20048c0969b79931604462ba142a34aed06fd8cfab3baf03eab89c0168f",
        "sha256:bbc3940f77ab51cad2fa3c4807733d97b98fe9af051803328a5a5cda0ead85e4",
    )


def test_identity_ownership_oracle_keeps_domains_independent(tmp_path: Path) -> None:
    transition = _transition()
    serious = _execution()
    provenance = _provenance()
    changed_transition = replace(transition, num_leapfrog_steps=6)
    changed_execution = replace(serious, root_seed=(20260711, 702))
    changed_provenance = replace(provenance, tuning_config_hash="sha256:" + "e" * 64)

    assert changed_transition.identity_hash != transition.identity_hash
    assert changed_execution.identity_hash != serious.identity_hash
    assert changed_provenance.identity_hash != provenance.identity_hash
    assert serious.transition_identity_hash == transition.identity_hash
    rebound_execution = replace(
        serious,
        transition_identity_hash=changed_transition.identity_hash,
    )
    assert rebound_execution.identity_hash != serious.identity_hash
    assert provenance.identity_hash == _provenance().identity_hash

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"value":1}\n', encoding="utf-8")
    second.write_text('{ "value" : 1 }\n', encoding="utf-8")
    assert json.loads(first.read_text()) == json.loads(second.read_text())
    assert canonical_artifact_payload_hash(json.loads(first.read_text())) == (
        canonical_artifact_payload_hash(json.loads(second.read_text()))
    )
    assert snapshot_governed_inputs((first,))[0]["file_sha256"] != (
        snapshot_governed_inputs((second,))[0]["file_sha256"]
    )


def test_real_phase3_opt_in_path_persists_evidence_then_reraises_legacy_veto(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    config = DeterministicLGSSMPhase7Config.load(HISTORICAL_V1_CONFIG_PATH)
    paths = {
        "private_sidecar_path": tmp_path / "private_sidecar.json",
        "input_manifest_path": tmp_path / "input_manifest.json",
        "public_record_path": tmp_path / "public_record.json",
        "output_manifest_path": tmp_path / "output_manifest.json",
    }

    with pytest.raises(
        DeterministicLGSSMPhase7Error,
        match="public final kernel hash mismatch",
    ):
        generate_phase3_candidate_identity_evidence(config, **paths)

    assert all(path.is_file() for path in paths.values())
    sidecar = json.loads(paths["private_sidecar_path"].read_text(encoding="utf-8"))
    input_manifest = json.loads(
        paths["input_manifest_path"].read_text(encoding="utf-8")
    )
    public = json.loads(paths["public_record_path"].read_text(encoding="utf-8"))
    output = json.loads(paths["output_manifest_path"].read_text(encoding="utf-8"))

    parse_private_identity_sidecar(sidecar)
    parse_input_integrity_manifest(input_manifest)
    verify_input_integrity_manifest(input_manifest)
    parse_public_validation_record(public)
    verify_output_integrity_manifest(
        output,
        sidecar_path=paths["private_sidecar_path"],
        input_manifest_path=paths["input_manifest_path"],
        public_record_path=paths["public_record_path"],
    )
    assert all(public["candidate_checks"].values())
    assert public["legacy_gate"]["remains_binding"] is True
    assert sidecar["legacy_validator_result"]["message"] == (
        "public final kernel hash mismatch"
    )
    assert input_manifest["schema"] == HMC_IDENTITY_INPUT_INTEGRITY_MANIFEST_SCHEMA_V1


def test_legacy_phase7_validator_retains_exact_fail_closed_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    config = DeterministicLGSSMPhase7Config.load(HISTORICAL_V1_CONFIG_PATH)

    with pytest.raises(
        DeterministicLGSSMPhase7Error,
        match="^public final kernel hash mismatch$",
    ):
        validate_phase7_inputs(config)
