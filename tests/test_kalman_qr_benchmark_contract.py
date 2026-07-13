from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path

import pytest

from scripts import kalman_qr_benchmark_contract as contract


def _fingerprints() -> dict[str, str]:
    return {field: field.replace("_fingerprint", "-hash") for field in contract.FINGERPRINT_FIELDS}


def _event(attempt_id: str, stage: str = "fixture", **updates) -> dict[str, object]:
    payload: dict[str, object] = {
        "attempt_id": attempt_id,
        "case_id": "case-a",
        "method_id": contract.METHOD_IDS[0],
        "stage": stage,
        "resume_key": "resume-hash",
        **_fingerprints(),
    }
    payload.update(updates)
    return payload


def _expected(attempt_id: str, **updates) -> dict[str, object]:
    event = _event(attempt_id, **updates)
    event.pop("stage")
    return event


def _config(**updates) -> dict[str, object]:
    payload: dict[str, object] = {
        "method_id": contract.METHOD_IDS[0],
        "method_contract_version": contract.METHOD_CONTRACT_VERSION,
        "dimension": 2,
        "parameter_count": 2,
        "timesteps": 2,
        "batch_size": 1,
        "dtype": "float32",
        "device": "cpu",
        "jit_compile": False,
        "cpu_threads": 1,
        "repeats": 1,
        "subprocess_timeout_seconds": 30.0,
        "xla_flags": "UNSET",
        "tf32_enabled": True,
        "jitter": 1.0e-9,
        "jitter_updates_filtered_covariance": True,
        "fixture_contract_version": "phase1-v1",
        "timing_boundary_version": contract.TIMING_BOUNDARY_VERSION,
        "method_options": {},
    }
    payload.update(updates)
    return payload


def _fixture(**updates) -> dict[str, object]:
    payload: dict[str, object] = {
        "fixture_contract_version": "phase1-v1",
        "randomness": "deterministic",
        "seed": None,
        "dimension": 2,
        "parameter_count": 2,
        "timesteps": 2,
        "batch_size": 1,
        "dtype": "float32",
        "parameter_batch_version": "historical-batch-offset-v1",
        "observation_generation_version": "historical-sine-v1",
        "external_input_hashes": {},
    }
    payload.update(updates)
    return payload


def _runtime(**updates) -> dict[str, object]:
    payload: dict[str, object] = {
        "interpreter": "/env/bin/python",
        "python_implementation": "CPython",
        "python_version": "3.13.13",
        "platform": "test-platform",
        "distributions": {
            "tensorflow": {"distribution": "tensorflow", "version": "2.20.0"},
            "tensorflow_probability": {"distribution": "tfp-nightly", "version": "0.25.0"},
            "numpy": {"distribution": "numpy", "version": "2.1.3"},
        },
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, {"nested": [math.nan]}])
def test_strict_json_rejects_nonfinite_values(value) -> None:
    with pytest.raises(contract.ContractError):
        contract.strict_json_dumps(value)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_rejects_raw_nonstandard_constants(token: str) -> None:
    with pytest.raises(contract.ContractError):
        contract.strict_json_loads('{"value":' + token + "}")


def test_strict_json_rejects_duplicate_object_keys() -> None:
    with pytest.raises(contract.ContractError):
        contract.strict_json_loads('{"value":1,"value":2}')


def test_atomic_write_preserves_previous_artifact_when_encoding_fails(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    contract.atomic_write_json(path, {"state": "valid"})
    original = path.read_bytes()
    with pytest.raises(contract.ContractError):
        contract.atomic_write_json(path, {"state": math.nan})
    assert path.read_bytes() == original


def test_canonical_fingerprints_are_order_stable_and_field_sensitive() -> None:
    left = contract.canonical_sha256({"a": 1, "b": 2})
    right = contract.canonical_sha256({"b": 2, "a": 1})
    assert left == right
    assert left != contract.canonical_sha256({"a": 1, "b": 3})
    for field in contract.CONFIG_FIELDS:
        changed = _config()
        if field == "method_options":
            changed[field] = {"unknown": True}
            with pytest.raises(contract.ContractError):
                contract.config_manifest(changed)
            continue
        if field == "method_contract_version":
            changed[field] = "stale-method-contract"
            with pytest.raises(contract.ContractError, match="stale method_contract_version"):
                contract.config_manifest(changed)
            continue
        if field == "timing_boundary_version":
            changed[field] = "stale-timing-boundary"
            with pytest.raises(contract.ContractError, match="stale timing_boundary_version"):
                contract.config_manifest(changed)
            continue
        changed[field] = (
            contract.METHOD_IDS[1] if field == "method_id" else f"changed-{field}"
        )
        assert contract.config_manifest(changed)["config_fingerprint"] != contract.config_manifest(
            _config()
        )["config_fingerprint"]


def test_closed_manifests_reject_missing_or_extra_fields() -> None:
    for builder, payload in (
        (contract.config_manifest, _config()),
        (contract.fixture_manifest, _fixture()),
        (contract.runtime_manifest, _runtime()),
    ):
        missing = dict(payload)
        missing.pop(next(iter(missing)))
        with pytest.raises(contract.ContractError):
            builder(missing)
        extra = dict(payload, unexpected=True)
        with pytest.raises(contract.ContractError):
            builder(extra)


def test_fixture_and_runtime_fingerprints_are_field_sensitive() -> None:
    base_fixture = contract.fixture_manifest(_fixture())["fixture_fingerprint"]
    for field in contract.FIXTURE_FIELDS:
        if field in {"randomness", "seed", "external_input_hashes"}:
            continue
        changed = _fixture(**{field: f"changed-{field}"})
        assert contract.fixture_manifest(changed)["fixture_fingerprint"] != base_fixture
    base_runtime = contract.runtime_manifest(_runtime())["runtime_fingerprint"]
    changed_runtime = _runtime(platform="changed-platform")
    assert contract.runtime_manifest(changed_runtime)["runtime_fingerprint"] != base_runtime


def test_source_manifest_is_sorted_sensitive_and_missing_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(contract, "SOURCE_PATHS", ("b.py", "a.py"))
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "b.py").write_text("b", encoding="utf-8")
    first = contract.source_manifest(tmp_path, include_supervisor=False)
    assert [row["path"] for row in first["files"]] == ["a.py", "b.py"]
    (tmp_path / "a.py").write_text("changed", encoding="utf-8")
    second = contract.source_manifest(tmp_path, include_supervisor=False)
    assert second["source_fingerprint"] != first["source_fingerprint"]
    (tmp_path / "b.py").unlink()
    with pytest.raises(contract.ContractError):
        contract.source_manifest(tmp_path, include_supervisor=False)


def test_schedule_is_predeclared_unique_and_check_complete() -> None:
    base = {field: f"hash-{field}" for field in contract.FINGERPRINT_FIELDS[:-1]}
    identity = {"case_id": "case-a", "method_id": contract.METHOD_IDS[0], **base}
    manifest = contract.build_schedule_manifest(
        [identity],
        contract.HARNESS_ONLY_CHECKS,
        harness_contract_test_only=True,
    )
    assert manifest["expected_identities"] == [
        identity
    ]
    with pytest.raises(contract.ContractError):
        contract.build_schedule_manifest(
            [identity, identity],
            contract.HARNESS_ONLY_CHECKS,
            harness_contract_test_only=True,
        )
    with pytest.raises(contract.ContractError):
        contract.build_schedule_manifest(
            [identity],
            contract.HARNESS_ONLY_CHECKS,
            harness_contract_test_only=False,
        )
    with pytest.raises(contract.ContractError):
        contract.build_schedule_manifest(
            [], contract.HARNESS_ONLY_CHECKS, harness_contract_test_only=True
        )
    changed = dict(identity, config_fingerprint="changed")
    assert contract.build_schedule_manifest(
        [changed], contract.HARNESS_ONLY_CHECKS, harness_contract_test_only=True
    )["schedule_fingerprint"] != manifest["schedule_fingerprint"]


def test_phase5_method_and_measurement_contract_is_exact() -> None:
    assert contract.SCHEMA == "bayesfilter.kalman_qr_batched_xla_repair.v4"
    assert contract.METHOD_CONTRACT_VERSION == "measurement-boundaries-phase5-v1"
    assert contract.TIMING_BOUNDARY_VERSION == (
        "separated-trace-execution-materialization-phase5-v1"
    )
    assert contract.PRIMARY_METHOD_IDS == (
        "batch_native_analytical_qr_score",
        "batch_native_autodiff_qr_score",
    )
    assert contract.REFERENCE_METHOD_IDS == (
        "scalar_analytical_row_loop",
        "autodiff_row_loop_qr_score",
    )
    assert set(contract.PRIMARY_METHOD_IDS).isdisjoint(contract.REFERENCE_METHOD_IDS)
    assert contract.METHOD_IDS == contract.PRIMARY_METHOD_IDS + contract.REFERENCE_METHOD_IDS
    assert "batched_static_autodiff_probe" not in contract.METHOD_IDS


@pytest.mark.parametrize(
    ("methods", "mode", "checks"),
    [
        (contract.PRIMARY_METHOD_IDS, "primary_pair", contract.PRIMARY_PAIR_CHECKS),
        ((contract.PRIMARY_METHOD_IDS[0],), "method_local_only", contract.METHOD_LOCAL_CHECKS),
        ((contract.PRIMARY_METHOD_IDS[1],), "method_local_only", contract.METHOD_LOCAL_CHECKS),
        (contract.REFERENCE_METHOD_IDS, "method_local_only", contract.METHOD_LOCAL_CHECKS),
        (
            (contract.PRIMARY_METHOD_IDS[0], contract.REFERENCE_METHOD_IDS[0]),
            "method_local_only",
            contract.METHOD_LOCAL_CHECKS,
        ),
        (
            (*contract.PRIMARY_METHOD_IDS, contract.REFERENCE_METHOD_IDS[0]),
            "primary_pair",
            contract.PRIMARY_PAIR_CHECKS,
        ),
    ],
)
def test_schedule_modes_and_canonical_checks(methods, mode, checks) -> None:
    base = {field: f"hash-{field}" for field in contract.FINGERPRINT_FIELDS[:-1]}
    identities = [
        {"case_id": "case-a", "method_id": method, **base}
        for method in methods
    ]
    manifest = contract.build_schedule_manifest(
        identities,
        checks,
        harness_contract_test_only=False,
    )
    assert manifest["comparison_mode"] == mode
    assert manifest["mandatory_aggregate_checks"] == list(checks)
    assert manifest["primary_pair_complete"] is (mode == "primary_pair")
    assert manifest["comparator_parity_applicable"] is (mode == "primary_pair")
    assert manifest["comparator_parity_reason"] == (
        None if mode == "primary_pair" else "primary_method_pair_incomplete"
    )
    with pytest.raises(contract.ContractError, match="aggregate checks must equal"):
        contract.build_schedule_manifest(
            identities,
            tuple(reversed(checks)),
            harness_contract_test_only=False,
        )


def test_harness_only_schedule_rejects_complete_primary_pair() -> None:
    base = {field: f"hash-{field}" for field in contract.FINGERPRINT_FIELDS[:-1]}
    identities = [
        {"case_id": "case-a", "method_id": method, **base}
        for method in contract.PRIMARY_METHOD_IDS
    ]
    with pytest.raises(contract.ContractError, match="cannot claim a primary pair"):
        contract.build_schedule_manifest(
            identities,
            contract.HARNESS_ONLY_CHECKS,
            harness_contract_test_only=True,
        )


def test_attempt_journal_recovery_is_identity_bound_and_partial_safe(tmp_path: Path) -> None:
    attempt_id, path = contract.new_attempt(tmp_path, "case-a", contract.METHOD_IDS[0])
    contract.append_progress_event(path, _event(attempt_id, "fixture"))
    contract.append_progress_event(path, _event(attempt_id, "trace"))
    with path.open("ab") as handle:
        handle.write(b'{"partial":')
        handle.flush()
        os.fsync(handle.fileno())
    assert contract.recover_last_stage(path, _expected(attempt_id)) == "trace"
    assert contract.recover_last_stage(path, _expected("stale-attempt")) is None


def test_retry_uses_fresh_journal_and_cannot_inherit_stale_or_partial_line(tmp_path: Path) -> None:
    old_attempt, old_path = contract.new_attempt(tmp_path, "case-a", contract.METHOD_IDS[0])
    contract.append_progress_event(old_path, _event(old_attempt, "warm_execution"))
    with old_path.open("ab") as handle:
        handle.write(b"partial")
    new_attempt, new_path = contract.new_attempt(tmp_path, "case-a", contract.METHOD_IDS[0])
    assert new_path != old_path
    assert contract.recover_last_stage(new_path, _expected(new_attempt)) is None
    contract.append_progress_event(new_path, _event(new_attempt, "fixture"))
    assert contract.recover_last_stage(new_path, _expected(new_attempt)) == "fixture"


def test_new_attempt_paths_are_exclusive(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(contract.secrets, "token_hex", lambda _: "fixed")
    contract.new_attempt(tmp_path, "case-a", contract.METHOD_IDS[0])
    with pytest.raises(FileExistsError):
        contract.new_attempt(tmp_path, "case-a", contract.METHOD_IDS[0])


@pytest.mark.parametrize(
    ("timed_out", "returncode", "expected_state"),
    [(True, None, "timed_out"), (False, -9, "crashed"), (False, 1, "failed")],
)
def test_process_record_recovers_exact_stage(tmp_path: Path, timed_out, returncode, expected_state) -> None:
    attempt_id, path = contract.new_attempt(tmp_path, "case-a", contract.METHOD_IDS[0])
    identity = _expected(attempt_id)
    contract.append_progress_event(path, _event(attempt_id, "first_executable_call"))
    record = contract.synthesize_process_record(
        identity=identity,
        progress_path=path,
        timed_out=timed_out,
        returncode=returncode,
        error_tail="failure",
    )
    assert record["state"] == expected_state
    assert record["last_entered_stage"] == "first_executable_call"


def _measurement(repeats: int = 1) -> dict[str, object]:
    events = []
    for index, stage in enumerate(contract.MEASUREMENT_EVENT_STAGES):
        events.append(
            {
                "sequence_index": index,
                "stage": stage,
                "entered_ns": index * 20,
                "finished_ns": index * 20 + 10,
            }
        )
    return {
        "timing_boundary_version": contract.TIMING_BOUNDARY_VERSION,
        "requested_repeats": repeats,
        "stage_events": events,
        "durations": {
            "fixture_seconds": 1.0e-8,
            "trace_seconds": 1.0e-8,
            "first_executable_call_seconds": 1.0e-8,
            "warm_execution_seconds": [1.0e-9] * repeats,
            "materialization_seconds": 1.0e-8,
            "parity_seconds": 1.0e-8,
            "payload_encoding_seconds": 1.0e-8,
            "artifact_write_seconds": 1.0e-8,
        },
        "synchronization": {
            "method": "scalar_sentinel",
            "sentinel_definition": "reduce_sum(value)+reduce_sum(score)",
            "scalar_materialization_count": 1 + repeats,
            "full_output_materialization_count": 1,
            "parity_residual_materialization_count": 1,
        },
        "invocation_counts": {
            "before_first_executable_call": 0,
            "after_first_executable_call": 1,
            "after_warm_execution": 1 + repeats,
            "after_reference_call": 2 + repeats,
        },
        "graphdef": {"node_count": 2, "serialized_bytes": 100},
        "direct_output_parity": {
            "passed": True,
            "dtype": "float32",
            "value_rtol": 2.0e-4,
            "value_atol": 2.0e-4,
            "score_rtol": 2.0e-4,
            "score_atol": 2.0e-4,
            "value_reference_max_abs": 1.0,
            "score_reference_max_abs": 1.0,
            "value_max_abs_residual": 0.0,
            "score_max_abs_residual": 0.0,
        },
        "payload_sidecar": {
            "path": "/tmp/method.payload.json",
            "sha256": "0" * 64,
            "write_count": 1,
        },
        "envelope_write_measured": False,
    }


def _passed_record() -> dict[str, object]:
    fingerprints = _fingerprints()
    return {
        "schema": contract.SCHEMA,
        "method_contract_version": contract.METHOD_CONTRACT_VERSION,
        "case_id": "case-a",
        "method_id": contract.METHOD_IDS[0],
        **fingerprints,
        "resume_key": "resume-hash",
        "state": "passed",
        "attempt_id": "attempt-a",
        "last_entered_stage": "envelope_write",
        "terminal_stage": "envelope_write",
        "failure_stage": None,
        "invoked_method_ids": [contract.METHOD_IDS[0]],
        "measurement": _measurement(),
        "output_metadata": {
            "all_finite": True,
            "value_shape": [1],
            "score_shape": [1, 2],
            "value_dtype": "float32",
            "score_dtype": "float32",
        },
    }


def test_resume_requires_exact_passed_finite_record() -> None:
    expected = _fingerprints()
    reusable, reason = contract.method_record_reuse_decision(
        _passed_record(),
        expected_case_id="case-a",
        expected_method_id=contract.METHOD_IDS[0],
        expected_fingerprints=expected,
        expected_resume_key="resume-hash",
    )
    assert (reusable, reason) == (True, "reusable_exact_match")
    mutations = {
        "schema": "v1",
        "state": "failed",
        "resume_key": "stale",
        "invoked_method_ids": [],
        "output_metadata": {"all_finite": False},
        "attempt_id": None,
        "last_entered_stage": "parity",
        "terminal_stage": "parity",
        "failure_stage": "trace",
    }
    for field, value in mutations.items():
        record = _passed_record()
        record[field] = value
        assert contract.method_record_reuse_decision(
            record,
            expected_case_id="case-a",
            expected_method_id=contract.METHOD_IDS[0],
            expected_fingerprints=expected,
            expected_resume_key="resume-hash",
        )[0] is False
    for fingerprint_field in contract.FINGERPRINT_FIELDS:
        record = _passed_record()
        record[fingerprint_field] = "mismatch"
        assert contract.method_record_reuse_decision(
            record,
            expected_case_id="case-a",
            expected_method_id=contract.METHOD_IDS[0],
            expected_fingerprints=expected,
            expected_resume_key="resume-hash",
        )[0] is False
    for output_metadata in (
        None,
        {"all_finite": False},
        {
            "all_finite": True,
            "value_shape": None,
            "score_shape": [1, 2],
            "value_dtype": "float32",
            "score_dtype": "float32",
        },
    ):
        record = _passed_record()
        record["output_metadata"] = output_metadata
        assert contract.method_record_reuse_decision(
            record,
            expected_case_id="case-a",
            expected_method_id=contract.METHOD_IDS[0],
            expected_fingerprints=expected,
            expected_resume_key="resume-hash",
        )[0] is False


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("schema", "bayesfilter.kalman_qr_batched_xla_repair.v2", "schema_mismatch"),
        ("method_contract_version", "old", "method_contract_version_mismatch"),
        ("method_id", "batched_static_autodiff_probe", "method_id_mismatch"),
        ("source_fingerprint", "stale", "source_fingerprint_mismatch"),
        ("config_fingerprint", "stale", "config_fingerprint_mismatch"),
        ("schedule_fingerprint", "stale", "schedule_fingerprint_mismatch"),
    ],
)
def test_stale_resume_reason_is_named_and_read_only(
    tmp_path: Path, field: str, value: str, reason: str
) -> None:
    record = _passed_record()
    record[field] = value
    path = tmp_path / "method.json"
    contract.atomic_write_json(path, record)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = contract.read_strict_json(path)
    reusable, actual_reason = contract.method_record_reuse_decision(
        loaded,
        expected_case_id="case-a",
        expected_method_id=contract.METHOD_IDS[0],
        expected_fingerprints=_fingerprints(),
        expected_resume_key="resume-hash",
    )
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert reusable is False
    assert actual_reason == reason
    assert before == after


def test_top_level_status_rejects_missing_extra_duplicate_and_nonterminal() -> None:
    expected = [{"case_id": "case-a", "method_id": contract.METHOD_IDS[0]}]
    passed = [{"case_id": "case-a", "method_id": contract.METHOD_IDS[0], "state": "passed"}]
    checks = {name: True for name in contract.HARNESS_ONLY_CHECKS}
    mandatory = contract.HARNESS_ONLY_CHECKS
    assert contract.classify_top_level_status(expected, passed, checks, mandatory) == "complete"
    failed = [dict(passed[0], state="failed")]
    assert contract.classify_top_level_status(expected, failed, checks, mandatory) == "complete_with_failures"
    assert contract.classify_top_level_status(expected, [], checks, mandatory) == "failed"
    assert contract.classify_top_level_status(expected, passed * 2, checks, mandatory) == "failed"
    extra = passed + [{"case_id": "extra", "method_id": contract.METHOD_IDS[1], "state": "passed"}]
    assert contract.classify_top_level_status(expected, extra, checks, mandatory) == "failed"
    running = [dict(passed[0], state="running")]
    assert contract.classify_top_level_status(expected, running, checks, mandatory) == "failed"
    assert contract.classify_top_level_status(
        expected, passed, checks, mandatory, interrupted=True
    ) == "interrupted"
    assert contract.classify_top_level_status(
        expected, passed, checks, mandatory, structural_failure=True
    ) == "failed"


@pytest.mark.parametrize(
    "checks",
    [
        {},
        {"identity_integrity": True},
        {
            "identity_integrity": True,
            "record_integrity": True,
            "extra": True,
        },
        {"identity_integrity": 1, "record_integrity": True},
        {"record_integrity": True, "identity_integrity": True},
    ],
)
def test_top_level_status_rejects_noncanonical_check_contract(checks) -> None:
    expected = [{"case_id": "case-a", "method_id": contract.METHOD_IDS[0]}]
    passed = [{"case_id": "case-a", "method_id": contract.METHOD_IDS[0], "state": "passed"}]
    assert contract.classify_top_level_status(
        expected, passed, checks, contract.HARNESS_ONLY_CHECKS
    ) == "failed"


def test_only_complete_has_zero_exit_code() -> None:
    assert contract.exit_code_for_status("complete") == 0
    for status in ("complete_with_failures", "failed", "interrupted"):
        assert contract.exit_code_for_status(status) == 1


def test_strict_decoder_rejects_malformed_resume_input(tmp_path: Path) -> None:
    path = tmp_path / "resume.json"
    path.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(contract.ContractError):
        contract.read_strict_json(path)
