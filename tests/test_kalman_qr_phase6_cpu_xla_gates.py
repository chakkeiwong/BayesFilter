from __future__ import annotations

import copy
import base64
import functools
import hashlib
import importlib.util
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from scripts import kalman_qr_benchmark_contract as contract


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py"
)


def _load_runner(name: str):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _blob(payload: Any | None, path: str) -> dict[str, Any]:
    if payload is None:
        return {
            "path": path,
            "present": False,
            "byte_count": 0,
            "sha256": None,
            "base64": None,
            "strict_json": None,
        }
    encoded = contract.strict_json_dumps(payload).encode("utf-8")
    return {
        "path": path,
        "present": True,
        "byte_count": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "base64": base64.b64encode(encoded).decode("ascii"),
        "strict_json": payload,
    }


def _raw_blob(raw: bytes, path: str) -> dict[str, Any]:
    return {
        "path": path,
        "present": True,
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode("ascii"),
        "strict_json": None,
    }


def _manifest(entries: Sequence[Mapping[str, str]] = ()) -> dict[str, Any]:
    rows = [dict(entry) for entry in entries]
    return {
        "schema": contract.PHASE6_DEPENDENCY_SCHEMA,
        "repository_root": str(ROOT.resolve()),
        "entries": rows,
        "manifest_sha256": contract.canonical_sha256(rows),
    }


def _discovery_manifest() -> dict[str, Any]:
    return _manifest(
        [
            {
                "module": f"phase6.test_dependency_{index}",
                "path": path,
                "sha256": hashlib.sha256(path.encode("utf-8")).hexdigest(),
            }
            for index, path in enumerate(contract.PHASE6_REQUIRED_SOURCE_PATHS)
        ]
    )


def _progress_journal_blob(
    schedule_row: Mapping[str, Any],
    *,
    stages: Sequence[str],
    attempt_id: str | None = None,
    path: str = "/tmp/phase6-progress.jsonl",
) -> dict[str, Any]:
    raw = b"".join(
        (
            contract.strict_json_dumps(
                {
                    "attempt_id": attempt_id or schedule_row["attempt_id"],
                    "case_id": schedule_row["case_id"],
                    "method_id": schedule_row["identity"]["method_id"],
                    "stage": stage,
                    "resume_key": schedule_row["resume_key"],
                    **schedule_row["fingerprints"],
                }
            )
            + "\n"
        ).encode("utf-8")
        for stage in stages
    )
    return _raw_blob(raw, path)


def _schedule_record(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": dict(identity),
        "case_id": identity["identity_id"],
        "attempt_id": f"phase6-test-{contract.canonical_sha256(identity)[:16]}",
        "config": {
            "dimension": identity["dimension"],
            "parameter_count": identity["parameter_count"],
            "batch_size": identity["batch_size"],
            "timesteps": 120,
            "dtype": identity["dtype"],
            "device": "cpu",
            "cpu_threads": 1,
            "jit_compile": identity["operation"] == "xla",
        },
        "fingerprints": {
            field: hashlib.sha256(field.encode("ascii")).hexdigest()
            for field in contract.FINGERPRINT_FIELDS
        },
        "resume_key": hashlib.sha256(identity["identity_id"].encode("utf-8")).hexdigest(),
        "child_command_argv": ["/bin/true", identity["identity_id"]],
    }


def _schedule(
    identities: Sequence[Mapping[str, Any]], *, gate: str, ledger_schema: str
) -> dict[str, Any]:
    records = [_schedule_record(identity) for identity in identities]
    core = {
        "schema": contract.PHASE6_SCHEDULE_SCHEMA,
        "ledger_schema": ledger_schema,
        "gate": gate,
        "records": records,
    }
    payload = {**core, "schedule_sha256": contract.canonical_sha256(core)}
    return {"payload": payload, "sha256": contract.canonical_sha256(payload)}


@functools.lru_cache(maxsize=None)
def _reviewed_schedule(ledger_schema: str, gate: str) -> dict[str, Any]:
    runner = _load_runner("kalman_qr_phase6_fixture_schedule_builder")
    return runner.phase6_build_schedule(
        ledger_schema, gate=gate, child_timeout_seconds=60
    )


_SYNTHETIC_PREDECESSOR_CACHE: dict[str, dict[str, Any]] = {}


def _bindings(
    identities: Sequence[Mapping[str, Any]],
    *,
    gate: str,
    ledger_schema: str,
    authority_inputs: Sequence[Mapping[str, Any]] = (),
    runtime_predecessors: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    authority_id = "a" * 64
    discovery = _discovery_manifest()
    active_payload = copy.deepcopy(_reviewed_schedule(ledger_schema, gate))
    expected_identities = contract.phase6_expected_roster(ledger_schema)
    if list(identities) != expected_identities:
        wanted = {identity["identity_id"] for identity in identities}
        active_payload["records"] = [
            row
            for row in active_payload["records"]
            if row["identity"]["identity_id"] in wanted
        ]
        core = {
            key: active_payload[key]
            for key in ("schema", "ledger_schema", "gate", "records")
        }
        active_payload["schedule_sha256"] = contract.canonical_sha256(core)
    active_schedule = {
        "payload": active_payload,
        "sha256": contract.canonical_sha256(active_payload),
    }
    sibling_schemas = (
        (contract.PHASE6_TRACE_SCHEMA, contract.PHASE6_PILOT_SCHEMA)
        if gate == "gate_b"
        else (contract.PHASE6_SCALAR_SCHEMA, contract.PHASE6_FINAL_SCHEMA)
    )
    schedules = {
        schema: (
            active_schedule["payload"]
            if schema == ledger_schema
            else copy.deepcopy(_reviewed_schedule(schema, gate))
        )
        for schema in sibling_schemas
    }
    proposal = {
        "authority_id": authority_id,
        "gate": gate,
        "dependency_discovery": {"manifest": discovery},
        "schedules": copy.deepcopy(schedules),
        "inputs": [
            {"path": blob["path"], "sha256": blob["sha256"]}
            for blob in authority_inputs
        ],
        "artifacts": dict(
            contract.PHASE6_GATE_B_ARTIFACTS
            if gate == "gate_b"
            else contract.PHASE6_GATE_C_ARTIFACTS
        ),
    }
    attestation = {"authority_id": authority_id, "gate": gate}
    bindings = {
        "authority_id": authority_id,
        "proposal": _blob(proposal, "/tmp/phase6-proposal.json"),
        "attestation": _blob(attestation, "/tmp/phase6-attestation.json"),
        "schedule": active_schedule,
        "phase45_evidence": [
            _blob({"schema": "phase45-test-evidence"}, "/tmp/phase45.json")
        ],
        "authority_inputs": list(authority_inputs),
        "runtime_predecessors": (
            list(runtime_predecessors)
            if runtime_predecessors is not None
            else []
        ),
    }
    producer_schema = {
        contract.PHASE6_PILOT_SCHEMA: contract.PHASE6_TRACE_SCHEMA,
        contract.PHASE6_FINAL_SCHEMA: contract.PHASE6_SCALAR_SCHEMA,
    }.get(ledger_schema)
    if (
        runtime_predecessors is None
        and producer_schema is not None
        and producer_schema in schedules
        and list(identities) == expected_identities
    ):
        producer_bindings = copy.deepcopy(bindings)
        producer_schedule = copy.deepcopy(schedules[producer_schema])
        producer_bindings["schedule"] = {
            "payload": producer_schedule,
            "sha256": contract.canonical_sha256(producer_schedule),
        }
        producer_bindings["runtime_predecessors"] = []
        cache_key = contract.canonical_sha256(
            {
                "schema": producer_schema,
                "bindings": producer_bindings,
            }
        )
        producer = _SYNTHETIC_PREDECESSOR_CACHE.get(cache_key)
        if producer is None:
            producer_gate, producer_kind, _ = contract.PHASE6_SCHEMA_CONTRACTS[
                producer_schema
            ]
            producer = contract.new_phase6_ledger(
                schema=producer_schema,
                gate=producer_gate,
                artifact_kind=producer_kind,
                identities=contract.phase6_expected_roster(producer_schema),
                bindings=producer_bindings,
            )
            for identity, row in zip(
                producer["roster"], producer_schedule["records"], strict=True
            ):
                producer = contract.transition_phase6_record(
                    producer,
                    identity_id=identity["identity_id"],
                    new_state="running",
                    timestamp_utc="2026-07-11T00:00:00+00:00",
                    process=_running_process(row["child_command_argv"]),
                )
                producer = contract.transition_phase6_record(
                    producer,
                    identity_id=identity["identity_id"],
                    new_state="failed",
                    timestamp_utc="2026-07-11T00:00:01+00:00",
                    reason="invalid_child_evidence",
                    process=_terminal_process(
                        row["child_command_argv"], returncode=0
                    ),
                    evidence=_failure_evidence(classification="common_invalidity"),
                )
            producer = contract.finalize_phase6_ledger(producer)
            _SYNTHETIC_PREDECESSOR_CACHE[cache_key] = copy.deepcopy(producer)
        else:
            producer = copy.deepcopy(producer)
        artifact_key = (
            "trace_output_json"
            if producer_schema == contract.PHASE6_TRACE_SCHEMA
            else "scalar_output_json"
        )
        artifact_path = str((ROOT / proposal["artifacts"][artifact_key]).resolve())
        bindings["runtime_predecessors"] = [
            {
                "kind": "same_authority_runtime_predecessor",
                "producer_ledger_schema": producer_schema,
                "authority_id": authority_id,
                "schedule_sha256": producer_schedule["schedule_sha256"],
                "artifact": _blob(producer, artifact_path),
            }
        ]
    return bindings


def _running_process(command: Sequence[str]) -> dict[str, Any]:
    return {
        "command_argv": list(command),
        "cwd": str(ROOT),
        "environment": {"CUDA_VISIBLE_DEVICES": "-1"},
        "pid": 12345,
        "pgid": 12345,
        "process_start_ticks": 100,
        "started_ns": 10,
        "deadline_seconds": 60.0,
    }


def _terminal_process(
    command: Sequence[str], *, returncode: int = 1, timed_out: bool = False
) -> dict[str, Any]:
    empty_digest = hashlib.sha256(b"").hexdigest()
    return {
        "command_argv": list(command),
        "cwd": str(ROOT),
        "environment": {"CUDA_VISIBLE_DEVICES": "-1"},
        "pid": 12345,
        "pgid": 12345,
        "process_start_ticks": 100,
        "started_ns": 10,
        "finished_ns": 20,
        "elapsed_seconds": 1.0e-8,
        "deadline_seconds": 60.0,
        "term_sent": timed_out,
        "kill_sent": False,
        "reaped": True,
        "reap_status": "reaped_direct_child",
        "process_group_gone": True,
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout_bytes": 0,
        "stdout_total_bytes": 0,
        "stdout_capture_status": "complete",
        "stdout_sha256": empty_digest,
        "stdout_base64": "",
        "stdout_tail": "",
        "stderr_bytes": 0,
        "stderr_total_bytes": 0,
        "stderr_capture_status": "complete",
        "stderr_sha256": empty_digest,
        "stderr_base64": "",
        "stderr_tail": "",
    }


def _failure_evidence(
    *,
    classification: str = "cpu_backend_or_method_failure",
    identity: Mapping[str, Any] | None = None,
    schedule_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = _discovery_manifest()
    child_blob = _blob(None, "/tmp/child.json")
    journal_blob = _blob(None, "/tmp/progress.json")
    if identity is not None and schedule_row is not None:
        attempt_id = schedule_row["attempt_id"]
        child = {
            "schema": contract.SCHEMA,
            "method_contract_version": contract.METHOD_CONTRACT_VERSION,
            "case_id": schedule_row["case_id"],
            "method_id": identity["method_id"],
            "resume_key": schedule_row["resume_key"],
            "attempt_id": attempt_id,
            "state": "failed",
            "returncode": 1,
            "timed_out": False,
            "last_entered_stage": "envelope_write",
            "terminal_stage": "envelope_write",
            "failure_stage": "fixture",
            "error": {"type": "RuntimeError", "message": "synthetic failure"},
            "measurement": None,
            "output_metadata": None,
            "outputs": None,
            **schedule_row["fingerprints"],
        }
        child_blob = _blob(child, "/tmp/child.json")
        journal_blob = _progress_journal_blob(
            schedule_row,
            stages=contract.STAGES,
            attempt_id=attempt_id,
            path="/tmp/progress.jsonl",
        )
    return {
        "classification": classification,
        "child_artifact": child_blob,
        "payload_sidecar": _blob(None, "/tmp/payload.json"),
        "progress_journal": journal_blob,
        "dependency_manifest_before_builder": manifest,
        "dependency_manifest_after_terminal": manifest,
        "dependency_coverage_before": True,
        "dependency_coverage_after": True,
    }


def _new_ledger(schema: str) -> dict[str, Any]:
    gate, artifact_kind, _ = contract.PHASE6_SCHEMA_CONTRACTS[schema]
    identities = contract.phase6_expected_roster(schema)
    return contract.new_phase6_ledger(
        schema=schema,
        gate=gate,
        artifact_kind=artifact_kind,
        identities=identities,
        bindings=_bindings(identities, gate=gate, ledger_schema=schema),
    )


def _closed_failed_pilot() -> dict[str, Any]:
    ledger = _new_ledger(contract.PHASE6_PILOT_SCHEMA)
    schedule = ledger["bindings"]["schedule"]["payload"]["records"]
    for identity, row in zip(ledger["roster"], schedule, strict=True):
        command = row["child_command_argv"]
        ledger = contract.transition_phase6_record(
            ledger,
            identity_id=identity["identity_id"],
            new_state="running",
            timestamp_utc="2026-07-11T00:00:00+00:00",
            process=_running_process(command),
        )
        ledger = contract.transition_phase6_record(
            ledger,
            identity_id=identity["identity_id"],
            new_state="failed",
            timestamp_utc="2026-07-11T00:00:01+00:00",
            reason="child_nonzero_exit",
            process=_terminal_process(command),
            evidence=_failure_evidence(
                classification="method_local_failure",
                identity=identity,
                schedule_row=row,
            ),
        )
    return contract.finalize_phase6_ledger(ledger)


def _imported_pilot_fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pilot = _closed_failed_pilot()
    pilot_blob = _blob(pilot, "/tmp/phase6-pilot.json")
    bindings = _bindings(
        contract.phase6_expected_roster(contract.PHASE6_FINAL_SCHEMA),
        gate="gate_c",
        ledger_schema=contract.PHASE6_FINAL_SCHEMA,
        authority_inputs=(pilot_blob,),
    )
    original = pilot["records"][0]
    imported = copy.deepcopy(original)
    imported["imported_from"] = {
        "kind": "gate_b_pilot",
        "pilot_artifact_sha256": pilot_blob["sha256"],
        "pilot_record_sha256": contract.canonical_sha256(original),
    }
    return imported, bindings, pilot_blob


def test_imported_pilot_helper_requires_exact_immutable_original() -> None:
    imported, bindings, _ = _imported_pilot_fixture()
    assert contract.phase6_imported_pilot_record_valid(
        imported, bindings=bindings
    )
    assert contract.phase6_terminal_record_semantics_valid(
        imported, bindings=bindings
    )
    assert imported["state"] == "failed"
    assert imported["imported_from"]["kind"] == "gate_b_pilot"


@pytest.mark.parametrize(
    "mutation",
    [
        "artifact_digest",
        "record_digest",
        "process",
        "evidence",
        "missing_import_metadata",
        "extra_import_metadata",
        "duplicate_pilot_blob",
    ],
)
def test_imported_pilot_helper_rejects_independent_tampering(mutation: str) -> None:
    imported, bindings, pilot_blob = _imported_pilot_fixture()
    if mutation == "artifact_digest":
        imported["imported_from"]["pilot_artifact_sha256"] = "0" * 64
    elif mutation == "record_digest":
        imported["imported_from"]["pilot_record_sha256"] = "0" * 64
    elif mutation == "process":
        imported["process"]["pid"] += 1
    elif mutation == "evidence":
        imported["evidence"]["classification"] = "common_invalidity"
    elif mutation == "missing_import_metadata":
        imported["imported_from"] = None
    elif mutation == "extra_import_metadata":
        imported["imported_from"]["unreviewed"] = True
    elif mutation == "duplicate_pilot_blob":
        bindings["authority_inputs"].append(copy.deepcopy(pilot_blob))
    else:  # pragma: no cover - parameter list is closed above.
        raise AssertionError(mutation)
    assert not contract.phase6_imported_pilot_record_valid(
        imported, bindings=bindings
    )


def test_imported_pilot_helper_rejects_tampered_embedded_pilot_bytes() -> None:
    imported, bindings, _ = _imported_pilot_fixture()
    tampered = copy.deepcopy(bindings["authority_inputs"][0])
    tampered["strict_json"]["records"][0]["process"]["pid"] += 1
    bindings["authority_inputs"] = [tampered]
    assert not contract.phase6_imported_pilot_record_valid(
        imported, bindings=bindings
    )


def test_imported_transition_is_the_only_legal_pending_to_terminal_shortcut() -> None:
    imported, bindings, _ = _imported_pilot_fixture()
    ledger = contract.new_phase6_ledger(
        schema=contract.PHASE6_FINAL_SCHEMA,
        gate="gate_c",
        artifact_kind="cpu_xla_final",
        identities=contract.phase6_expected_roster(contract.PHASE6_FINAL_SCHEMA),
        bindings=bindings,
    )
    identity_id = imported["identity"]["identity_id"]
    with pytest.raises(contract.ContractError, match="illegal Phase 6 transition"):
        contract.transition_phase6_record(
            ledger,
            identity_id=identity_id,
            new_state=imported["state"],
            timestamp_utc="2026-07-11T00:00:00+00:00",
            reason=imported["reason"],
            process=imported["process"],
            evidence=imported["evidence"],
        )

    closed = contract.transition_phase6_record(
        ledger,
        identity_id=identity_id,
        new_state=imported["state"],
        timestamp_utc="2026-07-11T00:00:00+00:00",
        reason=imported["reason"],
        process=imported["process"],
        evidence=imported["evidence"],
        imported_from=imported["imported_from"],
    )
    assert closed["events"] == [
        {
            "update_index": 1,
            "identity_id": identity_id,
            "prior_state": "pending",
            "new_state": imported["state"],
            "timestamp_utc": "2026-07-11T00:00:00+00:00",
            "evidence_sha256": contract.canonical_sha256(
                {
                    "process": imported["process"],
                    "evidence": imported["evidence"],
                    "imported_from": imported["imported_from"],
                }
            ),
        }
    ]
    stripped = copy.deepcopy(closed)
    stripped["records"][0]["imported_from"] = None
    checks = contract.phase6_ledger_checks(stripped, final=False)
    assert checks["transition_log"] is False or checks["records_identity"] is False


@pytest.mark.parametrize(
    ("schema", "count", "first", "last"),
    [
        (contract.PHASE6_TRACE_SCHEMA, 36, (10, 50, 1), (30, 150, 16)),
        (contract.PHASE6_PILOT_SCHEMA, 2, (10, 50, 1), (10, 50, 1)),
        (contract.PHASE6_SCALAR_SCHEMA, 4, (10, 50, 1), (10, 50, 4)),
        (contract.PHASE6_FINAL_SCHEMA, 36, (10, 50, 1), (30, 150, 16)),
        (contract.PHASE6_ROUTING_SCHEMA, 18, (10, 150, 1), (30, 150, 16)),
    ],
)
def test_phase6_rosters_are_exact_and_stably_ordered(
    schema: str,
    count: int,
    first: tuple[int, int, int],
    last: tuple[int, int, int],
) -> None:
    roster = contract.phase6_expected_roster(schema)
    assert len(roster) == count
    assert len({row["identity_id"] for row in roster}) == count
    assert tuple(roster[0][key] for key in ("dimension", "parameter_count", "batch_size")) == first
    assert tuple(roster[-1][key] for key in ("dimension", "parameter_count", "batch_size")) == last
    assert roster[0]["method_id"] == (
        contract.REFERENCE_METHOD_IDS[0]
        if schema == contract.PHASE6_SCALAR_SCHEMA
        else contract.PRIMARY_METHOD_IDS[0]
    )


def test_new_ledger_rejects_roster_or_schedule_binding_drift() -> None:
    schema = contract.PHASE6_PILOT_SCHEMA
    identities = contract.phase6_expected_roster(schema)
    bindings = _bindings(
        identities, gate="gate_b", ledger_schema=contract.PHASE6_PILOT_SCHEMA
    )
    with pytest.raises(contract.ContractError, match="roster"):
        contract.new_phase6_ledger(
            schema=schema,
            gate="gate_b",
            artifact_kind="cpu_xla_pilot",
            identities=list(reversed(identities)),
            bindings=bindings,
        )

    incomplete_bindings = _bindings(
        identities[:-1],
        gate="gate_b",
        ledger_schema=contract.PHASE6_PILOT_SCHEMA,
    )
    with pytest.raises(contract.ContractError, match="bindings|schedule"):
        contract.new_phase6_ledger(
            schema=schema,
            gate="gate_b",
            artifact_kind="cpu_xla_pilot",
            identities=identities,
            bindings=incomplete_bindings,
        )


def test_new_ledger_rejects_binding_authority_from_another_gate() -> None:
    schema = contract.PHASE6_PILOT_SCHEMA
    identities = contract.phase6_expected_roster(schema)
    wrong_gate_bindings = _bindings(
        identities, gate="gate_c", ledger_schema=contract.PHASE6_PILOT_SCHEMA
    )
    with pytest.raises(contract.ContractError, match="bindings|gate"):
        contract.new_phase6_ledger(
            schema=schema,
            gate="gate_b",
            artifact_kind="cpu_xla_pilot",
            identities=identities,
            bindings=wrong_gate_bindings,
        )


def test_mixed_format_authority_inputs_bind_exact_bytes_and_construct_ledger() -> None:
    identities = contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA)
    authority_inputs = (
        _blob({"schema": "r2-invalid-harness-archive"}, "/tmp/r2-archive.json"),
        _raw_blob(b"# Repair result\n", "/tmp/repair-result.md"),
        _raw_blob(b"# Plan review\n", "/tmp/plan-review.md"),
        _raw_blob(b"# Result review\n", "/tmp/result-review.md"),
    )
    bindings = _bindings(
        identities,
        gate="gate_b",
        ledger_schema=contract.PHASE6_TRACE_SCHEMA,
        authority_inputs=authority_inputs,
    )

    assert [blob["strict_json"] is not None for blob in bindings["authority_inputs"]] == [
        True,
        False,
        False,
        False,
    ]
    assert contract._phase6_bindings_valid(bindings)
    ledger = contract.new_phase6_ledger(
        schema=contract.PHASE6_TRACE_SCHEMA,
        gate="gate_b",
        artifact_kind="trace_census",
        identities=identities,
        bindings=bindings,
    )
    assert all(contract.phase6_ledger_checks(ledger, final=False).values())


def test_real_files_one_json_three_markdown_construct_mixed_format_ledger(
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_real_mixed_format_files")
    identities = contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA)
    archive = tmp_path / "r2-archive.json"
    result = tmp_path / "repair-result.md"
    plan_review = tmp_path / "plan-review.md"
    result_review = tmp_path / "result-review.md"
    proposal_path = tmp_path / "proposal.json"
    attestation_path = tmp_path / "attestation.json"
    phase4 = tmp_path / "phase4.json"
    phase5 = tmp_path / "phase5.json"
    contract.durable_atomic_write_json(
        archive, {"schema": "r2-invalid-harness-archive"}
    )
    for path, text in (
        (result, "# Repair result\n"),
        (plan_review, "# Plan review\n"),
        (result_review, "# Result review\n"),
    ):
        path.write_text(text, encoding="ascii")
    schedule = copy.deepcopy(
        _reviewed_schedule(contract.PHASE6_TRACE_SCHEMA, "gate_b")
    )
    sibling = copy.deepcopy(
        _reviewed_schedule(contract.PHASE6_PILOT_SCHEMA, "gate_b")
    )
    inputs = (archive, result, plan_review, result_review)
    authority_id = "b" * 64
    proposal = {
        "authority_id": authority_id,
        "gate": "gate_b",
        "dependency_discovery": {"manifest": _discovery_manifest()},
        "schedules": {
            contract.PHASE6_TRACE_SCHEMA: schedule,
            contract.PHASE6_PILOT_SCHEMA: sibling,
        },
        "inputs": [contract.path_digest_record(path) for path in inputs],
        "artifacts": dict(contract.PHASE6_GATE_B_ARTIFACTS),
    }
    contract.durable_atomic_write_json(proposal_path, proposal)
    contract.durable_atomic_write_json(
        attestation_path, {"authority_id": authority_id, "gate": "gate_b"}
    )
    contract.durable_atomic_write_json(phase4, {"schema": "phase4"})
    contract.durable_atomic_write_json(phase5, {"schema": "phase5"})

    bindings = runner.phase6_build_bindings(
        proposal_path=proposal_path,
        attestation_path=attestation_path,
        schedule=schedule,
        phase45_paths=(phase4, phase5),
        authority_input_paths=inputs,
    )
    assert [
        blob["strict_json"] is not None for blob in bindings["authority_inputs"]
    ] == [True, False, False, False]
    assert contract._phase6_bindings_valid(bindings)
    ledger = contract.new_phase6_ledger(
        schema=contract.PHASE6_TRACE_SCHEMA,
        gate="gate_b",
        artifact_kind="trace_census",
        identities=identities,
        bindings=bindings,
    )
    assert all(contract.phase6_ledger_checks(ledger, final=False).values())


@pytest.mark.parametrize(
    "mutation",
    ("bytes", "base64", "path", "digest", "order", "presence"),
)
def test_mixed_format_authority_inputs_reject_independent_mutations(
    mutation: str,
) -> None:
    identities = contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA)
    inputs = [
        _blob({"schema": "r2-invalid-harness-archive"}, "/tmp/r2-archive.json"),
        _raw_blob(b"# Repair result\n", "/tmp/repair-result.md"),
        _raw_blob(b"# Plan review\n", "/tmp/plan-review.md"),
        _raw_blob(b"# Result review\n", "/tmp/result-review.md"),
    ]
    bindings = _bindings(
        identities,
        gate="gate_b",
        ledger_schema=contract.PHASE6_TRACE_SCHEMA,
        authority_inputs=inputs,
    )
    if mutation == "bytes":
        bindings["authority_inputs"][1]["byte_count"] += 1
    elif mutation == "base64":
        bindings["authority_inputs"][1]["base64"] = "not-base64"
    elif mutation == "path":
        bindings["authority_inputs"][1]["path"] = "/tmp/substitute.md"
    elif mutation == "digest":
        bindings["authority_inputs"][1]["sha256"] = "0" * 64
    elif mutation == "order":
        bindings["authority_inputs"][1:3] = reversed(
            bindings["authority_inputs"][1:3]
        )
    else:
        bindings["authority_inputs"][1] = _blob(None, "/tmp/repair-result.md")

    assert not contract._phase6_bindings_valid(bindings)


@pytest.mark.parametrize(
    "category",
    ("proposal", "attestation", "phase45", "runtime_predecessor"),
)
def test_semantically_parsed_binding_categories_still_require_strict_json(
    category: str,
) -> None:
    schema = (
        contract.PHASE6_PILOT_SCHEMA
        if category == "runtime_predecessor"
        else contract.PHASE6_TRACE_SCHEMA
    )
    identities = contract.phase6_expected_roster(schema)
    bindings = _bindings(identities, gate="gate_b", ledger_schema=schema)
    assert contract._phase6_bindings_valid(bindings)
    if category in {"proposal", "attestation"}:
        bindings[category]["strict_json"] = None
    elif category == "phase45":
        bindings["phase45_evidence"][0]["strict_json"] = None
    else:
        bindings["runtime_predecessors"][0]["artifact"]["strict_json"] = None

    assert not contract._phase6_bindings_valid(bindings)


def test_child_authority_snapshot_binds_one_exact_schedule_row() -> None:
    identities = contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA)
    bindings = _bindings(
        identities, gate="gate_b", ledger_schema=contract.PHASE6_TRACE_SCHEMA
    )
    row = bindings["schedule"]["payload"]["records"][0]
    snapshot = contract.phase6_child_authority_snapshot(bindings, row)

    assert contract.phase6_child_authority_snapshot_valid(snapshot)
    assert contract.durable_json_sha256(snapshot) == hashlib.sha256(
        (contract.strict_json_dumps(snapshot, indent=2) + "\n").encode("utf-8")
    ).hexdigest()
    changed = copy.deepcopy(snapshot)
    changed["schedule_row"] = bindings["schedule"]["payload"]["records"][1]
    assert not contract.phase6_child_authority_snapshot_valid(changed)


@pytest.mark.parametrize(
    ("schema", "gate", "mutation"),
    [
        (contract.PHASE6_PILOT_SCHEMA, "gate_b", "missing"),
        (contract.PHASE6_PILOT_SCHEMA, "gate_b", "wrong_schema"),
        (contract.PHASE6_FINAL_SCHEMA, "gate_c", "wrong_authority"),
        (contract.PHASE6_FINAL_SCHEMA, "gate_c", "duplicate"),
    ],
)
def test_bindings_require_exact_same_authority_runtime_predecessor_roster(
    schema: str,
    gate: str,
    mutation: str,
) -> None:
    identities = contract.phase6_expected_roster(schema)
    bindings = _bindings(identities, gate=gate, ledger_schema=schema)
    assert contract._phase6_bindings_valid(bindings)
    if mutation == "missing":
        bindings["runtime_predecessors"] = []
    elif mutation == "wrong_schema":
        bindings["runtime_predecessors"][0]["producer_ledger_schema"] = (
            contract.PHASE6_SCALAR_SCHEMA
        )
    elif mutation == "wrong_authority":
        bindings["runtime_predecessors"][0]["authority_id"] = "b" * 64
    elif mutation == "duplicate":
        bindings["runtime_predecessors"].append(
            copy.deepcopy(bindings["runtime_predecessors"][0])
        )
    else:  # pragma: no cover - parameter list is closed above.
        raise AssertionError(mutation)
    assert not contract._phase6_bindings_valid(bindings)
    with pytest.raises(contract.ContractError, match="bindings"):
        contract.new_phase6_ledger(
            schema=schema,
            gate=gate,
            artifact_kind=contract.PHASE6_SCHEMA_CONTRACTS[schema][1],
            identities=identities,
            bindings=bindings,
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing_sibling", "substitute_sibling", "command_drift", "schema_relabel"],
)
def test_binding_schedule_must_equal_exact_reviewed_schema_schedule(
    mutation: str,
) -> None:
    schema = contract.PHASE6_PILOT_SCHEMA
    identities = contract.phase6_expected_roster(schema)
    bindings = _bindings(identities, gate="gate_b", ledger_schema=schema)
    if mutation == "missing_sibling":
        proposal = bindings["proposal"]["strict_json"]
        proposal["schedules"].pop(
            contract.PHASE6_TRACE_SCHEMA
        )
        bindings["proposal"] = _blob(proposal, "/tmp/phase6-proposal.json")
    elif mutation == "substitute_sibling":
        sibling = bindings["proposal"]["strict_json"]["schedules"][
            contract.PHASE6_TRACE_SCHEMA
        ]
        bindings["schedule"] = {
            "payload": sibling,
            "sha256": contract.canonical_sha256(sibling),
        }
    elif mutation == "command_drift":
        payload = bindings["schedule"]["payload"]
        payload["records"][0]["child_command_argv"].append("--drift")
        core = {
            key: payload[key]
            for key in ("schema", "ledger_schema", "gate", "records")
        }
        payload["schedule_sha256"] = contract.canonical_sha256(core)
        bindings["schedule"]["sha256"] = contract.canonical_sha256(payload)
    elif mutation == "schema_relabel":
        payload = bindings["schedule"]["payload"]
        payload["ledger_schema"] = contract.PHASE6_TRACE_SCHEMA
        core = {
            key: payload[key]
            for key in ("schema", "ledger_schema", "gate", "records")
        }
        payload["schedule_sha256"] = contract.canonical_sha256(core)
        bindings["schedule"]["sha256"] = contract.canonical_sha256(payload)
    else:  # pragma: no cover - parameter list is closed above.
        raise AssertionError(mutation)
    with pytest.raises(contract.ContractError, match="binding|schedule|roster"):
        contract.new_phase6_ledger(
            schema=schema,
            gate="gate_b",
            artifact_kind="cpu_xla_pilot",
            identities=identities,
            bindings=bindings,
        )


def test_phase6_transition_is_prefix_serial_and_event_replay_is_binding() -> None:
    ledger = _new_ledger(contract.PHASE6_PILOT_SCHEMA)
    first, second = ledger["roster"]
    first_command = ledger["bindings"]["schedule"]["payload"]["records"][0][
        "child_command_argv"
    ]
    running = contract.transition_phase6_record(
        ledger,
        identity_id=first["identity_id"],
        new_state="running",
        timestamp_utc="2026-07-11T00:00:00+00:00",
        process=_running_process(first_command),
    )
    second_command = ledger["bindings"]["schedule"]["payload"]["records"][1][
        "child_command_argv"
    ]
    with pytest.raises(contract.ContractError, match="transition|prefix|running"):
        contract.transition_phase6_record(
            running,
            identity_id=second["identity_id"],
            new_state="running",
            timestamp_utc="2026-07-11T00:00:01+00:00",
            process=_running_process(second_command),
        )

    closed = contract.transition_phase6_record(
        running,
        identity_id=first["identity_id"],
        new_state="failed",
        timestamp_utc="2026-07-11T00:00:02+00:00",
        reason="child_nonzero_exit",
        process=_terminal_process(first_command),
        evidence=_failure_evidence(
            identity=first,
            schedule_row=ledger["bindings"]["schedule"]["payload"]["records"][0],
        ),
    )
    assert all(contract.phase6_ledger_checks(closed, final=False).values())
    next_running = contract.transition_phase6_record(
        closed,
        identity_id=second["identity_id"],
        new_state="running",
        timestamp_utc="2026-07-11T00:00:03+00:00",
        process=_running_process(second_command),
    )
    assert all(contract.phase6_ledger_checks(next_running, final=False).values())

    tampered = copy.deepcopy(next_running)
    tampered["events"][0]["identity_id"] = second["identity_id"]
    assert contract.phase6_ledger_checks(tampered, final=False)["transition_log"] is False


def test_not_launched_records_require_exact_reason_and_finalize_cleanly() -> None:
    ledger = _new_ledger(contract.PHASE6_PILOT_SCHEMA)
    for identity in ledger["roster"]:
        ledger = contract.transition_phase6_record(
            ledger,
            identity_id=identity["identity_id"],
            new_state="not_launched:trace_gate_not_passed",
            timestamp_utc="2026-07-11T00:00:00+00:00",
            reason="trace_gate_not_passed",
        )
    finalized = contract.finalize_phase6_ledger(ledger)
    assert finalized["state"] == "complete_with_failures"
    assert all(contract.phase6_ledger_checks(finalized, final=True).values())
    tampered = copy.deepcopy(finalized)
    tampered["records"][0]["reason"] = "common_invalidity"
    assert contract.phase6_ledger_checks(tampered, final=True)["records_identity"] is False


def test_process_and_evidence_records_fail_closed() -> None:
    command = ["/bin/true", "child"]
    process = _terminal_process(command)
    evidence = _failure_evidence()
    assert contract.phase6_process_record_valid(process, terminal_state="failed")
    assert contract.phase6_evidence_record_valid(evidence, terminal_state="failed")

    for field, value in (
        ("process_group_gone", False),
        ("reaped", False),
        ("stdout_tail", "x" * 4001),
    ):
        changed = copy.deepcopy(process)
        changed[field] = value
        assert not contract.phase6_process_record_valid(changed, terminal_state="failed")
    changed = copy.deepcopy(evidence)
    changed["dependency_coverage_after"] = "yes"
    assert not contract.phase6_evidence_record_valid(changed, terminal_state="failed")

    passed = copy.deepcopy(evidence)
    passed["classification"] = "method_pass"
    assert not contract.phase6_evidence_record_valid(passed, terminal_state="passed")


@pytest.mark.parametrize(
    ("status", "total_bytes", "terminal_state", "valid"),
    [
        ("complete", 0, "failed", True),
        ("truncated_at_cap", 1, "failed", True),
        ("unavailable_after_recovery", None, "interrupted", True),
        ("truncated_at_cap", 1, "passed", False),
        ("unavailable_after_recovery", None, "passed", False),
        ("complete", 1, "failed", False),
        ("truncated_at_cap", 0, "failed", False),
        ("unavailable_after_recovery", 0, "interrupted", False),
    ],
)
def test_stream_capture_status_is_exact_and_cannot_promote_incomplete_output(
    status: str,
    total_bytes: int | None,
    terminal_state: str,
    valid: bool,
) -> None:
    returncode = 0 if terminal_state == "passed" else 1
    process = _terminal_process(["/bin/true"], returncode=returncode)
    for prefix in ("stdout", "stderr"):
        process[f"{prefix}_capture_status"] = status
        process[f"{prefix}_total_bytes"] = total_bytes
    assert (
        contract.phase6_process_record_valid(process, terminal_state=terminal_state)
        is valid
    )


def test_stream_embedded_bytes_hash_base64_and_tail_are_one_identity() -> None:
    process = _terminal_process(["/bin/true"])
    mutations = [
        ("stdout_bytes", 1),
        ("stdout_total_bytes", -1),
        ("stdout_capture_status", "partial"),
        ("stdout_sha256", "0" * 64),
        ("stdout_base64", "YQ=="),
        ("stdout_tail", "changed"),
    ]
    for field, value in mutations:
        changed = copy.deepcopy(process)
        changed[field] = value
        assert not contract.phase6_process_record_valid(
            changed, terminal_state="failed"
        )


def _trace_child(
    identity: Mapping[str, Any],
    manifest: Mapping[str, Any],
    schedule_row: Mapping[str, Any],
) -> dict[str, Any]:
    from tensorflow.core.framework import graph_pb2, types_pb2

    graph = graph_pb2.GraphDef()
    node = graph.node.add(name="parameters_batch", op="Placeholder")
    node.attr["dtype"].type = types_pb2.DT_FLOAT
    node.attr["shape"].shape.dim.add().size = identity["batch_size"]
    node.attr["shape"].shape.dim.add().size = identity["parameter_count"]
    raw = graph.SerializeToString(deterministic=True)
    tokens = contract.graphdef_token_stream(raw)
    return {
        "schema": "bayesfilter.kalman_qr_batched_xla_repair.phase6.trace_child.v1",
        "state": "passed",
        "identity": dict(identity),
        "case_id": schedule_row["case_id"],
        "attempt_id": "phase6-test-trace-attempt",
        **schedule_row["fingerprints"],
        "resume_key": schedule_row["resume_key"],
        "stage": "terminal_provenance",
        "started_ns": 10,
        "finished_ns": 20,
        "elapsed_seconds": 1.0e-8,
        "command_argv": schedule_row["child_command_argv"],
        "dependency_manifest_before_builder": dict(manifest),
        "dependency_manifest_after_terminal": dict(manifest),
        "evidence": {
            "identity": dict(identity),
            "timesteps": 120,
            "requested_device": "cpu",
            "cuda_visible_devices": "-1",
            "jit_compile": False,
            "tf32_queried": False,
            "device_enumeration_api_calls": 0,
            "invoked_method_ids": [identity["method_id"]],
            "get_concrete_function_calls": 1,
            "concrete_function_invocations": 0,
            "structured_user_input": {
                "name": "parameters_batch",
                "dtype": "float32",
                "shape": [identity["batch_size"], identity["parameter_count"]],
            },
            "concrete_outputs": [
                {
                    "name": "value:0",
                    "dtype": "float32",
                    "shape": [identity["batch_size"]],
                    "result_position": "value",
                },
                {
                    "name": "score:0",
                    "dtype": "float32",
                    "shape": [identity["batch_size"], identity["parameter_count"]],
                    "result_position": "score",
                },
            ],
            "graphdef_bytes": contract.graphdef_bytes_record(raw),
            "typed_token_stream": tokens,
            "typed_token_stream_sha256": contract.canonical_sha256(tokens),
            "top_level_node_count": 1,
            "function_count": 0,
        },
        "error": None,
        "nonclaims": list(contract.PHASE6_NONCLAIMS),
    }


def test_passed_trace_evidence_binds_command_provenance_and_no_execution() -> None:
    identity = contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA)[0]
    bindings = _bindings(
        contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA),
        gate="gate_b",
        ledger_schema=contract.PHASE6_TRACE_SCHEMA,
    )
    schedule_row = bindings["schedule"]["payload"]["records"][0]
    manifest = bindings["proposal"]["strict_json"]["dependency_discovery"]["manifest"]
    child = _trace_child(identity, manifest, schedule_row)
    evidence = {
        "classification": "trace_pass",
        "child_artifact": _blob(child, "/tmp/trace-child.json"),
        "payload_sidecar": _blob(None, "/tmp/trace-sidecar.json"),
        "progress_journal": _progress_journal_blob(
            schedule_row,
            stages=contract.PHASE6_TRACE_STAGES,
            attempt_id=child["attempt_id"],
            path="/tmp/trace-progress.jsonl",
        ),
        "dependency_manifest_before_builder": manifest,
        "dependency_manifest_after_terminal": manifest,
        "dependency_coverage_before": True,
        "dependency_coverage_after": True,
    }
    command = bindings["schedule"]["payload"]["records"][0]["child_command_argv"]
    record = {
        "identity": identity,
        "state": "passed",
        "reason": "child_passed",
        "process": _terminal_process(command, returncode=0),
        "evidence": evidence,
    }
    assert contract.phase6_terminal_record_semantics_valid(record, bindings=bindings)
    assert contract.phase6_evidence_record_valid(evidence, terminal_state="passed")

    invoked = copy.deepcopy(record)
    invoked_child = invoked["evidence"]["child_artifact"]["strict_json"]
    invoked_child["evidence"]["concrete_function_invocations"] = 1
    invoked["evidence"]["child_artifact"] = _blob(
        invoked_child, "/tmp/trace-child.json"
    )
    assert not contract.phase6_terminal_record_semantics_valid(
        invoked, bindings=bindings
    )
    wrong_command = copy.deepcopy(record)
    wrong_command["process"]["command_argv"] = ["/bin/false"]
    assert not contract.phase6_terminal_record_semantics_valid(
        wrong_command, bindings=bindings
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_stage",
        "skipped_stage",
        "reordered_stage",
        "wrong_attempt_id",
        "wrong_fingerprint",
        "malformed_line",
        "missing_terminal_newline",
        "wrong_last_stage",
    ],
)
def test_trace_progress_journal_rejects_noncontiguous_or_unbound_events(
    mutation: str,
) -> None:
    bindings = _bindings(
        contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA),
        gate="gate_b",
        ledger_schema=contract.PHASE6_TRACE_SCHEMA,
    )
    row = bindings["schedule"]["payload"]["records"][0]
    attempt_id = "phase6-trace-journal-attempt"
    events = [
        {
            "attempt_id": attempt_id,
            "case_id": row["case_id"],
            "method_id": row["identity"]["method_id"],
            "stage": stage,
            "resume_key": row["resume_key"],
            **row["fingerprints"],
        }
        for stage in contract.PHASE6_TRACE_STAGES
    ]
    if mutation == "duplicate_stage":
        events.insert(2, copy.deepcopy(events[1]))
    elif mutation == "skipped_stage":
        events.pop(2)
    elif mutation == "reordered_stage":
        events[1], events[2] = events[2], events[1]
    elif mutation == "wrong_attempt_id":
        events[2]["attempt_id"] = "another-attempt"
    elif mutation == "wrong_fingerprint":
        events[2][contract.FINGERPRINT_FIELDS[0]] = "0" * 64
    elif mutation == "wrong_last_stage":
        events.pop()
    elif mutation not in {"malformed_line", "missing_terminal_newline"}:
        raise AssertionError(mutation)

    raw = b"".join(
        (contract.strict_json_dumps(event) + "\n").encode("utf-8")
        for event in events
    )
    if mutation == "malformed_line":
        raw = raw.splitlines(keepends=True)[0] + b"{malformed}\n"
    elif mutation == "missing_terminal_newline":
        raw = raw.rstrip(b"\n")
    blob = _raw_blob(raw, "/tmp/mutated-trace-progress.jsonl")
    assert not contract.phase6_progress_journal_valid(
        blob,
        schedule_row=row,
        trace=True,
        expected_last_stage="envelope_write",
        expected_attempt_id=attempt_id,
    )


def test_trace_progress_journal_accepts_only_the_exact_contiguous_prefix() -> None:
    bindings = _bindings(
        contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA),
        gate="gate_b",
        ledger_schema=contract.PHASE6_TRACE_SCHEMA,
    )
    row = bindings["schedule"]["payload"]["records"][0]
    attempt_id = "phase6-trace-prefix-attempt"
    for prefix_length in range(1, len(contract.PHASE6_TRACE_STAGES) + 1):
        stages = contract.PHASE6_TRACE_STAGES[:prefix_length]
        assert contract.phase6_progress_journal_valid(
            _progress_journal_blob(
                row,
                stages=stages,
                attempt_id=attempt_id,
                path=f"/tmp/trace-prefix-{prefix_length}.jsonl",
            ),
            schedule_row=row,
            trace=True,
            expected_last_stage=stages[-1],
            expected_attempt_id=attempt_id,
        )


def _failed_trace_record(
    *,
    identity: Mapping[str, Any],
    bindings: Mapping[str, Any],
    malformed_journal: bool = False,
) -> dict[str, Any]:
    row = bindings["schedule"]["payload"]["records"][0]
    manifest = bindings["proposal"]["strict_json"]["dependency_discovery"]["manifest"]
    attempt_id = row["attempt_id"]
    child = {
        "schema": "bayesfilter.kalman_qr_batched_xla_repair.phase6.trace_child.v1",
        "state": "failed",
        "identity": dict(identity),
        "case_id": row["case_id"],
        "attempt_id": attempt_id,
        **row["fingerprints"],
        "resume_key": row["resume_key"],
        "stage": "selected_method_construction",
        "started_ns": 10,
        "finished_ns": 20,
        "elapsed_seconds": 1.0e-8,
        "command_argv": row["child_command_argv"],
        "dependency_manifest_before_builder": manifest,
        "dependency_manifest_after_terminal": manifest,
        "evidence": None,
        "error": {"type": "RuntimeError", "message": "trace construction failed"},
        "nonclaims": list(contract.PHASE6_NONCLAIMS),
    }
    stages = [
        "fixture",
        "pre_builder_provenance",
        "selected_method_construction",
        "envelope_write",
    ]
    journal = _progress_journal_blob(
        row,
        stages=stages,
        attempt_id=attempt_id,
        path="/tmp/failed-trace-progress.jsonl",
    )
    if malformed_journal:
        raw = base64.b64decode(journal["base64"], validate=True)
        journal = _raw_blob(raw[:-1], "/tmp/malformed-failed-trace-progress.jsonl")
    evidence = {
        "classification": (
            "common_invalidity" if malformed_journal else "trace_structural_failure"
        ),
        "child_artifact": _blob(child, "/tmp/failed-trace-child.json"),
        "payload_sidecar": _blob(None, "/tmp/failed-trace-sidecar.json"),
        "progress_journal": journal,
        "dependency_manifest_before_builder": manifest,
        "dependency_manifest_after_terminal": manifest,
        "dependency_coverage_before": True,
        "dependency_coverage_after": True,
    }
    return {
        "identity": identity,
        "state": "failed",
        "reason": "invalid_child_evidence" if malformed_journal else "child_nonzero_exit",
        "process": _terminal_process(
            row["child_command_argv"], returncode=0 if malformed_journal else 1
        ),
        "evidence": evidence,
        "imported_from": None,
    }


def test_honest_trace_failure_is_not_reclassified_as_common_invalidity() -> None:
    identity = contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA)[0]
    bindings = _bindings(
        contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA),
        gate="gate_b",
        ledger_schema=contract.PHASE6_TRACE_SCHEMA,
    )
    record = _failed_trace_record(identity=identity, bindings=bindings)
    assert contract.phase6_terminal_record_semantics_valid(record, bindings=bindings)
    malformed = _failed_trace_record(
        identity=identity,
        bindings=bindings,
        malformed_journal=True,
    )
    assert contract.phase6_terminal_record_semantics_valid(
        malformed, bindings=bindings
    )


@pytest.mark.parametrize(
    ("schema", "state", "reason", "classification", "returncode", "timed_out"),
    [
        (
            contract.PHASE6_TRACE_SCHEMA,
            "timed_out",
            "child_execution_deadline_exceeded",
            "trace_timeout",
            -signal.SIGTERM,
            True,
        ),
        (
            contract.PHASE6_PILOT_SCHEMA,
            "crashed",
            "child_signal_exit",
            "cpu_backend_or_method_failure",
            -signal.SIGSEGV,
            False,
        ),
    ],
)
def test_early_timeout_or_crash_accepts_truthfully_absent_dependency_sidecars(
    schema: str,
    state: str,
    reason: str,
    classification: str,
    returncode: int,
    timed_out: bool,
) -> None:
    gate = contract.PHASE6_SCHEMA_CONTRACTS[schema][0]
    identity = contract.phase6_expected_roster(schema)[0]
    bindings = _bindings(
        contract.phase6_expected_roster(schema),
        gate=gate,
        ledger_schema=schema,
    )
    row = bindings["schedule"]["payload"]["records"][0]
    evidence = {
        "classification": classification,
        "child_artifact": _blob(None, "/tmp/early-child.json"),
        "payload_sidecar": _blob(None, "/tmp/early-sidecar.json"),
        "progress_journal": _progress_journal_blob(
            row,
            stages=("fixture",),
            attempt_id=row["attempt_id"],
            path="/tmp/early-progress.jsonl",
        ),
        "dependency_manifest_before_builder": None,
        "dependency_manifest_after_terminal": None,
        "dependency_coverage_before": False,
        "dependency_coverage_after": False,
    }
    record = {
        "identity": identity,
        "state": state,
        "reason": reason,
        "process": _terminal_process(
            row["child_command_argv"],
            returncode=returncode,
            timed_out=timed_out,
        ),
        "evidence": evidence,
        "imported_from": None,
    }
    assert contract.phase6_evidence_record_valid(evidence, terminal_state=state)
    assert contract.phase6_terminal_record_semantics_valid(record, bindings=bindings)

    dishonest = copy.deepcopy(record)
    dishonest["evidence"]["dependency_coverage_before"] = True
    assert not contract.phase6_terminal_record_semantics_valid(
        dishonest, bindings=bindings
    )


@pytest.mark.parametrize(
    ("returncode", "timed_out", "expected"),
    [
        (1, False, ("failed", "child_nonzero_exit", "trace_structural_failure")),
        (1, True, ("timed_out", "child_execution_deadline_exceeded", "trace_timeout")),
        (-signal.SIGSEGV, False, ("crashed", "child_signal_exit", "trace_crash")),
    ],
)
def test_trace_process_outcomes_have_lane_specific_initial_classification(
    returncode: int,
    timed_out: bool,
    expected: tuple[str, str, str],
) -> None:
    runner = _load_runner(
        f"kalman_qr_phase6_trace_classification_{returncode}_{timed_out}"
    )
    identity = contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA)[0]
    process = _terminal_process(
        ["/bin/true", identity["identity_id"]],
        returncode=returncode,
        timed_out=timed_out,
    )
    assert runner._phase6_terminal_classification(identity, process) == expected


def test_blob_strict_json_is_reparsed_and_bound_to_exact_bytes() -> None:
    blob = _blob({"state": "passed", "checks": {"a": True}}, "/tmp/result.json")
    assert contract.phase6_blob_record_valid(blob)
    changed = copy.deepcopy(blob)
    changed["strict_json"]["state"] = "failed"
    assert not contract.phase6_blob_record_valid(changed)
    changed = copy.deepcopy(blob)
    changed["base64"] += "\n"
    assert not contract.phase6_blob_record_valid(changed)


def test_dependency_manifest_is_subset_bound_and_required_path_complete() -> None:
    entries = [
        {"module": "lane.a", "path": "scripts/a.py", "sha256": "1" * 64},
        {"module": "lane.b", "path": "scripts/b.py", "sha256": "2" * 64},
    ]
    discovery = _manifest(entries)
    actual = _manifest(entries[:1])
    assert contract.dependency_manifest_covers(
        discovery, actual, required_paths=["scripts/a.py", "scripts/b.py"]
    )
    unseen = _manifest(
        [
            {
                "module": "lane.c",
                "path": "scripts/c.py",
                "sha256": "3" * 64,
            }
        ]
    )
    assert not contract.dependency_manifest_covers(discovery, unseen)
    assert not contract.dependency_manifest_covers(
        discovery, actual, required_paths=["scripts/missing.py"]
    )
    corrupt = copy.deepcopy(actual)
    corrupt["entries"][0]["sha256"] = "4" * 64
    assert not contract.dependency_manifest_covers(discovery, corrupt)


def test_repository_manifest_rejects_symlinked_repository_module(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n", encoding="ascii")
    manifest = contract.repository_module_manifest(
        tmp_path, modules={"local_module": SimpleNamespace(__file__=str(source))}
    )
    assert manifest["entries"][0]["path"] == "module.py"
    link = tmp_path / "link.py"
    link.symlink_to(source)
    with pytest.raises(contract.ContractError, match="non-symlink"):
        contract.repository_module_manifest(
            tmp_path, modules={"linked_module": SimpleNamespace(__file__=str(link))}
        )


def test_opening_hash_ledger_is_exact_frozen_lane_inventory() -> None:
    path = Path(contract.PHASE6_OPENING_HASH_LEDGER)
    record = contract.phase6_opening_hash_ledger_record(path)
    entries = record["entries"]
    assert record["sha256"] == contract.PHASE6_OPENING_HASH_LEDGER_SHA256
    assert record["entries_sha256"] == (
        "abf01b4989b320a6dd5c1dc22137e45a89e5d8a7cb65471b591826b3a29508f9"
    )
    assert len(entries) == 144
    assert sum(row["opening_state"] == "present" for row in entries) == 131
    assert sum(row["opening_state"] == "absent" for row in entries) == 13
    expected_nonhistorical = {
        *contract.PHASE6_OPENING_MUTABLE_PATHS,
        *contract.PHASE6_OPENING_FIXED_PATHS,
        *contract.PHASE6_OPENING_ABSENT_PATHS,
    }
    historical = {row["path"] for row in entries} - expected_nonhistorical
    assert len(historical) == 106
    assert all(
        path.startswith(("docs/benchmarks/", "docs/plans/"))
        and "2026-07-09" in path
        for path in historical
    )
    assert contract.phase6_opening_hash_ledger_record_matches(record)


@pytest.mark.parametrize(
    "mutation", ["raw_digest", "entries_digest", "entry_path", "entry_state", "extra_field"]
)
def test_opening_hash_ledger_record_rejects_embedded_mutation(mutation: str) -> None:
    record = contract.phase6_opening_hash_ledger_record(
        Path(contract.PHASE6_OPENING_HASH_LEDGER)
    )
    if mutation == "raw_digest":
        record["sha256"] = "0" * 64
    elif mutation == "entries_digest":
        record["entries_sha256"] = "0" * 64
    elif mutation == "entry_path":
        record["entries"][0]["path"] = "../escape"
        record["entries_sha256"] = contract.canonical_sha256(record["entries"])
    elif mutation == "entry_state":
        record["entries"][-1]["opening_state"] = "present"
        record["entries"][-1]["sha256"] = "0" * 64
        record["entries_sha256"] = contract.canonical_sha256(record["entries"])
    elif mutation == "extra_field":
        record["unexpected"] = True
    else:  # pragma: no cover - parameter list is closed above.
        raise AssertionError(mutation)
    assert not contract.phase6_opening_hash_ledger_record_matches(record)


def test_opening_hash_ledger_parser_rejects_changed_bytes_and_symlink(
    tmp_path: Path,
) -> None:
    frozen = Path(contract.PHASE6_OPENING_HASH_LEDGER)
    changed = tmp_path / "changed.sha256"
    changed.write_bytes(frozen.read_bytes() + b"\n")
    with pytest.raises(contract.ContractError, match="frozen entry digest"):
        contract.phase6_parse_opening_hash_ledger(changed)
    link = tmp_path / "ledger-link.sha256"
    link.symlink_to(frozen)
    with pytest.raises(contract.ContractError, match="non-symlink"):
        contract.phase6_parse_opening_hash_ledger(link)


def test_graphdef_base64_contract_rejects_noncanonical_or_over_cap_records(
    monkeypatch,
) -> None:
    raw = b"phase6-graphdef"
    record = contract.graphdef_bytes_record(raw)
    assert contract.decode_graphdef_bytes_record(record) == raw

    for field, value in (
        ("decoded_bytes", len(raw) + 1),
        ("sha256", "0" * 64),
        ("base64", record["base64"] + "\n"),
        ("encoding", "base64"),
    ):
        changed = copy.deepcopy(record)
        changed[field] = value
        with pytest.raises(contract.ContractError):
            contract.decode_graphdef_bytes_record(changed)

    with pytest.raises(contract.ContractError):
        contract.decode_graphdef_bytes_record(
            record,
            prior_total_decoded_bytes=contract.PHASE6_GRAPHDEF_MAX_TOTAL_DECODED_BYTES,
        )
    monkeypatch.setattr(contract, "PHASE6_GRAPHDEF_MAX_DECODED_BYTES", len(raw) - 1)
    with pytest.raises(contract.ContractError, match="size limits"):
        contract.graphdef_bytes_record(raw)


def test_bounded_trace_reader_rejects_symlinks_and_preparse_size_overflow(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = tmp_path / "trace.json"
    artifact.write_text('{"ok":true}', encoding="ascii")
    assert contract.read_bounded_phase6_trace_json(artifact) == {"ok": True}
    link = tmp_path / "trace-link.json"
    link.symlink_to(artifact)
    with pytest.raises(contract.ContractError, match="regular-file"):
        contract.read_bounded_phase6_trace_json(link)
    monkeypatch.setattr(contract, "PHASE6_TRACE_MAX_JSON_BYTES", 2)
    with pytest.raises(contract.ContractError, match="size cap"):
        contract.read_bounded_phase6_trace_json(artifact)


def _graphdef_cohort(*, changed_const: tuple[int, int] | None = None):
    from tensorflow.core.framework import graph_pb2, types_pb2

    records = []
    for parameter_count in (50, 150):
        for batch_size in (1, 4, 16):
            graph = graph_pb2.GraphDef()
            placeholder = graph.node.add(name="parameters_batch", op="Placeholder")
            placeholder.attr["dtype"].type = types_pb2.DT_FLOAT
            placeholder.attr["shape"].shape.dim.add().size = batch_size
            placeholder.attr["shape"].shape.dim.add().size = parameter_count
            constant = graph.node.add(name="fixed_constant", op="Const")
            constant.attr["dtype"].type = types_pb2.DT_INT32
            tensor = constant.attr["value"].tensor
            tensor.dtype = types_pb2.DT_INT32
            tensor.int_val.append(
                8 if changed_const == (parameter_count, batch_size) else 7
            )
            raw = graph.SerializeToString(deterministic=True)
            records.append(
                {
                    "identity": contract.phase6_identity(
                        dimension=10,
                        parameter_count=parameter_count,
                        batch_size=batch_size,
                        dtype="float32",
                        method_id=contract.PRIMARY_METHOD_IDS[0],
                        operation="trace",
                    ),
                    "graphdef_bytes": contract.graphdef_bytes_record(raw),
                }
            )
    return records


def test_graphdef_cohort_accepts_only_declared_shape_axes() -> None:
    result = contract.compare_graphdef_cohort(_graphdef_cohort())
    assert result["passed"] is True
    assert result["normalized_graphdefs_equal"] is True
    assert not result["rejected_differences"]
    assert {row["axis"] for row in result["accepted_differences"]} == {"B", "P"}
    assert all(row["rule_id"] in {"static_shape_dimension_B", "static_shape_dimension_P"} for row in result["accepted_differences"])


def test_graphdef_cohort_rejects_const_or_roster_differences() -> None:
    result = contract.compare_graphdef_cohort(
        _graphdef_cohort(changed_const=(150, 16))
    )
    assert result["passed"] is False
    assert any(row["inside_const"] for row in result["rejected_differences"])
    with pytest.raises(contract.ContractError, match="exactly six"):
        contract.compare_graphdef_cohort(_graphdef_cohort()[:-1])
    wrong = _graphdef_cohort()
    wrong[0]["identity"] = contract.phase6_identity(
        dimension=20,
        parameter_count=50,
        batch_size=1,
        dtype="float32",
        method_id=contract.PRIMARY_METHOD_IDS[0],
        operation="trace",
    )
    with pytest.raises(contract.ContractError, match="must fix dimension"):
        contract.compare_graphdef_cohort(wrong)


def _gate_b_command() -> dict[str, Any]:
    argv = [
        str(Path(sys.executable).resolve()),
        contract.PHASE6_SUPERVISOR_RELATIVE,
        "--phase6-pilot",
        "--dimensions",
        "10",
        "20",
        "30",
        "--parameter-counts",
        "50",
        "150",
        "--batch-sizes",
        "1",
        "4",
        "16",
        "--timesteps",
        "120",
        "--dtype",
        "float32",
        "--device",
        "cpu",
        "--cpu-threads",
        "1",
        "--jit-compile",
        "--trace-child-timeout-seconds",
        "60",
        "--xla-child-timeout-seconds",
        "60",
        "--xla-cell-timeout-seconds",
        "160",
        "--budget-contract",
        contract.PHASE6_GATE_B_BUDGET_RELATIVE,
        "--budget-attestation",
        contract.PHASE6_GATE_B_ATTESTATION_RELATIVE,
        "--trace-output-json",
        contract.PHASE6_GATE_B_ARTIFACTS["trace_output_json"],
        "--output-json",
        contract.PHASE6_GATE_B_ARTIFACTS["pilot_output_json"],
    ]
    return {
        "name": "trace_census_and_pilot",
        "argv": argv,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
        },
        "term_deadline_seconds": 3000,
        "kill_grace_seconds": 45,
    }


def _synthetic_gate_b_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, str]]:
    paths = [tmp_path / f"gate-b-input-{index}.txt" for index in range(4)]
    for index, path in enumerate(paths):
        path.write_text(f"gate-b-input-{index}\n", encoding="ascii")
    records = [contract.path_digest_record(path) for path in paths]
    monkeypatch.setattr(
        contract,
        "phase6_gate_b_input_records",
        lambda **kwargs: copy.deepcopy(records),
    )
    monkeypatch.setattr(
        contract,
        "phase6_gate_b_inputs_valid",
        lambda value: value == records,
    )
    return records


def _valid_gate_b_proposal(tmp_path: Path, monkeypatch) -> dict[str, Any]:
    runner = _load_runner("kalman_qr_phase6_gate_b_budget_schedules")
    inputs = _synthetic_gate_b_inputs(tmp_path, monkeypatch)
    opening = Path(contract.PHASE6_OPENING_HASH_LEDGER)
    plan_path = ROOT / contract.PHASE6_PLAN_RELATIVE
    source_hashes = [
        contract.path_digest_record(ROOT / relative)
        for relative in contract.PHASE6_REQUIRED_SOURCE_PATHS
    ]
    entries = [
        {
            "module": f"phase6.source_{index}",
            "path": relative,
            "sha256": contract.file_sha256(ROOT / relative),
        }
        for index, relative in enumerate(contract.PHASE6_REQUIRED_SOURCE_PATHS)
    ]
    manifest = _manifest(entries)
    proposal = {
        "schema": contract.PHASE6_BUDGET_SCHEMA,
        "authority_id": "0" * 64,
        "gate": "gate_b",
        "plan": contract.path_digest_record(plan_path),
        "opening_hash_ledger": contract.phase6_opening_hash_ledger_record(opening),
        "dependency_discovery": {
            "schema": "bayesfilter.kalman_qr_batched_xla_repair.phase6.import_discovery.v1",
            "kind": "import_only_no_fixture_trace_or_execution",
            "command_argv": list(contract.PHASE6_IMPORT_DISCOVERY_ARGV),
            "environment": dict(contract.PHASE6_ENVIRONMENT),
            "fixture_constructed": False,
            "trace_requested": False,
            "selected_method_constructed": False,
            "concrete_function_invocations": 0,
            "manifest": manifest,
            "nonclaims": list(contract.PHASE6_NONCLAIMS),
        },
        "source_hashes": source_hashes,
        "commands": [_gate_b_command()],
        "schedules": {
            schema: runner.phase6_build_schedule(
                schema, gate="gate_b", child_timeout_seconds=60
            )
            for schema in (
                contract.PHASE6_TRACE_SCHEMA,
                contract.PHASE6_PILOT_SCHEMA,
            )
        },
        "artifacts": dict(contract.PHASE6_GATE_B_ARTIFACTS),
        "budget": {
            "child_execution_deadline_seconds": 60,
            "child_term_grace_seconds": 5,
            "child_kill_reap_grace_seconds": 5,
            "child_lifecycle_cap_seconds": 70,
            "cell_cap_seconds": 160,
            "outer_term_deadline_seconds": 3000,
            "outer_kill_grace_seconds": 45,
            "hard_ceiling_seconds": 3045,
        },
        "inputs": inputs,
        "nonclaims": list(contract.PHASE6_NONCLAIMS),
    }
    identity = {
        key: proposal[key]
        for key in contract.PHASE6_BUDGET_FIELDS
        if key != "authority_id"
    }
    proposal["authority_id"] = contract.canonical_sha256(identity)
    return proposal


def _valid_import_discovery_payload() -> dict[str, Any]:
    entries = [
        {
            "module": f"phase6.proposal_source_{index}",
            "path": relative,
            "sha256": contract.file_sha256(ROOT / relative),
        }
        for index, relative in enumerate(contract.PHASE6_REQUIRED_SOURCE_PATHS)
    ]
    return {
        "schema": (
            "bayesfilter.kalman_qr_batched_xla_repair.phase6.import_discovery.v1"
        ),
        "kind": "import_only_no_fixture_trace_or_execution",
        "command_argv": list(contract.PHASE6_IMPORT_DISCOVERY_ARGV),
        "environment": dict(contract.PHASE6_ENVIRONMENT),
        "fixture_constructed": False,
        "trace_requested": False,
        "selected_method_constructed": False,
        "concrete_function_invocations": 0,
        "manifest": _manifest(entries),
        "nonclaims": list(contract.PHASE6_NONCLAIMS),
    }


def test_construction_only_gate_b_proposal_builder_is_closed_and_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner("kalman_qr_phase6_proposal_builder")
    inputs = _synthetic_gate_b_inputs(tmp_path, monkeypatch)
    proposal = runner.phase6_build_budget_proposal(
        "gate_b", _valid_import_discovery_payload()
    )
    contract.validate_phase6_budget_proposal(proposal, expected_gate="gate_b")
    assert proposal["commands"] == [runner._phase6_gate_b_command()]
    assert proposal["inputs"] == inputs
    assert set(proposal["schedules"]) == {
        contract.PHASE6_TRACE_SCHEMA,
        contract.PHASE6_PILOT_SCHEMA,
    }
    assert proposal["budget"]["hard_ceiling_seconds"] == 3045
    assert proposal["opening_hash_ledger"]["sha256"] == (
        contract.PHASE6_OPENING_HASH_LEDGER_SHA256
    )


def test_prepare_proposal_requires_exact_cli_and_refuses_changed_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("kalman_qr_phase6_prepare_proposal_cli")
    _synthetic_gate_b_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner, "_phase6_assert_gate_b_work_root_absent", lambda: None
    )
    relative = contract.PHASE6_GATE_B_BUDGET_RELATIVE
    output = tmp_path / "proposal.json"
    discovery_calls = 0

    def discovery() -> dict[str, Any]:
        nonlocal discovery_calls
        discovery_calls += 1
        return _valid_import_discovery_payload()

    monkeypatch.setattr(runner, "_phase6_run_import_discovery", discovery)
    monkeypatch.setattr(runner, "_phase6_repo_path", lambda path: output)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--phase6-prepare-proposal",
            "gate_b",
            "--output-json",
            relative,
        ],
    )
    args = SimpleNamespace(
        phase6_prepare_proposal="gate_b",
        output_json=Path(relative),
    )
    assert runner.run_phase6_prepare_proposal(args) == 0
    proposal = contract.read_strict_json(output)
    contract.validate_phase6_budget_proposal(proposal, expected_gate="gate_b")
    assert discovery_calls == 1
    assert not (tmp_path / "attestation.json").exists()

    before_bytes = output.read_bytes()
    with pytest.raises(contract.ContractError, match="strictly absent"):
        runner.run_phase6_prepare_proposal(args)
    assert discovery_calls == 1
    assert output.read_bytes() == before_bytes

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--phase6-prepare-proposal",
            "gate_b",
            "--output-json",
            relative,
            "--phase6-pilot",
        ],
    )
    before = discovery_calls
    with pytest.raises(contract.ContractError, match="exact closed invocation"):
        runner.run_phase6_prepare_proposal(args)
    assert discovery_calls == before


def test_budget_proposal_and_detached_review_are_exactly_bound(
    tmp_path: Path, monkeypatch
) -> None:
    proposal = _valid_gate_b_proposal(tmp_path, monkeypatch)
    assert all(
        contract.phase6_budget_proposal_checks(
            proposal, expected_gate="gate_b"
        ).values()
    )
    contract.validate_phase6_budget_proposal(proposal, expected_gate="gate_b")

    proposal_path = tmp_path / "proposal.json"
    contract.durable_atomic_write_json(proposal_path, proposal)
    review_path = tmp_path / "review.md"
    review_path.write_text(
        "\n".join(
            [
                "Review strength: `codex_substitute_weaker`",
                f"PROPOSAL_PATH: {proposal_path.resolve()}",
                f"PROPOSAL_SHA256: {contract.file_sha256(proposal_path)}",
                f"PLAN_PATH: {proposal['plan']['path']}",
                f"PLAN_SHA256: {proposal['plan']['sha256']}",
                "VERDICT: AGREE",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    attestation = {
        "schema": contract.PHASE6_ATTESTATION_SCHEMA,
        "authority_id": proposal["authority_id"],
        "gate": "gate_b",
        "proposal": contract.path_digest_record(proposal_path),
        "plan": proposal["plan"],
        "review": contract.path_digest_record(review_path),
        "verdict": "AGREE",
        "review_strength": "codex_substitute_weaker",
        "timestamp_utc": "2026-07-11T00:00:00+00:00",
    }
    assert all(
        contract.phase6_attestation_checks(
            attestation, proposal_path=proposal_path, expected_gate="gate_b"
        ).values()
    )
    attestation_path = tmp_path / "attestation.json"
    contract.durable_atomic_write_json(attestation_path, attestation)
    loaded_proposal, loaded_attestation = contract.validate_phase6_runtime_authority(
        proposal_path, attestation_path, expected_gate="gate_b"
    )
    assert loaded_proposal == proposal
    assert loaded_attestation == attestation

    wrong_budget = copy.deepcopy(proposal)
    wrong_budget["budget"]["cell_cap_seconds"] = 161
    assert not all(
        contract.phase6_budget_proposal_checks(
            wrong_budget, expected_gate="gate_b"
        ).values()
    )
    wrong_command = copy.deepcopy(proposal)
    wrong_command["commands"][0]["argv"].extend(["--unexpected", "1"])
    assert contract.phase6_budget_proposal_checks(
        wrong_command, expected_gate="gate_b"
    )["command_argv_identity"] is False
    missing_schedule = copy.deepcopy(proposal)
    missing_schedule["schedules"].pop(contract.PHASE6_TRACE_SCHEMA)
    assert contract.phase6_budget_proposal_checks(
        missing_schedule, expected_gate="gate_b"
    )["schedules_closed"] is False

    review_path.write_text(
        "Review strength: `codex_substitute_weaker`\n"
        f"PROPOSAL_PATH: {proposal_path.resolve()}\n"
        "PROPOSAL_SHA256: " + "0" * 64 + "\n"
        f"PLAN_PATH: {proposal['plan']['path']}\n"
        f"PLAN_SHA256: {proposal['plan']['sha256']}\n"
        "VERDICT: AGREE\n",
        encoding="ascii",
    )
    wrong_review = copy.deepcopy(attestation)
    wrong_review["review"] = contract.path_digest_record(review_path)
    checks = contract.phase6_attestation_checks(
        wrong_review, proposal_path=proposal_path, expected_gate="gate_b"
    )
    assert checks["review_digest"] is True
    assert checks["verdict_agree"] is False


def test_directed_float32_comparison_locks_shape_tolerance_and_direction() -> None:
    passed = contract.directed_float32_comparison(
        [1.0001], [1.0], expected_shape=[1], output_kind="value"
    )
    assert passed["passed"] is True
    assert passed["rtol"] == pytest.approx(2.0e-4)
    assert passed["atol"] == pytest.approx(2.0e-4)
    assert contract.directed_float32_comparison(
        [1.0, 2.0], [1.0], expected_shape=[1], output_kind="value"
    )["passed"] is False
    assert contract.directed_float32_comparison(
        [1_000_200.04],
        [1_000_000.0],
        expected_shape=[1],
        output_kind="value",
    )["passed"] is False
    assert contract.directed_float32_comparison(
        [1_000_000.0],
        [1_000_200.04],
        expected_shape=[1],
        output_kind="value",
    )["passed"] is True
    with pytest.raises(contract.ContractError, match="non-finite"):
        contract.directed_float32_comparison(
            [float("nan")], [0.0], expected_shape=[1], output_kind="value"
        )


def _minimal_child_outputs(batch_size: int) -> dict[str, Any]:
    return {
        "outputs": {
            "value": [0.0] * batch_size,
            "score": [[0.0] * 50 for _ in range(batch_size)],
        }
    }


def test_scalar_status_cannot_pass_from_invalid_or_unbound_ledgers() -> None:
    scalar_records = []
    final_records = []
    for batch_size in (1, 4):
        for method in contract.REFERENCE_METHOD_IDS:
            scalar_records.append(
                {
                    "identity": {
                        "batch_size": batch_size,
                        "method_id": method,
                    },
                    "state": "passed",
                    "evidence": {
                        "child_artifact": {
                            "strict_json": _minimal_child_outputs(batch_size)
                        }
                    },
                }
            )
        for method in contract.PRIMARY_METHOD_IDS:
            final_records.append(
                {
                    "identity": {
                        "dimension": 10,
                        "parameter_count": 50,
                        "batch_size": batch_size,
                        "method_id": method,
                    },
                    "state": "passed",
                    "evidence": {
                        "child_artifact": {
                            "strict_json": _minimal_child_outputs(batch_size)
                        }
                    },
                }
            )
    result = contract.evaluate_phase6_scalar_status(
        {"records": scalar_records}, {"records": final_records}
    )
    assert result["target_scalar_status"] == "failed_common_or_cpu_xla_backend_unlocalized"
    assert not all(result["scalar_ledger_checks"].values())
    assert not all(result["final_ledger_checks"].values())


def _handoff_inputs(**updates: Any) -> dict[str, Any]:
    payload = {
        "phase45_common_correctness_valid": True,
        "dependency_provenance_valid": True,
        "trace_common_valid": True,
        "cpu_xla_common_invalidity": False,
        "target_scalar_status": "passed",
        "cpu_xla_lane_local_only": False,
        "fair_pair_cells": [
            {
                "dimension": 10,
                "parameter_count": 50,
                "batch_size": 4,
                "dtype": "float32",
                "completed_scalar_comparisons": True,
            },
            {
                "dimension": 10,
                "parameter_count": 50,
                "batch_size": 1,
                "dtype": "float32",
                "completed_scalar_comparisons": True,
            },
        ],
        "valid_trace_cohorts": [
            {
                "dimension": 10,
                "parameter_count": 50,
                "batch_size": 1,
                "dtype": "float32",
            }
        ],
    }
    payload.update(updates)
    return payload


def test_handoff_is_total_nonexpanding_and_lexicographically_deterministic() -> None:
    handoff = contract._phase6_handoff_from_derived(_handoff_inputs())
    assert handoff == {
        "schema": contract.PHASE6_HANDOFF_SCHEMA,
        "phase7_scope": "target_numerical_gate",
        "selected_phase7_cell": {
            "dimension": 10,
            "parameter_count": 50,
            "batch_size": 1,
            "dtype": "float32",
        },
        "phase7_expansion_authorized": False,
        "phase7_nonclaims": list(contract.PHASE7_NONCLAIMS),
    }

    diagnostic = contract._phase6_handoff_from_derived(
        _handoff_inputs(
            target_scalar_status="partial_missing_evidence",
            cpu_xla_lane_local_only=True,
            fair_pair_cells=[],
            valid_trace_cohorts=[
                {
                    "dimension": 20,
                    "parameter_count": 150,
                    "batch_size": 4,
                    "dtype": "float32",
                },
                {
                    "dimension": 10,
                    "parameter_count": 150,
                    "batch_size": 1,
                    "dtype": "float32",
                },
            ],
        )
    )
    assert diagnostic["phase7_scope"] == "diagnostic_smallest_gpu_only"
    assert diagnostic["selected_phase7_cell"] == {
        "dimension": 10,
        "parameter_count": 150,
        "batch_size": 1,
        "dtype": "float32",
    }
    assert diagnostic["phase7_expansion_authorized"] is False


@pytest.mark.parametrize(
    "updates",
    [
        {"phase45_common_correctness_valid": False},
        {"dependency_provenance_valid": False},
        {"trace_common_valid": False},
        {"cpu_xla_common_invalidity": True},
        {"target_scalar_status": "failed_scalar_reference_disagreement_unlocalized"},
        {"target_scalar_status": f"failed_method_local:{contract.PRIMARY_METHOD_IDS[0]}"},
        {"fair_pair_cells": [], "valid_trace_cohorts": []},
    ],
)
def test_handoff_blockers_precede_any_promotion(updates: Mapping[str, Any]) -> None:
    handoff = contract._phase6_handoff_from_derived(_handoff_inputs(**updates))
    assert handoff["phase7_scope"] == "blocked"
    assert handoff["selected_phase7_cell"] is None
    assert handoff["phase7_expansion_authorized"] is False


def test_handoff_rejects_duplicate_cohorts_and_extra_derived_fields() -> None:
    duplicate = _handoff_inputs()
    duplicate["valid_trace_cohorts"] *= 2
    assert contract._phase6_handoff_from_derived(duplicate)["phase7_scope"] == "blocked"
    extra = _handoff_inputs(unreviewed_override=True)
    assert contract._phase6_handoff_from_derived(extra)["phase7_scope"] == "blocked"


def test_phase6_cli_modes_are_mutually_exclusive_and_exposed(monkeypatch) -> None:
    runner = _load_runner("kalman_qr_phase6_cli_contract")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(RUNNER_PATH), "--phase6-pilot", "--phase6-remaining"],
    )
    with pytest.raises(SystemExit):
        runner.parse_args()
    monkeypatch.setattr(sys, "argv", [str(RUNNER_PATH), "--phase6-pilot"])
    args = runner.parse_args()
    assert args.phase6_pilot is True
    assert args.phase6_scalar_references is False
    assert args.phase6_remaining is False
    assert args.phase6_evaluate is False


@pytest.mark.parametrize(
    ("schema", "gate", "expected_operation"),
    [
        (contract.PHASE6_TRACE_SCHEMA, "gate_b", "trace"),
        (contract.PHASE6_PILOT_SCHEMA, "gate_b", "xla"),
        (contract.PHASE6_SCALAR_SCHEMA, "gate_c", "scalar_reference"),
        (contract.PHASE6_FINAL_SCHEMA, "gate_c", "xla"),
    ],
)
def test_supervisor_schedule_is_exactly_rostered_and_method_isolated(
    schema: str, gate: str, expected_operation: str
) -> None:
    runner = _load_runner(f"kalman_qr_phase6_schedule_{expected_operation}")
    schedule = runner.phase6_build_schedule(
        schema, gate=gate, child_timeout_seconds=60
    )
    roster = contract.phase6_expected_roster(schema)
    assert [row["identity"] for row in schedule["records"]] == roster
    assert schedule["schema"] == contract.PHASE6_SCHEDULE_SCHEMA
    assert schedule["ledger_schema"] == schema
    assert schedule["gate"] == gate
    core = {
        "schema": schedule["schema"],
        "ledger_schema": schedule["ledger_schema"],
        "gate": schedule["gate"],
        "records": schedule["records"],
    }
    assert schedule["schedule_sha256"] == contract.canonical_sha256(core)
    assert all(contract.phase6_schedule_checks(schedule).values())
    for identity, row in zip(roster, schedule["records"], strict=True):
        assert identity["operation"] == expected_operation
        assert row["config"]["timesteps"] == 120
        assert row["config"]["device"] == "cpu"
        assert row["config"]["cpu_threads"] == 1
        assert row["config"]["jit_compile"] is (expected_operation == "xla")
        command = row["child_command_argv"]
        assert command[command.index("--method") + 1] == identity["method_id"]
        assert all(
            sibling not in command
            for sibling in set(contract.METHOD_IDS) - {identity["method_id"]}
        )


@pytest.mark.parametrize(
    "mutation",
    ["config_fingerprint", "schedule_fingerprint", "resume_key", "gate", "roster_order"],
)
def test_schedule_evaluator_rejects_coherently_rehashed_semantic_drift(
    mutation: str,
) -> None:
    runner = _load_runner(f"kalman_qr_phase6_schedule_drift_{mutation}")
    schedule = runner.phase6_build_schedule(
        contract.PHASE6_PILOT_SCHEMA,
        gate="gate_b",
        child_timeout_seconds=60,
    )
    if mutation == "config_fingerprint":
        schedule["records"][0]["fingerprints"]["config_fingerprint"] = "0" * 64
    elif mutation == "schedule_fingerprint":
        schedule["records"][0]["fingerprints"]["schedule_fingerprint"] = "0" * 64
    elif mutation == "resume_key":
        schedule["records"][0]["resume_key"] = "0" * 64
    elif mutation == "gate":
        schedule["gate"] = "gate_c"
    elif mutation == "roster_order":
        schedule["records"].reverse()
    else:  # pragma: no cover - parameter list is closed above.
        raise AssertionError(mutation)
    core = {
        key: schedule[key]
        for key in ("schema", "ledger_schema", "gate", "records")
    }
    schedule["schedule_sha256"] = contract.canonical_sha256(core)
    assert not all(contract.phase6_schedule_checks(schedule).values())


def test_supervisor_binding_builder_preserves_exact_artifact_bytes(
    tmp_path: Path,
) -> None:
    runner = _load_runner("kalman_qr_phase6_binding_builder")
    identities = contract.phase6_expected_roster(contract.PHASE6_PILOT_SCHEMA)
    schedule = _schedule(
        identities, gate="gate_b", ledger_schema=contract.PHASE6_PILOT_SCHEMA
    )["payload"]
    authority_id = "a" * 64
    proposal_path = tmp_path / "proposal.json"
    attestation_path = tmp_path / "attestation.json"
    phase45_path = tmp_path / "phase45.json"
    contract.durable_atomic_write_json(
        proposal_path, {"authority_id": authority_id, "gate": "gate_b"}
    )
    contract.durable_atomic_write_json(
        attestation_path, {"authority_id": authority_id, "gate": "gate_b"}
    )
    contract.durable_atomic_write_json(phase45_path, {"state": "passed"})
    bindings = runner.phase6_build_bindings(
        proposal_path=proposal_path,
        attestation_path=attestation_path,
        schedule=schedule,
        phase45_paths=[phase45_path],
    )
    assert bindings["authority_id"] == authority_id
    assert bindings["schedule"]["payload"] == schedule
    assert bindings["schedule"]["sha256"] == contract.canonical_sha256(schedule)
    assert contract.phase6_blob_record_valid(bindings["proposal"])
    assert contract.phase6_blob_record_valid(bindings["attestation"])
    assert contract.phase6_blob_record_valid(bindings["phase45_evidence"][0])
    proposal_path.write_text('{"authority_id":"stale"}\n', encoding="ascii")
    assert bindings["proposal"]["strict_json"]["authority_id"] == authority_id
    assert not contract.path_digest_record_matches(
        {
            "path": bindings["proposal"]["path"],
            "sha256": bindings["proposal"]["sha256"],
        }
    )


@pytest.mark.parametrize(
    ("state", "reason", "classification", "returncode", "timed_out"),
    [
        (
            "passed",
            "child_passed",
            "method_pass",
            0,
            False,
        ),
        (
            "failed",
            "child_nonzero_exit",
            "method_local_failure",
            1,
            False,
        ),
        (
            "timed_out",
            "child_execution_deadline_exceeded",
            "cpu_backend_or_cell_timeout",
            -15,
            True,
        ),
        (
            "crashed",
            "child_signal_exit",
            "cpu_backend_or_method_failure",
            -9,
            False,
        ),
    ],
)
def test_imported_pilot_is_one_direct_exact_terminal_transition_without_launch(
    tmp_path: Path,
    monkeypatch,
    state: str,
    reason: str,
    classification: str,
    returncode: int,
    timed_out: bool,
) -> None:
    runner = _load_runner(f"kalman_qr_phase6_direct_import_{state}")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    roster = contract.phase6_expected_roster(contract.PHASE6_FINAL_SCHEMA)
    pilot_roster = contract.phase6_expected_roster(contract.PHASE6_PILOT_SCHEMA)
    imported_identity = pilot_roster[0]
    terminal_process = _terminal_process(
        ["/bin/true", imported_identity["identity_id"]],
        returncode=returncode,
        timed_out=timed_out,
    )
    imported_evidence = {"classification": classification}
    original_record = {
        "identity": imported_identity,
        "state": state,
        "reason": reason,
        "process": terminal_process,
        "evidence": imported_evidence,
        "imported_from": None,
    }
    second_record = {
        "identity": pilot_roster[1],
        "state": "not_launched:trace_gate_not_passed",
        "reason": "trace_gate_not_passed",
        "process": None,
        "evidence": None,
        "imported_from": None,
    }
    imported = {
        record["identity"]["identity_id"]: record
        for record in (original_record, second_record)
    }
    pilot_payload = {
        "schema": contract.PHASE6_PILOT_SCHEMA,
        "records": [original_record, second_record],
    }
    pilot_blob = _blob(
        pilot_payload,
        "/tmp/phase6-parametrized-pilot.json",
    )
    bindings = {
        "schedule": {
            "payload": {
                "records": [
                    {
                        "identity": identity,
                        "child_command_argv": ["/bin/true", identity["identity_id"]],
                    }
                    for identity in roster
                ]
            }
        },
        "proposal": {
            "strict_json": {"dependency_discovery": {"manifest": _manifest()}}
        },
        "authority_inputs": [pilot_blob],
        "runtime_predecessors": [],
    }
    initial = {
        "schema": contract.PHASE6_FINAL_SCHEMA,
        "records": [
            {"identity": identity, "state": "pending"} for identity in roster
        ],
    }
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(contract, "new_phase6_ledger", lambda **kwargs: initial)
    monkeypatch.setattr(
        runner,
        "phase6_persist_and_validate",
        lambda path, payload, final: payload,
    )

    def fake_transition(payload, **kwargs):
        calls.append(kwargs)
        for record in payload["records"]:
            if record["identity"]["identity_id"] == kwargs["identity_id"]:
                record["state"] = kwargs["new_state"]
                break
        return payload

    monkeypatch.setattr(contract, "transition_phase6_record", fake_transition)
    monkeypatch.setattr(
        contract,
        "finalize_phase6_ledger",
        lambda payload: (_ for _ in ()).throw(RuntimeError("stop before closure")),
    )
    original_ledger_checks = contract.phase6_ledger_checks
    monkeypatch.setattr(
        contract,
        "phase6_ledger_checks",
        lambda payload, final: (
            {"synthetic_exact_pilot_roster": True}
            if payload is pilot_payload
            else original_ledger_checks(payload, final=final)
        ),
    )
    monkeypatch.setattr(
        runner,
        "phase6_new_routing_ledger",
        lambda current_bindings: {
            "schema": contract.PHASE6_ROUTING_SCHEMA,
            "authority_id": "a" * 64,
            "state": "running",
            "update_index": 0,
            "records": [
                {
                    "identity": identity,
                    "state": "decided",
                    "reason": None,
                    "dependencies": {},
                    "prelaunch_snapshot": {},
                    "fingerprints": {},
                    "rule_id": "synthetic_direct_import",
                    "action": "eligible_under_gate_c_budget",
                }
                for identity in contract.phase6_expected_roster(
                    contract.PHASE6_ROUTING_SCHEMA
                )
            ],
        },
    )
    monkeypatch.setattr(
        runner,
        "phase6_persist_routing",
        lambda path, payload, final: payload,
    )
    monkeypatch.setattr(
        runner,
        "phase6_routing_decision",
        lambda payload, final_payload, identity: payload,
    )
    monkeypatch.setattr(
        runner,
        "_phase6_routing_checks",
        lambda payload, final: {"synthetic_direct_import": True},
    )
    monkeypatch.setattr(
        runner,
        "run_managed_process_group",
        lambda *args, **kwargs: pytest.fail("imported pilot must never relaunch"),
    )

    with pytest.raises(RuntimeError, match="stop before closure"):
        runner.phase6_execute_ledger(
            schema=contract.PHASE6_FINAL_SCHEMA,
            output_path=tmp_path / "unused.json",
            bindings=bindings,
            child_timeout_seconds=60,
            eligible_identity_ids=set(),
            imported_records=imported,
            routing_path=tmp_path / "routing.json",
        )
    imported_calls = [
        call
        for call in calls
        if call["identity_id"] == imported_identity["identity_id"]
    ]
    assert len(imported_calls) == 1
    terminal = imported_calls[0]
    assert terminal["new_state"] == state
    assert terminal["reason"] == reason
    assert terminal["process"] == terminal_process
    assert terminal["evidence"] is imported_evidence
    assert terminal["imported_from"] == {
        "kind": "gate_b_pilot",
        "pilot_artifact_sha256": pilot_blob["sha256"],
        "pilot_record_sha256": contract.canonical_sha256(original_record),
    }
    second_calls = [
        call
        for call in calls
        if call["identity_id"] == second_record["identity"]["identity_id"]
    ]
    assert len(second_calls) == 1
    assert second_calls[0]["new_state"] == "not_launched:trace_gate_not_passed"
    imported_ids = set(imported)
    assert all(
        call["new_state"] == "not_launched:not_in_gate_b_pilot"
        for call in calls
        if call["identity_id"] not in imported_ids
    )


def _pruning_payload() -> dict[str, Any]:
    return {
        "schema": contract.PHASE6_FINAL_SCHEMA,
        "records": [
            {"identity": identity, "state": "pending"}
            for identity in contract.phase6_expected_roster(contract.PHASE6_FINAL_SCHEMA)
        ],
    }


def _set_cell_state(
    payload: Mapping[str, Any],
    *,
    dimension: int,
    parameter_count: int,
    batch_size: int,
    method_id: str,
    state: str,
) -> None:
    for record in payload["records"]:
        identity = record["identity"]
        if (
            identity["dimension"],
            identity["parameter_count"],
            identity["batch_size"],
            identity["method_id"],
        ) == (dimension, parameter_count, batch_size, method_id):
            record["state"] = state
            return
    raise AssertionError("test cell missing from Phase 6 final roster")


def _cell_identity(
    *, dimension: int, parameter_count: int, batch_size: int, method_id: str
) -> dict[str, Any]:
    return contract.phase6_identity(
        dimension=dimension,
        parameter_count=parameter_count,
        batch_size=batch_size,
        dtype="float32",
        method_id=method_id,
        operation="xla",
    )


def test_phase6_pruning_is_method_local_and_dependency_order_is_deterministic() -> None:
    runner = _load_runner("kalman_qr_phase6_pruning_contract")
    analytical, autodiff = contract.PRIMARY_METHOD_IDS
    payload = _pruning_payload()
    _set_cell_state(
        payload,
        dimension=10,
        parameter_count=50,
        batch_size=1,
        method_id=analytical,
        state="failed",
    )
    _set_cell_state(
        payload,
        dimension=10,
        parameter_count=50,
        batch_size=1,
        method_id=autodiff,
        state="passed",
    )
    assert runner._phase6_should_prune(
        payload,
        _cell_identity(
            dimension=10,
            parameter_count=50,
            batch_size=4,
            method_id=analytical,
        ),
    ) == "after_smaller_batch_failure"
    assert (
        runner._phase6_should_prune(
            payload,
            _cell_identity(
                dimension=10,
                parameter_count=50,
                batch_size=4,
                method_id=autodiff,
            ),
        )
        is None
    )

    p150 = _pruning_payload()
    _set_cell_state(
        p150,
        dimension=10,
        parameter_count=50,
        batch_size=1,
        method_id=analytical,
        state="not_launched:after_smaller_batch_failure",
    )
    assert runner._phase6_should_prune(
        p150,
        _cell_identity(
            dimension=10,
            parameter_count=150,
            batch_size=1,
            method_id=analytical,
        ),
    ) == "p50_dependency_not_launched"
    _set_cell_state(
        p150,
        dimension=10,
        parameter_count=50,
        batch_size=1,
        method_id=analytical,
        state="failed",
    )
    assert runner._phase6_should_prune(
        p150,
        _cell_identity(
            dimension=10,
            parameter_count=150,
            batch_size=1,
            method_id=analytical,
        ),
    ) == "p50_dependency_failed"

    _set_cell_state(
        p150,
        dimension=10,
        parameter_count=150,
        batch_size=1,
        method_id=analytical,
        state="failed",
    )
    _set_cell_state(
        p150,
        dimension=10,
        parameter_count=50,
        batch_size=4,
        method_id=analytical,
        state="failed",
    )
    assert runner._phase6_should_prune(
        p150,
        _cell_identity(
            dimension=10,
            parameter_count=150,
            batch_size=4,
            method_id=analytical,
        ),
    ) == "after_smaller_p150_batch_failure"


def test_managed_process_group_reaps_a_harmless_descendant(tmp_path: Path) -> None:
    runner = _load_runner("kalman_qr_phase6_process_group")
    if not hasattr(runner, "run_managed_process_group"):
        pytest.fail("Gate A must expose run_managed_process_group")
    child_script = tmp_path / "tree.py"
    child_script.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "time.sleep(60)\n",
        encoding="ascii",
    )
    result = runner.run_managed_process_group(
        [sys.executable, str(child_script)],
        cwd=tmp_path,
        environment={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        deadline_seconds=0.2,
        term_grace_seconds=0.2,
        kill_reap_grace_seconds=1.0,
    )
    assert result["timed_out"] is True
    assert result["term_sent"] is True
    assert result["reaped"] is True
    assert result["process_group_gone"] is True
    assert contract.phase6_process_record_valid(result, terminal_state="timed_out")


@pytest.mark.parametrize("raised", [RuntimeError("callback failed"), KeyboardInterrupt()])
def test_managed_process_group_cleans_tree_when_started_callback_raises(
    tmp_path: Path, raised: BaseException
) -> None:
    runner = _load_runner(
        f"kalman_qr_phase6_callback_cleanup_{type(raised).__name__}"
    )
    child_script = tmp_path / "callback-tree.py"
    child_script.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "time.sleep(60)\n",
        encoding="ascii",
    )
    started: dict[str, Any] = {}

    def reject(identity: Mapping[str, Any]) -> None:
        started.update(identity)
        raise raised

    with pytest.raises(type(raised), match=None if isinstance(raised, KeyboardInterrupt) else "callback failed"):
        runner.run_managed_process_group(
            [sys.executable, str(child_script)],
            cwd=tmp_path,
            environment={"CUDA_VISIBLE_DEVICES": "-1"},
            deadline_seconds=5,
            term_grace_seconds=0.2,
            kill_reap_grace_seconds=1.0,
            on_started=reject,
        )
    assert set(started) == set(contract.PHASE6_RUNNING_PROCESS_FIELDS)
    assert started["process_start_ticks"] > 0
    assert not runner._phase6_process_group_exists(started["pgid"])
    assert not Path(f"/proc/{started['pid']}").exists()


def test_process_identity_uses_kernel_pgid_and_start_ticks() -> None:
    runner = _load_runner("kalman_qr_phase6_kernel_process_identity")
    pgid, start_ticks = runner._phase6_process_identity(os.getpid())
    assert pgid == os.getpgid(os.getpid())
    assert type(start_ticks) is int and start_ticks > 0
    with pytest.raises(contract.ContractError, match="cannot establish process identity"):
        runner._phase6_process_identity(2**31 - 1)


def test_recovery_of_dead_process_is_unavailable_but_interrupted_compatible() -> None:
    runner = _load_runner("kalman_qr_phase6_dead_recovery")
    process = _running_process(["/bin/true"])
    process.update(pid=2**31 - 1, pgid=2**31 - 1, process_start_ticks=1)
    recovered = runner._phase6_recover_running_process(process)
    assert recovered["term_sent"] is False
    assert recovered["kill_sent"] is False
    assert recovered["returncode"] is None
    assert recovered["reaped"] is False
    assert recovered["reap_status"] == "already_gone_not_waitable_after_recovery"
    assert recovered["stdout_total_bytes"] is None
    assert recovered["stderr_total_bytes"] is None
    assert recovered["stdout_capture_status"] == "unavailable_after_recovery"
    assert recovered["stderr_capture_status"] == "unavailable_after_recovery"
    assert contract.phase6_process_record_valid(
        recovered, terminal_state="interrupted"
    )


@pytest.mark.parametrize("ambiguity", ["start_ticks", "group_without_pid"])
def test_recovery_ambiguity_raises_without_signaling_current_group(
    monkeypatch, ambiguity: str
) -> None:
    runner = _load_runner(f"kalman_qr_phase6_recovery_ambiguity_{ambiguity}")
    current_pgid, current_ticks = runner._phase6_process_identity(os.getpid())
    process = _running_process(["/bin/true"])
    if ambiguity == "start_ticks":
        process.update(
            pid=os.getpid(),
            pgid=current_pgid,
            process_start_ticks=current_ticks + 1,
        )
        message = "identity is ambiguous"
    else:
        process.update(
            pid=2**31 - 1,
            pgid=current_pgid,
            process_start_ticks=1,
        )
        message = "group exists without its recorded PID"
        monkeypatch.setattr(runner, "_phase6_process_group_exists", lambda pgid: True)
    monkeypatch.setattr(
        runner,
        "_phase6_signal_group",
        lambda *args: pytest.fail("ambiguous recovery must not signal"),
    )
    with pytest.raises(contract.ContractError, match=message):
        runner._phase6_recover_running_process(process)
    assert Path(f"/proc/{os.getpid()}").exists()


def test_recovery_terminates_exact_orphaned_process_identity(tmp_path: Path) -> None:
    runner = _load_runner("kalman_qr_phase6_live_recovery")
    launcher = (
        "import subprocess, sys; "
        "p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL, start_new_session=True); print(p.pid)"
    )
    launched = subprocess.run(
        [sys.executable, "-c", launcher],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    pid = int(launched.stdout.strip())
    pgid, start_ticks = runner._phase6_process_identity(pid)
    process = _running_process([sys.executable, "-c", "import time; time.sleep(60)"])
    process.update(pid=pid, pgid=pgid, process_start_ticks=start_ticks)
    try:
        recovered = runner._phase6_recover_running_process(process)
        assert recovered["term_sent"] is True
        assert recovered["process_group_gone"] is True
        assert not runner._phase6_process_group_exists(pgid)
        assert contract.phase6_process_record_valid(
            recovered, terminal_state="interrupted"
        )
    finally:
        if runner._phase6_process_group_exists(pgid):
            os.killpg(pgid, signal.SIGKILL)


def test_managed_process_stream_cap_preserves_exact_tail_and_total(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner("kalman_qr_phase6_stream_cap")
    monkeypatch.setattr(contract, "PHASE6_STREAM_MAX_BYTES", 8)
    result = runner.run_managed_process_group(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'0123456789abcdef'); "
            "sys.stderr.buffer.write(b'ABCDEFGHIJKLMNOP')",
        ],
        cwd=tmp_path,
        environment={"CUDA_VISIBLE_DEVICES": "-1"},
        deadline_seconds=5,
        term_grace_seconds=0.2,
        kill_reap_grace_seconds=1.0,
    )
    assert result["returncode"] == 0
    assert result["stdout_total_bytes"] == 16
    assert result["stderr_total_bytes"] == 16
    assert result["stdout_capture_status"] == "truncated_at_cap"
    assert result["stderr_capture_status"] == "truncated_at_cap"
    assert result["stdout_base64"] == "ODlhYmNkZWY="
    assert result["stderr_base64"] == "SUpLTE1OT1A="
    assert result["stdout_tail"] == "89abcdef"
    assert result["stderr_tail"] == "IJKLMNOP"
    assert contract.phase6_process_record_valid(result)
    assert not contract.phase6_process_record_valid(result, terminal_state="passed")


def test_phase6_persist_reparses_the_exact_written_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner("kalman_qr_phase6_persist_contract")
    if not hasattr(runner, "phase6_persist_and_validate"):
        pytest.fail("Gate A must expose phase6_persist_and_validate")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    ledger = _new_ledger(contract.PHASE6_PILOT_SCHEMA)
    output = tmp_path / "ledger.json"
    reparsed = runner.phase6_persist_and_validate(output, ledger, final=False)
    assert reparsed == ledger
    assert contract.read_strict_json(output) == ledger

    closing = ledger
    for identity in closing["roster"]:
        closing = contract.transition_phase6_record(
            closing,
            identity_id=identity["identity_id"],
            new_state="not_launched:trace_gate_not_passed",
            timestamp_utc="2026-07-11T00:00:00+00:00",
            reason="trace_gate_not_passed",
        )
    closing = contract.finalize_phase6_ledger(closing)
    assert runner.phase6_persist_and_validate(output, closing, final=True) == closing
    assert contract.read_strict_json(output) == closing

    tampered = copy.deepcopy(ledger)
    tampered["records"].reverse()
    with pytest.raises(contract.ContractError):
        runner.phase6_persist_and_validate(output, tampered, final=False)
    assert contract.read_strict_json(output) == closing
