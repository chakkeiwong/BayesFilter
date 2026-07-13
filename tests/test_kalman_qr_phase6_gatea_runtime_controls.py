from __future__ import annotations

import copy
import base64
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
BUDGET_STATE_SCHEMA = (
    "bayesfilter.kalman_qr_batched_xla_repair.phase6.budget_state.v2"
)


def _load_runner(name: str):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _require_api(runner: Any, name: str) -> Any:
    implementation = getattr(runner, name, None)
    if implementation is None:
        pytest.fail(f"Gate A runtime control must expose {name}")
    return implementation


def _reviewed_schedule(runner: Any, schema: str, gate: str) -> dict[str, Any]:
    schedule = runner.phase6_build_schedule(
        schema,
        gate=gate,
        child_timeout_seconds=60,
    )
    assert all(contract.phase6_schedule_checks(schedule).values())
    return schedule


def _blob(payload: Any, path: str) -> dict[str, Any]:
    raw = contract.strict_json_dumps(payload).encode("utf-8")
    return {
        "path": path,
        "present": True,
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode("ascii"),
        "strict_json": payload,
    }


def _absent_blob(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "present": False,
        "byte_count": 0,
        "sha256": None,
        "base64": None,
        "strict_json": None,
    }


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


def _recovered_process(command: Sequence[str]) -> dict[str, Any]:
    empty_digest = hashlib.sha256(b"").hexdigest()
    return {
        **_running_process(command),
        "finished_ns": 20,
        "elapsed_seconds": 1.0e-8,
        "term_sent": False,
        "kill_sent": False,
        "reaped": False,
        "reap_status": "already_gone_not_waitable_after_recovery",
        "process_group_gone": True,
        "returncode": None,
        "timed_out": False,
        "stdout_bytes": 0,
        "stdout_total_bytes": None,
        "stdout_capture_status": "unavailable_after_recovery",
        "stdout_sha256": empty_digest,
        "stdout_base64": "",
        "stdout_tail": "",
        "stderr_bytes": 0,
        "stderr_total_bytes": None,
        "stderr_capture_status": "unavailable_after_recovery",
        "stderr_sha256": empty_digest,
        "stderr_base64": "",
        "stderr_tail": "",
    }


def _completed_process(
    command: Sequence[str], *, deadline_seconds: float
) -> dict[str, Any]:
    empty_digest = hashlib.sha256(b"").hexdigest()
    return {
        **_running_process(command),
        "deadline_seconds": deadline_seconds,
        "finished_ns": 20,
        "elapsed_seconds": 1.0e-8,
        "term_sent": False,
        "kill_sent": False,
        "reaped": True,
        "reap_status": "reaped_direct_child",
        "process_group_gone": True,
        "returncode": 0,
        "timed_out": False,
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


def _failed_process(command: Sequence[str]) -> dict[str, Any]:
    process = _completed_process(command, deadline_seconds=60.0)
    process["returncode"] = 1
    return process


def _interruption_evidence(identity: Mapping[str, Any]) -> dict[str, Any]:
    paths = contract._phase6_child_artifact_paths(identity)
    return {
        "classification": "supervisor_interruption",
        "child_artifact": _absent_blob(str(paths["artifact"].resolve())),
        "payload_sidecar": _absent_blob(str(paths["sidecar"].resolve())),
        "progress_journal": _absent_blob(str(paths["journal"].resolve())),
        "dependency_manifest_before_builder": None,
        "dependency_manifest_after_terminal": None,
        "dependency_coverage_before": False,
        "dependency_coverage_after": False,
    }


def _common_invalid_evidence(identity: Mapping[str, Any]) -> dict[str, Any]:
    paths = contract._phase6_child_artifact_paths(identity)
    return {
        "classification": "common_invalidity",
        "child_artifact": _absent_blob(str(paths["artifact"].resolve())),
        "payload_sidecar": _absent_blob(str(paths["sidecar"].resolve())),
        "progress_journal": _absent_blob(str(paths["journal"].resolve())),
        "dependency_manifest_before_builder": None,
        "dependency_manifest_after_terminal": None,
        "dependency_coverage_before": False,
        "dependency_coverage_after": False,
    }


_PREDECESSOR_CACHE: dict[str, dict[str, Any]] = {}


def _closed_interrupted_ledger(
    schema: str,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    cache_key = contract.canonical_sha256({"schema": schema, "bindings": bindings})
    cached = _PREDECESSOR_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)
    gate, artifact_kind, _ = contract.PHASE6_SCHEMA_CONTRACTS[schema]
    ledger = contract.new_phase6_ledger(
        schema=schema,
        gate=gate,
        artifact_kind=artifact_kind,
        identities=contract.phase6_expected_roster(schema),
        bindings=bindings,
    )
    for identity, row in zip(
        ledger["roster"], bindings["schedule"]["payload"]["records"], strict=True
    ):
        ledger = contract.transition_phase6_record(
            ledger,
            identity_id=identity["identity_id"],
            new_state="running",
            timestamp_utc="2026-07-11T00:00:00+00:00",
            process=_running_process(row["child_command_argv"]),
        )
        ledger = contract.transition_phase6_record(
            ledger,
            identity_id=identity["identity_id"],
            new_state="interrupted",
            timestamp_utc="2026-07-11T00:00:01+00:00",
            reason="supervisor_recovery",
            process=_recovered_process(row["child_command_argv"]),
            evidence=_interruption_evidence(identity),
        )
    ledger = contract.finalize_phase6_ledger(ledger)
    _PREDECESSOR_CACHE[cache_key] = copy.deepcopy(ledger)
    return ledger


def _minimal_bindings(
    runner: Any,
    schema: str,
    gate: str,
    *,
    authority_inputs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    schedule = _reviewed_schedule(runner, schema, gate)
    sibling_schemas = (
        (contract.PHASE6_TRACE_SCHEMA, contract.PHASE6_PILOT_SCHEMA)
        if gate == "gate_b"
        else (contract.PHASE6_SCALAR_SCHEMA, contract.PHASE6_FINAL_SCHEMA)
    )
    schedules = {
        sibling_schema: (
            copy.deepcopy(schedule)
            if sibling_schema == schema
            else _reviewed_schedule(runner, sibling_schema, gate)
        )
        for sibling_schema in sibling_schemas
    }
    authority_id = "a" * 64
    proposal = {
        "authority_id": authority_id,
        "gate": gate,
        "dependency_discovery": {"manifest": {}},
        "schedules": schedules,
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
    bindings = {
        "authority_id": authority_id,
        "proposal": _blob(proposal, "/tmp/proposal.json"),
        "attestation": _blob(
            {"authority_id": authority_id, "gate": gate},
            "/tmp/attestation.json",
        ),
        "schedule": {
            "payload": schedule,
            "sha256": contract.canonical_sha256(schedule),
        },
        "phase45_evidence": [
            _blob({"schema": "phase45-test-evidence"}, "/tmp/phase45.json")
        ],
        "authority_inputs": list(authority_inputs),
        "runtime_predecessors": [],
    }
    producer_schema = {
        contract.PHASE6_PILOT_SCHEMA: contract.PHASE6_TRACE_SCHEMA,
        contract.PHASE6_FINAL_SCHEMA: contract.PHASE6_SCALAR_SCHEMA,
    }.get(schema)
    if producer_schema is not None:
        producer_schedule = copy.deepcopy(schedules[producer_schema])
        producer_bindings = copy.deepcopy(bindings)
        producer_bindings["schedule"] = {
            "payload": producer_schedule,
            "sha256": contract.canonical_sha256(producer_schedule),
        }
        producer = _closed_interrupted_ledger(producer_schema, producer_bindings)
        artifact_key = (
            "trace_output_json"
            if producer_schema == contract.PHASE6_TRACE_SCHEMA
            else "scalar_output_json"
        )
        bindings["runtime_predecessors"] = [
            {
                "kind": "same_authority_runtime_predecessor",
                "producer_ledger_schema": producer_schema,
                "authority_id": authority_id,
                "schedule_sha256": producer_schedule["schedule_sha256"],
                "artifact": _blob(
                    producer,
                    str((ROOT / proposal["artifacts"][artifact_key]).resolve()),
                ),
            }
        ]
    return bindings


def _fake_persisted_ledger(
    schema: str,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    gate, artifact_kind, _ = contract.PHASE6_SCHEMA_CONTRACTS[schema]
    return {
        "schema": schema,
        "gate": gate,
        "artifact_kind": artifact_kind,
        "state": "running",
        "bindings": copy.deepcopy(dict(bindings)),
        "roster": contract.phase6_expected_roster(schema),
        "records": [
            {
                "identity": identity,
                "state": "pending",
                "reason": None,
                "process": None,
                "evidence": None,
                "imported_from": None,
            }
            for identity in contract.phase6_expected_roster(schema)
        ],
        "events": [],
        "update_index": 0,
        "aggregate": {},
        "nonclaims": list(contract.PHASE6_NONCLAIMS),
    }


def _terminal_not_launched_pilot(runner: Any) -> dict[str, Any]:
    bindings = _minimal_bindings(
        runner,
        contract.PHASE6_PILOT_SCHEMA,
        "gate_b",
    )
    ledger = _fake_persisted_ledger(contract.PHASE6_PILOT_SCHEMA, bindings)
    for identity in ledger["roster"]:
        ledger = contract.transition_phase6_record(
            ledger,
            identity_id=identity["identity_id"],
            new_state="not_launched:trace_gate_not_passed",
            timestamp_utc="2026-07-11T00:00:00+00:00",
            reason="trace_gate_not_passed",
        )
    return contract.finalize_phase6_ledger(ledger)


def _terminal_common_invalid_pilot(runner: Any) -> dict[str, Any]:
    bindings = _minimal_bindings(
        runner,
        contract.PHASE6_PILOT_SCHEMA,
        "gate_b",
    )
    ledger = contract.new_phase6_ledger(
        schema=contract.PHASE6_PILOT_SCHEMA,
        gate="gate_b",
        artifact_kind="cpu_xla_pilot",
        identities=contract.phase6_expected_roster(contract.PHASE6_PILOT_SCHEMA),
        bindings=bindings,
    )
    first, second = ledger["roster"]
    first_row = bindings["schedule"]["payload"]["records"][0]
    ledger = contract.transition_phase6_record(
        ledger,
        identity_id=first["identity_id"],
        new_state="running",
        timestamp_utc="2026-07-11T00:00:00+00:00",
        process=_running_process(first_row["child_command_argv"]),
    )
    ledger = contract.transition_phase6_record(
        ledger,
        identity_id=first["identity_id"],
        new_state="failed",
        timestamp_utc="2026-07-11T00:00:01+00:00",
        reason="invalid_child_evidence",
        process=_completed_process(
            first_row["child_command_argv"], deadline_seconds=60.0
        ),
        evidence=_common_invalid_evidence(first),
    )
    ledger = contract.transition_phase6_record(
        ledger,
        identity_id=second["identity_id"],
        new_state="not_launched:common_invalidity",
        timestamp_utc="2026-07-11T00:00:02+00:00",
        reason="common_invalidity",
    )
    final = contract.finalize_phase6_ledger(ledger)
    assert all(contract.phase6_ledger_checks(final, final=True).values())
    return final


def _imported_from(
    pilot: Mapping[str, Any],
    original: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "kind": "gate_b_pilot",
        "pilot_artifact_sha256": contract.canonical_sha256(pilot),
        "pilot_record_sha256": contract.canonical_sha256(original),
    }


def _find_schedule_row(
    schedule: Mapping[str, Any],
    *,
    dimension: int,
    parameter_count: int,
    batch_size: int,
    method_id: str,
) -> dict[str, Any]:
    return next(
        row
        for row in schedule["records"]
        if (
            row["identity"]["dimension"],
            row["identity"]["parameter_count"],
            row["identity"]["batch_size"],
            row["identity"]["method_id"],
        )
        == (dimension, parameter_count, batch_size, method_id)
    )


def test_schedule_rejects_coherently_rehashed_exact_child_argv_mutation() -> None:
    runner = _load_runner("phase6_runtime_controls_schedule")
    schedule = _reviewed_schedule(
        runner,
        contract.PHASE6_PILOT_SCHEMA,
        "gate_b",
    )
    mutated = copy.deepcopy(schedule)
    row = mutated["records"][0]
    option_index = row["child_command_argv"].index("--cpu-threads") + 1
    row["child_command_argv"][option_index] = "2"
    core = {
        key: mutated[key]
        for key in ("schema", "ledger_schema", "gate", "records")
    }
    mutated["schedule_sha256"] = contract.canonical_sha256(core)

    checks = contract.phase6_schedule_checks(mutated)
    assert checks["schedule_digest"] is True
    assert checks["row_semantics"] is False
    assert not all(checks.values())


def test_imported_pilot_not_launched_is_preserved_directly_with_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_import_not_launched")
    pilot = _terminal_not_launched_pilot(runner)
    pilot_blob = _blob(pilot, str(tmp_path / "pilot.json"))
    original = pilot["records"][0]
    identity = original["identity"]
    final_bindings = _minimal_bindings(
        runner,
        contract.PHASE6_FINAL_SCHEMA,
        "gate_c",
        authority_inputs=[pilot_blob],
    )
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    initial = _fake_persisted_ledger(contract.PHASE6_FINAL_SCHEMA, final_bindings)
    transition_calls: list[dict[str, Any]] = []
    original_transition = contract.transition_phase6_record

    def transition(payload: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        transition_calls.append(kwargs)
        return original_transition(payload, **kwargs)

    monkeypatch.setattr(contract, "new_phase6_ledger", lambda **kwargs: initial)
    monkeypatch.setattr(contract, "transition_phase6_record", transition)
    monkeypatch.setattr(
        contract,
        "finalize_phase6_ledger",
        lambda payload: (_ for _ in ()).throw(RuntimeError("stop before closure")),
    )
    monkeypatch.setattr(
        runner,
        "phase6_persist_and_validate",
        lambda path, payload, final: payload,
    )
    monkeypatch.setattr(
        runner,
        "run_managed_process_group",
        lambda *args, **kwargs: pytest.fail("an imported pilot record must not spawn"),
    )

    with pytest.raises(RuntimeError, match="stop before closure"):
        runner.phase6_execute_ledger(
            schema=contract.PHASE6_FINAL_SCHEMA,
            output_path=tmp_path / "final.json",
            bindings=final_bindings,
            child_timeout_seconds=60,
            eligible_identity_ids=set(),
            imported_records={
                record["identity"]["identity_id"]: record
                for record in pilot["records"]
            },
            routing_path=tmp_path / "routing.json",
        )

    imported = next(
        call for call in transition_calls if call["identity_id"] == identity["identity_id"]
    )
    assert imported == {
        "identity_id": identity["identity_id"],
        "new_state": "not_launched:trace_gate_not_passed",
        "timestamp_utc": imported["timestamp_utc"],
        "reason": "trace_gate_not_passed",
        "process": None,
        "evidence": None,
        "imported_from": _imported_from(pilot, original),
    }

    imported_record = copy.deepcopy(original)
    imported_record["imported_from"] = imported["imported_from"]
    assert contract.phase6_imported_pilot_record_valid(
        imported_record,
        bindings=final_bindings,
    )


def test_prelaunch_authority_revalidation_failure_is_durable_common_invalidity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_authority")
    _require_api(runner, "phase6_revalidate_launch_authority")
    bindings = _minimal_bindings(
        runner,
        contract.PHASE6_PILOT_SCHEMA,
        "gate_b",
    )
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "PHASE6_WORK_DIR", tmp_path / "work")
    initial = _fake_persisted_ledger(contract.PHASE6_PILOT_SCHEMA, bindings)
    authority_calls = 0

    def reject_authority(current_bindings: Mapping[str, Any]) -> None:
        nonlocal authority_calls
        authority_calls += 1
        assert current_bindings is bindings
        raise contract.ContractError("proposal changed after ledger creation")

    monkeypatch.setattr(contract, "new_phase6_ledger", lambda **kwargs: initial)
    monkeypatch.setattr(
        runner,
        "phase6_persist_and_validate",
        lambda path, payload, final: payload,
    )
    monkeypatch.setattr(runner, "phase6_revalidate_launch_authority", reject_authority)
    monkeypatch.setattr(
        runner,
        "run_managed_process_group",
        lambda *args, **kwargs: pytest.fail("authority drift must block spawn"),
    )

    result = runner.phase6_execute_ledger(
        schema=contract.PHASE6_PILOT_SCHEMA,
        output_path=tmp_path / "pilot.json",
        bindings=bindings,
        child_timeout_seconds=60,
        authority_validator=runner.phase6_revalidate_launch_authority,
    )
    assert authority_calls == 1
    assert result["state"] == "failed"
    assert all(
        record["state"] == "not_launched:common_invalidity"
        for record in result["records"]
    )
    assert all(contract.phase6_ledger_checks(result, final=True).values())


def test_shared_budget_state_persists_authority_start_elapsed_and_command_order(
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_budget_persistence")
    budget_open = _require_api(runner, "phase6_budget_state_open")
    budget_remaining = _require_api(runner, "phase6_budget_state_remaining")
    budget_close = _require_api(runner, "phase6_budget_state_close_command")
    path = tmp_path / "gate-c-budget-state.json"
    authority_id = "b" * 64

    with runner.phase6_budget_lease(path, "scalar_references") as scalar_lease:
        state = budget_open(
            path,
            authority_id,
            "gate_c",
            3120.0,
            "scalar_references",
            lease=scalar_lease,
            now_ns=1_000_000_000,
        )
        assert state == contract.read_strict_json(path)
        assert state["schema"] == BUDGET_STATE_SCHEMA
        assert state["authority_id"] == authority_id
        assert state["gate"] == "gate_c"
        assert state["hard_ceiling_seconds"] == 3120.0
        assert state["boot_id"] == runner._phase6_boot_id()
        assert state["started_ns"] == 1_000_000_000
        assert state["deadline_ns"] == 3_121_000_000_000
        assert state["last_observed_ns"] == 1_000_000_000
        assert state["elapsed_seconds"] == 0.0
        assert state["state"] == "running"
        assert state["update_index"] == 0
        assert state["commands"] == [
            {
                "name": "scalar_references",
                "started_ns": 1_000_000_000,
                "finished_ns": None,
                "elapsed_seconds": None,
                "state": "running",
            }
        ]
        assert budget_remaining(state, 11_000_000_000) == pytest.approx(3110.0)

        state = budget_close(
            path,
            state,
            "scalar_references",
            lease=scalar_lease,
            now_ns=11_000_000_000,
        )
    assert state == contract.read_strict_json(path)
    assert state["elapsed_seconds"] == pytest.approx(10.0)
    assert state["last_observed_ns"] == 11_000_000_000
    assert state["commands"][0] == {
        "name": "scalar_references",
        "started_ns": 1_000_000_000,
        "finished_ns": 11_000_000_000,
        "elapsed_seconds": pytest.approx(10.0),
        "state": "closed",
    }
    assert state["update_index"] == 1
    assert state["state"] == "running"

    with runner.phase6_budget_lease(path, "remaining_lattice") as remaining_lease:
        resumed = budget_open(
            path,
            authority_id,
            "gate_c",
            3120.0,
            "remaining_lattice",
            lease=remaining_lease,
            now_ns=21_000_000_000,
        )
        assert resumed["started_ns"] == 1_000_000_000
        assert resumed["deadline_ns"] == 3_121_000_000_000
        assert resumed["last_observed_ns"] == 21_000_000_000
        assert resumed["elapsed_seconds"] == pytest.approx(20.0)
        assert [row["name"] for row in resumed["commands"]] == [
            "scalar_references",
            "remaining_lattice",
        ]
        assert resumed["commands"][1]["started_ns"] == 21_000_000_000
        assert resumed["update_index"] == 2

        closed = budget_close(
            path,
            resumed,
            "remaining_lattice",
            lease=remaining_lease,
            now_ns=31_000_000_000,
        )
    assert closed["state"] == "closed"
    assert closed["elapsed_seconds"] == pytest.approx(30.0)
    assert closed["last_observed_ns"] == 31_000_000_000
    assert closed["update_index"] == 3
    assert closed == contract.read_strict_json(path)


def test_exhausted_shared_budget_prunes_without_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_budget_pruning")
    budget_open = _require_api(runner, "phase6_budget_state_open")
    bindings = _minimal_bindings(
        runner,
        contract.PHASE6_PILOT_SCHEMA,
        "gate_b",
    )
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    initial = _fake_persisted_ledger(contract.PHASE6_PILOT_SCHEMA, bindings)
    budget_path = tmp_path / "gate-b-budget-state.json"
    transitions: list[dict[str, Any]] = []

    def transition(payload: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        transitions.append(kwargs)
        updated = copy.deepcopy(dict(payload))
        record = next(
            row
            for row in updated["records"]
            if row["identity"]["identity_id"] == kwargs["identity_id"]
        )
        record["state"] = kwargs["new_state"]
        record["reason"] = kwargs.get("reason")
        return updated

    monkeypatch.setattr(contract, "new_phase6_ledger", lambda **kwargs: initial)
    monkeypatch.setattr(contract, "transition_phase6_record", transition)
    monkeypatch.setattr(contract, "finalize_phase6_ledger", lambda payload: payload)
    monkeypatch.setattr(
        runner,
        "phase6_persist_and_validate",
        lambda path, payload, final: payload,
    )
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: 3_000_000_000)
    monkeypatch.setattr(
        runner,
        "run_managed_process_group",
        lambda *args, **kwargs: pytest.fail("exhausted shared budget must not spawn"),
    )

    with runner.phase6_budget_lease(
        budget_path, "trace_census_and_pilot"
    ) as lease:
        budget_open(
            budget_path,
            bindings["authority_id"],
            "gate_b",
            1.0,
            "trace_census_and_pilot",
            lease=lease,
            now_ns=1_000_000_000,
        )
        runner.phase6_execute_ledger(
            schema=contract.PHASE6_PILOT_SCHEMA,
            output_path=tmp_path / "pilot.json",
            bindings=bindings,
            child_timeout_seconds=60,
            budget_path=budget_path,
            budget_lease=lease,
            expected_budget_command_name="trace_census_and_pilot",
            gate_hard_ceiling_seconds=1.0,
        )
    assert len(transitions) == len(initial["roster"])
    assert {
        (call["new_state"], call["reason"])
        for call in transitions
    } == {("not_launched:global_budget_exhausted", "global_budget_exhausted")}


def test_budget_lease_rejects_second_live_supervisor_handle(tmp_path: Path) -> None:
    runner = _load_runner("phase6_runtime_controls_budget_lease_exclusion")
    path = tmp_path / "budget.json"

    with runner.phase6_budget_lease(path, "scalar_references") as first:
        first.assert_current("scalar_references")
        with pytest.raises(contract.ContractError, match="live supervisor"):
            with runner.phase6_budget_lease(path, "scalar_references"):
                pytest.fail("second lease must not be acquired")

    with runner.phase6_budget_lease(path, "scalar_references") as resumed:
        assert resumed.record is not None
        assert resumed.record["generation"] == 2


def test_budget_lease_releases_lock_when_enter_fails_after_flock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_budget_lease_enter_failure")
    path = tmp_path / "budget.json"
    original_identity = runner._phase6_process_identity
    lease = runner.phase6_budget_lease(path, "scalar_references")
    monkeypatch.setattr(
        runner,
        "_phase6_process_identity",
        lambda pid: (_ for _ in ()).throw(contract.ContractError("identity failure")),
    )

    with pytest.raises(contract.ContractError, match="identity failure"):
        lease.__enter__()
    assert lease.fd is None
    assert lease.record is None

    monkeypatch.setattr(runner, "_phase6_process_identity", original_identity)
    with runner.phase6_budget_lease(path, "scalar_references") as recovered:
        recovered.assert_current("scalar_references")


def test_budget_resume_rejects_clock_rollback_boot_drift_and_wrong_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_budget_epoch")
    path = tmp_path / "budget.json"
    authority_id = "c" * 64

    with runner.phase6_budget_lease(path, "scalar_references") as scalar_lease:
        state = runner.phase6_budget_state_open(
            path,
            authority_id,
            "gate_c",
            3120.0,
            "scalar_references",
            lease=scalar_lease,
            now_ns=1_000_000_000,
        )
        state = runner.phase6_budget_state_checkpoint(
            path,
            scalar_lease,
            authority_id=authority_id,
            gate="gate_c",
            hard_ceiling_seconds=3120.0,
            command_name="scalar_references",
            now_ns=11_000_000_000,
        )
        with pytest.raises(contract.ContractError, match="checkpoint"):
            runner.phase6_budget_state_checkpoint(
                path,
                scalar_lease,
                authority_id=authority_id,
                gate="gate_c",
                hard_ceiling_seconds=3120.0,
                command_name="scalar_references",
                now_ns=10_000_000_000,
            )
        with pytest.raises(contract.ContractError, match="command mismatch"):
            scalar_lease.assert_current("remaining_lattice")
        state = runner.phase6_budget_state_close_command(
            path,
            state,
            "scalar_references",
            lease=scalar_lease,
            now_ns=12_000_000_000,
        )

    original_boot = runner._phase6_boot_id()
    monkeypatch.setattr(runner, "_phase6_boot_id", lambda: f"drift-{original_boot}")
    with runner.phase6_budget_lease(path, "remaining_lattice") as remaining_lease:
        with pytest.raises(contract.ContractError, match="authority or clock drift"):
            runner.phase6_budget_state_open(
                path,
                authority_id,
                "gate_c",
                3120.0,
                "remaining_lattice",
                lease=remaining_lease,
                now_ns=13_000_000_000,
            )


def test_two_method_cell_cap_uses_remaining_allowance_and_prunes_sibling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_cell_cap")
    bindings = _minimal_bindings(
        runner,
        contract.PHASE6_PILOT_SCHEMA,
        "gate_b",
    )
    monkeypatch.setattr(runner, "PHASE6_WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    initial = _fake_persisted_ledger(contract.PHASE6_PILOT_SCHEMA, bindings)
    transitions: list[dict[str, Any]] = []
    launch_deadlines: list[float] = []
    monotonic_values = iter(
        [
            1_000_000_000,
            161_000_000_000,
            161_000_000_000,
        ]
    )

    def transition(payload: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        transitions.append(kwargs)
        updated = copy.deepcopy(dict(payload))
        record = next(
            row
            for row in updated["records"]
            if row["identity"]["identity_id"] == kwargs["identity_id"]
        )
        record.update(
            state=kwargs["new_state"],
            reason=kwargs.get("reason"),
            process=kwargs.get("process"),
            evidence=kwargs.get("evidence"),
        )
        return updated

    def launch(
        command: Sequence[str],
        *,
        deadline_seconds: float,
        on_started: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        launch_deadlines.append(deadline_seconds)
        on_started(
            {
                "command_argv": list(command),
                "cwd": str(ROOT),
                "environment": {"CUDA_VISIBLE_DEVICES": "-1"},
                "pid": 1234,
                "pgid": 1234,
                "process_start_ticks": 1,
                "started_ns": 1_000_000_000,
                "deadline_seconds": deadline_seconds,
            }
        )
        return {
            "command_argv": list(command),
            "cwd": str(ROOT),
            "environment": {"CUDA_VISIBLE_DEVICES": "-1"},
            "pid": 1234,
            "pgid": 1234,
            "process_start_ticks": 1,
            "started_ns": 1_000_000_000,
            "finished_ns": 161_000_000_000,
            "elapsed_seconds": 160.0,
            "deadline_seconds": deadline_seconds,
            "term_sent": True,
            "kill_sent": False,
            "reaped": True,
            "reap_status": "reaped_direct_child",
            "process_group_gone": True,
            "returncode": -15,
            "timed_out": True,
            "stdout_bytes": 0,
            "stdout_total_bytes": 0,
            "stdout_capture_status": "complete",
            "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "stdout_base64": "",
            "stdout_tail": "",
            "stderr_bytes": 0,
            "stderr_total_bytes": 0,
            "stderr_capture_status": "complete",
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_base64": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(contract, "new_phase6_ledger", lambda **kwargs: initial)
    monkeypatch.setattr(contract, "transition_phase6_record", transition)
    monkeypatch.setattr(contract, "finalize_phase6_ledger", lambda payload: payload)
    monkeypatch.setattr(
        contract,
        "phase6_terminal_record_semantics_valid",
        lambda record, bindings: True,
    )
    monkeypatch.setattr(
        runner,
        "phase6_persist_and_validate",
        lambda path, payload, final: payload,
    )
    monkeypatch.setattr(
        runner,
        "_phase6_record_evidence",
        lambda identity, classification, discovery: {"classification": classification},
    )
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: next(monotonic_values))
    monkeypatch.setattr(runner, "run_managed_process_group", launch)

    runner.phase6_execute_ledger(
        schema=contract.PHASE6_PILOT_SCHEMA,
        output_path=tmp_path / "pilot.json",
        bindings=bindings,
        child_timeout_seconds=60,
        cell_cap_seconds=160.0,
    )

    assert launch_deadlines == [60.0]
    assert any(call["new_state"] == "timed_out" for call in transitions)
    sibling_prunes = [
        call
        for call in transitions
        if call["new_state"] == "not_launched:global_budget_exhausted"
    ]
    assert len(sibling_prunes) == 1
    assert sibling_prunes[0]["reason"] == "global_budget_exhausted"


def test_outer_sigterm_guard_cleans_once_raises_and_restores_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner("phase6_runtime_controls_sigterm")
    guard = _require_api(runner, "phase6_outer_sigterm_guard")
    termination = _require_api(runner, "Phase6OuterTermination")
    installed: dict[int, Any] = {signal.SIGTERM: object()}
    changes: list[tuple[int, Any]] = []
    callbacks: list[int] = []

    monkeypatch.setattr(signal, "getsignal", lambda signum: installed[signum])

    def set_handler(signum: int, handler: Any) -> Any:
        prior = installed[signum]
        installed[signum] = handler
        changes.append((signum, handler))
        return prior

    monkeypatch.setattr(signal, "signal", set_handler)
    previous = installed[signal.SIGTERM]

    with pytest.raises(termination):
        with guard(lambda signum: callbacks.append(signum)):
            handler = installed[signal.SIGTERM]
            assert callable(handler)
            handler(signal.SIGTERM, None)

    assert callbacks == [signal.SIGTERM]
    assert changes[0][0] == signal.SIGTERM
    assert callable(changes[0][1])
    assert changes[-1] == (signal.SIGTERM, previous)
    assert installed[signal.SIGTERM] is previous


def test_sigterm_during_popen_is_deferred_until_running_is_durable_and_cleans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_sigterm_popen_window")
    original_popen = runner.subprocess.Popen
    spawned: list[Any] = []
    durable_running: list[dict[str, Any]] = []

    def popen_and_signal(*args: Any, **kwargs: Any) -> Any:
        process = original_popen(*args, **kwargs)
        spawned.append(process)
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", popen_and_signal)

    with pytest.raises(runner.Phase6OuterTermination):
        with runner.phase6_outer_sigterm_guard(lambda signum: None) as termination:
            runner.run_managed_process_group(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                cwd=tmp_path,
                environment={"CUDA_VISIBLE_DEVICES": "-1"},
                deadline_seconds=5,
                term_grace_seconds=0.2,
                kill_reap_grace_seconds=1.0,
                on_started=lambda process: durable_running.append(dict(process)),
                termination=termination,
            )

    assert len(spawned) == 1
    assert len(durable_running) == 1
    process = durable_running[0]
    assert not runner._phase6_process_group_exists(process["pgid"])
    assert not Path(f"/proc/{process['pid']}").exists()


def test_process_identity_failure_after_popen_always_cleans_owned_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_identity_failure_cleanup")
    original_popen = runner.subprocess.Popen
    spawned: list[Any] = []

    def capture_popen(*args: Any, **kwargs: Any) -> Any:
        process = original_popen(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(
        runner,
        "_phase6_process_identity",
        lambda pid: (_ for _ in ()).throw(contract.ContractError("identity failed")),
    )

    with pytest.raises(contract.ContractError, match="identity failed"):
        runner.run_managed_process_group(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            environment={"CUDA_VISIBLE_DEVICES": "-1"},
            deadline_seconds=5,
            term_grace_seconds=0.2,
            kill_reap_grace_seconds=1.0,
        )

    assert len(spawned) == 1
    process = spawned[0]
    assert not runner._phase6_process_group_exists(process.pid)
    assert not Path(f"/proc/{process.pid}").exists()


def test_recovered_interruption_preserves_malformed_present_blobs_and_never_relaunches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_partial_recovery")
    monkeypatch.setattr(runner, "PHASE6_WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(runner, "_phase6_repo_path", lambda path: Path(path))
    monkeypatch.setattr(
        contract,
        "_phase6_child_artifact_paths",
        lambda current: runner._phase6_child_paths(current),
    )
    bindings = _minimal_bindings(
        runner,
        contract.PHASE6_PILOT_SCHEMA,
        "gate_b",
    )
    identity = contract.phase6_expected_roster(contract.PHASE6_PILOT_SCHEMA)[0]
    row = bindings["schedule"]["payload"]["records"][0]
    ledger = _fake_persisted_ledger(contract.PHASE6_PILOT_SCHEMA, bindings)
    ledger = contract.transition_phase6_record(
        ledger,
        identity_id=identity["identity_id"],
        new_state="running",
        timestamp_utc="2026-07-11T00:00:00+00:00",
        process=_running_process(row["child_command_argv"]),
    )
    paths = runner._phase6_child_paths(identity)
    malformed = {
        paths["artifact"]: b'{"partial":',
        paths["sidecar"]: b"not-json\x00",
        paths["journal"]: b'{"attempt_id":"truncated"',
    }
    for path, raw in malformed.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    snapshots: list[dict[str, Any]] = []

    def persist(path: Path, payload: Mapping[str, Any], final: bool) -> dict[str, Any]:
        del path, final
        snapshot = copy.deepcopy(dict(payload))
        snapshots.append(snapshot)
        return snapshot

    output_path = tmp_path / "pilot.json"
    output_path.write_text("{}", encoding="ascii")
    original_read_strict_json = contract.read_strict_json
    monkeypatch.setattr(
        contract,
        "read_strict_json",
        lambda path: (
            copy.deepcopy(ledger)
            if Path(path) == output_path
            else original_read_strict_json(Path(path))
        ),
    )
    monkeypatch.setattr(runner, "phase6_persist_and_validate", persist)
    monkeypatch.setattr(
        runner,
        "_phase6_recover_running_process",
        lambda process: _recovered_process(process["command_argv"]),
    )
    monkeypatch.setattr(
        runner,
        "run_managed_process_group",
        lambda *args, **kwargs: pytest.fail("recovered identity must never relaunch"),
    )
    monkeypatch.setattr(
        contract,
        "finalize_phase6_ledger",
        lambda payload: payload,
    )

    runner.phase6_execute_ledger(
        schema=contract.PHASE6_PILOT_SCHEMA,
        output_path=output_path,
        bindings=bindings,
        child_timeout_seconds=60,
        eligible_identity_ids=set(),
    )

    recovered = snapshots[0]["records"][0]
    assert recovered["state"] == "interrupted"
    assert recovered["reason"] == "supervisor_recovery"
    for field, path in (
        ("child_artifact", paths["artifact"]),
        ("payload_sidecar", paths["sidecar"]),
        ("progress_journal", paths["journal"]),
    ):
        blob = recovered["evidence"][field]
        raw = malformed[path]
        assert blob["present"] is True
        assert blob["strict_json"] is None
        assert blob["byte_count"] == len(raw)
        assert blob["sha256"] == hashlib.sha256(raw).hexdigest()
        assert base64.b64decode(blob["base64"]) == raw
        assert path.read_bytes() == raw


def test_fresh_attempt_refuses_stale_child_file_without_deleting_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_stale_file_refusal")
    monkeypatch.setattr(runner, "PHASE6_WORK_DIR", tmp_path / "work")
    identity = contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA)[0]
    artifact = runner._phase6_child_paths(identity)["artifact"]
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"preserve-me")

    with pytest.raises(contract.ContractError, match="reviewed recovery"):
        runner._phase6_prepare_child_artifacts(identity)
    assert artifact.read_bytes() == b"preserve-me"


def test_absolute_deadline_includes_spawn_and_durable_callback_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_absolute_deadline")
    observed_timeouts: list[float] = []

    class FakeProcess:
        pid = 424242
        returncode = 0

        def communicate(self, timeout: float) -> tuple[bytes, bytes]:
            observed_timeouts.append(timeout)
            return b"", b""

        def poll(self) -> int:
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(runner, "_phase6_process_identity", lambda pid: (pid, 99))
    monkeypatch.setattr(runner, "_phase6_process_group_exists", lambda pgid: False)
    clock = iter((1_000_000_000, 4_000_000_000, 4_100_000_000))
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: next(clock))

    result = runner.run_managed_process_group(
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
        environment={"CUDA_VISIBLE_DEVICES": "-1"},
        deadline_seconds=5,
        absolute_deadline_ns=6_000_000_000,
    )

    assert observed_timeouts == [pytest.approx(2.0)]
    assert result["deadline_seconds"] == pytest.approx(5.0)
    assert result["elapsed_seconds"] == pytest.approx(3.1)


def test_outer_sigterm_unwinds_through_real_process_group_cleanup(
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_real_sigterm_cleanup")
    started: dict[str, Any] = {}

    def terminate_after_start(process: Mapping[str, Any]) -> None:
        started.update(process)
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)

    with pytest.raises(runner.Phase6OuterTermination):
        with runner.phase6_outer_sigterm_guard(lambda signum: None):
            runner.run_managed_process_group(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                cwd=tmp_path,
                environment={"CUDA_VISIBLE_DEVICES": "-1"},
                deadline_seconds=5,
                term_grace_seconds=0.2,
                kill_reap_grace_seconds=1.0,
                on_started=terminate_after_start,
            )

    assert started
    assert not runner._phase6_process_group_exists(started["pgid"])
    assert not Path(f"/proc/{started['pid']}").exists()


def test_execute_ledger_persists_outer_termination_after_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_outer_terminal_record")
    bindings = _minimal_bindings(
        runner,
        contract.PHASE6_PILOT_SCHEMA,
        "gate_b",
    )
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    initial = _fake_persisted_ledger(contract.PHASE6_PILOT_SCHEMA, bindings)
    snapshots: list[dict[str, Any]] = []

    monkeypatch.setattr(contract, "new_phase6_ledger", lambda **kwargs: initial)

    def persist(path: Path, payload: Mapping[str, Any], final: bool) -> dict[str, Any]:
        del path, final
        snapshot = copy.deepcopy(dict(payload))
        snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(runner, "phase6_persist_and_validate", persist)
    monkeypatch.setattr(runner, "_phase6_prepare_child_artifacts", lambda identity: None)
    monkeypatch.setattr(
        runner,
        "_phase6_record_evidence",
        lambda identity, classification, discovery: _interruption_evidence(identity),
    )

    def interrupted_launch(
        command: Sequence[str],
        *,
        deadline_seconds: float,
        on_started: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        running = _running_process(command)
        running["deadline_seconds"] = deadline_seconds
        on_started(running)
        raise runner.Phase6OuterTermination("synthetic outer TERM")

    monkeypatch.setattr(runner, "run_managed_process_group", interrupted_launch)
    monkeypatch.setattr(
        runner,
        "_phase6_recover_running_process",
        lambda process: _recovered_process(process["command_argv"]),
    )

    with pytest.raises(runner.Phase6OuterTermination):
        runner.phase6_execute_ledger(
            schema=contract.PHASE6_PILOT_SCHEMA,
            output_path=tmp_path / "pilot.json",
            bindings=bindings,
            child_timeout_seconds=60,
        )

    terminal = snapshots[-1]["records"][0]
    assert terminal["state"] == "interrupted"
    assert terminal["reason"] == "outer_termination"
    assert terminal["evidence"]["classification"] == "supervisor_interruption"
    assert terminal["process"]["process_group_gone"] is True


def test_routing_ledger_preallocates_updates_persists_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_routing")
    new_routing = _require_api(runner, "phase6_new_routing_ledger")
    routing_decision = _require_api(runner, "phase6_routing_decision")
    persist_routing = _require_api(runner, "phase6_persist_routing")
    bindings = _minimal_bindings(
        runner,
        contract.PHASE6_FINAL_SCHEMA,
        "gate_c",
    )
    routing_path = tmp_path / "routing.json"

    routing = new_routing(bindings)
    expected_roster = contract.phase6_expected_roster(contract.PHASE6_ROUTING_SCHEMA)
    assert routing["schema"] == contract.PHASE6_ROUTING_SCHEMA
    assert len(routing["records"]) == 18
    assert [record["identity"] for record in routing["records"]] == expected_roster
    assert routing["update_index"] == 0
    assert routing["state"] == "running"
    assert all(
        record
        == {
            "identity": identity,
                "state": "pending_dependency",
                "reason": None,
                "dependencies": None,
                "prelaunch_snapshot": None,
                "fingerprints": None,
            "rule_id": None,
            "action": None,
        }
        for record, identity in zip(routing["records"], expected_roster, strict=True)
    )
    routing = persist_routing(routing_path, routing, final=False)
    assert routing == contract.read_strict_json(routing_path)

    analytical = contract.PRIMARY_METHOD_IDS[0]
    final_schedule = bindings["schedule"]["payload"]
    p50_row = _find_schedule_row(
        final_schedule,
        dimension=10,
        parameter_count=50,
        batch_size=1,
        method_id=analytical,
    )
    target = next(
        identity
        for identity in expected_roster
        if (
            identity["dimension"],
            identity["batch_size"],
            identity["method_id"],
        )
        == (10, 1, analytical)
    )
    final_payload = _fake_persisted_ledger(contract.PHASE6_FINAL_SCHEMA, bindings)
    for identity in final_payload["roster"][:6]:
        final_payload = contract.transition_phase6_record(
            final_payload,
            identity_id=identity["identity_id"],
            new_state="not_launched:global_budget_exhausted",
            timestamp_utc="2026-07-11T00:00:00+00:00",
            reason="global_budget_exhausted",
        )
    p50_record = next(
        record
        for record in final_payload["records"]
        if record["identity"] == p50_row["identity"]
    )
    p50_record["state"] = "passed"
    p50_record["reason"] = "child_passed"
    p50_record["process"] = {"synthetic": "passed dependency"}
    p50_record["evidence"] = {"classification": "method_pass"}

    routing = routing_decision(routing, final_payload, target)
    decided = next(record for record in routing["records"] if record["identity"] == target)
    assert decided["state"] == "decided"
    assert decided["reason"] is None
    assert decided["rule_id"] == (
        "p50_passed_and_preceding_p150_passed_or_not_applicable"
    )
    assert decided["action"] == "eligible_under_gate_c_budget"
    assert decided["dependencies"] is not None
    assert decided["prelaunch_snapshot"]["ledger_update_index"] == 6
    assert decided["prelaunch_snapshot"]["closed_record_count"] == 6
    assert decided["prelaunch_snapshot"]["common_invalidity_present"] is False
    assert decided["fingerprints"] is not None
    assert routing["update_index"] == 1
    routing = persist_routing(routing_path, routing, final=False)
    assert routing == contract.read_strict_json(routing_path)

    tampered = copy.deepcopy(routing)
    tampered["records"][0]["action"] = "eligible_under_gate_c_budget"
    with pytest.raises(contract.ContractError):
        persist_routing(routing_path, tampered, final=False)
    assert contract.read_strict_json(routing_path) == routing

    drifted = copy.deepcopy(final_payload)
    changed_p50 = next(
        record
        for record in drifted["records"]
        if record["identity"] == p50_row["identity"]
    )
    changed_p50["process"]["synthetic"] = "dependency drift"
    with pytest.raises(contract.ContractError, match="no longer matches"):
        routing_decision(routing, drifted, target)

    backdated = copy.deepcopy(routing)
    backdated_record = next(
        record for record in backdated["records"] if record["identity"] == target
    )
    backdated_record["prelaunch_snapshot"]["common_invalidity_present"] = True
    with pytest.raises(contract.ContractError):
        persist_routing(routing_path, backdated, final=False)


def test_imported_common_invalidity_closes_real_final_and_routing_without_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_common_invalid_integration")
    pilot = _terminal_common_invalid_pilot(runner)
    pilot_blob = _blob(pilot, str(tmp_path / "pilot.json"))
    bindings = _minimal_bindings(
        runner,
        contract.PHASE6_FINAL_SCHEMA,
        "gate_c",
        authority_inputs=[pilot_blob],
    )
    output_path = tmp_path / "final.json"
    routing_path = tmp_path / "routing.json"
    monkeypatch.setattr(runner, "_phase6_repo_path", lambda path: Path(path))
    imported = {
        record["identity"]["identity_id"]: record for record in pilot["records"]
    }
    monkeypatch.setattr(
        runner,
        "run_managed_process_group",
        lambda *args, **kwargs: pytest.fail(
            "imported common invalidity must prevent every spawn"
        ),
    )

    final = runner.phase6_execute_ledger(
        schema=contract.PHASE6_FINAL_SCHEMA,
        output_path=output_path,
        bindings=bindings,
        child_timeout_seconds=60,
        imported_records=imported,
        routing_path=routing_path,
    )
    routing = contract.read_strict_json(routing_path)

    assert all(contract.phase6_ledger_checks(final, final=True).values())
    assert sum(record["imported_from"] is not None for record in final["records"]) == 2
    assert all(
        record["state"] == "not_launched:common_invalidity"
        for record in final["records"][2:]
    )
    assert routing["schema"].endswith("p150_routing.v3")
    assert routing["state"] == "closed"
    assert routing["update_index"] == 18
    assert all(record["state"] == "decided" for record in routing["records"])
    assert all(record["rule_id"] == "common_invalidity" for record in routing["records"])
    assert all(
        record["prelaunch_snapshot"]["common_invalidity_present"] is True
        for record in routing["records"]
    )
    overlay = routing["terminal_overlay"]
    assert overlay["mode"] == "common_invalidity"
    assert all(
        row["effective_action"] == "globally_invalidated_by_common_invalidity"
        for row in overlay["dispositions"]
    )
    assert all(runner.phase6_final_routing_checks(final, routing).values())

    malformed_overlay = copy.deepcopy(routing)
    malformed_overlay["terminal_overlay"]["dispositions"][0].pop("final_reason")
    assert not all(runner._phase6_routing_checks(malformed_overlay, final=True).values())
    assert not all(runner.phase6_final_routing_checks(final, malformed_overlay).values())


def test_imported_common_invalidity_recovers_owned_running_record_before_rejection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_common_invalid_recovery_order")
    pilot = _terminal_common_invalid_pilot(runner)
    pilot_blob = _blob(pilot, str(tmp_path / "pilot.json"))
    bindings = _minimal_bindings(
        runner,
        contract.PHASE6_FINAL_SCHEMA,
        "gate_c",
        authority_inputs=[pilot_blob],
    )
    ledger = contract.new_phase6_ledger(
        schema=contract.PHASE6_FINAL_SCHEMA,
        gate="gate_c",
        artifact_kind="cpu_xla_final",
        identities=contract.phase6_expected_roster(contract.PHASE6_FINAL_SCHEMA),
        bindings=bindings,
    )
    first = ledger["roster"][0]
    first_row = bindings["schedule"]["payload"]["records"][0]
    ledger = contract.transition_phase6_record(
        ledger,
        identity_id=first["identity_id"],
        new_state="running",
        timestamp_utc="2026-07-11T00:00:00+00:00",
        process=_running_process(first_row["child_command_argv"]),
    )
    output_path = tmp_path / "final.json"
    contract.durable_atomic_write_json(output_path, ledger)
    monkeypatch.setattr(runner, "_phase6_repo_path", lambda path: Path(path))
    recoveries: list[Mapping[str, Any]] = []

    def recover(process: Mapping[str, Any]) -> dict[str, Any]:
        recoveries.append(process)
        return _recovered_process(process["command_argv"])

    monkeypatch.setattr(runner, "_phase6_recover_running_process", recover)
    monkeypatch.setattr(
        runner,
        "run_managed_process_group",
        lambda *args, **kwargs: pytest.fail("inconsistent prior execution must not spawn"),
    )
    imported = {
        record["identity"]["identity_id"]: record for record in pilot["records"]
    }

    with pytest.raises(contract.ContractError, match="already launched work"):
        runner.phase6_execute_ledger(
            schema=contract.PHASE6_FINAL_SCHEMA,
            output_path=output_path,
            bindings=bindings,
            child_timeout_seconds=60,
            imported_records=imported,
            routing_path=tmp_path / "routing.json",
        )

    assert len(recoveries) == 1
    recovered = contract.read_strict_json(output_path)["records"][0]
    assert recovered["state"] == "interrupted"
    assert recovered["reason"] == "supervisor_recovery"


def test_prelaunch_deadline_expiry_prunes_durably_without_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_prelaunch_expiry")
    bindings = _minimal_bindings(
        runner,
        contract.PHASE6_PILOT_SCHEMA,
        "gate_b",
    )
    monkeypatch.setattr(runner, "PHASE6_WORK_DIR", tmp_path / "work")
    budget_path = tmp_path / "budget.json"
    monkeypatch.setattr(runner, "_phase6_repo_path", lambda path: Path(path))
    clock = [1_000_000_000]
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: clock[0])
    monkeypatch.setattr(
        runner,
        "run_managed_process_group",
        lambda *args, **kwargs: pytest.fail("expired prelaunch must not spawn"),
    )

    def revalidate(current: Mapping[str, Any]) -> None:
        assert current is bindings
        clock[0] = 63_000_000_000

    with runner.phase6_budget_lease(
        budget_path, "trace_census_and_pilot"
    ) as lease:
        runner.phase6_budget_state_open(
            budget_path,
            bindings["authority_id"],
            "gate_b",
            71.0,
            "trace_census_and_pilot",
            lease=lease,
            now_ns=1_000_000_000,
        )
        result = runner.phase6_execute_ledger(
            schema=contract.PHASE6_PILOT_SCHEMA,
            output_path=tmp_path / "pilot.json",
            bindings=bindings,
            child_timeout_seconds=60,
            authority_validator=revalidate,
            budget_path=budget_path,
            budget_lease=lease,
            expected_budget_command_name="trace_census_and_pilot",
            gate_hard_ceiling_seconds=71.0,
        )

    assert all(
        record["state"] == "not_launched:global_budget_exhausted"
        for record in result["records"]
    )
    assert all(contract.phase6_ledger_checks(result, final=True).values())


def test_p150_route_is_durable_before_the_only_mocked_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_route_before_spawn")
    pilot_records = [
        {
            "identity": identity,
            "state": "passed",
            "reason": "child_passed",
            "process": {"synthetic": "pilot pass"},
            "evidence": {"classification": "method_pass"},
            "imported_from": None,
        }
        for identity in contract.phase6_expected_roster(contract.PHASE6_PILOT_SCHEMA)
    ]
    pilot_blob = _blob(
        {"schema": contract.PHASE6_PILOT_SCHEMA, "records": pilot_records},
        str(tmp_path / "pilot.json"),
    )
    bindings = _minimal_bindings(
        runner,
        contract.PHASE6_FINAL_SCHEMA,
        "gate_c",
        authority_inputs=[pilot_blob],
    )
    monkeypatch.setattr(runner, "PHASE6_WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    initial = _fake_persisted_ledger(contract.PHASE6_FINAL_SCHEMA, bindings)
    routing_path = tmp_path / "routing.json"
    target = next(
        identity
        for identity in initial["roster"]
        if (
            identity["dimension"],
            identity["parameter_count"],
            identity["batch_size"],
            identity["method_id"],
        )
        == (10, 150, 1, contract.PRIMARY_METHOD_IDS[0])
    )
    transitions: list[dict[str, Any]] = []
    launches: list[list[str]] = []
    original_transition = contract.transition_phase6_record

    def transition(payload: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        transitions.append(kwargs)
        updated = copy.deepcopy(dict(payload))
        record = next(
            row
            for row in updated["records"]
            if row["identity"]["identity_id"] == kwargs["identity_id"]
        )
        prior_state = record["state"]
        record.update(
            state=kwargs["new_state"],
            reason=kwargs.get("reason"),
            process=copy.deepcopy(kwargs.get("process")),
            evidence=copy.deepcopy(kwargs.get("evidence")),
            imported_from=copy.deepcopy(kwargs.get("imported_from")),
        )
        updated["update_index"] += 1
        updated["events"].append(
            {
                "update_index": updated["update_index"],
                "identity_id": kwargs["identity_id"],
                "prior_state": prior_state,
                "new_state": kwargs["new_state"],
                "timestamp_utc": kwargs["timestamp_utc"],
                "evidence_sha256": contract._phase6_record_event_digest(record),
            }
        )
        return updated

    monkeypatch.setattr(contract, "new_phase6_ledger", lambda **kwargs: initial)
    monkeypatch.setattr(contract, "transition_phase6_record", transition)
    monkeypatch.setattr(
        contract,
        "finalize_phase6_ledger",
        lambda payload: (_ for _ in ()).throw(RuntimeError("stop before closure")),
    )
    monkeypatch.setattr(
        contract,
        "phase6_terminal_record_semantics_valid",
        lambda record, bindings: True,
    )
    monkeypatch.setattr(
        runner,
        "phase6_persist_and_validate",
        lambda path, payload, final: payload,
    )
    monkeypatch.setattr(runner, "_phase6_prepare_child_artifacts", lambda identity: None)
    monkeypatch.setattr(
        runner,
        "_phase6_record_evidence",
        lambda identity, classification, discovery: {"classification": classification},
    )

    def launch(
        command: Sequence[str],
        *,
        deadline_seconds: float,
        on_started: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        routing = contract.read_strict_json(routing_path)
        route_identity = runner._phase6_routing_identity(target)
        decision = next(
            record
            for record in routing["records"]
            if record["identity"] == route_identity
        )
        assert decision["action"] == "eligible_under_gate_c_budget"
        launches.append(list(command))
        running = _running_process(command)
        running["deadline_seconds"] = deadline_seconds
        on_started(running)
        return _completed_process(command, deadline_seconds=deadline_seconds)

    monkeypatch.setattr(runner, "run_managed_process_group", launch)
    original_ledger_checks = contract.phase6_ledger_checks
    monkeypatch.setattr(
        contract,
        "phase6_ledger_checks",
        lambda payload, final: (
            {"synthetic_pilot_valid": True}
            if payload is pilot_blob["strict_json"]
            else original_ledger_checks(payload, final=final)
        ),
    )
    imported = {
        record["identity"]["identity_id"]: record for record in pilot_records
    }
    with pytest.raises(RuntimeError, match="stop before closure"):
        runner.phase6_execute_ledger(
            schema=contract.PHASE6_FINAL_SCHEMA,
            output_path=tmp_path / "final.json",
            bindings=bindings,
            child_timeout_seconds=60,
            eligible_identity_ids={target["identity_id"]},
            imported_records=imported,
            authority_validator=lambda current: None,
            routing_path=routing_path,
        )

    assert len(launches) == 1
    command = launches[0]
    assert command[command.index("--dimensions") + 1] == "10"
    assert command[command.index("--parameter-counts") + 1] == "150"
    assert command[command.index("--batch-size") + 1] == "1"
    assert command[command.index("--method") + 1] == target["method_id"]
    assert contract.read_strict_json(routing_path)["state"] == "decisions_complete"


def test_gate_c_scalar_wires_shared_budget_and_continues_after_valid_nonpass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_scalar_entrypoint")
    authority_id = "c" * 64
    proposal = {
        "authority_id": authority_id,
        "budget": {"hard_ceiling_seconds": 3120, "cell_cap_seconds": 160},
    }
    proposal_path = tmp_path / "proposal.json"
    attestation_path = tmp_path / "attestation.json"
    budget_path = tmp_path / "budget-state.json"
    output_path = tmp_path / "scalar.json"
    calls: list[tuple[str, Any]] = []
    bindings = {"synthetic": "scalar bindings"}

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(
        runner,
        "_phase6_resolve_authority_path",
        lambda path, label: proposal_path if "proposal" in label else attestation_path,
    )
    monkeypatch.setattr(
        contract,
        "validate_phase6_runtime_authority",
        lambda *args, **kwargs: (proposal, {"gate": "gate_c"}),
    )
    monkeypatch.setattr(runner, "_phase6_budget_state_path", lambda *args: budget_path)
    monkeypatch.setattr(
        runner,
        "phase6_budget_state_open",
        lambda *args, **kwargs: calls.append(("open", args)) or {"running": True},
    )
    monkeypatch.setattr(
        runner,
        "phase6_budget_state_close_command",
        lambda *args, **kwargs: calls.append(("close", args)) or {"closed": True},
    )
    monkeypatch.setattr(
        runner,
        "_phase6_bindings_for_gate",
        lambda **kwargs: calls.append(("bindings", kwargs)) or bindings,
    )
    monkeypatch.setattr(runner, "_phase6_repo_path", lambda path: output_path)
    monkeypatch.setattr(
        runner,
        "phase6_revalidate_launch_authority",
        lambda current: calls.append(("revalidate", current)),
    )
    monkeypatch.setattr(
        runner,
        "phase6_execute_ledger",
        lambda **kwargs: calls.append(("execute", kwargs))
        or {"state": "complete_with_failures"},
    )
    args = SimpleNamespace(
        dimensions=[10],
        parameter_counts=[50],
        batch_sizes=[1, 4],
        timesteps=120,
        dtype="float32",
        device="cpu",
        cpu_threads=1,
        jit_compile=False,
        child_timeout_seconds=60,
        budget_contract=proposal_path,
        budget_attestation=attestation_path,
        output_json=output_path,
    )

    assert runner.run_phase6_scalar_references(args) == 0
    execute = next(value for name, value in calls if name == "execute")
    assert execute["authority_validator"] is runner.phase6_revalidate_launch_authority
    assert execute["budget_path"] == budget_path
    assert execute["gate_hard_ceiling_seconds"] == 3120
    assert execute["cell_cap_seconds"] == 160
    assert [name for name, _ in calls] == [
        "open",
        "bindings",
        "execute",
        "revalidate",
        "close",
    ]


def test_gate_c_scalar_exception_keeps_budget_command_open_for_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_scalar_exception")
    proposal = {
        "authority_id": "d" * 64,
        "budget": {"hard_ceiling_seconds": 3120, "cell_cap_seconds": 160},
    }
    close_calls: list[Any] = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(
        runner,
        "_phase6_resolve_authority_path",
        lambda path, label: tmp_path / ("proposal.json" if "proposal" in label else "attestation.json"),
    )
    monkeypatch.setattr(
        contract,
        "validate_phase6_runtime_authority",
        lambda *args, **kwargs: (proposal, {}),
    )
    monkeypatch.setattr(runner, "_phase6_budget_state_path", lambda *args: tmp_path / "budget.json")
    monkeypatch.setattr(
        runner,
        "phase6_budget_state_open",
        lambda *args, **kwargs: {"running": True},
    )
    monkeypatch.setattr(
        runner,
        "phase6_budget_state_close_command",
        lambda *args, **kwargs: close_calls.append(args),
    )
    monkeypatch.setattr(runner, "_phase6_bindings_for_gate", lambda **kwargs: {})
    monkeypatch.setattr(runner, "_phase6_repo_path", lambda path: tmp_path / "scalar.json")
    monkeypatch.setattr(
        runner,
        "phase6_execute_ledger",
        lambda **kwargs: (_ for _ in ()).throw(
            runner.Phase6OuterTermination("outer TERM")
        ),
    )
    args = SimpleNamespace(
        dimensions=[10],
        parameter_counts=[50],
        batch_sizes=[1, 4],
        timesteps=120,
        dtype="float32",
        device="cpu",
        cpu_threads=1,
        jit_compile=False,
        child_timeout_seconds=60,
        budget_contract=tmp_path / "proposal.json",
        budget_attestation=tmp_path / "attestation.json",
        output_json=tmp_path / "scalar.json",
    )

    with pytest.raises(runner.Phase6OuterTermination):
        runner.run_phase6_scalar_references(args)
    assert close_calls == []


def test_gate_b_entrypoint_shares_one_budget_across_trace_and_pilot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_gate_b_entrypoint")
    trace_bindings = _minimal_bindings(
        runner, contract.PHASE6_TRACE_SCHEMA, "gate_b"
    )
    proposal = copy.deepcopy(trace_bindings["proposal"]["strict_json"])
    proposal["budget"] = {"hard_ceiling_seconds": 3045, "cell_cap_seconds": 160}
    proposal["commands"] = [runner._phase6_gate_b_command()]
    authority_id = proposal["authority_id"]
    budget_path = tmp_path / "gate-b-budget.json"
    calls: list[tuple[str, Any]] = []

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(
        runner,
        "_phase6_exact_supervisor_argv",
        lambda: list(runner._phase6_gate_b_command()["argv"]),
    )
    monkeypatch.setattr(
        runner, "_phase6_assert_fresh_gate_b_namespace", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "_phase6_resolve_authority_path",
        lambda path, label: tmp_path / ("proposal.json" if "proposal" in label else "attestation.json"),
    )
    monkeypatch.setattr(
        runner,
        "_phase6_repo_path",
        lambda path: tmp_path / Path(path).name,
    )
    monkeypatch.setattr(
        contract,
        "validate_phase6_runtime_authority",
        lambda *args, **kwargs: (proposal, {}),
    )
    monkeypatch.setattr(runner, "_phase6_budget_state_path", lambda *args: budget_path)

    class Lease:
        def __enter__(self):
            calls.append(("lease", None))
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    monkeypatch.setattr(runner, "phase6_budget_lease", lambda *args: Lease())
    monkeypatch.setattr(
        runner,
        "phase6_budget_state_open",
        lambda *args, **kwargs: calls.append(("open", args)) or {"running": True},
    )
    monkeypatch.setattr(
        runner,
        "phase6_budget_state_close_command",
        lambda *args, **kwargs: calls.append(("close", args)) or {"closed": True},
    )

    def bindings(**kwargs: Any) -> dict[str, Any]:
        calls.append(("bindings", kwargs))
        return (
            trace_bindings
            if kwargs["schema"] == contract.PHASE6_TRACE_SCHEMA
            else {"schema": kwargs["schema"]}
        )

    monkeypatch.setattr(runner, "_phase6_bindings_for_gate", bindings)
    monkeypatch.setattr(
        contract,
        "evaluate_phase6_trace_census",
        lambda payload: {"trace_common_valid": True},
    )
    monkeypatch.setattr(
        contract,
        "phase6_terminal_summary",
        lambda payload: {"has_common_invalidity": False},
    )
    monkeypatch.setattr(
        runner,
        "phase6_revalidate_launch_authority",
        lambda current: calls.append(("revalidate", current)),
    )

    def execute(**kwargs: Any) -> dict[str, Any]:
        calls.append(("execute", kwargs))
        return {
            "schema": kwargs["schema"],
            "state": "passed",
        }

    monkeypatch.setattr(runner, "phase6_execute_ledger", execute)
    args = SimpleNamespace(
        dimensions=[10, 20, 30],
        parameter_counts=[50, 150],
        batch_sizes=[1, 4, 16],
        timesteps=120,
        dtype="float32",
        device="cpu",
        cpu_threads=1,
        jit_compile=True,
        trace_child_timeout_seconds=60,
        xla_child_timeout_seconds=60,
        xla_cell_timeout_seconds=160,
        budget_contract=tmp_path / "proposal.json",
        budget_attestation=tmp_path / "attestation.json",
        trace_output_json=Path("trace.json"),
        output_json=Path("pilot.json"),
    )

    assert runner.run_phase6_pilot(args) == 0
    executions = [value for name, value in calls if name == "execute"]
    assert [row["schema"] for row in executions] == [
        contract.PHASE6_TRACE_SCHEMA,
        contract.PHASE6_PILOT_SCHEMA,
    ]
    assert all(row["budget_path"] == budget_path for row in executions)
    assert all(row["gate_hard_ceiling_seconds"] == 3045 for row in executions)
    assert "cell_cap_seconds" not in executions[0]
    assert executions[1]["cell_cap_seconds"] == 160
    pilot_binding = [value for name, value in calls if name == "bindings"][1]
    assert pilot_binding["runtime_predecessor_paths"] == (tmp_path / "trace.json",)
    assert [name for name, _ in calls].count("open") == 1
    assert [name for name, _ in calls].count("close") == 1
    names = [name for name, _ in calls]
    assert names[:3] == ["bindings", "revalidate", "lease"]
    assert names.index("lease") < names.index("open") < names.index("execute")
    assert executions[0]["initial_payload"]["bindings"] == trace_bindings


def test_gate_b_deterministic_preflight_failure_creates_no_runtime_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_r3_preflight_before_budget")
    proposal = {
        "authority_id": "e" * 64,
        "budget": {"hard_ceiling_seconds": 3045, "cell_cap_seconds": 160},
        "commands": [runner._phase6_gate_b_command()],
    }
    work = tmp_path / "work"
    trace = tmp_path / "trace.json"
    pilot = tmp_path / "pilot.json"
    proposal_path = tmp_path / "proposal.json"
    attestation_path = tmp_path / "attestation.json"
    proposal_path.write_text("{}\n", encoding="ascii")
    attestation_path.write_text("{}\n", encoding="ascii")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(runner, "PHASE6_WORK_DIR", work)
    monkeypatch.setattr(
        runner,
        "_phase6_exact_supervisor_argv",
        lambda: list(runner._phase6_gate_b_command()["argv"]),
    )
    monkeypatch.setattr(
        runner, "_phase6_assert_fresh_gate_b_namespace", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "_phase6_resolve_authority_path",
        lambda path, label: proposal_path if "proposal" in label else attestation_path,
    )
    monkeypatch.setattr(runner, "_phase6_repo_path", lambda path: Path(path))
    monkeypatch.setattr(
        contract,
        "validate_phase6_runtime_authority",
        lambda *args, **kwargs: (proposal, {}),
    )
    monkeypatch.setattr(
        runner,
        "_phase6_bindings_for_gate",
        lambda **kwargs: (_ for _ in ()).throw(
            contract.ContractError("deterministic mixed-format binding failure")
        ),
    )
    monkeypatch.setattr(
        runner,
        "phase6_budget_lease",
        lambda *args, **kwargs: pytest.fail("preflight failure must precede lease"),
    )
    args = SimpleNamespace(
        dimensions=[10, 20, 30],
        parameter_counts=[50, 150],
        batch_sizes=[1, 4, 16],
        timesteps=120,
        dtype="float32",
        device="cpu",
        cpu_threads=1,
        jit_compile=True,
        trace_child_timeout_seconds=60,
        xla_child_timeout_seconds=60,
        xla_cell_timeout_seconds=160,
        budget_contract=proposal_path,
        budget_attestation=attestation_path,
        trace_output_json=trace,
        output_json=pilot,
    )

    with pytest.raises(contract.ContractError, match="deterministic mixed-format"):
        runner.run_phase6_pilot(args)

    assert not work.exists() and not work.is_symlink()
    assert not trace.exists() and not trace.is_symlink()
    assert not pilot.exists() and not pilot.is_symlink()


def test_gate_c_remaining_resumes_budget_and_wires_routing_and_predecessors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("phase6_runtime_controls_remaining_entrypoint")
    authority_id = "f" * 64
    proposal = {
        "authority_id": authority_id,
        "budget": {"hard_ceiling_seconds": 3120, "cell_cap_seconds": 160},
    }
    budget_path = tmp_path / "gate-c-budget.json"
    trace_path = tmp_path / "trace.json"
    pilot_path = tmp_path / "pilot.json"
    scalar_path = tmp_path / "scalar.json"
    routing_path = tmp_path / "routing.json"
    final_path = tmp_path / "final.json"
    calls: list[tuple[str, Any]] = []
    pilot_payload = {
        "records": [
            {"identity": {"identity_id": "pilot-cell"}, "state": "passed"}
        ]
    }

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    def resolve(path: Path | None, label: str) -> Path:
        mapping = {
            "Gate C proposal": tmp_path / "proposal.json",
            "Gate C attestation": tmp_path / "attestation.json",
            "trace input": trace_path,
            "pilot input": pilot_path,
            "scalar input": scalar_path,
        }
        return mapping[label]

    monkeypatch.setattr(runner, "_phase6_resolve_authority_path", resolve)
    monkeypatch.setattr(
        runner,
        "_phase6_repo_path",
        lambda path: routing_path if Path(path).name == routing_path.name else final_path,
    )
    monkeypatch.setattr(
        contract,
        "validate_phase6_runtime_authority",
        lambda *args, **kwargs: (proposal, {}),
    )
    monkeypatch.setattr(runner, "_phase6_budget_state_path", lambda *args: budget_path)
    monkeypatch.setattr(
        runner,
        "phase6_budget_state_open",
        lambda *args, **kwargs: calls.append(("open", args)) or {"running": True},
    )
    monkeypatch.setattr(
        runner,
        "phase6_budget_state_close_command",
        lambda *args, **kwargs: calls.append(("close", args)) or {"closed": True},
    )
    monkeypatch.setattr(
        contract,
        "read_bounded_phase6_trace_json",
        lambda path: {"schema": contract.PHASE6_TRACE_SCHEMA},
    )
    monkeypatch.setattr(
        contract,
        "evaluate_phase6_trace_census",
        lambda payload: {"trace_common_valid": True},
    )
    monkeypatch.setattr(
        contract,
        "read_strict_json",
        lambda path: pilot_payload if path == pilot_path else {"schema": contract.PHASE6_SCALAR_SCHEMA},
    )
    monkeypatch.setattr(
        contract,
        "phase6_ledger_checks",
        lambda payload, final: {"synthetic_valid": True},
    )
    bindings = {"synthetic": "final bindings"}
    monkeypatch.setattr(
        runner,
        "_phase6_bindings_for_gate",
        lambda **kwargs: calls.append(("bindings", kwargs)) or bindings,
    )
    monkeypatch.setattr(
        runner,
        "phase6_revalidate_launch_authority",
        lambda current: calls.append(("revalidate", current)),
    )
    monkeypatch.setattr(
        runner,
        "phase6_execute_ledger",
        lambda **kwargs: calls.append(("execute", kwargs)) or {"state": "passed"},
    )
    monkeypatch.setattr(
        contract,
        "evaluate_phase6_handoff",
        lambda payload: {"phase7_scope": "blocked"},
    )
    args = SimpleNamespace(
        dimensions=[10, 20, 30],
        parameter_counts=[50, 150],
        batch_sizes=[1, 4, 16],
        timesteps=120,
        dtype="float32",
        device="cpu",
        cpu_threads=1,
        jit_compile=True,
        child_timeout_seconds=60,
        cell_timeout_seconds=160,
        budget_contract=tmp_path / "proposal.json",
        budget_attestation=tmp_path / "attestation.json",
        trace_input=trace_path,
        pilot_input=pilot_path,
        scalar_reference_input=scalar_path,
        routing_output_json=routing_path,
        output_json=final_path,
    )

    assert runner.run_phase6_remaining(args) == 1
    open_args = next(value for name, value in calls if name == "open")
    assert open_args[4] == "remaining_lattice"
    binding_args = next(value for name, value in calls if name == "bindings")
    assert binding_args["authority_input_paths"] == (trace_path, pilot_path)
    assert binding_args["runtime_predecessor_paths"] == (scalar_path,)
    execute = next(value for name, value in calls if name == "execute")
    assert execute["budget_path"] == budget_path
    assert execute["gate_hard_ceiling_seconds"] == 3120
    assert execute["cell_cap_seconds"] == 160
    assert execute["routing_path"] == routing_path
    assert execute["authority_validator"] is runner.phase6_revalidate_launch_authority
    assert [name for name, _ in calls][-2:] == ["revalidate", "close"]
