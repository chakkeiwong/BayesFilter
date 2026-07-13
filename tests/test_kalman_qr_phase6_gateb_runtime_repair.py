from __future__ import annotations

import copy
import base64
import inspect
import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Mapping

import pytest

from scripts import kalman_qr_benchmark_contract as contract
from tests.test_kalman_qr_phase6_cpu_xla_gates import (
    _bindings,
    _failure_evidence,
    _running_process,
    _terminal_process,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "scripts/benchmark_kalman_qr_parameter_count_scaling.py"
RUNNER_PATH = (
    ROOT / "docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py"
)
PYTHON = "/home/ubuntu/anaconda3/envs/tfgpu/bin/python"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_process_argv_preserves_reviewed_interpreter_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _load(BENCHMARK_PATH, "phase6_runtime_repair_exact_argv")
    arguments = ["--phase6-trace-only", "--method", contract.PRIMARY_METHOD_IDS[0]]
    monkeypatch.setattr(sys, "argv", [str(BENCHMARK_PATH), *arguments])
    monkeypatch.setattr(
        sys,
        "orig_argv",
        [PYTHON, str(BENCHMARK_PATH), *arguments],
        raising=False,
    )

    assert benchmark._phase6_exact_process_argv() == [
        PYTHON,
        str(BENCHMARK_PATH),
        *arguments,
    ]


def test_real_python_orig_argv_preserves_symlink_spelling(tmp_path: Path) -> None:
    probe = tmp_path / "orig_argv_probe.py"
    probe.write_text(
        "import json, sys\nprint(json.dumps(sys.orig_argv))\n",
        encoding="ascii",
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    completed = subprocess.run(
        [PYTHON, str(probe), "sentinel"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [PYTHON, str(probe), "sentinel"]


def test_exact_argv_source_shape_does_not_accept_resolved_equivalence() -> None:
    benchmark = _load(BENCHMARK_PATH, "phase6_runtime_repair_source_shape")
    helper = inspect.getsource(benchmark._phase6_preimport_exact_process_argv)
    trace = inspect.getsource(benchmark.run_phase6_trace_only)
    terminal = inspect.getsource(contract.phase6_terminal_record_semantics_valid)

    assert "raw[0] != PHASE6_IMPORT_DISCOVERY_PYTHON" in helper
    assert "return list(raw)" in helper
    assert "command_argv = _phase6_exact_process_argv()" in trace
    assert '"command_argv": command_argv' in trace
    assert 'process.get("command_argv") != schedule_row.get("child_command_argv")' in terminal
    assert 'Path(process.get("command_argv")' not in terminal
    assert 'Path(schedule_row.get("child_command_argv")' not in terminal
    assert "realpath" not in terminal
    assert "samefile" not in terminal


def test_exact_supervisor_argv_matches_reviewed_gate_b_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load(RUNNER_PATH, "phase6_runtime_repair_supervisor_exact_argv")
    expected = runner._phase6_gate_b_command()["argv"]
    monkeypatch.setattr(sys, "argv", [expected[1], *expected[2:]])
    monkeypatch.setattr(sys, "orig_argv", expected, raising=False)

    assert runner._phase6_exact_supervisor_argv() == expected


@pytest.mark.parametrize("mutation", ["missing", "interpreter", "script", "output"])
def test_exact_supervisor_argv_rejects_unreviewed_runtime_tokens(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    runner = _load(
        RUNNER_PATH, f"phase6_runtime_repair_supervisor_argv_{mutation}"
    )
    expected = runner._phase6_gate_b_command()["argv"]
    raw = list(expected)
    if mutation == "missing":
        monkeypatch.delattr(sys, "orig_argv", raising=False)
    elif mutation == "interpreter":
        raw[0] = str(Path(PYTHON).resolve())
        monkeypatch.setattr(sys, "orig_argv", raw, raising=False)
    elif mutation == "script":
        raw[1] = str(BENCHMARK_PATH)
        monkeypatch.setattr(sys, "orig_argv", raw, raising=False)
    else:
        raw[-1] = "docs/benchmarks/unreviewed-pilot.json"
        monkeypatch.setattr(sys, "orig_argv", raw, raising=False)
    monkeypatch.setattr(sys, "argv", [raw[1], *raw[2:]])

    if mutation in {"missing", "interpreter"}:
        with pytest.raises(contract.ContractError, match="exact process argv"):
            runner._phase6_exact_supervisor_argv()
    elif mutation == "script":
        with pytest.raises(contract.ContractError, match="different script"):
            runner._phase6_exact_supervisor_argv()
    else:
        assert runner._phase6_exact_supervisor_argv() != expected


@pytest.mark.parametrize("mutation", ["missing", "malformed", "interpreter", "script"])
def test_exact_process_argv_fails_closed(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    benchmark = _load(BENCHMARK_PATH, f"phase6_runtime_repair_argv_{mutation}")
    arguments = ["--phase6-trace-only"]
    monkeypatch.setattr(sys, "argv", [str(BENCHMARK_PATH), *arguments])
    raw: object = [PYTHON, str(BENCHMARK_PATH), *arguments]
    if mutation == "missing":
        monkeypatch.delattr(sys, "orig_argv", raising=False)
    elif mutation == "malformed":
        raw = [PYTHON, 3]
        monkeypatch.setattr(sys, "orig_argv", raw, raising=False)
    elif mutation == "interpreter":
        raw = [str(Path(PYTHON).resolve()), str(BENCHMARK_PATH), *arguments]
        monkeypatch.setattr(sys, "orig_argv", raw, raising=False)
    else:
        raw = [PYTHON, str(RUNNER_PATH), *arguments]
        monkeypatch.setattr(sys, "orig_argv", raw, raising=False)

    with pytest.raises(ValueError, match="exact process argv|different script"):
        benchmark._phase6_exact_process_argv()


def test_zero_return_invalid_child_evidence_is_reason_scoped() -> None:
    command = [PYTHON, str(BENCHMARK_PATH), "--phase6-trace-only"]
    process = _terminal_process(command, returncode=0)

    assert contract.phase6_process_record_valid(
        process,
        terminal_state="failed",
        terminal_reason="invalid_child_evidence",
    )
    assert not contract.phase6_process_record_valid(
        process,
        terminal_state="failed",
        terminal_reason="child_nonzero_exit",
    )

    positive = copy.deepcopy(process)
    positive["returncode"] = 1
    assert contract.phase6_process_record_valid(
        positive,
        terminal_state="failed",
        terminal_reason="child_nonzero_exit",
    )
    assert not contract.phase6_process_record_valid(
        positive,
        terminal_state="failed",
        terminal_reason="invalid_child_evidence",
    )


def test_zero_return_invalid_evidence_terminal_transition_is_valid_once() -> None:
    runner = _load(RUNNER_PATH, "phase6_runtime_repair_zero_return_transition")
    identities = contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA)
    bindings = _bindings(
        identities,
        gate="gate_b",
        ledger_schema=contract.PHASE6_TRACE_SCHEMA,
    )
    identity = identities[0]
    command = bindings["schedule"]["payload"]["records"][0]["child_command_argv"]
    ledger = contract.new_phase6_ledger(
        schema=contract.PHASE6_TRACE_SCHEMA,
        gate="gate_b",
        artifact_kind="trace_census",
        identities=identities,
        bindings=bindings,
    )
    ledger = contract.transition_phase6_record(
        ledger,
        identity_id=identity["identity_id"],
        new_state="running",
        timestamp_utc="2026-07-12T00:00:00+00:00",
        process=_running_process(command),
    )
    evidence = _failure_evidence(classification="common_invalidity")
    ledger = contract.transition_phase6_record(
        ledger,
        identity_id=identity["identity_id"],
        new_state="failed",
        timestamp_utc="2026-07-12T00:00:01+00:00",
        reason="invalid_child_evidence",
        process=_terminal_process(command, returncode=0),
        evidence=evidence,
    )

    record = ledger["records"][0]
    assert record["state"] == "failed"
    assert record["reason"] == "invalid_child_evidence"
    assert record["process"]["returncode"] == 0
    assert record["evidence"]["classification"] == "common_invalidity"
    assert all(contract.phase6_ledger_checks(ledger, final=False).values())
    with pytest.raises(contract.ContractError, match="illegal Phase 6 transition"):
        contract.transition_phase6_record(
            ledger,
            identity_id=identity["identity_id"],
            new_state="interrupted",
            timestamp_utc="2026-07-12T00:00:02+00:00",
            reason="supervisor_recovery",
            process=_terminal_process(command, returncode=0),
            evidence=evidence,
        )

    assert runner._phase6_should_prune(ledger, identities[1]) == "common_invalidity"


def test_real_executor_commits_zero_return_invalid_evidence_once_and_prunes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load(RUNNER_PATH, "phase6_runtime_repair_executor_integration")
    identities = contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA)
    bindings = _bindings(
        identities,
        gate="gate_b",
        ledger_schema=contract.PHASE6_TRACE_SCHEMA,
    )
    launches: list[list[str]] = []
    recoveries: list[object] = []

    monkeypatch.setattr(runner, "_phase6_repo_path", lambda path: Path(path))
    monkeypatch.setattr(runner, "_phase6_prepare_child_artifacts", lambda identity: None)
    monkeypatch.setattr(
        runner,
        "_phase6_record_evidence",
        lambda identity, *, classification, discovery: _failure_evidence(
            classification=classification
        ),
    )

    def recover(process: object) -> object:
        recoveries.append(process)
        raise AssertionError("executor entered recovery after a durable terminal transition")

    monkeypatch.setattr(runner, "_phase6_recover_running_process", recover)

    def managed(
        command: list[str],
        *,
        on_started,
        on_completed,
        **kwargs,
    ):
        del kwargs
        launches.append(list(command))
        on_started(_running_process(command))
        process = _terminal_process(command, returncode=0)
        on_completed(process)
        return process

    monkeypatch.setattr(runner, "run_managed_process_group", managed)
    output = tmp_path / "trace.json"
    payload = runner.phase6_execute_ledger(
        schema=contract.PHASE6_TRACE_SCHEMA,
        output_path=output,
        bindings=bindings,
        child_timeout_seconds=60,
    )

    first = payload["records"][0]
    assert first["state"] == "failed"
    assert first["reason"] == "invalid_child_evidence"
    assert first["process"]["returncode"] == 0
    assert first["evidence"]["classification"] == "common_invalidity"
    assert launches == [
        bindings["schedule"]["payload"]["records"][0]["child_command_argv"]
    ]
    assert recoveries == []
    assert [event["new_state"] for event in payload["events"][:2]] == [
        "running",
        "failed",
    ]
    assert all(
        record["state"] == "not_launched:common_invalidity"
        for record in payload["records"][1:]
    )
    assert all(contract.phase6_ledger_checks(payload, final=True).values())


def test_parent_boundary_drift_closes_budget_with_zero_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load(RUNNER_PATH, "phase6_r3_parent_boundary_drift")
    identities = contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA)
    authority_path = tmp_path / "repair-result.md"
    authority_path.write_text("# Original repair result\n", encoding="ascii")
    authority_blob = contract.phase6_blob_record(authority_path)
    bindings = _bindings(
        identities,
        gate="gate_b",
        ledger_schema=contract.PHASE6_TRACE_SCHEMA,
        authority_inputs=(authority_blob,),
    )
    initial = contract.new_phase6_ledger(
        schema=contract.PHASE6_TRACE_SCHEMA,
        gate="gate_b",
        artifact_kind="trace_census",
        identities=identities,
        bindings=bindings,
    )
    work = tmp_path / "work"
    budget_path = tmp_path / "budget" / "gate-b.json"
    output = tmp_path / "trace.json"
    launches: list[list[str]] = []
    monkeypatch.setattr(runner, "PHASE6_WORK_DIR", work)
    monkeypatch.setattr(runner, "_phase6_repo_path", lambda path: Path(path))
    monkeypatch.setattr(
        runner,
        "run_managed_process_group",
        lambda command, **kwargs: launches.append(list(command)),
    )

    def reject(current: Mapping[str, object]) -> None:
        if contract.phase6_blob_record(authority_path) != current["authority_inputs"][0]:
            raise contract.ContractError("authority input changed at parent boundary")

    with runner.phase6_budget_lease(
        budget_path, "trace_census_and_pilot"
    ) as lease:
        opened = runner.phase6_budget_state_open(
            budget_path,
            bindings["authority_id"],
            "gate_b",
            3045.0,
            "trace_census_and_pilot",
            lease=lease,
        )
        authority_path.write_text("# Mutated after budget open\n", encoding="ascii")
        result = runner.phase6_execute_ledger(
            schema=contract.PHASE6_TRACE_SCHEMA,
            output_path=output,
            bindings=bindings,
            child_timeout_seconds=60,
            authority_validator=reject,
            budget_path=budget_path,
            budget_lease=lease,
            expected_budget_command_name="trace_census_and_pilot",
            gate_hard_ceiling_seconds=3045.0,
            initial_payload=initial,
        )
        runner.phase6_budget_state_close_command(
            budget_path,
            opened,
            "trace_census_and_pilot",
            lease=lease,
        )

    assert launches == []
    assert result["state"] == "failed"
    assert all(
        record["state"] == "not_launched:common_invalidity"
        for record in result["records"]
    )
    assert contract.read_strict_json(output) == result
    assert contract.read_strict_json(budget_path)["state"] == "closed"
    lease_record = contract.read_strict_json(
        budget_path.with_name(f"{budget_path.name}.lease")
    )
    assert lease_record["state"] == "released"
    assert not any(path.is_file() for path in work.rglob("*"))


def test_terminal_authority_drift_preserves_real_process_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load(RUNNER_PATH, "phase6_r3_terminal_authority_drift")
    identities = contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA)
    bindings = _bindings(
        identities, gate="gate_b", ledger_schema=contract.PHASE6_TRACE_SCHEMA
    )
    calls = 0
    monkeypatch.setattr(runner, "PHASE6_WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(runner, "_phase6_repo_path", lambda path: Path(path))
    monkeypatch.setattr(
        runner,
        "_phase6_record_evidence",
        lambda identity, *, classification, discovery: _failure_evidence(
            classification=classification
        ),
    )

    def revalidate(_: Mapping[str, object]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise contract.ContractError("persistent terminal authority drift")

    def managed(command: list[str], *, on_started, on_completed, **kwargs):
        del kwargs
        on_started(_running_process(command))
        process = _terminal_process(command, returncode=7)
        on_completed(process)
        return process

    monkeypatch.setattr(runner, "run_managed_process_group", managed)
    result = runner.phase6_execute_ledger(
        schema=contract.PHASE6_TRACE_SCHEMA,
        output_path=tmp_path / "trace.json",
        bindings=bindings,
        child_timeout_seconds=60,
        authority_validator=revalidate,
    )

    first = result["records"][0]
    assert calls == 2
    assert first["state"] == "failed"
    assert first["reason"] == "authority_revalidation_failed"
    assert first["process"]["returncode"] == 7
    assert first["evidence"]["classification"] == "common_invalidity"
    assert all(
        record["state"] == "not_launched:common_invalidity"
        for record in result["records"][1:]
    )
    assert all(contract.phase6_ledger_checks(result, final=True).values())


def test_child_entry_authority_drift_exits_before_tensorflow_or_target_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load(RUNNER_PATH, "phase6_r3_child_entry_authority_drift")
    identities = contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA)
    authority_path = tmp_path / "repair-result.md"
    original_authority = b"# Original repair result\n"
    authority_path.write_bytes(original_authority)
    bindings = _bindings(
        identities,
        gate="gate_b",
        ledger_schema=contract.PHASE6_TRACE_SCHEMA,
        authority_inputs=(contract.phase6_blob_record(authority_path),),
    )
    for field, path in (
        ("proposal", tmp_path / "proposal.json"),
        ("attestation", tmp_path / "attestation.json"),
    ):
        path.write_bytes(base64.b64decode(bindings[field]["base64"], validate=True))
        bindings[field] = contract.phase6_blob_record(path)
    phase45_path = tmp_path / "phase45.json"
    phase45_path.write_bytes(
        base64.b64decode(bindings["phase45_evidence"][0]["base64"], validate=True)
    )
    bindings["phase45_evidence"][0] = contract.phase6_blob_record(phase45_path)
    assert contract._phase6_bindings_valid(bindings)
    row = bindings["schedule"]["payload"]["records"][0]
    paths = contract._phase6_child_artifact_paths(row["identity"])
    root = Path(contract.PHASE6_WORK_ROOT)
    assert not root.exists() and not root.is_symlink()
    assert all(not path.exists() and not path.is_symlink() for path in paths.values())
    snapshot = contract.phase6_child_authority_snapshot(bindings, row)
    marker = tmp_path / "tensorflow-imported"
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "tensorflow.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported', encoding='ascii')\n"
        "raise RuntimeError('TensorFlow import crossed child authority guard')\n",
        encoding="ascii",
    )
    monkeypatch.setenv("PYTHONPATH", str(poison))
    monkeypatch.setattr(runner, "_phase6_repo_path", lambda path: Path(path))
    launches: list[list[str]] = []
    original_run = runner.run_managed_process_group

    def revalidate(current: Mapping[str, object]) -> None:
        for field in ("proposal", "attestation"):
            blob = current[field]
            if contract.phase6_blob_record(Path(blob["path"])) != blob:
                raise contract.ContractError(f"{field} drifted")
        for field in ("phase45_evidence", "authority_inputs"):
            for blob in current[field]:
                if contract.phase6_blob_record(Path(blob["path"])) != blob:
                    raise contract.ContractError(f"{field} drifted")

    def mutate_after_parent_guard(command: list[str], **kwargs):
        launches.append(list(command))
        authority_path.write_text("# Mutated after parent guard\n", encoding="ascii")
        on_completed = kwargs["on_completed"]

        def restore_before_terminal_guard(process):
            authority_path.write_bytes(original_authority)
            on_completed(process)

        kwargs["on_completed"] = restore_before_terminal_guard
        return original_run(command, **kwargs)

    monkeypatch.setattr(
        runner, "run_managed_process_group", mutate_after_parent_guard
    )
    try:
        result = runner.phase6_execute_ledger(
            schema=contract.PHASE6_TRACE_SCHEMA,
            output_path=tmp_path / "trace.json",
            bindings=bindings,
            child_timeout_seconds=60,
            authority_validator=revalidate,
        )

        assert launches == [row["child_command_argv"]]
        first = result["records"][0]
        assert first["state"] == "failed"
        assert first["reason"] == "invalid_child_evidence"
        assert first["process"]["returncode"] == 0
        assert first["evidence"]["classification"] == "common_invalidity"
        assert all(
            record["state"] == "not_launched:common_invalidity"
            for record in result["records"][1:]
        )
        assert authority_path.read_bytes() == original_authority
        assert not marker.exists()
        failure = contract.read_strict_json(paths["artifact"])
        assert failure["schema"] == contract.PHASE6_CHILD_AUTHORITY_FAILURE_SCHEMA
        assert failure["stage"] == "child_entry_authority_guard"
        assert failure["command_argv"] == row["child_command_argv"]
        assert failure["target_work"] == {
            "tensorflow_imported": False,
            "fixture_constructed": False,
            "selected_method_constructed": False,
            "trace_requested": False,
            "xla_requested": False,
            "kalman_invocations": 0,
        }
        assert paths["journal"].is_file()
        assert not paths["sidecar"].exists()
        assert not paths["dependency_before"].exists()
        assert not paths["dependency_after"].exists()
    finally:
        for path in paths.values():
            path.unlink(missing_ok=True)
        paths["artifact"].parent.rmdir()
        root.rmdir()


def test_historical_namespaces_and_active_r3_are_disjoint_and_stale_evidence_is_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load(RUNNER_PATH, "phase6_runtime_repair_disjoint_paths")
    r1_paths = {str(record[0]) for record in contract.PHASE6_R1_ARCHIVE_FILE_SPECS}
    assert len(
        {
            contract.PHASE6_R1_WORK_ROOT,
            contract.PHASE6_R2_WORK_ROOT,
            contract.PHASE6_WORK_ROOT,
        }
    ) == 3
    assert all(contract.PHASE6_WORK_ROOT not in path for path in r1_paths)
    assert all(
        "gateb_r3" in relative
        for relative in (
            contract.PHASE6_GATE_B_BUDGET_RELATIVE,
            contract.PHASE6_GATE_B_ATTESTATION_RELATIVE,
            *contract.PHASE6_GATE_B_ARTIFACTS.values(),
        )
    )
    for schema in (contract.PHASE6_TRACE_SCHEMA, contract.PHASE6_PILOT_SCHEMA):
        schedule = runner.phase6_build_schedule(
            schema, gate="gate_b", child_timeout_seconds=60
        )
        assert all(
            contract.PHASE6_WORK_ROOT in " ".join(row["child_command_argv"])
            for row in schedule["records"]
        )

    monkeypatch.setattr(runner, "PHASE6_WORK_DIR", tmp_path / "r3")
    identity = contract.phase6_expected_roster(contract.PHASE6_TRACE_SCHEMA)[0]
    artifact = runner._phase6_child_paths(identity)["artifact"]
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"stale-r2-evidence\n")
    before = artifact.read_bytes()
    with pytest.raises(contract.ContractError, match="stale Phase 6 child evidence"):
        runner._phase6_prepare_child_artifacts(identity)
    assert artifact.read_bytes() == before


def test_gate_b_inputs_require_exact_archive_result_and_reviews(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = tmp_path / "archive.json"
    result = tmp_path / "result.md"
    plan = tmp_path / "plan.md"
    plan_review = tmp_path / "plan-review.md"
    result_review = tmp_path / "result-review.md"
    archive.write_text("{}\n", encoding="ascii")
    result.write_text("repair result\n", encoding="ascii")
    plan.write_text("repair plan\n", encoding="ascii")
    plan_review.write_text("plan review\n", encoding="ascii")
    result_review.write_text(
        "\n".join(
            [
                "Review strength: `codex_substitute_weaker`",
                f"RESULT_PATH: {result.resolve()}",
                f"RESULT_SHA256: {contract.file_sha256(result)}",
                f"PLAN_PATH: {plan.resolve()}",
                f"PLAN_SHA256: {contract.file_sha256(plan)}",
                "VERDICT: AGREE",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    monkeypatch.setattr(contract, "PHASE6_R2_ARCHIVE_RELATIVE", str(archive))
    monkeypatch.setattr(contract, "PHASE6_REPAIR_RESULT_RELATIVE", str(result))
    monkeypatch.setattr(contract, "PHASE6_PLAN_RELATIVE", str(plan))
    monkeypatch.setattr(contract, "PHASE6_PLAN_SHA256", contract.file_sha256(plan))
    monkeypatch.setattr(contract, "PHASE6_PLAN_REVIEW_RELATIVE", str(plan_review))
    monkeypatch.setattr(
        contract, "PHASE6_PLAN_REVIEW_SHA256", contract.file_sha256(plan_review)
    )
    monkeypatch.setattr(
        contract, "PHASE6_REPAIR_RESULT_REVIEW_RELATIVE", str(result_review)
    )
    monkeypatch.setattr(
        contract,
        "PHASE6_GATE_B_INPUT_RELATIVES",
        (str(archive), str(result), str(plan_review), str(result_review)),
    )
    monkeypatch.setattr(contract, "validate_phase6_r2_archive", lambda payload: None)

    records = contract.phase6_gate_b_input_records(repo_root=tmp_path)
    assert contract.phase6_gate_b_inputs_valid(records)
    for index in range(4):
        changed = copy.deepcopy(records)
        changed[index]["sha256"] = "0" * 64
        assert not contract.phase6_gate_b_inputs_valid(changed)
    result_review.write_text(
        result_review.read_text(encoding="ascii").replace(
            "VERDICT: AGREE", "VERDICT: REVISE"
        ),
        encoding="ascii",
    )
    assert not contract.phase6_gate_b_inputs_valid(
        contract.phase6_gate_b_input_records(repo_root=tmp_path)
    )


def test_closed_attestation_cli_writes_once_and_binds_review_strength(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load(RUNNER_PATH, "phase6_runtime_repair_attestation_cli")
    proposal_path = tmp_path / "proposal.json"
    review_path = tmp_path / "review.md"
    output_path = tmp_path / "attestation.json"
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("reviewed plan\n", encoding="ascii")
    proposal = {
        "authority_id": "a" * 64,
        "plan": contract.path_digest_record(plan_path),
    }
    contract.durable_atomic_write_json(proposal_path, proposal)
    review_path.write_text(
        "Review strength: `codex_substitute_weaker`\nVERDICT: AGREE\n",
        encoding="ascii",
    )
    paths = {
        "Gate B proposal": proposal_path,
        "Gate B review": review_path,
    }
    monkeypatch.setattr(
        runner, "_phase6_resolve_authority_path", lambda path, label: paths[label]
    )
    monkeypatch.setattr(runner, "_phase6_repo_path", lambda path: output_path)
    monkeypatch.setattr(
        contract, "validate_phase6_budget_proposal", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        contract,
        "phase6_attestation_checks",
        lambda *args, **kwargs: {"synthetic_closed_check": True},
    )
    validations: list[tuple[Path, Path, str]] = []
    monkeypatch.setattr(
        contract,
        "validate_phase6_runtime_authority",
        lambda proposal, attestation, *, expected_gate: validations.append(
            (proposal, attestation, expected_gate)
        ),
    )
    expected = [
        "--phase6-create-attestation",
        "gate_b",
        "--budget-contract",
        contract.PHASE6_GATE_B_BUDGET_RELATIVE,
        "--review-path",
        contract.PHASE6_GATE_B_REVIEW_RELATIVE,
        "--output-json",
        contract.PHASE6_GATE_B_ATTESTATION_RELATIVE,
    ]
    monkeypatch.setattr(sys, "argv", [str(RUNNER_PATH), *expected])
    args = SimpleNamespace(
        phase6_create_attestation="gate_b",
        budget_contract=Path(contract.PHASE6_GATE_B_BUDGET_RELATIVE),
        review_path=Path(contract.PHASE6_GATE_B_REVIEW_RELATIVE),
        output_json=Path(contract.PHASE6_GATE_B_ATTESTATION_RELATIVE),
    )

    assert runner.run_phase6_create_attestation(args) == 0
    attestation = contract.read_strict_json(output_path)
    assert attestation["authority_id"] == proposal["authority_id"]
    assert attestation["review_strength"] == "codex_substitute_weaker"
    assert attestation["proposal"] == contract.path_digest_record(proposal_path)
    assert attestation["review"] == contract.path_digest_record(review_path)
    assert validations == [(proposal_path, output_path, "gate_b")]
    before = output_path.read_bytes()
    with pytest.raises(contract.ContractError, match="strictly absent"):
        runner.run_phase6_create_attestation(args)
    assert output_path.read_bytes() == before


def test_closed_authority_validation_cli_checks_absence_and_live_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load(RUNNER_PATH, "phase6_runtime_repair_validate_authority_cli")
    proposal_path = tmp_path / "proposal.json"
    attestation_path = tmp_path / "attestation.json"
    proposal_path.write_text("{}\n", encoding="ascii")
    attestation_path.write_text("{}\n", encoding="ascii")
    paths = {
        "Gate B proposal": proposal_path,
        "Gate B attestation": attestation_path,
    }
    monkeypatch.setattr(
        runner, "_phase6_resolve_authority_path", lambda path, label: paths[label]
    )
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "PHASE6_WORK_DIR", tmp_path / "work")
    discovery_path = tmp_path / "work" / "import_discovery.json"
    discovery_path.parent.mkdir()
    discovery = {"schema": "synthetic-import-discovery"}
    contract.durable_atomic_write_json(discovery_path, discovery)
    monkeypatch.setattr(
        contract, "PHASE6_IMPORT_DISCOVERY_OUTPUT", str(discovery_path)
    )
    validations: list[tuple[Path, Path, str]] = []
    monkeypatch.setattr(
        contract,
        "validate_phase6_runtime_authority",
        lambda proposal, attestation, *, expected_gate: (
            validations.append((proposal, attestation, expected_gate))
            or ({"dependency_discovery": discovery}, {})
        ),
    )
    monkeypatch.setattr(runner, "_phase6_live_target_pids", lambda: [])
    expected = [
        "--phase6-validate-authority",
        "gate_b",
        "--budget-contract",
        contract.PHASE6_GATE_B_BUDGET_RELATIVE,
        "--budget-attestation",
        contract.PHASE6_GATE_B_ATTESTATION_RELATIVE,
    ]
    monkeypatch.setattr(sys, "argv", [str(RUNNER_PATH), *expected])
    args = SimpleNamespace(
        phase6_validate_authority="gate_b",
        budget_contract=Path(contract.PHASE6_GATE_B_BUDGET_RELATIVE),
        budget_attestation=Path(contract.PHASE6_GATE_B_ATTESTATION_RELATIVE),
    )

    assert runner.run_phase6_validate_authority(args) == 0
    assert validations == [(proposal_path, attestation_path, "gate_b")]
    monkeypatch.setattr(runner, "_phase6_live_target_pids", lambda: [4242])
    with pytest.raises(contract.ContractError, match="target worker is live"):
        runner.run_phase6_validate_authority(args)


@pytest.mark.parametrize(
    "mutation", ["missing_root", "wrong_discovery", "extra_file", "trace_output"]
)
def test_fresh_gate_b_namespace_rejects_prelaunch_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    runner = _load(RUNNER_PATH, f"phase6_runtime_repair_namespace_{mutation}")
    work = tmp_path / "work"
    discovery_path = work / "import_discovery.json"
    discovery = {"schema": "synthetic-import-discovery"}
    if mutation != "missing_root":
        work.mkdir()
        contract.durable_atomic_write_json(discovery_path, discovery)
    trace = tmp_path / "trace.json"
    pilot = tmp_path / "pilot.json"
    if mutation == "wrong_discovery":
        contract.durable_atomic_write_json(
            discovery_path, {"schema": "wrong-discovery"}
        )
    elif mutation == "extra_file":
        (work / "stale-child.json").write_text("{}\n", encoding="ascii")
    elif mutation == "trace_output":
        trace.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(runner, "PHASE6_WORK_DIR", work)
    monkeypatch.setattr(contract, "PHASE6_IMPORT_DISCOVERY_OUTPUT", str(discovery_path))
    monkeypatch.setattr(runner, "_phase6_live_target_pids", lambda: [])

    with pytest.raises(contract.ContractError, match="strictly absent|namespace|unexpected"):
        runner._phase6_assert_fresh_gate_b_namespace(
            {"dependency_discovery": discovery},
            trace_output=trace,
            pilot_output=pilot,
        )


def test_proposal_work_root_must_be_strictly_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load(RUNNER_PATH, "phase6_runtime_repair_proposal_root_absence")
    root = tmp_path / "r3"
    monkeypatch.setattr(runner, "PHASE6_WORK_DIR", root)
    runner._phase6_assert_gate_b_work_root_absent()
    root.mkdir()
    with pytest.raises(contract.ContractError, match="strictly absent"):
        runner._phase6_assert_gate_b_work_root_absent()


def test_r1_archive_rehashes_complete_lineage_and_localizes_argv() -> None:
    runner = _load(RUNNER_PATH, "phase6_runtime_repair_archive")
    archive = runner.build_phase6_r1_archive()

    assert len(archive["files"]) == len(contract.PHASE6_R1_ARCHIVE_FILE_SPECS) == 17
    assert all(contract.phase6_r1_archive_checks(archive).values())
    assert archive["diagnosis"]["argv_differences"] == [
        {
            "index": 0,
            "schedule": PYTHON,
            "child": str(Path(PYTHON).resolve()),
        }
    ]
    assert archive["diagnosis"]["other_argv_elements_equal"] is True
    assert archive["diagnosis"]["full_child_validity_recomputed"] is False
    assert archive["no_live_process"]["matching_pids"] == []


@pytest.mark.parametrize(
    "mutation",
    ["file_digest", "argv_difference", "classification", "live_process", "extra_field"],
)
def test_r1_archive_validator_rejects_independent_mutations(mutation: str) -> None:
    runner = _load(RUNNER_PATH, f"phase6_runtime_repair_archive_{mutation}")
    archive = runner.build_phase6_r1_archive()
    if mutation == "file_digest":
        archive["files"][0]["sha256"] = "0" * 64
    elif mutation == "argv_difference":
        archive["diagnosis"]["argv_differences"][0]["schedule"] = str(
            Path(PYTHON).resolve()
        )
    elif mutation == "classification":
        archive["diagnosis"]["classification"] = "method_failure"
    elif mutation == "live_process":
        archive["no_live_process"] = {
            "scan": "proc_cmdline_exact_supervisor_target_modes",
            "matching_pids": [123],
            "passed": False,
        }
    else:
        archive["unreviewed"] = True

    with pytest.raises(contract.ContractError, match="invalid Phase 6 r1 archive"):
        contract.validate_phase6_r1_archive(archive)


def test_r1_archive_builder_rejects_live_target_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load(RUNNER_PATH, "phase6_runtime_repair_archive_live")
    monkeypatch.setattr(runner, "_phase6_live_target_pids", lambda: [4242])

    with pytest.raises(contract.ContractError, match="invalid Phase 6 r1 archive"):
        runner.build_phase6_r1_archive()


def test_r1_archive_cli_refuses_nonidentical_existing_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load(RUNNER_PATH, "phase6_runtime_repair_archive_overwrite")
    archive = runner.build_phase6_r1_archive()
    output = tmp_path / "archive.json"
    output.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(contract, "PHASE6_R1_ARCHIVE_RELATIVE", "archive.json")
    monkeypatch.setattr(runner, "_phase6_repo_path", lambda path: output)
    monkeypatch.setattr(runner, "build_phase6_r1_archive", lambda: archive)
    monkeypatch.setattr(sys, "argv", [str(RUNNER_PATH), "--phase6-archive-r1", "--output-json", "archive.json"])
    args = type("Args", (), {"output_json": Path("archive.json")})()

    with pytest.raises(contract.ContractError, match="invalid Phase 6 r1 archive"):
        runner.run_phase6_archive_r1(args)
    assert output.read_bytes() == b"{}\n"


def test_r2_archive_rehashes_exact_failed_generation_without_mutation() -> None:
    runner = _load(RUNNER_PATH, "phase6_r3_r2_archive")
    before = {
        record[0]: contract.file_sha256(Path(record[0]))
        for record in contract.PHASE6_R2_ARCHIVE_FILE_SPECS
        if Path(record[0]).is_absolute()
    }
    archive = runner.build_phase6_r2_archive()

    assert len(archive["files"]) == len(contract.PHASE6_R2_ARCHIVE_FILE_SPECS) == 13
    assert all(contract.phase6_r2_archive_checks(archive).values())
    assert archive["work_root_entries"] == ["budget_state", "import_discovery.json"]
    assert archive["diagnosis"]["mixed_format_inputs"] == [
        "json",
        "markdown",
        "markdown",
        "markdown",
    ]
    assert archive["no_live_process"]["passed"] is True
    assert before == {
        path: contract.file_sha256(Path(path)) for path in before
    }


@pytest.mark.parametrize(
    "mutation",
    ("file_digest", "absence", "inventory", "diagnosis", "live_process", "extra"),
)
def test_r2_archive_validator_rejects_independent_mutations(mutation: str) -> None:
    runner = _load(RUNNER_PATH, f"phase6_r3_r2_archive_{mutation}")
    archive = runner.build_phase6_r2_archive()
    if mutation == "file_digest":
        archive["files"][0]["sha256"] = "0" * 64
    elif mutation == "absence":
        archive["absent_paths"] = archive["absent_paths"][:-1]
    elif mutation == "inventory":
        archive["work_root_entries"].append("trace")
    elif mutation == "diagnosis":
        archive["diagnosis"]["target_trace_requested"] = True
    elif mutation == "live_process":
        archive["no_live_process"] = {
            "scan": "proc_cmdline_exact_supervisor_target_modes",
            "matching_pids": [123],
            "passed": False,
        }
    else:
        archive["unreviewed"] = True

    with pytest.raises(contract.ContractError, match="invalid Phase 6 r2 archive"):
        contract.validate_phase6_r2_archive(archive)
