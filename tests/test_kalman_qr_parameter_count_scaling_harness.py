from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import kalman_qr_benchmark_contract as contract
from tests.test_kalman_qr_benchmark_contract import _measurement


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "scripts/benchmark_kalman_qr_parameter_count_scaling.py"
RUNNER_PATH = ROOT / "docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("selected", contract.METHOD_IDS)
def test_selected_builder_dispatches_only_selected_method(monkeypatch, selected: str) -> None:
    benchmark = _load(BENCHMARK_PATH, "kalman_qr_benchmark_harness")
    calls = {method: 0 for method in contract.METHOD_IDS}

    def builder(method):
        def build():
            calls[method] += 1
            return object()

        return build

    registry = {method: builder(method) for method in contract.METHOD_IDS}
    result, ledger = benchmark._selected_method_builder(
        selected,
        fixture=object(),
        batch_size=1,
        jit_compile=False,
        builder_registry=registry,
    )
    assert result is not None
    assert ledger == [selected]
    assert calls[selected] == 1
    assert all(count == (1 if method == selected else 0) for method, count in calls.items())


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dimensions=[2],
        parameter_counts=[2],
        timesteps=2,
        batch_size=1,
        dtype="float32",
        device="cpu",
        cpu_threads=1,
        repeats=1,
        timeout_seconds=1.0,
        methods=[contract.PRIMARY_METHOD_IDS[0]],
        output_dir=tmp_path,
        harness_contract_test_only=True,
        no_resume=True,
        jit_compile=False,
        tf32_enabled=True,
    )


def test_child_command_names_exactly_one_method(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "kalman_qr_runner_command")
    args = _args(tmp_path)
    schedule = runner.build_schedule(args)
    identity = schedule["expected_identities"][0]
    attempt_id, progress = contract.new_attempt(tmp_path / "progress", identity["case_id"], identity["method_id"])
    command = runner._child_command(
        args,
        identity=identity,
        attempt_id=attempt_id,
        progress_path=progress,
        output_json=tmp_path / "out.json",
        output_md=tmp_path / "out.md",
        schedule_fingerprint=schedule["schedule_fingerprint"],
    )
    assert command.count("--method") == 1
    assert command[command.index("--method") + 1] == identity["method_id"]
    assert all(method not in command for method in set(contract.METHOD_IDS) - {identity["method_id"]})


@pytest.mark.parametrize(
    ("completed", "expected_state"),
    [
        (subprocess.CompletedProcess([], -9, "", "crash"), "crashed"),
        (subprocess.CompletedProcess([], 1, "", "failure"), "failed"),
    ],
)
def test_runner_synthesizes_failure_without_erasing_sibling(
    tmp_path: Path, completed, expected_state
) -> None:
    runner = _load(RUNNER_PATH, f"kalman_qr_runner_{expected_state}")
    args = _args(tmp_path)
    schedule = runner.build_schedule(args)
    identity = schedule["expected_identities"][0]
    sibling = tmp_path / "sibling.json"
    contract.atomic_write_json(sibling, {"state": "preserved"})
    before = sibling.read_bytes()

    def fake_run(command, **kwargs):
        progress = Path(command[command.index("--progress-journal") + 1])
        attempt_id = command[command.index("--attempt-id") + 1]
        expected = runner.expected_progress_identity(
            identity, attempt_id, schedule["schedule_fingerprint"]
        )
        contract.append_progress_event(progress, {**expected, "stage": "trace"})
        return completed

    record, decision = runner.run_identity(
        args,
        identity=identity,
        schedule_fingerprint=schedule["schedule_fingerprint"],
        progress_dir=tmp_path / "progress",
        runner=fake_run,
    )
    assert decision.endswith("executed_synthesized_record")
    assert record["state"] == expected_state
    assert record["last_entered_stage"] == "trace"
    assert sibling.read_bytes() == before


def test_runner_synthesizes_timeout_at_exact_stage(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "kalman_qr_runner_timeout")
    args = _args(tmp_path)
    schedule = runner.build_schedule(args)
    identity = schedule["expected_identities"][0]

    def fake_run(command, **kwargs):
        progress = Path(command[command.index("--progress-journal") + 1])
        attempt_id = command[command.index("--attempt-id") + 1]
        expected = runner.expected_progress_identity(
            identity, attempt_id, schedule["schedule_fingerprint"]
        )
        contract.append_progress_event(progress, {**expected, "stage": "first_executable_call"})
        raise subprocess.TimeoutExpired(command, timeout=1.0, stderr="timeout")

    record, decision = runner.run_identity(
        args,
        identity=identity,
        schedule_fingerprint=schedule["schedule_fingerprint"],
        progress_dir=tmp_path / "progress",
        runner=fake_run,
    )
    assert decision.endswith("executed_timeout")
    assert record["state"] == "timed_out"
    assert record["last_entered_stage"] == "first_executable_call"


def test_aggregate_parity_requires_all_sibling_outputs() -> None:
    runner = _load(RUNNER_PATH, "kalman_qr_runner_parity")
    common = {
        "case_id": (
            "dimension=2-parameter_count=2-timesteps=2-batch_size=1-"
            "dtype=float32-device=cpu"
        ),
        "schema": contract.SCHEMA,
        "state": "passed",
        "output_metadata": {
            "all_finite": True,
            "value_shape": [1],
            "score_shape": [1, 2],
            "value_dtype": "float32",
            "score_dtype": "float32",
        },
    }
    records = [
        {**common, "method_id": contract.METHOD_IDS[0], "outputs": {"value": [1.0], "score": [[2.0]]}},
        {**common, "method_id": contract.METHOD_IDS[1], "outputs": None},
    ]
    assert runner._comparator_parity(records) is False
    records[1]["outputs"] = {"value": [1.0], "score": [[2.0]]}
    assert runner._comparator_parity(records) is True
    records[1]["outputs"] = {"value": [1.1], "score": [[2.0]]}
    assert runner._comparator_parity(records) is False


def _parity_record(method_id: str, *, dtype: str = "float64") -> dict[str, object]:
    return {
        "case_id": (
            "dimension=2-parameter_count=1-timesteps=2-batch_size=1-"
            f"dtype={dtype}-device=cpu"
        ),
        "method_id": method_id,
        "schema": contract.SCHEMA,
        "method_contract_version": contract.METHOD_CONTRACT_VERSION,
        "state": "passed",
        "output_metadata": {
            "all_finite": True,
            "value_shape": [1],
            "score_shape": [1, 1],
            "value_dtype": dtype,
            "score_dtype": dtype,
        },
        "outputs": {"value": [10.0], "score": [[10.0]]},
    }


@pytest.mark.parametrize(
    ("candidate_id", "reference_id"),
    [
        (
            "batch_native_autodiff_qr_score",
            "batch_native_analytical_qr_score",
        ),
        ("scalar_analytical_row_loop", "batch_native_analytical_qr_score"),
        ("autodiff_row_loop_qr_score", "batch_native_autodiff_qr_score"),
    ],
)
@pytest.mark.parametrize("field", ["value", "score"])
def test_directed_parity_mapping_boundaries(candidate_id: str, reference_id: str, field: str) -> None:
    runner = _load(RUNNER_PATH, f"parity_boundary_{candidate_id}_{field}")
    reference = _parity_record(reference_id)
    candidate = _parity_record(candidate_id)
    tolerance = runner._TOLERANCES["float64"][field]
    limit = tolerance["atol"] + tolerance["rtol"] * 10.0

    assert runner._record_pair_matches(candidate, reference) is True
    if field == "value":
        candidate["outputs"][field][0] = 10.0 + limit * (1.0 - 1.0e-6)
    else:
        candidate["outputs"][field][0][0] = 10.0 + limit * (1.0 - 1.0e-6)
    assert runner._record_pair_matches(candidate, reference) is True
    if field == "value":
        candidate["outputs"][field][0] = 10.0 + limit * (1.0 + 1.0e-6)
    else:
        candidate["outputs"][field][0][0] = 10.0 + limit * (1.0 + 1.0e-6)
    assert runner._record_pair_matches(candidate, reference) is False


def test_directed_parity_uses_analytical_reference_magnitude() -> None:
    runner = _load(RUNNER_PATH, "parity_direction")
    tolerance = {"rtol": 0.1, "atol": 0.0}
    assert runner._directed_allclose(9.0, 10.0, **tolerance) is True
    assert runner._directed_allclose(10.0, 9.0, **tolerance) is False
    assert runner._directed_allclose(-9.0, -10.0, **tolerance) is True


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_directed_parity_rejects_nonfinite(value: float) -> None:
    runner = _load(RUNNER_PATH, f"parity_nonfinite_{str(value)}")
    assert runner._directed_allclose(value, 1.0, rtol=1.0, atol=1.0) is False
    assert runner._directed_allclose(1.0, value, rtol=1.0, atol=1.0) is False


@pytest.mark.parametrize(
    "mutation",
    ["missing_output", "value_shape", "score_shape", "dtype", "actual_shape"],
)
def test_record_pair_rejects_output_metadata_and_shape_mutations(mutation: str) -> None:
    runner = _load(RUNNER_PATH, f"parity_metadata_{mutation}")
    reference = _parity_record("batch_native_analytical_qr_score")
    candidate = _parity_record("batch_native_autodiff_qr_score")
    if mutation == "missing_output":
        candidate["outputs"] = None
    elif mutation == "value_shape":
        candidate["output_metadata"]["value_shape"] = [2]
    elif mutation == "score_shape":
        candidate["output_metadata"]["score_shape"] = [1, 2]
    elif mutation == "dtype":
        candidate["output_metadata"]["score_dtype"] = "float32"
    elif mutation == "actual_shape":
        candidate["outputs"]["score"] = [[10.0, 10.0]]
    assert runner._record_pair_matches(candidate, reference) is False


def test_resume_reuses_only_exact_passed_method_record(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "kalman_qr_runner_resume")
    args = _args(tmp_path)
    args.no_resume = False
    schedule = runner.build_schedule(args)
    identity = schedule["expected_identities"][0]
    fingerprints = runner._fingerprints(identity, schedule["schedule_fingerprint"])
    path = runner.method_artifact_path(tmp_path, identity)
    sidecar_path = path.with_suffix(".payload.json")
    measurement = _measurement()
    measurement["payload_sidecar"]["path"] = str(sidecar_path)
    record = {
        "schema": contract.SCHEMA,
        "method_contract_version": contract.METHOD_CONTRACT_VERSION,
        "case_id": identity["case_id"],
        "method_id": identity["method_id"],
        **fingerprints,
        "resume_key": contract.resume_key(
            case_identity=identity["case_id"],
            method_id=identity["method_id"],
            fingerprints=fingerprints,
        ),
        "state": "passed",
        "attempt_id": "attempt-a",
        "last_entered_stage": "envelope_write",
        "terminal_stage": "envelope_write",
        "failure_stage": None,
        "invoked_method_ids": [identity["method_id"]],
        "measurement": measurement,
        "output_metadata": {
            "all_finite": True,
            "value_shape": [1],
            "score_shape": [1, 2],
            "value_dtype": "float32",
            "score_dtype": "float32",
        },
        "outputs": {"value": [1.0], "score": [[1.0, 2.0]]},
    }
    sidecar_payload = {
        "case_id": record["case_id"],
        "method_id": record["method_id"],
        "output_metadata": record["output_metadata"],
        "outputs": record["outputs"],
        "graphdef": measurement["graphdef"],
        "direct_output_parity": measurement["direct_output_parity"],
    }
    contract.atomic_write_json(sidecar_path, sidecar_payload)
    measurement["payload_sidecar"]["sha256"] = contract.file_sha256(sidecar_path)
    contract.atomic_write_json(path, record)

    def forbidden_run(*args, **kwargs):
        raise AssertionError("exact passed record should be reused")

    reused, decision = runner.run_identity(
        args,
        identity=identity,
        schedule_fingerprint=schedule["schedule_fingerprint"],
        progress_dir=tmp_path / "progress",
        runner=forbidden_run,
    )
    assert decision == "reusable_exact_match"
    assert reused == record

    sidecar_path.write_text('{"corrupt":true}\n', encoding="utf-8")
    rejected, reason = runner._read_reusable_record(
        path,
        identity=identity,
        schedule_fingerprint=schedule["schedule_fingerprint"],
    )
    assert rejected is None
    assert reason == "payload_sidecar_invalid"


def test_historical_v1_and_nonfinite_resume_inputs_are_rejected(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "kalman_qr_runner_reject")
    args = _args(tmp_path)
    schedule = runner.build_schedule(args)
    identity = schedule["expected_identities"][0]
    path = runner.method_artifact_path(tmp_path, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema":"v1","value":NaN}', encoding="utf-8")
    record, reason = runner._read_reusable_record(
        path,
        identity=identity,
        schedule_fingerprint=schedule["schedule_fingerprint"],
    )
    assert record is None
    assert reason.startswith("strict_json_rejected:")


def test_stale_method_artifact_is_not_accepted_as_current_attempt(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "kalman_qr_runner_stale")
    args = _args(tmp_path)
    schedule = runner.build_schedule(args)
    identity = schedule["expected_identities"][0]
    stale_path = runner.method_artifact_path(tmp_path, identity)
    contract.atomic_write_json(stale_path, {"attempt_id": "stale", "state": "passed"})

    def fake_crash(command, **kwargs):
        return subprocess.CompletedProcess(command, -9, "", "crashed")

    record, decision = runner.run_identity(
        args,
        identity=identity,
        schedule_fingerprint=schedule["schedule_fingerprint"],
        progress_dir=tmp_path / "progress",
        runner=fake_crash,
    )
    assert decision.endswith("executed_synthesized_record")
    assert record["state"] == "crashed"
    assert record["attempt_id"] != "stale"


@pytest.mark.parametrize("stage", contract.STAGES)
@pytest.mark.parametrize("mode", ["hard_exit", "timeout"])
def test_real_subprocess_stage_journal_recovery(tmp_path: Path, stage: str, mode: str) -> None:
    attempt_id, progress = contract.new_attempt(
        tmp_path / f"progress-{stage}-{mode}", "case-a", contract.METHOD_IDS[0]
    )
    event = {
        "attempt_id": attempt_id,
        "case_id": "case-a",
        "method_id": contract.METHOD_IDS[0],
        "stage": stage,
        "resume_key": "resume-hash",
        **{field: f"hash-{field}" for field in contract.FINGERPRINT_FIELDS},
    }
    expected = dict(event)
    expected.pop("stage")
    code = (
        "import os,sys,time; "
        "from scripts import kalman_qr_benchmark_contract as c; "
        "event=c.strict_json_loads(sys.argv[2]); "
        "c.append_progress_event(__import__('pathlib').Path(sys.argv[1]),event); "
        + ("os._exit(7)" if mode == "hard_exit" else "time.sleep(5)")
    )
    command = [
        sys.executable,
        "-c",
        code,
        str(progress),
        contract.strict_json_dumps(event),
    ]
    if mode == "hard_exit":
        completed = subprocess.run(command, cwd=ROOT, check=False)
        assert completed.returncode == 7
        timed_out = False
        returncode = completed.returncode
    else:
        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(command, cwd=ROOT, check=False, timeout=0.15)
        timed_out = True
        returncode = None
    record = contract.synthesize_process_record(
        identity=expected,
        progress_path=progress,
        timed_out=timed_out,
        returncode=returncode,
        error_tail=mode,
    )
    assert record["last_entered_stage"] == stage
    assert record["state"] == ("timed_out" if timed_out else "failed")


def test_atomic_target_survives_hard_exit_before_replace(tmp_path: Path) -> None:
    target = tmp_path / "status.json"
    contract.atomic_write_json(target, {"state": "valid"})
    before = target.read_bytes()
    code = (
        "import os,sys; from pathlib import Path; "
        "p=Path(sys.argv[1]); t=p.with_name('.'+p.name+'.partial.tmp'); "
        "h=t.open('x'); h.write('partial'); h.flush(); os.fsync(h.fileno()); os._exit(9)"
    )
    completed = subprocess.run([sys.executable, "-c", code, str(target)], check=False)
    assert completed.returncode == 9
    assert target.read_bytes() == before


def test_schedule_identity_stability_covers_config_and_fixture(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "kalman_qr_runner_stability")
    args = _args(tmp_path)
    schedule = runner.build_schedule(args)
    assert runner._schedule_identity_stable(args, schedule) is True
    args.repeats = 2
    assert runner._schedule_identity_stable(args, schedule) is False
    args.repeats = 1
    args.timesteps = 3
    assert runner._schedule_identity_stable(args, schedule) is False


def test_gpu_method_failure_produces_nonzero_top_level_exit(tmp_path: Path, monkeypatch) -> None:
    runner = _load(RUNNER_PATH, "kalman_qr_runner_gpu_failure")
    args = _args(tmp_path)
    args.device = "gpu"
    args.methods = [contract.METHOD_IDS[0]]
    schedule = runner.build_schedule(args)
    identity = schedule["expected_identities"][0]
    fingerprints = runner._fingerprints(identity, schedule["schedule_fingerprint"])
    failed_record = {
        "schema": contract.SCHEMA,
        "method_contract_version": contract.METHOD_CONTRACT_VERSION,
        "case_id": identity["case_id"],
        "method_id": identity["method_id"],
        **fingerprints,
        "resume_key": contract.resume_key(
            case_identity=identity["case_id"],
            method_id=identity["method_id"],
            fingerprints=fingerprints,
        ),
        "attempt_id": "attempt-gpu",
        "state": "failed",
        "last_entered_stage": "first_executable_call",
        "terminal_stage": "first_executable_call",
        "failure_stage": "first_executable_call",
        "invoked_method_ids": [identity["method_id"]],
        "output_metadata": None,
        "outputs": None,
    }
    monkeypatch.setattr(runner, "build_schedule", lambda _args: schedule)
    monkeypatch.setattr(runner, "_schedule_identity_stable", lambda _args, _schedule: True)
    monkeypatch.setattr(
        runner,
        "run_identity",
        lambda *args, **kwargs: (failed_record, "synthetic_gpu_failure"),
    )
    payload, returncode = runner.execute_schedule(args)
    assert payload["status"] == "complete_with_failures"
    assert returncode == 1


def test_record_fingerprint_corruption_is_structural_failure(tmp_path: Path) -> None:
    runner = _load(RUNNER_PATH, "kalman_qr_runner_record_integrity")
    args = _args(tmp_path)
    schedule = runner.build_schedule(args)
    identity = schedule["expected_identities"][0]
    record = {
        "schema": contract.SCHEMA,
        "case_id": identity["case_id"],
        "method_id": identity["method_id"],
        "attempt_id": "attempt-a",
        "state": "failed",
        "invoked_method_ids": [],
        "resume_key": "wrong",
        **runner._fingerprints(identity, schedule["schedule_fingerprint"]),
    }
    checks = runner._aggregate_checks(schedule, [record])
    assert checks["record_integrity"] is False
