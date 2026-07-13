from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

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


@pytest.fixture(scope="module")
def benchmark():
    return _load(BENCHMARK_PATH, "kalman_qr_phase5_measurement_benchmark")


@pytest.fixture(scope="module")
def runner():
    return _load(RUNNER_PATH, "kalman_qr_phase5_measurement_runner")


def test_timed_stage_records_monotonic_exact_interval(benchmark) -> None:
    ticks = iter((10, 25))
    events = []
    result, seconds = benchmark._timed_stage(
        events,
        "fixture",
        lambda: "result",
        clock_ns=lambda: next(ticks),
    )
    assert result == "result"
    assert seconds == 15.0e-9
    assert events == [
        {
            "sequence_index": 0,
            "stage": "fixture",
            "entered_ns": 10,
            "finished_ns": 25,
        }
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_stage",
        "wrong_index",
        "negative",
        "reverse_interval",
        "reverse_sequence_time",
        "missing_event",
        "extra_field",
    ],
)
def test_stage_event_contract_fails_closed(mutation: str) -> None:
    events = copy.deepcopy(_measurement()["stage_events"])
    if mutation == "wrong_stage":
        events[1]["stage"] = "fixture"
    elif mutation == "wrong_index":
        events[1]["sequence_index"] = 0
    elif mutation == "negative":
        events[0]["entered_ns"] = -1
    elif mutation == "reverse_interval":
        events[1]["finished_ns"] = events[1]["entered_ns"] - 1
    elif mutation == "reverse_sequence_time":
        events[1]["entered_ns"] = events[0]["finished_ns"] - 1
    elif mutation == "missing_event":
        events.pop()
    else:
        events[0]["unexpected"] = True
    assert contract.validate_stage_events(events) is False


@pytest.mark.parametrize(
    ("mutation", "check"),
    [
        ("timing_version", "timing_boundary_identity"),
        ("event", "ordered_stage_events"),
        ("duration_missing", "duration_contract"),
        ("duration_boolean", "duration_contract"),
        ("duration_negative", "duration_contract"),
        ("duration_nonfinite", "duration_contract"),
        ("warm_count", "duration_contract"),
        ("synchronization_count", "synchronization_counts"),
        ("full_count", "synchronization_counts"),
        ("parity_count", "synchronization_counts"),
        ("invocation", "invocation_counts"),
        ("graph", "graphdef_metadata"),
        ("parity", "direct_output_parity"),
        ("sidecar_hash", "payload_sidecar_identity"),
        ("sidecar_write_count", "payload_sidecar_identity"),
        ("envelope", "outer_envelope_unmeasured"),
        ("forbidden_compile", "no_compile_subtraction_field"),
    ],
)
def test_measurement_record_mutations_fail_named_gate(mutation: str, check: str) -> None:
    measurement = _measurement(repeats=2)
    if mutation == "timing_version":
        measurement["timing_boundary_version"] = "stale"
    elif mutation == "event":
        measurement["stage_events"][1]["stage"] = "fixture"
    elif mutation == "duration_missing":
        measurement["durations"].pop("trace_seconds")
    elif mutation == "duration_boolean":
        measurement["durations"]["trace_seconds"] = True
    elif mutation == "duration_negative":
        measurement["durations"]["trace_seconds"] = -1.0
    elif mutation == "duration_nonfinite":
        measurement["durations"]["trace_seconds"] = float("inf")
    elif mutation == "warm_count":
        measurement["durations"]["warm_execution_seconds"] = [0.1]
    elif mutation == "synchronization_count":
        measurement["synchronization"]["scalar_materialization_count"] = 0
    elif mutation == "full_count":
        measurement["synchronization"]["full_output_materialization_count"] = 2
    elif mutation == "parity_count":
        measurement["synchronization"]["parity_residual_materialization_count"] = 2
    elif mutation == "invocation":
        measurement["invocation_counts"]["before_first_executable_call"] = 1
    elif mutation == "graph":
        measurement["graphdef"]["node_count"] = 0
    elif mutation == "parity":
        measurement["direct_output_parity"]["passed"] = False
    elif mutation == "sidecar_hash":
        measurement["payload_sidecar"]["sha256"] = "short"
    elif mutation == "sidecar_write_count":
        measurement["payload_sidecar"]["write_count"] = 2
    elif mutation == "envelope":
        measurement["envelope_write_measured"] = True
    else:
        measurement["compilation_seconds"] = 1.0
    checks = contract.measurement_record_checks({"measurement": measurement})
    assert checks[check] is False
    assert contract.measurement_record_is_valid({"measurement": measurement}) is False


def test_synchronization_paths_are_counted_without_full_materialization(benchmark) -> None:
    outputs = (tf.constant([1.0, 2.0]), tf.constant([[3.0], [4.0]]))
    calls = []
    method, count, definition = benchmark._synchronize_outputs(
        outputs, async_wait=lambda: calls.append("wait")
    )
    assert (method, count, definition) == ("tf.experimental.async_wait", 0, None)
    assert calls == ["wait"]

    method, count, definition = benchmark._synchronize_outputs(
        outputs, async_wait=None
    )
    assert method in {"tf.experimental.async_wait", "scalar_sentinel"}
    if method == "scalar_sentinel":
        assert (count, definition) == (1, "reduce_sum(value)+reduce_sum(score)")
    else:
        assert (count, definition) == (0, None)


def test_materialize_uses_one_packed_host_transfer(benchmark, monkeypatch) -> None:
    value = tf.constant([1.0, 2.0], dtype=tf.float32)
    score = tf.constant([[3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
    real_concat = tf.concat
    transfers = []

    class Packed:
        def __init__(self, tensor):
            self.tensor = tensor

        def numpy(self):
            transfers.append("packed")
            return self.tensor.numpy()

    monkeypatch.setattr(benchmark.tf, "concat", lambda values, axis: Packed(real_concat(values, axis)))
    materialized = benchmark._materialize((value, score))
    assert transfers == ["packed"]
    np.testing.assert_allclose(materialized["value"], [1.0, 2.0])
    np.testing.assert_allclose(materialized["score"], [[3.0, 4.0], [5.0, 6.0]])


def test_encoded_sidecar_is_strict_atomic_and_not_reencoded(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "method.payload.json"
    encoded = contract.strict_json_dumps({"value": [1.0]}, indent=2) + "\n"
    calls = 0
    real_dumps = contract.strict_json_dumps

    def counting_dumps(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(contract, "strict_json_dumps", counting_dumps)
    contract.atomic_write_encoded_json(path, encoded)
    assert calls == 0
    assert path.read_text(encoding="utf-8") == encoded
    before = path.read_bytes()
    with pytest.raises(contract.ContractError):
        contract.atomic_write_encoded_json(path, '{"bad":NaN}\n')
    assert path.read_bytes() == before


def _runner_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dimensions=[2],
        parameter_counts=[3],
        timesteps=4,
        batch_size=4,
        dtype="float32",
        device="cpu",
        cpu_threads=1,
        repeats=2,
        timeout_seconds=90.0,
        methods=list(contract.PRIMARY_METHOD_IDS),
        output_dir=tmp_path,
        harness_contract_test_only=False,
        no_resume=True,
        jit_compile=True,
        tf32_enabled=True,
    )


def _synthetic_status(runner, tmp_path: Path) -> tuple[dict[str, object], dict[str, str]]:
    args = _runner_args(tmp_path)
    schedule = runner.build_schedule(args)
    input_path = tmp_path / "status.json"
    execution = runner._phase5_expected_execution(input_path, tmp_path / "smoke.log")
    execution["git_commit"] = runner._git_commit()
    execution["command_argv"] = [
        str(runner.PYTHON.resolve()),
        str(runner.Path(runner.__file__).resolve()),
        "--dimensions", "2",
        "--parameter-counts", "3",
        "--timesteps", "4",
        "--batch-size", "4",
        "--dtype", "float32",
        "--device", "cpu",
        "--cpu-threads", "1",
        "--repeats", "2",
        "--timeout-seconds", "90",
        "--methods", *contract.PRIMARY_METHOD_IDS,
        "--output-dir", str(tmp_path),
        "--no-resume",
        "--jit-compile",
        "--tf32-enabled",
    ]
    schedule["execution_contract"] = execution
    schedule_fingerprint = schedule["schedule_fingerprint"]
    records = []
    sidecars: dict[str, str] = {}
    for identity in schedule["expected_identities"]:
        fingerprints = runner._fingerprints(identity, schedule_fingerprint)
        sidecar_path = runner.method_artifact_path(tmp_path, identity).with_suffix(
            ".payload.json"
        )
        measurement = _measurement(repeats=2)
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
                "attempt_id": "attempt-" + identity["method_id"],
                "last_entered_stage": "envelope_write",
                "terminal_stage": "envelope_write",
                "failure_stage": None,
                "returncode": 0,
                "timed_out": False,
                "error": None,
                "invoked_method_ids": [identity["method_id"]],
                "measurement": measurement,
                "output_metadata": {
                    "all_finite": True,
                    "value_shape": [4],
                    "score_shape": [4, 3],
                    "value_dtype": "float32",
                    "score_dtype": "float32",
                },
                "outputs": {
                    "value": [1.0, 2.0, 3.0, 4.0],
                    "score": [[1.0, 2.0, 3.0]] * 4,
                },
                "device_manifest": {
                    "requested_device": "cpu",
                    "selected_device": "/CPU:0",
                    "physical_gpus": [],
                    "logical_gpus": [],
                    "cpu_only_exception": True,
                    "trust_basis": "cpu_debug_or_reference_exception",
                },
                "cpu_thread_manifest": {
                    "requested_cpu_threads": 1,
                    "tf_intra_op_parallelism_threads": 1,
                    "tf_inter_op_parallelism_threads": 1,
                    "intra_op_set_status": "set",
                    "inter_op_set_status": "set",
                    "omp_num_threads": "1",
                    "tf_num_intraop_threads_env": "1",
                    "tf_num_interop_threads_env": "1",
                },
            }
        payload = {
            "case_id": record["case_id"],
            "method_id": record["method_id"],
            "output_metadata": record["output_metadata"],
            "outputs": record["outputs"],
            "graphdef": measurement["graphdef"],
            "direct_output_parity": measurement["direct_output_parity"],
        }
        content = contract.strict_json_dumps(payload, indent=2) + "\n"
        sidecars[str(sidecar_path)] = content
        measurement["payload_sidecar"] = {
            "path": str(sidecar_path),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "write_count": 1,
        }
        records.append(record)
    aggregate = runner._aggregate_checks(schedule, records)
    status = {
        "schema": contract.SCHEMA,
        "status": "complete",
        "schedule": schedule,
        "records": records,
        "execution_contract": execution,
        "aggregate_checks": aggregate,
        "comparison_summary": {"comparison_complete": True},
    }
    return status, sidecars


def test_phase5_evaluator_recomputes_raw_gates_and_embeds_sidecars(
    runner, tmp_path: Path, monkeypatch
) -> None:
    status, sidecars = _synthetic_status(runner, tmp_path)
    monkeypatch.setattr(runner, "_git_commit", lambda: status["execution_contract"]["git_commit"])
    checks, embedded = runner.evaluate_phase5_smoke_raw(
        status,
        input_path=tmp_path / "status.json",
        log_path=tmp_path / "smoke.log",
        sidecar_reader=lambda path: sidecars[str(path)],
    )
    assert all(checks.values()), checks
    assert len(embedded) == 2

    for mutation, gate in (
        (("execution_contract", "jit_compile"), "execution_identity"),
        (("schedule", "unexpected"), "schedule_source_runtime_identity"),
        (("records", 0, "measurement", "stage_events", 1, "stage"), f"{contract.PRIMARY_METHOD_IDS[0]}:ordered_stage_events"),
        (("records", 0, "measurement", "payload_sidecar", "sha256"), "sidecar_content_hash_identity"),
        (("aggregate_checks", "record_integrity"), "aggregate_checks_recomputed_identity"),
    ):
        mutated = copy.deepcopy(status)
        target = mutated
        for key in mutation[:-1]:
            target = target[key]
        leaf = mutation[-1]
        original = target.get(leaf) if isinstance(target, dict) else target[leaf]
        target[leaf] = False if original is True else "corrupt"
        mutated_checks, _ = runner.evaluate_phase5_smoke_raw(
            mutated,
            input_path=tmp_path / "status.json",
            log_path=tmp_path / "smoke.log",
            sidecar_reader=lambda path: sidecars[str(path)],
        )
        assert mutated_checks[gate] is False

    forged = copy.deepcopy(status)
    forged["checks"] = {"everything": True}
    forged["records"][0]["measurement"]["durations"]["trace_seconds"] = -1.0
    forged_checks, _ = runner.evaluate_phase5_smoke_raw(
        forged,
        input_path=tmp_path / "status.json",
        log_path=tmp_path / "smoke.log",
        sidecar_reader=lambda path: sidecars[str(path)],
    )
    assert forged_checks[f"{contract.PRIMARY_METHOD_IDS[0]}:duration_contract"] is False

    log_path = tmp_path / "smoke.log"
    log_path.write_text("smoke\n", encoding="utf-8")
    output_path = tmp_path / "export.json"
    payload, returncode = runner.build_phase5_smoke_export(
        status,
        input_path=tmp_path / "status.json",
        log_path=log_path,
        output_path=output_path,
        expected_input=tmp_path / "status.json",
        expected_log=log_path,
        expected_output=output_path,
        sidecar_reader=lambda path: sidecars[str(path)],
    )
    assert (payload["state"], returncode) == ("passed", 0)
    assert all(payload["checks"].values())

    for mutation in (
        ("records", 0, "measurement", "durations", "trace_seconds"),
        ("records", 0, "measurement", "synchronization", "full_output_materialization_count"),
        ("records", 0, "measurement", "payload_sidecar", "sha256"),
        ("execution_contract", "jit_compile"),
    ):
        invalid = copy.deepcopy(status)
        invalid["checks"] = {"forged": True}
        target = invalid
        for key in mutation[:-1]:
            target = target[key]
        leaf = mutation[-1]
        target[leaf] = False if target[leaf] is True else "invalid"
        failed_payload, failed_returncode = runner.build_phase5_smoke_export(
            invalid,
            input_path=tmp_path / "status.json",
            log_path=log_path,
            output_path=output_path,
            expected_input=tmp_path / "status.json",
            expected_log=log_path,
            expected_output=output_path,
            sidecar_reader=lambda path: sidecars[str(path)],
        )
        assert (failed_payload["state"], failed_returncode) == ("failed", 1)
        assert not all(failed_payload["checks"].values())


def test_gpu_hidden_non_jit_supervisor_smoke_obeys_v4_boundaries(tmp_path: Path) -> None:
    output_dir = tmp_path / "non_jit_smoke"
    command = [
        sys.executable,
        str(RUNNER_PATH),
        "--dimensions", "2",
        "--parameter-counts", "3",
        "--timesteps", "4",
        "--batch-size", "4",
        "--dtype", "float32",
        "--device", "cpu",
        "--cpu-threads", "1",
        "--repeats", "2",
        "--timeout-seconds", "60",
        "--methods", *contract.PRIMARY_METHOD_IDS,
        "--output-dir", str(output_dir),
        "--no-resume",
        "--no-jit-compile",
        "--tf32-enabled",
    ]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
        }
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    status = contract.read_strict_json(output_dir / "status.json")
    assert status["status"] == "complete"
    assert status["aggregate_checks"] == {
        name: True for name in contract.PRIMARY_PAIR_CHECKS
    }
    assert [record["method_id"] for record in status["records"]] == list(
        contract.PRIMARY_METHOD_IDS
    )
    for record in status["records"]:
        assert contract.measurement_record_is_valid(record)
        expected_sidecar = Path(record["measurement"]["payload_sidecar"]["path"])
        assert contract.payload_sidecar_matches_record(
            record, expected_path=expected_sidecar
        )
        assert record["measurement"]["invocation_counts"] == {
            "before_first_executable_call": 0,
            "after_first_executable_call": 1,
            "after_warm_execution": 3,
            "after_reference_call": 4,
        }
        assert record["measurement"]["synchronization"][
            "full_output_materialization_count"
        ] == 1
        assert record["measurement"]["envelope_write_measured"] is False


def _child_args(tmp_path: Path) -> argparse.Namespace:
    attempt_id, progress = contract.new_attempt(
        tmp_path / "progress", "case-a", contract.PRIMARY_METHOD_IDS[0]
    )
    return argparse.Namespace(
        attempt_id=attempt_id,
        case_id="case-a",
        method=contract.PRIMARY_METHOD_IDS[0],
        source_fingerprint="source",
        config_fingerprint="config",
        runtime_fingerprint="runtime",
        fixture_fingerprint="fixture",
        schedule_fingerprint="schedule",
        resume_key="resume",
        progress_journal=str(progress),
        output_json=str(tmp_path / "method.json"),
        timesteps=4,
        batch_size=4,
        repeats=2,
        jit_compile=False,
        phase6_dependency_before=None,
    )


class _FakeGraphDef:
    node = (object(),)

    @staticmethod
    def SerializeToString() -> bytes:
        return b"graph"


class _FakeGraph:
    @staticmethod
    def as_graph_def(*, add_shapes: bool):
        assert add_shapes is True
        return _FakeGraphDef()


class _FakeMethod:
    graph = _FakeGraph()

    def __init__(self, fail_call: int | None = None):
        self.calls = 0
        self.fail_call = fail_call

    def get_concrete_function(self, parameters):
        del parameters
        return self

    def __call__(self, parameters):
        del parameters
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError("injected invocation failure")
        return (
            tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32),
            tf.ones([4, 3], dtype=tf.float32),
        )


@pytest.mark.parametrize(
    ("failure", "expected_stage"),
    [
        ("builder", "trace"),
        ("get_concrete", "trace"),
        ("first_invocation", "first_executable_call"),
        ("first_synchronization", "first_executable_call"),
        ("warm_1", "warm_execution"),
        ("warm_2", "warm_execution"),
        ("materialization", "materialization"),
        ("encoding", "payload_encoding"),
        ("payload_write", "payload_write"),
    ],
)
def test_measurement_orchestrator_failures_are_stage_specific(
    benchmark, tmp_path: Path, monkeypatch, failure: str, expected_stage: str
) -> None:
    args = _child_args(tmp_path)
    fail_call = {"first_invocation": 1, "warm_1": 2, "warm_2": 3}.get(failure)
    method = _FakeMethod(fail_call=fail_call)
    monkeypatch.setattr(benchmark, "make_fixture", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        benchmark, "_make_parameter_batch", lambda fixture, batch_size: tf.zeros([4, 3])
    )
    if failure == "builder":
        monkeypatch.setattr(
            benchmark,
            "_selected_method_builder",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("builder")),
        )
    else:
        if failure == "get_concrete":
            monkeypatch.setattr(
                method,
                "get_concrete_function",
                lambda parameters: (_ for _ in ()).throw(RuntimeError("trace")),
            )
        monkeypatch.setattr(
            benchmark,
            "_selected_method_builder",
            lambda *args, **kwargs: (method, [contract.PRIMARY_METHOD_IDS[0]]),
        )

    synchronization_calls = 0

    def synchronize(outputs):
        nonlocal synchronization_calls
        del outputs
        synchronization_calls += 1
        if failure == "first_synchronization" and synchronization_calls == 1:
            raise RuntimeError("synchronization")
        return "scalar_sentinel", 1, "reduce_sum(value)+reduce_sum(score)"

    monkeypatch.setattr(benchmark, "_synchronize_outputs", synchronize)
    if failure == "materialization":
        monkeypatch.setattr(
            benchmark,
            "_materialize",
            lambda outputs: (_ for _ in ()).throw(RuntimeError("materialization")),
        )
    if failure == "encoding":
        real_dumps = benchmark.benchmark_contract.strict_json_dumps

        def fail_payload_encoding(value, **kwargs):
            if kwargs.get("indent") == 2:
                raise contract.ContractError("encoding")
            return real_dumps(value, **kwargs)

        monkeypatch.setattr(
            benchmark.benchmark_contract, "strict_json_dumps", fail_payload_encoding
        )
    if failure == "payload_write":
        monkeypatch.setattr(
            benchmark.benchmark_contract,
            "atomic_write_encoded_json",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write")),
        )

    record = benchmark.benchmark_selected_method_case(
        args=args,
        dimension=2,
        parameter_count=3,
        device_name="/CPU:0",
        dtype=tf.float32,
    )
    assert record["state"] == "failed"
    assert record["failure_stage"] == expected_stage
    assert record["last_entered_stage"] == "envelope_write"
    assert record["measurement"] is None
    assert contract.recover_last_stage(
        Path(args.progress_journal), benchmark._progress_identity(args)
    ) == "envelope_write"


def test_progress_append_fsync_failure_is_fail_closed(tmp_path: Path, monkeypatch) -> None:
    attempt_id, path = contract.new_attempt(
        tmp_path / "progress", "case-a", contract.PRIMARY_METHOD_IDS[0]
    )
    event = {
        "attempt_id": attempt_id,
        "case_id": "case-a",
        "method_id": contract.PRIMARY_METHOD_IDS[0],
        "stage": "fixture",
        "resume_key": "resume",
        **{field: field for field in contract.FINGERPRINT_FIELDS},
    }
    monkeypatch.setattr(contract.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync")))
    with pytest.raises(OSError, match="fsync"):
        contract.append_progress_event(path, event)
    expected = dict(event)
    expected.pop("stage")
    assert contract.recover_last_stage(path, expected) == "fixture"
