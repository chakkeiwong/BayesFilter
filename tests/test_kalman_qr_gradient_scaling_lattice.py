from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "docs/benchmarks/run_kalman_qr_gradient_scaling_lattice_2026_07_14.py"
)


def _load():
    name = "kalman_qr_gradient_scaling_lattice_test_subject"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_schedule_is_exact_requested_lattice() -> None:
    runner = _load()
    specs = runner._schedule_specs()
    assert len(specs) == 15
    assert {spec["batch_size"] for spec in specs} == {1, 4, 16}
    assert {spec["cpu_threads"] for spec in specs if spec["device"] == "cpu"} == {
        1,
        4,
        16,
    }
    assert {spec["dtype"] for spec in specs if spec["device"] == "gpu"} == {
        "float32",
        "float64",
    }


def test_command_uses_true_batched_pair_xla_and_five_warm_calls(tmp_path: Path) -> None:
    runner = _load()
    spec = runner._schedule_specs()[0]
    command = runner._command(spec, tmp_path, 600)
    assert command[
        command.index("--dimensions") + 1 : command.index("--parameter-counts")
    ] == ["10", "20", "30"]
    assert command[
        command.index("--parameter-counts") + 1 : command.index("--timesteps")
    ] == ["50", "150"]
    assert command[
        command.index("--methods") + 1 : command.index("--output-dir")
    ] == list(runner.METHODS)
    assert command[command.index("--repeats") + 1] == "5"
    assert "--jit-compile" in command
    assert "--no-resume" in command
    assert command[command.index("--plan-path") + 1] == runner.PLAN
    assert command[command.index("--result-path") + 1] == runner.RESULT


def _record(runner, *, device: str) -> dict:
    record = {
        "state": "passed",
        "measurement": {"durations": {"warm_execution_seconds": [1.0] * 5}},
        "device_manifest": {"selected_device": "/CPU:0"},
        "xla_flags": "UNSET",
    }
    if device == "gpu":
        record.update(
            {
                "device_manifest": {
                    "selected_device": "/GPU:0",
                    "trust_basis": runner.GPU_TRUST_BASIS,
                },
                "gpu_xla_triton_gemm_policy": {
                    "action": "benchmark_default_no_triton_applied",
                    "input_xla_flags": "UNSET",
                    "effective_xla_flags": runner.GPU_XLA_FLAG,
                },
                "gpu_memory_growth_policy": {
                    "policy": "required_no_full_device_preallocation",
                    "environment_variable": "TF_FORCE_GPU_ALLOW_GROWTH",
                    "environment_value": "true",
                },
                "gpu_allocator_memory": {
                    "device": "/GPU:0",
                    "current_bytes": 1024,
                    "peak_bytes": 2048,
                },
                "xla_flags": runner.GPU_XLA_FLAG,
            }
        )
    return record


def _status(runner, *, device: str) -> dict:
    return {
        "status": "complete",
        "comparison_summary": {"comparison_complete": True},
        "aggregate_checks": {"a": True, "b": True},
        "records": [_record(runner, device=device) for _ in range(12)],
    }


def test_completed_status_validation_fails_closed() -> None:
    runner = _load()
    spec = {"device": "cpu"}
    valid = _status(runner, device="cpu")
    assert runner._status_valid(valid, spec)
    assert not runner._status_valid(
        {**valid, "status": "complete_with_failures"}, spec
    )
    assert not runner._status_valid({**valid, "records": valid["records"][:11]}, spec)
    assert not runner._status_valid(
        {**valid, "aggregate_checks": {"a": True, "b": False}}, spec
    )


def test_gpu_status_requires_exact_policy_and_placement() -> None:
    runner = _load()
    spec = {"device": "gpu"}
    valid = _status(runner, device="gpu")
    assert runner._status_valid(valid, spec)
    invalid = {**valid, "records": [dict(row) for row in valid["records"]]}
    invalid["records"][0]["xla_flags"] = "UNSET"
    assert not runner._status_valid(invalid, spec)
    missing_growth = {**valid, "records": [dict(row) for row in valid["records"]]}
    missing_growth["records"][0].pop("gpu_memory_growth_policy")
    assert not runner._status_valid(missing_growth, spec)


def test_summary_preserves_raw_warm_calls() -> None:
    runner = _load()
    status = {
        "records": [
            {
                "case_id": (
                    "dimension=10-parameter_count=50-timesteps=120-"
                    "batch_size=4-dtype=float32-device=cpu"
                ),
                "method_id": runner.METHODS[0],
                "measurement": {
                    "durations": {
                        "trace_seconds": 1.0,
                        "first_executable_call_seconds": 2.0,
                        "warm_execution_seconds": [1.0, 2.0, 3.0, 4.0, 5.0],
                    },
                    "graphdef": {"node_count": 10, "serialized_bytes": 100},
                },
            }
        ]
    }
    rows = runner._record_warm_rows(
        {
            "batch_size": 4,
            "cpu_threads": 1,
            "device": "cpu",
            "dtype": "float32",
        },
        status,
    )
    assert rows[0]["warm_execution_seconds"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert rows[0]["warm_median_seconds"] == 3.0


def test_gpu_shared_prelaunch_gate_requires_authorized_display_baseline() -> None:
    runner = _load()
    idle = {
        "gpu_query_returncode": 0,
        "process_query_returncode": 0,
        "gpu_rows": [
            "0, uuid0, GPU, 32760, 1200, 40, 36",
            "1, uuid1, GPU, 32760, 30000, 80, 90",
        ],
        "compute_process_rows": [
            "uuid0, 5955, display, 251",
            "uuid0, 6575, nx, 312",
        ],
    }
    assert runner._gpu_target_prelaunch_admissible(idle)
    assert not runner._gpu_target_prelaunch_admissible(
        {**idle, "compute_process_rows": ["uuid0, 123, python, 100"]}
    )
    busy = {
        **idle,
        "gpu_rows": ["0, uuid0, GPU, 32760, 1200, 50, 36", idle["gpu_rows"][1]],
    }
    assert not runner._gpu_target_prelaunch_admissible(busy)
    assert not runner._gpu_target_prelaunch_admissible(
        {**idle, "process_query_returncode": 1}
    )


def test_gpu_runtime_gate_allows_display_and_owned_process_group(monkeypatch) -> None:
    runner = _load()
    snapshot = {
        "load_average": [8.0, 8.0, 8.0],
        "foreign_compute_processes": [],
        "gpu": {
            "gpu_query_returncode": 0,
            "process_query_returncode": 0,
            "gpu_rows": [
                "0, uuid0, GPU, 32760, 30682, 99, 70",
                "1, uuid1, GPU, 32760, 30000, 80, 90",
            ],
            "compute_process_rows": [
                "uuid0, 5955, display, 251",
                "uuid0, 6575, nx, 312",
                "uuid0, 123, python, 30682",
            ],
        },
    }
    monkeypatch.setattr(
        runner, "_pid_is_in_process_group", lambda pid, pgid: (pid, pgid) == (123, 99)
    )
    assert not runner._resources_idle(snapshot, "gpu")
    assert runner._runtime_resources_idle(snapshot, "gpu", owned_pgid=99)
    snapshot["gpu"]["compute_process_rows"].append("uuid0, 456, python, 100")
    assert not runner._runtime_resources_idle(snapshot, "gpu", owned_pgid=99)
    snapshot["gpu"]["compute_process_rows"] = [
        "uuid0, 5955, display, 251",
        "uuid0, 6575, nx, 312",
    ]
    assert not runner._runtime_resources_idle(snapshot, "gpu", owned_pgid=99)


def test_gpu_post_run_requires_display_baseline_and_memory_release() -> None:
    runner = _load()
    released = {
        "gpu_query_returncode": 0,
        "process_query_returncode": 0,
        "gpu_rows": [
            "0, uuid0, GPU, 32760, 1200, 80, 60",
            "1, uuid1, GPU, 32760, 30000, 80, 60",
        ],
        "compute_process_rows": [
            "uuid0, 5955, display, 251",
            "uuid0, 6575, nx, 312",
        ],
    }
    assert runner._gpu_target_released(released)
    assert not runner._gpu_target_released(
        {**released, "compute_process_rows": ["uuid0, 123, python, 100"]}
    )
    assert not runner._gpu_target_released(
        {
            **released,
            "gpu_rows": [
                "0, uuid0, GPU, 32760, 2200, 0, 60",
                released["gpu_rows"][1],
            ],
        }
    )


def test_resource_gate_is_device_relevant_and_bounded() -> None:
    runner = _load()
    base = {
        "load_average": [8.0, 8.0, 8.0],
        "foreign_compute_processes": [
            {"pid": 101, "cpu_percent": 100.0},
            {"pid": 102, "cpu_percent": 100.0},
        ],
        "gpu": {
            "gpu_query_returncode": 0,
            "process_query_returncode": 0,
            "gpu_rows": [
                "0, uuid0, GPU, 32760, 1200, 40, 36",
                "1, uuid1, GPU, 32760, 30000, 80, 90",
            ],
            "compute_process_rows": [
                "uuid0, 5955, display, 251",
                "uuid0, 6575, nx, 312",
            ],
        },
    }
    assert runner._resources_idle(base, "cpu")
    assert runner._resources_idle(base, "gpu")
    assert runner._resources_idle({**base, "load_average": [33.0]}, "cpu")
    assert not runner._resources_idle({**base, "load_average": [65.0]}, "cpu")
    assert not runner._resources_idle(
        {
            **base,
            "foreign_compute_processes": [{"pid": 103, "cpu_percent": 1601.0}],
        },
        "cpu",
    )
    assert runner._resources_idle(
        {
            **base,
            "foreign_compute_processes": [{"pid": 103, "cpu_percent": 3200.0}],
        },
        "gpu",
    )
    assert not runner._resources_idle(
        {**base, "foreign_compute_processes": [{"pid": None}]}, "cpu"
    )
    assert not runner._resources_idle(
        {
            **base,
            "gpu": {
                **base["gpu"],
                "compute_process_rows": ["uuid0, 123, python, 100"],
            },
        },
        "gpu",
    )


def test_cpu_inheritance_revalidates_exact_nine_schedules(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    fake_hashes = {path: f"hash-{index}" for index, path in enumerate(runner.SOURCE_PATHS)}
    fake_manifest = {
        "files": [
            {"path": path, "sha256": digest}
            for path, digest in fake_hashes.items()
        ],
        "fingerprint": "fake-current-fingerprint",
    }
    monkeypatch.setattr(runner, "_source_manifest", lambda: fake_manifest)

    schedules = []
    for spec in (row for row in runner._schedule_specs() if row["device"] == "cpu"):
        status_path = tmp_path / "prior" / spec["schedule_id"] / "status.json"
        runner._write_json(status_path, _status(runner, device="cpu"))
        attempt = {
            "attempt": 1,
            "valid": True,
            "structured_status_valid": True,
            "overlap_veto": False,
            "returncode": 0,
            "status_path": str(status_path.relative_to(tmp_path)),
            "device_environment": {
                "CUDA_VISIBLE_DEVICES": "-1",
                "OMP_NUM_THREADS": str(spec["cpu_threads"]),
                "TF_NUM_INTRAOP_THREADS": str(spec["cpu_threads"]),
                "TF_NUM_INTEROP_THREADS": str(spec["cpu_threads"]),
                "XLA_FLAGS": "UNSET",
            },
        }
        schedules.append({"spec": spec, "state": "passed", "attempts": [attempt]})

    prior_status = tmp_path / "prior" / "status.json"
    prior = {
        "schema": runner.SCHEMA,
        "status": "complete_with_failures",
        "source_manifest": json.loads(json.dumps(fake_manifest)),
        "execution_contract": runner._execution_contract(tmp_path / "prior"),
        "schedules": schedules,
    }
    prior_status.write_text(json.dumps(prior), encoding="utf-8")
    inherited, provenance = runner._validated_inherited_cpu_schedules(prior_status)
    assert len(inherited) == 9
    assert all(schedule["state"] == "passed" for schedule in inherited)
    assert provenance["method_record_count"] == 108
    inherited_master = {"inheritance": provenance, "schedules": inherited}
    assert runner._inherited_artifacts_valid(inherited_master)
    first_status = tmp_path / inherited[0]["attempts"][0]["status_path"]
    first_status.write_text("{}", encoding="utf-8")
    assert not runner._inherited_artifacts_valid(inherited_master)
    runner._write_json(first_status, _status(runner, device="cpu"))

    prior["source_manifest"]["files"][0]["sha256"] = "drifted"
    prior_status.write_text(json.dumps(prior), encoding="utf-8")
    try:
        runner._validated_inherited_cpu_schedules(prior_status)
    except RuntimeError as exc:
        assert "drifted" in str(exc)
    else:
        raise AssertionError("measurement-source drift must reject CPU inheritance")
