from __future__ import annotations

import sys
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from bayesfilter.runtime.parallel_tuning import (
    ParallelTuningError,
    ParallelTuningTask,
    run_parallel_tuning_wave,
    validate_tasks,
    worker_environment,
)


def _task(tmp_path: Path, name: str, index: int, gpu: str, code: str) -> ParallelTuningTask:
    return ParallelTuningTask(
        task_id=name,
        scope_index=index,
        gpu_uuid=gpu,
        output_dir=tmp_path / name,
        command=(sys.executable, "-c", code),
    )


def test_worker_environment_pins_one_gpu_and_growth_before_framework_import(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    environment = worker_environment({"BASE": "yes"}, "GPU-test-1")
    assert environment["BASE"] == "yes"
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-test-1"
    assert environment["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
    assert environment["BAYESFILTER_SELECTED_GPU_UUID"] == "GPU-test-1"


def test_validate_tasks_rejects_duplicate_gpu_assignment(tmp_path: Path):
    tasks = (
        _task(tmp_path, "a", 0, "GPU-test-1", "pass"),
        _task(tmp_path, "b", 1, "GPU-test-1", "pass"),
    )
    with pytest.raises(ParallelTuningError, match="GPU UUID reused"):
        validate_tasks(tasks)


def test_parallel_wave_runs_independent_processes_and_preserves_private_artifacts(
    tmp_path: Path,
):
    code = (
        "from pathlib import Path; import os, json, time; "
        "root = Path(os.environ['BAYESFILTER_WORKER_OUTPUT_DIR']); "
        "root.mkdir(exist_ok=False); "
        "root.joinpath('ready').write_text('ready'); "
        "deadline = time.monotonic() + 4; "
        "\nwhile len(list(root.parent.glob('scope-*/ready'))) != 2:\n"
        " assert time.monotonic() < deadline, 'second worker was not concurrent'\n"
        " time.sleep(0.01)\n"
        "root.joinpath('run_manifest.json').write_text(json.dumps({"
        "'status': 'completed', 'gpu': os.environ['CUDA_VISIBLE_DEVICES']}))"
    )
    tasks = (
        _task(tmp_path, "scope-0", 0, "GPU-test-0", code),
        _task(tmp_path, "scope-1", 1, "GPU-test-1", code),
    )
    result = run_parallel_tuning_wave(
        tasks, timeout_seconds=10.0, terminate_grace_seconds=0.1, cwd=tmp_path
    )
    assert result["status"] == "completed"
    assert result["completed_count"] == 2
    assert json.loads((tmp_path / "scope-0" / "run_manifest.json").read_text())["gpu"] == "GPU-test-0"
    assert json.loads((tmp_path / "scope-1" / "run_manifest.json").read_text())["gpu"] == "GPU-test-1"
    assert all(row["status"] == "completed" for row in result["results"])
    assert len({row["pid"] for row in result["results"]}) == 2
    assert result["worker_seconds"] == pytest.approx(sum(row["elapsed_seconds"] for row in result["results"]))


def test_parallel_wave_times_out_and_records_worker_result(tmp_path: Path):
    code = "import time; time.sleep(10)"
    task = _task(tmp_path, "slow", 0, "GPU-test-0", code)
    result = run_parallel_tuning_wave(
        (task,),
        timeout_seconds=0.2,
        terminate_grace_seconds=0.1,
        cwd=tmp_path,
    )
    assert result["status"] == "timed_out"
    assert result["results"][0]["status"] == "timed_out"
    assert (task.supervision_dir / "parallel_worker_result.json").is_file()


@pytest.mark.parametrize("code,expected", (
    ("raise SystemExit(3)", "failed"),
    ("pass", "invalid_required_artifact"),
    ("from pathlib import Path; import os; root = Path(os.environ['BAYESFILTER_WORKER_OUTPUT_DIR']); "
     "root.mkdir(); root.joinpath('run_manifest.json').write_text('not json')", "invalid_required_artifact"),
))
def test_worker_failure_or_bad_artifact_never_passes_wave(tmp_path, code, expected):
    task = _task(tmp_path, "bad", 0, "GPU-test-0", code)
    result = run_parallel_tuning_wave((task,), timeout_seconds=5, terminate_grace_seconds=0.1)
    assert result["status"] == "failed"
    assert result["results"][0]["status"] == expected


def test_timeout_preserves_early_success_and_individual_elapsed_time(tmp_path):
    code = (
        "from pathlib import Path; import os; root = Path(os.environ['BAYESFILTER_WORKER_OUTPUT_DIR']); "
        "root.mkdir(); root.joinpath('run_manifest.json').write_text('{\"status\": \"completed\"}')"
    )
    fast = _task(tmp_path, "fast", 0, "GPU-test-0", code)
    slow = _task(tmp_path, "slow", 1, "GPU-test-1", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)")
    result = run_parallel_tuning_wave((fast, slow), timeout_seconds=1.0, terminate_grace_seconds=0.1)
    first, second = result["results"]
    assert first["status"] == "completed"
    assert second["status"] == "timed_out"
    assert second["returncode"] == -9
    assert first["elapsed_seconds"] < second["elapsed_seconds"]


def test_partial_launch_failure_reaps_started_worker_and_reports_unlaunched(tmp_path):
    tasks = (_task(tmp_path, "first", 0, "GPU-test-0", "import time; time.sleep(30)"),
             _task(tmp_path, "second", 1, "GPU-test-1", "pass"))
    processes = []

    def launch(*args, **kwargs):
        if processes:
            raise OSError("synthetic launch failure")
        process = subprocess.Popen(*args, **kwargs)
        processes.append(process)
        return process

    result = run_parallel_tuning_wave(tasks, timeout_seconds=5, terminate_grace_seconds=0.1, popen_factory=launch)
    assert result["status"] == "failed"
    assert result["results"][1]["status"] == "not_launched"
    assert processes[0].poll() is not None


def test_monitor_exception_reaps_workers(tmp_path):
    def interrupted():
        raise KeyboardInterrupt("synthetic parent interruption")

    task = _task(tmp_path, "interrupt", 0, "GPU-test-0", "import time; time.sleep(30)")
    result = run_parallel_tuning_wave((task,), timeout_seconds=5, terminate_grace_seconds=0.1,
                                      monitor_callback=interrupted)
    assert result["status"] == "failed"
    assert result["results"][0]["returncode"] is not None


def test_task_roots_cannot_overlap_or_reuse_control_root(tmp_path):
    first = _task(tmp_path, "first", 0, "GPU-test-0", "pass")
    second = replace(_task(tmp_path, "second", 1, "GPU-test-1", "pass"), output_dir=first.output_dir / "nested")
    with pytest.raises(ParallelTuningError, match="overlap"):
        validate_tasks((first, second))
    first.supervision_dir.mkdir()
    with pytest.raises(ParallelTuningError, match="reuse"):
        validate_tasks((first,))
