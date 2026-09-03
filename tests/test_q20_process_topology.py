"""Focused tests for the staged q=20 process topology contract."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from pathlib import Path

import pytest

from bayesfilter.inference.process_topology import (
    BarrierTopology,
    ProcessTopologyError,
    Q20ProcessTopology,
    WorkerAssignment,
)


def test_default_topology_is_exactly_the_requested_72_worker_cores() -> None:
    topology = Q20ProcessTopology()
    assert topology.total_worker_cores == 72
    assert topology.screen.worker_core_total == 8 * 4
    assert topology.selection.worker_core_total == 2 * 8
    assert topology.scope_finalize.worker_core_total == 6 * 4
    assert topology.peak_barrier_cores == 32
    assert topology.payload()["barriers_are_sequential"] is True


def test_barrier_assignments_are_disjoint_and_exact() -> None:
    topology = Q20ProcessTopology()
    available = tuple(range(128))
    for name, workers, cores in (
        ("screen", 8, 4),
        ("selection", 2, 8),
        ("scope_finalize", 6, 4),
    ):
        assignments = topology.assignments(name, available)
        assert len(assignments) == workers
        assert all(len(row.cpu_ids) == cores for row in assignments)
        flattened = [cpu for row in assignments for cpu in row.cpu_ids]
        assert len(flattened) == len(set(flattened))


def test_assignment_fails_closed_when_affinity_is_too_small() -> None:
    topology = Q20ProcessTopology()
    with pytest.raises(ProcessTopologyError, match="need 72"):
        topology.validate_available_cpu_ids(range(71))


def test_invalid_barrier_values_fail_closed() -> None:
    with pytest.raises(ProcessTopologyError, match="work_unit_count"):
        BarrierTopology("bad", worker_count=1, cores_per_worker=1, work_unit_count=0)


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[1] / (
        "docs/benchmarks/run_ssl_lstm_q20_72core_process_parallel_2026_09_03.py"
    )
    spec = importlib.util.spec_from_file_location("q20_72core_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_declares_eight_candidate_pairs_and_staged_task_counts() -> None:
    module = _load_benchmark_module()
    assert len(module._candidate_rows()) == 8
    assert module.TOPOLOGY.total_worker_cores == 72
    assert module.CANARY_COUNTS["selection_num_results"] >= 4


def test_process_artifact_encoding_preserves_nonfinite_diagnostics() -> None:
    module = _load_benchmark_module()
    payload = module._canonical({"nan": float("nan"), "finite": 1.0})
    assert b'"__nonfinite__":"nan"' in payload
    assert b'"finite":1.0' in payload


def test_deadline_closeout_preserves_partial_task_coverage(tmp_path, monkeypatch) -> None:
    module = _load_benchmark_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    output = tmp_path / "selection"
    worker = output / "worker-0"
    task_dir = worker / "tasks"
    task_dir.mkdir(parents=True)
    module._write_json(
        task_dir / "task-0.json",
        {
            "schema": "bayesfilter.q20.process_task_result.v1",
            "status": "COMPLETE",
            "task": {"task_id": "task-0"},
        },
    )
    module._write_json(
        task_dir / "task-1.start.json",
        {"schema": "start", "status": "RUNNING", "task": {"task_id": "task-1"}},
    )

    class FakeProcess:
        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    process = FakeProcess()
    barrier = module.BarrierTopology(
        name="selection", worker_count=1, cores_per_worker=8, work_unit_count=2
    )
    payload = module._barrier_deadline_payload(
        phase="selection",
        barrier=barrier,
        tasks=(
            {"task_id": "task-0"},
            {"task_id": "task-1"},
        ),
        output_dir=output,
        processes=(process,),
        stage="worker_summary",
    )
    assert payload["status"] == "CAP_STOP_INCOMPLETE"
    assert payload["partial_coverage"]["returned_task_count"] == 1
    assert payload["partial_coverage"]["missing_task_ids"] == ["task-1"]
    assert (output / "barrier_timeout.json").is_file()
    assert process.returncode == -15


def test_launch_barrier_raises_typed_deadline_after_readiness_cap(tmp_path, monkeypatch) -> None:
    module = _load_benchmark_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)

    class FakeProcess:
        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        module,
        "TOPOLOGY",
        SimpleNamespace(
            assignments=lambda name, available: (
                WorkerAssignment(worker_index=0, cpu_ids=(0, 1, 2, 3)),
            )
        ),
    )
    barrier = module.BarrierTopology(
        name="screen", worker_count=1, cores_per_worker=4, work_unit_count=1
    )
    with pytest.raises(module.ParallelCampaignDeadline) as caught:
        module._launch_barrier(
            phase="screen",
            barrier=barrier,
            tasks=({"task_id": "task-0"},),
            output_dir=tmp_path / "screen",
            available_cpu_ids=tuple(range(72)),
            deadline=0.0,
        )
    assert caught.value.payload["stage"] == "worker_readiness"
    assert caught.value.payload["status"] == "CAP_STOP_INCOMPLETE"
    assert (tmp_path / "screen" / "barrier_timeout.json").is_file()
