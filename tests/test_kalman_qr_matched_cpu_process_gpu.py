from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_kalman_qr_matched_cpu_process_gpu_2026_07_15.py"


def _load():
    name = "matched_kalman_qr_runner_test_subject"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_all_arms_cover_same_sixteen_rows_once() -> None:
    runner = _load()
    for arm in runner.ARMS:
        rows = [row for spec in runner.arm_worker_specs(arm) for row in spec["row_ids"]]
        assert sorted(rows) == list(range(16))
        assert len(rows) == len(set(rows)) == 16


def test_cpu_resources_are_same_physical_pool() -> None:
    runner = _load()
    native = runner.arm_worker_specs("cpu_native_b16_xla")
    sharded = runner.arm_worker_specs("cpu_processes_16xb1_xla")
    assert native == [{"row_ids": list(range(16)), "cpus": list(range(16, 32)), "intra": 16, "device": "cpu"}]
    assert [spec["cpus"][0] for spec in sharded] == list(range(16, 32))


def test_every_block_order_contains_each_arm_once_and_is_balanced() -> None:
    runner = _load()
    orders = runner.balanced_orders()
    assert len(orders) == 6
    assert all(sorted(order) == sorted(runner.ARMS) for order in orders)
    for position in range(3):
        assert sorted(order[position] for order in orders) == sorted(runner.ARMS * 2)


def _outputs(runner, offset: float = 0.0):
    return {row: ([float(row) + offset], [float(index) for index in range(runner.PARAMETER_COUNT)]) for row in runner.ROW_IDS}


def test_parity_fails_closed_on_rows_shapes_nonfinite_and_residual() -> None:
    runner = _load()
    assert runner.parity_summary(_outputs(runner), _outputs(runner))["passed"]
    missing = _outputs(runner)
    missing.pop(3)
    assert not runner.parity_summary(missing, _outputs(runner))["passed"]
    wrong_shape = _outputs(runner)
    wrong_shape[0] = ([0.0], [0.0])
    assert not runner.parity_summary(wrong_shape, _outputs(runner))["passed"]
    nonfinite = _outputs(runner)
    nonfinite[0] = ([math.nan], nonfinite[0][1])
    assert not runner.parity_summary(nonfinite, _outputs(runner))["passed"]
    assert not runner.parity_summary(_outputs(runner, 1.0), _outputs(runner))["passed"]


def test_gpu_admission_requires_less_than_fifty_percent_and_bounded_memory() -> None:
    runner = _load()
    base = {"gpu_returncode": 0, "gpus": [["0", "uuid", "name", "32760", "1000", "49", "40"]]}
    assert runner.gpu0_admissible(base)
    assert not runner.gpu0_admissible({**base, "gpus": [["0", "uuid", "name", "32760", "1000", "50", "40"]]})
    assert not runner.gpu0_admissible({**base, "gpus": [["0", "uuid", "name", "32760", "3000", "0", "40"]]})


def test_gpu_process_census_is_scoped_to_gpu_zero() -> None:
    runner = _load()
    snapshot = {"gpu_returncode": 0, "app_returncode": 0, "gpus": [["0", "gpu0"]], "compute_apps": [["gpu0", "10", "display", "200"], ["gpu1", "20", "other", "100"]]}
    assert runner.gpu0_compute_pids(snapshot) == {10}
    assert runner.gpu0_compute_pids({**snapshot, "app_returncode": 1}) is None


def test_worker_environments_enforce_cpu_hiding_and_gpu_growth(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load()
    monkeypatch.setenv("XLA_FLAGS", "foreign")
    cpu = runner.worker_environment(runner.arm_worker_specs("cpu_native_b16_xla")[0])
    gpu = runner.worker_environment(runner.arm_worker_specs("gpu_native_b16_xla")[0])
    assert cpu["CUDA_VISIBLE_DEVICES"] == "-1"
    assert "XLA_FLAGS" not in cpu
    assert gpu["CUDA_VISIBLE_DEVICES"] == "0"
    assert gpu["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
    assert gpu["XLA_FLAGS"] == runner.GPU_XLA_FLAG


def test_paired_statistics_are_deterministic() -> None:
    runner = _load()
    first = runner.paired_statistics([0.5] * 6, [1.0] * 6, resamples=1000)
    second = runner.paired_statistics([0.5] * 6, [1.0] * 6, resamples=1000)
    assert first == second
    assert first["bootstrap_95_interval"][1] < 1.0


def test_cpu_busy_and_contamination_accounting() -> None:
    runner = _load()
    before = {16: (100, 50), 17: (200, 100)}
    after = {16: (200, 140), 17: (300, 150)}
    assert runner.cpu_busy_fractions(before, after) == pytest.approx({16: 0.1, 17: 0.5})
    # 60 busy ticks at 100 Hz minus 0.4 owned CPU-second leaves 0.2 second.
    assert runner.target_cpu_contamination_seconds(before, after, 0.4) == pytest.approx(0.2)
    assert runner.cpu_busy_fractions(before, {16: after[16]}) is None


def test_source_manifest_includes_plan_and_harness() -> None:
    runner = _load()
    paths = {row["path"] for row in runner.source_manifest()["files"]}
    assert runner.PLAN in paths
    assert "docs/benchmarks/run_kalman_qr_matched_cpu_process_gpu_2026_07_15.py" in paths
