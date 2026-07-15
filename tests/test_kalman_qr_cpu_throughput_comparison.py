from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "docs/benchmarks/run_kalman_qr_cpu_throughput_comparison_2026_07_14.py"


def _load():
    name = "kalman_qr_cpu_throughput_comparison_test_subject"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("core_budget", [1, 2, 4, 8, 16])
def test_partitions_cover_every_canonical_row_once(core_budget: int) -> None:
    runner = _load()
    partitions = runner.partition_rows(core_budget)
    flattened = [row for partition in partitions for row in partition]
    assert sorted(flattened) == list(range(16))
    assert len(flattened) == len(set(flattened)) == 16


def test_b1_fixture_row_seven_cannot_define_sharded_work() -> None:
    runner = _load()
    assert runner.partition_rows(1) == [list(range(16))]
    assert runner.partition_rows(4) == [
        [0, 4, 8, 12],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15],
    ]


def test_cpu_lists_exclude_smt_siblings() -> None:
    runner = _load()
    assert runner.cpu_list(1) == (16,)
    assert runner.cpu_list(16) == tuple(range(16, 32))
    assert not set(runner.cpu_list(16)).intersection(range(144, 160))


@pytest.mark.parametrize("core_budget", [1, 2, 4, 8, 16])
def test_batched_and_sharded_specs_use_the_same_fixed_cpu_pool(core_budget: int) -> None:
    runner = _load()
    expected = runner.cpu_list(core_budget)
    batch = runner.worker_specs("batch_native", core_budget)
    sharded = runner.worker_specs("sharded", core_budget)
    assert batch[0][0] == expected
    assert tuple(spec[0][0] for spec in sharded) == expected
    assert [row for spec in sharded for row in spec[2]] == [
        row for partition in runner.partition_rows(core_budget) for row in partition
    ]


def test_parse_cpu_list() -> None:
    runner = _load()
    assert runner.parse_cpu_list("0-3,8,10-11") == {0, 1, 2, 3, 8, 10, 11}


def test_worker_placement_fails_closed() -> None:
    runner = _load()
    worker = {
        "affinity": [0, 1],
        "task_affinities": {"task_cpu_lists": {"1": [0, 1], "2": [0, 1]}},
        "numa": {"valid": True},
    }
    assert runner.validate_worker_placement(worker, {0, 1})
    assert not runner.validate_worker_placement({**worker, "affinity": [0]}, {0, 1})
    assert not runner.validate_worker_placement({**worker, "numa": {"valid": False}}, {0, 1})


def _outputs(offset: float = 0.0):
    return {row: ([float(row) + offset], [float(row), float(row + 1)]) for row in range(16)}


def test_parity_requires_exact_rows_shapes_finite_and_tolerance() -> None:
    runner = _load()
    assert runner.parity_summary(_outputs(), _outputs())["passed"]
    missing = _outputs()
    missing.pop(3)
    assert not runner.parity_summary(missing, _outputs())["passed"]
    assert not runner.parity_summary(_outputs(1.0), _outputs())["passed"]
    nonfinite = _outputs()
    nonfinite[0] = ([math.nan], [0.0, 1.0])
    assert not runner.parity_summary(nonfinite, _outputs())["passed"]


def test_target_cpu_contamination_subtracts_owned_work() -> None:
    runner = _load()
    before = {0: (1000, 500), 1: (2000, 1000)}
    after = {0: (1200, 550), 1: (2200, 1050)}
    # 300 busy ticks at 100 Hz less 2.5 owned seconds leaves 0.5 seconds.
    assert runner.target_cpu_contamination_seconds(before, after, 2.5, clock_ticks=100) == pytest.approx(0.5)
    assert runner.target_cpu_contamination_seconds(before, {0: after[0]}, 0.0, clock_ticks=100) is None


def test_cpu_busy_fraction() -> None:
    runner = _load()
    before = {0: (100, 50), 1: (200, 100)}
    after = {0: (200, 140), 1: (300, 150)}
    assert runner.cpu_busy_fractions(before, after) == pytest.approx({0: 0.1, 1: 0.5})
    assert runner.cpu_busy_fractions(before, {0: after[0]}) is None


def test_paired_statistics_and_holm_are_deterministic() -> None:
    runner = _load()
    candidate = [0.8, 0.9, 0.85, 0.82, 0.88]
    comparator = [1.0] * 5
    first = runner.paired_statistics(candidate, comparator, resamples=1000)
    second = runner.paired_statistics(candidate, comparator, resamples=1000)
    assert first == second
    assert first["bootstrap_95_interval"][1] < 1.0
    adjusted = runner.holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_worker_commands_bind_memory_then_exact_os_cpu_ids(tmp_path: Path) -> None:
    runner = _load()
    command = runner._worker_command(
        cpus=(16, 17),
        intra=2,
        mode="batch",
        method="analytical",
        row_ids=tuple(range(16)),
        dimension=2,
        parameter_count=3,
        timesteps=4,
    )
    assert command[:5] == [str(runner.HWLOC_BIND), "--membind", "node:0", "--", "taskset"]
    assert command[command.index("-c") + 1] == "16,17"
    assert command[command.index("--row-ids") + 1] == ",".join(str(value) for value in range(16))


def test_terminate_processes_cleans_descendant_group() -> None:
    runner = _load()
    process = subprocess.Popen(["sleep", "60"], start_new_session=True)
    runner.terminate_processes([process])
    assert process.poll() is not None


def test_phase_specs_are_bounded() -> None:
    runner = _load()
    assert runner._phase_specs("smoke") == ([(2, 3, 4)], (1, 2), 1)
    assert runner._phase_specs("nominate") == ([(20, 150, 120)], runner.CORE_BUDGETS, 5)


def test_topology_requires_distinct_primary_cores_and_declared_siblings(monkeypatch) -> None:
    runner = _load()
    lines = ["CPU NODE SOCKET CORE ONLINE"]
    lines.extend(f"{cpu} 0 0 {cpu} yes" for cpu in range(16, 32))
    lines.extend(f"{cpu + 128} 0 0 {cpu} yes" for cpu in range(16, 32))
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "\n".join(lines)})(),
    )
    assert runner.topology_contract()["valid"]
    lines[-1] = "159 0 0 99 yes"
    assert not runner.topology_contract()["valid"]


def test_confirmation_primary_is_selected_by_identity_not_sort_order() -> None:
    runner = _load()
    decisions = [
        {"workload": [10, 50, 120], "nominee_over_comparator_interval": [0.7, 0.8]},
        {"workload": [30, 50, 120], "nominee_over_comparator_interval": [0.8, 0.9]},
        {"workload": [30, 150, 120], "nominee_over_comparator_interval": [0.9, 1.1]},
    ]
    primary = next(row for row in decisions if row["workload"] == [30, 150, 120])
    assert primary["nominee_over_comparator_interval"] == [0.9, 1.1]
