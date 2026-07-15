from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "docs/benchmarks/run_kalman_qr_cpu_xla_formulation_shootout_2026_07_15.py"


def _load():
    name = "kalman_qr_cpu_xla_formulation_shootout_test_subject"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _output(offset: float = 0.0):
    return {
        "value": [float(row) + offset for row in range(4)],
        "score": [[float(row), float(row + 1)] for row in range(4)],
    }


def _record(name: str, seconds: float, *, passed: bool = True):
    return {
        "formulation": name,
        "status": "passed" if passed else "failed",
        "parity": {"passed": passed},
        "timing": {"measured_median_seconds": seconds},
    }


def test_formulation_boundaries_keep_references_out_of_nomination() -> None:
    runner = _load()
    assert "native_batch" not in runner.SINGLE_PROCESS_CANDIDATES
    assert "sequential_b1_calls" not in runner.SINGLE_PROCESS_CANDIDATES
    assert "vectorized_strict" in runner.SINGLE_PROCESS_CANDIDATES
    assert "vectorized_fallback" in runner.SINGLE_PROCESS_CANDIDATES
    assert len(runner.FORMULATIONS) == len(set(runner.FORMULATIONS))


def test_cpu_pool_uses_distinct_primary_cores_and_excludes_siblings() -> None:
    runner = _load()
    assert runner.cpu_list(4) == (16, 17, 18, 19)
    assert runner.cpu_list(16) == tuple(range(16, 32))
    assert not set(runner.cpu_list(16)).intersection(runner.EXCLUDED_SMT_SIBLINGS)
    with pytest.raises(ValueError):
        runner.cpu_list(17)


def test_worker_command_binds_memory_and_exact_cpu_ids(tmp_path: Path) -> None:
    runner = _load()
    command = runner.worker_command(
        "vectorized_strict",
        dimension=2,
        parameter_count=3,
        timesteps=4,
        batch_size=4,
        record_path=tmp_path / "record.json",
    )
    assert command[:5] == [str(runner.HWLOC_BIND), "--membind", "node:0", "--", "taskset"]
    assert command[command.index("-c") + 1] == "16,17,18,19"
    assert command[command.index("--intra") + 1] == "4"
    assert command[command.index("--formulation") + 1] == "vectorized_strict"


def test_repo_relative_resolves_relative_output_paths() -> None:
    runner = _load()
    path = Path("docs/benchmarks/example/record.json")
    assert runner.repo_relative(path) == "docs/benchmarks/example/record.json"


def test_parity_requires_shapes_finiteness_and_tolerance() -> None:
    runner = _load()
    assert runner.parity_summary(_output(), _output())["passed"]
    assert not runner.parity_summary(_output(1.0), _output())["passed"]
    missing = _output()
    missing["score"] = missing["score"][:-1]
    assert not runner.parity_summary(missing, _output())["passed"]
    nonfinite = _output()
    nonfinite["score"][0][0] = math.nan
    assert not runner.parity_summary(nonfinite, _output())["passed"]


def test_nomination_requires_twenty_percent_single_process_repair() -> None:
    runner = _load()
    records = [
        _record("native_batch", 10.0),
        _record("vectorized_strict", 7.9),
        _record("map_parallel_16", 8.1),
        _record("static_unrolled", 7.0, passed=False),
        _record("sequential_b1_calls", 2.0),
    ]
    result = runner.nomination_summary(records)
    assert result["status"] == "candidate_nominated"
    assert result["candidate"] == {"formulation": "vectorized_strict", "ratio": 0.79}
    assert [row["formulation"] for row in result["eligible"]] == ["vectorized_strict"]


def test_nomination_fails_closed_without_native_baseline() -> None:
    runner = _load()
    result = runner.nomination_summary([_record("vectorized_strict", 1.0)])
    assert result == {"status": "invalid_native_baseline", "candidate": None, "eligible": []}


def test_hlo_census_counts_only_named_operations() -> None:
    runner = _load()
    hlo = """
      %a = f32[] custom-call(%x), custom_call_target="Qr"
      %b = f32[] triangular-solve(%x, %y)
      %c = f32[] while(%b), condition=%cond, body=%body
      %d = f32[] dot(%x, %y)
      %e = f32[] fusion(%d), kind=kLoop
      %f = f32[] map(%e), dimensions={0}
    """
    assert runner.hlo_census(hlo) == {
        "while": 1,
        "custom_call": 1,
        "qr_target": 1,
        "triangular_solve": 1,
        "dot": 1,
        "fusion": 1,
        "map": 1,
    }


def test_contamination_subtracts_owned_process_cpu_time() -> None:
    runner = _load()
    before = {16: (1000, 500), 17: (2000, 1000)}
    after = {16: (1200, 550), 17: (2200, 1050)}
    assert runner.contamination_seconds(before, after, 2.5, clock_ticks=100) == pytest.approx(0.5)
    assert runner.contamination_seconds(before, {16: after[16]}, 0.0, clock_ticks=100) is None


def test_paired_statistics_are_deterministic() -> None:
    runner = _load()
    candidate = [0.70, 0.72, 0.71, 0.69, 0.73, 0.70, 0.71, 0.72]
    baseline = [1.0] * 8
    first = runner.paired_statistics(candidate, baseline, resamples=1000)
    second = runner.paired_statistics(candidate, baseline, resamples=1000)
    assert first == second
    assert first["bootstrap_95_interval"][1] < 0.90
    assert first["paired_block_count"] == 8


def test_phase_specs_bound_the_ladder() -> None:
    runner = _load()
    assert runner.phase_specs("smoke") == (2, 3, 4, 4)
    assert runner.phase_specs("canary") == (10, 50, 120, 16)
    assert runner.phase_specs("transfer") == (30, 50, 120, 16)
