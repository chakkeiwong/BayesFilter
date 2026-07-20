from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/benchmark_ssl_lstm_precision_gpu_xla_2026_07_20.py"


def _load():
    name = "ssl_lstm_precision_gpu_xla_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load()


def _worker(arm: str, *, seconds: float, shift: float = 0.0, floors: int = 0):
    dtype = "float64" if arm != "all_float32_tf32" else "float32"
    rows = [
        {
            "value": float(i) + shift,
            "score": [float(i + j) + shift for j in range(4)],
            "placement_floor_count": floors,
            "innovation_floor_count": 0,
        }
        for i in range(runner.WARM_POINTS)
    ]
    return {
        "warm_rows": rows,
        "warm_seconds_median": seconds,
        "gpu_allocator_memory": {"peak_bytes": 100},
        "run_manifest": {"storage_dtype": dtype},
    }


def _cells(*, mixed_shift: float = 0.0, float32_floors: int = 1):
    cells = []
    for q in runner.Q_VALUES:
        for repetition in range(runner.REPETITIONS):
            for arm in runner.ARMS:
                shift = mixed_shift if arm == "mixed_lstm32_filter64" else 0.0
                floors = float32_floors if arm == "all_float32_tf32" else 0
                cells.append(
                    {
                        "q": q,
                        "repetition": repetition,
                        "arm": arm,
                        "prelaunch": {"timing_contaminated": False},
                        "worker": _worker(
                            arm,
                            seconds={
                                "all_float64": 1.0,
                                "mixed_lstm32_filter64": 0.8,
                                "all_float32_tf32": 0.5,
                            }[arm],
                            shift=shift,
                            floors=floors,
                        ),
                    }
                )
    return cells


def test_contract_is_bounded_and_alternates_arm_order() -> None:
    cells = runner.planned_cells()
    assert len(cells) == 18
    assert cells[:6] == [
        (5, 0, "all_float64"),
        (5, 0, "mixed_lstm32_filter64"),
        (5, 0, "all_float32_tf32"),
        (5, 1, "all_float32_tf32"),
        (5, 1, "mixed_lstm32_filter64"),
        (5, 1, "all_float64"),
    ]


def test_mixed_passes_while_float32_floor_branch_vetoes() -> None:
    summary = runner.summarize(_cells())
    assert summary["hard_vetoes"] == []
    for row in summary["rows"]:
        assert row["arms"]["mixed_lstm32_filter64"]["accuracy_passed"] is True
        assert row["arms"]["all_float32_tf32"]["accuracy_passed"] is False
        assert row["arms"]["all_float32_tf32"]["floor_branch_changed"] is True


def test_accuracy_error_vetoes_mixed_candidate() -> None:
    summary = runner.summarize(_cells(mixed_shift=1.0e-2))
    assert all(
        not row["arms"]["mixed_lstm32_filter64"]["accuracy_passed"]
        for row in summary["rows"]
    )
