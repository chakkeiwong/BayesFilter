from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/benchmark_ssl_lstm_matrix_free_gpu_xla_2026_07_20.py"


def load_runner():
    name = "ssl_lstm_matrix_free_gpu_xla_benchmark_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def _worker(*, seconds: float, value_shift: float = 0.0, allocator: int = 100, host: int = 200):
    rows = []
    for index in range(runner.WARM_POINTS):
        rows.append(
            {
                "point_index": index,
                "seconds": seconds,
                "value": float(index) + value_shift,
                "score": [float(index + offset) + value_shift for offset in range(4)],
            }
        )
    return {
        "warm_rows": rows,
        "warm_seconds_median": seconds,
        "gpu_allocator_memory": {"peak_bytes": allocator},
        "process_memory": {"vmhwm_bytes": host, "ru_maxrss_bytes": host},
    }


def _cells(*, ratio: float = 0.75, contaminated: bool = False, value_shift: float = 0.0):
    cells = []
    for q in runner.Q_VALUES:
        for repetition in range(runner.REPETITIONS):
            for arm in runner.ARMS:
                cells.append(
                    {
                        "q": q,
                        "repetition": repetition,
                        "arm": arm,
                        "prelaunch": {"timing_contaminated": contaminated},
                        "worker": _worker(
                            seconds=1.0 if arm == "dense" else ratio,
                            value_shift=value_shift if arm == "jvp" else 0.0,
                        ),
                    }
                )
    return cells


def test_contract_and_schedule_are_bounded_and_alternating() -> None:
    payload = runner.contract_payload()
    assert payload["q_values"] == [5, 10, 20]
    assert payload["arms"] == ["dense", "jvp"]
    assert payload["repetitions"] == 3
    assert payload["warm_points"] == 5
    assert payload["cell_count"] == 18
    assert payload["material_execution_authorized"] is False
    assert runner.planned_cells()[:6] == [
        (5, 0, "dense"),
        (5, 0, "jvp"),
        (5, 1, "jvp"),
        (5, 1, "dense"),
        (5, 2, "dense"),
        (5, 2, "jvp"),
    ]


def test_q20_large_uncontaminated_effect_nominates_bounded_rerun() -> None:
    summary = runner.summarize_cells(_cells(ratio=0.75))
    assert summary["hard_vetoes"] == []
    assert summary["downstream_q20_rerun_nominated"] is True
    assert summary["decision"] == "NOMINATE_BOUNDED_Q20_TARGET_NEUTRA_CAPACITY_RERUN"


def test_parity_failure_is_hard_veto_and_blocks_nomination() -> None:
    summary = runner.summarize_cells(_cells(ratio=0.75, value_shift=1.0e-4))
    assert summary["hard_vetoes"]
    assert summary["downstream_q20_rerun_nominated"] is False


def test_contamination_blocks_nomination_without_numerical_veto() -> None:
    summary = runner.summarize_cells(_cells(ratio=0.50, contaminated=True))
    assert summary["hard_vetoes"] == []
    assert summary["downstream_q20_rerun_nominated"] is False


def test_small_q_slowdown_does_not_stop_q20_nomination() -> None:
    cells = _cells(ratio=0.75)
    for cell in cells:
        if cell["q"] in (5, 10) and cell["arm"] == "jvp":
            cell["worker"] = _worker(seconds=1.20)
    summary = runner.summarize_cells(cells)
    assert all(row["small_q_regression_signal"] for row in summary["rows"][:2])
    assert summary["downstream_q20_rerun_nominated"] is True
