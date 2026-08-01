from __future__ import annotations

from docs.benchmarks import run_lgssm_n5000_seed_batch_capacity as capacity


def _microbatch(seed: int, value: float, score: list[float]) -> dict[str, object]:
    return {
        "seeds": [seed],
        "result": {
            "per_seed_value": [value],
            "per_seed_physical_score": [score],
            "timing_seconds": {
                "trace": 1.0,
                "compile_plus_first_execution": 2.0,
                "warm_execution": 3.0,
            },
        },
    }


def test_parity_compares_per_seed_values_and_total_scores() -> None:
    baseline = [
        _microbatch(1, -2.0, [1.0, 2.0]),
        _microbatch(2, -3.0, [3.0, 4.0]),
    ]
    batched = {
        "per_seed_value": [-2.0, -3.0 + 5.0e-5],
        "per_seed_physical_score": [[1.0, 2.0], [3.0, 4.0 + 5.0e-5]],
    }
    result = capacity._parity(batched, baseline)
    assert result["pass"] is True
    assert result["maximum_absolute_value_difference"] < capacity.PARITY_ATOL
    assert result["maximum_absolute_score_difference"] < capacity.PARITY_ATOL


def test_parity_rejects_batch_dependent_score_drift() -> None:
    baseline = [_microbatch(1, -2.0, [1.0, 2.0])]
    batched = {
        "per_seed_value": [-2.0],
        "per_seed_physical_score": [[1.0, 2.0 + 2.0e-4]],
    }
    assert capacity._parity(batched, baseline)["pass"] is False


def test_speed_compares_equal_seed_work() -> None:
    baseline = {
        "trace_seconds": 8.0,
        "compile_plus_first_execution_seconds": 80.0,
        "warm_replay_seconds": 40.0,
        "trace_cold_and_replay_seconds": 128.0,
    }
    batched = {
        "timing_seconds": {
            "trace": 2.0,
            "compile_plus_first_execution": 20.0,
            "warm_execution": 10.0,
        }
    }
    result = capacity._speed(batched, baseline)
    assert result["cold_execution_speedup"] == 4.0
    assert result["warm_execution_speedup"] == 4.0
    assert result["trace_cold_and_replay_speedup"] == 4.0

