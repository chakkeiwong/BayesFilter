from __future__ import annotations

import math

from docs.benchmarks.run_genut_austria_antithetic_ensemble import (
    FD_MINIMUM_STEPS,
    FD_RELATIVE_STEPS,
    LABELS,
    _assemble_ensembles,
    _bootstrap_log_ratio,
    _percentile,
    _select_prior_genut_row,
)


def test_zero_parameter_fd_steps_use_absolute_floors() -> None:
    realized = [
        max(minimum, relative * 0.0)
        for relative, minimum in zip(FD_RELATIVE_STEPS, FD_MINIMUM_STEPS)
    ]
    assert realized == [0.0004, 0.0008]


def _row(replicate: int, arm: str, slot: int, value: float, score: list[float]):
    return {
        "replicate": replicate,
        "arm": arm,
        "slot": slot,
        "value": value,
        "score": score,
        "runtime_seconds": 1.0,
        "valid": True,
        "severe_score_tail": False,
    }


def test_prefix_ensembles_are_equal_cost_and_average_complete_runs() -> None:
    rows = []
    for slot in range(8):
        rows.append(_row(0, "antithetic", slot, float(slot), [slot, slot, slot]))
        rows.append(
            _row(0, "independent", slot, float(10 + slot), [10 + slot] * 3)
        )
    ensembles = _assemble_ensembles(rows)
    assert len(ensembles) == 6
    by_key = {(row["arm"], row["K"]): row for row in ensembles}
    for k in (1, 2, 4):
        assert by_key[("antithetic", k)]["complete_run_count"] == 2 * k
        assert by_key[("independent", k)]["complete_run_count"] == 2 * k
    assert by_key[("antithetic", 2)]["value"] == 1.5
    assert by_key[("independent", 2)]["value"] == 11.5


def test_invalid_constituent_invalidates_prefix_without_dropping_ensemble() -> None:
    rows = []
    for slot in range(8):
        anti = _row(0, "antithetic", slot, float(slot), [slot, slot, slot])
        if slot == 1:
            anti["valid"] = False
            anti["value"] = None
            anti["score"] = None
        rows.append(anti)
        rows.append(_row(0, "independent", slot, float(slot), [slot, slot, slot]))
    ensembles = _assemble_ensembles(rows)
    antithetic = [row for row in ensembles if row["arm"] == "antithetic"]
    independent = [row for row in ensembles if row["arm"] == "independent"]
    assert all(not row["valid"] for row in antithetic)
    assert all(row["valid"] for row in independent)
    assert all(row["value"] is None and row["score"] is None for row in antithetic)


def test_bootstrap_detects_exact_antithetic_cancellation() -> None:
    independent = [[float(index), *[float(index)] * 3] for index in range(1, 9)]
    antithetic = [[0.0, 0.0, 0.0, 0.0] for _ in independent]
    rows = _bootstrap_log_ratio(
        antithetic,
        independent,
        [2.0] * len(independent),
        [2.0] * len(independent),
    )
    assert [row["label"] for row in rows] == list(LABELS)
    assert all(row["coordinate_nominated"] for row in rows)
    assert all(row["variance_ratio_antithetic_over_independent"] < 1e-10 for row in rows)


def test_percentile_uses_linear_interpolation() -> None:
    assert _percentile([0.0, 10.0], 0.25) == 2.5
    assert math.isclose(_percentile([0.0, 1.0, 2.0], 0.975), 1.95)


def test_prior_checkpoint_selects_genut_among_comparator_rows() -> None:
    selected = _select_prior_genut_row(
        {
            "row_id": "austria_sir_T20",
            "rows": [
                {"method": "genut", "controls": {"epsilon": 8.0}},
                {"method": "sgqf", "value": -682.0},
                {"method": "zhao_cui", "status": "blocked"},
            ],
        }
    )
    assert selected["method"] == "genut"
