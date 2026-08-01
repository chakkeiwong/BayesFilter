from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_two_lane_highdim_leaderboard_applicability", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load highdim leaderboard module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _full_matrix(module, default_status: str = "blocked") -> list[dict[str, object]]:
    return [
        {
            "row_id": row_id,
            "algorithm_id": algorithm_id,
            "comparison_status": default_status,
        }
        for row_id in module.HIGHDIM_ROWS
        for algorithm_id in module.HIGHDIM_ALGOS
    ]


def _set_status(
    rows: list[dict[str, object]], row_id: str, algorithm_id: str, status: str
) -> None:
    row = next(
        item
        for item in rows
        if item["row_id"] == row_id and item["algorithm_id"] == algorithm_id
    )
    row["comparison_status"] = status


def test_contract_classifies_fixed_and_scoped_sir_without_false_score_requirements() -> None:
    module = _load_module()
    rows = module._apply_applicability_contract(_full_matrix(module))
    by_key = {(row["row_id"], row["algorithm_id"]): row for row in rows}

    fixed_sir = by_key[(module.FIXED_SIR_ROW, "fixed_sgqf")]
    assert fixed_sir["algorithm_applicability"] == "applicable"
    assert fixed_sir["required_result_kind"] == "value_only_no_free_theta"
    assert fixed_sir["score_admitted"] is None

    scoped_sgqf = by_key[(module.PARAMETERIZED_SIR_ROW, "fixed_sgqf")]
    assert scoped_sgqf["algorithm_applicability"] == "not_applicable_scoped_row"
    assert scoped_sgqf["required_result_kind"] == "not_applicable"
    assert scoped_sgqf["cell_result_complete"] is True

    scoped_zhao_cui = by_key[
        (module.PARAMETERIZED_SIR_ROW, "zhao_cui_scalar_or_multistate")
    ]
    assert scoped_zhao_cui["algorithm_applicability"] == "applicable"
    assert scoped_zhao_cui["required_result_kind"] == "value_score"


def test_fixed_sir_is_comparison_ready_only_as_value_only() -> None:
    module = _load_module()
    rows = _full_matrix(module)
    for algorithm_id in module.HIGHDIM_ALGOS:
        _set_status(rows, module.FIXED_SIR_ROW, algorithm_id, "executed_value_only")
    contracted = module._apply_applicability_contract(rows)
    summary = next(
        item
        for item in module._row_summary_from_rows(contracted)
        if item["row_id"] == module.FIXED_SIR_ROW
    )

    assert summary["value_complete"] is True
    assert summary["score_complete"] is True
    assert summary["comparison_ready"] is True
    assert summary["full_three_way_ready"] is False

    _set_status(rows, module.FIXED_SIR_ROW, "fixed_sgqf", "executed_value_score")
    contracted = module._apply_applicability_contract(rows)
    summary = next(
        item
        for item in module._row_summary_from_rows(contracted)
        if item["row_id"] == module.FIXED_SIR_ROW
    )
    assert summary["value_complete"] is True
    assert summary["comparison_ready"] is False
    assert summary["sgqf_complete"] is False


def test_scoped_not_applicable_cells_do_not_count_as_blocked() -> None:
    module = _load_module()
    rows = _full_matrix(module)
    _set_status(
        rows,
        module.PARAMETERIZED_SIR_ROW,
        "zhao_cui_scalar_or_multistate",
        "executed_value_score",
    )
    contracted = module._apply_applicability_contract(rows)
    summary = next(
        item
        for item in module._row_summary_from_rows(contracted)
        if item["row_id"] == module.PARAMETERIZED_SIR_ROW
    )

    assert summary["row_scope"] == "scoped_component_row"
    assert summary["scoped_component_ready"] is True
    assert summary["comparison_ready"] is False
    assert summary["blocked_or_missing_algorithms"] == []


def test_sgqf_column_completion_is_independent_of_other_algorithms() -> None:
    module = _load_module()
    rows = _full_matrix(module)
    for row_id in module.HIGHDIM_ROWS:
        if row_id == module.PARAMETERIZED_SIR_ROW:
            continue
        status = (
            "executed_value_only"
            if row_id == module.FIXED_SIR_ROW
            else "executed_value_score"
        )
        _set_status(rows, row_id, "fixed_sgqf", status)
    contracted = module._apply_applicability_contract(rows)
    summary = module._row_summary_from_rows(contracted)

    assert module._sgqf_column_complete(summary) is True
    assert any(
        row["row_scope"] == "main_observed_data_filtering_row"
        and row["comparison_ready"] is False
        for row in summary
    )


def test_applicability_contract_fails_closed_on_incomplete_or_unknown_matrix() -> None:
    module = _load_module()
    rows = _full_matrix(module)

    with pytest.raises(ValueError, match="missing highdim leaderboard cells"):
        module._apply_applicability_contract(rows[:-1])
    with pytest.raises(ValueError, match="duplicate highdim leaderboard cell"):
        module._apply_applicability_contract(rows + [dict(rows[0])])

    unknown_row = [dict(row) for row in rows]
    unknown_row[0]["row_id"] = "unclassified_row"
    with pytest.raises(ValueError, match="unclassified highdim leaderboard row"):
        module._apply_applicability_contract(unknown_row)

    unknown_algorithm = [dict(row) for row in rows]
    unknown_algorithm[0]["algorithm_id"] = "unclassified_algorithm"
    with pytest.raises(ValueError, match="unclassified highdim leaderboard algorithm"):
        module._apply_applicability_contract(unknown_algorithm)


def test_live_fixed_sir_sgqf_cell_is_value_only_with_sealed_identity() -> None:
    module = _load_module()
    row = module._apply_score_status(module._cell_for_fixed_sgqf(module.FIXED_SIR_ROW))
    contracted = module._apply_applicability_contract(
        [
            row
            if row_id == module.FIXED_SIR_ROW and algorithm_id == "fixed_sgqf"
            else {
                "row_id": row_id,
                "algorithm_id": algorithm_id,
                "comparison_status": "blocked",
            }
            for row_id in module.HIGHDIM_ROWS
            for algorithm_id in module.HIGHDIM_ALGOS
        ]
    )
    actual = next(
        item
        for item in contracted
        if item["row_id"] == module.FIXED_SIR_ROW
        and item["algorithm_id"] == "fixed_sgqf"
    )

    assert actual["comparison_status"] == "executed_value_only"
    assert actual["numeric_execution_status"] == (
        "executed_fixed_sir_source_order_sgqf_value_only"
    )
    assert actual["log_likelihood"] is not None
    assert actual["score"] is None
    assert actual["score_status"] == "not_applicable_no_free_theta"
    assert actual["required_result_kind"] == "value_only_no_free_theta"
    assert actual["cell_result_complete"] is True
    assert len(actual["route_identity"]) == 64
    assert actual["time_order"] == "x0_then_20_transition_then_observe_steps_y1_y20"
    assert actual["cloud_point_count"] == 37
