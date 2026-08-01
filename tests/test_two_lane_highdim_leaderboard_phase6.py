from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py"
ROW_ID = "zhao_cui_generalized_sv_synthetic_from_estimated_values"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_two_lane_highdim_leaderboard", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load highdim leaderboard module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase6_fixed_sgqf_generalized_sv_uses_raw_y_manual_score_route() -> None:
    module = _load_module()
    row = module._apply_score_status(module._cell_for_fixed_sgqf(ROW_ID))

    assert row["comparison_status"] == "executed_value_score"
    assert row["numeric_execution_status"] == (
        "executed_generalized_sv_raw_y_sgqf_value_score"
    )
    assert row["target_contract_status"] == (
        "target_compatible_generalized_sv_prior_mean_raw_y_gaussian_projection"
    )
    assert row["score_status"] == "analytical_score_emitted"
    assert isinstance(row["log_likelihood"], float)
    assert len(row["score"]) == 3
    assert row["cloud_level"] == 3
    assert "GENERALIZED_SV_RAW_Y_SGQF_MANUAL_VALUE_SCORE_ROUTE" in row["reason_codes"]
    assert any("not exact nonlinear likelihood" in item for item in row["nonclaims"])


def test_phase6_zhao_cui_generalized_sv_remains_exact_source_row_blocker() -> None:
    module = _load_module()
    p8d_cell = module._p8d_cells()[("zhao_cui_scalar_or_multistate", ROW_ID)]
    row = module._apply_p91_zhao_cui_status(
        module._cell_from_p8d("zhao_cui_scalar_or_multistate", ROW_ID, p8d_cell)
    )

    assert row["comparison_status"] == "blocked_or_status_only"
    assert row["numeric_execution_status"] == (
        "blocked_generalized_sv_zhao_cui_source_row_evaluator_adapter_required"
    )
    assert row["target_contract_status"] == "blocked_exact_source_row_evaluator_missing"
    assert row["log_likelihood"] is None
    assert row["score_coordinate_system"] is None
    assert row["score_derivative_provenance"] is None
    assert row["score_l2_norm"] is None
    assert "GENERALIZED_SV_EXACT_SOURCE_ROW_ZHAO_CUI_EVALUATOR_REQUIRED" in row["reason_codes"]
    assert "PRECURSOR_NATIVE_ORACLE_AUXILIARY_ACTUAL_SV_KSC_NOT_ADMISSION_EVIDENCE" in row["reason_codes"]
    assert "blocked_source_row_evaluator_missing" in row["reason"]
    assert any("do not admit this Zhao-Cui source-row cell" in item for item in row["nonclaims"])
