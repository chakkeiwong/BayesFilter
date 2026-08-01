from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "docs/benchmarks/build_contract_e_tp_phase7_comparison.py"
LEDGER_PATH = ROOT / (
    "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/"
    "phase7_same_target_comparison_20260715/comparison_ledger_v2.json"
)


def _builder_module():
    spec = importlib.util.spec_from_file_location("contract_e_tp_phase7_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase7_builder_fails_closed_on_target_mismatch() -> None:
    module = _builder_module()
    payload = {
        "row_id": module.ACTUAL_ROW,
        "target": {
            "time_steps": 2,
            "theta": module.EXPECTED[module.ACTUAL_ROW]["theta"],
            "parameter_names": ["log_beta", "gamma_unconstrained"],
            "target_observation_policy": "exact_log_y_square_log_chi_square",
            "transition_before_first_observation": False,
        },
    }
    try:
        module._validate_scalar_target(payload, module.ACTUAL_ROW, 2)
    except ValueError as error:
        assert "parameter order mismatch" in str(error)
    else:
        raise AssertionError("target mismatch did not fail closed")


def test_phase7_ledger_preserves_method_boundaries() -> None:
    payload = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    assert payload["status"] == "PHASE7_COMPARISON_COMPLETE_WITH_EXPLICIT_GAPS"
    assert payload["row_count"] == 15
    assert payload["method_eligibility"]["zhao_cui_source_parameter_learning"] == "unavailable_all_rows"
    for row in payload["rows"]:
        assert "comparison_classification" in row
        assert row["contract_e_chol"]["status"] == "unavailable"
        assert row["zhao_cui_source_parameter_learning"]["status"] == "unavailable"
        if row.get("status") == "blocked_target_measure_mismatch":
            continue
        assert row["reference"]["status"] == "available"
        assert row["contract_e_tp"]["own_scalar_fd_status"] == "pass"
        assert row["comparison_classification"] == "descriptive_only_margin_unavailable"
        extension = row["fixed_parameter_adjacent_state_extension"]
        if extension["status"] == "available":
            assert extension["route_classification"] == "extension_or_invention"
            assert extension["own_scalar_fd_status"] == "pass"
            assert extension["first_step_time_order_status"] == "pass"


def test_generalized_t10_remains_negative_feature_result() -> None:
    payload = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    row = next(
        item
        for item in payload["rows"]
        if item["row_id"] == "zhao_cui_generalized_sv_synthetic_from_estimated_values"
        and item["horizon"] == 10
    )
    assert row["contract_e_tp"]["scientific_status"].startswith("negative_result")
    assert row["fixed_parameter_adjacent_state_extension"]["status"] == "available"
    assert row["zhao_cui_source_parameter_learning"]["status"] == "unavailable"
