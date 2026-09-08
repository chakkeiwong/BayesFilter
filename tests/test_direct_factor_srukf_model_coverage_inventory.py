from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "docs/plans/artifacts/direct-factor-srukf-model-coverage-20260817"
ALLOWED_STATUSES = {
    "eligible_score",
    "eligible_value_only",
    "adapter_required",
    "not_applicable_contract",
    "owner_excluded",
    "historical_only",
    "blocked",
}
EXPECTED_CANONICAL_IDS = {
    "lgssm_2d_h25_rich",
    "sv_1d_h18_rich",
    "range_bearing_4d_h20_rich",
    "structural_ar1_quadratic_h16",
    "spatial_sir_j3_rk4",
    "predator_prey_rk4",
    "LGSSM-EXACT",
    "PP-UKF",
    "PP-SGQF",
    "SIR-SGQF",
    "STR-UKF",
    "SVX-SGQF",
    "KSC-UKF",
    "PP-ZC",
    "STR-ZC",
    "SIR-ZC",
    "SVX-ZC",
    "SIR-UKF",
    "SSL-LSTM",
}


def _inventory() -> list[dict[str, object]]:
    payload = json.loads((ARTIFACT_ROOT / "model_inventory.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "bayesfilter.direct_factor_srukf_inventory.v1"
    rows = payload["rows"]
    assert isinstance(rows, list)
    return rows


def test_coverage_inventory_is_unique_and_classified() -> None:
    rows = _inventory()
    ids = [row["model_id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert all(row["status"] in ALLOWED_STATUSES for row in rows)
    assert all(row.get("source", {}).get("path") for row in rows)
    assert all(row.get("reason") for row in rows)


def test_canonical_registries_are_present_without_silent_promotion() -> None:
    rows = _inventory()
    ids = {row["model_id"] for row in rows}
    assert EXPECTED_CANONICAL_IDS <= ids
    assert all(
        row["status"] != "eligible_score" or row["contract"] == "TFFactorSRUKFModel"
        for row in rows
    )


def test_executed_rows_have_result_artifacts_and_score_boundaries() -> None:
    rows = _inventory()
    result_names = {
        "model_a_affine": "model_a_affine_result.json",
        "model_b_nonlinear_accumulation": "model_b_nonlinear_accumulation_result.json",
        "model_c_nonlinear_growth": "model_c_nonlinear_growth_result.json",
        "PP-UKF": "pp_ukf_result.json",
        "STR-UKF": "str_ukf_result.json",
        "structural_ar1_quadratic_h16": "structural_ar1_quadratic_h16_result.json",
    }
    for row in rows:
        if row["model_id"] not in result_names:
            continue
        result = json.loads((ARTIFACT_ROOT / result_names[row["model_id"]]).read_text(encoding="utf-8"))
        assert result["model_id"] == row["model_id"]
        if row["status"] == "eligible_score":
            assert result["score_claim"]
            assert result["branch_status"] == "fixed_full_rank_positive_pivot"
        if row["status"] == "eligible_value_only":
            assert result["score_claim"].startswith("none;")
            assert result["branch_status"] == "value_only_rank_discovery"
