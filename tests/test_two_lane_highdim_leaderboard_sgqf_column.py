from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sgqf_column_runner", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load highdim leaderboard module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sgqf_column_artifact_is_applicability_aware(monkeypatch) -> None:
    module = _load_module()

    def fake_cell(row_id: str):
        if row_id == module.PARAMETERIZED_SIR_ROW:
            return {
                "lane": "highdim_source_scope",
                "row_id": row_id,
                "algorithm_id": "fixed_sgqf",
                "comparison_status": "not_applicable",
                "numeric_execution_status": "not_applicable",
                "score_status": "not_applicable_to_scoped_component_row",
                "score": None,
                "score_l2_norm": None,
                "nonclaims": [],
            }
        value_only = row_id == module.FIXED_SIR_ROW
        return {
            "lane": "highdim_source_scope",
            "row_id": row_id,
            "algorithm_id": "fixed_sgqf",
            "comparison_status": (
                "executed_value_only" if value_only else "executed_value_score"
            ),
            "numeric_execution_status": "executed_test",
            "value_route_id": f"test_route_{row_id}",
            "score_route_id": f"test_route_{row_id}",
            "value_score_route_status": "same_route_value_score",
            "log_likelihood": -1.0,
            "score": None if value_only else [0.1],
            "score_l2_norm": None if value_only else 0.1,
            "score_coordinate_system": None if value_only else "theta",
            "score_derivative_provenance": (
                None if value_only else "manual_fixed_sgqf_test_score"
            ),
            "score_status": (
                "not_applicable_no_free_theta"
                if value_only
                else "analytical_score_emitted"
            ),
            "nonclaims": [],
        }

    monkeypatch.setattr(module, "_cell_for_fixed_sgqf", fake_cell)
    monkeypatch.setattr(module, "_attach_reset_source_data_identity", lambda row: row)
    monkeypatch.setattr(module, "_apply_score_status", lambda row: row)
    monkeypatch.setattr(module, "_enforce_analytical_score_admission", lambda rows: rows)
    monkeypatch.setattr(module, "_apply_value_score_route_contract", lambda rows: rows)
    monkeypatch.setattr(module, "_validate_analytical_score_contract", lambda rows: None)
    monkeypatch.setattr(module, "SGQF_GPU_EVIDENCE", {})

    artifact = module.build_sgqf_column_artifact()

    assert artifact["manifest"]["execution_mode"] == "row_selective_sgqf_column_only"
    assert artifact["sgqf_column_complete"] is True
    assert len(artifact["rows"]) == 7
    scoped = next(
        row for row in artifact["rows"] if row["row_id"] == module.PARAMETERIZED_SIR_ROW
    )
    assert scoped["algorithm_applicability"] == "not_applicable_scoped_row"
    assert scoped["required_result_kind"] == "not_applicable"
    assert scoped["comparison_status"] == "not_applicable"
    assert scoped["cell_result_complete"] is True


def test_sgqf_manual_provenance_names_analytical_route() -> None:
    module = _load_module()
    for row_id in (
        "zhao_cui_predator_prey_T20",
        "zhao_cui_generalized_sv_synthetic_from_estimated_values",
    ):
        row = module._cell_for_fixed_sgqf(row_id)
        row = module._apply_score_status(row)
        (row,) = module._apply_value_score_route_contract([row])
        module._validate_analytical_score_contract([row])
        provenance = row["score_derivative_provenance"].lower()
        assert "manual" in provenance
        assert "analytical" in provenance


def test_repository_issues_legacy_sgqf_route_identities() -> None:
    module = _load_module()
    for row_id in (
        "benchmark_lgssm_exact_oracle_m3_T50",
        "zhao_cui_sv_actual_nongaussian_T1000",
        "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
    ):
        row = module._apply_score_status(module._cell_for_fixed_sgqf(row_id))
        row = module._attach_reset_source_data_identity(row)
        (row,) = module._apply_value_score_route_contract([row])
        row = module._attach_repository_sgqf_route_identity(row)
        assert len(row["route_identity"]) == 64
        assert row["route_identity_manifest"]["row_id"] == row_id
        assert row["route_identity_manifest"]["data_observation_sha256"]
