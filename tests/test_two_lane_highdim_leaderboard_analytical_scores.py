from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py"
_MODULE = None
_PAYLOAD = None


def _load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    spec = importlib.util.spec_from_file_location("benchmark_two_lane_highdim_leaderboard", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load highdim leaderboard module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE = module
    return module


def _build_payload():
    global _PAYLOAD
    if _PAYLOAD is None:
        _PAYLOAD = _load_module().build_artifact()
    return _PAYLOAD


def test_actual_leaderboard_score_rows_exclude_autodiff_provenance() -> None:
    payload = _build_payload()

    score_rows = [row for row in payload["rows"] if row["comparison_status"] == "executed_value_score"]
    assert score_rows
    for row in score_rows:
        provenance = str(row.get("score_derivative_provenance") or "").lower()
        assert "autodiff" not in provenance, (row["row_id"], row["algorithm_id"], provenance)
        assert "gradienttape" not in provenance, (row["row_id"], row["algorithm_id"], provenance)
        assert "gradient_tape" not in provenance, (row["row_id"], row["algorithm_id"], provenance)
        if row["algorithm_id"] == "ukf":
            assert "svd" not in provenance, (row["row_id"], row["algorithm_id"], provenance)
            assert "eigenderivative" not in provenance, (row["row_id"], row["algorithm_id"], provenance)


def test_admitted_value_score_rows_use_one_declared_route() -> None:
    module = _load_module()
    score_rows = module._apply_value_score_route_contract(
        [
            {
                "row_id": "zhao_cui_sv_actual_nongaussian_T1000",
                "algorithm_id": "ukf",
                "comparison_status": "executed_value_score",
                "score": [0.1, 0.2],
                "score_derivative_provenance": (
                    "actual_sv_augmented_noise_factor_propagating_srukf_manual_score"
                ),
                "nonclaims": [],
            },
            {
                "row_id": "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
                "algorithm_id": "ukf",
                "comparison_status": "executed_value_score",
                "score": [0.1, 0.2],
                "score_derivative_provenance": (
                    "principal_sqrt_ukf_independent_panel_transformed_sv_gaussian_mixture_"
                    "analytical_component_score_logsumexp_aggregation"
                ),
                "nonclaims": [],
            },
        ]
    )

    module._validate_analytical_score_contract(score_rows)
    for row in score_rows:
        assert row["value_route_id"] == row["score_route_id"], (
            row["row_id"],
            row["algorithm_id"],
            row.get("value_route_id"),
            row.get("score_route_id"),
        )
        assert row["value_score_route_status"] == "same_route_value_score"


def test_score_contract_rejects_missing_or_mixed_value_score_routes() -> None:
    module = _load_module()
    base = {
        "row_id": "benchmark_lgssm_exact_oracle_m3_T50",
        "algorithm_id": "fixed_sgqf",
        "comparison_status": "executed_value_score",
        "numeric_execution_status": "executed_direct_affine_sgqf_value_score",
        "score": [1.0],
        "score_l2_norm": 1.0,
        "score_coordinate_system": "theta",
        "score_derivative_provenance": "fixed_sgqf_analytic_first_order_fixed_branch_affine_lgssm_score",
    }

    with pytest.raises(ValueError, match="requires value_route_id"):
        module._validate_analytical_score_contract([dict(base)])

    mixed = dict(
        base,
        value_route_id="fixed_sgqf_direct_affine_lgssm",
        score_route_id="other_score_route",
        value_score_route_status="same_route_value_score",
    )
    with pytest.raises(ValueError, match="mixes value route"):
        module._validate_analytical_score_contract([mixed])


def test_score_contract_rejects_zhao_cui_without_fixed_variant_hmc_identity() -> None:
    module = _load_module()
    row = {
        "row_id": "zhao_cui_sv_actual_nongaussian_T1000",
        "algorithm_id": "zhao_cui_scalar_or_multistate",
        "comparison_status": "executed_value_score",
        "score": [1.0],
        "score_derivative_provenance": "manual_parameter_score_methods_only",
        "value_route_id": "historical_route",
        "score_route_id": "historical_route",
        "value_score_route_status": "same_route_value_score",
    }
    with pytest.raises(ValueError, match="fixed-variant HMC filtering route"):
        module._validate_analytical_score_contract([row])


def test_autodiff_score_rows_are_value_only_diagnostic_rows() -> None:
    payload = _build_payload()

    demoted = [
        row
        for row in payload["rows"]
        if row.get("score_status") == "blocked_autodiff_not_admitted"
    ]
    assert demoted
    for row in demoted:
        assert row["comparison_status"] == "executed_value_only"
        assert row["numeric_execution_status"] == "executed_numeric_value_only_autodiff_score_not_admitted"
        assert row["score_l2_norm"] is None
        assert row["score_coordinate_system"] is None
        assert row["phase7_batch_gpu_xla_status"]["timing_rank_status"] == (
            "not_rankable_correctness_gate_open"
        )
        assert "analytical gradient accuracy" in row["score_status_reason"]


def test_historical_svd_ukf_score_rows_are_value_only_diagnostics() -> None:
    module = _load_module()
    rows = [
        {
            "row_id": "synthetic_row",
            "algorithm_id": "ukf",
            "comparison_status": "executed_value_score",
            "numeric_execution_status": "executed_numeric_value_score",
            "score": [1.0],
            "score_l2_norm": 1.0,
            "score_coordinate_system": "theta",
            "score_derivative_provenance": "tf_svd_ukf_score_historical_eigenderivative",
            "nonclaims": [],
        }
    ]

    demoted = module._enforce_analytical_score_admission(rows)

    assert demoted[0]["comparison_status"] == "executed_value_only"
    assert demoted[0]["numeric_execution_status"] == (
        "executed_numeric_value_only_historical_svd_ukf_score_not_admitted"
    )
    assert demoted[0]["score_status"] == "blocked_historical_svd_ukf_not_admitted"
    assert demoted[0]["score"] is None
    assert "principal-square-root UKF" in demoted[0]["score_status_reason"]


def test_zhao_cui_score_admission_requires_fixed_variant_filter_identity() -> None:
    module = _load_module()
    base = {
        "row_id": "zhao_cui_sv_actual_nongaussian_T1000",
        "algorithm_id": "zhao_cui_scalar_or_multistate",
        "comparison_status": "executed_value_score",
        "numeric_execution_status": "executed_test_value_score",
        "score": [1.0, -0.25],
        "score_l2_norm": 1.0308,
        "score_coordinate_system": "theta",
        "score_derivative_provenance": (
            "zhao_cui_manual_parameter_score_methods_only"
        ),
        "nonclaims": [],
    }
    historical = dict(
        base,
        route_id="zhao_cui_predator_prey_t20_multistate_fixed_design_tt",
        route_role="historical_diagnostic_only",
    )
    fixed = dict(
        base,
        route_id="zhao_cui_exact_transformed_sv_fixed_branch_tt",
        route_role="fixed_variant_analytical_hmc_route",
        hmc_route_policy_id="zhao_cui_fixed_variant_hmc_default_v1",
        hmc_target_scope_admitted=True,
    )

    historical_row, fixed_row = module._enforce_analytical_score_admission(
        [historical, fixed]
    )

    assert historical_row["comparison_status"] == "executed_value_only"
    assert historical_row["score"] is None
    assert fixed_row["comparison_status"] == "executed_value_score"
    assert fixed_row["score"] == [1.0, -0.25]


def test_legacy_zhao_cui_sv_signature_receives_repository_route_identity() -> None:
    module = _load_module()
    row = {
        "row_id": "zhao_cui_sv_actual_nongaussian_T1000",
        "algorithm_id": "zhao_cui_scalar_or_multistate",
        "comparison_status": "executed_value_score",
        "numeric_execution_status": (
            "executed_zhao_cui_exact_transformed_sv_fixed_branch_tt_value_score"
        ),
        "score": [1.0, -0.25],
        "score_l2_norm": 1.0308,
        "score_coordinate_system": "theta",
        "score_derivative_provenance": (
            "zhao_cui_scalar_fixed_branch_tt_exact_transformed_sv_"
            "manual_parameter_score_methods_only"
        ),
        "nonclaims": [],
    }

    (normalized,) = module._enforce_analytical_score_admission([row])

    assert normalized["comparison_status"] == "executed_value_score"
    assert normalized["route_id"] == "zhao_cui_exact_transformed_sv_fixed_branch_tt"
    assert normalized["hmc_target_scope_admitted"] is True


def test_row_summary_full_ready_requires_all_algorithms_with_admitted_scores() -> None:
    payload = _build_payload()

    by_row = {row["row_id"]: row for row in payload["row_summary"]}
    assert by_row["benchmark_lgssm_exact_oracle_m3_T50"]["full_three_way_ready"] is False
    assert by_row["zhao_cui_sv_actual_nongaussian_T1000"]["full_three_way_ready"] is True
    assert by_row["zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000"]["full_three_way_ready"] is True
    assert by_row["zhao_cui_predator_prey_T20"]["full_three_way_ready"] is False
    assert by_row["zhao_cui_generalized_sv_synthetic_from_estimated_values"][
        "full_three_way_ready"
    ] is False


def test_ukf_sv_rows_use_only_admitted_principal_sqrt_scores() -> None:
    payload = _build_payload()

    rows = {
        (row["row_id"], row["algorithm_id"]): row
        for row in payload["rows"]
    }
    actual_sv = rows[("zhao_cui_sv_actual_nongaussian_T1000", "ukf")]
    assert actual_sv["comparison_status"] == "executed_value_score"
    assert actual_sv["numeric_execution_status"] == "executed_actual_sv_augmented_noise_srukf_value_score"
    assert actual_sv["score_status"] == "analytical_score_emitted"
    assert actual_sv["score"] is not None
    assert actual_sv["average_log_likelihood"] is not None
    provenance = str(actual_sv["score_derivative_provenance"]).lower()
    assert "factor_propagating_srukf_manual_score" in provenance
    assert "svd" not in provenance
    assert "autodiff" not in provenance
    assert "gradient" not in provenance
    assert "factor-propagating SR-UKF manual route" in actual_sv["score_status_reason"]
    assert any(
        "gamma score nearly zero structurally"
        in nonclaim
        for nonclaim in actual_sv["nonclaims"]
    )

    ksc = rows[("zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000", "ukf")]
    assert ksc["comparison_status"] == "executed_value_score"
    assert ksc["numeric_execution_status"] == "executed_principal_sqrt_ukf_ksc_mixture_value_score"
    assert ksc["score_status"] == "analytical_score_emitted"
    provenance = str(ksc["score_derivative_provenance"]).lower()
    assert "principal_sqrt" in provenance
    assert "svd" not in provenance
    assert "autodiff" not in provenance
    assert ksc["score"] is not None


def test_actual_sv_ukf_cell_uses_reviewed_srukf_score_without_full_payload_rebuild() -> None:
    module = _load_module()
    row = module._actual_sv_augmented_noise_srukf_ukf_cell()

    assert row["comparison_status"] == "executed_value_score"
    assert row["numeric_execution_status"] == "executed_actual_sv_augmented_noise_srukf_value_score"
    assert row["score_status"] == "analytical_score_emitted"
    assert row["score"] is not None
    provenance = str(row["score_derivative_provenance"]).lower()
    assert "factor_propagating_srukf_manual_score" in provenance
    assert "svd" not in provenance
    assert "autodiff" not in provenance
    assert "gradient" not in provenance
    assert any("gamma score nearly zero structurally" in item for item in row["nonclaims"])


def test_zhao_cui_sv_cells_use_fixed_variant_manual_scores_without_full_payload_rebuild(monkeypatch) -> None:
    module = _load_module()
    full_observations = module._sv_observations()
    original_config = module._zhao_cui_scalar_tt_config
    monkeypatch.setattr(module, "_sv_observations", lambda: full_observations[:4])
    monkeypatch.setattr(
        module,
        "_zhao_cui_scalar_tt_config",
        lambda seed: original_config(
            seed,
            basis_degree=16,
            quadrature_order=41,
        ),
    )
    actual_sv = module._zhao_cui_actual_sv_tt_cell()
    ksc = module._zhao_cui_ksc_tt_cell()

    for row, target_phrase in (
        (actual_sv, "exact transformed SV"),
        (ksc, "KSC"),
    ):
        assert row["algorithm_id"] == "zhao_cui_scalar_or_multistate"
        assert row["row_id"] in {
            "zhao_cui_sv_actual_nongaussian_T1000",
            "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
        }
        assert row["comparison_status"] == "executed_value_score"
        assert row["route_role"] == "fixed_variant_analytical_hmc_route"
        assert row["hmc_target_scope_admitted"] is True
        assert row["score_coordinate_system"] == "theta=[probit_gamma,log_beta]"
        assert row["score_status"] == "analytical_score_emitted"
        assert row["score"] is not None
        provenance = str(row["score_derivative_provenance"]).lower()
        assert "manual_parameter_score_methods_only" in provenance
        assert "autodiff" not in provenance
        assert "gradienttape" not in provenance
        assert "finite_difference" not in provenance
        assert "fd_" not in provenance
        assert any(target_phrase.lower() in item.lower() for item in row["nonclaims"])
