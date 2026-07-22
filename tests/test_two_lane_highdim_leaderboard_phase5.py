from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_two_lane_highdim_leaderboard", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load highdim leaderboard module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase5_zhao_cui_sir_keeps_p91_sidecar_but_blocks_full_filtering_row() -> None:
    module = _load_module()
    p8d_cell = module._p8d_cells()[
        ("zhao_cui_scalar_or_multistate", "zhao_cui_spatial_sir_austria_j9_T20")
    ]
    row = module._apply_p91_zhao_cui_status(
        module._cell_from_p8d(
            "zhao_cui_scalar_or_multistate",
            "zhao_cui_spatial_sir_austria_j9_T20",
            p8d_cell,
        )
    )

    assert row["comparison_status"] == "blocked_or_status_only"
    assert row["numeric_execution_status"] == (
        "blocked_full_filtering_evaluator_pending_p91_local_component_ready"
    )
    assert row["target_contract_status"] == (
        "full_filtering_blocked_local_complete_data_component_ready"
    )
    assert "P91_SCOPED_LOCAL_COMPLETE_DATA_READY" in row["reason_codes"]
    assert "FULL_FILTERING_LEADERBOARD_CELL_STILL_BLOCKED" in row["reason_codes"]
    assert "full observed-data/filtering leaderboard evaluator remains blocked" in row["reason"]

    sidecar = row["p91_scoped_evidence"]
    assert sidecar["scope"] == "local_complete_data_zhao_cui_sir_d18_component"
    assert sidecar["status"] == "P91_SCOPED_PRODUCTION_READY_CLOSED"
    assert sidecar["score_identity_status"] == "PASS_P91_PHASE4_LOCAL_COMPONENT_SCORE_IDENTITY"
    assert sidecar["gpu_xla_status"] == "PASS_P91_PHASE5_GPU_XLA_JIT_LOCAL_COMPLETE_DATA"
    assert sidecar["benchmark_status"] == "PASS_P91_PHASE6_PERFORMANCE_BENCHMARK"
    assert sidecar["hmc_smoke_status"] == "PASS_P91_PHASE7_HMC_SMOKE"

    joined_nonclaims = " ".join(row["nonclaims"] + sidecar["nonclaims"])
    assert "not full observed-data/filtering score identity" in joined_nonclaims
    assert "not exact likelihood correctness" in joined_nonclaims
    assert "not posterior correctness or convergence" in joined_nonclaims


def test_phase5_parameterized_sir_scoped_component_row_uses_manual_score() -> None:
    module = _load_module()
    row = module._zhao_cui_parameterized_sir_local_complete_data_cell()

    assert row["row_id"] == module.PARAMETERIZED_SIR_ROW
    assert row["algorithm_id"] == "zhao_cui_scalar_or_multistate"
    assert row["comparison_status"] == "executed_value_score"
    assert row["row_admission_status"] == "scoped_component_row_admitted"
    assert row["target_scope"] == "local_complete_data_zhao_cui_sir_d18_component"
    assert row["hmc_target_scope_admitted"] is True
    assert row["target_contract_status"] == "target_compatible_scoped_local_complete_data_component"
    assert row["route_role"] == "fixed_variant_zhao_cui_source_route"
    assert row["retained_grid_leaderboard_admission"] == (
        "not_admitted_for_production_leaderboard_use_fixed_variant_zhao_cui"
    )
    assert row["score_status"] == "analytical_score_emitted"
    assert row["score_derivative_provenance"] == (
        "zhao_cui_sir_d18_local_complete_data_manual_parameter_score_methods"
    )
    assert isinstance(row["log_likelihood"], float)
    assert isinstance(row["average_log_likelihood"], float)
    assert isinstance(row["score"], list)
    assert len(row["score"]) == 3
    assert row["score_l2_norm"] is not None

    joined = " ".join(row["reason_codes"] + row["nonclaims"])
    assert "FULL_FILTERING_LEADERBOARD_CELL_STILL_NOT_CLAIMED" in joined
    assert "row id encodes parameterization only" in joined
    assert "not full observed-data/filtering" in joined


def test_phase5_scoped_parameterized_sir_requires_metadata_guard_for_score_admission() -> None:
    module = _load_module()
    row = module._zhao_cui_parameterized_sir_local_complete_data_cell()
    broken = dict(row)
    broken.pop("target_scope")

    (demoted,) = module._enforce_analytical_score_admission([broken])

    assert demoted["comparison_status"] == "blocked"
    assert demoted["numeric_execution_status"] == (
        "blocked_scoped_component_metadata_guard_failed"
    )
    assert demoted["score_status"] == "blocked_scoped_component_metadata_guard_failed"
    assert demoted["score"] is None
    assert "row id alone is insufficient" in " ".join(demoted["nonclaims"])


def test_phase5_split_merge_cached_baseline_admits_scoped_row_with_nonclaim() -> None:
    module = _load_module()
    baseline_path = (
        module.ROOT
        / "docs/plans/bayesfilter-two-lane-highdim-leaderboard-results-2026-07-01.json"
    )
    patch_path = (
        module.ROOT
        / "docs/plans/bayesfilter-two-lane-highdim-leaderboard-results-2026-07-03-scoped-zhaocui-sir-row.json"
    )
    payload = module.build_artifact_from_cached_baseline(
        module._load(baseline_path),
        module._load(patch_path),
        baseline_path=baseline_path,
        scoped_patch_path=patch_path,
    )

    assert payload["metadata_date"] == "2026-07-03"
    assert payload["manifest"]["execution_mode"] == (
        "split_merge_cached_july1_full_leaderboard_plus_validated_"
        "scoped_zhaocui_sir_component_row"
    )
    assert "not evidence that unrelated expensive rows were rerun" in " ".join(
        payload["nonclaims"]
    )

    row = next(
        item
        for item in payload["rows"]
        if item["row_id"] == module.PARAMETERIZED_SIR_ROW
        and item["algorithm_id"] == "zhao_cui_scalar_or_multistate"
    )
    assert row["comparison_status"] == "executed_value_score"
    assert isinstance(row["score"], list)
    assert row["score_status"] == "analytical_score_emitted"
    assert row["row_admission_status"] == "scoped_component_row_admitted"
    assert row["target_scope"] == "local_complete_data_zhao_cui_sir_d18_component"
    assert row["score_derivative_provenance"] == (
        "zhao_cui_sir_d18_local_complete_data_manual_parameter_score_methods"
    )
    assert row["retained_grid_leaderboard_admission"] == (
        "not_admitted_for_production_leaderboard_use_fixed_variant_zhao_cui"
    )

    scoped_summary = next(
        item
        for item in payload["row_summary"]
        if item["row_id"] == module.PARAMETERIZED_SIR_ROW
    )
    assert scoped_summary["row_scope"] == "scoped_component_row"
    assert scoped_summary["scoped_component_ready"] is True
    assert scoped_summary["full_three_way_ready"] is False

    for admitted in payload["rows"]:
        if admitted["comparison_status"] != "executed_value_score":
            continue
        provenance = str(admitted.get("score_derivative_provenance") or "").lower()
        assert "autodiff" not in provenance
        assert "gradienttape" not in provenance
        assert "gradient_tape" not in provenance
