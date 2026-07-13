from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/benchmark_two_lane_highdim_ledh_leaderboard.py"
SAME_TARGET_LGSSM_SCRIPT = (
    ROOT / "docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py"
)
RESULTS_SCRIPT = (
    ROOT / "docs/benchmarks/benchmark_two_lane_highdim_ledh_inclusive_results.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_two_lane_highdim_ledh_leaderboard",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load LEDH-inclusive leaderboard module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_results_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_two_lane_highdim_ledh_inclusive_results",
        RESULTS_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load LEDH-inclusive results module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ledh_dry_run_emits_all_rows_and_algorithms() -> None:
    module = _load_module()
    payload = module.build_artifact()

    assert payload["comparator_mode"] == "frozen_non_ledh_baseline_plus_fresh_ledh"
    assert payload["runtime_cross_ranking_allowed"] is False
    assert len(payload["row_summary"]) == 7

    for summary in payload["row_summary"]:
        assert set(summary["comparison_algorithms"]) == {
            "fixed_sgqf",
            "ukf",
            "zhao_cui_scalar_or_multistate",
            "ledh_pfpf_ot",
        }
        assert summary["full_four_way_ready"] is False
        assert summary["ledh_runtime_rankable"] is False


def test_ledh_dry_run_has_explicit_blocked_reasons_for_nonexecuted_arms() -> None:
    module = _load_module()
    payload = module.build_artifact()
    ledh_rows = [row for row in payload["rows"] if row["algorithm_id"] == "ledh_pfpf_ot"]

    assert len(ledh_rows) == 7
    assert all(row["reason"] for row in ledh_rows)
    assert all(row["score_status_reason"] for row in ledh_rows)
    assert all(row["runtime_rankable"] is False for row in ledh_rows)
    assert all(row["phase7_batch_gpu_xla_status"]["timing_rank_status"] == "not_rankable_phase2_schema_only" for row in ledh_rows)


def test_ledh_dry_run_initial_executable_arms_are_value_only() -> None:
    module = _load_module()
    payload = module.build_artifact()

    assert payload["manifest"]["initial_executable_arms"] == [
        "benchmark_lgssm_exact_oracle_m3_T50:ledh_value_dry_run_or_tiny_value_gate_only",
        "zhao_cui_spatial_sir_austria_j9_T20:fixed_spatial_sir_value_arm_only",
    ]
    assert "all_ledh_score_arms" in payload["manifest"]["initially_blocked_arms"]

    by_row = {
        row["row_id"]: row
        for row in payload["rows"]
        if row["algorithm_id"] == "ledh_pfpf_ot"
    }
    assert by_row["benchmark_lgssm_exact_oracle_m3_T50"]["comparison_status"] == "dry_run_value_candidate"
    assert by_row["benchmark_lgssm_exact_oracle_m3_T50"]["score_status"].startswith("blocked_score")
    assert by_row["zhao_cui_spatial_sir_austria_j9_T20"]["comparison_status"] in {
        "dry_run_value_candidate",
        "blocked",
    }
    assert by_row["zhao_cui_spatial_sir_austria_j9_T20"]["row_admission_status"] == (
        "amended_sir_log_scale_theta_full_row_candidate"
    )
    assert by_row["zhao_cui_spatial_sir_austria_j9_T20"]["score_status"].startswith("blocked_score")


def test_ledh_dry_run_parameterized_sir_is_scoped_not_full_row() -> None:
    module = _load_module()
    payload = module.build_artifact()
    row = next(
        item
        for item in payload["rows"]
        if item["algorithm_id"] == "ledh_pfpf_ot"
        and item["row_id"] == "zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale"
    )

    assert row["comparison_status"] == "scoped_component_status_only"
    assert row["row_scope"] == "scoped_component_row"
    assert "not_full_observed_data_filtering_row" in row["target_match_status"]
    assert any("full observed-data" in item for item in row["forbidden_claims"])


def test_ledh_dry_run_frozen_baseline_rows_are_not_runtime_rankable_with_ledh() -> None:
    module = _load_module()
    payload = module.build_artifact()
    non_ledh_rows = [row for row in payload["rows"] if row["algorithm_id"] != "ledh_pfpf_ot"]

    assert non_ledh_rows
    assert all(row["comparator_provenance"] == "frozen_non_ledh_baseline" for row in non_ledh_rows)
    assert all(row["runtime_rankable_with_ledh"] is False for row in non_ledh_rows)


def test_same_target_lgssm_runner_preserves_leaderboard_identity_without_cpu_hiding_import() -> None:
    source = SAME_TARGET_LGSSM_SCRIPT.read_text(encoding="utf-8")

    assert 'ROW_ID = "benchmark_lgssm_exact_oracle_m3_T50"' in source
    assert "DATASET_SEED = 81100" in source
    assert "FULL_ROW_TIME_STEPS = 50" in source
    assert "STATE_DIM = 3" in source
    assert "OBS_DIM = 3" in source
    assert "_lgssm_dataset" in source
    assert "_lgssm_benchmark_model" in source
    assert "import benchmark_two_lane_highdim_leaderboard" not in source
    assert "from docs.benchmarks import benchmark_two_lane_highdim_leaderboard" not in source
    assert "admitted_same_target_compact_score" in source
    assert "compact_forward_sensitivity_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot" in source
    assert "runtime_rankable_with_frozen_non_ledh" in source


def test_ledh_inclusive_results_merge_blocks_legacy_score_memory_candidates() -> None:
    module = _load_results_module()
    payload = module.build_artifact()

    assert payload["comparator_mode"] == "frozen_non_ledh_baseline_plus_fresh_ledh"
    assert payload["runtime_cross_ranking_allowed"] is False
    assert len(payload["row_summary"]) == 7

    by_key = {(row["row_id"], row["algorithm_id"]): row for row in payload["rows"]}
    lgssm = by_key[("benchmark_lgssm_exact_oracle_m3_T50", "ledh_pfpf_ot")]
    sir = by_key[("zhao_cui_spatial_sir_austria_j9_T20", "ledh_pfpf_ot")]

    assert payload["metadata_date"] == "2026-07-06"
    assert payload["manifest"]["score_admission_policy"] == (
        "phase1_validated_compact_score_artifact_required_for_admission"
    )

    assert lgssm["comparison_status"] == "executed_value_only_score_blocked"
    assert lgssm["target_match_status"] == "same_target_value_only"
    assert lgssm["score"] is None
    assert lgssm["score_l2_norm"] is None
    assert lgssm["score_status"] == "blocked_score_until_same_target_no_tape_gate"
    assert lgssm["score_derivative_provenance"] is None
    assert lgssm["score_candidate_derivative_provenance"] == (
        "compact_forward_sensitivity_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot"
    )
    assert lgssm["score_candidate_admission_status"] == "legacy_raw_score_memory_not_admitted"
    assert lgssm["score_candidate_artifact"].endswith(
        "ledh-phase5-lgssm-score-memory-n10000-2026-07-06.json"
    )
    assert "lacks the Phase 1 score artifact schema" in lgssm["score_status_reason"]
    assert lgssm["runtime_rankable"] is False
    assert abs(lgssm["average_log_likelihood"] + 2.719201477050781) < 1e-12

    assert sir["comparison_status"] == "executed_value_only_score_blocked"
    assert sir["score"] is None
    assert sir["score_l2_norm"] is None
    assert sir["score_status"] == "blocked_score_until_same_target_no_tape_gate"
    assert sir["score_derivative_provenance"] is None
    assert sir["score_candidate_derivative_provenance"] == (
        "manual_total_vjp_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot"
    )
    assert sir["score_candidate_admission_status"] == "historical_diagnostic_not_admitted"
    assert sir["score_candidate_artifact"].endswith(
        "ledh-phase5-fixed-sir-score-memory-n10000-2026-07-06.json"
    )
    assert sir["score_evidence_artifact"] is None
    assert sir["score_status_reason"].startswith("historical manual_total_vjp")
    assert sir["target_match_status"] == "sir_log_scale_theta_observed_data_value_score"
    assert sir["runtime_rankable"] is False
    assert sir["mc_standard_error"] is not None


def test_ledh_inclusive_results_frozen_non_ledh_rows_are_labeled() -> None:
    module = _load_results_module()
    payload = module.build_artifact()
    non_ledh_rows = [row for row in payload["rows"] if row["algorithm_id"] != "ledh_pfpf_ot"]

    assert non_ledh_rows
    assert all(row["comparator_provenance"] == "frozen_non_ledh_baseline" for row in non_ledh_rows)
    assert all(row["runtime_rankable_with_ledh"] is False for row in non_ledh_rows)
    assert all("source_baseline" in row for row in non_ledh_rows)


def test_ledh_inclusive_results_preserve_blocked_and_scoped_ledh_rows() -> None:
    module = _load_results_module()
    payload = module.build_artifact()
    ledh_by_row = {
        row["row_id"]: row
        for row in payload["rows"]
        if row["algorithm_id"] == "ledh_pfpf_ot"
    }

    value_only_blocked_score = {
        "benchmark_lgssm_exact_oracle_m3_T50",
        "zhao_cui_spatial_sir_austria_j9_T20",
    }
    blocked = {
        "zhao_cui_sv_actual_nongaussian_T1000",
        "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
        "zhao_cui_predator_prey_T20",
        "zhao_cui_generalized_sv_synthetic_from_estimated_values",
    }

    for row_id in value_only_blocked_score:
        assert ledh_by_row[row_id]["comparison_status"] == "executed_value_only_score_blocked"
        assert ledh_by_row[row_id]["score_status"] == "blocked_score_until_same_target_no_tape_gate"
        assert ledh_by_row[row_id]["score"] is None
        assert ledh_by_row[row_id]["score_derivative_provenance"] is None

    for row_id in blocked:
        assert ledh_by_row[row_id]["comparison_status"] == "blocked"
        assert ledh_by_row[row_id]["score_status"] == "blocked_score"
        assert ledh_by_row[row_id]["score"] is None

    scoped = ledh_by_row["zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale"]
    assert scoped["comparison_status"] == "scoped_component_status_only"
    assert scoped["row_scope"] == "scoped_component_row"
    assert scoped["score_status"] == "scoped_score_diagnostic_not_full_observed_data_score"
