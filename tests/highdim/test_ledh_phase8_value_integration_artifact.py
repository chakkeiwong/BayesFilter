from __future__ import annotations

import json
import math
from pathlib import Path

from bayesfilter.highdim.ledh_forward_contract import (
    ACTUAL_SV_ROW_ID,
    FIXED_SIR_AUSTRIA_ROW_ID,
    GENERALIZED_SV_ROW_ID,
    KSC_SV_ROW_ID,
    LEDH_FORWARD_ADMISSION_STATUS_ADMITTED,
    LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
    LGSSM_M3_T50_ROW_ID,
    PARAMETERIZED_SIR_DIAGNOSTIC_ROW_ID,
    PREDATOR_PREY_ROW_ID,
    validate_ledh_forward_scalar_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    ROOT / "docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json"
)

EXPECTED_ROW_IDS = [
    LGSSM_M3_T50_ROW_ID,
    FIXED_SIR_AUSTRIA_ROW_ID,
    PREDATOR_PREY_ROW_ID,
    ACTUAL_SV_ROW_ID,
    GENERALIZED_SV_ROW_ID,
    KSC_SV_ROW_ID,
]
EXPECTED_SOURCE_ARTIFACTS = {
    LGSSM_M3_T50_ROW_ID: "docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json",
    FIXED_SIR_AUSTRIA_ROW_ID: (
        "docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json"
    ),
    PREDATOR_PREY_ROW_ID: (
        "docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json"
    ),
    ACTUAL_SV_ROW_ID: "docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json",
    GENERALIZED_SV_ROW_ID: (
        "docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json"
    ),
    KSC_SV_ROW_ID: "docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json",
}
FORBIDDEN_SCORE_KEYS = {
    "score",
    "score_l2_norm",
    "score_status",
    "score_derivative_provenance",
    "score_coordinate_system",
    "score_evidence_artifact",
}


def test_phase8_value_integration_artifact_has_only_admitted_main_rows() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    assert artifact["schema_version"] == (
        "bayesfilter.highdim.ledh_forward_scalar_value_integration.v1"
    )
    assert artifact["phase"] == 8
    assert artifact["target_scalar"] == LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD
    assert artifact["target_output_tensor_field"] == "log_likelihood"
    assert artifact["score_integration_status"] == "blocked_out_of_scope_forward_scalar_only"
    assert artifact["runtime_cross_ranking_allowed"] is False
    assert artifact["all_algorithm_comparison_allowed"] is False
    assert artifact["main_row_count"] == 6
    assert artifact["main_row_ids"] == EXPECTED_ROW_IDS

    rows = artifact["rows"]
    assert [row["row_id"] for row in rows] == EXPECTED_ROW_IDS
    assert PARAMETERIZED_SIR_DIAGNOSTIC_ROW_ID not in [row["row_id"] for row in rows]

    for row in rows:
        row_id = row["row_id"]
        assert FORBIDDEN_SCORE_KEYS.isdisjoint(row)
        assert row["algorithm_id"] == "ledh_pfpf_ot"
        assert row["source_artifact"] == EXPECTED_SOURCE_ARTIFACTS[row_id]
        assert row["admission_status"] == LEDH_FORWARD_ADMISSION_STATUS_ADMITTED
        assert row["value_status"] == "admitted_forward_scalar_value_only"
        assert row["score_integration_status"] == "blocked_out_of_scope_forward_scalar_only"
        assert row["target_scalar"] == LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD
        assert row["target_output_tensor_field"] == "log_likelihood"
        assert row["target_density_used_for_correction"] is True
        assert row["runtime_rankable"] is False
        assert row["runtime_cross_ranking_allowed"] is False
        assert row["row_semantics"]["target_policy_label_preserved"] is True
        assert len(row["batch_seeds"]) == len(row["log_likelihood_by_seed"])
        assert len(row["batch_seeds"]) == len(row["average_log_likelihood_by_seed"])
        assert all(math.isfinite(value) for value in row["log_likelihood_by_seed"])
        assert all(math.isfinite(value) for value in row["average_log_likelihood_by_seed"])

        source = json.loads((ROOT / row["source_artifact"]).read_text(encoding="utf-8"))
        normalized = validate_ledh_forward_scalar_artifact(
            source,
            expected_row_id=row_id,
            require_admitted=True,
        )
        assert row["log_likelihood_by_seed"] == normalized["log_likelihood_by_seed"]
        assert row["average_log_likelihood_by_seed"] == normalized[
            "average_log_likelihood_by_seed"
        ]


def test_phase8_value_integration_preserves_ksc_surrogate_boundary() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    row = next(item for item in artifact["rows"] if item["row_id"] == KSC_SV_ROW_ID)
    semantics = row["row_semantics"]

    assert row["target_observation_policy"] == "ksc_log_chi_square_gaussian_mixture_surrogate"
    assert semantics["target_family"] == "ksc_finite_gaussian_mixture_surrogate"
    assert semantics["target_observation_density"] == (
        "finite_ksc_log_chi_square_gaussian_mixture_log_density"
    )
    assert semantics["target_transform"] == "log_y_square_plus_offset"
    assert semantics["transform_offset"] == 1.0e-8
    assert semantics["ksc_mixture_is_target_likelihood"] is True
    assert semantics["actual_sv_exact_log_chi_square_target_used"] is False
    assert semantics["generalized_sv_target_used"] is False
    assert semantics["legacy_raw_gaussian_callback_used"] is False
    assert "KSC row is finite-mixture surrogate target evidence, not exact native actual-SV likelihood" in artifact[
        "nonclaims"
    ]


def test_phase8_value_integration_records_diagnostic_sir_exclusion() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    diagnostics = artifact["diagnostic_rows"]

    assert diagnostics == [
        {
            "row_id": PARAMETERIZED_SIR_DIAGNOSTIC_ROW_ID,
            "status": "excluded_from_main_value_leaderboard",
            "reason": (
                "legacy scoped parameterized SIR diagnostic row; no separate "
                "Phase 8 main-row admission artifact"
            ),
        }
    ]
