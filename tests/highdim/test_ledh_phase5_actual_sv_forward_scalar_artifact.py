from __future__ import annotations

import json
import math
from pathlib import Path

from bayesfilter.highdim.ledh_forward_contract import (
    ACTUAL_SV_ROW_ID,
    LEDH_FORWARD_ADMISSION_STATUS_ADMITTED,
    LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
    validate_ledh_forward_scalar_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    ROOT / "docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json"
)


def test_phase5_actual_sv_full_forward_scalar_artifact_replays_admission_gate() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    normalized = validate_ledh_forward_scalar_artifact(
        artifact,
        expected_row_id=ACTUAL_SV_ROW_ID,
        require_admitted=True,
    )

    assert normalized["row_id"] == ACTUAL_SV_ROW_ID
    assert normalized["target_scalar"] == LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD
    assert normalized["target_output_tensor_field"] == "log_likelihood"
    assert normalized["admission_status"] == LEDH_FORWARD_ADMISSION_STATUS_ADMITTED
    assert normalized["num_particles"] == 10000
    assert normalized["time_steps"] == 1000
    assert normalized["batch_seeds"] == [81120, 81121, 81122, 81123, 81124]
    assert normalized["theta_coordinate_system"] == "synthetic_unconstrained"
    assert normalized["theta_values"] == [0.2533471031357997, -0.916290731874155]
    assert normalized["target_observation_policy"] == "transformed_actual_sv_log_y_square"
    assert normalized["flow_observation_policy"] != normalized["target_observation_policy"]
    assert normalized["target_density_used_for_correction"] is True
    assert len(normalized["log_likelihood_by_seed"]) == 5
    assert all(math.isfinite(value) for value in normalized["log_likelihood_by_seed"])

    semantics = artifact["actual_sv_semantics"]
    assert artifact["run_scope"] == "full-row-admission"
    assert semantics["row_id"] == ACTUAL_SV_ROW_ID
    assert semantics["target_transform"] == "exact_log_y_square"
    assert semantics["transform_offset"] == 0.0
    assert semantics["target_observation_density"] == "exact_log_chi_square_log_density"
    assert semantics["legacy_raw_gaussian_callback_used"] is False
    assert semantics["ksc_mixture_used"] is False
    assert semantics["augmented_noise_gaussian_closure_used"] is False
    assert artifact["transport"]["plan_mode"] == "streaming"
    assert artifact["transport"]["dense_transport_matrix_materialized"] is False
    assert artifact["normalization_checks"]["exact_full_row_requested"] is True
    assert artifact["normalization_checks"]["full_row_admission_scope"] is True
    assert artifact["output_devices"]
    assert all("GPU" in device.upper() for device in artifact["output_devices"])


def test_phase5_actual_sv_full_artifact_nonclaims_exclude_score_and_science_claims() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    nonclaims = set(artifact["nonclaims"])
    validator_nonclaims = set(artifact["validator_normalized_core"]["nonclaims"])

    assert validator_nonclaims == nonclaims
    assert "not full actual-SV row admission" not in nonclaims
    assert "not full actual-SV row admission" not in validator_nonclaims
    assert "not score admission" in nonclaims
    assert "not score correctness" in nonclaims
    assert "not KSC surrogate likelihood evidence" in nonclaims
    assert "not raw Gaussian observation likelihood evidence" in nonclaims
    assert "not augmented-noise Gaussian-closure evidence" in nonclaims
    assert "not generalized-SV admission" in nonclaims
    assert "not HMC readiness evidence" in nonclaims
    assert "not posterior correctness evidence" in nonclaims
    assert "not scientific superiority evidence" in nonclaims
    assert "not runtime ranking evidence" in nonclaims
