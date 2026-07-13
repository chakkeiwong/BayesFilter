from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from bayesfilter.highdim.ledh_forward_contract import (
    ACTUAL_SV_ROW_ID,
    LEDH_FORWARD_ADMISSION_STATUS_TINY,
    LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
    validate_ledh_forward_scalar_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    ROOT
    / "docs/plans/ledh-phase5-actual-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json"
)


def test_phase5_actual_sv_tiny_forward_scalar_artifact_replays_without_admission() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    normalized = validate_ledh_forward_scalar_artifact(
        artifact,
        expected_row_id=ACTUAL_SV_ROW_ID,
        require_admitted=False,
    )

    assert normalized["row_id"] == ACTUAL_SV_ROW_ID
    assert normalized["target_scalar"] == LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD
    assert normalized["target_output_tensor_field"] == "log_likelihood"
    assert normalized["admission_status"] == LEDH_FORWARD_ADMISSION_STATUS_TINY
    assert normalized["num_particles"] == 128
    assert normalized["time_steps"] == 4
    assert normalized["batch_seeds"] == [81120]
    assert normalized["theta_coordinate_system"] == "synthetic_unconstrained"
    assert normalized["theta_values"] == [0.2533471031357997, -0.916290731874155]
    assert normalized["target_observation_policy"] == "transformed_actual_sv_log_y_square"
    assert normalized["flow_observation_policy"] != normalized["target_observation_policy"]
    assert normalized["target_density_used_for_correction"] is True
    assert all(math.isfinite(value) for value in normalized["log_likelihood_by_seed"])

    semantics = artifact["actual_sv_semantics"]
    assert semantics["row_id"] == ACTUAL_SV_ROW_ID
    assert semantics["target_transform"] == "exact_log_y_square"
    assert semantics["transform_offset"] == 0.0
    assert semantics["target_observation_density"] == "exact_log_chi_square_log_density"
    assert semantics["legacy_raw_gaussian_callback_used"] is False
    assert semantics["ksc_mixture_used"] is False
    assert semantics["augmented_noise_gaussian_closure_used"] is False
    assert artifact["transport"]["plan_mode"] == "streaming"
    assert artifact["transport"]["dense_transport_matrix_materialized"] is False
    assert artifact["normalization_checks"]["not_full_row_requested"] is True


def test_phase5_actual_sv_tiny_artifact_is_rejected_as_admission() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="not admitted"):
        validate_ledh_forward_scalar_artifact(
            artifact,
            expected_row_id=ACTUAL_SV_ROW_ID,
            require_admitted=True,
        )


def test_phase5_actual_sv_tiny_artifact_nonclaims_exclude_score_and_science_claims() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    nonclaims = set(artifact["nonclaims"])

    assert "not full actual-SV row admission" in nonclaims
    assert "not score admission" in nonclaims
    assert "not score correctness" in nonclaims
    assert "not KSC surrogate likelihood evidence" in nonclaims
    assert "not raw Gaussian observation likelihood evidence" in nonclaims
    assert "not augmented-noise Gaussian-closure evidence" in nonclaims
    assert "not HMC readiness evidence" in nonclaims
    assert "not posterior correctness evidence" in nonclaims
    assert "not scientific superiority evidence" in nonclaims
    assert "not runtime ranking evidence" in nonclaims
