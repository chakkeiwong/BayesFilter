from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from bayesfilter.highdim.ledh_forward_contract import (
    LEDH_FORWARD_ADMISSION_STATUS_HISTORICAL_RAW,
    LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
    PREDATOR_PREY_ROW_ID,
    validate_ledh_forward_scalar_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    ROOT
    / "docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json"
)


def test_phase4_predator_prey_forward_scalar_artifact_replays_as_historical() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    normalized = validate_ledh_forward_scalar_artifact(
        artifact,
        expected_row_id=PREDATOR_PREY_ROW_ID,
        require_admitted=False,
    )

    assert normalized["row_id"] == PREDATOR_PREY_ROW_ID
    assert normalized["target_scalar"] == LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD
    assert normalized["target_output_tensor_field"] == "log_likelihood"
    assert normalized["admission_status"] == LEDH_FORWARD_ADMISSION_STATUS_HISTORICAL_RAW
    assert normalized["num_particles"] == 10000
    assert normalized["time_steps"] == 20
    assert normalized["batch_seeds"] == [81120, 81121, 81122, 81123, 81124]
    assert normalized["theta_coordinate_system"] == "physical"
    assert normalized["theta_values"] == [0.6, 114.0, 25.0, 0.3, 0.5, 0.5]
    assert normalized["target_observation_policy"] == "additive_gaussian_predator_prey"
    assert normalized["target_density_used_for_correction"] is True
    assert all(math.isfinite(value) for value in normalized["log_likelihood_by_seed"])

    semantics = artifact["predator_prey_semantics"]
    assert semantics["row_id"] == PREDATOR_PREY_ROW_ID
    assert semantics["target_density_used_for_correction"] is True
    assert semantics["flow_observation_contract"] == "identity_state_gaussian_flow_observation"
    assert artifact["transport"]["plan_mode"] == "streaming"
    assert artifact["transport"]["dense_transport_matrix_materialized"] is False
    assert artifact["output_devices"]
    assert all("GPU" in device.upper() for device in artifact["output_devices"])
    with pytest.raises(ValueError, match="cannot be canonically admitted"):
        validate_ledh_forward_scalar_artifact(
            artifact,
            expected_row_id=PREDATOR_PREY_ROW_ID,
            require_admitted=True,
        )


def test_phase4_predator_prey_artifact_is_not_score_or_exactness_evidence() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    nonclaims = set(artifact["nonclaims"])

    assert "not score admission" in nonclaims
    assert "not score correctness" in nonclaims
    assert "not exact nonlinear likelihood correctness evidence" in nonclaims
    assert "not Zhao-Cui TT/SIRT source-faithfulness evidence" in nonclaims
    assert "not HMC readiness evidence" in nonclaims
    assert "not posterior correctness evidence" in nonclaims
    assert "not scientific superiority evidence" in nonclaims
    assert "not runtime ranking evidence" in nonclaims
