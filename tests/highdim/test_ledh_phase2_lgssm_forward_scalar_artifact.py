from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from bayesfilter.highdim.ledh_forward_contract import (
    LEDH_FORWARD_ADMISSION_STATUS_HISTORICAL_RAW,
    LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
    LGSSM_M3_T50_ROW_ID,
    validate_ledh_forward_scalar_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json"


def test_phase2_lgssm_forward_scalar_artifact_replays_as_historical() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    normalized = validate_ledh_forward_scalar_artifact(
        artifact,
        expected_row_id=LGSSM_M3_T50_ROW_ID,
        require_admitted=False,
    )

    assert normalized["row_id"] == LGSSM_M3_T50_ROW_ID
    assert (
        LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD
        == "observed_data_log_likelihood_estimator"
    )
    assert normalized["target_scalar"] == LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD
    assert normalized["target_output_tensor_field"] == "log_likelihood"
    assert normalized["admission_status"] == LEDH_FORWARD_ADMISSION_STATUS_HISTORICAL_RAW
    assert normalized["num_particles"] == 10000
    assert normalized["time_steps"] == 50
    assert normalized["batch_seeds"] == [81120, 81121, 81122, 81123, 81124]
    assert normalized["theta_values"] == [0.72, 0.55, 0.35, 0.35, 0.45]
    assert all(math.isfinite(value) for value in normalized["log_likelihood_by_seed"])
    assert artifact["exact_value_comparator"] == (
        "tf_kalman_log_likelihood on same observations/model"
    )
    assert artifact["normalization_checks"] == {
        "batch_seeds": True,
        "exact_comparator": True,
        "finite_output": True,
        "finite_total_log_likelihood_by_seed": True,
        "num_particles": True,
        "primary_pass_same_target_value_execution": True,
        "row_id": True,
        "seed_count_matches_values": True,
        "time_steps": True,
        "truth_theta": True,
    }
    with pytest.raises(ValueError, match="cannot be canonically admitted"):
        validate_ledh_forward_scalar_artifact(
            artifact,
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_phase2_lgssm_artifact_is_not_score_or_nonlinear_evidence() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    nonclaims = set(artifact["nonclaims"])

    assert "not nonlinear-row evidence" in nonclaims
    assert "not score admission" in nonclaims
    assert "not score correctness" in nonclaims
    assert "not HMC readiness evidence" in nonclaims
