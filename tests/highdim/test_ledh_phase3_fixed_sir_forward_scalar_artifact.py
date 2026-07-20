from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from bayesfilter.highdim.ledh_forward_contract import (
    FIXED_SIR_AUSTRIA_ROW_ID,
    LEDH_FORWARD_ADMISSION_STATUS_HISTORICAL_RAW,
    LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
    validate_ledh_forward_scalar_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json"


def test_phase3_fixed_sir_forward_scalar_artifact_replays_as_historical() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    normalized = validate_ledh_forward_scalar_artifact(
        artifact,
        expected_row_id=FIXED_SIR_AUSTRIA_ROW_ID,
        require_admitted=False,
    )

    assert normalized["row_id"] == FIXED_SIR_AUSTRIA_ROW_ID
    assert (
        LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD
        == "observed_data_log_likelihood_estimator"
    )
    assert normalized["target_scalar"] == LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD
    assert normalized["target_output_tensor_field"] == "log_likelihood"
    assert normalized["admission_status"] == LEDH_FORWARD_ADMISSION_STATUS_HISTORICAL_RAW
    assert normalized["num_particles"] == 10000
    assert normalized["time_steps"] == 20
    assert normalized["batch_seeds"] == [81120, 81121, 81122, 81123, 81124]
    assert normalized["theta_coordinate_system"] == "sir_log_scale_theta"
    assert normalized["theta_values"] == [0.0, 0.0, 0.0]
    assert normalized["target_density_used_for_correction"] is True
    assert all(math.isfinite(value) for value in normalized["log_likelihood_by_seed"])
    assert artifact["sir_semantics"]["target_density_used_for_correction"] is True
    assert artifact["sir_semantics"]["row_id"] == FIXED_SIR_AUSTRIA_ROW_ID
    with pytest.raises(ValueError, match="cannot be canonically admitted"):
        validate_ledh_forward_scalar_artifact(
            artifact,
            expected_row_id=FIXED_SIR_AUSTRIA_ROW_ID,
            require_admitted=True,
        )


def test_phase3_fixed_sir_artifact_is_not_old_no_free_theta_or_score_evidence() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    nonclaims = set(artifact["nonclaims"])

    assert artifact["theta_coordinate_system"] == "sir_log_scale_theta"
    assert "not old no_free_theta admission" in nonclaims
    assert "not score admission" in nonclaims
    assert "not score correctness" in nonclaims
    assert "not exact nonlinear likelihood correctness evidence" in nonclaims
    assert "not Zhao-Cui TT/SIRT source-faithfulness evidence" in nonclaims
    assert "not HMC readiness evidence" in nonclaims
