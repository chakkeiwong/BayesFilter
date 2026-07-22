from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bayesfilter.highdim.ledh_forward_contract import (
    LEDH_FORWARD_ADMISSION_STATUS_HISTORICAL_RAW,
    LGSSM_M3_T50_ROW_ID,
    validate_ledh_forward_scalar_artifact,
)
from bayesfilter.highdim.ledh_score_artifact import build_ledh_score_artifact
from bayesfilter.highdim.ledh_score_contract import (
    LEDH_SCORE_ADMISSION_STATUS_FULL,
    LEDH_SCORE_ADMISSION_STATUS_HISTORICAL_RAW,
    LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
    LEDH_SCORE_COMPACT_LGSSM_PROVENANCE,
    LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
    validate_ledh_score_artifact,
)
from docs.benchmarks.benchmark_two_lane_highdim_ledh_inclusive_results import (
    _score_payload,
)


ROOT = Path(__file__).resolve().parents[2]
VALUE_PATH = ROOT / "docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json"


def _value() -> dict:
    return json.loads(VALUE_PATH.read_text(encoding="utf-8"))


def _score() -> dict:
    return {
        "schema_version": LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
        "row_id": LGSSM_M3_T50_ROW_ID,
        "source_value_artifact": str(VALUE_PATH.relative_to(ROOT)),
        "score_target_kind": LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
        "target_scalar": "observed_data_log_likelihood_estimator",
        "target_output_tensor_field": "log_likelihood",
        "target_observation_policy": "lgssm_gaussian_observation_density",
        "theta_coordinate_system": "physical_benchmark_exact_oracle",
        "score_parameter_names": ["phi1", "phi2", "phi3", "q_scale", "r_scale"],
        "score": [1.0, -2.0, 0.5, 3.0, 4.0],
        "score_derivative_provenance": LEDH_SCORE_COMPACT_LGSSM_PROVENANCE,
        "value_score_route_status": "same_route_value_score",
        "value_score_same_transport_algorithm": True,
        "no_autodiff_score_route": True,
        "uses_gradient_tape": False,
        "uses_forward_accumulator": False,
        "uses_stopped_partial_derivative": False,
        "score_correctness": {
            "kind": "same_scalar_finite_difference",
            "status": "pass",
            "max_abs_error": 1.0e-8,
        },
        "score_admission_status": LEDH_SCORE_ADMISSION_STATUS_FULL,
        "score_precision": {
            "dtype": "float32",
            "active_dtype": "float32",
            "tf_dtype": "float32",
            "tf32_mode": "enabled",
            "tf32_execution_enabled": True,
        },
        "memory_diagnostics": {
            "n10000_memory_pass": True,
            "source": "score_gpu_memory_info_after",
            "peak_mib": 512.0,
            "budget_mib": 14000.0,
        },
    }


def test_v1_forward_full_status_is_historical_and_never_admitted() -> None:
    normalized = validate_ledh_forward_scalar_artifact(_value())

    assert normalized["admission_status"] == LEDH_FORWARD_ADMISSION_STATUS_HISTORICAL_RAW
    assert normalized["canonical_admission_eligible"] is False
    with pytest.raises(ValueError, match="lacks factory-issued Contract E"):
        validate_ledh_forward_scalar_artifact(_value(), require_admitted=True)


def test_forged_contract_e_metadata_cannot_upgrade_v1_forward() -> None:
    forged = copy.deepcopy(_value())
    forged["reset_contract_id"] = "contract_e_chol_v1"
    forged["forward_contract"]["metadata"]["reset_contract_id"] = "contract_e_chol_v1"

    normalized = validate_ledh_forward_scalar_artifact(forged)
    assert normalized["admission_status"] == LEDH_FORWARD_ADMISSION_STATUS_HISTORICAL_RAW
    with pytest.raises(ValueError, match="lacks factory-issued Contract E"):
        validate_ledh_forward_scalar_artifact(forged, require_admitted=True)


def test_v1_score_full_status_is_historical_and_never_admitted() -> None:
    normalized = validate_ledh_score_artifact(
        _score(),
        source_value_artifact=_value(),
        expected_row_id=LGSSM_M3_T50_ROW_ID,
    )

    assert normalized["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_HISTORICAL_RAW
    assert normalized["admitted"] is False
    with pytest.raises(ValueError, match="lacks factory-issued Contract E"):
        validate_ledh_score_artifact(
            _score(),
            source_value_artifact=_value(),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_forged_contract_e_metadata_cannot_upgrade_v1_score() -> None:
    forged = _score()
    forged["reset_contract_id"] = "contract_e_chol_v1"
    forged["canonical_route_factory_id"] = "forged_by_caller"

    with pytest.raises(ValueError, match="lacks factory-issued Contract E"):
        validate_ledh_score_artifact(
            forged,
            source_value_artifact=_value(),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_v1_emitter_cannot_create_new_full_admission() -> None:
    with pytest.raises(ValueError, match="full admission is revoked"):
        build_ledh_score_artifact(
            source_value_artifact=_value(),
            source_value_artifact_path=str(VALUE_PATH.relative_to(ROOT)),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            score_parameter_names=["phi1", "phi2", "phi3", "q_scale", "r_scale"],
            score=[1.0, -2.0, 0.5, 3.0, 4.0],
            score_derivative_provenance=LEDH_SCORE_COMPACT_LGSSM_PROVENANCE,
            score_correctness={
                "kind": "same_scalar_finite_difference",
                "status": "pass",
            },
            score_admission_status=LEDH_SCORE_ADMISSION_STATUS_FULL,
            score_precision={
                "dtype": "float32",
                "active_dtype": "float32",
                "tf_dtype": "float32",
                "tf32_mode": "enabled",
                "tf32_execution_enabled": True,
            },
            memory_diagnostics={
                "n10000_memory_pass": True,
                "source": "score_gpu_memory_info_after",
                "peak_mib": 512.0,
                "budget_mib": 14000.0,
            },
        )


def test_inclusive_aggregator_fails_closed_on_v1_score() -> None:
    admitted, candidate, reason = _score_payload(
        _score(),
        expected_row_id=LGSSM_M3_T50_ROW_ID,
        source_value_artifact=_value(),
    )

    assert admitted is None
    assert candidate["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_HISTORICAL_RAW
    assert "not full admission" in reason
