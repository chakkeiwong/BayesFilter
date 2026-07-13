from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bayesfilter.highdim.ledh_forward_contract import LGSSM_M3_T50_ROW_ID
from bayesfilter.highdim.ledh_score_artifact import build_ledh_score_artifact
from bayesfilter.highdim.ledh_score_contract import (
    LEDH_SCORE_ADMISSION_STATUS_FULL,
    LEDH_SCORE_ADMISSION_STATUS_TINY,
    LEDH_SCORE_COMPACT_LGSSM_PROVENANCE,
    LEDH_SCORE_MEMORY_STYLE_LGSSM_PROVENANCE,
    validate_ledh_score_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
LGSSM_VALUE_PATH = ROOT / "docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json"


def _load_value() -> dict:
    return json.loads(LGSSM_VALUE_PATH.read_text(encoding="utf-8"))


def _correctness() -> dict:
    return {
        "kind": "same_scalar_finite_difference",
        "status": "pass",
        "max_abs_error": 1.0e-8,
        "max_relative_error": 1.0e-8,
    }


def _full_artifact(**overrides) -> dict:
    kwargs = {
        "source_value_artifact": _load_value(),
        "source_value_artifact_path": "docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json",
        "expected_row_id": LGSSM_M3_T50_ROW_ID,
        "score_parameter_names": ["phi1", "phi2", "phi3", "q_scale", "r_scale"],
        "score": [1.0, -2.0, 0.5, 3.0, 4.0],
        "score_derivative_provenance": LEDH_SCORE_COMPACT_LGSSM_PROVENANCE,
        "score_correctness": _correctness(),
        "score_admission_status": LEDH_SCORE_ADMISSION_STATUS_FULL,
        "memory_diagnostics": {
            "n10000_memory_pass": True,
            "source": "score_gpu_memory_info_after",
            "peak_mib": 512.0,
            "budget_mib": 14000.0,
        },
        "score_precision": {
            "dtype": "float32",
            "active_dtype": "float32",
            "tf_dtype": "float32",
            "tf32_mode": "enabled",
            "tf32_execution_enabled": True,
        },
    }
    kwargs.update(overrides)
    return build_ledh_score_artifact(**kwargs)


def test_phase1_shared_emitter_builds_full_compact_admitted_artifact() -> None:
    artifact = _full_artifact()
    normalized = validate_ledh_score_artifact(
        artifact,
        source_value_artifact=_load_value(),
        expected_row_id=LGSSM_M3_T50_ROW_ID,
        require_admitted=True,
    )

    assert normalized["admitted"] is True
    assert artifact["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_FULL
    assert artifact["target_scalar"] == "observed_data_log_likelihood_estimator"


def test_phase1_shared_emitter_rejects_missing_n10000_memory_gate() -> None:
    with pytest.raises(ValueError, match="n10000_memory_pass"):
        _full_artifact(memory_diagnostics={})


def test_phase1_shared_emitter_rejects_historical_route_full_admission() -> None:
    with pytest.raises(ValueError, match="historical memory_style/manual_total_vjp"):
        _full_artifact(
            score_derivative_provenance=LEDH_SCORE_MEMORY_STYLE_LGSSM_PROVENANCE
        )


def test_phase1_shared_emitter_rejects_parameter_order_mismatch() -> None:
    with pytest.raises(ValueError, match="parameter"):
        _full_artifact(score_parameter_names=["phi1", "phi2", "q_scale", "r_scale", "phi3"])


def test_phase1_shared_emitter_rejects_row_mismatch() -> None:
    with pytest.raises(ValueError, match="expected row"):
        _full_artifact(expected_row_id="wrong_row")


def test_phase1_shared_emitter_rejects_raw_legacy_payload_without_schema() -> None:
    raw = {
        "row_id": LGSSM_M3_T50_ROW_ID,
        "score_route": "compact_forward_sensitivity_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot",
        "primary_pass": True,
    }
    with pytest.raises(ValueError, match="schema_version"):
        validate_ledh_score_artifact(
            raw,
            source_value_artifact=_load_value(),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_phase1_shared_emitter_keeps_tiny_artifact_non_admitted() -> None:
    artifact = _full_artifact(
        score_admission_status=LEDH_SCORE_ADMISSION_STATUS_TINY,
        memory_diagnostics={},
        require_admitted=False,
    )

    assert artifact["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_TINY
    with pytest.raises(ValueError, match="not admitted"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load_value(),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_phase1_shared_emitter_rejects_source_target_tampering() -> None:
    source = copy.deepcopy(_load_value())
    source["target_scalar"] = "proposal_log_likelihood"

    with pytest.raises(ValueError, match="target_scalar"):
        _full_artifact(source_value_artifact=source)
