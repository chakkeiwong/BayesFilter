from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bayesfilter.highdim.ledh_forward_contract import (
    KSC_SV_ROW_ID,
    LGSSM_M3_T50_ROW_ID,
    PARAMETERIZED_SIR_DIAGNOSTIC_ROW_ID,
)
from bayesfilter.highdim.ledh_score_contract import (
    LEDH_SCORE_ADMISSION_STATUS_FULL,
    LEDH_SCORE_ADMISSION_STATUS_TINY,
    LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
    LEDH_SCORE_COMPACT_ACTUAL_SV_PROVENANCE,
    LEDH_SCORE_COMPACT_FIXED_SIR_PROVENANCE,
    LEDH_SCORE_COMPACT_GENERALIZED_SV_PROVENANCE,
    LEDH_SCORE_COMPACT_KSC_SV_PROVENANCE,
    LEDH_SCORE_COMPACT_LGSSM_PROVENANCE,
    LEDH_SCORE_COMPACT_PREDATOR_PREY_PROVENANCE,
    LEDH_SCORE_MEMORY_STYLE_ACTUAL_SV_PROVENANCE,
    LEDH_SCORE_MEMORY_STYLE_FIXED_SIR_PROVENANCE,
    LEDH_SCORE_MEMORY_STYLE_LGSSM_PROVENANCE,
    LEDH_SCORE_MEMORY_STYLE_PREDATOR_PREY_PROVENANCE,
    LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
    validate_ledh_score_production_precision,
    validate_ledh_score_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
LGSSM_VALUE_PATH = ROOT / "docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json"
KSC_VALUE_PATH = ROOT / "docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _lgssm_score_fixture(*, admitted: bool = False) -> dict:
    return {
        "schema_version": LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
        "row_id": LGSSM_M3_T50_ROW_ID,
        "source_value_artifact": "docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json",
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
        "score_admission_status": (
            LEDH_SCORE_ADMISSION_STATUS_FULL
            if admitted
            else LEDH_SCORE_ADMISSION_STATUS_TINY
        ),
        "score_precision": (
            {
                "dtype": "float32",
                "active_dtype": "float32",
                "tf_dtype": "float32",
                "tf32_mode": "enabled",
                "tf32_execution_enabled": True,
            }
            if admitted
            else None
        ),
        "memory_diagnostics": (
            {
                "n10000_memory_pass": True,
                "source": "score_gpu_memory_info_after",
                "peak_mib": 512.0,
                "budget_mib": 14000.0,
            }
            if admitted
            else {}
        ),
    }


def test_phase1_score_contract_validates_production_precision_metadata() -> None:
    normalized = validate_ledh_score_production_precision(
        {
            "dtype": "float32",
            "active_dtype": "float32",
            "tf_dtype": "float32",
            "tf32_mode": "enabled",
            "tf32_execution_enabled": True,
        }
    )

    assert normalized["dtype"] == "float32"
    assert normalized["tf32_execution_enabled"] is True


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("dtype", "float64", "dtype"),
        ("active_dtype", "float64", "active_dtype"),
        ("tf_dtype", "float64", "tf_dtype"),
        ("tf32_mode", "disabled", "tf32_mode"),
        ("tf32_execution_enabled", False, "tf32_execution_enabled"),
    ],
)
def test_phase1_score_contract_rejects_nonproduction_precision_metadata(
    field: str,
    value,
    match: str,
) -> None:
    precision = {
        "dtype": "float32",
        "active_dtype": "float32",
        "tf_dtype": "float32",
        "tf32_mode": "enabled",
        "tf32_execution_enabled": True,
    }
    precision[field] = value

    with pytest.raises(ValueError, match=match):
        validate_ledh_score_production_precision(precision)


@pytest.mark.parametrize("field", ["active_dtype", "tf_dtype"])
def test_phase1_score_contract_rejects_missing_explicit_precision_fields(
    field: str,
) -> None:
    precision = {
        "dtype": "float32",
        "active_dtype": "float32",
        "tf_dtype": "float32",
        "tf32_mode": "enabled",
        "tf32_execution_enabled": True,
    }
    precision.pop(field)

    with pytest.raises(ValueError, match=field):
        validate_ledh_score_production_precision(precision)


def test_phase1_score_contract_accepts_tiny_not_admitted_lgssm_fixture() -> None:
    normalized = validate_ledh_score_artifact(
        _lgssm_score_fixture(admitted=False),
        source_value_artifact=_load(LGSSM_VALUE_PATH),
        expected_row_id=LGSSM_M3_T50_ROW_ID,
        require_admitted=False,
    )

    assert normalized["row_id"] == LGSSM_M3_T50_ROW_ID
    assert normalized["admitted"] is False
    assert normalized["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_TINY


def test_phase1_score_contract_accepts_full_lgssm_fixture_with_memory_gate() -> None:
    normalized = validate_ledh_score_artifact(
        _lgssm_score_fixture(admitted=True),
        source_value_artifact=_load(LGSSM_VALUE_PATH),
        expected_row_id=LGSSM_M3_T50_ROW_ID,
        require_admitted=True,
    )

    assert normalized["admitted"] is True
    assert normalized["memory_diagnostics"]["n10000_memory_pass"] is True


def test_phase1_score_contract_accepts_row_matched_compact_route_for_full_admission() -> None:
    artifact = _lgssm_score_fixture(admitted=True)

    normalized = validate_ledh_score_artifact(
        artifact,
        source_value_artifact=_load(LGSSM_VALUE_PATH),
        expected_row_id=LGSSM_M3_T50_ROW_ID,
        require_admitted=True,
    )

    assert normalized["score_derivative_provenance"] == LEDH_SCORE_COMPACT_LGSSM_PROVENANCE


@pytest.mark.parametrize(
    "wrong_compact_route",
    [
        LEDH_SCORE_COMPACT_FIXED_SIR_PROVENANCE,
        LEDH_SCORE_COMPACT_PREDATOR_PREY_PROVENANCE,
        LEDH_SCORE_COMPACT_ACTUAL_SV_PROVENANCE,
        LEDH_SCORE_COMPACT_GENERALIZED_SV_PROVENANCE,
        LEDH_SCORE_COMPACT_KSC_SV_PROVENANCE,
    ],
)
def test_phase1_score_contract_rejects_wrong_row_compact_route_for_full_admission(
    wrong_compact_route: str,
) -> None:
    artifact = _lgssm_score_fixture(admitted=True)
    artifact["score_derivative_provenance"] = wrong_compact_route

    with pytest.raises(ValueError, match="row-matched compact provenance"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load(LGSSM_VALUE_PATH),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


@pytest.mark.parametrize(
    "old_route",
    [
        LEDH_SCORE_MEMORY_STYLE_LGSSM_PROVENANCE,
        LEDH_SCORE_MEMORY_STYLE_FIXED_SIR_PROVENANCE,
        LEDH_SCORE_MEMORY_STYLE_PREDATOR_PREY_PROVENANCE,
        LEDH_SCORE_MEMORY_STYLE_ACTUAL_SV_PROVENANCE,
        "manual_total_vjp_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot",
        "manual_total_vjp_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot",
        "manual_total_vjp_no_autodiff_same_scalar_actual_sv_ledh_pfpf_ot",
        "manual_total_vjp_no_autodiff_same_scalar_generalized_sv_ledh_pfpf_ot",
        "manual_total_vjp_no_autodiff_same_scalar_ksc_sv_ledh_pfpf_ot",
    ],
)
def test_phase1_score_contract_rejects_historical_routes_full_admission(
    old_route: str,
) -> None:
    artifact = _lgssm_score_fixture(admitted=True)
    artifact["score_derivative_provenance"] = old_route

    with pytest.raises(ValueError, match="historical memory_style/manual_total_vjp"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load(LGSSM_VALUE_PATH),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_phase1_score_contract_allows_historical_routes_only_when_not_admitted() -> None:
    artifact = _lgssm_score_fixture(admitted=False)
    artifact["score_derivative_provenance"] = LEDH_SCORE_MEMORY_STYLE_LGSSM_PROVENANCE

    normalized = validate_ledh_score_artifact(
        artifact,
        source_value_artifact=_load(LGSSM_VALUE_PATH),
        expected_row_id=LGSSM_M3_T50_ROW_ID,
        require_admitted=False,
    )

    assert normalized["admitted"] is False
    assert normalized["score_derivative_provenance"] == artifact["score_derivative_provenance"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("row_id", "wrong_row", "row_id"),
        ("target_scalar", "proposal_log_likelihood", "target_scalar"),
        ("target_output_tensor_field", "flow_log_likelihood", "output field"),
        (
            "target_observation_policy",
            "transformed_actual_sv_log_y_square",
            "target_observation_policy",
        ),
        ("theta_coordinate_system", "wrong_theta", "theta_coordinate_system"),
        ("score_parameter_names", ["phi1", "phi2", "q_scale", "r_scale", "phi3"], "parameter"),
    ],
)
def test_phase1_score_contract_rejects_value_identity_mismatches(
    field: str,
    value,
    match: str,
) -> None:
    artifact = _lgssm_score_fixture(admitted=False)
    artifact[field] = value

    with pytest.raises(ValueError, match=match):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load(LGSSM_VALUE_PATH),
            expected_row_id=LGSSM_M3_T50_ROW_ID if field != "row_id" else None,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "score_derivative_provenance",
            "GradientTape_same_scalar",
            "unsupported no-tape",
        ),
        ("uses_gradient_tape", True, "uses_gradient_tape"),
        ("uses_forward_accumulator", True, "uses_forward_accumulator"),
        ("uses_stopped_partial_derivative", True, "uses_stopped_partial_derivative"),
        ("value_score_route_status", "different_route", "same_route_value_score"),
        ("score_target_kind", "true_likelihood_score", "realized finite-N"),
    ],
)
def test_phase1_score_contract_rejects_forbidden_score_provenance(
    field: str,
    value,
    match: str,
) -> None:
    artifact = _lgssm_score_fixture(admitted=False)
    artifact[field] = value

    with pytest.raises(ValueError, match=match):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load(LGSSM_VALUE_PATH),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
        )


def test_phase1_score_contract_rejects_tiny_as_full_admission() -> None:
    with pytest.raises(ValueError, match="not admitted"):
        validate_ledh_score_artifact(
            _lgssm_score_fixture(admitted=False),
            source_value_artifact=_load(LGSSM_VALUE_PATH),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_phase1_score_contract_rejects_full_admission_without_memory_gate() -> None:
    artifact = _lgssm_score_fixture(admitted=True)
    artifact["memory_diagnostics"] = {}

    with pytest.raises(ValueError, match="n10000_memory_pass"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load(LGSSM_VALUE_PATH),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_phase1_score_contract_rejects_full_admission_without_score_precision() -> None:
    artifact = _lgssm_score_fixture(admitted=True)
    artifact.pop("score_precision")

    with pytest.raises(ValueError, match="score_precision"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load(LGSSM_VALUE_PATH),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_phase1_score_contract_rejects_full_admission_with_float64_score_precision() -> None:
    artifact = _lgssm_score_fixture(admitted=True)
    artifact["score_precision"]["dtype"] = "float64"

    with pytest.raises(ValueError, match="score_precision.dtype"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load(LGSSM_VALUE_PATH),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_phase1_score_contract_rejects_full_admission_with_tf32_disabled() -> None:
    artifact = _lgssm_score_fixture(admitted=True)
    artifact["score_precision"]["tf32_mode"] = "disabled"
    artifact["score_precision"]["tf32_execution_enabled"] = False

    with pytest.raises(ValueError, match="score_precision.tf32_mode"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load(LGSSM_VALUE_PATH),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_phase1_score_contract_rejects_full_admission_without_numeric_memory_source() -> None:
    artifact = _lgssm_score_fixture(admitted=True)
    artifact["memory_diagnostics"] = {
        "n10000_memory_pass": True,
        "peak_mib": 512.0,
        "budget_mib": 14000.0,
    }

    with pytest.raises(ValueError, match="memory_diagnostics.source"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load(LGSSM_VALUE_PATH),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_phase1_score_contract_rejects_full_admission_when_peak_exceeds_budget() -> None:
    artifact = _lgssm_score_fixture(admitted=True)
    artifact["memory_diagnostics"] = {
        "n10000_memory_pass": True,
        "source": "score_gpu_memory_info_after",
        "peak_mib": 15000.0,
        "budget_mib": 14000.0,
    }

    with pytest.raises(ValueError, match="exceeds budget"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load(LGSSM_VALUE_PATH),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_phase1_score_contract_rejects_exact_reference_for_full_admission() -> None:
    artifact = _lgssm_score_fixture(admitted=True)
    artifact["score_correctness"] = {
        "kind": "exact_reference",
        "status": "pass",
        "max_abs_error": 0.0,
    }

    with pytest.raises(ValueError, match="same-scalar finite-difference"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load(LGSSM_VALUE_PATH),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
            require_admitted=True,
        )


def test_phase1_score_contract_rejects_parameterized_sir_diagnostic_row() -> None:
    artifact = _lgssm_score_fixture(admitted=False)
    artifact["row_id"] = PARAMETERIZED_SIR_DIAGNOSTIC_ROW_ID

    with pytest.raises(ValueError, match="row_id"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load(LGSSM_VALUE_PATH),
        )


def test_phase1_score_contract_rejects_ksc_exact_native_actual_sv_overclaim() -> None:
    artifact = {
        "schema_version": LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
        "row_id": KSC_SV_ROW_ID,
        "source_value_artifact": "docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json",
        "score_target_kind": LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
        "target_scalar": "observed_data_log_likelihood_estimator",
        "target_output_tensor_field": "log_likelihood",
        "target_observation_policy": "ksc_log_chi_square_gaussian_mixture_surrogate",
        "theta_coordinate_system": "synthetic_unconstrained",
        "score_parameter_names": ["gamma_unconstrained", "log_beta"],
        "score": [1.0, 2.0],
        "score_derivative_provenance": (
            "manual_total_vjp_no_autodiff_same_scalar_ksc_sv_ledh_pfpf_ot"
        ),
        "value_score_route_status": "same_route_value_score",
        "value_score_same_transport_algorithm": True,
        "no_autodiff_score_route": True,
        "uses_gradient_tape": False,
        "uses_forward_accumulator": False,
        "uses_stopped_partial_derivative": False,
        "claims_exact_native_actual_sv_likelihood": True,
        "score_correctness": {"kind": "same_scalar_finite_difference", "status": "pass"},
        "score_admission_status": LEDH_SCORE_ADMISSION_STATUS_TINY,
        "memory_diagnostics": {},
    }

    with pytest.raises(ValueError, match="claims_exact_native_actual_sv_likelihood"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load(KSC_VALUE_PATH),
            expected_row_id=KSC_SV_ROW_ID,
        )


def test_phase1_score_contract_rejects_nonfinite_score() -> None:
    artifact = copy.deepcopy(_lgssm_score_fixture(admitted=False))
    artifact["score"][2] = float("nan")

    with pytest.raises(ValueError, match="nonfinite"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load(LGSSM_VALUE_PATH),
            expected_row_id=LGSSM_M3_T50_ROW_ID,
        )
