"""LEDH same-target score artifact contract.

The score contract is intentionally an admission guard.  It does not compute a
score and it does not admit a row by itself.  It validates that a serialized
score artifact is tied to an admitted LEDH forward-scalar artifact and that the
claimed score is a no-tape total derivative of the same realized finite-N
estimator.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from bayesfilter.highdim.ledh_forward_contract import (
    ACTUAL_SV_ROW_ID,
    FIXED_SIR_AUSTRIA_ROW_ID,
    GENERALIZED_SV_ROW_ID,
    KSC_SV_ROW_ID,
    LEDH_FORWARD_ADMISSION_STATUS_ADMITTED,
    LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD,
    LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
    LGSSM_M3_T50_ROW_ID,
    PARAMETERIZED_SIR_DIAGNOSTIC_ROW_ID,
    PREDATOR_PREY_ROW_ID,
    validate_ledh_forward_scalar_artifact,
)


LEDH_SCORE_ARTIFACT_SCHEMA_VERSION = "bayesfilter.highdim.ledh_score_artifact.v1"
LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR = (
    "realized_finite_N_ledh_log_likelihood_estimator"
)
LEDH_SCORE_ADMISSION_STATUS_FULL = "n10000_same_target_no_tape_score_admitted"
LEDH_SCORE_ADMISSION_STATUS_TINY = "tiny_score_diagnostic_not_admitted"
LEDH_SCORE_ADMISSION_STATUS_BLOCKED_NOT_RUN = "blocked_score_not_run"
LEDH_SCORE_VALUE_ROUTE_STATUS_SAME = "same_route_value_score"
LEDH_SCORE_PRODUCTION_DTYPE = "float32"
LEDH_SCORE_PRODUCTION_TF32_MODE = "enabled"

_ALLOWED_SCORE_ADMISSION_STATUSES = frozenset(
    {
        LEDH_SCORE_ADMISSION_STATUS_FULL,
        LEDH_SCORE_ADMISSION_STATUS_TINY,
        LEDH_SCORE_ADMISSION_STATUS_BLOCKED_NOT_RUN,
    }
)
LEDH_SCORE_COMPACT_LGSSM_PROVENANCE = (
    "compact_forward_sensitivity_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot"
)
LEDH_SCORE_COMPACT_ACTUAL_SV_PROVENANCE = (
    "compact_forward_sensitivity_no_autodiff_same_scalar_actual_sv_ledh_pfpf_ot"
)
LEDH_SCORE_COMPACT_FIXED_SIR_PROVENANCE = (
    "compact_forward_sensitivity_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot"
)
LEDH_SCORE_COMPACT_PREDATOR_PREY_PROVENANCE = (
    "compact_forward_sensitivity_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot"
)
LEDH_SCORE_COMPACT_GENERALIZED_SV_PROVENANCE = (
    "compact_forward_sensitivity_no_autodiff_same_scalar_generalized_sv_ledh_pfpf_ot"
)
LEDH_SCORE_COMPACT_KSC_SV_PROVENANCE = (
    "compact_forward_sensitivity_no_autodiff_same_scalar_ksc_sv_ledh_pfpf_ot"
)
LEDH_SCORE_MEMORY_STYLE_LGSSM_PROVENANCE = (
    "memory_style_reverse_vjp_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot"
)
LEDH_SCORE_MEMORY_STYLE_FIXED_SIR_PROVENANCE = (
    "memory_style_reverse_vjp_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot"
)
LEDH_SCORE_MEMORY_STYLE_PREDATOR_PREY_PROVENANCE = (
    "memory_style_reverse_vjp_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot"
)
LEDH_SCORE_MEMORY_STYLE_ACTUAL_SV_PROVENANCE = (
    "memory_style_reverse_vjp_no_autodiff_same_scalar_actual_sv_ledh_pfpf_ot"
)
_COMPACT_ADMISSIBLE_NO_TAPE_PROVENANCE = frozenset(
    {
        LEDH_SCORE_COMPACT_LGSSM_PROVENANCE,
        LEDH_SCORE_COMPACT_ACTUAL_SV_PROVENANCE,
        LEDH_SCORE_COMPACT_FIXED_SIR_PROVENANCE,
        LEDH_SCORE_COMPACT_PREDATOR_PREY_PROVENANCE,
        LEDH_SCORE_COMPACT_GENERALIZED_SV_PROVENANCE,
        LEDH_SCORE_COMPACT_KSC_SV_PROVENANCE,
    }
)
_ROW_COMPACT_PROVENANCE = {
    LGSSM_M3_T50_ROW_ID: LEDH_SCORE_COMPACT_LGSSM_PROVENANCE,
    ACTUAL_SV_ROW_ID: LEDH_SCORE_COMPACT_ACTUAL_SV_PROVENANCE,
    FIXED_SIR_AUSTRIA_ROW_ID: LEDH_SCORE_COMPACT_FIXED_SIR_PROVENANCE,
    PREDATOR_PREY_ROW_ID: LEDH_SCORE_COMPACT_PREDATOR_PREY_PROVENANCE,
    GENERALIZED_SV_ROW_ID: LEDH_SCORE_COMPACT_GENERALIZED_SV_PROVENANCE,
    KSC_SV_ROW_ID: LEDH_SCORE_COMPACT_KSC_SV_PROVENANCE,
}
_HISTORICAL_DIAGNOSTIC_NO_TAPE_PROVENANCE = frozenset(
    {
        LEDH_SCORE_MEMORY_STYLE_LGSSM_PROVENANCE,
        LEDH_SCORE_MEMORY_STYLE_FIXED_SIR_PROVENANCE,
        LEDH_SCORE_MEMORY_STYLE_PREDATOR_PREY_PROVENANCE,
        LEDH_SCORE_MEMORY_STYLE_ACTUAL_SV_PROVENANCE,
        "manual_total_vjp_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot",
        "manual_total_vjp_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot",
        "manual_total_vjp_no_autodiff_same_scalar_actual_sv_ledh_pfpf_ot",
        "manual_total_vjp_no_autodiff_same_scalar_generalized_sv_ledh_pfpf_ot",
        "manual_total_vjp_no_autodiff_same_scalar_ksc_sv_ledh_pfpf_ot",
    }
)
_ALLOWED_NO_TAPE_PROVENANCE = (
    _COMPACT_ADMISSIBLE_NO_TAPE_PROVENANCE
    | _HISTORICAL_DIAGNOSTIC_NO_TAPE_PROVENANCE
)
_FORBIDDEN_DERIVATIVE_PROVENANCE_TOKENS = (
    "GradientTape",
    "ForwardAccumulator",
    "stopped",
    "stop_gradient",
    "partial_derivative",
)


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_bool_true(name: str, value: object) -> None:
    if value is not True:
        raise ValueError(f"{name} must be true")


def _require_bool_false(name: str, value: object) -> None:
    if value is not False:
        raise ValueError(f"{name} must be false")


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _require_text_tuple(name: str, values: object) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty string sequence")
    output = tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))
    if not output:
        raise ValueError(f"{name} must be nonempty")
    return output


def _require_finite_float_tuple(name: str, values: object) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty numeric sequence")
    output = tuple(float(value) for value in values)
    if not output:
        raise ValueError(f"{name} must be nonempty")
    nonfinite = [index for index, value in enumerate(output) if not math.isfinite(value)]
    if nonfinite:
        raise ValueError(f"{name} contains nonfinite values at indices {nonfinite}")
    return output


def _optional_finite_float(name: str, value: object) -> float | None:
    if value is None:
        return None
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{name} must be finite when provided")
    return output


def _provenance_has_forbidden_token(provenance: str) -> bool:
    lowered = provenance.lower()
    return any(token.lower() in lowered for token in _FORBIDDEN_DERIVATIVE_PROVENANCE_TOKENS)


def validate_ledh_score_production_precision(precision: Mapping[str, Any]) -> dict[str, Any]:
    """Validate production LEDH score precision metadata.

    Full LEDH score admission is for the production GPU TF32 lane. Reference or
    tiny diagnostic runs may use separate precision, but an admitted score
    artifact must disclose that the score computation itself used float32
    tensors with TensorFlow TF32 execution enabled.
    """

    payload = _require_mapping("score_precision", precision)
    dtype = _require_text("score_precision.dtype", payload.get("dtype"))
    active_dtype = _require_text(
        "score_precision.active_dtype",
        payload.get("active_dtype"),
    )
    tf_dtype = _require_text(
        "score_precision.tf_dtype",
        payload.get("tf_dtype"),
    )
    tf32_mode = _require_text("score_precision.tf32_mode", payload.get("tf32_mode"))
    tf32_execution_enabled = _require_bool(
        "score_precision.tf32_execution_enabled",
        payload.get("tf32_execution_enabled"),
    )
    if dtype != LEDH_SCORE_PRODUCTION_DTYPE:
        raise ValueError("score_precision.dtype must be float32 for production LEDH score")
    if active_dtype != LEDH_SCORE_PRODUCTION_DTYPE:
        raise ValueError(
            "score_precision.active_dtype must be float32 for production LEDH score"
        )
    if tf_dtype != LEDH_SCORE_PRODUCTION_DTYPE:
        raise ValueError("score_precision.tf_dtype must be float32 for production LEDH score")
    if tf32_mode != LEDH_SCORE_PRODUCTION_TF32_MODE:
        raise ValueError("score_precision.tf32_mode must be enabled for production LEDH score")
    if tf32_execution_enabled is not True:
        raise ValueError(
            "score_precision.tf32_execution_enabled must be true for production LEDH score"
        )
    return {
        "dtype": dtype,
        "active_dtype": active_dtype,
        "tf_dtype": tf_dtype,
        "tf32_mode": tf32_mode,
        "tf32_execution_enabled": tf32_execution_enabled,
    }


def validate_ledh_score_artifact(
    artifact: Mapping[str, Any],
    *,
    source_value_artifact: Mapping[str, Any],
    expected_row_id: str | None = None,
    require_admitted: bool = False,
) -> dict[str, Any]:
    """Validate and normalize an LEDH score artifact.

    The source value artifact is part of the validation input so score
    admission cannot drift away from the admitted finite-N value scalar.
    """

    payload = _require_mapping("artifact", artifact)
    if payload.get("schema_version") != LEDH_SCORE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("invalid LEDH score artifact schema_version")

    value_core = validate_ledh_forward_scalar_artifact(
        source_value_artifact,
        expected_row_id=expected_row_id,
        require_admitted=True,
    )

    row_id = _require_text("row_id", payload.get("row_id"))
    if row_id != value_core["row_id"]:
        raise ValueError("score row_id must match admitted value artifact")
    if row_id == PARAMETERIZED_SIR_DIAGNOSTIC_ROW_ID:
        raise ValueError("parameterized SIR diagnostic row cannot be score-admitted")

    target_scalar = _require_text("target_scalar", payload.get("target_scalar"))
    if target_scalar != LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD:
        raise ValueError("score target_scalar must be observed_data_log_likelihood_estimator")
    if target_scalar != value_core["target_scalar"]:
        raise ValueError("score target_scalar must match admitted value artifact")

    output_field = _require_text("target_output_tensor_field", payload.get("target_output_tensor_field"))
    if output_field != LEDH_OUTPUT_TENSOR_FIELD_LOG_LIKELIHOOD:
        raise ValueError("score output field target_output_tensor_field must be log_likelihood")
    if output_field != value_core["target_output_tensor_field"]:
        raise ValueError("score output field must match admitted value artifact")

    target_policy = _require_text("target_observation_policy", payload.get("target_observation_policy"))
    if target_policy != value_core["target_observation_policy"]:
        raise ValueError("score target_observation_policy must match admitted value artifact")

    theta_coordinate_system = _require_text(
        "theta_coordinate_system",
        payload.get("theta_coordinate_system"),
    )
    if theta_coordinate_system != value_core["theta_coordinate_system"]:
        raise ValueError("score theta_coordinate_system must match admitted value artifact")

    parameter_names = _require_text_tuple("score_parameter_names", payload.get("score_parameter_names"))
    value_parameter_order = tuple(value_core["forward_contract"]["theta_contract"]["parameter_order"])
    if parameter_names != value_parameter_order:
        raise ValueError("score_parameter_names must match admitted value parameter order")

    score_values = _require_finite_float_tuple("score", payload.get("score"))
    if len(score_values) != len(parameter_names):
        raise ValueError("score length must match score_parameter_names")

    score_target_kind = _require_text("score_target_kind", payload.get("score_target_kind"))
    if score_target_kind != LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR:
        raise ValueError("score_target_kind must be realized finite-N LEDH estimator")

    value_score_route_status = _require_text(
        "value_score_route_status",
        payload.get("value_score_route_status"),
    )
    if value_score_route_status != LEDH_SCORE_VALUE_ROUTE_STATUS_SAME:
        raise ValueError("value_score_route_status must be same_route_value_score")
    _require_bool_true(
        "value_score_same_transport_algorithm",
        payload.get("value_score_same_transport_algorithm"),
    )

    derivative_provenance = _require_text(
        "score_derivative_provenance",
        payload.get("score_derivative_provenance"),
    )
    if derivative_provenance not in _ALLOWED_NO_TAPE_PROVENANCE:
        raise ValueError("unsupported no-tape score_derivative_provenance")
    if _provenance_has_forbidden_token(derivative_provenance):
        raise ValueError("score_derivative_provenance contains forbidden autodiff/partial token")
    _require_bool_true("no_autodiff_score_route", payload.get("no_autodiff_score_route"))
    _require_bool_false("uses_gradient_tape", payload.get("uses_gradient_tape", False))
    _require_bool_false("uses_forward_accumulator", payload.get("uses_forward_accumulator", False))
    _require_bool_false("uses_stopped_partial_derivative", payload.get("uses_stopped_partial_derivative", False))

    fd_or_exact = _require_mapping("score_correctness", payload.get("score_correctness"))
    correctness_status = _require_text("score_correctness.status", fd_or_exact.get("status"))
    if correctness_status != "pass":
        raise ValueError("score correctness status must pass")
    correctness_kind = _require_text("score_correctness.kind", fd_or_exact.get("kind"))
    if correctness_kind not in {"same_scalar_finite_difference", "exact_reference"}:
        raise ValueError("score correctness kind must be same-scalar FD or exact reference")

    admission_status = _require_text("score_admission_status", payload.get("score_admission_status"))
    if admission_status not in _ALLOWED_SCORE_ADMISSION_STATUSES:
        raise ValueError(f"unsupported score_admission_status: {admission_status}")
    if (
        admission_status == LEDH_SCORE_ADMISSION_STATUS_FULL
        and derivative_provenance in _HISTORICAL_DIAGNOSTIC_NO_TAPE_PROVENANCE
    ):
        raise ValueError(
            "historical memory_style/manual_total_vjp score routes cannot "
            "be full LEDH score admission; compact no-tape recurrence is required"
        )
    if (
        admission_status == LEDH_SCORE_ADMISSION_STATUS_FULL
        and derivative_provenance not in _COMPACT_ADMISSIBLE_NO_TAPE_PROVENANCE
    ):
        raise ValueError("full LEDH score admission requires compact no-tape provenance")
    if admission_status == LEDH_SCORE_ADMISSION_STATUS_FULL:
        expected_compact_provenance = _ROW_COMPACT_PROVENANCE.get(row_id)
        if expected_compact_provenance is None:
            raise ValueError("full LEDH score admission requires known row compact provenance")
        if derivative_provenance != expected_compact_provenance:
            raise ValueError(
                "full LEDH score admission requires row-matched compact provenance"
            )
    if (
        admission_status == LEDH_SCORE_ADMISSION_STATUS_FULL
        and correctness_kind != "same_scalar_finite_difference"
    ):
        raise ValueError(
            "full LEDH score admission requires same-scalar finite-difference "
            "correctness; exact references are diagnostic only unless a reviewed "
            "contract proves they are the same realized finite-N LEDH scalar"
        )
    if require_admitted and admission_status != LEDH_SCORE_ADMISSION_STATUS_FULL:
        raise ValueError("score artifact is not admitted")

    source_path = _require_text("source_value_artifact", payload.get("source_value_artifact"))
    if not source_path.endswith(".json"):
        raise ValueError("source_value_artifact must name a JSON artifact path")

    memory = _require_mapping("memory_diagnostics", payload.get("memory_diagnostics", {}))
    precision = None
    if admission_status == LEDH_SCORE_ADMISSION_STATUS_FULL:
        precision = validate_ledh_score_production_precision(
            _require_mapping("score_precision", payload.get("score_precision")),
        )
        if value_core["admission_status"] != LEDH_FORWARD_ADMISSION_STATUS_ADMITTED:
            raise ValueError("full score admission requires admitted source value artifact")
        if value_core["num_particles"] < 10000:
            raise ValueError("full score admission requires N>=10000 source value artifact")
        _require_bool_true("memory_diagnostics.n10000_memory_pass", memory.get("n10000_memory_pass"))
        source = _require_text("memory_diagnostics.source", memory.get("source"))
        allowed_sources = {
            "score_gpu_memory_info_after",
            "max_per_seed_score_gpu_memory_info_after",
            "trusted_gpu_score_memory_artifact",
        }
        if source not in allowed_sources:
            raise ValueError("memory_diagnostics.source must identify a trusted score memory measurement")
        peak_mib = _optional_finite_float("memory_diagnostics.peak_mib", memory.get("peak_mib"))
        budget_mib = _optional_finite_float("memory_diagnostics.budget_mib", memory.get("budget_mib"))
        if peak_mib is None:
            raise ValueError("memory_diagnostics.peak_mib must be finite for full score admission")
        if budget_mib is None:
            raise ValueError("memory_diagnostics.budget_mib must be finite for full score admission")
        if peak_mib > budget_mib:
            raise ValueError("memory_diagnostics peak_mib exceeds budget_mib")
    else:
        if require_admitted:
            raise ValueError("score artifact is not admitted")

    if row_id.endswith("ksc_gaussian_mixture_surrogate_T1000"):
        _require_bool_false(
            "claims_exact_native_actual_sv_likelihood",
            payload.get("claims_exact_native_actual_sv_likelihood", False),
        )

    return {
        "schema_version": LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
        "row_id": row_id,
        "source_value_artifact": source_path,
        "score_target_kind": score_target_kind,
        "target_scalar": target_scalar,
        "target_output_tensor_field": output_field,
        "target_observation_policy": target_policy,
        "theta_coordinate_system": theta_coordinate_system,
        "score_parameter_names": list(parameter_names),
        "score": list(score_values),
        "score_derivative_provenance": derivative_provenance,
        "value_score_route_status": value_score_route_status,
        "value_score_same_transport_algorithm": True,
        "no_autodiff_score_route": True,
        "score_correctness": dict(fd_or_exact),
        "score_admission_status": admission_status,
        "memory_diagnostics": dict(memory),
        "score_precision": precision,
        "admitted": admission_status == LEDH_SCORE_ADMISSION_STATUS_FULL,
    }
