"""Builders for LEDH score artifacts.

The helpers in this module only assemble and validate score-artifact payloads.
They do not compute scores and they do not weaken the admission gate in
``ledh_score_contract``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from bayesfilter.highdim.ledh_forward_contract import validate_ledh_forward_scalar_artifact
from bayesfilter.highdim.ledh_score_contract import (
    LEDH_SCORE_ADMISSION_STATUS_FULL,
    LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
    LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
    LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
    validate_ledh_score_artifact,
)


def _finite_float_list(name: str, values: Sequence[float]) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a numeric sequence")
    output = [float(value) for value in values]
    if not output:
        raise ValueError(f"{name} must be nonempty")
    nonfinite = [index for index, value in enumerate(output) if not math.isfinite(value)]
    if nonfinite:
        raise ValueError(f"{name} contains nonfinite values at indices {nonfinite}")
    return output


def _text_list(name: str, values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a string sequence")
    output = [str(value) for value in values]
    if not output or any(not value for value in output):
        raise ValueError(f"{name} must be a nonempty string sequence")
    return output


def build_ledh_score_artifact(
    *,
    source_value_artifact: Mapping[str, Any],
    source_value_artifact_path: str,
    expected_row_id: str,
    score_parameter_names: Sequence[str],
    score: Sequence[float],
    score_derivative_provenance: str,
    score_correctness: Mapping[str, Any],
    score_admission_status: str,
    memory_diagnostics: Mapping[str, Any] | None = None,
    score_precision: Mapping[str, Any] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
    require_admitted: bool | None = None,
) -> dict[str, Any]:
    """Build and validate an LEDH same-target score artifact.

    ``require_admitted`` defaults to true only for full-admission artifacts.
    The returned payload is the serializable artifact, not the validator's
    normalized summary, so model-specific boundary fields such as the KSC
    exact-native-likelihood nonclaim can be preserved.
    """

    value_core = validate_ledh_forward_scalar_artifact(
        source_value_artifact,
        expected_row_id=expected_row_id,
        require_admitted=True,
    )
    source_path = str(source_value_artifact_path)
    parameter_names = _text_list("score_parameter_names", score_parameter_names)
    score_values = _finite_float_list("score", score)
    if len(parameter_names) != len(score_values):
        raise ValueError("score length must match score_parameter_names")
    if tuple(parameter_names) != tuple(
        value_core["forward_contract"]["theta_contract"]["parameter_order"]
    ):
        raise ValueError("score_parameter_names must match admitted value parameter order")

    artifact: dict[str, Any] = {
        "schema_version": LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
        "row_id": value_core["row_id"],
        "source_value_artifact": source_path,
        "score_target_kind": LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
        "target_scalar": value_core["target_scalar"],
        "target_output_tensor_field": value_core["target_output_tensor_field"],
        "target_observation_policy": value_core["target_observation_policy"],
        "theta_coordinate_system": value_core["theta_coordinate_system"],
        "score_parameter_names": parameter_names,
        "score": score_values,
        "score_derivative_provenance": str(score_derivative_provenance),
        "value_score_route_status": LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
        "value_score_same_transport_algorithm": True,
        "no_autodiff_score_route": True,
        "uses_gradient_tape": False,
        "uses_forward_accumulator": False,
        "uses_stopped_partial_derivative": False,
        "score_correctness": dict(score_correctness),
        "score_admission_status": str(score_admission_status),
        "memory_diagnostics": dict(memory_diagnostics or {}),
    }
    if score_precision is not None:
        artifact["score_precision"] = dict(score_precision)
    if extra_fields:
        artifact.update(dict(extra_fields))

    should_require_admitted = (
        score_admission_status == LEDH_SCORE_ADMISSION_STATUS_FULL
        if require_admitted is None
        else bool(require_admitted)
    )
    validate_ledh_score_artifact(
        artifact,
        source_value_artifact=source_value_artifact,
        expected_row_id=expected_row_id,
        require_admitted=should_require_admitted,
    )
    return artifact
