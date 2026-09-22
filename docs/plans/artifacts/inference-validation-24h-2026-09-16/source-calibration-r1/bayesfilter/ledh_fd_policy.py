"""Owner-directed individual-direction finite-difference diagnostic policy."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


LEDH_FD_POLICY_ID = "owner_directed_individual_direction_relative_5pct_sqrt_p_v1"
LEDH_FD_BASE_RELATIVE_TOLERANCE = 0.05
LEDH_FD_RELATIVE_ERROR_DENOMINATOR_FLOOR = 1.0e-12
LEDH_FD_DENOMINATOR = "max(abs(score), abs(finite_difference), 1e-12)"
LEDH_FD_PASS_RULE = (
    "max_coordinate_relative_error <= 0.05 * sqrt(num_parameters)"
)
LEDH_FD_DIAGNOSTIC_SCOPE = "finite_difference_only"
LEDH_FD_STATISTICAL_STATUS = (
    "five_percent_selected_to_mirror_the_conventional_95pct_threshold; "
    "no_confidence_interval_or_coverage_calibration_is_computed"
)
LEDH_FD_STEP_POLICY_ID = "float32_central_difference_cuberoot_epsilon_coordinate_scale_v1"
LEDH_FD_STEP_DTYPE = "float32"
LEDH_FD_FLOAT32_EPSILON = 2.0**-23
LEDH_FD_STEP_COEFFICIENT = LEDH_FD_FLOAT32_EPSILON ** (1.0 / 3.0)
LEDH_FD_STEP_SCALE = "max(1, abs(theta_j))"
LEDH_FD_STEP_FORMULA = "cbrt(float32_epsilon) * max(1, abs(theta_j))"


def _finite_vector(name: str, values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty numeric sequence")
    output = tuple(float(value) for value in values)
    if not output:
        raise ValueError(f"{name} must be nonempty")
    nonfinite = [index for index, value in enumerate(output) if not math.isfinite(value)]
    if nonfinite:
        raise ValueError(f"{name} contains nonfinite values at indices {nonfinite}")
    return output


def _parameter_names(values: Sequence[str], expected_length: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("parameter_names must be a nonempty string sequence")
    output = tuple(str(value) for value in values)
    if len(output) != expected_length or any(not value for value in output):
        raise ValueError("parameter_names must match the score dimension")
    if len(set(output)) != len(output):
        raise ValueError("parameter_names must be unique")
    return output


def coordinate_relative_error(score: float, finite_difference: float) -> tuple[float, float, float]:
    """Return the stored Phase 9 absolute, scale, and relative FD errors."""

    score_value = float(score)
    fd_value = float(finite_difference)
    if not math.isfinite(score_value) or not math.isfinite(fd_value):
        raise ValueError("score and finite_difference must be finite")
    absolute_error = abs(score_value - fd_value)
    relative_scale = max(
        abs(score_value),
        abs(fd_value),
        LEDH_FD_RELATIVE_ERROR_DENOMINATOR_FLOOR,
    )
    relative_error = absolute_error / relative_scale
    return absolute_error, relative_scale, relative_error


def coordinate_central_difference_step(theta: float) -> float:
    """Return the coordinate-scaled central-FD step for a float32 objective."""

    theta_value = float(theta)
    if not math.isfinite(theta_value):
        raise ValueError("theta must be finite")
    return LEDH_FD_STEP_COEFFICIENT * max(1.0, abs(theta_value))


def ledh_fd_step_policy_metadata() -> dict[str, Any]:
    """Return the numerical policy used to construct float32 FD endpoints."""

    return {
        "policy_id": LEDH_FD_STEP_POLICY_ID,
        "dtype": LEDH_FD_STEP_DTYPE,
        "machine_epsilon": LEDH_FD_FLOAT32_EPSILON,
        "coefficient": LEDH_FD_STEP_COEFFICIENT,
        "coordinate_scale": LEDH_FD_STEP_SCALE,
        "nominal_step_formula": LEDH_FD_STEP_FORMULA,
        "effective_denominator": "plus_theta_j - minus_theta_j",
    }


def evaluate_ledh_fd_policy(
    scores: Sequence[float],
    finite_differences: Sequence[float],
    parameter_names: Sequence[str],
) -> dict[str, Any]:
    """Evaluate the FD-only maximum individual-direction relative-error rule."""

    score_values = _finite_vector("scores", scores)
    fd_values = _finite_vector("finite_differences", finite_differences)
    if len(score_values) != len(fd_values):
        raise ValueError("scores and finite_differences must have equal length")
    names = _parameter_names(parameter_names, len(score_values))

    parameters = []
    relative_errors = []
    for name, score, finite_difference in zip(names, score_values, fd_values, strict=True):
        absolute_error, relative_scale, relative_error = coordinate_relative_error(
            score,
            finite_difference,
        )
        relative_errors.append(relative_error)
        parameters.append(
            {
                "parameter": name,
                "score": score,
                "finite_difference": finite_difference,
                "absolute_error": absolute_error,
                "relative_error_scale": relative_scale,
                "relative_error": relative_error,
            }
        )

    num_parameters = len(parameters)
    threshold = LEDH_FD_BASE_RELATIVE_TOLERANCE * math.sqrt(num_parameters)
    max_index = max(range(num_parameters), key=relative_errors.__getitem__)
    max_relative_error = relative_errors[max_index]
    passed = max_relative_error <= threshold
    return {
        "policy_id": LEDH_FD_POLICY_ID,
        "diagnostic_scope": LEDH_FD_DIAGNOSTIC_SCOPE,
        "status": "pass" if passed else "fail",
        "num_parameters": num_parameters,
        "base_relative_tolerance": LEDH_FD_BASE_RELATIVE_TOLERANCE,
        "dimension_scaling": "sqrt(num_parameters)",
        "coordinate_relative_error_denominator": LEDH_FD_DENOMINATOR,
        "pass_rule": LEDH_FD_PASS_RULE,
        "statistical_interpretation": LEDH_FD_STATISTICAL_STATUS,
        "max_coordinate_relative_error": max_relative_error,
        "max_coordinate_relative_error_threshold": threshold,
        "max_coordinate_relative_error_margin": threshold - max_relative_error,
        "max_error_parameter": names[max_index],
        "parameters": parameters,
    }


def validate_declared_ledh_fd_policy(
    declared: Mapping[str, Any],
    scores: Sequence[float],
    finite_differences: Sequence[float],
    parameter_names: Sequence[str],
) -> dict[str, Any]:
    """Recompute and validate a serialized FD-only policy result."""

    if not isinstance(declared, Mapping):
        raise ValueError("declared FD policy result must be a mapping")
    expected = evaluate_ledh_fd_policy(scores, finite_differences, parameter_names)
    if dict(declared) != expected:
        raise ValueError(
            "declared FD policy result does not match recomputed maximum-direction policy"
        )
    return expected


__all__ = [
    "LEDH_FD_BASE_RELATIVE_TOLERANCE",
    "LEDH_FD_DENOMINATOR",
    "LEDH_FD_DIAGNOSTIC_SCOPE",
    "LEDH_FD_PASS_RULE",
    "LEDH_FD_POLICY_ID",
    "LEDH_FD_RELATIVE_ERROR_DENOMINATOR_FLOOR",
    "LEDH_FD_STATISTICAL_STATUS",
    "LEDH_FD_FLOAT32_EPSILON",
    "LEDH_FD_STEP_COEFFICIENT",
    "LEDH_FD_STEP_DTYPE",
    "LEDH_FD_STEP_FORMULA",
    "LEDH_FD_STEP_POLICY_ID",
    "LEDH_FD_STEP_SCALE",
    "coordinate_central_difference_step",
    "coordinate_relative_error",
    "evaluate_ledh_fd_policy",
    "ledh_fd_step_policy_metadata",
    "validate_declared_ledh_fd_policy",
]
