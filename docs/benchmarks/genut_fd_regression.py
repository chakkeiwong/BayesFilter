"""Diagnostic quadratic-regression finite-difference checks for GenUT.

This module is benchmark-only. It estimates the zero-step derivative from
central differences at several step sizes by fitting ``d(h) = a + b h^2``.
It does not establish filtering accuracy or posterior correctness.
"""

from __future__ import annotations

import math
from typing import Sequence


FD_REGRESSION_POLICY_ID = "central_fd_h2_intercept_regression_v1"
FD_REGRESSION_STEPS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4)
FD_REGRESSION_RELATIVE_TOLERANCE = 0.05
FD_REGRESSION_DENOMINATOR_FLOOR = 1.0e-12


def _finite_sequence(name: str, values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be a nonempty finite sequence")
    return result


def fit_quadratic_step_regression(
    steps: Sequence[float], finite_differences: Sequence[float]
) -> dict[str, object]:
    """Fit the central-FD ladder ``d(h) = intercept + slope*h**2``."""

    h = _finite_sequence("steps", steps)
    y = _finite_sequence("finite_differences", finite_differences)
    if len(h) != len(y) or len(h) < 3:
        raise ValueError("steps and finite_differences need at least three pairs")
    if any(value <= 0.0 for value in h):
        raise ValueError("steps must be positive")
    x = tuple(value * value for value in h)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    centered_x = tuple(value - x_mean for value in x)
    centered_y = tuple(value - y_mean for value in y)
    sxx = sum(value * value for value in centered_x)
    if sxx <= 0.0:
        raise ValueError("steps must contain at least two distinct magnitudes")
    slope = sum(a * b for a, b in zip(centered_x, centered_y, strict=True)) / sxx
    intercept = y_mean - slope * x_mean
    fitted = tuple(intercept + slope * value for value in x)
    residuals = tuple(actual - predicted for actual, predicted in zip(y, fitted, strict=True))
    sse = sum(value * value for value in residuals)
    sst = sum(value * value for value in centered_y)
    r_squared = 1.0 - sse / sst if sst > 0.0 else 1.0
    x_span = max(x) - min(x)
    return {
        "policy_id": FD_REGRESSION_POLICY_ID,
        "steps": list(h),
        "step_squared": list(x),
        "finite_differences": list(y),
        "intercept": intercept,
        "slope_h_squared": slope,
        "fitted": list(fitted),
        "residuals": list(residuals),
        "sum_squared_residual": sse,
        "r_squared": r_squared,
        "step_squared_span": x_span,
        "regression_condition_proxy": max(x) / x_span,
    }


def evaluate_regression_derivative(
    score: float, regression: dict[str, object]
) -> dict[str, object]:
    """Compare a regression intercept to the manual score diagnostically."""

    score_value = float(score)
    intercept = float(regression["intercept"])
    if not math.isfinite(score_value) or not math.isfinite(intercept):
        raise ValueError("score and regression intercept must be finite")
    absolute_error = abs(intercept - score_value)
    scale = max(abs(intercept), abs(score_value), FD_REGRESSION_DENOMINATOR_FLOOR)
    relative_error = absolute_error / scale
    return {
        **regression,
        "score": score_value,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "relative_tolerance": FD_REGRESSION_RELATIVE_TOLERANCE,
        "diagnostic_pass": relative_error <= FD_REGRESSION_RELATIVE_TOLERANCE,
    }


__all__ = [
    "FD_REGRESSION_DENOMINATOR_FLOOR",
    "FD_REGRESSION_POLICY_ID",
    "FD_REGRESSION_RELATIVE_TOLERANCE",
    "FD_REGRESSION_STEPS",
    "evaluate_regression_derivative",
    "fit_quadratic_step_regression",
]
