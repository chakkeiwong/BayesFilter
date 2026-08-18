from __future__ import annotations

import math

import pytest

from docs.benchmarks.genut_fd_regression import (
    FD_REGRESSION_POLICY_ID,
    evaluate_regression_derivative,
    fit_quadratic_step_regression,
)


def test_quadratic_regression_recovers_zero_step_intercept() -> None:
    steps = (1.0e-1, 5.0e-2, 2.0e-2, 1.0e-2)
    values = tuple(2.5 + 7.0 * step * step for step in steps)
    result = fit_quadratic_step_regression(steps, values)
    assert result["policy_id"] == FD_REGRESSION_POLICY_ID
    assert math.isclose(float(result["intercept"]), 2.5, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(float(result["slope_h_squared"]), 7.0, rel_tol=0.0, abs_tol=1.0e-10)
    assert float(result["sum_squared_residual"]) <= 1.0e-24


def test_regression_derivative_reports_scale_aware_diagnostic() -> None:
    result = fit_quadratic_step_regression(
        (1.0e-2, 3.0e-3, 1.0e-3),
        (1.001, 1.00009, 1.00001),
    )
    checked = evaluate_regression_derivative(1.0, result)
    assert checked["diagnostic_pass"] is True
    assert float(checked["relative_error"]) < 1.0e-3


@pytest.mark.parametrize(
    "steps,values",
    [((1.0e-3, 1.0e-3, 1.0e-3), (1.0, 1.0, 1.0)), ((1.0, 2.0), (1.0, 2.0))],
)
def test_regression_rejects_ill_defined_ladder(steps, values) -> None:
    with pytest.raises(ValueError):
        fit_quadratic_step_regression(steps, values)
