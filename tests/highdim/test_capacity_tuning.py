from __future__ import annotations

import math

import pytest

from bayesfilter.highdim.capacity_tuning import (
    INVALID_INPUT,
    VALUE_STABLE,
    VALUE_UNSTABLE,
    SignificantPlacePolicy,
    assert_frozen_scope_equal,
    compare_likelihood_values,
    nominate_capacity,
    significant_place,
)


POLICY = SignificantPlacePolicy()


@pytest.mark.parametrize(
    ("scale", "expected"),
    [(21.0, 0.1), (0.21, 0.001), (2100.0, 10.0), (0.0, 0.001)],
)
def test_significant_place_is_scale_aware(scale: float, expected: float) -> None:
    assert significant_place(scale, POLICY) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("low", "high", "status"),
    [
        (-20.4475727, -20.1809302, VALUE_STABLE),
        (-20.4475727, -21.1809302, VALUE_UNSTABLE),
        (0.2100, 0.2199, VALUE_STABLE),
        (0.2100, 0.2201, VALUE_UNSTABLE),
        (0.0, 1.0e-13, VALUE_STABLE),
    ],
)
def test_total_value_decision(low: float, high: float, status: str) -> None:
    result = compare_likelihood_values(low_value=low, high_value=high)
    assert result["status"] == status


def test_exact_tolerance_boundary_passes() -> None:
    result = compare_likelihood_values(low_value=-20.0, high_value=-20.9)
    assert result["significant_prefix_equal"] is True
    assert result["stable"] is True


def test_third_digit_change_is_allowed_but_second_digit_change_is_not() -> None:
    allowed = compare_likelihood_values(low_value=-20.447, high_value=-20.180)
    rejected = compare_likelihood_values(low_value=-20.447, high_value=-21.180)
    assert allowed["stable"] is True
    assert rejected["stable"] is False


def test_increment_cancellation_is_warning_only() -> None:
    result = compare_likelihood_values(
        low_value=-20.0,
        high_value=-20.01,
        low_increments=(-10.0, -10.0),
        high_increments=(-9.0, -11.01),
    )
    assert result["status"] == VALUE_STABLE
    assert result["cancellation_warning"] is True
    assert any(not row["stable"] for row in result["increment_comparisons"])


def test_invalid_values_and_increment_shapes_fail_closed() -> None:
    nonfinite = compare_likelihood_values(low_value=0.0, high_value=math.inf)
    assert nonfinite["status"] == INVALID_INPUT
    mismatched = compare_likelihood_values(
        low_value=0.0,
        high_value=0.0,
        low_increments=(0.0,),
        high_increments=(0.0, 0.0),
    )
    assert mismatched["status"] == INVALID_INPUT


def test_frozen_scope_drift_is_rejected() -> None:
    assert_frozen_scope_equal({"model": "sv", "horizon": 10}, {"horizon": 10, "model": "sv"})
    with pytest.raises(ValueError, match="frozen scope differs"):
        assert_frozen_scope_equal({"model": "sv"}, {"model": "sir"})


def _cell(value: float, *, valid: bool = True, scope: str = "scope-a") -> dict[str, object]:
    return {
        "value": value,
        "increments": (0.4 * value, 0.6 * value),
        "invariant_pass": valid,
        "frozen_scope": {"scope": scope},
    }


def test_nomination_selects_minimum_coefficient_count() -> None:
    cells = {
        (4, 2): _cell(-20.00),
        (6, 2): _cell(-20.01),
        (4, 4): _cell(-20.02),
        (6, 4): _cell(-20.01),
        (8, 2): _cell(-20.01),
        (8, 4): _cell(-20.01),
    }
    result = nominate_capacity(cells, degrees=(4, 6, 8), ranks=(2, 4))
    assert result["status"] == "nominated"
    assert result["nominee"]["degree"] == 4
    assert result["nominee"]["rank"] == 2
    assert result["nominee"]["coefficient_count"] == 20
    assert result["axis_summary"]["degree_all_stable"] is True
    assert result["axis_summary"]["rank_all_stable"] is True
    assert result["axis_summary"]["degree_comparison_count"] == 4
    assert result["axis_summary"]["rank_comparison_count"] == 3


def test_boundary_invalid_and_drifted_cells_are_not_nominated() -> None:
    boundary_only = {(8, 4): _cell(-20.0)}
    assert (
        nominate_capacity(boundary_only, degrees=(4, 6, 8), ranks=(2, 4))["status"]
        == "no_nominee"
    )
    drifted = {
        (4, 2): _cell(-20.0),
        (6, 2): _cell(-20.0, scope="scope-b"),
        (4, 4): _cell(-20.0),
    }
    assert (
        nominate_capacity(drifted, degrees=(4, 6), ranks=(2, 4))["status"]
        == "no_nominee"
    )
    invalid = {
        (4, 2): _cell(-20.0),
        (6, 2): _cell(-20.0, valid=False),
        (4, 4): _cell(-20.0),
    }
    assert (
        nominate_capacity(invalid, degrees=(4, 6), ranks=(2, 4))["status"]
        == "no_nominee"
    )
