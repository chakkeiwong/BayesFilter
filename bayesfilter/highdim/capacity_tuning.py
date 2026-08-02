"""Model-independent likelihood-value capacity self-convergence utilities."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


VALUE_STABLE = "value_stable_first_2_of_3_significant_digits"
VALUE_UNSTABLE = "value_unstable"
INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class SignificantPlacePolicy:
    """Significant-digit-prefix comparison policy.

    With the default three-digit request, the first two significant digits
    must agree. The third significant digit is allowed to change.
    """

    significant_digits: int = 3
    stable_prefix_digits: int | None = None
    near_zero_threshold: float = 1.0e-12
    near_zero_place: float = 1.0e-3

    def __post_init__(self) -> None:
        if int(self.significant_digits) <= 0:
            raise ValueError("significant_digits must be positive")
        if not math.isfinite(float(self.near_zero_threshold)):
            raise ValueError("near_zero_threshold must be finite")
        if not math.isfinite(float(self.near_zero_place)):
            raise ValueError("near_zero_place must be finite")
        if float(self.near_zero_threshold) < 0.0:
            raise ValueError("near_zero_threshold must be nonnegative")
        if float(self.near_zero_place) <= 0.0:
            raise ValueError("near_zero_place must be positive")
        object.__setattr__(self, "significant_digits", int(self.significant_digits))
        prefix_digits = (
            self.significant_digits - 1
            if self.stable_prefix_digits is None
            else int(self.stable_prefix_digits)
        )
        if prefix_digits <= 0 or prefix_digits >= self.significant_digits:
            raise ValueError(
                "stable_prefix_digits must be positive and less than significant_digits"
            )
        object.__setattr__(self, "stable_prefix_digits", prefix_digits)
        object.__setattr__(self, "near_zero_threshold", float(self.near_zero_threshold))
        object.__setattr__(self, "near_zero_place", float(self.near_zero_place))

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "significant_digits": self.significant_digits,
            "stable_prefix_digits": self.stable_prefix_digits,
            "near_zero_threshold": self.near_zero_threshold,
            "near_zero_place": self.near_zero_place,
            "decision_rule": "leading_significant_digit_prefix_equal",
        }


def significant_place(scale: float, policy: SignificantPlacePolicy) -> float:
    """Return the absolute place value of the first omitted digit."""

    magnitude = float(scale)
    if not math.isfinite(magnitude) or magnitude < 0.0:
        raise ValueError("scale must be finite and nonnegative")
    if magnitude <= policy.near_zero_threshold:
        return policy.near_zero_place
    exponent = math.floor(math.log10(magnitude)) - policy.significant_digits + 1
    return 10.0**exponent


def _significant_prefix(value: float, policy: SignificantPlacePolicy) -> tuple[Any, ...]:
    """Return a sign/exponent/leading-digit prefix, without rounding."""

    magnitude = abs(float(value))
    if magnitude <= policy.near_zero_threshold:
        return (0, 0, 0)
    exponent = math.floor(math.log10(magnitude))
    place = 10.0 ** (exponent - policy.stable_prefix_digits + 1)
    leading_digits = int(math.floor(magnitude / place + 1.0e-12))
    return (
        -1 if value < 0.0 else 1,
        exponent,
        leading_digits,
    )


def _scalar_comparison(
    low: float,
    high: float,
    policy: SignificantPlacePolicy,
) -> dict[str, Any]:
    low_value = float(low)
    high_value = float(high)
    finite = math.isfinite(low_value) and math.isfinite(high_value)
    if not finite:
        return {
            "status": INVALID_INPUT,
            "stable": False,
            "low_value": low_value if math.isfinite(low_value) else str(low_value),
            "high_value": high_value if math.isfinite(high_value) else str(high_value),
            "reason": "values_must_be_finite",
        }
    scale = max(abs(low_value), abs(high_value))
    place = significant_place(scale, policy)
    delta = abs(high_value - low_value)
    boundary_slack = 8.0 * max(
        math.ulp(low_value),
        math.ulp(high_value),
        math.ulp(delta),
        math.ulp(place),
    )
    low_prefix = _significant_prefix(low_value, policy)
    high_prefix = _significant_prefix(high_value, policy)
    prefix_equal = low_prefix == high_prefix
    near_zero_stable = scale <= policy.near_zero_threshold and delta <= policy.near_zero_place + boundary_slack
    stable = bool(prefix_equal or near_zero_stable)
    return {
        "status": VALUE_STABLE if stable else VALUE_UNSTABLE,
        "stable": stable,
        "low_value": low_value,
        "high_value": high_value,
        "absolute_delta_v": delta,
        "scale": scale,
        "significant_place": place,
        "decision_prefix_digits": policy.stable_prefix_digits,
        "low_significant_prefix": low_prefix,
        "high_significant_prefix": high_prefix,
        "significant_prefix_equal": prefix_equal,
        "decision_rule": "first_two_of_three_significant_digits_equal",
        "boundary_slack": boundary_slack,
        "normalized_delta_v": delta / max(scale, 1.0e-300),
        "low_rounded_3sf": format(low_value, f".{policy.significant_digits}g"),
        "high_rounded_3sf": format(high_value, f".{policy.significant_digits}g"),
    }


def compare_likelihood_values(
    *,
    low_value: float,
    high_value: float,
    low_increments: Sequence[float] | None = None,
    high_increments: Sequence[float] | None = None,
    policy: SignificantPlacePolicy = SignificantPlacePolicy(),
) -> dict[str, Any]:
    """Compare adjacent likelihood values and optionally their increments.

    Increment comparisons are explanatory. They can set
    ``cancellation_warning`` but never change the total-likelihood decision.
    """

    total = _scalar_comparison(low_value, high_value, policy)
    if (low_increments is None) != (high_increments is None):
        return {
            **total,
            "status": INVALID_INPUT,
            "stable": False,
            "reason": "both_increment_sequences_are_required",
            "increment_comparisons": (),
            "cancellation_warning": False,
        }
    increment_rows: list[dict[str, Any]] = []
    if low_increments is not None and high_increments is not None:
        low_rows = tuple(float(value) for value in low_increments)
        high_rows = tuple(float(value) for value in high_increments)
        if len(low_rows) != len(high_rows):
            return {
                **total,
                "status": INVALID_INPUT,
                "stable": False,
                "reason": "increment_lengths_differ",
                "increment_comparisons": (),
                "cancellation_warning": False,
            }
        for index, (low_increment, high_increment) in enumerate(
            zip(low_rows, high_rows)
        ):
            increment_rows.append(
                {
                    "time_index": index,
                    **_scalar_comparison(low_increment, high_increment, policy),
                }
            )
    any_increment_unstable = any(
        row["status"] != VALUE_STABLE for row in increment_rows
    )
    return {
        **total,
        "increment_comparisons": tuple(increment_rows),
        "cancellation_warning": bool(total.get("stable") and any_increment_unstable),
        "policy": policy.manifest_payload(),
    }


def assert_frozen_scope_equal(
    low_scope: Mapping[str, Any],
    high_scope: Mapping[str, Any],
) -> None:
    """Reject non-capacity drift using stable JSON equality."""

    low_payload = json.dumps(low_scope, sort_keys=True, separators=(",", ":"))
    high_payload = json.dumps(high_scope, sort_keys=True, separators=(",", ":"))
    if low_payload != high_payload:
        raise ValueError("frozen scope differs across capacity cells")


def nominate_capacity(
    cells: Mapping[tuple[int, int], Mapping[str, Any]],
    *,
    degrees: Sequence[int],
    ranks: Sequence[int],
    policy: SignificantPlacePolicy = SignificantPlacePolicy(),
) -> dict[str, Any]:
    """Nominate the least-cost cell stable to degree and rank refinement."""

    degree_order = tuple(int(value) for value in degrees)
    rank_order = tuple(int(value) for value in ranks)
    if tuple(sorted(set(degree_order))) != degree_order:
        raise ValueError("degrees must be unique and increasing")
    if tuple(sorted(set(rank_order))) != rank_order:
        raise ValueError("ranks must be unique and increasing")

    degree_axis: dict[str, dict[str, Any]] = {}
    for degree, higher_degree in zip(degree_order[:-1], degree_order[1:]):
        for rank in rank_order:
            key = (degree, rank)
            higher_key = (higher_degree, rank)
            if key not in cells or higher_key not in cells:
                continue
            low = cells[key]
            high = cells[higher_key]
            if not all(bool(cell.get("invariant_pass", False)) for cell in (low, high)):
                continue
            try:
                assert_frozen_scope_equal(low["frozen_scope"], high["frozen_scope"])
            except (KeyError, TypeError, ValueError):
                continue
            degree_axis[f"d{degree}_to_d{higher_degree}_r{rank}"] = {
                "low": {"degree": degree, "rank": rank},
                "high": {"degree": higher_degree, "rank": rank},
                "comparison": compare_likelihood_values(
                    low_value=low["value"],
                    high_value=high["value"],
                    low_increments=low.get("increments"),
                    high_increments=high.get("increments"),
                    policy=policy,
                ),
            }

    rank_axis: dict[str, dict[str, Any]] = {}
    for degree in degree_order:
        for rank, higher_rank in zip(rank_order[:-1], rank_order[1:]):
            key = (degree, rank)
            higher_key = (degree, higher_rank)
            if key not in cells or higher_key not in cells:
                continue
            low = cells[key]
            high = cells[higher_key]
            if not all(bool(cell.get("invariant_pass", False)) for cell in (low, high)):
                continue
            try:
                assert_frozen_scope_equal(low["frozen_scope"], high["frozen_scope"])
            except (KeyError, TypeError, ValueError):
                continue
            rank_axis[f"d{degree}_r{rank}_to_r{higher_rank}"] = {
                "low": {"degree": degree, "rank": rank},
                "high": {"degree": degree, "rank": higher_rank},
                "comparison": compare_likelihood_values(
                    low_value=low["value"],
                    high_value=high["value"],
                    low_increments=low.get("increments"),
                    high_increments=high.get("increments"),
                    policy=policy,
                ),
            }

    comparisons: dict[str, dict[str, Any]] = {}
    nominees: list[dict[str, Any]] = []
    for degree_index, degree in enumerate(degree_order[:-1]):
        higher_degree = degree_order[degree_index + 1]
        for rank_index, rank in enumerate(rank_order[:-1]):
            higher_rank = rank_order[rank_index + 1]
            degree_axis_key = f"d{degree}_to_d{higher_degree}_r{rank}"
            rank_axis_key = f"d{degree}_r{rank}_to_r{higher_rank}"
            if degree_axis_key not in degree_axis or rank_axis_key not in rank_axis:
                continue
            degree_comparison = degree_axis[degree_axis_key]["comparison"]
            rank_comparison = rank_axis[rank_axis_key]["comparison"]
            comparison_key = f"d{degree}_r{rank}"
            comparisons[comparison_key] = {
                "base": {"degree": degree, "rank": rank},
                "degree_neighbor": {"degree": higher_degree, "rank": rank},
                "rank_neighbor": {"degree": degree, "rank": higher_rank},
                "degree_comparison": degree_comparison,
                "rank_comparison": rank_comparison,
            }
            if degree_comparison["stable"] and rank_comparison["stable"]:
                nominees.append(
                    {
                        "degree": degree,
                        "rank": rank,
                        "degree_neighbor": higher_degree,
                        "rank_neighbor": higher_rank,
                        "coefficient_count": 2 * (degree + 1) * rank,
                        "comparison_key": comparison_key,
                    }
                )
    nominees.sort(
        key=lambda row: (row["coefficient_count"], row["degree"], row["rank"])
    )
    degree_rows = tuple(row["comparison"] for row in degree_axis.values())
    rank_rows = tuple(row["comparison"] for row in rank_axis.values())
    return {
        "status": "nominated" if nominees else "no_nominee",
        "nominee": nominees[0] if nominees else None,
        "all_nominees": tuple(nominees),
        "comparisons": comparisons,
        "axis_comparisons": {
            "degree": degree_axis,
            "rank": rank_axis,
        },
        "axis_summary": {
            "degree_comparison_count": len(degree_rows),
            "degree_stable_count": sum(bool(row["stable"]) for row in degree_rows),
            "degree_all_stable": bool(degree_rows)
            and all(bool(row["stable"]) for row in degree_rows),
            "rank_comparison_count": len(rank_rows),
            "rank_stable_count": sum(bool(row["stable"]) for row in rank_rows),
            "rank_all_stable": bool(rank_rows)
            and all(bool(row["stable"]) for row in rank_rows),
        },
        "policy": policy.manifest_payload(),
    }


__all__ = [
    "INVALID_INPUT",
    "VALUE_STABLE",
    "VALUE_UNSTABLE",
    "SignificantPlacePolicy",
    "assert_frozen_scope_equal",
    "compare_likelihood_values",
    "nominate_capacity",
    "significant_place",
]
