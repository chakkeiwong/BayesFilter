"""Pure policy primitives for canonical ordinary HMC trajectory selection.

This module has no TensorFlow, TensorFlow Probability, NumPy, or inference
imports.  The canonical tuner and diagnostic compatibility helpers import the
same values so the reviewed broad grid and midpoint rule cannot drift.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence


ORDINARY_BROAD_FIXED_METRIC_POLICY_ID = (
    "ordinary_broad_fixed_metric_selection_v1"
)
ORDINARY_BROAD_PRIMARY_L_GRID = (3, 5, 9, 13, 18, 25)
ORDINARY_BROAD_MIN_LEAPFROG_STEPS = 3
ORDINARY_BROAD_MAX_LEAPFROG_STEPS = 25


def _strict_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def midpoint_refinement_l_values(
    initial_grid: Sequence[int],
    survivor_l_values: Sequence[int],
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, ...]:
    """Return untested integer midpoints adjacent to every survivor.

    Both floor and ceiling midpoints are included when an interval has an odd
    width.  This is the only implementation of the midpoint rule used by the
    ordinary canonical and fixed-metric diagnostic routes.
    """

    lower_bound = _strict_integer(minimum, name="minimum")
    upper_bound = _strict_integer(maximum, name="maximum")
    if lower_bound <= 0 or upper_bound < lower_bound:
        raise ValueError("midpoint bounds must be positive and ordered")
    grid = tuple(
        sorted(_strict_integer(item, name="initial_grid item") for item in initial_grid)
    )
    survivors = {
        _strict_integer(item, name="survivor_l_values item")
        for item in survivor_l_values
    }
    if len(grid) < 3 or len(set(grid)) != len(grid):
        raise ValueError("refinement requires at least three distinct grid values")
    if any(item < lower_bound or item > upper_bound for item in grid):
        raise ValueError("refinement grid values exceed the declared bounds")
    if not survivors.issubset(grid):
        raise ValueError("refinement survivors must come from the initial grid")

    additions: set[int] = set()
    for survivor in survivors:
        index = grid.index(survivor)
        adjacent_intervals: list[tuple[int, int]] = []
        if index > 0:
            adjacent_intervals.append((grid[index - 1], survivor))
        if index + 1 < len(grid):
            adjacent_intervals.append((survivor, grid[index + 1]))
        for lower, upper in adjacent_intervals:
            additions.update(
                midpoint
                for midpoint in (
                    (lower + upper) // 2,
                    (lower + upper + 1) // 2,
                )
                if lower_bound <= midpoint <= upper_bound and midpoint not in grid
            )
    return tuple(sorted(additions))


def ordinary_broad_refinement_l_values(
    survivor_l_values: Sequence[int],
) -> tuple[int, ...]:
    """Apply the canonical one-round refinement to primary-grid survivors."""

    return midpoint_refinement_l_values(
        ORDINARY_BROAD_PRIMARY_L_GRID,
        survivor_l_values,
        minimum=ORDINARY_BROAD_MIN_LEAPFROG_STEPS,
        maximum=ORDINARY_BROAD_MAX_LEAPFROG_STEPS,
    )


ORDINARY_BROAD_MAX_REFINEMENT_L_GRID = ordinary_broad_refinement_l_values(
    ORDINARY_BROAD_PRIMARY_L_GRID
)
ORDINARY_BROAD_MAX_CANDIDATE_COUNT = (
    len(ORDINARY_BROAD_PRIMARY_L_GRID) + len(ORDINARY_BROAD_MAX_REFINEMENT_L_GRID)
)


__all__ = [
    "ORDINARY_BROAD_MAX_CANDIDATE_COUNT",
    "ORDINARY_BROAD_FIXED_METRIC_POLICY_ID",
    "ORDINARY_BROAD_MAX_LEAPFROG_STEPS",
    "ORDINARY_BROAD_MAX_REFINEMENT_L_GRID",
    "ORDINARY_BROAD_MIN_LEAPFROG_STEPS",
    "ORDINARY_BROAD_PRIMARY_L_GRID",
    "midpoint_refinement_l_values",
    "ordinary_broad_refinement_l_values",
]
