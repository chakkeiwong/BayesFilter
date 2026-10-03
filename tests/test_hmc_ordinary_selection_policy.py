from __future__ import annotations

import pytest

import bayesfilter
from bayesfilter.hmc_ordinary_selection_policy import (
    ORDINARY_BROAD_FIXED_METRIC_POLICY_ID,
    ORDINARY_BROAD_PRIMARY_L_GRID,
    midpoint_refinement_l_values,
    ordinary_broad_refinement_l_values,
)
from bayesfilter.inference.hmc_fixed_metric_grid_search import (
    DEFAULT_L_GRID,
    refinement_l_values,
)


def test_ordinary_primary_grid_has_one_public_definition() -> None:
    assert ORDINARY_BROAD_FIXED_METRIC_POLICY_ID == (
        "ordinary_broad_fixed_metric_selection_v1"
    )
    assert ORDINARY_BROAD_PRIMARY_L_GRID == (3, 5, 9, 13, 18, 25)
    assert DEFAULT_L_GRID == ORDINARY_BROAD_PRIMARY_L_GRID
    assert bayesfilter.ORDINARY_BROAD_FIXED_METRIC_POLICY_ID == (
        ORDINARY_BROAD_FIXED_METRIC_POLICY_ID
    )
    assert bayesfilter.ORDINARY_BROAD_PRIMARY_L_GRID == (
        ORDINARY_BROAD_PRIMARY_L_GRID
    )


@pytest.mark.parametrize(
    ("survivors", "expected"),
    [
        ((), ()),
        ((3,), (4,)),
        ((5,), (4, 7)),
        ((9,), (7, 11)),
        ((13,), (11, 15, 16)),
        ((25,), (21, 22)),
        ((5, 13), (4, 7, 11, 15, 16)),
        ((3, 5, 9, 13, 18, 25), (4, 7, 11, 15, 16, 21, 22)),
    ],
)
def test_ordinary_refinement_is_adjacent_to_every_primary_survivor(
    survivors: tuple[int, ...],
    expected: tuple[int, ...],
) -> None:
    assert ordinary_broad_refinement_l_values(survivors) == expected
    assert refinement_l_values(ORDINARY_BROAD_PRIMARY_L_GRID, survivors) == expected


def test_midpoint_policy_rejects_nonprimary_survivors_and_invalid_grids() -> None:
    with pytest.raises(ValueError, match="survivors must come from"):
        ordinary_broad_refinement_l_values((4,))
    with pytest.raises(ValueError, match="distinct grid"):
        midpoint_refinement_l_values(
            (3, 3, 5),
            (3,),
            minimum=3,
            maximum=25,
        )
    with pytest.raises(ValueError, match="must be an integer"):
        midpoint_refinement_l_values(
            (3, 5, 9),
            (True,),
            minimum=3,
            maximum=25,
        )
