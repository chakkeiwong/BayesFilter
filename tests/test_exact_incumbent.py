from __future__ import annotations

import numpy as np

from bayesfilter.inference._exact_incumbent import (
    ExactCandidate,
    select_exact_incumbent,
)


def _candidate(
    position: float,
    value: float,
    *,
    index: int,
    source: str,
    eligible: bool = True,
) -> ExactCandidate:
    return ExactCandidate(
        position=np.array([position]),
        value=value,
        score=np.array([-position]),
        evaluation_index=index,
        source_role=source,
        eligible=eligible,
        canonical_replay=False,
    )


def test_exact_incumbent_excludes_nonfinite_and_typed_invalid_candidates() -> None:
    selected = select_exact_incumbent(
        (
            _candidate(0.0, -1.0, index=0, source="initial"),
            _candidate(1.0, np.inf, index=1, source="nonfinite"),
            _candidate(2.0, 10.0, index=2, source="finite_sentinel", eligible=False),
            _candidate(0.5, -0.25, index=3, source="design"),
        )
    )

    assert selected is not None
    assert selected.value == -0.25
    assert selected.source_role == "design"
    np.testing.assert_array_equal(selected.position, np.array([0.5]))


def test_exact_incumbent_keeps_earliest_candidate_on_exact_tie() -> None:
    selected = select_exact_incumbent(
        (
            _candidate(0.25, 1.0, index=4, source="pilot"),
            _candidate(0.50, 1.0, index=5, source="surrogate"),
        )
    )

    assert selected is not None
    assert selected.evaluation_index == 4
    assert selected.source_role == "pilot"
