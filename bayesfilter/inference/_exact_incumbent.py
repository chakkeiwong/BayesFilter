"""Internal exact-target incumbent records.

An incumbent is selected only from exact target evaluations. Surrogate model
predictions are never eligible. Exact ties retain the earlier evaluation so a
later replay cannot silently rewrite provenance.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExactCandidate:
    """One exact value/score evaluation and its eligibility metadata."""

    position: np.ndarray
    value: float
    score: np.ndarray
    evaluation_index: int
    source_role: str
    eligible: bool = True
    canonical_replay: bool = False

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float).reshape([-1]).copy()
        score = np.asarray(self.score, dtype=float).reshape([-1]).copy()
        if position.shape != score.shape:
            raise ValueError("position and score must have the same vector shape")
        position.setflags(write=False)
        score.setflags(write=False)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "evaluation_index", int(self.evaluation_index))
        object.__setattr__(self, "source_role", str(self.source_role))
        object.__setattr__(self, "eligible", bool(self.eligible))
        object.__setattr__(self, "canonical_replay", bool(self.canonical_replay))

    @property
    def strict_finite_eligible(self) -> bool:
        """Whether this record can compete for the exact incumbent."""

        return bool(
            self.eligible
            and np.isfinite(self.value)
            and np.all(np.isfinite(self.position))
            and np.all(np.isfinite(self.score))
        )


def select_exact_incumbent(
    candidates: Iterable[ExactCandidate],
) -> ExactCandidate | None:
    """Return the earliest highest-value strict-finite eligible candidate."""

    incumbent: ExactCandidate | None = None
    for candidate in candidates:
        if not candidate.strict_finite_eligible:
            continue
        if incumbent is None or candidate.value > incumbent.value:
            incumbent = candidate
    return incumbent


def candidates_from_rows(
    positions: np.ndarray,
    values: np.ndarray,
    scores: np.ndarray,
    *,
    start_index: int,
    source_role: str,
    eligibility: np.ndarray | None = None,
) -> tuple[ExactCandidate, ...]:
    """Build deterministic candidate records from one exact batched evaluation."""

    positions_np = np.asarray(positions, dtype=float)
    values_np = np.asarray(values, dtype=float).reshape([-1])
    scores_np = np.asarray(scores, dtype=float)
    if positions_np.ndim != 2 or scores_np.shape != positions_np.shape:
        raise ValueError("positions and scores must have shape [batch, dimension]")
    if values_np.shape != (positions_np.shape[0],):
        raise ValueError("values must have shape [batch]")
    if eligibility is None:
        eligible_np = np.ones(positions_np.shape[0], dtype=bool)
    else:
        eligible_np = np.asarray(eligibility, dtype=bool).reshape([-1])
        if eligible_np.shape != (positions_np.shape[0],):
            raise ValueError("eligibility must have shape [batch]")
    return tuple(
        ExactCandidate(
            position=position,
            value=value,
            score=score,
            evaluation_index=int(start_index) + row,
            source_role=source_role,
            eligible=bool(eligible),
        )
        for row, (position, value, score, eligible) in enumerate(
            zip(positions_np, values_np, scores_np, eligible_np, strict=True)
        )
    )
