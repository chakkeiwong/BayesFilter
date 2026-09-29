"""Candidate-selection helpers that preserve identity and tie behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CandidateResult:
    candidate_index: int
    step_size: float
    leapfrog_steps: int
    score: float
    status: str = "ok"
    payload: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_index", int(self.candidate_index))
        object.__setattr__(self, "step_size", float(self.step_size))
        object.__setattr__(self, "leapfrog_steps", int(self.leapfrog_steps))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "status", str(self.status))


def canonical_candidate_order(candidates: Iterable[CandidateResult]) -> tuple[CandidateResult, ...]:
    """Reassemble nondeterministic worker completions by candidate index."""

    return tuple(sorted(candidates, key=lambda candidate: candidate.candidate_index))


def select_first_tie_candidate(candidates: Iterable[CandidateResult]) -> CandidateResult:
    """Select the lowest score after canonical ordering, preserving first tie."""

    ordered = canonical_candidate_order(candidates)
    if not ordered:
        raise ValueError("at least one candidate is required")
    viable = [candidate for candidate in ordered if candidate.status == "ok"]
    if not viable:
        raise ValueError("at least one ok candidate is required")
    return min(viable, key=lambda candidate: candidate.score)
