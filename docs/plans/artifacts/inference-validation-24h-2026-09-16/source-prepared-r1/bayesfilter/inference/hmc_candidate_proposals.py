"""Pure finite proposal rules; no unmeasured epsilon receives qualification."""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class DirectionalEpsilonEvidence:
    epsilon: float
    decision: str
    candidate_id: str


def unvisited_interior(low: float, high: float, visited: Iterable[float]) -> float | None:
    """Bisect the largest unvisited log interval, breaking ties by lower bound."""
    points = sorted({low, high, *(value for value in visited if low < value < high)})
    gaps = sorted(zip(points[:-1], points[1:]),
                  key=lambda pair: (-math.log(pair[1] / pair[0]), pair[0]))
    for left, right in gaps:
        proposal = math.exp((math.log(left) + math.log(right)) / 2)
        if left < proposal < right:
            return proposal
    return None


def directional_proposal(*, epsilon: float, direction: str, evidence,
                         visited, domain, factor: float) -> float | None:
    """Use nearby opposing evidence before another multiplicative exploration."""
    higher = direction == "repair_step_higher"
    opposite = "repair_step_lower" if higher else "repair_step_higher"
    peers = [row for row in evidence if row.decision == opposite
             and (row.epsilon > epsilon if higher else row.epsilon < epsilon)]
    if peers:
        endpoint = min(peers, key=lambda row: abs(math.log(row.epsilon / epsilon))).epsilon
        return unvisited_interior(min(epsilon, endpoint), max(epsilon, endpoint), visited)
    low, high = domain
    proposed = min(high, max(low, epsilon * (factor if higher else 1 / factor)))
    if proposed == epsilon:
        return None
    if proposed in visited:
        return unvisited_interior(min(epsilon, proposed), max(epsilon, proposed), visited)
    return proposed


def unresolved_interiors(evidence, visited):
    """Nominate each adjacent directional reversal, including nonmonotone ones."""
    rows = sorted(evidence, key=lambda row: row.epsilon)
    for left, right in zip(rows[:-1], rows[1:]):
        if left.decision == right.decision:
            continue
        proposal = unvisited_interior(left.epsilon, right.epsilon, visited)
        if proposal is not None:
            yield proposal, (left.candidate_id, right.candidate_id)
