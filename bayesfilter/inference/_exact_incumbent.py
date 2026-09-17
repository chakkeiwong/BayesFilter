"""Internal exact-target incumbent records.

An incumbent is selected only from exact target evaluations. Surrogate model
predictions are never eligible. Exact ties retain the earlier evaluation so a
later replay cannot silently rewrite provenance.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import math

import tensorflow as tf

from bayesfilter.ops.host_tensor_io import numeric_tensor


@dataclass(frozen=True)
class ExactCandidate:
    """One exact value/score evaluation and its eligibility metadata."""

    position: tf.Tensor
    value: float
    score: tf.Tensor
    evaluation_index: int
    source_role: str
    eligible: bool = True
    canonical_replay: bool = False

    def __post_init__(self) -> None:
        position = tf.reshape(numeric_tensor(self.position, dtype=tf.float64), [-1])
        score = tf.reshape(numeric_tensor(self.score, dtype=tf.float64), [-1])
        if position.shape != score.shape:
            raise ValueError("position and score must have the same vector shape")
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
            and math.isfinite(self.value)
            and tf.reduce_all(tf.math.is_finite(self.position))
            and tf.reduce_all(tf.math.is_finite(self.score))
        )


def select_exact_incumbent(
    candidates: Iterable[ExactCandidate],
) -> ExactCandidate | None:
    """Return the earliest highest-value strict-finite eligible candidate."""

    records = tuple(candidates)
    if not records:
        return None
    eligible = tf.constant([candidate.strict_finite_eligible for candidate in records])
    if not bool(tf.reduce_any(eligible)):
        return None
    values = tf.constant([candidate.value for candidate in records], tf.float64)
    eligible_values = tf.where(eligible, values, tf.constant(float("-inf"), tf.float64))
    # TensorFlow argmax retains the first exact tie, including signed zero.
    return records[int(tf.argmax(eligible_values))]


def candidates_from_rows(
    positions: tf.Tensor,
    values: tf.Tensor,
    scores: tf.Tensor,
    *,
    start_index: int,
    source_role: str,
    eligibility: tf.Tensor | None = None,
) -> tuple[ExactCandidate, ...]:
    """Build deterministic candidate records from one exact batched evaluation."""

    positions_np = numeric_tensor(positions, dtype=tf.float64)
    values_np = tf.reshape(numeric_tensor(values, dtype=tf.float64), [-1])
    scores_np = numeric_tensor(scores, dtype=tf.float64)
    if positions_np.shape.rank != 2 or scores_np.shape != positions_np.shape:
        raise ValueError("positions and scores must have shape [batch, dimension]")
    if values_np.shape != (positions_np.shape[0],):
        raise ValueError("values must have shape [batch]")
    if eligibility is None:
        eligible_np = tf.ones(positions_np.shape[0], dtype=tf.bool)
    else:
        eligible_np = tf.reshape(numeric_tensor(eligibility, dtype=tf.bool), [-1])
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
