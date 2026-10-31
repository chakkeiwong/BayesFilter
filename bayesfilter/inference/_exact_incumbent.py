"""Internal exact-target incumbent records.

An incumbent is selected only from exact target evaluations. Surrogate model
predictions are never eligible. Exact ties retain the earlier evaluation so a
later replay cannot silently rewrite provenance.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import accumulate

import tensorflow as tf

from bayesfilter.ops.host_tensor_io import numeric_tensor


@tf.function(input_signature=[
    tf.TensorSpec([None], tf.float64), tf.TensorSpec([None], tf.float64),
    tf.TensorSpec([None], tf.float64), tf.TensorSpec([None], tf.bool),
    tf.TensorSpec([None], tf.int32),
], jit_compile=True, autograph=False)
def _incumbent_selection(positions, scores, values, eligible, row_ends):
    """Finite eligibility and earliest maximum for packed, possibly empty rows.

    Only the ragged record schema is prepared on the host. Every numerical
    predicate and selection executes here; empty vectors remain vacuously
    finite, as in the original selector. XLA specializes physical input sizes,
    while the TensorFlow tracing signature is shared across record counts.
    """
    rows = tf.searchsorted(row_ends, tf.range(tf.size(positions)), side="right")
    # Public packing pads coordinates with finite zeros. Assign this padding
    # to the final segment so it cannot create an out-of-range segment ID.
    rows = tf.minimum(rows, tf.size(values) - 1)
    finite_coordinates = tf.math.is_finite(positions) & tf.math.is_finite(scores)
    coordinate_eligible = tf.math.unsorted_segment_min(
        tf.cast(finite_coordinates, tf.int32), rows, tf.size(values)) > 0
    mask = eligible & tf.math.is_finite(values) & coordinate_eligible
    masked_values = tf.where(mask, values, tf.constant(float("-inf"), tf.float64))
    # Include a sentinel so the tensor endpoint also accepts no records. A
    # finite eligible record always wins, and argmax retains the first tie.
    with_sentinel = tf.concat([masked_values, tf.constant([float("-inf")], tf.float64)], 0)
    index = tf.where(tf.reduce_any(mask), tf.argmax(with_sentinel, output_type=tf.int32), -1)
    return mask, index


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
        position = numeric_tensor(self.position, dtype=tf.float64)
        score = numeric_tensor(self.score, dtype=tf.float64)
        if position.shape.rank != 1:
            position = tf.reshape(position, [-1])
        if score.shape.rank != 1:
            score = tf.reshape(score, [-1])
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

        mask, _ = _select_records((self,))
        return bool(mask[0])


def _select_records(records):
    """Pack immutable fields into geometric capacities, then select in XLA."""
    # These comprehensions pack fields and vector-width metadata. Powers of
    # two prevent every appended candidate from compiling a new executable.
    positions = tf.concat([candidate.position for candidate in records], 0)
    scores = tf.concat([candidate.score for candidate in records], 0)
    ends = tuple(accumulate(candidate.position.shape[0] for candidate in records))
    record_capacity = 1 << (len(records) - 1).bit_length()
    coordinate_capacity = 1 << (max(1, ends[-1]) - 1).bit_length()
    record_padding = record_capacity - len(records)
    coordinate_padding = coordinate_capacity - ends[-1]
    row_ends = tf.constant((*ends, *((ends[-1],) * record_padding)), tf.int32)
    eligible = tf.constant([candidate.eligible for candidate in records] + [False] * record_padding)
    values = tf.constant([candidate.value for candidate in records] + [float("-inf")] * record_padding, tf.float64)
    return _incumbent_selection(tf.pad(positions, [[0, coordinate_padding]]),
        tf.pad(scores, [[0, coordinate_padding]]), values, eligible, row_ends)


def select_exact_incumbent(
    candidates: Iterable[ExactCandidate],
) -> ExactCandidate | None:
    """Return the earliest highest-value strict-finite eligible candidate."""

    records = tuple(candidates)
    if not records:
        return None
    _, index = _select_records(records)
    selected = int(index)
    return None if selected < 0 else records[selected]


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
    # Materialize completed scalar fields once at the public record boundary.
    # Iterating tensors here launches one slice and host synchronization per
    # scalar. Bulk unstack/serialization only transports the completed values;
    # eligibility and ranking still execute in the compiled selector.
    position_rows = tf.unstack(positions_np, axis=0)
    score_rows = tf.unstack(scores_np, axis=0)
    host_values = values_np.numpy().tolist()
    host_eligibility = eligible_np.numpy().tolist()
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
            zip(position_rows, host_values, score_rows, host_eligibility, strict=True)
        )
    )
