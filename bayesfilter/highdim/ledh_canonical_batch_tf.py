"""Canonical LEDH batch lane (P5): batch-native value + analytical score.

Registered claim-bearing batch entry point. Batch semantics per the NeuTra
batch-native rule: the leading theta-batch dimension is preserved through
transport, target evaluation, and score computation. Implementation
strategy (recorded): the per-row program IS the single-cloud canonical
program — the batch lane maps the gated single-cloud implementation over
theta rows with identical arithmetic, which guarantees batch-size-1 parity
by construction and keeps ONE semantic authority (anti-fork rule).

NEUTRA ELIGIBILITY (honest limitation, AGENTS.md batch-native rule): a
Python row loop is a PARITY/REFERENCE implementation and is NOT eligible
as a NeuTra TRAINING backend — the batching rule forbids row-mapped scalar
targets for optimizer updates. Before the P7 NeuTra rebind, a fused
batch-tensor implementation must land and pass these same parity gates;
until then this entry point is claim-bearing for value/score evaluation
and parity, not for NeuTra training. Conformance G-3 tracks this cell.

NO autodiff (C-9): the score is the analytical recursion of
``ledh_canonical_score_tf``.
"""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel,
    canonical_value_and_analytical_score,
)

Tensor = tf.Tensor


def canonical_batch_value_score(
    model: NonlinearScoreModel,
    theta: Tensor,
    initial_states: Tensor,
    initial_covariances: Tensor,
    noises: Tensor,
    observations: Tensor,
    *,
    substeps: int | None = None,
    flow_substeps: int | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Batched value/score: theta [B, P] -> value [B], score [B, P].

    Frozen inputs (initial cloud, noises, observations) are shared across
    rows, matching the NeuTra target contract.  ``substeps`` is the historical
    batch-lane spelling; ``flow_substeps`` is accepted as the authority's
    spelling.  Supplying both is allowed only when they agree.
    """

    if substeps is None:
        substeps = flow_substeps
    elif flow_substeps is not None and int(substeps) != int(flow_substeps):
        raise ValueError("substeps and flow_substeps must agree")
    if substeps is None:
        raise TypeError("one of substeps or flow_substeps is required")
    substeps = int(substeps)
    if substeps < 1:
        raise ValueError("substeps must be positive")

    theta = tf.convert_to_tensor(theta)
    if theta.shape.rank != 2:
        raise ValueError("canonical batch lane requires theta of rank 2")
    batch_size = int(theta.shape[0])
    parameter_count = int(theta.shape[1])
    if parameter_count != 1:
        raise ValueError(
            "P5 slice supports one parameter direction per call; "
            "multi-parameter models loop directions (as the score "
            "definition prescribes)"
        )

    values = []
    scores = []
    valids = []
    for row in range(batch_size):
        row_theta = theta[row]
        value, score = canonical_value_and_analytical_score(
            model,
            row_theta,
            initial_states,
            initial_covariances,
            noises,
            observations,
            flow_substeps=substeps,
            with_score=True,
        )
        values.append(value)
        scores.append(score)
        valids.append(
            tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score))
        )
    value_batch = tf.stack(values)
    score_batch = tf.stack(scores)
    valid_batch = tf.stack(valids)
    nan = tf.cast(float("nan"), value_batch.dtype)
    return (
        tf.where(valid_batch, value_batch, nan),
        tf.where(valid_batch[:, None], score_batch, nan),
        {"program_valid": valid_batch},
    )


__all__ = ["canonical_batch_value_score"]
