"""Compatibility batch adapter over the canonical single-cloud score lane.

This entry point preserves the historical ``NonlinearScoreModel`` API. Rows are
executed by TensorFlow control flow rather than a Python loop, so graph size no
longer scales by duplicating the finite program once per batch row. The fused
``PerPointScoreModel`` lane remains the NeuTra-eligible batch backend.

NO autodiff (C-9): every row uses the canonical analytical recursion.
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
    """Evaluate the single-cloud authority for each theta row.

    ``theta`` has shape ``[B,1]`` and frozen simulation inputs are shared across
    rows. ``tf.map_fn`` traces one row program and executes it as TensorFlow
    control flow; it is a compatibility adapter, not the fused training lane.
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
    parameter_count = theta.shape[1]
    if parameter_count is None:
        raise ValueError("theta parameter count must be statically known")
    if int(parameter_count) != 1:
        raise ValueError(
            "compatibility batch lane supports one parameter direction; "
            "use canonical_batch_fused_value_score for explicit directions"
        )

    dtype = theta.dtype

    def evaluate_row(row_theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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
        valid = tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score))
        return value, score, valid

    value_batch, score_batch, valid_batch = tf.map_fn(
        evaluate_row,
        theta,
        fn_output_signature=(
            tf.TensorSpec([], dtype),
            tf.TensorSpec([1], dtype),
            tf.TensorSpec([], tf.bool),
        ),
        parallel_iterations=1,
    )
    nan = tf.cast(float("nan"), dtype)
    return (
        tf.where(valid_batch, value_batch, nan),
        tf.where(valid_batch[:, None], score_batch, nan),
        {"program_valid": valid_batch},
    )


__all__ = ["canonical_batch_value_score"]
