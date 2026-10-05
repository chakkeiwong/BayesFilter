"""Opt-in, fixed-batch TF/TFP location diagnostic with exact incumbent replay.

Each independent start uses theta = anchor + scale * R*tanh(u/R).
The minimized objective is -ell(theta), with gradient
-(scale * score_theta) * (1-tanh(u/R)**2). This optimizer chart contributes
no density Jacobian. Native batched TFP L-BFGS implements the search; bounded
re-anchoring is not a quadratic trust-region method or a MAP certificate.

Search, stopping, selection, and replay decisions stay in TensorFlow. Host
materialization occurs only in result reporting. No covariance, transport,
training, HMC, or posterior-convergence claim follows from an accepted center.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

BATCHED_LOCAL_CENTER_NONCLAIMS = (
    "bounded location diagnostic, not certified local/global MAP",
    "not posterior support or mode-coverage evidence",
    "not Hessian, covariance, or mass-matrix evidence",
    "not NeuTra training, whitening, or HMC readiness",
    "not GPU/default readiness, convergence, or identification evidence",
)

_STATUSES = (
    "initial_target_invalid",
    "callback_cap_exhausted",
    "replay_mismatch",
    "selected_optimizer_failed",
    "localized",
    "stationary",
)


@dataclass(frozen=True)
class BatchedLocalCenterConfig:
    """Frozen search limits and componentwise deterministic replay tolerances.

    The box radius is per coordinate in caller-supplied diagonal scale units,
    not a posterior standard deviation. Re-anchoring does not estimate that scale.
    The callback cap excludes one initial batch and endpoint/final replays.
    """

    box_radius: float = 4.0
    trust_refinement_rounds: int = 2
    num_correction_pairs: int = 10
    max_iterations: int = 50
    max_line_search_iterations: int = 20
    gradient_tolerance: float = 1e-8
    max_optimizer_callback_batches_per_round: int = 64
    replay_atol: float = 1e-10
    replay_rtol: float = 1e-10
    movement_tolerance: float = 1e-12
    jit_compile: bool = True
    stop_when_no_row_moves: bool = True

    def __post_init__(self) -> None:
        for name in (
            "box_radius",
            "gradient_tolerance",
            "replay_atol",
            "replay_rtol",
            "movement_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "trust_refinement_rounds",
            "num_correction_pairs",
            "max_iterations",
            "max_line_search_iterations",
            "max_optimizer_callback_batches_per_round",
        ):
            if operator.index(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def maximum_physical_rows_multiplier(self) -> int:
        return 2 + self.trust_refinement_rounds * (
            self.max_optimizer_callback_batches_per_round + 1
        )

    def payload(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "maximum_physical_rows_multiplier": self.maximum_physical_rows_multiplier,
        }


@dataclass(frozen=True)
class BatchedLocalCenterResult:
    """Tensor decisions and sufficient per-start exact-maxima ledger.

    Evaluation indices are zero-based flattened target-call/row indices. Strict
    improvements preserve earliest ties. Validation replays never replace records.
    An accepted nonstationary result is only a finite, reproducible location.
    """

    status_code: tf.Tensor
    accepted: tf.Tensor
    center: tf.Tensor
    center_value: tf.Tensor
    center_score: tf.Tensor
    selected_evaluation_index: tf.Tensor
    valid_rows: tf.Tensor
    best_values: tf.Tensor
    best_positions: tf.Tensor
    best_scores: tf.Tensor
    best_evaluation_indices: tf.Tensor
    endpoint_positions: tf.Tensor
    endpoint_values: tf.Tensor
    endpoint_valid: tf.Tensor
    endpoint_accepted: tf.Tensor
    optimizer_converged: tf.Tensor
    optimizer_failed: tf.Tensor
    chart_saturated: tf.Tensor
    rounds_completed: tf.Tensor
    target_callback_batches: tf.Tensor
    physical_target_rows: tf.Tensor
    optimizer_callback_attempts: tf.Tensor
    optimizer_target_batches: tf.Tensor
    replay_batches: tf.Tensor
    invalid_target_rows: tf.Tensor
    cap_exhausted: tf.Tensor
    replay_consistent: tf.Tensor
    trace_count: int
    nonclaims: tuple[str, ...] = BATCHED_LOCAL_CENTER_NONCLAIMS

    @property
    def status(self) -> str:
        return _STATUSES[int(self.status_code.numpy())]

    def payload(self) -> dict[str, Any]:
        """Materialize reporting only; absent/nonfinite numbers become null."""

        def convert(value: Any) -> Any:
            if tf.is_tensor(value):
                return convert(value.numpy().tolist())
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value

        return {
            "schema": "bayesfilter.batched_local_center.result.v1",
            "status": self.status,
            **{
                field.name: convert(getattr(self, field.name)) for field in fields(self)
            },
        }


def _bounded_chart(
    anchor: tf.Tensor,
    scale: tf.Tensor,
    unconstrained: tf.Tensor,
    radius: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return raw positions and diagonal d(theta)/du, with R cancellation."""
    tangent = tf.math.tanh(unconstrained / radius)
    return anchor + scale * (radius * tangent), scale * (1 - tf.square(tangent))


def locate_batched_local_center(
    batched_value_score_eligibility_fn: Callable[
        [tf.Tensor], tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    ],
    initial_positions: Any,
    scale: Any,
    *,
    config: BatchedLocalCenterConfig | None = None,
) -> BatchedLocalCenterResult:
    """Find the highest exact eligible point from independent [B,D] starts.

    The deterministic, row-independent callback takes float64 [B,D] and returns
    float64 values [B], raw scores [B,D], and bool eligibility [B]. All calls use
    the same shape, including duplicate final replay rows (not search starts).
    Native Tensor inputs must already be float64. Scale is positive [D].

    Invalid rows receive +inf optimizer objective and zero gradient but cannot
    be reported stationary. Exceptions propagate: callers must record the failure,
    not substitute another target or silently retry. Default JIT applies to the
    entire search/replay program, not just its target objective.
    """
    cfg = BatchedLocalCenterConfig() if config is None else config
    initial = tf.convert_to_tensor(initial_positions, dtype=tf.float64)
    scale_tensor = tf.convert_to_tensor(scale, dtype=tf.float64)
    if initial.shape.rank != 2 or not initial.shape.is_fully_defined():
        raise ValueError("initial_positions must have static rank-2 shape [B,D]")
    batch_size, dimension = initial.shape
    if batch_size <= 0 or dimension <= 0:
        raise ValueError("initial_positions must have positive dimensions")
    if scale_tensor.shape != tf.TensorShape([dimension]):
        raise ValueError("scale must have static shape [D]")
    tf.debugging.assert_all_finite(initial, "initial_positions must be finite")
    tf.debugging.assert_all_finite(scale_tensor, "scale must be finite")
    tf.debugging.assert_positive(scale_tensor, message="scale must be positive")

    def variable(value: Any, dtype: tf.DType) -> tf.Variable:
        return tf.Variable(value, trainable=False, dtype=dtype)

    negative_infinity = tf.constant(-math.inf, tf.float64)
    budget_abort = tf.constant(math.nan, tf.float64)
    best_values = variable(tf.fill([batch_size], negative_infinity), tf.float64)
    best_positions = variable(initial, tf.float64)
    best_scores = variable(tf.zeros_like(initial), tf.float64)
    # TensorFlow pins int32 resource variables to the host even inside a GPU
    # device scope. Int64 counters/indices allow the complete XLA optimizer
    # and its resource state to remain on the caller-selected accelerator.
    index_dtype = tf.int64
    best_indices = variable(tf.fill([batch_size], tf.constant(-1, index_dtype)), index_dtype)
    calls = variable(0, index_dtype)
    attempts = variable(0, index_dtype)
    optimizer_calls = variable(0, index_dtype)
    round_calls = variable(0, index_dtype)
    replays = variable(0, index_dtype)
    invalid_rows = variable(0, index_dtype)
    capped = variable(False, tf.bool)
    mismatch = variable(False, tf.bool)
    endpoint_positions = variable(initial, tf.float64)
    endpoint_values = variable(tf.fill([batch_size], negative_infinity), tf.float64)
    endpoint_valid = variable(tf.zeros([batch_size], tf.bool), tf.bool)
    endpoint_accepted = variable(tf.zeros([batch_size], tf.bool), tf.bool)
    optimizer_converged = variable(tf.zeros([batch_size], tf.bool), tf.bool)
    optimizer_failed = variable(tf.zeros([batch_size], tf.bool), tf.bool)
    saturated = variable(tf.zeros([batch_size], tf.bool), tf.bool)

    def close(actual: tf.Tensor, expected: tf.Tensor) -> tf.Tensor:
        return (
            tf.math.is_finite(actual)
            & tf.math.is_finite(expected)
            & (
                tf.abs(actual - expected)
                <= cfg.replay_atol + cfg.replay_rtol * tf.abs(expected)
            )
        )

    def evaluate(positions: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        positions = tf.ensure_shape(positions, [batch_size, dimension])
        call_number = calls.assign_add(1)
        with tf.control_dependencies([call_number]):
            values, scores, eligible = batched_value_score_eligibility_fn(positions)
            values = tf.ensure_shape(
                tf.convert_to_tensor(values, tf.float64), [batch_size]
            )
            scores = tf.ensure_shape(
                tf.convert_to_tensor(scores, tf.float64), initial.shape
            )
            eligible = tf.ensure_shape(
                tf.convert_to_tensor(eligible, tf.bool), [batch_size]
            )
            valid = (
                eligible
                & tf.math.is_finite(values)
                & tf.reduce_all(tf.math.is_finite(scores), axis=1)
                & tf.reduce_all(tf.math.is_finite(positions), axis=1)
            )
            invalid_rows.assign_add(tf.reduce_sum(tf.cast(~valid, index_dtype)))
        return (
            tf.where(valid, values, negative_infinity),
            tf.where(valid[:, None], scores, tf.zeros_like(scores)),
            valid,
        )

    def record(
        positions: tf.Tensor, values: tf.Tensor, scores: tf.Tensor, valid: tf.Tensor
    ) -> tf.Tensor:
        improve = valid & (values > best_values.read_value())
        indices = (calls.read_value() - 1) * batch_size + tf.range(batch_size, dtype=index_dtype)
        updates = (
            best_values.assign(tf.where(improve, values, best_values)),
            best_positions.assign(
                tf.where(improve[:, None], positions, best_positions)
            ),
            best_scores.assign(tf.where(improve[:, None], scores, best_scores)),
            best_indices.assign(tf.where(improve, indices, best_indices)),
        )
        with tf.control_dependencies(updates):
            return tf.identity(values)

    def search_round(round_index, anchor, anchor_values, continuing):
        del continuing
        round_calls.assign(0)

        def objective(unconstrained):
            attempts.assign_add(1)
            positions, derivative = _bounded_chart(
                anchor,
                scale_tensor,
                unconstrained,
                cfg.box_radius,
            )

            def available():
                round_calls.assign_add(1)
                optimizer_calls.assign_add(1)
                values, scores, valid = evaluate(positions)
                values = record(positions, values, scores, valid)
                return -values, tf.where(valid[:, None], -scores * derivative, 0.0)

            def exhausted():
                """Abort Hager-Zhang with NaN, not its bisectable +inf endpoint.

                This optimizer-only failure signal never enters target records.
                The separate cap veto rejects even a finite replayed incumbent.
                Invalid support remains +inf and may still be searched around.
                """
                capped.assign(True)
                return tf.fill([batch_size], budget_abort), tf.zeros_like(initial)

            return tf.cond(
                round_calls < cfg.max_optimizer_callback_batches_per_round,
                available,
                exhausted,
            )

        optimizer = tfp.optimizer.lbfgs_minimize(
            objective,
            initial_position=tf.zeros_like(initial),
            num_correction_pairs=cfg.num_correction_pairs,
            tolerance=tf.constant(cfg.gradient_tolerance, tf.float64),
            max_iterations=cfg.max_iterations,
            max_line_search_iterations=cfg.max_line_search_iterations,
            parallel_iterations=1,
            stopping_condition=tfp.optimizer.converged_all,
        )
        positions, derivative = _bounded_chart(
            anchor,
            scale_tensor,
            optimizer.position,
            cfg.box_radius,
        )
        replays.assign_add(1)
        values, scores, valid = evaluate(positions)
        reported_valid = (
            tf.math.is_finite(optimizer.objective_value)
            & tf.reduce_all(tf.math.is_finite(optimizer.objective_gradient), axis=1)
            & tf.reduce_all(tf.math.is_finite(optimizer.position), axis=1)
        )
        consistent = (valid == reported_valid) & (
            ~valid
            | (
                close(values, -optimizer.objective_value)
                & tf.reduce_all(
                    close(-scores * derivative, optimizer.objective_gradient), axis=1
                )
            )
        )
        mismatch.assign(mismatch | tf.reduce_any(~consistent))
        failed = optimizer.failed | ~valid | ~reported_valid
        accepted = valid & consistent & ~failed & ~capped & (values >= anchor_values)
        moved = accepted & (
            tf.reduce_max(tf.abs((positions - anchor) / scale_tensor), axis=1)
            > cfg.movement_tolerance
        )
        scaled_score_norm = tf.reduce_max(tf.abs(scores * scale_tensor), axis=1)
        endpoint_positions.assign(positions)
        endpoint_values.assign(values)
        endpoint_valid.assign(valid)
        endpoint_accepted.assign(accepted)
        optimizer_failed.assign(failed)
        optimizer_converged.assign(
            optimizer.converged
            & valid
            & consistent
            & ~failed
            & ~capped
            & (scaled_score_norm <= cfg.gradient_tolerance)
        )
        saturated.assign(
            valid
            & (tf.reduce_min(tf.abs(derivative / scale_tensor), axis=1) <= 1e-8)
            & (scaled_score_norm > cfg.gradient_tolerance)
        )
        continuing = ~capped & ~mismatch
        if cfg.stop_when_no_row_moves:
            continuing = continuing & tf.reduce_any(moved)
        return (
            round_index + 1,
            tf.where(accepted[:, None], positions, anchor),
            tf.where(accepted, values, anchor_values),
            continuing,
        )

    def run():
        values, scores, valid = evaluate(initial)
        values = record(initial, values, scores, valid)
        rounds, _, _, _ = tf.while_loop(
            lambda count, anchor, anchor_values, continuing: (
                (count < cfg.trust_refinement_rounds) & continuing
            ),
            search_round,
            (tf.constant(0), initial, values, tf.reduce_any(valid)),
            parallel_iterations=1,
        )
        valid_rows = best_indices >= 0
        has_incumbent = tf.reduce_any(valid_rows)
        highest = tf.reduce_max(best_values)
        tied_indices = tf.where(
            valid_rows & (best_values == highest),
            best_indices,
            2147483647,
        )
        selected = tf.argmin(tied_indices, output_type=tf.int32)
        center = tf.gather(best_positions, selected)
        center_value = tf.gather(best_values, selected)
        center_score = tf.gather(best_scores, selected)

        def replay():
            replays.assign_add(1)
            repeated = tf.broadcast_to(center, initial.shape)
            replay_values, replay_scores, replay_valid = evaluate(repeated)
            return tf.reduce_all(replay_valid & close(replay_values, center_value)) & (
                tf.reduce_all(close(replay_scores, center_score))
            )

        consistent = (
            tf.cond(has_incumbent, replay, lambda: tf.constant(False)) & ~mismatch
        )
        selected_failed = tf.gather(optimizer_failed, selected)
        accepted = has_incumbent & consistent & ~capped & ~selected_failed
        stationary = (
            tf.reduce_max(tf.abs(center_score * scale_tensor)) <= cfg.gradient_tolerance
        )
        status = tf.where(stationary, 5, 4)
        status = tf.where(selected_failed, 3, status)
        status = tf.where(~consistent, 2, status)
        status = tf.where(capped, 1, status)
        status = tf.where(~has_incumbent, 0, status)
        return {
            "status_code": status,
            "accepted": accepted,
            "center": center,
            "center_value": center_value,
            "center_score": center_score,
            "selected_evaluation_index": tf.gather(best_indices, selected),
            "valid_rows": valid_rows,
            "best_values": best_values.read_value(),
            "best_positions": best_positions.read_value(),
            "best_scores": best_scores.read_value(),
            "best_evaluation_indices": best_indices.read_value(),
            "endpoint_positions": endpoint_positions.read_value(),
            "endpoint_values": endpoint_values.read_value(),
            "endpoint_valid": endpoint_valid.read_value(),
            "endpoint_accepted": endpoint_accepted.read_value(),
            "optimizer_converged": optimizer_converged.read_value(),
            "optimizer_failed": optimizer_failed.read_value(),
            "chart_saturated": saturated.read_value(),
            "rounds_completed": rounds,
            "target_callback_batches": calls.read_value(),
            "physical_target_rows": calls * batch_size,
            "optimizer_callback_attempts": attempts.read_value(),
            "optimizer_target_batches": optimizer_calls.read_value(),
            "replay_batches": replays.read_value(),
            "invalid_target_rows": invalid_rows.read_value(),
            "cap_exhausted": capped.read_value(),
            "replay_consistent": consistent,
        }

    compiled = tf.function(run, jit_compile=cfg.jit_compile, autograph=False)
    tensors = compiled()
    return BatchedLocalCenterResult(
        **tensors, trace_count=compiled.experimental_get_tracing_count()
    )
