"""Reusable native batched locator; same search/replay with operand-bound starts."""

import math
from threading import RLock
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.batched_local_center import _bounded_chart


class BatchedLocalCenterProgram:
    """Own one fixed callback/batch/dimension locator and its resettable resources."""

    def __init__(self, batched_value_score_eligibility_fn, batch_size, dimension, cfg):
        self.invocation_lock = RLock()
        initial = tf.zeros([batch_size, dimension], tf.float64)
        def variable(value: Any, dtype: tf.DType) -> tf.Variable:
            return tf.Variable(value, trainable=False, dtype=dtype)

        negative_infinity = tf.constant(-math.inf, tf.float64)
        budget_abort = tf.constant(math.nan, tf.float64)
        best_values = variable(tf.fill([batch_size], negative_infinity), tf.float64)
        best_positions = variable(initial, tf.float64)
        best_scores = variable(tf.zeros_like(initial), tf.float64)
        # TensorFlow places int32 resources on CPU, which GPU XLA cannot read.
        # Only accounting uses int64; optimizer and floating arithmetic stay as-is.
        best_indices = variable(tf.fill([batch_size], tf.constant(-1, tf.int64)), tf.int64)
        calls = variable(0, tf.int64)
        attempts = variable(0, tf.int64)
        optimizer_calls = variable(0, tf.int64)
        round_calls = variable(0, tf.int64)
        replays = variable(0, tf.int64)
        invalid_rows = variable(0, tf.int64)
        capped = variable(False, tf.bool)
        mismatch = variable(False, tf.bool)
        endpoint_positions = variable(initial, tf.float64)
        endpoint_values = variable(tf.fill([batch_size], negative_infinity), tf.float64)
        endpoint_valid = variable(tf.zeros([batch_size], tf.bool), tf.bool)
        endpoint_accepted = variable(tf.zeros([batch_size], tf.bool), tf.bool)
        optimizer_converged = variable(tf.zeros([batch_size], tf.bool), tf.bool)
        optimizer_failed = variable(tf.zeros([batch_size], tf.bool), tf.bool)
        saturated = variable(tf.zeros([batch_size], tf.bool), tf.bool)

        @tf.function(input_signature=[tf.TensorSpec([batch_size, dimension], tf.float64),
            tf.TensorSpec([dimension], tf.float64)], jit_compile=cfg.jit_compile, autograph=False)
        def run(initial, scale_tensor):
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
                    invalid_rows.assign_add(tf.reduce_sum(tf.cast(~valid, tf.int64)))
                return (
                    tf.where(valid, values, negative_infinity),
                    tf.where(valid[:, None], scores, tf.zeros_like(scores)),
                    valid,
                )

            def record(
                positions: tf.Tensor, values: tf.Tensor, scores: tf.Tensor, valid: tf.Tensor
            ) -> tf.Tensor:
                improve = valid & (values > best_values.read_value())
                indices = (calls.read_value() - 1) * batch_size + tf.range(batch_size, dtype=tf.int64)
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

            resets = (
                best_values.assign(tf.fill([batch_size], negative_infinity)),
                best_positions.assign(initial),
                best_scores.assign(tf.zeros_like(initial)),
                best_indices.assign(tf.fill([batch_size], tf.constant(-1, tf.int64))),
                calls.assign(0),
                attempts.assign(0),
                optimizer_calls.assign(0),
                round_calls.assign(0),
                replays.assign(0),
                invalid_rows.assign(0),
                capped.assign(False),
                mismatch.assign(False),
                endpoint_positions.assign(initial),
                endpoint_values.assign(tf.fill([batch_size], negative_infinity)),
                endpoint_valid.assign(tf.zeros([batch_size], tf.bool)),
                endpoint_accepted.assign(tf.zeros([batch_size], tf.bool)),
                optimizer_converged.assign(tf.zeros([batch_size], tf.bool)),
                optimizer_failed.assign(tf.zeros([batch_size], tf.bool)),
                saturated.assign(tf.zeros([batch_size], tf.bool)),
            )
            with tf.control_dependencies(resets):
                initial = tf.identity(initial)
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


        self.compiled = run

    def __call__(self, initial, scale):
        with self.invocation_lock:
            result = self.compiled(initial, scale)
            int(result['status_code'])
            return result
