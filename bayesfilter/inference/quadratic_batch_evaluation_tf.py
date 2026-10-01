"""Native fixed-batch target evaluation for quadratic-center localization.

The input capacity bounds tracing and history storage. Only active chunks call
the analytical target; the last row is duplicated to fill a partial batch.
Python assembles the public report after this complete numerical program.
"""

from __future__ import annotations

import tensorflow as tf


def make_quadratic_batch_evaluator(callback, dimension, batch_size, capacity, *, jit_compile=True):
    """Compile ordered evaluation/selection; graph mode is a diagnostic option.

    ``point_count`` is in [1, capacity], validated by the caller before padding
    into the fixed-capacity input. Physical-row counts/indices use int64;
    bounded chunk slots use int32. The
    returned histories include every callback, including a final invalid batch;
    unused history slots are zero and are not reported by the caller.
    """
    if dimension < 1 or batch_size < 2 or capacity < 1:
        raise ValueError("positive dimension/capacity and batch_size>=2 required")
    batches = (capacity + batch_size - 1) // batch_size
    signature = (
        tf.TensorSpec([capacity, dimension], tf.float64),
        tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.bool),
        tf.TensorSpec([dimension], tf.float64), tf.TensorSpec([], tf.float64),
        tf.TensorSpec([dimension], tf.float64),
        tf.TensorSpec([], tf.int64), tf.TensorSpec([], tf.int64),
    )

    @tf.function(input_signature=signature, autograph=False, jit_compile=jit_compile)
    def evaluate(points, point_count, record, center, center_value, center_score, first_index, selected_index):
        initial = {
            "ok": tf.constant(True), "chunks": tf.constant(0),
            "callback_batches": tf.constant(0),
            "physical_rows": tf.constant(0, tf.int64),
            "padded_rows": tf.constant(0, tf.int64),
            "invalid_rows": tf.constant(0, tf.int64),
            "center": center, "center_value": center_value, "center_score": center_score,
            "selected_index": selected_index,
            "positions": tf.zeros([batches, batch_size, dimension], tf.float64),
            "values": tf.zeros([batches, batch_size], tf.float64),
            "scores": tf.zeros([batches, batch_size, dimension], tf.float64),
            "valid": tf.zeros([batches, batch_size], tf.bool),
        }

        def condition(state):
            return state["ok"] & (state["chunks"] * batch_size < point_count)

        def body(state):
            start = state["chunks"] * batch_size
            count = tf.minimum(batch_size, point_count - start)
            indices = tf.minimum(start + tf.range(batch_size), point_count - 1)
            chunk = tf.gather(points, indices)
            next_state = dict(state)
            next_state["chunks"] += 1
            next_state["physical_rows"] += tf.cast(batch_size, tf.int64)
            next_state["padded_rows"] += tf.cast(batch_size - count, tf.int64)

            def call_target():
                values, scores, eligible = callback(chunk)
                values = tf.convert_to_tensor(values)
                scores = tf.convert_to_tensor(scores)
                eligible = tf.convert_to_tensor(eligible)
                if values.dtype != tf.float64 or scores.dtype != tf.float64 or eligible.dtype != tf.bool:
                    raise TypeError("callback must return float64 values/scores and bool eligibility")
                if values.shape != (batch_size,) or scores.shape != (batch_size, dimension) or eligible.shape != values.shape:
                    raise ValueError("callback returned an invalid fixed-batch shape")
                valid = eligible & tf.math.is_finite(values) & tf.reduce_all(tf.math.is_finite(scores), axis=1)
                masked = tf.where(valid, values, tf.constant(-float("inf"), tf.float64))
                selected = tf.argmax(masked, output_type=tf.int32)
                # Scalar tensor slicing can make XLA specialize on the winner.
                improve = record & (tf.gather(masked, selected) > state["center_value"])
                output = dict(next_state)
                output["ok"] = tf.reduce_all(valid)
                output["callback_batches"] += 1
                output["invalid_rows"] += tf.reduce_sum(tf.cast(~valid, tf.int64))
                output["center"] = tf.where(improve, tf.gather(chunk, selected), state["center"])
                output["center_value"] = tf.where(improve, tf.gather(values, selected), state["center_value"])
                output["center_score"] = tf.where(improve, tf.gather(scores, selected), state["center_score"])
                output["selected_index"] = tf.where(
                    improve, first_index + state["physical_rows"] + tf.cast(selected, tf.int64),
                    state["selected_index"],
                )
                slot = tf.reshape(state["callback_batches"], [1, 1])
                output["positions"] = tf.tensor_scatter_nd_update(state["positions"], slot, chunk[None])
                output["values"] = tf.tensor_scatter_nd_update(state["values"], slot, values[None])
                output["scores"] = tf.tensor_scatter_nd_update(state["scores"], slot, scores[None])
                output["valid"] = tf.tensor_scatter_nd_update(state["valid"], slot, valid[None])
                return output

            def invalid_points():
                return {**next_state, "ok": tf.constant(False)}

            return (tf.cond(tf.reduce_all(tf.math.is_finite(chunk)), call_target, invalid_points),)

        result, = tf.while_loop(condition, body, (initial,), parallel_iterations=1)
        return result

    return evaluate
