"""Ordered quadratic-design preparation and batched evaluation in TensorFlow/XLA.

All designs in one call share the input anchor, even when an earlier design improves
the incumbent. Completed histories are returned for host reporting only.
"""

import tensorflow as tf

from bayesfilter.inference.paired_score_pilot_tf import make_paired_score_probe_program
from bayesfilter.inference.quadratic_batch_evaluation_tf import (
    make_quadratic_batch_evaluator,
)
from bayesfilter.ops.stateless_random_tf import philox_uniform_float64


def make_paired_probe_evaluator(callback, dimension, batch_size, *, steps=(0.001, 0.0001), jit_compile=True):
    """Compile all three probe partitions with fixed-capacity batch histories."""
    generate = make_paired_score_probe_program(dimension, steps=tuple(steps), jit_compile=jit_compile)
    return _make_probe_evaluator(callback, dimension, batch_size, generate,
        rows=2 * dimension, partitions=3, jit_compile=jit_compile)


def make_uniform_probe_evaluator(callback, dimension, batch_size, rows, *, half_width, jit_compile=True):
    """Preserve both original Philox clouds and their ordered target batches."""
    @tf.function(input_signature=[tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.int32)],
                 autograph=False, jit_compile=jit_compile)
    def generate(seed, round_index):
        width = tf.constant(half_width, tf.float64)
        first = philox_uniform_float64([rows, dimension], tf.stack((seed, 2 * round_index)))
        second = philox_uniform_float64([rows, dimension], tf.stack((seed, 2 * round_index + 1)))
        return first * (2 * width) - width, second * (2 * width) - width

    return _make_probe_evaluator(callback, dimension, batch_size, generate,
        rows=rows, partitions=2, jit_compile=jit_compile)


def _make_probe_evaluator(callback, dimension, batch_size, generate, *, rows, partitions, jit_compile):
    """Shared ordered evaluator; capacities and design family are static."""
    batches = (rows + batch_size - 1) // batch_size
    evaluate = make_quadratic_batch_evaluator(callback, dimension, batch_size, rows, jit_compile=jit_compile)
    signature = [tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.int32),
        tf.TensorSpec([dimension], tf.float64), tf.TensorSpec([dimension], tf.float64),
        tf.TensorSpec([], tf.float64), tf.TensorSpec([dimension], tf.float64),
        tf.TensorSpec([], tf.int64), tf.TensorSpec([], tf.int64)]

    @tf.function(input_signature=signature, autograph=False, jit_compile=jit_compile)
    def probes(seed, round_index, scale, center, center_value, center_score, first_index, selected_index):
        designs = tf.stack(generate(seed, round_index))
        initial = {
            "partitions": tf.constant(0), "ok": tf.constant(True),
            "physical_rows": tf.constant(0, tf.int64), "padded_rows": tf.constant(0, tf.int64),
            "invalid_rows": tf.constant(0, tf.int64), "callback_batches": tf.constant(0),
            "center": center, "center_value": center_value, "center_score": center_score,
            "selected_index": selected_index,
            "partition_batches": tf.zeros([partitions], tf.int32),
            "partition_first_index": tf.zeros([partitions], tf.int64),
            "positions": tf.zeros([partitions, batches, batch_size, dimension], tf.float64),
            "values": tf.zeros([partitions, batches, batch_size], tf.float64),
            "scores": tf.zeros([partitions, batches, batch_size, dimension], tf.float64),
            "valid": tf.zeros([partitions, batches, batch_size], tf.bool),
            "scaled_scores": tf.zeros([partitions, rows, dimension], tf.float64),
        }

        def step(state):
            index = state["partitions"]
            offsets = tf.gather(designs, index)
            # The anchor is the original input, not the updated incumbent.
            points = center[None, :] + offsets * scale
            start = first_index + state["physical_rows"]
            result = evaluate(points, rows, True, state["center"], state["center_value"],
                state["center_score"], start, state["selected_index"])
            slot = tf.reshape(index, [1, 1])
            return ({
                "partitions": index + 1, "ok": result["ok"],
                "physical_rows": state["physical_rows"] + result["physical_rows"],
                "padded_rows": state["padded_rows"] + result["padded_rows"],
                "invalid_rows": state["invalid_rows"] + result["invalid_rows"],
                "callback_batches": state["callback_batches"] + result["callback_batches"],
                "center": result["center"], "center_value": result["center_value"],
                "center_score": result["center_score"], "selected_index": result["selected_index"],
                "partition_batches": tf.tensor_scatter_nd_update(state["partition_batches"], slot, result["callback_batches"][None]),
                "partition_first_index": tf.tensor_scatter_nd_update(state["partition_first_index"], slot, start[None]),
                "positions": tf.tensor_scatter_nd_update(state["positions"], slot, result["positions"][None]),
                "values": tf.tensor_scatter_nd_update(state["values"], slot, result["values"][None]),
                "scores": tf.tensor_scatter_nd_update(state["scores"], slot, result["scores"][None]),
                "valid": tf.tensor_scatter_nd_update(state["valid"], slot, result["valid"][None]),
                "scaled_scores": tf.tensor_scatter_nd_update(state["scaled_scores"], slot,
                    (tf.reshape(result["scores"], [-1, dimension])[:rows] * scale)[None]),
            },)

        result, = tf.while_loop(lambda state: state["ok"] & (state["partitions"] < partitions),
            step, (initial,), maximum_iterations=partitions, parallel_iterations=1)
        result = {**result, "check_offsets": designs[partitions - 1]}
        return {**result, "fit_offsets": designs[0]} if partitions == 2 else result

    return probes
