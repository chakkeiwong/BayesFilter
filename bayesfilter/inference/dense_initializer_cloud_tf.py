"""Internal ordered dense-initializer cloud evaluation on prepared offsets.

This owns partition evaluation and strict incumbent selection only. Cloud RNG,
the attempt controller, fitting, and public integration are qualified separately.
Padding stores completed history; it never adds target-evaluation rows.
"""

import tensorflow as tf

from bayesfilter.inference.joint_center_tf import rounded_affine_position

D = tf.float64


def make_dense_initializer_cloud_program(callback, dimension, replicates,
        training_rows, selection_rows, audit_rows, *, jit_compile=True):
    """Preserve training/selection/audit order and first-invalid short circuit."""
    if min(dimension, training_rows, selection_rows, audit_rows) < 1 or replicates < 2:
        raise ValueError("positive dimensions and at least two replicates are required")
    capacity = max(training_rows, selection_rows, audit_rows)
    partitions = 2 * replicates + 1
    counts = tf.concat((tf.fill([replicates], training_rows),
        tf.fill([replicates], selection_rows), tf.constant([audit_rows])), axis=0)

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([], D), tf.TensorSpec([partitions, capacity, dimension], D)],
        jit_compile=jit_compile, autograph=False)
    def evaluate(center, scale, center_value, offsets):
        initial = {"partition_count": tf.constant(0), "exact_rows": tf.constant(0, tf.int64),
            "valid": tf.constant(True), "candidate_center": center, "candidate_value": center_value,
            "positions": tf.zeros([partitions, capacity, dimension], D),
            "values": tf.zeros([partitions, capacity], D),
            "scores": tf.zeros([partitions, capacity, dimension], D),
            "row_validity": tf.zeros([partitions, capacity], tf.bool),
            "scaled_scores": tf.zeros([partitions, capacity, dimension], D)}

        def step(state):
            index = state["partition_count"]

            def partition(rows):
                cloud = tf.ensure_shape(tf.gather(offsets, index)[:rows], [rows, dimension])
                points = rounded_affine_position(center[None, :], scale, cloud)
                values, scores, valid = callback(points)
                values = tf.ensure_shape(tf.convert_to_tensor(values, D), [rows])
                scores = tf.ensure_shape(tf.convert_to_tensor(scores, D), [rows, dimension])
                valid = tf.ensure_shape(tf.convert_to_tensor(valid, tf.bool), [rows])
                finite = tf.reduce_all(tf.math.is_finite(values)) & tf.reduce_all(tf.math.is_finite(scores))
                accepted = tf.reduce_all(valid) & finite
                best = tf.argmax(values, output_type=tf.int32)
                best_value = tf.gather(values, best)
                improve = accepted & (best_value > state["candidate_value"])
                return {"valid": accepted,
                    "candidate_center": tf.where(improve, tf.gather(points, best), state["candidate_center"]),
                    "candidate_value": tf.where(improve, best_value, state["candidate_value"]),
                    "positions": tf.pad(points, [[0, capacity - rows], [0, 0]]),
                    "values": tf.pad(values, [[0, capacity - rows]]),
                    "scores": tf.pad(scores, [[0, capacity - rows], [0, 0]]),
                    "row_validity": tf.pad(valid, [[0, capacity - rows]]),
                    "scaled_scores": tf.pad(tf.where(accepted, scores * scale, tf.zeros_like(scores)),
                        [[0, capacity - rows], [0, 0]])}

            result = tf.cond(index < replicates, lambda: partition(training_rows),
                lambda: tf.cond(index < 2 * replicates,
                    lambda: partition(selection_rows), lambda: partition(audit_rows)))
            slot = tf.reshape(index, [1, 1])

            def append(name):
                return tf.tensor_scatter_nd_update(state[name], slot, result[name][None])

            return ({"partition_count": index + 1,
                "exact_rows": state["exact_rows"] + tf.cast(tf.gather(counts, index), tf.int64),
                "valid": result["valid"], "candidate_center": result["candidate_center"],
                "candidate_value": result["candidate_value"], "positions": append("positions"),
                "values": append("values"), "scores": append("scores"),
                "row_validity": append("row_validity"), "scaled_scores": append("scaled_scores")},)

        result, = tf.while_loop(lambda state: state["valid"] & (state["partition_count"] < partitions),
            step, (initial,), maximum_iterations=partitions, parallel_iterations=1)
        return tf.nest.map_structure(tf.stop_gradient, result)

    return evaluate
