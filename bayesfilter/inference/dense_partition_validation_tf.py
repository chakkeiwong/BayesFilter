"""Native validation for disjoint, padded dense-initializer partitions.

Only active rows participate. Configuration fixes disjoint logical partitions;
the public preparation boundary must separately reject overlapping caller buffers.
The returned first error must gate fitting before any numerical fit executes.
"""

import tensorflow as tf

D = tf.float64


def normalized_offset_bits(offsets):
    """Normalize signed zero without floating comparisons that flush subnormals."""
    bits = tf.bitcast(offsets, tf.int64)
    magnitude_bits = tf.bitwise.bitwise_and(bits, tf.constant(0x7fffffffffffffff, tf.int64))
    return tf.where(magnitude_bits == 0, tf.zeros_like(bits), bits)


def make_dense_partition_validation_program(dimension, replicates, training_rows,
        selection_rows, audit_rows, *, jit_compile=True):
    """Check finite inputs, then the first exact cross-partition copied row."""
    if min(dimension, training_rows, selection_rows, audit_rows) < 1 or replicates < 2:
        raise ValueError("positive dimensions and at least two replicates are required")
    capacity = max(training_rows, selection_rows, audit_rows)
    partitions = 2 * replicates + 1
    counts = tf.concat((tf.fill([replicates], training_rows),
        tf.fill([replicates], selection_rows), tf.constant([audit_rows])), axis=0)
    active = tf.range(capacity)[None, :] < counts[:, None]

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([partitions, capacity, dimension], D),
        tf.TensorSpec([partitions, capacity, dimension], D)],
        jit_compile=jit_compile, autograph=False)
    def validate(center, center_score, offsets, scores):
        finite = tf.reduce_all(~active[:, :, None] |
            (tf.math.is_finite(offsets) & tf.math.is_finite(scores)), axis=[1, 2])
        error = tf.where(~tf.reduce_all(tf.math.is_finite(center)), 1,
            tf.where(~tf.reduce_all(tf.math.is_finite(center_score)), 2,
            tf.where(~tf.reduce_all(finite[:replicates]), 3,
            tf.where(~tf.reduce_all(finite[replicates:2 * replicates]), 4,
            tf.where(~finite[-1], 5, 0)))))
        # Compare finite float64 encodings, including subnormals. Only the two
        # signed zero encodings are normalized by the original row-key policy.
        bits = normalized_offset_bits(offsets)

        def find_overlap():
            def step(left, right, found, pair):
                matching = tf.reduce_all(tf.gather(bits, left)[:, None, :] ==
                    tf.gather(bits, right)[None, :, :], axis=2)
                matching &= tf.gather(active, left)[:, None] & tf.gather(active, right)[None, :]
                found = tf.reduce_any(matching)
                pair = tf.where(found, tf.stack([left, right]), pair)
                last_right = right + 1 == partitions
                return (tf.where(last_right, left + 1, left),
                    tf.where(last_right, left + 2, right + 1), found, pair)

            _, _, found, pair = tf.while_loop(
                lambda left, right, found, pair: (left < partitions - 1) & ~found,
                step, (tf.constant(0), tf.constant(1), tf.constant(False), tf.constant([-1, -1])),
                maximum_iterations=partitions * (partitions - 1) // 2, parallel_iterations=1)
            return tf.where(found, 6, 0), pair

        error, pair = tf.cond(error == 0, find_overlap, lambda: (error, tf.constant([-1, -1])))
        return {"error_code": error, "overlap_pair": pair}

    return validate
