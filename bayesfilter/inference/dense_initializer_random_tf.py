"""Compiled dense-initializer clouds with the original TensorFlow Philox words.

Seeds retain the external initializer's attempt/partition and radius offsets.
Explicit binary64 conversion avoids XLA's different uniform/normal seed map.
Libm rounding is qualified against the original TensorFlow reference separately.
"""

import tensorflow as tf

from bayesfilter.ops.stateless_random_tf import (
    philox_normal_float64,
    philox_uniform_float64,
)

D = tf.float64
STREAM_ID = 'dense_initializer_tf_philox_original_words_v1'


def make_dense_initializer_cloud_design(dimension, replicates, training_rows,
        selection_rows, audit_rows, max_attempts, *, jit_compile=True):
    """Prepare fixed-capacity attempt clouds without host numerical iteration."""
    if min(dimension, training_rows, selection_rows, audit_rows, max_attempts) < 1 or replicates < 2:
        raise ValueError('positive dimensions and at least two replicates are required')
    partitions = 2 * replicates + 1
    capacity = max(training_rows, selection_rows, audit_rows)
    total = max_attempts * partitions

    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32), tf.TensorSpec([], D)],
        jit_compile=jit_compile, autograph=False)
    def generate(seed, radius):
        def step(index, storage):
            attempt, partition_index = index // partitions, index % partitions
            key = tf.stack([seed[0], seed[1] + 1000 * attempt + partition_index])
            radius_key = tf.stack([key[0], key[1] + 100])

            def cloud(rows):
                directions = philox_normal_float64([rows, dimension], key)
                norms = tf.linalg.norm(directions, axis=1, keepdims=True)
                radii = radius * philox_uniform_float64([rows, 1], radius_key) ** (1.0 / dimension)
                offsets = directions / norms * radii
                return tf.pad(offsets, [[0, capacity - rows], [0, 0]])

            offsets = tf.cond(partition_index < replicates, lambda: cloud(training_rows),
                lambda: tf.cond(partition_index < 2 * replicates,
                    lambda: cloud(selection_rows), lambda: cloud(audit_rows)))
            return index + 1, storage.write(index, offsets)

        _, storage = tf.while_loop(lambda index, _: index < total, step,
            (tf.constant(0), tf.TensorArray(D, size=total, element_shape=[capacity, dimension])),
            maximum_iterations=total, parallel_iterations=1)
        return tf.stop_gradient(tf.reshape(storage.stack(), [max_attempts, partitions, capacity, dimension]))

    return generate
