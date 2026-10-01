"""CPU/XLA preparation with the existing posterior geometry random stream.

Python enumerates static seed identities only. Every numerical draw and cloud
assembly executes in TensorFlow, without target callbacks or host feedback.
"""

import tensorflow as tf

from bayesfilter.inference.posterior_local_initializer import _cloud_row_counts
from bayesfilter.inference.quadratic_geometry_full_tf import geometry_extents
from bayesfilter.ops.geometry_random_tf import (
    STREAM_ID,
    GeometryTensorStream,
    _ball_kernel,
    _draw_kernel,
)

D = tf.float64
I = tf.int32


def posterior_seed_keys(dimension, config, movement_config):
    """Hash only static attempt/partition/role metadata; perform no draws."""
    rank, _, _, _ = geometry_extents(dimension, movement_config)
    movement_keys, curvature_keys = [], []
    with tf.device("/CPU:0"):
        for attempt in range(config.max_movement_attempts):
            stream = GeometryTensorStream((movement_config.seed[0], movement_config.seed[1] + attempt))
            normal = stream._next_seed("normal") if rank else tf.zeros([2], I)
            movement_keys.append(tf.stack((normal, stream._next_seed("ball"), stream._next_seed("permutation"))))
        for attempt in range(config.max_curvature_attempts):
            partition_keys = []
            for partition in range(2 * config.replicate_count + 1):
                stream = GeometryTensorStream((config.seed[0] + attempt,
                    config.seed[1] + 1000 * attempt + partition))
                partition_keys.append(stream._next_seed("ball"))
            curvature_keys.append(tf.stack(partition_keys))
        return tf.stack(movement_keys), tf.stack(curvature_keys)


class PosteriorCloudPreparation:
    """Reusable native generator with CPU placement at its invocation boundary."""

    stream_id = STREAM_ID

    def __init__(self, dimension, config, movement_config, *, jit_compile=True):
        rank, _, samples, directions = geometry_extents(dimension, movement_config)
        training, selection, audit = _cloud_row_counts(config, dimension)
        movement_attempts, curvature_attempts = config.max_movement_attempts, config.max_curvature_attempts
        replicates, partitions = config.replicate_count, 2 * config.replicate_count + 1
        capacity = max(training, selection, audit)
        normal = _draw_kernel("normal", (directions, dimension)) if rank else None
        movement_ball = _ball_kernel(samples, dimension)
        training_ball, selection_ball, audit_ball = (_ball_kernel(training, dimension),
            _ball_kernel(selection, dimension), _ball_kernel(audit, dimension))

        @tf.function(input_signature=[tf.TensorSpec([movement_attempts, 3, 2], I),
            tf.TensorSpec([curvature_attempts, partitions, 2], I),
            tf.TensorSpec([], D), tf.TensorSpec([], D)], jit_compile=jit_compile, autograph=False)
        def generate(movement_keys, curvature_keys, movement_radius, curvature_radius):
            minimum = tf.constant(0., D)

            def movement_step(index, raw_history, offset_history):
                keys = tf.gather(movement_keys, index)
                raw = normal(keys[0]) if rank else tf.zeros([0, dimension], D)
                offsets = movement_ball(keys[1], movement_radius, minimum)
                return index + 1, raw_history.write(index, raw), offset_history.write(index, offsets)

            _, raw, movement = tf.while_loop(lambda index, *_: index < movement_attempts,
                movement_step, (tf.constant(0, I),
                    tf.TensorArray(D, size=movement_attempts, element_shape=[directions, dimension]),
                    tf.TensorArray(D, size=movement_attempts, element_shape=[samples, dimension])),
                parallel_iterations=1)

            def curvature_step(index, history):
                attempt, partition = index // partitions, index % partitions
                key = tf.gather_nd(curvature_keys, tf.stack((attempt, partition))[None])[0]

                def draw(kernel, rows):
                    return tf.pad(kernel(key, curvature_radius, tf.constant(.25, D)),
                        [[0, capacity - rows], [0, 0]])

                offsets = tf.cond(partition < replicates, lambda: draw(training_ball, training),
                    lambda: tf.cond(partition < 2 * replicates,
                        lambda: draw(selection_ball, selection), lambda: draw(audit_ball, audit)))
                return index + 1, history.write(index, offsets)

            _, curvature = tf.while_loop(lambda index, _: index < curvature_attempts * partitions,
                curvature_step, (tf.constant(0, I), tf.TensorArray(D,
                    size=curvature_attempts * partitions, element_shape=[capacity, dimension])),
                parallel_iterations=1)
            return tf.nest.map_structure(tf.stop_gradient, {
                "directions": raw.stack(), "movement_offsets": movement.stack(),
                "permutation_keys": movement_keys[:, 2],
                "curvature_offsets": tf.reshape(curvature.stack(),
                    [curvature_attempts, partitions, capacity, dimension]),
            })

        self.compiled = generate

    def __call__(self, movement_keys, curvature_keys, movement_radius, curvature_radius):
        with tf.device("/CPU:0"):
            return self.compiled(movement_keys, curvature_keys, movement_radius, curvature_radius)
