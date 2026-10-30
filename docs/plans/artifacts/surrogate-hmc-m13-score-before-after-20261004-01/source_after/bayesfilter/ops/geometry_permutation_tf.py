"""Fixed-capacity permutation for the geometry_tf_philox_cpu_xla_v1 stream.

The caller supplies the unchanged stream/call-index seed. The active prefix
matches the compact Fisher-Yates draw; the inactive suffix is the identity.
"""

import tensorflow as tf


def make_geometry_permutation_program(capacity, *, jit_compile=True):
    if capacity < 0 or capacity >= 2**31:
        raise ValueError('permutation capacity must be in [0, 2**31)')

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32), tf.TensorSpec([2], tf.int32)],
                 autograph=False, jit_compile=jit_compile)
    def draw(count, seed):
        valid = (count >= 0) & (count <= capacity)
        identity = tf.range(capacity)

        def swap(index, values):
            bound = tf.cast(index + 1, tf.uint64)
            space = tf.constant(1 << 32, tf.uint64)
            limit = space - space % bound
            step_seed = tf.random.experimental.stateless_fold_in(seed, index, alg='philox')

            def word(attempt):
                return tf.cast(tf.random.stateless_uniform([], tf.random.experimental.stateless_fold_in(
                    step_seed, attempt, alg='philox'), minval=None, maxval=None,
                    dtype=tf.uint32, alg='philox'), tf.uint64)

            _, chosen = tf.while_loop(lambda attempt, value: value >= limit,
                lambda attempt, value: (attempt + 1, word(attempt)), (tf.constant(1), word(tf.constant(0))))
            other = tf.cast(chosen % bound, tf.int32)
            values = tf.tensor_scatter_nd_update(values, [[index], [other]],
                [tf.gather(values, other), tf.gather(values, index)])
            return index - 1, values

        def permute():
            return tf.while_loop(lambda index, _: index > 0, swap,
                (count - 1, identity), maximum_iterations=capacity - 1)[1]

        values = tf.cond(valid, permute, lambda: identity) if capacity > 1 else identity
        return {'permutation': values, 'valid': valid}

    return draw
