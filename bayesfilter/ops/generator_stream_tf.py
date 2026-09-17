"""Native Philox call schedule matching TensorFlow Generator.from_seed.

TensorFlow's stateful generator reserves 256 counter blocks per requested
sample (stateful_random_ops._prepare_key_counter / RngReadAndSkip). This
implementation carries that same counter explicitly and does not create a
resource variable or change the existing seed-to-stream mapping.
"""

import math
from functools import lru_cache

import tensorflow as tf


@lru_cache(maxsize=16)
def normal_call_program(count, state_dimension, observation_dimension, *, jit_compile=True):
    @tf.function(input_signature=[tf.TensorSpec([3], tf.uint64)],
                 jit_compile=jit_compile, autograph=False)
    def generate(state):
        def at_date(index):
            def draw(offset, dimension):
                # Signed addition has a GPU AddN kernel; bitcasts retain the
                # same modular counter when Grappler combines additions.
                low = tf.bitcast(tf.bitcast(state[0], tf.int64)
                                 + tf.bitcast(offset, tf.int64), tf.uint64)
                high = state[1] + tf.cast(low < state[0], tf.uint64)
                words = tf.raw_ops.StatelessRandomUniformFullIntV2(
                    shape=[(dimension+1)//2, 4], key=state[2:3], counter=tf.stack([low, high]),
                    alg=1, dtype=tf.uint32)

                def uniform(first, second):
                    top = tf.cast(tf.bitwise.bitwise_and(first, 0xFFFFF), tf.uint64)
                    mantissa = tf.bitwise.bitwise_or(tf.bitwise.left_shift(top, 32), tf.cast(second, tf.uint64))
                    bits = tf.bitwise.bitwise_or(mantissa, tf.constant(1023 << 52, tf.uint64))
                    return tf.bitcast(bits, tf.float64) - 1.0

                # Preserve TSL BoxMullerDouble, including its small-U floor and
                # sin/cos output order. XLA's normal op uses a different map.
                radius = tf.sqrt(-2.0*tf.math.log(tf.maximum(uniform(words[:, 0], words[:, 1]), tf.constant(1e-7, tf.float64))))
                angle = tf.constant(2.0*math.pi, tf.float64)*uniform(words[:, 2], words[:, 3])
                return tf.reshape(tf.stack([tf.sin(angle)*radius, tf.cos(angle)*radius], axis=1), [-1])[:dimension]
            base = tf.cast(index, tf.uint64) * tf.constant(256*(state_dimension+observation_dimension), tf.uint64)
            return (draw(base, state_dimension),
                    draw(base + tf.constant(256*state_dimension, tf.uint64), observation_dimension))

        return tf.map_fn(at_date, tf.range(count), fn_output_signature=(
            tf.TensorSpec([state_dimension], tf.float64),
            tf.TensorSpec([observation_dimension], tf.float64)), parallel_iterations=1)
    return generate


def generator_seed_state(seed):
    value, mask = int(seed), (1 << 64)-1
    return tf.constant([value & mask, (value >> 64) & mask, (value >> 128) & mask], tf.uint64)


def generator_normal_calls(seed, count, state_dimension, observation_dimension, *, jit_compile=True):
    return normal_call_program(count, state_dimension, observation_dimension,
        jit_compile=jit_compile)(generator_seed_state(seed))
