"""Native FP64 gamma draws preserving TensorFlow 2.19.1's scalar-alpha stream.

The counter schedule, reversed sample buffers and rejection rule follow
tensorflow/core/kernels/stateless_random_gamma_op.cc:106--193 and its GPU
counterpart at the v2.19.1 tag. Both allocate 256 Philox blocks per output.
TSL random_distributions.h supplies Uint64ToDouble and BoxMullerDouble.
This is an execution port of the existing stream, not an RNG migration.
"""

import math

import tensorflow as tf

D = tf.float64


def _key_counter(seed):
    # Match random_ops_util._philox_scramble_seed explicitly. The XLA
    # StatelessRandomGetKeyCounter kernel changes negative int32 seed bits.
    mixed = tf.raw_ops.StatelessRandomUniformFullIntV2(shape=[4],
        key=tf.constant([0x02461E293EC8F720], tf.uint64), counter=tf.cast(seed, tf.uint64),
        alg=1, dtype=tf.uint32)
    words = tf.cast(tf.reshape(mixed, [2, 2]), tf.uint64)
    combined = tf.bitwise.bitwise_or(words[:, 0], tf.bitwise.left_shift(words[:, 1], 32))
    return combined[:1], tf.stack([tf.constant(0, tf.uint64), combined[1]])


def _uniform_pair(key, counter, offset):
    low = tf.bitcast(tf.bitcast(counter[0], tf.int64) + offset, tf.uint64)
    high = counter[1] + tf.cast(low < counter[0], tf.uint64)
    words = tf.raw_ops.StatelessRandomUniformFullIntV2(shape=[2, 2], key=key,
        counter=tf.stack([low, high]), alg=1, dtype=tf.uint32)
    top = tf.cast(tf.bitwise.bitwise_and(words[:, 0], 0xFFFFF), tf.uint64)
    mantissa = tf.bitwise.bitwise_or(tf.bitwise.left_shift(top, 32), tf.cast(words[:, 1], tf.uint64))
    bits = tf.bitwise.bitwise_or(mantissa, tf.constant(1023 << 52, tf.uint64))
    return tf.bitcast(bits, D) - 1.


def philox_gamma_float64(shape, seed, alpha, beta=1.):
    """Match scalar-concentration/rate stateless_gamma, including tiny clipping.

Callers validate positive finite scalar parameters at their configuration
boundary. Invalid dynamic parameters return NaNs without entering rejection.
The owning preparation factory encloses this primitive in a fixed XLA program.
"""
    count = math.prod(shape)
    alpha, beta = tf.convert_to_tensor(alpha, D), tf.convert_to_tensor(beta, D)
    if alpha.shape.rank != 0 or beta.shape.rank != 0:
        raise ValueError("gamma stream requires scalar alpha and beta")
    key, counter = _key_counter(seed)

    def sample(index):
        start = tf.cast(index, tf.int64) * tf.constant(256, tf.int64)

        def take(offset, buffer, remaining, normal):
            def refill():
                values = _uniform_pair(key, counter, offset)
                if normal:
                    radius = tf.sqrt(-2. * tf.math.log(tf.maximum(values[0], tf.constant(1e-7, D))))
                    angle = tf.constant(2. * math.pi, D) * values[1]
                    values = tf.stack([tf.sin(angle) * radius, tf.cos(angle) * radius])
                return offset + 1, values, tf.constant(1), values[1]

            return tf.cond(remaining > 0,
                lambda: (offset, buffer, remaining - 1, buffer[remaining - 1]), refill)

        def rejection():
            d = alpha + tf.where(alpha < 1., tf.constant(2. / 3., D), tf.constant(-1. / 3., D))
            c = tf.constant(1. / 3., D) / tf.sqrt(d)

            def attempt(offset, normals, nleft, uniforms, uleft, accepted, value):
                del accepted, value
                offset, normals, nleft, x = take(offset, normals, nleft, True)
                raw = 1. + c * x

                def positive():
                    v = raw * raw * raw
                    following, buffer, remaining, u = take(offset, uniforms, uleft, False)
                    accepted = ((u < 1. - .0331 * (x * x) * (x * x)) |
                                (tf.math.log(u) < .5 * x * x + d * (1. - v + tf.math.log(v))))

                    def augment():
                        position, samples, available, b = take(following, buffer, remaining, False)
                        return position, samples, available, d * v * tf.pow(b, 1. / alpha)

                    following, buffer, remaining, result = tf.cond(accepted & (alpha < 1.),
                        augment, lambda: (following, buffer, remaining, d * v))
                    return following, buffer, remaining, accepted, result

                offset, uniforms, uleft, accepted, value = tf.cond(raw > 0., positive,
                    lambda: (offset, uniforms, uleft, tf.constant(False), tf.constant(0., D)))
                return offset, normals, nleft, uniforms, uleft, accepted, value

            result = tf.while_loop(lambda *state: ~state[-2], attempt,
                (start, tf.zeros([2], D), tf.constant(0), tf.zeros([2], D), tf.constant(0),
                 tf.constant(False), tf.constant(0., D)), parallel_iterations=1)
            return result[-1]

        return tf.cond(alpha == 1., lambda: -tf.math.log1p(-_uniform_pair(key, counter, start)[1]), rejection)

    valid = (alpha > 0.) & (beta > 0.) & tf.math.is_finite(alpha) & tf.math.is_finite(beta)
    values = tf.cond(valid, lambda: tf.map_fn(sample, tf.range(count),
        fn_output_signature=D, parallel_iterations=1),
        lambda: tf.fill([count], tf.constant(float("nan"), D)))
    return tf.reshape(tf.maximum(tf.constant(float.fromhex("0x1.0p-1022"), D), values / beta), shape)
