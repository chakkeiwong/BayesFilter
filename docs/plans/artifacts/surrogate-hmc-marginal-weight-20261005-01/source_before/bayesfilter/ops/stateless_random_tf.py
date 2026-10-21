"""TensorFlow random primitives with explicit graph/XLA bit conversion."""

import math

import tensorflow as tf


def philox_uniform_float32(shape, seed):
    """Match TSL Uint32ToFloat, retaining the low 23 Philox bits."""
    words = tf.random.stateless_uniform(
        shape, seed, minval=None, maxval=None, dtype=tf.uint32, alg="philox")
    mantissa = tf.bitwise.bitwise_and(words, tf.constant(0x7FFFFF, tf.uint32))
    bits = tf.bitwise.bitwise_or(mantissa, tf.constant(127 << 23, tf.uint32))
    return tf.bitcast(bits, tf.float32) - tf.constant(1., tf.float32)


def philox_normal_float32(shape, seed):
    """Preserve TSL BoxMullerFloat's draw order and double angle product."""
    count = math.prod(shape)
    uniforms = philox_uniform_float32([(count + 1) // 2, 2], seed)
    radius = tf.sqrt(-2.0 * tf.math.log(tf.maximum(
        uniforms[:, 0], tf.constant(1e-7, tf.float32))))
    angle = tf.cast(tf.constant(2. * math.pi, tf.float64) *
                    tf.cast(uniforms[:, 1], tf.float64), tf.float32)
    values = tf.stack([tf.sin(angle) * radius, tf.cos(angle) * radius], axis=1)
    return tf.reshape(tf.reshape(values, [-1])[:count], shape)


def philox_uniform_float64(shape, seed):
    """Match TensorFlow's non-XLA Philox ``Uint64ToDouble`` conversion.

    XLA's uniform-double kernel combines the same Philox words differently.
    Constructing the mantissa explicitly preserves the existing stream. The
    source operation is TSL ``random_distributions_utils.h::Uint64ToDouble``:
    the low 20 bits of the first word precede all 32 bits of the second word.
    Raw Philox bits, shifts and bitcast are native TensorFlow/XLA operations.
    """
    shape = tf.convert_to_tensor(shape, tf.int32)
    words = tf.random.stateless_uniform(
        tf.concat([shape, [2]], axis=0), seed, minval=None, maxval=None,
        dtype=tf.uint32, alg="philox")
    high = tf.cast(tf.bitwise.bitwise_and(words[..., 0], 0xFFFFF), tf.uint64)
    mantissa = tf.bitwise.bitwise_or(tf.bitwise.left_shift(high, 32),
                                   tf.cast(words[..., 1], tf.uint64))
    bits = tf.bitwise.bitwise_or(mantissa, tf.constant(1023 << 52, tf.uint64))
    return tf.bitcast(bits, tf.float64) - tf.constant(1.0, tf.float64)


def philox_normal_float64(shape, seed):
    """Preserve TSL BoxMullerDouble's words, floor and sin/cos ordering.

    The ordinary XLA normal kernel changes the seed-to-draw map. Explicit
    conversion retains the non-XLA Philox stream, up to libm rounding.
    """
    count = math.prod(shape)
    uniforms = philox_uniform_float64([(count + 1) // 2, 2], seed)
    radius = tf.sqrt(-2.0 * tf.math.log(tf.maximum(
        uniforms[:, 0], tf.constant(1e-7, tf.float64))))
    angle = tf.constant(2.0 * math.pi, tf.float64) * uniforms[:, 1]
    values = tf.stack([tf.sin(angle) * radius, tf.cos(angle) * radius], axis=1)
    return tf.reshape(tf.reshape(values, [-1])[:count], shape)


def philox_shuffle_indices(size, seed):
    """Preserve TF shuffle_common.h's forward Fisher-Yates and Philox words.

    The original shuffle consumes exactly size-1 sequential uint32 samples,
    swapping i with i + sample % (size-i). Explicit words and tensor swaps
    preserve that finite program in both graph and XLA execution.
    """
    indices = tf.range(size, dtype=tf.int32)
    if size <= 1:
        return indices
    words = tf.random.stateless_uniform([size - 1], seed, minval=None, maxval=None,
        dtype=tf.uint32, alg="philox")

    def swap(index, permutation):
        offset = tf.cast(words[index] % tf.cast(size - index, tf.uint32), tf.int32)
        other = index + offset
        positions = tf.stack([index, other])
        values = tf.gather(permutation, tf.stack([other, index]))
        return index + 1, tf.tensor_scatter_nd_update(permutation, positions[:, None], values)

    return tf.while_loop(lambda index, _: index < size - 1, swap,
        (tf.constant(0), indices), maximum_iterations=size - 1, parallel_iterations=1)[1]


def stateless_categorical_cpu_stream(logits, count, seed):
    """One-row inverse-CDF draw matching TF's CPU categorical random stream.

    The stream name is explicit: GPU categorical kernels may use a different
    sampling engine. Consumers must qualify their existing execution device.
    """
    logits = tf.convert_to_tensor(logits, tf.float64)
    weights = tf.exp(logits - tf.reduce_max(logits))
    cumulative = tf.cumsum(weights)
    uniforms = philox_uniform_float64([count], seed) * cumulative[-1]
    return tf.searchsorted(cumulative, uniforms, side="right", out_type=tf.int32)
