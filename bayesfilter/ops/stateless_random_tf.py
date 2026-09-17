"""TensorFlow random primitives with explicit graph/XLA bit conversion."""

import math

import tensorflow as tf


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
