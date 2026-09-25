"""TensorFlow compatibility primitives for the existing LEDH seeded inputs.

Unspawned NumPy 2.1.3 SeedSequence/PCG64 and TensorFlow 2.19 Philox Box--Muller
semantics. These are numerical helpers for an enclosing XLA program, not a
new random stream. Source anchors and license notices are recorded in
docs/plans/filter_gradient_ledh_native_endpoint_20260925.md and its evidence.
Upstream notices: third_party/audit/ledh-random-compat-20260925/NOTICE.
Integer words, state and uniforms are exact; transcendental rounding depends
on the execution backend. No runtime NumPy dependency is permitted.
"""

from __future__ import annotations

import math
import struct

import tensorflow as tf


def integer_seed_words(seed: int) -> tuple[int, ...]:
    """Encode host configuration, with the old integer/nonnegative contract."""
    seed = int(seed)
    if seed < 0:
        raise ValueError("expected non-negative integer")
    count = max(1, (seed.bit_length() + 31) // 32)
    return struct.unpack("<" + "I" * count, seed.to_bytes(count * 4, "little"))


def seed_sequence_words(entropy: tf.Tensor, count: int = 8) -> tf.Tensor:
    """Unspawned, four-word SeedSequence pool; uint32 output of fixed count."""
    entropy = tf.convert_to_tensor(entropy, tf.uint32)
    size = tf.shape(entropy)[0]
    pool = tf.zeros([4], tf.uint32)

    def hashmix(value, multiplier):
        value = tf.bitwise.bitwise_xor(value, multiplier)
        multiplier = multiplier * tf.constant(0x931E8875, tf.uint32)
        value = value * multiplier
        return tf.bitwise.bitwise_xor(value, tf.bitwise.right_shift(value, 16)), multiplier

    def mix(left, right):
        value = tf.constant(0xCA01F9DD, tf.uint32) * left - tf.constant(0x4973F715, tf.uint32) * right
        return tf.bitwise.bitwise_xor(value, tf.bitwise.right_shift(value, 16))

    def initial(index, pool, multiplier):
        word = tf.cond(index < size, lambda: entropy[index], lambda: tf.constant(0, tf.uint32))
        value, multiplier = hashmix(word, multiplier)
        return index + 1, tf.tensor_scatter_nd_update(pool, [[index]], [value]), multiplier

    _, pool, multiplier = tf.while_loop(
        lambda index, *_: index < 4, initial,
        (tf.constant(0), pool, tf.constant(0x43B0D7E5, tf.uint32)), parallel_iterations=1)

    def cross(index, pool, multiplier):
        source, destination = index // 4, index % 4

        def update():
            word, following = hashmix(pool[source], multiplier)
            value = mix(pool[destination], word)
            return tf.tensor_scatter_nd_update(pool, [[destination]], [value]), following

        pool, multiplier = tf.cond(source != destination, update, lambda: (pool, multiplier))
        return index + 1, pool, multiplier

    _, pool, multiplier = tf.while_loop(
        lambda index, *_: index < 16, cross, (tf.constant(0), pool, multiplier), parallel_iterations=1)

    def extra(index, pool, multiplier):
        source, destination = index // 4, index % 4
        word, multiplier = hashmix(entropy[source], multiplier)
        value = mix(pool[destination], word)
        return index + 1, tf.tensor_scatter_nd_update(pool, [[destination]], [value]), multiplier

    _, pool, _ = tf.while_loop(
        lambda index, *_: index < size * 4, extra,
        (tf.constant(16), pool, multiplier), parallel_iterations=1)

    def output(index, values, multiplier):
        value = tf.bitwise.bitwise_xor(pool[index % 4], multiplier)
        multiplier = multiplier * tf.constant(0x58F38DED, tf.uint32)
        value = value * multiplier
        value = tf.bitwise.bitwise_xor(value, tf.bitwise.right_shift(value, 16))
        return index + 1, tf.tensor_scatter_nd_update(values, [[index]], [value]), multiplier

    return tf.while_loop(lambda index, *_: index < count, output,
        (tf.constant(0), tf.zeros([count], tf.uint32), tf.constant(0x8B51F9DD, tf.uint32)),
        parallel_iterations=1)[1]


def philox_replication_state(entropy: tf.Tensor) -> tf.Tensor:
    words = tf.cast(seed_sequence_words(entropy, 2), tf.uint64)
    seed = tf.bitwise.bitwise_xor(tf.bitwise.left_shift(words[0], 31), words[1])
    return tf.stack([seed, tf.constant(0, tf.uint64), tf.constant(0, tf.uint64)])


def _add128(left, right):
    low = left[1] + right[1]
    return tf.stack([left[0] + right[0] + tf.cast(low < left[1], tf.uint64), low])


def _multiply_high64(left, right):
    mask = tf.constant(0xFFFFFFFF, tf.uint64)
    a0, a1 = tf.bitwise.bitwise_and(left, mask), tf.bitwise.right_shift(left, 32)
    b0, b1 = tf.bitwise.bitwise_and(right, mask), tf.bitwise.right_shift(right, 32)
    p00, p01, p10 = a0 * b0, a0 * b1, a1 * b0
    middle = (tf.bitwise.right_shift(p00, 32) + tf.bitwise.bitwise_and(p01, mask)
              + tf.bitwise.bitwise_and(p10, mask))
    return a1 * b1 + tf.bitwise.right_shift(p01, 32) + tf.bitwise.right_shift(p10, 32) + tf.bitwise.right_shift(middle, 32)


def _pcg64_advance(state, increment):
    high_multiplier = tf.constant(2549297995355413924, tf.uint64)
    low_multiplier = tf.constant(4865540595714422341, tf.uint64)
    low = state[1] * low_multiplier
    high = (state[0] * low_multiplier + state[1] * high_multiplier
            + _multiply_high64(state[1], low_multiplier))
    return _add128(tf.stack([high, low]), increment)


def pcg64_initial_state(entropy: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    words = tf.cast(seed_sequence_words(entropy, 8), tf.uint64)
    packed = tf.bitwise.bitwise_or(words[0::2], tf.bitwise.left_shift(words[1::2], 32))
    state = packed[:2]
    increment = tf.stack([
        tf.bitwise.bitwise_or(tf.bitwise.left_shift(packed[2], 1), tf.bitwise.right_shift(packed[3], 63)),
        tf.bitwise.bitwise_or(tf.bitwise.left_shift(packed[3], 1), tf.constant(1, tf.uint64)),
    ])
    return _pcg64_advance(_add128(increment, state), increment), increment


def pcg64_next(state: tf.Tensor, increment: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return next state, raw word and exact NumPy-compatible float64 uniform."""
    state = _pcg64_advance(state, increment)
    word = tf.bitwise.bitwise_xor(state[0], state[1])
    rotation = tf.bitwise.right_shift(state[0], 58)
    complement = tf.bitwise.bitwise_and(tf.constant(64, tf.uint64) - rotation, tf.constant(63, tf.uint64))
    word = tf.bitwise.bitwise_or(tf.bitwise.right_shift(word, rotation),
                                 tf.bitwise.left_shift(word, complement))
    uniform = tf.cast(tf.bitwise.right_shift(word, 11), tf.float64) * tf.constant(2. ** -53, tf.float64)
    return state, word, uniform


def philox_box_muller_normal(state: tf.Tensor, shape: tuple[int, ...], dtype: tf.DType):
    """Original TF normal transform over exact Philox words, with native math."""
    size = math.prod(shape)
    pair_count = (size + 1) // 2
    double = dtype == tf.float64
    word_count = pair_count * (4 if double else 2)
    words = tf.raw_ops.StatelessRandomUniformFullIntV2(shape=[word_count],
        key=state[2:], counter=state[:2], alg=1, dtype=tf.uint32)
    if double:
        words = tf.cast(words, tf.uint64)
        mantissa = tf.bitwise.bitwise_or(
            tf.bitwise.left_shift(tf.bitwise.bitwise_and(words[0::2], tf.constant(0xFFFFF, tf.uint64)), 32),
            words[1::2])
        uniform = tf.bitcast(tf.bitwise.bitwise_or(mantissa, tf.constant(1023 << 52, tf.uint64)), tf.float64) - 1.
    else:
        mantissa = tf.bitwise.bitwise_and(words, tf.constant(0x7FFFFF, tf.uint32))
        uniform = tf.bitcast(tf.bitwise.bitwise_or(mantissa, tf.constant(127 << 23, tf.uint32)), tf.float32) - 1.
    radial = tf.sqrt(-2. * tf.math.log(tf.maximum(uniform[0::2], tf.constant(1e-7, dtype))))
    # Original float32 C++ expression promotes M_PI multiplication to double.
    angle = tf.cast(tf.constant(2. * math.pi, tf.float64) * tf.cast(uniform[1::2], tf.float64), dtype)
    normal = tf.reshape(tf.stack([tf.sin(angle) * radial, tf.cos(angle) * radial], axis=1), [-1])[:size]
    low = state[0] + tf.constant(256 * size, tf.uint64)
    following = tf.stack([low, state[1] + tf.cast(low < state[0], tf.uint64), state[2]])
    return tf.reshape(normal, shape), following
