"""Preserve tf.cast(Python integer quotient) coefficients inside XLA loops."""

import tensorflow as tf


def legacy_fraction(numerator, denominator: int, dtype):
    """Round a nonnegative stage fraction through binary32, even in GPU XLA.

    The old Python quotient is binary64, but ``tf.cast(Python_float, dtype)``
    first constructs a binary32 tensor. GPU XLA's excess-precision optimization
    can erase a float64 -> float32 -> float64 cast pair. Explicit nearest-even
    mantissa rounding retains that finite-program contract. For stage fractions
    0 <= numerator <= denominator <= 2**31-1, every nonzero value is normal
    binary32, so rounding the 52-bit significand to 23 fraction bits suffices.
    This is not a general subnormal/overflow floating-point conversion.
    """
    if not 1 <= denominator <= 2**31 - 1:
        raise ValueError("stage denominator must be a positive int32 extent")
    value = tf.cast(numerator, tf.float64) / tf.constant(denominator, tf.float64)
    # These nonnegative finite fractions have a zero sign bit. Signed int64
    # preserves the exact bits and supports graph-mode GPU AddN fusion.
    bits = tf.bitcast(value, tf.int64)
    retained_lsb = tf.bitwise.bitwise_and(tf.bitwise.right_shift(bits, 29), tf.constant(1, tf.int64))
    rounded = bits + tf.constant((1 << 28) - 1, tf.int64) + retained_lsb
    rounded = tf.bitwise.bitwise_and(rounded, tf.constant(-1 << 29, tf.int64))
    return tf.cast(tf.bitcast(rounded, tf.float64), dtype)
