"""TFP randomized Halton arithmetic with explicit ordinary/XLA RNG conversion.

This is the finite algorithm in TFP 0.25 sample_halton_sequence_lib.py,
lines181--246 and265--301: salted seed split, independently shuffled radix
digits and uniform trailing-zero correction. The tensor gather below is the
same permutation lookup without NumPy shape/index preparation. Callers that
move an ordinary generator under XLA select the existing explicit Philox
conversion; callers already under XLA retain its original conversion.
"""

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.ops.halton_primes import HALTON_PRIMES
from bayesfilter.ops.stateless_random_tf import (
    philox_uniform_float32,
    philox_uniform_float64,
)


def randomized_halton(num_results, dimension, seed, dtype, *, ordinary_stream=False):
    """Generate the same TFP point set; enclosing caller owns compilation."""
    if not isinstance(dimension, int) or not 1 <= dimension <= len(HALTON_PRIMES):
        raise ValueError('Halton dimension must be between 1 and 10000')
    if not isinstance(num_results, int) or num_results < 1:
        raise ValueError('Halton num_results must be a positive integer')
    dtype = tf.as_dtype(dtype)
    if dtype not in (tf.float32, tf.float64):
        raise ValueError('Halton dtype must be float32 or float64')
    radixes = tf.constant(HALTON_PRIMES[:dimension], dtype)[:, None]
    # TFP computes these static integer-base logarithms in NumPy float64,
    # then casts the resulting digit counts to the requested sample dtype.
    sizes = tf.cast(tf.floor(tf.math.log(tf.constant(num_results, tf.float64)) /
                             tf.math.log(tf.cast(radixes, tf.float64))) + 1., dtype)
    max_size = num_results.bit_length()
    exponents = tf.cast(tf.range(max_size)[None, :], dtype)
    relevant = exponents < sizes
    weights = tf.pow(radixes, tf.where(relevant, exponents, tf.zeros_like(exponents)))
    indices = tf.cast(tf.range(1, num_results + 1), dtype)[:, None, None]
    coefficients = tf.math.floordiv(indices, weights)
    coefficients *= tf.cast(relevant, dtype)
    coefficients = tf.cast(tf.math.floormod(coefficients, radixes), tf.int32)

    shuffle_seed, correction_seed = tfp.random.split_seed(seed, salt='MCMCSampleHaltonSequence')
    max_radix = HALTON_PRIMES[dimension - 1]
    permutation_shape = [max_size, dimension, max_radix]
    uniforms = (philox_uniform_float32(permutation_shape, shuffle_seed) if ordinary_stream
                else tf.random.stateless_uniform(permutation_shape, shuffle_seed))
    column = tf.range(max_radix)
    mask = column[None, :] >= tf.constant(HALTON_PRIMES[:dimension], tf.int32)[:, None]
    uniforms = tf.where(mask, tf.cast(column + 10, tf.float32), uniforms)
    permutations = tf.transpose(tf.argsort(uniforms, axis=-1), [1, 0, 2])
    permuted = tf.gather(permutations, tf.transpose(coefficients, [1, 2, 0]), axis=2, batch_dims=2)
    digits = tf.cast(tf.transpose(permuted, [2, 0, 1]), dtype) * tf.cast(relevant, dtype)
    values = tf.reduce_sum((digits / radixes) / weights, axis=-1)
    if ordinary_stream:
        uniform = philox_uniform_float64 if dtype == tf.float64 else philox_uniform_float32
        correction = uniform([dimension, 1], correction_seed)
    else:
        correction = tf.random.stateless_uniform([dimension, 1], correction_seed, dtype=dtype)
    correction /= tf.pow(radixes, sizes)
    return values + tf.reshape(correction, [-1])
