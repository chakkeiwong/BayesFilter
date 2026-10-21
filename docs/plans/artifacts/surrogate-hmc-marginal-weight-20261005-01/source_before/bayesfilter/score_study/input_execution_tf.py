"""Compiled score-study clouds preserving the existing Philox seed streams."""

from functools import lru_cache

import tensorflow as tf

from bayesfilter.ops.stateless_random_tf import (
    philox_normal_float32,
    philox_normal_float64,
    philox_uniform_float32,
    philox_uniform_float64,
)


@lru_cache(maxsize=12)
def make_score_inputs(d, N, T, dtype_name="float64", jit_compile=True, twist=False):
    """Draw initial/process/reset clouds and ordinary or twist uniforms.

    Seed rows are initial, process, resampling, reset_design and, for twist,
    twist_initial_ancestor. Configuration and seed labels stay on the host.
    Floating normal conversion retains the established backend libm rounding.
    """
    if type(d) is not int or d < 1 or type(N) is not int or N < 1 or type(T) is not int or T < 1:
        raise ValueError("score inputs require positive integer dimensions, particles and horizon")
    dtype = tf.as_dtype(dtype_name)
    if dtype not in (tf.float32, tf.float64):
        raise ValueError("score inputs require float32 or float64")
    normal = philox_normal_float64 if dtype == tf.float64 else philox_normal_float32
    uniform = philox_uniform_float64 if dtype == tf.float64 else philox_uniform_float32

    @tf.function(input_signature=[tf.TensorSpec([5 if twist else 4, 2], tf.int32)],
                 jit_compile=jit_compile, autograph=False)
    def inputs(seeds):
        uniforms = uniform([T, N], seeds[2])
        if twist:
            uniforms = tf.concat([uniform([1, N], seeds[4]), uniforms], axis=0)
        return (normal([N, d], seeds[0]), normal([T, N, d], seeds[1]),
                uniforms, normal([N, d], seeds[3]))

    return inputs
