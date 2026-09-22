"""Diagnostic controls centered by independent draws from the same noise law.

The reference innovations must be independent of the filter and calibration
and have the same marginal generator/dtype as the filter innovations. No
ideal-normal moment assumption is used. These controls preserve an estimator's
expectation when frozen coefficients are subtracted outside its normalization;
they cannot remove its existing bias or issue an HMC gradient.
"""
from functools import lru_cache

import tensorflow as tf


@lru_cache(maxsize=8)
def make_innovation_control_kernel(particle_count, time_count, dimension,
                                   reference_batches=16, dtype_name="float32",
                                   jit_compile=True):
    """Return first/second moment contrasts, flattened in time/coordinate order.

    Input time zero is the initial noise, followed by each transition's noise.
    FP64 accumulation is an explicit diagnostic precision exception. The
    caller owns the independence and identical-marginal sampling contract.
    """
    sizes = (particle_count, time_count, dimension, reference_batches)
    if any(type(size) is not int or size < 1 for size in sizes):
        raise ValueError("control dimensions and reference batches must be positive integers")
    if dtype_name not in ("float32", "float64"):
        raise ValueError("innovation input precision must be float32 or float64")
    n, t, d, m = sizes
    dtype = tf.as_dtype(dtype_name)

    @tf.function(input_signature=[tf.TensorSpec([t, n, d], dtype),
                                 tf.TensorSpec([m, t, n, d], dtype)],
                 jit_compile=jit_compile)
    def kernel(innovations, reference_innovations):
        z = tf.cast(innovations, tf.float64)
        reference = tf.cast(reference_innovations, tf.float64)
        empirical = tf.reduce_mean(tf.stack([z, z*z], -1), axis=1)
        independent = tf.reduce_mean(tf.reduce_mean(
            tf.stack([reference, reference*reference], -1), axis=2), axis=0)
        return tf.reshape(tf.sqrt(tf.cast(n, tf.float64)) * (empirical-independent), [t*d*2])

    return kernel
