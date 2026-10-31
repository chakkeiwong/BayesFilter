"""Compiled theta binding for the shared Gaussian analytical filters."""

from functools import lru_cache

import tensorflow as tf

from .gaussian_tf import make_gaussian_kernel, parameterized_model


@lru_cache(maxsize=32)
def make_gaussian_execution(dimension, observation_dimension, dtype_name="float64",
                            jit_compile=True, unscented=False):
    """Keep model transforms, initial tangents and filter recursion inside XLA."""
    dtype = tf.as_dtype(dtype_name)
    kernel = make_gaussian_kernel(dimension, observation_dimension, 6,
                                 dtype_name, jit_compile, unscented)

    @tf.function(input_signature=[tf.TensorSpec([6], dtype),
                                  tf.TensorSpec([None, observation_dimension], dtype)],
                 jit_compile=jit_compile, autograph=False)
    def execute(theta, observations):
        return kernel(observations, *parameterized_model(theta, dimension, observation_dimension))

    return execute
