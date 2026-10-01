"""Compiled frozen designs for the existing squared-TT fitting programs.

The Philox word conversion, Sobol points, seeds, salts and row transformations
are the original non-XLA definitions. No row law or fitting control is changed.
"""

from functools import lru_cache

import tensorflow as tf

from bayesfilter.ops.stateless_random_tf import philox_uniform_float64


@lru_cache(maxsize=32)
def design_row_program(row_design, count, dimension, date_count, salt, *,
                       gaussian_degree=None, jit_compile=True):
    if row_design not in ("mc", "sobol"):
        raise ValueError(f"unknown row_design {row_design!r}")

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32),
                                  tf.TensorSpec([date_count], tf.int32)],
                 jit_compile=jit_compile, autograph=False)
    def generate(seed, dates):
        base = (tf.math.sobol_sample(dimension, count, dtype=tf.float64)
                if row_design == "sobol" else None)

        def at_date(date):
            key = tf.stack([seed, salt + date])
            if row_design == "sobol":
                shift = philox_uniform_float64([1, dimension], key)
                uniform = 2.0 * tf.math.floormod(base + shift, 1.0) - 1.0
            else:
                uniform = 2.0 * philox_uniform_float64([count, dimension], key) - 1.0
            if gaussian_degree is None:
                return uniform
            from bayesfilter.highdim.squared_tt_engine_gaussian_tf import _christoffel_rows_from_uniform
            return _christoffel_rows_from_uniform(uniform, dimension, gaussian_degree)

        signature = tf.TensorSpec([count, dimension], tf.float64)
        if gaussian_degree is not None:
            signature = (signature, tf.TensorSpec([count], tf.float64),
                         tf.TensorSpec([], tf.float64), tf.TensorSpec([], tf.bool))
        return tf.map_fn(at_date, dates, fn_output_signature=signature, parallel_iterations=1)

    return generate


def frozen_design_rows(config, count, dimension, dates, salt, *,
                       gaussian_degree=None, jit_compile=True):
    return design_row_program(config.row_design, count, dimension, dates.shape[0], salt,
        gaussian_degree=gaussian_degree, jit_compile=jit_compile)(
            tf.convert_to_tensor(config.seed, tf.int32), dates)
