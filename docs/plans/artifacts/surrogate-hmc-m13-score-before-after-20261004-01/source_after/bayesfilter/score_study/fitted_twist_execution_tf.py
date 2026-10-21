"""Enclosing fixed-fit psi-APF execution with compatible stateless inputs.

The nominal fit is conditional on observations. The final analytical score
holds that fit fixed, exactly as the shared fitted-twist kernel specifies.
"""

from functools import lru_cache

import tensorflow as tf

from bayesfilter.ops.stateless_random_tf import (
    philox_normal_float32,
    philox_normal_float64,
    philox_uniform_float32,
    philox_uniform_float64,
)

from .fitted_twist_tf import make_fitted_twist_kernel, make_recursive_fit_kernel


@lru_cache(maxsize=12)
def make_fitted_twist_inputs(d, N, T, dtype_name="float64", jit_compile=True):
    """One initial/process/ancestor/mixture draw set; seed labels stay on host."""
    dtype = tf.as_dtype(dtype_name)
    if dtype not in (tf.float32, tf.float64):
        raise ValueError("fitted-twist inputs require float32 or float64")
    normal = philox_normal_float64 if dtype == tf.float64 else philox_normal_float32
    uniform = philox_uniform_float64 if dtype == tf.float64 else philox_uniform_float32

    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.int32)],
                 jit_compile=jit_compile, autograph=False)
    def inputs(seeds):
        return (normal([N, d], seeds[0]), normal([T, N, d], seeds[1]),
                uniform([T + 1, N], seeds[2]), uniform([T, N], seeds[3]))

    return inputs


@lru_cache(maxsize=12)
def make_fitted_twist_execution(d, o, N, T, iterations, initial_variance,
                                floor_ratio, dtype_name="float64", jit_compile=True,
                                transition_curve=0., observation_curve=0.):
    """Fit, reject or score in one stable graph; return histories for reporting.

    Histories after the first failed fit are NaN. ``attempted_iterations``
    includes that failed iteration; ``invalid_iteration`` is -1 on success.
    No final filter executes on failure. Coefficient histories are small fixed
    diagnostic arrays; observation/particle clouds are not retained per fit.
    """
    if type(iterations) is not int or not 1 <= iterations <= 20:
        raise ValueError("declare between one and twenty offline fit iterations")
    if d != 1 or o != 1:
        raise ValueError("fitted_twist currently requires dimension=observation_dimension=1")
    if not 0 < initial_variance < float("inf"):
        raise ValueError("positive finite initial variance required")
    dtype = tf.as_dtype(dtype_name)
    kernel = make_fitted_twist_kernel(d, o, N, T, dtype_name, jit_compile,
        transition_curve=transition_curve, observation_curve=observation_curve)
    fitter = make_recursive_fit_kernel(d, o, N, T, floor_ratio, dtype_name, jit_compile,
        transition_curve=transition_curve, observation_curve=observation_curve)
    inputs = make_fitted_twist_inputs(d, N, T, dtype_name, jit_compile)

    @tf.function(input_signature=[tf.TensorSpec([6], dtype), tf.TensorSpec([6], dtype),
        tf.TensorSpec([T, o], dtype), tf.TensorSpec([iterations + 1, 4, 2], tf.int32)],
        jit_compile=jit_compile, autograph=False)
    def execute(theta, fit_theta, observations, seeds):
        missing = tf.constant(float("nan"), dtype)
        centers = tf.zeros([T, d], dtype)
        covariance = tf.eye(d, batch_shape=[T], dtype=dtype) * tf.cast(initial_variance, dtype)
        floors = tf.fill([T], tf.math.log(tf.cast(floor_ratio, dtype)) -
            .5 * tf.cast(d, dtype) * tf.math.log(tf.cast(2 * 3.141592653589793 * initial_variance, dtype)))

        def step(i, invalid, centers, covariance, floors, values, errors,
                 center_history, covariance_history, floor_history):
            fitted = kernel(fit_theta, observations, *inputs(seeds[i]), centers, covariance, floors)
            centers, covariance, floors, valid, residual = fitter(fit_theta, observations, fitted[2])
            invalid = tf.where(valid, invalid, i)
            return (i + 1, invalid, centers, covariance, floors,
                tf.tensor_scatter_nd_update(values, [[i]], [fitted[0]]),
                tf.tensor_scatter_nd_update(errors, [[i]], [residual]),
                tf.tensor_scatter_nd_update(center_history, [[i]], [centers]),
                tf.tensor_scatter_nd_update(covariance_history, [[i]], [covariance]),
                tf.tensor_scatter_nd_update(floor_history, [[i]], [floors]))

        out = tf.while_loop(lambda i, invalid, *_: (i < iterations) & (invalid < 0), step,
            (tf.constant(0), tf.constant(-1), centers, covariance, floors,
             tf.fill([iterations], missing), tf.fill([iterations, T], missing),
             tf.fill([iterations, T, d], missing), tf.fill([iterations, T, d, d], missing),
             tf.fill([iterations, T], missing)), maximum_iterations=iterations, parallel_iterations=1)
        final = tf.cond(out[1] < 0,
            lambda: kernel(theta, observations, *inputs(seeds[iterations]), out[2], out[3], out[4]),
            lambda: (missing, tf.fill([6], missing), tf.fill([T, N, d], missing)))
        return {"final": final, "fit": (out[2], out[3], out[4]),
                "fit_log_values": out[5], "fit_errors": out[6],
                "fit_centers": out[7], "fit_covariances": out[8], "fit_log_floors": out[9],
                "attempted_iterations": out[0], "invalid_iteration": out[1]}

    return execute
