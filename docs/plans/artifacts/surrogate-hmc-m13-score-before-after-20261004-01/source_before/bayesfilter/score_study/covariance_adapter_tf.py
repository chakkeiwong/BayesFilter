"""Alternative covariance candidates using the single shared LEDH executor."""
from functools import lru_cache
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import _value_and_analytical_score_impl
from bayesfilter.nonlinear.sgqf_covariance_provider_tf import SGQFCovarianceProvider
from .canonical_adapter_tf import gaussian_direction_inputs, make_canonical_kernel


@lru_cache(maxsize=16)
def make_covariance_kernel(d, o, N, T, controls_tuple, level, dtype_name="float64", jit_compile=True):
    make_canonical_kernel(d, o, N, T, controls_tuple, dtype_name, jit_compile)
    provider = SGQFCovarianceProvider(d, level, dtype_name)
    dtype = tf.as_dtype(dtype_name)
    controls = dict(controls_tuple)
    @tf.function(input_signature=[tf.TensorSpec([6], dtype), tf.TensorSpec([6], dtype),
        tf.TensorSpec([T, o], dtype), tf.TensorSpec([N, d], dtype), tf.TensorSpec([T, N, d], dtype),
        tf.TensorSpec([N, d], dtype)], jit_compile=jit_compile)
    def kernel(theta, direction, observations, initial_noise, noise, reset_design):
        model, initial, dinitial, P, dP, *_ = gaussian_direction_inputs(theta, direction, initial_noise, d, o)
        value, score, trace = _value_and_analytical_score_impl(
            model, theta, initial, tf.broadcast_to(P, [N, d, d]), noise, observations,
            with_score=True, return_trace=True, initial_state_tangent=dinitial,
            initial_covariance_tangent=tf.broadcast_to(dP, [N, d, d]), reset_design=reset_design,
            moment_provider=(provider.predict, provider.update), **controls)
        # Trace verifies the same observation-conditioned covariance and reset
        # are carried into later predictions. No caller-issued canonical label.
        return value, score[0], tuple({k: step[k] for k in (
            "predicted_covariances", "post_covariances", "d_post_covariances",
            "covariances_after_reset", "d_covariances_after_reset")} for step in trace)
    return kernel, provider.metadata
