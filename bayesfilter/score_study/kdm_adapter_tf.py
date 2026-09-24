"""KDM consumers of the shared canonical loop; distinct finite targets.

Integrated observation factors and raw IWSG post-reset weights define different
finite programs. Neither is asserted to be an unbiased marginal model score.
The bandwidth is b²Q(theta); its full model-parameter derivative is included.
"""
from functools import lru_cache

import tensorflow as tf

from bayesfilter.highdim.ledh_younis_kdm_integrated_tf import (
    integrated_linear_gaussian_kdm_value_and_analytical_score,
)
from bayesfilter.highdim.ledh_younis_kdm_resampling_tf import (
    _factory_components, _run_resampling_program,
    RESPONSIBILITY_COVARIANCE_MARK_POLICY,
)
from .canonical_adapter_tf import gaussian_direction_inputs, make_canonical_kernel


@lru_cache(maxsize=16)
def make_kdm_kernel(d, o, N, T, controls_tuple, representation,
                    dtype_name="float64", jit_compile=True, replay=False):
    if representation not in ("integrated_kdm", "resampling_kdm"):
        raise ValueError("unknown KDM finite program")
    # Reuse canonical configuration and chunk validation, without evaluating it.
    make_canonical_kernel(d, o, N, T, controls_tuple, dtype_name, jit_compile)
    if replay and representation != "resampling_kdm":
        raise ValueError("frozen-sample replay applies only to resampling KDM")
    dtype = tf.as_dtype(dtype_name)
    controls = dict(controls_tuple)
    normalization_tolerance = 1e-5 if dtype == tf.float32 else 1e-8
    resampling, density = _factory_components(
        particle_count=N, state_dimension=d, dtype=dtype,
        rank_tolerance=1e-7 if dtype == tf.float32 else 1e-12,
        normalization_tolerance=normalization_tolerance,
        symmetry_tolerance=normalization_tolerance)

    @tf.function(input_signature=[
        tf.TensorSpec([6], dtype), tf.TensorSpec([6], dtype),
        tf.TensorSpec([T, o], dtype), tf.TensorSpec([N, d], dtype),
        tf.TensorSpec([T, N, d], dtype), tf.TensorSpec([N, d], dtype),
        tf.TensorSpec([T, N], dtype), tf.TensorSpec([T, N, d], dtype),
        tf.TensorSpec([], dtype)] + ([tf.TensorSpec([T, N, d], dtype),
        tf.TensorSpec([T, N], dtype), tf.TensorSpec([T, N], tf.int32)] if replay else []),
        jit_compile=jit_compile)
    def kernel(theta, direction, observations, initial_noise, noise, reset_design,
               uniforms, mixture_noise, bandwidth_scale, fixed_samples=None,
               fixed_proposal_log_densities=None, fixed_component_indices=None):
        model, initial, dinitial, P, dP, H, dH, Q, dQ = gaussian_direction_inputs(
            theta, direction, initial_noise, d, o)
        options = dict(controls, reset_design=reset_design,
                       initial_state_tangent=dinitial,
                       initial_covariance_tangent=tf.broadcast_to(dP, [N, d, d]))
        B, dB = bandwidth_scale**2 * Q, bandwidth_scale**2 * dQ
        if representation == "integrated_kdm":
            result = integrated_linear_gaussian_kdm_value_and_analytical_score(
                model, theta, initial, tf.broadcast_to(P, [N, d, d]), noise,
                observations, H, dH, tf.broadcast_to(B, [T, N, d, d]),
                tf.broadcast_to(dB, [T, N, d, d]), canonical_options=options)
        else:
            result = _run_resampling_program(
                model, theta, initial, tf.broadcast_to(P, [N, d, d]), noise,
                observations, tf.broadcast_to(B, [T, d, d]),
                tf.broadcast_to(dB, [T, d, d]), canonical_options=options,
                anchor_mode=not replay, fixed_samples=fixed_samples,
                fixed_proposal_log_densities=fixed_proposal_log_densities,
                fixed_component_indices=fixed_component_indices,
                stratified_uniforms=None if replay else uniforms,
                kdm_standard_noises=None if replay else mixture_noise,
                covariance_mark_policy=RESPONSIBILITY_COVARIANCE_MARK_POLICY,
                resampling_kernel=resampling, density_kernel=density,
                rank_tolerance=1e-7 if dtype == tf.float32 else 1e-12,
                normalization_tolerance=normalization_tolerance,
                symmetry_tolerance=normalization_tolerance)
        valid = result["valid"] & tf.math.is_finite(bandwidth_scale) & (bandwidth_scale > 0)
        nan = tf.constant(float("nan"), dtype)
        summary = (tf.where(valid, result["value"], nan),
                   tf.where(valid, result["score"][0], nan), valid)
        if representation == "resampling_kdm":
            return (*summary, result["fixed_samples"], result["fixed_proposal_log_densities"],
                    result["component_indices"])
        return summary
    return kernel
