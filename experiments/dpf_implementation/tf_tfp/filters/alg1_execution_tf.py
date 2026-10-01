"""Complete native execution of the diagnostic Algorithm-1 finite program.

Callbacks receive tensor dates. Execution repair does not admit canonical LEDH
or confer analytical-score provenance on a finite-program autodiff diagnostic.
"""

from functools import lru_cache

import tensorflow as tf

from bayesfilter.ops.stateless_random_tf import stateless_categorical_cpu_stream
from experiments.dpf_implementation.tf_tfp.filters.native_execution_tf import (
    conditional_step,
    compile_recomputed_diagnostic,
    scan_steps,
    tensor_diagnostics,
)


@lru_cache(maxsize=8)
def make_alg1_filter(observation_spec, particle_spec, pseudo_spec, *, jit_compile=True, **settings):
    """Compile the enclosing date, particle, pseudo-time and ESS recurrence."""
    from experiments.dpf_implementation.tf_tfp.filters import ledh_pfpf_alg1_ukf_tf as alg

    dtype = tf.float64
    horizon = observation_spec.shape[0]
    count, dimension = particle_spec.shape
    if horizon is None or horizon < 1 or count is None or dimension is None:
        raise ValueError("Algorithm 1 requires fixed positive horizon and state shapes")
    route = settings["resampling_route"]
    if route not in {"none", "classical_resampling", alg.OT_ANNEALED_COVARIANCE_CARRY_ROUTE,
                     alg.OT_SINKHORN_COVARIANCE_CARRY_ROUTE}:
        raise ValueError("Algorithm 1 core supports none, classical_resampling, and reviewed P8h OT routes")
    step_keys = ("transition_mean_fn", "transition_log_density_fn", "observation_mean_fn",
                 "observation_jacobian_fn", "process_noise_covariance_fn", "observation_covariance_fn",
                 "alpha", "beta", "kappa", "covariance_floor", "rank_tolerance")
    ot_keys = ("sinkhorn_epsilon", "sinkhorn_iterations", "sinkhorn_tolerance", "sinkhorn_epsilon_policy",
               "annealed_scaling", "annealed_convergence_threshold", "transport_gradient_mode", "covariance_floor")
    step_settings = {key: settings[key] for key in step_keys}
    ot_settings = {key: settings[key] for key in ot_keys}
    step_metadata, transport_metadata = {}, {}

    def resample(points, covariances, weights, log_weights, date):
        if route == "classical_resampling":
            indices = stateless_categorical_cpu_stream(
                tf.math.log(tf.maximum(weights, tf.constant(1e-300, dtype))), count,
                alg._seed_pair(settings["seed"], 9000 + date))
            points, covariances = alg.apply_classical_resampling_state_tf(
                particles=points, covariances=covariances, ancestor_indices=indices)
            diag = {"ancestor_indices": indices, "valid": tf.constant(True)}
            transport_metadata["resampling_method"] = "stateless_multinomial_with_covariance_gather"
        elif route != "none":
            points, covariances, full_diag = alg.apply_ot_resampling_state_tf(
                particles=points, covariances=covariances, weights=weights, log_weights=log_weights,
                resampling_route=route, **ot_settings)
            diag = tensor_diagnostics(full_diag, transport_metadata)
        else:
            diag = {"valid": tf.constant(True)}
        return points, covariances, diag

    def skip(specs, points, covariances, weights, log_weights, date):
        diag = tf.nest.map_structure(lambda spec: tf.zeros(spec.shape, spec.dtype), specs[2])
        diag["valid"] = tf.constant(True)
        return points, covariances, diag

    resample, result_specs = conditional_step(resample, [particle_spec,
        tf.TensorSpec([count, dimension, dimension], dtype), tf.TensorSpec([count], dtype),
        tf.TensorSpec([count], dtype), tf.TensorSpec([], tf.int32)], jit_compile=jit_compile, fallback=skip)
    diagnostic_specs = result_specs[2]

    def evaluate(observations, particles, covariance, pseudo_time):
        uniform = tf.fill([count], -tf.math.log(tf.cast(count, dtype)))
        covariances = tf.broadcast_to(alg.symmetrize(covariance), [count, dimension, dimension])
        pseudo_valid = (tf.reduce_all(tf.math.is_finite(pseudo_time)) & tf.reduce_all(pseudo_time > 0.)
                        & (tf.abs(tf.reduce_sum(pseudo_time) - 1.) <= tf.constant(1e-12, dtype)))

        def step(date, state):
            particles, covariances, log_weights, total = state
            ancestors = particles
            pre_flow = tf.cast(settings["transition_sample"](ancestors, settings["seed"], date), dtype)
            result = alg.li_coates_ledh_alg1_time_step_tf(ancestors=ancestors,
                previous_covariances=covariances, pre_flow_particles=pre_flow,
                observation=observations[date], time_index=date, pseudo_time_steps=pseudo_time,
                graph_reference=not jit_compile, **step_settings)
            particles = result.post_flow_particles
            corrected = (log_weights + settings["transition_log_density_fn"](particles, ancestors, date)
                + settings["observation_log_density_fn"](particles, observations[date], date)
                - result.pre_flow_log_density + result.forward_log_det)
            weights, increment = alg.normalize_log_weights_tf(corrected)
            ess = 1. / tf.reduce_sum(weights * weights)
            mean, variance = alg.weighted_mean_and_variance_tf(particles, weights)
            normalized = tf.math.log(tf.maximum(weights, tf.constant(1e-300, dtype)))
            trigger = (ess < settings["ess_threshold_ratio"] * count) & (route != "none")

            if route == "none":
                particles, covariances, transport = skip(result_specs,
                    particles, result.updated_covariances, weights, normalized, date)
            else:
                particles, covariances, transport = resample(trigger,
                    particles, result.updated_covariances, weights, normalized, date)
            diag = tensor_diagnostics(result.diagnostics, step_metadata)
            diag.update(time_index=date, ess=ess, ess_ratio=ess / count,
                finite_corrected_log_weights=tf.reduce_all(tf.math.is_finite(corrected)),
                min_corrected_log_weight=tf.reduce_min(corrected), max_corrected_log_weight=tf.reduce_max(corrected),
                resampled=trigger, ess_triggered=trigger, valid_pseudo_time=pseudo_valid)
            return (particles, covariances, tf.where(trigger, uniform, normalized), total + increment), (
                mean, variance, result.updated_covariances, result.predicted_covariances, corrected, ess, diag, transport)

        state, history = scan_steps(step, (particles, covariances, uniform, tf.constant(0., dtype)), horizon)
        return state[3], history

    call = compile_recomputed_diagnostic(evaluate, [observation_spec, particle_spec,
        tf.TensorSpec([dimension, dimension], dtype), pseudo_spec], jit_compile=jit_compile)
    call.step_metadata, call.transport_metadata = step_metadata, transport_metadata
    return call
