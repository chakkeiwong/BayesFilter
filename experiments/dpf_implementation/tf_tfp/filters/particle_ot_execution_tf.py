"""Native date and ESS execution shared by the existing particle OT routes.

These routes retain their declared numerical algorithms. Compiling a legacy
LEDH route does not make it canonical or give its AD diagnostic an analytical
score provenance.
"""

from functools import lru_cache

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import (
    DTYPE, normalize_log_weights_tf, weighted_mean_and_variance_tf,
)
from experiments.dpf_implementation.tf_tfp.filters.native_execution_tf import (
    compile_recomputed_diagnostic, conditional_step, host_reports, scan_steps, tensor_diagnostics,
)


@lru_cache(maxsize=8)
def make_particle_ot_filter(observation_spec, particle_spec, *, transition_sample,
    observation_log_density, seed, ess_threshold_ratio, ledh_flow=None,
    transition_log_density=None, jit_compile=True, **transport_settings):
    """Bind a stable compiled complete calculation with tensor-date callbacks."""
    from experiments.dpf_implementation.tf_tfp.filters.dpf_ot_tf import _resample_particles

    horizon, count = observation_spec.shape[0], particle_spec.shape[0]
    if horizon is None or horizon < 1 or count is None:
        raise ValueError("Particle OT requires fixed positive horizon and particle count")
    transport_labels, step_labels = {}, {}

    def resample(points, weights, log_weights, date):
        result, method = _resample_particles(particles=points, weights=weights,
            log_weights=log_weights, time_index=date, **transport_settings)
        diagnostics = tensor_diagnostics(result.diagnostics, transport_labels)
        diagnostics["valid"] = (diagnostics.get("valid", tf.constant(True))
            & diagnostics.get("finite_transport", tf.constant(True))
            & diagnostics.get("finite_particles", tf.constant(True)))
        transport_labels["resampling_method"] = method
        return result.particles, diagnostics

    def skip(specs, particles, weights, log_weights, date):
        diag = tf.nest.map_structure(lambda spec: tf.zeros(spec.shape, spec.dtype), specs[1])
        diag["valid"] = tf.constant(True)
        return particles, diag

    resample, schemas = conditional_step(resample, [particle_spec,
        tf.TensorSpec([count], DTYPE), tf.TensorSpec([count], DTYPE),
        tf.TensorSpec([], tf.int32)], jit_compile=jit_compile, fallback=skip)

    def evaluate(observations, particles):
        uniform = tf.fill([count], -tf.math.log(tf.cast(count, DTYPE)))

        def step(date, state):
            particles, log_weights, total = state
            proposed = tf.cast(transition_sample(particles, seed, date), DTYPE)
            diagnostics = {}
            if ledh_flow is None:
                particles = proposed
                corrected = log_weights + tf.cast(
                    observation_log_density(particles, observations[date], date), DTYPE)
            else:
                flow = ledh_flow(proposed, particles, observations[date], date)
                target_transition = tf.cast(transition_log_density(flow.post_flow_particles, particles, date), DTYPE)
                target_observation = tf.cast(observation_log_density(flow.post_flow_particles, observations[date], date), DTYPE)
                corrected = log_weights + target_transition + target_observation - flow.pre_flow_log_density + flow.forward_log_det
                particles = flow.post_flow_particles
                diagnostics.update(tensor_diagnostics(flow.diagnostics, step_labels))
                diagnostics.update(finite_target_transition=tf.reduce_all(tf.math.is_finite(target_transition)),
                    finite_target_observation=tf.reduce_all(tf.math.is_finite(target_observation)),
                    max_abs_corrected_log_weight=tf.reduce_max(tf.abs(corrected)))
                step_labels["pfpf_correction"] = "log_target_transition_plus_observation_minus_q0_plus_forward_logdet"
            weights, increment = normalize_log_weights_tf(corrected)
            ess = 1. / tf.reduce_sum(weights * weights)
            mean, variance = weighted_mean_and_variance_tf(particles, weights)
            normalized = tf.math.log(tf.maximum(weights, tf.constant(1e-300, DTYPE)))
            trigger = ess < ess_threshold_ratio * count

            particles, transport = resample(trigger, particles, weights, normalized, date)
            diagnostics.update(time_index=date, ess=ess, ess_ratio=ess / count, resampled=trigger,
                finite_corrected_log_weights=tf.reduce_all(tf.math.is_finite(corrected)),
                min_corrected_log_weight=tf.reduce_min(corrected), max_corrected_log_weight=tf.reduce_max(corrected))
            return (particles, tf.where(trigger, uniform, normalized), total + increment), (
                mean, variance, ess, diagnostics, transport)

        state, history = scan_steps(step, (particles, uniform, tf.constant(0., DTYPE)), horizon)
        return state[2], history

    call = compile_recomputed_diagnostic(evaluate, [observation_spec, particle_spec], jit_compile=jit_compile)
    call.step_metadata, call.transport_metadata = step_labels, transport_labels
    return call


def particle_ot_reports(history, call, *, jit_compile):
    """Serialize completed histories and enforce the original numerical vetoes."""
    reports = host_reports(history[3], call.step_metadata)
    transport = host_reports(history[4], call.transport_metadata)
    for report, row in zip(reports, transport):
        if report["resampled"]:
            report.update(row)
        else:
            report.update(resampling_method="none", resampling_status="no_resampling_ess_above_threshold")
        report.update(backend="tensorflow", jit_compile=jit_compile, time_loop_route="tf_while_loop",
                      canonical_algorithm_admitted=False)
        if not (row["valid"] and report["finite_corrected_log_weights"] and report.get("valid_flow", True)):
            raise FloatingPointError("Particle OT filter violated a flow, weight, or resampling veto")
    return reports
