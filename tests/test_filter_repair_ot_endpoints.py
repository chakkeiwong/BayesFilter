"""Full endpoint diagnostics against an independent host date recurrence."""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import (
    normalize_log_weights_tf, weighted_mean_and_variance_tf,
)
from experiments.dpf_implementation.tf_tfp.filters.dpf_ot_tf import run_ot_dpf_tf, _resample_particles
from experiments.dpf_implementation.tf_tfp.filters.ledh_pfpf_ot_tf import run_ledh_pfpf_ot_tf
from experiments.dpf_implementation.tf_tfp.flows.ledh_tf import ledh_flow_batch_tf, gaussian_logpdf_tf
from experiments.dpf_implementation.tf_tfp.structural.contracts_tf import StructuralFlowProposalTF
from experiments.dpf_implementation.tf_tfp.structural.structural_filter_tf import (
    run_structural_ledh_pfpf_tf, make_structural_filter_tf,
)
from experiments.dpf_implementation.tf_tfp.structural.resampling_policies_tf import (
    SUPPORTED_POLICIES, apply_structural_resampling_policy_tf,
)
from experiments.dpf_implementation.tf_tfp.structural.particle_state_tf import StructuralParticleStateTF

D = tf.float64


def callbacks(scale):
    initial = lambda n, seed: tf.reshape(tf.linspace(tf.constant(-.2, D), tf.constant(.3, D), n), [-1, 1])
    transition = lambda x, seed, date: x * scale + tf.cast(date + 1, D) * .03
    transition_log = lambda x, old, date: gaussian_logpdf_tf(x - old * scale, tf.constant([[.2]], D))
    observation_log = lambda x, y, date: gaussian_logpdf_tf(x - y, tf.constant([[.15]], D))
    def flow(x, old, y, date):
        return ledh_flow_batch_tf(pre_flow_particles=x, ancestors=old, observation=y,
            transition_matrix=tf.reshape(scale, [1, 1]), transition_covariance=tf.constant([[.2]], D),
            observation_covariance=tf.constant([[.15]], D), observation_fn=lambda x: x,
            observation_jacobian_fn=lambda x: tf.ones([1, 1], D), observation_residual_fn=lambda predicted, y: y - predicted)
    return initial, transition, transition_log, observation_log, flow


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("ledh", [False, True])
@pytest.mark.parametrize("threshold", [0., 1.1])
@pytest.mark.parametrize("method", ["fixed_target_sinkhorn", "annealed_transport"])
def test_ot_endpoint_matches_host_recurrence_and_preserves_gradient(jit, ledh, threshold, method):
    scale = tf.Variable(.72, dtype=D)
    initial, transition, transition_log, observation_log, flow = callbacks(scale)
    observations = tf.constant([[.1], [.2], [-.1]], D)
    transport = dict(transport_method=method, epsilon=.8, sinkhorn_iterations=20,
        sinkhorn_tolerance=1e-6, annealed_scaling=.9, annealed_convergence_threshold=1e-3,
        retained_teacher_warmstart_fn=None)
    with tf.GradientTape() as tape:
        tape.watch(observations)
        points = initial(4, 17)
        uniform = tf.fill([4], -tf.math.log(tf.constant(4., D)))
        log_weights, total = uniform, tf.constant(0., D)
        means, variances, esses, triggers = [], [], [], []
        for date, observation in enumerate(tf.unstack(observations)):
            old = points
            points = transition(points, 17, date)
            if ledh:
                result = flow(points, old, observation, date)
                points = result.post_flow_particles
                correction = transition_log(points, old, date) - result.pre_flow_log_density + result.forward_log_det
            else:
                correction = 0.
            weights, increment = normalize_log_weights_tf(log_weights + observation_log(points, observation, date) + correction)
            total += increment
            mean, variance = weighted_mean_and_variance_tf(points, weights)
            ess = 1. / tf.reduce_sum(weights**2)
            means.append(mean)
            variances.append(variance)
            esses.append(ess)
            trigger = bool(ess < threshold * 4)
            triggers.append(trigger)
            log_weights = tf.math.log(tf.maximum(weights, tf.constant(1e-300, D)))
            if trigger:
                points = _resample_particles(particles=points, weights=weights, log_weights=log_weights,
                    time_index=date, **transport)[0].particles
                log_weights = uniform
    expected_gradient = tape.gradient(total, (scale, observations))
    with tf.GradientTape() as tape:
        tape.watch(observations)
        kwargs = dict(observations=observations, initial_sample=initial, transition_sample=transition,
            observation_log_density=observation_log, seed=17, num_particles=4, ess_threshold_ratio=threshold,
            sinkhorn_epsilon=.8, sinkhorn_iterations=20, sinkhorn_tolerance=1e-6,
            transport_method=method, jit_compile=jit)
        actual = run_ledh_pfpf_ot_tf(**kwargs, ledh_flow=flow, transition_log_density=transition_log) if ledh else run_ot_dpf_tf(**kwargs)
    actual_gradient = tape.gradient(actual.log_likelihood_estimate, (scale, observations))
    for value, expected in zip((actual.log_likelihood_estimate, actual.filtered_means, actual.filtered_variances, actual.ess_by_time),
                               (total, tf.stack(means), tf.stack(variances), tf.stack(esses))):
        np.testing.assert_allclose(value, expected, atol=1e-10, rtol=1e-10)
    for actual_component, expected_component in zip(actual_gradient, expected_gradient):
        assert actual_component is not None
        np.testing.assert_allclose(actual_component, expected_component, atol=1e-10, rtol=1e-10)
    assert actual.resampling_count == sum(triggers)
    assert actual.finite


def test_structural_factory_reuses_callback_and_shape_binding():
    # Construction is independent of observation values; a different callback
    # identity must never reuse the previous model's cached graph.
    specs = (tf.TensorSpec([3, 1], D), tf.TensorSpec([4, 1], D), tf.TensorSpec([4, 1], D))
    _, transition, transition_log, observation_log, _ = callbacks(tf.constant(.72, D))
    complete = lambda s, old, current: s + current - old
    residual = lambda s, old, current, new: new-complete(s, old, current)
    flow = lambda z, old, s, y, date: StructuralFlowProposalTF(z, transition_log(z, old, date), tf.zeros([4], D), {})
    callbacks_a = (transition, flow, complete, transition_log,
                   lambda z, s, y, date: observation_log(z+s, y, date), residual)
    callbacks_b = (lambda x, seed, date: transition(x, seed, date), *callbacks_a[1:])
    options = dict(seed=17, resampling_policy_id=SUPPORTED_POLICIES[0])
    first = make_structural_filter_tf(*specs, callbacks_a, **options)
    assert first is make_structural_filter_tf(*specs, callbacks_a, **options)
    assert first is not make_structural_filter_tf(*specs, callbacks_b, **options)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("policy", SUPPORTED_POLICIES)
def test_structural_endpoint_preserves_completed_context(policy, jit):
    scale = tf.constant(.72, D)
    initial, transition, transition_log, observation_log, _ = callbacks(scale)
    def complete(s, old, current):
        return s + .3 * old + current**2
    model = SimpleNamespace(initial_z_sample=initial, initial_s_from_z=lambda z: z**2,
        transition_z_sample=transition, transition_z_log_prob=transition_log,
        complete_s=complete, completion_residual=lambda s, old, current, new: new - complete(s, old, current),
        observation_log_prob=lambda z, s, y, date: observation_log(z + s, y, date),
        local_flow_proposal=lambda z, old, s, y, date: StructuralFlowProposalTF(
            z, transition_log(z, old, date), tf.zeros([4], D), {}))
    observations = tf.constant([[.1], [.2], [-.1]], D)
    z = initial(4, 17)
    s = model.initial_s_from_z(z)
    logs, total, means = tf.fill([4], -tf.math.log(tf.constant(4., D))), tf.constant(0., D), []
    for date, y in enumerate(tf.unstack(observations)):
        next_z = transition(z, 17, date)
        next_s = complete(s, z, next_z)
        weights, increment = normalize_log_weights_tf(logs + model.observation_log_prob(next_z, next_s, y, date))
        total += increment
        state = StructuralParticleStateTF(z, s, next_z, next_s)
        means.append(weighted_mean_and_variance_tf(state.completed_state(), weights)[0])
        result = apply_structural_resampling_policy_tf(model=model, state=state, weights=weights,
            policy_id=policy, seed=17, time_index=date, sinkhorn_epsilon=.8, sinkhorn_iterations=20, sinkhorn_tolerance=1e-6)
        z, s, logs = result.next_z, result.next_s, result.next_log_weights
    actual = run_structural_ledh_pfpf_tf(model=model, observations=observations, seed=17, num_particles=4,
        resampling_policy_id=policy, sinkhorn_epsilon=.8, sinkhorn_iterations=20, sinkhorn_tolerance=1e-6, jit_compile=jit)
    np.testing.assert_allclose(actual.log_likelihood_estimate, total, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual.filtered_means, tf.stack(means), rtol=1e-10, atol=1e-10)
    assert actual.finite
