"""Default-XLA C2 K=1 branch-construction owner."""

from types import SimpleNamespace

import tensorflow as tf

from bayesfilter.highdim.c2_mixture_ukf_apf_tf import (
    DTYPE,
    gaussian_log_density,
    make_batched_ukf_kernel,
    make_k1_apf_sampler_from_random_inputs,
)
from bayesfilter.highdim.c2_transformed_observation_student_proposal_tf import (
    LOG_CHI_SQUARE_VARIANCE,
    _transformed_observation_core,
)
from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import _evaluate_core
from bayesfilter.ops.stateless_random_tf import (
    philox_normal_float64,
    philox_uniform_float64,
)


def make_k1_preparation(model, horizon, count, config, jit_compile=True):
    """Bind static model/configuration, keeping observations/theta/seed live.

    The caller retains the owner in a bounded model cache. Prefix evaluation
    calls the same analytical APF authority with a full-capacity tensor view;
    it neither constructs host artifacts nor changes the prefix recurrence.
    """
    dimension = model.state_dim()

    @tf.function(input_signature=(
        tf.TensorSpec([horizon, dimension], DTYPE),
        tf.TensorSpec([2], DTYPE), tf.TensorSpec([], tf.int64),
    ), jit_compile=jit_compile, autograph=False)
    def prepare(observed, theta, seed):
        def words(offset):
            return tf.stack([seed, tf.cast(offset, tf.int64)])

        transition = model.transition_matrix(theta)
        process_covariance = tf.eye(dimension, dtype=DTYPE) * float(model.sigma)**2
        stationary, _ = model.stationary_covariance_and_derivative(theta)
        initial_cholesky = tf.linalg.cholesky(stationary)
        initial_normal = philox_normal_float64([count, dimension], words(1001))
        initial_states = tf.einsum('ij,nj->ni', initial_cholesky, initial_normal)
        initial_log_q = model.initial_log_density(theta, initial_states)
        covariances = tf.broadcast_to(stationary[None], [count, dimension, dimension])
        process = tf.broadcast_to(process_covariance[None], [count, dimension, dimension])
        observation_covariance = tf.eye(dimension, dtype=DTYPE) * tf.constant(LOG_CHI_SQUARE_VARIANCE, DTYPE)
        noise = tf.broadcast_to(observation_covariance[None], [count, dimension, dimension])

        def transition_fn(points):
            return tf.einsum('ij,bpj->bpi', transition, points)

        kernel = make_batched_ukf_kernel(batch_size=count, state_dim=dimension,
            observation_dim=dimension, transition_fn=transition_fn,
            observation_fn=tf.identity, config=config, jit_compile=jit_compile)
        sampler = make_k1_apf_sampler_from_random_inputs(batch_size=count,
            state_dim=dimension, jit_compile=jit_compile)
        states = tf.tensor_scatter_nd_update(tf.zeros([horizon, count, dimension], DTYPE),
            [[0]], initial_states[None])
        ancestors = tf.zeros([horizon-1, count], tf.int32)
        auxiliary = tf.zeros([horizon-1, count], DTYPE)
        log_q = tf.zeros([horizon-1, count], DTYPE)
        initial_mass = tf.fill([count], -tf.math.log(tf.cast(count, DTYPE)))
        transition_mass = tf.fill([horizon-1, count], -tf.math.log(tf.cast(count, DTYPE)))

        def prefix(time, particles, genealogy, ancestor_log, proposal_log):
            # Internal tensor view only; the public branch/fingerprint is built
            # after completion and the original validation boundary is retained.
            branch = SimpleNamespace(dtype=DTYPE, particle_count=count,
                time_steps=horizon, observations=observed, states=particles,
                initial_log_proposal_density=initial_log_q, ancestors=genealogy,
                auxiliary_log_probabilities=ancestor_log,
                transition_log_proposal_density=proposal_log,
                initial_log_base_mass=initial_mass,
                transition_log_base_mass=transition_mass)
            return _evaluate_core(model, branch, theta, evaluation_steps=time)

        initial = prefix(tf.constant(1), states, ancestors, auxiliary, log_q)
        # Match host preparation validation before its exact-prefix finite test.
        status = tf.where(tf.reduce_all(tf.math.is_finite(observed[0])), 0, 1)
        status = tf.where((status == 0) & ~tf.reduce_all(tf.math.is_finite(initial_states)), 2, status)
        status = tf.where((status == 0) & ~tf.reduce_all(tf.math.is_finite(initial_log_q)), 3, status)
        status = tf.where((status == 0) & ~initial['finite'], 4, status)
        diagnostics = {
            'time_index': tf.zeros([horizon-1], tf.int32),
            'transformed_observation': tf.zeros([horizon-1, dimension], DTYPE),
            'lookahead_log_likelihood': tf.zeros([horizon-1, count], DTYPE),
            'ancestor_log_probabilities': tf.zeros([horizon-1, count], DTYPE),
            'selected_mean': tf.zeros([horizon-1, count, dimension], DTYPE),
            'selected_cholesky': tf.zeros([horizon-1, count, dimension, dimension], DTYPE),
            'selected_log_q': tf.zeros([horizon-1, count], DTYPE),
            'proposal_density_recomposition_max_abs': tf.zeros([horizon-1], DTYPE),
            'posterior_mean': tf.zeros([horizon-1, count, dimension], DTYPE),
            'posterior_min_eigenvalue': tf.zeros([horizon-1, count], DTYPE),
            'proposal_finite': tf.zeros([horizon-1], tf.bool),
            'exact_prefix_finite': tf.zeros([horizon-1], tf.bool),
        }

        def step(time, particles, genealogy, ancestor_log, proposal_log, covariance,
                 log_weights, diagnostic_rows, status, failed_time, invalid_rows):
            transformed = _transformed_observation_core(observed[time], theta)
            ukf = kernel(particles[time-1], covariance, process, transformed, noise)
            uniform = philox_uniform_float64([count], words(5100+41*time))
            normal = philox_normal_float64([count, dimension], words(5200+43*time))
            sampled = sampler(ukf['posterior_mean'], ukf['posterior_covariance'],
                ukf['posterior_cholesky'], ukf['innovation_log_likelihood'],
                log_weights, uniform, normal)
            particles = tf.tensor_scatter_nd_update(particles, tf.reshape(time, [1,1]), sampled['samples'][None])
            index = tf.reshape(time-1, [1,1])
            genealogy = tf.tensor_scatter_nd_update(genealogy, index, sampled['ancestor_indices'][None])
            ancestor_log = tf.tensor_scatter_nd_update(ancestor_log, index, sampled['log_ancestor_probabilities'][None])
            proposal_log = tf.tensor_scatter_nd_update(proposal_log, index, sampled['complete_log_q'][None])
            exact = prefix(time+1, particles, genealogy, ancestor_log, proposal_log)
            row = {
                'time_index': time,
                'transformed_observation': transformed,
                'lookahead_log_likelihood': ukf['innovation_log_likelihood'],
                'ancestor_log_probabilities': sampled['log_ancestor_probabilities'],
                'selected_mean': sampled['selected_mean'],
                'selected_cholesky': sampled['selected_cholesky'],
                'selected_log_q': sampled['selected_log_q'],
                'proposal_density_recomposition_max_abs': tf.reduce_max(tf.abs(sampled['complete_log_q']
                    - gaussian_log_density(sampled['samples'], sampled['selected_mean'], sampled['selected_cholesky']))),
                'posterior_mean': ukf['posterior_mean'],
                'posterior_min_eigenvalue': ukf['posterior_min_eigenvalue'],
                'proposal_finite': sampled['finite'],
                'exact_prefix_finite': exact['finite'],
            }
            diagnostic_rows = tf.nest.map_structure(
                lambda history, value: tf.tensor_scatter_nd_update(history, index, value[None]),
                diagnostic_rows, row)
            status = tf.where(tf.reduce_any(observed[time] == 0.), 10, 0)
            status = tf.where((status == 0) & ~tf.reduce_all(tf.math.is_finite(transformed)), 11, status)
            status = tf.where((status == 0) & ~ukf['valid'], 5, status)
            status = tf.where((status == 0) & ~sampled['finite'], 6, status)
            status = tf.where((status == 0) & ~tf.reduce_all(tf.math.is_finite(sampled['log_ancestor_probabilities'])), 7, status)
            status = tf.where((status == 0) & (tf.abs(tf.reduce_logsumexp(sampled['log_ancestor_probabilities'])) > 1e-10), 8, status)
            status = tf.where((status == 0) & ~exact['finite'], 9, status)
            invalid_rows = tf.reduce_sum(tf.cast(~ukf['valid_rows'], tf.int32))
            return (time+1, particles, genealogy, ancestor_log, proposal_log,
                sampled['next_covariances'], exact['final_log_weights'], diagnostic_rows,
                status, tf.where(status == 0, failed_time, time), invalid_rows)

        result = tf.while_loop(lambda time, *args: (time < horizon) & (args[-3] == 0),
            step, (tf.constant(1), states, ancestors, auxiliary, log_q, covariances,
                initial['final_log_weights'], diagnostics, status, tf.constant(0), tf.constant(0)),
            maximum_iterations=horizon-1, parallel_iterations=1)
        return {'states': result[1], 'ancestors': result[2],
            'auxiliary_log_probabilities': result[3], 'transition_log_proposal_density': result[4],
            'initial_log_proposal_density': initial_log_q, 'diagnostics': result[7],
            'status': result[8], 'failed_time': result[9], 'invalid_rows': result[10],
            'process_covariance': process_covariance, 'observation_covariance': observation_covariance}

    return prepare
