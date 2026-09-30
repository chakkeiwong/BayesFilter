"""Native equal-bank C2 DMIS step using the shared proposal authorities."""

from functools import partial

import tensorflow as tf

from bayesfilter.highdim.c2_independent_preparation_tf import _hermite_tensor_view, _sample_configuration
from bayesfilter.highdim.c2_student_preparation_tf import sample_student
from bayesfilter.highdim.c2_transformed_observation_student_proposal_tf import _student_tensor_view
from bayesfilter.ops.stateless_random_tf import philox_uniform_float64

D = tf.float64


def _dispatch(time, configurations, indices, groups, evaluate):
    def execute(configuration, packed):
        payload = tf.nest.map_structure(lambda value: value[time-1], packed)
        return evaluate(configuration, payload)
    functions = tuple(partial(execute, configuration, packed)
                      for configuration, packed in zip(configurations, groups, strict=True))
    return tf.switch_case(indices[time-1], branch_fns=functions)


def diagnostic_specs():
    return {'alpha': tf.TensorSpec([], D), 'nu': tf.TensorSpec([], D),
            'tt_bank_count': tf.TensorSpec([], tf.int32),
            'defensive_bank_count': tf.TensorSpec([], tf.int32),
            'tt_finite': tf.TensorSpec([], tf.bool), 'defensive_finite': tf.TensorSpec([], tf.bool),
            'mixture_finite': tf.TensorSpec([], tf.bool), 'finite': tf.TensorSpec([], tf.bool)}


def dmis_step(retained_configurations, student_configurations, count):
    bank_count = count // 2

    def sample(time, previous_states, auxiliary, seed, theta, operands):
        del theta
        retained, defensive, alpha, reported_nu = operands
        retained_indices, _, retained_groups = retained
        student_indices, student_groups = defensive

        def words(offset):
            return tf.stack([seed, tf.cast(offset, tf.int64)])

        cdf = tf.math.cumsum(tf.exp(auxiliary))
        cdf = tf.concat([cdf[:-1], tf.ones([1], D)], axis=0)
        ancestor_tt = tf.searchsorted(cdf, philox_uniform_float64(
            [bank_count], words(2000+37*time)), side='right', out_type=tf.int32)
        ancestor_defensive = tf.searchsorted(cdf, philox_uniform_float64(
            [bank_count], words(2100+37*time)), side='right', out_type=tf.int32)
        parents_tt = tf.gather(previous_states, ancestor_tt)
        parents_defensive = tf.gather(previous_states, ancestor_defensive)

        tt_states, tt_log_q, tt_diagnostic = _dispatch(
            time, retained_configurations, retained_indices, retained_groups,
            lambda configuration, payload: _sample_configuration(
                configuration, payload, bank_count, words(3000+41*time)))
        defensive_sampled = _dispatch(
            time, student_configurations, student_indices, student_groups,
            lambda configuration, payload: sample_student(
                configuration, payload, bank_count, parents_defensive, words(4000+43*time)))
        defensive_states = defensive_sampled['physical_points']
        defensive_log_q = defensive_sampled['physical_log_density']
        tt_log_defensive = _dispatch(
            time, student_configurations, student_indices, student_groups,
            lambda configuration, payload: _student_tensor_view(configuration[1], payload).log_density(
                tt_states, parents_tt))
        defensive_log_tt = _dispatch(
            time, retained_configurations, retained_indices, retained_groups,
            lambda configuration, payload: _hermite_tensor_view(configuration, payload).physical_log_density(
                defensive_states))
        log_alpha = tf.math.log(alpha)
        log_one_minus_alpha = tf.math.log(1.-alpha)
        log_bank_count = tf.math.log(tf.cast(bank_count, D))
        tt_mixture_log_q = tf.reduce_logsumexp(tf.stack(
            [log_one_minus_alpha+tt_log_q, log_alpha+tt_log_defensive], axis=1), axis=1)
        defensive_mixture_log_q = tf.reduce_logsumexp(tf.stack(
            [log_one_minus_alpha+defensive_log_tt, log_alpha+defensive_log_q], axis=1), axis=1)
        density = tf.concat([tt_mixture_log_q, defensive_mixture_log_q], axis=0)
        mass = tf.concat([tf.fill([bank_count], log_one_minus_alpha-log_bank_count),
                          tf.fill([bank_count], log_alpha-log_bank_count)], axis=0)
        mixture_finite = tf.reduce_all(tf.math.is_finite(density))
        row = {'alpha': alpha, 'nu': reported_nu,
               'tt_bank_count': tf.constant(bank_count, tf.int32),
               'defensive_bank_count': tf.constant(bank_count, tf.int32),
               'tt_finite': tt_diagnostic['finite'],
               'defensive_finite': defensive_sampled['finite'],
               'mixture_finite': mixture_finite,
               'finite': tt_diagnostic['finite'] & defensive_sampled['finite'] & mixture_finite}
        return tf.concat([tt_states, defensive_states], axis=0), density, row, tf.concat(
            [ancestor_tt, ancestor_defensive], axis=0), mass
    return sample
