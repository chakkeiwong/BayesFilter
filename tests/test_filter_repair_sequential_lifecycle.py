"""Explicit lifecycle boundary fixtures; callbacks isolate controller mechanics."""

import dataclasses

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_lifecycle_tf as native
from bayesfilter.inference.sequential_map_covariance import (
    SequentialMapCovarianceConfig,
)

D = tf.float64


def _program(config, terminal_usable=(True, True), projections=(0., 0.),
             best=(False, False), terminal_scores=(0., 0.), refined_score=0., refine_stop=False):
    # TensorFlow places int32 variables on CPU even for GPU kernels. int64
    # observation resources follow device placement and can be captured by XLA.
    terminal_calls = tf.Variable(0, dtype=tf.int64, trainable=False)
    refine_calls = tf.Variable(0, dtype=tf.int64, trainable=False)
    total = tf.Variable(0, dtype=tf.int64, trainable=False)
    order = tf.Variable(tf.zeros([8], tf.int64), trainable=False)
    seeds = tf.Variable(tf.zeros([8, 2], tf.int64), trainable=False)

    @tf.function(input_signature=[tf.TensorSpec([3], D), tf.TensorSpec([3], D),
        tf.TensorSpec([3], D), tf.TensorSpec([], D), tf.TensorSpec([2], tf.int32)],
        jit_compile=True, autograph=False)
    def terminal(center, score, scale, radius, seed):
        index = terminal_calls.read_value()
        terminal_calls.assign_add(1)
        seeds.scatter_nd_update(tf.reshape(index, [1, 1]), tf.cast(seed[None, :], tf.int64))
        order.scatter_nd_update(tf.reshape(total.read_value(), [1, 1]), tf.constant([1], tf.int64))
        total.assign_add(1)
        return {'usable': tf.gather(tf.constant(terminal_usable), index),
            'projection': tf.gather(tf.constant(projections, D), index),
            'has_best': tf.gather(tf.constant(best), index),
            'best_value': tf.constant(2., D), 'best_position': center + 1.,
            'best_score': tf.fill([3], tf.gather(tf.constant(terminal_scores, D), index)),
            'evaluations': tf.constant(config.terminal_sample_count)}

    @tf.function(input_signature=[tf.TensorSpec([3], D), tf.TensorSpec([], D),
        tf.TensorSpec([3], D), tf.TensorSpec([3], D), tf.TensorSpec([], D),
        tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.int32)], jit_compile=True, autograph=False)
    def refine(center, value, score, scale, radius, stalled, index):
        refine_calls.assign_add(1)
        order.scatter_nd_update(tf.reshape(total.read_value(), [1, 1]), tf.constant([2], tf.int64))
        total.assign_add(1)
        return {'center': center + .1, 'center_value': value + 1.,
            'center_score': tf.fill([3], tf.constant(refined_score, D)),
            'radius_after': radius, 'stalled': stalled + 1, 'evaluations': tf.constant(29),
            'stop': tf.constant(refine_stop)}

    program = native.lifecycle_program(refine, terminal, 3, config, config.search_sample_count)
    return program, terminal_calls, refine_calls, order, seeds


# Expected decisions come from the original public loop's explicit exit and
# budget boundaries, not from a second implementation of the tensor controller.
@pytest.mark.parametrize('case', [
    'healthy_terminal', 'projection_veto', 'terminal_limit', 'terminal_retry_final',
    'terminal_recenter_then_refine', 'terminal_budget_factor', 'terminal_budget_full',
    'refine_budget', 'minimum_radius_final_retry', 'refine_stop_stationary', 'refine_stop_moving',
    'final_fit_winner_is_not_recentered',
    'budget_equal_above_int32', 'budget_exceeded_above_int32',
])
def test_lifecycle_order_budgets_retries_and_terminal_seeds(case):
    cfg = SequentialMapCovarianceConfig(max_attempts=2, terminal_sample_count=24,
        regression_sample_count=24, search_sample_count=4, max_exact_evaluations=100)
    options, score, evaluations = {}, 0., 0
    status, events, expected_seeds, terminal_count, expected_value = 0, [1], [100718], 1, 0.
    if case == 'projection_veto':
        options['projections'] = (1., 0.)
        status = 3
    elif case == 'terminal_limit':
        cfg = dataclasses.replace(cfg, max_terminal_fit_attempts=1)
        options['terminal_usable'] = (False, True)
        status = 4
    elif case in ('terminal_retry_final', 'minimum_radius_final_retry'):
        cfg = dataclasses.replace(cfg, max_attempts=1,
            minimum_radius=.2 if case == 'minimum_radius_final_retry' else .0001)
        options['terminal_usable'] = (False, True)
        events, expected_seeds, terminal_count = [1, 1], [100718, 200718], 2
    elif case == 'terminal_recenter_then_refine':
        options.update(best=(True, False), terminal_scores=(.5, 0.))
        events, expected_seeds, terminal_count, expected_value = [1, 2, 1], [100718, 200718], 2, 3.
    elif case in ('terminal_budget_factor', 'terminal_budget_full'):
        evaluations = 77
        if case == 'terminal_budget_factor':
            cfg = dataclasses.replace(cfg, refinement_geometry_policy='factor_correlation')
            status, events, expected_seeds, terminal_count = 1, [], [], 0
    elif case == 'refine_budget':
        score, evaluations = .5, 80
        status, events, expected_seeds, terminal_count = 2, [], [], 0
    elif case in ('budget_equal_above_int32', 'budget_exceeded_above_int32'):
        evaluations = 2 ** 31 - 10
        exceeds = case == 'budget_exceeded_above_int32'
        cfg = dataclasses.replace(cfg, refinement_geometry_policy='factor_correlation',
            max_exact_evaluations=evaluations + 24 - int(exceeds))
        if exceeds:
            status, events, expected_seeds, terminal_count = 1, [], [], 0
    elif case in ('refine_stop_stationary', 'refine_stop_moving', 'final_fit_winner_is_not_recentered'):
        score = .5
        options['refine_stop'] = True
        events, expected_seeds, expected_value = [2, 1], [200718], 1.
        if case == 'refine_stop_moving':
            options['refined_score'] = .5
            status, events, expected_seeds, terminal_count = 4, [2], [], 0
        elif case == 'final_fit_winner_is_not_recentered':
            options['best'] = (True, False)
    program, terminal_calls, refine_calls, order, seeds = _program(cfg, **options)
    result = program(tf.zeros([3], D), tf.constant(0., D), tf.fill([3], tf.constant(score, D)),
        tf.ones([3], D), tf.constant(evaluations, tf.int64))
    assert int(result['status']) == status
    assert int(terminal_calls) == terminal_count
    assert int(result['terminal_attempts']) == terminal_count
    assert int(tf.reduce_sum(tf.cast(result['history']['terminal_called'], tf.int32))) == (
        terminal_count - int(result['final_terminal_attempt']))
    assert int(refine_calls) == events.count(2)
    assert int(result['evaluations']) == evaluations + 24 * terminal_count + 29 * events.count(2)
    assert float(result['value']) == expected_value
    np.testing.assert_array_equal(order[:len(events)], events)
    np.testing.assert_array_equal(seeds[:terminal_count, 0], [2026] * terminal_count)
    np.testing.assert_array_equal(seeds[:terminal_count, 1], expected_seeds)
    assert program.experimental_get_tracing_count() == 1
