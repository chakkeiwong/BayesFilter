"""Native refinement/terminal lifecycle with fixed-signature numerical steps.

Callback results retain complete tensor records for post-execution reporting.
The controller preserves the original asymmetric terminal budget checks and
the final terminal-fit attempt, including its original lack of recentering.
"""

from functools import lru_cache

import tensorflow as tf

D = tf.float64


@lru_cache(maxsize=32)
def lifecycle_program(refine, terminal, dimension, config, search_count, *, jit_compile=True):
    refine_shape = refine.get_concrete_function().structured_outputs
    terminal_shape = terminal.get_concrete_function().structured_outputs
    # The public route resolves its existing dimension-scaled selector before
    # graph construction and passes the identical count to the numerical step.
    fit_count = (config.structured_fresh_sample_multiplier * dimension
        if config.refinement_geometry_policy == 'factor_correlation' else config.regression_sample_count)
    reserve = config.structured_max_factors if config.refinement_geometry_policy == 'factor_correlation' else 1
    required = search_count + fit_count + reserve

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([], D),
        tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D), tf.TensorSpec([], tf.int64)],
        jit_compile=jit_compile, autograph=False)
    def execute(center, value, score, scale, evaluations):
        terminal_zero = tf.nest.map_structure(lambda v: tf.zeros(v.shape, v.dtype), terminal_shape)
        refine_zero = tf.nest.map_structure(lambda v: tf.zeros(v.shape, v.dtype), refine_shape)
        record_zero = {'event': tf.constant(0), 'terminal_called': tf.constant(False),
            'center_before': center, 'value_before': value,
            'score_before': score, 'radius_before': tf.constant(config.initial_radius, D),
            'evaluations_before': evaluations, 'terminal': terminal_zero, 'refine': refine_zero}
        history = tf.nest.map_structure(lambda v: tf.TensorArray(v.dtype, size=config.max_attempts,
            element_shape=v.shape, clear_after_read=False), record_zero)

        def initialize(index, history):
            return index + 1, tf.nest.map_structure(lambda rows, v: rows.write(index, v), history, record_zero)

        _, history = tf.while_loop(lambda i, _: i < config.max_attempts, initialize,
            (tf.constant(0), history), maximum_iterations=config.max_attempts, parallel_iterations=1)
        state = {'center': center, 'value': value, 'score': score,
            'radius': tf.constant(config.initial_radius, D), 'stalled': tf.constant(0),
            'evaluations': evaluations, 'terminal_attempts': tf.constant(0),
            'has_terminal': tf.constant(False), 'terminal': terminal_zero,
            'stop': tf.constant(False), 'early_status': tf.constant(0)}

        def terminal_step(state, seed, recenter):
            fitted = terminal(state['center'], state['score'], scale, state['radius'], seed)
            moved = (fitted['has_best'] & (fitted['best_value'] > state['value'])) if recenter else tf.constant(False)
            updated = {**state, 'terminal': fitted, 'has_terminal': ~moved,
                'terminal_attempts': state['terminal_attempts'] + 1,
                'evaluations': state['evaluations'] + tf.cast(fitted['evaluations'], tf.int64),
                'center': tf.where(moved, fitted['best_position'], state['center']),
                'value': tf.where(moved, fitted['best_value'], state['value']),
                'score': tf.where(moved, fitted['best_score'], state['score']),
                'stalled': tf.where(moved, 0, state['stalled'])}
            return updated, fitted, moved

        def step(index, state, history):
            row = {**record_zero, 'center_before': state['center'], 'value_before': state['value'],
                'score_before': state['score'], 'radius_before': state['radius'],
                'evaluations_before': state['evaluations']}

            def stationary():
                limit = (state['terminal_attempts'] >= config.max_terminal_fit_attempts
                    if config.max_terminal_fit_attempts is not None else tf.constant(False))
                budget = (state['evaluations'] + config.terminal_sample_count > config.max_exact_evaluations
                    if config.refinement_geometry_policy == 'factor_correlation' else tf.constant(False))

                def fit():
                    seed = tf.constant(config.seed, tf.int32) + tf.stack([0, 100003 + index])
                    updated, fitted, moved = terminal_step(state, seed, True)
                    rejected = ~moved & ~fitted['usable']
                    radius = tf.where(rejected, state['radius'] * tf.constant(config.shrink_factor, D), state['radius'])
                    veto = ~moved & fitted['usable'] & (fitted['projection'] > tf.constant(
                        config.terminal_projection_relative_frobenius_cap, D))
                    stop = ~moved & (fitted['usable'] | (radius < tf.constant(config.minimum_radius, D)))
                    return ({**updated, 'radius': radius, 'stop': stop,
                        'early_status': tf.where(veto, 3, 0)},
                        {**row, 'event': tf.where(moved, 1, tf.where(rejected, 2, 0)),
                            'terminal_called': tf.constant(True), 'terminal': fitted})

                return tf.cond(limit | budget,
                    lambda: ({**state, 'stop': tf.constant(True),
                        'early_status': tf.where(~limit & budget, 1, 0)}, row), fit)

            def moving():
                def advance():
                    result = refine(state['center'], state['value'], state['score'], scale,
                        state['radius'], state['stalled'], index)
                    return ({**state, 'center': result['center'], 'value': result['center_value'],
                        'score': result['center_score'], 'radius': result['radius_after'],
                        'stalled': result['stalled'],
                        'evaluations': state['evaluations'] + tf.cast(result['evaluations'], tf.int64),
                        'stop': result['stop']}, {**row, 'event': tf.constant(3), 'refine': result})

                return tf.cond(state['evaluations'] + required > config.max_exact_evaluations,
                    lambda: ({**state, 'stop': tf.constant(True), 'early_status': tf.constant(2)}, row), advance)

            updated, row = tf.cond(tf.reduce_max(tf.abs(scale * state['score'])) <= tf.constant(
                config.terminal_score_max_abs, D), stationary, moving)
            return index + 1, updated, tf.nest.map_structure(lambda rows, v: rows.write(index, v), history, row)

        count, state, history = tf.while_loop(lambda i, state, _: (i < config.max_attempts) & ~state['stop'],
            step, (tf.constant(0), state, history), maximum_iterations=config.max_attempts, parallel_iterations=1)
        max_score = tf.reduce_max(tf.abs(scale * state['score']))
        below_limit = (state['terminal_attempts'] < config.max_terminal_fit_attempts
            if config.max_terminal_fit_attempts is not None else tf.constant(True))
        finish = ((state['early_status'] == 0) & (max_score <= tf.constant(config.terminal_score_max_abs, D))
            & (~state['has_terminal'] | ~state['terminal']['usable'])
            & (state['evaluations'] + config.terminal_sample_count <= config.max_exact_evaluations) & below_limit)
        state = tf.cond(finish, lambda: terminal_step(state,
            tf.constant([config.seed[0], config.seed[1] + 200003], tf.int32), False)[0], lambda: state)
        failed = (max_score > tf.constant(config.terminal_score_max_abs, D)) | ~state['has_terminal'] | ~state['terminal']['usable']
        projection_veto = state['terminal']['projection'] > tf.constant(config.terminal_projection_relative_frobenius_cap, D)
        status = tf.where(state['early_status'] != 0, state['early_status'],
            tf.where(failed, 4, tf.where(projection_veto, 3, 0)))
        return {**state, 'status': status, 'max_score': max_score, 'attempt_count': count,
            'final_terminal_attempt': finish, 'history': tf.nest.map_structure(lambda rows: rows.stack(), history)}

    return execute
