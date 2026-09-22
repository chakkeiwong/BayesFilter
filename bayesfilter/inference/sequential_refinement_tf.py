"""One native search, fresh fit, exact recenter and ordered proposal step."""


import tensorflow as tf

from bayesfilter.inference.factor_correlation_geometry import (
    FactorCorrelationGeometryConfig,
)
from bayesfilter.inference.factor_decisions_tf import factor_decisions_program
from bayesfilter.inference.program_cache_scope import scoped_program_cache
from bayesfilter.inference.sequential_attempts_tf import attempts_program
from bayesfilter.inference.sequential_factor_attempt_tf import (
    empty_second_program,
    second_factor_program,
)
from bayesfilter.inference.sequential_selection_tf import search_program
from bayesfilter.inference.sequential_structured_fit_tf import (
    structured_fit_data_program,
)
from bayesfilter.inference.sequential_structured_preparation_tf import (
    structured_data_program,
)
from bayesfilter.inference.sequential_terminal_tf import terminal_program

D = tf.float64


@scoped_program_cache(maxsize=32)
def refinement_program(scalar, batched, dimension, config, search_count, *, jit_compile=True):
    search = search_program(scalar, batched, search_count, dimension,
        config.orthogonal_antithetic_search, jit_compile=jit_compile)
    structured = config.refinement_geometry_policy == 'factor_correlation'
    escalate = structured and config.structured_max_factors == 2
    if structured:
        fresh = config.structured_fresh_sample_multiplier * dimension
        capacity = fresh // 2 + (search_count if config.reuse_search_scores else 0)
        preparation = structured_data_program(scalar, batched, dimension, fresh, search_count,
            config.reuse_search_scores, jit_compile=jit_compile)
        factor_cfg = FactorCorrelationGeometryConfig(factor_count=1,
            max_condition_number=config.max_condition_number,
            holdout_score_relative_rmse=config.structured_holdout_score_relative_rmse)
        fit = structured_fit_data_program(dimension, capacity, fresh // 2, factor_cfg, jit_compile=jit_compile)
        decide = factor_decisions_program(dimension, factor_cfg, jit_compile=jit_compile)
    else:
        # Only the sample count differs between refinement and terminal score
        # fitting; all existing symmetric-fit settings and support rules match.
        from dataclasses import replace

        fit = terminal_program(scalar, batched, dimension,
            replace(config, terminal_sample_count=config.regression_sample_count), jit_compile=jit_compile)
    second = (second_factor_program(dimension, capacity, fresh // 2, config, jit_compile=jit_compile)
        if escalate else empty_second_program(dimension))
    proposals = attempts_program(scalar, second, dimension, config, jit_compile=jit_compile)
    empty_shape = proposals.get_concrete_function().structured_outputs

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([], D),
        tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D), tf.TensorSpec([], D),
        tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.int32)], jit_compile=jit_compile, autograph=False)
    def step(center, value, score, scale, radius, stalled, index):
        seed = tf.constant(config.seed, tf.int32)
        selected = search(center, value, score, scale, radius, seed + tf.stack([0, 1000 + index]))
        fit_seed = seed + tf.stack([0, 10000 + index])
        if structured:
            data = preparation(selected['position'], selected['score'], scale, radius, fit_seed,
                selected['search_positions'], selected['search_scores'])
            first_inputs = (data['center_score_z'], data['training_offsets_z'], data['training_scores_z'],
                data['holdout_offsets_z'], data['holdout_scores_z'], data['training_weights'], data['active_training_rows'])
            first = fit(*first_inputs)
            computed = first['fit']
            decision = decide(computed['invalid_covariance_evaluations'], computed['finite'], computed['eigenvalues'],
                computed['condition_number'], computed['jacobian_rank'], computed['holdout_relative'],
                computed['optimizer'].failed, computed['loadings'])
            first = {**first, 'decision': decision}
            usable = (first['input_status'] == 0) & (decision['status_code'] == 0)
            precision = computed['precision']
            has_best, best_value, best_position, best_score = (
                data['best_index'] >= 0, data['best_value'], data['best_position'], data['best_score'])
            evaluations = tf.constant(search_count + fresh)
            second_inputs = first_inputs if escalate else ()
            # Preserve only the fields used by the original public fit record.
            preparation_record = {'best_index': data['best_index'], 'best_value': best_value,
                'best_position': best_position, 'best_score': best_score,
                'active_training_rows': data['active_training_rows'], 'fresh_training_count': data['fresh_training_count'],
                'fresh_holdout_count': data['fresh_holdout_count'], 'reused_training_count': data['reused_training_count']}
        else:
            first = fit(selected['position'], selected['score'], scale, radius, fit_seed)
            usable, precision = first['usable'], first['record']['projected_precision_z']
            has_best, best_value, best_position, best_score = (
                first['has_best'], first['best_value'], first['best_position'], first['best_score'])
            evaluations = search_count + first['evaluations']
            second_inputs = ()
            preparation_record = {}
        moved = has_best & (best_value > selected['value'])
        fit_rejected = ~usable & ~tf.constant(escalate)
        empty_attempts = tf.nest.map_structure(lambda v: tf.zeros(v.shape, v.dtype), empty_shape)
        attempt = tf.cond(moved | fit_rejected, lambda: empty_attempts,
            lambda: proposals(selected['position'], selected['value'], selected['score'], scale, radius,
                stalled, usable, tf.where(usable, precision, tf.zeros_like(precision)), second_inputs))
        next_radius = tf.where(moved, radius, tf.where(fit_rejected,
            radius * tf.constant(config.shrink_factor, D), attempt['radius_after']))
        next_stalled = tf.where(moved, 0, tf.where(fit_rejected, stalled + 1, attempt['stalled']))
        stop = ~moved & ((next_radius < tf.constant(config.minimum_radius, D))
            | (tf.constant(config.stop_on_stalled_attempts) & (next_stalled >= config.max_stalled_attempts)))
        return {'center': tf.where(moved, best_position, tf.where(fit_rejected, selected['position'], attempt['center'])),
            'center_value': tf.where(moved, best_value, tf.where(fit_rejected, selected['value'], attempt['center_value'])),
            'center_score': tf.where(moved, best_score, tf.where(fit_rejected, selected['score'], attempt['center_score'])),
            'radius_after': next_radius, 'stalled': next_stalled, 'stop': stop,
            'evaluations': evaluations + tf.where(moved | fit_rejected, 0, attempt['evaluations']),
            'action': tf.where(moved, 0, tf.where(fit_rejected, 1, tf.where(attempt['accepted'], 2, 3))),
            'first_fit': first, 'preparation': preparation_record, 'fit_seed': fit_seed, 'attempts': attempt,
            'selected_center': selected['position'], 'selected_value': selected['value'],
            'selected_score': selected['score'], 'search_recentered': selected['recentered']}

    return step
