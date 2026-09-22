"""Native terminal score fit with the original static support decision."""


import tensorflow as tf

from bayesfilter.inference.program_cache_scope import scoped_program_cache
from bayesfilter.inference.sequential_score_fit_tf import (
    partition_schema,
    score_fit_program,
)

D = tf.float64


@scoped_program_cache(maxsize=32)
def terminal_program(scalar, batched, dimension, config, *, jit_compile=True):
    count = config.terminal_sample_count
    if config.pair_disjoint_score_holdout:
        training, holdout = partition_schema(count, config.holdout_fraction, pair_disjoint=True)
        support = len(training) // 2
    else:
        support = count
    insufficient = support * dimension < dimension * (dimension + 1) // 2 + dimension
    fit = None
    if not insufficient:
        training, holdout = partition_schema(count, config.holdout_fraction,
            pair_disjoint=config.pair_disjoint_score_holdout)
        fit = score_fit_program(scalar, batched, count, dimension, training, holdout,
            jit_compile=jit_compile)

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([dimension], D), tf.TensorSpec([], D), tf.TensorSpec([2], tf.int32)],
        jit_compile=jit_compile, autograph=False)
    def terminal(center, score, scale, radius, seed):
        if insufficient:
            computed = {'status': tf.constant(-1), 'rank': tf.constant(0),
                'train_score_rmse': tf.constant(0., D), 'holdout_score_relative_rmse': tf.constant(0., D),
                'raw_eigenvalues': tf.zeros([dimension], D), 'projected_eigenvalues': tf.zeros([dimension], D),
                'projection_relative_frobenius': tf.constant(0., D),
                'projected_precision_z': tf.zeros([dimension, dimension], D),
                'best_index': tf.constant(-1), 'best_value': tf.constant(float('-inf'), D),
                'best_position': center, 'best_score': score}
        else:
            computed = fit(center, score, scale, radius, seed,
                tf.constant(config.ridge, D), tf.constant(config.eigenvalue_floor, D),
                tf.constant(config.max_condition_number, D), tf.constant(config.score_holdout_relative_rmse, D))
        return {'usable': computed['status'] == 1, 'projection': computed['projection_relative_frobenius'],
            'has_best': computed['best_index'] >= 0, 'best_value': computed['best_value'],
            'best_position': computed['best_position'], 'best_score': computed['best_score'],
            'evaluations': tf.constant(0 if insufficient else count), 'seed': seed, 'record': computed}

    return terminal
