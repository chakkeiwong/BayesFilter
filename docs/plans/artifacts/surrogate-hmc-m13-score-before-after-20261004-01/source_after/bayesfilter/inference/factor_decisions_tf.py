"""Native factor-fit rejection precedence and numerical report fields."""

from functools import lru_cache

import tensorflow as tf

D = tf.float64
STATUS_NAMES = (
    'usable', 'factor_optimizer_failed', 'nonfinite_or_non_spd_fit',
    'condition_number_above_cap', 'second_factor_unidentified',
    'holdout_score_fit_rejected', 'factor_optimizer_failed',
)


@lru_cache(maxsize=32)
def factor_decisions_program(dimension, config, *, jit_compile=True):
    parameter_count = 2 * dimension if config.factor_count == 1 else 3 * dimension - 1
    # Preserve the original Python-rounded configuration threshold.
    condition_cap = config.max_condition_number * (1. + 1.e-8)

    @tf.function(input_signature=[tf.TensorSpec([], tf.int64), tf.TensorSpec([], tf.bool),
        tf.TensorSpec([dimension], D), tf.TensorSpec([], D), tf.TensorSpec([], tf.int32),
        tf.TensorSpec([], D), tf.TensorSpec([], tf.bool),
        tf.TensorSpec([dimension, config.factor_count], D)], jit_compile=jit_compile, autograph=False)
    def decide(invalid_evaluations, finite, eigenvalues, condition, rank, holdout, failed, loadings):
        identified = (config.factor_count == 1) | (rank == parameter_count)
        status = tf.where(failed, 6, 0)
        status = tf.where(holdout > tf.constant(config.holdout_score_relative_rmse, D), 5, status)
        if config.factor_count == 2:
            status = tf.where(~identified, 4, status)
        status = tf.where(condition > tf.constant(condition_cap, D), 3, status)
        # XLA minimum can discard NaNs and return +inf for an all-NaN
        # vector. Explicit positive comparisons preserve the rejection.
        status = tf.where(~finite | ~tf.reduce_all(eigenvalues > 0.), 2, status)
        status = tf.where(invalid_evaluations > 0, 1, status)
        return {'status_code': status, 'second_factor_identified': identified,
            'jacobian_condition_available': rank != 0,
            'loading_row_squared_norms': tf.reduce_sum(tf.square(loadings), axis=1)}

    return decide
