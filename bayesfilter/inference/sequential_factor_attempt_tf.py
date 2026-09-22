"""Native second-factor callback for the native proposal controller."""

from functools import lru_cache

import tensorflow as tf

from bayesfilter.inference.factor_correlation_geometry import (
    FactorCorrelationGeometryConfig,
)
from bayesfilter.inference.factor_decisions_tf import factor_decisions_program
from bayesfilter.inference.program_cache_scope import scoped_program_cache
from bayesfilter.inference.sequential_structured_fit_tf import (
    structured_fit_data_program,
)

D = tf.float64


@scoped_program_cache(maxsize=32)
def empty_second_program(dimension):
    """Fixed empty schema for configurations with exactly one proposal."""
    @tf.function(input_signature=[], jit_compile=True, autograph=False)
    def empty():
        return {'precision': tf.zeros([dimension, dimension], D), 'usable': tf.constant(False)}

    return empty


@lru_cache(maxsize=32)
def second_factor_program(dimension, capacity, holdout_rows, config, *, jit_compile=True):
    cfg = FactorCorrelationGeometryConfig(factor_count=2,
        max_condition_number=config.max_condition_number,
        holdout_score_relative_rmse=config.structured_holdout_score_relative_rmse)
    decide = factor_decisions_program(dimension, cfg, jit_compile=jit_compile).python_function

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([capacity, dimension], D),
        tf.TensorSpec([capacity, dimension], D), tf.TensorSpec([holdout_rows, dimension], D),
        tf.TensorSpec([holdout_rows, dimension], D), tf.TensorSpec([capacity], D),
        tf.TensorSpec([], tf.int32)], jit_compile=jit_compile, autograph=False)
    def evaluate(center, training, scores, holdout, holdout_scores, weights, active):
        computed = structured_fit_data_program(dimension, capacity, holdout_rows, cfg,
            jit_compile=jit_compile).python_function(center, training, scores, holdout, holdout_scores, weights, active)
        fit = computed['fit']
        decision = decide(fit['invalid_covariance_evaluations'], fit['finite'], fit['eigenvalues'],
            fit['condition_number'], fit['jacobian_rank'], fit['holdout_relative'],
            fit['optimizer'].failed, fit['loadings'])
        return {'usable': (computed['input_status'] == 0) & (decision['status_code'] == 0),
            'precision': fit['precision'], 'computed': computed, 'decision': decision}

    return evaluate
