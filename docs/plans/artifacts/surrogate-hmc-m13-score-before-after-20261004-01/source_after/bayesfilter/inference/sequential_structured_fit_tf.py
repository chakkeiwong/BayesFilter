"""Native finite-input gate and fixed-capacity factor fit for prepared rows."""

from collections import namedtuple
from functools import lru_cache

import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor

D = tf.float64
OptimizerSummary = namedtuple('OptimizerSummary',
    'converged failed num_iterations num_objective_evaluations objective_value')


def _empty_result(dimension, factor_count):
    zero = tf.constant(0., D)
    return {'covariance': tf.zeros([dimension, dimension], D),
        'precision': tf.zeros([dimension, dimension], D), 'deviations': tf.zeros([dimension], D),
        'loadings': tf.zeros([dimension, factor_count], D), 'eigenvalues': tf.zeros([dimension], D),
        'finite': tf.constant(False), 'condition_number': zero, 'train_rmse': zero,
        'holdout_error': zero, 'holdout_relative': zero, 'jacobian_rank': tf.constant(0),
        'jacobian_condition': zero, 'anchors': tf.zeros([factor_count], tf.int32),
        'invalid_covariance_evaluations': tf.constant(0, tf.int64),
        'optimizer': OptimizerSummary(tf.constant(False), tf.constant(False),
            tf.constant(0), tf.constant(0), zero)}


def structured_fit_data_program(dimension, training_capacity, holdout_rows, config, *, jit_compile=True):
    """Keep active-row eligibility native; reject invalid data before fitting."""
    count = 2 * dimension if config.factor_count == 1 else 3 * dimension - 1
    # Bind the guard to every enclosing consumer graph, including cached reuse.
    kernel = None if count > dimension * (dimension + 1) // 2 else factor._make_factor_program(
        dimension, training_capacity, holdout_rows, config, jit_compile,
        factor._prediction_jacobian_diagnostics, padded_training=True)
    return _cached_structured_fit(kernel, dimension, training_capacity, holdout_rows, config, jit_compile)


@lru_cache(maxsize=32)
def _cached_structured_fit(kernel, dimension, training_capacity, holdout_rows, config, jit_compile):
    @tf.function(input_signature=[tf.TensorSpec([dimension], D),
        tf.TensorSpec([training_capacity, dimension], D), tf.TensorSpec([training_capacity, dimension], D),
        tf.TensorSpec([holdout_rows, dimension], D), tf.TensorSpec([holdout_rows, dimension], D),
        tf.TensorSpec([training_capacity], D), tf.TensorSpec([], tf.int32)],
        jit_compile=jit_compile, autograph=False)
    def fit(center, training, scores, holdout, holdout_scores, weights, active_count):
        if kernel is None:
            return {'input_status': tf.constant(1), 'fit': _empty_result(dimension, config.factor_count)}
        active = tf.range(training_capacity) < active_count
        finite = (tf.reduce_all(tf.math.is_finite(center))
            & tf.reduce_all(~active[:, None] | (tf.math.is_finite(training) & tf.math.is_finite(scores)))
            & tf.reduce_all(tf.math.is_finite(holdout)) & tf.reduce_all(tf.math.is_finite(holdout_scores)))
        valid_weights = tf.reduce_all(tf.math.is_finite(weights)
            & tf.where(active, weights > 0., weights == 0.))
        valid_count = (active_count >= 2 * dimension) & (active_count <= training_capacity)
        status = tf.where(~valid_count, 4, tf.where(~finite, 2, tf.where(~valid_weights, 3, 0)))

        def compute():
            result = kernel(center, training, scores, holdout, holdout_scores, weights, active_count)
            optimizer = result['optimizer']
            return {**result, 'optimizer': OptimizerSummary(optimizer.converged, optimizer.failed,
                optimizer.num_iterations, optimizer.num_objective_evaluations, optimizer.objective_value)}

        return tf.nest.map_structure(tf.stop_gradient,
            {'input_status': status, 'fit': tf.cond(status == 0, compute,
                lambda: _empty_result(dimension, config.factor_count))})

    return fit
