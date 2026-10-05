"""Seeded dense initialization in one enclosing TensorFlow/XLA program.

The versioned Philox preparation and the qualified attempt recurrence share a
stable operand signature. Completed reports and archives are decoded separately.
"""

import tensorflow as tf

from bayesfilter.inference.dense_initializer_controller_tf import (
    DenseInitializerProgram,
)
from bayesfilter.inference.dense_initializer_random_tf import (
    STREAM_ID,
    make_dense_initializer_cloud_design,
)


class SeededDenseInitializerProgram:
    """Own one seeded recurrence and its resettable locator resources."""

    stream_id = STREAM_ID

    def __init__(self, callback, dimension, replicates, training_rows, selection_rows,
            audit_rows, *, locator_config, thresholds, max_attempts,
            max_exact_evaluations, center_score_max, factor_max=2,
            dense_eigenvalue_floor=1e-8, max_condition_number=1e8,
            shrinkage_weights=(0., .25, .5, .75, 1.), structured_target_family=None):
        self.controller = DenseInitializerProgram(callback, dimension, replicates,
            training_rows, selection_rows, audit_rows, locator_config=locator_config,
            thresholds=thresholds, max_attempts=max_attempts,
            max_exact_evaluations=max_exact_evaluations, center_score_max=center_score_max,
            factor_max=factor_max, dense_eigenvalue_floor=dense_eigenvalue_floor,
            max_condition_number=max_condition_number, shrinkage_weights=shrinkage_weights,
            structured_target_family=structured_target_family)
        self.cloud_design = make_dense_initializer_cloud_design(dimension, replicates,
            training_rows, selection_rows, audit_rows, max_attempts,
            jit_compile=locator_config.jit_compile)
        self.invocation_lock = self.controller.invocation_lock
        self.planned_rows = self.controller.planned_rows
        generate, initialize = self.cloud_design, self.controller.compiled

        @tf.function(input_signature=[tf.TensorSpec([dimension], tf.float64),
            tf.TensorSpec([dimension], tf.float64), tf.TensorSpec([2], tf.int32),
            tf.TensorSpec([], tf.float64)], jit_compile=locator_config.jit_compile, autograph=False)
        def run(initial, scale, seed, radius):
            return initialize(initial, scale, generate(seed, radius))

        self.compiled = run

    def __call__(self, initial, scale, seed, radius):
        with self.invocation_lock:
            result = self.compiled(initial, scale, seed, radius)
            int(result['status_code'])
            return result
