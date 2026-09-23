"""Native dense initializer cloud-to-fitting attempt boundary.

This composes the already qualified cloud evaluator and validated fixed-center
fitter. It accepts prepared offset clouds as operands; RNG, locator retries,
archives and public serialization remain separate responsibilities.
"""

import tensorflow as tf

from bayesfilter.inference.dense_initializer_cloud_tf import (
    make_dense_initializer_cloud_program,
)
from bayesfilter.inference.dense_validated_fit_tf import (
    make_dense_validated_fit_program,
)

D = tf.float64


def make_dense_initializer_attempt_program(callback, dimension, replicates,
        training_rows, selection_rows, audit_rows, *, thresholds, factor_max=2,
        dense_eigenvalue_floor=1e-8, max_condition_number=1e8,
        shrinkage_weights=(0., .25, .5, .75, 1.), structured_target_family=None,
        jit_compile=True):
    """Evaluate clouds, then fit; center_score is in the target coordinates."""
    cloud = make_dense_initializer_cloud_program(callback, dimension, replicates,
        training_rows, selection_rows, audit_rows, jit_compile=jit_compile)
    fit = make_dense_validated_fit_program(dimension, replicates, training_rows,
        selection_rows, audit_rows, thresholds=thresholds, factor_max=factor_max,
        dense_eigenvalue_floor=dense_eigenvalue_floor,
        max_condition_number=max_condition_number, shrinkage_weights=shrinkage_weights,
        structured_target_family=structured_target_family, jit_compile=jit_compile)
    fit_template = fit.get_concrete_function().structured_outputs
    zero_fit = tf.nest.map_structure(lambda value: tf.zeros(value.shape, value.dtype), fit_template)
    partitions = 2 * replicates + 1
    capacity = max(training_rows, selection_rows, audit_rows)

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([partitions, capacity, dimension], D)],
        jit_compile=jit_compile, autograph=False)
    def evaluate(center, scale, center_value, center_score, offsets):
        clouds = cloud(center, scale, center_value, offsets)
        moved = clouds["valid"] & (clouds["candidate_value"] > center_value)
        centered = clouds["valid"] & ~moved

        def fit_centered():
            return fit(center, center_score * scale, offsets, clouds["scaled_scores"])

        fit_result = tf.cond(centered, fit_centered, lambda: zero_fit)
        usable = centered & fit_result["usable"]

        def scale_from_covariance():
            covariance = fit_result["fit"]["covariance"]
            marginal = scale * tf.sqrt(tf.linalg.diag_part(covariance))
            finite = tf.reduce_all(tf.math.is_finite(marginal) & (marginal > 0.))
            return marginal, finite

        marginal, scale_valid = tf.cond(usable, scale_from_covariance,
            lambda: (tf.zeros([dimension], D), tf.constant(False)))
        status = tf.where(~clouds["valid"], tf.constant(1),
            tf.where(moved, tf.constant(2),
            tf.where(~fit_result["fit_ran"], tf.constant(3),
            tf.where((fit_result["fit_error_code"] != 0) | ~fit_result["usable"], tf.constant(4),
            tf.where(~scale_valid, tf.constant(5), tf.constant(6))))))
        return tf.nest.map_structure(tf.stop_gradient, {
            "status_code": status, "cloud": clouds, "fit": fit_result,
            "candidate_center": clouds["candidate_center"],
            "candidate_value": clouds["candidate_value"],
            "marginal_standard_deviations": marginal,
            "initial_output_shift": tf.where(status == 6, clouds["candidate_center"],
                tf.zeros_like(center)),
            "initial_output_scale_log": tf.where(status == 6, tf.math.log(marginal),
                tf.zeros_like(center)),
            "exact_evaluation_rows": clouds["exact_rows"],
            "fit_ran": fit_result["fit_ran"], "usable": status == 6,
        })

    return evaluate
