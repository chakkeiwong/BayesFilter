"""Validated fixed-center fitting for a prepared dense-initializer cloud.

Finite/overlap decisions and fitting execute in one XLA program. Only disjoint
prepared tensor partitions are eligible; original caller-buffer alias checks
remain at the preparation boundary. Full initializer integration is separate.
"""

import tensorflow as tf

from bayesfilter.inference import fixed_center_curvature as fixed
from bayesfilter.inference import fixed_center_fitting_tf as fitting
from bayesfilter.inference.dense_partition_validation_tf import (
    make_dense_partition_validation_program,
)
from bayesfilter.inference.program_cache_scope import ProgramCacheScope

D = tf.float64


def fit_error_code(result):
    """Preserve one-factor, family-stability, then selection error precedence."""
    one = result["one_stability"]
    one_error = tf.where(one["complete"], one["error"], 0)
    family = result["selection"]["stability"]
    family_errors = tf.where(family["complete"], family["error"], 0)
    first = tf.argmax(tf.cast(family_errors != 0, tf.int32), output_type=tf.int32)
    family_error = tf.gather(family_errors, first)
    return tf.where(one_error != 0, one_error, tf.where(family_error != 0, family_error,
        tf.where(result["selection"]["error"] != 0, 4, 0)))


def make_dense_validated_fit_program(dimension, replicates, training_rows, selection_rows,
        audit_rows, *, thresholds, factor_max=2, dense_eigenvalue_floor=1e-8,
        max_condition_number=1e8, shrinkage_weights=(0., .25, .5, .75, 1.),
        structured_target_family=None, jit_compile=True):
    """Build a reusable fitting boundary with explicit data operands."""
    validate = make_dense_partition_validation_program(dimension, replicates, training_rows,
        selection_rows, audit_rows, jit_compile=jit_compile)
    if training_rows + selection_rows < 4 * dimension:
        raise ValueError("each training plus selection replicate must total at least 4N")
    if audit_rows < 2 * dimension:
        raise ValueError("audit rows must total at least 2N")
    if factor_max not in (1, 2):
        raise ValueError("factor_max must be one or two")
    if structured_target_family not in (None, "factor_1", "factor_2"):
        raise ValueError("structured_target_family must be factor_1 or factor_2")
    if structured_target_family == "factor_2" and factor_max != 2:
        raise ValueError("factor_2 structured target requires factor_max=2")
    weights = fixed._shrinkage_weights(shrinkage_weights)
    caps = (thresholds.generalized_eigenvalue_spread_cap,
        thresholds.trace_normalized_frobenius_cap, thresholds.trace_normalized_operator_cap,
        thresholds.principal_angle_degrees_cap)
    cap_values = tf.constant(tf.nest.map_structure(lambda x: 0. if x is None else x, caps), D)
    enabled = tf.constant(tf.nest.map_structure(lambda x: x is not None, caps), tf.bool)
    rank = tf.constant(dimension if thresholds.principal_subspace_rank is None
        else thresholds.principal_subspace_rank, tf.int32)
    weight_values = tf.constant(weights, D)
    eigen_floor = tf.constant(dense_eigenvalue_floor, D)
    projection_cap = tf.constant(thresholds.projection_relative_frobenius_cap, D)
    require_raw_spd = tf.constant(thresholds.require_raw_spd)
    audit_cap = tf.constant(thresholds.audit_relative_rmse_cap, D)
    scope = ProgramCacheScope()
    with scope.activate():
        # This owner retains its own complete fitter. Shape-only lower-level
        # programs retain their existing caches; no callback enters a global LRU.
        fit = fitting.fit_program.__wrapped__(fixed._dense_fit_kernel, fitting._structured_fit,
            fixed._precision_geometry_kernel, fixed._score_error_kernel, dimension, replicates,
            training_rows, selection_rows, audit_rows, factor_max, max_condition_number,
            thresholds.selection_holdout_relative_rmse_cap, structured_target_family,
            len(weights), jit_compile=jit_compile)
        fit_outputs = fit.get_concrete_function().structured_outputs

    @tf.function(input_signature=validate.input_signature, jit_compile=jit_compile, autograph=False)
    def compute(center, center_score, offsets, scores):
        validation = validate(center, center_score, offsets, scores)
        permitted = validation["error_code"] == 0

        def run_fit():
            return fit(center_score, offsets[:replicates, :training_rows], scores[:replicates, :training_rows],
                offsets[replicates:2 * replicates, :selection_rows],
                scores[replicates:2 * replicates, :selection_rows],
                offsets[-1, :audit_rows], scores[-1, :audit_rows], cap_values, enabled, rank,
                weight_values, eigen_floor, projection_cap, require_raw_spd, audit_cap)

        result = tf.cond(permitted, run_fit,
            lambda: tf.nest.map_structure(lambda value: tf.zeros(value.shape, value.dtype), fit_outputs))
        error = fit_error_code(result)
        return tf.nest.map_structure(tf.stop_gradient, {"validation": validation, "fit_ran": permitted,
            "fit_error_code": error, "usable": permitted & (error == 0) & (result["status"] == 1),
            "fit": result})

    with scope.activate():
        compute.get_concrete_function()
    compute.dependency_scope = scope
    return compute
