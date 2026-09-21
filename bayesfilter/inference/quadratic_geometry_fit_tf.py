"""Internal native fit and final decisions on an evaluated quadratic design.

Pilot clouds and partition preparation are inputs to this execution boundary.
The public initializer is unchanged pending whole-endpoint qualification.
"""

from threading import RLock

import tensorflow as tf

from bayesfilter.inference.mass_matrix_tf import summary_program
from bayesfilter.inference.quadratic_geometry import (
    _predict_quadratic,
    _quadratic_fit_kernel,
)
from bayesfilter.inference.quadratic_geometry_control_tf import (
    make_center_refinement_program,
    make_exact_replay_program,
)

D = tf.float64
STATUSES = ("fit_nonfinite", "usable", "holdout_fit_rejected",
            "precision_not_spd", "precision_condition_above_cap")
SOURCE_ROLES = ("center", "pilot", "design", "surrogate_replay")
_LOCK = RLock()
_LAST_FIT = None


def geometry_fit_program(callback, dimension, rank, train_rows, holdout_rows, config, *, jit_compile=True):
    """Reuse one numerical signature without retaining prior callback graphs."""
    global _LAST_FIT
    settings = (dimension, rank, train_rows, holdout_rows, jit_compile,
        config.eigenvalue_floor, config.max_condition_number,
        config.holdout_rmse_abs_tolerance, config.holdout_rmse_rel_tolerance,
        config.constrain_center_refinement_to_trust_region, config.trust_radius,
        config.center_log_prob_tolerance, config.center_score_improvement_factor)
    with _LOCK:
        previous = _LAST_FIT
        if previous is not None and previous[0] is callback and previous[1] == settings:
            return previous[2]
        _LAST_FIT = None
        del previous
        program = make_geometry_fit_program(callback, dimension, rank, train_rows, holdout_rows,
            config, jit_compile=jit_compile)
        _LAST_FIT = (callback, settings, program)
        return program


def clear_geometry_fit_cache():
    """Release Python ownership; this does not assert native code eviction."""
    global _LAST_FIT
    with _LOCK:
        _LAST_FIT = None


def make_geometry_fit_program(callback, dimension, rank, train_rows, holdout_rows, config, *, jit_compile=True):
    """Compile fitting through exact replay with fixed design extents.

    Training scores and center_score are already in whitened coordinates.
    Incumbent scores retain the original parameter coordinates.
    """
    if dimension < 1 or not 0 <= rank < dimension or train_rows < 1 or holdout_rows < 0:
        raise ValueError("invalid geometry fit extents")
    if train_rows * dimension < rank + 1:
        raise ValueError("training design has fewer rows than curvature parameters")
    refinement = make_center_refinement_program(callback, dimension, config, jit_compile=jit_compile)
    replay = make_exact_replay_program(callback, dimension, jit_compile=jit_compile)
    summary = summary_program(dimension, jit_compile=jit_compile)
    refinement_template = refinement.get_concrete_function().structured_outputs
    replay_template = replay.get_concrete_function().structured_outputs

    @tf.function(input_signature=[
        tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([dimension, rank], D), tf.TensorSpec([train_rows, dimension], D),
        tf.TensorSpec([train_rows], D), tf.TensorSpec([train_rows, dimension], D),
        tf.TensorSpec([holdout_rows, dimension], D), tf.TensorSpec([holdout_rows], D),
        tf.TensorSpec([], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([dimension], D), tf.TensorSpec([], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([], tf.int64), tf.TensorSpec([], tf.int32),
        tf.TensorSpec([], tf.bool), tf.TensorSpec([], tf.int64),
    ], autograph=False, jit_compile=jit_compile)
    def evaluate(center, scale, basis, z_train, y_train, score_train, z_holdout, y_holdout,
                 center_value, center_score, best_position, best_value, best_score,
                 best_index, best_source, has_incumbent, evaluation_count):
        fit = _quadratic_fit_kernel(z_train, y_train, score_train, basis, center_score,
            tf.constant(config.eigenvalue_floor, D), tf.constant(config.max_condition_number, D),
            use_xla_svd=jit_compile)
        valid_singular = tf.math.is_finite(fit["singular_values"]) & (fit["singular_values"] > 0.)
        design_condition = tf.reduce_max(tf.where(valid_singular, fit["singular_values"], tf.constant(0., D))) / (
            tf.reduce_min(tf.where(valid_singular, fit["singular_values"], tf.constant(float("inf"), D))))
        usable = fit["finite"]
        # Failed fits must not initialize invalid inverse/eigenvalue operations.
        safe_precision = tf.where(usable, fit["precision"], tf.eye(dimension, dtype=D))
        covariance = tf.linalg.inv(safe_precision)
        covariance = .5 * (covariance + tf.transpose(covariance))
        precision_summary = summary(safe_precision)
        covariance_summary = summary(covariance)
        predicted_train = _predict_quadratic(z_train, intercept=fit["intercept"],
            linear=fit["linear_term"], lambda0=fit["lambda0"], mu=fit["mu"], q_basis=basis)
        train_rmse = tf.sqrt(tf.reduce_mean((y_train - predicted_train) ** 2))
        if holdout_rows:
            predicted_holdout = _predict_quadratic(z_holdout, intercept=fit["intercept"],
                linear=fit["linear_term"], lambda0=fit["lambda0"], mu=fit["mu"], q_basis=basis)
            holdout_rmse = tf.sqrt(tf.reduce_mean((y_holdout - predicted_holdout) ** 2))
            holdout_scale = tf.maximum(tf.constant(1., D), tf.math.reduce_std(y_train - center_value))
            holdout_threshold = tf.maximum(tf.constant(config.holdout_rmse_abs_tolerance, D),
                tf.constant(config.holdout_rmse_rel_tolerance, D) * holdout_scale)
            holdout_passed = holdout_rmse <= holdout_threshold
        else:
            holdout_rmse = holdout_threshold = tf.constant(float("nan"), D)
            holdout_passed = tf.constant(True)

        proposal = tf.cond(usable,
            lambda: refinement(center, scale, fit["precision"], fit["linear_term"],
                               center_value, tf.linalg.norm(center_score)),
            lambda: tf.nest.map_structure(lambda value: tf.zeros(value.shape, value.dtype), refinement_template))
        count = evaluation_count + tf.cast(proposal["target_called"], tf.int64)
        eligible = (usable & (proposal["status"] == 0) & proposal["target_called"]
            & tf.reduce_all(tf.math.is_finite(proposal["refined_center"]))
            & tf.math.is_finite(proposal["refined_value"])
            & tf.reduce_all(tf.math.is_finite(proposal["refined_score"])))
        # Any finite exact proposal can win, even if refinement was rejected.
        # Exact ties preserve the incumbent's earlier index and source.
        wins = eligible & (~has_incumbent | (proposal["refined_value"] > best_value))
        position = tf.where(wins, proposal["refined_center"], best_position)
        value = tf.where(wins, proposal["refined_value"], best_value)
        score = tf.where(wins, proposal["refined_score"], best_score)
        index = tf.where(wins, count - 1, best_index)
        source = tf.where(wins, 3, best_source)
        present = has_incumbent | eligible
        exact = tf.cond(usable, lambda: replay(position, value, score, present),
            lambda: tf.nest.map_structure(lambda value: tf.zeros(value.shape, value.dtype), replay_template))
        status = tf.where(~usable, 0, tf.where(~holdout_passed, 2,
            tf.where(~precision_summary[3], 3, tf.where(
                precision_summary[1][2] > tf.constant(config.max_condition_number * (1. + 1e-8), D), 4, 1))))
        return {"status": status, "fit": fit, "basis": basis, "covariance": covariance,
            "design_condition": design_condition, "design_condition_defined": tf.reduce_any(valid_singular),
            "lstsq_residual_defined": (fit["score_design_rank"] == rank + 1) & (train_rows * dimension > rank + 1),
            "train_rmse": train_rmse, "holdout_rmse": holdout_rmse,
            "holdout_threshold": holdout_threshold, "holdout_passed": holdout_passed,
            "precision_summary": precision_summary, "covariance_summary": covariance_summary,
            "refinement": proposal, "replay": exact, "best_position": position,
            "best_value": value, "best_score": score, "best_index": index, "best_source": source,
            "has_incumbent": present, "replay_index": count,
            "evaluation_count": count + tf.cast(exact["attempted"], tf.int64)}

    return evaluate
