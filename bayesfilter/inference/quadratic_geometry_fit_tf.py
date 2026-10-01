"""Internal native fit and final decisions on an evaluated quadratic design.

Pilot clouds and partition preparation are inputs to this execution boundary.
The complete geometry program calls it inside its public numerical boundary.
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
from bayesfilter.inference.quadratic_geometry_rows_tf import (
    compact_holdout_rmse,
    compact_train_metrics,
)

D = tf.float64
STATUSES = ("fit_nonfinite", "usable", "holdout_fit_rejected",
            "precision_not_spd", "precision_condition_above_cap", "fit_design_ill_conditioned")
SOURCE_ROLES = ("center", "pilot", "design", "surrogate_replay")
_LOCK = RLock()
_LAST_FIT = None


def geometry_fit_program(callback, dimension, rank, train_rows, holdout_rows, config, *, jit_compile=True, active_rows=False,
                         minimum_train_rows=1):
    """Reuse one numerical signature without retaining prior callback graphs."""
    global _LAST_FIT
    settings = (dimension, rank, train_rows, holdout_rows, jit_compile, active_rows, minimum_train_rows,
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
            config, jit_compile=jit_compile, active_rows=active_rows, minimum_train_rows=minimum_train_rows)
        _LAST_FIT = (callback, settings, program)
        return program


def clear_geometry_fit_cache():
    """Release Python ownership; this does not assert native code eviction."""
    global _LAST_FIT
    with _LOCK:
        _LAST_FIT = None


def make_geometry_fit_program(callback, dimension, rank, train_rows, holdout_rows, config, *, jit_compile=True, active_rows=False,
                              minimum_train_rows=1):
    """Compile fitting through exact replay with fixed design extents.

    Training scores and center_score are already in whitened coordinates.
    Incumbent scores retain the original parameter coordinates.
    With active_rows=True the extents are capacities and two final int32 scalar
    inputs give the retained training/holdout counts. Invalid counts fail the fit
    without calling the target; inactive rows never become observations.
    """
    if dimension < 1 or not 0 <= rank < dimension or train_rows < 1 or holdout_rows < 0:
        raise ValueError("invalid geometry fit extents")
    if train_rows * dimension < rank + 1:
        raise ValueError("training design has fewer rows than curvature parameters")
    if not 1 <= minimum_train_rows <= train_rows:
        raise ValueError("invalid minimum training count")
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
    ] + ([tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.int32)] if active_rows else []),
        autograph=False, jit_compile=jit_compile)
    def evaluate(center, scale, basis, z_train, y_train, score_train, z_holdout, y_holdout,
                 center_value, center_score, best_position, best_value, best_score,
                 best_index, best_source, has_incumbent, evaluation_count, *counts):
        training_count, holdout_count = counts if active_rows else (train_rows, holdout_rows)
        fit = _quadratic_fit_kernel(z_train, y_train, score_train, basis, center_score,
            tf.constant(config.eigenvalue_floor, D), tf.constant(config.max_condition_number, D),
            use_xla_svd=jit_compile, active_rows=training_count if active_rows else None,
            minimum_active_rows=minimum_train_rows)
        if active_rows:
            fit = {**fit, "finite": fit["finite"] & (holdout_count >= 0) & (holdout_count <= holdout_rows)}
            training_mask = tf.range(train_rows) < training_count
            holdout_mask = tf.range(holdout_rows) < holdout_count
            z_train = tf.where(training_mask[:, None], z_train, tf.zeros_like(z_train))
            y_train = tf.where(training_mask, y_train, tf.zeros_like(y_train))
            z_holdout = tf.where(holdout_mask[:, None], z_holdout, tf.zeros_like(z_holdout))
            y_holdout = tf.where(holdout_mask, y_holdout, tf.zeros_like(y_holdout))
        valid_singular = tf.math.is_finite(fit["singular_values"]) & (fit["singular_values"] > 0.)
        design_condition = tf.reduce_max(tf.where(valid_singular, fit["singular_values"], tf.constant(0., D))) / (
            tf.reduce_min(tf.where(valid_singular, fit["singular_values"], tf.constant(float("inf"), D))))
        usable = fit["finite"] & fit["design_resolved"]
        # Failed fits must not initialize invalid inverse/eigenvalue operations.
        safe_precision = tf.where(usable, fit["precision"], tf.eye(dimension, dtype=D))
        covariance = tf.linalg.inv(safe_precision)
        covariance = .5 * (covariance + tf.transpose(covariance))
        precision_summary = summary(safe_precision)
        covariance_summary = summary(covariance)
        predicted_train = _predict_quadratic(z_train, intercept=fit["intercept"],
            linear=fit["linear_term"], lambda0=fit["lambda0"], mu=fit["mu"], q_basis=basis)
        if active_rows:
            train_rmse, train_std = compact_train_metrics(y_train, predicted_train, center_value, training_count, minimum_train_rows)
        else:
            train_rmse = tf.sqrt(tf.reduce_mean((y_train - predicted_train) ** 2))
            train_std = tf.math.reduce_std(y_train - center_value)
        if holdout_rows:
            predicted_holdout = _predict_quadratic(z_holdout, intercept=fit["intercept"],
                linear=fit["linear_term"], lambda0=fit["lambda0"], mu=fit["mu"], q_basis=basis)
            holdout_rmse = (compact_holdout_rmse(y_holdout, predicted_holdout, holdout_count) if active_rows
                else tf.sqrt(tf.reduce_mean((y_holdout - predicted_holdout) ** 2)))
            holdout_scale = tf.maximum(tf.constant(1., D), train_std)
            holdout_threshold = tf.maximum(tf.constant(config.holdout_rmse_abs_tolerance, D),
                tf.constant(config.holdout_rmse_rel_tolerance, D) * holdout_scale)
            holdout_passed = holdout_rmse <= holdout_threshold
            if active_rows:
                holdout_threshold = tf.where(holdout_count > 0, holdout_threshold, tf.constant(float('nan'), D))
                holdout_passed = (holdout_count == 0) | holdout_passed
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
        status = tf.where(~usable, tf.where(fit["finite"], 5, 0), tf.where(~holdout_passed, 2,
            tf.where(~precision_summary[3], 3, tf.where(
                precision_summary[1][2] > tf.constant(config.max_condition_number * (1. + 1e-8), D), 4, 1))))
        return {"status": status, "fit": fit, "basis": basis, "covariance": covariance,
            "design_condition": design_condition, "design_condition_defined": tf.reduce_any(valid_singular),
            "lstsq_residual_defined": (fit["score_design_rank"] == rank + 1) & (training_count * dimension > rank + 1),
            "train_rmse": train_rmse, "holdout_rmse": holdout_rmse,
            "holdout_threshold": holdout_threshold, "holdout_passed": holdout_passed,
            "precision_summary": precision_summary, "covariance_summary": covariance_summary,
            "refinement": proposal, "replay": exact, "best_position": position,
            "best_value": value, "best_score": score, "best_index": index, "best_source": source,
            "has_incumbent": present, "replay_index": count,
            "evaluation_count": count + tf.cast(exact["attempted"], tf.int64)}

    return evaluate
