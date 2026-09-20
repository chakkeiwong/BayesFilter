"""Complete native replicate fitting, selection and audit at a fixed center.

This encloses the existing dense kernel and factor optimizer without changing
their numerical policy. Public wrappers own schema validation and reporting.
"""

from functools import lru_cache

import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from bayesfilter.inference import fixed_center_selection_tf as selection
from bayesfilter.inference import fixed_center_stability_tf as stability
from bayesfilter.inference.mass_matrix_tf import _eigenpairs

D = tf.float64
FIT_STATUSES = (
    "usable", "raw_curvature_not_spd", "projection_burden_rejected", "selection_holdout_rejected",
    "factor_parameterization_dimensionally_unidentified", "nonfinite_or_non_spd_fit",
    "condition_number_above_cap", "second_factor_unidentified", "holdout_score_fit_rejected",
    "factor_optimizer_failed",
)
RESULT_STATUSES = ("geometry_readiness_blocked", "eligible_for_exact_hmc_canary",
    "diagnostic_only", "audit_holdout_rejected")
FACTOR_METRICS = ("train_score_rmse", "holdout_score_rmse", "holdout_score_relative_rmse",
    "condition_number", "prediction_jacobian_rank", "prediction_jacobian_condition_number",
    "optimizer_converged", "optimizer_failed", "optimizer_iterations",
    "optimizer_objective_evaluations", "final_loss", "second_factor_identified")


def empty_fit(dimension):
    return {"raw": tf.zeros([dimension, dimension], D), "precision": tf.zeros([dimension, dimension], D),
        "covariance": tf.zeros([dimension, dimension], D), "raw_values": tf.zeros([dimension], D),
        "flags": tf.zeros([3], tf.bool), "status": tf.constant(4), "projection": tf.constant(0., D),
        "holdout": tf.constant(0., D), "holdout_passed": tf.constant(False), "nonpositive": tf.constant(0, tf.int64),
        "factor_metrics": tf.zeros([len(FACTOR_METRICS)], D), "factor_eigenvalues": tf.zeros([dimension], D),
        "loading_norms": tf.zeros([dimension], D), "anchors": tf.zeros([2], tf.int32)}


def _dense_fit(kernel, center, training, scores, offsets, selection_scores,
               eigenvalue_floor, condition_cap, holdout_cap, projection_cap, require_raw_spd, *, jit_compile=True):
    raw, precision, covariance, values, projection, holdout, nonpositive = kernel(
        center, training, scores, offsets, selection_scores, eigenvalue_floor, condition_cap, jit_compile=jit_compile)
    raw_ok = ~require_raw_spd | (nonpositive == 0)
    admissible = raw_ok & (projection <= projection_cap)
    accepted = admissible & (holdout <= holdout_cap)
    status = tf.where(accepted, 0, tf.where(~raw_ok, 1, tf.where(projection > projection_cap, 2, 3)))
    return {**empty_fit(center.shape[0]), "raw": raw, "precision": precision, "covariance": covariance,
        "raw_values": values, "flags": tf.stack([True, admissible, accepted]), "status": status,
        "projection": projection, "holdout": holdout, "holdout_passed": holdout <= holdout_cap, "nonpositive": nonpositive}


def _structured_fit(center, training, scores, offsets, selection_scores, *, config, jit_compile):
    dimension = int(center.shape[0])
    parameter_count = 2 * dimension if config.factor_count == 1 else 3 * dimension - 1
    if parameter_count > dimension * (dimension + 1) // 2:
        return empty_fit(dimension)
    program = factor._make_factor_program(dimension, int(training.shape[0]), int(offsets.shape[0]),
        config, jit_compile, factor._prediction_jacobian_diagnostics)
    weights = tf.fill([training.shape[0]], tf.constant(1. / training.shape[0], D))
    value = program(center, training, scores, offsets, selection_scores, weights)
    optimizer = value["optimizer"]
    identified = (config.factor_count == 1) | (value["jacobian_rank"] == parameter_count)
    status = tf.where(~value["finite"] | ~(tf.reduce_min(value["eigenvalues"]) > 0.), 5,
        tf.where(value["condition_number"] > config.max_condition_number * (1. + 1e-8), 6,
        tf.where(~identified, 7, tf.where(value["holdout_relative"] > config.holdout_score_relative_rmse, 8,
        tf.where(optimizer.failed, 9, 0)))))
    admissible = ((status == 0) | (status == 8)) & identified
    raw_values = _eigenpairs(value["precision"], jit_compile)[0]
    metrics = tf.stack([value["train_rmse"], value["holdout_error"], value["holdout_relative"],
        value["condition_number"], tf.cast(value["jacobian_rank"], D), value["jacobian_condition"],
        tf.cast(optimizer.converged, D), tf.cast(optimizer.failed, D), tf.cast(optimizer.num_iterations, D),
        tf.cast(optimizer.num_objective_evaluations, D), optimizer.objective_value, tf.cast(identified, D)])
    return {**empty_fit(dimension), "raw": value["precision"], "precision": value["precision"],
        "covariance": value["covariance"], "raw_values": raw_values,
        "flags": tf.stack([True, admissible, status == 0]), "status": status,
        "holdout": value["holdout_relative"], "holdout_passed": value["holdout_relative"] <= config.holdout_score_relative_rmse,
        "nonpositive": tf.math.count_nonzero(raw_values <= 0.),
        "factor_metrics": metrics, "factor_eigenvalues": value["eigenvalues"],
        "loading_norms": tf.reduce_sum(tf.square(value["loadings"]), axis=1),
        "anchors": tf.pad(tf.cast(value["anchors"], tf.int32), [[0, 2 - config.factor_count]])}


@lru_cache(maxsize=32)
def fit_program(dense_kernel, structured_kernel, comparison_kernel, score_kernel, dimension, replicates,
        training_rows, selection_rows, audit_rows, factor_max, max_condition_number,
        holdout_cap, structured_target_family, weight_count, *, jit_compile=True):
    configs = tuple(factor.FactorCorrelationGeometryConfig(factor_count=count,
        max_condition_number=max_condition_number, holdout_score_relative_rmse=holdout_cap)
        for count in (1, 2))
    one_stability = stability.stability_program(comparison_kernel, dimension, replicates, jit_compile=jit_compile)
    select_two = selection.selection_program(comparison_kernel, score_kernel, dimension,
        ("dense", "factor_1"), (replicates,) * 2, (selection_rows,) * replicates,
        weight_count, structured_target_family, jit_compile=jit_compile)
    select_three = selection.selection_program(comparison_kernel, score_kernel, dimension,
        ("dense", "factor_1", "factor_2"), (replicates,) * 3, (selection_rows,) * replicates,
        weight_count, structured_target_family, jit_compile=jit_compile)

    @tf.function(input_signature=[tf.TensorSpec([dimension], D),
        tf.TensorSpec([replicates, training_rows, dimension], D), tf.TensorSpec([replicates, training_rows, dimension], D),
        tf.TensorSpec([replicates, selection_rows, dimension], D), tf.TensorSpec([replicates, selection_rows, dimension], D),
        tf.TensorSpec([audit_rows, dimension], D), tf.TensorSpec([audit_rows, dimension], D),
        tf.TensorSpec([4], D), tf.TensorSpec([4], tf.bool), tf.TensorSpec([], tf.int32),
        tf.TensorSpec([weight_count], D), tf.TensorSpec([], D), tf.TensorSpec([], D),
        tf.TensorSpec([], tf.bool), tf.TensorSpec([], D)], jit_compile=jit_compile, autograph=False)
    def compute(center, training, scores, offsets, selection_scores, audit, audit_scores,
                caps, enabled, rank, weights, eigenvalue_floor, projection_cap, require_raw_spd, audit_cap):
        fits = tf.nest.map_structure(lambda value: tf.zeros([3, replicates, *value.shape], value.dtype), empty_fit(dimension))

        def put(fits, family, index, fit):
            return tf.nest.map_structure(lambda rows, row: tf.tensor_scatter_nd_update(rows, [[family, index]], [row]), fits, fit)

        def initial(index, fits):
            dense = _dense_fit(dense_kernel, center, training[index], scores[index], offsets[index],
                selection_scores[index], eigenvalue_floor, tf.constant(max_condition_number, D),
                tf.constant(holdout_cap, D), projection_cap, require_raw_spd, jit_compile=jit_compile)
            one = structured_kernel(center, training[index], scores[index], offsets[index],
                selection_scores[index], config=configs[0], jit_compile=jit_compile)
            return index + 1, put(put(fits, 0, index, dense), 1, index, one)

        _, fits = tf.while_loop(lambda index, _: index < replicates, initial, (tf.constant(0), fits),
            maximum_iterations=replicates, parallel_iterations=1)
        one_passed = tf.reduce_all(fits["flags"][1, :, 2])
        one_report = one_stability(fits["precision"][1], fits["flags"][1, :, :2], caps, enabled, rank)
        attempted = (factor_max == 2) & (~(one_passed & one_report["passed"]) | (structured_target_family == "factor_2"))

        def additional():
            def step(index, fits):
                two = structured_kernel(center, training[index], scores[index], offsets[index],
                    selection_scores[index], config=configs[1], jit_compile=jit_compile)
                return index + 1, put(fits, 2, index, two)
            return tf.while_loop(lambda index, _: index < replicates, step, (tf.constant(0), fits),
                maximum_iterations=replicates, parallel_iterations=1)[1]

        if factor_max == 2:
            fits = tf.cond(attempted & (one_report["error"] == 0), additional, lambda: fits)
        arguments = (center, offsets, selection_scores, caps, enabled, rank, weights, tf.constant(holdout_cap, D))

        def without_second():
            result = select_two(fits["precision"][:2], fits["flags"][:2], *arguments)
            reports = tf.nest.map_structure(lambda value: tf.pad(value,
                [[0, 1]] + [[0, 0]] * (len(value.shape) - 1)), result["stability"])
            return {**result, "stability": reports}

        selected = (tf.cond(attempted,
            lambda: select_three(fits["precision"], fits["flags"], *arguments), without_second)
            if factor_max == 2 else without_second())

        def qualify():
            error = score_kernel(selected["precision"], center, audit, audit_scores)
            complete = tf.reduce_all(enabled)
            passed = error <= audit_cap
            status = tf.where(complete & passed & ~selected["diagonal_only"], 1, tf.where(passed, 2, 3))
            return error, tf.linalg.inv(selected["precision"]), status

        audit_error, covariance, status = tf.cond(selected["family_code"] != 0, qualify,
            lambda: (tf.constant(0., D), tf.zeros([dimension, dimension], D), tf.constant(0)))
        # These fitted records were frozen at the original host boundary.
        # Preserve that contract before an outer tape builds optimizer or
        # eigensolver pullbacks; internal objective derivatives remain intact.
        return tf.nest.map_structure(tf.stop_gradient,
            {"fits": fits, "one_stability": one_report, "one_passed": one_passed,
            "two_attempted": attempted, "selection": selected, "audit_error": audit_error,
            "covariance": covariance, "status": status})

    return compute
