"""Internal prepared-cloud posterior curvature recurrence.

Partition evaluation, fixed-center fitting and exact selection reuse their shared
authorities. The enclosing public initializer is qualified separately.
"""

import tensorflow as tf

from bayesfilter.inference.dense_initializer_cloud_tf import (
    make_dense_initializer_cloud_program,
)
from bayesfilter.inference.dense_partition_validation_tf import (
    make_dense_partition_validation_program,
)
from bayesfilter.inference.dense_validated_fit_tf import (
    make_dense_validated_fit_program,
)
from bayesfilter.inference.factor_correlation_geometry import (
    FactorCorrelationGeometryConfig,
)
from bayesfilter.inference.posterior_candidate_ledger_tf import candidate_ledger_program
from bayesfilter.inference.posterior_local_initializer import _cloud_row_counts
from bayesfilter.inference.program_cache_scope import ProgramCacheScope
from bayesfilter.inference.quadratic_geometry import _compiled_cloud
from bayesfilter.inference.quadratic_geometry_control_tf import _callback_program

D = tf.float64
I = tf.int32
STATUSES = (
    "pending", "usable_posterior_local_initializer", "curvature_center_invalid",
    "curvature_cloud_invalid", "eligibility_contract_mismatch",
    "exact_evaluation_budget_exhausted", "curvature_not_centered_within_attempt_budget",
    "curvature_result", "physical_marginal_scale_invalid", "curvature_fit_exception",
)


def _deferred_configuration_fit(dimension, replicates, training, selection, audit, *, jit_compile):
    """Preserve partition-validation precedence before a known static fit error."""
    validate = make_dense_partition_validation_program(dimension, replicates, training, selection,
        audit, jit_compile=jit_compile)

    @tf.function(input_signature=validate.input_signature, jit_compile=jit_compile, autograph=False)
    def rejected(center, center_score, offsets, scores):
        validation = validate(center, center_score, offsets, scores)
        return {"validation": validation, "fit_ran": tf.constant(False),
            "fit_error_code": tf.constant(0, I), "usable": tf.constant(False),
            "fit": {"selection": {"precision": tf.zeros([dimension, dimension], D)},
                    "covariance": tf.zeros([dimension, dimension], D)}}

    return rejected


def make_posterior_curvature_program(evaluator, dimension, config, thresholds, *, jit_compile=True):
    """Continue from a finite incumbent without resetting the owner's tracker.

    Clouds are fixed-capacity operands. The owner serializes invocations and
    resets shared resources only at the complete initializer boundary.
    """
    attempts, replicates = config.max_curvature_attempts, config.replicate_count
    training, selection, audit = _cloud_row_counts(config, dimension)
    partitions, capacity = 2 * replicates + 1, max(training, selection, audit)
    rows_per_attempt = replicates * (training + selection) + audit
    maximum = 1 + attempts * (1 + rows_per_attempt)
    scope = ProgramCacheScope()
    scalar = evaluator.scalar
    batched = evaluator.batched if evaluator.batched_fn is not None else None

    def evaluate_rows(points):
        # Build the existing callback body without putting resource-owning
        # callbacks into its global LRU. Partition extents are static at trace.
        callback = _compiled_cloud.__wrapped__(scalar, batched, points.shape[0], dimension)
        values, scores = callback(points)
        finite = tf.math.is_finite(values) & tf.reduce_all(tf.math.is_finite(scores), axis=1)
        nan = tf.constant(float("nan"), D)
        return tf.where(finite, values, nan), tf.where(finite[:, None], scores, nan), finite

    with scope.activate():
        target = _callback_program(scalar, dimension, jit_compile)
        cloud = make_dense_initializer_cloud_program(evaluate_rows, dimension, replicates,
            training, selection, audit, jit_compile=jit_compile)
        # The original creates this configuration only after a complete cloud
        # passes partition validation. Preserve that exception boundary.
        try:
            FactorCorrelationGeometryConfig(factor_count=1,
                max_condition_number=config.max_condition_number,
                holdout_score_relative_rmse=thresholds.selection_holdout_relative_rmse_cap)
            configuration_error = False
        except ValueError:
            configuration_error = True
        fitter = (_deferred_configuration_fit(dimension, replicates, training, selection,
            audit, jit_compile=jit_compile) if configuration_error else
            make_dense_validated_fit_program(dimension, replicates, training, selection,
                audit, thresholds=thresholds, factor_max=config.factor_max,
                dense_eigenvalue_floor=config.dense_eigenvalue_floor,
                max_condition_number=config.max_condition_number,
                shrinkage_weights=config.shrinkage_weights,
                structured_target_family=config.structured_target_family, jit_compile=jit_compile))
        choose = candidate_ledger_program(maximum, dimension, jit_compile=jit_compile)
        cloud_template = cloud.get_concrete_function().structured_outputs
        fit_template = fitter.get_concrete_function().structured_outputs

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([], D),
        tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([attempts, partitions, capacity, dimension], D)],
        jit_compile=jit_compile, autograph=False)
    def run(initial, value, score, scale, offsets):
        zero_cloud = tf.nest.map_structure(lambda x: tf.zeros(x.shape, x.dtype), cloud_template)
        zero_fit = tf.nest.map_structure(lambda x: tf.zeros(x.shape, x.dtype), fit_template)
        row = tf.range(rows_per_attempt)
        training_end = replicates * training
        selection_end = training_end + replicates * selection
        partition_indices = tf.where(row < training_end, row // training,
            tf.where(row < selection_end, replicates + (row - training_end) // selection, 2 * replicates))
        local_rows = tf.where(row < training_end, row % training,
            tf.where(row < selection_end, (row - training_end) % selection, row - selection_end))
        cloud_addresses = tf.stack((partition_indices, local_rows), axis=1)
        state = {
            "attempt_count": tf.constant(0, I), "record_count": tf.constant(0, I),
            "status": tf.constant(0, I), "center": initial, "value": value, "score": score,
            "center_score_z": score * scale,
            "ledger_count": tf.constant(1, I),
            "ledger_positions": tf.concat((initial[None], tf.zeros([maximum - 1, dimension], D)), 0),
            "ledger_values": tf.concat((value[None], tf.zeros([maximum - 1], D)), 0),
            "ledger_scores": tf.concat((score[None], tf.zeros([maximum - 1, dimension], D)), 0),
            "ledger_attempts": tf.fill([maximum], tf.constant(-1, I)),
            "ledger_partitions": tf.fill([maximum], tf.constant(-2, I)),
            "ledger_promoted": tf.concat((tf.constant([True]), tf.zeros([maximum - 1], tf.bool)), 0),
            "ledger_scaled_score_l2": tf.concat((tf.linalg.norm(score * scale)[None], tf.zeros([maximum - 1], D)), 0),
            "centers": tf.zeros([attempts, dimension], D),
            "center_moved": tf.zeros([attempts], tf.bool),
            "scaled_moves": tf.zeros([attempts], D), "improvements": tf.zeros([attempts], D),
            "fit_attempts": tf.zeros([attempts], tf.bool),
            "cloud_history": tf.nest.map_structure(lambda x: tf.zeros([attempts, *x.shape], x.dtype), cloud_template),
            "fit": zero_fit, "fit_attempted": tf.constant(False),
            "precision_theta": tf.zeros([dimension, dimension], D),
            "covariance_theta": tf.zeros([dimension, dimension], D),
            "marginal": tf.zeros([dimension], D), "scale_log": tf.zeros([dimension], D),
        }

        def tracker_status(otherwise):
            return tf.where(evaluator.mismatch_rows.read_value() > 0, 4,
                tf.where(evaluator.budget_exhausted.read_value(), 5, otherwise))

        def step(current):
            index = current["attempt_count"]
            center = current["center"]
            center_value, center_score = target(center)
            center_valid = tf.math.is_finite(center_value) & tf.reduce_all(tf.math.is_finite(center_score))
            prepared = tf.gather(offsets, index)
            raw = tf.cond(center_valid, lambda: cloud(center, scale, center_value, prepared), lambda: zero_cloud)

            def append_valid_partitions():
                # The failed partition contributes no rows, including any
                # finite prefix. Earlier valid partitions remain in the ledger.
                complete = raw["partition_count"] - tf.cast(~raw["valid"], I)
                active_rows = tf.reduce_sum(tf.cast(partition_indices < complete, I))
                start = current["ledger_count"]
                addresses = (start + tf.range(1 + rows_per_attempt))[:, None]
                positions = tf.concat((center[None], tf.gather_nd(raw["positions"], cloud_addresses)), 0)
                values = tf.concat((center_value[None], tf.gather_nd(raw["values"], cloud_addresses)), 0)
                scores = tf.concat((center_score[None], tf.gather_nd(raw["scores"], cloud_addresses)), 0)
                ledger_positions = tf.tensor_scatter_nd_update(current["ledger_positions"], addresses, positions)
                ledger_values = tf.tensor_scatter_nd_update(current["ledger_values"], addresses, values)
                ledger_scores = tf.tensor_scatter_nd_update(current["ledger_scores"], addresses, scores)
                count = start + 1 + active_rows
                chosen = choose(ledger_positions, ledger_values, ledger_scores,
                    tf.ones([maximum], tf.bool), scale, count)
                return {
                    "ledger_positions": ledger_positions, "ledger_values": ledger_values,
                    "ledger_scores": ledger_scores, "ledger_count": count,
                    "ledger_attempts": tf.tensor_scatter_nd_update(current["ledger_attempts"], addresses,
                        tf.fill([1 + rows_per_attempt], index)),
                    "ledger_partitions": tf.tensor_scatter_nd_update(current["ledger_partitions"], addresses,
                        tf.concat((tf.constant([-1], I), partition_indices), 0)),
                    "ledger_promoted": chosen["promoted"],
                    "ledger_scaled_score_l2": chosen["scaled_score_l2"],
                    "chosen_position": chosen["position"], "chosen_value": chosen["value"],
                    "chosen_score": chosen["score"],
                }

            appended = tf.cond(center_valid, append_valid_partitions, lambda: {
                "ledger_positions": current["ledger_positions"], "ledger_values": current["ledger_values"],
                "ledger_scores": current["ledger_scores"], "ledger_count": current["ledger_count"],
                "ledger_attempts": current["ledger_attempts"], "ledger_partitions": current["ledger_partitions"],
                "ledger_promoted": current["ledger_promoted"],
                "ledger_scaled_score_l2": current["ledger_scaled_score_l2"],
                "chosen_position": center, "chosen_value": current["value"], "chosen_score": current["score"],
            })
            chosen_position = appended.pop("chosen_position")
            chosen_value = appended.pop("chosen_value")
            chosen_score = appended.pop("chosen_score")
            complete_cloud = center_valid & raw["valid"]
            moved = (chosen_value > center_value) & ~tf.reduce_all(chosen_position == center)
            fit_attempted = complete_cloud & ~moved
            fit = tf.cond(fit_attempted,
                lambda: fitter(center, center_score * scale, prepared, raw["scaled_scores"]), lambda: zero_fit)
            usable = fit_attempted & fit["usable"]

            def physical():
                precision = fit["fit"]["selection"]["precision"] / (scale[:, None] * scale[None, :])
                covariance = fit["fit"]["covariance"] * scale[:, None] * scale[None, :]
                marginal = tf.sqrt(tf.linalg.diag_part(covariance))
                finite = tf.reduce_all(tf.math.is_finite(marginal) & (marginal > 0.))
                return precision, covariance, marginal, tf.math.log(marginal), finite

            precision, covariance, marginal, scale_log, physical_valid = tf.cond(usable, physical,
                lambda: (tf.zeros([dimension, dimension], D), tf.zeros([dimension, dimension], D),
                         tf.zeros([dimension], D), tf.zeros([dimension], D), tf.constant(False)))
            with tf.control_dependencies(tf.nest.flatten(raw)):
                invalid_status = tracker_status(tf.where(center_valid, 3, 2))
            fit_status = tf.where(~fit["fit_ran"] | (fit["fit_error_code"] != 0), 9,
                tf.where(~fit["usable"], 7, tf.where(physical_valid, 1, 8)))
            status = tf.where(~complete_cloud, invalid_status,
                tf.where(moved, tf.where(index + 1 == attempts, 6, 0), fit_status))
            address = tf.reshape(index, [1, 1])

            def record(history, item):
                if item.shape.num_elements() == 0:
                    return history
                return tf.tensor_scatter_nd_update(history, address, item[None])

            result = {**current, **appended,
                "attempt_count": index + 1,
                "record_count": current["record_count"] + tf.cast(complete_cloud, I),
                "status": status,
                "center": tf.where(complete_cloud & moved, chosen_position, center),
                "value": tf.where(complete_cloud & moved, chosen_value, center_value),
                "score": tf.where(complete_cloud & moved, chosen_score, center_score),
                "center_score_z": center_score * scale,
                "centers": record(current["centers"], center),
                "center_moved": record(current["center_moved"], complete_cloud & moved),
                "scaled_moves": record(current["scaled_moves"], tf.linalg.norm((chosen_position - center) / scale)),
                "improvements": record(current["improvements"], chosen_value - center_value),
                "fit_attempts": record(current["fit_attempts"], fit_attempted),
                "cloud_history": tf.nest.map_structure(record, current["cloud_history"], raw),
                "fit": fit, "fit_attempted": fit_attempted,
                "precision_theta": precision, "covariance_theta": covariance,
                "marginal": marginal, "scale_log": scale_log,
            }
            return (result,)

        result, = tf.while_loop(lambda current: (current["status"] == 0) & (current["attempt_count"] < attempts),
            step, (state,), parallel_iterations=1)
        return tf.nest.map_structure(tf.stop_gradient, {
            **result, "configuration_error": tf.constant(configuration_error),
            "fit_jit_compile": tf.constant(jit_compile)})

    with scope.activate():
        run.get_concrete_function()
    run.dependency_scope = scope
    return run
