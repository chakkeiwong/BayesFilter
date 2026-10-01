"""Native fixed-center regional curvature controller.

The callback supplies the original batched analytical target. All attempted
rows, partition order, replicate fits and holdout decisions stay in the graph.
Callers receive tensor records; completed host reporting is separate.
"""

from dataclasses import replace
from threading import RLock

import tensorflow as tf

from bayesfilter.inference.mass_matrix_tf import eigenpair_program
from bayesfilter.inference.score_curvature_tf import (
    _relative_response_rmse,
    fit_dense_score_precision_tf,
)
from bayesfilter.ops.stateless_random_tf import (
    philox_normal_float64,
    philox_uniform_float64,
)

D = tf.float64
STATUS = (
    "ok", "eligible_for_local_position_factor", "nonfinite_position",
    "ineligible_target_row", "nonfinite_target_value", "nonfinite_target_score",
    "nonfinite_transformed_score", "curvature_fit_numerical_failure",
    "curvature_fit_rejected", "replicate_numerical_failure", "replicate_instability",
    "nonfinite_consensus_precision", "audit_rejected", "factorization_failed",
    "factor_reconstruction_failed", "nonfinite_refined_center_score",
    "refined_proposal_rejected",
)
COUNT_NAMES = (
    "physical_rows", "logical_rows", "padded_rows", "eligibility_batches",
    "callback_batches", "target_rows", "target_logical_rows", "completed_logical_rows",
)
_LOCK = RLock()
_LAST_CONTROLLER = None


def _finite(value):
    return tf.reduce_all(tf.math.is_finite(value))


def _within(value, cap):
    return tf.math.is_finite(value) & (value <= tf.constant(cap, D))


def _symmetric(matrix):
    return 0.5 * matrix + 0.5 * tf.transpose(matrix)


def _project_scores(scores, factor):
    """Use the same contraction for the center and every batched score.

    GPU GEMV and GEMM can round identical rows differently. A one-ULP
    discrepancy then makes an exactly zero response report relative error
    one. This is the same score-times-factor product with a common sum axis.
    """
    return tf.reduce_sum(scores[:, :, None] * factor[None, :, :], axis=1)


def partition_seed(seed, index):
    return tf.random.experimental.stateless_fold_in(tf.stack((seed, 0)), index, alg="philox")


def draw_offsets(rows, dimension, seed, width, design):
    """Preserve the original non-XLA Philox box/ball stream."""
    radius = tf.constant(width, D)
    if design == "uniform_box":
        return philox_uniform_float64([rows, dimension], seed) * (2.0 * radius) - radius
    direction_seed = tf.random.experimental.stateless_fold_in(seed, 0, alg="philox")
    radius_seed = tf.random.experimental.stateless_fold_in(seed, 1, alg="philox")
    direction = philox_normal_float64([rows, dimension], direction_seed)
    norms = tf.linalg.norm(direction, axis=1, keepdims=True)
    unit = direction / norms
    radial = philox_uniform_float64([rows, 1], radius_seed)
    return unit * (radius * tf.pow(radial, tf.constant(1.0 / dimension, D)))


def make_partition_evaluator(callback, eligibility, dimension, batch_size, capacity, *, jit_compile=True):
    """Evaluate ordered fixed batches, charging failed attempts before gates."""
    if dimension < 1 or batch_size < 2 or capacity < 1:
        raise ValueError("positive dimension/capacity and batch_size>=2 required")
    batches = (capacity + batch_size - 1) // batch_size

    @tf.function(input_signature=[tf.TensorSpec([capacity, dimension], D), tf.TensorSpec([], tf.int32)],
                 autograph=False, jit_compile=jit_compile)
    def evaluate(points, point_count):
        initial = {
            "status": tf.constant(0), "batches": tf.constant(0),
            "counts": tf.zeros([8], tf.int64),
            "scores": tf.zeros([batches, batch_size, dimension], D),
        }

        def step(state):
            start = state["batches"] * batch_size
            count = tf.minimum(batch_size, point_count - start)
            indices = tf.minimum(start + tf.range(batch_size), point_count - 1)
            chunk = tf.gather(points, indices)
            counts = state["counts"] + tf.cast(tf.stack((batch_size, count, batch_size - count, 0, 0, 0, 0, 0)), tf.int64)
            attempted = {**state, "batches": state["batches"] + 1, "counts": counts}

            def check_eligibility():
                eligible = tf.convert_to_tensor(eligibility(chunk))
                if eligible.dtype != tf.bool:
                    raise TypeError("eligibility callback must return bool")
                if eligible.shape != (batch_size,):
                    raise ValueError("eligibility callback must return [batch_size]")
                checked = {**attempted, "counts": counts + tf.constant([0, 0, 0, 1, 0, 0, 0, 0], tf.int64)}

                def call_target():
                    values, scores = callback(chunk)
                    values, scores = tf.convert_to_tensor(values), tf.convert_to_tensor(scores)
                    if values.dtype != D or scores.dtype != D:
                        raise TypeError("target callback must return float64 values and scores")
                    if values.shape != (batch_size,) or scores.shape != chunk.shape:
                        raise ValueError("target callback returned an invalid batch shape")
                    status = tf.where(~_finite(values), 4, tf.where(~_finite(scores), 5, 0))
                    completed = tf.where(status == 0, count, 0)
                    charged = checked["counts"] + tf.cast(tf.stack((0, 0, 0, 0, 1, batch_size, count, completed)), tf.int64)
                    saved = tf.tensor_scatter_nd_update(state["scores"], tf.reshape(state["batches"], [1, 1]), scores[None])
                    return {**checked, "status": status, "counts": charged, "scores": saved}

                return tf.cond(tf.reduce_all(eligible), call_target, lambda: {**checked, "status": tf.constant(3)})

            return (tf.cond(_finite(chunk), check_eligibility, lambda: {**attempted, "status": tf.constant(2)}),)

        result, = tf.while_loop(lambda state: (state["status"] == 0) & (state["batches"] * batch_size < point_count),
            step, (initial,), maximum_iterations=batches, parallel_iterations=1)
        return {**result, "scores": tf.reshape(result["scores"], [-1, dimension])[:capacity]}

    return evaluate


def precision_spread(precisions, *, jit_compile=True):
    """Check every ordered combination once, including reciprocal eigenvalues."""
    replicates, dimension = precisions.shape[:2]

    def pair(first, second, spread):
        cholesky = tf.linalg.cholesky(tf.gather(precisions, first))
        solved = tf.linalg.triangular_solve(cholesky, tf.gather(precisions, second))
        transformed = _symmetric(tf.transpose(tf.linalg.triangular_solve(cholesky, tf.transpose(solved))))
        eigenvalues = (eigenpair_program(dimension)(transformed)[0] if jit_compile
                       else tf.linalg.eigvalsh(transformed))
        valid = _finite(eigenvalues) & tf.reduce_all(eigenvalues > 0.0)
        spread = tf.where(valid, tf.maximum(spread, tf.reduce_max(tf.maximum(eigenvalues, 1.0 / eigenvalues))),
                          tf.constant(float("inf"), D))
        next_first = tf.where(second + 1 < replicates, first, first + 1)
        next_second = tf.where(second + 1 < replicates, second + 1, first + 2)
        return next_first, next_second, spread

    return tf.while_loop(lambda first, _second, spread: (first < replicates - 1) & tf.math.is_finite(spread),
        pair, (tf.constant(0), tf.constant(1), tf.constant(1.0, D)),
        maximum_iterations=replicates * (replicates - 1) // 2, parallel_iterations=1)[2]


def posterior_curvature_controller(callback, eligibility, dimension, config, *, jit_compile=True):
    """Keep one target/configuration; seed and input geometry remain dynamic."""
    global _LAST_CONTROLLER
    static = replace(config, seed=0, lineage={})
    with _LOCK:
        previous = _LAST_CONTROLLER
        if (previous is not None and previous[0] is callback and previous[1] is eligibility
                and previous[2:5] == (dimension, static, jit_compile)):
            return previous[5]
        _LAST_CONTROLLER = None
        del previous
        program = make_posterior_curvature_controller(callback, eligibility, dimension, static, jit_compile=jit_compile)
        _LAST_CONTROLLER = (callback, eligibility, dimension, static, jit_compile, program)
        return program


def clear_posterior_curvature_controller_cache():
    """Release the Python cache entry; native executable eviction is not claimed."""
    global _LAST_CONTROLLER
    with _LOCK:
        _LAST_CONTROLLER = None


def make_posterior_curvature_controller(callback, eligibility, dimension, config, *, jit_compile=True):
    """Compile the whole fixed-center calculation; graph mode is a reference."""
    cfg = config
    rows = max(32, 4 * dimension) if cfg.rows_per_partition is None else cfg.rows_per_partition
    replicates, batch = cfg.replicate_count, cfg.batch_size
    fit_partitions = 2 * replicates
    partitions = fit_partitions + 3
    planned = batch + (fit_partitions + 2) * ((rows + batch - 1) // batch) * batch
    if dimension < 1 or rows < dimension or planned > cfg.max_physical_rows:
        raise ValueError("invalid dimension/design or whole refinement physical-row budget")
    evaluate = make_partition_evaluator(callback, eligibility, dimension, batch, rows, jit_compile=jit_compile)
    nan = float("nan")

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension, dimension], D),
                                 tf.TensorSpec([], tf.int32)], autograph=False, jit_compile=jit_compile)
    def refine(center, factor, seed):
        state = {
            "status": tf.constant(0), "role": tf.constant(0), "partition_count": tf.constant(0),
            "counts": tf.zeros([8], tf.int64), "partition_counts": tf.zeros([partitions, 8], tf.int64),
            "partition_seeds": tf.zeros([partitions, 2], tf.int32),
            "center_score": tf.zeros([dimension], D), "center_score_z": tf.zeros([dimension], D),
            "designs": tf.zeros([fit_partitions, rows, dimension], D),
            "scores_z": tf.zeros([fit_partitions, rows, dimension], D),
            "fit_count": tf.constant(0), "fit_precisions": tf.zeros([replicates, dimension, dimension], D),
            "fit_metrics": tf.zeros([replicates, 3], D), "fit_spd": tf.zeros([replicates], tf.bool),
            "fit_rank": tf.zeros([replicates], tf.int32), "fit_accepted": tf.zeros([replicates], tf.bool),
            "spread_done": tf.constant(False), "spread": tf.constant(nan, D),
            "audit_done": tf.constant(False), "audit_rmse": tf.constant(nan, D),
            "reconstruction_done": tf.constant(False), "factor_abs": tf.constant(nan, D), "factor_rel": tf.constant(nan, D),
            "norm_done": tf.constant(False), "center_norm": tf.constant(nan, D),
            "proposal_done": tf.constant(False), "proposal_rmse": tf.constant(nan, D),
            "precision": tf.zeros([dimension, dimension], D), "covariance": tf.zeros([dimension, dimension], D),
            "refined_factor": tf.zeros([dimension, dimension], D),
        }

        def record_partition(current, points, count, role, design_seed):
            result = evaluate(points, count)
            slot = tf.reshape(role, [1, 1])
            return ({**current, "status": result["status"], "role": role,
                "partition_count": current["partition_count"] + 1,
                "counts": current["counts"] + result["counts"],
                "partition_counts": tf.tensor_scatter_nd_update(current["partition_counts"], slot, result["counts"][None]),
                "partition_seeds": tf.tensor_scatter_nd_update(current["partition_seeds"], slot, design_seed[None]),
            }, result["scores"])

        state, center_scores = record_partition(state, tf.broadcast_to(center[None], [rows, dimension]),
            tf.constant(1), tf.constant(0), tf.zeros([2], tf.int32))
        center_score = center_scores[0]
        center_z = _project_scores(center_score[None], factor)[0]
        state = {**state, "center_score": center_score, "center_score_z": center_z,
                 "status": tf.where((state["status"] == 0) & ~_finite(center_z), 6, state["status"])}

        def partition(current):
            index = current["partition_count"] - 1
            design_seed = partition_seed(seed, index)
            offsets = draw_offsets(rows, dimension, design_seed, cfg.coordinate_half_width, cfg.fit_design)
            points = center[None] + tf.matmul(offsets, factor, transpose_b=True)
            output, scores = record_partition(current, points, tf.constant(rows), index + 1, design_seed)
            transformed = _project_scores(scores, factor)
            slot = tf.reshape(index, [1, 1])
            return ({**output,
                "status": tf.where((output["status"] == 0) & ~_finite(transformed), 6, output["status"]),
                "designs": tf.tensor_scatter_nd_update(current["designs"], slot, offsets[None]),
                "scores_z": tf.tensor_scatter_nd_update(current["scores_z"], slot, transformed[None]),
            },)

        state, = tf.while_loop(lambda current: (current["status"] == 0) & (current["partition_count"] <= fit_partitions),
            partition, (state,), maximum_iterations=fit_partitions, parallel_iterations=1)
        state = {**state, "role": tf.where(state["status"] == 0, fit_partitions + 3, state["role"])}

        def fit_replicate(current):
            index = current["fit_count"]
            fit = fit_dense_score_precision_tf(current["center_score_z"],
                tf.gather(current["designs"], index), tf.gather(current["scores_z"], index),
                selection_offsets=tf.gather(current["designs"], index + replicates),
                selection_scores=tf.gather(current["scores_z"], index + replicates))
            valid = (_finite(fit["raw_precision"]) & _finite(fit["raw_eigenvalues"]) & fit["raw_spd"]
                & (fit["design_rank"] == dimension)
                & _within(fit["design_condition"], cfg.max_design_condition_number)
                & _within(fit["precision_condition"], cfg.max_precision_condition_number)
                & _within(fit["selection_relative_rmse"], cfg.selection_relative_rmse_cap))
            slot = tf.reshape(index, [1, 1])
            metrics = tf.stack((fit["design_condition"], fit["precision_condition"], fit["selection_relative_rmse"]))
            return ({**current, "status": tf.where(valid, 0, 8), "fit_count": index + 1,
                "fit_precisions": tf.tensor_scatter_nd_update(current["fit_precisions"], slot, fit["raw_precision"][None]),
                "fit_metrics": tf.tensor_scatter_nd_update(current["fit_metrics"], slot, metrics[None]),
                "fit_spd": tf.tensor_scatter_nd_update(current["fit_spd"], slot, fit["raw_spd"][None]),
                "fit_rank": tf.tensor_scatter_nd_update(current["fit_rank"], slot, fit["design_rank"][None]),
                "fit_accepted": tf.tensor_scatter_nd_update(current["fit_accepted"], slot, valid[None]),
            },)

        state, = tf.while_loop(lambda current: (current["status"] == 0) & (current["fit_count"] < replicates),
            fit_replicate, (state,), maximum_iterations=replicates, parallel_iterations=1)

        def consensus(current):
            spread = precision_spread(current["fit_precisions"], jit_compile=jit_compile)
            precision = _symmetric(tf.reduce_sum(current["fit_precisions"] / replicates, axis=0))
            status = tf.where(~_within(spread, cfg.replicate_generalized_eigenvalue_spread_cap), 10,
                              tf.where(~_finite(precision), 11, 0))
            return {**current, "role": tf.constant(fit_partitions + 4), "status": status,
                    "spread_done": tf.constant(True), "spread": spread, "precision": precision}

        state = tf.cond(state["status"] == 0, lambda: consensus(state), lambda: state)

        def audit(current):
            design_seed = partition_seed(seed, tf.constant(fit_partitions))
            offsets = draw_offsets(rows, dimension, design_seed, cfg.coordinate_half_width, cfg.fit_design)
            points = center[None] + tf.matmul(offsets, factor, transpose_b=True)
            output, scores = record_partition(current, points, tf.constant(rows), tf.constant(fit_partitions + 1), design_seed)

            def compare():
                rmse = _relative_response_rmse(current["precision"], current["center_score_z"], offsets, _project_scores(scores, factor))
                return {**output, "audit_done": tf.constant(True), "audit_rmse": rmse,
                        "status": tf.where(_within(rmse, cfg.audit_relative_rmse_cap), 0, 12)}

            return tf.cond(output["status"] == 0, compare, lambda: output)

        state = tf.cond(state["status"] == 0, lambda: audit(state), lambda: state)

        def factorize(current):
            current = {**current, "role": tf.constant(fit_partitions + 5)}
            cholesky = tf.linalg.cholesky(current["precision"])
            inverse_action = tf.linalg.triangular_solve(cholesky, tf.transpose(factor))
            covariance = _symmetric(tf.matmul(inverse_action, inverse_action, transpose_a=True))
            refined = tf.linalg.cholesky(covariance)
            valid = _finite(cholesky) & _finite(covariance) & _finite(refined) & tf.reduce_all(tf.linalg.diag_part(refined) > 0.0)

            def reconstruction():
                residual = refined @ tf.transpose(refined) - covariance
                absolute = tf.reduce_max(tf.abs(residual))
                scale = tf.reduce_max(tf.abs(covariance))
                relative = tf.linalg.norm(residual / scale) / tf.linalg.norm(covariance / scale)
                passed = (tf.math.is_finite(absolute) & tf.math.is_finite(relative)
                    & ((absolute <= tf.constant(cfg.factor_absolute_tolerance, D)) | (relative <= tf.constant(cfg.factor_relative_tolerance, D))))
                output = {**current, "covariance": covariance, "refined_factor": refined,
                    "reconstruction_done": tf.constant(True), "factor_abs": absolute, "factor_rel": relative,
                    "status": tf.where(passed, 0, 14)}

                def center_norm():
                    normalized = tf.linalg.matvec(refined, current["center_score"], transpose_a=True)
                    scale = tf.reduce_max(tf.abs(normalized))
                    norm = scale * tf.linalg.norm(tf.math.divide_no_nan(normalized, scale))
                    return {**output, "norm_done": tf.constant(True), "center_norm": norm,
                            "status": tf.where(tf.math.is_finite(norm), 0, 15)}

                return tf.cond(passed, center_norm, lambda: output)

            return tf.cond(valid, reconstruction, lambda: {**current, "status": tf.constant(13)})

        state = tf.cond(state["status"] == 0, lambda: factorize(state), lambda: state)

        def proposal(current):
            design_seed = partition_seed(seed, tf.constant(fit_partitions + 1))
            latent = philox_normal_float64([rows, dimension], design_seed)
            delta = tf.matmul(latent, current["refined_factor"], transpose_b=True)
            output, scores = record_partition(current, center[None] + delta, tf.constant(rows),
                tf.constant(fit_partitions + 2), design_seed)

            def compare():
                offsets = tf.transpose(tf.linalg.triangular_solve(factor, tf.transpose(delta)))
                rmse = _relative_response_rmse(current["precision"], current["center_score_z"], offsets, _project_scores(scores, factor))
                return {**output, "proposal_done": tf.constant(True), "proposal_rmse": rmse,
                        "status": tf.where(_within(rmse, cfg.proposal_relative_rmse_cap), 1, 16)}

            return tf.cond(output["status"] == 0, compare, lambda: output)

        state = tf.cond(state["status"] == 0, lambda: proposal(state), lambda: state)
        del state["designs"], state["scores_z"], state["center_score"], state["center_score_z"]
        return state

    return refine
