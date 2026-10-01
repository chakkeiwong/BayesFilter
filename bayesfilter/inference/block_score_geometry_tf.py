"""Complete XLA programs for the existing fixed-center block score fit.

The public wrapper owns validation of static schemas and report construction.
Numerical decisions, including ordered early exits, stay inside these programs.
This preparation helper is neither a target score nor a tuning authority.
"""

import math
import sys
from functools import lru_cache

import tensorflow as tf

from bayesfilter.inference.fixed_center_curvature import _precision_geometry_kernel
from bayesfilter.ops.accurate_svd_tf import accurate_svd
from bayesfilter.ops.qr_lstsq_tf import complete_orthogonal_lstsq
from bayesfilter.ops.symmetric_matrix_tf import symmetric_score_design, unpack_symmetric

D = tf.float64
STATUSES = (
    "qualified_for_hmc_initialization",
    "nonfinite_fit_inputs",
    "block_design_rank_deficient",
    "raw_block_precision_not_spd",
    "block_condition_number_rejected",
    "selection_score_fit_rejected",
    "replicate_stability_rejected",
    "audit_score_fit_rejected",
    "offblock_curvature_rejected",
    "precision_covariance_identity_rejected",
    "invalid_principal_subspace_rank",
)
CONTROL_FIELDS = (
    "ridge", "max_condition_number", "selection_relative_rmse_cap",
    "audit_relative_rmse_cap", "unexplained_response_fraction_cap",
    "generalized_eigenvalue_spread_cap", "trace_normalized_frobenius_cap",
    "trace_normalized_operator_cap", "principal_angle_degrees_cap",
)


def _block_fit(offsets, response, ridge_value, condition_cap):
    rows, dimension = offsets.shape
    if rows == 0:
        # The empty design and offset matrices have rank zero. XLA's SVD
        # lowering cannot slice a zero row extent; preserve the early rejection.
        return tf.zeros([dimension, dimension], D), tf.constant([0., 0., 0., 0., 0., 0., 2.], D), tf.constant(2)
    count = dimension * (dimension + 1) // 2
    design = tf.reshape(symmetric_score_design(offsets, dimension), [-1, count])
    singular = accurate_svd(design, compute_uv=False)
    tolerance = tf.reduce_max(singular) * tf.cast(rows * dimension, D) * sys.float_info.epsilon
    rank = tf.math.count_nonzero(singular > tolerance, dtype=tf.int32)
    row_singular = accurate_svd(offsets, compute_uv=False)
    row_tolerance = tf.reduce_max(row_singular) * tf.cast(rows, D) * sys.float_info.epsilon
    row_rank = tf.math.count_nonzero(row_singular > row_tolerance, dtype=tf.int32)

    def fit():
        ridge = tf.sqrt(ridge_value) * tf.eye(count, dtype=D)
        coefficients = complete_orthogonal_lstsq(
            tf.concat((design, ridge), axis=0),
            tf.concat((tf.reshape(response, [-1, 1]), tf.zeros([count, 1], D)), axis=0),
        )[:, 0]
        precision = unpack_symmetric(coefficients, dimension)
        eigenvalues = tf.linalg.eigvalsh(precision)
        minimum, maximum = tf.reduce_min(eigenvalues), tf.reduce_max(eigenvalues)
        nonpositive = tf.math.count_nonzero(eigenvalues <= 0., dtype=tf.int32)
        condition = tf.where(minimum <= 0., tf.constant(math.inf, D), maximum / minimum)
        status = tf.where(nonpositive > 0, 3,
            tf.where(~tf.math.is_finite(condition) | (condition > condition_cap), 4, 0))
        return precision, tf.stack([minimum, maximum, tf.cast(nonpositive, D), condition]), status

    precision, summary, status = tf.cond((rank < count) | (row_rank < dimension),
        lambda: (tf.zeros([dimension, dimension], D), tf.zeros([4], D), tf.constant(2)), fit)
    report = tf.concat([tf.cast(tf.stack([rank, row_rank]), D), summary, [tf.cast(status, D)]], 0)
    return precision, report, status


def _score_diagnostics(precision, center, offsets, scores):
    response = center[None, :] - scores
    prediction = tf.matmul(offsets, precision, transpose_b=True)
    residual = prediction - response
    response_scale = tf.maximum(tf.sqrt(tf.reduce_mean(tf.square(response))), tf.constant(1e-15, D))
    rmse = tf.sqrt(tf.reduce_mean(tf.square(residual))) / response_scale
    unexplained = tf.linalg.norm(residual) / tf.maximum(tf.linalg.norm(response), tf.constant(1e-15, D))
    return tf.stack([rmse, unexplained])


def _pair_diagnostics(first, second, rank, controls, *, jit_compile=True):
    # Match the public comparison's symmetry projection before its eigensystems.
    first, second = .5 * (first + tf.transpose(first)), .5 * (second + tf.transpose(second))
    left_values, right_values, positive_rank, angles, generalized, frobenius, operator = (
        _precision_geometry_kernel(first, second, tf.constant(0., D), rank, jit_compile=jit_compile))
    left_nonpositive = tf.math.count_nonzero(left_values <= 0., dtype=tf.int32)
    right_nonpositive = tf.math.count_nonzero(right_values <= 0., dtype=tf.int32)
    spd = tf.reduce_all(left_values > 0.) & tf.reduce_all(right_values > 0.)
    maximum_angle = tf.reduce_max(tf.where(tf.range(first.shape[0]) < positive_rank,
        angles, tf.constant(-math.inf, D)))
    checks = tf.stack([
        spd & (generalized[2] <= controls[5]), frobenius <= controls[6],
        operator <= controls[7], (positive_rank > 0) & (maximum_angle <= controls[8]),
    ])
    report = tf.concat([left_values, right_values, angles,
        tf.stack([tf.cast(positive_rank, D), *tf.unstack(generalized), frobenius, operator,
            maximum_angle, tf.cast(left_nonpositive, D), tf.cast(right_nonpositive, D), tf.cast(spd, D)])], 0)
    return report, checks


def _qualification_status(selection_passed, stability_passed, summary, controls):
    """Preserve the public precedence, including strict > versus <= tests."""
    return tf.where(~selection_passed | (summary[0] > controls[2]), 5,
        tf.where(~stability_passed, 6,
        tf.where(summary[2] > controls[3], 7,
        tf.where(summary[3] > controls[4], 8,
        tf.where(summary[4] > 1e-8, 9, 0)))))


@lru_cache(maxsize=64)
def fit_program(dimension, replicate_count, training_rows, selection_rows, audit_rows,
                partition, *, jit_compile=True):
    """One traced body per loop; only distinct block widths specialize fits."""
    widths = tuple(sorted({stop - start for start, stop in partition}))
    branch_ids = tuple(widths.index(stop - start) for start, stop in partition)
    starts = tuple(start for start, _ in partition)
    block_count = len(partition)
    pair_count = replicate_count * (replicate_count - 1) // 2
    pair_width = 3 * dimension + 10

    @tf.function(input_signature=[
        tf.TensorSpec([dimension], D),
        tf.TensorSpec([replicate_count, training_rows, dimension], D),
        tf.TensorSpec([replicate_count, training_rows, dimension], D),
        tf.TensorSpec([replicate_count, selection_rows, dimension], D),
        tf.TensorSpec([replicate_count, selection_rows, dimension], D),
        tf.TensorSpec([audit_rows, dimension], D), tf.TensorSpec([audit_rows, dimension], D),
        tf.TensorSpec([len(CONTROL_FIELDS)], D), tf.TensorSpec([], tf.int32),
    ], jit_compile=jit_compile, autograph=False)
    def compute(center, training, training_scores, selection, selection_scores,
                audit, audit_scores, controls, requested_rank):
        finite = tf.reduce_all(tf.math.is_finite(tf.concat([
            tf.reshape(center, [-1]), tf.reshape(training, [-1]), tf.reshape(training_scores, [-1]),
            tf.reshape(selection, [-1]), tf.reshape(selection_scores, [-1]),
            tf.reshape(audit, [-1]), tf.reshape(audit_scores, [-1]),
        ], 0)))

        def replicate_step(index, _status, precisions, block_reports, block_counts, selections):
            response = center[None, :] - training_scores[index]

            def block_step(block_index, _failure, precision, reports):
                start = tf.gather(tf.constant(starts), block_index)

                def branch(width):
                    def update():
                        offsets = tf.slice(training[index], [0, start], [training_rows, width])
                        block_response = tf.slice(response, [0, start], [training_rows, width])
                        block, report, status = _block_fit(offsets, block_response, controls[0], controls[1])
                        indices = tf.range(width) + start
                        grid = tf.stack(tf.meshgrid(indices, indices, indexing="ij"), -1)
                        updated = tf.tensor_scatter_nd_update(precision, tf.reshape(grid, [-1, 2]),
                            tf.reshape(block, [-1]))
                        return tf.where(status == 0, updated, precision), report, status
                    return update

                updated, report, status = tf.switch_case(tf.gather(tf.constant(branch_ids), block_index),
                    branch_fns=tuple(branch(width) for width in widths))
                return block_index + 1, status, updated, tf.tensor_scatter_nd_update(
                    reports, [[block_index]], [report])

            completed, status, precision, reports = tf.while_loop(
                lambda block_index, failure, *_: (block_index < block_count) & (failure == 0),
                block_step, (tf.constant(0), tf.constant(0), tf.zeros([dimension, dimension], D),
                    tf.zeros([block_count, 7], D)),
                maximum_iterations=block_count, parallel_iterations=1)
            metrics = tf.cond(status == 0, lambda: _score_diagnostics(precision, center,
                selection[index], selection_scores[index]), lambda: tf.zeros([2], D))
            return (index + 1, status, tf.tensor_scatter_nd_update(precisions, [[index]], [precision]),
                tf.tensor_scatter_nd_update(block_reports, [[index]], [reports]),
                tf.tensor_scatter_nd_update(block_counts, [[index]], [completed]),
                tf.tensor_scatter_nd_update(selections, [[index]], [metrics]))

        completed, status, precisions, reports, counts, selections = tf.while_loop(
            lambda index, failure, *_: (index < replicate_count) & (failure == 0),
            replicate_step, (tf.constant(0), tf.where(finite, 0, 1),
                tf.zeros([replicate_count, dimension, dimension], D),
                tf.zeros([replicate_count, block_count, 7], D), tf.zeros([replicate_count], tf.int32),
                tf.zeros([replicate_count, 2], D)),
            maximum_iterations=replicate_count, parallel_iterations=1)
        rank = tf.minimum(requested_rank, dimension - 1)
        status = tf.where((status == 0) & (rank <= 0), 10, status)

        def qualify():
            def pair_step(index, left, right, pair_reports, checks):
                report, passed = _pair_diagnostics(precisions[left], precisions[right], rank, controls,
                    jit_compile=jit_compile)
                next_left = tf.where(right + 1 == replicate_count, left + 1, left)
                next_right = tf.where(right + 1 == replicate_count, left + 2, right + 1)
                return (index + 1, next_left, next_right,
                    tf.tensor_scatter_nd_update(pair_reports, [[index]], [report]),
                    tf.tensor_scatter_nd_update(checks, [[index]], [passed]))

            _, _, _, pair_reports, checks = tf.while_loop(lambda index, *_: index < pair_count,
                pair_step, (tf.constant(0), tf.constant(0), tf.constant(1),
                    tf.zeros([pair_count, pair_width], D), tf.zeros([pair_count, 4], tf.bool)),
                maximum_iterations=pair_count, parallel_iterations=1)
            consensus = tf.reduce_mean(precisions, axis=0)

            def selection_step(index, values):
                metrics = _score_diagnostics(consensus, center, selection[index], selection_scores[index])
                return index + 1, tf.tensor_scatter_nd_update(values, [[index]], [metrics])

            _, consensus_selections = tf.while_loop(lambda index, _: index < replicate_count,
                selection_step, (tf.constant(0), tf.zeros([replicate_count, 2], D)),
                maximum_iterations=replicate_count, parallel_iterations=1)
            audit_metrics = _score_diagnostics(consensus, center, audit, audit_scores)
            covariance = tf.linalg.cholesky_solve(tf.linalg.cholesky(consensus), tf.eye(dimension, dtype=D))
            inverse_residual = tf.reduce_max(tf.abs(tf.matmul(consensus, covariance) - tf.eye(dimension, dtype=D)))
            summary = tf.concat([tf.reduce_mean(consensus_selections, axis=0), audit_metrics, [inverse_residual]], 0)
            final_status = _qualification_status(tf.reduce_all(selections[:, 0] <= controls[2]),
                tf.reduce_all(checks), summary, controls)
            return final_status, consensus, covariance, pair_reports, checks, summary

        final_status, precision, covariance, pair_reports, checks, summary = tf.cond(status == 0, qualify,
            lambda: (status, tf.zeros([dimension, dimension], D), tf.zeros([dimension, dimension], D),
                tf.zeros([pair_count, pair_width], D), tf.zeros([pair_count, 4], tf.bool), tf.zeros([5], D)))
        # Public preparation records were frozen NumPy arrays in the baseline.
        return {"status": final_status, "replicate_count": completed, "block_counts": counts,
            "blocks": reports, "selection": selections, "pairs": pair_reports, "checks": checks,
            "pair_passed": tf.reduce_all(checks, axis=1),
            "stability_passed": tf.reduce_all(checks), "summary": summary,
            "precision": tf.stop_gradient(precision), "covariance": tf.stop_gradient(covariance)}

    return compute


@lru_cache(maxsize=64)
def position_program(dimension, *, jit_compile=True):
    @tf.function(input_signature=[tf.TensorSpec([dimension, dimension], D),
        tf.TensorSpec([dimension, dimension], D), tf.TensorSpec([dimension], D)],
        jit_compile=jit_compile, autograph=False)
    def compute(precision, covariance, scale):
        valid = tf.reduce_all(tf.math.is_finite(scale) & (scale > 0.))

        def transform():
            scale_matrix, inverse_scale = tf.linalg.diag(scale), tf.linalg.diag(1. / scale)
            scaled_covariance = tf.matmul(tf.matmul(scale_matrix, covariance), scale_matrix)
            scaled_precision = tf.matmul(tf.matmul(inverse_scale, precision), inverse_scale)
            return scaled_precision, scaled_covariance, tf.linalg.cholesky(scaled_covariance)

        scaled = tf.cond(valid, transform, lambda: (tf.zeros_like(precision),
            tf.zeros_like(covariance), tf.zeros_like(covariance)))
        return *scaled, valid

    return compute
