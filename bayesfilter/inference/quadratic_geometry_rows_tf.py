"""Compact-row operations with fixed storage and a runtime sample count.

Only shape-dependent QR, projections and reductions use static shape dispatch.
The enclosing geometry kernel owns the shared SVD and numerical decisions.
"""

import tensorflow as tf

from bayesfilter.ops.qr_lstsq_tf import _active_row_operation


def compact_qr(design, count, dimension):
    capacity = design.shape[0]

    def operation(samples):
        rows = samples * dimension
        q, r = tf.linalg.qr(design[:rows], full_matrices=False)
        return tf.pad(q, [[0, capacity - rows], [0, 0]]), r

    return _active_row_operation(operation, count, 1, capacity // dimension)


def compact_solution(q, left_r, right, inverse, response, count, dimension):
    def operation(samples):
        rows = samples * dimension
        left = tf.matmul(q[:rows], left_r)
        return tf.linalg.matvec(right, inverse * tf.linalg.matvec(left, response[:rows], transpose_a=True))

    return _active_row_operation(operation, count, 1, q.shape[0] // dimension)


def compact_fit_statistics(y, without_intercept, design, response, raw, coefficients, count, dimension):
    def operation(samples):
        rows = samples * dimension
        intercept = tf.reduce_mean(y[:samples] - without_intercept[:samples])
        residual = intercept + without_intercept[:samples] - y[:samples]
        score_residual = tf.linalg.matvec(design[:rows], coefficients) - response[:rows]
        loss = tf.reduce_mean(residual ** 2)
        score_rmse = tf.sqrt(tf.reduce_mean(score_residual ** 2))
        squares = tf.reduce_sum((tf.linalg.matvec(design[:rows], raw) - response[:rows]) ** 2)
        return intercept, loss, score_rmse, squares

    return _active_row_operation(operation, count, 1, y.shape[0])


def compact_train_metrics(y, predictions, center_value, count):
    def operation(samples):
        return (tf.sqrt(tf.reduce_mean((y[:samples] - predictions[:samples]) ** 2)),
                tf.math.reduce_std(y[:samples] - center_value))

    return _active_row_operation(operation, count, 1, y.shape[0])


def compact_holdout_rmse(y, predictions, count):
    def operation(samples):
        if samples == 0:
            return tf.constant(float('nan'), y.dtype)
        return tf.sqrt(tf.reduce_mean((y[:samples] - predictions[:samples]) ** 2))

    return _active_row_operation(operation, count, 0, y.shape[0])
