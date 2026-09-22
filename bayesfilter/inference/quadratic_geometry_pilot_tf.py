"""Internal XLA cloud/pilot evaluation on prepared fixed-shape directions."""

import math

import tensorflow as tf

from bayesfilter.inference.mass_matrix_tf import eigenpair_program
from bayesfilter.inference.quadratic_geometry import _pilot_sketch_kernel
from bayesfilter.inference.quadratic_geometry_control_tf import (
    _callback_program,
    validate_callback_graph,
)
from bayesfilter.ops.qr_lstsq_tf import _active_row_operation

D = tf.float64


def _basis_resolution(eigenvalues, positive_count, rank, direction_count):
    """Relative eigenvector sensitivity guard; no regularization or error bound."""
    limit = tf.constant(math.sqrt(math.ulp(1.)), D)
    if not rank:
        return tf.constant(True), tf.constant(0., D), limit, tf.constant(float('inf'), D)
    dimension = eigenvalues.shape[0]
    gaps = eigenvalues[1:] - eigenvalues[:-1]
    minimum_gap = tf.reduce_min(gaps[dimension - rank - 1:])
    scale = tf.reduce_max(tf.abs(eigenvalues))
    relative = tf.constant(math.ulp(1.), D) * tf.cast(tf.maximum(direction_count, dimension), D)
    indicator = tf.where(minimum_gap > 0., relative * scale / tf.where(minimum_gap > 0., minimum_gap, 1.),
        tf.constant(float('inf'), D))
    # No positive curvature takes the explicit identity branch, not an
    # eigendecomposition of the zero matrix, so its orientation is resolved.
    indicator = tf.where(positive_count == 0, tf.constant(0., D), indicator)
    resolved = tf.math.is_finite(indicator) & (indicator <= limit)
    return resolved, indicator, limit, minimum_gap


def make_geometry_cloud_program(callback, dimension, rows, *, batched=False, jit_compile=True,
                                active_rows=False, row_multiple=1):
    """Evaluate in original order, retaining raw scalar target records.

    Batched evaluation normalizes invalid rows to NaNs as the original did.
    Scalar normalization, when needed for design clouds, remains explicit.
    """
    if dimension < 1 or rows < 0 or row_multiple < 1 or rows % row_multiple:
        raise ValueError("invalid cloud shape")
    if active_rows and batched:
        # Each branch binds only the callback's original compact input shape.
        # Shared pilot algebra belongs outside this exact-extent dispatcher.
        @tf.function(input_signature=[tf.TensorSpec([rows, dimension], D), tf.TensorSpec([], tf.int32)],
                     autograph=False, jit_compile=jit_compile)
        def evaluate_active(points, count):
            valid_count = (count >= 0) & (count <= rows) & (count % row_multiple == 0)

            def operation(units):
                extent = units * row_multiple
                call = make_geometry_cloud_program(callback, dimension, extent,
                    batched=True, jit_compile=jit_compile)
                cloud = call(points[:extent])
                return {'values': tf.pad(cloud['values'], [[0, rows - extent]]),
                    'scores': tf.pad(cloud['scores'], [[0, rows - extent], [0, 0]]),
                    'valid': tf.pad(cloud['valid'], [[0, rows - extent]])}

            empty = {'values': tf.zeros([rows], D), 'scores': tf.zeros([rows, dimension], D),
                'valid': tf.zeros([rows], tf.bool)}
            cloud = tf.cond(valid_count,
                lambda: _active_row_operation(operation, count // row_multiple, 0, rows // row_multiple),
                lambda: empty)
            return {**cloud, 'count_valid': valid_count}

        return evaluate_active
    if batched:
        def call(points):
            try:
                values, scores = callback(points)
                values = tf.ensure_shape(tf.convert_to_tensor(values, D), [rows])
                scores = tf.ensure_shape(tf.convert_to_tensor(scores, D), [rows, dimension])
            except Exception:  # noqa: BLE001 - original construction/conversion failure boundary only.
                values = tf.fill([rows], tf.constant(float("nan"), D))
                scores = tf.fill([rows, dimension], tf.constant(float("nan"), D))
            return values, scores

        with tf.init_scope():
            target = tf.function(call, input_signature=[tf.TensorSpec([rows, dimension], D)],
                autograph=False, jit_compile=jit_compile)
            validate_callback_graph(target, jit_compile=jit_compile)
    else:
        target = _callback_program(callback, dimension, jit_compile)

    @tf.function(input_signature=[tf.TensorSpec([rows, dimension], D)] +
        ([tf.TensorSpec([], tf.int32)] if active_rows else []), autograph=False, jit_compile=jit_compile)
    def evaluate(points, *counts):
        count = counts[0] if active_rows else rows
        valid_count = (count >= 0) & (count <= rows) & (count % row_multiple == 0)
        if batched:
            values, scores = target(points)
        elif rows:
            def row(index, values, scores):
                point = tf.gather(points, index)
                value, score = target(point)
                return (index + 1, tf.tensor_scatter_nd_update(values, [[index]], value[None]),
                    tf.tensor_scatter_nd_update(scores, [[index]], score[None]))

            _, values, scores = tf.while_loop(lambda index, *_: valid_count & (index < count), row,
                (tf.constant(0), tf.zeros([rows], D), tf.zeros([rows, dimension], D)),
                maximum_iterations=rows, parallel_iterations=1)
        else:
            values, scores = tf.zeros([0], D), tf.zeros([0, dimension], D)
        valid = tf.math.is_finite(values) & tf.reduce_all(tf.math.is_finite(scores), axis=1)
        if active_rows:
            valid &= valid_count & (tf.range(rows) < count)
        if batched:
            values = tf.where(valid, values, tf.constant(float("nan"), D))
            scores = tf.where(valid[:, None], scores, tf.constant(float("nan"), D))
        cloud = {"values": values, "scores": scores, "valid": valid}
        return {**cloud, 'count_valid': valid_count} if active_rows else cloud

    return evaluate


def make_geometry_pilot_program(callback, dimension, rank, direction_count, *, batched=False, jit_compile=True, active_rows=False):
    """Enclose ordered pilot evaluation and the unchanged curvature sketch."""
    if dimension < 1 or not 0 <= rank < dimension or direction_count < 0:
        raise ValueError("invalid pilot extents")
    if rank == 0 and direction_count:
        raise ValueError("rank-zero pilots must not prepare unused directions")
    rows = 2 * direction_count
    evaluate = make_geometry_cloud_program(callback, dimension, rows, batched=batched,
        jit_compile=jit_compile, active_rows=active_rows, row_multiple=2) if rank else None
    eigenpairs = eigenpair_program(dimension) if jit_compile and rank else tf.linalg.eigh

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([direction_count, dimension], D), tf.TensorSpec([], D),
        tf.TensorSpec([dimension], D)] + ([tf.TensorSpec([], tf.int32)] if active_rows else []),
        autograph=False, jit_compile=jit_compile)
    def pilot(center, scale, directions, step, center_score, *counts):
        count = counts[0] if active_rows else direction_count
        valid_count = (count >= 0) & (count <= direction_count)
        if not rank:
            result = {"q_basis": tf.zeros([dimension, 0], D), "eigenvalues": tf.zeros([dimension], D),
                "positive_count": tf.constant(0, tf.int64), "curvature_min": tf.constant(float("inf"), D),
                "curvature_max": tf.constant(0., D), "center_score_norm": tf.linalg.norm(center_score),
                "positions": tf.zeros([0, dimension], D), "values": tf.zeros([0], D),
                "scores": tf.zeros([0, dimension], D), "valid": tf.zeros([0], tf.bool),
                'basis_resolved': valid_count, 'basis_roundoff_indicator': tf.constant(0., D),
                'basis_roundoff_limit': tf.constant(math.sqrt(math.ulp(1.)), D),
                'basis_minimum_eigen_gap': tf.constant(float('inf'), D)}
            return {**result, 'direction_count': count, 'count_valid': valid_count} if active_rows else result
        if active_rows:
            directions = tf.where((tf.range(direction_count) < count)[:, None], directions, tf.zeros_like(directions))
        plus = center[None] + step * directions * scale[None]
        minus = center[None] - step * directions * scale[None]
        points = (tf.concat((plus, minus), axis=0) if batched else
                  tf.reshape(tf.stack((plus, minus), axis=1), [rows, dimension]))
        if active_rows:
            if batched and direction_count:
                slots = tf.range(rows)
                source = tf.where(slots < count, slots, direction_count + slots - count)
                points = tf.gather(points, tf.clip_by_value(source, 0, rows - 1))
            points = tf.where((tf.range(rows) < 2 * count)[:, None], points, tf.zeros_like(points))
        cloud = (evaluate(points, tf.where(valid_count, 2 * count, -1)) if active_rows
                 else evaluate(points))
        scores = tf.where(cloud["valid"][:, None], cloud["scores"], tf.constant(float("nan"), D))
        if active_rows and batched and direction_count:
            plus_score = scores[:direction_count]
            minus_score = tf.gather(scores, tf.clip_by_value(count + tf.range(direction_count), 0, rows - 1))
        else:
            plus_score, minus_score = ((scores[:direction_count], scores[direction_count:]) if batched else
                                       (scores[::2], scores[1::2]))
        if active_rows:
            active = valid_count & (tf.range(direction_count) < count)
            plus_score = tf.where(active[:, None], plus_score, tf.zeros_like(plus_score))
            minus_score = tf.where(active[:, None], minus_score, tf.zeros_like(minus_score))
        if direction_count:
            frame, eigenvalues, count, minimum, maximum = _pilot_sketch_kernel(
                directions, plus_score, minus_score, scale, step, eigenpairs=eigenpairs)
        else:
            frame, eigenvalues = tf.eye(dimension, dtype=D), tf.zeros([dimension], D)
            count = tf.constant(0, tf.int64)
            minimum, maximum = tf.constant(float("inf"), D), tf.constant(0., D)
        basis = tf.linalg.qr(frame[:, :rank], full_matrices=False)[0]
        resolved, indicator, limit, gap = _basis_resolution(eigenvalues, count, rank,
            counts[0] if active_rows else direction_count)
        resolved &= valid_count
        result = {"q_basis": tf.where(resolved, basis, tf.zeros_like(basis)),
            'basis_resolved': resolved, 'basis_roundoff_indicator': indicator,
            'basis_roundoff_limit': limit, 'basis_minimum_eigen_gap': gap,
            "eigenvalues": eigenvalues[::-1], "positive_count": count,
            "curvature_min": minimum, "curvature_max": maximum,
            "center_score_norm": tf.linalg.norm(center_score), "positions": points, **cloud}
        return {**result, 'direction_count': counts[0]} if active_rows else result

    return pilot
