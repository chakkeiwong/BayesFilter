"""Internal XLA cloud/pilot evaluation on prepared fixed-shape directions."""

import tensorflow as tf

from bayesfilter.inference.mass_matrix_tf import eigenpair_program
from bayesfilter.inference.quadratic_geometry import _pilot_sketch_kernel
from bayesfilter.inference.quadratic_geometry_control_tf import (
    _callback_program,
    validate_callback_graph,
)

D = tf.float64


def make_geometry_cloud_program(callback, dimension, rows, *, batched=False, jit_compile=True):
    """Evaluate in original order, retaining raw scalar target records.

    Batched evaluation normalizes invalid rows to NaNs as the original did.
    Scalar normalization, when needed for design clouds, remains explicit.
    """
    if dimension < 1 or rows < 0:
        raise ValueError("invalid cloud shape")
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

    @tf.function(input_signature=[tf.TensorSpec([rows, dimension], D)], autograph=False, jit_compile=jit_compile)
    def evaluate(points):
        if batched:
            values, scores = target(points)
        elif rows:
            def row(index, values, scores):
                point = tf.gather(points, index)
                value, score = target(point)
                return (index + 1, tf.tensor_scatter_nd_update(values, [[index]], value[None]),
                    tf.tensor_scatter_nd_update(scores, [[index]], score[None]))

            _, values, scores = tf.while_loop(lambda index, *_: index < rows, row,
                (tf.constant(0), tf.zeros([rows], D), tf.zeros([rows, dimension], D)),
                maximum_iterations=rows, parallel_iterations=1)
        else:
            values, scores = tf.zeros([0], D), tf.zeros([0, dimension], D)
        valid = tf.math.is_finite(values) & tf.reduce_all(tf.math.is_finite(scores), axis=1)
        if batched:
            values = tf.where(valid, values, tf.constant(float("nan"), D))
            scores = tf.where(valid[:, None], scores, tf.constant(float("nan"), D))
        return {"values": values, "scores": scores, "valid": valid}

    return evaluate


def make_geometry_pilot_program(callback, dimension, rank, direction_count, *, batched=False, jit_compile=True):
    """Enclose ordered pilot evaluation and the unchanged curvature sketch."""
    if dimension < 1 or not 0 <= rank < dimension or direction_count < 0:
        raise ValueError("invalid pilot extents")
    if rank == 0 and direction_count:
        raise ValueError("rank-zero pilots must not prepare unused directions")
    rows = 2 * direction_count
    evaluate = make_geometry_cloud_program(callback, dimension, rows, batched=batched, jit_compile=jit_compile) if rank else None
    eigenpairs = eigenpair_program(dimension) if jit_compile and rank else tf.linalg.eigh

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D),
        tf.TensorSpec([direction_count, dimension], D), tf.TensorSpec([], D),
        tf.TensorSpec([dimension], D)], autograph=False, jit_compile=jit_compile)
    def pilot(center, scale, directions, step, center_score):
        if not rank:
            return {"q_basis": tf.zeros([dimension, 0], D), "eigenvalues": tf.zeros([dimension], D),
                "positive_count": tf.constant(0, tf.int64), "curvature_min": tf.constant(float("inf"), D),
                "curvature_max": tf.constant(0., D), "center_score_norm": tf.linalg.norm(center_score),
                "positions": tf.zeros([0, dimension], D), "values": tf.zeros([0], D),
                "scores": tf.zeros([0, dimension], D), "valid": tf.zeros([0], tf.bool)}
        plus = center[None] + step * directions * scale[None]
        minus = center[None] - step * directions * scale[None]
        points = (tf.concat((plus, minus), axis=0) if batched else
                  tf.reshape(tf.stack((plus, minus), axis=1), [rows, dimension]))
        cloud = evaluate(points)
        scores = tf.where(cloud["valid"][:, None], cloud["scores"], tf.constant(float("nan"), D))
        plus_score, minus_score = ((scores[:direction_count], scores[direction_count:]) if batched else
                                   (scores[::2], scores[1::2]))
        if direction_count:
            frame, eigenvalues, count, minimum, maximum = _pilot_sketch_kernel(
                directions, plus_score, minus_score, scale, step, eigenpairs=eigenpairs)
        else:
            frame, eigenvalues = tf.eye(dimension, dtype=D), tf.zeros([dimension], D)
            count = tf.constant(0, tf.int64)
            minimum, maximum = tf.constant(float("inf"), D), tf.constant(0., D)
        basis = tf.linalg.qr(frame[:, :rank], full_matrices=False)[0]
        return {"q_basis": basis, "eigenvalues": eigenvalues[::-1], "positive_count": count,
            "curvature_min": minimum, "curvature_max": maximum,
            "center_score_norm": tf.linalg.norm(center_score), "positions": points, **cloud}

    return pilot
