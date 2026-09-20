"""Native all-replicate stability checks for fixed-center geometry selection."""

import math
from functools import lru_cache

import tensorflow as tf

D = tf.float64
CHECK_NAMES = ("generalized_eigenvalue_spread", "trace_normalized_frobenius",
    "trace_normalized_operator", "principal_angle_degrees")
CAP_NAMES = tuple(name + "_cap" for name in CHECK_NAMES)


@lru_cache(maxsize=64)
def stability_program(comparison_kernel, dimension, count, *, jit_compile=True):
    """The caller supplies the shared repository precision comparison kernel."""
    pair_count = count * (count - 1) // 2
    width = 3 * dimension + 10

    @tf.function(input_signature=[tf.TensorSpec([count, dimension, dimension], D),
        tf.TensorSpec([count, 2], tf.bool), tf.TensorSpec([4], D),
        tf.TensorSpec([4], tf.bool), tf.TensorSpec([], tf.int32)],
        jit_compile=jit_compile, autograph=False)
    def compute(matrices, usable, caps, enabled, requested_rank):
        usable_count = tf.math.count_nonzero(tf.reduce_all(usable, axis=1), dtype=tf.int32)
        complete = (usable_count == count) & (count >= 2)
        if count < 2:
            return {"complete": complete, "usable_count": usable_count, "error": tf.constant(0),
                "reports": tf.zeros([pair_count, width], D), "checks": tf.zeros([pair_count, 4], tf.bool),
                "pair_passed": tf.zeros([pair_count], tf.bool), "passed": tf.constant(False)}

        def validate(matrix):
            return tf.reduce_all(tf.math.is_finite(matrix)) & tf.reduce_all(
                tf.abs(matrix - tf.transpose(matrix)) <= 1e-12 + 1e-10 * tf.abs(tf.transpose(matrix)))

        def pairs():
            def step(index, left, right, _error, reports, checks):
                first, second = matrices[left], matrices[right]
                valid_left, valid_right = validate(first), validate(second)
                rank_valid = (requested_rank > 0) & (requested_rank <= dimension)
                error = tf.where(~valid_left, 1, tf.where(~valid_right, 2, tf.where(~rank_valid, 3, 0)))

                def compare():
                    values_left, values_right, rank, angles, generalized, frobenius, operator = comparison_kernel(
                        .5 * (first + tf.transpose(first)), .5 * (second + tf.transpose(second)),
                        tf.constant(0., D), requested_rank, jit_compile=jit_compile)
                    spd = tf.reduce_all(values_left > 0.) & tf.reduce_all(values_right > 0.)
                    maximum_angle = tf.reduce_max(tf.where(tf.range(dimension) < rank, angles,
                        tf.constant(-math.inf, D)))
                    raw_checks = tf.stack([spd & (generalized[2] <= caps[0]),
                        frobenius <= caps[1], operator <= caps[2],
                        (rank > 0) & (maximum_angle <= caps[3])])
                    report = tf.concat([values_left, values_right, angles,
                        tf.stack([tf.cast(rank, D), *tf.unstack(generalized), frobenius, operator, maximum_angle,
                            tf.cast(tf.math.count_nonzero(values_left <= 0.), D),
                            tf.cast(tf.math.count_nonzero(values_right <= 0.), D), tf.cast(spd, D)])], 0)
                    return report, ~enabled | raw_checks

                report, pair_checks = tf.cond(error == 0, compare,
                    lambda: (tf.zeros([width], D), tf.zeros([4], tf.bool)))
                next_left = tf.where(right + 1 == count, left + 1, left)
                next_right = tf.where(right + 1 == count, left + 2, right + 1)
                return (index + 1, next_left, next_right, error,
                    tf.tensor_scatter_nd_update(reports, [[index]], [report]),
                    tf.tensor_scatter_nd_update(checks, [[index]], [pair_checks]))

            _, _, _, error, reports, checks = tf.while_loop(
                lambda index, _left, _right, error, *_: (index < pair_count) & (error == 0), step,
                (tf.constant(0), tf.constant(0), tf.constant(1), tf.constant(0),
                    tf.zeros([pair_count, width], D), tf.zeros([pair_count, 4], tf.bool)),
                maximum_iterations=pair_count, parallel_iterations=1)
            return error, reports, checks

        error, reports, checks = tf.cond(complete, pairs, lambda: (tf.constant(0),
            tf.zeros([pair_count, width], D), tf.zeros([pair_count, 4], tf.bool)))
        return {"complete": complete, "usable_count": usable_count, "error": error,
            "reports": reports, "checks": checks, "pair_passed": tf.reduce_all(checks, axis=1),
            "passed": complete & (error == 0) & tf.reduce_all(checks)}

    return compute
