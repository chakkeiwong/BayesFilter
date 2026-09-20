"""Ordered scalar multistart localization and exact replay in TensorFlow.

The optimizer settings and scalar target authority belong to the sequential
locator. This program preserves their order; it does not batch independent
optimizers or substitute endpoint objective values for exact replay.
"""

from functools import lru_cache

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.sequential_selection_tf import selection_numerics

D = tf.float64


@lru_cache(maxsize=64)
def scalar_locator_program(scalar, count, dimension, box_radius, tolerance,
                           max_iterations, max_line_search_iterations,
                           stopping_condition, *, jit_compile=True):
    stopping = (tfp.optimizer.converged_all if stopping_condition == "converged_all"
                else tfp.optimizer.converged_any)

    def evaluate(point):
        value, score = scalar(point)
        value = tf.reshape(tf.convert_to_tensor(value, D), [])
        score = tf.reshape(tf.convert_to_tensor(score, D), [-1])
        if score.shape != (dimension,):
            raise ValueError("target score must have one entry per parameter")
        return value, score

    @tf.function(input_signature=[tf.TensorSpec([count, dimension], D),
                                  tf.TensorSpec([dimension], D)],
                 jit_compile=jit_compile, autograph=False)
    def locate(starts, scale):
        endpoints = tf.TensorArray(D, size=count, element_shape=[dimension])
        valid = tf.TensorArray(tf.bool, size=count, element_shape=[])
        diagnostics = tf.TensorArray(tf.int32, size=count, element_shape=[4])
        norms = tf.TensorArray(D, size=count, element_shape=[])
        radius = tf.constant(box_radius, D)

        def optimize(index, endpoints, valid, diagnostics, norms):
            start = starts[index]

            @tf.function(input_signature=[tf.TensorSpec([dimension], D)],
                         jit_compile=jit_compile, autograph=False)
            def objective(unconstrained):
                z = radius * tf.math.tanh(unconstrained / radius)
                value, score = evaluate(start + scale * z)
                derivative = 1.0 - tf.square(z / radius)
                return -value, -(scale * score * derivative)

            optimizer = tfp.optimizer.lbfgs_minimize(objective,
                initial_position=tf.zeros([dimension], D),
                tolerance=tf.constant(tolerance, D), max_iterations=max_iterations,
                max_line_search_iterations=max_line_search_iterations,
                parallel_iterations=1, stopping_condition=stopping)
            endpoint_z = radius * tf.math.tanh(optimizer.position / radius)
            endpoint = start + scale * endpoint_z
            value, score = evaluate(endpoint)
            finite = tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score))
            row = tf.stack([tf.cast(optimizer.converged, tf.int32),
                            tf.cast(optimizer.failed, tf.int32),
                            optimizer.num_iterations, optimizer.num_objective_evaluations])
            return (index + 1, endpoints.write(index, endpoint), valid.write(index, finite),
                    diagnostics.write(index, row), norms.write(index, tf.linalg.norm(endpoint_z)))

        if count:
            _, endpoints, valid, diagnostics, norms = tf.while_loop(
                lambda index, *_: index < count, optimize,
                (tf.constant(0), endpoints, valid, diagnostics, norms),
                maximum_iterations=count, parallel_iterations=1)
            endpoints, valid, diagnostics, norms = (
                endpoints.stack(), valid.stack(), diagnostics.stack(), norms.stack())
        else:
            endpoints, valid, diagnostics, norms = (tf.zeros([0, dimension], D),
                tf.zeros([0], tf.bool), tf.zeros([0, 4], tf.int32), tf.zeros([0], D))

        positions = tf.concat([starts, endpoints], axis=0)
        eligible = tf.concat([tf.ones([count], tf.bool), valid], axis=0)
        values = tf.TensorArray(D, size=2 * count, element_shape=[])
        scores = tf.TensorArray(D, size=2 * count, element_shape=[dimension])

        def replay(index, values, scores):
            value, score = tf.cond(eligible[index], lambda: evaluate(positions[index]),
                lambda: (tf.constant(float("nan"), D), tf.zeros([dimension], D)))
            return index + 1, values.write(index, value), scores.write(index, score)

        if count:
            _, values, scores = tf.while_loop(lambda index, *_: index < 2 * count, replay,
                (tf.constant(0), values, scores), maximum_iterations=2 * count, parallel_iterations=1)
            values, scores = values.stack(), scores.stack()
        else:
            values, scores = tf.zeros([0], D), tf.zeros([0, dimension], D)

        # Public preparation originally returned frozen host records. Stop an
        # outer tape before it builds unavailable optimizer-loop derivatives.
        return tf.nest.map_structure(tf.stop_gradient,
            {"selected": selection_numerics(positions, values, scores),
                "endpoint_finite": valid, "optimizer_diagnostics": diagnostics,
                "endpoint_standardized_norm": norms,
                "exact_evaluations": count + tf.math.count_nonzero(eligible, dtype=tf.int32),
                "objective_evaluations": tf.reduce_sum(diagnostics[:, 3])})

    return locate
