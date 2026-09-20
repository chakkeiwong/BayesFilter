"""Native XLA kernels for the sequential locator's existing preparation rules.

Cloud mapping here is independent preparation, not a batch-native NeuTra
training target. The enclosing locator's host search lifecycle is separate.
"""

import math
from functools import lru_cache

import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_self_adjoint_eig

from bayesfilter.ops.stateless_random_tf import philox_normal_float64


@lru_cache(maxsize=64)
def evaluation_program(scalar_function, batched_function, row_count, dimension):
    """Compile one complete cloud with a bounded, fixed input signature."""

    @tf.function(input_signature=[tf.TensorSpec([row_count, dimension], tf.float64)],
                 jit_compile=True, autograph=False)
    def evaluate(rows):
        if batched_function is not None:
            values, scores = batched_function(rows)
            return (tf.ensure_shape(tf.convert_to_tensor(values, tf.float64), [row_count]),
                tf.ensure_shape(tf.convert_to_tensor(scores, tf.float64), [row_count, dimension]))
        values = tf.TensorArray(tf.float64, size=row_count, element_shape=[])
        scores = tf.TensorArray(tf.float64, size=row_count, element_shape=[dimension])

        def step(index, values, scores):
            value, score = scalar_function(rows[index])
            value = tf.reshape(tf.convert_to_tensor(value, tf.float64), [])
            score = tf.reshape(tf.convert_to_tensor(score, tf.float64), [-1])
            if score.shape != (dimension,):
                raise ValueError("target score must have one entry per parameter")
            return index + 1, values.write(index, value), scores.write(index, score)

        _, values, scores = tf.while_loop(lambda index, *_: index < row_count, step,
            (tf.constant(0), values, scores), maximum_iterations=row_count, parallel_iterations=1)
        return values.stack(), scores.stack()

    return evaluate


@lru_cache(maxsize=64)
def cloud_program(sample_count, dimension, orthogonal):
    """Keep the original per-frame seeds, pair layout and Philox stream."""
    pair_count = (sample_count + 1) // 2
    frame_count = (pair_count + dimension - 1) // dimension

    @tf.function(input_signature=[tf.TensorSpec([], tf.float64), tf.TensorSpec([2], tf.int32)],
                 jit_compile=True, autograph=False)
    def generate(radius, seed):
        if orthogonal:
            frames = tf.TensorArray(tf.float64, size=frame_count, element_shape=[dimension, dimension])

            def frame_step(index, frames):
                frame_seed = tf.stack([seed[0], seed[1] + 7919 * index])
                normal = philox_normal_float64([dimension, dimension], frame_seed)
                orthogonal_frame, diagonal = tf.linalg.qr(normal)
                signs = tf.where(tf.linalg.diag_part(diagonal) >= 0.0,
                    tf.ones([dimension], tf.float64), -tf.ones([dimension], tf.float64))
                return index + 1, frames.write(index, tf.transpose(orthogonal_frame * signs[None, :]))

            _, frames = tf.while_loop(lambda index, _: index < frame_count, frame_step,
                (tf.constant(0), frames), maximum_iterations=frame_count, parallel_iterations=1)
            directions = tf.reshape(frames.stack(), [-1, dimension])[:pair_count]
        else:
            directions = philox_normal_float64([pair_count, dimension], seed)
            directions /= tf.maximum(tf.linalg.norm(directions, axis=1, keepdims=True), 1.0e-15)
        radii = tf.linspace(radius / float(pair_count), radius, pair_count)[:, None]
        if orthogonal:
            cloud = tf.reshape(tf.stack((directions * radii, -directions * radii), axis=1), [-1, dimension])
        else:
            cloud = tf.concat([directions * radii, -directions * radii], axis=0)
        return cloud[:sample_count]

    return generate


@lru_cache(maxsize=64)
def trust_region_program(dimension, *, jit_compile=True):
    """Original SPD bracket/bisection with the unchanged 80/80 iteration caps."""

    @tf.function(input_signature=[tf.TensorSpec([dimension, dimension], tf.float64),
        tf.TensorSpec([dimension], tf.float64), tf.TensorSpec([], tf.float64)],
        jit_compile=jit_compile, autograph=False)
    def solve(precision, linear, radius):
        # tf.linalg.eigh's default XLA stopping tolerance is too loose for the
        # existing binary64 gate. Use the project's explicit Jacobi precision.
        if jit_compile:
            eigenvalues, eigenvectors = xla_self_adjoint_eig(
                precision, lower=True, max_iter=100, epsilon=math.ulp(1.0))
        else:
            # Explicit graph-reference exception; the default is GPU/XLA.
            eigenvalues, eigenvectors = tf.linalg.eigh(precision)
        eigenvalues = tf.ensure_shape(eigenvalues, [dimension])
        eigenvectors = tf.ensure_shape(eigenvectors, [dimension, dimension])
        projected = tf.linalg.matvec(eigenvectors, linear, transpose_a=True)

        def solution(multiplier):
            return tf.linalg.matvec(eigenvectors, projected / (eigenvalues + multiplier))

        unconstrained = tf.linalg.matvec(eigenvectors, projected / eigenvalues)
        boundary = tf.linalg.norm(unconstrained) > radius

        def constrained():
            _, upper = tf.while_loop(
                lambda index, upper: tf.linalg.norm(solution(upper)) > radius,
                lambda index, upper: (index + 1, upper * 2.0),
                (tf.constant(0), tf.constant(1.0, tf.float64)),
                maximum_iterations=80, parallel_iterations=1)

            def bisect(index, lower, upper):
                middle = 0.5 * (lower + upper)
                outside = tf.linalg.norm(solution(middle)) > radius
                return index + 1, tf.where(outside, middle, lower), tf.where(outside, upper, middle)

            _, _, upper = tf.while_loop(lambda index, *_: index < 80, bisect,
                (tf.constant(0), tf.constant(0.0, tf.float64), upper),
                maximum_iterations=80, parallel_iterations=1)
            return solution(upper)

        step = tf.cond(boundary, constrained, lambda: unconstrained)
        predicted = tf.tensordot(linear, step, 1) - 0.5 * tf.tensordot(
            step, tf.linalg.matvec(precision, step), 1)
        return step, boundary, predicted

    return solve
