"""Compiled numerical programs for the existing mass preparation contract.

Configuration and result records live in ``mass_matrix``. These programs keep
floor selection, spectral projection, inversion and block repetition in XLA.
"""

import math
from functools import lru_cache

import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_self_adjoint_eig

from bayesfilter.ops.compiled_tensor_program_tf import tensor_program

D = tf.float64


@tf.custom_gradient
def _eigh(matrix):
    """Binary64 XLA eigenpairs with TensorFlow 2.19's real lower-triangle VJP.

    The gap broadening and triangular convention reproduce SelfAdjointEigV2's
    existing derivative (tensorflow/python/ops/linalg_grad.py), including its
    convention at repeated eigenvalues. This is not a new regularization rule.
    """
    values, vectors = xla_self_adjoint_eig(
        matrix, lower=True, max_iter=100, epsilon=math.ulp(1.0))
    values = tf.ensure_shape(values, matrix.shape[:-1])
    vectors = tf.ensure_shape(vectors, matrix.shape)
    values, vectors = _refine_eigenpairs(matrix, values, vectors)

    def pullback(value_gradient, vector_gradient):
        gaps = values[..., None, :] - values[..., :, None]
        inverse_gaps = gaps * tf.math.reciprocal(gaps * gaps + 1e-20)
        inverse_gaps = tf.linalg.set_diag(inverse_gaps, tf.zeros_like(values))
        local = tf.linalg.diag(value_gradient) + inverse_gaps * tf.matmul(
            vectors, vector_gradient, transpose_a=True)
        gradient = tf.matmul(vectors, tf.matmul(local, vectors, transpose_b=True))
        gradient = tf.linalg.band_part(gradient + tf.linalg.matrix_transpose(gradient), -1, 0)
        return tf.linalg.set_diag(gradient, 0.5 * tf.linalg.diag_part(gradient))

    return (values, vectors), pullback


def _refine_eigenpairs(matrix, values, vectors):
    """Resolve residual rotations missed by the backend's stopping check.

    Jacobi similarity rotations solve the same symmetric eigenproblem. Test
    explicit off-diagonal entries, avoiding subtraction of nearly equal norms.
    This kernel is internal to the eigensystem custom VJP, so it does not change
    TensorFlow's established spectral derivative convention.
    """
    dimension = matrix.shape[0]
    if dimension <= 1:
        return values, vectors
    lower = tf.linalg.band_part(matrix, -1, 0)
    symmetric = lower + tf.transpose(lower) - tf.linalg.diag(tf.linalg.diag_part(lower))
    rotated = _symmetrize(tf.matmul(vectors, tf.matmul(symmetric, vectors), transpose_a=True))

    def off_diagonal(square):
        return tf.linalg.set_diag(square, tf.zeros([dimension], D))

    def unresolved(index, square, _frame):
        scale = tf.reduce_max(tf.abs(square))
        return tf.reduce_max(tf.abs(off_diagonal(square))) > math.ulp(1.) * scale

    def rotate(index, square, frame):
        largest = tf.argmax(tf.reshape(tf.abs(off_diagonal(square)), [-1]), output_type=tf.int32)
        row, column = largest // dimension, largest % dimension
        cross = square[row, column]
        delta = 0.5 * square[column, column] - 0.5 * square[row, row]
        scale = tf.maximum(tf.abs(delta), tf.abs(cross))
        scaled_delta, scaled_cross = delta / scale, cross / scale
        sign = tf.where(delta >= 0., tf.constant(1., D), tf.constant(-1., D))
        tangent = sign * scaled_cross / (tf.abs(scaled_delta) + tf.sqrt(scaled_delta ** 2 + scaled_cross ** 2))
        cosine = tf.math.rsqrt(1. + tangent ** 2)
        sine = tangent * cosine
        row_basis, column_basis = tf.one_hot(row, dimension, dtype=D), tf.one_hot(column, dimension, dtype=D)

        def rotate_columns(value):
            first, second = value[:, row], value[:, column]
            return (value + (cosine * first - sine * second - first)[:, None] * row_basis[None, :]
                + (sine * first + cosine * second - second)[:, None] * column_basis[None, :])

        updated = tf.transpose(rotate_columns(tf.transpose(rotate_columns(square))))
        # Exact Jacobi off-diagonal zeros avoid recycling cancellation noise.
        updated = tf.tensor_scatter_nd_update(updated, tf.stack([[row, column], [column, row]]), tf.zeros([2], D))
        return index + 1, _symmetrize(updated), rotate_columns(frame)

    _, rotated, vectors = tf.while_loop(unresolved, rotate, (tf.constant(0), rotated, vectors),
        maximum_iterations=100 * dimension * dimension, parallel_iterations=1)
    values = tf.linalg.diag_part(rotated)
    order = tf.argsort(values, stable=True)
    return tf.gather(values, order), tf.gather(vectors, order, axis=1)


def _symmetrize(matrix):
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def _eigenpairs(matrix, jit_compile):
    return _eigh(matrix) if jit_compile else tf.linalg.eigh(matrix)


@lru_cache(maxsize=64)
def eigenpair_program(dimension):
    """Trace the spectral custom derivative without retaining a caller graph."""
    # The custom-gradient registry retains its traced tensor closure. Creating
    # that closure in a resource-owning caller also retains its factor guards.
    # Keep this graph independent, as for the shared fresh-position program.
    with tf.init_scope():
        @tf.function(input_signature=[tf.TensorSpec([dimension, dimension], D)],
                     jit_compile=True, autograph=False)
        def eigenpairs(matrix):
            return _eigh(matrix)

        eigenpairs.get_concrete_function()
    return eigenpairs


@tf.function(input_signature=[tf.TensorSpec([None, None], D)], jit_compile=True, autograph=False)
def finite_matrix(matrix):
    return tf.reduce_all(tf.math.is_finite(matrix))


def precision_numerics(matrix, jitter, requested_floor, condition_cap, *, jit_compile=True):
    """Original regularization, with the original frozen effective floor."""
    finite = tf.reduce_all(tf.math.is_finite(matrix))
    safe_matrix = tf.where(finite, matrix, tf.zeros_like(matrix))
    asymmetry = tf.reduce_max(tf.abs(safe_matrix - tf.linalg.matrix_transpose(safe_matrix)))
    symmetric = _symmetrize(safe_matrix)
    jittered = symmetric + jitter * tf.eye(tf.shape(matrix)[0], dtype=D)
    values, vectors = _eigenpairs(jittered, jit_compile)
    finite_values = tf.reduce_all(tf.math.is_finite(values))
    raw_max = tf.reduce_max(values)
    positive = tf.reduce_any(values > 0.)
    implicit_floor = math.ulp(1.) * tf.maximum(tf.constant(1., D), raw_max)
    floor = tf.where(requested_floor == 0., implicit_floor, requested_floor)
    floor = tf.where(condition_cap > 0., tf.maximum(floor, raw_max / condition_cap), floor)
    floor = tf.where(floor <= 0., tf.constant(math.ulp(1.), D), floor)
    # Before migration this scalar crossed .numpy() and became a constant.
    floor = tf.stop_gradient(floor)
    projected_values = tf.maximum(values, floor)
    regularized = _symmetrize(tf.matmul(vectors * projected_values[None, :], vectors, transpose_b=True))
    diagnostics = tf.stop_gradient(tf.stack([floor, tf.reduce_min(values), raw_max,
        tf.reduce_min(projected_values), tf.reduce_max(projected_values),
        tf.cast(tf.math.count_nonzero(values <= 0.), D),
        tf.cast(tf.math.count_nonzero(projected_values > values), D), asymmetry]))
    flags = tf.stack([finite, finite_values, (requested_floor != 0.) | positive, asymmetry > 0.])
    return regularized, diagnostics, flags


@lru_cache(maxsize=64)
def precision_program(dimension, *, dense=None, jit_compile=True):
    """Enclose regularization and optional dense/diagonal covariance inversion."""
    @tf.function(input_signature=[tf.TensorSpec([dimension, dimension], D),
        tf.TensorSpec([], D), tf.TensorSpec([], D), tf.TensorSpec([], D)],
        jit_compile=jit_compile, autograph=False)
    def compute(matrix, jitter, floor, condition_cap):
        regularized, diagnostics, flags = precision_numerics(matrix, jitter, floor, condition_cap,
            jit_compile=jit_compile)
        if dense is None:
            return regularized, diagnostics, flags
        diagonal = tf.linalg.diag_part(regularized)
        diagonal_valid = tf.reduce_all(tf.math.is_finite(diagonal) & (diagonal > 0.))
        covariance = (_symmetrize(tf.linalg.inv(regularized)) if dense else
            tf.linalg.diag(tf.math.reciprocal(diagonal)))
        return regularized, covariance, diagnostics, flags, diagonal_valid

    return compute


@lru_cache(maxsize=64)
def covariance_program(dimension, *, whitening=False, jit_compile=True):
    @tf.function(input_signature=[tf.TensorSpec([dimension, dimension], D), tf.TensorSpec([], tf.float32)],
        jit_compile=jit_compile, autograph=False)
    def compute(matrix, jitter):
        # Preserve the prior Python-float -> tf.cast binary32 jitter boundary.
        regularized = matrix + tf.cast(jitter, D) * tf.eye(dimension, dtype=D)
        return tf.linalg.cholesky(regularized) if whitening else regularized

    return compute


@lru_cache(maxsize=64)
def summary_program(dimension, *, jit_compile=True):
    @tf.function(input_signature=[tf.TensorSpec([dimension, dimension], D)],
        jit_compile=jit_compile, autograph=False)
    def compute(matrix):
        values, _ = _eigenpairs(_symmetrize(matrix), jit_compile)
        finite = tf.reduce_all(tf.math.is_finite(values))
        minimum = tf.where(finite, tf.reduce_min(values), tf.constant(math.nan, D))
        maximum = tf.where(finite, tf.reduce_max(values), tf.constant(math.nan, D))
        positive = finite & (minimum > 0.)
        condition = tf.where(positive, maximum / minimum, tf.constant(math.inf, D))
        return tf.stop_gradient(values), tf.stop_gradient(tf.stack([minimum, maximum, condition])), finite, positive

    return compute


@lru_cache(maxsize=64)
def structured_program(dimension, partition, *, jit_compile=True):
    """One native block loop; only distinct widths require static branches."""
    widths = tuple(sorted({stop - start for start, stop in partition}))
    branch_ids = tuple(widths.index(stop - start) for start, stop in partition)
    starts = tuple(start for start, _ in partition)
    block_count = len(partition)

    def compute(matrix, weight, floor, condition_cap):
        finite = tf.reduce_all(tf.math.is_finite(matrix))
        matrix = tf.where(finite, matrix, tf.zeros_like(matrix))

        def step(index, projected, summaries, valid):
            start = tf.gather(tf.constant(starts), index)

            def branch(width):
                def update():
                    block = tf.slice(matrix, [start, start], [width, width])
                    raw = _symmetrize(block)
                    target = tf.linalg.diag(tf.linalg.diag_part(raw))
                    shrunk = _symmetrize((1. - weight) * raw + weight * target)
                    values, vectors = _eigenpairs(shrunk, jit_compile)
                    block_valid = tf.reduce_all(tf.math.is_finite(values))
                    effective = tf.stop_gradient(tf.maximum(floor,
                        tf.maximum(tf.reduce_max(values), floor) / condition_cap))
                    clipped = tf.maximum(values, effective)
                    regularized = _symmetrize(tf.matmul(vectors * clipped[None, :], vectors, transpose_b=True))
                    rows, columns = tf.meshgrid(tf.range(width) + start, tf.range(width) + start, indexing="ij")
                    coordinates = tf.stack([tf.reshape(rows, [-1]), tf.reshape(columns, [-1])], -1)
                    updated = tf.tensor_scatter_nd_update(projected, coordinates, tf.reshape(regularized, [-1]))
                    summary = tf.stack([tf.reduce_min(values), tf.reduce_min(clipped), tf.reduce_max(clipped)])
                    return updated, tf.stop_gradient(summary), block_valid
                return update

            updated, summary, block_valid = tf.switch_case(tf.gather(tf.constant(branch_ids), index),
                tuple(branch(width) for width in widths))
            return index + 1, updated, summaries.write(index, summary), valid & block_valid

        _, projected, summaries, valid = tf.while_loop(lambda index, *_: index < block_count, step,
            (tf.constant(0), tf.zeros([dimension, dimension], D),
             tf.TensorArray(D, size=block_count, element_shape=[3]), finite),
            maximum_iterations=block_count, parallel_iterations=1)
        summaries = summaries.stack()
        summary = tf.stack([tf.reduce_min(summaries[:, 0]),
            tf.reduce_min(summaries[:, 1]), tf.reduce_max(summaries[:, 2])])
        return projected, tf.stop_gradient(summary), tf.cast(valid, D)

    return tensor_program(compute, [tf.TensorSpec([dimension, dimension], D),
        tf.TensorSpec([], D), tf.TensorSpec([], D), tf.TensorSpec([], D)], jit_compile)
