"""Complete XLA score-fit preparation for the existing sequential locator."""

import math
from functools import lru_cache

import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_svd

from bayesfilter.inference._exact_incumbent import _incumbent_selection
from bayesfilter.inference.mass_matrix_tf import eigenpair_program
from bayesfilter.inference.program_cache_scope import (
    independent_trace_scope,
    scoped_program_cache,
)
from bayesfilter.inference.sequential_preparation_tf import (
    cloud_program,
    evaluation_program,
)
from bayesfilter.ops.qr_lstsq_tf import complete_orthogonal_lstsq
from bayesfilter.ops.symmetric_matrix_tf import symmetric_score_design, unpack_symmetric


@lru_cache(maxsize=64)
def _score_lstsq_program(rows, columns, jit_compile):
    # The custom-gradient registry keeps its tensors alive. This shape-only
    # primitive must therefore have no ancestry into a callback-owning graph.
    with independent_trace_scope():
        solve = tf.function(complete_orthogonal_lstsq,
            input_signature=[tf.TensorSpec([rows, columns], tf.float64),
                tf.TensorSpec([rows, 1], tf.float64)],
            jit_compile=jit_compile, autograph=False)
        solve.get_concrete_function()
    return solve


def partition_schema(sample_count, holdout_fraction, *, pair_disjoint):
    """Resolve immutable row indices from configuration before tracing."""
    count, fraction = int(sample_count), float(holdout_fraction)
    if count <= 1 or not 0.0 < fraction < 1.0:
        raise ValueError("score-fit partition requires count > 1 and 0 < fraction < 1")
    if not pair_disjoint:
        train_count = count - max(1, round(count * fraction))
        return tuple(range(train_count)), tuple(range(train_count, count))
    if count % 2:
        raise ValueError("pair-disjoint score holdout requires an even sample count")
    pairs = count // 2
    holdout_pairs = max(1, round(pairs * fraction))
    if holdout_pairs >= pairs:
        raise ValueError("pair-disjoint score holdout requires at least one training pair")
    train_pairs = pairs - holdout_pairs
    return (tuple(range(train_pairs)) + tuple(range(pairs, pairs + train_pairs)),
        tuple(range(train_pairs, pairs)) + tuple(range(pairs + train_pairs, count)))


def fit_numerics(z, scores, center_score, scale, ridge, floor, condition_cap,
                 holdout_tolerance, *, training_indices, holdout_indices, jit_compile=True):
    """Original symmetric COD fit and diagnostics, with fixed-shape rejection."""
    dimension = z.shape[1]
    coefficients = dimension * (dimension + 1) // 2
    response = scale[None, :] * center_score[None, :] - scale[None, :] * scores
    design = symmetric_score_design(z, dimension)
    train_design_rows = tf.gather(design, training_indices)
    train_response_rows = tf.gather(response, training_indices)
    holdout_design_rows = tf.gather(design, holdout_indices)
    holdout_response_rows = tf.gather(response, holdout_indices)
    train_design = tf.reshape(train_design_rows, [-1, coefficients])
    train_response = tf.reshape(train_response_rows, [-1, 1])
    # XlaSvd computes full factors. The triangular factor has the same singular
    # values and avoids a square factor in the much larger observation space.
    reduced = tf.linalg.qr(train_design, full_matrices=False)[1]
    singular = (xla_svd(reduced, max_iter=100, epsilon=math.ulp(1.0), precision_config="").s
        if jit_compile else tf.linalg.svd(reduced, compute_uv=False))
    tolerance = tf.reduce_max(singular) * tf.cast(tf.shape(train_design)[0], tf.float64) * math.ulp(1.0)
    rank = tf.reduce_sum(tf.cast(singular > tolerance, tf.int32))

    def rejected():
        return (tf.constant(0), tf.constant(0., tf.float64), tf.constant(0., tf.float64),
            tf.zeros([dimension], tf.float64), tf.zeros([dimension], tf.float64),
            tf.constant(0., tf.float64), tf.zeros([dimension, dimension], tf.float64))

    def solve():
        regularizer = tf.sqrt(ridge) * tf.eye(coefficients, dtype=tf.float64)
        least_squares = _score_lstsq_program(int(train_design.shape[0]) + coefficients,
            coefficients, jit_compile)
        beta = least_squares(tf.concat([train_design, regularizer], 0),
            tf.concat([train_response, tf.zeros([coefficients, 1], tf.float64)], 0))[:, 0]
        precision = unpack_symmetric(beta, dimension)
        train_prediction = tf.einsum("nrc,c->nr", train_design_rows, beta)
        holdout_prediction = tf.einsum("nrc,c->nr", holdout_design_rows, beta)
        train_rmse = tf.sqrt(tf.reduce_mean((train_prediction - train_response_rows) ** 2))
        holdout_error = tf.sqrt(tf.reduce_mean((holdout_prediction - holdout_response_rows) ** 2))
        holdout_scale = tf.maximum(tf.sqrt(tf.reduce_mean(holdout_response_rows ** 2)), 1e-15)
        relative = holdout_error / holdout_scale
        eigenvalues, vectors = eigenpair_program(dimension)(precision) if jit_compile else tf.linalg.eigh(precision)
        eigenvalues = tf.ensure_shape(eigenvalues, [dimension])
        vectors = tf.ensure_shape(vectors, [dimension, dimension])
        effective_floor = tf.maximum(floor, tf.reduce_max(eigenvalues) / condition_cap)
        projected_values = tf.maximum(eigenvalues, effective_floor)
        projected = tf.matmul(vectors * projected_values[None, :], vectors, transpose_b=True)
        projection = tf.linalg.norm(projected - precision) / tf.maximum(tf.linalg.norm(precision), 1e-15)
        return (tf.where(relative <= holdout_tolerance, 1, 2), train_rmse, relative,
            eigenvalues, projected_values, projection, projected)

    status, train_rmse, relative, eigenvalues, projected_values, projection, projected = tf.cond(
        rank < coefficients, rejected, solve)
    return {"status": status, "rank": rank, "train_score_rmse": train_rmse,
        "holdout_score_relative_rmse": relative, "raw_eigenvalues": eigenvalues,
        "projected_eigenvalues": projected_values, "projection_relative_frobenius": projection,
        "projected_precision_z": projected}


@scoped_program_cache(maxsize=64)
def score_fit_program(scalar, batched, sample_count, dimension, training_indices, holdout_indices,
                      *, jit_compile=True):
    """Enclose seeded cloud, full target evaluation, fit and exact selection."""
    generate = cloud_program(sample_count, dimension, False).python_function
    evaluate = evaluation_program(scalar, batched, sample_count, dimension).python_function

    @tf.function(input_signature=[tf.TensorSpec([dimension], tf.float64),
        tf.TensorSpec([dimension], tf.float64), tf.TensorSpec([dimension], tf.float64),
        tf.TensorSpec([], tf.float64), tf.TensorSpec([2], tf.int32),
        tf.TensorSpec([], tf.float64), tf.TensorSpec([], tf.float64),
        tf.TensorSpec([], tf.float64), tf.TensorSpec([], tf.float64)],
        jit_compile=jit_compile, autograph=False)
    def fit(center, center_score, scale, radius, seed, ridge, floor, condition_cap, holdout_tolerance):
        z = generate(radius, seed)
        positions = center[None, :] + z * scale[None, :]
        values, scores = evaluate(positions)
        result = fit_numerics(z, scores, center_score, scale, ridge, floor, condition_cap,
            holdout_tolerance, training_indices=training_indices, holdout_indices=holdout_indices,
            jit_compile=jit_compile)
        _, winner = _incumbent_selection.python_function(tf.reshape(positions, [-1]),
            tf.reshape(scores, [-1]), values, tf.ones([sample_count], tf.bool),
            tf.range(1, sample_count + 1) * dimension)
        safe_winner = tf.maximum(winner, 0)
        # The original result crossed NumPy/.numpy() boundaries. Preserve that
        # frozen preparation contract inside the function, before an external
        # tape can request unavailable eigensolver derivatives.
        return tf.nest.map_structure(tf.stop_gradient,
            {**result, "best_index": winner, "best_value": tf.gather(values, safe_winner),
             "best_position": tf.gather(positions, safe_winner), "best_score": tf.gather(scores, safe_winner)})

    return fit
