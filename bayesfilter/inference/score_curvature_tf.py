"""TensorFlow kernels for dense fixed-center score-curvature fits.

For a fixed center and standardized offset ``z``, the local score model is
``g_z(c) - g_z(c + z) = K z``.  This module computes the unrestricted dense
least-squares coefficient ``K`` without choosing a positive-definite fallback.
The numerical kernel is intentionally independent of HMC and NumPy so it can
be used by accepted TensorFlow inference paths.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_svd

from bayesfilter.inference.mass_matrix_tf import eigenpair_program
from bayesfilter.ops.compiled_tensor_program_tf import in_xla_context
from bayesfilter.ops.qr_lstsq_tf import complete_orthogonal_lstsq


@tf.custom_gradient
def _singular_values_xla(matrix):
    """Binary64 singular values with TensorFlow's values-only SVD pullback.

    The backend's default tolerance can leave a 5e-10 condition error even
    for a well-conditioned D5 design. Match the binary64 tolerance used by
    the repository condition-number kernel. For A=U diag(s) V', the pullback
    is U diag(ds) V', as in TF 2.19 linalg_grad.py::_SvdGrad(compute_uv=False).
    """
    decomposition = xla_svd(matrix, max_iter=100, epsilon=math.ulp(1.), precision_config="")
    values = tf.ensure_shape(decomposition.s, matrix.shape[:-1])
    left = tf.ensure_shape(decomposition.u, matrix.shape)
    right = tf.ensure_shape(decomposition.v, matrix.shape)

    def pullback(upstream):
        return tf.matmul(left * upstream[None, :], right, transpose_b=True)

    return values, pullback


@lru_cache(maxsize=64)
def _singular_value_program(dimension):
    # Keep the custom-gradient closure independent of resource-owning callers.
    with tf.init_scope():
        program = tf.function(_singular_values_xla, autograph=False, jit_compile=True,
            input_signature=[tf.TensorSpec([dimension, dimension], tf.float64)])
        program.get_concrete_function()
    return program


def _as_float64(value: Any, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value)
    if tensor.dtype != tf.float64:
        raise TypeError(f"{name} must have dtype float64")
    return tensor


def _require_matrix(value: Any, name: str, dimension: int | None = None) -> tf.Tensor:
    tensor = _as_float64(value, name)
    if tensor.shape.rank != 2:
        raise ValueError(f"{name} must have rank two")
    if tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be nonempty")
    if dimension is not None and tensor.shape[1] != dimension:
        raise ValueError(f"{name} must have shape [rows,{dimension}]")
    tf.debugging.assert_all_finite(tensor, f"{name} must be finite")
    return tensor


def _symmetric(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.transpose(matrix))


def _relative_response_rmse(
    precision: tf.Tensor,
    center_score: tf.Tensor,
    offsets: tf.Tensor,
    scores: tf.Tensor,
) -> tf.Tensor:
    response = center_score[tf.newaxis, :] - scores
    prediction = tf.matmul(offsets, precision, transpose_b=True)
    scale = tf.maximum(
        tf.reduce_max(tf.abs(response)), tf.reduce_max(tf.abs(prediction))
    )
    scaled_response = tf.math.divide_no_nan(response, scale)
    scaled_prediction = tf.math.divide_no_nan(prediction, scale)
    error = tf.linalg.norm(scaled_prediction - scaled_response)
    response_norm = tf.linalg.norm(scaled_response)
    return tf.where(
        response_norm > 0.0,
        error / response_norm,
        tf.where(
            error == 0.0,
            tf.constant(0.0, tf.float64),
            tf.constant(float("inf"), tf.float64),
        ),
    )


def fit_dense_score_precision_tf(
    center_score: Any,
    training_offsets: Any,
    training_scores: Any,
    *,
    selection_offsets: Any | None = None,
    selection_scores: Any | None = None,
) -> dict[str, tf.Tensor]:
    """Fit ``K`` from batched analytical scores using TensorFlow only.

    ``training_scores`` and ``center_score`` must already be expressed in the
    coordinate system represented by ``training_offsets``. The returned coefficient is
    symmetric by construction, matching the historical dense fit.  No
    eigenvalue projection is performed. Symmetrizing unrestricted least squares
    is not the exact symmetry-constrained optimum for nonquadratic scores.
    Callers that require a position factor
    must reject a non-SPD raw result rather than silently manufacture one.

    ``design_condition`` is infinite when the existing numerical rank policy
    (singular values greater than 1e-12 times the largest) finds deficient rank.
    Otherwise it is the largest/smallest singular-value ratio. Infinity under
    that policy does not assert exact algebraic singularity.
    """

    center = _as_float64(center_score, "center_score")
    if center.shape.rank != 1 or center.shape[0] == 0:
        raise ValueError("center_score must have shape [dimension]")
    dimension = int(center.shape[0])
    tf.debugging.assert_all_finite(center, "center_score must be finite")
    offsets = _require_matrix(training_offsets, "training_offsets", dimension)
    scores = _require_matrix(training_scores, "training_scores", dimension)
    if scores.shape != offsets.shape:
        raise ValueError("training_scores must match training_offsets")
    if offsets.shape[0] is not None and offsets.shape[0] < dimension:
        raise ValueError("training design must have at least dimension rows")
    tf.debugging.assert_equal(tf.shape(scores), tf.shape(offsets))
    tf.debugging.assert_greater_equal(tf.shape(offsets)[0], dimension)
    response = center[tf.newaxis, :] - scores
    coefficient = complete_orthogonal_lstsq(offsets, response)
    raw_precision = _symmetric(coefficient)
    # Q has orthonormal columns, so A and the reduced R have the same singular
    # values. Avoid XLA's much costlier tall Jacobi SVD without forming A'A.
    design_for_svd = offsets
    if in_xla_context() and offsets.shape[0] is not None and offsets.shape[0] > dimension:
        design_for_svd = tf.linalg.qr(offsets, full_matrices=False)[1]
    singular_values = (_singular_value_program(dimension)(design_for_svd)
        if in_xla_context() and design_for_svd.shape == (dimension, dimension)
        else tf.linalg.svd(design_for_svd, compute_uv=False))
    largest_singular = tf.reduce_max(singular_values)
    smallest_singular = tf.reduce_min(singular_values)
    design_rank = tf.reduce_sum(
        tf.cast(
            singular_values > largest_singular * tf.constant(1.0e-12, tf.float64),
            tf.int32,
        )
    )
    design_condition = tf.where(
        design_rank == dimension,
        largest_singular / smallest_singular,
        tf.constant(float("inf"), tf.float64),
    )
    # The XLA backend can stop before resolving nearly repeated eigenvalues.
    # Reuse the same residual-refined eigensystem as the paired score fitter;
    # graph execution retains TensorFlow's original reference operation.
    eigenvalues = (eigenpair_program(dimension)(raw_precision)[0]
                   if in_xla_context() else tf.linalg.eigvalsh(raw_precision))
    minimum_eigenvalue = tf.reduce_min(eigenvalues)
    maximum_eigenvalue = tf.reduce_max(eigenvalues)
    precision_condition = tf.where(
        minimum_eigenvalue > 0.0,
        maximum_eigenvalue / minimum_eigenvalue,
        tf.constant(float("inf"), tf.float64),
    )
    result = {
        "raw_precision": raw_precision,
        "raw_eigenvalues": eigenvalues,
        "design_condition": design_condition,
        "design_rank": design_rank,
        "minimum_eigenvalue": minimum_eigenvalue,
        "maximum_eigenvalue": maximum_eigenvalue,
        "precision_condition": precision_condition,
        "raw_spd": tf.reduce_all(eigenvalues > 0.0),
    }
    if (selection_offsets is None) != (selection_scores is None):
        raise ValueError("selection offsets and scores must be supplied together")
    if selection_offsets is not None:
        heldout_offsets = _require_matrix(
            selection_offsets, "selection_offsets", dimension
        )
        heldout_scores = _require_matrix(
            selection_scores, "selection_scores", dimension
        )
        if heldout_scores.shape != heldout_offsets.shape:
            raise ValueError("selection_scores must match selection_offsets")
        result["selection_relative_rmse"] = _relative_response_rmse(
            raw_precision, center, heldout_offsets, heldout_scores
        )
    return result


__all__ = ["fit_dense_score_precision_tf"]
