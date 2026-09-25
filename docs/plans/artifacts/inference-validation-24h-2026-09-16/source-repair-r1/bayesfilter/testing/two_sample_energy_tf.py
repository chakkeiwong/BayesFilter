"""Diagnostic-only TensorFlow/XLA two-sample energy permutation test."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import tensorflow as tf


class EnergyDiagnosticError(ValueError):
    """Raised when the diagnostic two-sample contract is invalid."""


@dataclass(frozen=True)
class EnergyPermutationResult:
    statistic: tf.Tensor
    p_value: tf.Tensor
    exceedance_count: tf.Tensor
    permutation_statistics: tf.Tensor
    permutation_count: int
    sample_size_per_arm: int
    horizon: int
    seed: tuple[int, int]
    jit_compile: bool
    status: str


def _distance_matrix_kernel(paths: tf.Tensor) -> tf.Tensor:
    squared_norm = tf.reduce_sum(tf.square(paths), axis=1)
    squared = (
        squared_norm[:, tf.newaxis]
        + squared_norm[tf.newaxis, :]
        - 2.0 * tf.matmul(paths, paths, transpose_b=True)
    )
    return tf.sqrt(tf.maximum(squared, tf.constant(0.0, tf.float64)))


_distance_matrix_xla_core = tf.function(
    _distance_matrix_kernel, autograph=False, jit_compile=True
)
_distance_matrix_graph_core = tf.function(
    _distance_matrix_kernel, autograph=False, jit_compile=False
)


def _project_symmetric(distances: tf.Tensor) -> tf.Tensor:
    """Project XLA pairwise rounding onto the exact symmetric distance target."""

    values = 0.5 * (distances + tf.transpose(distances))
    return tf.linalg.set_diag(
        values, tf.zeros([values.shape[0]], tf.float64)
    )


def _distance_matrix_xla(paths: tf.Tensor) -> tf.Tensor:
    # Keep the quadratic kernel compiled, but place the projection outside the
    # XLA cluster so the compiler cannot undo the explicit symmetrization.
    return _project_symmetric(_distance_matrix_xla_core(paths))


def _distance_matrix_eager(paths: tf.Tensor) -> tf.Tensor:
    return _project_symmetric(_distance_matrix_graph_core(paths))


def _energy_from_labels_kernel(
    distances: tf.Tensor,
    labels: tf.Tensor,
    denominator: tf.Tensor,
) -> tf.Tensor:
    products = tf.matmul(labels, distances)
    return -tf.reduce_sum(products * labels, axis=1) / denominator


_energy_from_labels_xla = tf.function(
    _energy_from_labels_kernel, autograph=False, jit_compile=True
)
_energy_from_labels_eager = tf.function(
    _energy_from_labels_kernel, autograph=False, jit_compile=False
)


def _balanced_labels(
    *,
    permutation_count: int,
    pooled_count: int,
    sample_size: int,
    seed: tf.Tensor,
) -> tf.Tensor:
    scores = tf.random.stateless_uniform(
        [permutation_count, pooled_count],
        seed=seed,
        dtype=tf.float64,
        alg="philox",
    )
    order = tf.argsort(scores, axis=1, stable=True)
    selected = order[:, :sample_size]
    row = tf.repeat(tf.range(permutation_count, dtype=tf.int32), sample_size)
    indices = tf.stack((row, tf.reshape(selected, [-1])), axis=1)
    positive = tf.scatter_nd(
        indices,
        tf.fill([permutation_count * sample_size], tf.constant(2.0, tf.float64)),
        [permutation_count, pooled_count],
    )
    return positive - 1.0


def whole_path_energy_permutation_test(
    left_paths: Any,
    right_paths: Any,
    *,
    permutation_count: int,
    seed: Any,
    permutation_batch_size: int = 250,
    jit_compile: bool = True,
) -> EnergyPermutationResult:
    """Test equality of two iid complete-path samples using energy distance.

    The biased empirical energy statistic includes zero diagonal distances. The
    Monte Carlo p-value uses independently generated balanced label
    permutations and the standard plus-one correction.
    """

    if isinstance(permutation_count, bool) or not isinstance(permutation_count, int):
        raise EnergyDiagnosticError("permutation_count must be an integer")
    if permutation_count < 1:
        raise EnergyDiagnosticError("permutation_count must be positive")
    if isinstance(permutation_batch_size, bool) or not isinstance(
        permutation_batch_size, int
    ):
        raise EnergyDiagnosticError("permutation_batch_size must be an integer")
    if permutation_batch_size < 1:
        raise EnergyDiagnosticError("permutation_batch_size must be positive")
    if type(jit_compile) is not bool:
        raise EnergyDiagnosticError("jit_compile must be a Python bool")
    left = tf.convert_to_tensor(left_paths, tf.float64)
    right = tf.convert_to_tensor(right_paths, tf.float64)
    if left.shape.rank != 2 or right.shape.rank != 2:
        raise EnergyDiagnosticError("path samples must have rank two")
    if not left.shape.is_fully_defined() or not right.shape.is_fully_defined():
        raise EnergyDiagnosticError("path samples require fully defined static shapes")
    if left.shape != right.shape:
        raise EnergyDiagnosticError("two path samples must have equal shapes")
    sample_size, horizon = (int(value) for value in left.shape)
    if sample_size < 2 or horizon < 1:
        raise EnergyDiagnosticError("each arm requires at least two nonempty paths")
    root = tf.convert_to_tensor(seed, tf.int32)
    if root.shape != (2,):
        raise EnergyDiagnosticError("seed must have shape (2,)")
    seed_values = tuple(int(value) for value in root.numpy().tolist())
    try:
        tf.debugging.assert_all_finite(left, "left paths must be finite")
        tf.debugging.assert_all_finite(right, "right paths must be finite")
    except tf.errors.InvalidArgumentError as exc:
        raise EnergyDiagnosticError("path samples must be finite") from exc

    pooled = tf.concat((left, right), axis=0)
    distance_kernel = _distance_matrix_xla if jit_compile else _distance_matrix_eager
    statistic_kernel = (
        _energy_from_labels_xla if jit_compile else _energy_from_labels_eager
    )
    distances = distance_kernel(pooled)
    observed_labels = tf.concat(
        (tf.ones([sample_size], tf.float64), -tf.ones([sample_size], tf.float64)),
        axis=0,
    )[tf.newaxis, :]
    denominator = tf.constant(float(sample_size * sample_size), tf.float64)
    statistic = statistic_kernel(distances, observed_labels, denominator)[0]
    scale = tf.maximum(tf.constant(1.0, tf.float64), tf.reduce_max(distances))
    tolerance = tf.constant(4096.0 * 2.220446049250313e-16, tf.float64) * scale
    try:
        tf.debugging.assert_all_finite(distances, "distance matrix must be finite")
        tf.debugging.assert_near(
            distances,
            tf.transpose(distances),
            atol=tolerance,
            rtol=tf.constant(0.0, tf.float64),
            message="distance matrix must be symmetric",
        )
        tf.debugging.assert_near(
            tf.linalg.diag_part(distances),
            tf.zeros([2 * sample_size], tf.float64),
            atol=tolerance,
            rtol=tf.constant(0.0, tf.float64),
            message="distance diagonal must be zero",
        )
        tf.debugging.assert_greater_equal(
            statistic,
            -tolerance,
            message="energy statistic is negative beyond roundoff",
        )
    except tf.errors.InvalidArgumentError as exc:
        raise EnergyDiagnosticError("energy distance validity check failed") from exc
    statistic = tf.maximum(statistic, tf.constant(0.0, tf.float64))

    batches = []
    completed = 0
    batch_index = 0
    while completed < permutation_count:
        count = min(permutation_batch_size, permutation_count - completed)
        batch_seed = tf.random.experimental.stateless_fold_in(
            root, tf.constant(batch_index, tf.int32), alg="philox"
        )
        labels = _balanced_labels(
            permutation_count=count,
            pooled_count=2 * sample_size,
            sample_size=sample_size,
            seed=batch_seed,
        )
        batches.append(statistic_kernel(distances, labels, denominator))
        completed += count
        batch_index += 1
    permutation_statistics = tf.concat(batches, axis=0)
    try:
        tf.debugging.assert_equal(
            tf.shape(permutation_statistics, out_type=tf.int32)[0],
            permutation_count,
            message="permutation count mismatch",
        )
        tf.debugging.assert_all_finite(
            permutation_statistics, "permutation statistics must be finite"
        )
    except tf.errors.InvalidArgumentError as exc:
        raise EnergyDiagnosticError("permutation statistic validity check failed") from exc
    exceedance = tf.reduce_sum(
        tf.cast(permutation_statistics >= statistic, tf.int64)
    )
    p_value = tf.cast(1 + exceedance, tf.float64) / tf.constant(
        float(permutation_count + 1), tf.float64
    )
    if not math.isfinite(float(p_value)):
        raise EnergyDiagnosticError("permutation p-value is nonfinite")
    return EnergyPermutationResult(
        statistic=statistic,
        p_value=p_value,
        exceedance_count=exceedance,
        permutation_statistics=permutation_statistics,
        permutation_count=permutation_count,
        sample_size_per_arm=sample_size,
        horizon=horizon,
        seed=seed_values,
        jit_compile=jit_compile,
        status="VALID",
    )


__all__ = [
    "EnergyDiagnosticError",
    "EnergyPermutationResult",
    "whole_path_energy_permutation_test",
]
