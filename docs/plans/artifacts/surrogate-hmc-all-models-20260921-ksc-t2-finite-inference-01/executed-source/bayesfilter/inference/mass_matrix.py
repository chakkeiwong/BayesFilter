"""TensorFlow mass-matrix construction with explicit provenance.

TensorFlow performs every numerical operation in this module. Python scalars
are materialized only for fail-closed validation and artifact metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import tensorflow as tf


_FLOAT64_EPSILON = 2.220446049250313e-16  # IEEE-754 binary64 machine epsilon.


@dataclass(frozen=True)
class MassMatrixResult:
    covariance: tf.Tensor
    source: str
    matrix_kind: str
    jitter: float
    eigenvalue_floor: float | None = None
    regularized_precision: tf.Tensor | None = None
    precision_eigen_summary: dict[str, Any] | None = None
    covariance_eigen_summary: dict[str, Any] | None = None
    regularization_report: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        covariance = _square_tensor(self.covariance, "covariance")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "matrix_kind", str(self.matrix_kind))
        object.__setattr__(self, "jitter", float(self.jitter))
        if self.eigenvalue_floor is not None:
            object.__setattr__(self, "eigenvalue_floor", float(self.eigenvalue_floor))
        if self.regularized_precision is not None:
            precision = _square_tensor(self.regularized_precision, "regularized_precision")
            object.__setattr__(self, "regularized_precision", precision)
            object.__setattr__(self, "precision_eigen_summary", _eigen_summary(precision))
        elif self.precision_eigen_summary is not None:
            object.__setattr__(
                self,
                "precision_eigen_summary",
                _normalize_eigen_summary(self.precision_eigen_summary),
            )
        object.__setattr__(self, "covariance_eigen_summary", _eigen_summary(covariance))
        report = {} if self.regularization_report is None else dict(self.regularization_report)
        object.__setattr__(self, "regularization_report", report)


def regularize_covariance(covariance: Any, *, jitter: float = 1e-9) -> tf.Tensor:
    matrix = _square_tensor(covariance, "covariance")
    jitter_value = _nonnegative_finite(jitter, "jitter")
    dimension = tf.shape(matrix, out_type=tf.int32)[0]
    return matrix + tf.cast(jitter_value, matrix.dtype) * tf.eye(
        dimension, dtype=matrix.dtype
    )


def structured_covariance_from_empirical(
    covariance: Any,
    *,
    blocks: Sequence[Mapping[str, Any]] | None = None,
    diagonal: bool = False,
    shrinkage: float = 0.10,
    eigenvalue_floor: float = 1.0e-6,
    max_condition_number: float = 1.0e6,
    source: str = "discarded_pilot_empirical_covariance",
) -> MassMatrixResult:
    """Project a discarded-pilot covariance to diagonal or fixed blocks.

    This is the BayesFilter-owned structural-mass constructor.  It first
    projects the empirical covariance onto either a diagonal family or a
    caller-declared, complete block partition, then shrinks each block toward
    its diagonal and clamps covariance eigenvalues.  The function does not
    estimate a center, run HMC, or decide whether pilot evidence is sufficient;
    callers must make those provenance and campaign decisions before invoking
    it.
    """

    empirical = _square_tensor(covariance, "empirical covariance")
    if not _scalar_bool(tf.reduce_all(tf.math.is_finite(empirical))):
        raise ValueError("empirical covariance must be finite")
    dimension = _scalar_int(tf.shape(empirical, out_type=tf.int32)[0])
    if dimension == 0:
        raise ValueError("empirical covariance must have positive dimension")
    use_diagonal = bool(diagonal)
    if use_diagonal and blocks is not None:
        raise ValueError("diagonal and blocks are mutually exclusive")
    if not use_diagonal and blocks is None:
        raise ValueError("blocks are required unless diagonal=True")
    weight = float(shrinkage)
    floor = float(eigenvalue_floor)
    condition_cap = float(max_condition_number)
    if not _python_finite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("shrinkage must be finite and in [0, 1]")
    if not _python_finite(floor) or floor <= 0.0:
        raise ValueError("eigenvalue_floor must be positive and finite")
    if not _python_finite(condition_cap) or condition_cap <= 1.0:
        raise ValueError("max_condition_number must be finite and greater than one")
    source_label = str(source)
    if not source_label:
        raise ValueError("source must be non-empty")

    if use_diagonal:
        normalized_blocks = tuple(
            {"name": f"coordinate_{index}", "start": index, "stop": index + 1}
            for index in range(dimension)
        )
        family = "diagonal"
    else:
        normalized: list[dict[str, Any]] = []
        cursor = 0
        for index, item in enumerate(tuple(blocks or ())):
            if not isinstance(item, Mapping):
                raise TypeError("every structural block must be a mapping")
            name = str(item.get("name", f"block_{index}"))
            start = int(item.get("start", -1))
            stop = int(item.get("stop", -1))
            if not name or start != cursor or stop <= start or stop > dimension:
                raise ValueError(
                    "structural blocks must be named, contiguous, ordered, and in bounds"
                )
            normalized.append({"name": name, "start": start, "stop": stop})
            cursor = stop
        if cursor != dimension:
            raise ValueError("structural blocks must form a complete partition")
        normalized_blocks = tuple(normalized)
        family = "structural_block"

    regularized_blocks: list[tf.Tensor] = []
    raw_block_eigenvalues: list[float] = []
    regularized_block_eigenvalues: list[float] = []
    for block in normalized_blocks:
        start = int(block["start"])
        stop = int(block["stop"])
        empirical_block = empirical[start:stop, start:stop]
        raw = 0.5 * (empirical_block + tf.linalg.matrix_transpose(empirical_block))
        diagonal_target = tf.linalg.diag(tf.linalg.diag_part(raw))
        shrunk = (1.0 - weight) * raw + weight * diagonal_target
        shrunk = 0.5 * (shrunk + tf.linalg.matrix_transpose(shrunk))
        values, vectors = tf.linalg.eigh(shrunk)
        if not _scalar_bool(tf.reduce_all(tf.math.is_finite(values))):
            raise ValueError("structured covariance eigenvalues must be finite")
        raw_values = tuple(float(value) for value in values.numpy().tolist())
        raw_block_eigenvalues.extend(raw_values)
        largest = max(_scalar_float(tf.reduce_max(values)), floor)
        effective_floor = max(floor, largest / condition_cap)
        regularized_values = tf.maximum(
            values, tf.constant(effective_floor, dtype=tf.float64)
        )
        regularized_block_eigenvalues.extend(
            float(value) for value in regularized_values.numpy().tolist()
        )
        regularized = tf.matmul(
            vectors * regularized_values[tf.newaxis, :],
            vectors,
            transpose_b=True,
        )
        regularized_blocks.append(
            0.5 * (regularized + tf.linalg.matrix_transpose(regularized))
        )

    projected = tf.linalg.LinearOperatorBlockDiag(
        tuple(tf.linalg.LinearOperatorFullMatrix(block) for block in regularized_blocks)
    ).to_dense()

    report = {
        "method": "empirical_covariance_structural_projection_shrinkage_eigen_clamp",
        "numerical_backend": "tensorflow",
        "family": family,
        "blocks": normalized_blocks,
        "shrinkage": weight,
        "requested_eigenvalue_floor": floor,
        "max_condition_number": condition_cap,
        "raw_min_block_eigenvalue": float(min(raw_block_eigenvalues)),
        "regularized_min_block_eigenvalue": float(
            min(regularized_block_eigenvalues)
        ),
        "regularized_max_block_eigenvalue": float(
            max(regularized_block_eigenvalues)
        ),
        "cross_block_entries_zero": True,
    }
    return MassMatrixResult(
        covariance=projected,
        source=source_label,
        matrix_kind=family,
        jitter=0.0,
        eigenvalue_floor=floor,
        regularization_report=report,
    )


def covariance_from_precision(
    precision: Any,
    *,
    source: str,
    jitter: float = 1e-9,
    eigenvalue_floor: float | None = None,
    max_condition_number: float | None = None,
    dense: bool = True,
) -> MassMatrixResult:
    regularized, report = regularize_precision(
        precision,
        jitter=jitter,
        eigenvalue_floor=eigenvalue_floor,
        max_condition_number=max_condition_number,
    )
    if dense:
        covariance = tf.linalg.inv(regularized)
        covariance = 0.5 * (covariance + tf.linalg.matrix_transpose(covariance))
        matrix_kind = "dense"
    else:
        diagonal = tf.linalg.diag_part(regularized)
        if not _scalar_bool(tf.reduce_all(tf.math.is_finite(diagonal))):
            raise ValueError("regularized precision diagonal must be positive finite")
        if not _scalar_bool(tf.reduce_all(diagonal > 0.0)):
            raise ValueError("regularized precision diagonal must be positive finite")
        covariance = tf.linalg.diag(tf.math.reciprocal(diagonal))
        matrix_kind = "diagonal"
        report = {
            **report,
            "diagonal_fallback_used": True,
            "diagonal_fallback_source": "regularized_precision_diagonal",
        }
    return MassMatrixResult(
        covariance=covariance,
        source=source,
        matrix_kind=matrix_kind,
        jitter=float(jitter),
        eigenvalue_floor=report["effective_eigenvalue_floor"],
        regularized_precision=regularized,
        precision_eigen_summary=_eigen_summary(regularized),
        covariance_eigen_summary=_eigen_summary(covariance),
        regularization_report=report,
    )


def covariance_from_negative_hessian(
    negative_hessian: Any,
    *,
    source: str = "negative_hessian",
    jitter: float = 1e-9,
    eigenvalue_floor: float | None = None,
    max_condition_number: float | None = None,
    dense: bool = True,
) -> MassMatrixResult:
    """Convert an explicit negative log-posterior Hessian to covariance."""

    return covariance_from_precision(
        negative_hessian,
        source=source,
        jitter=jitter,
        eigenvalue_floor=eigenvalue_floor,
        max_condition_number=max_condition_number,
        dense=dense,
    )


def regularize_precision(
    precision: Any,
    *,
    jitter: float = 1e-9,
    eigenvalue_floor: float | None = None,
    max_condition_number: float | None = None,
) -> tuple[tf.Tensor, dict[str, Any]]:
    """Return a positive-definite precision tensor and regularization report."""

    matrix = _square_tensor(precision, "precision")
    if not _scalar_bool(tf.reduce_all(tf.math.is_finite(matrix))):
        raise ValueError("precision must be finite")
    jitter_value = _nonnegative_finite(jitter, "jitter")
    floor = 0.0 if eigenvalue_floor is None else _nonnegative_finite(
        eigenvalue_floor, "eigenvalue_floor"
    )
    if max_condition_number is None:
        max_condition = None
    else:
        max_condition = float(max_condition_number)
        if not _python_finite(max_condition) or max_condition <= 1.0:
            raise ValueError("max_condition_number must be finite and greater than 1")

    asymmetry = matrix - tf.linalg.matrix_transpose(matrix)
    asymmetry_max_abs = _scalar_float(tf.reduce_max(tf.abs(asymmetry)))
    symmetric = 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))
    dimension = tf.shape(symmetric, out_type=tf.int32)[0]
    jittered = symmetric + tf.cast(jitter_value, symmetric.dtype) * tf.eye(
        dimension, dtype=symmetric.dtype
    )
    raw_eigvals, eigvecs = tf.linalg.eigh(jittered)
    if not _scalar_bool(tf.reduce_all(tf.math.is_finite(raw_eigvals))):
        raise ValueError("precision eigenvalues must be finite")

    positive_raw = tf.boolean_mask(raw_eigvals, raw_eigvals > 0.0)
    if floor == 0.0:
        if _scalar_int(tf.size(positive_raw)) == 0:
            raise ValueError("precision must have a positive eigenvalue; pass eigenvalue_floor")
        maximum_positive = _scalar_float(tf.reduce_max(positive_raw))
        floor = max(floor, _FLOAT64_EPSILON * max(1.0, maximum_positive))
    raw_max = _scalar_float(tf.reduce_max(raw_eigvals))
    if max_condition is not None:
        floor = max(floor, raw_max / max_condition)
    if floor <= 0.0:
        floor = _FLOAT64_EPSILON

    floor_tensor = tf.cast(floor, raw_eigvals.dtype)
    regularized_eigvals = tf.maximum(raw_eigvals, floor_tensor)
    regularized = tf.matmul(
        eigvecs * regularized_eigvals[tf.newaxis, :],
        eigvecs,
        transpose_b=True,
    )
    regularized = 0.5 * (regularized + tf.linalg.matrix_transpose(regularized))
    clipped = regularized_eigvals > raw_eigvals
    report = {
        "method": "symmetric_eigendecomposition_floor",
        "numerical_backend": "tensorflow",
        "jitter": jitter_value,
        "requested_eigenvalue_floor": (
            None if eigenvalue_floor is None else float(eigenvalue_floor)
        ),
        "effective_eigenvalue_floor": float(floor),
        "max_condition_number": max_condition,
        "raw_min_eigenvalue": _scalar_float(tf.reduce_min(raw_eigvals)),
        "raw_max_eigenvalue": raw_max,
        "regularized_min_eigenvalue": _scalar_float(tf.reduce_min(regularized_eigvals)),
        "regularized_max_eigenvalue": _scalar_float(tf.reduce_max(regularized_eigvals)),
        "raw_nonpositive_eigenvalue_count": _scalar_int(
            tf.reduce_sum(tf.cast(raw_eigvals <= 0.0, tf.int32))
        ),
        "clipped_eigenvalue_count": _scalar_int(
            tf.reduce_sum(tf.cast(clipped, tf.int32))
        ),
        "symmetry_projection": "average_with_transpose",
        "input_asymmetry_max_abs": asymmetry_max_abs,
        "input_asymmetric": bool(asymmetry_max_abs > 0.0),
        "diagonal_fallback_used": False,
        "silent_eigenvalue_reflection": False,
    }
    return regularized, report


def whitening_from_covariance(covariance: Any, *, jitter: float = 1e-9) -> tf.Tensor:
    """Return `F` with covariance equal to `F @ F.T` up to roundoff."""

    return tf.linalg.cholesky(regularize_covariance(covariance, jitter=jitter))


def _square_tensor(value: Any, name: str) -> tf.Tensor:
    matrix = tf.convert_to_tensor(value, dtype=tf.float64)
    if matrix.shape.rank != 2:
        raise ValueError(f"{name} must be a square matrix")
    rows = _scalar_int(tf.shape(matrix, out_type=tf.int32)[0])
    columns = _scalar_int(tf.shape(matrix, out_type=tf.int32)[1])
    if rows != columns:
        raise ValueError(f"{name} must be a square matrix")
    return matrix


def _eigen_summary(matrix: Any) -> dict[str, Any]:
    square = _square_tensor(matrix, "matrix")
    symmetric = 0.5 * (square + tf.linalg.matrix_transpose(square))
    eigenvalues = tf.linalg.eigvalsh(symmetric)
    finite = _scalar_bool(tf.reduce_all(tf.math.is_finite(eigenvalues)))
    minimum = _scalar_float(tf.reduce_min(eigenvalues)) if finite else float("nan")
    maximum = _scalar_float(tf.reduce_max(eigenvalues)) if finite else float("nan")
    positive = bool(finite and minimum > 0.0)
    return {
        "finite": finite,
        "positive": positive,
        "min": minimum,
        "max": maximum,
        "condition_number": maximum / minimum if positive else float("inf"),
        "eigenvalues": tuple(float(value) for value in eigenvalues.numpy().tolist()),
    }


def _normalize_eigen_summary(summary: dict[str, Any]) -> dict[str, Any]:
    eigenvalues = summary.get("eigenvalues", ())
    return {
        "finite": bool(summary.get("finite")),
        "positive": bool(summary.get("positive")),
        "min": float(summary.get("min")),
        "max": float(summary.get("max")),
        "condition_number": float(summary.get("condition_number")),
        "eigenvalues": tuple(float(value) for value in eigenvalues),
    }


def _nonnegative_finite(value: Any, name: str) -> float:
    number = float(value)
    if not _python_finite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _python_finite(value: float) -> bool:
    return bool(tf.math.is_finite(tf.convert_to_tensor(value, dtype=tf.float64)).numpy())


def _scalar_bool(value: tf.Tensor) -> bool:
    return bool(tf.convert_to_tensor(value).numpy())


def _scalar_float(value: tf.Tensor) -> float:
    return float(tf.convert_to_tensor(value).numpy())


def _scalar_int(value: tf.Tensor) -> int:
    return int(tf.convert_to_tensor(value).numpy())


__all__ = [
    "MassMatrixResult",
    "covariance_from_negative_hessian",
    "covariance_from_precision",
    "regularize_covariance",
    "regularize_precision",
    "structured_covariance_from_empirical",
    "whitening_from_covariance",
]
