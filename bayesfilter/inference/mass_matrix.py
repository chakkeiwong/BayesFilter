"""TensorFlow mass-matrix construction with explicit provenance.

Numerical programs default to XLA with stable signatures. Python scalars are
materialized only for fail-closed validation and artifact metadata.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from bayesfilter.inference import mass_matrix_tf as native


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
    return native.covariance_program(matrix.shape[0])(matrix, tf.constant(jitter_value, tf.float32))


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
    if not _scalar_bool(native.finite_matrix(empirical)):
        raise ValueError("empirical covariance must be finite")
    dimension = empirical.shape[0]
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

    partition = tuple((block["start"], block["stop"]) for block in normalized_blocks)
    projected, summary, valid = native.structured_program(dimension, partition)(
        empirical, tf.constant(weight, tf.float64), tf.constant(floor, tf.float64),
        tf.constant(condition_cap, tf.float64))
    if not _scalar_bool(valid):
        raise ValueError("structured covariance eigenvalues must be finite")
    raw_minimum, regularized_minimum, regularized_maximum = summary.numpy().tolist()

    report = {
        "method": "empirical_covariance_structural_projection_shrinkage_eigen_clamp",
        "numerical_backend": "tensorflow",
        "family": family,
        "blocks": normalized_blocks,
        "shrinkage": weight,
        "requested_eigenvalue_floor": floor,
        "max_condition_number": condition_cap,
        "raw_min_block_eigenvalue": raw_minimum,
        "regularized_min_block_eigenvalue": regularized_minimum,
        "regularized_max_block_eigenvalue": regularized_maximum,
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
    matrix, jitter_value, floor, max_condition = _precision_inputs(
        precision, jitter, eigenvalue_floor, max_condition_number)
    regularized, covariance, diagnostics, flags, diagonal_valid = native.precision_program(
        matrix.shape[0], dense=bool(dense))(matrix, tf.constant(jitter_value, tf.float64),
            tf.constant(floor, tf.float64), tf.constant(max_condition or 0., tf.float64))
    report = _precision_report(diagnostics, flags, jitter_value, eigenvalue_floor, max_condition)
    if dense:
        matrix_kind = "dense"
    else:
        if not _scalar_bool(diagonal_valid):
            raise ValueError("regularized precision diagonal must be positive finite")
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

    matrix, jitter_value, floor, max_condition = _precision_inputs(
        precision, jitter, eigenvalue_floor, max_condition_number)
    regularized, diagnostics, flags = native.precision_program(matrix.shape[0])(
        matrix, tf.constant(jitter_value, tf.float64), tf.constant(floor, tf.float64),
        tf.constant(max_condition or 0., tf.float64))
    report = _precision_report(diagnostics, flags, jitter_value, eigenvalue_floor, max_condition)
    return regularized, report


def _precision_inputs(precision, jitter, eigenvalue_floor, max_condition_number):
    matrix = _square_tensor(precision, "precision")
    if not _scalar_bool(native.finite_matrix(matrix)):
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

    return matrix, jitter_value, floor, max_condition


def _precision_report(diagnostics, flags, jitter, requested_floor, max_condition):
    finite, finite_values, has_floor, asymmetric = flags.numpy().tolist()
    if not finite:
        raise ValueError("precision must be finite")
    if not finite_values:
        raise ValueError("precision eigenvalues must be finite")
    if not has_floor:
        raise ValueError("precision must have a positive eigenvalue; pass eigenvalue_floor")
    floor, raw_min, raw_max, clipped_min, clipped_max, nonpositive, clipped, asymmetry = diagnostics.numpy().tolist()
    return {
        "method": "symmetric_eigendecomposition_floor",
        "numerical_backend": "tensorflow",
        "jitter": jitter,
        "requested_eigenvalue_floor": None if requested_floor is None else float(requested_floor),
        "effective_eigenvalue_floor": float(floor),
        "max_condition_number": max_condition,
        "raw_min_eigenvalue": raw_min,
        "raw_max_eigenvalue": raw_max,
        "regularized_min_eigenvalue": clipped_min,
        "regularized_max_eigenvalue": clipped_max,
        "raw_nonpositive_eigenvalue_count": int(nonpositive),
        "clipped_eigenvalue_count": int(clipped),
        "symmetry_projection": "average_with_transpose",
        "input_asymmetry_max_abs": asymmetry,
        "input_asymmetric": asymmetric,
        "diagonal_fallback_used": False,
        "silent_eigenvalue_reflection": False,
    }


def whitening_from_covariance(covariance: Any, *, jitter: float = 1e-9) -> tf.Tensor:
    """Return `F` with covariance equal to `F @ F.T` up to roundoff."""

    matrix = _square_tensor(covariance, "covariance")
    jitter_value = _nonnegative_finite(jitter, "jitter")
    return native.covariance_program(matrix.shape[0], whitening=True)(matrix, tf.constant(jitter_value, tf.float32))


def _square_tensor(value: Any, name: str) -> tf.Tensor:
    matrix = tf.convert_to_tensor(value, dtype=tf.float64)
    if matrix.shape.rank != 2:
        raise ValueError(f"{name} must be a square matrix")
    rows, columns = matrix.shape
    if rows != columns:
        raise ValueError(f"{name} must be a square matrix")
    return matrix


def _eigen_summary(matrix: Any) -> dict[str, Any]:
    square = _square_tensor(matrix, "matrix")
    eigenvalues, summary, finite, positive = native.summary_program(square.shape[0])(square)
    minimum, maximum, condition = summary.numpy().tolist()
    return {
        "finite": bool(finite),
        "positive": bool(positive),
        "min": minimum,
        "max": maximum,
        "condition_number": condition,
        "eigenvalues": tuple(eigenvalues.numpy().tolist()),
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
    return math.isfinite(value)


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
