"""Dense fixed-center precision reconstruction from antithetic score directions.

For standardized offsets ``z = r q`` around a fixed center ``c``, let ``g`` be
the standardized score and define the central response

``y(q) = (g(c - z) - g(c + z)) / 2``.

If ``P = -H`` is the local precision, then ``y(q) = P z + O(r^3)``. With the
positive offsets stacked as rows of a square full-rank matrix ``Z`` and the
responses stacked as rows of ``Y``, this module reconstructs the unconstrained
directional map from ``Z P_raw.T = Y``. Raw asymmetry is measured before any
interpretation of ``(P_raw + P_raw.T) / 2``. No eigenvalue projection,
shrinkage, covariance, factor, mass artifact, initializer, or HMC operation is
implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import tensorflow as tf

from bayesfilter.inference.fixed_center_curvature import compare_precision_geometry


DENSE_DIRECTIONAL_GEOMETRY_NONCLAIMS = (
    "fixed-center finite-radius directional diagnostic only",
    "not a MAP or exact Hessian claim",
    "not a posterior covariance or mass artifact",
    "not HMC, convergence, identification, or scientific evidence",
    "load-time HMC exposure from the comparison module is not an HMC operation",
)


@dataclass(frozen=True)
class DenseDirectionalScoreGeometryConfig:
    """Prospectively fixed numerical and candidate-qualification thresholds."""

    symmetry_projection_relative_frobenius_cap: float = 0.02
    max_condition_number: float = 1.0e8
    solve_relative_residual_cap: float = 1.0e-10
    orthogonality_max_abs_cap: float = 1.0e-12
    generalized_eigenvalue_spread_cap: float = 100.0
    trace_normalized_frobenius_cap: float = 0.01
    trace_normalized_operator_cap: float = 0.01
    principal_angle_degrees_cap: float = 5.0
    principal_subspace_rank: int = 64
    antithetic_absolute_tolerance: float = 1.0e-14

    def __post_init__(self) -> None:
        for name in (
            "symmetry_projection_relative_frobenius_cap",
            "max_condition_number",
            "solve_relative_residual_cap",
            "orthogonality_max_abs_cap",
            "generalized_eigenvalue_spread_cap",
            "trace_normalized_frobenius_cap",
            "trace_normalized_operator_cap",
            "principal_angle_degrees_cap",
            "antithetic_absolute_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if self.max_condition_number < 1.0:
            raise ValueError("max_condition_number must be at least one")
        if self.generalized_eigenvalue_spread_cap < 1.0:
            raise ValueError("generalized_eigenvalue_spread_cap must be at least one")
        if self.principal_angle_degrees_cap > 90.0:
            raise ValueError("principal_angle_degrees_cap must not exceed 90")
        rank = int(self.principal_subspace_rank)
        if rank <= 0:
            raise ValueError("principal_subspace_rank must be positive")
        object.__setattr__(self, "principal_subspace_rank", rank)

    def payload(self) -> Mapping[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class DenseDirectionalScoreGeometryResult:
    """Raw reconstruction plus fail-closed central-symmetric interpretation."""

    valid: bool
    accepted: bool
    status: str
    directional_precision_raw: np.ndarray
    central_symmetric_precision: np.ndarray | None
    central_symmetric_raw_eigenvalues: np.ndarray | None
    diagnostics: Mapping[str, Any]
    nonclaims: tuple[str, ...] = DENSE_DIRECTIONAL_GEOMETRY_NONCLAIMS

    def __post_init__(self) -> None:
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "status", str(self.status))
        for name in (
            "directional_precision_raw",
            "central_symmetric_precision",
            "central_symmetric_raw_eigenvalues",
        ):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=np.float64).copy()
                array.setflags(write=False)
                object.__setattr__(self, name, array)
        object.__setattr__(self, "diagnostics", _json_ready(dict(self.diagnostics)))
        object.__setattr__(self, "nonclaims", tuple(str(item) for item in self.nonclaims))

    def payload(self, *, include_matrices: bool = False) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "schema": "bayesfilter.dense_directional_score_geometry.v1",
            "valid": self.valid,
            "accepted": self.accepted,
            "status": self.status,
            "diagnostics": self.diagnostics,
            "nonclaims": self.nonclaims,
        }
        if include_matrices:
            payload.update(
                {
                    "directional_precision_raw": self.directional_precision_raw,
                    "central_symmetric_precision": self.central_symmetric_precision,
                    "central_symmetric_raw_eigenvalues": (
                        self.central_symmetric_raw_eigenvalues
                    ),
                }
            )
        return _json_ready(payload)


def fit_dense_directional_score_geometry(
    *,
    center_score_z: Any,
    antithetic_offsets_z: Any,
    antithetic_scores_z: Any,
    config: DenseDirectionalScoreGeometryConfig | None = None,
) -> DenseDirectionalScoreGeometryResult:
    """Reconstruct one raw dense directional precision without SPD rescue.

    Rows must be interleaved as ``(+z_0, -z_0, +z_1, -z_1, ...)`` with exactly
    one positive direction per dimension. Malformed inputs are harness errors
    and raise ``ValueError``. Excessive finite-radius asymmetry is instead a
    valid candidate rejection and does not authorize interpreting eigenvalues
    of the symmetric projection.
    """

    cfg = DenseDirectionalScoreGeometryConfig() if config is None else config
    center = tf.reshape(tf.convert_to_tensor(center_score_z, tf.float64), [-1])
    if center.shape[0] is None:
        raise ValueError("center_score_z must have static dimension")
    dimension = int(center.shape[0])
    offsets = _antithetic_matrix(
        antithetic_offsets_z, dimension, "antithetic_offsets_z"
    )
    scores = _antithetic_matrix(
        antithetic_scores_z, dimension, "antithetic_scores_z"
    )
    if offsets.shape != scores.shape:
        raise ValueError("antithetic offsets and scores must have matching shapes")
    finite = tf.reduce_all(
        tf.math.is_finite(
            tf.concat(
                (tf.reshape(center, [-1]), tf.reshape(offsets, [-1]), tf.reshape(scores, [-1])),
                axis=0,
            )
        )
    )
    if not bool(finite.numpy()):
        raise ValueError("dense directional inputs must be finite")

    positive, frame = _validate_antithetic_frame(
        offsets,
        dimension=dimension,
        antithetic_absolute_tolerance=cfg.antithetic_absolute_tolerance,
        orthogonality_max_abs_cap=cfg.orthogonality_max_abs_cap,
        reject_orthogonality=False,
    )

    plus_score = scores[0::2]
    minus_score = scores[1::2]
    response = 0.5 * (minus_score - plus_score)
    even_response = plus_score + minus_score - 2.0 * center[None, :]
    solved_transpose = tf.linalg.solve(positive, response)
    directional_raw = tf.transpose(solved_transpose)
    prediction = tf.matmul(positive, directional_raw, transpose_b=True)
    solve_residual = float(
        (
            tf.linalg.norm(prediction - response)
            / tf.maximum(tf.linalg.norm(response), tf.constant(1.0e-15, tf.float64))
        ).numpy()
    )
    even_fraction = float(
        (
            tf.linalg.norm(even_response)
            / tf.maximum(
                tf.linalg.norm(minus_score - plus_score),
                tf.constant(1.0e-15, tf.float64),
            )
        ).numpy()
    )
    projection_burden = float(
        (
            tf.linalg.norm(directional_raw - tf.transpose(directional_raw))
            / tf.maximum(
                2.0 * tf.linalg.norm(directional_raw),
                tf.constant(1.0e-15, tf.float64),
            )
        ).numpy()
    )
    diagnostics: dict[str, Any] = {
        "dimension": dimension,
        "positive_direction_count": dimension,
        **frame,
        "solve_relative_frobenius_residual": solve_residual,
        "symmetry_projection_relative_frobenius": projection_burden,
        "even_response_relative_frobenius": even_fraction,
        "central_symmetric_interpreted": False,
        "eigenvalue_projection_used": False,
        "shrinkage_used": False,
        "factor_route_used": False,
        "covariance_built": False,
        "mass_artifact_built": False,
        "hmc_operation_called": False,
        "config": cfg.payload(),
    }
    raw_np = directional_raw.numpy()
    if (
        float(frame["orthogonality_max_abs_error"])
        > cfg.orthogonality_max_abs_cap
        or solve_residual > cfg.solve_relative_residual_cap
    ):
        return DenseDirectionalScoreGeometryResult(
            valid=False,
            accepted=False,
            status="directional_reconstruction_numerically_invalid",
            directional_precision_raw=raw_np,
            central_symmetric_precision=None,
            central_symmetric_raw_eigenvalues=None,
            diagnostics=diagnostics,
        )
    if projection_burden > cfg.symmetry_projection_relative_frobenius_cap:
        return DenseDirectionalScoreGeometryResult(
            valid=True,
            accepted=False,
            status="symmetry_projection_burden_rejected",
            directional_precision_raw=raw_np,
            central_symmetric_precision=None,
            central_symmetric_raw_eigenvalues=None,
            diagnostics=diagnostics,
        )

    symmetric = 0.5 * (directional_raw + tf.transpose(directional_raw))
    eigenvalues = tf.linalg.eigvalsh(symmetric)
    minimum = float(tf.reduce_min(eigenvalues).numpy())
    maximum = float(tf.reduce_max(eigenvalues).numpy())
    nonpositive = int(
        tf.reduce_sum(tf.cast(eigenvalues <= 0.0, tf.int32)).numpy()
    )
    condition = np.inf if nonpositive else maximum / minimum
    diagnostics.update(
        {
            "central_symmetric_interpreted": True,
            "central_symmetric_raw_minimum_eigenvalue": minimum,
            "central_symmetric_raw_maximum_eigenvalue": maximum,
            "central_symmetric_raw_nonpositive_eigenvalue_count": nonpositive,
            "central_symmetric_raw_condition_number": condition,
            "central_symmetric_raw_spd": nonpositive == 0,
        }
    )
    if nonpositive:
        status = "central_symmetric_raw_not_spd"
    elif not np.isfinite(condition) or condition > cfg.max_condition_number:
        status = "central_symmetric_condition_number_rejected"
    else:
        status = "dense_directional_candidate_usable"
    return DenseDirectionalScoreGeometryResult(
        valid=True,
        accepted=status == "dense_directional_candidate_usable",
        status=status,
        directional_precision_raw=raw_np,
        central_symmetric_precision=symmetric.numpy(),
        central_symmetric_raw_eigenvalues=eigenvalues.numpy(),
        diagnostics=diagnostics,
    )


def directional_prediction_relative_frobenius(
    precision: Any,
    antithetic_offsets_z: Any,
    antithetic_scores_z: Any,
    *,
    antithetic_absolute_tolerance: float = 1.0e-14,
    orthogonality_max_abs_cap: float = 1.0e-12,
) -> float:
    """Evaluate ``||Z P.T - Y||_F / max(||Y||_F, 1e-15)`` exactly."""

    matrix = _symmetric_matrix(precision, "precision")
    dimension = int(matrix.shape[0])
    offsets = _antithetic_matrix(
        antithetic_offsets_z, dimension, "antithetic_offsets_z"
    )
    scores = _antithetic_matrix(
        antithetic_scores_z, dimension, "antithetic_scores_z"
    )
    if offsets.shape != scores.shape:
        raise ValueError("antithetic offsets and scores must have matching shapes")
    if not bool(tf.reduce_all(tf.math.is_finite(matrix)).numpy()):
        raise ValueError("precision must be finite")
    if not bool(tf.reduce_all(tf.math.is_finite(offsets)).numpy()) or not bool(
        tf.reduce_all(tf.math.is_finite(scores)).numpy()
    ):
        raise ValueError("holdout offsets and scores must be finite")
    positive, _ = _validate_antithetic_frame(
        offsets,
        dimension=dimension,
        antithetic_absolute_tolerance=antithetic_absolute_tolerance,
        orthogonality_max_abs_cap=orthogonality_max_abs_cap,
        reject_orthogonality=True,
    )
    response = 0.5 * (scores[1::2] - scores[0::2])
    residual = tf.matmul(positive, matrix, transpose_b=True) - response
    return float(
        (
            tf.linalg.norm(residual)
            / tf.maximum(tf.linalg.norm(response), tf.constant(1.0e-15, tf.float64))
        ).numpy()
    )


def arithmetic_precision_consensus(precisions: Sequence[Any]) -> np.ndarray:
    """Return the unweighted arithmetic consensus without projection or shrinkage."""

    matrices = tuple(_symmetric_matrix(value, "precision") for value in precisions)
    if len(matrices) < 2:
        raise ValueError("at least two precision matrices are required")
    if any(matrix.shape != matrices[0].shape for matrix in matrices[1:]):
        raise ValueError("precision matrices must have matching shapes")
    consensus = tf.reduce_mean(tf.stack(matrices, axis=0), axis=0)
    return consensus.numpy()


def compare_dense_directional_precision_stability(
    left: DenseDirectionalScoreGeometryResult,
    right: DenseDirectionalScoreGeometryResult,
    *,
    config: DenseDirectionalScoreGeometryConfig | None = None,
) -> Mapping[str, Any]:
    """Compare two accepted raw-SPD symmetric candidates and apply fixed gates."""

    cfg = DenseDirectionalScoreGeometryConfig() if config is None else config
    if (
        not left.valid
        or not right.valid
        or not left.accepted
        or not right.accepted
        or left.central_symmetric_precision is None
        or right.central_symmetric_precision is None
        or left.diagnostics.get("central_symmetric_raw_spd") is not True
        or right.diagnostics.get("central_symmetric_raw_spd") is not True
    ):
        raise ValueError("stability comparison requires two accepted raw-SPD candidates")
    dimension = int(left.central_symmetric_precision.shape[0])
    rank = min(cfg.principal_subspace_rank, dimension)
    metrics = dict(
        compare_precision_geometry(
            left.central_symmetric_precision,
            right.central_symmetric_precision,
            subspace_rank=rank,
        )
    )
    generalized = metrics.get("generalized_eigenvalues")
    checks = {
        "generalized_eigenvalue_spread": bool(
            isinstance(generalized, Mapping)
            and float(generalized["spread"])
            <= cfg.generalized_eigenvalue_spread_cap
        ),
        "trace_normalized_frobenius": bool(
            float(metrics["trace_normalized_frobenius"])
            <= cfg.trace_normalized_frobenius_cap
        ),
        "trace_normalized_operator": bool(
            float(metrics["trace_normalized_operator"])
            <= cfg.trace_normalized_operator_cap
        ),
        "principal_angle_degrees": bool(
            metrics["maximum_principal_angle_degrees"] is not None
            and float(metrics["maximum_principal_angle_degrees"])
            <= cfg.principal_angle_degrees_cap
        ),
    }
    return _json_ready(
        {
            "passed": all(checks.values()),
            "principal_subspace_rank": rank,
            "checks": checks,
            "metrics": metrics,
        }
    )


def _antithetic_matrix(value: Any, dimension: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, tf.float64)
    if tensor.shape.rank != 2 or tensor.shape[0] is None or tensor.shape[1] is None:
        raise ValueError(f"{name} must have static shape [2 * dimension, dimension]")
    if tensor.shape != (2 * dimension, dimension):
        raise ValueError(f"{name} must have shape [2 * dimension, dimension]")
    return tensor


def _validate_antithetic_frame(
    offsets: tf.Tensor,
    *,
    dimension: int,
    antithetic_absolute_tolerance: float,
    orthogonality_max_abs_cap: float,
    reject_orthogonality: bool,
) -> tuple[tf.Tensor, Mapping[str, Any]]:
    """Validate the shared square central-direction frame contract."""

    pair_tolerance = float(antithetic_absolute_tolerance)
    orthogonality_cap = float(orthogonality_max_abs_cap)
    if (
        not np.isfinite(pair_tolerance)
        or pair_tolerance <= 0.0
        or not np.isfinite(orthogonality_cap)
        or orthogonality_cap <= 0.0
    ):
        raise ValueError("frame tolerances must be positive and finite")
    positive = offsets[0::2]
    negative = offsets[1::2]
    pair_error = float(tf.reduce_max(tf.abs(positive + negative)).numpy())
    if pair_error > pair_tolerance:
        raise ValueError("offset rows must be exactly interleaved antithetic pairs")

    singular = tf.linalg.svd(positive, compute_uv=False)
    tolerance = (
        tf.reduce_max(singular)
        * tf.cast(tf.shape(positive)[0], tf.float64)
        * tf.constant(np.finfo(np.float64).eps, tf.float64)
    )
    rank = int(tf.reduce_sum(tf.cast(singular > tolerance, tf.int32)).numpy())
    if rank != dimension:
        raise ValueError("positive directional offsets must have full rank")

    radii = tf.linalg.norm(positive, axis=1)
    radius = tf.reduce_mean(radii)
    if not bool((radius > 0.0).numpy()):
        raise ValueError("directional radius must be positive")
    radius_spread = float(tf.reduce_max(tf.abs(radii - radius)).numpy())
    if radius_spread > pair_tolerance:
        raise ValueError("positive directional offsets must have one common radius")
    normalized = positive / radius
    orthogonality_error = float(
        tf.reduce_max(
            tf.abs(
                tf.matmul(normalized, normalized, transpose_b=True)
                - tf.eye(dimension, dtype=tf.float64)
            )
        ).numpy()
    )
    if reject_orthogonality and orthogonality_error > orthogonality_cap:
        raise ValueError("positive directional offsets must be orthogonal")
    return positive, {
        "offset_rank": rank,
        "radius": float(radius.numpy()),
        "radius_max_abs_spread": radius_spread,
        "antithetic_max_abs_error": pair_error,
        "orthogonality_max_abs_error": orthogonality_error,
    }


def _square_matrix(value: Any, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, tf.float64)
    if (
        tensor.shape.rank != 2
        or tensor.shape[0] is None
        or tensor.shape[1] is None
        or tensor.shape[0] != tensor.shape[1]
    ):
        raise ValueError(f"{name} must be a static square matrix")
    return tensor


def _symmetric_matrix(value: Any, name: str) -> tf.Tensor:
    """Require a finite symmetric matrix without projecting an invalid input."""

    tensor = _square_matrix(value, name)
    if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
        raise ValueError(f"{name} must be finite")
    symmetry_error = float(
        tf.reduce_max(tf.abs(tensor - tf.transpose(tensor))).numpy()
    )
    scale = max(float(tf.reduce_max(tf.abs(tensor)).numpy()), 1.0)
    if symmetry_error > 1.0e-12 + 1.0e-10 * scale:
        raise ValueError(f"{name} must be symmetric")
    return tensor


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "DENSE_DIRECTIONAL_GEOMETRY_NONCLAIMS",
    "DenseDirectionalScoreGeometryConfig",
    "DenseDirectionalScoreGeometryResult",
    "arithmetic_precision_consensus",
    "compare_dense_directional_precision_stability",
    "directional_prediction_relative_frobenius",
    "fit_dense_directional_score_geometry",
]
