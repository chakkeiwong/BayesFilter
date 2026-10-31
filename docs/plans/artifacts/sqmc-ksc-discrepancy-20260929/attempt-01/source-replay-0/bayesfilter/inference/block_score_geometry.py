"""Block-native fixed-center score geometry for HMC initialization.

For standardized coordinates ``theta = c + diag(scale) z``, this module fits
the local precision model

``g_z(c) - g_z(c + z) ~= P_z z``.

Only the caller-declared diagonal blocks of ``P_z`` are estimated. The fit is
evaluated against the complete score response, so omitted cross-block curvature
is visible in the selection and audit residuals. This module neither searches
the block family nor moves the center.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import tensorflow as tf

from bayesfilter.inference.fixed_center_curvature import compare_precision_geometry


BLOCK_SCORE_GEOMETRY_NONCLAIMS = (
    "fixed-center block score geometry is an HMC initializer only",
    "center stationarity is not required or established",
    "not a MAP claim",
    "not a posterior covariance claim",
    "not final mass or tuning evidence",
    "not posterior convergence or default-readiness evidence",
)


@dataclass(frozen=True)
class ScoreGeometryBlock:
    """One contiguous, non-overlapping parameter-family block."""

    name: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        name = str(self.name)
        start = int(self.start)
        stop = int(self.stop)
        if not name:
            raise ValueError("block name must be nonempty")
        if start < 0 or stop <= start:
            raise ValueError("block bounds must satisfy 0 <= start < stop")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)

    @property
    def size(self) -> int:
        return self.stop - self.start

    def payload(self) -> Mapping[str, Any]:
        return {"name": self.name, "start": self.start, "stop": self.stop}


@dataclass(frozen=True)
class BlockScoreGeometryConfig:
    """Prospectively fixed numerical and qualification policy."""

    ridge: float = 1.0e-8
    max_condition_number: float = 1.0e8
    selection_relative_rmse_cap: float = 0.20
    audit_relative_rmse_cap: float = 0.20
    unexplained_response_fraction_cap: float = 0.20
    generalized_eigenvalue_spread_cap: float = 100.0
    trace_normalized_frobenius_cap: float = 0.01
    trace_normalized_operator_cap: float = 0.01
    principal_angle_degrees_cap: float = 5.0
    principal_subspace_rank: int = 64

    def __post_init__(self) -> None:
        for name in (
            "ridge",
            "max_condition_number",
            "selection_relative_rmse_cap",
            "audit_relative_rmse_cap",
            "unexplained_response_fraction_cap",
            "generalized_eigenvalue_spread_cap",
            "trace_normalized_frobenius_cap",
            "trace_normalized_operator_cap",
            "principal_angle_degrees_cap",
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
class BlockScoreGeometryResult:
    """Qualified consensus or fail-closed diagnostics for one fixed family."""

    accepted: bool
    status: str
    blocks: tuple[ScoreGeometryBlock, ...]
    precision_z: np.ndarray | None
    covariance_z: np.ndarray | None
    diagnostics: Mapping[str, Any]
    nonclaims: tuple[str, ...] = BLOCK_SCORE_GEOMETRY_NONCLAIMS

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        for name in ("precision_z", "covariance_z"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=np.float64).copy()
                array.setflags(write=False)
                object.__setattr__(self, name, array)
        object.__setattr__(self, "diagnostics", _json_ready(dict(self.diagnostics)))

    def position_geometry(self, scale: Any) -> Mapping[str, np.ndarray]:
        """Convert qualified standardized geometry to parameter coordinates."""

        if not self.accepted or self.precision_z is None or self.covariance_z is None:
            raise ValueError("block score geometry is not qualified")
        scale_tf = tf.reshape(tf.convert_to_tensor(scale, tf.float64), [-1])
        dimension = int(self.precision_z.shape[0])
        if scale_tf.shape != (dimension,):
            raise ValueError("scale shape must match geometry dimension")
        if not bool(tf.reduce_all(tf.math.is_finite(scale_tf) & (scale_tf > 0.0)).numpy()):
            raise ValueError("scale must be positive and finite")
        scale_matrix = tf.linalg.diag(scale_tf)
        inverse_scale = tf.linalg.diag(1.0 / scale_tf)
        covariance = tf.matmul(
            tf.matmul(scale_matrix, tf.convert_to_tensor(self.covariance_z, tf.float64)),
            scale_matrix,
        )
        precision = tf.matmul(
            tf.matmul(inverse_scale, tf.convert_to_tensor(self.precision_z, tf.float64)),
            inverse_scale,
        )
        factor = tf.linalg.cholesky(covariance)
        return {
            "precision": precision.numpy(),
            "covariance": covariance.numpy(),
            "factor": factor.numpy(),
        }

    def payload(self, *, include_matrices: bool = False) -> Mapping[str, Any]:
        payload = {
            "schema": "bayesfilter.block_score_geometry.v1",
            "accepted": self.accepted,
            "status": self.status,
            "blocks": [block.payload() for block in self.blocks],
            "diagnostics": self.diagnostics,
            "nonclaims": self.nonclaims,
        }
        if include_matrices:
            payload.update(
                {"precision_z": self.precision_z, "covariance_z": self.covariance_z}
            )
        return _json_ready(payload)


def fit_block_diagonal_score_geometry(
    *,
    center_score_z: Any,
    training_offsets_z: Any,
    training_scores_z: Any,
    selection_offsets_z: Any,
    selection_scores_z: Any,
    audit_offsets_z: Any,
    audit_scores_z: Any,
    blocks: Sequence[ScoreGeometryBlock],
    config: BlockScoreGeometryConfig | None = None,
) -> BlockScoreGeometryResult:
    """Fit one predeclared block family and qualify it without fallback."""

    cfg = BlockScoreGeometryConfig() if config is None else config
    center = tf.reshape(tf.convert_to_tensor(center_score_z, tf.float64), [-1])
    if center.shape[0] is None:
        raise ValueError("center_score_z must have static dimension")
    dimension = int(center.shape[0])
    declared_blocks = _validate_blocks(blocks, dimension)
    training_z = _replicate_tensor(training_offsets_z, dimension, "training_offsets_z")
    training_scores = _replicate_tensor(training_scores_z, dimension, "training_scores_z")
    selection_z = _replicate_tensor(selection_offsets_z, dimension, "selection_offsets_z")
    selection_scores = _replicate_tensor(selection_scores_z, dimension, "selection_scores_z")
    audit_z = _matrix_tensor(audit_offsets_z, dimension, "audit_offsets_z")
    audit_scores = _matrix_tensor(audit_scores_z, dimension, "audit_scores_z")
    if training_z.shape != training_scores.shape:
        raise ValueError("training offset and score shapes must match")
    if selection_z.shape != selection_scores.shape:
        raise ValueError("selection offset and score shapes must match")
    if audit_z.shape != audit_scores.shape:
        raise ValueError("audit offset and score shapes must match")
    replicate_count = int(training_z.shape[0])
    if replicate_count < 2 or int(selection_z.shape[0]) != replicate_count:
        raise ValueError("at least two matching training/selection replicates are required")
    finite_inputs = tf.reduce_all(
        tf.math.is_finite(
            tf.concat(
            (
                tf.reshape(center, [-1]),
                tf.reshape(training_z, [-1]),
                tf.reshape(training_scores, [-1]),
                tf.reshape(selection_z, [-1]),
                tf.reshape(selection_scores, [-1]),
                tf.reshape(audit_z, [-1]),
                tf.reshape(audit_scores, [-1]),
            ),
            axis=0,
            )
        )
    )
    if not bool(finite_inputs.numpy()):
        return _rejected("nonfinite_fit_inputs", declared_blocks)

    replicates: list[Mapping[str, Any]] = []
    replicate_precisions: list[tf.Tensor] = []
    for replicate_index in range(replicate_count):
        precision, block_reports, failure = _fit_replicate(
            center,
            training_z[replicate_index],
            training_scores[replicate_index],
            declared_blocks,
            cfg,
        )
        report: dict[str, Any] = {
            "replicate_index": replicate_index,
            "blocks": block_reports,
        }
        if failure is not None:
            report["status"] = failure
            replicates.append(report)
            return _rejected(failure, declared_blocks, {"replicates": replicates})
        selection_rmse, selection_unexplained = _score_diagnostics(
            precision,
            center,
            selection_z[replicate_index],
            selection_scores[replicate_index],
        )
        report.update(
            {
                "status": "usable",
                "selection_relative_rmse": selection_rmse,
                "selection_unexplained_response_fraction": selection_unexplained,
            }
        )
        replicates.append(report)
        replicate_precisions.append(precision)

    stability = _stability(replicate_precisions, cfg, dimension)
    consensus = tf.reduce_mean(tf.stack(replicate_precisions, axis=0), axis=0)
    selection_passed = all(
        report["selection_relative_rmse"] <= cfg.selection_relative_rmse_cap
        for report in replicates
    )
    consensus_selection = [
        _score_diagnostics(consensus, center, selection_z[index], selection_scores[index])
        for index in range(replicate_count)
    ]
    consensus_selection_rmse = float(
        tf.reduce_mean(tf.convert_to_tensor([item[0] for item in consensus_selection], tf.float64)).numpy()
    )
    consensus_selection_unexplained = float(
        tf.reduce_mean(tf.convert_to_tensor([item[1] for item in consensus_selection], tf.float64)).numpy()
    )
    audit_rmse, audit_unexplained = _score_diagnostics(
        consensus, center, audit_z, audit_scores
    )
    covariance = tf.linalg.cholesky_solve(
        tf.linalg.cholesky(consensus), tf.eye(dimension, dtype=tf.float64)
    )
    inverse_residual = float(
        tf.reduce_max(tf.abs(tf.matmul(consensus, covariance) - tf.eye(dimension, dtype=tf.float64))).numpy()
    )
    diagnostics = {
        "family": "block_diagonal_symmetric_score_regression",
        "dimension": dimension,
        "block_count": len(declared_blocks),
        "within_block_symmetric_coefficient_count": sum(
            block.size * (block.size + 1) // 2 for block in declared_blocks
        ),
        "training_rows_per_replicate": int(training_z.shape[1]),
        "selection_rows_per_replicate": int(selection_z.shape[1]),
        "audit_rows": int(audit_z.shape[0]),
        "replicates": replicates,
        "stability": stability,
        "consensus_selection_relative_rmse": consensus_selection_rmse,
        "consensus_selection_unexplained_response_fraction": consensus_selection_unexplained,
        "audit_relative_rmse": audit_rmse,
        "audit_unexplained_response_fraction": audit_unexplained,
        "precision_covariance_identity_max_abs": inverse_residual,
        "config": cfg.payload(),
        "audit_used_after_consensus": True,
        "dense_global_fit_used": False,
        "factor_fit_used": False,
        "fallback_used": False,
    }
    status = "qualified_for_hmc_initialization"
    if not selection_passed or consensus_selection_rmse > cfg.selection_relative_rmse_cap:
        status = "selection_score_fit_rejected"
    elif not bool(stability["passed"]):
        status = "replicate_stability_rejected"
    elif audit_rmse > cfg.audit_relative_rmse_cap:
        status = "audit_score_fit_rejected"
    elif audit_unexplained > cfg.unexplained_response_fraction_cap:
        status = "offblock_curvature_rejected"
    elif inverse_residual > 1.0e-8:
        status = "precision_covariance_identity_rejected"
    return BlockScoreGeometryResult(
        accepted=status == "qualified_for_hmc_initialization",
        status=status,
        blocks=declared_blocks,
        precision_z=consensus.numpy(),
        covariance_z=covariance.numpy(),
        diagnostics=diagnostics,
    )


def _fit_replicate(
    center: tf.Tensor,
    offsets: tf.Tensor,
    scores: tf.Tensor,
    blocks: tuple[ScoreGeometryBlock, ...],
    config: BlockScoreGeometryConfig,
) -> tuple[tf.Tensor, list[Mapping[str, Any]], str | None]:
    dimension = int(center.shape[0])
    precision = tf.zeros([dimension, dimension], tf.float64)
    reports: list[Mapping[str, Any]] = []
    response = center[None, :] - scores
    for block in blocks:
        z_block = offsets[:, block.start : block.stop]
        response_block = response[:, block.start : block.stop]
        design = _symmetric_score_design(z_block, block.size)
        coefficient_count = block.size * (block.size + 1) // 2
        flat_design = tf.reshape(design, [-1, coefficient_count])
        flat_response = tf.reshape(response_block, [-1, 1])
        singular = tf.linalg.svd(flat_design, compute_uv=False)
        tolerance = (
            tf.reduce_max(singular)
            * tf.cast(tf.shape(flat_design)[0], tf.float64)
            * tf.experimental.numpy.finfo(tf.float64.as_numpy_dtype).eps
        )
        rank = int(tf.reduce_sum(tf.cast(singular > tolerance, tf.int32)).numpy())
        row_singular = tf.linalg.svd(z_block, compute_uv=False)
        row_tolerance = (
            tf.reduce_max(row_singular)
            * tf.cast(tf.shape(z_block)[0], tf.float64)
            * tf.experimental.numpy.finfo(tf.float64.as_numpy_dtype).eps
        )
        row_rank = int(tf.reduce_sum(tf.cast(row_singular > row_tolerance, tf.int32)).numpy())
        if rank < coefficient_count or row_rank < block.size:
            reports.append(
                {
                    **block.payload(),
                    "coefficient_count": coefficient_count,
                    "design_rank": rank,
                    "offset_rank": row_rank,
                    "status": "rank_deficient",
                }
            )
            return precision, reports, "block_design_rank_deficient"
        ridge = tf.sqrt(tf.constant(config.ridge, tf.float64)) * tf.eye(
            coefficient_count, dtype=tf.float64
        )
        coefficients = tf.linalg.lstsq(
            tf.concat((flat_design, ridge), axis=0),
            tf.concat((flat_response, tf.zeros([coefficient_count, 1], tf.float64)), axis=0),
            fast=False,
        )[:, 0]
        block_precision = _unpack_symmetric(coefficients, block.size)
        eigenvalues = tf.linalg.eigvalsh(block_precision)
        minimum = float(tf.reduce_min(eigenvalues).numpy())
        maximum = float(tf.reduce_max(eigenvalues).numpy())
        nonpositive = int(tf.reduce_sum(tf.cast(eigenvalues <= 0.0, tf.int32)).numpy())
        condition = np.inf if minimum <= 0.0 else maximum / minimum
        reports.append(
            {
                **block.payload(),
                "coefficient_count": coefficient_count,
                "design_rank": rank,
                "offset_rank": row_rank,
                "raw_minimum_eigenvalue": minimum,
                "raw_maximum_eigenvalue": maximum,
                "raw_nonpositive_eigenvalue_count": nonpositive,
                "raw_condition_number": condition,
                "status": "usable"
                if nonpositive == 0 and condition <= config.max_condition_number
                else "raw_not_spd_or_conditioned",
            }
        )
        if nonpositive:
            return precision, reports, "raw_block_precision_not_spd"
        if not np.isfinite(condition) or condition > config.max_condition_number:
            return precision, reports, "block_condition_number_rejected"
        indices = tf.range(block.start, block.stop, dtype=tf.int32)
        grid = tf.stack(tf.meshgrid(indices, indices, indexing="ij"), axis=-1)
        precision = tf.tensor_scatter_nd_update(
            precision, tf.reshape(grid, [-1, 2]), tf.reshape(block_precision, [-1])
        )
    return precision, reports, None


def _score_diagnostics(
    precision: tf.Tensor,
    center: tf.Tensor,
    offsets: tf.Tensor,
    scores: tf.Tensor,
) -> tuple[float, float]:
    response = center[None, :] - scores
    prediction = tf.matmul(offsets, precision, transpose_b=True)
    residual = prediction - response
    response_scale = tf.maximum(
        tf.sqrt(tf.reduce_mean(tf.square(response))), tf.constant(1.0e-15, tf.float64)
    )
    relative_rmse = tf.sqrt(tf.reduce_mean(tf.square(residual))) / response_scale
    unexplained = tf.linalg.norm(residual) / tf.maximum(
        tf.linalg.norm(response), tf.constant(1.0e-15, tf.float64)
    )
    return float(relative_rmse.numpy()), float(unexplained.numpy())


def _stability(
    precisions: Sequence[tf.Tensor],
    config: BlockScoreGeometryConfig,
    dimension: int,
) -> Mapping[str, Any]:
    comparisons = []
    passed = True
    rank = min(config.principal_subspace_rank, dimension - 1)
    for left_index, left in enumerate(precisions):
        for right_index in range(left_index + 1, len(precisions)):
            metrics = dict(
                compare_precision_geometry(
                    left.numpy(), precisions[right_index].numpy(), subspace_rank=rank
                )
            )
            generalized = metrics["generalized_eigenvalues"]
            checks = {
                "generalized_eigenvalue_spread": bool(
                    generalized is not None
                    and generalized["spread"] <= config.generalized_eigenvalue_spread_cap
                ),
                "trace_normalized_frobenius": bool(
                    metrics["trace_normalized_frobenius"]
                    <= config.trace_normalized_frobenius_cap
                ),
                "trace_normalized_operator": bool(
                    metrics["trace_normalized_operator"]
                    <= config.trace_normalized_operator_cap
                ),
                "principal_angle_degrees": bool(
                    metrics["maximum_principal_angle_degrees"] is not None
                    and metrics["maximum_principal_angle_degrees"]
                    <= config.principal_angle_degrees_cap
                ),
            }
            pair_passed = all(checks.values())
            passed = passed and pair_passed
            comparisons.append(
                {
                    "left_replicate": left_index,
                    "right_replicate": right_index,
                    "passed": pair_passed,
                    "checks": checks,
                    "metrics": metrics,
                }
            )
    return {"passed": passed, "principal_subspace_rank": rank, "comparisons": comparisons}


def _validate_blocks(
    blocks: Sequence[ScoreGeometryBlock], dimension: int
) -> tuple[ScoreGeometryBlock, ...]:
    declared = tuple(blocks)
    if not declared:
        raise ValueError("at least one score geometry block is required")
    names = [block.name for block in declared]
    if len(set(names)) != len(names):
        raise ValueError("score geometry block names must be unique")
    cursor = 0
    for block in declared:
        if block.start != cursor:
            raise ValueError("score geometry blocks must be contiguous and ordered")
        cursor = block.stop
    if cursor != dimension:
        raise ValueError("score geometry blocks must cover the full dimension")
    return declared


def _replicate_tensor(value: Any, dimension: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, tf.float64)
    if tensor.shape.rank != 3 or tensor.shape[0] is None or tensor.shape[1] is None:
        raise ValueError(f"{name} must have static shape [replicate, row, dimension]")
    if tensor.shape[2] is None or int(tensor.shape[2]) != dimension:
        raise ValueError(f"{name} trailing dimension mismatch")
    return tensor


def _matrix_tensor(value: Any, dimension: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, tf.float64)
    if tensor.shape.rank != 2 or tensor.shape[0] is None:
        raise ValueError(f"{name} must have static shape [row, dimension]")
    if tensor.shape[1] is None or int(tensor.shape[1]) != dimension:
        raise ValueError(f"{name} trailing dimension mismatch")
    return tensor


def _symmetric_score_design(z: tf.Tensor, dimension: int) -> tf.Tensor:
    columns = []
    for row in range(dimension):
        for column in range(row, dimension):
            contribution = z[:, column, None] * tf.one_hot(
                row, dimension, dtype=tf.float64
            )[None, :]
            if row != column:
                contribution += z[:, row, None] * tf.one_hot(
                    column, dimension, dtype=tf.float64
                )[None, :]
            columns.append(contribution)
    return tf.stack(columns, axis=2)


def _unpack_symmetric(coefficients: tf.Tensor, dimension: int) -> tf.Tensor:
    matrix = tf.zeros([dimension, dimension], tf.float64)
    index = 0
    for row in range(dimension):
        for column in range(row, dimension):
            value = coefficients[index]
            matrix += tf.scatter_nd([[row, column]], [value], [dimension, dimension])
            if row != column:
                matrix += tf.scatter_nd([[column, row]], [value], [dimension, dimension])
            index += 1
    return matrix


def _rejected(
    status: str,
    blocks: tuple[ScoreGeometryBlock, ...],
    diagnostics: Mapping[str, Any] | None = None,
) -> BlockScoreGeometryResult:
    return BlockScoreGeometryResult(
        accepted=False,
        status=status,
        blocks=blocks,
        precision_z=None,
        covariance_z=None,
        diagnostics={} if diagnostics is None else diagnostics,
    )


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
    "BLOCK_SCORE_GEOMETRY_NONCLAIMS",
    "BlockScoreGeometryConfig",
    "BlockScoreGeometryResult",
    "ScoreGeometryBlock",
    "fit_block_diagonal_score_geometry",
]
