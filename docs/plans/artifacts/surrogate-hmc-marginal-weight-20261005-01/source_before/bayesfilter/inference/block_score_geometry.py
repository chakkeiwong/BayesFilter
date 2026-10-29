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

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from bayesfilter.inference import block_score_geometry_tf as native
from bayesfilter.ops.host_tensor_io import numeric_tensor

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
            if not math.isfinite(value) or value <= 0.0:
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
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class BlockScoreGeometryResult:
    """Qualified consensus or fail-closed diagnostics for one fixed family."""

    accepted: bool
    status: str
    blocks: tuple[ScoreGeometryBlock, ...]
    precision_z: tf.Tensor | None
    covariance_z: tf.Tensor | None
    diagnostics: Mapping[str, Any]
    nonclaims: tuple[str, ...] = BLOCK_SCORE_GEOMETRY_NONCLAIMS

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        for name in ("precision_z", "covariance_z"):
            value = getattr(self, name)
            if value is not None:
                array = numeric_tensor(value, tf.float64)
                object.__setattr__(self, name, array)
        object.__setattr__(self, "diagnostics", _json_ready(dict(self.diagnostics)))

    def position_geometry(self, scale: Any) -> Mapping[str, tf.Tensor]:
        """Convert qualified standardized geometry to parameter coordinates."""

        if not self.accepted or self.precision_z is None or self.covariance_z is None:
            raise ValueError("block score geometry is not qualified")
        scale_tf = tf.reshape(tf.convert_to_tensor(scale, tf.float64), [-1])
        dimension = int(self.precision_z.shape[0])
        if scale_tf.shape != (dimension,):
            raise ValueError("scale shape must match geometry dimension")
        precision, covariance, factor, valid = native.position_program(dimension)(
            self.precision_z, self.covariance_z, scale_tf)
        if not bool(valid):
            raise ValueError("scale must be positive and finite")
        return {
            "precision": precision,
            "covariance": covariance,
            "factor": factor,
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
    training_scores = _replicate_tensor(
        training_scores_z, dimension, "training_scores_z"
    )
    selection_z = _replicate_tensor(
        selection_offsets_z, dimension, "selection_offsets_z"
    )
    selection_scores = _replicate_tensor(
        selection_scores_z, dimension, "selection_scores_z"
    )
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
        raise ValueError(
            "at least two matching training/selection replicates are required"
        )
    program = native.fit_program(dimension, replicate_count, int(training_z.shape[1]),
        int(selection_z.shape[1]), int(audit_z.shape[0]),
        tuple((block.start, block.stop) for block in declared_blocks))
    result = program(center, training_z, training_scores, selection_z, selection_scores,
        audit_z, audit_scores,
        tf.constant([getattr(cfg, name) for name in native.CONTROL_FIELDS], tf.float64),
        tf.constant(cfg.principal_subspace_rank, tf.int32))
    # Bulk transport of completed diagnostics; no host numerical decisions.
    report = {name: result[name].numpy().tolist() for name in (
        "status", "replicate_count", "block_counts", "blocks", "selection", "pairs",
        "checks", "pair_passed", "stability_passed", "summary")}
    status_code = report["status"]
    if status_code == 10:
        raise ValueError("subspace_rank must lie in [1, dimension]")
    status = native.STATUSES[status_code]
    if status_code == 1:
        return _rejected(status, declared_blocks)
    replicates = _replicate_reports(report, declared_blocks)
    if status_code in (2, 3, 4):
        return _rejected(status, declared_blocks, {"replicates": replicates})
    summary = report["summary"]
    diagnostics = {
        "family": "block_diagonal_symmetric_score_regression",
        "dimension": dimension,
        "block_count": len(declared_blocks),
        "within_block_symmetric_coefficient_count": sum(
            block.size * (block.size + 1) // 2 for block in declared_blocks),
        "training_rows_per_replicate": int(training_z.shape[1]),
        "selection_rows_per_replicate": int(selection_z.shape[1]),
        "audit_rows": int(audit_z.shape[0]),
        "replicates": replicates,
        "stability": _stability_report(report, cfg, dimension, replicate_count),
        "consensus_selection_relative_rmse": summary[0],
        "consensus_selection_unexplained_response_fraction": summary[1],
        "audit_relative_rmse": summary[2],
        "audit_unexplained_response_fraction": summary[3],
        "precision_covariance_identity_max_abs": summary[4],
        "config": cfg.payload(),
        "audit_used_after_consensus": True,
        "dense_global_fit_used": False,
        "factor_fit_used": False,
        "fallback_used": False,
    }
    return BlockScoreGeometryResult(
        accepted=status_code == 0, status=status, blocks=declared_blocks,
        precision_z=result["precision"], covariance_z=result["covariance"], diagnostics=diagnostics)


def _replicate_reports(report, blocks):
    """Decode the already ordered/truncated tensor reports into the public schema."""
    replicates = []
    for index in range(report["replicate_count"]):
        block_reports = []
        code = 0
        for block_index in range(report["block_counts"][index]):
            block = blocks[block_index]
            values = report["blocks"][index][block_index]
            code = int(values[6])
            block_report = {**block.payload(),
                "coefficient_count": block.size * (block.size + 1) // 2,
                "design_rank": int(values[0]), "offset_rank": int(values[1])}
            if code == 2:
                block_report["status"] = "rank_deficient"
            else:
                block_report.update({"raw_minimum_eigenvalue": values[2],
                    "raw_maximum_eigenvalue": values[3],
                    "raw_nonpositive_eigenvalue_count": int(values[4]),
                    "raw_condition_number": values[5],
                    "status": "usable" if code == 0 else "raw_not_spd_or_conditioned"})
            block_reports.append(block_report)
        record = {"replicate_index": index, "blocks": block_reports,
            "status": "usable" if code == 0 else native.STATUSES[code]}
        if code == 0:
            record.update({"selection_relative_rmse": report["selection"][index][0],
                "selection_unexplained_response_fraction": report["selection"][index][1]})
        replicates.append(record)
    return replicates


def _stability_report(report, config, dimension, replicate_count):
    comparisons = []
    pair_index = 0
    for left in range(replicate_count):
        for right in range(left + 1, replicate_count):
            values = report["pairs"][pair_index]
            summary = values[3 * dimension:]
            rank = int(summary[0])
            metrics = {"left_raw_eigenvalues": values[:dimension],
                "right_raw_eigenvalues": values[dimension:2 * dimension],
                "left_nonpositive_count": int(summary[7]),
                "right_nonpositive_count": int(summary[8]),
                "trace_normalized_frobenius": summary[4],
                "trace_normalized_operator": summary[5],
                "positive_subspace_rank": rank,
                "principal_angles_degrees": values[2 * dimension:2 * dimension + rank],
                "maximum_principal_angle_degrees": summary[6] if rank else None,
                "generalized_eigenvalues": {"minimum": summary[1], "maximum": summary[2],
                    "spread": summary[3]} if bool(summary[9]) else None}
            checks = dict(zip(("generalized_eigenvalue_spread", "trace_normalized_frobenius",
                "trace_normalized_operator", "principal_angle_degrees"), report["checks"][pair_index], strict=True))
            comparisons.append({"left_replicate": left, "right_replicate": right,
                "passed": report["pair_passed"][pair_index], "checks": checks, "metrics": metrics})
            pair_index += 1
    return {"passed": report["stability_passed"],
        "principal_subspace_rank": min(config.principal_subspace_rank, dimension - 1),
        "comparisons": comparisons}


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
    if tf.is_tensor(value) or hasattr(value, "__array_interface__"):
        return numeric_tensor(value).numpy().tolist()
    return value


__all__ = [
    "BLOCK_SCORE_GEOMETRY_NONCLAIMS",
    "BlockScoreGeometryConfig",
    "BlockScoreGeometryResult",
    "ScoreGeometryBlock",
    "fit_block_diagonal_score_geometry",
]
