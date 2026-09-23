"""Fixed-center score-curvature stability and consensus diagnostics.

For standardized offsets ``z`` around a reviewed center ``c``, this module
fits and compares the local score model

``g_z(c) - g_z(c + z) ~= P_z z``.

Dense fits estimate the symmetric precision ``P_z`` directly. Structured fits
optimize an SPD factor covariance ``C_z`` and predict with ``P_z = C_z^{-1}``.
The fixed center need not be stationary, so no result from this module is a MAP
claim. A caller must supply disjoint training, selection-holdout, and audit
clouds; the audit cloud is evaluated only after candidate selection.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tensorflow as tf

from bayesfilter.inference import dense_partition_validation_tf as partition_validation
from bayesfilter.inference import fixed_center_fitting_tf as fitting_native
from bayesfilter.inference import fixed_center_selection_tf as selection_native
from bayesfilter.inference import fixed_center_stability_tf as stability_native
from bayesfilter.inference.factor_correlation_geometry import (
    FactorCorrelationGeometryConfig,
    fit_factor_correlation_score_geometry,
)
from bayesfilter.inference.hmc import PrecomputedMassArtifact
from bayesfilter.inference.mass_matrix_tf import _eigenpairs
from bayesfilter.inference.score_curvature_tf import fit_dense_score_precision_tf
from bayesfilter.ops.host_tensor_io import (
    buffer_byte_regions,
    buffer_regions_overlap,
    canonical_numeric_bytes,
    numeric_tensor,
)

FIXED_CENTER_CURVATURE_NONCLAIMS = (
    "fixed-center local score-curvature diagnostic only",
    "center stationarity is not required or established",
    "not a MAP claim",
    "not a posterior covariance claim",
    "not HMC readiness or convergence evidence",
    "not default-readiness evidence",
)


@dataclass(frozen=True)
class FixedCenterCurvatureThresholds:
    """Caller-reviewed gates; omitted stability caps imply diagnostic-only output."""

    selection_holdout_relative_rmse_cap: float
    audit_relative_rmse_cap: float
    projection_relative_frobenius_cap: float
    generalized_eigenvalue_spread_cap: float | None = None
    trace_normalized_frobenius_cap: float | None = None
    trace_normalized_operator_cap: float | None = None
    principal_angle_degrees_cap: float | None = None
    principal_subspace_rank: int | None = None
    require_raw_spd: bool = True

    def __post_init__(self) -> None:
        for name in (
            "selection_holdout_relative_rmse_cap",
            "audit_relative_rmse_cap",
            "projection_relative_frobenius_cap",
        ):
            _positive_finite(getattr(self, name), name)
        for name in (
            "generalized_eigenvalue_spread_cap",
            "trace_normalized_frobenius_cap",
            "trace_normalized_operator_cap",
            "principal_angle_degrees_cap",
        ):
            value = getattr(self, name)
            if value is not None:
                _positive_finite(value, name)
                object.__setattr__(self, name, float(value))
        if (
            self.generalized_eigenvalue_spread_cap is not None
            and self.generalized_eigenvalue_spread_cap < 1.0
        ):
            raise ValueError("generalized_eigenvalue_spread_cap must be at least one")
        if (
            self.principal_angle_degrees_cap is not None
            and self.principal_angle_degrees_cap > 90.0
        ):
            raise ValueError("principal_angle_degrees_cap must not exceed 90")
        if self.principal_subspace_rank is not None:
            rank = int(self.principal_subspace_rank)
            if rank <= 0:
                raise ValueError("principal_subspace_rank must be positive")
            object.__setattr__(self, "principal_subspace_rank", rank)
        if (self.principal_angle_degrees_cap is None) != (
            self.principal_subspace_rank is None
        ):
            raise ValueError(
                "principal angle cap and principal_subspace_rank must be set together"
            )
        object.__setattr__(self, "require_raw_spd", bool(self.require_raw_spd))

    @property
    def stability_caps_complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.generalized_eigenvalue_spread_cap,
                self.trace_normalized_frobenius_cap,
                self.trace_normalized_operator_cap,
                self.principal_angle_degrees_cap,
                self.principal_subspace_rank,
            )
        )


@dataclass(frozen=True)
class FixedCenterCurvatureFit:
    """One dense or structured precision estimate at an unchanged center."""

    family: str
    replicate_index: int
    factor_count: int | None
    accepted: bool
    status: str
    raw_precision_z: tf.Tensor | None
    precision_z: tf.Tensor | None
    covariance_z: tf.Tensor | None
    raw_eigenvalues: tf.Tensor | None
    raw_nonpositive_count: int | None
    projection_relative_frobenius: float | None
    selection_holdout_relative_rmse: float | None
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", str(self.family))
        object.__setattr__(self, "replicate_index", int(self.replicate_index))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "accepted", bool(self.accepted))
        if self.factor_count is not None:
            object.__setattr__(self, "factor_count", int(self.factor_count))
        for name in (
            "raw_precision_z",
            "precision_z",
            "covariance_z",
            "raw_eigenvalues",
        ):
            value = getattr(self, name)
            if value is not None:
                array = numeric_tensor(value, tf.float64)
                object.__setattr__(self, name, array)
        object.__setattr__(self, "diagnostics", _json_ready(dict(self.diagnostics)))

    def payload(self) -> Mapping[str, Any]:
        return _json_ready(
            {
                "family": self.family,
                "replicate_index": self.replicate_index,
                "factor_count": self.factor_count,
                "accepted": self.accepted,
                "status": self.status,
                "raw_precision_z": self.raw_precision_z,
                "precision_z": self.precision_z,
                "covariance_z": self.covariance_z,
                "raw_eigenvalues": self.raw_eigenvalues,
                "raw_nonpositive_count": self.raw_nonpositive_count,
                "projection_relative_frobenius": self.projection_relative_frobenius,
                "selection_holdout_relative_rmse": self.selection_holdout_relative_rmse,
                "diagnostics": self.diagnostics,
            }
        )


@dataclass(frozen=True)
class FixedCenterCurvatureResult:
    """Selected fixed-center precision plus evidence and fail-closed status."""

    accepted: bool
    status: str
    center: tf.Tensor
    center_score_z: tf.Tensor
    selected_family: str | None
    selected_precision_z: tf.Tensor | None
    selected_covariance_z: tf.Tensor | None
    audit_relative_rmse: float | None
    fits: tuple[FixedCenterCurvatureFit, ...]
    diagnostics: Mapping[str, Any]
    nonclaims: tuple[str, ...] = FIXED_CENTER_CURVATURE_NONCLAIMS

    def __post_init__(self) -> None:
        for name in ("center", "center_score_z"):
            array = numeric_tensor(getattr(self, name), tf.float64)
            object.__setattr__(self, name, array)
        for name in ("selected_precision_z", "selected_covariance_z"):
            value = getattr(self, name)
            if value is not None:
                array = numeric_tensor(value, tf.float64)
                object.__setattr__(self, name, array)
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "fits", tuple(self.fits))
        object.__setattr__(self, "diagnostics", _json_ready(dict(self.diagnostics)))

    def payload(self) -> Mapping[str, Any]:
        return _json_ready(
            {
                "schema": "bayesfilter.fixed_center_curvature.v1",
                "accepted": self.accepted,
                "status": self.status,
                "center": self.center,
                "center_score_z": self.center_score_z,
                "selected_family": self.selected_family,
                "selected_precision_z": self.selected_precision_z,
                "selected_covariance_z": self.selected_covariance_z,
                "audit_relative_rmse": self.audit_relative_rmse,
                "fits": [fit.payload() for fit in self.fits],
                "diagnostics": self.diagnostics,
                "nonclaims": self.nonclaims,
            }
        )

    def build_mass_artifact(
        self,
        *,
        scale: Any,
        adapter_signature: str,
        covariance_source: str = "fixed_center_consensus_curvature",
    ) -> PrecomputedMassArtifact:
        """Build a diagnostic-center mass artifact only after every gate passes."""

        if not self.accepted or self.status != "eligible_for_exact_hmc_canary":
            raise ValueError("fixed-center curvature is not eligible for HMC handoff")
        scale_array = numeric_tensor(scale, tf.float64)
        if scale_array.shape != self.center.shape or not bool(
            tf.reduce_all(tf.math.is_finite(scale_array) & (scale_array > 0.0))
        ):
            raise ValueError("scale must be positive finite with center shape")
        covariance_z = tf.convert_to_tensor(self.selected_covariance_z, tf.float64)
        scale_tf = tf.convert_to_tensor(scale_array, tf.float64)
        covariance_theta = _run_kernel(_scale_covariance, covariance_z, scale_tf)
        lineage = self.diagnostics.get("lineage", {})
        return PrecomputedMassArtifact.from_covariance(
            position=self.center,
            covariance=covariance_theta,
            adapter_signature=adapter_signature,
            position_role="diagnostic_center",
            covariance_source=covariance_source,
            source="fixed_center_curvature_handoff",
            jitter=0.0,
            regularization_report={
                "method": "fixed_center_curvature_selected_precision",
                "position_role": "diagnostic_center",
                "selected_family": self.selected_family,
                "audit_relative_rmse": self.audit_relative_rmse,
                "lineage": lineage,
                "nonclaims": list(self.nonclaims),
            },
        )


def fit_fixed_center_curvature(
    center: Any,
    center_score_z: Any,
    training_offsets_z: Any,
    training_scores_z: Any,
    selection_offsets_z: Any,
    selection_scores_z: Any,
    audit_offsets_z: Any,
    audit_scores_z: Any,
    *,
    thresholds: FixedCenterCurvatureThresholds,
    factor_max: int = 2,
    dense_eigenvalue_floor: float = 1.0e-8,
    max_condition_number: float = 1.0e8,
    shrinkage_weights: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    structured_target_family: str | None = None,
    lineage: Mapping[str, Any] | None = None,
) -> FixedCenterCurvatureResult:
    """Fit and select geometry without moving or stationarity-gating ``center``.

    Candidate fitting and selection use only training and selection-holdout
    rows. The audit rows are touched once after the family and shrinkage weight
    have been selected. Partition arrays must neither share memory nor contain
    exact copied offset rows; deterministic generators should also record
    disjoint seeds in ``lineage``.
    """

    fixed_center = _vector(center, "center")
    center_score = _vector(center_score_z, "center_score_z")
    if center_score.shape != fixed_center.shape:
        raise ValueError("center and center_score_z must have matching shape")
    dimension = int(fixed_center.shape[0])
    train_partitions = _cloud_partitions(
        training_offsets_z, training_scores_z, dimension, "training"
    )
    select_partitions = _cloud_partitions(
        selection_offsets_z, selection_scores_z, dimension, "selection"
    )
    audit_z, audit_scores = _cloud_pair(
        audit_offsets_z, audit_scores_z, dimension, "audit"
    )
    if len(train_partitions) != len(select_partitions):
        raise ValueError("training and selection must have the same replicate count")
    if len(train_partitions) < 2:
        raise ValueError("at least two independent fit replicates are required")
    for (train_z, _train_scores), (select_z, _select_scores) in zip(
        train_partitions, select_partitions, strict=True
    ):
        if int(train_z.shape[0]) + int(select_z.shape[0]) < 4 * dimension:
            raise ValueError(
                "each training plus selection replicate must total at least 4N"
            )
    named_partition_offsets = [
        *(
            (f"training[{index}]", offsets)
            for index, (offsets, _scores) in enumerate(train_partitions)
        ),
        *(
            (f"selection[{index}]", offsets)
            for index, (offsets, _scores) in enumerate(select_partitions)
        ),
        ("audit", audit_z),
    ]
    raw_regions = []
    for name, value in (
        ("training", training_offsets_z),
        ("selection", selection_offsets_z),
        ("audit", audit_offsets_z),
    ):
        shape = getattr(value, "shape", ())
        split = len(shape) == 3
        for index, regions in enumerate(
            buffer_byte_regions(value, split_first_axis=split)
        ):
            raw_regions.append((f"{name}[{index}]" if split else name, regions))
    _require_independent_partitions(named_partition_offsets, raw_regions=raw_regions)
    if int(audit_z.shape[0]) < 2 * dimension:
        raise ValueError("audit rows must total at least 2N")
    if factor_max not in (1, 2):
        raise ValueError("factor_max must be one or two")
    if structured_target_family not in (None, "factor_1", "factor_2"):
        raise ValueError("structured_target_family must be factor_1 or factor_2")
    if structured_target_family == "factor_2" and factor_max != 2:
        raise ValueError("factor_2 structured target requires factor_max=2")
    weights = _shrinkage_weights(shrinkage_weights)

    caps = tuple(getattr(thresholds, name) for name in stability_native.CAP_NAMES)
    replicates = len(train_partitions)
    training_rows, selection_rows = int(train_partitions[0][0].shape[0]), int(select_partitions[0][0].shape[0])
    program = fitting_native.fit_program(_dense_fit_kernel, fitting_native._structured_fit, _precision_geometry_kernel, _score_error_kernel,
        dimension, replicates, training_rows, selection_rows, int(audit_z.shape[0]), factor_max,
        max_condition_number, thresholds.selection_holdout_relative_rmse_cap,
        structured_target_family, len(weights))
    result = program(center_score, tf.stack(tuple(rows for rows, _ in train_partitions)),
        tf.stack(tuple(scores for _, scores in train_partitions)), tf.stack(tuple(rows for rows, _ in select_partitions)),
        tf.stack(tuple(scores for _, scores in select_partitions)), audit_z, audit_scores,
        tf.constant([0. if cap is None else cap for cap in caps], tf.float64),
        tf.constant([cap is not None for cap in caps], tf.bool),
        tf.constant(dimension if thresholds.principal_subspace_rank is None else thresholds.principal_subspace_rank, tf.int32),
        tf.constant(weights, tf.float64), tf.constant(dense_eigenvalue_floor, tf.float64),
        tf.constant(thresholds.projection_relative_frobenius_cap, tf.float64), tf.constant(thresholds.require_raw_spd),
        tf.constant(thresholds.audit_relative_rmse_cap, tf.float64))
    return _fixed_center_result_from_native(result, fixed_center, center_score,
        dimension=dimension, replicates=replicates, training_rows=training_rows,
        selection_rows=selection_rows, audit_rows=int(audit_z.shape[0]), thresholds=thresholds,
        factor_max=factor_max, weights=weights, structured_target_family=structured_target_family,
        lineage=lineage)


def _fixed_center_result_from_native(result, fixed_center, center_score, *, dimension,
        replicates, training_rows, selection_rows, audit_rows, thresholds, factor_max,
        weights, structured_target_family, lineage=None):
    """Format completed fit tensors without evaluating or selecting geometry."""
    report = tf.nest.map_structure(lambda value: value.numpy().tolist(), result)
    families = ("dense", "factor_1", "factor_2") if report["two_attempted"] else ("dense", "factor_1")
    groups = tuple(tuple(_native_fit_record(family, index,
        {name: values[family_index][index] for name, values in report["fits"].items()},
        dimension, training_rows, selection_rows, thresholds.selection_holdout_relative_rmse_cap)
        for index in range(replicates)) for family_index, family in enumerate(families))
    fits = [fit for pair in zip(groups[0], groups[1], strict=True) for fit in pair]
    if report["two_attempted"]:
        fits.extend(groups[2])
    # Raise the original one-factor stability error before selector errors.
    one_report = _stability_report(groups[1], thresholds, report["one_stability"])
    selected, selection = _selection_report(families, groups, thresholds, weights, structured_target_family,
        result["selection"], report["selection"])
    selection["factor_escalation"] = {
        "one_factor_passed_all_replicates": report["one_passed"],
        "one_factor_stability_passed": one_report["passed"],
        "two_factor_attempted": report["two_attempted"],
        "reason": "explicit_factor_2_target" if structured_target_family == "factor_2"
            else "one_factor_fit_holdout_or_stability_rejected" if report["two_attempted"]
            else "one_factor_fit_holdout_and_stability_passed" if factor_max == 2 else "factor_max_one",
    }
    if selected is None:
        return _blocked_result(fixed_center, center_score, fits, "geometry_readiness_blocked",
            lineage=lineage, extra={"selection": selection})
    audit_error = report["audit_error"]
    selection.update(audit_relative_rmse=audit_error, audit_row_count=audit_rows,
        audit_used_after_selection=True, audit_changed_selection=False)
    return FixedCenterCurvatureResult(accepted=report["status"] == 1,
        status=fitting_native.RESULT_STATUSES[report["status"]], center=fixed_center, center_score_z=center_score,
        selected_family=selected["family"], selected_precision_z=result["selection"]["precision"],
        selected_covariance_z=result["covariance"], audit_relative_rmse=audit_error, fits=tuple(fits),
        diagnostics={"center_score_role": "explanatory_only", "center_stationarity_required": False,
            "partition_contract": {"replicate_count": replicates,
                "training_rows_per_replicate": [training_rows] * replicates,
                "selection_rows_per_replicate": [selection_rows] * replicates,
                "audit_rows": audit_rows, "audit_used_after_selection": True,
                "offset_overlap_check": "shared_memory_and_exact_float64_rows"},
            "thresholds_complete": thresholds.stability_caps_complete,
            "diagonal_only_cannot_be_automatically_eligible": report["selection"]["diagonal_only"],
            "selection": selection, "lineage": {} if lineage is None else dict(lineage)})


def _native_fit_record(family, index, row, dimension, training_rows, selection_rows, holdout_cap):
    """Materialize the unchanged public schema from a completed native fit."""
    present, admissible, accepted = row["flags"]
    factor_count = None if family == "dense" else int(family[-1])
    if factor_count is None:
        diagnostics = {"precision_parameterization": "direct_symmetric_least_squares",
            "geometry_admissible": admissible, "selection_holdout_passed": row["holdout_passed"]}
    else:
        count = 2 * dimension if factor_count == 1 else 3 * dimension - 1
        domain_violations = row["invalid_covariance_evaluations"]
        anchors = row["anchors"][:factor_count] if present or domain_violations else ()
        if present:
            metrics = dict(zip(fitting_native.FACTOR_METRICS, row["factor_metrics"], strict=True))
            for name in ("prediction_jacobian_rank", "optimizer_iterations", "optimizer_objective_evaluations"):
                metrics[name] = int(metrics[name])
            for name in ("second_factor_identified", "optimizer_converged", "optimizer_failed"):
                metrics[name] = bool(metrics[name])
            if not metrics["prediction_jacobian_rank"]:
                metrics["prediction_jacobian_condition_number"] = None
            diagnostics = {**metrics, "jit_compile": True, "training_row_count": training_rows,
                "holdout_row_count": selection_rows, "training_score_equation_count": training_rows * dimension,
                "holdout_score_equation_count": selection_rows * dimension, "parameter_count": count,
                "holdout_score_relative_rmse_cap": holdout_cap, "covariance_eigenvalues": row["factor_eigenvalues"],
                "loading_row_squared_norms": row["loading_norms"],
                "covariance_parameterization": "D[diag(1-row_norm(L)^2)+LL^T]D",
                "score_model": "center_score_minus_local_score_equals_precision_times_offset"}
        elif domain_violations:
            diagnostics = {"exception_type": "InvalidArgumentError", "jit_compile": True,
                "invalid_covariance_evaluations": domain_violations,
                "failure_reason": "factor_covariance_domain_violation"}
        else:
            diagnostics = {"parameter_count": count, "symmetric_covariance_entry_count": dimension * (dimension + 1) // 2}
        diagnostics.update(covariance_parameterized_precision_prediction=True, parameter_count=count,
            anchor_indices=anchors, geometry_admissible=admissible, selection_holdout_passed=row["holdout_passed"])
    return FixedCenterCurvatureFit(family=family, replicate_index=index, factor_count=factor_count,
        accepted=accepted, status=fitting_native.FIT_STATUSES[row["status"]],
        raw_precision_z=row["raw"] if present else None, precision_z=row["precision"] if present else None,
        covariance_z=row["covariance"] if present else None, raw_eigenvalues=row["raw_values"] if present else None,
        raw_nonpositive_count=row["nonpositive"] if present else None,
        projection_relative_frobenius=row["projection"] if present else None,
        selection_holdout_relative_rmse=row["holdout"] if present else None, diagnostics=diagnostics)


def compare_precision_geometry(
    left: Any,
    right: Any,
    *,
    positive_subspace_tolerance: float = 0.0,
    subspace_rank: int | None = None,
) -> Mapping[str, Any]:
    """Return weak-direction-sensitive diagnostics for two symmetric matrices."""

    first = _symmetric_matrix(left, "left")
    second = _symmetric_matrix(right, "right")
    if first.shape != second.shape:
        raise ValueError("precision matrices must have matching shapes")
    dimension = int(first.shape[0])
    requested_rank = dimension if subspace_rank is None else int(subspace_rank)
    if requested_rank <= 0 or requested_rank > dimension:
        raise ValueError("subspace_rank must lie in [1, dimension]")
    values = _run_kernel(
        _precision_geometry_kernel,
        first,
        second,
        tf.constant(float(positive_subspace_tolerance), tf.float64),
        tf.constant(requested_rank, tf.int32),
    )
    first_values, second_values, rank, angles, generalized, frobenius, operator = values
    positive_rank = int(rank)
    principal_angles = angles.numpy().tolist()[:positive_rank]
    spd = bool(tf.reduce_all(first_values > 0.0) & tf.reduce_all(second_values > 0.0))
    return _json_ready(
        {
            "left_raw_eigenvalues": first_values,
            "right_raw_eigenvalues": second_values,
            "left_nonpositive_count": int(tf.math.count_nonzero(first_values <= 0.0)),
            "right_nonpositive_count": int(tf.math.count_nonzero(second_values <= 0.0)),
            "trace_normalized_frobenius": float(frobenius),
            "trace_normalized_operator": float(operator),
            "positive_subspace_rank": positive_rank,
            "principal_angles_degrees": principal_angles,
            "maximum_principal_angle_degrees": None
            if not principal_angles
            else max(principal_angles),
            "generalized_eigenvalues": None
            if not spd
            else {
                "minimum": float(generalized[0]),
                "maximum": float(generalized[1]),
                "spread": float(generalized[2]),
            },
        }
    )


def consensus_shrunk_precision(
    precisions: Sequence[Any],
    *,
    target: Any,
    weight: float,
) -> tf.Tensor:
    """Return ``(1-weight) * mean(precisions) + weight * target``."""

    if len(precisions) == 0:
        raise ValueError("at least one precision is required")
    try:
        matrices = numeric_tensor(precisions, tf.float64)
    except (ValueError, tf.errors.InvalidArgumentError) as exc:
        raise ValueError("precision matrices must have matching shapes") from exc
    if matrices.shape.rank != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("precision must be a finite symmetric square matrix")
    target_matrix = numeric_tensor(target, tf.float64)
    if target_matrix.shape.rank != 2 or target_matrix.shape[0] != target_matrix.shape[1]:
        raise ValueError("target must be a finite symmetric square matrix")
    if target_matrix.shape != matrices.shape[1:]:
        raise ValueError("target shape must match precisions")
    shrinkage = float(weight)
    if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("weight must be finite and in [0, 1]")
    candidate, precision_valid, target_valid, positive = _run_kernel(
        _checked_consensus_kernel,
        matrices,
        target_matrix,
        tf.convert_to_tensor(shrinkage, tf.float64),
    )
    if not bool(precision_valid):
        raise ValueError("precision must be a finite symmetric square matrix")
    if not bool(target_valid):
        raise ValueError("target must be a finite symmetric square matrix")
    if not bool(positive):
        raise ValueError("consensus shrinkage requires SPD inputs and target")
    return candidate


def _checked_consensus_kernel(precisions, target, weight):
    """Enclose the public endpoint's numerical validation and computation."""
    transposed = tf.linalg.matrix_transpose(precisions)
    target_transposed = tf.transpose(target)
    precision_valid = tf.reduce_all(tf.math.is_finite(precisions)) & tf.reduce_all(
        tf.abs(precisions - transposed) <= 1.0e-12 + 1.0e-10 * tf.abs(transposed))
    target_valid = tf.reduce_all(tf.math.is_finite(target)) & tf.reduce_all(
        tf.abs(target - target_transposed) <= 1.0e-12 + 1.0e-10 * tf.abs(target_transposed))
    candidate = consensus_shrunk_precision_tf(
        0.5 * (precisions + transposed), 0.5 * (target + target_transposed), weight)
    # Invalid inputs used to stop before the eigensystem. Preserve that order.
    positive = tf.cond(precision_valid & target_valid & tf.reduce_all(tf.math.is_finite(candidate)),
        lambda: tf.reduce_min(tf.linalg.eigvalsh(candidate)) > 0.0,
        lambda: tf.constant(False))
    return candidate, precision_valid, target_valid, positive


def consensus_shrunk_precision_tf(
    precisions: tf.Tensor,
    target: tf.Tensor,
    weight: tf.Tensor,
) -> tf.Tensor:
    """TensorFlow/XLA kernel for convex consensus shrinkage."""

    matrices = tf.convert_to_tensor(precisions, tf.float64)
    target_matrix = tf.convert_to_tensor(target, tf.float64)
    shrinkage = tf.convert_to_tensor(weight, tf.float64)
    consensus = tf.reduce_mean(matrices, axis=0)
    candidate = (1.0 - shrinkage) * consensus + shrinkage * target_matrix
    return 0.5 * (candidate + tf.transpose(candidate))


@lru_cache(maxsize=128)
def _compiled_kernel(kernel, signature):
    return tf.function(
        kernel, input_signature=signature, jit_compile=True, autograph=False
    )


def _run_kernel(kernel, *tensors):
    signature = tuple(tf.TensorSpec(value.shape, value.dtype) for value in tensors)
    return _compiled_kernel(kernel, signature)(*tensors)


def _scale_covariance(covariance, scale):
    return covariance * scale[:, None] * scale[None, :]


def _precision_eigenvalues(precision):
    return _eigenpairs(precision, True)[0]


def _precision_geometry_kernel(first, second, tolerance, requested_rank, *, jit_compile=True):
    # Principal angles depend on eigenvectors as well as eigenvalues. The
    # backend's loose default stopping check can leave O(1e-7) residuals.
    first_values, first_vectors = _eigenpairs(first, jit_compile)
    second_values, second_vectors = _eigenpairs(second, jit_compile)
    rank = tf.minimum(
        requested_rank,
        tf.minimum(
            tf.math.count_nonzero(first_values > tolerance, dtype=tf.int32),
            tf.math.count_nonzero(second_values > tolerance, dtype=tf.int32),
        ),
    )
    dimension = first.shape[0]
    # Keep the subspace matrix fixed-size for XLA. Its first `rank` singular
    # values are exactly those of the two selected positive eigenspaces.
    selected = tf.cast(tf.range(dimension) >= dimension - rank, first.dtype)
    overlap = tf.matmul(
        first_vectors * selected, second_vectors * selected, transpose_a=True
    )
    singular = tf.linalg.svd(overlap, compute_uv=False)
    singular = tf.where(
        tf.abs(1.0 - singular) <= 1.0e-12, tf.ones_like(singular), singular
    )
    angles = tf.acos(tf.clip_by_value(singular, -1.0, 1.0)) * tf.constant(
        180.0 / math.pi, tf.float64
    )
    spd = tf.reduce_all(first_values > 0.0) & tf.reduce_all(second_values > 0.0)

    def generalized():
        chol = tf.linalg.cholesky(first)
        solved = tf.linalg.triangular_solve(chol, second)
        transformed = tf.transpose(
            tf.linalg.triangular_solve(chol, tf.transpose(solved))
        )
        values, _ = _eigenpairs(0.5 * (transformed + tf.transpose(transformed)), jit_compile)
        minimum, maximum = tf.reduce_min(values), tf.reduce_max(values)
        return tf.stack((minimum, maximum, maximum / minimum))

    generalized_values = tf.cond(spd, generalized, lambda: tf.zeros([3], first.dtype))
    scale = tf.maximum(
        tf.maximum(tf.abs(tf.linalg.trace(first)), tf.abs(tf.linalg.trace(second))),
        tf.constant(1.0e-15, tf.float64),
    )
    difference = first - second
    return (
        first_values,
        second_values,
        rank,
        angles,
        generalized_values,
        tf.linalg.norm(difference) / scale,
        tf.reduce_max(tf.linalg.svd(difference, compute_uv=False)) / scale,
    )


def _score_error_kernel(precision, center_score, offsets, scores):
    response = center_score[None, :] - scores
    prediction = tf.matmul(offsets, precision, transpose_b=True)
    error = tf.sqrt(tf.reduce_mean(tf.square(prediction - response)))
    scale = tf.maximum(
        tf.sqrt(tf.reduce_mean(tf.square(response))), tf.constant(1.0e-15, tf.float64)
    )
    return error / scale


def _dense_fit_kernel(
    center, train, scores, selection, selection_scores, eigenvalue_floor, condition_cap, *, jit_compile=True
):
    raw = fit_dense_score_precision_tf(center, train, scores)["raw_precision"]
    values, vectors = _eigenpairs(raw, jit_compile)
    floor = tf.maximum(eigenvalue_floor, tf.reduce_max(tf.abs(values)) / condition_cap)
    projected = tf.matmul(
        vectors * tf.maximum(values, floor)[None, :], vectors, transpose_b=True
    )
    projection = tf.linalg.norm(projected - raw) / tf.maximum(
        tf.linalg.norm(raw), tf.constant(1.0e-15, tf.float64)
    )
    return (
        raw,
        projected,
        tf.linalg.inv(projected),
        values,
        projection,
        _score_error_kernel(projected, center, selection, selection_scores),
        tf.math.count_nonzero(values <= 0.0),
    )


def _fit_dense_precision(
    center_score: tf.Tensor,
    train_z: tf.Tensor,
    train_scores: tf.Tensor,
    select_z: tf.Tensor,
    select_scores: tf.Tensor,
    *,
    replicate_index: int,
    eigenvalue_floor: float,
    max_condition_number: float,
    holdout_cap: float,
    projection_cap: float,
    require_raw_spd: bool,
) -> FixedCenterCurvatureFit:
    raw, projected, covariance, raw_values, projection_tf, holdout_tf, nonpositive = (
        _run_kernel(
            _dense_fit_kernel,
            numeric_tensor(center_score, tf.float64),
            numeric_tensor(train_z, tf.float64),
            numeric_tensor(train_scores, tf.float64),
            numeric_tensor(select_z, tf.float64),
            numeric_tensor(select_scores, tf.float64),
            tf.constant(float(eigenvalue_floor), tf.float64),
            tf.constant(float(max_condition_number), tf.float64),
        )
    )
    projection, holdout, raw_nonpositive = (
        float(projection_tf),
        float(holdout_tf),
        int(nonpositive),
    )
    accepted = (
        (not require_raw_spd or raw_nonpositive == 0)
        and projection <= projection_cap
        and holdout <= holdout_cap
    )
    status = (
        "usable"
        if accepted
        else "raw_curvature_not_spd"
        if require_raw_spd and raw_nonpositive
        else "projection_burden_rejected"
        if projection > projection_cap
        else "selection_holdout_rejected"
    )
    return FixedCenterCurvatureFit(
        family="dense",
        replicate_index=replicate_index,
        factor_count=None,
        accepted=accepted,
        status=status,
        raw_precision_z=raw,
        precision_z=projected,
        covariance_z=covariance,
        raw_eigenvalues=raw_values,
        raw_nonpositive_count=raw_nonpositive,
        projection_relative_frobenius=projection,
        selection_holdout_relative_rmse=holdout,
        diagnostics={
            "precision_parameterization": "direct_symmetric_least_squares",
            "geometry_admissible": bool(
                (not require_raw_spd or raw_nonpositive == 0)
                and projection <= projection_cap
            ),
            "selection_holdout_passed": bool(holdout <= holdout_cap),
        },
    )


def _fit_structured_precision(
    center_score: tf.Tensor,
    train_z: tf.Tensor,
    train_scores: tf.Tensor,
    select_z: tf.Tensor,
    select_scores: tf.Tensor,
    *,
    replicate_index: int,
    factor_count: int,
    max_condition_number: float,
    holdout_cap: float,
) -> FixedCenterCurvatureFit:
    result = fit_factor_correlation_score_geometry(
        center_score,
        train_z,
        train_scores,
        select_z,
        select_scores,
        config=FactorCorrelationGeometryConfig(
            factor_count=factor_count,
            max_condition_number=max_condition_number,
            holdout_score_relative_rmse=holdout_cap,
        ),
    )
    precision = (
        None
        if result.precision_z is None
        else numeric_tensor(result.precision_z, tf.float64)
    )
    values = None if precision is None else _run_kernel(_precision_eigenvalues, precision)
    return FixedCenterCurvatureFit(
        family=f"factor_{factor_count}",
        replicate_index=replicate_index,
        factor_count=factor_count,
        accepted=result.accepted,
        status=result.status,
        raw_precision_z=precision,
        precision_z=precision,
        covariance_z=result.covariance_z,
        raw_eigenvalues=values,
        raw_nonpositive_count=(
            None if values is None else int(tf.math.count_nonzero(values <= 0.0))
        ),
        projection_relative_frobenius=0.0 if precision is not None else None,
        selection_holdout_relative_rmse=result.diagnostics.get(
            "holdout_score_relative_rmse"
        ),
        diagnostics={
            **dict(result.diagnostics),
            "covariance_parameterized_precision_prediction": True,
            "parameter_count": result.parameter_count,
            "anchor_indices": result.anchor_indices,
            "geometry_admissible": bool(
                precision is not None
                and result.status in {"usable", "holdout_score_fit_rejected"}
                and (
                    factor_count == 1
                    or result.diagnostics.get("second_factor_identified")
                )
            ),
            "selection_holdout_passed": bool(
                result.diagnostics.get("holdout_score_relative_rmse", float("inf"))
                <= holdout_cap
            ),
        },
    )


def _select_candidate(
    fits: Sequence[FixedCenterCurvatureFit],
    center_score: tf.Tensor,
    selection_partitions: Sequence[tuple[tf.Tensor, tf.Tensor]],
    *,
    thresholds: FixedCenterCurvatureThresholds,
    shrinkage_weights: tuple[float, ...],
    structured_target_family: str | None,
) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    families = tuple(sorted({fit.family for fit in fits}))
    family_fits = tuple(tuple(fit for fit in fits if fit.family == family) for family in families)
    counts = tuple(len(group) for group in family_fits)
    capacity = max(counts, default=0)
    center = numeric_tensor(center_score, tf.float64)
    dimension = int(center.shape[0])
    matrices = tuple(tf.stack(tuple(tf.zeros([dimension, dimension], tf.float64)
        if fit.precision_z is None else numeric_tensor(fit.precision_z, tf.float64) for fit in group))
        for group in family_fits)
    packed = tf.stack(tuple(tf.pad(values, [[0, capacity - count], [0, 0], [0, 0]])
        for values, count in zip(matrices, counts, strict=True))) if families else tf.zeros([0, 0, dimension, dimension], tf.float64)
    flags = tuple(tf.constant([(fit.precision_z is not None,
        bool(fit.diagnostics.get("geometry_admissible")), fit.accepted) for fit in group], tf.bool)
        for group in family_fits)
    flags = tf.stack(tuple(tf.pad(values, [[0, capacity - count], [0, 0]])
        for values, count in zip(flags, counts, strict=True))) if families else tf.zeros([0, 0, 3], tf.bool)
    offsets, scores, partition_rows = _pack_selection_partitions(selection_partitions, dimension)
    caps = tuple(getattr(thresholds, name) for name in stability_native.CAP_NAMES)
    program = selection_native.selection_program(_precision_geometry_kernel, _score_error_kernel,
        dimension, families, counts, partition_rows, len(shrinkage_weights), structured_target_family)
    result = program(packed, flags, center, offsets, scores,
        tf.constant([0. if cap is None else cap for cap in caps], tf.float64),
        tf.constant([cap is not None for cap in caps], tf.bool),
        tf.constant(dimension if thresholds.principal_subspace_rank is None
            else thresholds.principal_subspace_rank, tf.int32),
        tf.constant(shrinkage_weights, tf.float64), tf.constant(thresholds.selection_holdout_relative_rmse_cap, tf.float64))
    report = tf.nest.map_structure(lambda value: value.numpy().tolist(), result)
    return _selection_report(families, family_fits, thresholds, shrinkage_weights, structured_target_family, result, report)


def _selection_report(families, family_fits, thresholds, shrinkage_weights, structured_target_family, result, report):
    """Reconstruct native selector output; no numerical decisions are repeated."""
    stability = {family: _stability_report(group, thresholds,
        {name: value[index] for name, value in report["stability"].items()})
        for index, (family, group) in enumerate(zip(families, family_fits, strict=True))}
    if report["error"]:
        raise ValueError("consensus shrinkage requires SPD inputs and target")
    selection = {"order": ["factor_1", "factor_2", "consensus_structured", "consensus_diagonal"],
        "audit_used_for_selection": False, "requested_structured_target_family": structured_target_family,
        "candidates": [], "stability": stability}
    code = report["family_code"]
    if code in (1, 2):
        family = f"factor_{code}"
        selection["candidates"].append({"family": family, "selected": True})
        return {"family": family, "precision_z": result["precision"]}, selection
    target_family = "diagonal_consensus" if structured_target_family is None else structured_target_family
    family = f"consensus_{target_family}"
    selection["candidates"] = [{"family": family, "weight": weight,
        "selection_holdout_relative_rmse": error, "selected": index == report["selected_index"]}
        for index, (weight, error) in enumerate(zip(shrinkage_weights[:report["visited"]],
            report["errors"][:report["visited"]], strict=True))]
    if not code:
        return None, selection
    selection.update(selected_weight=report["weight"], selected_target=target_family,
        diagonal_only=report["diagonal_only"])
    return {"family": family, "precision_z": result["precision"]}, selection


def _score_relative_rmse(
    precision: tf.Tensor,
    center_score: tf.Tensor,
    offsets: tf.Tensor,
    scores: tf.Tensor,
) -> float:
    return float(
        _run_kernel(
            _score_error_kernel,
            *(
                numeric_tensor(value, tf.float64)
                for value in (precision, center_score, offsets, scores)
            ),
        )
    )


def _mean_selection_error(
    precision: tf.Tensor,
    center_score: tf.Tensor,
    partitions: Sequence[tuple[tf.Tensor, tf.Tensor]],
) -> float:
    center = numeric_tensor(center_score, tf.float64)
    dimension = int(center.shape[0])
    offsets, scores, partition_rows = _pack_selection_partitions(partitions, dimension)
    program = selection_native.mean_error_program(_score_error_kernel, dimension, partition_rows)
    return float(program(numeric_tensor(precision, tf.float64), center, offsets, scores))


def _pack_selection_partitions(partitions, dimension):
    """Pack static report/input schemas without evaluating their numerical scores."""
    rows = tuple(int(offsets.shape[0]) for offsets, _scores in partitions)
    extent = max(rows, default=0)
    offsets = tuple(tf.pad(numeric_tensor(offsets, tf.float64), [[0, extent - count], [0, 0]])
        for (offsets, _scores), count in zip(partitions, rows, strict=True))
    scores = tuple(tf.pad(numeric_tensor(scores, tf.float64), [[0, extent - count], [0, 0]])
        for (_offsets, scores), count in zip(partitions, rows, strict=True))
    return (tf.stack(offsets) if rows else tf.zeros([0, 0, dimension], tf.float64),
        tf.stack(scores) if rows else tf.zeros([0, 0, dimension], tf.float64), rows)


def _family_stability(
    fits: Sequence[FixedCenterCurvatureFit],
    thresholds: FixedCenterCurvatureThresholds,
) -> Mapping[str, Any]:
    # Pack every completed fit, including absent/unusable records. Eligibility
    # counting and all pairwise numerical decisions occur in the native program.
    dimension = next((int(fit.precision_z.shape[0]) for fit in fits
        if fit.precision_z is not None), 1)
    matrices = tuple(tf.zeros([dimension, dimension], tf.float64) if fit.precision_z is None
        else numeric_tensor(fit.precision_z, tf.float64) for fit in fits)
    usable = tf.reshape(tf.constant([(fit.precision_z is not None, bool(fit.diagnostics.get("geometry_admissible")))
        for fit in fits], tf.bool), [len(fits), 2])
    caps = tuple(getattr(thresholds, name) for name in stability_native.CAP_NAMES)
    program = stability_native.stability_program(_precision_geometry_kernel, dimension, len(fits))
    result = program(tf.stack(matrices) if matrices else tf.zeros([0, dimension, dimension], tf.float64),
        usable, tf.constant([0. if cap is None else cap for cap in caps], tf.float64),
        tf.constant([cap is not None for cap in caps], tf.bool),
        tf.constant(dimension if thresholds.principal_subspace_rank is None
            else thresholds.principal_subspace_rank, tf.int32))
    report = {name: value.numpy().tolist() for name, value in result.items()}
    return _stability_report(fits, thresholds, report)


def _stability_report(fits, thresholds, report):
    """Reconstruct the completed native comparison records without reevaluation."""
    dimension = next((int(fit.precision_z.shape[0]) for fit in fits
        if fit.precision_z is not None), 1)
    caps = tuple(getattr(thresholds, name) for name in stability_native.CAP_NAMES)
    if not report["complete"]:
        return {"passed": False, "reason": "fewer_than_two_usable_replicates",
            "replicate_count": len(fits), "usable_count": report["usable_count"], "comparisons": []}
    if report["error"]:
        messages = {1: "left must be a finite symmetric square matrix",
            2: "right must be a finite symmetric square matrix", 3: "subspace_rank must lie in [1, dimension]"}
        raise ValueError(messages[report["error"]])
    comparisons = []
    pair_index = 0
    for left in range(len(fits)):
        for right in range(left + 1, len(fits)):
            values = report["reports"][pair_index]
            summary = values[3 * dimension:]
            rank = int(summary[0])
            metrics = {"left_raw_eigenvalues": values[:dimension],
                "right_raw_eigenvalues": values[dimension:2 * dimension],
                "left_nonpositive_count": int(summary[7]), "right_nonpositive_count": int(summary[8]),
                "trace_normalized_frobenius": summary[4], "trace_normalized_operator": summary[5],
                "positive_subspace_rank": rank,
                "principal_angles_degrees": values[2 * dimension:2 * dimension + rank],
                "maximum_principal_angle_degrees": summary[6] if rank else None,
                "generalized_eigenvalues": {"minimum": summary[1], "maximum": summary[2],
                    "spread": summary[3]} if bool(summary[9]) else None}
            checks = {name: None if cap is None else passed for name, cap, passed in zip(
                stability_native.CHECK_NAMES, caps, report["checks"][pair_index], strict=True)}
            comparisons.append({"left_replicate": fits[left].replicate_index,
                "right_replicate": fits[right].replicate_index, "metrics": metrics,
                "checks": checks, "passed": report["pair_passed"][pair_index]})
            pair_index += 1
    return {"passed": report["passed"], "thresholds_complete": thresholds.stability_caps_complete,
        "replicate_count": len(fits), "usable_count": report["usable_count"], "comparisons": comparisons}


def _blocked_result(
    center: tf.Tensor,
    center_score: tf.Tensor,
    fits: Sequence[FixedCenterCurvatureFit],
    status: str,
    *,
    lineage: Mapping[str, Any] | None,
    extra: Mapping[str, Any] | None = None,
) -> FixedCenterCurvatureResult:
    return FixedCenterCurvatureResult(
        accepted=False,
        status=status,
        center=center,
        center_score_z=center_score,
        selected_family=None,
        selected_precision_z=None,
        selected_covariance_z=None,
        audit_relative_rmse=None,
        fits=tuple(fits),
        diagnostics={
            "center_score_role": "explanatory_only",
            "center_stationarity_required": False,
            "lineage": {} if lineage is None else dict(lineage),
            **({} if extra is None else dict(extra)),
        },
    )


def _cloud_pair(
    offsets: Any, scores: Any, dimension: int, name: str
) -> tuple[tf.Tensor, tf.Tensor]:
    offset_array = numeric_tensor(offsets, tf.float64)
    score_array = numeric_tensor(scores, tf.float64)
    if (
        offset_array.shape.rank != 2
        or offset_array.shape[1] != dimension
        or score_array.shape != offset_array.shape
        or offset_array.shape[0] == 0
    ):
        raise ValueError(f"{name} offsets/scores must have matching [rows, N] shape")
    if not bool(tf.reduce_all(tf.math.is_finite(offset_array))) or not bool(
        tf.reduce_all(tf.math.is_finite(score_array))
    ):
        raise ValueError(f"{name} offsets/scores must be finite")
    return offset_array, score_array


def _cloud_partitions(
    offsets: Any,
    scores: Any,
    dimension: int,
    name: str,
) -> tuple[tuple[tf.Tensor, tf.Tensor], ...]:
    offset_array = numeric_tensor(offsets, tf.float64)
    score_array = numeric_tensor(scores, tf.float64)
    if offset_array.shape.rank == 2:
        return (_cloud_pair(offset_array, score_array, dimension, name),)
    if (
        offset_array.shape.rank != 3
        or offset_array.shape[2] != dimension
        or score_array.shape != offset_array.shape
        or offset_array.shape[0] == 0
    ):
        raise ValueError(
            f"{name} offsets/scores must have matching [replicates, rows, N] shape"
        )
    if not bool(tf.reduce_all(tf.math.is_finite(offset_array))) or not bool(
        tf.reduce_all(tf.math.is_finite(score_array))
    ):
        raise ValueError(f"{name} offsets/scores must be finite")
    return tuple(
        (offset_array[index], score_array[index])
        for index in range(offset_array.shape[0])
    )


def _require_independent_partitions(
    named_arrays: Sequence[tuple[str, tf.Tensor]],
    *,
    raw_regions=None,
) -> None:
    if raw_regions is None:
        raw_regions = [
            (name, regions)
            for name, array in named_arrays
            for regions in buffer_byte_regions(array)
        ]
    for left_index, (left_name, left) in enumerate(raw_regions):
        for right_name, right in raw_regions[left_index + 1 :]:
            if buffer_regions_overlap(left, right):
                raise ValueError(
                    f"partition offsets must be disjoint arrays: {left_name} and {right_name} share memory"
                )
    row_keys = []
    for name, array in named_arrays:
        normalized = numeric_tensor(array, tf.float64)
        normalized = _run_kernel(partition_validation.normalized_offset_bits, normalized)
        _, data = canonical_numeric_bytes(normalized)
        row_bytes = int(normalized.shape[1]) * 8
        row_keys.append(
            (
                name,
                {
                    data[start : start + row_bytes]
                    for start in range(0, len(data), row_bytes)
                },
            )
        )
    for left_index, (left_name, left_keys) in enumerate(row_keys):
        for right_name, right_keys in row_keys[left_index + 1 :]:
            if left_keys.intersection(right_keys):
                raise ValueError(
                    f"partition offsets contain copied rows: {left_name} and {right_name} overlap"
                )


def _vector(value: Any, name: str) -> tf.Tensor:
    array = numeric_tensor(value, tf.float64)
    if (
        array.shape.rank != 1
        or array.shape[0] == 0
        or not bool(tf.reduce_all(tf.math.is_finite(array)))
    ):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def _symmetric_matrix(value: Any, name: str) -> tf.Tensor:
    matrix = numeric_tensor(value, tf.float64)
    if (
        matrix.shape.rank != 2
        or matrix.shape[0] != matrix.shape[1]
        or not bool(tf.reduce_all(tf.math.is_finite(matrix)))
        or not bool(
            tf.reduce_all(
                tf.abs(matrix - tf.transpose(matrix))
                <= 1.0e-12 + 1.0e-10 * tf.abs(tf.transpose(matrix))
            )
        )
    ):
        raise ValueError(f"{name} must be a finite symmetric square matrix")
    return 0.5 * (matrix + tf.transpose(matrix))


def _shrinkage_weights(values: Sequence[float]) -> tuple[float, ...]:
    weights = tuple(float(value) for value in values)
    if (
        not weights
        or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in weights)
        or 0.0 not in weights
        or 1.0 not in weights
    ):
        raise ValueError(
            "shrinkage_weights must be finite in [0,1] and include 0 and 1"
        )
    return tuple(sorted(set(weights)))


def _positive_finite(value: Any, name: str) -> None:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive finite")


def _json_ready(value: Any) -> Any:
    if tf.is_tensor(value) or hasattr(value, "__array_interface__"):
        return numeric_tensor(value).numpy().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value
