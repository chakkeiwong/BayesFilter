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

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import tensorflow as tf

from bayesfilter.inference.factor_correlation_geometry import (
    FactorCorrelationGeometryConfig,
    fit_factor_correlation_score_geometry,
)
from bayesfilter.inference.hmc import PrecomputedMassArtifact


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
    raw_precision_z: np.ndarray | None
    precision_z: np.ndarray | None
    covariance_z: np.ndarray | None
    raw_eigenvalues: np.ndarray | None
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
                array = np.asarray(value, dtype=float).copy()
                array.setflags(write=False)
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
    center: np.ndarray
    center_score_z: np.ndarray
    selected_family: str | None
    selected_precision_z: np.ndarray | None
    selected_covariance_z: np.ndarray | None
    audit_relative_rmse: float | None
    fits: tuple[FixedCenterCurvatureFit, ...]
    diagnostics: Mapping[str, Any]
    nonclaims: tuple[str, ...] = FIXED_CENTER_CURVATURE_NONCLAIMS

    def __post_init__(self) -> None:
        for name in ("center", "center_score_z"):
            array = np.asarray(getattr(self, name), dtype=float).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in ("selected_precision_z", "selected_covariance_z"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float).copy()
                array.setflags(write=False)
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
        scale_array = np.asarray(scale, dtype=float)
        if scale_array.shape != self.center.shape or not np.all(
            np.isfinite(scale_array) & (scale_array > 0.0)
        ):
            raise ValueError("scale must be positive finite with center shape")
        covariance_z = tf.convert_to_tensor(self.selected_covariance_z, tf.float64)
        scale_tf = tf.convert_to_tensor(scale_array, tf.float64)
        covariance_theta = (
            covariance_z * scale_tf[:, None] * scale_tf[None, :]
        ).numpy()
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
            raise ValueError("each training plus selection replicate must total at least 4N")
    named_partition_offsets = [
        *((f"training[{index}]", offsets) for index, (offsets, _scores) in enumerate(train_partitions)),
        *((f"selection[{index}]", offsets) for index, (offsets, _scores) in enumerate(select_partitions)),
        ("audit", audit_z),
    ]
    _require_independent_partitions(named_partition_offsets)
    if int(audit_z.shape[0]) < 2 * dimension:
        raise ValueError("audit rows must total at least 2N")
    if factor_max not in (1, 2):
        raise ValueError("factor_max must be one or two")
    if structured_target_family not in (None, "factor_1", "factor_2"):
        raise ValueError("structured_target_family must be factor_1 or factor_2")
    if structured_target_family == "factor_2" and factor_max != 2:
        raise ValueError("factor_2 structured target requires factor_max=2")
    weights = _shrinkage_weights(shrinkage_weights)

    fits = []
    one_factor_fits = []
    for replicate_index, (train, selection_partition) in enumerate(
        zip(train_partitions, select_partitions, strict=True)
    ):
        train_z, train_scores = train
        select_z, select_scores = selection_partition
        fits.append(
            _fit_dense_precision(
                center_score,
                train_z,
                train_scores,
                select_z,
                select_scores,
                replicate_index=replicate_index,
                eigenvalue_floor=dense_eigenvalue_floor,
                max_condition_number=max_condition_number,
                holdout_cap=thresholds.selection_holdout_relative_rmse_cap,
                projection_cap=thresholds.projection_relative_frobenius_cap,
                require_raw_spd=thresholds.require_raw_spd,
            )
        )
        one_factor = _fit_structured_precision(
            center_score,
            train_z,
            train_scores,
            select_z,
            select_scores,
            replicate_index=replicate_index,
            factor_count=1,
            max_condition_number=max_condition_number,
            holdout_cap=thresholds.selection_holdout_relative_rmse_cap,
        )
        fits.append(one_factor)
        one_factor_fits.append(one_factor)

    one_factor_passed_all_replicates = all(fit.accepted for fit in one_factor_fits)
    one_factor_stability = _family_stability(one_factor_fits, thresholds)
    one_factor_adequate = (
        one_factor_passed_all_replicates and bool(one_factor_stability["passed"])
    )
    two_factor_attempted = factor_max == 2 and (
        not one_factor_adequate or structured_target_family == "factor_2"
    )
    if two_factor_attempted:
        for replicate_index, (train, selection_partition) in enumerate(
            zip(train_partitions, select_partitions, strict=True)
        ):
            train_z, train_scores = train
            select_z, select_scores = selection_partition
            fits.append(
                _fit_structured_precision(
                    center_score,
                    train_z,
                    train_scores,
                    select_z,
                    select_scores,
                    replicate_index=replicate_index,
                    factor_count=2,
                    max_condition_number=max_condition_number,
                    holdout_cap=thresholds.selection_holdout_relative_rmse_cap,
                )
            )

    selected, selection = _select_candidate(
        fits,
        center_score,
        select_partitions,
        thresholds=thresholds,
        shrinkage_weights=weights,
        structured_target_family=structured_target_family,
    )
    selection["factor_escalation"] = {
        "one_factor_passed_all_replicates": one_factor_passed_all_replicates,
        "one_factor_stability_passed": bool(one_factor_stability["passed"]),
        "two_factor_attempted": two_factor_attempted,
        "reason": (
            "explicit_factor_2_target"
            if structured_target_family == "factor_2"
            else "one_factor_fit_holdout_or_stability_rejected"
            if two_factor_attempted
            else "one_factor_fit_holdout_and_stability_passed"
            if factor_max == 2
            else "factor_max_one"
        ),
    }
    if selected is None:
        return _blocked_result(
            fixed_center,
            center_score,
            fits,
            "geometry_readiness_blocked",
            lineage=lineage,
            extra={"selection": selection},
        )

    selected_precision = np.asarray(selected["precision_z"], dtype=float)
    audit_error = _score_relative_rmse(
        selected_precision, center_score, audit_z, audit_scores
    )
    selection["audit_relative_rmse"] = audit_error
    selection["audit_row_count"] = int(audit_z.shape[0])
    selection["audit_used_after_selection"] = True
    selection["audit_changed_selection"] = False
    complete = thresholds.stability_caps_complete
    audit_passed = audit_error <= thresholds.audit_relative_rmse_cap
    diagonal_only = bool(selection.get("diagonal_only", False))
    status = (
        "eligible_for_exact_hmc_canary"
        if complete and audit_passed and not diagonal_only
        else "diagnostic_only"
        if audit_passed
        else "audit_holdout_rejected"
    )
    accepted = status == "eligible_for_exact_hmc_canary"
    return FixedCenterCurvatureResult(
        accepted=accepted,
        status=status,
        center=fixed_center,
        center_score_z=center_score,
        selected_family=str(selected["family"]),
        selected_precision_z=selected_precision,
        selected_covariance_z=tf.linalg.inv(
            tf.convert_to_tensor(selected_precision, tf.float64)
        ).numpy(),
        audit_relative_rmse=audit_error,
        fits=tuple(fits),
        diagnostics={
            "center_score_role": "explanatory_only",
            "center_stationarity_required": False,
            "partition_contract": {
                "replicate_count": len(train_partitions),
                "training_rows_per_replicate": [
                    int(offsets.shape[0]) for offsets, _scores in train_partitions
                ],
                "selection_rows_per_replicate": [
                    int(offsets.shape[0]) for offsets, _scores in select_partitions
                ],
                "audit_rows": int(audit_z.shape[0]),
                "audit_used_after_selection": True,
                "offset_overlap_check": "shared_memory_and_exact_float64_rows",
            },
            "thresholds_complete": complete,
            "diagonal_only_cannot_be_automatically_eligible": diagonal_only,
            "selection": selection,
            "lineage": {} if lineage is None else dict(lineage),
        },
    )


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
    first_tf = tf.convert_to_tensor(first, tf.float64)
    second_tf = tf.convert_to_tensor(second, tf.float64)
    first_values_tf, first_vectors_tf = tf.linalg.eigh(first_tf)
    second_values_tf, second_vectors_tf = tf.linalg.eigh(second_tf)
    first_values = first_values_tf.numpy()
    second_values = second_values_tf.numpy()
    first_vectors = first_vectors_tf.numpy()
    second_vectors = second_vectors_tf.numpy()
    first_positive = first_values > float(positive_subspace_tolerance)
    second_positive = second_values > float(positive_subspace_tolerance)
    positive_rank = min(int(first_positive.sum()), int(second_positive.sum()))
    if subspace_rank is not None:
        requested_rank = int(subspace_rank)
        if requested_rank <= 0 or requested_rank > first.shape[0]:
            raise ValueError("subspace_rank must lie in [1, dimension]")
        positive_rank = min(positive_rank, requested_rank)
    principal_angles: list[float] = []
    if positive_rank:
        first_space = first_vectors[:, -positive_rank:]
        second_space = second_vectors[:, -positive_rank:]
        singular = tf.linalg.svd(
            tf.convert_to_tensor(first_space.T @ second_space, tf.float64),
            compute_uv=False,
        )
        singular = tf.where(
            tf.abs(1.0 - singular) <= tf.constant(1.0e-12, tf.float64),
            tf.ones_like(singular),
            singular,
        )
        principal_angles = (
            tf.acos(tf.clip_by_value(singular, -1.0, 1.0))
            * tf.constant(180.0 / np.pi, tf.float64)
        ).numpy().tolist()
    generalized = None
    if np.all(first_values > 0.0) and np.all(second_values > 0.0):
        chol = tf.linalg.cholesky(first_tf)
        left_solved = tf.linalg.triangular_solve(chol, second_tf)
        transformed = tf.linalg.triangular_solve(
            chol, tf.transpose(left_solved), adjoint=False
        )
        transformed = tf.transpose(transformed)
        generalized_values = tf.linalg.eigvalsh(
            0.5 * (transformed + tf.transpose(transformed))
        ).numpy()
        generalized = {
            "minimum": float(generalized_values.min()),
            "maximum": float(generalized_values.max()),
            "spread": float(generalized_values.max() / generalized_values.min()),
        }
    scale = max(
        abs(float(tf.linalg.trace(first_tf).numpy())),
        abs(float(tf.linalg.trace(second_tf).numpy())),
        1.0e-15,
    )
    difference = first_tf - second_tf
    return _json_ready(
        {
            "left_raw_eigenvalues": first_values,
            "right_raw_eigenvalues": second_values,
            "left_nonpositive_count": int(np.sum(first_values <= 0.0)),
            "right_nonpositive_count": int(np.sum(second_values <= 0.0)),
            "trace_normalized_frobenius": float(
                (tf.linalg.norm(difference) / scale).numpy()
            ),
            "trace_normalized_operator": float(
                (tf.reduce_max(tf.linalg.svd(difference, compute_uv=False)) / scale).numpy()
            ),
            "positive_subspace_rank": positive_rank,
            "principal_angles_degrees": principal_angles,
            "maximum_principal_angle_degrees": (
                None if not principal_angles else float(max(principal_angles))
            ),
            "generalized_eigenvalues": generalized,
        }
    )


def consensus_shrunk_precision(
    precisions: Sequence[Any],
    *,
    target: Any,
    weight: float,
) -> np.ndarray:
    """Return ``(1-weight) * mean(precisions) + weight * target``."""

    if not precisions:
        raise ValueError("at least one precision is required")
    matrices = [_symmetric_matrix(value, "precision") for value in precisions]
    if any(matrix.shape != matrices[0].shape for matrix in matrices[1:]):
        raise ValueError("precision matrices must have matching shapes")
    target_matrix = _symmetric_matrix(target, "target")
    if target_matrix.shape != matrices[0].shape:
        raise ValueError("target shape must match precisions")
    shrinkage = float(weight)
    if not np.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("weight must be finite and in [0, 1]")
    stack = tf.convert_to_tensor(np.stack(matrices, axis=0), tf.float64)
    target_tf = tf.convert_to_tensor(target_matrix, tf.float64)
    candidate = consensus_shrunk_precision_tf(
        stack, target_tf, tf.convert_to_tensor(shrinkage, tf.float64)
    )
    if float(tf.reduce_min(tf.linalg.eigvalsh(candidate)).numpy()) <= 0.0:
        raise ValueError("consensus shrinkage requires SPD inputs and target")
    return candidate.numpy()


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


def _fit_dense_precision(
    center_score: np.ndarray,
    train_z: np.ndarray,
    train_scores: np.ndarray,
    select_z: np.ndarray,
    select_scores: np.ndarray,
    *,
    replicate_index: int,
    eigenvalue_floor: float,
    max_condition_number: float,
    holdout_cap: float,
    projection_cap: float,
    require_raw_spd: bool,
) -> FixedCenterCurvatureFit:
    train_z_tf = tf.convert_to_tensor(train_z, tf.float64)
    response_tf = tf.convert_to_tensor(
        center_score[None, :] - train_scores, tf.float64
    )
    raw_tf = tf.linalg.lstsq(train_z_tf, response_tf, fast=False)
    raw_tf = 0.5 * (raw_tf + tf.transpose(raw_tf))
    raw_values_tf, vectors_tf = tf.linalg.eigh(raw_tf)
    raw = raw_tf.numpy()
    raw_values = raw_values_tf.numpy()
    floor = max(float(eigenvalue_floor), float(np.max(np.abs(raw_values))) / max_condition_number)
    projected_values = np.maximum(raw_values, floor)
    projected_tf = tf.matmul(
        vectors_tf * tf.convert_to_tensor(projected_values[None, :], tf.float64),
        vectors_tf,
        transpose_b=True,
    )
    projected = projected_tf.numpy()
    projection = float(
        (
            tf.linalg.norm(projected_tf - raw_tf)
            / tf.maximum(tf.linalg.norm(raw_tf), tf.constant(1.0e-15, tf.float64))
        ).numpy()
    )
    holdout = _score_relative_rmse(
        projected, center_score, select_z, select_scores
    )
    raw_nonpositive = int(np.sum(raw_values <= 0.0))
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
        covariance_z=tf.linalg.inv(projected_tf).numpy(),
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
    center_score: np.ndarray,
    train_z: np.ndarray,
    train_scores: np.ndarray,
    select_z: np.ndarray,
    select_scores: np.ndarray,
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
    precision = None if result.precision_z is None else np.asarray(result.precision_z)
    values = None if precision is None else np.linalg.eigvalsh(precision)
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
        raw_nonpositive_count=(None if values is None else int(np.sum(values <= 0.0))),
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
                and result.status
                in {"usable", "holdout_score_fit_rejected"}
                and (factor_count == 1 or result.diagnostics.get("second_factor_identified"))
            ),
            "selection_holdout_passed": bool(
                result.diagnostics.get("holdout_score_relative_rmse", float("inf"))
                <= holdout_cap
            ),
        },
    )


def _select_candidate(
    fits: Sequence[FixedCenterCurvatureFit],
    center_score: np.ndarray,
    selection_partitions: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    thresholds: FixedCenterCurvatureThresholds,
    shrinkage_weights: tuple[float, ...],
    structured_target_family: str | None,
) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    selection: dict[str, Any] = {
        "order": ["factor_1", "factor_2", "consensus_structured", "consensus_diagonal"],
        "audit_used_for_selection": False,
        "requested_structured_target_family": structured_target_family,
        "candidates": [],
    }
    families = sorted({fit.family for fit in fits})
    stability: dict[str, Any] = {}
    stable_families: set[str] = set()
    for family in families:
        family_fits = [fit for fit in fits if fit.family == family]
        family_stability = _family_stability(family_fits, thresholds)
        stability[family] = family_stability
        if family_stability["passed"]:
            stable_families.add(family)
    selection["stability"] = stability

    for family in (() if structured_target_family is not None else ("factor_1", "factor_2")):
        family_fits = [
            fit
            for fit in fits
            if fit.family == family and fit.accepted and fit.precision_z is not None
        ]
        if family in stable_families and len(family_fits) >= 2:
            consensus = tf.reduce_mean(
                tf.convert_to_tensor(
                    np.stack([fit.precision_z for fit in family_fits], axis=0),
                    tf.float64,
                ),
                axis=0,
            ).numpy()
            selection["candidates"].append({"family": family, "selected": True})
            return {"family": family, "precision_z": consensus}, selection

    admissible = [
        fit
        for fit in fits
        if fit.precision_z is not None
        and fit.family in stable_families
        and bool(fit.diagnostics.get("geometry_admissible"))
    ]
    if not admissible:
        return None, selection
    dense_precisions = [
        fit.precision_z for fit in admissible if fit.family == "dense"
    ]
    if not dense_precisions:
        return None, selection
    consensus = tf.reduce_mean(
        tf.convert_to_tensor(np.stack(dense_precisions), tf.float64), axis=0
    ).numpy()
    structured = None
    if structured_target_family is not None:
        candidates_for_target = [
            fit
            for fit in admissible
            if fit.family == structured_target_family and fit.accepted
        ]
        if structured_target_family not in stable_families or not candidates_for_target:
            return None, selection
        target = tf.reduce_mean(
            tf.convert_to_tensor(
                np.stack([fit.precision_z for fit in candidates_for_target]),
                tf.float64,
            ),
            axis=0,
        ).numpy()
        target_family = structured_target_family
    else:
        target = np.diag(np.diag(consensus))
        target_family = "diagonal_consensus"
    candidates = []
    for weight in shrinkage_weights:
        candidate = consensus_shrunk_precision(
            dense_precisions, target=target, weight=weight
        )
        error = _mean_selection_error(
            candidate, center_score, selection_partitions
        )
        candidates.append((error, weight, candidate))
        selection["candidates"].append(
            {
                "family": f"consensus_{target_family}",
                "weight": weight,
                "selection_holdout_relative_rmse": error,
                "selected": False,
            }
        )
    error, weight, candidate = min(candidates, key=lambda row: (row[0], row[1]))
    if error > thresholds.selection_holdout_relative_rmse_cap:
        return None, selection
    selected_index = min(
        range(len(candidates)), key=lambda index: (candidates[index][0], candidates[index][1])
    )
    selection["candidates"][-len(candidates) + selected_index]["selected"] = True
    selection["selected_weight"] = weight
    selection["selected_target"] = target_family
    selection["diagonal_only"] = bool(
        target_family == "diagonal_consensus" and weight == 1.0
    )
    return {
        "family": f"consensus_{target_family}",
        "precision_z": candidate,
    }, selection


def _score_relative_rmse(
    precision: np.ndarray,
    center_score: np.ndarray,
    offsets: np.ndarray,
    scores: np.ndarray,
) -> float:
    response = tf.convert_to_tensor(center_score[None, :] - scores, tf.float64)
    prediction = tf.matmul(
        tf.convert_to_tensor(offsets, tf.float64),
        tf.convert_to_tensor(precision, tf.float64),
        transpose_b=True,
    )
    error = tf.sqrt(tf.reduce_mean(tf.square(prediction - response)))
    scale = tf.maximum(
        tf.sqrt(tf.reduce_mean(tf.square(response))),
        tf.constant(1.0e-15, tf.float64),
    )
    return float((error / scale).numpy())


def _mean_selection_error(
    precision: np.ndarray,
    center_score: np.ndarray,
    partitions: Sequence[tuple[np.ndarray, np.ndarray]],
) -> float:
    errors = [
        _score_relative_rmse(precision, center_score, offsets, scores)
        for offsets, scores in partitions
    ]
    return float(tf.reduce_mean(tf.convert_to_tensor(errors, tf.float64)).numpy())


def _family_stability(
    fits: Sequence[FixedCenterCurvatureFit],
    thresholds: FixedCenterCurvatureThresholds,
) -> Mapping[str, Any]:
    usable = [
        fit
        for fit in fits
        if fit.precision_z is not None
        and bool(fit.diagnostics.get("geometry_admissible"))
    ]
    if len(usable) != len(fits) or len(usable) < 2:
        return {
            "passed": False,
            "reason": "fewer_than_two_usable_replicates",
            "replicate_count": len(fits),
            "usable_count": len(usable),
            "comparisons": [],
        }
    comparisons = []
    all_passed = True
    for left_index, left in enumerate(usable):
        for right in usable[left_index + 1 :]:
            metrics = dict(
                compare_precision_geometry(
                    left.precision_z,
                    right.precision_z,
                    subspace_rank=thresholds.principal_subspace_rank,
                )
            )
            generalized = metrics["generalized_eigenvalues"]
            checks = {
                "generalized_eigenvalue_spread": (
                    None
                    if thresholds.generalized_eigenvalue_spread_cap is None
                    else generalized is not None
                    and generalized["spread"]
                    <= thresholds.generalized_eigenvalue_spread_cap
                ),
                "trace_normalized_frobenius": (
                    None
                    if thresholds.trace_normalized_frobenius_cap is None
                    else metrics["trace_normalized_frobenius"]
                    <= thresholds.trace_normalized_frobenius_cap
                ),
                "trace_normalized_operator": (
                    None
                    if thresholds.trace_normalized_operator_cap is None
                    else metrics["trace_normalized_operator"]
                    <= thresholds.trace_normalized_operator_cap
                ),
                "principal_angle_degrees": (
                    None
                    if thresholds.principal_angle_degrees_cap is None
                    else metrics["maximum_principal_angle_degrees"] is not None
                    and metrics["maximum_principal_angle_degrees"]
                    <= thresholds.principal_angle_degrees_cap
                ),
            }
            passed = all(value is not False for value in checks.values())
            all_passed = all_passed and passed
            comparisons.append(
                {
                    "left_replicate": left.replicate_index,
                    "right_replicate": right.replicate_index,
                    "metrics": metrics,
                    "checks": checks,
                    "passed": passed,
                }
            )
    return {
        "passed": all_passed,
        "thresholds_complete": thresholds.stability_caps_complete,
        "replicate_count": len(fits),
        "usable_count": len(usable),
        "comparisons": comparisons,
    }


def _blocked_result(
    center: np.ndarray,
    center_score: np.ndarray,
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
) -> tuple[np.ndarray, np.ndarray]:
    offset_array = np.asarray(offsets, dtype=float)
    score_array = np.asarray(scores, dtype=float)
    if (
        offset_array.ndim != 2
        or offset_array.shape[1] != dimension
        or score_array.shape != offset_array.shape
        or offset_array.shape[0] == 0
    ):
        raise ValueError(f"{name} offsets/scores must have matching [rows, N] shape")
    if not np.all(np.isfinite(offset_array)) or not np.all(np.isfinite(score_array)):
        raise ValueError(f"{name} offsets/scores must be finite")
    return offset_array, score_array


def _cloud_partitions(
    offsets: Any,
    scores: Any,
    dimension: int,
    name: str,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    offset_array = np.asarray(offsets, dtype=float)
    score_array = np.asarray(scores, dtype=float)
    if offset_array.ndim == 2:
        return (_cloud_pair(offset_array, score_array, dimension, name),)
    if (
        offset_array.ndim != 3
        or offset_array.shape[2] != dimension
        or score_array.shape != offset_array.shape
        or offset_array.shape[0] == 0
    ):
        raise ValueError(
            f"{name} offsets/scores must have matching [replicates, rows, N] shape"
        )
    if not np.all(np.isfinite(offset_array)) or not np.all(np.isfinite(score_array)):
        raise ValueError(f"{name} offsets/scores must be finite")
    return tuple(
        (offset_array[index], score_array[index])
        for index in range(offset_array.shape[0])
    )


def _require_independent_partitions(
    named_arrays: Sequence[tuple[str, np.ndarray]],
) -> None:
    row_keys = []
    for name, array in named_arrays:
        normalized = np.ascontiguousarray(array, dtype=np.float64).copy()
        normalized[normalized == 0.0] = 0.0
        row_keys.append((name, {row.tobytes() for row in normalized}))
    for left_index, (left_name, left) in enumerate(named_arrays):
        for right_name, right in named_arrays[left_index + 1 :]:
            if np.shares_memory(left, right):
                raise ValueError(
                    f"partition offsets must be disjoint arrays: {left_name} and {right_name} share memory"
                )
    for left_index, (left_name, left_keys) in enumerate(row_keys):
        for right_name, right_keys in row_keys[left_index + 1 :]:
            if left_keys.intersection(right_keys):
                raise ValueError(
                    f"partition offsets contain copied rows: {left_name} and {right_name} overlap"
                )


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def _symmetric_matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=1.0e-12)
    ):
        raise ValueError(f"{name} must be a finite symmetric square matrix")
    return 0.5 * (matrix + matrix.T)


def _shrinkage_weights(values: Sequence[float]) -> tuple[float, ...]:
    weights = tuple(float(value) for value in values)
    if (
        not weights
        or any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in weights)
        or 0.0 not in weights
        or 1.0 not in weights
    ):
        raise ValueError("shrinkage_weights must be finite in [0,1] and include 0 and 1")
    return tuple(sorted(set(weights)))


def _positive_finite(value: Any, name: str) -> None:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive finite")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value
