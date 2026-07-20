"""Deterministic fixed-branch TT fitting for Phase-3 highdim contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.bases import BoundedInterval, LegendreBasis1D, ProductBasis
from bayesfilter.highdim.diagnostics import (
    HighDimStatus,
    MeasureConvention,
    assert_density_matches_mass,
    assert_tf_float64,
    freeze_mapping,
)
from bayesfilter.highdim.fixed_branch import BranchHash, BranchIdentity, BranchManifest
from bayesfilter.highdim.tt import FunctionalTT, TTCore
from bayesfilter.highdim.validation import ComplexityBudget


_FLOAT64_EPS = float(tf.experimental.numpy.finfo(tf.float64.as_numpy_dtype).eps)
_DEFAULT_COLUMN_SCALE_FLOOR = _FLOAT64_EPS
_SCALE_FLOOR_RULE = (
    "max(sqrt(float64_eps) * max_weighted_column_norm, column_scale_floor)"
)
_STABILIZATION_POLICY_ID = "objective_preserving_column_scaled_augmented_ridge_v1"
_SOLVER_BACKEND = "tensorflow.linalg.lstsq(fast=False)"
_SOLVER_MODE = "objective_preserving_column_scaled_augmented_ridge"
_TRANSFORMED_RIDGE_RULE = "rho_times_S_inverse_squared"
_CONDITION_GATE_TARGET = "scaled_augmented_solved_system"
_UNSCALED_CONDITION_ROLE = "diagnostic_only"


@dataclass(frozen=True)
class FixedTTFitConfig:
    """Declared fixed-rank weighted ridge ALS configuration."""

    ranks: tuple[int, ...]
    ridge: float
    max_sweeps: int
    sweep_order: tuple[int, ...]
    row_budget: int
    column_budget: int
    dense_matrix_byte_budget: int
    normal_matrix_byte_budget: int
    condition_number_warning: float
    condition_number_veto: float
    holdout_tolerance: float
    dtype: tf.DType = tf.float64
    stabilization_policy_id: str = _STABILIZATION_POLICY_ID
    solver_backend: str = _SOLVER_BACKEND
    column_scale_floor: float = _DEFAULT_COLUMN_SCALE_FLOOR

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", tuple(int(rank) for rank in self.ranks))
        object.__setattr__(
            self,
            "sweep_order",
            tuple(int(axis) for axis in self.sweep_order),
        )
        if self.dtype != tf.float64:
            raise ValueError("FixedTTFitConfig requires tf.float64")
        if len(self.ranks) < 2 or any(rank <= 0 for rank in self.ranks):
            raise ValueError(f"ranks: {HighDimStatus.INVALID_SHAPE.value}")
        if self.ranks[0] != 1 or self.ranks[-1] != 1:
            raise ValueError(f"ranks: {HighDimStatus.INVALID_SHAPE.value}")
        if not self.sweep_order:
            raise ValueError("sweep_order must be nonempty")
        if self.max_sweeps <= 0:
            raise ValueError("max_sweeps must be positive")
        for name in (
            "row_budget",
            "column_budget",
            "dense_matrix_byte_budget",
            "normal_matrix_byte_budget",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "ridge",
            "condition_number_warning",
            "condition_number_veto",
            "holdout_tolerance",
            "column_scale_floor",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if self.condition_number_warning <= 0.0 or self.condition_number_veto <= 0.0:
            raise ValueError("condition-number thresholds must be positive")
        if self.column_scale_floor <= 0.0:
            raise ValueError("column_scale_floor must be positive")
        stabilization_policy_id = str(self.stabilization_policy_id)
        if not stabilization_policy_id.strip():
            raise ValueError("stabilization_policy_id must be nonempty")
        object.__setattr__(self, "stabilization_policy_id", stabilization_policy_id)
        solver_backend = str(self.solver_backend)
        if solver_backend != _SOLVER_BACKEND:
            raise ValueError(f"solver_backend must be {_SOLVER_BACKEND!r}")
        object.__setattr__(self, "solver_backend", solver_backend)


@dataclass(frozen=True)
class FixedTTFitSampleBatch:
    """Weighted sample batch for fixed-design TT fitting."""

    points: tf.Tensor
    target_values: tf.Tensor
    weights: tf.Tensor
    holdout_points: tf.Tensor | None = None
    holdout_values: tf.Tensor | None = None
    holdout_weights: tf.Tensor | None = None

    def __post_init__(self) -> None:
        points = tf.convert_to_tensor(self.points, dtype=tf.float64)
        targets = tf.convert_to_tensor(self.target_values, dtype=tf.float64)
        weights = tf.convert_to_tensor(self.weights, dtype=tf.float64)
        _validate_sample_tensors("samples", points, targets, weights)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "target_values", targets)
        object.__setattr__(self, "weights", weights)
        if self.holdout_points is None:
            if self.holdout_values is not None or self.holdout_weights is not None:
                raise ValueError(f"holdout: {HighDimStatus.INVALID_SHAPE.value}")
            object.__setattr__(self, "holdout_values", None)
            object.__setattr__(self, "holdout_weights", None)
            return
        if self.holdout_values is None:
            raise ValueError(f"holdout_values: {HighDimStatus.INVALID_SHAPE.value}")
        holdout_points = tf.convert_to_tensor(self.holdout_points, dtype=tf.float64)
        holdout_values = tf.convert_to_tensor(self.holdout_values, dtype=tf.float64)
        holdout_weights = (
            tf.ones_like(holdout_values, dtype=tf.float64)
            if self.holdout_weights is None
            else tf.convert_to_tensor(self.holdout_weights, dtype=tf.float64)
        )
        _validate_sample_tensors(
            "holdout",
            holdout_points,
            holdout_values,
            holdout_weights,
        )
        object.__setattr__(self, "holdout_points", holdout_points)
        object.__setattr__(self, "holdout_values", holdout_values)
        object.__setattr__(self, "holdout_weights", holdout_weights)


@dataclass(frozen=True)
class FixedTTFitResult:
    """Result of one deterministic fixed-branch TT fit."""

    fitted_tt: FunctionalTT
    status: HighDimStatus
    termination_reason: str
    stop_condition_triggered: str
    fit_residual: tf.Tensor | None
    holdout_residual: tf.Tensor | None
    branch_identity: BranchIdentity
    core_update_statuses: tuple[Mapping[str, object], ...]
    diagnostics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fitted_tt, FunctionalTT):
            raise TypeError("fitted_tt must be a FunctionalTT")
        if not isinstance(self.status, HighDimStatus):
            raise TypeError("status must be a HighDimStatus")
        if not isinstance(self.branch_identity, BranchIdentity):
            raise TypeError("branch_identity must be a BranchIdentity")
        for name in ("termination_reason", "stop_condition_triggered"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be nonempty")
        for name in ("fit_residual", "holdout_residual"):
            value = getattr(self, name)
            if value is not None:
                tensor = tf.convert_to_tensor(value, dtype=tf.float64)
                if tensor.shape.rank != 0:
                    raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")
                object.__setattr__(self, name, tensor)
        object.__setattr__(
            self,
            "core_update_statuses",
            tuple(MappingProxyType(dict(record)) for record in self.core_update_statuses),
        )
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))

    @property
    def branch_hash(self) -> BranchHash:
        """Full realized branch hash."""

        return self.branch_identity.hash


@dataclass(frozen=True)
class _CoreUpdateSystem:
    design_matrix: tf.Tensor
    normal_matrix: tf.Tensor
    rhs: tf.Tensor
    condition_number: float
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class _ScaledAugmentedSolveResult:
    solution: tf.Tensor
    column_scales: tf.Tensor
    raw_column_norms: tf.Tensor
    scale_floor: tf.Tensor
    scaled_augmented_matrix: tf.Tensor
    scaled_augmented_rhs: tf.Tensor
    scaled_augmented_condition_number: float
    diagnostics: Mapping[str, object]


class FixedTTFitter:
    """Fixed-rank weighted ridge ALS fitter with replayable branch manifests."""

    def fit(
        self,
        product_basis: ProductBasis,
        samples: FixedTTFitSampleBatch,
        config: FixedTTFitConfig,
        initial_cores: Sequence[TTCore | tf.Tensor],
        branch_seed: int | str,
        measure_convention: MeasureConvention,
        initialization_rule: str = "supplied_initial_cores",
    ) -> FixedTTFitResult:
        if not isinstance(product_basis, ProductBasis):
            raise TypeError("product_basis must be a ProductBasis")
        if not isinstance(samples, FixedTTFitSampleBatch):
            raise TypeError("samples must be a FixedTTFitSampleBatch")
        if not isinstance(config, FixedTTFitConfig):
            raise TypeError("config must be a FixedTTFitConfig")
        if measure_convention != product_basis.convention:
            raise ValueError(f"measure_convention: {HighDimStatus.MEASURE_MISMATCH.value}")
        assert_density_matches_mass(measure_convention)
        initialization_rule = str(initialization_rule)
        if not initialization_rule.strip():
            raise ValueError("initialization_rule must be nonempty")
        self._validate_config_for_basis(product_basis, config)
        self._validate_batch_dimension(product_basis, samples)
        cores = list(
            self._validate_initial_cores(
                product_basis,
                measure_convention,
                config,
                initial_cores,
            )
        )
        initial_core_hash = _hash_tensors("initial_cores_hash.v1", [core.values for core in cores])
        update_records: list[Mapping[str, object]] = []
        environment_rebuild_hashes: list[str] = []
        status = HighDimStatus.OK
        termination_reason = "max_sweeps_exhausted"
        stop_condition = "max_sweeps_exhausted"

        for sweep_index in range(config.max_sweeps):
            for axis in config.sweep_order:
                pre_hash = _hash_tensors("pre_update_cores_hash.v1", [core.values for core in cores])
                environment_rebuild_hashes.append(pre_hash)
                updated_core, record = self._fit_core_update(
                    product_basis=product_basis,
                    points=samples.points,
                    target_values=samples.target_values,
                    weights=samples.weights,
                    cores=tuple(cores),
                    core_index=axis,
                    config=config,
                    sweep_index=sweep_index,
                )
                update_records.append(record)
                record_status = HighDimStatus(str(record["status"]))
                if record_status is not HighDimStatus.OK:
                    status = record_status
                    termination_reason = str(record["termination_reason"])
                    stop_condition = str(record["stop_condition_triggered"])
                    return self._finalize_result(
                        product_basis=product_basis,
                        samples=samples,
                        config=config,
                        cores=tuple(cores),
                        branch_seed=branch_seed,
                        measure_convention=measure_convention,
                        initial_core_hash=initial_core_hash,
                        initialization_rule=initialization_rule,
                        update_records=tuple(update_records),
                        environment_rebuild_hashes=tuple(environment_rebuild_hashes),
                        status=status,
                        termination_reason=termination_reason,
                        stop_condition=stop_condition,
                        fit_residual=None,
                        holdout_residual=None,
                    )
                cores[axis] = updated_core

        fitted_tt_without_identity = FunctionalTT(
            cores,
            product_basis,
            measure_convention,
            complexity_budget=_evaluation_budget_from_fit_config(config),
        )
        fit_residual = _weighted_rms_residual(
            fitted_tt_without_identity.evaluate(samples.points),
            samples.target_values,
            samples.weights,
        )
        if not bool(tf.math.is_finite(fit_residual).numpy()):
            status = HighDimStatus.NONFINITE_VALUE
            termination_reason = "nonfinite_fit_residual"
            stop_condition = HighDimStatus.NONFINITE_VALUE.value
        holdout_residual = None
        if status is HighDimStatus.OK and samples.holdout_points is not None:
            holdout_residual = _weighted_rms_residual(
                fitted_tt_without_identity.evaluate(samples.holdout_points),
                samples.holdout_values,
                samples.holdout_weights,
            )
            if not bool(tf.math.is_finite(holdout_residual).numpy()):
                status = HighDimStatus.NONFINITE_VALUE
                termination_reason = "nonfinite_holdout_residual"
                stop_condition = HighDimStatus.NONFINITE_VALUE.value
            elif bool((holdout_residual > config.holdout_tolerance).numpy()):
                status = HighDimStatus.HOLDOUT_RESIDUAL_VETO
                termination_reason = "holdout_residual_veto"
                stop_condition = HighDimStatus.HOLDOUT_RESIDUAL_VETO.value
        return self._finalize_result(
            product_basis=product_basis,
            samples=samples,
            config=config,
            cores=tuple(cores),
            branch_seed=branch_seed,
            measure_convention=measure_convention,
            initial_core_hash=initial_core_hash,
            initialization_rule=initialization_rule,
            update_records=tuple(update_records),
            environment_rebuild_hashes=tuple(environment_rebuild_hashes),
            status=status,
            termination_reason=termination_reason,
            stop_condition=stop_condition,
            fit_residual=fit_residual,
            holdout_residual=holdout_residual,
        )

    def build_core_update_system(
        self,
        product_basis: ProductBasis,
        points: tf.Tensor,
        target_values: tf.Tensor,
        weights: tf.Tensor,
        cores: Sequence[TTCore],
        core_index: int,
        config: FixedTTFitConfig,
    ) -> _CoreUpdateSystem:
        """Build the declared core-design system for direct reference tests."""

        points = tf.convert_to_tensor(points, dtype=tf.float64)
        target_values = tf.convert_to_tensor(target_values, dtype=tf.float64)
        weights = tf.convert_to_tensor(weights, dtype=tf.float64)
        self._validate_batch_dimension(
            product_basis,
            FixedTTFitSampleBatch(points, target_values, weights),
        )
        checked_cores = self._validate_initial_cores(
            product_basis,
            product_basis.convention,
            config,
            cores,
        )
        gate = self._check_design_budget(product_basis, points, checked_cores, core_index, config)
        if gate["status"] != HighDimStatus.OK.value:
            raise ValueError(HighDimStatus.COMPLEXITY_GATE.value)
        design = self._build_design_matrix(product_basis, points, checked_cores, core_index)
        normal, rhs = _normal_equations(design, target_values, weights, config.ridge)
        condition_number = _condition_number(normal)
        return _CoreUpdateSystem(
            design_matrix=design,
            normal_matrix=normal,
            rhs=rhs,
            condition_number=condition_number,
            diagnostics={
                **gate,
                "condition_number": _finite_number_or_text(condition_number),
            },
        )

    def _fit_core_update(
        self,
        product_basis: ProductBasis,
        points: tf.Tensor,
        target_values: tf.Tensor,
        weights: tf.Tensor,
        cores: tuple[TTCore, ...],
        core_index: int,
        config: FixedTTFitConfig,
        sweep_index: int,
    ) -> tuple[TTCore, Mapping[str, object]]:
        gate = self._check_design_budget(product_basis, points, cores, core_index, config)
        if gate["status"] != HighDimStatus.OK.value:
            gate_status = HighDimStatus(str(gate["status"]))
            return cores[core_index], {
                **gate,
                "core_index": core_index,
                "sweep_index": sweep_index,
                "termination_reason": str(gate["gate"]),
                "stop_condition_triggered": gate_status.value,
            }
        design = self._build_design_matrix(product_basis, points, cores, core_index)
        if not bool(tf.reduce_all(tf.math.is_finite(design)).numpy()):
            return cores[core_index], _failed_update_record(
                status=HighDimStatus.NONFINITE_VALUE,
                core_index=core_index,
                sweep_index=sweep_index,
                termination_reason="nonfinite_design_matrix",
            )
        normal, rhs = _normal_equations(design, target_values, weights, config.ridge)
        if not bool(
            tf.reduce_all(tf.math.is_finite(normal)).numpy()
            and tf.reduce_all(tf.math.is_finite(rhs)).numpy()
        ):
            return cores[core_index], _failed_update_record(
                status=HighDimStatus.NONFINITE_VALUE,
                core_index=core_index,
                sweep_index=sweep_index,
                termination_reason="nonfinite_normal_equations",
            )
        unscaled_condition_number = _condition_number(normal)
        try:
            solve_result = _solve_scaled_augmented_ridge(
                design=design,
                target_values=target_values,
                weights=weights,
                ridge=config.ridge,
                column_scale_floor=config.column_scale_floor,
                stabilization_policy_id=config.stabilization_policy_id,
                solver_backend=config.solver_backend,
            )
        except Exception:
            return cores[core_index], {
                **gate,
                "condition_number": "unavailable",
                "condition_warning": False,
                "condition_number_warning": float(config.condition_number_warning),
                "condition_number_veto": float(config.condition_number_veto),
                "condition_number_semantics": "scaled_augmented_solve_condition",
                "unscaled_normal_condition_number": _finite_number_or_text(
                    unscaled_condition_number
                ),
                "unscaled_normal_condition_warning": bool(
                    unscaled_condition_number > config.condition_number_warning
                ),
                "unscaled_normal_condition_veto": bool(
                    (not math.isfinite(unscaled_condition_number))
                    or unscaled_condition_number > config.condition_number_veto
                ),
                **_stabilization_policy_payload(config),
                "status": HighDimStatus.CONDITION_NUMBER_VETO.value,
                "core_index": core_index,
                "sweep_index": sweep_index,
                "termination_reason": "scaled_augmented_lstsq_failure",
                "stop_condition_triggered": HighDimStatus.CONDITION_NUMBER_VETO.value,
            }
        condition_number = solve_result.scaled_augmented_condition_number
        condition_record = {
            "condition_number": _finite_number_or_text(condition_number),
            "condition_warning": bool(condition_number > config.condition_number_warning),
            "condition_number_warning": float(config.condition_number_warning),
            "condition_number_veto": float(config.condition_number_veto),
            "condition_number_semantics": "scaled_augmented_solve_condition",
            "unscaled_normal_condition_number": _finite_number_or_text(
                unscaled_condition_number
            ),
            "unscaled_normal_condition_warning": bool(
                unscaled_condition_number > config.condition_number_warning
            ),
            "unscaled_normal_condition_veto": bool(
                (not math.isfinite(unscaled_condition_number))
                or unscaled_condition_number > config.condition_number_veto
            ),
            **solve_result.diagnostics,
        }
        if (not math.isfinite(condition_number)) or condition_number > config.condition_number_veto:
            return cores[core_index], {
                **gate,
                **condition_record,
                "status": HighDimStatus.CONDITION_NUMBER_VETO.value,
                "core_index": core_index,
                "sweep_index": sweep_index,
                "termination_reason": "scaled_augmented_condition_number_veto",
                "stop_condition_triggered": HighDimStatus.CONDITION_NUMBER_VETO.value,
            }
        solution = solve_result.solution
        if not bool(tf.reduce_all(tf.math.is_finite(solution)).numpy()):
            return cores[core_index], _failed_update_record(
                status=HighDimStatus.NONFINITE_VALUE,
                core_index=core_index,
                sweep_index=sweep_index,
                termination_reason="nonfinite_solution",
            )
        old_core = cores[core_index]
        new_values = tf.reshape(
            solution,
            [old_core.left_rank, old_core.basis_dim, old_core.right_rank],
        )
        return TTCore(new_values), {
            **gate,
            **condition_record,
            "status": HighDimStatus.OK.value,
            "core_index": core_index,
            "sweep_index": sweep_index,
            "termination_reason": "core_update_accepted",
            "stop_condition_triggered": "none",
            "solver_backend": config.solver_backend,
            "solver_mode": _SOLVER_MODE,
        }

    def _build_design_matrix(
        self,
        product_basis: ProductBasis,
        points: tf.Tensor,
        cores: Sequence[TTCore],
        core_index: int,
    ) -> tf.Tensor:
        matrices = _core_matrices(product_basis, points, cores)
        left = _left_environments(matrices)[core_index]
        right = _right_environments(matrices)[core_index]
        basis_values = product_basis.evaluate_axis(core_index, points[:, core_index])
        blocks = tf.einsum("na,nl,nb->nalb", left, basis_values, right)
        return tf.reshape(
            blocks,
            [
                int(points.shape[0]),
                cores[core_index].left_rank
                * cores[core_index].basis_dim
                * cores[core_index].right_rank,
            ],
        )

    def _check_design_budget(
        self,
        product_basis: ProductBasis,
        points: tf.Tensor,
        cores: Sequence[TTCore],
        core_index: int,
        config: FixedTTFitConfig,
    ) -> Mapping[str, object]:
        if points.shape.rank != 2 or points.shape[0] is None:
            return {
                "status": HighDimStatus.INVALID_SHAPE.value,
                "gate": "points_shape",
                "message": "static row count is required",
            }
        n_rows = int(points.shape[0])
        core = cores[core_index]
        n_cols = core.left_rank * product_basis.bases[core_index].basis_dim * core.right_rank
        dtype_size = tf.float64.size
        dense_bytes = n_rows * n_cols * dtype_size
        normal_matrix_bytes = n_cols * n_cols * dtype_size
        common = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "dense_matrix_bytes": dense_bytes,
            "normal_matrix_bytes": normal_matrix_bytes,
            "row_budget": int(config.row_budget),
            "column_budget": int(config.column_budget),
            "dense_matrix_byte_budget": int(config.dense_matrix_byte_budget),
            "normal_matrix_byte_budget": int(config.normal_matrix_byte_budget),
        }
        if n_rows > config.row_budget:
            return {
                **common,
                "status": HighDimStatus.COMPLEXITY_GATE.value,
                "gate": "row_budget",
            }
        if n_cols > config.column_budget:
            return {
                **common,
                "status": HighDimStatus.COMPLEXITY_GATE.value,
                "gate": "column_budget",
            }
        if dense_bytes > config.dense_matrix_byte_budget:
            return {
                **common,
                "status": HighDimStatus.COMPLEXITY_GATE.value,
                "gate": "dense_matrix_byte_budget",
            }
        if normal_matrix_bytes > config.normal_matrix_byte_budget:
            return {
                **common,
                "status": HighDimStatus.COMPLEXITY_GATE.value,
                "gate": "normal_matrix_byte_budget",
            }
        return {**common, "status": HighDimStatus.OK.value, "gate": "ok"}

    def _validate_config_for_basis(
        self,
        product_basis: ProductBasis,
        config: FixedTTFitConfig,
    ) -> None:
        if len(config.ranks) != product_basis.dimension + 1:
            raise ValueError(f"ranks: {HighDimStatus.INVALID_SHAPE.value}")
        if not _is_valid_sweep_order(config.sweep_order, product_basis.dimension):
            raise ValueError(f"sweep_order: {HighDimStatus.INVALID_SHAPE.value}")

    def _validate_batch_dimension(
        self,
        product_basis: ProductBasis,
        samples: FixedTTFitSampleBatch,
    ) -> None:
        if samples.points.shape[1] != product_basis.dimension:
            raise ValueError(f"points: {HighDimStatus.INVALID_SHAPE.value}")
        if samples.holdout_points is not None and samples.holdout_points.shape[1] != product_basis.dimension:
            raise ValueError(f"holdout_points: {HighDimStatus.INVALID_SHAPE.value}")

    def _validate_initial_cores(
        self,
        product_basis: ProductBasis,
        measure_convention: MeasureConvention,
        config: FixedTTFitConfig,
        initial_cores: Sequence[TTCore | tf.Tensor],
    ) -> tuple[TTCore, ...]:
        cores = tuple(core if isinstance(core, TTCore) else TTCore(core) for core in initial_cores)
        ftt = FunctionalTT(cores, product_basis, measure_convention)
        if ftt.rank_tuple() != config.ranks:
            raise ValueError(f"initial_cores: {HighDimStatus.INVALID_SHAPE.value}")
        return cores

    def _finalize_result(
        self,
        product_basis: ProductBasis,
        samples: FixedTTFitSampleBatch,
        config: FixedTTFitConfig,
        cores: tuple[TTCore, ...],
        branch_seed: int | str,
        measure_convention: MeasureConvention,
        initial_core_hash: str,
        initialization_rule: str,
        update_records: tuple[Mapping[str, object], ...],
        environment_rebuild_hashes: tuple[str, ...],
        status: HighDimStatus,
        termination_reason: str,
        stop_condition: str,
        fit_residual: tf.Tensor | None,
        holdout_residual: tf.Tensor | None,
    ) -> FixedTTFitResult:
        manifest = BranchManifest(
            version="fixed_tt_fit.v1",
            payload={
                "family": "FixedTTFitter",
                "scope": "fixed_branch_weighted_ridge_als_only",
                "product_basis": _product_basis_payload(product_basis),
                "measure_convention": _measure_convention_payload(measure_convention),
                "samples": _sample_payload(samples),
                "sample_hashes": _sample_hashes(samples),
                "ranks": config.ranks,
                "ridge": float(config.ridge),
                "dtype": config.dtype.name,
                "sweep_order": config.sweep_order,
                "coordinate_order": tuple(range(product_basis.dimension)),
                "max_sweeps": int(config.max_sweeps),
                "initial_core_hash": initial_core_hash,
                "initialization_rule": initialization_rule,
                "fitted_cores": tuple(core.values for core in cores),
                "per_core_update_statuses": tuple(dict(record) for record in update_records),
                "environment_rebuild_hashes": environment_rebuild_hashes,
                "stabilization_choices": _stabilization_policy_payload(config),
                "stabilization_policy": _stabilization_policy_payload(config),
                "stabilization_policy_id": config.stabilization_policy_id,
                "objective_preserving_column_scaling": True,
                "column_scale_floor": float(config.column_scale_floor),
                "transformed_ridge_rule": _TRANSFORMED_RIDGE_RULE,
                "condition_number_gate_target": _CONDITION_GATE_TARGET,
                "original_unscaled_normal_condition_role": _UNSCALED_CONDITION_ROLE,
                "complexity_budgets": _config_budget_payload(config),
                "solver_backend": config.solver_backend,
                "deterministic_seed": branch_seed,
                "status": status.value,
                "termination_reason": termination_reason,
                "stop_condition_triggered": stop_condition,
                "fit_residual": fit_residual,
                "holdout_residual": holdout_residual,
                "what_is_not_claimed": (
                    "adaptive_tt_cross",
                    "rank_adaptation",
                    "filtering",
                    "derivative_correctness",
                ),
            },
        )
        branch_identity = BranchIdentity(manifest=manifest, hash=manifest.sha256())
        fitted_tt = FunctionalTT(
            cores,
            product_basis,
            measure_convention,
            complexity_budget=_evaluation_budget_from_fit_config(config),
            branch_identity=branch_identity,
        )
        diagnostics = {
            "status": status.value,
            "termination_reason": termination_reason,
            "stop_condition_triggered": stop_condition,
            "rank_tuple": config.ranks,
            "ridge": float(config.ridge),
            "sweep_count": int(config.max_sweeps),
            "sweep_order": config.sweep_order,
            "initialization_rule": initialization_rule,
            "coordinate_order": tuple(range(product_basis.dimension)),
            "coordinate_order_sensitivity_reported": True,
            "environment_rebuild_count": len(environment_rebuild_hashes),
            "environment_rebuild_hashes": environment_rebuild_hashes,
            "branch_hash": branch_identity.hash.value,
            "fit_residual": fit_residual,
            "holdout_residual": holdout_residual,
            "fixed_branch_only": True,
            "solver_backend": config.solver_backend,
            "solver_mode": _SOLVER_MODE,
            "stabilization_policy": _stabilization_policy_payload(config),
            "stabilization_diagnostics_summary": _stabilization_diagnostics_summary(
                update_records
            ),
            "stabilization_policy_id": config.stabilization_policy_id,
            "objective_preserving_column_scaling": True,
            "column_scale_floor": float(config.column_scale_floor),
            "transformed_ridge_rule": _TRANSFORMED_RIDGE_RULE,
            "condition_number_gate_target": _CONDITION_GATE_TARGET,
            "original_unscaled_normal_condition_role": _UNSCALED_CONDITION_ROLE,
            "scale_floor_rule": _SCALE_FLOOR_RULE,
        }
        return FixedTTFitResult(
            fitted_tt=fitted_tt,
            status=status,
            termination_reason=termination_reason,
            stop_condition_triggered=stop_condition,
            fit_residual=fit_residual,
            holdout_residual=holdout_residual,
            branch_identity=branch_identity,
            core_update_statuses=update_records,
            diagnostics=diagnostics,
        )


def _validate_sample_tensors(
    name: str,
    points: tf.Tensor,
    targets: tf.Tensor,
    weights: tf.Tensor,
) -> None:
    assert_tf_float64(f"{name}.points", points)
    assert_tf_float64(f"{name}.target_values", targets)
    assert_tf_float64(f"{name}.weights", weights)
    if points.shape.rank != 2 or targets.shape.rank != 1 or weights.shape.rank != 1:
        raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")
    if points.shape[0] is None or targets.shape[0] is None or weights.shape[0] is None:
        raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")
    if int(points.shape[0]) != int(targets.shape[0]) or int(points.shape[0]) != int(weights.shape[0]):
        raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")
    if not bool(tf.reduce_all(tf.math.is_finite(points)).numpy()):
        raise ValueError(f"{name}.points: {HighDimStatus.NONFINITE_VALUE.value}")
    if not bool(tf.reduce_all(tf.math.is_finite(targets)).numpy()):
        raise ValueError(f"{name}.target_values: {HighDimStatus.NONFINITE_VALUE.value}")
    if not bool(tf.reduce_all(tf.math.is_finite(weights)).numpy()):
        raise ValueError(f"{name}.weights: {HighDimStatus.NONFINITE_VALUE.value}")
    if bool(tf.reduce_any(weights < 0.0).numpy()):
        raise ValueError(f"{name}.weights: {HighDimStatus.NONFINITE_VALUE.value}")
    if bool((tf.reduce_sum(weights) <= 0.0).numpy()):
        raise ValueError(f"{name}.weights: {HighDimStatus.NONFINITE_VALUE.value}")


def _core_matrices(
    product_basis: ProductBasis,
    points: tf.Tensor,
    cores: Sequence[TTCore],
) -> tuple[tf.Tensor, ...]:
    matrices = []
    for axis, core in enumerate(cores):
        basis_values = product_basis.evaluate_axis(axis, points[:, axis])
        matrices.append(tf.einsum("nl,alb->nab", basis_values, core.values))
    return tuple(matrices)


def _left_environments(matrices: Sequence[tf.Tensor]) -> tuple[tf.Tensor, ...]:
    n_rows = tf.shape(matrices[0])[0]
    environments = [tf.ones([n_rows, 1], dtype=tf.float64)]
    for matrix in matrices[:-1]:
        environments.append(tf.einsum("na,nab->nb", environments[-1], matrix))
    return tuple(environments)


def _right_environments(matrices: Sequence[tf.Tensor]) -> tuple[tf.Tensor, ...]:
    n_rows = tf.shape(matrices[0])[0]
    environments = [None] * len(matrices)
    accumulator = tf.ones([n_rows, 1], dtype=tf.float64)
    for axis in range(len(matrices) - 1, -1, -1):
        environments[axis] = accumulator
        if axis > 0:
            accumulator = tf.einsum("nab,nb->na", matrices[axis], accumulator)
    return tuple(environments)


def _normal_equations(
    design: tf.Tensor,
    target_values: tf.Tensor,
    weights: tf.Tensor,
    ridge: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    weighted_design = design * tf.reshape(weights, [-1, 1])
    normal = tf.matmul(design, weighted_design, transpose_a=True)
    normal = normal + tf.cast(ridge, tf.float64) * tf.eye(
        int(design.shape[1]),
        dtype=tf.float64,
    )
    rhs = tf.linalg.matvec(design, weights * target_values, transpose_a=True)
    return normal, rhs


def _stabilization_policy_payload(config: FixedTTFitConfig) -> Mapping[str, object]:
    return {
        "stabilization_policy_id": config.stabilization_policy_id,
        "solver_backend": config.solver_backend,
        "solver_mode": _SOLVER_MODE,
        "objective_preserving_column_scaling": True,
        "column_scale_floor": float(config.column_scale_floor),
        "column_scale_floor_rule": _SCALE_FLOOR_RULE,
        "scale_floor_rule": _SCALE_FLOOR_RULE,
        "transformed_ridge_rule": _TRANSFORMED_RIDGE_RULE,
        "condition_number_gate_target": _CONDITION_GATE_TARGET,
        "original_unscaled_normal_condition_role": _UNSCALED_CONDITION_ROLE,
        "ridge": float(config.ridge),
        "ridge_coordinate_system": "u_coordinates",
        "ridge_metric_coordinate_system": "scaled_z_coordinates",
    }


def _stabilization_diagnostics_summary(
    records: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    records_tuple = tuple(records)
    scaled_conditions: list[float] = []
    scaled_condition_inf = False
    unscaled_conditions: list[float] = []
    unscaled_condition_inf = False
    scale_mins: list[float] = []
    scale_maxes: list[float] = []
    scale_spreads: list[float] = []
    column_hashes = []
    zero_counts = []
    ridge_metric_mins: list[float] = []
    ridge_metric_maxes: list[float] = []
    first_policy = None
    for record in records_tuple:
        condition = record.get("transformed_system_condition_number")
        if isinstance(condition, (int, float)):
            scaled_conditions.append(float(condition))
        elif condition == "inf":
            scaled_condition_inf = True
        unscaled_condition = record.get("unscaled_normal_condition_number")
        if isinstance(unscaled_condition, (int, float)):
            unscaled_conditions.append(float(unscaled_condition))
        elif unscaled_condition == "inf":
            unscaled_condition_inf = True
        for key, target in (
            ("column_scale_min", scale_mins),
            ("column_scale_max", scale_maxes),
            ("column_scale_spread", scale_spreads),
        ):
            value = record.get(key)
            if isinstance(value, (int, float)):
                target.append(float(value))
        column_hash = record.get("column_scale_hash")
        if column_hash is not None:
            column_hashes.append(str(column_hash))
        zero_count = record.get("raw_column_norm_zero_count")
        if isinstance(zero_count, int):
            zero_counts.append(int(zero_count))
        ridge_summary = record.get("ridge_metric_summary")
        if isinstance(ridge_summary, Mapping):
            min_diag = ridge_summary.get("min_diagonal")
            max_diag = ridge_summary.get("max_diagonal")
            if isinstance(min_diag, (int, float)):
                ridge_metric_mins.append(float(min_diag))
            if isinstance(max_diag, (int, float)):
                ridge_metric_maxes.append(float(max_diag))
        if first_policy is None and isinstance(record.get("stabilization_policy"), Mapping):
            first_policy = dict(record["stabilization_policy"])
    return {
        "available": bool(records_tuple),
        "record_count": len(records_tuple),
        "stabilized_record_count": len(column_hashes),
        "stabilization_policy": first_policy,
        "transformed_system_condition_number_max": (
            "inf"
            if scaled_condition_inf
            else max(scaled_conditions)
            if scaled_conditions
            else None
        ),
        "original_unscaled_normal_condition_number_max": (
            "inf"
            if unscaled_condition_inf
            else max(unscaled_conditions)
            if unscaled_conditions
            else None
        ),
        "original_unscaled_normal_condition_role": _UNSCALED_CONDITION_ROLE,
        "condition_number_gate_target": _CONDITION_GATE_TARGET,
        "column_scale_min": min(scale_mins) if scale_mins else None,
        "column_scale_max": max(scale_maxes) if scale_maxes else None,
        "column_scale_spread_max": max(scale_spreads) if scale_spreads else None,
        "column_scale_hashes": tuple(column_hashes),
        "raw_column_norm_zero_count_total": sum(zero_counts),
        "ridge_metric_summary": {
            "transformed_ridge_rule": _TRANSFORMED_RIDGE_RULE,
            "coordinate_system": "scaled_z_coordinates",
            "min_diagonal": min(ridge_metric_mins) if ridge_metric_mins else None,
            "max_diagonal": max(ridge_metric_maxes) if ridge_metric_maxes else None,
        },
    }


def _weighted_column_scales(
    design: tf.Tensor,
    weights: tf.Tensor,
    column_scale_floor: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    design = tf.convert_to_tensor(design, dtype=tf.float64)
    weights = tf.convert_to_tensor(weights, dtype=tf.float64)
    raw_norms = tf.sqrt(
        tf.reduce_sum(
            tf.reshape(weights, [-1, 1]) * tf.square(design),
            axis=0,
        )
    )
    eps = tf.constant(float(column_scale_floor), dtype=tf.float64)
    max_norm = tf.reduce_max(raw_norms)
    float64_eps = tf.constant(_FLOAT64_EPS, dtype=tf.float64)
    floor_from_max = tf.sqrt(float64_eps) * max_norm
    scale_floor = tf.maximum(floor_from_max, eps)
    scales = tf.maximum(raw_norms, scale_floor)
    return scales, raw_norms, scale_floor


def _solve_scaled_augmented_ridge(
    *,
    design: tf.Tensor,
    target_values: tf.Tensor,
    weights: tf.Tensor,
    ridge: float,
    column_scale_floor: float = _DEFAULT_COLUMN_SCALE_FLOOR,
    stabilization_policy_id: str = _STABILIZATION_POLICY_ID,
    solver_backend: str = _SOLVER_BACKEND,
) -> _ScaledAugmentedSolveResult:
    design = tf.convert_to_tensor(design, dtype=tf.float64)
    target_values = tf.convert_to_tensor(target_values, dtype=tf.float64)
    weights = tf.convert_to_tensor(weights, dtype=tf.float64)
    if design.shape.rank != 2 or target_values.shape.rank != 1 or weights.shape.rank != 1:
        raise ValueError("scaled augmented solve requires matrix/vector inputs")
    if int(design.shape[0]) != int(target_values.shape[0]) or int(design.shape[0]) != int(weights.shape[0]):
        raise ValueError("scaled augmented solve row mismatch")
    if not bool(
        tf.reduce_all(tf.math.is_finite(design)).numpy()
        and tf.reduce_all(tf.math.is_finite(target_values)).numpy()
        and tf.reduce_all(tf.math.is_finite(weights)).numpy()
    ):
        raise ValueError("scaled augmented solve nonfinite input")
    if bool(tf.reduce_any(weights < 0.0).numpy()):
        raise ValueError("scaled augmented solve negative weights")
    ridge_value = float(ridge)
    if ridge_value < 0.0 or not math.isfinite(ridge_value):
        raise ValueError("scaled augmented solve ridge must be finite nonnegative")
    floor_value = float(column_scale_floor)
    if floor_value <= 0.0 or not math.isfinite(floor_value):
        raise ValueError("scaled augmented solve column scale floor must be finite positive")
    policy_id = str(stabilization_policy_id)
    if not policy_id.strip():
        raise ValueError("scaled augmented solve policy id must be nonempty")
    if str(solver_backend) != _SOLVER_BACKEND:
        raise ValueError(f"scaled augmented solve backend must be {_SOLVER_BACKEND!r}")

    column_scales, raw_column_norms, scale_floor = _weighted_column_scales(
        design,
        weights,
        floor_value,
    )
    scaled_design = design / tf.reshape(column_scales, [1, -1])
    sqrt_weights = tf.sqrt(weights)
    weighted_scaled_design = scaled_design * tf.reshape(sqrt_weights, [-1, 1])
    weighted_target = target_values * sqrt_weights
    n_cols = int(design.shape[1])
    ridge_sqrt = tf.sqrt(tf.constant(ridge_value, dtype=tf.float64))
    ridge_block = tf.linalg.diag(ridge_sqrt / column_scales)
    augmented_matrix = tf.concat([weighted_scaled_design, ridge_block], axis=0)
    augmented_rhs = tf.concat(
        [
            tf.reshape(weighted_target, [-1, 1]),
            tf.zeros([n_cols, 1], dtype=tf.float64),
        ],
        axis=0,
    )
    if not bool(
        tf.reduce_all(tf.math.is_finite(augmented_matrix)).numpy()
        and tf.reduce_all(tf.math.is_finite(augmented_rhs)).numpy()
    ):
        raise ValueError("scaled augmented solve nonfinite augmented system")
    v_solution = _stable_overdetermined_lstsq(augmented_matrix, augmented_rhs)[:, 0]
    solution = v_solution / column_scales
    singular_values = tf.linalg.svd(augmented_matrix, compute_uv=False)
    condition_number = _condition_number_from_singular_values(singular_values)
    min_scale = float(tf.reduce_min(column_scales).numpy())
    max_scale = float(tf.reduce_max(column_scales).numpy())
    scale_spread = float("inf") if min_scale <= 0.0 else max_scale / min_scale
    ridge_metric = ridge_value / tf.square(column_scales)
    ridge_metric_min = float(tf.reduce_min(ridge_metric).numpy())
    ridge_metric_max = float(tf.reduce_max(ridge_metric).numpy())
    diagnostics = {
        "stabilization_policy_id": policy_id,
        "stabilization_policy": {
            "stabilization_policy_id": policy_id,
            "solver_backend": _SOLVER_BACKEND,
            "solver_mode": _SOLVER_MODE,
            "objective_preserving_column_scaling": True,
            "column_scale_floor": floor_value,
            "column_scale_floor_rule": _SCALE_FLOOR_RULE,
            "transformed_ridge_rule": _TRANSFORMED_RIDGE_RULE,
            "condition_number_gate_target": _CONDITION_GATE_TARGET,
            "original_unscaled_normal_condition_role": _UNSCALED_CONDITION_ROLE,
            "ridge_coordinate_system": "u_coordinates",
            "ridge_metric_coordinate_system": "scaled_z_coordinates",
        },
        "solver_backend": _SOLVER_BACKEND,
        "solver_mode": _SOLVER_MODE,
        "objective_preserving_column_scaling": True,
        "scale_floor": float(scale_floor.numpy()),
        "scale_floor_rule": _SCALE_FLOOR_RULE,
        "column_scale_floor": floor_value,
        "column_scale_min": min_scale,
        "column_scale_max": max_scale,
        "column_scale_spread": scale_spread,
        "column_scale_hash": _hash_tensor_or_none(
            "fixed_tt_fit_column_scales.v1",
            column_scales,
        ),
        "raw_column_norm_min": float(tf.reduce_min(raw_column_norms).numpy()),
        "raw_column_norm_max": float(tf.reduce_max(raw_column_norms).numpy()),
        "raw_column_norm_zero_count": int(
            tf.reduce_sum(tf.cast(raw_column_norms == 0.0, tf.int32)).numpy()
        ),
        "ridge_metric_summary": {
            "ridge": ridge_value,
            "transformed_ridge_rule": _TRANSFORMED_RIDGE_RULE,
            "coordinate_system": "scaled_z_coordinates",
            "min_diagonal": ridge_metric_min,
            "max_diagonal": ridge_metric_max,
        },
        "transformed_system_condition_number": _finite_number_or_text(condition_number),
        "scaled_augmented_condition_number": _finite_number_or_text(condition_number),
        "scaled_augmented_rows": int(augmented_matrix.shape[0]),
        "scaled_augmented_cols": int(augmented_matrix.shape[1]),
        "condition_number_gate_target": _CONDITION_GATE_TARGET,
        "original_unscaled_normal_condition_role": _UNSCALED_CONDITION_ROLE,
        "transformed_ridge_rule": _TRANSFORMED_RIDGE_RULE,
        "ridge_coordinate_system": "u_coordinates",
        "nonclaims": (
            "stable solve is not a Phase 6 diagnostic pass",
            "stable solve is fixed_hmc_adaptation not source_faithful",
        ),
    }
    return _ScaledAugmentedSolveResult(
        solution=solution,
        column_scales=column_scales,
        raw_column_norms=raw_column_norms,
        scale_floor=scale_floor,
        scaled_augmented_matrix=augmented_matrix,
        scaled_augmented_rhs=augmented_rhs,
        scaled_augmented_condition_number=condition_number,
        diagnostics=diagnostics,
    )


@tf.custom_gradient
def _stable_overdetermined_lstsq(
    matrix: tf.Tensor,
    rhs: tf.Tensor,
) -> tuple[tf.Tensor, object]:
    """Use the stable least-squares primal with its exact full-rank pullback."""

    a = tf.convert_to_tensor(matrix, dtype=tf.float64)
    b = tf.convert_to_tensor(rhs, dtype=tf.float64)
    solution = tf.linalg.lstsq(a, b, fast=False)

    def grad(upstream: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        g = tf.convert_to_tensor(upstream, dtype=tf.float64)
        _, upper = tf.linalg.qr(a, full_matrices=False)
        intermediate = tf.linalg.triangular_solve(
            upper,
            g,
            lower=False,
            adjoint=True,
        )
        z = tf.linalg.triangular_solve(
            upper,
            intermediate,
            lower=False,
        )
        solution_z_t = tf.linalg.matmul(solution, z, transpose_b=True)
        symmetric = solution_z_t + tf.linalg.matrix_transpose(solution_z_t)
        grad_a = -tf.linalg.matmul(a, symmetric) + tf.linalg.matmul(
            b,
            z,
            transpose_b=True,
        )
        grad_b = tf.linalg.matmul(a, z)
        return grad_a, grad_b

    return solution, grad


def _condition_number(matrix: tf.Tensor) -> float:
    singular_values = tf.linalg.svd(matrix, compute_uv=False)
    return _condition_number_from_singular_values(singular_values)


def _condition_number_from_singular_values(singular_values: tf.Tensor) -> float:
    singular_values = tf.reshape(tf.convert_to_tensor(singular_values, dtype=tf.float64), [-1])
    min_value = float(tf.reduce_min(singular_values).numpy())
    max_value = float(tf.reduce_max(singular_values).numpy())
    if min_value <= 0.0 or not math.isfinite(min_value) or not math.isfinite(max_value):
        return float("inf")
    return max_value / min_value


def _weighted_rms_residual(
    predicted: tf.Tensor,
    target: tf.Tensor,
    weights: tf.Tensor,
) -> tf.Tensor:
    denominator = tf.reduce_sum(weights)
    residual = predicted - target
    return tf.sqrt(tf.reduce_sum(weights * tf.square(residual)) / denominator)


def _failed_update_record(
    status: HighDimStatus,
    core_index: int,
    sweep_index: int,
    termination_reason: str,
) -> Mapping[str, object]:
    return {
        "status": status.value,
        "core_index": core_index,
        "sweep_index": sweep_index,
        "termination_reason": termination_reason,
        "stop_condition_triggered": status.value,
    }


def _evaluation_budget_from_fit_config(config: FixedTTFitConfig) -> ComplexityBudget:
    max_bytes = max(
        int(config.dense_matrix_byte_budget),
        int(config.normal_matrix_byte_budget),
        tf.float64.size,
    )
    return ComplexityBudget(max_elements=max_bytes // tf.float64.size, max_bytes=max_bytes)


def _measure_convention_payload(convention: MeasureConvention) -> Mapping[str, object]:
    return {
        "density_measure": convention.density_measure.value,
        "mass_measure": convention.mass_measure.value,
        "reference_weight_name": convention.reference_weight_name,
        "physical_coordinate_name": convention.physical_coordinate_name,
        "reference_coordinate_name": convention.reference_coordinate_name,
        "dtype_name": convention.dtype_name,
    }


def _basis_payload(basis) -> Mapping[str, object]:
    if hasattr(basis, "manifest_payload"):
        payload = dict(basis.manifest_payload())
        if isinstance(basis, LegendreBasis1D) and isinstance(basis.domain, BoundedInterval):
            payload.update(
                {
                    "family": "LegendreBasis1D",
                    "left": basis.domain.left,
                    "right": basis.domain.right,
                    "reference_measure": "UniformReferenceMeasure",
                    "max_degree": int(basis.max_degree),
                    "normalized": bool(basis.normalized),
                }
            )
        return payload
    return {
        "family": type(basis).__name__,
        "dtype": basis.dtype.name,
        "basis_dim": int(basis.basis_dim),
    }


def _product_basis_payload(product_basis: ProductBasis) -> Mapping[str, object]:
    return {
        "family": "ProductBasis",
        "dimension": int(product_basis.dimension),
        "basis_dim_tuple": product_basis.basis_dim_tuple(),
        "convention": _measure_convention_payload(product_basis.convention),
        "bases": tuple(_basis_payload(basis) for basis in product_basis.bases),
    }


def _sample_payload(samples: FixedTTFitSampleBatch) -> Mapping[str, object]:
    return {
        "points": samples.points,
        "target_values": samples.target_values,
        "weights": samples.weights,
        "holdout_points": samples.holdout_points,
        "holdout_values": samples.holdout_values,
        "holdout_weights": samples.holdout_weights,
    }


def _sample_hashes(samples: FixedTTFitSampleBatch) -> Mapping[str, object]:
    return {
        "sample_points_hash": _hash_tensor_or_none("sample_points_hash.v1", samples.points),
        "target_values_hash": _hash_tensor_or_none("target_values_hash.v1", samples.target_values),
        "weights_hash": _hash_tensor_or_none("weights_hash.v1", samples.weights),
        "holdout_hash": BranchManifest(
            version="holdout_hash.v1",
            payload={
                "points": samples.holdout_points,
                "values": samples.holdout_values,
                "weights": samples.holdout_weights,
            },
        ).sha256().value,
    }


def _config_budget_payload(config: FixedTTFitConfig) -> Mapping[str, object]:
    return {
        "row_budget": int(config.row_budget),
        "column_budget": int(config.column_budget),
        "dense_matrix_byte_budget": int(config.dense_matrix_byte_budget),
        "normal_matrix_byte_budget": int(config.normal_matrix_byte_budget),
    }


def _is_valid_sweep_order(sweep_order: tuple[int, ...], dimension: int) -> bool:
    dim = int(dimension)
    order = tuple(int(axis) for axis in sweep_order)
    if dim <= 0 or not order:
        return False
    if any(axis < 0 or axis >= dim for axis in order):
        return False
    legacy_permutation = tuple(range(dim))
    if len(order) == dim and sorted(order) == list(legacy_permutation):
        return True
    canonical_alternating = legacy_permutation + tuple(reversed(legacy_permutation))
    return order == canonical_alternating


def _hash_tensor_or_none(version: str, value: tf.Tensor | None) -> str | None:
    if value is None:
        return None
    return BranchManifest(version=version, payload={"value": value}).sha256().value


def _hash_tensors(version: str, tensors: Sequence[tf.Tensor]) -> str:
    return BranchManifest(version=version, payload={"tensors": tuple(tensors)}).sha256().value


def _finite_number_or_text(value: float) -> float | str:
    return float(value) if math.isfinite(value) else "inf"
