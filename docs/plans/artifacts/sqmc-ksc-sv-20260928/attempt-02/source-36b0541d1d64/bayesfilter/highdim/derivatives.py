"""Fixed-branch derivative contracts for P37-M6 highdim filters."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.diagnostics import (
    HighDimStatus,
    HighDimValidationResult,
    freeze_mapping,
)
from bayesfilter.highdim.fixed_branch import BranchHash, BranchIdentity, BranchManifest
from bayesfilter.highdim.squared_tt import SquaredTTDensity
from bayesfilter.highdim.tt import FunctionalTT, TTCore
from bayesfilter.highdim.validation import FiniteDifferenceRowStatus


_M6_ROW_DECISIONS = frozenset(
    {"DERIVATIVE_PASSED", "DERIVATIVE_BLOCKED", "DERIVATIVE_NOT_APPLICABLE"}
)
_M6_STABLE_WINDOW_STATUSES = frozenset(
    {
        "PASS_DECREASING_OR_ROUNDOFF_PLATEAU",
        "BLOCKED_VALUE_OR_BRANCH_CONTRACT",
        "FAIL_NO_STABLE_WINDOW",
        "NOT_APPLICABLE",
    }
)
_M6_REQUIRED_NONCLAIM_TERMS = (
    "adaptive",
    "score API",
    "HMC",
    "DSGE",
    "GPU",
    "general nonlinear",
)


def _require_nonempty_text(name: str, value: object) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be nonempty")
    return normalized


def _normalize_text_tuple(name: str, values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{name} must contain nonempty strings")
    return normalized


@dataclass(frozen=True)
class FixedBranchDerivativeConfig:
    """Declared fixed-branch derivative policy."""

    parameter_indices: tuple[int, ...]
    finite_difference_h: tuple[float, ...] = (1e-2, 3e-3, 1e-3, 3e-4)
    derivative_ridge_floor: float = 1e-12
    solve_condition_number_veto: float = 1e12
    allow_parameter_dependent_coordinate_map: bool = False
    allow_moving_basis: bool = False
    dtype: tf.DType = tf.float64

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_indices", tuple(int(i) for i in self.parameter_indices))
        object.__setattr__(self, "finite_difference_h", tuple(float(h) for h in self.finite_difference_h))
        if self.dtype != tf.float64:
            raise ValueError("FixedBranchDerivativeConfig requires tf.float64")
        if not self.parameter_indices:
            raise ValueError("parameter_indices must be nonempty")
        if any(index < 0 for index in self.parameter_indices):
            raise ValueError("parameter_indices must be nonnegative")
        if any((not math.isfinite(h)) or h <= 0.0 for h in self.finite_difference_h):
            raise ValueError("finite_difference_h values must be positive finite")
        for name in ("derivative_ridge_floor", "solve_condition_number_veto"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive finite")

    def unsupported_status(self) -> HighDimValidationResult:
        """Return the deterministic unsupported status for moving-basis paths."""

        if self.allow_moving_basis:
            return HighDimValidationResult(
                status=HighDimStatus.UNSUPPORTED_MOVING_BASIS_DERIVATIVE,
                message="moving-basis derivatives are unsupported in P37-M6",
                diagnostics={
                    "allow_moving_basis": True,
                    "fixed_branch_only": True,
                },
            )
        return HighDimValidationResult(
            status=HighDimStatus.OK,
            message="fixed branch derivative policy supported",
            diagnostics={
                "allow_moving_basis": False,
                "allow_parameter_dependent_coordinate_map": bool(
                    self.allow_parameter_dependent_coordinate_map
                ),
            },
        )


@dataclass(frozen=True)
class CoreDerivativeState:
    """One TT core and its derivative under a fixed branch."""

    core_index: int
    core: TTCore
    dot_core: TTCore
    pre_update_core_hash: str
    post_update_core_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.core, TTCore) or not isinstance(self.dot_core, TTCore):
            raise TypeError("core and dot_core must be TTCore")
        if self.core.values.shape != self.dot_core.values.shape:
            raise ValueError(f"dot_core: {HighDimStatus.INVALID_SHAPE.value}")
        object.__setattr__(self, "core_index", int(self.core_index))
        for name in ("pre_update_core_hash", "post_update_core_hash"):
            value = str(getattr(self, name))
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SweepDerivativeDiagnostics:
    """Diagnostics for one differentiated fixed-design core solve."""

    time_index: int
    sweep_index: int
    sweep_direction: str
    core_index: int
    status: HighDimStatus
    condition_number: float
    normal_matrix_hash: str
    rhs_hash: str
    coefficient_hash: str
    environment_provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.status, HighDimStatus):
            raise TypeError("status must be a HighDimStatus")
        object.__setattr__(self, "time_index", int(self.time_index))
        object.__setattr__(self, "sweep_index", int(self.sweep_index))
        object.__setattr__(self, "core_index", int(self.core_index))
        object.__setattr__(self, "sweep_direction", str(self.sweep_direction))
        object.__setattr__(self, "condition_number", float(self.condition_number))
        object.__setattr__(self, "environment_provenance", freeze_mapping(self.environment_provenance))


@dataclass(frozen=True)
class FixedBranchReplayTape:
    """Replay tape bound to a pre-existing fixed-branch identity."""

    version: str
    branch_identity: BranchIdentity
    entries: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        if not str(self.version).strip():
            raise ValueError("version must be nonempty")
        if not isinstance(self.branch_identity, BranchIdentity):
            raise TypeError("branch_identity must be a BranchIdentity")
        object.__setattr__(
            self,
            "entries",
            tuple(MappingProxyType({str(k): v for k, v in entry.items()}) for entry in self.entries),
        )

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "version": self.version,
            "branch_hash": self.branch_identity.hash.value,
            "entry_count": len(self.entries),
            "entries": tuple(dict(entry) for entry in self.entries),
            "moving_basis_supported": False,
            "fixed_branch_only": True,
        }

    def manifest(self) -> BranchManifest:
        return BranchManifest("fixed_branch_replay_tape.v1", self.manifest_payload())

    def sha256(self) -> BranchHash:
        return self.manifest().sha256()

    def assert_matches_branch(self, branch_identity: BranchIdentity) -> None:
        if branch_identity != self.branch_identity:
            raise ValueError(HighDimStatus.REPLAY_TAPE_MISMATCH.value)


@dataclass(frozen=True)
class FiniteDifferenceRow:
    """One finite-difference row with fixed-branch compatibility hashes."""

    parameter_index: int
    h: tf.Tensor
    value_plus: tf.Tensor
    value_minus: tf.Tensor
    branch_hash_plus: str
    branch_hash_minus: str
    branch_hash_base: str
    centered_difference: tf.Tensor
    analytic_gradient: tf.Tensor
    abs_error: tf.Tensor
    rel_error: tf.Tensor
    row_status: FiniteDifferenceRowStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_index", int(self.parameter_index))
        for name in (
            "h",
            "value_plus",
            "value_minus",
            "centered_difference",
            "analytic_gradient",
            "abs_error",
            "rel_error",
        ):
            value = tf.convert_to_tensor(getattr(self, name), dtype=tf.float64)
            if value.shape.rank != 0:
                raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")
            object.__setattr__(self, name, value)
        if not isinstance(self.row_status, FiniteDifferenceRowStatus):
            raise TypeError("row_status must be a FiniteDifferenceRowStatus")


@dataclass(frozen=True)
class FiniteDifferenceTable:
    """Finite-difference table that excludes invalid rows from summaries."""

    rows: tuple[FiniteDifferenceRow, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))

    def valid_rows(self) -> tuple[FiniteDifferenceRow, ...]:
        return tuple(row for row in self.rows if row.row_status is FiniteDifferenceRowStatus.VALID)

    def max_abs_error(self) -> tf.Tensor:
        rows = self.valid_rows()
        if not rows:
            return tf.constant(float("nan"), dtype=tf.float64)
        return tf.reduce_max(tf.stack([row.abs_error for row in rows]))


@dataclass(frozen=True)
class FixedDesignLSDerivativeResult:
    """Derivative of one fixed-design weighted ridge least-squares solve."""

    dot_coefficients: tf.Tensor
    dot_normal_matrix: tf.Tensor
    dot_rhs: tf.Tensor
    condition_number: float
    status: HighDimStatus
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.status, HighDimStatus):
            raise TypeError("status must be a HighDimStatus")
        object.__setattr__(self, "dot_coefficients", tf.convert_to_tensor(self.dot_coefficients, dtype=tf.float64))
        object.__setattr__(self, "dot_normal_matrix", tf.convert_to_tensor(self.dot_normal_matrix, dtype=tf.float64))
        object.__setattr__(self, "dot_rhs", tf.convert_to_tensor(self.dot_rhs, dtype=tf.float64))
        object.__setattr__(self, "condition_number", float(self.condition_number))
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))


@dataclass(frozen=True)
class FixedBranchScoreResult:
    """Fixed-branch score result with replay and finite-difference evidence."""

    log_likelihood: tf.Tensor
    score: tf.Tensor
    branch_identity: BranchIdentity
    replay_tape_hash: str
    finite_difference_table: FiniteDifferenceTable
    status: HighDimStatus
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.branch_identity, BranchIdentity):
            raise TypeError("branch_identity must be a BranchIdentity")
        if not isinstance(self.finite_difference_table, FiniteDifferenceTable):
            raise TypeError("finite_difference_table must be a FiniteDifferenceTable")
        if not isinstance(self.status, HighDimStatus):
            raise TypeError("status must be a HighDimStatus")
        object.__setattr__(self, "log_likelihood", tf.convert_to_tensor(self.log_likelihood, dtype=tf.float64))
        object.__setattr__(self, "score", tf.convert_to_tensor(self.score, dtype=tf.float64))
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))


@dataclass(frozen=True)
class P30FixedBranchGradientTableManifest:
    """P37-M6 first-gate manifest for a fixed-branch derivative table."""

    phase_id: str
    model_row: str
    value_result_artifact: str
    value_prerequisite_status: str
    perturbation_coordinate: str
    parameterization: str
    finite_difference_h: tuple[float, ...]
    tolerance_policy: Mapping[str, object]
    branch_policy: str
    finite_difference_table: FiniteDifferenceTable
    stable_window_status: str
    row_decision: str
    non_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        phase_id = _require_nonempty_text("phase_id", self.phase_id)
        if phase_id != "P37-M6":
            raise ValueError("phase_id must be P37-M6")
        for field_name in (
            "model_row",
            "value_result_artifact",
            "value_prerequisite_status",
            "perturbation_coordinate",
            "parameterization",
            "branch_policy",
            "stable_window_status",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_nonempty_text(field_name, getattr(self, field_name)),
            )
        row_decision = _require_nonempty_text("row_decision", self.row_decision)
        if row_decision not in _M6_ROW_DECISIONS:
            raise ValueError(
                "row_decision must be DERIVATIVE_PASSED, DERIVATIVE_BLOCKED, "
                "or DERIVATIVE_NOT_APPLICABLE"
            )
        if not isinstance(self.finite_difference_table, FiniteDifferenceTable):
            raise TypeError("finite_difference_table must be a FiniteDifferenceTable")
        finite_difference_h = tuple(float(value) for value in self.finite_difference_h)
        if any((not math.isfinite(value)) or value <= 0.0 for value in finite_difference_h):
            raise ValueError("finite_difference_h values must be positive finite")
        if row_decision == "DERIVATIVE_PASSED" and not finite_difference_h:
            raise ValueError("DERIVATIVE_PASSED requires a nonempty finite_difference_h ladder")
        row_h = tuple(float(row.h.numpy()) for row in self.finite_difference_table.rows)
        if row_h != finite_difference_h:
            raise ValueError("finite_difference_h must match finite_difference_table row h values")
        tolerance_policy = {str(key): value for key, value in self.tolerance_policy.items()}
        for key in ("abs_error_tolerance", "stable_window_policy"):
            if key not in tolerance_policy:
                raise ValueError(f"tolerance_policy missing {key}")
        abs_tolerance = float(tolerance_policy["abs_error_tolerance"])
        if not math.isfinite(abs_tolerance) or abs_tolerance <= 0.0:
            raise ValueError("abs_error_tolerance must be positive finite")

        valid_rows = self.finite_difference_table.valid_rows()
        all_rows_valid = len(valid_rows) == len(self.finite_difference_table.rows)
        has_stable_window = self._has_adjacent_window_below_tolerance(valid_rows, abs_tolerance)
        stable_window_status = self.stable_window_status
        if stable_window_status not in _M6_STABLE_WINDOW_STATUSES:
            raise ValueError("stable_window_status is not an allowed P37-M6 status")
        status_upper = self.value_prerequisite_status.upper()
        expected_stable_window_status = self._expected_stable_window_status(
            row_decision=row_decision,
            value_prerequisite_passed=status_upper.startswith("PASS"),
            all_rows_valid=all_rows_valid,
            has_rows=bool(self.finite_difference_table.rows),
            has_stable_window=has_stable_window,
        )
        if stable_window_status != expected_stable_window_status:
            raise ValueError("stable_window_status does not match computed finite-difference evidence")
        if row_decision == "DERIVATIVE_PASSED":
            if not status_upper.startswith("PASS"):
                raise ValueError("DERIVATIVE_PASSED requires passed value prerequisite")
            if not all_rows_valid:
                raise ValueError("DERIVATIVE_PASSED requires all finite-difference rows branch compatible and finite")
            if not has_stable_window:
                raise ValueError("DERIVATIVE_PASSED requires a stable finite-difference window")
            if stable_window_status != "PASS_DECREASING_OR_ROUNDOFF_PLATEAU":
                raise ValueError("DERIVATIVE_PASSED requires PASS_DECREASING_OR_ROUNDOFF_PLATEAU status")
        if row_decision in {"DERIVATIVE_BLOCKED", "DERIVATIVE_NOT_APPLICABLE"} and status_upper.startswith("PASS"):
            if valid_rows and has_stable_window:
                raise ValueError("passed derivative evidence cannot be marked blocked or not applicable")
        non_claims = _normalize_text_tuple("non_claims", self.non_claims)
        non_claim_text = " ".join(non_claims).lower()
        for term in _M6_REQUIRED_NONCLAIM_TERMS:
            if term.lower() not in non_claim_text:
                raise ValueError(f"non_claims must include {term} non-claim")

        object.__setattr__(self, "phase_id", phase_id)
        object.__setattr__(self, "finite_difference_h", finite_difference_h)
        object.__setattr__(self, "tolerance_policy", MappingProxyType(tolerance_policy))
        object.__setattr__(self, "row_decision", row_decision)
        object.__setattr__(self, "non_claims", non_claims)

    @staticmethod
    def _expected_stable_window_status(
        *,
        row_decision: str,
        value_prerequisite_passed: bool,
        all_rows_valid: bool,
        has_rows: bool,
        has_stable_window: bool,
    ) -> str:
        if row_decision == "DERIVATIVE_NOT_APPLICABLE":
            return "NOT_APPLICABLE"
        if not value_prerequisite_passed or not all_rows_valid or not has_rows:
            return "BLOCKED_VALUE_OR_BRANCH_CONTRACT"
        if not has_stable_window:
            return "FAIL_NO_STABLE_WINDOW"
        return "PASS_DECREASING_OR_ROUNDOFF_PLATEAU"

    @staticmethod
    def _has_adjacent_window_below_tolerance(
        rows: tuple[FiniteDifferenceRow, ...],
        tolerance: float,
    ) -> bool:
        if len(rows) < 2:
            return False
        ordered = sorted(rows, key=lambda row: float(row.h.numpy()), reverse=True)
        for left, right in zip(ordered[:-1], ordered[1:]):
            left_error = float(left.abs_error.numpy())
            right_error = float(right.abs_error.numpy())
            if left_error <= tolerance and right_error <= tolerance:
                if right_error <= left_error or abs(right_error - left_error) <= 0.1 * tolerance:
                    return True
        return False

    def valid_row_count(self) -> int:
        return len(self.finite_difference_table.valid_rows())


def fixed_branch_compatibility_hash(payload: Mapping[str, object]) -> str:
    """Hash branch-defining fields while excluding perturbed target values."""

    return BranchManifest("fixed_branch_fd_compatibility.v1", payload).sha256().value


def make_finite_difference_row(
    parameter_index: int,
    h: float,
    value_plus: tf.Tensor,
    value_minus: tf.Tensor,
    branch_hash_plus: str,
    branch_hash_minus: str,
    branch_hash_base: str,
    analytic_gradient: tf.Tensor,
) -> FiniteDifferenceRow:
    """Build one finite-difference row and invalidate branch mismatches."""

    h_tensor = tf.convert_to_tensor(h, dtype=tf.float64)
    plus = tf.convert_to_tensor(value_plus, dtype=tf.float64)
    minus = tf.convert_to_tensor(value_minus, dtype=tf.float64)
    analytic = tf.convert_to_tensor(analytic_gradient, dtype=tf.float64)
    centered = (plus - minus) / (2.0 * h_tensor)
    finite = bool(
        tf.reduce_all(
            tf.math.is_finite(tf.stack([h_tensor, plus, minus, centered, analytic]))
        ).numpy()
    )
    if branch_hash_plus != branch_hash_base or branch_hash_minus != branch_hash_base:
        status = FiniteDifferenceRowStatus.INVALID_BRANCH_MISMATCH
    elif not finite:
        status = FiniteDifferenceRowStatus.INVALID_NONFINITE_VALUE
    else:
        status = FiniteDifferenceRowStatus.VALID
    abs_error = tf.abs(centered - analytic)
    rel_error = abs_error / tf.maximum(tf.constant(1.0, dtype=tf.float64), tf.abs(analytic))
    return FiniteDifferenceRow(
        parameter_index=parameter_index,
        h=h_tensor,
        value_plus=plus,
        value_minus=minus,
        branch_hash_plus=branch_hash_plus,
        branch_hash_minus=branch_hash_minus,
        branch_hash_base=branch_hash_base,
        centered_difference=centered,
        analytic_gradient=analytic,
        abs_error=abs_error,
        rel_error=rel_error,
        row_status=status,
    )


def differentiate_design_matrix(
    product_basis: ProductBasis,
    points: tf.Tensor,
    cores: Sequence[TTCore],
    dot_cores: Sequence[TTCore],
    core_index: int,
) -> tf.Tensor:
    """Differentiate the fixed-design core matrix with frozen bases."""

    values = tf.convert_to_tensor(points, dtype=tf.float64)
    checked_cores = tuple(cores)
    checked_dot_cores = tuple(dot_cores)
    matrices = _core_matrices(product_basis, values, checked_cores)
    dot_matrices = _core_matrices(product_basis, values, checked_dot_cores)
    left, dot_left = _left_and_dot_environments(matrices, dot_matrices)
    right, dot_right = _right_and_dot_environments(matrices, dot_matrices)
    basis_values = product_basis.evaluate_axis(core_index, values[:, core_index])
    blocks = (
        tf.einsum("na,nl,nb->nalb", dot_left[core_index], basis_values, right[core_index])
        + tf.einsum("na,nl,nb->nalb", left[core_index], basis_values, dot_right[core_index])
    )
    core = checked_cores[core_index]
    return tf.reshape(
        blocks,
        [
            int(values.shape[0]),
            core.left_rank * core.basis_dim * core.right_rank,
        ],
    )


def fixed_design_lsq_derivative(
    design_matrix: tf.Tensor,
    target_values: tf.Tensor,
    weights: tf.Tensor,
    coefficients: tf.Tensor,
    dot_target_values: tf.Tensor,
    ridge: float,
    dot_design_matrix: tf.Tensor | None = None,
    dot_weights: tf.Tensor | None = None,
    dot_ridge: float = 0.0,
    condition_number_veto: float = 1e12,
) -> FixedDesignLSDerivativeResult:
    """Differentiate ``(A'WA + rho I)c = A'Wy`` for a fixed branch."""

    design = tf.convert_to_tensor(design_matrix, dtype=tf.float64)
    target = tf.convert_to_tensor(target_values, dtype=tf.float64)
    weights_tensor = tf.convert_to_tensor(weights, dtype=tf.float64)
    coeff = tf.convert_to_tensor(coefficients, dtype=tf.float64)
    dot_target = tf.convert_to_tensor(dot_target_values, dtype=tf.float64)
    dot_design = (
        tf.zeros_like(design)
        if dot_design_matrix is None
        else tf.convert_to_tensor(dot_design_matrix, dtype=tf.float64)
    )
    dot_w = (
        tf.zeros_like(weights_tensor)
        if dot_weights is None
        else tf.convert_to_tensor(dot_weights, dtype=tf.float64)
    )
    normal = _normal_matrix(design, weights_tensor, ridge)
    dot_normal = (
        tf.matmul(dot_design, design * weights_tensor[:, tf.newaxis], transpose_a=True)
        + tf.matmul(design, dot_design * weights_tensor[:, tf.newaxis], transpose_a=True)
        + tf.matmul(design, design * dot_w[:, tf.newaxis], transpose_a=True)
        + tf.cast(dot_ridge, tf.float64) * tf.eye(int(design.shape[1]), dtype=tf.float64)
    )
    dot_rhs = (
        tf.linalg.matvec(dot_design, weights_tensor * target, transpose_a=True)
        + tf.linalg.matvec(design, weights_tensor * dot_target, transpose_a=True)
        + tf.linalg.matvec(design, dot_w * target, transpose_a=True)
    )
    condition = _condition_number(normal)
    diagnostics = {
        "normal_matrix_hash": _hash_tensor("normal_matrix_hash.v1", normal),
        "dot_normal_matrix_hash": _hash_tensor("dot_normal_matrix_hash.v1", dot_normal),
        "rhs_hash": _hash_tensor("dot_rhs_hash.v1", dot_rhs),
        "condition_number": condition,
        "fixed_design": dot_design_matrix is None,
        "dot_weights_zero": dot_weights is None,
        "dot_ridge": float(dot_ridge),
    }
    if (not math.isfinite(condition)) or condition > condition_number_veto:
        return FixedDesignLSDerivativeResult(
            dot_coefficients=tf.zeros_like(coeff),
            dot_normal_matrix=dot_normal,
            dot_rhs=dot_rhs,
            condition_number=condition,
            status=HighDimStatus.DERIVATIVE_SOLVE_FAILURE,
            diagnostics={**diagnostics, "stop_condition_triggered": HighDimStatus.DERIVATIVE_SOLVE_FAILURE.value},
        )
    try:
        dot_coeff = tf.linalg.solve(normal, tf.reshape(dot_rhs - tf.linalg.matvec(dot_normal, coeff), [-1, 1]))[:, 0]
    except Exception:
        return FixedDesignLSDerivativeResult(
            dot_coefficients=tf.zeros_like(coeff),
            dot_normal_matrix=dot_normal,
            dot_rhs=dot_rhs,
            condition_number=condition,
            status=HighDimStatus.DERIVATIVE_SOLVE_FAILURE,
            diagnostics={**diagnostics, "stop_condition_triggered": HighDimStatus.DERIVATIVE_SOLVE_FAILURE.value},
        )
    if not bool(tf.reduce_all(tf.math.is_finite(dot_coeff)).numpy()):
        status = HighDimStatus.NONFINITE_RETAINED_DERIVATIVE
    else:
        status = HighDimStatus.OK
    return FixedDesignLSDerivativeResult(
        dot_coefficients=dot_coeff,
        dot_normal_matrix=dot_normal,
        dot_rhs=dot_rhs,
        condition_number=condition,
        status=status,
        diagnostics={**diagnostics, "stop_condition_triggered": "none" if status is HighDimStatus.OK else status.value},
    )


def tt_evaluation_derivative(
    product_basis: ProductBasis,
    points: tf.Tensor,
    cores: Sequence[TTCore],
    dot_cores: Sequence[TTCore],
) -> tf.Tensor:
    """Differentiate TT evaluation using left/right product-rule propagation."""

    values = tf.convert_to_tensor(points, dtype=tf.float64)
    matrices = _core_matrices(product_basis, values, tuple(cores))
    dot_matrices = _core_matrices(product_basis, values, tuple(dot_cores))
    vector = tf.ones([tf.shape(values)[0], 1], dtype=tf.float64)
    dot_vector = tf.zeros_like(vector)
    for matrix, dot_matrix in zip(matrices, dot_matrices):
        dot_vector = tf.einsum("na,nab->nb", dot_vector, matrix) + tf.einsum(
            "na,nab->nb",
            vector,
            dot_matrix,
        )
        vector = tf.einsum("na,nab->nb", vector, matrix)
    return tf.reshape(dot_vector, [tf.shape(values)[0]])


def squared_tt_normalizer_derivative(
    sqrt_tt: FunctionalTT,
    dot_cores: Sequence[TTCore],
) -> tf.Tensor:
    """Differentiate ``integral h(z)^2 dM(z)`` for frozen bases."""

    vector = tf.ones([1], dtype=tf.float64)
    dot_vector = tf.zeros([1], dtype=tf.float64)
    active_measure = sqrt_tt.measure_convention.mass_measure
    for axis, (core, dot_core) in enumerate(zip(sqrt_tt.cores, dot_cores)):
        mass = sqrt_tt.product_basis.bases[axis].mass_matrix(active_measure)
        matrix = _paired_core_matrix(core, core, mass)
        dot_matrix = _paired_core_matrix(dot_core, core, mass) + _paired_core_matrix(core, dot_core, mass)
        dot_vector = tf.einsum("a,ab->b", dot_vector, matrix) + tf.einsum("a,ab->b", vector, dot_matrix)
        vector = tf.einsum("a,ab->b", vector, matrix)
    return tf.reshape(dot_vector, [])


def squared_tt_log_normalizer_derivative(
    density: SquaredTTDensity,
    dot_cores: Sequence[TTCore],
) -> tf.Tensor:
    """Differentiate the squared-TT log normalizer with fixed defensive mass."""

    dot_z = squared_tt_normalizer_derivative(density.sqrt_tt, dot_cores)
    return dot_z / density.normalizer()


def retained_filter_quotient_derivative(
    numerator: tf.Tensor,
    dot_numerator: tf.Tensor,
    normalizer: tf.Tensor,
    dot_normalizer: tf.Tensor,
) -> tf.Tensor:
    """Differentiate a retained normalized quantity ``m/Z``."""

    m = tf.convert_to_tensor(numerator, dtype=tf.float64)
    dot_m = tf.convert_to_tensor(dot_numerator, dtype=tf.float64)
    z = tf.convert_to_tensor(normalizer, dtype=tf.float64)
    dot_z = tf.convert_to_tensor(dot_normalizer, dtype=tf.float64)
    return dot_m / z - m * dot_z / tf.square(z)


def replay_tape_from_filter_result(branch_identity: BranchIdentity, entries: Sequence[Mapping[str, object]]) -> FixedBranchReplayTape:
    """Build a replay tape for a realized fixed-branch result."""

    return FixedBranchReplayTape(
        version="fixed_branch_replay_tape.phase5.v1",
        branch_identity=branch_identity,
        entries=tuple(entries),
    )


def scalar_one_step_lgssm_prior_mean_score(
    observation: float = 0.2,
    mu0: float = 0.0,
    initial_variance: float = 1.0,
    observation_variance: float = 0.09,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Exact score of ``log p(y0)`` with respect to the initial mean."""

    y = tf.constant(observation, dtype=tf.float64)
    mu = tf.constant(mu0, dtype=tf.float64)
    variance = tf.constant(initial_variance + observation_variance, dtype=tf.float64)
    log_evidence = tfp.distributions.Normal(mu, tf.sqrt(variance)).log_prob(y)
    score = (y - mu) / variance
    return log_evidence, score


def scalar_two_step_lgssm_transition_score(
    transition_scale: float = 0.7,
    observations: Sequence[float] = (0.2, -0.1),
    initial_variance: float = 1.0,
    transition_variance: float = 0.25,
    observation_variance: float = 0.09,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Exact two-step scalar LGSSM score with respect to transition scale."""

    obs = tf.convert_to_tensor(observations, dtype=tf.float64)
    with tf.GradientTape() as tape:
        a = tf.Variable(transition_scale, dtype=tf.float64)
        p0 = tf.constant(initial_variance, dtype=tf.float64)
        r = tf.constant(observation_variance, dtype=tf.float64)
        q = tf.constant(transition_variance, dtype=tf.float64)
        mean = tf.constant(0.0, dtype=tf.float64)
        covariance = p0
        innovation0 = obs[0] - mean
        s0 = covariance + r
        log0 = tfp.distributions.Normal(tf.constant(0.0, dtype=tf.float64), tf.sqrt(s0)).log_prob(innovation0)
        gain0 = covariance / s0
        mean = mean + gain0 * innovation0
        covariance = (1.0 - gain0) * covariance
        mean = a * mean
        covariance = tf.square(a) * covariance + q
        innovation1 = obs[1] - mean
        s1 = covariance + r
        log1 = tfp.distributions.Normal(tf.constant(0.0, dtype=tf.float64), tf.sqrt(s1)).log_prob(innovation1)
        log_evidence = log0 + log1
    score = tape.gradient(log_evidence, a)
    return log_evidence, score


def scalar_nonlinear_dense_quadrature_score(
    theta: float = 0.4,
    observation: float = 0.1,
    observation_variance: float = 0.25,
    quadrature_order: int = 160,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Dense Gauss-Hermite score for ``y|x,theta ~ N(theta sin(x),0.25)``."""

    nodes, weights = gauss_hermite_nodes_weights(quadrature_order)
    x = tf.sqrt(tf.constant(2.0, dtype=tf.float64)) * nodes
    normal_weights = weights / tf.sqrt(tf.constant(math.pi, dtype=tf.float64))
    y = tf.constant(observation, dtype=tf.float64)
    scale = tf.sqrt(tf.constant(observation_variance, dtype=tf.float64))
    with tf.GradientTape() as tape:
        theta_tensor = tf.Variable(theta, dtype=tf.float64)
        likelihood = tfp.distributions.Normal(theta_tensor * tf.sin(x), scale)
        evidence = tf.reduce_sum(normal_weights * tf.exp(likelihood.log_prob(y)))
        log_evidence = tf.math.log(evidence)
    score = tape.gradient(log_evidence, theta_tensor)
    return log_evidence, score


def exact_score_result(
    name: str,
    log_likelihood: tf.Tensor,
    score: tf.Tensor,
    finite_difference_table: FiniteDifferenceTable,
    diagnostics: Mapping[str, object] | None = None,
) -> FixedBranchScoreResult:
    """Package an exact-score fixture in the Phase-5 result container."""

    manifest = BranchManifest(
        "fixed_branch_score_fixture.v1",
        {
            "name": name,
            "log_likelihood": tf.convert_to_tensor(log_likelihood, dtype=tf.float64),
            "score": tf.convert_to_tensor(score, dtype=tf.float64),
            "fixed_branch_only": True,
            "moving_basis_supported": False,
        },
    )
    identity = BranchIdentity(manifest=manifest, hash=manifest.sha256())
    tape = FixedBranchReplayTape(
        version="fixed_branch_replay_tape.phase5.v1",
        branch_identity=identity,
        entries=(
            {
                "manifest_version": manifest.version,
                "pre_replay_branch_hash": identity.hash.value,
                "solver_backend": "tensorflow",
                "moving_basis_supported": False,
                "fixed_branch_only": True,
            },
        ),
    )
    return FixedBranchScoreResult(
        log_likelihood=log_likelihood,
        score=score,
        branch_identity=identity,
        replay_tape_hash=tape.sha256().value,
        finite_difference_table=finite_difference_table,
        status=HighDimStatus.OK,
        diagnostics={
            **dict(diagnostics or {}),
            "fixed_branch_only": True,
            "replay_tape_version": tape.version,
        },
    )


def gauss_hermite_nodes_weights(order: int) -> tuple[tf.Tensor, tf.Tensor]:
    """Gauss-Hermite nodes and weights for ``exp(-x^2)`` integrals."""

    if int(order) < 2:
        raise ValueError("order must be at least 2")
    k = tf.cast(tf.range(1, int(order), dtype=tf.int32), tf.float64)
    beta = tf.sqrt(k / 2.0)
    jacobi = tf.linalg.diag(beta, k=1) + tf.linalg.diag(beta, k=-1)
    eigenvalues, eigenvectors = tf.linalg.eigh(jacobi)
    weights = tf.sqrt(tf.constant(math.pi, dtype=tf.float64)) * tf.square(eigenvectors[0, :])
    return eigenvalues, weights


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


def _left_and_dot_environments(
    matrices: Sequence[tf.Tensor],
    dot_matrices: Sequence[tf.Tensor],
) -> tuple[tuple[tf.Tensor, ...], tuple[tf.Tensor, ...]]:
    n_rows = tf.shape(matrices[0])[0]
    left = [tf.ones([n_rows, 1], dtype=tf.float64)]
    dot_left = [tf.zeros([n_rows, 1], dtype=tf.float64)]
    for matrix, dot_matrix in zip(matrices[:-1], dot_matrices[:-1]):
        dot_left.append(
            tf.einsum("na,nab->nb", dot_left[-1], matrix)
            + tf.einsum("na,nab->nb", left[-1], dot_matrix)
        )
        left.append(tf.einsum("na,nab->nb", left[-1], matrix))
    return tuple(left), tuple(dot_left)


def _right_and_dot_environments(
    matrices: Sequence[tf.Tensor],
    dot_matrices: Sequence[tf.Tensor],
) -> tuple[tuple[tf.Tensor, ...], tuple[tf.Tensor, ...]]:
    n_rows = tf.shape(matrices[0])[0]
    right = [None] * len(matrices)
    dot_right = [None] * len(matrices)
    accumulator = tf.ones([n_rows, 1], dtype=tf.float64)
    dot_accumulator = tf.zeros([n_rows, 1], dtype=tf.float64)
    for axis in range(len(matrices) - 1, -1, -1):
        right[axis] = accumulator
        dot_right[axis] = dot_accumulator
        if axis > 0:
            dot_accumulator = (
                tf.einsum("nab,nb->na", dot_matrices[axis], accumulator)
                + tf.einsum("nab,nb->na", matrices[axis], dot_accumulator)
            )
            accumulator = tf.einsum("nab,nb->na", matrices[axis], accumulator)
    return tuple(right), tuple(dot_right)


def _normal_matrix(design: tf.Tensor, weights: tf.Tensor, ridge: float) -> tf.Tensor:
    weighted_design = design * tf.reshape(weights, [-1, 1])
    normal = tf.matmul(design, weighted_design, transpose_a=True)
    return normal + tf.cast(ridge, tf.float64) * tf.eye(int(design.shape[1]), dtype=tf.float64)


def _condition_number(matrix: tf.Tensor) -> float:
    singular_values = tf.linalg.svd(matrix, compute_uv=False)
    min_value = float(tf.reduce_min(singular_values).numpy())
    max_value = float(tf.reduce_max(singular_values).numpy())
    if min_value <= 0.0 or not math.isfinite(min_value) or not math.isfinite(max_value):
        return float("inf")
    return max_value / min_value


def _paired_core_matrix(left_core: TTCore, right_core: TTCore, mass: tf.Tensor) -> tf.Tensor:
    paired = tf.einsum("alb,A mB,lm->aAbB", left_core.values, right_core.values, mass)
    return tf.reshape(
        paired,
        [
            left_core.left_rank * right_core.left_rank,
            left_core.right_rank * right_core.right_rank,
        ],
    )


def _hash_tensor(version: str, tensor: tf.Tensor) -> str:
    return BranchManifest(version, {"tensor": tf.convert_to_tensor(tensor)}).sha256().value
