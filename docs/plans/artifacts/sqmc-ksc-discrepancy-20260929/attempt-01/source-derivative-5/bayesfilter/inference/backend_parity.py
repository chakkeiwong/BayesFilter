"""Target-agnostic backend parity gates for scientific validation fixtures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


_HESSIAN_ROLES = {"explanatory_only", "hard_reviewed"}


def _as_optional_array(value: Any, *, field_name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.shape == ():
        array = np.reshape(array, ())
    if array.dtype == object:
        raise ValueError(f"{field_name} must be numeric")
    return array


def _as_shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.shape(value))


def _nonempty_string(value: Any, *, field_name: str) -> str:
    text = str(value)
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.shape == ():
        return abs(float(left) - float(right))
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _max_rel_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    denom = np.maximum(1.0, np.abs(right))
    if left.shape == ():
        return abs(float(left) - float(right)) / float(denom)
    return float(np.max(np.abs(left - right) / denom)) if left.size else 0.0


@dataclass(frozen=True)
class BackendParityRow:
    """One backend evaluation of one scalar target at one coordinate point.

    ``score`` and ``hessian`` are interpreted as derivatives of this row's
    ``value`` under ``target_scope`` and ``coordinate_scope``.  If a caller has
    transformed a parameter-space score into a latent score, it should use a
    separate row with ``coordinate_scope="latent"`` so the gate never compares
    derivatives in different coordinates.
    """

    backend_name: str
    target_scope: str
    value: Any | None = None
    score: Any | None = None
    hessian: Any | None = None
    coordinate_scope: str = "parameter"
    derivative_target_scope: str | None = None
    hessian_target_scope: str | None = None
    position: Any | None = None
    latent_position: Any | None = None
    shape: Any | None = None
    branch_label: str = "valid"
    failure_policy_label: str = "none"
    regularization_label: str = "none"
    regularization_changes_target: bool = False
    ok: bool = True
    role: str = "direct"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        backend_name = _nonempty_string(self.backend_name, field_name="backend_name")
        target_scope = _nonempty_string(self.target_scope, field_name="target_scope")
        coordinate_scope = _nonempty_string(
            self.coordinate_scope,
            field_name="coordinate_scope",
        )
        derivative_target_scope = (
            target_scope
            if self.derivative_target_scope is None
            else _nonempty_string(
                self.derivative_target_scope,
                field_name="derivative_target_scope",
            )
        )
        hessian_target_scope = (
            target_scope
            if self.hessian_target_scope is None
            else _nonempty_string(
                self.hessian_target_scope,
                field_name="hessian_target_scope",
            )
        )
        if self.score is not None and derivative_target_scope != target_scope:
            raise ValueError("score must be a derivative of the row target_scope")
        if self.hessian is not None and hessian_target_scope != target_scope:
            raise ValueError("hessian must be a derivative of the row target_scope")

        value = _as_optional_array(self.value, field_name="value")
        if value is not None and value.shape != ():
            raise ValueError("value must be a scalar")
        score = _as_optional_array(self.score, field_name="score")
        hessian = _as_optional_array(self.hessian, field_name="hessian")
        position = _as_optional_array(self.position, field_name="position")
        latent_position = _as_optional_array(
            self.latent_position,
            field_name="latent_position",
        )

        if self.shape is None:
            if score is not None:
                row_shape = tuple(int(dim) for dim in score.shape)
            elif position is not None:
                row_shape = tuple(int(dim) for dim in position.shape)
            elif latent_position is not None:
                row_shape = tuple(int(dim) for dim in latent_position.shape)
            elif value is not None:
                row_shape = ()
            else:
                row_shape = ()
        else:
            row_shape = tuple(int(dim) for dim in self.shape)

        metadata = {} if self.metadata is None else dict(self.metadata)
        object.__setattr__(self, "backend_name", backend_name)
        object.__setattr__(self, "target_scope", target_scope)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "hessian", hessian)
        object.__setattr__(self, "coordinate_scope", coordinate_scope)
        object.__setattr__(self, "derivative_target_scope", derivative_target_scope)
        object.__setattr__(self, "hessian_target_scope", hessian_target_scope)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "latent_position", latent_position)
        object.__setattr__(self, "shape", row_shape)
        object.__setattr__(
            self,
            "branch_label",
            _nonempty_string(self.branch_label, field_name="branch_label"),
        )
        object.__setattr__(
            self,
            "failure_policy_label",
            _nonempty_string(
                self.failure_policy_label,
                field_name="failure_policy_label",
            ),
        )
        object.__setattr__(
            self,
            "regularization_label",
            _nonempty_string(
                self.regularization_label,
                field_name="regularization_label",
            ),
        )
        object.__setattr__(self, "regularization_changes_target", bool(self.regularization_changes_target))
        object.__setattr__(self, "ok", bool(self.ok))
        object.__setattr__(self, "role", _nonempty_string(self.role, field_name="role"))
        object.__setattr__(self, "metadata", metadata)

    @property
    def value_finite(self) -> bool:
        return self.value is not None and bool(np.all(np.isfinite(self.value)))

    @property
    def score_finite(self) -> bool:
        return self.score is not None and bool(np.all(np.isfinite(self.score)))

    @property
    def hessian_finite(self) -> bool:
        return self.hessian is not None and bool(np.all(np.isfinite(self.hessian)))

    @property
    def branch_policy(self) -> tuple[bool, str, str]:
        return (self.ok, self.branch_label, self.failure_policy_label)

    def payload(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "target_scope": self.target_scope,
            "coordinate_scope": self.coordinate_scope,
            "derivative_target_scope": self.derivative_target_scope,
            "hessian_target_scope": self.hessian_target_scope,
            "shape": self.shape,
            "branch_label": self.branch_label,
            "failure_policy_label": self.failure_policy_label,
            "regularization_label": self.regularization_label,
            "regularization_changes_target": self.regularization_changes_target,
            "ok": self.ok,
            "role": self.role,
            "value_finite": self.value_finite,
            "score_finite": self.score_finite,
            "hessian_finite": self.hessian_finite if self.hessian is not None else None,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class BackendParityResult:
    """Auditable result from a backend parity gate."""

    passed: bool
    target_scope: str
    baseline_backend_name: str
    baseline_selection: str
    checks: Mapping[str, bool]
    failed_checks: tuple[str, ...]
    row_results: tuple[Mapping[str, Any], ...]
    max_value_abs_diff: float | None
    max_score_abs_diff: float | None
    max_hessian_abs_diff: float | None
    hessian_role: str
    nonclaims: tuple[str, ...]
    source: str = "backend_parity_gate"

    def payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "passed": self.passed,
            "target_scope": self.target_scope,
            "baseline_backend_name": self.baseline_backend_name,
            "baseline_selection": self.baseline_selection,
            "checks": dict(self.checks),
            "failed_checks": self.failed_checks,
            "row_results": tuple(dict(row) for row in self.row_results),
            "max_value_abs_diff": self.max_value_abs_diff,
            "max_score_abs_diff": self.max_score_abs_diff,
            "max_hessian_abs_diff": self.max_hessian_abs_diff,
            "hessian_role": self.hessian_role,
            "nonclaims": self.nonclaims,
        }


@dataclass(frozen=True)
class BackendParityGate:
    """Compare backend rows for one declared scalar target and coordinate."""

    rows: tuple[BackendParityRow, ...]
    target_scope: str | None = None
    baseline_backend_name: str | None = None
    value_atol: float = 1.0e-9
    value_rtol: float = 0.0
    score_atol: float = 1.0e-7
    score_rtol: float = 0.0
    hessian_atol: float = 1.0e-6
    hessian_rtol: float = 0.0
    require_value_parity: bool = True
    require_score_parity: bool = False
    require_shape_parity: bool = True
    require_branch_policy_parity: bool = True
    hessian_role: str = "explanatory_only"
    reviewed_hessian_contract: str | None = None
    nonclaims: tuple[str, ...] = (
        "backend parity is local pointwise fixture evidence only",
        "no global backend equivalence claim",
        "no posterior convergence claim",
        "no default backend readiness claim",
    )

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if len(rows) < 2:
            raise ValueError("BackendParityGate requires at least two rows")
        if not all(isinstance(row, BackendParityRow) for row in rows):
            raise TypeError("rows must contain BackendParityRow instances")
        hessian_role = str(self.hessian_role)
        if hessian_role not in _HESSIAN_ROLES:
            raise ValueError(
                "hessian_role must be 'explanatory_only' or 'hard_reviewed'"
            )
        if hessian_role == "hard_reviewed" and not self.reviewed_hessian_contract:
            raise ValueError(
                "hard Hessian parity requires a reviewed_hessian_contract"
            )
        if not self.nonclaims:
            raise ValueError("nonclaims must be non-empty")
        if self.target_scope is not None:
            _nonempty_string(self.target_scope, field_name="target_scope")
        if self.baseline_backend_name is not None:
            _nonempty_string(
                self.baseline_backend_name,
                field_name="baseline_backend_name",
            )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "hessian_role", hessian_role)
        object.__setattr__(self, "value_atol", float(self.value_atol))
        object.__setattr__(self, "value_rtol", float(self.value_rtol))
        object.__setattr__(self, "score_atol", float(self.score_atol))
        object.__setattr__(self, "score_rtol", float(self.score_rtol))
        object.__setattr__(self, "hessian_atol", float(self.hessian_atol))
        object.__setattr__(self, "hessian_rtol", float(self.hessian_rtol))
        object.__setattr__(
            self,
            "nonclaims",
            tuple(str(nonclaim) for nonclaim in self.nonclaims),
        )

    def evaluate(self) -> BackendParityResult:
        target_scope = self.target_scope or self.rows[0].target_scope
        baseline, baseline_selection = self._baseline_row()
        checks: dict[str, bool] = {}
        row_results: list[Mapping[str, Any]] = []

        checks["target_scope_matches"] = all(
            row.target_scope == target_scope for row in self.rows
        )
        checks["baseline_target_scope_matches"] = baseline.target_scope == target_scope
        checks["regularization_target_changes_labeled"] = all(
            (not row.regularization_changes_target)
            or row.regularization_label not in {"none", "unlabeled"}
            for row in self.rows
        )

        if self.require_value_parity:
            checks["required_values_present_and_finite"] = all(
                row.value_finite for row in self.rows
            )
        else:
            checks["required_values_present_and_finite"] = True
        if self.require_score_parity:
            checks["required_scores_present_and_finite"] = all(
                row.score_finite for row in self.rows
            )
        else:
            checks["required_scores_present_and_finite"] = True

        value_abs_diffs: list[float] = []
        score_abs_diffs: list[float] = []
        hessian_abs_diffs: list[float] = []
        value_rows_pass = True
        score_rows_pass = True
        shape_rows_pass = True
        branch_rows_pass = True
        hessian_rows_pass = True

        for row in self.rows:
            comparison = self._compare_row(row, baseline)
            row_results.append(comparison)
            if not comparison["is_baseline"] and comparison["value_abs_diff"] is not None:
                value_abs_diffs.append(float(comparison["value_abs_diff"]))
            if not comparison["is_baseline"] and comparison["score_max_abs_diff"] is not None:
                score_abs_diffs.append(float(comparison["score_max_abs_diff"]))
            if not comparison["is_baseline"] and comparison["hessian_max_abs_diff"] is not None:
                hessian_abs_diffs.append(float(comparison["hessian_max_abs_diff"]))
            value_rows_pass = value_rows_pass and bool(comparison["value_parity_passed"])
            score_rows_pass = score_rows_pass and bool(comparison["score_parity_passed"])
            shape_rows_pass = shape_rows_pass and bool(comparison["shape_parity_passed"])
            branch_rows_pass = branch_rows_pass and bool(comparison["branch_policy_parity_passed"])
            hessian_rows_pass = hessian_rows_pass and bool(comparison["hessian_parity_passed"])

        checks["value_parity"] = value_rows_pass
        checks["score_parity"] = score_rows_pass
        checks["shape_parity"] = shape_rows_pass
        checks["branch_policy_parity"] = branch_rows_pass
        checks["hessian_parity"] = (
            hessian_rows_pass if self.hessian_role == "hard_reviewed" else True
        )
        checks["hessian_explanatory_only"] = self.hessian_role == "explanatory_only"

        failed_checks = tuple(name for name, passed in checks.items() if not passed)
        return BackendParityResult(
            passed=not failed_checks,
            target_scope=target_scope,
            baseline_backend_name=baseline.backend_name,
            baseline_selection=baseline_selection,
            checks=checks,
            failed_checks=failed_checks,
            row_results=tuple(row_results),
            max_value_abs_diff=max(value_abs_diffs) if value_abs_diffs else None,
            max_score_abs_diff=max(score_abs_diffs) if score_abs_diffs else None,
            max_hessian_abs_diff=max(hessian_abs_diffs) if hessian_abs_diffs else None,
            hessian_role=self.hessian_role,
            nonclaims=self.nonclaims,
        )

    def _baseline_row(self) -> tuple[BackendParityRow, str]:
        if self.baseline_backend_name is None:
            return self.rows[0], "first_row"
        matches = [
            row for row in self.rows if row.backend_name == self.baseline_backend_name
        ]
        if not matches:
            raise ValueError("baseline_backend_name does not match any row")
        return matches[0], "provided"

    def _compare_row(
        self,
        row: BackendParityRow,
        baseline: BackendParityRow,
    ) -> Mapping[str, Any]:
        same_row = row is baseline
        value_abs_diff: float | None = None
        value_rel_diff: float | None = None
        if row.value is not None and baseline.value is not None:
            value_abs_diff = _max_abs_difference(row.value, baseline.value)
            value_rel_diff = _max_rel_difference(row.value, baseline.value)
        if not self.require_value_parity and value_abs_diff is None:
            value_parity_passed = True
        elif self.require_value_parity and value_abs_diff is None:
            value_parity_passed = False
        else:
            value_parity_passed = bool(
                value_abs_diff <= self.value_atol
                + self.value_rtol * max(1.0, abs(float(baseline.value)))
            )

        score_abs_diff: float | None = None
        score_rel_diff: float | None = None
        score_coordinate_match = row.coordinate_scope == baseline.coordinate_scope
        if row.score is not None and baseline.score is not None and score_coordinate_match:
            score_abs_diff = _max_abs_difference(row.score, baseline.score)
            score_rel_diff = _max_rel_difference(row.score, baseline.score)
        if self.require_score_parity and not score_coordinate_match:
            score_parity_passed = False
        elif not self.require_score_parity and score_abs_diff is None:
            score_parity_passed = True
        elif self.require_score_parity and score_abs_diff is None:
            score_parity_passed = False
        else:
            score_parity_passed = bool(
                score_abs_diff <= self.score_atol
                + self.score_rtol * max(
                    1.0,
                    float(np.max(np.abs(baseline.score))) if baseline.score is not None and baseline.score.size else 1.0,
                )
            )

        hessian_abs_diff: float | None = None
        hessian_rel_diff: float | None = None
        hessian_coordinate_match = row.coordinate_scope == baseline.coordinate_scope
        if row.hessian is not None and baseline.hessian is not None and hessian_coordinate_match:
            hessian_abs_diff = _max_abs_difference(row.hessian, baseline.hessian)
            hessian_rel_diff = _max_rel_difference(row.hessian, baseline.hessian)
        if row.hessian is not None and baseline.hessian is not None and not hessian_coordinate_match:
            hessian_parity_passed = False
        elif hessian_abs_diff is None:
            hessian_parity_passed = True
        else:
            hessian_parity_passed = bool(
                hessian_abs_diff <= self.hessian_atol
                + self.hessian_rtol
                * max(
                    1.0,
                    float(np.max(np.abs(baseline.hessian))) if baseline.hessian is not None and baseline.hessian.size else 1.0,
                )
            )

        shape_parity_passed = (
            (row.shape == baseline.shape)
            if self.require_shape_parity
            else True
        )
        branch_policy_parity_passed = (
            (row.branch_policy == baseline.branch_policy)
            if self.require_branch_policy_parity
            else True
        )
        return {
            "backend_name": row.backend_name,
            "role": row.role,
            "baseline_backend_name": baseline.backend_name,
            "is_baseline": same_row,
            "target_scope": row.target_scope,
            "coordinate_scope": row.coordinate_scope,
            "baseline_coordinate_scope": baseline.coordinate_scope,
            "score_coordinate_scope_matches": score_coordinate_match,
            "hessian_coordinate_scope_matches": hessian_coordinate_match,
            "shape": row.shape,
            "baseline_shape": baseline.shape,
            "branch_label": row.branch_label,
            "failure_policy_label": row.failure_policy_label,
            "regularization_label": row.regularization_label,
            "regularization_changes_target": row.regularization_changes_target,
            "value_abs_diff": value_abs_diff,
            "value_rel_diff": value_rel_diff,
            "value_parity_passed": value_parity_passed,
            "score_max_abs_diff": score_abs_diff,
            "score_max_rel_diff": score_rel_diff,
            "score_parity_passed": score_parity_passed,
            "hessian_max_abs_diff": hessian_abs_diff,
            "hessian_max_rel_diff": hessian_rel_diff,
            "hessian_parity_passed": (
                hessian_parity_passed
                if self.hessian_role == "hard_reviewed"
                else True
            ),
            "hessian_parity_explanatory_passed": hessian_parity_passed,
            "shape_parity_passed": shape_parity_passed,
            "branch_policy_parity_passed": branch_policy_parity_passed,
        }
