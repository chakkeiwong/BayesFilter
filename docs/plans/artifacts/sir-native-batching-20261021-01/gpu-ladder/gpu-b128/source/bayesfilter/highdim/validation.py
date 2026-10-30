"""Validation containers for the internal high-dimensional filtering lane."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from bayesfilter.highdim.diagnostics import HighDimStatus, HighDimValidationResult


class FiniteDifferenceRowStatus(str, Enum):
    """Finite-difference row validity statuses."""

    VALID = "VALID"
    INVALID_BRANCH_MISMATCH = "INVALID_BRANCH_MISMATCH"
    INVALID_NONFINITE_VALUE = "INVALID_NONFINITE_VALUE"
    INVALID_MEASURE_MISMATCH = "INVALID_MEASURE_MISMATCH"
    INVALID_COMPLEXITY_GATE = "INVALID_COMPLEXITY_GATE"


class StressRunStatus(str, Enum):
    """Phase-6 stress/performance decision statuses."""

    PASS_EXACT_REFERENCE = "PASS_EXACT_REFERENCE"
    PASS_DIAGNOSTIC_ONLY = "PASS_DIAGNOSTIC_ONLY"
    FAIL_IMPLEMENTATION = "FAIL_IMPLEMENTATION"
    FAIL_TUNING = "FAIL_TUNING"
    FAIL_APPROXIMATION = "FAIL_APPROXIMATION"
    FAIL_RESOURCE = "FAIL_RESOURCE"
    FAIL_NUMERICAL_VETO = "FAIL_NUMERICAL_VETO"
    BLOCKED_BY_PHASE_REGRESSION = "BLOCKED_BY_PHASE_REGRESSION"


class ModelSuiteTraceabilityStatus(str, Enum):
    """Governed traceability statuses for P30 model-suite claims."""

    SOURCE_MATCHED = "SOURCE_MATCHED"
    MATLAB_BEHAVIOR_MATCHED = "MATLAB_BEHAVIOR_MATCHED"
    BAYESFILTER_EXTENSION = "BAYESFILTER_EXTENSION"
    DOCUMENTED_DEVIATION = "DOCUMENTED_DEVIATION"
    REFERENCE_ONLY = "REFERENCE_ONLY"
    BLOCKED_UNTRACEABLE = "BLOCKED_UNTRACEABLE"
    BLOCKED_UNVALIDATED = "BLOCKED_UNVALIDATED"


class P30ModelSuiteModelID(str, Enum):
    """Stable identifiers for the P30 validation model suite."""

    LGSSM_EXACT = "lgssm_exact"
    STOCHASTIC_VOLATILITY_SYNTHETIC = "stochastic_volatility_synthetic"
    STOCHASTIC_VOLATILITY_REAL_OPTIONAL = "stochastic_volatility_real_optional"
    SPATIAL_SIR = "spatial_sir"
    PREDATOR_PREY = "predator_prey"
    BAYESFILTER_GENERIC_STRESS = "bayesfilter_generic_stress"


class P30Cut4ComparatorStatus(str, Enum):
    """Governed P38 CUT4 comparator row statuses."""

    EQUIVALENCE_PASSED = "EQUIVALENCE_PASSED"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
    COMPARATOR_NOT_APPLICABLE = "COMPARATOR_NOT_APPLICABLE"
    BLOCKED_BY_SEMANTIC_MISMATCH = "BLOCKED_BY_SEMANTIC_MISMATCH"


_PROMOTED_TRACEABILITY_STATUSES = frozenset(
    {
        ModelSuiteTraceabilityStatus.SOURCE_MATCHED,
        ModelSuiteTraceabilityStatus.MATLAB_BEHAVIOR_MATCHED,
        ModelSuiteTraceabilityStatus.BAYESFILTER_EXTENSION,
        ModelSuiteTraceabilityStatus.DOCUMENTED_DEVIATION,
    }
)


def _normalize_text_tuple(name: str, values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{name} must contain nonempty strings")
    return normalized


def _require_nonempty_text(name: str, value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be nonempty")
    return normalized


def _freeze_mapping(name: str, values: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return MappingProxyType({str(key): value for key, value in values.items()})


@dataclass(frozen=True)
class P30ModelSuiteRegistryRow:
    """Source-governed registry row for one P30 model-suite fixture family."""

    model_id: P30ModelSuiteModelID
    source_equations: tuple[str, ...]
    p30_anchor: str
    paper_anchor: str
    matlab_anchor: str
    bayesfilter_code_anchor: str
    bayesfilter_test_anchor: str
    status: ModelSuiteTraceabilityStatus
    implementation_status: str
    test_status: str
    reference_method: str
    dimension_convention: str
    non_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, P30ModelSuiteModelID):
            raise TypeError("model_id must be a P30ModelSuiteModelID")
        if not isinstance(self.status, ModelSuiteTraceabilityStatus):
            raise TypeError("status must be a ModelSuiteTraceabilityStatus")
        object.__setattr__(
            self,
            "source_equations",
            _normalize_text_tuple("source_equations", self.source_equations),
        )
        for field_name in (
            "p30_anchor",
            "paper_anchor",
            "matlab_anchor",
            "bayesfilter_code_anchor",
            "bayesfilter_test_anchor",
            "implementation_status",
            "test_status",
            "reference_method",
            "dimension_convention",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_nonempty_text(field_name, getattr(self, field_name)),
            )
        object.__setattr__(
            self,
            "non_claims",
            _normalize_text_tuple("non_claims", self.non_claims),
        )
        if self.status in _PROMOTED_TRACEABILITY_STATUSES:
            for anchor_name in ("bayesfilter_code_anchor", "bayesfilter_test_anchor"):
                if getattr(self, anchor_name).lower() == "none":
                    raise ValueError(f"{anchor_name} is required for promoted statuses")


@dataclass(frozen=True)
class P30ModelSuiteFixtureManifest:
    """Versioned fixture contract for one planned P30 model-suite test row."""

    version: str
    model_id: P30ModelSuiteModelID
    source_equations: tuple[str, ...]
    paper_anchor: str
    matlab_anchor: str
    parameter_values: Mapping[str, Any]
    prior: Mapping[str, Any]
    state_dimension: int
    parameter_dimension: int
    horizon: int
    basis: Mapping[str, Any]
    rank: tuple[int, ...]
    sweeps: int
    seed: str
    dtype: str
    reference_method: str
    expected_metrics: tuple[str, ...]
    vetoes: tuple[str, ...]
    non_claims: tuple[str, ...]
    clean_room_status: str
    dimension_convention: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, P30ModelSuiteModelID):
            raise TypeError("model_id must be a P30ModelSuiteModelID")
        object.__setattr__(self, "version", _require_nonempty_text("version", self.version))
        object.__setattr__(
            self,
            "source_equations",
            _normalize_text_tuple("source_equations", self.source_equations),
        )
        for field_name in (
            "paper_anchor",
            "matlab_anchor",
            "seed",
            "dtype",
            "reference_method",
            "clean_room_status",
            "dimension_convention",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_nonempty_text(field_name, getattr(self, field_name)),
            )
        if self.dtype != "tf.float64":
            raise ValueError("dtype must be tf.float64")
        if self.state_dimension <= 0:
            raise ValueError("state_dimension must be positive")
        if self.parameter_dimension < 0:
            raise ValueError("parameter_dimension must be nonnegative")
        if self.horizon < 0:
            raise ValueError("horizon must be nonnegative")
        if self.sweeps < 0:
            raise ValueError("sweeps must be nonnegative")
        object.__setattr__(
            self,
            "parameter_values",
            _freeze_mapping("parameter_values", self.parameter_values),
        )
        object.__setattr__(self, "prior", _freeze_mapping("prior", self.prior))
        object.__setattr__(self, "basis", _freeze_mapping("basis", self.basis))
        object.__setattr__(self, "rank", tuple(int(value) for value in self.rank))
        if not self.rank or any(value <= 0 for value in self.rank):
            raise ValueError("rank must contain positive integers")
        for field_name in ("expected_metrics", "vetoes", "non_claims"):
            object.__setattr__(
                self,
                field_name,
                _normalize_text_tuple(field_name, getattr(self, field_name)),
            )


@dataclass(frozen=True)
class P30ModelSuiteResultManifest:
    """Result contract for a P30 model-suite row without promoting proxies."""

    version: str
    fixture_version: str
    model_id: P30ModelSuiteModelID
    source_governance_status: ModelSuiteTraceabilityStatus
    bayesfilter_evidence_anchors: tuple[str, ...]
    accuracy_metrics: Mapping[str, Any]
    resource_metrics: Mapping[str, Any]
    finite_diagnostics: Mapping[str, Any]
    branch_replay_status: str
    failure_classification: str
    clean_room_status: str
    non_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, P30ModelSuiteModelID):
            raise TypeError("model_id must be a P30ModelSuiteModelID")
        if not isinstance(self.source_governance_status, ModelSuiteTraceabilityStatus):
            raise TypeError("source_governance_status must be a ModelSuiteTraceabilityStatus")
        for field_name in (
            "version",
            "fixture_version",
            "branch_replay_status",
            "failure_classification",
            "clean_room_status",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_nonempty_text(field_name, getattr(self, field_name)),
            )
        object.__setattr__(
            self,
            "bayesfilter_evidence_anchors",
            tuple(str(value).strip() for value in self.bayesfilter_evidence_anchors),
        )
        if any(not value for value in self.bayesfilter_evidence_anchors):
            raise ValueError("bayesfilter_evidence_anchors must be nonempty strings")
        if (
            self.source_governance_status in _PROMOTED_TRACEABILITY_STATUSES
            and not self.bayesfilter_evidence_anchors
        ):
            raise ValueError("promoted result status requires BayesFilter evidence anchors")
        object.__setattr__(
            self,
            "accuracy_metrics",
            _freeze_mapping("accuracy_metrics", self.accuracy_metrics),
        )
        object.__setattr__(
            self,
            "resource_metrics",
            _freeze_mapping("resource_metrics", self.resource_metrics),
        )
        object.__setattr__(
            self,
            "finite_diagnostics",
            _freeze_mapping("finite_diagnostics", self.finite_diagnostics),
        )
        object.__setattr__(
            self,
            "non_claims",
            _normalize_text_tuple("non_claims", self.non_claims),
        )


_CUT4_REQUIRED_NONCLAIM_TERMS = (
    "ground truth",
    "paper-scale",
    "adaptive MATLAB",
    "GPU",
    "HMC",
    "DSGE",
    "public API",
    "score API",
)


@dataclass(frozen=True)
class P30Cut4StatisticalComparatorManifest:
    """Governed manifest for P38 CUT4 comparator/equivalence rows."""

    version: str
    model_id: P30ModelSuiteModelID
    comparator_status: P30Cut4ComparatorStatus
    traceability_status: ModelSuiteTraceabilityStatus
    comparator_description: str
    candidate_description: str
    audit_design: Mapping[str, Any]
    equivalence_band: tuple[float, float] | None
    ci_low: float | None
    ci_high: float | None
    max_abs_error: float | None
    outlier_band: float | None
    finite_diagnostics: Mapping[str, Any]
    veto_status: str
    promotion_decision: str
    non_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, P30ModelSuiteModelID):
            raise TypeError("model_id must be a P30ModelSuiteModelID")
        if not isinstance(self.comparator_status, P30Cut4ComparatorStatus):
            raise TypeError("comparator_status must be a P30Cut4ComparatorStatus")
        if not isinstance(self.traceability_status, ModelSuiteTraceabilityStatus):
            raise TypeError("traceability_status must be a ModelSuiteTraceabilityStatus")
        for field_name in (
            "version",
            "comparator_description",
            "candidate_description",
            "veto_status",
            "promotion_decision",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_nonempty_text(field_name, getattr(self, field_name)),
            )
        object.__setattr__(
            self,
            "audit_design",
            _freeze_mapping("audit_design", self.audit_design),
        )
        object.__setattr__(
            self,
            "finite_diagnostics",
            _freeze_mapping("finite_diagnostics", self.finite_diagnostics),
        )
        non_claims = _normalize_text_tuple("non_claims", self.non_claims)
        non_claim_text = " ".join(non_claims).lower()
        for term in _CUT4_REQUIRED_NONCLAIM_TERMS:
            if term.lower() not in non_claim_text:
                raise ValueError(f"non_claims must include {term} non-claim")
        object.__setattr__(self, "non_claims", non_claims)

        if self.comparator_status is P30Cut4ComparatorStatus.EQUIVALENCE_PASSED:
            if self.equivalence_band is None:
                raise ValueError("equivalence_band is required for EQUIVALENCE_PASSED")
            band = tuple(float(value) for value in self.equivalence_band)
            if len(band) != 2 or band[0] >= band[1]:
                raise ValueError("equivalence_band must be an increasing pair")
            numeric_fields = {
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
                "max_abs_error": self.max_abs_error,
                "outlier_band": self.outlier_band,
            }
            for name, value in numeric_fields.items():
                if not _is_finite_number(value):
                    raise ValueError(f"{name} must be finite for EQUIVALENCE_PASSED")
            ci_low = float(self.ci_low)
            ci_high = float(self.ci_high)
            max_abs_error = float(self.max_abs_error)
            outlier_band = float(self.outlier_band)
            if ci_low < band[0] or ci_high > band[1]:
                raise ValueError("CI must lie inside equivalence_band")
            if max_abs_error > outlier_band:
                raise ValueError("max_abs_error exceeds outlier_band")
            if not self.veto_status.upper().startswith("PASS"):
                raise ValueError("EQUIVALENCE_PASSED requires PASS veto_status")
            if "PROMOTE" in self.promotion_decision.upper():
                raise ValueError("P38 first gate records equivalence but cannot promote defaults")
            object.__setattr__(self, "equivalence_band", band)
            return

        if any(
            value is not None
            for value in (
                self.equivalence_band,
                self.ci_low,
                self.ci_high,
                self.max_abs_error,
                self.outlier_band,
            )
        ):
            raise ValueError("non-equivalence rows must not provide equivalence metrics")
        if self.comparator_status is P30Cut4ComparatorStatus.DIAGNOSTIC_ONLY:
            if "DIAGNOSTIC" not in self.promotion_decision.upper():
                raise ValueError("DIAGNOSTIC_ONLY rows require diagnostic promotion_decision")
        if self.comparator_status in (
            P30Cut4ComparatorStatus.COMPARATOR_NOT_APPLICABLE,
            P30Cut4ComparatorStatus.BLOCKED_BY_SEMANTIC_MISMATCH,
        ):
            text = f"{self.comparator_description} {self.veto_status} {self.promotion_decision}".lower()
            if "not applicable" not in text and "semantic" not in text:
                raise ValueError("inapplicable rows must state not applicable or semantic mismatch")


_PREDATOR_PREY_MATCHED_SETTING_FIELDS = frozenset(
    {
        "observations_seed",
        "truth_seed",
        "prior",
        "parameter_box",
        "initial_state_prior",
        "delta",
        "rk4_internal_step",
        "process_covariance",
        "observation_covariance",
        "dtype",
        "basis_family",
        "basis_size",
        "nominal_rank_cap",
        "sweep_count",
        "stopping_tolerance",
        "sample_count",
        "wall_time_accounting_policy",
    }
)

_PREDATOR_PREY_REQUIRED_METRICS = frozenset(
    {
        "q_ess_linear_0p50",
        "q_ess_nonlinear_0p50",
        "wall_time_linear_seconds",
        "wall_time_nonlinear_seconds",
        "delta_ess",
        "delta_cost",
        "trajectory_rmse_linear",
        "trajectory_rmse_nonlinear",
    }
)


@dataclass(frozen=True)
class P30PredatorPreyComparisonManifest:
    """First-gate schema for fair predator-prey preconditioning comparisons."""

    version: str
    linear_settings: Mapping[str, Any]
    nonlinear_settings: Mapping[str, Any]
    metrics: Mapping[str, Any]
    promotion_decision: str
    non_claims: tuple[str, ...]
    model_id: P30ModelSuiteModelID = P30ModelSuiteModelID.PREDATOR_PREY

    def __post_init__(self) -> None:
        if self.model_id is not P30ModelSuiteModelID.PREDATOR_PREY:
            raise ValueError("model_id must be predator_prey")
        object.__setattr__(self, "version", _require_nonempty_text("version", self.version))
        linear_settings = {str(key): value for key, value in self.linear_settings.items()}
        nonlinear_settings = {str(key): value for key, value in self.nonlinear_settings.items()}
        metrics = {str(key): value for key, value in self.metrics.items()}
        missing_linear = sorted(_PREDATOR_PREY_MATCHED_SETTING_FIELDS.difference(linear_settings))
        missing_nonlinear = sorted(_PREDATOR_PREY_MATCHED_SETTING_FIELDS.difference(nonlinear_settings))
        missing_metrics = sorted(_PREDATOR_PREY_REQUIRED_METRICS.difference(metrics))
        if missing_linear:
            raise ValueError(f"linear_settings missing fields: {', '.join(missing_linear)}")
        if missing_nonlinear:
            raise ValueError(f"nonlinear_settings missing fields: {', '.join(missing_nonlinear)}")
        if missing_metrics:
            raise ValueError(f"metrics missing fields: {', '.join(missing_metrics)}")
        mismatched = sorted(
            field
            for field in _PREDATOR_PREY_MATCHED_SETTING_FIELDS
            if linear_settings[field] != nonlinear_settings[field]
        )
        if mismatched:
            raise ValueError(f"unmatched comparison settings: {', '.join(mismatched)}")
        for name in (
            "q_ess_linear_0p50",
            "q_ess_nonlinear_0p50",
            "wall_time_linear_seconds",
            "wall_time_nonlinear_seconds",
            "delta_ess",
            "delta_cost",
            "trajectory_rmse_linear",
            "trajectory_rmse_nonlinear",
        ):
            value = metrics[name]
            if not isinstance(value, (int, float)) or value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"{name} must be finite numeric")
        if metrics["wall_time_linear_seconds"] <= 0.0 or metrics["wall_time_nonlinear_seconds"] <= 0.0:
            raise ValueError("wall time metrics must be positive")
        decision = _require_nonempty_text("promotion_decision", self.promotion_decision)
        if decision == "PROMOTE_NONLINEAR_USEFULNESS":
            if metrics["delta_ess"] <= 0.0 or metrics["delta_cost"] <= 0.0:
                raise ValueError("promotion requires positive delta_ess and delta_cost")
            raise ValueError("M4 first gate cannot promote nonlinear preconditioning usefulness")
        object.__setattr__(self, "linear_settings", MappingProxyType(linear_settings))
        object.__setattr__(self, "nonlinear_settings", MappingProxyType(nonlinear_settings))
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "promotion_decision", decision)
        object.__setattr__(self, "non_claims", _normalize_text_tuple("non_claims", self.non_claims))


def p30_model_suite_registry() -> Mapping[str, P30ModelSuiteRegistryRow]:
    """Return the current source-governed P30 model-suite registry."""

    rows = (
        P30ModelSuiteRegistryRow(
            model_id=P30ModelSuiteModelID.LGSSM_EXACT,
            source_equations=("eq:p27-lg1", "eq:p27-lg2", "eq:p27-lg3", "eq:p27-lg9"),
            p30_anchor="P30 exact-reference linear Gaussian benchmark",
            paper_anchor="Zhao--Cui linear Gaussian benchmark, Section 6.1",
            matlab_anchor="eg1_kalman/main_script.m; eg1_kalman/computeHL2.m; eg1_kalman/computeL1.m",
            bayesfilter_code_anchor="bayesfilter/highdim/models.py; bayesfilter/highdim/filtering.py",
            bayesfilter_test_anchor="tests/highdim/test_filtering_kalman_exact.py; tests/highdim/test_scaling_smoke.py",
            status=ModelSuiteTraceabilityStatus.SOURCE_MATCHED,
            implementation_status="partial_exact_value_path",
            test_status="tiny_exact_and_stress_smoke",
            reference_method="exact Kalman evidence and moments",
            dimension_convention="state dimension m plus parameter grid when tested",
            non_claims=("not full Zhao--Cui reproduction grid", "no derivative claim"),
        ),
        P30ModelSuiteRegistryRow(
            model_id=P30ModelSuiteModelID.STOCHASTIC_VOLATILITY_SYNTHETIC,
            source_equations=(
                "eq:p27-sv1",
                "eq:p27-sv2",
                "eq:p27-sv3",
                "eq:p27-sv5a",
                "eq:p27-sv6",
            ),
            p30_anchor="P30 long-horizon stochastic-volatility benchmark",
            paper_anchor="Zhao--Cui stochastic-volatility benchmark",
            matlab_anchor="eg2_sv/mainscript.m",
            bayesfilter_code_anchor="bayesfilter/highdim/models.py; bayesfilter/highdim/filtering.py",
            bayesfilter_test_anchor="tests/highdim/test_p30_stochastic_volatility.py",
            status=ModelSuiteTraceabilityStatus.BAYESFILTER_EXTENSION,
            implementation_status="scalar_dense_value_path_only",
            test_status="tiny_dense_reference_and_scalar_value_path_smoke",
            reference_method="tiny dense joint quadrature and sequential dense-grid reference; scalar dense BayesFilter value path",
            dimension_convention="synthetic rows include x_0:x_T; joint dimension is 2+(T+1)",
            non_claims=(
                "not TT posterior accuracy",
                "no Zhao--Cui T=1000 reproduction",
                "no long-horizon TT filtering stability claim",
            ),
        ),
        P30ModelSuiteRegistryRow(
            model_id=P30ModelSuiteModelID.STOCHASTIC_VOLATILITY_REAL_OPTIONAL,
            source_equations=("eq:p27-sv5b", "eq:p27-sv9", "eq:p27-sv10"),
            p30_anchor="P30 real-data stochastic-volatility benchmark",
            paper_anchor="Zhao--Cui S&P 500 stochastic-volatility example",
            matlab_anchor="eg2_sv/mainscriptSP500.m; eg2_sv/SP500.txt",
            bayesfilter_code_anchor="none",
            bayesfilter_test_anchor="none",
            status=ModelSuiteTraceabilityStatus.REFERENCE_ONLY,
            implementation_status="not_implemented",
            test_status="not_tested",
            reference_method="stability and reasonableness only unless separate reference is planned",
            dimension_convention="real-data transformed coordinates must be declared",
            non_claims=("not an accuracy proof", "not BayesFilter evidence"),
        ),
        P30ModelSuiteRegistryRow(
            model_id=P30ModelSuiteModelID.SPATIAL_SIR,
            source_equations=(
                "eq:p27-sir1",
                "eq:p27-sir2",
                "eq:p27-sir3",
                "eq:p27-sir4",
                "eq:p27-sir5",
                "eq:p27-sir6",
                "eq:p27-sir7",
                "eq:p27-sir9",
                "eq:p27-sir10",
                "eq:p27-sir11",
                "eq:p27-sir13",
            ),
            p30_anchor="P30 spatial SIR benchmark",
            paper_anchor="Zhao--Cui spatial SIR benchmark",
            matlab_anchor="eg3_sir/mainscript.m; deep-tensor.dev/tests/test_sparse_sirt/sir_ll_ftt.m",
            bayesfilter_code_anchor="bayesfilter/highdim/models.py",
            bayesfilter_test_anchor="tests/highdim/test_p30_spatial_sir.py",
            status=ModelSuiteTraceabilityStatus.BAYESFILTER_EXTENSION,
            implementation_status="first_gate_model_contract_only",
            test_status="rk4_observation_likelihood_simulation_rmse_diagnostics",
            reference_method="independent RK4 hand/small-step checks and synthetic truth diagnostics",
            dimension_convention="state is (S_1,I_1,...,S_J,I_J) in R^{2J}",
            non_claims=(
                "not production TT/SIRT SIR filtering",
                "not paper-scale J=9 accuracy",
                "no rank ladder evidence",
                "no partial-observation scalability claim",
            ),
        ),
        P30ModelSuiteRegistryRow(
            model_id=P30ModelSuiteModelID.PREDATOR_PREY,
            source_equations=(
                "eq:p27-pp1",
                "eq:p27-pp2",
                "eq:p27-pp3",
                "eq:p27-pp4",
                "eq:p27-pp5",
                "eq:p27-pp5a",
                "eq:p27-pp5b",
                "eq:p27-pp5c",
                "eq:p27-pp5d",
                "eq:p27-pp6",
                "eq:p27-pp7",
                "eq:p27-pp8",
            ),
            p30_anchor="P30 predator-prey nonlinear preconditioning benchmark",
            paper_anchor="Zhao--Cui predator-prey validation example; Section 5 preconditioning context",
            matlab_anchor="eg4_predatorprey/mainscript.m; models/pre_sol.m",
            bayesfilter_code_anchor="bayesfilter/highdim/models.py; bayesfilter/highdim/validation.py",
            bayesfilter_test_anchor="tests/highdim/test_p30_predator_prey.py; tests/highdim/test_p30_model_suite_contracts.py",
            status=ModelSuiteTraceabilityStatus.BAYESFILTER_EXTENSION,
            implementation_status="first_gate_model_contract_and_comparison_schema_only",
            test_status="rk4_prior_likelihood_simulation_rmse_manifest_schema",
            reference_method="independent RK4 checks and matched-comparison schema before preconditioner rows",
            dimension_convention="state is (P,Q), parameter is (r,K,a,s,u,v)",
            non_claims=(
                "no nonlinear preconditioning usefulness claim",
                "no matched linear/nonlinear comparison success claim",
                "no paper-scale predator-prey result",
                "no adaptive MATLAB behavior claim",
                "no high-dimensional scalability claim",
            ),
        ),
        P30ModelSuiteRegistryRow(
            model_id=P30ModelSuiteModelID.BAYESFILTER_GENERIC_STRESS,
            source_equations=("eq:p27-bf1", "eq:p27-bf2", "eq:p27-bf3", "eq:p27-r8"),
            p30_anchor="P30 BayesFilter stress ladders beyond the paper",
            paper_anchor="BayesFilter extension beyond Zhao--Cui reproduction",
            matlab_anchor="none required; paper model suite informs stress shape only",
            bayesfilter_code_anchor="bayesfilter/highdim/validation.py",
            bayesfilter_test_anchor="tests/highdim/test_scaling_smoke.py; tests/highdim/test_p30_stress_ladders.py",
            status=ModelSuiteTraceabilityStatus.BAYESFILTER_EXTENSION,
            implementation_status="p37_m5_first_gate_stress_schema_and_tiny_cpu_smoke",
            test_status="m5_manifest_schema_one_axis_guardrail_and_tiny_cpu_smoke",
            reference_method="resource and failure manifest, exact references only when available",
            dimension_convention="generic nonlinear SSM dimension is declared per row",
            non_claims=(
                "not paper-model reproduction",
                "no production scalability claim",
                "no correctness claim from stress rows",
                "no GPU/HMC/DSGE readiness claim",
            ),
        ),
    )
    return MappingProxyType({row.model_id.value: row for row in rows})


_REQUIRED_STRESS_MANIFEST_FIELDS = frozenset(
    {
        "git_commit",
        "command",
        "environment",
        "cpu_gpu_status",
        "random_seeds",
        "dtype",
        "model_equations",
        "dimension_rank_degree_horizon_grid",
        "resource_budgets",
        "expected_memory_model",
        "measured_peak_memory",
        "wall_time_seconds",
        "exact_reference_error",
        "fit_residual",
        "holdout_residual",
        "normalizer_diagnostics",
        "branch_hash",
        "deterministic_replay_status",
        "decision_status",
        "termination_reason",
        "stop_condition_triggered",
        "what_is_not_concluded",
    }
)


@dataclass(frozen=True)
class StressRunManifest:
    """Immutable P37-M5 stress-smoke manifest with required resource fields.

    The class name is preserved for compatibility with earlier highdim plans
    that called this the Phase-6 stress lane.
    """

    fields: Mapping[str, object]

    def __post_init__(self) -> None:
        mapping = {str(key): value for key, value in self.fields.items()}
        missing = sorted(_REQUIRED_STRESS_MANIFEST_FIELDS.difference(mapping))
        if missing:
            raise ValueError(f"stress manifest missing fields: {', '.join(missing)}")
        try:
            decision_status = StressRunStatus(str(mapping["decision_status"]))
        except ValueError as exc:
            raise ValueError("decision_status must be a StressRunStatus value") from exc
        if not str(mapping["branch_hash"]).strip():
            raise ValueError("branch_hash must be nonempty")
        if not str(mapping["deterministic_replay_status"]).strip():
            raise ValueError("deterministic_replay_status must be nonempty")
        if not str(mapping["termination_reason"]).strip():
            raise ValueError("termination_reason must be nonempty")
        if not str(mapping["stop_condition_triggered"]).strip():
            raise ValueError("stop_condition_triggered must be nonempty")
        resource_budgets = mapping["resource_budgets"]
        if not isinstance(resource_budgets, Mapping):
            raise ValueError("resource_budgets must be a mapping")
        for name in (
            "row_budget",
            "column_budget",
            "dense_matrix_byte_budget",
            "normal_matrix_byte_budget",
            "retained_storage_byte_budget",
        ):
            if name not in resource_budgets:
                raise ValueError(f"resource_budgets missing {name}")
        object.__setattr__(
            self,
            "fields",
            MappingProxyType({**mapping, "decision_status": decision_status.value}),
        )

    @property
    def decision_status(self) -> StressRunStatus:
        """Return the classified P37-M5 stress decision."""

        return StressRunStatus(str(self.fields["decision_status"]))


_STRESS_LADDER_AXES = frozenset({"dimension", "horizon", "rank", "basis"})
_STRESS_LADDER_AXIS_VALUES = frozenset((*_STRESS_LADDER_AXES, "none"))
_STRESS_EVIDENCE_INTERPRETATIONS = frozenset({"EXACT_REFERENCE", "DIAGNOSTIC_ONLY"})
_STRESS_PROMOTION_DECISIONS = frozenset({"STRESS_SCHEMA_ONLY", "RECORD_DIAGNOSTIC_ROW"})
_STRESS_REQUIRED_NONCLAIM_TERMS = (
    "correctness",
    "scalability",
    "GPU",
    "HMC",
    "DSGE",
    "paper",
)


def _normalize_optional_text_tuple(name: str, values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{name} must contain nonempty strings")
    return normalized


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value and value not in (
        float("inf"),
        float("-inf"),
    )


@dataclass(frozen=True)
class P30StressLadderRowManifest:
    """P37-M5 first-gate wrapper enforcing stress-ladder interpretation."""

    phase_id: str
    stress_manifest: StressRunManifest
    ladder_axis: str
    varied_axes: tuple[str, ...]
    fixed_axes: Mapping[str, Any]
    lower_phase_guardrail_status: str
    evidence_interpretation: str
    promotion_decision: str
    non_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        phase_id = _require_nonempty_text("phase_id", self.phase_id)
        if phase_id != "P37-M5":
            raise ValueError("phase_id must be P37-M5")
        if not isinstance(self.stress_manifest, StressRunManifest):
            raise TypeError("stress_manifest must be a StressRunManifest")
        ladder_axis = _require_nonempty_text("ladder_axis", self.ladder_axis)
        if ladder_axis not in _STRESS_LADDER_AXIS_VALUES:
            raise ValueError("ladder_axis must be one of dimension, horizon, rank, basis, none")
        varied_axes = _normalize_optional_text_tuple("varied_axes", self.varied_axes)
        invalid_varied = sorted(set(varied_axes).difference(_STRESS_LADDER_AXES))
        if invalid_varied:
            raise ValueError(f"varied_axes contains unsupported axes: {', '.join(invalid_varied)}")
        if len(varied_axes) > 1:
            raise ValueError("M5 first gate permits only one active ladder axis")
        if ladder_axis == "none":
            if varied_axes:
                raise ValueError("ladder_axis none cannot vary ladder axes")
        elif varied_axes != (ladder_axis,):
            raise ValueError("varied_axes must exactly match ladder_axis")
        fixed_axes = {str(key): value for key, value in self.fixed_axes.items()}
        varied_set = set(varied_axes)
        for axis in sorted(_STRESS_LADDER_AXES.difference(varied_set)):
            if axis not in fixed_axes:
                raise ValueError(f"fixed_axes missing {axis}")
        for axis in varied_set:
            if axis in fixed_axes:
                raise ValueError("fixed_axes must not include varied_axes")

        lower_phase_guardrail_status = _require_nonempty_text(
            "lower_phase_guardrail_status",
            self.lower_phase_guardrail_status,
        )
        lower_guardrail_passed = lower_phase_guardrail_status.upper().startswith("PASS")
        is_blocked_by_regression = (
            self.stress_manifest.decision_status is StressRunStatus.BLOCKED_BY_PHASE_REGRESSION
        )
        if (
            not lower_guardrail_passed
            and not is_blocked_by_regression
        ):
            raise ValueError("lower phase regression must block stress row")
        if lower_guardrail_passed and is_blocked_by_regression:
            raise ValueError("passing lower phase guardrail cannot use regression-blocked status")

        evidence_interpretation = _require_nonempty_text(
            "evidence_interpretation",
            self.evidence_interpretation,
        )
        if evidence_interpretation not in _STRESS_EVIDENCE_INTERPRETATIONS:
            raise ValueError("evidence_interpretation must be EXACT_REFERENCE or DIAGNOSTIC_ONLY")
        if (
            evidence_interpretation == "EXACT_REFERENCE"
            and self.stress_manifest.decision_status is not StressRunStatus.PASS_EXACT_REFERENCE
        ):
            raise ValueError("EXACT_REFERENCE interpretation requires PASS_EXACT_REFERENCE status")
        if (
            self.stress_manifest.decision_status is StressRunStatus.PASS_EXACT_REFERENCE
            and evidence_interpretation != "EXACT_REFERENCE"
        ):
            raise ValueError("PASS_EXACT_REFERENCE status requires EXACT_REFERENCE interpretation")

        promotion_decision = _require_nonempty_text("promotion_decision", self.promotion_decision)
        if promotion_decision not in _STRESS_PROMOTION_DECISIONS:
            if "PROMOTE" in promotion_decision.upper():
                raise ValueError("M5 first gate cannot promote correctness or scalability")
            raise ValueError("promotion_decision must be STRESS_SCHEMA_ONLY or RECORD_DIAGNOSTIC_ROW")

        fields = self.stress_manifest.fields
        wall_time = fields["wall_time_seconds"]
        if not _is_finite_number(wall_time) or float(wall_time) < 0.0:
            raise ValueError("wall_time_seconds must be finite and nonnegative")
        if evidence_interpretation == "EXACT_REFERENCE":
            exact_reference_error = fields["exact_reference_error"]
            if not _is_finite_number(exact_reference_error) or float(exact_reference_error) < 0.0:
                raise ValueError("exact_reference_error must be finite and nonnegative")
        resource_budgets = fields["resource_budgets"]
        for name, value in resource_budgets.items():
            if not _is_finite_number(value) or float(value) < 0.0:
                raise ValueError(f"resource_budgets {name} must be finite and nonnegative")

        non_claims = _normalize_text_tuple("non_claims", self.non_claims)
        non_claim_text = " ".join(non_claims).lower()
        for term in _STRESS_REQUIRED_NONCLAIM_TERMS:
            if term.lower() not in non_claim_text:
                raise ValueError(f"non_claims must include {term} non-claim")

        object.__setattr__(self, "phase_id", phase_id)
        object.__setattr__(self, "ladder_axis", ladder_axis)
        object.__setattr__(self, "varied_axes", varied_axes)
        object.__setattr__(self, "fixed_axes", MappingProxyType(fixed_axes))
        object.__setattr__(self, "lower_phase_guardrail_status", lower_phase_guardrail_status)
        object.__setattr__(self, "evidence_interpretation", evidence_interpretation)
        object.__setattr__(self, "promotion_decision", promotion_decision)
        object.__setattr__(self, "non_claims", non_claims)


def classify_stress_failure(status: HighDimStatus | str, reason: str) -> StressRunStatus:
    """Classify low-level highdim statuses into P37-M5 stress decisions."""

    highdim_status = status if isinstance(status, HighDimStatus) else HighDimStatus(str(status))
    reason_text = str(reason).lower()
    if highdim_status is HighDimStatus.OK:
        return StressRunStatus.PASS_EXACT_REFERENCE
    if highdim_status in (
        HighDimStatus.NONFINITE_VALUE,
        HighDimStatus.NORMALIZER_FLOOR_EXCEEDED,
        HighDimStatus.CONDITIONAL_DENOMINATOR_FLOOR_EXCEEDED,
        HighDimStatus.EXACT_REFERENCE_MISMATCH,
    ):
        return StressRunStatus.FAIL_NUMERICAL_VETO
    if highdim_status in (
        HighDimStatus.COMPLEXITY_GATE,
        HighDimStatus.RETAINED_STORAGE_BUDGET_EXCEEDED,
    ):
        return StressRunStatus.FAIL_RESOURCE
    if highdim_status in (
        HighDimStatus.CONDITION_NUMBER_VETO,
        HighDimStatus.HOLDOUT_RESIDUAL_VETO,
    ):
        if "holdout" in reason_text or highdim_status is HighDimStatus.HOLDOUT_RESIDUAL_VETO:
            return StressRunStatus.FAIL_APPROXIMATION
        return StressRunStatus.FAIL_TUNING
    if highdim_status in (
        HighDimStatus.INVALID_SHAPE,
        HighDimStatus.MEASURE_MISMATCH,
        HighDimStatus.RETAINED_MEASURE_MISMATCH,
        HighDimStatus.RETAINED_AXES_MISMATCH,
        HighDimStatus.INVALID_BRANCH_MISMATCH,
        HighDimStatus.REPLAY_TAPE_MISMATCH,
        HighDimStatus.REPLAY_CORE_HASH_MISMATCH,
        HighDimStatus.REPLAY_ENVIRONMENT_STALE,
    ):
        return StressRunStatus.FAIL_IMPLEMENTATION
    return StressRunStatus.FAIL_IMPLEMENTATION


def stress_ladder_blocked_by_phase_regression(regression_detected: bool) -> StressRunStatus:
    """Return the P37-M5 ladder gate status before stress runs start."""

    if regression_detected:
        return StressRunStatus.BLOCKED_BY_PHASE_REGRESSION
    return StressRunStatus.PASS_DIAGNOSTIC_ONLY


@dataclass(frozen=True)
class ComplexityBudget:
    """Dense allocation budget used before high-dimensional materialization."""

    max_elements: int
    max_bytes: int

    def __post_init__(self) -> None:
        if self.max_elements <= 0:
            raise ValueError("max_elements must be positive")
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

    def validate(self, estimated_elements: int, dtype_size: int) -> HighDimValidationResult:
        estimated_bytes = int(estimated_elements) * int(dtype_size)
        if estimated_elements > self.max_elements or estimated_bytes > self.max_bytes:
            return HighDimValidationResult(
                status=HighDimStatus.COMPLEXITY_GATE,
                message="complexity budget exceeded",
                diagnostics={
                    "estimated_elements": int(estimated_elements),
                    "estimated_bytes": estimated_bytes,
                    "max_elements": self.max_elements,
                    "max_bytes": self.max_bytes,
                },
            )
        return HighDimValidationResult(
            status=HighDimStatus.OK,
            message="complexity budget ok",
            diagnostics={
                "estimated_elements": int(estimated_elements),
                "estimated_bytes": estimated_bytes,
            },
        )
