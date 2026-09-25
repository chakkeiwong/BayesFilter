"""Adapters for MacroFinance-shaped linear Gaussian state-space objects.

The functions here use structural typing: BayesFilter does not import
MacroFinance at package import time.  Client projects keep ownership of their
financial model construction and derivative recursions; BayesFilter only
normalizes generic LGSSM objects and result metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Protocol

import numpy as np

from bayesfilter.filters.kalman import (
    KalmanResult,
    LinearGaussianStateSpace,
    kalman_log_likelihood,
)
from bayesfilter.structural import StatePartition


class MacroFinanceLGSSMLike(Protocol):
    initial_mean: Any
    initial_covariance: Any
    transition_offset: Any
    transition_matrix: Any
    transition_covariance: Any
    observation_offset: Any
    observation_matrix: Any
    observation_covariance: Any


class MacroFinanceProviderLike(Protocol):
    parameter_dim: int

    def build_state_space(self, theta: np.ndarray) -> MacroFinanceLGSSMLike: ...

    def parameter_names(self) -> list[str]: ...


class MacroFinanceDerivativeProviderLike(MacroFinanceProviderLike, Protocol):
    def build_state_space_with_derivatives(
        self,
        theta: np.ndarray,
        order: int,
    ) -> tuple[MacroFinanceLGSSMLike, Any]: ...


@dataclass(frozen=True)
class MacroFinanceLikelihoodResult:
    """Value-side likelihood result with adapter provenance."""

    log_likelihood: float
    kalman_result: KalmanResult
    parameter_names: tuple[str, ...]
    theta: np.ndarray
    jitter: float
    source: str = "macrofinance_provider"


@dataclass(frozen=True)
class MacroFinanceDerivativeResult:
    """Derivative result delegated to a MacroFinance-compatible backend."""

    log_likelihood: float
    score: np.ndarray
    hessian: np.ndarray | None
    parameter_names: tuple[str, ...]
    theta: np.ndarray
    derivative_order: int
    backend: str
    jitter: float | None = None
    initial_mean_policy: str | None = None


@dataclass(frozen=True)
class MacroFinanceHMCReadinessResult:
    """Finite-operation conformance result for future HMC targets."""

    parameter_names: tuple[str, ...]
    theta: np.ndarray
    log_prob: float
    score: np.ndarray
    negative_log_prob: float
    negative_gradient: np.ndarray
    negative_hessian: np.ndarray
    backend: str
    readiness_label: str = "hmc_contract_ready_smoke"
    convergence_claim: str = "not_claimed"


@dataclass(frozen=True)
class ParameterUnitMetadata:
    parameter_names: tuple[str, ...]
    units: np.ndarray
    source: str = "provider.parameter_units"

    def __post_init__(self) -> None:
        units = np.asarray(self.units, dtype=float).copy()
        units.setflags(write=False)
        object.__setattr__(self, "parameter_names", tuple(str(name) for name in self.parameter_names))
        object.__setattr__(self, "units", units)


@dataclass(frozen=True)
class ObservationMaskMetadata:
    shape: tuple[int, int]
    observed_count: int
    missing_count: int
    all_observed: bool
    convention: str = "true_means_observed"
    source: str = "provider.observation_mask"

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))
        object.__setattr__(self, "observed_count", int(self.observed_count))
        object.__setattr__(self, "missing_count", int(self.missing_count))
        object.__setattr__(self, "all_observed", bool(self.all_observed))


@dataclass(frozen=True)
class DerivativeCoverageMetadata:
    rows: tuple[tuple[tuple[str, Any], ...], ...]
    source: str = "provider.derivative_coverage_matrix"

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", _freeze_metadata_value(self.rows))


@dataclass(frozen=True)
class FiniteDifferenceOracleMetadata:
    available: bool
    step: float | None = None
    source: str = "provider.finite_difference_oracle"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(self, "step", None if self.step is None else float(self.step))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))


@dataclass(frozen=True)
class ReadinessBlockerMetadata:
    blockers: tuple[str, ...]
    blocker_table: tuple[tuple[str, str], ...]
    blocker_summary: str
    final_ready: bool
    final_readiness_error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "blockers", tuple(str(blocker) for blocker in self.blockers))
        object.__setattr__(
            self,
            "blocker_table",
            tuple((str(name), str(status)) for name, status in self.blocker_table),
        )
        object.__setattr__(self, "blocker_summary", str(self.blocker_summary))
        object.__setattr__(self, "final_ready", bool(self.final_ready))
        if self.final_readiness_error is not None:
            object.__setattr__(self, "final_readiness_error", str(self.final_readiness_error))


@dataclass(frozen=True)
class IdentificationEvidenceMetadata:
    rows: tuple[tuple[tuple[str, Any], ...], ...]
    source: str = "provider.identification_evidence_status"

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", _freeze_metadata_value(self.rows))


@dataclass(frozen=True)
class SparseBackendPolicyMetadata:
    rows: tuple[tuple[tuple[str, Any], ...], ...]
    source: str = "provider.sparse_derivative_backend_policy"

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", _freeze_metadata_value(self.rows))


@dataclass(frozen=True)
class MacroFinanceHMCGateResult:
    readiness: MacroFinanceHMCReadinessResult
    value_finite: bool
    score_finite: bool
    negative_hessian_finite: bool
    negative_hessian_symmetric: bool
    eager_compiled_parity: bool | None = None
    max_abs_value_diff: float | None = None
    max_abs_score_diff: float | None = None
    target_ready: bool = False
    convergence_claim: str = "not_claimed"


@dataclass(frozen=True)
class LargeScaleAdaptationGateResult:
    mask_metadata: ObservationMaskMetadata
    requested_derivative_order: int
    masked_derivative_order_supported: int
    masked_support_source: str
    production_mode: bool
    likelihood_adaptation_ready: bool
    blockers: tuple[str, ...]
    source: str = "large_scale_adaptation_gate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_derivative_order", int(self.requested_derivative_order))
        object.__setattr__(
            self,
            "masked_derivative_order_supported",
            int(self.masked_derivative_order_supported),
        )
        object.__setattr__(self, "masked_support_source", str(self.masked_support_source))
        object.__setattr__(self, "production_mode", bool(self.production_mode))
        object.__setattr__(self, "likelihood_adaptation_ready", bool(self.likelihood_adaptation_ready))
        object.__setattr__(self, "blockers", tuple(str(blocker) for blocker in self.blockers))


@dataclass(frozen=True)
class CrossCurrencyDerivativeGateResult:
    coverage: DerivativeCoverageMetadata
    provider_parameter_names: tuple[str, ...]
    covered_parameter_names: tuple[str, ...]
    missing_parameter_names: tuple[str, ...]
    extra_parameter_names: tuple[str, ...]
    coverage_complete: bool
    oracle_checked: bool
    oracle_passed: bool | None
    max_abs_oracle_discrepancy: float | None
    oracle_tolerance: float
    required_oracle_blocks: tuple[str, ...]
    checked_oracle_blocks: tuple[str, ...]
    missing_oracle_blocks: tuple[str, ...]
    adaptation_ready: bool
    blockers: tuple[str, ...]
    source: str = "cross_currency_derivative_gate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_parameter_names", tuple(str(name) for name in self.provider_parameter_names))
        object.__setattr__(self, "covered_parameter_names", tuple(str(name) for name in self.covered_parameter_names))
        object.__setattr__(self, "missing_parameter_names", tuple(str(name) for name in self.missing_parameter_names))
        object.__setattr__(self, "extra_parameter_names", tuple(str(name) for name in self.extra_parameter_names))
        object.__setattr__(self, "coverage_complete", bool(self.coverage_complete))
        object.__setattr__(self, "oracle_checked", bool(self.oracle_checked))
        if self.oracle_passed is not None:
            object.__setattr__(self, "oracle_passed", bool(self.oracle_passed))
        if self.max_abs_oracle_discrepancy is not None:
            object.__setattr__(
                self,
                "max_abs_oracle_discrepancy",
                float(self.max_abs_oracle_discrepancy),
            )
        object.__setattr__(self, "oracle_tolerance", float(self.oracle_tolerance))
        object.__setattr__(self, "required_oracle_blocks", tuple(str(block) for block in self.required_oracle_blocks))
        object.__setattr__(self, "checked_oracle_blocks", tuple(str(block) for block in self.checked_oracle_blocks))
        object.__setattr__(self, "missing_oracle_blocks", tuple(str(block) for block in self.missing_oracle_blocks))
        object.__setattr__(self, "adaptation_ready", bool(self.adaptation_ready))
        object.__setattr__(self, "blockers", tuple(str(blocker) for blocker in self.blockers))


@dataclass(frozen=True)
class ProductionExposureGateResult:
    readiness: ReadinessBlockerMetadata
    identification: IdentificationEvidenceMetadata
    sparse_policy: SparseBackendPolicyMetadata
    final_identification_ready: bool
    sparse_policy_available: bool
    exposure_ready: bool
    blockers: tuple[str, ...]
    source: str = "production_exposure_gate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "final_identification_ready", bool(self.final_identification_ready))
        object.__setattr__(self, "sparse_policy_available", bool(self.sparse_policy_available))
        object.__setattr__(self, "exposure_ready", bool(self.exposure_ready))
        object.__setattr__(self, "blockers", tuple(str(blocker) for blocker in self.blockers))


@dataclass(frozen=True)
class MacroFinanceHMCDiagnosticGateResult:
    target_gate: MacroFinanceHMCGateResult
    acceptance_rate: float
    divergence_count: int
    split_rhat: np.ndarray
    ess: np.ndarray
    min_ess: float
    max_split_rhat: float
    min_acceptance_rate: float
    max_acceptance_rate: float
    diagnostics_ready: bool
    blockers: tuple[str, ...]
    convergence_claim: str = "not_claimed"
    source: str = "hmc_diagnostic_gate"

    def __post_init__(self) -> None:
        split_rhat = np.asarray(self.split_rhat, dtype=float).copy()
        split_rhat.setflags(write=False)
        ess = np.asarray(self.ess, dtype=float).copy()
        ess.setflags(write=False)
        object.__setattr__(self, "acceptance_rate", float(self.acceptance_rate))
        object.__setattr__(self, "divergence_count", int(self.divergence_count))
        object.__setattr__(self, "split_rhat", split_rhat)
        object.__setattr__(self, "ess", ess)
        object.__setattr__(self, "min_ess", float(self.min_ess))
        object.__setattr__(self, "max_split_rhat", float(self.max_split_rhat))
        object.__setattr__(self, "min_acceptance_rate", float(self.min_acceptance_rate))
        object.__setattr__(self, "max_acceptance_rate", float(self.max_acceptance_rate))
        object.__setattr__(self, "diagnostics_ready", bool(self.diagnostics_ready))
        object.__setattr__(self, "blockers", tuple(str(blocker) for blocker in self.blockers))
        object.__setattr__(self, "convergence_claim", str(self.convergence_claim))


@dataclass(frozen=True)
class MacroFinanceHMCBackendComparisonResult:
    backend_results: tuple[tuple[str, MacroFinanceHMCDiagnosticGateResult], ...]
    comparison_ready: bool
    acceptance_rate_min: float
    acceptance_rate_max: float
    min_ess: float
    max_split_rhat: float
    blockers: tuple[str, ...]
    convergence_claim: str = "not_claimed"
    source: str = "hmc_backend_comparison_gate"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend_results",
            tuple((str(name), result) for name, result in self.backend_results),
        )
        object.__setattr__(self, "comparison_ready", bool(self.comparison_ready))
        object.__setattr__(self, "acceptance_rate_min", float(self.acceptance_rate_min))
        object.__setattr__(self, "acceptance_rate_max", float(self.acceptance_rate_max))
        object.__setattr__(self, "min_ess", float(self.min_ess))
        object.__setattr__(self, "max_split_rhat", float(self.max_split_rhat))
        object.__setattr__(self, "blockers", tuple(str(blocker) for blocker in self.blockers))
        object.__setattr__(self, "convergence_claim", str(self.convergence_claim))


def _as_array(value: Any) -> np.ndarray:
    """Convert NumPy, TensorFlow eager tensors, or array-like objects."""

    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def _freeze_metadata_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _freeze_metadata_value(value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            (field.name, _freeze_metadata_value(getattr(value, field.name)))
            for field in fields(value)
        )
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_metadata_value(nested_value))
            for key, nested_value in value.items()
        )
    if isinstance(value, tuple):
        return tuple(_freeze_metadata_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_metadata_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_metadata_value(item) for item in value), key=repr))
    return value


def _row_to_items(row: Any) -> tuple[tuple[str, Any], ...]:
    if is_dataclass(row) and not isinstance(row, type):
        values = {field.name: getattr(row, field.name) for field in fields(row)}
    elif hasattr(row, "_asdict"):
        values = row._asdict()
    elif isinstance(row, dict):
        values = row
    else:
        public = {
            name: getattr(row, name)
            for name in dir(row)
            if not name.startswith("_") and not callable(getattr(row, name))
        }
        values = public if public else {"value": row}
    return tuple((str(key), _freeze_metadata_value(value)) for key, value in values.items())


def _rows_to_tuple(rows: Any) -> tuple[tuple[tuple[str, Any], ...], ...]:
    return tuple(_row_to_items(row) for row in rows)


def macrofinance_lgssm_to_bayesfilter(
    model: MacroFinanceLGSSMLike,
    *,
    partition: StatePartition | None = None,
) -> LinearGaussianStateSpace:
    """Convert a MacroFinance-shaped LGSSM object to BayesFilter's value object."""

    return LinearGaussianStateSpace(
        initial_mean=_as_array(model.initial_mean),
        initial_covariance=_as_array(model.initial_covariance),
        transition_offset=_as_array(model.transition_offset),
        transition_matrix=_as_array(model.transition_matrix),
        transition_covariance=_as_array(model.transition_covariance),
        observation_offset=_as_array(model.observation_offset),
        observation_matrix=_as_array(model.observation_matrix),
        observation_covariance=_as_array(model.observation_covariance),
        partition=partition,
    )


def _parameter_names(provider: Any, parameter_dim: int) -> tuple[str, ...]:
    if hasattr(provider, "parameter_names"):
        names = tuple(str(name) for name in provider.parameter_names())
        if len(names) != parameter_dim:
            raise ValueError("provider parameter_names length does not match theta")
        return names
    return tuple(f"theta_{idx}" for idx in range(parameter_dim))


def evaluate_macrofinance_provider_likelihood(
    provider: MacroFinanceProviderLike,
    theta: np.ndarray,
    observations: np.ndarray,
    *,
    partition: StatePartition | None = None,
    mask: np.ndarray | None = None,
    jitter: float = 0.0,
) -> MacroFinanceLikelihoodResult:
    """Evaluate a MacroFinance provider through BayesFilter's exact LGSSM path."""

    theta_array = np.asarray(theta, dtype=float)
    model = macrofinance_lgssm_to_bayesfilter(
        provider.build_state_space(theta_array),
        partition=partition,
    )
    result = kalman_log_likelihood(
        observations,
        model,
        mask=mask,
        jitter=jitter,
    )
    return MacroFinanceLikelihoodResult(
        log_likelihood=result.log_likelihood,
        kalman_result=result,
        parameter_names=_parameter_names(provider, theta_array.shape[0]),
        theta=theta_array.copy(),
        jitter=float(jitter),
    )


def evaluate_macrofinance_provider_derivatives(
    provider: MacroFinanceDerivativeProviderLike,
    theta: np.ndarray,
    observations: np.ndarray,
    backend: Any,
    *,
    derivative_order: int,
    backend_name: str,
    jitter: float | None = None,
    initial_mean_policy: str | None = None,
) -> MacroFinanceDerivativeResult:
    """Delegate value/score/Hessian evaluation to a MacroFinance-compatible backend.

    `backend` must accept `(observations, model, derivatives, **kwargs)` and
    return either `(loglik, score)` for first order or `(loglik, score, hessian)`
    for second order.  BayesFilter records the result shape and provenance but
    does not implement the derivative recursion here.
    """

    if derivative_order not in {1, 2}:
        raise ValueError("derivative_order must be 1 or 2")
    theta_array = np.asarray(theta, dtype=float)
    model, derivatives = provider.build_state_space_with_derivatives(
        theta_array,
        order=derivative_order,
    )
    kwargs = {}
    if jitter is not None:
        kwargs["jitter"] = float(jitter)
    raw = backend(np.asarray(observations, dtype=float), model, derivatives, **kwargs)
    if len(raw) < 2:
        raise ValueError("derivative backend must return at least log likelihood and score")
    if derivative_order == 1:
        log_likelihood, score = raw[:2]
        hessian = None
    else:
        if len(raw) < 3:
            raise ValueError("second-order derivative backend must return a Hessian")
        log_likelihood, score, hessian = raw[:3]

    score_array = np.asarray(_as_array(score), dtype=float)
    hessian_array = None if hessian is None else np.asarray(_as_array(hessian), dtype=float)
    parameter_names = _parameter_names(provider, theta_array.shape[0])
    if score_array.shape != (len(parameter_names),):
        raise ValueError("score shape does not match parameter names")
    if hessian_array is not None and hessian_array.shape != (len(parameter_names), len(parameter_names)):
        raise ValueError("hessian shape does not match parameter names")

    return MacroFinanceDerivativeResult(
        log_likelihood=float(_as_array(log_likelihood)),
        score=score_array,
        hessian=hessian_array,
        parameter_names=parameter_names,
        theta=theta_array.copy(),
        derivative_order=derivative_order,
        backend=str(backend_name),
        jitter=None if jitter is None else float(jitter),
        initial_mean_policy=initial_mean_policy,
    )


def evaluate_macrofinance_hmc_readiness(
    adapter: Any,
    theta: np.ndarray | None = None,
    *,
    backend_name: str | None = None,
) -> MacroFinanceHMCReadinessResult:
    """Check finite HMC-target operations without running or endorsing a sampler."""

    if theta is None:
        if not hasattr(adapter, "initial_position"):
            raise ValueError("theta is required when adapter has no initial_position")
        theta = adapter.initial_position()
    theta_array = np.asarray(theta, dtype=float)
    if hasattr(adapter, "log_prob_and_grad"):
        log_prob, score = adapter.log_prob_and_grad(theta_array)
    else:
        log_prob = adapter.log_prob(theta_array)
        score = adapter.grad_log_prob(theta_array)
    if not hasattr(adapter, "negative_log_prob_and_gradient"):
        raise ValueError("adapter must expose negative_log_prob_and_gradient")
    if not hasattr(adapter, "negative_log_prob_hessian"):
        raise ValueError("adapter must expose negative_log_prob_hessian")
    negative_log_prob, negative_gradient = adapter.negative_log_prob_and_gradient(theta_array)
    negative_hessian = adapter.negative_log_prob_hessian(theta_array)

    score_array = np.asarray(_as_array(score), dtype=float)
    negative_gradient_array = np.asarray(_as_array(negative_gradient), dtype=float)
    negative_hessian_array = np.asarray(_as_array(negative_hessian), dtype=float)
    names = tuple(str(name) for name in adapter.parameter_names())
    if score_array.shape != (len(names),):
        raise ValueError("score shape does not match parameter names")
    if negative_gradient_array.shape != (len(names),):
        raise ValueError("negative gradient shape does not match parameter names")
    if negative_hessian_array.shape != (len(names), len(names)):
        raise ValueError("negative Hessian shape does not match parameter names")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("score is not finite")
    if not np.all(np.isfinite(negative_gradient_array)):
        raise ValueError("negative gradient is not finite")
    if not np.all(np.isfinite(negative_hessian_array)):
        raise ValueError("negative Hessian is not finite")
    log_prob_float = float(_as_array(log_prob))
    negative_log_prob_float = float(_as_array(negative_log_prob))
    if not np.isfinite(log_prob_float) or not np.isfinite(negative_log_prob_float):
        raise ValueError("log probability values must be finite")

    return MacroFinanceHMCReadinessResult(
        parameter_names=names,
        theta=theta_array.copy(),
        log_prob=log_prob_float,
        score=score_array,
        negative_log_prob=negative_log_prob_float,
        negative_gradient=negative_gradient_array,
        negative_hessian=negative_hessian_array,
        backend=backend_name or str(getattr(adapter, "analytic_backend", "unknown")),
    )


def extract_parameter_unit_metadata(provider: Any) -> ParameterUnitMetadata:
    """Extract parameter-unit metadata from a MacroFinance-like provider."""

    if not hasattr(provider, "parameter_units"):
        raise ValueError("provider does not expose parameter_units")
    if not hasattr(provider, "parameter_names"):
        raise ValueError("provider does not expose parameter_names")
    names = tuple(str(name) for name in provider.parameter_names())
    units = np.asarray(_as_array(provider.parameter_units()), dtype=float)
    if units.shape != (len(names),):
        raise ValueError("parameter_units shape does not match parameter_names")
    if not np.all(np.isfinite(units)):
        raise ValueError("parameter_units must be finite")
    return ParameterUnitMetadata(parameter_names=names, units=units)


def extract_observation_mask_metadata(
    provider: Any,
    *,
    mask: Any | None = None,
    source: str = "provider.observation_mask",
) -> ObservationMaskMetadata:
    """Extract observation-mask policy metadata using True-means-observed convention."""

    if mask is None:
        if hasattr(provider, "observation_mask"):
            mask = provider.observation_mask()
        elif hasattr(provider, "observations"):
            mask = np.isfinite(_as_array(provider.observations()))
            source = "finite_observations"
        else:
            raise ValueError("provider exposes neither observation_mask nor observations")
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim != 2:
        raise ValueError("observation mask must be two-dimensional")
    observed_count = int(np.count_nonzero(mask_array))
    missing_count = int(mask_array.size - observed_count)
    return ObservationMaskMetadata(
        shape=tuple(int(v) for v in mask_array.shape),
        observed_count=observed_count,
        missing_count=missing_count,
        all_observed=missing_count == 0,
        source=source,
    )


def evaluate_large_scale_adaptation_gate(
    provider: Any,
    *,
    mask: Any | None = None,
    requested_derivative_order: int = 2,
    masked_derivative_order_supported: int | None = None,
    production_mode: bool = False,
) -> LargeScaleAdaptationGateResult:
    """Gate large-scale likelihood adaptation on dense panels or masked support."""

    requested_order = int(requested_derivative_order)
    if requested_order < 0:
        raise ValueError("requested_derivative_order must be nonnegative")
    mask_metadata = extract_observation_mask_metadata(provider, mask=mask)
    if masked_derivative_order_supported is None:
        supported = int(getattr(provider, "masked_derivative_order_supported", 0))
        support_source = (
            "provider.masked_derivative_order_supported"
            if hasattr(provider, "masked_derivative_order_supported")
            else "not_declared"
        )
    else:
        supported = int(masked_derivative_order_supported)
        support_source = "caller_override"
    blockers: list[str] = []
    if not mask_metadata.all_observed and supported < requested_order:
        blockers.append(
            "sparse observations require masked derivative support through "
            f"order {requested_order}; provider reports order {supported}"
        )
    if (
        bool(production_mode)
        and not mask_metadata.all_observed
        and support_source == "caller_override"
    ):
        blockers.append("production sparse readiness requires provider-owned masked derivative support")
    return LargeScaleAdaptationGateResult(
        mask_metadata=mask_metadata,
        requested_derivative_order=requested_order,
        masked_derivative_order_supported=supported,
        masked_support_source=support_source,
        production_mode=bool(production_mode),
        likelihood_adaptation_ready=not blockers,
        blockers=tuple(blockers),
    )


def extract_derivative_coverage_metadata(
    provider: Any,
    *,
    source: str = "provider.derivative_coverage_matrix",
) -> DerivativeCoverageMetadata:
    """Normalize derivative-coverage rows without assigning readiness claims."""

    if not hasattr(provider, "derivative_coverage_matrix"):
        raise ValueError("provider does not expose derivative_coverage_matrix")
    rows = _rows_to_tuple(provider.derivative_coverage_matrix())
    if not rows:
        raise ValueError("derivative_coverage_matrix returned no rows")
    return DerivativeCoverageMetadata(rows=rows, source=source)


def extract_finite_difference_oracle_metadata(
    provider: Any,
    *,
    source: str = "provider.finite_difference_oracle",
) -> FiniteDifferenceOracleMetadata:
    """Record finite-difference oracle provenance when a provider exposes it."""

    if hasattr(provider, "finite_difference_oracle"):
        oracle = provider.finite_difference_oracle()
        step = getattr(oracle, "finite_difference_step", getattr(provider, "finite_difference_step", None))
        return FiniteDifferenceOracleMetadata(
            available=True,
            step=None if step is None else float(step),
            source=source,
            notes=(type(oracle).__name__,),
        )
    if hasattr(provider, "finite_difference_step"):
        return FiniteDifferenceOracleMetadata(
            available=False,
            step=float(getattr(provider, "finite_difference_step")),
            source="provider.finite_difference_step",
            notes=("provider_exposes_step_but_no_oracle_method",),
        )
    return FiniteDifferenceOracleMetadata(
        available=False,
        source=source,
        notes=("not_exposed",),
    )


def _row_value(row: tuple[tuple[str, Any], ...], key: str, default: Any = None) -> Any:
    return dict(row).get(key, default)


def _covered_parameter_names(coverage: DerivativeCoverageMetadata) -> tuple[str, ...]:
    covered: set[str] = set()
    for row in coverage.rows:
        names = _row_value(row, "parameter_names", tuple())
        if isinstance(names, str):
            covered.add(names)
        else:
            covered.update(str(name) for name in names)
    return tuple(sorted(covered))


def evaluate_cross_currency_derivative_gate(
    provider: Any,
    *,
    oracle_check: Any | None = None,
    oracle_tolerance: float = 1e-6,
    required_oracle_blocks: tuple[str, ...] = (),
) -> CrossCurrencyDerivativeGateResult:
    """Gate cross-currency adaptation on parameter coverage and oracle evidence."""

    coverage = extract_derivative_coverage_metadata(provider)
    provider_names = tuple(str(name) for name in provider.parameter_names())
    covered_names = _covered_parameter_names(coverage)
    provider_set = set(provider_names)
    covered_set = set(covered_names)
    missing = tuple(name for name in provider_names if name not in covered_set)
    extra = tuple(name for name in covered_names if name not in provider_set)
    coverage_complete = not missing
    blockers: list[str] = []
    if missing:
        blockers.append(f"derivative coverage missing {len(missing)} provider parameters")

    oracle_checked = oracle_check is not None
    oracle_passed: bool | None = None
    max_abs_oracle_discrepancy: float | None = None
    required_blocks = tuple(str(block) for block in required_oracle_blocks)
    checked_blocks: tuple[str, ...] = tuple()
    missing_blocks: tuple[str, ...] = tuple(required_blocks)
    if oracle_check is not None:
        raw = oracle_check(provider)
        if isinstance(raw, Mapping):
            discrepancy = raw.get("max_abs_oracle_discrepancy", raw.get("max_abs_discrepancy"))
            raw_blocks = raw.get("checked_blocks", raw.get("oracle_blocks", tuple()))
            checked_blocks = tuple(str(block) for block in raw_blocks)
        else:
            discrepancy = raw
        max_abs_oracle_discrepancy = float(discrepancy)
        oracle_passed = bool(np.isfinite(max_abs_oracle_discrepancy) and max_abs_oracle_discrepancy <= float(oracle_tolerance))
        if not oracle_passed:
            blockers.append(
                "finite-difference oracle discrepancy "
                f"{max_abs_oracle_discrepancy:.3e} exceeds tolerance {float(oracle_tolerance):.3e}"
            )
        checked_set = set(checked_blocks)
        missing_blocks = tuple(block for block in required_blocks if block not in checked_set)
    if required_blocks and not oracle_checked:
        blockers.append("required finite-difference oracle blocks were not checked")
    elif missing_blocks:
        blockers.append(
            "finite-difference oracle missing required blocks: "
            + ", ".join(missing_blocks)
        )

    return CrossCurrencyDerivativeGateResult(
        coverage=coverage,
        provider_parameter_names=provider_names,
        covered_parameter_names=covered_names,
        missing_parameter_names=missing,
        extra_parameter_names=extra,
        coverage_complete=coverage_complete,
        oracle_checked=oracle_checked,
        oracle_passed=oracle_passed,
        max_abs_oracle_discrepancy=max_abs_oracle_discrepancy,
        oracle_tolerance=float(oracle_tolerance),
        required_oracle_blocks=required_blocks,
        checked_oracle_blocks=checked_blocks,
        missing_oracle_blocks=missing_blocks,
        adaptation_ready=not blockers,
        blockers=tuple(blockers),
    )


def _call_or_value(provider: Any, name: str) -> Any:
    value = getattr(provider, name)
    return value() if callable(value) else value


def _string_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return tuple()
    if isinstance(values, str):
        return (values,)
    return tuple(str(value) for value in values)


def _blocker_table(provider: Any, blockers: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    if hasattr(provider, "readiness_blocker_table"):
        rows = _call_or_value(provider, "readiness_blocker_table")
    elif hasattr(provider, "blocker_table"):
        rows = _call_or_value(provider, "blocker_table")
    else:
        rows = tuple((f"blocker_{idx}", blocker) for idx, blocker in enumerate(blockers))
    return tuple((str(row[0]), str(row[1])) for row in rows)


def extract_readiness_blocker_metadata(provider: Any) -> ReadinessBlockerMetadata:
    """Normalize production-readiness blockers and fail closed on final readiness."""

    blockers = _string_tuple(_call_or_value(provider, "blockers")) if hasattr(provider, "blockers") else tuple()
    blocker_table = _blocker_table(provider, blockers)
    if hasattr(provider, "blocker_summary"):
        blocker_summary = str(_call_or_value(provider, "blocker_summary"))
    else:
        blocker_summary = "; ".join(blockers)

    final_ready = False
    final_readiness_error: str | None = None
    if hasattr(provider, "validate_final_ten_country_ready"):
        try:
            provider.validate_final_ten_country_ready()
            final_ready = True
        except Exception as exc:
            final_readiness_error = str(exc)
    else:
        final_readiness_error = "provider does not expose validate_final_ten_country_ready"

    if blockers:
        final_ready = False
        if final_readiness_error is None:
            final_readiness_error = "provider reports readiness blockers"

    return ReadinessBlockerMetadata(
        blockers=blockers,
        blocker_table=blocker_table,
        blocker_summary=blocker_summary,
        final_ready=final_ready,
        final_readiness_error=final_readiness_error,
    )


def extract_identification_evidence_metadata(
    provider: Any,
    *,
    source: str = "provider.identification_evidence_status",
) -> IdentificationEvidenceMetadata:
    """Normalize provider identification-evidence rows."""

    if not hasattr(provider, "identification_evidence_status"):
        raise ValueError("provider does not expose identification_evidence_status")
    rows = _rows_to_tuple(provider.identification_evidence_status())
    if not rows:
        raise ValueError("identification_evidence_status returned no rows")
    return IdentificationEvidenceMetadata(rows=rows, source=source)


def extract_sparse_backend_policy_metadata(
    provider: Any,
    *,
    source: str = "provider.sparse_derivative_backend_policy",
) -> SparseBackendPolicyMetadata:
    """Normalize sparse derivative backend policy rows."""

    if not hasattr(provider, "sparse_derivative_backend_policy"):
        raise ValueError("provider does not expose sparse_derivative_backend_policy")
    rows = _rows_to_tuple(provider.sparse_derivative_backend_policy())
    if not rows:
        raise ValueError("sparse_derivative_backend_policy returned no rows")
    return SparseBackendPolicyMetadata(rows=rows, source=source)


def _identification_is_final(row: tuple[tuple[str, Any], ...]) -> bool:
    trust_status = str(_row_value(row, "trust_status", ""))
    if trust_status != "Identified":
        return False
    text_parts = [trust_status]
    for _, value in row:
        if isinstance(value, tuple):
            text_parts.extend(str(item) for item in value)
        else:
            text_parts.append(str(value))
    text = " ".join(text_parts).lower()
    blocked_tokens = ("weak", "fixture", "synthetic", "blocked", "not_final")
    return not any(token in text for token in blocked_tokens)


def evaluate_production_exposure_gate(provider: Any) -> ProductionExposureGateResult:
    """Gate production exposure on final readiness and final identification evidence."""

    readiness = extract_readiness_blocker_metadata(provider)
    identification = extract_identification_evidence_metadata(provider)
    sparse_policy = extract_sparse_backend_policy_metadata(provider)
    final_identification_ready = bool(identification.rows) and all(
        _identification_is_final(row) for row in identification.rows
    )
    sparse_policy_available = bool(sparse_policy.rows)
    blockers: list[str] = []
    if not readiness.final_ready:
        blockers.append("final readiness validation failed")
    if readiness.blockers:
        blockers.append(f"provider reports {len(readiness.blockers)} readiness blockers")
    if not final_identification_ready:
        blockers.append("identification evidence is not final-data Identified evidence")
    if not sparse_policy_available:
        blockers.append("sparse derivative backend policy is missing")

    return ProductionExposureGateResult(
        readiness=readiness,
        identification=identification,
        sparse_policy=sparse_policy,
        final_identification_ready=final_identification_ready,
        sparse_policy_available=sparse_policy_available,
        exposure_ready=not blockers,
        blockers=tuple(blockers),
    )


def evaluate_macrofinance_hmc_gate(
    adapter: Any,
    theta: np.ndarray | None = None,
    *,
    backend_name: str | None = None,
    compiled_log_prob_and_grad: Any | None = None,
    parity_tolerance: float = 1e-8,
    symmetry_tolerance: float = 1e-8,
) -> MacroFinanceHMCGateResult:
    """Evaluate BayesFilter-context HMC target gates without running a sampler."""

    readiness = evaluate_macrofinance_hmc_readiness(
        adapter,
        theta,
        backend_name=backend_name,
    )
    value_finite = bool(np.isfinite(readiness.log_prob) and np.isfinite(readiness.negative_log_prob))
    score_finite = bool(
        np.all(np.isfinite(readiness.score))
        and np.all(np.isfinite(readiness.negative_gradient))
    )
    negative_hessian_finite = bool(np.all(np.isfinite(readiness.negative_hessian)))
    negative_hessian_symmetric = bool(
        np.allclose(
            readiness.negative_hessian,
            readiness.negative_hessian.T,
            rtol=0.0,
            atol=float(symmetry_tolerance),
        )
    )

    eager_compiled_parity: bool | None = None
    max_abs_value_diff: float | None = None
    max_abs_score_diff: float | None = None
    if compiled_log_prob_and_grad is not None:
        compiled_value, compiled_score = compiled_log_prob_and_grad(readiness.theta.copy())
        compiled_value_float = float(_as_array(compiled_value))
        compiled_score_array = np.asarray(_as_array(compiled_score), dtype=float)
        if compiled_score_array.shape != readiness.score.shape:
            raise ValueError("compiled score shape does not match eager score")
        max_abs_value_diff = abs(compiled_value_float - readiness.log_prob)
        max_abs_score_diff = float(np.max(np.abs(compiled_score_array - readiness.score)))
        eager_compiled_parity = bool(
            max_abs_value_diff <= float(parity_tolerance)
            and max_abs_score_diff <= float(parity_tolerance)
        )

    target_ready = bool(
        value_finite
        and score_finite
        and negative_hessian_finite
        and negative_hessian_symmetric
        and eager_compiled_parity is not False
    )
    return MacroFinanceHMCGateResult(
        readiness=readiness,
        value_finite=value_finite,
        score_finite=score_finite,
        negative_hessian_finite=negative_hessian_finite,
        negative_hessian_symmetric=negative_hessian_symmetric,
        eager_compiled_parity=eager_compiled_parity,
        max_abs_value_diff=max_abs_value_diff,
        max_abs_score_diff=max_abs_score_diff,
        target_ready=target_ready,
        convergence_claim=readiness.convergence_claim,
    )


def _diagnostic_value(diagnostics: Any, name: str) -> Any:
    if isinstance(diagnostics, Mapping):
        return diagnostics[name]
    return getattr(diagnostics, name)


def evaluate_macrofinance_hmc_diagnostic_gate(
    target_gate: MacroFinanceHMCGateResult,
    diagnostics: Any,
    *,
    min_ess: float = 50.0,
    max_split_rhat: float = 1.01,
    min_acceptance_rate: float = 0.6,
    max_acceptance_rate: float = 0.95,
) -> MacroFinanceHMCDiagnosticGateResult:
    """Gate HMC sampler readiness on target readiness plus chain diagnostics."""

    acceptance_rate = float(_diagnostic_value(diagnostics, "acceptance_rate"))
    divergence_count = int(_diagnostic_value(diagnostics, "divergence_count"))
    split_rhat = np.asarray(_diagnostic_value(diagnostics, "split_rhat"), dtype=float)
    ess = np.asarray(_diagnostic_value(diagnostics, "ess"), dtype=float)
    blockers: list[str] = []
    if not target_gate.target_ready:
        blockers.append("target gate is not ready")
    if not np.isfinite(acceptance_rate):
        blockers.append("acceptance rate is not finite")
    elif not (float(min_acceptance_rate) <= acceptance_rate <= float(max_acceptance_rate)):
        blockers.append(
            f"acceptance rate {acceptance_rate:.3f} outside "
            f"[{float(min_acceptance_rate):.3f}, {float(max_acceptance_rate):.3f}]"
        )
    if divergence_count != 0:
        blockers.append(f"divergence count is {divergence_count}")
    if split_rhat.size == 0 or not np.all(np.isfinite(split_rhat)):
        blockers.append("split R-hat diagnostics are missing or nonfinite")
    elif float(np.max(split_rhat)) > float(max_split_rhat):
        blockers.append(
            f"max split R-hat {float(np.max(split_rhat)):.3f} exceeds {float(max_split_rhat):.3f}"
        )
    if ess.size == 0 or not np.all(np.isfinite(ess)):
        blockers.append("ESS diagnostics are missing or nonfinite")
    elif float(np.min(ess)) < float(min_ess):
        blockers.append(f"min ESS {float(np.min(ess)):.3f} below {float(min_ess):.3f}")

    diagnostics_ready = not blockers
    return MacroFinanceHMCDiagnosticGateResult(
        target_gate=target_gate,
        acceptance_rate=acceptance_rate,
        divergence_count=divergence_count,
        split_rhat=split_rhat,
        ess=ess,
        min_ess=float(min_ess),
        max_split_rhat=float(max_split_rhat),
        min_acceptance_rate=float(min_acceptance_rate),
        max_acceptance_rate=float(max_acceptance_rate),
        diagnostics_ready=diagnostics_ready,
        blockers=tuple(blockers),
        convergence_claim="diagnostics_thresholds_passed" if diagnostics_ready else "not_claimed",
    )


def compare_macrofinance_hmc_backend_diagnostics(
    backend_diagnostics: Mapping[str, tuple[MacroFinanceHMCGateResult, Any]],
    *,
    min_ess: float = 50.0,
    max_split_rhat: float = 1.01,
    min_acceptance_rate: float = 0.6,
    max_acceptance_rate: float = 0.95,
) -> MacroFinanceHMCBackendComparisonResult:
    """Compare supplied HMC diagnostics across named backend targets."""

    if not backend_diagnostics:
        raise ValueError("backend_diagnostics must not be empty")
    results: list[tuple[str, MacroFinanceHMCDiagnosticGateResult]] = []
    blockers: list[str] = []
    for backend_name, (target_gate, diagnostics) in backend_diagnostics.items():
        result = evaluate_macrofinance_hmc_diagnostic_gate(
            target_gate,
            diagnostics,
            min_ess=min_ess,
            max_split_rhat=max_split_rhat,
            min_acceptance_rate=min_acceptance_rate,
            max_acceptance_rate=max_acceptance_rate,
        )
        results.append((str(backend_name), result))
        if not result.diagnostics_ready:
            blockers.append(
                f"{backend_name} diagnostics blocked: " + "; ".join(result.blockers)
            )

    acceptance_rates = [result.acceptance_rate for _, result in results]
    ess_values = [
        float(np.min(result.ess)) if result.ess.size else float("nan")
        for _, result in results
    ]
    rhat_values = [
        float(np.max(result.split_rhat)) if result.split_rhat.size else float("nan")
        for _, result in results
    ]
    comparison_ready = not blockers
    return MacroFinanceHMCBackendComparisonResult(
        backend_results=tuple(results),
        comparison_ready=comparison_ready,
        acceptance_rate_min=float(np.min(acceptance_rates)),
        acceptance_rate_max=float(np.max(acceptance_rates)),
        min_ess=float(np.min(ess_values)),
        max_split_rhat=float(np.max(rhat_values)),
        blockers=tuple(blockers),
        convergence_claim=(
            "diagnostics_thresholds_passed" if comparison_ready else "not_claimed"
        ),
    )
