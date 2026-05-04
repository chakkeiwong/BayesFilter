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
