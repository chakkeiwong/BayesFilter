"""Adapters for MacroFinance-shaped linear Gaussian state-space objects.

The functions here use structural typing: BayesFilter does not import
MacroFinance at package import time.  Client projects keep ownership of their
financial model construction and derivative recursions; BayesFilter only
normalizes generic LGSSM objects and result metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def _as_array(value: Any) -> np.ndarray:
    """Convert NumPy, TensorFlow eager tensors, or array-like objects."""

    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


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
