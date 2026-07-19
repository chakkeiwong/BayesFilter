"""Operational interleaved windowed warmup for fixed-trajectory TF/TFP HMC.

Each window executes real HMC transitions under one immutable affine transform.
At slow-window boundaries, covariance evidence may rebuild the transform; the
canonical theta endpoint is then mapped into the new coordinates and the next
TF/TFP runner is rebuilt. Warmup draws are adaptation inputs, not posterior
samples or convergence evidence.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from bayesfilter.hmc_route_contract import (
    HMC_ROUTE_CONTRACT_VERSION,
    OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
)
from bayesfilter.inference.batched_value_score import reviewed_value_score_target_fn
from bayesfilter.inference.hmc_coordinates import (
    AffineCoordinateTransform,
    KernelState,
    MomentumMetric,
    PositionCovarianceEstimate,
    WarmupTrajectoryPolicy,
)
from bayesfilter.inference.hmc_tuning import (
    WindowedMassAdaptationConfig,
    WindowedWarmupWindow,
    build_windowed_warmup_schedule,
    welford_covariance,
)
from bayesfilter.inference.hmc_verification import (
    _evaluate_retained_target_health,
    target_status_telemetry_has_failure,
)
from bayesfilter.inference.posterior_adapter import (
    ValueScoreCapability,
    value_score_capability,
)


OPERATIONAL_WARMUP_NONCLAIMS = (
    "operational interleaved HMC warmup engineering evidence only",
    "warmup draws are adaptation inputs and not posterior samples",
    "native divergence may be unavailable from the active TFP HMC kernel",
    "no posterior convergence claim",
    "no sampler superiority claim",
    "no GPU or XLA readiness claim",
)


def _stable_hash(label: str, payload: Mapping[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{label}:{normalized}".encode("utf-8")).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _strict_integer(
    value: Any,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer scalar")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _target_status_policy(value: Any) -> str:
    policy = str(value)
    if policy not in {"none", "per_chain_step"}:
        raise ValueError(
            "target_status_trace_policy must be 'none' or 'per_chain_step'"
        )
    return policy


def _strict_seed(value: Any, *, name: str) -> tuple[int, int]:
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly two integer scalars") from exc
    if len(raw) != 2:
        raise ValueError(f"{name} must contain exactly two integer scalars")
    return tuple(
        _strict_integer(item, name=f"{name} item", minimum=0) for item in raw
    )


def _seed(root: tuple[int, int], index: int, lane: int = 0) -> tuple[int, int]:
    normalized = _strict_seed(root, name="seed root")
    normalized_index = _strict_integer(index, name="seed index")
    normalized_lane = _strict_integer(lane, name="seed lane", minimum=0)
    return (
        normalized[0],
        normalized[1]
        + 1009 * (normalized_index + 1)
        + 7919 * normalized_lane,
    )


@dataclass(frozen=True)
class MetricAdequacyDecision:
    """Dense/diagonal/retain decision for one slow warmup window."""

    outcome: str
    covariance: Any | None
    estimator_family: str | None
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        outcome = str(self.outcome)
        allowed = {
            "dense_update",
            "diagonal_fallback",
            "no_update_insufficient_metric_evidence",
            "candidate_metric_rejected",
        }
        if outcome not in allowed:
            raise ValueError("unsupported metric adequacy outcome")
        covariance = None
        if self.covariance is not None:
            covariance = np.asarray(self.covariance, dtype=float).copy()
            if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
                raise ValueError("adequate covariance must be square")
            if not np.all(np.isfinite(covariance)):
                raise ValueError("adequate covariance must be finite")
            covariance.setflags(write=False)
        if outcome in {
            "no_update_insufficient_metric_evidence",
            "candidate_metric_rejected",
        }:
            if covariance is not None or self.estimator_family is not None:
                raise ValueError("no-update decision cannot carry an estimator")
        elif covariance is None or not self.estimator_family:
            raise ValueError("metric update requires covariance and estimator family")
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(
            self,
            "estimator_family",
            None if self.estimator_family is None else str(self.estimator_family),
        )
        object.__setattr__(self, "report", dict(self.report))

    @property
    def update_applied(self) -> bool:
        return self.outcome in {"dense_update", "diagonal_fallback"}

    def payload(self) -> Mapping[str, Any]:
        return {
            "outcome": self.outcome,
            "estimator_family": self.estimator_family,
            "update_applied": self.update_applied,
            "report": self.report,
        }


def assess_metric_covariance(
    latent_states: Any,
    *,
    shrinkage: float = 0.25,
    dense_min_states: int | None = None,
    diagonal_min_states: int | None = None,
) -> MetricAdequacyDecision:
    """Apply the R0 dense-information and diagonal-fallback gates."""

    temporal = np.asarray(latent_states, dtype=float)
    if (
        temporal.ndim not in {2, 3}
        or temporal.shape[0] < 2
        or temporal.shape[-1] <= 0
    ):
        raise ValueError(
            "latent_states must be draw/dimension or draw/chain/dimension"
        )
    if not np.all(np.isfinite(temporal)):
        raise ValueError("latent_states must be finite")
    explicit_chains = temporal.ndim == 3
    chain_states = temporal[:, None, :] if not explicit_chains else temporal
    states = chain_states.reshape((-1, chain_states.shape[-1]))
    n, dimension = (int(states.shape[0]), int(states.shape[1]))
    dense_required = (
        max(64, 4 * dimension)
        if dense_min_states is None
        else _strict_integer(
            dense_min_states,
            name="dense_min_states",
            minimum=2,
        )
    )
    diagonal_required = (
        max(32, 2 * int(np.ceil(np.log2(dimension + 1))))
        if diagonal_min_states is None
        else _strict_integer(
            diagonal_min_states,
            name="diagonal_min_states",
            minimum=2,
        )
    )
    if dense_required < 2 or diagonal_required < 2:
        raise ValueError("metric adequacy minimums must be at least two")
    weight = float(shrinkage)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("shrinkage must be finite and in [0, 1]")

    ess_by_coordinate = _summed_chain_ess(chain_states)
    min_ess = float(np.min(ess_by_coordinate))
    dense_min_ess = max(8, dimension + 1)
    diagonal_min_ess = max(4, int(np.ceil(np.log2(dimension + 1))))
    split_rhat = _split_rhat_by_coordinate(chain_states)
    split_rhat_finite = bool(
        split_rhat is not None and np.all(np.isfinite(split_rhat))
    )
    max_split_rhat = (
        float(np.max(split_rhat)) if split_rhat_finite else None
    )
    dense_chain_compatible = bool(
        not explicit_chains
        or (np.all(np.isfinite(split_rhat)) and max_split_rhat <= 1.10)
    ) if split_rhat is not None else not explicit_chains
    diagonal_chain_compatible = bool(
        not explicit_chains
        or (np.all(np.isfinite(split_rhat)) and max_split_rhat <= 1.25)
    ) if split_rhat is not None else not explicit_chains

    welford = welford_covariance(states)
    empirical = np.asarray(welford.covariance, dtype=float)
    raw_rank = int(np.linalg.matrix_rank(empirical))
    raw_eigenvalues = np.linalg.eigvalsh(0.5 * (empirical + empirical.T))
    raw_positive = bool(np.all(raw_eigenvalues > 0.0))
    raw_condition = (
        float(np.max(raw_eigenvalues) / np.min(raw_eigenvalues))
        if raw_positive
        else float("inf")
    )
    identity = np.eye(dimension)
    dense_shrunk = (1.0 - weight) * empirical + weight * identity
    dense_discrepancy = float(
        np.linalg.norm(dense_shrunk - empirical, ord="fro")
        / max(np.linalg.norm(empirical, ord="fro"), np.finfo(float).eps)
    )
    dense_checks = {
        "state_count_sufficient": n >= dense_required,
        "effective_information_sufficient": min_ess >= dense_min_ess,
        "cross_chain_location_compatible": dense_chain_compatible,
        "full_raw_rank": raw_rank == dimension,
        "raw_condition_acceptable": raw_condition <= 1.0e8,
        "regularization_discrepancy_acceptable": dense_discrepancy <= 0.50,
    }
    common = {
        "state_count": n,
        "dimension": dimension,
        "dense_min_states": dense_required,
        "diagonal_min_states": diagonal_required,
        "ess_method": "geyer_initial_positive_sequence_fft_autocorrelation",
        "effective_sample_size_by_coordinate": tuple(
            float(item) for item in ess_by_coordinate
        ),
        "minimum_effective_sample_size": min_ess,
        "dense_min_effective_sample_size": dense_min_ess,
        "diagonal_min_effective_sample_size": diagonal_min_ess,
        "cross_chain_compatibility_method": (
            "not_applicable_single_chain"
            if not explicit_chains
            else "split_rhat_adaptation_adequacy_only"
        ),
        "cross_chain_compatibility_status": (
            "not_applicable"
            if not explicit_chains
            else "available" if split_rhat_finite else "undefined_fail_closed"
        ),
        "split_rhat_by_coordinate": (
            None
            if split_rhat is None
            else tuple(
                float(item) if np.isfinite(item) else None for item in split_rhat
            )
        ),
        "maximum_split_rhat": max_split_rhat,
        "raw_numerical_rank": raw_rank,
        "raw_condition_number": raw_condition,
        "shrinkage": weight,
        "dense_relative_frobenius_discrepancy": dense_discrepancy,
        "dense_checks": dense_checks,
    }
    if all(dense_checks.values()):
        median_diagonal = float(np.median(np.diag(dense_shrunk)))
        floor = max(1.0e-9, 1.0e-8 * median_diagonal)
        eigenvalues, eigenvectors = np.linalg.eigh(dense_shrunk)
        regularized = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
        return MetricAdequacyDecision(
            outcome="dense_update",
            covariance=0.5 * (regularized + regularized.T),
            estimator_family="dense_welford_identity_shrinkage",
            report={
                **common,
                "eigenvalue_floor": floor,
                "clipped_eigenvalue_count": int(np.sum(eigenvalues < floor)),
                "dense_information_gate_passed": True,
                "diagonal_fallback_used": False,
            },
        )

    empirical_diagonal = np.diag(empirical)
    diagonal_finite_positive = bool(
        np.all(np.isfinite(empirical_diagonal)) and np.all(empirical_diagonal > 0.0)
    )
    shrunk_diagonal = (1.0 - weight) * empirical_diagonal + weight
    diagonal_discrepancy = float(
        np.linalg.norm(shrunk_diagonal - empirical_diagonal)
        / max(np.linalg.norm(empirical_diagonal), np.finfo(float).eps)
    )
    diagonal_checks = {
        "state_count_sufficient": n >= diagonal_required,
        "effective_information_sufficient": min_ess >= diagonal_min_ess,
        "cross_chain_location_compatible": diagonal_chain_compatible,
        "raw_variances_finite_positive": diagonal_finite_positive,
        "regularization_discrepancy_acceptable": diagonal_discrepancy <= 0.75,
    }
    report = {
        **common,
        "diagonal_relative_euclidean_discrepancy": diagonal_discrepancy,
        "diagonal_checks": diagonal_checks,
        "dense_information_gate_passed": False,
    }
    if all(diagonal_checks.values()):
        median_diagonal = float(np.median(shrunk_diagonal))
        floor = max(1.0e-9, 1.0e-8 * median_diagonal)
        regularized_diagonal = np.maximum(shrunk_diagonal, floor)
        return MetricAdequacyDecision(
            outcome="diagonal_fallback",
            covariance=np.diag(regularized_diagonal),
            estimator_family="diagonal_welford_identity_shrinkage",
            report={
                **report,
                "eigenvalue_floor": floor,
                "clipped_eigenvalue_count": int(np.sum(shrunk_diagonal < floor)),
                "diagonal_fallback_used": True,
            },
        )
    return MetricAdequacyDecision(
        outcome="no_update_insufficient_metric_evidence",
        covariance=None,
        estimator_family=None,
        report={
            **report,
            "diagonal_fallback_used": False,
            "shrinkage_spd_not_treated_as_adequacy": True,
        },
    )


def _rejected_metric_candidate(
    decision: MetricAdequacyDecision,
    *,
    stage: str,
    error: Exception,
) -> MetricAdequacyDecision:
    """Convert a qualified but unusable proposal into a rollback decision.

    The completed HMC window and its adequacy evidence remain valid. Only the
    proposed coordinate boundary is rejected, so the incumbent transform and
    dual-averaging state may continue without exposing exception text.
    """

    if not decision.update_applied:
        raise ValueError("only an adequate metric proposal can be rejected")
    rejection_stage = str(stage)
    if rejection_stage not in {
        "transform_construction",
        "affine_target_parity",
        "reasonable_epsilon",
    }:
        raise ValueError("unsupported metric candidate rejection stage")
    return MetricAdequacyDecision(
        outcome="candidate_metric_rejected",
        covariance=None,
        estimator_family=None,
        report={
            **dict(decision.report),
            "candidate_metric_evidence_passed": True,
            "candidate_original_outcome": decision.outcome,
            "candidate_estimator_family": decision.estimator_family,
            "candidate_rejection_stage": rejection_stage,
            "candidate_rejection_error_type": type(error).__name__,
            "candidate_rejection_reason": (
                f"candidate_boundary_{rejection_stage}_failed"
            ),
            "incumbent_metric_retained": True,
        },
    )


def _summed_chain_ess(chain_states: np.ndarray) -> np.ndarray:
    """Conservative per-coordinate ESS with time kept within each chain."""

    draw_count, chain_count, dimension = chain_states.shape
    total = np.zeros(dimension, dtype=float)
    fft_size = 1 << int(np.ceil(np.log2(max(2, 2 * draw_count))))
    for chain_index in range(chain_count):
        centered = chain_states[:, chain_index, :] - np.mean(
            chain_states[:, chain_index, :], axis=0, keepdims=True
        )
        spectrum = np.fft.rfft(centered, n=fft_size, axis=0)
        autocovariance = np.fft.irfft(
            spectrum * np.conjugate(spectrum), n=fft_size, axis=0
        )[:draw_count]
        variance = autocovariance[0]
        valid = variance > np.finfo(float).eps
        rho = np.zeros_like(autocovariance, dtype=float)
        rho[:, valid] = autocovariance[:, valid] / variance[valid]
        pair_sum = np.zeros(dimension, dtype=float)
        active = valid.copy()
        previous = np.full(dimension, np.inf, dtype=float)
        for lag in range(1, draw_count - 1, 2):
            pair = rho[lag] + rho[lag + 1]
            pair = np.minimum(pair, previous)
            positive = active & np.isfinite(pair) & (pair > 0.0)
            pair_sum[positive] += pair[positive]
            active &= positive
            previous[positive] = pair[positive]
            if not np.any(active):
                break
        tau = np.maximum(1.0, 1.0 + 2.0 * pair_sum)
        chain_ess = np.zeros(dimension, dtype=float)
        chain_ess[valid] = np.clip(draw_count / tau[valid], 1.0, draw_count)
        total += chain_ess
    return total


def _split_rhat_by_coordinate(chain_states: np.ndarray) -> np.ndarray | None:
    """Return split R-hat only when multiple explicit chains are available."""

    draw_count, chain_count, _dimension = chain_states.shape
    if chain_count < 2 or draw_count < 4:
        return None
    half = draw_count // 2
    split = np.concatenate(
        (chain_states[:half], chain_states[draw_count - half :]),
        axis=1,
    )
    split_means = np.mean(split, axis=0)
    within = np.mean(np.var(split, axis=0, ddof=1), axis=0)
    between = half * np.var(split_means, axis=0, ddof=1)
    variance = ((half - 1.0) / half) * within + between / half
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(variance / within)


def normalize_operational_warmup_config(
    config: WindowedMassAdaptationConfig,
) -> WindowedMassAdaptationConfig:
    """Reserve four final-coordinate draws without changing the total budget."""

    if not isinstance(config, WindowedMassAdaptationConfig):
        raise TypeError("config must be WindowedMassAdaptationConfig")
    if config.initial_buffer <= 0:
        raise ValueError("operational warmup requires a non-empty initial fast window")
    if config.final_buffer >= 4:
        return config
    final_buffer = 4
    slow_steps = config.warmup_steps - config.initial_buffer - final_buffer
    if slow_steps < config.min_window_samples:
        raise ValueError(
            "operational warmup cannot reserve four final-coordinate transitions"
        )
    first_window_size = min(config.first_window_size, slow_steps)
    if first_window_size < config.min_window_samples:
        raise ValueError(
            "operational warmup cannot preserve the slow-window sample minimum"
        )
    return replace(
        config,
        final_buffer=final_buffer,
        first_window_size=first_window_size,
    )


class _AffineWarmupAdapter:
    """Transform a canonical value/score adapter into one active coordinate."""

    def __init__(
        self,
        *,
        base_adapter: Any,
        transform: AffineCoordinateTransform,
        target_scope: str,
    ) -> None:
        if not hasattr(base_adapter, "log_prob_and_grad"):
            raise TypeError("base_adapter must expose log_prob_and_grad")
        self.base_adapter = base_adapter
        self.transform = transform
        self.parameter_dim = transform.dimension
        self.target_scope = str(target_scope)
        if not self.target_scope:
            raise ValueError("target_scope must be non-empty")
        self.supports_retained_draw_batch = bool(
            getattr(base_adapter, "supports_retained_draw_batch", False)
        )
        self.supports_retained_flat_batch = bool(
            getattr(base_adapter, "supports_retained_flat_batch", False)
        )
        if self.supports_retained_draw_batch and self.supports_retained_flat_batch:
            raise ValueError(
                "base adapter cannot declare two retained batching contracts"
            )
        self.runtime_backend = "bayesfilter.inference.hmc_warmup._AffineWarmupAdapter"

    def adapter_signature(self) -> str:
        return _stable_hash(
            "bayesfilter.affine_warmup_adapter.v2",
            {
                "base_adapter_signature": _base_adapter_signature(self.base_adapter),
                "transform_signature": self.transform.signature,
                "target_scope": self.target_scope,
            },
        )

    def value_score_capability(self) -> ValueScoreCapability:
        base = value_score_capability(self.base_adapter)
        return ValueScoreCapability(
            value_score_authority=base.value_score_authority,
            xla_hmc_ready=False,
            full_chain_xla_diagnostic_ready=False,
            runtime_backend=self.runtime_backend,
            evidence_path=base.evidence_path,
            target_scope=self.target_scope,
            nonclaims=OPERATIONAL_WARMUP_NONCLAIMS,
        )

    def initial_position(self) -> Any:
        import tensorflow as tf

        return tf.zeros((self.parameter_dim,), dtype=tf.float64)

    def log_prob_and_grad(self, latent: Any) -> tuple[Any, Any]:
        import tensorflow as tf

        z = tf.convert_to_tensor(latent, dtype=tf.float64)
        theta = self.transform.latent_to_theta(z)
        value, theta_score = self.base_adapter.log_prob_and_grad(theta)
        return (
            tf.convert_to_tensor(value, dtype=z.dtype),
            self.transform.theta_score_to_latent_score(theta_score),
        )

    def target_status_telemetry(self, latent: Any) -> Mapping[str, Any]:
        telemetry = getattr(self.base_adapter, "target_status_telemetry", None)
        if not callable(telemetry):
            raise TypeError("base_adapter must expose target_status_telemetry")
        theta = self.transform.latent_to_theta(latent)
        payload = telemetry(theta)
        if not isinstance(payload, Mapping):
            raise TypeError("target_status_telemetry must return a mapping")
        return payload


def _base_adapter_signature(adapter: Any) -> str:
    signature = getattr(adapter, "adapter_signature", None)
    if callable(signature):
        value = str(signature())
    else:
        value = str(signature or type(adapter).__qualname__)
    if not value:
        raise ValueError("base adapter signature must be non-empty")
    return value


@dataclass(frozen=True)
class ReasonableEpsilonAttempt:
    step_size: float
    mean_acceptance_probability: float | None
    finite: bool
    seed: tuple[int, int]
    engineering_health_failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        step = float(self.step_size)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("reasonable-epsilon step_size must be positive and finite")
        if not isinstance(self.finite, (bool, np.bool_)):
            raise ValueError("reasonable-epsilon finite flag must be boolean")
        finite = bool(self.finite)
        mean = self.mean_acceptance_probability
        if finite:
            if mean is None:
                raise ValueError("finite reasonable-epsilon attempt requires acceptance")
            mean = float(mean)
            if not np.isfinite(mean) or not 0.0 <= mean <= 1.0:
                raise ValueError("reasonable-epsilon acceptance must lie inside [0, 1]")
        elif mean is not None:
            raise ValueError("nonfinite reasonable-epsilon attempt must normalize to None")
        health_failures = tuple(
            dict.fromkeys(str(item) for item in self.engineering_health_failures)
        )
        if any(not item for item in health_failures):
            raise ValueError(
                "reasonable-epsilon health failures must contain nonempty codes"
            )
        if not set(health_failures).issubset({"target_status_telemetry_failure"}):
            raise ValueError("unsupported reasonable-epsilon health failure")
        object.__setattr__(self, "step_size", step)
        object.__setattr__(self, "mean_acceptance_probability", mean)
        object.__setattr__(self, "finite", finite)
        object.__setattr__(self, "seed", _strict_seed(self.seed, name="attempt seed"))
        object.__setattr__(self, "engineering_health_failures", health_failures)

    @property
    def usable(self) -> bool:
        return self.finite and not self.engineering_health_failures

    def payload(self) -> Mapping[str, Any]:
        return {
            "step_size": self.step_size,
            "mean_acceptance_probability": self.mean_acceptance_probability,
            "finite": self.finite,
            "engineering_health_failures": self.engineering_health_failures,
            "usable": self.usable,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ReasonableEpsilonResult:
    status: str
    selected_step_size: float | None
    attempts: tuple[ReasonableEpsilonAttempt, ...]

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in {"passed", "inconclusive_bracket"}:
            raise ValueError("unsupported reasonable-epsilon status")
        attempts = tuple(self.attempts)
        if not attempts or not all(
            isinstance(item, ReasonableEpsilonAttempt) for item in attempts
        ):
            raise ValueError("reasonable-epsilon result requires typed attempts")
        selected = self.selected_step_size
        if status == "passed":
            if selected is None:
                raise ValueError("passed reasonable-epsilon result requires a step")
            selected = float(selected)
            if (
                not np.isfinite(selected)
                or selected <= 0.0
                or not attempts[-1].usable
                or attempts[-1].mean_acceptance_probability is None
                or not np.isclose(selected, attempts[-1].step_size, rtol=0.0, atol=0.0)
            ):
                raise ValueError("selected reasonable epsilon lacks final finite evidence")
        elif selected is not None:
            raise ValueError("inconclusive reasonable-epsilon result cannot select a step")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "selected_step_size", selected)
        object.__setattr__(self, "attempts", attempts)

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def payload(self) -> Mapping[str, Any]:
        return {
            "status": self.status,
            "selected_step_size": self.selected_step_size,
            "attempts": tuple(attempt.payload() for attempt in self.attempts),
            "diagnostic_role": "reasonable_epsilon_engineering_bracket",
            "nonclaims": OPERATIONAL_WARMUP_NONCLAIMS,
        }


def find_reasonable_epsilon(
    *,
    adapter: Any,
    current_state: Any,
    initial_step_size: float,
    seed: tuple[int, int],
    max_attempts: int = 20,
    lower_acceptance: float = 0.25,
    upper_acceptance: float = 0.75,
    target_status_trace_policy: str = "none",
) -> ReasonableEpsilonResult:
    """Bracket a finite one-step HMC epsilon using mean accept probability."""

    import tensorflow as tf
    import tensorflow_probability as tfp

    step = float(initial_step_size)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("initial_step_size must be positive and finite")
    attempt_limit = _strict_integer(max_attempts, name="max_attempts", minimum=1)
    lower = float(lower_acceptance)
    upper = float(upper_acceptance)
    if (
        not np.all(np.isfinite((lower, upper)))
        or not 0.0 < lower < upper < 1.0
    ):
        raise ValueError("reasonable-epsilon acceptance bracket must lie inside (0, 1)")
    normalized_seed = _strict_seed(seed, name="seed")
    target_status_policy = _target_status_policy(target_status_trace_policy)
    state = tf.convert_to_tensor(current_state, dtype=tf.float64)
    if (
        state.shape.rank is None
        or state.shape.rank < 1
        or any(dim == 0 for dim in state.shape)
        or not bool(tf.reduce_all(tf.math.is_finite(state)).numpy())
    ):
        raise ValueError("reasonable-epsilon current_state must be non-empty and finite")
    target = reviewed_value_score_target_fn(adapter, dtype=state.dtype)
    initial_value, initial_score = adapter.log_prob_and_grad(state)
    if not _all_finite_tensors((initial_value, initial_score)):
        raise ValueError(
            "reasonable-epsilon current_state target value and score must be finite"
        )
    target_status_shape = tuple(int(item) for item in tf.shape(state)[:-1].numpy())
    if target_status_policy == "per_chain_step" and _target_status_failed(
        adapter,
        state,
        expected_shape=target_status_shape,
    ):
        raise ValueError(
            "reasonable-epsilon current_state target-status telemetry is nonvalid"
        )
    attempts: list[ReasonableEpsilonAttempt] = []
    high_acceptance_step: float | None = None
    low_acceptance_step: float | None = None
    proposal_seed = _seed(normalized_seed, 0)
    for index in range(attempt_limit):
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target,
            step_size=tf.constant(step, dtype=state.dtype),
            num_leapfrog_steps=1,
        )
        results = kernel.bootstrap_results(state)
        if not _kernel_result_value_score_finite(results.accepted_results):
            raise ValueError("reasonable-epsilon bootstrap target evidence is nonfinite")
        try:
            next_state, next_results = kernel.one_step(
                state,
                results,
                seed=tf.constant(proposal_seed, dtype=tf.int32),
            )
            log_accept = tf.convert_to_tensor(next_results.log_accept_ratio, tf.float64)
            retained_finite = bool(
                _all_finite_tensors((next_state,))
                and _kernel_result_value_score_finite(next_results.accepted_results)
            )
            finite = bool(
                tf.reduce_all(tf.math.is_finite(log_accept)).numpy()
                and _all_finite_tensors((next_results.proposed_state,))
                and _kernel_result_value_score_finite(next_results.proposed_results)
            )
            mean_accept = (
                float(tf.reduce_mean(tf.exp(tf.minimum(log_accept, 0.0))).numpy())
                if finite
                else None
            )
        except Exception as exc:  # noqa: BLE001 - shared runner failure is fail-closed.
            raise RuntimeError(
                "reasonable-epsilon HMC proposal execution failed"
            ) from exc
        if not retained_finite:
            raise ValueError(
                "reasonable-epsilon accepted or retained state is nonfinite"
            )
        health_failures: tuple[str, ...] = ()
        if finite and target_status_policy == "per_chain_step":
            proposal_failed = _target_status_failed(
                adapter,
                next_results.proposed_state,
                expected_shape=target_status_shape,
            )
            retained_failed = _target_status_failed(
                adapter,
                next_state,
                expected_shape=target_status_shape,
            )
            if retained_failed:
                raise ValueError(
                    "reasonable-epsilon accepted or retained target status is nonvalid"
                )
            if proposal_failed:
                health_failures = ("target_status_telemetry_failure",)
        attempt = ReasonableEpsilonAttempt(
            step_size=step,
            mean_acceptance_probability=mean_accept,
            finite=finite,
            seed=proposal_seed,
            engineering_health_failures=health_failures,
        )
        attempts.append(attempt)
        if attempt.usable and mean_accept is not None and lower <= mean_accept <= upper:
            return ReasonableEpsilonResult("passed", step, tuple(attempts))
        if attempt.usable and mean_accept is not None and mean_accept > upper:
            high_acceptance_step = step
            next_step = (
                step * 2.0
                if low_acceptance_step is None
                else float(np.sqrt(step * low_acceptance_step))
            )
        else:
            low_acceptance_step = step
            next_step = (
                step * 0.5
                if high_acceptance_step is None
                else float(np.sqrt(step * high_acceptance_step))
            )
        if (
            not np.isfinite(next_step)
            or next_step <= 0.0
            or np.isclose(next_step, step, rtol=1.0e-12, atol=0.0)
        ):
            break
        step = next_step
    return ReasonableEpsilonResult("inconclusive_bracket", None, tuple(attempts))


def _all_finite_tensors(values: Sequence[Any]) -> bool:
    import tensorflow as tf

    for value in values:
        if isinstance(value, (tuple, list)):
            if not _all_finite_tensors(value):
                return False
            continue
        tensor = tf.convert_to_tensor(value)
        if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
            return False
    return True


def _kernel_result_value_score_finite(kernel_result: Any) -> bool:
    return _all_finite_tensors(
        (
            kernel_result.target_log_prob,
            kernel_result.grads_target_log_prob,
        )
    )


def _target_status_failed(
    adapter: Any,
    state: Any,
    *,
    expected_shape: tuple[int, ...],
) -> bool:
    telemetry = getattr(adapter, "target_status_telemetry", None)
    if not callable(telemetry):
        raise TypeError("requested target-status telemetry is unavailable")
    payload = telemetry(state)
    if not isinstance(payload, Mapping):
        raise TypeError("target_status_telemetry must return a mapping")
    return target_status_telemetry_has_failure(
        {
            key: np.asarray(value.numpy() if hasattr(value, "numpy") else value)
            for key, value in payload.items()
        },
        expected_shape=expected_shape,
    )


@dataclass(frozen=True)
class OperationalWarmupWindowResult:
    window: WindowedWarmupWindow
    transition_count_before_window: int
    transition_count_after_window: int
    coordinate_signature_used: str
    metric_signature_used: str
    epsilon_start: float
    epsilon_end: float
    mean_acceptance_probability: float
    binary_acceptance_rate: float
    native_divergence_status: str
    native_divergence_count: int | None
    target_status_trace_policy: str
    target_status_failure_count: int | None
    max_abs_log_accept_energy_proxy: float
    final_latent_state: Any
    final_canonical_theta: Any
    adaptation_latent_states: Any
    adaptation_canonical_states: Any
    log_accept_ratio: Any
    is_accepted: Any
    target_log_prob: Any
    step_size_trace: Any
    metric_decision: MetricAdequacyDecision | None
    next_coordinate_signature: str | None
    next_metric_signature: str | None
    state_map_residual: float | None
    target_value_map_residual: float | None
    target_score_map_residual: float | None
    next_reasonable_epsilon: ReasonableEpsilonResult | None
    dual_averaging_generation: int
    runner_generation: int
    runner_trace_count: int | None
    runtime_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.window, WindowedWarmupWindow):
            raise TypeError("window must be a WindowedWarmupWindow")
        before = _strict_integer(
            self.transition_count_before_window,
            name="transition_count_before_window",
            minimum=0,
        )
        after = _strict_integer(
            self.transition_count_after_window,
            name="transition_count_after_window",
            minimum=1,
        )
        if before != self.window.start or after != self.window.end:
            raise ValueError("warmup transition counts must match the window")
        coordinate = str(self.coordinate_signature_used)
        metric = str(self.metric_signature_used)
        if not coordinate or not metric:
            raise ValueError("warmup coordinate and metric signatures must be non-empty")
        epsilon_start = float(self.epsilon_start)
        epsilon_end = float(self.epsilon_end)
        if (
            not np.all(np.isfinite((epsilon_start, epsilon_end)))
            or epsilon_start <= 0.0
            or epsilon_end <= 0.0
        ):
            raise ValueError("warmup epsilon endpoints must be positive and finite")

        final_latent = np.asarray(self.final_latent_state, dtype=float).copy()
        final_theta = np.asarray(self.final_canonical_theta, dtype=float).copy()
        latent_draws = np.asarray(self.adaptation_latent_states, dtype=float).copy()
        theta_draws = np.asarray(self.adaptation_canonical_states, dtype=float).copy()
        log_accept = np.asarray(self.log_accept_ratio, dtype=float).copy()
        accepted_input = np.asarray(self.is_accepted)
        if not np.issubdtype(accepted_input.dtype, np.bool_):
            raise ValueError("is_accepted must be boolean")
        accepted = accepted_input.astype(bool, copy=True)
        target_value = np.asarray(self.target_log_prob, dtype=float).copy()
        step_trace = np.asarray(self.step_size_trace, dtype=float).copy()
        dimension = int(final_latent.size)
        expected_draw_shape = (self.window.length, dimension)
        expected_trace_shape = (self.window.length,)
        if (
            final_latent.ndim != 1
            or dimension <= 0
            or final_theta.shape != (dimension,)
            or latent_draws.shape != expected_draw_shape
            or theta_draws.shape != expected_draw_shape
            or log_accept.shape != expected_trace_shape
            or accepted.shape != expected_trace_shape
            or target_value.shape != expected_trace_shape
            or step_trace.shape != expected_trace_shape
        ):
            raise ValueError("operational warmup window arrays are misaligned")
        arrays = (
            final_latent,
            final_theta,
            latent_draws,
            theta_draws,
            log_accept,
            target_value,
            step_trace,
        )
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("operational warmup window arrays must be finite")
        if np.any(step_trace <= 0.0):
            raise ValueError("step_size_trace must be positive")
        if not np.allclose(final_latent, latent_draws[-1], rtol=0.0, atol=0.0):
            raise ValueError("final latent state must equal the last warmup draw")
        if not np.allclose(final_theta, theta_draws[-1], rtol=0.0, atol=0.0):
            raise ValueError("final canonical state must equal the last warmup draw")

        mean_acceptance = float(self.mean_acceptance_probability)
        binary_acceptance = float(self.binary_acceptance_rate)
        proxy = float(self.max_abs_log_accept_energy_proxy)
        expected_mean = float(np.mean(np.exp(np.minimum(log_accept, 0.0))))
        expected_binary = float(np.mean(accepted))
        expected_proxy = float(np.max(np.abs(log_accept)))
        if (
            not np.all(np.isfinite((mean_acceptance, binary_acceptance, proxy)))
            or not 0.0 <= mean_acceptance <= 1.0
            or not 0.0 <= binary_acceptance <= 1.0
            or proxy < 0.0
            or not np.isclose(mean_acceptance, expected_mean, rtol=1.0e-12, atol=1.0e-12)
            or not np.isclose(binary_acceptance, expected_binary, rtol=1.0e-12, atol=1.0e-12)
            or not np.isclose(proxy, expected_proxy, rtol=1.0e-12, atol=1.0e-12)
        ):
            raise ValueError("operational warmup acceptance summary is inconsistent")

        divergence_status = str(self.native_divergence_status)
        divergence_count = self.native_divergence_count
        if divergence_status == "available":
            divergence_count = _strict_integer(
                divergence_count,
                name="native_divergence_count",
                minimum=0,
            )
        elif divergence_status in {"not_exposed_by_kernel", "not_collected"}:
            if divergence_count is not None:
                raise ValueError("unavailable divergence status cannot carry a count")
        else:
            raise ValueError("invalid native divergence status")
        target_status_policy = _target_status_policy(self.target_status_trace_policy)
        target_status_failure_count = self.target_status_failure_count
        if target_status_policy == "none":
            if target_status_failure_count is not None:
                raise ValueError("disabled target-status policy cannot carry a count")
        else:
            target_status_failure_count = _strict_integer(
                target_status_failure_count,
                name="target_status_failure_count",
                minimum=0,
            )
            if target_status_failure_count:
                raise ValueError("operational warmup target-status telemetry vetoed a window")

        decision = self.metric_decision
        if decision is not None and not isinstance(decision, MetricAdequacyDecision):
            raise TypeError("metric_decision must be a MetricAdequacyDecision")
        if decision is not None and not self.window.update_mass:
            raise ValueError("a non-mass warmup window cannot carry a metric decision")
        update_applied = decision is not None and decision.update_applied
        next_coordinate = (
            None
            if self.next_coordinate_signature is None
            else str(self.next_coordinate_signature)
        )
        next_metric = (
            None if self.next_metric_signature is None else str(self.next_metric_signature)
        )
        next_reasonable = self.next_reasonable_epsilon
        if update_applied:
            if (
                not next_coordinate
                or not next_metric
                or not isinstance(next_reasonable, ReasonableEpsilonResult)
                or not next_reasonable.passed
                or next_reasonable.selected_step_size is None
                or not np.isclose(
                    epsilon_end,
                    next_reasonable.selected_step_size,
                    rtol=1.0e-12,
                    atol=0.0,
                )
            ):
                raise ValueError("metric update lacks a complete epsilon handoff")
        elif any(
            item is not None
            for item in (next_coordinate, next_metric, next_reasonable)
        ):
            raise ValueError("no-update warmup window carries a false handoff")
        elif not np.isclose(epsilon_end, step_trace[-1], rtol=1.0e-12, atol=0.0):
            raise ValueError("warmup epsilon endpoint does not match its step trace")

        residual_names = (
            "state_map_residual",
            "target_value_map_residual",
            "target_score_map_residual",
        )
        for name in residual_names:
            value = getattr(self, name)
            if value is None:
                if name == "state_map_residual":
                    raise ValueError("state_map_residual is required")
                continue
            normalized = float(value)
            if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0e-10:
                raise ValueError(f"{name} violates the coordinate-map tolerance")
            object.__setattr__(self, name, normalized)
        generation = _strict_integer(
            self.dual_averaging_generation,
            name="dual_averaging_generation",
            minimum=0,
        )
        runner_generation = _strict_integer(
            self.runner_generation,
            name="runner_generation",
            minimum=0,
        )
        trace_count = self.runner_trace_count
        if trace_count is not None:
            trace_count = _strict_integer(
                trace_count,
                name="runner_trace_count",
                minimum=1,
            )
        runtime = float(self.runtime_s)
        if not np.isfinite(runtime) or runtime < 0.0:
            raise ValueError("runtime_s must be finite and nonnegative")

        for array in (*arrays, accepted):
            array.setflags(write=False)
        object.__setattr__(self, "transition_count_before_window", before)
        object.__setattr__(self, "transition_count_after_window", after)
        object.__setattr__(self, "coordinate_signature_used", coordinate)
        object.__setattr__(self, "metric_signature_used", metric)
        object.__setattr__(self, "epsilon_start", epsilon_start)
        object.__setattr__(self, "epsilon_end", epsilon_end)
        object.__setattr__(self, "mean_acceptance_probability", mean_acceptance)
        object.__setattr__(self, "binary_acceptance_rate", binary_acceptance)
        object.__setattr__(self, "native_divergence_status", divergence_status)
        object.__setattr__(self, "native_divergence_count", divergence_count)
        object.__setattr__(self, "target_status_trace_policy", target_status_policy)
        object.__setattr__(
            self,
            "target_status_failure_count",
            target_status_failure_count,
        )
        object.__setattr__(self, "max_abs_log_accept_energy_proxy", proxy)
        object.__setattr__(self, "final_latent_state", final_latent)
        object.__setattr__(self, "final_canonical_theta", final_theta)
        object.__setattr__(self, "adaptation_latent_states", latent_draws)
        object.__setattr__(self, "adaptation_canonical_states", theta_draws)
        object.__setattr__(self, "log_accept_ratio", log_accept)
        object.__setattr__(self, "is_accepted", accepted)
        object.__setattr__(self, "target_log_prob", target_value)
        object.__setattr__(self, "step_size_trace", step_trace)
        object.__setattr__(self, "next_coordinate_signature", next_coordinate)
        object.__setattr__(self, "next_metric_signature", next_metric)
        object.__setattr__(self, "dual_averaging_generation", generation)
        object.__setattr__(self, "runner_generation", runner_generation)
        object.__setattr__(self, "runner_trace_count", trace_count)
        object.__setattr__(self, "runtime_s", runtime)

    def public_payload(self) -> Mapping[str, Any]:
        return {
            "window": self.window.payload(),
            "transition_count_before_window": self.transition_count_before_window,
            "transition_count_after_window": self.transition_count_after_window,
            "coordinate_signature_used": self.coordinate_signature_used,
            "metric_signature_used": self.metric_signature_used,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "mean_acceptance_probability": self.mean_acceptance_probability,
            "binary_acceptance_rate": self.binary_acceptance_rate,
            "native_divergence_status": self.native_divergence_status,
            "native_divergence_count": self.native_divergence_count,
            "target_status_trace_policy": self.target_status_trace_policy,
            "target_status_failure_count": self.target_status_failure_count,
            "max_abs_log_accept_energy_proxy": self.max_abs_log_accept_energy_proxy,
            "metric_decision": None
            if self.metric_decision is None
            else self.metric_decision.payload(),
            "next_coordinate_signature": self.next_coordinate_signature,
            "next_metric_signature": self.next_metric_signature,
            "state_map_residual": self.state_map_residual,
            "target_value_map_residual": self.target_value_map_residual,
            "target_score_map_residual": self.target_score_map_residual,
            "next_reasonable_epsilon": None
            if self.next_reasonable_epsilon is None
            else self.next_reasonable_epsilon.payload(),
            "dual_averaging_generation": self.dual_averaging_generation,
            "runner_generation": self.runner_generation,
            "runner_trace_count": self.runner_trace_count,
            "runtime_s": self.runtime_s,
            "raw_states_exposed": False,
        }


@dataclass(frozen=True)
class OperationalWindowedWarmupResult:
    config: WindowedMassAdaptationConfig
    initial_coordinate_signature: str
    final_kernel_state: KernelState
    reasonable_epsilon: ReasonableEpsilonResult
    windows: tuple[OperationalWarmupWindowResult, ...]
    private_start_bank_theta: Any
    seed_root: tuple[int, int]
    target_scope: str
    target_status_trace_policy: str
    elapsed_s: float
    status: str = "passed"
    algorithm_id: str = OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
    route_contract_version: str = HMC_ROUTE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.config, WindowedMassAdaptationConfig):
            raise TypeError("config must be WindowedMassAdaptationConfig")
        if not isinstance(self.final_kernel_state, KernelState):
            raise TypeError("final_kernel_state must be KernelState")
        if not isinstance(self.reasonable_epsilon, ReasonableEpsilonResult):
            raise TypeError("reasonable_epsilon must be ReasonableEpsilonResult")
        if not self.reasonable_epsilon.passed:
            raise ValueError("operational warmup requires a passed epsilon bracket")
        if str(self.status) != "passed":
            raise ValueError("operational warmup supports only passed results")
        windows = tuple(self.windows)
        if not windows:
            raise ValueError("operational warmup requires window results")
        expected_schedule = build_windowed_warmup_schedule(self.config)
        if tuple(item.window for item in windows) != expected_schedule:
            raise ValueError("operational warmup windows do not match the schedule")
        initial_signature = str(self.initial_coordinate_signature)
        if not initial_signature or windows[0].coordinate_signature_used != initial_signature:
            raise ValueError("operational warmup initial coordinate lineage is invalid")
        expected_coordinate = initial_signature
        expected_metric = windows[0].metric_signature_used
        expected_epsilon = self.reasonable_epsilon.selected_step_size
        applied_updates = 0
        for index, window in enumerate(windows):
            if (
                window.coordinate_signature_used != expected_coordinate
                or window.metric_signature_used != expected_metric
                or window.dual_averaging_generation != applied_updates
                or window.runner_generation != applied_updates
                or expected_epsilon is None
                or not np.isclose(
                    window.epsilon_start,
                    expected_epsilon,
                    rtol=1.0e-12,
                    atol=0.0,
                )
            ):
                raise ValueError("operational warmup window lineage is discontinuous")
            update_applied = (
                window.metric_decision is not None
                and window.metric_decision.update_applied
            )
            if update_applied:
                if index + 1 >= len(windows):
                    raise ValueError("terminal metric update lacks a later transition")
                expected_coordinate = str(window.next_coordinate_signature)
                expected_metric = str(window.next_metric_signature)
                applied_updates += 1
            expected_epsilon = window.epsilon_end
        bank = np.asarray(self.private_start_bank_theta, dtype=float).copy()
        if bank.ndim != 2 or bank.shape[0] != 4:
            raise ValueError("private start bank must contain four rank-2 rows")
        if bank.shape[1] != self.final_kernel_state.transform.dimension:
            raise ValueError("private start bank dimension mismatch")
        if not np.all(np.isfinite(bank)):
            raise ValueError("private start bank must be finite")
        final_state = self.final_kernel_state
        if (
            final_state.transform.signature != expected_coordinate
            or final_state.momentum_metric.signature != expected_metric
            or final_state.adaptation_generation != applied_updates
            or final_state.epsilon is None
            or not np.isclose(
                final_state.epsilon,
                float(expected_epsilon),
                rtol=1.0e-12,
                atol=0.0,
            )
            or not np.allclose(
                final_state.canonical_theta,
                windows[-1].final_canonical_theta,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            or not np.allclose(
                final_state.active_latent,
                windows[-1].final_latent_state,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ):
            raise ValueError("operational warmup final kernel lineage is invalid")
        scale = max(float(np.linalg.norm(np.std(bank, axis=0))), 1.0)
        tolerance = 1.0e-10 * scale
        pairwise = np.linalg.norm(bank[:, None, :] - bank[None, :, :], axis=-1)
        if (
            not np.allclose(
                bank[-1],
                final_state.canonical_theta,
                rtol=1.0e-10,
                atol=1.0e-10,
            )
            or np.any(pairwise[np.triu_indices(4, k=1)] <= tolerance)
        ):
            raise ValueError("private start bank must be dispersed and include the endpoint")
        seed_root = _strict_seed(self.seed_root, name="seed_root")
        target_scope = str(self.target_scope)
        target_status_policy = _target_status_policy(self.target_status_trace_policy)
        elapsed = float(self.elapsed_s)
        algorithm_id = str(self.algorithm_id)
        route_contract_version = str(self.route_contract_version)
        if not target_scope:
            raise ValueError("target_scope must be non-empty")
        if (
            algorithm_id != OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
            or route_contract_version != HMC_ROUTE_CONTRACT_VERSION
        ):
            raise ValueError("operational warmup route identity is invalid")
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("elapsed_s must be finite and nonnegative")
        bank.setflags(write=False)
        object.__setattr__(self, "initial_coordinate_signature", initial_signature)
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "private_start_bank_theta", bank)
        object.__setattr__(self, "seed_root", seed_root)
        object.__setattr__(self, "target_scope", target_scope)
        object.__setattr__(self, "target_status_trace_policy", target_status_policy)
        object.__setattr__(self, "elapsed_s", elapsed)
        object.__setattr__(self, "status", "passed")
        object.__setattr__(self, "algorithm_id", algorithm_id)
        object.__setattr__(self, "route_contract_version", route_contract_version)
        if not self.every_update_used_by_later_transition:
            raise ValueError("metric update was not used by the next real transition")

    @property
    def private_start_bank_signature(self) -> str:
        digest = hashlib.sha256(np.ascontiguousarray(self.private_start_bank_theta).tobytes())
        digest.update(self.final_kernel_state.transform.signature.encode("ascii"))
        return digest.hexdigest()

    @property
    def operational_metric_update_count(self) -> int:
        return sum(
            1
            for window in self.windows
            if window.metric_decision is not None and window.metric_decision.update_applied
        )

    @property
    def every_update_used_by_later_transition(self) -> bool:
        for index, window in enumerate(self.windows):
            update_applied = (
                window.metric_decision is not None
                and window.metric_decision.update_applied
            )
            if not update_applied:
                continue
            if index + 1 >= len(self.windows):
                return False
            next_window = self.windows[index + 1]
            if (
                next_window.coordinate_signature_used
                != window.next_coordinate_signature
                or next_window.metric_signature_used != window.next_metric_signature
            ):
                return False
        return True

    def public_payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_operational_windowed_warmup.v2",
            "status": self.status,
            "algorithm_id": self.algorithm_id,
            "route_contract_version": self.route_contract_version,
            "config": self.config.payload(),
            "initial_coordinate_signature": self.initial_coordinate_signature,
            "final_coordinate_signature": self.final_kernel_state.transform.signature,
            "final_metric_signature": self.final_kernel_state.momentum_metric.signature,
            "final_epsilon": self.final_kernel_state.epsilon,
            "trajectory_policy_signature": self.final_kernel_state.trajectory_policy.signature,
            "reasonable_epsilon": self.reasonable_epsilon.payload(),
            "windows": tuple(window.public_payload() for window in self.windows),
            "operational_metric_update_count": self.operational_metric_update_count,
            "every_update_used_by_later_transition": self.every_update_used_by_later_transition,
            "private_start_bank": {
                "schema": "bayesfilter.hmc_private_start_bank.v2",
                "signature": self.private_start_bank_signature,
                "count": 4,
                "raw_values_exposed": False,
                "paths_exposed": False,
            },
            "seed_root": self.seed_root,
            "target_scope": self.target_scope,
            "target_status_trace_policy": self.target_status_trace_policy,
            "elapsed_s": self.elapsed_s,
            "reports_posterior_convergence": False,
            "nonclaims": OPERATIONAL_WARMUP_NONCLAIMS,
        }


@dataclass(frozen=True)
class OperationalWindowedWarmupCloseout:
    """Public-only snapshot taken before an unstarted warmup window."""

    algorithm_id: str
    route_contract_version: str
    boundary: str
    completed_windows: tuple[Mapping[str, Any], ...]
    planned_window_count: int
    completed_transition_count: int
    planned_transition_count: int
    completed_segment_count: int
    planned_segment_count: int
    boundary_payload: Mapping[str, Any]
    elapsed_s: float
    status: str = "partial_timeout_closeout"

    def __post_init__(self) -> None:
        algorithm_id = str(self.algorithm_id)
        version = str(self.route_contract_version)
        boundary = str(self.boundary)
        completed = tuple(dict(item) for item in self.completed_windows)
        planned_count = int(self.planned_window_count)
        transition_count = int(self.completed_transition_count)
        planned_transition_count = int(self.planned_transition_count)
        completed_segment_count = int(self.completed_segment_count)
        planned_segment_count = int(self.planned_segment_count)
        elapsed = float(self.elapsed_s)
        if (
            algorithm_id != OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
            or version != HMC_ROUTE_CONTRACT_VERSION
            or not boundary
        ):
            raise ValueError("operational closeout route identity is invalid")
        if planned_count <= 0 or len(completed) >= planned_count:
            raise ValueError("operational closeout must be partial")
        if (
            transition_count < 0
            or planned_transition_count <= 0
            or transition_count > planned_transition_count
            or completed_segment_count < 0
            or planned_segment_count <= 0
            or completed_segment_count >= planned_segment_count
            or not np.isfinite(elapsed)
            or elapsed < 0.0
        ):
            raise ValueError("operational closeout counters must be nonnegative")
        if any(item.get("raw_states_exposed") is not False for item in completed):
            raise ValueError("operational closeout window ledgers must be public-only")
        object.__setattr__(self, "algorithm_id", algorithm_id)
        object.__setattr__(self, "route_contract_version", version)
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "completed_windows", completed)
        object.__setattr__(self, "planned_window_count", planned_count)
        object.__setattr__(self, "completed_transition_count", transition_count)
        object.__setattr__(self, "planned_transition_count", planned_transition_count)
        object.__setattr__(self, "completed_segment_count", completed_segment_count)
        object.__setattr__(self, "planned_segment_count", planned_segment_count)
        object.__setattr__(self, "boundary_payload", dict(self.boundary_payload))
        object.__setattr__(self, "elapsed_s", elapsed)
        object.__setattr__(self, "status", "partial_timeout_closeout")

    def public_payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_operational_windowed_warmup_closeout.v1",
            "status": self.status,
            "algorithm_id": self.algorithm_id,
            "route_contract_version": self.route_contract_version,
            "boundary": self.boundary,
            "completed_windows": self.completed_windows,
            "completed_window_count": len(self.completed_windows),
            "planned_window_count": self.planned_window_count,
            "completed_transition_count": self.completed_transition_count,
            "planned_transition_count": self.planned_transition_count,
            "remaining_transition_count": (
                self.planned_transition_count - self.completed_transition_count
            ),
            "completed_segment_count": self.completed_segment_count,
            "planned_segment_count": self.planned_segment_count,
            "observed_transitions_per_second": (
                None
                if self.completed_transition_count == 0 or self.elapsed_s <= 0.0
                else self.completed_transition_count / self.elapsed_s
            ),
            "stop_source": self.boundary_payload.get("stop_source"),
            "stop_reason": self.boundary_payload.get("stop_reason"),
            "supervision_counter_baseline": self.boundary_payload.get(
                "supervision_counter_baseline"
            ),
            "boundary_payload": self.boundary_payload,
            "elapsed_s": self.elapsed_s,
            "completed_warmup_result": False,
            "private_start_bank_exposed": False,
            "candidate_selection_authorized": False,
            "retuning_authorized": False,
            "verification_authorized": False,
            "promotion_authorized": False,
            "legacy_fallback_used": False,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "nonclaims": OPERATIONAL_WARMUP_NONCLAIMS,
        }


OperationalWarmupBoundaryCallback = Callable[
    [str, tuple[Mapping[str, Any], ...]], Mapping[str, Any] | None
]
OperationalWarmupSegmentCallback = Callable[
    [str, Mapping[str, Any]], Mapping[str, Any] | None
]
OperationalWarmupStageCallback = Callable[[str, Mapping[str, Any]], None]


def run_operational_windowed_warmup(
    *,
    adapter: Any,
    initial_transform: AffineCoordinateTransform,
    initial_canonical_theta: Any,
    initial_step_size: float,
    trajectory_policy: WarmupTrajectoryPolicy,
    config: WindowedMassAdaptationConfig,
    target_accept_prob: float,
    seed: tuple[int, int],
    target_scope: str,
    chain_execution_mode: str = "tf_function",
    target_status_trace_policy: str = "none",
    algorithm_id: str = OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
    route_contract_version: str = HMC_ROUTE_CONTRACT_VERSION,
    boundary_callback: OperationalWarmupBoundaryCallback | None = None,
    execution_segment_size: int | None = None,
    segment_callback: OperationalWarmupSegmentCallback | None = None,
    stage_callback: OperationalWarmupStageCallback | None = None,
) -> OperationalWindowedWarmupResult | OperationalWindowedWarmupCloseout:
    """Run real interleaved TF/TFP HMC warmup with operational metric rebuilds."""

    import tensorflow as tf
    import tensorflow_probability as tfp

    if not isinstance(initial_transform, AffineCoordinateTransform):
        raise TypeError("initial_transform must be AffineCoordinateTransform")
    if not isinstance(trajectory_policy, WarmupTrajectoryPolicy):
        raise TypeError("trajectory_policy must be WarmupTrajectoryPolicy")
    normalized_seed = _strict_seed(seed, name="seed")
    if chain_execution_mode not in {"eager", "tf_function"}:
        raise ValueError("chain_execution_mode must be eager or tf_function")
    target_status_policy = _target_status_policy(target_status_trace_policy)
    if target_status_policy == "per_chain_step" and not callable(
        getattr(adapter, "target_status_telemetry", None)
    ):
        raise TypeError(
            "target_status_trace_policy='per_chain_step' requires adapter telemetry"
        )
    config = normalize_operational_warmup_config(config)
    target_accept = float(target_accept_prob)
    if not np.isfinite(target_accept) or not 0.0 < target_accept < 1.0:
        raise ValueError("target_accept_prob must be finite and in (0, 1)")
    theta = np.asarray(initial_canonical_theta, dtype=float)
    if theta.shape != (initial_transform.dimension,) or not np.all(np.isfinite(theta)):
        raise ValueError("initial_canonical_theta must be one finite transform vector")
    latent = np.asarray(initial_transform.theta_to_latent(theta).numpy(), dtype=float)
    metric = MomentumMetric.identity_for(initial_transform)
    kernel_state = KernelState(
        canonical_theta=theta,
        active_latent=latent,
        transform=initial_transform,
        momentum_metric=metric,
        epsilon=None,
        trajectory_policy=trajectory_policy,
        adaptation_generation=0,
        seed_lineage=normalized_seed,
        evidence_status="initialized",
    )
    first_adapter = _AffineWarmupAdapter(
        base_adapter=adapter,
        transform=initial_transform,
        target_scope=target_scope,
    )
    reasonable = find_reasonable_epsilon(
        adapter=first_adapter,
        current_state=latent,
        initial_step_size=initial_step_size,
        seed=_seed(normalized_seed, -1, lane=1),
        target_status_trace_policy=target_status_policy,
    )
    if not reasonable.passed or reasonable.selected_step_size is None:
        raise ValueError("operational warmup reasonable epsilon search was inconclusive")
    epsilon = float(reasonable.selected_step_size)
    windows = build_windowed_warmup_schedule(config)
    segmented_execution = execution_segment_size is not None
    segment_size = (
        max(window.length for window in windows)
        if execution_segment_size is None
        else _strict_integer(
            execution_segment_size,
            name="execution_segment_size",
            minimum=1,
        )
    )
    planned_segment_count = sum(
        (window.length + segment_size - 1) // segment_size for window in windows
    )
    results: list[OperationalWarmupWindowResult] = []
    canonical_history: list[np.ndarray] = []
    transition_count = 0
    start = time.perf_counter()
    active_adaptive_kernel: Any | None = None
    active_runner: Any | None = None
    previous_kernel_results: Any | None = None
    runner_generation = -1

    def emit_stage(
        event: str,
        *,
        stage: str,
        window: WindowedWarmupWindow,
        stage_started: float | None = None,
    ) -> None:
        if stage_callback is None:
            return
        payload: dict[str, Any] = {
            "stage": str(stage),
            "window_index": int(window.index),
            "window_kind": str(window.kind),
            "logical_draw_count": int(window.length),
            "supports_retained_draw_batch": bool(
                getattr(adapter, "supports_retained_draw_batch", False)
            ),
            "supports_retained_flat_batch": bool(
                getattr(adapter, "supports_retained_flat_batch", False)
            ),
            "progress_only": True,
            "states_exposed": False,
            "scores_exposed": False,
            "metric_exposed": False,
            "epsilon_exposed": False,
        }
        if stage_started is not None:
            payload["stage_elapsed_s"] = time.perf_counter() - stage_started
        callback_result = stage_callback(str(event), payload)
        if callback_result is not None:
            raise ValueError("operational warmup stage callback must return None")

    def build_closeout(
        boundary: str,
        payload: Mapping[str, Any] | None,
        *,
        completed_transitions: int,
        completed_segments: int,
    ) -> OperationalWindowedWarmupCloseout | None:
        if payload is None:
            return None
        public_windows = tuple(item.public_payload() for item in results)
        return OperationalWindowedWarmupCloseout(
            algorithm_id=algorithm_id,
            route_contract_version=route_contract_version,
            boundary=str(boundary),
            completed_windows=public_windows,
            planned_window_count=len(windows),
            completed_transition_count=completed_transitions,
            planned_transition_count=config.warmup_steps,
            completed_segment_count=completed_segments,
            planned_segment_count=planned_segment_count,
            boundary_payload=dict(payload),
            elapsed_s=time.perf_counter() - start,
        )

    closeout = build_closeout(
        "before_first_window",
        None
        if boundary_callback is None
        else boundary_callback("before_first_window", ()),
        completed_transitions=0,
        completed_segments=0,
    )
    if closeout is not None:
        return closeout

    for window in windows:
        if results:
            public_windows = tuple(item.public_payload() for item in results)
            closeout = build_closeout(
                "before_next_window",
                None
                if boundary_callback is None
                else boundary_callback("before_next_window", public_windows),
                completed_transitions=transition_count,
                completed_segments=sum(
                    (item.window.length + segment_size - 1) // segment_size
                    for item in results
                ),
            )
            if closeout is not None:
                return closeout
        active_transform = kernel_state.transform
        active_metric = kernel_state.momentum_metric
        active_adapter = _AffineWarmupAdapter(
            base_adapter=adapter,
            transform=active_transform,
            target_scope=target_scope,
        )
        current_latent = tf.convert_to_tensor(kernel_state.active_latent, dtype=tf.float64)
        if active_adaptive_kernel is None:
            target = reviewed_value_score_target_fn(active_adapter, dtype=current_latent.dtype)
            base_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target,
                step_size=tf.constant(epsilon, dtype=current_latent.dtype),
                num_leapfrog_steps=trajectory_policy.num_leapfrog_steps,
            )
            active_adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=base_kernel,
                num_adaptation_steps=config.warmup_steps - transition_count,
                target_accept_prob=tf.constant(target_accept, dtype=current_latent.dtype),
            )
            previous_kernel_results = active_adaptive_kernel.bootstrap_results(
                current_latent
            )
            runner_generation += 1

        def trace_fn(_state: Any, kernel_results: Any) -> Mapping[str, Any]:
            inner = kernel_results.inner_results
            trace = {
                "is_accepted": inner.is_accepted,
                "log_accept_ratio": inner.log_accept_ratio,
                "step_size": kernel_results.new_step_size,
                "target_log_prob": inner.accepted_results.target_log_prob,
            }
            divergence = _native_divergence(inner)
            if divergence is not None:
                trace["divergence"] = divergence
            if target_status_policy == "per_chain_step":
                trace["target_status_telemetry"] = (
                    active_adapter.target_status_telemetry(_state)
                )
            return trace

        def run_window(
            state: Any,
            kernel_results: Any,
            num_results: Any,
            run_seed: Any,
        ) -> tuple[Any, Mapping[str, Any]]:
            return tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=0,
                current_state=state,
                previous_kernel_results=kernel_results,
                kernel=active_adaptive_kernel,
                trace_fn=trace_fn,
                return_final_kernel_results=True,
                seed=run_seed,
            )

        if active_runner is None:
            active_runner = (
                run_window
                if chain_execution_mode == "eager"
                else tf.function(run_window, reduce_retracing=True)
            )
        window_start = time.perf_counter()
        segment_states: list[Any] = []
        segment_traces: list[Mapping[str, Any]] = []
        completed_in_window = 0
        completed_before_window = sum(
            (item.window.length + segment_size - 1) // segment_size for item in results
        )
        segment_count = (window.length + segment_size - 1) // segment_size
        final_kernel_results = previous_kernel_results
        while completed_in_window < window.length:
            segment_index = completed_in_window // segment_size
            active_results = min(segment_size, window.length - completed_in_window)
            completed_transitions = transition_count + completed_in_window
            completed_segments = completed_before_window + segment_index
            public_segment = {
                "window_index": int(window.index),
                "window_kind": str(window.kind),
                "window_segment_index": int(segment_index),
                "window_segment_count": int(segment_count),
                "completed_transition_count": int(completed_transitions),
                "planned_transition_count": int(config.warmup_steps),
                "completed_segment_count": int(completed_segments),
                "planned_segment_count": int(planned_segment_count),
                "segment_transition_count": int(active_results),
                "progress_only": True,
                "hmc_mechanics_exposed": False,
            }
            closeout_payload = (
                None
                if segment_callback is None
                else segment_callback("segment_start", public_segment)
            )
            closeout = build_closeout(
                "before_next_segment",
                closeout_payload,
                completed_transitions=completed_transitions,
                completed_segments=completed_segments,
            )
            if closeout is not None:
                return closeout
            segment_started = time.perf_counter()
            checkpoint = active_runner(
                current_latent,
                final_kernel_results,
                tf.constant(active_results, dtype=tf.int32),
                tf.constant(
                    _seed(
                        normalized_seed,
                        window.index
                        if not segmented_execution
                        else window.index * 100_000 + segment_index,
                        lane=2,
                    ),
                    dtype=tf.int32,
                ),
            )
            segment_states.append(checkpoint.all_states)
            segment_traces.append(checkpoint.trace)
            final_kernel_results = checkpoint.final_kernel_results
            current_latent = checkpoint.all_states[-1]
            completed_in_window += active_results
            if segment_callback is not None:
                segment_callback(
                    "segment_complete",
                    {
                        **public_segment,
                        "completed_transition_count": int(
                            transition_count + completed_in_window
                        ),
                        "completed_segment_count": int(completed_segments + 1),
                        "segment_elapsed_s": time.perf_counter() - segment_started,
                    },
                )
        stage_started = time.perf_counter()
        emit_stage(
            "stage_start",
            stage="post_window_conversion",
            window=window,
        )
        latent_draws_tensor = tf.concat(segment_states, axis=0)
        trace = tf.nest.map_structure(
            lambda *parts: tf.concat(parts, axis=0),
            *segment_traces,
        )
        runtime_s = time.perf_counter() - window_start
        latent_draws = np.asarray(latent_draws_tensor.numpy(), dtype=float)
        if not np.all(np.isfinite(latent_draws)):
            raise ValueError("operational warmup produced nonfinite latent states")
        canonical_draws = np.asarray(active_transform.latent_to_theta(latent_draws).numpy())
        if not np.all(np.isfinite(canonical_draws)):
            raise ValueError("operational warmup produced nonfinite canonical states")
        emit_stage(
            "stage_complete",
            stage="post_window_conversion",
            window=window,
            stage_started=stage_started,
        )
        stage_started = time.perf_counter()
        emit_stage(
            "stage_start",
            stage="retained_target_health",
            window=window,
        )
        target_health = _evaluate_retained_target_health(
            adapter=active_adapter,
            samples=latent_draws,
            target_status_trace_policy=target_status_policy,
        )
        if target_health["shared_invalidity_reasons"]:
            raise ValueError("operational warmup retained target authority is invalid")
        if target_health["candidate_data_invalidity_reasons"]:
            raise ValueError("operational warmup retained target value/score is nonvalid")
        emit_stage(
            "stage_complete",
            stage="retained_target_health",
            window=window,
            stage_started=stage_started,
        )
        log_accept = np.asarray(trace["log_accept_ratio"].numpy(), dtype=float)
        if not np.all(np.isfinite(log_accept)):
            raise ValueError("operational warmup produced nonfinite log acceptance")
        target_values = np.asarray(trace["target_log_prob"].numpy(), dtype=float)
        if not np.all(np.isfinite(target_values)):
            raise ValueError("operational warmup produced nonfinite target values")
        step_trace = np.asarray(trace["step_size"].numpy(), dtype=float)
        if not np.all(np.isfinite(step_trace)) or np.any(step_trace <= 0.0):
            raise ValueError("operational warmup produced invalid dual-averaging step")
        epsilon_end = float(np.reshape(step_trace, [-1])[-1])
        accepted = np.asarray(trace["is_accepted"].numpy(), dtype=bool)
        mean_accept = float(np.mean(np.exp(np.minimum(log_accept, 0.0))))
        binary_accept = float(np.mean(accepted))
        if "divergence" in trace:
            divergence_count = int(np.sum(np.asarray(trace["divergence"].numpy(), bool)))
            divergence_status = "available"
        else:
            divergence_count = None
            divergence_status = "not_exposed_by_kernel"
        target_status_failure_count = None
        if target_status_policy == "per_chain_step":
            target_status_trace = trace.get("target_status_telemetry")
            if target_status_trace is None:
                raise ValueError("operational warmup target-status trace is missing")
            target_status_numpy = {
                key: np.asarray(value.numpy() if hasattr(value, "numpy") else value)
                for key, value in target_status_trace.items()
            }
            try:
                target_status_failed = target_status_telemetry_has_failure(
                    target_status_numpy,
                    expected_shape=(window.length,),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "operational warmup target-status trace is invalid"
                ) from exc
            status = np.asarray(target_status_numpy["status_code"])
            valid = np.asarray(
                target_status_numpy["valid_pre_regularized_score"],
                dtype=bool,
            )
            target_status_failure_count = int(np.sum((status != 0) | (~valid)))
            if target_status_failure_count != target_health[
                "target_status_failure_count"
            ]:
                raise ValueError(
                    "operational warmup target-status trace disagrees with retained states"
                )
            if target_status_failed:
                raise ValueError(
                    "operational warmup target-status telemetry vetoed a window"
                )

        final_latent = np.asarray(latent_draws[-1], dtype=float)
        final_theta = np.asarray(canonical_draws[-1], dtype=float)
        map_residual = float(
            np.max(
                np.abs(
                    np.asarray(active_transform.latent_to_theta(final_latent).numpy())
                    - final_theta
                )
            )
        )
        metric_decision = None
        next_transform = active_transform
        next_metric = active_metric
        next_coordinate_signature = None
        next_metric_signature = None
        next_reasonable_epsilon = None
        target_value_map_residual = None
        target_score_map_residual = None
        if window.update_mass:
            stage_started = time.perf_counter()
            emit_stage(
                "stage_start",
                stage="metric_assessment",
                window=window,
            )
            metric_decision = assess_metric_covariance(
                latent_draws.reshape((-1, active_transform.dimension)),
                shrinkage=config.mass_shrinkage,
            )
            emit_stage(
                "stage_complete",
                stage="metric_assessment",
                window=window,
                stage_started=stage_started,
            )
            if metric_decision.update_applied:
                stage_started = time.perf_counter()
                emit_stage(
                    "stage_start",
                    stage="metric_candidate_construction",
                    window=window,
                )
                latent_covariance = np.asarray(metric_decision.covariance, dtype=float)
                canonical_covariance = (
                    active_transform.factor
                    @ latent_covariance
                    @ active_transform.factor.T
                )
                canonical_mean = np.mean(
                    canonical_draws.reshape((-1, active_transform.dimension)),
                    axis=0,
                )
                try:
                    estimate = PositionCovarianceEstimate(
                        center=canonical_mean,
                        covariance=canonical_covariance,
                        source_coordinate_signature=active_transform.signature,
                        estimator_family=str(metric_decision.estimator_family),
                        state_count=int(
                            latent_draws.reshape(
                                (-1, active_transform.dimension)
                            ).shape[0]
                        ),
                        effective_rank=int(np.linalg.matrix_rank(latent_covariance)),
                        regularization_report=metric_decision.report,
                        adequacy_report={
                            "outcome": metric_decision.outcome,
                            "passed": True,
                        },
                    )
                    candidate_transform = (
                        AffineCoordinateTransform.from_covariance_estimate(estimate)
                    )
                    candidate_metric = MomentumMetric.identity_for(
                        candidate_transform
                    )
                    if candidate_transform.signature == active_transform.signature:
                        raise ValueError(
                            "metric update did not produce a new coordinate signature"
                        )
                except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
                    metric_decision = _rejected_metric_candidate(
                        metric_decision,
                        stage="transform_construction",
                        error=exc,
                    )
                emit_stage(
                    "stage_complete",
                    stage="metric_candidate_construction",
                    window=window,
                    stage_started=stage_started,
                )

            if metric_decision.update_applied:
                stage_started = time.perf_counter()
                emit_stage(
                    "stage_start",
                    stage="metric_candidate_affine_parity",
                    window=window,
                )
                averaged_log_step = np.asarray(
                    final_kernel_results.log_averaging_step[0], dtype=float
                )
                candidate_epsilon_start = float(
                    np.exp(np.reshape(averaged_log_step, [-1])[-1])
                )
                try:
                    mapped_latent = np.asarray(
                        candidate_transform.theta_to_latent(final_theta).numpy(),
                        dtype=float,
                    )
                    candidate_adapter = _AffineWarmupAdapter(
                        base_adapter=adapter,
                        transform=candidate_transform,
                        target_scope=target_scope,
                    )
                    base_value, base_score = adapter.log_prob_and_grad(
                        tf.convert_to_tensor(final_theta, dtype=tf.float64)
                    )
                    old_value, old_score = active_adapter.log_prob_and_grad(
                        tf.convert_to_tensor(final_latent, dtype=tf.float64)
                    )
                    new_value, new_score = candidate_adapter.log_prob_and_grad(
                        tf.convert_to_tensor(mapped_latent, dtype=tf.float64)
                    )
                    base_value_array = np.asarray(base_value.numpy(), dtype=float)
                    candidate_value_residual = float(
                        max(
                            np.max(
                                np.abs(
                                    np.asarray(old_value.numpy()) - base_value_array
                                )
                            ),
                            np.max(
                                np.abs(
                                    np.asarray(new_value.numpy()) - base_value_array
                                )
                            ),
                        )
                    )
                    expected_old_score = np.asarray(
                        active_transform.theta_score_to_latent_score(
                            base_score
                        ).numpy(),
                        dtype=float,
                    )
                    expected_new_score = np.asarray(
                        candidate_transform.theta_score_to_latent_score(
                            base_score
                        ).numpy(),
                        dtype=float,
                    )
                    candidate_score_residual = float(
                        max(
                            np.max(
                                np.abs(
                                    np.asarray(old_score.numpy())
                                    - expected_old_score
                                )
                            ),
                            np.max(
                                np.abs(
                                    np.asarray(new_score.numpy())
                                    - expected_new_score
                                )
                            ),
                        )
                    )
                    if (
                        candidate_value_residual > 1.0e-10
                        or candidate_score_residual > 1.0e-10
                    ):
                        raise ValueError(
                            "target value/score changed across the metric boundary"
                        )
                except (
                    TypeError,
                    ValueError,
                    np.linalg.LinAlgError,
                    tf.errors.InvalidArgumentError,
                ) as exc:
                    metric_decision = _rejected_metric_candidate(
                        metric_decision,
                        stage="affine_target_parity",
                        error=exc,
                    )
                emit_stage(
                    "stage_complete",
                    stage="metric_candidate_affine_parity",
                    window=window,
                    stage_started=stage_started,
                )

            if metric_decision.update_applied:
                stage_started = time.perf_counter()
                emit_stage(
                    "stage_start",
                    stage="metric_boundary_reasonable_epsilon",
                    window=window,
                )
                try:
                    candidate_reasonable = find_reasonable_epsilon(
                        adapter=candidate_adapter,
                        current_state=mapped_latent,
                        initial_step_size=candidate_epsilon_start,
                        seed=_seed(normalized_seed, window.index, lane=3),
                        target_status_trace_policy=target_status_policy,
                    )
                    if (
                        not candidate_reasonable.passed
                        or candidate_reasonable.selected_step_size is None
                    ):
                        raise ValueError(
                            "metric-boundary reasonable epsilon search was inconclusive"
                        )
                except (
                    TypeError,
                    ValueError,
                    RuntimeError,
                    tf.errors.InvalidArgumentError,
                ) as exc:
                    metric_decision = _rejected_metric_candidate(
                        metric_decision,
                        stage="reasonable_epsilon",
                        error=exc,
                    )
                emit_stage(
                    "stage_complete",
                    stage="metric_boundary_reasonable_epsilon",
                    window=window,
                    stage_started=stage_started,
                )

            if metric_decision.update_applied:
                next_transform = candidate_transform
                next_metric = candidate_metric
                next_coordinate_signature = next_transform.signature
                next_metric_signature = next_metric.signature
                target_value_map_residual = candidate_value_residual
                target_score_map_residual = candidate_score_residual
                next_reasonable_epsilon = candidate_reasonable
                epsilon_end = float(candidate_reasonable.selected_step_size)
                active_adaptive_kernel = None
                active_runner = None
                previous_kernel_results = None

        metric_update_applied = (
            metric_decision is not None and metric_decision.update_applied
        )
        if not metric_update_applied:
            previous_kernel_results = final_kernel_results

        mapped_latent = np.asarray(
            next_transform.theta_to_latent(final_theta).numpy(), dtype=float
        )
        state_map_residual = float(
            np.max(
                np.abs(
                    np.asarray(next_transform.latent_to_theta(mapped_latent).numpy())
                    - final_theta
                )
            )
        )

        results.append(
            OperationalWarmupWindowResult(
                window=window,
                transition_count_before_window=transition_count,
                transition_count_after_window=transition_count + window.length,
                coordinate_signature_used=active_transform.signature,
                metric_signature_used=active_metric.signature,
                epsilon_start=epsilon,
                epsilon_end=epsilon_end,
                mean_acceptance_probability=mean_accept,
                binary_acceptance_rate=binary_accept,
                native_divergence_status=divergence_status,
                native_divergence_count=divergence_count,
                target_status_trace_policy=target_status_policy,
                target_status_failure_count=target_status_failure_count,
                max_abs_log_accept_energy_proxy=float(np.max(np.abs(log_accept))),
                final_latent_state=final_latent,
                final_canonical_theta=final_theta,
                adaptation_latent_states=latent_draws,
                adaptation_canonical_states=canonical_draws,
                log_accept_ratio=log_accept,
                is_accepted=accepted,
                target_log_prob=target_values,
                step_size_trace=step_trace,
                metric_decision=metric_decision,
                next_coordinate_signature=next_coordinate_signature,
                next_metric_signature=next_metric_signature,
                state_map_residual=max(map_residual, state_map_residual),
                target_value_map_residual=target_value_map_residual,
                target_score_map_residual=target_score_map_residual,
                next_reasonable_epsilon=next_reasonable_epsilon,
                dual_averaging_generation=kernel_state.adaptation_generation,
                runner_generation=runner_generation,
                runner_trace_count=(
                    None
                    if chain_execution_mode == "eager"
                    else int(active_runner.experimental_get_tracing_count())
                    if active_runner is not None
                    else 1
                ),
                runtime_s=runtime_s,
            )
        )
        canonical_history.extend(
            np.asarray(canonical_draws, dtype=float).reshape((-1, active_transform.dimension))
        )
        transition_count += window.length
        kernel_state = KernelState(
            canonical_theta=final_theta,
            active_latent=mapped_latent,
            transform=next_transform,
            momentum_metric=next_metric,
            epsilon=None,
            trajectory_policy=trajectory_policy,
            adaptation_generation=kernel_state.adaptation_generation
            + (1 if metric_update_applied else 0),
            seed_lineage=normalized_seed,
            evidence_status="metric_updated"
            if metric_update_applied
            else "warmup_window_complete",
        )
        # A transform change invalidates epsilon; every window starts a fresh
        # dual-averaging generation from the previous stable scalar proposal.
        epsilon = epsilon_end

    kernel_state = kernel_state.with_epsilon(
        epsilon,
        evidence_status="metric_and_step_frozen",
    )
    history = np.asarray(results[-1].adaptation_canonical_states, dtype=float).reshape(
        (-1, initial_transform.dimension)
    )
    bank = build_private_start_bank(
        history,
        reference_transform=kernel_state.transform,
    )
    result = OperationalWindowedWarmupResult(
        config=config,
        initial_coordinate_signature=initial_transform.signature,
        final_kernel_state=kernel_state,
        reasonable_epsilon=reasonable,
        windows=tuple(results),
        private_start_bank_theta=bank,
        seed_root=normalized_seed,
        target_scope=str(target_scope),
        target_status_trace_policy=target_status_policy,
        elapsed_s=time.perf_counter() - start,
        algorithm_id=algorithm_id,
        route_contract_version=route_contract_version,
    )
    return result


def build_private_start_bank(
    canonical_states: Any,
    *,
    reference_transform: AffineCoordinateTransform | None = None,
    minimum_relative_separation: float = 1.0e-4,
) -> np.ndarray:
    """Select canonical starts with material separation in reference geometry."""

    states = np.asarray(canonical_states, dtype=float)
    if states.ndim != 2 or states.shape[0] < 4 or not np.all(np.isfinite(states)):
        raise ValueError("start bank source must contain at least four finite states")
    if reference_transform is not None and not isinstance(
        reference_transform, AffineCoordinateTransform
    ):
        raise TypeError("reference_transform must be an AffineCoordinateTransform")
    separation = float(minimum_relative_separation)
    if not np.isfinite(separation) or separation <= 0.0:
        raise ValueError("minimum_relative_separation must be finite and positive")
    reference = (
        states
        if reference_transform is None
        else np.asarray(reference_transform.theta_to_latent(states).numpy(), dtype=float)
    )
    reference_scale = max(
        float(np.sqrt(states.shape[1])),
        float(np.linalg.norm(np.std(reference, axis=0))),
    )
    tolerance = separation * reference_scale
    endpoint_index = states.shape[0] - 1
    eligible_indices: list[int] = []
    for index in range(endpoint_index):
        if np.linalg.norm(reference[index] - reference[endpoint_index]) <= tolerance:
            continue
        if all(
            np.linalg.norm(reference[index] - reference[existing]) > tolerance
            for existing in eligible_indices
        ):
            eligible_indices.append(index)
    if len(eligible_indices) < 3:
        raise ValueError("operational warmup start bank is not sufficiently dispersed")
    selected = np.linspace(0, len(eligible_indices) - 1, 3, dtype=int)
    bank_indices = [eligible_indices[index] for index in selected] + [endpoint_index]
    bank = states[bank_indices].astype(float, copy=True)
    reference_bank = reference[bank_indices]
    pairwise = np.linalg.norm(
        reference_bank[:, None, :] - reference_bank[None, :, :], axis=-1
    )
    if np.any(pairwise[np.triu_indices(4, k=1)] <= tolerance):
        raise ValueError("operational warmup start bank is not sufficiently dispersed")
    bank.setflags(write=False)
    return bank


def _native_divergence(kernel_results: Any) -> Any | None:
    for name in ("is_divergent", "has_divergence", "divergence", "divergences"):
        value = getattr(kernel_results, name, None)
        if value is not None:
            return value
    return None


def compose_base_transform_with_nested_artifact(
    *,
    base_transform: AffineCoordinateTransform,
    nested_artifact: Any,
    source_coordinate_signature: str,
) -> AffineCoordinateTransform:
    """Compose ``theta=c0+A0 z0`` with historical ``z0=c1+A1 z``."""

    nested_center = np.asarray(nested_artifact.position, dtype=float)
    nested_factor = np.asarray(nested_artifact.factor, dtype=float)
    if nested_center.shape != (base_transform.dimension,):
        raise ValueError("nested artifact center dimension mismatch")
    if nested_factor.shape != (base_transform.dimension, base_transform.dimension):
        raise ValueError("nested artifact factor dimension mismatch")
    canonical_center = np.asarray(
        base_transform.latent_to_theta(nested_center).numpy(), dtype=float
    )
    canonical_factor = base_transform.factor @ nested_factor
    canonical_covariance = canonical_factor @ canonical_factor.T
    estimate = PositionCovarianceEstimate(
        center=canonical_center,
        covariance=canonical_covariance,
        source_coordinate_signature=str(source_coordinate_signature),
        estimator_family="historical_nested_affine_composition",
        state_count=max(1, base_transform.dimension),
        effective_rank=int(np.linalg.matrix_rank(canonical_covariance)),
        regularization_report={
            "method": "exact_forward_affine_composition",
            "base_coordinate_signature": base_transform.signature,
            "nested_artifact_source": str(getattr(nested_artifact, "source", "unknown")),
        },
        adequacy_report={
            "legacy_compatibility_adapter": True,
            "operational_metric_adequacy_not_inferred": True,
        },
        evidence_role="carried_operational_transform_lineage",
    )
    transform = AffineCoordinateTransform(
        center=canonical_center,
        factor=canonical_factor,
        covariance_signature=estimate.signature,
    )
    probe = np.stack(
        [np.zeros(base_transform.dimension), np.linspace(-0.2, 0.3, base_transform.dimension)]
    )
    nested_theta = base_transform.latent_to_theta(
        nested_artifact.build_latent_transform().latent_to_position(probe)
    )
    direct_theta = transform.latent_to_theta(probe)
    if not np.allclose(nested_theta, direct_theta, rtol=1.0e-10, atol=1.0e-10):
        raise ValueError("forward operational compatibility composition failed")
    return transform


def compose_operational_transform_in_base_coordinates(
    *,
    base_transform: AffineCoordinateTransform,
    final_transform: AffineCoordinateTransform,
    adapter_signature: str,
    source: str = "operational_windowed_warmup_composition",
) -> Any:
    """Express a canonical final transform as a transform inside base latent space.

    If ``theta = c0 + A0 z0`` and ``theta = c1 + A1 z1``, the compatibility
    artifact represents ``z0 = b + B z1`` with
    ``b = A0^{-1}(c1-c0)`` and ``B = A0^{-1} A1``. Composing the historical
    Phase 5 adapters then recovers exactly the final canonical transform.
    """

    from bayesfilter.inference.hmc import PrecomputedMassArtifact

    if base_transform.dimension != final_transform.dimension:
        raise ValueError("base and final transform dimensions must match")
    center_delta = final_transform.center - base_transform.center
    nested_center = np.linalg.solve(base_transform.factor, center_delta)
    nested_factor = np.linalg.solve(base_transform.factor, final_transform.factor)
    nested_covariance = nested_factor @ nested_factor.T
    artifact = PrecomputedMassArtifact(
        position=nested_center,
        covariance=nested_covariance,
        factor=nested_factor,
        adapter_signature=str(adapter_signature),
        position_role="operational_final_center_in_geometry_latent",
        covariance_source="operational_final_covariance_in_geometry_latent",
        matrix_used_for_square_root="affine_composition_factor",
        source=str(source),
        regularization_report={
            "method": "exact_affine_composition",
            "base_coordinate_signature": base_transform.signature,
            "final_coordinate_signature": final_transform.signature,
            "double_composition_forbidden": True,
        },
        nonclaims=OPERATIONAL_WARMUP_NONCLAIMS,
    )
    probe = np.stack(
        [np.zeros(base_transform.dimension), np.linspace(-0.3, 0.4, base_transform.dimension)]
    )
    nested_theta = base_transform.latent_to_theta(
        artifact.build_latent_transform().latent_to_position(probe)
    )
    direct_theta = final_transform.latent_to_theta(probe)
    if not np.allclose(nested_theta, direct_theta, rtol=1.0e-10, atol=1.0e-10):
        raise ValueError("operational compatibility composition failed")
    return artifact
