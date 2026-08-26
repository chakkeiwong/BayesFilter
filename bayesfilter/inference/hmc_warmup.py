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
import math
import sys
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from bayesfilter.hmc_route_contract import (
    HMC_ROUTE_CONTRACT_VERSION,
    OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
)
from bayesfilter.inference.batched_value_score import (
    _call_mapping_with_batch_rank_bridge,
    _call_value_score_with_batch_rank_bridge,
    reviewed_value_score_target_fn,
)
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

_START_BANK_SCOPE_SCHEMA = "bayesfilter.hmc_start_bank_scope_assessment.v1"
_START_BANK_QUALIFICATION_SCHEMA = "bayesfilter.hmc_start_bank_qualification.v1"
_START_BANK_POLICY_ID = "bayesfilter.greedy_four_start_bank.v1"
_START_BANK_DIAGNOSTIC_ATTRIBUTE = (
    "_bayesfilter_start_bank_qualification_diagnostic_v1"
)
_START_BANK_SCOPE_NAMES = frozenset(
    {"authoritative_final_window", "shadow_all_windows"}
)
_START_BANK_FAILURE_CODES = frozenset(
    {
        "none",
        "insufficient_greedy_eligible",
        "post_selection_pairwise_failure",
        "shadow_input_conversion_failure",
        "shadow_invalid_shape",
        "shadow_nonfinite_source",
        "shadow_nonfinite_reference",
        "shadow_reference_conversion_failure",
        "shadow_assessment_failure",
    }
)
_START_BANK_INTERPRETATIONS = frozenset(
    {
        "final_pass",
        "final_fail_shadow_pass",
        "both_fail",
        "post_selection_invariant_failure",
    }
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


def _unbiased_covariance_and_correlation(states: Any) -> tuple[Any, Any, Any]:
    """Return TensorFlow `N-1` covariance, variances, and correlation."""

    import tensorflow as tf

    tensor = tf.cast(tf.convert_to_tensor(states), tf.float64)
    if tensor.shape.rank != 2 or any(dim is None for dim in tensor.shape):
        raise ValueError("covariance states must have a static rank-2 shape")
    count = int(tensor.shape[0])
    if count < 2 or int(tensor.shape[1]) <= 0:
        raise ValueError("covariance requires at least two nonempty states")
    tf.debugging.assert_all_finite(tensor, "covariance states must be finite")
    centered = tensor - tf.reduce_mean(tensor, axis=0, keepdims=True)
    covariance = tf.matmul(centered, centered, transpose_a=True) / tf.cast(
        count - 1,
        tf.float64,
    )
    covariance = 0.5 * (covariance + tf.transpose(covariance))
    variances = tf.linalg.diag_part(covariance)
    safe_scale = tf.where(
        tf.math.is_finite(variances) & (variances > 0.0),
        tf.sqrt(variances),
        tf.ones_like(variances),
    )
    correlation = covariance / (safe_scale[:, None] * safe_scale[None, :])
    correlation = 0.5 * (correlation + tf.transpose(correlation))
    return covariance, variances, correlation


def assess_metric_covariance(
    latent_states: Any,
    *,
    shrinkage: float = 0.25,
    dense_min_states: int | None = None,
    diagonal_min_states: int | None = None,
) -> MetricAdequacyDecision:
    """Assess a unit-equivariant position covariance in correlation space.

    Numerical decisions in this active adaptation path are TensorFlow-owned.
    The covariance is materialized only when constructing the immutable public
    decision boundary.
    """

    import tensorflow as tf

    temporal = tf.cast(tf.convert_to_tensor(latent_states), tf.float64)
    rank = temporal.shape.rank
    if rank not in {2, 3} or any(dim is None for dim in temporal.shape):
        raise ValueError(
            "latent_states must be draw/dimension or draw/chain/dimension"
        )
    if int(temporal.shape[0]) < 2 or int(temporal.shape[-1]) <= 0:
        raise ValueError(
            "latent_states must be draw/dimension or draw/chain/dimension"
        )
    if not bool(tf.reduce_all(tf.math.is_finite(temporal)).numpy()):
        raise ValueError("latent_states must be finite")
    explicit_chains = rank == 3
    chain_states = temporal[:, None, :] if not explicit_chains else temporal
    dimension = int(chain_states.shape[-1])
    states = tf.reshape(chain_states, (-1, dimension))
    n = int(states.shape[0])
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
        max(32, 2 * int(math.ceil(math.log2(dimension + 1))))
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
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("shrinkage must be finite and in [0, 1]")

    ess_tensor = _summed_chain_ess(chain_states)
    ess_by_coordinate = tuple(float(item) for item in ess_tensor.numpy().tolist())
    min_ess = float(tf.reduce_min(ess_tensor).numpy())
    dense_min_ess = max(8, dimension + 1)
    diagonal_min_ess = max(4, int(math.ceil(math.log2(dimension + 1))))
    split_rhat = _split_rhat_by_coordinate(chain_states)
    split_rhat_finite = bool(
        split_rhat is not None
        and tf.reduce_all(tf.math.is_finite(split_rhat)).numpy()
    )
    max_split_rhat = (
        float(tf.reduce_max(split_rhat).numpy()) if split_rhat_finite else None
    )
    dense_chain_compatible = bool(
        not explicit_chains
        or (split_rhat_finite and max_split_rhat <= 1.10)
    ) if split_rhat is not None else not explicit_chains
    diagonal_chain_compatible = bool(
        not explicit_chains
        or (split_rhat_finite and max_split_rhat <= 1.25)
    ) if split_rhat is not None else not explicit_chains

    empirical, empirical_diagonal, correlation = (
        _unbiased_covariance_and_correlation(states)
    )
    diagonal_finite_positive = bool(
        tf.reduce_all(
            tf.math.is_finite(empirical_diagonal) & (empirical_diagonal > 0.0)
        ).numpy()
    )
    standardized_eigenvalues = tf.linalg.eigvalsh(correlation)
    spectrum_finite = bool(
        tf.reduce_all(tf.math.is_finite(standardized_eigenvalues)).numpy()
    )
    maximum_abs_eigenvalue = tf.reduce_max(tf.abs(standardized_eigenvalues))
    rank_tolerance = (
        tf.cast(dimension, tf.float64)
        * tf.constant(sys.float_info.epsilon, tf.float64)
        * maximum_abs_eigenvalue
    )
    standardized_rank = int(
        tf.reduce_sum(
            tf.cast(standardized_eigenvalues > rank_tolerance, tf.int32)
        ).numpy()
    )
    standardized_positive = bool(
        spectrum_finite
        and tf.reduce_all(standardized_eigenvalues > 0.0).numpy()
    )
    standardized_condition = (
        float(
            (
                tf.reduce_max(standardized_eigenvalues)
                / tf.reduce_min(standardized_eigenvalues)
            ).numpy()
        )
        if standardized_positive
        else float("inf")
    )
    identity = tf.eye(dimension, dtype=tf.float64)
    dense_standardized = (1.0 - weight) * correlation + weight * identity
    dense_shrunk = (1.0 - weight) * empirical + weight * tf.linalg.diag(
        empirical_diagonal
    )
    dense_shrunk = 0.5 * (dense_shrunk + tf.transpose(dense_shrunk))
    dense_discrepancy = float(
        (
            tf.norm(dense_standardized - correlation)
            / tf.maximum(
                tf.norm(correlation),
                tf.constant(sys.float_info.epsilon, tf.float64),
            )
        ).numpy()
    )
    dense_candidate_eigenvalues = tf.linalg.eigvalsh(dense_shrunk)
    dense_candidate_finite_positive = bool(
        tf.reduce_all(
            tf.math.is_finite(dense_candidate_eigenvalues)
            & (dense_candidate_eigenvalues > 0.0)
        ).numpy()
    )
    dense_checks = {
        "state_count_sufficient": n >= dense_required,
        "effective_information_sufficient": min_ess >= dense_min_ess,
        "cross_chain_location_compatible": dense_chain_compatible,
        "full_raw_rank": standardized_rank == dimension,
        "raw_condition_acceptable": standardized_condition <= 1.0e8,
        "regularization_discrepancy_acceptable": dense_discrepancy <= 0.50,
        "candidate_covariance_finite_positive": dense_candidate_finite_positive,
    }
    common = {
        "state_count": n,
        "dimension": dimension,
        "dense_min_states": dense_required,
        "diagonal_min_states": diagonal_required,
        "covariance_backend": "tensorflow_float64_unbiased_n_minus_1",
        "ess_method": "geyer_initial_positive_sequence_fft_autocorrelation",
        "ess_positive_variance_rule": "finite_and_strictly_positive_scale_free",
        "effective_sample_size_by_coordinate": ess_by_coordinate,
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
                float(item) if math.isfinite(float(item)) else None
                for item in split_rhat.numpy().tolist()
            )
        ),
        "maximum_split_rhat": max_split_rhat,
        "adequacy_geometry_space": "correlation",
        "rank_tolerance": float(rank_tolerance.numpy()),
        "raw_numerical_rank": standardized_rank,
        "standardized_numerical_rank": standardized_rank,
        "raw_condition_number": standardized_condition,
        "standardized_condition_number": standardized_condition,
        "standardized_eigenvalues": tuple(
            float(item) for item in standardized_eigenvalues.numpy().tolist()
        ),
        "shrinkage": weight,
        "dense_relative_frobenius_discrepancy": dense_discrepancy,
        "dense_discrepancy_space": "correlation",
        "dense_checks": dense_checks,
    }
    if all(dense_checks.values()):
        return MetricAdequacyDecision(
            outcome="dense_update",
            covariance=dense_shrunk.numpy(),
            estimator_family="dense_unbiased_correlation_shrinkage",
            report={
                **common,
                "eigenvalue_floor": None,
                "clipped_eigenvalue_count": 0,
                "regularization_method": "shrink_correlations_preserve_variances",
                "absolute_regularization_applied": False,
                "dense_information_gate_passed": True,
                "diagonal_fallback_used": False,
            },
        )

    diagonal_discrepancy = float(
        (
            tf.norm(identity - correlation)
            / tf.maximum(
                tf.norm(correlation),
                tf.constant(sys.float_info.epsilon, tf.float64),
            )
        ).numpy()
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
        "diagonal_discrepancy_space": "correlation",
        "diagonal_checks": diagonal_checks,
        "dense_information_gate_passed": False,
    }
    if all(diagonal_checks.values()):
        return MetricAdequacyDecision(
            outcome="diagonal_fallback",
            covariance=tf.linalg.diag(empirical_diagonal).numpy(),
            estimator_family="diagonal_unbiased_variance_preserving",
            report={
                **report,
                "eigenvalue_floor": None,
                "clipped_eigenvalue_count": 0,
                "regularization_method": "diagonal_empirical_variances",
                "absolute_regularization_applied": False,
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


def _summed_chain_ess(chain_states: Any) -> Any:
    """TensorFlow Geyer ESS with time kept within each chain."""

    import tensorflow as tf

    tensor = tf.cast(tf.convert_to_tensor(chain_states), tf.float64)
    if tensor.shape.rank != 3 or any(dim is None for dim in tensor.shape):
        raise ValueError("ESS states must have static draw/chain/coordinate shape")
    draw_count, chain_count, dimension = (int(dim) for dim in tensor.shape)
    total = tf.zeros((dimension,), dtype=tf.float64)
    fft_size = 1 << int(math.ceil(math.log2(max(2, 2 * draw_count))))
    for chain_index in range(chain_count):
        chain = tensor[:, chain_index, :]
        centered = chain - tf.reduce_mean(chain, axis=0, keepdims=True)
        spectrum = tf.signal.rfft(tf.transpose(centered), fft_length=[fft_size])
        autocovariance = tf.transpose(
            tf.signal.irfft(
                spectrum * tf.math.conj(spectrum),
                fft_length=[fft_size],
            )
        )[:draw_count]
        variance = autocovariance[0]
        valid = tf.math.is_finite(variance) & (variance > 0.0)
        denominator = tf.where(valid, variance, tf.ones_like(variance))
        rho = tf.where(
            valid[None, :],
            autocovariance / denominator[None, :],
            tf.zeros_like(autocovariance),
        )
        pair_sum = tf.zeros((dimension,), dtype=tf.float64)
        active = valid
        previous = tf.fill((dimension,), tf.constant(float("inf"), tf.float64))
        for lag in range(1, draw_count - 1, 2):
            pair = tf.minimum(rho[lag] + rho[lag + 1], previous)
            positive = active & tf.math.is_finite(pair) & (pair > 0.0)
            pair_sum = pair_sum + tf.where(positive, pair, 0.0)
            active = active & positive
            previous = tf.where(positive, pair, previous)
        tau = tf.maximum(1.0, 1.0 + 2.0 * pair_sum)
        chain_ess = tf.where(
            valid,
            tf.clip_by_value(float(draw_count) / tau, 1.0, float(draw_count)),
            tf.zeros_like(tau),
        )
        total = total + chain_ess
    tf.debugging.assert_all_finite(total, "metric ESS must be finite")
    return total


def _split_rhat_by_coordinate(chain_states: Any) -> Any | None:
    """Return TensorFlow split R-hat for multiple explicit chains."""

    import tensorflow as tf

    tensor = tf.cast(tf.convert_to_tensor(chain_states), tf.float64)
    if tensor.shape.rank != 3 or any(dim is None for dim in tensor.shape):
        raise ValueError("split R-hat states require a static rank-3 shape")
    draw_count, chain_count, _dimension = (int(dim) for dim in tensor.shape)
    if chain_count < 2 or draw_count < 4:
        return None
    half = draw_count // 2
    split = tf.concat(
        (tensor[:half], tensor[draw_count - half :]),
        axis=1,
    )
    split_means = tf.reduce_mean(split, axis=0)
    within_chain_centered = split - split_means[None, :, :]
    within_chain_variance = tf.reduce_sum(
        tf.square(within_chain_centered), axis=0
    ) / tf.cast(half - 1, tf.float64)
    within = tf.reduce_mean(within_chain_variance, axis=0)
    grand_mean = tf.reduce_mean(split_means, axis=0, keepdims=True)
    split_chain_count = int(split.shape[1])
    between = tf.cast(half, tf.float64) * tf.reduce_sum(
        tf.square(split_means - grand_mean), axis=0
    ) / tf.cast(split_chain_count - 1, tf.float64)
    variance = ((half - 1.0) / half) * within + between / half
    return tf.sqrt(variance / within)


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
        self.supports_retained_value_score_status = bool(
            getattr(base_adapter, "supports_retained_value_score_status", False)
            and callable(getattr(base_adapter, "log_prob_and_grad_status", None))
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
        value, score = self._log_prob_and_grad_status(latent)[:2]
        return value, score

    def _log_prob_and_grad_status(
        self, latent: Any
    ) -> tuple[Any, Any, Mapping[str, Any] | None]:
        import tensorflow as tf

        z = tf.convert_to_tensor(latent, dtype=tf.float64)
        theta = self.transform.latent_to_theta(z)
        value, theta_score, status = _call_value_score_with_batch_rank_bridge(
            self.base_adapter,
            theta,
            with_status=self.supports_retained_value_score_status,
        )
        return (
            tf.convert_to_tensor(value, dtype=z.dtype),
            self.transform.theta_score_to_latent_score(theta_score),
            status,
        )

    def log_prob_and_grad_status(
        self, latent: Any
    ) -> tuple[Any, Any, Mapping[str, Any]]:
        if not self.supports_retained_value_score_status:
            raise TypeError("base adapter does not expose combined value/score/status")
        value, score, status = self._log_prob_and_grad_status(latent)
        if not isinstance(status, Mapping):
            raise TypeError("combined value/score/status target must return a mapping")
        return value, score, dict(status)

    def target_status_telemetry(self, latent: Any) -> Mapping[str, Any]:
        telemetry = getattr(self.base_adapter, "target_status_telemetry", None)
        if not callable(telemetry):
            raise TypeError("base_adapter must expose target_status_telemetry")
        theta = self.transform.latent_to_theta(latent)
        return _call_mapping_with_batch_rank_bridge(
            self.base_adapter,
            telemetry,
            theta,
        )

    def classify_target_exception(self, error: BaseException) -> bool:
        """Forward only an adapter-declared target-domain failure classifier."""

        classifier = getattr(self.base_adapter, "classify_target_exception", None)
        if not callable(classifier):
            return False
        result = classifier(error)
        if not isinstance(result, (bool, np.bool_)):
            raise TypeError("classify_target_exception must return a boolean")
        return bool(result)


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
    num_leapfrog_steps: int = 1
    probe_seeds: tuple[tuple[int, int], ...] = ()
    minimum_acceptance_probability: float | None = None
    maximum_acceptance_probability: float | None = None
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
        if not set(health_failures).issubset(
            {"target_status_telemetry_failure", "target_domain_execution_failure"}
        ):
            raise ValueError("unsupported reasonable-epsilon health failure")
        seed = _strict_seed(self.seed, name="attempt seed")
        probe_seeds = tuple(
            _strict_seed(item, name="probe seed") for item in self.probe_seeds
        )
        if not probe_seeds:
            probe_seeds = (seed,)
        if probe_seeds[0] != seed or len(set(probe_seeds)) != len(probe_seeds):
            raise ValueError("reasonable-epsilon probe seeds must be distinct and led by seed")
        minimum = self.minimum_acceptance_probability
        maximum = self.maximum_acceptance_probability
        if finite:
            minimum = mean if minimum is None else float(minimum)
            maximum = mean if maximum is None else float(maximum)
            if (
                not np.all(np.isfinite((minimum, maximum)))
                or not 0.0 <= minimum <= mean <= maximum <= 1.0
            ):
                raise ValueError("reasonable-epsilon probe acceptance range is invalid")
        elif minimum is not None or maximum is not None:
            raise ValueError("nonfinite reasonable-epsilon attempt cannot carry acceptance range")
        object.__setattr__(self, "step_size", step)
        object.__setattr__(self, "mean_acceptance_probability", mean)
        object.__setattr__(self, "finite", finite)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(
            self,
            "num_leapfrog_steps",
            _strict_integer(
                self.num_leapfrog_steps,
                name="num_leapfrog_steps",
                minimum=1,
            ),
        )
        object.__setattr__(self, "probe_seeds", probe_seeds)
        object.__setattr__(self, "minimum_acceptance_probability", minimum)
        object.__setattr__(self, "maximum_acceptance_probability", maximum)
        object.__setattr__(self, "engineering_health_failures", health_failures)

    @property
    def usable(self) -> bool:
        return self.finite and not self.engineering_health_failures

    def payload(self) -> Mapping[str, Any]:
        return {
            "step_size": self.step_size,
            "mean_acceptance_probability": self.mean_acceptance_probability,
            "finite": self.finite,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "probe_count": len(self.probe_seeds),
            "probe_seeds": self.probe_seeds,
            "minimum_acceptance_probability": self.minimum_acceptance_probability,
            "maximum_acceptance_probability": self.maximum_acceptance_probability,
            "engineering_health_failures": self.engineering_health_failures,
            "usable": self.usable,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ReasonableEpsilonResult:
    status: str
    selected_step_size: float | None
    attempts: tuple[ReasonableEpsilonAttempt, ...]
    qualification_source: str | None = None

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in {"passed", "externally_qualified", "inconclusive_bracket"}:
            raise ValueError("unsupported reasonable-epsilon status")
        attempts = tuple(self.attempts)
        if attempts and not all(
            isinstance(item, ReasonableEpsilonAttempt) for item in attempts
        ):
            raise ValueError("reasonable-epsilon result attempts must be typed")
        selected = self.selected_step_size
        if status == "passed":
            if selected is None or not attempts:
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
        elif status == "externally_qualified":
            if attempts or selected is None:
                raise ValueError(
                    "externally qualified epsilon requires one selected step and no probes"
                )
            selected = float(selected)
            if not np.isfinite(selected) or selected <= 0.0:
                raise ValueError("externally qualified epsilon must be positive and finite")
        elif selected is not None:
            raise ValueError("inconclusive reasonable-epsilon result cannot select a step")
        source = None if self.qualification_source is None else str(self.qualification_source)
        if status == "externally_qualified":
            if not source:
                raise ValueError("externally qualified epsilon requires provenance")
        elif source is not None:
            raise ValueError("epsilon qualification source is only valid for external evidence")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "selected_step_size", selected)
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "qualification_source", source)

    @property
    def passed(self) -> bool:
        return self.status in {"passed", "externally_qualified"}

    def payload(self) -> Mapping[str, Any]:
        return {
            "status": self.status,
            "selected_step_size": self.selected_step_size,
            "attempts": tuple(attempt.payload() for attempt in self.attempts),
            "qualification_source": self.qualification_source,
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
    num_leapfrog_steps: int = 1,
    momentum_probe_count: int = 1,
    target_status_trace_policy: str = "none",
    jit_compile: bool = False,
) -> ReasonableEpsilonResult:
    """Bracket epsilon using the fixed trajectory that will consume it."""

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
    leapfrog_steps = _strict_integer(
        num_leapfrog_steps,
        name="num_leapfrog_steps",
        minimum=1,
    )
    probe_count = _strict_integer(
        momentum_probe_count,
        name="momentum_probe_count",
        minimum=1,
    )
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
    proposal_seeds = tuple(_seed(normalized_seed, index) for index in range(probe_count))

    def is_declared_target_domain_failure(error: BaseException) -> bool:
        classifier = getattr(adapter, "classify_target_exception", None)
        if not callable(classifier):
            return False
        try:
            result = classifier(error)
        except Exception as exc:  # noqa: BLE001 - classifier is adapter authority.
            raise RuntimeError(
                "target exception classifier failed during epsilon search"
            ) from exc
        if not isinstance(result, (bool, np.bool_)):
            raise TypeError("classify_target_exception must return a boolean")
        return bool(result)

    for index in range(attempt_limit):
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target,
            step_size=tf.constant(step, dtype=state.dtype),
            num_leapfrog_steps=leapfrog_steps,
        )
        if bool(jit_compile):
            bootstrap_results = tf.function(
                kernel.bootstrap_results,
                jit_compile=True,
                reduce_retracing=True,
            )
            results = bootstrap_results(state)
        else:
            results = kernel.bootstrap_results(state)
        if not _kernel_result_value_score_finite(results.accepted_results):
            raise ValueError("reasonable-epsilon bootstrap target evidence is nonfinite")
        if bool(jit_compile):
            @tf.function(jit_compile=True, reduce_retracing=True)
            def one_step(proposal_seed: tf.Tensor):
                return kernel.one_step(
                    state,
                    results,
                    seed=tf.convert_to_tensor(proposal_seed, tf.int32),
                )
        else:
            def one_step(proposal_seed: tf.Tensor):
                return kernel.one_step(
                    state,
                    results,
                    seed=tf.convert_to_tensor(proposal_seed, tf.int32),
                )
        acceptance_probabilities: list[float] = []
        finite = True
        health_failure_list: list[str] = []
        for proposal_seed in proposal_seeds:
            try:
                next_state, next_results = one_step(
                    tf.constant(proposal_seed, dtype=tf.int32)
                )
            except tf.errors.InvalidArgumentError as exc:
                if not is_declared_target_domain_failure(exc):
                    raise RuntimeError(
                        "unclassified TensorFlow InvalidArgumentError during "
                        "reasonable-epsilon search"
                    ) from exc
                finite = False
                health_failure_list.append("target_domain_execution_failure")
                break
            except tf.errors.InternalError as exc:
                if (
                    "covariance must be positive definite" not in str(exc)
                    or not is_declared_target_domain_failure(exc)
                ):
                    raise RuntimeError(
                        "reasonable-epsilon HMC proposal execution failed"
                    ) from exc
                finite = False
                health_failure_list.append("target_domain_execution_failure")
                break
            except tf.errors.OpError as exc:
                raise RuntimeError(
                    "reasonable-epsilon HMC proposal execution failed"
                ) from exc
            except Exception as exc:  # noqa: BLE001 - non-target runner failure is fatal.
                raise RuntimeError(
                    "reasonable-epsilon HMC proposal execution failed"
                ) from exc
            log_accept = tf.convert_to_tensor(next_results.log_accept_ratio, tf.float64)
            retained_finite = bool(
                _all_finite_tensors((next_state,))
                and _kernel_result_value_score_finite(next_results.accepted_results)
            )
            if not retained_finite:
                raise ValueError(
                    "reasonable-epsilon accepted or retained state is nonfinite"
                )
            probe_finite = bool(
                tf.reduce_all(tf.math.is_finite(log_accept)).numpy()
                and _all_finite_tensors((next_results.proposed_state,))
                and _kernel_result_value_score_finite(next_results.proposed_results)
            )
            if not probe_finite:
                finite = False
                break
            acceptance_probabilities.append(
                float(tf.reduce_mean(tf.exp(tf.minimum(log_accept, 0.0))).numpy())
            )
            if target_status_policy == "per_chain_step":
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
                    health_failure_list.append("target_status_telemetry_failure")
        mean_accept = (
            float(np.mean(acceptance_probabilities))
            if finite and len(acceptance_probabilities) == probe_count
            else None
        )
        health_failures = tuple(dict.fromkeys(health_failure_list))
        attempt = ReasonableEpsilonAttempt(
            step_size=step,
            mean_acceptance_probability=mean_accept,
            finite=finite,
            seed=proposal_seeds[0],
            num_leapfrog_steps=leapfrog_steps,
            probe_seeds=proposal_seeds,
            minimum_acceptance_probability=None
            if mean_accept is None
            else min(acceptance_probabilities),
            maximum_acceptance_probability=None
            if mean_accept is None
            else max(acceptance_probabilities),
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
    proposed_step_size_trace: Any
    consumed_step_size_trace: Any
    step_size_upper_bound: float
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
        proposed_step_trace = np.asarray(
            self.proposed_step_size_trace, dtype=float
        ).copy()
        consumed_step_trace = np.asarray(
            self.consumed_step_size_trace, dtype=float
        ).copy()
        step_upper_bound = float(self.step_size_upper_bound)
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
            or proposed_step_trace.shape != expected_trace_shape
            or consumed_step_trace.shape != expected_trace_shape
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
            proposed_step_trace,
            consumed_step_trace,
        )
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("operational warmup window arrays must be finite")
        if (
            not np.isfinite(step_upper_bound)
            or step_upper_bound <= 0.0
            or np.any(step_trace <= 0.0)
            or np.any(proposed_step_trace <= 0.0)
            or np.any(consumed_step_trace <= 0.0)
            or np.any(step_trace > step_upper_bound * (1.0 + 1.0e-12))
            or np.any(consumed_step_trace > step_upper_bound * (1.0 + 1.0e-12))
        ):
            raise ValueError("operational warmup step ceiling is invalid")
        if not np.allclose(
            step_trace,
            np.minimum(proposed_step_trace, step_upper_bound),
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError("bounded step trace does not match proposed step and ceiling")
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
        object.__setattr__(self, "proposed_step_size_trace", proposed_step_trace)
        object.__setattr__(self, "consumed_step_size_trace", consumed_step_trace)
        object.__setattr__(self, "step_size_upper_bound", step_upper_bound)
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
            "step_size_upper_bound": self.step_size_upper_bound,
            "maximum_bounded_next_step_size": float(np.max(self.step_size_trace)),
            "maximum_proposed_step_size": float(
                np.max(self.proposed_step_size_trace)
            ),
            "maximum_consumed_step_size": float(
                np.max(self.consumed_step_size_trace)
            ),
            "step_ceiling_hit_count": int(
                np.sum(
                    self.proposed_step_size_trace
                    > self.step_size_upper_bound * (1.0 + 1.0e-12)
                )
            ),
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


def _start_bank_optional_nonnegative_float(
    value: Any,
    *,
    name: str,
) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative when present")
    return result


def _start_bank_optional_count(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _strict_integer(value, name=name, minimum=0)


@dataclass(frozen=True)
class _StartBankScopeDiagnostic:
    """Bounded scalar evidence from one execution of the existing selector."""

    scope: str
    source_row_count: int
    dimension: int
    minimum_relative_separation: float
    sqrt_dimension_scale_component: float | None
    reference_coordinate_std_norm: float | None
    reference_scale: float | None
    absolute_tolerance: float | None
    finite_status: bool
    pre_endpoint_candidate_count: int | None
    endpoint_exclusion_count: int | None
    prior_eligible_exclusion_count: int | None
    final_greedy_eligible_count: int | None
    endpoint_distance_minimum: float | None
    endpoint_distance_maximum: float | None
    endpoint_distance_count_at_or_below_tolerance: int | None
    all_pair_distance_minimum: float | None
    all_pair_distance_maximum: float | None
    all_pair_distance_count_at_or_below_tolerance: int | None
    selection_attempted: bool
    selection_succeeded: bool
    selected_row_count: int
    failure_code: str

    def __post_init__(self) -> None:
        scope = str(self.scope)
        if scope not in _START_BANK_SCOPE_NAMES:
            raise ValueError("unsupported start-bank diagnostic scope")
        rows = _strict_integer(
            self.source_row_count,
            name="start-bank source_row_count",
            minimum=0,
        )
        dimension = _strict_integer(
            self.dimension,
            name="start-bank dimension",
            minimum=0,
        )
        separation = float(self.minimum_relative_separation)
        if not np.isfinite(separation) or separation <= 0.0:
            raise ValueError(
                "start-bank minimum_relative_separation must be finite and positive"
            )
        optional_float_names = (
            "sqrt_dimension_scale_component",
            "reference_coordinate_std_norm",
            "reference_scale",
            "absolute_tolerance",
            "endpoint_distance_minimum",
            "endpoint_distance_maximum",
            "all_pair_distance_minimum",
            "all_pair_distance_maximum",
        )
        optional_count_names = (
            "pre_endpoint_candidate_count",
            "endpoint_exclusion_count",
            "prior_eligible_exclusion_count",
            "final_greedy_eligible_count",
            "endpoint_distance_count_at_or_below_tolerance",
            "all_pair_distance_count_at_or_below_tolerance",
        )
        for name in optional_float_names:
            object.__setattr__(
                self,
                name,
                _start_bank_optional_nonnegative_float(
                    getattr(self, name),
                    name=f"start-bank {name}",
                ),
            )
        for name in optional_count_names:
            object.__setattr__(
                self,
                name,
                _start_bank_optional_count(
                    getattr(self, name),
                    name=f"start-bank {name}",
                ),
            )
        if not isinstance(self.finite_status, (bool, np.bool_)):
            raise TypeError("start-bank finite_status must be boolean")
        if not isinstance(self.selection_attempted, (bool, np.bool_)):
            raise TypeError("start-bank selection_attempted must be boolean")
        if not isinstance(self.selection_succeeded, (bool, np.bool_)):
            raise TypeError("start-bank selection_succeeded must be boolean")
        selected_count = _strict_integer(
            self.selected_row_count,
            name="start-bank selected_row_count",
            minimum=0,
        )
        failure_code = str(self.failure_code)
        if failure_code not in _START_BANK_FAILURE_CODES:
            raise ValueError("unsupported start-bank failure code")
        counts = (
            self.pre_endpoint_candidate_count,
            self.endpoint_exclusion_count,
            self.prior_eligible_exclusion_count,
            self.final_greedy_eligible_count,
        )
        if all(value is not None for value in counts) and counts[0] != sum(
            counts[1:]
        ):
            raise ValueError("start-bank greedy exclusion counts are inconsistent")
        for lower_name, upper_name in (
            ("endpoint_distance_minimum", "endpoint_distance_maximum"),
            ("all_pair_distance_minimum", "all_pair_distance_maximum"),
        ):
            lower = getattr(self, lower_name)
            upper = getattr(self, upper_name)
            if (lower is None) != (upper is None) or (
                lower is not None and upper is not None and lower > upper
            ):
                raise ValueError("start-bank distance summary is inconsistent")
        attempted = bool(self.selection_attempted)
        succeeded = bool(self.selection_succeeded)
        if succeeded and (
            not attempted or selected_count != 4 or failure_code != "none"
        ):
            raise ValueError("successful start-bank selection evidence is inconsistent")
        if failure_code == "none" and not succeeded:
            raise ValueError("failure-free start-bank evidence must report success")
        if failure_code == "insufficient_greedy_eligible" and (
            attempted or selected_count != 0
        ):
            raise ValueError("insufficient start-bank evidence cannot report selection")
        if failure_code == "post_selection_pairwise_failure" and (
            not attempted or succeeded or selected_count != 4
        ):
            raise ValueError("post-selection start-bank evidence is inconsistent")
        if failure_code.startswith("shadow_") and (
            scope != "shadow_all_windows"
            or attempted
            or succeeded
            or selected_count != 0
        ):
            raise ValueError("shadow assessment failure evidence is inconsistent")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "source_row_count", rows)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "minimum_relative_separation", separation)
        object.__setattr__(self, "finite_status", bool(self.finite_status))
        object.__setattr__(self, "selection_attempted", attempted)
        object.__setattr__(self, "selection_succeeded", succeeded)
        object.__setattr__(self, "selected_row_count", selected_count)
        object.__setattr__(self, "failure_code", failure_code)

    def public_payload(self) -> Mapping[str, Any]:
        payload = {
            "schema": _START_BANK_SCOPE_SCHEMA,
            "policy_id": _START_BANK_POLICY_ID,
            "scope": self.scope,
            "source_row_count": self.source_row_count,
            "dimension": self.dimension,
            "minimum_relative_separation": self.minimum_relative_separation,
            "sqrt_dimension_scale_component": self.sqrt_dimension_scale_component,
            "reference_coordinate_std_norm": self.reference_coordinate_std_norm,
            "reference_scale": self.reference_scale,
            "absolute_tolerance": self.absolute_tolerance,
            "finite_status": self.finite_status,
            "pre_endpoint_candidate_count": self.pre_endpoint_candidate_count,
            "endpoint_exclusion_count": self.endpoint_exclusion_count,
            "prior_eligible_exclusion_count": self.prior_eligible_exclusion_count,
            "final_greedy_eligible_count": self.final_greedy_eligible_count,
            "endpoint_distance_minimum": self.endpoint_distance_minimum,
            "endpoint_distance_maximum": self.endpoint_distance_maximum,
            "endpoint_distance_count_at_or_below_tolerance": (
                self.endpoint_distance_count_at_or_below_tolerance
            ),
            "all_pair_distance_minimum": self.all_pair_distance_minimum,
            "all_pair_distance_maximum": self.all_pair_distance_maximum,
            "all_pair_distance_count_at_or_below_tolerance": (
                self.all_pair_distance_count_at_or_below_tolerance
            ),
            "selection_attempted": self.selection_attempted,
            "selection_succeeded": self.selection_succeeded,
            "selected_row_count": self.selected_row_count,
            "failure_code": self.failure_code,
        }
        return _validate_start_bank_scope_payload(payload)


_START_BANK_SCOPE_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "policy_id",
        "scope",
        "source_row_count",
        "dimension",
        "minimum_relative_separation",
        "sqrt_dimension_scale_component",
        "reference_coordinate_std_norm",
        "reference_scale",
        "absolute_tolerance",
        "finite_status",
        "pre_endpoint_candidate_count",
        "endpoint_exclusion_count",
        "prior_eligible_exclusion_count",
        "final_greedy_eligible_count",
        "endpoint_distance_minimum",
        "endpoint_distance_maximum",
        "endpoint_distance_count_at_or_below_tolerance",
        "all_pair_distance_minimum",
        "all_pair_distance_maximum",
        "all_pair_distance_count_at_or_below_tolerance",
        "selection_attempted",
        "selection_succeeded",
        "selected_row_count",
        "failure_code",
    }
)


def _validate_start_bank_scope_payload(payload: Any) -> Mapping[str, Any]:
    if (
        not isinstance(payload, Mapping)
        or frozenset(payload) != _START_BANK_SCOPE_PAYLOAD_KEYS
    ):
        raise ValueError("start-bank scope diagnostic schema mismatch")
    if (
        payload.get("schema") != _START_BANK_SCOPE_SCHEMA
        or payload.get("policy_id") != _START_BANK_POLICY_ID
        or payload.get("scope") not in _START_BANK_SCOPE_NAMES
        or payload.get("failure_code") not in _START_BANK_FAILURE_CODES
    ):
        raise ValueError("start-bank scope diagnostic identity mismatch")
    field_names = tuple(_StartBankScopeDiagnostic.__dataclass_fields__)
    validated = _StartBankScopeDiagnostic(
        **{name: payload[name] for name in field_names}
    )
    return {
        "schema": _START_BANK_SCOPE_SCHEMA,
        "policy_id": _START_BANK_POLICY_ID,
        **{name: getattr(validated, name) for name in field_names},
    }


@dataclass(frozen=True)
class _StartBankAssessment:
    """Ephemeral selector state; serialized output is scalar-only."""

    canonical_states: Any
    reference_states: Any
    selected_row_indices: tuple[int, ...]
    diagnostic: _StartBankScopeDiagnostic

    def __post_init__(self) -> None:
        if type(self.diagnostic) is not _StartBankScopeDiagnostic:
            raise TypeError(
                "diagnostic must be a concrete start-bank scope diagnostic"
            )
        states = np.asarray(self.canonical_states, dtype=float).copy()
        reference = np.asarray(self.reference_states, dtype=float).copy()
        if states.ndim != 2 or reference.shape != states.shape:
            raise ValueError(
                "start-bank assessment arrays must be aligned rank-2 matrices"
            )
        indices = tuple(
            _strict_integer(index, name="start-bank selected row index", minimum=0)
            for index in self.selected_row_indices
        )
        if any(index >= states.shape[0] for index in indices):
            raise ValueError("start-bank assessment selected row index is out of range")
        if len(indices) != self.diagnostic.selected_row_count:
            raise ValueError("start-bank assessment selected row count is inconsistent")
        states.setflags(write=False)
        reference.setflags(write=False)
        object.__setattr__(self, "canonical_states", states)
        object.__setattr__(self, "reference_states", reference)
        object.__setattr__(self, "selected_row_indices", indices)

    def public_payload(self) -> Mapping[str, Any]:
        return self.diagnostic.public_payload()


@dataclass(frozen=True)
class _StartBankQualificationDiagnostic:
    """Validated scalar-only carrier shared by result and exception paths."""

    authoritative: _StartBankScopeDiagnostic
    shadow: _StartBankScopeDiagnostic

    def __post_init__(self) -> None:
        if (
            type(self.authoritative) is not _StartBankScopeDiagnostic
            or type(self.shadow) is not _StartBankScopeDiagnostic
        ):
            raise TypeError(
                "start-bank qualification scopes must use the concrete type"
            )
        if (
            self.authoritative.scope != "authoritative_final_window"
            or self.shadow.scope != "shadow_all_windows"
        ):
            raise ValueError("start-bank qualification scope roles are invalid")
        if self.interpretation not in _START_BANK_INTERPRETATIONS:
            raise ValueError("unsupported start-bank qualification interpretation")

    @property
    def interpretation(self) -> str:
        if self.authoritative.selection_succeeded:
            return "final_pass"
        if self.authoritative.failure_code == "post_selection_pairwise_failure":
            return "post_selection_invariant_failure"
        if self.shadow.selection_succeeded:
            return "final_fail_shadow_pass"
        return "both_fail"

    def public_payload(self) -> Mapping[str, Any]:
        payload = {
            "schema": _START_BANK_QUALIFICATION_SCHEMA,
            "policy_id": _START_BANK_POLICY_ID,
            "authoritative_scope": "authoritative_final_window",
            "shadow_decision_effect": False,
            "interpretation": self.interpretation,
            "scopes": {
                "authoritative_final_window": self.authoritative.public_payload(),
                "shadow_all_windows": self.shadow.public_payload(),
            },
        }
        return _validate_start_bank_qualification_payload(payload)


_START_BANK_QUALIFICATION_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "policy_id",
        "authoritative_scope",
        "shadow_decision_effect",
        "interpretation",
        "scopes",
    }
)


def _validate_start_bank_qualification_payload(payload: Any) -> Mapping[str, Any]:
    if (
        not isinstance(payload, Mapping)
        or frozenset(payload) != _START_BANK_QUALIFICATION_PAYLOAD_KEYS
        or payload.get("schema") != _START_BANK_QUALIFICATION_SCHEMA
        or payload.get("policy_id") != _START_BANK_POLICY_ID
        or payload.get("authoritative_scope") != "authoritative_final_window"
        or payload.get("shadow_decision_effect") is not False
        or payload.get("interpretation") not in _START_BANK_INTERPRETATIONS
    ):
        raise ValueError("start-bank qualification diagnostic schema mismatch")
    scopes = payload.get("scopes")
    if (
        not isinstance(scopes, Mapping)
        or frozenset(scopes) != _START_BANK_SCOPE_NAMES
    ):
        raise ValueError("start-bank qualification scope mapping is invalid")
    validated_scopes = {
        name: _validate_start_bank_scope_payload(scopes[name])
        for name in sorted(_START_BANK_SCOPE_NAMES)
    }
    if any(validated_scopes[name]["scope"] != name for name in validated_scopes):
        raise ValueError("start-bank qualification nested scope identity mismatch")
    field_names = tuple(_StartBankScopeDiagnostic.__dataclass_fields__)
    reconstructed = {
        name: _StartBankScopeDiagnostic(
            **{field: validated_scopes[name][field] for field in field_names}
        )
        for name in validated_scopes
    }
    expected_interpretation = _StartBankQualificationDiagnostic(
        authoritative=reconstructed["authoritative_final_window"],
        shadow=reconstructed["shadow_all_windows"],
    ).interpretation
    if payload.get("interpretation") != expected_interpretation:
        raise ValueError("start-bank qualification interpretation is inconsistent")
    return {**dict(payload), "scopes": validated_scopes}


def start_bank_qualification_payload_from_exception(
    exc: BaseException,
) -> Mapping[str, Any] | None:
    """Return only a concrete, schema-valid bounded selector diagnostic."""

    try:
        candidate = getattr(exc, _START_BANK_DIAGNOSTIC_ATTRIBUTE, None)
    except Exception:  # noqa: BLE001 - invalid carriers must remain ignorable.
        return None
    if type(candidate) is not _StartBankQualificationDiagnostic:
        return None
    try:
        return candidate.public_payload()
    except Exception:  # noqa: BLE001 - schema validation is fail-closed.
        return None


@dataclass(frozen=True)
class OperationalWindowedWarmupResult:
    config: WindowedMassAdaptationConfig
    initial_coordinate_signature: str
    final_kernel_state: KernelState
    reasonable_epsilon: ReasonableEpsilonResult
    windows: tuple[OperationalWarmupWindowResult, ...]
    private_start_bank_theta: Any
    start_bank_qualification: _StartBankQualificationDiagnostic
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
        if self.config.mass_policy == "fixed_identity" and any(
            item.window.update_mass or item.metric_decision is not None
            for item in windows
        ):
            raise ValueError(
                "fixed-identity operational warmup cannot assess or update mass"
            )
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
        qualification = self.start_bank_qualification
        if (
            type(qualification) is not _StartBankQualificationDiagnostic
            or not qualification.authoritative.selection_succeeded
        ):
            raise ValueError(
                "operational warmup requires a successful start-bank qualification"
            )
        qualification.public_payload()
        final_state = self.final_kernel_state
        if self.config.mass_policy == "fixed_identity" and (
            final_state.transform.signature != initial_signature
            or final_state.momentum_metric.signature != expected_metric
            or final_state.adaptation_generation != 0
        ):
            raise ValueError(
                "fixed-identity operational warmup changed coordinate or metric state"
            )
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
        object.__setattr__(self, "start_bank_qualification", qualification)
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
    def metric_adaptation_status(self) -> str:
        return (
            "metric_updated"
            if self.operational_metric_update_count > 0
            else "no_metric_update"
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
            "metric_adaptation_status": self.metric_adaptation_status,
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
    initial_step_size_upper_bound: float | None = None,
    initial_step_qualification_source: str | None = None,
    trajectory_policy: WarmupTrajectoryPolicy,
    config: WindowedMassAdaptationConfig,
    target_accept_prob: float,
    seed: tuple[int, int],
    target_scope: str,
    chain_execution_mode: str = "tf_function",
    jit_compile: bool = False,
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
    initial_bound = (
        None
        if initial_step_size_upper_bound is None
        else float(initial_step_size_upper_bound)
    )
    if initial_bound is not None and (
        not np.isfinite(initial_bound)
        or initial_bound <= 0.0
        or float(initial_step_size) > initial_bound * (1.0 + 1.0e-12)
    ):
        raise ValueError(
            "initial_step_size_upper_bound must be positive and bound the initial step"
        )
    qualification_source = (
        None
        if initial_step_qualification_source is None
        else str(initial_step_qualification_source)
    )
    if initial_bound is None and qualification_source is not None:
        raise ValueError("initial step qualification source requires an upper bound")
    if initial_bound is not None and not qualification_source:
        raise ValueError("initial step upper bound requires qualification provenance")
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
    reasonable = (
        ReasonableEpsilonResult(
            status="externally_qualified",
            selected_step_size=float(initial_step_size),
            attempts=(),
            qualification_source=qualification_source,
        )
        if initial_bound is not None
        else find_reasonable_epsilon(
            adapter=first_adapter,
            current_state=latent,
            initial_step_size=initial_step_size,
            seed=_seed(normalized_seed, -1, lane=1),
            num_leapfrog_steps=trajectory_policy.num_leapfrog_steps,
            momentum_probe_count=4,
            target_status_trace_policy=target_status_policy,
            jit_compile=jit_compile,
        )
    )
    if not reasonable.passed or reasonable.selected_step_size is None:
        raise ValueError("operational warmup reasonable epsilon search was inconclusive")
    epsilon = float(reasonable.selected_step_size)
    active_step_upper_bound = epsilon if initial_bound is None else initial_bound
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
        window_step_upper_bound = active_step_upper_bound
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
            def bounded_step_setter(
                kernel_results: Any, new_step_size: Any
            ) -> Any:
                from tensorflow_probability.python.internal import unnest

                bounded = tf.minimum(
                    tf.convert_to_tensor(new_step_size, dtype=current_latent.dtype),
                    tf.constant(window_step_upper_bound, dtype=current_latent.dtype),
                )
                return unnest.replace_innermost(
                    kernel_results,
                    step_size=bounded,
                )

            active_adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=base_kernel,
                num_adaptation_steps=config.warmup_steps - transition_count,
                target_accept_prob=tf.constant(target_accept, dtype=current_latent.dtype),
                step_size_setter_fn=bounded_step_setter,
                shrinkage_target=tf.constant(
                    min(10.0 * epsilon, window_step_upper_bound),
                    dtype=current_latent.dtype,
                ),
            )
            previous_kernel_results = active_adaptive_kernel.bootstrap_results(
                current_latent
            )
            previous_kernel_results = previous_kernel_results._replace(
                new_step_size=tf.minimum(
                    previous_kernel_results.new_step_size,
                    tf.constant(window_step_upper_bound, current_latent.dtype),
                )
            )
            runner_generation += 1

        def trace_fn(_state: Any, kernel_results: Any) -> Mapping[str, Any]:
            inner = kernel_results.inner_results
            trace = {
                "is_accepted": inner.is_accepted,
                "log_accept_ratio": inner.log_accept_ratio,
                "step_size": tf.minimum(
                    kernel_results.new_step_size,
                    tf.constant(window_step_upper_bound, current_latent.dtype),
                ),
                "proposed_step_size": kernel_results.new_step_size,
                "consumed_step_size": (
                    kernel_results.inner_results.accepted_results.step_size
                ),
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
                else tf.function(
                    run_window,
                    jit_compile=bool(jit_compile),
                    reduce_retracing=True,
                )
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
            final_kernel_results = final_kernel_results._replace(
                new_step_size=tf.minimum(
                    final_kernel_results.new_step_size,
                    tf.constant(window_step_upper_bound, current_latent.dtype),
                )
            )
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
        proposed_step_trace = np.asarray(
            trace["proposed_step_size"].numpy(), dtype=float
        )
        consumed_step_trace = np.asarray(
            trace["consumed_step_size"].numpy(), dtype=float
        )
        if (
            not np.all(np.isfinite(step_trace))
            or not np.all(np.isfinite(proposed_step_trace))
            or not np.all(np.isfinite(consumed_step_trace))
            or np.any(step_trace <= 0.0)
            or np.any(proposed_step_trace <= 0.0)
            or np.any(consumed_step_trace <= 0.0)
            or np.any(
                consumed_step_trace
                > window_step_upper_bound * (1.0 + 1.0e-12)
            )
        ):
            raise ValueError("operational warmup produced an invalid bounded step")
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
                latent_covariance = tf.convert_to_tensor(
                    metric_decision.covariance,
                    dtype=tf.float64,
                )
                active_factor = tf.convert_to_tensor(
                    active_transform.factor,
                    dtype=tf.float64,
                )
                canonical_covariance = tf.matmul(
                    tf.matmul(active_factor, latent_covariance),
                    active_factor,
                    transpose_b=True,
                )
                canonical_covariance = 0.5 * (
                    canonical_covariance + tf.transpose(canonical_covariance)
                )
                canonical_state_tensor = tf.reshape(
                    tf.convert_to_tensor(canonical_draws, dtype=tf.float64),
                    (-1, active_transform.dimension),
                )
                canonical_mean = tf.reduce_mean(
                    canonical_state_tensor,
                    axis=0,
                )
                try:
                    estimate = PositionCovarianceEstimate(
                        center=canonical_mean.numpy(),
                        covariance=canonical_covariance.numpy(),
                        source_coordinate_signature=active_transform.signature,
                        estimator_family=str(metric_decision.estimator_family),
                        state_count=int(canonical_state_tensor.shape[0]),
                        effective_rank=int(
                            tf.linalg.matrix_rank(latent_covariance).numpy()
                        ),
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
                except (
                    TypeError,
                    ValueError,
                    tf.errors.InvalidArgumentError,
                ) as exc:
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
                        num_leapfrog_steps=trajectory_policy.num_leapfrog_steps,
                        momentum_probe_count=4,
                        target_status_trace_policy=target_status_policy,
                        jit_compile=jit_compile,
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
                active_step_upper_bound = epsilon_end
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
                proposed_step_size_trace=proposed_step_trace,
                consumed_step_size_trace=consumed_step_trace,
                step_size_upper_bound=window_step_upper_bound,
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
    authoritative_assessment = _assess_private_start_bank(
        history,
        reference_transform=kernel_state.transform,
        scope="authoritative_final_window",
    )
    shadow_diagnostic = _best_effort_shadow_start_bank_scope(
        canonical_history,
        reference_transform=kernel_state.transform,
        minimum_relative_separation=(
            authoritative_assessment.diagnostic.minimum_relative_separation
        ),
    )
    start_bank_qualification = _StartBankQualificationDiagnostic(
        authoritative=authoritative_assessment.diagnostic,
        shadow=shadow_diagnostic,
    )
    bank = _materialize_private_start_bank(
        authoritative_assessment,
        qualification=start_bank_qualification,
    )
    result = OperationalWindowedWarmupResult(
        config=config,
        initial_coordinate_signature=initial_transform.signature,
        final_kernel_state=kernel_state,
        reasonable_epsilon=reasonable,
        windows=tuple(results),
        private_start_bank_theta=bank,
        start_bank_qualification=start_bank_qualification,
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

    assessment = _assess_private_start_bank(
        canonical_states,
        reference_transform=reference_transform,
        minimum_relative_separation=minimum_relative_separation,
        scope="authoritative_final_window",
    )
    return _materialize_private_start_bank(assessment)


def _assess_private_start_bank(
    canonical_states: Any,
    *,
    reference_transform: AffineCoordinateTransform | None = None,
    minimum_relative_separation: float = 1.0e-4,
    scope: str,
) -> _StartBankAssessment:
    """Run the existing selector calculation without materializing its bank."""

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
        else np.asarray(
            reference_transform.theta_to_latent(states).numpy(),
            dtype=float,
        )
    )
    return _assess_prepared_start_bank(
        states,
        reference,
        minimum_relative_separation=separation,
        scope=scope,
    )


def _finite_nonnegative_or_none(value: Any) -> float | None:
    result = float(value)
    return result if np.isfinite(result) and result >= 0.0 else None


def _start_bank_distance_summary(
    distances: Any,
    *,
    tolerance: float,
) -> tuple[float | None, float | None, int]:
    values = np.asarray(distances, dtype=float).reshape(-1)
    count = int(np.sum(values <= tolerance))
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None, None, count
    return float(np.min(values)), float(np.max(values)), count


def _assess_prepared_start_bank(
    states: np.ndarray,
    reference: np.ndarray,
    *,
    minimum_relative_separation: float,
    scope: str,
) -> _StartBankAssessment:
    """Preserve the selector's endpoint-first chronological greedy ordering."""

    sqrt_dimension = float(np.sqrt(states.shape[1]))
    reference_std_norm = float(np.linalg.norm(np.std(reference, axis=0)))
    reference_scale = max(
        sqrt_dimension,
        reference_std_norm,
    )
    tolerance = minimum_relative_separation * reference_scale
    endpoint_index = states.shape[0] - 1
    eligible_indices: list[int] = []
    endpoint_exclusion_count = 0
    prior_eligible_exclusion_count = 0
    for index in range(endpoint_index):
        if np.linalg.norm(reference[index] - reference[endpoint_index]) <= tolerance:
            endpoint_exclusion_count += 1
            continue
        if all(
            np.linalg.norm(reference[index] - reference[existing]) > tolerance
            for existing in eligible_indices
        ):
            eligible_indices.append(index)
        else:
            prior_eligible_exclusion_count += 1

    endpoint_distances = np.linalg.norm(
        reference[:endpoint_index] - reference[endpoint_index],
        axis=-1,
    )
    endpoint_minimum, endpoint_maximum, endpoint_close_count = (
        _start_bank_distance_summary(endpoint_distances, tolerance=tolerance)
    )
    all_pair_matrix = np.linalg.norm(
        reference[:, None, :] - reference[None, :, :],
        axis=-1,
    )
    all_pair_distances = all_pair_matrix[
        np.triu_indices(states.shape[0], k=1)
    ]
    all_pair_minimum, all_pair_maximum, all_pair_close_count = (
        _start_bank_distance_summary(all_pair_distances, tolerance=tolerance)
    )

    bank_indices: list[int] = []
    selection_attempted = len(eligible_indices) >= 3
    selection_succeeded = False
    failure_code = "insufficient_greedy_eligible"
    if selection_attempted:
        selected = np.linspace(0, len(eligible_indices) - 1, 3, dtype=int)
        bank_indices = [eligible_indices[index] for index in selected] + [
            endpoint_index
        ]
        reference_bank = reference[bank_indices]
        pairwise = np.linalg.norm(
            reference_bank[:, None, :] - reference_bank[None, :, :], axis=-1
        )
        if np.any(pairwise[np.triu_indices(4, k=1)] <= tolerance):
            failure_code = "post_selection_pairwise_failure"
        else:
            selection_succeeded = True
            failure_code = "none"

    diagnostic = _StartBankScopeDiagnostic(
        scope=scope,
        source_row_count=int(states.shape[0]),
        dimension=int(states.shape[1]),
        minimum_relative_separation=minimum_relative_separation,
        sqrt_dimension_scale_component=_finite_nonnegative_or_none(sqrt_dimension),
        reference_coordinate_std_norm=_finite_nonnegative_or_none(reference_std_norm),
        reference_scale=_finite_nonnegative_or_none(reference_scale),
        absolute_tolerance=_finite_nonnegative_or_none(tolerance),
        finite_status=bool(
            np.all(np.isfinite(states)) and np.all(np.isfinite(reference))
        ),
        pre_endpoint_candidate_count=endpoint_index,
        endpoint_exclusion_count=endpoint_exclusion_count,
        prior_eligible_exclusion_count=prior_eligible_exclusion_count,
        final_greedy_eligible_count=len(eligible_indices),
        endpoint_distance_minimum=endpoint_minimum,
        endpoint_distance_maximum=endpoint_maximum,
        endpoint_distance_count_at_or_below_tolerance=endpoint_close_count,
        all_pair_distance_minimum=all_pair_minimum,
        all_pair_distance_maximum=all_pair_maximum,
        all_pair_distance_count_at_or_below_tolerance=all_pair_close_count,
        selection_attempted=selection_attempted,
        selection_succeeded=selection_succeeded,
        selected_row_count=len(bank_indices),
        failure_code=failure_code,
    )
    return _StartBankAssessment(
        canonical_states=states,
        reference_states=reference,
        selected_row_indices=tuple(bank_indices),
        diagnostic=diagnostic,
    )


def _shadow_start_bank_failure_diagnostic(
    states: np.ndarray | None,
    *,
    minimum_relative_separation: float,
    failure_code: str,
    finite_status_override: bool | None = None,
) -> _StartBankScopeDiagnostic:
    rows = 0
    dimension = 0
    finite_status = False
    sqrt_dimension: float | None = None
    if states is not None:
        finite_status = bool(np.all(np.isfinite(states)))
        if states.ndim >= 1:
            rows = int(states.shape[0])
        if states.ndim == 2:
            dimension = int(states.shape[1])
            sqrt_dimension = float(np.sqrt(dimension))
    if finite_status_override is not None:
        finite_status = bool(finite_status_override)
    return _StartBankScopeDiagnostic(
        scope="shadow_all_windows",
        source_row_count=rows,
        dimension=dimension,
        minimum_relative_separation=minimum_relative_separation,
        sqrt_dimension_scale_component=sqrt_dimension,
        reference_coordinate_std_norm=None,
        reference_scale=None,
        absolute_tolerance=None,
        finite_status=finite_status,
        pre_endpoint_candidate_count=None,
        endpoint_exclusion_count=None,
        prior_eligible_exclusion_count=None,
        final_greedy_eligible_count=None,
        endpoint_distance_minimum=None,
        endpoint_distance_maximum=None,
        endpoint_distance_count_at_or_below_tolerance=None,
        all_pair_distance_minimum=None,
        all_pair_distance_maximum=None,
        all_pair_distance_count_at_or_below_tolerance=None,
        selection_attempted=False,
        selection_succeeded=False,
        selected_row_count=0,
        failure_code=failure_code,
    )


def _best_effort_shadow_start_bank_scope(
    canonical_states: Any,
    *,
    reference_transform: AffineCoordinateTransform | None,
    minimum_relative_separation: float,
) -> _StartBankScopeDiagnostic:
    """Assess accumulated history without allowing shadow errors to escape."""

    separation = float(minimum_relative_separation)
    try:
        states = np.asarray(canonical_states, dtype=float)
    except Exception:  # noqa: BLE001 - fixed bounded shadow failure code.
        return _shadow_start_bank_failure_diagnostic(
            None,
            minimum_relative_separation=separation,
            failure_code="shadow_input_conversion_failure",
        )
    if states.ndim != 2 or states.shape[0] < 4:
        return _shadow_start_bank_failure_diagnostic(
            states,
            minimum_relative_separation=separation,
            failure_code="shadow_invalid_shape",
        )
    if not np.all(np.isfinite(states)):
        return _shadow_start_bank_failure_diagnostic(
            states,
            minimum_relative_separation=separation,
            failure_code="shadow_nonfinite_source",
        )
    try:
        reference = (
            states
            if reference_transform is None
            else np.asarray(
                reference_transform.theta_to_latent(states).numpy(),
                dtype=float,
            )
        )
    except Exception:  # noqa: BLE001 - fixed bounded shadow failure code.
        return _shadow_start_bank_failure_diagnostic(
            states,
            minimum_relative_separation=separation,
            failure_code="shadow_reference_conversion_failure",
            finite_status_override=False,
        )
    if reference.shape != states.shape:
        return _shadow_start_bank_failure_diagnostic(
            states,
            minimum_relative_separation=separation,
            failure_code="shadow_reference_conversion_failure",
            finite_status_override=False,
        )
    if not np.all(np.isfinite(reference)):
        return _shadow_start_bank_failure_diagnostic(
            states,
            minimum_relative_separation=separation,
            failure_code="shadow_nonfinite_reference",
            finite_status_override=False,
        )
    try:
        return _assess_prepared_start_bank(
            states,
            reference,
            minimum_relative_separation=separation,
            scope="shadow_all_windows",
        ).diagnostic
    except Exception:  # noqa: BLE001 - shadow assessment is decision-inert.
        return _shadow_start_bank_failure_diagnostic(
            states,
            minimum_relative_separation=separation,
            failure_code="shadow_assessment_failure",
        )


def _materialize_private_start_bank(
    assessment: _StartBankAssessment,
    *,
    qualification: _StartBankQualificationDiagnostic | None = None,
) -> np.ndarray:
    if type(assessment) is not _StartBankAssessment:
        raise TypeError("assessment must be a concrete start-bank assessment")
    if (
        qualification is not None
        and type(qualification) is not _StartBankQualificationDiagnostic
    ):
        raise TypeError("qualification must be a concrete start-bank diagnostic")
    if not assessment.diagnostic.selection_succeeded:
        error = ValueError(
            "operational warmup start bank is not sufficiently dispersed"
        )
        if qualification is not None:
            setattr(error, _START_BANK_DIAGNOSTIC_ATTRIBUTE, qualification)
        raise error
    bank = assessment.canonical_states[
        list(assessment.selected_row_indices)
    ].astype(float, copy=True)
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
    import tensorflow as tf

    if base_transform.dimension != final_transform.dimension:
        raise ValueError("base and final transform dimensions must match")
    base_factor = tf.convert_to_tensor(base_transform.factor, dtype=tf.float64)
    final_factor = tf.convert_to_tensor(final_transform.factor, dtype=tf.float64)
    center_delta = tf.convert_to_tensor(
        final_transform.center - base_transform.center,
        dtype=tf.float64,
    )
    nested_center = tf.linalg.triangular_solve(
        base_factor,
        center_delta[:, None],
        lower=True,
    )[:, 0]
    nested_factor = tf.linalg.triangular_solve(
        base_factor,
        final_factor,
        lower=True,
    )
    nested_covariance = tf.matmul(nested_factor, nested_factor, transpose_b=True)
    tf.debugging.assert_all_finite(
        nested_covariance,
        "operational compatibility covariance must be finite",
    )
    artifact = PrecomputedMassArtifact(
        position=nested_center.numpy(),
        covariance=nested_covariance.numpy(),
        factor=nested_factor.numpy(),
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
    probe = tf.stack(
        [
            tf.zeros((base_transform.dimension,), dtype=tf.float64),
            tf.linspace(
                tf.constant(-0.3, tf.float64),
                tf.constant(0.4, tf.float64),
                base_transform.dimension,
            ),
        ]
    )
    nested_theta = base_transform.latent_to_theta(
        artifact.build_latent_transform().latent_to_position(probe)
    )
    direct_theta = final_transform.latent_to_theta(probe)
    try:
        tf.debugging.assert_near(
            nested_theta,
            direct_theta,
            rtol=1.0e-10,
            atol=1.0e-10,
            message="operational compatibility composition failed",
        )
    except tf.errors.InvalidArgumentError as exc:
        raise ValueError("operational compatibility composition failed") from exc
    return artifact
