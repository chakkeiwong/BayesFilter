"""TensorFlow/TFP-only fixed-kernel HMC for reviewed value/score adapters."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.batched_value_score import reviewed_value_score_target_fn
from bayesfilter.inference.posterior_adapter import value_score_capability


_REVIEWED_AUTHORITIES = frozenset(
    {"graph_native", "reviewed_gradient_tape_xla_exception"}
)
_NONCLAIMS = (
    "fixed-kernel TFP engineering runner only",
    "no HMC tuning claim",
    "no posterior convergence claim",
    "no posterior validity claim",
    "no performance superiority claim",
)


@dataclass(frozen=True)
class NativeTFPFixedKernelHMCConfig:
    """Static fixed-kernel contract with no adaptation or XLA escape hatch."""

    num_results: int
    num_burnin_steps: int
    step_size: float
    num_leapfrog_steps: int
    seed: tuple[int, int]
    target_scope: str

    def __post_init__(self) -> None:
        for name in ("num_results", "num_burnin_steps", "num_leapfrog_steps"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        step_size = float(self.step_size)
        if not math.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")
        object.__setattr__(self, "step_size", step_size)
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain exactly two integers")
        object.__setattr__(self, "seed", seed)
        scope = str(self.target_scope)
        if not scope:
            raise ValueError("target_scope must be non-empty")
        object.__setattr__(self, "target_scope", scope)

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "num_results": self.num_results,
            "num_burnin_steps": self.num_burnin_steps,
            "step_size": self.step_size,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "seed": self.seed,
            "target_scope": self.target_scope,
            "target_status_trace_policy": "per_chain_step_required",
            "runtime": "tfp.mcmc.sample_chain",
            "kernel": "tfp.mcmc.HamiltonianMonteCarlo",
            "adaptation_policy": "fixed_kernel_no_adaptation",
            "chain_execution_mode": "tf_function",
            "use_xla": False,
        }


@dataclass(frozen=True)
class NativeTFPHMCRunResult:
    """Tensor-valued samples, traces, and bounded engineering diagnostics."""

    samples: tf.Tensor
    trace: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any]


def run_native_tfp_fixed_kernel_hmc(
    adapter: Any,
    initial_state: Any,
    config: NativeTFPFixedKernelHMCConfig,
) -> NativeTFPHMCRunResult:
    """Run one fixed TensorFlow Probability HMC chain in a stable graph."""

    if not isinstance(config, NativeTFPFixedKernelHMCConfig):
        raise TypeError("config must be NativeTFPFixedKernelHMCConfig")
    capability = value_score_capability(adapter)
    if capability.value_score_authority not in _REVIEWED_AUTHORITIES:
        raise ValueError(
            "native TFP HMC requires reviewed graph value/score authority; got "
            f"{capability.value_score_authority!r}"
        )
    if capability.target_scope != config.target_scope:
        raise ValueError("value/score target_scope mismatch")
    if not callable(getattr(adapter, "target_status_telemetry", None)):
        raise TypeError(
            "native TFP HMC requires adapter target_status_telemetry"
        )

    state = tf.cast(tf.convert_to_tensor(initial_state), tf.float64)
    if state.shape.rank is None or any(dim is None for dim in state.shape):
        raise ValueError("initial_state must have a fully static shape")
    target_log_prob = reviewed_value_score_target_fn(adapter, dtype=state.dtype)
    trace_fn = _standard_trace_fn(adapter)
    tfm = tfp.mcmc
    kernel = tfm.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=tf.constant(config.step_size, dtype=state.dtype),
        num_leapfrog_steps=config.num_leapfrog_steps,
    )

    @tf.function(input_signature=(), autograph=False, reduce_retracing=True)
    def run_chain() -> tuple[tf.Tensor, Mapping[str, Any]]:
        return tfm.sample_chain(
            num_results=config.num_results,
            num_burnin_steps=config.num_burnin_steps,
            current_state=state,
            kernel=kernel,
            trace_fn=trace_fn,
            seed=tf.constant(config.seed, dtype=tf.int32),
        )

    started = time.perf_counter()
    samples, trace = run_chain()
    sample_chain_call_s = time.perf_counter() - started
    diagnostics = _diagnostics(samples, trace)
    adapter_signature = _adapter_signature(adapter)
    metadata = {
        "runtime": "tfp.mcmc.sample_chain",
        "kernel": "tfp.mcmc.HamiltonianMonteCarlo",
        "implementation_module": "bayesfilter.inference.native_tfp_hmc",
        "implementation_backend": "tensorflow_tensorflow_probability_only",
        "adaptation_policy": "fixed_kernel_no_adaptation",
        "chain_execution_mode": "tf_function",
        "tf_function_input_signature": (),
        "use_xla": False,
        "jit_compile": False,
        "sample_chain_invocation_count": 1,
        "sample_chain_call_s": sample_chain_call_s,
        "sample_chain_timing_role": "explanatory_only_compile_plus_execute",
        "initial_state_shape": tuple(int(dim) for dim in state.shape),
        "initial_state_dtype": state.dtype.name,
        "value_score_authority": capability.value_score_authority,
        "target_scope": capability.target_scope,
        "adapter_signature": adapter_signature,
        "program_signature": _program_signature(
            {
                "adapter_signature": adapter_signature,
                "capability": {
                    "value_score_authority": capability.value_score_authority,
                    "runtime_backend": capability.runtime_backend,
                    "target_scope": capability.target_scope,
                    "xla_hmc_ready": capability.xla_hmc_ready,
                    "full_chain_xla_diagnostic_ready": (
                        capability.full_chain_xla_diagnostic_ready
                    ),
                },
                "config": config.signature_payload(),
                "initial_state_shape": tuple(int(dim) for dim in state.shape),
                "initial_state_dtype": state.dtype.name,
            }
        ),
        "trace_unavailability": (
            {}
            if "divergence" in trace
            else {
                "divergence": (
                    "native boolean divergence field not exposed by "
                    "TensorFlow Probability HMC kernel results"
                )
            }
        ),
        "nonclaims": _NONCLAIMS,
    }
    return NativeTFPHMCRunResult(
        samples=samples,
        trace=trace,
        diagnostics=diagnostics,
        metadata=metadata,
    )


def _standard_trace_fn(adapter: Any):
    def trace_fn(state: Any, kernel_results: Any) -> Mapping[str, Any]:
        trace = {
            "is_accepted": kernel_results.is_accepted,
            "log_accept_ratio": kernel_results.log_accept_ratio,
            "target_log_prob": kernel_results.accepted_results.target_log_prob,
            "proposed_target_log_prob": (
                kernel_results.proposed_results.target_log_prob
            ),
        }
        correction = getattr(
            kernel_results.proposed_results,
            "log_acceptance_correction",
            None,
        )
        if correction is not None:
            trace["log_acceptance_correction"] = correction
        divergence = _native_divergence_tensor(kernel_results)
        if divergence is not None:
            trace["divergence"] = divergence
        trace["target_status_telemetry"] = adapter.target_status_telemetry(state)
        return trace

    return trace_fn


def _native_divergence_tensor(kernel_results: Any) -> tf.Tensor | None:
    result_objects = (
        kernel_results,
        getattr(kernel_results, "proposed_results", None),
        getattr(kernel_results, "accepted_results", None),
    )
    for result_object in result_objects:
        if result_object is None:
            continue
        for field_name in (
            "is_divergent",
            "has_divergence",
            "divergence",
            "divergences",
        ):
            value = getattr(result_object, field_name, None)
            if value is None:
                continue
            try:
                tensor = tf.convert_to_tensor(value)
            except (TypeError, ValueError):
                continue
            if tensor.dtype == tf.bool:
                return tensor
    return None


def _diagnostics(samples: tf.Tensor, trace: Mapping[str, Any]) -> Mapping[str, Any]:
    finite_by_sample = tf.reduce_all(tf.math.is_finite(samples), axis=-1)
    accepted = tf.cast(trace["is_accepted"], tf.float64)
    diagnostics: dict[str, Any] = {
        "finite_sample_count": tf.reduce_sum(tf.cast(finite_by_sample, tf.int32)),
        "nonfinite_sample_count": tf.reduce_sum(
            tf.cast(tf.logical_not(finite_by_sample), tf.int32)
        ),
        "sample_shape": tuple(int(dim) for dim in samples.shape),
        "trace_policy": "standard",
        "acceptance_rate": tf.reduce_mean(accepted),
        "native_divergence_status": "not_exposed_by_kernel",
        "divergence_status": "not_exposed_by_kernel",
        "divergence_count": None,
        "divergence_source": None,
        "hmc_health_diagnostics": _health_diagnostics(trace),
        "descriptive_movement_telemetry": _movement_telemetry(samples, trace),
        "nonclaims": (
            "finite fixed-kernel diagnostics only",
            "native divergence unavailability is not zero divergences",
            "no sampler convergence claim",
        ),
    }
    if "divergence" in trace:
        divergence = tf.cast(trace["divergence"], tf.bool)
        diagnostics.update(
            {
                "native_divergence_status": "available",
                "divergence_status": "available",
                "divergence_count": tf.reduce_sum(tf.cast(divergence, tf.int32)),
                "divergence_source": "native_boolean_tfp_kernel_result",
            }
        )
    if "target_log_prob" in trace:
        diagnostics["min_target_log_prob"] = tf.reduce_min(trace["target_log_prob"])
        diagnostics["max_target_log_prob"] = tf.reduce_max(trace["target_log_prob"])
    if "target_status_telemetry" in trace:
        diagnostics["target_status_telemetry"] = _target_status_diagnostics(
            trace["target_status_telemetry"]
        )
    return diagnostics


def _health_diagnostics(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    accepted = tf.cast(trace["is_accepted"], tf.float64)
    health: dict[str, Any] = {
        "diagnostic_role": "hmc_health_diagnostics_not_native_divergence",
        "acceptance_rate": tf.reduce_mean(accepted),
        "acceptance_finite": tf.reduce_all(tf.math.is_finite(accepted)),
        "nonclaims": (
            "acceptance/log-accept/target-log-prob are not native divergence telemetry",
            "no sampler convergence claim",
        ),
    }
    for name in ("log_accept_ratio", "log_acceptance_correction"):
        if name not in trace:
            health[name] = {"available": False}
            continue
        values = tf.cast(trace[name], tf.float64)
        finite = tf.math.is_finite(values)
        finite_values = tf.boolean_mask(values, finite)
        health[name] = {
            "available": True,
            "finite": tf.reduce_all(finite),
            "finite_count": tf.reduce_sum(tf.cast(finite, tf.int32)),
            "nonfinite_count": tf.reduce_sum(
                tf.cast(tf.logical_not(finite), tf.int32)
            ),
            "max_abs_finite": tf.cond(
                tf.size(finite_values) > 0,
                lambda: tf.reduce_max(tf.abs(finite_values)),
                lambda: tf.constant(float("nan"), tf.float64),
            ),
        }
    for name in ("target_log_prob", "proposed_target_log_prob"):
        values = tf.cast(trace[name], tf.float64)
        health[name] = {
            "available": True,
            "finite": tf.reduce_all(tf.math.is_finite(values)),
            "min": tf.reduce_min(values),
            "max": tf.reduce_max(values),
        }
    return health


def _movement_telemetry(
    samples: tf.Tensor,
    trace: Mapping[str, Any],
) -> Mapping[str, Any]:
    values = tf.cast(samples, tf.float64)
    log_accept = tf.cast(trace["log_accept_ratio"], tf.float64)
    accepted = tf.cast(trace["is_accepted"], tf.float64)
    finite = tf.math.is_finite(log_accept)
    acceptance_probability = tf.where(
        finite,
        tf.exp(tf.minimum(log_accept, tf.constant(0.0, tf.float64))),
        tf.fill(tf.shape(log_accept), tf.constant(float("nan"), tf.float64)),
    )
    displacement = values[1:] - values[:-1]
    return {
        "diagnostic_role": "descriptive_only_not_promotion_or_convergence",
        "mean_acceptance_probability": tf.reduce_mean(acceptance_probability),
        "binary_acceptance_rate": tf.reduce_mean(accepted),
        "log_accept_ratio_finite_count": tf.reduce_sum(tf.cast(finite, tf.int32)),
        "log_accept_ratio_nonfinite_count": tf.reduce_sum(
            tf.cast(tf.logical_not(finite), tf.int32)
        ),
        "max_abs_log_accept_ratio": tf.reduce_max(tf.abs(log_accept)),
        "mean_displacement_l2": tf.cond(
            tf.shape(values, out_type=tf.int32)[0] > 1,
            lambda: tf.reduce_mean(tf.linalg.norm(displacement, axis=-1)),
            lambda: tf.constant(float("nan"), tf.float64),
        ),
        "nonclaims": (
            "descriptive fixed-kernel telemetry only",
            "absolute log acceptance is not Hamiltonian energy error",
            "no posterior convergence claim",
        ),
    }


def _target_status_diagnostics(telemetry: Mapping[str, Any]) -> Mapping[str, Any]:
    required = (
        "status_code",
        "valid_pre_regularized_score",
        "floor_count_value",
        "min_innovation_eigenvalue",
        "innovation_condition_estimate",
    )
    missing = tuple(name for name in required if name not in telemetry)
    if missing:
        raise ValueError(
            "target_status_telemetry missing required fields: " + ", ".join(missing)
        )
    status = tf.cast(telemetry["status_code"], tf.int32)
    valid = tf.cast(telemetry["valid_pre_regularized_score"], tf.bool)
    floors = tf.cast(telemetry["floor_count_value"], tf.int32)
    min_eigen = tf.cast(telemetry["min_innovation_eigenvalue"], tf.float64)
    condition = tf.cast(telemetry["innovation_condition_estimate"], tf.float64)
    nonvalid = tf.logical_or(tf.not_equal(status, 0), tf.logical_not(valid))
    return {
        "trace_entry_count": tf.size(status),
        "status_nonvalid_count": tf.reduce_sum(tf.cast(nonvalid, tf.int32)),
        "all_status_valid": tf.reduce_all(tf.logical_not(nonvalid)),
        "floor_count_total": tf.reduce_sum(floors),
        "max_floor_count_value": tf.reduce_max(floors),
        "min_min_innovation_eigenvalue": tf.reduce_min(min_eigen),
        "max_innovation_condition_estimate": tf.reduce_max(condition),
        "telemetry_failure_veto": tf.reduce_any(nonvalid),
    }


def _adapter_signature(adapter: Any) -> str:
    explicit = getattr(adapter, "adapter_signature", None)
    if explicit is None:
        raise ValueError("native TFP HMC requires an explicit adapter_signature")
    signature = str(explicit() if callable(explicit) else explicit)
    if not signature:
        raise ValueError("adapter_signature must be non-empty")
    return signature


def _program_signature(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(
        _json_normalize(payload),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def _json_normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_normalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"program signature metadata is not JSON-safe: {type(value)!r}")


__all__ = [
    "NativeTFPFixedKernelHMCConfig",
    "NativeTFPHMCRunResult",
    "run_native_tfp_fixed_kernel_hmc",
]
