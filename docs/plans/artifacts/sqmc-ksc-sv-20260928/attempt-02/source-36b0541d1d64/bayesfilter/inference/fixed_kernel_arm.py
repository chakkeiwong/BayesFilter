"""Direct fixed-kernel HMC arms for bounded mechanism diagnostics.

This module deliberately wraps the existing ``FixedSizeHMCChunkRunner`` rather
than reimplementing HMC.  An arm is a discarded, fixed Metropolis-corrected
chain with an explicit starting state, step size, leapfrog count, seed, and
transition count.  The returned diagnostic is a JSON-safe minimum coordinate
ESS summary; raw draws are consumed in memory and are not part of the result.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Mapping

from bayesfilter.inference.hmc import (
    FixedSizeHMCChunkConfig,
    FixedSizeHMCChunkRunResult,
    build_fixed_size_hmc_chunk_runner,
)


@dataclass(frozen=True)
class FixedKernelArmConfig:
    """Immutable arm controls; adaptation is intentionally not representable."""

    label: str
    transition_count: int
    step_size: float
    num_leapfrog_steps: int
    seed: tuple[int, int]
    target_scope: str
    use_xla: bool = True
    chain_execution_mode: str = "tf_function"
    ess_checkpoints: tuple[int, ...] = ()
    ess_threshold: float = 9.0

    def __post_init__(self) -> None:
        label = str(self.label).strip()
        if not label:
            raise ValueError("label must be non-empty")
        count = int(self.transition_count)
        if count <= 0:
            raise ValueError("transition_count must be positive")
        step = float(self.step_size)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be positive and finite")
        leapfrog = int(self.num_leapfrog_steps)
        if leapfrog <= 0:
            raise ValueError("num_leapfrog_steps must be positive")
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain exactly two integers")
        scope = str(self.target_scope).strip()
        if not scope:
            raise ValueError("target_scope must be non-empty")
        mode = str(self.chain_execution_mode)
        if mode not in {"tf_function", "eager"}:
            raise ValueError("chain_execution_mode must be tf_function or eager")
        checkpoints = tuple(int(item) for item in self.ess_checkpoints)
        if checkpoints and (
            any(item < 4 or item > count for item in checkpoints)
            or tuple(sorted(set(checkpoints))) != checkpoints
        ):
            raise ValueError(
                "ess_checkpoints must be strictly increasing unique values "
                "between four and transition_count"
            )
        ess_threshold = float(self.ess_threshold)
        if not math.isfinite(ess_threshold) or ess_threshold <= 0.0:
            raise ValueError("ess_threshold must be positive and finite")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "transition_count", count)
        object.__setattr__(self, "step_size", step)
        object.__setattr__(self, "num_leapfrog_steps", leapfrog)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "target_scope", scope)
        object.__setattr__(self, "chain_execution_mode", mode)
        object.__setattr__(self, "ess_checkpoints", checkpoints)
        object.__setattr__(self, "ess_threshold", ess_threshold)

    def payload(self) -> Mapping[str, Any]:
        payload = {
            "label": self.label,
            "transition_count": self.transition_count,
            "step_size": self.step_size,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "seed": self.seed,
            "target_scope": self.target_scope,
            "use_xla": bool(self.use_xla),
            "chain_execution_mode": self.chain_execution_mode,
            "adaptation_policy": "fixed_kernel_no_adaptation",
            "num_burnin_steps": 0,
        }
        if self.ess_checkpoints:
            payload["ess_checkpoints"] = self.ess_checkpoints
            payload["ess_threshold"] = self.ess_threshold
        return payload


@dataclass(frozen=True)
class FixedKernelArmResult:
    """Public-safe arm result; samples and terminal state are discarded."""

    config: FixedKernelArmConfig
    diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def payload(self) -> Mapping[str, Any]:
        return {
            "arm": self.config.payload(),
            "diagnostics": dict(self.diagnostics),
            "metadata": dict(self.metadata),
            "raw_samples_publicized": False,
            "final_state_publicized": False,
            "adaptation_policy": "fixed_kernel_no_adaptation",
            "nonclaims": (
                "discarded fixed-kernel mechanism diagnostic only",
                "no posterior convergence claim",
                "no sampler ranking claim",
                "no default-readiness claim",
            ),
        }


def _valid_samples(result: FixedSizeHMCChunkRunResult) -> Any:
    import tensorflow as tf

    samples = tf.cast(tf.convert_to_tensor(result.samples), tf.float64)
    mask = tf.cast(tf.convert_to_tensor(result.valid_mask), tf.bool)
    if samples.shape.rank not in {2, 3} or any(dim is None for dim in samples.shape):
        raise ValueError(
            "fixed-kernel samples must have shape [draw, coordinate] or "
            "[draw, chain, coordinate]"
        )
    if mask.shape.rank != 1 or int(mask.shape[0]) != int(samples.shape[0]):
        raise ValueError("fixed-kernel valid mask must match draw count")
    valid_count = int(tf.reduce_sum(tf.cast(mask, tf.int32)).numpy())
    if valid_count <= 0:
        raise ValueError("fixed-kernel arm produced no valid draws")
    valid = tf.boolean_mask(samples, mask, axis=0)
    tf.debugging.assert_all_finite(valid, "fixed-kernel valid samples must be finite")
    return valid


def minimum_latent_ess(samples: Any) -> Mapping[str, Any]:
    """Compute a BayesFilter-owned coordinate ESS summary.

    ``samples`` has shape ``[draw, chain, coordinate]``.  This is the same
    per-chain Geyer initial-positive-sequence estimator used by BayesFilter's
    metric adequacy gate: each chain is kept separate and its coordinate ESS
    values are summed.  A single chain is accepted for a mechanism nomination,
    but the result explicitly makes no posterior convergence claim.
    """

    import tensorflow as tf
    tensor = tf.cast(tf.convert_to_tensor(samples), tf.float64)
    if tensor.shape.rank == 2:
        tensor = tensor[:, None, :]
    if tensor.shape.rank != 3 or any(dim is None for dim in tensor.shape):
        raise ValueError("samples must have shape [draw, chain, coordinate]")
    draws, chains, coordinates = (int(dim) for dim in tensor.shape)
    if draws < 4 or chains < 1 or coordinates < 1:
        raise ValueError("ESS requires at least four draws and one coordinate")
    tf.debugging.assert_all_finite(tensor, "ESS samples must be finite")
    ess_by_chain = []
    fft_size = 1 << math.ceil(math.log2(max(2, 2 * draws)))
    for chain_index in range(chains):
        chain = tensor[:, chain_index, :]
        centered = chain - tf.reduce_mean(chain, axis=0, keepdims=True)
        spectrum = tf.signal.rfft(tf.transpose(centered), fft_length=[fft_size])
        power = spectrum * tf.math.conj(spectrum)
        autocovariance = tf.transpose(
            tf.signal.irfft(power, fft_length=[fft_size])
        )[:draws]
        variance = autocovariance[0]
        valid = variance > tf.constant(sys.float_info.epsilon, tf.float64)
        denominator = tf.where(valid, variance, tf.ones_like(variance))
        rho = tf.where(
            valid[None, :],
            autocovariance / denominator[None, :],
            tf.zeros_like(autocovariance),
        )
        pair_sum = tf.zeros((coordinates,), tf.float64)
        active = valid
        previous = tf.fill((coordinates,), tf.constant(float("inf"), tf.float64))
        for lag in range(1, draws - 1, 2):
            pair = tf.minimum(rho[lag] + rho[lag + 1], previous)
            positive = active & tf.math.is_finite(pair) & (pair > 0.0)
            pair_sum = pair_sum + tf.where(positive, pair, 0.0)
            active = active & positive
            previous = tf.where(positive, pair, previous)
        tau = tf.maximum(1.0, 1.0 + 2.0 * pair_sum)
        chain_ess = tf.where(
            valid,
            tf.clip_by_value(float(draws) / tau, 1.0, float(draws)),
            tf.zeros_like(tau),
        )
        ess_by_chain.append(chain_ess)
    ess = tf.reduce_sum(tf.stack(ess_by_chain, axis=0), axis=0)
    tf.debugging.assert_all_finite(ess, "ESS result must be finite")
    values = tuple(float(item) for item in tf.reshape(ess, [-1]).numpy().tolist())
    return {
        "ess_method": "geyer_initial_positive_sequence_fft_autocorrelation",
        "sample_shape": (draws, chains, coordinates),
        "chain_count": chains,
        "state_count": draws,
        "coordinate_count": coordinates,
        "effective_sample_size_by_coordinate": values,
        "minimum_effective_sample_size": min(values),
        "maximum_effective_sample_size": max(values),
        "finite": True,
        "diagnostic_role": "mechanism_nomination_only",
        "single_chain_allowed": True,
        "posterior_convergence_claim": False,
    }


def minimum_latent_ess_checkpoints(
    samples: Any,
    checkpoints: tuple[int, ...],
    *,
    ess_threshold: float = 9.0,
) -> Mapping[str, Mapping[str, Any]]:
    """Summarize same-chain ESS prefixes without exposing coordinate vectors."""

    import tensorflow as tf

    tensor = tf.cast(tf.convert_to_tensor(samples), tf.float64)
    if tensor.shape.rank == 2:
        tensor = tensor[:, None, :]
    if tensor.shape.rank != 3 or any(dim is None for dim in tensor.shape):
        raise ValueError("samples must have shape [draw, chain, coordinate]")
    draws = int(tensor.shape[0])
    normalized = tuple(int(item) for item in checkpoints)
    if not normalized or (
        any(item < 4 or item > draws for item in normalized)
        or tuple(sorted(set(normalized))) != normalized
    ):
        raise ValueError(
            "checkpoints must be strictly increasing unique values between "
            "four and the sample draw count"
        )
    threshold = float(ess_threshold)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("ess_threshold must be positive and finite")

    summaries: dict[str, Mapping[str, Any]] = {}
    for checkpoint in normalized:
        report = minimum_latent_ess(tensor[:checkpoint])
        coordinate_ess = tuple(report["effective_sample_size_by_coordinate"])
        summaries[str(checkpoint)] = {
            "ess_method": report["ess_method"],
            "state_count": int(report["state_count"]),
            "chain_count": int(report["chain_count"]),
            "coordinate_count": int(report["coordinate_count"]),
            "minimum_effective_sample_size": float(
                report["minimum_effective_sample_size"]
            ),
            "maximum_effective_sample_size": float(
                report["maximum_effective_sample_size"]
            ),
            "coordinate_count_below_threshold": sum(
                float(value) < threshold for value in coordinate_ess
            ),
            "ess_threshold": threshold,
            "finite": bool(report["finite"]),
            "diagnostic_role": "mechanism_nomination_only",
            "posterior_convergence_claim": False,
        }
    return summaries


def run_fixed_kernel_arm(
    adapter: Any,
    initial_state: Any,
    config: FixedKernelArmConfig,
) -> FixedKernelArmResult:
    """Run one explicit-state, no-adaptation arm and discard its draws."""

    if not isinstance(config, FixedKernelArmConfig):
        raise TypeError("config must be FixedKernelArmConfig")
    import tensorflow as tf

    state = tf.cast(tf.convert_to_tensor(initial_state), tf.float64)
    runner_config = FixedSizeHMCChunkConfig(
        max_results=config.transition_count,
        num_burnin_steps=0,
        step_size=config.step_size,
        num_leapfrog_steps=config.num_leapfrog_steps,
        seed=config.seed,
        use_xla=config.use_xla,
        trace_policy="standard",
        target_status_trace_policy="none",
        target_scope=config.target_scope,
        chain_execution_mode=config.chain_execution_mode,
    )
    runner = build_fixed_size_hmc_chunk_runner(adapter, state, runner_config)
    result = runner.run(
        active_results=config.transition_count,
        current_state=state,
        seed=config.seed,
        step_size=config.step_size,
    )
    if int(result.diagnostics["valid_sample_count"].numpy()) != config.transition_count:
        raise ValueError("fixed-kernel arm did not complete its transition budget")
    if int(result.diagnostics["nonfinite_valid_sample_count"].numpy()) != 0:
        raise ValueError("fixed-kernel arm produced nonfinite samples")
    divergence_status = str(
        result.diagnostics.get(
            "divergence_status",
            result.diagnostics.get("native_divergence_status", "unavailable"),
        )
    )
    if divergence_status not in {"available", "not_exposed_by_kernel", "unavailable"}:
        raise ValueError("fixed-kernel arm returned invalid divergence provenance")
    divergence_available = divergence_status == "available"
    if divergence_available:
        divergence_count = int(result.diagnostics["divergence_count"].numpy())
    else:
        divergence_count = None
    target_nonfinite = int(
        result.diagnostics["target_log_prob_nonfinite_count"].numpy()
    )
    log_accept_nonfinite = int(
        result.diagnostics["log_accept_ratio_nonfinite_count"].numpy()
    )
    if target_nonfinite != 0:
        raise ValueError("fixed-kernel arm target evaluation was nonfinite")
    if log_accept_nonfinite != 0:
        raise ValueError("fixed-kernel arm log acceptance ratio was nonfinite")
    valid_samples = _valid_samples(result)
    ess_report = minimum_latent_ess(valid_samples)
    diagnostics = {
        **ess_report,
        "valid_sample_count": config.transition_count,
        "target_log_prob_nonfinite_count": target_nonfinite,
        "log_accept_ratio_nonfinite_count": log_accept_nonfinite,
        "acceptance_rate": float(result.diagnostics["acceptance_rate"].numpy()),
        "native_divergence_available": divergence_available,
        "divergence_status": divergence_status,
        "divergence_count": divergence_count,
    }
    if config.ess_checkpoints:
        diagnostics["ess_checkpoint_summaries"] = minimum_latent_ess_checkpoints(
            valid_samples,
            config.ess_checkpoints,
            ess_threshold=config.ess_threshold,
        )
    metadata = {
        **dict(result.metadata),
        "arm_label": config.label,
        "adaptation_policy": "fixed_kernel_no_adaptation",
        "num_burnin_steps": 0,
        "raw_samples_publicized": False,
        "final_state_publicized": False,
        "privacy_contract": {
            "public_summary_contains_raw_samples": False,
            "public_summary_contains_final_state": False,
            "public_summary_contains_hmc_mechanics": True,
        },
    }
    return FixedKernelArmResult(
        config=config,
        diagnostics=diagnostics,
        metadata=metadata,
    )
