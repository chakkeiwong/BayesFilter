"""Shared TensorFlow/TFP sequential HMC controller for frozen NeuTra targets."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.batched_value_score import reviewed_value_score_target_fn
from bayesfilter.inference.hmc_convergence import (
    rank_normalized_split_rhat_summary,
)
from bayesfilter.inference.hmc_posterior_diagnostics import (
    rank_normalized_bulk_tail_ess,
    rank_normalized_split_rhat,
)
from bayesfilter.inference.neutra_hmc_policy import (
    NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
)
from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
    FixedTransportFullChainConfig,
    FixedTransportHMCPolicy,
    build_fixed_transport_reusable_runner,
    fixed_transport_base_adapter_signature,
)
from bayesfilter.inference.posterior_adapter import value_score_capability


MAX_RESULTS_PER_CHAIN = 10_000
DEFAULT_ENERGY_ERROR_LOG_ACCEPT_THRESHOLD = -1000.0

ArchiveCallback = Callable[..., Mapping[str, Any]]
TargetStatusSummaryCallback = Callable[[Any], Mapping[str, Any]]
RetainedDiagnosticCallback = Callable[[tf.Tensor], Mapping[str, Any]]


@dataclass(frozen=True)
class SequentialNeuTraHMCXLAQualificationReceipt:
    """Evidence-bound authority for one canonical sequential XLA program."""

    status: str
    policy_id: str
    adapter_signature: str
    initial_state_shape: tuple[int, int]
    chunk_results: int
    program_signature: str
    tracing_count: int
    target_value_max_abs_residual: float
    target_score_max_abs_residual: float
    all_chains_moved: bool
    final_state_equals_last_sample: bool
    sequential_handoff_verified: bool
    target_status_passed: bool
    evidence_path: str
    evidence_sha256: str
    qualification_code_hash: str = ""
    qualified_programs: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.status != "passed" or self.policy_id != NEUTRA_SEQUENTIAL_HMC_POLICY_ID:
            raise ValueError("sequential XLA qualification receipt did not pass")
        shape = tuple(int(item) for item in self.initial_state_shape)
        if len(shape) != 2 or any(item <= 0 for item in shape):
            raise ValueError("sequential XLA qualification state shape is invalid")
        object.__setattr__(self, "initial_state_shape", shape)
        if int(self.chunk_results) <= 0 or int(self.tracing_count) != 1:
            raise ValueError("sequential XLA qualification program counts are invalid")
        for name in (
            "all_chains_moved",
            "final_state_equals_last_sample",
            "sequential_handoff_verified",
            "target_status_passed",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"sequential XLA qualification failed {name}")
        if not self.adapter_signature or not self.program_signature:
            raise ValueError("sequential XLA qualification signatures are required")
        if not self.evidence_path or not self.evidence_sha256:
            raise ValueError("sequential XLA qualification evidence is required")
        programs = tuple(dict(item) for item in self.qualified_programs)
        if programs:
            sizes = tuple(int(item.get("chunk_results", 0)) for item in programs)
            if any(size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
                raise ValueError("sequential XLA qualified programs are invalid")
            if int(self.chunk_results) not in sizes:
                raise ValueError("primary sequential XLA program is missing")
        object.__setattr__(self, "qualified_programs", programs)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.neutra.sequential_hmc_xla_qualification.v1",
            **asdict(self),
        }


class NeuTraHMCError(RuntimeError):
    """Raised when shared NeuTra HMC inputs or callbacks violate the policy."""


@dataclass(frozen=True)
class BatchedHMCConfig:
    """Static configuration for one batched fixed-kernel HMC invocation."""

    num_results: int
    num_burnin_steps: int
    step_size: float
    num_leapfrog_steps: int
    seed: tuple[int, int]
    jit_compile: bool = True
    energy_error_log_accept_threshold: float = (
        DEFAULT_ENERGY_ERROR_LOG_ACCEPT_THRESHOLD
    )

    def __post_init__(self) -> None:
        if int(self.num_results) <= 0:
            raise ValueError("num_results must be positive")
        if int(self.num_burnin_steps) < 0:
            raise ValueError("num_burnin_steps must be non-negative")
        if not math.isfinite(float(self.step_size)) or float(self.step_size) <= 0.0:
            raise ValueError("step_size must be finite and positive")
        if int(self.num_leapfrog_steps) <= 0:
            raise ValueError("num_leapfrog_steps must be positive")
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must have exactly two integers")
        threshold = float(self.energy_error_log_accept_threshold)
        if not math.isfinite(threshold) or threshold >= 0.0:
            raise ValueError(
                "energy_error_log_accept_threshold must be finite and negative"
            )
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "energy_error_log_accept_threshold", threshold)

    def payload(self, *, chain_count: int | None = None) -> Mapping[str, Any]:
        return {
            "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
            "num_results": int(self.num_results),
            "num_burnin_steps": int(self.num_burnin_steps),
            "step_size": float(self.step_size),
            "num_leapfrog_steps": int(self.num_leapfrog_steps),
            "seed": self.seed,
            "chain_count": None if chain_count is None else int(chain_count),
            "jit_compile": bool(self.jit_compile),
            "dtype": "float64",
            "energy_error_log_accept_threshold": (
                self.energy_error_log_accept_threshold
            ),
            "execution": "one_batched_tfp_sample_chain_invocation",
        }


# Compatibility name retained for campaign callers migrated from the first proof.
TensorHMCConfig = BatchedHMCConfig


@dataclass(frozen=True)
class _SharedSequentialNeuTraHMCConfig:
    """Bounded modern-R-hat controller for warm-up and retained sampling."""

    step_size: float
    num_leapfrog_steps: int
    warmup_seed: tuple[int, int]
    retained_seed: tuple[int, int]
    warmup_chunk_results: int = 1000
    warmup_min_results: int = 2000
    warmup_check_window_results: int = 1000
    warmup_max_results: int = MAX_RESULTS_PER_CHAIN
    warmup_rhat_max: float = 1.05
    retained_chunk_results: int = 1000
    retained_min_results: int = 1000
    retained_max_results: int = MAX_RESULTS_PER_CHAIN
    retained_rhat_max: float = 1.01
    minimum_chain_count: int = 4
    jit_compile: bool = True
    energy_error_log_accept_threshold: float = (
        DEFAULT_ENERGY_ERROR_LOG_ACCEPT_THRESHOLD
    )

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.step_size)) or float(self.step_size) <= 0.0:
            raise ValueError("step_size must be finite and positive")
        if int(self.num_leapfrog_steps) <= 0:
            raise ValueError("num_leapfrog_steps must be positive")
        for name in ("warmup_seed", "retained_seed"):
            seed = tuple(int(item) for item in getattr(self, name))
            if len(seed) != 2:
                raise ValueError(f"{name} must have exactly two integers")
            object.__setattr__(self, name, seed)
        if self.warmup_seed == self.retained_seed:
            raise ValueError("warmup_seed and retained_seed must be distinct")
        for name in (
            "warmup_chunk_results",
            "warmup_min_results",
            "warmup_check_window_results",
            "warmup_max_results",
            "retained_chunk_results",
            "retained_min_results",
            "retained_max_results",
            "minimum_chain_count",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.minimum_chain_count < 4:
            raise ValueError("minimum_chain_count must be at least four")
        if self.warmup_min_results > self.warmup_max_results:
            raise ValueError("warmup_min_results must not exceed warmup_max_results")
        if self.warmup_check_window_results > self.warmup_max_results:
            raise ValueError(
                "warmup_check_window_results must not exceed warmup_max_results"
            )
        if self.warmup_chunk_results > self.warmup_max_results:
            raise ValueError("warmup_chunk_results must not exceed warmup_max_results")
        if self.retained_min_results > self.retained_max_results:
            raise ValueError("retained_min_results must not exceed retained_max_results")
        if self.retained_chunk_results > self.retained_max_results:
            raise ValueError("retained_chunk_results must not exceed retained_max_results")
        if self.warmup_max_results > MAX_RESULTS_PER_CHAIN:
            raise ValueError("warmup_max_results must not exceed 10000 per chain")
        if self.retained_max_results > MAX_RESULTS_PER_CHAIN:
            raise ValueError("retained_max_results must not exceed 10000 per chain")
        for name in ("warmup_rhat_max", "retained_rhat_max"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 1.0:
                raise ValueError(f"{name} must be finite and greater than 1")
            object.__setattr__(self, name, value)
        threshold = float(self.energy_error_log_accept_threshold)
        if not math.isfinite(threshold) or threshold >= 0.0:
            raise ValueError(
                "energy_error_log_accept_threshold must be finite and negative"
            )
        object.__setattr__(self, "energy_error_log_accept_threshold", threshold)

    def payload(self, *, chain_count: int | None = None) -> Mapping[str, Any]:
        return {
            "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
            "step_size": float(self.step_size),
            "num_leapfrog_steps": int(self.num_leapfrog_steps),
            "warmup_seed": self.warmup_seed,
            "retained_seed": self.retained_seed,
            "warmup_chunk_results": self.warmup_chunk_results,
            "warmup_min_results": self.warmup_min_results,
            "warmup_check_window_results": self.warmup_check_window_results,
            "warmup_max_results": self.warmup_max_results,
            "warmup_rhat_max": self.warmup_rhat_max,
            "retained_chunk_results": self.retained_chunk_results,
            "retained_min_results": self.retained_min_results,
            "retained_max_results": self.retained_max_results,
            "retained_rhat_max": self.retained_rhat_max,
            "rhat_definition": (
                "max(rank-normalized split R-hat, "
                "folded rank-normalized split R-hat)"
            ),
            "minimum_chain_count": self.minimum_chain_count,
            "chain_count": None if chain_count is None else int(chain_count),
            "jit_compile": bool(self.jit_compile),
            "dtype": "float64",
            "energy_error_log_accept_threshold": (
                self.energy_error_log_accept_threshold
            ),
        }


def run_batched_hmc(
    *,
    adapter: Any,
    initial_state: Any,
    config: BatchedHMCConfig,
    target_status_summary_fn: TargetStatusSummaryCallback | None = None,
) -> Mapping[str, Any]:
    """Run all chains in one fixed-size TensorFlow/TFP HMC invocation."""

    state, chain_count, _ = _validated_initial_state(initial_state)
    compiled = _build_batched_hmc_program(
        adapter=adapter,
        num_results=int(config.num_results),
        num_burnin_steps=int(config.num_burnin_steps),
        step_size=float(config.step_size),
        num_leapfrog_steps=int(config.num_leapfrog_steps),
        jit_compile=bool(config.jit_compile),
    )
    started = time.monotonic()
    samples, trace = compiled(state, tf.constant(config.seed, tf.int32))
    return _summarize_batched_hmc_output(
        initial_state=state,
        samples=samples,
        trace=trace,
        config=config,
        chain_count=chain_count,
        elapsed_seconds=time.monotonic() - started,
        target_status_summary_fn=target_status_summary_fn,
    )


def _build_batched_hmc_program(
    *,
    adapter: Any,
    num_results: int,
    num_burnin_steps: int,
    step_size: float,
    num_leapfrog_steps: int,
    jit_compile: bool,
) -> Callable[[tf.Tensor, tf.Tensor], Any]:
    """Build one reusable fixed-size batched HMC program."""

    target = reviewed_value_score_target_fn(adapter, dtype=tf.float64, require_batched=True)
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target,
        step_size=tf.constant(step_size, tf.float64),
        num_leapfrog_steps=int(num_leapfrog_steps),
        state_gradients_are_stopped=True,
    )

    def trace_fn(state: Any, kernel_results: Any) -> Mapping[str, tf.Tensor]:
        result = {
            "is_accepted": kernel_results.is_accepted,
            "log_accept_ratio": kernel_results.log_accept_ratio,
            "target_log_prob": kernel_results.accepted_results.target_log_prob,
        }
        telemetry = getattr(adapter, "target_status_telemetry", None)
        if callable(telemetry):
            result["target_status"] = telemetry(state)
        return result

    @tf.function(jit_compile=bool(jit_compile), reduce_retracing=True)
    def compiled(current_state: tf.Tensor, seed: tf.Tensor):
        return tfp.mcmc.sample_chain(
            num_results=int(num_results),
            num_burnin_steps=int(num_burnin_steps),
            current_state=current_state,
            kernel=kernel,
            trace_fn=trace_fn,
            seed=seed,
        )

    return compiled


def _summarize_batched_hmc_output(
    *,
    initial_state: tf.Tensor,
    samples: Any,
    trace: Mapping[str, Any],
    config: BatchedHMCConfig,
    chain_count: int,
    elapsed_seconds: float,
    target_status_summary_fn: TargetStatusSummaryCallback | None = None,
) -> Mapping[str, Any]:
    samples = tf.convert_to_tensor(samples, tf.float64)
    accepted = tf.convert_to_tensor(trace["is_accepted"], tf.bool)
    log_accept = tf.convert_to_tensor(trace["log_accept_ratio"], tf.float64)
    target_log_prob = tf.convert_to_tensor(trace["target_log_prob"], tf.float64)
    target_status = trace.get("target_status")
    samples_finite = bool(tf.reduce_all(tf.math.is_finite(samples)).numpy())
    log_accept_finite = bool(tf.reduce_all(tf.math.is_finite(log_accept)).numpy())
    target_finite = bool(tf.reduce_all(tf.math.is_finite(target_log_prob)).numpy())
    extreme_log_accept = tf.logical_and(
        tf.math.is_finite(log_accept),
        log_accept < tf.constant(
            config.energy_error_log_accept_threshold, tf.float64
        ),
    )
    status_summary = _summarize_target_status(
        target_status, target_status_summary_fn=target_status_summary_fn
    )
    diagnostics = {
        "sample_shape": tuple(int(item) for item in samples.shape),
        "samples_all_finite": samples_finite,
        "log_accept_ratio_all_finite": log_accept_finite,
        "target_log_prob_all_finite": target_finite,
        "acceptance_rate": float(
            tf.reduce_mean(tf.cast(accepted, tf.float64)).numpy()
        ),
        "acceptance_rate_by_chain": tuple(
            float(item)
            for item in tf.reduce_mean(tf.cast(accepted, tf.float64), axis=0)
            .numpy()
            .tolist()
        ),
        "extreme_log_accept_count": int(
            tf.reduce_sum(tf.cast(extreme_log_accept, tf.int32)).numpy()
        ),
        "extreme_log_accept_definition": (
            "finite log_accept_ratio < "
            f"{config.energy_error_log_accept_threshold:g}"
        ),
        "extreme_log_accept_role": "explanatory_only_not_a_veto_or_divergence",
        # Compatibility aliases for existing readers. These values are not a
        # divergence count and must not be used as a health veto.
        "energy_error_divergence_count": int(
            tf.reduce_sum(tf.cast(extreme_log_accept, tf.int32)).numpy()
        ),
        "energy_error_divergence_definition": (
            "historical field name; counts finite log_accept_ratio < "
            f"{config.energy_error_log_accept_threshold:g}; explanatory only"
        ),
        "native_divergence_status": "not_exposed_by_tfp_hamiltonian_monte_carlo",
        "all_states_moved": bool(
            tf.reduce_all(
                tf.reduce_any(tf.not_equal(samples[-1], initial_state), axis=-1)
            ).numpy()
        ),
        "elapsed_seconds": float(elapsed_seconds),
        "jit_compile": bool(config.jit_compile),
        "single_batched_sample_chain_invocation": True,
        "target_status_telemetry": status_summary,
    }
    diagnostics["health_passed"] = bool(
        samples_finite
        and log_accept_finite
        and target_finite
        and diagnostics["all_states_moved"]
        and (
            status_summary.get("available") is not True
            or status_summary.get("all_status_valid") is True
        )
    )
    return {
        "samples": samples,
        "trace": {
            "is_accepted": accepted,
            "log_accept_ratio": log_accept,
            "target_log_prob": target_log_prob,
            "target_status": target_status,
        },
        "diagnostics": diagnostics,
        "config": config.payload(chain_count=chain_count),
    }


def _run_shared_sequential_neutra_hmc(
    *,
    adapter: Any,
    initial_state: Any,
    model_transform: Callable[[tf.Tensor], Any] | None = None,
    raw_transform: Callable[[tf.Tensor], Any] | None = None,
    parameter_names: Sequence[str],
    config: _SharedSequentialNeuTraHMCConfig,
    retained_diagnostic_fn: RetainedDiagnosticCallback | None = None,
    archive_callback: ArchiveCallback | None = None,
    target_status_summary_fn: TargetStatusSummaryCallback | None = None,
) -> Mapping[str, Any]:
    """Retain warm-up and sample cumulatively until declared gates or caps."""

    state, chain_count, dimension = _validated_initial_state(initial_state)
    if chain_count < config.minimum_chain_count:
        raise NeuTraHMCError(
            f"sequential HMC requires at least {config.minimum_chain_count} chains"
        )
    names = tuple(str(item) for item in parameter_names)
    if len(names) != dimension:
        raise NeuTraHMCError("parameter_names must match the HMC dimension")
    if model_transform is not None and raw_transform is not None:
        raise NeuTraHMCError("provide only one of model_transform or raw_transform")
    transform_fn = model_transform or raw_transform or (lambda values: values)
    programs: dict[int, Callable[[tf.Tensor, tf.Tensor], Any]] = {}

    def transform(samples: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(transform_fn(samples), tf.float64)
        if values.shape != samples.shape:
            raise NeuTraHMCError(
                "model_transform must preserve [draw, chain, parameter] shape"
            )
        return values

    def run_chunk(active_results: int, seed: tuple[int, int]) -> Mapping[str, Any]:
        active = int(active_results)
        if active not in programs:
            programs[active] = _build_batched_hmc_program(
                adapter=adapter,
                num_results=active,
                num_burnin_steps=0,
                step_size=config.step_size,
                num_leapfrog_steps=config.num_leapfrog_steps,
                jit_compile=config.jit_compile,
            )
        chunk_config = BatchedHMCConfig(
            num_results=active,
            num_burnin_steps=0,
            step_size=config.step_size,
            num_leapfrog_steps=config.num_leapfrog_steps,
            seed=seed,
            jit_compile=config.jit_compile,
            energy_error_log_accept_threshold=(
                config.energy_error_log_accept_threshold
            ),
        )
        started = time.monotonic()
        samples, trace = programs[active](state, tf.constant(seed, tf.int32))
        return _summarize_batched_hmc_output(
            initial_state=state,
            samples=samples,
            trace=trace,
            config=chunk_config,
            chain_count=chain_count,
            elapsed_seconds=time.monotonic() - started,
            target_status_summary_fn=target_status_summary_fn,
        )

    started = time.monotonic()
    warmup_latent_chunks: list[tf.Tensor] = []
    warmup_model_chunks: list[tf.Tensor] = []
    warmup_checks: list[Mapping[str, Any]] = []
    warmup_archives: list[Mapping[str, Any]] = []
    hard_vetoes: list[str] = []
    warmup_count = 0
    warmup_passed = False
    warmup_index = 0

    while warmup_count < config.warmup_max_results:
        active = min(
            config.warmup_chunk_results, config.warmup_max_results - warmup_count
        )
        seed = _shared_sequential_chunk_seed(config.warmup_seed, warmup_index)
        chunk = run_chunk(active, seed)
        latent_samples = tf.convert_to_tensor(chunk["samples"], tf.float64)
        model_samples = transform(latent_samples)
        state = latent_samples[-1]
        warmup_latent_chunks.append(latent_samples)
        warmup_model_chunks.append(model_samples)
        warmup_count += active
        if archive_callback is not None:
            warmup_archives.append(
                _call_archive(
                    archive_callback,
                    stage="warmup",
                    chunk_index=warmup_index,
                    latent_samples=latent_samples,
                    model_samples=model_samples,
                    seed=seed,
                    cumulative=False,
                )
            )
        if chunk["diagnostics"]["health_passed"] is not True:
            hard_vetoes.append("warmup_chunk_health_failed")
            warmup_checks.append(
                {
                    "chunk_index": warmup_index,
                    "completed_results_per_chain": warmup_count,
                    "seed": seed,
                    "health": chunk["diagnostics"],
                    "modern_rhat": None,
                    "passed": False,
                }
            )
            break
        rhat = None
        if warmup_count >= max(
            config.warmup_min_results, config.warmup_check_window_results
        ):
            cumulative = tf.concat(warmup_model_chunks, axis=0)
            window = cumulative[-config.warmup_check_window_results :]
            rhat = rank_normalized_split_rhat_summary(
                window, rhat_max=config.warmup_rhat_max
            )
            warmup_passed = bool(rhat["passed"])
        warmup_checks.append(
            {
                "chunk_index": warmup_index,
                "completed_results_per_chain": warmup_count,
                "check_window_results_per_chain": (
                    config.warmup_check_window_results if rhat is not None else 0
                ),
                "seed": seed,
                "health": chunk["diagnostics"],
                "modern_rhat": rhat,
                "passed": warmup_passed,
            }
        )
        warmup_index += 1
        if warmup_passed:
            break

    retained_latent_chunks: list[tf.Tensor] = []
    retained_model_chunks: list[tf.Tensor] = []
    retained_checks: list[Mapping[str, Any]] = []
    retained_archives: list[Mapping[str, Any]] = []
    retained_count = 0
    retained_passed = False
    retained_index = 0

    while warmup_passed and retained_count < config.retained_max_results:
        active = min(
            config.retained_chunk_results,
            config.retained_max_results - retained_count,
        )
        seed = _shared_sequential_chunk_seed(config.retained_seed, retained_index)
        chunk = run_chunk(active, seed)
        latent_samples = tf.convert_to_tensor(chunk["samples"], tf.float64)
        model_samples = transform(latent_samples)
        state = latent_samples[-1]
        retained_latent_chunks.append(latent_samples)
        retained_model_chunks.append(model_samples)
        retained_count += active
        if archive_callback is not None:
            retained_archives.append(
                _call_archive(
                    archive_callback,
                    stage="retained",
                    chunk_index=retained_index,
                    latent_samples=latent_samples,
                    model_samples=model_samples,
                    seed=seed,
                    cumulative=False,
                )
            )
        if chunk["diagnostics"]["health_passed"] is not True:
            hard_vetoes.append("retained_chunk_health_failed")
            retained_checks.append(
                {
                    "chunk_index": retained_index,
                    "completed_results_per_chain": retained_count,
                    "seed": seed,
                    "health": chunk["diagnostics"],
                    "modern_rhat": None,
                    "passed": False,
                }
            )
            break
        cumulative = tf.concat(retained_model_chunks, axis=0)
        if retained_diagnostic_fn is None:
            diagnostic = rank_normalized_split_rhat_summary(
                cumulative, rhat_max=config.retained_rhat_max
            )
            diagnostic_role = "modern_rhat"
        else:
            diagnostic = retained_diagnostic_fn(cumulative)
            if not isinstance(diagnostic, Mapping) or "passed" not in diagnostic:
                raise NeuTraHMCError(
                    "retained_diagnostic_fn must return a mapping with passed"
                )
            diagnostic_role = "full_convergence"
        retained_passed = bool(
            retained_count >= config.retained_min_results and diagnostic["passed"]
        )
        check = {
            "chunk_index": retained_index,
            "completed_results_per_chain": retained_count,
            "seed": seed,
            "health": chunk["diagnostics"],
            "diagnostic_role": diagnostic_role,
            "passed": retained_passed,
        }
        check[diagnostic_role] = diagnostic
        retained_checks.append(check)
        retained_index += 1
        if retained_passed:
            break

    warmup_latent = tf.concat(warmup_latent_chunks, axis=0)
    warmup_model = tf.concat(warmup_model_chunks, axis=0)
    empty = tf.zeros((0, chain_count, dimension), tf.float64)
    retained_latent = (
        tf.concat(retained_latent_chunks, axis=0)
        if retained_latent_chunks
        else empty
    )
    retained_model = (
        tf.concat(retained_model_chunks, axis=0)
        if retained_model_chunks
        else empty
    )
    cumulative_archives = None
    if archive_callback is not None:
        cumulative_archives = {
            "warmup": _call_archive(
                archive_callback,
                stage="warmup",
                chunk_index=None,
                latent_samples=warmup_latent,
                model_samples=warmup_model,
                seed=None,
                cumulative=True,
            )
        }
        if retained_count:
            cumulative_archives = {
                **cumulative_archives,
                "retained": _call_archive(
                    archive_callback,
                    stage="retained",
                    chunk_index=None,
                    latent_samples=retained_latent,
                    model_samples=retained_model,
                    seed=None,
                    cumulative=True,
                ),
            }

    passed = bool(warmup_passed and retained_passed and not hard_vetoes)
    return {
        "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        "passed": passed,
        "decision": (
            "ADMIT_SEQUENTIAL_FIXED_NEUTRA_HMC_KERNEL"
            if passed
            else "REJECT_SEQUENTIAL_FIXED_NEUTRA_HMC_KERNEL"
        ),
        "config": config.payload(chain_count=chain_count),
        "warmup_passed": warmup_passed,
        "warmup_cap_hit": bool(
            not warmup_passed and warmup_count >= config.warmup_max_results
        ),
        "warmup_results_per_chain": warmup_count,
        "warmup_check_count": len(warmup_checks),
        "warmup_checks": tuple(warmup_checks),
        "retained_passed": retained_passed,
        "retained_cap_hit": bool(
            warmup_passed
            and not retained_passed
            and retained_count >= config.retained_max_results
        ),
        "retained_results_per_chain": retained_count,
        "retained_check_count": len(retained_checks),
        "retained_checks": tuple(retained_checks),
        "hard_vetoes": tuple(dict.fromkeys(hard_vetoes)),
        "warmup_archives": tuple(warmup_archives),
        "retained_archives": tuple(retained_archives),
        "cumulative_archives": cumulative_archives,
        "warmup_excluded_from_posterior": True,
        "warmup_samples_retained": True,
        "elapsed_seconds": time.monotonic() - started,
        "private_warmup_z": warmup_latent,
        "private_warmup_raw": warmup_model,
        "private_retained_z": retained_latent,
        "private_retained_raw": retained_model,
    }


def _shared_sequential_chunk_seed(
    root_seed: tuple[int, int], chunk_index: int
) -> tuple[int, int]:
    """Derive deterministic, stage-separated fixed-size chunk seeds."""

    return int(root_seed[0]), int(root_seed[1]) + 1009 * (int(chunk_index) + 1)


def _validated_initial_state(initial_state: Any) -> tuple[tf.Tensor, int, int]:
    state = tf.cast(tf.convert_to_tensor(initial_state), tf.float64)
    if state.shape.rank != 2:
        raise NeuTraHMCError("initial_state must have shape [chain, dimension]")
    chain_count, dimension = state.shape.as_list()
    if chain_count is None or chain_count <= 0:
        raise NeuTraHMCError("initial_state chain count must be static and positive")
    if dimension is None or dimension <= 0:
        raise NeuTraHMCError("initial_state dimension must be static and positive")
    return state, int(chain_count), int(dimension)


def _summarize_target_status(
    target_status: Any,
    *,
    target_status_summary_fn: TargetStatusSummaryCallback | None,
) -> Mapping[str, Any]:
    if target_status is None:
        return {"available": False}
    if target_status_summary_fn is not None:
        summary = target_status_summary_fn(target_status)
        if not isinstance(summary, Mapping):
            raise NeuTraHMCError("target_status_summary_fn must return a mapping")
        if summary.get("available") is not True:
            raise NeuTraHMCError(
                "target status summary must declare available=True"
            )
        if not isinstance(summary.get("all_status_valid"), bool):
            raise NeuTraHMCError(
                "target status summary must declare boolean all_status_valid"
            )
        return dict(summary)
    if not isinstance(target_status, Mapping):
        raise NeuTraHMCError(
            "target telemetry requires target_status_summary_fn or standard mapping"
        )
    try:
        status_code = tf.convert_to_tensor(target_status["status_code"], tf.int32)
        valid_score = tf.convert_to_tensor(
            target_status["valid_pre_regularized_score"], tf.bool
        )
    except KeyError as exc:
        raise NeuTraHMCError(
            "standard target telemetry lacks status_code or valid score"
        ) from exc
    invalid = tf.logical_or(tf.not_equal(status_code, 0), tf.logical_not(valid_score))
    return {
        "available": True,
        "all_status_valid": bool(tf.reduce_all(tf.logical_not(invalid)).numpy()),
        "status_nonvalid_count": int(
            tf.reduce_sum(tf.cast(invalid, tf.int32)).numpy()
        ),
        "trace_scope": "sampled_transition_states",
    }


def _call_archive(
    callback: ArchiveCallback,
    **kwargs: Any,
) -> Mapping[str, Any]:
    result = callback(**kwargs)
    if not isinstance(result, Mapping):
        raise NeuTraHMCError("archive_callback must return a mapping")
    return result

from bayesfilter.inference.batched_value_score import reviewed_value_score_target_fn
from bayesfilter.inference.hmc_posterior_diagnostics import (
    rank_normalized_bulk_tail_ess,
    rank_normalized_split_rhat,
)


_ARCHIVED_SEQUENTIAL_NEUTRA_HMC_SCHEMA = "bayesfilter.neutra.sequential_hmc_result.v1"


@dataclass(frozen=True)
class _ArchivedSequentialNeuTraHMCConfig:
    """Fixed-kernel sequential warm-up and retained-sampling policy."""

    step_size: float
    num_leapfrog_steps: int
    seed: tuple[int, int]
    warmup_chunk_size: int = 500
    warmup_min_results: int = 2000
    warmup_window_results: int = 1000
    warmup_max_results: int = 10000
    retained_chunk_size: int = 500
    retained_min_results: int = 1000
    retained_max_results: int = 10000
    retained_check_interval_results: int | None = None
    warmup_rhat_max: float = 1.05
    retained_rhat_max: float = 1.01
    bulk_ess_min: float = 400.0
    tail_ess_min: float = 400.0
    delta_h_abs_max: float = 1000.0
    acceptance_min: float = 0.35
    acceptance_max: float = 0.95
    chain_count: int = 4
    use_xla: bool = True
    target_status_required: bool = True
    primary_diagnostic_coordinate: str = "maximum_over_z_and_model"
    retained_ess_required: bool = True
    xla_qualification_required: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.step_size)) or float(self.step_size) <= 0.0:
            raise ValueError("step_size must be positive and finite")
        object.__setattr__(self, "step_size", float(self.step_size))
        for name in (
            "num_leapfrog_steps",
            "warmup_chunk_size",
            "warmup_min_results",
            "warmup_window_results",
            "warmup_max_results",
            "retained_chunk_size",
            "retained_min_results",
            "retained_max_results",
            "chain_count",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.num_leapfrog_steps < 2:
            raise ValueError("num_leapfrog_steps must be greater than or equal to 2")
        if self.chain_count < 2:
            raise ValueError("sequential HMC requires at least two chains")
        if self.warmup_window_results > self.warmup_min_results:
            raise ValueError("warmup window cannot exceed minimum warm-up")
        if self.warmup_window_results < 4:
            raise ValueError("warmup diagnostic window requires at least four draws")
        if self.warmup_min_results > self.warmup_max_results:
            raise ValueError("warmup minimum cannot exceed maximum")
        if self.retained_min_results > self.retained_max_results:
            raise ValueError("retained minimum cannot exceed maximum")
        if self.retained_min_results < 4:
            raise ValueError("retained diagnostics require at least four draws")
        for total_name, chunk_name in (
            ("warmup_min_results", "warmup_chunk_size"),
            ("warmup_window_results", "warmup_chunk_size"),
            ("warmup_max_results", "warmup_chunk_size"),
            ("retained_min_results", "retained_chunk_size"),
            ("retained_max_results", "retained_chunk_size"),
        ):
            if int(getattr(self, total_name)) % int(getattr(self, chunk_name)):
                raise ValueError(f"{total_name} must be a multiple of {chunk_name}")
        check_interval = (
            self.retained_chunk_size
            if self.retained_check_interval_results is None
            else int(self.retained_check_interval_results)
        )
        if check_interval <= 0 or check_interval % self.retained_chunk_size:
            raise ValueError(
                "retained_check_interval_results must be a positive multiple "
                "of retained_chunk_size"
            )
        object.__setattr__(self, "retained_check_interval_results", check_interval)
        for name in (
            "warmup_rhat_max",
            "retained_rhat_max",
            "bulk_ess_min",
            "tail_ess_min",
            "delta_h_abs_max",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if self.warmup_rhat_max <= 1.0 or self.retained_rhat_max <= 1.0:
            raise ValueError("R-hat thresholds must exceed one")
        for name in ("acceptance_min", "acceptance_max"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, value)
        if self.acceptance_min >= self.acceptance_max:
            raise ValueError("acceptance_min must be less than acceptance_max")
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain two integers")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "use_xla", bool(self.use_xla))
        object.__setattr__(self, "target_status_required", bool(self.target_status_required))
        object.__setattr__(self, "retained_ess_required", bool(self.retained_ess_required))
        object.__setattr__(
            self, "xla_qualification_required", bool(self.xla_qualification_required)
        )
        coordinate = str(self.primary_diagnostic_coordinate)
        if coordinate not in {"hmc_coordinates_z", "maximum_over_z_and_model"}:
            raise ValueError("primary_diagnostic_coordinate is invalid")
        object.__setattr__(self, "primary_diagnostic_coordinate", coordinate)

    def payload(self) -> Mapping[str, Any]:
        return {
            "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
            **asdict(self),
        }


@dataclass(frozen=True)
class _ArchivedSequentialNeuTraHMCResult:
    """Public-safe sequential run result; raw draws remain in private shards."""

    passed: bool
    stop_reason: str
    warmup_results_per_chain: int
    retained_results_per_chain: int
    diagnostics: Mapping[str, Any]
    archive: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def payload(self) -> Mapping[str, Any]:
        return {"schema": _ARCHIVED_SEQUENTIAL_NEUTRA_HMC_SCHEMA, **asdict(self)}


def _archived_sequential_chunk_seed(
    root_seed: tuple[int, int], *, phase_index: int, chunk_index: int
) -> tuple[int, int]:
    """Return deterministic, phase-separated chunk seeds."""

    first, second = (int(item) for item in root_seed)
    return first, second + 1000003 * int(phase_index) + 1009 * (int(chunk_index) + 1)

class SequentialNeuTraHMCConfig:
    """Construct either supported sequential-controller configuration.

    The shared callback controller uses distinct warm-up and retained seeds.
    The older archived-output controller uses one root seed and archive fields.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        shared_fields = {"warmup_seed", "retained_seed"}
        archived_fields = {"seed", "warmup_chunk_size", "archive_root"}
        if shared_fields.intersection(kwargs):
            return _SharedSequentialNeuTraHMCConfig(*args, **kwargs)
        if archived_fields.intersection(kwargs):
            return _ArchivedSequentialNeuTraHMCConfig(*args, **kwargs)
        raise TypeError(
            "sequential HMC config requires warmup_seed/retained_seed or seed"
        )


def run_sequential_neutra_hmc(*args: Any, **kwargs: Any) -> Any:
    """Dispatch to the shared or archived-output controller by call contract."""

    config = kwargs.get("config")
    if config is None and len(args) >= 3:
        config = args[2]
    if isinstance(config, _ArchivedSequentialNeuTraHMCConfig):
        receipt = kwargs.pop("xla_qualification_receipt", None)
        adapter = kwargs.get("adapter") if "adapter" in kwargs else args[0]
        initial_state = (
            kwargs.get("initial_state") if "initial_state" in kwargs else args[1]
        )
        if config.xla_qualification_required:
            if receipt is None:
                raise NeuTraHMCError(
                    "sequential XLA production requires an exact qualification receipt"
                )
            validate_sequential_neutra_hmc_xla_receipt(
                receipt,
                adapter=adapter,
                initial_state=initial_state,
                config=config,
            )
        return _run_archived_sequential_neutra_hmc(*args, **kwargs)
    return _run_shared_sequential_neutra_hmc(*args, **kwargs)


def sequential_chunk_seed(
    root_seed: tuple[int, int],
    chunk_index: int | None = None,
    *,
    phase_index: int | None = None,
) -> tuple[int, int]:
    """Dispatch deterministic seed derivation by the requested signature."""

    if phase_index is not None:
        if chunk_index is None:
            raise TypeError("chunk_index is required")
        return _archived_sequential_chunk_seed(
            root_seed, phase_index=phase_index, chunk_index=chunk_index
        )
    if chunk_index is None:
        raise TypeError("chunk_index is required")
    return _shared_sequential_chunk_seed(root_seed, chunk_index)


def _tensor_tree_python(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _tensor_tree_python(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tensor_tree_python(item) for item in value]
    if tf.is_tensor(value):
        return _tensor_tree_python(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_tensor(path: Path, value: Any) -> Mapping[str, Any]:
    tensor = tf.convert_to_tensor(value)
    serialized = tf.io.serialize_tensor(tensor).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise NeuTraHMCError(f"archive shard already exists: {path}")
    path.write_bytes(serialized)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "bytes": len(serialized),
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": tensor.dtype.name,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise NeuTraHMCError(f"artifact already exists: {path}")
    text = json.dumps(_tensor_tree_python(payload), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _tensor_tree_python(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_tensor_tree_python(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_tensor_receipt(receipt: Mapping[str, Any]) -> tf.Tensor:
    path = Path(str(receipt["path"]))
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != str(receipt["sha256"]):
        raise NeuTraHMCError(f"tensor receipt hash mismatch: {path}")
    dtype = tf.dtypes.as_dtype(str(receipt["dtype"]))
    tensor = tf.io.parse_tensor(data, out_type=dtype)
    expected_shape = tuple(int(item) for item in receipt["shape"])
    if tuple(tensor.shape) != expected_shape:
        raise NeuTraHMCError(f"tensor receipt shape mismatch: {path}")
    return tensor


def _read_json_receipt(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    path = Path(str(receipt["path"]))
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != str(receipt["sha256"]):
        raise NeuTraHMCError(f"JSON receipt hash mismatch: {path}")
    payload = json.loads(data)
    if not isinstance(payload, Mapping):
        raise NeuTraHMCError(f"JSON receipt is not a mapping: {path}")
    return payload


def _maximum_absolute_residual(left: Any, right: Any) -> float:
    left_tensor = tf.cast(tf.convert_to_tensor(left), tf.float64)
    right_tensor = tf.cast(tf.convert_to_tensor(right), tf.float64)
    if left_tensor.shape != right_tensor.shape:
        raise NeuTraHMCError("XLA parity tensors have different shapes")
    return float(tf.reduce_max(tf.abs(left_tensor - right_tensor)).numpy())


def _sequential_xla_qualification_code_hash() -> str:
    """Bind qualification receipts to the exact target and runner implementation."""

    from bayesfilter.inference import batched_value_score
    from bayesfilter.inference import fixed_transport_hmc_mechanics_tf

    digest = hashlib.sha256()
    for path in sorted(
        (
            Path(__file__).resolve(),
            Path(batched_value_score.__file__).resolve(),
            Path(fixed_transport_hmc_mechanics_tf.__file__).resolve(),
        )
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def qualify_sequential_neutra_hmc_xla(
    *,
    adapter: Any,
    initial_state: Any,
    step_size: float,
    num_leapfrog_steps: int,
    seed: tuple[int, int],
    evidence_path: str | Path,
    chunk_results: int = 500,
    additional_chunk_results: Sequence[int] = (),
    value_score_atol: float = 2.0e-9,
) -> SequentialNeuTraHMCXLAQualificationReceipt:
    """Qualify the exact canonical sequential XLA graph and state handoff."""

    output = Path(evidence_path).resolve()
    if output.exists():
        raise FileExistsError(f"XLA qualification evidence exists: {output}")
    state, chain_count, dimension = _validated_initial_state(initial_state)
    if (chain_count, dimension) != (4, 9):
        raise NeuTraHMCError("DZ5 exact qualification requires state shape [4,9]")
    capability = value_score_capability(adapter)
    if not capability.is_accepted_xla_hmc_authority:
        raise NeuTraHMCError(
            "sequential XLA qualification requires accepted target-XLA authority"
        )
    target = reviewed_value_score_target_fn(adapter, dtype=tf.float64, require_batched=True)

    def target_value_score(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(values)
            target_value = target(values)
            target_total = tf.reduce_sum(target_value)
        return target_value, tape.gradient(target_total, values)

    eager_value, eager_score = target_value_score(state)
    compiled_target = tf.function(
        target_value_score,
        input_signature=(tf.TensorSpec((4, 9), tf.float64),),
        jit_compile=True,
        reduce_retracing=True,
    )
    compiled_value, compiled_score = compiled_target(state)
    value_residual = _maximum_absolute_residual(eager_value, compiled_value)
    score_residual = _maximum_absolute_residual(eager_score, compiled_score)
    if value_residual > value_score_atol or score_residual > value_score_atol:
        raise NeuTraHMCError("canonical sequential target XLA parity failed")

    config = _ArchivedSequentialNeuTraHMCConfig(
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
        seed=seed,
        warmup_chunk_size=chunk_results,
        warmup_min_results=chunk_results,
        warmup_window_results=chunk_results,
        warmup_max_results=chunk_results,
        retained_chunk_size=chunk_results,
        retained_min_results=chunk_results,
        retained_max_results=chunk_results,
        bulk_ess_min=1.0,
        tail_ess_min=1.0,
        acceptance_min=0.0,
        acceptance_max=1.0,
        chain_count=4,
        use_xla=True,
        target_status_required=True,
        primary_diagnostic_coordinate="hmc_coordinates_z",
        retained_ess_required=False,
    )
    sizes = tuple(dict.fromkeys((int(chunk_results), *(int(item) for item in additional_chunk_results))))
    if any(size <= 0 for size in sizes):
        raise ValueError("qualified chunk sizes must be positive")
    runners = {
        size: _ChunkRunner(adapter, state, config, num_results=size) for size in sizes
    }
    runner = runners[int(chunk_results)]
    first_seed = tf.constant(_archived_sequential_chunk_seed(seed, phase_index=0, chunk_index=0), tf.int32)
    second_seed = tf.constant(_archived_sequential_chunk_seed(seed, phase_index=0, chunk_index=1), tf.int32)
    first_samples, first_trace = runner.run(state, first_seed)
    first_final = tf.cast(first_samples[-1], tf.float64)
    second_samples, second_trace = runner.run(first_final, second_seed)
    second_final = tf.cast(second_samples[-1], tf.float64)
    if runner.tracing_count != 1:
        raise NeuTraHMCError("canonical sequential XLA runner retraced")
    moved = tf.logical_and(
        _chain_moved(state, first_samples),
        _chain_moved(first_final, second_samples),
    )
    all_moved = bool(tf.reduce_all(moved).numpy())
    final_equal = bool(tf.reduce_all(tf.equal(second_final, second_samples[-1])).numpy())
    handoff = bool(tf.reduce_all(tf.equal(first_final, first_samples[-1])).numpy())
    required_trace = (
        "log_accept_ratio",
        "target_log_prob",
        "proposed_target_log_prob",
        "target_score",
        "delta_h",
    )
    trace_finite = all(
        bool(
            tf.reduce_all(
                tf.math.is_finite(tf.cast(tf.convert_to_tensor(trace[name]), tf.float64))
            ).numpy()
        )
        for trace in (first_trace, second_trace)
        for name in required_trace
    )
    samples_finite = bool(
        tf.reduce_all(tf.math.is_finite(first_samples)).numpy()
        and tf.reduce_all(tf.math.is_finite(second_samples)).numpy()
    )
    status_rows = []
    for trace, samples in ((first_trace, first_samples), (second_trace, second_samples)):
        status = _target_status_from_trace(trace)
        if status is None:
            status = _target_status(adapter, samples)
        status_rows.append(status)
    program_records: list[Mapping[str, Any]] = [
        {
            "chunk_results": int(chunk_results),
            "program_signature": runner.program_signature,
            "tracing_count": runner.tracing_count,
            "call_count": 2,
        }
    ]
    continuation_state = second_final
    for size in sizes[1:]:
        additional_runner = runners[size]
        additional_samples, additional_trace = additional_runner.run(
            continuation_state,
            tf.constant(
                _archived_sequential_chunk_seed(
                    seed, phase_index=1, chunk_index=size
                ),
                tf.int32,
            ),
        )
        all_moved = bool(
            all_moved
            and tf.reduce_all(
                _chain_moved(continuation_state, additional_samples)
            ).numpy()
        )
        samples_finite = bool(
            samples_finite
            and tf.reduce_all(tf.math.is_finite(additional_samples)).numpy()
        )
        trace_finite = bool(
            trace_finite
            and all(
                tf.reduce_all(
                    tf.math.is_finite(
                        tf.cast(tf.convert_to_tensor(additional_trace[name]), tf.float64)
                    )
                ).numpy()
                for name in required_trace
            )
        )
        status = _target_status_from_trace(additional_trace)
        if status is None:
            status = _target_status(adapter, additional_samples)
        status_rows.append(status)
        if additional_runner.tracing_count != 1:
            raise NeuTraHMCError("canonical sequential XLA runner retraced")
        program_records.append(
            {
                "chunk_results": size,
                "program_signature": additional_runner.program_signature,
                "tracing_count": additional_runner.tracing_count,
                "call_count": 1,
            }
        )
        continuation_state = tf.cast(additional_samples[-1], tf.float64)
    status_passed = bool(all(row["passed"] is True for row in status_rows))
    if not (all_moved and final_equal and handoff and trace_finite and samples_finite and status_passed):
        raise NeuTraHMCError("canonical sequential XLA mechanics qualification failed")

    code_hash = _sequential_xla_qualification_code_hash()
    evidence = {
        "schema": "bayesfilter.neutra.sequential_hmc_xla_evidence.v1",
        "status": "passed",
        "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        "adapter_signature": fixed_transport_base_adapter_signature(adapter),
        "initial_state_shape": [4, 9],
        "chunk_results": int(chunk_results),
        "program_signature": runner.program_signature,
        "qualified_programs": program_records,
        "qualification_code_hash": code_hash,
        "tracing_count": runner.tracing_count,
        "target_value_max_abs_residual": value_residual,
        "target_score_max_abs_residual": score_residual,
        "all_chains_moved": all_moved,
        "final_state_equals_last_sample": final_equal,
        "sequential_handoff_verified": handoff,
        "target_status_passed": status_passed,
        "trace_all_finite": trace_finite,
        "samples_all_finite": samples_finite,
        "seeds": [first_seed.numpy().tolist(), second_seed.numpy().tolist()],
        "nonclaims": [
            "discarded exact-program XLA qualification only",
            "no candidate, convergence, retained-sampling, or posterior claim",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    return SequentialNeuTraHMCXLAQualificationReceipt(
        status="passed",
        policy_id=NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        adapter_signature=evidence["adapter_signature"],
        initial_state_shape=(4, 9),
        chunk_results=int(chunk_results),
        program_signature=runner.program_signature,
        tracing_count=int(runner.tracing_count),
        target_value_max_abs_residual=value_residual,
        target_score_max_abs_residual=score_residual,
        all_chains_moved=all_moved,
        final_state_equals_last_sample=final_equal,
        sequential_handoff_verified=handoff,
        target_status_passed=status_passed,
        evidence_path=output.as_posix(),
        evidence_sha256=evidence_sha256,
        qualification_code_hash=code_hash,
        qualified_programs=tuple(program_records),
    )


def validate_sequential_neutra_hmc_xla_receipt(
    receipt: SequentialNeuTraHMCXLAQualificationReceipt,
    *,
    adapter: Any,
    initial_state: Any,
    config: _ArchivedSequentialNeuTraHMCConfig,
) -> None:
    """Fail closed unless a receipt binds the exact production program."""

    if not isinstance(receipt, SequentialNeuTraHMCXLAQualificationReceipt):
        raise TypeError("receipt must be SequentialNeuTraHMCXLAQualificationReceipt")
    state, _, _ = _validated_initial_state(initial_state)
    if not config.use_xla:
        raise NeuTraHMCError("sequential production requires XLA")
    required_sizes = tuple(
        dict.fromkeys((config.warmup_chunk_size, config.retained_chunk_size))
    )
    runners = {
        size: _ChunkRunner(adapter, state, config, num_results=size)
        for size in required_sizes
    }
    runner = runners[config.warmup_chunk_size]
    expected = {
        "adapter_signature": fixed_transport_base_adapter_signature(adapter),
        "initial_state_shape": tuple(int(item) for item in state.shape),
        "chunk_results": config.warmup_chunk_size,
        "program_signature": runner.program_signature,
    }
    mismatches = tuple(
        name for name, value in expected.items() if getattr(receipt, name) != value
    )
    if mismatches:
        raise NeuTraHMCError(
            "sequential XLA qualification receipt mismatch: " + ", ".join(mismatches)
        )
    qualified = {
        int(item["chunk_results"]): str(item["program_signature"])
        for item in receipt.qualified_programs
    }
    if not qualified:
        qualified = {receipt.chunk_results: receipt.program_signature}
    program_mismatches = tuple(
        size
        for size, required_runner in runners.items()
        if qualified.get(size) != required_runner.program_signature
    )
    if program_mismatches:
        raise NeuTraHMCError(
            "sequential XLA qualification program set mismatch: "
            + ", ".join(str(size) for size in program_mismatches)
        )
    if receipt.qualification_code_hash != _sequential_xla_qualification_code_hash():
        raise NeuTraHMCError("sequential XLA qualification code hash is stale")
    evidence = Path(receipt.evidence_path)
    if not evidence.is_file() or hashlib.sha256(evidence.read_bytes()).hexdigest() != receipt.evidence_sha256:
        raise NeuTraHMCError("sequential XLA qualification evidence hash mismatch")
    try:
        payload = json.loads(evidence.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NeuTraHMCError(
            "sequential XLA qualification evidence is not valid JSON"
        ) from error
    expected_payload = {
        "schema": "bayesfilter.neutra.sequential_hmc_xla_evidence.v1",
        "status": "passed",
        "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        "adapter_signature": receipt.adapter_signature,
        "initial_state_shape": list(receipt.initial_state_shape),
        "chunk_results": receipt.chunk_results,
        "program_signature": receipt.program_signature,
        "tracing_count": receipt.tracing_count,
        "qualification_code_hash": receipt.qualification_code_hash,
        "qualified_programs": [dict(item) for item in receipt.qualified_programs],
        "all_chains_moved": True,
        "final_state_equals_last_sample": True,
        "sequential_handoff_verified": True,
        "target_status_passed": True,
    }
    observed = {name: payload.get(name) for name in expected_payload}
    if observed != expected_payload:
        raise NeuTraHMCError("sequential XLA qualification evidence payload mismatch")


def load_sequential_neutra_hmc_xla_receipt(
    evidence_path: str | Path,
) -> SequentialNeuTraHMCXLAQualificationReceipt:
    """Load a hash-bound qualification receipt from its immutable evidence."""

    path = Path(evidence_path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NeuTraHMCError("sequential XLA qualification evidence is unreadable") from error
    if payload.get("schema") != "bayesfilter.neutra.sequential_hmc_xla_evidence.v1":
        raise NeuTraHMCError("sequential XLA qualification evidence schema mismatch")
    return SequentialNeuTraHMCXLAQualificationReceipt(
        status=str(payload.get("status")),
        policy_id=str(payload.get("policy_id")),
        adapter_signature=str(payload.get("adapter_signature")),
        initial_state_shape=tuple(payload.get("initial_state_shape", ())),
        chunk_results=int(payload.get("chunk_results", 0)),
        program_signature=str(payload.get("program_signature")),
        tracing_count=int(payload.get("tracing_count", 0)),
        target_value_max_abs_residual=float(
            payload.get("target_value_max_abs_residual", float("nan"))
        ),
        target_score_max_abs_residual=float(
            payload.get("target_score_max_abs_residual", float("nan"))
        ),
        all_chains_moved=payload.get("all_chains_moved") is True,
        final_state_equals_last_sample=(
            payload.get("final_state_equals_last_sample") is True
        ),
        sequential_handoff_verified=(
            payload.get("sequential_handoff_verified") is True
        ),
        target_status_passed=payload.get("target_status_passed") is True,
        evidence_path=path.as_posix(),
        evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        qualification_code_hash=str(payload.get("qualification_code_hash", "")),
        qualified_programs=tuple(payload.get("qualified_programs", ())),
    )


def _coordinate_diagnostics(
    samples: tf.Tensor, *, rhat_max: float
) -> Mapping[str, Any]:
    chain_major = tf.transpose(samples, (1, 0, 2))
    rhat = rank_normalized_split_rhat(chain_major)
    ess = rank_normalized_bulk_tail_ess(chain_major)
    maximum = tf.convert_to_tensor(rhat["maximum"], tf.float64)
    bulk = tf.convert_to_tensor(ess["bulk"], tf.float64)
    tail = tf.convert_to_tensor(ess["tail"], tf.float64)
    finite = tf.reduce_all(
        tf.math.is_finite(tf.concat((maximum, bulk, tail), axis=0))
    )
    return {
        "all_finite": bool(finite.numpy()),
        "max_rhat": float(tf.reduce_max(maximum).numpy()),
        "min_bulk_ess": float(tf.reduce_min(bulk).numpy()),
        "min_tail_ess": float(tf.reduce_min(tail).numpy()),
        "rhat_by_parameter": _tensor_tree_python(maximum),
        "bulk_ess_by_parameter": _tensor_tree_python(bulk),
        "tail_ess_by_parameter": _tensor_tree_python(tail),
        "rhat_threshold": float(rhat_max),
    }


def _mapped_model_samples(adapter: Any, samples: tf.Tensor) -> tf.Tensor:
    mapper = getattr(adapter, "latent_to_position", None)
    if not callable(mapper):
        raise NeuTraHMCError(
            "NeuTra admission requires a latent-to-model-parameter map"
        )
    if samples.shape.rank != 3 or any(dim is None for dim in samples.shape):
        raise NeuTraHMCError("samples must have static [draw, chain, parameter] shape")
    flat = tf.reshape(samples, (-1, int(samples.shape[-1])))
    mapped = tf.convert_to_tensor(mapper(flat), tf.float64)
    if mapped.shape != flat.shape:
        raise NeuTraHMCError("mapped model-parameter samples changed the sample shape")
    return tf.reshape(mapped, tf.shape(samples))


def _diagnostics(
    adapter: Any,
    samples: tf.Tensor,
    *,
    rhat_max: float,
    primary_diagnostic_coordinate: str = "maximum_over_z_and_model",
) -> Mapping[str, Any]:
    latent = _coordinate_diagnostics(samples, rhat_max=rhat_max)
    model = _coordinate_diagnostics(
        _mapped_model_samples(adapter, samples), rhat_max=rhat_max
    )
    if primary_diagnostic_coordinate == "hmc_coordinates_z":
        primary = latent
    elif primary_diagnostic_coordinate == "maximum_over_z_and_model":
        primary = {
            "all_finite": bool(latent["all_finite"] and model["all_finite"]),
            "max_rhat": max(float(latent["max_rhat"]), float(model["max_rhat"])),
            "min_bulk_ess": min(
                float(latent["min_bulk_ess"]), float(model["min_bulk_ess"])
            ),
            "min_tail_ess": min(
                float(latent["min_tail_ess"]), float(model["min_tail_ess"])
            ),
        }
    else:
        raise NeuTraHMCError("primary diagnostic coordinate is invalid")
    return {
        "hmc_coordinates": latent,
        "model_parameters": model,
        "primary_diagnostic_coordinate": primary_diagnostic_coordinate,
        "physical_coordinate_role": "explanatory_only",
        "all_finite": bool(primary["all_finite"]),
        "max_rhat": float(primary["max_rhat"]),
        "min_bulk_ess": float(primary["min_bulk_ess"]),
        "min_tail_ess": float(primary["min_tail_ess"]),
        "rhat_threshold": float(rhat_max),
    }


def _target_status(adapter: Any, samples: tf.Tensor) -> Mapping[str, Any]:
    status_method = getattr(adapter, "log_prob_and_grad_status", None)
    if not callable(status_method):
        raise NeuTraHMCError("adapter lacks combined value/score/status telemetry")
    if samples.shape.rank != 3 or any(dim is None for dim in samples.shape):
        raise NeuTraHMCError(
            "target status samples must have static [draw, chain, parameter] shape"
        )
    # Compile/evaluate small fixed batches.  The q=20 status target maps over
    # a static leading dimension; flattening a full 500-draw chunk would build
    # an unnecessarily large XLA graph, while one call per draw adds avoidable
    # host overhead.  Four transitions per chain is the reviewed compromise.
    draw_count = int(samples.shape[0])
    chain_count = int(samples.shape[1])
    flat = tf.reshape(samples, (-1, int(samples.shape[-1])))
    audit_rows = chain_count * min(4, draw_count)
    audit_batches = [
        flat[start : start + audit_rows]
        for start in range(0, int(flat.shape[0]), audit_rows)
    ]
    values_by_batch: list[tf.Tensor] = []
    scores_by_batch: list[tf.Tensor] = []
    status_by_batch: dict[str, list[tf.Tensor]] = {}
    for audit_batch in audit_batches:
        values, scores, status = status_method(audit_batch)
        if not isinstance(status, Mapping):
            raise NeuTraHMCError("target status telemetry schema mismatch")
        values_by_batch.append(tf.convert_to_tensor(values, tf.float64))
        scores_by_batch.append(tf.convert_to_tensor(scores, tf.float64))
        for key, value in status.items():
            status_by_batch.setdefault(str(key), []).append(tf.convert_to_tensor(value))
    required = (
        "status_code",
        "valid_pre_regularized_score",
        "floor_count_value",
        "min_innovation_eigenvalue",
    )
    if any(key not in status_by_batch for key in required):
        raise NeuTraHMCError("target status telemetry schema mismatch")
    combined_status = {
        key: tf.concat(values, axis=0) for key, values in status_by_batch.items()
    }
    combined_values = tf.concat(values_by_batch, axis=0)
    combined_scores = tf.concat(scores_by_batch, axis=0)
    code = tf.convert_to_tensor(combined_status["status_code"], tf.int32)
    valid = tf.convert_to_tensor(
        combined_status["valid_pre_regularized_score"], tf.bool
    )
    floors = tf.convert_to_tensor(combined_status["floor_count_value"], tf.int32)
    min_eigen = tf.convert_to_tensor(
        combined_status["min_innovation_eigenvalue"], tf.float64
    )
    condition_raw = combined_status.get("innovation_condition_estimate")
    condition = (
        None
        if condition_raw is None
        else tf.convert_to_tensor(condition_raw, tf.float64)
    )
    target_finite = tf.logical_and(
        tf.reduce_all(tf.math.is_finite(combined_values)),
        tf.reduce_all(tf.math.is_finite(combined_scores)),
    )
    status_valid = tf.reduce_all(tf.logical_and(tf.equal(code, 0), valid))
    diagnostics_finite = tf.reduce_all(tf.math.is_finite(min_eigen))
    if condition is not None:
        diagnostics_finite = tf.logical_and(
            diagnostics_finite,
            tf.reduce_all(tf.math.is_finite(condition)),
        )
    passed = tf.logical_and(target_finite, tf.logical_and(status_valid, diagnostics_finite))
    return {
        "passed": bool(passed.numpy()),
        "target_value_score_all_finite": bool(target_finite.numpy()),
        "status_nonvalid_count": int(
            tf.reduce_sum(tf.cast(tf.logical_or(tf.not_equal(code, 0), ~valid), tf.int32)).numpy()
        ),
        "floor_count_total": int(tf.reduce_sum(floors).numpy()),
        "minimum_innovation_eigenvalue": float(tf.reduce_min(min_eigen).numpy()),
        "maximum_innovation_condition_estimate": (
            None
            if condition is None
            else float(tf.reduce_max(condition).numpy())
        ),
        "innovation_condition_estimate_status": (
            "not_exposed_by_target" if condition is None else "available"
        ),
        "audit_batch_size": audit_rows,
        "audit_batch_count": len(audit_batches),
        "audited_state_count": int(samples.shape[0]) * int(samples.shape[1]),
    }


class _ChunkRunner:
    """Canonical sequential wrapper around the shared fixed-transport runner."""

    def __init__(
        self,
        adapter: Any,
        state: tf.Tensor,
        config: _ArchivedSequentialNeuTraHMCConfig,
        *,
        num_results: int | None = None,
    ):
        self.adapter = adapter
        self.config = config
        if config.xla_qualification_required:
            capability = value_score_capability(adapter)
            if not capability.is_accepted_full_chain_xla_diagnostic_authority:
                raise NeuTraHMCError(
                    "sequential production requires full-chain XLA authority"
                )
        self.num_results = int(num_results or config.warmup_chunk_size)
        policy = FixedTransportHMCPolicy.fixed(
            source=NEUTRA_SEQUENTIAL_HMC_POLICY_ID
        )
        fixed_config = FixedTransportFullChainConfig(
            num_results=self.num_results,
            num_burnin_steps=0,
            step_size=config.step_size,
            num_leapfrog_steps=config.num_leapfrog_steps,
            seed=config.seed,
            use_xla=config.use_xla,
            trace_policy="standard",
            # Only adapters exposing graph-safe per-state telemetry can trace
            # status inside XLA.  Other supported adapters are audited after
            # the fixed chunk by _target_status().
            target_status_trace_policy=(
                "per_chain_step"
                if config.target_status_required
                and callable(getattr(adapter, "target_status_telemetry", None))
                else "none"
            ),
            tuning_policy=policy,
            target_scope=str(getattr(adapter, "target_scope", "neutra_hmc")),
            chain_execution_mode="tf_function",
        )
        self._runner = build_fixed_transport_reusable_runner(
            adapter, state, fixed_config
        )
        self.program_signature = self._runner.program_signature

    @property
    def tracing_count(self) -> int | None:
        return self._runner.tracing_count

    def run(self, current_state: tf.Tensor, seed: tf.Tensor):
        result = self._runner.run(
            current_state=current_state,
            seed=seed,
            step_size=tf.constant(self.config.step_size, tf.float64),
            num_leapfrog_steps=tf.constant(
                self.config.num_leapfrog_steps, tf.int32
            ),
        )
        trace = dict(result.trace)
        telemetry = trace.pop("target_status_telemetry", None)
        if isinstance(telemetry, Mapping):
            for source, target in (
                ("status_code", "target_status_code"),
                (
                    "valid_pre_regularized_score",
                    "target_valid_pre_regularized_score",
                ),
                ("floor_count_value", "target_floor_count_value"),
                (
                    "min_innovation_eigenvalue",
                    "target_min_innovation_eigenvalue",
                ),
            ):
                if source in telemetry:
                    trace[target] = telemetry[source]
        return result.samples, trace


def _chain_moved(pre_chunk_state: tf.Tensor, samples: tf.Tensor) -> tf.Tensor:
    sequence = tf.concat((pre_chunk_state[None, ...], samples), axis=0)
    return tf.reduce_any(tf.not_equal(sequence[1:], sequence[:-1]), axis=(0, 2))


def _target_status_from_trace(trace: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if not all(
        key in trace
        for key in ("target_status_code", "target_valid_pre_regularized_score")
    ):
        return None
    code = tf.cast(tf.convert_to_tensor(trace["target_status_code"]), tf.int32)
    valid = tf.cast(
        tf.convert_to_tensor(trace["target_valid_pre_regularized_score"]), tf.bool
    )
    nonvalid = tf.logical_or(tf.not_equal(code, 0), tf.logical_not(valid))
    floors = tf.cast(
        tf.convert_to_tensor(
            trace.get("target_floor_count_value", tf.zeros_like(code))
        ),
        tf.int32,
    )
    eigen_raw = trace.get("target_min_innovation_eigenvalue")
    eigen = (
        None
        if eigen_raw is None
        else tf.cast(tf.convert_to_tensor(eigen_raw), tf.float64)
    )
    eigen_valid = (
        tf.constant(True)
        if eigen is None
        else tf.reduce_all(tf.logical_and(tf.math.is_finite(eigen), eigen > 0.0))
    )
    passed = tf.logical_and(
        tf.reduce_all(tf.logical_not(nonvalid)),
        tf.logical_and(tf.reduce_all(tf.equal(floors, 0)), eigen_valid),
    )
    return {
        "passed": bool(passed.numpy()),
        "source": "per_transition_hmc_trace",
        "status_nonvalid_count": int(
            tf.reduce_sum(tf.cast(nonvalid, tf.int32)).numpy()
        ),
        "floor_count_total": int(tf.reduce_sum(floors).numpy()),
        "minimum_innovation_eigenvalue": (
            None if eigen is None else float(tf.reduce_min(eigen).numpy())
        ),
        "audited_state_count": int(tf.size(code).numpy()),
        "target_value_score_all_finite": None,
    }


def _chunk_policy_vetoes(
    *,
    samples_finite: bool,
    log_accept_finite: bool,
    target_finite: bool,
    proposed_finite: bool,
    target_score_finite: bool,
    delta_h_finite: bool,
    target_status_passed: bool,
    chain_moved: Any,
    native_divergence_status: str,
    native_divergence_count: int | None,
) -> tuple[str, ...]:
    """Classify only the declared sequential mechanics gates."""

    vetoes = []
    if not samples_finite:
        vetoes.append("nonfinite_state")
    if not log_accept_finite:
        vetoes.append("nonfinite_log_accept_ratio")
    if not target_finite or not proposed_finite:
        vetoes.append("nonfinite_target_log_prob")
    if not target_score_finite:
        vetoes.append("nonfinite_target_score")
    if not delta_h_finite:
        vetoes.append("nonfinite_delta_h")
    if not target_status_passed:
        vetoes.append("target_status_veto")
    movement = tf.convert_to_tensor(chain_moved, dtype=tf.bool)
    if not bool(tf.reduce_all(movement).numpy()):
        vetoes.append("chain_without_movement")
    if (
        str(native_divergence_status) == "available"
        and native_divergence_count is not None
        and int(native_divergence_count) > 0
    ):
        vetoes.append("positive_native_divergence")
    return tuple(dict.fromkeys(vetoes))


def _run_archived_sequential_neutra_hmc(
    adapter: Any,
    initial_state: Any,
    config: _ArchivedSequentialNeuTraHMCConfig,
    *,
    archive_root: str | Path,
    archive_label: str,
    budget_check: Callable[[int], bool | None] | None = None,
    run_chunk: Callable[
        [tf.Tensor, tuple[int, int], _ArchivedSequentialNeuTraHMCConfig],
        tuple[Any, Mapping[str, Any]],
    ]
    | None = None,
    resume: bool = False,
) -> _ArchivedSequentialNeuTraHMCResult:
    """Run fixed-kernel sequential warm-up and retained HMC."""

    state = tf.convert_to_tensor(initial_state, tf.float64)
    if state.shape.rank != 2 or int(state.shape[0]) != config.chain_count:
        raise ValueError("initial_state must have shape [chain, parameter]")
    if any(dim is None for dim in state.shape):
        raise ValueError("initial_state must have a fully static shape")
    root = Path(archive_root)
    checkpoint_path = root / f"{archive_label}-checkpoint.json"
    manifest_path = root / f"{archive_label}-manifest.json"
    if manifest_path.exists():
        raise NeuTraHMCError("sequential HMC archive is already terminal")
    if root.exists() and any(root.iterdir()) and not resume:
        raise NeuTraHMCError("archive_root must be new or empty unless resume=True")
    if resume and not checkpoint_path.is_file():
        raise NeuTraHMCError("resume requires an existing block checkpoint")
    root.mkdir(parents=True, exist_ok=True)
    runners = (
        {}
        if run_chunk is not None
        else {
            size: _ChunkRunner(adapter, state, config, num_results=size)
            for size in {config.warmup_chunk_size, config.retained_chunk_size}
        }
    )
    started = time.perf_counter()
    phase_rows: dict[str, list[Mapping[str, Any]]] = {"warmup": [], "retained": []}
    phase_samples: dict[str, list[tf.Tensor]] = {"warmup": [], "retained": []}
    hard_vetoes: list[str] = []
    warmup_ready = False
    retained_passed = False
    warmup_count = 0
    retained_count = 0
    last_warmup_diagnostics: Mapping[str, Any] = {}
    last_retained_diagnostics: Mapping[str, Any] = {}
    next_phase = "warmup"
    next_chunk_index = 0
    resumed = False

    initial_state_payload = _tensor_tree_python(state)
    run_contract = {
        "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        "adapter_signature": fixed_transport_base_adapter_signature(adapter),
        "config": config.payload(),
        "archive_label": archive_label,
        "initial_state_shape": [int(item) for item in state.shape],
        "initial_state_hash": _payload_hash({"state": initial_state_payload}),
        "runner_programs": (
            []
            if run_chunk is not None
            else [
                {
                    "num_results": size,
                    "program_signature": runners[size].program_signature,
                }
                for size in sorted(runners)
            ]
        ),
    }
    run_contract_hash = _payload_hash(run_contract)
    if resume:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("schema") != "bayesfilter.neutra.sequential_hmc_checkpoint.v1":
            raise NeuTraHMCError("sequential HMC checkpoint schema mismatch")
        if checkpoint.get("terminal") is not False:
            raise NeuTraHMCError("sequential HMC checkpoint is terminal")
        checkpoint_contract = checkpoint.get("run_contract")
        if not isinstance(checkpoint_contract, Mapping) or _payload_hash(
            checkpoint_contract
        ) != checkpoint.get("run_contract_hash"):
            raise NeuTraHMCError("sequential HMC checkpoint contract hash is invalid")
        if checkpoint.get("run_contract_hash") != run_contract_hash:
            raise NeuTraHMCError("sequential HMC checkpoint run contract mismatch")
        rows = checkpoint.get("phase_rows")
        if not isinstance(rows, Mapping):
            raise NeuTraHMCError("sequential HMC checkpoint rows are missing")
        phase_rows = {
            phase: [dict(item) for item in rows.get(phase, ())]
            for phase in ("warmup", "retained")
        }
        for phase in ("warmup", "retained"):
            for expected_index, row in enumerate(phase_rows[phase]):
                if row.get("phase") != phase or int(row.get("chunk_index", -1)) != expected_index:
                    raise NeuTraHMCError("sequential HMC checkpoint chunk ordering mismatch")
                samples = _read_tensor_receipt(row["sample_receipt"])
                phase_samples[phase].append(tf.cast(samples, tf.float64))
                receipt_payload = _read_json_receipt(row["receipt"])
                expected_receipt_payload = {
                    key: value for key, value in row.items() if key != "receipt"
                }
                if _tensor_tree_python(receipt_payload) != _tensor_tree_python(
                    expected_receipt_payload
                ):
                    raise NeuTraHMCError(
                        "sequential HMC checkpoint receipt payload mismatch"
                    )
                for receipt in row["trace_receipts"].values():
                    _read_tensor_receipt(receipt)
        expected_paths = {checkpoint_path.resolve()}
        for phase in ("warmup", "retained"):
            for row in phase_rows[phase]:
                expected_paths.add(Path(row["sample_receipt"]["path"]).resolve())
                expected_paths.add(Path(row["receipt"]["path"]).resolve())
                expected_paths.update(
                    Path(receipt["path"]).resolve()
                    for receipt in row["trace_receipts"].values()
                )
                prefix = f"{archive_label}-{phase}-{int(row['chunk_index']):03d}"
                expected_paths.add(
                    (root / "checkpoints" / f"{prefix}-final-state.tftensor").resolve()
                )
        expected_paths.add(Path(checkpoint["final_state"]["path"]).resolve())
        observed_paths = {path.resolve() for path in root.rglob("*") if path.is_file()}
        orphan_paths = sorted(observed_paths - expected_paths)
        if orphan_paths:
            raise NeuTraHMCError(
                "sequential HMC archive contains orphan partial-block artifacts: "
                + ", ".join(path.as_posix() for path in orphan_paths)
            )
        state = tf.cast(_read_tensor_receipt(checkpoint["final_state"]), tf.float64)
        if tuple(state.shape) != tuple(int(item) for item in initial_state.shape):
            raise NeuTraHMCError("sequential HMC checkpoint state shape mismatch")
        warmup_count = int(checkpoint["warmup_results_per_chain"])
        retained_count = int(checkpoint["retained_results_per_chain"])
        warmup_ready = bool(checkpoint["warmup_ready"])
        retained_passed = bool(checkpoint["retained_passed"])
        last_warmup_diagnostics = dict(checkpoint.get("last_warmup_diagnostics", {}))
        last_retained_diagnostics = dict(checkpoint.get("last_retained_diagnostics", {}))
        next_phase = str(checkpoint["next_phase"])
        next_chunk_index = int(checkpoint["next_chunk_index"])
        if next_phase not in {"warmup", "retained"}:
            raise NeuTraHMCError("sequential HMC checkpoint next phase is invalid")
        resumed = True

    for phase_index, phase in enumerate(("warmup", "retained")):
        if phase == "warmup" and next_phase == "retained":
            continue
        maximum = (
            config.warmup_max_results if phase == "warmup" else config.retained_max_results
        )
        chunk_size = (
            config.warmup_chunk_size if phase == "warmup" else config.retained_chunk_size
        )
        chunk_count = maximum // chunk_size
        start_index = next_chunk_index if phase == next_phase else 0
        for chunk_index in range(start_index, chunk_count):
            if budget_check is not None:
                budget_allowed = budget_check(
                    chunk_size * config.num_leapfrog_steps
                )
                if budget_allowed is False:
                    hard_vetoes.append("campaign_resource_cap")
                    break
            seed = _archived_sequential_chunk_seed(
                config.seed, phase_index=phase_index, chunk_index=chunk_index
            )
            pre_chunk_state = state
            chunk_started = time.perf_counter()
            if run_chunk is None:
                runner = runners.get(chunk_size)
                if runner is None:
                    raise NeuTraHMCError("internal chunk runner is unavailable")
                samples, trace = runner.run(state, tf.constant(seed, tf.int32))
            else:
                samples, trace = run_chunk(state, seed, config)
            chunk_seconds = time.perf_counter() - chunk_started
            samples = tf.convert_to_tensor(samples, tf.float64)
            state = samples[-1]
            trace = {key: tf.convert_to_tensor(value) for key, value in trace.items()}
            moved = _chain_moved(pre_chunk_state, samples)
            samples_finite = tf.reduce_all(tf.math.is_finite(samples))
            log_accept_finite = tf.reduce_all(
                tf.math.is_finite(tf.convert_to_tensor(trace["log_accept_ratio"], tf.float64))
            )
            target_finite = tf.reduce_all(
                tf.math.is_finite(tf.convert_to_tensor(trace["target_log_prob"], tf.float64))
            )
            proposed_finite = tf.reduce_all(
                tf.math.is_finite(
                    tf.convert_to_tensor(trace["proposed_target_log_prob"], tf.float64)
                )
            )
            target_score_finite = tf.reduce_all(
                tf.math.is_finite(
                    tf.convert_to_tensor(trace["target_score"], tf.float64)
                )
            )
            delta_h = tf.convert_to_tensor(trace["delta_h"], tf.float64)
            delta_h_finite = tf.reduce_all(tf.math.is_finite(delta_h))
            delta_h_within_limit = tf.reduce_all(
                tf.abs(delta_h) <= tf.constant(config.delta_h_abs_max, tf.float64)
            )
            acceptance_probability_by_chain = tf.reduce_mean(
                tf.exp(
                    tf.minimum(
                        tf.convert_to_tensor(trace["log_accept_ratio"], tf.float64),
                        0.0,
                    )
                ),
                axis=0,
            )
            native_divergence = trace.get("divergence")
            native_divergence_status = (
                "available" if native_divergence is not None else "not_exposed_by_kernel"
            )
            native_divergence_count = (
                None
                if native_divergence is None
                else int(
                    tf.reduce_sum(
                        tf.cast(tf.convert_to_tensor(native_divergence), tf.int32)
                    ).numpy()
                )
            )
            status = (
                _target_status_from_trace(trace)
                if config.target_status_required
                else {"passed": True, "source": "not_required"}
            )
            if config.target_status_required and status is None:
                status = _target_status(adapter, samples)
            chunk_hard = list(
                _chunk_policy_vetoes(
                    samples_finite=bool(samples_finite.numpy()),
                    log_accept_finite=bool(log_accept_finite.numpy()),
                    target_finite=bool(target_finite.numpy()),
                    proposed_finite=bool(proposed_finite.numpy()),
                    target_score_finite=bool(target_score_finite.numpy()),
                    delta_h_finite=bool(delta_h_finite.numpy()),
                    target_status_passed=bool(status["passed"]),
                    chain_moved=moved,
                    native_divergence_status=native_divergence_status,
                    native_divergence_count=native_divergence_count,
                )
            )
            checkpoint_diagnostics: Mapping[str, Any] = {}
            if not chunk_hard:
                phase_samples[phase].append(samples)
                if phase == "warmup":
                    warmup_count += chunk_size
                    if warmup_count >= config.warmup_min_results:
                        combined = tf.concat(phase_samples[phase], axis=0)
                        window = combined[-config.warmup_window_results :]
                        last_warmup_diagnostics = _diagnostics(
                            adapter,
                            window,
                            rhat_max=config.warmup_rhat_max,
                            primary_diagnostic_coordinate=(
                                config.primary_diagnostic_coordinate
                            ),
                        )
                        warmup_ready = bool(
                            last_warmup_diagnostics["all_finite"]
                            and last_warmup_diagnostics["max_rhat"]
                            < config.warmup_rhat_max
                        )
                        checkpoint_diagnostics = last_warmup_diagnostics
                else:
                    retained_count += chunk_size
                    if (
                        retained_count >= config.retained_min_results
                        and (
                            retained_count == config.retained_min_results
                            or (retained_count - config.retained_min_results)
                            % config.retained_check_interval_results
                            == 0
                        )
                    ):
                        combined = tf.concat(phase_samples[phase], axis=0)
                        last_retained_diagnostics = _diagnostics(
                            adapter,
                            combined,
                            rhat_max=config.retained_rhat_max,
                            primary_diagnostic_coordinate=(
                                config.primary_diagnostic_coordinate
                            ),
                        )
                        retained_passed = bool(
                            last_retained_diagnostics["all_finite"]
                            and last_retained_diagnostics["max_rhat"]
                            < config.retained_rhat_max
                            and (
                                not config.retained_ess_required
                                or (
                                    last_retained_diagnostics["min_bulk_ess"]
                                    >= config.bulk_ess_min
                                    and last_retained_diagnostics["min_tail_ess"]
                                    >= config.tail_ess_min
                                )
                            )
                        )
                        checkpoint_diagnostics = last_retained_diagnostics
            prefix = f"{archive_label}-{phase}-{chunk_index:03d}"
            sample_receipt = _write_tensor(root / phase / f"{prefix}-samples.tftensor", samples)
            trace_receipts = {
                key: _write_tensor(root / phase / f"{prefix}-{key}.tftensor", value)
                for key, value in trace.items()
            }
            receipt_path = root / phase / f"{prefix}-receipt.json"
            row = {
                "phase": phase,
                "chunk_index": chunk_index,
                "seed": list(seed),
                "sample_receipt": sample_receipt,
                "trace_receipts": trace_receipts,
                "chunk_seconds": chunk_seconds,
                "acceptance_probability_by_chain": _tensor_tree_python(
                    acceptance_probability_by_chain
                ),
                "acceptance_bounds": [config.acceptance_min, config.acceptance_max],
                "acceptance_role": "explanatory_only_not_a_convergence_veto",
                "chain_moved": _tensor_tree_python(moved),
                "chain_movement_role": "hard_validity_veto",
                "target_status": status,
                "energy_error": {
                    "identity": "delta_h_equals_negative_log_accept_ratio",
                    "maximum_absolute_delta_h": float(
                        tf.reduce_max(tf.abs(delta_h)).numpy()
                    ),
                    "finite_tail_alert_threshold": config.delta_h_abs_max,
                    "finite_tail_role": "explanatory_alert_only",
                    "all_finite": bool(delta_h_finite.numpy()),
                    "within_alert_threshold": bool(delta_h_within_limit.numpy()),
                },
                "checkpoint_diagnostics": checkpoint_diagnostics,
                "hard_vetoes": chunk_hard,
                "native_divergence_status": native_divergence_status,
                "native_divergence_count": native_divergence_count,
                "native_divergence_interpretation": (
                    "available native boolean/count"
                    if native_divergence_status == "available"
                    else "unavailable is not zero divergences"
                ),
            }
            _write_json(receipt_path, row)
            row = {
                **row,
                "receipt": {
                    "path": receipt_path.as_posix(),
                    "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                },
            }
            phase_rows[phase].append(row)
            hard_vetoes.extend(chunk_hard)
            if not chunk_hard:
                final_state_receipt = _write_tensor(
                    root / "checkpoints" / f"{prefix}-final-state.tftensor",
                    state,
                )
                if phase == "warmup" and warmup_ready:
                    checkpoint_next_phase = "retained"
                    checkpoint_next_index = 0
                else:
                    checkpoint_next_phase = phase
                    checkpoint_next_index = chunk_index + 1
                checkpoint = {
                    "schema": "bayesfilter.neutra.sequential_hmc_checkpoint.v1",
                    "run_contract": run_contract,
                    "run_contract_hash": run_contract_hash,
                    "phase_rows": phase_rows,
                    "final_state": final_state_receipt,
                    "warmup_results_per_chain": warmup_count,
                    "retained_results_per_chain": retained_count,
                    "warmup_ready": warmup_ready,
                    "retained_passed": retained_passed,
                    "last_warmup_diagnostics": last_warmup_diagnostics,
                    "last_retained_diagnostics": last_retained_diagnostics,
                    "next_phase": checkpoint_next_phase,
                    "next_chunk_index": checkpoint_next_index,
                    "terminal": False,
                }
                _write_json_atomic(checkpoint_path, checkpoint)
            if chunk_hard:
                break
            if (phase == "warmup" and warmup_ready) or (
                phase == "retained" and retained_passed
            ):
                break
        if hard_vetoes or (phase == "warmup" and not warmup_ready):
            break

    stop_reason = (
        "hard_veto"
        if hard_vetoes
        else "warmup_cap_not_ready"
        if not warmup_ready
        else "retained_diagnostics_passed"
        if retained_passed
        else "retained_cap_not_passed"
    )
    manifest = {
        "schema": _ARCHIVED_SEQUENTIAL_NEUTRA_HMC_SCHEMA,
        "policy": config.payload(),
        "archive_label": archive_label,
        "warmup_chunks": phase_rows["warmup"],
        "retained_chunks": phase_rows["retained"],
        "warmup_excluded_from_posterior": True,
        "primary_diagnostic_coordinate": config.primary_diagnostic_coordinate,
        "physical_coordinate_role": "explanatory_only",
        "retained_ess_role": (
            "promotion_veto"
            if config.retained_ess_required
            else "explanatory_only"
        ),
        "runner_programs": (
            []
            if run_chunk is not None
            else [
                {
                    "num_results": size,
                    "program_signature": runners[size].program_signature,
                    "tracing_count": runners[size].tracing_count,
                }
                for size in sorted(runners)
            ]
        ),
        "native_divergence_status": "not_exposed_by_kernel",
        "nonclaims": [
            "native divergence unavailability is not zero divergences",
            "finite-sample operational screen only",
            "no posterior correctness or stationarity proof",
        ],
    }
    _write_json(manifest_path, manifest)
    terminal_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {
        "schema": "bayesfilter.neutra.sequential_hmc_checkpoint.v1",
        "run_contract": run_contract,
        "run_contract_hash": run_contract_hash,
        "phase_rows": phase_rows,
    }
    terminal_checkpoint.update(
        terminal=True,
        stop_reason=stop_reason,
        manifest_path=manifest_path.as_posix(),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    _write_json_atomic(checkpoint_path, terminal_checkpoint)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return _ArchivedSequentialNeuTraHMCResult(
        passed=retained_passed and not hard_vetoes,
        stop_reason=stop_reason,
        warmup_results_per_chain=warmup_count,
        retained_results_per_chain=retained_count,
        diagnostics={
            "warmup": last_warmup_diagnostics,
            "retained": last_retained_diagnostics,
            "hard_vetoes": list(dict.fromkeys(hard_vetoes)),
            "acceptance_role": "explanatory_only_not_a_convergence_veto",
            "acceptance_bounds": [config.acceptance_min, config.acceptance_max],
            "primary_diagnostic_coordinate": config.primary_diagnostic_coordinate,
            "physical_coordinate_role": "explanatory_only",
            "retained_ess_role": (
                "promotion_veto"
                if config.retained_ess_required
                else "explanatory_only"
            ),
            "finite_delta_h_tail_role": "explanatory_alert_only",
            "movement_role": "hard_validity_veto",
            "native_divergence_status": "not_exposed_by_kernel",
        },
        archive={
            "root": root.as_posix(),
            "manifest_path": manifest_path.as_posix(),
            "manifest_sha256": manifest_hash,
            "warmup_chunk_count": len(phase_rows["warmup"]),
            "retained_chunk_count": len(phase_rows["retained"]),
            "checkpoint_path": checkpoint_path.as_posix(),
            "resumed": resumed,
        },
        metadata={
            "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
            "wall_seconds": time.perf_counter() - started,
            "use_xla": config.use_xla,
            "warmup_excluded_from_posterior": True,
            "resumed": resumed,
            "run_contract_hash": run_contract_hash,
        },
    )


__all__ = [
    "NEUTRA_SEQUENTIAL_HMC_POLICY_ID",
    "NeuTraHMCError",
    "SequentialNeuTraHMCConfig",
    "SequentialNeuTraHMCXLAQualificationReceipt",
    "qualify_sequential_neutra_hmc_xla",
    "load_sequential_neutra_hmc_xla_receipt",
    "run_sequential_neutra_hmc",
    "validate_sequential_neutra_hmc_xla_receipt",
    "_ArchivedSequentialNeuTraHMCConfig",
    "_ArchivedSequentialNeuTraHMCResult",
    "_run_archived_sequential_neutra_hmc",
    "_archived_sequential_chunk_seed",
]
