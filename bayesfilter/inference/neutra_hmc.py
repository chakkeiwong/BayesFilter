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


MAX_RESULTS_PER_CHAIN = 10_000
DEFAULT_ENERGY_ERROR_LOG_ACCEPT_THRESHOLD = -1000.0

ArchiveCallback = Callable[..., Mapping[str, Any]]
TargetStatusSummaryCallback = Callable[[Any], Mapping[str, Any]]
RetainedDiagnosticCallback = Callable[[tf.Tensor], Mapping[str, Any]]


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
    state = tf.convert_to_tensor(initial_state, dtype=tf.float64)
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
    warmup_rhat_max: float = 1.05
    retained_rhat_max: float = 1.01
    bulk_ess_min: float = 400.0
    tail_ess_min: float = 400.0
    delta_h_abs_max: float = 1000.0
    chain_count: int = 4
    use_xla: bool = True
    target_status_required: bool = True

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
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain two integers")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "use_xla", bool(self.use_xla))
        object.__setattr__(self, "target_status_required", bool(self.target_status_required))

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
        return {"schema": SEQUENTIAL_NEUTRA_HMC_SCHEMA, **asdict(self)}


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
    adapter: Any, samples: tf.Tensor, *, rhat_max: float
) -> Mapping[str, Any]:
    latent = _coordinate_diagnostics(samples, rhat_max=rhat_max)
    model = _coordinate_diagnostics(
        _mapped_model_samples(adapter, samples), rhat_max=rhat_max
    )
    return {
        "hmc_coordinates": latent,
        "model_parameters": model,
        "all_finite": bool(latent["all_finite"] and model["all_finite"]),
        "max_rhat": max(float(latent["max_rhat"]), float(model["max_rhat"])),
        "min_bulk_ess": min(
            float(latent["min_bulk_ess"]), float(model["min_bulk_ess"])
        ),
        "min_tail_ess": min(
            float(latent["min_tail_ess"]), float(model["min_tail_ess"])
        ),
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
        "innovation_condition_estimate",
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
    condition = tf.convert_to_tensor(
        combined_status["innovation_condition_estimate"], tf.float64
    )
    target_finite = tf.logical_and(
        tf.reduce_all(tf.math.is_finite(combined_values)),
        tf.reduce_all(tf.math.is_finite(combined_scores)),
    )
    status_valid = tf.reduce_all(tf.logical_and(tf.equal(code, 0), valid))
    diagnostics_finite = tf.reduce_all(
        tf.math.is_finite(tf.concat((min_eigen, condition), axis=0))
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
        "maximum_innovation_condition_estimate": float(tf.reduce_max(condition).numpy()),
        "audit_batch_size": audit_rows,
        "audit_batch_count": len(audit_batches),
        "audited_state_count": int(samples.shape[0]) * int(samples.shape[1]),
    }


class _ChunkRunner:
    def __init__(self, adapter: Any, state: tf.Tensor, config: _ArchivedSequentialNeuTraHMCConfig):
        self.adapter = adapter
        self.config = config
        self.state_shape = tuple(int(dim) for dim in state.shape)
        self.dtype = state.dtype
        target = reviewed_value_score_target_fn(adapter, dtype=self.dtype, require_batched=True)

        def run(current_state: tf.Tensor, seed: tf.Tensor):
            kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target,
                step_size=tf.constant(config.step_size, self.dtype),
                num_leapfrog_steps=config.num_leapfrog_steps,
            )

            def trace_fn(_state: Any, results: Any) -> Mapping[str, tf.Tensor]:
                return {
                    "is_accepted": results.is_accepted,
                    "log_accept_ratio": results.log_accept_ratio,
                    "target_log_prob": results.accepted_results.target_log_prob,
                    "proposed_target_log_prob": results.proposed_results.target_log_prob,
                    # For identity-mass TFP HMC, log(alpha ratio) = -Delta H.
                    "delta_h": -results.log_accept_ratio,
                }

            return tfp.mcmc.sample_chain(
                num_results=config.warmup_chunk_size,
                num_burnin_steps=0,
                current_state=current_state,
                kernel=kernel,
                trace_fn=trace_fn,
                seed=seed,
            )

        self.run = tf.function(
            run,
            input_signature=(
                tf.TensorSpec(self.state_shape, self.dtype),
                tf.TensorSpec((2,), tf.int32),
            ),
            jit_compile=config.use_xla,
            reduce_retracing=True,
        )


def _chain_moved(pre_chunk_state: tf.Tensor, samples: tf.Tensor) -> tf.Tensor:
    sequence = tf.concat((pre_chunk_state[None, ...], samples), axis=0)
    return tf.reduce_any(tf.not_equal(sequence[1:], sequence[:-1]), axis=(0, 2))


def _run_archived_sequential_neutra_hmc(
    adapter: Any,
    initial_state: Any,
    config: _ArchivedSequentialNeuTraHMCConfig,
    *,
    archive_root: str | Path,
    archive_label: str,
    budget_check: Callable[[int], bool | None] | None = None,
) -> _ArchivedSequentialNeuTraHMCResult:
    """Run fixed-kernel sequential warm-up and retained HMC."""

    state = tf.convert_to_tensor(initial_state, tf.float64)
    if state.shape.rank != 2 or int(state.shape[0]) != config.chain_count:
        raise ValueError("initial_state must have shape [chain, parameter]")
    if any(dim is None for dim in state.shape):
        raise ValueError("initial_state must have a fully static shape")
    if int(config.retained_chunk_size) != int(config.warmup_chunk_size):
        raise ValueError("current compiled controller requires equal chunk sizes")
    root = Path(archive_root)
    if root.exists() and any(root.iterdir()):
        raise NeuTraHMCError("archive_root must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    runner = _ChunkRunner(adapter, state, config)
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

    for phase_index, phase in enumerate(("warmup", "retained")):
        maximum = (
            config.warmup_max_results if phase == "warmup" else config.retained_max_results
        )
        chunk_size = (
            config.warmup_chunk_size if phase == "warmup" else config.retained_chunk_size
        )
        chunk_count = maximum // chunk_size
        for chunk_index in range(chunk_count):
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
            samples, trace = runner.run(state, tf.constant(seed, tf.int32))
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
            delta_h = tf.convert_to_tensor(trace["delta_h"], tf.float64)
            delta_h_finite = tf.reduce_all(tf.math.is_finite(delta_h))
            delta_h_within_limit = tf.reduce_all(
                tf.abs(delta_h) <= tf.constant(config.delta_h_abs_max, tf.float64)
            )
            status = _target_status(adapter, samples) if config.target_status_required else {"passed": True}
            chunk_hard = []
            if not bool(samples_finite.numpy()):
                chunk_hard.append("nonfinite_state")
            if not bool(log_accept_finite.numpy()):
                chunk_hard.append("nonfinite_log_accept_ratio")
            if not bool(target_finite.numpy()) or not bool(proposed_finite.numpy()):
                chunk_hard.append("nonfinite_target_log_prob")
            if not bool(delta_h_finite.numpy()):
                chunk_hard.append("nonfinite_delta_h")
            elif not bool(delta_h_within_limit.numpy()):
                chunk_hard.append("absolute_delta_h_above_hard_limit")
            if not bool(tf.reduce_all(moved).numpy()):
                chunk_hard.append("unmoved_chain")
            if not bool(status["passed"]):
                chunk_hard.append("target_status_veto")
            checkpoint_diagnostics: Mapping[str, Any] = {}
            if not chunk_hard:
                phase_samples[phase].append(samples)
                if phase == "warmup":
                    warmup_count += chunk_size
                    if warmup_count >= config.warmup_min_results:
                        combined = tf.concat(phase_samples[phase], axis=0)
                        window = combined[-config.warmup_window_results :]
                        last_warmup_diagnostics = _diagnostics(
                            adapter, window, rhat_max=config.warmup_rhat_max
                        )
                        warmup_ready = bool(
                            last_warmup_diagnostics["all_finite"]
                            and last_warmup_diagnostics["max_rhat"]
                            <= config.warmup_rhat_max
                        )
                        checkpoint_diagnostics = last_warmup_diagnostics
                else:
                    retained_count += chunk_size
                    if retained_count >= config.retained_min_results:
                        combined = tf.concat(phase_samples[phase], axis=0)
                        last_retained_diagnostics = _diagnostics(
                            adapter, combined, rhat_max=config.retained_rhat_max
                        )
                        retained_passed = bool(
                            last_retained_diagnostics["all_finite"]
                            and last_retained_diagnostics["max_rhat"]
                            <= config.retained_rhat_max
                            and last_retained_diagnostics["min_bulk_ess"]
                            >= config.bulk_ess_min
                            and last_retained_diagnostics["min_tail_ess"]
                            >= config.tail_ess_min
                        )
                        checkpoint_diagnostics = last_retained_diagnostics
            prefix = f"{archive_label}-{phase}-{chunk_index:03d}"
            sample_receipt = _write_tensor(root / phase / f"{prefix}-samples.tftensor", samples)
            trace_receipts = {
                key: _write_tensor(root / phase / f"{prefix}-{key}.tftensor", value)
                for key, value in trace.items()
            }
            row = {
                "phase": phase,
                "chunk_index": chunk_index,
                "seed": list(seed),
                "sample_receipt": sample_receipt,
                "trace_receipts": trace_receipts,
                "chunk_seconds": chunk_seconds,
                "acceptance_probability_by_chain": _tensor_tree_python(
                    tf.reduce_mean(
                        tf.exp(tf.minimum(tf.convert_to_tensor(trace["log_accept_ratio"], tf.float64), 0.0)),
                        axis=0,
                    )
                ),
                "chain_moved": _tensor_tree_python(moved),
                "target_status": status,
                "energy_error": {
                    "identity": "delta_h_equals_negative_log_accept_ratio",
                    "maximum_absolute_delta_h": float(
                        tf.reduce_max(tf.abs(delta_h)).numpy()
                    ),
                    "absolute_delta_h_hard_limit": config.delta_h_abs_max,
                    "all_finite": bool(delta_h_finite.numpy()),
                    "within_hard_limit": bool(delta_h_within_limit.numpy()),
                },
                "checkpoint_diagnostics": checkpoint_diagnostics,
                "hard_vetoes": chunk_hard,
                "native_divergence_status": "not_exposed_by_kernel",
            }
            _write_json(root / phase / f"{prefix}-receipt.json", row)
            phase_rows[phase].append(row)
            hard_vetoes.extend(chunk_hard)
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
        "native_divergence_status": "not_exposed_by_kernel",
        "nonclaims": [
            "native divergence unavailability is not zero divergences",
            "finite-sample operational screen only",
            "no posterior correctness or stationarity proof",
        ],
    }
    manifest_path = root / f"{archive_label}-manifest.json"
    _write_json(manifest_path, manifest)
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
            "acceptance_role": "explanatory_only",
            "native_divergence_status": "not_exposed_by_kernel",
        },
        archive={
            "root": root.as_posix(),
            "manifest_path": manifest_path.as_posix(),
            "manifest_sha256": manifest_hash,
            "warmup_chunk_count": len(phase_rows["warmup"]),
            "retained_chunk_count": len(phase_rows["retained"]),
        },
        metadata={
            "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
            "wall_seconds": time.perf_counter() - started,
            "use_xla": config.use_xla,
            "warmup_excluded_from_posterior": True,
        },
    )


__all__ = [
    "NEUTRA_SEQUENTIAL_HMC_POLICY_ID",
    "NeuTraHMCError",
    "_ArchivedSequentialNeuTraHMCConfig",
    "_ArchivedSequentialNeuTraHMCResult",
    "_run_archived_sequential_neutra_hmc",
    "_archived_sequential_chunk_seed",
]
