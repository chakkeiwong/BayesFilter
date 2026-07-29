"""Shared TensorFlow/TFP sequential HMC controller for frozen NeuTra targets."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.batched_value_score import reviewed_value_score_target_fn
from bayesfilter.inference.hmc_convergence import (
    rank_normalized_split_rhat_summary,
)
from bayesfilter.inference.neutra_hmc_policy import (
    NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
)


MAX_RESULTS_PER_CHAIN = 10_000
DEFAULT_ENERGY_ERROR_LOG_ACCEPT_THRESHOLD = -1000.0

ArchiveCallback = Callable[..., Mapping[str, Any]]
TargetStatusSummaryCallback = Callable[[Any], Mapping[str, Any]]
RetainedDiagnosticCallback = Callable[[tf.Tensor], Mapping[str, Any]]
RetainedCheckpointCallback = Callable[[Mapping[str, Any]], None]
StopRequestedCallback = Callable[[], bool]


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
class SequentialNeuTraHMCConfig:
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


def run_sequential_neutra_hmc(
    *,
    adapter: Any,
    initial_state: Any,
    model_transform: Callable[[tf.Tensor], Any] | None = None,
    raw_transform: Callable[[tf.Tensor], Any] | None = None,
    parameter_names: Sequence[str],
    config: SequentialNeuTraHMCConfig,
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
        seed = sequential_chunk_seed(config.warmup_seed, warmup_index)
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
        seed = sequential_chunk_seed(config.retained_seed, retained_index)
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


def run_retained_neutra_hmc_continuation(
    *,
    adapter: Any,
    prefix_latent: Any,
    prefix_model: Any,
    model_transform: Callable[[tf.Tensor], Any] | None = None,
    raw_transform: Callable[[tf.Tensor], Any] | None = None,
    parameter_names: Sequence[str],
    config: SequentialNeuTraHMCConfig,
    next_chunk_index: int,
    retained_diagnostic_fn: RetainedDiagnosticCallback,
    archive_callback: ArchiveCallback | None = None,
    checkpoint_callback: RetainedCheckpointCallback | None = None,
    stop_requested_fn: StopRequestedCallback | None = None,
    target_status_summary_fn: TargetStatusSummaryCallback | None = None,
) -> Mapping[str, Any]:
    """Continue retained HMC from a verified, already-discarded-warmup prefix."""

    latent = tf.convert_to_tensor(prefix_latent, tf.float64)
    model = tf.convert_to_tensor(prefix_model, tf.float64)
    if latent.shape.rank != 3 or model.shape != latent.shape:
        raise NeuTraHMCError(
            "continuation prefixes must share [draw, chain, parameter] shape"
        )
    static_shape = latent.shape.as_list()
    if any(value is None for value in static_shape):
        raise NeuTraHMCError("continuation prefixes must have fully static shape")
    prefix_count, chain_count, dimension = (int(value) for value in static_shape)
    if prefix_count <= 0:
        raise NeuTraHMCError("continuation prefix must contain retained draws")
    if chain_count < config.minimum_chain_count:
        raise NeuTraHMCError(
            f"sequential HMC requires at least {config.minimum_chain_count} chains"
        )
    names = tuple(str(item) for item in parameter_names)
    if len(names) != dimension:
        raise NeuTraHMCError("parameter_names must match the HMC dimension")
    if not bool(tf.reduce_all(tf.math.is_finite(latent)).numpy()) or not bool(
        tf.reduce_all(tf.math.is_finite(model)).numpy()
    ):
        raise NeuTraHMCError("continuation prefixes must be finite")
    chunk_index = int(next_chunk_index)
    if chunk_index < 0:
        raise NeuTraHMCError("next_chunk_index must be non-negative")
    if prefix_count != chunk_index * config.retained_chunk_results:
        raise NeuTraHMCError(
            "continuation prefix count does not match the next fixed chunk index"
        )
    if prefix_count >= config.retained_max_results:
        raise NeuTraHMCError("continuation prefix must be below retained_max_results")
    if model_transform is not None and raw_transform is not None:
        raise NeuTraHMCError("provide only one of model_transform or raw_transform")
    transform_fn = model_transform or raw_transform or (lambda values: values)
    if not callable(retained_diagnostic_fn):
        raise NeuTraHMCError("retained_diagnostic_fn is required for continuation")

    def transform(samples: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(transform_fn(samples), tf.float64)
        if values.shape != samples.shape:
            raise NeuTraHMCError(
                "model_transform must preserve [draw, chain, parameter] shape"
            )
        return values

    state = tf.convert_to_tensor(latent[-1], tf.float64)
    programs: dict[int, Callable[[tf.Tensor, tf.Tensor], Any]] = {}
    latent_chunks = [latent]
    model_chunks = [model]
    checks: list[Mapping[str, Any]] = []
    archives: list[Mapping[str, Any]] = []
    hard_vetoes: list[str] = []
    retained_count = prefix_count
    retained_passed = False
    stopped_before_chunk = False
    started = time.monotonic()

    while retained_count < config.retained_max_results:
        if stop_requested_fn is not None and bool(stop_requested_fn()):
            stopped_before_chunk = True
            break
        active = min(
            config.retained_chunk_results,
            config.retained_max_results - retained_count,
        )
        if active not in programs:
            programs[active] = _build_batched_hmc_program(
                adapter=adapter,
                num_results=active,
                num_burnin_steps=0,
                step_size=config.step_size,
                num_leapfrog_steps=config.num_leapfrog_steps,
                jit_compile=config.jit_compile,
            )
        seed = sequential_chunk_seed(config.retained_seed, chunk_index)
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
        chunk_started = time.monotonic()
        samples, trace = programs[active](state, tf.constant(seed, tf.int32))
        chunk = _summarize_batched_hmc_output(
            initial_state=state,
            samples=samples,
            trace=trace,
            config=chunk_config,
            chain_count=chain_count,
            elapsed_seconds=time.monotonic() - chunk_started,
            target_status_summary_fn=target_status_summary_fn,
        )
        latent_samples = tf.convert_to_tensor(chunk["samples"], tf.float64)
        model_samples = transform(latent_samples)
        model_finite = bool(tf.reduce_all(tf.math.is_finite(model_samples)).numpy())
        state = latent_samples[-1]
        latent_chunks.append(latent_samples)
        model_chunks.append(model_samples)
        retained_count += active
        if archive_callback is not None:
            archives.append(
                _call_archive(
                    archive_callback,
                    stage="retained",
                    chunk_index=chunk_index,
                    latent_samples=latent_samples,
                    model_samples=model_samples,
                    seed=seed,
                    cumulative=False,
                )
            )

        diagnostic = None
        diagnostic_vetoes: tuple[str, ...] = ()
        if chunk["diagnostics"]["health_passed"] is not True:
            hard_vetoes.append("retained_continuation_chunk_health_failed")
        elif not model_finite:
            hard_vetoes.append("retained_continuation_model_samples_nonfinite")
        else:
            cumulative_model = tf.concat(model_chunks, axis=0)
            diagnostic = retained_diagnostic_fn(cumulative_model)
            if not isinstance(diagnostic, Mapping) or not isinstance(
                diagnostic.get("passed"), bool
            ):
                raise NeuTraHMCError(
                    "retained_diagnostic_fn must return a mapping with boolean passed"
                )
            diagnostic_vetoes = tuple(
                str(item) for item in diagnostic.get("hard_vetoes", ())
            )
            hard_vetoes.extend(diagnostic_vetoes)
            retained_passed = bool(
                retained_count >= config.retained_min_results
                and diagnostic["passed"]
                and not diagnostic_vetoes
            )
        check = {
            "chunk_index": chunk_index,
            "completed_results_per_chain": retained_count,
            "seed": seed,
            "health": chunk["diagnostics"],
            "model_samples_all_finite": model_finite,
            "diagnostic_role": "full_convergence",
            "full_convergence": diagnostic,
            "passed": retained_passed,
        }
        checks.append(check)
        cap_hit_at_checkpoint = bool(
            not retained_passed
            and not hard_vetoes
            and retained_count >= config.retained_max_results
        )
        if retained_passed:
            checkpoint_status = "passed"
            checkpoint_decision = "ADMIT_SEQUENTIAL_FIXED_NEUTRA_HMC_KERNEL"
        elif hard_vetoes:
            checkpoint_status = "hard_veto"
            checkpoint_decision = "REJECT_SEQUENTIAL_FIXED_NEUTRA_HMC_KERNEL"
        elif cap_hit_at_checkpoint:
            checkpoint_status = "retained_cap_reached"
            checkpoint_decision = "REJECT_SEQUENTIAL_FIXED_NEUTRA_HMC_KERNEL"
        else:
            checkpoint_status = "continuation_in_progress"
            checkpoint_decision = "INCOMPLETE_SEQUENTIAL_FIXED_NEUTRA_HMC_KERNEL"
        checkpoint = {
            "passed": bool(retained_passed and not hard_vetoes),
            "decision": checkpoint_decision,
            "completion_status": checkpoint_status,
            "retained_passed": retained_passed,
            "retained_cap_hit": cap_hit_at_checkpoint,
            "retained_results_per_chain": retained_count,
            "retained_checks": tuple(checks),
            "hard_vetoes": tuple(dict.fromkeys(hard_vetoes)),
            "retained_archives": tuple(archives),
            "last_completed_chunk_index": chunk_index,
            "elapsed_seconds": time.monotonic() - started,
            "terminal": bool(
                retained_passed or hard_vetoes or cap_hit_at_checkpoint
            ),
        }
        if checkpoint_callback is not None:
            checkpoint_callback(checkpoint)
        if retained_passed or hard_vetoes:
            break
        chunk_index += 1

    cumulative_latent = tf.concat(latent_chunks, axis=0)
    cumulative_model = tf.concat(model_chunks, axis=0)
    cumulative_archive = None
    if archive_callback is not None:
        cumulative_archive = _call_archive(
            archive_callback,
            stage="retained",
            chunk_index=None,
            latent_samples=cumulative_latent,
            model_samples=cumulative_model,
            seed=None,
            cumulative=True,
        )
    retained_cap_hit = bool(
        not retained_passed
        and not hard_vetoes
        and retained_count >= config.retained_max_results
    )
    passed = bool(retained_passed and not hard_vetoes)
    if passed:
        completion_status = "passed"
        decision = "ADMIT_SEQUENTIAL_FIXED_NEUTRA_HMC_KERNEL"
    elif hard_vetoes:
        completion_status = "hard_veto"
        decision = "REJECT_SEQUENTIAL_FIXED_NEUTRA_HMC_KERNEL"
    elif retained_cap_hit:
        completion_status = "retained_cap_reached"
        decision = "REJECT_SEQUENTIAL_FIXED_NEUTRA_HMC_KERNEL"
    else:
        completion_status = "stopped_before_chunk"
        decision = "INCOMPLETE_SEQUENTIAL_FIXED_NEUTRA_HMC_KERNEL"
    return {
        "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        "passed": passed,
        "decision": decision,
        "completion_status": completion_status,
        "retained_passed": retained_passed,
        "retained_cap_hit": retained_cap_hit,
        "stopped_before_chunk": stopped_before_chunk,
        "retained_results_per_chain": retained_count,
        "retained_check_count": len(checks),
        "retained_checks": tuple(checks),
        "hard_vetoes": tuple(dict.fromkeys(hard_vetoes)),
        "retained_archives": tuple(archives),
        "cumulative_archive": cumulative_archive,
        "elapsed_seconds": time.monotonic() - started,
        "private_retained_z": cumulative_latent,
        "private_retained_raw": cumulative_model,
    }


def sequential_chunk_seed(
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
