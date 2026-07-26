"""TensorFlow/TFP-only fixed-kernel HMC for reviewed value/score adapters."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.hmc_posterior_diagnostics import (
    posterior_mean_diagnostics,
    rank_normalized_bulk_tail_ess,
    rank_normalized_split_rhat,
)
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
_INDEPENDENT_CHAIN_PROBE_STAGES = frozenset(
    {"target", "status", "bootstrap", "one_step", "one_step_status"}
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
            "parallel_iterations": 1,
            "use_xla": False,
        }


@dataclass(frozen=True)
class NativeTFPHMCRunResult:
    """Tensor-valued samples, traces, and bounded engineering diagnostics."""

    samples: tf.Tensor
    trace: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any]
    initial_state: tf.Tensor | None = None


@dataclass(frozen=True)
class NativeTFPIndependentChainHMCConfig:
    """Static independent-chain fixed-kernel contract."""

    num_results: int
    num_burnin_steps: int
    step_size: float
    num_leapfrog_steps: int
    seed: tuple[int, int]
    chain_count: int
    target_scope: str

    def __post_init__(self) -> None:
        for name in (
            "num_results",
            "num_burnin_steps",
            "num_leapfrog_steps",
            "chain_count",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.num_results < 4 or self.num_results % 2:
            raise ValueError(
                "independent-chain HMC requires an even num_results of at least four"
            )
        if self.chain_count < 2:
            raise ValueError("independent-chain HMC requires at least two chains")
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
            "chain_count": self.chain_count,
            "target_scope": self.target_scope,
            "target_status_trace_policy": "per_chain_step_required",
            "runtime": "tfp.mcmc.sample_chain",
            "kernel": "tfp.mcmc.HamiltonianMonteCarlo",
            "target_batching": "scalar_rows_tf_while_loop",
            "adaptation_policy": "fixed_kernel_no_adaptation",
            "chain_execution_mode": "tf_function",
            "parallel_iterations": 1,
            "sample_chain_partition": (
                "one_result_reused_graph_exact_tfp_continuation_v1"
            ),
            "use_xla": False,
        }


@dataclass(frozen=True)
class NativeTFPRetainedArtifact:
    """Reloaded retained tensors and their verified manifest."""

    root: str
    initial_state: tf.Tensor
    samples: tf.Tensor
    trace: Mapping[str, Any]
    manifest: Mapping[str, Any]
    manifest_sha256: str


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
    if state.shape.rank != 1 or any(dim is None for dim in state.shape):
        raise ValueError("single-chain initial_state must have static shape [parameter]")
    target_log_prob = _reviewed_scalar_target_fn(
        adapter,
        parameter_dim=int(state.shape[0]),
        dtype=state.dtype,
    )
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
            parallel_iterations=1,
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
        "parallel_iterations": 1,
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
        initial_state=state,
    )


def _reviewed_scalar_target_fn(
    adapter: Any,
    *,
    parameter_dim: int,
    dtype: Any,
):
    """Build the scalar custom-gradient target owned by this runtime closure."""

    parameter_dim = int(parameter_dim)
    if parameter_dim <= 0:
        raise ValueError("parameter_dim must be positive")
    if not hasattr(adapter, "log_prob_and_grad"):
        raise TypeError("adapter must expose log_prob_and_grad")

    def target_value(theta: Any) -> tf.Tensor:
        positions = tf.ensure_shape(
            tf.cast(tf.convert_to_tensor(theta), dtype),
            (parameter_dim,),
        )

        @tf.custom_gradient
        def value_with_score(values: tf.Tensor) -> tuple[tf.Tensor, Any]:
            value, score = adapter.log_prob_and_grad(values)
            target = tf.ensure_shape(
                tf.cast(tf.convert_to_tensor(value), values.dtype),
                (),
            )
            target_score = tf.ensure_shape(
                tf.cast(tf.convert_to_tensor(score), values.dtype),
                (parameter_dim,),
            )

            def grad(upstream: Any) -> tf.Tensor:
                weight = tf.ensure_shape(
                    tf.cast(tf.convert_to_tensor(upstream), values.dtype),
                    (),
                )
                return weight * target_score

            return target, grad

        return value_with_score(positions)

    return target_value


def reviewed_independent_chain_target_fn(
    adapter: Any,
    *,
    chain_count: int,
    parameter_dim: int,
    dtype: Any = tf.float64,
):
    """Build a row-local value/score target without pfor or callbacks."""

    chain_count = int(chain_count)
    parameter_dim = int(parameter_dim)
    if chain_count < 2 or parameter_dim <= 0:
        raise ValueError("chain_count must be at least two and parameter_dim positive")
    if not hasattr(adapter, "log_prob_and_grad"):
        raise TypeError("adapter must expose log_prob_and_grad")

    def target_value(theta: Any) -> tf.Tensor:
        positions = tf.ensure_shape(
            tf.cast(tf.convert_to_tensor(theta), dtype),
            (chain_count, parameter_dim),
        )

        @tf.custom_gradient
        def value_with_row_scores(
            values: tf.Tensor,
        ) -> tuple[tf.Tensor, Any]:
            value_rows = tf.TensorArray(
                dtype=values.dtype,
                size=chain_count,
                element_shape=tf.TensorShape([]),
            )
            score_rows = tf.TensorArray(
                dtype=values.dtype,
                size=chain_count,
                element_shape=tf.TensorShape([parameter_dim]),
            )

            def body(index: tf.Tensor, value_ta: Any, score_ta: Any):
                value, score = adapter.log_prob_and_grad(values[index])
                return (
                    index + 1,
                    value_ta.write(index, tf.reshape(tf.cast(value, values.dtype), ())),
                    score_ta.write(
                        index,
                        tf.ensure_shape(
                            tf.cast(tf.convert_to_tensor(score), values.dtype),
                            (parameter_dim,),
                        ),
                    ),
                )

            _, value_rows, score_rows = tf.while_loop(
                lambda index, *_: index < chain_count,
                body,
                (tf.constant(0, tf.int32), value_rows, score_rows),
                parallel_iterations=1,
            )
            target = value_rows.stack()
            scores = score_rows.stack()

            def grad(upstream: Any) -> tf.Tensor:
                weights = tf.ensure_shape(
                    tf.cast(tf.convert_to_tensor(upstream), values.dtype),
                    (chain_count,),
                )
                return weights[:, tf.newaxis] * scores

            return target, grad

        return value_with_row_scores(positions)

    return target_value


def run_native_tfp_independent_chains(
    adapter: Any,
    initial_state: Any,
    config: NativeTFPIndependentChainHMCConfig,
) -> NativeTFPHMCRunResult:
    """Run fixed-kernel independent chains through one stable TFP graph."""

    if not isinstance(config, NativeTFPIndependentChainHMCConfig):
        raise TypeError("config must be NativeTFPIndependentChainHMCConfig")
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
    if state.shape.rank != 2 or any(dim is None for dim in state.shape):
        raise ValueError(
            "independent-chain initial_state must have static shape [chain, parameter]"
        )
    if int(state.shape[0]) != config.chain_count:
        raise ValueError("initial_state chain dimension does not match chain_count")
    parameter_dim = int(state.shape[1])
    target_log_prob = reviewed_independent_chain_target_fn(
        adapter,
        chain_count=config.chain_count,
        parameter_dim=parameter_dim,
        dtype=state.dtype,
    )
    trace_fn = _independent_chain_trace_fn(
        adapter,
        chain_count=config.chain_count,
    )
    tfm = tfp.mcmc
    kernel = tfm.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=tf.constant(config.step_size, state.dtype),
        num_leapfrog_steps=config.num_leapfrog_steps,
    )

    started = time.perf_counter()
    kernel_results = kernel.bootstrap_results(state)
    kernel_result_signature = tf.nest.map_structure(
        lambda value: tf.TensorSpec(value.shape, value.dtype),
        kernel_results,
    )

    @tf.function(
        input_signature=(
            tf.TensorSpec(state.shape, state.dtype),
            kernel_result_signature,
            tf.TensorSpec((2,), tf.int32),
            tf.TensorSpec((), tf.int32),
        ),
        autograph=False,
        reduce_retracing=True,
    )
    def run_segment(
        segment_state: tf.Tensor,
        segment_kernel_results: Any,
        segment_seed: tf.Tensor,
        segment_burnin_steps: tf.Tensor,
    ) -> tuple[tf.Tensor, Mapping[str, Any], Any, tf.Tensor]:
        result = tfm.sample_chain(
            num_results=1,
            num_burnin_steps=segment_burnin_steps,
            current_state=segment_state,
            previous_kernel_results=segment_kernel_results,
            kernel=kernel,
            trace_fn=trace_fn,
            return_final_kernel_results=True,
            parallel_iterations=1,
            seed=segment_seed,
        )
        next_seed = _next_sample_chain_segment_seed(
            segment_seed, segment_burnin_steps + tf.constant(1, tf.int32)
        )
        return (
            result.all_states[0],
            tf.nest.map_structure(lambda value: value[0], result.trace),
            result.final_kernel_results,
            next_seed,
        )

    segment_state = state
    segment_seed = tf.constant(config.seed, tf.int32)
    sample_rows = []
    trace_rows = []
    for result_index in range(config.num_results):
        burnin_steps = config.num_burnin_steps if result_index == 0 else 0
        segment_state, segment_trace, kernel_results, segment_seed = run_segment(
            segment_state,
            kernel_results,
            segment_seed,
            tf.constant(burnin_steps, tf.int32),
        )
        sample_rows.append(segment_state)
        trace_rows.append(segment_trace)
    samples = tf.stack(sample_rows, axis=0)
    trace = tf.nest.map_structure(
        lambda *values: tf.stack(values, axis=0), *trace_rows
    )
    elapsed = time.perf_counter() - started
    diagnostics = dict(_diagnostics(samples, trace))
    displacement = samples[1:] - samples[:-1]
    movement = tf.reduce_any(
        tf.linalg.norm(displacement, axis=-1) > tf.constant(0.0, samples.dtype),
        axis=0,
    )
    diagnostics.update(
        {
            "acceptance_rate_by_chain": tf.reduce_mean(
                tf.cast(trace["is_accepted"], tf.float64), axis=0
            ),
            "movement_by_chain": movement,
            "all_chains_moved": tf.reduce_all(movement),
        }
    )
    adapter_signature = _adapter_signature(adapter)
    capability_payload = _capability_payload(capability)
    program_signature = _independent_program_signature(
        adapter_signature=adapter_signature,
        capability_payload=capability_payload,
        config=config,
        initial_state_shape=tuple(int(dim) for dim in state.shape),
        initial_state_dtype=state.dtype.name,
        target_status_trace_source=_target_status_trace_source(adapter),
    )
    metadata = {
        "runtime": "tfp.mcmc.sample_chain",
        "kernel": "tfp.mcmc.HamiltonianMonteCarlo",
        "implementation_module": "bayesfilter.inference.native_tfp_hmc",
        "implementation_backend": "tensorflow_tensorflow_probability_only",
        "target_batching": "scalar_rows_tf_while_loop",
        "adaptation_policy": "fixed_kernel_no_adaptation",
        "chain_execution_mode": "tf_function",
        "tf_function_input_signature": (
            "state_kernel_results_seed_burnin_steps"
        ),
        "parallel_iterations": 1,
        "sample_chain_partition": (
            "one_result_reused_graph_exact_tfp_continuation_v1"
        ),
        "use_xla": False,
        "jit_compile": False,
        "sample_chain_invocation_count": config.num_results,
        "sample_chain_call_s": elapsed,
        "sample_chain_timing_role": "explanatory_only_compile_plus_execute",
        "target_status_trace_source": _target_status_trace_source(adapter),
        "trace_count": run_segment.experimental_get_tracing_count(),
        "initial_state_shape": tuple(int(dim) for dim in state.shape),
        "initial_state_dtype": state.dtype.name,
        "value_score_authority": capability.value_score_authority,
        "target_scope": capability.target_scope,
        "adapter_signature": adapter_signature,
        "capability": capability_payload,
        "program_signature": program_signature,
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
        initial_state=state,
    )


def probe_native_tfp_independent_chain_graph(
    adapter: Any,
    initial_state: Any,
    config: NativeTFPIndependentChainHMCConfig,
    *,
    stage: str,
) -> Mapping[str, Any]:
    """Trace and execute one bounded independent-chain graph stage.

    This diagnostic does not sample a chain or qualify a kernel. It isolates
    target, status, bootstrap, and one-step graph costs so a failed full-chain
    compile can be attributed without pfor, callbacks, or another sampler.
    """

    if not isinstance(config, NativeTFPIndependentChainHMCConfig):
        raise TypeError("config must be NativeTFPIndependentChainHMCConfig")
    stage = str(stage)
    if stage not in _INDEPENDENT_CHAIN_PROBE_STAGES:
        raise ValueError(
            "stage must be one of "
            + ", ".join(sorted(_INDEPENDENT_CHAIN_PROBE_STAGES))
        )
    capability = value_score_capability(adapter)
    if capability.value_score_authority not in _REVIEWED_AUTHORITIES:
        raise ValueError(
            "native TFP HMC probe requires reviewed graph value/score authority"
        )
    if capability.target_scope != config.target_scope:
        raise ValueError("value/score target_scope mismatch")
    if not callable(getattr(adapter, "target_status_telemetry", None)):
        raise TypeError(
            "native TFP HMC probe requires adapter target_status_telemetry"
        )

    state = tf.cast(tf.convert_to_tensor(initial_state), tf.float64)
    if state.shape.rank != 2 or any(dim is None for dim in state.shape):
        raise ValueError(
            "independent-chain initial_state must have static shape [chain, parameter]"
        )
    if int(state.shape[0]) != config.chain_count:
        raise ValueError("initial_state chain dimension does not match chain_count")
    parameter_dim = int(state.shape[1])
    target_log_prob = reviewed_independent_chain_target_fn(
        adapter,
        chain_count=config.chain_count,
        parameter_dim=parameter_dim,
        dtype=state.dtype,
    )
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=tf.constant(config.step_size, state.dtype),
        num_leapfrog_steps=config.num_leapfrog_steps,
    )

    @tf.function(input_signature=(), autograph=False, reduce_retracing=True)
    def target_probe() -> Mapping[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(state)
            values = target_log_prob(state)
            total = tf.reduce_sum(values)
        return {"value": values, "score": tape.gradient(total, state)}

    @tf.function(input_signature=(), autograph=False, reduce_retracing=True)
    def status_probe() -> Mapping[str, tf.Tensor]:
        return _independent_chain_status_telemetry(
            adapter, state, chain_count=config.chain_count
        )

    @tf.function(input_signature=(), autograph=False, reduce_retracing=True)
    def bootstrap_probe() -> Mapping[str, tf.Tensor]:
        results = kernel.bootstrap_results(state)
        return {
            "target_log_prob": results.accepted_results.target_log_prob,
            "proposed_target_log_prob": results.proposed_results.target_log_prob,
        }

    def one_step_outputs(*, include_status: bool) -> Mapping[str, tf.Tensor]:
        previous = kernel.bootstrap_results(state)
        next_state, results = kernel.one_step(
            state,
            previous,
            seed=tf.constant(config.seed, tf.int32),
        )
        output = {
            "next_state": next_state,
            "is_accepted": results.is_accepted,
            "log_accept_ratio": results.log_accept_ratio,
            "target_log_prob": results.accepted_results.target_log_prob,
            "proposed_target_log_prob": results.proposed_results.target_log_prob,
        }
        if include_status:
            status = _independent_chain_trace_status_telemetry(
                adapter,
                next_state,
                results.accepted_results.target_log_prob,
                chain_count=config.chain_count,
            )
            output.update(
                {f"status_{name}": value for name, value in status.items()}
            )
        return output

    @tf.function(input_signature=(), autograph=False, reduce_retracing=True)
    def one_step_probe() -> Mapping[str, tf.Tensor]:
        return one_step_outputs(include_status=False)

    @tf.function(input_signature=(), autograph=False, reduce_retracing=True)
    def one_step_status_probe() -> Mapping[str, tf.Tensor]:
        return one_step_outputs(include_status=True)

    probe = {
        "target": target_probe,
        "status": status_probe,
        "bootstrap": bootstrap_probe,
        "one_step": one_step_probe,
        "one_step_status": one_step_status_probe,
    }[stage]
    trace_started = time.perf_counter()
    concrete = probe.get_concrete_function()
    trace_elapsed = time.perf_counter() - trace_started
    execute_started = time.perf_counter()
    outputs = concrete()
    execute_elapsed = time.perf_counter() - execute_started
    finite = tuple(
        tf.reduce_all(tf.math.is_finite(tf.cast(value, tf.float64)))
        for value in outputs.values()
        if value.dtype != tf.bool
    )
    return {
        "stage": stage,
        "trace_seconds": trace_elapsed,
        "execute_seconds": execute_elapsed,
        "trace_count": probe.experimental_get_tracing_count(),
        "all_numeric_outputs_finite": tf.reduce_all(tf.stack(finite)),
        "outputs": outputs,
        "diagnostic_role": (
            "graph_attribution_only_not_sampling_tuning_or_convergence"
        ),
        "nonclaims": (
            "no retained HMC draws",
            "no kernel qualification",
            "no posterior or convergence claim",
        ),
    }


def native_tfp_rank_normalized_diagnostics(
    chain_major_samples: Any,
) -> Mapping[str, Any]:
    """Return TensorFlow/TFP rank-normalized diagnostics for retained chains."""

    samples = tf.cast(tf.convert_to_tensor(chain_major_samples), tf.float64)
    return {
        "rank_normalized_split_rhat": rank_normalized_split_rhat(samples),
        "rank_normalized_bulk_tail_ess": rank_normalized_bulk_tail_ess(samples),
        "posterior_mean": posterior_mean_diagnostics(samples),
        "diagnostic_role": "finite_sample_screen_not_convergence_proof",
        "nonclaims": (
            "finite-sample diagnostic screen only",
            "no posterior convergence proof",
            "no BGS posterior claim",
        ),
    }


def native_tfp_retained_diagnostics(
    retained: NativeTFPRetainedArtifact,
) -> Mapping[str, Any]:
    """Diagnose a verified draw-major native artifact in chain-major order."""

    if not isinstance(retained, NativeTFPRetainedArtifact):
        raise TypeError("retained must be NativeTFPRetainedArtifact")
    if retained.manifest.get("sample_layout") != "draw_chain_parameter":
        raise RuntimeError("native TFP retained sample layout mismatch")
    samples = tf.cast(tf.convert_to_tensor(retained.samples), tf.float64)
    if samples.shape.rank != 3 or any(dim is None for dim in samples.shape):
        raise RuntimeError(
            "native TFP retained samples must have static draw-chain-parameter shape"
        )
    chain_major = tf.transpose(samples, perm=(1, 0, 2))
    diagnostics = dict(native_tfp_rank_normalized_diagnostics(chain_major))
    diagnostics.update(
        {
            "source_manifest_sha256": retained.manifest_sha256,
            "source_sample_layout": "draw_chain_parameter",
            "diagnostic_sample_layout": "chain_draw_parameter",
        }
    )
    return diagnostics


def write_native_tfp_retained_artifact(
    output_dir: str | Path,
    run: NativeTFPHMCRunResult,
    *,
    adapter: Any,
    config: NativeTFPIndependentChainHMCConfig,
) -> Mapping[str, Any]:
    """Write immutable TensorFlow-serialized samples and trace tensors."""

    if not isinstance(run, NativeTFPHMCRunResult):
        raise TypeError("run must be NativeTFPHMCRunResult")
    if not isinstance(config, NativeTFPIndependentChainHMCConfig):
        raise TypeError("config must be NativeTFPIndependentChainHMCConfig")
    root = Path(output_dir).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        raise FileExistsError(f"native TFP retained artifact exists: {root}")
    adapter_signature = _adapter_signature(adapter)
    capability = value_score_capability(adapter)
    initial_state_shape = tuple(run.metadata.get("initial_state_shape", ()))
    initial_state_dtype = str(run.metadata.get("initial_state_dtype", ""))
    if len(initial_state_shape) != 2 or not initial_state_dtype:
        raise ValueError("native TFP retained run metadata is not independent-chain")
    if int(initial_state_shape[0]) != config.chain_count:
        raise ValueError("native TFP retained chain count does not match run metadata")
    expected_program_signature = _independent_program_signature(
        adapter_signature=adapter_signature,
        capability_payload=_capability_payload(capability),
        config=config,
        initial_state_shape=initial_state_shape,
        initial_state_dtype=initial_state_dtype,
        target_status_trace_source=_target_status_trace_source(adapter),
    )
    if run.metadata.get("program_signature") != expected_program_signature:
        raise ValueError("native TFP run/config program signature mismatch")
    if run.initial_state is None:
        raise ValueError("native TFP retained run is missing its initial state")
    tensors: dict[str, tf.Tensor] = {
        "initial_state": tf.convert_to_tensor(run.initial_state),
        "samples": tf.convert_to_tensor(run.samples),
    }
    if tuple(tensors["initial_state"].shape) != initial_state_shape:
        raise ValueError("native TFP retained initial state shape mismatch")
    if tensors["initial_state"].dtype.name != initial_state_dtype:
        raise ValueError("native TFP retained initial state dtype mismatch")
    expected_sample_shape = (
        config.num_results,
        config.chain_count,
        int(initial_state_shape[1]),
    )
    if tuple(tensors["samples"].shape) != expected_sample_shape:
        raise ValueError("native TFP retained sample shape does not match run config")
    _flatten_tensor_mapping("trace", run.trace, tensors)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        entries: dict[str, Any] = {}
        for logical_name, tensor in sorted(tensors.items()):
            relative = logical_name.replace("/", "__") + ".tensor"
            path = staging / relative
            tf.io.write_file(str(path), tf.io.serialize_tensor(tensor))
            entries[logical_name] = {
                "path": relative,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "dtype": tensor.dtype.name,
                "shape": tuple(int(dim) for dim in tensor.shape),
            }
        manifest = {
            "schema": "bayesfilter.native_tfp_retained_artifact.v1",
            "adapter_signature": adapter_signature,
            "program_signature": expected_program_signature,
            "config": config.signature_payload(),
            "sample_layout": "draw_chain_parameter",
            "trace_layout": "draw_chain_or_draw_chain_field",
            "tensor_count": len(entries),
            "tensors": entries,
            "nonclaims": (
                "retained tensor mechanics artifact only",
                "no posterior convergence claim",
                "no scientific validity claim",
            ),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="ascii",
        )
        manifest_sha256 = _sha256_file(manifest_path)
        staging.rename(root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "root": str(root),
        "manifest_path": str(root / "manifest.json"),
        "manifest_sha256": manifest_sha256,
        "tensor_count": len(entries),
    }


def load_native_tfp_retained_artifact(
    output_dir: str | Path,
    *,
    expected_adapter_signature: str | None = None,
    expected_program_signature: str | None = None,
) -> NativeTFPRetainedArtifact:
    """Verify and reload an immutable TensorFlow retained artifact."""

    root = Path(output_dir).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if manifest.get("schema") != "bayesfilter.native_tfp_retained_artifact.v1":
        raise RuntimeError("native TFP retained artifact schema mismatch")
    if manifest.get("sample_layout") != "draw_chain_parameter":
        raise RuntimeError("native TFP retained sample layout mismatch")
    if expected_adapter_signature is not None and manifest.get(
        "adapter_signature"
    ) != str(expected_adapter_signature):
        raise RuntimeError("native TFP retained adapter signature mismatch")
    if expected_program_signature is not None and manifest.get(
        "program_signature"
    ) != str(expected_program_signature):
        raise RuntimeError("native TFP retained program signature mismatch")
    tensors: dict[str, tf.Tensor] = {}
    entries = manifest.get("tensors")
    if not isinstance(entries, Mapping) or len(entries) != manifest.get("tensor_count"):
        raise RuntimeError("native TFP retained tensor manifest mismatch")
    expected_files = {"manifest.json"}
    for logical_name, entry in sorted(entries.items()):
        if not isinstance(entry, Mapping):
            raise RuntimeError("native TFP retained tensor entry mismatch")
        relative = str(entry["path"])
        if Path(relative).name != relative:
            raise RuntimeError("native TFP retained tensor path mismatch")
        if relative in expected_files:
            raise RuntimeError("native TFP retained tensor path collision")
        expected_files.add(relative)
        path = root / relative
        if not path.is_file() or _sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"native TFP retained tensor drift: {logical_name}")
        if path.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"native TFP retained tensor size drift: {logical_name}")
        dtype = tf.dtypes.as_dtype(entry["dtype"])
        tensor = tf.io.parse_tensor(tf.io.read_file(str(path)), out_type=dtype)
        tensors[str(logical_name)] = tf.ensure_shape(
            tensor, tuple(int(dim) for dim in entry["shape"])
        )
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    has_nonfiles = any(not path.is_file() for path in root.iterdir())
    if has_nonfiles or actual_files != expected_files:
        raise RuntimeError("native TFP retained artifact file-set drift")
    if "initial_state" not in tensors or "samples" not in tensors:
        raise RuntimeError("native TFP retained initial state or samples are missing")
    trace: dict[str, Any] = {}
    for logical_name, tensor in tensors.items():
        if logical_name in {"initial_state", "samples"}:
            continue
        _assign_tensor_path(trace, logical_name.split("/")[1:], tensor)
    return NativeTFPRetainedArtifact(
        root=str(root),
        initial_state=tensors["initial_state"],
        samples=tensors["samples"],
        trace=trace,
        manifest=manifest,
        manifest_sha256=_sha256_file(manifest_path),
    )


def _independent_chain_trace_fn(adapter: Any, *, chain_count: int):
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
        trace["target_status_telemetry"] = (
            _independent_chain_trace_status_telemetry(
                adapter,
                state,
                kernel_results.accepted_results.target_log_prob,
                chain_count=chain_count,
            )
        )
        return trace

    return trace_fn


def _independent_chain_trace_status_telemetry(
    adapter: Any,
    state: Any,
    target_log_prob: Any,
    *,
    chain_count: int,
) -> Mapping[str, tf.Tensor]:
    retained_status = getattr(adapter, "retained_target_status_telemetry", None)
    if callable(retained_status):
        return _required_target_status_fields(retained_status(target_log_prob))
    return _independent_chain_status_telemetry(
        adapter,
        state,
        chain_count=chain_count,
    )


def _next_sample_chain_segment_seed(
    seed: Any, transition_count: Any
) -> tf.Tensor:
    """Return the input seed whose salted value continues TFP's seed stream."""

    passalong = tfp.random.sanitize_seed(seed, salt="mcmc.sample_chain")
    count = tf.cast(tf.convert_to_tensor(transition_count), tf.int32)

    def body(index: tf.Tensor, current: tf.Tensor):
        _step_seed, next_seed = tfp.random.split_seed(current)
        return index + 1, next_seed

    _, passalong = tf.while_loop(
        lambda index, *_: index < count,
        body,
        (tf.constant(0, tf.int32), passalong),
        parallel_iterations=1,
    )
    # Stateless salting is an XOR fold-in, so applying the same salt gives the
    # preimage that the next sample_chain call will map back to `passalong`.
    return tfp.random.sanitize_seed(passalong, salt="mcmc.sample_chain")


def _independent_chain_status_telemetry(
    adapter: Any,
    state: Any,
    *,
    chain_count: int,
) -> Mapping[str, tf.Tensor]:
    status_codes = tf.TensorArray(tf.int32, chain_count, element_shape=())
    valid_scores = tf.TensorArray(tf.bool, chain_count, element_shape=())
    floor_counts = tf.TensorArray(tf.int32, chain_count, element_shape=())
    min_eigenvalues = tf.TensorArray(tf.float64, chain_count, element_shape=())
    condition_estimates = tf.TensorArray(tf.float64, chain_count, element_shape=())

    def body(
        index: tf.Tensor,
        status_ta: Any,
        valid_ta: Any,
        floor_ta: Any,
        eigen_ta: Any,
        condition_ta: Any,
    ):
        telemetry = adapter.target_status_telemetry(state[index])
        telemetry = _required_target_status_fields(telemetry)
        return (
            index + 1,
            status_ta.write(index, tf.cast(telemetry["status_code"], tf.int32)),
            valid_ta.write(
                index,
                tf.cast(telemetry["valid_pre_regularized_score"], tf.bool),
            ),
            floor_ta.write(
                index, tf.cast(telemetry["floor_count_value"], tf.int32)
            ),
            eigen_ta.write(
                index,
                tf.cast(telemetry["min_innovation_eigenvalue"], tf.float64),
            ),
            condition_ta.write(
                index,
                tf.cast(telemetry["innovation_condition_estimate"], tf.float64),
            ),
        )

    _, status_codes, valid_scores, floor_counts, min_eigenvalues, condition_estimates = (
        tf.while_loop(
            lambda index, *_: index < chain_count,
            body,
            (
                tf.constant(0, tf.int32),
                status_codes,
                valid_scores,
                floor_counts,
                min_eigenvalues,
                condition_estimates,
            ),
            parallel_iterations=1,
        )
    )
    return {
        "status_code": status_codes.stack(),
        "valid_pre_regularized_score": valid_scores.stack(),
        "floor_count_value": floor_counts.stack(),
        "min_innovation_eigenvalue": min_eigenvalues.stack(),
        "innovation_condition_estimate": condition_estimates.stack(),
    }


def _required_target_status_fields(
    telemetry: Mapping[str, Any],
) -> Mapping[str, Any]:
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
    return {name: telemetry[name] for name in required}


def _flatten_tensor_mapping(
    prefix: str,
    value: Mapping[str, Any],
    output: dict[str, tf.Tensor],
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("retained trace must be a tensor mapping")
    for name, item in sorted(value.items()):
        path = f"{prefix}/{name}"
        if isinstance(item, Mapping):
            _flatten_tensor_mapping(path, item, output)
        else:
            output[path] = tf.convert_to_tensor(item)


def _assign_tensor_path(
    output: dict[str, Any],
    parts: list[str],
    tensor: tf.Tensor,
) -> None:
    if not parts:
        raise RuntimeError("retained tensor path is empty")
    target = output
    for part in parts[:-1]:
        existing = target.setdefault(part, {})
        if not isinstance(existing, dict):
            raise RuntimeError("retained tensor path collides with a tensor")
        target = existing
    if parts[-1] in target:
        raise RuntimeError("duplicate retained tensor path")
    target[parts[-1]] = tensor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _capability_payload(capability: Any) -> Mapping[str, Any]:
    return {
        "value_score_authority": capability.value_score_authority,
        "runtime_backend": capability.runtime_backend,
        "target_scope": capability.target_scope,
        "xla_hmc_ready": capability.xla_hmc_ready,
        "full_chain_xla_diagnostic_ready": (
            capability.full_chain_xla_diagnostic_ready
        ),
    }


def _independent_program_signature(
    *,
    adapter_signature: str,
    capability_payload: Mapping[str, Any],
    config: NativeTFPIndependentChainHMCConfig,
    initial_state_shape: tuple[Any, ...],
    initial_state_dtype: str,
    target_status_trace_source: str,
) -> str:
    return _program_signature(
        {
            "adapter_signature": adapter_signature,
            "capability": capability_payload,
            "config": config.signature_payload(),
            "initial_state_shape": tuple(int(dim) for dim in initial_state_shape),
            "initial_state_dtype": str(initial_state_dtype),
            "target_status_trace_source": str(target_status_trace_source),
        }
    )


def _target_status_trace_source(adapter: Any) -> str:
    if callable(getattr(adapter, "retained_target_status_telemetry", None)):
        return "retained_accepted_target_log_prob"
    return "adapter_state_re_evaluation"


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
    "NativeTFPIndependentChainHMCConfig",
    "NativeTFPFixedKernelHMCConfig",
    "NativeTFPHMCRunResult",
    "NativeTFPRetainedArtifact",
    "load_native_tfp_retained_artifact",
    "native_tfp_rank_normalized_diagnostics",
    "native_tfp_retained_diagnostics",
    "reviewed_independent_chain_target_fn",
    "run_native_tfp_fixed_kernel_hmc",
    "run_native_tfp_independent_chains",
    "write_native_tfp_retained_artifact",
]
