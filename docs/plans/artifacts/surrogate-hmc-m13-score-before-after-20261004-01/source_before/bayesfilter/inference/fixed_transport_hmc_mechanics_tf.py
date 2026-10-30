"""Shared TensorFlow/TFP mechanics for fixed-transport HMC workflows.

This module owns numerical adapter construction and one fixed-length full-chain
execution.  It deliberately owns no grid, replication, candidate nomination,
selection, refinement, confirmation, or convergence policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.mcmc import simple_step_size_adaptation as _tfp_step_size

from bayesfilter.inference.batched_value_score import (
    FixedTransportValueScoreAdapter,
    reviewed_value_score_target_fn,
)


@dataclass(frozen=True)
class FixedTransportHMCPolicy:
    """Adaptation settings consumed by one numerical transition run."""

    label: str
    adaptation_policy: str
    num_adaptation_steps: int = 0
    target_accept_prob: float | None = None
    source: str = "bayesfilter.inference.fixed_transport_hmc_mechanics_tf"

    @property
    def uses_dual_averaging(self) -> bool:
        return self.num_adaptation_steps > 0

    def payload(self) -> Mapping[str, Any]:
        return asdict(self)

    @classmethod
    def fixed(cls, *, source: str) -> "FixedTransportHMCPolicy":
        return cls(
            label="fixed_kernel_screen",
            adaptation_policy="fixed_kernel_no_adaptation",
            source=source,
        )

    @classmethod
    def dual_averaging(
        cls, *, steps: int, target: float, source: str
    ) -> "FixedTransportHMCPolicy":
        return cls(
            label="fixed_mass_dual_averaging",
            adaptation_policy="dual_averaging_step_size",
            num_adaptation_steps=int(steps),
            target_accept_prob=float(target),
            source=source,
        )


@dataclass(frozen=True)
class FixedTransportFullChainConfig:
    """Configuration for one fixed-length, rank-2 TFP HMC call."""

    num_results: int
    num_burnin_steps: int
    step_size: float
    num_leapfrog_steps: int
    seed: tuple[int, int]
    use_xla: bool
    trace_policy: str
    target_status_trace_policy: str
    tuning_policy: FixedTransportHMCPolicy
    target_scope: str
    chain_execution_mode: str
    maximum_candidate_step_size: float | None = None

    def __post_init__(self) -> None:
        cap = self.maximum_candidate_step_size
        if cap is not None:
            cap = float(cap)
            if not math.isfinite(cap) or cap <= 0.0:
                raise ValueError(
                    "maximum_candidate_step_size must be finite and positive"
                )
            object.__setattr__(self, "maximum_candidate_step_size", cap)

    @property
    def adaptation_policy(self) -> str:
        return self.tuning_policy.adaptation_policy

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            **asdict(self),
            "tuning_policy": self.tuning_policy.payload(),
            "adaptation_policy": self.adaptation_policy,
        }


RunFullChainFn = Callable[[Any, Any, FixedTransportFullChainConfig], Any]
FullChainRunHook = Callable[[Any, Any, FixedTransportFullChainConfig], None]
FullChainResultHook = Callable[[Any, Any, FixedTransportFullChainConfig, Any], None]


def fixed_transport_shared_scalar_step_size(value: Any, *, label: str) -> tf.Tensor:
    """Validate a scalar or exactly replicated positive finite step size."""

    flattened = tf.reshape(tf.cast(tf.convert_to_tensor(value), tf.float64), (-1,))
    if int(tf.size(flattened).numpy()) == 0:
        raise ValueError(f"{label} is empty")
    if not bool(tf.reduce_all(tf.math.is_finite(flattened)).numpy()):
        raise ValueError(f"{label} is nonfinite")
    if not bool(tf.reduce_all(flattened > 0.0).numpy()):
        raise ValueError(f"{label} must be positive")
    if int(tf.size(flattened).numpy()) > 1 and not bool(
        tf.reduce_all(tf.equal(flattened, flattened[0])).numpy()
    ):
        raise ValueError(f"{label} replicas disagree")
    return tf.identity(flattened[0])


def fixed_transport_capped_step_size_setter(
    maximum_candidate_step_size: float,
) -> Callable[[Any, Any], Any]:
    """Build the BayesFilter-owned dual-averaging step-size setter.

    TFP's dual-averaging kernel reports an unconstrained proposal in
    ``new_step_size`` and calls this setter immediately before the next HMC
    transition.  The cap is therefore applied at the mechanics boundary, not
    by a model client filtering candidate rows after the run.  The setter
    preserves scalar or nested/per-chain step-size structures.
    """

    cap = float(maximum_candidate_step_size)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("maximum_candidate_step_size must be finite and positive")
    cap_tensor = tf.constant(cap, dtype=tf.float64)

    def setter(kernel_results: Any, new_step_size: Any) -> Any:
        bounded = tf.nest.map_structure(
            lambda value: tf.minimum(tf.cast(value, cap_tensor.dtype), cap_tensor),
            new_step_size,
        )
        return _tfp_step_size.hmc_like_step_size_setter_fn(kernel_results, bounded)

    return setter


def fixed_transport_step_size_telemetry(
    kernel_results: Any,
    *,
    maximum_candidate_step_size: float | None,
) -> Mapping[str, Any]:
    """Return requested/applied epsilon telemetry from adaptive kernel results."""

    requested = tf.nest.map_structure(
        lambda value: tf.cast(value, tf.float64), kernel_results.new_step_size
    )
    applied = tf.nest.map_structure(
        lambda value: tf.cast(value, tf.float64),
        _tfp_step_size.hmc_like_step_size_getter_fn(kernel_results.inner_results),
    )
    requested_flat = tf.nest.flatten(requested)
    applied_flat = tf.nest.flatten(applied)
    if len(requested_flat) != len(applied_flat):
        raise ValueError("requested/applied step-size structures disagree")
    requested_max = tf.reduce_max(tf.concat([tf.reshape(v, (-1,)) for v in requested_flat], axis=0))
    applied_max = tf.reduce_max(tf.concat([tf.reshape(v, (-1,)) for v in applied_flat], axis=0))
    applied_min = tf.reduce_min(tf.concat([tf.reshape(v, (-1,)) for v in applied_flat], axis=0))
    payload: dict[str, Any] = {
        "step_size": applied,
        "requested_step_size": requested,
        "applied_step_size": applied,
        "requested_step_size_max": requested_max,
        "applied_step_size_max": applied_max,
        "applied_step_size_min": applied_min,
    }
    if maximum_candidate_step_size is None:
        payload["step_size_cap_applied"] = tf.zeros_like(applied_max, dtype=tf.bool)
    else:
        cap_tensor = tf.constant(float(maximum_candidate_step_size), tf.float64)
        payload["maximum_candidate_step_size"] = cap_tensor
        payload["step_size_cap_applied"] = requested_max > cap_tensor
        payload["step_size_cap_within_bound"] = applied_max <= cap_tensor
    return payload


def fixed_transport_terminal_step_size(trace: Mapping[str, Any]) -> tf.Tensor:
    """Return one unambiguous adapted epsilon from the terminal trace row.

    TFP may expose one scalar terminal step size or one value per chain.  The
    latter is only reducible when every replicated value is exactly equal.  In
    particular, this must not flatten the whole trace and silently select its
    last element: that can turn a per-chain disagreement into a fake scalar.
    """

    trace_key = "applied_step_size" if "applied_step_size" in trace else "step_size"
    if trace_key not in trace:
        raise ValueError("adaptation trace does not contain step_size")
    values = tf.cast(tf.convert_to_tensor(trace[trace_key]), tf.float64)
    if values.shape.rank == 0:
        terminal = values
    else:
        terminal = values[-1]
    return fixed_transport_shared_scalar_step_size(
        terminal, label="terminal adapted step_size"
    )


def _step_size_diagnostics_from_trace(
    trace: Mapping[str, Any],
    *,
    maximum_candidate_step_size: float | None,
) -> Mapping[str, Any]:
    """Summarize adaptive requested/applied epsilon telemetry outside the graph."""

    requested = trace.get("requested_step_size")
    applied = trace.get("applied_step_size", trace.get("step_size"))
    if requested is None or applied is None:
        return {
            "step_size_cap_telemetry_complete": False,
            "step_size_cap_within_bound": False if maximum_candidate_step_size is not None else None,
            "step_size_cap_applied": None,
            "requested_step_size_max": None,
            "applied_step_size_max": None,
            "applied_step_size_min": None,
        }
    requested_values = tf.reshape(tf.cast(tf.convert_to_tensor(requested), tf.float64), (-1,))
    applied_values = tf.reshape(tf.cast(tf.convert_to_tensor(applied), tf.float64), (-1,))
    requested_max = tf.reduce_max(requested_values)
    applied_max = tf.reduce_max(applied_values)
    applied_min = tf.reduce_min(applied_values)
    payload: dict[str, Any] = {
        "step_size_cap_telemetry_complete": True,
        "requested_step_size_max": float(requested_max.numpy()),
        "applied_step_size_max": float(applied_max.numpy()),
        "applied_step_size_min": float(applied_min.numpy()),
        "applied_step_size_all_finite": bool(tf.reduce_all(tf.math.is_finite(applied_values)).numpy()),
        "requested_step_size_all_finite": bool(tf.reduce_all(tf.math.is_finite(requested_values)).numpy()),
    }
    if maximum_candidate_step_size is None:
        payload["step_size_cap_within_bound"] = None
        payload["step_size_cap_applied"] = False
    else:
        cap = float(maximum_candidate_step_size)
        payload["maximum_candidate_step_size"] = cap
        payload["step_size_cap_within_bound"] = bool(
            tf.reduce_all(applied_values <= tf.constant(cap, tf.float64)).numpy()
        )
        payload["step_size_cap_applied"] = bool(
            tf.reduce_any(requested_values > tf.constant(cap, tf.float64)).numpy()
        )
    return payload


def _fixed_step_size_diagnostics(
    step_size: Any,
    *,
    maximum_candidate_step_size: float | None,
) -> Mapping[str, Any]:
    """Summarize a fixed-kernel epsilon against the optional mechanics cap."""

    value = float(tf.cast(tf.convert_to_tensor(step_size), tf.float64).numpy())
    payload: dict[str, Any] = {
        "step_size_cap_telemetry_complete": True,
        "requested_step_size_max": value,
        "applied_step_size_max": value,
        "applied_step_size_min": value,
        "applied_step_size_all_finite": bool(tf.math.is_finite(tf.constant(value, tf.float64)).numpy()),
        "requested_step_size_all_finite": bool(tf.math.is_finite(tf.constant(value, tf.float64)).numpy()),
        "step_size_cap_applied": False,
    }
    if maximum_candidate_step_size is None:
        payload["step_size_cap_within_bound"] = None
    else:
        cap = float(maximum_candidate_step_size)
        payload["maximum_candidate_step_size"] = cap
        payload["step_size_cap_within_bound"] = value <= cap
    return payload


class FixedTransportReusableRunner:
    """Reusable full-chain HMC graph with tensor-valued epsilon and ``L``.

    The state shape, trace schema, budgets, adaptation mode, and XLA choice are
    static.  Current state, stateless seed, scalar step size, and scalar
    leapfrog count are explicit graph inputs, so a campaign can run every arm
    through the same compiled object without Python-captured tuning values.
    """

    def __init__(self, adapter: Any, initial_state_template: Any, config: FixedTransportFullChainConfig):
        from bayesfilter.inference.hmc import FullChainHMCRunResult

        del FullChainHMCRunResult
        template = tf.cast(tf.convert_to_tensor(initial_state_template), tf.float64)
        if template.shape.rank != 2 or any(dim is None for dim in template.shape):
            raise ValueError("reusable fixed-transport runner requires static rank-2 state")
        self.adapter = adapter
        self.config = config
        self.state_shape = tuple(int(dim) for dim in template.shape)
        self._target = reviewed_value_score_target_fn(
            adapter, dtype=template.dtype, require_batched=True
        )
        self._runner = self._build_runner()
        self._call_count = 0
        static_config = dict(config.signature_payload())
        static_config.pop("step_size", None)
        static_config.pop("num_leapfrog_steps", None)
        static_config.pop("seed", None)
        self.program_signature = fixed_transport_stable_hash(
            {
                "schema": "bayesfilter.fixed_transport_reusable_runner.v1",
                "adapter_signature": fixed_transport_base_adapter_signature(adapter),
                "state_shape": self.state_shape,
                "state_dtype": "float64",
                "static_config": static_config,
                "dynamic_inputs": (
                    "current_state",
                    "seed",
                    "step_size",
                    "num_leapfrog_steps",
                ),
            }
        )

    def _trace_fn(self, _state: Any, kernel_results: Any) -> Mapping[str, Any]:
        adaptive = self.config.tuning_policy.uses_dual_averaging
        results = kernel_results.inner_results if adaptive else kernel_results
        trace: dict[str, Any] = {
            "is_accepted": results.is_accepted,
            "log_accept_ratio": results.log_accept_ratio,
            # For identity-mass TFP HMC, log(alpha ratio) = -Delta H.
            "delta_h": -results.log_accept_ratio,
            "target_log_prob": results.accepted_results.target_log_prob,
            "proposed_target_log_prob": results.proposed_results.target_log_prob,
            "target_score": results.accepted_results.grads_target_log_prob[0],
        }
        divergence = _native_divergence(results)
        if divergence is not None:
            trace["divergence"] = divergence
        if adaptive:
            trace.update(
                fixed_transport_step_size_telemetry(
                    kernel_results,
                    maximum_candidate_step_size=self.config.maximum_candidate_step_size,
                )
            )
        if self.config.target_status_trace_policy == "per_chain_step":
            if bool(getattr(self.adapter, "target_status_invalid_rows_become_nonfinite", False)):
                valid = tf.logical_and(
                    tf.math.is_finite(trace["target_log_prob"]),
                    tf.reduce_all(tf.math.is_finite(trace["target_score"]), axis=-1),
                )
                trace["target_status_telemetry"] = {
                    "status_code": tf.where(valid, 0, 1),
                    "valid_pre_regularized_score": valid,
                }
            else:
                trace["target_status_telemetry"] = _target_status_trace(
                    self.adapter, _state
                )
        return trace

    def _build_runner(self) -> Callable[..., Any]:
        config = self.config

        def run_chain(state: Any, seed: Any, step_size: Any, leapfrog: Any) -> Any:
            hmc = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self._target,
                step_size=step_size,
                num_leapfrog_steps=leapfrog,
            )
            kernel: Any = hmc
            if config.tuning_policy.uses_dual_averaging:
                setter = None
                if config.maximum_candidate_step_size is not None:
                    setter = fixed_transport_capped_step_size_setter(
                        config.maximum_candidate_step_size
                    )
                kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                    hmc,
                    num_adaptation_steps=config.tuning_policy.num_adaptation_steps,
                    target_accept_prob=tf.constant(
                        config.tuning_policy.target_accept_prob, dtype=state.dtype
                    ),
                    **({"step_size_setter_fn": setter} if setter is not None else {}),
                )
            return tfp.mcmc.sample_chain(
                num_results=config.num_results,
                num_burnin_steps=config.num_burnin_steps,
                current_state=state,
                kernel=kernel,
                trace_fn=self._trace_fn,
                seed=seed,
            )

        return tf.function(
            run_chain,
            input_signature=(
                tf.TensorSpec(self.state_shape, tf.float64),
                tf.TensorSpec((2,), tf.int32),
                tf.TensorSpec((), tf.float64),
                tf.TensorSpec((), tf.int32),
            ),
            jit_compile=self.config.use_xla,
            reduce_retracing=True,
        )

    @property
    def tracing_count(self) -> int | None:
        getter = getattr(self._runner, "experimental_get_tracing_count", None)
        return None if getter is None else int(getter())

    @property
    def call_count(self) -> int:
        return self._call_count

    def run(
        self,
        *,
        current_state: Any,
        seed: tuple[int, int] | Any,
        step_size: float | Any,
        num_leapfrog_steps: int | Any,
    ) -> Any:
        from bayesfilter.inference.hmc import FullChainHMCRunResult

        state = tf.cast(tf.convert_to_tensor(current_state), tf.float64)
        if tuple(state.shape) != self.state_shape:
            raise ValueError("current_state shape does not match reusable runner")
        seed_tensor = tf.convert_to_tensor(seed, tf.int32)
        step_tensor = tf.convert_to_tensor(step_size, tf.float64)
        leapfrog_tensor = tf.convert_to_tensor(num_leapfrog_steps, tf.int32)
        if seed_tensor.shape != (2,) or step_tensor.shape.rank != 0 or leapfrog_tensor.shape.rank != 0:
            raise ValueError("seed, step_size, and num_leapfrog_steps must be scalar contracts")
        step_value = float(step_tensor.numpy())
        if not bool(tf.math.is_finite(step_tensor).numpy()) or step_value <= 0.0:
            raise ValueError("step_size must be finite and positive")
        if (
            self.config.maximum_candidate_step_size is not None
            and step_value > self.config.maximum_candidate_step_size
        ):
            raise ValueError(
                "step_size exceeds maximum_candidate_step_size"
            )
        if int(leapfrog_tensor.numpy()) <= 0:
            raise ValueError("num_leapfrog_steps must be positive")
        started = time.perf_counter()
        samples, trace = self._runner(state, seed_tensor, step_tensor, leapfrog_tensor)
        elapsed = time.perf_counter() - started
        self._call_count += 1
        trace = {str(key): value for key, value in trace.items()}
        diagnostics = fixed_transport_tensor_diagnostics(samples, trace)
        if self.config.tuning_policy.uses_dual_averaging:
            terminal = fixed_transport_terminal_step_size(trace)
            diagnostics["final_step_size"] = terminal
            diagnostics["final_step_size_finite"] = tf.math.is_finite(terminal)
            diagnostics.update(
                _step_size_diagnostics_from_trace(
                    trace,
                    maximum_candidate_step_size=self.config.maximum_candidate_step_size,
                )
            )
        else:
            diagnostics.update(
                _fixed_step_size_diagnostics(
                    step_tensor,
                    maximum_candidate_step_size=self.config.maximum_candidate_step_size,
                )
            )
        diagnostics["target_accept_prob"] = self.config.tuning_policy.target_accept_prob
        diagnostics["num_adaptation_steps"] = self.config.tuning_policy.num_adaptation_steps
        diagnostics["maximum_candidate_step_size"] = self.config.maximum_candidate_step_size
        trace_count = self.tracing_count
        metadata = {
            "runtime": "tfp.mcmc.sample_chain",
            "reusable_runner": True,
            "use_xla": self.config.use_xla,
            "jit_compile": self.config.use_xla,
            "runner_call_count": self._call_count,
            "runner_trace_count": trace_count,
            "runner_program_signature": self.program_signature,
            "initial_state_shape": self.state_shape,
            "dynamic_inputs": ("current_state", "seed", "step_size", "num_leapfrog_steps"),
            "step_size_source": "runtime_tensor_argument",
            "num_leapfrog_steps_source": "runtime_tensor_argument",
            "sample_chain_call_s": elapsed,
            "tuning_policy": self.config.tuning_policy.payload(),
            "shared_mechanics_route": "bayesfilter.inference.fixed_transport_hmc_mechanics_tf",
            "nonclaims": ("candidate-discovery mechanics only", "no convergence claim"),
        }
        return FullChainHMCRunResult(samples=samples, trace=trace, diagnostics=diagnostics, metadata=metadata)

    __call__ = run


def build_fixed_transport_reusable_runner(
    adapter: Any, initial_state_template: Any, config: FixedTransportFullChainConfig
) -> FixedTransportReusableRunner:
    """Build one reusable dynamic-epsilon/dynamic-``L`` runner."""

    return FixedTransportReusableRunner(adapter, initial_state_template, config)


class FixedTransportReusableRunnerPool:
    """Cache compiled runners by the genuinely static full-chain contract.

    A tuning campaign varies the current state, stateless seed, step size, and
    leapfrog count repeatedly. Rebuilding ``tf.function`` for those values
    recompiles an otherwise identical HMC graph. This pool keeps those four
    values as tensor inputs and creates a new runner only when a static loop
    budget, adaptation policy, trace schema, XLA mode, or target changes.

    One pool is deliberately bound to one adapter object and one state shape.
    Matching signatures are not enough to reuse a traced closure across two
    independently constructed targets.
    """

    def __init__(
        self,
        *,
        before_run: FullChainRunHook | None = None,
        after_run: FullChainResultHook | None = None,
    ) -> None:
        self._before_run = before_run
        self._after_run = after_run
        self._adapter: Any | None = None
        self._adapter_signature: str | None = None
        self._state_shape: tuple[int, int] | None = None
        self._runners: dict[str, FixedTransportReusableRunner] = {}

    @staticmethod
    def _static_config_payload(
        config: FixedTransportFullChainConfig,
    ) -> Mapping[str, Any]:
        payload = dict(config.signature_payload())
        for dynamic_name in ("step_size", "num_leapfrog_steps", "seed"):
            payload.pop(dynamic_name, None)
        return payload

    def _bind_campaign(self, adapter: Any, state: tf.Tensor) -> None:
        shape = tuple(int(dim) for dim in state.shape)
        if len(shape) != 2:
            raise ValueError("reusable runner pool requires rank-2 state")
        signature = fixed_transport_base_adapter_signature(adapter)
        if self._adapter is None:
            self._adapter = adapter
            self._adapter_signature = signature
            self._state_shape = shape
            return
        if adapter is not self._adapter:
            raise ValueError(
                "reusable runner pool cannot cross adapter object boundaries"
            )
        if signature != self._adapter_signature:
            raise ValueError("reusable runner pool adapter signature changed")
        if shape != self._state_shape:
            raise ValueError("reusable runner pool state shape changed")

    def __call__(
        self,
        adapter: Any,
        initial_state: Any,
        config: FixedTransportFullChainConfig,
    ) -> Any:
        if not isinstance(config, FixedTransportFullChainConfig):
            raise TypeError("config must be FixedTransportFullChainConfig")
        if config.chain_execution_mode != "tf_function":
            raise ValueError(
                "reusable runner pool requires chain_execution_mode='tf_function'"
            )
        state = tf.cast(tf.convert_to_tensor(initial_state), tf.float64)
        self._bind_campaign(adapter, state)
        adapter_scope = getattr(adapter, "target_scope", None)
        if adapter_scope is not None and str(adapter_scope) != config.target_scope:
            raise ValueError("reusable runner pool target_scope mismatch")
        if self._before_run is not None:
            self._before_run(adapter, state, config)
        static_payload = self._static_config_payload(config)
        static_signature = fixed_transport_stable_hash(static_payload)
        runner = self._runners.get(static_signature)
        if runner is None:
            runner = build_fixed_transport_reusable_runner(adapter, state, config)
            self._runners[static_signature] = runner
        elif runner.program_signature != fixed_transport_stable_hash(
            {
                "schema": "bayesfilter.fixed_transport_reusable_runner.v1",
                "adapter_signature": self._adapter_signature,
                "state_shape": self._state_shape,
                "state_dtype": "float64",
                "static_config": static_payload,
                "dynamic_inputs": (
                    "current_state",
                    "seed",
                    "step_size",
                    "num_leapfrog_steps",
                ),
            }
        ):
            raise RuntimeError("reusable runner pool program signature mismatch")
        result = runner.run(
            current_state=state,
            seed=config.seed,
            step_size=config.step_size,
            num_leapfrog_steps=config.num_leapfrog_steps,
        )
        if self._after_run is not None:
            self._after_run(adapter, state, config, result)
        return result

    def evidence(self) -> Mapping[str, Any]:
        """Return auditable call and trace counts for every static graph."""

        runners = tuple(
            {
                "static_config_signature": static_signature,
                "program_signature": runner.program_signature,
                "call_count": runner.call_count,
                "tracing_count": runner.tracing_count,
                "static_config": self._static_config_payload(runner.config),
            }
            for static_signature, runner in sorted(self._runners.items())
        )
        tracing_counts = tuple(row["tracing_count"] for row in runners)
        return {
            "schema": "bayesfilter.fixed_transport_reusable_runner_pool.v1",
            "adapter_signature": self._adapter_signature,
            "state_shape": self._state_shape,
            "runner_count": len(runners),
            "total_call_count": sum(int(row["call_count"]) for row in runners),
            "all_runners_traced_exactly_once": bool(runners)
            and all(count == 1 for count in tracing_counts),
            "dynamic_inputs": (
                "current_state",
                "seed",
                "step_size",
                "num_leapfrog_steps",
            ),
            "runners": runners,
        }


def fixed_transport_base_adapter_signature(adapter: Any) -> str:
    """Return the stable base-target identity used by both workflows."""

    explicit = getattr(adapter, "adapter_signature", None)
    if explicit is not None:
        return str(explicit() if callable(explicit) else explicit)
    return fixed_transport_stable_hash(
        {
            "module": adapter.__class__.__module__,
            "class": adapter.__class__.__qualname__,
            "parameter_dim": int(getattr(adapter, "parameter_dim")),
        }
    )


def fixed_transport_json_ready(value: Any) -> Any:
    """Normalize fixed-transport runtime values for deterministic JSON."""

    if tf.is_tensor(value):
        return fixed_transport_json_ready(value.numpy())
    if isinstance(value, Mapping):
        return {
            str(key): fixed_transport_json_ready(item) for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [fixed_transport_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def fixed_transport_stable_hash(value: Any) -> str:
    """Hash a JSON-normalized fixed-transport lineage payload."""

    encoded = json.dumps(
        fixed_transport_json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def offset_fixed_transport_seed(
    seed: tuple[int, int], offset: int
) -> tuple[int, int]:
    """Derive a deterministic stateless seed without assigning phase policy."""

    return seed[0], seed[1] + int(offset)


def build_fixed_transport_value_score_adapter(
    *,
    base_adapter: Any,
    fixed_transport: Any,
    target_scope: str,
    evidence_path: str | None,
    xla_hmc_ready: bool,
    full_chain_xla_diagnostic_ready: bool,
) -> FixedTransportValueScoreAdapter:
    """Bind a physical target to one frozen transport without setting policy."""

    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base_adapter,
        transport=fixed_transport,
        target_scope=target_scope,
        evidence_path=evidence_path,
        xla_hmc_ready=xla_hmc_ready,
        full_chain_xla_diagnostic_ready=full_chain_xla_diagnostic_ready,
        require_batch_native=True,
    )
    adapter.target_status_invalid_rows_become_nonfinite = bool(
        getattr(base_adapter, "target_status_invalid_rows_become_nonfinite", False)
    )
    return adapter


def _native_divergence(results: Any) -> Any | None:
    for container in (
        results,
        getattr(results, "proposed_results", None),
        getattr(results, "accepted_results", None),
    ):
        if container is None:
            continue
        for name in ("is_divergent", "has_divergence", "divergence", "divergences"):
            value = getattr(container, name, None)
            if value is not None:
                return tf.cast(value, tf.bool)
    return None


def _target_status_trace(adapter: Any, state: Any) -> Mapping[str, Any]:
    telemetry = adapter.target_status_telemetry(state)
    required = ("status_code", "valid_pre_regularized_score")
    missing = tuple(name for name in required if name not in telemetry)
    if missing:
        raise ValueError("target status telemetry missing: " + ", ".join(missing))
    return {str(key): value for key, value in telemetry.items() if tf.is_tensor(value)}


def fixed_transport_tensor_diagnostics(
    samples: Any, trace: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize tensors without deciding candidate or admission policy."""

    sample_tensor = tf.cast(tf.convert_to_tensor(samples), tf.float64)
    log_accept = tf.cast(tf.convert_to_tensor(trace["log_accept_ratio"]), tf.float64)
    finite_log = tf.math.is_finite(log_accept)
    finite_values = tf.boolean_mask(log_accept, finite_log)
    max_abs = tf.cond(
        tf.size(finite_values) > 0,
        lambda: tf.reduce_max(tf.abs(finite_values)),
        lambda: tf.constant(float("nan"), tf.float64),
    )
    target = tf.cast(tf.convert_to_tensor(trace["target_log_prob"]), tf.float64)
    acceptance_probability = tf.exp(tf.minimum(log_accept, 0.0))
    acceptance = tf.reduce_mean(acceptance_probability)
    acceptance_by_chain = (
        None
        if log_accept.shape.rank is None or log_accept.shape.rank < 2
        else tf.reduce_mean(acceptance_probability, axis=0)
    )
    binary_acceptance = (
        None
        if "is_accepted" not in trace
        else tf.cast(trace["is_accepted"], tf.float64)
    )
    binary_acceptance_by_chain = (
        None
        if binary_acceptance is None
        or binary_acceptance.shape.rank is None
        or binary_acceptance.shape.rank < 2
        else tf.reduce_mean(binary_acceptance, axis=0)
    )
    proposed = trace.get("proposed_target_log_prob")
    score = trace.get("target_score")
    divergence = trace.get("divergence")
    telemetry = trace.get("target_status_telemetry")
    return {
        "acceptance_rate": float(acceptance.numpy()),
        "acceptance_rate_semantics": "mean_metropolis_acceptance_probability",
        "acceptance_probability_by_chain": (
            None
            if acceptance_by_chain is None
            else acceptance_by_chain.numpy().tolist()
        ),
        "binary_acceptance_rate": (
            None
            if binary_acceptance is None
            else float(tf.reduce_mean(binary_acceptance).numpy())
        ),
        "binary_acceptance_by_chain": (
            None
            if binary_acceptance_by_chain is None
            else binary_acceptance_by_chain.numpy().tolist()
        ),
        "samples_all_finite": bool(
            tf.reduce_all(tf.math.is_finite(sample_tensor)).numpy()
        ),
        "log_accept_ratio_finite": bool(tf.reduce_all(finite_log).numpy()),
        "target_log_prob_finite": bool(
            tf.reduce_all(tf.math.is_finite(target)).numpy()
        ),
        "proposed_target_log_prob_finite": (
            None
            if proposed is None
            else bool(
                tf.reduce_all(tf.math.is_finite(tf.cast(proposed, tf.float64))).numpy()
            )
        ),
        "target_score_finite": (
            None
            if score is None
            else bool(
                tf.reduce_all(tf.math.is_finite(tf.cast(score, tf.float64))).numpy()
            )
        ),
        "max_abs_log_accept_ratio": float(max_abs.numpy()),
        "max_abs_log_accept_energy_proxy": float(max_abs.numpy()),
        "log_accept_energy_proxy_role": "explanatory_alert_only",
        "divergence_status": (
            "available" if divergence is not None else "not_exposed_by_kernel"
        ),
        "divergence_count": (
            None
            if divergence is None
            else int(tf.reduce_sum(tf.cast(divergence, tf.int32)).numpy())
        ),
        "native_divergence_interpretation": (
            "available native boolean/count"
            if divergence is not None
            else "unavailable is not zero divergences"
        ),
        "target_status_telemetry": fixed_transport_target_status_diagnostics(
            telemetry
        ),
    }


def fixed_transport_target_status_diagnostics(
    telemetry: Any,
) -> Mapping[str, Any] | None:
    """Reduce required target-status tensors without assigning workflow roles."""

    if telemetry is None:
        return None
    code = tf.cast(tf.convert_to_tensor(telemetry["status_code"]), tf.int32)
    valid = tf.cast(
        tf.convert_to_tensor(telemetry["valid_pre_regularized_score"]), tf.bool
    )
    nonvalid = tf.logical_or(tf.not_equal(code, 0), tf.logical_not(valid))
    return {
        "telemetry_failure_veto": bool(tf.reduce_any(nonvalid).numpy()),
        "all_status_valid": bool(tf.reduce_all(tf.logical_not(nonvalid)).numpy()),
        "status_nonvalid_count": int(
            tf.reduce_sum(tf.cast(nonvalid, tf.int32)).numpy()
        ),
    }


def run_fixed_transport_full_chain_tfp_hmc(
    adapter: Any,
    initial_state: Any,
    config: FixedTransportFullChainConfig,
) -> Any:
    """Run the shared batched TFP HMC transition for tuning workflows."""

    # Import here to avoid a module cycle: the result type remains the public
    # runtime record already used by BayesFilter HMC callbacks.
    from bayesfilter.inference.hmc import FullChainHMCRunResult

    state = tf.cast(tf.convert_to_tensor(initial_state), tf.float64)
    step_value = float(config.step_size)
    if not math.isfinite(step_value) or step_value <= 0.0:
        raise ValueError("step_size must be finite and positive")
    if (
        config.maximum_candidate_step_size is not None
        and step_value > config.maximum_candidate_step_size
    ):
        raise ValueError("step_size exceeds maximum_candidate_step_size")
    target = reviewed_value_score_target_fn(adapter, dtype=state.dtype, require_batched=True)
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target,
        step_size=tf.constant(config.step_size, state.dtype),
        num_leapfrog_steps=config.num_leapfrog_steps,
    )
    kernel: Any = hmc
    if config.tuning_policy.uses_dual_averaging:
        setter = None
        if config.maximum_candidate_step_size is not None:
            setter = fixed_transport_capped_step_size_setter(
                config.maximum_candidate_step_size
            )
        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            hmc,
            num_adaptation_steps=config.tuning_policy.num_adaptation_steps,
            target_accept_prob=tf.constant(
                config.tuning_policy.target_accept_prob, state.dtype
            ),
            **({"step_size_setter_fn": setter} if setter is not None else {}),
        )

    def trace_fn(chain_state: Any, kernel_results: Any) -> Mapping[str, Any]:
        adaptive = config.tuning_policy.uses_dual_averaging
        results = kernel_results.inner_results if adaptive else kernel_results
        trace = {
            "is_accepted": results.is_accepted,
            "log_accept_ratio": results.log_accept_ratio,
            "delta_h": -results.log_accept_ratio,
            "target_log_prob": results.accepted_results.target_log_prob,
            "proposed_target_log_prob": results.proposed_results.target_log_prob,
            "target_score": results.accepted_results.grads_target_log_prob[0],
        }
        divergence = _native_divergence(results)
        if divergence is not None:
            trace["divergence"] = divergence
        if adaptive:
            trace.update(
                fixed_transport_step_size_telemetry(
                    kernel_results,
                    maximum_candidate_step_size=config.maximum_candidate_step_size,
                )
            )
        if (
            config.target_status_trace_policy == "per_chain_step"
            and bool(
                getattr(adapter, "target_status_invalid_rows_become_nonfinite", False)
            )
        ):
            valid = tf.logical_and(
                tf.math.is_finite(trace["target_log_prob"]),
                tf.reduce_all(tf.math.is_finite(trace["target_score"]), axis=-1),
            )
            trace["target_status_telemetry"] = {
                "status_code": tf.where(valid, 0, 1),
                "valid_pre_regularized_score": valid,
            }
        elif config.target_status_trace_policy == "per_chain_step":
            trace["target_status_telemetry"] = _target_status_trace(adapter, chain_state)
        return trace

    def sample() -> Any:
        return tfp.mcmc.sample_chain(
            num_results=config.num_results,
            num_burnin_steps=config.num_burnin_steps,
            current_state=state,
            kernel=kernel,
            trace_fn=trace_fn,
            seed=tf.constant(config.seed, tf.int32),
        )

    runner = sample
    if config.chain_execution_mode == "tf_function":
        runner = tf.function(sample, jit_compile=config.use_xla, reduce_retracing=True)
    started = time.perf_counter()
    samples, trace = runner()
    elapsed = time.perf_counter() - started
    trace = {str(key): value for key, value in trace.items()}
    diagnostics = fixed_transport_tensor_diagnostics(samples, trace)
    if "step_size" in trace:
        final_step = fixed_transport_terminal_step_size(trace)
        diagnostics["final_step_size"] = final_step
        diagnostics["final_step_size_finite"] = tf.math.is_finite(final_step)
    if config.tuning_policy.uses_dual_averaging:
        diagnostics.update(
            _step_size_diagnostics_from_trace(
                trace,
                maximum_candidate_step_size=config.maximum_candidate_step_size,
            )
        )
    else:
        diagnostics.update(
            _fixed_step_size_diagnostics(
                config.step_size,
                maximum_candidate_step_size=config.maximum_candidate_step_size,
            )
        )
    diagnostics["target_accept_prob"] = config.tuning_policy.target_accept_prob
    diagnostics["num_adaptation_steps"] = config.tuning_policy.num_adaptation_steps
    diagnostics["maximum_candidate_step_size"] = config.maximum_candidate_step_size
    return FullChainHMCRunResult(
        samples=samples,
        trace=trace,
        diagnostics=diagnostics,
        metadata={
            "runtime": "tfp.mcmc.sample_chain",
            "jit_compile": config.use_xla,
            "use_xla": config.use_xla,
            "chain_execution_mode": config.chain_execution_mode,
            "initial_state_shape": tuple(int(value) for value in state.shape),
            "shared_scalar_step_across_chain_bank": True,
            "sample_chain_call_s": elapsed,
            "tuning_policy": config.tuning_policy.payload(),
            "runtime_numerical_backend": "tensorflow_tfp_only",
            "shared_mechanics_route": (
                "bayesfilter.inference.fixed_transport_hmc_mechanics_tf"
            ),
        },
    )


__all__ = [
    "FixedTransportFullChainConfig",
    "FixedTransportHMCPolicy",
    "FixedTransportReusableRunnerPool",
    "FullChainResultHook",
    "FullChainRunHook",
    "RunFullChainFn",
    "build_fixed_transport_value_score_adapter",
    "fixed_transport_base_adapter_signature",
    "fixed_transport_capped_step_size_setter",
    "fixed_transport_json_ready",
    "fixed_transport_stable_hash",
    "fixed_transport_step_size_telemetry",
    "fixed_transport_target_status_diagnostics",
    "fixed_transport_tensor_diagnostics",
    "fixed_transport_shared_scalar_step_size",
    "fixed_transport_terminal_step_size",
    "FixedTransportReusableRunner",
    "build_fixed_transport_reusable_runner",
    "offset_fixed_transport_seed",
    "run_fixed_transport_full_chain_tfp_hmc",
]
