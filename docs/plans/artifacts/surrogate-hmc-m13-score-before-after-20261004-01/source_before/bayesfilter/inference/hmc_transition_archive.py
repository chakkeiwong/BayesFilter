"""Fixed-kernel HMC continuation with complete transition instrumentation.

The runner mirrors :class:`FixedSizeHMCChunkRunner` mechanics: one TFP
``HamiltonianMonteCarlo`` kernel, a sequential ``tf.while_loop``, and one
stateless seed folded in for every transition.  It adds observational buffers
for the accepted/proposed states, scores, momenta, and Hamiltonians.  It never
adapts or tunes the kernel.

For a single tensor state in whitened HMC coordinates, the recorded energies
use ``K(p) = 0.5 * p' p`` and

``H_initial = -log_prob(pre_state) + K(initial_momentum)``

``H_proposed = -log_prob(proposed_state) + K(final_momentum)``.

The current proposal's momenta are recorded even when Metropolis rejects it.
This convention is required for exact energy-error and E-BFMI diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from bayesfilter.inference.hmc import (
    _make_tfp_target_log_prob_fn,
    _validate_full_chain_hmc_authority,
    stable_adapter_signature,
)


TRANSITION_TENSOR_KEYS = (
    "transition_index",
    "pre_state",
    "proposed_state",
    "post_state",
    "pre_target_log_prob",
    "proposed_target_log_prob",
    "post_target_log_prob",
    "pre_grad_target_log_prob",
    "proposed_grad_target_log_prob",
    "post_grad_target_log_prob",
    "is_accepted",
    "log_accept_ratio",
    "log_acceptance_correction",
    "initial_momentum",
    "final_momentum",
    "initial_kinetic_energy",
    "final_kinetic_energy",
    "initial_energy",
    "proposed_energy",
    "delta_h",
    "hamiltonian_identity_residual",
    "metropolis_seed",
    "proposal_seed",
    "effective_step_size",
    "effective_num_leapfrog_steps",
)


def _strict_scalar_integer(value: Any, *, name: str) -> int:
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be an integer scalar")
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)) or not isinstance(
        scalar,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer scalar")
    return int(scalar)


def _strict_seed(value: Any, *, name: str) -> tuple[int, int]:
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly two integer scalars") from exc
    if len(raw) != 2:
        raise ValueError(f"{name} must contain exactly two integer scalars")
    return tuple(
        _strict_scalar_integer(item, name=f"{name} item") for item in raw
    )


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


@dataclass(frozen=True)
class HMCTransitionArchiveConfig:
    """Static fixed-kernel transition archive contract."""

    max_results: int
    step_size: float
    num_leapfrog_steps: int
    master_seed: tuple[int, int]
    use_xla: bool = True
    target_scope: str | None = None
    chain_execution_mode: str = "tf_function"

    def __post_init__(self) -> None:
        max_results = _strict_scalar_integer(self.max_results, name="max_results")
        if max_results <= 0:
            raise ValueError("max_results must be positive")
        object.__setattr__(self, "max_results", max_results)
        step_size = float(self.step_size)
        if not math.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")
        object.__setattr__(self, "step_size", step_size)
        leapfrog = _strict_scalar_integer(
            self.num_leapfrog_steps,
            name="num_leapfrog_steps",
        )
        if leapfrog <= 0:
            raise ValueError("num_leapfrog_steps must be positive")
        object.__setattr__(self, "num_leapfrog_steps", leapfrog)
        seed = _strict_seed(self.master_seed, name="master_seed")
        object.__setattr__(self, "master_seed", seed)
        mode = str(self.chain_execution_mode)
        if mode not in {"tf_function", "eager"}:
            raise ValueError("chain_execution_mode must be 'tf_function' or 'eager'")
        use_xla = _strict_bool(self.use_xla, name="use_xla")
        if use_xla and mode != "tf_function":
            raise ValueError("XLA transition archive requires tf_function mode")
        object.__setattr__(self, "use_xla", use_xla)
        object.__setattr__(self, "chain_execution_mode", mode)
        if self.target_scope is not None:
            object.__setattr__(self, "target_scope", str(self.target_scope))

    def signature_payload(self) -> dict[str, Any]:
        return {
            "max_results": self.max_results,
            "num_burnin_steps": 0,
            "step_size": self.step_size,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "master_seed": self.master_seed,
            "seed_policy": "stateless_fold_in_master_seed_absolute_transition_index",
            "use_xla": self.use_xla,
            "chain_execution_mode": self.chain_execution_mode,
            "target_scope": self.target_scope,
            "adaptation_policy": "none",
            "tuning_policy": "none",
        }


@dataclass(frozen=True)
class HMCTransitionArchiveRunResult:
    """Tensor-valued full transition record and continuation handoff."""

    tensors: Mapping[str, Any]
    valid_mask: Any
    initial_state: Any
    final_state: Any
    diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class HMCTransitionShard:
    """Read-back record for one immutable transition shard."""

    path: str
    sha256: str
    metadata: Mapping[str, Any]
    tensors: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class HMCExactMechanicsIdentityPolicy:
    """Mixed absolute-relative tolerance for binary64 mechanics identities."""

    atol: float = 2.0e-10
    rtol_multiplier: float = 64.0
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.dtype != "float64":
            raise ValueError("exact mechanics currently supports only tested float64")
        atol = float(self.atol)
        multiplier = float(self.rtol_multiplier)
        if not np.isfinite(atol) or atol < 0.0:
            raise ValueError("identity atol must be finite and nonnegative")
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("identity rtol multiplier must be positive and finite")
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "rtol_multiplier", multiplier)

    @property
    def rtol(self) -> float:
        return self.rtol_multiplier * float(np.finfo(np.float64).eps)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_exact_mechanics_identity_policy.v1",
            "dtype": self.dtype,
            "atol": self.atol,
            "rtol": self.rtol,
            "rtol_multiplier": self.rtol_multiplier,
            "scale_definition": "max(1, abs(left), abs(right))",
            "pass_rule": "finite_operands_and_abs_residual_le_allowed_error",
        }


def summarize_hmc_exact_mechanics_identity(
    left: Any,
    right: Any,
    *,
    identity_name: str,
    policy: HMCExactMechanicsIdentityPolicy | None = None,
) -> Mapping[str, Any]:
    """Return a fail-closed worst-case summary for one elementwise identity."""

    active_policy = HMCExactMechanicsIdentityPolicy() if policy is None else policy
    if not isinstance(active_policy, HMCExactMechanicsIdentityPolicy):
        raise TypeError("identity policy has invalid type")
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.dtype != np.dtype(active_policy.dtype) or right_array.dtype != np.dtype(
        active_policy.dtype
    ):
        raise ValueError("exact mechanics operands must use the tested float64 dtype")
    if left_array.shape != right_array.shape:
        raise ValueError("exact mechanics operands must have identical shapes")
    if left_array.size == 0:
        raise ValueError("exact mechanics identity cannot evaluate empty operands")
    finite = np.isfinite(left_array) & np.isfinite(right_array)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        residual = left_array - right_array
        absolute = np.abs(residual)
        scale = np.maximum(1.0, np.maximum(np.abs(left_array), np.abs(right_array)))
        allowed = active_policy.atol + active_policy.rtol * scale
        scaled = absolute / allowed
    if np.all(finite):
        worst_flat = int(np.argmax(scaled))
        passed = bool(np.all(scaled <= 1.0))
        numeric_summary_available = True
    else:
        worst_flat = int(np.flatnonzero(~finite)[0])
        passed = False
        numeric_summary_available = False
    index = tuple(int(item) for item in np.unravel_index(worst_flat, left_array.shape))
    numeric = (
        {
            "max_absolute_residual": float(absolute[index]),
            "max_scaled_residual": float(scaled[index]),
            "worst_left_operand": float(left_array[index]),
            "worst_right_operand": float(right_array[index]),
            "worst_residual": float(residual[index]),
            "worst_scale": float(scale[index]),
            "worst_allowed_error": float(allowed[index]),
        }
        if numeric_summary_available
        else {
            "max_absolute_residual": None,
            "max_scaled_residual": None,
            "worst_left_operand": None,
            "worst_right_operand": None,
            "worst_residual": None,
            "worst_scale": None,
            "worst_allowed_error": None,
        }
    )
    return {
        "schema": "bayesfilter.hmc_exact_mechanics_diagnostic.v1",
        "identity_name": str(identity_name),
        "policy": active_policy.payload(),
        "shape": tuple(int(item) for item in left_array.shape),
        "element_count": int(left_array.size),
        "finite_operand_count": int(np.sum(finite)),
        "all_operands_finite": bool(np.all(finite)),
        "passed": passed,
        "numeric_summary_available": numeric_summary_available,
        "worst_index": index,
        "worst_left_operand_finite": bool(np.isfinite(left_array[index])),
        "worst_right_operand_finite": bool(np.isfinite(right_array[index])),
        **numeric,
    }


def _single_state_part(value: Any, *, name: str) -> Any:
    if not isinstance(value, (tuple, list)) or len(value) != 1:
        raise ValueError(f"TFP {name} must contain exactly one state part")
    return value[0]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_safe(value.numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class HMCTransitionArchiveRunner:
    """Reusable fixed-shape, absolute-index-seeded HMC transition runner."""

    def __init__(
        self,
        adapter: Any,
        initial_state_template: Any,
        config: HMCTransitionArchiveConfig,
    ) -> None:
        import tensorflow as tf

        self.adapter = adapter
        self.config = config
        self.capability = _validate_full_chain_hmc_authority(adapter, config)
        template = tf.cast(tf.convert_to_tensor(initial_state_template), tf.float64)
        if template.shape.rank != 2:
            raise ValueError(
                "transition archive requires rank-2 [chain, parameter] state"
            )
        if any(dim is None for dim in template.shape):
            raise ValueError("transition archive requires fully static state shape")
        if int(template.shape[0]) < 1 or int(template.shape[1]) < 1:
            raise ValueError("transition archive state dimensions must be positive")
        self._state_shape = tuple(int(dim) for dim in template.shape)
        self._dtype = template.dtype
        self._initial_state_template = template
        self._target_log_prob = _make_tfp_target_log_prob_fn(
            adapter, dtype=self._dtype
        )
        build_started = time.perf_counter()
        self._runner = self._build_runner()
        self._runner_build_seconds = time.perf_counter() - build_started
        self._call_count = 0
        self._first_call_seconds: float | None = None
        self._warm_call_seconds: float | None = None

    @property
    def state_shape(self) -> tuple[int, int]:
        return self._state_shape

    @property
    def call_count(self) -> int:
        return self._call_count

    def run(
        self,
        *,
        active_results: int | Any,
        global_start_index: int | Any,
        current_state: Any | None = None,
    ) -> HMCTransitionArchiveRunResult:
        """Run one continuation block and return every transition field."""

        import tensorflow as tf

        state = self._initial_state_template if current_state is None else current_state
        state_tensor = tf.convert_to_tensor(state, dtype=self._dtype)
        if tuple(state_tensor.shape.as_list()) != self._state_shape:
            raise ValueError("current_state shape must match the runner template")
        active_value = _strict_scalar_integer(
            active_results,
            name="active_results",
        )
        start_value = _strict_scalar_integer(
            global_start_index,
            name="global_start_index",
        )
        if active_value <= 0 or active_value > self.config.max_results:
            raise ValueError(
                "active_results must satisfy 1 <= active_results <= max_results"
            )
        if start_value < 0:
            raise ValueError("global_start_index must be nonnegative")
        active = tf.convert_to_tensor(active_value, dtype=tf.int32)
        start = tf.convert_to_tensor(start_value, dtype=tf.int32)

        call_started = time.perf_counter()
        tensors, valid_mask, final_state = self._runner(
            state_tensor, active, start
        )
        call_seconds = time.perf_counter() - call_started
        self._call_count += 1
        if self._call_count == 1:
            self._first_call_seconds = call_seconds
        else:
            self._warm_call_seconds = call_seconds
        diagnostics = self._diagnostics(tensors, valid_mask, state_tensor, final_state)
        return HMCTransitionArchiveRunResult(
            tensors=tensors,
            valid_mask=valid_mask,
            initial_state=state_tensor,
            final_state=final_state,
            diagnostics=diagnostics,
            metadata=self._metadata(
                active_results=active_value,
                global_start_index=start_value,
                call_seconds=call_seconds,
            ),
        )

    __call__ = run

    def _build_runner(
        self,
    ) -> Callable[[Any, Any, Any], tuple[Mapping[str, Any], Any, Any]]:
        import tensorflow as tf
        import tensorflow_probability as tfp

        config = self.config
        target_log_prob = self._target_log_prob
        state_shape = self._state_shape
        chain_count = state_shape[0]
        parameter_dim = state_shape[1]
        master_seed = tf.constant(config.master_seed, dtype=tf.int32)

        def run_block(
            current_state: Any,
            active_results: Any,
            global_start_index: Any,
        ) -> tuple[Mapping[str, Any], Any, Any]:
            active_results = tf.cast(active_results, tf.int32)
            global_start_index = tf.cast(global_start_index, tf.int32)
            with tf.control_dependencies(
                (
                    tf.debugging.assert_greater(
                        active_results, 0, message="active_results must be positive"
                    ),
                    tf.debugging.assert_less_equal(
                        active_results,
                        config.max_results,
                        message="active_results exceeds max_results",
                    ),
                    tf.debugging.assert_greater_equal(
                        global_start_index,
                        0,
                        message="global_start_index must be nonnegative",
                    ),
                )
            ):
                active_results = tf.identity(active_results)
                global_start_index = tf.identity(global_start_index)

            kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob,
                step_size=tf.constant(config.step_size, dtype=current_state.dtype),
                num_leapfrog_steps=config.num_leapfrog_steps,
            )
            kernel_results = kernel.bootstrap_results(current_state)
            # Fail at trace time if TFP changes the single-state-part contract.
            _single_state_part(
                kernel_results.accepted_results.grads_target_log_prob,
                name="accepted_results.grads_target_log_prob",
            )
            _single_state_part(
                kernel_results.accepted_results.initial_momentum,
                name="accepted_results.initial_momentum",
            )

            state_buffer_shape = (config.max_results, chain_count, parameter_dim)
            scalar_buffer_shape = (config.max_results, chain_count)
            seed_buffer_shape = (config.max_results, 2)
            nan = tf.constant(float("nan"), dtype=current_state.dtype)
            int_sentinel = tf.constant(-1, dtype=tf.int32)

            state_buffers = tuple(
                tf.fill(state_buffer_shape, nan) for _ in range(8)
            )
            scalar_buffers = tuple(
                tf.fill(scalar_buffer_shape, nan) for _ in range(11)
            )
            accepted_buffer = tf.zeros(scalar_buffer_shape, dtype=tf.bool)
            seed_buffers = tuple(
                tf.fill(seed_buffer_shape, int_sentinel) for _ in range(2)
            )
            transition_index_buffer = tf.fill((config.max_results,), int_sentinel)
            step_size_buffer = tf.fill(
                (config.max_results,), tf.cast(config.step_size, current_state.dtype)
            )
            leapfrog_buffer = tf.fill(
                (config.max_results,),
                tf.constant(config.num_leapfrog_steps, dtype=tf.int32),
            )

            def condition(index: Any, *_unused: Any) -> Any:
                return index < active_results

            def body(
                index: Any,
                state: Any,
                results: Any,
                pre_state_buffer: Any,
                proposed_state_buffer: Any,
                post_state_buffer: Any,
                pre_grad_buffer: Any,
                proposed_grad_buffer: Any,
                post_grad_buffer: Any,
                initial_momentum_buffer: Any,
                final_momentum_buffer: Any,
                pre_target_buffer: Any,
                proposed_target_buffer: Any,
                post_target_buffer: Any,
                log_accept_buffer: Any,
                correction_buffer: Any,
                initial_kinetic_buffer: Any,
                final_kinetic_buffer: Any,
                initial_energy_buffer: Any,
                proposed_energy_buffer: Any,
                delta_h_buffer: Any,
                identity_residual_buffer: Any,
                is_accepted_buffer: Any,
                metropolis_seed_buffer: Any,
                proposal_seed_buffer: Any,
                transition_indices: Any,
                step_sizes: Any,
                leapfrog_steps: Any,
            ) -> tuple[Any, ...]:
                absolute_index = global_start_index + index
                step_seed = tf.random.experimental.stateless_fold_in(
                    master_seed, absolute_index
                )
                pre_target = tf.cast(
                    results.accepted_results.target_log_prob, current_state.dtype
                )
                pre_grad = tf.cast(
                    _single_state_part(
                        results.accepted_results.grads_target_log_prob,
                        name="accepted_results.grads_target_log_prob",
                    ),
                    current_state.dtype,
                )
                next_state, next_results = kernel.one_step(
                    state, results, seed=step_seed
                )
                proposal_results = next_results.proposed_results
                accepted_results = next_results.accepted_results
                initial_momentum = tf.cast(
                    _single_state_part(
                        proposal_results.initial_momentum,
                        name="proposed_results.initial_momentum",
                    ),
                    current_state.dtype,
                )
                final_momentum = tf.cast(
                    _single_state_part(
                        proposal_results.final_momentum,
                        name="proposed_results.final_momentum",
                    ),
                    current_state.dtype,
                )
                proposed_state = tf.cast(
                    tf.convert_to_tensor(next_results.proposed_state),
                    current_state.dtype,
                )
                proposed_target = tf.cast(
                    proposal_results.target_log_prob, current_state.dtype
                )
                post_target = tf.cast(
                    accepted_results.target_log_prob, current_state.dtype
                )
                proposed_grad = tf.cast(
                    _single_state_part(
                        proposal_results.grads_target_log_prob,
                        name="proposed_results.grads_target_log_prob",
                    ),
                    current_state.dtype,
                )
                post_grad = tf.cast(
                    _single_state_part(
                        accepted_results.grads_target_log_prob,
                        name="accepted_results.grads_target_log_prob",
                    ),
                    current_state.dtype,
                )
                log_accept = tf.cast(
                    next_results.log_accept_ratio, current_state.dtype
                )
                correction = tf.cast(
                    proposal_results.log_acceptance_correction,
                    current_state.dtype,
                )
                initial_kinetic = 0.5 * tf.reduce_sum(
                    tf.square(initial_momentum), axis=-1
                )
                final_kinetic = 0.5 * tf.reduce_sum(
                    tf.square(final_momentum), axis=-1
                )
                initial_energy = -pre_target + initial_kinetic
                proposed_energy = -proposed_target + final_kinetic
                delta_h = proposed_energy - initial_energy
                identity_residual = delta_h + log_accept
                proposal_seed = tf.cast(proposal_results.seed, tf.int32)

                row = tf.reshape(index, (1, 1))

                def update(buffer: Any, value: Any) -> Any:
                    return tf.tensor_scatter_nd_update(
                        buffer, row, tf.expand_dims(value, axis=0)
                    )

                return (
                    index + 1,
                    next_state,
                    next_results,
                    update(pre_state_buffer, state),
                    update(proposed_state_buffer, proposed_state),
                    update(post_state_buffer, next_state),
                    update(pre_grad_buffer, pre_grad),
                    update(proposed_grad_buffer, proposed_grad),
                    update(post_grad_buffer, post_grad),
                    update(initial_momentum_buffer, initial_momentum),
                    update(final_momentum_buffer, final_momentum),
                    update(pre_target_buffer, pre_target),
                    update(proposed_target_buffer, proposed_target),
                    update(post_target_buffer, post_target),
                    update(log_accept_buffer, log_accept),
                    update(correction_buffer, correction),
                    update(initial_kinetic_buffer, initial_kinetic),
                    update(final_kinetic_buffer, final_kinetic),
                    update(initial_energy_buffer, initial_energy),
                    update(proposed_energy_buffer, proposed_energy),
                    update(delta_h_buffer, delta_h),
                    update(identity_residual_buffer, identity_residual),
                    update(is_accepted_buffer, tf.cast(next_results.is_accepted, tf.bool)),
                    update(metropolis_seed_buffer, tf.cast(next_results.seed, tf.int32)),
                    update(proposal_seed_buffer, proposal_seed),
                    tf.tensor_scatter_nd_update(
                        transition_indices,
                        row,
                        tf.reshape(absolute_index, (1,)),
                    ),
                    step_sizes,
                    leapfrog_steps,
                )

            loop_values = (
                tf.constant(0, dtype=tf.int32),
                current_state,
                kernel_results,
                *state_buffers,
                *scalar_buffers,
                accepted_buffer,
                *seed_buffers,
                transition_index_buffer,
                step_size_buffer,
                leapfrog_buffer,
            )
            outputs = tf.while_loop(
                condition,
                body,
                loop_values,
                parallel_iterations=1,
            )
            (
                _index,
                final_state,
                _final_results,
                pre_state,
                proposed_state,
                post_state,
                pre_grad,
                proposed_grad,
                post_grad,
                initial_momentum,
                final_momentum,
                pre_target,
                proposed_target,
                post_target,
                log_accept,
                correction,
                initial_kinetic,
                final_kinetic,
                initial_energy,
                proposed_energy,
                delta_h,
                identity_residual,
                is_accepted,
                metropolis_seeds,
                proposal_seeds,
                transition_indices,
                step_sizes,
                leapfrog_steps,
            ) = outputs
            valid_mask = tf.range(config.max_results, dtype=tf.int32) < active_results
            tensors = {
                "transition_index": transition_indices,
                "pre_state": pre_state,
                "proposed_state": proposed_state,
                "post_state": post_state,
                "pre_target_log_prob": pre_target,
                "proposed_target_log_prob": proposed_target,
                "post_target_log_prob": post_target,
                "pre_grad_target_log_prob": pre_grad,
                "proposed_grad_target_log_prob": proposed_grad,
                "post_grad_target_log_prob": post_grad,
                "is_accepted": is_accepted,
                "log_accept_ratio": log_accept,
                "log_acceptance_correction": correction,
                "initial_momentum": initial_momentum,
                "final_momentum": final_momentum,
                "initial_kinetic_energy": initial_kinetic,
                "final_kinetic_energy": final_kinetic,
                "initial_energy": initial_energy,
                "proposed_energy": proposed_energy,
                "delta_h": delta_h,
                "hamiltonian_identity_residual": identity_residual,
                "metropolis_seed": metropolis_seeds,
                "proposal_seed": proposal_seeds,
                "effective_step_size": step_sizes,
                "effective_num_leapfrog_steps": leapfrog_steps,
            }
            return tensors, valid_mask, final_state

        input_signature = (
            tf.TensorSpec(self._state_shape, self._dtype),
            tf.TensorSpec((), tf.int32),
            tf.TensorSpec((), tf.int32),
        )
        if config.chain_execution_mode == "eager":
            return run_block
        return tf.function(
            run_block,
            input_signature=input_signature,
            jit_compile=config.use_xla,
            reduce_retracing=True,
        )

    def _diagnostics(
        self,
        tensors: Mapping[str, Any],
        valid_mask: Any,
        initial_state: Any,
        final_state: Any,
    ) -> Mapping[str, Any]:
        import tensorflow as tf

        active = {key: tf.boolean_mask(value, valid_mask) for key, value in tensors.items()}
        required_float = tuple(
            key
            for key in TRANSITION_TENSOR_KEYS
            if key not in {
                "transition_index",
                "is_accepted",
                "metropolis_seed",
                "proposal_seed",
                "effective_num_leapfrog_steps",
            }
        )
        finite_by_field = {
            key: tf.reduce_all(tf.math.is_finite(active[key]))
            for key in required_float
        }
        hamiltonian_identity = summarize_hmc_exact_mechanics_identity(
            np.asarray(active["delta_h"].numpy(), dtype=np.float64),
            -np.asarray(active["log_accept_ratio"].numpy(), dtype=np.float64),
            identity_name="delta_h_equals_negative_log_accept_ratio",
        )
        kinetic_identity = summarize_hmc_exact_mechanics_identity(
            np.asarray(
                active["log_acceptance_correction"].numpy(), dtype=np.float64
            ),
            np.asarray(
                (
                    active["initial_kinetic_energy"]
                    - active["final_kinetic_energy"]
                ).numpy(),
                dtype=np.float64,
            ),
            identity_name=(
                "log_acceptance_correction_equals_initial_minus_final_kinetic"
            ),
        )
        return {
            "active_results": tf.reduce_sum(tf.cast(valid_mask, tf.int32)),
            "all_required_float_tensors_finite": tf.reduce_all(
                tf.stack(tuple(finite_by_field.values()))
            ),
            "finite_by_field": finite_by_field,
            "max_abs_delta_h": tf.reduce_max(tf.abs(active["delta_h"])),
            "max_abs_hamiltonian_identity_residual": tf.reduce_max(
                tf.abs(active["hamiltonian_identity_residual"])
            ),
            "max_abs_kinetic_correction_identity_residual": tf.reduce_max(
                tf.abs(
                    active["log_acceptance_correction"]
                    - (
                        active["initial_kinetic_energy"]
                        - active["final_kinetic_energy"]
                    )
                )
            ),
            "hamiltonian_identity_diagnostic": hamiltonian_identity,
            "kinetic_correction_identity_diagnostic": kinetic_identity,
            "exact_mechanics_identities_passed": bool(
                hamiltonian_identity["passed"] and kinetic_identity["passed"]
            ),
            "acceptance_rate_by_chain": tf.reduce_mean(
                tf.cast(active["is_accepted"], tf.float64), axis=0
            ),
            "initial_state_matches_first_pre_state": tf.reduce_all(
                tf.equal(initial_state, active["pre_state"][0])
            ),
            "final_state_matches_last_post_state": tf.reduce_all(
                tf.equal(final_state, active["post_state"][-1])
            ),
            "transition_state_continuity": tf.reduce_all(
                tf.equal(active["post_state"][:-1], active["pre_state"][1:])
            ),
            "hamiltonian_identity": "delta_h = proposed_energy - initial_energy = -log_accept_ratio",
            "kinetic_correction_identity": "log_acceptance_correction = initial_kinetic_energy - final_kinetic_energy",
            "ebfmi_energy_field": "initial_energy",
        }

    def _metadata(
        self,
        *,
        active_results: int | None,
        global_start_index: int | None,
        call_seconds: float,
    ) -> Mapping[str, Any]:
        trace_count = getattr(self._runner, "experimental_get_tracing_count", None)
        return {
            "runtime": "tfp.mcmc.HamiltonianMonteCarlo.one_step_transition_archive_tf_while_loop",
            "uses_sample_chain": False,
            "transition_archive_runner": True,
            "hmc_execution_call_count": self._call_count,
            "adaptation_invocation_count": 0,
            "tuning_invocation_count": 0,
            "active_results": active_results,
            "global_start_index": global_start_index,
            "global_end_index_exclusive": (
                None
                if active_results is None or global_start_index is None
                else global_start_index + active_results
            ),
            "state_shape": self._state_shape,
            "dtype": self._dtype.name,
            "config": self.config.signature_payload(),
            "adapter_signature": stable_adapter_signature(self.adapter),
            "value_score_authority": self.capability.value_score_authority,
            "target_scope": self.capability.target_scope,
            "runner_build_seconds": self._runner_build_seconds,
            "call_seconds": call_seconds,
            "first_call_compile_plus_execute_seconds": self._first_call_seconds,
            "latest_warm_call_seconds": self._warm_call_seconds,
            "compile_trace_count": (
                None if trace_count is None else int(trace_count())
            ),
            "tfp_static_kernel_result_fields": {
                "accepted_step_size": "empty_state_part_list; effective scalar archived explicitly",
                "accepted_num_leapfrog_steps": "empty_state_part_list; effective scalar archived explicitly",
                "accepted_seed": "empty_state_part_list",
                "proposal_seed": "archived",
                "metropolis_seed": "archived",
            },
            "nonclaims": (
                "transition instrumentation and continuation only",
                "no tuning or adaptation",
                "no posterior convergence or scientific-validity claim",
            ),
        }


def build_hmc_transition_archive_runner(
    adapter: Any,
    initial_state_template: Any,
    config: HMCTransitionArchiveConfig,
) -> HMCTransitionArchiveRunner:
    """Build one reusable BayesFilter transition archive runner."""

    return HMCTransitionArchiveRunner(adapter, initial_state_template, config)


def write_hmc_transition_shard(
    result: HMCTransitionArchiveRunResult,
    *,
    path: str | Path,
    role: str,
    block_index: int,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    """Atomically persist one active transition shard and verify its read-back."""

    if role not in {
        "warmup_diagnostic",
        "posterior_pilot",
        "posterior",
        "instrumentation_canary",
    }:
        raise ValueError("invalid immutable transition role")
    overwrite_enabled = _strict_bool(overwrite, name="overwrite")
    block = _strict_scalar_integer(block_index, name="block_index")
    if block < 0:
        raise ValueError("block_index must be nonnegative")
    output = Path(path)
    if output.suffix != ".npz":
        raise ValueError("transition shard path must use .npz")
    if output.exists() and not overwrite_enabled:
        raise FileExistsError(f"transition shard already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    active = np.asarray(result.valid_mask.numpy(), dtype=bool)
    active_count = int(np.sum(active))
    if active_count <= 0:
        raise ValueError("cannot persist an empty transition shard")
    arrays = {
        key: np.asarray(result.tensors[key].numpy())[active]
        for key in TRANSITION_TENSOR_KEYS
    }
    arrays["initial_state"] = np.asarray(result.initial_state.numpy())
    arrays["final_state"] = np.asarray(result.final_state.numpy())
    start = int(arrays["transition_index"][0])
    end = int(arrays["transition_index"][-1]) + 1
    expected = np.arange(start, end, dtype=arrays["transition_index"].dtype)
    if not np.array_equal(arrays["transition_index"], expected):
        raise ValueError("transition indices must be contiguous")
    payload = {
        "artifact_schema": "bayesfilter.hmc_transition_shard.v1",
        "role": role,
        "block_index": block,
        "active_results": active_count,
        "global_start_index": start,
        "global_end_index_exclusive": end,
        "tensor_axes": {
            "state_and_gradient": "[transition, chain, parameter]",
            "scalar_trace_and_energy": "[transition, chain]",
            "transition_seed": "[transition, seed_word]",
            "state_handoff": "[chain, parameter]",
        },
        "runner_metadata": _json_safe(result.metadata),
        "runner_diagnostics": _json_safe(result.diagnostics),
        "metadata": _json_safe({} if metadata is None else dict(metadata)),
        "tensor_manifest": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "byte_count": int(value.nbytes),
                "sha256": _sha256_array(value),
            }
            for key, value in arrays.items()
        },
    }
    metadata_bytes = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    archive_arrays = dict(arrays)
    archive_arrays["__metadata_json_utf8__"] = np.frombuffer(
        metadata_bytes, dtype=np.uint8
    )
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}.npz")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **archive_arrays)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite_enabled:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    readback = read_hmc_transition_shard(output)
    for key, expected_array in arrays.items():
        if not np.array_equal(
            readback.tensors[key], expected_array, equal_nan=True
        ):
            raise RuntimeError(f"transition shard read-back mismatch: {key}")
    return {
        **payload,
        "path": str(output.resolve()),
        "sha256": readback.sha256,
        "file_byte_count": output.stat().st_size,
        "readback_verified": True,
    }


def read_hmc_transition_shard(path: str | Path) -> HMCTransitionShard:
    """Read and validate one transition shard without pickle."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        files = set(archive.files)
        expected = set(TRANSITION_TENSOR_KEYS) | {
            "initial_state",
            "final_state",
            "__metadata_json_utf8__",
        }
        if files != expected:
            missing = sorted(expected - files)
            extra = sorted(files - expected)
            raise ValueError(
                f"transition shard tensor schema mismatch: missing={missing}, extra={extra}"
            )
        metadata = json.loads(
            np.asarray(archive["__metadata_json_utf8__"], dtype=np.uint8)
            .tobytes()
            .decode("utf-8")
        )
        if metadata.get("artifact_schema") != "bayesfilter.hmc_transition_shard.v1":
            raise ValueError("transition shard metadata schema mismatch")
        tensors = {
            key: np.asarray(archive[key]).copy()
            for key in expected
            if key != "__metadata_json_utf8__"
        }
    manifest = metadata.get("tensor_manifest", {})
    for key, tensor in tensors.items():
        record = manifest.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"transition shard manifest missing {key}")
        if list(tensor.shape) != record.get("shape"):
            raise ValueError(f"transition shard shape mismatch: {key}")
        if str(tensor.dtype) != record.get("dtype"):
            raise ValueError(f"transition shard dtype mismatch: {key}")
        byte_count = _strict_scalar_integer(
            record.get("byte_count"),
            name=f"transition shard {key} byte_count",
        )
        if int(tensor.nbytes) != byte_count:
            raise ValueError(f"transition shard byte-count mismatch: {key}")
        if _sha256_array(tensor) != record.get("sha256"):
            raise ValueError(f"transition shard tensor hash mismatch: {key}")
    indices = tensors["transition_index"]
    active_results = _strict_scalar_integer(
        metadata.get("active_results"),
        name="transition shard active_results",
    )
    if indices.ndim != 1 or indices.size != active_results:
        raise ValueError("transition shard active index shape mismatch")
    global_start = _strict_scalar_integer(
        metadata.get("global_start_index"),
        name="transition shard global_start_index",
    )
    global_end = _strict_scalar_integer(
        metadata.get("global_end_index_exclusive"),
        name="transition shard global_end_index_exclusive",
    )
    if global_start < 0 or global_end - global_start != active_results:
        raise ValueError("transition shard index metadata is inconsistent")
    expected_indices = np.arange(
        global_start,
        global_end,
        dtype=indices.dtype,
    )
    if not np.array_equal(indices, expected_indices):
        raise ValueError("transition shard indices are not contiguous")
    if not np.array_equal(tensors["initial_state"], tensors["pre_state"][0]):
        raise ValueError("transition shard initial-state handoff mismatch")
    if not np.array_equal(tensors["final_state"], tensors["post_state"][-1]):
        raise ValueError("transition shard final-state handoff mismatch")
    if tensors["post_state"].shape[0] > 1 and not np.array_equal(
        tensors["post_state"][:-1], tensors["pre_state"][1:]
    ):
        raise ValueError("transition shard internal state continuity mismatch")
    return HMCTransitionShard(
        path=str(source.resolve()),
        sha256=_sha256_file(source),
        metadata=metadata,
        tensors=tensors,
    )


__all__ = [
    "HMCExactMechanicsIdentityPolicy",
    "HMCTransitionArchiveConfig",
    "HMCTransitionArchiveRunResult",
    "HMCTransitionArchiveRunner",
    "HMCTransitionShard",
    "TRANSITION_TENSOR_KEYS",
    "build_hmc_transition_archive_runner",
    "read_hmc_transition_shard",
    "summarize_hmc_exact_mechanics_identity",
    "write_hmc_transition_shard",
]
