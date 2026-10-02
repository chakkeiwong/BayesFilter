"""First-error observation for non-XLA, traced, analytical-score HMC.

Resource writes precede the target, so its input survives an aborted TF call.
The host catches the original error; it never evaluates the target again.
The transparent kernel wrapper counts cached-score target points, not eager
value/gradient pairs (monograph eq:lf_full_theta and TFP maybe_call_fn_and_grads).
This is observability, not an integrator, tuner or runtime admission decision.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp


def _json_value(value: Any) -> Any:
    if tf.is_tensor(value):
        value = value.numpy().tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class TracedHMCFailureRecorder:
    """Retain a pending evaluation across a failed graph call, without replay.

    Supported state shapes are [parameter] and [chain, parameter]. A returned
    nonfinite value/score identifies failing rows. An inner batched assertion
    may not expose its failing row; in that case the complete input batch is
    retained with explicit unavailable attribution, never an invented index.
    One instance belongs to one serial reusable runner and becomes terminal
    after any failed call. Successful calls can reset the observation state.
    """

    def __init__(self, adapter: Any, state_template: Any) -> None:
        template = tf.convert_to_tensor(state_template)
        if template.shape.rank not in (1, 2) or any(
            dimension is None or dimension < 1 for dimension in template.shape
        ):
            raise ValueError("traced capture requires a static vector or chain/parameter matrix")
        if not callable(getattr(adapter, "log_prob_and_grad", None)):
            raise TypeError("traced capture requires the supplied analytical value/score")
        self.adapter = adapter
        self.chain_count = 1 if template.shape.rank == 1 else int(template.shape[0])
        self.state = tf.Variable(template, trainable=False)
        self.pre_state = tf.Variable(template, trainable=False)
        self.phase = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.transition = tf.Variable(-1, dtype=tf.int32, trainable=False)
        self.callbacks = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.target_pending = tf.Variable(False, trainable=False)
        self.bad_rows = tf.Variable(tf.zeros([self.chain_count], tf.bool), trainable=False)
        self._context: dict[str, Any] | None = None
        self._failure: dict[str, Any] | None = None

    @property
    def failure(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._failure)

    def begin_call(self, *, seed: Any, step_size: Any, leapfrog_steps: int,
                   burnin_steps: int, role: str) -> None:
        """Reset outside tracing only; a failed runner cannot silently retry."""
        if self._failure is not None:
            error = RuntimeError("first-failure capture is terminal; retry is forbidden")
            error.failure_record = self.failure
            raise error
        self._context = {
            "seed": _json_value(tf.convert_to_tensor(seed)),
            "initial_step_size": _json_value(tf.convert_to_tensor(step_size)),
            "leapfrog_steps": int(leapfrog_steps),
            "burnin_steps": int(burnin_steps),
            "lifecycle_stage": str(role),
        }
        self.phase.assign(0)
        self.transition.assign(-1)
        self.callbacks.assign(0)
        self.target_pending.assign(False)
        self.bad_rows.assign(tf.zeros_like(self.bad_rows))

    def _begin_phase(self, state: Any, phase: int, transition: Any) -> Any:
        return tf.group(
            self.pre_state.assign(state), self.state.assign(state),
            self.phase.assign(phase), self.transition.assign(transition),
            self.callbacks.assign(0), self.target_pending.assign(False),
            self.bad_rows.assign(tf.zeros_like(self.bad_rows)),
        )

    def _check_rows(self, finite_rows: Any, message: str) -> Any:
        bad = tf.reshape(tf.logical_not(finite_rows), [self.chain_count])
        with tf.control_dependencies([self.bad_rows.assign(bad)]):
            return tf.debugging.assert_equal(
                tf.reduce_any(bad), False, message="traced HMC target: " + message,
            )

    def log_prob_and_grad(self, position: Any) -> tuple[Any, Any]:
        """Observe the supplied score, with no reverse-mode or eager fallback."""
        with tf.control_dependencies([
            self.state.assign(position), self.callbacks.assign_add(1),
            self.target_pending.assign(True),
            self.bad_rows.assign(tf.zeros_like(self.bad_rows)),
        ]):
            position = tf.identity(position)
        state_check = self._check_rows(
            tf.reduce_all(tf.math.is_finite(position), axis=-1), "state is nonfinite",
        )
        with tf.control_dependencies([state_check]):
            value, score = self.adapter.log_prob_and_grad(tf.identity(position))
            value = tf.convert_to_tensor(value, dtype=position.dtype)
            score = tf.convert_to_tensor(score, dtype=position.dtype)
        with tf.control_dependencies([
            tf.debugging.assert_equal(tf.shape(value), tf.shape(position)[:-1]),
            tf.debugging.assert_equal(tf.shape(score), tf.shape(position)),
        ]):
            value_check = self._check_rows(tf.math.is_finite(value), "value is nonfinite")
        with tf.control_dependencies([value_check]):
            score_check = self._check_rows(
                tf.reduce_all(tf.math.is_finite(score), axis=-1), "score is nonfinite",
            )
        with tf.control_dependencies([score_check]):
            completed = self.target_pending.assign(False)
        with tf.control_dependencies([completed]):
            return tf.identity(value), tf.identity(score)

    def wrap_kernel(self, inner_kernel: Any, leapfrog_steps: Any) -> Any:
        """Add boundary observation without altering kernel-result structure."""
        recorder = self

        def complete(output: Any, expected_callbacks: Any) -> Any:
            with tf.control_dependencies(tf.nest.flatten(output)):
                check = tf.debugging.assert_equal(
                    recorder.callbacks.read_value(), expected_callbacks,
                    message="traced HMC callback count differs from cached-score contract",
                )
            with tf.control_dependencies([check]):
                return tf.nest.map_structure(tf.identity, output)

        class ObservedKernel(tfp.mcmc.TransitionKernel):
            @property
            def is_calibrated(self) -> bool:
                return inner_kernel.is_calibrated

            @property
            def parameters(self) -> dict[str, Any]:
                return {"inner_kernel": inner_kernel}

            def bootstrap_results(self, init_state: Any) -> Any:
                with tf.control_dependencies([recorder._begin_phase(init_state, 1, -1)]):
                    output = inner_kernel.bootstrap_results(tf.identity(init_state))
                return complete(output, 1)

            def one_step(self, current_state: Any, previous_kernel_results: Any,
                         seed: Any = None) -> Any:
                with tf.control_dependencies(tf.nest.flatten(previous_kernel_results)):
                    begin = recorder._begin_phase(
                        current_state, 2, recorder.transition.read_value() + 1,
                    )
                with tf.control_dependencies([begin]):
                    output = inner_kernel.one_step(
                        tf.identity(current_state), previous_kernel_results, seed=seed,
                    )
                return complete(output, leapfrog_steps)

        return ObservedKernel()

    def wrap_trace(self, trace_fn: Any) -> Any:
        """Keep trace failures distinct from failures inside a target point."""
        def trace(state: Any, results: Any) -> Any:
            with tf.control_dependencies(tf.nest.flatten(results)):
                begin = tf.group(self.phase.assign(3), self.state.assign(state))
            with tf.control_dependencies([begin]):
                return trace_fn(tf.identity(state), results)
        return trace

    def capture(self, error: Exception) -> dict[str, Any]:
        """Attach full original provenance to the same exception at the host."""
        if self._failure is None:
            phase = int(self.phase.numpy())
            pending = bool(self.target_pending.numpy())
            transition = int(self.transition.numpy())
            callbacks = int(self.callbacks.numpy())
            rows = tf.reshape(tf.where(self.bad_rows), [-1]).numpy().tolist()
            context = dict(self._context or {})
            attribution = "nonfinite_rows" if rows else (
                "single_chain" if self.chain_count == 1 else "unavailable_for_batch_exception"
            )
            indices = rows or ([0] if self.chain_count == 1 else None)
            self._failure = {
                "schema": "bayesfilter.traced_hmc_first_failure.v1",
                **context,
                "exception_type": f"{type(error).__module__}.{type(error).__name__}",
                "original_error": str(error),
                "target_state": _json_value(self.state.read_value()) if phase else None,
                "pre_transition_state": _json_value(self.pre_state.read_value()) if phase else None,
                "state_dtype": self.state.dtype.name,
                "chain_count": self.chain_count,
                "chain_index": indices[0] if indices is not None and len(indices) == 1 else None,
                "failed_chain_indices": indices,
                "chain_attribution": attribution,
                "transition_index": transition if phase else None,
                "failed_leapfrog_substep": (0 if phase == 1 else callbacks) if pending else None,
                "target_callbacks": callbacks,
                "evaluation_phase": {0: "before_execution", 1: "bootstrap", 2: "trajectory", 3: "trace"}[phase],
                "draw_role": "bootstrap" if phase == 1 else (
                    "discarded_burnin" if transition < context.get("burnin_steps", 0)
                    else context.get("lifecycle_stage", "unknown")
                ),
                "failure_location": "target_callback" if pending else (
                    "trace" if phase == 3 else "kernel_or_runner"
                ),
                "diagnostic_target_replayed": False,
                "scientific_authority": False,
            }
        error.failure_record = self.failure
        return self.failure
