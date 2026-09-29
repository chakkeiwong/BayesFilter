"""Eager, single-chain observation of TFP HMC failures, not another tuner.

TFP's cached-score leapfrog computes one new score after each position update
(monograph eq:lf_full_theta). Its eager value/gradient utility calls the target
twice at that point: first for value, then under its gradient tape. Counting
validated callback pairs locates the step without replacing the integrator. The outer kernel
must be eager; the adapter's numerical value/score function may be compiled.
This debugging helper issues no tuning handoff or posterior/XLA readiness claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.batched_value_score import reviewed_value_score_target_fn
from bayesfilter.inference.fixed_l_finite_bracket import (
    bounded_dual_averaging_kwargs,
    classify_tuning_exception,
    guard_finite_transitions,
)


def diagnostic_json(value: Any) -> Any:
    """Serialize diagnostic tensors without NumPy computation or invalid JSON."""

    if tf.is_tensor(value):
        value = value.numpy().tolist()
    if isinstance(value, dict):
        return {key: diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def state_sha256(state: Any) -> str:
    """Hash dtype, shape and exact tensor bytes through TensorFlow serialization."""

    return hashlib.sha256(tf.io.serialize_tensor(state).numpy()).hexdigest()


class FirstFailureRecorder:
    """Record the first original exception without retrying or changing a target.

    Only one chain is accepted: a failed batch assertion cannot generally reveal
    which row failed without replay. ``chain_index`` identifies that chain in a
    larger caller's ledger. Bootstrap evaluations have substep zero; trajectory
    substeps are one-based, global transition indices zero-based. Endpoint
    failures have no claimed failed leapfrog substep.
    """

    def __init__(
        self,
        adapter: Any,
        output_path: Path,
        *,
        chain_index: int = 0,
        diagnose: Callable[[Any], dict[str, Any]] | None = None,
    ) -> None:
        self.adapter = adapter
        self.output_path = Path(output_path)
        self.chain_index = chain_index
        self.diagnose = diagnose
        self.failure: dict[str, Any] | None = None
        self.context: dict[str, Any] | None = None
        self.target_evaluations = 0
        self.last_success: dict[str, Any] | None = None
        self.last_position: Any = None
        self.pre_state: Any = None
        self.point_hash: str | None = None
        self.target_fn = reviewed_value_score_target_fn(self, require_batched=True)

    def _begin(
        self, state: Any, *, transition_index: int, leapfrog_steps: int,
        epsilon: Any, lifecycle: str, phase: str,
    ) -> None:
        if not tf.executing_eagerly():
            raise RuntimeError("first-failure observation requires eager outer TFP execution")
        if self.failure is not None:
            raise RuntimeError("first-failure diagnostic is terminal; retry is forbidden")
        if state.shape.rank != 2 or state.shape[0] != 1:
            raise ValueError("first-failure diagnostic requires one rank-2 chain")
        if lifecycle not in {"bracketing", "dual_averaging", "verification"}:
            raise ValueError("unknown first-failure lifecycle")
        self.pre_state = tf.identity(state)
        self.last_position = state
        self.target_evaluations = 0
        self.last_success = None
        self.point_hash = None
        self.context = {
            "chain_index": self.chain_index,
            "global_transition_index": transition_index,
            "leapfrog_steps": leapfrog_steps,
            "consumed_epsilon": diagnostic_json(tf.convert_to_tensor(epsilon)),
            "lifecycle_stage": lifecycle,
            "evaluation_phase": phase,
            "pre_state_sha256": state_sha256(state),
        }

    def _capture(self, error: Exception, *, failed_substep: int | None) -> None:
        if self.failure is not None:
            return
        failure_class = classify_tuning_exception(self.adapter, error)
        if "first-failure target" in str(error):
            failure_class = "nonfinite_trajectory"
        record = {
            **self.context,
            "failure_class": failure_class,
            "exception_type": f"{type(error).__module__}.{type(error).__name__}",
            "first_failed_assertion": str(error),
            "failed_leapfrog_substep": failed_substep,
            "target_evaluations": self.target_evaluations,
            "tfp_callbacks_per_point": 2,
            "callback_role": None if failed_substep is None else (
                "value" if self.target_evaluations % 2 else "gradient"
            ),
            "failure_location": "kernel_or_endpoint" if failed_substep is None else "target_callback",
            "proposal_sha256": state_sha256(self.last_position),
            "last_successful_target_evaluation": self.last_success,
            "diagnostic_target_replayed": False,
        }
        if self.diagnose is not None:
            try:
                record["chart_diagnostics"] = diagnostic_json(self.diagnose(self.last_position))
            except Exception as diagnostic_error:  # noqa: BLE001 - preserve the original target failure
                record["chart_diagnostics_error"] = repr(diagnostic_error)
        self.failure = diagnostic_json(record)
        try:
            with self.output_path.open("x") as stream:
                json.dump(self.failure, stream, indent=2, allow_nan=False)
                stream.write("\n")
        except Exception as artifact_error:  # noqa: BLE001 - report secondary I/O failure without replacing the target error
            self.failure["artifact_write_error"] = repr(artifact_error)

    def log_prob_and_grad(self, position: Any) -> tuple[Any, Any]:
        if not tf.executing_eagerly() or self.context is None:
            raise RuntimeError("first-failure target requires an active eager observation")
        self.last_position = tf.identity(position)
        self.target_evaluations += 1
        substep = 0 if self.context["evaluation_phase"] == "bootstrap" else (self.target_evaluations + 1) // 2
        try:
            position_hash = state_sha256(position)
            if self.target_evaluations % 2:
                self.point_hash = position_hash
            elif self.point_hash != position_hash:
                raise RuntimeError("TFP eager value/gradient callback positions differ")
            value, score = self.adapter.log_prob_and_grad(position)
            tf.debugging.assert_all_finite(value, "first-failure target value is nonfinite")
            tf.debugging.assert_all_finite(score, "first-failure target score is nonfinite")
            tf.debugging.assert_equal(tf.shape(value), tf.shape(position)[:1])
            tf.debugging.assert_equal(tf.shape(score), tf.shape(position))
        except Exception as error:
            self._capture(error, failed_substep=substep)
            raise
        self.last_success = {
            "substep": substep, "position_sha256": state_sha256(position),
            "target_value": diagnostic_json(value),
        }
        return value, score

    def bootstrap(self, kernel: Any, state: Any, **context: Any) -> Any:
        self._begin(state, phase="bootstrap", **context)
        try:
            return kernel.bootstrap_results(state)
        except Exception as error:
            self._capture(error, failed_substep=0)
            raise

    def one_step(
        self, kernel: Any, state: Any, previous_results: Any, *, seed: Any,
        uses_dual_averaging: bool = False, **context: Any,
    ) -> tuple[Any, Any, dict[str, Any]]:
        self._begin(state, phase="trajectory", **context)
        try:
            next_state, results = kernel.one_step(state, previous_results, seed=seed)
            if self.target_evaluations != 2 * context["leapfrog_steps"]:
                raise RuntimeError("TFP target-evaluation count does not match leapfrog steps")
            endpoint = results.inner_results
            if uses_dual_averaging:
                endpoint = endpoint.inner_results
            tf.debugging.assert_equal(
                endpoint.proposed_results.step_size,
                tf.convert_to_tensor(context["epsilon"], dtype=state.dtype),
                message="recorded epsilon differs from TFP consumed epsilon",
            )
        except Exception as error:
            self._capture(error, failed_substep=None)
            raise
        trace = diagnostic_json({
            **self.context, "seed": seed,
            "target_evaluations": self.target_evaluations,
            "leapfrog_target_points": self.target_evaluations // 2,
            "tfp_callbacks_per_point": 2,
            "pre_state": state, "proposed_state": endpoint.proposed_state,
            "next_state": next_state, "health": results.health,
            "proposal_sha256": state_sha256(endpoint.proposed_state),
            "is_accepted": endpoint.is_accepted,
            "log_accept_ratio": endpoint.log_accept_ratio,
            "draw_role": "discarded_mechanics_diagnostic",
        })
        return next_state, results, trace


def diagnostic_hmc_kernel(
    recorder: FirstFailureRecorder, *, step_size: Any, leapfrog_steps: int,
    adaptation_steps: int = 0, upper_bound: float | None = None,
) -> Any:
    """Use stock TFP HMC/MH plus the repaired setter/endpoint guard.

    Identity mass is an explicit mechanics baseline. This helper does not search
    or select kernels, and cannot issue a public tuning artifact. A caller using
    adaptation must first evaluate the finite ceiling with the same actual L.
    """

    if adaptation_steps and upper_bound is None:
        raise ValueError("diagnostic adaptation requires an evaluated finite ceiling")
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        recorder.target_fn, step_size=step_size, num_leapfrog_steps=leapfrog_steps,
        store_parameters_in_results=True,
    )
    if adaptation_steps:
        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            kernel, num_adaptation_steps=adaptation_steps,
            target_accept_prob=tf.constant(0.70, tf.float64),
            **bounded_dual_averaging_kwargs(
                SimpleNamespace(step_size_upper_bound=upper_bound), step_size, tf.float64,
            ),
        )
    return guard_finite_transitions(kernel, uses_dual_averaging=bool(adaptation_steps))
