"""Experimental TF/XLA full-refresh corrected HMC and microcanonical kernels.

MAMS uses the isokinetic velocity-Verlet composition in the archived BlackJAX
integrators.py (coefficients [1/2, 1, 1/2]), with the surface Jacobian correction
from Robnik et al. (2025), Eq. 8 and Appendix A. Only identity geometry in the
already prepared coordinates is implemented. No persistent or ensemble kernel
is emulated by dropping its state. This module grants no tuning authority.
"""
from __future__ import annotations

import time
from typing import Any

import tensorflow as tf

from .hmc_transition_spec import HMCTransitionProfile


def sphere_force_substep(direction, score, duration):
    """Exact sphere force flow plus minus-log surface Jacobian, row by row.

    The scaled exponential form avoids cosh overflow. Normalizing the computed
    direction follows the pinned author's implementation, correcting roundoff
    on the sphere; arbitrary off-sphere input is not an admitted initial law.
    Exact poles and zero force use their analytic limits, including large steps.
    """
    d = direction.shape[-1]
    if d is None or d <= 1:
        raise ValueError("MAMS requires a static dimension greater than one")
    dtype = direction.dtype
    norm = tf.linalg.norm(score, axis=-1, keepdims=True)
    signed_score = score * tf.sign(duration)
    e = tf.math.divide_no_nan(signed_score, norm)
    delta = tf.abs(duration) * norm / tf.cast(d - 1, dtype)
    c = tf.reduce_sum(e * direction, axis=-1, keepdims=True)
    zeta = tf.exp(-delta)
    raw = e * (-tf.math.expm1(-delta)) * (1. + zeta + c * (-tf.math.expm1(-delta))) + 2. * zeta * direction
    updated = tf.math.divide_no_nan(raw, tf.linalg.norm(raw, axis=-1, keepdims=True))
    plus = .5 * tf.reduce_sum(tf.square(direction + e), axis=-1, keepdims=True)
    minus = .5 * tf.reduce_sum(tf.square(direction - e), axis=-1, keepdims=True)
    log_denominator = tf.reduce_logsumexp(tf.concat([
        tf.math.log(plus), tf.math.log(minus) - 2. * delta], axis=-1), axis=-1, keepdims=True)
    work = tf.cast(d - 1, dtype) * (delta - tf.math.log(tf.cast(2., dtype)) + log_denominator)
    pole = tf.reduce_all(direction == e, axis=-1, keepdims=True) | tf.reduce_all(direction == -e, axis=-1, keepdims=True)
    updated = tf.where(pole | (norm == 0.) | (duration == 0.), direction, updated)
    work = tf.where(pole, tf.cast(d - 1, dtype) * c * delta, work)
    work = tf.where((norm == 0.) | (duration == 0.), tf.zeros_like(work), work)
    return updated, work[..., 0]


def _finite_rows(q, value, score):
    return (tf.reduce_all(tf.math.is_finite(q), axis=-1)
            & tf.math.is_finite(value)
            & tf.reduce_all(tf.math.is_finite(score), axis=-1))


class ReusableModernHMCRunner:
    """Fixed-signature, batched full-refresh transition and chain execution.

    Count is fixed per compiled runner; epsilon, maximum steps, seed, starting
    position and transition offset are runtime tensors. The offset permits an
    exact same-seed split at outer-transition boundaries. No adaptive draws are
    produced here. All target evaluations, including inactive batch rows, are
    reported separately from realized trajectory counts.
    """

    def __init__(self, adapter: Any, initial_state_template: Any, *,
                 profile: HMCTransitionProfile, num_results: int,
                 use_xla: bool = True, target_status_trace_policy: str = "none"):
        from .hmc import FullChainHMCConfig, _validate_full_chain_hmc_authority
        if not isinstance(profile, HMCTransitionProfile):
            raise TypeError("typed transition profile required")
        if type(num_results) is not int or num_results < 1:
            raise ValueError("num_results must be a positive integer")
        if type(use_xla) is not bool or target_status_trace_policy not in {"none", "per_chain_step"}:
            raise ValueError("invalid numerical execution policy")
        q = tf.convert_to_tensor(initial_state_template, tf.float64)
        if q.shape.rank != 2 or not q.shape.is_fully_defined() or min(q.shape) < 1:
            raise ValueError("state must have a static [chain, parameter] shape")
        if profile.family == "mams" and q.shape[-1] <= 1:
            raise ValueError("MAMS requires dimension greater than one")
        # Use the same exact-score authority as existing full-chain HMC.
        from .posterior_adapter import value_score_capability
        cap = value_score_capability(adapter)
        _validate_full_chain_hmc_authority(adapter, FullChainHMCConfig(
            num_results=num_results, num_burnin_steps=0, step_size=1.,
            num_leapfrog_steps=1, seed=(0, 0), use_xla=use_xla, target_scope=cap.target_scope,
            target_status_trace_policy=target_status_trace_policy))
        self.adapter, self.profile = adapter, profile
        self.shape, self.num_results = q.shape, num_results
        self.use_xla, self.status_policy = use_xla, target_status_trace_policy
        self._call_count = 0
        sig = [tf.TensorSpec(q.shape, tf.float64), tf.TensorSpec([2], tf.int32),
               tf.TensorSpec([], tf.float64), tf.TensorSpec([], tf.int32)]
        self.one_step = tf.function(self._one_step, input_signature=sig,
                                    autograph=False, jit_compile=use_xla)
        self._sample = tf.function(self._chain, input_signature=[*sig, tf.TensorSpec([], tf.int32)],
                                   autograph=False, jit_compile=use_xla)

    def _evaluate(self, q):
        value, score = self.adapter.log_prob_and_grad(q)
        value, score = tf.convert_to_tensor(value), tf.convert_to_tensor(score)
        if value.dtype != tf.float64 or score.dtype != tf.float64 or value.shape != q.shape[:-1] or score.shape != q.shape:
            raise ValueError("target must return float64 rowwise values and full scores")
        if self.status_policy == "none":
            return value, score, {}, _finite_rows(q, value, score)
        from .hmc_verification import TARGET_STATUS_TELEMETRY_CORE_FIELDS
        status = self.adapter.target_status_telemetry(q)
        if any(k not in status for k in TARGET_STATUS_TELEMETRY_CORE_FIELDS):
            raise ValueError("incomplete target status")
        # Preserve numeric rowwise diagnostics (e.g. chart/filter residuals).
        # Only the declared health fields below affect admission.
        status = {k: tf.convert_to_tensor(v) for k, v in status.items()}
        if any(not (v.dtype.is_floating or v.dtype.is_integer or v.dtype == tf.bool) for v in status.values()):
            raise ValueError("target status must contain numeric rowwise tensors")
        if any(v.shape != value.shape for v in status.values()):
            raise ValueError("target status must preserve chain rows")
        if (not status["status_code"].dtype.is_integer or not status["floor_count_value"].dtype.is_integer
                or status["valid_pre_regularized_score"].dtype != tf.bool):
            raise ValueError("invalid target status dtypes")
        good = ((status["status_code"] == 0) & status["valid_pre_regularized_score"]
                & (status["floor_count_value"] == 0))
        for key in ("min_innovation_eigenvalue", "innovation_condition_estimate"):
            if key in status:
                good &= tf.math.is_finite(status[key]) & (status[key] > 0.)
        return value, score, status, good & _finite_rows(q, value, score)

    def _one_step(self, q, seed, epsilon, max_steps):
        shape = self.shape
        keys = tf.random.experimental.stateless_split(seed, 3)
        aux = tf.random.stateless_normal(shape, keys[0], dtype=q.dtype, alg="philox")
        if self.profile.family == "mams":
            aux = aux / tf.linalg.norm(aux, axis=-1, keepdims=True)
        if self.profile.trajectory == "uniform_duration":
            uniforms = tf.random.stateless_uniform(shape[:-1], keys[1], dtype=q.dtype, alg="philox")
            steps = tf.maximum(1, tf.cast(tf.math.ceil(uniforms * tf.cast(max_steps, q.dtype)), tf.int32))
        else:
            steps = tf.fill(shape[:-1], max_steps)
        value, score, status, good = self._evaluate(q)
        initial_value, initial_aux, initial_status = value, aux, status
        initial_score, initial_good = score, good
        work = tf.zeros(shape[:-1], q.dtype)
        last = tf.reduce_max(steps)
        # Stage codes: 0 none, 1 initial target, 2 first kick, 3 drift,
        # 4 proposed target, 5 second kick, 6 final energy correction.
        # Substeps are one-based; initial failures use 0, healthy rows -1.
        failure = {
            "substep": tf.where(good, -tf.ones_like(steps), tf.zeros_like(steps)),
            "stage": tf.where(good, tf.zeros_like(steps), tf.ones_like(steps)),
            "position": tf.where(good[:, None], tf.zeros_like(q), q),
            "position_finite": tf.reduce_all(tf.math.is_finite(q), axis=-1),
            "value_finite": tf.math.is_finite(value),
            "score_finite": tf.reduce_all(tf.math.is_finite(score), axis=-1),
            "momentum_finite": tf.reduce_all(tf.math.is_finite(aux), axis=-1),
            **{"target_" + k: tf.where(good, tf.zeros_like(v), v) for k, v in status.items()},
        }

        def record_failure(old, hit, substep, stage, position, value, score, momentum, status):
            proposed = {"substep": tf.fill(shape[:-1], substep), "stage": stage,
                "position": position,
                "position_finite": tf.reduce_all(tf.math.is_finite(position), axis=-1),
                "value_finite": tf.math.is_finite(value),
                "score_finite": tf.reduce_all(tf.math.is_finite(score), axis=-1),
                "momentum_finite": tf.reduce_all(tf.math.is_finite(momentum), axis=-1),
                **{"target_" + k: v for k, v in status.items()}}
            return {k: tf.where(hit[:, None] if v.shape.rank == 2 else hit, proposed[k], v)
                    for k, v in old.items()}

        def body(i, position, momentum, value, score, status, good, work, failure):
            active = i < steps
            if self.profile.family == "hmc":
                middle = momentum + .5 * epsilon * score
                first_work = tf.zeros_like(work)
            else:
                middle, first_work = sphere_force_substep(momentum, score, .5 * epsilon)
            next_q = tf.where(active[:, None], position + epsilon * middle, position)
            next_value, next_score, next_status, next_good = self._evaluate(next_q)
            if self.profile.family == "hmc":
                end = middle + .5 * epsilon * next_score
                second_work = tf.zeros_like(work)
            else:
                end, second_work = sphere_force_substep(middle, next_score, .5 * epsilon)
            first_ok = (tf.reduce_all(tf.math.is_finite(middle), axis=-1)
                        & tf.math.is_finite(first_work))
            drift_ok = tf.reduce_all(tf.math.is_finite(next_q), axis=-1)
            second_ok = (tf.reduce_all(tf.math.is_finite(end), axis=-1)
                         & tf.math.is_finite(second_work))
            step_good = first_ok & drift_ok & next_good & second_ok
            stage = tf.where(~first_ok, 2, tf.where(~drift_ok, 3,
                             tf.where(~next_good, 4, tf.where(~second_ok, 5, 0))))
            hit = active & (failure["stage"] == 0) & ~step_good
            failure = record_failure(failure, hit, i + 1, stage,
                tf.where(first_ok[:, None], next_q, position),
                tf.where(first_ok, next_value, value),
                tf.where(first_ok[:, None], next_score, score),
                tf.where((stage == 5)[:, None], end, middle),
                {k: tf.where(first_ok, next_status[k], status[k]) for k in status})
            good &= (~active) | step_good
            # Batched evaluations still occur for inactive rows (and are
            # charged), but must never overwrite a completed trajectory.
            return (i + 1, next_q, tf.where(active[:, None], end, momentum),
                    tf.where(active, next_value, value),
                    tf.where(active[:, None], next_score, score),
                    {k: tf.where(active, next_status[k], status[k]) for k in status}, good,
                    work + tf.where(active, first_work + second_work, tf.zeros_like(work)), failure)

        _, proposal, momentum, proposed_value, score, proposed_status, good, work, failure = tf.while_loop(
            lambda i, *_: i < last, body,
            (tf.constant(0), q, aux, value, score, status, good, work, failure), parallel_iterations=1)
        if self.profile.family == "hmc":
            correction = .5 * tf.reduce_sum(tf.square(initial_aux) - tf.square(momentum), axis=-1)
        else:
            correction = -work
        log_ratio = proposed_value - initial_value + correction
        ratio_finite = tf.math.is_finite(log_ratio)
        failure = record_failure(failure, (failure["stage"] == 0) & ~ratio_finite,
            last + 1, tf.fill(shape[:-1], 6), proposal, proposed_value, score, momentum, proposed_status)
        # Energy correction happens after each row's own terminal step.
        failure["substep"] = tf.where(failure["stage"] == 6, steps + 1, failure["substep"])
        good &= ratio_finite
        log_uniform = tf.math.log(tf.random.stateless_uniform(shape[:-1], keys[2], dtype=q.dtype, alg="philox"))
        accepted = good & (log_uniform < tf.minimum(tf.zeros_like(log_ratio), log_ratio))
        result = tf.where(accepted[:, None], proposal, q)
        # Report the rejection already made above without losing raw failure
        # evidence. A finite status failure stays a veto even if later steps
        # overflow: classify the first failure, never just the final NaN.
        numeric_failure = ((failure["stage"] == 2) | (failure["stage"] == 3)
            | (failure["stage"] == 5) | (failure["stage"] == 6)
            | ((failure["stage"] == 4) & (~failure["position_finite"]
                | ~failure["value_finite"] | ~failure["score_finite"])))
        numerical_rejection = initial_good & numeric_failure & ~accepted
        selected_score = tf.where(accepted[:, None], score, initial_score)
        trace = {"is_accepted": accepted,
                 "log_accept_ratio": tf.where(numerical_rejection,
                     tf.constant(-float("inf"), q.dtype), log_ratio),
                 "raw_log_accept_ratio": log_ratio,
                 "numerical_proposal_rejected": numerical_rejection,
                 "previous_target_valid": initial_good,
                 "previous_target_log_prob": initial_value,
                 "previous_target_score": initial_score,
                 "proposed_target_score": score,
                 "accepted_target_score": selected_score,
                 "accepted_target_score_finite": tf.reduce_all(tf.math.is_finite(selected_score), axis=-1),
                 "target_log_prob": tf.where(accepted, proposed_value, initial_value),
                 "proposed_target_log_prob": proposed_value, "proposed_state": proposal,
                 "initial_momentum": initial_aux, "final_momentum": momentum,
                 "log_acceptance_correction": correction, "target_score_finite": good,
                 "trajectory_steps": steps, "integration_work": work,
                 "value_score_calls_per_chain": tf.fill(shape[:-1], last + 1),
                 "target_status_calls_per_chain": tf.fill(shape[:-1], last + 1 if self.status_policy != "none" else 0)}
        trace.update({"first_failure_" + k: v for k, v in failure.items()})
        if self.status_policy != "none":
            trace["target_status_telemetry"] = {k: tf.where(accepted, proposed_status[k], initial_status[k]) for k in initial_status}
            trace["proposed_target_status_telemetry"] = proposed_status
        return result, trace

    def _chain(self, q, seed, epsilon, max_steps, offset):
        def step(current, index):
            key = tf.random.experimental.stateless_fold_in(seed, offset + index)
            return self._one_step(current, key, epsilon, max_steps)
        q, trace = step(q, tf.constant(0))
        states = tf.TensorArray(tf.float64, self.num_results, element_shape=self.shape).write(0, q)
        traces = tf.nest.map_structure(lambda x: tf.TensorArray(x.dtype, self.num_results,
                                        element_shape=x.shape).write(0, x), trace)
        def body(i, current, states, traces):
            current, trace = step(current, i)
            return (i+1, current, states.write(i, current),
                    tf.nest.map_structure(lambda array, x: array.write(i, x), traces, trace))
        _, _, states, traces = tf.while_loop(lambda i, *_: i < self.num_results,
            body, (tf.constant(1), q, states, traces), parallel_iterations=1)
        return states.stack(), tf.nest.map_structure(lambda array: array.stack(), traces)

    def run(self, *, current_state, seed, step_size, num_leapfrog_steps, transition_offset=0):
        from .hmc import FullChainHMCRunResult
        spec = self.profile.freeze(epsilon=step_size, max_steps=num_leapfrog_steps)
        if type(transition_offset) is not int or not 0 <= transition_offset <= 2**31-1-self.num_results:
            raise ValueError("invalid transition stream offset")
        state = tf.convert_to_tensor(current_state, tf.float64)
        if state.shape != self.shape or not bool(tf.reduce_all(tf.math.is_finite(state))):
            raise ValueError("invalid initial state")
        seed_tensor = tf.convert_to_tensor(seed, tf.int32)
        if seed_tensor.shape != (2,):
            raise ValueError("seed must have two int32 components")
        started = time.monotonic()
        samples, trace = self._sample(state, seed_tensor, tf.constant(spec.epsilon, tf.float64),
            tf.constant(spec.max_steps, tf.int32), tf.constant(transition_offset, tf.int32))
        # Materialization is a reporting boundary, not a NumPy computation path.
        calls = int(tf.reduce_sum(trace["value_score_calls_per_chain"]))
        status_calls = int(tf.reduce_sum(trace["target_status_calls_per_chain"]))
        self._call_count += 1
        return FullChainHMCRunResult(samples=samples, trace=trace, diagnostics={}, metadata={
            "transition_spec": spec.payload(), "sample_chain_call_s": time.monotonic()-started,
            "sample_chain_invocation_count": self._call_count, "execution_mode": "batched",
            "seed_layout": "independent_full_refresh_stateless_purpose_chain_rows",
            "value_score_row_evaluations": calls, "target_status_row_evaluations": status_calls,
            "maximum_reserved_gradient_work": self.num_results * int(self.shape[0]) * spec.conservative_gradient_work_per_transition,
            "samples_device": samples.device, "use_xla": self.use_xla,
            "transition_offset": transition_offset, "next_transition_offset": transition_offset+self.num_results})
