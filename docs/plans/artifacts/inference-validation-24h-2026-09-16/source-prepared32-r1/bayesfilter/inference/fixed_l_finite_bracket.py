"""Fail-closed contracts for finite, fixed-L epsilon initialization.

This is not a tuner. BayesFilter's ordinary tuner owns the candidate grid and
the TFP runner. These helpers keep an L-specific finite bracket from becoming
an unqualified global step-size claim. The step ceiling only bounds discarded
dual averaging; it never modifies a target, state, score, or MH correction.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any, Mapping


FiniteTransitionResults = namedtuple("FiniteTransitionResults", "inner_results health")


class FixedLFiniteBracketError(RuntimeError):
    """A terminal bracket failure, with a preserved lifecycle classification."""

    def __init__(self, failure_class: str, message: str, **record: Any) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.failure_record = {
            "failure_class": failure_class,
            "first_failed_assertion": message,
            "lifecycle_stage": "bracketing",
            **record,
        }


def classify_tuning_exception(adapter: Any, error: BaseException) -> str:
    """Do not relabel all backend exceptions as target-support failures."""

    if isinstance(error, FixedLFiniteBracketError):
        return error.failure_class
    classifier = getattr(adapter, "classify_target_exception", None)
    if callable(classifier):
        try:
            declared = classifier(error)
        except Exception:
            return "control_plane"
        if type(declared) is not bool:
            return "control_plane"
        if declared:
            return "target_domain"
    module = type(error).__module__
    if module.startswith("tensorflow"):
        if type(error).__name__ == "InvalidArgumentError" and any(marker in str(error) for marker in (
            "fixed-L finite transition:", "fixed-L adaptation step is nonfinite",
        )):
            return "nonfinite_trajectory"
        return "tensorflow"
    if module.startswith("bayesfilter"):
        return "bayesfilter"
    return "control_plane"


def bound_adaptation_step(step: Any, upper_bound: float) -> Any:
    """Bound a finite positive proposal step, never conceal NaN or Inf.

    This is the same setter ceiling used by BayesFilter's windowed warmup.
    The caller must derive the ceiling from its own finite fixed-L bracket.
    It is a tuning heuristic, not proof of finite future trajectories.
    """

    import tensorflow as tf

    value = tf.convert_to_tensor(step)
    checks = [
        tf.debugging.assert_all_finite(value, "fixed-L adaptation step is nonfinite"),
        tf.debugging.assert_positive(value, "fixed-L adaptation step must be positive"),
    ]
    with tf.control_dependencies(checks):
        return tf.minimum(value, tf.convert_to_tensor(upper_bound, dtype=value.dtype))


def bounded_dual_averaging_kwargs(policy: Any, initial_step: Any, dtype: Any) -> Mapping[str, Any]:
    """Bind the TFP setter and shrinkage target to the qualified ceiling."""

    if policy.step_size_upper_bound is None:
        return {}
    import tensorflow as tf
    from tensorflow_probability.python.internal import unnest

    def bounded_step_setter(kernel_results: Any, new_step_size: Any) -> Any:
        return unnest.replace_innermost(
            kernel_results,
            step_size=bound_adaptation_step(new_step_size, policy.step_size_upper_bound),
        )

    ceiling = tf.convert_to_tensor(policy.step_size_upper_bound, dtype=dtype)
    shrinkage = 10.0 * tf.minimum(tf.convert_to_tensor(initial_step, dtype=dtype), ceiling / 10.0)
    return {"step_size_setter_fn": bounded_step_setter, "shrinkage_target": shrinkage}


def finite_transition_health(pre_state: Any, kernel_results: Any) -> dict[str, Any]:
    """Compact endpoint health; missing evidence is false, not a success.

    Finiteness of proposal displacement is checked independently: subtracting
    two finite extreme coordinates can overflow. This trace does not recover
    a failed leapfrog substep; a future diagnostic must also record that
    target-side failure before it may claim complete first-failure coverage.
    """

    import tensorflow as tf

    def all_finite(value: Any) -> Any:
        if value is None:
            return tf.constant(False)
        leaves = tf.nest.flatten(value)
        if not leaves or any(leaf is None for leaf in leaves):
            return tf.constant(False)
        checks = [tf.reduce_all(tf.math.is_finite(leaf)) for leaf in leaves]
        return tf.reduce_all(checks)

    proposed = getattr(kernel_results, "proposed_results", None)
    accepted = getattr(kernel_results, "accepted_results", None)
    proposal = getattr(kernel_results, "proposed_state", None)
    displacement = None if proposal is None else tf.nest.map_structure(
        lambda proposed, previous: proposed - previous, proposal, pre_state
    )
    return {
        "target_score_finite": tf.logical_and(
            all_finite(getattr(proposed, "grads_target_log_prob", None)),
            all_finite(getattr(accepted, "grads_target_log_prob", None)),
        ),
        "proposal_finite": all_finite(proposal),
        "movement_finite": all_finite(displacement),
        "proposed_target_finite": all_finite(getattr(proposed, "target_log_prob", None)),
        "accepted_target_finite": all_finite(getattr(accepted, "target_log_prob", None)),
        "log_accept_ratio_finite": all_finite(getattr(kernel_results, "log_accept_ratio", None)),
    }


def guard_finite_transitions(inner_kernel: Any, *, uses_dual_averaging: bool) -> Any:
    """Check every attempted TFP transition, including discarded adaptation.

    ``sample_chain`` calls its trace after the MH decision and does not retain
    ordinary burn-in traces. Checking movement there would subtract a proposal
    from itself on acceptance, hiding overflow in ``proposal - pre_state``.
    This wrapper checks the real pre-state and preserves TFP's transition and
    MH correction unchanged. It is non-XLA pending separate qualification.
    A target exception before the inner kernel returns still requires a future
    target-side failed-substep recorder; endpoint checks cannot reconstruct it.
    """

    import tensorflow as tf
    import tensorflow_probability as tfp

    def endpoint_results(results: Any) -> Any:
        return results.inner_results if uses_dual_averaging else results

    class FiniteTransitionGuard(tfp.mcmc.TransitionKernel):
        @property
        def is_calibrated(self) -> bool:
            return inner_kernel.is_calibrated

        @property
        def parameters(self) -> Mapping[str, Any]:
            return {"inner_kernel": inner_kernel}

        def bootstrap_results(self, init_state: Any) -> Any:
            results = inner_kernel.bootstrap_results(init_state)
            return FiniteTransitionResults(
                results, finite_transition_health(init_state, endpoint_results(results))
            )

        def one_step(self, current_state: Any, previous_kernel_results: Any, seed: Any = None) -> Any:
            next_state, results = inner_kernel.one_step(
                current_state, previous_kernel_results.inner_results, seed=seed
            )
            health = finite_transition_health(current_state, endpoint_results(results))
            health["returned_state_finite"] = tf.reduce_all([
                tf.reduce_all(tf.math.is_finite(part)) for part in tf.nest.flatten(next_state)
            ])
            checks = [
                tf.debugging.assert_equal(value, True, message=f"fixed-L finite transition: {name}")
                for name, value in health.items()
            ]
            if uses_dual_averaging:
                checks.append(tf.debugging.assert_all_finite(
                    results.new_step_size, "fixed-L adaptation step is nonfinite"
                ))
            with tf.control_dependencies(checks):
                next_state = tf.nest.map_structure(tf.identity, next_state)
                results = tf.nest.map_structure(tf.identity, results)
            health.pop("returned_state_finite")
            return next_state, FiniteTransitionResults(results, health)

    return FiniteTransitionGuard()
