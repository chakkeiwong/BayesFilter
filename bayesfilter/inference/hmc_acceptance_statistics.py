"""Experimental bounded uncertainty for independent fixed-horizon trial scores.

TensorFlow computes summaries and intervals; host code validates the declared
protocol and classifies decisions. These statistics have no artifact authority.
See the October 2 repair plan for the finite-trial estimand and error derivation.
"""
from __future__ import annotations

from functools import lru_cache
import math
from typing import Any

import tensorflow as tf

from .hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy


STATISTICS_SCHEMA = "bayesfilter.hmc_replicated_acceptance_statistics.v1"
_EPS = 2.0**-52
_ROUNDING_GUARD = 64.0  # Conservative numerical hypothesis; reference-tested, not a hardware theorem.


def _tensor(values):
    return tf.cast(values, tf.float64) if tf.is_tensor(values) else tf.convert_to_tensor(values, tf.float64)


@lru_cache(maxsize=32)
def _bounds_program(columns: int, bets: tuple[float, ...], method: str, jit: bool):
    @tf.function(input_signature=[tf.TensorSpec([None, columns], tf.float64),
        tf.TensorSpec([], tf.float64)], autograph=False, jit_compile=jit)
    def calculate(values, sided_alpha):
        n = tf.cast(tf.shape(values)[0], tf.float64)
        mean = tf.reduce_mean(values, axis=0)
        log_threshold = -tf.math.log(sided_alpha)
        rounding = _ROUNDING_GUARD * _EPS * (n + len(bets) + 1.)
        if method == "bounded_hoeffding_rungs_v1":
            half = tf.sqrt(log_threshold / (2. * n)) + rounding
            return tf.maximum(mean-half, 0.), tf.minimum(mean+half, 1.)
        fractions = tf.constant(bets, tf.float64)[None, None, :]

        def crossed(threshold, sign):
            terms = sign * (values[:, :, None] - threshold[None, :, None]) * fractions
            # log1p(-1)=-inf is legitimate zero betting capital.
            logs = tf.reduce_sum(tf.math.log1p(terms), axis=0)
            capital = tf.reduce_logsumexp(logs, axis=1) - tf.math.log(tf.cast(len(bets), tf.float64))
            guard = rounding * (1. + tf.abs(log_threshold) + tf.abs(capital))
            return capital > log_threshold + guard

        def body(index, low_left, low_right, high_left, high_right):
            low_mid = .5 * (low_left + low_right)
            high_mid = .5 * (high_left + high_right)
            lower_crossed = crossed(low_mid, 1.)
            upper_crossed = crossed(high_mid, -1.)
            return (index+1, tf.where(lower_crossed, low_mid, low_left),
                tf.where(lower_crossed, low_right, low_mid),
                tf.where(upper_crossed, high_left, high_mid),
                tf.where(upper_crossed, high_mid, high_right))

        initial = (tf.constant(0), tf.zeros([columns], tf.float64), tf.ones([columns], tf.float64),
                   tf.zeros([columns], tf.float64), tf.ones([columns], tf.float64))
        _, lower, _, _, upper = tf.while_loop(lambda i, *bounds: i < 52, body, initial)
        # Keep the conservative side of each bisection bracket, with an extra
        # arithmetic margin. This is not a substitute for numerical parity tests.
        return tf.maximum(lower-rounding, 0.), tf.minimum(upper+rounding, 1.)
    return calculate


def bounded_trial_intervals(values, *, sided_alpha: float, method: str,
                            bets=(1., .5, .25, .125), jit_compile=True):
    """One scheduled-look interval, simultaneous over looks for the betting method.

    For Hoeffding the caller must spend sided_alpha across scheduled looks.
    Independent trial rows, a fixed conditional mean and [0,1] scores are
    assumptions; checking array shape cannot establish stochastic independence.
    """
    data = _tensor(values)
    if data.shape.rank != 2 or any(d is None or d < 1 for d in data.shape):
        raise ValueError("complete [repetition, statistic] scores are required")
    if not bool(tf.reduce_all(tf.math.is_finite(data) & (data >= 0.) & (data <= 1.))):
        raise ValueError("trial scores must be finite and within [0,1]")
    if type(sided_alpha) not in (int, float) or not math.isfinite(sided_alpha) or not 0 < sided_alpha < 1:
        raise ValueError("invalid one-sided error allocation")
    if method not in {"bounded_betting_mixture_v1", "bounded_hoeffding_rungs_v1"}:
        raise ValueError("unsupported interval method")
    if not bets or any(not math.isfinite(b) or not 0 < b <= 1 for b in bets):
        raise ValueError("fixed bets must lie in (0,1]")
    if type(jit_compile) is not bool:
        raise TypeError("jit_compile must be boolean")
    return _bounds_program(int(data.shape[1]), tuple(bets), method, jit_compile)(
        data, tf.constant(sided_alpha, tf.float64))


@lru_cache(maxsize=2)
def _statistics_program(jit: bool):
    @tf.function(input_signature=[tf.TensorSpec([None, 4], tf.float64), tf.TensorSpec([4], tf.float64)],
                 autograph=False, jit_compile=jit)
    def calculate(scores, weights):
        pooled = tf.reduce_sum(scores*weights, axis=1)
        values = tf.concat([pooled[:, None], scores], axis=1)
        n = tf.cast(tf.shape(values)[0], tf.float64)
        mean = tf.reduce_mean(values, axis=0)
        centered = values-mean
        covariance = tf.matmul(centered, centered, transpose_a=True) / tf.maximum(n-1., 1.) / n
        return values, mean, covariance
    return calculate


@lru_cache(maxsize=4)
def _intersection_program(columns: int, jit: bool):
    @tf.function(input_signature=[tf.TensorSpec([None, columns], tf.float64),
                                  tf.TensorSpec([None, columns], tf.float64)],
                 autograph=False, jit_compile=jit)
    def calculate(lower, upper):
        return tf.reduce_max(lower, axis=0), tf.reduce_min(upper, axis=0)
    return calculate


def _interval_sequence(values, policy, stage, rungs, *, jit_compile, diagnostic=False):
    n, columns = map(int, values.shape)
    targets = tuple(policy.repetition_target(r) for r in rungs)
    if n not in targets:
        raise ValueError("decisions require a complete declared repetition rung")
    alpha = (policy.diagnostic_family_alpha / (2 * 3 * policy.max_candidates * columns)
             if diagnostic else policy.one_sided_alpha(stage))
    if policy.method == "bounded_hoeffding_rungs_v1":
        alpha /= len(targets)
    lower, upper = [], []
    for target in targets:
        if target > n:
            break
        lo, hi = bounded_trial_intervals(values[:target], sided_alpha=alpha,
            method=policy.method, bets=policy.bet_fractions, jit_compile=jit_compile)
        lower.append(lo)
        upper.append(hi)
    lo, hi = _intersection_program(columns, jit_compile)(tf.stack(lower), tf.stack(upper))
    return lo, hi, alpha


def _list(values):
    return values.numpy().tolist()  # Host serialization only.


_WINDOW_PAIRS = tuple((a, b) for a in range(4) for b in range(a+1, 4))


@lru_cache(maxsize=2)
def _temporal_summary_program(jit: bool):
    @tf.function(input_signature=[tf.TensorSpec([None, 4, 4], tf.float64)],
                 autograph=False, jit_compile=jit)
    def calculate(windows):
        differences = tf.stack([windows[:, :, b]-windows[:, :, a] for a, b in _WINDOW_PAIRS], axis=2)
        return (tf.reshape(differences, [-1, 24])+1.)/2., tf.reduce_mean(differences, axis=0)
    return calculate


@lru_cache(maxsize=2)
def _temporal_result_program(jit: bool):
    @tf.function(input_signature=[tf.TensorSpec([24], tf.float64), tf.TensorSpec([24], tf.float64),
                                  tf.TensorSpec([], tf.float64)], autograph=False, jit_compile=jit)
    def calculate(lower, upper, tolerance):
        lo, hi = tf.reshape(2.*lower-1., [4, 6]), tf.reshape(2.*upper-1., [4, 6])
        return lo, hi, tf.reduce_any((lo > tolerance) | (hi < -tolerance), axis=1)
    return calculate


@lru_cache(maxsize=2)
def _trial_summary_program(jit: bool):
    @tf.function(input_signature=[tf.TensorSpec([None, 4], tf.float64)],
                 autograph=False, jit_compile=jit)
    def calculate(log_accept):
        probability = tf.exp(tf.minimum(log_accept, 0.))
        count = tf.shape(probability)[0]
        windows = tf.stack([
            tf.reduce_mean(probability[count*i//4:count*(i+1)//4], axis=0)
            for i in range(4)
        ], axis=1)
        return tf.reduce_mean(probability, axis=0), windows
    return calculate


def complete_trial_scores(log_accept_ratio: Any, *, policy: HMCReplicatedAcceptancePolicy,
                          jit_compile: bool = True) -> dict[str, Any]:
    """Summarize every measured transition; the caller has removed exactly W.

    The four temporal windows partition T even when it is not divisible by four.
    They are diagnostics, never independent observations of the trial mean.
    """
    if not isinstance(policy, HMCReplicatedAcceptancePolicy):
        raise TypeError("replicated acceptance policy required")
    data = _tensor(log_accept_ratio)
    if tuple(data.shape) != (policy.trial_num_results, policy.chain_count):
        raise ValueError("a complete measured trial with the frozen horizon is required")
    if not bool(tf.reduce_all(tf.math.is_finite(data))):
        raise ValueError("invalid acceptance trace cannot supply a trial score")
    if type(jit_compile) is not bool:
        raise TypeError("jit_compile must be boolean")
    means, windows = _trial_summary_program(jit_compile)(data)
    return {"start_scores": _list(means), "window_scores": _list(windows),
            "measured_transitions_per_start": policy.trial_num_results,
            "excluded_remainder_per_start": 0,
            "window_boundaries": tuple(policy.trial_num_results*i//4 for i in range(5))}


def replicated_acceptance_statistics(scores: Any, *, policy: HMCReplicatedAcceptancePolicy,
                                     stage: str, evidence_rungs: tuple[int, ...],
                                     window_scores: Any | None = None,
                                     jit_compile: bool = True) -> dict[str, Any]:
    """Classify complete trials; numerical/trajectory vetoes are applied separately."""
    if not isinstance(policy, HMCReplicatedAcceptancePolicy):
        raise TypeError("replicated acceptance policy required")
    data = _tensor(scores)
    if data.shape.rank != 2 or data.shape[1] != policy.chain_count:
        raise ValueError("scores must preserve [repetition, start] axes")
    if (data.shape[0] is None or not 1 <= int(data.shape[0]) <= policy.max_repetitions
            or not bool(tf.reduce_all(tf.math.is_finite(data) & (data >= 0.) & (data <= 1.)))):
        raise ValueError("invalid complete trial scores")
    # Validate the schedule even if the completed count happens to be a rung.
    from .hmc_acceptance_protocol import replicated_evidence_preflight
    replicated_evidence_preflight(policy, evidence_rungs=evidence_rungs,
        candidate_cap=policy.max_candidates, leapfrog_steps=(1,))
    statistics, means, covariance = _statistics_program(jit_compile)(data, tf.constant(policy.start_weights, tf.float64))
    lower, upper, alpha = _interval_sequence(statistics, policy, stage, evidence_rungs,
                                            jit_compile=jit_compile)
    lo, hi = _list(lower), _list(upper)
    pref_lo, pref_hi = policy.preferred_region
    qual_lo, qual_hi = policy.qualification_region
    reason = "boundary_or_precision_unresolved"
    if any(a > b for a, b in zip(lo, hi)):
        decision, reason = "inconclusive_preparation", "confidence_set_empty"
    elif hi[0] < pref_lo:
        decision, reason = "repair_step_lower", "supported_below_preferred_region"
    elif lo[0] > pref_hi:
        decision, reason = "repair_step_higher", "supported_above_preferred_region"
    elif any(b < qual_lo or a > qual_hi for a, b in zip(lo[1:], hi[1:])):
        decision, reason = "inconclusive_preparation", "supported_start_qualification_violation"
    elif all(a >= qual_lo and b <= qual_hi for a, b in zip(lo[1:], hi[1:])):
        decision, reason = "passed", "simultaneous_start_qualification_containment"
    else:
        decision = "inconclusive_evidence"
    result = {"schema": STATISTICS_SCHEMA, "policy_id": policy.identity,
        "stage": stage, "trial_count": int(data.shape[0]), "acceptance_decision": decision,
        "decision_reason": reason, "pooled_mean": float(means[0]),
        "start_means": _list(means[1:]), "pooled_interval": (lo[0], hi[0]),
        "start_intervals": tuple(zip(lo[1:], hi[1:])),
        "covariance_of_mean": _list(covariance) if data.shape[0] > 1 else None,
        "one_sided_alpha": alpha, "planned_candidate_cap": policy.max_candidates,
        "interval_method": policy.method, "rungs": evidence_rungs,
        "numerical_guard": {"float64_epsilon": _EPS, "rounding_multiplier": _ROUNDING_GUARD,
             "status": "conservative_reference_tested_hypothesis_not_formal_hardware_error_bound"},
        "qualification_claim": "each_finite_trial_start_expectation_in_declared_qualification_region",
        "tuning_artifact_authority": False, "temporal_diagnostic_role": "reporting_only"}
    if window_scores is not None:
        windows = _tensor(window_scores)
        if tuple(windows.shape) != (int(data.shape[0]), policy.chain_count, 4):
            raise ValueError("window scores must preserve [repetition, start, window]")
        if not bool(tf.reduce_all(tf.math.is_finite(windows) & (windows >= 0.) & (windows <= 1.))):
            raise ValueError("invalid temporal window scores")
        contrasts, temporal_means = _temporal_summary_program(jit_compile)(windows)
        dl, du, da = _interval_sequence(contrasts, policy, stage, evidence_rungs,
                                       jit_compile=jit_compile, diagnostic=True)
        dl, du, changed = _temporal_result_program(jit_compile)(dl, du, tf.constant(policy.temporal_tolerance, tf.float64))
        result["temporal"] = {"pairs": _WINDOW_PAIRS, "means_by_start": _list(temporal_means),
            "lower_by_start": _list(dl), "upper_by_start": _list(du),
            "supported_material_change_by_start": _list(changed),
            "one_sided_alpha": da, "admission_effect": "none"}
    return result
