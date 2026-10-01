"""Experimental uncertainty analysis of fixed-kernel acceptance traces.

This module has no tuning-artifact authority. It preserves the existing health
evaluator and reports a separate recommendation. Marginal batch-means intervals
are approximate; simultaneous subtraction avoids assuming independent windows.
No result establishes stationarity, convergence, burn-in sufficiency or a
finite-sample error guarantee. See the 2026-10-01 validation plan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import tensorflow as tf
import tensorflow_probability as tfp

from .mcmc_uncertainty import chain_batch_covariance

UNCERTAINTY_VERSION = "bayesfilter.acceptance_uncertainty.experimental.v1"
DECISIONS = (
    "invalid_probability_trace", "insufficient_information",
    "material_conflict", "repair_step_lower", "repair_step_higher",
    "compatible_for_fresh_verification", "unresolved_contrasts",
    "unresolved_acceptance",
)


@dataclass(frozen=True)
class AcceptanceUncertaintyPolicy:
    # Deliberately explicit: no universal information or discrepancy default.
    batch_size: int
    min_batches: int
    family_alpha: float
    temporal_tolerance: float
    chain_tolerance: float
    planned_looks: int
    planned_candidates: int
    method: str = "lugsail"
    practical_region: tuple[float, float] = (.65, .75)
    repair_region: tuple[float, float] = (.55, .85)
    jit_compile: bool = True

    def __post_init__(self):
        for name, minimum in (("batch_size", 3), ("min_batches", 2),
                              ("planned_looks", 1), ("planned_candidates", 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"invalid {name}")
        for name in ("family_alpha", "temporal_tolerance", "chain_tolerance"):
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be in (0,1)")
        if self.method not in {"batch_means", "lugsail"}:
            raise ValueError("unsupported uncertainty method")
        for name in ("practical_region", "repair_region"):
            value = tuple(getattr(self, name))
            if len(value) != 2 or not 0 < value[0] < value[1] < 1:
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        if not (self.repair_region[0] <= self.practical_region[0]
                and self.practical_region[1] <= self.repair_region[1]):
            raise ValueError("repair_region must contain practical_region")
        if type(self.jit_compile) is not bool:
            raise TypeError("jit_compile must be boolean")

    def payload(self):
        return {"schema": UNCERTAINTY_VERSION, **asdict(self),
                "batch_sizes": (self.batch_size, 2*self.batch_size),
                "window_count": 4, "lugsail_r": 3, "lugsail_c": .5,
                "interval_method": "approximate_t_batch_count_minus_one",
                "comparison_method": "simultaneous_marginal_interval_subtraction",
                "cross_window_covariance": "not_estimated; arbitrary_dependence_allowed_by_subtraction",
                "error_control": "Bonferroni_conditional_on_valid_marginal_intervals",
                "tuning_artifact_authority": False}

    @classmethod
    def from_payload(cls, payload):
        from dataclasses import fields
        policy = cls(**{f.name: payload[f.name] for f in fields(cls)})
        import json
        if json.dumps(policy.payload(), sort_keys=True) != json.dumps(payload, sort_keys=True):
            raise ValueError("uncertainty policy payload mismatch")
        return policy


@lru_cache(maxsize=128)
def _critical(df, tail):
    # Scalar configuration arithmetic, not a per-transition kernel. TFP's
    # inverse-beta implementation is traced once; the numerical trace kernels
    # below use XLA according to the caller's explicit policy.
    @tf.function(input_signature=[tf.TensorSpec([2], tf.float64)],
                 autograph=False, jit_compile=False)
    def compute(values):
        return tfp.distributions.StudentT(values[0], tf.constant(0., tf.float64),
            tf.constant(1., tf.float64)).quantile(values[1])
    return float(compute(tf.constant([max(df, 1), 1.-tail], tf.float64)))


@lru_cache(maxsize=16)
def _analysis_program(shape, policy):
    replications, n, chains = shape
    size = n // 4
    temporal_pairs = tuple((a, b) for a in range(4) for b in range(a+1, 4))
    chain_pairs = tuple((a, b) for a in range(chains) for b in range(a+1, chains))
    # Whole means, pooled mean, and windows, at two batch sizes. Contrasts
    # follow from these intervals without another comparison-independence claim.
    family_size = 2 * (1 + chains + 4*chains)
    tail = policy.family_alpha / (2 * family_size * policy.planned_looks * policy.planned_candidates)
    if not 0. < tail < .5 or not 1.-tail < 1.:
        raise ValueError("marginal alpha is below float64 probability resolution")
    criticals = tuple((_critical(n//b-1, tail), _critical(size//b-1, tail))
                     for b in (policy.batch_size, 2*policy.batch_size))

    @tf.function(input_signature=[tf.TensorSpec(shape, tf.float64)],
                 autograph=False, jit_compile=policy.jit_compile)
    def compute(probability):
        windows = tf.reshape(probability[:, :4*size], [replications*4, size, chains, 1])
        whole_reports, window_reports = [], []
        for b in (policy.batch_size, 2*policy.batch_size):
            args = dict(batch_size=b, min_batches=policy.min_batches, method=policy.method,
                        jit_compile=policy.jit_compile)
            whole_reports.append(chain_batch_covariance(probability[..., None], **args))
            window_reports.append(chain_batch_covariance(windows, **args))
        means = whole_reports[0]["mean_by_chain"][..., 0]
        block_means = tf.reshape(window_reports[0]["mean_by_chain"], [replications, 4, chains])
        pooled = tf.reduce_mean(means, axis=1)
        full_h = tf.reduce_max(tf.stack([c[0]*r["mcse_by_chain"][..., 0]
                                        for c, r in zip(criticals, whole_reports)]), axis=0)
        block_h = tf.reshape(tf.reduce_max(tf.stack([c[1]*r["mcse_by_chain"][..., 0]
                                        for c, r in zip(criticals, window_reports)]), axis=0),
                             [replications, 4, chains])
        pooled_h = tf.reduce_max(tf.stack([c[0]*tf.sqrt(r["pooled_variance_of_mean"][..., 0])
                                          for c, r in zip(criticals, whole_reports)]), axis=0)
        valid = tf.reduce_all(tf.stack([r["valid_by_chain"][..., 0] for r in whole_reports]), axis=(0, 2))
        valid &= tf.reduce_all(tf.reshape(tf.stack([r["valid_by_chain"][..., 0]
                             for r in window_reports]), [2, replications, 4, chains]), axis=(0, 2, 3))
        input_valid = tf.reduce_all(tf.math.is_finite(probability) & (probability >= 0.)
                                   & (probability <= 1.), axis=(1, 2))
        duplicate_chains = tf.reduce_any(tf.stack([
            tf.reduce_all(probability[:, :, a] == probability[:, :, b], axis=1)
            for a, b in chain_pairs]), axis=0)
        # Exact duplicate stochastic traces cannot support the declared
        # independent-chain variance reduction. Constants also fail above.
        valid &= ~duplicate_chains
        temporal = tf.stack([block_means[:, b] - block_means[:, a] for a, b in temporal_pairs], axis=1)
        temporal_h = tf.stack([block_h[:, b] + block_h[:, a] for a, b in temporal_pairs], axis=1)
        cross_chain = tf.stack([means[:, b] - means[:, a] for a, b in chain_pairs], axis=1)
        cross_h = tf.stack([full_h[:, b] + full_h[:, a] for a, b in chain_pairs], axis=1)
        temporal_conflict = tf.reduce_any(tf.abs(temporal) - temporal_h > policy.temporal_tolerance, axis=(1, 2))
        chain_conflict = tf.reduce_any(tf.abs(cross_chain) - cross_h > policy.chain_tolerance, axis=1)
        temporal_compatible = tf.reduce_all(tf.abs(temporal) + temporal_h <= policy.temporal_tolerance, axis=(1, 2))
        chain_compatible = tf.reduce_all(tf.abs(cross_chain) + cross_h <= policy.chain_tolerance, axis=1)
        lower, upper = pooled-pooled_h, pooled+pooled_h
        low, high = policy.practical_region
        repair_low, repair_high = policy.repair_region
        decision = tf.fill([replications], 7)
        decision = tf.where((lower <= high) & (upper >= low) & (lower >= repair_low)
                            & (upper <= repair_high), 5, decision)
        decision = tf.where(~(temporal_compatible & chain_compatible), 6, decision)
        decision = tf.where(lower > high, 4, decision)
        decision = tf.where(upper < low, 3, decision)
        decision = tf.where(temporal_conflict | chain_conflict, 2, decision)
        decision = tf.where(~valid, 1, decision)
        decision = tf.where(~input_valid, 0, decision)
        return {
            "decision_code": decision, "variance_estimate_available": valid & input_valid,
            "duplicate_chain_traces": duplicate_chains,
            "pooled_mean": pooled, "pooled_interval": tf.stack([lower, upper], axis=-1),
            "chain_means": means, "chain_half_width": full_h,
            "window_means": block_means, "window_half_width": block_h,
            "temporal_contrasts": temporal, "temporal_half_width": temporal_h,
            "chain_contrasts": cross_chain, "chain_contrast_half_width": cross_h,
            "temporal_conflict": temporal_conflict & valid & input_valid,
            "chain_conflict": chain_conflict & valid & input_valid,
            "temporal_compatible": temporal_compatible & valid & input_valid,
            "chain_compatible": chain_compatible & valid & input_valid,
            "chain_lrv_by_scale": tf.stack([r["long_run_covariance_by_chain"][..., 0, 0]
                                             for r in whole_reports], axis=1),
            "chain_mcse_by_scale": tf.stack([r["mcse_by_chain"][..., 0] for r in whole_reports], axis=1),
            "chain_information_by_scale": tf.stack([r["effective_information_by_chain"][..., 0]
                                                     for r in whole_reports], axis=1),
            "window_lrv_by_scale": tf.reshape(tf.stack([r["long_run_covariance_by_chain"][..., 0, 0]
                                for r in window_reports], axis=1), [replications, 4, 2, chains]),
            "window_information_by_scale": tf.reshape(tf.stack([r["effective_information_by_chain"][..., 0]
                                for r in window_reports], axis=1), [replications, 4, 2, chains]),
            "window_valid_by_scale": tf.reshape(tf.stack([r["valid_by_chain"][..., 0]
                                for r in window_reports], axis=1), [replications, 4, 2, chains]),
        }
    return compute


def acceptance_uncertainty_tensors(probability, *, policy):
    """Batched diagnostic interface [replication, draw, chain], no health claims."""
    if not isinstance(policy, AcceptanceUncertaintyPolicy):
        raise TypeError("typed uncertainty policy required")
    values = (tf.cast(probability, tf.float64) if tf.is_tensor(probability)
              else tf.convert_to_tensor(probability, tf.float64))
    if values.shape.rank != 3 or any(d is None or d < 1 for d in values.shape):
        raise ValueError("static [replication, draw, chain] required")
    if values.shape[1] < 8 or values.shape[2] < 2:
        raise ValueError("at least eight draws and two chains required")
    return _analysis_program(tuple(map(int, values.shape)), policy)(values)


def _json_finite(value):
    if tf.is_tensor(value):
        value = value.numpy().tolist()  # Serialization only, no NumPy arithmetic.
    if isinstance(value, (tuple, list)):
        return [_json_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def acceptance_uncertainty_report(probability, *, policy):
    values = (tf.cast(probability, tf.float64) if tf.is_tensor(probability)
              else tf.convert_to_tensor(probability, tf.float64))
    if values.shape.rank != 2:
        raise ValueError("[draw, chain] probability trace required")
    tensors = acceptance_uncertainty_tensors(values[None], policy=policy)
    result = {k: _json_finite(v[0]) for k, v in tensors.items()}
    n, m = map(int, values.shape)
    counts = [(n//4)//b for b in (policy.batch_size, 2*policy.batch_size)]
    reasons = []
    if min(counts) < policy.min_batches:
        reasons.append("insufficient_complete_batches_at_declared_scales")
    elif not result["variance_estimate_available"] and not result["duplicate_chain_traces"]:
        reasons.append("nonpositive_nonfinite_or_constant_trace_variance")
    if result["duplicate_chain_traces"]:
        reasons.append("duplicate_traces_do_not_support_independent_chain_variance")
    if result["decision_code"] == 0:
        reasons.append("invalid_probability_values")
    if result["decision_code"] == 6:
        reasons.append("contrast_intervals_do_not_resolve_declared_tolerances")
    if result["decision_code"] == 7:
        reasons.append("acceptance_interval_compatibility_unresolved")
    return {"schema": UNCERTAINTY_VERSION, "policy": policy.payload(), **result,
            "experimental_decision": DECISIONS[result["decision_code"]],
            "draws_per_chain": n, "chain_count": m,
            "excluded_window_remainder": n % 4,
            "batch_counts_by_scale": [n//b for b in (policy.batch_size, 2*policy.batch_size)],
            "window_batch_counts_by_scale": counts,
            "insufficiency_reasons": reasons,
            "tuning_artifact_authority": False,
            "interpretation": "conditional_on_local_stationarity_and_valid_marginal_intervals"}


def evaluate_acceptance_uncertainty(*, samples, log_accept_ratio, is_accepted,
                                  uncertainty_policy, health_policy=None,
                                  fixed_kernel, support_rejection=None, **health_kwargs):
    """Analyze a trace alongside the unchanged public health/admission evaluator.

    Explicit support rejections have probability zero and must be rejected.
    -inf is allowed only at those positions; other nonfinite ratios remain
    invalid under the existing evaluator. R-hat is intentionally not an input.
    This diagnostic cannot issue a candidate receipt or replace verification.
    """
    from .hmc_verification import HMCAcceptancePolicy, evaluate_hmc_acceptance_evidence
    ratios = (tf.cast(log_accept_ratio, tf.float64) if tf.is_tensor(log_accept_ratio)
              else tf.convert_to_tensor(log_accept_ratio, tf.float64))
    accepted = tf.convert_to_tensor(is_accepted)
    health_ratios = ratios
    probability = tf.exp(tf.minimum(ratios, 0.))
    if support_rejection is not None:
        mask = tf.convert_to_tensor(support_rejection)
        if mask.dtype != tf.bool or mask.shape != ratios.shape or accepted.shape != mask.shape:
            raise ValueError("aligned boolean support-rejection trace required")
        allowed = tf.math.is_finite(ratios) | (tf.math.is_inf(ratios) & (ratios < 0.))
        if bool(tf.reduce_any(mask & (accepted | ~allowed))):
            raise ValueError("invalid support rejection")
        probability = tf.where(mask, 0., probability)
        # Existing health arithmetic requires finite log ratios. This sentinel
        # represents numerical zero probability, not a finite Hamiltonian claim.
        health_ratios = tf.where(mask, tf.constant(-1.e300, tf.float64), ratios)
    policy = health_policy or HMCAcceptancePolicy()
    legacy = evaluate_hmc_acceptance_evidence(samples=samples, log_accept_ratio=health_ratios,
        is_accepted=accepted, policy=policy, **health_kwargs)
    if type(fixed_kernel) is not bool:
        raise TypeError("fixed_kernel must be boolean")
    if legacy.evidence_validity != "valid" or not fixed_kernel:
        diagnostic = {"experimental_decision": "unavailable_invalid_or_nonfixed_kernel"}
    else:
        diagnostic = acceptance_uncertainty_report(probability, policy=uncertainty_policy)
    recommendation = diagnostic["experimental_decision"]
    if legacy.candidate_promotion_vetoes or legacy.cost_stop_reasons:
        recommendation = "blocked_by_health_or_cost"
    return {"schema": UNCERTAINTY_VERSION, "legacy_evidence": legacy.payload(),
            "uncertainty": diagnostic, "recommendation": recommendation,
            "fixed_kernel_declared": fixed_kernel,
            "tuning_artifact_authority": False, "historical_decision_changed": False}
