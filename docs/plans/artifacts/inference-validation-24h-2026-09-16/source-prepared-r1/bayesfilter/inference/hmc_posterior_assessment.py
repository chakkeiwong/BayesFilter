"""Common posterior assessments for all sequential exact-transition routes.

The inherited warmup counts/thresholds are operational heuristics. Optional ESS,
persistence and precision requirements are explicit, never tuning admission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping
import tensorflow as tf

from bayesfilter.inference.hmc_precision import HMCPrecisionPolicy, mean_precision, precision_report
from bayesfilter.inference.hmc_posterior_diagnostics import rank_normalized_bulk_tail_ess


@dataclass(frozen=True)
class HMCPosteriorAssessmentPolicy:
    warmup_bulk_ess_min: float = 0.
    warmup_tail_ess_min: float = 0.
    retained_bulk_ess_min: float = 0.
    retained_tail_ess_min: float = 0.
    warmup_consecutive_checks: int = 1
    precision: HMCPrecisionPolicy | None = None
    quantities_id: str | None = None

    def __post_init__(self):
        for name in ("warmup_bulk_ess_min", "warmup_tail_ess_min", "retained_bulk_ess_min", "retained_tail_ess_min"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0.:
                raise ValueError("ESS information floors must be finite and nonnegative")
        if type(self.warmup_consecutive_checks) is not int or self.warmup_consecutive_checks < 1:
            raise ValueError("warmup_consecutive_checks must be positive")
        if self.precision is not None and not isinstance(self.precision, HMCPrecisionPolicy):
            raise TypeError("precision must be HMCPrecisionPolicy")
        if self.quantities_id is not None and (not isinstance(self.quantities_id, str) or not self.quantities_id):
            raise ValueError("quantities_id must be a nonempty string")

    def payload(self):
        return {"schema": "bayesfilter.hmc_posterior_assessment.v1", **asdict(self),
                "precision": None if self.precision is None else self.precision.payload(),
                "warmup_interpretation": "declared checks only; no stationarity proof",
                "defaults_provenance": "inherited R-hat-only policy; additional requirements explicitly configured"}


def assess_posterior(samples, names, *, policy, stage, rhat, extra=None, quantities_fn=None):
    policy = policy or HMCPosteriorAssessmentPolicy()
    if not isinstance(policy, HMCPosteriorAssessmentPolicy) or stage not in {"warmup", "retained"}:
        raise ValueError("invalid posterior assessment policy or stage")
    samples = tf.convert_to_tensor(samples, tf.float64)
    if samples.shape.rank != 3 or any(d is None for d in samples.shape):
        raise ValueError("posterior samples require static [draw, chain, quantity]")
    if int(samples.shape[0]) < 4:
        return {"passed": False, "stage": stage, "status": "insufficient_draws",
                "precision": {"status": "precision_unavailable", "passed": False}}
    names = tuple(names)
    if len(set(names)) != len(names) or len(names) != samples.shape[-1]:
        raise ValueError("posterior names must uniquely cover model coordinates")
    if extra is not None and (not isinstance(extra, Mapping) or type(extra.get("passed")) is not bool):
        raise ValueError("retained_diagnostic_fn must return a mapping with boolean passed")
    values = samples
    if policy.quantities_id is not None and quantities_fn is None:
        raise ValueError("declared quantities_id requires quantities_fn")
    if quantities_fn is not None:
        if not policy.quantities_id:
            raise ValueError("additional quantities require a declared quantities_id")
        quantities = quantities_fn(samples)
        if not isinstance(quantities, Mapping) or any(not isinstance(k, str) or not k or k in names for k in quantities):
            raise ValueError("additional quantities must have distinct names")
        parts = []
        for key, value in quantities.items():
            tensor = tf.cast(tf.convert_to_tensor(value), tf.float64)
            if tensor.shape != samples.shape[:2]:
                raise ValueError("each additional quantity must have [draw, chain] shape")
            parts.append(tensor[..., None])
        if parts:
            values = tf.concat((samples, *parts), axis=-1)
            names += tuple(quantities)
            from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
            rhat = rank_normalized_split_rhat_summary(values, rhat_max=rhat["rhat_threshold"])
    finite = bool(tf.reduce_all(tf.math.is_finite(values)))
    if not finite:
        return {"passed": False, "modern_rhat": rhat, "health_failures": ("nonfinite_monitored_quantity",),
                "precision": {"status": "precision_unavailable", "passed": False}, "stage": stage}
    ess = rank_normalized_bulk_tail_ess(tf.transpose(values, (1, 0, 2)))
    def safe_array(x):
        return tuple(float(v) if math.isfinite(float(v)) else None for v in tf.unstack(x))
    bulk_floor = getattr(policy, stage + "_bulk_ess_min")
    tail_floor = getattr(policy, stage + "_tail_ess_min")
    information = ((bulk_floor == 0. or bool(tf.reduce_all(tf.math.is_finite(ess["bulk"]) & (ess["bulk"] >= bulk_floor))))
                   and (tail_floor == 0. or bool(tf.reduce_all(tf.math.is_finite(ess["tail"]) & (ess["tail"] >= tail_floor)))))
    mean = mean_precision(values)
    precision = precision_report(values, names, policy.precision if stage == "retained" else None)
    # Preserve existing summary fields for consumers. The common decision and
    # its evidence below are authoritative; a callback cannot override them.
    return {**(rhat if extra is None else extra),
            "schema": "bayesfilter.hmc_posterior_assessment_result.v1", "stage": stage,
            "passed": bool(rhat["passed"] and information and precision["passed"]
                           and (extra is None or (extra["passed"] and not extra.get("hard_vetoes")))),
            "modern_rhat": rhat, "information_passed": information,
            "quantity_names": names, "bulk_ess": safe_array(ess["bulk"]), "tail_ess": safe_array(ess["tail"]),
            "mean_mcse": safe_array(mean["mcse"]), "mean_ess": safe_array(mean["mean_ess"]),
            "mcse_sd_ratio": safe_array(mean["mcse_sd_ratio"]),
            "mean_mcse_method": mean["estimator"], "precision": precision, "consumer_diagnostic": extra,
            "interpretation": "finite-window diagnostic evidence, not proof of global convergence"}


def validate_sequential_seeds(config, *, forbidden=()):
    """Check every actually executable chunk seed, roots and int32 range."""
    from bayesfilter.inference.neutra_hmc import _shared_sequential_chunk_seed
    used = set(tuple(s) for s in forbidden)
    roots = (tuple(config.warmup_seed), tuple(config.retained_seed))
    if roots[0] == roots[1]:
        raise ValueError("warmup and retained seeds must be distinct")
    if any(s in used for s in roots):
        raise ValueError("posterior seeds must be fresh relative to tuning")
    inventory = []
    for stage, root in zip(("warmup", "retained"), roots):
        if len(root) != 2 or any(type(v) is not int or not -(2**31) <= v < 2**31 for v in root):
            raise ValueError("posterior seeds must be int32 pairs")
        count = getattr(config, stage + "_max_results")
        chunk = getattr(config, stage + "_chunk_results")
        for i in range((count + chunk - 1) // chunk):
            seed = _shared_sequential_chunk_seed(root, i)
            if any(not -(2**31) <= v < 2**31 for v in seed):
                raise ValueError("derived posterior seed exceeds int32")
            if seed in used or seed in roots:
                raise ValueError("derived posterior seeds must be fresh and disjoint")
            used.add(seed)
            inventory.append((stage, i, seed))
    return tuple(inventory)


def run_hmc_posterior(*, member, config, **kwargs):
    """Assess an explicitly chosen, repository-verified exact HMC member."""
    from bayesfilter.inference.hmc_candidate_set_retained import HMCCandidateRetainedRunner
    if type(member) is not HMCCandidateRetainedRunner:
        raise TypeError("posterior execution requires a verified exact HMC member")
    result = dict(member.run_sequential(config=config, **kwargs))
    result["decision"] = ("POSTERIOR_DECLARED_CHECKS_PASSED" if result["passed"]
                          else "POSTERIOR_DECLARED_CHECKS_NOT_MET")
    result["assessment_role"] = "posterior_only"
    return result
