"""Diagnostic stopped/fixed comparisons against independently known quantities."""
from __future__ import annotations

import math
import numpy as np
from scipy import stats

from ..references import analytic
from .statistics import binomial_interval


def arm_quantities(draws, spec, params, data, *, jit_compile, method="lugsail",
                   batch_size=None, min_batches=20, lugsail_r=3, lugsail_c=.5):
    if not isinstance(method, str) or method not in {"lugsail", "autocorrelation", "batch_means"}:
        raise ValueError("unknown diagnostic mean estimator")
    from bayesfilter.inference.hmc_precision import mean_precision, quantile_precision
    import tensorflow as tf

    estimator_options = dict(method=method, jit_compile=jit_compile, batch_size=batch_size,
                             min_batches=min_batches, lugsail_r=lugsail_r, lugsail_c=lugsail_c)
    values = np.asarray(draws)
    references = analytic.exact_functionals(spec.target_id, params, data)
    rows = {}
    for (kind, index), truth in references.items():
        name = spec.parameters[index] + ":" + kind
        sample = values[..., index:index+1]
        available = len(sample) >= 4 and bool(np.all(np.isfinite(sample)))
        estimate = float(sample.mean() if kind == "mean" else np.median(sample)) if available else None
        report = (mean_precision(tf.constant(sample, tf.float64), **estimator_options)
                  if kind == "mean" else quantile_precision(tf.constant(sample, tf.float64), .5)) if available else {}
        se = float(report["mcse"][0]) if report else None
        available = available and se is not None and math.isfinite(se) and se > 0
        rows[name] = {"reference": truth, "estimate": estimate,
                      "method": method if kind == "mean" else "quantile",
                      "estimator": report.get("estimator"),
                      "error": None if estimate is None else estimate-truth,
                      "mcse": se if available else None, "available": available,
                      "covered": available and abs(estimate-truth) <= stats.norm.ppf(.975)*se}
    if spec.target_id == "mixture":
        # Mixture CDF at zero: w Phi(a) + (1-w) Phi(-a).
        a, w = params.get("separation", 5.), params.get("weight", .3)
        truth = float(w*stats.norm.cdf(a) + (1-w)*stats.norm.cdf(-a))
        sample = (values[..., :1] < 0).astype(float)
        estimate = float(sample.mean()) if len(sample) else None
        report = mean_precision(tf.constant(sample,tf.float64),**estimator_options) if len(sample)>=4 else {}
        se = float(report["mcse"][0]) if report else None
        valid = se is not None and math.isfinite(se) and se > 0
        rows["left_mode_probability"] = {"reference": truth, "estimate": estimate,
            "method": method, "estimator": report.get("estimator"),
            "error": None if estimate is None else estimate-truth, "mcse": se if valid else None,
            "available": valid, "covered": valid and abs(estimate-truth)<=stats.norm.ppf(.975)*se}
    return rows


def summarize_pairs(records, planned, *, declared_names=()):
    """One preselected member per independent replication; missing arms stay visible."""
    names = sorted(set(declared_names) | {key for pair in records for arm in ("stopped", "fixed")
                                       for key in pair.get(arm, {})})
    quantities = {}
    for name in names:
        arms = {}
        for arm in ("stopped", "fixed"):
            rows = [p.get(arm, {}).get(name, {}) for p in records]
            available = sum(bool(r.get("available")) for r in rows)
            covered = sum(bool(r.get("covered")) for r in rows)
            errors = [r["error"] for r in rows if r.get("error") is not None]
            arms[arm] = {"available":available,"covered":covered,"planned":planned,
                         "unavailable":planned-available,"coverage_interval":binomial_interval(covered,planned),
                         "conditional_coverage_interval":binomial_interval(covered,available) if available else None,
                         "coverage_denominator":"all planned fits; unavailable intervals do not count as covered",
                         "mean_absolute_error":float(np.mean(np.abs(errors))) if errors else None}
        differences = [abs(p["stopped"][name]["error"])-abs(p["fixed"][name]["error"])
                       for p in records if name in p.get("stopped",{}) and name in p.get("fixed",{})
                       and p["stopped"][name].get("error") is not None and p["fixed"][name].get("error") is not None]
        mean = float(np.mean(differences)) if differences else None
        se = float(np.std(differences,ddof=1)/math.sqrt(len(differences))) if len(differences)>1 else None
        radius = stats.t.ppf(.975,len(differences)-1)*se if se is not None else None
        quantities[name] = {"arms":arms,"paired_replications":len(differences),
                           "unpaired_or_missing":planned-len(differences),
                           "mean_absolute_error_difference":mean,
                           "paired_t_interval":None if radius is None else [mean-radius,mean+radius]}
    return {"quantities":quantities,"planned":planned,"recorded_pairs":len(records),
            "independent_unit":"complete tuning replication, same member, independent arm streams",
            "interval_interpretation":"pointwise diagnostic intervals; finite-moment t approximation, no multiplicity-adjusted ranking",
            "sequential_coverage_established":False,"ranking_supported":False}
