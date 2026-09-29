"""Independent finite-size stationary/transient controls for runtime diagnostics."""
from __future__ import annotations
import math
import time
import numpy as np
from scipy import stats
import tensorflow as tf

from ..designs import seed_for
from ..storage import write_json
from .statistics import binomial_interval


def reference_rhat(x):
    half=len(x)//2
    split=np.concatenate([x[:half],x[-half:]],axis=1)
    def component(v):
        rank=stats.rankdata(v,method="average").reshape(v.shape)
        z=stats.norm.ppf((rank-.375)/(v.size+.25))
        w=z.var(axis=0,ddof=1).mean()
        if w==0: return float("inf")
        return math.sqrt((half-1)/half+z.mean(0).var(ddof=1)/w)
    return max(component(split),component(abs(split-np.median(split))))


def run(design,root,deadline=None):
    from bayesfilter.inference.hmc_precision import mean_precision
    from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
    count=design.draws
    if count<64: raise ValueError("diagnostic arrays need at least 64 draws")
    rho=float(design.options.get("rho",.8))
    if not abs(rho)<1: raise ValueError("stationary AR(1) requires |rho|<1")
    kind=design.options.get("array_regime","stationary")
    if kind not in {"stationary","transient","constant","missed_mode"}: raise ValueError("unknown array regime")
    rows=[]
    for rep in range(design.replications):
        if deadline and time.monotonic()>=deadline: break
        rng=np.random.default_rng(seed_for(design.seed,design.design_id,rep,"arrays"))
        x=np.empty((count,4)); x[0]=rng.normal(size=4)
        for j in range(1,count): x[j]=rho*x[j-1]+math.sqrt(1-rho*rho)*rng.normal(size=4)
        if kind=="transient": x+=8*np.exp(-np.arange(count)[:,None]/(count/3))
        if kind=="constant": x[:]=1.
        if kind=="missed_mode": x-=5.
        values=tf.constant(x[...,None],tf.float64)
        expected=reference_rhat(x)
        actual=rank_normalized_split_rhat_summary(values)["rhat"][0]
        methods={}
        for method in ("autocorrelation","batch_means","lugsail"):
            result=mean_precision(values,method=method,jit_compile=design.device=="gpu")
            mcse=float(result["mcse"][0])
            valid=math.isfinite(mcse) and mcse>0
            methods[method]={"mcse":mcse if valid else None,"valid":valid,
                             "covered_zero":valid and abs(x.mean())<=stats.norm.ppf(.975)*mcse}
        rows.append({"rhat":actual,"reference_rhat":expected,"arithmetic_error":abs(actual-expected) if actual is not None and math.isfinite(actual) and math.isfinite(expected) else None,
            "availability_matches": (actual is not None) == math.isfinite(expected),
            "estimate":float(x.mean()),"methods":methods})
    # Covariance sum gives exact finite-size variance of the stationary sample mean.
    exact_var=(count+2*sum((count-k)*rho**k for k in range(1,count)))/(4*count*count)
    coverage={}
    for method in ("autocorrelation","batch_means","lugsail"):
        valid=sum(r["methods"][method]["valid"] for r in rows)
        covered=sum(r["methods"][method]["covered_zero"] for r in rows)
        coverage[method]={"available":valid,"total":len(rows),"covered":covered,
            "coverage_interval":binomial_interval(covered,len(rows)) if rows else None}
    complete=len(rows)==design.replications
    arithmetic_passed=all(r["availability_matches"] and (r["arithmetic_error"] is None or r["arithmetic_error"]<1e-12) for r in rows)
    result={"rows":rows,"coverage":coverage,"regime":kind,"rho":rho,
        "planned":design.replications,"completed":len(rows),"assessment_complete":complete,
        "exact_stationary_mean_se":math.sqrt(exact_var),
        "finding":("incomplete" if not complete else "diagnostics_assessed" if arithmetic_passed else "diagnostic_discrepancy"),
        "rhat_arithmetic_tolerance":1e-12,
        "comparison_target":"stationary zero mean" if kind=="stationary" else "specified nonstationary/stress control",
        "actual_controller_stopping_test":False,"ranking_supported":False,
        "nominal_interval":"fixed-size normal 95%; unavailable counts as noncoverage"}
    write_json(root/"diagnostics.json",result)
    return result
