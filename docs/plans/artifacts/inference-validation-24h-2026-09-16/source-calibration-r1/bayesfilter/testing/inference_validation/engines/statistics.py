"""Independent diagnostic statistics, with explicit independent observation units.

NumPy/SciPy are reference/post-run tools here, never runtime inference decisions.
Monte Carlo p-values include the observed statistic via the plus-one correction.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import stats


def binomial_interval(successes, total, alpha=.05):
    if not 0 <= successes <= total or total < 1: raise ValueError("invalid binomial counts")
    # Exact Clopper-Pearson bounds, including all-fail/all-pass cases.
    return [0. if successes==0 else float(stats.beta.ppf(alpha/2,successes,total-successes+1)),
            1. if successes==total else float(stats.beta.ppf(1-alpha/2,successes+1,total-successes))]


def randomized_rank(truth, draws, rng):
    values=np.asarray(draws)
    if not np.isfinite(truth) or not np.all(np.isfinite(values)):
        raise ValueError("rank requires finite quantities")
    return int(np.sum(values<truth)+rng.integers(int(np.sum(values==truth))+1))


def rank_uniform_test(ranks, rank_draws, *, null_draws, seed, alpha, multiplicity=1):
    values=np.asarray(ranks)
    if values.ndim!=1 or not len(values) or np.any(values!=values.astype(int)) or np.any(values<0) or np.any(values>rank_draws):
        raise ValueError("invalid integer ranks")
    bins=rank_draws+1
    count=np.bincount(values.astype(int),minlength=bins)
    expected=len(values)/bins
    stat=float(np.sum((count-expected)**2/expected))
    rng=np.random.default_rng(seed)
    null=rng.multinomial(len(values),[1/bins]*bins,size=null_draws)
    null_stat=np.sum((null-expected)**2/expected,axis=1)
    p=(1+int(np.sum(null_stat>=stat)))/(null_draws+1)
    return {"kind":"rank_uniformity","counts":count.tolist(),"replications":len(values),
            "statistic":stat,"p_value":p,"threshold":alpha/multiplicity,
            "finding":"discrepancy_detected" if p<=alpha/multiplicity else "no_discrepancy_detected",
            "null":"iid discrete uniform ranks; exact multinomial Monte Carlo comparison",
            "accuracy_established":False}


def two_sample_test(left,right,*,seed,permutations,alpha,multiplicity=1):
    """Permutation KS on independent scalar observations; not adjacent MCMC draws."""
    x,y=np.asarray(left).ravel(),np.asarray(right).ravel()
    if min(len(x),len(y))<2 or not np.all(np.isfinite(np.r_[x,y])):
        raise ValueError("two independent finite samples required")
    pooled=np.r_[x,y]
    order=np.argsort(pooled,kind="stable")
    # Evaluate CDF differences only at the end of each tie group.
    ends=np.r_[pooled[order][1:]!=pooled[order][:-1], True]
    def statistic(labels):
        cumulative=np.cumsum(labels[order])
        difference=cumulative/len(x)-(np.arange(len(pooled))+1-cumulative)/len(y)
        return float(np.max(np.abs(difference[ends])))
    labels=np.r_[np.ones(len(x)),np.zeros(len(y))]
    observed=statistic(labels)
    rng=np.random.default_rng(seed)
    exceed=sum(statistic(rng.permutation(labels))>=observed for _ in range(permutations))
    p=(exceed+1)/(permutations+1)
    return {"kind":"independent_two_sample_ks","statistic":observed,"p_value":p,
            "threshold":alpha/multiplicity,"finding":"discrepancy_detected" if p<=alpha/multiplicity else "no_discrepancy_detected",
            "accuracy_established":False,"independence_requirement":"independent sample rows; no consecutive MCMC draws"}


def accuracy_assessment(draws, reference, *, tolerance, finite_variance=True, reference_iid=True):
    """Bounded-function posterior discrepancies and replicate-chain uncertainty.

    Values are diagnostic unless uncertainty is small enough relative to the
    declared tolerance. Between-chain SE from few chains is explicitly weak.
    """
    x=np.asarray(draws); ref=np.asarray(reference)
    if x.ndim!=3 or ref.ndim!=2 or x.shape[-1]!=ref.shape[-1]: raise ValueError("draw/chain/parameter and reference-row/parameter shapes required")
    if x.shape[1]<1 or x.shape[-1]<1 or len(ref)<2: raise ValueError("nonempty chains and quantities and at least two reference rows required")
    if x.shape[0]<2: return {"finding":"unavailable","reason":"no retained posterior draws"}
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(ref)): return {"finding":"invalid","reason":"nonfinite draws"}
    rows=[]
    for j in range(x.shape[-1]):
        # Bounded transform has finite moments even for Cauchy.
        per_chain=np.arctan(x[:,:,j]).mean(0)
        r=np.arctan(ref[:,j])
        error=float(per_chain.mean()-r.mean())
        se=math.sqrt(float(per_chain.var(ddof=1)/len(per_chain)+r.var(ddof=1)/len(r))) if reference_iid and len(per_chain)>1 else None
        rows.append({"quantity":f"atan_parameter_{j}","error":error,"standard_error":se,
            "tolerance":tolerance,"descriptive_within_tolerance":abs(error)<=tolerance,
            "uncertainty_method":("between independent chain estimates plus iid reference variance; few-chain diagnostic"
                                  if reference_iid else "reference dependence not quantified; combined SE unavailable")})
        if finite_variance:
            rows.append({"quantity":f"mean_{j}","error":float(x[:,:,j].mean()-ref[:,j].mean()),
                         "role":"descriptive","reference_mean_se":float(ref[:,j].std(ddof=1)/math.sqrt(len(ref))) if reference_iid else None})
        rows.append({"quantity":f"median_{j}","error":float(np.median(x[:,:,j])-np.median(ref[:,j])),"role":"descriptive"})
    return {"finding":"reference_discrepancy" if any(r.get("descriptive_within_tolerance") is False for r in rows) else "within_descriptive_tolerance",
            "quantities":rows,"accuracy_established":False,"reference_iid":reference_iid,
            "limitation":"short correlated runs and few chains do not establish a fixed-error bound or convergence"}
