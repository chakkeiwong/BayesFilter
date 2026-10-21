#!/usr/bin/env python3
"""Independent CPU reference calibration of existing posterior MCSE policies.

Diagnostic NumPy/SciPy generate exact margins; TensorFlow evaluates the actual
posterior rules. This executable supplies neither sampler nor default authority.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

PLAN = "docs/plans/bayesfilter-mcse-stopping-calibration-2026-10-12.md"
METHODS = ("lugsail", "batch_means", "even_lugsail", "autocorrelation")
REPAIR_METHODS = ("lugsail", "lugsail_max_batch_means", "batch_means", "autocorrelation")
REPAIR_PLAN = "docs/plans/bayesfilter-modern-hmc-comparator-precision-continuation-2026-10-13.md"
RHOS = (0., .9, -.9)
NAMES = ("coordinate", "square", "abs_gt_2")
LOOKS = tuple(range(1000, 10001, 1000))


def write(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def finite_mean_variance(n, rho, chains=4):
    """Stationary variance-one Gaussian AR(1), deterministic sample count."""
    if type(n) is not int or n<1 or type(chains) is not int or chains<1 or not -1<rho<1:
        raise ValueError("invalid reference design")
    return (1+2*math.fsum((1-k/n)*rho**k for k in range(1,n)))/(n*chains)


def interval(count, total, alpha=.05):
    from scipy.stats import beta
    if not 0 <= count <= total or not 0. < alpha < 1.:
        raise ValueError("invalid binomial counts")
    if total == 0:
        return None
    return [0. if count==0 else float(beta.ppf(alpha/2,count,total-count+1)),
            1. if count==total else float(beta.ppf(1-alpha/2,count+1,total-count))]


def summarize(records, planned, methods=METHODS):
    """Preserve nonstoppers and invalid intervals in every denominator."""
    if len(records)>planned:
        raise ValueError("completed replications exceed planned denominator")
    result=dict(planned=planned, completed=len(records), unstarted=planned-len(records), methods={})
    for method in methods:
        rows=[r["methods"][method] for r in records]
        entry=dict(stopped=sum(r["stopped"] is not None for r in rows), endpoints={})
        for endpoint in ("fixed", "stopped"):
            present=[r[endpoint] for r in rows if r[endpoint] is not None]
            metrics=[]
            for index,name in enumerate(NAMES):
                valid=[r for r in present if r["valid"][index]]
                misses=sum(r["interval_miss"][index] for r in valid)
                metrics.append(dict(name=name, completed_replications=len(rows),
                    output_available=len(present), valid_intervals=len(valid),
                    missing_intervals=len(present)-len(valid), interval_misses=misses,
                    valid_interval_frequency=interval(len(valid),len(rows)),
                    miss_rate_among_valid_interval=interval(misses,len(valid)),
                    empirical_mean_squared_error=(sum(r["error"][index]**2 for r in present)/len(present)
                        if present else None),
                    mean_estimated_variance=(sum(r["mcse"][index]**2 for r in valid)/len(valid)
                        if valid else None)))
            joint=[r for r in present if all(r["valid"])]
            entry["endpoints"][endpoint]=dict(quantities=metrics,
                valid_joint_intervals=len(joint), joint_misses=sum(any(r["interval_miss"]) for r in joint),
                all_output_lengths=[r["draws"] for r in present])
        entry["stop_frequency_interval"]=interval(entry["stopped"],len(rows))
        result["methods"][method]=entry
    return result


def paired_mean(values):
    """Pointwise Monte Carlo uncertainty, not a simultaneous coverage claim."""
    from scipy.stats import t
    count = len(values)
    mean = math.fsum(values)/count if count else None
    halfwidth = None
    if count > 1:
        variance = math.fsum((v-mean)**2 for v in values)/(count-1)
        halfwidth = float(t.ppf(.975, count-1))*math.sqrt(variance/count)
    return dict(pairs=count, mean=mean,
                approximate_95_percent_interval=None if halfwidth is None
                else [mean-halfwidth, mean+halfwidth])


def paired_event_difference(values):
    """Conservative paired risk difference via two discordance probabilities.

    Each marginal Clopper--Pearson interval has coverage at least 97.5%.
    A union bound gives at least 95% for their simultaneous rectangle, so
    subtracting its endpoints covers p(+1)-p(-1), including all-tied samples.
    This is pointwise across independent replications, not across report cells.
    """
    if any(v not in (-1, 0, 1) for v in values):
        raise ValueError("paired events require -1, 0 or 1 differences")
    count = len(values)
    plus, minus = values.count(1), values.count(-1)
    positive = interval(plus, count, alpha=.025)
    negative = interval(minus, count, alpha=.025)
    return dict(pairs=count, maximum_only_miss=plus, original_only_miss=minus,
        mean=(plus-minus)/count if count else None,
        conservative_95_percent_interval=None if not count else
        [positive[0]-negative[1], positive[1]-negative[0]],
        uncertainty="Bonferroni Clopper-Pearson for paired discordance probabilities")


def summarize_pairs(records, *, law=None):
    """Compare common paths, keeping unmatched endpoints explicit."""
    result = dict(completed=len(records), direction="maximum_minus_original",
        uncertainty="continuous: paired Student-t; binary: exact discordance bounds; pointwise across cells",
        endpoints={})
    for endpoint in ("fixed", "stopped"):
        pairs = [(r["methods"]["lugsail"][endpoint], r["methods"]["lugsail_max_batch_means"][endpoint])
                 for r in records]
        present = [(a,b) for a,b in pairs if a is not None and b is not None]
        quantities = []
        for index, name in enumerate(NAMES):
            valid = [(a,b) for a,b in present if a["valid"][index] and b["valid"][index]]
            quantities.append(dict(name=name, common_valid_pairs=len(valid),
                interval_miss_change=paired_event_difference([int(b["interval_miss"][index])-int(a["interval_miss"][index])
                                                  for a,b in valid]),
                estimated_variance_change=paired_mean([b["mcse"][index]**2-a["mcse"][index]**2 for a,b in valid]),
                original_only_valid=sum(a["valid"][index] and not b["valid"][index] for a,b in present),
                maximum_only_valid=sum(b["valid"][index] and not a["valid"][index] for a,b in present)))
        result["endpoints"][endpoint] = dict(common_outputs=len(present),
            original_only_output=sum(a is not None and b is None for a,b in pairs),
            maximum_only_output=sum(b is not None and a is None for a,b in pairs),
            neither_output=sum(a is None and b is None for a,b in pairs),
            draw_count_change=paired_mean([b["draws"]-a["draws"] for a,b in present]), quantities=quantities)
    result["conditioning"] = "common valid outputs at each endpoint; not all planned paths"
    if law == "student_t5":
        # t(5) has finite fourth but not eighth moment. Mean precision for
        # X^2 is meaningful, but a variance estimate quartic in X need not
        # have finite variance across replications. Do not imply a t interval
        # for that statistic is calibrated merely because scipy computed one.
        for endpoint in result["endpoints"].values():
            square = next(q for q in endpoint["quantities"] if q["name"] == "square")
            delta = square["estimated_variance_change"]
            delta["approximate_95_percent_interval"] = None
            delta["uncertainty_status"] = "not_supported: t5 lacks eighth moment; quartic statistic variance not established"
            delta["mean_role"] = "descriptive_only"
    return result


def carry_completed_replicates(report, previous):
    """Reuse every completed record only under the identical statistical design."""
    if previous["status"] != "budget_exhausted":
        raise ValueError("continuation requires a terminal budget-exhausted diagnostic")
    if previous.get("profile", "historical") != report.get("profile", "historical"):
        raise ValueError("continuation changed calibration profile")
    if report.get("profile") == "batch_maximum_repair" and previous["driver_sha256"] != report["driver_sha256"]:
        raise ValueError("continuation changed repair driver")
    for key in ("seed", "replications", "methods", "looks", "chains", "source_sha256", "source",
                "versions", "stationary_initialization", "gpu_intentionally_hidden", "use_xla"):
        # JSON turns the original tuples into lists; compare their serialized form.
        if json.dumps(previous[key], sort_keys=True) != json.dumps(report[key], sort_keys=True):
            raise ValueError("continuation changed " + key)
    old_cells = previous["cells"]
    if [(c["law"], c["rho"], c["planned"]) for c in old_cells] != [
            (c["law"], c["rho"], c["planned"]) for c in report["cells"]]:
        raise ValueError("continuation changed condition inventory")
    prior_seconds = previous.get("cumulative_elapsed_seconds", previous["elapsed_seconds"])
    if not math.isfinite(prior_seconds) or prior_seconds < 0:
        raise ValueError("invalid prior compute charge")
    carried = []
    for index, (current, old) in enumerate(zip(report["cells"], old_cells)):
        records = old["records"]
        if len(records) > report["replications"]:
            raise ValueError("completed records exceed planned denominator")
        for replication, record in enumerate(records):
            if (record["replication"] != replication
                    or record["seed"] != [report["seed"], index//2, replication]
                    or set(record["methods"]) != set(report["methods"])
                    or any(r["fixed"] is None or r["fixed"]["draws"] != LOOKS[-1]
                           for r in record["methods"].values())):
                raise ValueError("completed records must be a unique complete prefix with original seeds")
        current["records"] = list(records)
        current["summary"] = summarize(records, report["replications"], report["methods"])
        carried.append(len(records))
    report.update(carried_completed_records=carried, prior_elapsed_seconds=prior_seconds,
                  attempt_count=previous.get("attempt_count", 1)+1)


def assess(values, method, rhat, *, truth, gaussian_rho=None):
    from bayesfilter.inference.hmc_precision import HMCPrecisionPolicy, HMCPrecisionTarget
    from bayesfilter.inference.hmc_posterior_assessment import HMCPosteriorAssessmentPolicy, assess_posterior
    n=int(values.shape[0])
    precision=HMCPrecisionPolicy(tuple(HMCPrecisionTarget(name,mcse_sd_ratio_max=.02) for name in NAMES),
        method="lugsail" if method=="even_lugsail" else method,
        batch_size=2*(math.isqrt(n)//2) if method=="even_lugsail" else None,
        min_batches=20,lugsail_r=3,lugsail_c=.5,jit_compile=True)
    policy=HMCPosteriorAssessmentPolicy(retained_bulk_ess_min=400.,retained_tail_ess_min=400.,
        precision=precision,binary_quantity_names=("abs_gt_2",))
    assessment=assess_posterior(values,NAMES,policy=policy,stage="retained",rhat=rhat)
    rows=assessment["precision"]["targets"]
    error=[r["estimate"]-value for r,value in zip(rows,truth)]
    valid=[r["valid"] for r in rows]
    mcse=[r["mcse"] for r in rows]
    z=1.959963984540054
    exact=None if gaussian_rho is None else finite_mean_variance(n,gaussian_rho)
    report = dict(draws=n, passed=assessment["passed"], failed_checks=assessment["failed_checks"],
        estimate=[r["estimate"] for r in rows], error=error, mcse=mcse, valid=valid,
        interval_miss=[bool(abs(e)>z*s) if ok else None for e,s,ok in zip(error,mcse,valid)],
        fixed_length_gaussian_coordinate_variance=exact,
        coordinate_variance_ratio=(mcse[0]**2/exact if exact is not None and valid[0] else None),
        gaussian_comparator_role="fixed_length_only_not_conditional_stopped_variance",
        precision_policy=precision.payload())
    if method == "lugsail_max_batch_means":
        report["batch_metadata"] = assessment["precision"]["batch_metadata"]
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--replications",type=int,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--max-wall-seconds",type=float,required=True)
    parser.add_argument("--resume-from",type=Path,
                        help="Preserve completed records from a terminal bounded attempt.")
    parser.add_argument("--profile", choices=("historical", "batch_maximum_repair"), default="historical")
    args=parser.parse_args()
    methods = REPAIR_METHODS if args.profile == "batch_maximum_repair" else METHODS
    if (os.environ.get("CUDA_VISIBLE_DEVICES")!="-1"
            or os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH","").lower()!="true"):
        raise ValueError("hide GPUs and set memory growth before framework import")
    if not 1<=args.replications<=128 or args.seed<0 or not 0<args.max_wall_seconds<=1800:
        raise ValueError("outside declared reference allocation")
    args.source_root=args.source_root.resolve()
    sys.path.insert(0,str(args.source_root))
    started=time.monotonic()
    import numpy as np  # independent reference generation only
    import scipy
    from scipy import signal, stats
    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
    from bayesfilter.testing.inference_validation.modern_bridge_audit import source_dependencies
    if not Path(sys.modules[rank_normalized_split_rhat_summary.__module__].__file__).resolve().is_relative_to(args.source_root):
        raise ValueError("wrong numerical source")
    paths=source_dependencies([args.source_root/"bayesfilter/inference/hmc_posterior_assessment.py",
        args.source_root/"bayesfilter/inference/hmc_convergence.py"],repository_root=args.source_root)
    hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in paths}
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status="running",plan=REPAIR_PLAN if args.profile == "batch_maximum_repair" else PLAN,
        profile=args.profile,command=[sys.executable,*sys.argv],environment=sys.executable,
        started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashes,
        source=json.loads((args.source_root/".validation-source.json").read_text()),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions=dict(tensorflow=tf.__version__,tfp=tfp.__version__,numpy=np.__version__,scipy=scipy.__version__),
        gpu_intentionally_hidden=True,use_xla=True,seed=args.seed,replications=args.replications,
        looks=LOOKS,methods=methods,chains=4,max_wall_seconds=args.max_wall_seconds,
        numerical_data="exact synthetic stationary reference laws",
        stationary_initialization=True,warmup_evaluated=False,actual_hmc_transitions=False,
        prior_elapsed_seconds=0.,attempt_count=1,
        posterior_default_promotion=False,nominal_coverage_established=False,method_ranking=None,
        cpu_threads={k:os.environ.get(k) for k in ("TF_NUM_INTEROP_THREADS","TF_NUM_INTRAOP_THREADS","OMP_NUM_THREADS")},
        cells=[dict(law=law,rho=rho,records=[],planned=args.replications)
               for rho in RHOS for law in ("gaussian","student_t5")])
    if args.resume_from is not None:
        parent = args.resume_from.resolve()/"result.json"
        carry_completed_replicates(report, json.loads(parent.read_text()))
        report["resume_from"] = dict(result_file=str(parent), result_sha256=hashlib.sha256(parent.read_bytes()).hexdigest())
    write(args.output/"manifest.json",{k:v for k,v in report.items() if k!="cells"})
    write(args.output/"result.json",report)
    try:
        for replication in range(min(len(c["records"]) for c in report["cells"]), args.replications):
            for rho_index,rho in enumerate(RHOS):
                if time.monotonic()-started>=args.max_wall_seconds:
                    break
                if all(len(c["records"]) > replication for c in report["cells"][2*rho_index:2*rho_index+2]):
                    continue
                seed=[args.seed,rho_index,replication]
                rng=np.random.default_rng(np.random.SeedSequence(seed))
                noise=rng.normal(size=(LOOKS[-1],4))
                noise[1:]*=math.sqrt(1-rho*rho)
                latent=signal.lfilter([1.],[1.,-rho],noise,axis=0)
                for law_index,law in enumerate(("gaussian","student_t5")):
                    cell=report["cells"][2*rho_index+law_index]
                    if len(cell["records"]) > replication:
                        continue
                    x=latent if law=="gaussian" else stats.t.ppf(stats.norm.cdf(latent),5)
                    if not np.isfinite(x).all():
                        raise ValueError("nonfinite exact marginal transform")
                    values=tf.stack((x,x*x,abs(x)>2.),axis=-1)
                    truth=[0.,1. if law=="gaussian" else 5/3,
                        float(2*(stats.norm.sf(2) if law=="gaussian" else stats.t.sf(2,5)))]
                    record=dict(replication=replication,seed=seed,truth=truth,
                        methods={method:dict(stopped=None,fixed=None) for method in methods})
                    for n in LOOKS:
                        if time.monotonic()-started>=args.max_wall_seconds:
                            break
                        active=[m for m in methods if record["methods"][m]["stopped"] is None or n==LOOKS[-1]]
                        if not active:
                            continue
                        rhat=rank_normalized_split_rhat_summary(values[:n],rhat_max=1.01)
                        for method in active:
                            assessment=assess(values[:n],method,rhat,truth=truth,
                                gaussian_rho=rho if law=="gaussian" else None)
                            if record["methods"][method]["stopped"] is None and assessment["passed"]:
                                record["methods"][method]["stopped"]=assessment
                            if n==LOOKS[-1]:
                                record["methods"][method]["fixed"]=assessment
                    if all(r["fixed"] is not None for r in record["methods"].values()):
                        cell["records"].append(record)
                        cell["summary"]=summarize(cell["records"],args.replications,methods)
                        if args.profile == "batch_maximum_repair":
                            cell["paired_summary"] = summarize_pairs(cell["records"], law=cell["law"])
                    else:
                        cell["interrupted_replicate"]=record
                    report["elapsed_seconds"]=time.monotonic()-started
                    write(args.output/"result.json",report)
            if time.monotonic()-started>=args.max_wall_seconds:
                break
        report["status"]="completed" if all(len(c["records"])==args.replications for c in report["cells"]) else "budget_exhausted"
        for name,expected in hashes.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
                raise ValueError("numerical source changed: "+name)
        if args.resume_from is not None and hashlib.sha256(parent.read_bytes()).hexdigest() != report["resume_from"]["result_sha256"]:
            raise ValueError("parent diagnostic changed during continuation")
    except BaseException as exc:
        report.update(status="repair_required",error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        for cell in report["cells"]:
            cell["summary"]=summarize(cell["records"],args.replications,methods)
            if args.profile == "batch_maximum_repair":
                cell["paired_summary"] = summarize_pairs(cell["records"], law=cell["law"])
        report.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        report["cumulative_elapsed_seconds"] = report["prior_elapsed_seconds"] + report["elapsed_seconds"]
        write(args.output/"result.json",report)
        write(args.output/"manifest.json",{k:v for k,v in report.items() if k!="cells"})
    print(json.dumps(dict(status=report["status"],seconds=report["elapsed_seconds"],
        completed=[len(c["records"]) for c in report["cells"]])))


if __name__=="__main__":
    main()
