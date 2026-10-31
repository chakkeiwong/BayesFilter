"""Null rejection and ignored-data detection for the diagnostic rank engine.

This independently simulated normal model calibrates the statistical engine;
it does not substitute for measured power against the public HMC procedure.
"""
from __future__ import annotations
import time
import numpy as np
from scipy import stats

from ..designs import seed_for
from ..storage import write_json
from .statistics import rank_uniform_test,binomial_interval,two_sample_test


def run(design,root,deadline=None):
    if "calibration_design" in design.options:
        return calibrate_experiment(design,root,deadline)
    n,m=design.draws,design.rank_draws
    trials=[]
    for trial in range(design.replications):
        if deadline and time.monotonic()>=deadline: break
        rng=np.random.default_rng(seed_for(design.seed,design.design_id,trial,"data"))
        k=int(design.options.get("normal_data_count",1))
        truth=rng.normal(scale=2.,size=n); y=truth[:,None]+rng.normal(size=(n,k))
        # Independent completion-of-square oracle, tau=2 and sigma=1.
        variance=1/(.25+k); mean=variance*y.sum(1); sd=np.sqrt(variance)
        draws=mean[:,None]+sd*rng.normal(size=(n,m))
        prior=2*rng.normal(size=(n,m))
        arm_results={}
        severity=design.options.get("location_severity",0.)
        arms=[("correct",draws),("ignored_data",prior)]
        if severity:
            arms.append(("location_defect",draws+severity*sd))
        for arm,values in arms:
            parameter=(values<truth[:,None]).sum(1)
            likelihood=(stats.norm.logpdf(y[:,None,:]-values[:,:,None]).sum(-1)<
                        stats.norm.logpdf(y-truth[:,None]).sum(-1)[:,None]).sum(1)
            quantities=[("parameter",parameter),("log_likelihood",likelihood)]
            if design.options.get("include_radius",False):
                quantities.append(("bounded_radius",(values**2<truth[:,None]**2).sum(1)))
            tests={name:rank_uniform_test(r,m,null_draws=design.null_draws,
                seed=seed_for(design.seed,design.design_id,trial,arm,name),alpha=design.alpha,multiplicity=len(quantities))
                for name,r in quantities}
            arm_results[arm]={"detected":any(t["finding"]=="discrepancy_detected" for t in tests.values()),"tests":tests}
        # Separate independent sample experiment calibrates the KS primitive.
        left=rng.normal(size=n); right=rng.normal(size=n)
        for arm,shift in ((("ks_null",0.),("ks_location_defect",1.)) if design.options.get("include_ks",True) else ()):
            test=two_sample_test(left,right+shift,seed=seed_for(design.seed,design.design_id,trial,arm),
                permutations=design.null_draws,alpha=design.alpha)
            arm_results[arm]={"detected":test["finding"]=="discrepancy_detected","tests":{"location":test}}
        trials.append(arm_results)
    count=len(trials)
    rates={}
    for arm in ("correct","ignored_data",*(("ks_null","ks_location_defect") if design.options.get("include_ks",True) else ()),*(
            ("location_defect",) if design.options.get("location_severity",0.) else ())):
        detected=sum(t[arm]["detected"] for t in trials)
        rates[arm]={"count":detected,"total":count,"rate":detected/count if count else None,
                    "interval":binomial_interval(detected,count) if count else None}
    result={"trials":trials,"rates":rates,"finding":"power_estimated" if count==design.replications else "incomplete",
        "independent_unit":"independent complete rank-test experiment",
        "tested_implementation":"rank statistical engine with independent analytic sampler controls",
        "mutation_activation":"posterior draws replaced by independent N(0,4) prior draws",
        "location_severity_posterior_sd":design.options.get("location_severity",0.),
        "hmc_mutation_power_established":False,"within_trial_multiplicity":3 if design.options.get("include_radius",False) else 2,
        "normal_data_count":int(design.options.get("normal_data_count",1)),
        "allocation":"normal conjugate tau=2 sigma=1; analytic error sensitivity, not proof of public sampler power"}
    write_json(root/"power.json",result)
    return result


def calibrate_experiment(design,root,deadline):
    """Repeat a declared engine, measuring valid executions and defect detection.

    This uses the same experiment engine and public procedure. It does not
    replace sampler observations with independently fabricated rank defects.
    """
    from dataclasses import replace
    from ..designs import ValidationDesign
    from ..storage import read_json
    from . import mechanics,invariance,pipeline,sbc,diagnostics

    base=ValidationDesign.from_payload(design.options["calibration_design"])
    arms=tuple(design.options.get("power_controls", ("baseline", "wrong_energy")))
    if "baseline" not in arms or len(set(arms))!=len(arms):
        raise ValueError("distinct power arms including baseline required")
    module={"mechanics":mechanics,"invariance":invariance,"search":pipeline,
            "accuracy":pipeline,"sbc":sbc,"stopping":diagnostics if base.scenario.route=="reference" else pipeline}[base.engine]
    trials=[]
    for trial in range(design.replications):
        path=root/f"power-trial-{trial:04d}.json"
        if path.exists():
            trials.append(read_json(path)); continue
        rows={}
        for arm in arms:
            if deadline and time.monotonic()>=deadline: break
            child=replace(base,design_id=f"trial-{trial}-{arm}",
                scenario=replace(base.scenario,control=arm),seed=seed_for(design.seed,trial,"experiment")[0])
            directory=root/f"trial-{trial:04d}"/arm
            directory.mkdir(parents=True,exist_ok=True)
            try:
                result=module.run(child,directory,deadline)
                finding=result["finding"]
                valid=(finding not in {"incomplete","calibration_incomplete","no_verified_members",
                                       "posterior_incomplete","unavailable","invalid"}
                       and result.get("assessment_complete",True))
                rows[arm]={"valid":valid,"detected":valid and finding in {"mechanics_discrepancy",
                    "inventory_discrepancy","pipeline_discrepancy","discrepancy_detected","diagnostic_discrepancy"},
                    "finding":finding,"observations":str(directory),"design":child.payload()}
            except Exception as exc:
                rows[arm]={"valid":False,"detected":False,"failure":str(exc),"exception":type(exc).__name__}
        write_json(path,rows); trials.append(rows)
        if len(rows)!=len(arms): break
    rates={}
    for arm in arms:
        valid=sum(t.get(arm,{}).get("valid",False) for t in trials)
        detected=sum(t.get(arm,{}).get("detected",False) for t in trials)
        rates[arm]={"valid":valid,"detected":detected,"planned":design.replications,
            "detection_fraction_of_planned":detected/design.replications,
            "interval":binomial_interval(detected,valid) if valid else None,
            "conditional_on_valid_execution":valid!=design.replications}
    result={"rates":rates,"trials":trials,"tested_engine":base.engine,
        "tested_procedure":base.scenario.route,"finding":"power_estimated" if all(
            row["valid"]==design.replications for row in rates.values()) else "incomplete",
        "independent_unit":"entire repeated experiment; no candidate siblings counted as replications",
        "ranking_supported":False,"power_precision":"read binomial intervals; small replications are coarse development evidence"}
    write_json(root/"power.json",result)
    return result
