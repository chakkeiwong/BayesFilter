"""SBC using independent complete fits for each conditional posterior draw.

The fit adapter receives observed data only. One output per independent fit
avoids treating correlated within-fit draws as iid posterior ranks. The null
still requires that each reported output have the correct posterior law.
"""
from __future__ import annotations
import time
import gc
import numpy as np

from ..designs import seed_for
from ..procedures import execute_pipeline
from ..references import analytic
from ..storage import read_tensor,write_json,read_json
from .statistics import randomized_rank,rank_uniform_test,binomial_interval


def reference_fit(design,data,seed):
    used=None if design.scenario.control=="ignore_data" else data
    return analytic.draw(design.scenario.target,1,seed,design.scenario.parameters,used)[0]


def one_output(design,data,root,dataset,fit,deadline):
    if design.scenario.route=="reference":
        return reference_fit(design,data,seed_for(design.seed,design.design_id,dataset,fit,"fit")),{"route":"independent_iid_reference"}
    output=execute_pipeline(design,root,data=data,dataset_id=dataset,fit_id=fit,deadline=deadline)
    # Stable predeclared L group; choose by candidate identity, never diagnostics,
    # retained accuracy, or the generating parameter. Every sibling remains saved.
    selected=sorted((m for m in output["members"] if
        design.options.get("member_rule","declared_l_first")=="first_verified" or m.get("L")==design.member_l),
        key=lambda m:m["candidate_id"])
    if not selected: return None,{"reason":("no_verified_member" if design.options.get("member_rule")=="first_verified"
                                             else "no_member_in_declared_L_group"),"pipeline":output}
    member=selected[0]
    if member["status"]!="assessed": return None,{"reason":"member_unassessed","pipeline":output}
    draws=read_tensor(member["draws_path"]).numpy()
    if len(draws)==0: return None,{"reason":"no_retained_output","pipeline":output}
    # Always the first chain at the last retained transition; no success screening.
    raw=analytic.active_coordinates(design.scenario.target,draws[-1,0])
    return raw,{"member_id":member["candidate_id"],"fit_path":str(root/"pipeline.json"),
                "runtime_passed":member["posterior"]["passed"]}


def run(design,root,deadline=None):
    records=[]; target=design.scenario.target
    for dataset in range(design.replications):
        path=root/f"dataset-{dataset:04d}.json"
        if path.exists(): records.append(read_json(path)); continue
        if deadline and time.monotonic()>=deadline: break
        truth,data=analytic.simulate(target,seed_for(design.seed,design.design_id,dataset,"data"),design.scenario.parameters)
        raw_truth=analytic.active_coordinates(target,truth)
        fits=[]; outputs=[]
        for fit in range(design.rank_draws):
            if deadline and time.monotonic()>=deadline: break
            try:
                value,record=one_output(design,data,root/f"dataset-{dataset:04d}"/f"fit-{fit:04d}",dataset,fit,deadline)
            except Exception as exc:
                # Failure remains a missing fit. It is never counted as a
                # detected statistical defect or removed from the denominator.
                value,record=None,{"reason":"fit_execution_failed", "exception":type(exc).__name__,
                                   "message":str(exc)}
            fits.append(record)
            if value is not None: outputs.append(value)
            gc.collect()
        ranks={}
        if len(outputs)==design.rank_draws:
            truth_q=analytic.test_quantities(target,np.asarray(raw_truth),design.scenario.parameters,data)
            fit_q=analytic.test_quantities(target,np.asarray(outputs),design.scenario.parameters,data)
            rng=np.random.default_rng(seed_for(design.seed,design.design_id,dataset,"rank_ties"))
            ranks={key:randomized_rank(truth_q[key],values,rng) for key,values in fit_q.items()}
        fits.extend({"reason":"unfunded", "fit_id":i} for i in range(len(fits),design.rank_draws))
        record={"dataset_id":dataset,"truth_assessor_only":truth,"data":data,"fits":fits,"ranks":ranks,
                "fit_outputs_active":np.asarray(outputs).tolist(),
                "status":"complete" if ranks else "missing_fit","output_rule":"last_retained_transition_first_chain",
                "fit_independence":"fresh full fit randomness conditional on dataset"}
        write_json(path,record); records.append(record)
    result=summarize_datasets(design,records)
    write_json(root/"sbc.json",result)
    return result


def summarize_datasets(design,records):
    """The same predeclared test for a single worker or an independent shard group."""
    if len(records)>design.replications:
        raise ValueError("more SBC dataset records than planned")
    available=[r for r in records if r["status"]=="complete"]
    for row in available:
        if (len(row["fits"])!=design.rank_draws or len(row["fit_outputs_active"])!=design.rank_draws
                or not row["ranks"] or any(f.get("reason") for f in row["fits"])):
            raise ValueError("inconsistent complete SBC dataset")
    tests={}
    if available:
        family=max(design.multiplicity,len(available[0]["ranks"]))
        if 1/(design.null_draws+1)>design.alpha/family: raise ValueError("insufficient null resolution for SBC family")
        tests={key:rank_uniform_test([r["ranks"][key] for r in available],design.rank_draws,
            null_draws=design.null_draws,seed=seed_for(design.seed,design.design_id,key,"null"),
            alpha=design.alpha,multiplicity=family) for key in available[0]["ranks"]}
    complete=len(available)==design.replications
    result={"datasets":records,"tests":tests,"planned":design.replications,"completed":len(available),
            "unstarted_datasets":design.replications-len(records),
            "planned_fits":design.replications*design.rank_draws,
            "missing_fit_policy":"conditional ranks are descriptive only; incomplete SBC cannot establish unconditional calibration",
            "availability_interval":binomial_interval(len(available),design.replications),
            "finding":("discrepancy_detected" if any(t["finding"]=="discrepancy_detected" for t in tests.values()) else "no_discrepancy_detected") if complete else "calibration_incomplete",
            "conditional_tests_only":not complete,"accuracy_established":False,
            "tested_procedure":design.scenario.route,"independent_unit":"simulated dataset; independent complete fits within each dataset",
            "null_assumptions":"independent fits return correct conditional posterior outputs; randomized ties; fixed reporting rule",
            "small_run_limit":"few datasets only demonstrate execution; inspect power before a calibration claim"}
    return result
