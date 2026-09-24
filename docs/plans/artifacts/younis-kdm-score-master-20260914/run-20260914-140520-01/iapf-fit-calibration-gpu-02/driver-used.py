"""Bounded target-specific nonlinear iAPF fit-control calibration."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path.cwd()))


class BudgetStop(SystemExit):
    """Stop the coordinator before starting an unaffordable numerical row."""


class AttemptBudget:
    def __init__(self, prior, limit, save):
        self.prior = prior
        self.limit = limit
        self.save = save
        self.charges = []
        self.deferred = []

    @property
    def used(self):
        return self.prior + sum(x["row_attempts"] + x["recursive_fits"] for x in self.charges)

    def wrap(self, endpoint, stage):
        def bounded(row, context):
            max_fits = row["iapf"]["max_iterations"] - 1 if row["proposal"] == "iapf" else 0
            upper = 1 + max_fits
            if self.used + upper > self.limit:
                self.deferred.append({"stage": stage, "row": row["id"],
                    "required_upper_bound": upper, "remaining": self.limit-self.used,
                    "numerical_work_started": False})
                self.save()
                raise BudgetStop(f"cannot admit {stage}/{row['id']}: {upper} charges required, "
                                 f"{self.limit-self.used} remain")
            fits, basis = max_fits, "conservative_failed_endpoint_upper_bound"
            try:
                result = endpoint(row, context)
                measured = result["diagnostics"].get("work_accounting", {}).get("offline_fit_calls")
                if row["proposal"] == "iapf":
                    if not isinstance(measured, int) or not 0 <= measured <= max_fits:
                        raise ValueError("missing or invalid recursive-fit accounting")
                    fits = measured
                else:
                    fits = 0
                basis = "executed_work_accounting"
                return result
            finally:
                self.charges.append({"stage": stage, "row": row["id"],
                    "row_attempts": 1, "recursive_fits": fits, "basis": basis})
                self.save()
        return bounded


def comparator_study(study, base, dataset, particles):
    comparator=deepcopy(study)
    comparator["settings"]["particles"]=particles
    comparator["evidence_class"]="mechanics"
    adversaries=("ekf","ukf","bootstrap","local_linear")
    comparator["required_proposals"]=list(adversaries)
    comparator["rows"]=[{**base,"id":name,"proposal":name,"role":"mechanics","dataset":dataset}
                        for name in adversaries]
    return comparator


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--source-revision",required=True)
    parser.add_argument("--resume-from",type=Path,
        help="Preserve a failed first launch and reuse its completed weak studies")
    args=parser.parse_args();root=args.output.resolve()
    if root.exists():raise ValueError("fresh versioned output required")
    revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    if revision!=args.source_revision:raise ValueError("source revision mismatch")
    if subprocess.check_output(["git","status","--porcelain"],text=True):raise ValueError("dirty source snapshot")
    from bayesfilter.score_study.coordinator import execute,report,fingerprint,write_json,load_endpoint
    from bayesfilter.score_study.contracts import digest,validate_result
    from bayesfilter.score_study.registry import default_registry
    from bayesfilter.score_study.runtime import configure_runtime
    from bayesfilter.score_study.tuning import issue_selection,consume_selection
    registry=default_registry();root.mkdir(parents=True);started=time.monotonic();cpu_started=time.process_time()
    manifest={"schema":"nonlinear_iapf_fit_calibration_v1","status":"running","git_commit":revision,
        "source_root":str(Path.cwd()),"command":sys.argv,"interpreter":sys.executable,
        "plan":"docs/plans/younis-score-iapf-fit-calibration-2026-09-16.md",
        "driver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment":{k:os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES","TF_FORCE_GPU_ALLOW_GROWTH",
            "BAYESFILTER_PRELOAD_CUSTOM_OP","TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS","OMP_NUM_THREADS")},
        "studies":[],"selections":{},"evidence_class":"target_specific_calibration","statistically_supported_ranking":False,
        "default_ready":False,"runtime_comparison":"forbidden_concurrent_workloads",
        "score_semantics":"finite_program_fixed_fit_realized_N_and_labels"}
    numerical=[]
    prior_root=args.resume_from.resolve() if args.resume_from else None
    prior_count=0
    if prior_root:
        previous=json.loads((prior_root/"run-manifest.json").read_text())
        previous_charges=json.loads((prior_root/"attempt-accounting.json").read_text())
        prior_count=sum(x["row_attempts"]+x["recursive_fits"] for x in previous_charges)
        if (previous["git_commit"]!=revision or previous["status"]!="failed" or
                previous.get("resume_from") or prior_count!=32 or
                previous["charged_row_or_recursive_fit_attempts"]!=prior_count):
            raise ValueError("resume requires the preserved first 32-charge failed launch")
        manifest.update(resume_from=str(prior_root),prior_charged_attempts=prior_count,
            prior_manifest_sha256=hashlib.sha256((prior_root/"run-manifest.json").read_bytes()).hexdigest(),
            prior_wall_seconds=previous["wall_seconds"],campaign_launch_number=2)
    else:
        manifest.update(prior_charged_attempts=0,prior_wall_seconds=0.,campaign_launch_number=1)

    def save_budget():
        manifest.update(charged_row_or_recursive_fit_attempts=budget.used,
            this_launch_charged_attempts=budget.used-budget.prior,remaining_charged_attempts=64-budget.used,
            budget_deferred_rows=budget.deferred,wall_seconds=time.monotonic()-started)
        write_json(root/"attempt-accounting.json",budget.charges)
        write_json(root/"run-manifest.json",manifest)

    budget=AttemptBudget(prior_count,64,save_budget)
    (root/"driver-used.py").write_bytes(Path(__file__).read_bytes())
    save_budget()

    def collect(name,spec,state,directory,reused=False):
        if state["fingerprint"]!=fingerprint(spec,registry):
            raise ValueError("study/source fingerprint mismatch")
        rows=[]
        for row in spec["rows"]:
            saved=state["rows"].get(row["id"],{})
            if saved.get("execution_status")!="complete":continue
            result=json.loads((directory/saved["result_path"]).read_text())
            if digest(result)!=saved["result_digest"]:raise ValueError("result digest mismatch")
            validate_result(result,row,registry)
            runtime=result["runtime"];diag=result["diagnostics"]
            if not (runtime["device"]=="GPU" and runtime["tf32"] and runtime["jit_compile"] and
                    runtime["traces"]==1 and runtime["memory_policy"]["all_physical_devices_memory_growth"]):
                raise ValueError("GPU/XLA/trace/memory evidence mismatch")
            if row["proposal"]=="iapf" and (any(n!=1 for n in diag["fit_trace_counts"]+diag["run_trace_counts"]) or
                    diag["fit_observation_digest"]!=diag["executed_observation_digest"]):
                raise ValueError("iAPF trace or fitting-data mismatch")
            rows.append({"stage":name,"row":row,"result":result,"path":str(directory/saved["result_path"]),
                "digest":saved["result_digest"],"reused_completed_evidence":reused})
        numerical.extend(rows);write_json(root/"consumer-evidence.json",{"rows":numerical})
        return rows

    def stage(name,spec,reuse=False):
        if reuse:
            directory=prior_root/name
            state=json.loads((directory/"state.json").read_text())
            if state["execution_status"]!="complete" or state["study"]!=spec:
                raise ValueError("only identical completed studies can be reused")
            manifest["studies"].append({"name":name,"output":str(directory),"rows":len(spec["rows"]),
                "status":"complete","reused_completed_evidence":True,"additional_charges":0})
            return collect(name,spec,state,directory,True)
        write_json(root/f"{name}-study.json",spec)
        deferred=None
        try:
            state=execute(spec,registry,root/name,
                endpoint_loader=lambda endpoint:budget.wrap(load_endpoint(endpoint),name))
        except BudgetStop as error:
            deferred=error
            state=json.loads((root/name/"state.json").read_text())
        summary=report(root/name,registry)
        manifest["studies"].append({"name":name,"output":str(root/name),"rows":len(spec["rows"]),
            "status":summary["execution_status"],"wall_seconds":summary["wall_seconds"]})
        rows=collect(name,spec,state,root/name)
        save_budget()
        if deferred is not None:raise deferred
        if summary["execution_status"]!="complete":raise ValueError(f"{name} incomplete; preserve failure for repair")
        return rows

    try:
        manifest["hardware"]=configure_runtime(device="GPU",tf32=True,jit_compile=True)
        heuristic_tables=[]
        for regime,c,b in (("weak",.12,.04),("curved",.35,.12)):
            settings=dict(dimension=1,observation_dimension=1,horizon=2,particles=16,
                dtype="float32",device="GPU",tf32=True,jit_compile=True,
                theta=[.62,-.8,-.6,.9,.25,-.3],data_theta=[.62,-.8,-.6,.9,.25,-.3],
                transition_curve=c,observation_curve=b,prepared_data_regime="nonlinear_iapf_fit_calibration_"+regime,
                reference=dict(points=401,radius=9.,relative_tolerance=1e-7,tail_tolerance=1e-9))
            config=dict(k=1,tau=100.,max_iterations=4,max_particles=128,mean_bound=4.,sd_lower=.2,sd_upper=4.,
                max_fit_steps=2000,max_backtracks=30,fit_tolerance=1e-7,floor_ratio=.01,fit_theta=settings["theta"],fit_dtype="float64")
            widened={**config,"tau":25.,"mean_bound":8.,"sd_lower":.1,"sd_upper":8.,
                "max_fit_steps":4000,"max_backtracks":40,"fit_tolerance":1e-8,"floor_ratio":.005}
            family=[{"iapf":config},{"iapf":widened},{"iapf":{**widened,"tau":10.,"k":2}}]
            base=dict(model="nonlinear_scalar",proposal="iapf",estimator="nonlinear_analytical",
                comparison_target="model_score",comparison="approximation_error",replicate=0,coupling_group="baseline")
            study=dict(schema="younis_score_study_v1",phase="0F",version=1,plan=manifest["plan"],evidence_class="target_specific_calibration",seed=1000,
                settings=settings,required_proposals=["iapf"],tuning_candidate_family=family,
                budget=dict(wall_seconds=600,max_attempts=6,max_attempts_per_row=1),
                partitions=dict(calibration=[1000 if regime=="weak" else 1001],
                    validation=[1010 if regime=="weak" else 1011],
                    claim=[1020 if regime=="weak" else 1021]),
                rows=[dict(base,**candidate,id=f"candidate{index}-{role}",role=role,dataset=dataset)
                    for index,candidate in enumerate(family)
                    for role,dataset in (("calibration",1000 if regime=="weak" else 1001),
                                         ("validation",1010 if regime=="weak" else 1011))])
            sources=fingerprint(study,registry)["sources"]
            reused=prior_root is not None and regime=="weak"
            stage(regime+"-calibration",study,reuse=reused)
            selection_path=(prior_root if reused else root)/(regime+"-selection.json")
            selected=(json.loads(selection_path.read_text()) if reused else
                issue_selection(root/(regime+"-calibration"),selection_path))
            manifest["selections"][regime]=selected["selected_controls"]
            evaluation=deepcopy(study)
            evaluation["rows"]=[dict(base,**selected["selected_controls"],id=f"claim-{rep}",role="claim",
                dataset=1020 if regime=="weak" else 1021,
                replicate=rep,tuning_selection=str(selection_path)) for rep in range(2)]
            if reused:
                for row in evaluation["rows"]:
                    if consume_selection(row,{"study":evaluation})!=selected["selected_controls"]:
                        raise ValueError("saved selection failed re-derivation")
            claims=stage(regime+"-evaluation",evaluation,reuse=reused)
            a,z=(item["result"]["diagnostics"] for item in claims)
            if a["fit_digest"]!=z["fit_digest"] or a["fit_seed_records"]["iapf_final_process"]==z["fit_seed_records"]["iapf_final_process"]:
                raise ValueError("frozen shared fit or independent final sample check failed")
            comparator=comparator_study(study,base,1020 if regime=="weak" else 1021,a["actual_particle_count"])
            adversaries=comparator["required_proposals"]
            others=stage(regime+"-comparators",comparator)
            errors={r["row"]["proposal"]:r["result"]["diagnostics"]["score_squared_error"] for r in others}
            errors["iapf_replicate_mean"]=sum(r["result"]["diagnostics"]["score_squared_error"] for r in claims)/len(claims)
            losses=[name for name in adversaries if errors[name]<errors["iapf_replicate_mean"]]
            heuristic_tables.append({"regime":regime,"score_squared_errors":errors,"observed_losses_to":losses,
                "heuristic_dominance_verdict":"promotion_veto_descriptive" if losses else "no_observed_veto_in_tiny_fixture",
                "statistically_supported_ranking":False,"realized_particles":a["actual_particle_count"]})
            write_json(root/"conditional-heuristics.json",heuristic_tables)
            if fingerprint(study,registry)["sources"]!=sources:raise ValueError("source drift")
            manifest["source_fingerprint"]=sources
        manifest.update(status="complete",numerical_rows=len(numerical),same_fit_across_final_replicates=True,
            independent_final_streams=True,conditional_heuristic_evidence=str(root/"conditional-heuristics.json"))
        print(json.dumps({k:manifest[k] for k in ("status","numerical_rows","charged_row_or_recursive_fit_attempts")}))
    except BudgetStop as error:
        manifest.update(status="partial_budget_exhausted",error=str(error),numerical_rows=len(numerical))
        print(json.dumps({"status":manifest["status"],"reason":str(error),"charged_attempts":budget.used}))
    except Exception as error:
        manifest.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["wall_seconds"]=time.monotonic()-started
        manifest["process_cpu_seconds"]=time.process_time()-cpu_started
        manifest["cumulative_driver_wall_seconds"]=manifest["prior_wall_seconds"]+manifest["wall_seconds"]
        write_json(root/"run-manifest.json",manifest)


if __name__=="__main__":run()
