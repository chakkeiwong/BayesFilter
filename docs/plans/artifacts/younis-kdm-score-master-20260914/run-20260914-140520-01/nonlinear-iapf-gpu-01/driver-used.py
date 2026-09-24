"""Bounded nonlinear iAPF mechanics and conditional heuristic diagnostics."""
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


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--source-revision",required=True)
    args=parser.parse_args();root=args.output.resolve()
    if root.exists():raise ValueError("fresh versioned output required")
    revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    if revision!=args.source_revision:raise ValueError("source revision mismatch")
    if subprocess.check_output(["git","status","--porcelain"],text=True):raise ValueError("dirty source snapshot")
    from bayesfilter.score_study.coordinator import execute,report,fingerprint,write_json
    from bayesfilter.score_study.registry import default_registry
    from bayesfilter.score_study.runtime import configure_runtime
    from bayesfilter.score_study.tuning import issue_selection
    registry=default_registry();root.mkdir(parents=True);started=time.monotonic()
    manifest={"schema":"nonlinear_iapf_mechanics_v1","status":"running","git_commit":revision,
        "source_root":str(Path.cwd()),"command":sys.argv,"interpreter":sys.executable,
        "plan":"docs/plans/younis-score-nonlinear-iapf-2026-09-16.md",
        "driver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment":{k:os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES","TF_FORCE_GPU_ALLOW_GROWTH",
            "BAYESFILTER_PRELOAD_CUSTOM_OP","TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS","OMP_NUM_THREADS")},
        "studies":[],"selections":{},"evidence_class":"mechanics_only","statistically_supported_ranking":False,
        "default_ready":False,"runtime_comparison":"forbidden_concurrent_workloads",
        "score_semantics":"finite_program_fixed_fit_realized_N_and_labels"}
    numerical=[];charges=[]
    write_json(root/"run-manifest.json",manifest)

    def stage(name,spec):
        write_json(root/f"{name}-study.json",spec)
        state=execute(spec,registry,root/name)
        summary=report(root/name,registry)
        spec_rows={row["id"]:row for row in spec["rows"]}
        for attempt in state["attempts"]:
            directory=root/name/attempt["path"];row=spec_rows[attempt["row"]]
            if (directory/"result.json").exists():
                details=json.loads((directory/"result.json").read_text())["diagnostics"]
                fits=details.get("work_accounting",{}).get("offline_fit_calls",0)
                basis="executed_work_accounting"
            elif (directory/"failure_diagnostics.json").exists():
                details=json.loads((directory/"failure_diagnostics.json").read_text())
                fits=sum("density_fit_diagnostics" in item for item in details.get("fit_iterations",[]))
                basis="preserved_failed_fit_diagnostics"
            else:
                fits=row.get("iapf",{}).get("max_iterations",1)-1
                basis="conservative_unknown_failure_upper_bound"
            charges.append({"stage":name,"row":attempt["row"],"row_attempts":1,"recursive_fits":fits,"basis":basis})
        manifest["studies"].append({"name":name,"output":str(root/name),"rows":len(spec["rows"]),
            "status":summary["execution_status"],"wall_seconds":summary["wall_seconds"]})
        manifest["wall_seconds"]=time.monotonic()-started
        manifest["charged_row_or_recursive_fit_attempts"]=sum(1+x["recursive_fits"] for x in charges)
        write_json(root/"attempt-accounting.json",charges);write_json(root/"run-manifest.json",manifest)
        if manifest["charged_row_or_recursive_fit_attempts"]>64:raise ValueError("per-launch attempt bound exceeded")
        if summary["execution_status"]!="complete":raise ValueError(f"{name} incomplete; preserve failure for repair")
        rows=[]
        for row in spec["rows"]:
            saved=state["rows"][row["id"]];result=json.loads((root/name/saved["result_path"]).read_text())
            runtime=result["runtime"];diag=result["diagnostics"]
            if not (runtime["device"]=="GPU" and runtime["tf32"] and runtime["jit_compile"] and
                    runtime["traces"]==1 and runtime["memory_policy"]["all_physical_devices_memory_growth"]):
                raise ValueError("GPU/XLA/trace/memory evidence mismatch")
            if row["proposal"]=="iapf" and (any(n!=1 for n in diag["fit_trace_counts"]+diag["run_trace_counts"]) or
                    diag["fit_observation_digest"]!=diag["executed_observation_digest"]):
                raise ValueError("iAPF trace or fitting-data mismatch")
            rows.append({"stage":name,"row":row,"result":result,"path":str(root/name/saved["result_path"]),"digest":saved["result_digest"]})
        numerical.extend(rows);write_json(root/"consumer-evidence.json",{"rows":numerical})
        return rows

    try:
        manifest["hardware"]=configure_runtime(device="GPU",tf32=True,jit_compile=True)
        heuristic_tables=[]
        for regime,c,b in (("weak",.12,.04),("curved",.35,.12)):
            settings=dict(dimension=1,observation_dimension=1,horizon=2,particles=16,
                dtype="float32",device="GPU",tf32=True,jit_compile=True,
                theta=[.62,-.8,-.6,.9,.25,-.3],data_theta=[.62,-.8,-.6,.9,.25,-.3],
                transition_curve=c,observation_curve=b,prepared_data_regime="nonlinear_iapf_mechanics_"+regime,
                reference=dict(points=401,radius=9.,relative_tolerance=1e-7,tail_tolerance=1e-9))
            config=dict(k=1,tau=100.,max_iterations=4,max_particles=128,mean_bound=4.,sd_lower=.2,sd_upper=4.,
                max_fit_steps=2000,max_backtracks=30,fit_tolerance=1e-7,floor_ratio=.01,fit_theta=settings["theta"],fit_dtype="float64")
            family=[{"iapf":config},{"iapf":{**config,"k":2,"max_iterations":5}}]
            base=dict(model="nonlinear_scalar",proposal="iapf",estimator="nonlinear_analytical",
                comparison_target="model_score",comparison="approximation_error",replicate=0,coupling_group="baseline")
            study=dict(schema="younis_score_study_v1",phase="0E",version=1,plan=manifest["plan"],evidence_class="mechanics",seed=941,
                settings=settings,required_proposals=["iapf"],tuning_candidate_family=family,
                budget=dict(wall_seconds=600,max_attempts=4,max_attempts_per_row=1),
                partitions=dict(calibration=[900],validation=[910],claim=[920]),
                rows=[dict(base,**candidate,id=f"candidate{index}-{role}",role=role,dataset=dataset)
                    for index,candidate in enumerate(family) for role,dataset in (("calibration",900),("validation",910))])
            sources=fingerprint(study,registry)["sources"]
            stage(regime+"-calibration",study)
            selection_path=root/(regime+"-selection.json")
            selected=issue_selection(root/(regime+"-calibration"),selection_path)
            manifest["selections"][regime]=selected["selected_controls"]
            evaluation=deepcopy(study)
            evaluation["rows"]=[dict(base,**selected["selected_controls"],id=f"claim-{rep}",role="claim",dataset=920,
                replicate=rep,tuning_selection=str(selection_path)) for rep in range(2)]
            claims=stage(regime+"-evaluation",evaluation)
            a,z=(item["result"]["diagnostics"] for item in claims)
            if a["fit_digest"]!=z["fit_digest"] or a["fit_seed_records"]["iapf_final_process"]==z["fit_seed_records"]["iapf_final_process"]:
                raise ValueError("frozen shared fit or independent final sample check failed")
            comparator=deepcopy(study);comparator["settings"]["particles"]=a["actual_particle_count"]
            adversaries=("ekf","ukf","bootstrap","local_linear");comparator["required_proposals"]=list(adversaries)
            comparator["rows"]=[{**base,"id":name,"proposal":name,"role":"mechanics","dataset":920} for name in adversaries]
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
    except Exception as error:
        manifest.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["wall_seconds"]=time.monotonic()-started
        write_json(root/"run-manifest.json",manifest)


if __name__=="__main__":run()
