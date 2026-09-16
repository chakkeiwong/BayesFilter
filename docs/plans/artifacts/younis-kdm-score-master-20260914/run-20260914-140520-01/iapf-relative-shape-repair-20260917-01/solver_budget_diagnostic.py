"""Single saved-validation replay: numerical diagnosis, no selection or claims."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from bayesfilter.score_study.contracts import DiagnosticFailure, digest, validate_result, validate_study
from bayesfilter.score_study.coordinator import execute, fingerprint, load_endpoint
from bayesfilter.score_study.registry import default_registry
from bayesfilter.score_study.runtime import configure_runtime, memory_usage

base=Path(__file__).resolve().parent
root=base/"launch03-solver-diagnostic"
read=lambda p:json.loads(p.read_text())
write=lambda p,x:p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
assert revision=="e6298503bf923a2aa8ee9fff53732b79eac205cb"
assert not subprocess.check_output(["git","status","--porcelain"],text=True)
study=read(base/"launch02/weak-selected-source-study.json")
prior_failure=read(base/"launch02/invalid-fits.json")[0]
assert prior_failure["row"]["id"]=="candidate0-validation-1212"
row=next(r for r in study["rows"] if r["id"]==prior_failure["row"]["id"])
row["iapf"]={**row["iapf"],"max_fit_steps":10000}
row["role"]="mechanics"
study["rows"]=[row]
study["evidence_class"]="mechanics"
study["plan"]=str(base.parents[3]/"younis-score-iapf-solver-budget-diagnostic-2026-09-17.md")
assert Path(study["plan"]).is_file()
study.pop("tuning_candidate_family")
study["budget"]={"wall_seconds":580,"max_attempts":1,"max_attempts_per_row":1}
registry=default_registry()
assert all(x["status"]=="runnable" for x in validate_study(study,registry))
before=fingerprint(study,registry)
root.mkdir(exist_ok=False)
write(root/"study.json",study)
started,cpu=time.monotonic(),time.process_time()
manifest=dict(status="running",git_commit=revision,source_checkout=str(Path.cwd()),
    command=sys.argv,environment_name="tftwogpu",plan=study["plan"],launch_number=3,
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    environment={k:os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES","TF_FORCE_GPU_ALLOW_GROWTH","BAYESFILTER_PRELOAD_CUSTOM_OP")},
    prior_charges=246,maximum_new_charges=4,charge_limit=280,
    seed=study["seed"],dataset=1212,data_version="same validation replay, not fresh evidence",
    inference_status="numerical_diagnosis_only",default_ready=False,
    statistically_supported_ranking=False,score_promotion=False,result_file=str(root/"comparison.json"),
    artifact_root=str(root),cpu_only=False,gpu_trust="escalated_local_gpu_access")
def save():
    manifest.update(wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu)
    write(root/"run-manifest.json",manifest)
save()
failure=[]
def loader(name):
    endpoint=load_endpoint(name)
    def checked(row,context):
        try:
            result=endpoint(row,context)
            validate_result(result,row,registry)
            runtime=result["runtime"]
            assert runtime["device"]=="GPU" and runtime["tf32"] and runtime["jit_compile"]
            assert runtime["memory_policy"]["all_physical_devices_memory_growth"]
            diag=result["diagnostics"]
            assert all(n==1 for n in diag["fit_trace_counts"]+diag["run_trace_counts"])
            manifest["new_charges"]=1+diag["work_accounting"]["offline_fit_calls"]
            return result
        except DiagnosticFailure as error:
            failure.append(dict(reason=str(error),diagnostics=error.diagnostics))
            write(root/"invalid-fit.json",failure)
            raise
        finally:
            manifest.setdefault("new_charges",4)
            save()
    return checked
try:
    manifest["runtime"]=configure_runtime(device="GPU",tf32=True,jit_compile=True)
    import tensorflow as tf
    manifest["physical_gpu_details"]=[dict(name=d.name,**tf.config.experimental.get_device_details(d))
        for d in tf.config.list_physical_devices("GPU")]
    state=execute(study,registry,root/"study",endpoint_loader=loader)
    assert fingerprint(study,registry)==before==state["fingerprint"]
    saved=state["rows"][row["id"]]
    if saved["execution_status"]=="complete":
        result=read(root/"study"/saved["result_path"])
        assert digest(result)==saved["result_digest"]
        diag=result["diagnostics"]
    else:
        assert len(failure)==1, "unclassified infrastructure or validity failure"
        diag=failure[0]["diagnostics"]["details"]
    previous=prior_failure["diagnostics"]["details"]
    assert all(diag["fit_seed_records"][k]==v for k,v in previous["fit_seed_records"].items())
    assert diag["fit_iterations"][0]["log_value"]==previous["fit_iterations"][0]["log_value"]
    comparison=dict(status="complete" if saved["execution_status"]=="complete" else "candidate_rejected",
        same_first_offline_streams=True,same_first_offline_log_value=True,
        baseline_max_fit_steps=2000,candidate_max_fit_steps=10000,unchanged_tolerance=1e-7,
        baseline_first_fit=previous["fit_iterations"][0],candidate_first_fit=diag["fit_iterations"][0],
        diagnostic_columns=diag["fit_diagnostic_columns"],
        score_or_default_promotion=False,source_fingerprint=before)
    write(root/"comparison.json",comparison)
    manifest.update(status=comparison["status"],total_charges=246+manifest["new_charges"],
        allocator=memory_usage("GPU"))
    assert manifest["new_charges"]<=4 and manifest["total_charges"]<=280
except BaseException as error:
    manifest.update(status="execution_stopped",error=f"{type(error).__name__}: {error}")
    raise
finally:
    save()
print(json.dumps({k:manifest[k] for k in ("status","new_charges","total_charges","wall_seconds")}))
