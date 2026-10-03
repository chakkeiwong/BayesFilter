"""R versus actual TF fixed-guide filter, including GPU/XLA execution.

Diagnostic only: tests this consumer's model/time/resampling contract, not the
paper-study adaptive learner and not the marginal-likelihood model score.
"""
import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def trace_comparison(baseline, candidate, uniforms, mixture, constant):
    """Chronological first changed decision; Python inspection only."""
    left, right = baseline["trace"], candidate["trace"]
    for t in range(len(uniforms)):
        if t and not constant:
            p, q = left["gaussian_probability"][t-1], right["gaussian_probability"][t-1]
            changed = [i for i,u in enumerate(mixture[t-1]) if (u<p[i]) != (u<q[i])]
            if changed:
                i=changed[0];u=mixture[t-1][i]
                return dict(kind="mixture",time=t-1,particle=i,changed_count=len(changed),
                    uniform=u,baseline_probability=p[i],candidate_probability=q[i],
                    baseline_margin=abs(u-p[i]),perturbation=abs(q[i]-p[i]))
        a,b=left["ancestor_indices"][t],right["ancestor_indices"][t]
        changed=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
        if changed:
            i=changed[0];u=uniforms[t][i]
            p,q=left["ancestor_cdf"][t],right["ancestor_cdf"][t]
            crossings=[dict(boundary=k,baseline=p[k],candidate=q[k],
                baseline_margin=abs(u-p[k]),perturbation=abs(q[k]-p[k]))
                for k in range(min(a[i],b[i]),max(a[i],b[i]))]
            return dict(kind="resampling",time=t,particle=i,changed_count=len(changed),
                uniform=u,baseline_ancestor=a[i],candidate_ancestor=b[i],crossings=crossings,
                max_cdf_difference=max(abs(x-y) for x,y in zip(p,q)))
    return dict(kind="none")

def worker(fixture, output, trace_mode=False):
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    policy = configure_tensorflow_gpu_memory_growth(tf)
    from bayesfilter.score_study import fitted_twist_tf as twist, iapf_adapter
    checks = []
    compilation = []
    traces = {}
    trace_checks = []
    def read(directory, name, shape, dtype):
        values = [float(row[0]) for row in csv.reader((directory/f"{name}.csv").open())]
        return tf.reshape(tf.constant(values, dtype=dtype), shape)
    with tf.device("/GPU:0"):
        for precision, tf32 in (("float64",False),("float32",False),("float32",True)):
            tf.config.experimental.enable_tensor_float_32_execution(tf32)
            for d in (1,2,5):
                directory=fixture/f"d{d}"
                N,T,o=64,5,d;dtype=tf.as_dtype(precision)
                arrays = [read(directory,name,shape,dtype) for name,shape in (
                    ("theta",[6]),("observations",[T,o]),("initial",[N,d]),("noise",[T,N,d]),
                    ("uniforms",[T+1,N]),("mixture",[T,N]),("centers",[T,d]),
                    ("covariances",[T,d,d]),("floors",[T]))]
                for constant in (False,True):
                    prefix="constant" if constant else "fitted"
                    # Fresh factory trace for each TF32 setting.
                    twist.make_fitted_twist_kernel.cache_clear()
                    kernel=twist.make_fitted_twist_kernel(d,o,N,T,precision,True,constant_twist=constant)
                    start=time.monotonic();actual=kernel(*arrays)
                    actual[0].numpy();seconds=time.monotonic()-start
                    hlo=kernel.experimental_get_compiler_ir(*arrays)(stage="hlo")
                    compilation.append(dict(dimension=d,dtype=precision,tf32=tf32,constant=constant,
                        compile_and_first_seconds=seconds,trace_count=kernel.experimental_get_tracing_count(),
                        hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(),device=actual[0].device))
                    if trace_mode:
                        traced_kernel=twist.make_fitted_twist_kernel(d,o,N,T,precision,True,
                            constant_twist=constant,include_numerical_trace=True)
                        observed=traced_kernel(*arrays)
                        differences=[float(tf.reduce_max(tf.abs(a-b)).numpy())
                            for a,b in zip(actual,observed[:3])]
                        trace={k:v.numpy().tolist() for k,v in observed[-1].items()}
                        supplied_uniforms=arrays[4].numpy().tolist()
                        supplied_mixture=arrays[5].numpy().tolist()
                        reconstructed=[[min(bisect.bisect_right(cdf,u),N-1) for u in us]
                            for cdf,us in zip(trace["ancestor_cdf"],supplied_uniforms)]
                        checks_ok=(reconstructed==trace["ancestor_indices"] and all(x==0 for x in differences))
                        trace_checks.append(dict(dimension=d,dtype=precision,tf32=tf32,constant=constant,
                            passed=checks_ok,observability_max_errors=differences,
                            cdf_reconstructs_actual_indices=reconstructed==trace["ancestor_indices"]))
                        traces[f"{d}/{precision}/{tf32}/{constant}"]=dict(trace=trace,
                            uniforms=supplied_uniforms,mixture=supplied_mixture,
                            clouds=actual[2].numpy().tolist())
                    for name,got in zip(("value","score","clouds"),actual):
                        expected=read(directory,prefix+"-"+name,got.shape,dtype)
                        error=tf.abs(got-expected)
                        if precision=="float64":
                            atol,rtol=(1e-5,1e-7) if name=="score" else (1e-9,1e-9)
                        else:
                            atol,rtol=(5e-4,1e-4) if name=="score" else (1e-4,1e-5)
                        valid=bool(tf.reduce_all(tf.math.is_finite(got)&
                            (error<=atol+rtol*tf.abs(expected))).numpy())
                        checks.append(dict(dimension=d,dtype=precision,tf32=tf32,constant=constant,
                            quantity=name,passed=valid,max_abs_error=float(tf.reduce_max(error).numpy()),
                            atol=atol,rtol=rtol,device=got.device))
        try:
            iapf_adapter.execute_iapf({"role":"diagnostic","model":"linear_gaussian"},
                dict(dimension=2,observation_dimension=2,particles=64,horizon=5),
                tf.zeros([6],tf.float64),tf.zeros([5,2],tf.float64),None)
        except ValueError as error:
            restriction="requires scalar state and observation" in str(error)
            reason=str(error)
        else:
            restriction=False;reason="Unexpected unrestricted consumer"
    report=dict(status="pass" if all(x["passed"] for x in checks) and restriction else "fail",
        checks=checks,compilation=compilation,tensorflow_version=tf.__version__,gpu_memory_policy=policy,
        allocator=tf.config.experimental.get_memory_info("GPU:0"),jit_compile=True,
        actual_adapter_scalar_veto=restriction,adapter_reason=reason,
        score_target="fixed-label finite-program derivative with guides frozen",
        model_score_checked=False,adaptive_learner_parity=False,paper_replication=False,default_readiness=False)
    if not all("GPU" in x["device"] for x in checks):
        report["status"]="fail";report["device_veto"]=True
    if trace_mode:
        localization=[]
        for d in (1,2,5):
            for constant in (False,True):
                baseline=traces[f"{d}/float32/False/{constant}"]
                candidate=traces[f"{d}/float32/True/{constant}"]
                localization.append(dict(dimension=d,constant=constant,
                    first_changed_decision=trace_comparison(baseline,candidate,
                        baseline["uniforms"],baseline["mixture"],constant),
                    max_cloud_error_by_time=[max(abs(x-y) for xs,ys in zip(a,b) for x,y in zip(xs,ys))
                        for a,b in zip(baseline["clouds"],candidate["clouds"])]))
        report.update(trace_checks=trace_checks,localization=localization,
            localization_valid=all(x["passed"] for x in trace_checks),
            precision_comparison_status=report["status"])
        # Known TF32 parity failures stay visible; successful diagnosis is a
        # different question from passing the original numerical comparison.
        diagnosis_ok=(report["localization_valid"] and restriction and
            all(x["passed"] for x in checks if not x["tf32"]) and not report.get("device_veto",False))
        report["status"]="diagnosis_complete" if diagnosis_ok else "diagnosis_invalid"
        output.with_name("numerical-traces.json").write_text(json.dumps(traces)+"\n")
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"checks":len(checks),
        "failed_count":sum(not x["passed"] for x in checks),"adapter_scalar_veto":restriction,
        "localization":report.get("localization")}))
    return 0 if report["status"] in ("pass","diagnosis_complete") else 1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker",action="store_true")
    p.add_argument("--trace",action="store_true",help="Localize precision failures in the actual consumer")
    p.add_argument("--center-probe",action="store_true")
    p.add_argument("--broadcast-probe",action="store_true")
    p.add_argument("--fixture",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.worker:
        if a.broadcast_probe:
            from diagnose_iapf_tf_center_probe import run_broadcast
            return run_broadcast(a.fixture,a.output)
        if a.center_probe:
            from diagnose_iapf_tf_center_probe import run
            return run(a.fixture,a.output)
        return worker(a.fixture,a.output,a.trace)
    repo=Path(__file__).resolve().parents[2]
    root=repo/"docs/plans/artifacts/iapf-renewed-mechanism-20260922-01"
    previous=sorted(root.glob("gpu-attempt*/manifest.json"))
    prior=[json.loads(x.read_text()) for x in previous]
    if any(x["status"]=="running" for x in prior):raise RuntimeError("GPU attempt already running")
    used=30+sum(x["GPU_process_seconds"] for x in prior)
    if used+1200>172800:raise RuntimeError("GPU budget exhausted")
    attempt=root/f"gpu-attempt{len(previous)+1:03d}"
    attempt.mkdir()
    source=attempt/"source"
    files=["bayesfilter/__init__.py","bayesfilter/runtime/__init__.py",
        "bayesfilter/runtime/gpu_memory_policy.py","docs/benchmarks/reference_iapf_paper.R",
        "docs/benchmarks/export_iapf_tf_filter_reference.R","docs/benchmarks/diagnose_iapf_tf_filter_bridge.py"]
    if a.center_probe or a.broadcast_probe:files.append("docs/benchmarks/diagnose_iapf_tf_center_probe.py")
    files += [str(x.relative_to(repo)) for x in (repo/"bayesfilter/score_study").glob("*.py")]
    for name in files:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="GPU-68251639-fe82-8f81-3ccc-2953c32e805b",
        TF_FORCE_GPU_ALLOW_GROWTH="true",BAYESFILTER_PRELOAD_CUSTOM_OP="0",PYTHONPATH=str(source),
        OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",TF_NUM_INTRAOP_THREADS="2",TF_NUM_INTEROP_THREADS="2")
    fixture=attempt/"fixture"
    commands=[["Rscript","--vanilla",str(source/"docs/benchmarks/export_iapf_tf_filter_reference.R"),str(fixture)],
        [sys.executable,str(source/"docs/benchmarks/diagnose_iapf_tf_filter_bridge.py"),"--worker",
         "--fixture",str(fixture),"--output",str(attempt/"bridge.json")]]
    if a.trace:commands[1].append("--trace")
    if a.center_probe:commands[1].append("--center-probe")
    if a.broadcast_probe:commands[1].append("--broadcast-probe")
    manifest=dict(status="running",commands=commands,
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
        dirty_worktree=True,plan="docs/plans/iapf-renewed-mechanism-campaign-2026-09-22.md",
        source_sha256={name:sha(source/name) for name in files},started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        environment={k:env.get(k, "") for k in ("CUDA_VISIBLE_DEVICES","TF_FORCE_GPU_ALLOW_GROWTH","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","XLA_FLAGS")},
        seeds=[93600001,93600002,93600005],role="independent_R_to_actual_TF_GPU_XLA_filter_diagnostic",
        target="TF parameterized LGSSM; x0 transition then first observation; every-step resampling",
        result=str(attempt/"bridge.json"),GPU_process_seconds=0,trace_localization=a.trace,
        CPU_reference_seconds=0)
    manifest["center_probe"]=a.center_probe
    manifest["broadcast_probe"]=a.broadcast_probe
    path=attempt/"manifest.json";path.write_text(json.dumps(manifest,indent=2)+"\n")
    start=time.monotonic()
    try:
        with (attempt/"run.log").open("w") as log:
            subprocess.run(commands[0],cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES="-1"),stdout=log,
                           stderr=subprocess.STDOUT,timeout=120,check=True)
            manifest["CPU_reference_seconds"]=time.monotonic()-start
            result=subprocess.run(commands[1],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,
                                  timeout=1080,check=False)
        manifest.update(status="complete" if result.returncode==0 else "failed",returncode=result.returncode)
    except Exception as error:
        manifest.update(status="failed",error=repr(error))
    finally:
        manifest.update(GPU_process_seconds=time.monotonic()-start,
                        finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        manifest["output_sha256"]={str(x.relative_to(attempt)):sha(x) for x in attempt.rglob("*")
            if x.is_file() and source not in x.parents and x!=path}
        path.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:manifest[k] for k in ("status","GPU_process_seconds")})+" "+str(path))
    return 0 if manifest["status"]=="complete" else 1

if __name__=="__main__":
    raise SystemExit(main())
