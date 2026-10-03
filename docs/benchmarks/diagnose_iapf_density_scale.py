"""Known-target density-scale diagnosis; all alternative scaling is diagnostic.

Uses the actual bounded TF fitter, unchanged default parameters, and a scoped
constant multiplier of the same objective. No production default is modified.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

from diagnose_iapf_adaptive_consumer import REPO, setup, r_literal, save, controls

ROOT=REPO/"docs/plans/artifacts/iapf-density-scale-20260922-01"


def diagnose(args):
    directory=ROOT/args.attempt; directory.mkdir(exist_ok=False)
    tf,environment=setup(args.device)
    from bayesfilter.score_study import iapf_fit_tf as module
    original=module._density_profile
    config=controls(); results=[]; rfixtures=[]; kernels={}
    def scaled(z,log_target,parameters,objective="density_l2"):
        if objective!="density_l2": raise ValueError("same density objective required")
        loss,gradient,log_scale,shape,amplitude=original(z,log_target,parameters,objective)
        initial_log_amplitude=-.5*tf.reduce_min(tf.reduce_sum(z*z,axis=1))
        multiplier=tf.exp(-2*initial_log_amplitude)
        return loss*multiplier,gradient*multiplier,log_scale,shape,amplitude
    def kernel(d,arm):
        key=(d,arm)
        if key not in kernels:
            @tf.function(input_signature=[tf.TensorSpec([1000,d],tf.float64),tf.TensorSpec([1000],tf.float64)],jit_compile=True)
            def evaluate(points,targets):
                return module.bounded_density_fit(points,targets,mean_bound=4.,sd_lower=.2,sd_upper=4.,
                    max_steps=2000,max_backtracks=30,tolerance=1e-7,floor_ratio=.01)
            kernels[key]=evaluate
        return kernels[key]
    with tf.device("/CPU:0" if args.device=="cpu" else "/GPU:0"):
        if args.source_attempt:
            inputs=json.loads((ROOT/args.source_attempt/"inputs.json").read_text())
        else:
            inputs=[]
            for d in (5,10,20,40,80):
                for seed in (71,72):
                    points=tf.random.stateless_normal([1000,d],[938000+seed,d],dtype=tf.float64)
                    j=tf.cast(tf.range(1,d+1),tf.float64)
                    for regime in ("healthy","shifted_narrow","shifted_broad"):
                        if regime=="healthy":
                            center=tf.reduce_mean(points,0);variance=tf.math.reduce_variance(points,0)
                        elif regime=="shifted_narrow":
                            center=.4*tf.sin(j);variance=tf.linspace(tf.constant(.6,tf.float64),tf.constant(.9,tf.float64),d)
                        else:
                            center=.4*tf.cos(j);variance=tf.linspace(tf.constant(1.4,tf.float64),tf.constant(1.8,tf.float64),d)
                        targets=-.5*tf.reduce_sum((points-center)**2/variance,axis=1)
                        inputs.append(dict(d=d,seed=seed,regime=regime,points=points.numpy().tolist(),
                            targets=targets.numpy().tolist(),center=center.numpy().tolist(),variance=variance.numpy().tolist()))
        save(directory/"inputs.json",inputs)
        for fixture in inputs:
            d=fixture["d"];points=tf.constant(fixture["points"],tf.float64);target=tf.constant(fixture["targets"],tf.float64)
            true_center=tf.constant(fixture["center"],tf.float64);true_variance=tf.constant(fixture["variance"],tf.float64)
            mu=tf.reduce_mean(points,0);sd=tf.math.reduce_std(points,0);z=(points-mu)/sd
            initial=original(z,target-tf.reduce_max(target),tf.zeros([2*d],tf.float64))
            p2=tf.exp(-tf.reduce_sum(z*z,axis=1));scores=tf.concat([z,z*z-1],axis=1)
            bound=2*tf.reduce_max(tf.abs(scores))*tf.reduce_mean(p2)
            true_par=tf.concat([(true_center-mu)/sd,tf.math.log(tf.sqrt(true_variance)/sd)],axis=0)
            optimum=original(z,target-tf.reduce_max(target),true_par)
            result={k:fixture[k] for k in ("d","seed","regime")}
            result.update(initial_gradient_max=float(tf.reduce_max(tf.abs(initial[1])).numpy()),
                initial_loss=float(initial[0].numpy()),initial_shape=float(initial[3].numpy()),
                initial_log_amplitude=float(initial[4].numpy()),gradient_bound=float(bound.numpy()),
                gradient_bound_pass=bool(tf.reduce_all(tf.abs(initial[1])<=bound*(1+1e-12)).numpy()),
                analytic_optimum_loss=float(optimum[0].numpy()),analytic_optimum_shape=float(optimum[3].numpy()),arms={})
            for arm in ("baseline","fixed_initial_density_scale"):
                started=time.monotonic()
                with patch.object(module,"_density_profile",scaled) if arm!="baseline" else nullcontext():
                    center,V,floor,info=kernel(d,arm)(points,target)
                variance=tf.linalg.diag_part(V)
                kl=.5*tf.reduce_sum((true_variance+(true_center-center)**2)/variance-1+tf.math.log(variance/true_variance))
                record={k:v.numpy().tolist() for k,v in info.items()}
                record.update(center=center.numpy().tolist(),variance=variance.numpy().tolist(),
                    log_floor=float(floor.numpy()),KL_target_to_guide=float(kl.numpy()),
                    center_error=float(tf.reduce_max(tf.abs(center-true_center)).numpy()),
                    variance_error=float(tf.reduce_max(tf.abs(variance-true_variance)).numpy()),
                    wall_seconds=time.monotonic()-started,
                    trace_count=kernel(d,arm).experimental_get_tracing_count())
                record["uninformative_zero_step_convergence"]=(record["valid"] and record["converged"] and
                    record["iterations"]==0 and (record["KL_target_to_guide"]>.01 or record["normalized_shape_residual"]>.01))
                result["arms"][arm]=record
            a,b=result["arms"].values()
            result["healthy_nonharm_pass"]=(fixture["regime"]!="healthy" or
                max(abs(x-y) for key in ("center","variance") for x,y in zip(a[key],b[key]))<=1e-10)
            results.append(result)
            save(directory/"results.json",results)
            print(json.dumps({k:result[k] for k in ("d","seed","regime","initial_gradient_max","gradient_bound")}),flush=True)
    rseconds=0.
    if args.device=="cpu":
        (directory/"R-input.R").write_text("fixtures <- "+r_literal(inputs)+"\n")
        command=["Rscript","--vanilla","docs/benchmarks/diagnose_iapf_density_scale.R",str(directory)]
        start=time.monotonic()
        with (directory/"R.log").open("w") as log:
            run=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        rseconds=time.monotonic()-start
        rrecord=dict(command=command,exit_code=run.returncode,wall_seconds=rseconds)
    else:
        rrecord=dict(classification="R comparison preserved in CPU source attempt",source_attempt=args.source_attempt)
        environment["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
    save(directory/"summary.json",dict(cases=len(results),environment=environment,R=rrecord,
        diagnostic_flags=sum(x["arms"]["baseline"]["uninformative_zero_step_convergence"] for x in results),
        all_gradient_bounds_pass=all(x["gradient_bound_pass"] for x in results),
        all_healthy_nonharm_pass=all(x["healthy_nonharm_pass"] for x in results),
        source_attempt=args.source_attempt))


def launch(args):
    if not args.attempt.isidentifier(): raise ValueError("simple versioned attempt name required")
    manifest=ROOT/(args.attempt+"-launch.json")
    if manifest.exists(): raise FileExistsError(manifest)
    budget=json.loads((ROOT/"budget.json").read_text())
    if len(list(ROOT.glob("*-launch.json")))>=6 or budget["phase_cpu_seconds"]>=3600 or budget["phase_gpu_seconds"]>=900:
        raise RuntimeError("bounded suballocation exhausted")
    command=[sys.executable,str(Path(__file__).resolve()),"--worker","--attempt",args.attempt,"--device",args.device]
    if args.source_attempt: command += ["--source-attempt",args.source_attempt]
    sources=[str(Path(__file__).resolve().relative_to(REPO)),"docs/benchmarks/diagnose_iapf_density_scale.R",
        "docs/benchmarks/diagnose_iapf_adaptive_consumer.py","docs/benchmarks/reference_iapf_paper.R",
        "bayesfilter/score_study/iapf_fit_tf.py","bayesfilter/score_study/fitted_twist_tf.py",
        "docs/plans/iapf-density-scale-diagnosis-2026-09-22.md"]
    hashes={}
    for name in sources:
        p=REPO/name; dest=ROOT/(args.attempt+"-source")/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    record=dict(command=command,device=args.device,sources=hashes,status="running",
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
        plan="docs/plans/iapf-density-scale-diagnosis-2026-09-22.md")
    save(manifest,record)
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"-1" if args.device=="cpu" else "GPU-68251639-fe82-8f81-3ccc-2953c32e805b",
        "TF_FORCE_GPU_ALLOW_GROWTH":"true","OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
        "TF_NUM_INTRAOP_THREADS":"1","TF_NUM_INTEROP_THREADS":"1","BAYESFILTER_PRELOAD_CUSTOM_OP":"0",
        "MPLCONFIGDIR":"/tmp/iapf-density-scale-matplotlib"}
    start=time.monotonic()
    with (ROOT/(args.attempt+".log")).open("x") as log:
        try:
            run=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            code=run.returncode
        except subprocess.TimeoutExpired: code=124
    elapsed=time.monotonic()-start
    record.update(status="finished",exit_code=code,wall_seconds=elapsed,
        cpu_seconds=elapsed if args.device=="cpu" else 0,gpu_seconds=elapsed if args.device=="gpu" else 0)
    save(manifest,record)
    budget["phase_cpu_seconds"]+=record["cpu_seconds"];budget["phase_gpu_seconds"]+=record["gpu_seconds"]
    budget["remaining_cpu_seconds"]-=record["cpu_seconds"];budget["remaining_gpu_seconds"]-=record["gpu_seconds"]
    save(ROOT/"budget.json",budget)
    print(json.dumps({k:record[k] for k in ("status","exit_code","wall_seconds")}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--attempt",required=True)
    p.add_argument("--device",choices=("cpu","gpu"),default="cpu");p.add_argument("--source-attempt")
    p.add_argument("--worker",action="store_true");a=p.parse_args()
    diagnose(a) if a.worker else launch(a)
