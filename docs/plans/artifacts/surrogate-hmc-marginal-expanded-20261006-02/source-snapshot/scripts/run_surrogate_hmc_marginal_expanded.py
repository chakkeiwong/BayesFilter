#!/usr/bin/env python3
"""Bounded, resumable fresh-data diagnostics for marginal LEDH weights.

No HMC admission or default selection authority. The parent does not import
TensorFlow. Each GPU worker has a manifest and a bounded lifetime.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/surrogate-hmc-marginal-expanded-execution-2026-10-06.md"
GPU = "GPU-68251639-fe82-8f81-3ccc-2953c32e805b"
POLICIES = ("ancestor", "marginal_mixture")
HORIZONS = (10,20,50,120,240)
MODELS = ("lg1","lg3","ksc","m13","lg10","lg20")


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def identity():
    paths = sorted((ROOT/"bayesfilter").rglob("*.py"))
    paths += [ROOT/name for name in ("scripts/run_surrogate_hmc_marginal_expanded.py",
                                    "scripts/diagnose_marginal_weight_campaign.py")]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def configuration(args):
    scopes = args.scopes.split(",") if args.scopes else [f"{m}-t{t}" for t in HORIZONS for m in MODELS]
    allowed = {f"{m}-t{t}" for m in MODELS for t in HORIZONS}
    if not scopes or len(set(scopes))!=len(scopes) or any(s not in allowed for s in scopes):
        raise ValueError("scopes must be distinct members of the declared grid")
    if args.budget_seconds<=0 or not math.isfinite(args.budget_seconds):
        raise ValueError("positive finite budget required")
    ns = [int(n) for n in args.particle_ladder.split(",")]
    if not ns or any(n not in (1024,4096,16384) for n in ns) or len(ns)!=len(set(ns)):
        raise ValueError("particle ladder must contain distinct declared counts")
    return dict(scopes=scopes,particles=ns,budget_seconds=args.budget_seconds,
                gpu=args.gpu,datasets=[4,8,16],designs=32,plan=PLAN,
                worker_scope_cap_seconds=7200,worker_check_cap_seconds=3600,
                harness_revision="dynamic_reference_graphs_matched_fp32_baselines_v2",
                dimension_policy="LG1_3_10_20_native_KSC1_range4_pending_owner_extension")


def setup(args):
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS","6")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS","2")
    sys.path.insert(0,str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf)
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPU worker requires trusted GPU visibility")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing import marginal_expanded as ex
    return tf,ex,dict(python=sys.executable,tensorflow=tf.__version__,gpu=args.gpu,
        physical_devices=[d.name for d in tf.config.list_physical_devices("GPU")],
        memory_policy=memory,jit_compile=True,tf32=True,cpu_only=False,
        comparison_lane="full_correction_observability")


def worker(args):
    tf,ex,environment=setup(args);m=ex.base
    directory=args.attempt
    started=time.monotonic()
    manifest=dict(status="RUNNING",worker=args.worker,scope=args.scope,particles=args.n,
        command=sys.argv,git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        source_sha256=identity(),environment=environment,plan=PLAN,result=str(directory/"result.json"),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        seed_policy="20261006 + 7919*(dataset_id+1), partition and replicate offsets")
    write(directory/"manifest.json",manifest)
    def check():
        if time.time() >= args.deadline: raise TimeoutError("worker/campaign budget exhausted")
    def evaluate(function,*values):
        check();tick=time.monotonic()
        value=m.serial(function(*values))
        if not isinstance(value,dict): value=dict(value=value)
        value["seconds"]=time.monotonic()-tick
        return value
    def save(path,value): m.save(path,value)

    try:
        if args.worker=="engineering":
            rows=[]
            for name in ("lg3-t2","ksc-t2","m13-t2"):
                spec=m.specification(name,tf.float64);n=32
                observed=tf.cast(tf.cast(m.dataset(spec,"engineering",dataset_id=0),tf.float32),tf.float64)
                design=m.designs(spec,n,"engineering",0,tf.float64,dataset_id=0)
                theta=tf.constant(spec["points"],tf.float64)
                controls=m.controls(1 if spec["kind"]=="m13" else 0)
                for policy in POLICIES:
                    batch=ex.compile_batch(spec,n,controls,policy,rows=len(spec["points"]),dtype=tf.float64)
                    scalar=ex.compile_batch(spec,n,controls,policy,dtype=tf.float64)
                    b=evaluate(batch,theta,*design,observed)
                    checks=[]
                    for point in range(len(spec["points"])):
                        single=ex.one(evaluate(scalar,theta[point:point+1],*design,observed))
                        parity=max([abs(single["value"]-b["value"][point])/(1+abs(single["value"]))]+
                            [abs(x-y)/(1+abs(x)) for x,y in zip(single["score"],b["score"][point])])
                        fd_checks=[]
                        for h in (1e-4,3e-5):
                            for j in range(spec["p"]):
                                step=tf.one_hot(j,spec["p"],dtype=tf.float64)[None]*h
                                plus=ex.one(evaluate(scalar,theta[point:point+1]+step,*design,observed))
                                minus=ex.one(evaluate(scalar,theta[point:point+1]-step,*design,observed))
                                fd=(plus["value"]-minus["value"])/(2*h)
                                relative=abs(fd-single["score"][j])/(1+abs(single["score"][j]))
                                fd_checks.append(dict(h=h,j=j,relative_error=relative,
                                    valid=plus["valid"] and minus["valid"],finite_difference=fd))
                        passed=(b["valid"][point] and single["valid"] and parity<3e-5 and
                                all(r["valid"] and r["relative_error"]<3e-4 for r in fd_checks))
                        checks.append(dict(point=point,parity_relative=parity,fd=fd_checks,passed=passed,single=single))
                    rows.append(dict(scope=name,policy=policy,batch=b,checks=checks,
                                     passed=all(r["passed"] for r in checks)))
                    save(directory/"progress.json",dict(rows=rows))
                    print(name,policy,"engineering",rows[-1]["passed"],flush=True)
            spec=m.specification("m13-t1",tf.float64)
            observed=m.dataset(spec,"engineering",dataset_id=0)
            quad,quad_meta=m.references(spec,observed,claim=True)
            kernels=ex.Kernels(spec)
            pf=ex.references(kernels,observed,0,"engineering",evaluate,True)
            check()
            consistency=[]
            for q,r,u in zip(quad,pf["rows"],pf["uncertainty"]):
                delta=ex.discrepancies([q],[r])[0]
                consistency.append(dict(delta=delta,sensitivity=u,
                    within_six_mcse=all(delta[k]<=2*u[k]+.001 for k in ("value","score"))))
            reference_check=dict(quadrature=quad_meta,particle_reference=pf,consistency=consistency,
                qualification="T=1 checks the reference implementation; it does not validate long-horizon references")
            result=dict(status="COMPLETE",passed=(all(r["passed"] for r in rows) and quad_meta["valid"]
                        and pf["valid"] and all(r["within_six_mcse"] for r in consistency)),
                        rows=rows,reference_check=reference_check)

        elif args.worker=="precision":
            rows=[]
            name="m13-t20";n=4096
            outputs={}
            # Both modes consume exactly the same rounded theta/data/design.
            for dtype in (tf.float32,tf.float64):
                spec=m.specification(name,dtype)
                observed=tf.cast(tf.cast(m.dataset(spec,"engineering",dataset_id=1),tf.float32),dtype)
                design=m.designs(spec,n,"engineering",1,dtype,dataset_id=1)
                theta=tf.cast(tf.constant(spec["points"],tf.float32),dtype)
                for policy in POLICIES:
                    function=ex.compile_batch(spec,n,m.controls(1),policy,dtype=dtype)
                    points=[]
                    for i in range(len(spec["points"])):
                        repeats=[ex.one(evaluate(function,theta[i:i+1],*design,observed)) for _ in range(3)]
                        points.append(repeats)
                    outputs[dtype.name+"_"+policy]=points
                    save(directory/"progress.json",outputs)
                    print(dtype.name,policy,"precision evaluated",flush=True)
            for policy in POLICIES:
                for i in range(2):
                    fp32=outputs["float32_"+policy][i];fp64=outputs["float64_"+policy][i]
                    delta=ex.discrepancies([fp32[0]],[fp64[0]])[0]
                    rows.append(dict(policy=policy,point=i,precision_delta=delta,
                        valid=all(r["valid"] for r in fp32+fp64),
                        repeat_value_spread=max(r["value"] for r in fp32)-min(r["value"] for r in fp32),
                        warning=max(delta.values())>.05))
            result=dict(status="COMPLETE",passed=all(r["valid"] for r in rows),rows=rows,outputs=outputs,
                        qualification="finite precision differences warn; finite-program derivative tests remain separate")

        else:
            spec=m.specification(args.scope);n=args.n
            kernels=ex.Kernels(spec);compiled={}
            def filt(setting,policy):
                key=(setting,policy)
                if key not in compiled: compiled[key]=ex.compile_batch(spec,n,m.controls(setting),policy)
                return compiled[key]
            def ref(partition,idx,observed,claim):
                path=args.output/"references"/args.scope/f"{partition}-{idx}.json"
                if path.exists():
                    cached=json.loads(path.read_text())
                    checksum=hashlib.sha256(json.dumps(m.serial(observed)).encode()).hexdigest()
                    if cached["data_sha256"]!=checksum: raise ValueError("reference data changed")
                    return cached["reference"]
                computed=ex.references(kernels,observed,idx,partition,evaluate,claim)
                save(path,dict(data_sha256=hashlib.sha256(json.dumps(m.serial(observed)).encode()).hexdigest(),
                               reference=computed))
                return computed
            scope_root=args.output/"scopes"/f"{args.scope}-n{n}"
            scope_root.mkdir(parents=True,exist_ok=True)
            frozen=scope_root/"controls.json"
            calibration=[]
            if frozen.exists():
                controls=json.loads(frozen.read_text());chosen=controls["chosen"]
            else:
                for setting in (0,1):
                    record=dict(setting=setting,valid=True,losses={},measurements=[])
                    for partition in ("calibration","validation"):
                        observed=m.dataset(spec,partition,dataset_id=0)
                        reference=ref(partition,0,observed,False)
                        losses=[]
                        for design_id in range(2):
                            design=m.designs(spec,n,partition,design_id,dataset_id=0)
                            for point,theta in enumerate(spec["points"]):
                                pair=[]
                                for policy in POLICIES:
                                    row=ex.one(evaluate(filt(setting,policy),tf.constant([theta],tf.float32),
                                                        *design,tf.cast(observed,tf.float32)))
                                    record["valid"] &= row["valid"]
                                    err=m.errors(row,reference["rows"][point]) if row["valid"] else None
                                    record["measurements"].append(dict(partition=partition,design=design_id,
                                        point=point,policy=policy,result=row,error=err))
                                    pair.append(sum(err.values()) if err else 1e100)
                                losses.append(max(pair))
                        record["valid"] &= reference["valid"]
                        record["losses"][partition]=statistics.fmean(losses)
                    calibration.append(record)
                    save(directory/"calibration.json",calibration)
                    print(args.scope,n,"control",setting,"valid",record["valid"],flush=True)
                viable=[r for r in calibration if r["valid"]]
                if not viable:
                    result=dict(status="COMPLETE",outcome="NO_VIABLE_CONTROLS",calibration=calibration,
                                candidate_rejected=True,scope=args.scope,particles=n)
                    save(directory/"result.json",result)
                    manifest.update(status="COMPLETE",wall_seconds=time.monotonic()-started,
                                    allocator=tf.config.experimental.get_memory_info("GPU:0"))
                    write(directory/"manifest.json",manifest);return
                chosen=min(viable,key=lambda r:r["losses"]["calibration"])["setting"]
                controls=dict(chosen=chosen,controls=m.controls(chosen),scope=args.scope,particles=n,
                    dimensions={k:spec[k] for k in ("d","m","p","t")},dtype="float32",tf32=True,
                    full_correction_observability=True,calibration=calibration,
                    status="frozen_diagnostic_comparison_not_canonical_admission",source_sha256=manifest["source_sha256"])
                save(frozen,controls)
            if args.worker=="pilot":
                # Measure fresh warm calls explicitly; never censor slow timings.
                observed=m.dataset(spec,"validation",dataset_id=0)
                design=m.designs(spec,n,"validation",0,dataset_id=0)
                timing={}
                for policy in POLICIES:
                    calls=[]
                    for _ in range(3):
                        row=ex.one(evaluate(filt(chosen,policy),
                            tf.constant([spec["points"][0]],tf.float32),*design,tf.cast(observed,tf.float32)))
                        calls.append(row["seconds"])
                    timing[policy]=calls
                per_pair=sum(statistics.median(v) for v in timing.values())
                result=dict(status="COMPLETE",scope=args.scope,particles=n,controls=str(frozen),
                    warm_call_seconds=timing,
                    four_dataset_minimum_candidate_seconds=4*32*len(spec["points"])*per_pair,
                    qualification="excludes compilation, reference refinement, other baselines and extensions to 8/16 datasets")

            else:
                datasets=[]
                bootstrap=kernels.get("bootstrap", n, dtype=tf.float32)
                prior=kernels.get("prior", n, dtype=tf.float32)
                gaussian=kernels.get("gaussian", dtype=tf.float32)
                for target in (4,8,16):
                    for idx in range(len(datasets),target):
                        path=scope_root/f"dataset-{idx:02d}.json"
                        if path.exists():
                            dataset=json.loads(path.read_text())
                            if dataset.get("complete"): datasets.append(dataset);continue
                        else:
                            observed=m.dataset(spec,"claim",dataset_id=idx)
                            reference=ref("claim",idx,observed,True)
                            dataset=dict(dataset_id=idx,observations=m.serial(observed),
                                data_sha256=hashlib.sha256(json.dumps(m.serial(observed)).encode()).hexdigest(),
                                truth=spec["truth"],reference=reference,
                                points=[dict(theta=p,rows=[]) for p in spec["points"]],complete=False)
                            save(path,dataset)
                        observed=tf.constant(dataset["observations"],tf.float64)
                        if not dataset["reference"]["valid"]: raise ValueError("reference refinement invalid")
                        observed32=tf.cast(observed,tf.float32)
                        for replicate in range(32):
                            design=m.designs(spec,n,"claim",replicate,dataset_id=idx)
                            for point,theta in enumerate(spec["points"]):
                                record=dataset["points"][point]
                                existing=len(record["rows"])>replicate
                                row=record["rows"][replicate] if existing else dict(replicate=replicate)
                                if row.get("comparator_precision")=="float32_tf32": continue
                                preserved_candidate=all(policy in row for policy in POLICIES)
                                for policy in POLICIES:
                                    if policy not in row:
                                        row[policy]=ex.one(evaluate(filt(chosen,policy),tf.constant([theta],tf.float32),
                                                                    *design,observed32))
                                param=tf.constant(theta,tf.float32)
                                seed=tf.constant([20261006+7919*(idx+1), 300000+1000*point+replicate],tf.int32)
                                row["bootstrap"]=evaluate(bootstrap,param,observed32,seed)
                                row["prior"]=evaluate(prior,param,observed32,seed+[0,500000])
                                row["gaussian"]=evaluate(gaussian,param,observed32,seed)
                                # The first invocation includes graph compilation and is
                                # reported as cold cost. Warm measurement fixes the budget.
                                if replicate==0 or preserved_candidate:
                                    warm=ex.one(evaluate(filt(chosen,"marginal_mixture"),
                                                        tf.constant([theta],tf.float32),*design,tf.cast(observed,tf.float32)))
                                    budget=warm["seconds"];row["candidate_warm_seconds"]=budget
                                    bpf_warm=evaluate(bootstrap,param,observed32,seed+[0,600000])["seconds"]
                                else:
                                    budget=row["marginal_mixture"]["seconds"]
                                    bpf_warm=row["bootstrap"]["seconds"]
                                # Fix replication count from a separate timing call,
                                # before drawing the likelihood estimates to combine.
                                requested=max(1,int(budget/max(bpf_warm,1e-9)))
                                count=min(256,requested);values=[];elapsed=0.
                                for repeat in range(count):
                                    b=evaluate(bootstrap,param,observed32,seed+[0,700000+10000*repeat])
                                    values.append(b["value"]);elapsed+=b["seconds"]
                                maximum=max(values)
                                combined=maximum+math.log(sum(math.exp(v-maximum) for v in values)/len(values))
                                ratio=elapsed/budget
                                row["bootstrap_equal_runtime"]=dict(value=combined,seconds=elapsed,repeats=len(values),
                                    requested_repeats=requested,target_seconds=budget,measured_cost_ratio=ratio,
                                    matched_budget=(requested<=256 and .8<=ratio<=1.2),
                                    aggregation="log of arithmetic mean likelihood, independent random designs")
                                row["comparator_precision"]="float32_tf32"
                                row["runtime_comparison_scope"]="likelihood accuracy; LEDH includes score, PF is value-only"
                                if existing:
                                    record["rows"][replicate]=row
                                else:
                                    record["rows"].append(row)
                                save(path,dataset)
                        dataset["complete"]=True;save(path,dataset);datasets.append(dataset)
                        print(args.scope,n,"dataset",idx,"complete",flush=True)
                    summaries=ex.summary(datasets,len(spec["points"]))
                    save(scope_root/f"look-{target}.json",dict(summaries=summaries))
                    unresolved=any(any(m["verdict"]=="INCONCLUSIVE" for m in r.get("metrics",{}).values()) for r in summaries)
                    if not unresolved: break
                result=dict(status="COMPLETE",scope=args.scope,particles=n,datasets=len(datasets),
                    summaries=summaries,controls=str(frozen),scope_root=str(scope_root),
                    default_ready=False,hmc_claim=False)

        save(directory/"result.json",result)
        manifest.update(status="COMPLETE",wall_seconds=time.monotonic()-started,
                        allocator=tf.config.experimental.get_memory_info("GPU:0"))
        write(directory/"manifest.json",manifest)
        print(args.worker,args.scope,"COMPLETE",round(manifest["wall_seconds"],2),flush=True)
    except Exception as exc:
        manifest.update(status="FAILED",wall_seconds=time.monotonic()-started,error=str(exc))
        write(directory/"manifest.json",manifest)
        traceback.print_exc()
        raise


def master(args):
    config=configuration(args);source=identity()
    args.output.mkdir(parents=True,exist_ok=True)
    campaign_path=args.output/"campaign.json"
    if campaign_path.exists():
        campaign=json.loads(campaign_path.read_text())
        if campaign["configuration"]!=config or campaign["source_sha256"]!=source:
            raise ValueError("resume configuration/source changed; use a new output root")
    else:
        campaign=dict(configuration=config,source_sha256=source,status="RUNNING",attempts=[],
                      created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
        write(campaign_path,campaign)
    jobs=[("engineering",None,32),("precision","m13-t20",4096),("pilot","lg20-t240",1024)]
    jobs += [("scope",scope,n) for n in config["particles"] for scope in config["scopes"]]
    all_jobs=list(jobs)
    if args.stage!="all": jobs=[j for j in jobs if j[0]==args.stage]
    for kind,scope,n in jobs:
        if identity()!=source:
            campaign["status"]="SOURCE_CHANGED_REPAIR_REQUIRED";write(campaign_path,campaign)
            raise RuntimeError("source changed while the master was running; preserve state and migrate")
        key=f"{kind}-{scope or 'all'}-n{n}"
        prior=[a for a in campaign["attempts"] if a["key"]==key]
        if any(a["status"]=="COMPLETE" for a in prior): continue
        if len(prior)>=3: continue  # initial attempt plus two bounded repairs
        consumed=campaign.get("prior_consumed_seconds",0.)+sum(a.get("wall_seconds",0.) for a in campaign["attempts"])
        remaining=config["budget_seconds"]-consumed
        if remaining<=0: break
        attempt=args.output/"attempts"/f"{len(campaign['attempts']):03d}-{key}"
        attempt.mkdir(parents=True,exist_ok=False)
        timeout=min(remaining,config["worker_scope_cap_seconds"] if kind=="scope" else config["worker_check_cap_seconds"])
        command=[sys.executable,str(Path(__file__).resolve()),"--output",str(args.output),
                 "--worker",kind,"--attempt",str(attempt),"--n",str(n),"--gpu",args.gpu,
                 "--deadline",str(time.time()+timeout),"--budget-seconds",str(config["budget_seconds"])]
        if scope: command += ["--scope",scope]
        record=dict(key=key,kind=kind,scope=scope,particles=n,attempt=str(attempt),
                    status="RUNNING",command=command,started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
        campaign["attempts"].append(record);write(campaign_path,campaign)
        write(args.output/"next-phase.json",dict(current=key,remaining_seconds=remaining,
            next_jobs=[dict(kind=k,scope=s,n=count) for k,s,count in jobs[jobs.index((kind,scope,n))+1:]],
            reason="part of the authorized expanded master",plan=PLAN))
        tick=time.monotonic()
        with (attempt/"command.log").open("w") as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try: code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                code=124
        record.update(wall_seconds=time.monotonic()-tick,exit_code=code,
                      status="COMPLETE" if code==0 else "FAILED")
        result_path=attempt/"result.json"
        if result_path.exists(): record["result"]=str(result_path)
        write(campaign_path,campaign)
        print(key,record["status"],round(record["wall_seconds"],2),flush=True)
        if kind=="engineering" and (code!=0 or not json.loads(result_path.read_text()).get("passed",False)):
            campaign["status"]="ENGINEERING_REPAIR_REQUIRED";write(campaign_path,campaign);return
        if kind=="precision" and (code!=0 or not json.loads(result_path.read_text()).get("passed",False)):
            campaign["status"]="PRECISION_REPAIR_REQUIRED";write(campaign_path,campaign);return
        # Scientific candidate rejection does not invalidate other scopes.
        # Infrastructure failures remain explicit and can resume locally.
    done={a["key"] for a in campaign["attempts"] if a["status"]=="COMPLETE"}
    missing=[f"{k}-{s or 'all'}-n{n}" for k,s,n in all_jobs if f"{k}-{s or 'all'}-n{n}" not in done]
    campaign["status"]="DEFINED_GRID_COMPLETE_DIMENSION_EXTENSION_PENDING" if not missing else "INCOMPLETE"
    campaign["selected_stage"]=args.stage
    campaign["unfinished_jobs"]=missing
    write(campaign_path,campaign)
    write(args.output/"next-phase.json",dict(status=campaign["status"],unfinished_jobs=missing,
        remaining_seconds=config["budget_seconds"]-campaign.get("prior_consumed_seconds",0.)-sum(a.get("wall_seconds",0.) for a in campaign["attempts"]),
        reason="continue the same expanded master; no new scientific direction",plan=PLAN))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--budget-seconds",type=float,default=86400)
    p.add_argument("--gpu",default=GPU)
    p.add_argument("--scopes")
    p.add_argument("--particle-ladder",default="1024,4096,16384")
    p.add_argument("--stage",choices=("all","engineering","precision","pilot","scope"),default="all")
    p.add_argument("--worker",choices=("engineering","precision","pilot","scope"))
    p.add_argument("--attempt",type=Path)
    p.add_argument("--scope")
    p.add_argument("--n",type=int,default=1024)
    p.add_argument("--deadline",type=float,default=float("inf"))
    args=p.parse_args();args.output=args.output.resolve()
    if args.worker:
        if args.attempt is None: p.error("--worker requires --attempt")
        worker(args)
    else: master(args)

if __name__=="__main__": main()

