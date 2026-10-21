#!/usr/bin/env python3
"""Resumable, bounded diagnostic campaign for the optional marginal weights."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
PLAN="docs/plans/surrogate-hmc-marginal-weight-execution-2026-10-05.md"
PHASES=("lg1-t2","lg3-t2","lg3-t10","ksc-t10","ksc-t20","m13-t1","m13-t20")
REGRESSION_PHASES=("regression-before","regression-after","engineering")


def write(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False));temporary.replace(path)


def source_identity():
    names=["bayesfilter/highdim/ledh_canonical_score_tf.py","bayesfilter/highdim/ledh_canonical_batch_fused_tf.py",
           "bayesfilter/highdim/ledh_marginal_weights_tf.py","bayesfilter/testing/marginal_weight_campaign.py",
           "scripts/run_surrogate_hmc_marginal_weight_campaign.py"]
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def run_configuration(args):
    """Validate comparison provenance before a worker can import TensorFlow."""
    sequence=tuple(args.only.split(",")) if args.only else REGRESSION_PHASES+PHASES
    if not sequence or len(set(sequence))!=len(sequence) or any(p not in REGRESSION_PHASES+PHASES for p in sequence):
        raise ValueError("--only must contain distinct supported phase names")
    if args.particles<=0 or not math.isfinite(args.budget_seconds) or args.budget_seconds<=0:
        raise ValueError("particle count and finite elapsed budget must be positive")
    needs_baseline=args.worker=="regression-before" or (args.worker is None and "regression-before" in sequence)
    baseline=(args.baseline_source or args.output/"source_before").resolve()
    identity=None
    if needs_baseline:
        if baseline==ROOT or (baseline/"bayesfilter").resolve()==(ROOT/"bayesfilter").resolve():
            raise ValueError("baseline must be the saved pre-change package, not the current checkout")
        required=("bayesfilter/__init__.py","bayesfilter/highdim/ledh_canonical_score_tf.py",
                  "bayesfilter/highdim/ledh_canonical_batch_fused_tf.py")
        if any(not (baseline/name).is_file() for name in required):
            raise ValueError("missing pre-change baseline; provide --baseline-source PATH (no current-code fallback)")
        paths=sorted(p for p in baseline.rglob("*") if p.is_file() and p.suffix in (".py",".so"))
        identity={str(p.relative_to(baseline)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        args.baseline_source=baseline
    if "regression-after" in sequence:
        index=sequence.index("regression-after")
        if (args.worker is None and "regression-before" not in sequence[:index]
                and not (args.output/"regression-before/result.json").is_file()):
            raise ValueError("regression-after requires a preceding saved-baseline regression")
        if (args.worker == "regression-after"
                and not (args.output/"regression-before/result.json").is_file()):
            raise ValueError("direct regression-after requires regression-before/result.json")
    return dict(particles=args.particles,gpu=args.gpu,phases=list(sequence),
                baseline_source=str(baseline) if needs_baseline else None,baseline_sha256=identity)


def setup(args):
    configuration=run_configuration(args)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS","6")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS","2")
    sys.path.insert(0,str(ROOT))
    if args.worker=="regression-before":
        sys.path.insert(0,str(args.baseline_source))
        origin=Path(importlib.util.find_spec("bayesfilter").origin).resolve()
        if origin!=(args.baseline_source/"bayesfilter/__init__.py").resolve():
            raise ValueError("baseline import resolved to the wrong package")
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    policy=configure_tensorflow_gpu_memory_growth(tf)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    module_spec=importlib.util.spec_from_file_location("marginal_campaign_support",ROOT/"bayesfilter/testing/marginal_weight_campaign.py")
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    environment=dict(python=sys.executable,tensorflow=tf.__version__,gpu_selector=args.gpu,
        gpu_memory_policy=policy,tf32=True,jit_compile=True,precision="FP32; explicit FP64 independent/derivative references",
        gpu_trust="escalated",input_generation="small CPU fixed-input diagnostic exception",numpy_runtime=False,
        baseline_source=configuration["baseline_source"],baseline_sha256=configuration["baseline_sha256"])
    return tf,module,environment


def run_worker(args):
    started=time.monotonic();tf,m,environment=setup(args)
    directory=args.output/args.worker;directory.mkdir(exist_ok=True)
    manifest=dict(command=sys.argv,git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        plan=PLAN,source_sha256=source_identity(),environment=environment,started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        seed_policy="20261005 plus saved partition/horizon/dimension/replicate",output=str(directory),status="RUNNING")
    write(directory/"manifest.json",manifest)
    def check_time():
        if time.time()>args.deadline:raise TimeoutError("campaign deadline reached")
    def evaluate(function,theta,design,observed):
        check_time();tick=time.monotonic();result=m.serial(function(theta,*design,observed))
        result["seconds"]=time.monotonic()-tick
        result["finite_value_score"] = all(math.isfinite(v) for v in [result["value"]]+result["score"])
        result["valid"] = result["valid"] and result["finite_value_score"]
        return result
    try:
        if args.worker.startswith("regression-"):
            rows=[]
            for name in ("lg1-t2","lg3-t2","ksc-t10"):
                spec=m.specification(name,tf.float64);observed=m.dataset(spec,"engineering")
                design=m.designs(spec,16,"engineering",0,tf.float64)
                function=m.compile_filter(spec,16,m.controls(0),None,tf.float64)
                rows.append(dict(name=name,result=evaluate(function,tf.constant(spec["truth"],tf.float64),design,observed)))
            result=dict(status="COMPLETE",rows=rows)
        elif args.worker=="engineering":
            rows=[]
            for name in PHASES:
                spec=m.specification(name,tf.float64);observed=m.dataset(spec,"engineering")
                design=m.designs(spec,16,"engineering",0,tf.float64)
                theta=tf.constant(spec["points"][1],tf.float64)
                for policy in ("ancestor","marginal_mixture"):
                    function=m.compile_filter(spec,16,m.controls(0),policy,tf.float64)
                    central=evaluate(function,theta,design,observed)
                    checks=[]
                    for h in (1e-4,3e-5):
                        for j in range(spec["p"]):
                            e=tf.one_hot(j,spec["p"],dtype=tf.float64)*h
                            plus=evaluate(function,theta+e,design,observed)
                            minus=evaluate(function,theta-e,design,observed)
                            fd=(plus["value"]-minus["value"])/(2*h)
                            error=abs(fd-central["score"][j])/(1+abs(central["score"][j]))
                            checks.append(dict(h=h,j=j,fd=fd,analytical=central["score"][j],relative_error=error,
                                valid=plus["valid"] and minus["valid"]))
                    passed=central["valid"] and all(x["valid"] and x["relative_error"]<3e-4 for x in checks)
                    rows.append(dict(name=name,policy=policy,central=central,checks=checks,passed=passed))
                    m.save(directory/"progress.json",rows)
                    print(name,policy,"FD",passed,flush=True)
            result=dict(status="COMPLETE",passed=all(row["passed"] for row in rows),rows=rows)
        else:
            name=args.worker;spec=m.specification(name);n=args.particles
            observations={part:m.dataset(spec,part) for part in ("calibration","validation","claim")}
            m.save(directory/"data.json",dict(observations=observations,points=spec["points"],truth=spec["truth"],n=n,
                dimensions={k:spec[k] for k in ("t","d","m","p")},partitions=m.PARTITIONS))
            reference={};reference_checks={}
            for part in observations:
                reference[part],reference_checks[part]=m.references(spec,observations[part],claim=part=="claim")
                m.save(directory/"references.json",dict(values=reference,checks=reference_checks))
                if not reference_checks[part]["valid"]:raise ValueError("reference refinement failed: "+part)
            compiled={};calibration=[]
            for setting in (0,1):
                for policy in ("ancestor","marginal_mixture"):
                    compiled[setting,policy]=m.compile_filter(spec,n,m.controls(setting),policy)
                rows=[]
                for part in ("calibration","validation"):
                    for replicate in range(4):
                        design=m.designs(spec,n,part,replicate)
                        for point,theta in enumerate(spec["points"]):
                            row=dict(partition=part,replicate=replicate,point=point)
                            for policy in ("ancestor","marginal_mixture"):
                                row[policy]=evaluate(compiled[setting,policy],tf.constant(theta,tf.float32),design,tf.cast(observations[part],tf.float32))
                            rows.append(row)
                    print(name,"scope",setting,part,"complete",flush=True)
                losses={}
                valid=all(row[p]["valid"] for row in rows for p in ("ancestor","marginal_mixture"))
                for part in ("calibration","validation"):
                    selected=[row for row in rows if row["partition"]==part]
                    part_valid=all(row[p]["valid"] for row in selected for p in ("ancestor","marginal_mixture"))
                    loss=([max(sum(m.errors(row[p],reference[part][row["point"]]).values())
                               for p in ("ancestor","marginal_mixture")) for row in selected] if part_valid else None)
                    losses[part]=sum(loss)/len(loss) if loss is not None else None
                calibration.append(dict(setting=setting,valid=valid,losses=losses,rows=rows))
                m.save(directory/"calibration.json",calibration)
            viable=[r for r in calibration if r["valid"]]
            if not viable:
                # Candidate rejection is evidence; continue to the next model.
                result=dict(status="COMPLETE",outcome="NO_NUMERICALLY_VIABLE_SCOPE_CONTROLS",
                    noninferiority_established=False,references=reference["claim"],reference_checks=reference_checks["claim"],
                    calibration=calibration,summaries=[],heuristic=[],claim_rows=[],
                    inference_scope="candidate numerically rejected, no accuracy or HMC conclusion")
                m.save(directory/"result.json",result)
                manifest.update(status="COMPLETE",wall_seconds=time.monotonic()-started,
                    allocator=tf.config.experimental.get_memory_info("GPU:0"))
                write(directory/"manifest.json",manifest)
                print(args.worker,"COMPLETE with candidate rejection",flush=True)
                return
            chosen=min(viable,key=lambda r:r["losses"]["calibration"])["setting"]
            scope=dict(model=name,n=n,dimensions={k:spec[k] for k in ("t","d","m","p")},dtype="float32",tf32=True,
                controls=m.controls(chosen),chosen=chosen,selection="minimize paired worst-method mean error on calibration only",
                validation="disjoint data, finite full-consumer outputs; diagnostic scope only",
                status="frozen_controlled_comparison_not_canonical_tuning_admission",source_sha256=source_identity())
            m.save(directory/"scope-controls.json",scope)
            # Untouched designs/data are consumed for the first time after freezing controls.
            claim_rows=[[] for _ in spec["points"]];looks=[]
            bootstrap=m.bootstrap_reference(spec,observations["claim"],n)
            counts=(32,64,96) if not name.startswith("m13") else (32,)
            completed=0
            for target_count in counts:
                for replicate in range(completed,target_count):
                    design=m.designs(spec,n,"claim",replicate)
                    for point,theta in enumerate(spec["points"]):
                        row=dict(replicate=replicate,point=point)
                        for policy in ("ancestor","marginal_mixture"):
                            row[policy]=evaluate(compiled[chosen,policy],tf.constant(theta,tf.float32),design,tf.cast(observations["claim"],tf.float32))
                        row["bootstrap_value"]=float(bootstrap(tf.constant(theta,tf.float64),tf.constant([20261008,replicate+point*1000],tf.int32)))
                        claim_rows[point].append(row)
                    if (replicate+1)%8==0:
                        m.save(directory/"claim-progress.json",claim_rows)
                        print(name,"retained paired designs",replicate+1,flush=True)
                completed=target_count
                summaries=[m.paired_summary(rows,ref) for rows,ref in zip(claim_rows,reference["claim"])]
                looks.append(dict(n=completed,summaries=summaries));m.save(directory/"looks.json",looks)
                if all(s["verdict"]=="NONINFERIOR" for s in summaries):break
                # Clear harm is evidence, not a reason to spend more seeds on that cell.
                unresolved=any(s["valid"] and any(x["verdict"]=="INCONCLUSIVE" for x in s["metrics"].values()) for s in summaries)
                if not unresolved:break
            heuristic=[]
            for point,rows in enumerate(claim_rows):
                ref=reference["claim"][point]
                bootstrap_error=sum(abs(row["bootstrap_value"]-ref["value"]) for row in rows)/len(rows)
                candidate_error=summaries[point]["metrics"].get("value",{}).get("candidate_mean")
                heuristic.append(dict(point=point,bootstrap_mean_value_error=bootstrap_error,candidate_mean_value_error=candidate_error,
                    verdict="DESCRIPTIVE_UNDERPERFORMANCE_VETO" if candidate_error is not None and candidate_error>bootstrap_error else "NO_DESCRIPTIVE_UNDERPERFORMANCE",
                    ranking_statistically_supported=False))
            result=dict(status="COMPLETE",scope=scope,references=reference["claim"],reference_checks=reference_checks["claim"],
                claim_rows=claim_rows,looks=looks,summaries=summaries,heuristic=heuristic,
                noninferiority_established=all(s["verdict"]=="NONINFERIOR" for s in summaries),
                inference_scope="integration error conditional on this saved dataset; not posterior/HMC admission")
        m.save(directory/"result.json",result)
        manifest.update(status="COMPLETE",wall_seconds=time.monotonic()-started,
            allocator=tf.config.experimental.get_memory_info("GPU:0"))
        write(directory/"manifest.json",manifest)
        print(args.worker,"COMPLETE",round(manifest["wall_seconds"],2),flush=True)
    except Exception as exc:
        manifest.update(status="FAILED",wall_seconds=time.monotonic()-started,error=str(exc))
        write(directory/"manifest.json",manifest);raise


def master(args):
    configuration=run_configuration(args)
    args.output.mkdir(parents=True,exist_ok=True)
    state_path=args.output/"campaign.json"
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(
        status="RUNNING",plan=PLAN,budget_seconds=args.budget_seconds,used_seconds=0.,attempts=[],source_sha256=source_identity(),configuration=configuration)
    if state["source_sha256"]!=source_identity():raise ValueError("source changed: record a repair and create a new attempt root before resuming")
    if state.get("configuration")!=configuration:raise ValueError("run configuration changed: create a new attempt root")
    sequence=configuration["phases"]
    for phase in sequence:
        result_path=args.output/phase/"result.json"
        if result_path.exists():
            result=json.loads(result_path.read_text())
            if result.get("status")=="COMPLETE":
                if phase=="engineering" and not result.get("passed",False):
                    state["status"]="REPAIR_REQUIRED";break
                if phase=="regression-after":
                    parity=args.output/"legacy-regression.json"
                    if not parity.exists() or not json.loads(parity.read_text()).get("passed",False):
                        state["status"]="REPAIR_REQUIRED";break
                continue
        remaining=state["budget_seconds"]-state["used_seconds"]
        if remaining<120:
            state["status"]="BUDGET_EXHAUSTED";break
        command=[sys.executable,str(Path(__file__).resolve()),"--output",str(args.output),"--worker",phase,
                 "--gpu",args.gpu,"--particles",str(args.particles),"--deadline",str(time.time()+remaining)]
        if phase=="regression-before":command.extend(("--baseline-source",str(args.baseline_source)))
        write(args.output/"next-phase.json",dict(phase=phase,status="RUNNING",remaining_seconds=remaining,command=command,
            reason="next stage of the authorized master program",plan=PLAN))
        started=time.monotonic();log=args.output/(phase+".log")
        with log.open("a") as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            write(args.output/"active-process.json",dict(pid=process.pid,phase=phase,log=str(log),command=command))
            try:code=process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                code=124
        elapsed=time.monotonic()-started;state["used_seconds"]+=elapsed
        write(args.output/"active-process.json",dict(pid=None,phase=phase,status="FINISHED",exit_code=code,log=str(log)))
        state["attempts"].append(dict(phase=phase,code=code,wall_seconds=elapsed,log=str(log)))
        write(state_path,state)
        print(phase,"exit",code,"elapsed",round(elapsed,1),"remaining",round(state["budget_seconds"]-state["used_seconds"],1),flush=True)
        if code:
            state["status"]="REPAIR_REQUIRED";break
        if phase=="regression-after":
            before=json.loads((args.output/"regression-before/result.json").read_text())["rows"]
            after=json.loads(result_path.read_text())["rows"]
            comparisons=[]
            for a,b in zip(before,after):
                diffs=[abs(a["result"]["value"]-b["result"]["value"])]+[abs(x-y) for x,y in zip(a["result"]["score"],b["result"]["score"])]
                comparisons.append(dict(name=a["name"],maximum_difference=max(diffs),valid=a["result"]["valid"] and b["result"]["valid"]))
            passed=all(x["maximum_difference"]<1e-10 and x["valid"] for x in comparisons)
            write(args.output/"legacy-regression.json",dict(passed=passed,comparisons=comparisons))
            if not passed:state["status"]="REPAIR_REQUIRED";break
        if phase=="engineering" and not json.loads(result_path.read_text())["passed"]:
            state["status"]="REPAIR_REQUIRED";break
    else:state["status"]="COMPLETE"
    write(state_path,state)
    write(args.output/"next-phase.json",dict(status=state["status"],used_seconds=state["used_seconds"],
        remaining_seconds=state["budget_seconds"]-state["used_seconds"],next_action="assemble and review results" if state["status"]=="COMPLETE" else "inspect the recorded implementation/reference failure and repair within budget"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--baseline-source",type=Path,help="saved pre-change source tree; required for regression-before unless OUTPUT/source_before exists")
    parser.add_argument("--worker",choices=("regression-before","regression-after","engineering")+PHASES)
    parser.add_argument("--budget-seconds",type=float,default=21600)
    parser.add_argument("--deadline",type=float,default=float("inf"))
    parser.add_argument("--particles",type=int,default=256)
    parser.add_argument("--gpu",default="GPU-68251639-fe82-8f81-3ccc-2953c32e805b")
    parser.add_argument("--only")
    args=parser.parse_args();args.output=args.output.resolve()
    if args.worker:run_worker(args)
    else:
        master(args)
        if json.loads((args.output/"campaign.json").read_text())["status"]!="COMPLETE":
            raise SystemExit(1)

if __name__=="__main__":main()
