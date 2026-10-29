#!/usr/bin/env python3
"""Bounded execution and phase refresh for the five-model shared-cost repair.

Unit fixtures are explicitly CPU-only diagnostics. Claim-scope engineering
comparisons use N4032, FP32 without TF32, GPU/XLA, and frozen paired inputs.
No short comparison is posterior or scope-tuning admission.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PLAN=ROOT/"docs/plans/surrogate-hmc-five-model-performance-repair-master-2026-10-11.md"
MODELS=("lgssm","ksc","predator_prey","sir_d18","range_bearing")
SOURCES=("analytical_jet_tf.py","covariance_proposal_score_tf.py", "covariance_proposal_score_batch_tf.py","covariance_proposal_score_moments_tf.py","covariance_proposal_tf.py","covariance_proposal_transport_tf.py","covariance_proposal_moments_tf.py","ledh_marginal_weights_tf.py")


def save(path,value):
    path=Path(path)
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def worker(args):
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    result=dict(status="STARTING",model=args.model,plan=str(PLAN),command=[sys.executable,*sys.argv],
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        started_utc=datetime.now(timezone.utc).isoformat(),python=sys.executable,cpu_only=args.unit,
        evidence_role="implementation_parity_and_descriptive_timing",seed=261008301,
        source_sha256={name:hashlib.sha256((ROOT/"bayesfilter/highdim"/name).read_bytes()).hexdigest() for name in SOURCES})
    source_paths=[ROOT/"bayesfilter/highdim"/name for name in SOURCES]
    source_paths += [Path(__file__).resolve(),ROOT/"bayesfilter/testing/surrogate_hmc_five_model.py"]
    for source in source_paths:
        snapshot=args.output/"source"/source.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,snapshot)
    result["source_snapshot"]=str(args.output/"source")
    destination=args.output/"result.json"
    save(destination,result)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1" if args.unit else args.gpu
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
        os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP","0")
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS","4")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS","2")
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        result["memory_policy"]=configure_tensorflow_gpu_memory_growth(tf,require_gpu=not args.unit)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        from bayesfilter.testing import surrogate_hmc_five_model as runtime
        from bayesfilter.highdim.covariance_proposal_tf import make_filter
        from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
        case=runtime.load_case(args.model)
        scope=case.scope()
        result["source_dependency_sha256"]=scope["source_sha256"]
        for relative in scope["source_sha256"]:
            snapshot=args.output/"source"/relative
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/relative,snapshot)
        dtype=tf.float64 if args.unit else tf.float32
        if args.unit and args.model=="range_bearing":
            from scripts.run_range_bearing_covariance_comparison import specs
            _,case.spec=specs(tf,dtype,case.horizon)
        n=64 if args.unit else runtime.N
        horizon=2 if args.unit else case.horizon
        controls=replace(case.controls,flow_steps=2,reset_steps=4,correction_steps=args.correction_steps) if args.unit else case.controls
        if not args.unit:
            # These controls are a paired performance baseline, NOT a new
            # repository-issued admission artifact after a source change.
            prepared=ROOT/"docs/plans/artifacts/surrogate-hmc-five-model-n4032-20261011-06"/args.model/"filter-tuning/filter-selection.json"
            selected=json.loads(prepared.read_text())
            case.controls=runtime.ProposalControls(**selected["selected"]["controls"])
            case.beta=tf.constant(selected["selected"]["beta"],tf.float32)
            controls=case.controls
            result["baseline_selection"]=str(prepared)
        values=(tf.cast(case.theta0,dtype),tf.cast(case.observations[:horizon],dtype),tf.cast(case.fixed[0][:n],dtype),tf.cast(case.fixed[1][:horizon,:n],dtype),tf.cast(case.fixed[2][:horizon,:n],dtype),tf.cast(case.beta,dtype))
        if args.offset:
            q=tf.fill([case.p],tf.cast(args.offset,dtype))
            values=(case.chart(q)[0],*values[1:])
        result.update(particles=n,horizon=horizon,parameter_count=case.p,state_dimension=case.spec.dimension,
            observation_dimension=case.observations.shape[1],dtype=dtype.name,tf32=False,jit_compile=True,
            controls=asdict(controls),beta=runtime.json_ready(case.beta),offset=args.offset,
            data_sha256=hashlib.sha256(case.data_path.read_bytes()).hexdigest(),
            gpu_devices=[tf.config.experimental.get_device_details(d) for d in tf.config.list_physical_devices("GPU")],
            tensorflow=tf.__version__,status="COMPILING")
        save(destination,runtime.json_ready(result))
        theta_second=case.chart(tf.fill([case.p],tf.cast(args.offset+.01,dtype)))[0]
        theta_batch=tf.stack([values[0]] if args.batch_size==1 else [values[0],theta_second])
        result["batch_size"]=args.batch_size
        new=make_batched_score_filter(case.spec,n,horizon,controls,dtype,True,batch_size=args.batch_size)
        old=make_filter(case.spec,n,horizon,controls,dtype,True)
        @tf.function(input_signature=[tf.TensorSpec([case.p],dtype)],jit_compile=True,autograph=False)
        def legacy(theta):
            scores=tf.TensorArray(dtype,case.p,element_shape=[])
            valid=tf.constant(True)
            def body(k,scores,value,valid):
                out=old(theta,tf.one_hot(k,case.p,dtype=dtype),*values[1:])
                return k+1,scores.write(k,out["score"]),out["value"],valid & out["valid"]
            _,scores,value,valid=tf.while_loop(lambda k,*_:k<case.p,body,(0,scores,tf.constant(0.,dtype),valid),parallel_iterations=1)
            return dict(value=value,score=scores.stack(),valid=valid)
        @tf.function(input_signature=[tf.TensorSpec([args.batch_size,case.p],dtype)],jit_compile=True,autograph=False)
        def baseline_endpoint(theta):
            rows=tuple(legacy(theta[b]) for b in range(args.batch_size))
            return {name:tf.stack([row[name] for row in rows]) for name in ("value","score","valid")}
        def evaluate(fn,*inputs):
            begin=time.monotonic(); output=fn(*inputs)
            for tensor in tf.nest.flatten(output): tensor.numpy()
            return output,time.monotonic()-begin
        @tf.function(input_signature=[tf.TensorSpec([args.batch_size,case.p],dtype)],jit_compile=True,autograph=False)
        def candidate_endpoint(theta):
            out=new(theta,*values[1:])
            return {k:out[k] for k in ("value","score","valid","steps_completed")}
        candidate,compile_new=evaluate(candidate_endpoint,theta_batch)
        result.update(status="COMPARING",new_first_call_seconds=compile_new,
            candidate=runtime.json_ready({k:candidate[k] for k in ("value","score","valid","steps_completed")}))
        save(destination,runtime.json_ready(result))
        baseline,compile_old=evaluate(baseline_endpoint,theta_batch)
        # FP64 is the algebra check; FP32 is arithmetic parity.
        rtol,atol=(2e-7,2e-7) if args.unit else (1e-3,1e-3)
        score_error=tf.abs(candidate["score"]-baseline["score"])
        score_ok=tf.reduce_all(score_error<=atol+rtol*tf.abs(baseline["score"]))
        value_error=tf.abs(candidate["value"]-baseline["value"])
        value_ok=tf.reduce_all(value_error<=atol+rtol*tf.abs(baseline["value"]))
        parity=bool(score_ok.numpy() and value_ok.numpy() and
            tf.reduce_all(candidate["valid"]).numpy() and
            tf.reduce_all(baseline["valid"]).numpy() and
            tf.reduce_all(candidate["steps_completed"]==horizon).numpy())
        if args.unit:
            fd=[]
            for row in range(args.batch_size):
                derivatives=[]
                for component in range(case.p):
                    step=tf.constant(1e-5,dtype)*tf.maximum(tf.constant(1.,dtype),
                                                              tf.abs(theta_batch[row,component]))
                    perturbation=(tf.one_hot(row,args.batch_size,dtype=dtype)[:,None] *
                                  tf.one_hot(component,case.p,dtype=dtype)[None,:])*step
                    upper=candidate_endpoint(theta_batch+perturbation)
                    lower=candidate_endpoint(theta_batch-perturbation)
                    derivatives.append((upper["value"][row]-lower["value"][row])/(2*step))
                    parity=parity and bool(tf.reduce_all(upper["valid"])) and bool(tf.reduce_all(lower["valid"]))
                fd.append(tf.stack(derivatives))
            fd=tf.stack(fd)
            fd_error=tf.abs(fd-candidate["score"])
            fd_pass=bool(tf.reduce_all(fd_error<=2e-5+2e-5*tf.abs(fd)))
            parity=parity and fd_pass
            result.update(finite_difference_score=runtime.json_ready(fd),
                finite_difference_error=runtime.json_ready(fd_error),finite_difference_pass=fd_pass)
        result.update(old_first_call_seconds=compile_old,baseline=runtime.json_ready(baseline),
            score_absolute_error=runtime.json_ready(score_error),value_absolute_error=runtime.json_ready(value_error),
            tolerance=dict(rtol=rtol,atol=atol),parity_pass=parity,status="TIMING")
        save(destination,runtime.json_ready(result))
        timings=[]
        for iteration in range(1 if args.unit else 2):
            pair={}
            for name in (("new","old") if iteration%2==0 else ("old","new")):
                _,elapsed=evaluate(candidate_endpoint,theta_batch) if name=="new" else evaluate(baseline_endpoint,theta_batch)
                pair[name]=elapsed
            timings.append(pair)
            result["paired_warmed_seconds"]=timings
            save(destination,runtime.json_ready(result))
        concrete=new.get_concrete_function()
        graph=concrete.graph.as_graph_def()
        nodes=list(graph.node)+[node for function in graph.library.function for node in function.node_def]
        forbidden=[node.op for node in nodes if node.op in ("PyFunc","EagerPyFunc","PyFuncStateless")]
        result.update(graph=dict(xla_must_compile=bool(concrete.function_def.attr["_XlaMustCompile"].b),traces=new.experimental_get_tracing_count(),host_callbacks=forbidden),
            median_speedup=statistics.median(t["old"] for t in timings)/statistics.median(t["new"] for t in timings),
            status="PASS" if parity and not forbidden and concrete.function_def.attr["_XlaMustCompile"].b else "FAIL")
        if not args.unit:
            result["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
        return 0 if result["status"]=="PASS" else 1
    except Exception:
        result.update(status="ERROR",error=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        result["wall_seconds"]=time.monotonic()-started
        save(destination,result)


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); completed=[]
    models=(args.model,) if args.model else MODELS
    for name in models:
        remaining=args.budget_seconds-(time.monotonic()-started)
        if remaining<=0: break
        command=[sys.executable,str(Path(__file__).resolve()),"worker","--model",name,"--output",str(args.output/name),"--gpu",args.gpu]
        command += ["--batch-size",str(args.batch_size)]
        if args.unit: command += ["--unit","--correction-steps",str(args.correction_steps),"--offset",str(args.offset)]
        save(args.output/"next-phase-plan.json",dict(status="RUNNING",active=name,completed=completed,remaining_seconds=remaining,next="paired analytical score and warmed runtime; HMC remains separate"))
        try:
            with (args.output/(name+".log")).open("w") as log:
                child=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=min(remaining,args.per_model_timeout))
            record=dict(model=name,exit_code=child.returncode,result=str(args.output/name/"result.json"))
        except subprocess.TimeoutExpired:
            record=dict(model=name,status="TIMEOUT")
            partial=args.output/name/"result.json"
            if partial.is_file():
                state=json.loads(partial.read_text())
                state.update(last_completed_status=state.get("status"),status="TIMEOUT",
                             timeout_seconds=min(remaining,args.per_model_timeout))
                save(partial,state)
        completed.append(record)
        save(args.output/"execution.json",dict(completed=completed,wall_seconds=time.monotonic()-started,budget_seconds=args.budget_seconds))
        if record.get("exit_code")!=0: break
    passed=len(completed)==len(models) and all(r.get("exit_code")==0 for r in completed)
    save(args.output/"next-phase-plan.json",dict(status="PASS" if passed else "REPAIR_REQUIRED",completed=completed,remaining_seconds=max(0,args.budget_seconds-(time.monotonic()-started)),next="review parity, wire consumer and run fresh scope tuning" if passed else "inspect failed result and repair within remaining budget"))
    return 0 if passed else 1


def regression(args):
    """CPU-only consumer/call-chain tests, isolated from GPU imports."""
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    os.environ["TF_NUM_INTRAOP_THREADS"]="4"
    os.environ["TF_NUM_INTEROP_THREADS"]="2"
    os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"]="0"
    started=time.monotonic()
    command=[sys.executable,"-m","pytest","-q",
        "tests/test_surrogate_hmc_five_model_master.py",
        "tests/highdim/test_covariance_proposal.py",
        "tests/highdim/test_covariance_proposal_precision_guard.py"]
    with (args.output/"pytest.log").open("w") as log:
        try:
            child=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                 timeout=args.budget_seconds)
            status="PASS" if child.returncode==0 else "FAIL"
        except subprocess.TimeoutExpired:
            status="TIMEOUT"
    save(args.output/"result.json",dict(status=status,cpu_only=True,command=command,
        wall_seconds=time.monotonic()-started,log=str(args.output/"pytest.log")))
    return 0 if status=="PASS" else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("worker","run","regression"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",choices=MODELS)
    parser.add_argument("--unit",action="store_true")
    parser.add_argument("--gpu",default="0")
    parser.add_argument("--budget-seconds",type=float,default=3600.)
    parser.add_argument("--per-model-timeout",type=float,default=900.)
    parser.add_argument("--correction-steps",type=int,default=0)
    parser.add_argument("--offset",type=float,default=0.)
    parser.add_argument("--batch-size",type=int,choices=(1,2),default=2)
    args=parser.parse_args()
    if args.command=="worker" and not args.model: parser.error("worker requires --model")
    if args.budget_seconds<=0 or args.per_model_timeout<=0: parser.error("budgets must be positive")
    if args.command=="regression": return regression(args)
    return worker(args) if args.command=="worker" else run(args)


if __name__=="__main__":
    sys.exit(main())
