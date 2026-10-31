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
import math
import resource
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
    def clean(v):
        if isinstance(v,float) and not math.isfinite(v): return str(v)
        if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [clean(x) for x in v]
        return v
    temporary.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+"\n")
    temporary.replace(path)



def summarize_profile(profile_dir):
    """Diagnostic-only aggregation of recorded GPU kernel durations."""
    from tensorflow.core.profiler.protobuf import xplane_pb2
    files=sorted(Path(profile_dir).rglob("*.xplane.pb"))
    planes=[]
    for path in files:
        space=xplane_pb2.XSpace()
        space.ParseFromString(path.read_bytes())
        for plane in space.planes:
            if "/device:GPU:" not in plane.name:
                continue
            names={}
            for line in plane.lines:
                # A kernel is on a CUDA stream. Ignore derived summary lines,
                # which can duplicate the same interval.
                if "stream" not in line.name.lower():
                    continue
                for event in line.events:
                    name=plane.event_metadata[event.metadata_id].name
                    row=names.setdefault(name,dict(name=name,count=0,seconds=0.))
                    row["count"]+=1
                    row["seconds"]+=event.duration_ps/1e12
            ordered=sorted(names.values(),key=lambda row:row["seconds"],reverse=True)
            planes.append(dict(name=plane.name,stream_kernel_count=sum(r["count"] for r in ordered),
                summed_stream_seconds=sum(r["seconds"] for r in ordered),top_kernels=ordered[:30]))
    return dict(status="RECORDED" if any(p["stream_kernel_count"] for p in planes) else "UNAVAILABLE_NO_DEVICE_EVENTS",
        files=[str(path) for path in files],gpu_planes=planes,
        role="descriptive summed stream durations; overlap can double-count wall time")



def assess_finite_difference(checks, machine_epsilon):
    """Materiality screen, conditional on a convergent central-difference ladder."""
    ordered=sorted(checks,key=lambda row:row["width"],reverse=True)
    if len(ordered)<3 or len({row["width"] for row in ordered})!=len(ordered):
        return dict(status="UNRESOLVED_WIDTH_CONVERGENCE",reason="need at least three distinct widths")
    a,b,c=ordered[-3:]
    if not all(math.isfinite(float(row[key])) for row in (a,b,c)
               for key in ("finite_difference","analytical_score")):
        return dict(status="INVALID_NONFINITE")
    def extrapolate(left,right):
        ratio=(left["width"]/right["width"])**2
        return (ratio*right["finite_difference"]-left["finite_difference"])/(ratio-1.)
    early=extrapolate(a,b); estimate=extrapolate(b,c)
    uncertainty=max(abs(estimate-early),abs(estimate-c["finite_difference"]))
    gap=abs(estimate-c["analytical_score"])
    scale=max(1.,abs(estimate),abs(c["analytical_score"]))
    tolerance=.01*scale
    roundoff=machine_epsilon*max(abs(v) for v in c["value_bracket"])/c["width"]
    resolved=uncertainty<=tolerance and roundoff<=tolerance
    return dict(status=("AGREES_WITHIN_MATERIALITY" if gap+uncertainty<=tolerance else
                        "SCORE_MISMATCH") if resolved else "UNRESOLVED_WIDTH_CONVERGENCE",
        extrapolated_reference=estimate,width_uncertainty=uncertainty,
        absolute_score_gap=gap,materiality_tolerance=tolerance,
        value_roundoff_indicator=roundoff,
        assumption="local smoothness and central-difference convergence; diagnostic, not HMC admission")


def worker(args):
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    result=dict(status="STARTING",model=args.model,plan=str(PLAN),command=[sys.executable,*sys.argv],
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        started_utc=datetime.now(timezone.utc).isoformat(),python=sys.executable,cpu_only=args.unit,
        evidence_role=("prefix_localization_reference" if args.debug_horizon else
                       "full_scope_precision_reference" if args.fp64_reference else
                       "candidate_score_diagnostic" if args.score_only else
                       "local_finite_value_derivative_diagnostic" if args.fd_sir else
                       "implementation_parity_and_descriptive_timing"),seed=261008301,
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
        from bayesfilter.highdim.covariance_proposal_tf import make_filter, TRACE_FIELDS
        from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
        case=runtime.load_case(args.model)
        scope=case.scope()
        result["source_dependency_sha256"]=scope["source_sha256"]
        for relative in scope["source_sha256"]:
            snapshot=args.output/"source"/relative
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/relative,snapshot)
        dtype=tf.float64 if args.unit or args.fp64_reference else tf.float32
        if dtype==tf.float64 and args.model=="range_bearing":
            from scripts.run_range_bearing_covariance_comparison import specs
            _,case.spec=specs(tf,dtype,case.horizon)
        n=64 if args.unit else runtime.N
        horizon=args.unit_horizon if args.unit else (args.debug_horizon or case.horizon)
        if horizon>case.horizon: raise ValueError("diagnostic horizon exceeds saved data")
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
            q=tf.fill([case.p],tf.cast(args.offset,tf.float32))
            values=(tf.cast(case.chart(q)[0],dtype),*values[1:])
        result.update(particles=n,horizon=horizon,parameter_count=case.p,state_dimension=case.spec.dimension,
            observation_dimension=case.observations.shape[1],dtype=dtype.name,tf32=False,jit_compile=True,
            controls=asdict(controls),beta=runtime.json_ready(case.beta),offset=args.offset,
            data_sha256=hashlib.sha256(case.data_path.read_bytes()).hexdigest(),
            gpu_devices=[tf.config.experimental.get_device_details(d) for d in tf.config.list_physical_devices("GPU")],
            tensorflow=tf.__version__,status="COMPILING")
        save(destination,runtime.json_ready(result))
        theta_second=tf.cast(case.chart(tf.fill([case.p],tf.cast(args.offset+.01,tf.float32)))[0],dtype)
        theta_batch=tf.stack([values[0]] if args.batch_size==1 else [values[0],theta_second])
        result["batch_size"]=args.batch_size
        result["parameter_positions"]=runtime.json_ready(theta_batch)
        new=make_batched_score_filter(case.spec,n,horizon,controls,dtype,True,batch_size=args.batch_size)
        old=make_filter(case.spec,n,horizon,controls,dtype,True)
        @tf.function(input_signature=[tf.TensorSpec([case.p],dtype)],jit_compile=True,autograph=False)
        def legacy(theta):
            scores=tf.TensorArray(dtype,case.p,element_shape=[])
            traces=tf.TensorArray(dtype,case.p,element_shape=[horizon])
            valid=tf.constant(True)
            def body(k,scores,traces,value,valid):
                out=old(theta,tf.one_hot(k,case.p,dtype=dtype),*values[1:])
                return k+1,scores.write(k,out["score"]),traces.write(k,out["trace"][:,1]),out["value"],valid & out["valid"]
            _,scores,traces,value,valid=tf.while_loop(lambda k,*_:k<case.p,body,(0,scores,traces,tf.constant(0.,dtype),valid),parallel_iterations=1)
            return dict(value=value,score=scores.stack(),score_trace=tf.transpose(traces.stack()),valid=valid)
        @tf.function(input_signature=[tf.TensorSpec([args.batch_size,case.p],dtype)],jit_compile=True,autograph=False)
        def baseline_endpoint(theta):
            rows=tuple(legacy(theta[b]) for b in range(args.batch_size))
            return {name:tf.stack([row[name] for row in rows]) for name in ("value","score","score_trace","valid")}
        def evaluate(fn,*inputs):
            begin=time.monotonic(); output=fn(*inputs)
            for tensor in tf.nest.flatten(output): tensor.numpy()
            return output,time.monotonic()-begin
        @tf.function(input_signature=[tf.TensorSpec([args.batch_size,case.p],dtype)],jit_compile=True,autograph=False)
        def candidate_endpoint(theta):
            out=new(theta,*values[1:])
            return {k:out[k] for k in ("value","score","score_trace","trace","valid","steps_completed","particle_tangent")}
        candidate,compile_new=evaluate(candidate_endpoint,theta_batch)
        result.update(status="COMPARING",new_first_call_seconds=compile_new,trace_fields=list(TRACE_FIELDS),
            candidate=runtime.json_ready({k:candidate[k] for k in ("value","score","score_trace","trace","valid","steps_completed")}))
        result["final_particle_tangent_max_abs"]=runtime.json_ready(
            tf.reduce_max(tf.abs(candidate["particle_tangent"]),axis=[2,3]))
        concrete=new.get_concrete_function()
        graph=concrete.graph.as_graph_def()
        nodes=list(graph.node)+[node for function in graph.library.function for node in function.node_def]
        forbidden=[node.op for node in nodes if node.op in ("PyFunc","EagerPyFunc","PyFuncStateless")]
        result["graph"]=dict(xla_must_compile=bool(concrete.function_def.attr["_XlaMustCompile"].b),
                            traces=new.experimental_get_tracing_count(),host_callbacks=forbidden)
        save(destination,runtime.json_ready(result))
        if forbidden or not result["graph"]["xla_must_compile"]:
            result.update(status="FAIL_EXECUTION_CONTRACT",performance_status="NOT_MEASURED")
            return 1
        if args.score_only:
            warmed=[]
            for repeat in range(args.paired_repeats):
                repeated,elapsed=evaluate(candidate_endpoint,theta_batch)
                warmed.append(elapsed)
                if not bool(tf.reduce_all(repeated["valid"]).numpy()):
                    raise RuntimeError("invalid warmed candidate")
                result["warmed_score_seconds"]=warmed
                save(destination,runtime.json_ready(result))
            if args.profile:
                profile_dir=args.output/"profile"
                options=tf.profiler.experimental.ProfilerOptions(
                    host_tracer_level=1,python_tracer_level=0,device_tracer_level=1)
                tf.profiler.experimental.start(str(profile_dir),options=options)
                try:
                    profiled,profile_seconds=evaluate(candidate_endpoint,theta_batch)
                finally:
                    tf.profiler.experimental.stop()
                if not bool(tf.reduce_all(profiled["valid"]).numpy()):
                    raise RuntimeError("invalid profiled candidate")
                result["profile"]=dict(path=str(profile_dir),wall_seconds=profile_seconds,
                    role="warmed kernel localization; tracing overhead excluded from timings")
                # Compiler IR is explanatory evidence, not a performance verdict.
                ir=new.experimental_get_compiler_ir(theta_batch,*values[1:])(stage="optimized_hlo")
                (args.output/"optimized_hlo.txt").write_text(ir)
                result["optimized_hlo"]=str(args.output/"optimized_hlo.txt")
                profile_summary=summarize_profile(profile_dir)
                save(args.output/"profile-summary.json",profile_summary)
                result["profile"]["summary"]=str(args.output/"profile-summary.json")
                save(destination,runtime.json_ready(result))
            healthy=bool(tf.reduce_all(candidate["valid"]).numpy()
                and tf.reduce_all(candidate["steps_completed"]==horizon).numpy()
                and tf.reduce_all(tf.math.is_finite(candidate["score"])).numpy()
                and tf.reduce_all(tf.math.is_finite(candidate["value"])).numpy())
            result.update(status="DIAGNOSTIC_FINISHED" if healthy else "INVALID_CANDIDATE_SCORE",
                scientific_status="PRECISION_DIAGNOSTIC_NO_HMC_PROMOTION",
                baseline_healthy=healthy,performance_status="NOT_MEASURED")
            return 0 if healthy else 1
        if args.fd_sir:
            direction=tf.one_hot(0,case.p,dtype=dtype)
            checks=[]
            for width in args.fd_widths:
                delta=tf.cast(width,dtype)*direction
                bracket=tf.stack([values[0]+delta,values[0]-delta])
                output,elapsed=evaluate(candidate_endpoint,bracket)
                fd=(output["value"][0]-output["value"][1])/(2*tf.cast(width,dtype))
                increments=(output["trace"][0,:,0]-output["trace"][1,:,0])/(2*tf.cast(width,dtype))
                checks.append(dict(width=width,finite_difference=float(fd.numpy()),
                    analytical_score=float(candidate["score"][0,0].numpy()),
                    absolute_error=float(tf.abs(fd-candidate["score"][0,0]).numpy()),
                    finite_difference_increments=runtime.json_ready(increments),
                    analytical_score_increments=runtime.json_ready(candidate["score_trace"][0,:,0]),
                    value_bracket=runtime.json_ready(output["value"]),
                    valid=runtime.json_ready(output["valid"]),
                    completed=runtime.json_ready(output["steps_completed"]),seconds=elapsed))
                result.update(status="DIAGNOSTIC_RUNNING",finite_difference=checks)
                save(destination,runtime.json_ready(result))
            baseline_healthy=bool(tf.reduce_all(candidate["valid"]).numpy()
                and tf.reduce_all(candidate["steps_completed"]==horizon).numpy()
                and tf.reduce_all(tf.math.is_finite(candidate["score"])).numpy()
                and tf.reduce_all(tf.math.is_finite(candidate["value"])).numpy())
            healthy=baseline_healthy and all(
                all(v for v in check["valid"]) and all(v==horizon for v in check["completed"])
                for check in checks)
            result["baseline_healthy"]=baseline_healthy
            result["finite_difference_width_spread"]=max(c["finite_difference"] for c in checks)-min(c["finite_difference"] for c in checks)
            result["minimum_score_gap"]=min(check["absolute_error"] for check in checks)
            result["derivative_assessment"]=assess_finite_difference(checks,
                2.**(-52 if dtype==tf.float64 else -23))
            result.update(status="DIAGNOSTIC_FINISHED" if healthy else "INVALID_FINITE_DIFFERENCE",
                scientific_status="DIAGNOSTIC_ONLY_NO_HMC_PROMOTION",
                performance_status="NOT_MEASURED",finite_difference=checks)
            return 0 if healthy else 1
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
        result["score_trace_absolute_error"]=runtime.json_ready(tf.abs(candidate["score_trace"]-baseline["score_trace"]))
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
                    parity=parity and all(bool(tf.reduce_all(out["valid"]).numpy()
                        and tf.reduce_all(out["steps_completed"]==horizon).numpy()
                        and tf.reduce_all(tf.math.is_finite(out["value"])).numpy())
                        for out in (upper,lower))
                fd.append(tf.stack(derivatives))
            fd=tf.stack(fd)
            fd_error=tf.abs(fd-candidate["score"])
            fd_pass=bool(tf.reduce_all(fd_error<=2e-5+2e-5*tf.abs(fd)))
            if args.fd_ladder:
                ladder=[]
                for width in (1e-5,3e-6,1e-6):
                    width_tensor=tf.constant(width,dtype)
                    row_results=[]
                    for row in range(args.batch_size):
                        directions=[]
                        for component in range(case.p):
                            step=width_tensor*tf.maximum(tf.constant(1.,dtype),tf.abs(theta_batch[row,component]))
                            perturbation=(tf.one_hot(row,args.batch_size,dtype=dtype)[:,None] *
                                tf.one_hot(component,case.p,dtype=dtype)[None,:])*step
                            upper=candidate_endpoint(theta_batch+perturbation)
                            lower=candidate_endpoint(theta_batch-perturbation)
                            derivative=(upper["value"][row]-lower["value"][row])/(2*step)
                            incremental=(upper["trace"][row,:,0]-lower["trace"][row,:,0])/(2*step)
                            directions.append(dict(parameter=component,finite_difference=float(derivative.numpy()),
                                absolute_error=float(tf.abs(derivative-candidate["score"][row,component]).numpy()),
                                trace_absolute_error=runtime.json_ready(tf.abs(incremental-candidate["score_trace"][row,:,component])),
                                upper_valid=bool(upper["valid"][row].numpy()),
                                lower_valid=bool(lower["valid"][row].numpy()),
                                upper_completed=int(upper["steps_completed"][row].numpy()),
                                lower_completed=int(lower["steps_completed"][row].numpy())))
                        row_results.append(directions)
                    ladder.append(dict(width=width,rows=row_results))
                    result["finite_difference_ladder"]=ladder
                    save(destination,runtime.json_ready(result))
                # Require the two finest extrapolants to agree with each
                # other and the analytical derivative. Keep the fixed-width
                # failure visible; never select the most favorable width.
                converged=True
                comparisons=[]
                for row in range(args.batch_size):
                    for component in range(case.p):
                        estimates=[]
                        healthy=True
                        for coarse,fine in zip(ladder,ladder[1:]):
                            a=coarse["rows"][row][component]
                            b=fine["rows"][row][component]
                            ratio=(coarse["width"]/fine["width"])**2
                            estimates.append((ratio*b["finite_difference"]-a["finite_difference"])/(ratio-1))
                            healthy=healthy and all(x["upper_valid"] and x["lower_valid"] and x["upper_completed"]==horizon and x["lower_completed"]==horizon for x in (a,b))
                        analytical=float(candidate["score"][row,component].numpy())
                        tolerance=2e-5+2e-5*abs(estimates[-1])
                        stable=all(abs(e-estimates[-1])<=tolerance for e in estimates)
                        agreement=all(abs(e-analytical)<=tolerance for e in estimates)
                        passed=healthy and stable and agreement
                        converged=converged and passed
                        comparisons.append(dict(row=row,parameter=component,extrapolated=estimates,
                            analytical=analytical,tolerance=tolerance,healthy=healthy,
                            stable=stable,agreement=agreement,pass_check=passed))
                result.update(finite_difference_fixed_width_pass=fd_pass,
                    finite_difference_convergence_pass=converged,
                    finite_difference_convergence=comparisons)
                fd_pass=converged
            result["analytical_implementation_parity_pass"]=parity
            parity=parity and fd_pass
            result.update(finite_difference_score=runtime.json_ready(fd),
                finite_difference_error=runtime.json_ready(fd_error),finite_difference_pass=fd_pass)
        result.update(old_first_call_seconds=compile_old,baseline=runtime.json_ready(baseline),
            score_absolute_error=runtime.json_ready(score_error),value_absolute_error=runtime.json_ready(value_error),
            tolerance=dict(rtol=rtol,atol=atol),parity_pass=parity,status="TIMING")
        save(destination,runtime.json_ready(result))
        if not parity:
            failed="FAIL_FINITE_DIFFERENCE" if args.unit and not result.get("finite_difference_pass",True) and bool(score_ok.numpy() and value_ok.numpy()) else "FAIL_SCORE_PARITY"
            result.update(status=failed,performance_status="NOT_MEASURED")
            return 1
        timings=[]
        for iteration in range(1 if args.unit else args.paired_repeats):
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
        speedup=(statistics.median(t["old"] for t in timings)/statistics.median(t["new"] for t in timings)) if timings else None
        result.update(graph=dict(xla_must_compile=bool(concrete.function_def.attr["_XlaMustCompile"].b),traces=new.experimental_get_tracing_count(),host_callbacks=forbidden),
            median_speedup=speedup,performance_status=("NOT_MEASURED" if speedup is None else "DESCRIPTIVE_FASTER" if speedup>1 else "DESCRIPTIVE_SLOWER"),
            status="PASS" if not forbidden and concrete.function_def.attr["_XlaMustCompile"].b else "FAIL")
        if not args.unit:
            result["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
        return 0 if result["status"]=="PASS" else 1
    except Exception:
        result.update(status="ERROR",error=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        result["wall_seconds"]=time.monotonic()-started
        result["peak_host_rss_bytes"]=1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if not args.unit and "tf" in locals():
            try:
                result["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
            except Exception as exc:
                result["allocator_error"]=str(exc)
        save(destination,result)


def campaign_remaining(path):
    ledger=json.loads(Path(path).read_text())
    spent=sum(float(row.get("wall_seconds",row.get("reserved_seconds",0.)))
              for row in ledger["attempts"])
    remaining=float(ledger["authorized_gpu_wall_seconds"])-spent
    if not math.isfinite(remaining):
        raise ValueError("campaign budget must be finite")
    if len(ledger["attempts"])>=ledger["max_attempts"]:
        remaining=0.
    return ledger,max(0.,remaining)


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); completed=[]
    models=(args.model,) if args.model else MODELS
    for name in models:
        remaining=args.budget_seconds-(time.monotonic()-started)
        if args.campaign_ledger:
            ledger,campaign_left=campaign_remaining(args.campaign_ledger)
            remaining=min(remaining,campaign_left)
        if remaining<=0: break
        command=[sys.executable,str(Path(__file__).resolve()),"worker","--model",name,"--output",str(args.output/name),"--gpu",args.gpu]
        command += ["--batch-size",str(args.batch_size),"--paired-repeats",str(args.paired_repeats)]
        command += ["--offset",str(args.offset)]
        if args.unit: command += ["--unit","--unit-horizon",str(args.unit_horizon),"--correction-steps",str(args.correction_steps)]
        if args.fd_ladder: command += ["--fd-ladder"]
        if args.fd_sir: command += ["--fd-sir"]
        if args.score_only: command += ["--score-only"]
        if args.profile: command += ["--profile"]
        if args.fp64_reference: command += ["--fp64-reference"]
        if args.debug_horizon: command += ["--debug-horizon",str(args.debug_horizon)]
        if args.fd_sir: command += ["--fd-widths",*[str(w) for w in args.fd_widths]]
        save(args.output/"next-phase-plan.json",dict(status="RUNNING",active=name,completed=completed,remaining_seconds=remaining,next="paired analytical score and warmed runtime; HMC remains separate"))
        attempt_started=time.monotonic()
        if args.campaign_ledger:
            entry=dict(output=str(args.output),model=name,command=command,status="RUNNING",
                started_utc=datetime.now(timezone.utc).isoformat(),
                reserved_seconds=min(remaining,args.per_model_timeout))
            ledger["attempts"].append(entry)
            ledger["next_action"]="running "+name+"; inspect this attempt before launching another worker"
            save(args.campaign_ledger,ledger)
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
        if args.campaign_ledger:
            entry["wall_seconds"]=time.monotonic()-attempt_started
            entry["exit_code"]=record.get("exit_code")
            entry["status"]=record.get("status","FINISHED" if record.get("exit_code")==0 else "FAILED")
            entry["result"]=str(args.output/name/"result.json")
            if Path(entry["result"]).is_file():
                entry["status"]=json.loads(Path(entry["result"]).read_text()).get("status",entry["status"])
            ledger["used_seconds"]=sum(float(row.get("wall_seconds",row.get("reserved_seconds",0.)))
                                       for row in ledger["attempts"])
            ledger["remaining_seconds"]=max(0.,ledger["authorized_gpu_wall_seconds"]-ledger["used_seconds"])
            ledger["next_action"]="inspect result and refresh phase; continue planned repair within remaining budget"
            save(args.campaign_ledger,ledger)
        completed.append(record)
        save(args.output/"execution.json",dict(completed=completed,wall_seconds=time.monotonic()-started,budget_seconds=args.budget_seconds))
        # A failed candidate for one model does not veto an independent model.
        # The total wall budget remains the stop condition.
    passed=len(completed)==len(models) and all(r.get("exit_code")==0 for r in completed)
    next_step=("interpret SIR finite-value derivative diagnostic; do not promote HMC"
               if args.fd_sir else "compare same-input arithmetic and per-step health; no HMC admission"
               if args.score_only else
               "inspect per-model parity and performance; tune only fully repaired scopes"
               if passed else "inspect failed model results and repair within remaining budget")
    save(args.output/"next-phase-plan.json",dict(status="DIAGNOSTIC_COMPLETE" if (args.fd_sir or args.score_only) and passed else "PARITY_PASS_PERFORMANCE_UNRESOLVED" if passed else "REPAIR_REQUIRED",completed=completed,remaining_seconds=max(0,args.budget_seconds-(time.monotonic()-started)),next=next_step))
    return 0 if passed else 1



def compare_precision_results(fp32, fp64):
    """Same-input final-output precision screen; no posterior authority."""
    if any(item.get("status")!="DIAGNOSTIC_FINISHED" for item in (fp32,fp64)):
        return dict(status="INVALID_COMPARISON",reason="both workers must finish")
    scope=("model","particles","horizon","batch_size","controls","beta",
           "data_sha256","parameter_positions","source_sha256","source_dependency_sha256")
    mismatches=[key for key in scope if key not in fp32 or fp32.get(key)!=fp64.get(key)]
    if mismatches:
        return dict(status="INVALID_COMPARISON",mismatched_scope=mismatches)
    if fp32.get("dtype")!="float32" or fp64.get("dtype")!="float64":
        return dict(status="INVALID_COMPARISON",reason="requires FP32 and FP64 pair")
    rows=[]
    for index in range(fp32["batch_size"]):
        a=fp32["candidate"]; b=fp64["candidate"]
        healthy=(a["valid"][index] and b["valid"][index]
                 and a["steps_completed"][index]==fp32["horizon"]
                 and b["steps_completed"][index]==fp64["horizon"])
        av=[a["value"][index],*a["score"][index]]
        bv=[b["value"][index],*b["score"][index]]
        if len(av)!=len(bv) or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in av+bv):
            return dict(status="INVALID_COMPARISON",reason="nonfinite or incompatible outputs")
        error=[abs(x-y) for x,y in zip(av,bv)]
        scaled=[e/max(1.,abs(y)) for e,y in zip(error,bv)]
        rows.append(dict(row=index,healthy=bool(healthy),
            fp32_value=av[0],fp64_value=bv[0],fp32_score=av[1:],fp64_score=bv[1:],
            absolute_value_error=error[0],absolute_score_error=error[1:],
            scaled_value_error=scaled[0],scaled_score_error=scaled[1:],
            materiality_pass=bool(healthy and max(scaled)<=.01)))
    return dict(status="PRECISION_PASS" if all(r["materiality_pass"] for r in rows) else "PRECISION_REPAIR_REQUIRED",
                rows=rows,tolerance=.01,scale="max(1,abs(FP64 reference component))",
                role="precision diagnostic only; FP64 is the same algorithm, not a model-likelihood oracle")


def precision(args):
    """Run both precisions for each scope, accounting every GPU attempt."""
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    results=[]
    models=(args.model,) if args.model else MODELS
    for name in models:
        paths={}
        for mode in ("fp32","fp64"):
            remaining=args.budget_seconds-(time.monotonic()-started)
            if args.campaign_ledger:
                _,left=campaign_remaining(args.campaign_ledger)
                remaining=min(remaining,left)
            if remaining<=0:
                break
            child=argparse.Namespace(**vars(args))
            child.command="run"; child.model=name; child.score_only=True
            child.output=args.output/(name+"-"+mode)
            child.fp64_reference=mode=="fp64"; child.budget_seconds=remaining
            run(child)
            result_path=child.output/name/"result.json"
            if result_path.exists():
                paths[mode]=result_path
        if set(paths)=={"fp32","fp64"}:
            data={mode:json.loads(path.read_text()) for mode,path in paths.items()}
            try:
                comparison=compare_precision_results(data["fp32"],data["fp64"])
            except (KeyError,TypeError,ValueError) as error:
                comparison=dict(status="INVALID_COMPARISON",reason=str(error))
        else:
            comparison=dict(status="INCOMPLETE_BUDGET")
        comparison.update(model=name,artifacts={k:str(v) for k,v in paths.items()})
        results.append(comparison)
        all_pass=(len(results)==len(models) and all(r["status"]=="PRECISION_PASS" for r in results))
        save(args.output/"precision.json",dict(status="PRECISION_PASS" if all_pass else "REPAIR_OR_COMPLETION_REQUIRED",
            models=results,wall_seconds=time.monotonic()-started,plan=str(PLAN)))
        save(args.output/"next-phase-plan.json",dict(
            status="PRECISION_PASS" if all_pass else "REPAIR_OR_COMPLETION_REQUIRED",
            models=results,
            next="fresh scope tuning and conditional HMC mechanics" if all_pass else
                 "complete the independent model checks; localize or retune failing scopes; no HMC promotion",
            remaining_seconds=max(0.,args.budget_seconds-(time.monotonic()-started))))
    return 0 if len(results)==len(models) and all(r["status"]=="PRECISION_PASS" for r in results) else 1


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
        "tests/test_surrogate_hmc_performance_repair.py",
        "tests/highdim/test_covariance_proposal.py",
        "tests/highdim/test_covariance_proposal_precision_guard.py",
        "tests/highdim/test_covariance_proposal_packed_score.py"]
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
    parser.add_argument("command",choices=("worker","run","regression","precision"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",choices=MODELS)
    parser.add_argument("--unit",action="store_true")
    parser.add_argument("--unit-horizon",type=int,default=2,help="CPU-only recursive diagnostic horizon, at most 10")
    parser.add_argument("--gpu",default="0")
    parser.add_argument("--budget-seconds",type=float,default=3600.)
    parser.add_argument("--campaign-ledger",type=Path,help="existing cumulative GPU budget; failures and compilation count")
    parser.add_argument("--per-model-timeout",type=float,default=900.)
    parser.add_argument("--correction-steps",type=int,default=0)
    parser.add_argument("--offset",type=float,default=0.)
    parser.add_argument("--batch-size",type=int,choices=(1,2),default=2)
    parser.add_argument("--paired-repeats",type=int,choices=(0,1,2),default=2)
    parser.add_argument("--fd-sir",action="store_true",help="bounded full-scope FP32 finite-value score diagnostic")
    parser.add_argument("--fd-ladder",action="store_true",help="CPU-only finite-difference width ladder and per-step trace")
    parser.add_argument("--score-only",action="store_true",help="single full-size candidate evaluation; diagnostic only")
    parser.add_argument("--profile",action="store_true",help="capture one warmed score-only GPU profile and optimized HLO")
    parser.add_argument("--fp64-reference",action="store_true",help="FP64 diagnostic exception; requires --score-only or --fd-sir")
    parser.add_argument("--debug-horizon",type=int,default=0,help="N4032 prefix localization only; no full-scope admission")
    parser.add_argument("--fd-widths",type=float,nargs="+",default=[.005,.001])
    args=parser.parse_args()
    if args.command=="worker" and not args.model: parser.error("worker requires --model")
    if args.campaign_ledger and (args.unit or args.command not in ("run","precision")):
        parser.error("--campaign-ledger applies only to GPU run orchestration")
    if args.profile and (not args.score_only or args.unit):
        parser.error("--profile requires a full-size --score-only GPU diagnostic")
    if args.fd_ladder and not args.unit: parser.error("--fd-ladder requires --unit")
    if args.score_only and (args.unit or args.fd_sir): parser.error("--score-only is a separate full-scope diagnostic")
    if args.fp64_reference and not (args.score_only or args.fd_sir): parser.error("--fp64-reference requires --score-only or --fd-sir")
    if args.debug_horizon and (not 1<=args.debug_horizon<=10 or args.unit or not(args.score_only or args.fd_sir)):
        parser.error("--debug-horizon requires a score-only or FD diagnostic with horizon 1..10")
    if len(args.fd_widths)<2 or any(not math.isfinite(w) or w<=0 for w in args.fd_widths):
        parser.error("--fd-widths requires at least two positive finite widths")
    if args.fd_sir and (args.model!="sir_d18" or args.unit or args.batch_size!=2):
        parser.error("--fd-sir requires full-size sir_d18 with batch size 2")
    if not (1<=args.unit_horizon<=10) or (not args.unit and args.unit_horizon!=2):
        parser.error("--unit-horizon must be 1..10 and requires --unit when not 2")
    if any(not math.isfinite(v) or v<=0 for v in (args.budget_seconds,args.per_model_timeout)): parser.error("budgets must be finite and positive")
    if not math.isfinite(args.offset): parser.error("--offset must be finite")
    if args.correction_steps<0: parser.error("--correction-steps must be nonnegative")
    if args.command=="precision":
        if args.unit or args.fd_sir or args.profile or args.debug_horizon or args.fp64_reference:
            parser.error("precision requires the full-scope paired GPU mode")
        return precision(args)
    if args.command=="regression": return regression(args)
    return worker(args) if args.command=="worker" else run(args)


if __name__=="__main__":
    sys.exit(main())
