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
SOURCES=("analytical_jet_tf.py","covariance_proposal_shared_gaussian_tf.py","covariance_proposal_score_tf.py", "covariance_proposal_score_batch_tf.py","covariance_proposal_score_moments_tf.py","covariance_proposal_tf.py","covariance_proposal_transport_tf.py","covariance_proposal_moments_tf.py","ledh_marginal_weights_tf.py")


def save(path,value):
    path=Path(path)
    temporary=path.with_suffix(".tmp")
    def clean(v):
        # Host serialization only: preserve tensor contents as Python values.
        if hasattr(v, "numpy"): return clean(v.numpy())
        if hasattr(v, "tolist"): return clean(v.tolist())
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



def pilot_timing_audit(pilot):
    """Separate a measured tiny chunk from an unmeasured long-chain rate."""
    if not pilot.get("hmc_executed"):
        return dict(status="NOT_MEASURED", role="explanatory_timing_only")
    steps = int(pilot["steps_per_chain"])
    return dict(status="INITIAL_FORCE_INCLUDED_IN_SHORT_CHUNK",
        role="explanatory_timing_only", steps_per_chain=steps,
        initial_force_evaluations_per_chunk=1,
        leapfrog_steps_per_transition=1,
        expected_force_evaluations_per_chunk=1+steps,
        measured_chunk_seconds=pilot["transition_wall_seconds"],
        measured_chunk_average_seconds=pilot["seconds_per_four_chain_transition"],
        raw_short_chunk_extrapolation_hours=pilot.get("minimum_2000_warmup_hours_at_measured_rate"),
        long_chain_rate_measured=False, warmup_lower_bound_established=False,
        interpretation="The pilot initializes its retained force on every call. "
            "Dividing a two-step chunk by two includes half an extra force call "
            "per transition. Its 2000-step extrapolation is not a warmup lower bound; "
            "use the separately measured force/endpoint cost model with explicit limitations.",
        source="sample_neural_force_hmc: one initial force plus one per L=1 transition")


def consumer_cost_model(diagnostic):
    """Descriptive fixed-position L=1 model; not a measured long chain."""
    rows={row["name"]:row for row in diagnostic.get("measurements",[])}
    names=("actual_force_B4_in_B2_groups", "independent_endpoint_B4")
    health=diagnostic.get("lane_health",{})
    if any(name not in rows or not rows[name].get("warmed_seconds") for name in names) or not all(
            health.get(name,False) for name in ("B4_grouped_force","B4_endpoint")):
        return dict(status="UNRESOLVED", role="explanatory_timing_only")
    force,endpoint=(statistics.mean(rows[name]["warmed_seconds"]) for name in names)
    return dict(status="DESCRIPTIVE_L1_COST_MODEL", role="explanatory_timing_only",
        initial_force_seconds=force, force_seconds_per_transition=force,
        endpoint_seconds_per_transition=endpoint,
        modeled_seconds_per_four_chain_transition=force+endpoint,
        modeled_2000_transition_hours_including_one_initial_force=(force+2000*(force+endpoint))/3600,
        long_chain_rate_measured=False, warmup_lower_bound_established=False,
        limitations="Fixed-position timings, few repeats, L=1 only; no adaptation, "
            "tuned trajectory length, changing-position costs or convergence claim.")


def consumer_cost_diagnostic(case, output, baseline_path=None):
    """Same-input execution-lane diagnostic; no HMC/tuning admission."""
    import tensorflow as tf
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
    starts=tf.constant([[v]*case.p for v in (-.15,-.05,.05,.15)],tf.float64)
    theta,jacobian=case.chart(tf.cast(starts,case.force_dtype))
    jacobian=tf.broadcast_to(jacobian,tf.shape(theta))
    result=dict(status="RUNNING",scope=case.scope(),selection_hash=case.selection_hash,
        positions=starts,measurements=[],scaled_gaps={},lane_health={},
        posterior_sampling_complete=False,evidence_role="same_input_consumer_cost_localization",
        nonclaims=["no statistical speed ranking","no posterior convergence",
                   "B4 is diagnostic only and requires new scope tuning for HMC"])
    destination=Path(output)/"consumer-cost.json"
    def measure(name,function,arguments):
        def call():
            begin=time.monotonic()
            value=function(*arguments)
            tf.nest.map_structure(lambda x:x.numpy() if tf.is_tensor(x) else x,value)
            return value,time.monotonic()-begin
        value,cold=call()
        times=[]
        row=dict(name=name,cold_compile_execute_seconds=cold,warmed_seconds=times,
                 allocator=tf.config.experimental.get_memory_info("GPU:0"))
        # Archive useful results before optional repetitions or the next lane.
        fields=(value if isinstance(value,dict) else value._asdict())
        retained={k:v for k,v in fields.items() if k in
                  ("value","score","valid","steps_completed","force","force_valid","potential","status")}
        row["output"]=retained
        result["measurements"].append(row)
        save(destination,result)
        for _ in range(2):
            value,seconds=call(); times.append(seconds)
            save(destination,result)
        return value
    def gap(a,b):
        return tf.reduce_max(tf.abs(a-b)/tf.maximum(tf.ones_like(a),tf.maximum(tf.abs(a),tf.abs(b))))
    def checked(name,healthy,comparisons=None):
        result["lane_health"][name]=bool(healthy)
        result["scaled_gaps"].update(comparisons or {})
        save(destination,result)
    program2=make_batched_score_filter(case.spec,runtime.N,case.horizon,case.controls,
                                       case.force_dtype,jit_compile=True,batch_size=2,
                                       precision_policy=case.precision_policy)
    inputs=(theta[:2],case.observations,*case.fixed,case.beta)
    direct2=measure("explicit_B2_full_output",program2,inputs)
    checked("B2_direct",tf.reduce_all(direct2["valid"] &
            (direct2["steps_completed"]==case.horizon)))
    expected2=starts[:2]-tf.cast(jacobian[:2]*direct2["score"],tf.float64)
    @tf.function(input_signature=program2.input_signature,jit_compile=True,autograph=False)
    def projected(*values):
        evaluated=program2(*values)
        return dict(score=evaluated["score"],valid=evaluated["valid"] &
                    (evaluated["steps_completed"]==case.horizon))
    projected2=measure("explicit_B2_score_projection",projected,inputs)
    checked("B2_projection",tf.reduce_all(projected2["valid"]),
            dict(B2_projection=gap(direct2["score"],projected2["score"])))
    components=runtime.bindings(case)
    force2=measure("actual_force_B2_closed_inputs",components["force"].function,(starts[:2],))
    checked("B2_force",tf.reduce_all(force2.force_valid),dict(B2_force=gap(expected2,force2.force)))
    force4=measure("actual_force_B4_in_B2_groups",components["force"].function,(starts,))
    checked("B4_grouped_force",tf.reduce_all(force4.force_valid),
            dict(B2_B4_grouped_force=gap(force2.force,force4.force[:2])))
    endpoint=measure("independent_endpoint_B4",components["target"].function,(starts,))
    checked("B4_endpoint",tf.reduce_all(tf.math.is_finite(endpoint.potential) &
            endpoint.status["value_valid"] & ~endpoint.status["numerical_failure"]))
    program4=make_batched_score_filter(case.spec,runtime.N,case.horizon,case.controls,
                                       case.force_dtype,jit_compile=True,batch_size=4,
                                       precision_policy=case.precision_policy)
    direct4=measure("diagnostic_native_B4_full_output",program4,
                    (theta,case.observations,*case.fixed,case.beta))
    expected4=starts-tf.cast(jacobian*direct4["score"],tf.float64)
    checked("B4_direct",tf.reduce_all(direct4["valid"] &
            (direct4["steps_completed"]==case.horizon)),dict(
            B4_force=gap(expected4,force4.force),
            B2_B4_score=gap(direct2["score"],direct4["score"][:2])))
    if baseline_path is not None:
        from bayesfilter.testing.surrogate_hmc_covariance_cost_diagnostic import run_consumer_pair
        paired=run_consumer_pair(case,output,baseline_path)
        result["paired_comparison"]=str(Path(output)/"consumer-paired.json")
        checked("paired_baseline",paired["status"]=="CONSUMER_PAIRED_PARITY_PASS_COST_MEASURED")
    healthy=all(result["lane_health"].values())
    precision=all(bool(v<=.01) for v in result["scaled_gaps"].values())
    result.update(status="CONSUMER_PARITY_PASS_COST_MEASURED" if healthy and precision else
                  "CONSUMER_PARITY_OR_HEALTH_VETO",healthy=healthy)
    result["long_chain_cost_model"]=consumer_cost_model(result)
    save(destination,result)
    return result


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
    source_paths += [Path(__file__).resolve(),ROOT/"bayesfilter/testing/surrogate_hmc_five_model.py",ROOT/"bayesfilter/testing/surrogate_hmc_scope_tuning.py",
                     ROOT/"bayesfilter/inference/neural_force_hmc.py",
                     ROOT/"bayesfilter/testing/surrogate_hmc_force_reuse_diagnostic.py",
                     ROOT/"bayesfilter/testing/surrogate_hmc_precision_islands_diagnostic.py",
                     ROOT/"bayesfilter/testing/surrogate_hmc_component_cost_diagnostic.py",
                     ROOT/"bayesfilter/testing/surrogate_hmc_transport_cache_diagnostic.py",
                     ROOT/"bayesfilter/testing/surrogate_hmc_transport_tile_batch_diagnostic.py",
                     ROOT/"bayesfilter/testing/surrogate_hmc_covariance_cost_diagnostic.py",
                     ROOT/"bayesfilter/testing/surrogate_hmc_force_health_diagnostic.py"]
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
        result["seed"]=31001 if args.model=="range_bearing" else 261008301
        scope=case.scope()
        result["source_dependency_sha256"]=scope["source_sha256"]
        for relative in scope["source_sha256"]:
            snapshot=args.output/"source"/relative
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/relative,snapshot)
        dtype=tf.float64 if args.unit or args.fp64_reference or getattr(args,"runtime_precision","uniform")!="uniform" else tf.float32
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
        if args.shared_gaussian_contraction:
            controls=replace(controls,shared_gaussian_contraction=args.shared_gaussian_contraction)
            case.controls=controls
        if getattr(args,"runtime_precision","uniform")!="uniform":
            case=runtime.configure_precision(case,args.runtime_precision,
                                             getattr(args,"tuning_seed_offset",0))
        else:
            case.tuning_seed_offset=getattr(args,"tuning_seed_offset",0)
        result.update(precision_policy=case.precision_policy,
                      pairwise_dtype="float32" if case.precision_policy!="uniform" else dtype.name,
                      tuning_seed_offset=case.tuning_seed_offset)
        if args.filter_tuning or args.hmc_mechanics:
            result.update(particles=n,horizon=horizon,parameter_count=case.p,
                          state_dimension=case.spec.dimension,dtype=case.force_dtype.name,tf32=False,jit_compile=True)
        if getattr(args,"force_health_replay",None):
            from bayesfilter.testing.surrogate_hmc_force_health_diagnostic import run_diagnostic
            result.update(status="FORCE_HEALTH_RUNNING",scope=case.scope(),
                particles=n,horizon=case.horizon,dtype=case.force_dtype.name,
                tf32=False,jit_compile=True,tensorflow=tf.__version__,
                gpu_devices=[tf.config.experimental.get_device_details(d) for d in tf.config.list_physical_devices("GPU")],
                evidence_role="exact_input_force_consumer_diagnostic")
            save(destination,runtime.json_ready(result))
            diagnostic=run_diagnostic(case,args.output,args.force_health_replay)
            result.update(status=diagnostic["status"],
                force_health=str(args.output/"force-health.json"))
            return 0 if diagnostic["status"]=="FORCE_CONSUMER_HEALTH_PASS" else 1
        if getattr(args,"transport_tile_batch_probe",False):
            from bayesfilter.testing.surrogate_hmc_transport_tile_batch_diagnostic import run_diagnostic
            result.update(status="TRANSPORT_TILE_BATCH_PROBE_RUNNING",scope=case.scope(),particles=n,horizon=case.horizon,dtype="float32",tf32=False,jit_compile=True,evidence_role="synthetic_exact_tile_schedule_nomination_only")
            save(destination,runtime.json_ready(result))
            diagnostic=run_diagnostic(case,args.output)
            result.update(status=diagnostic["status"],transport_tile_batch_probe=str(args.output/"transport-tile-batch.json"))
            return 0 if diagnostic["status"]=="TILE_PRIMITIVE_PARITY_PASS_COST_MEASURED" else 1
        if getattr(args,"transport_cache_probe",False) or getattr(args,"transport_combined_probe",False):
            from bayesfilter.testing.surrogate_hmc_transport_cache_diagnostic import run_diagnostic
            result.update(status="TRANSPORT_CACHE_PROBE_RUNNING",scope=case.scope(),particles=n,horizon=case.horizon,dtype="float32",tf32=False,jit_compile=True,evidence_role="synthetic_primitive_cost_nomination_only")
            save(destination,runtime.json_ready(result))
            method="combined" if args.transport_combined_probe else "cache"
            diagnostic=run_diagnostic(case,args.output,repetitions=100 if method=="combined" else 10,method=method)
            result.update(status=diagnostic["status"],transport_primitive_probe=str(args.output/("transport-combined-probe.json" if method=="combined" else "transport-cache-probe.json")))
            return 0 if diagnostic["status"]=="PRIMITIVE_PARITY_PASS_COST_MEASURED" else 1
        if getattr(args,"covariance_baseline",None):
            from bayesfilter.testing.surrogate_hmc_covariance_cost_diagnostic import run_diagnostic
            result.update(status="COVARIANCE_COMPARISON_RUNNING",scope=case.scope(),
                particles=n,horizon=case.horizon,dtype=case.force_dtype.name,
                tf32=False,jit_compile=True,tensorflow=tf.__version__,
                gpu_devices=[tf.config.experimental.get_device_details(d) for d in tf.config.list_physical_devices("GPU")],
                evidence_role="same_filter_implementation_comparison")
            save(destination,runtime.json_ready(result))
            diagnostic=run_diagnostic(case,args.output,args.covariance_baseline)
            result.update(status=diagnostic["status"],
                covariance_comparison=str(args.output/"covariance-comparison.json"))
            return 0 if diagnostic["status"]=="COVARIANCE_IMPLEMENTATION_PARITY_PASS_COST_MEASURED" else 1
        if getattr(args,"component_cost",False):
            from bayesfilter.testing.surrogate_hmc_component_cost_diagnostic import run_diagnostic, run_sequence_diagnostic
            sequence = getattr(args, "component_sequence", False)
            result.update(status="COMPONENT_COST_RUNNING",scope=case.scope(),
                particles=n,horizon=case.horizon if sequence else 1,requested_horizon=case.horizon,
                dtype=case.force_dtype.name,tf32=False,jit_compile=True,
                tensorflow=tf.__version__,
                gpu_devices=[tf.config.experimental.get_device_details(d) for d in tf.config.list_physical_devices("GPU")],
                evidence_role="full_sequence_component_cost_only" if sequence else "first_observation_component_cost_only")
            save(destination,runtime.json_ready(result))
            diagnostic=(run_sequence_diagnostic if sequence else run_diagnostic)(case,args.output)
            result.update(status=diagnostic["status"],component_cost=str(args.output/(
                "component-sequence-cost.json" if sequence else "component-cost.json")))
            expected="COMPONENT_SEQUENCE_PARITY_PASS_COST_MEASURED" if sequence else "COMPONENT_PARITY_PASS_COST_MEASURED"
            return 0 if diagnostic["status"]==expected else 1
        if args.filter_tuning:
            result.update(status="FILTER_TUNING_RUNNING",scope=case.scope(),
                          evidence_role="fresh_scope_control_nomination_with_precision_veto")
            save(destination,runtime.json_ready(result))
            selected=runtime.tune_filter_scope(case,args.output,smoother_reset=args.smoother_reset)
            result.update(status=selected["status"],selection=str(args.output/"filter-selection.json"),
                          selected=selected["selected"],scope=selected["scope"])
            return 0 if selected["selected"] is not None else 1
        if getattr(args,"force_reuse",False):
            from bayesfilter.testing.surrogate_hmc_force_reuse_diagnostic import run_diagnostic
            runtime.load_selection(case,args.selection)
            diagnostic=run_diagnostic(case,args.output)
            result.update(status=diagnostic["status"],force_reuse=str(args.output/"force-reuse.json"))
            return 0 if diagnostic["status"]=="FORCE_REUSE_PARITY_PASS" else 1
        if getattr(args,"consumer_cost",False):
            runtime.load_selection(case,args.selection)
            if getattr(args,"consumer_paired_only",False):
                from bayesfilter.testing.surrogate_hmc_covariance_cost_diagnostic import run_consumer_pair
                diagnostic=run_consumer_pair(case,args.output,args.consumer_baseline)
                result.update(status=diagnostic["status"],consumer_cost=str(args.output/"consumer-paired.json"))
                return 0 if diagnostic["status"]=="CONSUMER_PAIRED_PARITY_PASS_COST_MEASURED" else 1
            diagnostic=consumer_cost_diagnostic(case,args.output,getattr(args,"consumer_baseline",None))
            result.update(status=diagnostic["status"],consumer_cost=str(args.output/"consumer-cost.json"))
            return 0 if diagnostic["status"]=="CONSUMER_PARITY_PASS_COST_MEASURED" else 1
        if args.hmc_mechanics:
            if not args.selection: raise ValueError("HMC mechanics requires a fresh filter selection")
            runtime.load_selection(case,args.selection)
            result.update(status="HMC_MECHANICS_RUNNING",scope=case.scope(),
                          evidence_role="conditional_corrected_HMC_mechanics_only")
            save(destination,runtime.json_ready(result))
            endpoint=runtime.endpoint_audit(case,args.output)
            if endpoint["status"]!="PASS_ENDPOINT_MECHANICS":
                result.update(status="ENDPOINT_REPAIR_REQUIRED",endpoint_audit=endpoint)
                return 1
            pilot=runtime.pilot(case,args.output)
            timing=pilot_timing_audit(pilot)
            save(args.output/"pilot-timing-audit.json",timing)
            result.update(status=pilot["status"],pilot=pilot,endpoint_audit=endpoint,
                          pilot_timing_audit=timing)
            return 0 if pilot["status"]=="PASS_HMC_MECHANICS_ONLY" else 1
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
        if getattr(args,"precision_islands",None):
            from bayesfilter.testing.surrogate_hmc_precision_islands_diagnostic import make_program
            new,metadata=make_program(case.spec,n,horizon,controls,
                policy=args.precision_islands,batch_size=args.batch_size)
            result.update(precision_islands=metadata,
                          evidence_role="mixed_precision_arithmetic_localization")
        else:
            new=make_batched_score_filter(case.spec,n,horizon,controls,dtype,True,batch_size=args.batch_size,precision_policy=case.precision_policy)
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
    """Available worker-wall time, independent of the separate attempt limit."""
    ledger=json.loads(Path(path).read_text())
    authorized=float(ledger["authorized_gpu_wall_seconds"])
    charges=[float(row.get("wall_seconds",row.get("reserved_seconds",0.)))
             for row in ledger["attempts"]]
    if not all(math.isfinite(v) and v>=0 for v in [authorized,*charges]):
        raise ValueError("campaign budget and charges must be finite and nonnegative")
    campaign_attempts_remaining(ledger)
    return ledger,max(0.,authorized-sum(charges))


def campaign_attempts_remaining(ledger):
    limit=ledger["max_attempts"]
    if isinstance(limit,bool) or not isinstance(limit,int) or limit<0:
        raise ValueError("campaign attempt limit must be a nonnegative integer")
    return max(0,limit-len(ledger["attempts"]))


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); completed=[]; stop_reason=None
    models=(args.model,) if args.model else MODELS
    for name in models:
        remaining=args.budget_seconds-(time.monotonic()-started)
        if args.campaign_ledger:
            ledger,campaign_left=campaign_remaining(args.campaign_ledger)
            remaining=min(remaining,campaign_left)
            if campaign_left<=0:
                stop_reason="BUDGET_EXHAUSTED"; break
            if campaign_attempts_remaining(ledger)<=0:
                stop_reason="ATTEMPT_LIMIT_REACHED"; break
        if remaining<=0:
            stop_reason="PHASE_BUDGET_EXHAUSTED"; break
        command=[sys.executable,str(Path(__file__).resolve()),"worker","--model",name,"--output",str(args.output/name),"--gpu",args.gpu]
        command += ["--batch-size",str(args.batch_size),"--paired-repeats",str(args.paired_repeats)]
        command += ["--offset",str(args.offset)]
        if args.unit: command += ["--unit","--unit-horizon",str(args.unit_horizon),"--correction-steps",str(args.correction_steps)]
        if args.fd_ladder: command += ["--fd-ladder"]
        if args.fd_sir: command += ["--fd-sir"]
        if args.score_only: command += ["--score-only"]
        if getattr(args,"precision_islands",None):
            command += ["--precision-islands",args.precision_islands]
        if getattr(args,"runtime_precision","uniform")!="uniform":
            command += ["--runtime-precision",args.runtime_precision]
        if getattr(args,"tuning_seed_offset",0):
            command += ["--tuning-seed-offset",str(args.tuning_seed_offset)]
        if getattr(args,"filter_tuning",False): command += ["--filter-tuning"]
        if getattr(args,"smoother_reset",False): command += ["--smoother-reset"]
        if getattr(args,"hmc_mechanics",False): command += ["--hmc-mechanics","--selection",str(args.selection)]
        if getattr(args,"consumer_cost",False): command += ["--consumer-cost","--selection",str(args.selection)]
        if getattr(args,"consumer_baseline",None): command += ["--consumer-baseline",str(args.consumer_baseline)]
        if getattr(args,"consumer_paired_only",False): command += ["--consumer-paired-only"]
        if getattr(args,"force_health_replay",None): command += ["--force-health-replay",str(args.force_health_replay)]
        if getattr(args,"covariance_baseline",None): command += ["--covariance-baseline",str(args.covariance_baseline)]
        if getattr(args,"component_cost",False): command += ["--component-cost"]
        if getattr(args,"transport_cache_probe",False): command += ["--transport-cache-probe"]
        if getattr(args,"transport_combined_probe",False): command += ["--transport-combined-probe"]
        if getattr(args,"transport_tile_batch_probe",False): command += ["--transport-tile-batch-probe"]
        if getattr(args,"component_sequence",False): command += ["--component-sequence"]
        if getattr(args,"force_reuse",False): command += ["--force-reuse","--selection",str(args.selection)]
        if getattr(args,"shared_gaussian_contraction",None):
            command += ["--shared-gaussian-contraction",args.shared_gaussian_contraction]
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
    next_step=("inspect actual force consumer and exposed traces; repair disagreement before HMC"
               if getattr(args,"force_health_replay",None) else
               "assess full-horizon covariance parity and cost; retune changed scopes before HMC"
               if getattr(args,"covariance_baseline",None) else
               "inspect primitive timings and staged/fused ratio; continue the planned cost repair"
               if getattr(args,"component_cost",False) else
               "inspect primitive parity and timing; only a full-filter test can establish cost"
               if getattr(args,"transport_cache_probe",False) or getattr(args,"transport_combined_probe",False) or getattr(args,"transport_tile_batch_probe",False) else
               "run selected-force HMC mechanics; control nomination is not posterior evidence"
               if getattr(args,"filter_tuning",False) else
               "assess corrected-HMC mechanics and measured sampling cost; no convergence claim"
               if getattr(args,"hmc_mechanics",False) else
               "interpret SIR finite-value derivative diagnostic; do not promote HMC"
               if args.fd_sir else "compare same-input arithmetic and per-step health; no HMC admission"
               if args.score_only else
               "inspect per-model parity and performance; tune only fully repaired scopes"
               if passed else "inspect failed model results and repair within remaining budget")
    phase_left=max(0.,args.budget_seconds-(time.monotonic()-started))
    campaign_fields={}; available=phase_left
    if args.campaign_ledger:
        ledger,campaign_left=campaign_remaining(args.campaign_ledger)
        campaign_fields=dict(remaining_campaign_seconds=campaign_left,
            remaining_campaign_attempts=campaign_attempts_remaining(ledger))
        available=min(available,campaign_left)
    status=("EXPLORATORY_TUNING_COMPLETE" if getattr(args,"filter_tuning",False) and passed
            else "HMC_MECHANICS_COMPLETE" if getattr(args,"hmc_mechanics",False) and passed
            else "DIAGNOSTIC_COMPLETE" if (args.fd_sir or args.score_only or getattr(args,"component_cost",False) or getattr(args,"transport_cache_probe",False) or getattr(args,"transport_combined_probe",False) or getattr(args,"transport_tile_batch_probe",False) or getattr(args,"covariance_baseline",None) or getattr(args,"force_health_replay",None)) and passed
            else "PARITY_PASS_PERFORMANCE_UNRESOLVED" if passed else "REPAIR_REQUIRED")
    if stop_reason:
        status=stop_reason
        next_step=("review the attempt cap within existing compute authorization before retrying"
                   if stop_reason=="ATTEMPT_LIMIT_REACHED" else
                   "preserve partial results; review remaining authorized time before another phase")
    save(args.output/"next-phase-plan.json",dict(status=status,stop_reason=stop_reason,
        completed=completed,remaining_seconds=available,remaining_phase_seconds=phase_left,
        next=next_step,**campaign_fields))
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



def compare_precision_islands(candidate, reference):
    """Compare a mixed prefix/full run with the unchanged saved FP64 recursion."""
    invalid=lambda reason: dict(status="INVALID_COMPARISON",reason=reason)
    scope=("model","particles","batch_size","controls","beta","seed",
           "data_sha256","parameter_positions","source_sha256","source_dependency_sha256")
    if any(k not in candidate or candidate.get(k)!=reference.get(k) for k in scope):
        return invalid("source, inputs or controls differ")
    if any(d.get("status")!="DIAGNOSTIC_FINISHED" or d.get("dtype")!="float64"
           or d.get("tf32") is not False or d.get("jit_compile") is not True
           for d in (candidate,reference)):
        return invalid("completed XLA FP64-storage results required with TF32 off")
    policy=candidate.get("precision_islands",{})
    if (policy.get("policy") not in ("both_fp32","gaussian_fp32","transport_fp32") or
            policy.get("runtime_module_patched") is not False or
            reference.get("precision_islands")):
        return invalid("explicit mixed candidate and pure FP64 reference required")
    h=candidate["horizon"]; full=reference["horizon"]
    if not 0<h<=full:
        return invalid("candidate must be a prefix of the saved reference")
    if "log_increment" not in reference.get("trace_fields",[]):
        return invalid("missing reference likelihood increments")
    increment=reference["trace_fields"].index("log_increment")
    rows=[]
    try:
        a=candidate["candidate"]; b=reference["candidate"]
        for row in range(candidate["batch_size"]):
            p=len(candidate["parameter_positions"][row])
            trace=b["trace"][row]; score_trace=b["score_trace"][row]
            if len(trace)!=full or len(score_trace)!=full or any(len(v)!=p for v in score_trace):
                return invalid("reference trace shape mismatch")
            total=[math.fsum(v[increment] for v in trace),
                   *[math.fsum(v[j] for v in score_trace) for j in range(p)]]
            saved=[b["value"][row],*b["score"][row]]
            if len(saved)!=p+1 or any(abs(x-y)>1.e-10*max(1.,abs(y)) for x,y in zip(total,saved)):
                return invalid("reference increments do not reconstruct its final result")
            oracle=[math.fsum(v[increment] for v in trace[:h]),
                    *[math.fsum(v[j] for v in score_trace[:h]) for j in range(p)]]
            actual=[a["value"][row],*a["score"][row]]
            if len(actual)!=p+1 or not all(math.isfinite(x) for x in actual+oracle+saved):
                return invalid("nonfinite or incompatible outputs")
            healthy=(a["valid"][row] and b["valid"][row] and
                     a["steps_completed"][row]==h and b["steps_completed"][row]==full)
            gaps=[abs(x-y)/max(1.,abs(y)) for x,y in zip(actual,oracle)]
            rows.append(dict(row=row,healthy=bool(healthy),candidate_value=actual[0],
                reference_value=oracle[0],candidate_score=actual[1:],reference_score=oracle[1:],
                scaled_value_error=gaps[0],scaled_score_error=gaps[1:],
                materiality_pass=bool(healthy and max(gaps)<=.01)))
    except (KeyError,IndexError,TypeError,ValueError,OverflowError):
        return invalid("missing or invalid numerical evidence")
    passed=all(row["materiality_pass"] for row in rows)
    return dict(status=("PREFIX_PRECISION_PASS" if h<full else "FULL_SCOPE_PRECISION_PASS")
                if passed else "PRECISION_REPAIR_REQUIRED",rows=rows,tolerance=.01,
                reference_horizon=full,horizon=h,policy=policy,
                scale="max(1,abs(FP64 reference component))",
                evidence_role="arithmetic localization only; no tuning or HMC admission")


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
            child.command="run"; child.model=name
            child.score_only=True
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



def execution_scopes(path):
    if path is None:
        return {}
    plan=json.loads(Path(path).read_text())
    version=plan.get("schema")
    if version not in ("surrogate_hmc_scope_execution_v1","surrogate_hmc_scope_execution_v2"):
        raise ValueError("unknown scope execution plan")
    scopes=plan.get("scopes",{})
    if set(scopes)!=set(MODELS):
        raise ValueError("execution plan must declare every model exactly once")
    for name,scope in scopes.items():
        required={"contraction","tuning_timeout_seconds"}
        if version=="surrogate_hmc_scope_execution_v2":
            required.update(("runtime_precision","tuning_seed_offset","control_family"))
        if not isinstance(scope,dict) or set(scope)!=required:
            raise ValueError("unsupported execution controls for "+name)
        if version=="surrogate_hmc_scope_execution_v2":
            if scope["runtime_precision"] not in ("uniform","fp64_recursion_fp32_pairs"):
                raise ValueError("unknown runtime precision for "+name)
            seed=scope["tuning_seed_offset"]
            if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:
                raise ValueError("invalid tuning seed offset for "+name)
            if scope["control_family"] not in ("baseline_more_steps_half_epsilon","smoother_reset_1_2_4_8"):
                raise ValueError("unknown control family for "+name)
        if scope["contraction"] not in ("direct","matrix"):
            raise ValueError("unknown contraction for "+name)
        seconds=scope["tuning_timeout_seconds"]
        if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<=0:
            raise ValueError("invalid tuning timeout for "+name)
    return scopes


def advance(args):
    """Complete fresh tuning, mechanics and optional consumer cost in one ledger."""
    include_consumer = getattr(args, "consumer_cost_after_mechanics", False)
    success_status = ("COMPUTATION_CHECKS_COMPLETE_POSTERIOR_NOT_RUN" if include_consumer
                      else "HMC_MECHANICS_COMPLETE_POSTERIOR_NOT_RUN")
    phases = ("filter-tuning", "mechanics", "consumer-cost") if include_consumer else ("filter-tuning", "mechanics")
    scopes=execution_scopes(getattr(args,"execution_plan",None))
    baseline=json.loads(args.precision_results.read_text())
    evidence={row["model"]:row for row in baseline["models"]}
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); results=[]
    models=(args.model,) if args.model else MODELS
    for name in models:
        prior=evidence.get(name,{})
        row=dict(model=name,baseline_precision=prior.get("status","MISSING"))
        if prior.get("status") not in ("PRECISION_PASS","PRECISION_REPAIR_REQUIRED"):
            row.update(status="BASELINE_PRECISION_UNRESOLVED")
            results.append(row)
            save(args.output/"advance.json",dict(models=results,status="IN_PROGRESS",plan=str(PLAN)))
            save(args.output/"next-phase-plan.json",dict(status="BASELINE_PRECISION_UNRESOLVED",model=name,completed=results,next="continue independent scopes; repair missing precision evidence"))
            continue
        for phase in phases:
            ledger,left=campaign_remaining(args.campaign_ledger)
            remaining=min(left,args.budget_seconds-(time.monotonic()-started))
            if left<=0:
                row.update(status="BUDGET_EXHAUSTED",stopped_phase=phase); break
            if campaign_attempts_remaining(ledger)<=0:
                row.update(status="ATTEMPT_LIMIT_REACHED",stopped_phase=phase); break
            if remaining<=0:
                row.update(status="PHASE_BUDGET_EXHAUSTED",stopped_phase=phase); break
            child=argparse.Namespace(**vars(args))
            child.command="run"; child.model=name
            scope_execution=scopes.get(name,{})
            child.shared_gaussian_contraction=scope_execution.get(
                "contraction",getattr(args,"shared_gaussian_contraction",None))
            child.runtime_precision=scope_execution.get(
                "runtime_precision",getattr(args,"runtime_precision","uniform"))
            child.tuning_seed_offset=scope_execution.get(
                "tuning_seed_offset",getattr(args,"tuning_seed_offset",0))
            child.filter_tuning=phase=="filter-tuning"
            family=scope_execution.get("control_family")
            smoother=(family=="smoother_reset_1_2_4_8" if family is not None
                      else prior["status"]=="PRECISION_REPAIR_REQUIRED")
            child.smoother_reset=child.filter_tuning and smoother
            child.hmc_mechanics=phase=="mechanics"
            child.consumer_cost=phase=="consumer-cost"
            child.consumer_baseline=getattr(args,"consumer_baseline",None) if child.consumer_cost else None
            child.output=args.output/name/phase
            child.selection=args.output/name/"filter-tuning"/name/"filter-selection.json"
            child.budget_seconds=remaining
            phase_cap = (scope_execution.get("tuning_timeout_seconds", args.per_model_timeout)
                         if child.filter_tuning else getattr(args, "consumer_timeout", 2400.)
                         if child.consumer_cost else getattr(args, "mechanics_timeout", 1200.))
            child.per_model_timeout=min(phase_cap,remaining)
            save(args.output/"next-phase-plan.json",dict(status="RUNNING",model=name,phase=phase,
                completed=results,remaining_seconds=remaining,execution_controls=scope_execution,
                next="finish own-scope tuning, mechanics, then requested consumer cost with the same fresh selection"))
            code=run(child)
            result_path=child.output/name/"result.json"
            outcome=json.loads(result_path.read_text()) if result_path.exists() else {}
            row[phase]=dict(status=outcome.get("status","MISSING_RESULT"),result=str(result_path))
            if code!=0:
                child_plan=child.output/"next-phase-plan.json"
                child_state=json.loads(child_plan.read_text()) if child_plan.is_file() else {}
                timed_out=(outcome.get("status")=="TIMEOUT" or
                    any(r.get("status")=="TIMEOUT" for r in child_state.get("completed",[])))
                row["status"]=(child_state.get("stop_reason") or
                    (("TUNING_TIME_LIMIT_REACHED" if child.filter_tuning else
                       "CONSUMER_COST_TIME_LIMIT_REACHED" if child.consumer_cost else "MECHANICS_TIME_LIMIT_REACHED")
                     if timed_out else
                     ("TUNING_REPAIR_REQUIRED" if child.filter_tuning else
                      "CONSUMER_COST_REPAIR_REQUIRED" if child.consumer_cost else "MECHANICS_REPAIR_REQUIRED")))
                row["stopped_phase"]=phase
                break
            if child.hmc_mechanics:
                row["status"]="HMC_MECHANICS_COMPLETE_POSTERIOR_NOT_RUN"
            if child.consumer_cost:
                row["status"]=success_status
        results.append(row)
        ledger,left=campaign_remaining(args.campaign_ledger)
        attempts_left=campaign_attempts_remaining(ledger)
        save(args.output/"advance.json",dict(models=results,wall_seconds=time.monotonic()-started,
            remaining_campaign_seconds=left,remaining_campaign_attempts=attempts_left,plan=str(PLAN),
            status="IN_PROGRESS" if len(results)<len(models) else "BOUNDED_PHASES_FINISHED"))
        save(args.output/"next-phase-plan.json",dict(status=row.get("status","INCOMPLETE"),completed=results,
            remaining_campaign_seconds=left,remaining_campaign_attempts=attempts_left,
            next="continue remaining independent scopes, then assess numerical repairs and measured HMC cost"))
    all_pass=len(results)==len(models) and all(r.get("status")==success_status for r in results)
    ledger,left=campaign_remaining(args.campaign_ledger)
    save(args.output/"advance.json",dict(models=results,wall_seconds=time.monotonic()-started,
        remaining_campaign_seconds=left,remaining_campaign_attempts=campaign_attempts_remaining(ledger),
        plan=str(PLAN),status=success_status if all_pass else "REPAIR_OR_BUDGET_REQUIRED"))
    # Keep the last specific phase status for existing readers, but never let
    # the final successful model hide repairs needed by an earlier model.
    next_path=args.output/"next-phase-plan.json"
    next_state=json.loads(next_path.read_text()) if next_path.is_file() else {}
    next_state.update(overall_status=success_status if all_pass else "REPAIR_OR_BUDGET_REQUIRED",
        completed=results,remaining_campaign_seconds=left,
        remaining_campaign_attempts=campaign_attempts_remaining(ledger),
        unresolved_models=[r["model"] for r in results if r.get("status")!=success_status],
        next=("assess paired consumer cost; posterior work remains separately unfinished" if all_pass else
              "repair unresolved scopes within the remaining budget; preserve completed independent evidence"))
    save(next_path,next_state)
    return 0 if all_pass else 1


def regression(args):
    """CPU-only consumer/call-chain tests, isolated from GPU imports."""
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    os.environ["TF_NUM_INTRAOP_THREADS"]="4"
    os.environ["TF_NUM_INTEROP_THREADS"]="2"
    os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"]="0"
    started=time.monotonic()
    # Isolate TensorFlow compiler state by module. A failed module is preserved
    # and does not prevent independent regression evidence from being collected.
    files=[
        "tests/test_surrogate_hmc_five_model_master.py",
        "tests/test_surrogate_hmc_performance_repair.py",
        "tests/test_neural_force_hmc.py",
        "tests/test_neural_force_hmc_reuse.py",
        "tests/test_surrogate_hmc_consumer_cost_diagnostic.py",
        "tests/test_surrogate_hmc_component_cost_diagnostic.py",
        "tests/test_surrogate_hmc_precision_islands_diagnostic.py",
        "tests/test_surrogate_hmc_mixed_runtime.py",
        "tests/highdim/test_covariance_proposal.py",
        "tests/highdim/test_covariance_proposal_precision_guard.py",
        "tests/highdim/test_covariance_proposal_packed_score.py",
        "tests/highdim/test_covariance_proposal_tuning_callchain.py",
        "tests/highdim/test_covariance_proposal_shared_gaussian.py",
        "tests/test_surrogate_hmc_covariance_cost_diagnostic.py"]
    records=[]
    for index, path in enumerate(files):
        remaining=args.budget_seconds-(time.monotonic()-started)
        if remaining<=0:
            break
        command=[sys.executable,"-m","pytest","-q",path]
        log_path=args.output/(str(index).zfill(2)+"-"+Path(path).stem+".log")
        begin=time.monotonic()
        with log_path.open("w") as log:
            try:
                child=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                     timeout=remaining)
                status="PASS" if child.returncode==0 else "FAIL"
                exit_code=child.returncode
            except subprocess.TimeoutExpired:
                status="TIMEOUT"; exit_code=None
        records.append(dict(module=path,status=status,exit_code=exit_code,
                            wall_seconds=time.monotonic()-begin,command=command,log=str(log_path)))
        save(args.output/"result.json",dict(status="IN_PROGRESS",cpu_only=True,
            isolation="one Python process per test module",modules=records,
            wall_seconds=time.monotonic()-started))
    status=("PASS" if len(records)==len(files) and all(r["status"]=="PASS" for r in records)
            else "TIMEOUT" if len(records)<len(files) or any(r["status"]=="TIMEOUT" for r in records)
            else "FAIL")
    save(args.output/"result.json",dict(status=status,cpu_only=True,
        isolation="one Python process per test module",modules=records,
        wall_seconds=time.monotonic()-started))
    return 0 if status=="PASS" else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("worker","run","regression","precision","advance"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",choices=MODELS)
    parser.add_argument("--shared-gaussian-contraction",choices=("direct","matrix"))
    parser.add_argument("--filter-tuning",action="store_true")
    parser.add_argument("--smoother-reset",action="store_true")
    parser.add_argument("--hmc-mechanics",action="store_true")
    parser.add_argument("--consumer-cost",action="store_true",help="same-input consumer timing diagnostic")
    parser.add_argument("--consumer-paired-only",action="store_true",help="run only paired actual force after prior consumer localization")
    parser.add_argument("--consumer-baseline",type=Path,help="optional preserved complete batch source for paired actual-consumer diagnostic")
    parser.add_argument("--consumer-cost-after-mechanics",action="store_true",
                        help="advance continues each successful scope to same-input consumer cost")
    parser.add_argument("--consumer-timeout",type=float,default=2400.,
                        help="advance consumer cost wall cap, clipped to remaining budget")
    parser.add_argument("--force-health-replay",type=Path,help="old selection as diagnostic inputs only; inspect actual force endpoint")
    parser.add_argument("--covariance-baseline",type=Path,help="preserved map source for paired full-horizon implementation diagnostic")
    parser.add_argument("--component-cost",action="store_true",help="unchanged primitive timing on first observation; diagnostic only")
    parser.add_argument("--transport-cache-probe",action="store_true",help="synthetic fixed-kernel cache parity and cost probe; diagnostic only")
    parser.add_argument("--transport-combined-probe",action="store_true",help="one-reduction logsumexp/softmax synthetic probe; diagnostic only")
    parser.add_argument("--transport-tile-batch-probe",action="store_true",help="exact-tile batched reduction primitive; diagnostic only")
    parser.add_argument("--force-reuse",action="store_true",help="paired cached/uncached force mechanics diagnostic")
    parser.add_argument("--selection",type=Path)
    parser.add_argument("--precision-results",type=Path)
    parser.add_argument("--execution-plan",type=Path,help="frozen per-scope contraction and tuning time caps for advance")
    parser.add_argument("--unit",action="store_true")
    parser.add_argument("--unit-horizon",type=int,default=2,help="CPU-only recursive diagnostic horizon, at most 10")
    parser.add_argument("--gpu",default="0")
    parser.add_argument("--budget-seconds",type=float,default=3600.)
    parser.add_argument("--campaign-ledger",type=Path,help="existing cumulative GPU budget; failures and compilation count")
    parser.add_argument("--per-model-timeout",type=float,default=900.)
    parser.add_argument("--mechanics-timeout",type=float,default=1200.,help="advance mechanics phase wall cap, clipped to the cumulative budget")
    parser.add_argument("--correction-steps",type=int,default=0)
    parser.add_argument("--offset",type=float,default=0.)
    parser.add_argument("--batch-size",type=int,choices=(1,2),default=2)
    parser.add_argument("--paired-repeats",type=int,choices=(0,1,2),default=2)
    parser.add_argument("--fd-sir",action="store_true",help="bounded full-scope FP32 finite-value score diagnostic")
    parser.add_argument("--fd-ladder",action="store_true",help="CPU-only finite-difference width ladder and per-step trace")
    parser.add_argument("--score-only",action="store_true",help="single full-size candidate evaluation; diagnostic only")
    parser.add_argument("--runtime-precision",choices=("uniform","fp64_recursion_fp32_pairs"),
                        default="uniform",help="explicit optional arithmetic scope; uniform FP32 remains default")
    parser.add_argument("--tuning-seed-offset",type=int,default=0,
                        help="fresh partition offset, bound into the tuning scope")
    parser.add_argument("--precision-islands",choices=("both_fp32","gaussian_fp32","transport_fp32"),
        help="diagnostic FP64 recurrence with centered FP32 pair kernels; never tuning or HMC admission")
    parser.add_argument("--profile",action="store_true",help="capture one warmed score-only GPU profile and optimized HLO")
    parser.add_argument("--fp64-reference",action="store_true",help="FP64 diagnostic exception; requires --score-only or --fd-sir")
    parser.add_argument("--debug-horizon",type=int,default=0,help="N4032 prefix localization only; no full-scope admission")
    parser.add_argument("--fd-widths",type=float,nargs="+",default=[.005,.001])
    parser.add_argument("--component-sequence", action="store_true",
        help="full-horizon staged diagnostic; requires --component-cost")
    args=parser.parse_args()
    if args.component_sequence and not args.component_cost:
        parser.error("--component-sequence requires --component-cost")

    if args.tuning_seed_offset<0:
        parser.error("tuning seed offset must be nonnegative")
    if args.runtime_precision!="uniform" and (args.command not in ("run","worker")
            or not args.model or not (args.filter_tuning or args.hmc_mechanics or args.score_only or args.component_cost or args.covariance_baseline or args.force_health_replay or args.consumer_cost or args.force_reuse)
            or args.precision_islands or args.fp64_reference or args.fd_sir
            or args.profile):
        parser.error("mixed runtime precision requires one supported full-scope run/worker")
    if args.precision_islands and (args.command not in ("run","worker") or not args.model
            or not args.score_only or not args.fp64_reference or args.unit or args.fd_sir
            or args.fd_ladder or args.profile or args.filter_tuning or args.hmc_mechanics
            or args.consumer_cost or args.force_reuse or args.selection):
        parser.error("precision islands requires one run/worker score-only FP64-storage diagnostic")
    if args.command=="worker" and not args.model: parser.error("worker requires --model")
    if args.campaign_ledger and (args.unit or args.command not in ("run","precision","advance")):
        parser.error("--campaign-ledger applies only to GPU run orchestration")
    if (args.filter_tuning or args.hmc_mechanics) and args.command not in ("run","worker"):
        parser.error("tuning/mechanics flags require run or worker")
    if args.command=="advance" and args.selection:
        parser.error("advance issues its own fresh selections")
    if sum((args.filter_tuning,args.hmc_mechanics))>1:
        parser.error("tuning and mechanics are separate workers")
    if (args.filter_tuning or args.hmc_mechanics or args.command=="advance") and (
            args.unit or args.fd_sir or args.fd_ladder or args.profile or args.score_only
            or args.fp64_reference or args.debug_horizon or args.offset):
        parser.error("tuning/mechanics require the full scope without diagnostic overrides")
    if args.smoother_reset and not args.filter_tuning:
        parser.error("--smoother-reset requires --filter-tuning")
    if args.hmc_mechanics and not args.selection:
        parser.error("--hmc-mechanics requires --selection")
    if args.execution_plan and (args.command!="advance" or args.shared_gaussian_contraction
            or args.runtime_precision!="uniform" or args.tuning_seed_offset):
        parser.error("--execution-plan belongs to advance and cannot be combined with global scope overrides")
    if args.consumer_cost_after_mechanics and args.command!="advance":
        parser.error("--consumer-cost-after-mechanics belongs to advance")
    if args.command=="advance" and not (args.precision_results and args.campaign_ledger):
        parser.error("advance requires precision evidence and the cumulative campaign ledger")
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
    if any(not math.isfinite(v) or v<=0 for v in (args.budget_seconds,args.per_model_timeout,args.mechanics_timeout,args.consumer_timeout)): parser.error("budgets must be finite and positive")
    if not math.isfinite(args.offset): parser.error("--offset must be finite")
    if args.correction_steps<0: parser.error("--correction-steps must be nonnegative")
    if args.force_health_replay and (args.command not in ("run","worker") or not args.model
            or args.unit or args.selection or args.hmc_mechanics or args.filter_tuning
            or args.score_only or args.component_cost or args.covariance_baseline
            or args.fp64_reference or args.fd_sir or args.fd_ladder or args.profile
            or args.consumer_cost or args.force_reuse or args.debug_horizon or args.offset
            or args.batch_size!=2 or args.correction_steps or args.precision_islands):
        parser.error("force health replay is a separate full-horizon B2 diagnostic")
    if args.force_health_replay and not args.force_health_replay.is_file():
        parser.error("diagnostic force input selection must exist")
    if args.covariance_baseline and (args.command not in ("run","worker") or not args.model or args.unit
            or args.selection or args.hmc_mechanics or args.filter_tuning or args.score_only
            or args.component_cost or args.fp64_reference or args.fd_sir or args.fd_ladder
            or args.profile or args.consumer_cost or args.force_reuse or args.debug_horizon
            or args.offset or args.batch_size!=2 or args.correction_steps or args.precision_islands):
        parser.error("covariance comparison is a separate full-horizon B2 run/worker diagnostic")
    if args.covariance_baseline and not args.covariance_baseline.is_file():
        parser.error("preserved covariance baseline source must exist")
    if args.component_cost and (args.command not in ("run","worker") or args.unit or
            args.selection or args.hmc_mechanics or args.filter_tuning or args.score_only or
            args.fp64_reference or args.fd_sir or args.fd_ladder or args.profile or
            args.consumer_cost or args.force_reuse or args.debug_horizon or args.offset or
            args.batch_size!=2 or args.correction_steps or args.precision_islands):
        parser.error("component cost is a separate B2 first-observation diagnostic in run/worker mode")
    if (args.transport_cache_probe or args.transport_combined_probe or args.transport_tile_batch_probe) and (args.command not in ("run","worker") or not args.model or args.unit or args.selection or args.hmc_mechanics or args.filter_tuning or args.score_only or args.component_cost or args.covariance_baseline or args.fp64_reference or args.fd_sir or args.fd_ladder or args.profile or args.consumer_cost or args.force_reuse or args.force_health_replay or args.debug_horizon or args.offset or args.batch_size!=2 or args.correction_steps or args.precision_islands or args.runtime_precision!="uniform" or sum((args.transport_cache_probe,args.transport_combined_probe,args.transport_tile_batch_probe))!=1):
        parser.error("transport primitive probe requires one separate FP32 synthetic run")
    if args.consumer_baseline and (not (args.consumer_cost or
            (args.command=="advance" and args.consumer_cost_after_mechanics)) or not args.consumer_baseline.is_file()):
        parser.error("consumer baseline requires a consumer phase and an existing complete batch source")
    if args.consumer_paired_only and (args.command not in ("run","worker") or
            not args.consumer_cost or not args.consumer_baseline):
        parser.error("paired-only requires consumer-cost and preserved baseline in run/worker mode")
    if args.consumer_cost and (args.command not in ("run","worker") or args.unit or
            not args.model or not args.selection or args.hmc_mechanics or args.filter_tuning or
            args.score_only or args.fp64_reference or args.fd_sir or args.profile or args.force_reuse):
        parser.error("consumer cost requires one model and a fresh selection in run/worker mode")
    if args.force_reuse and (args.command not in ("run","worker") or args.unit or
            not args.model or not args.selection or args.hmc_mechanics or args.filter_tuning or
            args.score_only or args.fp64_reference or args.fd_sir or args.profile or
            args.consumer_cost or args.debug_horizon):
        parser.error("force reuse requires one full-scope model and a fresh selection in run/worker mode")
    if args.command=="precision":
        if args.unit or args.fd_sir or args.profile or args.debug_horizon or args.fp64_reference:
            parser.error("precision requires the full-scope paired GPU mode")
        return precision(args)
    if args.command=="advance": return advance(args)
    if args.command=="regression": return regression(args)
    return worker(args) if args.command=="worker" else run(args)


if __name__=="__main__":
    sys.exit(main())
