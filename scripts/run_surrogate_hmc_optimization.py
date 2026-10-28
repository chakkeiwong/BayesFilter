#!/usr/bin/env python3
"""Bounded GPU equivalence/cost campaign; no posterior or trained-map promotion.

Run this file without arguments to execute/resume the reviewed six-hour plan.
Every worker has a fresh process and versioned directory. The supervisor does
not import TensorFlow, so compiler/allocator state cannot leak between arms.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib
import importlib.util
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
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "docs/plans/artifacts/surrogate-hmc-optimization-20261021-01"
PLAN = "docs/plans/surrogate-hmc-optimization-implementation-2026-10-21.md"
MODELS = ("lgssm", "ksc", "predator_prey", "sir_d18", "range_bearing")
FILTER_FILES = (
    "bayesfilter/highdim/ledh_range_bearing_target_tf.py",
    "bayesfilter/highdim/covariance_proposal_native_models_tf.py",
    "bayesfilter/highdim/covariance_proposal_score_batch_tf.py",
)


def ready(value):
    if hasattr(value, "numpy"):
        return ready(value.numpy())
    if hasattr(value, "tolist"):
        return ready(value.tolist())
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(ready(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_baseline():
    """Load preserved authorities, verifying content before importing consumers."""
    expected = read(OUTPUT / "baseline.json")["sources"]
    for relative in FILTER_FILES:
        path = OUTPUT / "baseline" / relative
        if sha(path) != expected[relative]:
            raise ValueError("preserved baseline changed: " + relative)
        name = relative[:-3].replace("/", ".")
        parent_name, attribute = name.rsplit(".", 1)
        parent = importlib.import_module(parent_name)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        setattr(parent, attribute, module)
        spec.loader.exec_module(module)


def install_candidate():
    for relative in FILTER_FILES:
        name = relative[:-3].replace("/", ".")
        parent_name, attribute = name.rsplit(".", 1)
        parent = importlib.import_module(parent_name)
        spec = importlib.util.spec_from_file_location(name, ROOT/relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        setattr(parent, attribute, module)
        spec.loader.exec_module(module)


def selection_path(model):
    if model == "range_bearing":
        return ROOT / ("docs/plans/artifacts/surrogate-hmc-range-bearing-flow16-20261016-01/"
                       "preparation/range_bearing/filter-tuning/range_bearing/filter-selection.json")
    base = ROOT / "docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01"
    group = ("literature-prior-lgssm-advance-01" if model == "lgssm"
             else "literature-prior-other-four-advance-01")
    return base / group / model / "filter-tuning" / model / "filter-selection.json"


def load_case(tf, model):
    from bayesfilter.testing import surrogate_hmc_five_model as five
    from bayesfilter.testing import surrogate_hmc_literature_priors as priors
    from bayesfilter.highdim.covariance_proposal_tf import ProposalControls
    case = five.load_case(model)
    selected = read(selection_path(model))["selected"]
    return replace(case, controls=ProposalControls(**{"proposal_policy": "original_three_branch", **selected["controls"]}),
        beta=tf.constant(selected["beta"], tf.float32), precision_policy="uniform",
        prior_choice=priors.PRIOR_ID)


def initialize_gpu():
    # GPU 1 is occupied by another workload. Bind both arms to GPU 0 by UUID.
    os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    return tf, policy


def synchronize(tf, value):
    tf.nest.map_structure(lambda x: x.numpy() if tf.is_tensor(x) else x, value)


def measured(tf, function, *args):
    start = time.monotonic()
    value = function(*args)
    synchronize(tf, value)
    return value, time.monotonic() - start


def filter_worker(tf, args, destination):
    if args.arm == "baseline":
        install_baseline()
    from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
    case = load_case(tf, args.model)
    p = case.p
    q = tf.stack([tf.zeros([p]), tf.fill([p], .002), tf.fill([p], -.002),
                  tf.linspace(-.003, .003, p)])
    theta, _ = case.chart(q)
    function = make_batched_score_filter(case.spec, 4032, case.horizon,
        case.controls, tf.float32, jit_compile=True, batch_size=4)
    inputs = (theta, case.observations, *case.fixed, case.beta)
    tf.config.experimental.reset_memory_stats("GPU:0")
    first, cold = measured(tf, function, *inputs)
    save(destination / "progress.json", dict(stage="compiled_and_executed", seconds=cold,
        valid=first["valid"], completed=first["steps_completed"]))
    times = []
    for repeat in range(3):
        result, elapsed = measured(tf, function, *inputs)
        times.append(elapsed)
    files = {}
    for key, value in result.items():
        if tf.is_tensor(value):
            path = destination / (key + ".tensor")
            tf.io.write_file(str(path), tf.io.serialize_tensor(value))
            files[key] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                              shape=value.shape.as_list(), dtype=value.dtype.name)
    actual_sources = {}
    for relative in FILTER_FILES:
        module = sys.modules.get(relative[:-3].replace("/", "."))
        path = Path(module.__file__) if module is not None else ROOT / relative
        actual_sources[relative] = dict(path=str(path), sha256=sha(path),
                                        imported=module is not None)
    result_record = dict(kind="filter", arm=args.arm, model=args.model,
        particles=4032, batch_size=4, horizon=case.horizon, scope=case.scope(),
        controls=asdict(case.controls), beta=case.beta, q=q, theta=theta,
        controls_source=str(selection_path(args.model).relative_to(ROOT)),
        controls_status="frozen comparator; no new-scope admission claimed",
        data_path=str(case.data_path.relative_to(ROOT)), data_sha256=sha(case.data_path),
        fixed_design_seed=31001 if args.model == "range_bearing" else 261008301,
        fixed_design_sha256=[hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest()
                             for x in case.fixed],
        actual_filter_sources=actual_sources, compile_and_first_seconds=cold,
        warmed_seconds=times, warmed_median_seconds=statistics.median(times),
        allocator=tf.config.experimental.get_memory_info("GPU:0"),
        value=result["value"], score=result["score"], valid=result["valid"],
        steps_completed=result["steps_completed"], tensors=files,
        repeated_value_equal=bool(tf.reduce_all(first["value"] == result["value"])),
        repeated_score_equal=bool(tf.reduce_all(first["score"] == result["score"])))
    save(destination / "result.json", result_record)


def tensor_errors(tf, left, right):
    differences, scales = [], []
    for a, b in zip(tf.nest.flatten(left), tf.nest.flatten(right)):
        a, b = tf.cast(a, tf.float64), tf.cast(b, tf.float64)
        difference = tf.reduce_max(tf.abs(a-b))
        differences.append(difference)
        scales.append(tf.reduce_max(tf.abs(a-b)/tf.maximum(1.,tf.abs(a))))
    return dict(max_absolute=tf.reduce_max(differences), max_scaled=tf.reduce_max(scales))


def timing_worker(tf, args, destination):
    """Interleaved whole-filter timing; both graphs retained, not a memory test."""
    functions = {}
    results = {}
    for arm in ("baseline", "candidate"):
        install_baseline() if arm == "baseline" else install_candidate()
        from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
        case = load_case(tf,args.model)
        q = tf.stack([tf.zeros([case.p]),tf.fill([case.p],.002),tf.fill([case.p],-.002),
                      tf.linspace(-.003,.003,case.p)])
        theta,_ = case.chart(q)
        inputs = (theta,case.observations,*case.fixed,case.beta)
        functions[arm] = make_batched_score_filter(case.spec,4032,case.horizon,
            case.controls,tf.float32,jit_compile=True,batch_size=4)
        results[arm],_ = measured(tf,functions[arm],*inputs)
    times = {a:[] for a in functions}
    order = []
    for repeat in range(6):
        for arm in (("baseline","candidate") if repeat%2 == 0 else ("candidate","baseline")):
            results[arm],elapsed = measured(tf,functions[arm],*inputs)
            times[arm].append(elapsed)
            order.append(dict(arm=arm,seconds=elapsed))
    save(destination/"result.json",dict(kind="timing",model=args.model,particles=4032,
        batch_size=4,horizon=case.horizon,warmed_seconds=times,order=order,
        median_seconds={a:statistics.median(t) for a,t in times.items()},
        errors={k:tensor_errors(tf,results["baseline"][k],results["candidate"][k])
                for k in ("value","score")},
        both_valid=bool(tf.reduce_all(results["baseline"]["valid"] & results["candidate"]["valid"])),
        independent_process_memory_comparison=False,statistical_ranking_established=False))


def training_worker(tf, args, destination):
    from bayesfilter.testing.surrogate_hmc_sir_neutra import SIRNeuTraTarget
    from bayesfilter.inference.neutra_transport import (
        NeuTraTransport, NeuTraTransportConfig, NeuTraOptimizerConfig, NeuTraTransportTrainer)
    from bayesfilter.testing import surrogate_hmc_five_model as five
    from types import SimpleNamespace
    case = load_case(tf, "sir_d18")
    target = SIRNeuTraTarget(case, 4)
    # Explicit narrow integration fixture; it is not a coverage-trained map.
    config = replace(NeuTraTransportConfig.hoffman_author_iaf(3,
        conditional_scale_cap=2., seed=(261021,41), dtype="float32"),
        affine_scale=(.02,.02,.02))
    def trainer(shard, native):
        opt = NeuTraOptimizerConfig(8,"standard",.001,.9,.999,1e-7,1.,
                                   target_dtype="float32",shard_size=shard)
        return NeuTraTransportTrainer(NeuTraTransport(config), native.value_score,
                                       opt,target_signature=native.target_signature)
    whole = trainer(None, SIRNeuTraTarget(case,8))
    sharded = trainer(4,target)
    z = tf.random.stateless_normal([8,3],[261021,51])
    tf.config.experimental.reset_memory_stats("GPU:0")
    a, ta = measured(tf,whole.evaluate,z)
    b, tb = measured(tf,sharded.evaluate,z)
    comparisons = {key:tensor_errors(tf,a[key],b[key]) for key in ("loss","gradients","proposal_score")}
    save(destination/"progress.json",dict(stage="gradient_comparison", comparisons=comparisons,
        whole_valid=a["valid"],shard_valid=b["valid"],whole_seconds=ta,shard_seconds=tb))
    if not bool(a["valid"] & b["valid"]):
        raise ValueError("integration fixture outside healthy SIR training region")
    for item in comparisons.values():
        if float(item["max_scaled"]) > .002:
            raise ValueError("complete-update equivalence needs FP32 discrepancy localization")
    update_a, tua = measured(tf,whole.train_step,z)
    update_b, tub = measured(tf,sharded.train_step,z)
    if not bool(update_a["valid"] & update_b["valid"]):
        raise ValueError("invalid complete SIR optimizer update")
    state = lambda t: (*t.variables,*t.optimizer.variables)
    comparisons["optimizer_and_parameters"] = tensor_errors(tf,state(whole),state(sharded))
    if float(comparisons["optimizer_and_parameters"]["max_scaled"]) > .002:
        raise ValueError("whole/shard optimizer-state mismatch")
    checkpoint = sharded.checkpoint()
    save(destination/"checkpoint.json",dict(trainer=checkpoint, next_noise_seed=[261021,52],
        target_scope=target.scope,logical_batch=8,shard_size=4))
    resumed = trainer(4,target)
    resumed.restore(read(destination/"checkpoint.json")["trainer"])
    z2 = tf.random.stateless_normal([8,3],[261021,52])
    second, t2 = measured(tf,sharded.train_step,z2)
    second_whole, tw2 = measured(tf,whole.train_step,z2)
    second_resumed, tr = measured(tf,resumed.train_step,z2)
    if not bool(second["valid"] & second_whole["valid"] & second_resumed["valid"]):
        raise ValueError("resumed SIR update invalid")
    comparisons["second_optimizer_and_parameters"] = tensor_errors(tf,state(whole),state(sharded))
    if float(comparisons["second_optimizer_and_parameters"]["max_scaled"]) > .002:
        raise ValueError("second whole/shard optimizer-state mismatch")
    resume_difference = tensor_errors(tf,state(sharded),state(resumed))
    if float(resume_difference["max_absolute"]) != 0.:
        raise ValueError("checkpoint resume was not exact")
    heldout = tf.random.stateless_normal([8,3],[261021,53])
    monitor, tm = measured(tf,sharded.evaluate,heldout)
    if not bool(monitor["valid"]):
        raise ValueError("heldout integration fixture invalid")
    payload = sharded.transport.as_dtype("float64").frozen_payload(
        target_signature=target.target_signature,
        training_state_hash=sharded.checkpoint()["checkpoint_hash"])
    save(destination/"integration-only-frozen-map.json",payload)
    components = five.bindings(case,neutra_payload=payload,
                               training_target_signature=target.target_signature,
                               training_target_scope=target.scope)
    factor, center = tf.eye(3,dtype=tf.float64), tf.zeros([3],tf.float64)
    adapter = SimpleNamespace(transform=SimpleNamespace(factor=factor,center=center),
                               latent_to_position=lambda x:x)
    kernel = components["binding"].build_tensor_kernel(adapter,step_size=.01,
        num_leapfrog_steps=1,target_scope=components["scope"],target_status_trace_policy="none",
        chain_execution_mode="tf_function",use_xla=True)
    @tf.function(input_signature=[tf.TensorSpec([2,3],tf.float64)],jit_compile=True,autograph=False)
    def transition(x):
        return kernel.one_step(x,kernel.bootstrap_results(x),seed=(261021,61))
    (positions, health), th = measured(tf,transition,tf.cast(z[:2],tf.float64))
    if not bool(tf.reduce_all(health.finite_status)) or bool(tf.reduce_any(health.force_fallback)):
        raise ValueError("frozen-map SIR HMC mechanics invalid")
    save(destination/"result.json",dict(kind="training",model="sir_d18",particles=4032,
        horizon=50,logical_batch=8,native_shard_size=4,training_target_scope=target.scope,
        architecture=asdict(config),comparisons=comparisons,resume_difference=resume_difference,
        checkpoint_resume_exact=True,updates=2,optimizer_iteration=sharded.optimizer.iterations,
        training_dtype="float32",hmc_map_and_coordinate_dtype="float64",
        hmc_ledh_force_dtype=case.force_dtype.name,
        whole_evaluation_cold_seconds=ta,sharded_evaluation_cold_seconds=tb,
        whole_update_cold_seconds=tua,sharded_update_cold_seconds=tub,
        sharded_update_warm_seconds=t2,whole_update_warm_seconds=tw2,
        resumed_update_cold_seconds=tr,
        heldout=monitor,heldout_seconds=tm,frozen_hmc_seconds=th,
        hmc_finite=health.finite_status,hmc_force_fallback=health.force_fallback,
        hmc_accepted=health.is_accepted,hmc_delta_h=health.delta_h,
        map_hash=payload["transport_hash"],binding=components["binding"].payload(),
        allocator=tf.config.experimental.get_memory_info("GPU:0"),
        training_quality_established=False,posterior_qualified=False,
        default_ready=False,monitor_role="finite heldout integration check only",
        sample_wise_target_loop=False))


def worker(args):
    destination = Path(args.output)
    start = time.monotonic()
    manifest = dict(command=sys.argv,git_commit=subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),python=sys.executable,
        plan=PLAN,result=str(destination/"result.json"),backend="TensorFlow GPU XLA",
        dtype="float32",tf32=False,gpu_access="escalated",jit_compile=True,
        random_seeds=dict(fixed_design=31001 if args.model == "range_bearing" else 261008301,
            training_initialization=[261021,41] if args.worker == "training" else None,
            training_noise=[[261021,51],[261021,52],[261021,53]] if args.worker == "training" else [],
            hmc_transition=[261021,61] if args.worker == "training" else None),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (
            ROOT/__file__ if not Path(__file__).is_absolute() else Path(__file__),
            *(ROOT/f for f in FILTER_FILES),
            ROOT/"bayesfilter/highdim/covariance_proposal_nonshared_gaussian_tf.py",
            ROOT/"bayesfilter/testing/surrogate_hmc_sir_neutra.py",
            ROOT/"bayesfilter/testing/surrogate_hmc_five_model.py",
            ROOT/"bayesfilter/inference/neutra_transport.py",
            ROOT/"bayesfilter/inference/neutra_transport_core.py",
            ROOT/"bayesfilter/inference/neural_force_hmc.py")})
    try:
        tf, memory = initialize_gpu()
        manifest.update(memory_policy=memory,tensorflow_version=tf.__version__,
            devices=[str(d) for d in tf.config.list_physical_devices("GPU")],
            cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
            hardware=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader"],text=True).splitlines())
        save(destination/"manifest.json",manifest)
        if args.worker == "filter":
            filter_worker(tf,args,destination)
        elif args.worker == "timing":
            timing_worker(tf,args,destination)
        else:
            training_worker(tf,args,destination)
        manifest["status"]="COMPLETE"
    except Exception:
        manifest.update(status="FAILED",error=traceback.format_exc())
        raise
    finally:
        manifest["wall_seconds"]=time.monotonic()-start
        save(destination/"manifest.json",manifest)


def compare(records):
    """Stdlib reporting only; neither ratio is a scientific promotion rule."""
    rows=[]
    for model in MODELS:
        arms={r["arm"]:r for r in records if r.get("model")==model and r.get("kind")=="filter"}
        if len(arms)!=2:
            continue
        old,new=arms["baseline"],arms["candidate"]
        delta=lambda a,b:max(abs(x-y) for x,y in zip(a,b))
        flatten=lambda x:[v for row in x for v in row]
        old_score,new_score=flatten(old["score"]),flatten(new["score"])
        finite=all(isinstance(x,(int,float)) and math.isfinite(x)
                   for x in old["value"]+new["value"]+old_score+new_score)
        row=dict(model=model,valid_status_equal=old["valid"]==new["valid"],
            completed_equal=old["steps_completed"]==new["steps_completed"],
            healthy=all(old["valid"]) and all(new["valid"]) and finite,
            baseline_warm_seconds=old["warmed_median_seconds"],
            candidate_warm_seconds=new["warmed_median_seconds"],
            baseline_peak_bytes=old["allocator"]["peak"],candidate_peak_bytes=new["allocator"]["peak"],
            data_equal=old["data_sha256"]==new["data_sha256"],
            fixed_design_equal=old["fixed_design_sha256"]==new["fixed_design_sha256"],
            controls_equal=old["controls"]==new["controls"] and old["beta"]==new["beta"])
        if finite:
            row.update(max_value_absolute_difference=delta(old["value"],new["value"]),
                max_score_absolute_difference=delta(old_score,new_score),
                max_score_scaled_difference=max(abs(x-y)/max(1.,abs(x)) for x,y in zip(old_score,new_score)))
        rows.append(row)
    return rows


def current_result(entry, kind, arm):
    """Resume only evidence produced by the current relevant numerical sources."""
    if arm == "baseline":
        expected = read(OUTPUT/"baseline.json")["sources"]
        result = read(Path(entry["output"])/"result.json")
        return all(result["actual_filter_sources"][f]["sha256"] == expected[f]
                   for f in FILTER_FILES)
    sources = read(Path(entry["output"])/"manifest.json")["source_sha256"]
    required = list(FILTER_FILES) + ["bayesfilter/highdim/covariance_proposal_nonshared_gaussian_tf.py"]
    if kind == "training":
        required += ["bayesfilter/testing/surrogate_hmc_sir_neutra.py",
            "bayesfilter/testing/surrogate_hmc_five_model.py",
            "bayesfilter/inference/neutra_transport.py",
            "bayesfilter/inference/neutra_transport_core.py",
            "bayesfilter/inference/neural_force_hmc.py"]
    return all(sources.get(f) == sha(ROOT/f) for f in required)


def verify_saved():
    """CPU-only inspection of every saved tensor, including off-diagonal OT."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    ledger = read(OUTPUT/"execution.json")
    records = [read(Path(a["output"])/"result.json") for a in ledger["attempts"]
               if a["status"] == "COMPLETE"]
    comparisons = {}
    for model in MODELS:
        arms = {r["arm"]:r for r in records if r.get("model") == model and r["kind"] == "filter"}
        if len(arms) != 2:
            continue
        fields = {}
        for key, a in arms["baseline"]["tensors"].items():
            b = arms["candidate"]["tensors"][key]
            values = []
            for item in (a,b):
                path = ROOT/item["path"]
                if sha(path) != item["sha256"]:
                    raise ValueError("saved tensor checksum changed: " + str(path))
                values.append(tf.io.parse_tensor(tf.io.read_file(str(path)),tf.as_dtype(item["dtype"])))
            x,y = values
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError("tensor shape/dtype changed: " + key)
            if x.dtype.is_floating:
                finite = tf.math.is_finite(x) & tf.math.is_finite(y)
                flat_mask = tf.reshape(finite,[-1])
                xf,yf = [tf.cast(tf.boolean_mask(tf.reshape(v,[-1]),flat_mask),tf.float64) for v in (x,y)]
                diff = tf.abs(xf-yf)
                fields[key] = dict(finite_mask_equal=tf.reduce_all(tf.math.is_finite(x)==tf.math.is_finite(y)),
                    finite_count=tf.reduce_sum(tf.cast(finite,tf.int64)),
                    max_absolute=tf.reduce_max(diff),
                    max_scaled=tf.reduce_max(diff/tf.maximum(1.,tf.abs(xf))),
                    rms=tf.sqrt(tf.reduce_mean(diff*diff)),exact=tf.reduce_all(xf==yf))
            else:
                fields[key] = dict(exact=tf.reduce_all(x==y))
        comparisons[model] = fields
    # Compare the changed masking/reduction core directly with the saved core.
    from bayesfilter.inference import neutra_transport_core as new_core
    from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig
    path = OUTPUT/"baseline/bayesfilter/inference/neutra_transport_core.py"
    expected = read(OUTPUT/"baseline.json")["sources"]["bayesfilter/inference/neutra_transport_core.py"]
    if sha(path) != expected:
        raise ValueError("preserved training core changed")
    spec = importlib.util.spec_from_file_location("diagnostic_old_neutra_core",path)
    old_core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old_core
    spec.loader.exec_module(old_core)
    core_comparisons = []
    def target(q):
        a,b = q[:,0],q[:,1]
        v = -.5*(a*a+2*b*b)-.1*(a-b)**4
        t = .4*(a-b)**3
        return v,tf.stack([-a-t,-2*b+t],-1),tf.ones([7],tf.bool)
    for dtype in (tf.float32,tf.float64):
        transport = NeuTraTransport(NeuTraTransportConfig.hoffman_author_iaf(
            2,conditional_scale_cap=2.,seed=(11,23),dtype=dtype.name))
        z = tf.random.stateless_normal([7,2],[17,29],dtype=dtype)
        for estimator in ("standard","path"):
            def evaluate(core):
                return tf.function(lambda z:core.reverse_kl_evaluate(transport,target,z,estimator=estimator),
                    input_signature=[tf.TensorSpec([7,2],dtype)],jit_compile=True,autograph=False)(z)
            a,b = evaluate(old_core),evaluate(new_core)
            core_comparisons.append(dict(dtype=dtype.name,estimator=estimator,
                valid_equal=a["valid"]==b["valid"],
                errors={k:tensor_errors(tf,a[k],b[k]) for k in ("loss","gradients","proposal_score")}))
    save(OUTPUT/"saved-tensor-equivalence.json",dict(cpu_only=True,gpu_intentionally_hidden=True,
        model_fields=comparisons,unchanged_core_comparisons=core_comparisons,
        role="arithmetic equivalence diagnostics, not posterior or map-quality evidence"))


def supervisor(only_jobs=()):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    jobs=[("filter",m,a) for m in MODELS for a in ("baseline","candidate")]
    jobs.extend(("timing",m,"candidate") for m in MODELS)
    jobs.append(("training","sir_d18","candidate"))
    if only_jobs:
        known = {f"{k}-{m}-{a}" for k,m,a in jobs}
        if not set(only_jobs) <= known:
            raise ValueError("unknown selected job")
        jobs = [(k,m,a) for k,m,a in jobs if f"{k}-{m}-{a}" in only_jobs]
    ledger_path=OUTPUT/"execution.json"
    ledger=read(ledger_path) if ledger_path.exists() else dict(budget_seconds=21600,attempts=[])
    for kind,model,arm in jobs:
        key=f"{kind}-{model}-{arm}"
        attempts=[a for a in ledger["attempts"] if a["job"]==key]
        for attempt in attempts:
            if attempt.get("status") == "COMPLETE" and not current_result(attempt,kind,arm):
                attempt.update(status="SUPERSEDED",reason="numerical sources changed; evidence retained")
                save(ledger_path,ledger)
        if any(a.get("status")=="COMPLETE" for a in attempts):
            continue
        # Recover wall time for a crashed supervisor; never leave an orphan GPU worker.
        for attempt in ledger["attempts"]:
            if attempt.get("status")=="RUNNING":
                pid=attempt["pid"]
                try:
                    os.kill(pid,0)
                except ProcessLookupError:
                    attempt.update(status="INTERRUPTED",wall_seconds=time.time()-attempt["started_epoch"])
                else:
                    raise RuntimeError(f"worker {pid} still running; inspect instead of duplicating it")
        used=sum(a.get("wall_seconds",0.) for a in ledger["attempts"])
        remaining=ledger["budget_seconds"]-used
        save(OUTPUT/"next-phase-plan.json",dict(next_job=key,remaining_seconds=remaining,
            question="finite-algorithm equivalence and measured engineering cost",
            criterion="healthy matched outputs and complete updates; timings descriptive",
            continuation_veto="corrupt baseline/input or exhausted budget",
            training_quality_claim=False))
        if remaining<120 or len(attempts)>=3:
            continue
        destination=OUTPUT/(key+f"-{len(attempts)+1:02d}")
        destination.mkdir(exist_ok=False)
        command=[sys.executable,str(Path(__file__).resolve()),"--worker",kind,"--model",model,
                 "--arm",arm,"--output",str(destination)]
        limit=min(remaining,7200 if kind=="training" else 2400)
        started=time.monotonic()
        with (destination/"worker.log").open("w") as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            entry=dict(job=key,command=command,output=str(destination),status="RUNNING",pid=process.pid,
                       started_epoch=time.time(),timeout_seconds=limit)
            ledger["attempts"].append(entry)
            save(ledger_path,ledger)
            print("START",key,flush=True)
            try:
                code=process.wait(timeout=limit)
                entry.update(status="COMPLETE" if code==0 else "FAILED",exit_code=code)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
                entry.update(status="TIMEOUT",exit_code=process.returncode)
            entry["wall_seconds"]=time.monotonic()-started
            save(ledger_path,ledger)
            print(entry["status"],key,round(entry["wall_seconds"],1),flush=True)
        records=[read(Path(a["output"])/"result.json") for a in ledger["attempts"]
                 if a["status"]=="COMPLETE"]
        save(OUTPUT/"comparison.json",dict(filter_comparisons=compare(records),
            completed_jobs=[a["job"] for a in ledger["attempts"] if a["status"]=="COMPLETE"],
            wall_seconds=sum(a.get("wall_seconds",0) for a in ledger["attempts"]),
            statistical_ranking_established=False,posterior_qualified=False))
    save(OUTPUT/"next-phase-plan.json",dict(status="BOUNDED_EXECUTION_FINISHED_REVIEW_RESULTS",
        pending_jobs=[key for key in (f"{k}-{m}-{a}" for k,m,a in jobs)
                      if not any(x["job"]==key and x["status"]=="COMPLETE" for x in ledger["attempts"])],
        remaining_seconds=ledger["budget_seconds"]-sum(a.get("wall_seconds",0) for a in ledger["attempts"]),
        next_action="review measured equivalence/cost; local repairs may resume within remaining budget"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",choices=("filter","training","timing"))
    parser.add_argument("--model",choices=MODELS)
    parser.add_argument("--arm",choices=("baseline","candidate"))
    parser.add_argument("--output")
    parser.add_argument("--verify-saved",action="store_true")
    parser.add_argument("--only-job",action="append",default=[],
                        help="run selected existing jobs under the same campaign ledger")
    args=parser.parse_args()
    if args.verify_saved:
        verify_saved()
    elif args.worker:
        if not args.output or not args.model or not args.arm:
            parser.error("workers require model, arm, and output")
        worker(args)
    else:
        supervisor(args.only_job)


if __name__ == "__main__":
    main()
