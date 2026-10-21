"""Paired diagnostic of a compact covariance implementation and preserved source.

Both filters use the actual shared all-parameter recurrence. A preserved batch
module or map implementation is traced independently; no runtime fallback,
tuning authority, posterior claim, or NumPy computation is provided here.
"""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import tensorflow as tf
from bayesfilter.highdim import covariance_proposal_score_batch_tf as batch
from bayesfilter.testing.surrogate_hmc_component_cost_diagnostic import _timed


def compare_outputs(candidate, baseline, horizon, tolerance=.01):
    healthy = all(bool(tf.reduce_all(out["valid"]).numpy()) and
                  bool(tf.reduce_all(out["steps_completed"] == horizon).numpy())
                  for out in (candidate, baseline))
    differences = {}
    for key in ("value", "score", "particles", "particle_tangent"):
        a, b = candidate[key], baseline[key]
        finite = bool(tf.reduce_all(tf.math.is_finite(a)).numpy()) and bool(
            tf.reduce_all(tf.math.is_finite(b)).numpy())
        healthy = healthy and finite
        delta = tf.abs(a-b)
        differences[key] = dict(finite=finite,
            maximum_absolute=float(tf.reduce_max(delta).numpy()),
            maximum_scaled=float(tf.reduce_max(delta/tf.maximum(tf.ones_like(b),tf.abs(b))).numpy()))
    passed = healthy and all(differences[key]["maximum_scaled"] <= tolerance
                             for key in ("value", "score"))
    return dict(passed=passed, healthy=healthy, tolerance=tolerance,
                differences=differences, primary_outputs=["value", "score"],
                particle_errors_role="explanatory")



def compare_initial_region(candidate, baseline, horizon):
    """A pre-existing bad start is an HMC veto; a newly bad start vetoes repair."""
    def health(out):
        good = tf.logical_and(out["valid"], out["steps_completed"] == horizon)
        for key in ("value", "score", "particles", "particle_tangent"):
            finite = tf.reduce_all(tf.math.is_finite(out[key]),
                                   axis=tf.range(1, tf.rank(out[key])))
            good = tf.logical_and(good, finite)
        return good
    before, after = health(baseline), health(candidate)
    newly_invalid = tf.logical_and(before, tf.logical_not(after))
    differences = {}
    equivalent = True
    for key in ("value", "score"):
        a = tf.boolean_mask(candidate[key], before & after)
        b = tf.boolean_mask(baseline[key], before & after)
        error = tf.abs(a-b)/tf.maximum(tf.ones_like(b), tf.abs(b))
        maximum = float(tf.reduce_max(tf.concat([tf.reshape(error,[-1]),tf.zeros([1],error.dtype)],0)).numpy())
        differences[key] = dict(maximum_scaled=maximum)
        equivalent = equivalent and maximum <= .01
    return dict(passed=not bool(tf.reduce_any(newly_invalid).numpy()) and equivalent,
                differences_on_mutually_healthy_rows=differences,
                arithmetic_tolerance=.01, arithmetic_passed=equivalent,
                baseline_healthy=before.numpy().tolist(),
                candidate_healthy=after.numpy().tolist(),
                newly_invalid=newly_invalid.numpy().tolist(),
                hmc_health_passed=bool(tf.reduce_all(after).numpy()))


def make_pair(case, baseline_path):
    """Trace both consumers under their actual map implementation, then restore."""
    baseline_path = Path(baseline_path)
    module_spec = importlib.util.spec_from_file_location(
        "_covariance_proposal_preserved_reference", baseline_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    original = batch.build_maps
    n = int(case.fixed[0].shape[0])
    def build(factory=batch.make_batched_score_filter):
        f = factory(
            case.spec, n, case.horizon, case.controls, case.force_dtype,
            jit_compile=True, batch_size=2, precision_policy=case.precision_policy)
        # Resolve the global call chain now; restoring the module afterwards
        # cannot switch the already traced baseline to the candidate.
        f.get_concrete_function()
        return f
    if hasattr(module, "make_batched_score_filter"):
        # A preserved complete batch module retains its own global kernel
        # bindings. The two endpoint graphs are independent after tracing.
        baseline = build(module.make_batched_score_filter)
    else:
        try:
            batch.build_maps = module.build_maps
            baseline = build()
        finally:
            batch.build_maps = original
    candidate = build()
    return baseline, candidate


def run_diagnostic(case, output, baseline_path, repetitions=3):
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    if repetitions < 1:
        raise ValueError("positive repetitions required")
    output = Path(output)
    baseline_path = Path(baseline_path)
    baseline_copy = output / "baseline_covariance_proposal_score_tf.py"
    baseline_copy.write_bytes(baseline_path.read_bytes())
    current_path = Path(batch.__file__ if "def make_batched_score_filter(" in baseline_path.read_text() else batch.build_maps.__code__.co_filename)
    batch_baseline = "def make_batched_score_filter(" in baseline_path.read_text()
    result = dict(schema="surrogate_hmc_compact_covariance_comparison_v1",
        comparison_scope="complete_batch_kernels" if batch_baseline else "build_maps_only",
        status="RUNNING", evidence_role="same_filter_implementation_comparison",
        scope=case.scope(), controls=asdict(case.controls), beta=runtime.json_ready(case.beta),
        design_seed=31001 if case.name=="range_bearing" else 261008301,
        diagnostic_horizon=case.horizon, batch_size=2,
        baseline_source=str(baseline_copy),
        baseline_source_sha256=hashlib.sha256(baseline_copy.read_bytes()).hexdigest(),
        candidate_source_sha256=hashlib.sha256(current_path.read_bytes()).hexdigest(),
        cpu_only=not bool(tf.config.list_physical_devices("GPU")),
        jit_compile=True, posterior_sampling_complete=False,
        timing_includes_materialization=True, timing_interpretation="descriptive",
        first_call_seconds={}, warmed_seconds={"baseline": [], "candidate": []},
        allocator_by_arm={"baseline": [], "candidate": []},
        allocator_interpretation="per-call high-water mark with both traced graphs and comparison outputs resident")
    def save():
        (output/"covariance-comparison.json").write_text(
            json.dumps(runtime.json_ready(result), indent=2, allow_nan=False))
    save()
    dtype = case.force_dtype
    theta = tf.stack([case.theta0, case.chart(tf.fill([case.p],tf.cast(.01,dtype)))[0]])
    args = (theta,case.observations,*case.fixed,case.beta)
    result["parameter_positions"] = runtime.json_ready(theta)
    result["input_sha256"] = [hashlib.sha256(tf.io.serialize_tensor(a).numpy()).hexdigest()
                              for a in args]
    baseline, candidate = make_pair(case, baseline_copy)
    values = {}
    for name, function in (("baseline", baseline), ("candidate", candidate)):
        values[name], seconds = _timed(function, args)
        result["first_call_seconds"][name] = seconds
        result[name] = {key: runtime.json_ready(values[name][key])
                       for key in ("value","score","valid","steps_completed","trace")}
        save()
    result["comparison"] = compare_outputs(values["candidate"],values["baseline"],case.horizon)
    for field in ("score_trace", "trace"):
        a,b=values["candidate"][field],values["baseline"][field]
        finite=tf.math.is_finite(a)&tf.math.is_finite(b)
        delta=tf.where(finite,tf.abs(a-b)/tf.maximum(tf.ones_like(b),tf.abs(b)),tf.zeros_like(a))
        result[field+"_maximum_scaled_by_time"]=runtime.json_ready(tf.reduce_max(delta,[0,2]))

    save()
    if not result["comparison"]["passed"]:
        result["status"] = "COVARIANCE_IMPLEMENTATION_PARITY_FAILED"; save()
        return result
    jobs = (("baseline", baseline), ("candidate", candidate))
    for repeat in range(repetitions):
        for name, function in jobs[::1 if repeat%2==0 else -1]:
            gpu = bool(tf.config.list_physical_devices("GPU"))
            if gpu:
                tf.config.experimental.reset_memory_stats("GPU:0")
                before = tf.config.experimental.get_memory_info("GPU:0")["current"]
            answer, seconds = _timed(function, args)
            if gpu:
                memory = tf.config.experimental.get_memory_info("GPU:0")
                result["allocator_by_arm"][name].append(dict(
                    before_bytes=before, **memory, peak_increment_bytes=memory["peak"]-before))
            result["warmed_seconds"][name].append(seconds)
            # Exact deterministic replay is required within each traced graph.
            for key in ("value","score","valid","steps_completed"):
                tf.debugging.assert_equal(answer[key],values[name][key])
            save()
    # The range mechanics failed before sampling at its prescribed starts.
    # Reuse these same graphs to distinguish a new implementation failure
    # from a failure already present in the protected baseline.
    if batch_baseline or case.name == "range_bearing":
        result["initial_region_checks"] = []
        for offsets in ((-.15,-.05),(.05,.15)):
            positions = tf.stack([case.chart(tf.fill([case.p],tf.cast(q,dtype)))[0]
                                  for q in offsets])
            records, region_values = {}, {}
            for name, function in jobs:
                out, seconds = _timed(function,(positions,*args[1:]))
                region_values[name] = out
                records[name] = dict(seconds=seconds,
                    outputs={key:runtime.json_ready(out[key]) for key in
                             ("value","score","valid","steps_completed","trace")})
            check = compare_initial_region(region_values["candidate"],region_values["baseline"],case.horizon)
            result["initial_region_checks"].append(dict(offsets=list(offsets),comparison=check,**records))
            save()
    result["trace_counts"] = [f.experimental_get_tracing_count() for f in (baseline,candidate)]
    if tf.config.list_physical_devices("GPU"):
        result["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    result["host_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["candidate_to_baseline_warm_ratio"] = (
        sum(result["warmed_seconds"]["candidate"])/sum(result["warmed_seconds"]["baseline"]))
    region_pass = all(item["comparison"]["passed"] for item in result.get("initial_region_checks",[]))
    result["initial_region_no_new_failure"] = region_pass
    result["status"] = ("COVARIANCE_IMPLEMENTATION_PARITY_PASS_COST_MEASURED" if region_pass
                        else "COVARIANCE_IMPLEMENTATION_INITIAL_REGION_VETO")
    save()
    return result


def make_consumer_pair(case, baseline_path):
    """Trace the actual consumer against both factories; never return authority."""
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    module_spec = importlib.util.spec_from_file_location(
        "_surrogate_hmc_preserved_consumer_reference", Path(baseline_path))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if not hasattr(module, "make_batched_score_filter"):
        raise ValueError("consumer comparison requires a complete batch module")
    original = runtime.make_batched_score_filter
    try:
        runtime.make_batched_score_filter = module.make_batched_score_filter
        baseline = runtime.bindings(case)["force"].function
        baseline.get_concrete_function()
    finally:
        runtime.make_batched_score_filter = original
    candidate = runtime.bindings(case)["force"].function
    candidate.get_concrete_function()
    return baseline, candidate


def run_consumer_pair(case, output, baseline_path, repetitions=3):
    """Paired real-consumer diagnostic, not a force or tuning authority."""
    import time
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    output = Path(output)
    archived = output / "consumer_baseline_batch.py"
    archived.write_bytes(Path(baseline_path).read_bytes())
    positions = tf.constant([[v]*case.p for v in (-.15, -.05, .05, .15)], tf.float64)
    result = dict(status="RUNNING", evidence_role="actual_consumer_paired_diagnostic",
        scope=case.scope(), selection_hash=case.selection_hash,
        controls=asdict(case.controls), beta=runtime.json_ready(case.beta),
        baseline_source=str(archived),
        baseline_sha256=hashlib.sha256(archived.read_bytes()).hexdigest(),
        current_batch_sha256=hashlib.sha256(Path(batch.__file__).read_bytes()).hexdigest(),
        consumer_sha256=hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest(),
        input_sha256=[hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest()
                      for x in (positions, case.observations, *case.fixed, case.beta)],
        positions=runtime.json_ready(positions), precision_policy=case.precision_policy,
        cpu_only=not bool(tf.config.list_physical_devices("GPU")), jit_compile=True,
        posterior_sampling_complete=False, canonical_admission=False,
        baseline_tuning_authority=False, timing_interpretation="descriptive",
        contention_status="requires_external_process_observation",
        cold_compile_execute_seconds={}, warmed_seconds={"baseline":[],"candidate":[]},
        allocator_by_arm={"baseline":[],"candidate":[]},
        allocator_interpretation="both traced graphs and comparison outputs resident")
    destination=output/"consumer-paired.json"
    def save():
        destination.write_text(json.dumps(result, indent=2, allow_nan=False))
    save()
    begin=time.monotonic()
    baseline,candidate=make_consumer_pair(case,archived)
    result["python_graph_trace_seconds"]=time.monotonic()-begin
    outputs={}
    def call(function):
        begin=time.monotonic()
        answer=function(positions)
        tf.nest.map_structure(lambda x:x.numpy(),answer)
        return answer,time.monotonic()-begin
    for name,function in (("baseline",baseline),("candidate",candidate)):
        answer,seconds=call(function)
        outputs[name]=answer
        result["cold_compile_execute_seconds"][name]=seconds
        result[name]=runtime.json_ready(answer._asdict())
        save()
    healthy=all(bool(tf.reduce_all(out.force_valid).numpy()) and
                bool(tf.reduce_all(tf.math.is_finite(out.force)).numpy())
                for out in outputs.values())
    a,b=outputs["candidate"].force,outputs["baseline"].force
    error=tf.abs(a-b)/tf.maximum(tf.ones_like(b),tf.abs(b))
    max_error=float(tf.reduce_max(error).numpy()) if healthy else None
    result.update(healthy=healthy,maximum_scaled_force_gap=max_error,tolerance=.01,
                  force_gap_by_chain=runtime.json_ready(tf.reduce_max(error,axis=1)) if healthy else None)
    if not healthy or max_error>.01:
        result["status"]="CONSUMER_PAIRED_PARITY_OR_HEALTH_VETO"
        save(); return result
    jobs=(("baseline",baseline),("candidate",candidate))
    for repeat in range(repetitions):
        for name,function in jobs[::1 if repeat%2==0 else -1]:
            gpu=not result["cpu_only"]
            if gpu:
                tf.config.experimental.reset_memory_stats("GPU:0")
                before=tf.config.experimental.get_memory_info("GPU:0")["current"]
            answer,seconds=call(function)
            result["warmed_seconds"][name].append(seconds)
            if gpu:
                memory=tf.config.experimental.get_memory_info("GPU:0")
                result["allocator_by_arm"][name].append(dict(before_bytes=before,**memory))
            tf.debugging.assert_equal(answer.force,outputs[name].force)
            tf.debugging.assert_equal(answer.force_valid,outputs[name].force_valid)
            save()
    result["candidate_baseline_warm_ratio"]=sum(result["warmed_seconds"]["candidate"])/sum(result["warmed_seconds"]["baseline"])
    result["status"]="CONSUMER_PAIRED_PARITY_PASS_COST_MEASURED"
    save(); return result
