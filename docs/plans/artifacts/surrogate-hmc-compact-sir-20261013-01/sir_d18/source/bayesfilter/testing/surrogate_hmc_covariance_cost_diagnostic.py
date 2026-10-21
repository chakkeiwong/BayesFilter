"""Paired diagnostic of a compact covariance implementation and preserved source.

Both filters use the actual shared all-parameter recurrence. Only build_maps
is replaced while tracing the independent baseline graph; no runtime fallback,
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
    return dict(passed=not bool(tf.reduce_any(newly_invalid).numpy()),
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
    def build():
        f = batch.make_batched_score_filter(
            case.spec, n, case.horizon, case.controls, case.force_dtype,
            jit_compile=True, batch_size=2, precision_policy=case.precision_policy)
        # Resolve the global call chain now; restoring the module afterwards
        # cannot switch the already traced baseline to the candidate.
        f.get_concrete_function()
        return f
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
    current_path = Path(batch.build_maps.__code__.co_filename)
    result = dict(schema="surrogate_hmc_compact_covariance_comparison_v1",
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
        first_call_seconds={}, warmed_seconds={"baseline": [], "candidate": []})
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
    save()
    if not result["comparison"]["passed"]:
        result["status"] = "COVARIANCE_IMPLEMENTATION_PARITY_FAILED"; save()
        return result
    jobs = (("baseline", baseline), ("candidate", candidate))
    for repeat in range(repetitions):
        for name, function in jobs[::1 if repeat%2==0 else -1]:
            answer, seconds = _timed(function, args)
            result["warmed_seconds"][name].append(seconds)
            # Exact deterministic replay is required within each traced graph.
            for key in ("value","score","valid","steps_completed"):
                tf.debugging.assert_equal(answer[key],values[name][key])
            save()
    # The range mechanics failed before sampling at its prescribed starts.
    # Reuse these same graphs to distinguish a new implementation failure
    # from a failure already present in the protected baseline.
    if case.name == "range_bearing":
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
