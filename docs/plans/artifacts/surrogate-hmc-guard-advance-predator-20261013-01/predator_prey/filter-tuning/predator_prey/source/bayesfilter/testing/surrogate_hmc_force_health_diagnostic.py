"""Diagnostic replay of exact initial positions through the HMC force consumer.

Old selections supply reproducible inputs only, never current tuning authority.
The actual bound force is evaluated independently of a trace-exposing B2 graph.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
import resource
import time
import tensorflow as tf
from bayesfilter.testing import surrogate_hmc_five_model as runtime
from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
from bayesfilter.highdim.covariance_proposal_tf import TRACE_FIELDS, VALIDITY_FIELDS


def replay_inputs(case, selection_path):
    record = runtime.read(selection_path)
    signature = record.pop("selection_hash")
    if signature != runtime.hashed(record):
        raise ValueError("corrupt diagnostic selection")
    current = case.scope()
    for key in set(record["scope"]) | set(current):
        if key != "source_sha256" and record["scope"].get(key) != current.get(key):
            raise ValueError("diagnostic input scope mismatch: " + key)
    if record["status"] != "FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC" or not record["selected"]["valid"]:
        raise ValueError("diagnostic replay requires a selected input record")
    case.controls = runtime.ProposalControls(**record["selected"]["controls"])
    case.beta = tf.constant(record["selected"]["beta"], case.force_dtype)
    return dict(selection_path=str(selection_path), selection_sha256=hashlib.sha256(
        Path(selection_path).read_bytes()).hexdigest(), selection_hash=signature,
        tuning_authority=False,
        source_changes={key:dict(previous=record["scope"]["source_sha256"].get(key),
                                 current=current["source_sha256"].get(key))
            for key in set(record["scope"]["source_sha256"]) | set(current["source_sha256"])
            if record["scope"]["source_sha256"].get(key) != current["source_sha256"].get(key)})


def compare_paths(consumer_force, consumer_valid, direct_force, direct_valid, tolerance=.01):
    consumer_health = consumer_valid & tf.reduce_all(tf.math.is_finite(consumer_force),axis=-1)
    direct_health = direct_valid & tf.reduce_all(tf.math.is_finite(direct_force),axis=-1)
    scaled = tf.abs(consumer_force-direct_force)/tf.maximum(tf.constant(1.,consumer_force.dtype),
                                                         tf.abs(direct_force))
    gap = tf.reduce_max(scaled)
    healthy = bool(tf.reduce_all(consumer_health & direct_health))
    agreement = bool(tf.reduce_all(consumer_health == direct_health))
    finite_gap = bool(tf.math.is_finite(gap))
    return dict(consumer_healthy=consumer_health, direct_healthy=direct_health,
        same_health=agreement, all_healthy=healthy, maximum_scaled_force_gap=gap,
        tolerance=tolerance, passed=healthy and agreement and finite_gap and float(gap)<=tolerance)


def run_diagnostic(case, output, selection_path):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    inputs = replay_inputs(case, selection_path)
    starts = tf.constant([[q]*case.p for q in (-.15,-.05,.05,.15)],tf.float64)
    result = dict(schema="surrogate_hmc_exact_force_health_v1",
        status="EVALUATING_ACTUAL_CONSUMER", evidence_role="initial_force_call_chain_diagnostic",
        scope=case.scope(), replay_inputs=inputs, starts=starts, beta=case.beta,
        controls=asdict(case.controls), posterior_sampling_complete=False, hmc_executed=False,
        jit_compile=True, cpu_only=not bool(tf.config.list_physical_devices("GPU")),
        input_sha256=[hashlib.sha256(tf.io.serialize_tensor(a).numpy()).hexdigest()
                      for a in (starts,case.observations,*case.fixed,case.beta)])
    def save():
        runtime.save(output/"force-health.json",result)
    save()
    binding = runtime.bindings(case)
    begin = time.monotonic()
    actual = binding["force"].function(starts)
    actual.force.numpy(); actual.force_valid.numpy()
    result["consumer"] = dict(force=actual.force,valid=actual.force_valid,
        first_call_seconds=time.monotonic()-begin)
    result["status"] = "EVALUATING_TRACE_CONSUMER"
    save()
    program = make_batched_score_filter(case.spec,int(case.fixed[0].shape[0]),
        case.horizon,case.controls,case.force_dtype,jit_compile=True,batch_size=2,
        precision_policy=case.precision_policy)
    @tf.function(input_signature=[tf.TensorSpec([2,case.p],tf.float64)],
                 jit_compile=True,autograph=False)
    def exposed(q):
        theta,jacobian=case.chart(tf.cast(q,case.force_dtype))
        out=program(theta,case.observations,*case.fixed,case.beta)
        force=q-tf.cast(jacobian*out["score"],tf.float64)
        valid=out["valid"] & (out["steps_completed"]==case.horizon)
        valid &= tf.reduce_all(tf.math.is_finite(force),axis=1)
        return force,valid,out
    forces,validities=[],[]
    result["direct"]=[]
    result["trace_fields"]=list(TRACE_FIELDS)
    result["validity_fields"]=list(VALIDITY_FIELDS)
    for first in (0,2):
        begin=time.monotonic()
        force,valid,out=exposed(starts[first:first+2])
        force.numpy(); valid.numpy()
        forces.append(force); validities.append(valid)
        result["direct"].append(dict(first_chain=first,force=force,valid=valid,
            seconds=time.monotonic()-begin,
            outputs={key:out[key] for key in ("value","score","valid","steps_completed","trace","validity_trace","covariance_tolerance")}))
        save()
    result["comparison"]=compare_paths(actual.force,actual.force_valid,
        tf.concat(forces,axis=0),tf.concat(validities,axis=0))
    result["status"]=("FORCE_CONSUMER_HEALTH_PASS" if result["comparison"]["passed"]
                      else "FORCE_CONSUMER_REPAIR_REQUIRED")
    result["trace_counts"]=dict(consumer=binding["force"].function.experimental_get_tracing_count(),
                                exposed=exposed.experimental_get_tracing_count())
    if tf.config.list_physical_devices("GPU"):
        result["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
    result["host_peak_rss_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save()
    return result
