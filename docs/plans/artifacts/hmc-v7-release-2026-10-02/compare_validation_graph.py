"""Diagnostic comparison of metadata predicates; no sampler or runtime edits."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--evidence", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--gpu", help="Explicit trusted GPU diagnostic; omit for CPU reference")
args = parser.parse_args()
assert os.environ.get("CUDA_VISIBLE_DEVICES") == (args.gpu or "-1")
assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") == "true"
sys.path.insert(0, str(args.source.resolve()))
from scripts.run_hmc_v7_release_prices import check_source
signature = check_source(args.source.resolve())
import tensorflow as tf
memory = None
if args.gpu:
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
from bayesfilter.inference import hmc_verification as verification
from bayesfilter.inference.hmc_candidate_set_execution import (
    HMCCandidateExecutionBinding, HMCCandidateExecutionConfig,
    _tensor_from_payload, _trace_from_payload)
from bayesfilter.inference.hmc_candidate_set_tuning import HMCWorkItem, _json_native_sha256
from bayesfilter.inference.hmc_acceptance_trials import _assemble_trials

original = verification._all_close
cases = []
for rtol, atol in ((0., 1e-12), (1e-12, 1e-12)):
    for expected in (0., .1, .2, .3, .5, 1., 65., 1e-10, 1e10, 1e100, -1., -.3, -65.):
        width = atol + rtol*abs(expected)
        for sign in (-1., 1.):
            edge = expected + sign*width
            for actual in (expected, math.nextafter(edge, -math.inf), edge,
                           math.nextafter(edge, math.inf)):
                cases.append(("scalar_boundary", actual, expected, rtol, atol))
cases.extend([
    ("vector", [0., .1, 1., -1.], [0., .1, 1., -1.], 1e-12, 1e-12),
    ("vector_change", [0., .1, 1., -1.], [0., .1, 1.+1e-10, -1.], 1e-12, 1e-12),
    ("scalar_broadcast", [1., 1., 1., 1.], 1., 1e-12, 1e-12),
    ("matrix_broadcast", [[1., 2.], [1., 2.]], [1., 2.], 1e-12, 1e-12),
    ("empty", [], [], 1e-12, 1e-12),
    ("nan", math.nan, 0., 1e-12, 1e-12),
    ("same_inf", math.inf, math.inf, 1e-12, 1e-12),
    ("opposite_inf", math.inf, -math.inf, 1e-12, 1e-12),
    ("overflow_difference", -1e308, 1e308, 1e-12, 1e-12),
    ("invalid_shape", [1., 2.], [1., 2., 3.], 1e-12, 1e-12),
])


def result_of(function, case):
    _, actual, expected, rtol, atol = case
    try:
        return dict(value=function(actual, expected, rtol=rtol, atol=atol))
    except Exception as error:
        return dict(error_type=type(error).__name__)


def candidate(jit_compile):
    @tf.function(input_signature=[tf.TensorSpec(None, tf.float64),
        tf.TensorSpec(None, tf.float64), tf.TensorSpec([], tf.float64),
        tf.TensorSpec([], tf.float64)], autograph=False, jit_compile=jit_compile)
    def predicate(actual, expected, rtol, atol):
        return tf.reduce_all(tf.abs(actual-expected) <= atol+rtol*tf.abs(expected))

    def call(actual, expected, *, rtol, atol):
        return bool(predicate(verification._float64_tensor(actual),
            verification._float64_tensor(expected),
            tf.convert_to_tensor(rtol, dtype=tf.float64),
            tf.convert_to_tensor(atol, dtype=tf.float64)))
    return call, predicate


references = [result_of(original, case) for case in cases]
rows = []
variants = {}
graph_devices = {}
for use_xla in (True, False):
    function, graph = candidate(use_xla)
    before = time.monotonic()
    comparisons = [result_of(function, case) for case in cases]
    mismatches = [dict(index=i, case=cases[i][0], reference=a, actual=b)
        for i,(a,b) in enumerate(zip(references, comparisons)) if a != b]
    rows.append(dict(jit_compile=use_xla, cases=len(cases), mismatches=mismatches,
        boundary_parity=not mismatches, wall_seconds=time.monotonic()-before,
        trace_count=graph.experimental_get_tracing_count()))
    if not mismatches:
        variants[use_xla] = function
        graph_devices[str(use_xla)] = graph(tf.constant(1., tf.float64),
            tf.constant(1., tf.float64), tf.constant(1e-12, tf.float64),
            tf.constant(1e-12, tf.float64)).device

evidence = json.loads(args.evidence.read_text())
assert _json_native_sha256(evidence) == args.evidence.stem
spec = json.loads((args.evidence.parent.parent/"execution_spec.json").read_text())["execution"]
runtime = SimpleNamespace(config=HMCCandidateExecutionConfig.from_payload(spec["config"]),
    initial_active_state=_tensor_from_payload(spec["initial_active_state"]),
    scope=SimpleNamespace(payload=lambda:spec["scope"]))
runtime.health_failures=lambda initial,samples,trace:HMCCandidateExecutionBinding.health_failures(runtime,initial,samples,trace)
runtime.analyze_trial=lambda initial,samples,trace:HMCCandidateExecutionBinding.analyze_trial(runtime,initial,samples,trace)
work = HMCWorkItem.from_payload(evidence["work"])
chunks = evidence["chunks"]
decoded = [(_tensor_from_payload(c["samples"]),_trace_from_payload(c["trace"])) for c in chunks]
baseline = _assemble_trials(runtime,work,chunks,decoded_chunks=decoded)
before = time.monotonic()
baseline = _assemble_trials(runtime,work,chunks,decoded_chunks=decoded)
baseline_seconds = time.monotonic()-before
wire = lambda value: json.dumps(value,sort_keys=True,allow_nan=False)
for row in rows:
    flag = row["jit_compile"]
    if flag not in variants:
        continue
    try:
        verification._all_close = variants[flag]
        before = time.monotonic()
        actual = _assemble_trials(runtime,work,chunks,decoded_chunks=decoded)
        row["assembly_seconds"] = time.monotonic()-before
        row["full_record_parity"] = wire(actual) == wire(baseline)
    except Exception as error:
        row.update(full_record_parity=False, assembly_error=type(error).__name__+": "+str(error))
    finally:
        verification._all_close = original
placement_comparisons = []
if args.gpu:
    assert all("GPU:0" in device for device in graph_devices.values())
    with tf.device("/GPU:0"):
        native_decoded = [(tf.identity(samples), tf.nest.map_structure(tf.identity, trace))
            for samples, trace in decoded]
    for placement, values in (("archive_decoded", decoded), ("native_device", native_decoded)):
        reference = _assemble_trials(runtime, work, chunks, decoded_chunks=values)
        comparison = dict(placement=placement, input_device=values[0][0].device, timings=[])
        # Alternating order limits one obvious timing confound; these remain
        # descriptive saved-input measurements, not a full-search forecast.
        for order in ((None, True, False), (False, None, True), (True, False, None)):
            for variant in order:
                if variant is not None and variant not in variants:
                    continue
                try:
                    verification._all_close = original if variant is None else variants[variant]
                    before = time.monotonic()
                    actual = _assemble_trials(runtime, work, chunks, decoded_chunks=values)
                    elapsed = time.monotonic()-before
                    comparison["timings"].append(dict(variant="baseline" if variant is None else str(variant),
                        seconds=elapsed, full_record_parity=wire(actual) == wire(reference)))
                finally:
                    verification._all_close = original
        placement_comparisons.append(comparison)
check_source(args.source.resolve())
result = dict(status="completed_reference_comparison", comparisons=rows,
    source_manifest_sha256=signature, evidence_sha256=hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
    input_evidence=str(args.evidence.resolve()), trial_count=len(baseline),
    baseline_assembly_seconds=baseline_seconds, gpu_intentionally_hidden=not bool(args.gpu),
    gpu_uuid=args.gpu, memory_policy=memory, graph_devices=graph_devices,
    placement_comparisons=placement_comparisons,
    tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
    numerical_sampling=False, runtime_changed=False, release_ready=False,
    interpretation="Metadata-predicate and saved-record comparison on the recorded placements. Graph-only arm is diagnostic; no cross-placement equivalence, full-price saving or runtime adoption.",
    plan="docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md")
with (args.output/"result.json").open("x") as out:
    out.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
with (args.output/"runner.py").open("xb") as out:
    out.write(Path(__file__).read_bytes())
print(json.dumps(result))
