"""Diagnostic cost/precision evidence, never a canonical LEDH score.

NumPy reads fixtures, synchronizes results and writes diagnostic arrays only.
All executable kernels and differentiated objectives use TensorFlow.
"""

import gc
import hashlib
import os
import subprocess
import time
import weakref
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import dual_cap_genut_primal_tf as current
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_genut_transitive import _write

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "28cbdb536"
ARTIFACT_ROOT = Path("/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917")
CASES = ((1000, 3, 4114), (10000, 18, 4122))
CONTRACT = "genut_bounded_gradient_frozen_coefficients_v2"


def _frozen_source():
    path = "bayesfilter/highdim/dual_cap_genut_primal_tf.py"
    source = subprocess.check_output(["git", "show", f"{REFERENCE_COMMIT}:{path}"], cwd=ROOT, text=True)
    module = ModuleType("_genut_bounded_cost_reference")
    exec(compile(source, "<genut-bounded-cost-reference>", "exec"), module.__dict__)  # noqa: S102
    return module, hashlib.sha256(source.encode()).hexdigest()


def _load_case(count, dimension, source_run):
    path = ARTIFACT_ROOT / f"run-{source_run:05d}/genut-capacity-arrays.npz"
    with np.load(path, allow_pickle=False) as saved, tf.device("/CPU:0"):
        inputs = tuple(tf.constant(saved[key], tf.float32) for key in ("source", "weights", "reset"))
        # The existing precision test's coefficients are frozen in FP32 and
        # cast exactly to FP64, never recomputed in another precision.
        indices = tf.reshape(tf.cast(tf.range(count * dimension), tf.float32), [count, dimension])
        coefficients = tf.cos(indices * .11)
    assert tuple(inputs[0].shape) == (count, dimension)
    args = (*inputs, coefficients)
    identities = {"saved_input_path": str(path), "saved_input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "operand_sha256": [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in args]}
    return args, identities


def _evaluate(implementation, inputs, *, jit_compile=True):
    def evaluate(source, weights, reset, coefficients):
        with tf.GradientTape() as tape:
            tape.watch((source, weights, reset))
            values = implementation(source, weights, reset, diagonal_steps=4, pairwise_steps=4)
            loss = tf.reduce_sum(tf.sin(values["particles"]) * coefficients)
        gradients = tape.gradient(loss, (source, weights, reset))
        return {**values, "loss": loss, "source_gradient": gradients[0],
                "weight_gradient": gradients[1], "reset_gradient": gradients[2]}

    return tf.function(evaluate, input_signature=[tf.TensorSpec(value.shape, value.dtype) for value in inputs],
                       jit_compile=jit_compile, autograph=False)


def _materialize(result):
    return {key: value.numpy() for key, value in result.items()}


def _save_output(directory, record, coefficients):
    path = directory / "genut-bounded-gradient-output.npz"
    np.savez_compressed(path, **record, coefficients=coefficients.numpy())
    return {"output_file": path.name, "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize("arm", ["before_graph", "after_graph", "after_xla"])
@pytest.mark.parametrize("count,dimension,source_run", CASES, ids=["1000-3", "10000-18"])
def test_bounded_gradient_cost(arm, count, dimension, source_run, request):
    args, identities = _load_case(count, dimension, source_run)
    original, original_sha256 = _frozen_source()
    implementation = original.dual_cap_genut_primal if arm == "before_graph" else current.dual_cap_genut_primal
    implementation_sha256 = (original_sha256 if arm == "before_graph"
                             else hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest())
    jit_compile = arm == "after_xla"
    gpu = os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "visible"
    directory = Path(request.config.getoption("xmlpath")).parent
    if gpu:
        tf.config.experimental.reset_memory_stats("GPU:0")
    report = {"role": "descriptive_cost_requires_independent_array_analysis", "contract": CONTRACT,
              "reference_commit": REFERENCE_COMMIT, "reference_source_sha256": original_sha256,
              "implementation_source_sha256": implementation_sha256, "arm": arm, "jit_compile": jit_compile,
              "count": count, "dimension": dimension, "source_run": source_run, **identities,
              "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(), "records": []}
    with GPUProcessMonitor(gpu) as monitor:
        report["before"] = memory_snapshot(gpu)
        begin = time.perf_counter()
        owner = _evaluate(implementation, args, jit_compile=jit_compile)
        first = _materialize(owner(*args))
        report["cold_seconds"] = time.perf_counter() - begin
        report["after_first"] = memory_snapshot(gpu)
        for repeat in range(10):
            begin = time.perf_counter()
            replay = _materialize(owner(*args))
            elapsed = time.perf_counter() - begin
            exact = all(np.array_equal(value, replay[key]) for key, value in first.items())
            report["records"].append({"repeat": repeat, "seconds": elapsed, "exact": exact})
        report["after_replay"] = memory_snapshot(gpu)
    report["device_provenance"] = monitor.payload()
    report["trace_count"] = owner.experimental_get_tracing_count()
    report["first_record_finite"] = all(np.all(np.isfinite(value)) for value in first.values())
    report["valid"] = bool(first["valid"])
    report["before_hlo"] = memory_snapshot(gpu)
    if jit_compile:
        hlo = owner.experimental_get_compiler_ir(*args)(stage="optimized_hlo")
        (directory / "genut-bounded-gradient-optimized.hlo").write_text(hlo)
        report["hlo"] = {"sha256": hashlib.sha256(hlo.encode()).hexdigest(), "bytes": len(hlo.encode())}
        del hlo
    report["after_hlo"] = memory_snapshot(gpu)
    report.update(_save_output(directory, first, args[3]))
    owner_reference = weakref.ref(owner)
    del owner, first, replay
    gc.collect()
    report["after_collection"] = memory_snapshot(gpu)
    report["python_owner_collected"] = owner_reference() is None
    _write(request, "genut-bounded-gradient-cost.json", report)
    assert report["trace_count"] == 1 and report["first_record_finite"] and report["valid"]
    assert all(row["exact"] for row in report["records"])
    assert report["python_owner_collected"]


@pytest.mark.parametrize("count,dimension,source_run", CASES, ids=["1000-3", "10000-18"])
def test_bounded_gradient_fp64_reference(count, dimension, source_run, request):
    assert os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "cpu"
    args32, identities = _load_case(count, dimension, source_run)
    args = tuple(tf.cast(value, tf.float64) for value in args32)
    original, source_sha256 = _frozen_source()
    owner = _evaluate(original.dual_cap_genut_primal, args, jit_compile=False)
    result = _materialize(owner(*args))
    replay = _materialize(owner(*args))
    checks = []
    names = ("source_gradient", "weight_gradient", "reset_gradient")
    for index, name in enumerate(names):
        operand = args[index]
        direction = tf.reshape(tf.sin(tf.cast(tf.range(tf.size(operand)), tf.float64) + .31), operand.shape)
        direction /= tf.cast(tf.size(operand), tf.float64)
        projection = float(tf.reduce_sum(tf.constant(result[name]) * direction))
        for step in (1e-4, 5e-5):
            upper, lower = list(args), list(args)
            upper[index], lower[index] = operand + step * direction, operand - step * direction
            difference = float((owner(*upper)["loss"] - owner(*lower)["loss"]) / (2 * step))
            checks.append({"gradient": name, "step": step, "projection": projection,
                           "finite_difference": difference,
                           "passed": bool(np.isclose(difference, projection, rtol=1e-6, atol=1e-6))})
    report = {"role": "independent_fp64_graph_reference", "contract": CONTRACT,
              "reference_commit": REFERENCE_COMMIT, "reference_source_sha256": source_sha256,
              "count": count, "dimension": dimension, "source_run": source_run, **identities,
              "trace_count": owner.experimental_get_tracing_count(),
              "finite": all(np.all(np.isfinite(value)) for value in result.values()),
              "exact_replay": all(np.array_equal(value, replay[key]) for key, value in result.items()),
              "finite_differences": checks,
              **_save_output(Path(request.config.getoption("xmlpath")).parent, result, args[3])}
    _write(request, "genut-bounded-gradient-fp64.json", report)
    assert report["trace_count"] == 1 and report["finite"] and report["exact_replay"]
    assert all(row["passed"] for row in checks)
