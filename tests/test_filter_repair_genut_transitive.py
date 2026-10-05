"""Frozen-source and independent diagnostics for reduced GenUT execution.

These tests cannot admit a canonical LEDH algorithm or analytical score.
NumPy is used only for independent moment checks and complete-record comparison.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import dual_cap_genut_primal_tf as candidate
from bayesfilter.highdim import genut_guided_proposal_tf as reset_module
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_kdm_auxiliary import _assert_record_equal, _json

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "072959c00"


def _frozen(name):
    path = f"bayesfilter/highdim/{name}.py"
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"],
                                     cwd=ROOT, text=True)
    module = ModuleType("_genut_execution_reference_" + name)
    sys.modules[module.__name__] = module
    exec(compile(source, "<frozen-genut-execution-reference>", "exec"), module.__dict__)  # noqa: S102
    return module, hashlib.sha256(source.encode()).hexdigest()


def _write(request, name, report):
    path = Path(request.config.getoption("xmlpath")).parent / name
    path.write_text(json.dumps(_json(report), indent=2, allow_nan=False) + "\n")


def _fixture(dtype, dimension=3):
    # Identical CPU-generated operands are supplied to all execution arms.
    with tf.device("/CPU:0"):
        return (tf.random.stateless_normal([72, dimension], [101, 102], dtype=dtype),
                tf.nn.softmax(0.3 * tf.random.stateless_normal([72], [103, 104], dtype=dtype)),
                tf.random.stateless_normal([72, dimension], [105, 106], dtype=dtype))


def _owner(implementation, inputs, jit_compile=True, **controls):
    return tf.function(lambda *args: implementation(*args, **controls),
                       input_signature=[tf.TensorSpec(x.shape, x.dtype) for x in inputs],
                       jit_compile=jit_compile, autograph=False)


def _original_mode(dtype, jit_compile):
    # 04016 preserves the frozen FP32 GPU XLA compiler abort. Its original
    # graph remains a comparator, never evidence for original-XLA execution.
    unavailable = (dtype == tf.float32 and jit_compile
                   and os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "visible")
    return jit_compile and not unavailable, not unavailable


def _graph_evidence(owner, inputs, jit_compile):
    graph = owner.get_concrete_function().graph.as_graph_def()
    operations = Counter(node.op for node in graph.node)
    for function in graph.library.function:
        operations.update(node.op for node in function.node_def)
    assert not {"PyFunc", "EagerPyFunc", "PyFuncStateless"}.intersection(operations)
    hlo = owner.experimental_get_compiler_ir(*inputs)(stage="hlo") if jit_compile else ""
    return {"operations": dict(operations), "graph_bytes": graph.ByteSize(),
            "hlo_bytes": len(hlo), "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
            "traces": owner.experimental_get_tracing_count()}


def _independent_moments(result, inputs, tolerance):
    source, weights, _ = (x.numpy() for x in inputs)
    particles = result["particles"].numpy()
    mean = weights @ source
    centered = source - mean
    covariance = (centered.T * weights) @ centered
    actual_mean = particles.mean(axis=0)
    actual_covariance = (particles - actual_mean).T @ (particles - actual_mean) / len(particles)
    np.testing.assert_allclose(actual_mean, mean, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(actual_covariance, covariance, rtol=tolerance, atol=tolerance)
    assert bool(result["valid"])
    assert float(result["maximum_pairwise_post_cap_particle_rms"]) <= 2.0
    assert float(result["maximum_coordinatewise_post_cap_absolute"]) <= 0.98 + tolerance
    return {"target_mean": mean, "target_covariance": covariance,
            "output_mean": actual_mean, "output_covariance": actual_covariance}


@pytest.mark.parametrize("dtype", [tf.float64, tf.float32], ids=["f64", "f32"])
def test_complete_reduced_correction(dtype, request):
    original, source_sha = _frozen("dual_cap_genut_primal_tf")
    tolerance = 2.e-10 if dtype == tf.float64 else 2.e-5
    report = {"reference_sha256": source_sha, "dtype": dtype.name, "records": []}
    cases = [(3, 4, 4), (3, 0, 0), (3, 4, 0), (1, 4, 4)]
    try:
        for dimension, diagonal_steps, pairwise_steps in cases:
            inputs = _fixture(dtype, dimension)
            controls = {"diagonal_steps": diagonal_steps, "pairwise_steps": pairwise_steps}
            reference_graph = _owner(original.dual_cap_genut_primal, inputs, False, **controls)
            graph_expected = reference_graph(*inputs)
            for jit_compile in (False, True):
                reference_jit, same_mode = _original_mode(dtype, jit_compile)
                reference = _owner(original.dual_cap_genut_primal, inputs, reference_jit, **controls)
                owner = _owner(candidate.dual_cap_genut_primal, inputs, jit_compile, **controls)
                actual, expected = owner(*inputs), reference(*inputs)
                row = {"dimension": dimension, "controls": controls, "jit_compile": jit_compile,
                           "inputs": inputs, "actual": actual, "expected": expected,
                           "reference_jit_compile": reference_jit,
                           "same_mode_reference_available": same_mode, "graph_reference": graph_expected}
                report["records"].append(row)
                if same_mode:
                    _assert_record_equal(actual, expected, atol=tolerance, rtol=tolerance)
                # Cross-mode equality remains a separate, mandatory gate. The
                # frozen original also changes a thresholded FP32 report across
                # modes (04006); never discard that result or relax its bound.
                try:
                    _assert_record_equal(actual, graph_expected, atol=tolerance, rtol=tolerance)
                    row["cross_mode_passed"] = True
                except AssertionError as error:
                    row.update(cross_mode_passed=False, cross_mode_error=str(error))
                _assert_record_equal(actual, owner(*inputs), atol=0., rtol=0.)
                row["independent_moments"] = _independent_moments(actual, inputs, tolerance)
                changed = (inputs[0] * tf.constant(1.01, dtype) + tf.constant(.03, dtype),
                           tf.reverse(inputs[1], [0]), inputs[2] * tf.constant(.97, dtype))
                changed_result, changed_expected = owner(*changed), reference(*changed)
                row.update(changed_inputs=changed, changed=changed_result, changed_reference=changed_expected)
                try:
                    _assert_record_equal(changed_result, changed_expected, atol=tolerance, rtol=tolerance)
                    row["changed_reference_passed"] = True
                except AssertionError as error:
                    row.update(changed_reference_passed=False, changed_reference_error=str(error))
                    if same_mode:
                        raise
                _assert_record_equal(changed_result, owner(*changed), atol=0., rtol=0.)
                _independent_moments(changed_result, changed, tolerance)
                if dimension == 1:
                    scalar_no_pair = _owner(candidate.dual_cap_genut_primal, inputs, jit_compile,
                                            diagonal_steps=4, pairwise_steps=0,
                                            pairwise_particle_rms_cap=0.)(*inputs)
                    _assert_record_equal(actual, scalar_no_pair, atol=0., rtol=0.)
                if pairwise_steps == 0 or dimension == 1:
                    assert float(actual["maximum_pairwise_pre_cap_particle_rms"]) == 0.
                    assert float(actual["maximum_pairwise_post_cap_particle_rms"]) == 0.
                    assert float(actual["minimum_pairwise_particle_cap_scale"]) == 1.
                evidence = _graph_evidence(owner, inputs, jit_compile)
                row["execution"] = evidence
                assert evidence["traces"] == 1
                assert evidence["operations"].get("StatelessWhile", 0) + evidence["operations"].get("While", 0) > 0
                _assert_record_equal(actual, owner(*inputs), atol=0., rtol=0.)
    finally:
        _write(request, f"genut-transitive-{dtype.name}.json", report)


def test_f32_cross_mode_complete_record(request):
    original, source_sha = _frozen("dual_cap_genut_primal_tf")
    inputs = _fixture(tf.float32)
    graph = _owner(original.dual_cap_genut_primal, inputs, False)(*inputs)
    reference_jit, available = _original_mode(tf.float32, True)
    original_xla = (_owner(original.dual_cap_genut_primal, inputs, reference_jit)(*inputs)
                    if available else None)
    native_xla = _owner(candidate.dual_cap_genut_primal, inputs, True)(*inputs)
    _write(request, "genut-cross-mode-record.json", {"reference_sha256": source_sha,
           "inputs": inputs, "graph_reference": graph, "original_xla": original_xla,
           "native_xla": native_xla, "role": "unresolved_cross_mode_complete_record_gate"})
    if available:
        _assert_record_equal(native_xla, original_xla, atol=2.e-5, rtol=2.e-5)
    _assert_record_equal(native_xla, graph, atol=2.e-5, rtol=2.e-5)


def test_f32_moment_localization(request):
    """Diagnostic TF32 control; it cannot replace default GPU qualification."""
    original, source_sha = _frozen("dual_cap_genut_primal_tf")
    inputs = _fixture(tf.float32)
    points, weights, _ = (x.numpy().astype(np.float64) for x in inputs)
    target_mean = weights @ points
    centered = points - target_mean
    target_covariance = (centered.T * weights) @ centered
    report = {"reference_sha256": source_sha, "inputs": inputs,
              "target_mean_fp64": target_mean, "target_covariance_fp64": target_covariance,
              "role": "TF32_accuracy_localization_not_default_qualification", "arms": []}
    prior_tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
    try:
        for tf32, jit_compile, module in (
                (True, False, original), (True, True, candidate), (False, True, candidate)):
            tf.config.experimental.enable_tensor_float_32_execution(tf32)
            for steps in (0, 4):
                owner = _owner(module.dual_cap_genut_primal, inputs, jit_compile,
                               diagonal_steps=steps, pairwise_steps=steps)
                result = owner(*inputs)
                output = result["particles"].numpy().astype(np.float64)
                mean = output.mean(axis=0)
                covariance = (output - mean).T @ (output - mean) / len(output)
                row = {"tf32": tf32, "jit_compile": jit_compile,
                       "implementation": "original" if module is original else "native",
                       "steps": steps, "result": result, "output_mean_fp64": mean,
                       "output_covariance_fp64": covariance,
                       "maximum_mean_error": float(np.max(np.abs(mean - target_mean))),
                       "maximum_covariance_error": float(np.max(np.abs(covariance - target_covariance)))}
                report["arms"].append(row)
                _assert_record_equal(result, owner(*inputs), atol=0., rtol=0.)
                assert owner.experimental_get_tracing_count() == 1
    finally:
        tf.config.experimental.enable_tensor_float_32_execution(prior_tf32)
        _write(request, "genut-f32-moment-localization.json", report)


@pytest.mark.parametrize("dtype", [tf.float64, tf.float32], ids=["f64", "f32"])
def test_actual_reset_consumer(dtype, request):
    original, source_sha = _frozen("dual_cap_genut_primal_tf")
    frozen_reset, reset_sha = _frozen("genut_guided_proposal_tf")
    frozen_reset.dual_cap_genut_primal = original.dual_cap_genut_primal
    assert reset_module.dual_cap_genut_primal is candidate.dual_cap_genut_primal
    assert candidate.CANONICAL_LEDH_ADMITTED is False
    inputs = _fixture(dtype)
    controls = {"epsilon": 2., "sinkhorn_steps": 8, "balance_steps": 8, "ridge": 1.e-5,
                    "reset_policy": "contract_e", "dual_cap_enabled": True, "trust_region_enabled": False,
                    "transport_plan_mode": "streaming", "transport_row_chunk_size": 72,
                    "transport_col_chunk_size": 72}
    tolerance = 2.e-10 if dtype == tf.float64 else 2.e-5
    report = {"reference_sha256": source_sha, "reset_reference_sha256": reset_sha,
                  "controls": controls, "inputs": inputs, "records": [], "canonical_admission": False}
    try:
        for jit_compile in (False, True):
            reference_jit, same_mode = _original_mode(dtype, jit_compile)
            owner = _owner(reset_module._restore_cloud_primal, inputs, jit_compile, **controls)
            reference = _owner(frozen_reset._restore_cloud_primal, inputs, reference_jit, **controls)
            actual, expected = owner(*inputs), reference(*inputs)
            row = {"jit_compile": jit_compile, "actual": actual, "expected": expected,
                   "reference_jit_compile": reference_jit, "same_mode_reference_available": same_mode}
            report["records"].append(row)
            try:
                _assert_record_equal(actual, expected, atol=tolerance, rtol=tolerance)
                row["reference_passed"] = True
            except AssertionError as error:
                row.update(reference_passed=False, reference_error=str(error))
                if same_mode:
                    raise
            _assert_record_equal(actual, owner(*inputs), atol=0., rtol=0.)
            assert bool(actual["dual_cap_valid"])
            assert int(actual["transport_row_chunk_size"]) == 72
            assert int(actual["transport_col_chunk_size"]) == 72
            row["execution"] = _graph_evidence(owner, inputs, jit_compile)
            assert row["execution"]["traces"] == 1
    finally:
        _write(request, f"genut-reset-consumer-{dtype.name}.json", report)


def test_austria_observation_callback(request):
    from bayesfilter.highdim import ledh_pfpf_genut_model_callbacks_tf as callbacks
    from bayesfilter.highdim.sir_latent_preclip_tf import (
        latent_preclip_zhao_cui_sir_austria_model,
    )

    original, source_sha = _frozen("ledh_pfpf_genut_model_callbacks_tf")
    model = latent_preclip_zhao_cui_sir_austria_model()
    points = tf.reshape(tf.cast(tf.range(2 * 4 * 18), tf.float32), [2, 4, 18])
    theta = tf.zeros([3], tf.float32)

    def program(module):
        # Model static-spec validation is the existing host configuration
        # boundary. The callback's tensor computation is enclosed in XLA.
        selected = module.austria_sir_callbacks(model)
        observation, jacobian, residual = selected.observation_callbacks(theta, 1)

        def evaluate(values):
            value = observation(values)
            return value, jacobian(values), residual(value, tf.zeros([9], tf.float32))
        return _owner(evaluate, (points,))

    owner, reference = program(callbacks), program(original)
    result = owner(points)
    _assert_record_equal(result, reference(points), atol=0., rtol=0.)
    _assert_record_equal(result, owner(points), atol=0., rtol=0.)
    np.testing.assert_array_equal(result[0].numpy(), points.numpy()[..., 1::2])
    _assert_record_equal(owner(points + 1.), reference(points + 1.), atol=0., rtol=0.)
    evidence = _graph_evidence(owner, (points,), True)
    assert evidence["traces"] == 1
    _write(request, "genut-austria-callback.json", {"reference_sha256": source_sha,
                                                     "result": result, "execution": evidence})


@pytest.mark.parametrize("jit_compile", [False, True], ids=["graph", "xla"])
@pytest.mark.parametrize("arm", ["original", "native"])
def test_reduced_correction_cost(arm, jit_compile, request):
    original, source_sha = _frozen("dual_cap_genut_primal_tf")
    implementation = original.dual_cap_genut_primal if arm == "original" else candidate.dual_cap_genut_primal
    inputs = _fixture(tf.float32)
    gpu = os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "visible"
    if gpu:
        tf.config.experimental.reset_memory_stats("GPU:0")
    with GPUProcessMonitor(gpu) as monitor:
        before = memory_snapshot(gpu)
        begin = time.perf_counter()
        owner = _owner(implementation, inputs, jit_compile)
        result = _json(owner(*inputs))
        cold = time.perf_counter() - begin
        compiled = memory_snapshot(gpu)
        timings = []
        for _ in range(20):
            begin = time.perf_counter()
            replay = _json(owner(*inputs))
            timings.append(time.perf_counter() - begin)
            assert replay == result
        warm = memory_snapshot(gpu)
        evidence = _graph_evidence(owner, inputs, jit_compile)
    assert evidence["traces"] == 1
    _write(request, "genut-transitive-cost.json", {"arm": arm, "jit_compile": jit_compile,
        "reference_sha256": source_sha, "inputs": inputs, "result": result, "cold_seconds": cold,
        "warm_seconds": timings, "before": before, "compiled": compiled, "warm": warm,
        "execution": evidence, "device_provenance": monitor.payload(),
        "role": "single_process_descriptive_cost_diagnostic"})
