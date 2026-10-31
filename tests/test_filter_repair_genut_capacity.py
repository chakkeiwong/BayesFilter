"""Independent diagnostic capacity/moment checks; not canonical LEDH evidence."""

import hashlib
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import dual_cap_genut_primal_tf as candidate
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_genut_transitive import _frozen, _owner, _write


def _materialize(record):
    return {key: value.numpy() for key, value in record.items()}


@pytest.mark.parametrize("count,dimension", [(1000, 3), (10000, 3), (10000, 18)],
                         ids=["1000-3", "10000-3", "10000-18"])
@pytest.mark.parametrize("mode", ["graph", "xla"])
@pytest.mark.parametrize("arm", ["original", "native"])
def test_capacity_cost(arm, mode, count, dimension, request):
    original, source_sha = _frozen("dual_cap_genut_primal_tf")
    implementation = original.dual_cap_genut_primal if arm == "original" else candidate.dual_cap_genut_primal
    with tf.device("/CPU:0"):
        inputs = (tf.random.stateless_normal([count, dimension], [101, 102], dtype=tf.float32),
                  tf.nn.softmax(.3 * tf.random.stateless_normal([count], [103, 104], dtype=tf.float32)),
                  tf.random.stateless_normal([count, dimension], [105, 106], dtype=tf.float32))
    gpu = os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "visible"
    assert not (gpu and arm == "original" and mode == "xla"), "Preserved compiler abort is not retried"
    if gpu:
        tf.config.experimental.reset_memory_stats("GPU:0")
    with GPUProcessMonitor(gpu) as monitor:
        before = memory_snapshot(gpu)
        start = time.perf_counter()
        owner = _owner(implementation, inputs, mode == "xla")
        result = _materialize(owner(*inputs))
        cold = time.perf_counter() - start
        compiled = memory_snapshot(gpu)
        timings = []
        for _ in range(10):
            start = time.perf_counter()
            replay = _materialize(owner(*inputs))
            timings.append(time.perf_counter() - start)
            assert all(np.array_equal(value, replay[key]) for key, value in result.items())
        warm = memory_snapshot(gpu)
    assert owner.experimental_get_tracing_count() == 1
    source, weights, _ = (value.numpy().astype(np.float64) for value in inputs)
    mean = weights @ source
    centered = source - mean
    covariance = (centered.T * weights) @ centered
    particles = result["particles"].astype(np.float64)
    actual_mean = particles.mean(axis=0)
    actual_covariance = (particles-actual_mean).T @ (particles-actual_mean) / count
    checks = {"finite": bool(np.isfinite(particles).all()), "valid": bool(result["valid"]),
              "mean": bool(np.allclose(actual_mean, mean, atol=2e-5, rtol=2e-5)),
              "covariance": bool(np.allclose(actual_covariance, covariance, atol=2e-5, rtol=2e-5)),
              "radial_cap": float(result["maximum_pairwise_post_cap_particle_rms"]) <= 2. + 2e-5}
    output = Path(request.config.getoption("xmlpath")).parent
    # Independent post-run numerical evidence, outside the timed runtime kernel.
    arrays = output / "genut-capacity-arrays.npz"
    np.savez_compressed(arrays, **result, source=inputs[0].numpy(), weights=inputs[1].numpy(), reset=inputs[2].numpy())
    _write(request, "genut-capacity.json", {
        "role": "descriptive_reduced_primal_capacity_not_full_target_qualification",
        "arm": arm, "mode": mode, "count": count, "dimension": dimension,
        "reference_sha256": source_sha, "cold_seconds": cold, "warm_seconds": timings,
        "before": before, "compiled": compiled, "warm": warm, "device_provenance": monitor.payload(),
        "tf32": tf.config.experimental.tensor_float_32_execution_enabled(), "checks": checks,
        "independent_moment_max_error": {"mean": float(np.max(np.abs(actual_mean-mean))),
                                         "covariance": float(np.max(np.abs(actual_covariance-covariance)))},
        "numerical_qualified": all(checks.values()), "complete_cross_mode_gate_passed": False,
        "known_open_gate": "1e-7 thresholded cap-report cross-mode equivalence",
        "arrays_sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
        "input_sha256": [hashlib.sha256(value.numpy().tobytes()).hexdigest() for value in inputs],
        "scalar_record": {key: value for key, value in result.items() if key != "particles"},
    })
    assert checks["finite"] and checks["valid"] and checks["radial_cap"]
    if arm == "native":
        assert all(checks.values()), "Native independent moment gate failed; preserve costs without ranking"


@pytest.mark.parametrize("count,dimension,source_run", [(1000, 3, 4114), (10000, 3, 4118), (10000, 18, 4122)],
                         ids=["1000-3", "10000-3", "10000-18"])
def test_full_fp64_reference(count, dimension, source_run, request):
    """Exact saved FP32 operands cast to FP64; independent full-program diagnostic."""
    artifact_root = Path("/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917")
    saved_path = artifact_root / f"run-{source_run:05d}/genut-capacity-arrays.npz"
    with np.load(saved_path, allow_pickle=False) as saved:
        inputs = tuple(tf.constant(saved[key], tf.float64) for key in ("source", "weights", "reset"))
    assert inputs[0].shape == (count, dimension)
    assert os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "cpu"
    original, source_sha = _frozen("dual_cap_genut_primal_tf")
    owner = _owner(original.dual_cap_genut_primal, inputs, True)
    result = _materialize(owner(*inputs))
    replay = _materialize(owner(*inputs))
    assert all(np.array_equal(value, replay[key]) for key, value in result.items())
    output = Path(request.config.getoption("xmlpath")).parent
    arrays = output / "genut-capacity-fp64-arrays.npz"
    np.savez_compressed(arrays, **result)
    _write(request, "genut-capacity-fp64.json", {
        "role": "independent_fp64_full_program_on_frozen_fp32_operands",
        "reference_sha256": source_sha, "source_run": source_run, "count": count, "dimension": dimension,
        "saved_fp32_arrays_sha256": hashlib.sha256(saved_path.read_bytes()).hexdigest(),
        "arrays_sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
        "scalar_record": {key: value for key, value in result.items() if key != "particles"}})
    assert bool(result["valid"])
    assert result["mean_residual"] < 2e-10
    assert result["covariance_residual"] < 2e-10


def test_matched_cpu_xla_cost(request):
    """Interleaved attribution of the observed d=18 CPU warm-cost increase."""
    assert os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "cpu"
    artifact_root = Path("/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917")
    saved_path = artifact_root / "run-04122/genut-capacity-arrays.npz"
    with np.load(saved_path, allow_pickle=False) as saved:
        inputs = tuple(tf.constant(saved[key], tf.float32) for key in ("source", "weights", "reset"))
    original, source_sha = _frozen("dual_cap_genut_primal_tf")
    owners = {"original": _owner(original.dual_cap_genut_primal, inputs, True),
              "native": _owner(candidate.dual_cap_genut_primal, inputs, True)}
    expected = {name: _materialize(owner(*inputs)) for name, owner in owners.items()}
    rows = []
    for repeat in range(20):
        order = ("original", "native") if repeat % 2 == 0 else ("native", "original")
        timings = {}
        for name in order:
            start = time.perf_counter()
            result = _materialize(owners[name](*inputs))
            timings[name] = time.perf_counter() - start
            assert all(np.array_equal(value, result[key]) for key, value in expected[name].items())
        rows.append({"repeat": repeat, "order": order, "seconds": timings})
    output = Path(request.config.getoption("xmlpath")).parent
    hlo_evidence = {}
    for name, owner in owners.items():
        assert owner.experimental_get_tracing_count() == 1
        hlo = owner.experimental_get_compiler_ir(*inputs)(stage="optimized_hlo")
        path = output / f"genut-capacity-{name}-optimized.hlo"
        path.write_text(hlo)
        hlo_evidence[name] = {"sha256": hashlib.sha256(hlo.encode()).hexdigest(), "bytes": len(hlo),
                              "dot_operations": len(re.findall(r"\bdot\(", hlo)),
                              "reduce_operations": len(re.findall(r"\breduce\(", hlo))}
    _write(request, "genut-capacity-matched-cpu.json", {
        "role": "matched_single_process_cost_attribution_not_general_speed_ranking",
        "reference_sha256": source_sha, "saved_arrays_sha256": hashlib.sha256(saved_path.read_bytes()).hexdigest(),
        "rows": rows, "hlo": hlo_evidence})
