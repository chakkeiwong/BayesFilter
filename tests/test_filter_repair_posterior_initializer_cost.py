"""Matched complete posterior-initializer costs; diagnostic comparisons only."""

import dataclasses
import hashlib
import importlib.metadata
import os
import platform
import sys
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import posterior_initializer_controller_tf as native
from bayesfilter.inference import posterior_local_initializer as public
from bayesfilter.inference.joint_center import JointCenterLocatorConfig
from bayesfilter.inference.quadratic_geometry import LowRankSPDQuadraticGeometryConfig
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_posterior_initializer_controller import original_module
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_posterior_local_initializer import _thresholds

D = tf.float64


def numerical_payload(value):
    """Only execution-setting metadata differs intentionally between these arms."""
    if isinstance(value, dict):
        return {key: numerical_payload(item) for key, item in value.items() if key != "jit_compile"}
    if isinstance(value, list):
        return [numerical_payload(item) for item in value]
    return value


@pytest.mark.parametrize("dimension", [1, 3])
@pytest.mark.parametrize("arm", ["prior", "graph", "xla"])
def test_posterior_initializer_cost(arm, dimension, request):
    native.clear_posterior_initializer_cache()
    reference, hashes = original_module()
    # An identifiable one-factor Gaussian keeps fit condition diagnostics defined.
    scales = .8 + .11 * tf.cast(tf.range(dimension), D)
    loadings = .25 + .07 * tf.cast(tf.range(dimension), D)
    covariance = scales[:, None] * (tf.linalg.diag(1. - loadings ** 2)
        + loadings[:, None] * loadings[None, :]) * scales[None, :]
    precision = tf.linalg.inv(covariance)
    mode = .13 + .03 * tf.cast(tf.range(dimension), D)

    def batch(points):
        delta = points - mode
        score = -tf.linalg.matmul(delta, precision)
        return .5 * tf.reduce_sum(delta * score, axis=1), score

    def scalar(point):
        values, scores = batch(point[None])
        return values[0], scores[0]

    config = public.PosteriorLocalInitializerConfig(factor_max=1, seed=(31, 43),
        locator_config=JointCenterLocatorConfig(max_iterations=10,
            gradient_tolerance=1e-10, max_objective_evaluations=60))
    movement = LowRankSPDQuadraticGeometryConfig(rank=1, sample_count=12 * dimension,
        min_samples_per_parameter=1, fit_max_iterations=8, pilot_direction_count=6,
        trust_radius=.3, holdout_fraction=.25, holdout_rmse_abs_tolerance=.1,
        holdout_rmse_rel_tolerance=.1, constrain_center_refinement_to_trust_region=True, seed=(12, 34))
    thresholds = _thresholds(dimension)
    inputs = (mode - .15, .8 + .1 * tf.cast(tf.range(dimension), D))
    changed = (inputs[0] + .01, inputs[1] + .01)
    gpu = bool(tf.config.list_logical_devices("GPU"))
    candidate_config = dataclasses.replace(config,
        locator_config=dataclasses.replace(config.locator_config, jit_compile=arm != "graph"))
    options = {"batched_value_and_score_fn": batch, "movement_config": movement,
        "curvature_thresholds": thresholds}

    def snapshot():
        return {**memory_snapshot(gpu), "map_count": len(Path("/proc/self/maps").read_text().splitlines()),
            "host_load_average": list(os.getloadavg())}

    hardware = {"host": platform.node(), "platform": platform.platform(),
        "cpu_models": sorted({line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")}), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "python": sys.version, "executable": sys.executable, "tensorflow": tf.__version__,
        "tensorflow_probability": importlib.metadata.version("tensorflow-probability"),
        "xla_flags": os.environ.get("XLA_FLAGS", "")}
    warm_records = []
    stages = {"prepared": snapshot()}
    with GPUProcessMonitor(gpu) as sharing:
        def execute(operands):
            if gpu:
                tf.config.experimental.reset_memory_stats("GPU:0")
            tick = time.perf_counter()
            endpoint = reference.initialize_posterior_local_location_scale if arm == "prior" else public.initialize_posterior_local_location_scale
            result = endpoint(scalar, operands[0], scale=operands[1],
                config=config if arm == "prior" else candidate_config, **options)
            record = result.payload(include_arrays=True)
            elapsed = time.perf_counter() - tick
            return record, {"seconds": elapsed, "memory": snapshot()}

        first, cold = execute(inputs)
        stages["cold"] = snapshot()
        samples = []
        for _ in range(3):
            current, sample = execute(inputs)
            warm_records.append(current)
            samples.append(sample)
        stages["warm"] = snapshot()
        second, changed_cost = execute(changed)
        stages["changed"] = snapshot()

    # Preserve measured evidence even if a later reference or IR query fails.
    save(request, "posterior-initializer-cost-measured.json", {
        "arm": arm, "dimension": dimension, "hardware": hardware, "cold": cold, "samples": samples,
        "changed_cost": changed_cost, "stages": stages, "result": clean(first),
        "changed_result": clean(second), "warm_records": clean(warm_records),
        "gpu_process_observation": sharing.payload(), "qualification_pending": True})

    # Reference comparisons and compiler inspection cannot contaminate timed memory.
    original_records = ([first, second] if arm == "prior" else
        [reference.initialize_posterior_local_location_scale(scalar, operands[0], scale=operands[1],
            config=config, **options).payload(include_arrays=True) for operands in (inputs, changed)])
    program_info = None
    if arm != "prior":
        owner = native.posterior_initializer_owner(scalar, dimension, candidate_config, movement, thresholds,
            device=inputs[0].device, batched_callback=batch)
        graph = owner.compiled.get_concrete_function().graph.as_graph_def()
        nodes = [*graph.node, *(node for fn in graph.library.function for node in fn.node_def)]
        program_info = {"trace_count": owner.compiled.experimental_get_tracing_count(),
            "graph_nodes": len(nodes), "graph_bytes": graph.ByteSize(),
            "jit_compile": bool(owner.compiled.function_spec.jit_compile),
            "nested_xla_function_count": sum(bool(fn.attr.get("_XlaMustCompile", None) and fn.attr["_XlaMustCompile"].b)
                for fn in graph.library.function)}
        program_info["host_callback_ops"] = sorted({"PyFunc", "EagerPyFunc", "PyFuncStateless"} & {node.op for node in nodes})
        if arm == "xla":
            prepared = owner.prepare_clouds()
            cloud_operands = (prepared["directions"], prepared["movement_offsets"],
                prepared["permutation_keys"], prepared["curvature_offsets"])
            hlo = owner.compiled.experimental_get_compiler_ir(*inputs, *cloud_operands)(stage="hlo")
            other = owner.compiled.experimental_get_compiler_ir(*changed, *cloud_operands)(stage="hlo")
            program_info.update(hlo_bytes=len(hlo.encode()), hlo_unchanged=stable_hlo(hlo) == stable_hlo(other))
    save(request, "posterior-initializer-cost.json", {
        "schema": "filter_posterior_initializer_cost.v1", "arm": arm, "dimension": dimension,
        "reference_revision": "031692a0b", "original_source_sha256": hashes,
        "reference_entire_module": True,
        "reference_comparison_role": "This arm is the frozen reference" if arm == "prior" else "Independent complete frozen-module calls after timing",
        "config": dataclasses.asdict(config),
        "movement_config": dataclasses.asdict(movement), "thresholds": dataclasses.asdict(thresholds),
        "input_sha256": [hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest() for x in inputs],
        "changed_input_sha256": [hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest() for x in changed],
        "target_sha256": [hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest() for x in (mode, covariance)],
        "gpu": gpu, "jit_compile": arm != "graph", "public_installed": True, "hardware": hardware,
        "cold": cold, "samples": samples, "changed_cost": changed_cost, "stages": stages,
        "program": program_info, "result": clean(first), "changed_result": clean(second), "warm_records": clean(warm_records),
        "original_records": clean(original_records), "gpu_process_observation": sharing.payload(),
        "independent_gaussian": clean({"mean": mode, "covariance": covariance}),
        "timing_scope": "Full exported call, including construction, CPU cloud preparation, validation and completed payload materialization.",
        "comparison_rule": "Complete original numerical records; only explicit jit_compile metadata is compared separately. Same inputs, settings and verified CPU random stream.",
        "nonclaims": ["Three warm calls per process are descriptive; independent repeated cohorts remain required.",
            "The original mixes host control and compiled helpers; these are not identical-graph compiler ablations.",
            "The graph arm is an explicit non-default diagnostic and may retain declared compiled dependencies.",
            "The reference includes earlier repairs and does not replace oldest-original terminal gates.",
            "These factor_max=1 fixture costs do not bound the factor_max=2 default or identifiable D5 capacity.",
            "Memory snapshots do not prove native eviction or exact allocation peaks."]})
    assert all(record == first for record in warm_records)
    for actual, expected in zip((first, second), original_records, strict=True):
        _equal_records(numerical_payload(actual), numerical_payload(expected))
        assert actual["accepted"] and actual["diagnostics"]["config"]["locator_config"]["jit_compile"] is (arm != "graph")
        tf.debugging.assert_near(tf.constant(actual["center"], D), mode, atol=1e-7, rtol=1e-7)
        tf.debugging.assert_near(tf.constant(actual["covariance_theta"], D), covariance, atol=1e-7, rtol=1e-7)
    if program_info:
        assert program_info["trace_count"] == 1 and not program_info["host_callback_ops"]
        assert program_info["jit_compile"] is (arm == "xla")
        if arm == "xla":
            assert program_info["hlo_unchanged"]
    native.clear_posterior_initializer_cache()
