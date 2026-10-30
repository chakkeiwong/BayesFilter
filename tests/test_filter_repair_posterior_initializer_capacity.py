"""Diagnostic complete-initializer reuse and successful-owner memory accounting."""

import copy
import dataclasses
import gc
import hashlib
import json
import os
import platform
import sys
import time
import weakref
from pathlib import Path

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference import posterior_initializer_controller_tf as native
from bayesfilter.inference import posterior_local_initializer as public
from bayesfilter.inference.joint_center import JointCenterLocatorConfig
from bayesfilter.inference.quadratic_geometry import LowRankSPDQuadraticGeometryConfig
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_posterior_initializer_controller import original_module
from tests.test_filter_repair_posterior_initializer_cost import numerical_payload
from tests.test_filter_repair_posterior_residency import mapping_snapshot
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_posterior_local_initializer import _thresholds

D = tf.float64


class Gaussian:
    def __init__(self, mode, precision):
        self.mode = mode
        self.precision = precision

    def batch(self, points):
        delta = points - self.mode
        score = -tf.linalg.matmul(delta, self.precision)
        return .5 * tf.reduce_sum(delta * score, axis=1), score

    def scalar(self, point):
        values, scores = self.batch(point[None])
        return values[0], scores[0]


def fixture(dimension):
    scales = .8 + .11 * tf.cast(tf.range(dimension), D)
    loadings = .25 + .07 * tf.cast(tf.range(dimension), D)
    covariance = scales[:, None] * (tf.linalg.diag(1. - loadings ** 2)
        + loadings[:, None] * loadings[None, :]) * scales[None, :]
    mode = .13 + .03 * tf.cast(tf.range(dimension), D)
    config = public.PosteriorLocalInitializerConfig(factor_max=1, seed=(31, 43),
        locator_config=JointCenterLocatorConfig(max_iterations=10,
            gradient_tolerance=1e-10, max_objective_evaluations=60))
    movement = LowRankSPDQuadraticGeometryConfig(rank=1, sample_count=12 * dimension,
        min_samples_per_parameter=1, fit_max_iterations=8, pilot_direction_count=6,
        trust_radius=.3, holdout_fraction=.25, holdout_rmse_abs_tolerance=.1,
        holdout_rmse_rel_tolerance=.1, constrain_center_refinement_to_trust_region=True, seed=(12, 34))
    inputs = (mode - .15, .8 + .1 * tf.cast(tf.range(dimension), D))
    changed = (inputs[0] + .01, inputs[1] + .01)
    return mode, covariance, inputs, changed, config, movement, _thresholds(dimension)


def snapshot(gpu):
    return {**memory_snapshot(gpu),
        "map_count": len(Path("/proc/self/maps").read_text().splitlines()),
        "mapping_groups": mapping_snapshot(), "host_load_average": list(os.getloadavg())}


def fixture_identity(mode, covariance, inputs, changed, config, movement, thresholds):
    return {"config": dataclasses.asdict(config), "movement_config": dataclasses.asdict(movement),
        "thresholds": dataclasses.asdict(thresholds),
        "input_sha256": [hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest() for x in (*inputs, *changed)],
        "target_sha256": [hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest() for x in (mode, covariance)]}


def hardware():
    return {"host": platform.node(), "platform": platform.platform(), "python": sys.version,
        "executable": sys.executable, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "tensorflow": tf.__version__, "tensorflow_probability": tfp.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", "")}


def program_record(owner, inputs, changed):
    graph = owner.compiled.get_concrete_function().graph.as_graph_def()
    nodes = [*graph.node, *(node for fn in graph.library.function for node in fn.node_def)]
    result = {"trace_count": owner.compiled.experimental_get_tracing_count(),
        "jit_compile": bool(owner.compiled.function_spec.jit_compile),
        "graph_nodes": len(nodes), "graph_bytes": graph.ByteSize(),
        "host_callback_ops": sorted({"PyFunc", "EagerPyFunc", "PyFuncStateless"} & {n.op for n in nodes}),
        "nested_xla_function_count": sum(bool(fn.attr.get("_XlaMustCompile", None) and fn.attr["_XlaMustCompile"].b)
            for fn in graph.library.function)}
    if result["jit_compile"]:
        clouds = owner.prepare_clouds()
        operands = (clouds["directions"], clouds["movement_offsets"],
            clouds["permutation_keys"], clouds["curvature_offsets"])
        hlo = owner.compiled.experimental_get_compiler_ir(*inputs, *operands)(stage="hlo")
        second = owner.compiled.experimental_get_compiler_ir(*changed, *operands)(stage="hlo")
        result.update(hlo_bytes=len(hlo.encode()), hlo_unchanged=stable_hlo(hlo) == stable_hlo(second))
    return result


@pytest.mark.parametrize("dimension", [1, 3])
@pytest.mark.parametrize("arm", ["prior", "graph", "xla"])
def test_complete_initializer_twenty_call_reuse(arm, dimension, request, monkeypatch):
    native.clear_posterior_initializer_cache()
    reference, hashes = original_module()
    mode, covariance, inputs, changed, config, movement, thresholds = fixture(dimension)
    target = Gaussian(mode, tf.linalg.inv(covariance))
    identity = fixture_identity(mode, covariance, inputs, changed, config, movement, thresholds)
    candidate_config = dataclasses.replace(config,
        locator_config=dataclasses.replace(config.locator_config, jit_compile=arm != "graph"))
    gpu = bool(tf.config.list_logical_devices("GPU"))
    endpoint = reference.initialize_posterior_local_location_scale if arm == "prior" else public.initialize_posterior_local_location_scale
    records, elapsed, reuse, stages = [], [], [], {"prepared": snapshot(gpu)}
    call_memory, owner_returns = [], []
    original_call = native.PreparedPosteriorInitializer.__call__

    def observed_call(self, *args, **kwargs):
        result = original_call(self, *args, **kwargs)
        leaves = tf.nest.flatten(result)
        owner_returns.append({"memory": memory_snapshot(gpu),
            "output_bytes": sum(value.shape.num_elements() * value.dtype.size for value in leaves)})
        return result

    def execute(operands):
        if gpu:
            tf.config.experimental.reset_memory_stats("GPU:0")
        tick = time.perf_counter()
        payload = endpoint(target.scalar, operands[0], scale=operands[1], config=config if arm == "prior" else candidate_config,
            movement_config=movement, curvature_thresholds=thresholds,
            batched_value_and_score_fn=target.batch).payload(include_arrays=True)
        elapsed.append(time.perf_counter() - tick)
        records.append(payload)
        call_memory.append(memory_snapshot(gpu))

    with GPUProcessMonitor(gpu) as sharing:
        execute(inputs)
        stages["cold"] = snapshot(gpu)
        owner_id = id(native._LAST_OWNER[2]) if arm != "prior" else None
        # The uninstrumented cold call controls the one-return-boundary observer.
        if arm != "prior":
            monkeypatch.setattr(native.PreparedPosteriorInitializer, "__call__", observed_call)
        execute(changed)
        stages["changed"] = snapshot(gpu)
        for index in range(20):
            execute(inputs if index % 2 == 0 else changed)
            if arm != "prior":
                reuse.append({"owner_reused": id(native._LAST_OWNER[2]) == owner_id,
                    "traces": native._LAST_OWNER[2].compiled.experimental_get_tracing_count(),
                    "dependency_count": native._LAST_OWNER[2].dependency_scope.program_count})
            if index + 1 in (1, 5, 10, 15, 20):
                stages[f"warm_{index + 1}"] = snapshot(gpu)
    measured = {"schema": "filter_posterior_initializer_capacity.v1", "kind": "reuse",
        "arm": arm, "dimension": dimension, "gpu": gpu, "warm_calls": 20, "fixture": identity,
        "hardware": hardware(),
        "stages": stages, "elapsed_seconds": elapsed, "records": clean(records), "reuse": reuse,
        "call_memory": call_memory, "owner_returns": owner_returns,
        "gpu_process_observation": sharing.payload(), "qualification_pending": True}
    save(request, "posterior-initializer-capacity-measured.json", measured)
    # Compiler IR and original calls occur strictly after the memory window.
    program = program_record(native._LAST_OWNER[2], inputs, changed) if arm != "prior" else None
    originals = records[:2] if arm == "prior" else [reference.initialize_posterior_local_location_scale(
        target.scalar, operands[0], scale=operands[1], config=config, movement_config=movement,
        curvature_thresholds=thresholds, batched_value_and_score_fn=target.batch).payload(include_arrays=True)
        for operands in (inputs, changed)]
    report = {**{key: value for key, value in measured.items() if key != "qualification_pending"},
        "original_records": clean(originals), "original_source_sha256": hashes,
        "reference_revision": "031692a0b", "program": program,
        "independent_gaussian": clean({"mean": mode, "covariance": covariance}),
        "nonclaims": ["Bounded descriptive capacity; no leak-freedom or native eviction claim.",
            "D1/D3 factor_max=1 fixtures do not qualify actual DZ5 or factor_max=2 capacity."]}
    save(request, "posterior-initializer-capacity.json", report)
    for index, actual in enumerate(records):
        _equal_records(numerical_payload(actual), numerical_payload(originals[index % 2]))
        assert actual["accepted"]
        tf.debugging.assert_near(tf.constant(actual["center"], D), mode, atol=1e-7, rtol=1e-7)
        tf.debugging.assert_near(tf.constant(actual["covariance_theta"], D), covariance, atol=1e-7, rtol=1e-7)
        if index >= 2:
            assert actual == records[index % 2]
    if program:
        assert program["jit_compile"] is (arm == "xla") and program["trace_count"] == 1
        assert not program["host_callback_ops"]
        if arm == "xla":
            assert program["hlo_unchanged"]
        else:
            assert program["nested_xla_function_count"] > 0
        assert all(row["owner_reused"] and row["traces"] == 1 for row in reuse)
        assert len({row["dependency_count"] for row in reuse}) == 1
    native.clear_posterior_initializer_cache()


@pytest.mark.parametrize("dimension", [1, 3])
def test_successful_initializer_owner_replacement(dimension, request):
    native.clear_posterior_initializer_cache()
    reference, hashes = original_module()
    mode, covariance, inputs, changed, config, movement, thresholds = fixture(dimension)
    identity = fixture_identity(mode, covariance, inputs, changed, config, movement, thresholds)
    precision = tf.linalg.inv(covariance)
    gpu = bool(tf.config.list_logical_devices("GPU"))
    records, elapsed, references, collection, programs = [], [], [], [], []
    stages = {"prepared": snapshot(gpu)}
    with GPUProcessMonitor(gpu) as sharing:
        for index in range(4):
            target = Gaussian(mode, precision)
            tick = time.perf_counter()
            payload = public.initialize_posterior_local_location_scale(target.scalar, inputs[0],
                scale=inputs[1], config=config, movement_config=movement,
                curvature_thresholds=thresholds, batched_value_and_score_fn=target.batch).payload(include_arrays=True)
            elapsed.append(time.perf_counter() - tick)
            records.append(payload)
            stages[f"owner_{index + 1}_completed"] = snapshot(gpu)
            owner = native._LAST_OWNER[2]
            programs.append({"jit_compile": bool(owner.compiled.function_spec.jit_compile),
                "traces": owner.compiled.experimental_get_tracing_count(),
                "dependency_count": owner.dependency_scope.program_count})
            references.append(tuple(weakref.ref(item) for item in
                (target, owner, owner.compiled, owner.dependency_scope)))
            del owner, target, payload
            gc.collect()
            collection.append([[ref() is None for ref in group] for group in references[:-1]])
            stages[f"owner_{index + 1}_collected"] = snapshot(gpu)
        native.clear_posterior_initializer_cache()
        gc.collect()
        released = [[ref() is None for ref in group] for group in references]
        stages["all_python_owners_released"] = snapshot(gpu)
    measured = {"schema": "filter_posterior_initializer_capacity.v1", "kind": "owner_replacement",
        "arm": "xla", "dimension": dimension, "gpu": gpu, "owner_count": 4, "fixture": identity,
        "hardware": hardware(),
        "stages": stages, "elapsed_seconds": elapsed, "records": clean(records), "programs": programs,
        "superseded_collection": collection, "final_collection": released,
        "gpu_process_observation": sharing.payload(), "qualification_pending": True}
    save(request, "posterior-initializer-capacity-measured.json", measured)
    target = Gaussian(mode, precision)
    original = reference.initialize_posterior_local_location_scale(target.scalar, inputs[0], scale=inputs[1],
        config=config, movement_config=movement, curvature_thresholds=thresholds,
        batched_value_and_score_fn=target.batch).payload(include_arrays=True)
    save(request, "posterior-initializer-capacity.json", {
        **{key: value for key, value in measured.items() if key != "qualification_pending"},
        "original_records": clean([original]),
        "reference_revision": "031692a0b", "original_source_sha256": hashes,
        "independent_gaussian": clean({"mean": mode, "covariance": covariance}),
        "nonclaims": ["Python collection does not prove native executable eviction or a general capacity bound."]})
    assert all(all(all(group) for group in step) for step in collection)
    assert all(all(group) for group in released)
    assert all(row["jit_compile"] and row["traces"] == 1 for row in programs)
    assert len({row["dependency_count"] for row in programs}) == 1
    for record in records:
        assert record == records[0]
        _equal_records(record, original)
        assert record["accepted"]
        tf.debugging.assert_near(tf.constant(record["center"], D), mode, atol=1e-7, rtol=1e-7)
        tf.debugging.assert_near(tf.constant(record["covariance_theta"], D), covariance, atol=1e-7, rtol=1e-7)


def test_capacity_observer_without_numerical_calls(request):
    """Retain the same completed D3 payloads and snapshot count, without calls."""
    source = (Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
        / 'run-04493/posterior-initializer-cost.json')
    frozen = json.loads(source.read_text())
    assert frozen["dimension"] == 3 and frozen["arm"] == "prior"
    operands = fixture(3)  # Match input preparation; never call either endpoint.
    gpu = bool(tf.config.list_logical_devices("GPU"))
    stages, records = {"prepared": snapshot(gpu)}, []
    with GPUProcessMonitor(gpu) as sharing:
        for index in range(22):
            records.append(copy.deepcopy(frozen["result"] if index % 2 == 0 else frozen["changed_result"]))
            if index < 2 or index - 1 in (1, 5, 10, 15, 20):
                name = ("cold", "changed")[index] if index < 2 else f"warm_{index - 1}"
                stages[name] = snapshot(gpu)
        del records
        gc.collect()
        owner_stages, records = {"prepared": snapshot(gpu)}, []
        for index in range(4):
            records.append(copy.deepcopy(frozen["result"]))
            owner_stages[f"owner_{index + 1}_completed"] = snapshot(gpu)
            gc.collect()
            owner_stages[f"owner_{index + 1}_collected"] = snapshot(gpu)
        gc.collect()
        owner_stages["all_python_owners_released"] = snapshot(gpu)
    save(request, "posterior-initializer-capacity-observer.json", {
        "schema": "filter_posterior_initializer_capacity_observer.v1", "gpu": gpu,
        "hardware": hardware(),
        "stages": stages, "completed_records_retained": 22, "calls_between_snapshots": 0,
        "owner_stages": owner_stages, "owner_records_retained": len(records),
        "input_dimension": int(operands[0].shape[0]), "source_record": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "gpu_process_observation": sharing.payload(),
        "nonclaims": ["No numerical endpoint calls; this bounds observer overhead only."]})
