"""Diagnostic public single-locator costs with original same-mode records."""

import dataclasses
import hashlib
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.joint_center import (
    JointCenterLocatorConfig,
    locate_joint_center,
)
from bayesfilter.inference.joint_center_tf import (
    clear_joint_center_cache,
    joint_center_program,
)
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_joint_center_native import original_source
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


@pytest.mark.parametrize("dimension", [1, 3])
@pytest.mark.parametrize("arm", ["prior", "graph", "xla"])
def test_single_public_cost(arm, dimension, request):
    clear_joint_center_cache()
    original, prior, compatibility = original_source("joint_public_cost_original")
    precision = tf.linalg.diag(tf.cast(tf.range(dimension), D) + 1.3) + .07
    mode = .14 + tf.cast(tf.range(dimension), D) * .03

    def callback(point):
        delta = point - mode
        score = -tf.linalg.matvec(precision, delta)
        return .5 * tf.reduce_sum(delta * score), score

    config = JointCenterLocatorConfig(max_iterations=10, gradient_tolerance=1e-8,
                                      max_objective_evaluations=120)
    prior_config = prior.JointCenterLocatorConfig(**dataclasses.asdict(config))
    inputs = (.6 - tf.cast(tf.range(dimension), D) * .14,
              .8 + tf.cast(tf.range(dimension), D) * .09)
    changed = tuple(value + .11 for value in inputs)
    gpu = bool(tf.config.list_logical_devices("GPU"))

    def snapshot():
        return {**memory_snapshot(gpu), "map_count": len(Path("/proc/self/maps").read_text().splitlines())}

    stages = {"prepared": snapshot()}
    with GPUProcessMonitor(gpu) as sharing:
        def execute(operands):
            if gpu:
                tf.config.experimental.reset_memory_stats("GPU:0")
            tick = time.perf_counter()
            if arm == "prior":
                result = prior.locate_joint_center(callback, operands[0], scale=operands[1], config=prior_config)
            else:
                result = locate_joint_center(callback, operands[0], scale=operands[1],
                    config=dataclasses.replace(config, jit_compile=arm == "xla"))
            record = clean(dataclasses.asdict(result))
            elapsed = time.perf_counter() - tick
            return record, {"seconds": elapsed, "memory": snapshot()}

        first, cold = execute(inputs)
        stages["cold"] = snapshot()
        samples = []
        for _ in range(3):
            current, sample = execute(inputs)
            assert current == first
            samples.append(sample)
        stages["warm"] = snapshot()
        second, changed_cost = execute(changed)
        stages["changed"] = snapshot()

    # Independent original and IR inspection occur after timed memory observations.
    original_records = [clean(dataclasses.asdict(prior.locate_joint_center(
        callback, operands[0], scale=operands[1],
        config=dataclasses.replace(prior_config, jit_compile=arm != "graph"))))
        for operands in (inputs, changed)]
    cross_mode = None
    if arm == "graph":
        cross_mode = [clean(dataclasses.asdict(prior.locate_joint_center(
            callback, operands[0], scale=operands[1], config=prior_config)))
            for operands in (inputs, changed)]
    program_info = None
    if arm != "prior":
        owner = joint_center_program(callback, dimension,
            dataclasses.replace(config, jit_compile=arm == "xla"), device=inputs[0].device)
        graph = owner.get_concrete_function().graph.as_graph_def()
        program_info = {"trace_count": owner.experimental_get_tracing_count(),
            "graph_nodes": len(graph.node) + sum(len(f.node_def) for f in graph.library.function),
            "graph_bytes": graph.ByteSize(), "jit_compile": bool(owner.function_spec.jit_compile)}
        if arm == "xla":
            hlo = owner.experimental_get_compiler_ir(*inputs)(stage="hlo")
            other = owner.experimental_get_compiler_ir(*changed)(stage="hlo")
            program_info.update(hlo_bytes=len(hlo.encode()), hlo_unchanged=stable_hlo(hlo) == stable_hlo(other))
    save(request, "joint-public-cost.json", {"schema": "filter_joint_public_cost.v1",
        "arm": arm, "dimension": dimension, "numerical_authority": "3582b4ac",
        "mechanism_baseline": "3582b4ac", "original_source_sha256": original.hashes(),
        "original_compatibility": compatibility, "config": dataclasses.asdict(config),
        "input_sha256": [hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest() for x in inputs],
        "changed_input_sha256": [hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest() for x in changed],
        "gpu": gpu, "jit_compile": arm != "graph", "candidate_installed_publicly": True,
        "cold": cold, "samples": samples, "changed_cost": changed_cost, "stages": stages,
        "program": program_info, "result": first, "changed_result": second,
        "original_records": original_records, "cross_mode_original_records": cross_mode,
        "gpu_process_observation": sharing.payload(),
        "timing_scope": "Full public call including construction, validation and completed result materialization.",
        "comparison_rule": "Same-mode complete original records; cross-mode differences remain independent witnesses.",
        "nonclaims": ["One process per arm with three warm calls is descriptive only.",
            "Original rebuilds the optimizer per call; this is not an identical-graph compiler ablation.",
            "Snapshots do not establish native eviction, long-term residency or exact peaks."]})
    for actual, expected in zip((first, second), original_records, strict=True):
        _equal_records(actual, expected)
        assert actual["endpoint_accepted"] and actual["jit_compile"] is (arm != "graph")
    if program_info:
        assert program_info["trace_count"] == 1
        if arm == "xla":
            assert program_info["hlo_unchanged"]
    clear_joint_center_cache()
