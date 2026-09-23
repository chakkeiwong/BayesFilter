"""Diagnostic complete staged-locator costs; internal candidate, not admission."""

import dataclasses
import hashlib
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.joint_center import JointCenterStagedConfig
from bayesfilter.inference.joint_center_staged_tf import (
    StagedJointCenterProgram,
    run_staged_program,
)
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


@pytest.mark.parametrize("dimension", [1, 3])
@pytest.mark.parametrize("arm", ["prior", "graph", "xla"])
def test_staged_center_complete_costs(arm, dimension, request):
    original = FrozenCheckpoint("3582b4ac", "staged_center_cost_original")
    prior = original.load("bayesfilter.inference.joint_center")
    precision = tf.linalg.diag(tf.cast(tf.range(dimension), D) + 1.3) + .07
    mode = .14 + tf.cast(tf.range(dimension), D) * .03

    def callback(point):
        delta = point - mode
        score = -tf.linalg.matvec(precision, delta)
        return .5 * tf.reduce_sum(delta * score), score

    config = JointCenterStagedConfig(checkpoint_iterations=1, total_iterations=10,
        gradient_tolerance=1e-8, max_objective_evaluations=120)
    prior_config = prior.JointCenterStagedConfig(**dataclasses.asdict(config))
    inputs = (.6 - tf.cast(tf.range(dimension), D) * .14,
        .8 + tf.cast(tf.range(dimension), D) * .09)
    changed = tuple(value + .11 for value in inputs)
    gpu = bool(tf.config.list_logical_devices("GPU"))

    def snapshot():
        return {**memory_snapshot(gpu), "map_count": len(Path("/proc/self/maps").read_text().splitlines())}

    validator_calls = []

    def validator(checkpoint):
        validator_calls.append(checkpoint.endpoint_accepted)
        return True

    stages = {"prepared": snapshot()}
    with GPUProcessMonitor(gpu) as sharing:
        tick = time.perf_counter()
        owner = (None if arm == "prior" else StagedJointCenterProgram(callback, dimension,
            dataclasses.replace(config, jit_compile=arm == "xla")))
        build_seconds = time.perf_counter() - tick
        stages["built"] = snapshot()

        def execute(operands):
            validator_calls.clear()
            if gpu:
                tf.config.experimental.reset_memory_stats("GPU:0")
            tick = time.perf_counter()
            if owner is None:
                result = prior.locate_joint_center_staged(callback, operands[0], scale=operands[1],
                    config=prior_config, checkpoint_validator=validator)
            else:
                result = run_staged_program(owner, *operands, validator)
            record = clean(dataclasses.asdict(result))
            elapsed = time.perf_counter() - tick
            assert validator_calls == [True]
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

    # Numerical references and IR export follow every timed/memory observation.
    expected = prior.locate_joint_center_staged(callback, inputs[0], scale=inputs[1],
        config=prior_config, checkpoint_validator=lambda _: True)
    expected_changed = prior.locate_joint_center_staged(callback, changed[0], scale=changed[1],
        config=prior_config, checkpoint_validator=lambda _: True)
    original_records = [clean(dataclasses.asdict(value)) for value in (expected, expected_changed)]
    programs = {}
    if owner is not None:
        first_state = owner.checkpoint(*inputs)
        changed_state = owner.checkpoint(*changed)
        for name, program, args, other in (
            ("checkpoint", owner.checkpoint, inputs, changed),
            ("continuation", owner.continuation, (*inputs, first_state), (*changed, changed_state)),
        ):
            graph = program.get_concrete_function().graph.as_graph_def()
            entry = {"trace_count": program.experimental_get_tracing_count(),
                "graph_nodes": len(graph.node) + sum(len(f.node_def) for f in graph.library.function),
                "graph_bytes": graph.ByteSize(), "jit_compile": bool(program.function_spec.jit_compile)}
            if arm == "xla":
                hlo = program.experimental_get_compiler_ir(*args)(stage="hlo")
                other_hlo = program.experimental_get_compiler_ir(*other)(stage="hlo")
                entry.update(hlo_bytes=len(hlo.encode()), hlo_unchanged=stable_hlo(hlo) == stable_hlo(other_hlo))
            programs[name] = entry
    report = {"schema": "filter_staged_center_cost.v1", "arm": arm, "dimension": dimension,
        "numerical_authority": "3582b4ac", "mechanism_baseline": "3582b4ac",
        "original_source_sha256": original.hashes(), "config": dataclasses.asdict(config),
        "input_sha256": [hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest() for x in inputs],
        "changed_input_sha256": [hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest() for x in changed],
        "gpu": gpu, "jit_compile": arm != "graph", "candidate_installed_publicly": False,
        "execution_role": {"prior": "original_public_mixed_host_and_xla", "graph": "explicit_internal_graph_reference",
            "xla": "internal_enclosing_xla_candidate"}[arm],
        "build_seconds": build_seconds, "cold": cold, "samples": samples,
        "changed_cost": changed_cost, "stages": stages, "programs": programs,
        "result": first, "changed_result": second,
        "original_result": original_records[0], "original_changed_result": original_records[1],
        "validator_calls_per_execution": 1, "gpu_process_observation": sharing.payload(),
        "timing_scope": "Full result materialization and validator; owner construction (including checkpoint trace) enters total cold.",
        "comparison_rule": "All original fields unchanged; only graph diagnostic jit_compile reporting differs by design.",
        "nonclaims": ["One fresh process per arm/extent is descriptive, not a performance ranking.",
            "CPU and graph are explicit reference lanes; public candidate integration remains unqualified.",
            "Original recompiles stages per call; reuse comparison is not an identical-graph compiler ablation.",
            "Snapshots do not bound peaks, certify native eviction or close long-term residency."]}
    save(request, "staged-center-cost.json", report)
    for actual, expected_record in zip((first, second), original_records, strict=True):
        assert actual["jit_compile"] is (arm != "graph")
        adjusted = {**expected_record, "jit_compile": arm != "graph"}
        _equal_records(actual, adjusted)
        assert actual["endpoint_accepted"] and actual["continuation_started"]
    for info in programs.values():
        assert info["trace_count"] == 1 and info["jit_compile"] is (arm == "xla")
        if arm == "xla":
            assert info["hlo_unchanged"]
