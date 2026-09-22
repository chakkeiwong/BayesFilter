"""Diagnostic sequential startup/reuse attribution, including observer overhead."""

import dataclasses
import gc
import json
import time
import weakref
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_controller_tf as runtime
from bayesfilter.inference import sequential_map_covariance as public
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_lifecycle_original import _compare_original
from tests.test_filter_repair_posterior_residency import mapping_snapshot
from tests.test_filter_repair_sequential_public_memory import fixture


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('primed', [False, True])
def test_sequential_startup_and_reuse_residency(primed, dimension, request):
    scalar, batch, cfg, args = fixture(dimension)
    gpu = bool(tf.config.list_logical_devices('GPU'))
    stages, mappings, seconds = {}, {}, {}
    runtime.clear_sequential_controller_cache()

    def snapshot(name):
        stages[name] = memory_snapshot(gpu)
        mappings[name] = mapping_snapshot()

    with GPUProcessMonitor(gpu) as sharing:
        snapshot('prepared')
        if primed:
            @tf.function(input_signature=[tf.TensorSpec([16], tf.float64)], jit_compile=True, autograph=False)
            def compiler_control(value):
                return tf.math.sin(value) + value

            begin = time.perf_counter()
            compiler_control(tf.ones([16], tf.float64)).numpy()
            seconds['minimal_xla'] = time.perf_counter() - begin
            snapshot('minimal_xla')
            del compiler_control
            gc.collect()
            snapshot('minimal_xla_python_release')
        begin = time.perf_counter()
        owner = public.sequential_controller(scalar, batch, None, 1, dimension,
            cfg, cfg.search_sample_count, device=args[0].device)
        concrete = owner.compiled.get_concrete_function()
        seconds['build_and_trace'] = time.perf_counter() - begin
        references = {'root': weakref.ref(owner), 'graph': weakref.ref(concrete.graph),
            'scope': weakref.ref(owner.dependency_scope), 'scalar': weakref.ref(scalar), 'batch': weakref.ref(batch)}
        snapshot('traced')

        def execute(values):
            return public.estimate_sequential_map_covariance(scalar, values[0],
                batched_value_and_score_fn=batch, scale=values[1], config=cfg).payload()

        begin = time.perf_counter()
        first = execute(args)
        seconds['cold_public'] = time.perf_counter() - begin
        snapshot('cold')
        changed = (args[0] + .0005, args[1] * 1.1)
        second = execute(changed)
        snapshot('changed')
        begin = time.perf_counter()
        for index in range(1200):
            values, expected = (args, first) if index % 2 == 0 else (changed, second)
            assert execute(values) == expected
            if (index + 1) % 300 == 0:
                snapshot(f'reuse_{index + 1}')
        seconds['reuse_1200'] = time.perf_counter() - begin
        traces = owner.compiled.experimental_get_tracing_count()
        assert traces == 1
        runtime.clear_sequential_controller_cache()
        del execute, owner, concrete, scalar, batch
        gc.collect()
        released = {name: ref() is None for name, ref in references.items()}
        snapshot('python_release')
    # The independent numerical authority runs after all measured stages.
    scalar, batch, _, _ = fixture(dimension)
    checkpoint = FrozenCheckpoint('3582b4ac', 'sequential_residency_original')
    original = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    original_cfg = original.SequentialMapCovarianceConfig(**dataclasses.asdict(cfg))

    def reference(values):
        return original.estimate_sequential_map_covariance(scalar, values[0],
            batched_value_and_score_fn=batch, scale=values[1], config=original_cfg).payload()

    expected, expected_changed = reference(args), reference(changed)
    report = {'schema': 'filter_sequential_public_residency.v1', 'dimension': dimension,
        'minimal_xla_prewarm': primed, 'gpu': gpu, 'seconds': seconds, 'stages': stages,
        'mappings': mappings, 'python_released': released, 'trace_count': traces,
        'result': first, 'changed_result': second, 'original': expected, 'original_changed': expected_changed,
        'original_source_sha256': checkpoint.hashes(), 'reuse_calls': 1200,
        'gpu_process_observation': sharing.payload(), 'role': 'allocation_attribution_not_performance_ranking',
        'nonclaims': ['Single process per condition; not a matched performance comparison.',
            'Observed memory and Python collection do not establish native executable eviction or leak freedom.',
            'Minimal compiler priming is explanatory, not a production requirement.']}
    path = Path(request.config.getoption('xmlpath')).parent / 'sequential-public-residency.json'
    with path.open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')
    _compare_original(first, expected)
    _compare_original(second, expected_changed)
    assert all(released.values()), released


def test_sequential_residency_observer_retention_control(request):
    scalar, batch, cfg, args = fixture(3)
    gpu = bool(tf.config.list_logical_devices('GPU'))
    result = public.estimate_sequential_map_covariance(scalar, args[0],
        batched_value_and_score_fn=batch, scale=args[1], config=cfg).payload()
    stages, mappings = {}, {}
    with GPUProcessMonitor(gpu) as sharing:
        for index in range(11):
            stages[str(index)] = memory_snapshot(gpu)
            mappings[str(index)] = mapping_snapshot()
    checkpoint = FrozenCheckpoint('3582b4ac', 'sequential_observer_original')
    original = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    original_cfg = original.SequentialMapCovarianceConfig(**dataclasses.asdict(cfg))
    expected = original.estimate_sequential_map_covariance(scalar, args[0],
        batched_value_and_score_fn=batch, scale=args[1], config=original_cfg).payload()
    report = {'schema': 'filter_sequential_residency_observer_control.v1', 'gpu': gpu,
        'stages': stages, 'mappings': mappings, 'calls_between_snapshots': 0,
        'result': result, 'original': expected, 'original_source_sha256': checkpoint.hashes(),
        'gpu_process_observation': sharing.payload(), 'role': 'explanatory_observer_allocation_control',
        'nonclaims': ['Zero numerical calls between snapshots; only observer overhead is studied.',
            'Different fresh processes do not establish native eviction or general leak freedom.']}
    path = Path(request.config.getoption('xmlpath')).parent / 'sequential-residency-observer.json'
    with path.open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')
    _compare_original(result, expected)
