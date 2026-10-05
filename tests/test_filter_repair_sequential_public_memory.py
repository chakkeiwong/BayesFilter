"""Complete sequential API cost diagnostic; original records remain mandatory."""

import dataclasses
import hashlib
import json
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_controller_tf as runtime
from bayesfilter.inference import sequential_map_covariance as public
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_lifecycle_original import _compare_original

D = tf.float64


def fixture(dimension):
    cfg = public.SequentialMapCovarianceConfig(locator_policy='center_first', max_attempts=3,
        search_sample_count=4, regression_sample_count=24, terminal_sample_count=24,
        max_exact_evaluations=256, terminal_score_max_abs=1e-5,
        record_refinement_movement_diagnostics=True)

    def batch(rows):
        return -.5 * tf.reduce_sum(rows ** 2, axis=1) - .1 * tf.reduce_sum(rows ** 4, axis=1), -rows - .4 * rows ** 3

    def scalar(row):
        values, scores = batch(row[None])
        return values[0], scores[0]

    return scalar, batch, cfg, (tf.linspace(tf.constant(.002, D), tf.constant(.004, D), dimension)[None],
        tf.ones([dimension], D))


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('arm', ['prior', 'graph', 'xla'])
def test_complete_public_sequential_costs(arm, dimension, monkeypatch, request):
    scalar, batch, cfg, args = fixture(dimension)
    checkpoint = FrozenCheckpoint('48acf5e96', 'sequential_public_cost_prior')
    prior = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    original_checkpoint = FrozenCheckpoint('3582b4ac', 'sequential_public_cost_original')
    original = original_checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    gpu = bool(tf.config.list_logical_devices('GPU'))
    module = prior if arm == 'prior' else public
    options = module.SequentialMapCovarianceConfig(**dataclasses.asdict(cfg))
    if arm == 'graph':
        monkeypatch.setattr(public, 'sequential_controller',
            lambda *a, **kw: runtime.sequential_controller(*a, **kw, jit_compile=False))
    runtime.clear_sequential_controller_cache()
    stages = {'prepared': memory_snapshot(gpu)}
    owner, compiled = None, None
    with GPUProcessMonitor(gpu) as sharing:
        begin = time.perf_counter()
        if arm != 'prior':
            owner = public.sequential_controller(scalar, batch, None, 1, dimension,
                cfg, cfg.search_sample_count, device=args[0].device)
            compiled = owner.compiled
        build_seconds = time.perf_counter() - begin
        stages['built'] = memory_snapshot(gpu)
        begin = time.perf_counter()
        if compiled is not None:
            compiled.get_concrete_function()
        trace_seconds = time.perf_counter() - begin
        stages['traced'] = memory_snapshot(gpu)

        def execute(values):
            if gpu:
                tf.config.experimental.reset_memory_stats('GPU:0')
            start = time.perf_counter()
            payload = module.estimate_sequential_map_covariance(scalar, values[0],
                batched_value_and_score_fn=batch, scale=values[1], config=options).payload()
            elapsed = time.perf_counter() - start
            return payload, {'seconds': elapsed, 'memory': memory_snapshot(gpu)}

        first, cold = execute(args)
        stages['cold'] = memory_snapshot(gpu)
        samples = []
        for _ in range(20):
            actual, sample = execute(args)
            assert actual == first
            samples.append(sample)
        stages['warm'] = memory_snapshot(gpu)
        changed = (args[0] + .0005, args[1] * 1.1)
        second, changed_cost = execute(changed)
        stages['changed'] = memory_snapshot(gpu)
    nodes, graph_bytes, hlo_bytes, traces = None, None, None, None
    if compiled is not None:
        assert owner is runtime._LAST_CONTROLLER[4]
        graph = compiled.get_concrete_function().graph.as_graph_def()
        nodes = len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function)
        graph_bytes = graph.ByteSize()
        traces = compiled.experimental_get_tracing_count()
        assert traces == 1
        if arm == 'xla':
            hlo = compiled.experimental_get_compiler_ir(*args)(stage='hlo')
            hlo_bytes = len(hlo.encode())
        else:
            assert not compiled.get_concrete_function().function_def.attr['_XlaMustCompile'].b
    original_cfg = original.SequentialMapCovarianceConfig(**dataclasses.asdict(cfg))
    expected = original.estimate_sequential_map_covariance(scalar, args[0],
        batched_value_and_score_fn=batch, scale=args[1], config=original_cfg).payload()
    expected_changed = original.estimate_sequential_map_covariance(scalar, changed[0],
        batched_value_and_score_fn=batch, scale=changed[1], config=original_cfg).payload()
    report = {'schema': 'filter_sequential_public_cost.v1', 'arm': arm, 'dimension': dimension,
        'numerical_authority': '3582b4ac', 'mechanism_baseline': '48acf5e96',
        'prior_source_sha256': checkpoint.hashes(), 'original_source_sha256': original_checkpoint.hashes(),
        'input_sha256': [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in args],
        'changed_input_sha256': [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in changed],
        'config': dataclasses.asdict(cfg), 'gpu': gpu, 'jit_compile': arm == 'xla',
        'execution_role': 'mixed_host_and_compiled_prior' if arm == 'prior' else (
            'explicit_graph_reference_with_declared_compiled_dependencies' if arm == 'graph' else 'public_xla_default'),
        'build_seconds': build_seconds, 'trace_seconds': trace_seconds, 'stages': stages,
        'cold': cold, 'samples': samples, 'changed_cost': changed_cost,
        'result': first, 'changed_result': second, 'original_result': expected,
        'original_changed_result': expected_changed, 'trace_count': traces,
        'graph_nodes': nodes, 'graph_bytes': graph_bytes, 'hlo_bytes': hlo_bytes,
        'gpu_process_observation': sharing.payload(),
        'timing_scope': 'Entire public call/payload; construction and tracing included in total cold.',
        'nonclaims': ['CPU is an explicit reference; outer graph mode is non-default.',
            'Graph/XLA retain declared solvers/dependencies; not an identical-graph compiler ablation.',
            'Observed RSS/PSS and allocator counts are not exact peaks or leak-freedom evidence.',
            'Three processes per condition do not establish a general performance or scientific ranking.']}
    path = Path(request.config.getoption('xmlpath')).parent / 'sequential-public-memory.json'
    with path.open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')
    _compare_original(first, expected)
    _compare_original(second, expected_changed)
