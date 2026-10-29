"""Full public posterior costs in independent processes, including host reporting."""

import dataclasses
import hashlib
import json
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import posterior_curvature_refinement as public
from bayesfilter.inference import posterior_curvature_tf as runtime
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_posterior_curvature import reference, stable_hlo
from tests.test_filter_repair_posterior_curvature_extras import pure_fixture
from tests.test_filter_repair_quadratic_batches import _equal_records


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('arm', ['prior', 'graph', 'xla'])
def test_complete_public_posterior_costs(arm, dimension, monkeypatch, request):
    callback, eligibility, cfg, args = pure_fixture(dimension)
    checkpoint = FrozenCheckpoint('dba39e048', 'posterior_public_cost_prior')
    prior = checkpoint.load('bayesfilter.inference.posterior_curvature_refinement')
    gpu = bool(tf.config.list_logical_devices('GPU'))
    module = prior if arm == 'prior' else public
    options = module.PosteriorCurvatureRefinementConfig(**dataclasses.asdict(cfg))
    if arm == 'graph':
        monkeypatch.setattr(public, 'posterior_curvature_controller',
            lambda *a, **kw: runtime.posterior_curvature_controller(*a, **kw, jit_compile=False))
    runtime.clear_posterior_curvature_controller_cache()
    stages = {'prepared': memory_snapshot(gpu)}
    program = None
    with GPUProcessMonitor(gpu) as sharing:
        start = time.perf_counter()
        if arm != 'prior':
            program = public.posterior_curvature_controller(callback, eligibility, dimension, cfg)
        build_seconds = time.perf_counter() - start
        stages['built'] = memory_snapshot(gpu)
        start = time.perf_counter()
        if program is not None:
            program.get_concrete_function()
        trace_seconds = time.perf_counter() - start
        stages['traced'] = memory_snapshot(gpu)

        def execute(values, selected):
            if gpu:
                tf.config.experimental.reset_memory_stats('GPU:0')
            begin = time.perf_counter()
            result = module.refine_posterior_local_curvature(callback, values[0], values[1],
                batched_eligibility_fn=eligibility, config=selected).payload()
            elapsed = time.perf_counter() - begin
            return result, {'seconds': elapsed, 'memory': memory_snapshot(gpu)}

        first, cold = execute(args, options)
        stages['cold'] = memory_snapshot(gpu)
        samples = []
        for _ in range(20):
            actual, sample = execute(args, options)
            assert actual == first
            samples.append(sample)
        stages['warm'] = memory_snapshot(gpu)
        changed = (args[0] + .07, args[1] * 1.1, args[2] + 4)
        changed_options = dataclasses.replace(options, seed=int(changed[2]))
        second, changed_cost = execute(changed, changed_options)
        stages['changed'] = memory_snapshot(gpu)
    # Compiler-IR export and independent references follow all timed snapshots.
    nodes, graph_bytes, hlo_bytes, traces = None, None, None, None
    if program is not None:
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function)
        graph_bytes = graph.ByteSize()
        traces = program.experimental_get_tracing_count()
        assert traces == 1
        if arm == 'xla':
            hlo = program.experimental_get_compiler_ir(*args)(stage='hlo')
            changed_hlo = program.experimental_get_compiler_ir(*changed)(stage='hlo')
            assert stable_hlo(hlo) == stable_hlo(changed_hlo)
            hlo_bytes = len(hlo.encode())
        else:
            assert not program.get_concrete_function().function_def.attr['_XlaMustCompile'].b
    expected, original_hashes = reference(callback, eligibility, cfg, args)
    expected_changed, _ = reference(callback, eligibility, cfg, changed)
    report = {'schema': 'filter_posterior_public_cost.v1', 'arm': arm, 'dimension': dimension,
        'numerical_authority': '3582b4ac', 'mechanism_baseline': 'dba39e048',
        'prior_source_sha256': checkpoint.hashes(), 'original_source_sha256': original_hashes,
        'input_sha256': [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in args],
        'changed_input_sha256': [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in changed],
        'config': dataclasses.asdict(cfg), 'gpu': gpu, 'jit_compile': arm == 'xla',
        'execution_role': 'mixed_host_and_compiled_prior' if arm == 'prior' else (
            'explicit_graph_reference' if arm == 'graph' else 'public_xla_default'),
        'build_seconds': build_seconds, 'trace_seconds': trace_seconds, 'stages': stages,
        'cold': cold, 'samples': samples, 'changed_cost': changed_cost,
        'result': first, 'changed_result': second,
        'original_result': expected, 'original_changed_result': expected_changed,
        'trace_count': traces, 'graph_nodes': nodes, 'graph_bytes': graph_bytes, 'hlo_bytes': hlo_bytes,
        'gpu_process_observation': sharing.payload(),
        'timing_scope': 'Entire public call and payload; factory/trace separate and included in total cold.',
        'nonclaims': ['CPU is an explicit reference; graph is an explicit non-default reference.',
            'Graph/XLA retain their declared solvers; not an identical-graph compiler ablation.',
            'RSS/PSS snapshots are observations, not exact peaks or proof of leak freedom.',
            'Three process replicates support cost triggers, not a general performance or scientific ranking.']}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'posterior-public-memory.json').open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')
    _equal_records(first, expected)
    _equal_records(second, expected_changed)
