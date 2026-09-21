"""Complete native controller costs, preserving original records in each arm."""

import dataclasses
import hashlib
import json
import re
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.posterior_curvature_report import posterior_curvature_result
from bayesfilter.inference.posterior_curvature_tf import (
    make_posterior_curvature_controller,
)
from tests.test_filter_repair_posterior_curvature import original, reference, stable_hlo
from tests.test_filter_repair_posterior_curvature_extras import pure_fixture
from tests.test_filter_repair_quadratic_batch_memory import _memory
from tests.test_filter_repair_quadratic_batches import _equal_records


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('arm', ['before', 'graph', 'xla'])
def test_complete_native_posterior_costs(arm, dimension, request):
    checkpoint, baseline = original()
    gpu = bool(tf.config.list_logical_devices('GPU'))
    callback, eligibility, cfg, args = pure_fixture(dimension)
    stages = {'before': _memory(gpu)}
    started = time.perf_counter()
    program = None if arm == 'before' else make_posterior_curvature_controller(callback, eligibility, dimension, cfg, jit_compile=arm == 'xla')
    build_seconds = time.perf_counter() - started
    stages['built'] = _memory(gpu)
    trace_seconds = None
    if program is not None:
        started = time.perf_counter()
        program.get_concrete_function()
        trace_seconds = time.perf_counter() - started
    stages['traced'] = _memory(gpu)

    def execute(arguments):
        if gpu:
            tf.config.experimental.reset_memory_stats('GPU:0')
        started = time.perf_counter()
        options = dataclasses.replace(cfg, seed=int(arguments[2]))
        if program is None:
            result = baseline.refine_posterior_local_curvature(callback, arguments[0], arguments[1],
                batched_eligibility_fn=eligibility,
                config=baseline.PosteriorCurvatureRefinementConfig(**dataclasses.asdict(options))).payload()
        else:
            raw = program(*arguments)
            result = posterior_curvature_result(raw, arguments[0], arguments[1], options).payload()
        return result, {'seconds': time.perf_counter() - started, 'memory': _memory(gpu)}

    samples, first = [], None
    for _ in range(21):
        result, sample = execute(args)
        if first is None:
            first = result
        else:
            assert first == result
        samples.append(sample)
        del result
    stages['measured'] = _memory(gpu)
    changed = (args[0] + .07, args[1] * 1.1, args[2] + 4)
    second, changed_cost = execute(changed)
    nodes, hlo, traces = None, '', None
    if program is not None:
        concrete = program.get_concrete_function()
        graph = concrete.graph.as_graph_def()
        nodes = len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function)
        traces = program.experimental_get_tracing_count()
        assert traces == 1
        if arm == 'xla':
            hlo = program.experimental_get_compiler_ir(*args)(stage='hlo')
            assert stable_hlo(hlo) == stable_hlo(program.experimental_get_compiler_ir(*changed)(stage='hlo'))
            assert len(re.findall(r'\bparameter\((\d+)\)', hlo[hlo.rfind('\nENTRY '):])) == 3
        else:
            assert not any(fn.attr['_XlaMustCompile'].b for fn in graph.library.function if '_XlaMustCompile' in fn.attr)
    expected, _ = reference(callback, eligibility, cfg, args)
    expected_changed, _ = reference(callback, eligibility, cfg, changed)
    report = {'role': 'descriptive_complete_native_posterior_curvature_cost', 'baseline': '3582b4ac',
        'arm': arm, 'dimension': dimension, 'gpu': gpu, 'jit_compile': arm == 'xla',
        'original_source_sha256': checkpoint.hashes(),
        'input_sha256': [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in args],
        'build_seconds': build_seconds, 'trace_seconds': trace_seconds, 'stages': stages,
        'samples': samples, 'result': first, 'changed_result': second, 'changed_cost': changed_cost,
        'original_result': expected, 'original_changed_result': expected_changed,
        'trace_count': traces, 'graph_nodes': nodes, 'hlo_bytes': len(hlo.encode()),
        'non_jit_role': 'explicit_graph_reference_exception' if arm != 'xla' else None,
        'timing_scope': 'Complete fixed-center calculation and payload; factory/tracing reported separately and included in total cold.',
        'nonclaims': ['One process per arm/dimension; descriptive only.',
            'Public integration and ill-conditioned rejected diagnostic remain open.',
            'No exact process-peak, arbitrary target turnover, posterior or HMC claim.']}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'posterior-native-memory.json').open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')
    _equal_records(first, expected)
    _equal_records(second, expected_changed)
