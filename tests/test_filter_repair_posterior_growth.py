"""Bounded native-controller capacity and warm-allocation diagnostics."""

import dataclasses
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
from tests.test_filter_repair_posterior_curvature import reference, stable_hlo
from tests.test_filter_repair_posterior_curvature_extras import pure_fixture
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_filter_repair_quadratic_round_growth import memory


@pytest.mark.parametrize('replicates', [2, 4, 8])
def test_native_posterior_capacity_and_warm_memory(replicates, request):
    directory = Path(request.config.getoption('xmlpath')).parent

    def progress(stage, **values):
        with (directory / 'posterior-growth-progress.jsonl').open('a') as output:
            output.write(json.dumps({'stage': stage, **values}, allow_nan=False) + '\n')

    gpu = bool(tf.config.list_logical_devices('GPU'))
    callback, eligibility, config, args = pure_fixture(5, replicates=replicates)
    changed = (args[0] + .07, args[1] * 1.1, args[2] + 4)
    stages = {'before': memory(gpu)}
    start = time.perf_counter()
    program = make_posterior_curvature_controller(callback, eligibility, 5, config)
    build = time.perf_counter() - start
    stages['built'] = memory(gpu)
    progress('built', seconds=build, memory=stages['built'])
    start = time.perf_counter()
    program.get_concrete_function()
    trace = time.perf_counter() - start
    stages['traced'] = memory(gpu)
    progress('traced', seconds=trace, memory=stages['traced'])

    def execute(values):
        raw = program(*values)
        return posterior_curvature_result(raw, values[0], values[1],
            dataclasses.replace(config, seed=int(values[2]))).payload()

    start = time.perf_counter()
    first = execute(args)
    cold = time.perf_counter() - start
    stages['first_call'] = memory(gpu)
    progress('first_call', seconds=cold, memory=stages['first_call'])
    second = execute(changed)

    def repeat(index):
        values, expected = (args, first) if index % 2 == 0 else (changed, second)
        assert execute(values) == expected

    for index in range(20):
        repeat(index)
    if gpu:
        tf.config.experimental.reset_memory_stats('GPU:0')
    observations = [{'additional_calls': 0, 'memory': memory(gpu)}]
    progress('warm', **observations[-1])
    for block in range(3):
        for index in range(1000):
            repeat(index)
        observations.append({'additional_calls': (block + 1) * 1000, 'memory': memory(gpu)})
        progress('warm', **observations[-1])
    assert program.experimental_get_tracing_count() == 1
    concrete = program.get_concrete_function()
    graph = concrete.graph.as_graph_def()
    nodes = len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function)
    hlo = program.experimental_get_compiler_ir(*args)(stage='hlo')
    assert stable_hlo(hlo) == stable_hlo(program.experimental_get_compiler_ir(*changed)(stage='hlo'))
    operands = len(re.findall(r'\bparameter\((\d+)\)', hlo[hlo.rfind('\nENTRY '):]))
    assert operands == 3
    expected, hashes = reference(callback, eligibility, config, args)
    expected_changed, _ = reference(callback, eligibility, config, changed)
    report = {'role': 'bounded_native_posterior_capacity_and_memory', 'gpu': gpu,
        'dimension': 5, 'replicate_capacity': replicates, 'build_seconds': build,
        'trace_seconds': trace, 'first_execution_seconds': cold, 'stages': stages,
        'observations': observations, 'graph_nodes': nodes, 'hlo_bytes': len(hlo.encode()),
        'runtime_operands': operands, 'trace_count': program.experimental_get_tracing_count(),
        'result': first, 'changed_result': second, 'original': expected,
        'original_changed': expected_changed, 'original_source_sha256': hashes,
        'nonclaims': ['Capacity changes are diagnostic; no timing ranking.',
            'No exact peak RSS, arbitrary target turnover, executable eviction or general leak-freedom claim.']}
    with (directory / 'posterior-growth.json').open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    _equal_records(first, expected)
    _equal_records(second, expected_changed)
