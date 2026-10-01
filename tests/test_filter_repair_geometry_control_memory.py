"""Diagnostic costs of complete proposal/acceptance computation and reporting."""

import hashlib
import re
import time

import pytest
import tensorflow as tf

from bayesfilter.inference.quadratic_geometry_control_report import (
    center_refinement_report,
)
from bayesfilter.inference.quadratic_geometry_control_tf import (
    make_center_refinement_program,
)
from tests.test_filter_repair_geometry_control import (
    clean,
    fixture,
    reference,
    save,
    source,
    stable_hlo,
)
from tests.test_filter_repair_quadratic_batch_memory import _memory
from tests.test_filter_repair_quadratic_batches import _equal_records


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('arm', ['before', 'graph', 'xla'])
def test_complete_center_proposal_costs(arm, dimension, request):
    checkpoint, baseline = source('3582b4ac')
    callback, cfg, args, _, _ = fixture(dimension, 'interior', record_calls=False)
    gpu = bool(tf.config.list_logical_devices('GPU'))
    stages = {'before': _memory(gpu)}
    start = time.perf_counter()
    program = None if arm == 'before' else make_center_refinement_program(callback, dimension, cfg, jit_compile=arm == 'xla')
    build = time.perf_counter() - start
    stages['built'] = _memory(gpu)
    trace = None
    if program is not None:
        start = time.perf_counter()
        program.get_concrete_function()
        trace = time.perf_counter() - start
    stages['traced'] = _memory(gpu)

    def execute(inputs):
        if gpu:
            tf.config.experimental.reset_memory_stats('GPU:0')
        start = time.perf_counter()
        if program is None:
            result = reference(baseline, callback, cfg, inputs)
        else:
            result = center_refinement_report(program(*inputs), cfg, inputs[4], inputs[5])
        result = clean(result)
        return result, {'seconds': time.perf_counter() - start, 'memory': _memory(gpu)}

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
    changed = (args[0] + .03, args[1] * 1.1, args[2] * 1.03, args[3] * .9, args[4] - .1, args[5] * 1.1)
    second, changed_cost = execute(changed)
    # Sparse observation after identical fixed-signature workloads. This is a
    # bounded plateau diagnostic, not proof of arbitrary executable eviction.
    growth = [{'additional_calls': 0, 'memory': _memory(gpu)}]
    if gpu:
        tf.config.experimental.reset_memory_stats('GPU:0')
    for block in range(3):
        for index in range(1000):
            inputs, expected_record = (args, first) if index % 2 == 0 else (changed, second)
            if program is None:
                actual = reference(baseline, callback, cfg, inputs)
            else:
                actual = center_refinement_report(program(*inputs), cfg, inputs[4], inputs[5])
            assert clean(actual) == expected_record
        growth.append({'additional_calls': (block + 1) * 1000, 'memory': _memory(gpu)})
    stages['after_warm_reuse'] = _memory(gpu)
    nodes, hlo, traces = None, '', None
    if program is not None:
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function)
        traces = program.experimental_get_tracing_count()
        assert traces == 1
        if arm == 'xla':
            hlo = program.experimental_get_compiler_ir(*args)(stage='hlo')
            assert stable_hlo(hlo) == stable_hlo(program.experimental_get_compiler_ir(*changed)(stage='hlo'))
            assert len(re.findall(r'\bparameter\((\d+)\)', hlo[hlo.rfind('\nENTRY '):])) == 6
        else:
            assert not any(fn.attr['_XlaMustCompile'].b for fn in graph.library.function if '_XlaMustCompile' in fn.attr)
    expected = clean(reference(baseline, callback, cfg, args))
    expected_changed = clean(reference(baseline, callback, cfg, changed))
    save(request, 'geometry-control-memory.json', {
        'role': 'descriptive_complete_center_proposal_cost', 'baseline': '3582b4ac',
        'arm': arm, 'dimension': dimension, 'gpu': gpu, 'jit_compile': arm == 'xla',
        'original_source_sha256': checkpoint.hashes(),
        'input_sha256': [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in args],
        'build_seconds': build, 'trace_seconds': trace, 'stages': stages, 'samples': samples,
        'growth_observations': growth,
        'result': first, 'changed_result': second, 'changed_cost': changed_cost,
        'original_result': expected, 'original_changed_result': expected_changed,
        'trace_count': traces, 'graph_nodes': nodes, 'hlo_bytes': len(hlo.encode()),
        'non_jit_role': 'explicit_reference_exception' if arm != 'xla' else None,
        'timing_scope': 'Complete center proposal, target, acceptance and reporting; factory/trace included in total cold.',
        'nonclaims': ['One process per arm/dimension; descriptive only.',
            'Public integration, broader geometry/iterative control and complete endpoint costs remain open.',
            'No exact process-peak, arbitrary target turnover, posterior or HMC claim.']})
    _equal_records(first, expected)
    _equal_records(second, expected_changed)
