"""Matched complete fit-suffix costs; no whole-initializer timing claim."""

import hashlib
import time

import pytest
import tensorflow as tf

from bayesfilter.inference.quadratic_geometry_fit_report import (
    geometry_fit_payload,
    geometry_fit_result,
)
from bayesfilter.inference.quadratic_geometry_fit_tf import make_geometry_fit_program
from tests.filter_repair_geometry_fit_reference import (
    invoke_original_suffix,
    original_suffix,
    prefix_context,
    result_suffix,
)
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_geometry_fit import fixture, original_result
from tests.test_filter_repair_quadratic_batch_memory import _memory
from tests.test_filter_repair_quadratic_batches import _equal_records


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('arm', ['before', 'graph', 'xla'])
def test_complete_geometry_fit_costs(arm, dimension, monkeypatch, request):
    target, cfg, args, _ = fixture(dimension, 'nonquadratic', counter=False)
    _, _, changed, _ = fixture(dimension, 'nonquadratic', counter=False, shift=.07)
    function, body_hash, checkpoint = original_suffix()
    contexts = (prefix_context(target, cfg, args), prefix_context(target, cfg, changed))
    gpu = bool(tf.config.list_logical_devices('GPU'))
    stages = {'before': _memory(gpu)}
    start = time.perf_counter()
    program = None if arm == 'before' else make_geometry_fit_program(target, dimension,
        args[2].shape[1], args[3].shape[0], args[6].shape[0], cfg, jit_compile=arm == 'xla')
    build = time.perf_counter() - start
    stages['built'] = _memory(gpu)
    trace = None
    if program is not None:
        start = time.perf_counter()
        program.get_concrete_function()
        trace = time.perf_counter() - start
    stages['traced'] = _memory(gpu)

    def execute(inputs, context):
        if gpu:
            tf.config.experimental.reset_memory_stats('GPU:0')
        start = time.perf_counter()
        if program is None:
            result = invoke_original_suffix(function, context)
            call_seconds = time.perf_counter() - start
            result_seconds = 0.
        else:
            raw = program(*inputs)
            raw['status'].numpy()  # Synchronize the native call before separating host costs.
            call_seconds = time.perf_counter() - start
            result_start = time.perf_counter()
            result = geometry_fit_result(raw, cfg, inputs[0], inputs[1], context[2], holdout_rows=inputs[6].shape[0])
            result_seconds = time.perf_counter() - result_start
        # Both arms build the result class, diagnostics, artifact hash and full
        # array payload. Source-owned summaries are included in this boundary.
        payload_start = time.perf_counter()
        payload = clean(result.payload(include_arrays=True) if program is None else geometry_fit_payload(result))
        fields = clean(result_suffix(result))
        payload_seconds = time.perf_counter() - payload_start
        total_seconds = time.perf_counter() - start
        return fields, payload, {'seconds': total_seconds, 'memory': _memory(gpu),
            'components': {'source_call_seconds': call_seconds, 'native_result_construction_seconds': result_seconds,
                           'full_payload_seconds': payload_seconds}}

    samples, first, first_payload = [], None, None
    for _ in range(21):
        result, payload, sample = execute(args, contexts[0])
        if first is None:
            first, first_payload = result, payload
        else:
            assert result == first
            assert payload == first_payload
        samples.append(sample)
    stages['measured'] = _memory(gpu)
    second, second_payload, changed_cost = execute(changed, contexts[1])
    growth = [{'additional_calls': 0, 'memory': _memory(gpu)}]
    for block in range(3):
        for index in range(1000):
            inputs, context, expected_fields, expected_payload = ((args, contexts[0], first, first_payload)
                if index % 2 == 0 else (changed, contexts[1], second, second_payload))
            fields, payload, _ = execute(inputs, context)
            assert fields == expected_fields
            assert payload == expected_payload
        growth.append({'additional_calls': (block + 1) * 1000, 'memory': _memory(gpu)})
    stages['after_warm_reuse'] = _memory(gpu)
    expected, original_payload, _ = original_result(target, cfg, args, monkeypatch)
    expected_changed, original_changed_payload, _ = original_result(target, cfg, changed, monkeypatch)
    nodes, traces, hlo = None, None, ''
    if program is not None:
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function)
        traces = program.experimental_get_tracing_count()
        assert traces == 1
        if arm == 'xla':
            hlo = program.experimental_get_compiler_ir(*args)(stage='hlo')
        else:
            assert not any(fn.attr['_XlaMustCompile'].b for fn in graph.library.function if '_XlaMustCompile' in fn.attr)
    save(request, 'geometry-fit-memory.json', {'role': 'descriptive_complete_geometry_fit_suffix_cost',
        'baseline': '3582b4ac', 'arm': arm, 'dimension': dimension, 'gpu': gpu, 'jit_compile': arm == 'xla',
        'original_source_sha256': checkpoint.hashes(), 'original_unmodified_suffix_ast_sha256': body_hash,
        'input_sha256': [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in args],
        'build_seconds': build, 'trace_seconds': trace, 'stages': stages, 'samples': samples,
        'growth_observations': growth,
        'result': first, 'changed_result': second, 'changed_cost': changed_cost,
        'original_result': expected, 'original_changed_result': expected_changed,
        'payload': first_payload, 'changed_payload': second_payload,
        'original_payload': original_payload, 'original_changed_payload': original_changed_payload,
        'trace_count': traces, 'graph_nodes': nodes, 'hlo_bytes': len(hlo.encode()),
        'non_jit_role': 'explicit_reference_exception' if arm != 'xla' else None,
        'timing_scope': 'Original unchanged suffix AST versus enclosing native program, both with full result/hash/payload. Identical prefix preparation excluded.',
        'nonclaims': ['Single-process descriptive suffix comparison; no public whole-initializer or timing ranking claim.']})
    _equal_records(first, expected)
    _equal_records(second, expected_changed)
    for actual, original in ((first_payload, original_payload), (second_payload, original_changed_payload)):
        # Floating point payload hashes belong to each arm's own record. Every
        # numerical field and all remaining metadata still compare to original.
        actual['diagnostics'].pop('artifact_hash')
        original['diagnostics'].pop('artifact_hash')
        _equal_records(actual, original)
