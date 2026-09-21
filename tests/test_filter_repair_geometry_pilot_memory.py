"""Exact original pilot suffix versus native evaluation, with full records."""

import ast
import hashlib
import time
from functools import lru_cache

import pytest
import tensorflow as tf

from bayesfilter.inference.quadratic_geometry_pilot_report import geometry_pilot_report
from bayesfilter.inference.quadratic_geometry_pilot_tf import (
    make_geometry_pilot_program,
)
from tests.test_filter_repair_geometry_control import save, source
from tests.test_filter_repair_geometry_pilot import fixture, record_pilot, reference
from tests.test_filter_repair_quadratic_batch_memory import _memory
from tests.test_filter_repair_quadratic_batches import _equal_records


@lru_cache(maxsize=1)
def original_suffix():
    checkpoint, module = source('3582b4ac')
    tree = ast.parse(checkpoint.sources['bayesfilter/inference/quadratic_geometry.py'])
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_pilot_q_basis')
    index = next(index for index, node in enumerate(original.body) if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'sketch')
    body = original.body[index:]
    digest = hashlib.sha256(ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False).encode()).hexdigest()
    names = ('value_and_score_fn', 'batched_value_and_score_fn', 'center', 'scale', 'rank', 'cfg',
             'center_score_z', 'start_index', 'dim', 'count', 'directions')
    function = ast.FunctionDef(name='original_prepared_pilot', args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=name) for name in names], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=body, decorator_list=[])
    namespace = dict(vars(module))
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(tree, '3582b4ac:prepared_pilot', 'exec'), namespace)  # noqa: S102 - unchanged diagnostic source AST.
    return namespace['original_prepared_pilot'], digest, checkpoint


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('arm', ['before', 'graph', 'xla'])
@pytest.mark.parametrize('batched', [False, True])
def test_complete_pilot_costs(arm, dimension, batched, request):
    callback, cfg, args, directions, _, _ = fixture(dimension, 'nonquadratic', batched=batched, counter=False)
    changed = (args[0] + .02, args[1] * 1.1, -args[2], args[3], args[4] * .8)
    original, body_hash, checkpoint = original_suffix()
    host_inputs = tuple(tuple(value.numpy() for value in inputs) for inputs in (args, changed))
    gpu = bool(tf.config.list_logical_devices('GPU'))
    stages = {'before': _memory(gpu)}
    start = time.perf_counter()
    program = None if arm == 'before' else make_geometry_pilot_program(callback, dimension,
        dimension - 1, 9, batched=batched, jit_compile=arm == 'xla')
    build = time.perf_counter() - start
    stages['built'] = _memory(gpu)
    trace = None
    if program is not None:
        start = time.perf_counter()
        program.get_concrete_function()
        trace = time.perf_counter() - start
    stages['traced'] = _memory(gpu)

    def execute(inputs, host):
        if gpu:
            tf.config.experimental.reset_memory_stats('GPU:0')
        start = time.perf_counter()
        if program is None:
            result = original(callback, callback if batched else None, host[0], host[1],
                dimension - 1, cfg, host[4], 1, dimension, 9, host[2])
        else:
            result = geometry_pilot_report(program(*inputs), rank=dimension - 1,
                requested_direction_count=9, batched=batched)
        materialized = record_pilot(result)
        return materialized, {'seconds': time.perf_counter() - start, 'memory': _memory(gpu)}

    samples, first = [], None
    for _ in range(21):
        actual, sample = execute(args, host_inputs[0])
        if first is None:
            first = actual
        else:
            assert actual == first
        samples.append(sample)
    stages['measured'] = _memory(gpu)
    second, changed_cost = execute(changed, host_inputs[1])
    expected, _ = reference(callback, cfg, args, directions, batched=batched)
    expected_changed, _ = reference(callback, cfg, changed, -directions, batched=batched)
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
    lane = 'batch' if batched else 'scalar'
    save(request, f'geometry-pilot-memory-{lane}.json', {'role': f'descriptive_prepared_geometry_pilot_{lane}_cost',
        'baseline': '3582b4ac', 'arm': arm, 'dimension': dimension, 'gpu': gpu, 'jit_compile': arm == 'xla',
        'batched': batched, 'original_source_sha256': checkpoint.hashes(), 'original_suffix_ast_sha256': body_hash,
        'input_sha256': [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in args],
        'build_seconds': build, 'trace_seconds': trace, 'stages': stages, 'samples': samples,
        'result': first, 'changed_result': second, 'changed_cost': changed_cost,
        'original_result': expected, 'original_changed_result': expected_changed,
        'trace_count': traces, 'graph_nodes': nodes, 'hlo_bytes': len(hlo.encode()),
        'non_jit_role': 'explicit_reference_exception' if arm != 'xla' else None,
        'timing_scope': 'Same normalized directions; complete pilot evaluation/sketch/basis/candidate/report work in both arms.',
        'nonclaims': ['Single process per arm/extent/lane; no timing ranking, full initializer, or general memory bound.']})
    _equal_records(first, expected)
    _equal_records(second, expected_changed)
