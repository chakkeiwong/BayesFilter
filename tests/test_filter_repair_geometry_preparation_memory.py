"""Fresh-process descriptive costs for matched original preparation boundaries."""

import hashlib
import time

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.quadratic_geometry_prepare_tf import (
    make_direction_preparation_program,
    make_geometry_design_program,
    make_geometry_partition_program,
)
from tests.test_filter_repair_geometry_control import clean, save, source
from tests.test_filter_repair_geometry_preparation import (
    fixture,
    original_preparation,
    partition_record,
    partition_reference,
)
from tests.test_filter_repair_quadratic_batch_memory import _memory
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('arm', ['before', 'graph', 'xla'])
@pytest.mark.parametrize('kind', ['directions', 'partition', 'design_scalar', 'design_batch'])
def test_preparation_costs(kind, arm, dimension, request):
    args, required, fraction = fixture(dimension, 'partial')
    checkpoint, module = source('3582b4ac')
    original, body_hash, _ = original_preparation('directions' if kind == 'directions' else 'partition')
    batched = kind == 'design_batch'
    def target(point):
        cloud = point if batched else point[None]
        score = -cloud * (tf.cast(tf.range(dimension), D) + 2.)
        value = .5 * tf.reduce_sum(cloud * score, axis=1)
        return (value, score) if batched else (value[0], score[0])

    if kind == 'directions':
        inputs, changed = (args[0],), (-args[0],)
        build_program = lambda: make_direction_preparation_program(dimension, 24, jit_compile=arm == 'xla')
        def baseline(host):
            return clean(original(host[0]))
        def report(raw):
            return clean(raw['directions'][:int(raw['count'])])
    elif kind == 'partition':
        inputs = args
        changed = (args[0] + .01, args[1] + .1, args[2] * .95, args[3] * 1.05, args[4])
        build_program = lambda: make_geometry_partition_program(dimension, 24, required, fraction, jit_compile=arm == 'xla')
        def baseline(host):
            from types import SimpleNamespace
            offsets, values, scores, scale, order = host
            finite = np.isfinite(values) & np.all(np.isfinite(scores), axis=1)
            count = int(finite.sum())
            return clean(original(offsets, values, scores, scale, finite, count,
                SimpleNamespace(permutation=lambda n: order[:n].copy()),
                SimpleNamespace(holdout_fraction=fraction), required))
        report = partition_record
    else:
        body_hash = None
        inputs = (tf.cast(tf.range(dimension), D) * .1, args[3], args[0])
        changed = (inputs[0] + .1, inputs[1] * 1.05, -inputs[2])
        build_program = lambda: make_geometry_design_program(target, dimension, 24, batched=batched, jit_compile=arm == 'xla')
        def baseline(host):
            points = host[0][None] + host[2] * host[1][None]
            values, scores = module._evaluate_values_scores(target, points, batched_value_and_score_fn=target if batched else None)
            return clean({'positions': points, 'values': values, 'scores': scores})
        def report(raw):
            return clean({name: raw[name] for name in ('positions', 'values', 'scores')})

    host, changed_host = (tuple(value.numpy() for value in row) for row in (inputs, changed))
    gpu = bool(tf.config.list_logical_devices('GPU'))
    stages = {'before': _memory(gpu)}
    start = time.perf_counter()
    program = None if arm == 'before' else build_program()
    build = time.perf_counter() - start
    stages['built'] = _memory(gpu)
    trace = None
    if program is not None:
        start = time.perf_counter()
        program.get_concrete_function()
        trace = time.perf_counter() - start
    stages['traced'] = _memory(gpu)
    def execute(tensors, arrays):
        if gpu:
            tf.config.experimental.reset_memory_stats('GPU:0')
        start = time.perf_counter()
        result = baseline(arrays) if program is None else report(program(*tensors))
        return result, {'seconds': time.perf_counter() - start, 'memory': _memory(gpu)}

    samples, first = [], None
    for _ in range(21):
        actual, sample = execute(inputs, host)
        if first is None:
            first = actual
        else:
            assert first == actual
        samples.append(sample)
    stages['measured'] = _memory(gpu)
    second, changed_cost = execute(changed, changed_host)
    expected, expected_changed = baseline(host), baseline(changed_host)
    nodes, traces, hlo = None, None, ''
    if program is not None:
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function)
        traces = program.experimental_get_tracing_count()
        assert traces == 1
        if arm == 'xla':
            hlo = program.experimental_get_compiler_ir(*inputs)(stage='hlo')
        else:
            assert not any(fn.attr['_XlaMustCompile'].b for fn in graph.library.function if '_XlaMustCompile' in fn.attr)
    save(request, f'geometry-preparation-memory-{kind}.json', {
        'role': f'descriptive_geometry_preparation_{kind}_cost', 'baseline': '3582b4ac',
        'arm': arm, 'dimension': dimension, 'gpu': gpu, 'jit_compile': arm == 'xla',
        'original_source_sha256': checkpoint.hashes(), 'original_body_ast_sha256': body_hash,
        'input_sha256': [hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest() for value in inputs],
        'build_seconds': build, 'trace_seconds': trace, 'stages': stages, 'samples': samples,
        'result': first, 'changed_result': second, 'changed_cost': changed_cost,
        'original_result': expected, 'original_changed_result': expected_changed,
        'trace_count': traces, 'graph_nodes': nodes, 'hlo_bytes': len(hlo.encode()),
        'non_jit_role': 'explicit_reference_exception' if arm != 'xla' else None,
        'timing_scope': 'Same prepared inputs and common complete outputs; all active direction rows, fitting/holdout rows, or evaluated cloud. Fixed-capacity metadata remains extra candidate work.',
        'nonclaims': ['One process per arm/extent/kind; no whole initializer, statistical timing ranking, or general memory bound.']})
    _equal_records(first, expected)
    _equal_records(second, expected_changed)


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('kind', ['directions', 'partition'])
def test_preparation_components_and_changing_input_reuse(kind, dimension, request):
    """Localize standalone report overhead; preserve bounded reuse observations."""
    first, required, fraction = fixture(dimension)
    second, _, _ = fixture(dimension, 'partial')
    original, _, _ = original_preparation('directions' if kind == 'directions' else 'partition')
    gpu = bool(tf.config.list_logical_devices('GPU'))
    stages = {'before_build': _memory(gpu)}
    if kind == 'directions':
        altered = first[0].numpy()
        altered[0] = 1e-160
        altered[1] = 0.
        inputs = ((first[0],), (tf.constant(altered, D),))
        program = make_direction_preparation_program(dimension, 24)
        reference = [clean(original(row[0].numpy())) for row in inputs]
        def materialize(raw):
            return clean(raw['directions'][:int(raw['count'])])
        def synchronize(raw):
            return int(raw['count'])
    else:
        inputs = (first, second)
        program = make_geometry_partition_program(dimension, 24, required, fraction)
        reference = [clean(partition_reference(row, required, fraction)) for row in inputs]
        materialize = partition_record
        def synchronize(raw):
            return int(raw['finite_count'])
    stages['after_build'] = _memory(gpu)
    program.get_concrete_function()
    stages['after_trace'] = _memory(gpu)
    samples = []
    for index in range(21):
        start = time.perf_counter()
        result = program(*inputs[index % 2])
        synchronize(result)
        called = time.perf_counter()
        record = materialize(result)
        completed = time.perf_counter()
        _equal_records(record, reference[index % 2])
        samples.append({'native_synchronized_seconds': called - start,
                        'report_seconds': completed - called, 'total_seconds': completed - start})
    stages['after_21_full_records'] = _memory(gpu)
    reuse = []
    for interval in range(3):
        start = time.perf_counter()
        for index in range(1000):
            result = program(*inputs[index % 2])
            synchronize(result)
        elapsed = time.perf_counter() - start
        _equal_records(materialize(result), reference[1])
        reuse.append({'calls': (interval + 1) * 1000, 'seconds': elapsed, 'memory': _memory(gpu)})
    assert program.experimental_get_tracing_count() == 1
    save(request, f'geometry-preparation-reuse-{kind}-{dimension}.json', {
        'role': 'explanatory_native_report_components_and_bounded_reuse', 'dimension': dimension, 'kind': kind,
        'gpu': gpu, 'jit_compile': True, 'inputs': inputs, 'expected_records': reference,
        'stages': stages, 'samples': samples, 'reuse': reuse, 'traces': 1,
        'nonclaims': ['No raw original/native timing ranking; no general leak freedom or executable-eviction evidence.']})
