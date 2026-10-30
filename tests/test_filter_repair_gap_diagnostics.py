"""Diagnostic-only probes for unresolved enclosure and compilation costs.

The permutation trial is not an admitted runtime. Expected blocker observations
explain the next repair and do not certify repaired public behavior.
"""

import gc
import re
import resource
import time
import weakref
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.quadratic_geometry import _quadratic_fit_kernel
from bayesfilter.inference.quadratic_geometry_fit_report import (
    geometry_fit_payload,
    geometry_fit_result,
)
from bayesfilter.inference.quadratic_geometry_fit_tf import make_geometry_fit_program
from bayesfilter.ops.geometry_random_tf import STREAM_ID, _draw_kernel
from tests.test_filter_repair_geometry_control import save, stable_hlo
from tests.test_filter_repair_geometry_fit import fixture

D = tf.float64


def permutation_trial(capacity):
    """Fixed storage, same Fisher-Yates words; count remains a runtime operand."""
    @tf.function(input_signature=[tf.TensorSpec([], tf.int32), tf.TensorSpec([2], tf.int32)],
                 autograph=False, jit_compile=True)
    def draw(count, seed):
        def swap(index, values):
            bound = tf.cast(index + 1, tf.uint64)
            space = tf.constant(1 << 32, tf.uint64)
            limit = space - space % bound
            step_seed = tf.random.experimental.stateless_fold_in(seed, index, alg='philox')

            def word(attempt):
                return tf.cast(tf.random.stateless_uniform([], tf.random.experimental.stateless_fold_in(
                    step_seed, attempt, alg='philox'), minval=None, maxval=None,
                    dtype=tf.uint32, alg='philox'), tf.uint64)

            _, chosen = tf.while_loop(lambda attempt, value: value >= limit,
                lambda attempt, value: (attempt + 1, word(attempt)), (tf.constant(1), word(tf.constant(0))))
            other = tf.cast(chosen % bound, tf.int32)
            values = tf.tensor_scatter_nd_update(values, [[index], [other]],
                [tf.gather(values, other), tf.gather(values, index)])
            return index - 1, values

        return tf.while_loop(lambda index, _: index > 0, swap,
            (count - 1, tf.range(capacity)), maximum_iterations=capacity - 1)[1]

    return draw


def test_runtime_count_permutation_feasibility(request):
    capacity = 17
    program = permutation_trial(capacity)
    records, hlos = [], []
    for seed in ((173, 251), (739, 911)):
        for count in (0, 1, 2, 7, 16, 17):
            args = (tf.constant(count), tf.constant(seed))
            actual = program(*args)
            with tf.device('/CPU:0'):
                expected = _draw_kernel('permutation', (count,))(tf.constant(seed))
            records.append({'count': count, 'seed': seed, 'actual': actual, 'expected_active': expected})
            np.testing.assert_array_equal(actual[:count], expected)
            np.testing.assert_array_equal(actual[count:], np.arange(count, capacity))
            if count in (0, 17):
                hlos.append(program.experimental_get_compiler_ir(*args)(stage='hlo'))
    same_hlo = len({stable_hlo(text) for text in hlos}) == 1
    entry = hlos[0][hlos[0].rfind('\nENTRY '):]
    operands = len(re.findall(r'\bparameter\(\d+\)', entry))
    save(request, 'gap-permutation.json', {'role': 'diagnostic_feasibility_only',
        'stream_authority': STREAM_ID, 'cpu_reference': True, 'records': records,
        'trace_count': program.experimental_get_tracing_count(), 'same_hlo': same_hlo,
        'entry_operand_count': operands, 'hlo_bytes': len(hlos[0].encode())})
    assert program.experimental_get_tracing_count() == 1
    assert same_hlo and operands == 2


def test_unmasked_padding_changes_existing_fit(request):
    z = tf.constant([[.1, .2], [-.3, .2], [.4, -.5], [-.6, .7]], D)
    scores = -z * tf.constant([3., 2.], D)
    values = 5. + .5 * tf.reduce_sum(z * scores, axis=1)
    q = tf.constant([[1.], [0.]], D)
    records = {}
    for rows in (4, 8):
        inputs = (tf.pad(z, [[0, rows - 4], [0, 0]]), tf.pad(values, [[0, rows - 4]]),
                  tf.pad(scores, [[0, rows - 4], [0, 0]]))

        @tf.function(input_signature=[tf.TensorSpec([rows, 2], D), tf.TensorSpec([rows], D),
            tf.TensorSpec([rows, 2], D)], autograph=False, jit_compile=True)
        def fit(offsets, targets, gradients):
            return _quadratic_fit_kernel(offsets, targets, gradients, q,
                tf.zeros([2], D), tf.constant(1e-8, D), tf.constant(1e6, D))

        records[str(rows)] = fit(*inputs)
    save(request, 'gap-unmasked-padding.json', {'role': 'counterexample_not_repair', 'records': records,
        'interpretation': 'zero padding changes intercept and loss in the existing extent-based fit'})
    np.testing.assert_allclose(records['4']['precision'], records['8']['precision'], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(records['4']['intercept'], 5., atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(records['8']['intercept'], 2.5, atol=1e-10, rtol=1e-10)
    assert float(records['4']['loss']) < 1e-20
    assert float(records['8']['loss']) > 6.


def test_dynamic_batch_extent_is_not_reuse_evidence(request):
    @tf.function(input_signature=[tf.TensorSpec([17, 2], D), tf.TensorSpec([], tf.int32)],
                 autograph=False, jit_compile=True)
    def callback_extent(points, count):
        active = points[:count]
        return tf.shape(active)[0], tf.reduce_sum(active, axis=0)

    rows = tf.reshape(tf.cast(tf.range(34), D), [17, 2])
    observations = []
    for count in (3, 11):
        args = (rows, tf.constant(count))
        try:
            actual = callback_extent(*args)
            np.testing.assert_array_equal(actual[0], count)
            np.testing.assert_allclose(actual[1], tf.reduce_sum(rows[:count], axis=0), atol=1e-10, rtol=1e-10)
            hlo = callback_extent.experimental_get_compiler_ir(*args)(stage='hlo')
            entry = hlo[hlo.rfind('\nENTRY '):]
            observations.append({'count': count, 'compiled': True, 'result': actual,
                'entry_operand_count': len(re.findall(r'\bparameter\(\d+\)', entry)), 'hlo': hlo})
        except (tf.errors.InvalidArgumentError, tf.errors.UnimplementedError, ValueError) as exc:
            observations.append({'count': count, 'compiled': False,
                'error_type': type(exc).__name__, 'message': str(exc)[:12000]})
    compiled = all(row['compiled'] for row in observations)
    reusable = compiled and all(row['entry_operand_count'] == 2 for row in observations)
    if compiled:
        reusable = reusable and stable_hlo(observations[0]['hlo']) == stable_hlo(observations[1]['hlo'])
    save(request, 'gap-dynamic-batch.json', {'role': 'diagnostic_compile_boundary_not_runtime_qualification',
        'trace_count': callback_extent.experimental_get_tracing_count(),
        'demonstrated_runtime_count_reuse': reusable, 'observations': observations})
    assert callback_extent.experimental_get_tracing_count() == 1


def memory_snapshot(gpu):
    status = {line.split(':')[0]: int(line.split()[1]) * 1024
        for line in Path('/proc/self/status').read_text().splitlines()
        if line.startswith(('VmRSS:', 'VmHWM:'))}
    rollup = {line.split(':')[0]: int(line.split()[1]) * 1024
        for line in Path('/proc/self/smaps_rollup').read_text().splitlines()
        if line.startswith(('Rss:', 'Pss:', 'Private_Clean:', 'Private_Dirty:'))}
    return {'status': status, 'rollup': rollup,
        'ru_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'gpu': tf.config.experimental.get_memory_info('GPU:0') if gpu else None}


@pytest.mark.parametrize('jit', [False, True])
def test_fit_compiler_memory_release(jit, request):
    target, cfg, args, _ = fixture(3, 'nonquadratic', counter=False)
    prefix = {'center_log_prob': float(args[8]), 'center_score_norm': float(tf.linalg.norm(args[9])),
        'finite_sample_count': 32}
    gpu = bool(tf.config.list_logical_devices('GPU'))
    stages, timing = {}, {}
    stages['prepared'] = memory_snapshot(gpu)
    begin = time.perf_counter()
    program = make_geometry_fit_program(target, 3, 2, 24, 8, cfg, jit_compile=jit)
    timing['build'] = time.perf_counter() - begin
    stages['built'] = memory_snapshot(gpu)
    begin = time.perf_counter()
    concrete = program.get_concrete_function()
    timing['trace'] = time.perf_counter() - begin
    stages['traced'] = memory_snapshot(gpu)
    reference = weakref.ref(concrete.graph)
    begin = time.perf_counter()
    raw = program(*args)
    raw['status'].numpy()
    timing['first_execution'] = time.perf_counter() - begin
    stages['first_execution'] = memory_snapshot(gpu)
    first = geometry_fit_payload(geometry_fit_result(raw, cfg, args[0], args[1], prefix, holdout_rows=8))
    stages['reported'] = memory_snapshot(gpu)
    for _ in range(20):
        raw = program(*args)
        actual = geometry_fit_payload(geometry_fit_result(raw, cfg, args[0], args[1], prefix, holdout_rows=8))
        assert actual == first
    stages['warm_reuse'] = memory_snapshot(gpu)
    graph = concrete.graph.as_graph_def()
    metrics = {'nodes': len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function),
        'bytes': graph.ByteSize(), 'traces': program.experimental_get_tracing_count()}
    del raw, program, concrete, graph, actual
    gc.collect()
    stages['python_release'] = memory_snapshot(gpu)
    save(request, 'gap-fit-memory.json', {'role': 'diagnostic_compile_allocation_attribution',
        'jit_compile': jit, 'non_jit_role': None if jit else 'explicit_graph_reference',
        'input_records': args, 'result': first, 'gpu': gpu, 'timing_seconds': timing,
        'stages': stages, 'graph': metrics, 'python_graph_released': reference() is None,
        'limitation': 'Python graph release does not prove native executable or allocator release; small fixture only'})
