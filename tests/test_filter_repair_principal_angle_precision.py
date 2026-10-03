"""Independent principal-angle accuracy and descriptive XLA cost diagnostics.

NumPy is an independent eigensystem/SVD reference and post-run comparator only.
All timed kernels call the repository numerical authority on tensor operands.
"""

import gc
import hashlib
import json
import math
import os
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import fixed_center_curvature as current
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_geometry_control import clean, save

D = tf.float64
FIXTURE = Path(__file__).parent / 'fixtures/filter_repair_dz5_principal_angles_04578.json'
FIXTURE_SHA = 'd19248b80f8330dbf1f7ba9770880e1945a72616ac82f4c0c655785eba4f3b04'


def memory():
    status = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'VmHWM'):
            status[key] = int(value.split()[0]) * 1024
    return {'process_bytes': status, 'map_count': len(Path('/proc/self/maps').read_text().splitlines()),
        'allocator': tf.config.experimental.get_memory_info('GPU:0')
            if os.environ['CUDA_VISIBLE_DEVICES'] != '-1' else None}


def program(module, dimension, *, jit=True):
    @tf.function(input_signature=[tf.TensorSpec([dimension, dimension], D)] * 2
        + [tf.TensorSpec([], tf.int32)], jit_compile=jit, autograph=False)
    def evaluate(first, second, rank):
        return module._precision_geometry_kernel(first, second, tf.constant(1e-12, D), rank,
            jit_compile=jit)
    return evaluate


def synchronize(result):
    return tf.nest.map_structure(lambda value: value.numpy(), result)


@pytest.mark.parametrize('arm', ['before', 'after'])
def test_actual_d23_accuracy_and_costs(arm, request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA
    fixture = json.loads(FIXTURE.read_text())
    checkpoint = FrozenCheckpoint('2c80ecbcc', 'principal_angle_before')
    module = checkpoint.load('bayesfilter.inference.fixed_center_curvature') if arm == 'before' else current
    first, second = (tf.constant(fixture[key], D) for key in ('first_precision', 'second_precision'))
    rank = fixture['rank']
    operands = (first, second, tf.constant(rank, tf.int32))
    gpu = os.environ['CUDA_VISIBLE_DEVICES'] != '-1'
    before = memory()
    if gpu:
        tf.config.experimental.reset_memory_stats('GPU:0')
    tick = time.monotonic()
    kernel = program(module, first.shape[0])
    actual = synchronize(kernel(*operands))
    cold = time.monotonic() - tick
    after_cold = memory()
    times = []
    for _ in range(5):
        tick = time.monotonic()
        for _ in range(20):
            synchronize(kernel(*operands))
        times.append((time.monotonic() - tick) / 20)
    after_warm = memory()
    vectors_a = np.linalg.eigh(first.numpy())[1][:, -rank:]
    vectors_b = np.linalg.eigh(second.numpy())[1][:, -rank:]
    authority = np.linalg.svd(vectors_a.T @ vectors_b, compute_uv=False)
    cosine = np.cos(actual[3][:rank] * (math.pi / 180.))
    errors = np.abs(cosine - authority)
    gate = bool(np.all(errors <= 1e-14 + 1e-14 * np.abs(authority)))
    reference_angles = np.arccos(np.clip(authority, -1., 1.)) * (180. / math.pi)
    graph = synchronize(program(current, first.shape[0], jit=False)(*operands))
    graph_errors = np.abs(actual[3][:rank] - graph[3][:rank])
    graph_bounds = 1e-10 + 1e-10 * np.abs(graph[3][:rank])
    graph_definition = kernel.get_concrete_function().graph.as_graph_def()
    nodes = [*graph_definition.node,
        *(node for function in graph_definition.library.function for node in function.node_def)]
    callbacks = sorted({node.op for node in nodes if 'pyfunc' in node.op.lower() or 'hostcompute' in node.op.lower()})
    # Exercise both changed data and a changed runtime rank without retracing.
    changed = synchronize(kernel(first * tf.constant(1.001, D), second, tf.constant(1, tf.int32)))
    replay = synchronize(kernel(*operands))
    assert int(changed[2]) == 1
    assert kernel.experimental_get_tracing_count() == 1 and not callbacks
    for left, right in zip(actual, replay, strict=True):
        np.testing.assert_array_equal(left, right)
    hlo = kernel.experimental_get_compiler_ir(*operands)(stage='hlo')
    report = {'schema': 'filter_principal_angle_precision.v1', 'arm': arm,
        'device': 'GPU' if gpu else 'CPU', 'jit_compile': True, 'tf32': False,
        'cpu_reference_exception': not gpu, 'fixture_sha256': FIXTURE_SHA,
        'baseline_sources': checkpoint.hashes() if arm == 'before' else {},
        'accuracy_passed': gate, 'singular_error': errors.tolist(),
        'independent_singular_values': authority.tolist(), 'independent_angles': reference_angles.tolist(),
        'actual_angles': actual[3][:rank].tolist(), 'graph_angles': graph[3][:rank].tolist(),
        'full_record_angle_bounds': graph_bounds.tolist(), 'full_record_angle_errors': graph_errors.tolist(),
        'full_record_angles_equivalent': bool(np.all(graph_errors <= graph_bounds)),
        'cold_seconds': cold, 'warm_seconds_per_call': times,
        'warm_median_seconds': statistics.median(times),
        'memory_before': before, 'memory_after_cold': after_cold, 'memory_after_warm': after_warm,
        'memory_after_inspection': memory(), 'trace_count': kernel.experimental_get_tracing_count(),
        'host_callback_ops': callbacks, 'graph_bytes': graph_definition.ByteSize(),
        'hlo_bytes': len(hlo.encode()), 'hlo_sha256': hashlib.sha256(hlo.encode()).hexdigest(),
        'nonclaims': ['Defective baseline timing is diagnostic only; no performance ranking or full initializer admission.']}
    del kernel, graph_definition, nodes, hlo
    gc.collect()
    report['memory_after_python_release'] = memory()
    report['host_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    save(request, 'principal-angle-precision.json', report)
    if arm == 'after':
        assert gate, report['singular_error']
    else:
        assert not gate, 'The pinned defect must be reproduced on the actual overlap case'


def test_analytic_rotations_rank_and_gate(request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    kernel = program(current, 3)
    first = tf.linalg.diag(tf.constant([1., 2., 4.], D))
    records = []
    for angle in (0., .001, 4.9, 5.1, 40., 89.):
        radians = tf.constant(angle * math.pi / 180., D)
        c, s = tf.cos(radians), tf.sin(radians)
        zero = tf.constant(0., D)
        rotation = tf.stack((tf.constant([1., 0., 0.], D), tf.stack((zero, c, -s)), tf.stack((zero, s, c))))
        second = rotation @ first @ tf.transpose(rotation)
        result = synchronize(kernel(first, second, tf.constant(1, tf.int32)))
        observed = float(result[3][0])
        records.append({'angle': angle, 'observed': observed, 'passes_5_degree_gate': observed <= 5.})
        # Compare the well-conditioned singular value; also enforce the decision.
        np.testing.assert_allclose(np.cos(observed * math.pi / 180.), math.cos(angle * math.pi / 180.),
            atol=1e-14, rtol=1e-14)
        assert (observed <= 5.) == (angle <= 5.)
    for rank in (0, 1, 2, 3):
        result = synchronize(kernel(first, first, tf.constant(rank, tf.int32)))
        assert int(result[2]) == rank
        np.testing.assert_array_equal(result[3][:rank], np.zeros(rank))
    assert kernel.experimental_get_tracing_count() == 1
    save(request, 'principal-angle-analytic.json', {'records': clean(records), 'trace_count': 1})
