"""Bounded reuse and caller replacement of the accurate SVD composition."""

import gc
import os
import weakref

import numpy as np
import tensorflow as tf

from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_remaining_svd_cost import (
    D,
    PrecisionSource,
    build_endpoint,
    build_numerical,
    checks,
    fixture,
    source_functions,
)
from tests.test_filter_repair_resource_owners import memory


def sync(program, offsets):
    return tf.nest.map_structure(lambda x: np.asarray(x.numpy()), program(offsets))


def same(a, b):
    return all(np.array_equal(x, y, equal_nan=True) for x, y in
               zip(tf.nest.flatten(a), tf.nest.flatten(b), strict=True))


def test_composition_reuse_and_replacement(request):
    functions, _ = source_functions(False)
    gpu = os.environ['CUDA_VISIBLE_DEVICES'] != '-1'
    tf.constant(0.).numpy()
    if gpu:
        tf.config.experimental.reset_memory_stats('GPU:0')
    primitives = {}
    lifetime = []
    with GPUProcessMonitor(gpu) as monitor:
        before = memory()
        for dimension in (3, 5):
            left, right, separated, _, precision = fixture(dimension)
            offsets = (left*separated)@right.T
            arguments = [tf.constant(offsets*scale, D) for scale in (1., 1.1)]
            source = PrecisionSource(precision)
            primitive = build_numerical(functions, dimension, True)
            primitives[dimension] = primitive
            program = build_endpoint(primitive, source.read, dimension, True)
            original = sync(program, arguments[0])
            source.resource.assign(precision*1.05)
            changed = sync(program, arguments[1])
            assert all(checks(original, offsets, precision).values())
            assert all(checks(changed, offsets*1.1, precision*1.05).values())
            expected = [original, changed]
            snapshots = {0: memory()}
            exact = True
            for index in range(1, 257):
                variant = index % 2
                source.resource.assign(precision*(1.05 if variant else 1.))
                exact &= same(sync(program, arguments[variant]), expected[variant])
                if index in (16, 64, 128, 192, 256):
                    snapshots[index] = memory()
            trace_count = program.experimental_get_tracing_count()
            references = {'program': weakref.ref(program), 'resource_owner': weakref.ref(source),
                          'resource': weakref.ref(source.resource)}
            del program, source
            gc.collect()
            released = {key: value() is None for key, value in references.items()}
            late = snapshots[256]['VmRSS']-snapshots[128]['VmRSS']
            lifetime.append({'dimension': dimension, 'calls': 256, 'exact_replay': exact,
                'trace_count': trace_count, 'memory': snapshots, 'late_128_call_rss_growth_bytes': late,
                'released': released, 'original': original, 'changed': changed,
                'offsets': offsets, 'precision': precision})
            assert exact and trace_count == 1 and all(released.values())
            assert late <= 16*1024**2
            if gpu:
                assert snapshots[128]['allocator']['current'] == snapshots[256]['allocator']['current']
        capacity_before = memory()
        capacities = []
        for index in range(12):
            dimension = 3 if index % 2 == 0 else 5
            left, right, separated, _, precision = fixture(dimension)
            offsets = (left*separated)@right.T
            source = PrecisionSource(precision*(1.+index*.001))
            program = build_endpoint(primitives[dimension], source.read, dimension, True)
            result = sync(program, tf.constant(offsets, D))
            numerical_checks = checks(result, offsets, precision*(1.+index*.001))
            snapshot = memory()
            ref = weakref.ref(program)
            source_ref = weakref.ref(source)
            del program, source
            gc.collect()
            capacities.append({'index': index, 'dimension': dimension, 'checks': numerical_checks,
                'memory': snapshot, 'released': ref() is None and source_ref() is None})
            assert all(numerical_checks.values()) and capacities[-1]['released']
            assert snapshot['VmRSS']-capacity_before['VmRSS'] <= 2*1024**3
            if gpu:
                assert snapshot['allocator']['peak'] <= 2*1024**3
        primitive_records = {dimension: {'trace_count': program.experimental_get_tracing_count(),
            'captures': len(program.get_concrete_function().captured_inputs)}
            for dimension, program in primitives.items()}
        after = memory()
    record = {'schema': 'filter_repair.remaining_svd_lifetime.v1', 'device': 'GPU' if gpu else 'CPU',
        'lifetime': lifetime, 'replacement_count': 12, 'capacities': capacities,
        'primitives': primitive_records, 'memory': {'before': before, 'capacity_before': capacity_before,
            'after': after}, 'device_observation': monitor.payload(),
        'nonclaims': ['Finite D3/D5 owner replacement is not universal capacity.',
                     'Reusable COD primitive graphs may remain in TensorFlow native/custom-gradient registries.',
                     'Process exit contains residency; Python collection does not prove native eviction.']}
    save(request, 'remaining-svd-resource-lifetime.json', clean(record))
    assert all(row == {'trace_count': 1, 'captures': 0} for row in primitive_records.values())
    assert not monitor.errors
    assert all(process['pid'] == os.getpid() for sample in monitor.samples for process in sample['processes'])
