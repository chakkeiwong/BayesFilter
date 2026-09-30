"""Complete-public-call cost and finite reuse diagnostics for final XLA owners.

NumPy is used only for independent rotation fixtures and diagnostic comparisons.
The timed paths are repository TensorFlow implementations, with frozen inputs.
"""

import gc
import hashlib
import importlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_geometry_control import clean as base_clean
from tests.test_filter_repair_geometry_control import save
from tests.test_filter_repair_resource_owners import memory
from tests.test_filter_repair_sqmc_endpoints import incoming_module
from tests.test_filter_repair_sqmc_inputs import frozen_inputs

D = tf.float64
CASES = ('angle3', 'angle23', 'sqmc_iid', 'sqmc_halton', 'trace_static', 'trace_dynamic')
ANGLE_BASELINES = {'pre_angle': 'bf022dd90', 'pre_guard': '0bfae62f2'}
ROOT = Path(__file__).resolve().parents[1]


def clean(value):
    """Normalize diagnostic scalar booleans at the JSON boundary only."""
    if isinstance(value, np.generic):
        return base_clean(value.item())
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return base_clean(value)


def arms(case):
    return ('pre_angle', 'pre_guard', 'current') if case.startswith('angle') else ('before', 'current')


def synchronized(value):
    return tf.nest.map_structure(lambda x: x.numpy() if tf.is_tensor(x) else x, value)


def compare(actual, expected, bound):
    assert tf.nest.assert_same_structure(actual, expected) is None
    for a, b in zip(tf.nest.flatten(actual), tf.nest.flatten(expected), strict=True):
        if a is None or isinstance(a, (str, bool)):
            assert a == b
        else:
            np.testing.assert_allclose(a, b, rtol=bound, atol=bound, equal_nan=True)


def rotation_fixture(dimension, variant):
    # Exact resolved top eigenspaces: one rotation for D3, three for D23.
    rank = 1 if dimension == 3 else 3
    angles = np.array([10., 25., 40.][:rank]) + variant*.25
    diagonal = np.arange(1., dimension+1)
    rotation = np.eye(dimension)
    for index, angle in enumerate(angles):
        a, b = index, dimension-rank+index
        cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        rotation[a, a], rotation[b, b] = cosine, cosine
        rotation[a, b], rotation[b, a] = -sine, sine
    first = np.diag(diagonal)
    second = (rotation*diagonal)@rotation.T
    return first, second, rank, sorted(angles.tolist())


def angle_owner(case, arm):
    dimension = int(case.removeprefix('angle'))
    if arm == 'current':
        from bayesfilter.inference import fixed_center_curvature as module
        source = {'checkpoint': 'current'}
    else:
        checkpoint = FrozenCheckpoint(ANGLE_BASELINES[arm], 'resource_angle_'+arm)
        module = checkpoint.load('bayesfilter.inference.fixed_center_curvature')
        source = {'checkpoint': ANGLE_BASELINES[arm], 'sources': checkpoint.hashes()}
    fixtures = [rotation_fixture(dimension, variant) for variant in (0, 1)]
    operands = [(tf.constant(a, D), tf.constant(b, D), rank) for a, b, rank, _ in fixtures]

    def invoke(variant):
        left, right, rank = operands[variant]
        return module.compare_precision_geometry(left, right, subspace_rank=rank)

    def owners():
        signature = (tf.TensorSpec([dimension, dimension], D),)*2 + (tf.TensorSpec([], D), tf.TensorSpec([], tf.int32))
        return [module._compiled_kernel(module._precision_geometry_kernel, signature)]

    def validate(record, variant):
        expected = fixtures[variant][3]
        np.testing.assert_allclose(record['principal_angles_degrees'], expected, rtol=1e-10, atol=1e-10)
        assert record['positive_subspace_rank'] == len(expected)
        assert record['left_nonpositive_count'] == record['right_nonpositive_count'] == 0
        assert math.isfinite(record['trace_normalized_operator'])
        assert (record['maximum_principal_angle_degrees'] <= 5.) == (max(expected) <= 5.)

    fixture_id = hashlib.sha256(json.dumps(clean(fixtures), sort_keys=True).encode()).hexdigest()
    return invoke, owners, validate, {'case': case, 'fixture_sha256': fixture_id,
        'dimension': dimension, 'ranks': [item[2] for item in fixtures], 'source': source,
        'bound': 1e-10, 'scope': 'Public compare_precision_geometry with two resolved analytical rotations'}, module._compiled_kernel.cache_info


def sqmc_owner(case, arm):
    from bayesfilter.highdim import sqmc_campaign_tf as current
    from bayesfilter.highdim.sqmc_full_lgssm_tf import FullLGSSMSpec
    from tests.highdim.test_sqmc_campaign_repairs import CONTROLS

    spec = FullLGSSMSpec('full_matrix', 3)
    n = 12
    trace = case.startswith('trace')
    dynamic = case == 'trace_dynamic'
    horizon = 4 if dynamic else 2
    route = 'iid_dual_cap' if case == 'sqmc_iid' else 'repaired_permutation'
    observations = tf.reshape(tf.linspace(tf.constant(-.2, D), .3, horizon*3), [horizon, 3])
    theta = spec.default_theta()
    sources = {}
    program = None
    if trace:
        # Identical shared preparation is completed before measuring the changed
        # trace-summary owner. Preparation has its own public-endpoint costs.
        inputs = current.random_inputs(route, 81102, n, 3, horizon)
        if arm == 'current':
            module = importlib.import_module('run_sqmc_expanded_repair')
        else:
            module, source_hash = incoming_module('docs/benchmarks/run_sqmc_expanded_repair.py')
            sources['docs/benchmarks/run_sqmc_expanded_repair.py'] = source_hash

        def invoke(variant):
            nonlocal program
            if program is None:
                program = module.trace_summary_kernel(spec, route, CONTROLS, n, horizon,
                                                      dynamic_epsilon=dynamic)
            args = (theta+variant*.001, *inputs, observations)
            if dynamic:
                args += (tf.constant(64. if variant else 32., tf.float32),)
            raw = program(*args)
            return {'raw': synchronized(raw), 'summary': module.summarize_trace(raw)}

        def owners():
            return [program]

        def validate(record, _variant):
            assert record['summary']['valid']
            assert np.all(np.isfinite(record['raw'][0])) and np.all(np.isfinite(record['raw'][1]))

        cache = lambda: {'owners': int(program is not None)}
        bound = 1e-12
    else:
        if arm == 'current':
            module = current
        else:
            previous_inputs, hashes = frozen_inputs()
            module, source_hash = incoming_module('bayesfilter/highdim/sqmc_campaign_tf.py',
                                                 {'random_inputs': previous_inputs})
            # Preserve the exact incoming model preparation in the before arm.
            model_module, model_hash = incoming_module('bayesfilter/highdim/sqmc_full_lgssm_tf.py')
            spec = model_module.FullLGSSMSpec('full_matrix', 3)
            sources = {**hashes, 'bayesfilter/highdim/sqmc_campaign_tf.py': source_hash,
                       'bayesfilter/highdim/sqmc_full_lgssm_tf.py': model_hash}

        def invoke(variant):
            # Include default-parameter preparation, live seeded input generation,
            # and all analytical directions in the actual public endpoint.
            parameters = spec.default_theta()+variant*.001
            diagnostics = {}
            raw = module.value_and_score(spec, route, CONTROLS, parameters, observations,
                81102+variant, n, diagnostics=diagnostics)
            return {'raw': synchronized(raw), 'diagnostics': diagnostics}

        def owners():
            settings = module.numerical_settings(CONTROLS)
            result = [module._kernel(spec, route, n, horizon, D.name, tuple(sorted(settings.items())), True)]
            if arm == 'current':
                result.append(module._input_kernel(route == 'iid_dual_cap', n, 3, horizon, D.name, False, True))
            return result

        def validate(record, _variant):
            assert record['raw'][2]
            assert np.all(np.isfinite(record['raw'][0])) and np.all(np.isfinite(record['raw'][1]))
            assert not record['diagnostics']['invalid_value_coordinates']
            assert not record['diagnostics']['invalid_score_coordinates']

        def cache():
            result = {'score': module._kernel.cache_info()._asdict()['currsize']}
            if arm == 'current':
                result['input'] = module._input_kernel.cache_info()._asdict()['currsize']
            return result
        bound = 1e-9
    fixture_id = hashlib.sha256(json.dumps({'theta': theta.numpy().tolist(),
        'observations': observations.numpy().tolist(), 'seeds': [81102, 81103],
        'controls': CONTROLS, 'n': n, 'route': route, 'dynamic': dynamic}, sort_keys=True).encode()).hexdigest()
    return invoke, owners, validate, {'case': case, 'fixture_sha256': fixture_id,
        'horizon': horizon, 'particles': n, 'dimension': 3, 'controls': CONTROLS,
        'source': {'checkpoint': 'current' if arm == 'current' else '023e10610', 'sources': sources},
        'bound': bound, 'scope': 'Prepared-input trace summary' if trace else 'Live public SQMC value/score endpoint'}, cache


def build(case, arm):
    return angle_owner(case, arm) if case.startswith('angle') else sqmc_owner(case, arm)


def owner_records(owners):
    records = []
    for owner in owners():
        definition = owner.get_concrete_function().graph.as_graph_def()
        operations = {node.op for nodes in (definition.node, *(fn.node_def for fn in definition.library.function))
                      for node in nodes}
        records.append({'trace_count': owner.experimental_get_tracing_count(),
            'host_callbacks': sorted(operations & {'PyFunc', 'PyFuncStateless', 'EagerPyFunc', 'XlaHostCompute'}),
            'graph_bytes': definition.ByteSize()})
    assert all(r['trace_count'] == 1 and not r['host_callbacks'] for r in records)
    return records


def environment():
    return {'affinity': sorted(os.sched_getaffinity(0)),
        'threads': {key: os.environ.get(key) for key in ('TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS')},
        'tf32': bool(tf.config.experimental.tensor_float_32_execution_enabled())}


@pytest.mark.parametrize('case,arm,pair', [(case, arm, pair) for case in CASES for arm in arms(case) for pair in range(3)])
def test_public_cost(case, arm, pair, request, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'docs/benchmarks'))
    tf.config.experimental.enable_tensor_float_32_execution(False)
    gpu = os.environ['CUDA_VISIBLE_DEVICES'] != '-1'
    assert ('GPU:0' if gpu else 'CPU:0') in tf.constant(0.).device
    invoke, owners, validate, metadata, cache = build(case, arm)
    if gpu:
        tf.config.experimental.reset_memory_stats('GPU:0')
    with GPUProcessMonitor(gpu) as monitor:
        before = memory()
        tick = time.perf_counter()
        initial = invoke(0)
        cold = time.perf_counter()-tick
        after_cold = memory()
        times = []
        exact_replay = True
        for _ in range(30):
            tick = time.perf_counter()
            result = invoke(0)
            times.append(time.perf_counter()-tick)
            compare(result, initial, 0.)
        after_warm = memory()
        changed = invoke(1)
        replay = invoke(0)
        compare(replay, initial, 0.)
        validate(initial, 0)
        validate(changed, 1)
        graphs = owner_records(owners)
    record = {'schema': 'filter_repair.final_resource_cost.v1', 'case': case, 'arm': arm, 'pair': pair,
        'device': 'GPU' if gpu else 'CPU', 'metadata': metadata, 'environment': environment(),
        'cold_seconds': cold, 'warm_seconds': times, 'warm_median_seconds': statistics.median(times),
        'memory': {'before': before, 'cold': after_cold, 'warm': after_warm},
        'initial': clean(initial), 'changed': clean(changed), 'exact_replay': exact_replay,
        'owners': graphs, 'cache': str(cache()), 'device_observation': monitor.payload(),
        'measurement_boundary': 'Primary samples precede comparison-owner or HLO compilation.',
        'nonclaims': ['No ranking against inaccurate original D23 angles or canonical LEDH admission.',
                     'Three fresh-process pairs only; no universal performance or capacity.']}
    save(request, 'final-resource-cost.json', record)
    assert not monitor.errors
    assert all(p['pid'] == os.getpid() for s in monitor.samples for p in s['processes'])


@pytest.mark.parametrize('case', CASES)
def test_public_reuse(case, request, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'docs/benchmarks'))
    tf.config.experimental.enable_tensor_float_32_execution(False)
    gpu = os.environ['CUDA_VISIBLE_DEVICES'] != '-1'
    tf.constant(0.).numpy()
    invoke, owners, validate, metadata, cache = build(case, 'current')
    if gpu:
        tf.config.experimental.reset_memory_stats('GPU:0')
    with GPUProcessMonitor(gpu) as monitor:
        initial, changed = invoke(0), invoke(1)
        validate(initial, 0)
        validate(changed, 1)
        expected = [initial, changed]
        first_cache = cache()
        # Compare cache sizes, not hit counters.
        cache_size = first_cache.currsize if hasattr(first_cache, 'currsize') else first_cache
        samples = {0: memory()}
        for index in range(1, 257):
            compare(invoke(index % 2), expected[index % 2], 0.)
            if index in (16, 64, 128, 192, 256):
                samples[index] = memory()
        end_cache = cache()
        end_size = end_cache.currsize if hasattr(end_cache, 'currsize') else end_cache
        assert end_size == cache_size
        late = samples[256]['VmRSS']-samples[128]['VmRSS']
        graphs = owner_records(owners)
        assert late <= 16*1024**2
        if gpu:
            assert samples[128]['allocator']['current'] == samples[256]['allocator']['current']
            assert samples[256]['allocator']['peak'] <= 2*1024**3
        after = memory()
    del invoke, owners
    gc.collect()
    record = {'schema': 'filter_repair.final_resource_reuse.v1', 'case': case,
        'device': 'GPU' if gpu else 'CPU', 'metadata': metadata, 'environment': environment(),
        'calls': 256, 'exact_replay': True, 'owners': graphs, 'cache_size': cache_size,
        'memory': samples, 'late_128_call_rss_growth_bytes': late, 'after': after,
        'device_observation': monitor.payload(),
        'nonclaims': ['Module LRU owners remain cached intentionally; no native eviction claim.',
                     'Finite fixed-configuration reuse; no arbitrary shape capacity.']}
    save(request, 'final-resource-reuse.json', clean(record))
    assert not monitor.errors
    assert all(p['pid'] == os.getpid() for s in monitor.samples for p in s['processes'])
