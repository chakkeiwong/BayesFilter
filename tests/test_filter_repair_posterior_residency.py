"""Diagnostic attribution of first-XLA startup and complete posterior reuse."""

import dataclasses
import gc
import json
import re
import time
import weakref
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import posterior_curvature_refinement as public
from bayesfilter.inference import posterior_curvature_tf as runtime
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_posterior_curvature import reference
from tests.test_filter_repair_posterior_curvature_extras import pure_fixture
from tests.test_filter_repair_quadratic_batches import _equal_records


def mapping_snapshot():
    groups, current = {}, None
    with Path('/proc/self/smaps').open() as lines:
        for line in lines:
            if re.match(r'^[0-9a-f]+-[0-9a-f]+ ', line):
                fields = line.split(maxsplit=5)
                path = fields[5] if len(fields) == 6 else 'anonymous'
                category = ('anonymous_executable' if 'x' in fields[1] else 'anonymous_other') if path == 'anonymous' else path
                current = groups.setdefault(category, {'mappings': 0, 'Rss': 0, 'Pss': 0, 'Private_Clean': 0, 'Private_Dirty': 0})
                current['mappings'] += 1
            elif current is not None and line.split(':')[0] in ('Rss', 'Pss', 'Private_Clean', 'Private_Dirty'):
                current[line.split(':')[0]] += int(line.split()[1]) * 1024
    return groups


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('primed', [False, True])
def test_posterior_startup_and_reuse_residency(primed, dimension, request):
    callback, eligibility, cfg, args = pure_fixture(dimension)
    gpu = bool(tf.config.list_logical_devices('GPU'))
    stages, mappings, seconds = {}, {}, {}

    def snapshot(name):
        stages[name] = memory_snapshot(gpu)
        mappings[name] = mapping_snapshot()

    snapshot('prepared')
    if primed:
        @tf.function(input_signature=[tf.TensorSpec([16], tf.float64)], jit_compile=True, autograph=False)
        def compiler_control(value):
            return tf.math.sin(value) + value

        begin = time.perf_counter()
        compiler_control(tf.ones([16], tf.float64)).numpy()
        seconds['minimal_xla'] = time.perf_counter() - begin
        snapshot('minimal_xla')
        del compiler_control
        gc.collect()
        snapshot('minimal_xla_python_release')
    program = runtime.posterior_curvature_controller(callback, eligibility, dimension, cfg)
    begin = time.perf_counter()
    concrete = program.get_concrete_function()
    seconds['trace'] = time.perf_counter() - begin
    graph_ref = weakref.ref(concrete.graph)
    snapshot('traced')

    def execute(values, options):
        return public.refine_posterior_local_curvature(callback, values[0], values[1],
            batched_eligibility_fn=eligibility, config=options).payload()

    begin = time.perf_counter()
    first = execute(args, cfg)
    seconds['cold_public'] = time.perf_counter() - begin
    snapshot('cold')
    changed = (args[0] + .07, args[1] * 1.1, args[2] + 4)
    changed_cfg = dataclasses.replace(cfg, seed=int(changed[2]))
    second = execute(changed, changed_cfg)
    snapshot('changed')
    begin = time.perf_counter()
    for index in range(3000):
        values, options, expected = (args, cfg, first) if index % 2 == 0 else (changed, changed_cfg, second)
        actual = execute(values, options)
        assert actual == expected
        if (index + 1) % 500 == 0:
            snapshot(f'reuse_{index + 1}')
    seconds['reuse_3000'] = time.perf_counter() - begin
    assert program.experimental_get_tracing_count() == 1
    runtime.clear_posterior_curvature_controller_cache()
    del program, concrete
    gc.collect()
    snapshot('python_release')
    # Independent references follow the entire observed resource interval.
    original, hashes = reference(callback, eligibility, cfg, args)
    original_changed, _ = reference(callback, eligibility, cfg, changed)
    report = {'schema': 'filter_posterior_public_residency.v1', 'dimension': dimension,
        'minimal_xla_prewarm': primed, 'gpu': gpu, 'seconds': seconds, 'stages': stages,
        'mappings': mappings, 'python_graph_released': graph_ref() is None,
        'result': first, 'changed_result': second, 'original': original, 'original_changed': original_changed,
        'original_source_sha256': hashes, 'reuse_calls': 3000,
        'role': 'allocation_attribution_not_performance_ranking',
        'nonclaims': ['Single process per condition; not a paired cost qualification.',
            'Memory snapshots and Python graph collection do not certify native executable eviction or leak freedom.',
            'Minimal-XLA startup is a diagnostic control, not a production warmup requirement.']}
    path = Path(request.config.getoption('xmlpath')).parent / 'posterior-public-residency.json'
    with path.open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')
    _equal_records(first, original)
    _equal_records(second, original_changed)
