"""Public XLA call-chain qualification against the independent frozen controller."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import posterior_curvature_refinement as public
from bayesfilter.inference import posterior_curvature_tf as runtime
from tests.test_filter_repair_posterior_curvature import (
    CASES,
    fixture,
    reference,
    stable_hlo,
)
from tests.test_filter_repair_posterior_curvature_extras import (
    assert_ill_conditioned_rejection,
    assert_no_post_rejection_execution,
    original,
    pure_fixture,
)
from tests.test_filter_repair_quadratic_batches import _equal_records


def observe_controller(monkeypatch):
    """Observe the function actually invoked by the public entry point."""
    invocations = []
    factory = public.posterior_curvature_controller

    def build(*args, **kwargs):
        program = factory(*args, **kwargs)

        def invoke(*operands):
            record = program(*operands)
            invocations.append((program, record))
            return record

        return invoke

    monkeypatch.setattr(public, 'posterior_curvature_controller', build)
    return invocations


def assert_enclosed(program):
    concrete = program.get_concrete_function()
    assert concrete.function_def.attr['_XlaMustCompile'].b
    assert program.experimental_get_tracing_count() == 1
    graph = concrete.graph.as_graph_def()
    nodes = list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
    assert not {'PyFunc', 'EagerPyFunc', 'PyFuncStateless'} & {node.op for node in nodes}
    assert {'While', 'StatelessWhile'} & {node.op for node in nodes}


@pytest.mark.parametrize('dimension', [1, 3, 5])
@pytest.mark.parametrize('case', CASES)
def test_public_original_records(dimension, case, monkeypatch, request):
    callback, eligibility, cfg, args, counters, positions = fixture(dimension, case)
    expected, hashes = reference(callback, eligibility, cfg, args)
    counts = [int(counter) for counter in counters]
    original_positions = positions.numpy()[:counts[1]]
    for counter in counters:
        counter.assign(0)
    positions.assign(tf.zeros_like(positions))
    invoked = observe_controller(monkeypatch)
    result = public.refine_posterior_local_curvature(callback, args[0], args[1],
        batched_eligibility_fn=eligibility, config=cfg)
    actual = result.payload()
    assert len(invoked) == 1
    program, _record = invoked[0]
    assert_enclosed(program)
    actual_counts = [int(counter) for counter in counters]
    actual_positions = positions.numpy()[:actual_counts[1]]
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / f'posterior-public-{case}-{dimension}.json').open('x') as out:
        json.dump({'baseline': '3582b4ac', 'original_source_sha256': hashes,
            'original': expected, 'public': actual, 'original_counts': counts,
            'public_counts': actual_counts, 'original_positions': original_positions.tolist(),
            'public_positions': actual_positions.tolist(), 'jit_compile': True,
            'trace_count': program.experimental_get_tracing_count()}, out, indent=2, allow_nan=False)
        out.write('\n')
    assert actual_counts == counts
    np.testing.assert_allclose(actual_positions, original_positions, atol=1e-10, rtol=1e-10)
    _equal_records(actual, expected)


def test_public_reuses_changed_geometry_seed_and_lineage(monkeypatch, request):
    callback, eligibility, cfg, args, counters, _ = fixture(3, design='uniform_ball', replicates=3, rows=17, batch=7)
    invoked = observe_controller(monkeypatch)
    records, hlo = [], []
    changed = (args[0] + .07, args[1] * 1.1, args[2] + 4)
    for index, values in enumerate((args, changed)):
        options = dataclasses.replace(cfg, seed=int(values[2]), lineage={'public_call': index})
        for counter in counters:
            counter.assign(0)
        expected, _ = reference(callback, eligibility, options, values)
        counts = [int(counter) for counter in counters]
        for counter in counters:
            counter.assign(0)
        actual = public.refine_posterior_local_curvature(callback, values[0], values[1],
            batched_eligibility_fn=eligibility, config=options).payload()
        _equal_records(actual, expected)
        assert [int(counter) for counter in counters] == counts
        records.append(actual)
        hlo.append(invoked[-1][0].experimental_get_compiler_ir(*values)(stage='hlo'))
    assert len(invoked) == 2 and invoked[0][0] is invoked[1][0]
    assert_enclosed(invoked[0][0])
    assert stable_hlo(hlo[0]) == stable_hlo(hlo[1])
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory / 'posterior-public-first.hlo').write_text(hlo[0])
    (directory / 'posterior-public-changed.hlo').write_text(hlo[1])
    with (directory / 'posterior-public-reuse.json').open('x') as out:
        json.dump({'records': records, 'same_program': True, 'trace_count': 1}, out, indent=2, allow_nan=False)
        out.write('\n')


def test_public_rejected_ill_conditioned_geometry_is_unusable(monkeypatch, request):
    callback, eligibility, cfg, args = pure_fixture(3)
    offsets = tf.reshape(tf.sin(tf.range(99, dtype=tf.float64)), [33, 3]) * tf.constant([1., 1e-8, 1e-10], tf.float64)
    checkpoint, baseline = original()
    monkeypatch.setattr(baseline, '_draw_offsets', lambda *args: offsets)
    expected = baseline.refine_posterior_local_curvature(callback, args[0], args[1],
        batched_eligibility_fn=eligibility,
        config=baseline.PosteriorCurvatureRefinementConfig(**dataclasses.asdict(cfg))).payload()
    expected['diagnostics']['replicates'][0]['design_condition'] = None
    monkeypatch.setattr(runtime, 'draw_offsets', lambda *args: offsets)
    runtime.clear_posterior_curvature_controller_cache()
    invoked = observe_controller(monkeypatch)
    actual = public.refine_posterior_local_curvature(callback, args[0], args[1],
        batched_eligibility_fn=eligibility, config=cfg).payload()
    assert len(invoked) == 1
    assert_enclosed(invoked[0][0])
    assert_no_post_rejection_execution(invoked[0][1])
    assert_ill_conditioned_rejection(actual, expected)
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'posterior-public-rejection.json').open('x') as out:
        json.dump({'original': expected, 'public': actual, 'original_source_sha256': checkpoint.hashes(),
            'comparison': 'rejection_and_no_use; discarded_precision_is_diagnostic_only'}, out, indent=2, allow_nan=False)
        out.write('\n')


def test_public_unsupported_host_callback_never_replays_eagerly():
    host_calls = []
    _, eligibility, cfg, args = pure_fixture(3)

    def host(points):
        host_calls.append(True)
        return tf.reduce_sum(points, axis=1)

    def callback(points):
        values = tf.py_function(host, [points], tf.float64)
        values.set_shape([cfg.batch_size])
        return values, -points

    with pytest.raises((tf.errors.InvalidArgumentError, tf.errors.UnimplementedError)):
        public.refine_posterior_local_curvature(callback, args[0], args[1],
            batched_eligibility_fn=eligibility, config=cfg)
    assert host_calls == []
