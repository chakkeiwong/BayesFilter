"""Independent geometry, rejection, callback and ownership diagnostics."""

import dataclasses
import gc
import json
import weakref
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import posterior_curvature_tf as runtime
from bayesfilter.inference.posterior_curvature_report import posterior_curvature_result
from tests.test_filter_repair_posterior_curvature import fixture, original, reference
from tests.test_filter_repair_quadratic_batches import _equal_records


class EqualCallable:
    """Equal but distinct two-output callbacks must not share a cached target."""

    def __init__(self, callback):
        self.callback = callback

    def __eq__(self, other):
        return isinstance(other, EqualCallable)

    def __call__(self, points):
        return self.callback(points)


def pure_fixture(dimension, *, design='uniform_box', replicates=2):
    # Stateless target for unlimited warm repetitions and independent geometry.
    from bayesfilter.inference.posterior_curvature_refinement import (
        PosteriorCurvatureRefinementConfig,
    )

    cfg = PosteriorCurvatureRefinementConfig(rows_per_partition=33, batch_size=8,
        replicate_count=replicates, fit_design=design, seed=-73)
    args = (tf.range(dimension, dtype=tf.float64) * .03,
        tf.eye(dimension, dtype=tf.float64) * .9 + tf.linalg.band_part(tf.ones([dimension, dimension], tf.float64) * .05, -1, 0),
        tf.constant(cfg.seed))

    def callback(theta):
        precision = tf.linalg.diag(tf.cast(tf.range(dimension) + 1, tf.float64)) + .05
        delta = theta - .2
        product = delta @ precision
        return -.5 * tf.reduce_sum(delta * product, axis=1), -product

    def eligibility(theta):
        return tf.ones([cfg.batch_size], tf.bool)

    return callback, eligibility, cfg, args


@pytest.mark.parametrize('design', ['uniform_box', 'uniform_ball'])
@pytest.mark.parametrize('dimension', [1, 3, 5])
def test_independent_gaussian_geometry(design, dimension):
    callback, eligible, cfg, args = pure_fixture(dimension, design=design, replicates=3)
    result = runtime.make_posterior_curvature_controller(callback, eligible, dimension, cfg)(*args)
    assert int(result['status']) == 1
    precision = np.diag(np.arange(dimension) + 1.) + .05
    covariance = np.linalg.inv(precision)
    np.testing.assert_allclose(result['covariance'], covariance, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(result['precision'], args[1].numpy().T @ precision @ args[1].numpy(), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(result['refined_factor'], np.linalg.cholesky(covariance), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('design', ['zero', 'rank_one', 'ill_conditioned'])
def test_deficient_design_preserves_full_rejection(monkeypatch, design, request):
    callback, eligibility, cfg, args = pure_fixture(3)
    if design == 'zero':
        offsets = tf.zeros([33, 3], tf.float64)
    elif design == 'rank_one':
        offsets = tf.ones([33, 3], tf.float64)
    else:
        offsets = tf.reshape(tf.sin(tf.range(99, dtype=tf.float64)), [33, 3]) * tf.constant([1., 1e-8, 1e-10], tf.float64)
    _, baseline = original()
    monkeypatch.setattr(baseline, '_draw_offsets', lambda *args: offsets)
    expected = baseline.refine_posterior_local_curvature(callback, args[0], args[1],
        batched_eligibility_fn=eligibility, config=baseline.PosteriorCurvatureRefinementConfig(**dataclasses.asdict(cfg))).payload()
    for fit in expected['diagnostics'].get('replicates', []):
        if fit['design_rank'] < 3:
            fit['design_condition'] = None
    monkeypatch.setattr(runtime, 'draw_offsets', lambda *args: offsets)
    records = {}
    for jit in (False, True):
        raw = runtime.make_posterior_curvature_controller(callback, eligibility, 3, cfg, jit_compile=jit)(*args)
        actual = posterior_curvature_result(raw, args[0], args[1], cfg).payload()
        records[str(jit)] = actual
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / f'posterior-design-{design}.json').open('x') as out:
        json.dump({'original': expected, 'candidate': records, 'offsets': offsets.numpy().tolist()}, out, indent=2, allow_nan=False)
        out.write('\n')
    for actual in records.values():
        _equal_records(actual, expected)
        assert actual['status'] == 'curvature_fit_rejected'
        assert actual['refined_factor'] is None


@pytest.mark.parametrize('fault,expected', [
    ('eligibility_dtype', TypeError), ('eligibility_shape', ValueError),
    ('values_dtype', TypeError), ('scores_dtype', TypeError),
    ('values_shape', ValueError), ('scores_shape', ValueError), ('exception', RuntimeError),
])
def test_malformed_callbacks_raise(fault, expected):
    base, eligible, cfg, args = pure_fixture(3)

    def eligibility(theta):
        answer = eligible(theta)
        if fault == 'eligibility_dtype':
            answer = tf.cast(answer, tf.int32)
        if fault == 'eligibility_shape':
            answer = answer[:, None]
        return answer

    def callback(theta):
        if fault == 'exception':
            raise RuntimeError('callback programming error')
        values, scores = base(theta)
        if fault == 'values_dtype':
            values = tf.cast(values, tf.float32)
        if fault == 'scores_dtype':
            scores = tf.cast(scores, tf.float32)
        if fault == 'values_shape':
            values = values[:, None]
        if fault == 'scores_shape':
            scores = scores[:, 0]
        return values, scores

    with pytest.raises(expected):
        reference(callback, eligibility, cfg, args)
    with pytest.raises(expected):
        runtime.make_posterior_curvature_controller(callback, eligibility, 3, cfg)(*args)


def test_replicate_instability_preserves_all_fit_records():
    _unused, eligible, cfg, args, counters, _positions = fixture(3, replicates=3, rows=17, batch=7)
    counter = counters[1]
    args = (tf.zeros(3, tf.float64), tf.eye(3, dtype=tf.float64), args[2])

    def callback(theta):
        call = counter.assign_add(1)
        replicate = ((call - 2) // 3) % 3
        scale = tf.where(replicate == 0, tf.constant(1., tf.float64), tf.constant(2., tf.float64))
        return -.5 * scale * tf.reduce_sum(theta**2, axis=1), -scale * theta

    expected, _ = reference(callback, eligible, cfg, args)
    original_calls = int(counter)
    counter.assign(0)
    raw = runtime.make_posterior_curvature_controller(callback, eligible, 3, cfg)(*args)
    actual = posterior_curvature_result(raw, args[0], args[1], cfg).payload()
    _equal_records(actual, expected)
    assert actual['status'] == 'replicate_instability'
    assert int(counter) == original_calls == 19
    assert len(actual['diagnostics']['replicates']) == 3


def scaled_target(base, strength):
    def callback(points):
        values, scores = base(points)
        return strength * values, strength * scores
    return callback


def test_cache_identity_changed_target_and_resource_release(request):
    runtime.clear_posterior_curvature_controller_cache()
    base, eligibility, cfg, args = pure_fixture(3)
    strength = tf.Variable(1., dtype=tf.float64)
    target = EqualCallable(scaled_target(base, strength))
    program = runtime.posterior_curvature_controller(target, eligibility, 3, cfg)
    assert runtime.posterior_curvature_controller(target, eligibility, 3,
        dataclasses.replace(cfg, seed=17, lineage={'changed': 'reporting only'})) is program
    records = []
    for value in (1., 2.):
        strength.assign(value)
        expected, _ = reference(target, eligibility, cfg, args)
        raw = program(*args)
        actual = posterior_curvature_result(raw, args[0], args[1], cfg).payload()
        _equal_records(actual, expected)
        records.append(actual)
    assert records[0]['refined_covariance'] != records[1]['refined_covariance']
    assert program.experimental_get_tracing_count() == 1
    concrete = program.get_concrete_function()
    watched = {name: weakref.ref(value) for name, value in
        (('graph', concrete.graph), ('target', target), ('strength', strength))}
    other = EqualCallable(base)
    assert other == target
    assert runtime.posterior_curvature_controller(other, eligibility, 3, cfg) is not program
    runtime.clear_posterior_curvature_controller_cache()
    del concrete, program, target, strength, raw
    gc.collect()
    released = {name: ref() is None for name, ref in watched.items()}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'posterior-resource-ownership.json').open('x') as out:
        json.dump({'released': released, 'changed_target_records': records, 'cache_capacity': 1,
            'nonclaim': 'Python collection does not prove native executable eviction.'}, out, indent=2, allow_nan=False)
        out.write('\n')
    assert all(released.values())
