"""Independent rank-policy boundary checks and complete dense consumer parity."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import batched_quadratic_center as quadratic
from bayesfilter.inference import posterior_curvature_refinement as posterior
from bayesfilter.inference import posterior_curvature_tf as posterior_runtime
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_fixed_stability import _compare
from tests.test_filter_repair_quadratic_numerics import (
    equal_dense_fields,
    original,
    program,
    serializable,
)
from tests.test_filter_repair_quadratic_rounds import compare_public_records

D = tf.float64


def _design(dimension, case):
    diagonal = np.linspace(1., 2., dimension)
    if case == 'zero':
        diagonal[:] = 0.
    elif case in ('below', 'above'):
        # Largest singular value is one; exercise both sides of the fixed
        # 1e-12 rank threshold without SVD rounding at the boundary itself.
        diagonal[:] = 1.
        diagonal[-1] = .5e-12 if case == 'below' else 2e-12
    design = np.pad(np.diag(diagonal), ((0, 32 - dimension), (0, 0)))
    if case == 'duplicate':
        design[:, -1] = design[:, 0]
    elif case == 'rotated':
        frame = np.linalg.qr(np.random.default_rng(20260921 + dimension).normal(size=(dimension, dimension)))[0]
        design = design @ frame
    return tf.constant(design, D)


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('case', ['zero', 'duplicate', 'below', 'above', 'rotated'])
def test_rank_policy_condition_and_all_other_fields(dimension, case, request):
    offsets = _design(dimension, case)
    center = tf.zeros([dimension], D)
    precision = tf.linalg.diag(tf.linspace(tf.constant(.8, D), tf.constant(2., D), dimension))
    scores = -offsets @ precision
    arguments = (center, offsets, scores, offsets, scores)
    expected = program('dense', dimension, source='original')(*arguments)
    results = {mode: program('dense', dimension, jit=mode == 'xla')(*arguments) for mode in ('graph', 'xla')}
    rank = 0 if case == 'zero' else dimension - 1 if case in ('duplicate', 'below') else dimension
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / f'dense-condition-{case}-{dimension}.json').open('x') as handle:
        json.dump({'original_raw_ratio': serializable(expected),
            **{mode: serializable(result) for mode, result in results.items()},
            'expected_rank': rank, 'source_sha256': original()[0].hashes()}, handle, indent=2)
        handle.write('\n')
    assert int(expected['design_rank']) == rank
    for result in results.values():
        assert int(result['design_rank']) == rank
        equal_dense_fields(result, expected, dimension)
        if rank == dimension:
            np.testing.assert_allclose(result['design_condition'],
                np.linalg.cond(offsets.numpy()), atol=1e-10, rtol=1e-10)
        else:
            assert np.isposinf(float(result['design_condition']))


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('case', ['zero', 'duplicate', 'below', 'above', 'rotated'])
def test_posterior_consumer_complete_decisions(dimension, case, monkeypatch, request):
    checkpoint = FrozenCheckpoint('3582b4ac', 'dense_posterior_' + case)
    before = checkpoint.load(posterior.__name__)
    offsets = _design(dimension, case)
    matrix = tf.linalg.diag(tf.linspace(tf.constant(.8, D), tf.constant(2., D), dimension))

    def callback(points):
        return -.5 * tf.reduce_sum(points * (points @ matrix), axis=1), -points @ matrix

    def eligible(points):
        return tf.ones([tf.shape(points)[0]], tf.bool)

    records = {}
    for name, module in [('original', before), ('current', posterior)]:
        with monkeypatch.context() as context:
            # Frozen independent clouds isolate the numerical and consumer
            # contract; RNG equivalence is covered by separate initializer tests.
            if name == 'original':
                context.setattr(module, '_draw_offsets', lambda *_: offsets)
            else:
                posterior_runtime.clear_posterior_curvature_controller_cache()
                context.setattr(posterior_runtime, 'draw_offsets', lambda *_: offsets)
            cfg = module.PosteriorCurvatureRefinementConfig(rows_per_partition=32,
                batch_size=8, max_physical_rows=1000)
            records[name] = module.refine_posterior_local_curvature(callback,
                tf.zeros([dimension], D), tf.eye(dimension, dtype=D),
                batched_eligibility_fn=eligible, config=cfg).payload()
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / f'dense-posterior-{case}-{dimension}.json').open('x') as handle:
        json.dump({'records': records, 'source_sha256': checkpoint.hashes()}, handle, indent=2)
        handle.write('\n')
    expected = copy.deepcopy(records['original'])
    for row in expected['diagnostics']['replicates']:
        if row['design_rank'] < dimension:
            # Strict JSON maps the newly infinite diagnostic to null.
            row['design_condition'] = None
    _compare(records['current'], expected)
    assert records['current']['accepted'] is (case == 'rotated')


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('case', ['centered', 'move', 'nonquadratic', 'indefinite'])
def test_uniform_quadratic_complete_original_records(dimension, case, request):
    before = original()[1]['trust']
    diagonal = tf.linspace(tf.constant(.8, D), tf.constant(2., D), dimension)
    if case == 'indefinite':
        diagonal = tf.concat([tf.constant([-1.], D), diagonal[1:]], 0)
    matrix = tf.linalg.diag(diagonal)

    def callback(points):
        values = -.5 * tf.reduce_sum(points * (points @ matrix), axis=1)
        scores = -points @ matrix
        if case == 'nonquadratic':
            values -= .01 * tf.reduce_sum(points ** 4, axis=1)
            scores -= .04 * points ** 3
        return values, scores, tf.ones([4], tf.bool)

    center = tf.zeros([dimension], D) if case == 'centered' else tf.linspace(tf.constant(-.3, D), tf.constant(.4, D), dimension)
    records = {}
    for name, module in [('original', before), ('current', quadratic)]:
        cfg = module.BatchedQuadraticCenterConfig(max_fit_rounds=4, centeredness_cap=1e-8,
            jit_compile_trust=False)
        records[name] = module.refine_batched_quadratic_center(callback, center,
            tf.ones([dimension], D), config=cfg).payload()
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / f'dense-uniform-{case}-{dimension}.json').open('x') as handle:
        json.dump({'records': records, 'source_sha256': original()[0].hashes()}, handle, indent=2)
        handle.write('\n')
    compare_public_records(copy.deepcopy(records['current']), copy.deepcopy(records['original']), jit=False)
