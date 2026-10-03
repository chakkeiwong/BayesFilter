"""Independent spectral fixtures and public-consumer refusal diagnostics."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import dense_validated_fit_tf as dense
from bayesfilter.inference import fixed_center_curvature as current
from bayesfilter.inference import fixed_center_stability_tf as stability
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_dense_validated_fit import padded, report_result
from tests.test_filter_repair_dz5_exact_fit import (
    test_exact_saved_fit_inputs as execute_saved_fit,
)
from tests.test_filter_repair_dz5_initializer_fit_localization import differences
from tests.test_filter_repair_fixed_fitting import _inputs, _thresholds
from tests.test_filter_repair_fixed_stability import _fits
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_principal_angle_precision import program, synchronize

RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
BASELINE = '0bfae62f2'
D = tf.float64
ERROR = 'principal subspace is not numerically resolved at the requested rank'


def test_spectral_resolution_controls(request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    checkpoint = FrozenCheckpoint(BASELINE, 'subspace_identifiability_before')
    baseline = checkpoint.load('bayesfilter.inference.fixed_center_curvature')
    rows = []
    for dimension in (3, 23):
        rotation, _ = np.linalg.qr(np.arange(1., dimension*dimension+1).reshape(dimension, dimension)
            + 3*np.eye(dimension))
        separated = np.arange(1., dimension+1)
        repeated = separated.copy()
        repeated[-2:] = dimension+1
        for jit in (False, True):
            candidate, original = program(current, dimension, jit=jit), program(baseline, dimension, jit=jit)
            for scale in (1e-8, 1., 1e8):
                for label, eigenvalues, basis, rank, resolved in (
                    ('isotropic_partial', np.ones(dimension), np.eye(dimension), 1, False),
                    ('isotropic_full', np.ones(dimension), np.eye(dimension), dimension, True),
                    ('repeated_cut', repeated, np.eye(dimension), 1, False),
                    ('rotated_repeated_cut', repeated, rotation, 1, False),
                    ('repeated_whole_cluster', repeated, rotation, 2, True),
                    ('separated', separated, rotation, 1, True),
                    ('separated_full', separated, rotation, dimension, True)):
                    matrix = scale*(basis*eigenvalues)@basis.T
                    operands = (tf.constant(matrix, D), tf.constant(matrix, D), tf.constant(rank, tf.int32))
                    new, old = synchronize(candidate(*operands)), synchronize(original(*operands))
                    assert int(new[2]) == rank
                    if resolved:
                        assert np.all(np.isfinite(new[3][:rank]))
                        for actual, expected in zip(new, old, strict=True):
                            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
                    else:
                        assert np.all(np.isnan(new[3][:rank]))
                        for index in (0, 1, 2, 4, 5, 6):
                            np.testing.assert_allclose(new[index], old[index], rtol=1e-10, atol=1e-10)
                    rows.append({'dimension': dimension, 'jit_compile': jit, 'scale': scale,
                        'case': label, 'rank': rank, 'resolved': resolved,
                        'nonangle_metrics_preserved': True})
            assert candidate.experimental_get_tracing_count() == original.experimental_get_tracing_count() == 1
    owner = program(current, 3)
    for gap, resolved in ((np.nextafter(2., np.inf)-2., False), (1e-8, True)):
        matrix = tf.linalg.diag(tf.constant([1., 2., 2.+gap], D))
        result = owner(matrix, matrix, tf.constant(1))
        assert bool(tf.math.is_finite(result[3][0])) == resolved
    save(request, 'subspace-identifiability-controls.json', {'baseline': BASELINE,
        'baseline_sources': checkpoint.hashes(), 'cases': rows,
        'roundoff_scale_gap_refused': True, 'resolved_small_gap_preserved': True,
        'nonclaims': ['A numerical refusal is not a proof of exact eigenvalue multiplicity.']})


def test_public_and_native_error_precedence(request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    identity = np.eye(3)
    with pytest.raises(ValueError, match=ERROR):
        current.compare_precision_geometry(identity, identity, subspace_rank=1)
    assert current.compare_precision_geometry(identity, identity, subspace_rank=3)['maximum_principal_angle_degrees'] == 0.
    thresholds = current.FixedCenterCurvatureThresholds(selection_holdout_relative_rmse_cap=.2,
        audit_relative_rmse_cap=.2, projection_relative_frobenius_cap=.2,
        principal_subspace_rank=1, principal_angle_degrees_cap=5.)
    with pytest.raises(ValueError, match=ERROR):
        current._family_stability(_fits([identity, identity]), thresholds)
    rows = []
    for jit in (False, True):
        owner = stability.stability_program(current._precision_geometry_kernel, 3, 2, jit_compile=jit)
        for label, left, right, rank, usable, error in (
            ('unresolved', identity, identity, 1, [[True, True]]*2, 5),
            ('full', identity, identity, 3, [[True, True]]*2, 0),
            ('left_invalid', np.array([[1., .1, 0.], [0., 1., 0.], [0., 0., 1.]]), identity, 1, [[True, True]]*2, 1),
            ('right_invalid', identity, np.full((3, 3), np.nan), 1, [[True, True]]*2, 2),
            ('rank_invalid', identity, identity, 4, [[True, True]]*2, 3),
            ('incomplete', identity, identity, 1, [[True, True], [True, False]], 0)):
            raw = owner(tf.constant([left, right], D), tf.constant(usable),
                tf.constant([2., 1., 1., 5.], D), tf.constant([True]*4), tf.constant(rank))
            assert int(raw['error']) == error
            assert bool(raw['passed']) == (label == 'full')
            rows.append({'case': label, 'jit_compile': jit, 'error': int(raw['error']),
                'passed': bool(raw['passed'])})
    save(request, 'subspace-identifiability-precedence.json', {'cases': rows,
        'public_explicit_refusal': True, 'full_rank_allowed': True})


def test_enclosing_isotropic_refusal(request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    inputs = list(_inputs(dimension=3))
    # Exact isotropic score equations with frozen, disjoint diagnostic clouds.
    for cloud_index, score_index in ((2, 3), (4, 5), (6, 7)):
        inputs[score_index] = inputs[1]-2.*inputs[cloud_index]
    operands = padded(inputs)
    thresholds = _thresholds(current)
    owner = dense.make_dense_validated_fit_program(3, 2, 9, 6, 6,
        thresholds=thresholds, factor_max=1)
    raw = owner(*operands)
    assert int(raw['validation']['error_code']) == 0
    assert int(raw['fit_error_code']) == 5
    assert not bool(raw['usable'])
    report = report_result(raw, operands, thresholds, 3)
    assert report['error']['message'] == ERROR
    assert owner.experimental_get_tracing_count() == 1
    save(request, 'subspace-identifiability-enclosing.json', {'baseline': BASELINE,
        'raw': clean(raw), 'public': report, 'jit_compile': True,
        'trace_count': owner.experimental_get_tracing_count(),
        'output_device': raw['fit_error_code'].device})


def test_complete_saved_fit_preserved(request):
    execute_saved_fit('after', request)
    directory = Path(request.config.getoption('xmlpath')).parent
    actual = json.loads((directory/'dz5-exact-saved-fit.json').read_text())
    gpu = os.environ['CUDA_VISIBLE_DEVICES'] != '-1'
    previous = RAW/f'run-{4920 if gpu else 4919:05d}/dz5-exact-saved-fit.json'
    before = json.loads(previous.read_text())
    assert actual['operand_sha256'] == before['operand_sha256']
    delta = differences(actual['result'], before['result'])
    raw_delta = differences(actual['raw'], before['raw'])
    save(request, 'subspace-identifiability-saved-fit.json', {'baseline': BASELINE,
        'source_run': previous.parent.name, 'complete_record_differences': delta,
        'raw_differences': raw_delta, 'selected_family': actual['result']['selected_family'],
        'fit_error_code': actual['fit_error_code']})
    assert not delta and not raw_delta
    assert actual['fit_error_code'] == 0 and actual['fit_numerically_usable']
