"""Fresh test-scope calibration and untouched finite-program qualification.

These records cannot issue canonical tuning/admission identity. The failed
seed9292027 fixture never supplies operands to calibration or selection.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from tests.test_filter_repair_nonlinear_directions import _endpoint, _factory, _operands
from tests.test_filter_repair_score_directions import (
    DTYPE,
    _compare,
    _graph,
    _reference,
    _save,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
PROVIDERS = ('ledh', 'sgqf', 'kdm_covariance')
COUNTS = (8, 16, 32, 64)


def _partition(name):
    path = ROOT/f'tests/fixtures/filter_repair_nonlinear_{name}_20260929.json'
    data = json.loads(path.read_text())
    assert data['partition'] == name
    return data, hashlib.sha256(path.read_bytes()).hexdigest()


def _scope_fixture(source, count):
    fixture = copy.deepcopy(source)
    fixture['controls'].update(reset_sinkhorn_steps=count, reset_balance_steps=count)
    return fixture


def _latest(group):
    for path in sorted(RAW.glob('run-*/run.json'), reverse=True):
        number = int(path.parent.name[4:])
        if number <= 4805:
            break
        run = json.loads(path.read_text())
        if run['key'][1] == group:
            return number, run
    raise AssertionError(group)


def _check_calibration_fixture(fixture, provider):
    owner = _factory(provider+'_diagnostics', fixture=fixture)
    value, score, diagnostics = owner(tf.constant(fixture['theta'], DTYPE),
        tf.eye(6, dtype=DTYPE)[0], *_operands(provider, fixture=fixture))
    finite = bool(tf.math.is_finite(value) & tf.math.is_finite(score))
    finite &= all(bool(tf.reduce_all(tf.math.is_finite(x))) for x in diagnostics.values() if x.dtype.is_floating)
    positive = all(bool(tf.reduce_all(x > 0)) for name, x in diagnostics.items() if name.endswith('_minimum_eigenvalue_per_time'))
    correction = bool(tf.reduce_all(diagnostics['higher_moment_valid']))
    return {'case': fixture['case'], 'passed': finite and positive and correction,
        'value': float(value) if bool(tf.math.is_finite(value)) else str(float(value)),
        'score': float(score) if bool(tf.math.is_finite(score)) else str(float(score)),
        'finite': finite, 'positive_covariances': positive, 'correction_valid': correction,
        'maximum_column_weight_residual': float(tf.reduce_max(diagnostics['transport_column_weight_error_per_time'])),
        'trace_count': owner.experimental_get_tracing_count()}


@pytest.mark.parametrize('provider,count', [(p, n) for p in PROVIDERS for n in COUNTS])
def test_scope_calibration_candidate(provider, count, request):
    calibration, calibration_hash = _partition('calibration')
    # A later candidate is allowed only after the preceding one failed on the
    # calibration partition itself. Never react to failed validation by retuning.
    if count != COUNTS[0]:
        previous = COUNTS[COUNTS.index(count)-1]
        number, run = _latest(f'nonlinear_scope_calibrate_{provider}_{previous}_cpu')
        assert run['state'] == 'passed'
        before = json.loads((RAW/f'run-{number:05d}'/'nonlinear-scope-calibration.json').read_text())
        assert not before['calibration_passed'] and before['validation'] is None
    cases = [_check_calibration_fixture(_scope_fixture(row, count), provider)
             for row in calibration['providers'][provider]]
    passed = all(row['passed'] for row in cases)
    result = {'schema': 'filter_repair_test_scope_calibration.v1', 'provider': provider, 'count': count,
        'calibration_sha256': calibration_hash, 'calibration': cases, 'calibration_passed': passed,
        'validation': None, 'validation_sha256': None, 'validated_nomination': False,
        'scope': 'test fixture mechanics only; not a canonical tuning/admission artifact'}
    if passed:
        validation, validation_hash = _partition('validation')
        checked = [_check_calibration_fixture(_scope_fixture(row, count), provider)
                   for row in validation['providers'][provider]]
        result.update(validation=checked, validation_sha256=validation_hash,
                      validated_nomination=all(row['passed'] for row in checked))
    _save(request, 'nonlinear-scope-calibration', result)


def _nomination(provider):
    records = []
    for count in COUNTS:
        try:
            number, run = _latest(f'nonlinear_scope_calibrate_{provider}_{count}_cpu')
        except AssertionError:
            continue
        assert run['state'] == 'passed'
        result = json.loads((RAW/f'run-{number:05d}'/'nonlinear-scope-calibration.json').read_text())
        for name, digest in run['source_sha256'].items():
            if name.startswith('bayesfilter/'):
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
        records.append((number, result))
    successful = [(n, r) for n, r in records if r['validated_nomination']]
    assert len(successful) == 1
    number, result = successful[0]
    for _, other in records:
        if other['count'] < result['count']:
            assert not other['calibration_passed'] and other['validation'] is None
        else:
            assert other['count'] == result['count']
    assert result['calibration_sha256'] == _partition('calibration')[1]
    assert result['validation_sha256'] == _partition('validation')[1]
    return number, result['count']


@pytest.mark.parametrize('provider', PROVIDERS)
def test_untouched_scope(provider, request, monkeypatch):
    _qualify_untouched_scope(provider, request, monkeypatch, diagnostics=True)


@pytest.mark.parametrize('provider', PROVIDERS)
def test_untouched_plain_scope(provider, request, monkeypatch):
    _qualify_untouched_scope(provider, request, monkeypatch, diagnostics=False)


def _qualify_untouched_scope(provider, request, monkeypatch, *, diagnostics):
    nomination_run, count = _nomination(provider)
    untouched, untouched_hash = _partition('untouched')
    reports = []
    for row in untouched['providers'][provider]:
        fixture = _scope_fixture(row, count)
        case = provider+'_diagnostics' if diagnostics else provider
        scalar, owner = _factory(case, fixture=fixture), _factory(case, all_directions=True, fixture=fixture)
        theta = tf.constant(fixture['theta'], DTYPE)
        directions = tf.eye(6, dtype=DTYPE)
        args = _operands(case, fixture=fixture)
        actual = owner(theta, directions, *args)
        assert bool(actual['valid']) and bool(actual['value_invariant'])
        baseline_error = _compare(actual['outputs'], _reference(scalar, theta, directions, args))
        changed_theta = tf.constant(fixture['changed_theta'], DTYPE)
        changed_args = (tf.constant(fixture['changed_observations'], DTYPE), *args[1:])
        changed_directions = .7*tf.roll(directions, 1, 0)
        changed = owner(changed_theta, changed_directions, *changed_args)
        assert bool(changed['valid']) and bool(changed['value_invariant'])
        assert abs(float(actual['value']-changed['value'])) > 1e-7
        changed_error = _compare(changed['outputs'], _reference(scalar, changed_theta, changed_directions, changed_args))
        errors = []
        for step in (2e-4, 1e-4):
            fd = []
            for direction in tf.unstack(directions):
                h = tf.constant(step, DTYPE)*direction
                f = lambda k, h=h, direction=direction, scalar=scalar, theta=theta, args=args: float(scalar(theta+k*h, direction, *args)[0])
                fd.append((-f(2)+8*f(1)-8*f(-1)+f(-2))/(12*step))
            error = float(np.max(np.abs(np.asarray(fd)-actual['score'].numpy())))
            errors.append(error)
            assert error <= 2e-6, (provider, row['case'], step, error)
        endpoint = _endpoint(case, owner, actual, monkeypatch, fixture=fixture)
        suffix = '' if diagnostics else '-plain'
        graph = _graph(owner, (theta, directions, *args), request, f'nonlinear-scope{suffix}-{provider}-{row["case"]}')
        reports.append({'case': row['case'], 'baseline_error': baseline_error, 'changed_error': changed_error,
            'five_point_steps': [2e-4, 1e-4], 'five_point_errors': errors, 'graph': graph, 'endpoint': endpoint})
        _save(request, 'nonlinear-scope-untouched'+suffix, {'provider': provider, 'count': count,
            'diagnostics_enabled': diagnostics,
            'nomination_run': nomination_run, 'untouched_sha256': untouched_hash, 'cases': reports,
            'scope': 'new calibrated test scope; failed seed9292027 remains refused and excluded'})
