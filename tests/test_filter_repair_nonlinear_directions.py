"""Fresh scalar nonlinear endpoint qualification; references only use host loops."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from tests.test_filter_repair_score_directions import (
    DTYPE,
    _compare,
    _graph,
    _reference,
    _save,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT/'tests/fixtures/filter_repair_nonlinear_directions_20260929.json'
FIXTURE = json.loads(FIXTURE_PATH.read_text())
CASES = ('ekf', 'ukf', 'ledh', 'ledh_diagnostics', 'sgqf', 'sgqf_diagnostics',
         'kdm_covariance', 'kdm_covariance_diagnostics')


def _proposal(case):
    return case.removesuffix('_diagnostics')


def _factory(case, *, all_directions=False, fixture=FIXTURE):
    from bayesfilter.score_study.nonlinear_tf import (
        make_ledh_kernel,
        make_moment_filter,
    )
    c, b = fixture['transition_curve'], fixture['observation_curve']
    proposal = _proposal(case)
    if proposal in ('ekf', 'ukf'):
        return make_moment_filter(2, c, b, proposal, 'float64', True, all_directions=all_directions)
    control = {'ledh': 1., 'sgqf': fixture['sgqf_level'], 'kdm_covariance': fixture['within_fraction']}[proposal]
    return make_ledh_kernel(8, 2, tuple(sorted(fixture['controls'].items())), c, b, proposal, control,
        'float64', True, return_diagnostics=case.endswith('_diagnostics'), all_directions=all_directions)


def _operands(case, *, fixture=FIXTURE):
    names = ('observations',) if case in ('ekf', 'ukf') else ('observations', 'initial', 'process', 'reset_design')
    return tuple(tf.constant(fixture[name], DTYPE) for name in names)


def _endpoint(case, owner, numerical, monkeypatch, *, expected_valid=True, fixture=FIXTURE):
    from bayesfilter.score_study import (
        input_execution_tf,
        nonlinear_adapter,
        nonlinear_tf,
    )
    from bayesfilter.score_study.contracts import DiagnosticFailure, seed_pair
    from bayesfilter.score_study.registry import default_registry
    proposal = _proposal(case)
    device = 'CPU' if os.environ['CUDA_VISIBLE_DEVICES'] == '-1' else 'GPU'
    row = {'model': 'nonlinear_scalar', 'proposal': proposal, 'estimator': 'nonlinear_analytical',
        'comparison_target': 'model_score', 'comparison': 'approximation_error', 'dataset': 1,
        'replicate': 0, 'role': 'mechanics', 'controls': fixture['controls'],
        'sgqf_level': fixture['sgqf_level'], 'within_fraction': fixture['within_fraction'],
        'collect_control_diagnostics': case.endswith('_diagnostics')}
    settings = {key: fixture[key] for key in ('dimension', 'observation_dimension', 'horizon', 'particles',
        'dtype', 'theta', 'transition_curve', 'observation_curve', 'reference')}
    settings.update(device=device, tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
                    jit_compile=True, data_theta=fixture['theta'])
    context = {'study': {'seed': fixture['seed'], 'settings': settings, 'evidence_class': 'mechanics'},
               'registry': default_registry()}
    def seed(stream):
        return tuple(seed_pair(master_seed=fixture['seed'], model=row['model'], dataset=1,
                               replicate=0, stream=stream, coupling_group='baseline'))
    frozen = {seed(stream): tf.constant(fixture[name], DTYPE) for stream, name in
        {'initial': 'initial', 'process': 'process', 'reset_design': 'reset_design',
         'resampling': 'resampling_uniforms'}.items()}

    def random(shape, seed, dtype):
        result = frozen[tuple(seed)]
        assert result.shape == shape and result.dtype == dtype
        return result

    factory_name = 'make_moment_filter' if proposal in ('ekf', 'ukf') else 'make_ledh_kernel'
    original = getattr(nonlinear_tf, factory_name)
    seen = []
    def observed(*args, **kwargs):
        assert kwargs['all_directions'] is True
        result = original(*args, **kwargs)
        seen.append(result)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(nonlinear_tf, factory_name, observed)
        patch.setattr(nonlinear_adapter, 'configure_runtime', lambda **_: {
            'device': device, 'jit_compile': True, 'tf32': settings['tf32'],
            'initialization': 'campaign_worker_already_configured', 'reference_exception': device == 'CPU'})
        patch.setattr(nonlinear_tf, 'make_data_kernel', lambda *_:
            lambda *_: tf.constant(fixture['observations'], DTYPE))
        patch.setattr(tf.random, 'stateless_normal', random)
        patch.setattr(tf.random, 'stateless_uniform', random)
        patch.setattr(input_execution_tf, 'make_score_inputs', lambda *_, **__:
            lambda seeds: tuple(frozen[tuple(seed)] for seed in seeds.numpy().tolist()))
        if not expected_valid:
            for _ in range(2):
                with pytest.raises(tf.errors.InvalidArgumentError, match='nonlinear score validity veto'):
                    nonlinear_adapter.evaluate_nonlinear(row, context)
            assert len(seen) == 2 and seen[0] is seen[1] is owner
            assert owner.experimental_get_tracing_count() == 1
            return {'numerical_validity': 'rejected', 'exception': 'nonlinear score validity veto',
                    'actual_factory_owner_verified': True, 'trace_count': 1, 'device': device}
        first = nonlinear_adapter.evaluate_nonlinear(row, context)
        second = nonlinear_adapter.evaluate_nonlinear(row, context)
        assert len(seen) == 2 and seen[0] is seen[1] is owner
        if proposal in ('ekf', 'ukf'):
            def invalid_factory(*args, **kwargs):
                chosen = observed(*args, **kwargs)
                return lambda theta, directions, data: chosen(theta, directions, data*float('nan'))
            patch.setattr(nonlinear_tf, factory_name, invalid_factory)
        else:
            frozen[seed('reset_design')] = tf.fill([8, 1], tf.constant(float('nan'), DTYPE))
        if case.endswith('_diagnostics'):
            error, message = DiagnosticFailure, 'canonical control diagnostics invalid'
        else:
            error, message = tf.errors.InvalidArgumentError, 'nonlinear score validity veto'
        with pytest.raises(error, match=message):
            nonlinear_adapter.evaluate_nonlinear(row, context)
    assert first['runtime']['kernel_calls'] == 1 and first['runtime']['traces'] == 1
    assert second['runtime']['traces'] == 1
    assert device in first['runtime']['value_device'] and device in first['runtime']['score_device']
    diag = first['diagnostics']
    assert diag['direction_execution'] == 'enclosing_tensorflow_loop'
    assert 'CPU' in diag['reference_device']
    assert diag['reference_mesh_relative_error'] <= fixture['reference']['relative_tolerance']
    assert diag['reference_domain_relative_error'] <= fixture['reference']['relative_tolerance']
    assert diag['reference_max_tail_mass'] <= fixture['reference']['tail_tolerance']
    np.testing.assert_allclose(first['value'], numerical['value'], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(first['score'], numerical['score'], atol=1e-9, rtol=1e-9)
    if case.endswith('_diagnostics'):
        assert 'higher_moment_valid' in diag['control_diagnostics']
    return first


@pytest.mark.parametrize('case', CASES)
def test_nonlinear_direction_consumer(case, request, monkeypatch):
    theta = tf.constant(FIXTURE['theta'], DTYPE)
    directions = tf.eye(6, dtype=DTYPE)
    args = _operands(case)
    scalar, owner = _factory(case), _factory(case, all_directions=True)
    actual = owner(theta, directions, *args)
    if not bool(actual['valid']):
        assert case not in ('ekf', 'ukf'), 'previously healthy moment-filter fixture failed'
        reference = _reference(scalar, theta, directions, args)
        maximum_error = 0.
        for a, b in zip(tf.nest.flatten(reference), tf.nest.flatten(actual['outputs']), strict=True):
            if a.dtype.is_floating:
                np.testing.assert_array_equal(tf.math.is_finite(a), tf.math.is_finite(b))
                np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9, equal_nan=True)
                finite = np.isfinite(a.numpy())
                maximum_error = max(maximum_error, float(np.max(np.abs(a.numpy()[finite]-b.numpy()[finite]), initial=0)))
            else:
                np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(reference[0], np.full(6, -np.inf))
        np.testing.assert_array_equal(reference[1], np.zeros(6))
        endpoint = _endpoint(case, owner, actual, monkeypatch, expected_valid=False)
        graph = _graph(owner, (theta, directions, *args), request, 'nonlinear-directions-'+case)
        _save(request, 'nonlinear-directions-'+case, {
            'case': case, 'fixture_sha256': hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(),
            'qualification': 'original_and_enclosing_refusal_only_healthy_T2_gate_open',
            'finite_auxiliary_max_absolute_error': maximum_error,
            'scalar_values': reference[0].numpy().tolist(), 'scalar_scores': reference[1].numpy().tolist(),
            'host_nonfinite_rejection_checked': False, 'host_original_rejection_checked': True,
            'graph': graph, 'endpoint': endpoint})
        return
    assert bool(actual['value_invariant'])
    expected = _reference(scalar, theta, directions, args)
    baseline_error = _compare(actual['outputs'], expected)
    _compare(actual['first_auxiliary'], tf.nest.map_structure(lambda x: x[0], expected[2:]))
    changed_theta = tf.constant(FIXTURE['changed_theta'], DTYPE)
    changed_args = (tf.constant(FIXTURE['changed_observations'], DTYPE), *args[1:])
    changed_directions = .7*tf.roll(directions, 1, 0)
    changed = owner(changed_theta, changed_directions, *changed_args)
    assert bool(changed['valid']) and bool(changed['value_invariant'])
    changed_error = _compare(changed['outputs'], _reference(scalar, changed_theta, changed_directions, changed_args))
    assert abs(float(actual['value']-changed['value'])) > 1e-7
    fd_errors = []
    for step in (2e-4, 1e-4):
        fd = []
        for direction in tf.unstack(directions):
            h = tf.constant(step, DTYPE)*direction
            f = lambda k, h=h, direction=direction: float(scalar(theta+k*h, direction, *args)[0])
            fd.append((-f(2)+8*f(1)-8*f(-1)+f(-2))/(12*step))
        error = float(np.max(np.abs(np.asarray(fd)-actual['score'].numpy())))
        fd_errors.append(error)
        assert error <= 2e-6, (case, step, error)
    bad_args = (args[0]*float('nan'),) if case in ('ekf', 'ukf') else (*args[:-1], args[-1]*float('nan'))
    assert not bool(owner(theta, directions, *bad_args)['valid'])
    endpoint = _endpoint(case, owner, actual, monkeypatch)
    graph = _graph(owner, (theta, directions, *args), request, 'nonlinear-directions-'+case)
    _save(request, 'nonlinear-directions-'+case, {
        'case': case, 'fixture_sha256': hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(),
        'baseline_max_absolute_error': baseline_error, 'changed_max_absolute_error': changed_error,
        'five_point_steps': [2e-4, 1e-4], 'five_point_max_absolute_errors': fd_errors,
        'host_nonfinite_rejection_checked': True, 'graph': graph, 'endpoint': endpoint,
        'scope': 'scalar nonlinear execution-policy mechanics, own finite scalar, no scientific admission'})


@pytest.mark.parametrize('provider', ['ledh', 'sgqf', 'kdm_covariance'])
def test_nonlinear_ledh_invalid_fixture_localization(provider, request, monkeypatch):
    """Preserve the failed fixture; explain status without qualifying a score."""
    theta = tf.constant(FIXTURE['theta'], DTYPE)
    directions = tf.eye(6, dtype=DTYPE)
    args = _operands(provider)
    from bayesfilter.score_study import control_diagnostics_tf
    original = control_diagnostics_tf.compact_control_diagnostics

    def observe(trace):
        result = original(trace)
        trace_tensors = tf.nest.map_structure(lambda *values: tf.stack(values), *trace)
        residual = tf.reduce_mean(trace_tensors['reset_transport'], 1)-trace_tensors['posterior_weights']
        result['diagnostic_reset_column_tv'] = .5*tf.reduce_sum(tf.abs(residual), 1)
        result['diagnostic_program_valid'] = trace_tensors['program_valid']
        return result

    monkeypatch.setattr(control_diagnostics_tf, 'compact_control_diagnostics', observe)
    scalar = _factory(provider+'_diagnostics')
    owner = _factory(provider+'_diagnostics', all_directions=True)
    reference = _reference(scalar, theta, directions, args)
    result = owner(theta, directions, *args)
    finite = bool(tf.reduce_all(tf.math.is_finite(reference[0])))
    finite &= bool(tf.reduce_all(tf.math.is_finite(reference[1])))
    assert bool(result['valid']) == finite
    if not finite:
        assert bool(tf.reduce_any(reference[2]['diagnostic_reset_column_tv'] > 1e-4))
    for a, b in zip(tf.nest.flatten(reference), tf.nest.flatten(result['outputs']), strict=True):
        if a.dtype.is_floating:
            np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9, equal_nan=True)
        else:
            np.testing.assert_array_equal(a, b)
    report = {'provider': provider, 'fixture_sha256': hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(),
        'scalar_values': reference[0].numpy().tolist(), 'scalar_scores': reference[1].numpy().tolist(),
        'enclosing_valid': bool(result['valid']), 'enclosing_value_invariant': bool(result['value_invariant']),
        'scalar_diagnostics': tf.nest.map_structure(lambda x: x.numpy().tolist(), reference[2]),
        'scope': 'localization of original frozen fixture; no healthy finite-score qualification'}
    _save(request, 'nonlinear-'+provider+'-invalid-fixture', report)
