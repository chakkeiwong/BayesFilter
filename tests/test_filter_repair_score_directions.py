"""Fresh September fixtures for complete analytical score consumers.

Python/NumPy loops are independent references and artifact boundaries only.
No historical LEDH fixture, learned map, training or HMC execution is used.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.score_study.direction_assembly_tf import make_direction_kernel

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / 'tests/fixtures/filter_repair_score_directions_20260929.json'
FIXTURE = json.loads(FIXTURE_PATH.read_text())
DTYPE = tf.float64
CASES = ('ledh', 'ledh_diagnostics', 'sgqf', 'kdm_covariance', 'integrated_kdm', 'resampling_kdm')


def _directory(request):
    return Path(request.config.getoption('xmlpath')).parent


def _save(request, name, report):
    (_directory(request) / (name + '.json')).write_text(json.dumps(report, indent=2) + '\n')


def _graph(owner, args, request, name):
    assert owner.experimental_get_tracing_count() == 1
    concrete = owner.get_concrete_function()
    assert concrete.function_def.attr['_XlaMustCompile'].b
    graph = concrete.graph.as_graph_def()
    nodes = [*graph.node, *(n for f in graph.library.function for n in f.node_def)]
    assert not {n.op for n in nodes} & {'PyFunc', 'EagerPyFunc', 'PyFuncStateless', 'XlaHostCompute'}
    assert not any('/pfor/' in n.name for n in nodes)
    assert any(n.op in ('While', 'StatelessWhile') for n in nodes)
    hlo = owner.experimental_get_compiler_ir(*args)(stage='hlo')
    (_directory(request) / (name + '.hlo.txt')).write_text(hlo)
    return {'trace_count': 1, 'jit_compile': True, 'no_pfor_or_host_callback': True,
            'hlo_sha256': hashlib.sha256(hlo.encode()).hexdigest()}


def _compare(actual, expected):
    tf.nest.assert_same_structure(actual, expected)
    errors = []
    for lhs, rhs in zip(tf.nest.flatten(actual), tf.nest.flatten(expected), strict=True):
        a, b = lhs.numpy(), rhs.numpy()
        assert a.shape == b.shape and a.dtype == b.dtype
        if lhs.dtype.is_floating:
            np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9)
            errors.append(float(np.max(np.abs(a-b), initial=0)))
        else:
            np.testing.assert_array_equal(a, b)
    return max(errors, default=0.)


def _reference(scalar, theta, directions, args):
    return tf.nest.map_structure(lambda *values: tf.stack(values),
        *(scalar(theta, direction, *args) for direction in tf.unstack(directions)))


@pytest.mark.parametrize('jit', [True, False])
def test_generic_direction_owner(jit, request):
    @tf.function(input_signature=[tf.TensorSpec([3], DTYPE)]*3,
                 jit_compile=jit, autograph=False)
    def scalar(theta, direction, data):
        weights = tf.constant([1.3, 2.1, .8], DTYPE)
        value = tf.reduce_sum(weights*theta**2/2 + data*theta)
        gradient = weights*theta+data
        return (value, tf.reduce_sum(direction*gradient), tf.reduce_sum(direction) < 2,
                {'gradient': gradient, 'direction': direction,
                 'integer': tf.constant([2, 7], tf.int32)})

    owner = make_direction_kernel(scalar, 3, jit_compile=jit, validity_output_index=2)
    theta = tf.constant([.2, -.3, .4], DTYPE)
    data = tf.constant([.1, -.2, .3], DTYPE)
    directions = tf.eye(3, dtype=DTYPE)
    errors = []
    for point, rows, observations in ((theta, directions, data),
            (theta+.17, tf.roll(directions, 1, 0)*.7, data-.08)):
        result = owner(point, rows, observations)
        assert bool(result['valid']) and bool(result['value_invariant'])
        expected = _reference(scalar, point, rows, (observations,))
        errors.append(_compare(result['outputs'], expected))
        _compare(result['first_auxiliary'], tf.nest.map_structure(lambda x: x[0], expected[2:]))
    rejected = owner(theta, directions*3, data)
    assert not bool(rejected['valid']) and bool(rejected['value_invariant'])
    assert not bool(owner(theta*float('nan'), directions, data)['valid'])
    assert owner.experimental_get_tracing_count() == 1
    report = {'jit_compile': jit, 'reference_exception': not jit, 'errors': errors,
              'per_direction_invalidity_rejected': True, 'trace_count': 1}
    if jit:
        report['graph'] = _graph(owner, (theta, directions, data), request, 'directions-generic')
    _save(request, f'directions-generic-{jit}', report)


def test_direction_value_invariance_is_returned_not_an_ignored_assertion(request):
    @tf.function(input_signature=[tf.TensorSpec([2], DTYPE)]*2,
                 jit_compile=True, autograph=False)
    def inconsistent(theta, direction):
        return tf.reduce_sum(theta**2)+direction[0], tf.reduce_sum(2*theta*direction)

    owner = make_direction_kernel(inconsistent, 2)
    result = owner(tf.constant([.2, -.3], DTYPE), tf.eye(2, dtype=DTYPE))
    assert bool(result['valid']) and not bool(result['value_invariant'])
    _save(request, 'directions-invariance-veto', {'finite': True, 'value_invariant': False})


def _factory(case, *, all_directions=False, replay=False):
    controls = tuple(sorted(FIXTURE['controls'].items()))
    args = (2, 1, 8, 2, controls)
    if case.startswith('ledh'):
        from bayesfilter.score_study.canonical_adapter_tf import make_canonical_kernel
        return make_canonical_kernel(*args, 'float64', True,
            return_diagnostics=case == 'ledh_diagnostics', all_directions=all_directions)
    if case == 'sgqf':
        from bayesfilter.score_study.covariance_adapter_tf import make_covariance_kernel
        return make_covariance_kernel(*args, FIXTURE['sgqf_level'], 'float64', True,
            all_directions=all_directions)[0]
    if case == 'kdm_covariance':
        from bayesfilter.score_study.mixture_covariance_tf import (
            make_mixture_covariance_kernel,
        )
        return make_mixture_covariance_kernel(*args, FIXTURE['within_fraction'], 'float64', True,
            all_directions=all_directions)
    from bayesfilter.score_study.kdm_adapter_tf import make_kdm_kernel
    if replay:
        return make_kdm_kernel(*args, case, 'float64', True, replay=True, all_directions=all_directions)
    return make_kdm_kernel(*args, case, 'float64', True, all_directions=all_directions)


def _operands(case):
    args = tuple(tf.constant(FIXTURE[name], DTYPE) for name in
                 ('observations', 'initial', 'process', 'reset_design'))
    if case in ('integrated_kdm', 'resampling_kdm'):
        args += tuple(tf.constant(FIXTURE[name], DTYPE) for name in
                      ('mixture_uniforms', 'mixture_noise', 'bandwidth_scale'))
    return args


def _endpoint(case, owner, result, monkeypatch):
    """Keep the real endpoint/factories; inject only frozen streams/runtime."""
    from bayesfilter.score_study import adapters, gaussian_tf
    from bayesfilter.score_study.contracts import seed_pair
    from bayesfilter.score_study.registry import default_registry
    proposal = 'ledh' if case.startswith('ledh') else case
    device = 'CPU' if os.environ['CUDA_VISIBLE_DEVICES'] == '-1' else 'GPU'
    tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
    row = {'model': 'gaussian_all_parameters', 'proposal': proposal, 'estimator': 'analytical_filter',
        'comparison_target': 'model_score', 'comparison': 'approximation_error', 'dataset': 1,
        'replicate': 0, 'role': 'mechanics', 'controls': FIXTURE['controls'],
        'sgqf_level': FIXTURE['sgqf_level'], 'within_fraction': FIXTURE['within_fraction'],
        'bandwidth_scale': FIXTURE['bandwidth_scale'], 'collect_control_diagnostics': case == 'ledh_diagnostics'}
    settings = {'dimension': 2, 'observation_dimension': 1, 'horizon': 2, 'particles': 8,
        'dtype': 'float64', 'device': device, 'tf32': tf32, 'jit_compile': True,
        'theta': FIXTURE['theta'], 'data_theta': FIXTURE['theta']}
    context = {'study': {'seed': FIXTURE['seed'], 'settings': settings, 'evidence_class': 'mechanics'},
                   'registry': default_registry()}
    streams = {'initial': 'initial', 'process': 'process', 'resampling': 'resampling_uniforms',
               'reset_design': 'reset_design', 'kdm_noise': 'mixture_noise', 'kdm_components': 'mixture_uniforms'}
    frozen = {tuple(seed_pair(master_seed=FIXTURE['seed'], model=row['model'], dataset=1,
        replicate=0, stream=stream, coupling_group='baseline')): tf.constant(FIXTURE[name], DTYPE)
        for stream, name in streams.items()}

    def random(shape, seed, dtype):
        value = frozen[tuple(seed)]
        assert value.shape == shape and value.dtype == dtype
        return value

    modules = {'ledh': ('canonical_adapter_tf', 'make_canonical_kernel'),
        'sgqf': ('covariance_adapter_tf', 'make_covariance_kernel'),
        'kdm_covariance': ('mixture_covariance_tf', 'make_mixture_covariance_kernel'),
        'integrated_kdm': ('kdm_adapter_tf', 'make_kdm_kernel'),
        'resampling_kdm': ('kdm_adapter_tf', 'make_kdm_kernel')}
    import importlib
    module_name, function_name = modules[proposal]
    module = importlib.import_module('bayesfilter.score_study.'+module_name)
    original = getattr(module, function_name)
    seen = []

    def observed(*args, **kwargs):
        assert kwargs['all_directions'] is True
        chosen = original(*args, **kwargs)
        seen.append(chosen[0] if proposal == 'sgqf' else chosen)
        return chosen

    with monkeypatch.context() as patch:
        patch.setattr(module, function_name, observed)
        patch.setattr(adapters, 'configure_runtime', lambda **_: {
            'device': device, 'jit_compile': True, 'tf32': tf32,
            'initialization': 'campaign_worker_already_configured', 'reference_exception': device == 'CPU'})
        patch.setattr(gaussian_tf, 'make_data_kernel', lambda *_:
            lambda *_: tf.constant(FIXTURE['observations'], DTYPE))
        patch.setattr(tf.random, 'stateless_normal', random)
        patch.setattr(tf.random, 'stateless_uniform', random)
        first = adapters.evaluate_gaussian(row, context)
        second = adapters.evaluate_gaussian(row, context)
        reset_seed = seed_pair(master_seed=FIXTURE['seed'], model=row['model'], dataset=1,
            replicate=0, stream='reset_design', coupling_group='baseline')
        frozen[tuple(reset_seed)] = tf.fill([8, 2], tf.constant(float('nan'), DTYPE))
        if case == 'ledh_diagnostics':
            from bayesfilter.score_study.contracts import DiagnosticFailure
            error, message = DiagnosticFailure, 'canonical control diagnostics invalid'
        elif case in ('integrated_kdm', 'resampling_kdm'):
            error, message = ValueError, 'KDM consumer validity veto'
        else:
            error, message = tf.errors.InvalidArgumentError, 'analytical score validity veto'
        with pytest.raises(error, match=message):
            adapters.evaluate_gaussian(row, context)
    assert len(seen) == 3 and seen[0] is seen[1] is seen[2] is owner
    assert first['runtime']['kernel_calls'] == 1 and first['runtime']['traces'] == 1
    assert second['runtime']['traces'] == 1
    assert device in first['runtime']['value_device'] and device in first['runtime']['score_device']
    assert first['diagnostics']['direction_execution'] == 'enclosing_tensorflow_loop'
    np.testing.assert_allclose(first['value'], result['value'], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(first['score'], result['score'], atol=1e-9, rtol=1e-9)
    if case == 'ledh_diagnostics':
        assert 'higher_moment_valid' in first['diagnostics']['control_diagnostics']
    if case == 'kdm_covariance':
        assert len(first['diagnostics']['final_component_log_weights']) == 4
    return first


@pytest.mark.parametrize('case', CASES)
def test_actual_direction_consumer(case, request, monkeypatch):
    theta = tf.constant(FIXTURE['theta'], DTYPE)
    directions = tf.eye(6, dtype=DTYPE)
    args = _operands(case)
    scalar, owner = _factory(case), _factory(case, all_directions=True)
    actual = owner(theta, directions, *args)
    assert bool(actual['valid']) and bool(actual['value_invariant'])
    expected = _reference(scalar, theta, directions, args)
    baseline_error = _compare(actual['outputs'], expected)
    _compare(actual['first_auxiliary'], tf.nest.map_structure(lambda x: x[0], expected[2:]))
    changed_theta = tf.constant(FIXTURE['changed_theta'], DTYPE)
    changed_args = (tf.constant(FIXTURE['changed_observations'], DTYPE), *args[1:])
    changed_directions = tf.roll(directions, 1, 0)*.7
    changed = owner(changed_theta, changed_directions, *changed_args)
    assert bool(changed['valid']) and bool(changed['value_invariant'])
    changed_error = _compare(changed['outputs'], _reference(scalar, changed_theta, changed_directions, changed_args))
    assert abs(float(changed['value']-actual['value'])) > 1e-7
    derivative = scalar
    derivative_args = args
    if case == 'resampling_kdm':
        derivative = _factory(case, replay=True)
        derivative_args = (*args, *actual['first_auxiliary'][1:])
    fd_errors = []
    for step in (2e-4, 1e-4):
        differences = []
        for direction in tf.unstack(directions):
            h = tf.constant(step, DTYPE)*direction
            f = lambda multiplier, h=h, direction=direction: float(derivative(theta+multiplier*h, direction, *derivative_args)[0])
            differences.append((-f(2)+8*f(1)-8*f(-1)+f(-2))/(12*step))
        error = float(np.max(np.abs(np.asarray(differences)-actual['score'].numpy())))
        fd_errors.append(error)
        assert error <= 2e-6, (case, step, error)
    zero_args = (*args[:3], tf.zeros_like(args[3]), *args[4:])
    zero_reset = owner(theta, directions, *zero_args)
    zero_reference = _reference(scalar, theta, directions, zero_args)
    zero_valid = bool(tf.reduce_all(tf.math.is_finite(zero_reference[0])))
    zero_valid &= bool(tf.reduce_all(tf.math.is_finite(zero_reference[1])))
    if case in ('integrated_kdm', 'resampling_kdm'):
        zero_valid &= bool(tf.reduce_all(zero_reference[2]))
    assert bool(zero_reset['valid']) == zero_valid
    if zero_valid:
        _compare(zero_reset['outputs'], zero_reference)
    bad_args = (*args[:3], tf.fill(args[3].shape, tf.constant(float('nan'), DTYPE)), *args[4:])
    invalid_reset = owner(theta, directions, *bad_args)
    assert not bool(invalid_reset['valid']), 'nonfinite reset design must report invalidity'
    if case in ('integrated_kdm', 'resampling_kdm'):
        assert not bool(owner(theta, directions, *args[:-1], tf.constant(-.23, DTYPE))['valid'])
    endpoint = _endpoint(case, owner, actual, monkeypatch)
    graph = _graph(owner, (theta, directions, *args), request, 'directions-'+case)
    _save(request, 'directions-'+case, {'case': case, 'fixture_sha256': hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(),
        'baseline_max_absolute_error': baseline_error, 'changed_max_absolute_error': changed_error,
        'five_point_steps': [2e-4, 1e-4], 'five_point_max_absolute_errors': fd_errors,
        'zero_design_reference_valid': zero_valid, 'nonfinite_reset_rejected': True,
        'host_nonfinite_rejection_checked': True, 'graph': graph, 'endpoint': endpoint,
        'scope': 'fresh execution-policy mechanics only, no canonical or scientific admission'})
