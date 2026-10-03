"""Independent archived references for fitted-APF execution-policy repair."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.score_study.contracts import seed_pair
from bayesfilter.score_study.fitted_twist_execution_tf import (
    make_fitted_twist_execution,
    make_fitted_twist_inputs,
)
from bayesfilter.score_study.fitted_twist_tf import quadratic_features

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT/'tests/fixtures/filter_repair_fitted_apf_20260929'
FIXTURE = json.loads((FIXTURES/'fixture.json').read_text())


def _reference_module(name):
    path = FIXTURES/f'{name}_reference.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FIXTURE['baseline_sha256'][
        f'bayesfilter/score_study/{name}.py']
    spec = importlib.util.spec_from_file_location(f'bayesfilter.score_study._reference_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(model, variant=0):
    def seed(stream, *, replicate=None, group=None):
        return seed_pair(master_seed=FIXTURE['seed'] + variant, model=model, dataset=1,
            replicate=0 if replicate is None else replicate, stream=stream,
            coupling_group=group or 'baseline')
    return seed


def _seeds(model, variant=0):
    seed = _seed(model, variant)
    rows = [[seed(f'fit{i}_{name}', replicate=0, group='offline_fitted_twist')
             for name in ('initial', 'process', 'ancestors', 'mixture')] for i in range(2)]
    rows.append([seed(f'final_twist_{name}') for name in ('initial', 'process', 'ancestors', 'mixture')])
    return tf.constant(rows, tf.int32)


def _draws(seeds, dtype):
    return (tf.random.stateless_normal([16, 1], seeds[0], dtype=dtype),
            tf.random.stateless_normal([2, 16, 1], seeds[1], dtype=dtype),
            tf.random.stateless_uniform([3, 16], seeds[2], dtype=dtype),
            tf.random.stateless_uniform([2, 16], seeds[3], dtype=dtype))


def _host(value):
    if tf.is_tensor(value):
        return value.numpy().tolist()
    if isinstance(value, dict):
        return {key: _host(part) for key, part in value.items()}
    if isinstance(value, (list, tuple)):
        return [_host(part) for part in value]
    return value


def _save(request, name, report):
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory/f'{name}.json').write_text(json.dumps(_host(report), indent=2)+'\n')


def _graph(request, name, owner, args):
    concrete = owner.get_concrete_function()
    assert concrete.function_def.attr['_XlaMustCompile'].b
    graph = concrete.graph.as_graph_def()
    ops = [node.op for node in graph.node]
    ops += [node.op for function in graph.library.function for node in function.node_def]
    assert not {'PyFunc', 'EagerPyFunc', 'PyFuncStateless'}.intersection(ops)
    hlo = owner.experimental_get_compiler_ir(*args)(stage='hlo')
    assert 'while' in hlo.lower()
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory/f'{name}.hlo.txt').write_text(hlo)
    return {'trace_count': owner.experimental_get_tracing_count(),
            'hlo_sha256': hashlib.sha256(hlo.encode()).hexdigest(), 'jit_compile': True,
            'host_callbacks': False}


@pytest.mark.parametrize('dtype_name', ['float64', 'float32'])
def test_fitted_apf_input_and_feature_primitives(dtype_name, request):
    dtype = tf.as_dtype(dtype_name)
    inputs = make_fitted_twist_inputs(1, 16, 2, dtype_name)
    records = []
    for model in ('gaussian', 'nonlinear_scalar'):
        for seeds in tf.unstack(_seeds(model)):
            expected, actual = _draws(seeds, dtype), inputs(seeds)
            for index, (left, right) in enumerate(zip(expected, actual)):
                if index >= 2:
                    np.testing.assert_array_equal(left.numpy(), right.numpy())
                else:
                    np.testing.assert_allclose(left.numpy(), right.numpy(), rtol=0,
                        atol=1e-12 if dtype == tf.float64 else 2e-6)
            records.append({'model': model, 'seeds': seeds, 'reference': expected, 'candidate': actual})
    reference = _reference_module('fitted_twist_tf')
    features = tf.function(quadratic_features, input_signature=[tf.TensorSpec([7, 3], dtype)],
                           jit_compile=True)
    original_features = tf.function(reference.quadratic_features,
        input_signature=[tf.TensorSpec([7, 3], dtype)], jit_compile=True)
    points = tf.reshape(tf.linspace(tf.cast(-2., dtype), tf.cast(3., dtype), 21), [7, 3])
    eager = reference.quadratic_features(points)
    expected, actual = original_features(points), features(points)
    _save(request, f'fitted-apf-primitives-{dtype_name}', {
        'records': records, 'features': {'eager_reference': eager, 'reference': expected, 'candidate': actual}})
    np.testing.assert_array_equal(expected.numpy(), actual.numpy())
    assert inputs.experimental_get_tracing_count() == 1
    assert inputs.get_concrete_function().function_def.attr['_XlaMustCompile'].b
    _save(request, f'fitted-apf-primitives-{dtype_name}', {
        'records': records, 'features': {'eager_reference': eager, 'reference': expected, 'candidate': actual},
        'trace_count': 1, 'dtype': dtype_name, 'device': actual.device,
        'nonclaim': 'Primitive compatibility; no endpoint or performance admission.'})


def _reference_program(dtype_name, curves, theta, fit_theta, observations, seeds):
    original = _reference_module('fitted_twist_tf')
    dtype = tf.as_dtype(dtype_name)
    kernel = original.make_fitted_twist_kernel(1, 1, 16, 2, dtype_name, True, **curves)
    fitter = original.make_recursive_fit_kernel(1, 1, 16, 2, .01, dtype_name, True, **curves)
    centers, covariance = tf.zeros([2, 1], dtype), tf.eye(1, batch_shape=[2], dtype=dtype)
    floors = tf.fill([2], tf.math.log(tf.cast(.01, dtype)) -
        .5 * tf.math.log(tf.cast(2 * 3.141592653589793, dtype)))
    histories = []
    for i in range(2):
        fitted = kernel(fit_theta, observations, *_draws(seeds[i], dtype), centers, covariance, floors)
        centers, covariance, floors, valid, errors = fitter(fit_theta, observations, fitted[2])
        assert bool(valid.numpy()), f'independent fixture invalid at {i}'
        histories.append((fitted[0], errors, centers, covariance, floors))
    return {'final': kernel(theta, observations, *_draws(seeds[2], dtype), centers, covariance, floors),
            'fit': (centers, covariance, floors), 'fit_log_values': tf.stack([h[0] for h in histories]),
            'fit_errors': tf.stack([h[1] for h in histories]),
            'fit_centers': tf.stack([h[2] for h in histories]),
            'fit_covariances': tf.stack([h[3] for h in histories]),
            'fit_log_floors': tf.stack([h[4] for h in histories]),
            'attempted_iterations': tf.constant(2), 'invalid_iteration': tf.constant(-1)}


def _assert_record(expected, actual, dtype_name, exact=False):
    maximum = 0.
    for left, right in zip(tf.nest.flatten(expected), tf.nest.flatten(actual), strict=True):
        left, right = np.asarray(left), np.asarray(right)
        if exact or left.dtype.kind in 'biu':
            np.testing.assert_array_equal(left, right)
        else:
            assert np.isfinite(left).all() and np.isfinite(right).all()
            error = float(np.max(np.abs(left-right)))
            maximum = max(maximum, error)
            tolerance = 2e-10 if dtype_name == 'float64' else 5e-5
            np.testing.assert_allclose(left, right, rtol=tolerance, atol=tolerance)
    return maximum


@pytest.mark.parametrize('dtype_name', ['float64', 'float32'])
@pytest.mark.parametrize('model', ['gaussian', 'nonlinear_scalar'])
def test_fitted_apf_complete_owner(model, dtype_name, request):
    dtype = tf.as_dtype(dtype_name)
    curves = FIXTURE['nonlinear_curves'] if model == 'nonlinear_scalar' else {}
    owner = make_fitted_twist_execution(1, 1, 16, 2, 2, 1., .01, dtype_name, True, **curves)
    theta = tf.constant(FIXTURE['theta'], dtype)
    observations = tf.constant(FIXTURE['observations'], dtype)
    rows = []
    for variant in (0, 1):
        seeds = _seeds(model, variant)
        current_theta = theta + tf.cast(variant * .002, dtype)
        fit_theta = theta - tf.cast(variant * .001, dtype)
        data = observations + tf.cast(variant * .003, dtype)
        args = (current_theta, fit_theta, data, seeds)
        expected = _reference_program(dtype_name, curves, *args)
        actual = owner(*args)
        row = {'variant': variant, 'reference': expected, 'candidate': actual}
        rows.append(row)
        _save(request, f'fitted-apf-owner-{model}-{dtype_name}', {'rows': rows})
        row['maximum_absolute_error'] = _assert_record(expected, actual, dtype_name)
        _assert_record(actual, owner(*args), dtype_name, exact=True)
        if dtype_name == 'float64':
            from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
            final = make_fitted_twist_kernel(1, 1, 16, 2, dtype_name, True, **curves)
            fixed = (data, *make_fitted_twist_inputs(1, 16, 2, dtype_name)(seeds[2]), *actual['fit'])
            errors = []
            for h in (1e-5, 5e-6):
                directions = np.eye(6) * h
                fd = [(float(final(current_theta + direction, *fixed)[0]) -
                       float(final(current_theta - direction, *fixed)[0])) / (2*h) for direction in directions]
                np.testing.assert_allclose(fd, actual['final'][1].numpy(), rtol=3e-5, atol=2e-7)
                errors.append(float(np.max(np.abs(np.asarray(fd)-actual['final'][1].numpy()))))
            row['fixed_fit_fd_errors'] = errors
    assert owner.experimental_get_tracing_count() == 1
    graph = _graph(request, f'fitted-apf-owner-{model}-{dtype_name}', owner, args)
    _save(request, f'fitted-apf-owner-{model}-{dtype_name}', {'rows': rows, 'graph': graph,
        'device': actual['final'][0].device, 'dtype': dtype_name,
        'nonclaim': 'Fixed-fit mechanics qualification; costs/adaptive iAPF remain open.'})


@pytest.mark.parametrize('model', ['gaussian', 'nonlinear_scalar'])
def test_fitted_apf_public_endpoint(model, request, monkeypatch):
    from bayesfilter.score_study import (
        adapters,
        fitted_twist_adapter,
        fitted_twist_execution_tf,
        fitted_twist_tf,
        gaussian_tf,
        nonlinear_adapter,
        nonlinear_tf,
    )
    from bayesfilter.score_study.contracts import digest
    from bayesfilter.score_study.registry import default_registry

    dtype = tf.float64
    device = 'CPU' if os.environ['CUDA_VISIBLE_DEVICES'] == '-1' else 'GPU'
    row = {'model': model, 'proposal': 'fitted_twist', 'dataset': 1, 'replicate': 0,
           'role': 'mechanics', 'comparison_target': 'finite_program_score',
           'comparison': 'approximation_error', 'fit_theta': FIXTURE['theta'],
           'estimator': 'analytical_filter' if model == 'gaussian' else 'nonlinear_analytical',
           **{k: FIXTURE[k] for k in ('fit_iterations', 'fit_initial_variance', 'fit_floor_ratio')}}
    settings = {k: FIXTURE[k] for k in ('dimension', 'observation_dimension', 'particles', 'horizon', 'theta')}
    settings.update(data_theta=FIXTURE['theta'], dtype='float64', device=device, jit_compile=True,
        tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
        reference={'points': 257, 'radius': 8., 'relative_tolerance': 1e-7, 'tail_tolerance': 1e-9})
    if model == 'nonlinear_scalar':
        settings.update(FIXTURE['nonlinear_curves'])
    context = {'study': {'seed': FIXTURE['seed'], 'settings': settings, 'evidence_class': 'mechanics'},
               'registry': default_registry()}
    adapter = adapters if model == 'gaussian' else nonlinear_adapter
    endpoint = adapter.evaluate_gaussian if model == 'gaussian' else adapter.evaluate_nonlinear
    data_module = gaussian_tf if model == 'gaussian' else nonlinear_tf
    monkeypatch.setattr(adapter, 'configure_runtime', lambda **_: {
        'device': device, 'jit_compile': True, 'tf32': settings['tf32'],
        'initialization': 'campaign_worker_already_configured', 'reference_exception': device == 'CPU'})
    # Freeze only the physical dataset. Every fitted stream uses its live label
    # and the actual public adapter. This is not data-generator qualification.
    monkeypatch.setattr(data_module, 'make_data_kernel', lambda *args:
        lambda *args: tf.constant(FIXTURE['observations'], dtype))
    original_tf = _reference_module('fitted_twist_tf')
    original_adapter = _reference_module('fitted_twist_adapter')
    with monkeypatch.context() as patch:
        patch.setattr(fitted_twist_adapter, 'execute_fitted_twist', original_adapter.execute_fitted_twist)
        patch.setattr(fitted_twist_tf, 'make_fitted_twist_kernel', original_tf.make_fitted_twist_kernel)
        patch.setattr(fitted_twist_tf, 'make_recursive_fit_kernel', original_tf.make_recursive_fit_kernel)
        expected = endpoint(row, context)
    original_factory = fitted_twist_execution_tf.make_fitted_twist_execution
    seen = []
    def observed(*args, **kwargs):
        owner = original_factory(*args, **kwargs)
        seen.append(owner)
        return owner
    monkeypatch.setattr(fitted_twist_execution_tf, 'make_fitted_twist_execution', observed)
    actual, replay = endpoint(row, context), endpoint(row, context)
    assert len(seen) == 2 and seen[0] is seen[1]
    assert seen[0].experimental_get_tracing_count() == 1
    assert actual['runtime']['kernel_calls'] == expected['runtime']['kernel_calls'] == 5
    assert actual['runtime']['traces'] == 1
    assert actual['diagnostics']['fit_enclosing_calls'] == 1
    assert actual['diagnostics']['fit_execution'] == 'enclosing_tensorflow_loop'
    for record in (expected, actual, replay):
        assert digest(record['diagnostics']['fit']) == record['diagnostics']['fit_digest']
    def numerical_record(record):
        result = json.loads(json.dumps(record))
        result.pop('runtime')
        for key in ('fit_execution', 'fit_enclosing_calls', 'fit_digest'):
            result['diagnostics'].pop(key, None)
        return result
    def compare(left, right):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                compare(left[key], right[key])
        elif isinstance(left, list):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                compare(a, b)
        elif isinstance(left, float):
            np.testing.assert_allclose(left, right, rtol=2e-10, atol=2e-10)
        else:
            assert left == right
    _save(request, f'fitted-apf-endpoint-{model}', {'reference': expected, 'candidate': actual, 'replay': replay})
    compare(numerical_record(expected), numerical_record(actual))
    assert numerical_record(actual) == numerical_record(replay)
    _save(request, f'fitted-apf-endpoint-{model}', {'reference': expected, 'candidate': actual, 'replay': replay,
        'passed': True, 'factory_identity_verified': True,
        'scope': 'Ordinary public endpoint; frozen physical data, actual fit/final RNG streams.'})


def test_fitted_apf_intermediate_failure(request, monkeypatch):
    from bayesfilter.score_study import fitted_twist_adapter, fitted_twist_execution_tf
    original = make_fitted_twist_inputs(1, 16, 2)
    table = _seeds('gaussian')
    rejected = table[1, 0, 0]

    @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.int32)], jit_compile=True)
    def inputs(seeds):
        initial, process, ancestors, mixture = original(seeds)
        # Identical particles make the real second QR fit rank deficient.
        failed = seeds[0, 0] == rejected
        return (tf.where(failed, tf.zeros_like(initial), initial),
                tf.where(failed, tf.zeros_like(process), process), ancestors, mixture)

    with monkeypatch.context() as patch:
        patch.setattr(fitted_twist_execution_tf, 'make_fitted_twist_inputs', lambda *args: inputs)
        owner = make_fitted_twist_execution.__wrapped__(1, 1, 16, 2, 3, 1., .01)
        extended = tf.concat([table[:2], table[:1], table[2:]], axis=0)
        theta = tf.constant(FIXTURE['theta'], tf.float64)
        observations = tf.constant(FIXTURE['observations'], tf.float64)
        result = owner(theta, theta, observations, extended)
        _save(request, 'fitted-apf-failure', result)
        assert int(result['invalid_iteration']) == 1
        assert int(result['attempted_iterations']) == 2
        assert np.isfinite(result['fit_log_values'].numpy()[:2]).all()
        assert np.isnan(result['fit_log_values'].numpy()[2])
        assert np.isnan(result['fit_centers'].numpy()[2]).all()
        assert all(np.isnan(x.numpy()).all() for x in result['final'])
        patch.setattr(fitted_twist_execution_tf, 'make_fitted_twist_execution', lambda *args, **kwargs: owner)
        # Keep the four-row table identical when exercising the host error.
        labels = {f'fit{i}_{name}': extended[i, j].numpy().tolist()
                  for i in range(3) for j, name in enumerate(('initial', 'process', 'ancestors', 'mixture'))}
        labels.update({f'final_twist_{name}': extended[3, j].numpy().tolist()
                       for j, name in enumerate(('initial', 'process', 'ancestors', 'mixture'))})
        row = {'model': 'gaussian', 'fit_theta': FIXTURE['theta'], 'fit_iterations': 3,
               'fit_initial_variance': 1., 'fit_floor_ratio': .01}
        settings = {'dimension': 1, 'observation_dimension': 1, 'particles': 16,
                    'horizon': 2, 'dtype': 'float64', 'jit_compile': True}
        with pytest.raises(ValueError, match='veto at iteration 1'):
            fitted_twist_adapter.execute_fitted_twist(row, settings, theta, observations,
                lambda label, **_: labels[label])
    _graph(request, 'fitted-apf-failure', owner, (theta, theta, observations, extended))
