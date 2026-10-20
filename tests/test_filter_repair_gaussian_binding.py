"""Original-source and independent derivative diagnostics for Gaussian binding."""

import hashlib
import inspect
import json
import os
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.score_study.gaussian_execution_tf import make_gaussian_execution
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'ea5319db9'
THETA = [.55, -.45, -.3, 1.05, .2, -.4]
DEPENDENCIES = ('bayesfilter/score_study/gaussian_execution_tf.py',
                'bayesfilter/score_study/gaussian_tf.py',
                'bayesfilter/score_study/adapters.py')


def directory(request):
    return Path(request.config.getoption('xmlpath')).parent


def host(value):
    if tf.is_tensor(value):
        return value.numpy().tolist()
    if isinstance(value, (tuple, list)):
        return [host(part) for part in value]
    if isinstance(value, dict):
        return {key: host(part) for key, part in value.items()}
    return value


def save(request, name, record):
    (directory(request) / name).write_text(json.dumps(host(record), indent=2, allow_nan=False) + '\n')


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in DEPENDENCIES}


def compare(actual, expected, tolerance):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        return max((compare(actual[k], expected[k], tolerance) for k in actual), default=0.)
    if isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        return max((compare(a, b, tolerance) for a, b in zip(actual, expected, strict=True)), default=0.)
    if isinstance(actual, (str, bool)) or actual is None:
        assert type(actual) is type(expected) and actual == expected
        return 0.
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)
    return float(np.max(np.abs(a-b), initial=0.))


def graph(request, owner, operands, label):
    concrete = owner.get_concrete_function()
    assert concrete.function_def.attr['_XlaMustCompile'].b
    definition = concrete.graph.as_graph_def()
    nodes = [*definition.node, *(node for function in definition.library.function for node in function.node_def)]
    assert not any('/pfor/' in node.name for node in nodes)
    assert not {node.op for node in nodes} & {'PyFunc', 'EagerPyFunc', 'PyFuncStateless', 'XlaHostCompute'}
    assert any(node.op in ('While', 'StatelessWhile') for node in nodes)
    # Model parameter transforms must be in the outer graph, before its call.
    assert 'Exp' in {node.op for node in definition.node}
    hlo = owner.experimental_get_compiler_ir(*operands)(stage='hlo')
    (directory(request) / f'{label}.hlo.txt').write_text(hlo)
    return {'jit_compile': True, 'trace_count': owner.experimental_get_tracing_count(),
            'host_callbacks': False, 'pfor': False, 'model_transforms_enclosed': True,
            'hlo_sha256': hashlib.sha256(hlo.encode()).hexdigest()}


@pytest.mark.parametrize('dtype_name', ['float64', 'float32'])
def test_complete_model_binding(dtype_name, request):
    checkpoint = FrozenCheckpoint(BASELINE, 'gaussian_binding_reference')
    original = checkpoint.load('bayesfilter.score_study.gaussian_tf')
    assert checkpoint.hashes()['bayesfilter/score_study/gaussian_tf.py'] == hashes()[
        'bayesfilter/score_study/gaussian_tf.py']
    assert inspect.signature(make_gaussian_execution).parameters['jit_compile'].default is True
    dtype = tf.as_dtype(dtype_name)
    tolerance = 1e-9 if dtype_name == 'float64' else 1e-5
    cases = []
    for d, o in ((1, 1), (3, 2)):
        observations = tf.reshape(tf.linspace(tf.cast(-.3, dtype), tf.cast(.4, dtype), 4*o), [4, o])
        theta = tf.constant(THETA, dtype)
        for unscented in (False, True):
            owner = make_gaussian_execution(d, o, dtype_name, unscented=unscented)
            reference = original.make_gaussian_kernel(d, o, 6, dtype_name, True, unscented)
            outputs, errors, fd_rows = [], [], []
            for variant in (0., .07):
                point, data = theta + tf.cast(variant, dtype), observations - tf.cast(variant, dtype)
                expected = reference(data, *original.parameterized_model(point, d, o))
                actual = owner(point, data)
                if os.environ['CUDA_VISIBLE_DEVICES'] != '-1':
                    assert all('GPU:0' in part.device for part in actual)
                errors.append(compare(host(actual), host(expected), tolerance))
                assert float(actual[4]) > 0
                replay = owner(point, data)
                assert all(a.numpy().tobytes() == b.numpy().tobytes() for a, b in zip(actual, replay, strict=True))
                outputs.append({'theta': point, 'observations': data, 'original': expected, 'candidate': actual})
                if dtype_name == 'float64':
                    for step in (.002, .001):
                        derivatives = []
                        for direction in tf.unstack(tf.eye(6, dtype=dtype)):
                            minus2 = owner(point - 2*step*direction, data)[0]
                            minus1 = owner(point - step*direction, data)[0]
                            plus1 = owner(point + step*direction, data)[0]
                            plus2 = owner(point + 2*step*direction, data)[0]
                            derivatives.append((minus2 - 8*minus1 + 8*plus1 - plus2)/(12*step))
                        derivative = tf.stack(derivatives)
                        error = compare(host(actual[1]), host(derivative), 1e-7)
                        fd_rows.append({'variant': variant, 'step': step, 'score': actual[1],
                                        'finite_difference': derivative, 'maximum_error': error})
            assert owner.experimental_get_tracing_count() == 1
            assert make_gaussian_execution(d, o, dtype_name, unscented=unscented) is owner
            # Invalid model values remain observable to the existing public veto.
            invalid = owner(tf.constant([float('nan'), *THETA[1:]], dtype), observations)
            assert not bool(tf.reduce_all(tf.math.is_finite(invalid[1])))
            cases.append({'dimension': d, 'observations': o, 'unscented': unscented,
                'errors': errors, 'outputs': outputs, 'finite_differences': fd_rows,
                'nonfinite_score_preserved': True,
                'graph': graph(request, owner, (theta, observations), f'gaussian-{d}-{o}-{unscented}-{dtype_name}')})
    if dtype_name == 'float64' and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
        graph_reference = make_gaussian_execution(1, 1, dtype_name, jit_compile=False)
        theta, data = tf.constant(THETA, dtype), tf.constant([[.1], [.3]], dtype)
        compare(host(graph_reference(theta, data)), host(make_gaussian_execution(1, 1)(theta, data)), tolerance)
        assert not graph_reference.function_spec.jit_compile
    save(request, f'gaussian-binding-{dtype_name}.json', {
        'schema': 'filter_gaussian_binding.v1', 'dtype': dtype_name, 'baseline': BASELINE,
        'device': 'CPU' if os.environ['CUDA_VISIBLE_DEVICES'] == '-1' else 'GPU',
        'cpu_reference_exception': os.environ['CUDA_VISIBLE_DEVICES'] == '-1',
        'source_sha256': hashes(), 'baseline_source_sha256': checkpoint.hashes(), 'cases': cases,
        'last_output_devices': [part.device for part in actual]})


def original_adapter():
    name = 'bayesfilter/score_study/adapters.py'
    source = subprocess.check_output(['git', 'show', f'{BASELINE}:{name}'], cwd=ROOT, text=True)
    module = types.ModuleType('bayesfilter.score_study._gaussian_binding_diagnostic_reference')
    module.__package__ = 'bayesfilter.score_study'
    exec(compile(source, f'{BASELINE}:{name}', 'exec'), module.__dict__)  # noqa: S102 - exact Git-pinned diagnostic source
    baseline_model = subprocess.check_output(['git', 'show', f'{BASELINE}:bayesfilter/score_study/gaussian_tf.py'], cwd=ROOT)
    assert baseline_model == (ROOT / 'bayesfilter/score_study/gaussian_tf.py').read_bytes()
    return module, hashlib.sha256(source.encode()).hexdigest()


def endpoint_case(proposal, variant=0.):
    from bayesfilter.score_study.registry import default_registry
    device = 'CPU' if os.environ['CUDA_VISIBLE_DEVICES'] == '-1' else 'GPU'
    settings = {'dimension': 3, 'observation_dimension': 2, 'horizon': 4, 'particles': 16,
        'theta': [value + variant for value in THETA], 'data_theta': THETA,
        'device': device, 'dtype': 'float64', 'jit_compile': True,
        'tf32': tf.config.experimental.tensor_float_32_execution_enabled()}
    row = {'model': 'gaussian', 'proposal': proposal, 'dataset': 1, 'replicate': 0,
        'role': 'mechanics', 'comparison_target': 'finite_program_score',
        'comparison': 'approximation_error', 'estimator': 'analytical_filter'}
    context = {'study': {'seed': 20260929, 'settings': settings, 'evidence_class': 'mechanics'},
               'registry': default_registry()}
    return row, context


def runtime_setup(**settings):
    return {**settings, 'initialization': 'trusted_campaign_worker_already_configured',
            'reference_exception': settings['device'] == 'CPU'}


def numerical_record(result):
    return {key: value for key, value in result.items() if key != 'runtime'}


def test_public_gaussian_binding(request, monkeypatch):
    from bayesfilter.score_study import adapters, gaussian_execution_tf
    original, source_digest = original_adapter()
    for module in (original, adapters):
        monkeypatch.setattr(module, 'configure_runtime', runtime_setup)
    factory = gaussian_execution_tf.make_gaussian_execution
    seen = []

    def observed(*args, **kwargs):
        owner = factory(*args, **kwargs)
        seen.append(owner)
        return owner

    monkeypatch.setattr(gaussian_execution_tf, 'make_gaussian_execution', observed)
    cases = []
    for proposal in ('kalman', 'ukf'):
        for variant in (0., .07):
            row, context = endpoint_case(proposal, variant)
            expected = original.evaluate_gaussian(row, context)
            actual = adapters.evaluate_gaussian(row, context)
            replay = adapters.evaluate_gaussian(row, context)
            error = compare(numerical_record(actual), numerical_record(expected), 1e-9)
            assert numerical_record(actual) == numerical_record(replay)
            assert actual['runtime']['traces'] == 1 and actual['runtime']['jit_compile']
            cases.append({'proposal': proposal, 'variant': variant, 'maximum_error': error,
                          'original': expected, 'candidate': actual})
        # Preserve the actual finite-score host veto; do not accept invalid output.
        row, context = endpoint_case(proposal)
        context['study']['settings']['theta'][0] = float('nan')
        errors = []
        for endpoint in (original.evaluate_gaussian, adapters.evaluate_gaussian):
            with pytest.raises((ValueError, tf.errors.InvalidArgumentError)) as error:
                endpoint(row, context)
            errors.append(type(error.value).__name__)
        assert errors[0] == errors[1]
    assert seen and all(owner.experimental_get_tracing_count() == 1 for owner in seen)
    save(request, 'gaussian-binding-public.json', {'schema': 'filter_gaussian_binding_public.v1',
        'baseline': BASELINE, 'baseline_adapter_sha256': source_digest,
        'source_sha256': hashes(), 'cases': cases, 'live_dataset_seeds': True,
        'preserved_public_invalidity': True, 'single_trace': True,
        'deferred_iapf_not_executed': True})
