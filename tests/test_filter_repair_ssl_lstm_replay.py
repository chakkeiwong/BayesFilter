"""Pinned-original and independent diagnostics for the fixed replay XLA repair."""

import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.nonlinear import ssl_lstm_zhaocui_fixed_adapter as candidate
from bayesfilter.nonlinear.ssl_lstm_protocol import SSLLSTMStaticConfig

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'c4950a827'
SOURCE = 'bayesfilter/nonlinear/ssl_lstm_zhaocui_fixed_adapter.py'
DEPENDENCIES = (
    'bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py',
    'bayesfilter/nonlinear/ssl_lstm_protocol.py',
    'bayesfilter/ops/compiled_tensor_program_tf.py',
    'bayesfilter/ops/stateless_random_tf.py',
)
EVIDENCE = 'docs/plans/filter_gradient_ssl_lstm_replay_execution_20260929.md'


@pytest.fixture(scope='module', autouse=True)
def precision():
    tf.config.experimental.enable_tensor_float_32_execution(False)


def reference():
    for path in DEPENDENCIES:
        assert subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'], cwd=ROOT) == (ROOT/path).read_bytes()
    source = subprocess.check_output(['git', 'show', f'{BASELINE}:{SOURCE}'], cwd=ROOT)
    name = 'bayesfilter.nonlinear._fixed_replay_pinned_diagnostic'
    module = types.ModuleType(name)
    module.__package__ = 'bayesfilter.nonlinear'
    sys.modules[name] = module
    exec(compile(source, f'{BASELINE}:{SOURCE}', 'exec'), module.__dict__)  # noqa: S102
    return module, hashlib.sha256(source).hexdigest()


def case(horizon, latent=1):
    config = SSLLSTMStaticConfig(horizon=horizon, latent_dim=latent,
                                hidden_dim=1, observation_dim=latent)
    theta = tf.linspace(tf.constant(-.18, tf.float64), tf.constant(.23, tf.float64),
                        config.parameter_dim)
    observations = tf.reshape(tf.linspace(tf.constant(-.12, tf.float64),
        tf.constant(.17, tf.float64), horizon*latent), [horizon, latent])
    manifest = candidate.SSLLSTMZhaoCuiFixedManifest(reference_sample_count=9,
        initial_seed=(20260705, 41), process_seed=(20260705, 43))
    return config, theta, observations, manifest


def tensor_record(result, components):
    return {
        'result': {name: getattr(result, name) for name in candidate._RESULT_TENSOR_FIELDS},
        'diagnostics': {name: result.diagnostics[name] for name in candidate._NUMERICAL_DIAGNOSTIC_FIELDS},
        'parameters': {field.name: getattr(components.parameters, field.name)
            for field in dataclasses.fields(components.parameters)
            if field.name not in ('config', 'slices', 'std_floor')},
    }


def evaluate(module, config, theta, observations, manifest, **kwargs):
    result, components = module.tf_ssl_lstm_zhaocui_fixed_score(
        observations, theta, config, evidence_path=EVIDENCE, manifest=manifest, **kwargs)
    return tensor_record(result, components)


def host(value):
    if tf.is_tensor(value):
        return value.numpy().tolist()
    if isinstance(value, dict):
        return {key: host(part) for key, part in value.items()}
    if isinstance(value, (list, tuple)):
        return [host(part) for part in value]
    return value


def save(request, name, record):
    output = Path(request.config.getoption('xmlpath')).parent/name
    record = {'baseline': BASELINE, 'source_sha256': {
        p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (SOURCE, *DEPENDENCIES)},
        'device': 'CPU' if os.environ['CUDA_VISIBLE_DEVICES'] == '-1' else 'GPU',
        'tf32_enabled': tf.config.experimental.tensor_float_32_execution_enabled(),
        **record}
    output.write_text(json.dumps(host(record), indent=2, allow_nan=True)+'\n')


def compare(actual, expected):
    assert tf.nest.map_structure(lambda _: None, actual) == tf.nest.map_structure(lambda _: None, expected)
    for a, b in zip(tf.nest.flatten(actual), tf.nest.flatten(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('horizon,latent', [(1, 1), (5, 1), (5, 2)])
def test_complete_public_records_and_streams(horizon, latent, request):
    original, digest = reference()
    config, theta, observations, manifest = case(horizon, latent)
    expected = evaluate(original, config, theta, observations, manifest)
    actual = evaluate(candidate, config, theta, observations, manifest)
    graph_reference = evaluate(candidate, config, theta, observations, manifest, jit_compile=False)
    save(request, f'public-{horizon}-{latent}.json', {'baseline_sha256': digest,
        'expected': expected, 'candidate': actual, 'candidate_graph_reference': graph_reference})
    compare(actual, expected)
    compare(graph_reference, expected)
    repeated = evaluate(candidate, config, theta, observations, manifest)
    for a, b in zip(tf.nest.flatten(actual), tf.nest.flatten(repeated), strict=True):
        np.testing.assert_array_equal(a, b)
    changed = evaluate(candidate, config, theta+.015, observations-.013, manifest)
    compare(changed, evaluate(original, config, theta+.015, observations-.013, manifest))
    assert not np.array_equal(actual['result']['score'], changed['result']['score'])
    owner = candidate._fixed_replay_program(config, manifest, horizon, 1e-4, True, False)
    assert owner.experimental_get_tracing_count() == 1
    assert owner.get_concrete_function().function_def.attr['_XlaMustCompile'].b


@pytest.mark.parametrize('jit', [False, True])
def test_enclosing_context_preserves_original_stream(jit, request):
    original, digest = reference()
    config, theta, observations, manifest = case(5)
    signature = [tf.TensorSpec(theta.shape, theta.dtype), tf.TensorSpec(observations.shape, observations.dtype)]
    before = tf.function(lambda q, y: evaluate(original, config, q, y, manifest),
        input_signature=signature, jit_compile=jit, autograph=False)
    after = tf.function(lambda q, y: evaluate(candidate, config, q, y, manifest),
        input_signature=signature, jit_compile=jit, autograph=False)
    expected, actual = before(theta, observations), after(theta, observations)
    save(request, f'enclosing-{jit}.json', {'jit_compile': jit,
        'baseline_sha256': digest, 'expected': expected, 'candidate': actual})
    compare(actual, expected)
    compare(after(theta+.03, observations-.01), before(theta+.03, observations-.01))
    assert after.experimental_get_tracing_count() == 1


def test_native_graph_and_independent_derivatives(request):
    config, theta, observations, manifest = case(2)
    value, _ = candidate.tf_ssl_lstm_zhaocui_fixed_score(observations, theta, config,
        evidence_path=EVIDENCE, manifest=manifest)
    direction = tf.math.l2_normalize(tf.linspace(tf.constant(-1., tf.float64),
        tf.constant(1., tf.float64), config.parameter_dim))
    step = 1e-4
    values = []
    for multiple in (-2, -1, 1, 2):
        result, _ = candidate.tf_ssl_lstm_zhaocui_fixed_score(observations,
            theta+multiple*step*direction, config, evidence_path=EVIDENCE, manifest=manifest)
        values.append(result.log_likelihood)
    finite_difference = (values[0]-8*values[1]+8*values[2]-values[3])/(12*step)
    analytical = tf.reduce_sum(value.score*direction)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        result, _ = candidate.tf_ssl_lstm_zhaocui_fixed_score(observations, theta, config,
            evidence_path=EVIDENCE, manifest=manifest)
    diagnostic_autodiff = tape.gradient(result.log_likelihood, theta)
    owner = candidate._fixed_replay_program(config, manifest, 2, 1e-4, True, False)
    graph = owner.get_concrete_function().graph.as_graph_def()
    nodes = [*graph.node, *(node for function in graph.library.function for node in function.node_def)]
    operations = {node.op for node in nodes}
    hlo = owner.experimental_get_compiler_ir(observations, theta)(stage='hlo')
    save(request, 'derivative-graph.json', {'analytical_direction': analytical,
        'finite_difference_direction': finite_difference, 'score': value.score,
        'diagnostic_autodiff_score': diagnostic_autodiff, 'graph_node_count': len(nodes),
        'operations': sorted(operations), 'trace_count': owner.experimental_get_tracing_count(),
        'hlo_sha256': hashlib.sha256(hlo.encode()).hexdigest(),
        'scope': 'Analytical score remains runtime authority; tape and finite differences are independent diagnostics.'})
    np.testing.assert_allclose(analytical, finite_difference, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(value.score, diagnostic_autodiff, rtol=1e-10, atol=1e-10)
    assert operations & {'While', 'StatelessWhile'}
    assert not operations & {'PyFunc', 'EagerPyFunc', 'PyFuncStateless'}
    assert 'HloModule' in hlo
    assert owner.experimental_get_tracing_count() == 1
    assert candidate._fixed_replay_program.cache_info().maxsize == 16


def test_original_error_boundaries():
    original, _ = reference()
    config, theta, observations, manifest = case(2)
    cases = [(theta[None, :], observations), (theta[:-1], observations),
        (theta, observations[:, 0]), (theta, observations[:0]),
        (theta, tf.concat([observations, observations], 0)),
        (theta, tf.concat([observations, observations], 1))]
    for q, y in cases:
        with pytest.raises(ValueError) as expected:
            evaluate(original, config, q, y, manifest)
        with pytest.raises(ValueError) as actual:
            evaluate(candidate, config, q, y, manifest)
        assert str(actual.value) == str(expected.value)


def test_original_manifest_coercions_and_metadata(request):
    original, digest = reference()
    config, theta, observations, manifest = case(2)
    manifest = dataclasses.replace(manifest, reference_sample_count=9.0,
        initial_seed=list(manifest.initial_seed), process_seed=list(manifest.process_seed),
        recenter_ridge='0.00001', nonclaims=['custom diagnostic metadata'])
    expected = evaluate(original, config, theta, observations, manifest)
    save(request, 'manifest-reference.json', {'baseline_sha256': digest,
        'manifest': manifest.as_dict(), 'expected': expected})
    result, components = candidate.tf_ssl_lstm_zhaocui_fixed_score(observations, theta, config,
        evidence_path=EVIDENCE, manifest=manifest)
    actual = tensor_record(result, components)
    save(request, 'manifest-candidate.json', {'candidate': actual,
        'manifest': result.diagnostics['manifest']})
    compare(actual, expected)
    assert components.manifest is manifest
    assert result.diagnostics['manifest'] == manifest.as_dict()
    owner_count = candidate._fixed_replay_program.cache_info().currsize
    metadata_only = dataclasses.replace(manifest, nonclaims=['other diagnostic metadata'])
    other, other_components = candidate.tf_ssl_lstm_zhaocui_fixed_score(
        observations, theta, config, evidence_path=EVIDENCE, manifest=metadata_only)
    compare(tensor_record(other, other_components), expected)
    assert candidate._fixed_replay_program.cache_info().currsize == owner_count
    assert other_components.manifest is metadata_only
    assert other.diagnostics['manifest'] == metadata_only.as_dict()
    # A populated integer-seed cache must not accept numerically equal floats.
    invalid = dataclasses.replace(manifest, initial_seed=[float(v) for v in manifest.initial_seed])
    with pytest.raises(TypeError):
        evaluate(original, config, theta, observations, invalid)
    with pytest.raises(TypeError):
        evaluate(candidate, config, theta, observations, invalid)


def test_actual_benchmark_reports_executed_mode(request):
    path = ROOT/'docs/benchmarks/benchmark_ssl_lstm_filter_hmc_phase6.py'
    spec = importlib.util.spec_from_file_location('_fixed_replay_benchmark_diagnostic', path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    config, theta, observations, manifest = case(2)
    rows = []
    for jit in (True, False):
        result, components = candidate.tf_ssl_lstm_zhaocui_fixed_score(
            observations, theta, config, evidence_path=EVIDENCE,
            manifest=manifest, jit_compile=jit)
        row = benchmark._candidate_row(filter_name='zhaocui_fixed', status='admitted',
            protocol=components.protocol, score_result=result,
            train_log_likelihood=result.log_likelihood, full_log_likelihood=result.log_likelihood,
            decoded_means=result.filtered_means[:, :1], truth_state_path=observations,
            heldout_obs=observations, predicted_obs=observations, fd_error=1e-12,
            artifact_builder=candidate.build_ssl_lstm_zhaocui_fixed_value_score_artifact,
            manifest=manifest)
        rows.append(row)
        assert row['artifact']['jit_compile'] == jit
        assert row['artifact']['compile_mode'] == ('xla' if jit else 'graph')
        assert row['artifact_role'] == ('target' if jit else 'debug_reference')
        assert row['status'] == ('admitted' if jit else 'debug_reference')
        assert row['artifact']['device'] == result.log_likelihood.device
    save(request, 'benchmark-modes.json', {'rows': rows,
        'benchmark_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
