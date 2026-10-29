"""Fresh reference checks of repaired helpers; no old campaign or HMC runs."""

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
BENCH = 'docs/benchmarks/'
DTYPE = tf.float64


def _helper(path, name, **dependencies):
    """Compile the exact function AST without executing a historical entry point."""
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    namespace = {'tf': tf, 'np': np, 'Any': Any, **dependencies}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), 'exec'), namespace)  # noqa: S102 -- exact repository helper AST
    return namespace[name]


def _save(request, name, report):
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory / (name + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    return directory


def _graph(owner, args, directory, name):
    assert owner.experimental_get_tracing_count() == 1
    graph = owner.get_concrete_function().graph.as_graph_def()
    nodes = [*graph.node, *(n for f in graph.library.function for n in f.node_def)]
    assert not {n.op for n in nodes} & {'PyFunc', 'EagerPyFunc', 'PyFuncStateless', 'XlaHostCompute'}
    assert not any('/pfor/' in n.name for n in nodes)
    hlo = owner.experimental_get_compiler_ir(*args)(stage='hlo')
    (directory / (name + '.hlo.txt')).write_text(hlo)
    return {'trace_count': 1, 'jit_compile': True, 'no_pfor_or_host_callback': True,
        'hlo_sha256': hashlib.sha256(hlo.encode()).hexdigest()}


def _fd(function, theta, step):
    columns = []
    for axis in np.eye(int(theta.shape[0])):
        direction = tf.constant(axis, theta.dtype) * step
        columns.append((-np.asarray(function(theta + 2 * direction))
            + 8 * np.asarray(function(theta + direction))
            - 8 * np.asarray(function(theta - direction))
            + np.asarray(function(theta - 2 * direction))) / (12 * step))
    return np.stack(columns, axis=-1)


@pytest.mark.parametrize('filename,function_name', [
    ('benchmark_identifiable_ssl_lstm_oracle_geometry_2026_07_08.py', 'dense_negative_hessian'),
    ('benchmark_minimal_ssl_lstm_zhaocui_hmc_tuning_phase5_2026_07_06.py', 'score_jacobian'),
])
def test_diagnostic_curvature_helpers(filename, function_name, request):
    precision = tf.constant([[2., .3], [.3, 1.5]], DTYPE)

    class Quadratic:
        def log_prob(self, theta):
            return -.5 * tf.einsum('i,ij,j->', theta, precision, theta)

        def log_prob_and_grad(self, theta):
            return self.log_prob(theta), -tf.linalg.matvec(precision, theta)

    function = _helper(BENCH + filename, function_name)
    records = []
    for point in ([.17, -.23], [-.31, .19]):
        actual = np.asarray(function(Quadratic(), tf.constant(point, DTYPE)))
        np.testing.assert_allclose(actual, precision, atol=1e-9, rtol=1e-9)
        records.append(actual.tolist())
    _save(request, 'pfor-helper-' + function_name, {'path': BENCH + filename,
        'scope': 'exact_helper_with_fresh_analytic_quadratic_no_tuning', 'records': records})


def test_generic_value_jacobian_helpers(request):
    jacobian = _helper(BENCH + 'contract_e_score_aware_teacher_projection_2d_lgssm.py', '_jacobian')
    prefix = _helper(BENCH + 'run_contract_e_tp_scalar_sv_prefix.py', '_result_value_score_and_increment_score')

    def values(x):
        return tf.stack([x[0] ** 2 + x[1], tf.sin(x[0]) - 2 * x[1] ** 2])

    def program(x):
        v = values(x)
        return {'objective': tf.reduce_sum(v), 'increment_history': v}

    records = []
    for point in ([.21, -.17], [-.11, .37]):
        theta = tf.constant(point, DTYPE)
        expected = tf.stack([[2 * theta[0], tf.constant(1., DTYPE)],
                             [tf.cos(theta[0]), -4 * theta[1]]])
        value, actual = jacobian(values, theta)
        result, score, increments = prefix(program, theta)
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(increments, expected, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(score, tf.reduce_sum(expected, 0), atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(result['increment_history'], value, atol=1e-9, rtol=1e-9)
        records.append({'point': point, 'jacobian': actual.numpy().tolist()})
    _save(request, 'pfor-helper-generic-jacobians', {'records': records,
        'scope': 'generic_wrappers_only_no_historical_LEDH'})


def test_generic_dense_moment_wrapper(request):
    def forward(observations, theta, *, order, radius):
        values = observations[:, 0] * theta[0] ** 2 + tf.sin(theta[1])
        return tf.reduce_sum(values), values

    function = _helper(BENCH + 'run_zhao_cui_moment_teacher_actual_sv.py', '_dense_arm',
        _dense_source_order_value=forward)
    observations = tf.constant([[.2], [-.3], [.4]], DTYPE)
    theta = tf.constant([.13, -.21], DTYPE)
    actual = function(observations, theta, order=8, radius=5.)
    expected = tf.stack([2 * theta[0] * observations[:, 0],
        tf.fill([3], tf.cos(theta[1]))], axis=-1)
    np.testing.assert_allclose(actual['increment_scores'], expected, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(actual['score'], tf.reduce_sum(expected, 0), atol=1e-9, rtol=1e-9)
    _save(request, 'pfor-helper-moment-wrapper', {'scope': 'exact_wrapper_generic_formula_only',
        'increment_scores': actual['increment_scores'].numpy().tolist()})


def test_generic_batch_jacobian_adapter(request):
    class AnalyticAdapter:
        def initial_value(self, theta, noise):
            return noise[None, :, :] * (1 + theta[:, None, :1]) + theta[:, None, 1:2]

        def initial_tangent(self, theta, noise):
            return tf.stack([noise, tf.ones_like(noise)], axis=-1)[None, ...]

        def transition_value(self, theta, initial, noise, time):
            return initial**2 + theta[:, None, :1] + noise[None, :, :]

        def transition_tangent(self, theta, initial, noise, tangent, time):
            return 2 * initial[..., None] * tangent + tf.constant([1., 0.])

        def observation_value(self, theta, transitioned, observation, time):
            return transitioned + theta[:, None, 1:2]**2

        def observation_tangent(self, theta, transitioned, tangent, observation, time):
            return tangent + tf.stack([tf.zeros_like(theta[:, 1]), 2 * theta[:, 1]], -1)[:, None, None, :]

    def compare(manual, reference):
        np.testing.assert_allclose(manual, reference, atol=1e-9, rtol=1e-9)
        return {'shape': manual.shape.as_list(), 'max_error': float(tf.reduce_max(tf.abs(manual-reference)))}

    function = _helper(BENCH + 'run_genut_austria_endpoint_root_cause_20260817.py',
        '_local_autodiff_checks', _comparison=compare)
    noise = tf.reshape(tf.linspace(-.3, .3, 72), [36, 2])
    target = SimpleNamespace(filter_adapter=AnalyticAdapter(), parameter_dim=2,
        initial_noise=noise, process_noise=noise[None, ...] * .1, observations=tf.zeros([1, 2]))
    actual = function(target)
    _save(request, 'pfor-helper-batch-jacobian', {'scope': 'exact_helper_fresh_analytic_adapter_not_Austria_target',
        'actual': actual})


def test_reference_scalar_map(request):
    function = _helper(BENCH + 'benchmark_minimal_ssl_lstm_zhaocui_hmc_oracle_2026_07_06.py',
        'target_log_prob_batch')
    adapter = SimpleNamespace(_scalar_log_prob_and_grad=lambda x: (-tf.reduce_sum(x*x), -2*x))
    points = np.array([[.2, -.1], [.5, .3], [-.2, .4]])
    actual = function(adapter, points)
    np.testing.assert_allclose(actual, -np.sum(points**2, axis=1), atol=1e-9, rtol=1e-9)
    _save(request, 'pfor-helper-reference-map', {'values': actual.tolist(),
        'scope': 'scalar_reference_only_not_batch_native_training'})


@pytest.mark.parametrize('filename', [
    'run_cubature_exact_sv_score_ladder.py', 'run_exact_sv_fixed_gaussian_genut_paired.py',
])
def test_dense_sv_reference_derivative(filename, request):
    function = _helper(BENCH + filename, '_dense_reference')
    theta = tf.constant([.17, -.11], DTYPE)
    observations = tf.constant([[.19], [-.31], [.07]], DTYPE)

    def evaluate(x):
        return function(observations, x, order=16, radius=5.)

    actual = evaluate(theta)
    errors = []
    for step in (1e-3, 5e-4):
        expected = _fd(lambda x: evaluate(x)['value_increments'], theta, step)
        np.testing.assert_allclose(actual['score_increments'], expected, atol=2e-6, rtol=2e-6)
        errors.append(float(np.max(np.abs(np.asarray(actual['score_increments']) - expected))))
    np.testing.assert_allclose(np.sum(actual['score_increments'], axis=0), actual['score'],
        atol=1e-9, rtol=1e-9)
    _save(request, 'pfor-helper-' + filename.removesuffix('.py'), {'actual': actual,
        'fd_errors': errors, 'scope': 'fresh_dense_reference_derivatives_only_not_filter_accuracy'})


@pytest.mark.parametrize('arm', ['jit_check', 'benchmark', 'mapped_target'])
def test_p91_enclosing_xla(arm, request):
    import bayesfilter.highdim.models as model
    from scripts import p91_gpu_xla_jit_check as jit_check
    from scripts import p91_hmc_smoke as smoke
    from scripts import p91_performance_benchmark as benchmark

    rng = np.random.default_rng(81129)
    initial = model._zhao_cui_sir_austria_initial_mean_xla()
    states = tf.constant(initial.numpy()[None, None, :] + rng.normal(size=(4, 5, 18)) * .04, DTYPE)
    observations = states[:, :, 1::2] + tf.constant(rng.normal(size=(4, 5, 9)) * .1, DTYPE)
    theta = tf.constant([.013, -.021, .032], DTYPE)
    if arm == 'jit_check':
        owner = jit_check._batched_value_and_score_compiled()
    elif arm == 'benchmark':
        owner = benchmark._batched_compiled(jit_compile=True)
    else:
        target = smoke._make_target_log_prob(states[0], observations[0])

        @tf.function(input_signature=[tf.TensorSpec([4, 3], DTYPE)], jit_compile=True, autograph=False)
        def owner(parameters):
            with tf.GradientTape() as tape:
                tape.watch(parameters)
                values = target(parameters)
                total = tf.reduce_sum(values)
            return values, tape.gradient(total, parameters)

    records = []
    for shift in (0., .004):
        parameters = theta + shift
        if arm == 'mapped_target':
            parameters = parameters[None, :] + tf.constant(rng.normal(size=(4, 3)) * .002, DTYPE)
            args = (parameters,)
            scalar_values, scalar_scores = [], []
            for row in parameters:
                with tf.GradientTape() as tape:
                    tape.watch(row)
                    v = model.zhao_cui_sir_austria_local_complete_data_log_density_xla(row, states[0], observations[0])
                scalar_values.append(v)
                scalar_scores.append(tape.gradient(v, row))
            expected = (tf.stack(scalar_values), tf.stack(scalar_scores))
        else:
            args = (parameters, states + shift, observations - shift)
            scalar_values, scalar_scores = [], []
            for index in range(4):
                with tf.GradientTape() as tape:
                    tape.watch(parameters)
                    v = model.zhao_cui_sir_austria_local_complete_data_log_density_xla(
                        parameters, args[1][index], args[2][index])
                scalar_values.append(v)
                scalar_scores.append(tape.gradient(v, parameters))
            expected = (tf.stack(scalar_values), tf.stack(scalar_scores))
        actual = owner(*args)
        np.testing.assert_allclose(actual[0], expected[0], atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(actual[1], expected[1], atol=1e-9, rtol=1e-9)
        errors = []
        for step in (1e-3, 5e-4):
            if arm == 'mapped_target':
                fd = _fd(lambda x: model.zhao_cui_sir_austria_local_complete_data_log_density_xla(
                    x, states[0], observations[0]), parameters[0], step)
                compared = actual[1][0]
            else:
                fd = _fd(lambda x, inputs=args[1:]: owner(x, *inputs)[0], parameters, step)
                compared = actual[1]
            np.testing.assert_allclose(compared, fd, atol=2e-6, rtol=2e-6)
            errors.append(float(np.max(np.abs(np.asarray(compared) - fd))))
        records.append({'shift': shift, 'values': actual[0].numpy().tolist(),
            'scores': actual[1].numpy().tolist(), 'fd_errors': errors,
            'devices': [v.device for v in actual]})
    name = 'pfor-p91-' + arm
    directory = _save(request, name, {'seed': 81129, 'records': records})
    graph = _graph(owner, args, directory, name)
    _save(request, name, {'seed': 81129, 'records': records, 'graph': graph,
        'scope': 'current_complete_data_diagnostic_no_sampler_training_or_source_faithfulness_claim'})


def _kalman_runner():
    name = '_pfor_repaired_kalman_runner'
    path = ROOT / BENCH / 'run_kalman_qr_cpu_xla_formulation_shootout_2026_07_15.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p91_derivative_graph_localization(request):
    """Export the failing generated graph before choosing another repair."""
    from bayesfilter.testing.p91_complete_data_diagnostics_tf import (
        complete_data_values_and_scores_tf,
    )

    signature = [tf.TensorSpec([3], DTYPE), tf.TensorSpec([4, 5, 18], DTYPE),
        tf.TensorSpec([4, 5, 9], DTYPE)]
    owner = tf.function(complete_data_values_and_scores_tf, input_signature=signature,
        jit_compile=True, autograph=False)
    graph = owner.get_concrete_function().graph.as_graph_def()
    definitions = {f.signature.name for f in graph.library.function}
    loops = []
    for scope, nodes in [('<root>', graph.node), *((f.signature.name, f.node_def)
                                                   for f in graph.library.function)]:
        for node in nodes:
            if node.op in ('While', 'StatelessWhile'):
                refs = {name: node.attr[name].func.name for name in ('body', 'cond')}
                loops.append({'scope': scope, 'node': node.name, 'op': node.op,
                    'references': refs, 'all_defined_before_xla': all(x in definitions for x in refs.values())})
    directory = _save(request, 'pfor-p91-graph-localization', {'loops': loops,
        'function_count': len(definitions), 'scope': 'traced_graph_before_failed_compiler_rewrite'})
    (directory / 'pfor-p91-failing-graph.pbtxt').write_text(str(graph))
    assert loops and all(row['all_defined_before_xla'] for row in loops)


@pytest.mark.parametrize('mode', ['vectorized_strict', 'vectorized_fallback', 'unknown'])
def test_kalman_rejected_modes_before_numerics(mode, tmp_path):
    runner = _kalman_runner()
    for call in (lambda: runner._build_formulation(None, None, mode, 2),
                 lambda: runner._worker(SimpleNamespace(formulation=mode)),
                 lambda: runner.worker_command(mode, dimension=2, parameter_count=2,
                     timesteps=2, batch_size=2, record_path=tmp_path / 'unused.json')):
        with pytest.raises(ValueError, match='retired|unknown formulation'):
            call()
    with pytest.raises(SystemExit):
        runner.parse_args(['--worker', '--formulation', mode])


def test_invalidated_p44_retires_before_framework_import(request):
    path = 'experiments/dpf_implementation/tf_tfp/runners/run_ledh_pfpf_source_faithful_repair_tf.py'
    code = '''
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'tensorflow', 'numpy', 'bayesfilter', 'experiments'}:
            raise AssertionError('framework imported before retirement: ' + fullname)
sys.meta_path.insert(0, Block())
try:
    runpy.run_path(sys.argv[1], run_name='__main__')
except RuntimeError as error:
    assert 'HISTORICAL_LEDH_P44_RETIRED' in str(error)
    print(str(error))
else:
    raise AssertionError('retired benchmark executed')
'''
    result = subprocess.run([sys.executable, '-c', code, str(ROOT / path)],
        capture_output=True, text=True, check=True, timeout=10)
    _save(request, 'pfor-retired-p44', {'path': path, 'stdout': result.stdout,
        'framework_import_blocked': True})
