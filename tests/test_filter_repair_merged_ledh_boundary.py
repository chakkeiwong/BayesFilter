"""Diagnostic callback parity and training-boundary checks for merged LEDH code."""

import ast
import hashlib
import inspect
import json
import math
import subprocess
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_canonical_neutra_targets_tf as targets
from bayesfilter.highdim.ledh_alg1_contract import ENTRY_POINTS
from bayesfilter.inference.neutra_batching import (
    InvalidNeuTraBatchTarget,
    require_batch_native_neutra_target,
)

BASELINE = '9d8202b779888f6150d219bc971df832cfe9c887'
DTYPE = tf.float64


def test_row_mapped_target_is_rejected_by_public_training_binder(monkeypatch):
    target = targets.make_canonical_neutra_target('lgssm', particle_count=6, substeps=1)
    assert not target.neutra_training_eligible
    assert target.batching_policy_id == 'row_mapped_scalar_target_diagnostic_only_v1'
    signature = target.target_signature()
    assert signature == target.target_signature()

    def forbidden_evaluation(*args, **kwargs):
        raise AssertionError('ineligible target reached numerical evaluation')

    monkeypatch.setattr(targets.CanonicalNeuTraTarget, 'batch_value_score', forbidden_evaluation)
    with pytest.raises(InvalidNeuTraBatchTarget, match='requires bound method'):
        require_batch_native_neutra_target(target, target_signature=signature, batch_size=2)
    entries = {entry.lane: entry for entry in ENTRY_POINTS}
    assert 'ineligible for NeuTra training' in entries['batch_fused'].notes
    assert 'no training admission' in entries['neutra_target'].notes


def test_target_builder_has_no_numpy_import():
    source = inspect.getsource(targets)
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            assert all(alias.name.split('.')[0] != 'numpy' for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or '').split('.')[0] != 'numpy'


def test_ksc_callbacks_match_original_and_independent_tangents(request):
    relative = 'bayesfilter/highdim/ledh_canonical_neutra_targets_tf.py'
    source = subprocess.check_output(['git', 'show', f'{BASELINE}:{relative}'],
        cwd=Path(__file__).resolve().parents[1], text=True)
    node = next(node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == '_ksc_fused_model')
    namespace = {'tf': tf, 'DTYPE': DTYPE, 'PerPointScoreModel': targets.PerPointScoreModel}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'frozen_ksc_reference', 'exec'), namespace)  # noqa: S102 -- pinned diagnostic reference
    original, candidate = namespace['_ksc_fused_model'](), targets._ksc_fused_model()
    signature = [tf.TensorSpec([3, 2], DTYPE), tf.TensorSpec([3, 1], DTYPE),
        tf.TensorSpec([1], DTYPE), tf.TensorSpec([3, 1], DTYPE), tf.TensorSpec([3, 2], DTYPE)]

    def outputs(model, theta, points, observation, d_points, d_theta):
        return (model.transition_mean_fn(theta, points),
            model.transition_mean_tangent_fn(theta, points, d_points, d_theta),
            model.observation_fn(points), model.observation_jacobian_fn(points),
            model.observation_tangent_fn(points, d_points), model.process_covariance,
            model.observation_covariance, model.observation_log_density_fn(theta, points, observation),
            model.observation_log_density_tangent_fn(theta, points, observation, d_points, d_theta))

    compiled = tf.function(lambda *args: outputs(candidate, *args),
        input_signature=signature, jit_compile=True, autograph=False)
    reference = tf.function(lambda *args: outputs(original, *args),
        input_signature=signature, jit_compile=False, autograph=False)
    theta = tf.constant([[-1.3, .15], [.2, -.1], [1.8, .7]], DTYPE)
    points = tf.constant([[-.5], [.4], [1.2]], DTYPE)
    observation = tf.constant([.3], DTYPE)
    d_points = tf.constant([[.1], [-.3], [.2]], DTYPE)
    d_theta = tf.constant([[.4, -.2], [.3, .5], [-.1, .2]], DTYPE)
    initial = (theta, points, observation, d_points, d_theta)
    changed = (theta + .01, points - .04, observation + .02, d_points * .8, d_theta + .03)
    reports, first = [], None
    hlo = compiled.experimental_get_compiler_ir(*initial)(stage='hlo')
    for label, arguments in (('initial', initial), ('changed', changed), ('replay', initial)):
        actual, expected = compiled(*arguments), reference(*arguments)
        errors = []
        for a, b in zip(actual, expected, strict=True):
            tf.debugging.assert_all_finite(a, 'callback output')
            delta = tf.abs(a - b)
            limit = tf.constant(1e-12, DTYPE) * (1. + tf.abs(b))
            tf.debugging.assert_less_equal(delta, limit)
            errors.append(float(tf.reduce_max(delta / limit)))
        if first is None:
            first = actual
        if label == 'replay':
            for a, b in zip(actual, first, strict=True):
                tf.debugging.assert_equal(a, b)
        reports.append({'label': label, 'max_scaled_error': max(errors),
            'outputs': [a.numpy().tolist() for a in actual]})

    finite_differences = []
    for step in (1e-4, 5e-5):
        upper = compiled(theta + step * d_theta, points + step * d_points,
            observation, d_points, d_theta)
        lower = compiled(theta - step * d_theta, points - step * d_points,
            observation, d_points, d_theta)
        errors = []
        for primal, tangent in ((0, 1), (7, 8)):
            numeric = (upper[primal] - lower[primal]) / (2. * step)
            analytical = first[tangent]
            delta = tf.abs(numeric - analytical)
            limit = tf.constant(1e-7, DTYPE) * (1. + tf.abs(analytical))
            tf.debugging.assert_less_equal(delta, limit)
            errors.append(float(tf.reduce_max(delta / limit)))
        finite_differences.append({'step': step, 'max_scaled_error': max(errors)})

    # Independent closed-form derivative of Phi(theta_0) * state.
    expected_tangent = []
    for t, p, dp, dt in zip(theta.numpy().tolist(), points.numpy().tolist(),
            d_points.numpy().tolist(), d_theta.numpy().tolist(), strict=True):
        phi = math.exp(-.5 * t[0] ** 2) / math.sqrt(2. * math.pi)
        cdf = .5 * (1. + math.erf(t[0] / math.sqrt(2.)))
        expected_tangent.append([phi * dt[0] * p[0] + cdf * dp[0]])
    tf.debugging.assert_near(first[1], tf.constant(expected_tangent, DTYPE), atol=1e-12, rtol=1e-12)
    assert compiled.experimental_get_tracing_count() == 1
    assert hlo == compiled.experimental_get_compiler_ir(*changed)(stage='hlo')
    definition = compiled.get_concrete_function().graph.as_graph_def()
    operations = {node.op for node in definition.node}
    operations.update(node.op for function in definition.library.function for node in function.node_def)
    assert not operations & {'PyFunc', 'PyFuncStateless', 'EagerPyFunc', 'XlaHostCompute'}
    report = {'schema': 'filter_repair_merged_ksc_callbacks.v1', 'passed': True,
        'baseline': BASELINE, 'baseline_module_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'comparisons': reports, 'independent_tangents': finite_differences,
        'trace_count': 1, 'hlo_sha256': hashlib.sha256(hlo.encode()).hexdigest(),
        'nonclaims': ['No LEDH filter, NeuTra training, target admission or scientific qualification.']}
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory / 'ksc-callbacks.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (directory / 'ksc-callbacks.hlo.txt').write_text(hlo)
