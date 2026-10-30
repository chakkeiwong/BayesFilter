"""Diagnostic comparison of unchanged sequential scores; never execute pfor."""

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_canonical_batch_fused_tf as current
from bayesfilter.highdim.ledh_canonical_score_tf import (
    canonical_value_and_analytical_score,
    make_analytical_score_program,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '020d794be'
DTYPE = tf.float64


@pytest.mark.parametrize('mode', ['pfor', 'unknown'])
def test_unapproved_modes_reject_before_tensor_work(mode, monkeypatch):
    def fail(*_args, **_kwargs):
        pytest.fail('unsupported mode reached TensorFlow numerical work')
    monkeypatch.setattr(tf, 'convert_to_tensor', fail)
    with pytest.raises(ValueError, match='unapproved pfor execution is unavailable'):
        current.canonical_batch_fused_value_score(None, None, None, None, None, None, None,
                                                  substeps=2, k_batch_mode=mode)


@pytest.mark.parametrize('name', ['ledh_k_batch_parity_and_timing.py', 'ledh_execution_mode_matrix.py'])
def test_exploratory_harness_retired_before_framework_import(name, request):
    # A fresh interpreter is necessary: TF is already imported by the test worker.
    command = [sys.executable, '-c', '''
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'tensorflow','numpy','bayesfilter'}:
            raise AssertionError('framework imported before retirement: '+fullname)
sys.meta_path.insert(0, Block())
try:
    runpy.run_path(sys.argv[1], run_name='__main__')
except RuntimeError as error:
    assert 'HISTORICAL pfor exploration is retired' in str(error)
    print(str(error))
else:
    raise AssertionError('retired benchmark executed')
''', str(ROOT / 'docs/benchmarks' / name)]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory / f'retired-{name}.json').write_text(json.dumps({
        'path': name, 'returncode': result.returncode, 'stdout': result.stdout,
        'framework_import_blocked': True}, indent=2) + '\n')


def _model():
    eye = tf.eye(2, dtype=DTYPE)
    return current.PerPointScoreModel(
        transition_mean_fn=lambda theta, points: points + theta[:, :1] * tf.sin(points),
        transition_mean_tangent_fn=lambda theta, points, tangent, direction: (
            tangent + direction[:, :1] * tf.sin(points) + theta[:, :1] * tf.cos(points) * tangent),
        observation_fn=lambda points: points,
        observation_jacobian_fn=lambda points: tf.broadcast_to(eye, [tf.shape(points)[0], 2, 2]),
        observation_tangent_fn=lambda points, tangent: tangent,
        process_covariance=.4 * eye, observation_covariance=.6 * eye)


def test_sequential_complete_program_and_direction_scores(request):
    path = 'bayesfilter/highdim/ledh_canonical_batch_fused_tf.py'
    source = subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'], cwd=ROOT, text=True)
    frozen = ModuleType('_ledh_sequential_only_frozen_reference')
    sys.modules[frozen.__name__] = frozen
    try:
        exec(compile(source, '<frozen-sequential-only-reference>', 'exec'), frozen.__dict__)  # noqa: S102 -- fixed Git test authority
        assert inspect.signature(current.canonical_batch_fused_value_score).parameters['k_batch_mode'].default == 'sequential'
        model = _model()
        rng = np.random.default_rng(81100)
        initial = tf.constant(rng.normal(size=(8, 2)) * .1, DTYPE)
        covs = tf.broadcast_to(tf.eye(2, dtype=DTYPE), [8, 2, 2])
        noises = tf.constant(rng.normal(size=(2, 8, 2)) * .1, DTYPE)
        observations = tf.constant(rng.normal(size=(2, 2)) * .1, DTYPE)
        theta = tf.constant([[.15], [.2]], DTYPE)
        directions = tf.constant([[[1.], [-.7]], [[1.], [.3]]], DTYPE)
        signature = [tf.TensorSpec([2, 1], DTYPE), tf.TensorSpec([2, 2, 1], DTYPE),
                     tf.TensorSpec([8, 2], DTYPE), tf.TensorSpec([8, 2, 2], DTYPE),
                     tf.TensorSpec([2, 8, 2], DTYPE), tf.TensorSpec([2, 2], DTYPE)]

        def factory(function, explicit):
            @tf.function(input_signature=signature, jit_compile=True, autograph=False)
            def owner(theta, directions, initial, covs, noises, observations):
                return function(model, theta, directions, initial, covs, noises, observations,
                                substeps=2, **({'k_batch_mode': 'sequential'} if explicit else {}))
            return owner

        owner = factory(current.canonical_batch_fused_value_score, False)
        explicit = factory(current.canonical_batch_fused_value_score, True)
        previous = factory(frozen.canonical_batch_fused_value_score, True)
        native = make_analytical_score_program(
            lambda theta, direction: current._single_cloud_model(model, theta, direction),
            dtype=DTYPE, theta_shape=(1,), initial_state_shape=(8, 2), horizon=2,
            observation_dimension=2, flow_substeps=2, reset_policy='none')
        records = []
        for shift in (0., .01):
            args = (theta + shift, directions, initial, covs, noises, observations + shift)
            actual = owner(*args)
            for comparison in (previous(*args), explicit(*args)):
                for lhs, rhs in zip(tf.nest.flatten(actual), tf.nest.flatten(comparison), strict=True):
                    np.testing.assert_allclose(lhs, rhs, atol=1e-9, rtol=1e-9)
            assert bool(tf.reduce_all(actual[2]['program_valid']))
            np.testing.assert_allclose(actual[1][:, 1], actual[1][:, 0] * [-.7, .3], atol=1e-9, rtol=1e-9)
            finite_differences = []
            for row in range(2):
                for direction in range(2):
                    point, tangent = args[0][row], directions[row, direction]
                    value, score = native(point, tangent, initial, covs, noises, args[5],
                                          tf.zeros_like(initial), tf.zeros_like(covs))
                    np.testing.assert_allclose(actual[0][row], value, atol=1e-9, rtol=1e-9)
                    np.testing.assert_allclose(actual[1][row, direction], score[0], atol=1e-9, rtol=1e-9)
                    # Independent five-point VALUE derivative of the finite program.
                    step = 5e-4
                    values = []
                    for scale in (-2, -1, 1, 2):
                        shifted = point + scale * step * tangent
                        bound = current._single_cloud_model(model, shifted, tangent)
                        values.append(float(canonical_value_and_analytical_score(
                            bound, shifted, initial, covs, noises, args[5],
                            flow_substeps=2, with_score=False, reset_policy='none')[0]))
                    derivative = (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step)
                    np.testing.assert_allclose(actual[1][row, direction], derivative, atol=2e-6, rtol=2e-6)
                    finite_differences.append({'row': row, 'direction': direction, 'derivative': derivative})
            records.append({'shift': shift, 'value': actual[0].numpy().tolist(),
                            'score': actual[1].numpy().tolist(), 'five_point': finite_differences})
        assert owner.experimental_get_tracing_count() == 1
        graph = owner.get_concrete_function().graph.as_graph_def()
        ops = {n.op for n in graph.node} | {n.op for f in graph.library.function for n in f.node_def}
        assert not ops & {'PyFunc', 'EagerPyFunc', 'PyFuncStateless', 'XlaHostCompute'}
        hlo = owner.experimental_get_compiler_ir(*args)(stage='hlo')
        directory = Path(request.config.getoption('xmlpath')).parent
        (directory / 'sequential-score.hlo.txt').write_text(hlo)
        (directory / 'sequential-score.json').write_text(json.dumps({
            'baseline': BASELINE, 'baseline_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'records': records, 'trace_count': 1, 'jit_compile': True,
            'device': actual[0].device, 'pfor_executed': False,
            'hlo_sha256': hashlib.sha256(hlo.encode()).hexdigest(),
            'nonclaims': ['No-reset T>1 is a finite-program derivative diagnostic, not a likelihood estimator.',
                          'Row-mapped adapter is ineligible for NeuTra training.']}, indent=2) + '\n')
    finally:
        sys.modules.pop(frozen.__name__, None)
