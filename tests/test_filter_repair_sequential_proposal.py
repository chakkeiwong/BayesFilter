"""Frozen-source proposal arithmetic, decisions and enclosing XLA contracts."""

import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.sequential_proposal_tf import (
    acceptance_program,
    proposal_program,
)

D = tf.float64


@pytest.fixture(scope='module')
def frozen():
    source = subprocess.check_output(['git', 'show',
        'b3334646:bayesfilter/inference/sequential_map_covariance.py'], text=True)
    spec = importlib.util.spec_from_loader('sequential_proposal_frozen', loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, 'exec'), module.__dict__)  # noqa: S102 - frozen diagnostic authority
    dependency = subprocess.check_output(['git', 'show',
        'b3334646:bayesfilter/inference/sequential_preparation_tf.py'], text=True)
    namespace = {}
    exec(compile(dependency, 'proposal_frozen_trust_dependency', 'exec'), namespace)  # noqa: S102 - pin numerical dependency
    module.trust_region_program = namespace['trust_region_program']
    return module


def _target(row):
    return -.5 * tf.reduce_sum(row ** 2) - .1 * tf.reduce_sum(row ** 4), -row - .4 * row ** 3


def _arguments(dimension, radius=.2):
    center = tf.linspace(tf.constant(-.4, D), tf.constant(.7, D), dimension)
    scale = tf.linspace(tf.constant(.8, D), tf.constant(1.3, D), dimension)
    value, score = _target(center)
    precision = tf.linalg.diag(scale ** 2 * (1. + 1.2 * center ** 2))
    return (center, value, score, scale, precision, tf.constant(radius, D),
        tf.constant(.95, D), tf.constant(.1, D))


def _expected(frozen, target, args, policy, active):
    center, center_value, center_score, scale, precision, radius, fraction, ratio = args
    step = frozen._solve_trust_region_tf(precision, scale * center_score, float(radius))
    proposed = center + scale * step['step']
    value, score = frozen._scalar_value_score(target, proposed, int(center.shape[0]))
    old_norm, new_norm = float(tf.linalg.norm(scale * center_score)), float(tf.linalg.norm(scale * score))
    actual, predicted = float(value) - float(center_value), step['predicted_improvement']
    rho = actual / predicted if predicted > 0. else float('-inf')
    finite = bool(tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score)))
    gate = frozen._proposal_score_gate(old_norm, new_norm, policy=policy,
        fractional_factor=float(fraction), active=active)
    accepted = frozen._proposal_is_accepted(finite, actual=actual, predicted=predicted,
        rho=rho, acceptance_ratio=float(ratio), score_gate_passed=gate['passed'])
    return {'position': proposed, 'value': value, 'score': score, 'step': step['step'],
        'actual': actual, 'predicted': predicted, 'rho': rho, 'old_norm': old_norm, 'new_norm': new_norm,
        'finite': finite, 'boundary': step['boundary_active'], 'accepted': accepted,
        'score_passed': gate['passed'], 'legacy_passed': gate['legacy_fractional_passed'],
        'required_norm_max': 0. if gate['required_score_norm_max'] is None else gate['required_score_norm_max'],
        'resolution_floor': 0. if gate['numerical_resolution_floor'] is None else gate['numerical_resolution_floor']}


def _same(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, (bool, np.bool_)):
            assert bool(actual[key]) == value, key
        else:
            np.testing.assert_allclose(actual[key], value, atol=1e-10, rtol=1e-10, equal_nan=True, err_msg=key)


@pytest.mark.parametrize('policy,active', [('fractional', True), ('resolvable_decrease', True), ('ignored', False)])
def test_original_acceptance_boundaries(frozen, policy, active):
    decide = acceptance_program(policy, active)
    floor = math.sqrt(math.ulp(1.)) * 6.
    norms = [(6., 5.7), (6., np.nextafter(5.7, math.inf)), (6., np.nextafter(5.7, -math.inf)),
        (6., 6. - floor), (6., np.nextafter(6. - floor, -math.inf)),
        (6., np.nextafter(6. - floor, math.inf)), (0., 0.), (6., float('nan')),
        (float('nan'), 0.), (float('inf'), 0.), (6., float('inf'))]
    for old, new in norms:
        for actual, predicted, finite in [(1., 10., True), (np.nextafter(1., 0.), 10., True),
                (1., 0., True), (1., -1., True), (0., 1., True), (1., 1., False),
                (float('nan'), 1., True), (1., float('nan'), True)]:
            rho = actual / predicted if predicted > 0. else float('-inf')
            expected_gate = frozen._proposal_score_gate(old, new, policy=policy, fractional_factor=.95, active=active)
            expected = {'rho': rho, 'score_passed': expected_gate['passed'],
                'legacy_passed': expected_gate['legacy_fractional_passed'],
                'required_norm_max': 0. if expected_gate['required_score_norm_max'] is None else expected_gate['required_score_norm_max'],
                'resolution_floor': 0. if expected_gate['numerical_resolution_floor'] is None else expected_gate['numerical_resolution_floor'],
                'accepted': frozen._proposal_is_accepted(finite, actual=actual, predicted=predicted,
                    rho=rho, acceptance_ratio=.1, score_gate_passed=expected_gate['passed'])}
            _same(decide(*(tf.constant(v, D) for v in (old, new, actual, predicted, .95, .1)), tf.constant(finite)), expected)
    assert decide.experimental_get_tracing_count() == 1


@pytest.mark.parametrize('dimension', [1, 3, 5])
@pytest.mark.parametrize('policy,active', [('fractional', True), ('resolvable_decrease', True), ('ignored', False)])
def test_full_proposal_matches_frozen(frozen, dimension, policy, active):
    program = proposal_program(_target, dimension, policy, active)
    for radius in (.02, .2, 2.):
        args = _arguments(dimension, radius)
        _same(program(*args), _expected(frozen, _target, args, policy, active))
    assert program.experimental_get_tracing_count() == 1


def test_exact_scalar_callback_count_and_order(frozen):
    counter = tf.Variable(0, dtype=tf.int64, trainable=False)
    points = tf.Variable(tf.zeros([4, 3], D), trainable=False)

    def recorded(row):
        index = counter.read_value()
        points.scatter_nd_update(tf.reshape(index, [1, 1]), row[None, :])
        counter.assign_add(1)
        return _target(row)

    program = proposal_program(recorded, 3, 'fractional', True)
    expected = []
    for radius in (.05, .1, .2, 1.):
        args = _arguments(3, radius)
        expected.append(_expected(frozen, _target, args, 'fractional', True)['position'])
        program(*args)
    assert int(counter) == 4
    np.testing.assert_allclose(points, tf.stack(expected), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('fault', ['value', 'score'])
def test_nonfinite_proposals_preserve_rejection(frozen, fault):
    def target(row):
        value, score = _target(row)
        if fault == 'value':
            value = tf.constant(float('nan'), D)
        else:
            score = tf.fill(tf.shape(row), tf.constant(float('inf'), D))
        return value, score

    args = _arguments(3)
    expected = _expected(frozen, target, args, 'resolvable_decrease', True)
    actual = proposal_program(target, 3, 'resolvable_decrease', True)(*args)
    _same(actual, expected)
    assert not bool(actual['accepted'])


def test_callback_errors_have_no_eager_retry():
    counter = tf.Variable(0, dtype=tf.int64, trainable=False)

    def bad(row):
        counter.assign_add(1)
        return row.numpy().sum(), row

    with pytest.raises(AttributeError):
        proposal_program(bad, 3, 'fractional', True)(*_arguments(3))
    assert int(counter) == 0
    with pytest.raises(ValueError, match='unknown proposal'):
        proposal_program(_target, 3, 'unknown', True)


@pytest.mark.parametrize('dimension', [3, 5])
def test_enclosing_hlo_inputs_and_graph_reference(dimension, request):
    candidate = proposal_program(_target, dimension, 'resolvable_decrease', True)
    reference = proposal_program(_target, dimension, 'resolvable_decrease', True, jit_compile=False)

    @tf.function(input_signature=candidate.input_signature, jit_compile=True, autograph=False)
    def enclosing(*args):
        return candidate.python_function(*args)

    args = _arguments(dimension)
    _same(enclosing(*args), {key: value.numpy() for key, value in reference(*args).items()})
    hlo = enclosing.experimental_get_compiler_ir(*args)(stage='hlo')
    entry = hlo[hlo.rfind('\nENTRY '):]
    assert sorted(int(i) for i in re.findall(r'\bparameter\((\d+)\)', entry)) == list(range(8))
    changed = list(args)
    changed[0] = changed[0] * .9
    changed[1], changed[2] = _target(changed[0])
    changed[3] = changed[3] * 1.1
    changed[4] = changed[4] * 1.2
    changed[5] = changed[5] * .8
    changed[6], changed[7] = tf.constant(.97, D), tf.constant(.12, D)
    _same(enclosing(*changed), {key: value.numpy() for key, value in reference(*changed).items()})
    assert enclosing.experimental_get_compiler_ir(*changed)(stage='hlo') == hlo
    graph = enclosing.get_concrete_function().graph.as_graph_def()
    nodes = list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
    assert not any(node.op in ('PyFunc', 'EagerPyFunc', 'PyFuncStateless') for node in nodes)
    assert enclosing.experimental_get_tracing_count() == 1
    path = Path(request.config.getoption('xmlpath')).parent / f'proposal-enclosing-{dimension}.json'
    with path.open('x') as handle:
        json.dump({'dimension': dimension, 'runtime_inputs': 8, 'graph_nodes': len(nodes),
            'hlo_bytes': len(hlo.encode()), 'jit_compile': True}, handle, indent=2)
        handle.write('\n')

@pytest.mark.parametrize('dimension', [1, 3])
@pytest.mark.parametrize('policy', ['fractional', 'resolvable_decrease'])
def test_full_public_history_matches_frozen(frozen, dimension, policy, request):
    from bayesfilter.inference import sequential_map_covariance as current
    from tests.test_filter_repair_fixed_stability import _compare

    start = tf.linspace(tf.constant(.002, D), tf.constant(.004, D), dimension)

    def batched(rows):
        return -.5 * tf.reduce_sum(rows ** 2, axis=1) - .1 * tf.reduce_sum(rows ** 4, axis=1), -rows - .4 * rows ** 3

    options = {'locator_policy': 'center_first', 'terminal_score_max_abs': 1e-10,
        'initial_radius': .25, 'search_sample_count': 8, 'regression_sample_count': 24,
        'terminal_sample_count': 24, 'max_attempts': 4, 'max_exact_evaluations': 256,
        'proposal_score_acceptance_policy': policy, 'record_refinement_movement_diagnostics': True,
        'seed': (2026, 715)}
    records = []
    for module in (frozen, current):
        records.append(module.estimate_sequential_map_covariance(_target, [start],
            batched_value_and_score_fn=batched, config=module.SequentialMapCovarianceConfig(**options)).payload())
    path = Path(request.config.getoption('xmlpath')).parent / f'proposal-public-{dimension}-{policy}.json'
    with path.open('x') as handle:
        json.dump({'before': records[0], 'after': records[1]}, handle, indent=2)
        handle.write('\n')
    assert any(row['action'] in ('proposal_accepted', 'proposal_rejected') for row in records[0]['diagnostics']['history'])
    _compare(records[1], records[0])
