"""Independent host recurrence for native proposal-attempt order."""

import gc
import json
import re
import weakref
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_attempts_tf as native
from bayesfilter.inference import sequential_map_covariance as sequential
from tests.test_filter_repair_sequential_proposal import _expected, _target
from tests.test_filter_repair_sequential_proposal import frozen as _frozen

D = tf.float64


def _oracle(frozen, cfg, args, target=_target):
    center, value, score, scale, radius, stalled, usable, precision, second = args
    center_value = float(value)
    best_value, best_position, best_score = center_value, center, score
    best_exists, accepted = False, False
    old_norm = float(tf.linalg.norm(scale * score))
    last = {'actual': float('-inf'), 'predicted': float('-inf'), 'rho': float('-inf'),
        'old_norm': old_norm, 'new_norm': old_norm, 'boundary': False, 'accepted': False}
    second_calls, proposal_calls = 0, 0
    history = []
    for index in range(cfg.structured_max_factors):
        if index == 1:
            if accepted:
                break
            second_calls += 1
            precision, usable = second
        if not bool(usable):
            history.append(None)
            continue
        proposal_calls += 1
        packed = (center, value, score, scale, precision, radius,
            tf.constant(cfg.score_reduction_factor, D), tf.constant(cfg.acceptance_ratio, D))
        last = _expected(frozen, target, packed, cfg.proposal_score_acceptance_policy, cfg.require_proposal_score_reduction)
        history.append(last)
        if last['finite'] and float(last['value']) > best_value:
            best_exists, best_value, best_position, best_score = True, float(last['value']), last['position'], last['score']
        accepted = last['accepted']
        if accepted:
            break
    promoted = best_exists and (not accepted or best_value > float(last['value']))
    if last['rho'] < cfg.shrink_threshold or not accepted:
        after, action = float(radius) * cfg.shrink_factor, 0
    elif last['rho'] >= cfg.expansion_threshold and last['boundary']:
        after, action = min(cfg.maximum_radius, float(radius) * cfg.expansion_factor), 1
    else:
        after, action = float(radius), 2
    return {'accepted': accepted, 'attempted': len(history), 'evaluations': proposal_calls,
        'best_exists': best_exists, 'center_value': best_value, 'center': best_position, 'center_score': best_score,
        'stalled': 0 if best_exists else int(stalled) + 1, 'last_evaluated': proposal_calls > 0,
        'promoted_without_acceptance': promoted, 'radius_after': after, 'radius_action': action}, history, second_calls


@pytest.mark.parametrize('first_usable,second_usable,first_precision,second_precision',
    [(False, False, 1., 1.), (False, True, 1., 1.), (True, True, 1., 1.),
     (True, True, 100., 1.), (True, False, 100., 1.), (True, True, 100., 200.)])
@pytest.mark.parametrize('policy', ['fractional', 'resolvable_decrease'])
def test_attempt_order_and_independent_incumbent(first_usable, second_usable, first_precision, second_precision, policy):
    calls = tf.Variable(0, dtype=tf.int64, trainable=False)

    @tf.function(input_signature=[tf.TensorSpec([3, 3], D), tf.TensorSpec([], tf.bool)], jit_compile=True, autograph=False)
    def second_fit(precision, usable):
        calls.assign_add(1)
        return {'precision': precision, 'usable': usable}

    cfg = sequential.SequentialMapCovarianceConfig(refinement_geometry_policy='factor_correlation',
        structured_max_factors=2, proposal_score_acceptance_policy=policy)
    center = tf.constant([.5, -.4, .3], D)
    value, score = _target(center)
    args = (center, value, score, tf.ones([3], D), tf.constant(1., D), tf.constant(2),
        tf.constant(first_usable), tf.eye(3, dtype=D) * first_precision,
        (tf.eye(3, dtype=D) * second_precision, tf.constant(second_usable)))
    expected, history, second_calls = _oracle(_frozen.__wrapped__(), cfg, args)
    actual = native.attempts_program(_target, second_fit, 3, cfg)(*args)
    for key, value in expected.items():
        if isinstance(value, (bool, int)):
            assert actual[key].numpy().item() == value, key
        else:
            np.testing.assert_allclose(actual[key], value, atol=1e-10, rtol=1e-10, err_msg=key)
    assert int(calls) == second_calls
    for index, row in enumerate(history):
        assert bool(actual['used'][index])
        assert bool(actual['evaluated'][index]) == (row is not None)
        if row is not None:
            for key, value in row.items():
                np.testing.assert_allclose(actual['histories'][key][index], value, atol=1e-10, rtol=1e-10, equal_nan=True)


def test_single_factor_skips_second_callback_and_retains_graph_inputs(request):
    calls = tf.Variable(0, dtype=tf.int64, trainable=False)

    @tf.function(input_signature=[tf.TensorSpec([3, 3], D), tf.TensorSpec([], tf.bool)], jit_compile=True, autograph=False)
    def second_fit(precision, usable):
        calls.assign_add(1)
        return {'precision': precision, 'usable': usable}

    cfg = sequential.SequentialMapCovarianceConfig(refinement_geometry_policy='factor_correlation', structured_max_factors=1)
    center = tf.constant([.5, -.4, .3], D)
    value, score = _target(center)
    args = (center, value, score, tf.ones([3], D), tf.constant(.1, D), tf.constant(2),
        tf.constant(True), tf.eye(3, dtype=D), (tf.eye(3, dtype=D), tf.constant(True)))
    program = native.attempts_program(_target, second_fit, 3, cfg)
    actual = program(*args)
    expected, _, _ = _oracle(_frozen.__wrapped__(), cfg, args)
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, atol=1e-10, rtol=1e-10)
    assert int(calls) == 0
    assert int(actual['attempted']) == 1
    hlo = program.experimental_get_compiler_ir(*args)(stage='hlo')
    altered = list(args)
    altered[0] = altered[0] * .8
    altered[1], altered[2] = _target(altered[0])
    altered[3] = altered[3] * 1.1
    altered[4] = altered[4] * 1.2
    altered[5] = tf.constant(5)
    altered[6] = tf.constant(False)
    altered[7] = altered[7] * 1.3
    changed_result = program(*altered)
    changed_expected, _, _ = _oracle(_frozen.__wrapped__(), cfg, altered)
    for key, value in changed_expected.items():
        np.testing.assert_allclose(changed_result[key], value, atol=1e-10, rtol=1e-10)
    changed_hlo = program.experimental_get_compiler_ir(*altered)(stage='hlo')
    directory = Path(request.config.getoption('xmlpath')).parent
    for name, text in [('first', hlo), ('changed', changed_hlo)]:
        with (directory / f'attempts-inputs-{name}.hlo').open('x') as handle:
            handle.write(text)
    def normalized(text):
        # 01684 proves only Grappler's generated zero-constant metadata suffix
        # changes. Keep every operation, operand, literal and other annotation.
        return re.sub(r'op_name="(zeros(?:_\d+)?)/_\d+"(?= source_file="dummy_file_name" source_line=10)',
            r'op_name="\1/__generated"', text)

    assert normalized(changed_hlo) == normalized(hlo)
    entry = hlo[hlo.rfind('\nENTRY '):]
    assert sorted(int(i) for i in re.findall(r'\bparameter\((\d+)\)', entry)) == list(range(10))
    assert program.experimental_get_tracing_count() == 1


def test_better_first_incumbent_survives_later_model_acceptance():
    """Scripted exact outputs isolate controller mechanics, not score validity."""
    evaluations = tf.Variable(0, dtype=tf.int64, trainable=False)
    visited = tf.Variable(tf.zeros([2, 3], D), trainable=False)
    fits = tf.Variable(0, dtype=tf.int64, trainable=False)

    def target(row):
        first = row[0] > .4
        return tf.where(first, tf.constant(2., D), tf.constant(1., D)), tf.stack(
            [tf.where(first, tf.constant(1., D), tf.constant(.1, D)), tf.constant(0., D), tf.constant(0., D)])

    def recorded(row):
        visited.scatter_nd_update(tf.reshape(evaluations.read_value(), [1, 1]), row[None, :])
        evaluations.assign_add(1)
        return target(row)

    @tf.function(input_signature=[tf.TensorSpec([3, 3], D), tf.TensorSpec([], tf.bool)], jit_compile=True, autograph=False)
    def second_fit(precision, usable):
        fits.assign_add(1)
        return {'precision': precision, 'usable': usable}

    cfg = sequential.SequentialMapCovarianceConfig(refinement_geometry_policy='factor_correlation', structured_max_factors=2)
    args = (tf.zeros([3], D), tf.constant(0., D), tf.constant([1., 0., 0.], D),
        tf.ones([3], D), tf.constant(1., D), tf.constant(2), tf.constant(True),
        tf.eye(3, dtype=D) * 2., (tf.eye(3, dtype=D) * 4., tf.constant(True)))
    expected, _, _ = _oracle(_frozen.__wrapped__(), cfg, args, target)
    result = native.attempts_program(recorded, second_fit, 3, cfg)(*args)
    for key, value in expected.items():
        np.testing.assert_allclose(result[key], value, atol=1e-10, rtol=1e-10)
    assert bool(result['accepted']) and bool(result['promoted_without_acceptance'])
    assert float(result['center_value']) == 2.
    assert int(evaluations) == 2 and int(fits) == 1
    np.testing.assert_array_equal(visited, [[.5, 0., 0.], [.25, 0., 0.]])


@pytest.mark.parametrize('capacity', [4, 32])
def test_actual_second_factor_records_and_resource_lifetime(capacity, request):
    from bayesfilter.inference import factor_correlation_geometry as factor
    from bayesfilter.inference import sequential_factor_attempt_tf as provider
    from bayesfilter.inference import sequential_structured_fit_tf as fit_native
    from bayesfilter.inference.sequential_structured_preparation_tf import (
        structured_data_program,
    )
    from tests.test_filter_repair_fixed_stability import _compare
    from tests.test_filter_repair_structured_fit import _prepared_arguments
    from tests.test_filter_repair_structured_fit import frozen as frozen_structured
    from tests.test_filter_repair_structured_memory import _inputs
    from tests.test_filter_repair_structured_preparation import _before

    scalar, batched, preparation_args = _inputs(5, capacity)
    data = structured_data_program(scalar, batched, 5, 20, capacity, True)(*preparation_args)
    cfg = sequential.SequentialMapCovarianceConfig(refinement_geometry_policy='factor_correlation',
        structured_max_factors=2, require_proposal_score_reduction=False)
    frozen = frozen_structured.__wrapped__()
    original_data, _ = _before(frozen, scalar, batched, preparation_args, 20, True)
    original = frozen._fit_factor_from_data(original_data, factor_count=2, config=cfg)
    assert original['status'] == 'usable'
    second = provider.second_factor_program(5, 10 + capacity, 10, cfg)
    center, score, scale, radius = preparation_args[:4]
    value = scalar(center)[0]
    program = native.attempts_program(scalar, second, 5, cfg)
    args = (center, value, score, scale, radius, tf.constant(2), tf.constant(False),
        tf.eye(5, dtype=D), _prepared_arguments(data))
    actual = program(*args)
    expected_args = (*args[:8], (tf.constant(original['precision_z'], D), tf.constant(True)))
    expected, _, _ = _oracle(_frozen.__wrapped__(), cfg, expected_args, scalar)
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, atol=1e-10, rtol=1e-10, err_msg=key)
    computed = actual['second_fit']['computed']
    assert int(computed['input_status']) == 0
    result = factor._factor_result_from_computed(computed['fit'],
        factor.FactorCorrelationGeometryConfig(factor_count=2,
            holdout_score_relative_rmse=cfg.structured_holdout_score_relative_rmse),
        5, int(data['active_training_rows']), 10, True).payload()
    _compare(result, {key: original[key] for key in result})
    concrete = program.get_concrete_function()
    hlo = program.experimental_get_compiler_ir(*args)(stage='hlo')
    entry = hlo[hlo.rfind('\nENTRY '):]
    # 8 first-step operands, 7 prepared inputs, callback precision and guard.
    assert len(re.findall(r'\bparameter\((\d+)\)', entry)) == 17
    graph = concrete.graph
    refs = [weakref.ref(v) for v in graph.variables]
    graph_ref = weakref.ref(graph)
    factor._make_factor_program.cache_clear()
    fit_native._cached_structured_fit.cache_clear()
    native.attempts_program.cache_clear()
    provider.second_factor_program.cache_clear()
    del program, second
    gc.collect()
    repeated = concrete(*args)
    _compare(tf.nest.map_structure(lambda value: value.numpy(), repeated),
        tf.nest.map_structure(lambda value: value.numpy(), actual))
    del graph, concrete
    gc.collect()
    assert graph_ref() is None
    assert all(ref() is None for ref in refs)
    path = Path(request.config.getoption('xmlpath')).parent / f'actual-factor-attempt-{capacity}.json'
    with path.open('x') as handle:
        json.dump({'capacity': capacity, 'original': original, 'actual_fit': result,
            'runtime_operands': 17, 'resource_released': True}, handle, indent=2)
        handle.write('\n')
