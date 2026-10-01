"""Mutation checks for the narrowly scoped, approved equivalence rule."""

from copy import deepcopy

import pytest

from tests.filter_repair_record_comparison import (
    FACTOR_ARRAYS,
    FACTOR_DIAGNOSTICS,
    FACTOR_SCHEMA,
    compare_records,
)


def record():
    return {'schema': FACTOR_SCHEMA, 'accepted': True, 'status': 'usable',
        'anchor_indices': [0, 4], 'factor_count': 2,
        **{key: [[1.0, .1], [.1, 2.0]] for key in FACTOR_ARRAYS},
        'diagnostics': {**{key: 1.0 for key in FACTOR_DIAGNOSTICS},
            'final_loss': 1e-12, 'optimizer_iterations': 73,
            'optimizer_objective_evaluations': 217, 'prediction_jacobian_rank': 14,
            'holdout_score_relative_rmse_cap': .001}}


@pytest.mark.parametrize('key', sorted(FACTOR_ARRAYS | FACTOR_DIAGNOSTICS))
def test_only_declared_factor_fields_receive_allowance(key):
    expected, actual = record(), record()
    if key in FACTOR_ARRAYS:
        actual[key][0][0] += 5e-9
    else:
        actual['diagnostics'][key] += 5e-9
    compare_records({'history': [{'fit': actual}]}, {'history': [{'fit': expected}]})
    if key in FACTOR_ARRAYS:
        actual[key][0][0] += 3e-8
    else:
        actual['diagnostics'][key] += 3e-8
    with pytest.raises(AssertionError):
        compare_records(actual, expected)


@pytest.mark.parametrize('mutation', ['count', 'decision', 'missing', 'extra', 'shape',
    'anchor', 'rank', 'final_loss', 'threshold', 'schema', 'boolean_type'])
def test_bad_records_still_fail(mutation):
    expected, actual = record(), record()
    if mutation == 'count':
        actual['diagnostics']['optimizer_iterations'] += 1
    elif mutation == 'decision':
        actual['accepted'] = False
    elif mutation == 'missing':
        del actual['precision_z']
    elif mutation == 'extra':
        actual['unchecked'] = 1.0
    elif mutation == 'shape':
        actual['precision_z'] = [[1.0, .1]]
    elif mutation == 'anchor':
        actual['anchor_indices'].reverse()
    elif mutation == 'rank':
        actual['diagnostics']['prediction_jacobian_rank'] -= 1
    elif mutation == 'final_loss':
        actual['diagnostics']['final_loss'] += 5e-9
    elif mutation == 'threshold':
        actual['diagnostics']['holdout_score_relative_rmse_cap'] += 5e-9
    elif mutation == 'schema':
        actual['schema'] += '_other'
    else:
        actual['accepted'] = 1
    with pytest.raises(AssertionError):
        compare_records(actual, expected)


@pytest.mark.parametrize('key', ['value', 'score', 'center', 'precision_z', 'design_condition'])
def test_unrelated_nested_fields_keep_strict_tolerance(key):
    expected = {'fit': record(), 'other': {key: 1.0}}
    actual = deepcopy(expected)
    actual['other'][key] += 5e-9
    with pytest.raises(AssertionError):
        compare_records(actual, expected)


def test_unrecognized_factor_schema_keeps_strict_tolerance():
    expected, actual = record(), record()
    expected['schema'] = actual['schema'] = 'unrecognized'
    actual['loadings'][0][0] += 5e-9
    with pytest.raises(AssertionError):
        compare_records(actual, expected)
