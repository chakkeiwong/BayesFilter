"""Adverse evidence checks for SVD cost comparison eligibility."""

import copy
from types import SimpleNamespace

import pytest

from tests.test_filter_repair_campaign import load


def report(valid=True):
    return {'numerical_passed': valid, 'comparisons': [
        {'passed': valid, 'actual': {'likelihood': [.5 if valid else .6],
            'mean': [[[.1, .2, .3]]], 'covariance': [[[[1., 0.], [0., 1.]]]],
            'on_support': [True], 'rank': [3]},
         'independent_reference': {'likelihood': [.5], 'mean': [[[.1, .2, .3]]],
            'covariance': [[[[1., 0.], [0., 1.]]]]}} for _ in range(2)]}


@pytest.mark.parametrize('fault', ['claimed_pass', 'hidden_nan', 'rank', 'support', 'shape', 'aggregate'])
def test_svd_costs_reject_false_numerical_evidence(fault):
    result = copy.deepcopy(report())
    actual = result['comparisons'][0]['actual']
    if fault == 'claimed_pass':
        actual['likelihood'][0] += .01
    elif fault == 'hidden_nan':
        actual['mean'][0][0][0] = float('nan')
    elif fault == 'rank':
        actual['rank'] = [2]
    elif fault == 'support':
        actual['on_support'] = [False]
    elif fault == 'shape':
        actual['mean'][0].append([.1, .2, .3])
    else:
        result['numerical_passed'] = False
    with pytest.raises(AssertionError):
        load('analyze_filter_repair_svd_costs').numerical_status(result)


def test_svd_costs_keep_inaccurate_baseline_but_forbid_ranking():
    analyzer = load('analyze_filter_repair_svd_costs')
    assert analyzer.numerical_status(report())
    assert not analyzer.numerical_status(report(False))
    comparison = analyzer.comparison_metrics({'numerical_passed': False}, {'numerical_passed': True})
    assert comparison['eligible'] is False and 'warm_ratio' not in comparison


def test_svd_costs_decline_shared_preflight(tmp_path, monkeypatch):
    runner = load('run_filter_repair_campaign')
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    for group in ('svd_graph_attribution_gpu', *runner.TEST_BATCHES['svd_cost_gpu']):
        args = SimpleNamespace(action='test', device='GPU', group=group,
            gpu_uuid='GPU-device-2', gpu_preflight=[{'performance_preflight_uncontended': False}] * 2)
        with pytest.raises(RuntimeError, match='declined before launch'):
            runner.require_unshared_cost_preflight(args)
