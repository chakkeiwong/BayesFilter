"""Campaign prices must cover the full search and every verified export."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'release_prices', Path(__file__).resolve().parents[1] / 'scripts/run_hmc_v7_release_prices.py')
prices = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prices)


def complete_result():
    return dict(completion_status='complete', checkpoint_recomputed=True,
                expectation_met=True, verified_candidate_ids=['a', 'b'],
                candidate_states={'a': 'verified', 'b': 'verified', 'c': 'rejected'})


def test_complete_search_with_every_verified_member_has_a_delivery_price():
    assert prices.full_search_delivered(complete_result())


@pytest.mark.parametrize('completion', ['partial_budget', 'paused_infrastructure',
                                      'shared_invalidity', None])
def test_partial_member_delivery_does_not_price_the_complete_search(completion):
    result = complete_result()
    result['completion_status'] = completion
    assert not prices.full_search_delivered(result)
    assert result['verified_candidate_ids'] == ['a', 'b']


@pytest.mark.parametrize('change', [
    {'verified_candidate_ids': []},
    {'verified_candidate_ids': ['a']},
    {'verified_candidate_ids': ['a', 'b', 'b']},
    {'candidate_states': {'a': 'verified', 'b': 'validating'}},
    {'checkpoint_recomputed': False},
    {'expectation_met': False},
])
def test_completion_alone_does_not_establish_checked_positive_delivery(change):
    result = complete_result()
    result.update(change)
    assert not prices.full_search_delivered(result)


@pytest.mark.parametrize('cases', [
    ['lgssm_qr', 'nonlinear', 'funnel_residual'],
    ['nonlinear'], ['funnel_residual'],
    ['funnel_residual', 'lgssm_qr', 'nonlinear'],
])
def test_development_seed_is_independent_of_case_subset_and_order(cases):
    expected = {'lgssm_qr': (20261002, 2501), 'nonlinear': (20261002, 2502),
                'funnel_residual': (20261002, 2503)}
    assert prices.development_seeds(cases) == {case: expected[case] for case in cases}


@pytest.mark.parametrize('cases', [[], ['nonlinear', 'nonlinear'], ['unknown']])
def test_invalid_development_inventory_fails_before_launch(cases):
    with pytest.raises(ValueError):
        prices.development_seeds(cases)


def test_price_budget_includes_each_full_closeout():
    prices.validate_price_budget(['lgssm_qr'],1800,900,2800)
    with pytest.raises(ValueError,match='including closeout'):
        prices.validate_price_budget(['lgssm_qr'],1800,900,2000)
    with pytest.raises(ValueError,match='including closeout'):
        prices.validate_price_budget(['lgssm_qr','nonlinear'],1800,900,5399)
    with pytest.raises(ValueError,match='caps'):
        prices.validate_price_budget(['lgssm_qr'],1800,0,2800)
