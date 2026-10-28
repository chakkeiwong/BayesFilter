"""Lock scalar counterfactual roles and corruption detection."""
import pytest
from scripts.diagnostics.audit_modern_hmc_saved_tail import (
    check_interval, classify, same_multiplier_counterfactual,
)

def fixture():
    law=dict(posterior_truth=0.,fixed_expectation=0.,fixed_variance=1.)
    interval=dict(estimate=2.5,truth=0.,mcse=.5,multiplier=3.,width=3.,
                  estimated_to_exact_variance=.25,available=True,covered=False,
                  direction="above",transient_covered=False)
    return interval,law

def test_counterfactual_changes_variance_only_and_retains_original():
    interval,law=fixture();original=interval.copy()
    assert check_interval(interval,law,2.5)=="above"
    assert same_multiplier_counterfactual(interval,law)=="covered"
    assert interval==original
    assert classify(2.5,0.,1.959963984540054)=="above"

def test_unavailable_is_never_promoted_by_oracle_counterfactual():
    interval,law=fixture();interval.update(available=False,covered=None)
    assert check_interval(interval,law,2.5) is None
    assert same_multiplier_counterfactual(interval,law) is None

@pytest.mark.parametrize("field",["width","mcse","covered","direction","truth"])
def test_corrupt_interval_is_rejected(field):
    interval,law=fixture()
    interval[field]={"width":2.,"mcse":2.,"covered":True,"direction":"below","truth":1.}[field]
    with pytest.raises(ValueError):check_interval(interval,law,2.5)
