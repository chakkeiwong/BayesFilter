"""Reference and adversarial checks for replicated-trial acceptance inference."""
from dataclasses import replace
from fractions import Fraction
import itertools
import math

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_acceptance_statistics import (
    bounded_trial_intervals, replicated_acceptance_statistics,
)
from tests.test_hmc_acceptance_protocol import policy


def _policy():
    return replace(policy(), max_repetitions=1024)


def _report(rows, **kwargs):
    return replicated_acceptance_statistics(rows, policy=_policy(), stage="verification",
        evidence_rungs=(1,4,16,64,256), **kwargs)


def _exact_capital(values, mean, sign, bets=(1.,.5,.25,.125)):
    return sum((math.prod(Fraction(1)+sign*Fraction(b)*(Fraction(float(y))-Fraction(float(mean)))
                          for y in values) for b in bets), Fraction(0))/len(bets)


def test_small_support_optional_stopping_bound_by_exact_enumeration():
    # This reference uses rational arithmetic and the probability of every
    # Bernoulli path, not the implementation's interval or Monte Carlo seeds.
    threshold = Fraction(4)
    crossings = 0
    for path in itertools.product((0.,1.), repeat=8):
        crossed = any(_exact_capital(path[:n], .5, 1) >= threshold
                      or _exact_capital(path[:n], .5, -1) >= threshold
                      for n in range(1,9))
        crossings += crossed
    assert Fraction(crossings, 256) <= Fraction(1,2)  # Two sides, each eta=1/4.


@pytest.mark.parametrize("values", [[.7]*4, [.7]*64, [0.,1.]*32, [.001]*64, [.999]*64])
@pytest.mark.parametrize("jit", [False,True])
def test_betting_bounds_are_outside_exact_rational_crossing(values, jit):
    alpha=.01
    lo,hi=bounded_trial_intervals([[v] for v in values], sided_alpha=alpha,
        method="bounded_betting_mixture_v1", jit_compile=jit)
    lower,upper=float(lo[0]),float(hi[0])
    assert 0 <= lower <= upper <= 1
    threshold=1/Fraction(alpha)
    if lower > 0:
        assert _exact_capital(values,lower,1) >= threshold
    if upper < 1:
        assert _exact_capital(values,upper,-1) >= threshold


def test_hoeffding_matches_independent_bound_and_does_not_claim_zero_variance():
    values=np.full((64,2),.7)
    lo,hi=bounded_trial_intervals(values, sided_alpha=.025,
        method="bounded_hoeffding_rungs_v1")
    h=math.sqrt(math.log(1/.025)/(2*64))
    np.testing.assert_allclose(lo, [.7-h]*2, atol=2e-11,rtol=0)
    np.testing.assert_allclose(hi, [.7+h]*2, atol=2e-11,rtol=0)
    short=_report(np.full((4,4),.7))
    assert short["acceptance_decision"] == "inconclusive_evidence"
    assert short["pooled_interval"][1]-short["pooled_interval"][0] > .1


@pytest.mark.parametrize("means,expected", [
    ([.7]*4,"passed"), ([.60,.68,.72,.80],"passed"),
    ([.4,.8,.8,.8],"inconclusive_preparation"),
    ([.4]*4,"repair_step_lower"), ([.95]*4,"repair_step_higher"),
    ([.65]*4,"passed"), ([.75]*4,"passed"),
])
def test_distinct_start_and_boundary_decisions(means,expected):
    result=_report(np.tile(means,(1024,1)))
    assert result["acceptance_decision"] == expected
    np.testing.assert_allclose(result["start_means"],means,atol=1e-12)
    assert result["tuning_artifact_authority"] is False


def test_qualification_boundary_remains_inconclusive_with_correct_pooled_mean():
    # Mean .70; one start at the qualification boundary. No finite observed
    # repetition count proves the exact boundary mean is strictly interior.
    result=_report(np.tile([.55,.75,.75,.75],(1024,1)))
    assert result["acceptance_decision"] == "inconclusive_evidence"


def test_opposing_temporal_changes_are_visible_and_cannot_gate_admission():
    scores=np.full((256,4),.7)
    windows=np.tile([[.5,.5,.9,.9],[.9,.9,.5,.5],
                     [.5,.5,.9,.9],[.9,.9,.5,.5]],(256,1,1))
    plain=_report(scores)
    diagnostic=_report(scores,window_scores=windows)
    assert plain["acceptance_decision"] == diagnostic["acceptance_decision"] == "passed"
    assert diagnostic["temporal"]["supported_material_change_by_start"] == [True]*4
    assert diagnostic["temporal"]["admission_effect"] == "none"


def test_cross_start_covariance_and_labels_survive_permutation():
    values=np.array([[.65,.68,.71,.74],[.7,.73,.75,.8]]*32)
    result=_report(values)
    vector=np.column_stack([values.mean(axis=1),values])
    np.testing.assert_allclose(result["covariance_of_mean"],np.cov(vector.T,ddof=1)/64,atol=1e-15)
    perm=[3,1,0,2]
    other=_report(values[:,perm])
    assert other["acceptance_decision"]==result["acceptance_decision"]
    np.testing.assert_allclose(other["pooled_interval"],result["pooled_interval"],atol=1e-12)
    np.testing.assert_allclose(other["start_intervals"],np.array(result["start_intervals"])[perm],atol=1e-12)


def test_extra_candidates_and_looks_cannot_invent_precision():
    values=np.full((64,4),.7)
    p=_policy()
    a=replicated_acceptance_statistics(values,policy=p,stage="verification",evidence_rungs=(1,4,16))
    b=replicated_acceptance_statistics(values,policy=replace(p,max_candidates=100),
        stage="verification",evidence_rungs=(1,4,16))
    assert b["pooled_interval"][0]<=a["pooled_interval"][0]
    assert b["pooled_interval"][1]>=a["pooled_interval"][1]
    h=replace(p,method="bounded_hoeffding_rungs_v1")
    c=replicated_acceptance_statistics(values,policy=h,stage="verification",evidence_rungs=(1,4,16))
    d=replicated_acceptance_statistics(values,policy=h,stage="verification",evidence_rungs=(1,4,16,64))
    assert d["pooled_interval"][0]<=c["pooled_interval"][0]
    assert d["pooled_interval"][1]>=c["pooled_interval"][1]


@pytest.mark.parametrize("mutation",["nan","outside","incomplete","wrong_axis"])
def test_invalid_or_partial_trials_cannot_be_selected_away(mutation):
    values=np.full((64,4),.7)
    if mutation=="nan": values[3,1]=np.nan
    elif mutation=="outside": values[3,1]=1.001
    elif mutation=="incomplete": values=values[:63]
    else: values=values.T
    with pytest.raises(ValueError): _report(values)
