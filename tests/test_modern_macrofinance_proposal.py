"""Independent diagnostic proposal mechanics, using exact Gaussian oracles."""
import numpy as np
import pytest
from bayesfilter.testing.inference_validation.references.macrofinance_proposal import (
    objective_gradient,checked_curvature,local_proposal,preserve_local_coverage)
from bayesfilter.testing.inference_validation.references.gaussian_mixture_reference import validate_proposal


def test_gradient_and_curvature_match_correlated_gaussian():
    precision=np.array([[4.,1.],[1.,2.]])
    mean=np.array([1.,-2.])
    def target(x): return -.5*np.einsum('bi,ij,bj->b',x-mean,precision,x-mean)
    x=np.array([.3,.2])
    _,gradient=objective_gradient(target,x)
    np.testing.assert_allclose(gradient,precision@(x-mean),atol=1.e-8)
    factor,receipt=checked_curvature(target,mean)
    np.testing.assert_allclose(factor@factor.T,np.linalg.inv(precision),atol=1.e-9)
    assert receipt['relative_hessian_discrepancy']<1.e-8


def test_invalid_curvature_is_not_silently_repaired():
    with pytest.raises(ValueError,match='nonpositive'):
        checked_curvature(lambda x:x[:,0]**2,np.zeros(2))


def test_local_search_retains_prior_and_broad_coverage():
    mean=np.arange(9)/10
    def target(x): return -.5*np.sum((x-mean)**2,axis=1)
    proposal,guards,receipt=local_proposal(target,np.zeros(9))
    validate_proposal(proposal)
    assert receipt['distinct_stationary_modes']==1
    assert all(r['max_gradient']<1.e-3 for r in receipt['starts'])
    np.testing.assert_allclose(proposal['centers'][1],mean,atol=1.e-6)
    updated=preserve_local_coverage(proposal,guards)
    validate_proposal(updated)
    assert updated['broad_local_components']==1
    np.testing.assert_allclose(updated['factors'][-1],2*np.eye(9),atol=1.e-6)
    assert proposal['factors'][-1]!=updated['factors'][-1]
