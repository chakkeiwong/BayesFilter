"""Independent exact and analytic checks for diagnostic numerical scores."""
import numpy as np
import pytest
from bayesfilter.testing.quadratic_score_reference import (
    fit_quadratic_score, log_likelihood_ratios, log_weight_likelihood_ratios, quadratic_design,
)


def points(d=3,n=1024):
    z=np.random.default_rng(71).uniform(-1,1,(n//2,d))
    return np.concatenate([z,-z])


def test_exact_cross_terms_and_physical_score_units():
    z=points(); scales=np.array([.6,114.,.3]); radius=.04
    score=np.array([2.,-.03,7.]); hessian=np.array([[2.,.1,-3.],[.1,.05,1.],[-3.,1.,-5.]])
    delta=radius*z*scales
    values=.73+delta@score+.5*np.einsum('ni,ij,nj->n',delta,hessian,delta)
    result=fit_quadratic_score(z,values,scales,radius)
    np.testing.assert_allclose(result.score,score,rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(result.hessian,hessian,rtol=1e-9,atol=1e-9)
    assert result.intercept==pytest.approx(.73,abs=1e-12)


def test_regression_converges_to_analytic_marginal_gaussian_score():
    # x~N(mu,1), y|x~N(x,exp(eta)); integrating x gives variance 1+exp(eta).
    theta=np.array([.2,-.3]); y=np.array([.7,-.1,1.2]); z=points(2)
    def loglike(p):
        variance=1+np.exp(p[:,1])
        return -.5*(len(y)*np.log(2*np.pi*variance)+np.sum((y[None,:]-p[:,0,None])**2,axis=1)/variance)
    variance=1+np.exp(theta[1]); residual=y-theta[0]
    exact=np.array([residual.sum()/variance,.5*np.exp(theta[1])*(np.sum(residual**2)/variance**2-len(y)/variance)])
    errors=[]
    for h in [.08,.04,.02,.01,.005]:
        fitted=fit_quadratic_score(z,loglike(theta+h*z)-loglike(theta[None,:])[0],np.ones(2),h)
        errors.append(np.max(np.abs(fitted.score-exact)))
    assert all(new<.27*old for old,new in zip(errors,errors[1:]))
    assert errors[-1]<1e-5


def test_common_path_ratio_cancels_unknown_proposal_normalization():
    delta=np.array([[0.,0.,0.],[.2,-.1,.7]])
    weights=np.log([.2,.5,.3])
    value,ess=log_likelihood_ratios(delta,weights)
    shifted,ess2=log_likelihood_ratios(delta,weights+1000)
    np.testing.assert_allclose(value,[0,np.log(np.dot([.2,.5,.3],np.exp(delta[1])))],atol=1e-13)
    np.testing.assert_allclose(shifted,value,atol=1e-12)
    np.testing.assert_allclose(ess,ess2,atol=1e-12)


def test_nonidentifiable_design_fails_without_ridge():
    with pytest.raises(ValueError,match='rank-deficient'):
        fit_quadratic_score(np.ones((64,3)),np.zeros(64),np.ones(3),.1)
    with pytest.raises(ValueError,match='underdetermined'):
        fit_quadratic_score(points(3,4),np.zeros(4),np.ones(3),.1)


def test_finite_checks_and_coefficient_count():
    assert quadratic_design(points(6)).shape==(1024,28)
    with pytest.raises(ValueError,match='finite'):
        log_likelihood_ratios([[np.nan]], [0.])


def test_direct_weights_retain_zero_paths_and_allow_weight_revival():
    baseline = np.array([-np.inf, np.log(2.)])
    weights = np.array([baseline, [np.log(3.), np.log(2.)]])
    ratio, ess = log_weight_likelihood_ratios(weights, baseline)
    np.testing.assert_allclose(ratio, [0., np.log(2.5)], atol=1e-14)
    np.testing.assert_allclose(ess, [1., 25./13.], atol=1e-14)
    shifted, shifted_ess = log_weight_likelihood_ratios(weights+1000, baseline+1000)
    np.testing.assert_allclose(shifted, ratio, atol=1e-12)
    np.testing.assert_allclose(shifted_ess, ess, atol=1e-12)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_direct_weights_reject_invalid_or_all_zero_samples(bad):
    with pytest.raises(ValueError):
        log_weight_likelihood_ratios([[bad, bad]], [0., 0.])
    with pytest.raises(ValueError):
        log_weight_likelihood_ratios([[0., 0.]], [bad, bad])


def test_direct_and_difference_forms_agree_for_finite_densities():
    base = np.log([.2, .5, .3]); delta = np.array([[0., 0., 0.], [.2, -.1, .7]])
    direct = log_weight_likelihood_ratios(delta+base, base)
    difference = log_likelihood_ratios(delta, base)
    np.testing.assert_allclose(direct, difference, atol=1e-14)


def test_delete_groups_match_literal_common_path_integration():
    from bayesfilter.testing.quadratic_score_reference import delete_group_log_ratios
    base = np.log(np.arange(1, 13, dtype=float))
    weights = np.vstack([base+.2, base+np.linspace(-.1,.1,12)])
    actual = delete_group_log_ratios(weights, base, 3)
    for g in range(3):
        keep = np.r_[0:4*g, 4*(g+1):12]
        expected = np.log(np.exp(weights[:,keep]).sum(axis=1)/np.exp(base[keep]).sum())
        np.testing.assert_allclose(actual[:,g], expected, atol=1e-14)


def test_jackknife_mean_has_known_standard_error():
    from bayesfilter.testing.quadratic_score_reference import equal_group_jackknife_se
    x = np.array([[1., 3.], [2., -4.], [5., 9.], [4., 8.]])
    deleted = (x.sum(axis=0)-x)/(len(x)-1)
    np.testing.assert_allclose(equal_group_jackknife_se(deleted), x.std(axis=0,ddof=1)/np.sqrt(len(x)))


def test_delete_group_zero_remaining_support_rejects():
    from bayesfilter.testing.quadratic_score_reference import delete_group_log_ratios
    with pytest.raises(ValueError, match='all-zero'):
        delete_group_log_ratios(np.array([[0.,-np.inf,-np.inf,-np.inf]]),np.zeros(4),2)
