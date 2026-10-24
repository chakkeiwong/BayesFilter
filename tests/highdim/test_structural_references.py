"""Independent reference diagnostics; no inference/runtime imports."""
import math
import numpy as np
from bayesfilter.testing.structural_reference_diagnostics import (
    model_b_quadrature, quadrature_value_score, replicated_reference)


def test_quadrature_linear_limit_matches_exact_kalman():
    q = np.array([math.atanh(.7), math.log(.25), -40.])
    y = np.array([.2, -.1, .3])
    f = np.diag([.7, .55])
    h = np.ones(2)
    m, p, total = np.zeros(2), np.diag([.25, .20]), 0.
    for obs in y:
        m, p = f@m, f@p@f.T + np.diag([.25**2, 0.])
        s, r = h@p@h+.09, obs-h@m
        total -= .5*(math.log(2*math.pi*s)+r*r/s)
        gain = p@h/s
        m, p = m+gain*r, p-np.outer(gain,h@p)
    assert abs(model_b_quadrature(q,y,32)-total)<1e-8


def test_path_score_reference_agrees_with_short_quadrature_with_mcse():
    q = np.array([math.atanh(.7), math.log(.25), math.log(.8)])
    y = np.array([.2, -.1])
    low, high = [quadrature_value_score(q,y,k) for k in (24,32)]
    expected = np.array([high['value'], *high['score']])
    refinement = np.abs(expected-np.array([low['value'], *low['score']]))
    assert refinement.max()<3e-4
    reference = replicated_reference(q,y,16384,list(range(12)))
    error = np.abs(np.array(reference['mean'])-expected)
    assert np.all(error < 5*np.array(reference['mcse']) + refinement + .002)


def test_gaussian_references_match_linear_limit():
    from bayesfilter.testing.structural_reference_diagnostics import gaussian_reference
    q=[math.atanh(.7),math.log(.25),-40.];y=[.2,-.1,.3]
    exact=model_b_quadrature(q,y,32)
    for method in ('ekf','cubature'):
        assert abs(gaussian_reference('model_b',q,y,method)-exact)<1e-8


def test_posterior_grid_recovers_prior_without_observations():
    from bayesfilter.testing.structural_reference_diagnostics import kalman_posterior_grid
    mean=[-.9,-1.2];scale=[.7,.4]
    result=kalman_posterior_grid([],mean,scale,size=129)
    np.testing.assert_allclose(result['mean'],mean,atol=1e-9)
    np.testing.assert_allclose(result['sd'],scale,atol=1e-9)
    assert result['boundary_mass']<1e-10
