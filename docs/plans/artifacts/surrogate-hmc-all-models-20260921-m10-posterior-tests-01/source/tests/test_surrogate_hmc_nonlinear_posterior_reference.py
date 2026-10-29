"""Independent analytical checks for M10 posterior-reference arithmetic."""
import math

import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_nonlinear_quadrature import make_reference
from bayesfilter.testing.surrogate_hmc_nonlinear_posterior_reference import (
    make_summary,cdf_at,evidence_lower_bound,omission_allowance,PRIOR_SD)


def test_partial_legendre_integrals_and_normal_posterior():
    points=tf.constant([[-1.,-.3,0.,.4,1.]],tf.float64)
    uniform=cdf_at(tf.constant([[.5,0.,0.]],tf.float64),points)
    tf.debugging.assert_near(uniform,(points+1)/2,atol=1e-14)
    # A constant likelihood leaves exactly the normal prior (up to 9-sigma tails).
    _,summary=make_summary(64)
    means=tf.zeros([2],tf.float64)
    scales=tf.constant([PRIOR_SD]*2,tf.float64)
    cuts=tf.constant([[-1.2815515655446004,0.,1.2815515655446004]]*2,tf.float64)*PRIOR_SD
    result=summary(tf.ones([64,64],tf.float64),
        tf.broadcast_to(tf.constant([.25,.75],tf.float64),[64,64,2]),means,scales,cuts)
    assert abs(float(result['evidence'])-1)<1e-12
    tf.debugging.assert_near(result['means'],means,atol=1e-12)
    tf.debugging.assert_near(result['scales'],scales,atol=1e-12)
    tf.debugging.assert_near(result['expectations'],
        tf.constant([0.,0.,1.,1.,0.,.1,.5,.9,.1,.5,.9,.25,.75],tf.float64),atol=1e-10)
    tf.debugging.assert_near(result['cuts'],cuts,atol=1e-9)
    assert bool(result['interpolation_valid'])


def test_value_only_likelihood_parity_and_positive_evidence_bound():
    data=tf.constant([[.5424982309341431],[2.675233840942383]],tf.float64)
    point=tf.zeros([2],tf.float64)
    score=make_reference()(point,data)
    value=make_reference(value_only=True)(point,data)
    assert 'score' not in value and bool(value['valid'])
    assert abs(float(value['likelihood'])-math.exp(float(score['log_likelihood'])))<1e-14
    predictive=value['predictive_integrals']
    assert 0<float(predictive[0])<float(predictive[1])<float(value['likelihood'])
    lower=evidence_lower_bound(data)
    assert 0<lower['lower_bound']<1e-2
    bounds=omission_allowance([0.,0.],[PRIOR_SD]*2,[0.]*13,lower['lower_bound'])
    assert 0<bounds['posterior_omitted_mass_bound']<1e-3
    assert all(math.isfinite(v) and v>=0 for v in bounds['function_allowances'])
