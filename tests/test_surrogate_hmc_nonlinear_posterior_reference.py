"""Independent analytical checks for M10 posterior-reference arithmetic."""
import math

import pytest
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_nonlinear_quadrature import (
    make_reference,predictive_integrals_valid)
from bayesfilter.testing.surrogate_hmc_nonlinear_posterior_reference import (
    make_summary,cdf_at,evidence_lower_bound,omission_allowance,PRIOR_SD)


@pytest.mark.parametrize('order',[64,96])
def test_partial_legendre_integrals_and_normal_posterior(order):
    points=tf.constant([[-1.,-.3,0.,.4,1.]],tf.float64)
    uniform=cdf_at(tf.constant([[.5,0.,0.]],tf.float64),points)
    tf.debugging.assert_near(uniform,(points+1)/2,atol=1e-14)
    # A constant likelihood leaves exactly the normal prior (up to 9-sigma tails).
    _,summary=make_summary(order)
    means=tf.zeros([2],tf.float64)
    scales=tf.constant([PRIOR_SD]*2,tf.float64)
    cuts=tf.constant([[-1.2815515655446004,0.,1.2815515655446004]]*2,tf.float64)*PRIOR_SD
    result=summary(tf.ones([order,order],tf.float64),
        tf.broadcast_to(tf.constant([.25,.75],tf.float64),[order,order,2]),means,scales,cuts)
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


def test_predictive_subnormal_rounding_guard():
    # Observed failure: independent reductions differed by one FP64 subnormal
    # increment. The guard must still reject substantive errors and nonfinites.
    likelihood=tf.constant(6.014e-320,tf.float64)
    assert bool(predictive_integrals_valid(likelihood,tf.constant([6.0143e-320]*2,tf.float64)))
    for wrong in ([.1,1.001],[-1e-15,.5],[float('nan'),.5],[float('inf'),.5]):
        assert not bool(predictive_integrals_valid(tf.constant(1.,tf.float64),tf.constant(wrong,tf.float64)))
    data=tf.constant([[.8160498738288879],[-.5775660872459412]],tf.float64)
    result=make_reference(32,128,value_only=True)(
        tf.constant([-3.3648781186133125,-3.192114385833288],tf.float64),data)
    assert bool(result['valid'])
    assert 0<=float(result['likelihood'])<1e-300
