"""Independent numerical checks for finite-box posterior diagnostics."""
import math
import pytest
import tensorflow as tf
from bayesfilter.testing.surrogate_hmc_m13_same_data_posterior import (
    PLAN, PRIOR_SD, RADIUS, posterior_integrator, precision_screen)
from bayesfilter.inference.surrogate_hmc_closure_program import jobs, PASS_CAPS


def test_constant_likelihood_recovers_truncated_prior_and_cancels_common_scale():
    _,integrate=posterior_integrator(65,RADIUS,4)
    scales=tf.math.log(tf.constant([1.,2.,4.,8.],tf.float64))
    result=integrate(tf.broadcast_to(scales,[65,65,4]))
    expected=PRIOR_SD*math.sqrt(1-8*math.exp(-8)/math.sqrt(2*math.pi)/math.erf(4/math.sqrt(2)))
    assert bool(result['finite'])
    assert abs(float(result['metrics'][0]))<1e-12
    assert float(result['metrics'][2])==pytest.approx(expected,abs=1e-7)
    assert float(tf.reduce_max(result['mcse'][:4]))<1e-12
    assert float(result['log_evidence_mcse'])>0.


def test_ratio_influence_matches_perturbing_replicate_likelihood_weight():
    axis,integrate=posterior_integrator(17,RADIUS,4)
    slopes=tf.constant([-.4,.2,.6,1.],tf.float64)
    logs=tf.broadcast_to(axis[:,None,None]*slopes[None,None,:],[17,17,4])
    logs+=tf.constant([0.,1.,-.5,.7],tf.float64)
    original=integrate(logs)
    epsilon=1e-5
    shift=tf.constant([0.,0.,epsilon,0.],tf.float64)
    derivative=(integrate(logs+shift)['metrics']-integrate(logs-shift)['metrics'])/(2*epsilon)
    tf.debugging.assert_near(derivative,original['influence'][2]/4,atol=1e-9,rtol=1e-7)


def test_precision_screen_detects_material_mean_shift():
    def row(shift):
        return dict(metrics=tf.constant([shift,0.,1.,1.,1.,1.,0.],tf.float64),
                    mcse=tf.zeros([7],tf.float64),finite=tf.constant(True))
    original=row(0.)
    grid=[(original,original,tf.zeros([7],tf.float64))]
    assert precision_screen(row(.01),original,grid)['passed']
    assert not precision_screen(row(.1),original,grid)['passed']


def test_master_registration_preserves_separate_diagnostic_budget():
    selected=jobs('m13-same-data-posterior')
    assert [x['name'] for x in selected]==['review','m13-same-data-posterior']
    assert selected[1]['plan']==PLAN
    assert selected[1]['device']=='gpu'
    assert PASS_CAPS['m13-same-data-posterior']==dict(cpu_worker_seconds=2500.,gpu_seconds=3000.)
