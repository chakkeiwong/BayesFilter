"""Independent CPU-only reference checks for the scalar intervention."""
from dataclasses import replace
import math

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import neutra_transport_core as core
from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig
from bayesfilter.testing.neutra_nonlinearity_attribution import configuration


@pytest.mark.parametrize('conditioner',['author_cmade','diagnostic_hoffman_made'])
@pytest.mark.parametrize('affine_readout',[True,False])
def test_paired_arrays_and_nontrivial_derivatives(conditioner,affine_readout):
    cfg=replace(configuration(2,409,conditioner,width=8),
                inverse_atol=1e-13,inverse_rtol=1e-13)
    nonlinear=NeuTraTransport(cfg)
    affine=NeuTraTransport(replace(cfg,diagnostic_naf_affine=True))
    assert nonlinear.parameter_state()==affine.parameter_state()
    if not affine_readout:
        affine=nonlinear
    for stage in affine.stages:
        for w in stage.weights[:-1]:
            w.assign(w*10.)
        if conditioner=='author_cmade':
            stage.projection[0].assign(stage.projection[0]*100.)
        else:
            stage.weights[-1].assign(stage.weights[-1]*3.)
    x=tf.constant([[-1.,.7],[2.,-1.3],[.2,.4]],tf.float64)
    y,ld=affine.forward_and_logdet(x)
    z,ild=affine.inverse_and_forward_logdet(y)
    np.testing.assert_allclose(z,x,atol=1e-11)
    np.testing.assert_allclose(ld,ild,atol=1e-11)
    h=1e-5
    jac=np.stack([((affine.forward(x+tf.one_hot(i,2,dtype=tf.float64)*h)
                  -affine.forward(x-tf.one_hot(i,2,dtype=tf.float64)*h))/(2*h)).numpy()
                  for i in range(2)],-1)
    np.testing.assert_allclose(ld,np.linalg.slogdet(jac)[1],atol=1e-8)
    # Independent directional finite difference of inverse-density parameters.
    variables=affine.trainable_variables
    directions=[tf.random.stateless_normal(v.shape,[19,i],dtype=tf.float64)*.02
                for i,v in enumerate(variables)]
    saved=[tf.identity(v) for v in variables]
    with tf.GradientTape() as tape:
        loss=-tf.reduce_mean(affine.log_prob(y))
    gradients=tape.gradient(loss,variables)
    assert all(g is not None for g in gradients)
    exact=sum(float(tf.reduce_sum(g*d)) for g,d in zip(gradients,directions))
    # Use a tighter reference inverse so residual/h does not dominate the
    # difference. Small weight-normalization directions also require h small
    # enough to avoid large O(h^2) truncation error.
    estimates=[]
    for gradient_h in (1e-4,3e-5,1e-5):
        values=[]
        for sign in (-1.,1.):
            for v,s,d in zip(variables,saved,directions):v.assign(s+sign*gradient_h*d)
            values.append(float(-tf.reduce_mean(affine.log_prob(y))))
        estimates.append((values[1]-values[0])/(2*gradient_h))
    for v,s in zip(variables,saved):v.assign(s)
    np.testing.assert_allclose(estimates,exact,atol=1e-7,rtol=1e-5)
    stage=nonlinear.stages[0]
    _,offsets,_=stage.pseudo_parameters(x)
    assert float(tf.math.reduce_std(offsets[:,0,:]))>0


def test_scalar_affine_formula_and_gaussian_lower_bound():
    x=tf.constant([-.4,.9],tf.float64)
    a=tf.constant([[.3,2.],[1.4,.8]],tf.float64)
    b=tf.constant([[1.,-.3],[.1,-2.]],tf.float64)
    logits=tf.constant([[.2,.9],[-.8,.4]],tf.float64)
    y,ld,_=core.affine_mixture(x,tf.math.log(a),b,logits)
    w=tf.nn.softmax(logits)
    np.testing.assert_allclose(y,tf.reduce_sum(w*(a*x[:,None]+b),-1),atol=1e-14)
    np.testing.assert_allclose(tf.exp(ld),tf.reduce_sum(w*a,-1),atol=1e-14)
    assert .5*math.log(10)-math.log(2)>.458


def test_existing_payload_unchanged_and_diagnostic_flag_roundtrips():
    cfg=configuration(2,409,'author_cmade')
    assert 'diagnostic_naf_affine' not in cfg.payload()
    assert NeuTraTransportConfig(**cfg.payload())==cfg
    diagnostic=replace(cfg,diagnostic_naf_affine=True)
    assert NeuTraTransportConfig(**diagnostic.payload())==diagnostic


def test_confirmation_sign_test_uses_training_pairs_and_multiplicity():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/neutra_attribution_campaign.py'
    spec=importlib.util.spec_from_file_location('attribution_test_controller',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    results=[{'paired_kl_benefit':{'mean':float(i+1),'normal_approximation_99_lower':.1}} for i in range(8)]
    result=module.sign_summary(results)
    assert result['passed'] and result['one_sided_sign_p']==1/256
    assert result['median_effect']==4.5
    results[0]['paired_kl_benefit']['normal_approximation_99_lower']=-.1
    assert not module.sign_summary(results)['passed']
    assert module.sign_summary([])['bonferroni_four_p']==1.
