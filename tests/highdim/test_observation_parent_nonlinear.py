"""CPU-only independent density and finite-program derivative diagnostics."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import math
from pathlib import Path
import sys
import pytest
import tensorflow as tf
from bayesfilter.highdim.analytical_jet_tf import Jet, logsumexp
from bayesfilter.highdim.covariance_proposal_native_models_tf import NativeModels
from bayesfilter.highdim.covariance_proposal_score_batch_tf import gaussian_mixture, make_batched_score_filter
from bayesfilter.highdim.covariance_proposal_tf import ProposalControls
from bayesfilter.highdim.observation_parent_proposal_tf import conditional_geometry, gaussian_log_density
from bayesfilter.highdim.sqmc_ksc_tf import KSCSpec
D=tf.float64
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))

def setup_model(kind, n=16):
    if kind=='ksc':
        spec=KSCSpec(); binding=spec; theta=spec.default_theta(D)
        initial=tf.random.stateless_normal([n,spec.dimension],[281301,11],dtype=D)
        noise=tf.random.stateless_normal([2,n,spec.dimension],[281301,12],dtype=D)
        uniform=tf.random.stateless_uniform([2,n,2],[281301,13],dtype=D)
    else:
        from run_range_bearing_covariance_comparison import specs
        _,binding=specs(tf,D,20)
        spec=binding
        theta=tf.constant([0.,0.],D)
        from run_range_bearing_covariance_comparison import inputs
        initial,noise,uniform,_=inputs(tf,binding.spec,n,281301,D)
        noise,uniform=noise[:2],uniform[:2]
    # Make the diagnostic self-contained: observations are deterministic
    # predictions from the same model/cloud, rather than external run data.
    model=NativeModels(spec,theta[None])
    state=model.initial_cloud(initial)
    observed=[]
    for _ in range(2):
        state=model.transition(state)
        observed.append(tf.reduce_mean(model.observation(state).v,axis=1)[0])
    return spec,theta,tf.stack(observed),initial,noise,uniform

@pytest.mark.parametrize('kind',['ksc','range_bearing'])
def test_exact_density_and_all_tangents(kind):
    spec,theta,obs,initial,noise,_=setup_model(kind)
    model=NativeModels(spec,theta[None]); n=initial.shape[0]
    means=model.transition(model.initial_cloud(initial))
    cm,cc,lc,alpha,z,_,valid=conditional_geometry(model,means,obs[0])
    tf.debugging.assert_equal(valid,[True])
    tf.debugging.assert_near(tf.reduce_sum(tf.exp(alpha.v),1),tf.ones([1],D),atol=1e-12)
    x=means+model.c(noise[0][None])
    direct=logsumexp(gaussian_log_density(x[:,:,None]-cm[:,None],lc[:,None],shared=False)+alpha[:,None],-1)
    if kind=='ksc':
        prior,pv=gaussian_mixture(x,means,model.Q[:,None],shared_covariance=True)
        actual=model.observation_density(x,obs[0])+prior-z[:,None]
    else:
        actual,pv=gaussian_mixture(x,cm,cc,log_weights=alpha)
    tf.debugging.assert_equal(pv,[True])
    tf.debugging.assert_near(direct.v,actual.v,rtol=1e-9,atol=1e-9)
    tf.debugging.assert_near(direct.t,actual.t,rtol=1e-8,atol=1e-8)

@pytest.mark.parametrize('kind',['ksc','range_bearing'])
@pytest.mark.parametrize('policy',['conditional_uniform','conditional_observation_weighted'])
def test_full_program_analytical_score_matches_finite_differences(kind,policy):
    spec,theta,obs,initial,noise,uniform=setup_model(kind,32)
    h=1e-6
    offsets=tf.concat([tf.zeros([1,2],D),h*tf.eye(2,dtype=D),-h*tf.eye(2,dtype=D)],0)
    controls=ProposalControls(flow_steps=2,reset_steps=5,proposal_policy=policy)
    fn=make_batched_score_filter(spec,32,2,controls,D,False,5)
    out=fn(theta[None]+offsets,obs,initial,noise,uniform,tf.constant([1/3]*3,D))
    tf.debugging.assert_equal(out['valid'],tf.ones([5],tf.bool))
    for j in range(2):
        tf.debugging.assert_equal(out['diagnostic_parent_indices'][0],out['diagnostic_parent_indices'][1+j])
        tf.debugging.assert_equal(out['diagnostic_parent_indices'][0],out['diagnostic_parent_indices'][3+j])
        fd=(out['value'][1+j]-out['value'][3+j])/(2*h)
        tf.debugging.assert_near(fd,out['score'][0,j],atol=2e-4,rtol=2e-4)
