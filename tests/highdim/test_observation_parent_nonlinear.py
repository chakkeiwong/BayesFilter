"""CPU-only independent density and finite-program derivative diagnostics."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import json
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
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
D=tf.float64

def setup_model(kind, n=16):
    from run_observation_parent_four_model_check import designs, OLD, RB
    if kind=='ksc':
        spec=KSCSpec(); fixture=None
        saved=json.loads((OLD/'004-filter-ksc-d26100831/manifest.json').read_text())
        theta=saved['theta']; data=saved['data']
    else:
        from run_range_bearing_covariance_comparison import specs
        fixture,spec=specs(tf,D,20)
        theta=[0.,0.];data=json.loads((RB/'data/1704-t20.json').read_text())
    initial,noise,uniform=designs(tf,spec,fixture,n,2,281301)
    return spec,tf.constant(theta,D),tf.constant(data['observations'][:2],D),initial,noise,uniform

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
