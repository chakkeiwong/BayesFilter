"""Default call-chain, independent SIR geometry, and finite-program checks."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import tensorflow as tf
from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, make_filter
from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
from bayesfilter.highdim.covariance_proposal_score_tf import make_score_filter
from bayesfilter.highdim.covariance_proposal_native_models_tf import NativeModels
from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
from bayesfilter.highdim.observation_parent_proposal_tf import conditional_geometry
from bayesfilter.testing.observation_parent_score_diagnostic_tf import conditional_geometry as independent, make_hook
D=tf.float64

def test_generic_geometry_matches_independent_sir_derivation():
    model=NativeModels(NonlinearSQMCSpec('sir_d18'),tf.constant([[.1,-.2,-.7],[0.,0.,0.]],D))
    x=model.initial_cloud(tf.random.stateless_normal([32,18],[28,11],dtype=D))
    means=model.transition(x)
    y=tf.reduce_mean(model.observation(means).v,[0,1])+1.
    cm,cov,lc,alpha,z,_,valid=conditional_geometry(model,means,y)
    ref=independent(model,means,y)
    for actual,expected in [(cm,ref[0]),(cov,ref[2]),(alpha,ref[3]),(z,ref[4])]:
        tf.debugging.assert_near(actual.v,expected.v,atol=1e-10,rtol=1e-10)
        tf.debugging.assert_near(actual.t,expected.t,atol=1e-9,rtol=1e-9)
    tf.debugging.assert_equal(valid,[True,True])

def test_default_scalar_batch_directional_and_fixed_design_share_recurrence():
    assert ProposalControls().proposal_policy=='conditional_observation_weighted'
    spec=NonlinearSQMCSpec('sir_d18'); n=32; t=2
    controls=ProposalControls(flow_steps=2,reset_steps=5)
    initial=tf.random.stateless_normal([n,18],[28,21],dtype=D)
    noise=tf.random.stateless_normal([t,n,18],[28,22],dtype=D)
    uniform=tf.random.stateless_uniform([t,n,2],[28,23],dtype=D)
    model=NativeModels(spec,tf.zeros([1,3],D))
    obs=tf.repeat(tf.reduce_mean(model.observation(model.transition(model.initial_cloud(initial))).v,axis=1),t,axis=0)
    beta=tf.constant([1/3]*3,D); args=(obs,initial,noise,uniform,beta)
    theta=tf.zeros([1,3],D)
    batch=make_batched_score_filter(spec,n,t,controls,D,False,1)(theta,*args)
    old=ProposalControls(flow_steps=2,reset_steps=5,proposal_policy='original_three_branch')
    independent_result=make_batched_score_filter(spec,n,t,old,D,False,1,diagnostic_proposal_hook=make_hook([2],old))(theta,*args)
    fixed=make_batched_score_filter(spec,n,t,controls,D,False,1,fixed_draw_design=(beta,uniform))(theta,obs,initial,noise)
    scalar=make_score_filter(spec,n,t,controls,D,False)(theta[0],*args)
    directional=make_filter(spec,n,t,controls,D,False)(theta[0],tf.constant([1.,0.,0.],D),*args)
    tf.debugging.assert_equal(batch['valid'],[True])
    for key in ('value','score','particles','particle_tangent'):
        tf.debugging.assert_near(batch[key],independent_result[key],atol=2e-7,rtol=1e-8)
        tf.debugging.assert_near(batch[key],fixed[key],atol=1e-10,rtol=1e-10)
        tf.debugging.assert_near(batch[key][0],scalar[key],atol=1e-10,rtol=1e-10)
    tf.debugging.assert_near(directional['score'],batch['score'][0,0],atol=1e-10)
    h=1e-6
    offsets=tf.constant([[h,0.,0.],[-h,0.,0.]],D)
    fdout=make_batched_score_filter(spec,n,t,controls,D,False,2)(offsets,*args)
    tf.debugging.assert_equal(fdout['diagnostic_parent_indices'][0],fdout['diagnostic_parent_indices'][1])
    fd=(fdout['value'][0]-fdout['value'][1])/(2*h)
    tf.debugging.assert_near(fd,batch['score'][0,0],atol=2e-4,rtol=2e-4)



def test_backward_guide_requires_explicit_original_policy():
    import pytest
    from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
    with pytest.raises(ValueError, match='explicit original_three_branch'):
        make_filter(LGSSMSpec('diagonal_ar', 2), 12, 2, ProposalControls(),
                    jit_compile=False, backward_guide=object())
