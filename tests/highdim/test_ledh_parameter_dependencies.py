"""Finite-program directional checks through the real model/batch boundary."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")
from dataclasses import replace
import tensorflow as tf
from bayesfilter.highdim.ledh_linear_parameter_adapter_tf import (
    stationary_covariance_and_tangent,linear_gaussian_parameter_model)
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import canonical_batch_fused_value_score


def test_stationary_covariance_requires_stability_and_spd_separately():
    dtype=tf.float64
    p,dp,status=stationary_covariance_and_tangent(
        tf.constant([[.8]],dtype),tf.constant([[1.]],dtype),
        tf.constant([[1.]],dtype),tf.constant([[0.]],dtype))
    tf.debugging.assert_near(p,tf.constant([[1/(1-.8**2)]],dtype),atol=1e-12)
    tf.debugging.assert_near(dp,tf.constant([[1.6/(1-.8**2)**2]],dtype),atol=1e-11)
    assert bool(status["value_valid"])
    _,_,failed=stationary_covariance_and_tangent(
        tf.constant([[1.1]],dtype),tf.constant([[1.]],dtype),
        tf.constant([[1.]],dtype),tf.constant([[0.]],dtype))
    assert not bool(failed["value_valid"])
    assert float(failed["stability_norm_squared"])>1
    # The unconstrained algebraic solve has zero residual and a negative P.
    algebraic=1/(1-1.1**2)
    assert algebraic<0 and abs(algebraic-1.1**2*algebraic-1)<1e-12


def test_real_consumer_initial_q_r_observation_and_jacobian_dependencies():
    dtype=tf.float64;d=2;n=12;t=2
    a=tf.constant([[.7,.2],[.1,.6]],dtype)
    q=tf.constant([[1.,.3],[.3,.8]],dtype)
    h=tf.constant([[1.,.2],[0.,1.]],dtype)
    r=.5*tf.eye(d,dtype=dtype)
    base=linear_gaussian_parameter_model(a,q,h,r,infer_process_scale=True)
    def initial(theta):
        mean,cov=base.initial_distribution_fn(theta)
        return mean+theta[2]*tf.constant([.3,-.2],dtype),cov
    def dinitial(theta,direction):
        mean,cov=base.initial_distribution_tangent_fn(theta,direction)
        return mean+direction[2]*tf.constant([.3,-.2],dtype),cov
    model=replace(base,
        initial_distribution_fn=initial,initial_distribution_tangent_fn=dinitial,
        observation_covariance_fn=lambda theta:tf.exp(2*theta[2])*r,
        observation_covariance_tangent_fn=lambda theta,v:2*v[2]*tf.exp(2*theta[2])*r,
        parameter_observation_fn=lambda theta,x:tf.exp(theta[:,2:3])*tf.linalg.matmul(x,h,transpose_b=True),
        parameter_observation_tangent_fn=lambda theta,x,dx,v:tf.exp(theta[:,2:3])*
            (tf.linalg.matmul(dx,h,transpose_b=True)+v[:,2:3]*tf.linalg.matmul(x,h,transpose_b=True)),
        parameter_observation_jacobian_fn=lambda theta,x:tf.exp(theta[:,2,None,None])*h[None],
        parameter_observation_jacobian_tangent_fn=lambda theta,x,dx,v:
            v[:,2,None,None]*tf.exp(theta[:,2,None,None])*h[None])
    states=tf.random.stateless_normal([n,d],[7041,1],dtype=dtype)
    covs=tf.eye(d,batch_shape=[n],dtype=dtype)
    noise=tf.random.stateless_normal([t,n,d],[7041,2],dtype=dtype)
    obs=tf.random.stateless_normal([t,d],[7041,3],dtype=dtype)
    design=tf.tile(tf.concat([tf.eye(d,dtype=dtype),-tf.eye(d,dtype=dtype)],0),[3,1])
    controls=dict(substeps=4,reset_policy="contract_e",reset_design=design,
        reset_sinkhorn_steps=4,reset_balance_steps=4,reset_ridge=1e-5,
        correction_steps=1,pairwise_steps=1,correction_trust_radius=.1,coordinate_cap=.98)
    directions=tf.constant([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[.3,-.2,.4]],dtype)
    @tf.function(input_signature=[tf.TensorSpec([3],dtype)],jit_compile=True)
    def evaluate(theta):
        return canonical_batch_fused_value_score(model,theta[None],directions[None],
            states,covs,noise,obs,**controls)
    theta=tf.constant([1.,.1,-.1],dtype)
    _,score,status=evaluate(theta)
    assert bool(status["program_valid"][0])
    for index,direction in enumerate(tf.unstack(directions)):
        errors=[]
        for step in [3e-4,1e-4,3e-5]:
            plus=evaluate(theta+step*direction)
            minus=evaluate(theta-step*direction)
            assert bool(plus[2]["program_valid"][0]) and bool(minus[2]["program_valid"][0])
            fd=(plus[0][0]-minus[0][0])/(2*step)
            errors.append(float(tf.abs(fd-score[0,index])/(1+tf.abs(fd))))
        assert max(errors)<3e-4, (index,errors)
    assert evaluate.experimental_get_tracing_count()==1
