"""Real canonical consumer regressions; CPU/GPU mechanics, never admission.

No numerical engine or batch wrapper is mocked. The constructed bad rows and
directions test numerical status, not scientific model support.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH','true')

from dataclasses import replace

import pytest
import tensorflow as tf
from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    PerPointScoreModel, canonical_batch_fused_value_score,
    canonical_batch_fused_value_score_whileloop, _single_cloud_model,
)
from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
from bayesfilter.inference.ledh_dual_parameter_target import DualParameterLEDHTarget


def fixture(dtype=tf.float64,horizon=3):
    n,d=8,2
    def transition(theta,x):
        return tf.where(theta[:,:1]>=0,theta[:,:1]*x,tf.constant(float('nan'),dtype))
    def tangent(theta,x,dx,dtheta):
        return tf.where(dtheta[:,:1]>=0,theta[:,:1]*dx+dtheta[:,:1]*x,
                        tf.constant(float('nan'),dtype))
    model=PerPointScoreModel(transition,tangent,lambda x:x,
        lambda x:tf.broadcast_to(tf.eye(d,dtype=dtype),[tf.shape(x)[0],d,d]),
        lambda x,dx:dx,.3*tf.eye(d,dtype=dtype),.6*tf.eye(d,dtype=dtype))
    states=tf.random.stateless_normal([n,d],[7031,1],dtype=dtype)
    covs=tf.eye(d,batch_shape=[n],dtype=dtype)
    noise=tf.random.stateless_normal([horizon,n,d],[7031,2],dtype=dtype)
    obs=tf.zeros([horizon,d],dtype)
    design=tf.tile(tf.concat([tf.eye(d,dtype=dtype),-tf.eye(d,dtype=dtype)],0),[2,1])
    controls=dict(substeps=2,reset_policy='contract_e',reset_design=design,
        reset_sinkhorn_steps=4,reset_balance_steps=4,reset_ridge=1e-5,
        correction_steps=1,pairwise_steps=1,correction_trust_radius=.1,
        coordinate_cap=.98)
    return model,(states,covs,noise,obs),controls


@pytest.mark.parametrize('mode',['eager','graph','xla'])
@pytest.mark.parametrize('dtype',[tf.float32,tf.float64])
def test_full_covariance_guard_batch_shape_and_scale(mode,dtype):
    matrices=tf.constant([[[[4.,1.],[1.,3.]]],[[[1.,float('nan')],[0.,1.]]],
        [[[float('inf'),0.],[0.,1.]]],[[[1.,2.],[2.,1.]]],[[[1e-20,0.],[0.,1e-20]]]],dtype)
    fn=safe_cholesky if mode=='eager' else tf.function(safe_cholesky,
        input_signature=[tf.TensorSpec([5,1,2,2],dtype)],jit_compile=mode=='xla')
    valid,factors=fn(matrices)
    assert valid.shape==[5,1]
    assert valid.numpy().tolist()==[[True],[False],[False],[False],[True]]
    tf.debugging.assert_all_finite(factors,'failed factor leaked')
    tf.debugging.assert_near(factors[1:4],tf.eye(2,batch_shape=[3,1],dtype=dtype))


@pytest.mark.parametrize('entry',[canonical_batch_fused_value_score,canonical_batch_fused_value_score_whileloop])
def test_real_full_feature_rows_and_direction_status(entry):
    model,inputs,controls=fixture()
    theta=tf.constant([[.6],[-.2],[.6]],tf.float64)
    directions=tf.constant([[[1.],[.5]],[[1.],[.5]],[[1.],[-1.]]],tf.float64)
    value,score,status=entry(model,theta,directions,*inputs,**controls)
    assert status['value_valid'].numpy().tolist()==[True,False,True]
    assert status['force_valid'].numpy().tolist()==[True,False,False]
    assert status['direction_force_valid'].numpy().tolist()==[[True,True],[False,False],[True,False]]
    assert float(value[1])==float('-inf')
    assert float(value[0])==float(value[2])
    tf.debugging.assert_equal(score[1:],tf.zeros([2,2],tf.float64))
    reference,force,_=entry(model,theta[:1],directions[:1],*inputs,**controls)
    tf.debugging.assert_equal(value[:1],reference)
    tf.debugging.assert_equal(score[:1],force)


@pytest.mark.parametrize('failure_time',[0,1,2])
def test_failure_is_cumulative_and_reports_first_time(failure_time):
    model,inputs,controls=fixture()
    states,covs,noise,obs=inputs
    obs=tf.tensor_scatter_nd_update(obs,[[failure_time,0]],[float('nan')])
    kwargs=dict(controls)
    kwargs['flow_substeps']=kwargs.pop('substeps')
    theta=tf.constant([.6],tf.float64)
    value,score,trace,status=canonical_value_and_analytical_score(
        _single_cloud_model(model,theta,tf.ones_like(theta)),theta,states,covs,noise,obs,
        with_score=True,return_status=True,return_trace=True,**kwargs)
    assert float(value)==float('-inf')
    tf.debugging.assert_equal(score,tf.zeros_like(score))
    assert int(status['first_failure_time'])==failure_time
    assert int(status['first_failure_stage'])==0
    for row in trace[failure_time:]:
        assert not bool(row['value_valid'])
        tf.debugging.assert_all_finite(row['states_after_reset'],'inactive state')
        tf.debugging.assert_all_finite(row['d_states_after_reset'],'inactive tangent')


def test_real_dual_target_invalid_value_and_custom_gradient():
    model,inputs,controls=fixture(horizon=1)
    target=DualParameterLEDHTarget(model,*inputs,exact_params=controls,biased_params=controls)
    theta=tf.constant([-.2],tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value=target(theta)
    gradient=tape.gradient(value,theta)
    assert float(value)==float('-inf')
    tf.debugging.assert_equal(gradient,tf.zeros_like(theta))
    _,_,status=target.value_score_and_status(theta)
    assert not bool(status['value_valid'])
    assert int(status['value_failure_code'])!=0


def test_actual_dual_force_failure_preserves_exact_value():
    model,inputs,controls=fixture(horizon=1)
    model=replace(model,transition_mean_tangent_fn=lambda theta,x,dx,dtheta:
        tf.fill(tf.shape(x),tf.constant(float("nan"),x.dtype)))
    target=DualParameterLEDHTarget(model,*inputs,exact_params=controls,biased_params=controls)
    theta=tf.constant([.6],tf.float64)
    value,force,status=target.value_score_and_status(theta)
    assert bool(status['value_valid'])
    assert not bool(status['force_valid'])
    assert bool(tf.math.is_finite(value))
    tf.debugging.assert_equal(force,tf.zeros_like(theta))
    tf.debugging.assert_equal(value,target.value_only(theta))


@pytest.mark.parametrize('jit',[False,True])
def test_real_compiled_consumer_returns_structured_status(jit):
    model,inputs,controls=fixture(horizon=1)
    # Full correction/reset remain enabled in the compiled consumer.
    target=DualParameterLEDHTarget(model,*inputs,exact_params=controls,biased_params=controls)
    fn=target.as_graph_value_score_callable(1,jit_compile=jit)
    healthy=fn(tf.constant([.6],tf.float64))
    failed=fn(tf.constant([-.2],tf.float64))
    assert bool(healthy[2]['program_valid'])
    assert float(failed[0])==float('-inf')
    tf.debugging.assert_equal(failed[1],tf.zeros([1],tf.float64))
    assert not bool(failed[2]['value_valid'])
    assert fn.experimental_get_tracing_count()==1
