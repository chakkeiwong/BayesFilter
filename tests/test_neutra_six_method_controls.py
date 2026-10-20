"""Small independent CPU references; never GPU training or posterior evidence."""
import dataclasses
import math
import os

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')

import pytest
import tensorflow as tf

from bayesfilter.testing.neutra_generic_targets import (
    ExactTargetEvaluator, fixed_specification, random_mixture_specification)
from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget
from bayesfilter.testing.neutra_six_method_controls import (
    IsotropicProposal, flow_config, initialize_affine_fixture, affine_fab_integrability)
from bayesfilter.inference.neutra_transport import NeuTraTransport
from bayesfilter.inference.neutra_warm_start_tf import AnnealedSMC, SMCConfig
from bayesfilter.inference.neutra_fab import FABConfig,FABTrainer,fab_weighted_loss,replay_log_correction

F64=tf.float64


@pytest.mark.parametrize('name',['gaussian','mixture','warped_mixture'])
def test_parameterized_density_matches_independent_existing_fixture(name):
    target=ExactTargetEvaluator(fixed_specification(name),jit_compile=False).target
    old=WarmStartTarget(name,jit_compile=False)
    x=tf.constant([[-5.,.2],[.7,-.4],[5.,2.]],F64)
    tf.debugging.assert_near(target.log_prob(x),old.log_prob(x),atol=1e-12,rtol=1e-12)
    tf.debugging.assert_near(target.value_score(x)[1],old.value_score(x)[1],atol=1e-11,rtol=1e-11)
    for field in ['centers','weights','specification','reference_sample','known_representatives','region_features','name']:
        assert not hasattr(target,field),field


@pytest.mark.parametrize('components',[2,3])
def test_random_design_bounds_and_density_gradient(components):
    spec=random_mixture_specification(components,191)
    assert spec==random_mixture_specification(components,191)
    assert spec!=random_mixture_specification(components,192)
    for i in range(components):
        for j in range(i):
            d=math.dist(spec['centers'][i],spec['centers'][j])
            assert 6-1e-12<=d<=10+1e-12
    assert all(.5<=v<=2 for v in spec['variances'])
    assert all(w>=.1 for w in spec['weights'])
    assert abs(sum(spec['weights'])-1)<1e-14
    target=ExactTargetEvaluator(spec,jit_compile=False).target
    x=tf.constant([[.1,.4],[-1.,2.]],F64)
    _,score,valid=target.value_score(x)
    assert bool(tf.reduce_all(valid))
    for j in range(2):
        delta=tf.one_hot(j,2,dtype=F64)*1e-5
        fd=(target.log_prob(x+delta)-target.log_prob(x-delta))/2e-5
        tf.debugging.assert_near(score[:,j],fd,atol=1e-8,rtol=1e-8)


def test_responsibility_identity_at_points_for_unequal_width_three_component_target():
    evaluator=ExactTargetEvaluator(random_mixture_specification(3,13),jit_compile=False)
    x=tf.constant([[0.,0.],[2.,-1.],[-4.,3.]],F64)
    spec=evaluator.specification
    densities=[]
    for row in x.numpy().tolist():
        parts=[w*math.exp(-sum((a-b)**2 for a,b in zip(row,mu))/(2*v))/(2*math.pi*v)
               for w,mu,v in zip(spec['weights'],spec['centers'],spec['variances'])]
        densities.append([v/sum(parts) for v in parts])
    tf.debugging.assert_near(evaluator.feature_program(x)[:,:3],tf.constant(densities,F64),atol=1e-12,rtol=1e-12)


def fixed_config(**kwargs):
    return dataclasses.replace(SMCConfig(32,2,4,.8,.5,.1,jit_compile=False,
        temperature_schedule=(0.,.25,.5,.75,1.)),**kwargs)


def test_ais_mutates_at_every_stage_without_resampling_and_preserves_exact_normalizer():
    proposal=IsotropicProposal(2,1.)
    class Target:
        parameter_dim=2
        log_prob=proposal.log_prob
    sampler=AnnealedSMC(Target(),proposal,fixed_config(use_resampling=False))
    seed=tf.constant([81,3]);initial=proposal.sample(32,seed)
    result=sampler.run(seed)
    assert result['complete'] and result['method']=='ais'
    assert all(row['mutation_steps']==2 and not row['resampled'] for row in result['stages'])
    assert float(tf.reduce_max(tf.abs(initial-result['particles'])))>1e-3
    tf.debugging.assert_near(result['log_normalizer'],tf.constant(0.,F64),atol=1e-12)
    tf.debugging.assert_near(result['log_weights'],tf.fill([32],tf.constant(-math.log(32),F64)),atol=1e-12)
    paired=AnnealedSMC(Target(),proposal,fixed_config(use_resampling=True)).run(seed)
    tf.debugging.assert_equal(result['particles'],paired['particles'])
    tf.debugging.assert_equal(result['log_weights'],paired['log_weights'])


@pytest.mark.parametrize('kwargs',[{'use_resampling':False,'temperature_schedule':()},
    {'temperature_schedule':(0.,.5,.4,1.)},{'temperature_schedule':(0.,.9)},
    {'temperature_schedule':(0.,float('nan'),1.)},{'waste_free':True}])
def test_invalid_ais_schedule_rejected(kwargs):
    with pytest.raises(ValueError): fixed_config(**kwargs)


def normal(x):
    return -.5*tf.reduce_sum(x*x,1)-math.log(2*math.pi),-x,tf.reduce_all(tf.math.is_finite(x),1)


def fab_config(**kwargs):
    return dataclasses.replace(FABConfig(8,2,2,1,.1,.001,.9,.999,1e-8,
        0,0,1,None,None,False,.65,1.02,jit_compile=False),**kwargs)


def test_fab_detached_gradient_and_posterior_correction_identity():
    logq=tf.Variable([-.2,-1.,-2.],dtype=F64)
    lw=tf.constant([1.,2.,3.],F64)
    with tf.GradientTape() as tape: loss=fab_weighted_loss(logq,lw)
    tf.debugging.assert_near(tape.gradient(loss,logq),-tf.nn.softmax(lw)/3,atol=1e-14)
    logp=tf.constant([-3.,-1.,-.5],F64)
    logg=2*logp-logq
    tf.debugging.assert_near(logg+(logq-logp),logp,atol=1e-14)
    tf.debugging.assert_near(replay_log_correction(logq+.7,logq),tf.fill([3],tf.constant(.7,F64)),atol=1e-14)


def test_fab_integrability_and_exact_resume_includes_rng_replay_and_optimizer():
    flow=NeuTraTransport(flow_config(2,31))
    initialize_affine_fixture(flow,scale=.5)
    assert affine_fab_integrability(flow,1.)['applicability']=='mathematically_inapplicable'
    initialize_affine_fixture(flow,scale=2.)
    assert affine_fab_integrability(flow,1.)['applicability']=='eligible'
    config=fab_config(replay_capacity=32,replay_min_size=16,updates_per_pass=2,correction_clip=10.)
    trainer=FABTrainer(flow,normal,config,target_signature='c'*64,seed=(31,61))
    for _ in range(4): trainer.step()
    checkpoint=trainer.checkpoint()
    other=FABTrainer(NeuTraTransport(flow_config(2,31)),normal,config,target_signature='c'*64,seed=(31,61))
    other.restore(checkpoint)
    a,b=trainer.step(),other.step()
    tf.debugging.assert_equal(a['x'],b['x'])
    assert trainer.checkpoint()==other.checkpoint()


def test_parameterized_target_matches_batched_evaluation_and_scalar_finite_difference():
    target=ExactTargetEvaluator(random_mixture_specification(3,991),jit_compile=False).target
    rows=tf.constant([[.1,.2],[1.,2.],[3.,-4.]],F64)
    batch=target.value_score(rows)
    for i in range(3):
        single=target.value_score(rows[i:i+1])
        tf.debugging.assert_near(batch[0][i:i+1],single[0],atol=1e-12)
        tf.debugging.assert_near(batch[1][i:i+1],single[1],atol=1e-12)
