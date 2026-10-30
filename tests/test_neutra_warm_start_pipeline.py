"""Independent CPU reference checks; not scientific trained-map evidence."""
import json
import math
import os
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget, BroadStudentProposal
from bayesfilter.inference.neutra_warm_start_tf import (
    independence_step, mala_step, value_score, next_temperature,
    systematic_indices, transport_increment, AnnealedSMC, SMCConfig, discover_modes, valid_log_weights)


@pytest.mark.parametrize('name',['gaussian','mixture','warped_mixture','wiggle','funnel'])
def test_target_gradient_and_hessian_against_independent_finite_differences(name):
    target=WarmStartTarget(name,jit_compile=False)
    x=tf.constant([[.3+i*.1 for i in range(target.parameter_dim)],
                   [-.5+i*.07 for i in range(target.parameter_dim)]],tf.float64)
    val,grad,valid=target.value_score(x)
    h=1e-5
    fd=[]; hessian=[]
    for j in range(target.parameter_dim):
        step=tf.one_hot(j,target.parameter_dim,dtype=tf.float64)*h
        fd.append((target.log_prob(x+step)-target.log_prob(x-step))/(2*h))
        hessian.append((target.value_score(x+step)[1]-target.value_score(x-step)[1])/(2*h))
    np.testing.assert_allclose(grad,tf.stack(fd,axis=1),atol=2e-8,rtol=1e-7)
    np.testing.assert_allclose(target.hessian(x),tf.stack(hessian,axis=2),atol=2e-7,rtol=1e-6)
    assert bool(tf.reduce_all(valid))


def test_warp_density_and_exact_inverse():
    base=WarmStartTarget('mixture',jit_compile=False)
    warped=WarmStartTarget('warped_mixture',jit_compile=False)
    x=tf.constant([[-5.,.3],[5.,-.7],[0.,1.]],tf.float64)
    np.testing.assert_allclose(base.log_prob(x),warped.log_prob(warped.warp(x)),atol=1e-13)


def test_weighted_cess_is_not_cumulative_ess_and_resampling_has_correct_support():
    lw=tf.math.log(tf.constant([.9,.1],tf.float64))
    ratio=tf.constant([-2.,2.],tf.float64)
    beta=next_temperature(lw,ratio,tf.constant(0.,tf.float64),tf.constant(.8,tf.float64))
    increment=float(beta)*np.array([-2.,2.])
    ces=(np.sum(np.array([.9,.1])*np.exp(increment))**2/
         np.sum(np.array([.9,.1])*np.exp(2*increment)))
    assert abs(ces-.8)<1e-10
    idx=systematic_indices(tf.constant([-math.inf,0.],tf.float64),.7,4)
    np.testing.assert_array_equal(idx,[1,1,1,1])


def test_author_controlled_global_mala_trajectory():
    path=Path(os.environ.get('WARM_START_AUTHOR_FIXTURE',
        'docs/plans/artifacts/neutra-warm-start-master-2026-09-29/author-reference-r2.json'))
    assert path.is_file(),'generate the actual author fixture before parity tests'
    fixture=json.loads(path.read_text())
    def log_target(x): return -.5*((x[:,0]-.7)**2+3*(x[:,1]+.2)**2)
    def log_q(x): return -.5*tf.reduce_sum(x*x,axis=1)-math.log(2*math.pi)
    x=tf.constant(fixture['starts'],tf.float64)
    positions=[]; acceptances=[]
    for i in range(2):
        x,acc,_,_=independence_step(log_target,log_q,x,
            tf.constant(fixture['proposals'][i],tf.float64),
            tf.constant(fixture['uniforms'][2*i],tf.float64))
        x,_,_,_=mala_step(lambda y:value_score(log_target,y),x,
            tf.constant(fixture['noises'][i],tf.float64),
            tf.constant(fixture['uniforms'][2*i+1],tf.float64),tf.constant(fixture['dt'],tf.float64))
        positions.append(x); acceptances.append(acc)
    np.testing.assert_allclose(tf.stack(positions),fixture['positions'],atol=2e-14,rtol=2e-14)
    np.testing.assert_array_equal(tf.stack(acceptances),fixture['global_accepted'])


def test_wiggle_density_and_score_match_actual_pinned_author_class():
    fixture=json.loads(Path('docs/plans/artifacts/neutra-warm-start-master-2026-09-29/author-reference-r2.json').read_text())['wiggle']
    target=WarmStartTarget('wiggle',jit_compile=False)
    v,g,_=target.value_score(tf.constant(fixture['points'],tf.float64))
    np.testing.assert_allclose(v,fixture['log_density'],rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(g,fixture['score'],rtol=1e-13,atol=1e-13)


def test_funnel_density_normalization_matches_exact_noncentered_chart():
    target=WarmStartTarget('funnel',jit_compile=False)
    z=np.array([[-1.2]+[.3]*9,[.7]+[-.4]*9])
    physical=np.concatenate((z[:,:1],np.exp(z[:,:1])*z[:,1:]),axis=1)
    expected=-.5*np.sum(z*z,axis=1)-5*math.log(2*math.pi)-9*z[:,0]
    np.testing.assert_allclose(target.log_prob(tf.constant(physical,tf.float64)),expected,atol=1e-13)


def test_transport_correction_has_right_orientation_and_jacobian():
    class Affine:
        def forward_and_logdet(self,x):
            return x*2+1,tf.fill([tf.shape(x)[0]],tf.constant(2*math.log(2),tf.float64))
    def prior(x): return -.5*tf.reduce_sum(x*x,axis=1)-math.log(2*math.pi)
    def target(y): return prior((y-1)/2)-2*math.log(2)
    x=tf.constant([[-2.,1.],[.3,-.7]],tf.float64)
    _,increment=transport_increment(prior,target,Affine(),x)
    np.testing.assert_allclose(increment,0.,atol=1e-14)


@pytest.mark.parametrize('waste_free',[False,True])
def test_smc_identity_bridge_preserves_normalizer(waste_free):
    proposal=BroadStudentProposal(2)
    class Target:
        parameter_dim=2
        log_prob=proposal.log_prob
    smc=AnnealedSMC(Target(),proposal,SMCConfig(32,4,16,.8,.5,.01,waste_free,False))
    result=smc.run(tf.constant([31,47]))
    assert result['complete']
    assert result['stages'][-1]['beta']==1.
    np.testing.assert_allclose(result['log_normalizer'],0.,atol=1e-13)


def test_adaptive_smc_weights_telescope_to_direct_importance_sampling_without_resampling():
    target=WarmStartTarget('gaussian',jit_compile=False);proposal=BroadStudentProposal(2)
    smc=AnnealedSMC(target,proposal,SMCConfig(32,4,128,.8,.001,.01,False,False))
    seed=tf.constant([57,81]);initial=proposal.sample(32,seed)
    ratio=(target.log_prob(initial)-proposal.log_prob(initial)).numpy()
    result=smc.run(seed)
    assert result['complete'] and len(result['stages'])>1
    assert not any(row['resampled'] for row in result['stages'])
    expected=np.max(ratio)+np.log(np.exp(ratio-np.max(ratio)).mean())
    np.testing.assert_allclose(result['log_normalizer'],expected,atol=1e-12)
    np.testing.assert_allclose(result['log_weights'],ratio-expected-math.log(32),atol=1e-12)


def test_multistart_finds_two_modes_without_known_mode_labels():
    target=WarmStartTarget('mixture',jit_compile=False)
    starts=tf.constant([[-8.,3.],[-3.,-1.],[3.,2.],[8.,-3.]],tf.float64)
    result=discover_modes(target,starts,max_iterations=100,score_tolerance=1e-8,
        merge_distance=1e-6,curvature_tolerance=1e-7)
    assert result['mode_count']==2
    assert not result['exhaustive_discovery']
    np.testing.assert_allclose(sorted(result['representatives'][:,0].numpy()),[-5.,5.],atol=1e-6)


def test_zero_weights_are_valid_and_never_resampled_even_at_zero_uniform():
    assert bool(valid_log_weights(tf.constant([-math.inf,0.],tf.float64)))
    for values in [[-math.inf,-math.inf],[0.,math.inf],[0.,math.nan]]:
        assert not bool(valid_log_weights(tf.constant(values,tf.float64)))
    np.testing.assert_array_equal(systematic_indices(tf.constant([-math.inf,0.],tf.float64),0.,4),[1,1,1,1])


def _flow_smc(target_name='gaussian',**overrides):
    from dataclasses import replace
    from bayesfilter.inference.neutra_flow_smc_tf import FlowTransportSMC
    from bayesfilter.inference.neutra_transport import NeuTraTransportConfig
    target=WarmStartTarget(target_name,jit_compile=False)
    cfg=replace(NeuTraTransportConfig.hoffman_author_iaf(2,conditional_scale_cap=2.,seed=(2,7)),hidden_layers=(4,4))
    options=dict(particles=8,stages=2,inner_updates=2,passes=2,learning_rate=.001,
        gradient_clip=100.,mutation_steps=1,step_size=.01,resampling_fraction=.5,jit_compile=False)
    options.update(overrides)
    return FlowTransportSMC(target,BroadStudentProposal(2),cfg,**options)


def test_aft_free_energy_gradient_matches_finite_difference_and_stops_sample_gradient():
    sampler=_flow_smc()
    x,lw,_=sampler.initial_population(tf.constant([23,5]))
    args=(x,lw,tf.constant(.2,tf.float64),tf.constant(.6,tf.float64))
    loss,grads,*_=sampler.evaluate(*args)
    variable=sampler.variables[-1];old=tf.identity(variable)
    direction=tf.ones_like(variable);h=1e-5
    variable.assign(old+h*direction);plus=sampler.evaluate(*args)[0]
    variable.assign(old-h*direction);minus=sampler.evaluate(*args)[0]
    variable.assign(old)
    np.testing.assert_allclose(tf.reduce_sum(grads[-1]*direction),(plus-minus)/(2*h),rtol=1e-6,atol=1e-7)
    with tf.GradientTape() as tape:
        tape.watch(x);value=sampler.evaluate(x,*args[1:])[0]
    assert tape.gradient(value,x) is None


def test_composed_target_preserves_flow_gradients_and_states_across_updates():
    nested=_flow_smc(target_name='mixture',inline_target=False)
    composed=_flow_smc(target_name='mixture',inline_target=True)
    seed=tf.constant([23,5]);population=nested.initial_population(seed)
    args=(*population[:2],tf.constant(0.,tf.float64),tf.constant(.5,tf.float64))
    for _ in range(3):
        a,b=nested.evaluate(*args),composed.evaluate(*args)
        for x,y in zip(tf.nest.flatten(a),tf.nest.flatten(b)):
            np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(nested.weights_only(*args)[0],composed.weights_only(*args)[0],rtol=1e-12)
        assert bool(nested.apply(*a[1])) and bool(composed.apply(*b[1]))
    a,_=nested.advance(population,args[2],args[3],seed)
    b,_=composed.advance(population,args[2],args[3],seed)
    for x,y in zip(a,b):np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-12)


def test_aft_uses_pre_update_selection_and_independent_three_populations():
    sampler=_flow_smc(inner_updates=1)
    initial=[v.numpy().copy() for v in sampler.variables]
    seeds=[]; original=sampler.initial_population
    def record(seed):
        seeds.append(tuple(seed.numpy()));return original(seed)
    sampler.initial_population=record
    result=sampler.run_aft(tf.constant([31,7]))
    assert result['complete'] and len(set(seeds))==3
    # With one objective observation the author's pre-update selection must
    # restore the original map at every stage, despite performing an update.
    for actual,expected in zip(sampler.variables,initial):
        np.testing.assert_array_equal(actual,expected)


def test_craft_freezes_all_maps_until_pass_finishes_and_uses_fresh_evaluation():
    sampler=_flow_smc()
    events=[]; original_advance=sampler.advance;original_apply=sampler.apply
    def advance(*args):
        events.append('advance');return original_advance(*args)
    def apply(*args):
        events.append('update');return original_apply(*args)
    sampler.advance=advance;sampler.apply=apply
    result=sampler.run_craft(tf.constant([47,9]))
    assert events==['advance','advance','update','update']*2+['advance','advance']
    assert result['evaluation_maps_frozen'] and result['independent_evaluation_population']


@pytest.mark.parametrize('kind',['forward','rkl','gabrie'])
def test_native_training_block_runs_real_shared_optimizer_without_scalar_target(kind):
    from bayesfilter.testing.neutra_warm_start_campaign import make_transport,TrainingBlock
    target=WarmStartTarget('gaussian',jit_compile=False)
    flow=make_transport(target,4,(17,3));before=[v.numpy().copy() for v in flow.trainable_variables]
    trainer=TrainingBlock(flow,target,batch=8,learning_rate=.001,clip=100.,kind=kind,
        walkers=4,walk_steps=2,jit_compile=False)
    pool=target.reference_sample(16,tf.constant([1,7]));current=pool[:4]
    result=trainer.run(tf.constant([19,1]),tf.constant(2),pool,tf.zeros([16],tf.float64),current,tf.constant(.01,tf.float64))
    assert int(result[0])==2 and bool(result[-1])
    assert any(np.any(v.numpy()!=old) for v,old in zip(flow.trainable_variables,before))


def test_forward_checkpoint_resume_matches_uninterrupted_adam_and_rejects_wrong_configuration():
    from bayesfilter.testing.neutra_warm_start_campaign import make_transport,TrainingBlock
    target=WarmStartTarget('mixture',jit_compile=False)
    def build(width=4):
        flow=make_transport(target,width,(17,3))
        return TrainingBlock(flow,target,batch=8,learning_rate=.001,clip=100.,kind='forward',
            walkers=4,walk_steps=2,jit_compile=False)
    original=build();resumed=build()
    pool=target.reference_sample(16,tf.constant([1,7]))
    def advance(trainer,key):
        return trainer.run(tf.constant(key),tf.constant(3),pool,tf.zeros([16],tf.float64),pool[:4],tf.constant(.01,tf.float64))
    advance(original,[19,1]);checkpoint=original.checkpoint()
    resumed.restore(json.loads(json.dumps(checkpoint)))
    advance(original,[19,4]);advance(resumed,[19,4])
    for a,b in zip(original.flow.trainable_variables,resumed.flow.trainable_variables):
        np.testing.assert_allclose(a,b,atol=1e-14,rtol=1e-14)
    for a,b in zip(original.trainer.optimizer.variables,resumed.trainer.optimizer.variables):
        np.testing.assert_allclose(a,b,atol=1e-14,rtol=1e-14)
    assert int(original.trainer.step)==int(resumed.trainer.step)==6
    wrong=build(width=8);before=[tf.identity(v) for v in wrong.flow.trainable_variables]
    with pytest.raises(ValueError,match='configuration mismatch'):wrong.restore(checkpoint)
    for a,b in zip(wrong.flow.trainable_variables,before):np.testing.assert_array_equal(a,b)
    corrupted=json.loads(json.dumps(checkpoint));corrupted['step']+=1
    with pytest.raises(ValueError,match='hash mismatch'):resumed.restore(corrupted)


def test_weighted_gradient_calibration_ignores_zero_mass_extreme_particle():
    from bayesfilter.testing.neutra_warm_start_campaign import make_transport,gradient_calibration
    target=WarmStartTarget('gaussian',jit_compile=False);flow=make_transport(target,4,(73,11))
    before=[v.numpy().copy() for v in flow.trainable_variables]
    pool=tf.constant([[.4,-.2],[1e200,1e200]],tf.float64)
    clip,report=gradient_calibration(flow,pool,dict(batch_size=4,gradient_pilot_batches=2,clip_pilot_multiplier=5.),
        17,log_weights=tf.constant([0.,-math.inf],tf.float64))
    assert math.isfinite(clip)
    assert float(report['batch_gradient_noise_to_squared_mean'])==0.
    for v,old in zip(flow.trainable_variables,before):np.testing.assert_array_equal(v,old)
