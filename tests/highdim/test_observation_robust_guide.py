"""CPU diagnostic identities and actual-consumer checks for optional A10."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import math
import pytest
import tensorflow as tf
from bayesfilter.highdim import observation_guided_tt_tf as obs
from bayesfilter.highdim import observation_robust_guide_tf as robust
from bayesfilter.highdim import pair_block_tt_tf as pair
from bayesfilter.highdim import sgqf_joint_consumer_tf as joint
from docs.benchmarks import observation_tt_sgqf_projection_diagnostic as projection
from docs.benchmarks import run_observation_aware_tt_complete as consumer
from docs.benchmarks import run_observation_tt_robust_guide as campaign

D = tf.float64


def model1():
    return obs.SVModel(tf.constant([[.6]],D), tf.constant([[1.5625]],D), .4, 1.)


def test_independent_scale_guard_rejects_single_node_collapse():
    chart = obs.Chart.from_moments(tf.zeros([1],D),tf.eye(1,dtype=D))
    points = tf.constant([[-1.],[0.],[1.]],D)
    weights = tf.constant([1e-31,1.,1e-31],D)
    old_cov = tf.constant([[2e-31]],D)
    obs.spd_factor(old_cov,'old scale-free test accepts')
    result, info = robust.checked_rule(chart,points,weights,tf.constant(0.,D),jit_compile=False)
    assert result is None and info['reason']=='covariance_unresolved_in_predictive_units'
    assert float(info['dominant_absolute_fraction'])==1.
    assert float(info['eigenvalues'][0]) < float(info['covariance_margin'])


def test_positive_rule_matches_gaussian_moments_and_no_fire():
    nodes, weights = robust.positive_gaussian_rule(2,5)
    m = tf.constant([.2,-.4],D); cov=tf.constant([[2.,.3],[.3,1.]],D)
    chart=obs.Chart.from_moments(m,cov)
    result,info=robust.checked_rule(chart,nodes,weights,tf.constant(0.,D),jit_compile=False)
    assert info['status']=='valid'
    tf.debugging.assert_near(result.mean,m,atol=1e-13,rtol=1e-13)
    tf.debugging.assert_near(result.factor@tf.transpose(result.factor),cov,atol=1e-13,rtol=1e-13)
    x=chart.forward(nodes);oldm,oldcov=obs.gaussian_moments(x,weights/tf.reduce_sum(weights))
    tf.debugging.assert_near(result.mean,oldm,atol=1e-13,rtol=1e-13)
    tf.debugging.assert_near(info['physical_covariance'],oldcov,atol=1e-13,rtol=1e-13)


def test_signed_cancellation_and_indefiniteness_remain_separate():
    chart=obs.Chart.from_moments(tf.zeros([1],D),tf.eye(1,dtype=D))
    _,info=robust.checked_rule(chart,tf.constant([[0.],[1.]],D),tf.constant([2.,-1.],D),tf.constant(0.,D),jit_compile=False)
    assert bool(info['mass_valid']) and not bool(info['covariance_valid'])
    _,info=robust.checked_rule(chart,tf.constant([[0.],[1.]],D),tf.constant([1.,-1.],D),tf.constant(0.,D),jit_compile=False)
    assert not bool(info['mass_valid'])


@pytest.mark.parametrize('y',[0.,.2,6.45])
def test_sv_mode_analytic_stationarity_and_positive_hessian(y):
    model=model1();m=tf.constant([.3],D);factor=tf.constant([[1.1]],D)
    x,cov,info=robust.laplace_mode_kernel(1,.4,False)(m,factor,tf.constant([y],D))
    assert bool(info['resolved'])
    precision=1/1.21;c=(y/.4)**2
    gradient=precision*(x-m)+.5*(1-c*tf.exp(-x))
    hessian=precision+.5*c*tf.exp(-x)
    tf.debugging.assert_near(gradient,tf.zeros_like(gradient),atol=2e-10,rtol=0.)
    tf.debugging.assert_near(cov[0,0],1/hessian[0],atol=1e-13,rtol=1e-13)
    assert float(hessian[0])>=precision


def test_laplace_change_of_measure_recovers_predictive_gaussian():
    # An arbitrary shifted/scaled integration chart requires the density ratio.
    predictive=obs.Chart.from_moments(tf.constant([.2],D),tf.constant([[1.]],D))
    laplace=obs.Chart.from_moments(tf.constant([.4],D),tf.constant([[1.3]],D))
    u,w=robust.positive_gaussian_rule(1,25);x=laplace.forward(u)
    ratio=tf.exp(predictive.log_prob(x)-laplace.log_prob(x));weights=w*ratio
    mass=tf.reduce_sum(weights)
    m,c=obs.gaussian_moments(x,weights/mass)
    tf.debugging.assert_near(m,predictive.mean,atol=1e-11,rtol=1e-11)
    tf.debugging.assert_near(c,tf.constant([[1.]],D),atol=1e-10,rtol=1e-10)
    tf.debugging.assert_near(mass,tf.constant(1.,D),atol=1e-12,rtol=1e-12)


def test_stable_chart_and_initializer_preserve_physical_joint():
    model=model1()
    past=obs.Chart.from_moments(tf.constant([-.2],D),tf.constant([[.3]],D))
    current=obs.Chart.from_moments(tf.constant([.4],D),tf.constant([[.2]],D))
    charts=robust.stable_charts(model,[(past,past),(current,current)])
    for chart in charts:
        tf.debugging.assert_near(chart.factor@tf.transpose(chart.factor),model.covariance0,atol=1e-14)
    m0,c0=projection.paired_gaussian(model,current,past)
    m1,c1=projection.paired_gaussian(model,charts[1],charts[0],guide_current=current,guide_condition=past)
    l0=tf.linalg.diag(tf.stack([current.factor[0,0],past.factor[0,0]]))
    l1=tf.linalg.diag(tf.stack([charts[1].factor[0,0],charts[0].factor[0,0]]))
    b=tf.concat([current.mean,past.mean],axis=0)
    tf.debugging.assert_near(b+tf.linalg.matvec(l0,m0),b+tf.linalg.matvec(l1,m1),atol=1e-13)
    tf.debugging.assert_near(l0@c0@tf.transpose(l0),l1@c1@tf.transpose(l1),atol=1e-13)
    # Identical random draws produce identical physical guided samples.
    a,_,_=obs.joint_sgqf_row_sampler(model,current,past,128,123,epsilon=1e-12)
    b,logw,diag=obs.joint_sgqf_row_sampler(model,charts[1],charts[0],128,123,epsilon=1e-12,
                                       guide_current=current,guide_condition=past)
    tf.debugging.assert_near(current.forward(a[:,:1]),charts[1].forward(b[:,:1]),atol=1e-13)
    tf.debugging.assert_near(past.forward(a[:,1:]),charts[0].forward(b[:,1:]),atol=1e-13)
    assert float(diag['maximum_rho_over_s'])<=1e12


def gaussian_pair_step(model):
    current=obs.Chart.from_moments(tf.constant([1.],D),tf.constant([[.4]],D))
    past=obs.Chart.from_moments(tf.constant([0.],D),tf.constant([[1.]],D))
    cores=pair.initial_pair_cores(1,1,1)
    z=pair.pair_total_mass(cores);tau=z*.01
    retained=obs.PairRetainedProposal(cores,current,z,tau,1)
    return obs.PairTTStep(cores,current,past,tau,{},1,retained)


def test_pair_density_evaluation_matches_sampler_and_mixture_both_branches():
    model=model1();step=gaussian_pair_step(model);previous=tf.zeros([128,1],D)
    x,logq,_=obs.sample_pair_tt_step(step,previous,38,False)
    tf.debugging.assert_near(robust.conditional_log_density(step,x,previous,False),logq,atol=1e-12)
    wrapped=robust.PhysicalDefenseStep(step,model,.5)
    x,logq,info=robust.sample_physical_defense(wrapped,previous,39,False)
    logtt=robust.conditional_log_density(step,x,previous,False)
    logf=model.transition_log_prob(x,previous)
    exact=tf.math.log(.5*tf.exp(logtt)+.5*tf.exp(logf))
    tf.debugging.assert_near(logq,exact,atol=1e-12)
    assert .1<float(info['physical_defense_fraction'])<.9
    assert float(info['log_density_lower_bound_margin'])>=-1e-12


def test_mixture_normalization_importance_and_second_moment_by_quadrature():
    model=model1();step=gaussian_pair_step(model);eps=.2
    x=tf.linspace(tf.constant(-14.,D),tf.constant(14.,D),20001)[:,None]
    previous=tf.zeros_like(x);dx=x[1,0]-x[0,0]
    logtt=robust.conditional_log_density(step,x,previous,False)
    logf=model.transition_log_prob(x,previous)
    q=(1-eps)*tf.exp(logtt)+eps*tf.exp(logf)
    tf.debugging.assert_near(tf.reduce_sum(q)*dx,tf.constant(1.,D),atol=1e-9)
    g=tf.exp(model.observation_log_prob(x,tf.constant([.8],D)))
    target=tf.exp(logf)*g;w=target/q
    tf.debugging.assert_near(tf.reduce_sum(q*w)*dx,tf.reduce_sum(target)*dx,atol=1e-13)
    second=tf.reduce_sum(q*w*w)*dx
    sharper=tf.reduce_sum(tf.exp(logf)*g*g)*dx/eps
    bound=(2*math.pi*.4**2)**-1*math.exp(.5)/eps
    assert float(second)<=float(sharper)+1e-12 and float(sharper)<=bound


def test_consumer_calls_physical_mixture_at_initial_and_later_steps(monkeypatch):
    model=model1();observations=tf.constant([[.3],[.6]],D)
    guide,_=obs.build_guide_path(model,observations)
    initial=joint.make_sgqf_joint_step(model,guide[0][1],None,0)
    path=[robust.PhysicalDefenseStep(s,model,.2) for s in (initial,gaussian_pair_step(model))]
    for name,value in [('tf',tf),('D',D),('lib',obs)]:monkeypatch.setattr(consumer,name,value,raising=False)
    calls=[];actual=robust.sample_physical_defense
    def record(step,*args,**kwargs):
        calls.append(step.proposal.time_index)
        return actual(step,*args,**kwargs)
    monkeypatch.setattr(robust,'sample_physical_defense',record)
    result,_=consumer.particle_filter(model,observations,guide,path,'tt_physical_defense',32,99,False)
    assert calls==[0,1]
    assert all('physical_defense_fraction' in s for s in result['steps'])
    tf.debugging.assert_all_finite(result['log_evidence'],'consumer evidence')


def test_a10_seed_blocks_are_disjoint_and_int32_safe():
    seen=set()
    for stage,count in [('calibration',3),('confirmation',12)]:
        for dimension in (1,4):
            for sequence in range(count):
                seed=campaign.sequence_seed(stage,dimension,sequence)
                seeds=[seed]
                seeds.extend(seed+2000000+offset+t for offset in (0,100000,200000) for t in range(20))
                seeds.extend(seed+3000000+1000*r+t for r in range(4) for t in range(20))
                seeds.extend(seed+4000000+1000000*level+1000*r+t for level in range(3) for r in range(4) for t in range(20))
                assert not seen.intersection(seeds) and len(seeds)==len(set(seeds))
                assert max(seeds)<2**31
                seen.update(seeds)


def test_calibration_does_not_discard_failures_or_choose_best_epsilon():
    entries=[]
    for d in (1,4):
        for i in range(3):
            metrics={}
            for family in ('guide','stable'):
                for l1 in (1e-5,.001):
                    metrics[f'{family}-l{l1:g}']=dict(regimes=dict(all=dict(mse=.1)),log_evidence_screen=dict(passed=True),
                        cdf_bracket_failures=0,consumer_invalid_steps=0)
            for l1 in (1e-5,.001):
                for eps in (.05,.2,.5):
                    metrics[f'full-l{l1:g}-e{eps:g}']=dict(regimes=dict(all=dict(mse=.1 if eps==.05 else .05)),
                        log_evidence_screen=dict(passed=True),cdf_bracket_failures=0,consumer_invalid_steps=0)
            if i==0:del metrics['stable-l1e-05']
            entries.append(dict(dimension=d,reference_pass=True,metrics=metrics))
    controls=campaign.freeze(entries)['controls']
    for value in controls.values():
        assert value['stable']['l1']==.001
        assert value['epsilon']==.05  # Smallest non-harming; not minimum MSE.
