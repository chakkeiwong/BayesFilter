"""CPU-only independent diagnostic checks for the optional A11 extension."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'

import json
from types import SimpleNamespace
import pytest
import tensorflow as tf
from bayesfilter.highdim import observation_guided_tt_tf as obs
from bayesfilter.highdim import observation_robust_guide_tf as robust
from bayesfilter.highdim import pair_block_tt_tf as pair
from bayesfilter.highdim import sgqf_joint_consumer_tf as joint
from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import _log_standard_normal as logrho
from docs.benchmarks import observation_tt_sgqf_projection_diagnostic as projection
from docs.benchmarks import run_observation_aware_tt_complete as consumer
from docs.benchmarks import run_observation_tt_protected_fitting as campaign
from docs.benchmarks import run_observation_tt_independent_filtering as prior
from docs.benchmarks import run_observation_tt_warm_improvement as warm

D=tf.float64


def fixture2():
    model=obs.SVModel(tf.constant([[.6,.2],[0.,.7]],D),tf.constant([[2.,.3],[.3,1.]],D),.4,.3)
    tiny=obs.Chart.from_moments(tf.constant([.1,-.2],D),1e-28*tf.eye(2,dtype=D))
    broad=obs.Chart.from_moments(tf.constant([.3,.2],D),tf.constant([[4.,.1],[.1,.4]],D))
    return model,[(tiny,tiny),(tiny,broad)]


def test_blend_retains_collapse_floor_and_exact_stable_endpoint():
    model,guide=fixture2();stable=robust.stable_charts(model,guide)
    exact=robust.blended_charts(model,guide,1.,jit_compile=False)
    for a,b in zip(stable,exact):tf.debugging.assert_equal(a.factor,b.factor)
    blend=robust.blended_charts(model,guide,.25,jit_compile=False)
    for (_,posterior),reference,chart in zip(guide,stable,blend):
        expected=.75*(posterior.factor@tf.transpose(posterior.factor))+.25*(reference.factor@tf.transpose(reference.factor))
        tf.debugging.assert_near(chart.factor@tf.transpose(chart.factor),expected,atol=1e-14,rtol=1e-14)
        normalized=tf.linalg.triangular_solve(reference.factor,chart.factor)
        assert float(tf.reduce_min(tf.linalg.eigvalsh(normalized@tf.transpose(normalized))))>=.25-1e-14
        tf.debugging.assert_equal(chart.mean,posterior.mean)
    assert float(tf.reduce_min(tf.linalg.eigvalsh(blend[0].factor@tf.transpose(blend[0].factor))))>.1


@pytest.mark.parametrize('fraction',[0.,-.1,1.01,float('nan'),float('inf')])
def test_invalid_blend_fraction_fails(fraction):
    model,guide=fixture2()
    with pytest.raises(ValueError,match='fraction'):robust.blended_charts(model,guide,fraction,jit_compile=False)


def test_blend_rejects_nonfinite_guide_factor():
    model,guide=fixture2();guide[0]=(guide[0][0],SimpleNamespace(mean=tf.zeros(2,D),factor=tf.fill([2,2],tf.constant(float('nan'),D))))
    with pytest.raises(ValueError,match='factor'):robust.blended_charts(model,guide,.25,jit_compile=False)


def test_cholesky_derivative_independent_finite_difference():
    covariance=tf.constant([[2.,.3],[.3,1.]],D);derivative=tf.constant([[.2,-.1],[-.1,.4]],D)
    factor=tf.linalg.cholesky(covariance)
    left=tf.linalg.triangular_solve(factor,derivative)
    b=tf.transpose(tf.linalg.triangular_solve(factor,tf.transpose(left)))
    phi=tf.linalg.band_part(b,-1,0)-.5*tf.linalg.diag(tf.linalg.diag_part(b))
    analytical=factor@phi;epsilon=1e-5
    numerical=(tf.linalg.cholesky(covariance+epsilon*derivative)-tf.linalg.cholesky(covariance-epsilon*derivative))/(2*epsilon)
    tf.debugging.assert_near(analytical,numerical,atol=1e-10,rtol=1e-8)
    tf.debugging.assert_near(analytical@tf.transpose(factor)+factor@tf.transpose(analytical),derivative,atol=1e-14,rtol=1e-14)


def test_seed_partitions_are_disjoint_and_signed_int32_safe():
    occupied=set()
    for attempt in (1,2):
        for stage,count in [('smoke',1),('calibration',6),('confirmation',12)]:
            for dimension in (1,4):
                for sequence in range(count):
                    seed=campaign.sequence_seed(stage,dimension,sequence,attempt)
                    offsets=[0]+[2000000+panel+t for panel in (0,100000,200000) for t in range(20)]
                    offsets += [3000000+1000*r+t for r in range(4) for t in range(20)]
                    offsets += [4000000+1000000*level+10000*r+t for level in range(4) for r in range(8) for t in range(20)]
                    values={seed+k for k in offsets}
                    assert len(values)==len(offsets) and not values.intersection(occupied)
                    assert min(values)>-2**31 and max(values)<2**31
                    occupied.update(values)


def test_scalar_candidate_capacity_is_deduplicated():
    scalar=campaign.candidates(1);four=campaign.candidates(4)
    assert len(scalar)==10 and len(four)==16
    assert not any(n.startswith(('rank-','combined-','rows-')) for n in scalar)


def entry_for_selection(d=4):
    metrics={}
    for name in campaign.candidates(d):
        metrics[name]=dict(regimes={'all':{'mse':1.}},log_evidence_screen={'passed':True},
                           cdf_bracket_failures=0,consumer_invalid_steps=0,low_ess_fraction=.1)
    return dict(dimension=d,reference_pass=True,metrics=metrics,failures={})


def test_nomination_requires_whole_panel_and_ignores_invalid_winner():
    entries=[entry_for_selection() for _ in range(6)]
    candidate=next(iter(campaign.candidates(4)))
    for e in entries:e['metrics'][candidate]['regimes']['all']['mse']=.001
    entries[-1]['metrics'][candidate]['log_evidence_screen']['passed']=False
    assert campaign.nominate(entries,4)['name']!=candidate
    entries[-1]['reference_pass']=False
    with pytest.raises(ValueError,match='references'):campaign.nominate(entries,4)


def test_defense_selection_uses_non_harm_and_low_ess_not_best_mse():
    entries=[entry_for_selection() for _ in range(6)]
    nomination=campaign.nominate(entries,4);name=nomination['name']
    for entry in entries:
        for epsilon,mse,low in [(.1,.8,.2),(.2,1.05,.1)]:
            entry['metrics'][f'{name}-e{epsilon:g}']=dict(entry['metrics'][name],
                    regimes={'all':{'mse':mse}},low_ess_fraction=low)
    chosen=campaign.select_epsilon(entries,4,nomination)
    assert chosen['epsilon']==.2
    assert not chosen['epsilon_curve'][1]['non_harm']


def test_reference_policy_executes_eight_replicates_and_records_limits(monkeypatch,tmp_path):
    calls=[]
    def fake_filter(model,observations,guide,path,method,N,seed,jit):
        calls.append((N,seed))
        return dict(steps=[{'mean':tf.zeros([4],D)} for _ in range(2)],log_evidence=tf.constant(-1.,D)),None
    monkeypatch.setattr(prior,'tf',tf,raising=False);monkeypatch.setattr(prior,'D',D,raising=False)
    monkeypatch.setattr(prior,'base',SimpleNamespace(particle_filter=fake_filter),raising=False)
    model=SimpleNamespace(dimension=4,covariance0=tf.eye(4,dtype=D))
    _,_,passed,info=prior.references(model,tf.zeros([2,4],D),tmp_path,-1500000000,lambda:None,**campaign.REFERENCE)
    assert passed and len(calls)==16 and len(set(seed for _,seed in calls))==16
    assert info['selected_N']==131072 and info['policy']['mean_mcse_limit']==.01
    assert (tmp_path/'reference-65536.json').exists() and (tmp_path/'reference-131072.json').exists()


def test_new_charts_reach_real_pair_fitter_and_physical_filter(monkeypatch,tmp_path):
    for module in (campaign,prior,warm,consumer):
        monkeypatch.setattr(module,'tf',tf,raising=False);monkeypatch.setattr(module,'D',D,raising=False)
    for name,value in dict(lib=obs,pair=pair,joint=joint,projection=projection,base=consumer,logrho=logrho).items():
        monkeypatch.setattr(warm,name,value,raising=False)
    for name,value in dict(lib=obs,robust=robust,joint=joint).items():monkeypatch.setattr(campaign,name,value,raising=False)
    monkeypatch.setattr(consumer,'lib',obs,raising=False)
    model=obs.SVModel(tf.constant([[.6]],D),tf.constant([[1.5625]],D),.4,1.)
    observations=tf.constant([[.3],[.6]],D)
    guide,_=robust.build_guide_path(model,observations,jit_compile=False)
    context=dict(dest=tmp_path,model=model,guide=guide,observations=observations,seed=-1200000000)
    cfg=dict(warm.config(2,64,.001),fraction=.25,rank=1,sweeps=1,proximal_steps=8)
    path=campaign.fit_path(context,'small',cfg,lambda:None)
    charts=robust.blended_charts(model,guide,.25,jit_compile=False)
    tf.debugging.assert_near(path[1].current_chart.factor,charts[1].factor,atol=1e-13,rtol=1e-13)
    tf.debugging.assert_near(path[1].conditioning_chart.factor,charts[0].factor,atol=1e-13,rtol=1e-13)
    wrapped=[robust.PhysicalDefenseStep(p,model,.05) for p in path]
    result,_=consumer.particle_filter(model,observations,guide,wrapped,'tt_physical_defense',32,-900000000,False)
    assert all(bool(s['finite']) and bool(s.get('cdf_bracket_valid',True)) for s in result['steps'])
    assert all(float(s['ess'])>0 for s in result['steps'])
    assert len(json.loads((tmp_path/'small/path.json').read_text())['charts'])==2
