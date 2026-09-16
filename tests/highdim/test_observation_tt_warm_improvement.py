"""CPU diagnostic checks of A09 selection and the actual particle call chain."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from bayesfilter.highdim import observation_guided_tt_tf as lib
from bayesfilter.highdim import pair_block_tt_tf as pair
from bayesfilter.highdim import sgqf_joint_consumer_tf as joint
from docs.benchmarks import run_observation_aware_tt_complete as base
from docs.benchmarks import run_observation_tt_warm_improvement as driver

D = tf.float64


def test_standalone_path_and_particle_consumer_reach_pair_sampler(tmp_path, monkeypatch):
    model = lib.SVModel(tf.constant([[.6]],D),tf.constant([[1.5625]],D),1.,1.)
    observations = tf.constant([[.8],[1.1],[.5]],D)
    guide,_ = lib.build_guide_path(model,observations)
    monkeypatch.setattr(driver,'tf',tf,raising=False)
    monkeypatch.setattr(driver,'lib',lib,raising=False)
    monkeypatch.setattr(driver,'joint',joint,raising=False)
    monkeypatch.setattr(base,'tf',tf,raising=False)
    monkeypatch.setattr(base,'D',D,raising=False)
    monkeypatch.setattr(base,'lib',lib,raising=False)
    incoming=[];steps=[]
    def fixed_step(model,observation,guide,retained,t,seed,cfg,budget):
        incoming.append(retained)
        cores=pair.initial_pair_cores(1,1,1)
        Z=pair.pair_total_mass(cores);tau=Z*.01
        retained_next=lib.PairRetainedProposal(cores,guide[t][1],Z,tau,t)
        step=lib.PairTTStep(cores,guide[t][1],guide[t-1][1],tau,{},t,retained_next)
        steps.append(step)
        return step,dict(time=t)
    monkeypatch.setattr(driver,'fit_step',fixed_step)
    path=driver.warm_path(model,observations,guide,91,driver.config(3,1024,1e-5),tmp_path,lambda:None)
    assert isinstance(path[0],joint.SGQFJointStep)
    assert incoming[0] is path[0].retained_proposal
    assert incoming[1] is steps[0].retained_proposal
    actual=lib.sample_pair_tt_step;calls=[]
    def recording_sampler(step,*args,**kwargs):
        calls.append(step.time_index)
        return actual(step,*args,**kwargs)
    monkeypatch.setattr(lib,'sample_pair_tt_step',recording_sampler)
    result,_=base.particle_filter(model,observations,guide,path,'tt_sgqf_initialized',32,93,False)
    assert calls==[1,2]
    assert len(result['steps'])==3
    tf.debugging.assert_all_finite(result['log_evidence'],'finite consumer evidence')


def test_selection_cannot_reward_missing_hard_sequence():
    baseline=driver.config(3,1024,0.)
    candidate=driver.config(4,4096,0.)
    contexts=[]
    for i in range(3):
        metrics={driver.key(baseline):dict(regimes=dict(all=dict(mse=.1)))}
        if i<2: metrics[driver.key(candidate)]=dict(regimes=dict(all=dict(mse=.01)))
        contexts.append(dict(guide=True,entry=dict(reference_pass=True,metrics=metrics)))
    chosen,table=driver.choose(contexts,[baseline,candidate])
    assert chosen==baseline
    assert table[1]['eligible'] is False


def test_paired_inference_excludes_missing_sequences_and_duplicate_evidence(monkeypatch):
    monkeypatch.setattr(driver,'tf',tf,raising=False)
    monkeypatch.setattr(driver,'D',D,raising=False)
    records=[]
    for d in (1,4):
        for i in range(12):
            metrics={}
            for name in ('baseline','capacity','preservation',*driver.HEURISTICS):
                mse=.1+i*.001
                if name=='capacity': mse-=.02+i*.0001
                metrics[name]=dict(regimes={r:dict(mse=mse,count=20) for r in ('all','near_zero','ordinary','large')})
            if d==4 and i==0: del metrics['capacity']
            records.append(dict(dimension=d,sequence=i,metrics=metrics,reference_pass=True))
    contrasts=driver.infer(records)['primary_contrasts']
    scalar=next(c for c in contrasts if c['dimension']==1 and c['candidate']=='capacity')
    missing=next(c for c in contrasts if c['dimension']==4 and c['candidate']=='capacity')
    assert scalar['upper']<0
    assert missing['sequences']==11 and not missing['interval_eligible']
    for c in contrasts:
        if c['candidate']=='preservation':
            assert c['exactly_identical'] and not c['mse_improvement_supported']


def test_seed_blocks_do_not_overlap_across_units_panels_or_replicates():
    used=set()
    for partition in (0,1,2):
        for dimension in (1,4):
            for sequence in range(12):
                seed=driver.sequence_seed(partition,dimension,sequence)
                offsets=[0]
                offsets += [2000000+panel+t for panel in (0,100000,200000) for t in range(20)]
                offsets += [3000000+1000*r+t for r in range(4) for t in range(20)]
                offsets += [4000000+1000000*level+1000*r+t for level in range(3) for r in range(4) for t in range(20)]
                streams={seed+offset for offset in offsets}
                assert len(streams)==len(offsets)
                assert not used.intersection(streams)
                assert max(streams)<2**31
                used.update(streams)


def test_reference_endpoint_uses_disjoint_repetition_stride(tmp_path, monkeypatch):
    from docs.benchmarks import run_observation_tt_independent_filtering as reference
    monkeypatch.setattr(reference,'tf',tf,raising=False)
    monkeypatch.setattr(reference,'D',D,raising=False)
    monkeypatch.setattr(reference,'base',base,raising=False)
    calls=[]
    def fake_particles(model, observations, guide, path, method, count, seed, jit):
        calls.append(seed)
        return dict(steps=[dict(mean=tf.zeros([4],D))],log_evidence=tf.constant(0.,D)),None
    monkeypatch.setattr(base,'particle_filter',fake_particles)
    model=lib.SVModel(tf.eye(4,dtype=D)*.6,tf.eye(4,dtype=D),1.,1.)
    reference.references(model,tf.zeros([1,4],D),tmp_path,700000000,lambda:None,replicate_seed_stride=1000)
    assert calls==[700000000+1000000*level+1000*r for level in range(2) for r in range(4)]
