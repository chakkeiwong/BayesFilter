"""CPU-only mechanics: branches, data streams and controller wiring."""
import importlib.util
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.testing.neutra_generic_targets import ExactTargetEvaluator
from bayesfilter.testing.neutra_source_fit_remedy import make_trainer, branch_trainer, train_block


def fixture():
    evaluator = ExactTargetEvaluator({'kind':'gaussian','mean':[0.,0.],
        'covariance':[[1.,0.],[0.,1.]]},jit_compile=False)
    profile = dict(width=4,batch=4,learning_rate=.001,jit_compile=False)
    return evaluator,profile


def test_branch_copies_map_and_resets_every_adam_state():
    evaluator,profile=fixture()
    parent=make_trainer(evaluator.target,11,profile)
    x=tf.constant([[1.,2.],[-2.,1.],[.5,-.1],[.2,.3]],tf.float64)
    train_block(parent,x,fresh=False,updates=2,seed=[11,1])
    saved=parent.checkpoint()
    branches=[branch_trainer(parent,evaluator.target,11,profile,o) for o in ('forward','rkl','joint')]
    for branch in branches:
        assert branch.checkpoint()['base']['parameters']==saved['base']['parameters']
        assert int(branch.optimizer.iterations)==0
        assert branch.checkpoint()['base']['optimizer']==branches[0].checkpoint()['base']['optimizer']
    train_block(branches[-1],x,fresh=True,updates=1,seed=[11,2])
    assert parent.checkpoint()==saved
    assert branches[0].checkpoint()['base']['parameters']==saved['base']['parameters']


def test_forward_update_matches_prior_weighted_authority():
    from bayesfilter.inference.neutra_weighted_training import WeightedForwardKLNeuTraTrainer,WeightedNeuTraConfig
    evaluator,profile=fixture()
    current=make_trainer(evaluator.target,11,profile)
    comparison=make_trainer(evaluator.target,11,profile)
    prior=WeightedForwardKLNeuTraTrainer(WeightedNeuTraConfig(dimension=2,
        hidden_layers=(4,4),stages=3,learning_rate=.001,beta1=.9,beta2=.999,
        epsilon=1e-8,gradient_clip_norm=1000.,jit_compile=False),transport=comparison.transport)
    x=tf.constant([[1.,2.],[-2.,1.],[.5,-.1],[.2,.3]],tf.float64)
    z=tf.constant([[.1,.2],[-.2,.1],[.5,-.1],[.2,.3]],tf.float64)
    current.train_joint_step(z,x,tf.zeros([4],tf.float64))
    prior._train_step_impl(x,tf.zeros([4],tf.float64))
    for a,b in zip(current.variables,prior.variables):
        tf.debugging.assert_near(a,b,atol=1e-12,rtol=1e-12)


def test_learning_rate_repair_checks_target_before_training(tmp_path):
    import json
    from bayesfilter.testing.neutra_source_fit_remedy import run_optimizer_repair
    evaluator,profile=fixture()
    parent=make_trainer(evaluator.target,11,profile)
    (tmp_path/'fresh-checkpoint.json').write_text(json.dumps(parent.checkpoint()))
    (tmp_path/'result.json').write_text(json.dumps({'seed':11,'same_initial_parameters':True}))
    changed={'kind':'gaussian','mean':[1.,0.],'covariance':[[1.,0.],[0.,1.]]}
    with pytest.raises(ValueError,match='target mismatch'):
        run_optimizer_repair(changed,{**profile,'parent':str(tmp_path)},11,tmp_path)


def test_optional_naf_uses_the_same_joint_training_authority():
    evaluator,profile=fixture()
    trainer=make_trainer(evaluator.target,11,{**profile,'kind':'naf_dsf'})
    assert trainer.transport.config.naf_conditioner=='author_cmade'
    assert trainer.transport.config.kind=='naf_dsf'
    x=tf.constant([[1.,2.],[-2.,1.],[.5,-.1],[.2,.3]],tf.float64)
    before=trainer.checkpoint()['base']['parameters']
    result=train_block(trainer,x,fresh=True,updates=1,seed=[11,2])
    assert result['finite'] and result['updates']==1
    assert trainer.checkpoint()['base']['parameters']!=before
    assert trainer.transport.config.stages==3


@pytest.mark.parametrize('fresh',[False,True])
def test_split_training_resumes_exact_data_and_optimizer_stream(fresh):
    evaluator,profile=fixture()
    rows=tf.reshape(tf.range(32,dtype=tf.float64),[16,2])/10.
    continuous=make_trainer(evaluator.target,37,profile)
    partial=make_trainer(evaluator.target,37,profile)
    train_block(continuous,rows,fresh=fresh,updates=4,seed=[37,3])
    train_block(partial,rows,fresh=fresh,updates=2,seed=[37,3])
    restored=make_trainer(evaluator.target,37,profile)
    restored.restore(partial.checkpoint())
    train_block(restored,rows,fresh=fresh,updates=2,seed=[37,3],counter=2)
    for lhs,rhs in zip(continuous.variables,restored.variables):
        tf.debugging.assert_equal(lhs,rhs)
    assert continuous.checkpoint()['base']['optimizer']==restored.checkpoint()['base']['optimizer']
    if fresh:
        with pytest.raises(ValueError,match='exhausted'):
            train_block(restored,rows,fresh=True,updates=1,seed=[37,3],counter=4)


def load_master():
    path=Path(__file__).resolve().parents[1]/'scripts/run_neutra_scientific_campaign_master.py'
    spec=importlib.util.spec_from_file_location('remedy_master_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_master_executes_complete_pairs_before_reporting(tmp_path,monkeypatch):
    master=load_master()
    shared=tmp_path/'shared'
    master.write(shared/'config.json',master.LIMITS)
    master.write(shared/'state.json',{'attempts':[]})
    monkeypatch.setattr(master,'CAMPAIGN',tmp_path/'campaign')
    monkeypatch.setattr(master,'SHARED',shared)
    monkeypatch.setattr(master,'PREVIOUS',())
    master.write(master.CAMPAIGN/'history-audit-r1.json',{'r3_prices':[]})
    controller=master.RemedyController()
    calls=[]
    monkeypatch.setattr(controller,'execute',lambda job,**kwargs:calls.append((job,kwargs)))
    monkeypatch.setattr(controller,'report',lambda:0)
    assert controller.run()==0
    assert [job for job,_ in calls[:2]]==['check','preflight']
    assert len(calls[2:])==6
    assert all(kwargs['phase']=='matched_fit' for _,kwargs in calls[2:])
    assert {(k['target'],k['seed']) for _,k in calls[2:]}=={
        (t,s) for t in master.DEV_TARGETS for s in master.REMEDY_SEEDS}


def test_terminal_resume_does_not_launch_new_training(tmp_path,monkeypatch):
    master=load_master()
    shared=tmp_path/'shared'
    master.write(shared/'config.json',master.LIMITS)
    master.write(shared/'state.json',{'attempts':[]})
    monkeypatch.setattr(master,'CAMPAIGN',tmp_path/'campaign')
    monkeypatch.setattr(master,'SHARED',shared)
    monkeypatch.setattr(master,'PREVIOUS',())
    controller=master.RemedyController()
    controller.state['status']='remedy_fit_complete'
    monkeypatch.setattr(controller,'execute',lambda *args,**kwargs:pytest.fail('unexpected relaunch'))
    monkeypatch.setattr(controller,'report',lambda:0)
    assert controller.run()==0
