"""Tiny CPU reference of the real fit consumer; no learned-quality claims."""
import hashlib
import json

import pytest
import tensorflow as tf

from bayesfilter.testing import neutra_warm_start_closure as closure
from bayesfilter.testing.neutra_warm_start_campaign import (
    TrainingBlock, make_transport, save_tensor, write_json)
from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget, F64


@pytest.fixture
def saved_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(closure, 'gpu_preflight', lambda *args: None)
    target=WarmStartTarget('mixture')
    prepared=tmp_path/'prepared';prepared.mkdir()
    teacher=tmp_path/'teacher';teacher.mkdir()
    parent=tmp_path/'previous'/'w4-lr0.001';parent.mkdir(parents=True)
    pool=target.reference_sample(128,tf.constant([61,2]));lw=tf.zeros([128],F64)
    for name in ('training','validation'):
        save_tensor(prepared/f'{name}.tensor',pool)
    save_tensor(prepared/'discovered_modes.tensor',tf.constant([[-5.,0.],[5.,0.]],F64))
    write_json(prepared/'preparation.json',{'log_normalizer':0.})
    save_tensor(teacher/'teacher-particles.tensor',pool)
    save_tensor(teacher/'teacher-log-weights.tensor',lw)
    write_json(teacher/'phase.json',{'mala_step_size':.01})
    flow=make_transport(target,4,(11,91))
    trainer=TrainingBlock(flow,target,batch=8,learning_rate=.001,clip=100.,
                          kind='forward',walkers=4,walk_steps=2)
    result=trainer.run(tf.constant([11,4]),tf.constant(3),pool,lw,
                       tf.zeros([4,2],F64),tf.constant(.01,F64))
    state=trainer.checkpoint()
    write_json(parent/'warm-3-checkpoint.json',state)
    write_json(parent/'warm-3-frozen.json',flow.frozen_payload(
        target_signature=target.signature,training_state_hash=state['state_hash']))
    save_tensor(parent/'walkers-3.tensor',result[1])
    write_json(parent/'gradient-calibration.json',{'test_fixture_clip':100.})
    inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in teacher.glob('*.tensor')}
    write_json(parent.parent/'manifest.json',{'input_sha256':inputs})
    cfg={'warm_parent':str(parent),'walkers':4,'walker_steps':2,'batch_size':8,
         'closure':{'fit_rungs':[4]}}
    out=tmp_path/'out';out.mkdir()
    return prepared,teacher,parent,out,cfg


@pytest.mark.parametrize('reset',[False,True])
def test_real_fit_restores_or_resets_adam_and_keeps_failed_probe(saved_fit,reset):
    prepared,teacher,parent,out,cfg=saved_fit
    cfg['closure']['reset_forward_optimizer']=reset
    report=closure.fit('mixture',prepared,teacher,out,cfg,11)
    candidate=out/'w4-lr0.001'
    restoration=json.loads((candidate/'restoration.json').read_text())
    checkpoint=json.loads((candidate/'warm-4-checkpoint.json').read_text())
    assert restoration['optimizer_restored'] is (not reset)
    assert restoration['optimizer_step']==(0 if reset else 3)
    assert restoration['lifetime_forward_updates']==3
    assert checkpoint['step']==(1 if reset else 4)
    assert not report['passed']
    probe=json.loads((candidate/'post-training-1000.json').read_text())
    assert probe['rows']==1000 and probe['complete']
    terminal=json.loads((candidate/'terminal-diagnostic.json').read_text())
    assert terminal['updates']==4 and terminal['role']=='explanatory_only'


@pytest.mark.parametrize('changed',['teacher-particles.tensor','teacher-log-weights.tensor'])
def test_fit_rejects_changed_teacher_before_any_optimizer(saved_fit,monkeypatch,changed):
    prepared,teacher,parent,out,cfg=saved_fit
    values=closure.read_tensor(teacher/changed)
    save_tensor(teacher/changed,values+tf.cast(.01,F64))
    def forbidden(*args,**kwargs):
        raise AssertionError('optimizer constructed before checking changed teacher')
    monkeypatch.setattr(closure,'TrainingBlock',forbidden)
    with pytest.raises(ValueError,match='teacher.*mismatch'):
        closure.fit('mixture',prepared,teacher,out,cfg,11)


def test_numerical_update_failure_is_not_reported_as_plateau(saved_fit,monkeypatch):
    prepared,teacher,parent,out,cfg=saved_fit
    original=closure.TrainingBlock
    def inject_failure(*args,**kwargs):
        trainer=original(*args,**kwargs);run=trainer.run
        def failed(*args):
            result=run(*args)
            return (*result[:-1],tf.constant(False))
        trainer.run=failed
        return trainer
    monkeypatch.setattr(closure,'TrainingBlock',inject_failure)
    report=closure.fit('mixture',prepared,teacher,out,cfg,11)
    assert report['reason']=='numerical_failure'
    assert report['repair']=='numerical_diagnosis'
    assert report['continuation_veto']
    assert not report['continuation_candidates']


@pytest.mark.parametrize('eligible',[False,True])
def test_nonfinite_post_training_probe_vetoes_the_actual_fit(saved_fit,monkeypatch,eligible):
    prepared,teacher,parent,out,cfg=saved_fit
    assessment=closure.flow_assessment
    def screened(*args,**kwargs):
        result=assessment(*args,**kwargs)
        result['passed']=eligible
        return result
    monkeypatch.setattr(closure,'flow_assessment',screened)
    # Inject failure exactly at the probe boundary; actual training and fit
    # assessment still execute. Probe mathematics is checked by other tests.
    monkeypatch.setattr(closure,'PostTrainingProbe',lambda *a,**k:lambda seed:{
        'complete':True,'finite':False,'rows':1000,'valid_rows':999})
    report=closure.fit('mixture',prepared,teacher,out,cfg,11)
    assert report['reason']=='numerical_failure' and not report['passed']
    assert report['continuation_veto'] and not report['continuation_candidates']


def test_refinement_rejects_a_nonfinite_terminal_probe(saved_fit,monkeypatch):
    prepared,teacher,parent,out,cfg=saved_fit
    frozen=json.loads((parent/'warm-3-frozen.json').read_text())
    write_json(parent/'selected-frozen.json',frozen)
    save_tensor(parent/'selected-walkers.tensor',closure.read_tensor(parent/'walkers-3.tensor'))
    write_json(parent/'selected-assessment.json',{'heldout_forward_kl':1.})
    write_json(parent/'phase.json',{'learning_rate':.001,'mala_step_size':.01,'passed':True})
    monkeypatch.setattr(closure,'gradient_calibration',lambda *a,**k:(100.,{'role':'CPU fixture'}))
    monkeypatch.setattr(closure,'PostTrainingProbe',lambda *a,**k:lambda seed:{
        'complete':True,'finite':False,'rows':1000,'valid_rows':999})
    report=closure.refine('mixture',prepared,parent,out,cfg,11)
    assert not report['passed'] and report['reason']=='numerical_failure'
    assert report['continuation_veto']


def test_qualification_does_not_consume_rejected_refinement(tmp_path,monkeypatch):
    from bayesfilter.testing import neutra_warm_start_qualification as qualification
    training=tmp_path/'invalid';training.mkdir()
    write_json(training/'phase.json',{'passed':False,'reason':'numerical_failure'})
    def forbidden(*args,**kwargs):raise AssertionError('invalid map reached HMC')
    monkeypatch.setattr(qualification,'qualify',forbidden)
    report=closure.qualify_selected('mixture',tmp_path/'unused-reference',training,tmp_path/'result',{},11)
    assert not report['passed'] and not report['heldout_consumed']


def test_refinement_preserves_shape_passing_rkl_even_when_forward_kl_is_worse(saved_fit,monkeypatch):
    prepared,teacher,parent,out,cfg=saved_fit
    frozen=json.loads((parent/'warm-3-frozen.json').read_text())
    write_json(parent/'selected-frozen.json',frozen)
    save_tensor(parent/'selected-walkers.tensor',closure.read_tensor(parent/'walkers-3.tensor'))
    write_json(parent/'selected-assessment.json',{'passed':True,'heldout_forward_kl':0.})
    write_json(parent/'phase.json',{'learning_rate':.001,'mala_step_size':.01,'passed':True})
    monkeypatch.setattr(closure,'gradient_calibration',lambda *a,**k:(100.,{'role':'CPU mechanics fixture'}))
    monkeypatch.setattr(closure,'flow_assessment',lambda *a,**k:{'passed':True,'heldout_forward_kl':tf.constant(1.,F64)})
    monkeypatch.setattr(closure,'PostTrainingProbe',lambda *a,**k:lambda seed:{
        'complete':True,'finite':True,'rows':1000,'valid_rows':1000})
    report=closure.refine('mixture',prepared,parent,out,cfg,11)
    assert report['passed'] and report['selected_stage']=='warm'
    listing=json.loads((out/'checkpoint-candidates.json').read_text())
    assert [r['stage'] for r in listing['candidates']]==['rkl-2048','rkl-1024','rkl-256','warm']
    assert all(row['eligible'] for row in listing['candidates'])
    for row in listing['candidates']:
        assert json.loads((out/row['probe_file']).read_text())['transport_hash']==row['transport_hash']
