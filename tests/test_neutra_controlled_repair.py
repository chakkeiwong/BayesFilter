"""Tiny CPU-hidden mechanics/reference checks, never training-quality evidence."""
import json
import math
from pathlib import Path
import pytest
import tensorflow as tf

from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget,F64
from bayesfilter.testing.neutra_warm_start_campaign import make_transport
from bayesfilter.inference.neutra_transport import NeuTraOptimizerConfig
from bayesfilter.inference.neutra_joint_training import JointNeuTraTrainer
from bayesfilter.testing.neutra_controlled_reference import ExactMixtureTransport,oracle_bank
from bayesfilter.testing.neutra_rare_regions_tf import event_features,known_masses


def trainer(f=1.,r=1.):
    target=WarmStartTarget('mixture');flow=make_transport(target,4,(31,2),variance_scale=.02)
    cfg=NeuTraOptimizerConfig(8,'standard',.001,.9,.999,1e-8,None,True)
    return JointNeuTraTrainer(flow,target.value_score,cfg,target_signature=target.signature,
                            teacher_id='test-only',forward_weight=f,reverse_weight=r)


def inputs():
    z=tf.random.stateless_normal([8,2],[21,3],dtype=F64)
    x=WarmStartTarget('mixture').reference_sample(8,tf.constant([32,1]))
    return z,x,tf.zeros([8],F64)


def test_joint_gradient_is_sum_and_has_directional_derivative():
    joint,forward,reverse=trainer(),trainer(1.,0.),trainer(0.,1.)
    args=inputs();j=joint.evaluate_joint(*args);f=forward.evaluate_joint(*args);r=reverse.evaluate_joint(*args)
    assert bool(j['valid'])
    for a,b,c in zip(j['gradients'],f['gradients'],r['gradients']):
        assert float(tf.reduce_max(tf.abs(a-b-c)))<1e-9
    var=joint.variables[-1];initial=var.read_value();direction=tf.ones_like(var)
    h=tf.constant(1e-5,F64)
    var.assign(initial+h*direction);plus=joint.evaluate_joint(*args)['loss']
    var.assign(initial-h*direction);minus=joint.evaluate_joint(*args)['loss'];var.assign(initial)
    fd=(plus-minus)/(2*h);ad=tf.reduce_sum(j['gradients'][-1]*direction)
    assert float(tf.abs(fd-ad)/(1+tf.abs(ad)))<1e-6


def test_restore_produces_same_next_update_and_rejects_changed_teacher():
    a=trainer();args=inputs();assert bool(a.train_joint_step(*args)['valid'])
    checkpoint=json.loads(json.dumps(a.checkpoint()))
    b=trainer();b.restore(checkpoint)
    a.train_joint_step(*args);b.train_joint_step(*args)
    for x,y in zip(a.variables,b.variables):assert float(tf.reduce_max(tf.abs(x-y)))<1e-12
    wrong={**checkpoint,'teacher_id':'changed'}
    with pytest.raises(ValueError,match='teacher mismatch'):b.restore(wrong)


@pytest.mark.parametrize('warped',[False,True])
def test_exact_transport_value_and_score(warped):
    flow=ExactMixtureTransport(warped)
    z=tf.constant([[-4.,1.],[-.430727,0.],[0.,-1.],[4.,2.]],F64)
    x,v,s,c=flow.check(z)
    assert float(tf.reduce_max(tf.abs(v)))<1e-10
    assert float(tf.reduce_max(tf.abs(s)))<1e-7
    assert float(tf.reduce_max(tf.abs(c)))<1e-12


def test_oracle_strata_preserve_rare_probability():
    target=WarmStartTarget('mixture');x,w=oracle_bank(target,16,[12,3])
    masses=tf.reduce_sum(tf.exp(w)[:,None]*event_features(target,x),0)
    for i in (0,1):assert math.isclose(float(masses[i]),known_masses()[i],rel_tol=1e-8,abs_tol=1e-14)
    assert math.isclose(float(tf.reduce_sum(tf.exp(w))),1.,rel_tol=1e-12)


def test_invalid_update_rolls_back_all_state():
    a=trainer();before=a.checkpoint();z,x,w=inputs()
    result=a.train_joint_step(z,tf.fill([8,2],tf.constant(float('nan'),F64)),w)
    assert not bool(result['valid']);assert a.checkpoint()==before


def test_actual_fit_block_is_restorable_and_batch_native():
    from bayesfilter.testing.neutra_controlled_repair import StratifiedJointBlock
    target=WarmStartTarget('mixture');x,w=oracle_bank(target,8,[11,2])
    a=StratifiedJointBlock(make_transport(target,4,(21,3)),target,x,w,'fixture',batch=8,lr=.001,forward=1.,reverse=1.)
    result=a.run(tf.constant([12,1]),tf.constant(2))
    assert int(result[0])==2 and bool(result[-1])
    state=a.trainer.checkpoint()
    telemetry=a.gradient_telemetry(12)
    assert telemetry['independent_batches']==4
    assert len(telemetry['layers'])==len(a.flow.trainable_variables)
    assert a.trainer.checkpoint()==state
    b=StratifiedJointBlock(make_transport(target,4,(21,3)),target,x,w,'fixture',batch=8,lr=.001,forward=1.,reverse=1.)
    b.trainer.restore(state)
    a.run(tf.constant([12,1]),tf.constant(1));b.run(tf.constant([12,1]),tf.constant(1))
    for va,vb in zip(a.flow.trainable_variables,b.flow.trainable_variables):
        assert float(tf.reduce_max(tf.abs(va-vb)))<1e-12


def test_gaussian_control_reaches_public_numerical_tuner(tmp_path,monkeypatch):
    from bayesfilter.testing import neutra_warm_start_qualification as q
    from bayesfilter.testing.neutra_controlled_repair import exact_hmc
    public=q.tune_fixed_transport_hmc_kernel
    calls=[]
    def bounded(**kwargs):
        calls.append(kwargs['config'].target_scope)
        return public(**kwargs,max_work_items=1)
    monkeypatch.setattr(q,'tune_fixed_transport_hmc_kernel',bounded)
    cfg={'reference_count':32,'hmc':{'leapfrogs':[3],'initial_epsilon':.5,
        'max_candidates':3,'work_units':9,'job_wall_seconds':45.,
        'measurement_num_results':64,'verification_num_results':64}}
    result=exact_hmc('mixture',tmp_path,tmp_path,cfg,10011)
    assert calls==['standard_gaussian_reference']
    assert result['reference_control'] and not result['heldout_consumed']
    assert (tmp_path/'control-reference.tensor').exists()
    assert not (tmp_path/'confirmation.tensor').exists()


def test_shortlist_preserves_earlier_checkpoint_and_confirms_only_selected(tmp_path,monkeypatch):
    from bayesfilter.testing import neutra_warm_start_qualification as q
    from bayesfilter.testing.neutra_controlled_repair import qualify_training
    candidates=[]
    for step in (2048,8192):
        name=f'step-{step}-frozen.json'
        (tmp_path/name).write_text(json.dumps({'transport_hash':str(step)}))
        candidates.append({'stage':str(step),'filename':name,'eligible':True,'transport_hash':str(step)})
    (tmp_path/'checkpoint-candidates.json').write_text(json.dumps({'candidates':candidates}))
    calls=[]
    def qualify(*args,**kwargs):
        calls.append(kwargs)
        return {'qualified':kwargs['confirm'],'selection_screen_passed':True,'reason':'fixture'}
    monkeypatch.setattr(q,'qualify',qualify)
    cfg={'hmc':{'job_wall_seconds':120.}}
    result=qualify_training('mixture',tmp_path,tmp_path,tmp_path/'screen',cfg,11)
    assert result['selected_frozen']=='step-2048-frozen.json'
    assert len(calls)==1 and not calls[0]['confirm']
    terminal=qualify_training('mixture',tmp_path,tmp_path,tmp_path/'terminal',cfg,14,
                              frozen_filename='step-8192-frozen.json')
    assert terminal['selected_frozen']=='step-8192-frozen.json'
    assert calls[-1]['frozen_filename']=='step-8192-frozen.json'
    qualify_training('mixture',tmp_path,tmp_path,tmp_path/'confirm',cfg,12,confirm=True,
                     frozen_filename=result['selected_frozen'])
    assert calls[-1]['frozen_filename']=='step-2048-frozen.json'
    assert calls[-1]['confirm'] and calls[-1]['start_scope']=='modes'
    with pytest.raises(ValueError,match='preselected'):
        qualify_training('mixture',tmp_path,tmp_path,tmp_path/'bad',cfg,12,confirm=True)


def test_tiny_parent_writes_full_resume_and_geometry_evidence(tmp_path):
    from bayesfilter.testing.neutra_controlled_repair import save_bank,train
    from bayesfilter.testing.neutra_warm_start_campaign import save_tensor
    target=WarmStartTarget('mixture');x,w=oracle_bank(target,8,[21,32])
    save_bank(tmp_path/'oracle',x,w);save_bank(tmp_path/'directed-reference',x,w)
    save_tensor(tmp_path/'validation.tensor',target.reference_sample(1000,tf.constant([12,11])))
    cfg={'widths':[4],'learning_rates':[.001],'batch_size':8,'pilot_updates':1,
        'train_wall_seconds':180.,'training_rungs':[2,4],'plateau_delta':.001}
    out=tmp_path/'fit';out.mkdir()
    result=train('mixture',tmp_path,out,cfg,11,'oracle')
    assert result['finite'] and len(result['history'])==2
    checkpoint=json.loads((out/result['selected_checkpoint']).read_text())
    assert checkpoint['teacher_id']==result['teacher_id']
    metrics=result['history'][-1]['metrics']
    assert metrics['post_training_1000']['rows']==1000
    assert len(metrics['directed']['conditional_offsets'])==5
    assert (out/'step-4-optimizer-diagnostic.json').exists()


def test_metric_graph_reuse_reads_current_parameters():
    from bayesfilter.testing.neutra_controlled_repair import metric_programs
    target=WarmStartTarget('mixture');flow=make_transport(target,4,(12,13))
    programs=metric_programs(flow,target);x=tf.zeros([8,2],F64)
    initial=programs['logq'](x)
    flow.trainable_variables[-1].assign_add(tf.ones_like(flow.trainable_variables[-1])*.1)
    current=programs['logq'](x)
    assert bool(tf.reduce_any(tf.abs(current-initial)>1e-8))
    assert metric_programs(flow,target) is programs
    assert programs['logq'].experimental_get_tracing_count()==1
    assert float(tf.reduce_max(tf.abs(current-flow.log_prob(x))))<1e-10


def test_finalization_recovery_preserves_failed_reference_and_rejects_mismatch(tmp_path):
    from bayesfilter.testing.neutra_controlled_repair import recover_confirmed_control
    payloads={'manifest.json':{'error_type':'FileExistsError','error':str(tmp_path/'verified-member.json'),
                              'source_manifest':'saved-numerical-source.json'},
        'qualification.json':{'reference_control':True,'qualified':False},
        'member-screen.json':[{'passed':True,'candidate_id':'checked-member','warmup_results':2000,'retained_results':9000}],
        'verified-member.json':{'candidate_id':'checked-member'},
        'posterior.json':{'passed':False,'selection_screen_passed':True,'final_reference_check':{'passed':False}},
        'final-reference-check.json':{'passed':False}}
    for name,value in payloads.items():(tmp_path/name).write_text(json.dumps(value))
    result=recover_confirmed_control(tmp_path)
    assert not result['qualified'] and result['reason']=='final_reference_failed'
    (tmp_path/'verified-member.json').write_text(json.dumps({'candidate_id':'different-member'}))
    with pytest.raises(ValueError,match='inconsistent'):recover_confirmed_control(tmp_path)


def test_objective_progress_does_not_confuse_reverse_improvement_with_forward_loss():
    from bayesfilter.testing.neutra_controlled_repair import paired_objective_progress
    before={'forward':tf.constant([1.,2.,3.],F64),'reverse':tf.constant([4.,5.,6.],F64)}
    after={'forward':before['forward']+1.,'reverse':before['reverse']-2.}
    assert paired_objective_progress(before,after,0.,1.)[:2]==(2.,0.)
    assert paired_objective_progress(before,after,1.,1.)[:2]==(1.,0.)
    assert paired_objective_progress(before,after,1.,0.)[:2]==(-1.,0.)
