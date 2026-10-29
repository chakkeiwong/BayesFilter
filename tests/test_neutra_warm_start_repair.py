"""CPU reference and orchestration regressions; no training-quality claims."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.testing.neutra_warm_start_policy import MALACalibrationFailure, select_mala_candidate, preserve_deferred
from bayesfilter.testing import neutra_warm_start_closure as closure
from bayesfilter.testing.neutra_warm_start_closure import LaplaceMixtureProposal, cloud_assessment, key, load_flow
from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget, BroadStudentProposal, F64
from bayesfilter.testing.neutra_warm_start_campaign import make_transport

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('repair_master',ROOT/'scripts/run_neutra_warm_start_repair_master.py')
repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)
master=repair.master


@pytest.mark.parametrize('invalid,acceptance',[(1,.9),(0,.1)])
def test_no_kernel_selected_when_every_calibration_candidate_fails(invalid,acceptance):
    with pytest.raises(ValueError,match='calibration failed'):
        select_mala_candidate([dict(invalid_proposals=invalid,acceptance=acceptance,
            squared_displacement=1.,step_size=.001)])


def test_curvature_search_repairs_stiff_gaussian_pilot_without_relaxing_screen(tmp_path):
    """Tiny CPU graph reference; demonstrates a missed scale, not teacher quality."""
    distribution=tfp.distributions.MultivariateNormalDiag(
        tf.zeros([2],F64),tf.fill([2],tf.constant(1e-4,F64)))
    proposal=SimpleNamespace(log_prob=distribution.log_prob,
        sample=lambda count,seed:distribution.sample(count,seed=seed),
        covariance=tf.eye(2,batch_shape=[1],dtype=F64)*tf.constant(1e-8,F64))
    target=SimpleNamespace(parameter_dim=2,log_prob=distribution.log_prob)
    original=closure.MALAProgram
    with patch.object(closure,'MALAProgram',side_effect=lambda *args,**kwargs:original(*args,**kwargs,jit_compile=False)):
        dt=closure.mutation_calibration(target,proposal,tmp_path,11)
    rows=json.loads((tmp_path/'mutation-calibration.json').read_text())
    with pytest.raises(MALACalibrationFailure):
        select_mala_candidate(rows[:4])
    assert 0<dt<1e-4 and len(rows)==7
    selected=select_mala_candidate(rows)
    assert selected['step_size']==dt and selected['acceptance']>=.5
    assert selected['invalid_proposals']==0 and selected['squared_displacement']>0


@pytest.mark.parametrize('phase',['teacher','fit'])
def test_calibration_rejection_returns_failed_phase_without_running_sampler_or_training(tmp_path,phase):
    prepared=tmp_path/'prepared';prepared.mkdir()
    output=tmp_path/'output';output.mkdir()
    for name in ('discovered_modes','training','validation'):
        closure.save_tensor(prepared/(name+'.tensor'),tf.constant([[-5.,0.],[5.,0.]],F64))
    (prepared/'preparation.json').write_text(json.dumps({'log_normalizer':0.}))
    with patch.object(closure,'gpu_preflight'), \
         patch.object(closure,'mutation_calibration',side_effect=MALACalibrationFailure('no eligible kernel')), \
         patch.object(closure,'AnnealedSMC',side_effect=AssertionError('sampler started after rejection')), \
         patch.object(closure,'make_transport',side_effect=AssertionError('training started after rejection')):
        if phase=='teacher':
            result=closure.teacher('mixture',prepared,output,master.configuration(),11)
        else:
            result=closure.fit('mixture',prepared,prepared,output,master.configuration(),11,arm='gabrie')
    assert not result['passed'] and result['failure_class']=='kernel_calibration'
    assert not result.get('continuation_veto',False)
    assert json.loads((output/'phase.json').read_text())['reason']=='mutation_calibration_failed'


def test_completed_job_reuse_requires_same_inputs_and_configuration(tmp_path):
    cfg=master.configuration();state={'attempts':[],'active_job':None}
    class Process:
        pid=999999
        def wait(self,timeout=None):return 0
    with patch.object(master.subprocess,'Popen',return_value=Process()) as launch:
        master.run_one(tmp_path,state,cfg,'job','train',[])
        master.run_one(tmp_path,state,cfg,'job','train',[])
        assert launch.call_count==1
        changed={**cfg,'batch_size':cfg['batch_size']*2}
        master.run_one(tmp_path,state,changed,'job','train',[])
        assert launch.call_count==2
        assert state['attempts'][0]['invocation_identity']!=state['attempts'][1]['invocation_identity']


def test_reference_content_change_invalidates_reuse(tmp_path):
    prepared=tmp_path/'prepared';prepared.mkdir()
    data=prepared/'validation.tensor';data.write_bytes(b'first')
    args=['--prepared',str(prepared)];cfg=master.configuration()
    first=master.invocation_identity(cfg,'train',args)
    data.write_bytes(b'second')
    assert first!=master.invocation_identity(cfg,'train',args)


def test_deferred_work_survives_filtered_refresh_and_resolves_explicitly():
    row={'target':'mixture','arm':'smc','seed':11,'phase':'repair'}
    assert preserve_deferred([row],[])==[row]
    assert preserve_deferred([row],[row])==[row]
    assert preserve_deferred([row],[],['repair-mixture-smc-s11'])==[]


def test_laplace_proposal_density_matches_known_defensive_mixture():
    target=WarmStartTarget('mixture',jit_compile=False)
    proposal=LaplaceMixtureProposal(target,tf.constant([[-5.,0.],[5.,0.]],F64))
    x=tf.constant([[-5.,0.],[0.,0.],[5.,1.],[20.,10.]],F64)
    expected=tf.reduce_logsumexp(tf.stack((tf.math.log(tf.constant(.9,F64))+target.log_prob(x),
        tf.math.log(tf.constant(.1,F64))+BroadStudentProposal(2).log_prob(x))),axis=0)
    tf.debugging.assert_near(proposal.log_prob(x),expected,atol=1e-10,rtol=1e-10)
    assert proposal.sample(8,key(1,2)).shape==(8,2)


def test_physical_shape_screen_rejects_moment_matched_unimodal_fit():
    target=WarmStartTarget('mixture',jit_compile=False)
    reference=target.reference_sample(8192,key(7,8))
    z=tf.random.stateless_normal([8192,2],key(9,10),dtype=F64)
    broad=z*tf.constant([((209/9)**.5),1.],F64)+tf.constant([5/3,0.],F64)
    result=cloud_assessment(target,[broad],[tf.zeros([8192],F64)],reference,iid=True)
    assert result['finite'] and not result['passed']
    assert not result['agreement_passed']


def test_refinement_copy_really_has_trainable_gradients(tmp_path):
    target=WarmStartTarget('gaussian',jit_compile=False)
    original=make_transport(target,8,(11,91))
    path=tmp_path/'map.json';path.write_text(json.dumps(original.frozen_payload(target_signature=target.signature)))
    restored=load_flow(path,target,trainable=True)
    with tf.GradientTape() as tape:
        loss=tf.reduce_sum(restored.forward_and_logdet(tf.ones([4,2],F64))[0])
    gradients=tape.gradient(loss,restored.trainable_variables)
    assert all(g is not None for g in gradients)


@pytest.mark.parametrize('final_failure',[False,True])
def test_controller_repairs_fit_before_creating_fresh_final_reference(tmp_path,final_failure):
    master.write(tmp_path/'config.json',master.configuration())
    master.write(tmp_path/'state.json',{'status':'ready','attempts':[],'active_job':None})
    calls=[];reference_seeds=[]
    def execute(root,state,cfg,job,phase,args):
        calls.append((job,phase))
        out=root/'attempts'/job;out.mkdir(parents=True)
        passed=not (phase=='closure_fit' and job.endswith('-v0'))
        if final_failure and phase=='closure_qualify' and job.endswith('-v0'):passed=False
        if phase=='closure_reference':reference_seeds.append(args[args.index('--seed')+1])
        master.write(out/'phase.json',{'passed':passed,'reason':'fixture_pass' if passed else 'no_useful_warm_fit'})
        if phase=='closure_refine':master.write(out/'selected-frozen.json',{'transport_hash':'fixture-hash'})
        state['attempts'].append(dict(job=job,phase=phase,status='complete',output=str(out),
            cpu_core_seconds=0.,gpu_process_seconds=0.))
        return True
    controller=repair.Controller(tmp_path,execute=execute)
    controller.run(['gaussian'],[11],['smc'])
    phases=[phase for _,phase in calls]
    assert phases==['closure_prepare','closure_teacher','closure_fit','closure_fit',
                    'closure_refine','closure_reference','closure_qualify']+(
                        ['closure_reference','closure_qualify'] if final_failure else [])
    assert len(set(reference_seeds))==len(reference_seeds)
    assert controller.record['outcomes'][0]['status']=='posterior_screen_passed'
    assert (tmp_path/'repair-next-phase.json').exists()


def test_teacher_failure_cannot_launch_training_or_final_tests(tmp_path):
    master.write(tmp_path/'config.json',master.configuration())
    master.write(tmp_path/'state.json',{'status':'ready','attempts':[],'active_job':None})
    calls=[]
    def execute(root,state,cfg,job,phase,args):
        calls.append(phase);out=root/'attempts'/job;out.mkdir(parents=True)
        master.write(out/'phase.json',{'passed':phase=='closure_prepare','reason':'fixture'})
        state['attempts'].append(dict(job=job,phase=phase,status='complete',output=str(out),cpu_core_seconds=0.,gpu_process_seconds=0.))
        return True
    controller=repair.Controller(tmp_path,execute=execute)
    controller.run(['mixture'],[11],['smc'])
    assert calls==['closure_prepare','closure_teacher','closure_teacher','closure_teacher']
    assert controller.record['outcomes'][0]['status']=='teacher_repair_exhausted'
