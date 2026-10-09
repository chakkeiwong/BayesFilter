"""Recovery and candidate-failure behavior of the local research supervisor."""
import importlib.util
import json
from pathlib import Path
import signal
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('warm_start_master',ROOT/'scripts/run_neutra_warm_start_master.py')
master=importlib.util.module_from_spec(spec);spec.loader.exec_module(master)


def test_recovery_records_finished_orphan_without_relaunch_or_double_charge(tmp_path):
    output=tmp_path/'orphan';output.mkdir()
    master.write(output/'manifest.json',dict(status='complete',wall_seconds=7.,cpu_core_seconds=9.,gpu_process_seconds=7.))
    state={'attempts':[],'active_job':{'job':'train-gaussian-rkl-s11','phase':'train','output':str(output),
        'pid':999999999,'started_unix':0.,'timeout':10.}}
    master.recover_active(tmp_path,state,master.configuration())
    assert state['active_job'] is None and len(state['attempts'])==1
    master.recover_active(tmp_path,state,master.configuration())
    assert len(state['attempts'])==1
    assert master.remaining(state,master.configuration())['gpu_process_seconds']==7193.


@pytest.mark.parametrize('code,status,continue_allowed',[(0,'complete',True),(20,'candidate_failed',True),(1,'failed',False),(21,'budget_limited',True),(124,'budget_limited',True),(-signal.SIGXCPU,'budget_limited',True),(-signal.SIGKILL,'failed',False)])
def test_failure_classification_preserves_other_candidates(tmp_path,code,status,continue_allowed):
    class Process:
        pid=999999999
        def wait(self,timeout=None):return code
    cfg=master.configuration();state={'attempts':[],'active_job':None}
    with patch.object(master.subprocess,'Popen',return_value=Process()):
        actual=master.run_one(tmp_path,state,cfg,'train-fixture','train',[])
    assert actual is continue_allowed
    assert state['attempts'][0]['status']==status and state['active_job'] is None
    if code in (0,20):
        with patch.object(master.subprocess,'Popen',side_effect=AssertionError('duplicate launch')):
            assert master.run_one(tmp_path,state,cfg,'train-fixture','train',[])


def test_budget_exhaustion_does_not_launch_or_overwrite_prior_evidence(tmp_path):
    state={'attempts':[{'gpu_process_seconds':7200.,'cpu_core_seconds':1.}], 'active_job':None}
    state['attempts'][0].update(job='old',status='complete')
    with patch.object(master.subprocess,'Popen',side_effect=AssertionError('launch exceeds allowance')):
        assert not master.run_one(tmp_path,state,master.configuration(),'new','train',[])
    assert state['status']=='budget_exhausted'


def test_uncertain_cpu_charge_cannot_fund_more_work(tmp_path):
    cfg=master.configuration()
    state={'attempts':[dict(job='interrupted',status='interrupted',gpu_process_seconds=240.,
        cpu_core_seconds=6000.,cpu_accounting_estimated=True,
        cpu_core_seconds_hard_upper_bound=7300.)],'active_job':None}
    accounting=master.resource_accounting(state,cfg)
    assert accounting['estimated_remaining']['cpu_core_seconds']==1200.
    assert accounting['remaining_for_launch']['cpu_core_seconds']==-100.
    assert not accounting['cpu_accounting_exact']
    assert not accounting['ceiling_compliance_established']
    with patch.object(master.subprocess,'Popen',side_effect=AssertionError('uncertain budget spent')):
        assert not master.run_one(tmp_path,state,cfg,'new','train',[])


def test_missing_orphan_counter_reserves_launch_limit_not_thread_count(tmp_path):
    cfg={**master.configuration(),'cpu_core_seconds':60.}
    output=tmp_path/'interrupted';output.mkdir()
    state={'attempts':[],'active_job':dict(job='train-fixture',phase='train',output=str(output),
        pid=999999999,started_unix=time.time()-20.,timeout=10.,
        command=['worker','--cpu-limit','60'])}
    master.recover_active(tmp_path,state,cfg)
    row=state['attempts'][0]
    assert row['cpu_accounting_estimated']
    assert row['cpu_core_seconds_hard_upper_bound']==60.
    assert master.remaining(state,cfg)['cpu_core_seconds']==0.
    assert row['gpu_process_seconds']>=20.  # Do not hide orphan timeout overruns.


def test_budget_stop_survives_phase_wrapper_and_summary(tmp_path):
    cfg=master.configuration()
    master.write(tmp_path/'config.json',cfg)
    state={'status':'pricing_complete','active_job':None,'attempts':[dict(
        job='old',phase='train',status='interrupted',output=str(tmp_path/'old'),
        wall_seconds=240.,gpu_process_seconds=240.,cpu_core_seconds=6000.,
        cpu_accounting_estimated=True,cpu_core_seconds_hard_upper_bound=7300.)]}
    master.write(tmp_path/'state.json',state)
    with patch.object(master.subprocess,'Popen',side_effect=AssertionError('unexpected worker')):
        master.run(SimpleNamespace(output=str(tmp_path),targets='mixture',arms='gabrie',seeds='11',through='repair'))
    actual=json.loads((tmp_path/'state.json').read_text())
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert actual['status']==summary['state']=='budget_exhausted'
    assert summary['remaining']['cpu_core_seconds']==-100.
    assert summary['resource_accounting']['estimated_remaining']['cpu_core_seconds']==1200.
    assert 'train-mixture-gabrie-s11' in summary['forecast']['pending_training_job_ids']


def test_failed_checks_exit_nonzero_for_durable_queue(tmp_path):
    master.write(tmp_path/'state.json',{'status':'checks_require_repair'})
    with patch.object(master,'run'),patch.object(master.sys,'argv',[
        'master','run','--output',str(tmp_path),'--through','qualify']):
        with pytest.raises(SystemExit) as error:master.main()
    assert error.value.code==2


def test_filtered_resumes_share_repair_allowance_and_keep_existing_retries():
    state={'attempts':[
        {'phase':'train','job':f'repair-{target}-oracle-s11'}
        for target in ('gaussian','mixture','funnel')]}
    # A second attempt is charged to the same job's retry cap, not a new slot.
    state['attempts'].append({'phase':'train','job':'repair-mixture-oracle-s11'})
    state['attempts'].append({'phase':'train','job':'repair-wiggle-aft-s23'})
    repairs=[('mixture','craft'),('mixture','gabrie_discovered'),
             ('warped_mixture','aft'),('funnel','oracle')]
    selected,deferred=master.select_repairs(state,repairs,11,5)
    assert selected==[repairs[0],repairs[1],repairs[3]]
    assert deferred==[repairs[2]]


def test_qualification_uses_frozen_canonical_codec_and_batch_native_adapter():
    import tensorflow as tf
    from bayesfilter.testing.neutra_warm_start_qualification import BenchmarkAdapter,features
    from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget
    from bayesfilter.testing.neutra_warm_start_campaign import make_transport
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    target=WarmStartTarget('mixture',jit_compile=False);adapter=BenchmarkAdapter(target)
    flow=make_transport(target,4,(11,91))
    payload=flow.frozen_payload(target_signature=target.signature)
    loaded=load_frozen_neutra_artifact(payload,expected_target_signature=adapter.adapter_signature()).transport
    x=tf.constant([[-5.,0.],[5.,0.]],tf.float64)
    tf.debugging.assert_near(loaded.forward_batch(loaded.inverse_theta_to_z_batch(x)),x,atol=1e-12)
    tf.debugging.assert_near(adapter.log_prob_and_grad(x)[0],target.log_prob(x))
    assert features(target,x).shape==(2,5)


def test_mixture_posterior_regions_are_actual_events_not_saturated_responsibility_quantiles():
    import tensorflow as tf
    from bayesfilter.testing.neutra_warm_start_diagnostics import features
    from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget
    target=WarmStartTarget('mixture',jit_compile=False)
    x=tf.constant([[[-5.,.1],[5.,-.1]]],tf.float64)
    values=features(target,x)
    assert values.shape==(1,2,5)
    tf.debugging.assert_equal(values[...,-1],tf.constant([[0.,1.]],tf.float64))


def test_new_qualification_route_has_shared_controller_classification():
    ledger=json.loads((ROOT/'docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/c0/route_ledger.json').read_text())
    records=[r for r in ledger['routes'] if r['path']=='bayesfilter/testing/neutra_warm_start_qualification.py']
    assert len(records)==1
    assert records[0]['core_binding']=='delegated'
    assert records[0]['delegate_path']=='bayesfilter/inference/hmc_candidate_set_retained.py'
