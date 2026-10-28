"""Coordinator regression tests; mocked evidence has no scientific authority."""
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from scripts import continue_modern_hmc_budget_extension as study


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    study.write(path, value)


def fixture(tmp_path, status='completed'):
    campaign, output = tmp_path/'campaign', tmp_path/'extension'
    output.mkdir()
    source = campaign/'source/BayesFilter'
    source_file = source/'scripts/run_modern_hmc_macrofinance_frozen_diagnostic.py'
    source_file.parent.mkdir(parents=True)
    source_file.write_text('# mock numerical supervisor, never executed\n')
    consumer = campaign/'source/MacroFinance'
    consumer.mkdir()
    put(campaign/'isolated-source.json', dict(source_root=str(source), consumer_root=str(consumer),
        source_sha256={str(source_file): study.digest(source_file)}))
    put(campaign/'frozen-cohort.json', dict(epsilon_by_l=[[3, [.4, .8]]]))
    parent = dict(active_run=False, status=status, elapsed_seconds=80., max_wall_seconds=100.,
        started_utc=(study.utcnow()-timedelta(seconds=81)).isoformat(),
        finished_utc=(study.utcnow()-timedelta(seconds=1)).isoformat(),
        exit_code=0 if status == 'completed' else 1)
    put(campaign/'manifest.json', parent)
    put(campaign/'qualification-01/manifest.json', parent)
    evidence = campaign/'qualification-01/evidence'
    complete = 'complete' if status == 'completed' else 'partial_budget'
    states = {'a': 'verified', 'b': 'promotion_failed'}
    saved = dict(completion_status=complete, candidate_states=states,
        candidates=[dict(candidate_id='a', leapfrog_steps=3, epsilon=.4),
                    dict(candidate_id='b', leapfrog_steps=3, epsilon=.8)],
        verified_candidate_ids=['a'])
    put(evidence/'tuning/tuning_checkpoint.json', dict(result=saved))
    put(evidence/'tuning/execution_spec.json', dict(execution=dict(config=dict(use_xla=True),
        runtime_policy=dict(device_type='GPU', cuda_visible_devices='GPU-mock'))))
    put(evidence/'manifest.json', dict(source_sha256={str(source_file): study.digest(source_file)},
        physical_gpu=dict(uuid='GPU-mock'), memory_policy=dict(all_physical_devices_memory_growth=True)))
    archive = evidence/'reload/member-0/archive.json'
    member = evidence/'members/a.json'
    put(archive, {})
    put(member, {})
    put(evidence/'members.json', {'a': str(member)})
    put(evidence/'native-audit.json', dict(rhat_role='reporting_only', verified_candidate_ids=['a'],
        candidate_states=states, unobserved_attempted_chunk_count=0))
    report = dict(status=status, sources_unchanged=True,
        loaded_local_modules={'scripts.mock':str(source_file)},
        posterior_qualification=False, external_consumer_admitted=False,
        default_promoted=False, no_ranking=True,
        execution=dict(completion_status=complete, candidate_states=states, verified=['a'],
                       all_verified_members_exported=True),
        reload=dict(verified=['a'], all_verified_members_reloaded=True,
                    rows=[dict(candidate_id='a', endpoint_preserved=True,
                               finite_continuation=True, archive=str(archive))]))
    put(evidence/'result.json', report)
    put(output/'prior-settlement.json', dict(remaining_seconds=100.))
    allocation = dict(output=str(output), prior_campaign=str(campaign), plan='mock-plan.md',
        credit_seconds=5000., inherited_unsettled_seconds=100., prior_reserved_seconds=100.,
        separate_sir_seconds=7000., diagnostic_charges=[dict(elapsed_seconds=5.)],
        numerical_attempt_limit=1, prior_ledger_sha256=study.digest(output/'prior-settlement.json'))
    put(output/'allocation.json', allocation)
    return campaign, output, evidence, allocation


def test_settlement_credits_once_and_ignores_inclusive_child_costs(tmp_path):
    campaign, _, _, allocation = fixture(tmp_path)
    parent = study.read(campaign/'manifest.json')
    parent['stages'] = [{'elapsed_seconds':100000}]
    waiting = study.settlement(allocation)
    assert waiting['unreserved_seconds'] == 4995
    assert waiting['prior_reservation_unsettled_seconds'] == 100
    first = study.settlement(allocation, parent, 50.)
    assert first == study.settlement(allocation, parent, 50.)
    assert first['remaining_seconds'] == 4965
    assert first['separate_sir_seconds'] == 7000
    assert study.settlement(allocation, parent, 10000)['shortfall_seconds'] == 4985


@pytest.mark.parametrize('value', [True, -1, float('inf'), float('nan'), '100'])
def test_bad_budget_fails(value):
    with pytest.raises(ValueError):
        study.seconds(value)


def test_live_checkpoint_does_not_trigger_resume():
    assert study.terminal_action({'active_run':True}, {}, None,
        {'result':{'completion_status':'partial_budget'}}) == 'wait'


def test_settled_parent_balance_is_carried_without_new_credit_or_old_charge(tmp_path):
    campaign, _, _, allocation = fixture(tmp_path)
    allocation.update(prior_parent_already_settled=True, credit_seconds=0,
                      inherited_unsettled_seconds=4000)
    parent=study.read(campaign/'manifest.json')
    assert study.settlement(allocation)['unreserved_seconds']==3995
    balance=study.settlement(allocation,parent,10)
    assert balance['remaining_seconds']==3985
    assert balance['prior_outer_charged_seconds']==0


def test_error_before_checkpoint_preserves_primary_worker_cause(tmp_path):
    put(tmp_path/'result.json',dict(status='error',error='explicit live check required'))
    with pytest.raises(RuntimeError,match='explicit live check required'):
        study.child_terminal_action(dict(status='error',active_run=False),tmp_path)


def test_external_recovery_entrypoint_is_hashed_and_does_not_replace_snapshot(tmp_path):
    campaign, output, evidence, allocation=fixture(tmp_path,'resource_stopped')
    host=output/'recovery.py';host.write_text('# host orchestration only')
    allocation.update(recovery_entrypoint=str(host),recovery_entrypoint_sha256=study.digest(host))
    command,env,root,_=study.resume_command(allocation,4000.)
    assert command[1]==str(host) and '--plan' in command
    assert root==campaign/'source/BayesFilter' and env['PYTHONPATH']==str(root)
    host.write_text('changed')
    with pytest.raises(ValueError,match='entrypoint changed'):
        study.resume_command(allocation,4000.)


@pytest.mark.parametrize('status,state,action', [
    ('completed','complete','audit'), ('resource_stopped','partial_budget','resume'),
    ('wall_budget_exhausted','partial_budget','resume'),
    ('wall_budget_exhausted','complete','resume')])
def test_explicit_terminal_decisions(status, state, action):
    terminal = dict(active_run=False, status=status, exit_code=0 if status=='completed' else 1)
    result = None if status=='wall_budget_exhausted' else dict(status=status,
                                                              execution=dict(completion_status=state))
    assert study.terminal_action(terminal, terminal, result,
                                 dict(result=dict(completion_status=state))) == action


@pytest.mark.parametrize('status', ['shared_invalidity','paused_infrastructure','error','unknown'])
def test_invalidity_and_unclassified_failures_are_not_budget_stops(status):
    terminal = dict(active_run=False, status=status)
    with pytest.raises(ValueError, match='diagnosis'):
        study.terminal_action(terminal, terminal, None, {'result':{'completion_status':'partial_budget'}})


def test_resource_wrapper_cannot_hide_error_or_invalid_checkpoint():
    terminal = dict(active_run=False, status='resource_stopped')
    checkpoint = dict(result=dict(completion_status='partial_budget'))
    with pytest.raises(ValueError, match='disagree'):
        study.terminal_action(terminal, terminal, dict(status='error'), checkpoint)
    with pytest.raises(ValueError, match='not resumable'):
        study.terminal_action(terminal, terminal, dict(status='resource_stopped',
            execution=dict(completion_status='shared_invalidity')),
            dict(result=dict(completion_status='shared_invalidity')))


def test_resume_uses_original_snapshot_and_growth_without_rerunning_preflight(tmp_path, monkeypatch):
    campaign, output, evidence, allocation = fixture(tmp_path, 'resource_stopped')
    monkeypatch.setenv('PYTHONPATH', '/wrong/shared/checkout')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    command, env, root, child = study.resume_command(allocation, 4000.)
    assert root == campaign/'source/BayesFilter'
    assert command[1] == str(root/'scripts/run_modern_hmc_macrofinance_frozen_diagnostic.py')
    assert command[command.index('--source')+1] == str(evidence)
    assert child == output/'qualification-resume-01' and not child.exists()
    assert env['PYTHONPATH'] == str(root) and env['TF_FORCE_GPU_ALLOW_GROWTH'] == 'true'
    assert 'CUDA_VISIBLE_DEVICES' not in env
    assert env['BAYESFILTER_OP_LIB_DIR'] == str(root/'bayesfilter/ops')
    child.mkdir()
    with pytest.raises(ValueError, match='already exists'):
        study.resume_command(allocation, 4000.)


def test_mutated_frozen_source_blocks_launch(tmp_path):
    campaign, _, _, allocation = fixture(tmp_path)
    source = study.read(campaign/'isolated-source.json')
    Path(next(iter(source['source_sha256']))).write_text('changed')
    with pytest.raises(ValueError, match='source changed'):
        study.resume_command(allocation, 4000.)


@pytest.mark.parametrize('corruption', ['member','reload','source','rhat','calls','authority','gpu','growth','candidate'])
def test_terminal_audit_rejects_incomplete_or_invalid_evidence(tmp_path, corruption):
    campaign, _, evidence, _ = fixture(tmp_path)
    assert study.terminal_audit(campaign, evidence)['verified_count'] == 1
    if corruption=='member':
        put(evidence/'members.json', {})
    elif corruption in {'reload','authority','source'}:
        report = study.read(evidence/'result.json')
        if corruption=='reload': report['reload']['rows'] = []
        elif corruption=='authority': report['posterior_qualification'] = True
        else: report['sources_unchanged'] = False
        put(evidence/'result.json', report)
    elif corruption in {'rhat','calls'}:
        audit = study.read(evidence/'native-audit.json')
        audit['rhat_role' if corruption=='rhat' else 'unobserved_attempted_chunk_count'] = (
            'qualification' if corruption=='rhat' else 1)
        put(evidence/'native-audit.json', audit)
    elif corruption in {'gpu','growth'}:
        manifest = study.read(evidence/'manifest.json')
        if corruption=='gpu': manifest['physical_gpu']['uuid'] = 'GPU-wrong'
        else: manifest['memory_policy']['all_physical_devices_memory_growth'] = False
        put(evidence/'manifest.json', manifest)
    else:
        checkpoint = study.read(evidence/'tuning/tuning_checkpoint.json')
        checkpoint['result']['candidates'].pop()
        put(evidence/'tuning/tuning_checkpoint.json', checkpoint)
    with pytest.raises(ValueError):
        study.terminal_audit(campaign, evidence)


def no_launch(*args, **kwargs):
    pytest.fail('unexpected numerical launch')


def test_finished_parent_is_audited_without_new_numerical_work(tmp_path, monkeypatch):
    _, output, _, _ = fixture(tmp_path)
    monkeypatch.setattr(study, 'run_child', no_launch)
    result = study.run(output/'allocation.json')
    assert result['status']=='completed' and result['audit']['verified_count']==1
    assert study.read(output/'settlement.json')['prior_outer_charged_seconds']==80
    assert study.read(output/'monitor-manifest.json')['active_run'] is False


def test_wait_then_resume_once_and_settle_without_child_double_charge(tmp_path, monkeypatch):
    campaign, output, evidence, _ = fixture(tmp_path, 'resource_stopped')
    terminal = study.read(campaign/'manifest.json')
    put(campaign/'manifest.json', dict(terminal, active_run=True, status='running'))
    monkeypatch.setattr(study.time, 'sleep', lambda _: put(campaign/'manifest.json', terminal))
    calls = []
    def finish(command, env, root, child, cap, mark):
        calls.append(command)
        shutil.copytree(evidence.parent, child)
        receipt = dict(terminal, status='completed', exit_code=0, elapsed_seconds=100000)
        put(child/'manifest.json', receipt)
        checkpoint = study.read(child/'evidence/tuning/tuning_checkpoint.json')
        checkpoint['result']['completion_status'] = 'complete'
        put(child/'evidence/tuning/tuning_checkpoint.json', checkpoint)
        report = study.read(child/'evidence/result.json')
        report['status']='completed'
        report['execution']['completion_status']='complete'
        put(child/'evidence/result.json', report)
        mark({'pid':123, 'elapsed_seconds':0.01})
        return 0
    monkeypatch.setattr(study, 'run_child', finish)
    result = study.run(output/'allocation.json')
    assert result['status']=='completed' and len(calls)==1
    balance = study.read(output/'settlement.json')
    assert 4900 < balance['remaining_seconds'] < 5015
    assert not balance['child_times_added'] and not balance['overlapping_passive_watch_charged']
    assert study.read(evidence/'tuning/tuning_checkpoint.json')['result']['completion_status']=='partial_budget'


def test_parent_error_is_reported_without_resume(tmp_path, monkeypatch):
    _, output, _, _ = fixture(tmp_path, 'error')
    monkeypatch.setattr(study, 'run_child', no_launch)
    result = study.run(output/'allocation.json')
    assert result['status']=='repair_required'
    assert (output/'repair-alert.json').exists()
    assert study.read(output/'settlement.json')['prior_outer_charged_seconds']==80


def test_real_harmless_subprocess_and_timeout_cleanup(tmp_path):
    marks = []
    command = [sys.executable, '-c', 'print("mechanics only")']
    success = tmp_path/'success'; success.mkdir()
    assert study.run_child(command, os.environ.copy(), tmp_path, success/'run',
                           5, marks.append, poll_seconds=.01)==0
    assert marks and (success/'resume-supervisor.log').read_text().strip()=='mechanics only'
    failure = tmp_path/'timeout'; failure.mkdir()
    command = [sys.executable, '-c', 'import time; time.sleep(10)']
    with pytest.raises(TimeoutError):
        study.run_child(command, os.environ.copy(), tmp_path, failure/'run',
                        .1, marks.append, poll_seconds=.01)


def test_master_update_preserves_other_phases(tmp_path):
    master = tmp_path/'master.md'
    master.write_text('prefix\n'+study.MASTER_BEGIN+'\nold\n'+study.MASTER_END+'\nother phase\n')
    study.update_master(master, dict(status='completed', next_action='C2 review'),
                        dict(remaining_seconds=123.))
    body=master.read_text()
    assert body.startswith('prefix\n') and body.endswith('\nother phase\n')
    assert '**completed**' in body and '123.000' in body and '\nold\n' not in body
